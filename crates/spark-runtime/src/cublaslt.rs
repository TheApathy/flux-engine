// SPDX-License-Identifier: AGPL-3.0-only
//! cuBLASLt FFI and autotuned plan cache for Flux Engine CUDA GEMMs.
//!
//! Covers BF16, F16, F32, FP8 (tensor- and outer-vector-scaled), and CUDA 13.0
//! block-scaled NVFP4 / MXFP8 matrix multiplications, plus dtype-/stride-parameterized
//! pinned GEMMs (`typed`, `typed_algo`), reduction-policy diagnostics (`diagnostic`),
//! persistent on-disk plan caching (`plan_cache`), and opt-in CUDA-event stage timing
//! (`stage_timer`).

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

// DeepSeek-V4.1 needs dtype-parameterised (bf16/fp32) GEMMs on top of this module's bf16-only
// entry points -- see typed.rs's own doc. Ported from dsv41/integration; every symbol it
// references via `use super::*` (the cublasLt type aliases, CUDA_R_* constants) already exists
// below unchanged.
mod plan_cache;
mod typed;
mod typed_algo;
// Opt-in CUDA-event stage attribution for prefill chunks (ATLAS_QWEN4_PREFILL_STAGES=1).
pub mod stage_timer;
pub use typed::{
    GemmDtype, gemm_act_weight_t_typed, gemm_act_weight_t_typed_ex, gemm_act_weight_t_typed_pinned,
    register_stream_workspace,
};
pub use typed_algo::pinned_algo_config;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_16F: i32 = 2;
const CUDA_R_32F: i32 = 0;
/// `library_types.h`: real as an `nv_fp8_e4m3`.
const CUDA_R_8F_E4M3: i32 = 28;
/// `library_types.h`: real as an unsigned `nv_fp8_e4m3` (`CUDA_R_8F_UE4M3 = CUDA_R_8F_E4M3`).
const CUDA_R_8F_UE4M3: i32 = 28;
/// `library_types.h`: real as an exponent-only unsigned `nv_fp8_e8m0` (MXFP8 block scales).
const CUDA_R_8F_UE8M0: i32 = 30;
/// `library_types.h`: real as a `nv_fp4_e2m1` (NVFP4, packed 2 elements per byte).
const CUDA_R_4F_E2M1: i32 = 33;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
/// `cublasLt.h`: `CUBLASLT_MATMUL_DESC_POINTER_MODE` (`int32_t`).
const DESC_POINTER_MODE: u32 = 2;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
/// `cublasLt.h`: `CUBLASLT_MATMUL_DESC_FAST_ACCUM` (`int8_t`, default 0).
const DESC_FAST_ACCUM: u32 = 25;
/// `cublasLt.h`: `CUBLASLT_POINTER_MODE_HOST` (`0`) and `CUBLASLT_POINTER_MODE_DEVICE` (`1`).
const POINTER_MODE_HOST: i32 = 0;
const POINTER_MODE_DEVICE: i32 = 1;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;
/// `cublasLt.h`: `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` (`uint32_t`, default 256).
const PREF_MIN_ALIGNMENT_A_BYTES: u32 = 5;
const PREF_MIN_ALIGNMENT_B_BYTES: u32 = 6;
const PREF_MIN_ALIGNMENT_C_BYTES: u32 = 7;
const PREF_MIN_ALIGNMENT_D_BYTES: u32 = 8;
const LAYOUT_BATCH_COUNT: u32 = 5;
const LAYOUT_STRIDED_BATCH_OFFSET: u32 = 6;

// GLM-5.3 reduction-policy diagnostics (explicit-policy API only).
mod diagnostic;
mod diagnostic_contract;
pub use diagnostic::Bf16GemmReceipt;
use diagnostic_contract::HeuristicResult;
pub use diagnostic_contract::ReductionPolicy;

/// Strided batch description for `bf16_gemm_batched`: element strides.
#[derive(Clone, Copy, Debug)]
pub struct StridedBatch {
    pub count: i32,
    pub stride_act: i64,
    pub stride_weight: i64,
    pub stride_out: i64,
    /// Row stride (elements) of `act` / `out`; 0 = dense (K / N).
    pub ld_act: i64,
    pub ld_out: i64,
}

// SAFETY: Symbols and C ABI signatures match `cublasLt.h` and `cuda.h` in the linked
// `libcublasLt` and `libcuda` libraries.
unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtDestroy(handle: cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatrixLayoutSetAttribute(
        layout: cublasLtMatrixLayout_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoInit(
        handle: cublasLtHandle_t,
        compute_type: i32,
        scale_type: i32,
        a_type: i32,
        b_type: i32,
        c_type: i32,
        d_type: i32,
        algo_id: i32,
        algo: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoCheck(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        result: *mut HeurResult,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size_in_bytes: usize,
        size_written: *mut usize,
    ) -> i32;
    fn cublasLtMatmulAlgoConfigSetAttribute(
        algo: *mut c_void,
        attr: u32,
        buf: *const c_void,
        size_in_bytes: usize,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, byte_count: usize) -> i32;
    fn cuMemsetD8_v2(dptr: u64, uc: u8, n: usize) -> i32;
    fn cuMemsetD8Async(dptr: u64, uc: u8, n: usize, stream: u64) -> i32;
    fn cuStreamCreate(stream: *mut u64, flags: u32) -> i32;
    fn cuStreamDestroy_v2(stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuStreamIsCapturing(stream: u64, capture_status: *mut u32) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 8-byte-aligned storage matching `cublasLtMatmulAlgo_t` (`uint64_t data[8]`).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AlignedAlgo(pub(super) [u8; 64]);

impl AlignedAlgo {
    pub(super) const ZERO: Self = Self([0u8; 64]);

    #[inline]
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    #[inline]
    pub(super) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
}

/// True ABI layout of `cublasLtMatmulHeuristicResult_t` (96 bytes, 8-byte aligned).
/// Used for both single-result and multi-candidate heuristic queries so alignment
/// and array stride match `cublasLt.h` exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct HeurResult {
    pub(super) algo: AlignedAlgo,
    pub(super) workspace_size: usize,
    pub(super) state: i32,
    pub(super) waves_count: f32,
    pub(super) reserved: [i32; 4],
}

impl HeurResult {
    pub(super) const ZERO: Self = Self {
        algo: AlignedAlgo::ZERO,
        workspace_size: 0,
        state: 0,
        waves_count: 0.0,
        reserved: [0; 4],
    };
}

struct StreamWorkspaces {
    /// Pre-allocated primary workspace assigned to the first stream that issues a GEMM,
    /// avoiding `cuMemAlloc_v2` inside CUDA graph capture on the primary stream.
    primary_stream: Option<u64>,
    primary_ws: u64,
    /// Dedicated workspaces for any additional concurrent streams.
    extra: std::collections::HashMap<u64, u64>,
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspaces: std::sync::Mutex<StreamWorkspaces>,
    ws_size: usize,
    /// Process-lifetime 8-byte device buffer storing `[0.0f32, 1.0f32]` so
    /// `CUBLASLT_POINTER_MODE_DEVICE` calls with `beta = 0.0` or `1.0` require
    /// zero per-call allocations or H2D transfers (safe under CUDA graph capture).
    device_scalars_buf: u64,
    autotune: bool,
}

impl Ctx {
    const fn device_zero_f32(&self) -> u64 {
        self.device_scalars_buf
    }

    const fn device_one_f32(&self) -> u64 {
        self.device_scalars_buf + std::mem::size_of::<f32>() as u64
    }

    /// Return a dedicated `self.ws_size`-byte device workspace for `stream`.
    ///
    /// Distinct CUDA streams receive distinct device workspaces so concurrent
    /// launches across streams never race on workspace memory; launches on the
    /// same stream execute in FIFO stream order and safely share that stream's
    /// workspace.
    pub(super) fn workspace_for_stream(&self, stream: u64) -> Result<u64> {
        let mut guard = self
            .workspaces
            .lock()
            .map_err(|_| anyhow::anyhow!("cuBLASLt workspace table poisoned"))?;
        if guard.primary_stream == Some(stream) {
            return Ok(guard.primary_ws);
        }
        if let Some(&ws) = guard.extra.get(&stream) {
            return Ok(ws);
        }
        if guard.primary_stream.is_none() {
            guard.primary_stream = Some(stream);
            return Ok(guard.primary_ws);
        }
        let mut ws: u64 = 0;
        // SAFETY: `&mut ws` is a valid out-pointer for allocating `self.ws_size` bytes on the
        // active CUDA context; holding `self.workspaces` prevents duplicate allocations per stream.
        let st = unsafe { cuMemAlloc_v2(&mut ws, self.ws_size) };
        if st != 0 {
            bail!(
                "cuMemAlloc cuBLASLt workspace for stream {stream:#x} failed: {st} \
                 (pre-register side streams with register_stream_workspace before graph capture)"
            );
        }
        crate::startup_ledger::record(
            "cublaslt-workspace",
            format!("stream {stream:#x} workspace"),
            Some(self.ws_size),
            None,
        );
        guard.extra.insert(stream, ws);
        Ok(ws)
    }

    /// Pre-allocate a dedicated cuBLASLt workspace for `stream` (idempotent).
    pub(super) fn register_stream_workspace(&self, stream: u64) -> Result<()> {
        let mut guard = self
            .workspaces
            .lock()
            .map_err(|_| anyhow::anyhow!("cuBLASLt workspace table poisoned"))?;
        if guard.primary_stream == Some(stream) || guard.extra.contains_key(&stream) {
            return Ok(());
        }
        let mut ws: u64 = 0;
        // SAFETY: `&mut ws` is a valid out-pointer for allocating `self.ws_size` bytes on the
        // active CUDA context; holding `self.workspaces` prevents duplicate allocations per stream.
        let st = unsafe { cuMemAlloc_v2(&mut ws, self.ws_size) };
        if st != 0 {
            bail!("cuMemAlloc side-stream cuBLASLt workspace failed: {st}");
        }
        crate::startup_ledger::record(
            "cublaslt-workspace",
            format!("side stream {stream:#x} workspace"),
            Some(self.ws_size),
            None,
        );
        guard.extra.insert(stream, ws);
        Ok(())
    }
}
// SAFETY: `cublasLtHandle_t` is documented thread-safe for concurrent `cublasLtMatmul` calls;
// per-stream workspaces are synchronized via `self.workspaces` (`Mutex<StreamWorkspaces>`) so
// distinct streams never share workspace memory; `device_scalars_buf` is immutable after init.
unsafe impl Send for Ctx {}
// SAFETY: See `Send` impl above — all mutable state inside `Ctx` is guarded by a `Mutex` and
// `handle` / `device_scalars_buf` / `ws_size` / `autotune` are immutable for process lifetime.
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    // SAFETY: `handle` is a valid out-pointer for `cublasLtCreate`.
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    // SAFETY: `ws` is a valid out-pointer for `cuMemAlloc_v2`.
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        // SAFETY: `handle` was created by `cublasLtCreate` above and is not published.
        unsafe {
            let _ = cublasLtDestroy(handle);
        }
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let scalar_vals: [f32; 2] = [0.0, 1.0];
    let mut device_scalars_buf: u64 = 0;
    // SAFETY: `device_scalars_buf` is a valid out-pointer for allocating 8 bytes of device memory.
    let st = unsafe { cuMemAlloc_v2(&mut device_scalars_buf, std::mem::size_of_val(&scalar_vals)) };
    if st != 0 {
        // SAFETY: `ws` and `handle` were created above and are not published.
        unsafe {
            let _ = cuMemFree_v2(ws);
            let _ = cublasLtDestroy(handle);
        }
        bail!("cuMemAlloc cuBLASLt device scalars failed: {st}");
    }
    // SAFETY: `device_scalars_buf` has 8 bytes allocated and `scalar_vals` is an initialized 8-byte host array.
    let st = unsafe {
        cuMemcpyHtoD_v2(
            device_scalars_buf,
            scalar_vals.as_ptr().cast(),
            std::mem::size_of_val(&scalar_vals),
        )
    };
    if st != 0 {
        // SAFETY: `device_scalars_buf`, `ws`, and `handle` were created above and are not published.
        unsafe {
            let _ = cuMemFree_v2(device_scalars_buf);
            let _ = cuMemFree_v2(ws);
            let _ = cublasLtDestroy(handle);
        }
        bail!("cuMemcpyHtoD cuBLASLt device scalars failed: {st}");
    }
    crate::startup_ledger::record(
        "cublaslt-workspace",
        "primary handle workspace",
        Some(ws_size),
        None,
    );
    let autotune = std::env::var("ATLAS_CUBLASLT_AUTOTUNE").ok().as_deref() == Some("1")
        || std::env::var("ATLAS_CUBLAS_TUNED").ok().as_deref() == Some("1");
    if let Err(unused) = CTX.set(Ctx {
        handle,
        workspaces: std::sync::Mutex::new(StreamWorkspaces {
            primary_stream: None,
            primary_ws: ws,
            extra: std::collections::HashMap::new(),
        }),
        ws_size,
        device_scalars_buf,
        autotune,
    }) {
        // Another thread initialized `CTX` first; release our duplicate allocations and handle.
        let unused_ws = unused.workspaces.into_inner().map_or(ws, |w| w.primary_ws);
        // SAFETY: `unused` was never published to any other thread.
        unsafe {
            let _ = cuMemFree_v2(unused.device_scalars_buf);
            let _ = cuMemFree_v2(unused_ws);
            let _ = cublasLtDestroy(unused.handle);
        }
    }
    Ok(CTX.get().expect("CTX initialized above"))
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// Maximum number of cuBLASLt heuristic candidates requested during live
/// autotuning (16 exposes SM121a high-TFLOPS kernels beyond slots 0..7 on
/// large golden prefill shapes).
const MAX_HEURISTIC_CANDIDATES: usize = 16;
/// `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` attribute ID.
const ALGO_CONFIG_SPLITK_NUM: u32 = 2;
/// `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` attribute ID.
const ALGO_CONFIG_REDUCTION_SCHEME: u32 = 3;
/// `CUBLASLT_REDUCTION_SCHEME_NONE` attribute value.
const REDUCTION_SCHEME_NONE: u32 = 0;

/// Canonical identity for deduplicating cuBLASLt heuristic candidates before
/// live GPU timing. Uses all 9 config attributes (`ALGO_ID` through Blackwell
/// `CLUSTER_SHAPE_ID`) so distinct tiles/stages sharing an `ALGO_ID` are
/// preserved while exact duplicate configurations are timed only once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateIdentity {
    Config(plan_cache::AlgoConfig),
    Raw(AlignedAlgo),
}

fn candidate_identity(algo: &AlignedAlgo) -> CandidateIdentity {
    match plan_cache::extract_algo_config(algo) {
        Some(cfg) => CandidateIdentity::Config(cfg),
        None => CandidateIdentity::Raw(*algo),
    }
}

/// Returns `true` when `algo` uses no split-K (`split_k <= 1`) and no
/// cross-CTA reduction (`CUBLASLT_REDUCTION_SCHEME_NONE`).
fn is_deterministic_algo(algo: &AlignedAlgo) -> bool {
    let mut split_k: i32 = 1;
    let mut written: usize = 0;
    // SAFETY: `algo` is an initialized 8-byte-aligned 64-byte `cublasLtMatmulAlgo_t`;
    // `split_k` is 4-byte `i32`.
    let sk_rc = unsafe {
        cublasLtMatmulAlgoConfigGetAttribute(
            algo.as_ptr().cast(),
            ALGO_CONFIG_SPLITK_NUM,
            (&mut split_k as *mut i32).cast(),
            std::mem::size_of::<i32>(),
            &mut written,
        )
    };
    let mut red_scheme: u32 = REDUCTION_SCHEME_NONE;
    let mut rs_written: usize = 0;
    // SAFETY: `algo` is an initialized 8-byte-aligned 64-byte `cublasLtMatmulAlgo_t`;
    // `red_scheme` is 4-byte `u32`.
    let rs_rc = unsafe {
        cublasLtMatmulAlgoConfigGetAttribute(
            algo.as_ptr().cast(),
            ALGO_CONFIG_REDUCTION_SCHEME,
            (&mut red_scheme as *mut u32).cast(),
            std::mem::size_of::<u32>(),
            &mut rs_written,
        )
    };
    sk_rc == 0 && split_k <= 1 && (rs_rc != 0 || red_scheme == REDUCTION_SCHEME_NONE)
}

/// Minimum fractional speedup required for a non-`heur0` candidate to override
/// `heur0` on standard (non-large-SM121a) FP8 shapes when `!ctx.autotune`.
const DEFAULT_HEUR0_OVERRIDE_MARGIN: f32 = 0.02;
/// Minimum fractional speedup (3%) required for a non-`heur0` candidate to
/// override a healthy `heur0` on large SM121a dense FP8 shapes
/// (`m >= 4096 && n >= 2560 && k >= 2560`).
const SM121A_LARGE_HEUR0_OVERRIDE_MARGIN: f32 = 0.03;

/// Validate that `algo` passes `cublasLtMatmulAlgoCheck` within `ws_size`.
///
/// # Safety
/// `handle`, `desc`, `la`, `lb`, and `ld_` must be valid initialized cuBLASLt handles.
unsafe fn algo_check_ok(
    handle: cublasLtHandle_t,
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld_: cublasLtMatrixLayout_t,
    algo: &AlignedAlgo,
    ws_size: usize,
) -> bool {
    let mut check = HeurResult::ZERO;
    // SAFETY: caller guarantees valid cuBLASLt handles; `algo` is 8-byte aligned
    // `cublasLtMatmulAlgo_t` storage and `check` matches `cublasLtMatmulHeuristicResult_t`.
    let rc = unsafe {
        cublasLtMatmulAlgoCheck(
            handle,
            desc,
            la,
            lb,
            ld_,
            ld_,
            algo.as_ptr().cast(),
            &mut check,
        )
    };
    rc == 0 && check.state == 0 && check.workspace_size <= ws_size
}

/// Reconstruct a `cublasLtMatmulAlgo_t` from a cached 9-attribute [`plan_cache::AlgoConfig`]
/// and validate it against `(desc, la, lb, ld_)` via CPU-only `cublasLtMatmulAlgoCheck`
/// (0 GPU launches, 0 stream syncs).
///
/// # Safety
/// `handle`, `desc`, `la`, `lb`, and `ld_` must be valid initialized cuBLASLt handles.
#[allow(clippy::too_many_arguments)]
unsafe fn reconstruct_and_check_algo(
    handle: cublasLtHandle_t,
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld_: cublasLtMatrixLayout_t,
    a_type: i32,
    d_type: i32,
    config: &plan_cache::AlgoConfig,
    ws_size: usize,
) -> Option<AlignedAlgo> {
    let mut aligned = AlignedAlgo::ZERO;
    let algo_ptr: *mut c_void = aligned.as_mut_ptr().cast();
    // SAFETY: caller guarantees valid `handle`; `aligned` is 8-byte aligned 64-byte storage
    // matching `cublasLtMatmulAlgo_t`.
    let init_rc = unsafe {
        cublasLtMatmulAlgoInit(
            handle,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
            a_type,
            a_type,
            d_type,
            d_type,
            config[0] as i32,
            algo_ptr,
        )
    };
    if init_rc != 0 {
        return None;
    }
    // Attr 0 (`CUBLASLT_ALGO_CONFIG_ID`) is read-only and set by `cublasLtMatmulAlgoInit`.
    // Attrs 1..=6 are 4-byte scalars (`TILE_ID`..=`STAGES_ID`).
    for attr in 1u32..=6u32 {
        let val = config[attr as usize];
        // SAFETY: `aligned` was initialized by `cublasLtMatmulAlgoInit`; `val` matches the
        // 4-byte storage of attributes 1..=6 in `cublasLt.h`.
        let rc = unsafe {
            cublasLtMatmulAlgoConfigSetAttribute(
                algo_ptr,
                attr,
                (&val as *const u32).cast(),
                std::mem::size_of::<u32>(),
            )
        };
        if rc != 0 {
            return None;
        }
    }
    // Attrs 7..=8 (`INNER_SHAPE_ID`, `CLUSTER_SHAPE_ID`) use 2-byte `uint16_t` storage.
    for attr in 7u32..=8u32 {
        let val16 = u16::try_from(config[attr as usize]).ok()?;
        // SAFETY: `aligned` was initialized by `cublasLtMatmulAlgoInit`; `val16` matches the
        // 2-byte `uint16_t` storage of attributes 7..=8 in `cublasLt.h`.
        let rc = unsafe {
            cublasLtMatmulAlgoConfigSetAttribute(
                algo_ptr,
                attr,
                (&val16 as *const u16).cast(),
                std::mem::size_of::<u16>(),
            )
        };
        if rc != 0 {
            return None;
        }
    }
    // SAFETY: caller guarantees valid cuBLASLt handles; `aligned` is initialized and 8-byte aligned.
    if unsafe { algo_check_ok(handle, desc, la, lb, ld_, &aligned, ws_size) } {
        Some(aligned)
    } else {
        None
    }
}

/// Attempt zero-GPU-launch cross-`M` donor reuse for `ckey`: locate the nearest
/// healthy cached donor, obtain a `cublasLtMatmulAlgo_t` matching its 9-attribute
/// `AlgoConfig` (preferring an already-returned heuristic candidate, otherwise
/// reconstructing via `cublasLtMatmulAlgoInit` + `SetAttribute`), and validate it
/// against `(desc, la, lb, ld_)` via CPU-only `cublasLtMatmulAlgoCheck`.
///
/// # Safety
/// `handle`, `desc`, `la`, `lb`, and `ld_` must be valid initialized cuBLASLt handles.
#[allow(clippy::too_many_arguments)]
unsafe fn resolve_cross_m_donor(
    ckey: &plan_cache::PlanCacheKey,
    candidates: &[HeurResult],
    handle: cublasLtHandle_t,
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld_: cublasLtMatrixLayout_t,
    a_type: i32,
    d_type: i32,
    ws_size: usize,
) -> Option<(u32, AlignedAlgo)> {
    let donor = plan_cache::lookup_nearest_m_donor(ckey)?;
    if ckey.deterministic && !plan_cache::is_deterministic_config(&donor.config) {
        return None;
    }
    for (idx, cand) in candidates.iter().enumerate() {
        if cand.state != 0 || cand.workspace_size > ws_size {
            continue;
        }
        if plan_cache::extract_algo_config(&cand.algo).as_ref() != Some(&donor.config) {
            continue;
        }
        if ckey.deterministic && idx > 0 && !is_deterministic_algo(&cand.algo) {
            continue;
        }
        // SAFETY: caller guarantees valid cuBLASLt handles; `cand.algo` is initialized.
        if unsafe { algo_check_ok(handle, desc, la, lb, ld_, &cand.algo, ws_size) } {
            return Some((donor.donor_m, cand.algo));
        }
    }
    // SAFETY: caller guarantees valid cuBLASLt handles.
    let algo = unsafe {
        reconstruct_and_check_algo(
            handle,
            desc,
            la,
            lb,
            ld_,
            a_type,
            d_type,
            &donor.config,
            ws_size,
        )
    }?;
    if ckey.deterministic && !is_deterministic_algo(&algo) {
        return None;
    }
    Some((donor.donor_m, algo))
}

/// Reduce repeated per-iteration GPU timings into a steady-state latency
/// estimate in milliseconds.
///
/// Filters to finite positive samples, sorts ascending, and selects
/// `sorted[len / 2]` (exact median for 3 or 5 trials; conservative slower
/// reading for 2 trials so a single fast cold-L2 iteration cannot fool
/// candidate ranking, while 3+ trials also filter single-iteration DMA stalls).
fn reduce_trial_times_ms(raw_samples: &[f32]) -> Option<f32> {
    let mut valid: Vec<f32> = raw_samples
        .iter()
        .copied()
        .filter(|t| t.is_finite() && *t > 0.0)
        .collect();
    if valid.is_empty() {
        return None;
    }
    valid.sort_by(f32::total_cmp);
    Some(valid[valid.len() / 2])
}

/// Select the winning candidate index from live-timed `fp8_gemm_device_scaled`
/// candidates after all trials complete.
///
/// On large SM121a dense FP8 shapes (`guard_sm121a_large == true`), guards
/// against cold-L2 `sm89_xmma` (`CUBLASLT_MATMUL_STAGES_64x*`) poisoning:
/// - Whenever at least one native Blackwell (`!is_legacy_64x_stages`) candidate
///   achieves a healthy timing, legacy `STAGES_64x*` candidates cannot override it.
/// - Any non-`heur0` candidate must beat a healthy `heur0` by at least 3%
///   (`SM121A_LARGE_HEUR0_OVERRIDE_MARGIN`) before overriding `heur0`.
fn select_fp8_live_winner<H>(
    times: &[f32],
    configs: &[Option<plan_cache::AlgoConfig>],
    ctx_autotune: bool,
    guard_sm121a_large: bool,
    mut is_healthy: H,
) -> Option<usize>
where
    H: FnMut(f32) -> bool,
{
    let is_valid = |t: f32| t.is_finite() && t > 0.0;
    let is_legacy_64x = |idx: usize| {
        configs.get(idx).copied().flatten().is_some_and(|cfg| {
            plan_cache::is_legacy_64x_stages(cfg[plan_cache::ALGO_CONFIG_IDX_STAGES_ID])
        })
    };

    let exclude_legacy_64x = guard_sm121a_large
        && times
            .iter()
            .enumerate()
            .any(|(idx, &t)| is_valid(t) && !is_legacy_64x(idx) && is_healthy(t));

    let t0 = times.first().copied().unwrap_or(f32::NAN);
    let heur0_eligible = is_valid(t0) && !(exclude_legacy_64x && is_legacy_64x(0));
    let heur0_healthy = heur0_eligible && is_healthy(t0);

    let mut best_idx: Option<usize> = None;
    let mut best_ms = f32::INFINITY;
    for (idx, &t) in times.iter().enumerate() {
        if !is_valid(t) {
            continue;
        }
        if exclude_legacy_64x && is_legacy_64x(idx) {
            continue;
        }
        let beats_heur0 = idx == 0
            || !heur0_eligible
            || if guard_sm121a_large {
                !heur0_healthy || t < t0 * (1.0 - SM121A_LARGE_HEUR0_OVERRIDE_MARGIN)
            } else {
                ctx_autotune || t < t0 * (1.0 - DEFAULT_HEUR0_OVERRIDE_MARGIN)
            };
        if beats_heur0 && t < best_ms {
            best_ms = t;
            best_idx = Some(idx);
        }
    }
    best_idx
}

/// Filter heuristic candidates by runtime state, workspace budget, eligibility
/// predicate, and configuration deduplication, preserving cuBLASLt's heuristic
/// priority order (lowest index kept for duplicate configs).
fn select_distinct_candidates<F, P>(
    candidates: &[HeurResult],
    ws_size: usize,
    mut fingerprint: F,
    mut eligible: P,
) -> Vec<usize>
where
    F: FnMut(&AlignedAlgo) -> CandidateIdentity,
    P: FnMut(usize, &HeurResult) -> bool,
{
    let mut selected = Vec::with_capacity(candidates.len());
    let mut seen = Vec::with_capacity(candidates.len());
    for (idx, cand) in candidates.iter().enumerate() {
        if cand.state != 0 || cand.workspace_size > ws_size {
            continue;
        }
        if !eligible(idx, cand) {
            continue;
        }
        let id = fingerprint(&cand.algo);
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        selected.push(idx);
    }
    selected
}

/// Minimum `(m, n)` for default FP8 / block-scaled cuBLASLt heuristic search
/// on non-SM120 GPUs (preserves GB10 SM121a prefill-only default gate).
const DEFAULT_TUNE_MIN_M: u32 = 512;
const DEFAULT_TUNE_MIN_N: u32 = 512;
/// Minimum `(m, n)` for SM120 (`cc_major == 12 && cc_minor == 0`) decode/verify
/// cuBLASLt heuristic search (cached to disk via `plan_cache` outside graph
/// capture so split-K / CTA candidates saturate 110-188 SMs at `m >= 16`).
const SM120_DECODE_TUNE_MIN_M: u32 = 16;
const SM120_DECODE_TUNE_MIN_N: u32 = 256;

#[inline]
const fn is_sm120_cc(device_cc: Option<(i32, i32)>) -> bool {
    matches!(device_cc, Some((12, 0)))
}

/// Whether live cuBLASLt heuristic search may evaluate non-deterministic
/// split-K / cross-CTA reduction candidates for `(m, device_cc)`.
///
/// - Explicit `ATLAS_CUBLAS_TUNED=1` / `ATLAS_CUBLASLT_AUTOTUNE=1`
///   (`ctx_autotune == true`) enables split-K candidates across all shapes.
/// - On SM120 (`cc == (12, 0)`), decode/verify shapes `m in 16..512` also
///   explore split-K CTA candidates when autotune is not disabled (`!= "0"`),
///   while prefill (`m >= 512`) and GB10 SM121a (`cc == (12, 1)`) remain
///   strictly deterministic unless `ctx_autotune` is enabled.
#[inline]
const fn allow_splitk_candidates(
    ctx_autotune: bool,
    auto_tune_enabled: bool,
    m: u32,
    device_cc: Option<(i32, i32)>,
) -> bool {
    ctx_autotune
        || (auto_tune_enabled
            && is_sm120_cc(device_cc)
            && m >= SM120_DECODE_TUNE_MIN_M
            && m < DEFAULT_TUNE_MIN_M)
}

/// Number of heuristic candidates to request for `fp8_gemm_device_scaled`,
/// parameterized by the active device compute capability. Returns `1` during
/// CUDA graph capture or when autotuning is disabled, and
/// [`MAX_HEURISTIC_CANDIDATES`] (`16`) for eligible prefill shapes (and
/// SM120 decode/verify shapes `m >= 16, n >= 256`).
fn fp8_requested_candidates_for_cc(
    ctx_autotune: bool,
    fp8_tensor_tune: bool,
    is_capturing: bool,
    outer: bool,
    m: u32,
    n: u32,
    device_cc: Option<(i32, i32)>,
) -> i32 {
    if is_capturing {
        return 1;
    }
    let (min_m, min_n) = if is_sm120_cc(device_cc) {
        (SM120_DECODE_TUNE_MIN_M, SM120_DECODE_TUNE_MIN_N)
    } else {
        (DEFAULT_TUNE_MIN_M, DEFAULT_TUNE_MIN_N)
    };
    let do_tune = ctx_autotune || (!outer && fp8_tensor_tune && m >= min_m && n >= min_n);
    if do_tune {
        MAX_HEURISTIC_CANDIDATES as i32
    } else {
        1
    }
}

/// Number of heuristic candidates to request for `fp8_gemm_device_scaled`
/// under the baseline (non-SM120) gate.
#[cfg(test)]
fn fp8_requested_candidates(
    ctx_autotune: bool,
    fp8_tensor_tune: bool,
    is_capturing: bool,
    outer: bool,
    m: u32,
    n: u32,
) -> i32 {
    fp8_requested_candidates_for_cc(
        ctx_autotune,
        fp8_tensor_tune,
        is_capturing,
        outer,
        m,
        n,
        None,
    )
}

/// Immortal per-shape matmul plan: desc + layouts + the algo that won the
/// first-use autotune. cublasLt objects are immutable after setup, so sharing
/// one plan across calls (and inside CUDA graph capture) is safe.
struct TunedPlan {
    desc: usize,
    la: usize,
    lb: usize,
    ld: usize,
    algo: AlignedAlgo,
}
// SAFETY: All cuBLASLt descriptor, layout, and `AlignedAlgo` handles inside `TunedPlan` are
// immutable after construction and live for the process lifetime.
unsafe impl Send for TunedPlan {}
// SAFETY: See `Send` impl above — `TunedPlan` performs no interior mutation after publication.
unsafe impl Sync for TunedPlan {}

type PlanShape = (u32, u32, u32, i32, u32);
type PlanCache = std::collections::HashMap<PlanShape, &'static TunedPlan>;

static PLANS: OnceLock<std::sync::Mutex<PlanCache>> = OnceLock::new();

/// Build desc+layouts for `out[M,N]=act[M,K]@W[N,K]ᵀ` (same mapping as
/// `bf16_gemm_act_weight_t`), autotune over the heuristic's top-16 algos on a
/// PRIVATE stream with zeroed dummy operands (runs once per shape; safe while
/// another stream is mid graph-capture), and cache the winner. The naive
/// heuristic[0] pick left the K=8 verify o_proj at 113 GB/s (~24 CTAs on 48
/// SMs, no split-K) — tuning recovers the split-K/tile choice per shape.
fn tuned_plan(m: u32, n: u32, k: u32, ldc: u32) -> Result<&'static TunedPlan> {
    tuned_plan_dt(m, n, k, CUDA_R_16BF, 2, ldc)
}

/// As [`tuned_plan`], but with the A/B operand element type chosen by the
/// caller (`at`, `ab` = its size in bytes). D stays BF16. The plan cache is
/// keyed by operand type AND the output row stride as well as shape, so FP8
/// and BF16 plans — and plans for different `ldc` — never collide.
fn tuned_plan_dt(
    m: u32,
    n: u32,
    k: u32,
    at: i32,
    ab: usize,
    ldc: u32,
) -> Result<&'static TunedPlan> {
    let plans = PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(&p) = plans.lock().unwrap().get(&(m, n, k, at, ldc)) {
        return Ok(p);
    }
    let ctx = ctx()?;
    // SAFETY: All cuBLASLt descriptors, layouts, preferences, temporary device buffers, stream,
    // and events are wrapped in RAII guards so any early `?` error cleans up every resource cleanly.
    // `results` is an array of 8-byte-aligned `HeurResult` matching `cublasLtMatmulHeuristicResult_t`.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, at, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut lb_raw, at, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        let lb = LayoutGuard(lb_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_raw, CUDA_R_16BF, n as u64, m as u64, ldc as i64),
            "LayoutD",
        )?;
        let ld_ = LayoutGuard(ld_raw);
        let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
        let pref = PrefGuard(pref_raw);
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut results = [HeurResult::ZERO; MAX_HEURISTIC_CANDIDATES];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc.0,
                la.0,
                lb.0,
                ld_.0,
                ld_.0,
                pref.0,
                MAX_HEURISTIC_CANDIDATES as i32,
                results.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        drop(pref);
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        let returned_count = (returned as usize).min(results.len());

        const TIE_EPS: f32 = 0.20;
        let disk_key = plan_cache::PlanCacheKey::for_current_device(plan_cache::PlanSpec {
            ws_size: ctx.ws_size,
            kind: plan_cache::PlanKind::TunedActWeightT,
            m,
            n,
            k,
            at,
            ldc,
            outer: false,
            tie_eps: TIE_EPS,
            requested: MAX_HEURISTIC_CANDIDATES as u32,
            deterministic: false,
        });
        let selected_algo: AlignedAlgo = if let Some(ref ckey) = disk_key
            && let Some(hit) = plan_cache::lookup_candidate(
                ckey,
                &results[..returned_count],
                ctx.ws_size,
                |_idx, cand| {
                    algo_check_ok(
                        ctx.handle,
                        desc.0,
                        la.0,
                        lb.0,
                        ld_.0,
                        &cand.algo,
                        ctx.ws_size,
                    )
                },
            ) {
            let source = if hit.from_legacy_key {
                "disk-cache(legacy) hit"
            } else {
                "disk-cache hit"
            };
            tracing::info!(
                "cuBLASLt {source} {m}x{n}x{k} dtype={at} ldc={ldc}: algo[{}] of {returned_count} (prior {:.3}ms, 0ms GPU tune)",
                hit.candidate_idx,
                hit.measured_ms,
            );
            results[hit.candidate_idx].algo
        } else if let Some(ref ckey) = disk_key
            && let Some((donor_m, algo)) = resolve_cross_m_donor(
                ckey,
                &results[..returned_count],
                ctx.handle,
                desc.0,
                la.0,
                lb.0,
                ld_.0,
                at,
                CUDA_R_16BF,
                ctx.ws_size,
            )
        {
            tracing::info!(
                "cuBLASLt {m}x{n}x{k} dtype={at} ldc={ldc}: cross-M donor hit donor_m={donor_m} -> m={m} (0ms GPU tune)",
            );
            algo
        } else {
            // Dummy operands (zeroed) + private stream + private workspace:
            // tune without touching the caller's stream (which may be mid
            // graph-capture) AND without racing on per-stream workspaces while
            // live GEMMs execute on other streams.
            struct TempDevBuf(u64);
            impl Drop for TempDevBuf {
                fn drop(&mut self) {
                    if self.0 != 0 {
                        // SAFETY: buffer was allocated by `cuMemAlloc_v2` and
                        // `TuneStreamGuard` is declared after `TempDevBuf` so
                        // the private tune stream synchronizes before free.
                        unsafe {
                            let _ = cuMemFree_v2(self.0);
                        }
                    }
                }
            }
            struct TuneStreamGuard(u64);
            impl Drop for TuneStreamGuard {
                fn drop(&mut self) {
                    if self.0 != 0 {
                        // SAFETY: private stream created by `cuStreamCreate`; sync drains all
                        // queued work before destroying the stream and freeing `TempDevBuf`s.
                        unsafe {
                            let _ = cuStreamSynchronize(self.0);
                            let _ = cuStreamDestroy_v2(self.0);
                        }
                    }
                }
            }
            struct TuneEventGuard(u64);
            impl Drop for TuneEventGuard {
                fn drop(&mut self) {
                    if self.0 != 0 {
                        // SAFETY: event created by `cuEventCreate` and uniquely owned.
                        unsafe {
                            let _ = cuEventDestroy_v2(self.0);
                        }
                    }
                }
            }
            let (mut dw_raw, mut da_raw, mut dd_raw, mut dws_raw): (u64, u64, u64, u64) =
                (0, 0, 0, 0);
            chk(
                cuMemAlloc_v2(&mut dw_raw, n as usize * k as usize * ab),
                "tuneAllocW",
            )?;
            let dw = TempDevBuf(dw_raw);
            chk(
                cuMemAlloc_v2(&mut da_raw, m as usize * k as usize * ab),
                "tuneAllocA",
            )?;
            let da = TempDevBuf(da_raw);
            chk(
                cuMemAlloc_v2(&mut dd_raw, m as usize * ldc as usize * 2),
                "tuneAllocD",
            )?;
            let dd = TempDevBuf(dd_raw);
            chk(cuMemAlloc_v2(&mut dws_raw, ctx.ws_size), "tuneAllocWS")?;
            let dws = TempDevBuf(dws_raw);
            let mut ts_raw: u64 = 0;
            chk(cuStreamCreate(&mut ts_raw, 1), "tuneStream")?;
            let ts = TuneStreamGuard(ts_raw);
            let _ = cuMemsetD8Async(dw.0, 0, n as usize * k as usize * ab, ts.0);
            let _ = cuMemsetD8Async(da.0, 0, m as usize * k as usize * ab, ts.0);
            let _ = cuMemsetD8Async(dws.0, 0, ctx.ws_size, ts.0);
            let (mut e0_raw, mut e1_raw): (u64, u64) = (0, 0);
            chk(cuEventCreate(&mut e0_raw, 0), "tuneEventCreate0")?;
            let e0 = TuneEventGuard(e0_raw);
            chk(cuEventCreate(&mut e1_raw, 0), "tuneEventCreate1")?;
            let e1 = TuneEventGuard(e1_raw);
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            // Candidate timings, indexed 1:1 with `results`. NaN = candidate was
            // rejected (bad state, duplicate config, runtime refusal, or event
            // failure) and is not eligible. Selection happens AFTER the loop so
            // the winner is a pure function of the ordering, not of measurement
            // sequence.
            let mut times = vec![f32::NAN; returned_count];
            let iters = 10;
            // Bursts are median-reduced: rejects both a single transient slow
            // DMA burst and a single artificially fast cold-L2 first burst.
            const BURSTS: usize = 3;
            let distinct = select_distinct_candidates(
                &results[..returned_count],
                ctx.ws_size,
                candidate_identity,
                |_idx, _cand| true,
            );
            for &i in &distinct {
                let r = &results[i];
                let run = |st: u64| {
                    cublasLtMatmul(
                        ctx.handle,
                        desc.0,
                        &alpha as *const f32 as *const c_void,
                        dw.0 as *const c_void,
                        la.0,
                        da.0 as *const c_void,
                        lb.0,
                        &beta as *const f32 as *const c_void,
                        dd.0 as *const c_void,
                        ld_.0,
                        dd.0 as *mut c_void,
                        ld_.0,
                        r.algo.as_ptr() as *const c_void,
                        dws.0 as *mut c_void,
                        ctx.ws_size,
                        st as *mut c_void,
                    )
                };
                if run(ts.0) != 0 {
                    continue; // algo rejected at runtime — skip
                }
                // Warmup burst, untimed: first-touch of this algo's workspace and
                // any lazy kernel load must not land inside a measured window.
                let mut warmup_ok = true;
                for _ in 0..iters {
                    if run(ts.0) != 0 {
                        warmup_ok = false;
                        break;
                    }
                }
                if !warmup_ok || cuStreamSynchronize(ts.0) != 0 {
                    continue;
                }
                let mut burst_times = [f32::NAN; BURSTS];
                let mut ok = true;
                for slot in &mut burst_times {
                    if cuEventRecord(e0.0, ts.0) != 0 {
                        ok = false;
                        break;
                    }
                    for _ in 0..iters {
                        if run(ts.0) != 0 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok || cuEventRecord(e1.0, ts.0) != 0 || cuEventSynchronize(e1.0) != 0 {
                        ok = false;
                        break;
                    }
                    let mut ms = 0f32;
                    if cuEventElapsedTime(&mut ms, e0.0, e1.0) == 0 {
                        *slot = ms / iters as f32;
                    }
                }
                if ok && let Some(per) = reduce_trial_times_ms(&burst_times) {
                    times[i] = per;
                }
            }
            // Drop events, sync+destroy stream, then free dummy device buffers explicitly.
            drop(e1);
            drop(e0);
            drop(ts);
            drop(dws);
            drop(dd);
            drop(da);
            drop(dw);

            // Deterministic selection: take the fastest time, then keep the
            // LOWEST-INDEXED candidate within TIE_EPS of it so ties resolve to
            // cuBLASLt's own deterministic preference order.
            let t_min = times
                .iter()
                .copied()
                .filter(|t| t.is_finite())
                .fold(f32::INFINITY, f32::min);
            if !t_min.is_finite() {
                bail!("cuBLASLt: every algorithm for {m}x{n}x{k} failed to run");
            }
            let band = t_min * (1.0 + TIE_EPS);
            let best = times
                .iter()
                .position(|t| t.is_finite() && *t <= band)
                .expect("t_min is finite, so at least one candidate is within the band");
            let best_ms = times[best];
            let in_band = times
                .iter()
                .filter(|t| t.is_finite() && **t <= band)
                .count();
            let bytes = n as u64 * k as u64 * ab as u64;
            tracing::info!(
                "cuBLASLt tune {m}x{n}x{k} dtype={at}: algo[{best}] of {returned_count} ({} distinct) @ {best_ms:.3}ms \
                 ({:.0} GB/s weight-read) fastest={t_min:.3}ms tie_band={in_band} \
                 eps={TIE_EPS} times={times:.4?}",
                distinct.len(),
                bytes as f64 / (best_ms as f64 / 1e3) / 1e9,
            );
            if let Some(ref ckey) = disk_key {
                plan_cache::record_winner(ckey, &results[best].algo, best, best_ms);
            }
            results[best].algo
        };
        let mut guard = plans.lock().unwrap();
        if let Some(&existing) = guard.get(&(m, n, k, at, ldc)) {
            // Another thread raced us and inserted first; RAII guards destroy our handles.
            return Ok(existing);
        }
        let plan: &'static TunedPlan = Box::leak(Box::new(TunedPlan {
            desc: desc.into_raw() as usize,
            la: la.into_raw() as usize,
            lb: lb.into_raw() as usize,
            ld: ld_.into_raw() as usize,
            algo: selected_algo,
        }));
        guard.insert((m, n, k, at, ldc), plan);
        Ok(plan)
    }
}

/// Tuned variant of [`bf16_gemm_act_weight_t`]: per-shape cached plan
/// (autotuned on first use), zero per-call descriptor/heuristic overhead.
pub fn bf16_gemm_act_weight_t_tuned(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_act_weight_t_tuned_ex(act, weight, out, m, n, k, n, 1.0, stream)
}

/// Tuned variant with explicit output stride and alpha (plans cached per (m,n,k,ldc)).
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_tuned_ex(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    if ldc < n {
        bail!("cuBLASLt: ldc {ldc} < n {n}");
    }
    let plan = tuned_plan(m, n, k, ldc)?;
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    let beta: f32 = 0.0;
    // SAFETY: `plan` holds immutable process-lifetime cuBLASLt handles and an 8-byte-aligned
    // `AlignedAlgo`; `ws` is a dedicated per-stream workspace of `ctx.ws_size` bytes; caller
    // supplies valid device buffers and stream.
    unsafe {
        chk(
            cublasLtMatmul(
                ctx.handle,
                plan.desc as cublasLtMatmulDesc_t,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr() as *const c_void,
                ws as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            "MatmulTuned",
        )
    }
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16 — the standard
/// projection GEMM (activation × transposed weight). Maps to cuBLASLt's
/// column-major convention as `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_act_weight_t_ldc(act, weight, out, m, n, k, n, 1.0, stream)
}

/// [`bf16_gemm_act_weight_t`] with an explicit output row stride `ldc >= n`
/// (elements), so a projection can land inside a wider interleaved row.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_ldc(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    if ldc < n {
        bail!("cuBLASLt: ldc {ldc} < n {n}");
    }
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    // SAFETY: All cuBLASLt descriptors, layouts, and preferences are wrapped in RAII guards so
    // early error returns never leak handles. `result` is an 8-byte-aligned `HeurResult` matching
    // `cublasLtMatmulHeuristicResult_t`; `ws` is dedicated to `stream`; caller supplies valid
    // device pointers and stream.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut lb_raw, CUDA_R_16BF, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        let lb = LayoutGuard(lb_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_raw, CUDA_R_16BF, n as u64, m as u64, ldc as i64),
            "LayoutD",
        )?;
        let ld_ = LayoutGuard(ld_raw);
        let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
        let pref = PrefGuard(pref_raw);
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = HeurResult::ZERO;
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc.0,
                la.0,
                lb.0,
                ld_.0,
                ld_.0,
                pref.0,
                1,
                (&mut result as *mut HeurResult).cast(),
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = alpha;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc.0,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la.0,
            act as *const c_void,
            lb.0,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_.0,
            out as *mut c_void,
            ld_.0,
            result.algo.as_ptr().cast(),
            ws as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// [`bf16_gemm_act_weight_t`] on `m` rows with the algorithm cuBLASLt's
/// heuristic picks for the same call at `ref_m` rows (queried exactly as that
/// call queries it, then cached). Row blocks of one `ref_m`-row GEMM run with
/// this reproduce the whole GEMM's per-row arithmetic; the Qwen4 PLE prefill
/// pipeline (`ATLAS_QWEN4_PLE_PREFILL_PIPELINE`) projects a chunk per token
/// block this way.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_ref_m(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ref_m: u32,
    stream: u64,
) -> Result<()> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    type AlgoCache = Mutex<Option<HashMap<(u32, u32, u32), HeurResult>>>;
    static ALGOS: AlgoCache = Mutex::new(None);
    if m == 0 || ref_m < m {
        bail!("cuBLASLt ref_m GEMM: rows {m} outside reference rows {ref_m}");
    }
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    // SAFETY: All cuBLASLt descriptors, layouts, and preferences are wrapped in RAII guards so
    // early error returns never leak handles. `result` is an 8-byte-aligned `HeurResult` matching
    // `cublasLtMatmulHeuristicResult_t`; `ws` is dedicated to `stream`; caller supplies valid
    // device pointers and stream.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        let layouts = |rows: u32| -> Result<(LayoutGuard, LayoutGuard)> {
            let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut lb_raw,
                    CUDA_R_16BF,
                    k as u64,
                    rows as u64,
                    k as i64,
                ),
                "LayoutB",
            )?;
            let lb = LayoutGuard(lb_raw);
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut ld_raw,
                    CUDA_R_16BF,
                    n as u64,
                    rows as u64,
                    n as i64,
                ),
                "LayoutD",
            )?;
            let ld_ = LayoutGuard(ld_raw);
            Ok((lb, ld_))
        };
        let key = (ref_m, n, k);
        let cached = ALGOS
            .lock()
            .ok()
            .and_then(|algos| algos.as_ref().and_then(|map| map.get(&key).copied()));
        let result = match cached {
            Some(result) => result,
            None => {
                let (lb, ld_) = layouts(ref_m)?;
                let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
                chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
                let pref = PrefGuard(pref_raw);
                let ws_size = ctx.ws_size;
                chk(
                    cublasLtMatmulPreferenceSetAttribute(
                        pref.0,
                        PREF_MAX_WORKSPACE_BYTES,
                        &ws_size as *const usize as *const c_void,
                        std::mem::size_of::<usize>(),
                    ),
                    "PrefWorkspace",
                )?;
                let mut result = HeurResult::ZERO;
                let mut returned: i32 = 0;
                let status = cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    desc.0,
                    la.0,
                    lb.0,
                    ld_.0,
                    ld_.0,
                    pref.0,
                    1,
                    (&mut result as *mut HeurResult).cast(),
                    &mut returned,
                );
                chk(status, "AlgoGetHeuristic")?;
                if returned < 1 {
                    bail!("cuBLASLt: no algorithm for {ref_m}x{n}x{k}");
                }
                if let Ok(mut algos) = ALGOS.lock() {
                    algos.get_or_insert_with(HashMap::new).insert(key, result);
                }
                result
            }
        };
        let (lb, ld_) = layouts(m)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc.0,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la.0,
            act as *const c_void,
            lb.0,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_.0,
            out as *mut c_void,
            ld_.0,
            result.algo.as_ptr().cast(),
            ws as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ` with **FP8 e4m3 operands and
/// a BF16 result**, using the same per-shape immortal plan cache as
/// [`bf16_gemm_act_weight_t_tuned`], so there is no per-call descriptor or
/// heuristic work. The first call at a given shape autotunes over the
/// heuristic's top-16 algorithms on a private stream; every later call is a
/// single `cublasLtMatmul`.
///
/// The hand-written prefill kernels round both operands to e4m3 before
/// `mma.sync...f32.e4m3.e4m3.f32` (the W4A16 path dequantises NVFP4 and
/// re-rounds via `cvt.rn.satfinite.e4m3x2.f32`), so passing those same e4m3
/// bytes here computes identical products and differs only in accumulation
/// order. Not bit-exact by construction; gated and gate-tested.
///
/// `k` and `n` must be multiples of 16 (cuBLASLt FP8 constraint).
pub fn fp8_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if !k.is_multiple_of(16) || !n.is_multiple_of(16) {
        bail!("cuBLASLt FP8 requires k and n multiples of 16, got k={k} n={n}");
    }
    // FP8 path writes a packed [m, n] output, so the row stride is n.
    let plan = tuned_plan_dt(m, n, k, CUDA_R_8F_E4M3, 1, n)?;
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    // SAFETY: `plan` holds immutable process-lifetime cuBLASLt handles and an 8-byte-aligned
    // `AlignedAlgo`; `ws` is a dedicated per-stream workspace of `ctx.ws_size` bytes; caller
    // supplies valid device buffers and stream.
    unsafe {
        chk(
            cublasLtMatmul(
                ctx.handle,
                plan.desc as cublasLtMatmulDesc_t,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr() as *const c_void,
                ws as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            "MatmulFp8Tuned",
        )
    }
}

/// `CUBLASLT_MATMUL_DESC_{A,B}_SCALE_POINTER` / `_SCALE_MODE` and
/// `cublasLtMatmulMatrixScale_t` values (`cublasLt.h`, CUDA 13.0).
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
/// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`: 16-element `UE4M3` block scaling for NVFP4.
const MATRIX_SCALE_VEC16_UE4M3: i32 = 1;
/// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`: 32-element `UE8M0` block scaling for MXFP8.
const MATRIX_SCALE_VEC32_UE8M0: i32 = 2;
const MATRIX_SCALE_OUTER_VEC_32F: i32 = 3;

/// Cached plan for device-scaled (scalar or outer-vector) FP8 GEMM, keyed by
/// `(m, n, k, ldc, outer)`. The matmul descriptor is guarded by a `Mutex` so
/// per-call scale-pointer updates (`DESC_A_SCALE_POINTER` / `DESC_B_SCALE_POINTER`)
/// and kernel enqueue are serialized across concurrent host threads.
struct Fp8OuterPlan {
    desc: std::sync::Mutex<usize>,
    la: usize,
    lb: usize,
    ld: usize,
    algo: AlignedAlgo,
}
// SAFETY: cuBLASLt layout handles and `AlignedAlgo` are immutable after creation; per-call
// mutations to `desc` attributes are serialized by `self.desc` (`Mutex<usize>`).
unsafe impl Send for Fp8OuterPlan {}
// SAFETY: See `Send` impl above — `self.desc` prevents concurrent host mutation of the C++
// descriptor across threads.
unsafe impl Sync for Fp8OuterPlan {}

type Fp8OuterCache = std::collections::HashMap<(u32, u32, u32, u32, bool), &'static Fp8OuterPlan>;
static FP8_OUTER_PLANS: OnceLock<std::sync::Mutex<Fp8OuterCache>> = OnceLock::new();

/// Row-major `out[M,N] = alpha * (sa[m] * sw[n]) * (act[M,K] @ weight[N,K]^T)` with
/// e4m3 operands, an f32 per-token activation scale vector `sa` (M) and an f32
/// per-output-channel weight scale vector `sw` (N), BF16 result with row stride
/// `ldc` (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F). `k` and `n` must be
/// multiples of 16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_outer_scaled(
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    fp8_gemm_device_scaled(
        act,
        act_scale,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        ldc,
        alpha,
        true,
        stream,
    )
}

/// Device-scaled FP8 `e4m3` GEMM producing BF16 output with row stride `ldc`.
///
/// - When `outer == false`: each scale pointer addresses a single device `f32`
///   (`CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F`, per-tensor scaling).
/// - When `outer == true`: `act_scale` is a length-`m` `f32` vector and
///   `weight_scale` is a length-`n` `f32` vector
///   (`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_device_scaled(
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    outer: bool,
    stream: u64,
) -> Result<()> {
    let mode_label = if outer {
        "outer-vector-scaled"
    } else {
        "scalar-scaled"
    };
    if !k.is_multiple_of(16) || !n.is_multiple_of(16) || ldc < n {
        bail!("cuBLASLt FP8 {mode_label}: bad shape m={m} n={n} k={k} ldc={ldc}");
    }
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    let plans =
        FP8_OUTER_PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (m, n, k, ldc, outer);
    let cached = plans.lock().unwrap().get(&key).copied();
    let plan: &'static Fp8OuterPlan = match cached {
        Some(p) => p,
        // SAFETY: All cuBLASLt descriptors, layouts, and preferences are wrapped in RAII guards
        // so early error returns destroy every handle cleanly. `results` is an array of
        // 8-byte-aligned `HeurResult` matching `cublasLtMatmulHeuristicResult_t`; `ws` is
        // dedicated to `stream`.
        None => unsafe {
            let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
            chk(
                cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                "DescCreate",
            )?;
            let desc = DescGuard(desc_raw);
            let ta = CUBLAS_OP_T;
            let tb = CUBLAS_OP_N;
            chk(
                cublasLtMatmulDescSetAttribute(
                    desc.0,
                    DESC_TRANSA,
                    &ta as *const i32 as *const c_void,
                    4,
                ),
                "TRANSA",
            )?;
            chk(
                cublasLtMatmulDescSetAttribute(
                    desc.0,
                    DESC_TRANSB,
                    &tb as *const i32 as *const c_void,
                    4,
                ),
                "TRANSB",
            )?;
            if outer {
                let mode = MATRIX_SCALE_OUTER_VEC_32F;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_A_SCALE_MODE,
                        &mode as *const i32 as *const c_void,
                        4,
                    ),
                    "AScaleMode",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_B_SCALE_MODE,
                        &mode as *const i32 as *const c_void,
                        4,
                    ),
                    "BScaleMode",
                )?;
            }
            // Heuristic needs the pointers present (any valid device address).
            chk(
                cublasLtMatmulDescSetAttribute(
                    desc.0,
                    DESC_A_SCALE_POINTER,
                    &weight_scale as *const u64 as *const c_void,
                    8,
                ),
                "AScalePtr",
            )?;
            chk(
                cublasLtMatmulDescSetAttribute(
                    desc.0,
                    DESC_B_SCALE_POINTER,
                    &act_scale as *const u64 as *const c_void,
                    8,
                ),
                "BScalePtr",
            )?;
            let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut la_raw,
                    CUDA_R_8F_E4M3,
                    k as u64,
                    n as u64,
                    k as i64,
                ),
                "LayoutA",
            )?;
            let la = LayoutGuard(la_raw);
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut lb_raw,
                    CUDA_R_8F_E4M3,
                    k as u64,
                    m as u64,
                    k as i64,
                ),
                "LayoutB",
            )?;
            let lb = LayoutGuard(lb_raw);
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut ld_raw,
                    CUDA_R_16BF,
                    n as u64,
                    m as u64,
                    ldc as i64,
                ),
                "LayoutD",
            )?;
            let ld_ = LayoutGuard(ld_raw);
            let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
            chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
            let pref = PrefGuard(pref_raw);
            let ws_size = ctx.ws_size;
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    pref.0,
                    PREF_MAX_WORKSPACE_BYTES,
                    &ws_size as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                ),
                "PrefWorkspace",
            )?;
            let mut results = [HeurResult::ZERO; MAX_HEURISTIC_CANDIDATES];
            static FP8_TENSOR_TUNE: OnceLock<bool> = OnceLock::new();
            let fp8_tensor_tune = *FP8_TENSOR_TUNE.get_or_init(|| {
                std::env::var("ATLAS_CUBLASLT_AUTOTUNE").ok().as_deref() != Some("0")
            });
            let mut capture_status: u32 = 0;
            let is_capturing =
                cuStreamIsCapturing(stream, &mut capture_status) == 0 && capture_status != 0;
            let device_cc = plan_cache::current_device_cc();
            let requested = fp8_requested_candidates_for_cc(
                ctx.autotune,
                fp8_tensor_tune,
                is_capturing,
                outer,
                m,
                n,
                device_cc,
            );
            let allow_splitk = allow_splitk_candidates(ctx.autotune, fp8_tensor_tune, m, device_cc);
            let do_tune = requested > 1;
            let mut returned: i32 = 0;
            chk(
                cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    desc.0,
                    la.0,
                    lb.0,
                    ld_.0,
                    ld_.0,
                    pref.0,
                    requested,
                    results.as_mut_ptr() as *mut c_void,
                    &mut returned,
                ),
                "AlgoGetHeuristic(fp8 scaled)",
            )?;
            drop(pref);
            if returned < 1 {
                bail!("cuBLASLt: no FP8 {mode_label} algorithm for {m}x{n}x{k}");
            }
            let returned_count = (returned as usize).min(results.len());
            let mut selected_algo = results[0].algo;
            if do_tune && returned_count > 1 {
                let disk_key = plan_cache::PlanCacheKey::for_current_device(plan_cache::PlanSpec {
                    ws_size: ctx.ws_size,
                    kind: plan_cache::PlanKind::Fp8DeviceScaled,
                    m,
                    n,
                    k,
                    at: CUDA_R_8F_E4M3,
                    ldc,
                    outer,
                    tie_eps: 0.0,
                    requested: requested as u32,
                    deterministic: !allow_splitk,
                });
                if let Some(ref ckey) = disk_key
                    && let Some(hit) = plan_cache::lookup_candidate(
                        ckey,
                        &results[..returned_count],
                        ctx.ws_size,
                        |idx, cand| {
                            if !allow_splitk && idx > 0 && !is_deterministic_algo(&cand.algo) {
                                return false;
                            }
                            algo_check_ok(
                                ctx.handle,
                                desc.0,
                                la.0,
                                lb.0,
                                ld_.0,
                                &cand.algo,
                                ctx.ws_size,
                            )
                        },
                    )
                {
                    let best = hit.candidate_idx;
                    selected_algo = results[best].algo;
                    let source = if hit.from_legacy_key {
                        "disk-cache(legacy) hit"
                    } else {
                        "disk-cache hit"
                    };
                    tracing::info!(
                        "cuBLASLt fp8_device_scaled {source} {m}x{n}x{k} ldc={ldc} outer={outer}: algo[{best}] of {returned_count} (prior {:.3}ms, 0ms GPU tune)",
                        hit.measured_ms,
                    );
                } else if let Some(ref ckey) = disk_key
                    && let Some((donor_m, algo)) = resolve_cross_m_donor(
                        ckey,
                        &results[..returned_count],
                        ctx.handle,
                        desc.0,
                        la.0,
                        lb.0,
                        ld_.0,
                        CUDA_R_8F_E4M3,
                        CUDA_R_16BF,
                        ctx.ws_size,
                    )
                {
                    selected_algo = algo;
                    tracing::info!(
                        "cuBLASLt fp8_device_scaled {m}x{n}x{k} ldc={ldc} outer={outer}: cross-M donor hit donor_m={donor_m} -> m={m} (0ms GPU tune)",
                    );
                } else {
                    // When `!allow_splitk`, restrict non-heur0 candidates strictly to
                    // deterministic `split_k <= 1` with `CUBLASLT_REDUCTION_SCHEME_NONE`,
                    // and deduplicate identical 9-attribute configs before live GPU timing.
                    let candidates = select_distinct_candidates(
                        &results[..returned_count],
                        ctx.ws_size,
                        candidate_identity,
                        |idx, cand| allow_splitk || idx == 0 || is_deterministic_algo(&cand.algo),
                    );
                    if candidates.len() <= 1 {
                        let only = candidates.first().copied().unwrap_or(0);
                        selected_algo = results[only].algo;
                    } else {
                        const DEFAULT_TIMED_REPS: usize = 3;
                        const AUTOTUNE_TIMED_REPS: usize = 5;
                        const MAX_TUNE_EVENTS: usize = AUTOTUNE_TIMED_REPS + 1;
                        let timed_reps = if ctx.autotune {
                            AUTOTUNE_TIMED_REPS
                        } else {
                            DEFAULT_TIMED_REPS
                        };
                        let mut events = [0u64; MAX_TUNE_EVENTS];
                        let mut events_ok = true;
                        for ev in events.iter_mut().take(timed_reps + 1) {
                            if cuEventCreate(ev, 0) != 0 {
                                events_ok = false;
                                break;
                            }
                        }
                        if events_ok {
                            let tune_alpha: f32 = alpha;
                            let tune_beta: f32 = 0.0;
                            let run_algo = |algo: &AlignedAlgo| {
                                cublasLtMatmul(
                                    ctx.handle,
                                    desc.0,
                                    &tune_alpha as *const f32 as *const c_void,
                                    weight as *const c_void,
                                    la.0,
                                    act as *const c_void,
                                    lb.0,
                                    &tune_beta as *const f32 as *const c_void,
                                    out as *const c_void,
                                    ld_.0,
                                    out as *mut c_void,
                                    ld_.0,
                                    algo.as_ptr() as *const c_void,
                                    ws as *mut c_void,
                                    ctx.ws_size,
                                    stream as *mut c_void,
                                )
                            };
                            // Extra untimed launch before timing `results[0]` so GPU idle->boost
                            // clock ramp does not penalize cuBLASLt's primary heuristic (`heur0`).
                            if let Some(&first_idx) = candidates.first() {
                                let _ = run_algo(&results[first_idx].algo);
                            }
                            let mut times = vec![f32::NAN; returned_count];
                            let mut configs: Vec<Option<plan_cache::AlgoConfig>> =
                                vec![None; returned_count];
                            for &i in &candidates {
                                let r = &results[i];
                                configs[i] = plan_cache::extract_algo_config(&r.algo);
                                // 1 untimed warmup launch per distinct candidate.
                                if run_algo(&r.algo) != 0 {
                                    continue;
                                }
                                let mut ok = cuEventRecord(events[0], stream) == 0;
                                if ok {
                                    for rep in 0..timed_reps {
                                        if run_algo(&r.algo) != 0
                                            || cuEventRecord(events[rep + 1], stream) != 0
                                        {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                                if !ok || cuEventSynchronize(events[timed_reps]) != 0 {
                                    continue;
                                }
                                // Median-reduce individual back-to-back trials so neither a
                                // single cold-L2 fast outlier (`sm89_xmma`) nor a single
                                // transient LPDDR5X / NVMe DMA stall poisons candidate ranking.
                                let mut samples = [f32::NAN; AUTOTUNE_TIMED_REPS];
                                for rep in 0..timed_reps {
                                    let mut ms = 0.0f32;
                                    if cuEventElapsedTime(&mut ms, events[rep], events[rep + 1])
                                        == 0
                                    {
                                        samples[rep] = ms;
                                    }
                                }
                                if let Some(per) = reduce_trial_times_ms(&samples[..timed_reps]) {
                                    times[i] = per;
                                }
                            }
                            let guard_sm121a_large = disk_key
                                .as_ref()
                                .is_some_and(plan_cache::is_sm121a_large_fp8_key);
                            let best_opt = select_fp8_live_winner(
                                &times,
                                &configs,
                                ctx.autotune,
                                guard_sm121a_large,
                                |ms| {
                                    disk_key.as_ref().is_none_or(|ckey| {
                                        plan_cache::is_canonical_healthy(ckey, ms)
                                    })
                                },
                            );
                            let (best, best_ms) = match best_opt {
                                Some(idx) => (idx, times[idx]),
                                None => (0, f32::INFINITY),
                            };
                            selected_algo = results[best].algo;
                            tracing::info!(
                                "cuBLASLt fp8_device_scaled tune {m}x{n}x{k} ldc={ldc} outer={outer}: algo[{best}] of {returned_count} ({} distinct) @ {best_ms:.3}ms (heur0={:.3}ms) times={times:.3?}",
                                candidates.len(),
                                times[0],
                            );
                            if best_ms.is_finite()
                                && let Some(ref ckey) = disk_key
                            {
                                plan_cache::record_winner(ckey, &selected_algo, best, best_ms);
                            }
                        }
                        for &ev in events.iter().take(timed_reps + 1) {
                            if ev != 0 {
                                cuEventDestroy_v2(ev);
                            }
                        }
                    }
                }
            }
            let mut guard = plans.lock().unwrap();
            if let Some(&existing) = guard.get(&key) {
                // Another thread raced us and inserted first; RAII guards destroy our handles.
                existing
            } else {
                let p: &'static Fp8OuterPlan = Box::leak(Box::new(Fp8OuterPlan {
                    desc: std::sync::Mutex::new(desc.into_raw() as usize),
                    la: la.into_raw() as usize,
                    lb: lb.into_raw() as usize,
                    ld: ld_.into_raw() as usize,
                    algo: selected_algo,
                }));
                guard.insert(key, p);
                p
            }
        },
    };
    let alpha: f32 = alpha;
    let beta: f32 = 0.0;
    let desc_guard = plan
        .desc
        .lock()
        .expect("fp8 scaled descriptor lock poisoned");
    let desc = *desc_guard as cublasLtMatmulDesc_t;
    // SAFETY: `desc` is exclusively locked for this call across both scale-pointer updates and
    // `cublasLtMatmul`; `plan.algo` is an 8-byte-aligned `AlignedAlgo`; `ws` is dedicated to
    // `stream`; caller supplies valid device pointers and stream.
    unsafe {
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_A_SCALE_POINTER,
                &weight_scale as *const u64 as *const c_void,
                8,
            ),
            "AScalePtr",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_B_SCALE_POINTER,
                &act_scale as *const u64 as *const c_void,
                8,
            ),
            "BScalePtr",
        )?;
        chk(
            cublasLtMatmul(
                ctx.handle,
                desc,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr() as *const c_void,
                ws as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            "MatmulFp8Outer",
        )
    }
}

// ── CUDA 13.0 block-scaled NVFP4 & MXFP8 GEMMs ───────────────────────────────

/// Number of `E2M1` elements sharing one `UE4M3` block scale along `K` (`VEC16_UE4M3`).
pub const NVFP4_BLOCK_SIZE: u32 = 16;
/// Number of `E4M3` elements sharing one `UE8M0` block scale along `K` (`VEC32_UE8M0`).
pub const MXFP8_BLOCK_SIZE: u32 = 32;
/// Minimum device pointer alignment (in bytes) required by cuBLASLt block-scaled GEMMs.
const BLOCK_SCALED_PTR_ALIGN_BYTES: u64 = 16;
/// Minimum alignment (in bytes) for a device FP32 `alpha` / `beta` scalar pointer.
const DEVICE_SCALAR_PTR_ALIGN_BYTES: u64 = 4;
/// Required `K` granularity for both NVFP4 (`32 * 4b = 16B`, 2 `VEC16` blocks) and
/// MXFP8 (`32 * 8b = 32B`, 1 `VEC32` block).
const BLOCK_SCALED_K_MULTIPLE: u32 = 32;
/// Required `N` / `ldc` granularity for BF16 output rows (`8 * 2B = 16B` alignment).
const BLOCK_SCALED_BF16_ROW_MULTIPLE: u32 = 8;
/// Height of one cuBLASLt `128x4` 1D block-scale macro-tile along the outer dimension.
const SCALE_TILE_OUTER: u32 = 128;
/// Width of one cuBLASLt `128x4` 1D block-scale macro-tile along the K-block dimension.
const SCALE_TILE_K_BLOCKS: u32 = 4;
/// Byte size of one cuBLASLt `128x4` block-scale macro-tile (`128 * 4`).
const SCALE_TILE_BYTES: usize = (SCALE_TILE_OUTER as usize) * (SCALE_TILE_K_BLOCKS as usize);

/// Compute the byte size required for a cuBLASLt `128x4`-tiled 1D block-scale tensor.
///
/// For a row-major `[outer, k]` matrix with `block_size` elements per scale factor
/// ([`NVFP4_BLOCK_SIZE`] = 16 or [`MXFP8_BLOCK_SIZE`] = 32), cuBLASLt expects scales
/// packed into `128x4` macro-tiles padded up to `ceil(outer / 128) * 128` rows and
/// `ceil((k / block_size) / 4) * 4` scale columns.
///
/// Returns `0` when `block_size == 0`, `outer == 0`, or `k == 0`.
#[must_use]
pub const fn block_scale_padded_bytes(outer: u32, k: u32, block_size: u32) -> usize {
    if block_size == 0 || outer == 0 || k == 0 {
        return 0;
    }
    let padded_outer =
        (outer as usize).div_ceil(SCALE_TILE_OUTER as usize) * (SCALE_TILE_OUTER as usize);
    let k_blocks = (k as usize) / (block_size as usize);
    let padded_k_blocks =
        k_blocks.div_ceil(SCALE_TILE_K_BLOCKS as usize) * (SCALE_TILE_K_BLOCKS as usize);
    padded_outer * padded_k_blocks
}

/// Map logical `(outer_idx, k_block_idx)` to its byte offset inside cuBLASLt's
/// `128x4` swizzled block-scale layout (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`
/// and `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`).
///
/// `k_blocks` is the unpadded scale column count (`k / 16` for NVFP4 or `k / 32`
/// for MXFP8). Matches the `32x4` sub-tile interleaving inside each `128x4` tile:
/// `tile * 512 + (r % 32) * 16 + (r / 32) * 4 + (k_block_idx % 4)`.
#[must_use]
pub const fn block_scale_swizzle_offset(outer_idx: u32, k_block_idx: u32, k_blocks: u32) -> usize {
    let k_tiles = (k_blocks as usize).div_ceil(SCALE_TILE_K_BLOCKS as usize);
    let tile_row = (outer_idx as usize) / (SCALE_TILE_OUTER as usize);
    let tile_k = (k_block_idx as usize) / (SCALE_TILE_K_BLOCKS as usize);
    let r = (outer_idx as usize) % (SCALE_TILE_OUTER as usize);
    let g = (k_block_idx as usize) % (SCALE_TILE_K_BLOCKS as usize);
    let tile_idx = tile_row * k_tiles + tile_k;
    tile_idx * SCALE_TILE_BYTES + (r % 32) * 16 + (r / 32) * 4 + g
}

/// Source of the FP32 `beta` scalar when `alpha` is provided as a device pointer
/// (`CUBLASLT_POINTER_MODE_DEVICE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceBeta {
    /// `beta = 0.0` backed by an internal static device-resident `0.0f32` constant.
    Zero,
    /// `beta = 1.0` backed by an internal static device-resident `1.0f32` constant.
    One,
    /// Caller-owned device pointer to a single 4-byte-aligned FP32 `beta` value.
    DevicePtr(u64),
}

/// FP32 `alpha` and `beta` scalar source for block-scaled cuBLASLt GEMMs
/// (`out = alpha * (act @ weight^T) + beta * out`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockScaleScalars {
    /// Host-resident FP32 `alpha` and `beta` (`CUBLASLT_POINTER_MODE_HOST`).
    Host { alpha: f32, beta: f32 },
    /// Device pointer to a single 4-byte-aligned FP32 `alpha` (for NVFP4, typically
    /// the global tensor scale product `s_a_global * s_b_global`) and a device-resident
    /// `beta` source (`CUBLASLT_POINTER_MODE_DEVICE`). Safe inside CUDA graph capture.
    Device { alpha_ptr: u64, beta: DeviceBeta },
}

impl Default for BlockScaleScalars {
    fn default() -> Self {
        Self::Host {
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

impl BlockScaleScalars {
    /// Host-resident `alpha` and `beta` scalars.
    #[must_use]
    pub const fn host(alpha: f32, beta: f32) -> Self {
        Self::Host { alpha, beta }
    }

    /// Device-resident FP32 `alpha` pointer with `beta = 0.0`.
    #[must_use]
    pub const fn device_alpha(alpha_ptr: u64) -> Self {
        Self::Device {
            alpha_ptr,
            beta: DeviceBeta::Zero,
        }
    }

    /// Device-resident FP32 `alpha` pointer with `beta = 1.0` (in-place residual accumulate).
    #[must_use]
    pub const fn device_alpha_accumulate(alpha_ptr: u64) -> Self {
        Self::Device {
            alpha_ptr,
            beta: DeviceBeta::One,
        }
    }

    /// Device-resident FP32 `alpha` and `beta` pointers.
    #[must_use]
    pub const fn device(alpha_ptr: u64, beta_ptr: u64) -> Self {
        Self::Device {
            alpha_ptr,
            beta: DeviceBeta::DevicePtr(beta_ptr),
        }
    }

    const fn is_device(self) -> bool {
        matches!(self, Self::Device { .. })
    }

    const fn beta_is_known_zero(self) -> bool {
        match self {
            Self::Host { beta, .. } => beta == 0.0,
            Self::Device {
                beta: DeviceBeta::Zero,
                ..
            } => true,
            Self::Device { .. } => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BlockScaledFormat {
    /// NVFP4 (`CUDA_R_4F_E2M1`, packed 2 elements/byte) + `VEC16_UE4M3` scales.
    Nvfp4,
    /// MXFP8 (`CUDA_R_8F_E4M3`, 1 element/byte) + `VEC32_UE8M0` scales.
    Mxfp8,
}

impl BlockScaledFormat {
    const fn label(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4_block_scaled",
            Self::Mxfp8 => "mxfp8_block_scaled",
        }
    }

    const fn cuda_data_type(self) -> i32 {
        match self {
            Self::Nvfp4 => CUDA_R_4F_E2M1,
            Self::Mxfp8 => CUDA_R_8F_E4M3,
        }
    }

    const fn cuda_scale_data_type(self) -> i32 {
        match self {
            Self::Nvfp4 => CUDA_R_8F_UE4M3,
            Self::Mxfp8 => CUDA_R_8F_UE8M0,
        }
    }

    const fn cuda_scale_mode(self) -> i32 {
        match self {
            Self::Nvfp4 => MATRIX_SCALE_VEC16_UE4M3,
            Self::Mxfp8 => MATRIX_SCALE_VEC32_UE8M0,
        }
    }

    /// Disjoint `plan_cache::PlanSpec::at` tag so NVFP4 (`33`) and MXFP8 (`30`)
    /// never collide with scalar/outer FP8 (`28`) or BF16 (`14`) disk-cache keys.
    const fn plan_cache_at_tag(self) -> i32 {
        match self {
            Self::Nvfp4 => CUDA_R_4F_E2M1,
            Self::Mxfp8 => CUDA_R_8F_UE8M0,
        }
    }
}

pub(super) struct DescGuard(pub(super) cublasLtMatmulDesc_t);
impl DescGuard {
    fn into_raw(mut self) -> cublasLtMatmulDesc_t {
        std::mem::replace(&mut self.0, std::ptr::null_mut())
    }
}
impl Drop for DescGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` was created by `cublasLtMatmulDescCreate` and is uniquely owned.
            unsafe {
                let _ = cublasLtMatmulDescDestroy(self.0);
            }
        }
    }
}

pub(super) struct LayoutGuard(pub(super) cublasLtMatrixLayout_t);
impl LayoutGuard {
    fn into_raw(mut self) -> cublasLtMatrixLayout_t {
        std::mem::replace(&mut self.0, std::ptr::null_mut())
    }
}
impl Drop for LayoutGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` was created by `cublasLtMatrixLayoutCreate` and is uniquely owned.
            unsafe {
                let _ = cublasLtMatrixLayoutDestroy(self.0);
            }
        }
    }
}

pub(super) struct PrefGuard(pub(super) cublasLtMatmulPreference_t);
impl Drop for PrefGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` was created by `cublasLtMatmulPreferenceCreate` and is uniquely owned.
            unsafe {
                let _ = cublasLtMatmulPreferenceDestroy(self.0);
            }
        }
    }
}

struct DeviceBufGuard {
    ptr: u64,
    sync_stream: Option<u64>,
}
impl Drop for DeviceBufGuard {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: `self.ptr` was allocated by `cuMemAlloc_v2` and is uniquely owned;
            // synchronizing `sync_stream` ensures no in-flight tuning kernel still accesses it.
            unsafe {
                if let Some(st) = self.sync_stream {
                    let _ = cuStreamSynchronize(st);
                }
                let _ = cuMemFree_v2(self.ptr);
            }
        }
    }
}

/// Cached plan for a CUDA 13.0 block-scaled (`NVFP4` or `MXFP8`) cuBLASLt GEMM.
///
/// The matmul descriptor is guarded by a mutex so per-call scale-pointer updates
/// (`DESC_A_SCALE_POINTER` / `DESC_B_SCALE_POINTER`) and kernel enqueue remain
/// free of CPU-side data races even if invoked from concurrent host threads.
struct BlockScaledPlan {
    desc: std::sync::Mutex<usize>,
    la: usize,
    lb: usize,
    ld: usize,
    algo: AlignedAlgo,
}
// SAFETY: cuBLASLt layout descriptors and algo structs are immutable after creation;
// mutable descriptor pointer attributes are serialized by `self.desc`.
unsafe impl Send for BlockScaledPlan {}
unsafe impl Sync for BlockScaledPlan {}

type BlockScaledPlanKey = (BlockScaledFormat, u32, u32, u32, u32, bool);
type BlockScaledPlanCache = std::collections::HashMap<BlockScaledPlanKey, &'static BlockScaledPlan>;
static BLOCK_SCALED_PLANS: OnceLock<std::sync::Mutex<BlockScaledPlanCache>> = OnceLock::new();

#[allow(clippy::too_many_arguments)]
fn validate_block_scaled_args(
    format: BlockScaledFormat,
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    scalars: BlockScaleScalars,
) -> Result<()> {
    let label = format.label();
    if m == 0 || n == 0 || k == 0 {
        bail!("cuBLASLt {label}: zero dimension not allowed (m={m}, n={n}, k={k})");
    }
    if !k.is_multiple_of(BLOCK_SCALED_K_MULTIPLE) {
        bail!(
            "cuBLASLt {label}: k={k} must be a multiple of {BLOCK_SCALED_K_MULTIPLE} (m={m}, n={n})"
        );
    }
    if ldc < n {
        bail!("cuBLASLt {label}: ldc ({ldc}) < n ({n})");
    }
    if !n.is_multiple_of(BLOCK_SCALED_BF16_ROW_MULTIPLE)
        || !ldc.is_multiple_of(BLOCK_SCALED_BF16_ROW_MULTIPLE)
    {
        bail!(
            "cuBLASLt {label}: n={n} and ldc={ldc} must be multiples of {BLOCK_SCALED_BF16_ROW_MULTIPLE} for 16-byte BF16 alignment"
        );
    }
    for (name, ptr) in [
        ("act", act),
        ("act_scale", act_scale),
        ("weight", weight),
        ("weight_scale", weight_scale),
        ("out", out),
    ] {
        if ptr == 0 {
            bail!("cuBLASLt {label}: null device pointer for `{name}`");
        }
        if !ptr.is_multiple_of(BLOCK_SCALED_PTR_ALIGN_BYTES) {
            bail!(
                "cuBLASLt {label}: `{name}` pointer {ptr:#x} must be {BLOCK_SCALED_PTR_ALIGN_BYTES}-byte aligned"
            );
        }
    }
    match scalars {
        BlockScaleScalars::Host { alpha, beta } => {
            if !alpha.is_finite() || !beta.is_finite() {
                bail!("cuBLASLt {label}: non-finite host scalar (alpha={alpha}, beta={beta})");
            }
        }
        BlockScaleScalars::Device { alpha_ptr, beta } => {
            if alpha_ptr == 0 || !alpha_ptr.is_multiple_of(DEVICE_SCALAR_PTR_ALIGN_BYTES) {
                bail!(
                    "cuBLASLt {label}: device `alpha_ptr` {alpha_ptr:#x} must be non-null and {DEVICE_SCALAR_PTR_ALIGN_BYTES}-byte aligned"
                );
            }
            if let DeviceBeta::DevicePtr(beta_ptr) = beta
                && (beta_ptr == 0 || !beta_ptr.is_multiple_of(DEVICE_SCALAR_PTR_ALIGN_BYTES))
            {
                bail!(
                    "cuBLASLt {label}: device `beta_ptr` {beta_ptr:#x} must be non-null and {DEVICE_SCALAR_PTR_ALIGN_BYTES}-byte aligned"
                );
            }
        }
    }
    Ok(())
}

/// Number of heuristic candidates to request for block-scaled GEMMs,
/// parameterized by the active device compute capability. Returns `1` during
/// CUDA graph capture or when autotuning is disabled, and
/// [`MAX_HEURISTIC_CANDIDATES`] (`16`) for wide GEMMs (`m >= 512 && n >= 512`),
/// SM120 decode/verify shapes (`m >= 16 && n >= 256`), or when `ctx.autotune`
/// is enabled.
fn block_scaled_requested_candidates_for_cc(
    ctx_autotune: bool,
    auto_tune_enabled: bool,
    is_capturing: bool,
    m: u32,
    n: u32,
    device_cc: Option<(i32, i32)>,
) -> i32 {
    if is_capturing {
        return 1;
    }
    let (min_m, min_n) = if is_sm120_cc(device_cc) {
        (SM120_DECODE_TUNE_MIN_M, SM120_DECODE_TUNE_MIN_N)
    } else {
        (DEFAULT_TUNE_MIN_M, DEFAULT_TUNE_MIN_N)
    };
    let do_tune = ctx_autotune || (auto_tune_enabled && m >= min_m && n >= min_n);
    if do_tune {
        MAX_HEURISTIC_CANDIDATES as i32
    } else {
        1
    }
}

/// Number of heuristic candidates to request for block-scaled GEMMs under the
/// baseline (non-SM120) gate.
#[cfg(test)]
fn block_scaled_requested_candidates(
    ctx_autotune: bool,
    auto_tune_enabled: bool,
    is_capturing: bool,
    m: u32,
    n: u32,
) -> i32 {
    block_scaled_requested_candidates_for_cc(
        ctx_autotune,
        auto_tune_enabled,
        is_capturing,
        m,
        n,
        None,
    )
}

/// Shared execution core for CUDA 13.0 block-scaled (`NVFP4` and `MXFP8`) GEMMs:
/// row-major `out[m, n] (BF16) = alpha * (act[m, k] * act_scale) @ (weight[n, k] * weight_scale)^T + beta * out[m, n]`.
///
/// Maps to cuBLASLt's column-major TN convention:
/// - `A_lt = weight` (`[K, N]` col-major, `ld = K`, `CUBLAS_OP_T`, scale = `weight_scale`),
/// - `B_lt = act` (`[K, M]` col-major, `ld = K`, `CUBLAS_OP_N`, scale = `act_scale`),
/// - `C = D = out` (`[N, M]` col-major BF16, `ld = ldc`).
#[allow(clippy::too_many_arguments)]
fn block_scaled_gemm_impl(
    format: BlockScaledFormat,
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    scalars: BlockScaleScalars,
    stream: u64,
) -> Result<()> {
    validate_block_scaled_args(
        format,
        act,
        act_scale,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        ldc,
        scalars,
    )?;
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    let device_scalars = scalars.is_device();
    let label = format.label();
    let plans =
        BLOCK_SCALED_PLANS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key: BlockScaledPlanKey = (format, m, n, k, ldc, device_scalars);
    let cached = plans
        .lock()
        .expect("block-scaled plan cache lock poisoned")
        .get(&key)
        .copied();
    let plan: &'static BlockScaledPlan = match cached {
        Some(p) => p,
        None => {
            // SAFETY: all cuBLASLt descriptors are initialized before use and wrapped in RAII guards
            // so early error returns destroy every handle cleanly.
            unsafe {
                let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
                chk(
                    cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                    "DescCreate(block_scaled)",
                )?;
                let desc = DescGuard(desc_raw);
                let ta = CUBLAS_OP_T;
                let tb = CUBLAS_OP_N;
                let ptr_mode = if device_scalars {
                    POINTER_MODE_DEVICE
                } else {
                    POINTER_MODE_HOST
                };
                // Block-scaled Blackwell tensor cores accumulate natively in FP32.
                // Verified on CUDA 13.0 (SM121): NVFP4 (`VEC16_UE4M3`) rejects
                // `fast_accum = 1` with `CUBLAS_STATUS_INVALID_VALUE` (7), while
                // MXFP8 (`VEC32_UE8M0`) returns identical heuristics for 0 and 1.
                let fast_accum: i8 = 0;
                let scale_mode = format.cuda_scale_mode();
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_TRANSA,
                        (&ta as *const i32).cast(),
                        std::mem::size_of::<i32>(),
                    ),
                    "TRANSA(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_TRANSB,
                        (&tb as *const i32).cast(),
                        std::mem::size_of::<i32>(),
                    ),
                    "TRANSB(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_POINTER_MODE,
                        (&ptr_mode as *const i32).cast(),
                        std::mem::size_of::<i32>(),
                    ),
                    "PointerMode(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_FAST_ACCUM,
                        (&fast_accum as *const i8).cast(),
                        std::mem::size_of::<i8>(),
                    ),
                    "FastAccum(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_A_SCALE_MODE,
                        (&scale_mode as *const i32).cast(),
                        std::mem::size_of::<i32>(),
                    ),
                    "AScaleMode(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_B_SCALE_MODE,
                        (&scale_mode as *const i32).cast(),
                        std::mem::size_of::<i32>(),
                    ),
                    "BScaleMode(block_scaled)",
                )?;
                // Heuristic inspects scale pointers for presence/alignment without dereferencing.
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_A_SCALE_POINTER,
                        (&weight_scale as *const u64).cast(),
                        std::mem::size_of::<u64>(),
                    ),
                    "AScalePtr(block_scaled)",
                )?;
                chk(
                    cublasLtMatmulDescSetAttribute(
                        desc.0,
                        DESC_B_SCALE_POINTER,
                        (&act_scale as *const u64).cast(),
                        std::mem::size_of::<u64>(),
                    ),
                    "BScalePtr(block_scaled)",
                )?;

                let a_type = format.cuda_data_type();
                let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
                let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
                // For `CUDA_R_4F_E2M1`, rows/cols/ld are specified in logical elements
                // (2 elements packed per byte along the contiguous K dimension).
                chk(
                    cublasLtMatrixLayoutCreate(
                        &mut la_raw,
                        a_type,
                        u64::from(k),
                        u64::from(n),
                        i64::from(k),
                    ),
                    "LayoutA(block_scaled)",
                )?;
                let la = LayoutGuard(la_raw);
                chk(
                    cublasLtMatrixLayoutCreate(
                        &mut lb_raw,
                        a_type,
                        u64::from(k),
                        u64::from(m),
                        i64::from(k),
                    ),
                    "LayoutB(block_scaled)",
                )?;
                let lb = LayoutGuard(lb_raw);
                chk(
                    cublasLtMatrixLayoutCreate(
                        &mut ld_raw,
                        CUDA_R_16BF,
                        u64::from(n),
                        u64::from(m),
                        i64::from(ldc),
                    ),
                    "LayoutD(block_scaled)",
                )?;
                let ld_ = LayoutGuard(ld_raw);

                let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
                chk(
                    cublasLtMatmulPreferenceCreate(&mut pref_raw),
                    "PrefCreate(block_scaled)",
                )?;
                let pref = PrefGuard(pref_raw);
                let ws_size = ctx.ws_size;
                chk(
                    cublasLtMatmulPreferenceSetAttribute(
                        pref.0,
                        PREF_MAX_WORKSPACE_BYTES,
                        (&ws_size as *const usize).cast(),
                        std::mem::size_of::<usize>(),
                    ),
                    "PrefWorkspace(block_scaled)",
                )?;
                let min_align = BLOCK_SCALED_PTR_ALIGN_BYTES as u32;
                for (attr, attr_name) in [
                    (PREF_MIN_ALIGNMENT_A_BYTES, "PrefMinAlignA"),
                    (PREF_MIN_ALIGNMENT_B_BYTES, "PrefMinAlignB"),
                    (PREF_MIN_ALIGNMENT_C_BYTES, "PrefMinAlignC"),
                    (PREF_MIN_ALIGNMENT_D_BYTES, "PrefMinAlignD"),
                ] {
                    chk(
                        cublasLtMatmulPreferenceSetAttribute(
                            pref.0,
                            attr,
                            (&min_align as *const u32).cast(),
                            std::mem::size_of::<u32>(),
                        ),
                        attr_name,
                    )?;
                }

                static BLOCK_SCALED_TUNE: OnceLock<bool> = OnceLock::new();
                let tune_enabled = *BLOCK_SCALED_TUNE.get_or_init(|| {
                    std::env::var("ATLAS_CUBLASLT_AUTOTUNE").ok().as_deref() != Some("0")
                });
                let mut capture_status: u32 = 0;
                let is_capturing =
                    cuStreamIsCapturing(stream, &mut capture_status) == 0 && capture_status != 0;
                let device_cc = plan_cache::current_device_cc();
                let requested = block_scaled_requested_candidates_for_cc(
                    ctx.autotune,
                    tune_enabled,
                    is_capturing,
                    m,
                    n,
                    device_cc,
                );
                let allow_splitk =
                    allow_splitk_candidates(ctx.autotune, tune_enabled, m, device_cc);
                let mut results = [HeurResult::ZERO; MAX_HEURISTIC_CANDIDATES];
                let mut returned: i32 = 0;
                chk(
                    cublasLtMatmulAlgoGetHeuristic(
                        ctx.handle,
                        desc.0,
                        la.0,
                        lb.0,
                        ld_.0,
                        ld_.0,
                        pref.0,
                        requested,
                        results.as_mut_ptr().cast(),
                        &mut returned,
                    ),
                    "AlgoGetHeuristic(block_scaled)",
                )?;
                if returned < 1 {
                    bail!(
                        "cuBLASLt: no {label} algorithm (data_type={a_type}, scale_type={}, scale_mode={scale_mode}) for {m}x{n}x{k} ldc={ldc}",
                        format.cuda_scale_data_type(),
                    );
                }
                let returned_count = (returned as usize).min(results.len());
                let mut selected_algo = results[0].algo;
                let do_tune = requested > 1;
                if do_tune && returned_count > 1 {
                    let disk_key =
                        plan_cache::PlanCacheKey::for_current_device(plan_cache::PlanSpec {
                            ws_size: ctx.ws_size,
                            kind: plan_cache::PlanKind::Fp8DeviceScaled,
                            m,
                            n,
                            k,
                            at: format.plan_cache_at_tag(),
                            ldc,
                            outer: device_scalars,
                            tie_eps: 0.0,
                            requested: requested as u32,
                            deterministic: !allow_splitk,
                        });
                    if let Some(ref ckey) = disk_key
                        && let Some(hit) = plan_cache::lookup_candidate(
                            ckey,
                            &results[..returned_count],
                            ctx.ws_size,
                            |idx, cand| {
                                if !allow_splitk && idx > 0 && !is_deterministic_algo(&cand.algo) {
                                    return false;
                                }
                                algo_check_ok(
                                    ctx.handle,
                                    desc.0,
                                    la.0,
                                    lb.0,
                                    ld_.0,
                                    &cand.algo,
                                    ctx.ws_size,
                                )
                            },
                        )
                    {
                        selected_algo = results[hit.candidate_idx].algo;
                        tracing::info!(
                            "cuBLASLt {label} disk-cache hit {m}x{n}x{k} ldc={ldc} dev_scalars={device_scalars}: algo[{}] of {returned_count} (prior {:.3}ms, 0ms GPU tune)",
                            hit.candidate_idx,
                            hit.measured_ms,
                        );
                    } else if let Some(ref ckey) = disk_key
                        && let Some((donor_m, algo)) = resolve_cross_m_donor(
                            ckey,
                            &results[..returned_count],
                            ctx.handle,
                            desc.0,
                            la.0,
                            lb.0,
                            ld_.0,
                            a_type,
                            CUDA_R_16BF,
                            ctx.ws_size,
                        )
                    {
                        selected_algo = algo;
                        tracing::info!(
                            "cuBLASLt {label} {m}x{n}x{k} ldc={ldc} dev_scalars={device_scalars}: cross-M donor hit donor_m={donor_m} -> m={m} (0ms GPU tune)",
                        );
                    } else {
                        let candidates = select_distinct_candidates(
                            &results[..returned_count],
                            ctx.ws_size,
                            candidate_identity,
                            |idx, cand| {
                                allow_splitk || idx == 0 || is_deterministic_algo(&cand.algo)
                            },
                        );
                        if candidates.len() <= 1 {
                            let only = candidates.first().copied().unwrap_or(0);
                            selected_algo = results[only].algo;
                        } else {
                            // Never clobber caller's pre-existing `out` when `beta` may be non-zero:
                            // allocate a temporary zeroed BF16 output buffer for live timing.
                            let scratch_out_guard = if scalars.beta_is_known_zero() {
                                DeviceBufGuard {
                                    ptr: 0,
                                    sync_stream: None,
                                }
                            } else {
                                let out_bytes = (m as usize) * (ldc as usize) * 2;
                                let mut tmp_out: u64 = 0;
                                chk(
                                    cuMemAlloc_v2(&mut tmp_out, out_bytes),
                                    "tuneAllocOut(block_scaled)",
                                )?;
                                let guard = DeviceBufGuard {
                                    ptr: tmp_out,
                                    sync_stream: Some(stream),
                                };
                                let _ = cuMemsetD8_v2(guard.ptr, 0, out_bytes);
                                guard
                            };
                            let tune_out = if scratch_out_guard.ptr != 0 {
                                scratch_out_guard.ptr
                            } else {
                                out
                            };
                            const DEFAULT_TIMED_REPS: usize = 3;
                            const AUTOTUNE_TIMED_REPS: usize = 5;
                            const MAX_TUNE_EVENTS: usize = AUTOTUNE_TIMED_REPS + 1;
                            let timed_reps = if ctx.autotune {
                                AUTOTUNE_TIMED_REPS
                            } else {
                                DEFAULT_TIMED_REPS
                            };
                            let mut events = [0u64; MAX_TUNE_EVENTS];
                            let mut events_ok = true;
                            for ev in events.iter_mut().take(timed_reps + 1) {
                                if cuEventCreate(ev, 0) != 0 {
                                    events_ok = false;
                                    break;
                                }
                            }
                            if events_ok {
                                let host_alpha_one: f32 = 1.0;
                                let host_beta_zero: f32 = 0.0;
                                let (tune_alpha_ptr, tune_beta_ptr): (
                                    *const c_void,
                                    *const c_void,
                                ) = match scalars {
                                    BlockScaleScalars::Host { .. } => (
                                        (&host_alpha_one as *const f32).cast(),
                                        (&host_beta_zero as *const f32).cast(),
                                    ),
                                    BlockScaleScalars::Device { alpha_ptr, .. } => (
                                        alpha_ptr as *const c_void,
                                        ctx.device_zero_f32() as *const c_void,
                                    ),
                                };
                                let run_algo = |algo: &AlignedAlgo| {
                                    cublasLtMatmul(
                                        ctx.handle,
                                        desc.0,
                                        tune_alpha_ptr,
                                        weight as *const c_void,
                                        la.0,
                                        act as *const c_void,
                                        lb.0,
                                        tune_beta_ptr,
                                        tune_out as *const c_void,
                                        ld_.0,
                                        tune_out as *mut c_void,
                                        ld_.0,
                                        algo.as_ptr().cast(),
                                        ws as *mut c_void,
                                        ctx.ws_size,
                                        stream as *mut c_void,
                                    )
                                };
                                if let Some(&first_idx) = candidates.first() {
                                    let _ = run_algo(&results[first_idx].algo);
                                }
                                let mut times = vec![f32::NAN; returned_count];
                                let mut configs: Vec<Option<plan_cache::AlgoConfig>> =
                                    vec![None; returned_count];
                                for &i in &candidates {
                                    let r = &results[i];
                                    configs[i] = plan_cache::extract_algo_config(&r.algo);
                                    if run_algo(&r.algo) != 0 {
                                        continue;
                                    }
                                    let mut ok = cuEventRecord(events[0], stream) == 0;
                                    if ok {
                                        for rep in 0..timed_reps {
                                            if run_algo(&r.algo) != 0
                                                || cuEventRecord(events[rep + 1], stream) != 0
                                            {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if !ok || cuEventSynchronize(events[timed_reps]) != 0 {
                                        continue;
                                    }
                                    let mut samples = [f32::NAN; AUTOTUNE_TIMED_REPS];
                                    for rep in 0..timed_reps {
                                        let mut ms = 0.0f32;
                                        if cuEventElapsedTime(&mut ms, events[rep], events[rep + 1])
                                            == 0
                                        {
                                            samples[rep] = ms;
                                        }
                                    }
                                    if let Some(per) = reduce_trial_times_ms(&samples[..timed_reps])
                                    {
                                        times[i] = per;
                                    }
                                }
                                let best_opt = select_fp8_live_winner(
                                    &times,
                                    &configs,
                                    ctx.autotune,
                                    false,
                                    |_| true,
                                );
                                let (best, best_ms) = match best_opt {
                                    Some(idx) => (idx, times[idx]),
                                    None => (0, f32::INFINITY),
                                };
                                selected_algo = results[best].algo;
                                tracing::info!(
                                    "cuBLASLt {label} tune {m}x{n}x{k} ldc={ldc} dev_scalars={device_scalars}: algo[{best}] of {returned_count} ({} distinct) @ {best_ms:.3}ms (heur0={:.3}ms)",
                                    candidates.len(),
                                    times[0],
                                );
                                if best_ms.is_finite()
                                    && let Some(ref ckey) = disk_key
                                {
                                    plan_cache::record_winner(ckey, &selected_algo, best, best_ms);
                                }
                            }
                            for &ev in events.iter().take(timed_reps + 1) {
                                if ev != 0 {
                                    let _ = cuEventDestroy_v2(ev);
                                }
                            }
                        }
                    }
                }
                let mut guard = plans.lock().expect("block-scaled plan cache lock poisoned");
                if let Some(&existing) = guard.get(&key) {
                    // Another thread raced us and inserted first; RAII guards destroy our handles.
                    existing
                } else {
                    let built: &'static BlockScaledPlan = Box::leak(Box::new(BlockScaledPlan {
                        desc: std::sync::Mutex::new(desc.into_raw() as usize),
                        la: la.into_raw() as usize,
                        lb: lb.into_raw() as usize,
                        ld: ld_.into_raw() as usize,
                        algo: selected_algo,
                    }));
                    guard.insert(key, built);
                    built
                }
            }
        }
    };

    let desc_guard = plan
        .desc
        .lock()
        .expect("block-scaled descriptor lock poisoned");
    let desc = *desc_guard as cublasLtMatmulDesc_t;
    let (host_alpha, host_beta) = match scalars {
        BlockScaleScalars::Host { alpha, beta } => (alpha, beta),
        BlockScaleScalars::Device { .. } => (0.0f32, 0.0f32),
    };
    let (alpha_ptr, beta_ptr): (*const c_void, *const c_void) = match scalars {
        BlockScaleScalars::Host { .. } => (
            (&host_alpha as *const f32).cast(),
            (&host_beta as *const f32).cast(),
        ),
        BlockScaleScalars::Device { alpha_ptr, beta } => {
            let b_dev = match beta {
                DeviceBeta::Zero => ctx.device_zero_f32(),
                DeviceBeta::One => ctx.device_one_f32(),
                DeviceBeta::DevicePtr(ptr) => ptr,
            };
            (alpha_ptr as *const c_void, b_dev as *const c_void)
        }
    };
    // SAFETY: `desc` is exclusively locked for this call; layouts, `AlignedAlgo`, dedicated
    // per-stream workspace `ws`, and validated device pointers match the configured block-scaled
    // matmul descriptor.
    unsafe {
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_A_SCALE_POINTER,
                (&weight_scale as *const u64).cast(),
                std::mem::size_of::<u64>(),
            ),
            "AScalePtr(block_scaled)",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_B_SCALE_POINTER,
                (&act_scale as *const u64).cast(),
                std::mem::size_of::<u64>(),
            ),
            "BScalePtr(block_scaled)",
        )?;
        chk(
            cublasLtMatmul(
                ctx.handle,
                desc,
                alpha_ptr,
                weight as *const c_void,
                plan.la as cublasLtMatrixLayout_t,
                act as *const c_void,
                plan.lb as cublasLtMatrixLayout_t,
                beta_ptr,
                out as *const c_void,
                plan.ld as cublasLtMatrixLayout_t,
                out as *mut c_void,
                plan.ld as cublasLtMatrixLayout_t,
                plan.algo.as_ptr().cast(),
                ws as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            ),
            label,
        )
    }
}

/// Row-major native CUDA 13.0 block-scaled NVFP4 GEMM (`CUDA_R_4F_E2M1` +
/// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`) with BF16 output and FP32 accumulation.
///
/// Computes `out[m, n] = alpha * (act[m, k] * act_scale) @ (weight[n, k] * weight_scale)^T + beta * out[m, n]`:
/// - `act`: row-major packed `E2M1` (`[m, k / 2]` bytes, 2 nibbles/byte, low nibble = even `k`).
/// - `act_scale`: `128x4`-swizzled `UE4M3` block scales (1 byte per 16 elements along `k`,
///   sized [`block_scale_padded_bytes`]`(m, k, NVFP4_BLOCK_SIZE)`).
/// - `weight`: row-major packed `E2M1` (`[n, k / 2]` bytes).
/// - `weight_scale`: `128x4`-swizzled `UE4M3` block scales
///   (sized [`block_scale_padded_bytes`]`(n, k, NVFP4_BLOCK_SIZE)`).
/// - `out`: row-major BF16 `[m, n]` with row stride `ldc >= n` (elements).
/// - `scalars`: host `(alpha, beta)` or device-pointer `alpha` (global tensor scale product
///   `s_a_global * s_b_global`) with [`DeviceBeta`].
///
/// # Errors
/// Returns an error if any dimension is zero, `k` is not a multiple of 32, `n` or `ldc` is not
/// a multiple of 8, `ldc < n`, any device pointer is null or misaligned, host scalars are
/// non-finite, or cuBLASLt fails to select or launch a kernel on the active GPU.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_block_scaled(
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    scalars: BlockScaleScalars,
    stream: u64,
) -> Result<()> {
    block_scaled_gemm_impl(
        BlockScaledFormat::Nvfp4,
        act,
        act_scale,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        ldc,
        scalars,
        stream,
    )
}

/// Row-major native CUDA 13.0 block-scaled MXFP8 GEMM (`CUDA_R_8F_E4M3` +
/// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`) with BF16 output and FP32 accumulation.
///
/// Computes `out[m, n] = alpha * (act[m, k] * act_scale) @ (weight[n, k] * weight_scale)^T + beta * out[m, n]`:
/// - `act`: row-major FP8 `E4M3` (`[m, k]` bytes).
/// - `act_scale`: `128x4`-swizzled `UE8M0` block scales (1 byte per 32 elements along `k`,
///   sized [`block_scale_padded_bytes`]`(m, k, MXFP8_BLOCK_SIZE)`).
/// - `weight`: row-major FP8 `E4M3` (`[n, k]` bytes).
/// - `weight_scale`: `128x4`-swizzled `UE8M0` block scales
///   (sized [`block_scale_padded_bytes`]`(n, k, MXFP8_BLOCK_SIZE)`).
/// - `out`: row-major BF16 `[m, n]` with row stride `ldc >= n` (elements).
///
/// # Errors
/// Returns an error if any dimension is zero, `k` is not a multiple of 32, `n` or `ldc` is not
/// a multiple of 8, `ldc < n`, any device pointer is null or misaligned, `alpha`/`beta` are
/// non-finite, or cuBLASLt fails to select or launch a kernel on the active GPU.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemm_block_scaled(
    act: u64,
    act_scale: u64,
    weight: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    alpha: f32,
    beta: f32,
    stream: u64,
) -> Result<()> {
    block_scaled_gemm_impl(
        BlockScaledFormat::Mxfp8,
        act,
        act_scale,
        weight,
        weight_scale,
        out,
        m,
        n,
        k,
        ldc,
        BlockScaleScalars::Host { alpha, beta },
        stream,
    )
}

// ── GLM-5.3 GEMM entry points ────────────────────────────────────────────────
// Ported from perf/glm-5.3-flash-prefill-20260920. They use their own
// first-heuristic core (`gemm_impl_batched`) so the GLM path keeps its exact
// algorithm choice; `bf16_gemm_act_weight_t` above is shared and unchanged
// (it too takes the first heuristic, so GLM's calls to it are numerically the
// same as on the GLM branch).

/// Row-major `out[M,N] = act[M,K] @ weight[K,N]`, all BF16.
pub fn bf16_gemm_act_weight(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm(act, weight, out, m, n, k, stream, false)
}

#[allow(clippy::too_many_arguments)]
fn bf16_gemm(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
) -> Result<()> {
    bf16_gemm_impl(
        act,
        weight,
        out,
        m,
        n,
        k,
        stream,
        weight_is_nk,
        None,
        CUDA_R_16BF,
    )
    .map(|_| ())
}

/// Strided-batched row-major BF16 GEMM: per batch `out = act @ weight[K,N]` (or
/// `@ weight[N,K]ᵀ` when `weight_is_nk`). Used by the GLM DSA absorption banks
/// (64 heads).
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_batched(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    weight_is_nk: bool,
    batch: StridedBatch,
    stream: u64,
) -> Result<()> {
    gemm_impl_batched(
        act,
        weight,
        out,
        m,
        n,
        k,
        stream,
        weight_is_nk,
        None,
        CUDA_R_16BF,
        Some(batch),
    )
    .map(|_| ())
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all F32 (full fp32, no TF32).
/// Used by the GLM prompt-scope router logits and mHC mixing projections.
pub fn f32_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_impl(act, weight, out, m, n, k, stream, true, None, CUDA_R_32F).map(|_| ())
}

/// Row-major `out[M,N] = act[M,K] @ weight[K,N]`, all F16 (fp32 accumulate).
/// Used by the GLM EXL3 reconstructed-weight prefill projections.
pub fn f16_gemm_act_weight(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    bf16_gemm_impl(act, weight, out, m, n, k, stream, false, None, CUDA_R_16F).map(|_| ())
}

/// Same projection core with explicit experimental reduction selection and a
/// receipt for its actual heuristic. Caller owns buffers and completion, and
/// must serialize use of the process-global workspace just as in production.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_diagnostic(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    policy: ReductionPolicy,
) -> Result<Bf16GemmReceipt> {
    anyhow::ensure!(
        act != 0 && weight != 0 && out != 0 && m > 0 && n > 0 && k > 0,
        "invalid diagnostic GEMM pointers/dimensions"
    );
    bf16_gemm_impl(
        act,
        weight,
        out,
        m,
        n,
        k,
        stream,
        true,
        Some(policy),
        CUDA_R_16BF,
    )?
    .ok_or_else(|| anyhow::anyhow!("missing explicit GEMM receipt"))
}

#[allow(clippy::too_many_arguments)]
fn bf16_gemm_impl(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
    policy: Option<ReductionPolicy>,
    dtype: i32,
) -> Result<Option<Bf16GemmReceipt>> {
    gemm_impl_batched(
        act,
        weight,
        out,
        m,
        n,
        k,
        stream,
        weight_is_nk,
        policy,
        dtype,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl_batched(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    weight_is_nk: bool,
    policy: Option<ReductionPolicy>,
    dtype: i32,
    batch: Option<StridedBatch>,
) -> Result<Option<Bf16GemmReceipt>> {
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    // SAFETY: All cuBLASLt descriptors, layouts, and preferences are wrapped in RAII guards so
    // early error returns destroy every handle cleanly. `result` is an 8-byte-aligned
    // `HeuristicResult` matching `cublasLtMatmulHeuristicResult_t`; `ws` is a dedicated per-stream
    // device workspace of `ctx.ws_size` bytes; caller supplies valid device pointers and stream.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let ta = if weight_is_nk {
            CUBLAS_OP_T
        } else {
            CUBLAS_OP_N
        };
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let (weight_rows, weight_cols, weight_ld) = if weight_is_nk {
            (k as u64, n as u64, k as i64)
        } else {
            (n as u64, k as u64, n as i64)
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, dtype, weight_rows, weight_cols, weight_ld),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        let ld_act = match batch {
            Some(b) if b.ld_act > 0 => b.ld_act,
            _ => k as i64,
        };
        let ld_out = match batch {
            Some(b) if b.ld_out > 0 => b.ld_out,
            _ => n as i64,
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut lb_raw, dtype, k as u64, m as u64, ld_act),
            "LayoutB",
        )?;
        let lb = LayoutGuard(lb_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_raw, dtype, n as u64, m as u64, ld_out),
            "LayoutD",
        )?;
        let ld_ = LayoutGuard(ld_raw);
        if let Some(batch) = batch {
            for (layout, stride) in [
                (la.0, batch.stride_weight),
                (lb.0, batch.stride_act),
                (ld_.0, batch.stride_out),
            ] {
                chk(
                    cublasLtMatrixLayoutSetAttribute(
                        layout,
                        LAYOUT_BATCH_COUNT,
                        &batch.count as *const i32 as *const c_void,
                        4,
                    ),
                    "LayoutBatchCount",
                )?;
                chk(
                    cublasLtMatrixLayoutSetAttribute(
                        layout,
                        LAYOUT_STRIDED_BATCH_OFFSET,
                        &stride as *const i64 as *const c_void,
                        8,
                    ),
                    "LayoutBatchStride",
                )?;
            }
        }
        let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
        let pref = PrefGuard(pref_raw);
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        // Always retain the original first-heuristic control. Only the explicit
        // diagnostic API supplies a policy or asks for algorithm attributes.
        if let Some(policy) = policy {
            diagnostic::configure(pref.0, policy)?;
        }
        let mut result = HeuristicResult::default();
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc.0,
                la.0,
                lb.0,
                ld_.0,
                ld_.0,
                pref.0,
                1,
                (&mut result as *mut HeuristicResult).cast(),
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        result
            .admit(returned, ctx.ws_size)
            .map_err(anyhow::Error::msg)?;
        let receipt = policy
            .map(|policy| diagnostic::receipt(&result, policy, [m, n, k], ctx.ws_size))
            .transpose()?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc.0,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la.0,
            act as *const c_void,
            lb.0,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_.0,
            out as *mut c_void,
            ld_.0,
            result.algo.as_ptr().cast(),
            ws as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "Matmul")?;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_requested_candidates_widens_golden_prefill_shapes_to_16() {
        let golden_shapes = [
            (8192u32, 16384u32),
            (8192, 2560),
            (8192, 12288),
            (16384, 2560),
            (8192, 512),
            (16384, 16384),
        ];
        for (m, n) in golden_shapes {
            assert_eq!(
                fp8_requested_candidates(false, true, false, false, m, n),
                16,
                "golden prefill shape {m}x{n} must search 16 heuristic candidates"
            );
        }
        // CUDA graph capture must never run live autotuning.
        assert_eq!(
            fp8_requested_candidates(true, true, true, false, 8192, 16384),
            1
        );
        // Explicit disable (`ATLAS_CUBLASLT_AUTOTUNE=0`) disables default tuning.
        assert_eq!(
            fp8_requested_candidates(false, false, false, false, 8192, 16384),
            1
        );
        // Small or outer-scaled shapes stay single-heuristic unless `ctx.autotune` is forced.
        assert_eq!(
            fp8_requested_candidates(false, true, false, false, 256, 4096),
            1
        );
        assert_eq!(
            fp8_requested_candidates(false, true, false, true, 8192, 16384),
            1
        );
        assert_eq!(
            fp8_requested_candidates(true, true, false, true, 8192, 16384),
            16
        );
    }

    #[test]
    fn select_distinct_candidates_deduplicates_configs_and_preserves_order() {
        let ws_limit = 64 * 1024 * 1024;
        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 16 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 9];

        // 0: valid heur0 (algo_id=67, tile_id=23, split_k=1)
        cands[0].algo.0[0] = 67;
        cands[0].algo.0[1] = 23;
        cands[0].algo.0[2] = 1;
        // 1: invalid state -> skipped
        cands[1].algo.0[0] = 67;
        cands[1].algo.0[1] = 30;
        cands[1].algo.0[2] = 1;
        cands[1].state = 7;
        // 2: workspace exceeds budget -> skipped
        cands[2].algo.0[0] = 67;
        cands[2].algo.0[1] = 31;
        cands[2].algo.0[2] = 1;
        cands[2].workspace_size = 128 * 1024 * 1024;
        // 3: exact duplicate config of index 0 -> deduplicated
        cands[3].algo.0[0] = 67;
        cands[3].algo.0[1] = 23;
        cands[3].algo.0[2] = 1;
        // 4: same algo_id=67 but distinct tile_id=46 -> preserved
        cands[4].algo.0[0] = 67;
        cands[4].algo.0[1] = 46;
        cands[4].algo.0[2] = 1;
        // 5: non-deterministic split_k=2 -> filtered out by `eligible`
        cands[5].algo.0[0] = 35;
        cands[5].algo.0[1] = 20;
        cands[5].algo.0[2] = 2;
        // 6: exact duplicate config of index 4 -> deduplicated
        cands[6].algo.0[0] = 67;
        cands[6].algo.0[1] = 46;
        cands[6].algo.0[2] = 1;
        // 7: raw fallback candidate (attribute extraction unavailable) -> preserved
        cands[7].algo.0[0] = 0xff;
        cands[7].algo.0[2] = 1;
        cands[7].algo.0[5] = 42;
        // 8: duplicate raw fallback of index 7 -> deduplicated
        cands[8].algo.0[0] = 0xff;
        cands[8].algo.0[2] = 1;
        cands[8].algo.0[5] = 42;

        let selected = select_distinct_candidates(
            &cands,
            ws_limit,
            |a| {
                if a.0[0] == 0xff {
                    CandidateIdentity::Raw(*a)
                } else {
                    CandidateIdentity::Config([
                        u32::from(a.0[0]),
                        u32::from(a.0[1]),
                        u32::from(a.0[2]),
                        0,
                        0,
                        0,
                        36,
                        0,
                        0,
                    ])
                }
            },
            |idx, cand| idx == 0 || cand.algo.0[2] <= 1,
        );
        assert_eq!(selected, [0, 4, 7]);
    }

    #[test]
    fn reduce_trial_times_ms_rejects_cold_l2_outliers_and_dma_stalls() {
        // 3 trials with a cold-L2 fast first trial: median rejects the 3.08ms illusion.
        let cold_l2 = reduce_trial_times_ms(&[3.08, 4.64, 4.68]).expect("valid 3-sample median");
        assert!((cold_l2 - 4.64).abs() < 1e-6, "got {cold_l2}");

        // 3 trials with a single transient DMA stall: median rejects the 9.20ms spike.
        let dma_stall = reduce_trial_times_ms(&[3.55, 9.20, 3.58]).expect("valid 3-sample median");
        assert!((dma_stall - 3.58).abs() < 1e-6, "got {dma_stall}");

        // 2 trials: conservative upper-middle prevents cold-L2 first iter from winning.
        let two_rep = reduce_trial_times_ms(&[3.08, 4.64]).expect("valid 2-sample reduction");
        assert!((two_rep - 4.64).abs() < 1e-6, "got {two_rep}");

        // 5 trials: exact median rejects both fast and slow tails.
        let five_rep =
            reduce_trial_times_ms(&[3.08, 3.12, 4.64, 4.70, 9.10]).expect("valid 5-sample median");
        assert!((five_rep - 4.64).abs() < 1e-6, "got {five_rep}");

        // Invalid readings are filtered out; empty/all-invalid returns None.
        let filtered = reduce_trial_times_ms(&[f32::NAN, 0.0, -1.0, 3.50, f32::INFINITY])
            .expect("single valid reading");
        assert!((filtered - 3.50).abs() < 1e-6, "got {filtered}");
        assert!(reduce_trial_times_ms(&[f32::NAN, 0.0, -2.0]).is_none());
    }

    #[test]
    fn select_fp8_live_winner_blocks_sm89_xmma_and_enforces_3pct_margin_on_sm121a_large() {
        let nvjet_a: plan_cache::AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];
        let nvjet_b: plan_cache::AlgoConfig = [67, 23, 1, 0, 0, 31, 36, 0, 0];
        let sm89_xmma: plan_cache::AlgoConfig = [35, 20, 1, 0, 1, 0, 15, 0, 0];

        // 1. Cold-L0 sm89_xmma illusion (3.08ms) must NOT beat healthy native nvjet heur0 (3.50ms),
        //    in either default mode or `ctx.autotune = true`.
        let times = [3.50f32, 3.08, 3.42];
        let configs = [Some(nvjet_a), Some(sm89_xmma), Some(nvjet_b)];
        for autotune in [false, true] {
            let winner = select_fp8_live_winner(&times, &configs, autotune, true, |ms| ms < 5.0);
            assert_eq!(
                winner,
                Some(0),
                "heur0 nvjet must beat sm89_xmma illusion and <3%-faster sibling (autotune={autotune})"
            );
        }

        // 2. Native Blackwell candidate >3% faster than heur0 (3.35ms vs 3.50ms = -4.3%) wins.
        let times_faster_nvjet = [3.50f32, 3.08, 3.35];
        let winner =
            select_fp8_live_winner(&times_faster_nvjet, &configs, false, true, |ms| ms < 5.0);
        assert_eq!(winner, Some(2));

        // 3. Degraded heur0 (13.8ms, unhealthy) is cleanly replaced by healthy native Blackwell.
        let times_degraded_h0 = [13.80f32, 3.08, 3.55];
        let winner =
            select_fp8_live_winner(&times_degraded_h0, &configs, false, true, |ms| ms < 5.0);
        assert_eq!(winner, Some(2));

        // 4. Fallback: if NO native Blackwell candidate is healthy, sm89_xmma can still run.
        let times_only_xmma = [13.80f32, 3.50, f32::NAN];
        let winner = select_fp8_live_winner(&times_only_xmma, &configs, false, true, |ms| ms < 5.0);
        assert_eq!(winner, Some(1));

        // 5. Non-large / non-SM121a (`guard_sm121a_large = false`) preserves 2% default hysteresis
        //    and fastest-wins under `ctx_autotune = true`.
        let times_small = [1.00f32, 0.985, 0.975];
        assert_eq!(
            select_fp8_live_winner(&times_small, &configs, false, false, |_| true),
            Some(2),
            "2.5% speedup beats 2% default hysteresis"
        );
        let times_small_tie = [1.00f32, 0.985];
        assert_eq!(
            select_fp8_live_winner(&times_small_tie, &configs[..2], false, false, |_| true),
            Some(0),
            "1.5% speedup stays on heur0 under 2% default hysteresis"
        );
        assert_eq!(
            select_fp8_live_winner(&times_small_tie, &configs[..2], true, false, |_| true),
            Some(1),
            "ctx_autotune picks fastest when not under SM121a large guard"
        );
    }

    #[test]
    fn cuda13_block_scaled_enum_constants_match_headers() {
        // Verified directly against `/usr/local/cuda/include/library_types.h` and `cublasLt.h`.
        assert_eq!(CUDA_R_8F_E4M3, 28);
        assert_eq!(CUDA_R_8F_UE4M3, 28);
        assert_eq!(CUDA_R_8F_UE8M0, 30);
        assert_eq!(CUDA_R_4F_E2M1, 33);
        assert_eq!(CUDA_R_16BF, 14);
        assert_eq!(CUDA_R_32F, 0);
        assert_eq!(CUBLAS_COMPUTE_32F, 68);
        assert_eq!(MATRIX_SCALE_VEC16_UE4M3, 1);
        assert_eq!(MATRIX_SCALE_VEC32_UE8M0, 2);
        assert_eq!(DESC_POINTER_MODE, 2);
        assert_eq!(DESC_A_SCALE_POINTER, 17);
        assert_eq!(DESC_B_SCALE_POINTER, 18);
        assert_eq!(DESC_FAST_ACCUM, 25);
        assert_eq!(DESC_A_SCALE_MODE, 31);
        assert_eq!(DESC_B_SCALE_MODE, 32);
        assert_eq!(POINTER_MODE_HOST, 0);
        assert_eq!(POINTER_MODE_DEVICE, 1);
    }

    #[test]
    fn block_scale_swizzle_layout_is_bijective_and_matches_128x4_tiling() {
        let cases = [
            (1u32, 32u32, NVFP4_BLOCK_SIZE),
            (16, 64, NVFP4_BLOCK_SIZE),
            (128, 128, NVFP4_BLOCK_SIZE),
            (137, 160, NVFP4_BLOCK_SIZE),
            (256, 256, NVFP4_BLOCK_SIZE),
            (1, 32, MXFP8_BLOCK_SIZE),
            (128, 128, MXFP8_BLOCK_SIZE),
            (137, 160, MXFP8_BLOCK_SIZE),
            (257, 288, MXFP8_BLOCK_SIZE),
        ];
        for (outer, k, blk) in cases {
            let total = block_scale_padded_bytes(outer, k, blk);
            let padded_outer = outer.div_ceil(128) * 128;
            let k_blocks = k / blk;
            let padded_k_blocks = k_blocks.div_ceil(4) * 4;
            assert_eq!(
                total,
                (padded_outer as usize) * (padded_k_blocks as usize),
                "padded bytes mismatch for outer={outer} k={k} blk={blk}"
            );
            let mut seen = vec![false; total];
            for row in 0..padded_outer {
                for g in 0..padded_k_blocks {
                    let off = block_scale_swizzle_offset(row, g, k_blocks);
                    assert!(
                        off < total,
                        "offset {off} >= {total} for row={row} g={g} outer={outer} k={k}"
                    );
                    assert!(
                        !seen[off],
                        "duplicate offset {off} for row={row} g={g} outer={outer} k={k}"
                    );
                    seen[off] = true;
                }
            }
            assert!(
                seen.iter().all(|&v| v),
                "swizzle did not cover full buffer for outer={outer} k={k} blk={blk}"
            );
        }
        assert_eq!(block_scale_padded_bytes(0, 64, NVFP4_BLOCK_SIZE), 0);
        assert_eq!(block_scale_padded_bytes(64, 0, NVFP4_BLOCK_SIZE), 0);
        assert_eq!(block_scale_padded_bytes(64, 64, 0), 0);
    }

    #[test]
    fn block_scaled_plan_cache_tags_are_disjoint_from_fp8_and_bf16() {
        let nvfp4_tag = BlockScaledFormat::Nvfp4.plan_cache_at_tag();
        let mxfp8_tag = BlockScaledFormat::Mxfp8.plan_cache_at_tag();
        assert_ne!(nvfp4_tag, mxfp8_tag);
        for existing in [CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F] {
            assert_ne!(
                nvfp4_tag, existing,
                "NVFP4 tag must not collide with {existing}"
            );
            assert_ne!(
                mxfp8_tag, existing,
                "MXFP8 tag must not collide with {existing}"
            );
        }
    }

    #[test]
    fn block_scaled_arg_validation_rejects_bad_shapes_alignment_and_scalars() {
        let ok_ptr = 0x1000u64;
        // Zero dimensions.
        assert!(
            nvfp4_gemm_block_scaled(
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                0,
                32,
                32,
                32,
                BlockScaleScalars::default(),
                0
            )
            .is_err()
        );
        assert!(
            mxfp8_gemm_block_scaled(
                ok_ptr, ok_ptr, ok_ptr, ok_ptr, ok_ptr, 32, 0, 32, 32, 1.0, 0.0, 0
            )
            .is_err()
        );
        assert!(
            mxfp8_gemm_block_scaled(
                ok_ptr, ok_ptr, ok_ptr, ok_ptr, ok_ptr, 32, 32, 0, 32, 1.0, 0.0, 0
            )
            .is_err()
        );
        // k not a multiple of 32.
        for bad_k in [16u32, 24, 48] {
            assert!(
                nvfp4_gemm_block_scaled(
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    32,
                    32,
                    bad_k,
                    32,
                    BlockScaleScalars::default(),
                    0
                )
                .is_err(),
                "NVFP4 must reject k={bad_k}"
            );
            assert!(
                mxfp8_gemm_block_scaled(
                    ok_ptr, ok_ptr, ok_ptr, ok_ptr, ok_ptr, 32, 32, bad_k, 32, 1.0, 0.0, 0
                )
                .is_err(),
                "MXFP8 must reject k={bad_k}"
            );
        }
        // ldc < n and non-multiple-of-8 n/ldc.
        assert!(
            nvfp4_gemm_block_scaled(
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                16,
                32,
                32,
                16,
                BlockScaleScalars::default(),
                0
            )
            .is_err()
        );
        assert!(
            nvfp4_gemm_block_scaled(
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                16,
                12,
                32,
                16,
                BlockScaleScalars::default(),
                0
            )
            .is_err()
        );
        assert!(
            mxfp8_gemm_block_scaled(
                ok_ptr, ok_ptr, ok_ptr, ok_ptr, ok_ptr, 16, 16, 32, 20, 1.0, 0.0, 0
            )
            .is_err()
        );
        // Null and misaligned data/scale pointers.
        for bad_ptr in [0u64, ok_ptr + 8] {
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Nvfp4,
                    bad_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    32,
                    32,
                    32,
                    32,
                    BlockScaleScalars::default(),
                )
                .is_err()
            );
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Nvfp4,
                    ok_ptr,
                    bad_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    32,
                    32,
                    32,
                    32,
                    BlockScaleScalars::default(),
                )
                .is_err()
            );
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Mxfp8,
                    ok_ptr,
                    ok_ptr,
                    bad_ptr,
                    ok_ptr,
                    ok_ptr,
                    32,
                    32,
                    32,
                    32,
                    BlockScaleScalars::default(),
                )
                .is_err()
            );
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Mxfp8,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    bad_ptr,
                    ok_ptr,
                    32,
                    32,
                    32,
                    32,
                    BlockScaleScalars::default(),
                )
                .is_err()
            );
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Mxfp8,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    bad_ptr,
                    32,
                    32,
                    32,
                    32,
                    BlockScaleScalars::default(),
                )
                .is_err()
            );
        }
        // Invalid host / device scalars.
        for bad_scalars in [
            BlockScaleScalars::host(f32::NAN, 0.0),
            BlockScaleScalars::host(1.0, f32::INFINITY),
            BlockScaleScalars::device_alpha(0),
            BlockScaleScalars::device_alpha(ok_ptr + 2),
            BlockScaleScalars::device(ok_ptr, 0),
            BlockScaleScalars::device(ok_ptr, ok_ptr + 2),
        ] {
            assert!(
                validate_block_scaled_args(
                    BlockScaledFormat::Nvfp4,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    ok_ptr,
                    32,
                    32,
                    32,
                    32,
                    bad_scalars,
                )
                .is_err(),
                "must reject invalid scalars {bad_scalars:?}"
            );
        }
        // Valid args pass cleanly.
        assert!(
            validate_block_scaled_args(
                BlockScaledFormat::Nvfp4,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                ok_ptr,
                1,
                16,
                32,
                16,
                BlockScaleScalars::device_alpha_accumulate(ok_ptr + 4),
            )
            .is_ok()
        );
    }

    #[test]
    fn block_scaled_requested_candidates_respects_capture_and_shape() {
        assert_eq!(
            block_scaled_requested_candidates(true, true, true, 1024, 1024),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates(false, false, false, 1024, 1024),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates(false, true, false, 256, 1024),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates(false, true, false, 512, 512),
            16
        );
        assert_eq!(
            block_scaled_requested_candidates(true, false, false, 32, 32),
            16
        );
    }

    #[test]
    fn sm120_candidate_gate_widens_decode_verify_shapes_and_respects_capture() {
        let sm120 = Some((12, 0));
        let sm121 = Some((12, 1));

        // SM120 (12, 0) widens decode/verify shapes m >= 16, n >= 256 to 16 candidates
        // for both FP8 and block-scaled GEMMs outside CUDA graph capture.
        for (m, n) in [
            (16u32, 256u32),
            (24, 320),
            (32, 2560),
            (48, 2560),
            (64, 12288),
            (128, 16384),
            (256, 4096),
            (512, 512),
        ] {
            assert_eq!(
                fp8_requested_candidates_for_cc(false, true, false, false, m, n, sm120),
                16,
                "SM120 FP8 shape {m}x{n} must search 16 candidates"
            );
            assert_eq!(
                block_scaled_requested_candidates_for_cc(false, true, false, m, n, sm120),
                16,
                "SM120 block-scaled shape {m}x{n} must search 16 candidates"
            );
        }

        // Below m=16 or n=256 stays single-heuristic unless ctx.autotune is active.
        assert_eq!(
            fp8_requested_candidates_for_cc(false, true, false, false, 15, 2560, sm120),
            1
        );
        assert_eq!(
            fp8_requested_candidates_for_cc(false, true, false, false, 32, 128, sm120),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates_for_cc(false, true, false, 15, 2560, sm120),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates_for_cc(false, true, false, 32, 128, sm120),
            1
        );
        assert_eq!(
            fp8_requested_candidates_for_cc(true, false, false, false, 16, 256, sm120),
            16
        );

        // CUDA graph capture and explicit ATLAS_CUBLASLT_AUTOTUNE=0 never tune.
        assert_eq!(
            fp8_requested_candidates_for_cc(true, true, true, false, 64, 2560, sm120),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates_for_cc(true, true, true, 64, 2560, sm120),
            1
        );
        assert_eq!(
            fp8_requested_candidates_for_cc(false, false, false, false, 64, 2560, sm120),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates_for_cc(false, false, false, 64, 2560, sm120),
            1
        );

        // GB10 SM121a (12, 1) keeps exact m >= 512 && n >= 512 default gate unchanged.
        assert_eq!(
            fp8_requested_candidates_for_cc(false, true, false, false, 256, 4096, sm121),
            1
        );
        assert_eq!(
            block_scaled_requested_candidates_for_cc(false, true, false, 256, 1024, sm121),
            1
        );
        assert_eq!(
            fp8_requested_candidates_for_cc(false, true, false, false, 512, 512, sm121),
            16
        );

        // Split-K exploration engages on SM120 decode/verify shapes (m in 16..512) and
        // under ctx.autotune, while keeping SM120 prefill and SM121a deterministic.
        assert!(allow_splitk_candidates(false, true, 16, sm120));
        assert!(allow_splitk_candidates(false, true, 64, sm120));
        assert!(allow_splitk_candidates(false, true, 256, sm120));
        assert!(!allow_splitk_candidates(false, true, 15, sm120));
        assert!(!allow_splitk_candidates(false, true, 512, sm120));
        assert!(!allow_splitk_candidates(false, false, 64, sm120));
        assert!(!allow_splitk_candidates(false, true, 64, sm121));
        assert!(allow_splitk_candidates(true, false, 64, sm121));
        assert!(allow_splitk_candidates(true, false, 512, sm120));
    }

    fn decode_e2m1(nibble: u8) -> f32 {
        const LUT: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        LUT[(nibble & 0x0f) as usize]
    }

    fn decode_fp8_e4m3(byte: u8) -> f32 {
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 3) & 0x0f;
        let man = byte & 0x07;
        if exp == 15 && man == 7 {
            return f32::NAN;
        }
        if exp == 0 {
            sign * f32::from(man) * 0.001_953_125
        } else {
            sign * (1.0 + f32::from(man) * 0.125) * 2.0f32.powi(i32::from(exp) - 7)
        }
    }

    fn decode_ue8m0(byte: u8) -> f32 {
        if byte == 255 {
            return f32::NAN;
        }
        2.0f32.powi(i32::from(byte) - 127)
    }

    fn swizzle_block_scales(logical: &[u8], outer: u32, k_blocks: u32) -> Vec<u8> {
        let padded_outer = outer.div_ceil(128) * 128;
        let padded_k_blocks = k_blocks.div_ceil(4) * 4;
        let mut out = vec![0u8; (padded_outer as usize) * (padded_k_blocks as usize)];
        for row in 0..outer {
            for g in 0..k_blocks {
                let src = (row as usize) * (k_blocks as usize) + (g as usize);
                let dst = block_scale_swizzle_offset(row, g, k_blocks);
                out[dst] = logical[src];
            }
        }
        out
    }

    fn compare_bf16_vs_ref(
        got_bf16: &[u16],
        expected_f32: &[f32],
        m: u32,
        n: u32,
        ldc: u32,
    ) -> (f32, f64, u32) {
        let mut max_abs_err = 0.0f32;
        let mut max_bf16_ulp = 0u32;
        let mut dot = 0.0f64;
        let mut norm_g = 0.0f64;
        let mut norm_e = 0.0f64;
        for row in 0..(m as usize) {
            for col in 0..(n as usize) {
                let g_bits = got_bf16[row * (ldc as usize) + col];
                let g = half::bf16::from_bits(g_bits).to_f32();
                let e_f32 = expected_f32[row * (n as usize) + col];
                let e_bf16 = half::bf16::from_f32(e_f32);
                let e_rounded = e_bf16.to_f32();
                assert!(
                    g.is_finite(),
                    "non-finite GPU output at ({row}, {col}): bits={g_bits:#06x}"
                );
                let abs_err = (g - e_rounded).abs();
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                }
                let ulp = if g == 0.0 && e_rounded == 0.0 {
                    0
                } else {
                    g_bits.abs_diff(e_bf16.to_bits()) as u32
                };
                if ulp > max_bf16_ulp {
                    max_bf16_ulp = ulp;
                }
                dot += f64::from(g) * f64::from(e_f32);
                norm_g += f64::from(g) * f64::from(g);
                norm_e += f64::from(e_f32) * f64::from(e_f32);
            }
        }
        let cos_sim = if norm_g > 0.0 && norm_e > 0.0 {
            dot / (norm_g.sqrt() * norm_e.sqrt())
        } else {
            1.0
        };
        (max_abs_err, cos_sim, max_bf16_ulp)
    }

    #[test]
    #[ignore = "requires CUDA 13.0 Blackwell GPU (SM100+/SM120+/SM121)"]
    fn real_gpu_nvfp4_block_scaled_gemm_matches_cpu_fp32_reference() {
        use crate::gpu::GpuBackend;

        let gpu =
            crate::cuda_backend::AtlasCudaBackend::new(0, &[]).expect("init AtlasCudaBackend");
        let stream = 0u64;

        // Distinct positive UE4M3 scales spanning 0.25 .. 2.5 (sign bit = 0).
        const SCALE_POOL: [u8; 8] = [0x28, 0x2c, 0x30, 0x34, 0x38, 0x3a, 0x3c, 0x42];

        let run_case = |label: &str,
                        m: u32,
                        n: u32,
                        k: u32,
                        ldc: u32,
                        alpha_val: f32,
                        beta_val: f32,
                        use_device_alpha: bool,
                        device_beta_mode: Option<DeviceBeta>,
                        ptr_offset_16: bool| {
            let k_blocks = k / NVFP4_BLOCK_SIZE;
            let mut a_packed = vec![0u8; (m as usize) * ((k / 2) as usize)];
            let mut b_packed = vec![0u8; (n as usize) * ((k / 2) as usize)];
            let mut a_scales_log = vec![0u8; (m as usize) * (k_blocks as usize)];
            let mut b_scales_log = vec![0u8; (n as usize) * (k_blocks as usize)];

            // Deterministic LCG seeded by label + shape so repeated calls on identical shapes
            // generate distinct data and scales to exercise cached-plan scale-pointer updates.
            let label_mix = label.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(131).wrapping_add(u64::from(b))
            });
            let mut state: u64 = 0x9e37_79b9_7f4a_7c15
                ^ label_mix
                ^ (u64::from(m) << 32)
                ^ (u64::from(n) << 16)
                ^ u64::from(k);
            let mut next_u32 = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 32) as u32
            };
            for byte in &mut a_packed {
                let lo = (next_u32() & 0x0f) as u8;
                let hi = (next_u32() & 0x0f) as u8;
                *byte = (hi << 4) | lo;
            }
            for byte in &mut b_packed {
                let lo = (next_u32() & 0x0f) as u8;
                let hi = (next_u32() & 0x0f) as u8;
                *byte = (hi << 4) | lo;
            }
            for (idx, s) in a_scales_log.iter_mut().enumerate() {
                *s = SCALE_POOL[((next_u32() as usize) ^ idx) % SCALE_POOL.len()];
            }
            for (idx, s) in b_scales_log.iter_mut().enumerate() {
                *s = SCALE_POOL[((next_u32() as usize) ^ (idx * 3)) % SCALE_POOL.len()];
            }

            let a_scales_swiz = swizzle_block_scales(&a_scales_log, m, k_blocks);
            let b_scales_swiz = swizzle_block_scales(&b_scales_log, n, k_blocks);
            assert_eq!(
                a_scales_swiz.len(),
                block_scale_padded_bytes(m, k, NVFP4_BLOCK_SIZE)
            );
            assert_eq!(
                b_scales_swiz.len(),
                block_scale_padded_bytes(n, k, NVFP4_BLOCK_SIZE)
            );

            // Pre-populate `out` with non-zero sentinel BF16 values to verify both `beta == 0`
            // overwrite, `beta != 0` accumulation, and `ldc > n` padding preservation.
            let mut init_out_bf16 = vec![0u16; (m as usize) * (ldc as usize)];
            for (idx, slot) in init_out_bf16.iter_mut().enumerate() {
                let val = (((idx % 17) as f32) - 8.0) * 0.25;
                *slot = half::bf16::from_f32(val).to_bits();
            }

            // CPU exact FP32 reference:
            // ref[row, col] = alpha * sum_j (deq_a[row, j] * deq_b[col, j]) + beta * init_c[row, col]
            let mut ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
            for row in 0..(m as usize) {
                let mut a_row = vec![0.0f32; k as usize];
                for j in 0..(k as usize) {
                    let byte = a_packed[row * ((k / 2) as usize) + (j / 2)];
                    let nib = if (j & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                    let sc = decode_fp8_e4m3(a_scales_log[row * (k_blocks as usize) + (j / 16)]);
                    a_row[j] = decode_e2m1(nib) * sc;
                }
                for col in 0..(n as usize) {
                    let mut acc = 0.0f32;
                    for j in 0..(k as usize) {
                        let byte = b_packed[col * ((k / 2) as usize) + (j / 2)];
                        let nib = if (j & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                        let sc =
                            decode_fp8_e4m3(b_scales_log[col * (k_blocks as usize) + (j / 16)]);
                        acc += a_row[j] * (decode_e2m1(nib) * sc);
                    }
                    let c_init =
                        half::bf16::from_bits(init_out_bf16[row * (ldc as usize) + col]).to_f32();
                    ref_f32[row * (n as usize) + col] = alpha_val * acc + beta_val * c_init;
                }
            }

            let pad = if ptr_offset_16 { 16usize } else { 0usize };
            let d_a_base = gpu.alloc(a_packed.len() + pad).expect("alloc a");
            let d_as_base = gpu.alloc(a_scales_swiz.len() + pad).expect("alloc a_scale");
            let d_b_base = gpu.alloc(b_packed.len() + pad).expect("alloc b");
            let d_bs_base = gpu.alloc(b_scales_swiz.len() + pad).expect("alloc b_scale");
            let out_bytes = init_out_bf16.len() * 2;
            let d_out_base = gpu.alloc(out_bytes + pad).expect("alloc out");

            let d_a = d_a_base.offset(pad);
            let d_as = d_as_base.offset(pad);
            let d_b = d_b_base.offset(pad);
            let d_bs = d_bs_base.offset(pad);
            let d_out = d_out_base.offset(pad);

            gpu.copy_h2d(&a_packed, d_a).expect("h2d a");
            gpu.copy_h2d(&a_scales_swiz, d_as).expect("h2d a_scale");
            gpu.copy_h2d(&b_packed, d_b).expect("h2d b");
            gpu.copy_h2d(&b_scales_swiz, d_bs).expect("h2d b_scale");
            let out_init_bytes: Vec<u8> =
                init_out_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d(&out_init_bytes, d_out).expect("h2d out");

            let mut d_alpha_opt: Option<crate::gpu::DevicePtr> = None;
            let mut d_beta_opt: Option<crate::gpu::DevicePtr> = None;
            let scalars = if use_device_alpha {
                let d_alpha = gpu.alloc(4).expect("alloc alpha");
                gpu.copy_h2d(&alpha_val.to_le_bytes(), d_alpha)
                    .expect("h2d alpha");
                d_alpha_opt = Some(d_alpha);
                match device_beta_mode.unwrap_or(DeviceBeta::Zero) {
                    DeviceBeta::Zero => BlockScaleScalars::device_alpha(d_alpha.0),
                    DeviceBeta::One => BlockScaleScalars::device_alpha_accumulate(d_alpha.0),
                    DeviceBeta::DevicePtr(_) => {
                        let d_beta = gpu.alloc(4).expect("alloc beta");
                        gpu.copy_h2d(&beta_val.to_le_bytes(), d_beta)
                            .expect("h2d beta");
                        d_beta_opt = Some(d_beta);
                        BlockScaleScalars::device(d_alpha.0, d_beta.0)
                    }
                }
            } else {
                BlockScaleScalars::host(alpha_val, beta_val)
            };

            nvfp4_gemm_block_scaled(
                d_a.0, d_as.0, d_b.0, d_bs.0, d_out.0, m, n, k, ldc, scalars, stream,
            )
            .unwrap_or_else(|e| panic!("NVFP4 case `{label}` failed: {e:#}"));
            gpu.synchronize(stream).expect("sync");

            let mut got_bytes = vec![0u8; out_bytes];
            gpu.copy_d2h(d_out, &mut got_bytes).expect("d2h out");
            let got_bf16: Vec<u16> = got_bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();

            // Verify strided padding columns `n..ldc` were left untouched.
            for row in 0..(m as usize) {
                for col in (n as usize)..(ldc as usize) {
                    assert_eq!(
                        got_bf16[row * (ldc as usize) + col],
                        init_out_bf16[row * (ldc as usize) + col],
                        "NVFP4 `{label}` clobbered ldc padding at ({row}, {col})"
                    );
                }
            }

            let (max_abs_err, cos_sim, max_ulp) =
                compare_bf16_vs_ref(&got_bf16, &ref_f32, m, n, ldc);
            println!(
                "NVFP4 [{label}] {m}x{n}x{k} ldc={ldc}: max_abs_err={max_abs_err:.6e} max_bf16_ulp={max_ulp} cos_sim={cos_sim:.9}"
            );
            assert!(
                cos_sim > 0.999_99,
                "NVFP4 `{label}` cosine similarity too low: {cos_sim}"
            );
            assert!(
                max_ulp <= 1,
                "NVFP4 `{label}` BF16 ULP error too large: {max_ulp} (max_abs_err={max_abs_err})"
            );

            if let Some(p) = d_alpha_opt {
                gpu.free(p).expect("free alpha");
            }
            if let Some(p) = d_beta_opt {
                gpu.free(p).expect("free beta");
            }
            gpu.free(d_a_base).expect("free a");
            gpu.free(d_as_base).expect("free a_scale");
            gpu.free(d_b_base).expect("free b");
            gpu.free(d_bs_base).expect("free b_scale");
            gpu.free(d_out_base).expect("free out");
        };

        run_case(
            "aligned_host",
            128,
            128,
            128,
            128,
            0.75,
            0.0,
            false,
            None,
            false,
        );
        // Second call on identical cached shape with distinct alpha & inputs verifies plan-cache
        // reuse updates `A_SCALE_POINTER` / `B_SCALE_POINTER` cleanly.
        run_case(
            "aligned_host_cache_hit",
            128,
            128,
            128,
            128,
            1.5,
            0.0,
            false,
            None,
            false,
        );
        run_case(
            "unaligned_strided",
            137,
            144,
            96,
            160,
            1.25,
            0.0,
            false,
            None,
            false,
        );
        run_case(
            "decode_m1_n8_k32",
            1,
            8,
            32,
            8,
            1.0,
            0.0,
            false,
            None,
            false,
        );
        run_case("decode_m1_k32", 1, 32, 32, 32, 1.0, 0.0, false, None, false);
        run_case(
            "small_m16_n24_k64",
            16,
            24,
            64,
            24,
            0.5,
            0.0,
            false,
            None,
            false,
        );
        run_case(
            "multiseq_m24_strided",
            24,
            128,
            128,
            192,
            1.125,
            0.0,
            false,
            None,
            false,
        );
        run_case(
            "multiseq_m24_device_alpha",
            24,
            256,
            128,
            256,
            0.875,
            0.0,
            true,
            Some(DeviceBeta::Zero),
            false,
        );
        run_case(
            "host_beta_residual",
            64,
            64,
            64,
            64,
            0.5,
            0.25,
            false,
            None,
            false,
        );
        run_case(
            "dev_alpha_beta0",
            128,
            128,
            128,
            128,
            0.3125,
            0.0,
            true,
            Some(DeviceBeta::Zero),
            false,
        );
        run_case(
            "dev_alpha_beta1",
            128,
            128,
            128,
            128,
            0.3125,
            1.0,
            true,
            Some(DeviceBeta::One),
            false,
        );
        run_case(
            "dev_alpha_dev_beta",
            64,
            96,
            64,
            96,
            0.375,
            0.5,
            true,
            Some(DeviceBeta::DevicePtr(1)),
            false,
        );
        run_case(
            "align16_subslice",
            64,
            64,
            64,
            64,
            1.0,
            0.0,
            false,
            None,
            true,
        );
        run_case(
            "autotune_wide_beta1",
            512,
            512,
            128,
            512,
            0.25,
            1.0,
            false,
            None,
            false,
        );
    }

    #[test]
    #[ignore = "requires CUDA 13.0 Blackwell GPU (SM100+/SM120+/SM121)"]
    fn real_gpu_mxfp8_block_scaled_gemm_matches_cpu_fp32_reference() {
        use crate::gpu::GpuBackend;

        let gpu =
            crate::cuda_backend::AtlasCudaBackend::new(0, &[]).expect("init AtlasCudaBackend");
        let stream = 0u64;

        // Valid finite FP8 E4M3 data values (both signs, normal & subnormal).
        const E4M3_POOL: [u8; 16] = [
            0x00, 0x28, 0x30, 0x34, 0x38, 0x3c, 0x40, 0x44, 0x80, 0xa8, 0xb0, 0xb4, 0xb8, 0xbc,
            0xc0, 0xc4,
        ];
        // Distinct UE8M0 power-of-two exponents spanning 2^-3 .. 2^+3.
        const UE8M0_POOL: [u8; 7] = [124, 125, 126, 127, 128, 129, 130];

        let run_case = |label: &str,
                        m: u32,
                        n: u32,
                        k: u32,
                        ldc: u32,
                        alpha_val: f32,
                        beta_val: f32,
                        ptr_offset_16: bool| {
            let k_blocks = k / MXFP8_BLOCK_SIZE;
            let mut a_bytes = vec![0u8; (m as usize) * (k as usize)];
            let mut b_bytes = vec![0u8; (n as usize) * (k as usize)];
            let mut a_scales_log = vec![0u8; (m as usize) * (k_blocks as usize)];
            let mut b_scales_log = vec![0u8; (n as usize) * (k_blocks as usize)];

            let label_mix = label.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(131).wrapping_add(u64::from(b))
            });
            let mut state: u64 = 0xd1b5_4a32_d192_ed03
                ^ label_mix
                ^ (u64::from(m) << 32)
                ^ (u64::from(n) << 16)
                ^ u64::from(k);
            let mut next_u32 = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 32) as u32
            };
            for byte in &mut a_bytes {
                *byte = E4M3_POOL[(next_u32() as usize) % E4M3_POOL.len()];
            }
            for byte in &mut b_bytes {
                *byte = E4M3_POOL[(next_u32() as usize) % E4M3_POOL.len()];
            }
            for (idx, s) in a_scales_log.iter_mut().enumerate() {
                *s = UE8M0_POOL[((next_u32() as usize) ^ idx) % UE8M0_POOL.len()];
            }
            for (idx, s) in b_scales_log.iter_mut().enumerate() {
                *s = UE8M0_POOL[((next_u32() as usize) ^ (idx * 5)) % UE8M0_POOL.len()];
            }

            let a_scales_swiz = swizzle_block_scales(&a_scales_log, m, k_blocks);
            let b_scales_swiz = swizzle_block_scales(&b_scales_log, n, k_blocks);
            assert_eq!(
                a_scales_swiz.len(),
                block_scale_padded_bytes(m, k, MXFP8_BLOCK_SIZE)
            );
            assert_eq!(
                b_scales_swiz.len(),
                block_scale_padded_bytes(n, k, MXFP8_BLOCK_SIZE)
            );

            let mut init_out_bf16 = vec![0u16; (m as usize) * (ldc as usize)];
            for (idx, slot) in init_out_bf16.iter_mut().enumerate() {
                let val = (((idx % 19) as f32) - 9.0) * 0.25;
                *slot = half::bf16::from_f32(val).to_bits();
            }

            let mut ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
            for row in 0..(m as usize) {
                let mut a_row = vec![0.0f32; k as usize];
                for j in 0..(k as usize) {
                    let sc = decode_ue8m0(a_scales_log[row * (k_blocks as usize) + (j / 32)]);
                    a_row[j] = decode_fp8_e4m3(a_bytes[row * (k as usize) + j]) * sc;
                }
                for col in 0..(n as usize) {
                    let mut acc = 0.0f32;
                    for j in 0..(k as usize) {
                        let sc = decode_ue8m0(b_scales_log[col * (k_blocks as usize) + (j / 32)]);
                        acc += a_row[j] * (decode_fp8_e4m3(b_bytes[col * (k as usize) + j]) * sc);
                    }
                    let c_init =
                        half::bf16::from_bits(init_out_bf16[row * (ldc as usize) + col]).to_f32();
                    ref_f32[row * (n as usize) + col] = alpha_val * acc + beta_val * c_init;
                }
            }

            let pad = if ptr_offset_16 { 16usize } else { 0usize };
            let d_a_base = gpu.alloc(a_bytes.len() + pad).expect("alloc a");
            let d_as_base = gpu.alloc(a_scales_swiz.len() + pad).expect("alloc a_scale");
            let d_b_base = gpu.alloc(b_bytes.len() + pad).expect("alloc b");
            let d_bs_base = gpu.alloc(b_scales_swiz.len() + pad).expect("alloc b_scale");
            let out_bytes = init_out_bf16.len() * 2;
            let d_out_base = gpu.alloc(out_bytes + pad).expect("alloc out");

            let d_a = d_a_base.offset(pad);
            let d_as = d_as_base.offset(pad);
            let d_b = d_b_base.offset(pad);
            let d_bs = d_bs_base.offset(pad);
            let d_out = d_out_base.offset(pad);

            gpu.copy_h2d(&a_bytes, d_a).expect("h2d a");
            gpu.copy_h2d(&a_scales_swiz, d_as).expect("h2d a_scale");
            gpu.copy_h2d(&b_bytes, d_b).expect("h2d b");
            gpu.copy_h2d(&b_scales_swiz, d_bs).expect("h2d b_scale");
            let out_init_bytes: Vec<u8> =
                init_out_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d(&out_init_bytes, d_out).expect("h2d out");

            mxfp8_gemm_block_scaled(
                d_a.0, d_as.0, d_b.0, d_bs.0, d_out.0, m, n, k, ldc, alpha_val, beta_val, stream,
            )
            .unwrap_or_else(|e| panic!("MXFP8 case `{label}` failed: {e:#}"));
            gpu.synchronize(stream).expect("sync");

            let mut got_bytes = vec![0u8; out_bytes];
            gpu.copy_d2h(d_out, &mut got_bytes).expect("d2h out");
            let got_bf16: Vec<u16> = got_bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();

            for row in 0..(m as usize) {
                for col in (n as usize)..(ldc as usize) {
                    assert_eq!(
                        got_bf16[row * (ldc as usize) + col],
                        init_out_bf16[row * (ldc as usize) + col],
                        "MXFP8 `{label}` clobbered ldc padding at ({row}, {col})"
                    );
                }
            }

            let (max_abs_err, cos_sim, max_ulp) =
                compare_bf16_vs_ref(&got_bf16, &ref_f32, m, n, ldc);
            println!(
                "MXFP8 [{label}] {m}x{n}x{k} ldc={ldc}: max_abs_err={max_abs_err:.6e} max_bf16_ulp={max_ulp} cos_sim={cos_sim:.9}"
            );
            assert!(
                cos_sim > 0.999_99,
                "MXFP8 `{label}` cosine similarity too low: {cos_sim}"
            );
            assert!(
                max_ulp <= 1,
                "MXFP8 `{label}` BF16 ULP error too large: {max_ulp} (max_abs_err={max_abs_err})"
            );

            gpu.free(d_a_base).expect("free a");
            gpu.free(d_as_base).expect("free a_scale");
            gpu.free(d_b_base).expect("free b");
            gpu.free(d_bs_base).expect("free b_scale");
            gpu.free(d_out_base).expect("free out");
        };

        run_case("aligned_host", 128, 128, 128, 128, 1.25, 0.0, false);
        run_case(
            "aligned_host_cache_hit",
            128,
            128,
            128,
            128,
            0.625,
            0.0,
            false,
        );
        run_case("unaligned_strided", 137, 144, 96, 160, 0.5, 0.0, false);
        run_case("decode_m1_n8_k32", 1, 8, 32, 8, 1.0, 0.0, false);
        run_case("decode_m1_k32", 1, 32, 32, 32, 1.0, 0.0, false);
        run_case("small_m16_n24_k64", 16, 24, 64, 24, 0.75, 0.0, false);
        run_case("host_beta_residual", 64, 64, 64, 64, 0.5, 1.0, false);
        run_case("host_beta_half", 64, 96, 64, 96, 0.375, 0.5, false);
        run_case("align16_subslice", 64, 64, 64, 64, 1.0, 0.0, true);
        run_case("autotune_wide_beta1", 512, 512, 128, 512, 0.25, 1.0, false);
    }
}
