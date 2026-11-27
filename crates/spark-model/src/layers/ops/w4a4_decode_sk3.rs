// SPDX-License-Identifier: AGPL-3.0-only

//! 48-SM Roofline W4A4 Decode GEMM (`split_k=1..=8`, default `3` on GB10)
//! with deterministic fixed-order reduction and `W=1 == W=8 == W=16` /
//! `B=1 == B=4/8` bit-exact invariance.
//!
//! CUDA entrypoints live in `kernels/gb10/common/w4a4_decode_roofline_sm120.cu`
//! (auto-discovered by `atlas-kernels` under module `"w4a4_decode_roofline_sm120"`):
//! - `w4a4_gemm_m16_sk3_sm120`: native Blackwell `m16n8k64.scale_vec::4X`
//!   NVFP4xNVFP4 decode GEMM (`M_TILE=16, N_TILE=64, K_STEP=64`, 2-stage
//!   register pipeline, `LDG.E.EF.128` weights, `LDG.E.128.CONSTANT` activations,
//!   3-op branchless SWAR E2M1 `-0.0 -> +0.0` canonicalization, exact scalar
//!   fallback on `sm_103a`, zero `atomicAdd`).
//! - `reduce_w4a4_sk3_bf16_sm120`: fixed ascending-order (`s = 0..split_k-1`)
//!   FP32 workspace reduction + `scale2_ab` multiply + explicit inline PTX
//!   `cvt.rn.bf16.f32`.
//!
//! Opt-in gate: `ATLAS_W4A4_DECODE_SK3=1` (strict `ZERO_OR_ONE`, default OFF
//! for 100% bit-identical baseline and zero extra VRAM when unset).

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt;
use std::sync::{Mutex, Once, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;
use crate::weight_map::dual_plane::{DUAL_PLANE_E2M1_LUT, e4m3_byte_to_f32};

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/w4a4_decode_roofline_sm120.cu`.
pub const W4A4_DECODE_SK3_MODULE: &str = "w4a4_decode_roofline_sm120";

/// Environment variable opting into the 48-SM roofline W4A4 decode GEMM.
///
/// Strict `ZERO_OR_ONE` (`"0"` or `"1"`, unset = disabled). Default OFF.
pub const W4A4_DECODE_SK3_ENV: &str = "ATLAS_W4A4_DECODE_SK3";

/// Maximum batch / verify rows `M` covered by one `m16` tile height
/// (`W=1`, `W=8`, `W=16` all execute in a single tile height).
pub const W4A4_DECODE_SK3_MAX_M: u32 = 16;

/// Output columns per CTA (`N_TILE = 64`: 80 tiles at `N=5120`, 272 at `N=17408`).
pub const W4A4_DECODE_SK3_N_TILE: u32 = 64;

/// Reduction step per K-tile (`K_STEP = 64`, 4 UE4M3 block-16 groups).
pub const W4A4_DECODE_SK3_K_STEP: u32 = 64;

/// Maximum supported `split_k` factor.
pub const W4A4_DECODE_SK3_MAX_SPLIT_K: u32 = 8;

/// Default `split_k` on 48-SM GB10 (`sm_count <= 64`):
/// `N=5120 -> 240 CTAs (5.0/SM)`, `N=4096 -> 192 (4.0/SM)`,
/// `N=17408 -> 816 (17.0/SM)`, `N=6144 -> 288 (6.0/SM)`.
pub const W4A4_DECODE_SK3_DEFAULT_GB10: u32 = 3;

/// Default `split_k` on wider discrete SM120/SM103 GPUs (`sm_count > 64`).
pub const W4A4_DECODE_SK3_DEFAULT_WIDE: u32 = 4;

/// SM count ceiling classifying GB10-class GPUs (`<= 64` SMs).
pub const W4A4_DECODE_SK3_GB10_SM_CEILING: u32 = 64;

/// Error returned when parsing [`W4A4_DECODE_SK3_ENV`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum W4a4DecodeSk3ConfigError {
    /// Environment value is not valid UTF-8.
    NonUtf8,
    /// Environment value is not strictly `"0"` or `"1"`.
    InvalidToken(String),
}

impl fmt::Display for W4a4DecodeSk3ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => {
                write!(
                    f,
                    "{W4A4_DECODE_SK3_ENV} must contain valid UTF-8 and be exactly 0 or 1"
                )
            }
            Self::InvalidToken(raw) => write!(
                f,
                "{W4A4_DECODE_SK3_ENV} must be strictly 0 or 1, got {raw:?}"
            ),
        }
    }
}

impl std::error::Error for W4a4DecodeSk3ConfigError {}

/// Opt-in runtime policy parsed from [`W4A4_DECODE_SK3_ENV`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum W4a4DecodeSk3Policy {
    /// Disabled (default): baseline decode/verify FFN path is bit-identical.
    #[default]
    Disabled,
    /// Enabled (`ATLAS_W4A4_DECODE_SK3=1`): opt-in roofline W4A4 decode GEMM.
    Enabled,
}

impl W4a4DecodeSk3Policy {
    /// Strictly parse [`W4A4_DECODE_SK3_ENV`] from the process environment.
    ///
    /// Returns [`Self::Disabled`] when unset or `"0"`, [`Self::Enabled`] when
    /// `"1"`, and fails closed on non-UTF-8 or any other token.
    ///
    /// # Errors
    /// Returns [`W4a4DecodeSk3ConfigError`] on non-UTF-8 or unrecognized tokens.
    pub fn from_env() -> std::result::Result<Self, W4a4DecodeSk3ConfigError> {
        let raw = std::env::var_os(W4A4_DECODE_SK3_ENV);
        Self::parse_os(raw.as_deref())
    }

    /// Parse an optional OS string (used by [`Self::from_env`] and unit tests).
    ///
    /// # Errors
    /// Returns [`W4a4DecodeSk3ConfigError::NonUtf8`] on non-UTF-8 or
    /// [`W4a4DecodeSk3ConfigError::InvalidToken`] on any value other than `"0"`/`"1"`.
    pub fn parse_os(raw: Option<&OsStr>) -> std::result::Result<Self, W4a4DecodeSk3ConfigError> {
        let Some(os_val) = raw else {
            return Ok(Self::Disabled);
        };
        let utf8 = os_val.to_str().ok_or(W4a4DecodeSk3ConfigError::NonUtf8)?;
        Self::parse_optional(Some(utf8))
    }

    /// Parse an optional UTF-8 string slice strictly against `ZERO_OR_ONE`.
    ///
    /// # Errors
    /// Returns [`W4a4DecodeSk3ConfigError::InvalidToken`] on any value other
    /// than `"0"` or `"1"` (including empty strings or booleans like `"true"`).
    pub fn parse_optional(
        raw: Option<&str>,
    ) -> std::result::Result<Self, W4a4DecodeSk3ConfigError> {
        let Some(val) = raw else {
            return Ok(Self::Disabled);
        };
        match val {
            "0" => Ok(Self::Disabled),
            "1" => Ok(Self::Enabled),
            other => Err(W4a4DecodeSk3ConfigError::InvalidToken(other.to_owned())),
        }
    }
}

/// Cached fail-closed runtime check for `ATLAS_W4A4_DECODE_SK3=1`.
///
/// Logs a warning once and stays [`W4a4DecodeSk3Policy::Disabled`] if the env
/// var contains non-UTF-8 bytes or a token outside `{"0", "1"}`.
#[must_use]
pub fn w4a4_decode_sk3_policy_fail_closed() -> W4a4DecodeSk3Policy {
    static POLICY: OnceLock<W4a4DecodeSk3Policy> = OnceLock::new();
    *POLICY.get_or_init(|| match W4a4DecodeSk3Policy::from_env() {
        Ok(policy) => policy,
        Err(err) => {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    env = W4A4_DECODE_SK3_ENV,
                    error = %err,
                    "invalid ATLAS_W4A4_DECODE_SK3 value — failing closed to Disabled"
                );
            });
            W4a4DecodeSk3Policy::Disabled
        }
    })
}

/// Convenience predicate returning `true` only when `ATLAS_W4A4_DECODE_SK3=1`.
#[must_use]
pub fn w4a4_decode_sk3_enabled() -> bool {
    w4a4_decode_sk3_policy_fail_closed() == W4a4DecodeSk3Policy::Enabled
}

/// Select the deterministic default `split_k` factor from hardware SM count
/// and reduction width `k`.
///
/// Deliberately **independent of `M`** so `W=1`, `W=8`, `W=16` and
/// `B=1`, `B=4`, `B=8` always use the identical K-slice boundaries and produce
/// bit-for-bit identical BF16 outputs.
#[must_use]
pub const fn default_w4a4_decode_split_k(sm_count: u32, k: u32) -> u32 {
    let k_tiles = k / W4A4_DECODE_SK3_K_STEP;
    if k_tiles <= 1 {
        return 1;
    }
    let target = if sm_count <= W4A4_DECODE_SK3_GB10_SM_CEILING {
        W4A4_DECODE_SK3_DEFAULT_GB10
    } else {
        W4A4_DECODE_SK3_DEFAULT_WIDE
    };
    if target < k_tiles { target } else { k_tiles }
}

/// Validated launch geometry and deterministic FP32 workspace sizing for one
/// roofline W4A4 decode GEMM (`[M, K] @ [N, K]^T -> [M, N]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4a4DecodeSk3Plan {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub split_k: u32,
    pub grid_n_tiles: u32,
    pub total_ctas: u32,
    pub reduce_grid_x: u32,
    /// Required FP32 workspace bytes (`split_k * m * n * 4` when `split_k > 1`,
    /// `0` when `split_k == 1`).
    pub workspace_bytes: usize,
}

/// Validate dimensions and compute checked launch/workspace geometry.
///
/// # Errors
/// Fails closed if `m` is outside `1..=16`, `n == 0`, `k` is zero or not a
/// multiple of `64`, `split_k` is outside `1..=8`, `split_k > k / 64`, or
/// workspace byte sizing overflows `usize`.
pub fn plan_w4a4_decode_sk3(m: u32, n: u32, k: u32, split_k: u32) -> Result<W4a4DecodeSk3Plan> {
    ensure!(
        (1..=W4A4_DECODE_SK3_MAX_M).contains(&m),
        "w4a4_decode_sk3: M={m} must lie in 1..={W4A4_DECODE_SK3_MAX_M}"
    );
    ensure!(n > 0, "w4a4_decode_sk3: N must be > 0");
    ensure!(
        k >= W4A4_DECODE_SK3_K_STEP && k.is_multiple_of(W4A4_DECODE_SK3_K_STEP),
        "w4a4_decode_sk3: K={k} must be >= {W4A4_DECODE_SK3_K_STEP} and divisible by {W4A4_DECODE_SK3_K_STEP}"
    );
    let num_k_tiles = k / W4A4_DECODE_SK3_K_STEP;
    ensure!(
        (1..=W4A4_DECODE_SK3_MAX_SPLIT_K).contains(&split_k),
        "w4a4_decode_sk3: split_k={split_k} must lie in 1..={W4A4_DECODE_SK3_MAX_SPLIT_K}"
    );
    ensure!(
        split_k <= num_k_tiles,
        "w4a4_decode_sk3: split_k={split_k} exceeds K-tile count {num_k_tiles} (K={k})"
    );

    let grid_n_tiles = div_ceil(n, W4A4_DECODE_SK3_N_TILE);
    let total_ctas = grid_n_tiles
        .checked_mul(split_k)
        .context("w4a4_decode_sk3: CTA grid count overflow")?;
    let total_outputs = m
        .checked_mul(n)
        .context("w4a4_decode_sk3: M * N output element count overflow")?;
    let reduce_grid_x = if split_k > 1 {
        div_ceil(total_outputs, 256)
    } else {
        0
    };
    let workspace_bytes = if split_k > 1 {
        (split_k as usize)
            .checked_mul(m as usize)
            .and_then(|v| v.checked_mul(n as usize))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .context("w4a4_decode_sk3: FP32 split-K workspace byte count overflow")?
    } else {
        0
    };

    Ok(W4a4DecodeSk3Plan {
        m,
        n,
        k,
        split_k,
        grid_n_tiles,
        total_ctas,
        reduce_grid_x,
        workspace_bytes,
    })
}

/// Resolved CUDA kernel handles for `w4a4_decode_roofline_sm120.cu`.
#[derive(Clone, Copy, Debug)]
pub struct W4a4DecodeSk3Kernels {
    pub gemm: KernelHandle,
    pub reduce: KernelHandle,
}

impl Default for W4a4DecodeSk3Kernels {
    fn default() -> Self {
        Self {
            gemm: KernelHandle(0),
            reduce: KernelHandle(0),
        }
    }
}

impl PartialEq for W4a4DecodeSk3Kernels {
    fn eq(&self, other: &Self) -> bool {
        self.gemm.0 == other.gemm.0 && self.reduce.0 == other.reduce.0
    }
}

impl Eq for W4a4DecodeSk3Kernels {}

impl W4a4DecodeSk3Kernels {
    /// Probe both entrypoints from module `"w4a4_decode_roofline_sm120"`.
    #[must_use]
    pub fn probe(gpu: &dyn GpuBackend) -> Self {
        Self {
            gemm: crate::layers::try_kernel(gpu, W4A4_DECODE_SK3_MODULE, "w4a4_gemm_m16_sk3_sm120"),
            reduce: crate::layers::try_kernel(
                gpu,
                W4A4_DECODE_SK3_MODULE,
                "reduce_w4a4_sk3_bf16_sm120",
            ),
        }
    }

    /// True when both GEMM and deterministic reduction kernels are loaded.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        self.gemm.0 != 0 && self.reduce.0 != 0
    }
}

/// Process-wide per-GPU cache of [`W4a4DecodeSk3Kernels`] handles so opt-in
/// decode dispatch requires zero edits to `DenseFfnLayer` structs and avoids
/// repeated symbol lookups.
#[must_use]
pub fn cached_w4a4_decode_sk3_kernels(gpu: &dyn GpuBackend) -> W4a4DecodeSk3Kernels {
    static CACHE: OnceLock<Mutex<HashMap<usize, W4a4DecodeSk3Kernels>>> = OnceLock::new();
    let key = std::ptr::from_ref(gpu) as *const () as usize;
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut guard) = map.lock() else {
        return W4a4DecodeSk3Kernels::probe(gpu);
    };
    *guard
        .entry(key)
        .or_insert_with(|| W4a4DecodeSk3Kernels::probe(gpu))
}

/// Whether allocation of a lazy fallback workspace is permitted on this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4a4WorkspaceAllocPolicy {
    /// Eager execution: may lazily allocate or grow the per-stream workspace.
    AllowAlloc,
    /// CUDA graph capture: never call `gpu.alloc`; fail closed if undersized.
    CaptureNoAlloc,
}

#[derive(Clone, Copy, Debug)]
struct CachedSplitKWorkspace {
    ptr: DevicePtr,
    bytes: usize,
}

/// Obtain a per-`(gpu, stream)` FP32 split-K workspace of at least `req_bytes`.
///
/// Zero VRAM is allocated when `req_bytes == 0` or when `ATLAS_W4A4_DECODE_SK3`
/// is disabled. Under [`W4a4WorkspaceAllocPolicy::CaptureNoAlloc`], returns
/// `Ok(None)` if no sufficiently large workspace was pre-allocated before
/// capture (never allocates mid-capture).
///
/// # Errors
/// Propagates GPU allocation/synchronization errors during eager growth.
pub fn ensure_w4a4_decode_sk3_workspace(
    gpu: &dyn GpuBackend,
    stream: u64,
    req_bytes: usize,
    policy: W4a4WorkspaceAllocPolicy,
) -> Result<Option<(DevicePtr, usize)>> {
    if req_bytes == 0 {
        return Ok(Some((DevicePtr::NULL, 0)));
    }
    static CACHE: OnceLock<Mutex<HashMap<(usize, u64), CachedSplitKWorkspace>>> = OnceLock::new();
    let key = (std::ptr::from_ref(gpu) as *const () as usize, stream);
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .map_err(|_| anyhow::anyhow!("w4a4_decode_sk3 workspace cache lock poisoned"))?;
    if let Some(existing) = guard.get(&key)
        && existing.bytes >= req_bytes
    {
        return Ok(Some((existing.ptr, existing.bytes)));
    }
    if policy == W4a4WorkspaceAllocPolicy::CaptureNoAlloc {
        return Ok(None);
    }
    if let Some(prev) = guard.get(&key).copied() {
        gpu.synchronize(stream)?;
        // Remove and free only after synchronization succeeds so a sync error
        // keeps `prev` tracked in `CACHE` instead of leaking device memory.
        guard.remove(&key);
        let _ = gpu.free(prev.ptr);
    }
    let ptr = gpu.alloc(req_bytes)?;
    guard.insert(
        key,
        CachedSplitKWorkspace {
            ptr,
            bytes: req_bytes,
        },
    );
    Ok(Some((ptr, req_bytes)))
}

/// Pre-launch operand, handle, and alignment preflight for a 3-projection
/// NVFP4 FFN (`gate_proj`, `up_proj`, `down_proj`).
///
/// Called before allocating scratch/workspace or issuing the first GPU kernel
/// so any missing handle, null scale pointer, or misaligned buffer falls back
/// cleanly with zero GPU side effects.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn w4a4_decode_sk3_ffn_preflight_ok(
    gate_proj: &QuantizedWeight,
    up_proj: &QuantizedWeight,
    down_proj: &QuantizedWeight,
    act_mul: KernelHandle,
    nvfp4_quantize_k: KernelHandle,
    input: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    output: DevicePtr,
) -> bool {
    let weight_ok = |w: &QuantizedWeight| {
        !w.is_null()
            && !w.weight_scale.is_null()
            && w.weight.0.is_multiple_of(16)
            && w.weight_scale.0.is_multiple_of(4)
            && w.weight_scale_2.is_finite()
            && w.weight_scale_2 > 0.0
    };
    let bf16_buf_ok = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(2);
    nvfp4_quantize_k.0 != 0
        && act_mul.0 != 0
        && weight_ok(gate_proj)
        && weight_ok(up_proj)
        && weight_ok(down_proj)
        && bf16_buf_ok(input)
        && bf16_buf_ok(gate_out)
        && bf16_buf_ok(up_out)
        && bf16_buf_ok(output)
}

/// Pre-launch pointer and alignment check for NVFP4 activation scratch
/// (`a_packed` 16B-aligned, `a_scale` 4B-aligned).
#[must_use]
pub fn w4a4_decode_sk3_activation_scratch_ok(a_packed: DevicePtr, a_scale: DevicePtr) -> bool {
    !a_packed.is_null()
        && a_packed.0.is_multiple_of(16)
        && !a_scale.is_null()
        && a_scale.0.is_multiple_of(4)
}

fn validate_gemm_operands(
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    scale2_ab: f32,
) -> Result<()> {
    ensure!(
        a_packed != DevicePtr::NULL && a_scale != DevicePtr::NULL,
        "w4a4_decode_sk3: A_packed or A_scale pointer is NULL"
    );
    ensure!(
        !weight.is_null() && weight.weight_scale != DevicePtr::NULL,
        "w4a4_decode_sk3: B_packed or B_scale pointer is NULL"
    );
    ensure!(
        a_packed.0.is_multiple_of(16) && weight.weight.0.is_multiple_of(16),
        "w4a4_decode_sk3: A_packed and B_packed must be 16B-aligned"
    );
    ensure!(
        a_scale.0.is_multiple_of(4) && weight.weight_scale.0.is_multiple_of(4),
        "w4a4_decode_sk3: A_scale and B_scale must be 4B-aligned"
    );
    ensure!(
        scale2_ab.is_finite() && scale2_ab > 0.0,
        "w4a4_decode_sk3: scale2_ab={scale2_ab} must be finite and positive"
    );
    Ok(())
}

/// Stage 1: Launch `w4a4_gemm_m16_sk3_sm120`.
///
/// Grid: `(ceil(N / 64), 1, split_k)`  Block: `(128, 1, 1)`
///
/// When `plan.split_k == 1`, writes BF16 directly to `out_bf16` (`workspace_f32`
/// is ignored). When `plan.split_k > 1`, writes FP32 partial slices to
/// `workspace_f32` (`[split_k, M, N]`).
///
/// # Errors
/// Fails closed before launch on invalid handle, pointers, alignment,
/// non-finite/non-positive `scale2_ab`, or undersized workspace.
#[allow(clippy::too_many_arguments)]
pub fn w4a4_gemm_m16_sk3_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    plan: &W4a4DecodeSk3Plan,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    workspace_f32: DevicePtr,
    workspace_bytes: usize,
    out_bf16: DevicePtr,
    scale2_ab: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "w4a4_gemm_m16_sk3_sm120 requires a non-zero kernel handle"
    );
    let checked = plan_w4a4_decode_sk3(plan.m, plan.n, plan.k, plan.split_k)?;
    validate_gemm_operands(a_packed, a_scale, weight, scale2_ab)?;

    if checked.split_k == 1 {
        ensure!(
            out_bf16 != DevicePtr::NULL && out_bf16.0.is_multiple_of(2),
            "w4a4_gemm_m16_sk3_sm120 (split_k=1): out_bf16 must be non-null and 2B-aligned"
        );
    } else {
        ensure!(
            workspace_f32 != DevicePtr::NULL && workspace_f32.0.is_multiple_of(4),
            "w4a4_gemm_m16_sk3_sm120 (split_k>1): workspace_f32 must be non-null and 4B-aligned"
        );
        ensure!(
            workspace_bytes >= checked.workspace_bytes,
            "w4a4_gemm_m16_sk3_sm120: workspace ({workspace_bytes} B) smaller than required {} B",
            checked.workspace_bytes
        );
    }

    KernelLaunch::new(gpu, kernel)
        .grid([checked.grid_n_tiles, 1, checked.split_k])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(workspace_f32)
        .arg_ptr(out_bf16)
        .arg_f32(scale2_ab)
        .arg_u32(checked.m)
        .arg_u32(checked.n)
        .arg_u32(checked.k)
        .arg_u32(checked.split_k)
        .launch(stream)
}

/// Stage 2: Launch `reduce_w4a4_sk3_bf16_sm120`.
///
/// Grid: `(ceil(M * N / 256), 1, 1)`  Block: `(256, 1, 1)`
///
/// Reduces `workspace_f32` (`[split_k, M, N]`) in fixed ascending order
/// `s = 0..split_k-1`, scales by `scale2_ab`, and rounds via inline PTX
/// `cvt.rn.bf16.f32`.
///
/// # Errors
/// Fails closed before launch on invalid handle, pointers, alignment,
/// non-finite/non-positive `scale2_ab`, or undersized workspace.
#[allow(clippy::too_many_arguments)]
pub fn reduce_w4a4_sk3_bf16_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    plan: &W4a4DecodeSk3Plan,
    workspace_f32: DevicePtr,
    workspace_bytes: usize,
    out_bf16: DevicePtr,
    scale2_ab: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "reduce_w4a4_sk3_bf16_sm120 requires a non-zero kernel handle"
    );
    let checked = plan_w4a4_decode_sk3(plan.m, plan.n, plan.k, plan.split_k)?;
    ensure!(
        checked.split_k > 1,
        "reduce_w4a4_sk3_bf16_sm120 requires split_k > 1"
    );
    ensure!(
        workspace_f32 != DevicePtr::NULL && workspace_f32.0.is_multiple_of(4),
        "reduce_w4a4_sk3_bf16_sm120: workspace_f32 must be non-null and 4B-aligned"
    );
    ensure!(
        workspace_bytes >= checked.workspace_bytes,
        "reduce_w4a4_sk3_bf16_sm120: workspace ({workspace_bytes} B) smaller than required {} B",
        checked.workspace_bytes
    );
    ensure!(
        out_bf16 != DevicePtr::NULL && out_bf16.0.is_multiple_of(2),
        "reduce_w4a4_sk3_bf16_sm120: out_bf16 must be non-null and 2B-aligned"
    );
    ensure!(
        scale2_ab.is_finite() && scale2_ab > 0.0,
        "reduce_w4a4_sk3_bf16_sm120: scale2_ab={scale2_ab} must be finite and positive"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([checked.reduce_grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(workspace_f32)
        .arg_ptr(out_bf16)
        .arg_f32(scale2_ab)
        .arg_u32(checked.m)
        .arg_u32(checked.n)
        .arg_u32(checked.split_k)
        .launch(stream)
}

/// Combined 1-launch (`split_k == 1`) or 2-launch (`split_k > 1`) roofline
/// W4A4 decode GEMM dispatch.
///
/// Validates ALL stage-1 and stage-2 preconditions up front so any invalid
/// input fails closed with zero GPU launches.
///
/// # Errors
/// Fails closed before issuing any launch on invalid handles, dimensions,
/// pointers, alignment, non-finite/non-positive `scale2_ab`, or undersized
/// workspace.
#[allow(clippy::too_many_arguments)]
pub fn w4a4_decode_roofline_gemm(
    gpu: &dyn GpuBackend,
    kernels: W4a4DecodeSk3Kernels,
    plan: &W4a4DecodeSk3Plan,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    workspace_f32: DevicePtr,
    workspace_bytes: usize,
    out_bf16: DevicePtr,
    scale2_ab: f32,
    stream: u64,
) -> Result<()> {
    let checked = plan_w4a4_decode_sk3(plan.m, plan.n, plan.k, plan.split_k)?;
    ensure!(
        kernels.gemm.0 != 0,
        "w4a4_decode_roofline_gemm requires a non-zero gemm kernel handle"
    );
    if checked.split_k > 1 {
        ensure!(
            kernels.reduce.0 != 0,
            "w4a4_decode_roofline_gemm (split_k>1) requires a non-zero reduce kernel handle"
        );
    }
    ensure!(
        out_bf16 != DevicePtr::NULL && out_bf16.0.is_multiple_of(2),
        "w4a4_decode_roofline_gemm: out_bf16 must be non-null and 2B-aligned"
    );
    validate_gemm_operands(a_packed, a_scale, weight, scale2_ab)?;

    w4a4_gemm_m16_sk3_sm120(
        gpu,
        kernels.gemm,
        &checked,
        a_packed,
        a_scale,
        weight,
        workspace_f32,
        workspace_bytes,
        out_bf16,
        scale2_ab,
        stream,
    )?;
    if checked.split_k > 1 {
        reduce_w4a4_sk3_bf16_sm120(
            gpu,
            kernels.reduce,
            &checked,
            workspace_f32,
            workspace_bytes,
            out_bf16,
            scale2_ab,
            stream,
        )?;
    }
    Ok(())
}

// ── Bit-Exact Host Reference Mirror (matches w4a4_decode_roofline_sm120.cu) ─

/// 3-op branchless SWAR E2M1 `-0.0` (`0x8`) -> `+0.0` (`0x0`) canonicalization
/// across all 8 packed 4-bit nibbles, mirroring `w4sk_canon_e2m1_u32`.
#[inline]
#[must_use]
pub const fn canon_e2m1_u32_cpu(w: u32) -> u32 {
    let m = w & 0x7777_7777;
    w & (m.wrapping_add(0x7777_7777) | 0x7777_7777)
}

/// Exact IEEE-754 FP32 -> BF16 round-to-nearest-even mirroring PTX
/// `cvt.rn.bf16.f32`.
#[inline]
#[must_use]
pub const fn f32_to_bf16_rn_cpu(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = bits & 0x7F80_0000;
    let mant = bits & 0x007F_FFFF;
    if exp == 0x7F80_0000 {
        if mant != 0 {
            // Canonical quiet NaN preserving sign bit.
            return ((bits >> 16) as u16) | 0x0040;
        }
        return (bits >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32.wrapping_add(lsb);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[inline]
#[must_use]
fn canon_e2m1_byte_cpu(b: u8) -> u8 {
    (canon_e2m1_u32_cpu(u32::from(b)) & 0xFF) as u8
}

/// Exact CPU mirror of stage 1 (`w4a4_gemm_m16_sk3_sm120`), producing the
/// `[split_k, M, N]` FP32 split-K workspace with identical K-tile boundaries
/// and 3-op SWAR E2M1 `-0.0` canonicalization.
///
/// # Errors
/// Fails closed if dimensions are invalid or slice lengths do not match.
#[allow(clippy::too_many_arguments)]
pub fn w4a4_gemm_m16_sk3_cpu(
    a_packed: &[u8],
    a_scale: &[u8],
    b_packed: &[u8],
    b_scale: &[u8],
    m: u32,
    n: u32,
    k: u32,
    split_k: u32,
) -> Result<Vec<f32>> {
    let plan = plan_w4a4_decode_sk3(m, n, k, split_k)?;
    let m_us = plan.m as usize;
    let n_us = plan.n as usize;
    let half_k = (plan.k / 2) as usize;
    let num_groups = (plan.k / 16) as usize;

    ensure!(
        a_packed.len() == m_us * half_k,
        "w4a4_gemm_m16_sk3_cpu: a_packed len {} != {}",
        a_packed.len(),
        m_us * half_k
    );
    ensure!(
        a_scale.len() == m_us * num_groups,
        "w4a4_gemm_m16_sk3_cpu: a_scale len {} != {}",
        a_scale.len(),
        m_us * num_groups
    );
    ensure!(
        b_packed.len() == n_us * half_k,
        "w4a4_gemm_m16_sk3_cpu: b_packed len {} != {}",
        b_packed.len(),
        n_us * half_k
    );
    ensure!(
        b_scale.len() == n_us * num_groups,
        "w4a4_gemm_m16_sk3_cpu: b_scale len {} != {}",
        b_scale.len(),
        n_us * num_groups
    );

    let num_k_tiles = plan.k / W4A4_DECODE_SK3_K_STEP;
    let sk_us = plan.split_k as usize;
    let mut workspace = vec![0.0_f32; sk_us * m_us * n_us];

    for sk in 0..plan.split_k {
        let tile_begin = (sk * num_k_tiles) / plan.split_k;
        let tile_end = ((sk + 1) * num_k_tiles) / plan.split_k;
        let slice_base = (sk as usize) * m_us * n_us;

        for r in 0..m_us {
            for c in 0..n_us {
                let mut acc = 0.0_f32;
                for tile in tile_begin..tile_end {
                    let kb = (tile * W4A4_DECODE_SK3_K_STEP) as usize;
                    let g_base = kb / 16;
                    let mut tile_sum = 0.0_f32;
                    for g in 0..4 {
                        let sa = e4m3_byte_to_f32(a_scale[r * num_groups + g_base + g] & 0x7F);
                        let sb = e4m3_byte_to_f32(b_scale[c * num_groups + g_base + g] & 0x7F);
                        let mut g_dot = 0.0_f32;
                        let byte_base = (kb + g * 16) / 2;
                        for bi in 0..8 {
                            let ab = canon_e2m1_byte_cpu(a_packed[r * half_k + byte_base + bi]);
                            let bb = canon_e2m1_byte_cpu(b_packed[c * half_k + byte_base + bi]);
                            let a_lo = DUAL_PLANE_E2M1_LUT[(ab & 0x0F) as usize];
                            let b_lo = DUAL_PLANE_E2M1_LUT[(bb & 0x0F) as usize];
                            let a_hi = DUAL_PLANE_E2M1_LUT[((ab >> 4) & 0x0F) as usize];
                            let b_hi = DUAL_PLANE_E2M1_LUT[((bb >> 4) & 0x0F) as usize];
                            g_dot += a_lo * b_lo;
                            g_dot += a_hi * b_hi;
                        }
                        tile_sum += g_dot * (sa * sb);
                    }
                    acc += tile_sum;
                }
                workspace[slice_base + r * n_us + c] = acc;
            }
        }
    }
    Ok(workspace)
}

/// Exact CPU mirror of stage 2 (`reduce_w4a4_sk3_bf16_sm120`): fixed ascending
/// `s = 0..split_k-1` FP32 sum -> `scale2_ab` multiply -> BF16 round-to-even.
///
/// # Errors
/// Fails closed if `scale2_ab` is non-finite/non-positive or workspace size
/// mismatches `split_k * m * n`.
pub fn reduce_w4a4_sk3_bf16_cpu(
    workspace_f32: &[f32],
    scale2_ab: f32,
    m: u32,
    n: u32,
    split_k: u32,
) -> Result<Vec<u16>> {
    ensure!(
        (1..=W4A4_DECODE_SK3_MAX_M).contains(&m)
            && n > 0
            && (1..=W4A4_DECODE_SK3_MAX_SPLIT_K).contains(&split_k),
        "reduce_w4a4_sk3_bf16_cpu: invalid m={m}, n={n}, split_k={split_k}"
    );
    ensure!(
        scale2_ab.is_finite() && scale2_ab > 0.0,
        "reduce_w4a4_sk3_bf16_cpu: scale2_ab={scale2_ab} must be finite and positive"
    );
    let total = (m as usize) * (n as usize);
    let expected = (split_k as usize) * total;
    ensure!(
        workspace_f32.len() == expected,
        "reduce_w4a4_sk3_bf16_cpu: workspace len {} != {expected}",
        workspace_f32.len()
    );

    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut sum = 0.0_f32;
        for s in 0..(split_k as usize) {
            sum += workspace_f32[s * total + idx];
        }
        out.push(f32_to_bf16_rn_cpu(sum * scale2_ab));
    }
    Ok(out)
}

/// Combined host mirror for `w4a4_decode_roofline_gemm` returning raw BF16
/// `u16` bits (`[M, N]`).
///
/// # Errors
/// Propagates validation errors from [`w4a4_gemm_m16_sk3_cpu`] and
/// [`reduce_w4a4_sk3_bf16_cpu`].
#[allow(clippy::too_many_arguments)]
pub fn w4a4_decode_roofline_gemm_cpu(
    a_packed: &[u8],
    a_scale: &[u8],
    b_packed: &[u8],
    b_scale: &[u8],
    scale2_ab: f32,
    m: u32,
    n: u32,
    k: u32,
    split_k: u32,
) -> Result<Vec<u16>> {
    let ws = w4a4_gemm_m16_sk3_cpu(a_packed, a_scale, b_packed, b_scale, m, n, k, split_k)?;
    reduce_w4a4_sk3_bf16_cpu(&ws, scale2_ab, m, n, split_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn make_valid_weight() -> QuantizedWeight {
        QuantizedWeight {
            weight: DevicePtr(0x2000),
            weight_scale: DevicePtr(0x4000),
            weight_scale_2: 0.25,
            input_scale: DevicePtr::NULL,
        }
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn synthetic_nvfp4_matrix(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let half_k = k / 2;
        let num_groups = k / 16;
        let mut packed = vec![0u8; rows * half_k];
        let mut scales = vec![0u8; rows * num_groups];
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u32 = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        for (i, byte) in packed.iter_mut().enumerate() {
            let v = next_u32();
            // Inject deliberate `-0.0` (0x8) nibbles every 7th byte to exercise SWAR canonicalization.
            if i % 7 == 0 {
                *byte = 0x80 | ((v & 0x0F) as u8);
            } else if i % 11 == 0 {
                *byte = 0x08 | (((v & 0x0F) as u8) << 4);
            } else {
                *byte = (v & 0xFF) as u8;
            }
        }
        // Valid positive UE4M3 scale codes (avoid NaN 0x7F and zero-only rows).
        const SCALE_CODES: [u8; 8] = [0x28, 0x30, 0x34, 0x38, 0x3C, 0x40, 0x44, 0x48];
        for sc in &mut scales {
            let idx = (next_u32() as usize) % SCALE_CODES.len();
            *sc = SCALE_CODES[idx];
        }
        (packed, scales)
    }

    #[test]
    fn swar_e2m1_neg_zero_canonicalization_is_exact_across_all_nibbles_and_carry_boundaries() {
        for nib in 0u32..16 {
            let expected = if nib == 0x8 { 0x0 } else { nib };
            // Test every nibble position 0..8 in a 32-bit word with adversarial neighbors.
            for pos in 0..8 {
                let fill_7 = 0x7777_7777u32 & !(0xFu32 << (pos * 4));
                let fill_f = !(0xFu32 << (pos * 4));
                let fill_8 = 0x8888_8888u32 & !(0xFu32 << (pos * 4));
                for (base, expected_other) in [
                    (fill_7, fill_7),
                    (fill_f, fill_f),
                    (fill_8, 0x0000_0000u32),
                    (0u32, 0u32),
                ] {
                    let word = base | (nib << (pos * 4));
                    let got = canon_e2m1_u32_cpu(word);
                    let want = expected_other | (expected << (pos * 4));
                    assert_eq!(
                        got, want,
                        "SWAR mismatch for nib={nib:#x} pos={pos} word={word:#010x}"
                    );
                }
            }
        }
        assert_eq!(canon_e2m1_u32_cpu(0x8888_8888), 0x0000_0000);
        assert_eq!(canon_e2m1_u32_cpu(0x7878_8787), 0x7070_0707);
        assert_eq!(canon_e2m1_u32_cpu(0x9A8B_C8DE), 0x9A0B_C0DE);
    }

    #[test]
    fn f32_to_bf16_rn_matches_ieee754_round_to_nearest_even() {
        assert_eq!(f32_to_bf16_rn_cpu(0.0), 0x0000);
        assert_eq!(f32_to_bf16_rn_cpu(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_rn_cpu(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_rn_cpu(-2.5), 0xC020);
        assert_eq!(f32_to_bf16_rn_cpu(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16_rn_cpu(f32::NEG_INFINITY), 0xFF80);
        assert_eq!(f32_to_bf16_rn_cpu(f32::NAN) & 0x7FC0, 0x7FC0);
        // Exact tie to even: 0x3F80_8000 (even LSB -> stay 0x3F80); 0x3F81_8000 (odd LSB -> 0x3F82).
        assert_eq!(f32_to_bf16_rn_cpu(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16_rn_cpu(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Just above tie: rounds up regardless of LSB.
        assert_eq!(f32_to_bf16_rn_cpu(f32::from_bits(0x3F80_8001)), 0x3F81);
    }

    #[test]
    fn verify_width_and_batch_size_invariance_is_bit_exact_zero_ulp() {
        // Non-multiple-of-64 N=131 (tests tail masking) and K=192, 320 (tests split_k=3 slicing).
        for &(n, k, split_k) in &[(131u32, 192u32, 3u32), (128u32, 320u32, 3u32)] {
            let max_m = 16usize;
            let half_k = (k / 2) as usize;
            let num_groups = (k / 16) as usize;
            let (a_all_p, a_all_s) = synthetic_nvfp4_matrix(max_m, k as usize, 0x1111_2222);
            let (b_p, b_s) = synthetic_nvfp4_matrix(n as usize, k as usize, 0x3333_4444);
            let scale2_ab = 0.03125_f32;

            // Single W=16 batch output.
            let out_w16 = w4a4_decode_roofline_gemm_cpu(
                &a_all_p, &a_all_s, &b_p, &b_s, scale2_ab, 16, n, k, split_k,
            )
            .unwrap();

            // W=1 (16 individual single-row runs) MUST match W=16 bit-for-bit (0 ULP).
            for r in 0..max_m {
                let ap = &a_all_p[r * half_k..(r + 1) * half_k];
                let asc = &a_all_s[r * num_groups..(r + 1) * num_groups];
                let out_w1 =
                    w4a4_decode_roofline_gemm_cpu(ap, asc, &b_p, &b_s, scale2_ab, 1, n, k, split_k)
                        .unwrap();
                assert_eq!(
                    out_w1.as_slice(),
                    &out_w16[r * (n as usize)..(r + 1) * (n as usize)],
                    "W=1 vs W=16 bit drift at row {r} (N={n}, K={k}, split_k={split_k})"
                );
            }

            // W=8 / B=8 (2 halves of 8 rows) MUST match W=16 bit-for-bit (0 ULP).
            for chunk in 0..2 {
                let r0 = chunk * 8;
                let r1 = r0 + 8;
                let ap = &a_all_p[r0 * half_k..r1 * half_k];
                let asc = &a_all_s[r0 * num_groups..r1 * num_groups];
                let out_w8 =
                    w4a4_decode_roofline_gemm_cpu(ap, asc, &b_p, &b_s, scale2_ab, 8, n, k, split_k)
                        .unwrap();
                assert_eq!(
                    out_w8.as_slice(),
                    &out_w16[r0 * (n as usize)..r1 * (n as usize)],
                    "W=8 vs W=16 bit drift in chunk {chunk} (N={n}, K={k})"
                );
            }

            // B=4 (4 chunks of 4 rows) MUST match B=1 and B=8 and W=16 bit-for-bit (0 ULP).
            for chunk in 0..4 {
                let r0 = chunk * 4;
                let r1 = r0 + 4;
                let ap = &a_all_p[r0 * half_k..r1 * half_k];
                let asc = &a_all_s[r0 * num_groups..r1 * num_groups];
                let out_b4 =
                    w4a4_decode_roofline_gemm_cpu(ap, asc, &b_p, &b_s, scale2_ab, 4, n, k, split_k)
                        .unwrap();
                assert_eq!(
                    out_b4.as_slice(),
                    &out_w16[r0 * (n as usize)..r1 * (n as usize)],
                    "B=4 vs W=16 bit drift in chunk {chunk} (N={n}, K={k})"
                );
            }
        }
    }

    #[test]
    fn split_k_variants_match_exact_fp64_reference_and_are_deterministic() {
        let m = 5usize;
        let n = 73usize;
        let k = 512usize;
        let scale2_ab = 0.0625_f32;
        let (a_p, a_s) = synthetic_nvfp4_matrix(m, k, 0xABCD_EF01);
        let (b_p, b_s) = synthetic_nvfp4_matrix(n, k, 0x5678_9ABC);

        // Compute FP64 ground truth over all K.
        let half_k = k / 2;
        let num_groups = k / 16;
        let mut ref_f64 = vec![0.0_f64; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut dot = 0.0_f64;
                for g in 0..num_groups {
                    let sa = f64::from(e4m3_byte_to_f32(a_s[r * num_groups + g] & 0x7F));
                    let sb = f64::from(e4m3_byte_to_f32(b_s[c * num_groups + g] & 0x7F));
                    let mut g_dot = 0.0_f64;
                    for bi in 0..8 {
                        let ab = canon_e2m1_byte_cpu(a_p[r * half_k + g * 8 + bi]);
                        let bb = canon_e2m1_byte_cpu(b_p[c * half_k + g * 8 + bi]);
                        g_dot += f64::from(DUAL_PLANE_E2M1_LUT[(ab & 0x0F) as usize])
                            * f64::from(DUAL_PLANE_E2M1_LUT[(bb & 0x0F) as usize]);
                        g_dot += f64::from(DUAL_PLANE_E2M1_LUT[((ab >> 4) & 0x0F) as usize])
                            * f64::from(DUAL_PLANE_E2M1_LUT[((bb >> 4) & 0x0F) as usize]);
                    }
                    dot += g_dot * sa * sb;
                }
                ref_f64[r * n + c] = dot * f64::from(scale2_ab);
            }
        }

        for split_k in [1u32, 2, 3, 4, 8] {
            let run1 = w4a4_decode_roofline_gemm_cpu(
                &a_p, &a_s, &b_p, &b_s, scale2_ab, m as u32, n as u32, k as u32, split_k,
            )
            .unwrap();
            let run2 = w4a4_decode_roofline_gemm_cpu(
                &a_p, &a_s, &b_p, &b_s, scale2_ab, m as u32, n as u32, k as u32, split_k,
            )
            .unwrap();
            assert_eq!(run1, run2, "non-deterministic output at split_k={split_k}");

            for (idx, (&got_bits, &truth)) in run1.iter().zip(&ref_f64).enumerate() {
                let got = f64::from(bf16_to_f32(got_bits));
                let tol = truth.abs() * 1.0e-2 + 1.0e-3;
                assert!(
                    (got - truth).abs() <= tol,
                    "split_k={split_k} idx={idx}: got={got} truth={truth} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn default_split_k_and_48sm_gb10_cta_grid_math_match_roofline_contract() {
        // 48-SM GB10: default split_k = 3 across all decode projections, independent of M.
        assert_eq!(default_w4a4_decode_split_k(48, 5120), 3);
        assert_eq!(default_w4a4_decode_split_k(48, 17408), 3);
        assert_eq!(default_w4a4_decode_split_k(64, 4096), 3);
        // Clamped when K has fewer than 3 tiles of 64.
        assert_eq!(default_w4a4_decode_split_k(48, 64), 1);
        assert_eq!(default_w4a4_decode_split_k(48, 128), 2);
        // Wider SM120 GPUs (>64 SMs): default split_k = 4.
        assert_eq!(default_w4a4_decode_split_k(170, 5120), 4);

        // Exact 48-SM wave multiples (0 tail quantization):
        for &(n, expected_ctas) in &[(5120u32, 240u32), (4096, 192), (17408, 816), (6144, 288)] {
            let plan = plan_w4a4_decode_sk3(16, n, 5120, 3).unwrap();
            assert_eq!(plan.total_ctas, expected_ctas);
            assert_eq!(plan.total_ctas % 48, 0);
        }
        // split_k = 1 needs 0 workspace bytes; split_k = 3 needs 3 * M * N * 4 bytes.
        assert_eq!(
            plan_w4a4_decode_sk3(8, 5120, 5120, 1)
                .unwrap()
                .workspace_bytes,
            0
        );
        assert_eq!(
            plan_w4a4_decode_sk3(16, 17408, 5120, 3)
                .unwrap()
                .workspace_bytes,
            3 * 16 * 17408 * 4
        );
    }

    #[test]
    fn env_parser_is_strict_zero_or_one_and_fails_closed() {
        assert_eq!(
            W4a4DecodeSk3Policy::parse_os(None).unwrap(),
            W4a4DecodeSk3Policy::Disabled
        );
        assert_eq!(
            W4a4DecodeSk3Policy::parse_optional(Some("0")).unwrap(),
            W4a4DecodeSk3Policy::Disabled
        );
        assert_eq!(
            W4a4DecodeSk3Policy::parse_optional(Some("1")).unwrap(),
            W4a4DecodeSk3Policy::Enabled
        );
        for bad in [
            "", " ", "0 ", " 1", "true", "false", "on", "off", "yes", "2", "-1",
        ] {
            let err = W4a4DecodeSk3Policy::parse_optional(Some(bad)).unwrap_err();
            assert_eq!(err, W4a4DecodeSk3ConfigError::InvalidToken(bad.to_owned()));
            assert!(err.to_string().contains(W4A4_DECODE_SK3_ENV));
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let non_utf8 = OsStr::from_bytes(&[0xFF, 0xFE]);
            let err = W4a4DecodeSk3Policy::parse_os(Some(non_utf8)).unwrap_err();
            assert_eq!(err, W4a4DecodeSk3ConfigError::NonUtf8);
            assert!(err.to_string().contains("UTF-8"));
        }
    }

    #[test]
    fn mock_gpu_launcher_validates_preconditions_and_emits_expected_launches() {
        let w = make_valid_weight();
        let kernels = W4a4DecodeSk3Kernels {
            gemm: KernelHandle(11),
            reduce: KernelHandle(22),
        };

        // 1. split_k = 1 -> 1 launch (no reduce).
        let gpu1 = MockGpuBackend::new();
        let plan1 = plan_w4a4_decode_sk3(8, 5120, 5120, 1).unwrap();
        w4a4_decode_roofline_gemm(
            &gpu1,
            kernels,
            &plan1,
            DevicePtr(0x1000),
            DevicePtr(0x3000),
            &w,
            DevicePtr::NULL,
            0,
            DevicePtr(0x5000),
            0.5,
            0x77,
        )
        .unwrap();
        assert_eq!(gpu1.launch_count(), 1);

        // 2. split_k = 3 -> 2 launches (GEMM + deterministic reduce).
        let gpu2 = MockGpuBackend::new();
        let plan3 = plan_w4a4_decode_sk3(16, 5120, 5120, 3).unwrap();
        w4a4_decode_roofline_gemm(
            &gpu2,
            kernels,
            &plan3,
            DevicePtr(0x1000),
            DevicePtr(0x3000),
            &w,
            DevicePtr(0x8000),
            plan3.workspace_bytes,
            DevicePtr(0x5000),
            0.5,
            0x88,
        )
        .unwrap();
        assert_eq!(gpu2.launch_count(), 2);

        // 3. Fail-closed checks MUST reject with 0 launches.
        let gpu_err = MockGpuBackend::new();
        assert!(plan_w4a4_decode_sk3(0, 64, 64, 1).is_err());
        assert!(plan_w4a4_decode_sk3(17, 64, 64, 1).is_err());
        assert!(plan_w4a4_decode_sk3(1, 0, 64, 1).is_err());
        assert!(plan_w4a4_decode_sk3(1, 64, 32, 1).is_err());
        assert!(plan_w4a4_decode_sk3(1, 64, 128, 0).is_err());
        assert!(plan_w4a4_decode_sk3(1, 64, 128, 9).is_err());
        assert!(plan_w4a4_decode_sk3(1, 64, 128, 3).is_err()); // split_k > K/64

        // Missing reduce handle for split_k=3 rejects BEFORE launching stage 1.
        assert!(
            w4a4_decode_roofline_gemm(
                &gpu_err,
                W4a4DecodeSk3Kernels {
                    gemm: KernelHandle(11),
                    reduce: KernelHandle(0),
                },
                &plan3,
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                &w,
                DevicePtr(0x8000),
                plan3.workspace_bytes,
                DevicePtr(0x5000),
                0.5,
                0,
            )
            .is_err()
        );
        // Undersized workspace rejects BEFORE launching stage 1.
        assert!(
            w4a4_decode_roofline_gemm(
                &gpu_err,
                kernels,
                &plan3,
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                &w,
                DevicePtr(0x8000),
                plan3.workspace_bytes - 4,
                DevicePtr(0x5000),
                0.5,
                0,
            )
            .is_err()
        );
        // Misaligned pointers / null / non-finite scale reject with 0 launches.
        for (ap, asc, ws, out, s2) in [
            (
                DevicePtr(0x1008),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3002),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8002),
                DevicePtr(0x5000),
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5001),
                0.5,
            ),
            (
                DevicePtr::NULL,
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr::NULL,
                DevicePtr(0x5000),
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr::NULL,
                0.5,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                f32::NAN,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                0.0,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                -1.0,
            ),
            (
                DevicePtr(0x1000),
                DevicePtr(0x3000),
                DevicePtr(0x8000),
                DevicePtr(0x5000),
                f32::INFINITY,
            ),
        ] {
            assert!(
                w4a4_decode_roofline_gemm(
                    &gpu_err,
                    kernels,
                    &plan3,
                    ap,
                    asc,
                    &w,
                    ws,
                    plan3.workspace_bytes,
                    out,
                    s2,
                    0,
                )
                .is_err()
            );
        }
        assert_eq!(gpu_err.launch_count(), 0);

        // 4. Graph-capture workspace guard never allocates mid-capture.
        let gpu_ws = Box::new(MockGpuBackend::new());
        let stream_ws = 0xDEAD_BEEF_u64;
        assert_eq!(
            ensure_w4a4_decode_sk3_workspace(
                gpu_ws.as_ref(),
                stream_ws,
                4096,
                W4a4WorkspaceAllocPolicy::CaptureNoAlloc
            )
            .unwrap(),
            None
        );
        assert_eq!(gpu_ws.alloc_count(), 0);
        let (ws_ptr, ws_sz) = ensure_w4a4_decode_sk3_workspace(
            gpu_ws.as_ref(),
            stream_ws,
            4096,
            W4a4WorkspaceAllocPolicy::AllowAlloc,
        )
        .unwrap()
        .unwrap();
        assert_eq!(ws_sz, 4096);
        assert_eq!(gpu_ws.alloc_count(), 1);
        // Subsequent CaptureNoAlloc with <= 4096 reuses existing buffer.
        assert_eq!(
            ensure_w4a4_decode_sk3_workspace(
                gpu_ws.as_ref(),
                stream_ws,
                2048,
                W4a4WorkspaceAllocPolicy::CaptureNoAlloc
            )
            .unwrap(),
            Some((ws_ptr, 4096))
        );
        assert_eq!(gpu_ws.alloc_count(), 1);
    }

    #[test]
    fn workspace_growth_frees_previous_and_preserves_cache_on_sync_error() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct SyncFailGpu {
            inner: MockGpuBackend,
            fail_sync: AtomicBool,
        }

        impl GpuBackend for SyncFailGpu {
            fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                func: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared_mem: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> Result<()> {
                self.inner
                    .launch(func, grid, block, shared_mem, stream, params)
            }
            fn synchronize(&self, stream: u64) -> Result<()> {
                if self.fail_sync.load(Ordering::Relaxed) {
                    anyhow::bail!("injected stream synchronize failure");
                }
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
                self.inner.kernel(module, func_name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> Result<usize> {
                self.inner.free_memory()
            }
        }

        let gpu = Box::new(SyncFailGpu {
            inner: MockGpuBackend::new(),
            fail_sync: AtomicBool::new(false),
        });
        let stream = 0xCAFE_BABE_u64;

        // Initial eager allocation (4096 B).
        let (p1, sz1) = ensure_w4a4_decode_sk3_workspace(
            gpu.as_ref(),
            stream,
            4096,
            W4a4WorkspaceAllocPolicy::AllowAlloc,
        )
        .unwrap()
        .unwrap();
        assert_eq!(sz1, 4096);
        assert_eq!(gpu.inner.alloc_count(), 1);

        // Growth fails on stream synchronize: previous buffer MUST remain cached (0 leak).
        gpu.fail_sync.store(true, Ordering::Relaxed);
        assert!(
            ensure_w4a4_decode_sk3_workspace(
                gpu.as_ref(),
                stream,
                8192,
                W4a4WorkspaceAllocPolicy::AllowAlloc,
            )
            .is_err()
        );
        assert_eq!(gpu.inner.alloc_count(), 1);
        assert_eq!(gpu.inner.invalid_free_count(), 0);
        // Cached `p1` is still tracked and reusable without allocation.
        assert_eq!(
            ensure_w4a4_decode_sk3_workspace(
                gpu.as_ref(),
                stream,
                4096,
                W4a4WorkspaceAllocPolicy::CaptureNoAlloc,
            )
            .unwrap(),
            Some((p1, 4096))
        );

        // Once synchronize succeeds, growth frees `p1` and installs `p2` (8192 B).
        gpu.fail_sync.store(false, Ordering::Relaxed);
        let (p2, sz2) = ensure_w4a4_decode_sk3_workspace(
            gpu.as_ref(),
            stream,
            8192,
            W4a4WorkspaceAllocPolicy::AllowAlloc,
        )
        .unwrap()
        .unwrap();
        assert_eq!(sz2, 8192);
        assert_ne!(p2, p1);
        assert_eq!(gpu.inner.alloc_count(), 1);
        assert_eq!(gpu.inner.invalid_free_count(), 0);
    }

    #[test]
    fn ffn_preflight_rejects_null_or_misaligned_weights_scales_buffers_and_handles() {
        let ok_w = make_valid_weight();
        let ok_act = KernelHandle(1);
        let ok_q = KernelHandle(2);
        let ok_in = DevicePtr(0x1000);
        let ok_g = DevicePtr(0x2000);
        let ok_u = DevicePtr(0x3000);
        let ok_out = DevicePtr(0x4000);

        assert!(w4a4_decode_sk3_ffn_preflight_ok(
            &ok_w, &ok_w, &ok_w, ok_act, ok_q, ok_in, ok_g, ok_u, ok_out
        ));
        // Missing kernel handles fail closed.
        assert!(!w4a4_decode_sk3_ffn_preflight_ok(
            &ok_w,
            &ok_w,
            &ok_w,
            KernelHandle(0),
            ok_q,
            ok_in,
            ok_g,
            ok_u,
            ok_out
        ));
        assert!(!w4a4_decode_sk3_ffn_preflight_ok(
            &ok_w,
            &ok_w,
            &ok_w,
            ok_act,
            KernelHandle(0),
            ok_in,
            ok_g,
            ok_u,
            ok_out
        ));
        // Null/misaligned weight_scale or weight on down_proj (or gate/up) fails BEFORE any launch.
        let bad_null_scale = QuantizedWeight {
            weight_scale: DevicePtr::NULL,
            ..ok_w
        };
        let bad_align_w = QuantizedWeight {
            weight: DevicePtr(0x2008),
            ..ok_w
        };
        let bad_align_s = QuantizedWeight {
            weight_scale: DevicePtr(0x4002),
            ..ok_w
        };
        let bad_scale2 = QuantizedWeight {
            weight_scale_2: f32::NAN,
            ..ok_w
        };
        for bad_w in [bad_null_scale, bad_align_w, bad_align_s, bad_scale2] {
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &bad_w, &ok_w, &ok_w, ok_act, ok_q, ok_in, ok_g, ok_u, ok_out
            ));
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &bad_w, &ok_w, ok_act, ok_q, ok_in, ok_g, ok_u, ok_out
            ));
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &ok_w, &bad_w, ok_act, ok_q, ok_in, ok_g, ok_u, ok_out
            ));
        }
        // Null or odd-aligned BF16 I/O buffers fail closed.
        for bad_ptr in [DevicePtr::NULL, DevicePtr(0x4001)] {
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &ok_w, &ok_w, ok_act, ok_q, bad_ptr, ok_g, ok_u, ok_out
            ));
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &ok_w, &ok_w, ok_act, ok_q, ok_in, bad_ptr, ok_u, ok_out
            ));
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &ok_w, &ok_w, ok_act, ok_q, ok_in, ok_g, bad_ptr, ok_out
            ));
            assert!(!w4a4_decode_sk3_ffn_preflight_ok(
                &ok_w, &ok_w, &ok_w, ok_act, ok_q, ok_in, ok_g, ok_u, bad_ptr
            ));
        }
        // Activation scratch alignment checks.
        assert!(w4a4_decode_sk3_activation_scratch_ok(
            DevicePtr(0x1000),
            DevicePtr(0x2000)
        ));
        assert!(!w4a4_decode_sk3_activation_scratch_ok(
            DevicePtr::NULL,
            DevicePtr(0x2000)
        ));
        assert!(!w4a4_decode_sk3_activation_scratch_ok(
            DevicePtr(0x1008),
            DevicePtr(0x2000)
        ));
        assert!(!w4a4_decode_sk3_activation_scratch_ok(
            DevicePtr(0x1000),
            DevicePtr::NULL
        ));
        assert!(!w4a4_decode_sk3_activation_scratch_ok(
            DevicePtr(0x1000),
            DevicePtr(0x2002)
        ));
    }

    #[test]
    fn cuda_source_preserves_isa_and_determinism_contracts() {
        let cu = include_str!("../../../../../kernels/gb10/common/w4a4_decode_roofline_sm120.cu");
        assert!(cu.contains("SPDX-License-Identifier: AGPL-3.0-only"));
        assert!(cu.contains("void w4a4_gemm_m16_sk3_sm120("));
        assert!(cu.contains("void reduce_w4a4_sk3_bf16_sm120("));
        assert!(cu.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."));
        assert!(cu.contains("m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"));
        assert!(cu.contains("ld.global.cs.L2::cache_hint.v4.u32"));
        assert!(cu.contains("ld.global.nc.v4.u32"));
        assert!(cu.contains("cvt.rn.bf16.f32"));
        assert!(cu.contains("0x77777777u"));
        // Check zero actual atomicAdd instructions in executable code (ignore comments).
        for line in cu.lines() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains("atomicAdd"),
                "non-deterministic atomicAdd found in CUDA kernel: {line}"
            );
        }
    }
}
