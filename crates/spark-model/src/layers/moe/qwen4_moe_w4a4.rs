// SPDX-License-Identifier: AGPL-3.0-only

//! Native Blackwell `W4A4` (`NVFP4 x NVFP4`, `m16n8k64` `4X` `e2m1 x e2m1`)
//! persistent MoE prefill control, scratch layout, and kernel dispatch (`v32_w4a4`).
//!
//! Controlled by `ATLAS_QWEN4_PREFILL_MOE_W4A4`:
//! - unset / `"0"` / `"off"`: disabled (default; preserves existing 0-ULP W4A8 path
//!   and leaves GB10 `active_sm_count() <= 64` defaults 100% unchanged).
//! - `"gateup"`: hybrid mode — `W4A4` GATEUP emits exact MXFP8 `oq`/`os` feeding
//!   `V31_WS` DOWN (eliminates DOWN activation outlier risk).
//! - `"1"` / `"all"`: full `W4A4` — `W4A4` GATEUP emits NVFP4 `oq [expanded, 320]`
//!   + UE4M3 `os [expanded, 40]` feeding `W4A4` DOWN (`slot_q`/`slot_s` MXFP8 or BF16).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::super::ExpertPtrTable;
use super::super::qwen4_compact_contract as abi;

/// Environment selector for native Blackwell `W4A4` MoE prefill (`v32_w4a4`).
pub(crate) const W4A4_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_W4A4";
/// Environment override for the number of sensitive boundary layers protected on exact W4A8
/// (`V31_WS`). Default protects first 2 (`0..2`) and last 3 (`num_layers-3..num_layers`).
/// Set `=0` to disable protection, `=N` for symmetric `(N, N)`, or `=F,L` for `(F, L)`.
pub(crate) const W4A4_PROTECT_LAYERS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_W4A4_PROTECT_LAYERS";

/// Minimum prefill row count admitted by `v32_w4a4` (`rows > 256`), guaranteeing speculative
/// decode/verify frames (`rows <= 256`) never enter `W4A4`.
pub(crate) const W4A4_MIN_PREFILL_ROWS: usize = 257;

/// CUDA module name auto-discovered from `qwen4_moe_w4a4_v32.cu`.
pub(crate) const W4A4_MODULE: &str = "qwen4_moe_w4a4_v32";

/// Kernel entry point names exported by [`W4A4_MODULE`].
pub(crate) const KERNEL_QUANT_GATE: &str = "qwen4_nvfp4_quant_rows_v32_gate";
pub(crate) const KERNEL_GATEUP_FLAT: &str = "qwen4_moe_w4a4_gateup_v32";
pub(crate) const KERNEL_GATEUP_TS4E: &str = "qwen4_moe_w4a4_gateup_v32_ts4e";
pub(crate) const KERNEL_GATEUP_OQ4_FLAT: &str = "qwen4_moe_w4a4_gateup_v32_oq4";
pub(crate) const KERNEL_GATEUP_OQ4_TS4E: &str = "qwen4_moe_w4a4_gateup_v32_oq4_ts4e";
pub(crate) const KERNEL_DOWN_MXFP8_FLAT: &str = "qwen4_moe_w4a4_down_v32";
pub(crate) const KERNEL_DOWN_MXFP8_TS4E: &str = "qwen4_moe_w4a4_down_v32_ts4e";
pub(crate) const KERNEL_DOWN_BF16_FLAT: &str = "qwen4_moe_w4a4_down_v32_bf16";
pub(crate) const KERNEL_DOWN_BF16_TS4E: &str = "qwen4_moe_w4a4_down_v32_bf16_ts4e";

/// Thread-block width for all `v32_w4a4` kernels (8 warps = 4 M x 2 N).
pub(crate) const W4A4_BLOCK: u32 = 256;

/// Target concurrent persistent CTAs per SM for `v32_w4a4` (`<= 48 KiB` SMEM, `<= 128` regs).
pub(crate) const W4A4_CTAS_PER_SM: u32 = 2;

/// Target concurrent CTAs per SM for `qwen4_nvfp4_quant_rows_v32_gate` on high-SM GPUs.
const W4A4_QUANT_CTAS_PER_SM: u32 = 6;
/// GB10 (`<= 64` SMs) grid cap for `qwen4_nvfp4_quant_rows_v32_gate`.
const W4A4_QUANT_GRID_CAP_GB10: u32 = 1024;
/// SM count threshold separating GB10-class (`<= 64` SMs) from high-SM Blackwell GPUs.
const HIGH_SM_THRESHOLD: u32 = 64;

/// Maximum dynamic SMEM budget per CTA to guarantee 2 CTAs/SM on SM120 / SM121.
pub(crate) const W4A4_MAX_2CTA_SMEM_BYTES: u32 = 49_152;
/// Dynamic SMEM for flat-scale `v32_w4a4` GATEUP (`39,488 B <= 49,152 B`).
pub(crate) const W4A4_FLAT_GATEUP_SMEM_BYTES: u32 = 39_488;
/// Dynamic SMEM for flat-scale `v32_w4a4` DOWN (`44,608 B <= 49,152 B`, incl. per-slice scale2 ratios).
pub(crate) const W4A4_FLAT_DOWN_SMEM_BYTES: u32 = 44_608;
/// Dynamic SMEM for Atlas-TS4E-128 `v32_w4a4` GATEUP (`41,600 B <= 49,152 B`).
pub(crate) const W4A4_TS4E_GATEUP_SMEM_BYTES: u32 = 41_600;
/// Dynamic SMEM for Atlas-TS4E-128 `v32_w4a4` DOWN (`45,376 B <= 49,152 B`, incl. per-slice scale2 ratios).
pub(crate) const W4A4_TS4E_DOWN_SMEM_BYTES: u32 = 45_376;

const _: () = {
    assert!(W4A4_FLAT_GATEUP_SMEM_BYTES <= W4A4_MAX_2CTA_SMEM_BYTES);
    assert!(W4A4_FLAT_DOWN_SMEM_BYTES <= W4A4_MAX_2CTA_SMEM_BYTES);
    assert!(W4A4_TS4E_GATEUP_SMEM_BYTES <= W4A4_MAX_2CTA_SMEM_BYTES);
    assert!(W4A4_TS4E_DOWN_SMEM_BYTES <= W4A4_MAX_2CTA_SMEM_BYTES);
};

/// Canonical Flash-Next MoE dimensions enforced by `v32_w4a4`.
pub(crate) const FLASH_NEXT_HIDDEN: u32 = 2_560;
pub(crate) const FLASH_NEXT_INTER: u32 = 640;
pub(crate) const FLASH_NEXT_TOPK: usize = 10;

/// Per-row NVFP4 input bytes (`xq`: `2560/2 = 1280`, `xs`: `2560/16 = 160`, `x_scale2`: `4`).
const XQ_NVFP4_BYTES_PER_ROW: usize = (FLASH_NEXT_HIDDEN as usize) / 2;
const XS_NVFP4_BYTES_PER_ROW: usize = (FLASH_NEXT_HIDDEN as usize) / 16;
const X_SCALE2_BYTES_PER_ROW: usize = std::mem::size_of::<f32>();

/// Per-expanded-row intermediate bytes for hybrid MXFP8 vs full NVFP4 `oq`/`os`/`oq_scale2`.
const OQ_MXFP8_BYTES_PER_PAIR: usize = FLASH_NEXT_INTER as usize;
const OS_MXFP8_BYTES_PER_PAIR: usize = (FLASH_NEXT_INTER as usize) / 32;
const OQ_NVFP4_BYTES_PER_PAIR: usize = (FLASH_NEXT_INTER as usize) / 2;
const OS_NVFP4_BYTES_PER_PAIR: usize = (FLASH_NEXT_INTER as usize) / 16;
/// Per-expanded-row dynamic FP32 `oq_scale2` slices (`I / 64 = 10` slices x 4 B = 40 B).
const OQ_SCALE2_SLICES_PER_PAIR: usize = (FLASH_NEXT_INTER as usize) / 64;
const OQ_SCALE2_BYTES_PER_PAIR: usize = OQ_SCALE2_SLICES_PER_PAIR * std::mem::size_of::<f32>();

/// Scalar fallback FP32 `oq_scale2` used only when `oq_scale2_buf` is NULL.
/// Normal full-W4A4 dispatch supplies dynamic per-`(orow, kh64)` `oq_scale2_buf` instead.
pub(crate) const DEFAULT_OQ_SCALE2: f32 = 0.5;

/// Boundary-layer protection policy keeping sensitive first/last layers on exact W4A8 (`V31_WS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct W4a4ProtectedBoundary {
    /// Leading layers `0..first` kept on exact W4A8.
    pub first: usize,
    /// Trailing layers `num_layers - last..num_layers` kept on exact W4A8.
    pub last: usize,
}

impl W4a4ProtectedBoundary {
    /// Default protection: first 2 layers (`0..2`) and last 3 layers (`num_layers-3..num_layers`).
    pub(crate) const DEFAULT: Self = Self { first: 2, last: 3 };
    /// Disables boundary-layer protection (`=0`).
    pub(crate) const NONE: Self = Self { first: 0, last: 0 };

    /// Returns `true` if `layer_idx` falls inside the protected leading or trailing band.
    #[must_use]
    pub(crate) const fn is_protected(
        self,
        layer_idx: Option<usize>,
        num_layers: Option<usize>,
    ) -> bool {
        let (Some(idx), Some(total)) = (layer_idx, num_layers) else {
            return false;
        };
        if idx < self.first {
            return true;
        }
        self.last > 0 && idx >= total.saturating_sub(self.last)
    }
}

impl Default for W4a4ProtectedBoundary {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Parses `ATLAS_QWEN4_PREFILL_MOE_W4A4_PROTECT_LAYERS` strictly (fail-closed on invalid values).
pub(crate) fn parse_w4a4_protected_boundary(
    value: Option<&str>,
) -> std::result::Result<W4a4ProtectedBoundary, String> {
    let Some(raw) = value.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(W4a4ProtectedBoundary::DEFAULT);
    };
    if let Some((f, l)) = raw.split_once(',') {
        let first = f.trim().parse::<usize>().map_err(|_| {
            format!(
                "{W4A4_PROTECT_LAYERS_SELECTOR}: invalid leading count {f:?} (expected integer)"
            )
        })?;
        let last = l.trim().parse::<usize>().map_err(|_| {
            format!(
                "{W4A4_PROTECT_LAYERS_SELECTOR}: invalid trailing count {l:?} (expected integer)"
            )
        })?;
        return Ok(W4a4ProtectedBoundary { first, last });
    }
    let n = raw.parse::<usize>().map_err(|_| {
        format!(
            "{W4A4_PROTECT_LAYERS_SELECTOR}: invalid value {raw:?} \
             (expected 0, N, or first,last)"
        )
    })?;
    if n == 0 {
        return Ok(W4a4ProtectedBoundary::NONE);
    }
    Ok(W4a4ProtectedBoundary { first: n, last: n })
}

/// Returns the cached process-wide [`W4a4ProtectedBoundary`].
///
/// # Errors
/// Returns an error if [`W4A4_PROTECT_LAYERS_SELECTOR`] is non-UTF-8 or malformed.
pub(crate) fn w4a4_protected_boundary() -> Result<W4a4ProtectedBoundary> {
    static BOUNDARY: OnceLock<std::result::Result<W4a4ProtectedBoundary, String>> = OnceLock::new();
    BOUNDARY
        .get_or_init(|| {
            let value = match std::env::var(W4A4_PROTECT_LAYERS_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => return Err(format!("invalid {W4A4_PROTECT_LAYERS_SELECTOR}: {e}")),
            };
            parse_w4a4_protected_boundary(value.as_deref())
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Operating mode selected via [`W4A4_SELECTOR`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum W4a4Mode {
    /// Disabled (default): keep existing W4A8 path (`V31_WS` / `V28`).
    #[default]
    Off,
    /// Hybrid mode: native `W4A4` GATEUP writing exact MXFP8 `oq`/`os` for `V31_WS` DOWN.
    GateUp,
    /// Full `W4A4` mode: `W4A4` GATEUP writing NVFP4 `oq`/`os` + `W4A4` DOWN.
    All,
}

impl W4a4Mode {
    /// Returns `true` when any `W4A4` kernel stage is enabled.
    #[must_use]
    pub(crate) const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Returns `true` when `W4A4` DOWN is also enabled (`mode == All`).
    #[must_use]
    pub(crate) const fn uses_w4a4_down(self) -> bool {
        matches!(self, Self::All)
    }

    /// Stable human-readable label for receipts and diagnostics.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::GateUp => "gateup",
            Self::All => "all",
        }
    }

    /// Evaluates per-dispatch eligibility for `v32_w4a4` with an explicit boundary-layer policy.
    #[must_use]
    pub(crate) const fn admits_with_boundary(
        self,
        eligibility: W4a4Eligibility,
        boundary: W4a4ProtectedBoundary,
    ) -> bool {
        self.is_enabled()
            && eligibility.rows >= W4A4_MIN_PREFILL_ROWS
            && !eligibility.verify_scope_active
            && eligibility.w4a8_is_v28
            && !eligibility.v28_m64_forced
            && !eligibility.has_corrections
            && !eligibility.l2_windowed
            && !eligibility.v29_m256_active
            && !eligibility.v30_2cta_active
            && eligibility.kernels_loaded
            && !boundary.is_protected(eligibility.layer_idx, eligibility.num_layers)
    }

    /// Evaluates per-dispatch eligibility for `v32_w4a4` using the default boundary-layer policy
    /// (protects first 2 layers `0..2` and last 3 layers `num_layers-3..num_layers`).
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn admits(self, eligibility: W4a4Eligibility) -> bool {
        self.admits_with_boundary(eligibility, W4a4ProtectedBoundary::DEFAULT)
    }
}

/// Dispatch-time eligibility inputs for [`W4a4Mode::admits_with_boundary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct W4a4Eligibility {
    pub rows: usize,
    pub w4a8_is_v28: bool,
    pub v28_m64_forced: bool,
    pub has_corrections: bool,
    pub l2_windowed: bool,
    pub v29_m256_active: bool,
    pub v30_2cta_active: bool,
    pub verify_scope_active: bool,
    pub kernels_loaded: bool,
    pub layer_idx: Option<usize>,
    pub num_layers: Option<usize>,
}

/// Weight-scale layout consumed by `v32_w4a4` GEMM kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum W4a4ScaleLayout {
    /// Standard transposed flat UE4M3 scales (`[K/16, N]`).
    Flat,
    /// Atlas-TS4E-128 compressed scale blobs decoded on the fly in shared memory.
    Ts4e128,
}

impl W4a4ScaleLayout {
    #[must_use]
    pub(crate) const fn is_ts4e(self) -> bool {
        matches!(self, Self::Ts4e128)
    }
}

/// Output storage format written by `qwen4_moe_w4a4_down_v32*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum W4a4DownOutput {
    /// BF16 expanded slot rows `[expanded, 2560]` written to `expert_down_out`.
    Bf16 { c_out: DevicePtr },
    /// MXFP8 expanded slot rows (`slot_q` + `slot_s`) matching `hc_combine`.
    Mxfp8Slots {
        slot_q: DevicePtr,
        slot_s: DevicePtr,
    },
}

/// Parses `ATLAS_QWEN4_PREFILL_MOE_W4A4` strictly (fail-closed on unknown tokens).
pub(crate) fn parse_w4a4_mode(value: Option<&str>) -> std::result::Result<W4a4Mode, &'static str> {
    match value {
        None | Some("0") | Some("off") => Ok(W4a4Mode::Off),
        Some("gateup") => Ok(W4a4Mode::GateUp),
        Some("1") | Some("all") => Ok(W4a4Mode::All),
        _ => Err("must be unset, 0, off, gateup, 1, or all"),
    }
}

/// Returns the cached process-wide [`W4a4Mode`] parsed from [`W4A4_SELECTOR`].
///
/// # Errors
/// Returns an error if [`W4A4_SELECTOR`] contains invalid UTF-8 or an unrecognized token.
pub(crate) fn w4a4_mode() -> Result<W4a4Mode> {
    static MODE: OnceLock<std::result::Result<W4a4Mode, String>> = OnceLock::new();
    MODE.get_or_init(|| {
        let value = match std::env::var(W4A4_SELECTOR) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(format!("invalid {W4A4_SELECTOR}: {e}")),
        };
        parse_w4a4_mode(value.as_deref()).map_err(|error| format!("{W4A4_SELECTOR}: {error}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Loaded kernel handle bundle for `qwen4_moe_w4a4_v32`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct W4a4V32Kernels {
    /// Single-read BF16 -> NVFP4 (`xq`, `xs`, `x_scale2`) + `shared_gate` dot quantizer.
    pub quant_gate: KernelHandle,
    /// Hybrid GATEUP (flat scales): NVFP4 x NVFP4 -> MXFP8 `oq`/`os` (for `V31_WS` DOWN).
    pub gateup_flat: KernelHandle,
    /// Hybrid GATEUP (TS4E-128 scales): NVFP4 x NVFP4 -> MXFP8 `oq`/`os`.
    pub gateup_ts4e: KernelHandle,
    /// Full-W4A4 GATEUP (flat scales): NVFP4 x NVFP4 -> NVFP4 `oq`/`os`.
    pub gateup_oq4_flat: KernelHandle,
    /// Full-W4A4 GATEUP (TS4E-128 scales): NVFP4 x NVFP4 -> NVFP4 `oq`/`os`.
    pub gateup_oq4_ts4e: KernelHandle,
    /// Full-W4A4 DOWN (flat scales): NVFP4 `oq`/`os` -> MXFP8 `slot_q`/`slot_s`.
    pub down_mxfp8_flat: KernelHandle,
    /// Full-W4A4 DOWN (TS4E-128 scales): NVFP4 `oq`/`os` -> MXFP8 `slot_q`/`slot_s`.
    pub down_mxfp8_ts4e: KernelHandle,
    /// Full-W4A4 DOWN (flat scales): NVFP4 `oq`/`os` -> BF16 `C`.
    pub down_bf16_flat: KernelHandle,
    /// Full-W4A4 DOWN (TS4E-128 scales): NVFP4 `oq`/`os` -> BF16 `C`.
    pub down_bf16_ts4e: KernelHandle,
}

impl W4a4V32Kernels {
    /// Selects the GATEUP kernel handle and dynamic SMEM footprint for `(mode, layout)`.
    #[must_use]
    pub(crate) const fn gateup_for(
        &self,
        mode: W4a4Mode,
        layout: W4a4ScaleLayout,
    ) -> (KernelHandle, u32) {
        match (mode, layout) {
            (W4a4Mode::All, W4a4ScaleLayout::Flat) => {
                (self.gateup_oq4_flat, W4A4_FLAT_GATEUP_SMEM_BYTES)
            }
            (W4a4Mode::All, W4a4ScaleLayout::Ts4e128) => {
                (self.gateup_oq4_ts4e, W4A4_TS4E_GATEUP_SMEM_BYTES)
            }
            (_, W4a4ScaleLayout::Flat) => (self.gateup_flat, W4A4_FLAT_GATEUP_SMEM_BYTES),
            (_, W4a4ScaleLayout::Ts4e128) => (self.gateup_ts4e, W4A4_TS4E_GATEUP_SMEM_BYTES),
        }
    }

    /// Selects the DOWN kernel handle and dynamic SMEM footprint for `(output, layout)`.
    #[must_use]
    pub(crate) const fn down_for(
        &self,
        output: W4a4DownOutput,
        layout: W4a4ScaleLayout,
    ) -> (KernelHandle, u32) {
        match (output, layout) {
            (W4a4DownOutput::Mxfp8Slots { .. }, W4a4ScaleLayout::Flat) => {
                (self.down_mxfp8_flat, W4A4_FLAT_DOWN_SMEM_BYTES)
            }
            (W4a4DownOutput::Mxfp8Slots { .. }, W4a4ScaleLayout::Ts4e128) => {
                (self.down_mxfp8_ts4e, W4A4_TS4E_DOWN_SMEM_BYTES)
            }
            (W4a4DownOutput::Bf16 { .. }, W4a4ScaleLayout::Flat) => {
                (self.down_bf16_flat, W4A4_FLAT_DOWN_SMEM_BYTES)
            }
            (W4a4DownOutput::Bf16 { .. }, W4a4ScaleLayout::Ts4e128) => {
                (self.down_bf16_ts4e, W4A4_TS4E_DOWN_SMEM_BYTES)
            }
        }
    }
}

/// Loads all 9 `v32_w4a4` kernel handles from [`W4A4_MODULE`].
///
/// All `v32_w4a4` GEMM kernels fit within 48 KiB (`<= 45,376 B <= 49,152 B`),
/// so 2 CTAs/SM occupancy requires zero dynamic shared-memory opt-in expansion.
///
/// # Errors
/// Returns an error if any kernel symbol is missing from the compiled PTX/cubin module.
pub(crate) fn load_w4a4_v32_kernels(gpu: &dyn GpuBackend) -> Result<W4a4V32Kernels> {
    let load = |name: &str| {
        gpu.kernel(W4A4_MODULE, name)
            .with_context(|| format!("failed to load {W4A4_MODULE}::{name}"))
    };
    Ok(W4a4V32Kernels {
        quant_gate: load(KERNEL_QUANT_GATE)?,
        gateup_flat: load(KERNEL_GATEUP_FLAT)?,
        gateup_ts4e: load(KERNEL_GATEUP_TS4E)?,
        gateup_oq4_flat: load(KERNEL_GATEUP_OQ4_FLAT)?,
        gateup_oq4_ts4e: load(KERNEL_GATEUP_OQ4_TS4E)?,
        down_mxfp8_flat: load(KERNEL_DOWN_MXFP8_FLAT)?,
        down_mxfp8_ts4e: load(KERNEL_DOWN_MXFP8_TS4E)?,
        down_bf16_flat: load(KERNEL_DOWN_BF16_FLAT)?,
        down_bf16_ts4e: load(KERNEL_DOWN_BF16_TS4E)?,
    })
}

/// Zero-allocation byte-offset layout slicing the existing compact MoE scratch arenas:
/// - `expert_up_out` (or `expert_down_out` during shared overlap; `>= rows * 2640 B`) hosts
///   NVFP4 `xq` + `xs` + `x_scale2` (`rows * 1444 B`).
/// - `expert_gate_out` (`>= expanded * 1280 B`) hosts either hybrid MXFP8 `oq`/`os`
///   (`expanded * 660 B`) or full NVFP4 `oq`/`os` + per-`(orow, kh64)` dynamic `oq_scale2`
///   (`expanded * 400 B`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct W4a4ScratchLayout {
    pub xq_offset: usize,
    pub xs_offset: usize,
    pub x_scale2_offset: usize,
    pub up_used_bytes: usize,
    pub oq_offset: usize,
    pub os_offset: usize,
    /// Byte offset of `[expanded, 10]` FP32 per-`(orow, kh64)` dynamic `oq_scale2` inside
    /// `expert_gate_out` (`Some` in [`W4a4Mode::All`], `None` otherwise).
    pub oq_scale2_offset: Option<usize>,
    pub gate_used_bytes: usize,
}

impl W4a4ScratchLayout {
    /// Computes and bounds-checks zero-alloc offsets inside `expert_up_out` and `expert_gate_out`.
    ///
    /// # Errors
    /// Returns an error on arithmetic overflow or if either pre-allocated arena is undersized.
    pub(crate) fn for_rows(
        rows: usize,
        mode: W4a4Mode,
        expert_up_out_bytes: usize,
        expert_gate_out_bytes: usize,
    ) -> Result<Self> {
        ensure!(rows > 0, "v32_w4a4 requires rows > 0");
        let pairs = rows
            .checked_mul(FLASH_NEXT_TOPK)
            .context("v32_w4a4 expanded row count overflow")?;

        let xq_bytes = rows
            .checked_mul(XQ_NVFP4_BYTES_PER_ROW)
            .context("v32_w4a4 xq byte size overflow")?;
        let xs_bytes = rows
            .checked_mul(XS_NVFP4_BYTES_PER_ROW)
            .context("v32_w4a4 xs byte size overflow")?;
        let x_scale2_offset = align_up_16(
            xq_bytes
                .checked_add(xs_bytes)
                .context("v32_w4a4 x_scale2 offset overflow")?,
        )?;
        let x_scale2_bytes = rows
            .checked_mul(X_SCALE2_BYTES_PER_ROW)
            .context("v32_w4a4 x_scale2 byte size overflow")?;
        let up_used_bytes = x_scale2_offset
            .checked_add(x_scale2_bytes)
            .context("v32_w4a4 expert_up_out footprint overflow")?;
        ensure!(
            up_used_bytes <= expert_up_out_bytes,
            "v32_w4a4 NVFP4 input ({up_used_bytes} B) exceeds expert_up_out ({expert_up_out_bytes} B)"
        );

        let (oq_stride, os_stride) = match mode {
            W4a4Mode::All => (OQ_NVFP4_BYTES_PER_PAIR, OS_NVFP4_BYTES_PER_PAIR),
            W4a4Mode::GateUp | W4a4Mode::Off => (OQ_MXFP8_BYTES_PER_PAIR, OS_MXFP8_BYTES_PER_PAIR),
        };
        let oq_bytes = pairs
            .checked_mul(oq_stride)
            .context("v32_w4a4 oq byte size overflow")?;
        let os_bytes = pairs
            .checked_mul(os_stride)
            .context("v32_w4a4 os byte size overflow")?;
        let oq_os_bytes = oq_bytes
            .checked_add(os_bytes)
            .context("v32_w4a4 expert_gate_out footprint overflow")?;
        let (oq_scale2_offset, gate_used_bytes) = if mode.uses_w4a4_down() {
            let s2_off = align_up_16(oq_os_bytes)?;
            let s2_bytes = pairs
                .checked_mul(OQ_SCALE2_BYTES_PER_PAIR)
                .context("v32_w4a4 oq_scale2 byte size overflow")?;
            let total = s2_off
                .checked_add(s2_bytes)
                .context("v32_w4a4 expert_gate_out footprint overflow")?;
            (Some(s2_off), total)
        } else {
            (None, oq_os_bytes)
        };
        ensure!(
            gate_used_bytes <= expert_gate_out_bytes,
            "v32_w4a4 intermediate ({gate_used_bytes} B) exceeds expert_gate_out ({expert_gate_out_bytes} B)"
        );

        Ok(Self {
            xq_offset: 0,
            xs_offset: xq_bytes,
            x_scale2_offset,
            up_used_bytes,
            oq_offset: 0,
            os_offset: oq_bytes,
            oq_scale2_offset,
            gate_used_bytes,
        })
    }

    /// Slices `(xq, xs, x_scale2)` device pointers inside `expert_up_out` with zero allocation.
    #[must_use]
    pub(crate) fn input_ptrs(self, expert_up_out: DevicePtr) -> (DevicePtr, DevicePtr, DevicePtr) {
        (
            expert_up_out.offset(self.xq_offset),
            expert_up_out.offset(self.xs_offset),
            expert_up_out.offset(self.x_scale2_offset),
        )
    }

    /// Slices `(oq, os, oq_scale2)` device pointers inside `expert_gate_out` with zero allocation.
    /// `oq_scale2` is [`DevicePtr::NULL`] outside [`W4a4Mode::All`].
    #[must_use]
    pub(crate) fn intermediate_ptrs(
        self,
        expert_gate_out: DevicePtr,
    ) -> (DevicePtr, DevicePtr, DevicePtr) {
        let oq_scale2 = self
            .oq_scale2_offset
            .map_or(DevicePtr::NULL, |off| expert_gate_out.offset(off));
        (
            expert_gate_out.offset(self.oq_offset),
            expert_gate_out.offset(self.os_offset),
            oq_scale2,
        )
    }
}

fn align_up_16(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(15)
        .map(|v| v & !15usize)
        .context("16-byte alignment overflow")
}

/// Computes persistent 2-CTA/SM grid width for `v32_w4a4` GATEUP (`10` N-tiles per work item).
#[must_use]
pub(crate) fn w4a4_grid_gateup_for_sm(max_items: usize, sm_count: u32) -> u32 {
    let resident = (sm_count.max(1) as usize).saturating_mul(W4A4_CTAS_PER_SM as usize);
    let tasks = max_items.saturating_mul(10).max(1);
    tasks.min(resident).clamp(1, abi::GRID as usize) as u32
}

/// Computes persistent 2-CTA/SM grid width for `v32_w4a4` DOWN (`20` N-tiles per work item).
#[must_use]
pub(crate) fn w4a4_grid_down_for_sm(max_items: usize, sm_count: u32) -> u32 {
    let resident = (sm_count.max(1) as usize).saturating_mul(W4A4_CTAS_PER_SM as usize);
    let tasks = max_items.saturating_mul(20).max(1);
    tasks.min(resident).clamp(1, abi::GRID as usize) as u32
}

/// Computes grid width for `qwen4_nvfp4_quant_rows_v32_gate` (`6 * sm` on high-SM GPUs).
#[must_use]
pub(crate) fn w4a4_quant_grid_for_sm(rows: usize, sm_count: u32) -> u32 {
    let cap = if sm_count > HIGH_SM_THRESHOLD {
        sm_count.saturating_mul(W4A4_QUANT_CTAS_PER_SM)
    } else {
        W4A4_QUANT_GRID_CAP_GB10
    };
    (rows.min(u32::MAX as usize) as u32).clamp(1, cap.max(1))
}

/// Device buffers shared across `v32_w4a4` GATEUP and DOWN launches.
#[derive(Clone, Copy, Debug)]
pub(crate) struct W4a4WorkBuffers {
    pub expert_offsets: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub contract: DevicePtr,
    pub workspace: DevicePtr,
    pub status: DevicePtr,
}

/// Launches `qwen4_nvfp4_quant_rows_v32_gate` (single BF16 read -> NVFP4 `xq`/`xs`/`x_scale2`
/// + optional fused `shared_gate` dot product into `ss_out`).
///
/// # Errors
/// Returns an error if `rows` overflows `u32` or the GPU launch fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_w4a4_quant(
    gpu: &dyn GpuBackend,
    kernels: &W4a4V32Kernels,
    src_bf16: DevicePtr,
    xq: DevicePtr,
    xs: DevicePtr,
    x_scale2: DevicePtr,
    shared_gate: DevicePtr,
    ss_out: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let rows_u32 = u32::try_from(rows).context("v32_w4a4 quant rows exceeds u32")?;
    let grid = w4a4_quant_grid_for_sm(rows, atlas_core::device::active_sm_count());
    KernelLaunch::new(gpu, kernels.quant_gate)
        .grid([grid, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .arg_ptr(src_bf16)
        .arg_ptr(xq)
        .arg_ptr(xs)
        .arg_ptr(x_scale2)
        .arg_ptr(shared_gate)
        .arg_ptr(ss_out)
        .arg_u32(rows_u32)
        .launch(stream)
}

/// Launches persistent 2-CTA/SM `qwen4_moe_w4a4_gateup_v32*`.
///
/// When `oq_scale2_buf` is non-null in [`W4a4Mode::All`], GATEUP reduces exact per-`(orow, nt)`
/// dynamic `oq_scale2 = slice_amax / 2688.0f` across the 2 N-warps and writes it to `oq_scale2_buf`.
/// When `oq_scale2_buf` is [`DevicePtr::NULL`], falls back to scalar `oq_scale2`.
///
/// # Errors
/// Returns an error if `mode == W4a4Mode::Off`, scalar `oq_scale2` is non-positive/non-finite
/// when `oq_scale2_buf` is NULL in `W4a4Mode::All`, or the GPU launch fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_w4a4_gateup(
    gpu: &dyn GpuBackend,
    kernels: &W4a4V32Kernels,
    mode: W4a4Mode,
    layout: W4a4ScaleLayout,
    xq: DevicePtr,
    xs: DevicePtr,
    x_scale2: DevicePtr,
    gate_table: &ExpertPtrTable,
    up_table: &ExpertPtrTable,
    oq: DevicePtr,
    os: DevicePtr,
    oq_scale2_buf: DevicePtr,
    work: W4a4WorkBuffers,
    max_items: usize,
    oq_scale2: f32,
    stream: u64,
) -> Result<(u32, u32)> {
    ensure!(
        mode.is_enabled(),
        "launch_w4a4_gateup called with W4a4Mode::Off"
    );
    if mode.uses_w4a4_down() && oq_scale2_buf.is_null() {
        ensure!(
            oq_scale2.is_finite() && oq_scale2 > 0.0,
            "v32_w4a4 full mode requires dynamic oq_scale2_buf or finite positive oq_scale2, got {oq_scale2}"
        );
    }
    let (handle, smem) = kernels.gateup_for(mode, layout);
    let grid = w4a4_grid_gateup_for_sm(max_items, atlas_core::device::active_sm_count());
    KernelLaunch::new(gpu, handle)
        .grid([grid, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .shared_mem(smem)
        .arg_ptr(xq)
        .arg_ptr(xs)
        .arg_ptr(x_scale2)
        .arg_ptr(gate_table.packed_ptrs)
        .arg_ptr(gate_table.scale_ptrs)
        .arg_ptr(gate_table.scale2_vals)
        .arg_ptr(oq)
        .arg_ptr(os)
        .arg_ptr(up_table.packed_ptrs)
        .arg_ptr(up_table.scale_ptrs)
        .arg_ptr(up_table.scale2_vals)
        .arg_ptr(work.expert_offsets)
        .arg_ptr(work.sorted_token_ids)
        .arg_ptr(work.contract)
        .arg_ptr(work.workspace)
        .arg_ptr(work.status)
        .arg_u32(FLASH_NEXT_INTER)
        .arg_u32(FLASH_NEXT_HIDDEN)
        .arg_f32(oq_scale2)
        .arg_ptr(oq_scale2_buf)
        .launch(stream)?;
    Ok((grid, smem))
}

/// Launches persistent 2-CTA/SM `qwen4_moe_w4a4_down_v32*`.
///
/// Consumes per-`(orow, kh64)` dynamic `oq_scale2_buf` when non-null (applying overflow-safe
/// online accumulator rescaling per `kh64` slice), or falls back to scalar `oq_scale2` when NULL.
///
/// # Errors
/// Returns an error if `oq_scale2_buf` is NULL and `oq_scale2` is non-positive/non-finite,
/// or if the GPU launch fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_w4a4_down(
    gpu: &dyn GpuBackend,
    kernels: &W4a4V32Kernels,
    layout: W4a4ScaleLayout,
    oq: DevicePtr,
    os: DevicePtr,
    oq_scale2_buf: DevicePtr,
    down_table: &ExpertPtrTable,
    output: W4a4DownOutput,
    work: W4a4WorkBuffers,
    max_items: usize,
    oq_scale2: f32,
    stream: u64,
) -> Result<(u32, u32)> {
    if oq_scale2_buf.is_null() {
        ensure!(
            oq_scale2.is_finite() && oq_scale2 > 0.0,
            "v32_w4a4 DOWN requires dynamic oq_scale2_buf or finite positive oq_scale2, got {oq_scale2}"
        );
    }
    let (handle, smem) = kernels.down_for(output, layout);
    let grid = w4a4_grid_down_for_sm(max_items, atlas_core::device::active_sm_count());
    let launch = KernelLaunch::new(gpu, handle)
        .grid([grid, 1, 1])
        .block([W4A4_BLOCK, 1, 1])
        .shared_mem(smem)
        .arg_ptr(oq)
        .arg_ptr(os)
        .arg_ptr(down_table.packed_ptrs)
        .arg_ptr(down_table.scale_ptrs)
        .arg_ptr(down_table.scale2_vals);
    let launch = match output {
        W4a4DownOutput::Mxfp8Slots { slot_q, slot_s } => launch.arg_ptr(slot_q).arg_ptr(slot_s),
        W4a4DownOutput::Bf16 { c_out } => launch.arg_ptr(c_out),
    };
    launch
        .arg_ptr(work.expert_offsets)
        .arg_ptr(work.sorted_token_ids)
        .arg_ptr(work.contract)
        .arg_ptr(work.workspace)
        .arg_ptr(work.status)
        .arg_u32(FLASH_NEXT_HIDDEN)
        .arg_u32(FLASH_NEXT_INTER)
        .arg_f32(oq_scale2)
        .arg_ptr(oq_scale2_buf)
        .launch(stream)?;
    Ok((grid, smem))
}

static W4A4_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Emits the one-shot `MOE_PREFILL_W4A4_V32_ENGAGED` observability receipt.
pub(crate) fn log_w4a4_receipt(
    mode: W4a4Mode,
    rows: usize,
    grid_gu: u32,
    grid_dn: u32,
    smem_gu: u32,
    smem_dn: u32,
    layout: W4a4ScaleLayout,
) {
    if !W4A4_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "MOE_PREFILL_W4A4_V32_ENGAGED selector={W4A4_SELECTOR} mode={} rows={rows} \
             grid={grid_gu}/{grid_dn} block={W4A4_BLOCK} smem={smem_gu}/{smem_dn} ts4e={}",
            mode.as_str(),
            layout.is_ts4e(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn selector_defaults_off_and_rejects_invalid_values() {
        assert_eq!(W4A4_SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_W4A4");
        assert_eq!(parse_w4a4_mode(None), Ok(W4a4Mode::Off));
        assert_eq!(parse_w4a4_mode(Some("0")), Ok(W4a4Mode::Off));
        assert_eq!(parse_w4a4_mode(Some("off")), Ok(W4a4Mode::Off));
        assert_eq!(parse_w4a4_mode(Some("gateup")), Ok(W4a4Mode::GateUp));
        assert_eq!(parse_w4a4_mode(Some("1")), Ok(W4a4Mode::All));
        assert_eq!(parse_w4a4_mode(Some("all")), Ok(W4a4Mode::All));
        for bad in [
            "", "2", "auto", "AUTO", "true", "false", "on", "ALL", "down",
        ] {
            assert!(
                parse_w4a4_mode(Some(bad)).is_err(),
                "must reject invalid selector value {bad:?}"
            );
        }
    }

    #[test]
    fn eligibility_enforces_prerequisites_verify_guard_row_floor_and_boundary_layers() {
        let ready = |rows| W4a4Eligibility {
            rows,
            w4a8_is_v28: true,
            v28_m64_forced: false,
            has_corrections: false,
            l2_windowed: false,
            v29_m256_active: false,
            v30_2cta_active: false,
            verify_scope_active: false,
            kernels_loaded: true,
            layer_idx: Some(12),
            num_layers: Some(48),
        };
        assert!(!W4a4Mode::Off.admits(ready(4096)));
        for mode in [W4a4Mode::GateUp, W4a4Mode::All] {
            // Prefill row floor: rows <= 256 (including speculative verify frames) never enter W4A4.
            for rows in [0usize, 1, 2, 16, 128, 256] {
                assert!(
                    !mode.admits(ready(rows)),
                    "mode={mode:?} rows={rows} must be rejected (<= 256)"
                );
            }
            for rows in [257usize, 512, 4096, 16_384] {
                assert!(mode.admits(ready(rows)), "mode={mode:?} rows={rows}");
            }
            // Speculative verify scope is strictly blocked regardless of row count.
            assert!(
                !mode.admits(W4a4Eligibility {
                    verify_scope_active: true,
                    ..ready(4096)
                }),
                "verify_scope_active must reject W4A4"
            );
            // Default boundary-layer protection: 0..2 and 45..47 stay on exact W4A8.
            for protected in [0usize, 1, 45, 46, 47] {
                assert!(
                    !mode.admits(W4a4Eligibility {
                        layer_idx: Some(protected),
                        num_layers: Some(48),
                        ..ready(4096)
                    }),
                    "layer {protected}/48 must be protected by default"
                );
            }
            for interior in [2usize, 3, 24, 44] {
                assert!(
                    mode.admits(W4a4Eligibility {
                        layer_idx: Some(interior),
                        num_layers: Some(48),
                        ..ready(4096)
                    }),
                    "interior layer {interior}/48 must be admitted"
                );
            }
            // ATLAS_QWEN4_PREFILL_MOE_W4A4_PROTECT_LAYERS=0 disables boundary protection.
            for layer in [0usize, 1, 45, 46, 47] {
                assert!(
                    mode.admits_with_boundary(
                        W4a4Eligibility {
                            layer_idx: Some(layer),
                            num_layers: Some(48),
                            ..ready(4096)
                        },
                        W4a4ProtectedBoundary::NONE,
                    ),
                    "override =0 must admit layer {layer}/48"
                );
            }
            // Missing layer info remains admissible when all other checks pass.
            assert!(mode.admits(W4a4Eligibility {
                layer_idx: None,
                num_layers: None,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                w4a8_is_v28: false,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                v28_m64_forced: true,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                has_corrections: true,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                l2_windowed: true,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                v29_m256_active: true,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                v30_2cta_active: true,
                ..ready(4096)
            }));
            assert!(!mode.admits(W4a4Eligibility {
                kernels_loaded: false,
                ..ready(4096)
            }));
        }
    }

    #[test]
    fn protected_boundary_parser_defaults_and_validates() {
        assert_eq!(
            parse_w4a4_protected_boundary(None),
            Ok(W4a4ProtectedBoundary { first: 2, last: 3 })
        );
        assert_eq!(
            parse_w4a4_protected_boundary(Some("")),
            Ok(W4a4ProtectedBoundary { first: 2, last: 3 })
        );
        assert_eq!(
            parse_w4a4_protected_boundary(Some("0")),
            Ok(W4a4ProtectedBoundary::NONE)
        );
        assert_eq!(
            parse_w4a4_protected_boundary(Some("4")),
            Ok(W4a4ProtectedBoundary { first: 4, last: 4 })
        );
        assert_eq!(
            parse_w4a4_protected_boundary(Some("1, 4")),
            Ok(W4a4ProtectedBoundary { first: 1, last: 4 })
        );
        for bad in ["-1", "off", "true", "2,a", "1,2,3"] {
            assert!(
                parse_w4a4_protected_boundary(Some(bad)).is_err(),
                "must reject {bad:?}"
            );
        }
    }

    #[test]
    fn scratch_layout_fits_existing_arenas_with_zero_extra_alloc() {
        for rows in [2usize, 64, 512, 4096, 16_384, 32_768] {
            let expanded = rows * FLASH_NEXT_TOPK;
            let up_cap = rows * 2_640; // existing W4A8 expert_up_out minimum
            let gate_cap = expanded * 1_280; // existing compact expert_gate_out minimum

            let hybrid = W4a4ScratchLayout::for_rows(rows, W4a4Mode::GateUp, up_cap, gate_cap)
                .expect("hybrid layout fits existing arenas");
            assert_eq!(hybrid.xq_offset, 0);
            assert_eq!(hybrid.xs_offset, rows * 1_280);
            assert_eq!(hybrid.x_scale2_offset % 16, 0);
            assert_eq!(hybrid.up_used_bytes, rows * 1_444);
            assert!(hybrid.up_used_bytes <= up_cap);
            assert_eq!(hybrid.oq_offset, 0);
            assert_eq!(hybrid.os_offset, expanded * 640);
            assert_eq!(hybrid.oq_scale2_offset, None);
            assert_eq!(hybrid.gate_used_bytes, expanded * 660);
            assert!(hybrid.gate_used_bytes <= gate_cap);

            let full = W4a4ScratchLayout::for_rows(rows, W4a4Mode::All, up_cap, gate_cap)
                .expect("full W4A4 layout fits existing arenas");
            assert_eq!(full.up_used_bytes, rows * 1_444);
            assert_eq!(full.oq_offset, 0);
            assert_eq!(full.os_offset, expanded * 320);
            let s2_off = full
                .oq_scale2_offset
                .expect("full mode allocates dynamic oq_scale2 inside expert_gate_out");
            assert_eq!(s2_off, expanded * 360);
            assert_eq!(s2_off % 16, 0);
            assert_eq!(full.gate_used_bytes, expanded * 400);
            assert!(full.gate_used_bytes <= gate_cap);
        }

        // Undersized arenas fail closed.
        assert!(W4a4ScratchLayout::for_rows(0, W4a4Mode::All, 1 << 20, 1 << 20).is_err());
        assert!(W4a4ScratchLayout::for_rows(16, W4a4Mode::All, 16 * 1_443, 160 * 400).is_err());
        assert!(W4a4ScratchLayout::for_rows(16, W4a4Mode::All, 16 * 1_444, 160 * 399).is_err());
        assert!(W4a4ScratchLayout::for_rows(16, W4a4Mode::GateUp, 16 * 1_444, 160 * 659).is_err());
    }

    #[test]
    fn grid_sizing_targets_two_ctas_per_sm_and_scales_cleanly() {
        // GB10 (48 SMs) -> 96 resident CTAs; SM120 (188 SMs) -> 376 resident CTAs.
        assert_eq!(w4a4_grid_gateup_for_sm(100, 48), 96);
        assert_eq!(w4a4_grid_down_for_sm(100, 48), 96);
        assert_eq!(w4a4_grid_gateup_for_sm(100, 188), 376);
        assert_eq!(w4a4_grid_down_for_sm(100, 188), 376);
        // Tiny worklists don't over-launch idle CTAs beyond task_count.
        assert_eq!(w4a4_grid_gateup_for_sm(2, 188), 20);
        assert_eq!(w4a4_grid_down_for_sm(2, 188), 40);
        // Quantizer scales to 6 CTAs/SM on high-SM GPUs while preserving GB10 cap.
        assert_eq!(w4a4_quant_grid_for_sm(16_384, 48), 1024);
        assert_eq!(w4a4_quant_grid_for_sm(16_384, 188), 1128);
        assert_eq!(w4a4_quant_grid_for_sm(64, 188), 64);
    }

    #[test]
    fn mock_kernel_load_and_zero_alloc_launches() {
        let gpu = MockGpuBackend::new();
        let kernels = load_w4a4_v32_kernels(&gpu).expect("mock loads all 9 v32_w4a4 kernels");

        let rows = 4096usize;
        let layout = W4a4ScratchLayout::for_rows(
            rows,
            W4a4Mode::All,
            rows * 2_640,
            rows * FLASH_NEXT_TOPK * 1_280,
        )
        .expect("layout");
        let (xq, xs, x_scale2) = layout.input_ptrs(DevicePtr(0x1000_0000));
        let (oq, os, oq_scale2_buf) = layout.intermediate_ptrs(DevicePtr(0x2000_0000));
        assert!(!oq_scale2_buf.is_null());
        let table = ExpertPtrTable {
            packed_ptrs: DevicePtr(0x3000_0000),
            scale_ptrs: DevicePtr(0x3100_0000),
            scale2_vals: DevicePtr(0x3200_0000),
        };
        let work = W4a4WorkBuffers {
            expert_offsets: DevicePtr(0x4000_0000),
            sorted_token_ids: DevicePtr(0x4100_0000),
            contract: DevicePtr(0x4200_0000),
            workspace: DevicePtr(0x4300_0000),
            status: DevicePtr(0x4400_0000),
        };

        launch_w4a4_quant(
            &gpu,
            &kernels,
            DevicePtr(0x0100_0000),
            xq,
            xs,
            x_scale2,
            DevicePtr(0x0200_0000),
            DevicePtr(0x0300_0000),
            rows,
            0,
        )
        .expect("quant launch");
        let (grid_gu, smem_gu) = launch_w4a4_gateup(
            &gpu,
            &kernels,
            W4a4Mode::All,
            W4a4ScaleLayout::Ts4e128,
            xq,
            xs,
            x_scale2,
            &table,
            &table,
            oq,
            os,
            oq_scale2_buf,
            work,
            512,
            DEFAULT_OQ_SCALE2,
            0,
        )
        .expect("gateup launch");
        let (grid_dn, smem_dn) = launch_w4a4_down(
            &gpu,
            &kernels,
            W4a4ScaleLayout::Ts4e128,
            oq,
            os,
            oq_scale2_buf,
            &table,
            W4a4DownOutput::Mxfp8Slots {
                slot_q: DevicePtr(0x5000_0000),
                slot_s: DevicePtr(0x5100_0000),
            },
            work,
            512,
            DEFAULT_OQ_SCALE2,
            0,
        )
        .expect("down launch");
        assert!(grid_gu >= 1 && grid_dn >= 1);
        assert_eq!(smem_gu, W4A4_TS4E_GATEUP_SMEM_BYTES);
        assert_eq!(smem_dn, W4A4_TS4E_DOWN_SMEM_BYTES);
        assert!(smem_gu <= W4A4_MAX_2CTA_SMEM_BYTES);
        assert!(smem_dn <= W4A4_MAX_2CTA_SMEM_BYTES);

        // Non-positive scalar fallback fails closed when oq_scale2_buf is NULL.
        assert!(
            launch_w4a4_gateup(
                &gpu,
                &kernels,
                W4a4Mode::All,
                W4a4ScaleLayout::Flat,
                xq,
                xs,
                x_scale2,
                &table,
                &table,
                oq,
                os,
                DevicePtr::NULL,
                work,
                512,
                0.0,
                0,
            )
            .is_err()
        );
        assert!(
            launch_w4a4_down(
                &gpu,
                &kernels,
                W4a4ScaleLayout::Flat,
                oq,
                os,
                DevicePtr::NULL,
                &table,
                W4a4DownOutput::Bf16 {
                    c_out: DevicePtr(0x5000_0000),
                },
                work,
                512,
                f32::NAN,
                0,
            )
            .is_err()
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_parity_tests {
    use super::*;
    use crate::layers::moe::helpers_a::ts4e128_encode_host;
    use crate::layers::moe::qwen4_compact_ops::{
        V31_WS_BLOCK, V31_WS_FLAT_DOWN_SMEM_BYTES, V31_WS_FLAT_GATEUP_SMEM_BYTES,
        load_v31_ws_kernels,
    };
    use crate::layers::moe::qwen4_prefill_compact::{SlotStorage, slot_layout};
    use half::bf16;
    use spark_runtime::cuda_backend::AtlasCudaBackend;

    struct DevBuf<'a> {
        gpu: &'a dyn GpuBackend,
        ptr: DevicePtr,
        bytes: usize,
    }

    impl<'a> DevBuf<'a> {
        fn alloc(gpu: &'a dyn GpuBackend, bytes: usize) -> Self {
            let alloc_bytes = bytes.max(16);
            let ptr = gpu.alloc(alloc_bytes).expect("GPU alloc");
            Self {
                gpu,
                ptr,
                bytes: alloc_bytes,
            }
        }

        fn upload(gpu: &'a dyn GpuBackend, data: &[u8]) -> Self {
            let buf = Self::alloc(gpu, data.len());
            if !data.is_empty() {
                gpu.copy_h2d(data, buf.ptr).expect("GPU copy_h2d");
            }
            buf
        }

        fn zeroed(gpu: &'a dyn GpuBackend, bytes: usize) -> Self {
            let buf = Self::alloc(gpu, bytes);
            let stream = gpu.default_stream();
            gpu.memset_async(buf.ptr, 0, buf.bytes, stream)
                .expect("GPU memset");
            buf
        }

        fn ptr(&self) -> DevicePtr {
            self.ptr
        }

        fn read_bytes(&self, len: usize) -> Vec<u8> {
            let mut out = vec![0u8; len];
            self.gpu
                .synchronize(self.gpu.default_stream())
                .expect("GPU sync");
            self.gpu.copy_d2h(self.ptr, &mut out).expect("GPU copy_d2h");
            out
        }
    }

    impl Drop for DevBuf<'_> {
        fn drop(&mut self) {
            let _ = self.gpu.free(self.ptr);
        }
    }

    fn try_gb10_flash_next() -> Option<AtlasCudaBackend> {
        let ptx = atlas_kernels::ptx_for_model("flash-next")?;
        AtlasCudaBackend::new(0, &ptx.modules).ok()
    }

    struct Rng(u64);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }

        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            let u = (self.next_u32() >> 8) as f32 * (1.0 / ((1u32 << 24) as f32));
            lo + (hi - lo) * u
        }
    }

    const E2M1_TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    fn e2m1_to_f32(nib: u8) -> f32 {
        let mag = E2M1_TABLE[(nib & 7) as usize];
        if (nib & 8) != 0 { -mag } else { mag }
    }

    fn e4m3_to_f32(byte: u8) -> f32 {
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 3) & 0x0f;
        let mant = byte & 0x07;
        if exp == 0 {
            sign * (f32::from(mant) * 2f32.powi(-9))
        } else if exp == 15 && mant == 7 {
            f32::NAN
        } else {
            sign * ((1.0 + f32::from(mant) * 0.125) * 2f32.powi(i32::from(exp) - 7))
        }
    }

    fn ue8m0_to_f32(byte: u8) -> f32 {
        if byte == 255 {
            f32::NAN
        } else {
            f32::from_bits(u32::from(byte) << 23)
        }
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn f32_to_e4m3_rn(x: f32) -> u8 {
        if !x.is_finite() {
            return 0x7f;
        }
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0u8 };
        let ax = x.abs().min(448.0);
        // Exact tie-to-even lookup over all 127 finite non-negative E4M3 magnitudes (0..=0x7e).
        let mut best_code = 0u8;
        let mut best_err = f32::INFINITY;
        for code in 0u8..=0x7eu8 {
            let val = e4m3_to_f32(code);
            let err = (ax - val).abs();
            if err < best_err || (err == best_err && (code & 1) == 0 && (best_code & 1) != 0) {
                best_err = err;
                best_code = code;
            }
        }
        sign | best_code
    }

    fn fake_quant_mxfp8_cpu(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let g_per_row = cols / 32;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for g in 0..g_per_row {
                let base = r * cols + g * 32;
                let amax = src[base..base + 32]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                if amax == 0.0 {
                    continue;
                }
                let scaled = amax * (1.0 / 448.0);
                // Match GPU frexpf -> ceil log2 power-of-two UE8M0 scale.
                let bits = scaled.to_bits();
                let exp_unbiased = ((bits >> 23) & 0xff) as i32 - 127;
                let mant = bits & 0x7f_ffff;
                let k = (if mant == 0 {
                    exp_unbiased
                } else {
                    exp_unbiased + 1
                })
                .clamp(-127, 127);
                let sc = f32::from_bits(((k + 127) as u32) << 23);
                let inv = f32::from_bits(((127 - k) as u32) << 23);
                for j in 0..32 {
                    let q = f32_to_e4m3_rn(src[base + j] * inv);
                    out[base + j] = e4m3_to_f32(q) * sc;
                }
            }
        }
        out
    }

    fn dequant_mxfp8_matrix(q: &[u8], s: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        let g_per_row = cols / 32;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for g in 0..g_per_row {
                let sc = ue8m0_to_f32(s[r * g_per_row + g]);
                for j in 0..32 {
                    let idx = r * cols + g * 32 + j;
                    out[idx] = e4m3_to_f32(q[idx]) * sc;
                }
            }
        }
        out
    }

    fn dequant_nvfp4_row_dynamic(
        oq: &[u8],
        os: &[u8],
        oq_s2: &[f32],
        row: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        let half = cols / 2;
        let grp = cols / 16;
        let slices = cols / 64;
        for c in 0..cols {
            let b = oq[row * half + (c >> 1)];
            let nib = if (c & 1) == 0 { b & 0x0f } else { b >> 4 };
            let sf = e4m3_to_f32(os[row * grp + (c >> 4)] & 0x7f);
            let s2 = oq_s2[row * slices + (c >> 6)];
            out[c] = e2m1_to_f32(nib) * sf * s2;
        }
        out
    }

    fn dequant_nvfp4_row_uniform(
        xq: &[u8],
        xs: &[u8],
        x_s2: &[f32],
        row: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        let half = cols / 2;
        let grp = cols / 16;
        let s2 = x_s2[row];
        for c in 0..cols {
            let b = xq[row * half + (c >> 1)];
            let nib = if (c & 1) == 0 { b & 0x0f } else { b >> 4 };
            let sf = e4m3_to_f32(xs[row * grp + (c >> 4)] & 0x7f);
            out[c] = e2m1_to_f32(nib) * sf * s2;
        }
        out
    }

    fn dequant_nvfp4_weight_row_major(
        packed_rm: &[u8],
        scales_rm: &[u8],
        scale2: f32,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let half = k / 2;
        let grp = k / 16;
        let mut out = vec![0.0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                let b = packed_rm[r * half + (c >> 1)];
                let nib = if (c & 1) == 0 { b & 0x0f } else { b >> 4 };
                // Matches pack_sf_stage sign mask (`sc &= 0x7f7f7f7fu`).
                let sf = e4m3_to_f32(scales_rm[r * grp + (c >> 4)] & 0x7f);
                out[r * k + c] = e2m1_to_f32(nib) * sf * scale2;
            }
        }
        out
    }

    fn transpose_u8(src_rm: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut dst = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src_rm[r * cols + c];
            }
        }
        dst
    }

    fn parity_metrics(cand: &[f32], reference: &[f32]) -> (f64, f32, usize) {
        assert_eq!(cand.len(), reference.len());
        let mut dot = 0.0f64;
        let mut n_c = 0.0f64;
        let mut n_r = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut non_finite = 0usize;
        for (&c, &r) in cand.iter().zip(reference) {
            if !c.is_finite() || !r.is_finite() {
                non_finite += 1;
                continue;
            }
            dot += f64::from(c) * f64::from(r);
            n_c += f64::from(c) * f64::from(c);
            n_r += f64::from(r) * f64::from(r);
            max_abs = max_abs.max((c - r).abs());
        }
        let cos = if n_c == 0.0 && n_r == 0.0 {
            1.0
        } else {
            dot / (n_c.sqrt() * n_r.sqrt()).max(1e-30)
        };
        (cos, max_abs, non_finite)
    }

    struct WeightSetHost {
        packed_rm: Vec<u8>,
        scales_rm: Vec<u8>,
        packed_t: Vec<u8>,
        scales_flat_t: Vec<u8>,
        scales_ts4e: Vec<u8>,
        scale2: f32,
    }

    fn make_nvfp4_weights(
        rng: &mut Rng,
        n: usize,
        k: usize,
        scale2: f32,
        row_scale_fn: impl Fn(usize) -> u8,
        set_sign_bits: bool,
    ) -> WeightSetHost {
        let half = k / 2;
        let grp = k / 16;
        let mut packed_rm = vec![0u8; n * half];
        for byte in &mut packed_rm {
            let lo = (rng.next_u32() & 0x0f) as u8;
            let hi = (rng.next_u32() & 0x0f) as u8;
            *byte = lo | (hi << 4);
        }
        let mut scales_rm = vec![0u8; n * grp];
        for r in 0..n {
            let base = row_scale_fn(r) & 0x7e;
            for g in 0..grp {
                let mut sb = base;
                if set_sign_bits && ((r + g) & 3) == 0 {
                    // Inject stray sign bit (bit 7) to verify UE4M3 sign-bit masking.
                    sb |= 0x80;
                }
                scales_rm[r * grp + g] = sb;
            }
        }
        let packed_t = transpose_u8(&packed_rm, n, half);
        let scales_flat_t = transpose_u8(&scales_rm, n, grp);
        let (scales_ts4e, _) = ts4e128_encode_host(&scales_rm, k, n).expect("ts4e encode");
        WeightSetHost {
            packed_rm,
            scales_rm,
            packed_t,
            scales_flat_t,
            scales_ts4e,
            scale2,
        }
    }

    struct SyntheticExpertTables<'a> {
        _packed_bufs: Vec<DevBuf<'a>>,
        _scale_flat_bufs: Vec<DevBuf<'a>>,
        _scale_ts4e_bufs: Vec<DevBuf<'a>>,
        _packed_ptrs: DevBuf<'a>,
        _scale_flat_ptrs: DevBuf<'a>,
        _scale_ts4e_ptrs: DevBuf<'a>,
        _scale2_vals: DevBuf<'a>,
        flat: ExpertPtrTable,
        ts4e: ExpertPtrTable,
    }

    fn upload_expert_tables<'a>(
        gpu: &'a dyn GpuBackend,
        unique: &[WeightSetHost],
    ) -> SyntheticExpertTables<'a> {
        let mut packed_bufs = Vec::with_capacity(unique.len());
        let mut scale_flat_bufs = Vec::with_capacity(unique.len());
        let mut scale_ts4e_bufs = Vec::with_capacity(unique.len());
        for w in unique {
            packed_bufs.push(DevBuf::upload(gpu, &w.packed_t));
            scale_flat_bufs.push(DevBuf::upload(gpu, &w.scales_flat_t));
            scale_ts4e_bufs.push(DevBuf::upload(gpu, &w.scales_ts4e));
        }
        let mut packed_ptr_bytes = Vec::with_capacity(512 * 8);
        let mut flat_ptr_bytes = Vec::with_capacity(512 * 8);
        let mut ts4e_ptr_bytes = Vec::with_capacity(512 * 8);
        let mut scale2_bytes = Vec::with_capacity(512 * 4);
        for e in 0..512usize {
            let u = e % unique.len();
            packed_ptr_bytes.extend_from_slice(&packed_bufs[u].ptr().0.to_le_bytes());
            flat_ptr_bytes.extend_from_slice(&scale_flat_bufs[u].ptr().0.to_le_bytes());
            ts4e_ptr_bytes.extend_from_slice(&scale_ts4e_bufs[u].ptr().0.to_le_bytes());
            scale2_bytes.extend_from_slice(&unique[u].scale2.to_le_bytes());
        }
        let packed_ptrs = DevBuf::upload(gpu, &packed_ptr_bytes);
        let scale_flat_ptrs = DevBuf::upload(gpu, &flat_ptr_bytes);
        let scale_ts4e_ptrs = DevBuf::upload(gpu, &ts4e_ptr_bytes);
        let scale2_vals = DevBuf::upload(gpu, &scale2_bytes);
        let flat = ExpertPtrTable {
            packed_ptrs: packed_ptrs.ptr(),
            scale_ptrs: scale_flat_ptrs.ptr(),
            scale2_vals: scale2_vals.ptr(),
        };
        let ts4e = ExpertPtrTable {
            packed_ptrs: packed_ptrs.ptr(),
            scale_ptrs: scale_ts4e_ptrs.ptr(),
            scale2_vals: scale2_vals.ptr(),
        };
        SyntheticExpertTables {
            _packed_bufs: packed_bufs,
            _scale_flat_bufs: scale_flat_bufs,
            _scale_ts4e_bufs: scale_ts4e_bufs,
            _packed_ptrs: packed_ptrs,
            _scale_flat_ptrs: scale_flat_ptrs,
            _scale_ts4e_ptrs: scale_ts4e_ptrs,
            _scale2_vals: scale2_vals,
            flat,
            ts4e,
        }
    }

    fn build_routing(rows: usize, active_experts: usize) -> (Vec<i32>, Vec<i32>, Vec<usize>) {
        assert!((FLASH_NEXT_TOPK..=512).contains(&active_experts));
        let mut per_expert: Vec<Vec<i32>> = vec![Vec::new(); 512];
        for t in 0..rows {
            let base = (t * 7) % active_experts;
            for slot in 0..FLASH_NEXT_TOPK {
                let e = (base + slot) % active_experts;
                per_expert[e].push(t as i32);
            }
        }
        let mut offsets = Vec::with_capacity(513);
        let mut sorted_ids = Vec::with_capacity(rows * FLASH_NEXT_TOPK);
        let mut row_expert = Vec::with_capacity(rows * FLASH_NEXT_TOPK);
        offsets.push(0i32);
        for (e, toks) in per_expert.iter().enumerate() {
            for &tok in toks {
                sorted_ids.push(tok);
                row_expert.push(e);
            }
            offsets.push(sorted_ids.len() as i32);
        }
        (offsets, sorted_ids, row_expert)
    }

    fn run_plan(
        gpu: &dyn GpuBackend,
        plan_k: KernelHandle,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        contract: DevicePtr,
        workspace: DevicePtr,
        status: DevicePtr,
        rows: usize,
        stream: u64,
    ) {
        let c_bytes = abi::contract(rows).expect("contract");
        let pending = (-1i32).to_le_bytes();
        let zero = 0u32.to_le_bytes();
        gpu.memset_async(workspace, 0, abi::ARENA_BYTES, stream)
            .expect("zero workspace");
        gpu.copy_h2d(&c_bytes, contract).expect("h2d contract");
        gpu.copy_h2d(&pending, status).expect("h2d status");
        gpu.copy_h2d(&zero, workspace.offset(36))
            .expect("h2d next_task_dn");
        KernelLaunch::new(gpu, plan_k)
            .grid([1, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(offsets)
            .arg_ptr(sorted_ids)
            .arg_ptr(contract)
            .arg_ptr(workspace)
            .arg_ptr(status)
            .launch(stream)
            .expect("plan launch");
    }

    fn reset_work_counters(gpu: &dyn GpuBackend, workspace: DevicePtr, stream: u64) {
        let zeros = [0u8; 8];
        gpu.synchronize(stream).expect("sync before reset");
        gpu.copy_h2d(&zeros, workspace.offset(32))
            .expect("reset next_task_gu/dn");
    }

    #[test]
    fn gpu_v32_w4a4_smem_and_quant_gate_parity() {
        let Some(gpu) = try_gb10_flash_next() else {
            return;
        };
        let stream = gpu.default_stream();
        let kernels = load_w4a4_v32_kernels(&gpu).expect("load v32_w4a4 kernels");

        // 1. Device-reported SMEM sizes match all 4 Rust constants byte-for-byte.
        let smem_k = gpu
            .kernel(W4A4_MODULE, "qwen4_moe_w4a4_v32_smem_bytes")
            .expect("smem query kernel");
        let smem_out = DevBuf::zeroed(&gpu, 16);
        KernelLaunch::new(&gpu, smem_k)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(smem_out.ptr())
            .launch(stream)
            .expect("smem query launch");
        let smem_bytes = smem_out.read_bytes(16);
        let got_smem: Vec<u32> = smem_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got_smem,
            vec![
                W4A4_FLAT_GATEUP_SMEM_BYTES,
                W4A4_FLAT_DOWN_SMEM_BYTES,
                W4A4_TS4E_GATEUP_SMEM_BYTES,
                W4A4_TS4E_DOWN_SMEM_BYTES,
            ],
            "device SMEM bytes must match Rust launcher constants"
        );

        // 2. Multi-iteration quantizer test (rows=4003 > grid=1024) covering:
        //    - SMEM race guard (`ss_out != NULL && shared_gate == NULL` across 4 loop iters/CTA)
        //    - Unbiased round-to-nearest UE4M3 scales (zero sign bit, <= 0.5 ULP vs exact sf)
        //    - Fused `shared_gate` dot product accuracy.
        let rows = 4003usize;
        let h = FLASH_NEXT_HIDDEN as usize;
        let mut rng = Rng(0x5eed_cafe_1234_5678);
        let mut x_host = vec![0.0f32; rows * h];
        for r in 0..rows {
            // Mix wildly varying row magnitudes to exercise UE4M3 ceiling & subnormals.
            let scale = match r % 5 {
                0 => 0.01,
                1 => 1.5,
                2 => 37.0,
                3 => 450.0,
                _ => 0.25,
            };
            for c in 0..h {
                let raw = rng.uniform(-scale, scale);
                x_host[r * h + c] = bf16::from_f32(raw).to_f32();
            }
        }
        let sg_host: Vec<f32> = (0..h)
            .map(|_| bf16::from_f32(rng.uniform(-0.08, 0.08)).to_f32())
            .collect();
        let x_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&x_host));
        let sg_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&sg_host));
        let xq_dev = DevBuf::zeroed(&gpu, rows * XQ_NVFP4_BYTES_PER_ROW);
        let xs_dev = DevBuf::zeroed(&gpu, rows * XS_NVFP4_BYTES_PER_ROW);
        let xs2_dev = DevBuf::zeroed(&gpu, rows * X_SCALE2_BYTES_PER_ROW);
        let ss_dev = DevBuf::zeroed(&gpu, rows * 4);

        // Case A: ss_out != NULL, shared_gate == NULL (exercises multi-iter SMEM barrier).
        launch_w4a4_quant(
            &gpu,
            &kernels,
            x_dev.ptr(),
            xq_dev.ptr(),
            xs_dev.ptr(),
            xs2_dev.ptr(),
            DevicePtr::NULL,
            ss_dev.ptr(),
            rows,
            stream,
        )
        .expect("quant launch (null shared_gate)");
        let xs_bytes = xs_dev.read_bytes(rows * XS_NVFP4_BYTES_PER_ROW);
        let xs2_vals = f32_bytes_to_vec(&xs2_dev.read_bytes(rows * 4));
        let ss_ones = f32_bytes_to_vec(&ss_dev.read_bytes(rows * 4));
        for (r, &ss) in ss_ones.iter().enumerate() {
            assert_eq!(ss, 1.0, "row {r}: ss_out must be 1.0 when shared_gate=NULL");
        }
        // Check exact row_amax -> x_scale2 AND unbiased round-to-nearest UE4M3 scale accuracy.
        let mut max_s2_rel = 0.0f32;
        for r in 0..rows {
            let row_slice = &x_host[r * h..(r + 1) * h];
            let row_amax = row_slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let expected_s2 = if row_amax > 0.0 {
                row_amax / 2688.0
            } else {
                0.0
            };
            let got_s2 = xs2_vals[r];
            assert!(got_s2.is_finite() && got_s2 >= 0.0);
            if expected_s2 > 0.0 {
                let rel = (got_s2 - expected_s2).abs() / expected_s2;
                max_s2_rel = max_s2_rel.max(rel);
                assert!(
                    rel < 1e-5,
                    "row {r}: x_scale2 mismatch got={got_s2} exp={expected_s2}"
                );
            }
            for g in 0..(h / 16) {
                let blk_amax = row_slice[g * 16..(g + 1) * 16]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                let sf_b = xs_bytes[r * (h / 16) + g];
                assert_eq!(sf_b & 0x80, 0, "UE4M3 scale sign bit must be 0");
                let cap = e4m3_to_f32(sf_b) * got_s2 * 6.0;
                // Unbiased round-to-nearest UE4M3 (3 mantissa bits -> max +/-6.25% rounding
                // around exact sf_f, zero systematic positive bias).
                if blk_amax > 0.0 && got_s2 > 0.0 {
                    let lo = blk_amax * 0.935 - 1e-5;
                    let hi = blk_amax * 1.07 + 1e-5;
                    assert!(
                        cap >= lo && cap <= hi,
                        "RN UE4M3 scale out of bounds at row={r} grp={g}: cap={cap} blk_amax={blk_amax}"
                    );
                }
            }
        }

        // Case B: fused shared_gate dot product.
        launch_w4a4_quant(
            &gpu,
            &kernels,
            x_dev.ptr(),
            xq_dev.ptr(),
            xs_dev.ptr(),
            xs2_dev.ptr(),
            sg_dev.ptr(),
            ss_dev.ptr(),
            rows,
            stream,
        )
        .expect("quant launch (with shared_gate)");
        let ss_got = f32_bytes_to_vec(&ss_dev.read_bytes(rows * 4));
        let ss_ref: Vec<f32> = (0..rows)
            .map(|r| {
                let dot: f32 = x_host[r * h..(r + 1) * h]
                    .iter()
                    .zip(&sg_host)
                    .map(|(&a, &b)| a * b)
                    .sum();
                1.0 / (1.0 + (-dot).exp())
            })
            .collect();
        let (ss_cos, ss_max_abs, ss_nan) = parity_metrics(&ss_got, &ss_ref);
        eprintln!(
            "v32_w4a4 quant_gate rows={rows}: max_s2_rel={max_s2_rel:.3e} ss_cos={ss_cos:.6} ss_max_abs={ss_max_abs:.3e} nan={ss_nan}"
        );
        assert_eq!(ss_nan, 0);
        assert!(ss_cos >= 0.99999, "ss_cos={ss_cos}");
        assert!(ss_max_abs < 1e-3, "ss_max_abs={ss_max_abs}");
    }

    #[test]
    fn gpu_v32_w4a4_exact_fake_quant_and_bimodal_parity() {
        let Some(gpu) = try_gb10_flash_next() else {
            return;
        };
        let stream = gpu.default_stream();
        let w4a4_k = load_w4a4_v32_kernels(&gpu).expect("load v32_w4a4 kernels");
        let v31_k = load_v31_ws_kernels(&gpu).expect("load v31_ws kernels");
        let plan_k = gpu
            .kernel(
                "qwen4_moe_compact_t_k32_v7",
                "qwen4_moe_compact_t_plan_k32_v7",
            )
            .expect("load compact plan kernel");
        let xq_mxfp8_k = gpu
            .kernel(
                "qwen4_moe_compact_t_k32_v28",
                "qwen4_mxfp8_quant_rows_v16q4_gate",
            )
            .expect("load mxfp8 quant kernel");

        let h = FLASH_NEXT_HIDDEN as usize;
        let inter = FLASH_NEXT_INTER as usize;
        let mut rng = Rng(0x1337_beef_cafe_4242);

        // =====================================================================
        // PART 1: Exact fake-quant parity (rows = 31, Flat & Ts4e128, with
        //         stray sign bits on Flat scales to verify UE4M3 sign-bit masking).
        // =====================================================================
        {
            let rows = 31usize;
            let pairs = rows * FLASH_NEXT_TOPK;
            let unique_gate: Vec<WeightSetHost> = (0..4)
                .map(|e| make_nvfp4_weights(&mut rng, inter, h, 0.05, |_| 0x38, e == 3))
                .collect();
            let unique_up: Vec<WeightSetHost> = (0..4)
                .map(|e| make_nvfp4_weights(&mut rng, inter, h, 0.05, |_| 0x38, e == 2))
                .collect();
            let unique_down: Vec<WeightSetHost> = (0..4)
                .map(|e| make_nvfp4_weights(&mut rng, h, inter, 0.05, |_| 0x38, e == 1))
                .collect();
            let gate_tbl = upload_expert_tables(&gpu, &unique_gate);
            let up_tbl = upload_expert_tables(&gpu, &unique_up);
            let down_tbl = upload_expert_tables(&gpu, &unique_down);

            let mut x_host = vec![0.0f32; rows * h];
            for v in &mut x_host {
                *v = bf16::from_f32(rng.uniform(-1.0, 1.0)).to_f32();
            }
            let x_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&x_host));
            let (offsets_h, sorted_h, row_expert) = build_routing(rows, 12);
            let offsets_bytes: Vec<u8> = offsets_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sorted_bytes: Vec<u8> = sorted_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let offsets_dev = DevBuf::upload(&gpu, &offsets_bytes);
            let sorted_dev = DevBuf::upload(&gpu, &sorted_bytes);
            let contract_dev = DevBuf::zeroed(&gpu, abi::CONTRACT_BYTES);
            let workspace_dev = DevBuf::zeroed(&gpu, abi::ARENA_BYTES);
            let status_dev = DevBuf::zeroed(&gpu, 4);
            let work = W4a4WorkBuffers {
                expert_offsets: offsets_dev.ptr(),
                sorted_token_ids: sorted_dev.ptr(),
                contract: contract_dev.ptr(),
                workspace: workspace_dev.ptr(),
                status: status_dev.ptr(),
            };
            let max_items = pairs.min(pairs.div_ceil(128) + 512);
            run_plan(
                &gpu,
                plan_k,
                offsets_dev.ptr(),
                sorted_dev.ptr(),
                contract_dev.ptr(),
                workspace_dev.ptr(),
                status_dev.ptr(),
                rows,
                stream,
            );

            let layout_all =
                W4a4ScratchLayout::for_rows(rows, W4a4Mode::All, rows * 2_640, pairs * 1_280)
                    .expect("layout all");
            let layout_gu =
                W4a4ScratchLayout::for_rows(rows, W4a4Mode::GateUp, rows * 2_640, pairs * 1_280)
                    .expect("layout gateup");
            let up_scratch = DevBuf::zeroed(&gpu, rows * 2_640);
            let gate_scratch = DevBuf::zeroed(&gpu, pairs * 1_280);
            let down_bf16 = DevBuf::zeroed(&gpu, pairs * h * 2);
            let slot_lay = slot_layout(pairs, SlotStorage::Mxfp8).expect("slot layout");
            let down_mx = DevBuf::zeroed(&gpu, slot_lay.used_bytes);
            let down_mx_scales = down_mx
                .ptr()
                .offset(slot_lay.scales_offset.expect("scales offset"));

            let (xq_ptr, xs_ptr, xs2_ptr) = layout_all.input_ptrs(up_scratch.ptr());
            launch_w4a4_quant(
                &gpu,
                &w4a4_k,
                x_dev.ptr(),
                xq_ptr,
                xs_ptr,
                xs2_ptr,
                DevicePtr::NULL,
                DevicePtr::NULL,
                rows,
                stream,
            )
            .expect("w4a4 quant");

            // Read back quantized x for exact CPU fake-quant reference.
            let xq_h = up_scratch.read_bytes(layout_all.up_used_bytes);
            let xq_slice =
                &xq_h[layout_all.xq_offset..layout_all.xq_offset + rows * XQ_NVFP4_BYTES_PER_ROW];
            let xs_slice =
                &xq_h[layout_all.xs_offset..layout_all.xs_offset + rows * XS_NVFP4_BYTES_PER_ROW];
            let xs2_slice = f32_bytes_to_vec(
                &xq_h[layout_all.x_scale2_offset..layout_all.x_scale2_offset + rows * 4],
            );
            let x_dq: Vec<Vec<f32>> = (0..rows)
                .map(|r| dequant_nvfp4_row_uniform(xq_slice, xs_slice, &xs2_slice, r, h))
                .collect();
            let wg_dq: Vec<Vec<f32>> = unique_gate
                .iter()
                .map(|w| {
                    dequant_nvfp4_weight_row_major(&w.packed_rm, &w.scales_rm, w.scale2, inter, h)
                })
                .collect();
            let wu_dq: Vec<Vec<f32>> = unique_up
                .iter()
                .map(|w| {
                    dequant_nvfp4_weight_row_major(&w.packed_rm, &w.scales_rm, w.scale2, inter, h)
                })
                .collect();
            let wd_dq: Vec<Vec<f32>> = unique_down
                .iter()
                .map(|w| {
                    dequant_nvfp4_weight_row_major(&w.packed_rm, &w.scales_rm, w.scale2, h, inter)
                })
                .collect();

            // Exact CPU SwiGLU fake-quant reference over the first 64 expanded rows.
            let check_pairs = pairs.min(64);
            let mut swiglu_cpu = vec![0.0f32; check_pairs * inter];
            for orow in 0..check_pairs {
                let tok = sorted_h[orow] as usize;
                let eu = row_expert[orow] % 4;
                let xr = &x_dq[tok];
                for n in 0..inter {
                    let mut g_acc = 0.0f32;
                    let mut u_acc = 0.0f32;
                    let w_off = n * h;
                    for k in 0..h {
                        let xv = xr[k];
                        g_acc += xv * wg_dq[eu][w_off + k];
                        u_acc += xv * wu_dq[eu][w_off + k];
                    }
                    let silu = g_acc / (1.0 + (-g_acc).exp());
                    swiglu_cpu[orow * inter + n] = silu * u_acc;
                }
            }

            // Test BOTH Flat (includes sign-bit-injected scales!) and Ts4e128 (clean scales).
            for scale_layout in [W4a4ScaleLayout::Flat, W4a4ScaleLayout::Ts4e128] {
                let (gt, ut, dt) = match scale_layout {
                    W4a4ScaleLayout::Flat => (&gate_tbl.flat, &up_tbl.flat, &down_tbl.flat),
                    W4a4ScaleLayout::Ts4e128 => (&gate_tbl.ts4e, &up_tbl.ts4e, &down_tbl.ts4e),
                };
                // 1a. GATEUP OQ_NVFP4=false (MXFP8 output) vs exact CPU fake-quant.
                reset_work_counters(&gpu, workspace_dev.ptr(), stream);
                let (oq_mx, os_mx, oq_s2_null) = layout_gu.intermediate_ptrs(gate_scratch.ptr());
                assert!(oq_s2_null.is_null());
                launch_w4a4_gateup(
                    &gpu,
                    &w4a4_k,
                    W4a4Mode::GateUp,
                    scale_layout,
                    xq_ptr,
                    xs_ptr,
                    xs2_ptr,
                    gt,
                    ut,
                    oq_mx,
                    os_mx,
                    oq_s2_null,
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 gateup mxfp8");
                let gu_mx_bytes = gate_scratch.read_bytes(layout_gu.gate_used_bytes);
                let oq_mx_dq = dequant_mxfp8_matrix(
                    &gu_mx_bytes[layout_gu.oq_offset..layout_gu.oq_offset + check_pairs * inter],
                    &gu_mx_bytes
                        [layout_gu.os_offset..layout_gu.os_offset + check_pairs * (inter / 32)],
                    check_pairs,
                    inter,
                );
                let swiglu_bf16: Vec<f32> = swiglu_cpu
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_f32())
                    .collect();
                let swiglu_fq = fake_quant_mxfp8_cpu(&swiglu_bf16, check_pairs, inter);
                let (cos_gu_fq, max_gu_fq, nan_gu_mx) = parity_metrics(&oq_mx_dq, &swiglu_fq);
                let (cos_gu_fp32, _, _) = parity_metrics(&oq_mx_dq, &swiglu_cpu);
                eprintln!(
                    "v32_w4a4 exact fake-quant GATEUP(MXFP8) layout={scale_layout:?} rows={rows}: \
                     fq_cos={cos_gu_fq:.6} fq_max={max_gu_fq:.3e} fp32_cos={cos_gu_fp32:.6} nan={nan_gu_mx}"
                );
                assert_eq!(nan_gu_mx, 0);
                assert!(
                    cos_gu_fq >= 0.9999,
                    "GATEUP(MXFP8) {scale_layout:?} fq_cos={cos_gu_fq}"
                );
                assert!(
                    cos_gu_fp32 >= 0.9994,
                    "GATEUP(MXFP8) {scale_layout:?} fp32_cos={cos_gu_fp32}"
                );

                // 1b. GATEUP OQ_NVFP4=true (dynamic oq_scale2) + DOWN (BF16 & slot_mxfp8) vs CPU exact fake-quant.
                reset_work_counters(&gpu, workspace_dev.ptr(), stream);
                let (oq_4, os_4, oq_s2_ptr) = layout_all.intermediate_ptrs(gate_scratch.ptr());
                assert!(!oq_s2_ptr.is_null());
                launch_w4a4_gateup(
                    &gpu,
                    &w4a4_k,
                    W4a4Mode::All,
                    scale_layout,
                    xq_ptr,
                    xs_ptr,
                    xs2_ptr,
                    gt,
                    ut,
                    oq_4,
                    os_4,
                    oq_s2_ptr,
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 gateup oq4");
                launch_w4a4_down(
                    &gpu,
                    &w4a4_k,
                    scale_layout,
                    oq_4,
                    os_4,
                    oq_s2_ptr,
                    dt,
                    W4a4DownOutput::Bf16 {
                        c_out: down_bf16.ptr(),
                    },
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 down bf16");
                let gu4_bytes = gate_scratch.read_bytes(layout_all.gate_used_bytes);
                let oq4_slice = &gu4_bytes
                    [layout_all.oq_offset..layout_all.oq_offset + pairs * OQ_NVFP4_BYTES_PER_PAIR];
                let os4_slice = &gu4_bytes
                    [layout_all.os_offset..layout_all.os_offset + pairs * OS_NVFP4_BYTES_PER_PAIR];
                let s2_off = layout_all.oq_scale2_offset.expect("s2 off");
                let oq4_s2 =
                    f32_bytes_to_vec(&gu4_bytes[s2_off..s2_off + pairs * OQ_SCALE2_BYTES_PER_PAIR]);
                let down_bf16_got = bf16_bytes_to_f32(&down_bf16.read_bytes(check_pairs * h * 2));

                // Exact CPU reference for DOWN from the GPU-emitted dynamic NVFP4 oq/os/oq_scale2.
                let mut down_cpu = vec![0.0f32; check_pairs * h];
                for orow in 0..check_pairs {
                    let eu = row_expert[orow] % 4;
                    let oq_row =
                        dequant_nvfp4_row_dynamic(oq4_slice, os4_slice, &oq4_s2, orow, inter);
                    for n in 0..h {
                        let mut acc = 0.0f32;
                        let w_off = n * inter;
                        for k in 0..inter {
                            acc += oq_row[k] * wd_dq[eu][w_off + k];
                        }
                        down_cpu[orow * h + n] = acc;
                    }
                }
                let (cos_dn_bf16, max_dn_bf16, nan_dn_bf16) =
                    parity_metrics(&down_bf16_got, &down_cpu);
                eprintln!(
                    "v32_w4a4 exact fake-quant DOWN(BF16) layout={scale_layout:?} rows={rows}: cos={cos_dn_bf16:.6} max_abs={max_dn_bf16:.3e} nan={nan_dn_bf16}"
                );
                assert_eq!(nan_dn_bf16, 0);
                assert!(
                    cos_dn_bf16 >= 0.9999,
                    "DOWN(BF16) {scale_layout:?} cos={cos_dn_bf16}"
                );

                // Also verify DOWN(slot_mxfp8) matches exact CPU fake-quant reference.
                reset_work_counters(&gpu, workspace_dev.ptr(), stream);
                launch_w4a4_down(
                    &gpu,
                    &w4a4_k,
                    scale_layout,
                    oq_4,
                    os_4,
                    oq_s2_ptr,
                    dt,
                    W4a4DownOutput::Mxfp8Slots {
                        slot_q: down_mx.ptr(),
                        slot_s: down_mx_scales,
                    },
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 down mxfp8");
                let mx_bytes = down_mx.read_bytes(slot_lay.used_bytes);
                let mx_s_off = slot_lay.scales_offset.expect("mx scales off");
                let down_mx_got = dequant_mxfp8_matrix(
                    &mx_bytes[..check_pairs * h],
                    &mx_bytes[mx_s_off..mx_s_off + check_pairs * (h / 32)],
                    check_pairs,
                    h,
                );
                let down_cpu_bf16: Vec<f32> = down_cpu
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_f32())
                    .collect();
                let down_cpu_fq = fake_quant_mxfp8_cpu(&down_cpu_bf16, check_pairs, h);
                let (cos_dn_mx_fq, max_dn_mx_fq, nan_dn_mx) =
                    parity_metrics(&down_mx_got, &down_cpu_fq);
                let (cos_dn_mx_fp32, _, _) = parity_metrics(&down_mx_got, &down_cpu);
                eprintln!(
                    "v32_w4a4 exact fake-quant DOWN(MXFP8) layout={scale_layout:?} rows={rows}: \
                     fq_cos={cos_dn_mx_fq:.6} fq_max={max_dn_mx_fq:.3e} fp32_cos={cos_dn_mx_fp32:.6} nan={nan_dn_mx}"
                );
                assert_eq!(nan_dn_mx, 0);
                assert!(
                    cos_dn_mx_fq >= 0.9999,
                    "DOWN(MXFP8) {scale_layout:?} fq_cos={cos_dn_mx_fq}"
                );
                assert!(
                    cos_dn_mx_fp32 >= 0.9994,
                    "DOWN(MXFP8) {scale_layout:?} fp32_cos={cos_dn_mx_fp32}"
                );
            }
        }

        // =====================================================================
        // PART 2: Bimodal post-SwiGLU (1e-4 .. 3000.0) SNR vs V31_WS W4A8 ref
        //         across non-aligned rows in {31, 64, 129, 4003}.
        // =====================================================================
        // Scale gate/up columns per kh64 slice (`nt = 0..9`) so post-SwiGLU spans 1e-4..3000.0:
        //   nt=0 (cols 0..64):    huge (~2000..3000; would severely clip under static 0.5!)
        //   nt=1..3 (64..256):    medium (~1..20)
        //   nt=4..6 (256..448):   tiny (~1e-4..1e-2; would underflow under static 0.5!)
        //   nt=7..8 (448..576):   mixed (~0.1..500)
        //   nt=9 (576..640):      exact 0.0 on all 4 experts (tests cur_s2 == 0 skip path!)
        let bimodal_gate_scale = |col: usize| -> u8 {
            match col / 64 {
                0 => 0x60,     // 32.0 -> huge SwiGLU (~2000..3000)
                1..=3 => 0x40, // 2.0  -> medium (~1..20)
                4..=6 => 0x10, // 0.03125 -> tiny (~1e-4..1e-2)
                7..=8 => 0x50, // 8.0  -> mixed (~50..500)
                _ => 0x00,     // 0.0  -> exact zero kh64 slice
            }
        };
        let bimodal_up_scale = |col: usize| -> u8 {
            match col / 64 {
                0 => 0x58,     // 16.0
                1..=3 => 0x38, // 1.0
                4..=6 => 0x10, // 0.03125
                7..=8 => 0x48, // 4.0
                _ => 0x00,     // 0.0
            }
        };
        // Weight down columns inversely so tiny kh64 slices still contribute meaningfully to output!
        let bimodal_down_weights = |rng: &mut Rng| -> WeightSetHost {
            let half = inter / 2;
            let grp = inter / 16;
            let mut packed_rm = vec![0u8; h * half];
            for byte in &mut packed_rm {
                let lo = (rng.next_u32() & 0x0f) as u8;
                let hi = (rng.next_u32() & 0x0f) as u8;
                *byte = lo | (hi << 4);
            }
            let mut scales_rm = vec![0u8; h * grp];
            for r in 0..h {
                for g in 0..grp {
                    let nt = g / 4; // kh64 slice index 0..9
                    scales_rm[r * grp + g] = match nt {
                        0 => 0x20,     // 0.125 (dampens huge slice so it doesn't drown medium/tiny)
                        1..=3 => 0x38, // 1.0
                        4..=6 => 0x60, // 32.0 (amplifies tiny 1e-4..1e-2 slice into output!)
                        7..=8 => 0x30, // 0.5
                        _ => 0x38,
                    };
                }
            }
            let packed_t = transpose_u8(&packed_rm, h, half);
            let scales_flat_t = transpose_u8(&scales_rm, h, grp);
            let (scales_ts4e, _) = ts4e128_encode_host(&scales_rm, inter, h).expect("ts4e");
            WeightSetHost {
                packed_rm,
                scales_rm,
                packed_t,
                scales_flat_t,
                scales_ts4e,
                scale2: 0.02,
            }
        };
        let bi_gate: Vec<WeightSetHost> = (0..4)
            .map(|_| make_nvfp4_weights(&mut rng, inter, h, 0.008, bimodal_gate_scale, false))
            .collect();
        let bi_up: Vec<WeightSetHost> = (0..4)
            .map(|_| make_nvfp4_weights(&mut rng, inter, h, 0.008, bimodal_up_scale, false))
            .collect();
        let bi_down: Vec<WeightSetHost> = (0..4).map(|_| bimodal_down_weights(&mut rng)).collect();
        let bi_gate_tbl = upload_expert_tables(&gpu, &bi_gate);
        let bi_up_tbl = upload_expert_tables(&gpu, &bi_up);
        let bi_down_tbl = upload_expert_tables(&gpu, &bi_down);

        for &rows in &[31usize, 64, 129, 4003] {
            let pairs = rows * FLASH_NEXT_TOPK;
            // Construct BF16 activations whose per-block values lie cleanly on both the MXFP8
            // and NVFP4 grids (power-of-two row scales) so V31_WS and v32_w4a4 GATEUP match
            // pre-SwiGLU inputs and the end-to-end comparison directly measures post-SwiGLU
            // bimodal 1e-4..3000.0 SNR (dynamic per-(orow, kh64) oq_scale2 vs V31_WS W4A8).
            let mut x_host = vec![0.0f32; rows * h];
            for r in 0..rows {
                let row_scale = match r % 3 {
                    0 => 0.125f32,
                    1 => 0.25,
                    _ => 0.5,
                };
                for c in 0..h {
                    // Ensure each 16-element block contains one peak 6.0*row_scale so every
                    // block scale equals exact 448.0 (zero x-quant diff between MXFP8 & NVFP4).
                    let nib = if (c & 15) == 0 {
                        0x07u8
                    } else {
                        (rng.next_u32() & 0x0f) as u8
                    };
                    x_host[r * h + c] = bf16::from_f32(e2m1_to_f32(nib) * row_scale).to_f32();
                }
            }
            let x_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&x_host));
            let (offsets_h, sorted_h, _) = build_routing(rows, 16);
            let offsets_bytes: Vec<u8> = offsets_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sorted_bytes: Vec<u8> = sorted_h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let offsets_dev = DevBuf::upload(&gpu, &offsets_bytes);
            let sorted_dev = DevBuf::upload(&gpu, &sorted_bytes);
            let contract_dev = DevBuf::zeroed(&gpu, abi::CONTRACT_BYTES);
            let workspace_dev = DevBuf::zeroed(&gpu, abi::ARENA_BYTES);
            let status_dev = DevBuf::zeroed(&gpu, 4);
            let work = W4a4WorkBuffers {
                expert_offsets: offsets_dev.ptr(),
                sorted_token_ids: sorted_dev.ptr(),
                contract: contract_dev.ptr(),
                workspace: workspace_dev.ptr(),
                status: status_dev.ptr(),
            };
            let max_items = pairs.min(pairs.div_ceil(128) + 512);
            run_plan(
                &gpu,
                plan_k,
                offsets_dev.ptr(),
                sorted_dev.ptr(),
                contract_dev.ptr(),
                workspace_dev.ptr(),
                status_dev.ptr(),
                rows,
                stream,
            );

            // Reference: V31_WS W4A8 (BF16 output).
            let ref_up = DevBuf::zeroed(&gpu, rows * 2_640);
            let ref_gate = DevBuf::zeroed(&gpu, pairs * 1_280);
            let ref_down_bf16 = DevBuf::zeroed(&gpu, pairs * h * 2);
            let ref_ss = DevBuf::zeroed(&gpu, rows * 4);
            let (ref_xq, ref_xs) = (ref_up.ptr(), ref_up.ptr().offset(rows * h));
            let (ref_aq, ref_as) = (ref_gate.ptr(), ref_gate.ptr().offset(pairs * inter));
            KernelLaunch::new(&gpu, xq_mxfp8_k)
                .grid([(rows.min(1024)) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(x_dev.ptr())
                .arg_ptr(ref_xq)
                .arg_ptr(ref_xs)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(ref_ss.ptr())
                .arg_u32(rows as u32)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .launch(stream)
                .expect("v31 xq quant");
            let v31_grid_gu = (max_items * 10).clamp(48, abi::GRID as usize) as u32;
            let v31_grid_dn = (max_items * 20).clamp(48, abi::GRID as usize) as u32;
            KernelLaunch::new(&gpu, v31_k.gateup_flat)
                .grid([v31_grid_gu, 1, 1])
                .block([V31_WS_BLOCK, 1, 1])
                .shared_mem(V31_WS_FLAT_GATEUP_SMEM_BYTES)
                .arg_ptr(ref_xq)
                .arg_ptr(ref_xs)
                .arg_ptr(bi_gate_tbl.flat.packed_ptrs)
                .arg_ptr(bi_gate_tbl.flat.scale_ptrs)
                .arg_ptr(bi_gate_tbl.flat.scale2_vals)
                .arg_ptr(ref_aq)
                .arg_ptr(ref_as)
                .arg_ptr(bi_up_tbl.flat.packed_ptrs)
                .arg_ptr(bi_up_tbl.flat.scale_ptrs)
                .arg_ptr(bi_up_tbl.flat.scale2_vals)
                .arg_ptr(offsets_dev.ptr())
                .arg_ptr(sorted_dev.ptr())
                .arg_ptr(contract_dev.ptr())
                .arg_ptr(workspace_dev.ptr())
                .arg_ptr(status_dev.ptr())
                .arg_u32(FLASH_NEXT_INTER)
                .arg_u32(FLASH_NEXT_HIDDEN)
                .launch(stream)
                .expect("v31 gateup");
            KernelLaunch::new(&gpu, v31_k.down_bf16_flat)
                .grid([v31_grid_dn, 1, 1])
                .block([V31_WS_BLOCK, 1, 1])
                .shared_mem(V31_WS_FLAT_DOWN_SMEM_BYTES)
                .arg_ptr(ref_aq)
                .arg_ptr(ref_as)
                .arg_ptr(bi_down_tbl.flat.packed_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale2_vals)
                .arg_ptr(ref_down_bf16.ptr())
                .arg_ptr(offsets_dev.ptr())
                .arg_ptr(sorted_dev.ptr())
                .arg_ptr(contract_dev.ptr())
                .arg_ptr(workspace_dev.ptr())
                .arg_ptr(status_dev.ptr())
                .arg_u32(FLASH_NEXT_HIDDEN)
                .arg_u32(FLASH_NEXT_INTER)
                .launch(stream)
                .expect("v31 down bf16");
            // Check only a representative slice on 4003 (first 512 + last 512 expanded rows)
            // plus full check on rows <= 129 to keep host-side reduction fast.
            let sample_pairs = if rows <= 129 { pairs } else { 512 };
            let ref_head = bf16_bytes_to_f32(&ref_down_bf16.read_bytes(sample_pairs * h * 2));
            let ref_tail = if rows > 129 {
                let mut buf = vec![0u8; sample_pairs * h * 2];
                gpu.copy_d2h(
                    ref_down_bf16.ptr().offset((pairs - sample_pairs) * h * 2),
                    &mut buf,
                )
                .expect("d2h ref tail");
                bf16_bytes_to_f32(&buf)
            } else {
                Vec::new()
            };

            // Candidate A: v32_w4a4 GateUp (OQ_NVFP4=false) + V31_WS DOWN (BF16).
            let layout_gu =
                W4a4ScratchLayout::for_rows(rows, W4a4Mode::GateUp, rows * 2_640, pairs * 1_280)
                    .expect("layout gu");
            let layout_all =
                W4a4ScratchLayout::for_rows(rows, W4a4Mode::All, rows * 2_640, pairs * 1_280)
                    .expect("layout all");
            let w4_up = DevBuf::zeroed(&gpu, rows * 2_640);
            let w4_gate = DevBuf::zeroed(&gpu, pairs * 1_280);
            let cand_bf16 = DevBuf::zeroed(&gpu, pairs * h * 2);
            let (xq_ptr, xs_ptr, xs2_ptr) = layout_all.input_ptrs(w4_up.ptr());
            launch_w4a4_quant(
                &gpu,
                &w4a4_k,
                x_dev.ptr(),
                xq_ptr,
                xs_ptr,
                xs2_ptr,
                DevicePtr::NULL,
                DevicePtr::NULL,
                rows,
                stream,
            )
            .expect("w4a4 quant bi");

            reset_work_counters(&gpu, workspace_dev.ptr(), stream);
            let (oq_mx, os_mx, oq_s2_null) = layout_gu.intermediate_ptrs(w4_gate.ptr());
            launch_w4a4_gateup(
                &gpu,
                &w4a4_k,
                W4a4Mode::GateUp,
                W4a4ScaleLayout::Flat,
                xq_ptr,
                xs_ptr,
                xs2_ptr,
                &bi_gate_tbl.flat,
                &bi_up_tbl.flat,
                oq_mx,
                os_mx,
                oq_s2_null,
                work,
                max_items,
                DEFAULT_OQ_SCALE2,
                stream,
            )
            .expect("w4a4 gateup bi mxfp8");
            KernelLaunch::new(&gpu, v31_k.down_bf16_flat)
                .grid([v31_grid_dn, 1, 1])
                .block([V31_WS_BLOCK, 1, 1])
                .shared_mem(V31_WS_FLAT_DOWN_SMEM_BYTES)
                .arg_ptr(oq_mx)
                .arg_ptr(os_mx)
                .arg_ptr(bi_down_tbl.flat.packed_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale2_vals)
                .arg_ptr(cand_bf16.ptr())
                .arg_ptr(offsets_dev.ptr())
                .arg_ptr(sorted_dev.ptr())
                .arg_ptr(contract_dev.ptr())
                .arg_ptr(workspace_dev.ptr())
                .arg_ptr(status_dev.ptr())
                .arg_u32(FLASH_NEXT_HIDDEN)
                .arg_u32(FLASH_NEXT_INTER)
                .launch(stream)
                .expect("v31 down after w4a4 gateup");
            let gu_head = bf16_bytes_to_f32(&cand_bf16.read_bytes(sample_pairs * h * 2));
            let (cos_hybrid, max_hybrid, nan_hybrid) = parity_metrics(&gu_head, &ref_head);
            assert_eq!(nan_hybrid, 0);

            // Candidate B: v32_w4a4 All (OQ_NVFP4=true + dynamic oq_scale2) + DOWN (BF16 & slot_mxfp8).
            reset_work_counters(&gpu, workspace_dev.ptr(), stream);
            let (oq_4, os_4, oq_s2_ptr) = layout_all.intermediate_ptrs(w4_gate.ptr());
            launch_w4a4_gateup(
                &gpu,
                &w4a4_k,
                W4a4Mode::All,
                W4a4ScaleLayout::Flat,
                xq_ptr,
                xs_ptr,
                xs2_ptr,
                &bi_gate_tbl.flat,
                &bi_up_tbl.flat,
                oq_4,
                os_4,
                oq_s2_ptr,
                work,
                max_items,
                DEFAULT_OQ_SCALE2,
                stream,
            )
            .expect("w4a4 gateup bi all");
            launch_w4a4_down(
                &gpu,
                &w4a4_k,
                W4a4ScaleLayout::Flat,
                oq_4,
                os_4,
                oq_s2_ptr,
                &bi_down_tbl.flat,
                W4a4DownOutput::Bf16 {
                    c_out: cand_bf16.ptr(),
                },
                work,
                max_items,
                DEFAULT_OQ_SCALE2,
                stream,
            )
            .expect("w4a4 down bi bf16");
            let all_head = bf16_bytes_to_f32(&cand_bf16.read_bytes(sample_pairs * h * 2));
            let (cos_all_bf16, max_all_bf16, nan_all_bf16) = parity_metrics(&all_head, &ref_head);
            let (cos_all_vs_gu, max_all_vs_gu, _) = parity_metrics(&all_head, &gu_head);
            assert_eq!(nan_all_bf16, 0);

            // Also run static oq_scale2=0.5 ablation at rows=31 to confirm bimodal input breaks static 0.5!
            if rows == 31 {
                reset_work_counters(&gpu, workspace_dev.ptr(), stream);
                let stat_gate = DevBuf::zeroed(&gpu, pairs * 1_280);
                let (soq, sos, _) = layout_all.intermediate_ptrs(stat_gate.ptr());
                launch_w4a4_gateup(
                    &gpu,
                    &w4a4_k,
                    W4a4Mode::All,
                    W4a4ScaleLayout::Flat,
                    xq_ptr,
                    xs_ptr,
                    xs2_ptr,
                    &bi_gate_tbl.flat,
                    &bi_up_tbl.flat,
                    soq,
                    sos,
                    DevicePtr::NULL,
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 gateup static");
                launch_w4a4_down(
                    &gpu,
                    &w4a4_k,
                    W4a4ScaleLayout::Flat,
                    soq,
                    sos,
                    DevicePtr::NULL,
                    &bi_down_tbl.flat,
                    W4a4DownOutput::Bf16 {
                        c_out: cand_bf16.ptr(),
                    },
                    work,
                    max_items,
                    DEFAULT_OQ_SCALE2,
                    stream,
                )
                .expect("w4a4 down static");
                let stat_head = bf16_bytes_to_f32(&cand_bf16.read_bytes(sample_pairs * h * 2));
                let (cos_stat_vs_gu, _, _) = parity_metrics(&stat_head, &gu_head);
                let (cos_stat_vs_v31, _, _) = parity_metrics(&stat_head, &ref_head);
                eprintln!(
                    "v32_w4a4 ABLATION static(0.5) vs dynamic oq_scale2 rows=31: \
                     static_vs_V31={cos_stat_vs_v31:.6} static_vs_GateUp={cos_stat_vs_gu:.6} | \
                     dynamic_vs_V31={cos_all_bf16:.6} dynamic_vs_GateUp={cos_all_vs_gu:.6}"
                );
                // Proves the bimodal workload genuinely breaks old static 0.5 while dynamic passes.
                assert!(
                    cos_stat_vs_v31 < 0.96 && cos_all_bf16 > cos_stat_vs_v31 + 0.03,
                    "expected dynamic oq_scale2 ({cos_all_bf16}) to massively beat static 0.5 ({cos_stat_vs_v31})"
                );
            }
            if rows > 129 {
                let mut tail_bytes = vec![0u8; sample_pairs * h * 2];
                gpu.copy_d2h(
                    cand_bf16.ptr().offset((pairs - sample_pairs) * h * 2),
                    &mut tail_bytes,
                )
                .expect("d2h cand tail");
                let all_tail = bf16_bytes_to_f32(&tail_bytes);
                let (cos_tail, _, nan_tail) = parity_metrics(&all_tail, &ref_tail);
                assert_eq!(nan_tail, 0);
                assert!(
                    cos_tail >= 0.996,
                    "bimodal All(BF16) tail vs V31_WS rows={rows} cos={cos_tail}"
                );
            }

            // Confirm dynamic oq_scale2 truly spanned bimodal range (tiny < 1e-4 to huge > 0.5).
            let s2_off = layout_all.oq_scale2_offset.expect("s2 off");
            let mut s2_sample_bytes = vec![0u8; sample_pairs * OQ_SCALE2_BYTES_PER_PAIR];
            gpu.copy_d2h(w4_gate.ptr().offset(s2_off), &mut s2_sample_bytes)
                .expect("d2h oq_scale2");
            let s2_sample = f32_bytes_to_vec(&s2_sample_bytes);
            let s2_min_pos = s2_sample
                .iter()
                .copied()
                .filter(|&v| v > 0.0)
                .fold(f32::INFINITY, f32::min);
            let s2_max = s2_sample.iter().copied().fold(0.0f32, f32::max);
            let s2_zeros = s2_sample.iter().filter(|&&v| v == 0.0).count();
            assert!(
                s2_min_pos < 1e-4 && s2_max > 0.5 && s2_zeros > 0,
                "expected bimodal oq_scale2 range: min_pos={s2_min_pos:.3e} max={s2_max:.3e} zeros={s2_zeros}"
            );

            // Candidate C: v32_w4a4 All DOWN (slot_mxfp8) vs V31_WS DOWN (slot_mxfp8) apples-to-apples.
            let slot_lay = slot_layout(pairs, SlotStorage::Mxfp8).expect("slot lay");
            let ref_mx = DevBuf::zeroed(&gpu, slot_lay.used_bytes);
            let ref_mx_s = ref_mx
                .ptr()
                .offset(slot_lay.scales_offset.expect("ref mx scales"));
            reset_work_counters(&gpu, workspace_dev.ptr(), stream);
            KernelLaunch::new(&gpu, v31_k.down_mxfp8_flat)
                .grid([v31_grid_dn, 1, 1])
                .block([V31_WS_BLOCK, 1, 1])
                .shared_mem(V31_WS_FLAT_DOWN_SMEM_BYTES)
                .arg_ptr(ref_aq)
                .arg_ptr(ref_as)
                .arg_ptr(bi_down_tbl.flat.packed_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale_ptrs)
                .arg_ptr(bi_down_tbl.flat.scale2_vals)
                .arg_ptr(ref_mx.ptr())
                .arg_ptr(ref_mx_s)
                .arg_ptr(offsets_dev.ptr())
                .arg_ptr(sorted_dev.ptr())
                .arg_ptr(contract_dev.ptr())
                .arg_ptr(workspace_dev.ptr())
                .arg_ptr(status_dev.ptr())
                .arg_u32(FLASH_NEXT_HIDDEN)
                .arg_u32(FLASH_NEXT_INTER)
                .launch(stream)
                .expect("v31 down mxfp8");
            let cand_mx = DevBuf::zeroed(&gpu, slot_lay.used_bytes);
            let cand_mx_s = cand_mx
                .ptr()
                .offset(slot_lay.scales_offset.expect("mx scales"));
            reset_work_counters(&gpu, workspace_dev.ptr(), stream);
            launch_w4a4_down(
                &gpu,
                &w4a4_k,
                W4a4ScaleLayout::Flat,
                oq_4,
                os_4,
                oq_s2_ptr,
                &bi_down_tbl.flat,
                W4a4DownOutput::Mxfp8Slots {
                    slot_q: cand_mx.ptr(),
                    slot_s: cand_mx_s,
                },
                work,
                max_items,
                DEFAULT_OQ_SCALE2,
                stream,
            )
            .expect("w4a4 down bi mxfp8");
            gpu.synchronize(stream).expect("sync mxfp8");
            let mut ref_mx_q_head = vec![0u8; sample_pairs * h];
            let mut ref_mx_s_head = vec![0u8; sample_pairs * (h / 32)];
            gpu.copy_d2h(ref_mx.ptr(), &mut ref_mx_q_head)
                .expect("d2h ref mx q");
            gpu.copy_d2h(ref_mx_s, &mut ref_mx_s_head)
                .expect("d2h ref mx s");
            let ref_mx_head = dequant_mxfp8_matrix(&ref_mx_q_head, &ref_mx_s_head, sample_pairs, h);
            let mut mx_q_head = vec![0u8; sample_pairs * h];
            let mut mx_s_head = vec![0u8; sample_pairs * (h / 32)];
            gpu.copy_d2h(cand_mx.ptr(), &mut mx_q_head)
                .expect("d2h mx q");
            gpu.copy_d2h(cand_mx_s, &mut mx_s_head).expect("d2h mx s");
            let all_mx_head = dequant_mxfp8_matrix(&mx_q_head, &mx_s_head, sample_pairs, h);
            let (cos_all_mx, max_all_mx, nan_all_mx) = parity_metrics(&all_mx_head, &ref_mx_head);
            let (cos_all_mx_vs_bf16, _, _) = parity_metrics(&all_mx_head, &ref_head);
            let (cos_all_mx_vs_all_bf16, _, _) = parity_metrics(&all_mx_head, &all_head);
            eprintln!(
                "v32_w4a4 bimodal(1e-4..3000) vs V31_WS rows={rows}: \
                 GateUp+V31 cos={cos_hybrid:.6} max={max_hybrid:.3e} | \
                 All(BF16) vs V31(BF16)={cos_all_bf16:.6} (max={max_all_bf16:.3e}) vs GateUp={cos_all_vs_gu:.6} (max={max_all_vs_gu:.3e}) | \
                 All(MXFP8) vs V31(BF16)={cos_all_mx_vs_bf16:.6} vs V31(MXFP8)={cos_all_mx:.6} (max={max_all_mx:.3e}) vs All(BF16)={cos_all_mx_vs_all_bf16:.6} | \
                 oq_s2=[{s2_min_pos:.2e}..{s2_max:.2e}] zeros={s2_zeros}"
            );
            assert_eq!(nan_all_mx, 0);
            assert!(
                cos_hybrid >= 0.9999,
                "bimodal GateUp+V31 vs V31_WS rows={rows} cos={cos_hybrid}"
            );
            assert!(
                cos_all_bf16 >= 0.996,
                "bimodal All(BF16) vs V31_WS(BF16) rows={rows} cos={cos_all_bf16}"
            );
            assert!(
                cos_all_mx_vs_bf16 >= 0.996,
                "bimodal All(MXFP8) vs V31_WS(BF16) rows={rows} cos={cos_all_mx_vs_bf16}"
            );
            assert!(
                cos_all_mx >= 0.9955 && cos_all_mx_vs_all_bf16 >= 0.9995,
                "bimodal All(MXFP8) parity rows={rows}: vs_v31_mx={cos_all_mx} vs_all_bf16={cos_all_mx_vs_all_bf16}"
            );
            assert!(
                cos_all_vs_gu >= 0.996,
                "bimodal All(BF16) vs GateUp+V31 rows={rows} cos={cos_all_vs_gu}"
            );
        }
    }
}
