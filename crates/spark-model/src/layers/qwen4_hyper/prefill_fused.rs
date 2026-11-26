// SPDX-License-Identifier: AGPL-3.0-only

//! Fused hyperconnection preparation for wide Flash-Next prefill
//! (`ATLAS_QWEN4_PREFILL_HC_FUSED=1|2`, default off; see
//! `kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_hc_fused.cu`).
//!
//! Same contract as [`Qwen4HyperConnection::prepare_prefill_fast`]: returns
//! the mixed `[rows, H]` core input in `norm_output`, stages it in the front
//! of each residual row, and leaves the injection scales in each residual
//! row's tail for `inject_saved_batched`. It never writes the normalized
//! four-stream rows (nothing downstream reads them on this path): the norm is
//! recomputed from `hyper` and a per-(row, stream) rstd, and the norm weight
//! is folded into the down/inject weights. Level 1 keeps the cuBLASLt up
//! projection and mixes from its logits; level 2 fuses the up projection and
//! the mix in one tensor-core kernel so the 10240-wide logits never reach DRAM.
//! Not bit-exact versus the fast path (different rounding points).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::cublaslt::GemmDtype;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Qwen4HyperConnection;
use super::pending::{MoeCombine, PendingInject};
use super::prefill_inject::disjoint;
use super::prefill_upmix_roof::UpmixLaunch;
#[allow(dead_code, unused_imports)]
pub(crate) use super::prefill_upmix_roof::{
    PreQuantizedMxfp8, UpmixMxfp8Arm, UpmixMxfp8Request, clear_prequant_mxfp8,
    launch_fp8_act_from_bf16, mxfp8_fused_enabled, take_prequant_mxfp8,
};
use super::prefill_upmix_roof_plan::OutputCachePolicy;
mod qwen4_fast_proj {
    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    pub(super) use crate::layers::qwen4_fast_proj::dequant_gemm;
    use crate::weight_map::QuantizedWeight;

    pub(super) fn dequant_into(
        gpu: &dyn GpuBackend,
        weight: &QuantizedWeight,
        out: DevicePtr,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<f32> {
        if super::fused_prep_selected() {
            Ok(weight.weight_scale_2)
        } else {
            crate::layers::qwen4_fast_proj::dequant_into(gpu, weight, out, n, k, stream)
        }
    }
}

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_FUSED";
pub(crate) const FUSED_PREP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_FUSED_PREP";
/// L2 strip height for `(inject/combine_moe -> upmix)` fusion on wide level-2
/// prefill (`0`/`off` disables strip-mining, which is the default on GB10 where
/// 2-kernel strip-mining regresses over unstriped W21; set e.g. `768` to opt in
/// with inline NVFP4 `folded`/`up` and fused W21-F MXFP8/gate).
pub(crate) const STRIP_ROWS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_STRIP_ROWS";
const DEFAULT_STRIP_ROWS: usize = 0;
/// Single-launch BM=64 cluster-fused inject + inline-W4 upmix (+ optional
/// zero-reread MXFP8/gate epilogue) for wide pending-Core prefill boundaries
/// (`default ON` for `num_tokens >= BM64_UPMIX_MIN_ROWS`; set `0`/`off`/`false`
/// to opt out).
pub(crate) const BM64_UPMIX_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_BM64_UPMIX";
/// Single-launch BM=64 cluster-fused MoE combine + inject + inline-W4 upmix
/// for wide pending-Combine prefill boundaries (`default OFF` on GB10
/// `active_sm_count() <= 64`, `default ON` on SM120/SM103 `active_sm_count() > 64`
/// where 96 MiB L2 holds `hyper'` + `moe_rows` + weights without thrashing;
/// override with `0`/`1`/`off`/`on`/`false`/`true`).
pub(crate) const BM64_COMBINE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_BM64_COMBINE";
/// Opt-in 2-CTA/SM (`47,232 B` SMEM, BM=32) Phase-2 cluster-fused mHC path
/// (`default OFF`; set `1`/`on`/`true`/`mxfp8` for 0-ULP/W4A8-exact MXFP8
/// `m16n8k32` `scale_vec::1X` UE8M0, or `nvfp4` for explicit E2M1 opt-in).
pub(crate) const MHC_NVFP4_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_NVFP4";
/// Minimum row count where BM=64 cluster-fused W4 beats 2-kernel W21 on GB10
/// (`active_sm_count() <= 64`: `T <= 4096` stays on W21; `T = 8192/14885/16384`
/// saves `-10.5/-32.0/-35.3 ms` across 48 MLP layers at strict 0-ULP).
const BM64_UPMIX_MIN_ROWS_GB10: usize = 4097;
/// Lower row floor on high-SM Blackwell (`active_sm_count() > 64`) so 4K
/// prefill also gets the BM=64 / 2-CTA BM=32 cluster-fused mHC speedup.
const BM64_UPMIX_MIN_ROWS_HIGH_SM: usize = 2048;
/// SM count threshold above which `BM64_COMBINE_SELECTOR` defaults ON and
/// `bm64_upmix_min_rows_for_sm` lowers to `2048` (SM120/SM103 96 MiB L2 targets;
/// GB10 with 48 SMs keeps default OFF and `4097`).
const BM64_COMBINE_HIGH_SM_THRESHOLD: u32 = 64;
/// `hc_bm64_upmix::kSmemBytes` in `qwen4_hc_combine_gemm.cu` (`static_assert` there).
const SMEM_BYTES_BM64_W4: u32 = 95_744;
/// `hc_bm32_nvfp4::kSmemBytes` in `qwen4_hc_combine_gemm.cu` (`static_assert` there).
const SMEM_BYTES_BM32_NVFP4: u32 = 47_232;
/// `hc_bm32_mxfp8::kSmemBytes` in `qwen4_hc_combine_gemm.cu` (`static_assert` there).
const SMEM_BYTES_BM32_MXFP8: u32 = 47_232;
/// `hc_w4::kSmemBytesW4` in `qwen4_hc_combine_gemm.cu` (`static_assert` there).
const SMEM_BYTES_STRIP_W4: u32 = 83_200;
const CLUSTER_TILE_ROWS: u32 = 64;
const CLUSTER_THREADS: u32 = 512;
const CLUSTER_TILE_ROWS_BM32: u32 = 32;
const CLUSTER_THREADS_BM32: u32 = 256;
const CLUSTER_STREAMS: u32 = 4;
const STRIP_W4_TOPK: u32 = 10;
/// Single-kernel fused 4-CTA cluster mHC selector (`default OFF` on GB10:
/// strict 0-ULP verified, but `BM=32` 1-CTA/SM amplifies L2 weight reads on
/// `folded+up` vs `inject_gemm(BM=64) + upmix_roof_t128(BM=128)` by +0.86..0.93 ms/call;
/// set `1`/`on`/`true` to opt in when `level >= 2`, `injgemm` + `gatepre` + `upmix`
/// are active, `strip_rows == 0`, and `rows >= 512`).
pub(crate) const CLUSTER_FUSE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_CLUSTER_FUSE";
const CLUSTER_FUSE_MIN_ROWS: usize = 512;

fn parse_bm64_toggle(selector: &str, value: Option<&str>, default: bool) -> Result<bool, String> {
    let Some(raw) = value else {
        return Ok(default);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => Ok(default),
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        other => Err(format!(
            "{selector}: unknown value {other:?} \
             (expected 0, 1, off, on, false, true)"
        )),
    }
}

fn parse_bm64_upmix(value: Option<&str>) -> Result<bool, String> {
    parse_bm64_toggle(BM64_UPMIX_SELECTOR, value, true)
}

fn bm64_upmix_enabled() -> Result<bool> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_bm64_upmix(std::env::var(BM64_UPMIX_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn parse_bm64_combine_for_sms(value: Option<&str>, sm_count: u32) -> Result<bool, String> {
    parse_bm64_toggle(
        BM64_COMBINE_SELECTOR,
        value,
        sm_count > BM64_COMBINE_HIGH_SM_THRESHOLD,
    )
}

#[cfg(test)]
fn parse_bm64_combine(value: Option<&str>) -> Result<bool, String> {
    parse_bm64_combine_for_sms(value, 48)
}

fn bm64_combine_enabled() -> Result<bool> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| {
            parse_bm64_combine_for_sms(
                std::env::var(BM64_COMBINE_SELECTOR).ok().as_deref(),
                atlas_core::device::active_sm_count(),
            )
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Operating mode for the 2-CTA/SM (`47,232 B`, BM=32) cluster-fused mHC kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MhcBm32Mode {
    /// Disabled (default): keep 1-CTA/SM BM=64 W4 path (`hc_bm64_upmix`).
    #[default]
    Off,
    /// 2-CTA/SM BM=32 W4A8 MXFP8 (`m16n8k32` `scale_vec::1X` UE8M0): W4A8-exact.
    Mxfp8,
    /// 2-CTA/SM BM=32 W4A4 NVFP4 (`m16n8k64` `scale_vec::4X` UE4M3): explicit opt-in.
    Nvfp4,
}

impl MhcBm32Mode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Mxfp8 => "mxfp8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

fn parse_mhc_bm32_mode(value: Option<&str>) -> Result<MhcBm32Mode, String> {
    let Some(raw) = value else {
        return Ok(MhcBm32Mode::Off);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => Ok(MhcBm32Mode::Off),
        "1" | "on" | "true" | "mxfp8" => Ok(MhcBm32Mode::Mxfp8),
        "nvfp4" => Ok(MhcBm32Mode::Nvfp4),
        other => Err(format!(
            "{MHC_NVFP4_SELECTOR}: unknown value {other:?} \
             (expected 0, 1, off, on, false, true, mxfp8, nvfp4)"
        )),
    }
}

fn mhc_bm32_mode() -> Result<MhcBm32Mode> {
    static MODE: OnceLock<Result<MhcBm32Mode, String>> = OnceLock::new();
    MODE.get_or_init(|| parse_mhc_bm32_mode(std::env::var(MHC_NVFP4_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

const fn bm64_upmix_min_rows_for_sm(sm_count: u32) -> usize {
    if sm_count > BM64_COMBINE_HIGH_SM_THRESHOLD {
        BM64_UPMIX_MIN_ROWS_HIGH_SM
    } else {
        BM64_UPMIX_MIN_ROWS_GB10
    }
}

fn bm64_combine_oracle_or_dump() -> bool {
    crate::layers::qwen4_oracle::root().is_some()
        || std::env::var("ATLAS_DUMP_EXPERT_IDS").ok().as_deref() == Some("1")
}

fn bm64_upmix_eligible_for_sm(
    enabled: bool,
    level: u32,
    strip_cap: usize,
    num_tokens: usize,
    sel: super::prefill_inject::Selection,
    sm_count: u32,
) -> bool {
    enabled
        && level >= 2
        && strip_cap == 0
        && num_tokens >= bm64_upmix_min_rows_for_sm(sm_count)
        && sel.injgemm
        && sel.upmix
}

#[cfg(test)]
fn bm64_upmix_eligible(
    enabled: bool,
    level: u32,
    strip_cap: usize,
    num_tokens: usize,
    sel: super::prefill_inject::Selection,
) -> bool {
    bm64_upmix_eligible_for_sm(enabled, level, strip_cap, num_tokens, sel, 48)
}

#[derive(Clone, Copy)]
struct Bm64UpmixKernels {
    plain: [KernelHandle; 2],
    mxfp8_gate: [KernelHandle; 2],
}

fn bm64_upmix_kernels(gpu: &dyn GpuBackend) -> Option<Bm64UpmixKernels> {
    static K: OnceLock<Option<Bm64UpmixKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM64_W4)
                .ok()?;
            Some(k)
        };
        Some(Bm64UpmixKernels {
            plain: [
                load("qwen4_hc_inject_upmix_bm64_w4")?,
                load("qwen4_hc_inject_upmix_bm64_w4_amax")?,
            ],
            mxfp8_gate: [
                load("qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate")?,
                load("qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate_amax")?,
            ],
        })
    })
}

#[derive(Clone, Copy)]
struct Bm64CombineKernels {
    mx_gatepre: [KernelHandle; 2],
}

#[derive(Clone, Copy)]
struct ClusterFusedLayout {
    tile_rows: u32,
    threads: u32,
    smem_bytes: u32,
    selector: &'static str,
    bm32_mode: MhcBm32Mode,
}

const BM64_W4_LAYOUT: ClusterFusedLayout = ClusterFusedLayout {
    tile_rows: CLUSTER_TILE_ROWS,
    threads: CLUSTER_THREADS,
    smem_bytes: SMEM_BYTES_BM64_W4,
    selector: BM64_UPMIX_SELECTOR,
    bm32_mode: MhcBm32Mode::Off,
};

const BM64_COMBINE_LAYOUT: ClusterFusedLayout = ClusterFusedLayout {
    tile_rows: CLUSTER_TILE_ROWS,
    threads: CLUSTER_THREADS,
    smem_bytes: SMEM_BYTES_BM64_W4,
    selector: BM64_COMBINE_SELECTOR,
    bm32_mode: MhcBm32Mode::Off,
};

const BM32_MXFP8_LAYOUT: ClusterFusedLayout = ClusterFusedLayout {
    tile_rows: CLUSTER_TILE_ROWS_BM32,
    threads: CLUSTER_THREADS_BM32,
    smem_bytes: SMEM_BYTES_BM32_MXFP8,
    selector: MHC_NVFP4_SELECTOR,
    bm32_mode: MhcBm32Mode::Mxfp8,
};

const BM32_NVFP4_LAYOUT: ClusterFusedLayout = ClusterFusedLayout {
    tile_rows: CLUSTER_TILE_ROWS_BM32,
    threads: CLUSTER_THREADS_BM32,
    smem_bytes: SMEM_BYTES_BM32_NVFP4,
    selector: MHC_NVFP4_SELECTOR,
    bm32_mode: MhcBm32Mode::Nvfp4,
};

fn bm32_mxfp8_upmix_kernels(gpu: &dyn GpuBackend) -> Option<Bm64UpmixKernels> {
    static K: OnceLock<Option<Bm64UpmixKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM32_MXFP8)
                .ok()?;
            Some(k)
        };
        Some(Bm64UpmixKernels {
            plain: [
                load("qwen4_hc_inject_upmix_bm32_mxfp8_2cta")?,
                load("qwen4_hc_inject_upmix_bm32_mxfp8_2cta_amax")?,
            ],
            mxfp8_gate: [
                load("qwen4_hc_inject_upmix_bm32_mxfp8_2cta_mxfp8_gate")?,
                load("qwen4_hc_inject_upmix_bm32_mxfp8_2cta_mxfp8_gate_amax")?,
            ],
        })
    })
}

fn bm32_mxfp8_combine_kernels(gpu: &dyn GpuBackend) -> Option<Bm64CombineKernels> {
    static K: OnceLock<Option<Bm64CombineKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM32_MXFP8)
                .ok()?;
            Some(k)
        };
        Some(Bm64CombineKernels {
            mx_gatepre: [
                load("qwen4_hc_combine_inject_upmix_bm32_mxfp8_2cta_mx_gatepre")?,
                load("qwen4_hc_combine_inject_upmix_bm32_mxfp8_2cta_mx_gatepre_amax")?,
            ],
        })
    })
}

fn bm32_nvfp4_upmix_kernels(gpu: &dyn GpuBackend) -> Option<Bm64UpmixKernels> {
    static K: OnceLock<Option<Bm64UpmixKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM32_NVFP4)
                .ok()?;
            Some(k)
        };
        Some(Bm64UpmixKernels {
            plain: [
                load("qwen4_hc_inject_upmix_bm32_nvfp4_2cta")?,
                load("qwen4_hc_inject_upmix_bm32_nvfp4_2cta_amax")?,
            ],
            mxfp8_gate: [
                load("qwen4_hc_inject_upmix_bm32_nvfp4_2cta_mxfp8_gate")?,
                load("qwen4_hc_inject_upmix_bm32_nvfp4_2cta_mxfp8_gate_amax")?,
            ],
        })
    })
}

fn bm32_nvfp4_combine_kernels(gpu: &dyn GpuBackend) -> Option<Bm64CombineKernels> {
    static K: OnceLock<Option<Bm64CombineKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM32_NVFP4)
                .ok()?;
            Some(k)
        };
        Some(Bm64CombineKernels {
            mx_gatepre: [
                load("qwen4_hc_combine_inject_upmix_bm32_nvfp4_2cta_mx_gatepre")?,
                load("qwen4_hc_combine_inject_upmix_bm32_nvfp4_2cta_mx_gatepre_amax")?,
            ],
        })
    })
}

fn bm64_combine_kernels(gpu: &dyn GpuBackend) -> Option<Bm64CombineKernels> {
    static K: OnceLock<Option<Bm64CombineKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_BM64_W4)
                .ok()?;
            Some(k)
        };
        Some(Bm64CombineKernels {
            mx_gatepre: [
                load("qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre")?,
                load("qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre_amax")?,
            ],
        })
    })
}

#[derive(Clone, Copy)]
struct StripW4Kernels {
    plain: KernelHandle,
    inject: KernelHandle,
    combine_moe: [KernelHandle; 2],
}

fn strip_w4_kernels(gpu: &dyn GpuBackend) -> Option<StripW4Kernels> {
    static K: OnceLock<Option<StripW4Kernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_STRIP_W4)
                .ok()?;
            Some(k)
        };
        Some(StripW4Kernels {
            plain: load("qwen4_hc_inject_gemm_ws_gatepre_w4")?,
            inject: load("qwen4_hc_inject_gemm_ws_l2h_gatepre_w4")?,
            combine_moe: [
                load("qwen4_hc_combine_moe_gemm_gatepre_w4")?,
                load("qwen4_hc_combine_moe_gemm_mx_gatepre_w4")?,
            ],
        })
    })
}

type StripGrowOnly = Mutex<Option<(usize, DevicePtr)>>;
static STRIP_FORMED_ROWS: StripGrowOnly = Mutex::new(None);
static STRIP_ROUTING_COPY: StripGrowOnly = Mutex::new(None);

fn grow_combine_scratch(
    cell: &StripGrowOnly,
    gpu: &dyn GpuBackend,
    bytes: usize,
    selector: &str,
    what: &str,
) -> Result<DevicePtr> {
    let mut guard = cell
        .lock()
        .map_err(|_| anyhow::anyhow!("{selector}: {what} scratch poisoned"))?;
    if let Some((have, ptr)) = *guard
        && have >= bytes
    {
        return Ok(ptr);
    }
    let ptr = gpu
        .alloc(bytes)
        .with_context(|| format!("{selector}: alloc {bytes} B for {what}"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, bytes, what)?;
    *guard = Some((bytes, ptr));
    Ok(ptr)
}

fn grow_strip_scratch(
    cell: &StripGrowOnly,
    gpu: &dyn GpuBackend,
    bytes: usize,
    what: &str,
) -> Result<DevicePtr> {
    grow_combine_scratch(cell, gpu, bytes, STRIP_ROWS_SELECTOR, what)
}

fn parse_cluster_fuse(value: Option<&str>) -> Result<bool, String> {
    let Some(raw) = value else {
        return Ok(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => Ok(false),
        "1" | "on" | "true" => Ok(true),
        other => Err(format!(
            "{CLUSTER_FUSE_SELECTOR}: unknown value {other:?} \
             (expected 0, 1, off, on, false, true)"
        )),
    }
}

fn cluster_fuse_enabled() -> Result<bool> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_cluster_fuse(std::env::var(CLUSTER_FUSE_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn cluster_fuse_eligible(
    enabled: bool,
    level: u32,
    strip_cap: usize,
    num_tokens: usize,
    sel: super::prefill_inject::Selection,
) -> bool {
    enabled
        && level >= 2
        && strip_cap == 0
        && num_tokens >= CLUSTER_FUSE_MIN_ROWS
        && sel.injgemm
        && sel.gatepre
        && sel.upmix
}

fn parse_strip_rows(value: Option<&str>) -> Result<usize, String> {
    let Some(raw) = value else {
        return Ok(DEFAULT_STRIP_ROWS);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(DEFAULT_STRIP_ROWS);
    }
    if trimmed.eq_ignore_ascii_case("off") {
        return Ok(0);
    }
    trimmed.parse::<usize>().map_err(|_| {
        format!(
            "{STRIP_ROWS_SELECTOR}: invalid strip row count {trimmed:?} \
             (expected 0, off, or a positive integer)"
        )
    })
}

fn strip_rows_cap() -> Result<usize> {
    static CAP: OnceLock<Result<usize, String>> = OnceLock::new();
    CAP.get_or_init(|| parse_strip_rows(std::env::var(STRIP_ROWS_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(crate) fn fused_prep_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var(FUSED_PREP_SELECTOR).ok().as_deref() == Some("1"))
}

fn prep_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_fused", "qwen4_hc_prep_weights")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FUSED_PREP_SELECTOR}: kernel unavailable: {e}"))
}

fn prep_up_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_fused", "qwen4_hc_prep_up")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FUSED_PREP_SELECTOR}: kernel unavailable: {e}"))
}

thread_local! {
    /// Row count whose per-stream GEMM configuration a cropped preparation
    /// must reproduce (`ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`).
    static REFERENCE_ROWS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// The row count that selects HC_FUSED's per-stream GEMM order: the injgemm
/// admission ([`super::prefill_inject_gemm_admit::gemm_order_verified`]) and
/// the cuBLASLt pinned `ref_m`. `rows` itself unless a [`ReferenceRowsGuard`]
/// is active, so a tail of a chunk runs the full chunk's arithmetic.
pub(crate) fn reference_rows(rows: usize) -> usize {
    REFERENCE_ROWS.with(std::cell::Cell::get).unwrap_or(rows)
}

/// Scoped [`reference_rows`] override.
pub(crate) struct ReferenceRowsGuard(Option<usize>);

impl ReferenceRowsGuard {
    pub(crate) fn enter(rows: usize) -> Self {
        Self(REFERENCE_ROWS.with(|c| c.replace(Some(rows))))
    }
}

impl Drop for ReferenceRowsGuard {
    fn drop(&mut self) {
        REFERENCE_ROWS.with(|c| c.set(self.0));
    }
}

/// 0 = off, 1 = fused down/inject + rstd mix, 2 = also the fused up+mix kernel.
pub(crate) fn level() -> u32 {
    static L: OnceLock<u32> = OnceLock::new();
    *L.get_or_init(|| match std::env::var(SELECTOR).ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

#[derive(Clone, Copy)]
struct Kernels {
    rstd: KernelHandle,
    fold: KernelHandle,
    combine: KernelHandle,
    mix_rstd: KernelHandle,
    up_mix: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let k = |name: &str| {
            gpu.kernel("qwen4_hc_fused", name)
                .map_err(|e| e.to_string())
        };
        Ok(Kernels {
            rstd: k("qwen4_hc_rstd")?,
            fold: k("qwen4_hc_fold")?,
            combine: k("qwen4_hc_combine")?,
            mix_rstd: k("qwen4_hc_mix_rstd")?,
            up_mix: k("qwen4_hc_up_mix_mma")?,
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernel unavailable: {e}"))
}

/// Grow-only process scratch, used only on the single prefill stream.
#[derive(Clone, Copy)]
struct Scratch {
    rows: usize,
    weights: DevicePtr, // dequantized down|inject rows [(R+hc), hc*H] bf16
    folded: DevicePtr,  // [hc][R+hc][H] bf16
    up: DevicePtr,      // [hc*H, R] bf16
    up_bytes: usize,
    rstd: DevicePtr,    // [rows, hc] f32
    down: DevicePtr,    // [rows, R] bf16 (disjoint from arena gate_logits / token_to_perm)
    partial: DevicePtr, // [hc][rows][R+hc] f32
    partial_bytes: usize,
}

fn scratch(
    gpu: &dyn GpuBackend,
    rows: usize,
    h: usize,
    hc: usize,
    rank: usize,
    need_full_partial: bool,
    need_separate_up: bool,
) -> Result<Scratch> {
    let rows = rows.max(crate::layers::qwen4_prefill_moe::max_chunk_rows());
    let n = rank + hc;
    let fused_prep = fused_prep_selected();
    let required_up_bytes = if !fused_prep || need_separate_up {
        hc * h * rank * 2
    } else {
        16
    };
    let required_partial_bytes = if need_full_partial {
        hc * rows * n * 4
    } else {
        let injgemm = super::prefill_inject::selection()
            .map(|sel| sel.injgemm)
            .unwrap_or(false);
        if injgemm { 16 } else { hc * rows * n * 4 }
    };
    static S: Mutex<Option<Scratch>> = Mutex::new(None);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scratch poisoned"))?;
    let alloc = |bytes: usize| -> Result<DevicePtr> {
        let ptr = gpu
            .alloc(bytes)
            .with_context(|| format!("{SELECTOR}: alloc {bytes} B"))?;
        // ATLAS_PREFILL_SCRATCH_INIT: no first-touch read of a previous owner's bytes.
        spark_runtime::scratch_init::init_grown(gpu, ptr, bytes, "hc_fused")?;
        Ok(ptr)
    };
    if let Some(mut s) = guard.as_ref().copied()
        && s.rows >= rows
    {
        let mut grown = false;
        if s.partial_bytes < required_partial_bytes {
            s.partial = alloc(required_partial_bytes)?;
            s.partial_bytes = required_partial_bytes;
            grown = true;
        }
        if s.up_bytes < required_up_bytes {
            s.up = alloc(required_up_bytes)?;
            s.up_bytes = required_up_bytes;
            grown = true;
        }
        if grown {
            tracing::info!(
                "{SELECTOR}: scratch grown in-place rows={} up_bytes={} partial_bytes={}",
                s.rows,
                s.up_bytes,
                s.partial_bytes
            );
            *guard = Some(s);
        }
        return Ok(s);
    }
    // Earlier blocks are intentionally leaked (grow-only, in-flight safety).
    let s = Scratch {
        rows,
        weights: alloc(if fused_prep { 16 } else { n * hc * h * 2 })?,
        folded: alloc(hc * n * h * 2)?,
        up: alloc(required_up_bytes)?,
        up_bytes: required_up_bytes,
        rstd: alloc(rows * hc * 4)?,
        down: alloc(rows * rank * 2)?,
        partial: alloc(required_partial_bytes)?,
        partial_bytes: required_partial_bytes,
    };
    tracing::info!(
        "{SELECTOR}: scratch allocated rows={rows} up_bytes={required_up_bytes} partial_bytes={required_partial_bytes}"
    );
    *guard = Some(s);
    Ok(s)
}

/// True when strip `k`'s `upmix` write to `mixed[row_start .. row_start + S]`
/// cannot clobber any input read by later strips' `pending` injection.
fn strip_pending_safe(
    pending: Option<&PendingInject>,
    hyper: DevicePtr,
    residual: DevicePtr,
    mixed: DevicePtr,
    num_tokens: usize,
    h: usize,
) -> bool {
    let Some(p) = pending else {
        return true;
    };
    if p.hyper != hyper || p.residual != residual || p.rows != num_tokens {
        return false;
    }
    let mixed_bytes = num_tokens * h * 2;
    let row_aligned_or_disjoint = |ptr: DevicePtr, bytes: usize| -> bool {
        ptr.is_null() || ptr == mixed || disjoint(ptr, bytes, mixed, mixed_bytes)
    };
    let must_be_disjoint = |ptr: DevicePtr, bytes: usize| -> bool {
        ptr.is_null() || disjoint(ptr, bytes, mixed, mixed_bytes)
    };
    if let Some(c) = p.combine {
        if !c.slot_corrections.is_null() {
            // slot_r2 preflight/commit tracks full-chunk row counts once per layer.
            return false;
        }
        let k = c.topk as usize;
        let expert_elem_bytes = if !c.slot_scales.is_null() { 1 } else { 2 };
        let gate_ok = if !c.gate_scalars.is_null() {
            must_be_disjoint(c.gate_scalars, num_tokens * 4)
        } else {
            row_aligned_or_disjoint(c.gate_input, num_tokens * h * 2)
                && must_be_disjoint(c.shared_gate, h * 2)
        };
        must_be_disjoint(c.expert_out, num_tokens * k * h * expert_elem_bytes)
            && must_be_disjoint(c.slot_scales, num_tokens * k * h / 32)
            && must_be_disjoint(c.token_to_perm, num_tokens * k * 4)
            && must_be_disjoint(c.topk_weights, num_tokens * k * 4)
            && row_aligned_or_disjoint(c.shared_out, num_tokens * h * 2)
            && gate_ok
    } else {
        row_aligned_or_disjoint(p.core, num_tokens * h * 2)
    }
}

/// Narrow `p` to strip `[row_start, row_start + strip_rows)` while keeping
/// global slot-indexed MoE buffers (`expert_out`, `slot_scales`,
/// `slot_corrections`) and static weights (`shared_gate`) unshifted.
fn slice_pending(
    p: &PendingInject,
    row_start: usize,
    strip_rows: usize,
    h: usize,
    r: usize,
) -> PendingInject {
    let row_bytes = r * 2;
    let h_bytes = h * 2;
    PendingInject {
        hyper: p.hyper.offset(row_start * row_bytes),
        core: if p.core.is_null() {
            DevicePtr::NULL
        } else {
            p.core.offset(row_start * h_bytes)
        },
        residual: p.residual.offset(row_start * row_bytes),
        rows: strip_rows,
        combine: p.combine.map(|c| {
            let topk_bytes = c.topk as usize * 4;
            MoeCombine {
                slot_scales: c.slot_scales,
                slot_corrections: c.slot_corrections,
                expert_out: c.expert_out,
                token_to_perm: c.token_to_perm.offset(row_start * topk_bytes),
                topk_weights: c.topk_weights.offset(row_start * topk_bytes),
                shared_out: c.shared_out.offset(row_start * h_bytes),
                gate_input: if c.gate_input.is_null() {
                    DevicePtr::NULL
                } else {
                    c.gate_input.offset(row_start * h_bytes)
                },
                shared_gate: c.shared_gate,
                gate_scalars: if c.gate_scalars.is_null() {
                    DevicePtr::NULL
                } else {
                    c.gate_scalars.offset(row_start * 4)
                },
                topk: c.topk,
            }
        }),
    }
}

/// Narrow a full-chunk MXFP8 `(xq, xs, ss_out)` request to strip
/// `[row_start, row_start + strip_rows)` while keeping static `shared_gate`
/// unshifted.
fn slice_mxfp8_request(
    req: UpmixMxfp8Request,
    row_start: usize,
    strip_rows: usize,
    h: usize,
) -> UpmixMxfp8Request {
    UpmixMxfp8Request {
        xq: req.xq.offset(row_start * h),
        xs: req.xs.offset(row_start * (h / 32)),
        shared_gate: req.shared_gate,
        ss_out: if req.ss_out.is_null() {
            DevicePtr::NULL
        } else {
            req.ss_out.offset(row_start * 4)
        },
        rows: strip_rows,
        stream: req.stream,
    }
}

#[derive(Clone, Copy)]
struct StripMxfp8CheckBufs {
    hyper: DevicePtr,
    residual: DevicePtr,
    down: DevicePtr,
    rstd: DevicePtr,
    folded: DevicePtr,
    num_tokens: usize,
    h: usize,
    hc: usize,
    rank: usize,
}

/// True when strip `k`'s MXFP8 `(xq, xs, ss_out)` writes cannot clobber any
/// buffer read by later strips (or vice versa) in strip-mined mode.
fn strip_mxfp8_safe(
    req: UpmixMxfp8Request,
    pending: Option<&PendingInject>,
    b: &StripMxfp8CheckBufs,
) -> bool {
    let r = b.h * b.hc;
    let n = b.rank + b.hc;
    let Some(hyper_bytes) = b.num_tokens.checked_mul(r * 2) else {
        return false;
    };
    let Some(xq_bytes) = b.num_tokens.checked_mul(b.h) else {
        return false;
    };
    let Some(xs_bytes) = b.num_tokens.checked_mul(b.h / 32) else {
        return false;
    };
    let Some(ss_bytes) = b.num_tokens.checked_mul(4) else {
        return false;
    };
    let Some(down_bytes) = b.num_tokens.checked_mul(b.rank * 2) else {
        return false;
    };
    let Some(rstd_bytes) = b.num_tokens.checked_mul(b.hc * 4) else {
        return false;
    };
    let folded_bytes = b.hc * n * b.h * 2;
    let mx_writes = [
        (req.xq, xq_bytes),
        (req.xs, xs_bytes),
        (req.ss_out, if req.ss_out.is_null() { 0 } else { ss_bytes }),
    ];
    let core_bufs = [
        (b.hyper, hyper_bytes),
        (b.residual, hyper_bytes),
        (b.down, down_bytes),
        (b.rstd, rstd_bytes),
        (b.folded, folded_bytes),
    ];
    let disjoint_nonempty = |a: DevicePtr, al: usize, c: DevicePtr, cl: usize| -> bool {
        a.is_null() || c.is_null() || al == 0 || cl == 0 || disjoint(a, al, c, cl)
    };
    if !mx_writes.iter().all(|&(m, ml)| {
        core_bufs
            .iter()
            .all(|&(c, cl)| disjoint_nonempty(m, ml, c, cl))
    }) {
        return false;
    }
    if !req.ss_out.is_null()
        && !req.shared_gate.is_null()
        && !core_bufs
            .iter()
            .all(|&(c, cl)| disjoint_nonempty(req.shared_gate, b.h * 2, c, cl))
    {
        return false;
    }
    let Some(p) = pending else {
        return true;
    };
    if let Some(c) = p.combine {
        let k = c.topk as usize;
        let expert_elem_bytes = if !c.slot_scales.is_null() { 1 } else { 2 };
        let gate_read = if !c.gate_scalars.is_null() {
            (c.gate_scalars, b.num_tokens * 4)
        } else {
            (c.gate_input, b.num_tokens * b.h * 2)
        };
        let reads = [
            (c.expert_out, b.num_tokens * k * b.h * expert_elem_bytes),
            (c.slot_scales, b.num_tokens * k * b.h / 32),
            (c.token_to_perm, b.num_tokens * k * 4),
            (c.topk_weights, b.num_tokens * k * 4),
            (c.shared_out, b.num_tokens * b.h * 2),
            gate_read,
            (
                c.shared_gate,
                if c.gate_scalars.is_null() { b.h * 2 } else { 0 },
            ),
        ];
        mx_writes.iter().all(|&(m, ml)| {
            reads
                .iter()
                .all(|&(rd, rl)| disjoint_nonempty(m, ml, rd, rl))
        })
    } else {
        mx_writes
            .iter()
            .all(|&(m, ml)| disjoint_nonempty(m, ml, p.core, b.num_tokens * b.h * 2))
    }
}

#[derive(Clone, Copy)]
struct Bm64UpmixAdmit {
    hyper: DevicePtr,
    residual: DevicePtr,
    folded: DevicePtr,
    mixed: DevicePtr,
    norm_w: DevicePtr,
    up_packed: DevicePtr,
    up_scales: DevicePtr,
    scale2_down: f32,
    scale2_inject: f32,
    alpha_up: f32,
    amax_slot: Option<DevicePtr>,
    mxfp8_req: Option<UpmixMxfp8Request>,
    num_tokens: usize,
    h: usize,
    hc: usize,
    rank: usize,
}

/// Pure geometry, alignment, pending-Core, and aliasing admission for
/// `qwen4_hc_inject_upmix_bm64_w4[_mxfp8_gate][_amax]`.
///
/// Returns `Some((core_ptr, admitted_mxfp8))` when admitted. If `a.mxfp8_req`
/// is `Some(_)` and fails MXFP8 validation, returns `None` so the caller falls
/// back rather than silently dropping W21-F MXFP8 fusion.
fn bm64_upmix_admit(
    a: &Bm64UpmixAdmit,
    pending: Option<&PendingInject>,
) -> Option<(DevicePtr, Option<UpmixMxfp8Request>)> {
    if (a.h, a.hc, a.rank) != (2560, 4, 320)
        || a.num_tokens == 0
        || u32::try_from(a.num_tokens).is_err()
    {
        return None;
    }
    let p = pending?;
    if p.combine.is_some()
        || p.core.is_null()
        || p.hyper != a.hyper
        || p.residual != a.residual
        || p.rows != a.num_tokens
    {
        return None;
    }
    let aligned16 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(16);
    let aligned8 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(8);
    let aligned4 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(4);
    if !aligned16(a.hyper)
        || !aligned16(p.core)
        || !aligned16(a.folded)
        || !aligned16(a.mixed)
        || !aligned8(a.up_packed)
        || a.up_scales.is_null()
        || !aligned4(a.residual)
        || !aligned4(a.norm_w)
        || !a.scale2_down.is_finite()
        || !a.scale2_inject.is_finite()
        || !a.alpha_up.is_finite()
        || !a.amax_slot.is_none_or(aligned4)
    {
        return None;
    }
    let r = a.h * a.hc;
    let n = a.rank + a.hc;
    let hyper_bytes = a.num_tokens.checked_mul(r * 2)?;
    let mixed_bytes = a.num_tokens.checked_mul(a.h * 2)?;
    let folded_bytes = a.hc * n * a.h * 2;
    let disjoint_nonempty = |x: DevicePtr, xl: usize, y: DevicePtr, yl: usize| -> bool {
        x.is_null() || y.is_null() || xl == 0 || yl == 0 || disjoint(x, xl, y, yl)
    };
    if !disjoint(a.hyper, hyper_bytes, a.residual, hyper_bytes)
        || !disjoint(a.hyper, hyper_bytes, a.mixed, mixed_bytes)
        || !disjoint(a.residual, hyper_bytes, a.mixed, mixed_bytes)
        || !disjoint(a.folded, folded_bytes, a.hyper, hyper_bytes)
        || !disjoint(a.folded, folded_bytes, a.residual, hyper_bytes)
        || !disjoint(a.folded, folded_bytes, a.mixed, mixed_bytes)
        || !disjoint(p.core, mixed_bytes, a.hyper, hyper_bytes)
        || !disjoint(p.core, mixed_bytes, a.residual, hyper_bytes)
        || !disjoint(p.core, mixed_bytes, a.folded, folded_bytes)
        || (p.core != a.mixed && !disjoint(p.core, mixed_bytes, a.mixed, mixed_bytes))
    {
        return None;
    }
    let Some(req) = a.mxfp8_req else {
        return Some((p.core, None));
    };
    if !aligned8(req.xq) || req.xs.is_null() {
        return None;
    }
    if !req.ss_out.is_null()
        && (!req.ss_out.0.is_multiple_of(4)
            || (!req.shared_gate.is_null() && !req.shared_gate.0.is_multiple_of(2)))
    {
        return None;
    }
    let xq_bytes = a.num_tokens.checked_mul(a.h)?;
    let xs_bytes = a.num_tokens.checked_mul(a.h / 32)?;
    let ss_bytes = if req.ss_out.is_null() {
        0
    } else {
        a.num_tokens.checked_mul(4)?
    };
    let writes = [
        (a.hyper, hyper_bytes),
        (a.residual, hyper_bytes),
        (a.mixed, mixed_bytes),
        (req.xq, xq_bytes),
        (req.xs, xs_bytes),
        (req.ss_out, ss_bytes),
    ];
    for i in 0..writes.len() {
        for j in (i + 1)..writes.len() {
            if !disjoint_nonempty(writes[i].0, writes[i].1, writes[j].0, writes[j].1) {
                return None;
            }
        }
    }
    let mx_writes = [
        (req.xq, xq_bytes),
        (req.xs, xs_bytes),
        (req.ss_out, ss_bytes),
    ];
    let reads = [(a.folded, folded_bytes), (p.core, mixed_bytes)];
    if !mx_writes.iter().all(|&(m, ml)| {
        reads
            .iter()
            .all(|&(rd, rl)| disjoint_nonempty(m, ml, rd, rl))
    }) {
        return None;
    }
    if !req.ss_out.is_null()
        && !req.shared_gate.is_null()
        && !writes
            .iter()
            .all(|&(w, wl)| disjoint_nonempty(req.shared_gate, a.h * 2, w, wl))
    {
        return None;
    }
    Some((p.core, Some(req)))
}

#[derive(Clone, Copy)]
struct Bm64CombineGates {
    upmix_enabled: bool,
    combine_enabled: bool,
    level: u32,
    strip_cap: usize,
    sel: super::prefill_inject::Selection,
    oracle_or_dump: bool,
    sm_count: u32,
}

/// Pure policy, geometry, alignment, pending-Combine, and aliasing admission
/// for `qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre[_amax]`.
///
/// Fails closed whenever BM=64 upmix or combine is disabled, `level < 2`,
/// strip-mining is active, `num_tokens < bm64_upmix_min_rows_for_sm(sm_count)`,
/// `!gatepre`, `postsplit` is selected, oracle/dump capture is active, geometry
/// differs from `(2560, 4, 320)`, `mxfp8_req` is armed, `combine` uses non-MXFP8
/// slots or R2 corrections, or any pointer fails alignment or disjointness checks.
fn bm64_combine_admit(
    gates: &Bm64CombineGates,
    a: &Bm64UpmixAdmit,
    pending: Option<&PendingInject>,
) -> Option<MoeCombine> {
    if !bm64_upmix_eligible_for_sm(
        gates.upmix_enabled,
        gates.level,
        gates.strip_cap,
        a.num_tokens,
        gates.sel,
        gates.sm_count,
    ) || !gates.combine_enabled
        || !gates.sel.gatepre
        || gates.sel.postsplit
        || gates.oracle_or_dump
    {
        return None;
    }
    if (a.h, a.hc, a.rank) != (2560, 4, 320)
        || u32::try_from(a.num_tokens).is_err()
        || a.mxfp8_req.is_some()
    {
        return None;
    }
    let p = pending?;
    let c = p.combine?;
    if p.hyper != a.hyper
        || p.residual != a.residual
        || p.rows != a.num_tokens
        || c.topk != STRIP_W4_TOPK
        || c.slot_scales.is_null()
        || !c.slot_corrections.is_null()
        || c.gate_scalars.is_null()
    {
        return None;
    }
    let aligned16 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(16);
    let aligned8 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(8);
    let aligned4 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(4);
    if !aligned16(a.hyper)
        || !aligned16(a.folded)
        || !aligned16(a.mixed)
        || !aligned8(a.up_packed)
        || a.up_scales.is_null()
        || !aligned4(a.residual)
        || !aligned4(a.norm_w)
        || !a.scale2_down.is_finite()
        || !a.scale2_inject.is_finite()
        || !a.alpha_up.is_finite()
        || !a.amax_slot.is_none_or(aligned4)
        || !aligned16(c.expert_out)
        || !aligned16(c.shared_out)
        || !aligned4(c.slot_scales)
        || !aligned4(c.token_to_perm)
        || !aligned4(c.topk_weights)
        || !aligned4(c.gate_scalars)
    {
        return None;
    }
    let r = a.h * a.hc;
    let n = a.rank + a.hc;
    let k = c.topk as usize;
    let hyper_bytes = a.num_tokens.checked_mul(r * 2)?;
    let mixed_bytes = a.num_tokens.checked_mul(a.h * 2)?;
    let folded_bytes = a.hc * n * a.h * 2;
    let expert_bytes = a.num_tokens.checked_mul(k)?.checked_mul(a.h)?;
    let scales_bytes = a.num_tokens.checked_mul(k)?.checked_mul(a.h / 32)?;
    let topk_bytes = a.num_tokens.checked_mul(k)?.checked_mul(4)?;
    let gate_bytes = a.num_tokens.checked_mul(4)?;
    if !disjoint(a.hyper, hyper_bytes, a.residual, hyper_bytes)
        || !disjoint(a.hyper, hyper_bytes, a.mixed, mixed_bytes)
        || !disjoint(a.residual, hyper_bytes, a.mixed, mixed_bytes)
        || !disjoint(a.folded, folded_bytes, a.hyper, hyper_bytes)
        || !disjoint(a.folded, folded_bytes, a.residual, hyper_bytes)
        || !disjoint(a.folded, folded_bytes, a.mixed, mixed_bytes)
    {
        return None;
    }
    let targets = [
        (a.hyper, hyper_bytes),
        (a.residual, hyper_bytes),
        (a.mixed, mixed_bytes),
        (a.folded, folded_bytes),
    ];
    let strict_reads = [
        (c.expert_out, expert_bytes),
        (c.slot_scales, scales_bytes),
        (c.topk_weights, topk_bytes),
        (c.gate_scalars, gate_bytes),
    ];
    if !strict_reads
        .iter()
        .all(|&(rd, rl)| targets.iter().all(|&(w, wl)| disjoint(rd, rl, w, wl)))
    {
        return None;
    }
    if !disjoint(c.token_to_perm, topk_bytes, a.folded, folded_bytes)
        || !disjoint(c.shared_out, mixed_bytes, a.hyper, hyper_bytes)
        || !disjoint(c.shared_out, mixed_bytes, a.residual, hyper_bytes)
        || !disjoint(c.shared_out, mixed_bytes, a.folded, folded_bytes)
        || (c.shared_out != a.mixed && !disjoint(c.shared_out, mixed_bytes, a.mixed, mixed_bytes))
    {
        return None;
    }
    Some(c)
}

#[derive(Clone, Copy)]
struct StripW4Admit {
    hyper: DevicePtr,
    residual: DevicePtr,
    down: DevicePtr,
    rstd: DevicePtr,
    down_packed: DevicePtr,
    down_scales: DevicePtr,
    scale2_down: f32,
    inject_packed: DevicePtr,
    inject_scales: DevicePtr,
    scale2_inject: f32,
    norm_w: DevicePtr,
    num_tokens: usize,
    h: usize,
    hc: usize,
    rank: usize,
}

/// Pure admission for inline-NVFP4 `folded` strip kernels (`hc_w4::*`).
fn strip_w4_folded_admit(
    a: &StripW4Admit,
    pending: Option<&PendingInject>,
    postpre_selected: bool,
) -> bool {
    if (a.h, a.hc, a.rank) != (2560, 4, 320)
        || a.num_tokens == 0
        || u32::try_from(a.num_tokens).is_err()
    {
        return false;
    }
    let aligned16 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(16);
    let aligned4 = |ptr: DevicePtr| !ptr.is_null() && ptr.0.is_multiple_of(4);
    if !aligned16(a.hyper)
        || !aligned16(a.norm_w)
        || !aligned4(a.residual)
        || !aligned4(a.down)
        || !aligned4(a.rstd)
        || !aligned4(a.down_packed)
        || a.down_scales.is_null()
        || !a.scale2_down.is_finite()
        || !aligned4(a.inject_packed)
        || a.inject_scales.is_null()
        || !a.scale2_inject.is_finite()
    {
        return false;
    }
    let Some(p) = pending else {
        return true;
    };
    if p.hyper != a.hyper || p.residual != a.residual || p.rows != a.num_tokens {
        return false;
    }
    let Some(c) = p.combine else {
        return p.core.is_null() || aligned16(p.core);
    };
    if !postpre_selected
        || c.topk != STRIP_W4_TOPK
        || c.gate_scalars.is_null()
        || !c.slot_corrections.is_null()
        || !aligned16(c.expert_out)
        || !aligned16(c.shared_out)
        || (!c.slot_scales.is_null() && !c.slot_scales.0.is_multiple_of(4))
        || !aligned4(c.token_to_perm)
        || !aligned4(c.topk_weights)
        || !aligned4(c.gate_scalars)
    {
        return false;
    }
    let r = a.h * a.hc;
    let k = c.topk as usize;
    let expert_elem_bytes = if !c.slot_scales.is_null() { 1 } else { 2 };
    let writes = [
        (a.hyper, a.num_tokens * r * 2),
        (a.residual, a.num_tokens * r * 2),
        (a.down, a.num_tokens * a.rank * 2),
        (a.rstd, a.num_tokens * a.hc * 4),
    ];
    let reads = [
        (c.expert_out, a.num_tokens * k * a.h * expert_elem_bytes),
        (c.slot_scales, a.num_tokens * k * a.h / 32),
        (c.topk_weights, a.num_tokens * k * 4),
        (c.shared_out, a.num_tokens * a.h * 2),
        (c.gate_scalars, a.num_tokens * 4),
    ];
    reads.iter().all(|&(rd, rl)| {
        rd.is_null() || rl == 0 || writes.iter().all(|&(w, wl)| disjoint(rd, rl, w, wl))
    })
}

fn launch_level2_upmix(
    upmix_launch: &UpmixLaunch,
    upmix: bool,
    h: usize,
    rows_usize: usize,
    fallback_up_mix: KernelHandle,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    // ATLAS_QWEN4_PREFILL_MHC_ROOF (default off): the same up+mix,
    // byte-identical, on the roof kernels where admitted.
    let roof = super::prefill_upmix_roof::launch(upmix_launch, upmix, h, rows_usize, gpu, stream)?;
    if !roof {
        let rows = u32::try_from(rows_usize)?;
        let up_mix = match upmix_launch.amax_slot {
            Some(_) => super::prefill_upmix::hcfused_amax_kernel(gpu)?,
            None if upmix => super::prefill_upmix::hcfused_kernel(gpu)?,
            None => fallback_up_mix,
        };
        let launch = KernelLaunch::new(gpu, up_mix)
            .grid([(h / 32) as u32, div_ceil(rows, 64), 1])
            .block([256, 1, 1])
            .arg_ptr(upmix_launch.act)
            .arg_ptr(upmix_launch.up)
            .arg_f32(upmix_launch.alpha)
            .arg_ptr(upmix_launch.hyper)
            .arg_ptr(upmix_launch.rstd)
            .arg_ptr(upmix_launch.norm_w)
            .arg_ptr(upmix_launch.mixed)
            .arg_u32(rows);
        match upmix_launch.amax_slot {
            Some(slot) => launch.arg_ptr(slot).launch(stream)?,
            None => launch.launch(stream)?,
        }
    }
    Ok(())
}

fn prep_weights_threads(h: usize, hc: usize, rank: usize, include_up: bool) -> u32 {
    let n = rank + hc;
    let fold_threads = ((hc * n * (h / 16)) / 4) as u32;
    if include_up {
        fold_threads + (hc * h * (rank / 16)) as u32
    } else {
        fold_threads
    }
}

#[derive(Clone, Copy)]
struct StripW4Launch {
    hyper: DevicePtr,
    residual: DevicePtr,
    rstd: DevicePtr,
    down: DevicePtr,
    scale2_down: f32,
    scale2_inject: f32,
    strip_rows: usize,
    eps: f32,
}

#[derive(Clone, Copy)]
struct Bm64CombineLaunch {
    hyper: DevicePtr,
    residual: DevicePtr,
    folded: DevicePtr,
    mixed: DevicePtr,
    amax_slot: Option<DevicePtr>,
    scale2_down: f32,
    scale2_inject: f32,
    num_tokens: usize,
    eps: f32,
}

impl Qwen4HyperConnection {
    fn launch_prep_weights(
        &self,
        gpu: &dyn GpuBackend,
        pk: KernelHandle,
        inject: &crate::weight_map::QuantizedWeight,
        folded_out: DevicePtr,
        up_out: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        let (h, hc, rank) = (self.hidden_size, self.hc_count, self.rank);
        let threads = prep_weights_threads(h, hc, rank, up_out.is_some());
        KernelLaunch::new(gpu, pk)
            .grid([div_ceil(threads, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.down.weight)
            .arg_ptr(self.down.weight_scale)
            .arg_ptr(inject.weight)
            .arg_ptr(inject.weight_scale)
            .arg_ptr(self.norm.weight)
            .arg_ptr(folded_out)
            .arg_ptr(self.up.weight)
            .arg_ptr(self.up.weight_scale)
            .arg_ptr(up_out.unwrap_or(DevicePtr::NULL))
            .arg_u32(rank as u32)
            .arg_u32(h as u32)
            .arg_u32(hc as u32)
            .launch(stream)
    }

    fn launch_strip_w4_phase1(
        &self,
        gpu: &dyn GpuBackend,
        wk: StripW4Kernels,
        inject: &crate::weight_map::QuantizedWeight,
        pending: Option<&PendingInject>,
        l: &StripW4Launch,
        stream: u64,
    ) -> Result<()> {
        let rows_u32 = u32::try_from(l.strip_rows)?;
        let grid = [CLUSTER_STREAMS, div_ceil(rows_u32, CLUSTER_TILE_ROWS), 1];
        if let Some(c) = pending.and_then(|p| p.combine) {
            let mx = !c.slot_scales.is_null();
            let cap_rows = l
                .strip_rows
                .max(crate::layers::qwen4_prefill_moe::max_chunk_rows());
            let formed = grow_strip_scratch(
                &STRIP_FORMED_ROWS,
                gpu,
                cap_rows * self.hidden_size * 2,
                "strip w4 formed MoE rows",
            )?;
            let r = self.hidden_size * self.hc_count;
            let routing_bytes = l.strip_rows * (STRIP_W4_TOPK as usize) * 4;
            let outputs = [
                (l.down, l.strip_rows * self.rank * 2),
                (l.hyper, l.strip_rows * r * 2),
                (l.residual, l.strip_rows * r * 2),
                (l.rstd, l.strip_rows * self.hc_count * 4),
            ];
            let routing = if outputs
                .iter()
                .all(|&(w, wb)| disjoint(c.token_to_perm, routing_bytes, w, wb))
            {
                c.token_to_perm
            } else {
                let dst = grow_strip_scratch(
                    &STRIP_ROUTING_COPY,
                    gpu,
                    cap_rows * (STRIP_W4_TOPK as usize) * 4,
                    "strip w4 routing copy",
                )?;
                gpu.copy_d2d_async(c.token_to_perm, dst, routing_bytes, stream)?;
                dst
            };
            return KernelLaunch::new(gpu, wk.combine_moe[usize::from(mx)])
                .grid(grid)
                .block([CLUSTER_THREADS, 1, 1])
                .shared_mem(SMEM_BYTES_STRIP_W4)
                .arg_ptr(l.hyper)
                .arg_ptr(l.residual)
                .arg_ptr(self.down.weight)
                .arg_ptr(self.down.weight_scale)
                .arg_ptr(inject.weight)
                .arg_ptr(inject.weight_scale)
                .arg_ptr(self.norm.weight)
                .arg_ptr(l.rstd)
                .arg_ptr(l.down)
                .arg_ptr(formed)
                .arg_ptr(c.expert_out)
                .arg_ptr(c.slot_scales)
                .arg_ptr(routing)
                .arg_ptr(c.topk_weights)
                .arg_ptr(c.shared_out)
                .arg_ptr(c.gate_scalars)
                .arg_u32(rows_u32)
                .arg_u32(self.rank as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_u32(c.topk)
                .arg_f32(l.scale2_down)
                .arg_f32(l.scale2_inject)
                .arg_f32(l.eps)
                .launch(stream);
        }
        let core = pending.map_or(DevicePtr::NULL, |p| p.core);
        let handle = if core.is_null() { wk.plain } else { wk.inject };
        KernelLaunch::new(gpu, handle)
            .grid(grid)
            .block([CLUSTER_THREADS, 1, 1])
            .shared_mem(SMEM_BYTES_STRIP_W4)
            .arg_ptr(l.hyper)
            .arg_ptr(core)
            .arg_ptr(l.residual)
            .arg_ptr(self.down.weight)
            .arg_ptr(self.down.weight_scale)
            .arg_ptr(inject.weight)
            .arg_ptr(inject.weight_scale)
            .arg_ptr(self.norm.weight)
            .arg_ptr(l.rstd)
            .arg_ptr(l.down)
            .arg_u32(rows_u32)
            .arg_u32(self.rank as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .arg_f32(l.scale2_down)
            .arg_f32(l.scale2_inject)
            .arg_f32(l.eps)
            .launch(stream)
    }

    fn launch_bm64_combine(
        &self,
        gpu: &dyn GpuBackend,
        ck: Bm64CombineKernels,
        layout: ClusterFusedLayout,
        c: &MoeCombine,
        l: &Bm64CombineLaunch,
        stream: u64,
    ) -> Result<()> {
        let (h, hc, rank) = (self.hidden_size, self.hc_count, self.rank);
        let r = h * hc;
        let n = rank + hc;
        let rows = u32::try_from(l.num_tokens)?;
        let cap_rows = l
            .num_tokens
            .max(crate::layers::qwen4_prefill_moe::max_chunk_rows());
        let formed_bytes = cap_rows * h * 2;
        let formed = grow_combine_scratch(
            &STRIP_FORMED_ROWS,
            gpu,
            formed_bytes,
            layout.selector,
            "bm64 combine formed MoE rows",
        )?;
        let routing_bytes = l.num_tokens * (STRIP_W4_TOPK as usize) * 4;
        let writes = [
            (l.hyper, l.num_tokens * r * 2),
            (l.residual, l.num_tokens * r * 2),
            (l.mixed, l.num_tokens * h * 2),
            (formed, formed_bytes),
        ];
        let routing = if writes
            .iter()
            .all(|&(w, w_bytes)| disjoint(c.token_to_perm, routing_bytes, w, w_bytes))
        {
            c.token_to_perm
        } else {
            let dst = grow_combine_scratch(
                &STRIP_ROUTING_COPY,
                gpu,
                cap_rows * (STRIP_W4_TOPK as usize) * 4,
                layout.selector,
                "bm64 combine routing copy",
            )?;
            gpu.copy_d2d_async(c.token_to_perm, dst, routing_bytes, stream)?;
            dst
        };
        if let Some(slot) = l.amax_slot {
            gpu.memset_async(slot, 0, 4, stream)?;
        }
        let amax_idx = usize::from(l.amax_slot.is_some());
        let kl = KernelLaunch::new(gpu, ck.mx_gatepre[amax_idx])
            .grid([CLUSTER_STREAMS, div_ceil(rows, layout.tile_rows), 1])
            .block([layout.threads, 1, 1])
            .shared_mem(layout.smem_bytes)
            .arg_ptr(l.hyper)
            .arg_ptr(l.residual)
            .arg_ptr(formed)
            .arg_ptr(c.expert_out)
            .arg_ptr(c.slot_scales)
            .arg_ptr(routing)
            .arg_ptr(c.topk_weights)
            .arg_ptr(c.shared_out)
            .arg_ptr(c.gate_scalars)
            .arg_ptr(l.folded)
            .arg_ptr(self.up.weight)
            .arg_ptr(self.up.weight_scale)
            .arg_ptr(self.norm.weight)
            .arg_ptr(l.mixed)
            .arg_u32(rows)
            .arg_u32(rank as u32)
            .arg_u32(h as u32)
            .arg_u32(hc as u32)
            .arg_u32(c.topk)
            .arg_f32(l.scale2_down)
            .arg_f32(l.scale2_inject)
            .arg_f32(self.up.weight_scale_2)
            .arg_f32(l.eps);
        match l.amax_slot {
            Some(slot) => kl.arg_ptr(slot).launch(stream)?,
            None => kl.launch(stream)?,
        }
        super::prefill_upmix_roof::discard_dead_l2(
            [
                (l.folded, hc * n * h * 2),
                (DevicePtr::NULL, 0),
                (DevicePtr::NULL, 0),
                (DevicePtr::NULL, 0),
            ],
            gpu,
            stream,
        )?;
        if l.amax_slot.is_some() {
            use crate::layers::qwen4_fp8_act_amax as act_amax;
            act_amax::note_ready(act_amax::Ready {
                act: l.mixed.0,
                rows: l.num_tokens,
                cols: h,
                stream,
                producer: act_amax::Producer::HcUpmix,
            });
        }
        static LOGGED_BM64_COMBINE: std::sync::Once = std::sync::Once::new();
        LOGGED_BM64_COMBINE.call_once(|| {
            tracing::info!(
                "HC_PREFILL_MHC_BM64_COMBINE_ENGAGED selector={} rows={} amax={} bm32_mode={} nvfp4_2cta={}",
                layout.selector,
                l.num_tokens,
                l.amax_slot.is_some(),
                layout.bm32_mode.as_str(),
                layout.bm32_mode == MhcBm32Mode::Nvfp4,
            );
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_prefill_fused(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        pending: Option<&PendingInject>,
        stage: bool,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let (h, hc, rank) = (self.hidden_size, self.hc_count, self.rank);
        ensure!(
            (h, hc, rank) == (2560, 4, 320),
            "{SELECTOR}: requires H2560/S4/R320, got H{h}/S{hc}/R{rank}"
        );
        ensure!(num_tokens > 0, "{SELECTOR}: empty prefill");
        let inject = self
            .inject
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR} requires an injecting hyperconnection"))?;
        let r = h * hc;
        let n = rank + hc;
        let level = level();
        ensure!(
            num_tokens * rank * 2 <= buffers.sizes().gate_logits
                && num_tokens * h * 2 <= buffers.sizes().norm_output
                && (level == 2 || num_tokens * r * 2 <= buffers.sizes().qkv_output),
            "{SELECTOR}: arena too small for {num_tokens} rows"
        );
        let rows = u32::try_from(num_tokens)?;
        let mixed = buffers.norm_output();
        // ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED: an armed up-mix also reduces
        // the absmax its FP8 tensor-scale consumer would otherwise read for.
        use crate::layers::qwen4_fp8_act_amax as act_amax;
        let amax_armed = act_amax::take_armed();
        act_amax::clear();
        let mxfp8_armed = super::prefill_upmix_roof::take_mxfp8_armed();
        super::prefill_upmix_roof::clear_prequant_mxfp8();

        let strip_cap = strip_rows_cap()?;
        let mhc_sel = super::prefill_inject::selection()?;
        let mut cluster_fused = false;
        if cluster_fuse_eligible(
            cluster_fuse_enabled()?,
            level,
            strip_cap,
            num_tokens,
            mhc_sel,
        ) && let Ok(pk) = prep_kernel(gpu)
        {
            let amax_slot = if amax_armed {
                Some(act_amax::slot(gpu)?)
            } else {
                None
            };
            let s = scratch(gpu, num_tokens, h, hc, rank, false, true)?;
            let launch = super::prefill_inject_gemm::ClusterFuseLaunch {
                hyper,
                residual,
                folded: s.folded,
                up: s.up,
                mixed,
                amax_slot,
                num_tokens,
                eps,
            };
            if let Some(plan) = self.plan_cluster_fuse(pending, &launch, gpu)? {
                self.launch_prep_weights(gpu, pk, inject, s.folded, Some(s.up), stream)?;
                self.launch_cluster_fuse(plan, pending, &launch, gpu, stream)?;
                super::prefill_upmix_roof::discard_dead_l2(
                    [
                        (s.folded, hc * n * h * 2),
                        (s.up, hc * h * rank * 2),
                        (DevicePtr::NULL, 0),
                        (DevicePtr::NULL, 0),
                    ],
                    gpu,
                    stream,
                )?;
                if amax_slot.is_some() {
                    act_amax::note_ready(act_amax::Ready {
                        act: mixed.0,
                        rows: num_tokens,
                        cols: h,
                        stream,
                        producer: act_amax::Producer::HcUpmix,
                    });
                }
                cluster_fused = true;
            }
        }

        if !cluster_fused {
            let k = kernels(gpu)?;
            let mut s = scratch(gpu, num_tokens, h, hc, rank, false, false)?;

            let fused_prep = fused_prep_selected();
            // L2 strip-mining for wide level-2 prefill (opt-in via
            // ATLAS_QWEN4_PREFILL_MHC_STRIP_ROWS=<N>; default 0 on GB10).
            let can_strip = level == 2
                && strip_cap > 0
                && num_tokens > strip_cap
                && strip_pending_safe(pending, hyper, residual, mixed, num_tokens, h)
                && self.inject_gemm_admitted(
                    pending,
                    hyper,
                    residual,
                    [s.folded, s.rstd, s.down],
                    num_tokens,
                )?;
            let strip_w4_k = if can_strip
                && fused_prep
                && mhc_sel.gatepre
                && strip_w4_folded_admit(
                    &StripW4Admit {
                        hyper,
                        residual,
                        down: s.down,
                        rstd: s.rstd,
                        down_packed: self.down.weight,
                        down_scales: self.down.weight_scale,
                        scale2_down: self.down.weight_scale_2,
                        inject_packed: inject.weight,
                        inject_scales: inject.weight_scale,
                        scale2_inject: inject.weight_scale_2,
                        norm_w: self.norm.weight,
                        num_tokens,
                        h,
                        hc,
                        rank,
                    },
                    pending,
                    mhc_sel.postpre,
                ) {
                strip_w4_kernels(gpu)
            } else {
                None
            };

            let (scale2_down, scale2_inject) = if fused_prep {
                ensure!(
                    !self.down.weight.is_null()
                        && !self.down.weight_scale.is_null()
                        && self.down.weight_scale_2.is_finite()
                        && !inject.weight.is_null()
                        && !inject.weight_scale.is_null()
                        && inject.weight_scale_2.is_finite()
                        && (level != 2
                            || (!self.up.weight.is_null()
                                && !self.up.weight_scale.is_null()
                                && self.up.weight_scale_2.is_finite())),
                    "{FUSED_PREP_SELECTOR}: invalid NVFP4 mHC weights"
                );
                if strip_w4_k.is_none() {
                    let pk = prep_kernel(gpu)?;
                    self.launch_prep_weights(gpu, pk, inject, s.folded, None, stream)?;
                    static LOGGED_PREP: std::sync::Once = std::sync::Once::new();
                    LOGGED_PREP.call_once(|| {
                        tracing::info!(
                            "HC_PREFILL_FUSED_PREP_ENGAGED selector={FUSED_PREP_SELECTOR} level={level} bit_identical_to_unfused"
                        );
                    });
                }
                (self.down.weight_scale_2, inject.weight_scale_2)
            } else {
                // Folded down|inject weights, one packed [n, H] slab per stream.
                let scale2_down =
                    qwen4_fast_proj::dequant_into(gpu, &self.down, s.weights, rank, r, stream)?;
                let scale2_inject = qwen4_fast_proj::dequant_into(
                    gpu,
                    inject,
                    s.weights.offset(rank * r * 2),
                    hc,
                    r,
                    stream,
                )?;
                KernelLaunch::new(gpu, k.fold)
                    .grid([div_ceil((hc * n * h) as u32, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(s.weights)
                    .arg_ptr(s.weights.offset(rank * r * 2))
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(s.folded)
                    .arg_u32(rank as u32)
                    .arg_u32(h as u32)
                    .arg_u32(hc as u32)
                    .launch(stream)?;
                (scale2_down, scale2_inject)
            };

            let mut bm64_done = false;
            let bm64_upmix_on = bm64_upmix_enabled()?;
            let bm64_combine_on = bm64_combine_enabled()?;
            let bm32_mode = mhc_bm32_mode()?;
            let active_sms = atlas_core::device::active_sm_count();
            if !can_strip
                && bm64_upmix_eligible_for_sm(
                    bm64_upmix_on,
                    level,
                    strip_cap,
                    num_tokens,
                    mhc_sel,
                    active_sms,
                )
            {
                let amax_slot = if amax_armed {
                    Some(act_amax::slot(gpu)?)
                } else {
                    None
                };
                let bm64_mxfp8 = match mxfp8_armed {
                    Some(req)
                        if req.rows == num_tokens
                            && req.stream == stream
                            && super::prefill_upmix_roof::mxfp8_fused_enabled()? =>
                    {
                        Some(req)
                    }
                    _ => None,
                };
                let admit_args = Bm64UpmixAdmit {
                    hyper,
                    residual,
                    folded: s.folded,
                    mixed,
                    norm_w: self.norm.weight,
                    up_packed: self.up.weight,
                    up_scales: self.up.weight_scale,
                    scale2_down,
                    scale2_inject,
                    alpha_up: self.up.weight_scale_2,
                    amax_slot,
                    mxfp8_req: bm64_mxfp8,
                    num_tokens,
                    h,
                    hc,
                    rank,
                };
                if let Some((core_ptr, admitted_mxfp8)) = bm64_upmix_admit(&admit_args, pending)
                    && let Some((bk, layout)) = (match bm32_mode {
                        MhcBm32Mode::Off => None,
                        MhcBm32Mode::Mxfp8 => {
                            bm32_mxfp8_upmix_kernels(gpu).map(|k| (k, BM32_MXFP8_LAYOUT))
                        }
                        MhcBm32Mode::Nvfp4 => {
                            if gpu.compute_capability().is_some_and(|(maj, _)| maj >= 12)
                                && admit_args.up_packed.0.is_multiple_of(16)
                                && admit_args.up_scales.0.is_multiple_of(4)
                            {
                                bm32_nvfp4_upmix_kernels(gpu).map(|k| (k, BM32_NVFP4_LAYOUT))
                            } else {
                                None
                            }
                        }
                    })
                    .or_else(|| bm64_upmix_kernels(gpu).map(|k| (k, BM64_W4_LAYOUT)))
                {
                    if let Some(slot) = amax_slot {
                        gpu.memset_async(slot, 0, 4, stream)?;
                    }
                    let amax_idx = usize::from(amax_slot.is_some());
                    let handle = match admitted_mxfp8 {
                        Some(_) => bk.mxfp8_gate[amax_idx],
                        None => bk.plain[amax_idx],
                    };
                    let kl = KernelLaunch::new(gpu, handle)
                        .grid([CLUSTER_STREAMS, div_ceil(rows, layout.tile_rows), 1])
                        .block([layout.threads, 1, 1])
                        .shared_mem(layout.smem_bytes)
                        .arg_ptr(hyper)
                        .arg_ptr(core_ptr)
                        .arg_ptr(residual)
                        .arg_ptr(s.folded)
                        .arg_ptr(self.up.weight)
                        .arg_ptr(self.up.weight_scale)
                        .arg_ptr(self.norm.weight)
                        .arg_ptr(mixed)
                        .arg_u32(rows)
                        .arg_u32(rank as u32)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .arg_f32(scale2_down)
                        .arg_f32(scale2_inject)
                        .arg_f32(self.up.weight_scale_2)
                        .arg_f32(eps);
                    let kl = match amax_slot {
                        Some(slot) => kl.arg_ptr(slot),
                        None => kl,
                    };
                    match admitted_mxfp8 {
                        Some(req) => kl
                            .arg_ptr(req.xq)
                            .arg_ptr(req.xs)
                            .arg_ptr(req.shared_gate)
                            .arg_ptr(req.ss_out)
                            .launch(stream)?,
                        None => kl.launch(stream)?,
                    }
                    super::prefill_upmix_roof::discard_dead_l2(
                        [
                            (s.folded, hc * n * h * 2),
                            (DevicePtr::NULL, 0),
                            (DevicePtr::NULL, 0),
                            (DevicePtr::NULL, 0),
                        ],
                        gpu,
                        stream,
                    )?;
                    if let Some(req) = admitted_mxfp8 {
                        super::prefill_upmix_roof::note_prequant_mxfp8(PreQuantizedMxfp8 {
                            mixed: mixed.0,
                            xq: req.xq.0,
                            xs: req.xs.0,
                            shared_gate: req.shared_gate.0,
                            ss_out: req.ss_out.0,
                            rows: num_tokens,
                            stream,
                        });
                    }
                    if amax_slot.is_some() {
                        act_amax::note_ready(act_amax::Ready {
                            act: mixed.0,
                            rows: num_tokens,
                            cols: h,
                            stream,
                            producer: act_amax::Producer::HcUpmix,
                        });
                    }
                    static LOGGED_BM64: std::sync::Once = std::sync::Once::new();
                    LOGGED_BM64.call_once(|| {
                        tracing::info!(
                            "HC_PREFILL_MHC_BM64_UPMIX_ENGAGED selector={} \
                             rows={num_tokens} amax={} mxfp8_gate={} gate_scalars={} bm32_mode={} nvfp4_2cta={}",
                            layout.selector,
                            amax_slot.is_some(),
                            admitted_mxfp8.is_some(),
                            admitted_mxfp8.is_some_and(|req| !req.ss_out.is_null()),
                            layout.bm32_mode.as_str(),
                            layout.bm32_mode == MhcBm32Mode::Nvfp4,
                        );
                    });
                    bm64_done = true;
                } else {
                    let combine_gates = Bm64CombineGates {
                        upmix_enabled: bm64_upmix_on,
                        combine_enabled: bm64_combine_on,
                        level,
                        strip_cap,
                        sel: mhc_sel,
                        oracle_or_dump: bm64_combine_oracle_or_dump(),
                        sm_count: active_sms,
                    };
                    if let Some(c) = bm64_combine_admit(&combine_gates, &admit_args, pending)
                        && let Some((ck, layout)) = (match bm32_mode {
                            MhcBm32Mode::Off => None,
                            MhcBm32Mode::Mxfp8 => {
                                bm32_mxfp8_combine_kernels(gpu).map(|k| (k, BM32_MXFP8_LAYOUT))
                            }
                            MhcBm32Mode::Nvfp4 => {
                                if gpu.compute_capability().is_some_and(|(maj, _)| maj >= 12)
                                    && admit_args.up_packed.0.is_multiple_of(16)
                                    && admit_args.up_scales.0.is_multiple_of(4)
                                {
                                    bm32_nvfp4_combine_kernels(gpu).map(|k| (k, BM32_NVFP4_LAYOUT))
                                } else {
                                    None
                                }
                            }
                        })
                        .or_else(|| bm64_combine_kernels(gpu).map(|k| (k, BM64_COMBINE_LAYOUT)))
                    {
                        self.launch_bm64_combine(
                            gpu,
                            ck,
                            layout,
                            &c,
                            &Bm64CombineLaunch {
                                hyper,
                                residual,
                                folded: s.folded,
                                mixed,
                                amax_slot,
                                scale2_down,
                                scale2_inject,
                                num_tokens,
                                eps,
                            },
                            stream,
                        )?;
                        bm64_done = true;
                    }
                }
            }

            if !bm64_done && can_strip {
                let upmix = mhc_sel.upmix;
                let amax_slot = if upmix && amax_armed {
                    Some(act_amax::slot(gpu)?)
                } else {
                    None
                };
                let output_cache = if num_tokens > 1024 {
                    OutputCachePolicy::StreamEvictFirst
                } else {
                    OutputCachePolicy::Default
                };
                let mut base_upmix = UpmixLaunch {
                    act: s.down,
                    up: s.up,
                    up_packed: self.up.weight,
                    up_scales: self.up.weight_scale,
                    alpha: self.up.weight_scale_2,
                    hyper,
                    rstd: s.rstd,
                    norm_w: self.norm.weight,
                    mixed,
                    amax_slot,
                    output_cache,
                    mxfp8_gate: mxfp8_armed,
                    prefer_w4: true,
                    suppress_mxfp8_note: true,
                };
                let w4 = super::prefill_upmix_roof::w4_admitted(&base_upmix, upmix)?;
                if !w4 && s.up_bytes < hc * h * rank * 2 {
                    s = scratch(gpu, num_tokens, h, hc, rank, false, true)?;
                    base_upmix.up = s.up;
                }
                let strip_mxfp8 = match mxfp8_armed {
                    Some(req)
                        if strip_mxfp8_safe(
                            req,
                            pending,
                            &StripMxfp8CheckBufs {
                                hyper,
                                residual,
                                down: s.down,
                                rstd: s.rstd,
                                folded: s.folded,
                                num_tokens,
                                h,
                                hc,
                                rank,
                            },
                        ) && super::prefill_upmix_roof::can_fuse_mxfp8_gate_strip(
                            &base_upmix,
                            upmix,
                            h,
                            num_tokens,
                            gpu,
                            stream,
                        )? =>
                    {
                        Some(req)
                    }
                    _ => None,
                };
                let alpha_up = if w4 {
                    if let Some(slot) = amax_slot {
                        gpu.memset_async(slot, 0, 4, stream)?;
                    }
                    self.up.weight_scale_2
                } else if fused_prep {
                    // Keep `s.up` disjoint from `s.folded` so strips 1..N still
                    // read `s.folded` while `s.up` is materialized once up front
                    // (and zeroes `amax_slot` once before strip 0).
                    let up_groups = (r * (rank / 16)) as u32;
                    let puk = prep_up_kernel(gpu)?;
                    KernelLaunch::new(gpu, puk)
                        .grid([div_ceil(up_groups, 256), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(self.up.weight)
                        .arg_ptr(self.up.weight_scale)
                        .arg_ptr(s.up)
                        .arg_u32(rank as u32)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .arg_ptr(amax_slot.unwrap_or(DevicePtr::NULL))
                        .launch(stream)?;
                    self.up.weight_scale_2
                } else {
                    if let Some(slot) = amax_slot {
                        gpu.memset_async(slot, 0, 4, stream)?;
                    }
                    qwen4_fast_proj::dequant_into(gpu, &self.up, s.up, r, rank, stream)?
                };

                let mut row_start = 0usize;
                while row_start < num_tokens {
                    let strip_rows = (num_tokens - row_start).min(strip_cap);
                    let hyper_strip = hyper.offset(row_start * r * 2);
                    let residual_strip = residual.offset(row_start * r * 2);
                    let mixed_strip = mixed.offset(row_start * h * 2);
                    let strip_pending =
                        pending.map(|p| slice_pending(p, row_start, strip_rows, h, r));
                    // Reuse `s.rstd` and `s.down` at offset 0 across strips so
                    // every strip shares the same 328 KiB L2 footprint.
                    if let Some(wk) = strip_w4_k {
                        self.launch_strip_w4_phase1(
                            gpu,
                            wk,
                            inject,
                            strip_pending.as_ref(),
                            &StripW4Launch {
                                hyper: hyper_strip,
                                residual: residual_strip,
                                rstd: s.rstd,
                                down: s.down,
                                scale2_down,
                                scale2_inject,
                                strip_rows,
                                eps,
                            },
                            stream,
                        )?;
                    } else {
                        let fused_strip = self.inject_gemm_prefill_strip(
                            strip_pending.as_ref(),
                            hyper_strip,
                            residual_strip,
                            [s.folded, s.rstd, s.down],
                            [scale2_down, scale2_inject],
                            strip_rows,
                            num_tokens,
                            gpu,
                            eps,
                            stream,
                        )?;
                        ensure!(
                            fused_strip,
                            "{STRIP_ROWS_SELECTOR}: inject_gemm refused admitted strip \
                             row_start={row_start} strip_rows={strip_rows}"
                        );
                    }
                    let upmix_launch = UpmixLaunch {
                        act: s.down,
                        up: s.up,
                        up_packed: self.up.weight,
                        up_scales: self.up.weight_scale,
                        alpha: alpha_up,
                        hyper: hyper_strip,
                        rstd: s.rstd,
                        norm_w: self.norm.weight,
                        mixed: mixed_strip,
                        amax_slot,
                        output_cache,
                        mxfp8_gate: strip_mxfp8
                            .map(|req| slice_mxfp8_request(req, row_start, strip_rows, h)),
                        prefer_w4: true,
                        suppress_mxfp8_note: true,
                    };
                    launch_level2_upmix(
                        &upmix_launch,
                        upmix,
                        h,
                        strip_rows,
                        k.up_mix,
                        gpu,
                        stream,
                    )?;
                    row_start += strip_rows;
                }
                if let Some(req) = strip_mxfp8 {
                    super::prefill_upmix_roof::note_prequant_mxfp8(PreQuantizedMxfp8 {
                        mixed: mixed.0,
                        xq: req.xq.0,
                        xs: req.xs.0,
                        shared_gate: req.shared_gate.0,
                        ss_out: req.ss_out.0,
                        rows: num_tokens,
                        stream,
                    });
                }
                let used_strip = strip_cap.min(num_tokens);
                let dead_folded = if strip_w4_k.is_some() {
                    (DevicePtr::NULL, 0)
                } else {
                    (s.folded, hc * n * h * 2)
                };
                let dead_up = if w4 {
                    (DevicePtr::NULL, 0)
                } else {
                    (s.up, hc * h * rank * 2)
                };
                super::prefill_upmix_roof::discard_dead_l2(
                    [
                        (s.down, used_strip * rank * 2),
                        (s.rstd, used_strip * hc * 4),
                        dead_folded,
                        dead_up,
                    ],
                    gpu,
                    stream,
                )?;
                if amax_slot.is_some() {
                    act_amax::note_ready(act_amax::Ready {
                        act: mixed.0,
                        rows: num_tokens,
                        cols: h,
                        stream,
                        producer: act_amax::Producer::HcUpmix,
                    });
                }
                static LOGGED_STRIP: std::sync::Once = std::sync::Once::new();
                LOGGED_STRIP.call_once(|| {
                    tracing::info!(
                        "HC_PREFILL_MHC_STRIP_ENGAGED selector={STRIP_ROWS_SELECTOR} \
                         strip_rows={strip_cap} rows={num_tokens} w4_folded={} w4_up={w4} \
                         mxfp8_gate={} output_cache={output_cache:?}",
                        strip_w4_k.is_some(),
                        strip_mxfp8.is_some(),
                    );
                });
            } else if !bm64_done {
                let mut down = s.down;
                // ATLAS_QWEN4_PREFILL_MHC_FUSED=injgemm: the pending injection, rstd,
                // the four projections and their combine in one kernel (byte-identical).
                let fused = self.inject_gemm_prefill(
                    pending,
                    hyper,
                    residual,
                    [s.folded, s.rstd, down],
                    [scale2_down, scale2_inject],
                    num_tokens,
                    gpu,
                    eps,
                    stream,
                )?;
                if !fused {
                    if s.partial_bytes < hc * num_tokens * n * 4 {
                        s = scratch(gpu, num_tokens, h, hc, rank, true, false)?;
                        down = s.down;
                    }
                    // A pending injection (ATLAS_QWEN4_PREFILL_MHC_FUSED) is applied
                    // by the pass that computes rstd (qwen4_hc_inject_rstd).
                    if let Some(p) = pending {
                        self.inject_rstd_prefill(p, s.rstd, gpu, eps, stream)?;
                    } else {
                        KernelLaunch::new(gpu, k.rstd)
                            .grid([rows, 1, 1])
                            .block([256, 1, 1])
                            .arg_ptr(hyper)
                            .arg_ptr(s.rstd)
                            .arg_u32(h as u32)
                            .arg_f32(eps)
                            .launch(stream)?;
                    }
                    let ref_rows = u32::try_from(reference_rows(num_tokens))?;
                    for stream_id in 0..hc {
                        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                            hyper.offset(stream_id * h * 2).0,
                            r as u32,
                            s.folded.offset(stream_id * n * h * 2).0,
                            s.partial.offset(stream_id * num_tokens * n * 4).0,
                            n as u32,
                            rows,
                            n as u32,
                            h as u32,
                            GemmDtype::Bf16,
                            GemmDtype::F32,
                            false,
                            ref_rows,
                            stream,
                        )?;
                    }
                    KernelLaunch::new(gpu, k.combine)
                        .grid([rows, 1, 1])
                        .block([128, 1, 1])
                        .arg_ptr(s.partial)
                        .arg_ptr(s.rstd)
                        .arg_ptr(down)
                        .arg_ptr(residual)
                        .arg_u32(rows)
                        .arg_u32(rank as u32)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .arg_f32(scale2_down)
                        .arg_f32(scale2_inject)
                        .launch(stream)?;
                }

                if level == 2 {
                    // ATLAS_QWEN4_PREFILL_MHC_FUSED=upmix: the same arithmetic on the
                    // qwen4_hc_upmix tiling (byte-identical, faster).
                    let upmix = mhc_sel.upmix;
                    let amax_slot = if upmix && amax_armed {
                        Some(act_amax::slot(gpu)?)
                    } else {
                        None
                    };
                    let mut upmix_launch = UpmixLaunch {
                        act: down,
                        up: s.up,
                        up_packed: self.up.weight,
                        up_scales: self.up.weight_scale,
                        alpha: self.up.weight_scale_2,
                        hyper,
                        rstd: s.rstd,
                        norm_w: self.norm.weight,
                        mixed,
                        amax_slot,
                        output_cache: OutputCachePolicy::Default,
                        mxfp8_gate: mxfp8_armed,
                        prefer_w4: false,
                        suppress_mxfp8_note: false,
                    };
                    let w4 = super::prefill_upmix_roof::w4_admitted(&upmix_launch, upmix)?;
                    let alpha_up = if w4 {
                        if let Some(slot) = amax_slot {
                            gpu.memset_async(slot, 0, 4, stream)?;
                        }
                        self.up.weight_scale_2
                    } else if fused_prep {
                        upmix_launch.up = s.folded;
                        let up_groups = (r * (rank / 16)) as u32;
                        let puk = prep_up_kernel(gpu)?;
                        KernelLaunch::new(gpu, puk)
                            .grid([div_ceil(up_groups, 256), 1, 1])
                            .block([256, 1, 1])
                            .arg_ptr(self.up.weight)
                            .arg_ptr(self.up.weight_scale)
                            .arg_ptr(s.folded)
                            .arg_u32(rank as u32)
                            .arg_u32(h as u32)
                            .arg_u32(hc as u32)
                            .arg_ptr(amax_slot.unwrap_or(DevicePtr::NULL))
                            .launch(stream)?;
                        self.up.weight_scale_2
                    } else {
                        if let Some(slot) = amax_slot {
                            gpu.memset_async(slot, 0, 4, stream)?;
                        }
                        qwen4_fast_proj::dequant_into(gpu, &self.up, s.up, r, rank, stream)?
                    };
                    upmix_launch.alpha = alpha_up;
                    launch_level2_upmix(
                        &upmix_launch,
                        upmix,
                        h,
                        num_tokens,
                        k.up_mix,
                        gpu,
                        stream,
                    )?;
                    let dead_up = if !w4 && !fused_prep {
                        (s.up, hc * h * rank * 2)
                    } else {
                        (DevicePtr::NULL, 0)
                    };
                    super::prefill_upmix_roof::discard_dead_l2(
                        [
                            (down, num_tokens * rank * 2),
                            (s.rstd, num_tokens * hc * 4),
                            (s.folded, hc * n * h * 2),
                            dead_up,
                        ],
                        gpu,
                        stream,
                    )?;
                    if amax_slot.is_some() {
                        act_amax::note_ready(act_amax::Ready {
                            act: mixed.0,
                            rows: num_tokens,
                            cols: h,
                            stream,
                            producer: act_amax::Producer::HcUpmix,
                        });
                    }
                } else {
                    let mix_logits = buffers.qkv_output();
                    qwen4_fast_proj::dequant_gemm(
                        gpu, down, &self.up, mix_logits, num_tokens, r, rank, stream,
                    )?;
                    KernelLaunch::new(gpu, k.mix_rstd)
                        .grid([rows, div_ceil(h as u32, 256), 1])
                        .block([256, 1, 1])
                        .arg_ptr(hyper)
                        .arg_ptr(s.rstd)
                        .arg_ptr(self.norm.weight)
                        .arg_ptr(mix_logits)
                        .arg_ptr(mixed)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                }
            }
        }
        // Exact-layout compatibility, as in the fast path; skipped for
        // packed-only consumers (ATLAS_QWEN4_PREFILL_MHC_FUSED=nostage).
        if stage {
            KernelLaunch::new(gpu, self.stage_mixed_k)
                .grid([div_ceil((num_tokens * h) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(mixed)
                .arg_ptr(residual)
                .arg_u32(0)
                .arg_u32(rows)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!("HC_PREFILL_FUSED_ENGAGED selector={SELECTOR} level={level} rows={num_tokens} non_bit_exact");
        });
        Ok(mixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pending(mixed: DevicePtr, rows: usize) -> PendingInject {
        PendingInject {
            hyper: DevicePtr(0x1_0000_0000),
            core: mixed,
            residual: DevicePtr(0x2_0000_0000),
            rows,
            combine: Some(MoeCombine {
                slot_scales: DevicePtr(0x3_0000_0000),
                slot_corrections: DevicePtr::NULL,
                expert_out: DevicePtr(0x4_0000_0000),
                token_to_perm: DevicePtr(0x5_0000_0000),
                topk_weights: DevicePtr(0x6_0000_0000),
                shared_out: DevicePtr(0x7_0000_0000),
                gate_input: mixed,
                shared_gate: DevicePtr(0x8_0000_0000),
                gate_scalars: DevicePtr(0x9_0000_0000),
                topk: 10,
            }),
        }
    }

    #[test]
    fn parse_strip_rows_defaults_and_validates() {
        assert_eq!(parse_strip_rows(None), Ok(DEFAULT_STRIP_ROWS));
        assert_eq!(parse_strip_rows(Some("")), Ok(DEFAULT_STRIP_ROWS));
        assert_eq!(parse_strip_rows(Some("0")), Ok(0));
        assert_eq!(parse_strip_rows(Some("off")), Ok(0));
        assert_eq!(parse_strip_rows(Some(" OFF ")), Ok(0));
        assert_eq!(parse_strip_rows(Some("512")), Ok(512));
        assert_eq!(parse_strip_rows(Some("768")), Ok(768));
        for bad in ["-1", "on", "yes", "512r"] {
            assert!(parse_strip_rows(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn slice_pending_shifts_per_token_buffers_only() {
        let mixed = DevicePtr(0xa_0000_0000);
        let p = sample_pending(mixed, 1024);
        let s = slice_pending(&p, 512, 256, 2560, 10_240);
        assert_eq!(s.rows, 256);
        assert_eq!(s.hyper, p.hyper.offset(512 * 10_240 * 2));
        assert_eq!(s.residual, p.residual.offset(512 * 10_240 * 2));
        assert_eq!(s.core, p.core.offset(512 * 2560 * 2));
        let orig = p.combine.expect("sample has combine");
        let sliced = s.combine.expect("sliced preserves combine");
        assert_eq!(sliced.expert_out, orig.expert_out);
        assert_eq!(sliced.slot_scales, orig.slot_scales);
        assert_eq!(sliced.slot_corrections, orig.slot_corrections);
        assert_eq!(sliced.shared_gate, orig.shared_gate);
        assert_eq!(
            sliced.token_to_perm,
            orig.token_to_perm.offset(512 * 10 * 4)
        );
        assert_eq!(sliced.topk_weights, orig.topk_weights.offset(512 * 10 * 4));
        assert_eq!(sliced.shared_out, orig.shared_out.offset(512 * 2560 * 2));
        assert_eq!(sliced.gate_input, orig.gate_input.offset(512 * 2560 * 2));
        assert_eq!(sliced.gate_scalars, orig.gate_scalars.offset(512 * 4));
        assert_eq!(sliced.topk, orig.topk);
    }

    #[test]
    fn strip_pending_safe_checks_aliasing_and_corrections() {
        let mixed = DevicePtr(0xa_0000_0000);
        let p = sample_pending(mixed, 1024);
        assert!(strip_pending_safe(
            None, p.hyper, p.residual, mixed, 1024, 2560
        ));
        assert!(strip_pending_safe(
            Some(&p),
            p.hyper,
            p.residual,
            mixed,
            1024,
            2560
        ));

        // Misaligned core overlap must refuse strip-mining.
        let mut bad_core = p;
        bad_core.combine = None;
        bad_core.core = mixed.offset(16);
        assert!(!strip_pending_safe(
            Some(&bad_core),
            p.hyper,
            p.residual,
            mixed,
            1024,
            2560,
        ));
        bad_core.core = mixed;
        assert!(strip_pending_safe(
            Some(&bad_core),
            p.hyper,
            p.residual,
            mixed,
            1024,
            2560,
        ));

        // Overlapping global expert_out must refuse strip-mining.
        let mut bad_expert = p;
        bad_expert.combine.as_mut().expect("combine").expert_out = mixed;
        assert!(!strip_pending_safe(
            Some(&bad_expert),
            p.hyper,
            p.residual,
            mixed,
            1024,
            2560,
        ));

        // Non-null slot_corrections must refuse strip-mining.
        let mut r2 = p;
        r2.combine.as_mut().expect("combine").slot_corrections = DevicePtr(0xb_0000_0000);
        assert!(!strip_pending_safe(
            Some(&r2),
            p.hyper,
            p.residual,
            mixed,
            1024,
            2560
        ));
    }

    #[test]
    fn parse_cluster_fuse_defaults_off_and_validates() {
        assert_eq!(parse_cluster_fuse(None), Ok(false));
        for on in ["1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_cluster_fuse(Some(on)), Ok(true), "{on:?}");
        }
        for off in ["", "0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(parse_cluster_fuse(Some(off)), Ok(false), "{off:?}");
        }
        for bad in ["2", "yes", "no", "cluster"] {
            assert!(parse_cluster_fuse(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn cluster_fuse_eligibility_checks_all_gates() {
        let ok_sel = super::super::prefill_inject::Selection {
            injgemm: true,
            gatepre: true,
            upmix: true,
            ..Default::default()
        };
        assert!(cluster_fuse_eligible(true, 2, 0, 512, ok_sel));
        assert!(cluster_fuse_eligible(true, 2, 0, 16_384, ok_sel));
        assert!(cluster_fuse_eligible(true, 2, 0, 14_885, ok_sel));
        assert!(!cluster_fuse_eligible(false, 2, 0, 16_384, ok_sel));
        assert!(!cluster_fuse_eligible(true, 1, 0, 16_384, ok_sel));
        assert!(!cluster_fuse_eligible(true, 2, 512, 16_384, ok_sel));
        assert!(!cluster_fuse_eligible(true, 2, 0, 511, ok_sel));
        assert!(!cluster_fuse_eligible(
            true,
            2,
            0,
            16_384,
            super::super::prefill_inject::Selection {
                injgemm: false,
                ..ok_sel
            }
        ));
        assert!(!cluster_fuse_eligible(
            true,
            2,
            0,
            16_384,
            super::super::prefill_inject::Selection {
                gatepre: false,
                ..ok_sel
            }
        ));
        assert!(!cluster_fuse_eligible(
            true,
            2,
            0,
            16_384,
            super::super::prefill_inject::Selection {
                upmix: false,
                ..ok_sel
            }
        ));
    }

    #[test]
    fn prep_weights_threads_covers_folded_and_optional_up() {
        assert_eq!(prep_weights_threads(2560, 4, 320, false), 51_840);
        assert_eq!(prep_weights_threads(2560, 4, 320, true), 51_840 + 204_800);
        assert_eq!(
            div_ceil(prep_weights_threads(2560, 4, 320, false), 256),
            203
        );
        assert_eq!(
            div_ceil(prep_weights_threads(2560, 4, 320, true), 256),
            1003
        );
    }

    #[test]
    fn parse_bm64_upmix_defaults_on_and_validates() {
        assert_eq!(parse_bm64_upmix(None), Ok(true));
        for on in ["", "1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_bm64_upmix(Some(on)), Ok(true), "{on:?}");
        }
        for off in ["0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(parse_bm64_upmix(Some(off)), Ok(false), "{off:?}");
        }
        for bad in ["2", "yes", "no", "bm64"] {
            assert!(parse_bm64_upmix(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parse_bm64_combine_defaults_off_and_validates() {
        assert_eq!(parse_bm64_combine(None), Ok(false));
        for off in ["", "0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(parse_bm64_combine(Some(off)), Ok(false), "{off:?}");
        }
        for on in ["1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_bm64_combine(Some(on)), Ok(true), "{on:?}");
        }
        for bad in ["2", "yes", "no", "combine"] {
            assert!(parse_bm64_combine(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parse_bm64_combine_defaults_on_for_high_sm_and_off_for_gb10() {
        for sm in [0u32, 48, 64] {
            assert_eq!(parse_bm64_combine_for_sms(None, sm), Ok(false), "sm={sm}");
            assert_eq!(
                parse_bm64_combine_for_sms(Some(""), sm),
                Ok(false),
                "sm={sm}"
            );
            assert_eq!(
                parse_bm64_combine_for_sms(Some("1"), sm),
                Ok(true),
                "sm={sm}"
            );
        }
        for sm in [65u32, 160, 188] {
            assert_eq!(parse_bm64_combine_for_sms(None, sm), Ok(true), "sm={sm}");
            assert_eq!(
                parse_bm64_combine_for_sms(Some(""), sm),
                Ok(true),
                "sm={sm}"
            );
            assert_eq!(
                parse_bm64_combine_for_sms(Some("0"), sm),
                Ok(false),
                "sm={sm}"
            );
        }
    }

    #[test]
    fn parse_mhc_nvfp4_defaults_off_and_validates() {
        assert_eq!(parse_mhc_bm32_mode(None), Ok(MhcBm32Mode::Off));
        for off in ["", "0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(
                parse_mhc_bm32_mode(Some(off)),
                Ok(MhcBm32Mode::Off),
                "{off:?}"
            );
        }
        for mxfp8 in ["1", "on", "true", " TRUE ", "On", "mxfp8", " MXFP8 "] {
            assert_eq!(
                parse_mhc_bm32_mode(Some(mxfp8)),
                Ok(MhcBm32Mode::Mxfp8),
                "{mxfp8:?}"
            );
        }
        for nvfp4 in ["nvfp4", " NVFP4 "] {
            assert_eq!(
                parse_mhc_bm32_mode(Some(nvfp4)),
                Ok(MhcBm32Mode::Nvfp4),
                "{nvfp4:?}"
            );
        }
        for bad in ["2", "yes", "no", "fp4"] {
            assert!(parse_mhc_bm32_mode(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn bm64_upmix_eligibility_checks_threshold_and_gates() {
        let ok_sel = super::super::prefill_inject::Selection {
            injgemm: true,
            upmix: true,
            ..Default::default()
        };
        // GB10 (<= 64 SMs): threshold stays at 4097 (zero GB10 regression).
        assert!(!bm64_upmix_eligible(true, 2, 0, 4096, ok_sel));
        assert!(bm64_upmix_eligible(true, 2, 0, 4097, ok_sel));
        // High-SM Blackwell (> 64 SMs): threshold lowers to 2048 so 4K prefill accelerates.
        assert!(!bm64_upmix_eligible_for_sm(true, 2, 0, 2047, ok_sel, 188));
        assert!(bm64_upmix_eligible_for_sm(true, 2, 0, 2048, ok_sel, 188));
        assert!(bm64_upmix_eligible_for_sm(true, 2, 0, 4096, ok_sel, 188));
        assert!(bm64_upmix_eligible(true, 2, 0, 8192, ok_sel));
        assert!(bm64_upmix_eligible(true, 2, 0, 14_885, ok_sel));
        assert!(bm64_upmix_eligible(true, 2, 0, 16_384, ok_sel));
        assert!(!bm64_upmix_eligible(false, 2, 0, 16_384, ok_sel));
        assert!(!bm64_upmix_eligible(true, 1, 0, 16_384, ok_sel));
        assert!(!bm64_upmix_eligible(true, 2, 768, 16_384, ok_sel));
        assert!(!bm64_upmix_eligible(
            true,
            2,
            0,
            16_384,
            super::super::prefill_inject::Selection {
                injgemm: false,
                ..ok_sel
            }
        ));
        assert!(!bm64_upmix_eligible(
            true,
            2,
            0,
            16_384,
            super::super::prefill_inject::Selection {
                upmix: false,
                ..ok_sel
            }
        ));
    }

    #[test]
    fn bm64_upmix_admit_checks_pending_alignment_and_mxfp8_safety() {
        let rows = 8192usize;
        let hyper = DevicePtr(0x1_0000_0000);
        let residual = DevicePtr(0x2_0000_0000);
        let folded = DevicePtr(0x3_0000_0000);
        let mixed = DevicePtr(0x4_0000_0000);
        let core = DevicePtr(0x5_0000_0000);
        let req = UpmixMxfp8Request {
            xq: DevicePtr(0x6_0000_0000),
            xs: DevicePtr(0x6_0400_0000),
            shared_gate: DevicePtr(0x7_0000_0000),
            ss_out: DevicePtr(0x8_0000_0000),
            rows,
            stream: 3,
        };
        let p_core = PendingInject {
            hyper,
            core,
            residual,
            rows,
            combine: None,
        };
        let base = Bm64UpmixAdmit {
            hyper,
            residual,
            folded,
            mixed,
            norm_w: DevicePtr(0x9_0000_0000),
            up_packed: DevicePtr(0xa_0000_0000),
            up_scales: DevicePtr(0xb_0000_0000),
            scale2_down: 0.01,
            scale2_inject: 0.02,
            alpha_up: 0.03,
            amax_slot: Some(DevicePtr(0xc_0000_0000)),
            mxfp8_req: Some(req),
            num_tokens: rows,
            h: 2560,
            hc: 4,
            rank: 320,
        };
        assert_eq!(
            bm64_upmix_admit(&base, Some(&p_core)),
            Some((core, Some(req)))
        );
        // Plain (no pending) or MoeCombine pending must refuse BM=64 path.
        assert_eq!(bm64_upmix_admit(&base, None), None);
        let p_moe = sample_pending(mixed, rows);
        assert_eq!(bm64_upmix_admit(&base, Some(&p_moe)), None);
        // core == mixed alias is admitted (Phase 1 reads core before Phase 2 writes mixed).
        let p_alias = PendingInject {
            core: mixed,
            ..p_core
        };
        assert_eq!(
            bm64_upmix_admit(&base, Some(&p_alias)),
            Some((mixed, Some(req)))
        );
        // Misaligned xq refuses rather than dropping W21-F fusion.
        let bad_xq = Bm64UpmixAdmit {
            mxfp8_req: Some(UpmixMxfp8Request {
                xq: req.xq.offset(4),
                ..req
            }),
            ..base
        };
        assert_eq!(bm64_upmix_admit(&bad_xq, Some(&p_core)), None);
        // Overlapping xq and core refuses.
        let overlap_xq = Bm64UpmixAdmit {
            mxfp8_req: Some(UpmixMxfp8Request { xq: core, ..req }),
            ..base
        };
        assert_eq!(bm64_upmix_admit(&overlap_xq, Some(&p_core)), None);
        // Without mxfp8_req, admits plain/amax mode.
        let no_mx = Bm64UpmixAdmit {
            mxfp8_req: None,
            ..base
        };
        assert_eq!(bm64_upmix_admit(&no_mx, Some(&p_core)), Some((core, None)));
    }

    #[test]
    fn bm64_combine_admit_checks_gates_thresholds_and_combine_contract() {
        let make_case = |rows: usize| {
            let mixed = DevicePtr(0x10_0000_0000);
            let p_moe = sample_pending(mixed, rows);
            let gates = Bm64CombineGates {
                upmix_enabled: true,
                combine_enabled: true,
                level: 2,
                strip_cap: 0,
                sel: super::super::prefill_inject::Selection {
                    injgemm: true,
                    upmix: true,
                    gatepre: true,
                    ..Default::default()
                },
                oracle_or_dump: false,
                sm_count: 48,
            };
            let args = Bm64UpmixAdmit {
                hyper: p_moe.hyper,
                residual: p_moe.residual,
                folded: DevicePtr(0x11_0000_0000),
                mixed,
                norm_w: DevicePtr(0x12_0000_0000),
                up_packed: DevicePtr(0x13_0000_0000),
                up_scales: DevicePtr(0x14_0000_0000),
                scale2_down: 0.01,
                scale2_inject: 0.02,
                alpha_up: 0.03,
                amax_slot: Some(DevicePtr(0x15_0000_0000)),
                mxfp8_req: None,
                num_tokens: rows,
                h: 2560,
                hc: 4,
                rank: 320,
            };
            (gates, args, p_moe)
        };

        // Admits Combine MXFP8 + gatepre at rows >= 4097 (with and without amax).
        for rows in [4097usize, 8192, 14_885, 16_384] {
            let (gates, args, p_moe) = make_case(rows);
            let expected = p_moe.combine;
            assert_eq!(
                bm64_combine_admit(&gates, &args, Some(&p_moe)),
                expected,
                "rows={rows} with amax"
            );
            let no_amax = Bm64UpmixAdmit {
                amax_slot: None,
                ..args
            };
            assert_eq!(
                bm64_combine_admit(&gates, &no_amax, Some(&p_moe)),
                expected,
                "rows={rows} without amax"
            );
        }

        // Rejects rows <= 4096 (exact same threshold as BM64 upmix).
        for rows in [0usize, 1, 2048, 4096] {
            let (gates, args, p_moe) = make_case(rows);
            assert_eq!(
                bm64_combine_admit(&gates, &args, Some(&p_moe)),
                None,
                "rows={rows} must reject"
            );
        }

        let (gates, args, p_moe) = make_case(16_384);

        // Respects =0 opt-out for BM64_COMBINE and BM64_UPMIX.
        assert_eq!(
            bm64_combine_admit(
                &Bm64CombineGates {
                    combine_enabled: parse_bm64_combine(Some("0")).expect("valid 0"),
                    ..gates
                },
                &args,
                Some(&p_moe),
            ),
            None
        );
        assert_eq!(
            bm64_combine_admit(
                &Bm64CombineGates {
                    upmix_enabled: parse_bm64_upmix(Some("0")).expect("valid 0"),
                    ..gates
                },
                &args,
                Some(&p_moe),
            ),
            None
        );

        // Rejects R2 slot corrections.
        let mut p_r2 = p_moe;
        p_r2.combine.as_mut().expect("combine").slot_corrections = DevicePtr(0x16_0000_0000);
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_r2)), None);

        // Rejects non-MXFP8 (BF16) slots.
        let mut p_bf16 = p_moe;
        p_bf16.combine.as_mut().expect("combine").slot_scales = DevicePtr::NULL;
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_bf16)), None);

        // Rejects !gatepre (selector off or missing gate_scalars).
        assert_eq!(
            bm64_combine_admit(
                &Bm64CombineGates {
                    sel: super::super::prefill_inject::Selection {
                        gatepre: false,
                        ..gates.sel
                    },
                    ..gates
                },
                &args,
                Some(&p_moe),
            ),
            None
        );
        let mut p_nogate = p_moe;
        p_nogate.combine.as_mut().expect("combine").gate_scalars = DevicePtr::NULL;
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_nogate)), None);

        // Rejects oracle/dump capture, non-2560 out_dim, postsplit, and armed mxfp8_req.
        assert_eq!(
            bm64_combine_admit(
                &Bm64CombineGates {
                    oracle_or_dump: true,
                    ..gates
                },
                &args,
                Some(&p_moe),
            ),
            None
        );
        assert_eq!(
            bm64_combine_admit(&gates, &Bm64UpmixAdmit { h: 2048, ..args }, Some(&p_moe)),
            None
        );
        assert_eq!(
            bm64_combine_admit(
                &Bm64CombineGates {
                    sel: super::super::prefill_inject::Selection {
                        postsplit: true,
                        ..gates.sel
                    },
                    ..gates
                },
                &args,
                Some(&p_moe),
            ),
            None
        );
        assert_eq!(
            bm64_combine_admit(
                &gates,
                &Bm64UpmixAdmit {
                    mxfp8_req: Some(UpmixMxfp8Request {
                        xq: DevicePtr(0x17_0000_0000),
                        xs: DevicePtr(0x17_0800_0000),
                        shared_gate: DevicePtr(0x18_0000_0000),
                        ss_out: DevicePtr(0x19_0000_0000),
                        rows: 16_384,
                        stream: 1,
                    }),
                    ..args
                },
                Some(&p_moe),
            ),
            None
        );

        // Rejects None / Core pending, misaligned slot_scales, and overlapping expert_out.
        assert_eq!(bm64_combine_admit(&gates, &args, None), None);
        let p_core = PendingInject {
            combine: None,
            ..p_moe
        };
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_core)), None);
        let mut p_bad_scales = p_moe;
        let orig_scales = p_bad_scales.combine.expect("combine").slot_scales;
        p_bad_scales.combine.as_mut().expect("combine").slot_scales = orig_scales.offset(2);
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_bad_scales)), None);
        let mut p_overlap_exp = p_moe;
        p_overlap_exp.combine.as_mut().expect("combine").expert_out = args.mixed;
        assert_eq!(
            bm64_combine_admit(&gates, &args, Some(&p_overlap_exp)),
            None
        );

        // Row-aligned shared_out == mixed is admitted; misaligned overlap is rejected.
        let mut p_sh_alias = p_moe;
        p_sh_alias.combine.as_mut().expect("combine").shared_out = args.mixed;
        assert_eq!(
            bm64_combine_admit(&gates, &args, Some(&p_sh_alias)),
            p_sh_alias.combine
        );
        let mut p_sh_bad = p_moe;
        p_sh_bad.combine.as_mut().expect("combine").shared_out = args.mixed.offset(16);
        assert_eq!(bm64_combine_admit(&gates, &args, Some(&p_sh_bad)), None);
    }

    #[test]
    fn slice_mxfp8_request_shifts_row_buffers_only() {
        let req = UpmixMxfp8Request {
            xq: DevicePtr(0x1_0000_0000),
            xs: DevicePtr(0x2_0000_0000),
            shared_gate: DevicePtr(0x3_0000_0000),
            ss_out: DevicePtr(0x4_0000_0000),
            rows: 16_384,
            stream: 9,
        };
        let s = slice_mxfp8_request(req, 768, 768, 2560);
        assert_eq!(s.xq, req.xq.offset(768 * 2560));
        assert_eq!(s.xs, req.xs.offset(768 * 80));
        assert_eq!(s.shared_gate, req.shared_gate);
        assert_eq!(s.ss_out, req.ss_out.offset(768 * 4));
        assert_eq!(s.rows, 768);
        assert_eq!(s.stream, 9);

        let no_gate = slice_mxfp8_request(
            UpmixMxfp8Request {
                ss_out: DevicePtr::NULL,
                ..req
            },
            768,
            512,
            2560,
        );
        assert_eq!(no_gate.ss_out, DevicePtr::NULL);
        assert_eq!(no_gate.rows, 512);
    }

    #[test]
    fn strip_mxfp8_safe_checks_disjointness() {
        let rows = 1536usize;
        let mixed = DevicePtr(0xa_0000_0000);
        let p = sample_pending(mixed, rows);
        let bufs = StripMxfp8CheckBufs {
            hyper: p.hyper,
            residual: p.residual,
            down: DevicePtr(0xb_0000_0000),
            rstd: DevicePtr(0xc_0000_0000),
            folded: DevicePtr(0xd_0000_0000),
            num_tokens: rows,
            h: 2560,
            hc: 4,
            rank: 320,
        };
        let req = UpmixMxfp8Request {
            xq: DevicePtr(0xe_0000_0000),
            xs: DevicePtr(0xe_0100_0000),
            shared_gate: DevicePtr(0xf_0000_0000),
            ss_out: DevicePtr(0xf_0100_0000),
            rows,
            stream: 1,
        };
        assert!(strip_mxfp8_safe(req, Some(&p), &bufs));
        // Overlap with hyper refuses.
        assert!(!strip_mxfp8_safe(
            UpmixMxfp8Request { xq: p.hyper, ..req },
            Some(&p),
            &bufs
        ));
        // Overlap with pending expert_out refuses.
        assert!(!strip_mxfp8_safe(
            UpmixMxfp8Request {
                xq: p.combine.expect("combine").expert_out,
                ..req
            },
            Some(&p),
            &bufs
        ));
    }

    #[test]
    fn strip_w4_folded_admit_checks_alignment_and_postpre() {
        let rows = 1536usize;
        let mixed = DevicePtr(0xa_0000_0000);
        let p = sample_pending(mixed, rows);
        let base = StripW4Admit {
            hyper: p.hyper,
            residual: p.residual,
            down: DevicePtr(0xb_0000_0000),
            rstd: DevicePtr(0xc_0000_0000),
            down_packed: DevicePtr(0xd_0000_0000),
            down_scales: DevicePtr(0xd_0100_0000),
            scale2_down: 0.01,
            inject_packed: DevicePtr(0xd_0200_0000),
            inject_scales: DevicePtr(0xd_0300_0000),
            scale2_inject: 0.02,
            norm_w: DevicePtr(0xd_0400_0000),
            num_tokens: rows,
            h: 2560,
            hc: 4,
            rank: 320,
        };
        assert!(strip_w4_folded_admit(&base, None, false));
        assert!(strip_w4_folded_admit(&base, Some(&p), true));
        assert!(!strip_w4_folded_admit(&base, Some(&p), false));
        // norm_w must be 16-byte aligned for ld_u128_ca_keep.
        assert!(!strip_w4_folded_admit(
            &StripW4Admit {
                norm_w: base.norm_w.offset(4),
                ..base
            },
            None,
            false
        ));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_parity_tests {
    use super::*;
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

    fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    fn f32_bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
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

    fn dequant_mxfp8(q: &[u8], s: &[u8], rows: usize, cols: usize) -> Vec<f32> {
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

    fn load_hc_kernel(gpu: &dyn GpuBackend, name: &str, smem: u32) -> KernelHandle {
        let k = gpu
            .kernel("qwen4_hc_combine_gemm", name)
            .unwrap_or_else(|e| panic!("load {name}: {e:#}"));
        gpu.set_kernel_max_dynamic_shared_memory(k, smem)
            .unwrap_or_else(|e| panic!("set smem {smem} for {name}: {e:#}"));
        k
    }

    #[test]
    fn gpu_mhc_bm32_mxfp8_and_nvfp4_vs_bm64_upmix_parity() {
        let Some(gpu) = try_gb10_flash_next() else {
            return;
        };
        let stream = gpu.default_stream();
        const H: usize = 2560;
        const HC: usize = 4;
        const RANK: usize = 320;
        const WIDTH: usize = H * HC;
        const NTOTAL: usize = RANK + HC;
        const TOPK: usize = STRIP_W4_TOPK as usize;

        let k_bm64_up = load_hc_kernel(
            &gpu,
            "qwen4_hc_inject_upmix_bm64_w4_mxfp8_gate_amax",
            SMEM_BYTES_BM64_W4,
        );
        let k_mxfp8_up = load_hc_kernel(
            &gpu,
            "qwen4_hc_inject_upmix_bm32_mxfp8_2cta_mxfp8_gate_amax",
            SMEM_BYTES_BM32_MXFP8,
        );
        let k_nvfp4_up = load_hc_kernel(
            &gpu,
            "qwen4_hc_inject_upmix_bm32_nvfp4_2cta_mxfp8_gate_amax",
            SMEM_BYTES_BM32_NVFP4,
        );
        let k_bm64_plain =
            load_hc_kernel(&gpu, "qwen4_hc_inject_upmix_bm64_w4", SMEM_BYTES_BM64_W4);
        let k_mxfp8_plain = load_hc_kernel(
            &gpu,
            "qwen4_hc_inject_upmix_bm32_mxfp8_2cta",
            SMEM_BYTES_BM32_MXFP8,
        );
        let k_nvfp4_plain = load_hc_kernel(
            &gpu,
            "qwen4_hc_inject_upmix_bm32_nvfp4_2cta",
            SMEM_BYTES_BM32_NVFP4,
        );
        let k_bm64_comb = load_hc_kernel(
            &gpu,
            "qwen4_hc_combine_inject_upmix_bm64_w4_mx_gatepre_amax",
            SMEM_BYTES_BM64_W4,
        );
        let k_mxfp8_comb = load_hc_kernel(
            &gpu,
            "qwen4_hc_combine_inject_upmix_bm32_mxfp8_2cta_mx_gatepre_amax",
            SMEM_BYTES_BM32_MXFP8,
        );
        let k_nvfp4_comb = load_hc_kernel(
            &gpu,
            "qwen4_hc_combine_inject_upmix_bm32_nvfp4_2cta_mx_gatepre_amax",
            SMEM_BYTES_BM32_NVFP4,
        );

        let mut rng = Rng(0x2026_1125_feed_beef);
        let folded_h: Vec<f32> = (0..HC * NTOTAL * H)
            .map(|_| rng.uniform(-0.35, 0.35))
            .collect();
        let up_packed_h: Vec<u8> = (0..WIDTH * (RANK / 2))
            .map(|_| {
                let lo = (rng.next_u32() & 0x0f) as u8;
                let hi = (rng.next_u32() & 0x0f) as u8;
                lo | (hi << 4)
            })
            .collect();
        let up_scales_h: Vec<u8> = (0..WIDTH * (RANK / 16))
            .map(|_| (0x28 + (rng.next_u32() % 24)) as u8)
            .collect();
        let norm_w_h: Vec<f32> = (0..WIDTH).map(|_| rng.uniform(-0.25, 0.25)).collect();
        let shared_gate_h: Vec<f32> = (0..H).map(|_| rng.uniform(-0.15, 0.15)).collect();

        let folded_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&folded_h));
        let up_packed_dev = DevBuf::upload(&gpu, &up_packed_h);
        let up_scales_dev = DevBuf::upload(&gpu, &up_scales_h);
        let norm_w_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&norm_w_h));
        let shared_gate_dev = DevBuf::upload(&gpu, &f32_to_bf16_bytes(&shared_gate_h));

        let scale2_down = 0.02f32;
        let scale2_inject = 0.02f32;
        let alpha_up = 0.025f32;
        let eps = 1e-6f32;

        for &rows in &[33usize, 4003, 4097] {
            let hyper_init = f32_to_bf16_bytes(
                &(0..rows * WIDTH)
                    .map(|_| rng.uniform(-1.5, 1.5))
                    .collect::<Vec<_>>(),
            );
            let core_init = f32_to_bf16_bytes(
                &(0..rows * H)
                    .map(|_| rng.uniform(-1.0, 1.0))
                    .collect::<Vec<_>>(),
            );
            let res_init = f32_to_bf16_bytes(
                &(0..rows * WIDTH)
                    .map(|_| rng.uniform(-1.0, 1.0))
                    .collect::<Vec<_>>(),
            );
            let core_dev = DevBuf::upload(&gpu, &core_init);

            let run_upmix = |handle: KernelHandle, layout: ClusterFusedLayout| {
                let hyper_d = DevBuf::upload(&gpu, &hyper_init);
                let res_d = DevBuf::upload(&gpu, &res_init);
                let mixed_d = DevBuf::zeroed(&gpu, rows * H * 2);
                let amax_d = DevBuf::zeroed(&gpu, 4);
                let xq_d = DevBuf::zeroed(&gpu, rows * H);
                let xs_d = DevBuf::zeroed(&gpu, rows * (H / 32));
                let ss_d = DevBuf::zeroed(&gpu, rows * 4);
                KernelLaunch::new(&gpu, handle)
                    .grid([CLUSTER_STREAMS, div_ceil(rows as u32, layout.tile_rows), 1])
                    .block([layout.threads, 1, 1])
                    .shared_mem(layout.smem_bytes)
                    .arg_ptr(hyper_d.ptr())
                    .arg_ptr(core_dev.ptr())
                    .arg_ptr(res_d.ptr())
                    .arg_ptr(folded_dev.ptr())
                    .arg_ptr(up_packed_dev.ptr())
                    .arg_ptr(up_scales_dev.ptr())
                    .arg_ptr(norm_w_dev.ptr())
                    .arg_ptr(mixed_d.ptr())
                    .arg_u32(rows as u32)
                    .arg_u32(RANK as u32)
                    .arg_u32(H as u32)
                    .arg_u32(HC as u32)
                    .arg_f32(scale2_down)
                    .arg_f32(scale2_inject)
                    .arg_f32(alpha_up)
                    .arg_f32(eps)
                    .arg_ptr(amax_d.ptr())
                    .arg_ptr(xq_d.ptr())
                    .arg_ptr(xs_d.ptr())
                    .arg_ptr(shared_gate_dev.ptr())
                    .arg_ptr(ss_d.ptr())
                    .launch(stream)
                    .expect("upmix launch");
                let sample = rows.min(256);
                let mixed_head = bf16_bytes_to_f32(&mixed_d.read_bytes(sample * H * 2));
                let mixed_tail = if rows > sample {
                    let mut b = vec![0u8; sample * H * 2];
                    gpu.copy_d2h(mixed_d.ptr().offset((rows - sample) * H * 2), &mut b)
                        .expect("d2h mixed tail");
                    bf16_bytes_to_f32(&b)
                } else {
                    Vec::new()
                };
                let hyper_head = bf16_bytes_to_f32(&hyper_d.read_bytes(sample * WIDTH * 2));
                let res_head = bf16_bytes_to_f32(&res_d.read_bytes(sample * WIDTH * 2));
                let xq_head = xq_d.read_bytes(sample * H);
                let xs_head = xs_d.read_bytes(sample * (H / 32));
                let mx_dq = dequant_mxfp8(&xq_head, &xs_head, sample, H);
                let ss_all = f32_bytes_to_vec(&ss_d.read_bytes(rows * 4));
                let amax_bits =
                    u32::from_le_bytes(amax_d.read_bytes(4).try_into().expect("4-byte amax"));
                let amax = f32::from_bits(amax_bits);
                (
                    mixed_head, mixed_tail, hyper_head, res_head, mx_dq, ss_all, amax,
                )
            };

            let ref_up = run_upmix(k_bm64_up, BM64_W4_LAYOUT);
            for (tag, handle, layout) in [
                ("bm32_mxfp8", k_mxfp8_up, BM32_MXFP8_LAYOUT),
                ("bm32_nvfp4", k_nvfp4_up, BM32_NVFP4_LAYOUT),
            ] {
                let cand = run_upmix(handle, layout);
                let (cos_m, max_m, nan_m) = parity_metrics(&cand.0, &ref_up.0);
                let (cos_h, max_h, nan_h) = parity_metrics(&cand.2, &ref_up.2);
                let (cos_r, _, nan_r) = parity_metrics(&cand.3, &ref_up.3);
                let (cos_xq, _, nan_xq) = parity_metrics(&cand.4, &ref_up.4);
                let (cos_ss, max_ss, nan_ss) = parity_metrics(&cand.5, &ref_up.5);
                let amax_rel = (cand.6 - ref_up.6).abs() / ref_up.6.max(1e-6);
                eprintln!(
                    "mHC upmix {tag} vs bm64_upmix rows={rows}: \
                     mixed cos={cos_m:.6} max={max_m:.3e} | hyper cos={cos_h:.6} max={max_h:.3e} | \
                     res cos={cos_r:.6} | xq cos={cos_xq:.6} | ss cos={cos_ss:.6} max={max_ss:.3e} | \
                     amax={:.4}/{:.4} (rel={amax_rel:.3e})",
                    cand.6, ref_up.6
                );
                assert_eq!(nan_m + nan_h + nan_r + nan_xq + nan_ss, 0);
                assert!(cand.6.is_finite() && cand.6 > 0.0);
                assert!(cos_m >= 0.9998, "{tag} upmix rows={rows} mixed cos={cos_m}");
                assert!(
                    cos_h >= 0.99999,
                    "{tag} upmix rows={rows} hyper cos={cos_h}"
                );
                assert!(cos_r >= 0.99999, "{tag} upmix rows={rows} res cos={cos_r}");
                assert!(cos_xq >= 0.9998, "{tag} upmix rows={rows} xq cos={cos_xq}");
                assert!(cos_ss >= 0.9998, "{tag} upmix rows={rows} ss cos={cos_ss}");
                assert!(
                    amax_rel < 0.02,
                    "{tag} upmix rows={rows} amax_rel={amax_rel}"
                );
                if !ref_up.1.is_empty() {
                    let (cos_mt, _, nan_mt) = parity_metrics(&cand.1, &ref_up.1);
                    assert_eq!(nan_mt, 0);
                    assert!(
                        cos_mt >= 0.9998,
                        "{tag} upmix rows={rows} mixed tail cos={cos_mt}"
                    );
                }
            }

            // Also verify plain (non-mxfp8-gate, non-amax) entry point at rows=33.
            if rows == 33 {
                let run_plain = |handle: KernelHandle, layout: ClusterFusedLayout| {
                    let hyper_d = DevBuf::upload(&gpu, &hyper_init);
                    let res_d = DevBuf::upload(&gpu, &res_init);
                    let mixed_d = DevBuf::zeroed(&gpu, rows * H * 2);
                    KernelLaunch::new(&gpu, handle)
                        .grid([CLUSTER_STREAMS, div_ceil(rows as u32, layout.tile_rows), 1])
                        .block([layout.threads, 1, 1])
                        .shared_mem(layout.smem_bytes)
                        .arg_ptr(hyper_d.ptr())
                        .arg_ptr(core_dev.ptr())
                        .arg_ptr(res_d.ptr())
                        .arg_ptr(folded_dev.ptr())
                        .arg_ptr(up_packed_dev.ptr())
                        .arg_ptr(up_scales_dev.ptr())
                        .arg_ptr(norm_w_dev.ptr())
                        .arg_ptr(mixed_d.ptr())
                        .arg_u32(rows as u32)
                        .arg_u32(RANK as u32)
                        .arg_u32(H as u32)
                        .arg_u32(HC as u32)
                        .arg_f32(scale2_down)
                        .arg_f32(scale2_inject)
                        .arg_f32(alpha_up)
                        .arg_f32(eps)
                        .launch(stream)
                        .expect("plain upmix launch");
                    bf16_bytes_to_f32(&mixed_d.read_bytes(rows * H * 2))
                };
                let ref_p = run_plain(k_bm64_plain, BM64_W4_LAYOUT);
                for (tag, handle, layout) in [
                    ("bm32_mxfp8_plain", k_mxfp8_plain, BM32_MXFP8_LAYOUT),
                    ("bm32_nvfp4_plain", k_nvfp4_plain, BM32_NVFP4_LAYOUT),
                ] {
                    let cand_p = run_plain(handle, layout);
                    let (cos_p, max_p, nan_p) = parity_metrics(&cand_p, &ref_p);
                    eprintln!("mHC {tag} vs bm64_w4 rows={rows}: cos={cos_p:.6} max={max_p:.3e}");
                    assert_eq!(nan_p, 0);
                    assert!(cos_p >= 0.9998, "{tag} rows={rows} cos={cos_p}");
                }
            }

            // Combine path (`_mx_gatepre_amax`): MoE MXFP8 slot combine + inject + upmix!
            let pairs = rows * TOPK;
            let exp_q_h: Vec<u8> = (0..pairs * H)
                .map(|_| {
                    let sign = ((rng.next_u32() & 1) << 7) as u8;
                    let mag = (0x20 + (rng.next_u32() % 40)) as u8;
                    sign | mag
                })
                .collect();
            let slot_s_h: Vec<u8> = (0..pairs * (H / 32))
                .map(|_| (122 + (rng.next_u32() % 5)) as u8)
                .collect();
            let mut t2p_h = Vec::with_capacity(pairs);
            let mut tw_h = Vec::with_capacity(pairs);
            for t in 0..rows {
                for s in 0..TOPK {
                    let perm_tok = (t * 7 + s) % rows;
                    t2p_h.push((perm_tok * TOPK + s) as i32);
                    tw_h.push(0.1f32);
                }
            }
            let shared_out_h = f32_to_bf16_bytes(
                &(0..rows * H)
                    .map(|_| rng.uniform(-0.8, 0.8))
                    .collect::<Vec<_>>(),
            );
            let gate_sc_h: Vec<f32> = (0..rows).map(|_| rng.uniform(0.2, 0.8)).collect();

            let exp_q_dev = DevBuf::upload(&gpu, &exp_q_h);
            let slot_s_dev = DevBuf::upload(&gpu, &slot_s_h);
            let t2p_dev = DevBuf::upload(
                &gpu,
                &t2p_h
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let tw_dev = DevBuf::upload(
                &gpu,
                &tw_h
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let shared_out_dev = DevBuf::upload(&gpu, &shared_out_h);
            let gate_sc_dev = DevBuf::upload(
                &gpu,
                &gate_sc_h
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );

            let run_combine = |handle: KernelHandle, layout: ClusterFusedLayout| {
                let hyper_d = DevBuf::upload(&gpu, &hyper_init);
                let res_d = DevBuf::upload(&gpu, &res_init);
                let moe_rows_d = DevBuf::zeroed(&gpu, rows * H * 2);
                let mixed_d = DevBuf::zeroed(&gpu, rows * H * 2);
                let amax_d = DevBuf::zeroed(&gpu, 4);
                KernelLaunch::new(&gpu, handle)
                    .grid([CLUSTER_STREAMS, div_ceil(rows as u32, layout.tile_rows), 1])
                    .block([layout.threads, 1, 1])
                    .shared_mem(layout.smem_bytes)
                    .arg_ptr(hyper_d.ptr())
                    .arg_ptr(res_d.ptr())
                    .arg_ptr(moe_rows_d.ptr())
                    .arg_ptr(exp_q_dev.ptr())
                    .arg_ptr(slot_s_dev.ptr())
                    .arg_ptr(t2p_dev.ptr())
                    .arg_ptr(tw_dev.ptr())
                    .arg_ptr(shared_out_dev.ptr())
                    .arg_ptr(gate_sc_dev.ptr())
                    .arg_ptr(folded_dev.ptr())
                    .arg_ptr(up_packed_dev.ptr())
                    .arg_ptr(up_scales_dev.ptr())
                    .arg_ptr(norm_w_dev.ptr())
                    .arg_ptr(mixed_d.ptr())
                    .arg_u32(rows as u32)
                    .arg_u32(RANK as u32)
                    .arg_u32(H as u32)
                    .arg_u32(HC as u32)
                    .arg_u32(TOPK as u32)
                    .arg_f32(scale2_down)
                    .arg_f32(scale2_inject)
                    .arg_f32(alpha_up)
                    .arg_f32(eps)
                    .arg_ptr(amax_d.ptr())
                    .launch(stream)
                    .expect("combine launch");
                let sample = rows.min(256);
                let mixed_head = bf16_bytes_to_f32(&mixed_d.read_bytes(sample * H * 2));
                let hyper_head = bf16_bytes_to_f32(&hyper_d.read_bytes(sample * WIDTH * 2));
                let amax_bits =
                    u32::from_le_bytes(amax_d.read_bytes(4).try_into().expect("4-byte amax"));
                (mixed_head, hyper_head, f32::from_bits(amax_bits))
            };

            let ref_c = run_combine(k_bm64_comb, BM64_COMBINE_LAYOUT);
            for (tag, handle, layout) in [
                ("bm32_mxfp8_combine", k_mxfp8_comb, BM32_MXFP8_LAYOUT),
                ("bm32_nvfp4_combine", k_nvfp4_comb, BM32_NVFP4_LAYOUT),
            ] {
                let cand = run_combine(handle, layout);
                let (cos_m, max_m, nan_m) = parity_metrics(&cand.0, &ref_c.0);
                let (cos_h, max_h, nan_h) = parity_metrics(&cand.1, &ref_c.1);
                let amax_rel = (cand.2 - ref_c.2).abs() / ref_c.2.max(1e-6);
                eprintln!(
                    "mHC {tag} vs bm64_combine rows={rows}: \
                     mixed cos={cos_m:.6} max={max_m:.3e} | hyper cos={cos_h:.6} max={max_h:.3e} | \
                     amax={:.4}/{:.4} (rel={amax_rel:.3e})",
                    cand.2, ref_c.2
                );
                assert_eq!(nan_m + nan_h, 0);
                assert!(cand.2.is_finite() && cand.2 > 0.0);
                assert!(cos_m >= 0.9998, "{tag} rows={rows} mixed cos={cos_m}");
                assert!(cos_h >= 0.99999, "{tag} rows={rows} hyper cos={cos_h}");
                assert!(amax_rel < 0.02, "{tag} rows={rows} amax_rel={amax_rel}");
            }
        }
    }
}
