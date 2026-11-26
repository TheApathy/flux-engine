// SPDX-License-Identifier: AGPL-3.0-only

//! V33 Thin-M Split-K NVFP4 MoE verify kernel dispatch (`rows = 4..=256`).
//!
//! Targets high-SM Blackwell GPUs (`active_sm_count() > 64`) across batched verify
//! frames where 4-CTA/SM `v28_m16` lacks Split-K occupancy (`rows <= 32`) and
//! 1-CTA/SM `v31_ws` leaves most consumer warps idle (`rows <= 256`):
//! - 4-warp (`block = 128`), `<= 10 KiB` dynamic SMEM -> 6 resident CTAs/SM
//!   (`1,128` resident CTAs on 188-SM SM120) with dynamic atomic work-stealing.
//! - Reads BF16 activations directly (zero `xq` / `oq` MXFP8 round-trip, never
//!   touches `expert_up_out` so shared-expert overlap remains race-free).
//! - Dynamic Split-K on GATEUP (`K = 2560`, `SK in {1, 2, 4}`) and DOWN
//!   (`K = 640`, `SK in {1, 2}`), reusing dead pre-allocated scratch inside
//!   `expert_down_out` (during GATEUP) and the free tail of `expert_gate_out`
//!   (during DOWN) with strict bounds checks and zero per-step `gpu.alloc()`.
//! - Auto-ON on high-SM Blackwell GPUs (`active_sm_count() > 64`, e.g. SM120) for
//!   `rows = 4..=96` (`THIN_SK_DEFAULT_MAX_ROWS`), handing off `rows >= 97` (`128..=256`)
//!   to `V31_WS` where warp-specialized TMA beats Split-K by +10..15%. GB10 (`<= 64` SMs)
//!   remains 100% untouched under unset/auto/`1`/`all`.
//! - Override via `ATLAS_QWEN4_MOE_THIN_SK=0|off|auto|1|on|all|force` and optional
//!   crossover ceiling `ATLAS_QWEN4_MOE_THIN_SK_MAX_ROWS=<4..=256>`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::moe::{
    ExpertPtrTable, RoutedScaleLayout, qwen4_compact_contract as abi, qwen4_prefill_compact,
};

/// Environment selector for the V33 thin-M Split-K MoE verify kernel.
///
/// - Unset / `""` / `"auto"`: auto-ON for `rows = 4..=max_rows` (default `96`) on
///   `active_sm_count() > 64` (SM120/SM103); OFF on GB10 (`<= 64` SMs).
/// - `"0"` / `"off"` / `"false"`: disabled everywhere.
/// - `"1"` / `"on"` / `"true"`: engages for `rows = 4..=max_rows` (default `96`) on
///   `active_sm_count() > 64`.
/// - `"all"`: engages across the full `rows = 4..=256` range on `active_sm_count() > 64`.
/// - `"force"`: engages across `rows = 4..=256` regardless of SM count (for A/B testing).
pub(crate) const THIN_SK_SELECTOR: &str = "ATLAS_QWEN4_MOE_THIN_SK";

/// Optional crossover row ceiling override for [`ThinSkPolicy::Auto`] / [`ThinSkPolicy::On`]
/// (`4..=256`, default [`THIN_SK_DEFAULT_MAX_ROWS`] = `96`).
pub(crate) const THIN_SK_MAX_ROWS_SELECTOR: &str = "ATLAS_QWEN4_MOE_THIN_SK_MAX_ROWS";

/// Compiled kernel module name auto-discovered by `atlas-kernels/build.rs`.
pub(crate) const THIN_SK_MODULE: &str = "qwen4_moe_thin_sk_v33";

/// Minimum batch rows admitted by `v33_thin_sk` (`C = 1` verify has `rows >= 4`).
pub(crate) const THIN_SK_MIN_ROWS: usize = 4;

/// Default SM120 crossover ceiling (`rows <= 32` keeps `C=1..4` verify on
/// `v33_thin_sk`, while `rows >= 33` hands off to `V31_WS`, winning by
/// +116..206 tok/s across `C=16..32` ragged `p_cum=0.25` verify frames).
pub(crate) const THIN_SK_DEFAULT_MAX_ROWS: usize = 32;

/// Hard maximum batch rows supported by `v33_thin_sk` (`rows = 4..=256`).
pub(crate) const THIN_SK_MAX_ROWS: usize = 256;

/// SM count threshold above which `ThinSkPolicy::On` engages (keeps GB10 `<= 64` SMs unchanged).
pub(crate) const THIN_SK_HIGH_SM_THRESHOLD: u32 = 64;

/// Thread-block width for all 6 `v33_thin_sk` persistent GEMM kernels (4 warps).
pub(crate) const THIN_SK_BLOCK: u32 = 128;

/// Dynamic SMEM footprint for flat-scale `v33_thin_sk` GEMM kernels (`9,824 B <= 12 KiB`).
pub(crate) const THIN_SK_FLAT_SMEM_BYTES: u32 = 9_824;

/// Dynamic SMEM footprint for Atlas-TS4E-128 `v33_thin_sk` GEMM kernels (`9,952 B <= 12 KiB`).
pub(crate) const THIN_SK_TS4E_SMEM_BYTES: u32 = 9_952;

const TOPK: usize = 10;
const HIDDEN: usize = 2560;
const INTER: usize = 640;
const GATEUP_N_TILES: usize = INTER / 64;
const DOWN_N_TILES: usize = HIDDEN / 128;
const RESIDENT_CTAS_PER_SM: usize = 6;
const PARTIALS_ALIGN_BYTES: usize = 256;
const REDUCE_SILU_GRID_Y: u32 = 5;
const REDUCE_SILU_BLOCK: u32 = 32;
const DOWN_REDUCE_GRID_Y: u32 = 20;
const DOWN_REDUCE_BLOCK: u32 = 128;
const SHARED_GATE_BLOCK: u32 = 256;

static THIN_SK_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Dispatch policy parsed from [`THIN_SK_SELECTOR`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThinSkPolicy {
    /// Disabled everywhere (explicit `"0"` / `"off"` / `"false"`).
    Off,
    /// Default when unset or `"auto"`: enabled for `rows = 4..=max_rows` (default `96`)
    /// on high-SM Blackwell GPUs (`sm_count > 64`), disabled on GB10 (`<= 64` SMs).
    Auto,
    /// Explicitly enabled (`"1"` / `"on"` / `"true"`): `rows = 4..=max_rows` (default `96`)
    /// on high-SM Blackwell GPUs (`sm_count > 64`).
    On,
    /// Full-range SM120 override (`"all"`): `rows = 4..=256` on `sm_count > 64`.
    All,
    /// Forced for `rows = 4..=256` on any SM count (explicit A/B override).
    Force,
}

/// Dispatch-time conditions checked before admitting `v33_thin_sk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThinSkEligibility {
    pub rows: usize,
    pub w4a8_is_v28: bool,
    pub has_corrections: bool,
    pub v29_m256_active: bool,
    pub oracle_capture: bool,
    pub kernels_loaded: bool,
}

impl ThinSkPolicy {
    /// Whether `v33_thin_sk` should handle this dispatch on a GPU with `sm_count` SMs
    /// and crossover ceiling `max_rows` (used by [`Self::Auto`] and [`Self::On`]).
    #[must_use]
    pub(crate) const fn admits_for_sm_with_max_rows(
        self,
        e: ThinSkEligibility,
        sm_count: u32,
        max_rows: usize,
    ) -> bool {
        let base_ok = e.rows >= THIN_SK_MIN_ROWS
            && e.rows <= THIN_SK_MAX_ROWS
            && e.w4a8_is_v28
            && !e.has_corrections
            && !e.v29_m256_active
            && !e.oracle_capture
            && e.kernels_loaded;
        if !base_ok {
            return false;
        }
        match self {
            Self::Off => false,
            Self::Auto | Self::On => sm_count > THIN_SK_HIGH_SM_THRESHOLD && e.rows <= max_rows,
            Self::All => sm_count > THIN_SK_HIGH_SM_THRESHOLD,
            Self::Force => true,
        }
    }

    /// Whether `v33_thin_sk` should handle this dispatch on a GPU with `sm_count` SMs
    /// under the default crossover ceiling ([`THIN_SK_DEFAULT_MAX_ROWS`] = `96`).
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn admits_for_sm(self, e: ThinSkEligibility, sm_count: u32) -> bool {
        self.admits_for_sm_with_max_rows(e, sm_count, THIN_SK_DEFAULT_MAX_ROWS)
    }

    /// Whether `v33_thin_sk` should handle this dispatch on the active GPU.
    #[must_use]
    pub(crate) fn admits(self, e: ThinSkEligibility) -> bool {
        self.admits_for_sm_with_max_rows(
            e,
            atlas_core::device::active_sm_count(),
            thin_sk_max_rows(),
        )
    }
}

/// Parse [`THIN_SK_SELECTOR`] into a typed [`ThinSkPolicy`].
pub(crate) fn parse_thin_sk_policy(raw: Option<&str>) -> Result<ThinSkPolicy> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(ThinSkPolicy::Auto);
    };
    match raw.to_ascii_lowercase().as_str() {
        "0" | "off" | "false" => Ok(ThinSkPolicy::Off),
        "auto" => Ok(ThinSkPolicy::Auto),
        "1" | "on" | "true" => Ok(ThinSkPolicy::On),
        "all" => Ok(ThinSkPolicy::All),
        "force" => Ok(ThinSkPolicy::Force),
        _ => bail!("{THIN_SK_SELECTOR}: expected 0/off, auto, 1/on, all, or force; got {raw:?}"),
    }
}

/// Parse [`THIN_SK_MAX_ROWS_SELECTOR`] into a crossover row ceiling in
/// `[THIN_SK_MIN_ROWS, THIN_SK_MAX_ROWS]` (default [`THIN_SK_DEFAULT_MAX_ROWS`] = `32`).
pub(crate) fn parse_thin_sk_max_rows(raw: Option<&str>) -> Result<usize> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(THIN_SK_DEFAULT_MAX_ROWS);
    };
    let parsed = raw.parse::<usize>().map_err(|_| {
        anyhow::anyhow!(
            "{THIN_SK_MAX_ROWS_SELECTOR}: expected integer in {THIN_SK_MIN_ROWS}..={THIN_SK_MAX_ROWS}; got {raw:?}"
        )
    })?;
    ensure!(
        (THIN_SK_MIN_ROWS..=THIN_SK_MAX_ROWS).contains(&parsed),
        "{THIN_SK_MAX_ROWS_SELECTOR}: expected integer in {THIN_SK_MIN_ROWS}..={THIN_SK_MAX_ROWS}; got {parsed}"
    );
    Ok(parsed)
}

/// Read the process-wide cached [`ThinSkPolicy`] from the environment.
pub(crate) fn thin_sk_policy() -> Result<ThinSkPolicy> {
    static POLICY: OnceLock<std::result::Result<ThinSkPolicy, String>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let raw = std::env::var(THIN_SK_SELECTOR).ok();
            parse_thin_sk_policy(raw.as_deref()).map_err(|e| e.to_string())
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Read the process-wide cached SM120 crossover ceiling from
/// [`THIN_SK_MAX_ROWS_SELECTOR`] (defaulting to [`THIN_SK_DEFAULT_MAX_ROWS`] = `32`).
#[must_use]
pub(crate) fn thin_sk_max_rows() -> usize {
    static MAX_ROWS: OnceLock<usize> = OnceLock::new();
    *MAX_ROWS.get_or_init(|| {
        let raw = std::env::var(THIN_SK_MAX_ROWS_SELECTOR).ok();
        match parse_thin_sk_max_rows(raw.as_deref()) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!("{err:#}; falling back to default {THIN_SK_DEFAULT_MAX_ROWS}");
                THIN_SK_DEFAULT_MAX_ROWS
            }
        }
    })
}

/// Resolved Split-K factors, scratch offsets, and persistent launch grids for one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ThinSkPlan {
    pub sk_gu: u32,
    pub sk_dn: u32,
    pub act_bf16_bytes: usize,
    pub down_partials_offset: usize,
    pub grid_gu: u32,
    pub grid_dn: u32,
}

/// Byte size of the BF16 activated intermediate `[pairs, 640]`.
#[must_use]
pub(crate) const fn act_bf16_bytes(rows: usize) -> usize {
    rows.saturating_mul(TOPK)
        .saturating_mul(INTER)
        .saturating_mul(2)
}

/// Choose GATEUP Split-K (`4` for `rows <= 16`, `2` for `17..=47`, `1` for `>= 48`),
/// stepping down cleanly if `down_buf_bytes` cannot fit the FP32 partials `[sk, pairs, 2, 640]`.
#[must_use]
pub(crate) const fn resolve_gateup_split_k(rows: usize, down_buf_bytes: usize) -> u32 {
    let target = if rows <= 16 {
        4u32
    } else if rows <= 47 {
        2u32
    } else {
        1u32
    };
    let bytes_per_slice = rows
        .saturating_mul(TOPK)
        .saturating_mul(INTER * 2)
        .saturating_mul(4);
    if target == 4 && bytes_per_slice.saturating_mul(4) <= down_buf_bytes {
        4
    } else if target >= 2 && bytes_per_slice.saturating_mul(2) <= down_buf_bytes {
        2
    } else {
        1
    }
}

/// Choose DOWN Split-K (`2` for `rows < 8`, `1` for `rows >= 8`), stepping down cleanly if the
/// free tail of `expert_gate_out` cannot fit the FP32 partials `[sk, pairs, 2560]`.
///
/// At `rows >= 8` (`pairs >= 80`, typically `~63` active experts), `N = 2560` produces
/// `20 * ~63 = ~1,260 >= 1,128` CTAs at `sk_dn = 1` — already saturating 188-SM SM120 at
/// 6 CTAs/SM — so writing MXFP8/BF16 directly eliminates `1.64 MB` FP32 partial traffic and
/// `48` `down_reduce` launches per round with zero occupancy loss.
#[must_use]
pub(crate) const fn resolve_down_split_k(rows: usize, gate_buf_bytes: usize) -> u32 {
    if rows >= 8 {
        return 1;
    }
    let act_aligned = act_bf16_bytes(rows).next_multiple_of(PARTIALS_ALIGN_BYTES);
    let avail = gate_buf_bytes.saturating_sub(act_aligned);
    let need_sk2 = rows
        .saturating_mul(TOPK)
        .saturating_mul(HIDDEN)
        .saturating_mul(4)
        .saturating_mul(2);
    if need_sk2 <= avail { 2 } else { 1 }
}

/// Persistent CTA grid width targeting 6 CTAs/SM, bounded by `[1, min(max_tasks, abi::GRID)]`.
#[must_use]
pub(crate) fn thin_sk_persistent_grid(max_tasks: usize, sm_count: u32) -> u32 {
    let resident_cap = (sm_count.max(1) as usize)
        .saturating_mul(RESIDENT_CTAS_PER_SM)
        .min(abi::GRID as usize);
    max_tasks.min(resident_cap).max(1) as u32
}

/// Plan Split-K factors, zero-alloc scratch offsets, and persistent grids for `rows`.
pub(crate) fn plan_thin_sk(
    rows: usize,
    gate_buf_bytes: usize,
    down_buf_bytes: usize,
    sm_count: u32,
) -> Result<ThinSkPlan> {
    ensure!(
        (THIN_SK_MIN_ROWS..=THIN_SK_MAX_ROWS).contains(&rows),
        "v33_thin_sk rows={rows} outside [{THIN_SK_MIN_ROWS}, {THIN_SK_MAX_ROWS}]"
    );
    let act_bytes = act_bf16_bytes(rows);
    ensure!(
        act_bytes <= gate_buf_bytes,
        "v33_thin_sk BF16 activations ({act_bytes} B) exceed expert_gate_out ({gate_buf_bytes} B)"
    );
    let down_partials_offset = act_bytes.next_multiple_of(PARTIALS_ALIGN_BYTES);
    let sk_gu = resolve_gateup_split_k(rows, down_buf_bytes);
    let sk_dn = resolve_down_split_k(rows, gate_buf_bytes);
    let pairs = rows * TOPK;
    let max_items = pairs.min(pairs.div_ceil(128) + 512);
    let grid_gu = thin_sk_persistent_grid(max_items * GATEUP_N_TILES * (sk_gu as usize), sm_count);
    let grid_dn = thin_sk_persistent_grid(max_items * DOWN_N_TILES * (sk_dn as usize), sm_count);
    Ok(ThinSkPlan {
        sk_gu,
        sk_dn,
        act_bf16_bytes: act_bytes,
        down_partials_offset,
        grid_gu,
        grid_dn,
    })
}

/// Loaded kernel handles for all 10 `qwen4_moe_thin_sk_v33` entrypoints.
#[derive(Clone, Copy)]
pub(super) struct V33ThinSkKernels {
    pub gateup_flat: KernelHandle,
    pub gateup_ts4e: KernelHandle,
    pub down_bf16_flat: KernelHandle,
    pub down_bf16_ts4e: KernelHandle,
    pub down_mxfp8_flat: KernelHandle,
    pub down_mxfp8_ts4e: KernelHandle,
    pub reduce_silu: KernelHandle,
    pub down_reduce_bf16: KernelHandle,
    pub down_reduce_mxfp8: KernelHandle,
    pub shared_gate: KernelHandle,
}

fn load_thin_sk_kernels(gpu: &dyn GpuBackend) -> std::result::Result<V33ThinSkKernels, String> {
    let load = |name: &str| {
        gpu.kernel(THIN_SK_MODULE, name)
            .map_err(|e| format!("{THIN_SK_MODULE}::{name}: {e}"))
    };
    let gateup_flat = load("qwen4_moe_thin_sk_gateup_v33")?;
    let gateup_ts4e = load("qwen4_moe_thin_sk_gateup_v33_ts4e")?;
    let down_bf16_flat = load("qwen4_moe_thin_sk_down_bf16_v33")?;
    let down_bf16_ts4e = load("qwen4_moe_thin_sk_down_bf16_v33_ts4e")?;
    let down_mxfp8_flat = load("qwen4_moe_thin_sk_down_mxfp8_v33")?;
    let down_mxfp8_ts4e = load("qwen4_moe_thin_sk_down_mxfp8_v33_ts4e")?;
    let reduce_silu = load("qwen4_moe_thin_sk_reduce_silu_v33")?;
    let down_reduce_bf16 = load("qwen4_moe_thin_sk_down_reduce_bf16_v33")?;
    let down_reduce_mxfp8 = load("qwen4_moe_thin_sk_down_reduce_mxfp8_v33")?;
    let shared_gate = load("qwen4_moe_thin_sk_shared_gate_v33")?;
    for (handle, bytes) in [
        (gateup_flat, THIN_SK_FLAT_SMEM_BYTES),
        (down_bf16_flat, THIN_SK_FLAT_SMEM_BYTES),
        (down_mxfp8_flat, THIN_SK_FLAT_SMEM_BYTES),
        (gateup_ts4e, THIN_SK_TS4E_SMEM_BYTES),
        (down_bf16_ts4e, THIN_SK_TS4E_SMEM_BYTES),
        (down_mxfp8_ts4e, THIN_SK_TS4E_SMEM_BYTES),
    ] {
        gpu.set_kernel_max_dynamic_shared_memory(handle, bytes)
            .map_err(|e| format!("{THIN_SK_MODULE} smem={bytes}: {e}"))?;
    }
    Ok(V33ThinSkKernels {
        gateup_flat,
        gateup_ts4e,
        down_bf16_flat,
        down_bf16_ts4e,
        down_mxfp8_flat,
        down_mxfp8_ts4e,
        reduce_silu,
        down_reduce_bf16,
        down_reduce_mxfp8,
        shared_gate,
    })
}

/// Return the cached [`V33ThinSkKernels`] bundle, loading and configuring SMEM caps once.
pub(super) fn thin_sk_kernels(gpu: &dyn GpuBackend) -> Result<V33ThinSkKernels> {
    static KERNELS: OnceLock<std::result::Result<V33ThinSkKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| load_thin_sk_kernels(gpu))
        .clone()
        .map_err(|e| anyhow::anyhow!("v33_thin_sk: {e}"))
}

/// Whether the cached [`V33ThinSkKernels`] bundle is available on `gpu`.
#[must_use]
pub(crate) fn thin_sk_kernels_loaded(gpu: &dyn GpuBackend) -> bool {
    thin_sk_kernels(gpu).is_ok()
}

/// Prewarm all 10 `v33_thin_sk` kernels at model load time (CUDA graph capture safe).
pub(super) fn prewarm_thin_sk_kernels(gpu: &dyn GpuBackend) {
    if let Err(err) = thin_sk_kernels(gpu) {
        tracing::warn!("v33_thin_sk prewarm failed: {err:#}");
    }
}

/// Optional shared-expert gate scalar precomputation request for fused unpermute/blend.
#[derive(Clone, Copy)]
pub(super) struct SharedGatePrecompute {
    pub shared_gate_weight: DevicePtr,
    pub ss_out: DevicePtr,
}

/// Bundled dispatch parameters for [`launch_thin_sk_routed`].
pub(super) struct ThinSkDispatchArgs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub kernels: V33ThinSkKernels,
    pub scale_layout: RoutedScaleLayout,
    pub slot_storage: qwen4_prefill_compact::SlotStorage,
    pub input: DevicePtr,
    pub offsets: DevicePtr,
    pub sorted_ids: DevicePtr,
    pub rows: usize,
    pub gate_table: &'a ExpertPtrTable,
    pub up_table: &'a ExpertPtrTable,
    pub down_table: &'a ExpertPtrTable,
    pub contract: DevicePtr,
    pub workspace: DevicePtr,
    pub status: DevicePtr,
    pub gate_buf: DevicePtr,
    pub gate_buf_bytes: usize,
    pub down_buf: DevicePtr,
    pub down_buf_bytes: usize,
    pub shared_gate_precompute: Option<SharedGatePrecompute>,
    pub stream: u64,
}

/// Launch the full routed `v33_thin_sk` pipeline (optional shared-gate precompute +
/// GATEUP [+ `reduce_silu`] + DOWN [+ `down_reduce_{bf16,mxfp8}`]) with zero dynamic allocations.
pub(super) fn launch_thin_sk_routed(
    args: ThinSkDispatchArgs<'_>,
) -> Result<qwen4_prefill_compact::SlotStorage> {
    ensure!(
        !args.slot_storage.has_corrections(),
        "v33_thin_sk does not support MXFP8 R2 correction overlays"
    );
    let sm_count = atlas_core::device::active_sm_count();
    let plan = plan_thin_sk(
        args.rows,
        args.gate_buf_bytes,
        args.down_buf_bytes,
        sm_count,
    )?;
    let use_ts4e = args.scale_layout == RoutedScaleLayout::Ts4e128;
    let smem_bytes = if use_ts4e {
        THIN_SK_TS4E_SMEM_BYTES
    } else {
        THIN_SK_FLAT_SMEM_BYTES
    };
    let pairs = args.rows * TOPK;
    let pairs_u32 = u32::try_from(pairs).context("v33_thin_sk pairs overflow u32")?;
    let rows_u32 = u32::try_from(args.rows).context("v33_thin_sk rows overflow u32")?;

    // Fold shared-expert gate scalar precomputation into the GATEUP prologue whenever
    // `grid_gu >= rows_u32` (0-ULP bit-identical to `qwen4_moe_thin_sk_shared_gate_v33`,
    // eliminating 1 launch per MoE layer). Fall back to the standalone launch only if
    // `grid_gu < rows_u32`.
    let (fused_sg_weight, fused_sg_out, fused_sg_rows) = match args.shared_gate_precompute {
        Some(sg) if plan.grid_gu >= rows_u32 => (sg.shared_gate_weight, sg.ss_out, rows_u32),
        Some(sg) => {
            KernelLaunch::new(args.gpu, args.kernels.shared_gate)
                .grid([rows_u32, 1, 1])
                .block([SHARED_GATE_BLOCK, 1, 1])
                .arg_ptr(args.input)
                .arg_ptr(sg.shared_gate_weight)
                .arg_ptr(sg.ss_out)
                .arg_u32(rows_u32)
                .launch(args.stream)?;
            (DevicePtr::NULL, DevicePtr::NULL, 0u32)
        }
        None => (DevicePtr::NULL, DevicePtr::NULL, 0u32),
    };

    // GATEUP: `down_buf` is 100% dead before DOWN, so reuse it as zero-alloc Split-K partials.
    // All four v33_thin_sk kernels carry explicit `griddepcontrol.wait` / `launch_dependents`
    // barriers so PDL overlap across `fused_sort_plan -> gateup -> reduce_silu -> down -> down_reduce`
    // is race-free on Blackwell (cc >= 9.0).
    let gu_partials = if plan.sk_gu > 1 {
        args.down_buf
    } else {
        DevicePtr::NULL
    };
    let gateup_handle = if use_ts4e {
        args.kernels.gateup_ts4e
    } else {
        args.kernels.gateup_flat
    };
    KernelLaunch::new(args.gpu, gateup_handle)
        .grid([plan.grid_gu, 1, 1])
        .block([THIN_SK_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .pdl(true)
        .arg_ptr(args.input)
        .arg_ptr(args.gate_table.packed_ptrs)
        .arg_ptr(args.gate_table.scale_ptrs)
        .arg_ptr(args.gate_table.scale2_vals)
        .arg_ptr(args.up_table.packed_ptrs)
        .arg_ptr(args.up_table.scale_ptrs)
        .arg_ptr(args.up_table.scale2_vals)
        .arg_ptr(args.gate_buf)
        .arg_ptr(gu_partials)
        .arg_ptr(args.offsets)
        .arg_ptr(args.sorted_ids)
        .arg_ptr(args.contract)
        .arg_ptr(args.workspace)
        .arg_ptr(args.status)
        .arg_u32(INTER as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(plan.sk_gu)
        .arg_ptr(fused_sg_weight)
        .arg_ptr(fused_sg_out)
        .arg_u32(fused_sg_rows)
        .launch(args.stream)?;

    if plan.sk_gu > 1 {
        KernelLaunch::new(args.gpu, args.kernels.reduce_silu)
            .grid([pairs_u32, REDUCE_SILU_GRID_Y, 1])
            .block([REDUCE_SILU_BLOCK, 1, 1])
            .pdl(true)
            .arg_ptr(gu_partials)
            .arg_ptr(args.gate_buf)
            .arg_u32(pairs_u32)
            .arg_u32(plan.sk_gu)
            .launch(args.stream)?;
    }
    spark_runtime::cublaslt::stage_timer::mark("compact_gateup", args.stream);

    // DOWN: reuse the aligned free tail of `gate_buf` after BF16 activations for Split-K partials.
    let dn_partials = if plan.sk_dn > 1 {
        args.gate_buf.offset(plan.down_partials_offset)
    } else {
        DevicePtr::NULL
    };
    let written_storage = if args.slot_storage.is_mxfp8() {
        let layout = qwen4_prefill_compact::slot_layout(pairs, args.slot_storage)?;
        ensure!(
            layout.used_bytes <= args.down_buf_bytes && layout.used_bytes <= pairs * HIDDEN * 2,
            "v33_thin_sk MXFP8 slot rows ({} B) do not fit expert_down_out ({} B)",
            layout.used_bytes,
            args.down_buf_bytes
        );
        let slot_scales = args.down_buf.offset(
            layout
                .scales_offset
                .context("MXFP8 slot layout omitted scales")?,
        );
        let down_handle = if use_ts4e {
            args.kernels.down_mxfp8_ts4e
        } else {
            args.kernels.down_mxfp8_flat
        };
        KernelLaunch::new(args.gpu, down_handle)
            .grid([plan.grid_dn, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(smem_bytes)
            .pdl(true)
            .arg_ptr(args.gate_buf)
            .arg_ptr(args.down_table.packed_ptrs)
            .arg_ptr(args.down_table.scale_ptrs)
            .arg_ptr(args.down_table.scale2_vals)
            .arg_ptr(args.down_buf)
            .arg_ptr(slot_scales)
            .arg_ptr(dn_partials)
            .arg_ptr(args.offsets)
            .arg_ptr(args.sorted_ids)
            .arg_ptr(args.contract)
            .arg_ptr(args.workspace)
            .arg_ptr(args.status)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(plan.sk_dn)
            .launch(args.stream)?;
        if plan.sk_dn > 1 {
            KernelLaunch::new(args.gpu, args.kernels.down_reduce_mxfp8)
                .grid([pairs_u32, DOWN_REDUCE_GRID_Y, 1])
                .block([DOWN_REDUCE_BLOCK, 1, 1])
                .pdl(true)
                .arg_ptr(dn_partials)
                .arg_ptr(args.down_buf)
                .arg_ptr(slot_scales)
                .arg_u32(pairs_u32)
                .arg_u32(plan.sk_dn)
                .launch(args.stream)?;
        }
        args.slot_storage
    } else {
        let bf16_bytes = pairs * HIDDEN * 2;
        ensure!(
            bf16_bytes <= args.down_buf_bytes,
            "v33_thin_sk BF16 output ({bf16_bytes} B) exceeds expert_down_out ({} B)",
            args.down_buf_bytes
        );
        let down_handle = if use_ts4e {
            args.kernels.down_bf16_ts4e
        } else {
            args.kernels.down_bf16_flat
        };
        KernelLaunch::new(args.gpu, down_handle)
            .grid([plan.grid_dn, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(smem_bytes)
            .pdl(true)
            .arg_ptr(args.gate_buf)
            .arg_ptr(args.down_table.packed_ptrs)
            .arg_ptr(args.down_table.scale_ptrs)
            .arg_ptr(args.down_table.scale2_vals)
            .arg_ptr(args.down_buf)
            .arg_ptr(dn_partials)
            .arg_ptr(args.offsets)
            .arg_ptr(args.sorted_ids)
            .arg_ptr(args.contract)
            .arg_ptr(args.workspace)
            .arg_ptr(args.status)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(plan.sk_dn)
            .launch(args.stream)?;
        if plan.sk_dn > 1 {
            KernelLaunch::new(args.gpu, args.kernels.down_reduce_bf16)
                .grid([pairs_u32, DOWN_REDUCE_GRID_Y, 1])
                .block([DOWN_REDUCE_BLOCK, 1, 1])
                .pdl(true)
                .arg_ptr(dn_partials)
                .arg_ptr(args.down_buf)
                .arg_u32(pairs_u32)
                .arg_u32(plan.sk_dn)
                .launch(args.stream)?;
        }
        qwen4_prefill_compact::SlotStorage::Bf16
    };
    spark_runtime::cublaslt::stage_timer::mark("compact_down", args.stream);

    if !THIN_SK_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "MOE_VERIFY_THIN_SK_V33_ENGAGED selector={THIN_SK_SELECTOR} rows={} sk_gu={} sk_dn={} grid_gu={} grid_dn={} block={THIN_SK_BLOCK} ts4e={use_ts4e} slot_mxfp8={} fused_shared_gate={} smem={smem_bytes}",
            args.rows,
            plan.sk_gu,
            plan.sk_dn,
            plan.grid_gu,
            plan.grid_dn,
            written_storage.is_mxfp8(),
            fused_sg_rows > 0,
        );
    }
    Ok(written_storage)
}

/// Execute the selected `moe_unpermute_reduce_blend_mxfp8_bf16[_w4]` kernel launch.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the exact 13-argument CUDA ABI of moe_unpermute_reduce_blend_mxfp8_bf16[_w4]"
)]
pub(super) fn launch_unpermute_blend_kernel(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    rows: u32,
    grid_y: u32,
    block_x: u32,
    slot_q: DevicePtr,
    slot_s: DevicePtr,
    shared_out: DevicePtr,
    ss_ptr: DevicePtr,
    output: DevicePtr,
    token_to_perm: DevicePtr,
    topk_weights: DevicePtr,
    hidden: u32,
    top_k: u32,
    routed_status: DevicePtr,
    shared_status: DevicePtr,
    sticky_err: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, grid_y, 1])
        .block([block_x, 1, 1])
        .pdl(atlas_core::device::active_sm_count() > THIN_SK_HIGH_SM_THRESHOLD)
        .arg_ptr(slot_q)
        .arg_ptr(slot_s)
        .arg_ptr(shared_out)
        .arg_ptr(ss_ptr)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_weights)
        .arg_u32(hidden)
        .arg_u32(rows)
        .arg_u32(top_k)
        .arg_ptr(routed_status)
        .arg_ptr(shared_status)
        .arg_ptr(sticky_err)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_eligibility(rows: usize) -> ThinSkEligibility {
        ThinSkEligibility {
            rows,
            w4a8_is_v28: true,
            has_corrections: false,
            v29_m256_active: false,
            oracle_capture: false,
            kernels_loaded: true,
        }
    }

    #[test]
    fn policy_defaults_auto_and_parses_all_documented_values() {
        assert_eq!(parse_thin_sk_policy(None).unwrap(), ThinSkPolicy::Auto);
        assert_eq!(parse_thin_sk_policy(Some("")).unwrap(), ThinSkPolicy::Auto);
        assert_eq!(
            parse_thin_sk_policy(Some(" auto ")).unwrap(),
            ThinSkPolicy::Auto
        );
        assert_eq!(parse_thin_sk_policy(Some("0")).unwrap(), ThinSkPolicy::Off);
        assert_eq!(
            parse_thin_sk_policy(Some("off")).unwrap(),
            ThinSkPolicy::Off
        );
        assert_eq!(
            parse_thin_sk_policy(Some("FALSE")).unwrap(),
            ThinSkPolicy::Off
        );
        assert_eq!(parse_thin_sk_policy(Some("1")).unwrap(), ThinSkPolicy::On);
        assert_eq!(parse_thin_sk_policy(Some("on")).unwrap(), ThinSkPolicy::On);
        assert_eq!(
            parse_thin_sk_policy(Some(" True ")).unwrap(),
            ThinSkPolicy::On
        );
        assert_eq!(
            parse_thin_sk_policy(Some("ALL")).unwrap(),
            ThinSkPolicy::All
        );
        assert_eq!(
            parse_thin_sk_policy(Some("force")).unwrap(),
            ThinSkPolicy::Force
        );
        assert!(parse_thin_sk_policy(Some("2")).is_err());
        assert!(parse_thin_sk_policy(Some("bogus")).is_err());

        // Crossover ceiling override parser.
        assert_eq!(
            parse_thin_sk_max_rows(None).unwrap(),
            THIN_SK_DEFAULT_MAX_ROWS
        );
        assert_eq!(
            parse_thin_sk_max_rows(Some("")).unwrap(),
            THIN_SK_DEFAULT_MAX_ROWS
        );
        assert_eq!(parse_thin_sk_max_rows(Some(" 64 ")).unwrap(), 64);
        assert_eq!(parse_thin_sk_max_rows(Some("128")).unwrap(), 128);
        assert_eq!(parse_thin_sk_max_rows(Some("256")).unwrap(), 256);
        assert!(parse_thin_sk_max_rows(Some("3")).is_err());
        assert!(parse_thin_sk_max_rows(Some("257")).is_err());
        assert!(parse_thin_sk_max_rows(Some("abc")).is_err());
    }

    #[test]
    fn admission_keeps_gb10_unchanged_and_applies_sm120_32_row_crossover() {
        for rows in [
            0, 1, 3, 4, 8, 16, 32, 33, 64, 96, 97, 128, 192, 256, 257, 512,
        ] {
            let e = base_eligibility(rows);
            // Explicit OFF never engages on either GB10 or SM120.
            assert!(!ThinSkPolicy::Off.admits_for_sm(e, 48));
            assert!(!ThinSkPolicy::Off.admits_for_sm(e, 188));
            // Auto / On / All NEVER alter GB10 (<= 64 SMs).
            for gb10_sms in [48u32, 64u32] {
                assert!(!ThinSkPolicy::Auto.admits_for_sm(e, gb10_sms));
                assert!(!ThinSkPolicy::On.admits_for_sm(e, gb10_sms));
                assert!(!ThinSkPolicy::All.admits_for_sm(e, gb10_sms));
            }
            // Auto & On engage on SM120 (> 64 SMs) strictly for rows in 4..=32
            // (rows >= 33 hand off to V31_WS).
            let in_crossover = (4..=THIN_SK_DEFAULT_MAX_ROWS).contains(&rows);
            assert_eq!(
                ThinSkPolicy::Auto.admits_for_sm(e, 188),
                in_crossover,
                "Auto sm120 rows={rows}"
            );
            assert_eq!(
                ThinSkPolicy::On.admits_for_sm(e, 188),
                in_crossover,
                "On sm120 rows={rows}"
            );
            // All & Force cover full 4..=256 range for A/B sweeps.
            let in_full = (4..=256).contains(&rows);
            assert_eq!(
                ThinSkPolicy::All.admits_for_sm(e, 188),
                in_full,
                "All sm120 rows={rows}"
            );
            assert_eq!(
                ThinSkPolicy::Force.admits_for_sm(e, 48),
                in_full,
                "Force gb10 rows={rows}"
            );
        }

        // Explicit crossover pins: rows 16, 32 -> admit; 48, 64, 96, 128, 192, 256 -> false (-> V31_WS).
        assert!(ThinSkPolicy::Auto.admits_for_sm(base_eligibility(16), 188));
        assert!(ThinSkPolicy::Auto.admits_for_sm(base_eligibility(32), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(48), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(64), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(96), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(128), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(192), 188));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(base_eligibility(256), 188));

        // Custom max_rows override shifts the crossover ceiling cleanly.
        assert!(ThinSkPolicy::Auto.admits_for_sm_with_max_rows(base_eligibility(128), 188, 128));
        assert!(!ThinSkPolicy::Auto.admits_for_sm_with_max_rows(base_eligibility(129), 188, 128));
        assert!(!ThinSkPolicy::Auto.admits_for_sm_with_max_rows(base_eligibility(64), 188, 32));

        // Guard rails reject incompatible states even on SM120.
        let ok = base_eligibility(16);
        assert!(!ThinSkPolicy::Auto.admits_for_sm(
            ThinSkEligibility {
                w4a8_is_v28: false,
                ..ok
            },
            188
        ));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(
            ThinSkEligibility {
                has_corrections: true,
                ..ok
            },
            188
        ));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(
            ThinSkEligibility {
                v29_m256_active: true,
                ..ok
            },
            188
        ));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(
            ThinSkEligibility {
                oracle_capture: true,
                ..ok
            },
            188
        ));
        assert!(!ThinSkPolicy::Auto.admits_for_sm(
            ThinSkEligibility {
                kernels_loaded: false,
                ..ok
            },
            188
        ));
    }

    #[test]
    fn split_k_schedule_and_scratch_step_down_are_bounds_safe() {
        // Large arena (k_max = 16384): 209.7 MiB gate, 432.5 MiB down.
        let gate_big = 16_384 * TOPK * INTER * 2;
        let down_big = 16_384 * TOPK * 2640;

        for (rows, exp_gu, exp_dn) in [
            (4, 4, 2),
            (7, 4, 2),
            (8, 4, 1),
            (16, 4, 1),
            (17, 2, 1),
            (24, 2, 1),
            (25, 2, 1),
            (32, 2, 1),
            (47, 2, 1),
            (48, 1, 1),
            (64, 1, 1),
            (96, 1, 1),
            (97, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
        ] {
            let p = plan_thin_sk(rows, gate_big, down_big, 188).unwrap();
            assert_eq!((p.sk_gu, p.sk_dn), (exp_gu, exp_dn), "rows={rows}");
            assert_eq!(p.down_partials_offset % PARTIALS_ALIGN_BYTES, 0);
            assert!(p.grid_gu >= 1 && p.grid_gu <= 188 * 6);
            assert!(p.grid_dn >= 1 && p.grid_dn <= 188 * 6);
        }

        // Tiny scratch arena steps Split-K down cleanly instead of overflowing.
        // At rows=16, bytes_per_slice = 16 * 10 * 1280 * 4 = 819,200 B (sk=4 needs 3.125 MiB, sk=2 needs 1.5625 MiB).
        assert_eq!(
            plan_thin_sk(16, gate_big, 1024 * 1024, 188).unwrap().sk_gu,
            1
        );
        assert_eq!(
            plan_thin_sk(16, gate_big, 2 * 1024 * 1024, 188)
                .unwrap()
                .sk_gu,
            2
        );
        assert_eq!(
            plan_thin_sk(16, gate_big, 4 * 1024 * 1024, 188)
                .unwrap()
                .sk_gu,
            4
        );

        let p_tiny_down = plan_thin_sk(7, gate_big, 2 * 1024 * 1024, 188).unwrap();
        assert_eq!(p_tiny_down.sk_gu, 4);
        assert_eq!(p_tiny_down.sk_dn, 2);

        let p_mid_down = plan_thin_sk(24, gate_big, 3 * 1024 * 1024, 188).unwrap();
        assert_eq!(p_mid_down.sk_gu, 2);
        assert_eq!(p_mid_down.sk_dn, 1);

        let p_tiny_gate = plan_thin_sk(7, act_bf16_bytes(7), down_big, 188).unwrap();
        assert_eq!(p_tiny_gate.sk_dn, 1);

        // Insufficient gate buffer for activations itself returns an error.
        assert!(plan_thin_sk(16, act_bf16_bytes(16) - 1, down_big, 188).is_err());
        // Out-of-range rows rejected.
        assert!(plan_thin_sk(3, gate_big, down_big, 188).is_err());
        assert!(plan_thin_sk(257, gate_big, down_big, 188).is_err());
    }

    #[test]
    fn smem_contract_stays_under_12_kib_and_persistent_grid_scales_with_sms() {
        const {
            assert!(THIN_SK_FLAT_SMEM_BYTES <= 12 * 1024);
            assert!(THIN_SK_TS4E_SMEM_BYTES <= 12 * 1024);
        }
        assert_eq!(thin_sk_persistent_grid(3200, 188), 1128);
        assert_eq!(thin_sk_persistent_grid(3200, 48), 288);
        assert_eq!(thin_sk_persistent_grid(400, 188), 400);
        assert_eq!(thin_sk_persistent_grid(0, 188), 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_v33_sk_dn1_vs_sk_dn2_and_fused_shared_gate() {
        use spark_runtime::cuda_backend::AtlasCudaBackend;

        let Some(ptx) = atlas_kernels::ptx_for_model("flash-next") else {
            eprintln!("skipping GPU parity test: flash-next PTX unavailable");
            return;
        };
        let Ok(gpu) = AtlasCudaBackend::new(0, &ptx.modules) else {
            eprintln!("skipping GPU parity test: CUDA device unavailable");
            return;
        };
        let Ok(kernels) = load_thin_sk_kernels(&gpu) else {
            eprintln!("skipping GPU parity test: qwen4_moe_thin_sk_v33 kernels unavailable");
            return;
        };
        let slice_as_bytes = |s: &[u16]| -> &[u8] {
            // SAFETY: `s` is a valid initialized `&[u16]` POD slice with no padding bytes;
            // `u8` has alignment 1 and the returned slice borrows `s`.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        };
        let slice_as_bytes_mut_u16 = |s: &mut [u16]| -> &mut [u8] {
            // SAFETY: `s` is a valid initialized `&mut [u16]` POD slice with no padding bytes;
            // any written byte pattern is a valid `u16` and `u8` has alignment 1.
            unsafe {
                std::slice::from_raw_parts_mut(
                    s.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(s),
                )
            }
        };
        let i32_as_bytes = |s: &[i32]| -> &[u8] {
            // SAFETY: `s` is a valid initialized `&[i32]` POD slice with no padding bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        };
        let u64_as_bytes = |s: &[u64]| -> &[u8] {
            // SAFETY: `s` is a valid initialized `&[u64]` POD slice with no padding bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        };
        let f32_as_bytes = |s: &[f32]| -> &[u8] {
            // SAFETY: `s` is a valid initialized `&[f32]` POD slice with no padding bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        };
        let f32_as_bytes_mut = |s: &mut [f32]| -> &mut [u8] {
            // SAFETY: `s` is a valid initialized `&mut [f32]` POD slice with no padding bytes;
            // any written byte pattern is a valid `f32` bit pattern.
            unsafe {
                std::slice::from_raw_parts_mut(
                    s.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(s),
                )
            }
        };

        let stream = gpu.default_stream();
        let rows: usize = 8;
        let pairs: usize = rows * TOPK;
        let num_experts: usize = 512;

        // Deterministic synthetic routing:
        // - Expert 0 gets 8 tokens (exercises hot Path B MMA M=8)
        // - Expert 1 gets 4 tokens (exercises thin Path A RM=4)
        // - Expert 2 gets 2 tokens (exercises thin Path A RM=2)
        // - Experts 3..68 get 1 token each (exercises thin Path A RM=1, 66 active experts)
        let mut offsets = vec![0i32; num_experts + 1];
        let mut expert_counts = vec![0usize; num_experts];
        expert_counts[0] = 8;
        expert_counts[1] = 4;
        expert_counts[2] = 2;
        for c in &mut expert_counts[3..69] {
            *c = 1;
        }
        let mut work_items: Vec<(u32, u32)> = Vec::new();
        let mut cursor = 0i32;
        for (e, &cnt) in expert_counts.iter().enumerate() {
            offsets[e] = cursor;
            if cnt > 0 {
                work_items.push((e as u32, 0));
            }
            cursor += cnt as i32;
        }
        offsets[num_experts] = cursor;
        assert_eq!(cursor as usize, pairs);

        let sorted_ids: Vec<i32> = (0..pairs).map(|i| (i % rows) as i32).collect();

        // Build valid Contract (64 B) and Workspace (86,064 B).
        let contract_bytes = abi::contract(rows).unwrap();
        let mut workspace_bytes = vec![0u8; abi::WORKSPACE_BYTES];
        workspace_bytes[0..8].copy_from_slice(&0x4f52494749363430u64.to_le_bytes());
        workspace_bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // version
        workspace_bytes[12..16].copy_from_slice(&(rows as u32).to_le_bytes());
        workspace_bytes[16..20].copy_from_slice(&(pairs as u32).to_le_bytes());
        workspace_bytes[20..24].copy_from_slice(&(work_items.len() as u32).to_le_bytes());
        workspace_bytes[24..28].copy_from_slice(&(work_items.len() as u32).to_le_bytes());
        workspace_bytes[28..32].copy_from_slice(&8u32.to_le_bytes()); // max_rows
        // next_task_gu (32..36) and next_task_dn (36..40) set before each run.
        for (idx, &(exp, mt)) in work_items.iter().enumerate() {
            let off = 48 + idx * 8;
            workspace_bytes[off..off + 4].copy_from_slice(&exp.to_le_bytes());
            workspace_bytes[off + 4..off + 8].copy_from_slice(&mt.to_le_bytes());
        }

        // Deterministic BF16 inputs and shared gate weights.
        let bf16_from_f32 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let f32_from_bf16 = |b: u16| -> f32 { f32::from_bits((b as u32) << 16) };

        let mut x_host = vec![0u16; rows * HIDDEN];
        for (i, slot) in x_host.iter_mut().enumerate() {
            let v = (((i.wrapping_mul(37) % 97) as f32) - 48.0) * 0.015;
            *slot = bf16_from_f32(v);
        }
        let mut sg_host = vec![0u16; HIDDEN];
        for (i, slot) in sg_host.iter_mut().enumerate() {
            let v = (((i.wrapping_mul(53) % 89) as f32) - 44.0) * 0.012;
            *slot = bf16_from_f32(v);
        }
        let mut act_host = vec![0u16; pairs * INTER];
        for (i, slot) in act_host.iter_mut().enumerate() {
            let v = (((i.wrapping_mul(29) % 73) as f32) - 36.0) * 0.025;
            *slot = bf16_from_f32(v);
        }

        // Shared single NVFP4 expert weight buffer reused across all 512 expert slots
        // (valid because each active expert reads read-only weight pointers).
        // GATE/UP shape: K=2560, N=640 -> packed [K/2, N] = 819,200 B, flat scales [K/16, N] = 102,400 B.
        // DOWN shape: K=640, N=2560 -> packed [K/2, N] = 819,200 B, flat scales [K/16, N] = 102,400 B.
        let packed_bytes = (HIDDEN / 2) * INTER;
        let scale_bytes = (HIDDEN / 16) * INTER;
        let mut packed_host = vec![0u8; packed_bytes];
        for (i, b) in packed_host.iter_mut().enumerate() {
            // Valid E2M1 nibbles (avoid -0 / 0x88 ambiguity; rich mix of positive & negative codes).
            let lo = ((i.wrapping_mul(17) + 3) & 0x7) as u8 | (((i >> 3) & 1) as u8) << 3;
            let hi = ((i.wrapping_mul(31) + 5) & 0x7) as u8 | (((i >> 4) & 1) as u8) << 3;
            *b = lo | (hi << 4);
        }
        // Valid FP8 E4M3 positive scales around [0.25..2.0] (0x30..0x40).
        let mut scale_host = vec![0u8; scale_bytes];
        for (i, b) in scale_host.iter_mut().enumerate() {
            *b = 0x30u8 + ((i.wrapping_mul(13) % 16) as u8);
        }

        let x_dev = gpu.alloc(x_host.len() * 2).unwrap();
        let sg_dev = gpu.alloc(sg_host.len() * 2).unwrap();
        let act_dev = gpu.alloc(act_host.len() * 2).unwrap();
        let packed_dev = gpu.alloc(packed_bytes).unwrap();
        let scale_dev = gpu.alloc(scale_bytes).unwrap();
        let offsets_dev = gpu.alloc(offsets.len() * 4).unwrap();
        let sorted_dev = gpu.alloc(sorted_ids.len() * 4).unwrap();
        let contract_dev = gpu.alloc(abi::CONTRACT_BYTES).unwrap();
        let workspace_dev = gpu.alloc(abi::WORKSPACE_BYTES).unwrap();
        let status_dev = gpu.alloc(4).unwrap();
        let ss_ref_dev = gpu.alloc(rows * 4).unwrap();
        let ss_fused_dev = gpu.alloc(rows * 4).unwrap();
        let ss_null_dev = gpu.alloc(rows * 4).unwrap();
        let gate_out_dev = gpu.alloc(pairs * INTER * 2).unwrap();
        let partials_gu_dev = gpu.alloc(4 * pairs * INTER * 2 * 4).unwrap();
        let partials_dn_dev = gpu.alloc(2 * pairs * HIDDEN * 4).unwrap();
        let bf16_sk1_dev = gpu.alloc(pairs * HIDDEN * 2).unwrap();
        let bf16_sk2_dev = gpu.alloc(pairs * HIDDEN * 2).unwrap();
        let mxfp8_q_sk1_dev = gpu.alloc(pairs * HIDDEN).unwrap();
        let mxfp8_s_sk1_dev = gpu.alloc(pairs * (HIDDEN / 32)).unwrap();
        let mxfp8_q_sk2_dev = gpu.alloc(pairs * HIDDEN).unwrap();
        let mxfp8_s_sk2_dev = gpu.alloc(pairs * (HIDDEN / 32)).unwrap();

        let packed_ptrs_host = vec![packed_dev.0; num_experts];
        let scale_ptrs_host = vec![scale_dev.0; num_experts];
        let scale2_host = vec![0.125f32; num_experts];
        let packed_ptrs_dev = gpu.alloc(num_experts * 8).unwrap();
        let scale_ptrs_dev = gpu.alloc(num_experts * 8).unwrap();
        let scale2_dev = gpu.alloc(num_experts * 4).unwrap();

        let upload = |dst: DevicePtr, bytes: &[u8]| {
            gpu.copy_h2d_async(bytes, dst, stream).unwrap();
        };
        upload(x_dev, slice_as_bytes(&x_host));
        upload(sg_dev, slice_as_bytes(&sg_host));
        upload(act_dev, slice_as_bytes(&act_host));
        upload(packed_dev, &packed_host);
        upload(scale_dev, &scale_host);
        upload(offsets_dev, i32_as_bytes(&offsets));
        upload(sorted_dev, i32_as_bytes(&sorted_ids));
        upload(contract_dev, &contract_bytes);
        upload(packed_ptrs_dev, u64_as_bytes(&packed_ptrs_host));
        upload(scale_ptrs_dev, u64_as_bytes(&scale_ptrs_host));
        upload(scale2_dev, f32_as_bytes(&scale2_host));
        upload(status_dev, &1i32.to_le_bytes());

        // ------------------------------------------------------------------
        // 1. Fused shared_gate in gateup_v33 vs standalone shared_gate_v33:
        //    must be 0-ULP BIT-IDENTICAL.
        // ------------------------------------------------------------------
        let grid_gu = thin_sk_persistent_grid(work_items.len() * GATEUP_N_TILES * 4, 48);
        assert!(grid_gu >= rows as u32);
        workspace_bytes[32..36].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);

        KernelLaunch::new(&gpu, kernels.shared_gate)
            .grid([rows as u32, 1, 1])
            .block([SHARED_GATE_BLOCK, 1, 1])
            .arg_ptr(x_dev)
            .arg_ptr(sg_dev)
            .arg_ptr(ss_ref_dev)
            .arg_u32(rows as u32)
            .launch(stream)
            .unwrap();

        KernelLaunch::new(&gpu, kernels.gateup_flat)
            .grid([grid_gu, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(x_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(gate_out_dev)
            .arg_ptr(partials_gu_dev)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(4u32)
            .arg_ptr(sg_dev)
            .arg_ptr(ss_fused_dev)
            .arg_u32(rows as u32)
            .launch(stream)
            .unwrap();

        // Also test NULL shared_gate weight -> writes 1.0f cleanly.
        workspace_bytes[32..36].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);
        KernelLaunch::new(&gpu, kernels.gateup_flat)
            .grid([grid_gu, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(x_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(gate_out_dev)
            .arg_ptr(partials_gu_dev)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(INTER as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32(4u32)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ss_null_dev)
            .arg_u32(rows as u32)
            .launch(stream)
            .unwrap();

        // ------------------------------------------------------------------
        // 2. DOWN sk_dn=1 (direct) vs sk_dn=2 (+ down_reduce), BF16 & MXFP8.
        // ------------------------------------------------------------------
        let grid_dn_sk1 = thin_sk_persistent_grid(work_items.len() * DOWN_N_TILES, 48);
        let grid_dn_sk2 = thin_sk_persistent_grid(work_items.len() * DOWN_N_TILES * 2, 48);

        // BF16 sk_dn=1
        workspace_bytes[36..40].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);
        KernelLaunch::new(&gpu, kernels.down_bf16_flat)
            .grid([grid_dn_sk1, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(act_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(bf16_sk1_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(1u32)
            .launch(stream)
            .unwrap();

        // BF16 sk_dn=2 + down_reduce_bf16
        workspace_bytes[36..40].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);
        KernelLaunch::new(&gpu, kernels.down_bf16_flat)
            .grid([grid_dn_sk2, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(act_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(bf16_sk2_dev)
            .arg_ptr(partials_dn_dev)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(2u32)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, kernels.down_reduce_bf16)
            .grid([pairs as u32, DOWN_REDUCE_GRID_Y, 1])
            .block([DOWN_REDUCE_BLOCK, 1, 1])
            .pdl(true)
            .arg_ptr(partials_dn_dev)
            .arg_ptr(bf16_sk2_dev)
            .arg_u32(pairs as u32)
            .arg_u32(2u32)
            .launch(stream)
            .unwrap();

        // MXFP8 sk_dn=1
        workspace_bytes[36..40].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);
        KernelLaunch::new(&gpu, kernels.down_mxfp8_flat)
            .grid([grid_dn_sk1, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(act_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(mxfp8_q_sk1_dev)
            .arg_ptr(mxfp8_s_sk1_dev)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(1u32)
            .launch(stream)
            .unwrap();

        // MXFP8 sk_dn=2 + down_reduce_mxfp8
        workspace_bytes[36..40].copy_from_slice(&0u32.to_le_bytes());
        upload(workspace_dev, &workspace_bytes);
        KernelLaunch::new(&gpu, kernels.down_mxfp8_flat)
            .grid([grid_dn_sk2, 1, 1])
            .block([THIN_SK_BLOCK, 1, 1])
            .shared_mem(THIN_SK_FLAT_SMEM_BYTES)
            .pdl(true)
            .arg_ptr(act_dev)
            .arg_ptr(packed_ptrs_dev)
            .arg_ptr(scale_ptrs_dev)
            .arg_ptr(scale2_dev)
            .arg_ptr(mxfp8_q_sk2_dev)
            .arg_ptr(mxfp8_s_sk2_dev)
            .arg_ptr(partials_dn_dev)
            .arg_ptr(offsets_dev)
            .arg_ptr(sorted_dev)
            .arg_ptr(contract_dev)
            .arg_ptr(workspace_dev)
            .arg_ptr(status_dev)
            .arg_u32(HIDDEN as u32)
            .arg_u32(INTER as u32)
            .arg_u32(2u32)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&gpu, kernels.down_reduce_mxfp8)
            .grid([pairs as u32, DOWN_REDUCE_GRID_Y, 1])
            .block([DOWN_REDUCE_BLOCK, 1, 1])
            .pdl(true)
            .arg_ptr(partials_dn_dev)
            .arg_ptr(mxfp8_q_sk2_dev)
            .arg_ptr(mxfp8_s_sk2_dev)
            .arg_u32(pairs as u32)
            .arg_u32(2u32)
            .launch(stream)
            .unwrap();

        gpu.synchronize(stream).unwrap();

        let mut status_out = [0u8; 4];
        gpu.copy_d2h(status_dev, &mut status_out).unwrap();
        assert_eq!(
            i32::from_le_bytes(status_out),
            1,
            "v33 status must stay PLANNED"
        );

        // Verify 0-ULP bit-identical fused shared_gate output.
        let mut ss_ref = vec![0f32; rows];
        let mut ss_fused = vec![0f32; rows];
        let mut ss_null = vec![0f32; rows];
        gpu.copy_d2h(ss_ref_dev, f32_as_bytes_mut(&mut ss_ref))
            .unwrap();
        gpu.copy_d2h(ss_fused_dev, f32_as_bytes_mut(&mut ss_fused))
            .unwrap();
        gpu.copy_d2h(ss_null_dev, f32_as_bytes_mut(&mut ss_null))
            .unwrap();
        for r in 0..rows {
            assert_eq!(
                ss_fused[r].to_bits(),
                ss_ref[r].to_bits(),
                "fused shared_gate row {r} differed: fused={} ref={}",
                ss_fused[r],
                ss_ref[r]
            );
            assert_eq!(ss_null[r].to_bits(), 1.0f32.to_bits());
        }

        // Verify BF16 DOWN sk_dn=1 vs sk_dn=2 (<= 1 BF16 ULP, zero NaN/Inf).
        let mut bf16_sk1 = vec![0u16; pairs * HIDDEN];
        let mut bf16_sk2 = vec![0u16; pairs * HIDDEN];
        gpu.copy_d2h(bf16_sk1_dev, slice_as_bytes_mut_u16(&mut bf16_sk1))
            .unwrap();
        gpu.copy_d2h(bf16_sk2_dev, slice_as_bytes_mut_u16(&mut bf16_sk2))
            .unwrap();
        let mut max_bf16_ulp = 0u32;
        let mut nonzero_count = 0usize;
        for (&a, &b) in bf16_sk1.iter().zip(bf16_sk2.iter()) {
            let fa = f32_from_bf16(a);
            let fb = f32_from_bf16(b);
            assert!(fa.is_finite() && fb.is_finite());
            if fa != 0.0 {
                nonzero_count += 1;
            }
            let ulp = if a == b || (fa == 0.0 && fb == 0.0) {
                0
            } else if (a ^ b) & 0x8000 == 0 {
                a.abs_diff(b) as u32
            } else {
                ((a & 0x7fff) as u32) + ((b & 0x7fff) as u32)
            };
            max_bf16_ulp = max_bf16_ulp.max(ulp);
        }
        assert!(
            nonzero_count > pairs * HIDDEN / 2,
            "expected non-trivial outputs"
        );
        assert!(
            max_bf16_ulp <= 1,
            "DOWN BF16 sk_dn=1 vs sk_dn=2 exceeded 1 BF16 ULP: max_ulp={max_bf16_ulp}"
        );

        // Verify MXFP8 DOWN sk_dn=1 vs sk_dn=2 (exact or <= 1 E4M3 ULP).
        let mut q1 = vec![0u8; pairs * HIDDEN];
        let mut s1 = vec![0u8; pairs * (HIDDEN / 32)];
        let mut q2 = vec![0u8; pairs * HIDDEN];
        let mut s2 = vec![0u8; pairs * (HIDDEN / 32)];
        gpu.copy_d2h(mxfp8_q_sk1_dev, &mut q1).unwrap();
        gpu.copy_d2h(mxfp8_s_sk1_dev, &mut s1).unwrap();
        gpu.copy_d2h(mxfp8_q_sk2_dev, &mut q2).unwrap();
        gpu.copy_d2h(mxfp8_s_sk2_dev, &mut s2).unwrap();
        let mut mxfp8_mismatches = 0usize;
        let mut max_q_diff = 0u8;
        let mut max_s_diff = 0u8;
        for i in 0..s1.len() {
            max_s_diff = max_s_diff.max(s1[i].abs_diff(s2[i]));
        }
        for i in 0..q1.len() {
            if q1[i] != q2[i] {
                mxfp8_mismatches += 1;
                max_q_diff = max_q_diff.max(q1[i].abs_diff(q2[i]));
            }
        }
        assert_eq!(
            max_s_diff, 0,
            "MXFP8 block scales differed between sk_dn=1 and sk_dn=2"
        );
        assert!(
            max_q_diff <= 1 && mxfp8_mismatches * 1000 <= q1.len(),
            "MXFP8 sk_dn=1 vs sk_dn=2 drift: mismatches={mxfp8_mismatches}/{} max_q_diff={max_q_diff}",
            q1.len()
        );

        for ptr in [
            x_dev,
            sg_dev,
            act_dev,
            packed_dev,
            scale_dev,
            offsets_dev,
            sorted_dev,
            contract_dev,
            workspace_dev,
            status_dev,
            ss_ref_dev,
            ss_fused_dev,
            ss_null_dev,
            gate_out_dev,
            partials_gu_dev,
            partials_dn_dev,
            bf16_sk1_dev,
            bf16_sk2_dev,
            mxfp8_q_sk1_dev,
            mxfp8_s_sk1_dev,
            mxfp8_q_sk2_dev,
            mxfp8_s_sk2_dev,
            packed_ptrs_dev,
            scale_ptrs_dev,
            scale2_dev,
        ] {
            let _ = gpu.free(ptr);
        }
    }
}
