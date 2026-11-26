// SPDX-License-Identifier: AGPL-3.0-only

//! HC_FUSED's rstd pass, per-stream down|inject projections and their combine
//! in one kernel per preparation (`ATLAS_QWEN4_PREFILL_MHC_FUSED=injgemm`,
//! default off). The pending injection (a core output, a deferred MoE output,
//! or none) is applied, normalised and projected while each hyper row tile is
//! on chip, so the four cuBLASLt GEMMs no longer re-read the hyper rows and
//! their f32 partials never reach DRAM (`qwen4_hc_combine_gemm.cu`). When
//! `ATLAS_QWEN4_PREFILL_HC_FUSED=0`, this is
//! gated to row counts whose pinned cuBLASLt algorithm accumulates as one
//! ascending-k16 chain (see [`super::prefill_inject_gemm_admit`]); other row
//! counts keep the GEMMs. Setting `ATLAS_QWEN4_PREFILL_HC_FUSED != 0`
//! explicitly opts out of that split-K parity gate and admits arbitrary row
//! counts.
//!
//! A pending MoE output (lever `combine`) is formed by `qwen4_hc_combine[_mx]_inject_rstd`,
//! after which the plain pass re-reads the injected rows. Levers `postpre` /
//! `postsplit` form it once per row into a scratch buffer instead and inject it
//! like a core output, in the projection kernel's own prologue
//! (`qwen4_hc_combine_moe_gemm[_mx]`) or one kernel earlier (`qwen4_hc_moe_rows[_mx]`).
//! In the single postpre kernel a tile's prologue may run after another tile's
//! epilogue, so it reads its routing map from a copy (see [`postpre_reads_disjoint`]).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Qwen4HyperConnection;
use super::pending::{MoeCombine, PendingInject};
use super::prefill_fused::reference_rows;
use super::prefill_inject::{SELECTOR, disjoint, selection};
use super::prefill_inject_gemm_admit::{HIDDEN, RANK, STREAMS, gemm_order_verified};

const TILE_ROWS: u32 = 64;
const TILE_ROWS_CLUSTER_FUSE: u32 = 32;
const THREADS: u32 = 512;
/// `kSmemBytes` in qwen4_hc_combine_gemm.cu (static_assert there).
const SMEM_BYTES: u32 = 100_416;
/// `hc_cluster_fuse::kSmemBytesF` in qwen4_hc_combine_gemm.cu (static_assert there).
const SMEM_BYTES_CLUSTER_FUSE: u32 = 92_672;
const TILE_ROWS_2CTA: u32 = 32;
const THREADS_2CTA: u32 = 256;
/// `hc_2cta::kSmemBytes2` in qwen4_hc_combine_gemm.cu (static_assert there).
const SMEM_BYTES_2CTA: u32 = 47_232;
/// Selects the 2-CTA/SM `qwen4_hc_inject_gemm_ws[_l2h]_gatepre_2cta` kernels
/// (default off on GB10 unstriped prefill because 1-CTA BM=32 halves folded-weight
/// DRAM traffic; set `1`/`on`/`true` to opt in).
const INJECT_2CTA_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_INJECT_2CTA";

fn parse_inject_2cta(value: Option<&str>) -> Result<bool, String> {
    let Some(raw) = value else {
        return Ok(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => Ok(false),
        "1" | "on" | "true" => Ok(true),
        other => Err(format!(
            "{INJECT_2CTA_SELECTOR}: unknown value {other:?} (expected 0, 1, off, on, false, true)"
        )),
    }
}

fn inject_2cta_enabled() -> Result<bool> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_inject_2cta(std::env::var(INJECT_2CTA_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// The warp-specialised cluster kernels of qwen4_hc_combine_gemm.cu: `inject`
/// (`qwen4_hc_inject_gemm_ws_l2h`, L2 row prefetch) for a pending core
/// injection, `plain` (`qwen4_hc_inject_gemm_ws`) for a read-only pass, plus
/// the 2-CTA/SM (`47,232 B` smem) `gatepre_2cta` twins. Two kernels on
/// purpose: one kernel choosing at run time ran 7.36 ms (micro9).
#[derive(Clone, Copy)]
struct Kernels {
    inject: KernelHandle,
    plain: KernelHandle,
    inject_gatepre: KernelHandle,
    plain_gatepre: KernelHandle,
    inject_gatepre_2cta: Option<KernelHandle>,
    plain_gatepre_2cta: Option<KernelHandle>,
}

fn kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |name: &str| {
            let k = gpu
                .kernel("qwen4_hc_combine_gemm", name)
                .map_err(|e| format!("{name}: {e}"))?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES)
                .map_err(|e| format!("{name}: {e}"))?;
            Ok::<_, String>(k)
        };
        let load_2cta = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_2CTA)
                .ok()?;
            Some(k)
        };
        Ok(Kernels {
            inject: load("qwen4_hc_inject_gemm_ws_l2h")?,
            plain: load("qwen4_hc_inject_gemm_ws")?,
            inject_gatepre: load("qwen4_hc_inject_gemm_ws_l2h_gatepre")?,
            plain_gatepre: load("qwen4_hc_inject_gemm_ws_gatepre")?,
            inject_gatepre_2cta: load_2cta("qwen4_hc_inject_gemm_ws_l2h_gatepre_2cta"),
            plain_gatepre_2cta: load_2cta("qwen4_hc_inject_gemm_ws_gatepre_2cta"),
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: inject_gemm kernels unavailable: {e}"))
}

/// `postpre` (`fused`, cluster kernels) and `postsplit` (`rows`, then the
/// `inject` kernel), each `[bf16 slots, MXFP8 slots]`.
#[derive(Clone, Copy)]
struct PostKernels {
    fused: [KernelHandle; 2],
    fused_gatepre: [KernelHandle; 2],
    fused_mx_gatepre_2cta: KernelHandle,
    rows: [KernelHandle; 2],
    rows_gatepre: [KernelHandle; 2],
}

fn post_kernels(gpu: &dyn GpuBackend) -> Result<PostKernels> {
    static K: OnceLock<Result<PostKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |name: &str| {
            gpu.kernel("qwen4_hc_combine_gemm", name)
                .map_err(|e| format!("{name}: {e}"))
        };
        let fused = |name: &str| {
            let k = load(name)?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES)
                .map_err(|e| format!("{name}: {e}"))?;
            Ok::<_, String>(k)
        };
        let fused_2cta = {
            let name = "qwen4_hc_combine_moe_gemm_mx_gatepre_2cta";
            let k = load(name)?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_2CTA)
                .map_err(|e| format!("{name}: {e}"))?;
            k
        };
        Ok(PostKernels {
            fused: [
                fused("qwen4_hc_combine_moe_gemm")?,
                fused("qwen4_hc_combine_moe_gemm_mx")?,
            ],
            fused_gatepre: [
                fused("qwen4_hc_combine_moe_gemm_gatepre")?,
                fused("qwen4_hc_combine_moe_gemm_mx_gatepre")?,
            ],
            fused_mx_gatepre_2cta: fused_2cta,
            rows: [load("qwen4_hc_moe_rows")?, load("qwen4_hc_moe_rows_mx")?],
            rows_gatepre: [
                load("qwen4_hc_moe_rows_gatepre")?,
                load("qwen4_hc_moe_rows_mx_gatepre")?,
            ],
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: postpre kernels unavailable: {e}"))
}

/// Single-kernel fused 4-CTA cluster kernels (`Phase 1 inject/down + Phase 2 upmix`)
/// in `qwen4_hc_combine_gemm.cu`, indexed by `[no_amax, amax]`.
#[derive(Clone, Copy)]
struct ClusterUpmixKernels {
    plain: [KernelHandle; 2],
    inject: [KernelHandle; 2],
    combine_moe_mx: [KernelHandle; 2],
}

fn cluster_upmix_kernels(gpu: &dyn GpuBackend) -> Option<ClusterUpmixKernels> {
    static K: OnceLock<Option<ClusterUpmixKernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let load = |name: &str| -> Option<KernelHandle> {
            let k = gpu.kernel("qwen4_hc_combine_gemm", name).ok()?;
            gpu.set_kernel_max_dynamic_shared_memory(k, SMEM_BYTES_CLUSTER_FUSE)
                .ok()?;
            Some(k)
        };
        Some(ClusterUpmixKernels {
            plain: [
                load("qwen4_hc_inject_upmix_ws_gatepre")?,
                load("qwen4_hc_inject_upmix_ws_gatepre_amax")?,
            ],
            inject: [
                load("qwen4_hc_inject_upmix_ws_l2h_gatepre")?,
                load("qwen4_hc_inject_upmix_ws_l2h_gatepre_amax")?,
            ],
            combine_moe_mx: [
                load("qwen4_hc_combine_moe_upmix_mx_gatepre")?,
                load("qwen4_hc_combine_moe_upmix_mx_gatepre_amax")?,
            ],
        })
    })
}

/// Parameters for one single-kernel fused 4-CTA cluster mHC preparation
/// (`hc_cluster_fuse_v2`: pre-dequantized BF16 `folded` + `up`).
#[derive(Clone, Copy)]
pub(super) struct ClusterFuseLaunch {
    pub hyper: DevicePtr,
    pub residual: DevicePtr,
    pub folded: DevicePtr,
    pub up: DevicePtr,
    pub mixed: DevicePtr,
    pub amax_slot: Option<DevicePtr>,
    pub num_tokens: usize,
    pub eps: f32,
}

/// Validated launch plan for `hc_cluster_fuse_v2`, ensuring `qwen4_hc_prep_weights`
/// only runs when the fused cluster kernel is admitted and loaded.
#[derive(Clone, Copy)]
pub(super) struct ClusterFusePlan {
    mode: ClusterFuseMode,
    kernels: ClusterUpmixKernels,
}

/// Static NVFP4 weight descriptors validated before `qwen4_hc_prep_weights`
/// materializes BF16 `folded` and `up` for the fused cluster kernels.
#[derive(Clone, Copy)]
struct ClusterFuseWeights {
    down_packed: DevicePtr,
    down_scales: DevicePtr,
    scale2_down: f32,
    inject_packed: DevicePtr,
    inject_scales: DevicePtr,
    scale2_inject: f32,
    up_packed: DevicePtr,
    up_scales: DevicePtr,
    alpha_up: f32,
    norm_w: DevicePtr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClusterFuseMode {
    Plain,
    Inject,
    PostPreMx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostPreVariant {
    OneCta { mx: bool, gatepre: bool },
    TwoCtaMxGatepre,
}

fn select_postpre_variant(mx: bool, gatepre: bool, use_2cta: bool) -> PostPreVariant {
    if gatepre && mx && use_2cta {
        PostPreVariant::TwoCtaMxGatepre
    } else {
        PostPreVariant::OneCta { mx, gatepre }
    }
}

fn cluster_fuse_grid(rows: u32) -> [u32; 3] {
    [STREAMS as u32, div_ceil(rows, TILE_ROWS_CLUSTER_FUSE), 1]
}

/// Pure pointer, geometry, and aliasing admission for the single-kernel fused
/// 4-CTA cluster path (`Plain`, `Inject`, or `PostPreMx` with `gatepre`).
fn cluster_fuse_admitted_mode(
    geometry: (usize, usize, usize),
    weights: &ClusterFuseWeights,
    launch: &ClusterFuseLaunch,
    pending: Option<&PendingInject>,
    postpre_selected: bool,
) -> Option<ClusterFuseMode> {
    if geometry != (HIDDEN, STREAMS, RANK)
        || launch.num_tokens == 0
        || u32::try_from(launch.num_tokens).is_err()
    {
        return None;
    }
    let aligned16 = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(16);
    let aligned8 = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(8);
    let aligned4 = |p: DevicePtr| !p.is_null() && p.0.is_multiple_of(4);
    if !aligned16(launch.hyper)
        || !aligned4(launch.residual)
        || !aligned16(launch.folded)
        || !aligned16(launch.up)
        || !aligned16(launch.mixed)
        || !aligned16(weights.norm_w)
        || !aligned8(weights.down_packed)
        || weights.down_scales.is_null()
        || !weights.scale2_down.is_finite()
        || !aligned8(weights.inject_packed)
        || weights.inject_scales.is_null()
        || !weights.scale2_inject.is_finite()
        || !aligned8(weights.up_packed)
        || weights.up_scales.is_null()
        || !weights.alpha_up.is_finite()
        || !launch.amax_slot.is_none_or(aligned4)
    {
        return None;
    }
    let row_bytes = HIDDEN * STREAMS * 2;
    let hyper_bytes = launch.num_tokens * row_bytes;
    let mixed_bytes = launch.num_tokens * HIDDEN * 2;
    let folded_bytes = STREAMS * (RANK + STREAMS) * HIDDEN * 2;
    let up_bytes = STREAMS * HIDDEN * RANK * 2;
    let prep_buffers = [(launch.folded, folded_bytes), (launch.up, up_bytes)];
    let io_writes = [
        (launch.hyper, hyper_bytes),
        (launch.residual, hyper_bytes),
        (launch.mixed, mixed_bytes),
    ];
    if !disjoint(launch.hyper, hyper_bytes, launch.mixed, mixed_bytes)
        || !disjoint(launch.residual, hyper_bytes, launch.mixed, mixed_bytes)
        || !disjoint(launch.folded, folded_bytes, launch.up, up_bytes)
        || !prep_buffers
            .iter()
            .all(|&(p, pb)| io_writes.iter().all(|&(w, wb)| disjoint(p, pb, w, wb)))
    {
        return None;
    }
    let Some(p) = pending else {
        return Some(ClusterFuseMode::Plain);
    };
    if p.hyper != launch.hyper || p.residual != launch.residual || p.rows != launch.num_tokens {
        return None;
    }
    let Some(c) = p.combine else {
        if !aligned16(p.core)
            || !disjoint(p.core, mixed_bytes, launch.hyper, hyper_bytes)
            || !disjoint(p.core, mixed_bytes, launch.residual, hyper_bytes)
            || !prep_buffers
                .iter()
                .all(|&(pb_ptr, pb_bytes)| disjoint(p.core, mixed_bytes, pb_ptr, pb_bytes))
            || (p.core != launch.mixed && !disjoint(p.core, mixed_bytes, launch.mixed, mixed_bytes))
        {
            return None;
        }
        return Some(ClusterFuseMode::Inject);
    };
    if !postpre_selected
        || c.slot_scales.is_null()
        || !c.slot_corrections.is_null()
        || c.gate_scalars.is_null()
        || !post_admitted(&c)
    {
        return None;
    }
    let k = TOPK as usize;
    let strict_reads = [
        (c.expert_out, launch.num_tokens * k * HIDDEN),
        (c.slot_scales, launch.num_tokens * k * HIDDEN / 32),
        (c.topk_weights, launch.num_tokens * k * 4),
        (c.gate_scalars, launch.num_tokens * 4),
    ];
    let strict_ok = strict_reads.iter().all(|&(r, r_bytes)| {
        io_writes
            .iter()
            .chain(prep_buffers.iter())
            .all(|&(w, w_bytes)| disjoint(r, r_bytes, w, w_bytes))
    });
    let shared_ok = disjoint(c.shared_out, mixed_bytes, launch.hyper, hyper_bytes)
        && disjoint(c.shared_out, mixed_bytes, launch.residual, hyper_bytes)
        && prep_buffers
            .iter()
            .all(|&(pb_ptr, pb_bytes)| disjoint(c.shared_out, mixed_bytes, pb_ptr, pb_bytes))
        && (c.shared_out == launch.mixed
            || disjoint(c.shared_out, mixed_bytes, launch.mixed, mixed_bytes));
    if strict_ok && shared_ok {
        Some(ClusterFuseMode::PostPreMx)
    } else {
        None
    }
}

/// Rows each `qwen4_hc_moe_rows` block forms (`kMoeRows`).
const MOE_ROWS: u32 = 16;
/// The kernels' fixed routing width (`kTopk`).
const TOPK: u32 = 10;

type GrowOnly = Mutex<Option<(usize, DevicePtr)>>;

/// `[rows, H]` bf16: the formed MoE output rows.
static FORMED_ROWS: GrowOnly = Mutex::new(None);
/// `[rows, top-k]` i32: postpre's copy of the pending combine's `token_to_perm`.
static ROUTING_COPY: GrowOnly = Mutex::new(None);

/// A grow-only device buffer, used only on the single prefill stream.
fn grow_only(cell: &GrowOnly, gpu: &dyn GpuBackend, bytes: usize, what: &str) -> Result<DevicePtr> {
    let mut guard = cell
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: {what} scratch poisoned"))?;
    if let Some((have, ptr)) = *guard
        && have >= bytes
    {
        return Ok(ptr);
    }
    // An earlier block is intentionally leaked (grow-only, in-flight safety).
    let ptr = gpu
        .alloc(bytes)
        .with_context(|| format!("{SELECTOR}: alloc {bytes} B for {what}"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, bytes, what)?;
    *guard = Some((bytes, ptr));
    Ok(ptr)
}

/// postpre forms the MoE rows and projects them in one kernel: a tile's
/// prologue can run after another tile's epilogue, so nothing it reads may
/// be one of the kernel's outputs. MoeLayer::forward_prefill keeps
/// `token_to_perm` inside the arena's `gate_logits`, which is also `down`
/// here (tiles finishing early overwrote the map later tiles gathered from:
/// CUDA 700 in-engine), so postpre reads a copy taken on the prefill stream
/// before the launch. Every other input must be disjoint from the outputs
/// (`writes`); otherwise the AfterCombine path runs (receipt `mode=AfterCombine`).
fn postpre_reads_disjoint(
    c: &MoeCombine,
    rows: usize,
    slot_rows: usize,
    writes: &[(DevicePtr, usize)],
) -> bool {
    let k = TOPK as usize;
    let gate_read = if !c.gate_scalars.is_null() {
        (c.gate_scalars, rows * 4)
    } else {
        (c.gate_input, rows * HIDDEN * 2)
    };
    let expert_elem_bytes = if !c.slot_scales.is_null() { 1 } else { 2 };
    let mut reads = vec![
        (c.expert_out, slot_rows * k * HIDDEN * expert_elem_bytes),
        (c.topk_weights, rows * k * 4),
        (c.shared_out, rows * HIDDEN * 2),
        gate_read,
    ];
    if !c.slot_scales.is_null() {
        reads.push((c.slot_scales, slot_rows * k * HIDDEN / 32));
    }
    reads.iter().all(|&(r, r_bytes)| {
        writes
            .iter()
            .all(|&(w, w_bytes)| disjoint(r, r_bytes, w, w_bytes))
    })
}

/// Pointer shapes the row formation reads with 16-byte (slot and shared
/// rows) and 4-byte (routing) vectors; the kernels trap on another top-k.
fn post_admitted(c: &MoeCombine) -> bool {
    let gate_ok = if !c.gate_scalars.is_null() {
        c.gate_scalars.0.is_multiple_of(4)
    } else {
        !c.gate_input.is_null()
            && c.gate_input.0.is_multiple_of(2)
            && c.shared_gate.0.is_multiple_of(2)
    };
    c.topk == TOPK
        && c.slot_corrections.is_null()
        && (c.slot_scales.is_null() || c.slot_scales.0.is_multiple_of(4))
        && [c.expert_out, c.shared_out]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(16))
        && [c.token_to_perm, c.topk_weights]
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(4))
        && gate_ok
}

/// Which pending injection a launch consumed (receipt only).
#[derive(Clone, Copy, Debug)]
enum Mode {
    Inject = 0,
    Plain = 1,
    AfterCombine = 2,
    PostPre = 3,
    PostPreMx = 4,
    PostSplit = 5,
    PostSplitMx = 6,
}

impl Qwen4HyperConnection {
    /// Whether `inject_gemm_prefill[_strip]` is admitted for these pointers and
    /// row count.
    pub(super) fn inject_gemm_admitted(
        &self,
        pending: Option<&PendingInject>,
        hyper: DevicePtr,
        residual: DevicePtr,
        outputs: [DevicePtr; 3],
        num_tokens: usize,
    ) -> Result<bool> {
        let [folded, rstd, down] = outputs;
        Ok(selection()?.injgemm
            && (self.hidden_size, self.hc_count, self.rank) == (HIDDEN, STREAMS, RANK)
            && pending.is_none_or(|p| {
                p.hyper == hyper
                    && p.residual == residual
                    && p.rows == num_tokens
                    && (p.combine.is_some() || p.core.is_null() || p.core.0.is_multiple_of(16))
            })
            && [hyper, folded]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(16))
            && [residual, rstd, down]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(4))
            && num_tokens > 0
            && u32::try_from(num_tokens).is_ok()
            && (super::prefill_fused::level() != 0
                || gemm_order_verified(reference_rows(num_tokens))?))
    }

    /// Apply `pending`, then write rstd, `down` and the residual tail's inject
    /// scales exactly as HC_FUSED's rstd pass (or `qwen4_hc_[combine_]inject_rstd`),
    /// four GEMMs and `qwen4_hc_combine` would. `outputs` is `[folded, rstd,
    /// down]`, `scale2` is `[down, inject]`. Returns false, launching nothing,
    /// when the lever is off or the rows are not admitted.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inject_gemm_prefill(
        &self,
        pending: Option<&PendingInject>,
        hyper: DevicePtr,
        residual: DevicePtr,
        outputs: [DevicePtr; 3],
        scale2: [f32; 2],
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<bool> {
        self.inject_gemm_prefill_strip(
            pending, hyper, residual, outputs, scale2, num_tokens, num_tokens, gpu, eps, stream,
        )
    }

    /// Strip-aware entry of [`Self::inject_gemm_prefill`]: `num_tokens` is the
    /// strip row count while `slot_rows` is the full batch row count backing
    /// global-slot-indexed MoE combine buffers (`expert_out`, `slot_scales`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inject_gemm_prefill_strip(
        &self,
        pending: Option<&PendingInject>,
        hyper: DevicePtr,
        residual: DevicePtr,
        outputs: [DevicePtr; 3],
        scale2: [f32; 2],
        num_tokens: usize,
        slot_rows: usize,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<bool> {
        let [folded, rstd, down] = outputs;
        if !self.inject_gemm_admitted(pending, hyper, residual, outputs, num_tokens)? {
            return Ok(false);
        }
        let k = kernels(gpu)?;
        let sel = selection()?;
        let use_2cta = sel.gatepre && inject_2cta_enabled()?;
        let rows = num_tokens as u32;
        let grid = [STREAMS as u32, div_ceil(rows, TILE_ROWS), 1];
        let grid_2cta = [STREAMS as u32, div_ceil(rows, TILE_ROWS_2CTA), 1];
        let launch_inject = |core: DevicePtr| {
            let (kh, launch_grid, launch_threads, launch_smem) =
                match (sel.gatepre, use_2cta, core.is_null()) {
                    (true, true, true) => match k.plain_gatepre_2cta {
                        Some(h) => (h, grid_2cta, THREADS_2CTA, SMEM_BYTES_2CTA),
                        None => (k.plain_gatepre, grid, THREADS, SMEM_BYTES),
                    },
                    (true, true, false) => match k.inject_gatepre_2cta {
                        Some(h) => (h, grid_2cta, THREADS_2CTA, SMEM_BYTES_2CTA),
                        None => (k.inject_gatepre, grid, THREADS, SMEM_BYTES),
                    },
                    (true, false, true) => (k.plain_gatepre, grid, THREADS, SMEM_BYTES),
                    (true, false, false) => (k.inject_gatepre, grid, THREADS, SMEM_BYTES),
                    (false, _, true) => (k.plain, grid, THREADS, SMEM_BYTES),
                    (false, _, false) => (k.inject, grid, THREADS, SMEM_BYTES),
                };
            KernelLaunch::new(gpu, kh)
                .grid(launch_grid)
                .block([launch_threads, 1, 1])
                .shared_mem(launch_smem)
                .arg_ptr(hyper)
                .arg_ptr(core)
                .arg_ptr(residual)
                .arg_ptr(folded)
                .arg_ptr(rstd)
                .arg_ptr(down)
                .arg_u32(rows)
                .arg_u32(self.rank as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(self.hc_count as u32)
                .arg_f32(scale2[0])
                .arg_f32(scale2[1])
                .arg_f32(eps)
                .launch(stream)
        };
        let row_bytes = self.hidden_size * self.hc_count * 2;
        let outputs = [
            (down, num_tokens * self.rank * 2),
            (hyper, num_tokens * row_bytes),
            (residual, num_tokens * row_bytes),
            (rstd, num_tokens * self.hc_count * 4),
        ];
        let post_selected = |c: &MoeCombine| {
            post_admitted(c)
                && (sel.postsplit
                    || (sel.postpre && postpre_reads_disjoint(c, num_tokens, slot_rows, &outputs)))
        };
        let mode = match pending.map(|p| (p, p.combine)) {
            Some((_, Some(c))) if post_selected(&c) => {
                let pk = post_kernels(gpu)?;
                let mx = !c.slot_scales.is_null();
                let gatepre = !c.gate_scalars.is_null();
                let cap_rows = num_tokens.max(crate::layers::qwen4_prefill_moe::max_chunk_rows());
                let formed = grow_only(
                    &FORMED_ROWS,
                    gpu,
                    cap_rows * self.hidden_size * 2,
                    "the formed MoE rows",
                )?;
                if sel.postpre {
                    let routing_bytes = num_tokens * TOPK as usize * 4;
                    let routing = if gatepre
                        && outputs.iter().all(|&(w, w_bytes)| {
                            disjoint(c.token_to_perm, routing_bytes, w, w_bytes)
                        }) {
                        c.token_to_perm
                    } else {
                        let routing = grow_only(
                            &ROUTING_COPY,
                            gpu,
                            cap_rows * TOPK as usize * 4,
                            "the postpre routing copy",
                        )?;
                        gpu.copy_d2d_async(c.token_to_perm, routing, routing_bytes, stream)?;
                        routing
                    };
                    let (handle, post_grid, post_threads, post_smem) =
                        match select_postpre_variant(mx, gatepre, use_2cta) {
                            PostPreVariant::TwoCtaMxGatepre => (
                                pk.fused_mx_gatepre_2cta,
                                grid_2cta,
                                THREADS_2CTA,
                                SMEM_BYTES_2CTA,
                            ),
                            PostPreVariant::OneCta { mx, gatepre } => {
                                let h = if gatepre {
                                    pk.fused_gatepre[mx as usize]
                                } else {
                                    pk.fused[mx as usize]
                                };
                                (h, grid, THREADS, SMEM_BYTES)
                            }
                        };
                    let launch = KernelLaunch::new(gpu, handle)
                        .grid(post_grid)
                        .block([post_threads, 1, 1])
                        .shared_mem(post_smem)
                        .arg_ptr(hyper)
                        .arg_ptr(residual)
                        .arg_ptr(folded)
                        .arg_ptr(rstd)
                        .arg_ptr(down)
                        .arg_ptr(formed)
                        .arg_ptr(c.expert_out)
                        .arg_ptr(c.slot_scales)
                        .arg_ptr(routing)
                        .arg_ptr(c.topk_weights)
                        .arg_ptr(c.shared_out);
                    let launch = if gatepre {
                        launch.arg_ptr(c.gate_scalars)
                    } else {
                        launch.arg_ptr(c.gate_input).arg_ptr(c.shared_gate)
                    };
                    launch
                        .arg_u32(rows)
                        .arg_u32(self.rank as u32)
                        .arg_u32(self.hidden_size as u32)
                        .arg_u32(self.hc_count as u32)
                        .arg_u32(c.topk)
                        .arg_f32(scale2[0])
                        .arg_f32(scale2[1])
                        .arg_f32(eps)
                        .launch(stream)?;
                    if mx { Mode::PostPreMx } else { Mode::PostPre }
                } else {
                    let handle = if gatepre {
                        pk.rows_gatepre[mx as usize]
                    } else {
                        pk.rows[mx as usize]
                    };
                    let launch = KernelLaunch::new(gpu, handle)
                        .grid([div_ceil(rows, MOE_ROWS), 1, 1])
                        .block([THREADS, 1, 1])
                        .arg_ptr(formed)
                        .arg_ptr(c.expert_out)
                        .arg_ptr(c.slot_scales)
                        .arg_ptr(c.token_to_perm)
                        .arg_ptr(c.topk_weights)
                        .arg_ptr(c.shared_out);
                    let launch = if gatepre {
                        launch.arg_ptr(c.gate_scalars)
                    } else {
                        launch.arg_ptr(c.gate_input).arg_ptr(c.shared_gate)
                    };
                    launch
                        .arg_u32(rows)
                        .arg_u32(self.hidden_size as u32)
                        .arg_u32(c.topk)
                        .launch(stream)?;
                    launch_inject(formed)?;
                    if mx {
                        Mode::PostSplitMx
                    } else {
                        Mode::PostSplit
                    }
                }
            }
            Some((p, Some(_))) => {
                // Formed and injected by its own fused pass
                // (qwen4_hc_combine_inject_rstd); the kernel then re-reads the
                // injected rows. Forming it inside the projection kernel
                // (qwen4_hc_combine_inject_gemm) was not faster (micro4).
                self.inject_rstd_prefill(p, rstd, gpu, eps, stream)?;
                launch_inject(DevicePtr::NULL)?;
                Mode::AfterCombine
            }
            Some((p, None)) => {
                launch_inject(p.core)?;
                Mode::Inject
            }
            None => {
                launch_inject(DevicePtr::NULL)?;
                Mode::Plain
            }
        };
        static LOGGED: [std::sync::Once; 7] = [const { std::sync::Once::new() }; 7];
        LOGGED[mode as usize].call_once(|| {
            tracing::info!(
                "MHC_FUSED_INJECT_GEMM_ENGAGED selector={SELECTOR} mode={mode:?} inject_2cta={use_2cta} rows={num_tokens}"
            );
        });
        Ok(true)
    }

    /// Validate admission and load kernel handles for the single-kernel fused
    /// 4-CTA cluster mHC v2 path (`BM=32`, pre-dequantized BF16 `folded` + `up`,
    /// 0 Phase-2 `cluster.sync()`s). Returns `Ok(None)` without launching when
    /// the mode, pointers, or kernel symbols are not admitted, so the caller
    /// skips `qwen4_hc_prep_weights(..., up_out=s.up)` and falls back cleanly.
    pub(super) fn plan_cluster_fuse(
        &self,
        pending: Option<&PendingInject>,
        launch: &ClusterFuseLaunch,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<ClusterFusePlan>> {
        let Some(inject) = self.inject.as_ref() else {
            return Ok(None);
        };
        let sel = selection()?;
        if !(sel.injgemm && sel.gatepre && sel.upmix) {
            return Ok(None);
        }
        if super::prefill_fused::level() == 0
            && !gemm_order_verified(reference_rows(launch.num_tokens))?
        {
            return Ok(None);
        }
        let weights = ClusterFuseWeights {
            down_packed: self.down.weight,
            down_scales: self.down.weight_scale,
            scale2_down: self.down.weight_scale_2,
            inject_packed: inject.weight,
            inject_scales: inject.weight_scale,
            scale2_inject: inject.weight_scale_2,
            up_packed: self.up.weight,
            up_scales: self.up.weight_scale,
            alpha_up: self.up.weight_scale_2,
            norm_w: self.norm.weight,
        };
        let Some(mode) = cluster_fuse_admitted_mode(
            (self.hidden_size, self.hc_count, self.rank),
            &weights,
            launch,
            pending,
            sel.postpre,
        ) else {
            return Ok(None);
        };
        let Some(kernels) = cluster_upmix_kernels(gpu) else {
            return Ok(None);
        };
        Ok(Some(ClusterFusePlan { mode, kernels }))
    }

    /// Launch a validated [`ClusterFusePlan`] after `qwen4_hc_prep_weights`
    /// has populated `launch.folded` and `launch.up`.
    pub(super) fn launch_cluster_fuse(
        &self,
        plan: ClusterFusePlan,
        pending: Option<&PendingInject>,
        launch: &ClusterFuseLaunch,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let inject = self
            .inject
            .as_ref()
            .expect("ClusterFusePlan validated inject weight");
        let mode = plan.mode;
        let ck = plan.kernels;
        if let Some(slot) = launch.amax_slot {
            gpu.memset_async(slot, 0, 4, stream)?;
        }
        let rows = launch.num_tokens as u32;
        let grid = cluster_fuse_grid(rows);
        let amax_idx = usize::from(launch.amax_slot.is_some());
        match mode {
            ClusterFuseMode::Plain | ClusterFuseMode::Inject => {
                let (handle, core) = match mode {
                    ClusterFuseMode::Plain => (ck.plain[amax_idx], DevicePtr::NULL),
                    ClusterFuseMode::Inject => (
                        ck.inject[amax_idx],
                        pending.expect("Inject mode requires pending").core,
                    ),
                    ClusterFuseMode::PostPreMx => unreachable!("handled in PostPreMx arm"),
                };
                let kl = KernelLaunch::new(gpu, handle)
                    .grid(grid)
                    .block([THREADS, 1, 1])
                    .shared_mem(SMEM_BYTES_CLUSTER_FUSE)
                    .arg_ptr(launch.hyper)
                    .arg_ptr(core)
                    .arg_ptr(launch.residual)
                    .arg_ptr(launch.folded)
                    .arg_ptr(launch.up)
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(launch.mixed)
                    .arg_u32(rows)
                    .arg_u32(self.rank as u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_f32(self.down.weight_scale_2)
                    .arg_f32(inject.weight_scale_2)
                    .arg_f32(self.up.weight_scale_2)
                    .arg_f32(launch.eps);
                match launch.amax_slot {
                    Some(slot) => kl.arg_ptr(slot).launch(stream)?,
                    None => kl.launch(stream)?,
                }
            }
            ClusterFuseMode::PostPreMx => {
                let c = pending
                    .and_then(|p| p.combine)
                    .expect("PostPreMx mode requires pending MoE combine");
                let cap_rows = launch
                    .num_tokens
                    .max(crate::layers::qwen4_prefill_moe::max_chunk_rows());
                let formed_bytes = cap_rows * self.hidden_size * 2;
                let formed = grow_only(&FORMED_ROWS, gpu, formed_bytes, "the formed MoE rows")?;
                let row_bytes = self.hidden_size * self.hc_count * 2;
                let hyper_bytes = launch.num_tokens * row_bytes;
                let mixed_bytes = launch.num_tokens * self.hidden_size * 2;
                let routing_bytes = launch.num_tokens * TOPK as usize * 4;
                let writes = [
                    (launch.hyper, hyper_bytes),
                    (launch.residual, hyper_bytes),
                    (launch.mixed, mixed_bytes),
                    (formed, formed_bytes),
                ];
                let routing = if writes
                    .iter()
                    .all(|&(w, w_bytes)| disjoint(c.token_to_perm, routing_bytes, w, w_bytes))
                {
                    c.token_to_perm
                } else {
                    let routing = grow_only(
                        &ROUTING_COPY,
                        gpu,
                        cap_rows * TOPK as usize * 4,
                        "the postpre routing copy",
                    )?;
                    gpu.copy_d2d_async(c.token_to_perm, routing, routing_bytes, stream)?;
                    routing
                };
                let kl = KernelLaunch::new(gpu, ck.combine_moe_mx[amax_idx])
                    .grid(grid)
                    .block([THREADS, 1, 1])
                    .shared_mem(SMEM_BYTES_CLUSTER_FUSE)
                    .arg_ptr(launch.hyper)
                    .arg_ptr(launch.residual)
                    .arg_ptr(formed)
                    .arg_ptr(c.expert_out)
                    .arg_ptr(c.slot_scales)
                    .arg_ptr(routing)
                    .arg_ptr(c.topk_weights)
                    .arg_ptr(c.shared_out)
                    .arg_ptr(c.gate_scalars)
                    .arg_ptr(launch.folded)
                    .arg_ptr(launch.up)
                    .arg_ptr(self.norm.weight)
                    .arg_ptr(launch.mixed)
                    .arg_u32(rows)
                    .arg_u32(self.rank as u32)
                    .arg_u32(self.hidden_size as u32)
                    .arg_u32(self.hc_count as u32)
                    .arg_u32(c.topk)
                    .arg_f32(self.down.weight_scale_2)
                    .arg_f32(inject.weight_scale_2)
                    .arg_f32(self.up.weight_scale_2)
                    .arg_f32(launch.eps);
                match launch.amax_slot {
                    Some(slot) => kl.arg_ptr(slot).launch(stream)?,
                    None => kl.launch(stream)?,
                }
            }
        }
        static LOGGED_CLUSTER_FUSE: std::sync::Once = std::sync::Once::new();
        LOGGED_CLUSTER_FUSE.call_once(|| {
            tracing::info!(
                "MHC_CLUSTER_FUSE_ENGAGED selector={} mode={mode:?} amax={} rows={} tile_rows={TILE_ROWS_CLUSTER_FUSE} smem={SMEM_BYTES_CLUSTER_FUSE}",
                super::prefill_fused::CLUSTER_FUSE_SELECTOR,
                launch.amax_slot.is_some(),
                launch.num_tokens,
            );
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> ClusterFuseWeights {
        ClusterFuseWeights {
            down_packed: DevicePtr(0x10_0000_0000),
            down_scales: DevicePtr(0x11_0000_0000),
            scale2_down: 1.0,
            inject_packed: DevicePtr(0x12_0000_0000),
            inject_scales: DevicePtr(0x13_0000_0000),
            scale2_inject: 1.0,
            up_packed: DevicePtr(0x14_0000_0000),
            up_scales: DevicePtr(0x15_0000_0000),
            alpha_up: 1.0,
            norm_w: DevicePtr(0x16_0000_0000),
        }
    }

    fn sample_launch(rows: usize) -> ClusterFuseLaunch {
        ClusterFuseLaunch {
            hyper: DevicePtr(0x20_0000_0000),
            residual: DevicePtr(0x30_0000_0000),
            folded: DevicePtr(0x34_0000_0000),
            up: DevicePtr(0x38_0000_0000),
            mixed: DevicePtr(0x40_0000_0000),
            amax_slot: Some(DevicePtr(0x50_0000_0000)),
            num_tokens: rows,
            eps: 1e-6,
        }
    }

    fn sample_moe_pending(launch: &ClusterFuseLaunch) -> PendingInject {
        PendingInject {
            hyper: launch.hyper,
            core: launch.mixed,
            residual: launch.residual,
            rows: launch.num_tokens,
            combine: Some(MoeCombine {
                slot_scales: DevicePtr(0x60_0000_0000),
                slot_corrections: DevicePtr::NULL,
                expert_out: DevicePtr(0x70_0000_0000),
                token_to_perm: DevicePtr(0x80_0000_0000),
                topk_weights: DevicePtr(0x90_0000_0000),
                shared_out: DevicePtr(0xa0_0000_0000),
                gate_input: launch.mixed,
                shared_gate: DevicePtr(0xb0_0000_0000),
                gate_scalars: DevicePtr(0xc0_0000_0000),
                topk: TOPK,
            }),
        }
    }

    #[test]
    fn parse_inject_2cta_defaults_off_and_validates() {
        assert_eq!(parse_inject_2cta(None), Ok(false));
        for on in ["1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_inject_2cta(Some(on)), Ok(true), "{on:?}");
        }
        for off in ["", "0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(parse_inject_2cta(Some(off)), Ok(false), "{off:?}");
        }
        for bad in ["2", "yes", "no", "2cta"] {
            assert!(parse_inject_2cta(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn postpre_mx_gates_2cta_behind_use_2cta() {
        assert_eq!(
            select_postpre_variant(true, true, false),
            PostPreVariant::OneCta {
                mx: true,
                gatepre: true,
            }
        );
        assert_eq!(
            select_postpre_variant(true, true, true),
            PostPreVariant::TwoCtaMxGatepre
        );
        assert_eq!(
            select_postpre_variant(false, true, true),
            PostPreVariant::OneCta {
                mx: false,
                gatepre: true,
            }
        );
        assert_eq!(
            select_postpre_variant(true, false, true),
            PostPreVariant::OneCta {
                mx: true,
                gatepre: false,
            }
        );
    }

    #[test]
    fn cluster_fuse_admits_plain_inject_and_postpre_mx_with_aliasing_guards() {
        let w = sample_weights();
        let l = sample_launch(1024);
        let geom = (HIDDEN, STREAMS, RANK);

        // Plain mode
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, None, true),
            Some(ClusterFuseMode::Plain)
        );

        // Inject mode (disjoint core and row-aligned core == mixed both admitted)
        let mut p_inj = sample_moe_pending(&l);
        p_inj.combine = None;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_inj), true),
            Some(ClusterFuseMode::Inject)
        );
        p_inj.core = DevicePtr(0xd0_0000_0000);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_inj), true),
            Some(ClusterFuseMode::Inject)
        );
        // Misaligned core overlap with mixed must be rejected.
        p_inj.core = l.mixed.offset(16);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_inj), true),
            None
        );

        // PostPreMx mode
        let p_moe = sample_moe_pending(&l);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_moe), true),
            Some(ClusterFuseMode::PostPreMx)
        );
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_moe), false),
            None
        );
        // Row-aligned shared_out == mixed is safe; overlapping expert_out is rejected.
        let mut p_sh = p_moe;
        p_sh.combine.as_mut().expect("combine").shared_out = l.mixed;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_sh), true),
            Some(ClusterFuseMode::PostPreMx)
        );
        let mut p_bad_exp = p_moe;
        p_bad_exp.combine.as_mut().expect("combine").expert_out = l.mixed;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_bad_exp), true),
            None
        );
        // slot_corrections (slot_r2), BF16 slots, and missing gate_scalars fall back.
        let mut p_r2 = p_moe;
        p_r2.combine.as_mut().expect("combine").slot_corrections = DevicePtr(0xe0_0000_0000);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_r2), true),
            None
        );
        let mut p_bf16 = p_moe;
        p_bf16.combine.as_mut().expect("combine").slot_scales = DevicePtr::NULL;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_bf16), true),
            None
        );
        let mut p_nogate = p_moe;
        p_nogate.combine.as_mut().expect("combine").gate_scalars = DevicePtr::NULL;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &l, Some(&p_nogate), true),
            None
        );
        // Non-finite weight scale or unaligned norm_w must be rejected.
        let mut bad_w = w;
        bad_w.alpha_up = f32::NAN;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &bad_w, &l, None, true),
            None
        );
        let mut bad_nw = w;
        bad_nw.norm_w = w.norm_w.offset(8);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &bad_nw, &l, None, true),
            None
        );
        // Overlapping or misaligned pre-dequantized folded/up buffers must be rejected.
        let mut overlap_up = l;
        overlap_up.up = l.folded;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &overlap_up, None, true),
            None
        );
        let mut bad_folded = l;
        bad_folded.folded = l.folded.offset(8);
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &bad_folded, None, true),
            None
        );
        let mut overlap_mixed = l;
        overlap_mixed.up = l.mixed;
        assert_eq!(
            cluster_fuse_admitted_mode(geom, &w, &overlap_mixed, None, true),
            None
        );
    }

    #[test]
    fn post_admitted_rejects_slot_corrections() {
        let l = sample_launch(1024);
        let mut c = sample_moe_pending(&l).combine.expect("combine");
        assert!(post_admitted(&c));
        c.slot_corrections = DevicePtr(0xe0_0000_0000);
        assert!(!post_admitted(&c));
    }

    #[test]
    fn cluster_fuse_grid_uses_bm32_tiles() {
        assert_eq!(TILE_ROWS_CLUSTER_FUSE, 32);
        assert_eq!(cluster_fuse_grid(512), [4, 16, 1]);
        assert_eq!(cluster_fuse_grid(513), [4, 17, 1]);
        assert_eq!(cluster_fuse_grid(14_885), [4, 466, 1]);
        assert_eq!(cluster_fuse_grid(16_384), [4, 512, 1]);
    }
}
