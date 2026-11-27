// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off: the Flash-Next shared expert through the V7 routed-expert kernels
//! (`ATLAS_QWEN4_PREFILL_SHARED_V7=1`, needs the compact V2 path).
//!
//! The shared expert runs as "expert 0" of a one-expert work list: every token is one
//! row, the gather table is the identity, and the host builds the V7 workspace
//! (128-row items) directly, so the routed planner, sort and contract are untouched.
//! Gate/up/SwiGLU run in the fused V7 kernel and down in the V7 GEMM, from the
//! transposed NVFP4 shared weights with exact operands (BF16 activations, scale2 in
//! the epilogue). The default path instead casts the activations to E4M3 without a
//! scale and rounds the weights to E4M3 (`fp8_gemm_t` over `predequant_nvfp4_to_fp8`),
//! so this changes numerics toward the BF16-activation reference. Output buffers are
//! the default path's (`ssm_deinterleaved` activated, `attn_output` down), so the
//! existing shared-gate blend consumes it unchanged.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{MoeLayer, qwen4_compact_contract as abi};
use crate::layer::ForwardContext;
use crate::weight_map::QuantizedWeight;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SHARED_V7";
/// Default-off: the same one-expert work list through the routed W4A8 kernels (V10, or V12
/// with ATLAS_QWEN4_PREFILL_MOE_W4A8_K64=1): b12x w4a8 numerics, MXFP8 activations.
pub(crate) const W4A8_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SHARED_W4A8";
/// Pre-transcode the shared expert's NVFP4 weights once per layer into a 4.83 MiB
/// L2-resident scratch buffer (`qwen4_shared_pretranscode_v28`) instead of re-transcoding inside
/// every 128-row M-tile. Bit-identical to `ATLAS_QWEN4_PREFILL_SHARED_W4A8=1` on V28.
///
/// Default OFF on GB10 (`active_sm_count() <= 64`) and auto-`Auto` on wide-SM GPUs
/// (`active_sm_count() > 64`; set `=0`/`off`/`false` to opt out). In `Auto` mode (`=1`),
/// engages only when `rows > 256` (`items >= 3`) where prefill reuses the pretranscoded weights
/// across multiple tiles; `rows <= 256` takes the direct W4A8 path to avoid writing
/// 5.07 MiB/layer to DRAM for 1-2 tiles. Set `=force` to override the row threshold.
pub(crate) const PRETRANSCODE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SHARED_W4A8_PRETRANSCODE";
/// Shared-expert side-stream overlap selector on prefill: default OFF on GB10
/// (`active_sm_count() <= 64`, where launching side-stream grids onto `prefill_stream` starves
/// concurrent prefill kernels and adds 48 cross-stream barriers; set `=1` to opt in), and
/// auto-ON on wide-SM GPUs (`active_sm_count() > 64`; set `=0`/`off`/`false` to opt out).
/// Decode overlap in `forward_k5.rs` remains independently enabled.
const PREFILL_SHARED_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_MOE_SHARED_OVERLAP";
/// Wave-17 full-overlap selector: defer joining `self.event_b` until AFTER
/// `run_routed_grouped_gemm` completes whenever `disjoint_overlap_safe` holds. Default OFF on
/// GB10 (`active_sm_count() <= 64`; set `=1` to opt in) and auto-ON on wide-SM GPUs
/// (`active_sm_count() > 64`; set `=0`/`off`/`false` to opt out and revert to pre-routed
/// joining in `validate_context`).
const PREFILL_SHARED_FULL_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SHARED_FULL_OVERLAP";
/// Post-router sort-overlap toggle: launch the post-router pre-transcoded shared-expert W4A8
/// path (`gateup_2stage` <= 48 KiB SMEM + 4-stage `gemm_discard` with L2 discard) on
/// `prefill_stream` right after the router GEMM so it overlaps `topk` and 1-CTA `sort_plan`,
/// joining via `SharedOverlapJoinPhase::BeforeRouted` before routed grouped GEMMs (used when
/// full `AfterRouted` overlap is not active). Default OFF on GB10 (`active_sm_count() <= 64`,
/// where serial 4-stage `gemm_discard` is faster; set `=1`/`on`/`true` to opt in) and auto-ON
/// on wide-SM GPUs (`active_sm_count() > 64`; set `=0`/`off`/`false` to opt out).
const PREFILL_SHARED_SORT_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SHARED_SORT_OVERLAP";
/// Minimum batch rows (`items >= 3`) for automatic shared-expert pre-transcode.
const PRETRANSCODE_MIN_ROWS: usize = 256;
/// Maximum shared-expert batch rows for 2-CTA/SM `v28_m32` (`TILE_M=32`, single weight pass).
const SHARED_V28_M32_MAX_ROWS: usize = 32;
/// Maximum shared-expert batch rows for `v28_m64` (`TILE_M=64`, single weight pass; `rows > 64`
/// promotes to `Twarp` `TILE_M=128` so weights are read once instead of 2-4x).
const SHARED_V28_M64_MAX_ROWS: usize = 64;
/// MXFP8 input activation bytes per token in `expert_up_out` (`2560` E4M3 + `80` UE8M0).
const SHARED_W4A8_XQ_BYTES_PER_ROW: usize = 2560 + 80;
/// MXFP8 activated intermediate bytes per row for the shared expert (`640` E4M3 + `20` UE8M0).
pub(super) const SHARED_W4A8_ACT_BYTES_PER_ROW: usize = 640 + 20;
const V7_TILE_M: usize = 128;
const WORKSPACE_MAGIC: u64 = 0x4f52_4947_4936_3430;
const WORKSPACE_HEADER: usize = 48;
const V28_SHARED_SMEM: u32 = 101_376;
/// Dedicated PTX module for 2-stage <= 48 KiB SMEM pre-transcoded `shared_gateup`.
const V28_SHARED_GATEUP_2STAGE_MODULE: &str = "qwen4_shared_w4a8_2stage";
/// 2-stage `shared_gateup` dynamic SMEM (`33.5 KiB <= 48 KiB` -> 2 CTAs/SM on GB10).
const V28_SHARED_GATEUP_2STAGE_SMEM: u32 = 34_304;
/// Thread-block width of 2-stage `shared_gateup` (`4 M x 2 N` = 8 warps, 2 CTAs/SM).
const V28_SHARED_GATEUP_2STAGE_BLOCK: u32 = 256;
/// 2-stage `shared_down` dynamic SMEM (`33.5 KiB <= 48 KiB` -> 2 CTAs/SM on GB10).
const V28_SHARED_DOWN_2STAGE_SMEM: u32 = 34_304;
/// Thread-block width of 2-stage `shared_down` (`4 M x 2 N` = 8 warps, 2 CTAs/SM).
const V28_SHARED_DOWN_2STAGE_BLOCK: u32 = 256;
/// Self-resetting GPU atomic counter slab for `shared_down` L2 discard (both 4-stage `nt10`
/// `_discard` and 2-stage `nt20` `_2stage`): `[0..32)` per-`nt` counters (128 B) + `[32..544)`
/// per-`wi` counters (covers up to `512` 128-row work items = `65,536` rows = `abi::MAX_ROWS`).
const SHARED_DISCARD_COUNTERS_BYTES: usize = 544 * 4;
const BT_GATEUP_BYTES: usize = 200 * 16_384;
const SF_GATEUP_BYTES: usize = 200 * 512;
const BT_DOWN_BYTES: usize = 100 * 16_384;
const SF_DOWN_BYTES: usize = 100 * 512;
const PRETRANSCODE_SCRATCH_BYTES: usize =
    BT_GATEUP_BYTES + SF_GATEUP_BYTES + BT_DOWN_BYTES + SF_DOWN_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PretranscodeMode {
    Off,
    Auto,
    Force,
}

impl PretranscodeMode {
    #[cfg_attr(not(test), allow(dead_code))]
    fn parse(raw: Option<&str>) -> Self {
        Self::parse_for_sm(raw, 0)
    }

    #[must_use]
    fn parse_for_sm(raw: Option<&str>, sm_count: u32) -> Self {
        match raw.map(str::trim) {
            None | Some("") => {
                if sm_count > PRETRANSCODE_HIGH_SM_THRESHOLD {
                    Self::Auto
                } else {
                    Self::Off
                }
            }
            Some(v)
                if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") =>
            {
                Self::Off
            }
            Some(v)
                if v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true") =>
            {
                Self::Auto
            }
            Some(v) if v.eq_ignore_ascii_case("force") => Self::Force,
            Some(_) => Self::Off,
        }
    }

    const fn active_for(self, rows: usize) -> bool {
        match self {
            Self::Off => false,
            Self::Auto => rows > PRETRANSCODE_MIN_ROWS,
            Self::Force => true,
        }
    }
}

fn pretranscode_mode() -> PretranscodeMode {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(PRETRANSCODE_SELECTOR).ok());
    PretranscodeMode::parse_for_sm(raw.as_deref(), atlas_core::device::active_sm_count())
}

pub(crate) fn selected() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    *SEL.get_or_init(|| std::env::var(SELECTOR).ok().as_deref() == Some("1")) || w4a8_selected()
}

pub(crate) fn w4a8_selected() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    *SEL.get_or_init(|| std::env::var(W4A8_SELECTOR).ok().as_deref() == Some("1"))
}

pub(crate) fn pretranscode_selected() -> bool {
    pretranscode_mode() != PretranscodeMode::Off
}

pub(crate) fn pretranscode_active_for(rows: usize) -> bool {
    pretranscode_selected() && pretranscode_mode().active_for(rows)
}

fn resolve_pretranscode_use(
    mode: PretranscodeMode,
    rows: usize,
    w4a8_tag: &str,
    ssm_bytes: usize,
) -> Result<bool> {
    if !mode.active_for(rows) || w4a8_tag != "v28" {
        return Ok(false);
    }
    if ssm_bytes < PRETRANSCODE_SCRATCH_BYTES {
        ensure!(
            mode != PretranscodeMode::Force,
            "{PRETRANSCODE_SELECTOR}=force requires ssm_deinterleaved >= {PRETRANSCODE_SCRATCH_BYTES} B, got {ssm_bytes} B (rows={rows})"
        );
        return Ok(false);
    }
    Ok(true)
}

#[must_use]
fn parse_wide_sm_bool_toggle(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        None | Some("") => sm_count > PRETRANSCODE_HIGH_SM_THRESHOLD,
        Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") => {
            false
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true") => {
            true
        }
        Some(_) => false,
    }
}

#[cfg(test)]
#[must_use]
fn parse_prefill_shared_overlap(raw: Option<&str>) -> bool {
    parse_wide_sm_bool_toggle(raw, 48)
}

#[must_use]
fn resolve_prefill_shared_overlap(raw: Option<&str>, sm_count: u32) -> bool {
    parse_wide_sm_bool_toggle(raw, sm_count)
}

fn prefill_shared_overlap_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(PREFILL_SHARED_OVERLAP_SELECTOR).ok());
    resolve_prefill_shared_overlap(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[cfg(test)]
#[must_use]
fn parse_prefill_shared_full_overlap(raw: Option<&str>) -> bool {
    parse_wide_sm_bool_toggle(raw, 48)
}

#[must_use]
fn resolve_prefill_shared_full_overlap(raw: Option<&str>, sm_count: u32) -> bool {
    parse_wide_sm_bool_toggle(raw, sm_count)
}

fn prefill_shared_full_overlap_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(PREFILL_SHARED_FULL_OVERLAP_SELECTOR).ok());
    resolve_prefill_shared_full_overlap(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[cfg(test)]
#[must_use]
fn parse_prefill_shared_sort_overlap(raw: Option<&str>) -> bool {
    parse_wide_sm_bool_toggle(raw, 48)
}

#[must_use]
fn resolve_prefill_shared_sort_overlap(raw: Option<&str>, sm_count: u32) -> bool {
    parse_wide_sm_bool_toggle(raw, sm_count)
}

fn prefill_shared_sort_overlap_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(PREFILL_SHARED_SORT_OVERLAP_SELECTOR).ok());
    resolve_prefill_shared_sort_overlap(raw.as_deref(), atlas_core::device::active_sm_count())
}

/// Join boundary for an in-flight side-stream shared-expert W4A8 launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedOverlapJoinPhase {
    /// Wave-16 pre-routed overlap: join in `qwen4_prefill_compact::validate_context`
    /// before routed grouped GEMMs run (e.g. when `xq_reuse` is off or
    /// `ATLAS_QWEN4_PREFILL_SHARED_FULL_OVERLAP=0`).
    BeforeRouted,
    /// Wave-17 full overlap: routed compact W4A8 reads `(xq, xs)` read-only and writes
    /// strictly disjoint arenas, so `validate_context` defers the join until
    /// [`SharedOverlapJoinGuard::join`] immediately after `run_routed_grouped_gemm`
    /// (or RAII drop on early return / error / unwind).
    AfterRouted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingSharedOverlapJoin {
    stream: u64,
    prefill_stream: u64,
    event: u64,
    phase: SharedOverlapJoinPhase,
}

thread_local! {
    static PENDING_SHARED_OVERLAP_JOIN: Cell<Option<PendingSharedOverlapJoin>> =
        const { Cell::new(None) };
    static FULL_OVERLAP_GUARD_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

fn full_overlap_guard_active() -> bool {
    FULL_OVERLAP_GUARD_ACTIVE.with(Cell::get)
}

fn record_pending_shared_overlap_join(
    stream: u64,
    prefill_stream: u64,
    event: u64,
    phase: SharedOverlapJoinPhase,
) {
    debug_assert!(
        phase != SharedOverlapJoinPhase::AfterRouted || full_overlap_guard_active(),
        "AfterRouted shared overlap join requires an active SharedOverlapJoinGuard"
    );
    let prev = PENDING_SHARED_OVERLAP_JOIN.with(|cell| {
        cell.replace(Some(PendingSharedOverlapJoin {
            stream,
            prefill_stream,
            event,
            phase,
        }))
    });
    debug_assert!(
        prev.is_none(),
        "shared W4A8 overlap join overwritten before prior layer joined"
    );
}

fn wait_pending_shared_overlap(
    gpu: &dyn GpuBackend,
    pending: PendingSharedOverlapJoin,
) -> Result<()> {
    if let Err(err) = gpu.stream_wait_event(pending.stream, pending.event) {
        if pending.prefill_stream != 0
            && let Err(sync_err) = gpu.synchronize(pending.prefill_stream)
        {
            tracing::error!(
                "shared_w4a8 overlap fallback synchronize(prefill_stream) failed: {sync_err:#}"
            );
        }
        return Err(err);
    }
    Ok(())
}

/// Wait on the main stream for any in-flight side-stream shared-expert W4A8 launch whose phase
/// requires joining before routed GEMMs (`BeforeRouted`).
///
/// Called by `qwen4_prefill_compact::validate_context` before routed compact GEMMs. When
/// Wave-17 full overlap (`AfterRouted`) is active under an enclosing [`SharedOverlapJoinGuard`],
/// this leaves the pending join intact so `shared_w4a8` stays in flight concurrently with
/// `run_routed_grouped_gemm` until `forward_prefill` joins it after routed GEMMs complete.
pub(super) fn join_pending_shared_overlap(gpu: &dyn GpuBackend) -> Result<()> {
    let pending = PENDING_SHARED_OVERLAP_JOIN.with(|cell| match cell.get() {
        Some(entry)
            if entry.phase == SharedOverlapJoinPhase::BeforeRouted
                || !full_overlap_guard_active() =>
        {
            cell.set(None);
            Some(entry)
        }
        _ => None,
    });
    let Some(pending) = pending else {
        return Ok(());
    };
    wait_pending_shared_overlap(gpu, pending)
}

/// Unconditionally join any pending side-stream shared-expert W4A8 launch regardless of phase.
fn join_pending_shared_overlap_all(gpu: &dyn GpuBackend) -> Result<()> {
    let Some(pending) = PENDING_SHARED_OVERLAP_JOIN.with(|cell| cell.replace(None)) else {
        return Ok(());
    };
    wait_pending_shared_overlap(gpu, pending)
}

/// RAII scope guard for `MoeLayer::forward_prefill` that guarantees any side-stream
/// `shared_w4a8` launch is joined back onto its main stream before returning — on normal
/// completion (via [`Self::join`]), early `Err` propagation, or panic unwind.
///
/// Arms `FULL_OVERLAP_GUARD_ACTIVE` while live so `SharedOverlapJoinPhase::AfterRouted` can
/// only be recorded when this guard owns the post-routed join invariant.
pub(super) struct SharedOverlapJoinGuard<'a> {
    gpu: &'a dyn GpuBackend,
    prev_active: bool,
    joined: bool,
}

impl<'a> SharedOverlapJoinGuard<'a> {
    #[must_use]
    pub(super) fn enter(gpu: &'a dyn GpuBackend) -> Self {
        let prev_active = FULL_OVERLAP_GUARD_ACTIVE.with(|cell| cell.replace(true));
        Self {
            gpu,
            prev_active,
            joined: false,
        }
    }

    /// Explicitly join any in-flight side-stream shared-W4A8 launch onto its recorded main
    /// stream, propagating any backend error.
    pub(super) fn join(mut self) -> Result<()> {
        self.joined = true;
        FULL_OVERLAP_GUARD_ACTIVE.with(|cell| cell.set(self.prev_active));
        join_pending_shared_overlap_all(self.gpu)
    }
}

impl Drop for SharedOverlapJoinGuard<'_> {
    fn drop(&mut self) {
        if !self.joined {
            FULL_OVERLAP_GUARD_ACTIVE.with(|cell| cell.set(self.prev_active));
            if let Err(err) = join_pending_shared_overlap_all(self.gpu) {
                tracing::error!(
                    "SharedOverlapJoinGuard::drop failed to join shared_w4a8 overlap: {err:#}"
                );
            }
        }
    }
}

/// Required `ssm_deinterleaved` scratch bytes for side-stream shared W4A8 overlap.
///
/// When `pretranscode` is active, the 4.83 MiB pre-transcoded weight slabs occupy
/// `[0..PRETRANSCODE_SCRATCH_BYTES)` (512-byte aligned) and the MXFP8 activated
/// intermediate `(aq, as_)` follows immediately at `PRETRANSCODE_SCRATCH_BYTES`.
#[must_use]
const fn overlap_ssm_scratch_bytes(rows: usize, pretranscode: bool) -> usize {
    let act_bytes = rows * SHARED_W4A8_ACT_BYTES_PER_ROW;
    if pretranscode {
        PRETRANSCODE_SCRATCH_BYTES + act_bytes
    } else {
        act_bytes
    }
}

/// Explicit `(base, byte_len)` views of every MoE prefill arena checked for side-stream
/// non-aliasing before forking `shared_w4a8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedOverlapArenas {
    ssm_deinterleaved: (DevicePtr, usize),
    attn_output: (DevicePtr, usize),
    expert_up_out: (DevicePtr, usize),
    input: (DevicePtr, usize),
    gate_logits: (DevicePtr, usize),
    scratch: (DevicePtr, usize),
    expert_gate_out: (DevicePtr, usize),
    expert_down_out: (DevicePtr, usize),
    moe_worklist: (DevicePtr, usize),
    moe_worklist_total: (DevicePtr, usize),
    moe_gate_scalars: Option<(DevicePtr, usize)>,
    moe_output: Option<(DevicePtr, usize)>,
}

/// Verify that all three device regions touched by side-stream `shared_w4a8`
/// (`ssm_deinterleaved` pretranscode + intermediate, `attn_output` output, and
/// `expert_up_out` read-only `xq`/`xs`) are pairwise disjoint and disjoint from every
/// routed/combine arena buffer.
#[must_use]
fn shared_overlap_regions_disjoint(
    arenas: SharedOverlapArenas,
    rows: usize,
    ssm_bytes: usize,
) -> bool {
    let Some(out_bytes) = rows.checked_mul(2560 * 2) else {
        return false;
    };
    let Some(xq_bytes) = rows.checked_mul(SHARED_W4A8_XQ_BYTES_PER_ROW) else {
        return false;
    };
    if rows == 0
        || ssm_bytes == 0
        || arenas.ssm_deinterleaved.1 < ssm_bytes
        || arenas.attn_output.1 < out_bytes
        || arenas.expert_up_out.1 < xq_bytes
        || arenas.input.1 < out_bytes
        || arenas.moe_worklist.1 < abi::ARENA_BYTES
        || arenas.moe_worklist_total.1 < 4
    {
        return false;
    }
    let Ok(ssm_region) = abi::region(arenas.ssm_deinterleaved.0.0, ssm_bytes, 16) else {
        return false;
    };
    let Ok(out_region) = abi::region(arenas.attn_output.0.0, out_bytes, 16) else {
        return false;
    };
    let Ok(xq_region) = abi::region(arenas.expert_up_out.0.0, arenas.expert_up_out.1, 16) else {
        return false;
    };
    if !abi::disjoint(ssm_region, out_region)
        || !abi::disjoint(ssm_region, xq_region)
        || !abi::disjoint(out_region, xq_region)
    {
        return false;
    }
    let peers = [
        (arenas.input.0, arenas.input.1, 16u64),
        (arenas.gate_logits.0, arenas.gate_logits.1, 16),
        (arenas.scratch.0, arenas.scratch.1, 16),
        (arenas.expert_gate_out.0, arenas.expert_gate_out.1, 16),
        (arenas.expert_down_out.0, arenas.expert_down_out.1, 16),
        (arenas.moe_worklist.0, abi::ARENA_BYTES, 16),
        (arenas.moe_worklist_total.0, 4, 4),
    ];
    for (ptr, bytes, align) in peers {
        let Ok(peer) = abi::region(ptr.0, bytes, align) else {
            return false;
        };
        if !abi::disjoint(ssm_region, peer)
            || !abi::disjoint(out_region, peer)
            || !abi::disjoint(xq_region, peer)
        {
            return false;
        }
    }
    for (opt_peer, align) in [(arenas.moe_gate_scalars, 4u64), (arenas.moe_output, 16u64)] {
        if let Some((ptr, bytes)) = opt_peer
            && !ptr.is_null()
            && bytes > 0
        {
            let Ok(peer) = abi::region(ptr.0, bytes, align) else {
                return false;
            };
            if !abi::disjoint(ssm_region, peer)
                || !abi::disjoint(out_region, peer)
                || !abi::disjoint(xq_region, peer)
            {
                return false;
            }
        }
    }
    true
}

fn shared_overlap_arenas_disjoint(
    ctx: &ForwardContext,
    input: DevicePtr,
    rows: usize,
    ssm_bytes: usize,
) -> bool {
    let sizes = ctx.buffers.sizes();
    let Some(out_bytes) = rows.checked_mul(2560 * 2) else {
        return false;
    };
    let input_bytes = if input == ctx.buffers.norm_output() {
        sizes.norm_output
    } else {
        out_bytes
    };
    let gate_scalars = ctx.buffers.moe_gate_scalars();
    let moe_out = ctx.buffers.moe_output();
    shared_overlap_regions_disjoint(
        SharedOverlapArenas {
            ssm_deinterleaved: (ctx.buffers.ssm_deinterleaved(), sizes.ssm_deinterleaved),
            attn_output: (ctx.buffers.attn_output(), sizes.attn_output),
            expert_up_out: (ctx.buffers.expert_up_out(), sizes.expert_up_out),
            input: (input, input_bytes),
            gate_logits: (ctx.buffers.gate_logits(), sizes.gate_logits),
            scratch: (ctx.buffers.scratch(), sizes.scratch),
            expert_gate_out: (ctx.buffers.expert_gate_out(), sizes.expert_gate_out),
            expert_down_out: (ctx.buffers.expert_down_out(), sizes.expert_down_out),
            moe_worklist: (ctx.buffers.moe_worklist(), sizes.moe_worklist),
            moe_worklist_total: (ctx.buffers.moe_worklist_total(), sizes.moe_worklist_total),
            moe_gate_scalars: (!gate_scalars.is_null() && sizes.moe_gate_scalars > 0)
                .then_some((gate_scalars, sizes.moe_gate_scalars)),
            moe_output: (!moe_out.is_null() && sizes.moe_output > 0)
                .then_some((moe_out, sizes.moe_output)),
        },
        rows,
        ssm_bytes,
    )
}

/// Explicit safety preconditions for deferring the side-stream shared-W4A8 join past
/// `run_routed_grouped_gemm` (Wave-17 full overlap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedFullOverlapChecks {
    base_overlap_eligible: bool,
    full_overlap_enabled: bool,
    guard_active: bool,
    routed_w4a8_active: bool,
    xq_reuse_active: bool,
    input_is_norm_output: bool,
    oracle_disabled: bool,
    arenas_disjoint: bool,
}

/// True iff every memory-disjointness, read-only `xq` reuse, and RAII-join safety invariant
/// holds so `shared_w4a8` on `prefill_stream` may overlap `run_routed_grouped_gemm` on
/// `main_stream` with zero stream or memory hazards.
#[must_use]
const fn disjoint_overlap_safe(checks: SharedFullOverlapChecks) -> bool {
    checks.base_overlap_eligible
        && checks.full_overlap_enabled
        && checks.guard_active
        && checks.routed_w4a8_active
        && checks.xq_reuse_active
        && checks.input_is_norm_output
        && checks.oracle_disabled
        && checks.arenas_disjoint
}

/// Explicit safety preconditions for post-router shared-W4A8 overlap with
/// `topk -> 1-CTA sort_plan` (joining `BeforeRouted` before routed grouped GEMMs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedSortOverlapChecks {
    sort_overlap_enabled: bool,
    use_pretranscode: bool,
    has_gateup_2stage: bool,
    has_down_discard: bool,
    common_overlap_eligible: bool,
    guard_active: bool,
    routed_w4a8_active: bool,
    xq_reuse_active: bool,
    input_is_norm_output: bool,
    oracle_disabled: bool,
    arenas_disjoint: bool,
}

/// True iff every memory-disjointness, read-only `xq` reuse, and RAII-join invariant holds
/// so pre-transcoded `shared_w4a8` may fork onto `prefill_stream` right after the router GEMM,
/// overlap `topk -> 1-CTA sort_plan`, and join before routed grouped GEMMs.
#[must_use]
const fn sort_overlap_safe(checks: SharedSortOverlapChecks) -> bool {
    checks.sort_overlap_enabled
        && checks.use_pretranscode
        && checks.has_gateup_2stage
        && checks.has_down_discard
        && checks.common_overlap_eligible
        && checks.guard_active
        && checks.routed_w4a8_active
        && checks.xq_reuse_active
        && checks.input_is_norm_output
        && checks.oracle_disabled
        && checks.arenas_disjoint
}

/// Selected shared-expert side-stream overlap strategy for one MoE prefill layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedOverlapPlan {
    /// Pre-router fork, `AfterRouted` join (overlaps router, topk, sort, and routed GEMMs).
    FullAfterRouted,
    /// Post-router deferred fork, `BeforeRouted` join (overlaps topk and 1-CTA sort_plan).
    PostRouterSortBeforeRouted,
    /// Pre-router fork, `BeforeRouted` join (legacy Wave-16 overlap when full overlap is off).
    PreRouterBeforeRouted,
    /// Serial execution on the main stream.
    Serial,
}

#[must_use]
const fn select_shared_overlap_plan_with_policy(
    full_overlap: bool,
    sort_overlap: bool,
    base_overlap: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> SharedOverlapPlan {
    if det_batch.is_enabled() {
        return SharedOverlapPlan::Serial;
    }
    select_shared_overlap_plan(full_overlap, sort_overlap, base_overlap)
}

#[must_use]
const fn select_shared_overlap_plan(
    full_overlap: bool,
    sort_overlap: bool,
    base_overlap: bool,
) -> SharedOverlapPlan {
    if full_overlap {
        SharedOverlapPlan::FullAfterRouted
    } else if sort_overlap {
        SharedOverlapPlan::PostRouterSortBeforeRouted
    } else if base_overlap {
        SharedOverlapPlan::PreRouterBeforeRouted
    } else {
        SharedOverlapPlan::Serial
    }
}

#[derive(Clone, Copy)]
struct PretranscodeKernels {
    pre: KernelHandle,
    gateup: KernelHandle,
    gateup_2stage: Option<KernelHandle>,
    gemm: KernelHandle,
    gemm_discard: Option<KernelHandle>,
    gemm_2stage: Option<KernelHandle>,
    discard_counters: DevicePtr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PretranscodeGateUpDispatch {
    use_2stage: bool,
    grid_x: u32,
    block_x: u32,
    smem_bytes: u32,
}

#[must_use]
const fn select_pretranscode_gateup_dispatch(
    rows: usize,
    use_2stage: bool,
    fallback_block: u32,
) -> PretranscodeGateUpDispatch {
    let items = rows.div_ceil(V7_TILE_M);
    if use_2stage {
        PretranscodeGateUpDispatch {
            use_2stage: true,
            grid_x: (items * 10) as u32,
            block_x: V28_SHARED_GATEUP_2STAGE_BLOCK,
            smem_bytes: V28_SHARED_GATEUP_2STAGE_SMEM,
        }
    } else {
        PretranscodeGateUpDispatch {
            use_2stage: false,
            grid_x: (items * 5) as u32,
            block_x: fallback_block,
            smem_bytes: V28_SHARED_SMEM,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PretranscodeDownPreference {
    /// Prefer 4-stage `TN=256` `_discard` (`items * 10` CTAs, half the `aq` reads and atomics).
    FourStageDiscard,
    /// Prefer 2-stage `TN=128` `_2stage` (`<= 48 KiB` SMEM for concurrent `AfterRouted` overlap).
    TwoStage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PretranscodeDownVariant {
    FourStageDiscard,
    TwoStage,
    FourStagePlain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PretranscodeDownDispatch {
    variant: PretranscodeDownVariant,
    grid_x: u32,
    block_x: u32,
    smem_bytes: u32,
}

#[must_use]
const fn select_pretranscode_down_dispatch(
    rows: usize,
    preference: PretranscodeDownPreference,
    has_discard: bool,
    has_2stage: bool,
    fallback_block: u32,
) -> PretranscodeDownDispatch {
    let items = rows.div_ceil(V7_TILE_M);
    let variant = match preference {
        PretranscodeDownPreference::FourStageDiscard => {
            if has_discard {
                PretranscodeDownVariant::FourStageDiscard
            } else if has_2stage {
                PretranscodeDownVariant::TwoStage
            } else {
                PretranscodeDownVariant::FourStagePlain
            }
        }
        PretranscodeDownPreference::TwoStage => {
            if has_2stage {
                PretranscodeDownVariant::TwoStage
            } else if has_discard {
                PretranscodeDownVariant::FourStageDiscard
            } else {
                PretranscodeDownVariant::FourStagePlain
            }
        }
    };
    match variant {
        PretranscodeDownVariant::TwoStage => PretranscodeDownDispatch {
            variant,
            grid_x: (items * 20) as u32,
            block_x: V28_SHARED_DOWN_2STAGE_BLOCK,
            smem_bytes: V28_SHARED_DOWN_2STAGE_SMEM,
        },
        PretranscodeDownVariant::FourStageDiscard | PretranscodeDownVariant::FourStagePlain => {
            PretranscodeDownDispatch {
                variant,
                grid_x: (items * 10) as u32,
                block_x: fallback_block,
                smem_bytes: V28_SHARED_SMEM,
            }
        }
    }
}

fn pretranscode_kernels(gpu: &dyn GpuBackend) -> Result<PretranscodeKernels> {
    static ST: OnceLock<Result<PretranscodeKernels, String>> = OnceLock::new();
    ST.get_or_init(|| {
        let m = "qwen4_moe_compact_t_k32_v28";
        let load = |f: &str| gpu.kernel(m, f).map_err(|e| format!("{m}::{f}: {e}"));
        let pre = load("qwen4_shared_pretranscode_v28")?;
        let gateup = load("qwen4_shared_w4a8_gateup_k32_v28")?;
        let gemm = load("qwen4_shared_w4a8_gemm_k32_v28")?;
        gpu.set_kernel_max_dynamic_shared_memory(gateup, V28_SHARED_SMEM)
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(gemm, V28_SHARED_SMEM))
            .map_err(|e| e.to_string())?;
        let gateup_2stage = gpu
            .kernel(
                V28_SHARED_GATEUP_2STAGE_MODULE,
                "qwen4_shared_w4a8_gateup_k32_v28_2stage",
            )
            .ok()
            .filter(|&k| {
                gpu.set_kernel_max_dynamic_shared_memory(k, V28_SHARED_GATEUP_2STAGE_SMEM)
                    .is_ok()
            });
        let gemm_discard = gpu
            .kernel(m, "qwen4_shared_w4a8_gemm_k32_v28_discard")
            .ok()
            .filter(|&k| {
                gpu.set_kernel_max_dynamic_shared_memory(k, V28_SHARED_SMEM)
                    .is_ok()
            });
        let gemm_2stage = gpu
            .kernel(m, "qwen4_shared_w4a8_gemm_k32_v28_2stage")
            .ok()
            .filter(|&k| {
                gpu.set_kernel_max_dynamic_shared_memory(k, V28_SHARED_DOWN_2STAGE_SMEM)
                    .is_ok()
            });
        let discard_counters = if gemm_discard.is_some() || gemm_2stage.is_some() {
            match gpu.alloc(SHARED_DISCARD_COUNTERS_BYTES) {
                Ok(ptr) if gpu.memset(ptr, 0, SHARED_DISCARD_COUNTERS_BYTES).is_ok() => ptr,
                Ok(ptr) => {
                    let _ = gpu.free(ptr);
                    DevicePtr::NULL
                }
                Err(_) => DevicePtr::NULL,
            }
        } else {
            DevicePtr::NULL
        };
        Ok(PretranscodeKernels {
            pre,
            gateup,
            gateup_2stage,
            gemm,
            gemm_discard,
            gemm_2stage,
            discard_counters,
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{PRETRANSCODE_SELECTOR}: {e}"))
}

/// One layer's one-entry pointer tables: [gate_packed, gate_scale, up_packed, up_scale,
/// down_packed, down_scale] (u64) followed by [gate, up, down] scale2 (f32).
struct SharedTables {
    device: DevicePtr,
    _down_t: Option<QuantizedWeight>,
}

fn tables_for(
    gpu: &dyn GpuBackend,
    layer: &MoeLayer,
    h: usize,
    inter: usize,
    stream: u64,
) -> Result<DevicePtr> {
    let gate = layer
        .shared_gate_t
        .as_ref()
        .context("shared V7: transposed shared gate absent")?;
    // Keyed by the layer's own transposed gate buffer: the unified loader frees the
    // original shared weights and NULLs their pointers, so those are not per-layer keys.
    let key = gate.weight.0;
    ensure!(key != 0, "shared V7: transposed shared gate is NULL");
    let mut guard = SHARED_TABLES
        .lock()
        .expect("shared V7 tables lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(t) = map.get(&key) {
        return Ok(t.device);
    }
    ensure!(
        !super::qwen4_moe_status::is_capturing(gpu, stream),
        "shared V7: pointer table not pre-allocated before CUDA graph capture"
    );
    let up = layer
        .shared_up_t
        .as_ref()
        .context("shared V7: transposed shared up absent")?;
    let (down, down_owned) = match layer.shared_down_t.as_ref() {
        Some(d) => (*d, None),
        None => {
            let d = layer
                .weights
                .shared_expert
                .down_proj
                .transpose_for_gemm(gpu, h, inter)?;
            (d, Some(d))
        }
    };
    let mut bytes = Vec::with_capacity(64);
    for w in [gate, up, &down] {
        ensure!(
            !w.weight.is_null() && !w.weight_scale.is_null() && w.weight_scale_2.is_finite(),
            "shared V7: invalid NVFP4 shared weight"
        );
        bytes.extend_from_slice(&w.weight.0.to_le_bytes());
        bytes.extend_from_slice(&w.weight_scale.0.to_le_bytes());
    }
    for w in [gate, up, &down] {
        bytes.extend_from_slice(&w.weight_scale_2.to_le_bytes());
    }
    let device = gpu.alloc(64).context("shared V7: alloc pointer tables")?;
    // Stream-ordered with the kernels that read it (a pageable copy_h2d on the legacy stream can
    // return before its DMA lands; the prefill stream is NonBlocking).
    gpu.copy_h2d_async(&bytes, device, stream)?;
    map.insert(
        key,
        SharedTables {
            device,
            _down_t: down_owned,
        },
    );
    Ok(device)
}

/// Grow-only `[offsets 0, rows][identity 0..rows)` u32 buffer (rows is fixed per call).
static SHARED_ID: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
static SHARED_TABLES: Mutex<Option<HashMap<u64, SharedTables>>> = Mutex::new(None);

fn identity(gpu: &dyn GpuBackend, rows: usize, stream: u64) -> Result<DevicePtr> {
    let mut guard = SHARED_ID.lock().expect("shared V7 identity lock poisoned");
    if let Some((ptr, cap)) = *guard
        && cap >= rows
    {
        return Ok(ptr);
    }
    ensure!(
        !super::qwen4_moe_status::is_capturing(gpu, stream),
        "shared V7: identity buffer not pre-allocated for rows={rows} before CUDA graph capture"
    );
    let cap = rows.next_multiple_of(8192);
    let mut host = Vec::with_capacity((cap + 16) * 4);
    host.extend_from_slice(&[0u8; 16]); // offsets written per call
    for i in 0..cap as u32 {
        host.extend_from_slice(&i.to_le_bytes());
    }
    let ptr = gpu.alloc(host.len()).context("shared V7: alloc identity")?;
    gpu.copy_h2d_async(&host, ptr, stream)?; // stream-ordered, as in tables_for
    *guard = Some((ptr, cap));
    Ok(ptr)
}

/// Pre-allocate and upload the `0..8192` identity index buffer once at model load time so CUDA
/// graph capture never allocates it lazily.
pub(crate) fn prewarm_identity(gpu: &dyn GpuBackend) -> Result<()> {
    let mut guard = SHARED_ID.lock().expect("shared V7 identity lock poisoned");
    if guard.is_some() {
        return Ok(());
    }
    let cap = 8192usize;
    let mut host = Vec::with_capacity((cap + 16) * 4);
    host.extend_from_slice(&[0u8; 16]);
    for i in 0..cap as u32 {
        host.extend_from_slice(&i.to_le_bytes());
    }
    let ptr = gpu
        .alloc(host.len())
        .context("shared V7: prewarm identity")?;
    gpu.copy_h2d(&host, ptr)?;
    *guard = Some((ptr, cap));
    Ok(())
}

/// True when this layer's shared-expert pointer table and identity index buffer are already
/// resident on device for `rows`, so launching inside CUDA graph capture allocates nothing.
pub(super) fn is_capture_ready(layer: &MoeLayer, rows: usize) -> bool {
    let Some(gate) = layer.shared_gate_t.as_ref() else {
        return false;
    };
    let key = gate.weight.0;
    if key == 0 {
        return false;
    }
    let id_ok = SHARED_ID
        .lock()
        .expect("shared V7 identity lock poisoned")
        .is_some_and(|(_, cap)| cap >= rows);
    if !id_ok {
        return false;
    }
    SHARED_TABLES
        .lock()
        .expect("shared V7 tables lock poisoned")
        .as_ref()
        .is_some_and(|map| map.contains_key(&key))
}

/// Exact byte length of the cached `"shared"` constant blob (`contract || workspace || offsets`)
/// for `rows`.
#[must_use]
pub(super) const fn shared_constant_len(rows: usize) -> usize {
    abi::CONTRACT_BYTES + WORKSPACE_HEADER + rows.div_ceil(V7_TILE_M) * 8 + 8
}

/// Build the single concatenated `"shared"` constant blob (`contract || workspace || offsets`)
/// with one allocation (called only on first cache miss or load-time prewarm).
pub(super) fn shared_constant_bytes(rows: usize) -> Result<Vec<u8>> {
    let contract_bytes = abi::contract(rows).map_err(anyhow::Error::msg)?;
    let items = rows.div_ceil(V7_TILE_M);
    let ws_len = WORKSPACE_HEADER + items * 8;
    ensure!(
        ws_len <= abi::WORKSPACE_BYTES,
        "shared V7: {rows} rows exceed the workspace"
    );
    let mut bytes = Vec::with_capacity(abi::CONTRACT_BYTES + ws_len + 8);
    bytes.extend_from_slice(&contract_bytes);
    append_workspace_bytes(&mut bytes, rows);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
    Ok(bytes)
}

fn append_workspace_bytes(w: &mut Vec<u8>, rows: usize) {
    let items = rows.div_ceil(V7_TILE_M);
    w.extend_from_slice(&WORKSPACE_MAGIC.to_le_bytes()); // ready_magic
    for word in [
        1u32,
        rows as u32,
        rows as u32 * 10,
        items as u32,
        1,
        rows as u32,
        0,
    ] {
        w.extend_from_slice(&word.to_le_bytes()); // version rows expanded work_count active max_rows reserved
    }
    w.extend_from_slice(&[0u8; 4]); // pad to census_hash
    w.extend_from_slice(&0u64.to_le_bytes()); // census_hash
    for t in 0..items as u32 {
        w.extend_from_slice(&0u32.to_le_bytes()); // expert 0
        w.extend_from_slice(&t.to_le_bytes());
    }
}

fn workspace_bytes(rows: usize) -> Vec<u8> {
    let items = rows.div_ceil(V7_TILE_M);
    let mut w = Vec::with_capacity(WORKSPACE_HEADER + items * 8);
    append_workspace_bytes(&mut w, rows);
    w
}

/// Pre-staged descriptors for launching the W4A8 shared expert (direct or V28 pre-transcoded)
/// on a secondary stream concurrently with the routed MoE path.
///
/// Every device-constant upload, status-slot claim, and BF16 -> MXFP8 input quantization
/// (`xq`, `xs` in `expert_up_out`) runs on the primary stream before the fork event so both
/// streams read `(xq, xs)` read-only while writing disjoint intermediates (`ssm_deinterleaved`
/// vs `expert_gate_out`) and disjoint outputs (`attn_output` vs `expert_down_out` -> `moe_output`).
pub(super) struct SharedW4a8OverlapPrepared {
    wide: super::qwen4_prefill_compact::WideKernels,
    pretranscode: Option<(PretranscodeKernels, DevicePtr)>,
    use_gateup_2stage: bool,
    down_preference: PretranscodeDownPreference,
    tables: DevicePtr,
    ids: DevicePtr,
    id_rows: DevicePtr,
    contract: DevicePtr,
    workspace: DevicePtr,
    status: DevicePtr,
    xq: DevicePtr,
    xs: DevicePtr,
    aq: DevicePtr,
    as_: DevicePtr,
    out: DevicePtr,
    rows: usize,
}

impl SharedW4a8OverlapPrepared {
    pub(super) fn launch(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if let Some((pt, bt_gateup)) = self.pretranscode {
            launch_pretranscode_w4a8_gemms(gpu, self, pt, bt_gateup, stream)?;
            log_pretranscode_engaged(self.rows);
        } else {
            launch_direct_w4a8_gemms(gpu, self, stream)?;
        }
        log_shared_engaged(true, self.rows, self.wide.w4a8_tag);
        Ok(())
    }
}

/// Post-router deferred side-stream shared-W4A8 launch descriptor.
///
/// Created by [`MoeLayer::run_qwen4_shared_v7_prefill`] once `(xq, xs)`, device constants, and
/// async status have been staged on `stream`, then handed to
/// [`MoeLayer::fork_qwen4_shared_w4a8_overlap`] right after the router GEMM completes so
/// pre-transcoded `shared_w4a8` runs concurrently with `topk -> 1-CTA sort_plan` and joins
/// before routed grouped GEMMs without contending with the router GEMM or colliding on SMEM.
pub(super) struct DeferredSharedW4a8Fork {
    prepared: SharedW4a8OverlapPrepared,
    join_phase: SharedOverlapJoinPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedDispatchMode {
    Immediate,
    AllowPostRouterDefer,
}

/// Total pre-transcode slices for the shared expert (`200` gate/up + `100` down).
const PRETRANSCODE_TOTAL_SLICES: u32 = 300;
/// GB10 pre-transcode grid width (`100` CTAs = 3 slices/CTA across 48 SMs @ 2 CTAs/SM).
const PRETRANSCODE_GRID_GB10: u32 = 100;
/// SM count above which `qwen4_shared_pretranscode_v28` launches all 300 slices in 1 wave.
const PRETRANSCODE_HIGH_SM_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
const fn pretranscode_grid_for_sm(sm_count: u32) -> u32 {
    if sm_count > PRETRANSCODE_HIGH_SM_THRESHOLD {
        PRETRANSCODE_TOTAL_SLICES
    } else {
        PRETRANSCODE_GRID_GB10
    }
}

fn launch_pretranscode_w4a8_gemms(
    gpu: &dyn GpuBackend,
    p: &SharedW4a8OverlapPrepared,
    pt: PretranscodeKernels,
    bt_gateup: DevicePtr,
    stream: u64,
) -> Result<()> {
    let sf_gateup = bt_gateup.offset(BT_GATEUP_BYTES);
    let bt_down = sf_gateup.offset(SF_GATEUP_BYTES);
    let sf_down = bt_down.offset(BT_DOWN_BYTES);
    let pre_grid = pretranscode_grid_for_sm(atlas_core::device::active_sm_count());
    KernelLaunch::new(gpu, pt.pre)
        .grid([pre_grid, 1, 1])
        .block([512, 1, 1])
        .arg_ptr(p.tables)
        .arg_ptr(bt_gateup)
        .arg_ptr(sf_gateup)
        .arg_ptr(bt_down)
        .arg_ptr(sf_down)
        .launch(stream)?;
    let (gateup_kernel, use_gateup_2stage) = match pt.gateup_2stage {
        Some(k) if p.use_gateup_2stage => (k, true),
        _ => (pt.gateup, false),
    };
    let gateup_dispatch =
        select_pretranscode_gateup_dispatch(p.rows, use_gateup_2stage, p.wide.w4a8_block);
    KernelLaunch::new(gpu, gateup_kernel)
        .grid([gateup_dispatch.grid_x, 1, 1])
        .block([gateup_dispatch.block_x, 1, 1])
        .shared_mem(gateup_dispatch.smem_bytes)
        .arg_ptr(p.xq)
        .arg_ptr(p.xs)
        .arg_ptr(bt_gateup)
        .arg_ptr(sf_gateup)
        .arg_ptr(p.tables.offset(48))
        .arg_ptr(p.aq)
        .arg_ptr(p.as_)
        .arg_ptr(p.tables.offset(52))
        .arg_ptr(p.ids)
        .arg_ptr(p.id_rows)
        .arg_ptr(p.contract)
        .arg_ptr(p.workspace)
        .arg_ptr(p.status)
        .arg_u32(640)
        .arg_u32(2560)
        .launch(stream)?;
    let down_dispatch = select_pretranscode_down_dispatch(
        p.rows,
        p.down_preference,
        pt.gemm_discard.is_some(),
        pt.gemm_2stage.is_some(),
        p.wide.w4a8_block,
    );
    let discard_kernel = match down_dispatch.variant {
        PretranscodeDownVariant::FourStageDiscard => pt.gemm_discard,
        PretranscodeDownVariant::TwoStage => pt.gemm_2stage,
        PretranscodeDownVariant::FourStagePlain => None,
    };
    if let Some(kernel) = discard_kernel {
        KernelLaunch::new(gpu, kernel)
            .grid([down_dispatch.grid_x, 1, 1])
            .block([down_dispatch.block_x, 1, 1])
            .shared_mem(down_dispatch.smem_bytes)
            .arg_ptr(p.aq)
            .arg_ptr(p.as_)
            .arg_ptr(bt_down)
            .arg_ptr(sf_down)
            .arg_ptr(p.tables.offset(56))
            .arg_ptr(p.out)
            .arg_ptr(p.ids)
            .arg_ptr(p.id_rows)
            .arg_ptr(p.contract)
            .arg_ptr(p.workspace)
            .arg_ptr(p.status)
            .arg_u32(2560)
            .arg_u32(640)
            .arg_ptr(bt_gateup)
            .arg_ptr(pt.discard_counters)
            .launch(stream)
    } else {
        KernelLaunch::new(gpu, pt.gemm)
            .grid([down_dispatch.grid_x, 1, 1])
            .block([down_dispatch.block_x, 1, 1])
            .shared_mem(down_dispatch.smem_bytes)
            .arg_ptr(p.aq)
            .arg_ptr(p.as_)
            .arg_ptr(bt_down)
            .arg_ptr(sf_down)
            .arg_ptr(p.tables.offset(56))
            .arg_ptr(p.out)
            .arg_ptr(p.ids)
            .arg_ptr(p.id_rows)
            .arg_ptr(p.contract)
            .arg_ptr(p.workspace)
            .arg_ptr(p.status)
            .arg_u32(2560)
            .arg_u32(640)
            .launch(stream)
    }
}

fn log_pretranscode_engaged(rows: usize) {
    static LOGGED_PT: std::sync::Once = std::sync::Once::new();
    LOGGED_PT.call_once(|| {
        tracing::info!(
            "MOE_PREFILL_SHARED_W4A8_PRETRANSCODE_ENGAGED selector={PRETRANSCODE_SELECTOR} rows={rows} scratch_bytes={PRETRANSCODE_SCRATCH_BYTES} bit_identical_to_v28"
        );
    });
}

fn log_prefill_shared_overlap_engaged(
    rows: usize,
    pretranscode: bool,
    full_overlap: bool,
    w4a8_tag: &str,
) {
    static LOGGED_OVERLAP: std::sync::Once = std::sync::Once::new();
    LOGGED_OVERLAP.call_once(|| {
        tracing::info!(
            "MOE_PREFILL_SHARED_W4A8_OVERLAP_ENGAGED selector={PREFILL_SHARED_OVERLAP_SELECTOR} rows={rows} pretranscode={pretranscode} full_overlap={full_overlap} kernels={w4a8_tag}"
        );
    });
    if full_overlap {
        static LOGGED_FULL: std::sync::Once = std::sync::Once::new();
        LOGGED_FULL.call_once(|| {
            tracing::info!(
                "MOE_PREFILL_SHARED_W4A8_FULL_OVERLAP_ENGAGED selector={PREFILL_SHARED_FULL_OVERLAP_SELECTOR} rows={rows} pretranscode={pretranscode} kernels={w4a8_tag}"
            );
        });
    }
}

fn log_prefill_shared_sort_overlap_engaged(rows: usize, down_smem: u32, w4a8_tag: &str) {
    static LOGGED_SORT_OVERLAP: std::sync::Once = std::sync::Once::new();
    LOGGED_SORT_OVERLAP.call_once(|| {
        tracing::info!(
            "MOE_PREFILL_SHARED_W4A8_SORT_OVERLAP_ENGAGED selector={PREFILL_SHARED_SORT_OVERLAP_SELECTOR} rows={rows} join_phase=before_routed gateup_smem={V28_SHARED_GATEUP_2STAGE_SMEM} down_smem={down_smem} kernels={w4a8_tag}"
        );
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SharedV28LaunchPlan {
    tile: super::qwen4_compact_ops::V28OpTile,
    grid_gateup: u32,
    grid_down: u32,
}

#[must_use]
fn select_shared_v28_launch_plan_with_policy(
    rows: usize,
    m32_active: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> SharedV28LaunchPlan {
    if det_batch.is_enabled() {
        let items = rows.div_ceil(V7_TILE_M);
        return SharedV28LaunchPlan {
            tile: super::qwen4_compact_ops::V28OpTile::M64,
            grid_gateup: (items * 5).clamp(1, abi::GRID as usize) as u32,
            grid_down: (items * 10).clamp(1, abi::GRID as usize) as u32,
        };
    }
    select_shared_v28_launch_plan(rows, m32_active)
}

#[must_use]
fn select_shared_v28_launch_plan(rows: usize, m32_active: bool) -> SharedV28LaunchPlan {
    let items = rows.div_ceil(V7_TILE_M);
    if rows <= SHARED_V28_M32_MAX_ROWS && m32_active {
        SharedV28LaunchPlan {
            tile: super::qwen4_compact_ops::V28OpTile::M32,
            grid_gateup: (items * 10).clamp(1, abi::GRID as usize) as u32,
            grid_down: (items * 20).clamp(1, abi::GRID as usize) as u32,
        }
    } else if rows <= SHARED_V28_M64_MAX_ROWS {
        SharedV28LaunchPlan {
            tile: super::qwen4_compact_ops::V28OpTile::M64,
            grid_gateup: (items * 5).clamp(1, abi::GRID as usize) as u32,
            grid_down: (items * 10).clamp(1, abi::GRID as usize) as u32,
        }
    } else {
        SharedV28LaunchPlan {
            tile: super::qwen4_compact_ops::V28OpTile::Twarp,
            grid_gateup: (items * 5).clamp(1, abi::GRID as usize) as u32,
            grid_down: (items * 10).clamp(1, abi::GRID as usize) as u32,
        }
    }
}

fn launch_direct_w4a8_gemms(
    gpu: &dyn GpuBackend,
    p: &SharedW4a8OverlapPrepared,
    stream: u64,
) -> Result<()> {
    let (grid_gateup, grid_down, block, gateup_k, gateup_smem, down_k, down_smem) =
        if p.wide.w4a8_tag == "v28" {
            let plan = select_shared_v28_launch_plan_with_policy(
                p.rows,
                super::qwen4_compact_ops::v28_m32_active_for_scope(),
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
            );
            match plan.tile {
                super::qwen4_compact_ops::V28OpTile::M32 => {
                    let (gu, dn, smem) = super::qwen4_compact_ops::v28_m32_shared_kernels(gpu)?;
                    (
                        plan.grid_gateup,
                        plan.grid_down,
                        super::qwen4_compact_ops::V28_M32_BLOCK,
                        gu,
                        smem,
                        dn,
                        smem,
                    )
                }
                super::qwen4_compact_ops::V28OpTile::M64 => {
                    let (gu, dn, smem) = super::qwen4_compact_ops::v28_m64_shared_kernels(gpu)?;
                    (
                        plan.grid_gateup,
                        plan.grid_down,
                        p.wide.w4a8_block,
                        gu,
                        smem,
                        dn,
                        smem,
                    )
                }
                super::qwen4_compact_ops::V28OpTile::Twarp => (
                    plan.grid_gateup,
                    plan.grid_down,
                    p.wide.w4a8_block,
                    p.wide.w4a8_gateup,
                    p.wide.w4a8_smem,
                    p.wide.w4a8_gemm,
                    p.wide.w4a8_smem_gemm,
                ),
            }
        } else {
            (
                abi::GRID,
                abi::GRID,
                p.wide.w4a8_block,
                p.wide.w4a8_gateup,
                p.wide.w4a8_smem,
                p.wide.w4a8_gemm,
                p.wide.w4a8_smem_gemm,
            )
        };
    KernelLaunch::new(gpu, gateup_k)
        .grid([grid_gateup, 1, 1])
        .block([block, 1, 1])
        .shared_mem(gateup_smem)
        .arg_ptr(p.xq)
        .arg_ptr(p.xs)
        .arg_ptr(p.tables)
        .arg_ptr(p.tables.offset(8))
        .arg_ptr(p.tables.offset(48))
        .arg_ptr(p.aq)
        .arg_ptr(p.as_)
        .arg_ptr(p.tables.offset(16))
        .arg_ptr(p.tables.offset(24))
        .arg_ptr(p.tables.offset(52))
        .arg_ptr(p.ids)
        .arg_ptr(p.id_rows)
        .arg_ptr(p.contract)
        .arg_ptr(p.workspace)
        .arg_ptr(p.status)
        .arg_u32(640)
        .arg_u32(2560)
        .launch(stream)?;
    KernelLaunch::new(gpu, down_k)
        .grid([grid_down, 1, 1])
        .block([block, 1, 1])
        .shared_mem(down_smem)
        .arg_ptr(p.aq)
        .arg_ptr(p.as_)
        .arg_ptr(p.tables.offset(32))
        .arg_ptr(p.tables.offset(40))
        .arg_ptr(p.tables.offset(56))
        .arg_ptr(p.out)
        .arg_ptr(p.ids)
        .arg_ptr(p.id_rows)
        .arg_ptr(p.contract)
        .arg_ptr(p.workspace)
        .arg_ptr(p.status)
        .arg_u32(2560)
        .arg_u32(640)
        .launch(stream)
}

fn log_shared_engaged(w4a8: bool, rows: usize, w4a8_tag: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        if w4a8 {
            tracing::info!(
                "MOE_PREFILL_SHARED_W4A8_ENGAGED selector={W4A8_SELECTOR} rows={rows} kernels={w4a8_tag}"
            );
        } else {
            tracing::info!(
                "MOE_PREFILL_SHARED_V7_ENGAGED selector={SELECTOR} rows={rows} kernels=v7_gateup+v7_down"
            );
        }
    });
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    fn stage_shared_xq(
        &self,
        wide: super::qwen4_prefill_compact::WideKernels,
        input: DevicePtr,
        xq: DevicePtr,
        xs: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
        allow_router_fp8: bool,
    ) -> Result<()> {
        #[rustfmt::skip]
        super::qwen4_compact_ops::launch_xq_quant(ctx.gpu, ctx.buffers, &wide, input, self.weights.shared_expert_gate.weight, xq, xs, rows, stream, allow_router_fp8)?;
        super::qwen4_compact_ops::note_xq(input, rows);
        Ok(())
    }

    /// Prepare the direct W4A8 shared expert on the primary stream so its two GEMMs can execute
    /// on a secondary stream concurrently with the routed compact W4A8 kernels.
    pub(super) fn prepare_qwen4_shared_w4a8_overlap(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<SharedW4a8OverlapPrepared> {
        ensure!(
            super::qwen4_prefill_compact::request_active(self.qwen4_compact.is_some()),
            "request-suppressed Qwen4 shared-V7 dispatch"
        );
        ensure!(
            super::qwen4_moe_status::selected(),
            "shared W4A8 side-stream overlap requires async MoE status constants"
        );
        let h = ctx.config.hidden_size;
        let inter = ctx.config.shared_expert_intermediate_size;
        ensure!(
            h == 2560 && inter == 640 && (2..=abi::MAX_ROWS).contains(&rows),
            "shared V7: Flash-Next geometry only"
        );
        let wide = self.qwen4_compact.and_then(|k| k.wide).context(
            "ATLAS_QWEN4_PREFILL_SHARED_V7 needs the compact V2 path (MOE_COMPACT_V2>0)",
        )?;
        let ws_len = WORKSPACE_HEADER + rows.div_ceil(V7_TILE_M) * 8;
        ensure!(
            ws_len <= abi::WORKSPACE_BYTES,
            "shared V7: {rows} rows exceed the workspace"
        );
        let sizes = ctx.buffers.sizes();
        ensure!(
            sizes.expert_up_out >= rows * SHARED_W4A8_XQ_BYTES_PER_ROW
                && sizes.ssm_deinterleaved >= rows * SHARED_W4A8_ACT_BYTES_PER_ROW
                && sizes.attn_output >= rows * h * 2,
            "shared W4A8 overlap: MXFP8 operands or output do not fit scratch arenas"
        );
        let tables = tables_for(ctx.gpu, self, h, inter, stream)?;
        let ids_buf = identity(ctx.gpu, rows, stream)?;
        let id_rows = ids_buf.offset(16);
        let c = abi::CONTRACT_BYTES;
        let constants = super::qwen4_moe_status::constant_with(
            ctx.gpu,
            "shared",
            rows,
            c + ws_len + 8,
            || shared_constant_bytes(rows),
            stream,
        )?;
        let status = if super::forward_k5::batched_verify_tc_moe_status_async_active() {
            super::qwen4_moe_status::sticky_shared_status(ctx.gpu, stream)?
        } else {
            super::qwen4_moe_status::defer_shared(ctx.gpu, "shared", stream)?
        };
        let (contract, workspace, ids) =
            (constants, constants.offset(c), constants.offset(c + ws_len));
        let (xq, xs) = (
            ctx.buffers.expert_up_out(),
            ctx.buffers.expert_up_out().offset(rows * 2560),
        );
        // Place shared activated intermediate in ssm_deinterleaved (dead during MoE) so it is
        // strictly disjoint from routed expert_gate_out during concurrent execution.
        let (aq, as_) = (
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.ssm_deinterleaved().offset(rows * 640),
        );
        self.stage_shared_xq(wide, input, xq, xs, rows, ctx, stream, false)?;
        Ok(SharedW4a8OverlapPrepared {
            wide,
            pretranscode: None,
            use_gateup_2stage: false,
            down_preference: PretranscodeDownPreference::FourStageDiscard,
            tables,
            ids,
            id_rows,
            contract,
            workspace,
            status,
            xq,
            xs,
            aq,
            as_,
            out: ctx.buffers.attn_output(),
            rows,
        })
    }

    fn fork_prepared_shared_w4a8(
        &self,
        prepared: &SharedW4a8OverlapPrepared,
        join_phase: SharedOverlapJoinPhase,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        gpu.record_event(self.event_a, stream)?;
        gpu.stream_wait_event(self.prefill_stream, self.event_a)?;
        let shared_launch = prepared.launch(gpu, self.prefill_stream);
        let rec_b = gpu.record_event(self.event_b, self.prefill_stream);
        if let Err(err) = shared_launch {
            let needs_sync = match rec_b {
                Ok(()) => match gpu.stream_wait_event(stream, self.event_b) {
                    Ok(()) => false,
                    Err(wait_err) => {
                        tracing::error!(
                            "fork_prepared_shared_w4a8: stream_wait_event failed during error recovery: {wait_err:#}"
                        );
                        true
                    }
                },
                Err(rec_err) => {
                    tracing::error!(
                        "fork_prepared_shared_w4a8: record_event failed during error recovery: {rec_err:#}"
                    );
                    true
                }
            };
            if needs_sync && let Err(sync_err) = gpu.synchronize(self.prefill_stream) {
                tracing::error!(
                    "fork_prepared_shared_w4a8: fallback synchronize(prefill_stream) failed: {sync_err:#}"
                );
            }
            return Err(err);
        }
        if let Err(err) = rec_b {
            if let Err(sync_err) = gpu.synchronize(self.prefill_stream) {
                tracing::error!(
                    "fork_prepared_shared_w4a8: fallback synchronize(prefill_stream) after record_event error failed: {sync_err:#}"
                );
            }
            return Err(err);
        }
        record_pending_shared_overlap_join(stream, self.prefill_stream, self.event_b, join_phase);
        Ok(())
    }

    /// Fork a post-router deferred pre-transcoded shared-W4A8 launch onto `self.prefill_stream`
    /// and register its `BeforeRouted` join handle on `stream`.
    pub(super) fn fork_qwen4_shared_w4a8_overlap(
        &self,
        deferred: DeferredSharedW4a8Fork,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.fork_prepared_shared_w4a8(&deferred.prepared, deferred.join_phase, gpu, stream)?;
        let down_smem = deferred
            .prepared
            .pretranscode
            .map_or(V28_SHARED_SMEM, |(pt, _)| {
                select_pretranscode_down_dispatch(
                    deferred.prepared.rows,
                    deferred.prepared.down_preference,
                    pt.gemm_discard.is_some(),
                    pt.gemm_2stage.is_some(),
                    deferred.prepared.wide.w4a8_block,
                )
                .smem_bytes
            });
        log_prefill_shared_sort_overlap_engaged(
            deferred.prepared.rows,
            down_smem,
            deferred.prepared.wide.w4a8_tag,
        );
        Ok(())
    }

    pub(super) fn run_qwen4_shared_v7(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let deferred = self.run_qwen4_shared_v7_inner(
            input,
            rows,
            ctx,
            stream,
            SharedDispatchMode::Immediate,
        )?;
        debug_assert!(deferred.is_none());
        Ok(())
    }

    pub(super) fn run_qwen4_shared_v7_prefill(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DeferredSharedW4a8Fork>> {
        self.run_qwen4_shared_v7_inner(
            input,
            rows,
            ctx,
            stream,
            SharedDispatchMode::AllowPostRouterDefer,
        )
    }

    fn run_qwen4_shared_v7_inner(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
        mode: SharedDispatchMode,
    ) -> Result<Option<DeferredSharedW4a8Fork>> {
        join_pending_shared_overlap_all(ctx.gpu)?;
        ensure!(
            super::qwen4_prefill_compact::request_active(self.qwen4_compact.is_some()),
            "request-suppressed Qwen4 shared-V7 dispatch"
        );
        let h = ctx.config.hidden_size;
        let inter = ctx.config.shared_expert_intermediate_size;
        ensure!(
            h == 2560 && inter == 640 && (2..=abi::MAX_ROWS).contains(&rows),
            "shared V7: Flash-Next geometry only"
        );
        let wide = self.qwen4_compact.and_then(|k| k.wide).context(
            "ATLAS_QWEN4_PREFILL_SHARED_V7 needs the compact V2 path (MOE_COMPACT_V2>0)",
        )?;
        let ws_len = WORKSPACE_HEADER + rows.div_ceil(V7_TILE_M) * 8;
        ensure!(
            ws_len <= abi::WORKSPACE_BYTES,
            "shared V7: {rows} rows exceed the workspace"
        );
        let tables = tables_for(ctx.gpu, self, h, inter, stream)?;
        let ids = identity(ctx.gpu, rows, stream)?;
        let check = super::qwen4_prefill_compact::check_selected().unwrap_or(false);
        let verify_sticky = super::forward_k5::batched_verify_tc_moe_status_async_active();
        let status_async = super::qwen4_moe_status::selected()
            && !check
            && (crate::model::qwen4_prefill_engagement::active_for(rows) || verify_sticky);
        let id_rows = ids.offset(16);
        let (contract, workspace, status, ids) = if status_async {
            // One upload per row count; pass read-only device constants and a pre-initialized
            // PLANNED status word directly to the shared expert kernels (0 per-layer D2D copies,
            // 0 per-layer host Vec allocations on cache hit).
            let c = abi::CONTRACT_BYTES;
            let constants = super::qwen4_moe_status::constant_with(
                ctx.gpu,
                "shared",
                rows,
                c + ws_len + 8,
                || shared_constant_bytes(rows),
                stream,
            )?;
            let status_slot = if verify_sticky {
                super::qwen4_moe_status::sticky_shared_status(ctx.gpu, stream)?
            } else {
                super::qwen4_moe_status::defer_shared(ctx.gpu, "shared", stream)?
            };
            (
                constants,
                constants.offset(c),
                status_slot,
                constants.offset(c + ws_len),
            )
        } else {
            let contract_bytes = abi::contract(rows).map_err(anyhow::Error::msg)?;
            let ws = workspace_bytes(rows);
            let mut offsets = [0u8; 8];
            offsets[0..4].copy_from_slice(&0u32.to_le_bytes());
            offsets[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
            let workspace = ctx.buffers.moe_worklist();
            let contract = workspace.offset(abi::WORKSPACE_BYTES);
            let status = ctx.buffers.moe_worklist_total();
            let planned = 1i32.to_le_bytes();
            ctx.gpu.copy_h2d_group_on_stream(
                &[
                    HostToDeviceCopy::new(&contract_bytes, contract),
                    HostToDeviceCopy::new(&ws, workspace),
                    HostToDeviceCopy::new(&planned, status),
                    HostToDeviceCopy::new(&offsets, ids),
                ],
                stream,
            )?;
            (contract, workspace, status, ids)
        };
        let out = ctx.buffers.attn_output();
        if w4a8_selected() {
            // Routed scratch is free until the routed path (which runs after this): MXFP8 x in
            // the up buffer, the MXFP8 activated intermediate in the gate buffer.
            let sizes = ctx.buffers.sizes();
            ensure!(
                sizes.expert_up_out >= rows * SHARED_W4A8_XQ_BYTES_PER_ROW
                    && sizes.expert_gate_out >= rows * SHARED_W4A8_ACT_BYTES_PER_ROW
                    && sizes.attn_output >= rows * h * 2,
                "shared W4A8: MXFP8 operands or output do not fit the routed scratch"
            );
            let (xq, xs) = (
                ctx.buffers.expert_up_out(),
                ctx.buffers.expert_up_out().offset(rows * 2560),
            );
            let (aq, as_) = (
                ctx.buffers.expert_gate_out(),
                ctx.buffers.expert_gate_out().offset(rows * 640),
            );
            let allow_router_fp8 = rows >= 1024
                && super::qwen4_prefill_compact::xq_reuse_selected()?
                && crate::layers::qwen4_fast_proj::fp8_tensor_scales()
                && self.gate_nvfp4.is_some()
                && self.weights.router_pre_norm.is_none()
                && !super::forward_prefill::router_bf16_tk64_selected(rows)
                && !crate::layers::qwen4_prefill_native::router_bf16(self.weights.gate.weight);
            self.stage_shared_xq(wide, input, xq, xs, rows, ctx, stream, allow_router_fp8)?;
            let use_pretranscode = resolve_pretranscode_use(
                pretranscode_mode(),
                rows,
                wide.w4a8_tag,
                sizes.ssm_deinterleaved,
            )?;
            let pt_kernels = if use_pretranscode {
                Some(pretranscode_kernels(ctx.gpu)?)
            } else {
                None
            };
            let overlap_ssm_bytes = overlap_ssm_scratch_bytes(rows, use_pretranscode);
            // Overlap shared W4A8 (`pretranscode` + `gateup` + `gemm`, or direct `gateup` + `gemm`)
            // on `self.prefill_stream` concurrently with the routed path on `stream`.
            // All inputs (`xq`, `xs`, pointer tables, cached constants, async status slot) are
            // already staged on `stream` before `event_a`; shared intermediates live exclusively
            // in `ssm_deinterleaved` and output in `attn_output`, strictly disjoint from every
            // routed arena. All three overlap selectors (`ATLAS_QWEN4_MOE_SHARED_OVERLAP`,
            // `ATLAS_QWEN4_PREFILL_SHARED_FULL_OVERLAP`, and
            // `ATLAS_QWEN4_PREFILL_SHARED_SORT_OVERLAP`) default OFF on `<= 64` SMs (GB10) and
            // auto-ON on `> 64` SMs (SM120/SM103). `select_shared_overlap_plan` resolves
            // precedence in order:
            // 1. `FullAfterRouted` (when `disjoint_overlap_safe` holds): pre-router fork,
            //    joining `AfterRouted` once routed GEMMs complete;
            // 2. `PostRouterSortBeforeRouted` (when `sort_overlap_safe` holds and full overlap
            //    is not active): post-router deferred fork using `gateup_2stage` + 4-stage
            //    `gemm_discard` to overlap `topk` and 1-CTA `sort_plan` and join `BeforeRouted`;
            // 3. `PreRouterBeforeRouted` (when `base_overlap_eligible` holds without full or
            //    sort overlap): pre-router fork joining `BeforeRouted` in `validate_context`;
            // 4. `Serial` on the main stream otherwise.
            let arenas_disjoint =
                shared_overlap_arenas_disjoint(ctx, input, rows, overlap_ssm_bytes);
            let common_overlap_eligible = status_async
                && !verify_sticky
                && !ctx.profile
                && !ctx.graph_capture
                && !super::qwen4_moe_status::is_capturing(ctx.gpu, stream)
                && !super::dump::enabled()
                && ctx.config.ep_world_size <= 4
                && ctx.config.tp_world_size <= 4
                && ctx.config.moe_intermediate_size == 640
                && ctx.config.num_experts == 512
                && ctx.config.num_experts_per_tok == 10
                && self.weights.router_pre_norm.is_none()
                && self.pre_expert_norm.is_none()
                && self.qwen4_compact.is_some_and(|k| !k.streamed)
                && super::qwen4_prefill_compact::v2_level().unwrap_or(0) > 0
                && self.prefill_stream != 0
                && self.prefill_stream != stream
                && self.event_a != 0
                && self.event_b != 0
                && arenas_disjoint;
            let routed_w4a8_active = common_overlap_eligible
                && !check
                && crate::layers::qwen4_fakequant::current()
                    .ok()
                    .is_some_and(|spec| {
                        (super::qwen4_prefill_compact::w4a8_selected().unwrap_or(false)
                            || spec.moe_kernel_w4a8)
                            && spec.moe_in.is_none()
                            && spec.moe_down.is_none()
                            && !spec.moe_weights_b12x
                            && !spec.moe_kernel_v7
                    });
            let xq_reuse_active = common_overlap_eligible
                && super::qwen4_prefill_compact::xq_reuse_selected().unwrap_or(false);
            let input_is_norm_output = input == ctx.buffers.norm_output();
            let oracle_disabled = crate::layers::qwen4_oracle::root().is_none();
            let base_overlap_eligible = common_overlap_eligible && prefill_shared_overlap_enabled();
            let full_overlap = disjoint_overlap_safe(SharedFullOverlapChecks {
                base_overlap_eligible,
                full_overlap_enabled: prefill_shared_full_overlap_enabled(),
                guard_active: full_overlap_guard_active(),
                routed_w4a8_active,
                xq_reuse_active,
                input_is_norm_output,
                oracle_disabled,
                arenas_disjoint,
            });
            let sort_overlap = pt_kernels.is_some_and(|pt| {
                matches!(mode, SharedDispatchMode::AllowPostRouterDefer)
                    && sort_overlap_safe(SharedSortOverlapChecks {
                        sort_overlap_enabled: prefill_shared_sort_overlap_enabled(),
                        use_pretranscode,
                        has_gateup_2stage: pt.gateup_2stage.is_some(),
                        has_down_discard: pt.gemm_discard.is_some() || pt.gemm_2stage.is_some(),
                        common_overlap_eligible,
                        guard_active: full_overlap_guard_active(),
                        routed_w4a8_active,
                        xq_reuse_active,
                        input_is_norm_output,
                        oracle_disabled,
                        arenas_disjoint,
                    })
            });
            match select_shared_overlap_plan_with_policy(
                full_overlap,
                sort_overlap,
                base_overlap_eligible,
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
            ) {
                SharedOverlapPlan::FullAfterRouted => {
                    let ssm = ctx.buffers.ssm_deinterleaved();
                    let (pretranscode, act_base, use_gateup_2stage) = if let Some(pt) = pt_kernels {
                        (
                            Some((pt, ssm)),
                            ssm.offset(PRETRANSCODE_SCRATCH_BYTES),
                            pt.gateup_2stage.is_some(),
                        )
                    } else {
                        (None, ssm, false)
                    };
                    let prepared = SharedW4a8OverlapPrepared {
                        wide,
                        pretranscode,
                        use_gateup_2stage,
                        down_preference: PretranscodeDownPreference::TwoStage,
                        tables,
                        ids,
                        id_rows,
                        contract,
                        workspace,
                        status,
                        xq,
                        xs,
                        aq: act_base,
                        as_: act_base.offset(rows * 640),
                        out,
                        rows,
                    };
                    self.fork_prepared_shared_w4a8(
                        &prepared,
                        SharedOverlapJoinPhase::AfterRouted,
                        ctx.gpu,
                        stream,
                    )?;
                    log_prefill_shared_overlap_engaged(rows, use_pretranscode, true, wide.w4a8_tag);
                    return Ok(None);
                }
                SharedOverlapPlan::PostRouterSortBeforeRouted => {
                    let pt = pt_kernels.expect("sort_overlap requires pretranscode kernels");
                    let ssm = ctx.buffers.ssm_deinterleaved();
                    let act_base = ssm.offset(PRETRANSCODE_SCRATCH_BYTES);
                    return Ok(Some(DeferredSharedW4a8Fork {
                        prepared: SharedW4a8OverlapPrepared {
                            wide,
                            pretranscode: Some((pt, ssm)),
                            use_gateup_2stage: true,
                            down_preference: PretranscodeDownPreference::FourStageDiscard,
                            tables,
                            ids,
                            id_rows,
                            contract,
                            workspace,
                            status,
                            xq,
                            xs,
                            aq: act_base,
                            as_: act_base.offset(rows * 640),
                            out,
                            rows,
                        },
                        join_phase: SharedOverlapJoinPhase::BeforeRouted,
                    }));
                }
                SharedOverlapPlan::PreRouterBeforeRouted => {
                    let ssm = ctx.buffers.ssm_deinterleaved();
                    let (pretranscode, act_base) = if let Some(pt) = pt_kernels {
                        (Some((pt, ssm)), ssm.offset(PRETRANSCODE_SCRATCH_BYTES))
                    } else {
                        (None, ssm)
                    };
                    let prepared = SharedW4a8OverlapPrepared {
                        wide,
                        pretranscode,
                        use_gateup_2stage: false,
                        down_preference: PretranscodeDownPreference::FourStageDiscard,
                        tables,
                        ids,
                        id_rows,
                        contract,
                        workspace,
                        status,
                        xq,
                        xs,
                        aq: act_base,
                        as_: act_base.offset(rows * 640),
                        out,
                        rows,
                    };
                    self.fork_prepared_shared_w4a8(
                        &prepared,
                        SharedOverlapJoinPhase::BeforeRouted,
                        ctx.gpu,
                        stream,
                    )?;
                    log_prefill_shared_overlap_engaged(
                        rows,
                        use_pretranscode,
                        false,
                        wide.w4a8_tag,
                    );
                    return Ok(None);
                }
                SharedOverlapPlan::Serial => {}
            }
            let serial = SharedW4a8OverlapPrepared {
                wide,
                pretranscode: pt_kernels.map(|pt| (pt, ctx.buffers.ssm_deinterleaved())),
                use_gateup_2stage: false,
                down_preference: PretranscodeDownPreference::FourStageDiscard,
                tables,
                ids,
                id_rows,
                contract,
                workspace,
                status,
                xq,
                xs,
                aq,
                as_,
                out,
                rows,
            };
            serial.launch(ctx.gpu, stream)?;
        } else {
            let act = ctx.buffers.ssm_deinterleaved();
            KernelLaunch::new(ctx.gpu, wide.gateup)
                .grid([abi::GRID, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(tables)
                .arg_ptr(tables.offset(8))
                .arg_ptr(tables.offset(48))
                .arg_ptr(act)
                .arg_ptr(tables.offset(16))
                .arg_ptr(tables.offset(24))
                .arg_ptr(tables.offset(52))
                .arg_ptr(ids)
                .arg_ptr(id_rows)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
            KernelLaunch::new(ctx.gpu, wide.gemm)
                .grid([abi::GRID, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(act)
                .arg_ptr(tables.offset(32))
                .arg_ptr(tables.offset(40))
                .arg_ptr(tables.offset(56))
                .arg_ptr(out)
                .arg_ptr(ids)
                .arg_ptr(id_rows)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(2560)
                .arg_u32(640)
                .launch(stream)?;
        }
        // The routed path resets the status word before its own plan, so a shared-launch
        // failure must be read here or it is lost (ordered D2H drains both GEMMs first).
        if !status_async {
            let mut result = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(status, &mut result, stream)?;
            let result = i32::from_le_bytes(result);
            abi::check_status(result)
                .map_err(|e| anyhow::anyhow!("shared V7: {e}: status={result}"))?;
        }
        log_shared_engaged(w4a8_selected(), rows, wide.w4a8_tag);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_matches_the_kernel_layout() {
        let w = workspace_bytes(300);
        assert_eq!(w.len(), WORKSPACE_HEADER + 3 * 8);
        assert_eq!(
            u64::from_le_bytes(w[0..8].try_into().unwrap()),
            WORKSPACE_MAGIC
        );
        let word = |o: usize| u32::from_le_bytes(w[o..o + 4].try_into().unwrap());
        assert_eq!((word(8), word(12), word(16), word(20)), (1, 300, 3000, 3));
        assert_eq!((word(24), word(28), word(32)), (1, 300, 0));
        assert_eq!((word(48), word(52), word(56), word(60)), (0, 0, 0, 1));
    }

    #[test]
    fn pretranscode_mode_gates_small_batches_and_supports_force_override() {
        assert_eq!(PretranscodeMode::parse(None), PretranscodeMode::Off);
        assert_eq!(PretranscodeMode::parse(Some("0")), PretranscodeMode::Off);
        assert_eq!(PretranscodeMode::parse(Some("1")), PretranscodeMode::Auto);
        assert_eq!(
            PretranscodeMode::parse(Some("force")),
            PretranscodeMode::Force
        );
        assert_eq!(
            PretranscodeMode::parse(Some("invalid")),
            PretranscodeMode::Off
        );

        for rows in [2usize, 16, 20, 40, 80, 128, 256] {
            assert!(!PretranscodeMode::Off.active_for(rows), "off rows={rows}");
            assert!(
                !PretranscodeMode::Auto.active_for(rows),
                "auto must skip pretranscode at rows={rows} <= 256"
            );
            assert!(
                PretranscodeMode::Force.active_for(rows),
                "force must override gate at rows={rows}"
            );
        }
        for rows in [257usize, 384, 512, 1024, 4096] {
            assert!(!PretranscodeMode::Off.active_for(rows), "off rows={rows}");
            assert!(
                PretranscodeMode::Auto.active_for(rows),
                "auto must engage pretranscode at rows={rows} > 256"
            );
            assert!(
                PretranscodeMode::Force.active_for(rows),
                "force must engage at rows={rows}"
            );
        }
    }

    #[test]
    fn shared_v28_launch_plan_selects_m32_m64_and_twarp_by_row_tier() {
        use super::super::qwen4_compact_ops::V28OpTile;
        // Tier 1 (rows <= 32): 2-CTA/SM M32 (10/20 CTAs) when m32 is active, else M64 (5/10 CTAs).
        for rows in [2usize, 6, 9, 10, 16, 20, 32] {
            assert_eq!(
                select_shared_v28_launch_plan(rows, true),
                SharedV28LaunchPlan {
                    tile: V28OpTile::M32,
                    grid_gateup: 10,
                    grid_down: 20,
                },
                "rows={rows} with m32 active must select M32 (10/20 CTAs)"
            );
            assert_eq!(
                select_shared_v28_launch_plan(rows, false),
                SharedV28LaunchPlan {
                    tile: V28OpTile::M64,
                    grid_gateup: 5,
                    grid_down: 10,
                },
                "rows={rows} with m32 disabled falls back to M64"
            );
        }
        // Tier 2 (33..=64): M64 (TILE_M=64, 1x weight read) regardless of m32_active.
        for rows in [33usize, 40, 48, 64] {
            for m32_active in [true, false] {
                assert_eq!(
                    select_shared_v28_launch_plan(rows, m32_active),
                    SharedV28LaunchPlan {
                        tile: V28OpTile::M64,
                        grid_gateup: 5,
                        grid_down: 10,
                    },
                    "rows={rows} (m32_active={m32_active}) must select M64 (5/10 CTAs)"
                );
            }
        }
        // Tier 3 (rows > 64): Twarp (TILE_M=128, 1x weight read instead of 2-4x).
        for rows in [65usize, 80, 96, 128] {
            for m32_active in [true, false] {
                assert_eq!(
                    select_shared_v28_launch_plan(rows, m32_active),
                    SharedV28LaunchPlan {
                        tile: V28OpTile::Twarp,
                        grid_gateup: 5,
                        grid_down: 10,
                    },
                    "rows={rows} (m32_active={m32_active}) must select Twarp (5/10 CTAs)"
                );
            }
        }
        assert_eq!(
            select_shared_v28_launch_plan(256, true),
            SharedV28LaunchPlan {
                tile: V28OpTile::Twarp,
                grid_gateup: 10,
                grid_down: 20,
            }
        );
    }

    #[test]
    fn pretranscode_down_dispatch_prefers_4stage_discard_with_2stage_and_plain_fallbacks() {
        // 2-stage shared_down fits 2 CTAs/SM on GB10 (<= 48 KiB = 49,152 B SMEM, 256 threads).
        assert_eq!(V28_SHARED_DOWN_2STAGE_SMEM, 34_304);
        const {
            assert!(V28_SHARED_DOWN_2STAGE_SMEM <= 49_152);
        }
        assert_eq!(V28_SHARED_DOWN_2STAGE_BLOCK, 256);
        assert!(SHARED_DISCARD_COUNTERS_BYTES >= (32 + abi::MAX_ROWS.div_ceil(V7_TILE_M)) * 4);
        assert_eq!(SHARED_DISCARD_COUNTERS_BYTES % 128, 0);

        for (rows, items) in [(384usize, 3u32), (2048, 16), (4096, 32), (16_384, 128)] {
            // Serial and BeforeRouted overlap prefer 4-stage TN=256 _discard (items * 10 CTAs).
            assert_eq!(
                select_pretranscode_down_dispatch(
                    rows,
                    PretranscodeDownPreference::FourStageDiscard,
                    true,
                    true,
                    512,
                ),
                PretranscodeDownDispatch {
                    variant: PretranscodeDownVariant::FourStageDiscard,
                    grid_x: items * 10,
                    block_x: 512,
                    smem_bytes: V28_SHARED_SMEM,
                },
                "4-stage discard dispatch mismatch for rows={rows}"
            );
            // If _discard is absent, fall back to 2-stage discard before plain 4-stage.
            assert_eq!(
                select_pretranscode_down_dispatch(
                    rows,
                    PretranscodeDownPreference::FourStageDiscard,
                    false,
                    true,
                    512,
                ),
                PretranscodeDownDispatch {
                    variant: PretranscodeDownVariant::TwoStage,
                    grid_x: items * 20,
                    block_x: 256,
                    smem_bytes: 34_304,
                },
                "2-stage fallback mismatch for rows={rows}"
            );
            // Explicit AfterRouted full overlap prefers 2-stage <= 48 KiB SMEM.
            assert_eq!(
                select_pretranscode_down_dispatch(
                    rows,
                    PretranscodeDownPreference::TwoStage,
                    true,
                    true,
                    512,
                ),
                PretranscodeDownDispatch {
                    variant: PretranscodeDownVariant::TwoStage,
                    grid_x: items * 20,
                    block_x: 256,
                    smem_bytes: 34_304,
                },
                "2-stage preferred dispatch mismatch for rows={rows}"
            );
            assert_eq!(
                select_pretranscode_down_dispatch(
                    rows,
                    PretranscodeDownPreference::TwoStage,
                    true,
                    false,
                    512,
                ),
                PretranscodeDownDispatch {
                    variant: PretranscodeDownVariant::FourStageDiscard,
                    grid_x: items * 10,
                    block_x: 512,
                    smem_bytes: V28_SHARED_SMEM,
                },
                "4-stage discard fallback when 2-stage absent mismatch for rows={rows}"
            );
            // Plain 4-stage fallback when neither discard entrypoint is available.
            assert_eq!(
                select_pretranscode_down_dispatch(
                    rows,
                    PretranscodeDownPreference::FourStageDiscard,
                    false,
                    false,
                    512,
                ),
                PretranscodeDownDispatch {
                    variant: PretranscodeDownVariant::FourStagePlain,
                    grid_x: items * 10,
                    block_x: 512,
                    smem_bytes: V28_SHARED_SMEM,
                },
                "plain 4-stage fallback mismatch for rows={rows}"
            );
        }
    }

    #[test]
    fn prefill_shared_overlap_selector_and_ssm_layout_alignment() {
        assert!(!parse_prefill_shared_overlap(None));
        assert!(parse_prefill_shared_overlap(Some("1")));
        assert!(!parse_prefill_shared_overlap(Some("0")));

        assert!(!parse_prefill_shared_full_overlap(None));
        assert!(parse_prefill_shared_full_overlap(Some("1")));
        assert!(!parse_prefill_shared_full_overlap(Some("0")));
        assert!(!parse_prefill_shared_full_overlap(Some(" 0 ")));

        // Pre-transcode slabs end on a 512-byte boundary so `aq` and `as_` stay 128-byte aligned.
        assert_eq!(PRETRANSCODE_SCRATCH_BYTES % 512, 0);
        assert_eq!(
            overlap_ssm_scratch_bytes(16_384, false),
            16_384 * SHARED_W4A8_ACT_BYTES_PER_ROW
        );
        assert_eq!(
            overlap_ssm_scratch_bytes(16_384, true),
            PRETRANSCODE_SCRATCH_BYTES + 16_384 * SHARED_W4A8_ACT_BYTES_PER_ROW
        );
    }

    struct EventRecordingGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        waits: Mutex<Vec<(u64, u64)>>,
        syncs: Mutex<Vec<u64>>,
        fail_wait: Mutex<bool>,
    }

    impl EventRecordingGpu {
        fn new() -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                waits: Mutex::new(Vec::new()),
                syncs: Mutex::new(Vec::new()),
                fail_wait: Mutex::new(false),
            }
        }

        fn set_fail_wait(&self, fail: bool) {
            *self.fail_wait.lock().expect("fail_wait lock poisoned") = fail;
        }

        fn take_waits(&self) -> Vec<(u64, u64)> {
            std::mem::take(&mut *self.waits.lock().expect("waits lock poisoned"))
        }

        fn take_syncs(&self) -> Vec<u64> {
            std::mem::take(&mut *self.syncs.lock().expect("syncs lock poisoned"))
        }
    }

    impl GpuBackend for EventRecordingGpu {
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
            self.syncs.lock().expect("syncs lock poisoned").push(stream);
            self.inner.synchronize(stream)
        }

        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }

        #[track_caller]
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, func_name)
        }

        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }

        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }

        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }

        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }

        fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
            self.waits
                .lock()
                .expect("waits lock poisoned")
                .push((stream, event));
            if *self.fail_wait.lock().expect("fail_wait lock poisoned") {
                anyhow::bail!("injected stream_wait_event error");
            }
            Ok(())
        }
    }

    fn sample_disjoint_arenas(rows: usize, pretranscode: bool) -> (SharedOverlapArenas, usize) {
        let ssm_bytes = overlap_ssm_scratch_bytes(rows, pretranscode);
        let out_bytes = rows * 2560 * 2;
        let xq_bytes = rows * SHARED_W4A8_XQ_BYTES_PER_ROW;
        let expanded = rows * 10;
        let mut cursor = 0x1000_0000u64;
        let mut alloc = |bytes: usize| -> (DevicePtr, usize) {
            let base = cursor;
            cursor += (bytes as u64).next_multiple_of(256);
            (DevicePtr(base), bytes)
        };
        let arenas = SharedOverlapArenas {
            ssm_deinterleaved: alloc(ssm_bytes),
            attn_output: alloc(out_bytes),
            expert_up_out: alloc(xq_bytes),
            input: alloc(out_bytes),
            gate_logits: alloc((expanded * 3 + 513) * 4),
            scratch: alloc(expanded * 8),
            expert_gate_out: alloc(expanded * 640 * 2),
            expert_down_out: alloc(expanded * 2640),
            moe_worklist: alloc(abi::ARENA_BYTES),
            moe_worklist_total: alloc(4),
            moe_gate_scalars: Some(alloc(rows * 4)),
            moe_output: Some(alloc(out_bytes)),
        };
        (arenas, ssm_bytes)
    }

    #[test]
    fn disjoint_overlap_safe_enforces_every_precondition() {
        let all_ok = SharedFullOverlapChecks {
            base_overlap_eligible: true,
            full_overlap_enabled: true,
            guard_active: true,
            routed_w4a8_active: true,
            xq_reuse_active: true,
            input_is_norm_output: true,
            oracle_disabled: true,
            arenas_disjoint: true,
        };
        assert!(disjoint_overlap_safe(all_ok));

        let cases = [
            (
                "base_overlap_eligible",
                SharedFullOverlapChecks {
                    base_overlap_eligible: false,
                    ..all_ok
                },
            ),
            (
                "full_overlap_enabled",
                SharedFullOverlapChecks {
                    full_overlap_enabled: false,
                    ..all_ok
                },
            ),
            (
                "guard_active",
                SharedFullOverlapChecks {
                    guard_active: false,
                    ..all_ok
                },
            ),
            (
                "routed_w4a8_active",
                SharedFullOverlapChecks {
                    routed_w4a8_active: false,
                    ..all_ok
                },
            ),
            (
                "xq_reuse_active",
                SharedFullOverlapChecks {
                    xq_reuse_active: false,
                    ..all_ok
                },
            ),
            (
                "input_is_norm_output",
                SharedFullOverlapChecks {
                    input_is_norm_output: false,
                    ..all_ok
                },
            ),
            (
                "oracle_disabled",
                SharedFullOverlapChecks {
                    oracle_disabled: false,
                    ..all_ok
                },
            ),
            (
                "arenas_disjoint",
                SharedFullOverlapChecks {
                    arenas_disjoint: false,
                    ..all_ok
                },
            ),
        ];
        for (name, checks) in cases {
            assert!(
                !disjoint_overlap_safe(checks),
                "disjoint_overlap_safe must fail closed when {name}=false"
            );
        }
    }

    #[test]
    fn shared_overlap_regions_disjoint_verifies_buffer_non_aliasing() {
        for (rows, pretranscode) in [(4096usize, false), (16_384, true)] {
            let (base, ssm_bytes) = sample_disjoint_arenas(rows, pretranscode);
            assert!(
                shared_overlap_regions_disjoint(base, rows, ssm_bytes),
                "fully disjoint arenas must pass (rows={rows}, pretranscode={pretranscode})"
            );
        }

        let rows = 4096usize;
        let (base, ssm_bytes) = sample_disjoint_arenas(rows, true);
        let ssm_tail = base.ssm_deinterleaved.0.offset(ssm_bytes - 16);
        let out_tail = base.attn_output.0.offset(rows * 2560 * 2 - 16);
        let xq_tail = base
            .expert_up_out
            .0
            .offset(rows * SHARED_W4A8_XQ_BYTES_PER_ROW - 16);

        // Pairwise aliasing among the three shared-stream buffers must fail closed.
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                attn_output: (ssm_tail, base.attn_output.1),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                expert_up_out: (ssm_tail, base.expert_up_out.1),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                expert_up_out: (out_tail, base.expert_up_out.1),
                ..base
            },
            rows,
            ssm_bytes
        ));

        // Aliasing any routed or combine arena against `ssm_deinterleaved`, `attn_output`,
        // or read-only `expert_up_out` (`xq`/`xs`) must fail closed.
        for (label, target_ptr) in [
            ("ssm_deinterleaved", ssm_tail),
            ("attn_output", out_tail),
            ("expert_up_out", xq_tail),
        ] {
            let aliased = [
                (
                    "input",
                    SharedOverlapArenas {
                        input: (target_ptr, base.input.1),
                        ..base
                    },
                ),
                (
                    "gate_logits",
                    SharedOverlapArenas {
                        gate_logits: (target_ptr, base.gate_logits.1),
                        ..base
                    },
                ),
                (
                    "scratch",
                    SharedOverlapArenas {
                        scratch: (target_ptr, base.scratch.1),
                        ..base
                    },
                ),
                (
                    "expert_gate_out",
                    SharedOverlapArenas {
                        expert_gate_out: (target_ptr, base.expert_gate_out.1),
                        ..base
                    },
                ),
                (
                    "expert_down_out",
                    SharedOverlapArenas {
                        expert_down_out: (target_ptr, base.expert_down_out.1),
                        ..base
                    },
                ),
                (
                    "moe_worklist",
                    SharedOverlapArenas {
                        moe_worklist: (target_ptr, base.moe_worklist.1),
                        ..base
                    },
                ),
                (
                    "moe_worklist_total",
                    SharedOverlapArenas {
                        moe_worklist_total: (target_ptr, base.moe_worklist_total.1),
                        ..base
                    },
                ),
                (
                    "moe_gate_scalars",
                    SharedOverlapArenas {
                        moe_gate_scalars: Some((target_ptr, rows * 4)),
                        ..base
                    },
                ),
                (
                    "moe_output",
                    SharedOverlapArenas {
                        moe_output: Some((target_ptr, rows * 2560 * 2)),
                        ..base
                    },
                ),
            ];
            for (peer_name, candidate) in aliased {
                assert!(
                    !shared_overlap_regions_disjoint(candidate, rows, ssm_bytes),
                    "alias between {label} and {peer_name} must be rejected"
                );
            }
        }

        // Undersized, null, misaligned, or zero-row arenas must fail closed.
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                ssm_deinterleaved: (base.ssm_deinterleaved.0, ssm_bytes - 1),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                attn_output: (base.attn_output.0, rows * 2560 * 2 - 1),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                expert_up_out: (
                    base.expert_up_out.0,
                    rows * SHARED_W4A8_XQ_BYTES_PER_ROW - 1
                ),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                ssm_deinterleaved: (DevicePtr::NULL, ssm_bytes),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(
            SharedOverlapArenas {
                expert_up_out: (base.expert_up_out.0.offset(8), base.expert_up_out.1),
                ..base
            },
            rows,
            ssm_bytes
        ));
        assert!(!shared_overlap_regions_disjoint(base, 0, ssm_bytes));
        assert!(!shared_overlap_regions_disjoint(base, rows, 0));
    }

    #[test]
    fn shared_overlap_join_guard_defers_after_routed_and_joins_on_error_and_unwind() {
        let gpu = EventRecordingGpu::new();
        let main_stream = 0x101u64;
        let prefill_stream = 0x102u64;
        let event_b = 0x202u64;

        // 1. Wave-17 full overlap: `validate_context` (`join_pending_shared_overlap`) defers
        //    `AfterRouted` across routed GEMMs, and `guard.join()` joins once after routed ends.
        assert!(!full_overlap_guard_active());
        {
            let guard = SharedOverlapJoinGuard::enter(&gpu);
            assert!(full_overlap_guard_active());
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::AfterRouted,
            );
            join_pending_shared_overlap(&gpu).expect("deferred validate_context check");
            assert!(
                gpu.take_waits().is_empty(),
                "AfterRouted overlap must not join early in validate_context"
            );
            guard.join().expect("post-routed explicit join");
        }
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert!(gpu.take_syncs().is_empty());
        assert!(!full_overlap_guard_active());

        // 2. Wave-16 fallback (`BeforeRouted`): `validate_context` joins immediately before routed
        //    GEMMs, and the later `guard.join()` is a no-op with no duplicate wait.
        {
            let guard = SharedOverlapJoinGuard::enter(&gpu);
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::BeforeRouted,
            );
            join_pending_shared_overlap(&gpu).expect("pre-routed validate_context join");
            assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
            guard.join().expect("post-routed no-op join");
        }
        assert!(gpu.take_waits().is_empty());
        assert!(gpu.take_syncs().is_empty());
        assert!(!full_overlap_guard_active());

        // 3. Early error return before `guard.join()` (e.g. router/topk/sort failure): RAII Drop
        //    must unconditionally join `event_b` back onto `main_stream` before returning `Err`.
        let early_err = || -> Result<()> {
            let _guard = SharedOverlapJoinGuard::enter(&gpu);
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::AfterRouted,
            );
            anyhow::bail!("simulated pre-routed error");
        };
        assert!(early_err().is_err());
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert!(gpu.take_syncs().is_empty());
        assert!(!full_overlap_guard_active());

        // 4. Error returned by `run_routed_grouped_gemm`: explicit `guard.join()` runs before
        //    propagating the routed error, joining `event_b` and preserving the root cause.
        let routed_err = || -> Result<()> {
            let guard = SharedOverlapJoinGuard::enter(&gpu);
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::AfterRouted,
            );
            join_pending_shared_overlap(&gpu)?;
            let routed: Result<()> = Err(anyhow::anyhow!("simulated routed GEMM failure"));
            let join = guard.join();
            routed?;
            join?;
            Ok(())
        };
        let err = routed_err().expect_err("routed failure must propagate");
        assert!(err.to_string().contains("simulated routed GEMM failure"));
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert!(gpu.take_syncs().is_empty());
        assert!(!full_overlap_guard_active());

        // 5. Panic unwind while side-stream work is in flight: RAII Drop still joins cleanly.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = SharedOverlapJoinGuard::enter(&gpu);
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::AfterRouted,
            );
            panic!("simulated unwind with in-flight shared_w4a8");
        }));
        assert!(panicked.is_err());
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert!(gpu.take_syncs().is_empty());
        assert!(!full_overlap_guard_active());
        assert_eq!(PENDING_SHARED_OVERLAP_JOIN.with(Cell::get), None);

        // 6. Backend error during explicit `join()` propagates, syncs fallback stream, and clears TLS.
        gpu.set_fail_wait(true);
        let guard = SharedOverlapJoinGuard::enter(&gpu);
        record_pending_shared_overlap_join(
            main_stream,
            prefill_stream,
            event_b,
            SharedOverlapJoinPhase::AfterRouted,
        );
        assert!(guard.join().is_err());
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert_eq!(gpu.take_syncs(), vec![prefill_stream]);
        assert!(!full_overlap_guard_active());
        assert_eq!(PENDING_SHARED_OVERLAP_JOIN.with(Cell::get), None);

        // 7. Backend error during Drop falls back to synchronize(prefill_stream) without panicking.
        {
            let _guard = SharedOverlapJoinGuard::enter(&gpu);
            record_pending_shared_overlap_join(
                main_stream,
                prefill_stream,
                event_b,
                SharedOverlapJoinPhase::AfterRouted,
            );
        }
        assert_eq!(gpu.take_waits(), vec![(main_stream, event_b)]);
        assert_eq!(gpu.take_syncs(), vec![prefill_stream]);
        assert!(!full_overlap_guard_active());
        assert_eq!(PENDING_SHARED_OVERLAP_JOIN.with(Cell::get), None);
        gpu.set_fail_wait(false);
    }

    #[test]
    fn resolve_pretranscode_use_checks_ssm_deinterleaved_capacity() {
        let undersized = PRETRANSCODE_SCRATCH_BYTES - 1;
        // Off or non-v28 never enables pretranscode.
        assert!(
            !resolve_pretranscode_use(
                PretranscodeMode::Off,
                512,
                "v28",
                PRETRANSCODE_SCRATCH_BYTES
            )
            .unwrap()
        );
        assert!(
            !resolve_pretranscode_use(PretranscodeMode::Force, 512, "v16", undersized).unwrap()
        );
        // Auto falls back cleanly to direct W4A8 when ssm_deinterleaved is undersized.
        assert!(!resolve_pretranscode_use(PretranscodeMode::Auto, 512, "v28", undersized).unwrap());
        assert!(
            resolve_pretranscode_use(
                PretranscodeMode::Auto,
                512,
                "v28",
                PRETRANSCODE_SCRATCH_BYTES
            )
            .unwrap()
        );
        assert!(
            !resolve_pretranscode_use(
                PretranscodeMode::Auto,
                256,
                "v28",
                PRETRANSCODE_SCRATCH_BYTES
            )
            .unwrap()
        );
        // Force fails closed when ssm_deinterleaved is undersized instead of overflowing GPU scratch.
        assert!(resolve_pretranscode_use(PretranscodeMode::Force, 128, "v28", undersized).is_err());
        assert!(
            resolve_pretranscode_use(
                PretranscodeMode::Force,
                128,
                "v28",
                PRETRANSCODE_SCRATCH_BYTES
            )
            .unwrap()
        );
    }

    #[test]
    fn prefill_shared_sort_overlap_selector_defaults_off_with_opt_in() {
        for disabled in [
            None,
            Some(""),
            Some("0"),
            Some(" 0 "),
            Some("off"),
            Some("OFF"),
            Some("false"),
            Some("FALSE"),
        ] {
            assert!(
                !parse_prefill_shared_sort_overlap(disabled),
                "expected sort-overlap disabled for {disabled:?}"
            );
        }
        for enabled in [
            Some("1"),
            Some(" 1 "),
            Some("on"),
            Some("ON"),
            Some("true"),
            Some("TRUE"),
        ] {
            assert!(
                parse_prefill_shared_sort_overlap(enabled),
                "expected sort-overlap enabled for {enabled:?}"
            );
        }
    }

    #[test]
    fn pretranscode_gateup_dispatch_uses_2cta_2stage_under_48kib_with_4stage_fallback() {
        assert_eq!(V28_SHARED_GATEUP_2STAGE_SMEM, 34_304);
        const {
            assert!(V28_SHARED_GATEUP_2STAGE_SMEM <= 49_152);
        }
        assert_eq!(V28_SHARED_GATEUP_2STAGE_BLOCK, 256);

        for (rows, items) in [(384usize, 3u32), (2048, 16), (4096, 32), (16_384, 128)] {
            assert_eq!(
                select_pretranscode_gateup_dispatch(rows, true, 512),
                PretranscodeGateUpDispatch {
                    use_2stage: true,
                    grid_x: items * 10,
                    block_x: 256,
                    smem_bytes: 34_304,
                },
                "2-stage gateup dispatch mismatch for rows={rows}"
            );
            assert_eq!(
                select_pretranscode_gateup_dispatch(rows, false, 512),
                PretranscodeGateUpDispatch {
                    use_2stage: false,
                    grid_x: items * 5,
                    block_x: 512,
                    smem_bytes: V28_SHARED_SMEM,
                },
                "4-stage gateup fallback mismatch for rows={rows}"
            );
        }
    }

    #[test]
    fn sort_overlap_safe_enforces_every_precondition() {
        let all_ok = SharedSortOverlapChecks {
            sort_overlap_enabled: true,
            use_pretranscode: true,
            has_gateup_2stage: true,
            has_down_discard: true,
            common_overlap_eligible: true,
            guard_active: true,
            routed_w4a8_active: true,
            xq_reuse_active: true,
            input_is_norm_output: true,
            oracle_disabled: true,
            arenas_disjoint: true,
        };
        assert!(sort_overlap_safe(all_ok));

        let cases = [
            (
                "sort_overlap_enabled",
                SharedSortOverlapChecks {
                    sort_overlap_enabled: false,
                    ..all_ok
                },
            ),
            (
                "use_pretranscode",
                SharedSortOverlapChecks {
                    use_pretranscode: false,
                    ..all_ok
                },
            ),
            (
                "has_gateup_2stage",
                SharedSortOverlapChecks {
                    has_gateup_2stage: false,
                    ..all_ok
                },
            ),
            (
                "has_down_discard",
                SharedSortOverlapChecks {
                    has_down_discard: false,
                    ..all_ok
                },
            ),
            (
                "common_overlap_eligible",
                SharedSortOverlapChecks {
                    common_overlap_eligible: false,
                    ..all_ok
                },
            ),
            (
                "guard_active",
                SharedSortOverlapChecks {
                    guard_active: false,
                    ..all_ok
                },
            ),
            (
                "routed_w4a8_active",
                SharedSortOverlapChecks {
                    routed_w4a8_active: false,
                    ..all_ok
                },
            ),
            (
                "xq_reuse_active",
                SharedSortOverlapChecks {
                    xq_reuse_active: false,
                    ..all_ok
                },
            ),
            (
                "input_is_norm_output",
                SharedSortOverlapChecks {
                    input_is_norm_output: false,
                    ..all_ok
                },
            ),
            (
                "oracle_disabled",
                SharedSortOverlapChecks {
                    oracle_disabled: false,
                    ..all_ok
                },
            ),
            (
                "arenas_disjoint",
                SharedSortOverlapChecks {
                    arenas_disjoint: false,
                    ..all_ok
                },
            ),
        ];
        for (name, checks) in cases {
            assert!(
                !sort_overlap_safe(checks),
                "sort_overlap_safe must fail closed when {name}=false"
            );
        }
    }

    #[test]
    fn pretranscode_grid_scales_to_single_wave_on_high_sm_and_preserves_gb10() {
        for sm in [0u32, 1, 48, 64] {
            assert_eq!(
                pretranscode_grid_for_sm(sm),
                100,
                "sm={sm} must keep GB10 grid=100"
            );
        }
        for sm in [65u32, 110, 148, 188, 256] {
            assert_eq!(
                pretranscode_grid_for_sm(sm),
                300,
                "sm={sm} must run all 300 pretranscode slices in one wave"
            );
        }
    }

    #[test]
    fn sm120_auto_enables_shared_overlap_and_pretranscode_while_preserving_gb10() {
        // GB10 (<= 64 SMs): all four selectors default OFF unless explicitly opted in.
        for sm in [0u32, 48, 64] {
            assert_eq!(
                PretranscodeMode::parse_for_sm(None, sm),
                PretranscodeMode::Off
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some(""), sm),
                PretranscodeMode::Off
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("1"), sm),
                PretranscodeMode::Auto
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("force"), sm),
                PretranscodeMode::Force
            );
            assert!(!resolve_prefill_shared_overlap(None, sm));
            assert!(resolve_prefill_shared_overlap(Some("1"), sm));
            assert!(!resolve_prefill_shared_full_overlap(None, sm));
            assert!(resolve_prefill_shared_full_overlap(Some("1"), sm));
            assert!(!resolve_prefill_shared_sort_overlap(None, sm));
            assert!(resolve_prefill_shared_sort_overlap(Some("1"), sm));
        }
        // SM120 / SM103 (> 64 SMs): all four auto-enable when unset and honor explicit opt-outs.
        for sm in [65u32, 110, 170, 188] {
            assert_eq!(
                PretranscodeMode::parse_for_sm(None, sm),
                PretranscodeMode::Auto
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some(""), sm),
                PretranscodeMode::Auto
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("0"), sm),
                PretranscodeMode::Off
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("off"), sm),
                PretranscodeMode::Off
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("false"), sm),
                PretranscodeMode::Off
            );
            assert_eq!(
                PretranscodeMode::parse_for_sm(Some("force"), sm),
                PretranscodeMode::Force
            );

            assert!(resolve_prefill_shared_overlap(None, sm));
            assert!(!resolve_prefill_shared_overlap(Some("0"), sm));
            assert!(!resolve_prefill_shared_overlap(Some("off"), sm));
            assert!(!resolve_prefill_shared_overlap(Some("FALSE"), sm));

            assert!(resolve_prefill_shared_full_overlap(None, sm));
            assert!(!resolve_prefill_shared_full_overlap(Some("0"), sm));
            assert!(!resolve_prefill_shared_full_overlap(Some("off"), sm));
            assert!(!resolve_prefill_shared_full_overlap(Some("false"), sm));

            assert!(resolve_prefill_shared_sort_overlap(None, sm));
            assert!(!resolve_prefill_shared_sort_overlap(Some("0"), sm));
            assert!(!resolve_prefill_shared_sort_overlap(Some("off"), sm));
            assert!(!resolve_prefill_shared_sort_overlap(Some("false"), sm));
        }
    }

    #[test]
    fn shared_overlap_plan_prefers_full_after_routed_then_sort_overlap_then_pre_routed() {
        assert_eq!(
            select_shared_overlap_plan(true, true, true),
            SharedOverlapPlan::FullAfterRouted
        );
        assert_eq!(
            select_shared_overlap_plan(false, true, true),
            SharedOverlapPlan::PostRouterSortBeforeRouted
        );
        assert_eq!(
            select_shared_overlap_plan(false, true, false),
            SharedOverlapPlan::PostRouterSortBeforeRouted
        );
        assert_eq!(
            select_shared_overlap_plan(false, false, true),
            SharedOverlapPlan::PreRouterBeforeRouted
        );
        assert_eq!(
            select_shared_overlap_plan(false, false, false),
            SharedOverlapPlan::Serial
        );
    }

    #[test]
    fn deterministic_batch_pins_shared_v7_serial_and_m64_tile_across_1_to_128() {
        use super::super::qwen4_compact_ops::V28OpTile;
        use super::{
            select_shared_overlap_plan_with_policy, select_shared_v28_launch_plan_with_policy,
        };
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        assert_eq!(
            select_shared_overlap_plan_with_policy(
                true,
                true,
                true,
                DeterministicBatchPolicy::Enabled,
            ),
            SharedOverlapPlan::Serial
        );
        for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            let plan = select_shared_v28_launch_plan_with_policy(
                m,
                true,
                DeterministicBatchPolicy::Enabled,
            );
            assert_eq!(
                plan.tile,
                V28OpTile::M64,
                "shared v28 tile must stay M-invariant at M={m}"
            );
        }
        assert_eq!(
            select_shared_v28_launch_plan_with_policy(8, true, DeterministicBatchPolicy::Disabled)
                .tile,
            V28OpTile::M32
        );
    }
}
