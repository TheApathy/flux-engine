// SPDX-License-Identifier: AGPL-3.0-only

//! Exact five-row NVFP4 MoE path for Qwen4 speculative verification.

use super::*;

const NATIVE_TARGET_MOE_MAX_TILE: usize = 256;
const WORKLIST_MAX_SLOTS: usize = 2560;
/// `moe_build_unique_worklist_e512` in `moe_shared_expert_fused_t.cu` stages
/// flattened slots in `__shared__ unsigned int s_idx[1280]`, and the worklist
/// GEMM kernels (`gate_up_worklist_t_lanes_rows_staged_body`,
/// `down_act_t_lanes_rows_t8_body`) guard `if (rows == 0u || rows > 128u) return;`,
/// so single-launch scalar worklist compaction is capped at `128` rows and
/// `1280` slots (`128` rows x `top_k=10`).
/// Wider frames (`129..=256`) run in a single pass on compact W4A8 TC MoE, or
/// split into two balanced `<= 128`-row halves when TC MoE is disabled.
const WORKLIST_E512_KERNEL_MAX_ROWS: u32 = 128;
const WORKLIST_E512_KERNEL_MAX_SLOTS: u32 = 1280;
/// Byte capacity of the shared worklist staging buffer:
/// `WORKLIST_MAX_SLOTS * (4 + 4 + 10 * 4) = 122,880` B (120 KiB, 4 KiB-aligned).
const WORKLIST_BUFFER_BYTES: usize =
    (WORKLIST_MAX_SLOTS * 12 * std::mem::size_of::<u32>()).next_multiple_of(4096);
/// Shared-expert V28 pre-transcode scratch footprint (`qwen4_shared_v7::PRETRANSCODE_SCRATCH_BYTES`).
const SHARED_W4A8_PRETRANSCODE_SCRATCH_BYTES: usize = 5_068_800;
/// `ATLAS_QWEN4_MOE_SHARED_OVERLAP`: overlap the shared expert on `MoeLayer::prefill_stream`
/// with the routed compact W4A8 path during batched verify. Defaults to OFF on `<= 64`-SM
/// GPUs (GB10, where 40..160 CTAs collide on SMEM/L2 with `v28_m32_ts4e` and add 47
/// fork/join graph branches) and ON on `> 64`-SM GPUs (SM120) when `shared_fast_v2` is
/// active, and ON for legacy `shared_w4a8`. Set `=1` to force overlap on both paths or
/// `=0` to disable overlap on both paths.
const MOE_SHARED_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_MOE_SHARED_OVERLAP";
/// `ATLAS_QWEN4_MOE_SHARED_GRAPH_OVERLAP=1` (default ON unless `=0`): allow fork/join
/// `prefill_stream` shared-expert overlap to record parallel DAG branches inside CUDA graph
/// capture when `ATLAS_QWEN4_MOE_SHARED_OVERLAP` is also enabled.
const MOE_SHARED_GRAPH_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_MOE_SHARED_GRAPH_OVERLAP";
/// `ATLAS_QWEN4_DECODE_SHARED_FAST_V2=1` (default ON unless `=0`): route the 1 shared expert
/// during batched decode (`rows in 2..=256`) onto high-CTA transposed NVFP4 W4A16 kernels
/// (`w4a16_shared_gateup_silu_t_n16_*` + `w4a16_shared_down_t_n32_*`, 40..160 CTAs) instead of
/// 5..20-CTA `v28_m32` W4A8 (`qwen4_shared_v7`).
const DECODE_SHARED_FAST_V2_SELECTOR: &str = "ATLAS_QWEN4_DECODE_SHARED_FAST_V2";
/// Cap deferred batched-verify layers per flush so `qwen4_moe_status`'s 4096-slot table
/// (2 slots/layer) never auto-flushes mid-layer while the side stream is in flight.
const MAX_DEFERRED_VERIFY_LAYERS_PER_FLUSH: usize = 2000;

static BATCHED_VERIFY_TC_MOE_RECEIPT_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Active while inside `run_qwen4_batched_verify_tc_moe_inner` with async MoE status enabled,
    /// admitting `qwen4_moe_status` deferred constants/slots without per-layer stream drains.
    static BATCHED_VERIFY_TC_MOE_STATUS_ASYNC_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Active when batched verify pre-quantized `(xq, xs)` into `expert_up_out` before forking
    /// the shared expert onto `prefill_stream`, requiring the routed W4A8 path to reuse `(xq, xs)`
    /// read-only instead of overwriting `expert_up_out` concurrently.
    static BATCHED_VERIFY_TC_MOE_XQ_PREQUANT_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Active when batched verify will consume precomputed `ss_ptr` gate scalars in the fused
    /// `moe_unpermute_reduce_blend_mxfp8_bf16` epilogue, requesting `v16q4_gate` during XQ quant.
    static BATCHED_VERIFY_TC_MOE_FUSED_BLEND_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Number of batched-verify TC MoE layer dispatches queued since the last status flush.
    static BATCHED_VERIFY_STATUS_PENDING_LAYERS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Layer identity key (`gate_ptrs.packed_ptrs.0`) of the first queued layer in the current
    /// round, used to flush early if a new round starts after an incomplete prior round.
    static BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

pub(super) fn batched_verify_tc_moe_status_async_active() -> bool {
    BATCHED_VERIFY_TC_MOE_STATUS_ASYNC_ACTIVE.with(std::cell::Cell::get)
}

pub(super) fn batched_verify_tc_moe_xq_prequant_active() -> bool {
    BATCHED_VERIFY_TC_MOE_XQ_PREQUANT_ACTIVE.with(std::cell::Cell::get)
}

pub(super) fn batched_verify_tc_moe_fused_blend_active() -> bool {
    BATCHED_VERIFY_TC_MOE_FUSED_BLEND_ACTIVE.with(std::cell::Cell::get)
}

fn reset_batched_verify_status_tracking() {
    BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(|c| c.set(0));
    BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY.with(|c| c.set(0));
}

#[must_use]
const fn should_flush_batched_verify_status(
    pending_layers: usize,
    num_hidden_layers: usize,
) -> bool {
    let target = if num_hidden_layers == 0 {
        1
    } else if num_hidden_layers > MAX_DEFERRED_VERIFY_LAYERS_PER_FLUSH {
        MAX_DEFERRED_VERIFY_LAYERS_PER_FLUSH
    } else {
        num_hidden_layers
    };
    pending_layers >= target
}

/// Drain and verify all deferred compact-MoE status words queued by batched verify.
/// No-op during CUDA graph capture (zero synchronous D2H copies while recording).
pub(crate) fn flush_batched_verify_moe_status(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    if super::qwen4_moe_status::is_capturing(gpu, stream) {
        return Ok(());
    }
    reset_batched_verify_status_tracking();
    super::qwen4_moe_status::flush(gpu, stream)
}

struct BatchedVerifyTcMoeScopeGuard {
    prev_status_async: bool,
    prev_xq_prequant: bool,
    prev_fused_blend: bool,
}

impl BatchedVerifyTcMoeScopeGuard {
    fn enter(status_async: bool, xq_prequant: bool, fused_blend: bool) -> Self {
        let prev_status_async =
            BATCHED_VERIFY_TC_MOE_STATUS_ASYNC_ACTIVE.with(|c| c.replace(status_async));
        let prev_xq_prequant =
            BATCHED_VERIFY_TC_MOE_XQ_PREQUANT_ACTIVE.with(|c| c.replace(xq_prequant));
        let prev_fused_blend =
            BATCHED_VERIFY_TC_MOE_FUSED_BLEND_ACTIVE.with(|c| c.replace(fused_blend));
        Self {
            prev_status_async,
            prev_xq_prequant,
            prev_fused_blend,
        }
    }
}

impl Drop for BatchedVerifyTcMoeScopeGuard {
    fn drop(&mut self) {
        BATCHED_VERIFY_TC_MOE_STATUS_ASYNC_ACTIVE.with(|c| c.set(self.prev_status_async));
        BATCHED_VERIFY_TC_MOE_XQ_PREQUANT_ACTIVE.with(|c| c.set(self.prev_xq_prequant));
        BATCHED_VERIFY_TC_MOE_FUSED_BLEND_ACTIVE.with(|c| c.set(self.prev_fused_blend));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedOverlapOverride {
    Auto,
    ForceOn,
    ForceOff,
}

#[must_use]
fn parse_moe_shared_overlap_override(raw: Option<&str>) -> SharedOverlapOverride {
    match raw.map(str::trim) {
        Some("1") => SharedOverlapOverride::ForceOn,
        Some("0") => SharedOverlapOverride::ForceOff,
        _ => SharedOverlapOverride::Auto,
    }
}

/// SM count threshold above which `Auto` overlaps `shared_fast_v2` onto
/// `prefill_stream` concurrently with routed MoE GEMMs (GB10 `<= 64` SMs stays
/// serialized when `shared_fast_v2` is active to avoid SMEM/L2 contention).
const MOE_SHARED_OVERLAP_HIGH_SM_THRESHOLD: u32 = 64;

/// Opt-in override (`ATLAS_QWEN4_SHARED_OVERLAP_THIN_SK=1`) to keep cross-stream
/// shared-expert overlap active while `v33_thin_sk` runs at `rows <= 16`.
///
/// Default is OFF because `v33_thin_sk` already launches `1,128` resident CTAs
/// across all 188 SM120 SMs, so serializing `shared_fast_v2` avoids 48 cross-stream
/// graph fork/joins fighting `v33_thin_sk` for SM occupancy and L2 capacity.
const MOE_SHARED_OVERLAP_THIN_SK_SELECTOR: &str = "ATLAS_QWEN4_SHARED_OVERLAP_THIN_SK";
const MOE_SHARED_OVERLAP_THIN_SK_MAX_ROWS: usize = 16;

#[must_use]
fn parse_moe_shared_overlap_thin_sk(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true")
    )
}

#[must_use]
fn moe_shared_overlap_thin_sk_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_moe_shared_overlap_thin_sk(
            std::env::var(MOE_SHARED_OVERLAP_THIN_SK_SELECTOR)
                .ok()
                .as_deref(),
        )
    })
}

#[must_use]
const fn moe_shared_overlap_enabled_for_path_and_sm(
    override_mode: SharedOverlapOverride,
    shared_fast_v2: bool,
    sm_count: u32,
) -> bool {
    match override_mode {
        SharedOverlapOverride::ForceOn => true,
        SharedOverlapOverride::ForceOff => false,
        SharedOverlapOverride::Auto => {
            !shared_fast_v2 || sm_count > MOE_SHARED_OVERLAP_HIGH_SM_THRESHOLD
        }
    }
}

#[must_use]
const fn moe_shared_overlap_enabled_for_thin_sk_and_sm(
    override_mode: SharedOverlapOverride,
    shared_fast_v2: bool,
    sm_count: u32,
    thin_sk_active: bool,
    rows: usize,
    thin_sk_overlap_override: bool,
) -> bool {
    if thin_sk_active
        && rows <= MOE_SHARED_OVERLAP_THIN_SK_MAX_ROWS
        && !thin_sk_overlap_override
        && !matches!(override_mode, SharedOverlapOverride::ForceOn)
    {
        return false;
    }
    moe_shared_overlap_enabled_for_path_and_sm(override_mode, shared_fast_v2, sm_count)
}

#[cfg(test)]
#[must_use]
fn moe_shared_overlap_enabled_for_path(
    override_mode: SharedOverlapOverride,
    shared_fast_v2: bool,
) -> bool {
    moe_shared_overlap_enabled_for_path_and_sm(
        override_mode,
        shared_fast_v2,
        atlas_core::device::active_sm_count(),
    )
}

fn moe_shared_overlap_override() -> SharedOverlapOverride {
    static OVERRIDE: std::sync::OnceLock<SharedOverlapOverride> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        parse_moe_shared_overlap_override(
            std::env::var(MOE_SHARED_OVERLAP_SELECTOR).ok().as_deref(),
        )
    })
}

fn moe_shared_graph_overlap_enabled(raw: Option<&str>) -> bool {
    raw != Some("0")
}

fn moe_shared_graph_overlap_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        moe_shared_graph_overlap_enabled(
            std::env::var(MOE_SHARED_GRAPH_OVERLAP_SELECTOR)
                .ok()
                .as_deref(),
        )
    })
}

fn decode_shared_fast_v2_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

fn decode_shared_fast_v2_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        decode_shared_fast_v2_enabled(
            std::env::var(DECODE_SHARED_FAST_V2_SELECTOR)
                .ok()
                .as_deref(),
        )
    })
}

#[must_use]
const fn moe_shared_overlap_admits_capture(
    overlap_enabled: bool,
    graph_overlap_enabled: bool,
    capturing: bool,
) -> bool {
    overlap_enabled && (!capturing || graph_overlap_enabled)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedVerifyTcMoePlan {
    use_ts4e: bool,
    use_m64: bool,
    use_m32_gateup: bool,
    use_m32_down: bool,
    shared_w4a8: bool,
    shared_fast_v2: bool,
    shared_overlap: bool,
    gatepre: bool,
    fused_unpermute_blend: bool,
}

/// Admit the single-sweep K-coalesced NVFP4 Tensor Core MoE router
/// (`w4a16_gemm_coalesced_n16_m*` for `num_experts <= 1024`) inside
/// batched-verify TC dense scopes (`rows in 4..=256`, `hidden % 32 == 0`).
///
/// The hard floor of `DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS` (`4`) paired with
/// `scope_active` (strictly `false` for `C = 1`) guarantees all `C=1` `0-ULP`
/// parity probes and `rows <= 3` stay on the exact bit-identical
/// `w4a16_gemv_batch_logits_exact_with` scalar path while enabling `rows >= 4`
/// (`C=2 r=4..8`, `C=2 K=5`, and `C=4..16`) to hit Warp-Split-K
/// `n8_sk8_m16`/`n32_sk4_m16`.
/// Coalesced NVFP4 Tensor Core routing reads `self.gate_nvfp4` in one kernel launch
/// with zero extra VRAM and zero lazy allocations, making it 100% CUDA-graph safe.
#[inline]
#[must_use]
const fn batched_verify_router_tc_admits_with_policy(
    scope_active: bool,
    tc_dense_enabled: bool,
    min_rows_override: Option<usize>,
    rows: usize,
    num_experts: u32,
    hidden: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    batched_verify_router_tc_admits(
        scope_active,
        tc_dense_enabled,
        min_rows_override,
        rows,
        num_experts,
        hidden,
    )
}

#[inline]
#[must_use]
const fn batched_verify_router_tc_admits(
    scope_active: bool,
    tc_dense_enabled: bool,
    min_rows_override: Option<usize>,
    rows: usize,
    num_experts: u32,
    hidden: u32,
) -> bool {
    let floor = crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS;
    let min_rows = match min_rows_override {
        Some(n) => {
            if n > floor {
                n
            } else {
                floor
            }
        }
        None => floor,
    };
    scope_active
        && tc_dense_enabled
        && rows >= min_rows
        && rows <= NATIVE_TARGET_MOE_MAX_TILE
        && num_experts > 0
        && hidden > 0
        && hidden.is_multiple_of(32)
}

/// Byte offset of `token_to_perm` inside `gate_logits`, matching `forward_prefill`
/// and `qwen4_compact_ops::run_qwen4_compact` (`launch_fused_sort_plan`).
#[must_use]
fn compact_token_to_perm_offset(rows: usize, gatepre: bool) -> usize {
    let expanded = rows * 10;
    let base_perm_off = expanded * 8 + 513 * 4;
    let gatepre_perm_off = base_perm_off.max(rows * 640);
    if gatepre && gatepre_perm_off + expanded * 4 <= rows * 512 * 2 {
        gatepre_perm_off
    } else {
        base_perm_off
    }
}

/// Verify that `[ptr, ptr + bytes)` (16-byte aligned) is strictly disjoint from all `peers`.
fn verify_tc_moe_region_disjoint(
    ptr: DevicePtr,
    bytes: usize,
    peers: &[(DevicePtr, usize)],
) -> bool {
    let Ok(region) = super::qwen4_compact_contract::region(ptr.0, bytes, 16) else {
        return false;
    };
    peers.iter().all(|&(other_ptr, other_bytes)| {
        super::qwen4_compact_contract::region(other_ptr.0, other_bytes, 4)
            .is_ok_and(|other| super::qwen4_compact_contract::disjoint(region, other))
    })
}

/// Verify that `output` (`[rows, 2560]` BF16) is well-aligned and strictly disjoint
/// from every arena buffer touched by the compact W4A8 + shared-expert pipeline.
fn verify_tc_moe_output_disjoint(
    output: DevicePtr,
    rows: usize,
    peers: &[(DevicePtr, usize)],
) -> bool {
    verify_tc_moe_region_disjoint(output, rows * 2560 * 2, peers)
}

/// Tight upper bound on distinct worklist groups across `num_experts` experts
/// when each expert's slots are chunked into groups of at most `rmax`.
const fn worklist_routed_ctas(rows: u32, top_k: u32, num_experts: u32, rmax: u32) -> u32 {
    let total = rows * top_k;
    let bound = num_experts + total / rmax;
    if total < bound { total } else { bound }
}

fn moe_build_unique_worklist_e512_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static K: std::sync::OnceLock<KernelHandle> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        gpu.kernel(
            "moe_shared_expert_fused_t",
            "moe_build_unique_worklist_e512",
        )
        .unwrap_or(KernelHandle(0))
    })
}

/// Effective row count fed into `[r3, r5, r8, r10]` worklist tier selection.
///
/// Selects `tier=1` (`r5`) for `rows <= 10` (unless `ATLAS_MOE_T_LANES_WORKLIST_R8=1`
/// explicitly requests `r8`), giving `rows = 10` (`C=2 K=5`) the same 2 shared CTAs
/// (`ceil(10/5) == ceil(10/8) == 2`) with higher SM occupancy on both gate/up and down_act.
#[must_use]
const fn worklist_tier_rows(rows: u32, force_r5: bool, force_r8: bool) -> u32 {
    let cap = if force_r5 || (rows <= 10 && !force_r8) {
        5
    } else {
        8
    };
    if rows < cap { rows } else { cap }
}

/// Dispatch mode chosen for worklist gate/up (`t_lanes_rows` fused `grid_z=1`, `OPT=4`, PDL
/// vs fallback `t8` unfused `grid_z=2`, `OPT=2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorklistGateUpMode {
    FusedTLanesRows,
    UnfusedT8,
}

#[must_use]
const fn select_worklist_gate_up_mode(
    prefer_t8: bool,
    has_fused: bool,
    has_t8: bool,
) -> Option<WorklistGateUpMode> {
    if prefer_t8 && has_t8 {
        Some(WorklistGateUpMode::UnfusedT8)
    } else if has_fused {
        Some(WorklistGateUpMode::FusedTLanesRows)
    } else if has_t8 {
        Some(WorklistGateUpMode::UnfusedT8)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct Ts4eWorklistKernels {
    gu_wl_r2: KernelHandle,
    gu_wl_r4: KernelHandle,
    gu_wl_tiers: [KernelHandle; 4],
    gu_t8_tiers: [KernelHandle; 4],
    dn_tiers: [KernelHandle; 4],
}

fn ts4e_worklist_kernels(gpu: &dyn GpuBackend) -> Ts4eWorklistKernels {
    static K: std::sync::OnceLock<Ts4eWorklistKernels> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        let module = "moe_shared_expert_fused_t";
        let load = |name: &str| gpu.kernel(module, name).unwrap_or(KernelHandle(0));
        Ts4eWorklistKernels {
            gu_wl_r2: load("moe_expert_gate_up_worklist_t_lanes_rows_r2_ts4e"),
            gu_wl_r4: load("moe_expert_gate_up_worklist_t_lanes_rows_r4_ts4e"),
            gu_wl_tiers: [
                load("moe_expert_gate_up_worklist_t_lanes_rows_r3_ts4e"),
                load("moe_expert_gate_up_worklist_t_lanes_rows_r5_ts4e"),
                load("moe_expert_gate_up_worklist_t_lanes_rows_r8_ts4e"),
                load("moe_expert_gate_up_worklist_t_lanes_rows_r10_ts4e"),
            ],
            gu_t8_tiers: [
                load("moe_expert_gate_up_worklist_t8_r3_k160_ts4e"),
                load("moe_expert_gate_up_worklist_t8_r5_k160_ts4e"),
                load("moe_expert_gate_up_worklist_t8_r8_k160_ts4e"),
                KernelHandle(0),
            ],
            dn_tiers: [
                load("moe_expert_down_act_shared_t_lanes_rows_t8_r3_k40_ts4e"),
                load("moe_expert_down_act_shared_t_lanes_rows_t8_r5_k40_ts4e"),
                load("moe_expert_down_act_shared_t_lanes_rows_t8_r8_k40_ts4e"),
                load("moe_expert_down_act_shared_t_lanes_rows_t8_r10_k40_ts4e"),
            ],
        }
    })
}

/// The exact unified-layout MoE worklist kernels chunk each distinct expert
/// into groups of `<= 5` or `<= 8` token slots (`RMAX = 5 or 8`) and tile the
/// shared expert across `ceil(rows / RMAX)` CTAs. A single worklist pass covers
/// `rows <= 128` (`WORKLIST_E512_KERNEL_MAX_ROWS`); wider `C129-C256` frames run
/// in a single pass on compact W4A8 TC MoE or split into two balanced `<= 128`-row
/// halves when TC MoE is disabled.
fn native_target_moe_tile_width(remaining: usize) -> std::result::Result<usize, &'static str> {
    if !(2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&remaining) {
        return Err("native Qwen4 MoE remaining rows must be in C2-C256");
    }
    let max_tile = if moe_t_lanes_worklist_selected() {
        NATIVE_TARGET_MOE_MAX_TILE
    } else {
        9
    };
    Ok(if remaining <= max_tile {
        remaining
    } else if remaining % max_tile == 1 {
        8
    } else {
        max_tile
    })
}

/// Decompose a wide `10..=256` MoE batch into non-singleton `2..=9` sub-tiles
/// when the `r8` worklist+down_act route is unavailable, guaranteeing the
/// `T_LANES_RMAX = 9` non-worklist kernels are never called with `rows > 9`.
fn non_worklist_subtile(remaining: usize) -> usize {
    if remaining <= 9 {
        remaining
    } else if remaining == 10 {
        5
    } else {
        9.min(remaining - 2)
    }
}

impl MoeLayer {
    pub fn forward_k5(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        self.forward_qwen4_exact(input, 5, ctx, stream)
    }

    pub fn forward_qwen4_exact(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_qwen4_exact_into(input, rows, ctx.buffers.moe_output(), ctx, stream)
    }

    fn forward_qwen4_exact_into(
        &self,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            matches!(rows, 4 | 5 | 9)
                || (crate::layers::qwen4_hybrid_all_k()
                    && (2..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&rows))
                || (self.t_lanes_rows_route() && (2..=NATIVE_TARGET_MOE_MAX_TILE).contains(&rows)),
            "Qwen4 exact MoE requires 4, 5 or 9 rows (2..=9 with ATLAS_QWEN4_HYBRID_ALL_K or 2..=256 on the unified _t_lanes route)"
        );
        anyhow::ensure!(
            ctx.config.ep_world_size <= 4 && ctx.config.tp_world_size <= 4,
            "Qwen4 exact NVFP4 MoE supports TP<=4 and EP<=4"
        );
        // Under ATLAS_UNIFIED_MOE_LAYOUT the untransposed expert weights are
        // freed and the gate/up/down pointer tables below go stale; only the
        // `_t` kernels may run. This path has no `_t` variant.
        // On the unified layout the exact path runs the multi-row `_t_lanes`
        // kernels, whose rows are byte-identical to the single-row `_t_lanes`
        // decode kernels; ATLAS_MOE_T_LANES must therefore also be on, so the
        // K=1 decode these rows are verified against uses the same arithmetic.
        let t_rows = self.use_t_layout_for_decode();
        if t_rows {
            anyhow::ensure!(
                super::forward_phase::t_lanes_selected()
                    && self.moe_expert_gate_up_shared_t_lanes_rows_k.0 != 0
                    && self.moe_expert_silu_down_shared_t_lanes_rows_k.0 != 0
                    && rows <= NATIVE_TARGET_MOE_MAX_TILE,
                "Qwen4 exact K4/K5/K9 MoE on ATLAS_UNIFIED_MOE_LAYOUT needs ATLAS_MOE_T_LANES=1 \
                 and the _t_lanes_rows kernels"
            );
        }
        let nvfp4 = self
            .gate_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact MoE requires an NVFP4 router"))?;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let tc_moe_plan = if t_rows {
            self.batched_verify_tc_moe_plan(input, rows, output, ctx, stream)?
        } else {
            None
        };
        if t_rows
            && tc_moe_plan.is_none()
            && rows > 9
            && !self.qwen4_worklist_covers_wide_rows(
                ctx.gpu,
                rows as u32,
                h,
                inter,
                num_experts,
                top_k,
            )
        {
            // When TC MoE is disabled and `rows > 128` exceeds the scalar e512
            // worklist kernel's 1280-slot SMEM array, split into two balanced
            // `<= 128`-row worklist halves instead of 9-row subtiles.
            let half = rows.div_ceil(2);
            if rows > 128
                && self.qwen4_worklist_covers_wide_rows(
                    ctx.gpu,
                    half as u32,
                    h,
                    inter,
                    num_experts,
                    top_k,
                )
            {
                let rem = rows - half;
                self.forward_qwen4_exact_into(input, half, output, ctx, stream)?;
                let byte_off = half * (h as usize) * 2;
                self.forward_qwen4_exact_into(
                    input.offset(byte_off),
                    rem,
                    output.offset(byte_off),
                    ctx,
                    stream,
                )?;
                return Ok(());
            }
            let mut off = 0usize;
            while off < rows {
                let tile = non_worklist_subtile(rows - off);
                let byte_off = off * (h as usize) * 2;
                self.forward_qwen4_exact_into(
                    input.offset(byte_off),
                    tile,
                    output.offset(byte_off),
                    ctx,
                    stream,
                )?;
                off += tile;
            }
            return Ok(());
        }
        let router_in = self.router_input(input, rows as u32, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();

        // Inside batched-verify TC dense scopes (`rows in 4..=256`, `hidden % 32 == 0`),
        // route `[rows, H] x [num_experts, H]^T` onto the 1-launch K-coalesced NVFP4
        // Tensor Core GEMM (`w4a16_gemm_coalesced_n16_m*` / `n8_sk8_m16` / `n32_sk4_m16`).
        // This eliminates both multi-pass scalar GEMV and uncoalesced `w4a16_gemm_pipe`
        // latency across all 48 layers with zero extra VRAM and full CUDA-graph safety.
        // Rows <= 3 and all `C=1` probes (`scope_active == false`) remain on the
        // bit-exact scalar GEMV path below.
        let use_rowpar = ops::w4a16_exact_rowpar_selected(self.w4a16_gemv_rows, num_experts);
        let ran_tc_router = !use_rowpar
            && batched_verify_router_tc_admits_with_policy(
                crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active(),
                crate::layers::qwen4_hyper::batched_verify_tc_dense_enabled(),
                crate::layers::qwen4_hyper::batched_verify_tc_dense_min_rows_override(),
                rows,
                num_experts,
                h,
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
            )
            && super::qwen4_router_pipe::try_launch_batched_verify_router_gemm(
                ctx.gpu,
                router_in,
                nvfp4,
                gate_logits,
                rows as u32,
                num_experts,
                h,
                stream,
            )?;

        if !ran_tc_router {
            // Preserve ordinary K=1 lane ownership and BF16 rounding exactly:
            // row-parallel K1 for small N (<= W4A16_EXACT_ROWPAR_MAX_N), single-read
            // exact multi-row GEMV (`m4/m8/m17/m32`) for wide N=512, or one K1 launch per row.
            let exact_kernels = ops::W4a16ExactLmHeadKernels::new(
                self.w4a16_gemv_exact_m4,
                self.w4a16_gemv_exact_m8,
                self.w4a16_gemv_exact_m17,
                self.w4a16_gemv_exact_m32,
            )
            .with_rowpar(self.w4a16_gemv_rows);
            if use_rowpar {
                ops::w4a16_gemv_rows(
                    ctx.gpu,
                    self.w4a16_gemv_rows,
                    router_in,
                    nvfp4,
                    gate_logits,
                    rows as u32,
                    num_experts,
                    h,
                    stream,
                )?;
            } else if matches!(
                exact_kernels.route_for_rows(rows as u32),
                Some(ops::ExactLmHeadRoute::Exact(_))
            ) {
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    exact_kernels,
                    router_in,
                    nvfp4,
                    gate_logits,
                    rows as u32,
                    num_experts,
                    h,
                    stream,
                    false,
                )?;
            } else {
                for row in 0..rows {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        router_in.offset(row * h as usize * 2),
                        nvfp4,
                        gate_logits.offset(row * num_experts as usize * 2),
                        num_experts,
                        h,
                        stream,
                    )?;
                }
            }
        }

        let indices_dev = ctx.buffers.scratch();
        let weights_dev = indices_dev.offset(rows * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                1.0,
                rows as u32,
                stream,
            )?;
        } else if !super::qwen4_topk_warp::try_launch(
            ctx.gpu,
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            rows as u32,
            stream,
        )? {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                rows as u32,
                stream,
            )?;
        }

        spark_runtime::cublaslt::stage_timer::mark("v_router", stream);
        if let Some(plan) = tc_moe_plan {
            return self.run_qwen4_batched_verify_tc_moe(
                plan,
                input,
                rows,
                output,
                indices_dev,
                weights_dev,
                ctx,
                stream,
            );
        }
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_gate_out = ctx.buffers.logits();
        let shared_up_out = ctx.buffers.ssm_qkvz();
        let shared_down_out = ctx.buffers.attn_output();
        if t_rows {
            self.qwen4_exact_t_lanes_rows(
                input,
                indices_dev,
                (expert_gate_out, expert_up_out, expert_down_out),
                (shared_gate_out, shared_up_out, shared_down_out),
                rows as u32,
                ctx,
                stream,
            )?;
            spark_runtime::cublaslt::stage_timer::mark("v_experts", stream);
            ops::moe_weighted_sum_blend_batch3(
                ctx.gpu,
                self.moe_weighted_sum_blend_batch3,
                output,
                expert_down_out,
                weights_dev,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                rows as u32,
                stream,
            )?;
            if ctx.config.ep_world_size > 1
                && let Some(comm) = ctx.comm
                && comm.world_size() > 1
            {
                if ctx.graph_capture {
                    comm.all_reduce(output.0, rows * h as usize * 2)?;
                } else {
                    comm.all_reduce_async(output.0, rows * h as usize * 2, stream)?;
                }
            }
            return Ok(());
        }

        // The formerly K=3 CUDA module is token-count parameterized. Its
        // per-row K16 reduction remains byte-identical to K=1/K=2/K=3.
        // ATLAS_QWEN4_MOE_DEDUP=1 swaps in the expert-deduplicated twins
        // (same per-row arithmetic, each distinct expert streamed once).
        anyhow::ensure!(
            self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
            "unported non-unified exact MoE path requires FlatE4m3; set ATLAS_MOE_SCALE_TS4E=0"
        );
        let dedup = qwen4_moe_dedup_selected()
            .then(|| {
                let tier = usize::from(rows > 8);
                let rows_max = if tier == 0 { 8u32 } else { 16u32 };
                let gu = self.moe_expert_gate_up_shared_dedup[tier];
                let dn = self.moe_expert_silu_down_shared_dedup[tier];
                (gu.0 != 0 && dn.0 != 0 && rows <= rows_max as usize).then_some((gu, dn, rows_max))
            })
            .flatten();
        let check = dedup.is_some() && qwen4_moe_dedup_check() && !ctx.graph_capture;
        let gate_up = |kernel: KernelHandle| {
            ops::moe_expert_gate_up_shared_batch3(
                ctx.gpu,
                kernel,
                input,
                self.gate_ptrs.packed_ptrs,
                self.gate_ptrs.scale_ptrs,
                self.gate_ptrs.scale2_vals,
                expert_gate_out,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                indices_dev,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                inter,
                h,
                top_k,
                rows as u32,
                stream,
            )
        };
        // ATLAS_QWEN4_MOE_DOWN_ACT=1: the dedup down reads SiLU*up rows computed
        // once (byte-identical) instead of re-staging them in every block.
        let act_down = dedup.and_then(|(_, _, rows_max)| {
            let (kernel, outp) =
                self.moe_expert_silu_down_shared_dedup_act[usize::from(rows_max > 8)];
            debug_assert!(rows_max as usize <= ops::MOE_DOWN_ACT_MAX_ROWS);
            (qwen4_moe_down_act_selected() && kernel.0 != 0 && self.moe_dedup_act_rows_k.0 != 0)
                .then_some((kernel, ops::moe_down_act_outputs_per_block(outp)))
        });
        let act_buf = match act_down {
            Some(_) => Some(self.moe_down_act_buffer(ctx, top_k as usize, inter as usize)?),
            None => None,
        };
        let down = |dedup_kernel: Option<(KernelHandle, u32)>| match dedup_kernel {
            Some((_, rows_max)) if act_down.is_some() && act_buf.is_some() => {
                let act = act_buf.unwrap_or(DevicePtr::NULL);
                ops::moe_dedup_act_rows(
                    ctx.gpu,
                    self.moe_dedup_act_rows_k,
                    expert_gate_out,
                    expert_up_out,
                    shared_gate_out,
                    shared_up_out,
                    act,
                    inter,
                    top_k,
                    rows as u32,
                    stream,
                )?;
                let (act_kernel, outputs_per_block) = act_down.unwrap_or((KernelHandle(0), 8));
                ops::moe_expert_silu_down_shared_dedup_act(
                    ctx.gpu,
                    act_kernel,
                    outputs_per_block,
                    rows_max,
                    act,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    indices_dev,
                    &self.weights.shared_expert.down_proj,
                    shared_down_out,
                    h,
                    inter,
                    top_k,
                    rows as u32,
                    stream,
                )
            }
            Some((kernel, rows_max)) => ops::moe_expert_silu_down_shared_dedup(
                ctx.gpu,
                kernel,
                rows_max,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_out,
                shared_up_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                rows as u32,
                stream,
            ),
            None => ops::moe_expert_silu_down_shared_batch3(
                ctx.gpu,
                self.moe_expert_silu_down_shared_batch3,
                expert_gate_out,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                indices_dev,
                shared_gate_out,
                shared_up_out,
                &self.weights.shared_expert.down_proj,
                shared_down_out,
                h,
                inter,
                top_k,
                rows as u32,
                stream,
            ),
        };
        let routed_inter = rows * top_k as usize * inter as usize * 2;
        let shared_inter = rows * inter as usize * 2;
        let routed_hidden = rows * top_k as usize * h as usize * 2;
        let shared_hidden = rows * h as usize * 2;
        match dedup {
            Some((gu, dn, rows_max)) if check => {
                // Self-check: reference kernels first, snapshot, then the
                // dedup twins over the same inputs; compare byte for byte.
                gate_up(self.moe_expert_gate_up_shared_batch3)?;
                let reference_up = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_gate_out, routed_inter),
                        (expert_up_out, routed_inter),
                        (shared_gate_out, shared_inter),
                        (shared_up_out, shared_inter),
                    ],
                )?;
                gate_up(gu)?;
                let mut dedup_up = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_gate_out, routed_inter),
                        (expert_up_out, routed_inter),
                        (shared_gate_out, shared_inter),
                        (shared_up_out, shared_inter),
                    ],
                )?;
                down(None)?;
                let reference_down = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_down_out, routed_hidden),
                        (shared_down_out, shared_hidden),
                    ],
                )?;
                down(Some((dn, rows_max)))?;
                let dedup_down = snapshot(
                    ctx,
                    stream,
                    &[
                        (expert_down_out, routed_hidden),
                        (shared_down_out, shared_hidden),
                    ],
                )?;
                if std::env::var("ATLAS_QWEN4_MOE_DEDUP_CHECK_CORRUPT")
                    .ok()
                    .as_deref()
                    == Some("1")
                    && let Some(byte) = dedup_up.first_mut().and_then(|b| b.first_mut())
                {
                    // Positive control: the checker must report this flip.
                    *byte ^= 0x01;
                }
                record_dedup_check(rows, &reference_up, &dedup_up, &reference_down, &dedup_down);
            }
            Some((gu, dn, rows_max)) => {
                gate_up(gu)?;
                down(Some((dn, rows_max)))?;
            }
            None => {
                gate_up(self.moe_expert_gate_up_shared_batch3)?;
                down(None)?;
            }
        }
        let is_ep = ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        let shared_for_blend = if is_ep && !shared_down_out.is_null() {
            let zero_bytes = rows * h as usize * 2;
            anyhow::ensure!(
                zero_bytes <= ctx.buffers.sizes().expert_gate_out,
                "forward_qwen4_exact_into EP zero buffer needs {zero_bytes} B in expert_gate_out, arena has {}",
                ctx.buffers.sizes().expert_gate_out,
            );
            ctx.gpu
                .memset_async(expert_gate_out, 0, zero_bytes, stream)?;
            expert_gate_out
        } else {
            shared_down_out
        };
        ops::moe_weighted_sum_blend_batch3(
            ctx.gpu,
            self.moe_weighted_sum_blend_batch3,
            output,
            expert_down_out,
            weights_dev,
            shared_for_blend,
            input,
            self.weights.shared_expert_gate.weight,
            h,
            top_k,
            h,
            rows as u32,
            stream,
        )?;
        if ctx.config.ep_world_size > 1
            && let Some(comm) = ctx.comm
            && comm.world_size() > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, rows * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, rows * h as usize * 2, stream)?;
            }
            if !shared_down_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add,
                        output,
                        shared_down_out,
                        rows as u32 * h,
                        stream,
                    )?;
                } else {
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_down_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        rows as u32,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Fail-closed preflight for routing batched verify MoE (`rows >= min_rows`, default 4)
    /// onto the in-tree W4A8 `QMMA.SF` compact MoE (`v28_m32_ts4e` / `v28_twarp_m64_ts4e` +
    /// `qwen4_shared_w4a8`). Returns `Ok(None)` to fall back cleanly to the exact `t8`
    /// worklist whenever any selector, threshold, capture-readiness, shape, weight, kernel, or
    /// buffer-disjointness requirement is unmet.
    fn batched_verify_tc_moe_plan(
        &self,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<BatchedVerifyTcMoePlan>> {
        if crate::layers::deterministic_batch::deterministic_batch_enabled() {
            return Ok(None);
        }
        let policy = super::qwen4_prefill_compact::batched_verify_tc_moe_policy()?;
        let capturing = ctx.graph_capture || super::qwen4_moe_status::is_capturing(ctx.gpu, stream);
        let compact_active =
            super::qwen4_prefill_compact::request_active(self.qwen4_compact.is_some());
        let kernels = self.qwen4_compact.filter(|k| k.transposed && !k.streamed);
        let wide = kernels.and_then(|k| k.wide);
        let w4a8_env = super::qwen4_prefill_compact::w4a8_selected().unwrap_or(false)
            && !super::qwen4_prefill_compact::check_selected().unwrap_or(false);
        let fakequant_ok = crate::layers::qwen4_fakequant::current().is_ok_and(|spec| {
            spec.moe_in.is_none()
                && spec.moe_down.is_none()
                && !spec.moe_weights_b12x
                && !spec.moe_kernel_v7
        });
        let prefill_receipt_active = crate::model::qwen4_prefill_engagement::active_for(rows);
        let use_ts4e = self.routed_scale_layout == RoutedScaleLayout::Ts4e128;
        let w4a8_ready = wide.is_some_and(|w| {
            let slot_storage = super::qwen4_prefill_compact::dispatch_slot_storage(
                w.slot_storage,
                crate::model::qwen4_prefill_engagement::slot_r2_active_for(rows),
            );
            let slot_ok = if slot_storage.is_mxfp8() {
                !slot_storage.has_corrections()
                    && w.w4a8_down_mxfp8.is_some()
                    && w.unpermute_mxfp8.is_some()
            } else {
                self.moe_unpermute_reduce.0 != 0
            };
            let layout_ok = !use_ts4e
                || super::qwen4_compact_ops::v28_ts4e_supported(
                    w.w4a8_tag,
                    w.v28q_level,
                    w.v28_m64,
                    slot_storage.has_corrections(),
                );
            w4a8_env && fakequant_ok && slot_ok && layout_ok
        });
        let qw_live = |w: &QuantizedWeight| {
            !w.weight.is_null()
                && !w.weight_scale.is_null()
                && w.weight_scale_2.is_finite()
                && w.weight_scale_2 > 0.0
        };
        let qw_fast_v2_live = |w: &QuantizedWeight| {
            !w.weight.is_null()
                && !w.weight_scale.is_null()
                && w.weight_scale_2.is_finite()
                && w.weight_scale_2 > 0.0
                && w.weight.0.is_multiple_of(16)
                && w.weight_scale.0.is_multiple_of(16)
        };
        let shared_ready = self.shared_gate_t.as_ref().is_some_and(qw_live)
            && self.shared_up_t.as_ref().is_some_and(qw_live)
            && self.shared_down_t.as_ref().is_some_and(qw_live)
            && !self.weights.shared_expert_gate.weight.is_null()
            && self.moe_batched_blend.0 != 0;
        let shared_fast_v2_base = decode_shared_fast_v2_selected()
            && (2..=NATIVE_TARGET_MOE_MAX_TILE).contains(&rows)
            && self.shared_gate_t.as_ref().is_some_and(qw_fast_v2_live)
            && self.shared_up_t.as_ref().is_some_and(qw_fast_v2_live)
            && self.shared_down_t.as_ref().is_some_and(qw_fast_v2_live)
            && !input.is_null()
            && input.0.is_multiple_of(16)
            && !ctx.buffers.ssm_deinterleaved().is_null()
            && ctx.buffers.ssm_deinterleaved().0.is_multiple_of(16)
            && !ctx.buffers.attn_output().is_null()
            && ctx.buffers.attn_output().0.is_multiple_of(16)
            && crate::layers::qwen4_fast_proj::shared_expert_nvfp4_t_fast_v2_available(
                ctx.gpu, rows,
            );
        let fused_sort_plan =
            super::qwen4_router_pipe::fused_sort_plan_active(true, self.qwen4_compact, rows)
                .unwrap_or(false);
        let capture_scratch_ready = !capturing
            || (super::qwen4_moe_status::selected()
                && super::qwen4_moe_status::is_capture_ready(
                    rows,
                    !fused_sort_plan,
                    !shared_fast_v2_base,
                )
                && (shared_fast_v2_base || super::qwen4_shared_v7::is_capture_ready(self, rows)));
        let admitted = policy.admits(
            super::qwen4_prefill_compact::BatchedVerifyTcMoeEligibility {
                rows,
                graph_capture: capturing,
                capture_scratch_ready,
                compact_active,
                w4a8_ready,
                shared_ready,
            },
        );
        if !admitted {
            return Ok(None);
        }
        let Some(wide) = wide else {
            return Ok(None);
        };
        if prefill_receipt_active
            || !crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
            || rows > NATIVE_TARGET_MOE_MAX_TILE
            || input != ctx.buffers.norm_output()
            || output != ctx.buffers.moe_output()
            || crate::layers::qwen4_oracle::root().is_some()
            || ctx.config.hidden_size != 2560
            || ctx.config.moe_intermediate_size != 640
            || ctx.config.shared_expert_intermediate_size != 640
            || ctx.config.num_experts != 512
            || ctx.config.num_experts_per_tok != 10
            || ctx.config.ep_world_size > 4
            || ctx.config.tp_world_size > 4
            || self.weights.router_pre_norm.is_some()
            || self.pre_expert_norm.is_some()
            || self.correction_bias_dev.is_some()
        {
            return Ok(None);
        }
        let Ok(gatepre) =
            crate::layers::qwen4_hyper::prefill_inject::selection().map(|s| s.gatepre)
        else {
            return Ok(None);
        };
        let shared_w4a8 = super::qwen4_shared_v7::w4a8_selected();
        let pretranscode = shared_w4a8
            && super::qwen4_shared_v7::pretranscode_active_for(rows)
            && wide.w4a8_tag == "v28";
        let shared_fast_v2_act_bytes = rows * 640 * 2;
        let legacy_ssm_deinterleaved_need = if !shared_w4a8 {
            rows * 640 * 2
        } else if pretranscode {
            SHARED_W4A8_PRETRANSCODE_SCRATCH_BYTES
        } else {
            0
        };
        let ssm_deinterleaved_need = if shared_fast_v2_base {
            shared_fast_v2_act_bytes
        } else {
            legacy_ssm_deinterleaved_need
        };
        let sizes = ctx.buffers.sizes();
        let perm_off = compact_token_to_perm_offset(rows, gatepre);
        if sizes.moe_output < rows * 2560 * 2
            || sizes.gate_logits < perm_off + rows * 10 * 4
            || sizes.ssm_deinterleaved < ssm_deinterleaved_need
        {
            return Ok(None);
        }
        if super::qwen4_prefill_compact::validate_context(ctx, rows).is_err()
            || self.validate_qwen4_prefill_moe().is_err()
        {
            return Ok(None);
        }
        let peers = [
            (ctx.buffers.norm_output(), sizes.norm_output),
            (ctx.buffers.attn_output(), sizes.attn_output),
            (ctx.buffers.gate_logits(), sizes.gate_logits),
            (ctx.buffers.scratch(), sizes.scratch),
            (ctx.buffers.expert_gate_out(), sizes.expert_gate_out),
            (ctx.buffers.expert_up_out(), sizes.expert_up_out),
            (ctx.buffers.expert_down_out(), sizes.expert_down_out),
            (
                ctx.buffers.moe_worklist(),
                super::qwen4_compact_contract::ARENA_BYTES,
            ),
            (ctx.buffers.moe_worklist_total(), 4),
        ];
        if !verify_tc_moe_output_disjoint(output, rows, &peers) {
            return Ok(None);
        }
        if ssm_deinterleaved_need > 0
            && !verify_tc_moe_output_disjoint(
                output,
                rows,
                &[(ctx.buffers.ssm_deinterleaved(), ssm_deinterleaved_need)],
            )
        {
            return Ok(None);
        }
        let ss_ptr = ctx.buffers.moe_gate_scalars();
        let ss_bytes = rows * 4;
        if gatepre && !verify_tc_moe_output_disjoint(output, rows, &[(ss_ptr, ss_bytes)]) {
            return Ok(None);
        }
        let slot_storage = super::qwen4_prefill_compact::dispatch_slot_storage(
            wide.slot_storage,
            crate::model::qwen4_prefill_engagement::slot_r2_active_for(rows),
        );
        let ss_arena_ok = !ss_ptr.is_null()
            && sizes.moe_gate_scalars >= ss_bytes
            && verify_tc_moe_region_disjoint(ss_ptr, ss_bytes, &peers)
            && verify_tc_moe_region_disjoint(ss_ptr, ss_bytes, &[(output, rows * 2560 * 2)]);
        let fused_unpermute_blend = slot_storage.is_mxfp8()
            && !slot_storage.has_corrections()
            && wide.w4a8_quant_gate.is_some()
            && wide.unpermute_blend_mxfp8.is_some()
            && ss_arena_ok;
        let shared_fast_v2 = shared_fast_v2_base
            && sizes.ssm_deinterleaved >= shared_fast_v2_act_bytes
            && sizes.attn_output >= rows * 2560 * 2
            && verify_tc_moe_region_disjoint(
                ctx.buffers.ssm_deinterleaved(),
                shared_fast_v2_act_bytes,
                &peers,
            )
            && verify_tc_moe_region_disjoint(
                ctx.buffers.ssm_deinterleaved(),
                shared_fast_v2_act_bytes,
                &[(output, rows * 2560 * 2)],
            )
            && (!(gatepre || fused_unpermute_blend)
                || verify_tc_moe_region_disjoint(
                    ctx.buffers.ssm_deinterleaved(),
                    shared_fast_v2_act_bytes,
                    &[(ss_ptr, ss_bytes)],
                ));
        if !shared_fast_v2
            && ((capturing
                && (!super::qwen4_shared_v7::is_capture_ready(self, rows)
                    || !super::qwen4_moe_status::is_capture_ready(rows, !fused_sort_plan, true)))
                || sizes.ssm_deinterleaved < legacy_ssm_deinterleaved_need
                || (legacy_ssm_deinterleaved_need > 0
                    && !verify_tc_moe_output_disjoint(
                        output,
                        rows,
                        &[(
                            ctx.buffers.ssm_deinterleaved(),
                            legacy_ssm_deinterleaved_need,
                        )],
                    )))
        {
            return Ok(None);
        }
        let thin_sk_active = rows <= MOE_SHARED_OVERLAP_THIN_SK_MAX_ROWS
            && super::qwen4_compact_ops::qwen4_moe_thin_sk::thin_sk_policy()
                .ok()
                .is_some_and(|policy| {
                    policy.admits(
                        super::qwen4_compact_ops::qwen4_moe_thin_sk::ThinSkEligibility {
                            rows,
                            w4a8_is_v28: wide.w4a8_tag == "v28",
                            has_corrections: slot_storage.has_corrections(),
                            v29_m256_active: false,
                            oracle_capture: crate::layers::qwen4_oracle::root().is_some(),
                            kernels_loaded:
                                super::qwen4_compact_ops::qwen4_moe_thin_sk::thin_sk_kernels_loaded(
                                    ctx.gpu,
                                ),
                        },
                    )
                });
        let shared_overlap_bytes = if shared_fast_v2 {
            shared_fast_v2_act_bytes
        } else {
            rows * super::qwen4_shared_v7::SHARED_W4A8_ACT_BYTES_PER_ROW
        };
        let shared_overlap = (shared_fast_v2 || (shared_w4a8 && !pretranscode))
            && super::qwen4_moe_status::selected()
            && moe_shared_overlap_admits_capture(
                moe_shared_overlap_enabled_for_thin_sk_and_sm(
                    moe_shared_overlap_override(),
                    shared_fast_v2,
                    atlas_core::device::active_sm_count(),
                    thin_sk_active,
                    rows,
                    moe_shared_overlap_thin_sk_selected(),
                ),
                moe_shared_graph_overlap_selected(),
                capturing,
            )
            && self.prefill_stream != 0
            && self.prefill_stream != stream
            && self.event_a != 0
            && self.event_b != 0
            && sizes.ssm_deinterleaved >= shared_overlap_bytes
            && verify_tc_moe_region_disjoint(
                ctx.buffers.ssm_deinterleaved(),
                shared_overlap_bytes,
                &peers,
            )
            && verify_tc_moe_region_disjoint(
                ctx.buffers.ssm_deinterleaved(),
                shared_overlap_bytes,
                &[(output, rows * 2560 * 2)],
            )
            && (!(gatepre || fused_unpermute_blend)
                || verify_tc_moe_region_disjoint(
                    ctx.buffers.ssm_deinterleaved(),
                    shared_overlap_bytes,
                    &[(ss_ptr, ss_bytes)],
                ));
        let v28_dispatch = super::qwen4_compact_ops::select_v28_dispatch(
            wide.w4a8_tag,
            wide.v28_m64,
            super::qwen4_compact_ops::v28_m32_active_for_scope(),
            rows,
            false,
        );
        let use_m32_gateup = v28_dispatch.gateup == super::qwen4_compact_ops::V28OpTile::M32;
        let use_m32_down = v28_dispatch.down == super::qwen4_compact_ops::V28OpTile::M32;
        let use_m64 = v28_dispatch.gateup == super::qwen4_compact_ops::V28OpTile::M64
            || v28_dispatch.down == super::qwen4_compact_ops::V28OpTile::M64;
        Ok(Some(BatchedVerifyTcMoePlan {
            use_ts4e,
            use_m64,
            use_m32_gateup,
            use_m32_down,
            shared_w4a8,
            shared_fast_v2,
            shared_overlap,
            gatepre,
            fused_unpermute_blend,
        }))
    }

    /// Drain and verify all deferred compact-MoE status words queued by batched verify.
    pub(crate) fn flush_batched_verify_moe_status(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        flush_batched_verify_moe_status(gpu, stream)
    }

    /// Execute batched verify MoE (`rows >= min_rows`, default 4) over the preflighted
    /// in-tree W4A8 `QMMA.SF` compact MoE (`v28_m32_ts4e` / `v28_twarp_m64_ts4e` +
    /// `qwen4_shared_w4a8`), reusing exact router indices/weights already placed in
    /// `ctx.buffers.scratch()`.
    #[allow(clippy::too_many_arguments)]
    fn run_qwen4_batched_verify_tc_moe(
        &self,
        plan: BatchedVerifyTcMoePlan,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let capturing = ctx.graph_capture || super::qwen4_moe_status::is_capturing(ctx.gpu, stream);
        let status_async = super::qwen4_moe_status::selected();
        let use_overlap = plan.shared_overlap
            && status_async
            && (!capturing || moe_shared_graph_overlap_selected())
            && self.prefill_stream != stream;
        let layer_key = self.gate_ptrs.packed_ptrs.0;
        if status_async && !capturing && layer_key != 0 {
            let restart = BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(|c| {
                c.get() > 0 && BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY.with(|k| k.get() == layer_key)
            });
            if restart {
                Self::flush_batched_verify_moe_status(ctx.gpu, stream)?;
            }
        }
        super::qwen4_compact_ops::forget_xq();
        let _ = super::qwen4_compact_ops::discard_stale_slot_storage();
        let result = {
            let _scope = BatchedVerifyTcMoeScopeGuard::enter(
                status_async,
                use_overlap && !plan.shared_fast_v2,
                plan.fused_unpermute_blend,
            );
            self.run_qwen4_batched_verify_tc_moe_inner(
                plan,
                input,
                rows,
                output,
                indices_dev,
                weights_dev,
                status_async,
                use_overlap,
                ctx,
                stream,
            )
        };
        super::qwen4_compact_ops::forget_xq();
        let (slot_storage, used_fused_blend) = match result {
            Ok(outcome) => outcome,
            Err(err) => {
                let _ = super::qwen4_compact_ops::discard_stale_slot_storage();
                if status_async {
                    let _ = super::qwen4_moe_status::discard_stale();
                    reset_batched_verify_status_tracking();
                }
                return Err(err);
            }
        };
        if status_async && !capturing {
            let pending = BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(|c| {
                let prev = c.get();
                if prev == 0 {
                    BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY.with(|k| k.set(layer_key));
                }
                let next = prev + 1;
                c.set(next);
                next
            });
            if should_flush_batched_verify_status(pending, ctx.config.num_hidden_layers) {
                Self::flush_batched_verify_moe_status(ctx.gpu, stream)?;
            }
        }
        spark_runtime::cublaslt::stage_timer::mark("v_experts", stream);
        if !BATCHED_VERIFY_TC_MOE_RECEIPT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "QWEN4_BATCHED_VERIFY_TC_MOE_ENGAGED rows={rows} ts4e={} m32_gu={} m32_dn={} m64={} shared_w4a8={} shared_fast_v2={} slot_mxfp8={} fused_blend={} status_async={} shared_overlap={} capturing={capturing}",
                plan.use_ts4e,
                plan.use_m32_gateup,
                plan.use_m32_down,
                plan.use_m64,
                plan.shared_w4a8,
                plan.shared_fast_v2,
                slot_storage.is_mxfp8(),
                used_fused_blend,
                status_async,
                use_overlap,
            );
        }
        Ok(())
    }

    fn launch_qwen4_shared_fast_v2(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gate_t = self
            .shared_gate_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("shared_fast_v2 requires live shared_gate_t"))?;
        let up_t = self
            .shared_up_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("shared_fast_v2 requires live shared_up_t"))?;
        let down_t = self
            .shared_down_t
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("shared_fast_v2 requires live shared_down_t"))?;
        let launched = crate::layers::qwen4_fast_proj::try_shared_expert_nvfp4_t_fast_v2(
            ctx.gpu,
            input,
            gate_t,
            up_t,
            down_t,
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.attn_output(),
            rows,
            ctx.config.shared_expert_intermediate_size,
            ctx.config.hidden_size,
            stream,
        )?;
        anyhow::ensure!(
            launched,
            "shared_fast_v2 preflight admitted rows={rows} but kernel launch failed closed"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen4_batched_verify_routed_gemms(
        &self,
        plan: BatchedVerifyTcMoePlan,
        input: DevicePtr,
        rows: usize,
        indices_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(super::qwen4_prefill_compact::SlotStorage, DevicePtr)> {
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let te = rows * top_k as usize;
        let gate_logits = ctx.buffers.gate_logits();
        let sorted_token_ids = gate_logits;
        let sorted_expert_ids = gate_logits.offset(te * 4);
        let expert_offsets = gate_logits.offset(te * 4 * 2);
        let perm_off = compact_token_to_perm_offset(rows, plan.gatepre);
        let token_to_perm = gate_logits.offset(perm_off);
        let is_ep = ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        if is_ep
            || !super::qwen4_router_pipe::fused_sort_plan_active(true, self.qwen4_compact, rows)?
        {
            ops::moe_sort_by_expert(
                ctx.gpu,
                self.moe_sort_by_expert,
                indices_dev,
                sorted_token_ids,
                sorted_expert_ids,
                expert_offsets,
                token_to_perm,
                te as u32,
                num_experts,
                top_k,
                stream,
            )?;
        }
        self.run_qwen4_compact(input, expert_offsets, sorted_token_ids, rows, ctx, stream)?;
        let slot_storage = super::qwen4_compact_ops::take_slot_storage();
        Ok((slot_storage, token_to_perm))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen4_batched_verify_tc_moe_inner(
        &self,
        plan: BatchedVerifyTcMoePlan,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        status_async: bool,
        use_overlap: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(super::qwen4_prefill_compact::SlotStorage, bool)> {
        let h = ctx.config.hidden_size as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let te = rows * top_k as usize;
        let is_ep = ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        let skip_shared = is_ep && ctx.config.ep_rank > 0;
        if is_ep {
            let (ep_start, ep_end) = ctx.config.local_expert_range();
            super::qwen4_compact_ops::launch_ep_mask_remote_weights(
                ctx.gpu,
                indices_dev,
                weights_dev,
                ep_start,
                ep_end,
                te as u32,
                stream,
            )?;
        }
        let (slot_storage, token_to_perm) = if skip_shared {
            self.run_qwen4_batched_verify_routed_gemms(plan, input, rows, indices_dev, ctx, stream)?
        } else if use_overlap && plan.shared_fast_v2 {
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu
                .stream_wait_event(self.prefill_stream, self.event_a)?;
            let shared_launch =
                self.launch_qwen4_shared_fast_v2(input, rows, ctx, self.prefill_stream);
            let rec_b = ctx.gpu.record_event(self.event_b, self.prefill_stream);
            if shared_launch.is_err() || rec_b.is_err() {
                let wait_res = match &rec_b {
                    Ok(()) => ctx.gpu.stream_wait_event(stream, self.event_b),
                    Err(rec_err) => {
                        Err(anyhow::anyhow!("record_event(event_b) failed: {rec_err:#}"))
                    }
                };
                if let Err(wait_err) = wait_res {
                    tracing::error!(
                        "batched_verify shared_fast_v2 overlap error cleanup stream_wait_event failed (falling back to synchronize): {wait_err:#}"
                    );
                    if let Err(sync_err) = ctx.gpu.synchronize(self.prefill_stream) {
                        tracing::error!(
                            "batched_verify shared_fast_v2 overlap error cleanup synchronize(prefill_stream) failed: {sync_err:#}"
                        );
                    }
                }
                shared_launch?;
                rec_b?;
            }
            let routed_result = self.run_qwen4_batched_verify_routed_gemms(
                plan,
                input,
                rows,
                indices_dev,
                ctx,
                stream,
            );
            let join_result = ctx.gpu.stream_wait_event(stream, self.event_b);
            let routed = routed_result?;
            join_result?;
            routed
        } else if use_overlap {
            let prepared = self.prepare_qwen4_shared_w4a8_overlap(input, rows, ctx, stream)?;
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu
                .stream_wait_event(self.prefill_stream, self.event_a)?;
            let shared_launch = prepared.launch(ctx.gpu, self.prefill_stream);
            let rec_b = ctx.gpu.record_event(self.event_b, self.prefill_stream);
            if shared_launch.is_err() || rec_b.is_err() {
                let wait_res = match &rec_b {
                    Ok(()) => ctx.gpu.stream_wait_event(stream, self.event_b),
                    Err(rec_err) => {
                        Err(anyhow::anyhow!("record_event(event_b) failed: {rec_err:#}"))
                    }
                };
                if let Err(wait_err) = wait_res {
                    tracing::error!(
                        "batched_verify shared_w4a8 overlap error cleanup stream_wait_event failed (falling back to synchronize): {wait_err:#}"
                    );
                    if let Err(sync_err) = ctx.gpu.synchronize(self.prefill_stream) {
                        tracing::error!(
                            "batched_verify shared_w4a8 overlap error cleanup synchronize(prefill_stream) failed: {sync_err:#}"
                        );
                    }
                }
                shared_launch?;
                rec_b?;
            }
            let routed_result = self.run_qwen4_batched_verify_routed_gemms(
                plan,
                input,
                rows,
                indices_dev,
                ctx,
                stream,
            );
            let join_result = ctx.gpu.stream_wait_event(stream, self.event_b);
            let routed = routed_result?;
            join_result?;
            routed
        } else {
            if plan.shared_fast_v2 {
                self.launch_qwen4_shared_fast_v2(input, rows, ctx, stream)?;
            } else {
                self.run_qwen4_shared_v7(input, rows, ctx, stream)?;
            }
            self.run_qwen4_batched_verify_routed_gemms(plan, input, rows, indices_dev, ctx, stream)?
        };
        let ss_ptr = if !skip_shared && plan.fused_unpermute_blend {
            super::qwen4_compact_ops::take_gate_scalars(
                input,
                self.weights.shared_expert_gate.weight,
                rows,
            )
        } else {
            DevicePtr::NULL
        };
        let expert_down_out = ctx.buffers.expert_down_out();
        let fused_kernel = if !skip_shared
            && slot_storage.is_mxfp8()
            && !slot_storage.has_corrections()
            && !ss_ptr.is_null()
        {
            self.qwen4_compact
                .and_then(|k| k.wide)
                .and_then(|w| w.unpermute_blend_mxfp8)
        } else {
            None
        };
        let used_fused_blend = if let Some(fused_k) = fused_kernel {
            let slot_layout = super::qwen4_prefill_compact::slot_layout(te, slot_storage)?;
            let scales_offset = anyhow::Context::context(
                slot_layout.scales_offset,
                "MXFP8 slot layout omitted scales",
            )?;
            let slot_scales = expert_down_out.offset(scales_offset);
            let (routed_status, shared_status, sticky_err) = if status_async {
                let (sh_st, err_ptr) = super::qwen4_moe_status::sticky_ptrs(ctx.gpu, stream)?;
                (ctx.buffers.moe_worklist_total(), sh_st, err_ptr)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL)
            };
            super::qwen4_compact_ops::launch_unpermute_reduce_blend_mxfp8_bf16(
                super::qwen4_compact_ops::UnpermuteBlendArgs {
                    gpu: ctx.gpu,
                    legacy_kernel: fused_k,
                    slot_q: expert_down_out,
                    slot_s: slot_scales,
                    shared_out: ctx.buffers.attn_output(),
                    ss_ptr,
                    output,
                    token_to_perm,
                    topk_weights: weights_dev,
                    hidden: h,
                    rows,
                    top_k,
                    routed_status,
                    shared_status,
                    sticky_err,
                    stream,
                },
            )?;
            true
        } else {
            if slot_storage.is_mxfp8() {
                anyhow::ensure!(
                    !slot_storage.has_corrections(),
                    "batched verify TC MoE does not support MXFP8 R2 corrections"
                );
                let slot_layout = super::qwen4_prefill_compact::slot_layout(te, slot_storage)?;
                let scales_offset = anyhow::Context::context(
                    slot_layout.scales_offset,
                    "MXFP8 slot layout omitted scales",
                )?;
                let slot_scales = expert_down_out.offset(scales_offset);
                let unpermute = anyhow::Context::context(
                    self.qwen4_compact
                        .and_then(|k| k.wide)
                        .and_then(|w| w.unpermute_mxfp8),
                    "MXFP8 slot rows without the MXFP8 unpermute kernel",
                )?;
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, unpermute)
                    .grid([rows as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(expert_down_out)
                    .arg_ptr(slot_scales)
                    .arg_ptr(output)
                    .arg_ptr(token_to_perm)
                    .arg_ptr(weights_dev)
                    .arg_u32(h)
                    .arg_u32(rows as u32)
                    .arg_u32(top_k)
                    .launch(stream)?;
            } else {
                ops::moe_unpermute_reduce_indexed(
                    ctx.gpu,
                    self.moe_unpermute_reduce,
                    expert_down_out,
                    output,
                    token_to_perm,
                    weights_dev,
                    h,
                    rows as u32,
                    top_k,
                    stream,
                )?;
            }
            if !skip_shared {
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    output,
                    ctx.buffers.attn_output(),
                    input,
                    self.weights.shared_expert_gate.weight,
                    h,
                    rows as u32,
                    stream,
                )?;
            }
            if status_async {
                super::qwen4_moe_status::accumulate_sticky_status(
                    ctx.gpu,
                    ctx.buffers.moe_worklist_total(),
                    stream,
                )?;
            }
            false
        };
        if is_ep && let Some(comm) = ctx.comm {
            if ctx.graph_capture {
                comm.all_reduce(output.0, rows * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, rows * h as usize * 2, stream)?;
            }
        }
        Ok((slot_storage, used_fused_blend))
    }
}

/// `ATLAS_QWEN4_MOE_DEDUP=1`: expert-deduplicated exact K5/K9 MoE kernels.
impl MoeLayer {
    /// Prove that C2-C256 will take the same unified `_t_lanes` arithmetic as
    /// C1 before any hyperconnection or recurrent state is modified. Frames
    /// above C128 run in a single pass on compact W4A8 TC MoE or split into two
    /// balanced `<= 128`-row halves when TC MoE is disabled (or `2..=9`-row
    /// sub-tiles when worklist kernels are unavailable), while the surrounding
    /// attention/SSM/hyperconnection frame remains C-wide.
    pub(crate) fn preflight_qwen4_native_exact_rows(
        &self,
        rows: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows),
            "native Qwen4 exact MoE preflight admits only C2-C256, got C{rows}"
        );
        anyhow::ensure!(
            ctx.config.is_qwen4_exp()
                && ctx.config.tp_world_size <= 4
                && ctx.config.ep_world_size <= 4,
            "native Qwen4 exact MoE requires TP<=4/EP<=4 target execution"
        );
        anyhow::ensure!(
            self.t_lanes_rows_route(),
            "native Qwen4 C{rows} requires the unified ATLAS_MOE_T_LANES exact-row route"
        );
        anyhow::ensure!(
            self.gate_nvfp4.is_some() && self.w4a16_gemv.0 != 0,
            "native Qwen4 exact MoE requires the NVFP4 router and K1 router kernel"
        );
        anyhow::ensure!(
            self.correction_bias_dev.is_none() && self.moe_topk_batched.0 != 0,
            "native Qwen4 exact MoE requires the softmax top-k route"
        );
        let h = u32::try_from(ctx.config.hidden_size)
            .map_err(|_| anyhow::anyhow!("native Qwen4 exact MoE hidden size exceeds u32"))?;
        let inter = u32::try_from(ctx.config.moe_intermediate_size)
            .map_err(|_| anyhow::anyhow!("native Qwen4 exact MoE intermediate size exceeds u32"))?;
        let num_experts = u32::try_from(ctx.config.num_experts)
            .map_err(|_| anyhow::anyhow!("native Qwen4 exact MoE expert count exceeds u32"))?;
        let top_k = u32::try_from(ctx.config.num_experts_per_tok)
            .map_err(|_| anyhow::anyhow!("native Qwen4 exact MoE top-k exceeds u32"))?;
        anyhow::ensure!(
            h != 0 && inter != 0 && num_experts != 0 && top_k != 0 && top_k <= num_experts,
            "native Qwen4 exact MoE dimensions are invalid"
        );
        let mut remaining = rows;
        while remaining > 0 {
            let tile = native_target_moe_tile_width(remaining).map_err(anyhow::Error::msg)?;
            super::qwen4_topk_warp::preflight(
                ctx.gpu,
                ctx.buffers.gate_logits(),
                num_experts,
                top_k,
                tile as u32,
            )?;
            remaining -= tile;
        }
        anyhow::ensure!(
            self.moe_expert_gate_up_shared_t_lanes_rows_k.0 != 0
                && self.moe_expert_silu_down_shared_t_lanes_rows_k.0 != 0
                && self.moe_weighted_sum_blend_batch3.0 != 0,
            "native Qwen4 exact MoE kernel set is incomplete"
        );
        let table_live = |table: &ExpertPtrTable| {
            !table.packed_ptrs.is_null()
                && !table.scale_ptrs.is_null()
                && !table.scale2_vals.is_null()
        };
        anyhow::ensure!(
            self.gate_ptrs_t.as_ref().is_some_and(table_live)
                && self.up_ptrs_t.as_ref().is_some_and(table_live)
                && self.down_ptrs_t.as_ref().is_some_and(table_live),
            "native Qwen4 exact MoE requires persistent unified expert pointer tables"
        );
        anyhow::ensure!(
            self.shared_gate_t
                .as_ref()
                .is_some_and(|weight| !weight.is_null())
                && self
                    .shared_up_t
                    .as_ref()
                    .is_some_and(|weight| !weight.is_null())
                && self
                    .shared_down_t
                    .as_ref()
                    .is_some_and(|weight| !weight.is_null())
                && !self.weights.shared_expert_gate.weight.is_null(),
            "native Qwen4 exact MoE requires live transposed shared-expert weights"
        );
        let tile_rows = rows.min(NATIVE_TARGET_MOE_MAX_TILE);
        let tile_bf16_extent = |width: usize| {
            tile_rows
                .checked_mul(width)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE extent overflow"))
        };
        let total_moe_out_bytes = rows
            .checked_mul(ctx.config.hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE output extent overflow"))?;
        let expanded = tile_rows
            .checked_mul(ctx.config.num_experts_per_tok)
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE expanded-row overflow"))?;
        let routed_inter_bytes = expanded
            .checked_mul(ctx.config.moe_intermediate_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE expert extent overflow"))?;
        let routed_down_bytes = expanded
            .checked_mul(ctx.config.hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE down extent overflow"))?;
        let routing_scratch = expanded
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 exact MoE routing extent overflow"))?;
        let arena = ctx.buffers.sizes();
        if self.weights.router_pre_norm.is_some() {
            anyhow::ensure!(
                self.pre_expert_norm_k.0 != 0
                    && tile_bf16_extent(ctx.config.hidden_size)? <= arena.qkv_output,
                "native Qwen4 exact MoE router pre-norm exceeds the QKV arena"
            );
        }
        anyhow::ensure!(
            tile_bf16_extent(ctx.config.num_experts)? <= arena.gate_logits
                && routing_scratch <= arena.scratch
                && routed_inter_bytes <= arena.expert_gate_out
                && routed_inter_bytes <= arena.expert_up_out
                && routed_down_bytes <= arena.expert_down_out
                && tile_bf16_extent(ctx.config.moe_intermediate_size)? <= arena.logits
                && tile_bf16_extent(ctx.config.moe_intermediate_size)? <= arena.ssm_qkvz
                && tile_bf16_extent(ctx.config.hidden_size)? <= arena.attn_output
                && total_moe_out_bytes <= arena.moe_output,
            "native Qwen4 exact MoE rows exceed the serving arenas"
        );
        Ok(())
    }

    /// Execute a preflighted C2-C256 target-only MoE frame. Single-pass
    /// worklist dispatch covers `<= 128` rows (`RMAX = 5 or 8`), while
    /// `129..=256` runs in one pass on compact W4A8 TC MoE or splits into two
    /// balanced `<= 128`-row halves when TC MoE is disabled (and `2..=9`-row
    /// sub-tiles when worklist kernels are unavailable). Inputs and outputs
    /// never alias, and no request is replayed through the C1 model loop.
    pub(crate) fn forward_qwen4_native_target_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.preflight_qwen4_native_exact_rows(rows, ctx)?;
        let row_bytes = ctx
            .config
            .hidden_size
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("native Qwen4 MoE row stride overflow"))?;
        let output = ctx.buffers.moe_output();
        let mut row = 0usize;
        while row < rows {
            let tile = native_target_moe_tile_width(rows - row).map_err(anyhow::Error::msg)?;
            self.forward_qwen4_exact_into(
                input.offset(row * row_bytes),
                tile,
                output.offset(row * row_bytes),
                ctx,
                stream,
            )?;
            row += tile;
        }
        Ok(())
    }

    /// Cross-sequence verify control: one exact launch per sequence, so each
    /// sequence's `k` rows share a MoE tile with no other sequence, exactly as
    /// in its single-sequence K-row verify. `k` must be a qualified exact
    /// width (2..=9).
    pub(crate) fn forward_qwen4_seq_aligned_rows(
        &self,
        input: DevicePtr,
        seqs: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = seqs
            .checked_mul(k)
            .ok_or_else(|| anyhow::anyhow!("sequence-aligned MoE row count overflow"))?;
        self.preflight_qwen4_native_exact_rows(rows, ctx)?;
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&k),
            "sequence-aligned MoE tile K{k} is not an exact width"
        );
        let row_bytes = ctx.config.hidden_size * 2;
        let output = ctx.buffers.moe_output();
        for s in 0..seqs {
            let base = s * k * row_bytes;
            self.forward_qwen4_exact_into(input.offset(base), k, output.offset(base), ctx, stream)?;
        }
        Ok(())
    }

    /// True when decode runs the exact `_t_lanes` kernels on the unified layout
    /// and the multi-row twins are loaded: every multi-row decode/verify MoE
    /// must then use them to stay byte-identical to K=1 decode.
    pub(crate) fn t_lanes_rows_route(&self) -> bool {
        self.use_t_layout_for_decode()
            && super::forward_phase::t_lanes_selected()
            && self.moe_expert_gate_up_shared_t_lanes_rows_k.0 != 0
            && self.moe_expert_silu_down_shared_t_lanes_rows_k.0 != 0
    }

    /// True when both gate/up and down worklist kernels cover `rows > 9` in a single pass.
    fn qwen4_worklist_covers_wide_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        top_k: u32,
    ) -> bool {
        let is_ts4e = self.routed_scale_layout == RoutedScaleLayout::Ts4e128;
        let force_r5 = std::env::var("ATLAS_MOE_T_LANES_ROWS_T8_R5_ONLY").is_ok_and(|v| v == "1");
        let max_slots = rows * top_k;
        let e512_ok = num_experts <= 512
            && rows <= WORKLIST_E512_KERNEL_MAX_ROWS
            && max_slots <= WORKLIST_E512_KERNEL_MAX_SLOTS.min(WORKLIST_MAX_SLOTS as u32)
            && moe_build_unique_worklist_e512_kernel(gpu).0 != 0;
        let r8_ok = rows <= 32 && max_slots <= 320 && self.moe_build_unique_worklist_r8_k.0 != 0;
        if force_r5
            || !moe_t_lanes_worklist_selected()
            || (!is_ts4e && !moe_t_lanes_rows_down_t8_selected())
            || !moe_t_lanes_down_act_selected()
            || !(e512_ok || r8_ok)
            || self.moe_silu_mul_rows_f32_k.0 == 0
        {
            return false;
        }
        let force_r8 = std::env::var("ATLAS_MOE_T_LANES_WORKLIST_R8")
            .ok()
            .as_deref()
            == Some("1");
        let tier_rows = worklist_tier_rows(rows, force_r5, force_r8);
        let Some(gu_t) = ops::t_lanes_gate_up_t8_tier_admits(tier_rows, h) else {
            return false;
        };
        let Some(dn_t) = ops::t_lanes_t8_tier_admits(tier_rows, inter) else {
            return false;
        };
        let (gu_ok, dn_ok) = if is_ts4e {
            let wl = ts4e_worklist_kernels(gpu);
            (
                wl.gu_wl_tiers[gu_t].0 != 0 || wl.gu_t8_tiers[gu_t].0 != 0,
                wl.dn_tiers[dn_t].0 != 0,
            )
        } else {
            (
                self.moe_expert_gate_up_worklist_t_lanes_rows_tiers[gu_t].0 != 0
                    || self.moe_expert_gate_up_worklist_t8_tiers[gu_t].0 != 0,
                self.moe_expert_down_act_shared_t_lanes_rows_t8_tiers[dn_t].0 != 0,
            )
        };
        gu_ok && dn_ok
    }

    /// Exact multi-row MoE on the unified (transposed) layout: one launch per
    /// projection, each distinct expert streamed once, every row byte-identical
    /// to the single-row `_t_lanes` kernel.
    #[allow(clippy::too_many_arguments)]
    fn qwen4_exact_t_lanes_rows(
        &self,
        input: DevicePtr,
        indices_dev: DevicePtr,
        (expert_gate_out, expert_up_out, expert_down_out): (DevicePtr, DevicePtr, DevicePtr),
        (shared_gate_out, shared_up_out, shared_down_out): (DevicePtr, DevicePtr, DevicePtr),
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let gate_t = self
            .gate_ptrs_t
            .as_ref()
            .expect("gate_ptrs_t under unified_t");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
        let down_t = self
            .down_ptrs_t
            .as_ref()
            .expect("down_ptrs_t under unified_t");
        let null_qw = QuantizedWeight::null();
        let skip_shared = ctx.config.ep_world_size > 1
            && ctx.config.ep_rank > 0
            && ctx.comm.is_some_and(|c| c.world_size() > 1);
        let sh_gate_t = if skip_shared {
            &null_qw
        } else {
            self.shared_gate_t.as_ref().unwrap_or(&null_qw)
        };
        let sh_up_t = if skip_shared {
            &null_qw
        } else {
            self.shared_up_t.as_ref().unwrap_or(&null_qw)
        };
        let sh_down_t = if skip_shared {
            &null_qw
        } else {
            self.shared_down_t.as_ref().unwrap_or(&null_qw)
        };
        let ob = ops::T_LANES_OUT_BLOCK;

        // Tier selection for [r3, r5, r8, r10] t8 / worklist kernels.
        // Only explicit `ATLAS_MOE_T_LANES_ROWS_T8_R5_ONLY=1` pins rows 4..=5 to
        // the standalone r5 tier; by default on the worklist path `2..=3` dispatch
        // to `r2/r3` (tier 0), `4..=10` to `r4/r5` (tier 1), and `11..=128` to `r8`
        // (tier 2) unless `ATLAS_MOE_T_LANES_WORKLIST_R8=1` requests `r8` from 6+.
        let force_r5 = std::env::var("ATLAS_MOE_T_LANES_ROWS_T8_R5_ONLY").is_ok_and(|v| v == "1");
        let force_r8 = std::env::var("ATLAS_MOE_T_LANES_WORKLIST_R8")
            .ok()
            .as_deref()
            == Some("1");
        let tier_rows = worklist_tier_rows(rows, force_r5, force_r8);
        let gu_tier = if force_r5 {
            ops::t_lanes_gate_up_t8_admits(rows, h).then_some(1usize)
        } else {
            ops::t_lanes_gate_up_t8_tier_admits(tier_rows, h)
        };
        let dn_tier = if force_r5 {
            ops::t_lanes_t8_admits(rows, inter).then_some(1usize)
        } else {
            ops::t_lanes_t8_tier_admits(tier_rows, inter)
        };

        let use_ts4e = self.routed_scale_layout == RoutedScaleLayout::Ts4e128;
        let ts4e_wl = use_ts4e.then(|| ts4e_worklist_kernels(ctx.gpu));
        let max_slots = rows * top_k;
        let e512_k = moe_build_unique_worklist_e512_kernel(ctx.gpu);
        let use_e512 = num_experts <= 512
            && rows <= WORKLIST_E512_KERNEL_MAX_ROWS
            && max_slots <= WORKLIST_E512_KERNEL_MAX_SLOTS.min(WORKLIST_MAX_SLOTS as u32)
            && e512_k.0 != 0;
        let use_r8 = !use_e512
            && rows <= 32
            && max_slots <= 320
            && self.moe_build_unique_worklist_r8_k.0 != 0;
        let use_worklist = moe_t_lanes_worklist_selected()
            && (use_e512 || use_r8)
            && gu_tier.is_some_and(|t| {
                if let Some(wl) = ts4e_wl {
                    wl.gu_wl_tiers[t].0 != 0 || wl.gu_t8_tiers[t].0 != 0
                } else {
                    self.moe_expert_gate_up_worklist_t_lanes_rows_tiers[t].0 != 0
                        || self.moe_expert_gate_up_worklist_t8_tiers[t].0 != 0
                }
            });

        let worklist_rmax: u32 = match gu_tier.unwrap_or(2) {
            0 => 3,
            1 => 5,
            2 => 8,
            _ => 10,
        };
        let routed_ctas = worklist_routed_ctas(rows, top_k, num_experts, worklist_rmax);
        let worklist_ptrs = if use_worklist {
            let base = self.moe_t8_worklist_buffer(ctx)?;
            let unique_first_slot = base;
            let unique_cnt = base.offset(WORKLIST_MAX_SLOTS * 4);
            let unique_flat = base.offset(WORKLIST_MAX_SLOTS * 2 * 4);
            if use_e512 {
                KernelLaunch::new(ctx.gpu, e512_k)
                    .grid([1, 1, 1])
                    .block([512, 1, 1])
                    .arg_ptr(indices_dev)
                    .arg_ptr(unique_first_slot)
                    .arg_ptr(unique_cnt)
                    .arg_ptr(unique_flat)
                    .arg_u32(rows)
                    .arg_u32(top_k)
                    .arg_u32(routed_ctas)
                    .arg_u32(worklist_rmax)
                    .arg_u32(num_experts)
                    .launch(stream)?;
            } else {
                KernelLaunch::new(ctx.gpu, self.moe_build_unique_worklist_r8_k)
                    .grid([1, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(indices_dev)
                    .arg_ptr(unique_first_slot)
                    .arg_ptr(unique_cnt)
                    .arg_ptr(unique_flat)
                    .arg_u32(rows)
                    .arg_u32(top_k)
                    .arg_u32(routed_ctas)
                    .arg_u32(worklist_rmax)
                    .launch(stream)?;
            }
            Some((unique_first_slot, unique_cnt, unique_flat))
        } else {
            None
        };

        if let (Some(tier), Some((unique_first_slot, unique_cnt, unique_flat))) =
            (gu_tier, worklist_ptrs)
        {
            let tier_rmax: u32 = match tier {
                0 => 3,
                1 => 5,
                2 => 8,
                _ => 10,
            };
            let shared_ctas = rows.div_ceil(tier_rmax);
            let (gu_wl_r2, gu_wl_r4, gu_wl_tier_k, gu_t8_tier_k) = if let Some(wl) = ts4e_wl {
                (
                    wl.gu_wl_r2,
                    wl.gu_wl_r4,
                    wl.gu_wl_tiers[tier],
                    wl.gu_t8_tiers[tier],
                )
            } else {
                (
                    self.moe_expert_gate_up_worklist_t_lanes_rows_r2_k,
                    self.moe_expert_gate_up_worklist_t_lanes_rows_r4_k,
                    self.moe_expert_gate_up_worklist_t_lanes_rows_tiers[tier],
                    self.moe_expert_gate_up_worklist_t8_tiers[tier],
                )
            };
            let prefer_t8 =
                moe_t_lanes_rows_gate_up_t8_selected() && rows <= NATIVE_TARGET_MOE_MAX_TILE as u32;
            let gu_mode = select_worklist_gate_up_mode(
                prefer_t8,
                gu_wl_tier_k.0 != 0,
                gu_t8_tier_k.0 != 0 && rows <= NATIVE_TARGET_MOE_MAX_TILE as u32,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "worklist gate/up kernel unavailable for tier={tier}, rows={rows}, ts4e={use_ts4e}"
                )
            })?;
            let (use_t8_gu, gu_wl_k, gu_out_block, gu_threads, gu_smem, gu_shared_ctas, gu_grid_z) =
                match gu_mode {
                    WorklistGateUpMode::UnfusedT8 => (
                        true,
                        gu_t8_tier_k,
                        ops::T_LANES_T8_OUT_BLOCK,
                        ops::T_LANES_T8_THREADS,
                        0,
                        shared_ctas,
                        2,
                    ),
                    WorklistGateUpMode::FusedTLanesRows => {
                        let (wl_k, smem_rows) = if !force_r5 && rows == 2 && gu_wl_r2.0 != 0 {
                            (gu_wl_r2, 2)
                        } else if !force_r5 && rows == 4 && gu_wl_r4.0 != 0 {
                            (gu_wl_r4, 4)
                        } else {
                            (gu_wl_tier_k, tier_rmax.min(8))
                        };
                        (
                            false,
                            wl_k,
                            ob,
                            ops::T_LANES_THREADS,
                            (smem_rows * h * 2) + (16 * 65 * 4),
                            rows.div_ceil(smem_rows),
                            1,
                        )
                    }
                };
            MOE_T_LANES_ROWS_GATE_UP_T8_LAUNCHED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_MOE_T_LANES_WORKLIST: worklist K-row gate_up launched (rows={rows}, tier={tier}, t8={use_t8_gu}, ts4e={use_ts4e})"
                );
            });
            KernelLaunch::new(ctx.gpu, gu_wl_k)
                .grid([
                    inter.div_ceil(gu_out_block),
                    routed_ctas + gu_shared_ctas,
                    gu_grid_z,
                ])
                .block([gu_threads, 1, 1])
                .shared_mem(gu_smem)
                .pdl(!use_t8_gu && rows <= 64)
                .arg_ptr(input)
                .arg_ptr(gate_t.packed_ptrs)
                .arg_ptr(gate_t.scale_ptrs)
                .arg_ptr(gate_t.scale2_vals)
                .arg_ptr(expert_gate_out)
                .arg_ptr(up_t.packed_ptrs)
                .arg_ptr(up_t.scale_ptrs)
                .arg_ptr(up_t.scale2_vals)
                .arg_ptr(expert_up_out)
                .arg_ptr(indices_dev)
                .arg_ptr(unique_first_slot)
                .arg_ptr(unique_cnt)
                .arg_ptr(unique_flat)
                .arg_ptr(sh_gate_t.weight)
                .arg_ptr(sh_gate_t.weight_scale)
                .arg_f32(sh_gate_t.weight_scale_2)
                .arg_ptr(shared_gate_out)
                .arg_ptr(sh_up_t.weight)
                .arg_ptr(sh_up_t.weight_scale)
                .arg_f32(sh_up_t.weight_scale_2)
                .arg_ptr(shared_up_out)
                .arg_u32(inter)
                .arg_u32(h)
                .arg_u32(top_k)
                .arg_u32(rows)
                .launch(stream)?;
        } else {
            anyhow::ensure!(
                self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
                "unported non-worklist _t_lanes_rows gate_up path requires FlatE4m3; set ATLAS_MOE_SCALE_TS4E=0"
            );
            anyhow::ensure!(
                rows <= 9,
                "non-worklist _t_lanes_rows gate_up requires rows <= T_LANES_RMAX (9), got {rows}"
            );
            let generic = std::env::var("ATLAS_MOE_T_LANES_ROWS_GENERIC").is_ok_and(|v| v == "1");
            let spec = (!generic)
                .then(|| {
                    self.t_lanes_rows_spec
                        .iter()
                        .find(|(cap, _, _)| rows <= *cap)
                })
                .flatten();
            let (gate_up_k, _, gate_up_smem) = match spec {
                Some(&(_, gu, dn)) => (gu, dn, rows * h * 2),
                None => (
                    self.moe_expert_gate_up_shared_t_lanes_rows_k,
                    self.moe_expert_silu_down_shared_t_lanes_rows_k,
                    0,
                ),
            };
            let non_wl_gu_tier = if force_r5 {
                ops::t_lanes_gate_up_t8_admits(rows, h).then_some(1usize)
            } else {
                ops::t_lanes_gate_up_t8_tier_admits(rows, h)
            };
            let gu_t8_k = non_wl_gu_tier
                .map(|t| self.moe_expert_gate_up_shared_t_lanes_rows_t8_tiers[t])
                .unwrap_or(KernelHandle(0));
            let gu_t8 = moe_t_lanes_rows_gate_up_t8_selected() && rows <= 10 && gu_t8_k.0 != 0;
            let (gate_up_k, gate_up_blocks, gate_up_threads, gate_up_smem) = if gu_t8 {
                MOE_T_LANES_ROWS_GATE_UP_T8_LAUNCHED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_MOE_T_LANES_ROWS_GATE_UP_T8: t8 K-row gate_up launched (rows={rows})"
                    );
                });
                (
                    gu_t8_k,
                    inter.div_ceil(ops::T_LANES_T8_OUT_BLOCK),
                    ops::T_LANES_T8_THREADS,
                    ops::t_lanes_gate_up_t8_smem_bytes(rows, h),
                )
            } else {
                (
                    gate_up_k,
                    inter.div_ceil(ob),
                    ops::T_LANES_THREADS,
                    gate_up_smem,
                )
            };
            KernelLaunch::new(ctx.gpu, gate_up_k)
                .grid([gate_up_blocks, rows * top_k + 1, 2])
                .block([gate_up_threads, 1, 1])
                .shared_mem(gate_up_smem)
                .arg_ptr(input)
                .arg_ptr(gate_t.packed_ptrs)
                .arg_ptr(gate_t.scale_ptrs)
                .arg_ptr(gate_t.scale2_vals)
                .arg_ptr(expert_gate_out)
                .arg_ptr(up_t.packed_ptrs)
                .arg_ptr(up_t.scale_ptrs)
                .arg_ptr(up_t.scale2_vals)
                .arg_ptr(expert_up_out)
                .arg_ptr(indices_dev)
                .arg_ptr(sh_gate_t.weight)
                .arg_ptr(sh_gate_t.weight_scale)
                .arg_f32(sh_gate_t.weight_scale_2)
                .arg_ptr(shared_gate_out)
                .arg_ptr(sh_up_t.weight)
                .arg_ptr(sh_up_t.weight_scale)
                .arg_f32(sh_up_t.weight_scale_2)
                .arg_ptr(shared_up_out)
                .arg_u32(inter)
                .arg_u32(h)
                .arg_u32(top_k)
                .arg_u32(rows)
                .launch(stream)?;
        }
        spark_runtime::cublaslt::stage_timer::mark("v_gate_up", stream);

        let dn_act_k_opt = if let (Some(tier), Some(wl)) = (dn_tier, ts4e_wl) {
            let k = wl.dn_tiers[tier];
            (k.0 != 0).then_some(k)
        } else if let Some(tier) = dn_tier
            && moe_t_lanes_rows_down_t8_selected()
        {
            let k = self.moe_expert_down_act_shared_t_lanes_rows_t8_tiers[tier];
            (k.0 != 0).then_some(k)
        } else {
            None
        };
        if let (Some(tier), Some((unique_first_slot, unique_cnt, unique_flat)), Some(dn_act_k)) =
            (dn_tier, worklist_ptrs, dn_act_k_opt)
            && moe_t_lanes_down_act_selected()
            && self.moe_silu_mul_rows_f32_k.0 != 0
        {
            let tier_rmax: u32 = match tier {
                0 => 3,
                1 => 5,
                2 => 8,
                _ => 10,
            };
            let shared_ctas = rows.div_ceil(tier_rmax);
            let act_base = self.moe_t8_act_buffer(ctx, top_k as usize, inter as usize)?;
            let routed_elems = rows * top_k * inter;
            let shared_elems = rows * inter;
            let sh_act_out = act_base.offset(routed_elems as usize * 4);
            let total_elems = routed_elems + shared_elems;
            KernelLaunch::new(ctx.gpu, self.moe_silu_mul_rows_f32_k)
                .grid([total_elems.div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .pdl(rows <= 64)
                .arg_ptr(expert_gate_out)
                .arg_ptr(expert_up_out)
                .arg_ptr(act_base)
                .arg_ptr(shared_gate_out)
                .arg_ptr(shared_up_out)
                .arg_ptr(sh_act_out)
                .arg_u32(routed_elems)
                .arg_u32(shared_elems)
                .launch(stream)?;
            MOE_T_LANES_ROWS_DOWN_T8_LAUNCHED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_MOE_T_LANES_DOWN_ACT: worklist+act t8 K-row down launched (rows={rows}, tier={tier}, ts4e={use_ts4e})"
                );
            });
            return KernelLaunch::new(ctx.gpu, dn_act_k)
                .grid([
                    h.div_ceil(ops::T_LANES_T8_OUT_BLOCK),
                    routed_ctas + shared_ctas,
                    1,
                ])
                .block([ops::T_LANES_T8_THREADS, 1, 1])
                .pdl(rows <= 64)
                .arg_ptr(act_base)
                .arg_ptr(down_t.packed_ptrs)
                .arg_ptr(down_t.scale_ptrs)
                .arg_ptr(down_t.scale2_vals)
                .arg_ptr(indices_dev)
                .arg_ptr(unique_first_slot)
                .arg_ptr(unique_cnt)
                .arg_ptr(unique_flat)
                .arg_ptr(expert_down_out)
                .arg_ptr(sh_act_out)
                .arg_ptr(sh_down_t.weight)
                .arg_ptr(sh_down_t.weight_scale)
                .arg_f32(sh_down_t.weight_scale_2)
                .arg_ptr(shared_down_out)
                .arg_u32(h)
                .arg_u32(inter)
                .arg_u32(top_k)
                .arg_u32(rows)
                .launch(stream);
        }

        anyhow::ensure!(
            self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
            "unported non-worklist _t_lanes_rows silu_down path requires FlatE4m3; set ATLAS_MOE_SCALE_TS4E=0"
        );
        anyhow::ensure!(
            rows <= 9,
            "non-worklist _t_lanes_rows silu_down requires rows <= T_LANES_RMAX (9), got {rows}"
        );
        let generic = std::env::var("ATLAS_MOE_T_LANES_ROWS_GENERIC").is_ok_and(|v| v == "1");
        let spec = (!generic)
            .then(|| {
                self.t_lanes_rows_spec
                    .iter()
                    .find(|(cap, _, _)| rows <= *cap)
            })
            .flatten();
        let down_k = match spec {
            Some(&(_, _, dn)) => dn,
            None => self.moe_expert_silu_down_shared_t_lanes_rows_k,
        };
        let non_wl_dn_tier = if force_r5 {
            ops::t_lanes_t8_admits(rows, inter).then_some(1usize)
        } else {
            ops::t_lanes_t8_tier_admits(rows, inter)
        };
        let t8_k = non_wl_dn_tier
            .map(|t| self.moe_expert_silu_down_shared_t_lanes_rows_t8_tiers[t])
            .unwrap_or(KernelHandle(0));
        let t8 = moe_t_lanes_rows_down_t8_selected() && rows <= 10 && t8_k.0 != 0;
        let (down_k, down_blocks, down_threads, down_smem) = if t8 {
            MOE_T_LANES_ROWS_DOWN_T8_LAUNCHED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_MOE_T_LANES_ROWS_DOWN_T8: t8 K-row down launched (rows={rows})"
                );
            });
            (
                t8_k,
                h.div_ceil(ops::T_LANES_T8_OUT_BLOCK),
                ops::T_LANES_T8_THREADS,
                ops::t_lanes_t8_smem_bytes(rows, inter),
            )
        } else {
            (
                down_k,
                h.div_ceil(ob),
                ops::T_LANES_THREADS,
                rows * inter * 4,
            )
        };
        KernelLaunch::new(ctx.gpu, down_k)
            .grid([down_blocks, rows * top_k + 1, 1])
            .block([down_threads, 1, 1])
            .shared_mem(down_smem)
            .arg_ptr(expert_gate_out)
            .arg_ptr(expert_up_out)
            .arg_ptr(down_t.packed_ptrs)
            .arg_ptr(down_t.scale_ptrs)
            .arg_ptr(down_t.scale2_vals)
            .arg_ptr(expert_down_out)
            .arg_ptr(indices_dev)
            .arg_ptr(shared_gate_out)
            .arg_ptr(shared_up_out)
            .arg_ptr(sh_down_t.weight)
            .arg_ptr(sh_down_t.weight_scale)
            .arg_f32(sh_down_t.weight_scale_2)
            .arg_ptr(shared_down_out)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(top_k)
            .arg_u32(rows)
            .launch(stream)
    }

    fn moe_t8_worklist_buffer(&self, ctx: &ForwardContext) -> Result<DevicePtr> {
        if let Some(&ptr) = self.moe_t8_worklist_buf.get() {
            return Ok(ptr);
        }
        // WORKLIST_MAX_SLOTS * (4 + 4 + 10 * 4) = 122,880 B (120 KiB, 4 KiB-aligned).
        let ptr = ctx.gpu.alloc(WORKLIST_BUFFER_BYTES)?;
        match self.moe_t8_worklist_buf.set(ptr) {
            Ok(()) => Ok(ptr),
            Err(_) => {
                let _ = ctx.gpu.free(ptr);
                Ok(*self
                    .moe_t8_worklist_buf
                    .get()
                    .expect("moe_t8_worklist_buf initialized by racing thread"))
            }
        }
    }

    fn moe_t8_act_buffer(
        &self,
        ctx: &ForwardContext,
        top_k: usize,
        inter: usize,
    ) -> Result<DevicePtr> {
        let bytes = (NATIVE_TARGET_MOE_MAX_TILE * top_k + NATIVE_TARGET_MOE_MAX_TILE)
            * inter
            * std::mem::size_of::<f32>();
        if let Some(&(cap, ptr)) = self.moe_t8_act_buf.get() {
            anyhow::ensure!(
                cap >= bytes,
                "moe_t8_act_buf capacity {cap} B is smaller than requested {bytes} B"
            );
            return Ok(ptr);
        }
        let ptr = ctx.gpu.alloc(bytes)?;
        match self.moe_t8_act_buf.set((bytes, ptr)) {
            Ok(()) => Ok(ptr),
            Err(_) => {
                let _ = ctx.gpu.free(ptr);
                let &(cap, existing) = self
                    .moe_t8_act_buf
                    .get()
                    .expect("moe_t8_act_buf initialized by racing thread");
                anyhow::ensure!(
                    cap >= bytes,
                    "moe_t8_act_buf capacity {cap} B is smaller than requested {bytes} B"
                );
                Ok(existing)
            }
        }
    }
}

/// `ATLAS_MOE_T_LANES_WORKLIST=1` (default ON unless `=0`): warp-ballot worklist compaction
/// for the t8 multi-row MoE path.
fn moe_t_lanes_worklist_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_T_LANES_WORKLIST").ok().as_deref() != Some("0"))
}

/// `ATLAS_MOE_T_LANES_DOWN_ACT=1` (default ON unless `=0`): pre-materialized FP32
/// `SiLU(gate) * up` activation rows for the worklist t8 down path.
fn moe_t_lanes_down_act_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MOE_T_LANES_DOWN_ACT").ok().as_deref() != Some("0"))
}

/// `ATLAS_MOE_T_LANES_ROWS_GATE_UP_T8=1`: eight-threads-per-group K-row gate/up
/// on the unified layout (byte-identical to the `_t_lanes_rows` gate/up).
/// Default OFF because `_t_lanes_rows_r{2..8}` beats `_t8_r5_k160` by 2.5..2.9 ms at k=4..5.
fn moe_t_lanes_rows_gate_up_t8_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_MOE_T_LANES_ROWS_GATE_UP_T8")
                .ok()
                .as_deref(),
            Some("1")
        )
    })
}

/// Receipt for the first t8 gate/up launch.
static MOE_T_LANES_ROWS_GATE_UP_T8_LAUNCHED: std::sync::Once = std::sync::Once::new();

/// `ATLAS_MOE_T_LANES_ROWS_DOWN_T8`: eight-threads-per-group K-row down on
/// the unified layout (verify-cost tlanes_t8_bench; byte-identical to the
/// `_t_lanes_rows` down).
///
/// Explicit `"1"` enables and `"0"` disables; when unset, implicitly turns ON
/// whenever `ATLAS_QWEN4_VERIFY_FAST_HOST=1`.
fn moe_t_lanes_rows_down_t8_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        match std::env::var("ATLAS_MOE_T_LANES_ROWS_DOWN_T8")
            .ok()
            .as_deref()
        {
            Some("1") => true,
            Some("0") => false,
            _ => crate::layers::qwen4_verify_fast_host_selected(),
        }
    })
}

/// Receipt for the first t8 down launch (the env read alone proves nothing).
static MOE_T_LANES_ROWS_DOWN_T8_LAUNCHED: std::sync::Once = std::sync::Once::new();

fn qwen4_moe_dedup_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_MOE_DEDUP").ok().as_deref() == Some("1");
        if on {
            tracing::info!("ENGAGED ATLAS_QWEN4_MOE_DEDUP: expert-deduplicated exact-row MoE");
        }
        on
    })
}

/// `ATLAS_QWEN4_MOE_DOWN_ACT=1`: activation-once dedup down (needs ATLAS_QWEN4_MOE_DEDUP=1).
fn qwen4_moe_down_act_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_MOE_DOWN_ACT").ok().as_deref() == Some("1");
        if on {
            tracing::info!(
                "ENGAGED ATLAS_QWEN4_MOE_DOWN_ACT: activation-once exact dedup MoE down"
            );
        }
        on
    })
}

impl MoeLayer {
    /// FP32 activation rows for the act down, sized for the r16 cap and
    /// allocated once per layer on first use. Verify graphs capture in
    /// relaxed mode (cuStreamBeginCapture mode 2), which permits the
    /// allocation, as for the GDN lazy-commit retain buffer.
    fn moe_down_act_buffer(
        &self,
        ctx: &ForwardContext,
        top_k: usize,
        inter: usize,
    ) -> Result<DevicePtr> {
        if let Some(&ptr) = self.moe_down_act_buf.get() {
            return Ok(ptr);
        }
        let bytes = ops::moe_down_act_elems(ops::MOE_DOWN_ACT_MAX_ROWS, top_k, inter)
            * std::mem::size_of::<f32>();
        let ptr = ctx.gpu.alloc(bytes)?;
        match self.moe_down_act_buf.set(ptr) {
            Ok(()) => Ok(ptr),
            Err(_) => {
                let _ = ctx.gpu.free(ptr);
                Ok(*self
                    .moe_down_act_buf
                    .get()
                    .expect("moe_down_act_buf initialized by racing thread"))
            }
        }
    }
}

/// `ATLAS_QWEN4_MOE_DEDUP_CHECK=1` (eager only): run the reference kernels and
/// the dedup twins on every call and compare their outputs byte for byte.
fn qwen4_moe_dedup_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4_MOE_DEDUP_CHECK").ok().as_deref() == Some("1"))
}

fn snapshot(
    ctx: &ForwardContext,
    stream: u64,
    regions: &[(DevicePtr, usize)],
) -> Result<Vec<Vec<u8>>> {
    ctx.gpu.synchronize(stream)?;
    regions
        .iter()
        .map(|&(ptr, bytes)| {
            let mut buf = vec![0u8; bytes];
            ctx.gpu.copy_d2h(ptr, &mut buf)?;
            Ok(buf)
        })
        .collect()
}

fn record_dedup_check(
    rows: usize,
    reference_up: &[Vec<u8>],
    dedup_up: &[Vec<u8>],
    reference_down: &[Vec<u8>],
    dedup_down: &[Vec<u8>],
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static BAD_CALLS: AtomicU64 = AtomicU64::new(0);
    let differing = |a: &[Vec<u8>], b: &[Vec<u8>]| -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
            .sum()
    };
    let up = differing(reference_up, dedup_up);
    let dn = differing(reference_down, dedup_down);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if up + dn > 0 {
        let bad = BAD_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            rows,
            up_bytes = up,
            down_bytes = dn,
            bad,
            calls,
            "MOE_DEDUP_CHECK mismatch"
        );
    }
    if calls.is_power_of_two() || calls.is_multiple_of(4096) {
        tracing::info!(
            calls,
            bad_calls = BAD_CALLS.load(Ordering::Relaxed),
            "MOE_DEDUP_CHECK summary"
        );
    }
}

#[cfg(test)]
mod native_target_tile_tests {
    use super::{NATIVE_TARGET_MOE_MAX_TILE, native_target_moe_tile_width, worklist_routed_ctas};

    #[test]
    fn c2_c128_partition_into_exact_non_singleton_tiles() {
        for rows in 2..=crate::layers::QWEN4_NATIVE_MAX_ROWS {
            let mut remaining = rows;
            let mut covered = 0usize;
            while remaining > 0 {
                let tile = native_target_moe_tile_width(remaining).unwrap();
                assert!((2..=NATIVE_TARGET_MOE_MAX_TILE).contains(&tile));
                covered += tile;
                remaining -= tile;
            }
            assert_eq!(covered, rows, "C{rows}");
        }
    }

    #[test]
    fn partition_boundaries_are_stable_and_fail_closed() {
        assert_eq!(native_target_moe_tile_width(9), Ok(9));
        assert_eq!(native_target_moe_tile_width(10), Ok(10));
        assert_eq!(native_target_moe_tile_width(15), Ok(15));
        assert_eq!(native_target_moe_tile_width(20), Ok(20));
        assert_eq!(native_target_moe_tile_width(32), Ok(32));
        assert_eq!(native_target_moe_tile_width(33), Ok(33));
        assert_eq!(native_target_moe_tile_width(64), Ok(64));
        assert_eq!(native_target_moe_tile_width(80), Ok(80));
        assert_eq!(native_target_moe_tile_width(128), Ok(128));
        assert_eq!(native_target_moe_tile_width(192), Ok(192));
        assert_eq!(native_target_moe_tile_width(256), Ok(256));
        for rows in [0, 1, 257, usize::MAX] {
            assert!(native_target_moe_tile_width(rows).is_err());
        }
        assert_eq!(super::WORKLIST_MAX_SLOTS, 2560);
        assert_eq!(super::WORKLIST_BUFFER_BYTES, 122_880);
        const {
            assert!(
                super::WORKLIST_BUFFER_BYTES >= super::WORKLIST_MAX_SLOTS * 12 * 4,
                "worklist buffer must cover unique_first_slot + unique_cnt + unique_flat"
            );
        }
    }

    #[test]
    fn worklist_routed_ctas_bounds_unique_groups_tightly() {
        // Small rows: total < num_experts, capped at total = rows * top_k.
        assert_eq!(worklist_routed_ctas(5, 10, 512, 5), 50);
        assert_eq!(worklist_routed_ctas(32, 10, 512, 8), 320);
        // Wide rows: total > num_experts, tightened to num_experts + total / rmax.
        assert_eq!(worklist_routed_ctas(64, 10, 512, 8), 592);
        assert_eq!(worklist_routed_ctas(80, 10, 512, 8), 612);
        assert_eq!(worklist_routed_ctas(128, 10, 512, 8), 672);
        assert_eq!(worklist_routed_ctas(256, 10, 512, 8), 832);
    }

    #[test]
    fn non_worklist_fallback_decomposes_10_to_256_into_rmax9_tiles() {
        for rows in 2..=NATIVE_TARGET_MOE_MAX_TILE {
            let mut rem = rows;
            let mut sum = 0usize;
            while rem > 0 {
                let tile = super::non_worklist_subtile(rem);
                assert!((2..=9).contains(&tile), "rows={rows} rem={rem} tile={tile}");
                sum += tile;
                rem -= tile;
            }
            assert_eq!(sum, rows);
        }
    }

    #[test]
    fn compact_token_to_perm_offset_fits_gate_logits_for_verify_batches() {
        for rows in [
            6usize, 7, 8, 10, 12, 14, 15, 16, 20, 40, 64, 80, 128, 192, 256,
        ] {
            let expanded = rows * 10;
            let gate_logits_bytes = rows * 512 * 2;
            for gatepre in [false, true] {
                let off = super::compact_token_to_perm_offset(rows, gatepre);
                assert!(
                    off >= expanded * 8 + 513 * 4,
                    "rows={rows} gatepre={gatepre} perm_off={off} aliases expert_offsets"
                );
                assert!(
                    off + expanded * 4 <= gate_logits_bytes,
                    "rows={rows} gatepre={gatepre} perm_off={off} overflows gate_logits ({gate_logits_bytes})"
                );
            }
        }
    }

    #[test]
    fn verify_tc_moe_output_disjoint_detects_aliasing_and_accepts_disjoint_arenas() {
        use spark_runtime::gpu::DevicePtr;
        let rows = 80usize;
        let out_bytes = rows * 2560 * 2;
        let output = DevicePtr(0x1000_0000);
        let disjoint_peers = [
            (DevicePtr(0x2000_0000), out_bytes),
            (DevicePtr(0x3000_0000), 86_128),
            (DevicePtr(0x4000_0000), 4),
        ];
        assert!(super::verify_tc_moe_output_disjoint(
            output,
            rows,
            &disjoint_peers
        ));
        // Direct overlap with output must fail closed.
        let overlapping = [
            (DevicePtr(0x2000_0000), out_bytes),
            (output.offset(out_bytes - 16), 64),
        ];
        assert!(!super::verify_tc_moe_output_disjoint(
            output,
            rows,
            &overlapping
        ));
        // Null or misaligned output must fail closed.
        assert!(!super::verify_tc_moe_output_disjoint(
            DevicePtr::NULL,
            rows,
            &disjoint_peers
        ));
        assert!(!super::verify_tc_moe_output_disjoint(
            DevicePtr(0x1000_0004),
            rows,
            &disjoint_peers
        ));
    }

    #[test]
    fn worklist_tier_selects_r5_through_rows_10_and_r8_above() {
        use crate::layers::ops;
        for rows in 2..=3u32 {
            let eff = super::worklist_tier_rows(rows, false, false);
            assert_eq!(
                ops::t_lanes_gate_up_t8_tier_admits(eff, 2560),
                Some(0),
                "rows={rows}"
            );
            assert_eq!(
                ops::t_lanes_t8_tier_admits(eff, 640),
                Some(0),
                "rows={rows}"
            );
        }
        // rows=4..=10 (including C=1 K=5 rows=5 and C=2 K=5 rows=10) select tier=1 (r5).
        for rows in 4..=10u32 {
            let eff = super::worklist_tier_rows(rows, false, false);
            assert_eq!(
                ops::t_lanes_gate_up_t8_tier_admits(eff, 2560),
                Some(1),
                "rows={rows}"
            );
            assert_eq!(
                ops::t_lanes_t8_tier_admits(eff, 640),
                Some(1),
                "rows={rows}"
            );
        }
        // Explicit WORKLIST_R8=1 restores tier=2 (r8) for 6..=10.
        for rows in 6..=10u32 {
            let eff = super::worklist_tier_rows(rows, false, true);
            assert_eq!(
                ops::t_lanes_gate_up_t8_tier_admits(eff, 2560),
                Some(2),
                "r8 rows={rows}"
            );
            assert_eq!(
                ops::t_lanes_t8_tier_admits(eff, 640),
                Some(2),
                "r8 rows={rows}"
            );
        }
        // rows=11..=128 select tier=2 (r8).
        for rows in [11u32, 15, 20, 40, 80, 128] {
            let eff = super::worklist_tier_rows(rows, false, false);
            assert_eq!(
                ops::t_lanes_gate_up_t8_tier_admits(eff, 2560),
                Some(2),
                "rows={rows}"
            );
            assert_eq!(
                ops::t_lanes_t8_tier_admits(eff, 640),
                Some(2),
                "rows={rows}"
            );
        }
    }

    #[test]
    fn worklist_gate_up_prefers_fused_and_falls_back_to_t8() {
        use super::{WorklistGateUpMode, select_worklist_gate_up_mode};
        // Default: fused t_lanes_rows (grid_z=1, OPT=4, PDL) when available.
        assert_eq!(
            select_worklist_gate_up_mode(false, true, true),
            Some(WorklistGateUpMode::FusedTLanesRows)
        );
        assert_eq!(
            select_worklist_gate_up_mode(false, true, false),
            Some(WorklistGateUpMode::FusedTLanesRows)
        );
        // Fallback to unfused t8 when fused handle is absent.
        assert_eq!(
            select_worklist_gate_up_mode(false, false, true),
            Some(WorklistGateUpMode::UnfusedT8)
        );
        // Explicit prefer_t8 selects t8 when available, else falls back to fused.
        assert_eq!(
            select_worklist_gate_up_mode(true, true, true),
            Some(WorklistGateUpMode::UnfusedT8)
        );
        assert_eq!(
            select_worklist_gate_up_mode(true, true, false),
            Some(WorklistGateUpMode::FusedTLanesRows)
        );
        assert_eq!(select_worklist_gate_up_mode(false, false, false), None);
    }

    #[test]
    fn batched_verify_tc_moe_scope_guard_restores_flags_and_flushes_at_final_layer() {
        assert!(!super::batched_verify_tc_moe_status_async_active());
        assert!(!super::batched_verify_tc_moe_xq_prequant_active());
        assert!(!super::batched_verify_tc_moe_fused_blend_active());
        {
            let _outer = super::BatchedVerifyTcMoeScopeGuard::enter(true, false, true);
            assert!(super::batched_verify_tc_moe_status_async_active());
            assert!(!super::batched_verify_tc_moe_xq_prequant_active());
            assert!(super::batched_verify_tc_moe_fused_blend_active());
            {
                let _inner = super::BatchedVerifyTcMoeScopeGuard::enter(false, true, false);
                assert!(!super::batched_verify_tc_moe_status_async_active());
                assert!(super::batched_verify_tc_moe_xq_prequant_active());
                assert!(!super::batched_verify_tc_moe_fused_blend_active());
            }
            assert!(super::batched_verify_tc_moe_status_async_active());
            assert!(!super::batched_verify_tc_moe_xq_prequant_active());
            assert!(super::batched_verify_tc_moe_fused_blend_active());
        }
        assert!(!super::batched_verify_tc_moe_status_async_active());
        assert!(!super::batched_verify_tc_moe_xq_prequant_active());
        assert!(!super::batched_verify_tc_moe_fused_blend_active());

        for pending in 1..48 {
            assert!(!super::should_flush_batched_verify_status(pending, 48));
        }
        assert!(super::should_flush_batched_verify_status(48, 48));
        assert!(super::should_flush_batched_verify_status(1, 0));
        assert!(!super::should_flush_batched_verify_status(1999, 4096));
        assert!(super::should_flush_batched_verify_status(2000, 4096));

        assert_eq!(
            super::parse_moe_shared_overlap_override(None),
            super::SharedOverlapOverride::Auto
        );
        assert_eq!(
            super::parse_moe_shared_overlap_override(Some("")),
            super::SharedOverlapOverride::Auto
        );
        assert_eq!(
            super::parse_moe_shared_overlap_override(Some("1")),
            super::SharedOverlapOverride::ForceOn
        );
        assert_eq!(
            super::parse_moe_shared_overlap_override(Some(" 1 ")),
            super::SharedOverlapOverride::ForceOn
        );
        assert_eq!(
            super::parse_moe_shared_overlap_override(Some("0")),
            super::SharedOverlapOverride::ForceOff
        );
        assert_eq!(
            super::parse_moe_shared_overlap_override(Some(" 0 ")),
            super::SharedOverlapOverride::ForceOff
        );
        // Auto on GB10 (<=64 SMs): serialize shared_fast_v2 by default while keeping legacy shared_w4a8 overlapped.
        assert!(!super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::Auto,
            true
        ));
        assert!(super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::Auto,
            false
        ));
        // Auto on SM120 (>64 SMs): enable shared_fast_v2 overlap concurrently on prefill_stream;
        // GB10 (<=64 SMs) remains serialized.
        for gb10_sms in [48u32, 64u32] {
            assert!(!super::moe_shared_overlap_enabled_for_path_and_sm(
                super::SharedOverlapOverride::Auto,
                true,
                gb10_sms
            ));
            assert!(super::moe_shared_overlap_enabled_for_path_and_sm(
                super::SharedOverlapOverride::Auto,
                false,
                gb10_sms
            ));
        }
        for sm120_sms in [65u32, 148u32, 188u32] {
            assert!(super::moe_shared_overlap_enabled_for_path_and_sm(
                super::SharedOverlapOverride::Auto,
                true,
                sm120_sms
            ));
            assert!(super::moe_shared_overlap_enabled_for_path_and_sm(
                super::SharedOverlapOverride::Auto,
                false,
                sm120_sms
            ));
            assert!(!super::moe_shared_overlap_enabled_for_path_and_sm(
                super::SharedOverlapOverride::ForceOff,
                true,
                sm120_sms
            ));
        }
        // Explicit =1 enables overlap on both paths; explicit =0 disables overlap on both paths.
        assert!(super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::ForceOn,
            true
        ));
        assert!(super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::ForceOn,
            false
        ));
        assert!(!super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::ForceOff,
            true
        ));
        assert!(!super::moe_shared_overlap_enabled_for_path(
            super::SharedOverlapOverride::ForceOff,
            false
        ));

        assert!(super::moe_shared_graph_overlap_enabled(None));
        assert!(super::moe_shared_graph_overlap_enabled(Some("1")));
        assert!(!super::moe_shared_graph_overlap_enabled(Some("0")));

        assert!(super::decode_shared_fast_v2_enabled(None));
        assert!(super::decode_shared_fast_v2_enabled(Some("")));
        assert!(super::decode_shared_fast_v2_enabled(Some("1")));
        assert!(!super::decode_shared_fast_v2_enabled(Some("0")));
        assert!(!super::decode_shared_fast_v2_enabled(Some(" 0 ")));

        // Eager overlap depends only on master overlap toggle; graph-captured overlap
        // requires both master overlap and graph overlap toggles (both default ON).
        assert!(super::moe_shared_overlap_admits_capture(true, true, false));
        assert!(super::moe_shared_overlap_admits_capture(true, true, true));
        assert!(super::moe_shared_overlap_admits_capture(true, false, false));
        assert!(!super::moe_shared_overlap_admits_capture(true, false, true));
        assert!(!super::moe_shared_overlap_admits_capture(
            false, true, false
        ));
        assert!(!super::moe_shared_overlap_admits_capture(false, true, true));

        // When v33_thin_sk is active at rows <= 16 on SM120, shared_overlap defaults OFF
        // (eliminating 48 cross-stream graph fork/joins fighting 1,128 thin-SK CTAs),
        // while staying overridable via ATLAS_QWEN4_SHARED_OVERLAP_THIN_SK=1 or ForceOn,
        // and keeping rows > 16 and non-thin-SK / GB10 paths unchanged.
        assert!(!super::parse_moe_shared_overlap_thin_sk(None));
        assert!(!super::parse_moe_shared_overlap_thin_sk(Some("0")));
        assert!(super::parse_moe_shared_overlap_thin_sk(Some("1")));
        assert!(super::parse_moe_shared_overlap_thin_sk(Some(" true ")));
        assert!(!super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            188,
            true,
            8,
            false,
        ));
        assert!(!super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            188,
            true,
            16,
            false,
        ));
        assert!(super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            188,
            true,
            17,
            false,
        ));
        assert!(super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            188,
            true,
            8,
            true,
        ));
        assert!(super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::ForceOn,
            true,
            188,
            true,
            8,
            false,
        ));
        assert!(super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            188,
            false,
            8,
            false,
        ));
        assert!(!super::moe_shared_overlap_enabled_for_thin_sk_and_sm(
            super::SharedOverlapOverride::Auto,
            true,
            48,
            false,
            8,
            false,
        ));

        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        super::BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(|c| c.set(17));
        super::BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY.with(|c| c.set(0x1234));
        // During CUDA graph capture, flush is a capture-safe no-op (no D2H sync).
        spark_runtime::gpu_capture_region::set_active_capture_region(Some("unit_test_verify_k5"));
        super::flush_batched_verify_moe_status(&gpu, 0).expect("capture-safe no-op flush");
        assert_eq!(
            super::BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(std::cell::Cell::get),
            17
        );
        spark_runtime::gpu_capture_region::set_active_capture_region(None);
        super::flush_batched_verify_moe_status(&gpu, 0).expect("flush on mock backend");
        assert_eq!(
            super::BATCHED_VERIFY_STATUS_PENDING_LAYERS.with(std::cell::Cell::get),
            0
        );
        assert_eq!(
            super::BATCHED_VERIFY_STATUS_FIRST_LAYER_KEY.with(std::cell::Cell::get),
            0
        );
    }

    #[test]
    fn batched_verify_router_tc_admits_wide_batches_and_preserves_c1_exact_parity() {
        use super::batched_verify_router_tc_admits;

        // Rows <= 3 stay on the 0-ULP exact GEMV path even when scope is active
        // or an override requests a smaller threshold.
        for rows in 0..4 {
            assert!(
                !batched_verify_router_tc_admits(true, true, None, rows, 512, 2560),
                "rows={rows} must stay exact"
            );
            assert!(
                !batched_verify_router_tc_admits(true, true, Some(2), rows, 512, 2560),
                "rows={rows} must stay exact even with low override"
            );
        }

        // Batched verify rows in 4..=256 (including C=2 r=4..8 and C=64 K=4 / C=32 K=8)
        // admit warp-split-K NVFP4 TC routing when scoped and enabled;
        // C=1 (scope_active = false) always stays on the exact 0-ULP scalar path.
        for rows in [4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 40, 80, 120, 128, 192, 256] {
            assert!(
                batched_verify_router_tc_admits(true, true, None, rows, 512, 2560),
                "rows={rows}"
            );
            assert!(!batched_verify_router_tc_admits(
                false, true, None, rows, 512, 2560
            ));
            assert!(!batched_verify_router_tc_admits(
                true, false, None, rows, 512, 2560
            ));
        }

        // Higher explicit override raises the threshold cleanly.
        assert!(!batched_verify_router_tc_admits(
            true,
            true,
            Some(32),
            20,
            512,
            2560
        ));
        assert!(batched_verify_router_tc_admits(
            true,
            true,
            Some(32),
            32,
            512,
            2560
        ));
        assert!(batched_verify_router_tc_admits(
            true,
            true,
            Some(32),
            80,
            512,
            2560
        ));

        // Tile ceiling and shape validation (coalesced kernel requires hidden % 32 == 0).
        assert!(!batched_verify_router_tc_admits(
            true, true, None, 257, 512, 2560
        ));
        assert!(!batched_verify_router_tc_admits(
            true, true, None, 80, 0, 2560
        ));
        assert!(!batched_verify_router_tc_admits(
            true, true, None, 80, 512, 0
        ));
        assert!(!batched_verify_router_tc_admits(
            true, true, None, 80, 512, 2552
        ));
        assert!(!batched_verify_router_tc_admits(
            true, true, None, 80, 512, 48
        ));

        use super::batched_verify_router_tc_admits_with_policy;
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;
        for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            assert!(
                !batched_verify_router_tc_admits_with_policy(
                    true,
                    true,
                    None,
                    m,
                    512,
                    2560,
                    DeterministicBatchPolicy::Enabled,
                ),
                "router TC must stay disabled at M={m}"
            );
        }
        assert!(batched_verify_router_tc_admits_with_policy(
            true,
            true,
            None,
            8,
            512,
            2560,
            DeterministicBatchPolicy::Disabled,
        ));
    }
}
