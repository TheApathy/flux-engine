// SPDX-License-Identifier: AGPL-3.0-only

//! MoE (Mixture of Experts) FFN component.
//!
//! Batched expert dispatch: top-K experts run in 2 fused kernel launches
//! (gate+up, silu+down) instead of 10 × 5 individual launches. Expert indices
//! and weights stay on device — zero D2H synchronization.

// PR #74 brought in `dump.rs` and `forward_prefill_routed.rs` as new
// submodules. We took ours on this file but need to declare the new files
// so their pub items are reachable.
pub(crate) mod bounded_transpose;
mod dump;
mod forward_prefill_routed;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::{ForwardContext, MoeStreamTransposeScratch};
use crate::layers::ops;
use crate::weight_map::{ExpertWeight, Fp8ExpertWeight, Fp8Weight, MoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// [num_experts] u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// [num_experts] u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// [num_experts] f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// [num_experts] u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// [num_experts] u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Scale layout for transposed routed NVFP4 expert weights (`gate_ptrs_t`, `up_ptrs_t`, `down_ptrs_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutedScaleLayout {
    /// Standard transposed flat E4M3 scales (`[K/16, N]`, 1 byte per 16-element group).
    #[default]
    FlatE4m3,
    /// Atlas-TS4E-128 compressed per-expert blobs stored directly at `scale_ptrs[e]`.
    Ts4e128,
}

/// Returns `true` when `ATLAS_MOE_SCALE_TS4E=1` is set (default `false` until live parity gate passes).
pub fn atlas_moe_scale_ts4e_enabled() -> bool {
    std::env::var("ATLAS_MOE_SCALE_TS4E").ok().as_deref() == Some("1")
}

/// MoE feed-forward network component.
///
/// Not a `TransformerLayer` — used as a component inside layers
/// for the FFN/MoE block after post-attention norm.
pub struct MoeLayer {
    pub weights: MoeWeights,
    /// Active scale layout for transposed routed expert pointer tables.
    pub routed_scale_layout: RoutedScaleLayout,
    // NVFP4-quantized gate weight (quarters bandwidth for routing)
    gate_nvfp4: Option<QuantizedWeight>,
    /// Pre-expert norm: applied to input AFTER routing but BEFORE expert dispatch.
    /// Gemma-4 26B: router sees raw residual, experts see pre_feedforward_layernorm_2(residual).
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pre_expert_norm_k: spark_runtime::gpu::KernelHandle,
    dense_gemv: KernelHandle,
    w4a16_gemv: KernelHandle,
    /// Row-parallel K1 GEMV (`w4a16_gemv_rows`) for the exact-rows router; zero if absent.
    w4a16_gemv_rows: KernelHandle,
    w4a16_gemv_exact_m4: KernelHandle,
    w4a16_gemv_exact_m8: KernelHandle,
    w4a16_gemv_exact_m17: KernelHandle,
    w4a16_gemv_exact_m32: KernelHandle,
    w4a16_gemm: KernelHandle,
    dense_gemm: KernelHandle,
    router_bf16_tk64: KernelHandle,
    moe_expert_gate_up_shared: KernelHandle,
    moe_expert_silu_down_shared: KernelHandle,
    moe_topk: KernelHandle,
    moe_weighted_sum_blend: KernelHandle,
    residual_add: KernelHandle,
    moe_topk_batched: KernelHandle,
    // K=2 fused MoE kernel handles
    moe_expert_gate_up_shared_batch2: KernelHandle,
    moe_expert_silu_down_shared_batch2: KernelHandle,
    moe_weighted_sum_blend_batch2: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    // K=3 fused MoE kernel handles
    moe_expert_gate_up_shared_batch3: KernelHandle,
    moe_expert_silu_down_shared_batch3: KernelHandle,
    /// Expert-deduplicated twins of the batch3 kernels ([r8, r16] row caps);
    /// KernelHandle(0) where the target's module does not carry them.
    moe_expert_gate_up_shared_dedup: [KernelHandle; 2],
    moe_expert_silu_down_shared_dedup: [KernelHandle; 2],
    /// Activation-once dedup down (`ATLAS_QWEN4_MOE_DOWN_ACT=1`): the SiLU*up
    /// row kernel and the [r8, r16] down twins; zero handles where absent.
    moe_dedup_act_rows_k: KernelHandle,
    /// (kernel, output pairs per warp) for the [r8, r16] act down twins.
    moe_expert_silu_down_shared_dedup_act: [(KernelHandle, u32); 2],
    /// FP32 activation rows for the act down, allocated on first use.
    moe_down_act_buf: std::sync::OnceLock<DevicePtr>,
    moe_weighted_sum_blend_batch3: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    // Sorted/grouped prefill path
    moe_sort_by_expert: KernelHandle,
    moe_grouped_gemm: KernelHandle,
    moe_silu_mul: KernelHandle,
    /// Activation kernel for sorted/unfused path. SiLU by default, GeGLU for Gemma-4.
    moe_act_mul: KernelHandle,
    /// True when the model uses GeGLU activation (e.g. Gemma-4); gates Qwen4 SiLU-only prefill admission.
    gelu_activation: bool,
    moe_unpermute_reduce: KernelHandle,
    moe_batched_blend: KernelHandle,
    /// Pointer tables for batched expert dispatch.
    gate_ptrs: ExpertPtrTable,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    /// Transposed pointer tables for coalesced prefill GEMM.
    gate_ptrs_t: Option<ExpertPtrTable>,
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    /// Owning bases for the canonical Qwen4 routed-expert transposed slabs.
    /// Pointer tables expose only 256-byte-aligned interior views; model
    /// teardown releases these backend-tracked bases with other allocations.
    routed_gate_up_transpose_slab: Option<DevicePtr>,
    routed_down_transpose_slab: Option<DevicePtr>,
    /// Owning bases for Atlas-TS4E-128 compressed routed scale slabs (one per projection).
    routed_ts4e_scale_slabs: Vec<DevicePtr>,
    /// Ordinary backend-tracked destination owners, never source-group claims.
    /// Retained immediately, including on partial construction failure.
    routed_bounded_transpose_owners: Vec<bounded_transpose::RetainedOwner>,
    /// Opt-in owners for source packed/scales. Individual expert pointers are
    /// non-owning views when this is present.
    routed_original_groups: Option<spark_runtime::weights::RoutedOriginalGroups>,
    /// Lazy down_proj transpose scratch — populated at the start of each
    /// prefill call when the persistent transpose pass couldn't fit
    /// down_proj. Decode keeps using `down_ptrs` (untransposed); prefill
    /// uses `down_ptrs_t` pointing into this scratch. Shared across all
    /// MoE layers (the same scratch is overwritten layer-by-layer during
    /// the sequential forward).
    ///
    /// `down_t_scratch_packed`: contiguous `[num_experts × N × K/2]` bytes.
    /// `down_t_scratch_scale`:  contiguous `[num_experts × N × K/16]` bytes.
    /// Both `None` when the persistent transpose pass already covered
    /// down (full-fits path) or when the layer doesn't need scratch
    /// transpose (FP8 experts, etc.).
    down_t_scratch_packed: Option<DevicePtr>,
    down_t_scratch_scale: Option<DevicePtr>,
    /// Complete gate/up/down scratch for exact per-layer Qwen4 prefill
    /// transposition. Unlike unified layout, original weights stay resident.
    stream_t_scratch: Option<MoeStreamTransposeScratch>,
    /// Kernel handle for the batched per-expert uint8 transpose.
    moe_transpose_u8_batched_k: KernelHandle,
    // Transposed-layout decode kernels used by unified-layout MoE decode and verify dispatch.
    moe_expert_gate_up_shared_t_k: KernelHandle,
    moe_expert_silu_down_shared_t_k: KernelHandle,
    /// Lane-parallel exact twins of the two kernels above (see
    /// `ops::T_LANES_OUT_BLOCK`); KernelHandle(0) where the target's module
    /// does not carry them.
    moe_expert_gate_up_shared_t_lanes_k: KernelHandle,
    moe_expert_silu_down_shared_t_lanes_k: KernelHandle,
    /// Eight-threads-per-group M=1 decode gate/up and down (`_t_lanes_t8_k160` / `_k40`),
    /// byte-identical to `_t_lanes_o4t16u1`; zero where absent.
    moe_expert_gate_up_shared_t_lanes_t8_k: KernelHandle,
    moe_expert_silu_down_shared_t_lanes_t8_k: KernelHandle,
    /// Multi-row, expert-deduplicated twins of the `_t_lanes` kernels for the
    /// Qwen4 exact K4/K5/K9 verify MoE on the unified layout (fn-decode).
    moe_expert_gate_up_shared_t_lanes_rows_k: KernelHandle,
    moe_expert_silu_down_shared_t_lanes_rows_k: KernelHandle,
    /// Row-count-specialized twins (rows cap 2/3/4/5/9; gate/up input staged in
    /// smem), byte-identical; (cap, gate_up, down). Empty when not loaded.
    t_lanes_rows_spec: Vec<(u32, KernelHandle, KernelHandle)>,
    /// `[r3, r5, r8, unset]` tiers of the t8 K-row gate/up and down kernels (`2..=3`, `4..=5`, `6..=8` rows; slot `[3]` is `KernelHandle(0)`).
    moe_expert_gate_up_shared_t_lanes_rows_t8_tiers: [KernelHandle; 4],
    moe_expert_silu_down_shared_t_lanes_rows_t8_tiers: [KernelHandle; 4],
    /// Warp-ballot unique-expert worklist builder (`moe_build_unique_worklist_r8`).
    moe_build_unique_worklist_r8_k: KernelHandle,
    /// `[r3, r5, r8, unset]` tiers of worklist-driven t8 gate/up (`moe_expert_gate_up_worklist_t8_r{3,5,8}_k160`; slot `[3]` is `KernelHandle(0)`).
    moe_expert_gate_up_worklist_t8_tiers: [KernelHandle; 4],
    /// `[r2, r3, r4, r5, r8, r10]` tiers of worklist-driven 32-K-lane gate/up (`moe_expert_gate_up_worklist_t_lanes_rows_r{2,3,4,5,8,10}`).
    moe_expert_gate_up_worklist_t_lanes_rows_r2_k: KernelHandle,
    moe_expert_gate_up_worklist_t_lanes_rows_r4_k: KernelHandle,
    moe_expert_gate_up_worklist_t_lanes_rows_tiers: [KernelHandle; 4],
    /// Pre-materialized FP32 `SiLU(gate) * up` row kernel (`moe_silu_mul_rows_f32`).
    moe_silu_mul_rows_f32_k: KernelHandle,
    /// `[r3, r5, r8, r10]` tiers of worklist+act-staged t8 down (`moe_expert_down_act_shared_t_lanes_rows_t8_r{3,5,8,10}_k40`).
    moe_expert_down_act_shared_t_lanes_rows_t8_tiers: [KernelHandle; 4],
    /// Lazy per-layer device worklist buffer (`WORKLIST_MAX_SLOTS = 2560`, `122,880 B` = 120 KiB).
    moe_t8_worklist_buf: std::sync::OnceLock<DevicePtr>,
    /// Lazy per-layer device FP32 activation buffer `(capacity_bytes, ptr)` sized for `NATIVE_TARGET_MOE_MAX_TILE = 256` rows.
    moe_t8_act_buf: std::sync::OnceLock<(usize, DevicePtr)>,
    moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    /// Unified-layout decode path: gate/up/down all use transposed `[K/2, N]`
    /// layout and decode dispatches to `moe_expert_*_shared_t` kernels.
    /// Defaults to ON for Qwen4 (`config.is_qwen4_exp()`) and OFF for other
    /// architectures; set `ATLAS_UNIFIED_MOE_LAYOUT=0` to force original
    /// `[N, K/2]` layout (or `=1` to opt in elsewhere).
    /// Resolved once at construction.
    unified_layout: bool,
    /// `ATLAS_NVFP4_GATE_UP_M128=1` opts in to the M=128 fused gate+up
    /// kernel (halving block count at large prefill for better SM amortization
    /// on GB10's 48 SMs). Currently only minimax-m2-229b ships the kernel;
    /// other models keep `moe_fused_gate_up_t_k64_m128 == KernelHandle(0)` and
    /// dispatch falls through to the M=64 path even when the env var is set.
    nvfp4_gate_up_m128: bool,
    /// `ATLAS_MOE_K3_FUSED_GATE_UP=1` opts in to the prefill-style fused
    /// gate+up kernel (`moe_w4a16_fused_gate_up_t_k64`) on the K=3 MTP
    /// verify path. Resolved once at construction; dispatch additionally
    /// requires `gate_ptrs_t`, `up_ptrs_t`, `down_ptrs_t` to be resident
    /// (full persistent transpose, NOT the lazy scratch path) AND
    /// `moe_fused_gate_up_t_k64 != KernelHandle(0)`. Falls through to
    /// the existing batch3 path when any precondition fails (incl. FP8
    /// experts, EP > 1, sigmoid+correction-bias routing).
    k3_fused_gate_up: bool,
    /// `ATLAS_HYBRID_MOE_LAYOUT=1` opts in to the hybrid-layout path:
    /// keep BOTH original `[N, K/2]` weights (for decode + MTP verify) AND
    /// transposed `[K/2, N]` weights (for prefill). Doubles MoE-weight
    /// memory but recovers the ~15 % decode regression that pure unified
    /// layout suffers from. Resolved once at construction; mutually
    /// exclusive with `unified_layout` at the dispatch level (hybrid wins
    /// on decode paths since it preserves untransposed warp-reduction
    /// parallelism).
    hybrid_layout: bool,
    /// Transposed shared expert weights for prefill.
    shared_gate_t: Option<QuantizedWeight>,
    /// `ATLAS_QWEN4_DENSE_FP8=1`: native MXFP8 shared expert (decode scope only).
    qwen4_shared_fp8: Option<crate::layers::qwen4_dense_fp8::proj::SharedFp8>,
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    moe_grouped_gemm_t_k64: KernelHandle,
    moe_fused_gate_up_t_k64: KernelHandle,
    /// M=128 variant of the K64 fused gate+up kernel. Loaded with `try_kernel`
    /// — falls back to `KernelHandle(0)` on models that don't ship the kernel;
    /// dispatch gates on `nvfp4_gate_up_m128` AND handle non-zero.
    moe_fused_gate_up_t_k64_m128: KernelHandle,
    /// Default-off same-stream compact ordinary-NVFP4 prefill route.
    nvfp4_moe_worklist: bool,
    /// F8-only original-layout compact planner/GEMM; absent by default.
    qwen4_compact: Option<qwen4_prefill_compact::Kernels>,
    /// Device builder plus compact M64 gate/up and down kernels. All
    /// three are an atomic capability bundle when the route is requested.
    moe_build_nvfp4_worklist_k: KernelHandle,
    moe_fused_gate_up_t_k64_worklist_k: KernelHandle,
    moe_grouped_gemm_t_k64_worklist_k: KernelHandle,
    w4a16_gemm_t: KernelHandle,
    /// Pre-dequanted FP8 weights for zero-overhead prefill GEMMs.
    gate_fp8: Option<DevicePtr>,
    shared_gate_fp8: Option<DevicePtr>,
    shared_up_fp8: Option<DevicePtr>,
    shared_down_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    /// Secondary CUDA stream for overlapping shared expert with routed experts.
    prefill_stream: u64,
    /// Event pair for stream synchronization (input_ready, shared_done).
    event_a: u64,
    event_b: u64,
    // ── Sigmoid + correction-bias routing (DeepSeek-V3 / MiniMax-M2 style) ──
    /// Device pointer to `[num_experts]` correction bias. Populated from
    /// `MoeWeights.correction_bias` in `new()` when the loader sets it.
    /// `None` = Atlas's default softmax path. When `Some`, every top-k
    /// dispatch site branches to `moe_topk_sigmoid` with this bias arg.
    correction_bias_dev: Option<DevicePtr>,
    /// Handle to `moe_topk_sigmoid` kernel. Lazy-loaded in `new()` even
    /// when bias is `None` (harmless if kernel isn't used).
    moe_topk_sigmoid_k: KernelHandle,
    /// Batched variant for prefill / MTP-verify (one block per token).
    /// Loaded via `try_kernel` — returns KernelHandle(0) on models whose
    /// KERNEL.toml doesn't register the sigmoid kernels (e.g. Mistral).
    /// Never dispatched on those paths because `correction_bias_dev` is
    /// `None` there.
    moe_topk_sigmoid_batched_k: KernelHandle,
    // FP8 fused MoE kernels (used when experts are FP8)
    moe_expert_gate_up_shared_fp8: KernelHandle,
    moe_expert_silu_down_shared_fp8: KernelHandle,
    // FP8 batch2/3 fused MoE kernels (for MTP K=2/K=3 verify)
    moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // FP8 grouped GEMM for sorted MoE prefill
    moe_fp8_grouped_gemm_k: KernelHandle,
    // FP8 grouped GEMM v2 — coalesced B/A load thread-remap. Default ON
    // (override via ATLAS_FP8_MOE_COALESCED=0 to restore v1). Kernel handle
    // may be 0 on older images (then dispatch falls back to v1). Same signature as v1.
    moe_fp8_grouped_gemm_v2_k: KernelHandle,
    // Resolved once per layer from ATLAS_FP8_MOE_COALESCED env var.
    fp8_moe_coalesced_enabled: bool,
    w8a16_gemm_k: KernelHandle, // for shared expert FP8 prefill
    // FP8 expert pointer tables (None when experts are NVFP4)
    fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // FP8 shared expert weights (None when shared expert is NVFP4)
    fp8_shared_expert: Option<Fp8ExpertWeight>,
    // Phase 2.7 Tier C — Frankenstein dispatch flag.
    // True when this layer's index is in `config.dflash_capture_layers`.
    // When the env var `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` is set,
    // `forward()` (single-token decode) will route through `forward_prefill`
    // (tensor-core grouped GEMM kernel) on this layer only, so the captured
    // hidden states use a different numerical recipe than the scalar GEMV
    // path. Used to test whether the kernel choice is the dominant cause
    // of low DFlash drafter acceptance on FP4/FP8 targets.
    pub is_dflash_capture_layer: bool,
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod collision_trace;
mod forward;
mod forward_batched;
mod forward_ep;
mod forward_k2;
mod forward_k3;
mod forward_k3_fused;
mod forward_k5;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod init;
mod qwen4_compact_check;
mod qwen4_compact_compare;
mod qwen4_compact_contract;
mod qwen4_compact_ops;
pub(crate) use qwen4_compact_ops::discard_stale_slot_storage;
pub(crate) mod qwen4_moe_status;
mod qwen4_plan_order_rule;
mod qwen4_prefill_admission;
mod qwen4_prefill_admission_router;
pub(crate) mod qwen4_prefill_compact;
pub use qwen4_prefill_compact::{
    DEFAULT_PREFILL_GRID, SMALL_BATCH_GRID, SMALL_BATCH_THRESHOLD, get_optimal_grid_dim,
};
pub(crate) mod qwen4_router_bf16;
mod qwen4_router_pipe;
pub(crate) mod qwen4_shared_bf16;
mod qwen4_shared_fp8_arm;
mod qwen4_shared_v7;
mod qwen4_stream_t;
mod qwen4_topk_warp;
mod qwen4_topk_warp_rule;
mod qwen4_xq_quant;
pub mod trait_decode;
pub use trait_decode::{MoeSmallBatchRoute, SMALL_BATCH_MOE_THRESHOLD};

/// Build a device-side pointer table from pre-transposed QuantizedWeight vec.
fn build_ptr_table_from_qw(
    weights: &[QuantizedWeight],
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = weights.len();
    let packed_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();

    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side pointer table for one projection across all experts.
fn build_ptr_table(
    experts: &[ExpertWeight],
    proj: impl Fn(&ExpertWeight) -> &crate::weight_map::QuantizedWeight,
    gpu: &dyn GpuBackend,
) -> Result<ExpertPtrTable> {
    let n = experts.len();

    // Build host-side arrays
    let packed_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight_scale_2.to_le_bytes())
        .collect();

    // Upload to device
    let packed_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    let scale2_vals = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;

    Ok(ExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Build a device-side FP8 pointer table for one projection across all experts.
///
/// FP8 experts store 2 arrays (weight + block_scale) per projection,
/// vs NVFP4's 3 (packed + scale + scale2).
fn build_fp8_ptr_table(
    experts: &[Fp8ExpertWeight],
    proj: impl Fn(&Fp8ExpertWeight) -> &Fp8Weight,
    gpu: &dyn GpuBackend,
) -> Result<Fp8ExpertPtrTable> {
    let n = experts.len();

    let weight_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| proj(e).row_scale.0.to_le_bytes())
        .collect();

    let weight_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&weight_bytes, weight_ptrs)?;

    let scale_ptrs = gpu.alloc(n * 8)?;
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;

    Ok(Fp8ExpertPtrTable {
        weight_ptrs,
        scale_ptrs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_moe_kernel_loading() {
        let gpu = MockGpuBackend::new();
        assert!(gpu.kernel("gemv", "dense_gemv_bf16").is_ok());
        assert!(gpu.kernel("w4a16_gemv", "w4a16_gemv").is_ok());
        assert!(gpu.kernel("moe_topk", "moe_topk_softmax").is_ok());
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up_2x")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_silu_down_2x")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")
                .is_ok()
        );
        assert!(
            gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")
                .is_ok()
        );
        // K=2 batch dispatch
        assert!(gpu.kernel("moe_topk", "moe_topk_softmax_batched").is_ok());
    }
}
