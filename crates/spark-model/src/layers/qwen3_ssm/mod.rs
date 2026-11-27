// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-Next SSM (Gated Delta Net) layer implementing TransformerLayer.
//!
//! Corrected pipeline matching the HuggingFace reference implementation:
//!   1. QKVZ projection (interleaved output)
//!   2. Deinterleave QKVZ → sequential [Q | K | V | Z]
//!   3. BA projection (interleaved output)
//!   4. Compute GDN gates: gate = exp(-A * softplus(alpha + dt_bias)), beta = sigmoid(b)
//!   5. Conv1d update on [Q | K | V] concatenated (d_inner=8192)
//!   6. Split conv output → Q', K', V'
//!   7. GDN decode (Q', K', V', gate, beta) — kernel handles GQA internally
//!   8. Gated RMS norm (GDN output, Z gate)
//!   9. Output projection [value_dim → hidden_size]
//!  10. MoE FFN

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle, PinnedHostBuffer};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{
    ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::layers::{FfnComponent, Qwen4HyperConnection};
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, SsmWeights};

const QWEN38_FLASHINFER_HIDDEN: usize = 5_120;
const QWEN38_FLASHINFER_SSM_QKVZ: usize = 16_384;
const QWEN38_FLASHINFER_SSM_VALUE: usize = 6_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SsmFlashinferPrefillRoute {
    Disabled,
    Ineligible,
    Missing,
    Complete,
}

const fn ssm_flashinfer_prefill_route(
    requested: bool,
    exact_geometry: bool,
    nvfp4_active: bool,
    prepared: bool,
) -> SsmFlashinferPrefillRoute {
    if !requested {
        SsmFlashinferPrefillRoute::Disabled
    } else if !exact_geometry || !nvfp4_active {
        SsmFlashinferPrefillRoute::Ineligible
    } else if !prepared {
        SsmFlashinferPrefillRoute::Missing
    } else {
        SsmFlashinferPrefillRoute::Complete
    }
}

const fn qwen38_ssm_flashinfer_geometry(
    rows: usize,
    hidden: usize,
    qkvz: usize,
    value: usize,
) -> bool {
    matches!(rows, 2_079 | 8_192)
        && hidden == QWEN38_FLASHINFER_HIDDEN
        && qkvz == QWEN38_FLASHINFER_SSM_QKVZ
        && value == QWEN38_FLASHINFER_SSM_VALUE
}

const fn qwen38_ssm_flashinfer_tactics(rows: usize) -> Option<(u8, u8)> {
    match rows {
        2_079 => Some((4, 2)),
        8_192 => Some((2, 2)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SsmFlashinferScratchLayout {
    packed_offset: usize,
    scales_offset: usize,
    global_max_offset: usize,
    scale2_offset: usize,
    status_offset: usize,
    alpha_offset: usize,
    total_bytes: usize,
}

fn ssm_flashinfer_scratch_layout(
    plan: ops::nvfp4_dynamic_scale::Nvfp4DynamicScalePlan,
    capacity: usize,
) -> Result<SsmFlashinferScratchLayout> {
    let align16 = |value: usize| -> Result<usize> {
        value
            .checked_add(15)
            .map(|next| next & !15)
            .ok_or_else(|| anyhow::anyhow!("FlashInfer SSM scratch alignment overflow"))
    };
    let packed_offset = 0;
    let scales_offset = align16(plan.packed_bytes)?;
    let global_max_offset = align16(
        scales_offset
            .checked_add(plan.physical_scale_bytes)
            .ok_or_else(|| anyhow::anyhow!("FlashInfer SSM scale scratch overflow"))?,
    )?;
    let scale2_offset = global_max_offset + 4;
    let status_offset = scale2_offset + 4;
    let alpha_offset = status_offset + 4;
    let total_bytes = alpha_offset + 4;
    ensure!(
        total_bytes <= capacity,
        "FlashInfer SSM activation scratch requires {total_bytes} bytes, arena has {capacity}"
    );
    Ok(SsmFlashinferScratchLayout {
        packed_offset,
        scales_offset,
        global_max_offset,
        scale2_offset,
        status_offset,
        alpha_offset,
        total_bytes,
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferSsmProjection {
    weight: DevicePtr,
    weight_scales_128x4: DevicePtr,
    weight_scales_hash: u64,
    weight_scale_2: f32,
    n: usize,
    k: usize,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferSsmPrefill {
    layer: usize,
    library: ops::flashinfer_sm121::FlashInferSm121,
    dynamic_scale_kernels: ops::nvfp4_dynamic_scale::Nvfp4DynamicScaleKernels,
    qkvz: FlashinferSsmProjection,
    output: FlashinferSsmProjection,
}

mod exact_flat_route;
mod gdn_c143_prefill;
#[cfg(test)]
mod gdn_c143_prefill_tests;
pub(crate) mod gdn_l2_window;
#[cfg(test)]
mod qkvz_e4m3_readiness_tests;
pub(crate) mod ssm_h_fp16;
use exact_flat_route::{ExactFlatSsmRoute, contiguous_intermediate_base, exact_flat_ssm_route};

/// Qwen3-Next SSM/GDN layer (36 of 48 layers).
///
/// Supports two QKVZ projection modes:
/// - **Interleaved** (80B): `w4a16_gemv_qkvz` or GEMV + `deinterleave_qkvz`
/// - **Sequential** (3.5-35B): plain GEMV → `[Q|K|V|Z]` already in order
#[allow(dead_code)]
pub struct Qwen3SsmLayer {
    input_norm: DenseWeight,
    ssm: SsmWeights,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    qwen4_attn_hyper: Option<Qwen4HyperConnection>,
    qwen4_mlp_hyper: Option<Qwen4HyperConnection>,
    // NVFP4-quantized QKVZ weight (quarters bandwidth vs BF16)
    qkvz_nvfp4: Option<QuantizedWeight>,
    // Transposed [K/2, N] copy for coalesced w4a16_gemm reads (prefill)
    qkvz_nvfp4_t: Option<QuantizedWeight>,
    // Transposed out_proj for prefill GEMM
    out_proj_nvfp4_t: Option<QuantizedWeight>,
    // BF16 out_proj for models where SSM weights are not pre-quantized
    pub out_proj_dense: Option<DenseWeight>,
    /// `ATLAS_QWEN4_GDN_BF16=1` (decode-only): BF16 out_proj read ONLY by
    /// the single-token decode GEMV in `ssm_forward.rs`, checked with
    /// higher priority than `out_proj_fp8w`/`out_proj_dense`/NVFP4 there.
    /// Deliberately a separate field from `out_proj_dense`: that field is
    /// also read by prefill (`trait_prefill_phase3.rs`, checked before
    /// FP8), so reusing it would silently change the prefill dispatch
    /// too. This field is `None` unless the control selected it, so the
    /// default decode dispatch order is unchanged byte-for-byte.
    pub(crate) out_proj_dense_decode_only: Option<DenseWeight>,
    /// `ATLAS_QWEN4_GDN_BF16=1` (decode-only): BF16 QKVZ read ONLY by the
    /// single-token decode GEMV in `ssm_forward.rs`, checked with higher
    /// priority than the fused/TC-parity/`qkvz_nvfp4`/dense arms there.
    /// `qkvz_nvfp4` itself stays `Some` unconditionally — see the loader
    /// comment in `linear_attn_arms.rs` for why nulling it is unsafe
    /// (several prefill/verify consumers assert it directly, independent
    /// of `qkvz_nvfp4_t`).
    pub(crate) qkvz_dense_decode_only: Option<DenseWeight>,
    /// Default-off FlashInfer SM121 projection operands. Construction is
    /// explicit and atomic; runtime never derives weight layouts or scalars.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    flashinfer_ssm_prefill: Option<FlashinferSsmPrefill>,
    // FP8 E4M3 checkpoint weights for native FP8 serving (w8a16_gemv LUT kernel)
    qkvz_fp8w: Option<Fp8Weight>,
    out_proj_fp8w: Option<Fp8Weight>,
    /// `ATLAS_QWEN4_DENSE_FP8=1`: native MXFP8 decode operands (decode scope only).
    qwen4_dense_fp8: Option<crate::layers::qwen4_dense_fp8::proj::GdnFp8>,
    /// When true, QKVZ projection output is already sequential [Q|K|V|Z].
    /// Skips the deinterleave kernel (used by Qwen3.5 where QKV+Z are
    /// concatenated at load time rather than interleaved per-group).
    sequential_qkvz: bool,
    // Kernels — decode path (single-token GEMV)
    rms_norm_residual_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    gated_rms_norm_f32_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    /// K=3 batched counterpart of `dense_gemv_k` for the SSM BA projection.
    /// Single launch handles all 3 tokens, eliminating ~48 μs of launch
    /// overhead per SSM layer per K=3 verify step. Gated by
    /// `ATLAS_SSM_BA_BATCHED=1`; NULL handle when the kernel isn't in the
    /// active target's PTX bundle, in which case the per-token loop runs.
    dense_gemv_batch3_k: KernelHandle,
    /// General batched (grid.y = token) counterpart of `dense_gemv_k` for
    /// the SSM BA projection at any num_tokens (DFlash K=γ=17 included).
    /// Bit-identical to the per-token loop (same kernel body per y-block).
    /// Gated by `ATLAS_SSM_BA_BATCH=1`; NULL handle → per-token loop.
    dense_gemv_batchn_k: KernelHandle,
    /// Exact multi-row BA projection with inline FP32 gate transforms.
    ba_gates_batchn_exact_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// Exact K1-order multi-row NVFP4 kernels shared by QKVZ and out_proj.
    /// A missing selected tier fails closed to ordinary row-major K1 GEMVs.
    w4a16_exact_projection_kernels: ops::W4a16ExactLmHeadKernels,
    /// Single-warp `w4a16_gemv_sw` — lossless, 8 outputs per 256-thread block
    /// instead of 4, no cross-warp smem round-trip. `KernelHandle(0)` on a
    /// miss falls back to the bit-identical 64-thread base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    /// `ATLAS_NO_GEMV_SW != "1"`, cached at construction.
    gemv_sw: bool,
    w8a16_gemv_k: KernelHandle,
    w4a16_gemv_qkvz_k: KernelHandle,
    /// Fused rms_norm_residual + w4a16_gemv for SSM QKVZ (sequential layout).
    /// Gated by ATLAS_FUSE_SSM_QKVZ=1 at decode dispatch time. NULL handle if
    /// the kernel module didn't load (older PTX bundles).
    fused_rms_qkvz_k: KernelHandle,
    /// K=3 batched counterpart of `fused_rms_qkvz_k` (3-token verify path).
    /// Same env gate; NULL handle when not built for the active target.
    fused_rms_qkvz_batch3_k: KernelHandle,
    deinterleave_k: KernelHandle,
    conv1d_k: KernelHandle,
    conv1d_l2norm_k: KernelHandle,
    conv1d_l2norm_f32_k: KernelHandle,
    /// Exact flat-chain multi-row FP32 conv recurrence + state snapshots.
    conv1d_l2norm_f32_sequence_k: KernelHandle,
    /// Exact flat-chain multi-row BF16 conv recurrence + state snapshots.
    conv1d_l2norm_bf16_sequence_k: KernelHandle,
    /// Tree-aware multi-row BF16 conv recurrence + state snapshots.
    conv1d_l2norm_bf16_tree_sequence_k: KernelHandle,
    gdn_k: KernelHandle,
    gdn_f32_k: KernelHandle,
    /// Exact flat-chain multi-row FP32 GDN recurrence + H snapshots.
    gdn_f32_sequence_k: KernelHandle,
    /// Register-resident H variant (identical arithmetic, H cached in regs).
    gdn_f32_sequence_persistent_k: KernelHandle,
    /// Speed-probe twin with per-token snapshot writes removed (ATLAS_SSM_GDN_NOSNAP).
    /// INVALIDATES rollback — measurement only.
    gdn_f32_sequence_nosnap_k: KernelHandle,
    /// LAZY-COMMIT variant (`ATLAS_SSM_GDN_LAZY=1`): registers-resident, writes
    /// NO per-token snapshots (58.6% of the kernel at k=16), writes final H to
    /// inter[num_tokens-1], and leaves `h_state` holding the step-initial H so
    /// the commit replay (the nosnap kernel over retained inputs) can
    /// reconstruct any accepted state bit-exactly. Requires the async commit
    /// integration in `async_chkpt.rs` — never enable without it.
    gdn_f32_sequence_lazyfinal_k: KernelHandle,
    /// Bit-exact Split-V (`grid=[nv, 4, 1]`, `block=32`) twins of
    /// `gdn_f32_sequence_{persistent,lazyfinal}_k` for wide-SM (`> 64` SM) GPUs.
    gdn_f32_sequence_persistent_splitv4_k: KernelHandle,
    gdn_f32_sequence_lazyfinal_splitv4_k: KernelHandle,
    /// Fused 1x1x1-CTA Qwen4 GDN decode (K=1) and speculative verify kernels
    /// (`ATLAS_QWEN4_GDN_VERIFY_FUSED`, default enabled).
    qwen4_gdn_decode_fused_k1_k: KernelHandle,
    qwen4_gdn_verify_fused_lazyfinal_k: KernelHandle,
    qwen4_gdn_verify_fused_persistent_k: KernelHandle,
    /// Multi-sequence (`grid=[nv, C, 1]`) fused Qwen4 GDN decode and verify kernels.
    #[allow(dead_code)]
    qwen4_gdn_decode_fused_k1_multi_k: KernelHandle,
    qwen4_gdn_verify_fused_lazyfinal_multi_k: KernelHandle,
    qwen4_gdn_verify_fused_persistent_multi_k: KernelHandle,
    /// Split-V (`grid=[nv, 4, C]`) multi-CTA fused Qwen4 GDN decode and verify
    /// kernels (`ATLAS_QWEN4_GDN_SPLIT_V`, auto on `sm_count > 64` for `C=1`).
    qwen4_gdn_decode_fused_k1_splitv4_k: KernelHandle,
    qwen4_gdn_verify_fused_lazyfinal_splitv4_k: KernelHandle,
    qwen4_gdn_verify_fused_persistent_splitv4_k: KernelHandle,
    qwen4_gdn_verify_fused_lazyfinal_multi_splitv4_k: KernelHandle,
    qwen4_gdn_verify_fused_persistent_multi_splitv4_k: KernelHandle,
    /// Zero-initialized self-resetting `u32` group completion counters
    /// (`[2 * MAX_LANES][32 key heads]`) for race-free GQA Q/K conv_state writeback.
    gdn_fused_group_done: DevicePtr,
    /// Per-slot stride (in `u32` elements, `>= linear_num_value_heads`) of
    /// `gdn_fused_split_done`.
    gdn_fused_split_done_stride: usize,
    /// Zero-initialized self-resetting `u32` Split-V completion counters
    /// (`[2 * MAX_LANES][gdn_fused_split_done_stride]`) for cross-CTA Step 4 RMS norm.
    gdn_fused_split_done: DevicePtr,
    /// Lazy-commit retention pool: [GDN_LAZY_MAX_K x qkvz] fp32 q/k/v(+z) +
    /// the gate/beta block at GDN_LAZY_MAX_K*qkvz*4 per buffer, or a single
    /// compact circular ring (`KEEP_LATE_RING_CAP = 64` raw-BF16 + FP32-gate
    /// rows, `gdn_lazy_retain_lane_stride_bytes`) paired with per-slot
    /// `GdnKeepLateCtrl` and single-seq descriptor device slots when
    /// `ATLAS_QWEN4_GDN_KEEP_LATE` is active. Lazily allocated on first engaged
    /// dispatch (or pre-allocated outside CUDA graph capture).
    gdn_lazy_retain: std::sync::OnceLock<GdnLazyRetainPool>,
    ba_gates_k: KernelHandle,
    residual_add_k: KernelHandle,
    l2_norm_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    gated_rms_norm_prefill_k: KernelHandle,
    // Kernels — batched verification path (multi-token GEMM)
    w4a16_gemm_k: KernelHandle,
    /// Byte-exact cp.async pipelined shadow of `w4a16_gemm` (see
    /// `prefill_proj_pipe_enabled`). Used by the SSM QKVZ + out_proj prefill
    /// projections at `ATLAS_PREFILL_PROJ_PIPE=1`; an eligible explicit
    /// request fails before projection when this handle is 0.
    w4a16_gemm_pipe_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle, // Transposed B layout [K/2, N] — K_STEP_T=32
    w4a16_gemm_t_k64_k: KernelHandle, // K64 variant: K_STEP_T=64, halves outer loop
    w4a16_gemm_t_m128_k: KernelHandle, // M128 variant: 2 M-chunks per CTA, halves B re-reads
    /// 8-warp 128x128 bit-identical shadows (ATLAS_PREFILL_FP8_W8=1); 0 when absent.
    w4a16_gemm_t_w8_k: KernelHandle,
    fp8_gemm_t_w8_k: KernelHandle,
    /// M16 variant: 1 CTA row × 4 warps × 32-N each (K=γ verify, M≤32).
    /// Gated by `ATLAS_TC_NVFP4_M16=1` env var. KernelHandle(0) if not compiled
    /// for this target (qwen3.6-27b NVFP4 shadow only as of 2026-05-19).
    w4a16_gemm_t_m16_k: KernelHandle,
    /// M32×N64 variant: K=γ verify small-M GEMM (single B read for M≤32 ×
    /// N_TILE=64 → full SM occupancy). Used by the SSM `out_proj`
    /// [M=17, N=5120, K=6144] at 3 < M ≤ 32 to replace the N_TILE=128
    /// `w4a16_gemm_t` (only 40 CTAs at N=5120, SM-starved). KernelHandle(0)
    /// if the kernel is not compiled for this target.
    w4a16_gemm_t_m32_n64_k: KernelHandle,
    /// Split-K variant of the m32_n64 kernel + its FP32→BF16 reduce.
    /// Used by the K=γ verify `out_proj` (ATLAS_SSM_OUT_SPLITK) and
    /// `qkvz` (ATLAS_SSM_QKVZ_SPLITK) routes: the single-slice out_proj
    /// fields only 80 CTAs on 48 SMs (floor-map measured 28% of the DRAM
    /// floor — 235µs vs 66µs); slicing K across gridDim.z into an FP32
    /// workspace reaches 84% (91µs). Mirrors the proven `ffn_down`
    /// split-K path (lossless FP32 partials, token-exact). Handle 0 when
    /// the PTX bundle lacks the kernels — routes silently stay off.
    w4a16_gemm_t_m32_n64_splitk_k: KernelHandle,
    reduce_splitk_k: KernelHandle,
    /// FP32 split-K workspace `[k_splits<=8, 32, max_n]`; pre-allocated at load
    /// time by `alloc_ssm_splitk_ws` (or lazily on first use outside graph capture).
    ssm_splitk_workspace: std::sync::Mutex<Option<DevicePtr>>,
    /// Lazily-allocated e4m3 activation scratch for the cuBLASLt FP8 projection
    /// route (`ATLAS_SSM_PROJ_CUBLASLT=1`). Holds `max_tokens * max_k` bytes,
    /// one e4m3 byte per BF16 activation element. It is sized once to the
    /// widest legal SSM projection and is never replaced while work is live.
    /// Shared by every SSM layer under `ATLAS_MEMLEAN_SHARED_SCRATCH=1`.
    ssm_act_e4m3_scratch: act_e4m3_arena::ActE4m3Arena,
    /// E4M3 copy of this layer's NVFP4 QKVZ weight, laid out [N, K] for the
    /// cuBLASLt FP8 route. When that route is requested by the dense-Qwen
    /// loader, its 80 MiB production-shape cache is fenced and published
    /// during model construction.
    ssm_qkvz_e4m3: std::sync::Mutex<Option<DevicePtr>>,
    w4a16_gemv_batch2_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    gdn_prefill_k: KernelHandle,
    gdn_prefill_split_k: KernelHandle,
    gdn_prefill_split4_k: KernelHandle,
    gdn_prefill_persistent_k: KernelHandle,
    gdn_prefill_persistent_wy4_k: KernelHandle,
    /// WY32 chunked prefill: processes 32 tokens per WY iteration with H in
    /// shared memory. ~30x faster than per-token for 14k+ sequences.
    gdn_prefill_wy32_k: KernelHandle,
    /// ABI-identical WY32 shadow that caches the thread-invariant gate-product
    /// triangle. Default off until live output-hash and TTFT qualification.
    gdn_prefill_wy32_gatecache_k: KernelHandle,
    /// bf16 -> e4m3 activation cast, exactly the MMA path's own conversion.
    cast_bf16_to_e4m3_k: KernelHandle,
    /// NVFP4 -> e4m3 weight materialisation, matching V2_DEQUANT exactly.
    dequant_nvfp4_to_e4m3_k: KernelHandle,
    /// Exact v2 shadow of the gate-cache kernel (ATLAS_GDN_PREFILL_GATECACHE_V2=1).
    gdn_prefill_wy32_gatecache_v2_k: KernelHandle,
    /// Flash-Next WY32 variant with four warp-parallel FP32 reductions.
    gdn_prefill_wy32_warp_k: KernelHandle,
    // ── Batched GDN prefill: same-chunk-len multi-stream kernels ──
    // Each takes `float* const* h_state_ptrs` plus stacked QKV/gate/beta/output.
    // Used by `Qwen3SsmLayer::prefill_batched` when N≥2 streams have matching
    // chunk_len. Null on targets that don't carry the corresponding kernel.
    gdn_prefill_wy32_batched_k: KernelHandle,
    gdn_prefill_persistent_batched_k: KernelHandle,
    gdn_prefill_persistent_wy4_batched_k: KernelHandle,
    gdn_prefill_split4_batched_k: KernelHandle,
    compute_gdn_gates_k: KernelHandle,
    /// Multi-seq variants: advance `c` SSM states in ONE launch. Gated
    /// by `ATLAS_SSM_MULTI_SEQ_KERNEL=1` (additive on top of
    /// `ATLAS_SSM_MULTI_SEQ_BATCHED=1`). KernelHandle(0) if the multi-seq
    /// PTX modules aren't in the active target's bundle, in which case
    /// the trait_decode_multi_seq path falls back to the per-seq loop.
    conv1d_multi_seq_k: KernelHandle,
    conv1d_l2norm_multi_seq_k: KernelHandle,
    gdn_decode_multi_seq_k: KernelHandle,
    compute_gdn_gates_multi_seq_k: KernelHandle,
    /// FP32-output multi-seq variants. These are the production-precision
    /// kernels that the Qwen3.6-27B decode path uses when
    /// `ATLAS_SSM_MULTI_SEQ_KERNEL=1`. The BF16-output variants above
    /// (conv1d_l2norm_multi_seq_k / gdn_decode_multi_seq_k) preserve API
    /// symmetry but aren't dispatched in production because the model is
    /// calibrated for FP32 recurrent precision. KernelHandle(0) when the
    /// PTX module isn't compiled for the active target.
    conv1d_l2norm_f32_multi_seq_k: KernelHandle,
    gdn_decode_f32_multi_seq_k: KernelHandle,
    gdn_decode_f16_multi_seq_k: KernelHandle,
    /// Multi-seq gated_rms_norm with FP32 input + per-seq strides. Lets
    /// the multi_seq decode collapse the per-seq gated_rms_norm loop into
    /// one launch, while writing into a value_dim-contig output buffer so
    /// the subsequent out_proj can fire as a batched w4a16_gemm at M=n.
    /// KernelHandle(0) when not compiled for the active target.
    gated_rms_norm_f32_multi_seq_k: KernelHandle,
    /// K=3 fused conv1d-update + L2 norm + intermediate-state save for the
    /// K=3 verify SSM forward. Replaces the per-token loop (3
    /// `conv1d_l2norm` + 3 d2d copies) with one launch. KernelHandle(0)
    /// when the PTX module isn't in the active target's bundle, in which
    /// case the per-token fallback runs.
    conv1d_l2norm_chunk3_k: KernelHandle,
    /// Scratch device buffer for per-seq state pointer arrays uploaded
    /// before each multi-seq kernel launch. Sized at init for
    /// `QWEN4_NATIVE_MAX_ROWS` (256) sequences: `[h_state_ptrs[c],
    /// conv_state_ptrs[c]]` u64 each (total `2 * max_c * 8` bytes).
    /// Pre-allocated to avoid per-call alloc.
    ssm_multi_seq_ptr_scratch: DevicePtr,
    /// Page-locked HOST staging buffer for the multi-seq pointer table H2D
    /// upload (64 u64s = 512 bytes, covering up to 32 sequences × 2 pointers).
    /// Backend-owned pinned storage gives a stable DMA source address across
    /// CUDA graph replays, unlike a transient stack array.
    ///
    /// Protected by `parking_lot::Mutex` so shared `&self` access is sound
    /// across threads without manual `unsafe impl Sync`. Uncontended lock/unlock
    /// is negligible compared to GPU kernel launch overhead.
    multi_seq_ptr_host: parking_lot::Mutex<PinnedHostBuffer>,
    /// Wide eager `C33..=QWEN4_NATIVE_MAX_ROWS` companion pinned pointer table
    /// (`2 * QWEN4_NATIVE_MAX_ROWS` u64s = 512 u64s).
    multi_seq_ptr_wide_host: parking_lot::Mutex<PinnedHostBuffer>,
    /// Layer-owned rotating 4-slot page-locked host staging ring with per-slot
    /// CUDA completion events for `refresh_multi_seq_ptr_table`.
    multi_seq_ptr_pinned_ring:
        std::sync::Mutex<Option<trait_decode_multi_seq::MultiSeqPtrPinnedRing>>,
    /// Row count of the last pointer table successfully submitted to
    /// `ssm_multi_seq_ptr_scratch`. Zero means the device contents cannot be
    /// reused. The native eager `C2..=QWEN4_NATIVE_MAX_ROWS` lane compares the
    /// stable pinned table before eliding an otherwise identical tiny H2D submission.
    ///
    /// Writers invalidate this before changing `multi_seq_ptr_host` and only
    /// publish a row count after the H2D enqueue succeeds. Layer-forward
    /// serialization makes the host bytes and this marker one cache entry.
    multi_seq_ptr_uploaded_rows: std::sync::atomic::AtomicUsize,
    /// Capacity of the pointer scratch in number of sequences. Acts as
    /// a hard cap on `num_seqs` for the multi-seq kernels (callers fall
    /// back to per-seq loop if `n > ssm_multi_seq_ptr_max`).
    ssm_multi_seq_ptr_max: usize,
    /// Cross-sequence verify twins of the ordered conv/GDN sequence kernels
    /// (`ATLAS_QWEN4_BATCHED_K5_GDN_MULTI`). KernelHandle(0) when absent;
    /// the cross-sequence frame then launches the single-sequence kernels
    /// once per sequence.
    gdn_f32_sequence_persistent_multi_k: KernelHandle,
    gdn_f32_sequence_lazyfinal_multi_k: KernelHandle,
    conv1d_l2norm_f32_sequence_multi_k: KernelHandle,
    /// Lazily allocated page-locked host staging, device copy and
    /// upload-done event of the cross-sequence pointer tables
    /// (`qwen4_cross_seq_verify.rs`).
    cross_seq_ptr_table: std::sync::Mutex<Option<qwen4_cross_seq_verify::CrossSeqPtrTable>>,
    ba_gates_prefill_k: KernelHandle,
    // Kernels — prefill (multi-token sequential)
    conv1d_prefill_k: KernelHandle,
    /// Exact conv1d-prefill shadow that also copies the projected Z channel
    /// beside the existing per-channel token loop. The measured direct-output
    /// route fails closed if the optional symbol is missing.
    conv1d_prefill_zcopy_k: KernelHandle,
    /// Default-off exact extension of `conv1d_prefill_zcopy_k` that retains
    /// the BF16 conv round trip and standalone L2 reduction tree in one
    /// 128-thread head CTA. An explicit eligible request fails closed when an
    /// older kernel bundle lacks the symbol.
    conv1d_prefill_l2norm_zcopy_k: KernelHandle,
    // Kernels — fused chunk2 path (2-token verification)
    gdn_chunk2_k: KernelHandle,
    conv1d_chunk2_k: KernelHandle,
    // Kernels — fused chunk3 path (3-token verification)
    gdn_chunk3_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    // Kernels — WY-chunkwise path (2-pass verification)
    gdn_wy2_k: KernelHandle,
    gdn_wy3_k: KernelHandle,
    gdn_wy4_k: KernelHandle,
    // ATLAS_SSM_H_FP16 twins of the three chunked-path WY kernels. The chunked
    // K>=5 route (`pick_chunk`) is what a k=13 verify actually runs — 4+4+3+2 —
    // so these three cover it. Zero when the PTX lacks the symbol; the call
    // sites turn that into a hard error rather than a silent FP32 fallback,
    // because reading an FP16 pool as floats produces fluent garbage, not a
    // crash.
    gdn_wy2_f16_k: KernelHandle,
    gdn_wy3_f16_k: KernelHandle,
    gdn_wy4_f16_k: KernelHandle,
    /// WY-Chunkwise K=17 GDN verify (DFlash γ+1). Only present in
    /// qwen3.6-35b-a3b's PTX module set; NULL handle for other targets,
    /// in which case decode_batched(K=17) falls through to the sequential
    /// per-token path.
    gdn_wy17_k: KernelHandle,
    /// V-dim-split wy17 (`gated_delta_rule_wy17_vsplit`): occupancy variant
    /// that fans the 48-head wy17 launch across `ATLAS_WY17_SPLIT` v-column
    /// bands (gridDim.z). Bit-identical output to `gdn_wy17_k`. NULL handle
    /// when the kernel isn't in the active target's PTX bundle → the K=17
    /// path uses `gdn_wy17_k` unchanged.
    gdn_wy17_vsplit_k: KernelHandle,
    /// LAZY Hi-writes wy17 (`gated_delta_rule_wy17_lazy`): takes runtime
    /// `lazy_j`; lazy_j==1 is bit-identical to `gdn_wy17_k`, lazy_j>1
    /// persists only checkpoint intermediate slots (86%-of-traffic cut).
    /// NULL handle when not in the active target's PTX bundle → dispatch
    /// uses `gdn_wy17_k` (all slots) unchanged. Gated by `ATLAS_WY17_LAZY`.
    gdn_wy17_lazy_k: KernelHandle,
    /// Combined LAZY + V-DIM SPLIT wy17 (`gated_delta_rule_wy17_lazy_vsplit`):
    /// fuses both benefits — checkpoint-gated Hi-writes (lazy_j) AND vsplit
    /// occupancy (96 CTAs on 48 SMs). Preferred over `gdn_wy17_lazy_k` when
    /// ATLAS_WY17_SPLIT is also set. NULL on targets that predate the kernel.
    gdn_wy17_lazy_vsplit_k: KernelHandle,
    /// Replay kernel (`gated_delta_rule_wy17_replay`) that reconstructs one
    /// skipped intermediate slot bit-exactly for the commit path under
    /// lazy_j>1. NULL when not compiled. Exposed for async_chkpt wiring.
    pub(crate) gdn_wy17_replay_k: KernelHandle,
    /// Tree-aware GDN verify kernel for DDTree verify with non-flat branches.
    /// Sequential per-token loop with parent_ids state load. NULL handle
    /// when not compiled for the active target.
    pub(crate) gdn_tree_k: KernelHandle,
    /// Tree-aware WY-fused GDN verify kernel. Bit-equivalent to wy17 on
    /// flat chains, supports arbitrary tree topology via ancestor walk in
    /// WY correction. Preferred over `gdn_tree_k` when present.
    pub(crate) gdn_tree_wy_k: KernelHandle,
    /// Tree-aware conv state re-root (ATLAS_DDTREE_TREE_CONV_EXACT=1).
    /// Before processing token t, copies conv_inter[parent[t]] → conv_state
    /// when parent[t] != t-1 (branch token). Makes the conv1d shift register
    /// ancestor-exact so FREE_SLOTS branch commits are byte-oracle.
    /// NULL on targets without causal_conv1d_tree_reroot compiled in.
    pub(crate) conv1d_tree_reroot_k: KernelHandle,
    // State allocation sizes (pre-computed from config)
    h_state_bytes: usize,
    conv_state_bytes: usize,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    qkvz_fp8: Option<DevicePtr>,
    out_proj_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    fp8_gemm_t_m128_k: KernelHandle, // M128: halves B re-reads for out_proj at ISL > 128
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Qwen3SsmLayer>();
};

// ── Sub-modules ───────────────────────────────────────────────────────────
mod act_e4m3_arena;
pub use act_e4m3_arena::SharedActE4m3Scratch;
mod debug;
mod exact_projection;
#[cfg(test)]
mod exact_projection_tests;
#[cfg(test)]
mod flashinfer_prefill_tests;
mod init;
#[cfg(test)]
mod prefill_gdn_gatecache_tests;
#[cfg(test)]
mod prefill_pack_tests;
#[cfg(test)]
mod prefill_sync_contract_tests;
mod qwen4_ba_gemm;
mod qwen4_dense_fp8_arm;
mod qwen4_gdn_conv_v2;
mod qwen4_gdn_fi;
pub(crate) use qwen4_gdn_fi::reserve_bridge as reserve_qwen4_gdn_fi_bridge;
mod qwen4_cross_seq_verify;
pub(crate) use qwen4_cross_seq_verify::CROSS_SEQ_TABLE_BYTES;
mod qwen4_gdn_fi_policy;
mod qwen4_gdn_norm_fp8;
mod qwen4_k5_ssm;
mod qwen4_native_multi_seq;
mod qwen4_prefill_check;
mod qwen4_prefill_check_raw;
pub(crate) mod qwen4_prefill_exact;
mod qwen4_prefill_exact_forward;
mod qwen4_prefill_exact_plan;
mod qwen4_prefill_gdn_fi_dump;
mod qwen4_prefill_gdn_tf;
pub(crate) mod qwen4_prefill_gemm;
mod qwen4_prefill_moe;
mod serial_diag;
mod ssm_forward;
mod trait_decode;
mod trait_decode_batched;
mod trait_decode_batched_conv_gdn;
mod trait_decode_cross_seq_dflash;
pub(crate) use trait_decode_cross_seq_dflash::DFLASH_SSM_PTR_TABLE_BYTES;
mod trait_decode_multi_seq;
mod trait_prefill;
mod trait_prefill_gdn;
mod trait_prefill_helper;
mod trait_prefill_phase1;
mod trait_prefill_phase3;
mod trait_prefill_proj;
mod trait_prefill_recur;

// ── Phase 2 profiling helpers (ATLAS_SSM_KERNEL_PROFILE=1) ────────────────
// Accumulate per-call wall-clock ns spent in wy17 across all SSM layers.
// Reporter logs total + call count every N=480 calls (10 verify steps at
// 48 SSM layers per step). Zero overhead when disabled.
static SSM_PROFILE_NS_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SSM_PROFILE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn ssm_profile_record(ns: u64) {
    SSM_PROFILE_NS_TOTAL.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    let n = SSM_PROFILE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // Report every 480 calls (= 10 verify steps × 48 SSM layers).
    if n.is_multiple_of(480) {
        let total_ns = SSM_PROFILE_NS_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let mean_us = (total_ns as f64) / (n as f64) / 1_000.0;
        tracing::info!(
            "ATLAS_SSM_KERNEL_PROFILE: wy17 calls={n}, total={:.3}ms, mean={:.2}us/call",
            (total_ns as f64) / 1_000_000.0,
            mean_us
        );
    }
}

// ── TransformerLayer impl (delegates to per-file inherent _inner methods) ──
impl TransformerLayer for Qwen3SsmLayer {
    fn qwen4_batch_verify_lane_ranges(
        &self,
        slot_idx: usize,
        gpu: &dyn GpuBackend,
        _kv_cache: &PagedKvCache,
        config: &atlas_core::config::ModelConfig,
        _writable_blocks: &[u32],
    ) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
        let configured_lanes = crate::layers::qwen4_verify_lanes::lanes();
        ensure!(
            config.is_qwen4_exp() && configured_lanes > 1 && slot_idx < configured_lanes,
            "batch GDN retain export requires a distinct physical Qwen4 lane"
        );
        ensure!(
            self.qwen4_attn_hyper.is_some()
                && self.qwen4_mlp_hyper.is_some()
                && self.sequential_qkvz
                && self.qkvz_nvfp4.is_some()
                && self.gdn_seq_lazy_engaged_inner(5),
            "batch GDN retain export requires the supported exact K5 lazy route"
        );
        let q = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim);
        let v = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim);
        let qkvz_size = q
            .zip(v)
            .and_then(|(q, v)| q.checked_add(v))
            .and_then(|half| half.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("batch GDN retain model geometry overflow"))?;
        ensure!(
            qkvz_size > 0 && config.linear_num_value_heads > 0,
            "batch GDN retain model geometry is empty"
        );
        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)?;
        let _ = self.gdn_lazy_retain_pool_with_layers(
            gpu,
            qkvz_size,
            config.linear_num_value_heads,
            Some(count_ssm_layers(config)),
        )?;
        let current =
            self.gdn_lazy_retain_current(gpu, qkvz_size, config.linear_num_value_heads)?;
        let pool = self
            .gdn_lazy_retain
            .get()
            .ok_or_else(|| anyhow::anyhow!("actual GDN retain allocation is unavailable"))?;
        ensure!(
            pool.lanes == configured_lanes,
            "GDN retain allocation belongs to a different lane registration"
        );
        let range = crate::layers::qwen4_qsa::checked_batch_verify_lane_range(
            pool.base,
            pool.allocation_bytes,
            pool.lane_stride,
            pool.lanes,
            slot_idx,
        )?;
        ensure!(
            range.2.offset(pool.active_buf_offset(slot_idx)) == current,
            "GDN retain export differs from the actual selected lane"
        );
        Ok(vec![range])
    }

    fn qwen4_prefill_finish_moe(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mlp = self.qwen4_mlp_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("expert-major prefill needs the Qwen4 MLP hyperconnection")
        })?;
        crate::layers::qwen4_prefill_moe::finish(
            mlp, &self.ffn, hidden, residual, rows, ctx, stream,
        )?;
        crate::model::qwen4_prefill_engagement::engage(
            crate::model::qwen4_prefill_engagement::PrefillPath::Ssm,
            rows,
        )
    }
    fn set_qwen4_hyperconnections(
        &mut self,
        attn: crate::layers::Qwen4HyperConnection,
        mlp: crate::layers::Qwen4HyperConnection,
    ) -> Result<()> {
        Qwen3SsmLayer::set_qwen4_hyperconnections(self, attn, mlp);
        Ok(())
    }

    fn qwen4_attn_hyper(&self) -> Option<&crate::layers::qwen4_hyper::Qwen4HyperConnection> {
        self.qwen4_attn_hyper.as_ref()
    }

    // ── MoE transpose hooks ──
    //
    // These were previously left to the trait's default `Ok(())`, which is a
    // SILENT no-op. Only `Qwen3AttentionLayer` implemented them, so on
    // Qwen3.8-Flash-Next — 36 SSM layers and 12 full-attention layers — a
    // whole-model transpose pass actually transposed 12 of 48 layers and
    // reported success. The two visible symptoms: `ATLAS_NVFP4_MOE_WORKLIST=1`
    // failed at "Prefill chunk layer 0 ... requires transposed gate/up expert
    // pointer tables" (layer 0 is an SSM layer), and `ATLAS_UNIFIED_MOE_LAYOUT=1`
    // measured as a pure loss because it switched decode to the `_t` kernels
    // while three quarters of the layers had no `_t` weights to use.
    //
    // Unlike the attention layer there is no `moe_ffn` here; the SSM layer owns
    // exactly one FFN.
    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_stream_transpose_scratch(
        &mut self,
        scratch: crate::layer::MoeStreamTransposeScratch,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_moe_stream_transpose_scratch(scratch);
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.flush_keep_late_for_layer_state(
            ctx.gpu,
            ctx.config,
            state,
            ctx.graph_capture,
            stream,
        )?;
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if h_intermediate.is_null()
            || num_tokens <= 1
            || !self.has_single_seq_staged_for_state(state)
        {
            self.flush_keep_late_for_layer_state(
                ctx.gpu,
                ctx.config,
                state,
                ctx.graph_capture,
                stream,
            )?;
        }
        self.decode_qwen4_batched_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            h_intermediate,
            conv_intermediate,
            h_intermediate_stride,
            conv_intermediate_stride,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_native_target_ssm<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.gdn_keep_late_any_pending() {
            for state in states.iter_mut() {
                self.flush_keep_late_for_layer_state(
                    ctx.gpu,
                    ctx.config,
                    *state,
                    ctx.graph_capture,
                    stream,
                )?;
            }
        }
        self.decode_qwen4_native_multi_seq(hidden, residual, num_seqs, states, ctx, stream)
    }

    fn release_qwen4_cross_seq_resources(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.release_keep_late_ctrl_slab();
        self.release_multi_seq_ptr_ring(gpu)?;
        self.release_cross_seq_ptr_table(gpu)
    }

    fn qwen4_cross_seq_admits_frames(&self) -> bool {
        self.cross_seq_table_admits_frames()
    }

    fn preflight_qwen4_cross_seq_verify(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        _qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_verify_inner(frame, ctx)
    }

    fn decode_qwen4_cross_seq_verify(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: crate::layer::Qwen4CrossSeqFrame,
        seqs: &mut [crate::layer::Qwen4CrossSeqRows<'_>],
        _kv_cache: &mut PagedKvCache,
        _staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_cross_seq_verify_inner(hidden, residual, frame, seqs, ctx, stream)
    }

    fn preflight_qwen4_cross_seq_mixer_only(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        _qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_mixer_only_inner(frame, ctx)
    }

    fn decode_qwen4_cross_seq_mixer_only(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: crate::layer::Qwen4CrossSeqFrame,
        seqs: &mut [crate::layer::Qwen4CrossSeqRows<'_>],
        _kv_cache: &mut PagedKvCache,
        _staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_cross_seq_mixer_only_inner(hidden, residual, frame, seqs, ctx, stream)
    }

    fn preflight_qwen4_native_target(
        &self,
        num_seqs: usize,
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_native_multi_seq(num_seqs, states, ctx)
    }

    fn qwen4_native_capability(
        &self,
        num_seqs: usize,
        ctx: &ForwardContext,
    ) -> std::result::Result<(), crate::qwen4_batch_policy::NativeGap> {
        use crate::qwen4_batch_policy::{NativeGap, SerialReason};
        // The MoE half first, so a MoE gap (including a bounded-transpose
        // layout) is named as such rather than as a recurrent-layer gap.
        self.ffn
            .preflight_qwen4_native_exact_rows(num_seqs, ctx)
            .map_err(|error| NativeGap::new(SerialReason::MoeExactRowsUnavailable, error))?;
        self.qwen4_native_multi_seq_capability(num_seqs, ctx)
            .map_err(|error| NativeGap::new(SerialReason::SsmLayer, error))
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.flush_keep_late_for_layer_state(
            ctx.gpu,
            ctx.config,
            state,
            ctx.graph_capture,
            stream,
        )?;
        self.decode_batched_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn run_deferred_ffn(
        &self,
        ffn_input: DevicePtr,
        hidden: DevicePtr,
        total_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.run_deferred_ffn_inner(ffn_input, hidden, total_rows, ctx, stream)
    }

    fn stage_dflash_cross_seq_ssm_ptr_table(
        &self,
        num_seqs: usize,
        k: usize,
        slot_ids: &[usize],
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
        dst: &mut [u8],
    ) -> Result<bool> {
        self.stage_dflash_cross_seq_ssm_ptr_table_inner(num_seqs, k, slot_ids, states, ctx, dst)
    }

    fn decode_ssm_batched_cross_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        k: usize,
        slot_ids: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.decode_ssm_batched_cross_seq_inner(
            hidden, residual, num_seqs, k, slot_ids, states, ctx, stream,
        )
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.gdn_keep_late_any_pending() {
            for state in states.iter_mut() {
                self.flush_keep_late_for_layer_state(
                    ctx.gpu,
                    ctx.config,
                    *state,
                    ctx.graph_capture,
                    stream,
                )?;
            }
        }
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.flush_keep_late_for_layer_state(
            ctx.gpu,
            ctx.config,
            state,
            ctx.graph_capture,
            stream,
        )?;
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    fn is_ssm_layer(&self) -> bool {
        self.is_ssm_layer_inner()
    }

    fn prefill_phase1(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.flush_keep_late_for_layer_state(
            ctx.gpu,
            ctx.config,
            state,
            ctx.graph_capture,
            stream,
        )?;
        self.prefill_phase1_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn prefill_gdn_full(
        &self,
        state: &mut dyn LayerState,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_inner(state, gdn_bufs, ctx, stream)
    }

    fn prefill_gdn_full_batched(
        &self,
        h_state_ptrs: DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_gdn_full_batched_inner(
            h_state_ptrs,
            gdn_bufs,
            batch_size,
            chunk_len,
            ctx,
            stream,
        )
    }

    fn prefill_phase3(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_phase3_inner(
            hidden,
            residual,
            num_tokens,
            gdn_bufs,
            token_offset,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        self.alloc_state_inner(gpu)
    }

    fn multiseq_graph_safe(&self, num_seqs: usize) -> bool {
        self.multi_seq_kernel_path_active(num_seqs)
    }

    fn multiseq_refresh_ptr_table<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        num_seqs: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.refresh_multi_seq_ptr_table(states, num_seqs, gpu, stream)
    }

    fn wy17_replay_kernel(&self) -> KernelHandle {
        self.gdn_wy17_replay_k
    }

    fn gdn_seq_lazy_engaged(&self, num_tokens: usize) -> bool {
        self.gdn_seq_lazy_engaged_inner(num_tokens)
    }

    fn gdn_seq_replay_kernel(&self) -> KernelHandle {
        self.gdn_f32_sequence_nosnap_k
    }

    fn gdn_seq_lazy_retain(&self) -> Option<DevicePtr> {
        let pool = self.gdn_lazy_retain.get()?;
        let lane = crate::layers::qwen4_verify_lanes::current_lane();
        // A lane past the count this buffer was built with has no retain
        // storage; the commit then fails closed ("retention buffer missing").
        (lane < pool.lanes).then(|| pool.retain_active_ptr(lane))
    }

    fn gdn_keep_late_supported(&self) -> bool {
        self.gdn_keep_late_supported_inner()
    }

    fn gdn_keep_late_armed(&self, slot: usize) -> bool {
        self.gdn_keep_late_armed_inner(slot)
    }

    fn gdn_keep_late_note_verify_completed(&self, slot: usize) {
        self.gdn_keep_late_note_verify_completed_inner(slot);
    }

    fn gdn_keep_late_reconcile_abort(&self, gpu: &dyn GpuBackend, slot: usize) -> Result<()> {
        self.gdn_keep_late_reconcile_abort_inner(gpu, slot)
    }

    fn gdn_keep_late_record(
        &self,
        slot: usize,
        num_committed: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        conv_inter_base: DevicePtr,
        conv_inter_stride: usize,
        h_scratch: DevicePtr,
    ) -> bool {
        self.gdn_keep_late_record_inner(
            slot,
            num_committed,
            h_state,
            conv_state,
            conv_inter_base,
            conv_inter_stride,
            h_scratch,
        )
    }

    fn gdn_keep_late_clear_pending(&self, slot: usize) {
        self.gdn_keep_late_clear_pending_inner(slot);
    }

    fn gdn_keep_late_reset_slot(&self, slot: usize) {
        self.gdn_keep_late_reset_slot_inner(slot);
    }

    fn gdn_keep_late_has_pending(&self, slot: usize) -> bool {
        self.gdn_keep_late_has_pending_inner(slot)
    }

    fn gdn_keep_late_has_scaled_fp16(&self, slot: usize) -> bool {
        self.gdn_keep_late_has_scaled_fp16_inner(slot)
    }

    fn gdn_keep_late_flush_slot(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        conv_ckpt: Option<DevicePtr>,
        conv_inter_base: DevicePtr,
        conv_inter_stride: usize,
        h_scratch: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.gdn_keep_late_flush_slot_inner(
            gpu,
            config,
            slot,
            h_state,
            conv_state,
            conv_ckpt,
            conv_inter_base,
            conv_inter_stride,
            h_scratch,
            stream,
        )
    }

    fn ensure_qwen4_cross_seq_fused_descs_ready(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        k: usize,
    ) -> Result<()> {
        self.ensure_qwen4_cross_seq_fused_descs_ready_inner(gpu, config, k)
    }

    fn stage_qwen4_cross_seq_fused_desc_bytes(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        seqs: &[crate::layer::Qwen4CrossSeqRows<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, Vec<u8>)>> {
        self.stage_qwen4_cross_seq_fused_desc_bytes_inner(frame, seqs, ctx, stream)
    }

    fn stage_qwen4_keep_late_ctrl_bytes(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        h_state: DevicePtr,
    ) -> Result<Option<(DevicePtr, [u8; 32])>> {
        self.stage_qwen4_keep_late_ctrl_bytes_inner(gpu, config, slot, h_state)
    }

    fn stage_qwen4_single_seq_fused_desc_bytes(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        k: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        conv_inter_base: DevicePtr,
        stream: u64,
    ) -> Result<Option<(DevicePtr, [u8; 64])>> {
        self.stage_qwen4_single_seq_fused_desc_bytes_inner(
            gpu,
            config,
            slot,
            k,
            h_state,
            conv_state,
            conv_inter_base,
            stream,
        )
    }

    fn gdn_tree_kernel_loaded(&self) -> bool {
        self.gdn_tree_k.0 != 0
    }

    fn ddtree_conv_state_exact(&self) -> bool {
        self.conv1d_tree_reroot_k.0 != 0
    }

    fn wy17_lazy_engaged(&self, num_tokens: usize) -> bool {
        // MUST mirror the `use_lazy` computation in
        // `trait_decode_batched_conv_gdn.rs` (wy17 branch) exactly: the lazy
        // wy17 kernel skips non-checkpoint intermediate H writes, and the
        // commit may replay a skipped slot ONLY when that kernel actually
        // produced this verify's intermediates. Everything here is a pure
        // function of `num_tokens` + process-constant env gates + kernel
        // handles, so it is safe under CUDA-graph replay (unlike per-step
        // mutable state, which would go stale on replayed steps).
        if num_tokens != 17 || self.gdn_wy17_k.0 == 0 {
            // The lazy kernel is only dispatched from the `num_tokens == 17`
            // branch; every other K runs chunked/fused kernels that persist
            // ALL intermediate slots.
            return false;
        }
        let use_vsplit = crate::layers::wy17_split() > 0 && self.gdn_wy17_vsplit_k.0 != 0;
        let lazy_base = crate::layers::wy17_lazy() > 1
            && self.gdn_wy17_replay_k.0 != 0
            && crate::layers::wy17_lazy_commit();
        // lazy is engaged when the plain lazy kernel runs (no vsplit) OR when
        // the combined lazy_vsplit kernel covers both benefits (vsplit active).
        lazy_base
            && ((self.gdn_wy17_lazy_k.0 != 0 && !use_vsplit)
                || (use_vsplit && self.gdn_wy17_lazy_vsplit_k.0 != 0))
    }
}

/// Retention rows for the lazy ExactSequence commit. γ+1 = 16 at the serving
/// γ=15; MTP verifies are smaller. `gdn_seq_lazy_engaged` refuses larger k.
pub const GDN_LAZY_MAX_K: usize = 16;

/// Device bytes of one lazy-commit retain lane: `GDN_LAZY_MAX_K` FP32 conv
/// rows of `qkvz_size` plus the gate/beta block (`num_v_heads * 2` FP32 per
/// row) at offset `GDN_LAZY_MAX_K * qkvz_size * 4`.
pub fn gdn_lazy_retain_lane_bytes(qkvz_size: usize, num_v_heads: usize) -> anyhow::Result<usize> {
    let rows = GDN_LAZY_MAX_K
        .checked_mul(qkvz_size)
        .and_then(|elements| elements.checked_mul(4));
    let gates = GDN_LAZY_MAX_K
        .checked_mul(num_v_heads)
        .and_then(|elements| elements.checked_mul(2 * 4));
    rows.zip(gates)
        .and_then(|(rows, gates)| rows.checked_add(gates))
        .ok_or_else(|| anyhow::anyhow!("GDN lazy retain lane extent overflow"))
}

/// Exact per-lane device byte stride allocated by [`GdnLazyRetainPool`] for
/// one SSM layer:
/// - When `keep_late_enabled == false`: one legacy 16-row FP32 retain buffer
///   ([`gdn_lazy_retain_lane_bytes`]).
/// - When `keep_late_enabled == true`: compact [`KEEP_LATE_RING_CAP`] (`64`)
///   row circular ring (`raw BF16 [Q, K, V]` + `FP32 [g, beta]`, 16-byte
///   aligned, e.g. `1,335,296 B/lane` on TP1 Flash-Next and `667,648 B/lane`
///   on TP2), lower-bounded by one legacy FP32 retain buffer so cold
///   non-`KEEP_LATE` fallback remains bounds-safe.
///
/// Both runtime allocation (`gdn_lazy_retain_pool_with_layers`) and server
/// preflight (`preflight.rs`) call this function so VRAM reservations match
/// runtime allocation byte-for-byte.
pub fn gdn_lazy_retain_lane_stride_bytes(
    qkvz_size: usize,
    num_v_heads: usize,
    keep_late_enabled: bool,
) -> anyhow::Result<usize> {
    let legacy_buf = gdn_lazy_retain_lane_bytes(qkvz_size, num_v_heads)?;
    if !keep_late_enabled {
        return Ok(legacy_buf);
    }
    // `KEEP_LATE` requires `vd == 128`, so `val_dim = num_v_heads * 128` and
    // `conv_dim = 2 * key_dim + val_dim = qkvz_size - val_dim`. Fall back to
    // `2 * legacy_buf` only for non-standard geometries where `qkvz_size <= val_dim`.
    let val_dim = num_v_heads
        .checked_mul(128)
        .ok_or_else(|| anyhow::anyhow!("GDN keep-late val_dim overflow"))?;
    let ring_bytes = if let Some(conv_dim) = qkvz_size.checked_sub(val_dim).filter(|&d| d > 0) {
        let row_bytes = conv_dim
            .checked_mul(2)
            .and_then(|b| num_v_heads.checked_mul(8).and_then(|g| b.checked_add(g)))
            .and_then(|raw| raw.checked_add(15).map(|aligned| aligned & !15))
            .ok_or_else(|| anyhow::anyhow!("GDN keep-late compact row extent overflow"))?;
        KEEP_LATE_RING_CAP
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("GDN keep-late ring extent overflow"))?
    } else {
        legacy_buf
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("GDN lazy retain ring stride overflow"))?
    };
    Ok(legacy_buf.max(ring_bytes))
}

/// Effective per-lane retain stride (bytes) for the current process's
/// `ATLAS_QWEN4_GDN_KEEP_LATE` configuration.
pub fn gdn_lazy_retain_effective_lane_bytes(
    qkvz_size: usize,
    num_v_heads: usize,
) -> anyhow::Result<usize> {
    gdn_lazy_retain_lane_stride_bytes(qkvz_size, num_v_heads, gdn_keep_late_enabled())
}

/// Flag bit in `GdnFusedMultiSeqDesc::flags` enabling zero-replay `KEEP_LATE`
/// circular-ring commit folding inside `qwen4_gdn_verify_fused_lazyfinal_multi_seq`.
pub(super) const GDN_FUSED_FLAG_KEEP_LATE: u32 = ops::GDN_FUSED_FLAG_KEEP_LATE;
/// Flag bit in `GdnFusedMultiSeqDesc::flags` indicating `desc.h_state_inter`
/// holds the slot-constant `GdnKeepLateCtrl*` device pointer (100% graph-safe).
pub(super) const GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR: u32 = ops::GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;
/// Flag bit in `GdnFusedMultiSeqDesc::flags` selecting in-place per-head
/// Scaled-FP16 `h_state` loads (`32 KiB` `__half` + `f32` scale at `32768`).
pub(super) const GDN_FUSED_FLAG_LOAD_SCALED_FP16: u32 = ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16;
/// Flag bit in `GdnFusedMultiSeqDesc::flags` selecting in-place per-head
/// Scaled-FP16 `h_state` stores on `KEEP_LATE` flush steps.
pub(super) const GDN_FUSED_FLAG_STORE_SCALED_FP16: u32 = ops::GDN_FUSED_FLAG_STORE_SCALED_FP16;
/// Flag bit in `GdnFusedMultiSeqDesc::flags` selecting in-place per-value-channel
/// Scaled-FP8 E4M3 `h_state` loads (`16 KiB` E4M3 + `256 B` FP16 scales at `16384`).
pub(super) const GDN_FUSED_FLAG_LOAD_SCALED_FP8: u32 = 64;
/// Flag bit in `GdnFusedMultiSeqDesc::flags` selecting in-place per-value-channel
/// Scaled-FP8 E4M3 `h_state` stores on `KEEP_LATE` flush steps.
pub(super) const GDN_FUSED_FLAG_STORE_SCALED_FP8: u32 = 128;
/// Exact byte size matching CUDA `sizeof(GdnKeepLateCtrl)`.
pub(super) const GDN_KEEP_LATE_CTRL_BYTES: usize = ops::GdnKeepLateCtrl::SIZE_BYTES;
/// Environment toggle for zero-replay `KEEP_LATE` DeltaNet commits (default ON
/// unless set to `"0"`, when `compact_lazy_h` + fused GDN verify are active).
pub const GDN_KEEP_LATE_ENV: &str = "ATLAS_QWEN4_GDN_KEEP_LATE";
/// Environment toggle for in-place per-head Scaled-FP16 `h_state` DRAM
/// compression under `KEEP_LATE` at `C >= 2` (default ON unless `"0"`).
pub const GDN_H_FP16_ENV: &str = "ATLAS_QWEN4_GDN_H_FP16";
/// Opt-in environment toggle for in-place per-value-channel (128-element)
/// Scaled-FP8 E4M3 `h_state` DRAM compression under `KEEP_LATE` at `C >= 2`
/// (`16.25 KiB/head`, default OFF until live needle-in-a-haystack verification).
pub const GDN_H_FP8_ENV: &str = "ATLAS_QWEN4_GDN_H_FP8";
/// Total active bytes per `[128, 128]` head under per-channel Scaled-FP8 E4M3
/// (`16,384 B` E4M3 payload + `128 * 2 B` FP16 scales = `16,640 B`).
pub(super) const GDN_H_FP8_HEAD_BYTES: usize = 16_384 + 128 * 2;
/// Minimum active verify cohort size (`C >= 2`) required to store `h_state` as
/// in-place per-head Scaled-FP16 (`C == 1` remains 100% 0-ULP FP32).
pub(super) const GDN_H_FP16_MIN_COHORT: usize = 2;
/// Environment override for staggered multi-step `KEEP_LATE` flush period `M`
/// (clamped to `1..=64`; auto mode uses `1` at `C = 1` and `2` at `C >= 2`).
pub const GDN_KEEP_LATE_FLUSH_PERIOD_ENV: &str = "ATLAS_QWEN4_GDN_KEEP_LATE_FLUSH_PERIOD";
/// Maximum compact rows addressed in one slot's circular `KEEP_LATE` ring
/// (`KEEP_LATE_L_MAX + GDN_LAZY_MAX_K = 48 + 16 = 64` rows: `1,335,296 B/lane`
/// on TP1 Flash-Next and `667,648 B/lane` on TP2, cutting `C=64` VRAM across
/// 36 SSM layers from `4.53 GiB` to `2.87 GiB` while guaranteeing
/// `pending_rows + k <= 47 + 16 = 63 < 64` with zero wrap-overlap).
pub(super) const KEEP_LATE_RING_CAP: usize = 64;
/// Pending-row threshold at which a slot forces a DRAM flush on the next verify
/// step (`pending_rows >= L_flush`). Guarantees accumulated rows stay `<= L_max`
/// (`31 + 16 = 47 <= 48`), leaving read+write `<= 47 + 16 = 63 < 64` inside the
/// compact 64-row ring.
pub(super) const KEEP_LATE_L_FLUSH: usize = 32;
/// Maximum unmaterialized committed rows accumulated in one slot's ring.
pub(super) const KEEP_LATE_L_MAX: usize = 48;
/// Default staggered flush period `M` (`(step_count + slot) % M == 0`) for
/// batched cohorts (`C >= 2`) on GB10-class GPUs (`sm_count <= 64`). With
/// in-place Scaled-FP16 `h_state` active at `C >= 2`, `M = 2` halves `h_state`
/// DRAM write traffic while capping deferred replay to 1 step.
pub(super) const KEEP_LATE_DEFAULT_FLUSH_PERIOD: usize = 2;
/// Default staggered flush period `M` for batched cohorts (`C >= 2`) on
/// high-SM Blackwell GPUs (`sm_count > 64`, e.g. SM120 188 SMs): cuts batched
/// `h_state` DRAM writes from 50% (`M = 2`) to 20% (`M = 5`) whenever the
/// retain ring satisfies `retain_capacity >= KEEP_LATE_L_MAX + GDN_LAZY_MAX_K`
/// (`64 <= 64`, since `should_flush_keep_late` caps `pending_rows < L_flush = 32`
/// before any deferred step).
pub(super) const KEEP_LATE_HIGH_SM_DEFAULT_FLUSH_PERIOD: usize = 5;
const KEEP_LATE_HIGH_SM_THRESHOLD: u32 = 64;

/// Parse an explicit `ATLAS_QWEN4_GDN_KEEP_LATE_FLUSH_PERIOD` override (`1..=64`).
#[inline]
#[must_use]
pub(super) fn parse_keep_late_flush_period_override(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&period| period >= 1)
        .map(|period| period.min(64))
}

/// Pure SM-aware resolver for the batched (`C >= 2`) `KEEP_LATE` flush period:
/// honors any explicit `ATLAS_QWEN4_GDN_KEEP_LATE_FLUSH_PERIOD` override
/// (`1..=64`), returning `2` for `sm_count <= 64` (GB10 unchanged) and `5` for
/// `sm_count > 64` (SM120/SM103).
#[inline]
#[must_use]
pub(super) fn keep_late_flush_period_for_sm(env: Option<&str>, sm_count: u32) -> usize {
    parse_keep_late_flush_period_override(env).unwrap_or(
        if sm_count > KEEP_LATE_HIGH_SM_THRESHOLD {
            KEEP_LATE_HIGH_SM_DEFAULT_FLUSH_PERIOD
        } else {
            KEEP_LATE_DEFAULT_FLUSH_PERIOD
        },
    )
}

/// Parse `ATLAS_QWEN4_GDN_KEEP_LATE_FLUSH_PERIOD` (`1..=64`, default `2`).
#[cfg(test)]
#[inline]
#[must_use]
pub(super) fn parse_keep_late_flush_period(raw: Option<&str>) -> usize {
    parse_keep_late_flush_period_override(raw).unwrap_or(KEEP_LATE_DEFAULT_FLUSH_PERIOD)
}

/// Effective staggered `KEEP_LATE` flush period for a cohort of `cohort_seqs`
/// active sequences on GB10 (`sm_count <= 64`): honors any explicit env
/// override; in auto mode uses `1` for `cohort_seqs <= 1` (`C = 1` 0-ULP FP32
/// exact path) and `M = 2` for all batched cohorts (`cohort_seqs >= 2`).
#[cfg(test)]
#[inline]
#[must_use]
pub(super) const fn resolve_keep_late_flush_period(
    override_period: Option<usize>,
    cohort_seqs: usize,
) -> usize {
    resolve_keep_late_flush_period_for_sm(
        override_period,
        cohort_seqs,
        KEEP_LATE_HIGH_SM_THRESHOLD,
        KEEP_LATE_RING_CAP,
        GDN_LAZY_MAX_K,
    )
}

/// Effective staggered `KEEP_LATE` flush period for a cohort of `cohort_seqs`
/// active sequences on a GPU with `sm_count` SMs and per-slot ring capacity
/// `retain_capacity`:
/// - Explicit env override (`Some(period)`) always wins.
/// - `cohort_seqs <= 1` (`C = 1`) stays `1` (0-ULP FP32 exact path).
/// - `sm_count <= 64` (GB10) returns `2`.
/// - `sm_count > 64` (SM120/SM103) returns `5` only when
///   `retain_capacity >= KEEP_LATE_L_MAX + max_k` (`64` rows, matching
///   `should_flush_keep_late`'s `pending_rows < 32 => pending_rows + k <= 63`),
///   otherwise falls back safely to `2`.
#[inline]
#[must_use]
pub(super) const fn resolve_keep_late_flush_period_for_sm(
    override_period: Option<usize>,
    cohort_seqs: usize,
    sm_count: u32,
    retain_capacity: usize,
    max_k: usize,
) -> usize {
    match override_period {
        Some(period) => period,
        None if cohort_seqs <= 1 => 1,
        None => {
            if sm_count > KEEP_LATE_HIGH_SM_THRESHOLD {
                match KEEP_LATE_L_MAX.checked_add(max_k) {
                    Some(needed) if retain_capacity >= needed => {
                        KEEP_LATE_HIGH_SM_DEFAULT_FLUSH_PERIOD
                    }
                    _ => KEEP_LATE_DEFAULT_FLUSH_PERIOD,
                }
            } else {
                KEEP_LATE_DEFAULT_FLUSH_PERIOD
            }
        }
    }
}

fn gdn_keep_late_flush_period_env_raw() -> Option<&'static str> {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    RAW.get_or_init(|| std::env::var(GDN_KEEP_LATE_FLUSH_PERIOD_ENV).ok())
        .as_deref()
}

fn gdn_keep_late_flush_period_override() -> Option<usize> {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *OVERRIDE
        .get_or_init(|| parse_keep_late_flush_period_override(gdn_keep_late_flush_period_env_raw()))
}

/// Process-constant staggered `KEEP_LATE` flush period `M` (env override or
/// SM-aware batched default: `2` on GB10 `sm_count <= 64`, `5` on SM120+
/// `sm_count > 64`).
pub fn gdn_keep_late_flush_period() -> usize {
    keep_late_flush_period_for_sm(
        gdn_keep_late_flush_period_env_raw(),
        atlas_core::device::active_sm_count(),
    )
}

thread_local! {
    static KEEP_LATE_COHORT_SEQS: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}

/// Scoped RAII guard exposing the active cross-sequence verify cohort size to
/// `stage_qwen4_keep_late_ctrl_bytes_inner` without widening `Model` traits.
pub(crate) struct KeepLateCohortSeqsGuard {
    prev: usize,
}

impl KeepLateCohortSeqsGuard {
    #[must_use]
    pub(crate) fn enter(cohort_seqs: usize) -> Self {
        let prev = KEEP_LATE_COHORT_SEQS.with(|cell| cell.replace(cohort_seqs.max(1)));
        Self { prev }
    }
}

impl Drop for KeepLateCohortSeqsGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        KEEP_LATE_COHORT_SEQS.with(|cell| cell.set(prev));
    }
}

#[inline]
fn current_keep_late_cohort_seqs() -> usize {
    KEEP_LATE_COHORT_SEQS.with(std::cell::Cell::get)
}

#[inline]
fn gdn_keep_late_flush_period_for_cohort(cohort_seqs: usize, retain_capacity: usize) -> usize {
    resolve_keep_late_flush_period_for_sm(
        gdn_keep_late_flush_period_override(),
        cohort_seqs,
        atlas_core::device::active_sm_count(),
        retain_capacity,
        GDN_LAZY_MAX_K,
    )
}

/// Parse `ATLAS_QWEN4_GDN_H_FP16` (default ON unless trimmed value is `"0"`).
#[inline]
#[must_use]
pub(super) fn parse_gdn_h_fp16_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// Process-constant gate for Tier-3 in-place per-head Scaled-FP16 `h_state`
/// DRAM compression under `KEEP_LATE` (`ATLAS_QWEN4_GDN_H_FP16`, default ON).
pub fn gdn_h_fp16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let raw = std::env::var(GDN_H_FP16_ENV).ok();
        parse_gdn_h_fp16_enabled(raw.as_deref())
    })
}

/// Parse `ATLAS_QWEN4_GDN_H_FP8` (default OFF; opt-in via `"1"`, `"true"`, `"on"`).
#[inline]
#[must_use]
pub(super) fn parse_gdn_h_fp8_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim),
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
    )
}

/// Process-constant opt-in gate for in-place per-value-channel Scaled-FP8 E4M3
/// `h_state` DRAM compression under `KEEP_LATE` (`ATLAS_QWEN4_GDN_H_FP8`, default OFF).
pub fn gdn_h_fp8_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let raw = std::env::var(GDN_H_FP8_ENV).ok();
        parse_gdn_h_fp8_enabled(raw.as_deref())
    })
}

/// Environment gate for Split-V (`SPLIT_V = 4`) multi-CTA GDN decode/verify
/// kernels (`"auto"` / unset enables for `C = 1` on `active_sm_count() > 64`;
/// `"0"` / `"false"` disables; `"1"` / `"true"` forces on for FP32 H-state).
const GDN_SPLIT_V_ENV: &str = "ATLAS_QWEN4_GDN_SPLIT_V";
const GDN_SPLIT_V_SM_THRESHOLD: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GdnSplitVPolicy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
pub(super) fn parse_gdn_split_v_policy(raw: Option<&str>) -> GdnSplitVPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => GdnSplitVPolicy::Disabled,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => GdnSplitVPolicy::Enabled,
        _ => GdnSplitVPolicy::Auto,
    }
}

/// Pure eligibility resolver for Split-V (`SPLIT_V = 4`) multi-CTA GDN kernels.
///
/// Always returns `false` when `h_format != KeepLateHFormat::Fp32` so
/// concurrent V-slice CTAs never race on an in-place Scaled-FP16 <-> FP32
/// transition. In `Auto` mode, engages only for single-sequence (`num_seqs == 1`)
/// workloads on GPUs with `sm_count > 64` (saturating 188-SM SM120 / SM103
/// with 192 CTAs instead of 48).
#[must_use]
pub(super) const fn resolve_gdn_split_v_enabled(
    policy: GdnSplitVPolicy,
    sm_count: u32,
    num_seqs: usize,
    h_format: KeepLateHFormat,
) -> bool {
    if !matches!(h_format, KeepLateHFormat::Fp32) || num_seqs == 0 {
        return false;
    }
    match policy {
        GdnSplitVPolicy::Disabled => false,
        GdnSplitVPolicy::Auto => sm_count > GDN_SPLIT_V_SM_THRESHOLD && num_seqs == 1,
        GdnSplitVPolicy::Enabled => num_seqs == 1,
    }
}

fn gdn_split_v_policy() -> GdnSplitVPolicy {
    static POLICY: std::sync::OnceLock<GdnSplitVPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var(GDN_SPLIT_V_ENV).ok();
        parse_gdn_split_v_policy(raw.as_deref())
    })
}

/// Whether Split-V (`SPLIT_V = 4`) multi-CTA GDN decode/verify is active for
/// canonical FP32 single-sequence (`C = 1`) execution on the current GPU.
#[must_use]
pub fn qwen4_gdn_split_v_enabled_for_single_seq() -> bool {
    resolve_gdn_split_v_enabled(
        gdn_split_v_policy(),
        atlas_core::device::active_sm_count(),
        1,
        KeepLateHFormat::Fp32,
    )
}

fn log_qwen4_gdn_h_fp16_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            "QWEN4_GDN_H_FP16_ENGAGED selector={} min_cohort={}",
            GDN_H_FP16_ENV,
            GDN_H_FP16_MIN_COHORT
        );
    });
}

fn log_ssm_gdn_h_fp8_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            "SSM_GDN_H_FP8_ENGAGED selector={} min_cohort={} head_bytes={}",
            GDN_H_FP8_ENV,
            GDN_H_FP16_MIN_COHORT,
            GDN_H_FP8_HEAD_BYTES
        );
    });
}

/// In-DRAM representation of one slot's live `h_state` buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum KeepLateHFormat {
    /// Canonical `64 KiB/head` FP32 `[128, 128]` matrix.
    #[default]
    Fp32,
    /// In-place per-head Scaled-FP16 (`32 KiB` row-major `__half` + `f32`
    /// scale at byte offset `32768` inside the existing `64 KiB` head slot).
    ScaledFp16,
    /// In-place per-value-channel Scaled-FP8 E4M3 (`16 KiB` row-major E4M3 +
    /// `128 x __half = 256 B` scales at byte offset `16384` inside the
    /// existing `64 KiB` head slot = `16.25 KiB/head`).
    ScaledFp8,
}

impl KeepLateHFormat {
    #[inline]
    #[must_use]
    pub(super) const fn is_scaled_fp16(self) -> bool {
        matches!(self, Self::ScaledFp16)
    }

    #[inline]
    #[must_use]
    pub(super) const fn is_scaled_fp8(self) -> bool {
        matches!(self, Self::ScaledFp8)
    }

    #[inline]
    #[must_use]
    pub(super) const fn is_compressed(self) -> bool {
        !matches!(self, Self::Fp32)
    }
}

/// Pure per-step `h_state` format and flush plan for one `KEEP_LATE` verify slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct KeepLateHFormatPlan {
    pub(super) load_format: KeepLateHFormat,
    pub(super) store_format: KeepLateHFormat,
    pub(super) flush_now: bool,
}

#[cfg(test)]
impl KeepLateHFormatPlan {
    /// Resulting `h_state` format after this step finishes on GPU.
    #[inline]
    #[must_use]
    pub(super) const fn next_format(self) -> KeepLateHFormat {
        if self.flush_now {
            self.store_format
        } else {
            self.load_format
        }
    }
}

/// Resolve load/store `h_state` format and flush flag for one verify step,
/// supporting canonical FP32, default-ON Scaled-FP16, and opt-in Scaled-FP8.
#[inline]
#[must_use]
pub(super) const fn plan_keep_late_h_format_ex(
    current: KeepLateHFormat,
    h_fp16_enabled: bool,
    h_fp8_enabled: bool,
    cohort_seqs: usize,
    base_flush_now: bool,
) -> KeepLateHFormatPlan {
    let target = if cohort_seqs >= GDN_H_FP16_MIN_COHORT {
        if h_fp8_enabled {
            KeepLateHFormat::ScaledFp8
        } else if h_fp16_enabled {
            KeepLateHFormat::ScaledFp16
        } else {
            KeepLateHFormat::Fp32
        }
    } else {
        KeepLateHFormat::Fp32
    };
    let force_flush = current.is_compressed()
        && !matches!(
            (current, target),
            (KeepLateHFormat::ScaledFp16, KeepLateHFormat::ScaledFp16)
                | (KeepLateHFormat::ScaledFp8, KeepLateHFormat::ScaledFp8)
        );
    let flush_now = base_flush_now || force_flush;
    KeepLateHFormatPlan {
        load_format: current,
        store_format: target,
        flush_now,
    }
}

/// Resolve load/store `h_state` format and flush flag for one verify step
/// (Scaled-FP16 helper kept for unit tests).
#[cfg(test)]
#[inline]
#[must_use]
pub(super) const fn plan_keep_late_h_format(
    current: KeepLateHFormat,
    h_fp16_enabled: bool,
    cohort_seqs: usize,
    base_flush_now: bool,
) -> KeepLateHFormatPlan {
    plan_keep_late_h_format_ex(current, h_fp16_enabled, false, cohort_seqs, base_flush_now)
}

/// Staged per-slot `KEEP_LATE` control pointer and compressed `h_state` format flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct KeepLateStagedCtrl {
    pub(super) ctrl_ptr: DevicePtr,
    pub(super) load_format: KeepLateHFormat,
    pub(super) store_format: KeepLateHFormat,
}

impl KeepLateStagedCtrl {
    /// Encode `GdnFusedMultiSeqDesc::flags` for this staged `KEEP_LATE` launch.
    #[inline]
    #[must_use]
    pub(super) const fn desc_flags(self) -> u32 {
        let mut flags = ops::GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN
            | GDN_FUSED_FLAG_KEEP_LATE
            | GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;
        if self.load_format.is_scaled_fp16() {
            flags |= GDN_FUSED_FLAG_LOAD_SCALED_FP16;
        } else if self.load_format.is_scaled_fp8() {
            flags |= GDN_FUSED_FLAG_LOAD_SCALED_FP8;
        }
        if self.store_format.is_scaled_fp16() {
            flags |= GDN_FUSED_FLAG_STORE_SCALED_FP16;
        } else if self.store_format.is_scaled_fp8() {
            flags |= GDN_FUSED_FLAG_STORE_SCALED_FP8;
        }
        flags
    }
}

/// Exact byte stride of one compact circular-ring row (`raw BF16 [Q, K, V]` +
/// `FP32 [raw_g, beta]`, rounded up to 16-byte alignment), matching
/// `geom.ring_row_bytes` in `qwen4_gdn_verify_fused.cu`.
#[inline]
#[must_use]
pub(super) const fn keep_late_ring_row_bytes(
    nk: usize,
    kd: usize,
    nv: usize,
    vd: usize,
) -> Option<usize> {
    let Some(key_dim) = nk.checked_mul(kd) else {
        return None;
    };
    let Some(val_dim) = nv.checked_mul(vd) else {
        return None;
    };
    let Some(two_key) = key_dim.checked_mul(2) else {
        return None;
    };
    let Some(conv_dim) = two_key.checked_add(val_dim) else {
        return None;
    };
    let Some(gates_off) = conv_dim.checked_mul(2) else {
        return None;
    };
    let Some(gates_bytes) = nv.checked_mul(8) else {
        return None;
    };
    let Some(raw_bytes) = gates_off.checked_add(gates_bytes) else {
        return None;
    };
    match raw_bytes.checked_add(15) {
        Some(aligned) => Some(aligned & !15),
        None => None,
    }
}

/// Number of compact rows that fit in one slot's `lane_stride` bytes, capped at
/// [`KEEP_LATE_RING_CAP`].
#[inline]
#[must_use]
pub(super) const fn keep_late_ring_capacity(
    lane_stride: usize,
    nk: usize,
    kd: usize,
    nv: usize,
    vd: usize,
) -> usize {
    let Some(row_bytes) = keep_late_ring_row_bytes(nk, kd, nv, vd) else {
        return 0;
    };
    if row_bytes == 0 {
        return 0;
    }
    let rows = lane_stride / row_bytes;
    if rows > KEEP_LATE_RING_CAP {
        KEEP_LATE_RING_CAP
    } else {
        rows
    }
}

/// Returns whether the slot's compact `KEEP_LATE` lane stride holds at least
/// `KEEP_LATE_L_MAX + GDN_LAZY_MAX_K` (`64`) compact ring rows (guaranteed
/// disjoint replay and verify write windows).
#[inline]
#[must_use]
pub(super) fn keep_late_ring_fits(
    nk: usize,
    kd: usize,
    nv: usize,
    vd: usize,
    qkvz_size: usize,
) -> bool {
    let Ok(lane_stride) = gdn_lazy_retain_lane_stride_bytes(qkvz_size, nv, true) else {
        return false;
    };
    keep_late_ring_capacity(lane_stride, nk, kd, nv, vd) >= KEEP_LATE_L_MAX + GDN_LAZY_MAX_K
}

/// Pure flush-scheduling policy for staggered multi-step ring `KEEP_LATE`:
/// flush `h_state` and `conv_state` to DRAM when:
/// - `pending_rows > 0` AND (`pending_rows >= L_flush` OR remaining ring
///   headroom cannot safely absorb another `GDN_LAZY_MAX_K` commit plus verify
///   window OR `flush_period <= 1` OR `(step_count + slot) % flush_period == 0`).
#[inline]
#[must_use]
pub(super) const fn should_flush_keep_late(
    pending_rows: usize,
    step_count: u32,
    slot: usize,
    flush_period: usize,
    ring_cap: usize,
) -> bool {
    if pending_rows == 0 {
        return false;
    }
    let headroom_low =
        ring_cap <= GDN_LAZY_MAX_K * 2 || pending_rows + GDN_LAZY_MAX_K * 2 > ring_cap;
    if pending_rows >= KEEP_LATE_L_FLUSH || headroom_low || flush_period <= 1 {
        return true;
    }
    (step_count as usize)
        .wrapping_add(slot)
        .is_multiple_of(flush_period)
}

/// Returns whether the raw BF16 pre-conv inputs (`conv_dim * 2` bytes) fit
/// inside the unused Z-tail (`(qkvz_size - conv_dim) * 4` bytes) of every
/// FP32 lazy-retain row.
#[cfg(test)]
#[inline]
#[must_use]
pub(super) const fn keep_late_raw_conv_fits_retain_tail(
    nk: usize,
    kd: usize,
    nv: usize,
    vd: usize,
    qkvz_size: usize,
) -> bool {
    let conv_dim = nk * kd * 2 + nv * vd;
    qkvz_size > conv_dim && (qkvz_size - conv_dim) * 2 >= conv_dim
}

/// Serialize a 32-byte `GdnKeepLateCtrl` struct in exact little-endian ABI layout.
#[inline]
#[must_use]
pub(super) fn keep_late_ctrl_le_bytes(
    ring_base: u64,
    pending_base: u32,
    pending_rows: u32,
    write_base: u32,
    ring_cap: u32,
    flush_now: bool,
) -> [u8; GDN_KEEP_LATE_CTRL_BYTES] {
    let ctrl = ops::GdnKeepLateCtrl {
        ring_base,
        pending_base,
        pending_rows,
        write_base,
        executed_flag: 0,
        ring_cap,
        flush_now: u32::from(flush_now),
    };
    let mut out = [0u8; GDN_KEEP_LATE_CTRL_BYTES];
    ctrl.write_le_bytes(&mut out);
    out
}

/// Per-lane host-side staggered circular-ring `KEEP_LATE` tracking state.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LatePendingSlot {
    /// Ring index (`0..ring_cap`) of the oldest unmaterialized committed row.
    pub(super) pending_base: u16,
    /// Unmaterialized committed rows (`0..=ring_cap`) accumulated across
    /// deferred verify steps since the last DRAM flush. Zero means `h_state`
    /// and `conv_state` already hold the canonical committed state.
    pub(super) pending_rows: u16,
    /// Compact-row capacity (`<= KEEP_LATE_RING_CAP`) of this slot's ring.
    pub(super) ring_cap: u16,
    /// Ring write index staged for the upcoming verify frame.
    pub(super) staged_write_base: u16,
    /// Ring write index written by the most recent `KEEP_LATE` verify frame.
    pub(super) last_verify_write_base: u16,
    /// Monotonic verify-step counter driving staggered periodic flushes.
    pub(super) step_count: u32,
    /// True when the staged verify frame is scheduled to commit replayed
    /// `h_state` and `conv_state` to DRAM.
    pub(super) staged_flush_now: bool,
    /// Current DRAM storage format of this slot's live `h_state`.
    pub(super) h_format: KeepLateHFormat,
    /// `h_state` load format staged for the upcoming verify frame.
    pub(super) staged_load_format: KeepLateHFormat,
    /// `h_state` store format staged for the upcoming verify frame.
    pub(super) staged_store_format: KeepLateHFormat,
    /// True when this lane's most recent verify executed with `KEEP_LATE`.
    pub(super) last_verify_keep_late: bool,
    /// True when `pre_verify_copy_async_dispatch` uploaded `late_ctrl_dev[lane]`
    /// for the upcoming verify frame.
    pub(super) ctrl_staged: bool,
    /// True when `pre_verify_copy_async_dispatch` staged `single_seq_descs_dev[lane]`
    /// for an upcoming single-sequence (`C=1`) `KEEP_LATE` verify.
    pub(super) single_seq_staged: bool,
    /// Live `SsmLayerState::h_state` device address bound to this lane.
    pub(super) h_state: u64,
    /// Live `SsmLayerState::conv_state` device address (for auto-flush).
    pub(super) conv_state: u64,
}

impl LatePendingSlot {
    /// True when live `h_state` or `conv_state` is not yet canonical FP32
    /// (unmaterialized committed rows in the ring or compressed Scaled-FP16/FP8 `h_state`).
    #[inline]
    #[must_use]
    pub(super) const fn needs_fp32_flush(&self) -> bool {
        self.pending_rows > 0 || self.h_format.is_compressed()
    }
}

/// Count `LinearAttention` layers declared in `config`.
#[inline]
pub(super) fn count_ssm_layers(config: &atlas_core::config::ModelConfig) -> usize {
    (0..config.num_hidden_layers)
        .filter(|&i| config.layer_type(i) == atlas_core::config::LayerType::LinearAttention)
        .count()
}

/// Byte offset of `(slot, layer_ordinal)` inside a slot-major contiguous
/// `[lanes][num_ssm_layers][GDN_KEEP_LATE_CTRL_BYTES]` `KEEP_LATE` control slab.
///
/// Slot-major layout keeps all `num_ssm_layers` layers of slot `s` adjacent,
/// so both single-sequence (`C=1`, 1.15 KB) and contiguous multi-sequence
/// cohorts (`C=2..16`, up to 18.4 KB) upload in ONE contiguous H2D copy.
#[inline]
#[must_use]
pub(crate) const fn keep_late_ctrl_slab_offset(
    slot: usize,
    layer_ordinal: usize,
    num_ssm_layers: usize,
) -> Option<usize> {
    if num_ssm_layers == 0 || layer_ordinal >= num_ssm_layers {
        return None;
    }
    let Some(slot_base) = slot.checked_mul(num_ssm_layers) else {
        return None;
    };
    let Some(entry_idx) = slot_base.checked_add(layer_ordinal) else {
        return None;
    };
    entry_idx.checked_mul(GDN_KEEP_LATE_CTRL_BYTES)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct KeepLateCtrlSlabKey {
    gpu_addr: usize,
    lanes: usize,
    num_ssm_layers: usize,
}

struct KeepLateCtrlSlabEntry {
    slab_id: u64,
    key: KeepLateCtrlSlabKey,
    base: DevicePtr,
    next_layer: usize,
    live_layers: usize,
}

#[derive(Default)]
struct KeepLateCtrlSlabRegistry {
    next_slab_id: u64,
    entries: Vec<KeepLateCtrlSlabEntry>,
}

static KEEP_LATE_CTRL_SLABS: std::sync::OnceLock<parking_lot::Mutex<KeepLateCtrlSlabRegistry>> =
    std::sync::OnceLock::new();

/// Claim one layer slice `(layer_base_dev, lane_stride_bytes, slab_id)` from a
/// model-lifetime contiguous `[lanes][num_ssm_layers][32 B]` `KEEP_LATE`
/// control slab so all SSM layers share a single immutable device allocation.
fn alloc_keep_late_ctrl_slab_slice(
    gpu: &dyn GpuBackend,
    lanes: usize,
    num_ssm_layers: usize,
    stream: u64,
) -> anyhow::Result<(DevicePtr, usize, u64)> {
    let lane_stride = num_ssm_layers
        .checked_mul(GDN_KEEP_LATE_CTRL_BYTES)
        .ok_or_else(|| anyhow::anyhow!("GDN keep-late ctrl slab lane stride overflow"))?;
    let slab_bytes = lanes
        .checked_mul(lane_stride)
        .ok_or_else(|| anyhow::anyhow!("GDN keep-late ctrl slab extent overflow"))?;
    let key = KeepLateCtrlSlabKey {
        gpu_addr: std::ptr::from_ref(gpu) as *const () as usize,
        lanes,
        num_ssm_layers,
    };
    let mut reg = KEEP_LATE_CTRL_SLABS
        .get_or_init(|| parking_lot::Mutex::new(KeepLateCtrlSlabRegistry::default()))
        .lock();
    if let Some(entry) = reg
        .entries
        .iter_mut()
        .rev()
        .find(|entry| entry.key == key && entry.next_layer < key.num_ssm_layers)
    {
        let layer_ordinal = entry.next_layer;
        entry.next_layer += 1;
        entry.live_layers += 1;
        let layer_offset = keep_late_ctrl_slab_offset(0, layer_ordinal, num_ssm_layers)
            .ok_or_else(|| anyhow::anyhow!("GDN keep-late ctrl slab layer offset overflow"))?;
        let ptr = entry.base.offset(layer_offset);
        return Ok((ptr, lane_stride, entry.slab_id));
    }
    let base = gpu.alloc(slab_bytes)?;
    if let Err(err) = gpu.memset_async(base, 0, slab_bytes, stream) {
        let _ = gpu.free(base);
        return Err(err);
    }
    reg.next_slab_id = reg.next_slab_id.wrapping_add(1).max(1);
    let slab_id = reg.next_slab_id;
    reg.entries.push(KeepLateCtrlSlabEntry {
        slab_id,
        key,
        base,
        next_layer: 1,
        live_layers: 1,
    });
    Ok((base, lane_stride, slab_id))
}

fn release_keep_late_ctrl_slab_slice(slab_id: u64) {
    if slab_id == 0 {
        return;
    }
    let Some(lock) = KEEP_LATE_CTRL_SLABS.get() else {
        return;
    };
    let mut reg = lock.lock();
    if let Some(pos) = reg.entries.iter().position(|e| e.slab_id == slab_id) {
        let entry = &mut reg.entries[pos];
        entry.next_layer = entry.key.num_ssm_layers;
        entry.live_layers = entry.live_layers.saturating_sub(1);
        if entry.live_layers == 0 {
            reg.entries.swap_remove(pos);
        }
    }
}

/// Layer-owned lazy-commit retain allocation. When `ATLAS_QWEN4_GDN_KEEP_LATE`
/// is active (`bufs_per_lane == 2`), each slot's `lane_stride` allocation
/// ([`gdn_lazy_retain_lane_stride_bytes`]) is sized for one compact circular
/// ring buffer (`KEEP_LATE_RING_CAP = 64` raw-BF16 + FP32-gate rows,
/// `>= buf_bytes` for cold fallback) paired with persistent per-slot
/// `GdnKeepLateCtrl` and single-seq descriptor device slots.
pub(super) struct GdnLazyRetainPool {
    pub(super) base: DevicePtr,
    /// Bytes of one legacy retain buffer (`gdn_lazy_retain_lane_bytes(qkvz, nv)`).
    pub(super) buf_bytes: usize,
    /// `2` when `gdn_keep_late_enabled()` (ring + ctrl/desc mode), else `1`.
    pub(super) bufs_per_lane: usize,
    /// Total per-lane byte stride (`gdn_lazy_retain_lane_stride_bytes(qkvz, nv, keep_late)`).
    pub(super) lane_stride: usize,
    pub(super) lanes: usize,
    /// `lanes * lane_stride`.
    pub(super) allocation_bytes: usize,
    /// Persistent slot-0 `GdnKeepLateCtrl` base pointer for this layer
    /// (sliced from the shared slot-major `[lanes][num_ssm_layers][32 B]` slab
    /// when `late_ctrl_slab_id != 0`, or standalone `[lanes][32 B]` otherwise).
    pub(super) late_ctrl_dev: DevicePtr,
    /// Byte stride between consecutive slots in `late_ctrl_dev`
    /// (`num_ssm_layers * 32` for slab-backed pools, `32` for standalone).
    pub(super) late_ctrl_lane_stride: usize,
    /// Nonzero registry ID when `late_ctrl_dev` belongs to a shared slab.
    pub(super) late_ctrl_slab_id: std::sync::atomic::AtomicU64,
    /// Persistent `[lanes]` 64-byte `GdnFusedMultiSeqDesc` array for C=1 verify
    /// and cold ring flushes.
    pub(super) single_seq_descs_dev: DevicePtr,
    /// Fast-path count of lanes where [`LatePendingSlot::needs_fp32_flush`] is
    /// true (`pending_rows > 0 || h_format != LateHFormat::Fp32`, single atomic load).
    pub(super) pending_count: std::sync::atomic::AtomicU32,
    pub(super) slots: parking_lot::Mutex<Vec<LatePendingSlot>>,
}

impl GdnLazyRetainPool {
    #[inline]
    pub(super) fn lane_base(&self, lane: usize) -> DevicePtr {
        self.base.offset(lane * self.lane_stride)
    }

    #[inline]
    pub(super) fn active_buf_offset(&self, _lane: usize) -> usize {
        0
    }

    #[inline]
    pub(super) fn retain_active_ptr(&self, lane: usize) -> DevicePtr {
        self.base.offset(lane * self.lane_stride)
    }

    #[inline]
    pub(super) fn late_ctrl_ptr(&self, lane: usize) -> DevicePtr {
        self.late_ctrl_dev.offset(lane * self.late_ctrl_lane_stride)
    }

    #[inline]
    pub(super) fn single_seq_desc_ptr(&self, lane: usize) -> DevicePtr {
        self.single_seq_descs_dev
            .offset(lane * ops::GdnFusedMultiSeqDesc::SIZE_BYTES)
    }
}

#[inline]
fn update_slot_flush_needed(pool: &GdnLazyRetainPool, was_needed: bool, now_needed: bool) {
    match (was_needed, now_needed) {
        (false, true) => {
            pool.pending_count
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        (true, false) => {
            let prev = pool
                .pending_count
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            debug_assert!(prev > 0, "KEEP_LATE pending_count underflow");
        }
        _ => {}
    }
}

fn complete_staged_verify_entry(pool: &GdnLazyRetainPool, entry: &mut LatePendingSlot) {
    let was_needed = entry.needs_fp32_flush();
    if entry.staged_flush_now && entry.needs_fp32_flush() {
        if entry.pending_rows > 0 {
            entry.pending_base = entry.staged_write_base;
            entry.pending_rows = 0;
        }
        entry.h_format = entry.staged_store_format;
        if entry.h_format.is_scaled_fp8() {
            log_ssm_gdn_h_fp8_engaged();
        } else if entry.h_format.is_scaled_fp16() {
            log_qwen4_gdn_h_fp16_engaged();
        }
    }
    update_slot_flush_needed(pool, was_needed, entry.needs_fp32_flush());
    entry.last_verify_keep_late = true;
    entry.last_verify_write_base = entry.staged_write_base;
    entry.step_count = entry.step_count.wrapping_add(1);
    entry.ctrl_staged = false;
    entry.single_seq_staged = false;
    entry.staged_flush_now = false;
    entry.staged_load_format = KeepLateHFormat::Fp32;
    entry.staged_store_format = KeepLateHFormat::Fp32;
}

impl Qwen3SsmLayer {
    /// Allocate or return the layer's lazy-commit retain pool.
    ///
    /// Refuses lazy allocation inside an active CUDA graph capture region so
    /// capture never fails mid-layer; callers must pre-warm eagerly before
    /// `begin_capture` (e.g. via `ensure_qwen4_cross_seq_fused_descs_ready`).
    pub(super) fn gdn_lazy_retain_pool(
        &self,
        gpu: &dyn GpuBackend,
        qkvz_size: usize,
        num_v_heads: usize,
    ) -> anyhow::Result<&GdnLazyRetainPool> {
        self.gdn_lazy_retain_pool_with_layers(gpu, qkvz_size, num_v_heads, None)
    }

    pub(super) fn gdn_lazy_retain_pool_with_layers(
        &self,
        gpu: &dyn GpuBackend,
        qkvz_size: usize,
        num_v_heads: usize,
        num_ssm_layers: Option<usize>,
    ) -> anyhow::Result<&GdnLazyRetainPool> {
        let buf_bytes = gdn_lazy_retain_lane_bytes(qkvz_size, num_v_heads)?;
        let pool = match self.gdn_lazy_retain.get() {
            Some(pool) => pool,
            None => {
                let default_stream = gpu.default_stream();
                #[rustfmt::skip]
                let capturing =
                    spark_runtime::gpu_capture_region::active_capture_region().is_some()
                        || gpu.stream_is_capturing(default_stream);
                anyhow::ensure!(
                    !capturing,
                    "GDN lazy retain pool must be pre-allocated outside CUDA graph capture \
                     (pre-warm via ensure_qwen4_cross_seq_fused_descs_ready before capture)"
                );
                let lanes = crate::layers::qwen4_verify_lanes::lanes();
                let keep_late = gdn_keep_late_enabled();
                let bufs_per_lane = if keep_late { 2usize } else { 1usize };
                let lane_stride =
                    gdn_lazy_retain_lane_stride_bytes(qkvz_size, num_v_heads, keep_late)?;
                let total = lanes
                    .checked_mul(lane_stride)
                    .ok_or_else(|| anyhow::anyhow!("GDN lazy retain lane total overflow"))?;
                let base = gpu.alloc(total)?;
                let (late_ctrl_dev, late_ctrl_lane_stride, late_ctrl_slab_id, single_seq_descs_dev) =
                    if bufs_per_lane > 1 {
                        let desc_bytes = lanes
                            .checked_mul(ops::GdnFusedMultiSeqDesc::SIZE_BYTES)
                            .ok_or_else(|| {
                            anyhow::anyhow!("GDN single-seq desc extent overflow")
                        })?;
                        let (ctrl, ctrl_stride, slab_id) = if let Some(n_layers) =
                            num_ssm_layers.filter(|&n| n > 0)
                        {
                            match alloc_keep_late_ctrl_slab_slice(
                                gpu,
                                lanes,
                                n_layers,
                                default_stream,
                            ) {
                                Ok(slice) => slice,
                                Err(err) => {
                                    let _ = gpu.free(base);
                                    return Err(err);
                                }
                            }
                        } else {
                            let ctrl_bytes =
                                lanes.checked_mul(GDN_KEEP_LATE_CTRL_BYTES).ok_or_else(|| {
                                    anyhow::anyhow!("GDN keep-late ctrl extent overflow")
                                })?;
                            let ctrl = match gpu.alloc(ctrl_bytes) {
                                Ok(ptr) => ptr,
                                Err(err) => {
                                    let _ = gpu.free(base);
                                    return Err(err);
                                }
                            };
                            if let Err(err) = gpu.memset_async(ctrl, 0, ctrl_bytes, default_stream)
                            {
                                let _ = gpu.free(ctrl);
                                let _ = gpu.free(base);
                                return Err(err);
                            }
                            (ctrl, GDN_KEEP_LATE_CTRL_BYTES, 0)
                        };
                        let descs = match gpu.alloc(desc_bytes) {
                            Ok(ptr) => ptr,
                            Err(err) => {
                                if slab_id != 0 {
                                    release_keep_late_ctrl_slab_slice(slab_id);
                                } else {
                                    let _ = gpu.free(ctrl);
                                }
                                let _ = gpu.free(base);
                                return Err(err);
                            }
                        };
                        (ctrl, ctrl_stride, slab_id, descs)
                    } else {
                        (
                            DevicePtr::NULL,
                            GDN_KEEP_LATE_CTRL_BYTES,
                            0,
                            DevicePtr::NULL,
                        )
                    };
                let created = GdnLazyRetainPool {
                    base,
                    buf_bytes,
                    bufs_per_lane,
                    lane_stride,
                    lanes,
                    allocation_bytes: total,
                    late_ctrl_dev,
                    late_ctrl_lane_stride,
                    late_ctrl_slab_id: std::sync::atomic::AtomicU64::new(late_ctrl_slab_id),
                    single_seq_descs_dev,
                    pending_count: std::sync::atomic::AtomicU32::new(0),
                    slots: parking_lot::Mutex::new(vec![LatePendingSlot::default(); lanes]),
                };
                if let Err(unused) = self.gdn_lazy_retain.set(created) {
                    let _ = gpu.free(unused.base);
                    let unused_slab = unused
                        .late_ctrl_slab_id
                        .swap(0, std::sync::atomic::Ordering::AcqRel);
                    if unused_slab != 0 {
                        release_keep_late_ctrl_slab_slice(unused_slab);
                    } else if !unused.late_ctrl_dev.is_null() {
                        let _ = gpu.free(unused.late_ctrl_dev);
                    }
                    if !unused.single_seq_descs_dev.is_null() {
                        let _ = gpu.free(unused.single_seq_descs_dev);
                    }
                }
                self.gdn_lazy_retain.get().ok_or_else(|| {
                    anyhow::anyhow!("GDN retain pool unavailable immediately after init")
                })?
            }
        };
        anyhow::ensure!(
            pool.buf_bytes == buf_bytes,
            "GDN lazy retain lane geometry changed ({} -> {buf_bytes} bytes)",
            pool.buf_bytes
        );
        anyhow::ensure!(
            pool.lane_stride.checked_mul(pool.lanes) == Some(pool.allocation_bytes),
            "GDN retain recorded allocation extent differs from its actual lane layout"
        );
        Ok(pool)
    }

    pub(super) fn release_keep_late_ctrl_slab(&self) {
        if let Some(pool) = self.gdn_lazy_retain.get() {
            let slab_id = pool
                .late_ctrl_slab_id
                .swap(0, std::sync::atomic::Ordering::AcqRel);
            release_keep_late_ctrl_slab_slice(slab_id);
        }
    }

    /// Active retain buffer of the calling thread's verify lane, allocating
    /// every configured lane on first use.
    pub(super) fn gdn_lazy_retain_current(
        &self,
        gpu: &dyn GpuBackend,
        qkvz_size: usize,
        num_v_heads: usize,
    ) -> anyhow::Result<DevicePtr> {
        let pool = self.gdn_lazy_retain_pool(gpu, qkvz_size, num_v_heads)?;
        let lane = crate::layers::qwen4_verify_lanes::current_lane();
        anyhow::ensure!(
            lane < pool.lanes,
            "GDN lazy retain has {} lanes; verify lane {lane} has none",
            pool.lanes
        );
        let lane_off = crate::layers::qwen4_verify_lanes::lane_offset(lane, pool.lane_stride)?;
        Ok(pool.base.offset(lane_off + pool.active_buf_offset(lane)))
    }

    #[inline]
    pub(super) fn gdn_keep_late_any_pending(&self) -> bool {
        self.gdn_lazy_retain.get().is_some_and(|pool| {
            pool.pending_count
                .load(std::sync::atomic::Ordering::Acquire)
                > 0
        })
    }

    pub(super) fn gdn_keep_late_supported_inner(&self) -> bool {
        gdn_keep_late_enabled()
            && self.qwen4_gdn_verify_fused_lazyfinal_multi_k.0 != 0
            && !self.gdn_fused_group_done.is_null()
    }

    pub(super) fn gdn_keep_late_armed_inner(&self, slot: usize) -> bool {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return false;
        };
        if pool.bufs_per_lane < 2 || slot >= pool.lanes {
            return false;
        }
        pool.slots
            .lock()
            .get(slot)
            .is_some_and(|entry| entry.last_verify_keep_late)
    }

    pub(super) fn gdn_keep_late_slot_ctrl_ready(&self, slot: usize) -> Option<KeepLateStagedCtrl> {
        let pool = self.gdn_lazy_retain.get()?;
        if pool.bufs_per_lane < 2 || slot >= pool.lanes || pool.late_ctrl_dev.is_null() {
            return None;
        }
        let slots = pool.slots.lock();
        slots
            .get(slot)
            .filter(|entry| entry.ctrl_staged)
            .map(|entry| KeepLateStagedCtrl {
                ctrl_ptr: pool.late_ctrl_ptr(slot),
                load_format: entry.staged_load_format,
                store_format: entry.staged_store_format,
            })
    }

    pub(super) fn gdn_keep_late_mark_verify_ran(&self, slot: usize, used_keep_late: bool) {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return;
        };
        let capturing = spark_runtime::gpu_capture_region::active_capture_region().is_some();
        let mut slots = pool.slots.lock();
        let Some(entry) = slots.get_mut(slot) else {
            return;
        };
        if !used_keep_late {
            entry.last_verify_keep_late = false;
            entry.ctrl_staged = false;
            entry.single_seq_staged = false;
            entry.staged_flush_now = false;
            entry.staged_load_format = KeepLateHFormat::Fp32;
            entry.staged_store_format = KeepLateHFormat::Fp32;
        } else if !capturing && entry.ctrl_staged {
            complete_staged_verify_entry(pool, entry);
        }
    }

    pub(super) fn gdn_keep_late_note_verify_completed_inner(&self, slot: usize) {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return;
        };
        let mut slots = pool.slots.lock();
        let Some(entry) = slots.get_mut(slot) else {
            return;
        };
        if entry.ctrl_staged {
            complete_staged_verify_entry(pool, entry);
        }
    }

    pub(super) fn gdn_keep_late_reconcile_abort_inner(
        &self,
        gpu: &dyn GpuBackend,
        slot: usize,
    ) -> anyhow::Result<()> {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return Ok(());
        };
        if pool.late_ctrl_dev.is_null() {
            return Ok(());
        }
        let flag_dev = {
            let mut slots = pool.slots.lock();
            let Some(entry) = slots.get_mut(slot) else {
                return Ok(());
            };
            // Only read GPU `executed_flag` when a DRAM flush/unpack was
            // actually scheduled (`staged_flush_now && needs_fp32_flush()`).
            // When `staged_flush_now` is false, the GPU never wrote `h_state`
            // or `conv_state`, so `pending_rows` and `h_format` stay intact.
            if !entry.ctrl_staged || !entry.staged_flush_now || !entry.needs_fp32_flush() {
                entry.ctrl_staged = false;
                entry.last_verify_keep_late = false;
                entry.single_seq_staged = false;
                entry.staged_flush_now = false;
                entry.staged_load_format = KeepLateHFormat::Fp32;
                entry.staged_store_format = KeepLateHFormat::Fp32;
                return Ok(());
            }
            // Offset 20..24 of GdnKeepLateCtrl is `executed_flag: u32`.
            pool.late_ctrl_ptr(slot).offset(20)
        };
        let mut flag_bytes = [0u8; 4];
        gpu.copy_d2h(flag_dev, &mut flag_bytes)?;
        let executed = u32::from_le_bytes(flag_bytes) != 0;
        let mut slots = pool.slots.lock();
        if let Some(entry) = slots.get_mut(slot) {
            if executed && entry.staged_flush_now && entry.needs_fp32_flush() {
                let was_needed = entry.needs_fp32_flush();
                if entry.pending_rows > 0 {
                    entry.pending_base = entry.staged_write_base;
                    entry.pending_rows = 0;
                }
                entry.h_format = entry.staged_store_format;
                update_slot_flush_needed(pool, was_needed, entry.needs_fp32_flush());
            }
            entry.ctrl_staged = false;
            entry.last_verify_keep_late = false;
            entry.single_seq_staged = false;
            entry.staged_flush_now = false;
            entry.staged_load_format = KeepLateHFormat::Fp32;
            entry.staged_store_format = KeepLateHFormat::Fp32;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_keep_late_record_inner(
        &self,
        slot: usize,
        num_committed: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        _conv_inter_base: DevicePtr,
        _conv_inter_stride: usize,
        _h_scratch: DevicePtr,
    ) -> bool {
        if num_committed == 0 || num_committed > GDN_LAZY_MAX_K {
            return false;
        }
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return false;
        };
        if pool.bufs_per_lane < 2 || slot >= pool.lanes {
            return false;
        }
        let mut slots = pool.slots.lock();
        let Some(entry) = slots.get_mut(slot) else {
            return false;
        };
        if !entry.last_verify_keep_late || entry.ring_cap == 0 {
            return false;
        }
        let ring_cap = usize::from(entry.ring_cap);
        let cur_pending = usize::from(entry.pending_rows);
        let new_pending = cur_pending + num_committed;
        if new_pending > ring_cap {
            return false;
        }
        if cur_pending == 0 {
            entry.pending_base = entry.last_verify_write_base;
        } else {
            let expected_write_base = (usize::from(entry.pending_base) + cur_pending) % ring_cap;
            if expected_write_base != usize::from(entry.last_verify_write_base) {
                return false;
            }
        }
        let was_needed = entry.needs_fp32_flush();
        entry.pending_rows = new_pending as u16;
        entry.last_verify_keep_late = false;
        entry.ctrl_staged = false;
        entry.single_seq_staged = false;
        entry.staged_flush_now = false;
        entry.staged_load_format = KeepLateHFormat::Fp32;
        entry.staged_store_format = KeepLateHFormat::Fp32;
        entry.h_state = h_state.0;
        entry.conv_state = conv_state.0;
        update_slot_flush_needed(pool, was_needed, entry.needs_fp32_flush());
        true
    }

    /// Disarm staged/verify flags for `slot` without dropping unmaterialized
    /// multi-step ring rows or altering `h_format` (`pending_rows` and
    /// `h_format` stay intact on deferred full-reject steps because live
    /// `h_state`/`conv_state` remain pinned at `P_base`).
    pub(super) fn gdn_keep_late_clear_pending_inner(&self, slot: usize) {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return;
        };
        let mut slots = pool.slots.lock();
        if let Some(entry) = slots.get_mut(slot) {
            entry.last_verify_keep_late = false;
            entry.ctrl_staged = false;
            entry.single_seq_staged = false;
            entry.staged_flush_now = false;
            entry.staged_load_format = KeepLateHFormat::Fp32;
            entry.staged_store_format = KeepLateHFormat::Fp32;
        }
    }

    pub(super) fn gdn_keep_late_reset_slot_inner(&self, slot: usize) {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return;
        };
        let mut slots = pool.slots.lock();
        if let Some(entry) = slots.get_mut(slot) {
            let was_needed = entry.needs_fp32_flush();
            *entry = LatePendingSlot::default();
            update_slot_flush_needed(pool, was_needed, false);
        }
    }

    pub(super) fn gdn_keep_late_has_pending_inner(&self, slot: usize) -> bool {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return false;
        };
        if pool
            .pending_count
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return false;
        }
        pool.slots
            .lock()
            .get(slot)
            .is_some_and(LatePendingSlot::needs_fp32_flush)
    }

    pub(super) fn gdn_keep_late_has_scaled_fp16_inner(&self, slot: usize) -> bool {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return false;
        };
        if pool
            .pending_count
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return false;
        }
        pool.slots
            .lock()
            .get(slot)
            .is_some_and(|entry| entry.h_format.is_compressed())
    }

    pub(super) fn single_seq_staged_is_fp32(&self, slot: usize) -> bool {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return false;
        };
        pool.slots.lock().get(slot).is_some_and(|entry| {
            entry.single_seq_staged
                && entry.ctrl_staged
                && entry.staged_load_format == KeepLateHFormat::Fp32
                && entry.staged_store_format == KeepLateHFormat::Fp32
        })
    }

    pub(super) fn gdn_split_done_ptr(&self, counter_slot: usize, nv: usize) -> Option<DevicePtr> {
        if self.gdn_fused_split_done.is_null() || nv == 0 || nv > self.gdn_fused_split_done_stride {
            return None;
        }
        let byte_offset = counter_slot
            .checked_mul(self.gdn_fused_split_done_stride)?
            .checked_mul(std::mem::size_of::<u32>())?;
        Some(self.gdn_fused_split_done.offset(byte_offset))
    }

    pub(super) fn gdn_split_v_scratch_ptrs(
        ctx: &ForwardContext,
        total_rows: usize,
        nv: usize,
    ) -> Option<(DevicePtr, DevicePtr)> {
        let layout = ops::gdn_split_v_scratch_layout(total_rows, nv)?;
        if layout.total_bytes > ctx.buffers.sizes().ssm_conv_out_f32 {
            return None;
        }
        let base = ctx.buffers.ssm_conv_out_f32();
        if base.is_null() {
            return None;
        }
        Some((base, base.offset(layout.ss_offset_bytes)))
    }

    /// Cold-flush all `pending_rows` from `slot`'s compact circular ring and/or
    /// unpack in-place Scaled-FP16 `h_state` back into canonical FP32 `h_state`
    /// and `conv_state` (and optionally `conv_ckpt`) using
    /// `qwen4_gdn_verify_fused_lazyfinal_multi_seq` with `num_tokens = 0,
    /// flush_now = 1`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_keep_late_flush_slot_inner(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        conv_ckpt: Option<DevicePtr>,
        _conv_inter_base: DevicePtr,
        _conv_inter_stride: usize,
        _h_scratch: DevicePtr,
        stream: u64,
    ) -> anyhow::Result<()> {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return Ok(());
        };
        if pool
            .pending_count
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return Ok(());
        }
        let pending = {
            let slots = pool.slots.lock();
            let Some(entry) = slots.get(slot) else {
                return Ok(());
            };
            if !entry.needs_fp32_flush() {
                return Ok(());
            }
            (
                usize::from(entry.pending_base),
                usize::from(entry.pending_rows),
                usize::from(entry.ring_cap),
                entry.h_format,
            )
        };
        let (pending_base, pending_rows, ring_cap, h_format) = pending;
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        anyhow::ensure!(
            self.qwen4_gdn_verify_fused_lazyfinal_multi_k.0 != 0
                && !self.gdn_fused_group_done.is_null(),
            "KEEP_LATE ring flush requires qwen4_gdn_verify_fused_lazyfinal_multi kernel"
        );
        anyhow::ensure!(
            slot < pool.lanes
                && slot < max_lanes
                && !pool.late_ctrl_dev.is_null()
                && !pool.single_seq_descs_dev.is_null()
                && !h_state.is_null()
                && !conv_state.is_null()
                && (pending_rows == 0 || (ring_cap > 0 && pending_rows <= ring_cap)),
            "KEEP_LATE ring flush invalid slot or ring state \
             (slot={slot}, pending_rows={pending_rows}, ring_cap={ring_cap}, h_format={h_format:?})"
        );

        let qkvz = config.ssm_qkvz_size();
        let nk = config.linear_num_key_heads;
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let eps = config.rms_norm_eps as f32;
        let next_base = if ring_cap > 0 {
            (pending_base + pending_rows) % ring_cap
        } else {
            0
        };
        let ring_base = pool.lane_base(slot);
        let ctrl_ptr = pool.late_ctrl_ptr(slot);
        let desc_ptr = pool.single_seq_desc_ptr(slot);
        let counter_slot = if stream == gpu.default_stream() {
            slot
        } else {
            max_lanes + slot
        };
        let group_done = self.gdn_fused_group_done.offset(counter_slot * 32 * 4);

        let ctrl_bytes = keep_late_ctrl_le_bytes(
            ring_base.0,
            pending_base as u32,
            pending_rows as u32,
            next_base as u32,
            ring_cap as u32,
            true,
        );
        // Cold flush ALWAYS stores canonical FP32 (`store_format = Fp32`) so
        // every FP32 consumer (`C=1`/`C=2`, checkpoints, snapshots, compaction)
        // observes canonical FP32 `h_state`.
        let flush_ctrl = KeepLateStagedCtrl {
            ctrl_ptr,
            load_format: h_format,
            store_format: KeepLateHFormat::Fp32,
        };
        let desc = ops::GdnFusedMultiSeqDesc {
            conv_state: conv_state.0,
            h_state: h_state.0,
            retain_qkv: ring_base.0,
            conv_inter: 0,
            h_state_inter: flush_ctrl.ctrl_ptr.0,
            group_done: group_done.0,
            row_base: 0,
            num_tokens: 0,
            flags: flush_ctrl.desc_flags(),
            _pad: 0,
        };
        let mut desc_bytes = [0u8; ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
        desc.write_le_bytes(&mut desc_bytes);

        let copies = [
            HostToDeviceCopy::new(&ctrl_bytes, ctrl_ptr),
            HostToDeviceCopy::new(&desc_bytes, desc_ptr),
        ];
        gpu.copy_h2d_group_on_stream(&copies, stream)?;
        ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
            gpu,
            self.qwen4_gdn_verify_fused_lazyfinal_multi_k,
            desc_ptr,
            DevicePtr::NULL,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            DevicePtr::NULL,
            &self.ssm.norm,
            DevicePtr::NULL,
            1,
            nk as u32,
            nv as u32,
            1e-6,
            eps,
            qkvz as u32,
            (nv * 2) as u32,
            (nv * vd) as u32,
            qkvz as u32,
            0,
            0,
            stream,
        )?;
        if let Some(ckpt) = conv_ckpt {
            gpu.copy_d2d_async(conv_state, ckpt, self.conv_state_bytes, stream)?;
        }
        let mut slots = pool.slots.lock();
        if let Some(entry) = slots.get_mut(slot) {
            let was_needed = entry.needs_fp32_flush();
            if entry.pending_rows > 0 {
                entry.pending_base = next_base as u16;
                entry.pending_rows = 0;
            }
            entry.h_format = KeepLateHFormat::Fp32;
            update_slot_flush_needed(pool, was_needed, entry.needs_fp32_flush());
            entry.last_verify_keep_late = false;
            entry.ctrl_staged = false;
            entry.single_seq_staged = false;
            entry.staged_flush_now = false;
            entry.staged_load_format = KeepLateHFormat::Fp32;
            entry.staged_store_format = KeepLateHFormat::Fp32;
        }
        Ok(())
    }

    pub(super) fn flush_keep_late_for_layer_state(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        state: &dyn LayerState,
        graph_capture: bool,
        stream: u64,
    ) -> anyhow::Result<()> {
        if !self.gdn_keep_late_any_pending() {
            return Ok(());
        }
        let Some(ssm) = state.as_any().downcast_ref::<SsmLayerState>() else {
            return Ok(());
        };
        self.flush_keep_late_for_h_state(gpu, config, ssm.h_state, graph_capture, stream)
    }

    pub(super) fn flush_keep_late_for_h_state(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        h_state: DevicePtr,
        graph_capture: bool,
        stream: u64,
    ) -> anyhow::Result<()> {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return Ok(());
        };
        if pool
            .pending_count
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
            || h_state.is_null()
        {
            return Ok(());
        }
        let matched = {
            let slots = pool.slots.lock();
            slots.iter().enumerate().find_map(|(lane, entry)| {
                (entry.needs_fp32_flush() && entry.h_state == h_state.0).then_some((
                    lane,
                    DevicePtr(entry.h_state),
                    DevicePtr(entry.conv_state),
                ))
            })
        };
        if let Some((lane, h, conv)) = matched {
            anyhow::ensure!(
                !graph_capture
                    && spark_runtime::gpu_capture_region::active_capture_region().is_none(),
                "cannot auto-flush pending KEEP_LATE DeltaNet state inside CUDA graph capture (lane={lane})"
            );
            self.gdn_keep_late_flush_slot_inner(
                gpu,
                config,
                lane,
                h,
                conv,
                None,
                DevicePtr::NULL,
                0,
                DevicePtr::NULL,
                stream,
            )?;
        }
        Ok(())
    }

    /// Guard single-lane (`lanes == 1`) multi-slot setups: before a different
    /// sequence writes into `lane`'s retain buffer, flush any pending rows or
    /// Scaled-FP16 `h_state` owned by the previous occupant of `lane`.
    pub(super) fn flush_keep_late_for_lane_if_different_h(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        lane: usize,
        h_state: DevicePtr,
        graph_capture: bool,
        stream: u64,
    ) -> anyhow::Result<()> {
        let Some(pool) = self.gdn_lazy_retain.get() else {
            return Ok(());
        };
        if pool
            .pending_count
            .load(std::sync::atomic::Ordering::Acquire)
            == 0
        {
            return Ok(());
        }
        let occupant = {
            let slots = pool.slots.lock();
            slots.get(lane).and_then(|entry| {
                (entry.needs_fp32_flush() && entry.h_state != h_state.0)
                    .then_some((DevicePtr(entry.h_state), DevicePtr(entry.conv_state)))
            })
        };
        if let Some((h, conv)) = occupant {
            anyhow::ensure!(
                !graph_capture
                    && spark_runtime::gpu_capture_region::active_capture_region().is_none(),
                "cannot flush colliding KEEP_LATE lane {lane} inside CUDA graph capture"
            );
            self.gdn_keep_late_flush_slot_inner(
                gpu,
                config,
                lane,
                h,
                conv,
                None,
                DevicePtr::NULL,
                0,
                DevicePtr::NULL,
                stream,
            )?;
        }
        Ok(())
    }

    pub(super) fn stage_qwen4_keep_late_ctrl_bytes_inner(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        h_state: DevicePtr,
    ) -> anyhow::Result<Option<(DevicePtr, [u8; GDN_KEEP_LATE_CTRL_BYTES])>> {
        if !self.gdn_keep_late_supported_inner() || !config.is_qwen4_exp() {
            return Ok(None);
        }
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let qkvz_size = config.ssm_qkvz_size();
        if kd != 128
            || vd != 128
            || config.linear_conv_kernel_dim != 4
            || nk == 0
            || nk > 32
            || nv < nk
            || !nv.is_multiple_of(nk)
            || config.output_gate_type != "sigmoid"
            || !keep_late_ring_fits(nk, kd, nv, vd, qkvz_size)
        {
            return Ok(None);
        }
        let pool = self.gdn_lazy_retain_pool_with_layers(
            gpu,
            qkvz_size,
            nv,
            Some(count_ssm_layers(config)),
        )?;
        if pool.bufs_per_lane < 2 || slot >= pool.lanes || pool.late_ctrl_dev.is_null() {
            return Ok(None);
        }
        let ring_cap = keep_late_ring_capacity(pool.lane_stride, nk, kd, nv, vd);
        if ring_cap < KEEP_LATE_L_MAX + GDN_LAZY_MAX_K {
            return Ok(None);
        }
        let mut slots = pool.slots.lock();
        let entry = &mut slots[slot];
        if entry.needs_fp32_flush() && entry.h_state != h_state.0 {
            return Ok(None);
        }
        let pending_base = usize::from(entry.pending_base) % ring_cap;
        let pending_rows = usize::from(entry.pending_rows);
        if pending_rows + GDN_LAZY_MAX_K > ring_cap {
            return Ok(None);
        }
        let cohort_seqs = current_keep_late_cohort_seqs();
        let base_flush_now = should_flush_keep_late(
            pending_rows,
            entry.step_count,
            slot,
            gdn_keep_late_flush_period_for_cohort(cohort_seqs, ring_cap),
            ring_cap,
        );
        let h_plan = plan_keep_late_h_format_ex(
            entry.h_format,
            gdn_h_fp16_enabled(),
            gdn_h_fp8_enabled(),
            cohort_seqs,
            base_flush_now,
        );
        let write_base = (pending_base + pending_rows) % ring_cap;
        let bytes = keep_late_ctrl_le_bytes(
            pool.lane_base(slot).0,
            pending_base as u32,
            pending_rows as u32,
            write_base as u32,
            ring_cap as u32,
            h_plan.flush_now,
        );
        entry.ring_cap = ring_cap as u16;
        entry.pending_base = pending_base as u16;
        entry.staged_write_base = write_base as u16;
        entry.staged_flush_now = h_plan.flush_now;
        entry.staged_load_format = h_plan.load_format;
        entry.staged_store_format = h_plan.store_format;
        entry.h_state = h_state.0;
        entry.ctrl_staged = true;
        Ok(Some((pool.late_ctrl_ptr(slot), bytes)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_qwen4_single_seq_fused_desc_bytes_inner(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        slot: usize,
        k: usize,
        h_state: DevicePtr,
        conv_state: DevicePtr,
        _conv_inter_base: DevicePtr,
        stream: u64,
    ) -> anyhow::Result<Option<(DevicePtr, [u8; ops::GdnFusedMultiSeqDesc::SIZE_BYTES])>> {
        if !self.gdn_keep_late_supported_inner()
            || !config.is_qwen4_exp()
            || !(2..=GDN_LAZY_MAX_K).contains(&k)
            || !self.gdn_seq_lazy_engaged_inner(k)
        {
            return Ok(None);
        }
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let qkvz_size = config.ssm_qkvz_size();
        if kd != 128
            || vd != 128
            || config.linear_conv_kernel_dim != 4
            || nk == 0
            || nk > 32
            || nv < nk
            || !nv.is_multiple_of(nk)
            || config.output_gate_type != "sigmoid"
            || !keep_late_ring_fits(nk, kd, nv, vd, qkvz_size)
        {
            return Ok(None);
        }
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        let pool = self.gdn_lazy_retain_pool_with_layers(
            gpu,
            qkvz_size,
            nv,
            Some(count_ssm_layers(config)),
        )?;
        if pool.bufs_per_lane < 2
            || slot >= pool.lanes
            || slot >= max_lanes
            || pool.late_ctrl_dev.is_null()
            || pool.single_seq_descs_dev.is_null()
        {
            return Ok(None);
        }
        let counter_slot = if stream == gpu.default_stream() {
            slot
        } else {
            max_lanes + slot
        };
        let group_done = self.gdn_fused_group_done.offset(counter_slot * 32 * 4);
        let (load_format, store_format) = {
            let mut slots = pool.slots.lock();
            let entry = &mut slots[slot];
            entry.h_state = h_state.0;
            entry.conv_state = conv_state.0;
            entry.single_seq_staged = true;
            if entry.ctrl_staged {
                (entry.staged_load_format, entry.staged_store_format)
            } else {
                (entry.h_format, KeepLateHFormat::Fp32)
            }
        };
        let staged_ctrl = KeepLateStagedCtrl {
            ctrl_ptr: pool.late_ctrl_ptr(slot),
            load_format,
            store_format,
        };
        let desc = ops::GdnFusedMultiSeqDesc {
            conv_state: conv_state.0,
            h_state: h_state.0,
            retain_qkv: pool.lane_base(slot).0,
            // Zero dead `conv_inter` writes under KEEP_LATE circular ring.
            conv_inter: 0,
            h_state_inter: staged_ctrl.ctrl_ptr.0,
            group_done: group_done.0,
            row_base: 0,
            num_tokens: k as u32,
            flags: staged_ctrl.desc_flags(),
            _pad: 0,
        };
        let mut bytes = [0u8; ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
        desc.write_le_bytes(&mut bytes);
        Ok(Some((pool.single_seq_desc_ptr(slot), bytes)))
    }

    pub(super) fn find_single_seq_staged_desc_for_h_state(
        &self,
        h_state: DevicePtr,
    ) -> Option<(usize, DevicePtr)> {
        let pool = self.gdn_lazy_retain.get()?;
        if pool.bufs_per_lane < 2 || pool.single_seq_descs_dev.is_null() || h_state.is_null() {
            return None;
        }
        let slots = pool.slots.lock();
        slots.iter().enumerate().find_map(|(lane, entry)| {
            (entry.single_seq_staged && entry.ctrl_staged && entry.h_state == h_state.0)
                .then_some((lane, pool.single_seq_desc_ptr(lane)))
        })
    }

    pub(super) fn has_single_seq_staged_for_state(&self, state: &dyn LayerState) -> bool {
        state
            .as_any()
            .downcast_ref::<SsmLayerState>()
            .is_some_and(|ssm| {
                self.find_single_seq_staged_desc_for_h_state(ssm.h_state)
                    .is_some()
            })
    }
}

impl Qwen3SsmLayer {
    /// Pure mirror of the lazy ExactSequence dispatch decision — MUST equal
    /// what `trait_decode_batched.rs` does for the same `num_tokens`, because
    /// the async commit uses this to decide replay-vs-copy. Everything here is
    /// process-constant env + kernel handles + num_tokens (graph-safe).
    pub(super) fn gdn_seq_lazy_engaged_inner(&self, num_tokens: usize) -> bool {
        let is_qwen4 = self.qwen4_attn_hyper.is_some() && self.qwen4_mlp_hyper.is_some();
        let lazy_requested = (is_qwen4 && replay_ssm_enabled())
            || std::env::var("ATLAS_SSM_GDN_LAZY").ok().as_deref() == Some("1");
        if !lazy_requested || !(2..=GDN_LAZY_MAX_K).contains(&num_tokens) {
            return false;
        }
        let qwen4_short_rows = is_qwen4
            && (2..4).contains(&num_tokens)
            && crate::layers::qwen4_hybrid_rows(num_tokens)
            && std::env::var("ATLAS_QWEN4_K5_BATCH_SSM").ok().as_deref() == Some("1")
            && std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1")
            && (replay_ssm_enabled()
                || crate::layers::qwen4_verify_fast_host_selected()
                || std::env::var("ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR")
                    .ok()
                    .as_deref()
                    == Some("1"));
        let route_tokens = if qwen4_short_rows { 4 } else { num_tokens };
        std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() != Some("1")
            && self.gdn_f32_sequence_lazyfinal_k.0 != 0
            && self.gdn_f32_sequence_nosnap_k.0 != 0
            && exact_flat_route::exact_flat_ssm_route(
                route_tokens,
                false, // tree handled separately: dispatch via ctx, commit via was_tree_mode
                self.ba_gates_batchn_exact_k.0 != 0,
                self.conv1d_l2norm_f32_sequence_k.0 != 0,
                self.gdn_f32_sequence_k.0 != 0,
                self.gated_rms_norm_f32_multi_seq_k.0 != 0,
            ) == exact_flat_route::ExactFlatSsmRoute::ExactSequence
    }
}

pub fn replay_ssm_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_QWEN4_REPLAY_SSM").ok().as_deref() != Some("0"))
}

/// `ATLAS_SSM_GDN_LAZY=1` (or `ATLAS_QWEN4_REPLAY_SSM != 0` on Qwen4) —
/// lazy-commit GDN verify (skip per-token H snapshots; commit reconstructs via
/// FP32 replay). Bit-exact by construction; dispatch + commit must agree, so
/// both call [`Qwen3SsmLayer::gdn_seq_lazy_engaged`].
pub fn gdn_seq_lazy_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        replay_ssm_enabled() || std::env::var("ATLAS_SSM_GDN_LAZY").ok().as_deref() == Some("1")
    })
}

/// `ATLAS_SSM_H_FP16=1` — store the GDN decode/verify h-state as FP16.
///
/// Storage-only narrowing — arithmetic, accumulation order, and gate clamps in
/// the FP16 twin kernels match their FP32 parents; only the h round-trip
/// rounding differs. OFF by default.
pub fn ssm_h_fp16_enabled() -> bool {
    std::env::var("ATLAS_SSM_H_FP16").ok().as_deref() == Some("1")
}

pub fn qwen4_gdn_verify_fused_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_GDN_VERIFY_FUSED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Zero-replay `KEEP_LATE` DeltaNet commit gate (`ATLAS_QWEN4_GDN_KEEP_LATE`,
/// default ON unless `"0"`, disabled when `ATLAS_QWEN4_BATCHED_K5_DEDUP=1` or
/// `ATLAS_SSM_GDN_NOSNAP=1` or fused GDN verify / lazy commit are off).
pub fn gdn_keep_late_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var(GDN_KEEP_LATE_ENV).ok().as_deref() != Some("0")
            && qwen4_gdn_verify_fused_enabled()
            && gdn_seq_lazy_enabled()
            && std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() != Some("1")
            && std::env::var("ATLAS_QWEN4_BATCHED_K5_DEDUP")
                .ok()
                .as_deref()
                != Some("1")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_ssm_state_allocation_sizes() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let nv = config.linear_num_value_heads; // 32
        let vd = config.linear_value_head_dim; // 128
        let nk = config.linear_num_key_heads; // 16
        let kd = config.linear_key_head_dim; // 128
        let d_conv = config.linear_conv_kernel_dim; // 4

        let h_bytes = nv * vd * kd * 4;
        assert_eq!(h_bytes, 32 * 128 * 128 * 4); // 2 MB

        // conv_dim = 2*key_dim + value_dim = 2*2048 + 4096 = 8192
        let conv_dim = nk * kd * 2 + nv * vd;
        let conv_bytes = conv_dim * d_conv * 4;
        assert_eq!(conv_bytes, 8192 * 4 * 4); // 128 KB

        // Verify allocations
        let gpu = MockGpuBackend::new();
        let h_state = gpu.alloc(h_bytes).unwrap();
        let conv_state = gpu.alloc(conv_bytes).unwrap();
        assert!(!h_state.is_null());
        assert!(!conv_state.is_null());
    }
}

impl Qwen3SsmLayer {
    /// `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1` (loader): detach the BF16 QKVZ
    /// concat once its NVFP4 copy exists. Every reader of `in_proj_qkvz`
    /// is a fallback taken only when `qkvz_nvfp4` is `None` (or under
    /// `ATLAS_GDN_BF16_WEIGHTS=1`, which the loader excludes). Returns the
    /// pointer for the caller to free; `None` when nothing was detached.
    ///
    /// Self-protecting: refuses when `qkvz_dense_decode_only` is `Some`
    /// (`ATLAS_QWEN4_GDN_BF16=1` engaged), because that field is a
    /// non-owning alias of this same `DenseWeight` pointer (`DenseWeight`
    /// is `Copy`, not a handle with its own lifetime) — detaching and
    /// freeing it here while decode still reads it through
    /// `qkvz_dense_decode_only` would be a use-after-free.
    pub(crate) fn take_bf16_qkvz(&mut self) -> Option<DevicePtr> {
        if self.qkvz_nvfp4.is_none()
            || self.ssm.in_proj_qkvz.weight.is_null()
            || self.qkvz_dense_decode_only.is_some()
        {
            return None;
        }
        Some(std::mem::replace(
            &mut self.ssm.in_proj_qkvz.weight,
            DevicePtr::NULL,
        ))
    }

    /// The BF16 QKVZ concat for the fallback routes that run only without an
    /// NVFP4 QKVZ copy. Once a loader released it at load
    /// (`ATLAS_QWEN4_RELEASE_BF16_QKVZ` / `ATLAS_MEMLEAN_SSM_BF16_QKVZ`), a
    /// route that reaches it fails closed instead of reading freed memory.
    fn bf16_qkvz(&self) -> anyhow::Result<&DenseWeight> {
        anyhow::ensure!(
            !self.ssm.in_proj_qkvz.weight.is_null(),
            "SSM BF16 QKVZ fallback weight was released at load (ATLAS_QWEN4_RELEASE_BF16_QKVZ / ATLAS_MEMLEAN_SSM_BF16_QKVZ)"
        );
        Ok(&self.ssm.in_proj_qkvz)
    }
}

#[cfg(test)]
mod gdn_lazy_retain_lane_tests {
    use super::{
        DevicePtr, GDN_FUSED_FLAG_KEEP_LATE, GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR,
        GDN_FUSED_FLAG_LOAD_SCALED_FP16, GDN_FUSED_FLAG_STORE_SCALED_FP16, GDN_H_FP16_ENV,
        GDN_H_FP16_MIN_COHORT, GDN_KEEP_LATE_CTRL_BYTES, GDN_LAZY_MAX_K, GdnLazyRetainPool,
        KEEP_LATE_DEFAULT_FLUSH_PERIOD, KEEP_LATE_HIGH_SM_DEFAULT_FLUSH_PERIOD, KEEP_LATE_L_FLUSH,
        KEEP_LATE_L_MAX, KEEP_LATE_RING_CAP, KeepLateCohortSeqsGuard, KeepLateHFormat,
        KeepLateHFormatPlan, KeepLateStagedCtrl, LatePendingSlot, complete_staged_verify_entry,
        current_keep_late_cohort_seqs, gdn_lazy_retain_lane_bytes,
        gdn_lazy_retain_lane_stride_bytes, keep_late_ctrl_le_bytes, keep_late_flush_period_for_sm,
        keep_late_raw_conv_fits_retain_tail, keep_late_ring_capacity, keep_late_ring_fits,
        keep_late_ring_row_bytes, ops, parse_gdn_h_fp16_enabled, parse_keep_late_flush_period,
        plan_keep_late_h_format, resolve_keep_late_flush_period,
        resolve_keep_late_flush_period_for_sm, should_flush_keep_late, update_slot_flush_needed,
    };

    #[test]
    fn lane_bytes_match_the_historical_single_retain_extent() {
        // Flash-Next: qkvz 16384, 48 value heads.
        let qkvz = 16_384;
        let nv = 48;
        let historical = GDN_LAZY_MAX_K * qkvz * 4 + GDN_LAZY_MAX_K * nv * 2 * 4;
        assert_eq!(gdn_lazy_retain_lane_bytes(qkvz, nv).unwrap(), historical);
        assert_eq!(historical, 1_054_720);
        assert_eq!(
            historical % 16,
            0,
            "lane-major retain lanes stay 16-byte aligned"
        );
    }

    #[test]
    fn lane_bytes_overflow_fails_closed() {
        assert!(gdn_lazy_retain_lane_bytes(usize::MAX, 1).is_err());
        assert!(gdn_lazy_retain_lane_bytes(1, usize::MAX).is_err());
    }

    #[test]
    fn keep_late_raw_conv_fits_qwen4_tp1_and_tp2_retain_tail() {
        // TP1 Flash-Next: conv_dim = 10240, qkvz_size = 16384 -> Z-tail = 6144 f32 = 12288 bf16.
        assert!(keep_late_raw_conv_fits_retain_tail(
            16, 128, 48, 128, 16_384
        ));
        // TP2 Flash-Next: conv_dim = 5120, qkvz_size = 8192 -> Z-tail = 3072 f32 = 6144 bf16.
        assert!(keep_late_raw_conv_fits_retain_tail(8, 128, 24, 128, 8_192));
        // Tight/no Z-tail rejects cleanly.
        assert!(!keep_late_raw_conv_fits_retain_tail(
            16, 128, 48, 128, 10_240
        ));
        assert!(!keep_late_raw_conv_fits_retain_tail(
            16, 128, 48, 128, 12_000
        ));
    }

    #[test]
    fn keep_late_compact_ring_geometry_sizes_tight_64_row_stride_and_saves_c64_vram() {
        assert_eq!(KEEP_LATE_RING_CAP, 64);
        assert_eq!(KEEP_LATE_RING_CAP, KEEP_LATE_L_MAX + GDN_LAZY_MAX_K);

        // TP1 Flash-Next: 20,864 B/row x 64 rows = 1,335,296 B (>= 1,054,720 B legacy FP32 buf).
        // Across 36 SSM layers x 64 lanes = 3,076,521,984 B (2.865 GiB vs old 4.526 GiB: -1.66 GiB).
        let tp1_row = keep_late_ring_row_bytes(16, 128, 48, 128).unwrap();
        assert_eq!(tp1_row, 20_864);
        assert_eq!(tp1_row % 16, 0);
        let tp1_legacy = gdn_lazy_retain_lane_bytes(16_384, 48).unwrap();
        assert_eq!(
            gdn_lazy_retain_lane_stride_bytes(16_384, 48, false).unwrap(),
            tp1_legacy
        );
        let tp1_stride = gdn_lazy_retain_lane_stride_bytes(16_384, 48, true).unwrap();
        assert_eq!(tp1_stride, 64 * 20_864);
        assert_eq!(tp1_stride, 1_335_296);
        assert!(tp1_stride >= tp1_legacy);
        assert_eq!(tp1_stride % 16, 0);
        assert_eq!(
            keep_late_ring_capacity(tp1_stride, 16, 128, 48, 128),
            KEEP_LATE_RING_CAP
        );
        assert!(keep_late_ring_fits(16, 128, 48, 128, 16_384));

        // TP2 Flash-Next: 10,432 B/row x 64 rows = 667,648 B (>= 527,360 B legacy FP32 buf).
        let tp2_row = keep_late_ring_row_bytes(8, 128, 24, 128).unwrap();
        assert_eq!(tp2_row, 10_432);
        assert_eq!(tp2_row % 16, 0);
        let tp2_stride = gdn_lazy_retain_lane_stride_bytes(8_192, 24, true).unwrap();
        assert_eq!(tp2_stride, 64 * 10_432);
        assert_eq!(tp2_stride, 667_648);
        assert_eq!(
            keep_late_ring_capacity(tp2_stride, 8, 128, 24, 128),
            KEEP_LATE_RING_CAP
        );
        assert!(keep_late_ring_fits(8, 128, 24, 128, 8_192));

        // Too-narrow retain allocation without headroom rejects cleanly.
        assert!(!keep_late_ring_fits(16, 128, 48, 128, 10_240));
    }

    #[test]
    fn keep_late_flush_period_parser_and_stagger_schedule() {
        assert_eq!(
            parse_keep_late_flush_period(None),
            KEEP_LATE_DEFAULT_FLUSH_PERIOD
        );
        assert_eq!(
            parse_keep_late_flush_period(Some("0")),
            KEEP_LATE_DEFAULT_FLUSH_PERIOD
        );
        assert_eq!(
            parse_keep_late_flush_period(Some("bogus")),
            KEEP_LATE_DEFAULT_FLUSH_PERIOD
        );
        assert_eq!(parse_keep_late_flush_period(Some("1")), 1);
        assert_eq!(parse_keep_late_flush_period(Some("8")), 8);
        assert_eq!(parse_keep_late_flush_period(Some("200")), 64);

        // Auto cohort sizing: C=1 stays M=1 (0-ULP FP32), all C>=2 cohorts use
        // M=2 (avoids multi-step pending-row replay regression); explicit env
        // overrides always win.
        assert_eq!(KEEP_LATE_DEFAULT_FLUSH_PERIOD, 2);
        assert_eq!(resolve_keep_late_flush_period(None, 0), 1);
        assert_eq!(resolve_keep_late_flush_period(None, 1), 1);
        for c in [2usize, 3, 4, 7, 8, 12, 16, 32] {
            assert_eq!(
                resolve_keep_late_flush_period(None, c),
                KEEP_LATE_DEFAULT_FLUSH_PERIOD,
                "cohort={c} must stay on M=2"
            );
        }
        assert_eq!(resolve_keep_late_flush_period(Some(4), 1), 4);
        assert_eq!(resolve_keep_late_flush_period(Some(6), 16), 6);
        assert_eq!(resolve_keep_late_flush_period(Some(2), 16), 2);
        assert_eq!(resolve_keep_late_flush_period(Some(1), 16), 1);
        {
            assert_eq!(current_keep_late_cohort_seqs(), 1);
            let _outer = KeepLateCohortSeqsGuard::enter(8);
            assert_eq!(current_keep_late_cohort_seqs(), 8);
            {
                let _inner = KeepLateCohortSeqsGuard::enter(2);
                assert_eq!(current_keep_late_cohort_seqs(), 2);
            }
            assert_eq!(current_keep_late_cohort_seqs(), 8);
        }
        assert_eq!(current_keep_late_cohort_seqs(), 1);

        // Zero pending rows never flushes.
        assert!(!should_flush_keep_late(0, 0, 0, 8, KEEP_LATE_RING_CAP));
        // Across 16 slots at default M=2, exactly 8 slots flush per step (16 per 2-step cycle).
        let mut m2_cycle_flushes = 0usize;
        for step in 0u32..2 {
            let flushers = (0usize..16)
                .filter(|&slot| {
                    should_flush_keep_late(
                        5,
                        step,
                        slot,
                        KEEP_LATE_DEFAULT_FLUSH_PERIOD,
                        KEEP_LATE_RING_CAP,
                    )
                })
                .count();
            assert_eq!(
                flushers, 8,
                "step={step} at M=2 should flush 8 of 16 slots, got {flushers}"
            );
            m2_cycle_flushes += flushers;
        }
        assert_eq!(m2_cycle_flushes, 16);
        // Across 16 slots at explicit override M=6, 2..=3 slots flush per step (16 per 6-step cycle).
        let mut m6_cycle_flushes = 0usize;
        for step in 0u32..6 {
            let flushers = (0usize..16)
                .filter(|&slot| should_flush_keep_late(5, step, slot, 6, KEEP_LATE_RING_CAP))
                .count();
            assert!(
                (2..=3).contains(&flushers),
                "step={step} at M=6 should flush 2..=3 of 16 slots, got {flushers}"
            );
            m6_cycle_flushes += flushers;
        }
        assert_eq!(m6_cycle_flushes, 16);
        // Across 16 slots at M=8, exactly 2 slots flush on every step.
        for step in 0u32..16 {
            let flushers = (0usize..16)
                .filter(|&slot| should_flush_keep_late(5, step, slot, 8, KEEP_LATE_RING_CAP))
                .count();
            assert_eq!(flushers, 2, "step={step} should flush exactly 2/16 slots");
        }
        // Threshold L_flush forces flush regardless of phase.
        assert!(should_flush_keep_late(
            KEEP_LATE_L_FLUSH,
            1,
            2,
            6,
            KEEP_LATE_RING_CAP
        ));
        // Period 1 flushes whenever pending_rows > 0.
        assert!(should_flush_keep_late(1, 3, 5, 1, KEEP_LATE_RING_CAP));
        // Low remaining headroom forces flush before wrap overlap.
        assert!(should_flush_keep_late(17, 1, 2, 8, 48));
    }

    #[test]
    fn keep_late_flush_period_for_sm_scales_sm120_and_enforces_retain_capacity() {
        // Pure SM-aware resolver: GB10 (<= 64 SMs) stays 2, SM120+ (> 64 SMs) uses 5.
        assert_eq!(KEEP_LATE_DEFAULT_FLUSH_PERIOD, 2);
        assert_eq!(KEEP_LATE_HIGH_SM_DEFAULT_FLUSH_PERIOD, 5);
        for sm in [0u32, 32, 48, 64] {
            assert_eq!(keep_late_flush_period_for_sm(None, sm), 2, "sm={sm}");
            assert_eq!(keep_late_flush_period_for_sm(Some("0"), sm), 2, "sm={sm}");
            assert_eq!(
                keep_late_flush_period_for_sm(Some("bogus"), sm),
                2,
                "sm={sm}"
            );
        }
        for sm in [65u32, 110, 160, 170, 188] {
            assert_eq!(keep_late_flush_period_for_sm(None, sm), 5, "sm={sm}");
            assert_eq!(keep_late_flush_period_for_sm(Some("0"), sm), 5, "sm={sm}");
            assert_eq!(
                keep_late_flush_period_for_sm(Some("bogus"), sm),
                5,
                "sm={sm}"
            );
        }
        // Explicit ATLAS_QWEN4_GDN_KEEP_LATE_FLUSH_PERIOD override always wins.
        assert_eq!(keep_late_flush_period_for_sm(Some("1"), 188), 1);
        assert_eq!(keep_late_flush_period_for_sm(Some("3"), 48), 3);
        assert_eq!(keep_late_flush_period_for_sm(Some("3"), 188), 3);
        assert_eq!(keep_late_flush_period_for_sm(Some(" 7 "), 188), 7);
        assert_eq!(keep_late_flush_period_for_sm(Some("200"), 188), 64);

        // Cohort + retain_capacity invariant:
        // - C <= 1 stays M=1 (0-ULP FP32 exact) even on SM120.
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(None, 1, 188, KEEP_LATE_RING_CAP, GDN_LAZY_MAX_K),
            1
        );
        // - GB10 (<= 64 SMs) stays M=2 at C >= 2 regardless of ring_cap.
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(None, 64, 48, KEEP_LATE_RING_CAP, GDN_LAZY_MAX_K),
            2
        );
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(None, 64, 64, KEEP_LATE_RING_CAP, GDN_LAZY_MAX_K),
            2
        );
        // - SM120 (> 64 SMs) uses M=5 when retain_capacity >= L_max + max_k (64 <= 64).
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(
                None,
                64,
                188,
                KEEP_LATE_RING_CAP,
                GDN_LAZY_MAX_K
            ),
            5
        );
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(
                None,
                2,
                188,
                KEEP_LATE_L_MAX + GDN_LAZY_MAX_K,
                GDN_LAZY_MAX_K
            ),
            5
        );
        // - Tight retain_capacity (< L_max + max_k) or addition overflow falls back safely to 2.
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(
                None,
                64,
                188,
                KEEP_LATE_L_MAX + GDN_LAZY_MAX_K - 1,
                GDN_LAZY_MAX_K
            ),
            2
        );
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(None, 64, 188, KEEP_LATE_RING_CAP, usize::MAX),
            2
        );
        // - Explicit env override wins across all cohorts and SM counts.
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(
                Some(2),
                64,
                188,
                KEEP_LATE_RING_CAP,
                GDN_LAZY_MAX_K
            ),
            2
        );
        assert_eq!(
            resolve_keep_late_flush_period_for_sm(Some(4), 1, 188, 64, GDN_LAZY_MAX_K),
            4
        );

        // Across 64 slots at SM120 M=5, exactly 20% of slots flush per 5-step cycle (64 total).
        let mut m5_cycle_flushes = 0usize;
        for step in 0u32..5 {
            let flushers = (0usize..64)
                .filter(|&slot| should_flush_keep_late(5, step, slot, 5, KEEP_LATE_RING_CAP))
                .count();
            assert!(
                (12..=13).contains(&flushers),
                "step={step} at M=5 should flush 12..=13 of 64 slots, got {flushers}"
            );
            m5_cycle_flushes += flushers;
        }
        assert_eq!(m5_cycle_flushes, 64);
    }

    #[test]
    fn keep_late_stagger_never_overflows_l_flush_or_ring_cap_under_worst_case_accept_all() {
        // Verify ring-wrap disjointness and L_max bounds under 200 consecutive worst-case
        // accept_all rounds across all 16 slots for K=5, K=7, K=8, and K=16 (GDN_LAZY_MAX_K),
        // testing both default auto M=2 and explicit env override M=6 on canonical ring_cap=100
        // and the minimum legal ring_cap (L_max + K_max = 64).
        let default_period = resolve_keep_late_flush_period(None, 16);
        assert_eq!(default_period, 2);
        let sm120_period = resolve_keep_late_flush_period_for_sm(
            None,
            64,
            188,
            KEEP_LATE_RING_CAP,
            GDN_LAZY_MAX_K,
        );
        assert_eq!(sm120_period, 5);
        let override_period = resolve_keep_late_flush_period(Some(6), 16);
        assert_eq!(override_period, 6);
        for period in [default_period, sm120_period, override_period] {
            for ring_cap in [KEEP_LATE_RING_CAP, KEEP_LATE_L_MAX + GDN_LAZY_MAX_K] {
                for k in [5usize, 7, 8, GDN_LAZY_MAX_K] {
                    for slot in 0usize..16 {
                        let mut pending_base = 0usize;
                        let mut pending_rows = 0usize;
                        let mut max_pending_seen = 0usize;
                        for step in 0u32..200 {
                            let flush_now =
                                should_flush_keep_late(pending_rows, step, slot, period, ring_cap);
                            let write_base = (pending_base + pending_rows) % ring_cap;
                            // Read window [pending_base, +pending_rows) and verify write window
                            // [write_base, +k) must never wrap-overlap in the circular ring.
                            assert!(
                                pending_rows + k <= ring_cap,
                                "period={period} ring_cap={ring_cap} k={k} slot={slot} step={step}: \
                                 pending_rows={pending_rows} + k={k} exceeded ring_cap"
                            );
                            if flush_now {
                                pending_base = write_base;
                                pending_rows = 0;
                            } else {
                                assert!(
                                    pending_rows < KEEP_LATE_L_FLUSH,
                                    "period={period} ring_cap={ring_cap} k={k} slot={slot} step={step}: \
                                     deferred step must satisfy pending_rows ({pending_rows}) < L_flush ({KEEP_LATE_L_FLUSH})"
                                );
                            }
                            // Worst-case accept_all commits all k verified tokens.
                            let next_pending = pending_rows + k;
                            assert!(
                                next_pending <= KEEP_LATE_L_MAX,
                                "period={period} ring_cap={ring_cap} k={k} slot={slot} step={step}: \
                                 next_pending={next_pending} exceeded L_max={KEEP_LATE_L_MAX}"
                            );
                            if k == 5 {
                                assert!(
                                    next_pending <= KEEP_LATE_L_FLUSH,
                                    "K=5 at M={period} must keep pending_rows ({next_pending}) <= L_flush ({KEEP_LATE_L_FLUSH})"
                                );
                            }
                            if k == 8 {
                                assert!(
                                    next_pending <= KEEP_LATE_L_FLUSH,
                                    "K=8 accept_all must force flush at L_flush ({KEEP_LATE_L_FLUSH}), got {next_pending}"
                                );
                            }
                            pending_rows = next_pending;
                            max_pending_seen = max_pending_seen.max(pending_rows);
                        }
                        assert!(max_pending_seen <= KEEP_LATE_L_MAX);
                    }
                }
            }
        }
    }

    #[test]
    fn keep_late_ctrl_serialization_matches_cuda_abi() {
        let bytes = keep_late_ctrl_le_bytes(0x1122_3344_5566_7788, 93, 12, 5, 100, true);
        assert_eq!(bytes.len(), GDN_KEEP_LATE_CTRL_BYTES);
        assert_eq!(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 93);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 12);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 1);
    }

    #[test]
    fn gdn_lazy_retain_pool_uses_async_memset_and_rejects_capture_alloc() {
        let source = include_str!("mod.rs");
        let pool_fn = source
            .split("pub(super) fn gdn_lazy_retain_pool(")
            .nth(1)
            .unwrap()
            .split("pub(super) fn gdn_lazy_retain_current(")
            .next()
            .unwrap();
        assert!(
            pool_fn.contains("gpu.memset_async(ctrl, 0, ctrl_bytes, default_stream)"),
            "gdn_lazy_retain_pool must zero ctrl with memset_async"
        );
        assert!(
            !pool_fn.contains("gpu.memset("),
            "gdn_lazy_retain_pool must never call synchronous gpu.memset"
        );
        assert!(
            pool_fn.contains("active_capture_region().is_some()")
                && pool_fn.contains("gpu.stream_is_capturing("),
            "gdn_lazy_retain_pool must refuse lazy alloc inside capture"
        );
    }

    #[test]
    fn gdn_h_fp16_env_parser_and_abi_flag_bits() {
        assert_eq!(GDN_H_FP16_ENV, "ATLAS_QWEN4_GDN_H_FP16");
        assert_eq!(GDN_H_FP16_MIN_COHORT, 2);
        // Default ON when unset or "1"; disabled only on "0".
        assert!(parse_gdn_h_fp16_enabled(None));
        assert!(parse_gdn_h_fp16_enabled(Some("1")));
        assert!(parse_gdn_h_fp16_enabled(Some("true")));
        assert!(parse_gdn_h_fp16_enabled(Some("")));
        assert!(!parse_gdn_h_fp16_enabled(Some("0")));
        assert!(!parse_gdn_h_fp16_enabled(Some(" 0 ")));

        // ABI flag bits match CUDA header defines and stay pairwise disjoint.
        assert_eq!(GDN_FUSED_FLAG_LOAD_SCALED_FP16, 16);
        assert_eq!(GDN_FUSED_FLAG_STORE_SCALED_FP16, 32);
        let base_flags = ops::GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN
            | GDN_FUSED_FLAG_KEEP_LATE
            | GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;
        assert_eq!(base_flags & GDN_FUSED_FLAG_LOAD_SCALED_FP16, 0);
        assert_eq!(base_flags & GDN_FUSED_FLAG_STORE_SCALED_FP16, 0);

        let ctrl_ptr = DevicePtr(0x1000);
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::Fp32,
                store_format: KeepLateHFormat::Fp32,
            }
            .desc_flags(),
            base_flags
        );
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::Fp32,
                store_format: KeepLateHFormat::ScaledFp16,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_STORE_SCALED_FP16
        );
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::ScaledFp16,
                store_format: KeepLateHFormat::ScaledFp16,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_LOAD_SCALED_FP16 | GDN_FUSED_FLAG_STORE_SCALED_FP16
        );
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::ScaledFp16,
                store_format: KeepLateHFormat::Fp32,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_LOAD_SCALED_FP16
        );
    }

    #[test]
    fn gdn_h_fp16_all_four_state_transitions() {
        // 1. C == 1 or env disabled: 100% untouched 0-ULP FP32 -> FP32.
        for base_flush in [false, true] {
            let plan = plan_keep_late_h_format(KeepLateHFormat::Fp32, true, 1, base_flush);
            assert_eq!(
                plan,
                KeepLateHFormatPlan {
                    load_format: KeepLateHFormat::Fp32,
                    store_format: KeepLateHFormat::Fp32,
                    flush_now: base_flush,
                },
                "C=1 base_flush={base_flush} must stay pure FP32"
            );
            assert_eq!(plan.next_format(), KeepLateHFormat::Fp32);
        }
        let disabled = plan_keep_late_h_format(KeepLateHFormat::Fp32, false, 16, true);
        assert_eq!(
            disabled,
            KeepLateHFormatPlan {
                load_format: KeepLateHFormat::Fp32,
                store_format: KeepLateHFormat::Fp32,
                flush_now: true,
            }
        );
        assert_eq!(disabled.next_format(), KeepLateHFormat::Fp32);

        // 2. First C >= 2 step from FP32: stays FP32 on deferred step, flips to
        //    ScaledFp16 ONLY when flush_now actually executes.
        for cohort in [2usize, 3] {
            let deferred = plan_keep_late_h_format(KeepLateHFormat::Fp32, true, cohort, false);
            assert_eq!(
                deferred,
                KeepLateHFormatPlan {
                    load_format: KeepLateHFormat::Fp32,
                    store_format: KeepLateHFormat::ScaledFp16,
                    flush_now: false,
                },
                "C={cohort} deferred step"
            );
            assert_eq!(deferred.next_format(), KeepLateHFormat::Fp32);
        }
        for cohort in [2usize, 16] {
            let first_flush = plan_keep_late_h_format(KeepLateHFormat::Fp32, true, cohort, true);
            assert_eq!(
                first_flush,
                KeepLateHFormatPlan {
                    load_format: KeepLateHFormat::Fp32,
                    store_format: KeepLateHFormat::ScaledFp16,
                    flush_now: true,
                },
                "C={cohort} first flush"
            );
            assert_eq!(first_flush.next_format(), KeepLateHFormat::ScaledFp16);
        }

        // 3. Steady-state C >= 2 from ScaledFp16: load & store ScaledFp16.
        for cohort in [2usize, 8] {
            for base_flush in [false, true] {
                let steady =
                    plan_keep_late_h_format(KeepLateHFormat::ScaledFp16, true, cohort, base_flush);
                assert_eq!(
                    steady,
                    KeepLateHFormatPlan {
                        load_format: KeepLateHFormat::ScaledFp16,
                        store_format: KeepLateHFormat::ScaledFp16,
                        flush_now: base_flush,
                    },
                    "C={cohort} steady state"
                );
                assert_eq!(steady.next_format(), KeepLateHFormat::ScaledFp16);
            }
        }

        // 4. Exit / C == 1 / disabled from ScaledFp16: forces flush_now=true to
        //    unpack back to canonical FP32 EVEN when base_flush_now == false (pending == 0).
        for (enabled, cohort) in [(true, 1usize), (false, 2), (false, 16)] {
            let unpack =
                plan_keep_late_h_format(KeepLateHFormat::ScaledFp16, enabled, cohort, false);
            assert_eq!(
                unpack,
                KeepLateHFormatPlan {
                    load_format: KeepLateHFormat::ScaledFp16,
                    store_format: KeepLateHFormat::Fp32,
                    flush_now: true,
                },
                "enabled={enabled} C={cohort} must force unpack to FP32 even with pending=0"
            );
            assert_eq!(unpack.next_format(), KeepLateHFormat::Fp32);
        }
    }

    #[test]
    fn gdn_h_fp16_slot_lifecycle_and_pending_count_invariant() {
        let pool = GdnLazyRetainPool {
            base: DevicePtr(0x1_0000),
            buf_bytes: 1_054_720,
            bufs_per_lane: 2,
            lane_stride: 1_335_296,
            lanes: 4,
            allocation_bytes: 4 * 1_335_296,
            late_ctrl_dev: DevicePtr(0x2_0000),
            late_ctrl_lane_stride: GDN_KEEP_LATE_CTRL_BYTES,
            late_ctrl_slab_id: std::sync::atomic::AtomicU64::new(0),
            single_seq_descs_dev: DevicePtr(0x3_0000),
            pending_count: std::sync::atomic::AtomicU32::new(0),
            slots: parking_lot::Mutex::new(vec![LatePendingSlot::default(); 4]),
        };
        let count = || {
            pool.pending_count
                .load(std::sync::atomic::Ordering::Acquire)
        };

        let mut slots = pool.slots.lock();
        let entry = &mut slots[0];
        entry.ring_cap = KEEP_LATE_RING_CAP as u16;
        assert!(!entry.needs_fp32_flush());
        assert_eq!(count(), 0);

        // Step 0 at C=8 with pending=0: no flush -> stays Fp32, pending_count=0.
        let p0 = plan_keep_late_h_format(entry.h_format, true, 8, false);
        entry.staged_flush_now = p0.flush_now;
        entry.staged_load_format = p0.load_format;
        entry.staged_store_format = p0.store_format;
        entry.ctrl_staged = true;
        complete_staged_verify_entry(&pool, entry);
        assert_eq!(entry.h_format, KeepLateHFormat::Fp32);
        assert!(!entry.needs_fp32_flush());
        assert_eq!(count(), 0);

        // Commit 4 rows -> pending_rows=4, Fp32, pending_count=1.
        let was = entry.needs_fp32_flush();
        entry.pending_rows = 4;
        update_slot_flush_needed(&pool, was, entry.needs_fp32_flush());
        assert_eq!(count(), 1);

        // Step 1 at C=8 deferred (flush=false): MUST stay Fp32, pending=4, count=1.
        let p1 = plan_keep_late_h_format(entry.h_format, true, 8, false);
        entry.staged_flush_now = p1.flush_now;
        entry.staged_load_format = p1.load_format;
        entry.staged_store_format = p1.store_format;
        entry.staged_write_base = 4;
        entry.ctrl_staged = true;
        complete_staged_verify_entry(&pool, entry);
        assert_eq!(entry.h_format, KeepLateHFormat::Fp32);
        assert_eq!(entry.pending_rows, 4);
        assert_eq!(count(), 1);

        // Step 2 at C=8 flush (flush=true): transitions Fp32 -> ScaledFp16,
        // pending_rows=0, yet needs_fp32_flush() stays TRUE and count stays 1!
        let p2 = plan_keep_late_h_format(entry.h_format, true, 8, true);
        assert_eq!(p2.load_format, KeepLateHFormat::Fp32);
        assert_eq!(p2.store_format, KeepLateHFormat::ScaledFp16);
        entry.staged_flush_now = p2.flush_now;
        entry.staged_load_format = p2.load_format;
        entry.staged_store_format = p2.store_format;
        entry.staged_write_base = 4;
        entry.ctrl_staged = true;
        complete_staged_verify_entry(&pool, entry);
        assert_eq!(entry.h_format, KeepLateHFormat::ScaledFp16);
        assert_eq!(entry.pending_rows, 0);
        assert!(entry.needs_fp32_flush());
        assert_eq!(count(), 1);

        // Commit 3 rows while already ScaledFp16: MUST NOT double-increment pending_count.
        let was = entry.needs_fp32_flush();
        entry.pending_rows = 3;
        update_slot_flush_needed(&pool, was, entry.needs_fp32_flush());
        assert_eq!(count(), 1);

        // Steady-state flush at C=8: ScaledFp16 -> ScaledFp16, pending=0, count stays 1.
        let p3 = plan_keep_late_h_format(entry.h_format, true, 8, true);
        assert_eq!(p3.load_format, KeepLateHFormat::ScaledFp16);
        assert_eq!(p3.store_format, KeepLateHFormat::ScaledFp16);
        entry.staged_flush_now = p3.flush_now;
        entry.staged_load_format = p3.load_format;
        entry.staged_store_format = p3.store_format;
        entry.staged_write_base = 7;
        entry.ctrl_staged = true;
        complete_staged_verify_entry(&pool, entry);
        assert_eq!(entry.h_format, KeepLateHFormat::ScaledFp16);
        assert_eq!(entry.pending_rows, 0);
        assert!(entry.needs_fp32_flush());
        assert_eq!(count(), 1);

        // Transition from C=8 to C=2 stays ScaledFp16 (no unpack flush needed).
        let p_c2 = plan_keep_late_h_format(entry.h_format, true, 2, false);
        assert!(!p_c2.flush_now);
        assert_eq!(p_c2.load_format, KeepLateHFormat::ScaledFp16);
        assert_eq!(p_c2.store_format, KeepLateHFormat::ScaledFp16);

        // Transition to C=1 with pending_rows == 0: forces flush_now=true to
        // unpack ScaledFp16 -> Fp32 and drops pending_count back to 0.
        let p4 = plan_keep_late_h_format(entry.h_format, true, 1, false);
        assert!(p4.flush_now);
        assert_eq!(p4.load_format, KeepLateHFormat::ScaledFp16);
        assert_eq!(p4.store_format, KeepLateHFormat::Fp32);
        entry.staged_flush_now = p4.flush_now;
        entry.staged_load_format = p4.load_format;
        entry.staged_store_format = p4.store_format;
        entry.staged_write_base = 7;
        entry.ctrl_staged = true;
        complete_staged_verify_entry(&pool, entry);
        assert_eq!(entry.h_format, KeepLateHFormat::Fp32);
        assert_eq!(entry.pending_rows, 0);
        assert!(!entry.needs_fp32_flush());
        assert_eq!(count(), 0);
    }

    #[test]
    fn gdn_h_fp16_all_fp32_flush_guards_check_needs_fp32_flush() {
        let source = include_str!("mod.rs");
        let prod = source.rsplit_once("#[cfg(test)]").unwrap().0;
        for fn_name in [
            "fn gdn_keep_late_reconcile_abort_inner(",
            "fn gdn_keep_late_reset_slot_inner(",
            "fn gdn_keep_late_has_pending_inner(",
            "fn gdn_keep_late_flush_slot_inner(",
            "fn flush_keep_late_for_h_state(",
            "fn flush_keep_late_for_lane_if_different_h(",
            "fn stage_qwen4_keep_late_ctrl_bytes_inner(",
        ] {
            let body = prod
                .split(fn_name)
                .nth(1)
                .unwrap_or_else(|| panic!("missing {fn_name}"))
                .split("\n    pub(super) fn ")
                .next()
                .unwrap();
            assert!(
                body.contains("needs_fp32_flush"),
                "{fn_name} must check needs_fp32_flush so Scaled-FP16 h_state is never read as FP32"
            );
        }
    }

    #[test]
    fn keep_late_ctrl_slab_layout_is_slot_major_contiguous_and_releases_cleanly() {
        use super::{
            alloc_keep_late_ctrl_slab_slice, keep_late_ctrl_slab_offset,
            release_keep_late_ctrl_slab_slice,
        };
        use spark_runtime::gpu::mock::MockGpuBackend;

        // Exact offset math: slot-major [lanes][num_ssm_layers][32 B].
        assert_eq!(keep_late_ctrl_slab_offset(0, 0, 36), Some(0));
        assert_eq!(keep_late_ctrl_slab_offset(0, 35, 36), Some(35 * 32));
        assert_eq!(keep_late_ctrl_slab_offset(1, 0, 36), Some(36 * 32));
        assert_eq!(
            keep_late_ctrl_slab_offset(15, 35, 36),
            Some((15 * 36 + 35) * 32)
        );
        assert_eq!(keep_late_ctrl_slab_offset(0, 36, 36), None);
        assert_eq!(keep_late_ctrl_slab_offset(0, 0, 0), None);

        // 36 layers share one contiguous GPU slab (`lanes=64`, `73,728 B`),
        // producing one zero-gap 18,432 B span across slots 0..16 and a
        // 1,152 B span for any single slot (C=1).
        let gpu = MockGpuBackend::new();
        let lanes = 64usize;
        let num_layers = 36usize;
        let mut slices: Vec<(DevicePtr, usize, u64)> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let (base, stride, slab_id) =
                alloc_keep_late_ctrl_slab_slice(&gpu, lanes, num_layers, 0)
                    .expect("alloc keep_late ctrl slab slice");
            assert_eq!(stride, num_layers * GDN_KEEP_LATE_CTRL_BYTES);
            assert_ne!(slab_id, 0);
            if layer > 0 {
                assert_eq!(slab_id, slices[0].2, "all 36 layers must share one slab_id");
                assert_eq!(
                    base.0,
                    slices[0].0.0 + (layer * GDN_KEEP_LATE_CTRL_BYTES) as u64,
                    "layer {layer} must sit at exact 32 B offset inside shared slab"
                );
            }
            slices.push((base, stride, slab_id));
        }

        // Verify slot-major adjacency across all 16 active slots x 36 layers.
        let slab_base = slices[0].0.0;
        for slot in 0..16usize {
            for (layer, &(layer_base, stride, _)) in slices.iter().enumerate() {
                let ptr = layer_base.offset(slot * stride);
                let expected =
                    slab_base + keep_late_ctrl_slab_offset(slot, layer, num_layers).unwrap() as u64;
                assert_eq!(ptr.0, expected, "slot={slot} layer={layer}");
            }
        }
        let end_of_16 = slices[35].0.offset(15 * slices[35].1).0 + GDN_KEEP_LATE_CTRL_BYTES as u64;
        assert_eq!(end_of_16 - slab_base, (16 * 36 * 32) as u64);

        // A second model (layer 37) gets a fresh slab_id with non-aliasing base.
        let (other_base, _, other_slab_id) =
            alloc_keep_late_ctrl_slab_slice(&gpu, lanes, num_layers, 0)
                .expect("second model allocates fresh slab");
        assert_ne!(other_slab_id, slices[0].2);
        assert_ne!(other_base, slices[0].0);

        for (_, _, slab_id) in slices {
            release_keep_late_ctrl_slab_slice(slab_id);
        }
        release_keep_late_ctrl_slab_slice(other_slab_id);
    }

    #[test]
    fn gdn_split_v_policy_auto_enables_for_c1_fp32_on_wide_sm_gpus_only() {
        use super::{
            GdnSplitVPolicy, KeepLateHFormat, parse_gdn_split_v_policy, resolve_gdn_split_v_enabled,
        };

        assert_eq!(parse_gdn_split_v_policy(None), GdnSplitVPolicy::Auto);
        assert_eq!(
            parse_gdn_split_v_policy(Some("auto")),
            GdnSplitVPolicy::Auto
        );
        assert_eq!(
            parse_gdn_split_v_policy(Some("1")),
            GdnSplitVPolicy::Enabled
        );
        assert_eq!(
            parse_gdn_split_v_policy(Some("true")),
            GdnSplitVPolicy::Enabled
        );
        assert_eq!(
            parse_gdn_split_v_policy(Some("0")),
            GdnSplitVPolicy::Disabled
        );
        assert_eq!(
            parse_gdn_split_v_policy(Some("false")),
            GdnSplitVPolicy::Disabled
        );

        // Auto: engages on >64 SMs (188-SM RTX PRO 6000 / SM103) for C=1 FP32.
        assert!(resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            188,
            1,
            KeepLateHFormat::Fp32,
        ));
        assert!(resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            128,
            1,
            KeepLateHFormat::Fp32,
        ));
        // Auto: stays off on <=64 SMs (48-SM GB10) and for C>=2 cohorts.
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            48,
            1,
            KeepLateHFormat::Fp32,
        ));
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            64,
            1,
            KeepLateHFormat::Fp32,
        ));
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            188,
            2,
            KeepLateHFormat::Fp32,
        ));
        // Explicit override: "1" forces on even on 48-SM GB10 for A/B & parity;
        // "0" disables even on 188-SM GPUs.
        assert!(resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Enabled,
            48,
            1,
            KeepLateHFormat::Fp32,
        ));
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Disabled,
            188,
            1,
            KeepLateHFormat::Fp32,
        ));
        // Never allows Scaled-FP16 H-state under any policy (avoids in-place unpack race).
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Enabled,
            188,
            1,
            KeepLateHFormat::ScaledFp16,
        ));
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            188,
            1,
            KeepLateHFormat::ScaledFp16,
        ));
        assert!(!resolve_gdn_split_v_enabled(
            GdnSplitVPolicy::Auto,
            188,
            1,
            KeepLateHFormat::ScaledFp8,
        ));
    }

    #[test]
    fn gdn_h_fp8_opt_in_env_flags_and_state_transitions() {
        use super::{
            GDN_FUSED_FLAG_LOAD_SCALED_FP8, GDN_FUSED_FLAG_STORE_SCALED_FP8, GDN_H_FP8_ENV,
            GDN_H_FP8_HEAD_BYTES, parse_gdn_h_fp8_enabled, plan_keep_late_h_format_ex,
        };

        assert_eq!(GDN_H_FP8_ENV, "ATLAS_QWEN4_GDN_H_FP8");
        assert_eq!(GDN_H_FP8_HEAD_BYTES, 16_640);
        // Default OFF until explicitly opted in.
        assert!(!parse_gdn_h_fp8_enabled(None));
        assert!(!parse_gdn_h_fp8_enabled(Some("")));
        assert!(!parse_gdn_h_fp8_enabled(Some("0")));
        assert!(!parse_gdn_h_fp8_enabled(Some("false")));
        assert!(parse_gdn_h_fp8_enabled(Some("1")));
        assert!(parse_gdn_h_fp8_enabled(Some(" 1 ")));
        assert!(parse_gdn_h_fp8_enabled(Some("true")));
        assert!(parse_gdn_h_fp8_enabled(Some("ON")));

        // Flag bits match CUDA defines (64 / 128) and stay pairwise disjoint from FP16 bits (16 / 32).
        assert_eq!(GDN_FUSED_FLAG_LOAD_SCALED_FP8, 64);
        assert_eq!(GDN_FUSED_FLAG_STORE_SCALED_FP8, 128);
        let base_flags = ops::GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN
            | GDN_FUSED_FLAG_KEEP_LATE
            | GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;
        assert_eq!(
            (base_flags | GDN_FUSED_FLAG_LOAD_SCALED_FP16 | GDN_FUSED_FLAG_STORE_SCALED_FP16)
                & (GDN_FUSED_FLAG_LOAD_SCALED_FP8 | GDN_FUSED_FLAG_STORE_SCALED_FP8),
            0
        );

        let ctrl_ptr = DevicePtr(0x2000);
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::Fp32,
                store_format: KeepLateHFormat::ScaledFp8,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_STORE_SCALED_FP8
        );
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::ScaledFp8,
                store_format: KeepLateHFormat::ScaledFp8,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_LOAD_SCALED_FP8 | GDN_FUSED_FLAG_STORE_SCALED_FP8
        );
        assert_eq!(
            KeepLateStagedCtrl {
                ctrl_ptr,
                load_format: KeepLateHFormat::ScaledFp8,
                store_format: KeepLateHFormat::Fp32,
            }
            .desc_flags(),
            base_flags | GDN_FUSED_FLAG_LOAD_SCALED_FP8
        );

        // C=1 stays canonical FP32 even when FP8 is enabled; dropping from ScaledFp8 to C=1 forces unpack.
        let c1_fp32 = plan_keep_late_h_format_ex(KeepLateHFormat::Fp32, true, true, 1, false);
        assert_eq!(c1_fp32.load_format, KeepLateHFormat::Fp32);
        assert_eq!(c1_fp32.store_format, KeepLateHFormat::Fp32);
        assert!(!c1_fp32.flush_now);

        let c1_unpack =
            plan_keep_late_h_format_ex(KeepLateHFormat::ScaledFp8, true, true, 1, false);
        assert_eq!(c1_unpack.load_format, KeepLateHFormat::ScaledFp8);
        assert_eq!(c1_unpack.store_format, KeepLateHFormat::Fp32);
        assert!(c1_unpack.flush_now);
        assert_eq!(c1_unpack.next_format(), KeepLateHFormat::Fp32);

        // C>=2 with opt-in FP8 supersedes default-ON FP16.
        let enter_fp8 = plan_keep_late_h_format_ex(KeepLateHFormat::Fp32, true, true, 64, true);
        assert_eq!(enter_fp8.load_format, KeepLateHFormat::Fp32);
        assert_eq!(enter_fp8.store_format, KeepLateHFormat::ScaledFp8);
        assert!(enter_fp8.flush_now);
        assert_eq!(enter_fp8.next_format(), KeepLateHFormat::ScaledFp8);

        let steady_fp8 =
            plan_keep_late_h_format_ex(KeepLateHFormat::ScaledFp8, true, true, 64, false);
        assert_eq!(steady_fp8.load_format, KeepLateHFormat::ScaledFp8);
        assert_eq!(steady_fp8.store_format, KeepLateHFormat::ScaledFp8);
        assert!(!steady_fp8.flush_now);
        assert_eq!(steady_fp8.next_format(), KeepLateHFormat::ScaledFp8);

        // Slot tracking reports needs_fp32_flush() == true while H is in ScaledFp8.
        let mut slot = LatePendingSlot::default();
        assert!(!slot.needs_fp32_flush());
        slot.h_format = KeepLateHFormat::ScaledFp8;
        assert!(slot.needs_fp32_flush());
    }
}
