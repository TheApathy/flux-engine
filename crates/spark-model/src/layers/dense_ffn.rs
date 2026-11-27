// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU/GeGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(act(gate) * up)
//! Single-token NVFP4 decode uses 2 fused launches for SiLU (dual GEMV +
//! SiLU-fused down GEMV) or 3 launches for GeLU (dual GEMV + GeLU-mul + down
//! GEMV); BF16 weights use 4 launches (separate gate/up GEMVs + activation + down).

mod cublaslt_ffn;
mod cublaslt_proj;
mod decode;
mod e2m1_flashinfer;
mod fused_rmsnorm;
mod kgamma;
mod layer;
mod policy;
mod prefill;
mod sm120_coalesced;

#[cfg(test)]
mod e2m1_prefill_reuse_tests;
#[cfg(test)]
mod exact_route_tests;
#[cfg(test)]
mod flashinfer_merged_ffn_tests;
#[cfg(test)]
mod gpu_parity_tests;
#[cfg(test)]
mod memlean_tests;
#[cfg(test)]
mod prefill_route_tests;
#[cfg(test)]
mod w3_route_tests;

pub(crate) use cublaslt_proj::{
    invalidate_precomputed_proj_absmax, prepare_precomputed_proj_absmax,
    proj_27b_cublaslt_nvfp4_can_replace_fp8, try_cublaslt_nvfp4_proj_qkv,
    try_cublaslt_nvfp4_proj_single,
};
pub use e2m1_flashinfer::SharedE2m1Scratch;
pub use fused_rmsnorm::PreparedRmsNormNvfp4;
pub use layer::{DenseFfnLayer, DenseFfnWeights, DenseFfnWeightsBf16, FfnActivation};
pub(crate) use policy::{CUBLASLT_NVFP4_MIN_M, prefill_fuse_absmax_enabled};
pub(crate) use sm120_coalesced::{
    sm120_coalesced_max_m, try_coalesced_nvfp4_sm120_small_m,
    try_coalesced_nvfp4_sm120_small_m_gateup_silu, warm_coalesced_nvfp4_sm120_kernels,
};

#[cfg(all(test, feature = "cuda"))]
use cublaslt_proj::{
    CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF, CUBLASLT_DYNAMIC_SCALARS_BYTES, CUBLASLT_DYNAMIC_SCALE2_OFF,
    CUBLASLT_DYNAMIC_STATUS_OFF, CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES,
    swizzle_nvfp4_scales_128x4_host,
};
#[cfg(test)]
use e2m1_flashinfer::{
    E2M1_KMAJOR_M256_MIN_M, E2m1KmajorKernel, E2m1PrefillRoute, E2m1Scope, E2m1SelectedKernel,
    FlashinferFfnPrefillRoute, e2m1_kmajor_kernel, e2m1_kmajor_projection_shape,
    e2m1_prefill_route, e2m1_scratch_layout, e2m1_selected_kernel, flashinfer_ffn_prefill_route,
    validated_e2m1_checkpoint_scales,
};
#[cfg(all(test, feature = "cuda", target_os = "linux"))]
use e2m1_flashinfer::{
    ensure_flashinfer_ffn_disjoint, flashinfer_ffn_activation_scale_extent, flashinfer_ffn_extent,
    merged_gate_up_views,
};
#[cfg(test)]
use policy::{
    CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT, CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES,
    CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES, CublasLtMergedGateUpPolicy,
    CublasLtNvfp4Policy, ExactFfnDispatch, ExactFfnMaterializedAvailability,
    ExactFfnMaterializedRoute, ExactFfnPhysicalRoute, ExactFfnRowOffsets, FfnTcPolicy,
    FlashInferFfnAvailability, PrefillFfnFastRoute, PrefillFusedEpilogueRoute,
    Proj27bCublasLtNvfp4Policy, cublaslt_fused_silu_quantize_scale,
    cublaslt_merged_gate_up_allow_materialize, cublaslt_merged_gate_up_vram_ok,
    cublaslt_merged_gate_up_vram_ok_for_sm, cublaslt_nvfp4_route_enabled,
    cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill, exact_ffn_auto_kgamma_applicable,
    exact_ffn_dispatch, exact_ffn_materialized_route, exact_ffn_physical_route,
    exact_ffn_row_offsets, is_blackwell_dc_compute_capability, is_sm120_compute_capability,
    parse_cublaslt_merged_gate_up_policy, parse_cublaslt_nvfp4_policy, parse_ffn_prefill_l2_strip,
    parse_ffn_tc_policy, parse_proj_27b_cublaslt_nvfp4_policy, prefill_ffn_fast_route,
    prefill_fused_epilogue_route, proj_27b_cublaslt_nvfp4_shape_ok, resolve_ffn_tc_for_rows,
    resolve_kgamma_cublaslt_opt_in, resolve_kgamma_sm_auto_flag, resolve_proj_27b_cublaslt_nvfp4,
    resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill, resolve_sm_auto_opt_in,
    select_cublaslt_quantize_128x4_kernel, supports_warp_nvfp4_mma, w3_kgamma_applicable,
};
#[cfg(test)]
use sm120_coalesced::{Sm120CoalescedSmallMRoute, select_sm120_coalesced_small_m_route};
