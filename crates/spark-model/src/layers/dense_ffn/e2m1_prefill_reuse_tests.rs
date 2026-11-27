// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::KernelHandle;

#[cfg(feature = "cuda")]
use super::swizzle_nvfp4_scales_128x4_host;
use super::{
    CublasLtNvfp4Policy, E2M1_KMAJOR_M256_MIN_M, E2m1KmajorKernel, E2m1PrefillRoute, E2m1Scope,
    E2m1SelectedKernel, FlashInferFfnAvailability, cublaslt_nvfp4_route_enabled,
    e2m1_kmajor_kernel, e2m1_kmajor_projection_shape, e2m1_prefill_route, e2m1_selected_kernel,
    is_blackwell_dc_compute_capability, is_sm120_compute_capability, parse_cublaslt_nvfp4_policy,
    supports_warp_nvfp4_mma, validated_e2m1_checkpoint_scales,
};

#[test]
fn base_w4a4_selector_is_full_authoritative_and_fail_closed() {
    assert_eq!(
        e2m1_prefill_route(false, false, true, false, false),
        E2m1PrefillRoute::Disabled
    );
    assert_eq!(
        e2m1_prefill_route(true, false, false, false, false),
        E2m1PrefillRoute::Ineligible
    );
    assert_eq!(
        e2m1_prefill_route(true, false, true, true, false),
        E2m1PrefillRoute::Complete(E2m1Scope::Full)
    );
    assert_eq!(
        e2m1_prefill_route(false, true, true, false, true),
        E2m1PrefillRoute::Complete(E2m1Scope::DownOnly)
    );
    assert_eq!(
        e2m1_prefill_route(true, false, true, false, true),
        E2m1PrefillRoute::Missing(E2m1Scope::Full)
    );
    assert_eq!(
        e2m1_prefill_route(false, true, true, true, false),
        E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)
    );
    assert_eq!(
        e2m1_prefill_route(true, true, true, false, true),
        E2m1PrefillRoute::Missing(E2m1Scope::Full),
        "full must never downgrade to a ready down-only route"
    );
}

#[test]
fn selected_w4a4_kernel_names_the_actual_m2048_boundary() {
    assert_eq!(
        e2m1_selected_kernel(false, true, 8192),
        E2m1SelectedKernel::RowMajorM64
    );
    assert_eq!(
        e2m1_selected_kernel(true, true, 2047),
        E2m1SelectedKernel::KmajorM128
    );
    assert_eq!(
        e2m1_selected_kernel(true, true, 2048),
        E2m1SelectedKernel::KmajorM256
    );
    assert_eq!(
        e2m1_selected_kernel(true, false, 8192),
        E2m1SelectedKernel::KmajorM128
    );
}

#[test]
fn base_w4a4_missing_route_precedes_projection_and_marker_is_actual_only() {
    let source = concat!(include_str!("prefill.rs"), include_str!("policy.rs"));
    let select = source.find("let e2m1_route =").expect("W4A4 selector");
    let projection = source[select..]
        .find("if e2m1_fast_path {")
        .map(|offset| select + offset)
        .expect("W4A4 projection");
    let selection = &source[select..projection];
    assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::Full)"));
    assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)"));
    assert!(selection.contains("before gate projection"));

    let log_start = source.find("fn log_e2m1_route_once").expect("W4A4 logger");
    // Inspect this logger, not unrelated helpers inserted after it.
    // Inner blocks are indented; this is its top-level closing brace.
    let log_end = source[log_start..]
        .find("\n}\n")
        .map(|offset| log_start + offset + 3)
        .expect("W4A4 logger boundary");
    let logger = &source[log_start..log_end];
    assert!(logger.contains("ENGAGED ATLAS_E2M1_PREFILL"));
    for field in [
        "scope={scope}",
        "gate_up={gate_up}",
        "e2m1_kernel={kernel}",
        "activation_scale={activation_scale}",
        "down_input={down_input}",
        "activation={activation}",
        "M={}",
        "H={}",
        "intermediate={}",
    ] {
        assert!(logger.contains(field), "missing actual field {field}");
    }
    assert!(!logger.contains("requested"));
    assert!(!logger.contains("has_e2m1"));
}

#[test]
fn kmajor_m128_accepts_only_native_ffn_tile_geometry() {
    assert!(e2m1_kmajor_projection_shape(8192, 17408, 5120));
    assert!(e2m1_kmajor_projection_shape(8192, 5120, 17408));
    assert!(!e2m1_kmajor_projection_shape(127, 17408, 5120));
    assert!(!e2m1_kmajor_projection_shape(128, 17409, 5120));
    assert!(!e2m1_kmajor_projection_shape(128, 17408, 5119));
}

#[test]
fn kmajor_kernel_stages_transformed_weights_for_eight_row_warps() {
    let source =
        include_str!("../../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
    let start = source
        .find("void nvfp4_nvfp4_gemm_kmajor_m128(")
        .expect("K-major W4A4 kernel is missing");
    let end = source[start..]
        .find("// K-major M256 shadow.")
        .map(|offset| start + offset)
        .expect("K-major kernel boundary is missing");
    let body = &source[start..end];

    assert!(source.contains("__launch_bounds__(256)\nvoid nvfp4_nvfp4_gemm_kmajor_m128"));
    assert!(body.contains("const unsigned char* __restrict__ B_packed_t"));
    assert!(body.contains("const unsigned char* __restrict__ B_scale_t"));
    assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
    assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
    assert!(body.contains("smem_Bp_km[2][N_TILE_LG][K_STEP / 2]"));
    assert!(body.contains("smem_Bs_km[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
    assert!(body.contains("warp_m_offset = warp_id * 16"));
    assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
    assert!(body.contains("acc[nt][0] * scale2_ab"));
    assert_eq!(body.matches("__syncthreads();").count(), 2);
}

#[test]
fn kmajor_m256_routes_only_at_the_saturated_long_prefill_threshold() {
    assert_eq!(E2M1_KMAJOR_M256_MIN_M, 2048);
    for m in [0, 127, 128, 256, 1024, 2047] {
        assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M128);
    }
    for m in [2048, 2049, 8192, u32::MAX] {
        assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M256);
        assert_eq!(e2m1_kmajor_kernel(m, false), E2m1KmajorKernel::M128);
    }
}

#[test]
fn kmajor_m256_reuses_one_weight_tile_across_sixteen_row_warps() {
    let source =
        include_str!("../../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
    let start = source
        .find("void nvfp4_nvfp4_gemm_kmajor_m256(")
        .expect("K-major M256 W4A4 kernel is missing");
    let end = source[start..]
        .find("// FIX PATH 3 LANDED")
        .map(|offset| start + offset)
        .expect("K-major M256 kernel boundary is missing");
    let body = &source[start..end];

    assert!(source.contains("__launch_bounds__(512, 1)\nvoid nvfp4_nvfp4_gemm_kmajor_m256"));
    assert!(body.contains("cta_m = blockIdx.y * 256u"));
    assert!(body.contains("smem_Ap_km256[2][256][K_STEP / 2]"));
    assert!(body.contains("smem_Bp_km256[2][N_TILE_LG][K_STEP / 2]"));
    assert!(body.contains("smem_Bs_km256[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
    assert!(body.contains("if (threadIdx.x < 256)"));
    assert!(body.contains("warp_m_offset = warp_id * 16"));
    assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
    assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
    assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
    assert!(body.contains("acc[nt][0] * scale2_ab"));
    assert_eq!(body.matches("__syncthreads();").count(), 2);
}

#[test]
fn kmajor_dispatch_fails_closed_and_uses_only_transformed_weights() {
    let source = concat!(
        include_str!("prefill.rs"),
        include_str!("e2m1_flashinfer.rs")
    );
    let start = source
        .find("let e2m1_kmajor_requested")
        .expect("K-major request gate is missing");
    let end = source[start..]
        .find("if v2_fast_path {")
        .map(|offset| start + offset)
        .expect("K-major dispatch boundary is missing");
    let body = &source[start..end];

    assert!(body.contains("self.has_e2m1_kmajor_runtime()"));
    assert!(body.contains("self.has_e2m1_kmajor_m256_runtime()"));
    assert!(body.contains("self.has_transposed_ffn()"));
    assert!(body.contains("e2m1_kmajor_projection_shape(m, inter, h)"));
    assert!(body.contains("self.gate_proj_t.as_ref().unwrap()"));
    assert!(body.contains("self.up_proj_t.as_ref().unwrap()"));
    assert!(body.contains("self.down_proj_t.as_ref().unwrap()"));
    assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m128("));
    assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m256("));

    let wrapper = include_str!("../ops/gemm_dense.rs");
    assert!(wrapper.contains("pub fn nvfp4_nvfp4_gemm_kmajor_m256("));
    assert!(wrapper.contains(".grid([n / 128, div_ceil(m, 256), 1])"));
    assert!(wrapper.contains(".block([512, 1, 1])"));
}

#[test]
fn static_scale_matches_merged_gate_up_contract_and_rejects_bad_metadata() {
    let scales = validated_e2m1_checkpoint_scales(0.001, 0.00125, 0.002).unwrap();
    assert_eq!(scales.gate_up.to_bits(), 0.00125_f32.to_bits());
    assert_eq!(scales.down.to_bits(), 0.002_f32.to_bits());

    for bad in [0.0, -0.1, f32::INFINITY, f32::NAN] {
        assert!(validated_e2m1_checkpoint_scales(bad, 0.001, 0.002).is_err());
        assert!(validated_e2m1_checkpoint_scales(0.001, bad, 0.002).is_err());
        assert!(validated_e2m1_checkpoint_scales(0.001, 0.002, bad).is_err());
    }
}

#[test]
fn static_scale_bypasses_runtime_absmax_and_sync_before_quantization() {
    let source = include_str!("e2m1_flashinfer.rs");
    let start = source
        .find("fn prepare_e2m1_input(")
        .expect("W4A4 activation preparation is missing");
    let end = source[start..]
        .find("fn forward_e2m1_prepared(")
        .map(|offset| start + offset)
        .expect("prepared W4A4 GEMM boundary is missing");
    let body = &source[start..end];

    let fixed = body
        .find("if let Some(scale2) = checkpoint_scale2")
        .expect("checkpoint-static branch is missing");
    let dynamic = body
        .find("ctx.gpu.synchronize(stream)?")
        .expect("dynamic absmax fallback is missing");
    let quantize = body
        .find("ops::quantize_bf16_to_nvfp4(")
        .expect("shared on-device quantizer is missing");
    assert!(fixed < dynamic && dynamic < quantize);
}

#[test]
fn full_w4a4_gate_and_up_share_one_prepared_activation() {
    let source = include_str!("prefill.rs");
    let start = source
        .find("if e2m1_fast_path {")
        .expect("full W4A4 dispatch is missing");
    let end = source[start..]
        .find("if e2m1_down_only_path {")
        .map(|offset| start + offset)
        .expect("down-only dispatch boundary is missing");
    let branch = &source[start..end];

    assert_eq!(branch.matches("self.prepare_e2m1_input(").count(), 1);
    assert_eq!(branch.matches("self.forward_e2m1_prepared(").count(), 3);
    assert_eq!(branch.matches("self.forward_e2m1_proj(").count(), 1);
    assert_eq!(branch.matches(".prepare_e2m1_silu_input(").count(), 1);
    assert!(branch.contains("&self.weights.gate_proj"));
    assert!(branch.contains("&self.weights.up_proj"));
    assert!(branch.contains("&self.weights.down_proj"));
}

#[test]
fn fused_silu_quant_preserves_bf16_boundary_and_fails_closed() {
    let cuda = include_str!("../../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
    let start = cuda
        .find("void quantize_silu_mul_bf16_to_nvfp4(")
        .expect("fused SwiGLU quantizer is missing");
    let body = &cuda[start..];
    assert!(cuda.contains("return __bfloat162float(__float2bfloat16(g * sigmoid_g * u));"));
    assert!(body.contains("float_to_fp8_e4m3(fp8_float)"));
    assert!(body.contains("quantize_e2m1(v0)"));
    assert!(body.contains("quantize_e2m1(v1)"));

    let rust = include_str!("prefill.rs");
    assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1"));
    assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"));
    assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4"));
}

#[test]
fn kmajor_transform_and_tile_scatter_reconstruct_parent_layout() {
    const N: usize = 256;
    const K: usize = 192;
    let packed_k = K / 2;
    let scale_k = K / 16;

    let packed_parent: Vec<u32> = (0..N * packed_k).map(|index| index as u32).collect();
    let scale_parent: Vec<u32> = (0..N * scale_k)
        .map(|index| 1_000_000 + index as u32)
        .collect();
    let mut packed_t = vec![0_u32; packed_parent.len()];
    let mut scale_t = vec![0_u32; scale_parent.len()];
    for column in 0..N {
        for k_byte in 0..packed_k {
            packed_t[k_byte * N + column] = packed_parent[column * packed_k + k_byte];
        }
        for group in 0..scale_k {
            scale_t[group * N + column] = scale_parent[column * scale_k + group];
        }
    }

    for n_base in [0, 128] {
        for k_base in [0, 64, 128] {
            for local_n in 0..128 {
                let column = n_base + local_n;
                for local_byte in 0..32 {
                    let k_byte = k_base / 2 + local_byte;
                    let staged = packed_t[k_byte * N + column];
                    assert_eq!(staged, packed_parent[column * packed_k + k_byte]);
                }
                for local_group in 0..4 {
                    let group = k_base / 16 + local_group;
                    let staged = scale_t[group * N + column];
                    assert_eq!(staged, scale_parent[column * scale_k + group]);
                }
            }
        }
    }
}

#[test]
fn sm103_disables_warp_nvfp4_mma_and_enables_cublaslt_nvfp4_route() {
    assert!(is_blackwell_dc_compute_capability(Some((10, 0))));
    assert!(is_blackwell_dc_compute_capability(Some((10, 3))));
    assert!(!is_blackwell_dc_compute_capability(Some((12, 0))));
    assert!(!is_blackwell_dc_compute_capability(Some((12, 1))));
    assert!(!is_blackwell_dc_compute_capability(None));

    assert!(is_sm120_compute_capability(Some((12, 0))));
    assert!(!is_sm120_compute_capability(Some((12, 1))));
    assert!(!is_sm120_compute_capability(Some((10, 0))));
    assert!(!is_sm120_compute_capability(Some((10, 3))));
    assert!(!is_sm120_compute_capability(None));

    assert!(!supports_warp_nvfp4_mma(Some((10, 3))));
    assert!(!supports_warp_nvfp4_mma(Some((10, 0))));
    assert!(supports_warp_nvfp4_mma(Some((12, 0))));
    assert!(supports_warp_nvfp4_mma(Some((12, 1))));
    assert!(supports_warp_nvfp4_mma(None));

    assert_eq!(parse_cublaslt_nvfp4_policy(None), CublasLtNvfp4Policy::Auto);
    assert_eq!(
        parse_cublaslt_nvfp4_policy(Some("1")),
        CublasLtNvfp4Policy::Enabled
    );
    assert_eq!(
        parse_cublaslt_nvfp4_policy(Some("true")),
        CublasLtNvfp4Policy::Enabled
    );
    assert_eq!(
        parse_cublaslt_nvfp4_policy(Some("0")),
        CublasLtNvfp4Policy::Disabled
    );
    assert_eq!(
        parse_cublaslt_nvfp4_policy(Some("false")),
        CublasLtNvfp4Policy::Disabled
    );

    // Auto mode engages on SM100 (B200/GB200) and SM103 (B300/GB300).
    for dc_cc in [Some((10, 0)), Some((10, 3))] {
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            dc_cc,
            FlashInferFfnAvailability::Absent,
            64,
            5120,
            17408,
            true,
            true,
        ));
    }
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((10, 3)),
        FlashInferFfnAvailability::Absent,
        128,
        5120,
        17408,
        true,
        true,
    ));
    // Auto mode engages on SM120 (RTX PRO 6000) when FlashInfer FFN is absent
    // (e.g. Qwen3.8-27B H=5120, I=17408 for M>=64), and defers when FlashInfer
    // is prepared.
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        64,
        5120,
        17408,
        true,
        true,
    ));
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Prepared,
        512,
        5120,
        17408,
        true,
        true,
    ));
    // Auto mode stays off on SM121 (GB10) and mock/None.
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 1)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        None,
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    // Explicit opt-out disables even on SM103/SM120; explicit opt-in allows SM121 testing.
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Disabled,
        Some((10, 3)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Disabled,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Enabled,
        Some((12, 1)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
    ));
    // Missing 128x4 quantizer, M < 24, or non-64-aligned shapes fall back cleanly to w4a16.
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        23,
        5120,
        17408,
        true,
        true,
    ));
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        24,
        5120,
        17408,
        true,
        true,
    ));
    assert!(cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((12, 0)),
        FlashInferFfnAvailability::Absent,
        32,
        5120,
        17408,
        true,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((10, 3)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        false,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled(
        CublasLtNvfp4Policy::Auto,
        Some((10, 3)),
        FlashInferFfnAvailability::Absent,
        512,
        5100,
        17408,
        true,
        true,
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cublaslt_nvfp4_scale_swizzle_matches_cutlass_128x4_layout() {
    use crate::weight_map::cutlass_scale_layout::{
        NVFP4_GROUP_SIZE, interleave_nvfp4_scales_128x4,
    };

    let n = 256u32;
    let k = 128u32;
    let groups = (k as usize) / NVFP4_GROUP_SIZE;
    let logical: Vec<u8> = (0..(n as usize) * groups)
        .map(|i| ((i.wrapping_mul(37) + 11) & 0xff) as u8)
        .collect();
    let via_cublaslt = swizzle_nvfp4_scales_128x4_host(&logical, n, k).unwrap();
    let via_cutlass =
        interleave_nvfp4_scales_128x4(&logical, &[n as usize, groups], NVFP4_GROUP_SIZE).unwrap();
    assert_eq!(via_cublaslt, via_cutlass);

    // Non-128-multiple N pads cleanly without error.
    let unaligned_n = 192u32;
    let logical_unaligned = vec![0x38u8; (unaligned_n as usize) * groups];
    let padded = swizzle_nvfp4_scales_128x4_host(&logical_unaligned, unaligned_n, k).unwrap();
    assert_eq!(padded.len(), 256 * groups);
}

#[test]
fn cublaslt_prefers_ptx_quantizer_and_gates_fused_silu_quantizer() {
    use super::{
        FfnActivation, cublaslt_fused_silu_quantize_scale, select_cublaslt_quantize_128x4_kernel,
    };

    let legacy = KernelHandle(11);
    let ptx = KernelHandle(22);
    let fused = KernelHandle(33);
    let missing = KernelHandle(0);

    assert_eq!(select_cublaslt_quantize_128x4_kernel(ptx, legacy).0, ptx.0);
    assert_eq!(
        select_cublaslt_quantize_128x4_kernel(missing, legacy).0,
        legacy.0
    );
    assert_eq!(select_cublaslt_quantize_128x4_kernel(ptx, missing).0, ptx.0);
    assert_eq!(
        select_cublaslt_quantize_128x4_kernel(missing, missing).0,
        missing.0
    );

    assert_eq!(
        cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, Some(0.125), fused),
        Some(0.125)
    );
    assert_eq!(
        cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, None, fused),
        None
    );
    assert_eq!(
        cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, Some(0.125), missing),
        None
    );
    assert_eq!(
        cublaslt_fused_silu_quantize_scale(FfnActivation::GeLU, Some(0.125), fused),
        None
    );

    let cuda = include_str!(
        "../../../../../kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu"
    );
    assert!(cuda.contains("quantize_bf16_to_nvfp4_atlas_128x4_ptx("));
    assert!(cuda.contains("silu_mul_quantize_nvfp4_atlas_128x4("));
    assert!(cuda.contains("cvt.rn.satfinite.e2m1x2.f32"));
}

#[test]
fn cublaslt_merged_gate_up_and_27b_proj_policies_protect_gb10_and_vram() {
    use super::{
        CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES, CublasLtMergedGateUpPolicy, CublasLtNvfp4Policy,
        Proj27bCublasLtNvfp4Policy, cublaslt_merged_gate_up_vram_ok,
        parse_cublaslt_merged_gate_up_policy, parse_proj_27b_cublaslt_nvfp4_policy,
        proj_27b_cublaslt_nvfp4_shape_ok, resolve_proj_27b_cublaslt_nvfp4,
    };

    assert_eq!(
        parse_cublaslt_merged_gate_up_policy(None),
        CublasLtMergedGateUpPolicy::Auto
    );
    assert_eq!(
        parse_cublaslt_merged_gate_up_policy(Some("0")),
        CublasLtMergedGateUpPolicy::Disabled
    );
    assert_eq!(
        parse_cublaslt_merged_gate_up_policy(Some("false")),
        CublasLtMergedGateUpPolicy::Disabled
    );
    assert_eq!(
        parse_cublaslt_merged_gate_up_policy(Some("1")),
        CublasLtMergedGateUpPolicy::Enabled
    );
    assert_eq!(
        parse_cublaslt_merged_gate_up_policy(Some("true")),
        CublasLtMergedGateUpPolicy::Enabled
    );

    let needed = 89 * 1024 * 1024;
    assert!(!cublaslt_merged_gate_up_vram_ok(None, needed));
    assert!(!cublaslt_merged_gate_up_vram_ok(
        Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed - 1),
        needed
    ));
    assert!(cublaslt_merged_gate_up_vram_ok(
        Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed),
        needed
    ));

    assert_eq!(
        parse_proj_27b_cublaslt_nvfp4_policy(None),
        Proj27bCublasLtNvfp4Policy::Auto
    );
    assert_eq!(
        parse_proj_27b_cublaslt_nvfp4_policy(Some("0")),
        Proj27bCublasLtNvfp4Policy::Disabled
    );
    assert_eq!(
        parse_proj_27b_cublaslt_nvfp4_policy(Some("1")),
        Proj27bCublasLtNvfp4Policy::Enabled
    );

    // GB10 (<= 64 SMs) stays default-OFF under Auto; explicit "1" enables.
    assert!(!resolve_proj_27b_cublaslt_nvfp4(
        Proj27bCublasLtNvfp4Policy::Auto,
        CublasLtNvfp4Policy::Auto,
        48,
        Some((12, 1))
    ));
    assert!(!resolve_proj_27b_cublaslt_nvfp4(
        Proj27bCublasLtNvfp4Policy::Auto,
        CublasLtNvfp4Policy::Enabled,
        64,
        Some((12, 1))
    ));
    assert!(resolve_proj_27b_cublaslt_nvfp4(
        Proj27bCublasLtNvfp4Policy::Enabled,
        CublasLtNvfp4Policy::Disabled,
        48,
        Some((12, 1))
    ));

    // SM120 / SM103 (> 64 SMs): auto-ON unless either gate opts out.
    for cc in [Some((12, 0)), Some((10, 3))] {
        assert!(resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Auto,
            CublasLtNvfp4Policy::Auto,
            188,
            cc
        ));
        assert!(!resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Disabled,
            CublasLtNvfp4Policy::Auto,
            188,
            cc
        ));
        assert!(!resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Auto,
            CublasLtNvfp4Policy::Disabled,
            188,
            cc
        ));
    }

    // Shape eligibility: Qwen3.8-27B SSM/Attn projections require M >= 64, N % 8 == 0, K % 64 == 0.
    assert!(proj_27b_cublaslt_nvfp4_shape_ok(64, 16_384, 5_120));
    assert!(proj_27b_cublaslt_nvfp4_shape_ok(7_942, 12_288, 5_120));
    assert!(proj_27b_cublaslt_nvfp4_shape_ok(8_192, 1_024, 5_120));
    assert!(proj_27b_cublaslt_nvfp4_shape_ok(8_192, 5_120, 6_144));
    assert!(!proj_27b_cublaslt_nvfp4_shape_ok(63, 16_384, 5_120));
    assert!(!proj_27b_cublaslt_nvfp4_shape_ok(8_192, 1_020, 5_120));
    assert!(!proj_27b_cublaslt_nvfp4_shape_ok(8_192, 5_120, 5_100));
}

#[test]
fn gb10_fast_prefill_enables_sm121_cublaslt_while_preserving_defaults_and_opt_out() {
    use super::{
        CublasLtNvfp4Policy, FlashInferFfnAvailability, Proj27bCublasLtNvfp4Policy,
        cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill,
        resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill,
    };

    // SM12.1 FFN stays OFF by default when ATLAS_GB10_FAST_PREFILL is off.
    assert!(!cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
        CublasLtNvfp4Policy::Auto,
        Some((12, 1)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
        false,
    ));
    // Under ATLAS_GB10_FAST_PREFILL=1, Auto enables SM12.1 when FlashInfer FFN is absent.
    assert!(cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
        CublasLtNvfp4Policy::Auto,
        Some((12, 1)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
        true,
    ));
    // FlashInfer FFN still takes precedence when prepared, and explicit =0 always disables.
    assert!(!cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
        CublasLtNvfp4Policy::Auto,
        Some((12, 1)),
        FlashInferFfnAvailability::Prepared,
        512,
        5120,
        17408,
        true,
        true,
        true,
    ));
    assert!(!cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
        CublasLtNvfp4Policy::Disabled,
        Some((12, 1)),
        FlashInferFfnAvailability::Absent,
        512,
        5120,
        17408,
        true,
        true,
        true,
    ));

    // SM12.1 (<= 64 SMs) SSM/Attn projections: OFF by default, ON under ATLAS_GB10_FAST_PREFILL=1,
    // and OFF when either projection or FFN policy explicitly opts out.
    assert!(!resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        Proj27bCublasLtNvfp4Policy::Auto,
        CublasLtNvfp4Policy::Auto,
        48,
        Some((12, 1)),
        false,
    ));
    assert!(resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        Proj27bCublasLtNvfp4Policy::Auto,
        CublasLtNvfp4Policy::Auto,
        48,
        Some((12, 1)),
        true,
    ));
    assert!(!resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        Proj27bCublasLtNvfp4Policy::Disabled,
        CublasLtNvfp4Policy::Auto,
        48,
        Some((12, 1)),
        true,
    ));
    assert!(!resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        Proj27bCublasLtNvfp4Policy::Auto,
        CublasLtNvfp4Policy::Disabled,
        48,
        Some((12, 1)),
        true,
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn merged_128x4_scales_equal_concatenated_halves_for_128_aligned_inter() {
    let inter = 256u32;
    let h = 128u32;
    let groups = (h / spark_runtime::cublaslt::NVFP4_BLOCK_SIZE) as usize;
    let gate_logical: Vec<u8> = (0..(inter as usize) * groups)
        .map(|i| ((i * 19 + 3) & 0xff) as u8)
        .collect();
    let up_logical: Vec<u8> = (0..(inter as usize) * groups)
        .map(|i| ((i * 41 + 7) & 0xff) as u8)
        .collect();
    let mut combined_logical = Vec::with_capacity(gate_logical.len() + up_logical.len());
    combined_logical.extend_from_slice(&gate_logical);
    combined_logical.extend_from_slice(&up_logical);

    let gate_s = swizzle_nvfp4_scales_128x4_host(&gate_logical, inter, h).unwrap();
    let up_s = swizzle_nvfp4_scales_128x4_host(&up_logical, inter, h).unwrap();
    let merged_s = swizzle_nvfp4_scales_128x4_host(&combined_logical, 2 * inter, h).unwrap();

    assert_eq!(&merged_s[..gate_s.len()], gate_s.as_slice());
    assert_eq!(&merged_s[gate_s.len()..], up_s.as_slice());
}

#[test]
fn sm120_kgamma_auto_tc_unlocks_m4_to_256_and_preserves_gb10_defaults() {
    use super::{
        FfnTcPolicy, parse_ffn_tc_policy, resolve_ffn_tc_for_rows, resolve_kgamma_sm_auto_flag,
    };

    assert_eq!(parse_ffn_tc_policy(None), FfnTcPolicy::Auto);
    assert_eq!(parse_ffn_tc_policy(Some("0")), FfnTcPolicy::Disabled);
    assert_eq!(parse_ffn_tc_policy(Some("false")), FfnTcPolicy::Disabled);
    assert_eq!(parse_ffn_tc_policy(Some("1")), FfnTcPolicy::Enabled);
    assert_eq!(parse_ffn_tc_policy(Some("true")), FfnTcPolicy::Enabled);

    // GB10 (<= 64 SMs): Auto keeps exact scalar GEMV and all opt-in flags OFF.
    for sm in [48u32, 64u32] {
        for rows in [1u32, 2, 3, 4, 8, 16, 32, 33, 64, 128, 256] {
            assert!(
                !resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, rows),
                "GB10 sm={sm} rows={rows} must stay exact under Auto"
            );
            assert!(
                resolve_ffn_tc_for_rows(FfnTcPolicy::Enabled, sm, rows),
                "explicit ATLAS_FFN_TC=1 must override on sm={sm}"
            );
        }
        assert!(!resolve_kgamma_sm_auto_flag(None, sm));
        assert!(resolve_kgamma_sm_auto_flag(Some("1"), sm));
        assert!(!resolve_kgamma_sm_auto_flag(Some("0"), sm));
    }

    // SM120 / SM103 (> 64 SMs): Auto enables TC batched kgamma for rows >= 4
    // (including M=4, 8, 16, 32, 64, 128, 256) while keeping rows < 4 on exact GEMV,
    // and honors explicit "0"/"false" opt-out for strict 0-ULP parity.
    for sm in [128u32, 188u32] {
        for small in [1u32, 2, 3] {
            assert!(
                !resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, small),
                "sm={sm} rows={small} should keep batched GEMV"
            );
        }
        for batched in [4u32, 8, 16, 32, 33, 64, 128, 256] {
            assert!(
                resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, batched),
                "sm={sm} rows={batched} must auto-enable TC batched kgamma"
            );
            assert!(
                !resolve_ffn_tc_for_rows(FfnTcPolicy::Disabled, sm, batched),
                "sm={sm} rows={batched} must honor ATLAS_FFN_TC=0 opt-out"
            );
        }
        assert!(resolve_kgamma_sm_auto_flag(None, sm));
        assert!(!resolve_kgamma_sm_auto_flag(Some("0"), sm));
        assert!(!resolve_kgamma_sm_auto_flag(Some("false"), sm));
        assert!(resolve_kgamma_sm_auto_flag(Some("1"), sm));
    }
}

#[test]
fn sm120_ffn_strip_reserve_and_kgamma_cublaslt_policies_keep_gb10_zero_regression() {
    use super::{
        CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT, CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES,
        CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES, CublasLtMergedGateUpPolicy,
        cublaslt_merged_gate_up_allow_materialize, cublaslt_merged_gate_up_vram_ok_for_sm,
        parse_ffn_prefill_l2_strip, resolve_kgamma_cublaslt_opt_in, resolve_sm_auto_opt_in,
    };

    let gib = 1024 * 1024 * 1024usize;
    let needed = 89 * 1024 * 1024usize; // 27B single-layer merged gate+up

    // GB10 (<= 64 SMs): 12 GiB reserve, Auto stays zero-copy, L2 strip OFF,
    // kgamma cuBLASLt OFF, fused absmax/norm OFF.
    for sm in [48u32, 64u32] {
        assert!(!cublaslt_merged_gate_up_allow_materialize(
            CublasLtMergedGateUpPolicy::Auto,
            sm
        ));
        assert!(cublaslt_merged_gate_up_allow_materialize(
            CublasLtMergedGateUpPolicy::Enabled,
            sm
        ));
        assert!(!cublaslt_merged_gate_up_vram_ok_for_sm(
            Some(11 * gib + 600 * 1024 * 1024),
            needed,
            sm
        ));
        assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
            Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed),
            needed,
            sm
        ));
        assert_eq!(parse_ffn_prefill_l2_strip(None, sm), None);
        assert_eq!(
            parse_ffn_prefill_l2_strip(Some("1"), sm),
            Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
        );
        assert_eq!(parse_ffn_prefill_l2_strip(Some("512"), sm), Some(512));
        assert!(!resolve_kgamma_cublaslt_opt_in(None, sm));
        assert!(resolve_kgamma_cublaslt_opt_in(Some("1"), sm));
        assert!(!resolve_sm_auto_opt_in(None, sm));
    }

    // SM120 / SM103 (> 64 SMs): 4 GiB reserve (C=64 with 11.6 GiB free fits!),
    // Auto allows materialization, L2 strip=1024, kgamma cuBLASLt ON, fused absmax/norm ON.
    for sm in [65u32, 188u32] {
        assert!(cublaslt_merged_gate_up_allow_materialize(
            CublasLtMergedGateUpPolicy::Auto,
            sm
        ));
        assert!(!cublaslt_merged_gate_up_allow_materialize(
            CublasLtMergedGateUpPolicy::Disabled,
            sm
        ));
        // 11.6 GiB free (C=64 with --kv-pool-tokens 131072) passes!
        assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
            Some(11 * gib + 600 * 1024 * 1024),
            needed,
            sm
        ));
        assert!(!cublaslt_merged_gate_up_vram_ok_for_sm(
            Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES + needed - 1),
            needed,
            sm
        ));
        assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
            Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES + needed),
            needed,
            sm
        ));
        assert_eq!(
            parse_ffn_prefill_l2_strip(None, sm),
            Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
        );
        assert_eq!(parse_ffn_prefill_l2_strip(Some("0"), sm), None);
        assert_eq!(parse_ffn_prefill_l2_strip(Some("false"), sm), None);
        assert_eq!(parse_ffn_prefill_l2_strip(Some("2048"), sm), Some(2048));
        assert!(resolve_kgamma_cublaslt_opt_in(None, sm));
        assert!(!resolve_kgamma_cublaslt_opt_in(Some("0"), sm));
        assert!(resolve_sm_auto_opt_in(None, sm));
        assert!(!resolve_sm_auto_opt_in(Some("0"), sm));
    }
}
