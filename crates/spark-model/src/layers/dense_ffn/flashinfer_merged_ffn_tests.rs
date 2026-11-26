// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    FlashinferFfnPrefillRoute as Route, e2m1_scratch_layout, flashinfer_ffn_prefill_route,
};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::{
    ensure_flashinfer_ffn_disjoint, flashinfer_ffn_activation_scale_extent, flashinfer_ffn_extent,
};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use spark_runtime::gpu::{DevicePtr, KernelHandle, mock::MockGpuBackend};

#[test]
fn explicit_route_is_exact_and_missing_is_fail_closed() {
    // `m_qualified` is now supplied by the caller (see the fn doc): the
    // frozen pair 2079/8192 qualifies, 2080 does not.
    let q = |m: u32| matches!(m, 2_079 | 8_192);
    assert_eq!(
        flashinfer_ffn_prefill_route(false, true, q(2_079), false),
        Route::Disabled
    );
    assert_eq!(
        flashinfer_ffn_prefill_route(true, false, q(2_079), true),
        Route::Ineligible
    );
    assert_eq!(
        flashinfer_ffn_prefill_route(true, true, q(2_080), false),
        Route::Ineligible
    );
    for m in [2_079u32, 8_192] {
        assert_eq!(
            flashinfer_ffn_prefill_route(true, true, q(m), false),
            Route::Missing
        );
        assert_eq!(
            flashinfer_ffn_prefill_route(true, true, q(m), true),
            Route::Complete
        );
    }
}

/// The extended ladder must stay OFF by default: with the env unset, an M
/// that is only in `QWEN38_FFN_EXTRA_M` must still be rejected, so default
/// behaviour is bit-for-bit the frozen table.
#[test]
fn extended_m_ladder_is_off_unless_requested() {
    use crate::weight_map::flashinfer_ffn_admission::{
        Qwen38FfnOperation, qwen38_ffn_extra_m_enabled, qwen38_ffn_launch_candidate,
    };
    if qwen38_ffn_extra_m_enabled() {
        return; // caller opted in; the negative assertion does not apply
    }
    for m in [1_024usize, 2_048, 4_096, 6_144, 7_168, 8_187] {
        assert!(
            qwen38_ffn_launch_candidate(Qwen38FfnOperation::MergedGateUp, m).is_err(),
            "M={m} must be unqualified while ATLAS_FLASHINFER_FFN_EXTRA_M is unset"
        );
    }
}

#[test]
fn construction_scratch_layout_matches_the_charged_maximum() {
    let layout = e2m1_scratch_layout(8_192, 17_408).unwrap();
    assert_eq!(layout.cap_m, 8_192);
    assert_eq!(layout.cap_k, 17_408);
    assert_eq!(layout.packed_bytes, 71_303_168);
    assert_eq!(layout.scale_bytes, 8_912_896);
    assert_eq!(layout.max_bytes, 4);
    assert_eq!(
        layout.packed_bytes + layout.scale_bytes + layout.max_bytes,
        80_216_068
    );
    assert!(e2m1_scratch_layout(usize::MAX, 17_408).is_err());
    assert!(e2m1_scratch_layout(8_192, usize::MAX - 15).is_err());
}

#[test]
fn production_order_preflights_then_merges_directly_quantizes_and_completes_down() {
    let source = include_str!("e2m1_flashinfer.rs");
    let start = source
        .find("fn try_forward_flashinfer_prefill(")
        .expect("FlashInfer FFN runtime route");
    let body = &source[start..];

    let missing = body.find("FlashinferFfnPrefillRoute::Missing").unwrap();
    let down_preflight = body.find("let mut down_launch").unwrap();
    let scratch_guard = body.find("self.lock_e2m1_use(stream)").unwrap();
    let physical_scale_extent = body
        .find("flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)")
        .unwrap();
    let quantize = body
        .find("ops::quantize_bf16_to_nvfp4_atlas_128x4(")
        .unwrap();
    let merged = body.find("merged_launch.launch_eager(").unwrap();
    let direct_silu_quantize = body
        .find("ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
        .unwrap();
    let down = body.find("down_launch.launch_eager(").unwrap();
    let receipt = body.find("log_flashinfer_ffn_success(").unwrap();
    assert!(
        missing < physical_scale_extent
            && physical_scale_extent < down_preflight
            && down_preflight < scratch_guard
            && scratch_guard < quantize
    );
    assert!(quantize < merged && merged < direct_silu_quantize && direct_silu_quantize < down);
    assert!(down < receipt);
    assert!(!body.contains("ops::silu_mul("));
    assert!(!body.contains("KernelLaunch::new(ctx.gpu, route.split_gate_up_k)"));
    assert!(!body.contains("ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(body.contains("f32::from_bits(route.down.input_scale_bits)"));
    assert!(body.contains("ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes"));
    assert!(body.contains("ctx.buffers.ffn_gate_up_bf16()"));
    assert!(!body.contains("Qwen38FfnOperation::Gate, m as usize"));
    assert!(!body.contains("Qwen38FfnOperation::Up, m as usize"));
    assert!(!body.contains("route.stream"));
    assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)"));
    assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)"));
}

#[test]
fn construction_stream_is_preflight_only_and_runtime_owns_launch_stream() {
    let source = include_str!("e2m1_flashinfer.rs");
    let struct_start = source.find("struct FlashinferFfnPrefill {").unwrap();
    let struct_end = source[struct_start..]
        .find("\n}")
        .map(|offset| struct_start + offset)
        .unwrap();
    assert!(!source[struct_start..struct_end].contains("stream: u64"));

    let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
    let runtime_start = source.find("fn try_forward_flashinfer_prefill(").unwrap();
    let prepare = &source[prepare_start..runtime_start];
    assert!(prepare.contains("let stream = gpu.default_stream();"));
    assert!(!prepare.contains("stream,\n        });"));

    let runtime = &source[runtime_start..];
    assert!(!runtime.contains("stream == route.stream"));
    assert!(runtime.contains("self.lock_e2m1_use(stream)"));

    let lock_start = source.find("fn lock_e2m1_use(").unwrap();
    let lock_end = source[lock_start..]
        .find("/// Quantize one BF16 activation matrix")
        .map(|offset| lock_start + offset)
        .unwrap();
    let lock = &source[lock_start..lock_end];
    assert!(lock.contains(".e2m1_use_stream"));
    assert!(lock.contains(".lock()"));
    assert!(lock.contains(".map_err("));
    assert!(!lock.contains(".unwrap()"));
}

#[test]
fn split_kernel_is_exact_row_major_gate_then_up() {
    let cuda = include_str!("../../../../../kernels/gb10/common/flashinfer_projection_split.cu");
    let start = cuda
        .find("void flashinfer_projection_split_ffn_gate_up(")
        .expect("merged FFN split kernel");
    let body = &cuda[start..];
    assert!(body.contains("constexpr uint32_t INTERMEDIATE = 17408"));
    assert!(body.contains("constexpr uint32_t TOTAL = 2 * INTERMEDIATE"));
    assert!(body.contains("merged + row * TOTAL"));
    assert!(body.contains("gate_dst[vector] = src[vector]"));
    assert!(body.contains("up_dst[vector] = src[VECTORS + vector]"));
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn physical_activation_scale_extent_is_exact_for_qualified_rows() {
    assert_eq!(
        flashinfer_ffn_activation_scale_extent(2_079, 17_408).unwrap(),
        2_367_488
    );
    assert_eq!(
        flashinfer_ffn_activation_scale_extent(8_192, 17_408).unwrap(),
        8_912_896
    );
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn physical_activation_scale_extent_rejects_padding_overflow() {
    assert!(flashinfer_ffn_activation_scale_extent(usize::MAX, 17_408).is_err());
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn physical_tail_overlap_is_rejected_before_launch() {
    let rows = 2_079;
    let cols = 17_408;
    let logical_bytes = flashinfer_ffn_extent(rows, cols / 16, 1).unwrap();
    let physical_bytes = flashinfer_ffn_activation_scale_extent(rows, cols).unwrap();
    assert_eq!(physical_bytes - logical_bytes, 105_536);

    let scales = DevicePtr(0x1_0000);
    let destination = DevicePtr(scales.0 + u64::try_from(logical_bytes).unwrap());
    assert!(
        ensure_flashinfer_ffn_disjoint(scales, logical_bytes, destination, 16).is_ok(),
        "the hostile destination begins exactly after the logical prefix"
    );
    assert!(
        ensure_flashinfer_ffn_disjoint(scales, physical_bytes, destination, 16).is_err(),
        "the padded physical tail must reject the same destination before launch"
    );
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn fused_physical_wrapper_accepts_exact_rows_and_rejects_hostile_inputs_before_launch() {
    let gpu = MockGpuBackend::new();
    let gate = DevicePtr(0x0010_0000);
    let up = DevicePtr(0x1000_0000);
    let packed = DevicePtr(0x2000_0000);
    let scales = DevicePtr(0x3000_0000);
    for rows in [2_079, 8_192] {
        crate::layers::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            KernelHandle(0x55),
            gate,
            up,
            packed,
            scales,
            0.25,
            rows,
            17_408,
            0x77,
        )
        .unwrap();
    }
    assert_eq!(gpu.launch_count(), 2);

    for (kernel, gate, up, packed, scales, scale2, rows, cols) in [
        (
            KernelHandle(0),
            gate,
            up,
            packed,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            DevicePtr::NULL,
            up,
            packed,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            gate,
            DevicePtr::NULL,
            packed,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            gate,
            up,
            DevicePtr::NULL,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            gate,
            up,
            packed,
            DevicePtr::NULL,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            gate,
            up,
            packed,
            scales,
            f32::NAN,
            2_079,
            17_408,
        ),
        (KernelHandle(1), gate, up, packed, scales, 0.25, 0, 17_408),
        (KernelHandle(1), gate, up, packed, scales, 0.25, 2_079, 16),
        (
            KernelHandle(1),
            gate,
            up,
            packed,
            scales,
            0.25,
            u32::MAX,
            17_408,
        ),
    ] {
        assert!(
            crate::layers::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu, kernel, gate, up, packed, scales, scale2, rows, cols, 0x77,
            )
            .is_err()
        );
    }
    assert_eq!(gpu.launch_count(), 2);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn direct_merged_wrapper_is_abi_separate_and_fails_before_launch() {
    let gpu = MockGpuBackend::new();
    let merged = DevicePtr(0x0010_0000);
    let packed = DevicePtr(0x2000_0000);
    let scales = DevicePtr(0x3000_0000);
    for rows in [2_079, 8_192] {
        crate::layers::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            KernelHandle(0x66),
            merged,
            packed,
            scales,
            0.25,
            rows,
            17_408,
            0x77,
        )
        .unwrap();
    }
    assert_eq!(gpu.launch_count(), 2);

    for (kernel, merged, packed, scales, scale2, rows, cols) in [
        (KernelHandle(0), merged, packed, scales, 0.25, 2_079, 17_408),
        (
            KernelHandle(1),
            DevicePtr::NULL,
            packed,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            merged,
            DevicePtr::NULL,
            scales,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            merged,
            packed,
            DevicePtr::NULL,
            0.25,
            2_079,
            17_408,
        ),
        (
            KernelHandle(1),
            merged,
            packed,
            scales,
            f32::NAN,
            2_079,
            17_408,
        ),
        (KernelHandle(1), merged, packed, scales, 0.25, 0, 17_408),
        (KernelHandle(1), merged, packed, scales, 0.25, 2_079, 16),
        (
            KernelHandle(1),
            merged,
            packed,
            scales,
            0.25,
            u32::MAX,
            17_408,
        ),
    ] {
        assert!(
            crate::layers::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu, kernel, merged, packed, scales, scale2, rows, cols, 0x77,
            )
            .is_err()
        );
    }
    assert_eq!(gpu.launch_count(), 2);
}

#[test]
fn fused_physical_symbol_layout_abi_handle_and_receipt_are_source_bound() {
    let cuda = include_str!("../../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
    let logical = cuda
        .find("void quantize_silu_mul_bf16_to_nvfp4(")
        .expect("existing logical W4A4 fused symbol");
    let physical = cuda
        .find("void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(")
        .expect("ABI-separated physical fused symbol");
    assert!(logical < physical);
    let physical_body = &cuda[physical..];
    assert!(physical_body.contains("((N + 127u) / 128u) * 128u"));
    assert!(physical_body.contains("if (row >= N)"));
    assert!(physical_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));
    assert!(physical_body.contains("silu_mul_bf16_round(gate, up"));
    assert!(physical_body.contains("float_to_fp8_e4m3(fp8_float)"));
    assert!(physical_body.contains("quantize_e2m1(v0)"));

    let direct = cuda
        .find("void quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
        .expect("ABI-separated direct merged physical symbol");
    assert!(physical < direct);
    let direct_body = &cuda[direct..];
    assert!(direct_body.contains("static_cast<unsigned long long>(row) * (2ull * K)"));
    assert!(direct_body.contains("const __nv_bfloat16* up = gate + K"));
    assert!(direct_body.contains("float rounded[GROUP_SIZE]"));
    assert_eq!(
        direct_body.matches("silu_mul_bf16_round(gate, up").count(),
        1
    );
    assert!(direct_body.contains("rounded[i] = value"));
    assert!(direct_body.contains("quantize_e2m1(rounded[i] * inv_eff)"));
    assert!(direct_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));

    let wrapper_source = include_str!("../ops/gemm_dense.rs");
    let wrapper_start = wrapper_source
        .find("pub fn quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
        .expect("direct merged physical Rust wrapper");
    let wrapper_end = wrapper_source[wrapper_start..]
        .find("/// Native NVFP4")
        .map(|offset| wrapper_start + offset)
        .expect("physical fused Rust wrapper boundary");
    let wrapper = &wrapper_source[wrapper_start..wrapper_end];
    let merged = wrapper.find(".arg_ptr(merged_gate_up)").unwrap();
    let packed = wrapper.find(".arg_ptr(packed_out)").unwrap();
    let scales = wrapper.find(".arg_ptr(scale_out)").unwrap();
    let scale2 = wrapper.find(".arg_f32(scale2)").unwrap();
    let rows = wrapper.find(".arg_u32(rows)").unwrap();
    let cols = wrapper.find(".arg_u32(cols)").unwrap();
    assert!(merged < packed && packed < scales && scales < scale2);
    assert!(scale2 < rows && rows < cols);
    assert!(wrapper.contains("cols >= 64 && cols.is_multiple_of(64)"));
    assert!(wrapper.contains(".checked_add(127)"));
    assert!(wrapper.contains(".grid([quantize_128x4_grid_x(padded_rows), 1, 1])"));

    let source = include_str!("e2m1_flashinfer.rs");
    let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
    let prepare_end = source[prepare_start..]
        .find("fn try_forward_flashinfer_prefill(")
        .map(|offset| prepare_start + offset)
        .unwrap();
    let prepare = &source[prepare_start..prepare_end];
    let symbol = prepare
        .find("\"quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4\"")
        .unwrap();
    let nonzero = prepare
        .find("quantize_merged_silu_atlas_128x4_k.0 != 0")
        .unwrap();
    let plan_freeze = prepare.find("for (operation, m) in [").unwrap();
    let scratch_preallocation = prepare
        .find("e2m1_scratch_layout(8_192, intermediate)")
        .unwrap();
    let scratch_capacity_check = prepare
        .find("retained.cap_m == scratch_layout.cap_m")
        .unwrap();
    let retained_copy = prepare.find("build_flashinfer_merged_gate_up(").unwrap();
    let route_retention = prepare.find("self.flashinfer_ffn_prefill = Some(").unwrap();
    assert!(
        symbol < nonzero
            && nonzero < plan_freeze
            && plan_freeze < scratch_preallocation
            && scratch_preallocation < scratch_capacity_check
            && scratch_capacity_check < retained_copy
            && retained_copy < route_retention
    );
    assert!(prepare.contains("\"quantize_nvfp4\""));
    assert!(prepare.contains("quantize_merged_silu_atlas_128x4_k,"));
    assert!(!prepare.contains("\"flashinfer_projection_split\""));
}
