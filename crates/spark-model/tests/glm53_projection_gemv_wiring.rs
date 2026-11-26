// SPDX-License-Identifier: AGPL-3.0-only
//! GPU launch seams supplement the behavioral plan; they are not GPU evidence.
fn source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/glm53_projection_probe/gemv.rs"
    ))
    .unwrap()
}
fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn resolves_real_registered_symbols_and_rejects_null_handles() {
    let raw = source();
    let source = compact(&raw);
    for lookup in [
        "gpu.kernel(\"gemv\",\"dense_gemv_bf16\")?",
        "gpu.kernel(\"gemv\",\"dense_gemv_bf16_batchn\")?",
        "gpu.kernel(\"dense_gemv_bf16_batchm\",\"dense_gemv_bf16_batchm\")?",
    ] {
        assert!(
            source.contains(lookup),
            "missing registered lookup {lookup}"
        );
    }
    assert!(source.contains("validate_handles("));
    let launch = raw.find("pub fn launch(").unwrap();
    assert!(
        !raw[launch..].contains("gpu.kernel("),
        "lookup inside projection"
    );
}

#[test]
fn batched_gemv_launches_use_registered_batchn_and_batchm_geometry() {
    let source = compact(&source());
    assert!(source.contains("ops::dense_gemv_batchn("));
    assert!(source.contains(".grid([OUTPUTS.div_ceil(4),1,1])"));
    assert!(source.contains(".block([256,1,1])"));
    assert!(source.contains(
        ".arg_ptr(DevicePtr(input.ptr)).arg_ptr(DevicePtr(plan.weight().ptr)).arg_ptr(DevicePtr(output.ptr)).arg_u32(plan.rows()).arg_u32(OUTPUTS).arg_u32(HIDDEN).arg_u32(HIDDEN).arg_u32(OUTPUTS)"
    ));
    assert!(!source.contains("lora_bgmv"));
    assert!(!source.contains("dense_gemv_bf16_batch2"));
    assert!(!source.contains("expand_fold"));
    assert!(!source.contains("apply_lora"));
}

#[test]
fn dual_row_and_single_row_operators_remain_explicit_new_family() {
    let source = source();
    assert!(source.contains("ops::dense_gemv("));
    assert!(source.contains("ops::dense_gemv_batchn("));
    assert!(compact(&source).contains("GemvMode::Batch2ifplan.rows()==2"));
    assert!(source.contains("batch2-with-m1-gemv"));
    for forbidden in [
        "cublas",
        "dense_gemm_tc",
        "copy_h2d",
        "copy_d2h",
        "gpu.alloc",
        "unsafe",
        "std::fs",
    ] {
        assert!(
            !source.contains(forbidden),
            "new GEMV helper contains {forbidden}"
        );
    }
}
