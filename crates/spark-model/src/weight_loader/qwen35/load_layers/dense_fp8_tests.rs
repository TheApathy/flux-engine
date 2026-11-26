// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{
    Ledger, Module, complete_receipt, layer_index, mxfp8_proj, receipt_line, validate_mxfp8_pair,
};

#[test]
fn layer_index_reads_the_trailing_component() {
    assert_eq!(layer_index("model.language_model.layers.17").unwrap(), 17);
    assert_eq!(layer_index("model.layers.0").unwrap(), 0);
    assert!(layer_index("model.layers.x").is_err());
    assert!(layer_index("").is_err());
}

#[test]
fn ledger_accumulates_bytes_and_layers_per_module_and_drains_on_take() {
    let ledger = Ledger::new();
    ledger.account(Module::Gdn, 100);
    ledger.account(Module::Gdn, 50);
    ledger.account(Module::Shared, 7);
    let snap = ledger.take();
    assert_eq!(snap[0], (Module::Gdn, 2, 150));
    assert_eq!(snap[1], (Module::Attn, 0, 0));
    assert_eq!(snap[2], (Module::Shared, 1, 7));

    // A second drain starts clean so reloads cannot mask missing module hooks.
    let drained = ledger.take();
    assert_eq!(drained[0], (Module::Gdn, 0, 0));
    assert_eq!(drained[1], (Module::Attn, 0, 0));
    assert_eq!(drained[2], (Module::Shared, 0, 0));
}

#[test]
fn receipt_names_scope_modules_and_extra_bytes() {
    let entries = [
        (Module::Gdn, 36, 2_000),
        (Module::Attn, 12, 300),
        (Module::Shared, 48, 40),
    ];
    let line = complete_receipt(&entries).unwrap();
    assert!(line.starts_with(
        "QWEN4_DENSE_FP8 engaged scope=decode-only modules=gdn,attn,shared extra_resident_bytes=0 aliased_checkpoint_bytes=2340 "
    ), "{line}");
    assert!(line.contains("gdn_layers=36 gdn_bytes=2000"), "{line}");
    assert!(
        line.contains("kernel=dense_gemv_mxfp8 weights=native_mxfp8_block32 requant=none"),
        "{line}"
    );
    assert!(line.contains("prefill=nvfp4_unchanged"), "{line}");
}

#[test]
fn receipt_refuses_an_engaged_pass_that_built_a_module_nowhere() {
    let entries = [
        (Module::Gdn, 36, 2_000),
        (Module::Attn, 0, 0),
        (Module::Shared, 48, 40),
    ];
    assert!(complete_receipt(&entries).is_err());
    assert!(receipt_line(&entries).contains("modules=gdn,shared "));
}

#[test]
fn undeclared_fp8_tensor_is_refused_before_any_device_work() {
    let gpu = MockGpuBackend::new();
    let name = "model.layers.0.linear_attn.out_proj.weight";
    let store = WeightStore::from_map(HashMap::from([(
        name.to_string(),
        WeightTensor {
            ptr: gpu.alloc(64 * 32).unwrap(),
            shape: vec![64, 32],
            dtype: WeightDtype::FP8E4M3,
        },
    )]));
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let before = gpu.alloc_count();
    let err = mxfp8_proj(
        &store,
        &config,
        &gpu,
        "model.layers.0.linear_attn.out_proj",
        [64, 32],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("not declared MXFP8"), "{err}");
    assert_eq!(gpu.alloc_count(), before);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn mxfp8_pair_validation_fails_closed_on_dtype_and_shape() {
    use WeightDtype::{BF16, FP8E4M3, UInt8};
    let ok = |w: (WeightDtype, &[usize]), s: (WeightDtype, &[usize]), e| {
        validate_mxfp8_pair("p", w, s, e)
    };
    assert_eq!(
        ok((FP8E4M3, &[64, 128]), (UInt8, &[64, 4]), Some([64, 128])).unwrap(),
        [64, 128]
    );
    assert_eq!(
        ok((FP8E4M3, &[64, 128]), (UInt8, &[64, 4]), None).unwrap(),
        [64, 128]
    );
    let refused = [
        ok((BF16, &[64, 128]), (UInt8, &[64, 4]), None),
        ok((UInt8, &[64, 128]), (UInt8, &[64, 4]), None),
        ok((FP8E4M3, &[64, 128, 1]), (UInt8, &[64, 4]), None),
        ok((FP8E4M3, &[64, 128]), (UInt8, &[64, 4]), Some([128, 64])),
        ok((FP8E4M3, &[64, 96 + 16]), (UInt8, &[64, 3]), None),
        ok((FP8E4M3, &[64, 128]), (FP8E4M3, &[64, 4]), None),
        ok((FP8E4M3, &[64, 128]), (UInt8, &[64, 1]), None),
        ok((FP8E4M3, &[64, 128]), (UInt8, &[4, 64]), None),
    ];
    for (i, r) in refused.into_iter().enumerate() {
        let err = r.unwrap_err().to_string();
        assert!(err.contains("refused"), "case {i}: {err}");
    }
}
