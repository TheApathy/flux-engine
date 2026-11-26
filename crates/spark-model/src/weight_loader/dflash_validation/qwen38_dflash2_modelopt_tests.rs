// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::weights::WeightDtype;

use super::qwen38_dflash2_tests::{FakeStore, official_config, official_store};
use super::{DflashProjectionFormat, validate_dflash_metadata, validate_dflash_metadata_admission};
use crate::weight_loader::dflash_loader::{
    DflashConfig, DflashQuantizationConfig, DflashQuantizationProducer,
};

fn modelopt_config() -> DflashConfig {
    let mut config = official_config();
    config.quantization_config = Some(DflashQuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "NVFP4".into(),
        group_size: 16,
        producer: Some(DflashQuantizationProducer {
            name: "modelopt".into(),
            version: "dflash2-nvfp4-rtn-calibrated-1.0".into(),
        }),
    });
    config
}

fn modelopt_store(prefix: &str) -> FakeStore {
    let mut store = official_store(prefix);
    for layer in 0..5 {
        let lp = format!("{prefix}layers.{layer}");
        for (stem, n, k) in [
            ("self_attn.q_proj", 4096, 5120),
            ("self_attn.k_proj", 1024, 5120),
            ("self_attn.v_proj", 1024, 5120),
            ("self_attn.o_proj", 5120, 4096),
            ("mlp.gate_proj", 17408, 5120),
            ("mlp.up_proj", 17408, 5120),
            ("mlp.down_proj", 5120, 17408),
        ] {
            let stem = format!("{lp}.{stem}");
            store.0.remove(&format!("{stem}.weight"));
            for (suffix, shape, dtype) in [
                ("weight", vec![n, k / 2], WeightDtype::UInt8),
                ("weight_scale", vec![n, k / 16], WeightDtype::FP8E4M3),
                ("weight_scale_2", vec![], WeightDtype::FP32),
                ("input_scale", vec![], WeightDtype::FP32),
            ] {
                assert!(
                    store
                        .0
                        .insert(format!("{stem}.{suffix}"), (shape, dtype))
                        .is_none()
                );
            }
        }
    }
    assert_eq!(store.0.len(), 186);
    store
}

#[test]
fn accepts_and_classifies_only_the_exact_186_tensor_modelopt_profile() {
    for prefix in ["", "model."] {
        let store = modelopt_store(prefix);
        let admission = validate_dflash_metadata_admission(&store, &modelopt_config())
            .unwrap()
            .expect("DFlash admission");
        assert_eq!(admission.prefix, prefix);
        assert_eq!(
            admission.projection_format,
            DflashProjectionFormat::ModelOptNvfp4
        );
    }
}

#[test]
fn every_modelopt_tensor_name_shape_and_dtype_is_load_bearing() {
    let names = modelopt_store("").0.into_keys().collect::<Vec<_>>();
    assert_eq!(names.len(), 186);
    for name in names {
        let mut missing = modelopt_store("");
        missing.0.remove(&name);
        assert!(validate_dflash_metadata(&missing, &modelopt_config()).is_err());

        let mut wrong_shape = modelopt_store("");
        wrong_shape.0.get_mut(&name).unwrap().0.push(1);
        assert!(validate_dflash_metadata(&wrong_shape, &modelopt_config()).is_err());

        let mut wrong_dtype = modelopt_store("");
        let entry = wrong_dtype.0.get_mut(&name).unwrap();
        entry.1 = if entry.1 == WeightDtype::BF16 {
            WeightDtype::FP32
        } else {
            WeightDtype::BF16
        };
        assert!(validate_dflash_metadata(&wrong_dtype, &modelopt_config()).is_err());
    }
}

#[test]
fn modelopt_profile_rejects_quantization_metadata_drift() {
    let mutations: [fn(&mut DflashQuantizationConfig); 3] = [
        |q| q.quant_method = "other".into(),
        |q| q.quant_algo = "INT8".into(),
        |q| q.group_size = 32,
    ];
    for mutate in mutations {
        let mut config = modelopt_config();
        mutate(config.quantization_config.as_mut().unwrap());
        assert!(validate_dflash_metadata(&modelopt_store(""), &config).is_err());
    }

    let mut missing = modelopt_config();
    missing.quantization_config = None;
    assert!(validate_dflash_metadata(&modelopt_store(""), &missing).is_err());
}

#[test]
fn rtncal_receipt_requires_the_exact_producer_identity() {
    let exact = modelopt_config();
    assert!(
        exact
            .quantization_config
            .as_ref()
            .unwrap()
            .is_exact_dflash2_rtncal()
    );

    for mutate in [
        |q: &mut DflashQuantizationConfig| q.producer = None,
        |q: &mut DflashQuantizationConfig| q.producer.as_mut().unwrap().name = "other".into(),
        |q: &mut DflashQuantizationConfig| {
            q.producer.as_mut().unwrap().version = "uncalibrated".into()
        },
    ] {
        let mut config = modelopt_config();
        mutate(config.quantization_config.as_mut().unwrap());
        assert!(
            !config
                .quantization_config
                .as_ref()
                .unwrap()
                .is_exact_dflash2_rtncal()
        );
    }
}
