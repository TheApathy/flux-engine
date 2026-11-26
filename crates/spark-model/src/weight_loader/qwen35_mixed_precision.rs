// SPDX-License-Identifier: AGPL-3.0-only

//! Shared mixed-precision helpers for Qwen3.5/3.6/3.8 loaders.

use anyhow::{Result, bail, ensure};
use atlas_core::config::{ModelConfig, ModelOptWeightFormat};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{
    DenseWeight, dense, dense_auto_fp8_or_bf16, dequant_nvfp4_to_bf16,
    quantized_modelopt_with_shape,
};

#[inline]
fn lm_head_needs_fp8_dequant(dtype: WeightDtype) -> bool {
    dtype == WeightDtype::FP8E4M3
}

fn lm_head_is_modelopt_nvfp4(config: &ModelConfig, prefix: &str) -> bool {
    config.quantization_config.as_ref().is_some_and(|quant| {
        quant.quant_method.eq_ignore_ascii_case("modelopt")
            && (quant.quant_algo.eq_ignore_ascii_case("NVFP4")
                || quant.modelopt_weight_format_for(prefix) == Some(ModelOptWeightFormat::Nvfp4))
    })
}

fn modelopt_lm_head_physical_rows(
    logical_rows: usize,
    hidden: usize,
    packed_shape: &[usize],
) -> Result<usize> {
    ensure!(
        logical_rows > 0,
        "logical LM-head row count must be non-zero"
    );
    ensure!(
        hidden > 0 && hidden.is_multiple_of(16),
        "LM-head hidden size must be a positive multiple of 16"
    );
    let [physical_rows, packed_cols] = packed_shape else {
        bail!("packed ModelOpt LM head must be rank 2, got {packed_shape:?}");
    };
    ensure!(
        *physical_rows >= logical_rows,
        "packed ModelOpt LM head has {physical_rows} physical rows, fewer than {logical_rows} logical rows"
    );
    ensure!(
        *packed_cols == hidden / 2,
        "packed ModelOpt LM head has {packed_cols} columns, expected {}",
        hidden / 2
    );
    Ok(*physical_rows)
}

/// Load an LM head while intercepting only the FP8 mixed-precision layout.
///
/// BF16 heads retain pointer-alias behavior. FP8 E4M3 and Standard-NVFP4
/// heads must be expanded to BF16: the architecture-neutral factory retains
/// a dense head for fallbacks and DFlash sharing before it creates its fast
/// NVFP4 copy. Returning a packed UInt8 pointer there would make the BF16
/// quantizer read four times past the allocation.
pub(super) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    for prefix in ["lm_head", "language_model.lm_head", "model.lm_head"] {
        let key = format!("{prefix}.weight");
        if !store.contains(&key) {
            continue;
        }
        let dtype = store.get(&key)?.dtype;
        if lm_head_needs_fp8_dequant(dtype) {
            return dense_auto_fp8_or_bf16(store, prefix, gpu);
        }
        if dtype == WeightDtype::UInt8 {
            if !lm_head_is_modelopt_nvfp4(config, prefix) {
                bail!("packed UInt8 LM head {prefix} lacks an exact ModelOpt NVFP4 declaration");
            }
            // Validate every packed extent and scalar before the dequantizer
            // sees a pointer. Qwen's server may cap the physical 248320-row
            // checkpoint to the tokenizer's 248077 logical rows. Validate the
            // full physical row stride, then dequantize its contiguous logical
            // prefix; model construction creates the runtime fast copy from
            // that exact prefix below.
            let packed = store.get(&key)?;
            let physical_rows = modelopt_lm_head_physical_rows(
                config.vocab_size,
                config.hidden_size,
                &packed.shape,
            )?;
            let _ = quantized_modelopt_with_shape(
                store,
                prefix,
                physical_rows,
                config.hidden_size,
                gpu,
            )?;
            return dequant_nvfp4_to_bf16(
                store,
                prefix,
                config.vocab_size,
                config.hidden_size,
                gpu,
            );
        }
        if dtype != WeightDtype::BF16 {
            bail!("unsupported LM head dtype {dtype:?} for {key}");
        }
        return dense(store, &key);
    }

    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

#[cfg(test)]
mod tests {
    use super::{
        lm_head_is_modelopt_nvfp4, lm_head_needs_fp8_dequant, modelopt_lm_head_physical_rows,
    };
    use atlas_core::config::{
        ModelConfig, ModelOptQuantizationGroup, ModelOptWeightFormat, QuantizationConfig,
    };
    use spark_runtime::weights::WeightDtype;

    #[test]
    fn lm_head_dequants_only_fp8() {
        assert!(lm_head_needs_fp8_dequant(WeightDtype::FP8E4M3));
        assert!(!lm_head_needs_fp8_dequant(WeightDtype::BF16));
        assert!(!lm_head_needs_fp8_dequant(WeightDtype::UInt8));
    }

    #[test]
    fn mixed_modelopt_requires_exact_nvfp4_lm_head_declaration() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.quantization_config = Some(QuantizationConfig {
            quant_method: "modelopt".into(),
            quant_algo: "MIXED_PRECISION".into(),
            format: String::new(),
            ignore_modules: Vec::new(),
            config_groups: vec![ModelOptQuantizationGroup {
                name: "nvfp4".into(),
                weight_format: ModelOptWeightFormat::Nvfp4,
                group_size: 16,
                targets: vec!["lm_head".into()],
            }],
        });
        assert!(lm_head_is_modelopt_nvfp4(&config, "lm_head"));
        assert!(!lm_head_is_modelopt_nvfp4(&config, "model.lm_head"));
    }

    #[test]
    fn modelopt_lm_head_allows_only_a_complete_logical_row_prefix() {
        assert_eq!(
            modelopt_lm_head_physical_rows(248_077, 5120, &[248_320, 2560]).unwrap(),
            248_320
        );
        assert_eq!(
            modelopt_lm_head_physical_rows(248_320, 5120, &[248_320, 2560]).unwrap(),
            248_320
        );
        assert!(modelopt_lm_head_physical_rows(248_321, 5120, &[248_320, 2560]).is_err());
        assert!(modelopt_lm_head_physical_rows(248_077, 5120, &[248_320, 2559]).is_err());
        assert!(modelopt_lm_head_physical_rows(248_077, 5120, &[248_320]).is_err());
    }
}
