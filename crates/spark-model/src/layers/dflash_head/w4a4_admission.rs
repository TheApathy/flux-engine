// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed admission and calibrated-scale retention for DFlash2 W4A4.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightDtype;

use crate::layers::DflashQuantization;
use crate::weight_map::QuantizedWeight;
use crate::weight_map::modelopt_scale_admission::{
    AdmittedModeloptScale, AdmittedSharedQkvScale, ModeloptScaleProjection, ModeloptScaleSource,
    admit_modelopt_checkpoint_scale, admit_shared_qkv_scale,
};

const W4A4_ENV: &str = "ATLAS_DFLASH2_W4A4_SHADOW";

/// Four calibrated activation scales retained for one DFlash2 layer.
///
/// Q/K/V and gate/up share one quantized activation each. The individual
/// admitted receipts remain attached so sharing is proved bit-for-bit rather
/// than inferred from rounded host values.
#[derive(Debug, Clone, Copy)]
pub(super) struct DflashNvfp4InputScales {
    qkv: AdmittedSharedQkvScale,
    output: AdmittedModeloptScale,
    gate_up: [AdmittedModeloptScale; 2],
    down: AdmittedModeloptScale,
}

impl DflashNvfp4InputScales {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read(
        gpu: &dyn GpuBackend,
        layer: usize,
        q: &QuantizedWeight,
        k: &QuantizedWeight,
        v: &QuantizedWeight,
        output: &QuantizedWeight,
        gate: &QuantizedWeight,
        up: &QuantizedWeight,
        down: &QuantizedWeight,
    ) -> Result<Self> {
        let q = read_scale(gpu, layer, ModeloptScaleProjection::AttentionQuery, q)?;
        let k = read_scale(gpu, layer, ModeloptScaleProjection::AttentionKey, k)?;
        let v = read_scale(gpu, layer, ModeloptScaleProjection::AttentionValue, v)?;
        let qkv = admit_shared_qkv_scale(layer, q, k, v)?;
        let output = read_scale(gpu, layer, ModeloptScaleProjection::AttentionOutput, output)?;
        let gate = read_scale(gpu, layer, ModeloptScaleProjection::FfnGate, gate)?;
        let up = read_scale(gpu, layer, ModeloptScaleProjection::FfnUp, up)?;
        ensure!(
            gate.has_same_value(up),
            "DFlash2 layer {layer} gate/up input scales must be bit-identical"
        );
        let down = read_scale(gpu, layer, ModeloptScaleProjection::FfnDown, down)?;
        Ok(Self {
            qkv,
            output,
            gate_up: [gate, up],
            down,
        })
    }

    pub(super) fn qkv(self) -> f32 {
        self.qkv.value()
    }

    pub(super) fn output(self) -> f32 {
        self.output.value()
    }

    pub(super) fn gate_up(self) -> f32 {
        self.gate_up[0].value()
    }

    pub(super) fn down(self) -> f32 {
        self.down.value()
    }

    pub(super) fn receipt_bits(self) -> [u32; 4] {
        [
            self.qkv.value_bits(),
            self.output.value_bits(),
            self.gate_up[0].value_bits(),
            self.down.value_bits(),
        ]
    }
}

fn read_scale(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: ModeloptScaleProjection,
    weight: &QuantizedWeight,
) -> Result<AdmittedModeloptScale> {
    ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && !weight.input_scale.is_null(),
        "DFlash2 layer {layer} {projection:?} requires complete ModelOpt operands"
    );
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "DFlash2 layer {layer} {projection:?} weight_scale_2 must be finite and positive"
    );
    let mut bytes = [0_u8; 4];
    gpu.copy_d2h(weight.input_scale, &mut bytes)
        .with_context(|| format!("read DFlash2 layer {layer} {projection:?} input_scale"))?;
    admit_modelopt_checkpoint_scale(
        ModeloptScaleSource { layer, projection },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        bytes,
    )
}

pub(super) fn w4a4_requested() -> Result<bool> {
    match std::env::var(W4A4_ENV) {
        Ok(value) => parse_w4a4_setting(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{W4A4_ENV} must be UTF-8 and exactly 0 or 1")
        }
    }
}

pub(super) fn parse_w4a4_setting(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{W4A4_ENV}={other:?} is invalid; expected 0 or 1"),
    }
}

pub(super) fn admit_w4a4_shadow(
    requested: bool,
    exact_rtncal_schema: bool,
    quantization: DflashQuantization,
    gamma: usize,
    physical_verify_k: usize,
    noise_only: bool,
) -> Result<bool> {
    if !requested {
        return Ok(false);
    }
    ensure!(
        exact_rtncal_schema,
        "{W4A4_ENV}=1 requires the exact calibrated ModelOpt RTNcal DFlash2 schema"
    );
    ensure!(
        quantization == DflashQuantization::Nvfp4,
        "{W4A4_ENV}=1 requires --dflash-quantization nvfp4"
    );
    ensure!(
        (gamma, physical_verify_k) == (15, 16),
        "{W4A4_ENV}=1 requires gamma=15 and physical verify K=16"
    );
    ensure!(
        noise_only,
        "{W4A4_ENV}=1 requires ATLAS_DFLASH_NOISE_ONLY=1"
    );
    Ok(true)
}

pub(super) fn validate_complete_layers(
    enabled: bool,
    direct: usize,
    expected: usize,
) -> Result<()> {
    ensure!(
        !enabled || (direct == expected && expected > 0),
        "DFlash2 W4A4 requires calibrated ModelOpt scales on every layer; got {direct}/{expected}"
    );
    Ok(())
}
