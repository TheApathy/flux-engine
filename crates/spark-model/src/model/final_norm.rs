// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level final-normalization dispatch.
//!
//! Qwen4-Exp owns its terminal normalization inside the learned model-level
//! hyperconnection mixer. Applying `final_norm` again changes every logit
//! scale and therefore probabilities, logprobs, and sampled output even when
//! greedy argmax happens to survive.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::layers::{Qwen4HyperConnection, ops};

#[derive(Clone, Copy)]
pub(super) enum PartialHead {
    EarlyExit,
    LayerResolutionProbe,
}

/// These partial heads consume ordinary H-wide rows and do not run Qwen4's
/// terminal mixer. Reject them before any recurrent state or GPU work changes.
pub(super) fn check_partial_head_admission(
    config: &ModelConfig,
    head: PartialHead,
    enabled: bool,
) -> Result<()> {
    let selector = match head {
        PartialHead::EarlyExit => "ATLAS_DFLASH_EARLY_EXIT",
        PartialHead::LayerResolutionProbe => "ATLAS_LAYER_RESOLUTION_PROBE",
    };
    ensure!(
        !(enabled && config.is_qwen4_exp()),
        "{selector} is unsupported for {}: partial H-wide heads do not preserve the Qwen4 hyperconnection residual and terminal mixer",
        config.model_type
    );
    Ok(())
}

fn validate_identity_geometry(
    config: &ModelConfig,
    mixer_geometry: Option<(usize, bool)>,
) -> Result<()> {
    if !config.final_norm_is_identity() {
        ensure!(
            mixer_geometry.is_none(),
            "{} unexpectedly installed a Qwen4 terminal mixer",
            config.model_type
        );
        return Ok(());
    }

    ensure!(
        config.hc_count > 1 && config.hc_lowrank > 0,
        "{} final norm is identity but hyperconnection geometry is incomplete: hc_count={} hc_lowrank={}",
        config.model_type,
        config.hc_count,
        config.hc_lowrank
    );
    let (residual_width, has_inject) = mixer_geometry.with_context(|| {
        format!(
            "{} final norm is identity but the terminal hyperconnection mixer is missing",
            config.model_type
        )
    })?;
    ensure!(
        residual_width == config.residual_width(),
        "{} terminal mixer width {residual_width} does not match residual width {}",
        config.model_type,
        config.residual_width()
    );
    ensure!(
        !has_inject,
        "{} terminal hyperconnection mixer must not carry an injection projection",
        config.model_type
    );
    Ok(())
}

/// Validate the construction-time invariant behind the identity dispatch.
pub(super) fn check_final_norm_identity(
    config: &ModelConfig,
    mixer: Option<&Qwen4HyperConnection>,
) -> Result<()> {
    validate_identity_geometry(
        config,
        mixer.map(|mixer| (mixer.residual_width(), mixer.inject.is_some())),
    )
}

impl TransformerModel {
    /// Apply the ordinary model-level RMSNorm, or validate the already-mixed
    /// Qwen4 terminal row. In this tree Qwen4's explicit terminal mixer writes
    /// H-wide rows directly into `output`; requiring pointer identity prevents
    /// an un-mixed 4H residual row from being silently truncated by a copy.
    pub(super) fn final_norm_rows(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        if self.final_norm_identity {
            ensure!(
                input == output,
                "{} final norm is identity only after the terminal hyperconnection mixer",
                self.config.model_type
            );
            return Ok(());
        }
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            input,
            &self.final_norm,
            output,
            rows,
            self.config.hidden_size as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{PartialHead, check_partial_head_admission, validate_identity_geometry};
    use atlas_core::config::ModelConfig;

    fn qwen4_exp_like() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.hc_count = 4;
        config.hc_lowrank = 320;
        config.hidden_size = 2560;
        config
    }

    #[test]
    fn partial_head_admission_rejects_qwen4_only_when_selected() {
        let qwen4 = qwen4_exp_like();
        let ordinary = ModelConfig::qwen3_next_80b_nvfp4();
        for (head, selector) in [
            (PartialHead::EarlyExit, "ATLAS_DFLASH_EARLY_EXIT"),
            (
                PartialHead::LayerResolutionProbe,
                "ATLAS_LAYER_RESOLUTION_PROBE",
            ),
        ] {
            let error = check_partial_head_admission(&qwen4, head, true)
                .unwrap_err()
                .to_string();
            assert!(error.contains(selector), "{error}");
            assert!(error.contains("terminal mixer"), "{error}");
            assert!(check_partial_head_admission(&qwen4, head, false).is_ok());
            for enabled in [false, true] {
                assert!(check_partial_head_admission(&ordinary, head, enabled).is_ok());
            }
        }
    }

    #[test]
    fn incomplete_qwen4_geometry_cannot_bypass_partial_head_admission() {
        let mut qwen4 = qwen4_exp_like();
        qwen4.hc_count = 1;
        qwen4.hc_lowrank = 0;
        for head in [PartialHead::EarlyExit, PartialHead::LayerResolutionProbe] {
            assert!(check_partial_head_admission(&qwen4, head, true).is_err());
        }
    }

    #[test]
    fn qwen4_identity_requires_the_exact_terminal_mixer_contract() {
        let config = qwen4_exp_like();
        assert!(validate_identity_geometry(&config, Some((10_240, false))).is_ok());

        for (geometry, needle) in [
            (None, "mixer is missing"),
            (Some((2560, false)), "does not match residual width"),
            (Some((10_240, true)), "must not carry an injection"),
        ] {
            let error = validate_identity_geometry(&config, geometry)
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn incomplete_qwen4_geometry_is_rejected() {
        let mut config = qwen4_exp_like();
        config.hc_count = 1;
        let error = validate_identity_geometry(&config, Some((2560, false)))
            .unwrap_err()
            .to_string();
        assert!(error.contains("geometry is incomplete"), "{error}");
    }

    #[test]
    fn ordinary_models_keep_rms_norm_and_reject_a_qwen4_mixer() {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(validate_identity_geometry(&config, None).is_ok());
        let error = validate_identity_geometry(&config, Some((config.hidden_size, false)))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unexpectedly installed"), "{error}");
    }
}
