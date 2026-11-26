// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic-only admission for a plain-AR Qwen3.8-27B TC-parity reference.
//!
//! Speculative verification intentionally refreezes several target operations
//! onto tensor-core arithmetic. A no-drafter reference normally has no verify
//! width, so `ATLAS_DECODE_TC_PARITY=1` remains inert and compares unlike
//! arithmetic. This control supplies the qualified 16-row geometry while the
//! ordinary target-only scheduler remains in charge.

use anyhow::Result;
use atlas_core::config::ModelConfig;

use crate::cli;

pub(super) const CONTROL_ENV: &str = "ATLAS_DECODE_TC_PARITY_REFERENCE_ROWS";
const QUALIFIED_ROWS: usize = 16;
const QUALIFIED_TARGET: &str = "qwen3.8-27b";

/// Route selectors required for a reference that actually mirrors the
/// qualified 16-row verifier. Requiring explicit values avoids a misleading
/// reference receipt when one tensor-core family silently stays on K1.
const REQUIRED_ROUTE_ENV: &[(&str, &str)] = &[
    ("ATLAS_DECODE_TC_PARITY", "1"),
    ("ATLAS_FFN_TC", "1"),
    ("ATLAS_FFN_KGAMMA_M16", "1"),
    ("ATLAS_FFN_KGAMMA_M128", "1"),
    ("ATLAS_FFN_M16_TRANSPOSED", "1"),
    ("ATLAS_SSM_PROJ_TC", "1"),
    ("ATLAS_SSM_QKVZ_SPLITK", "4"),
    ("ATLAS_SSM_OUT_SPLITK", "4"),
    ("ATLAS_TC_NVFP4_M16", "0"),
    ("ATLAS_LM_HEAD_TC", "1"),
    ("ATLAS_LM_HEAD_T", "1"),
];

#[derive(Clone, Copy)]
struct ReferenceContext<'a> {
    target_model: &'a str,
    model_type: &'a str,
    hidden_size: usize,
    num_experts: usize,
    speculative: bool,
    dflash: bool,
    self_speculative: bool,
    ngram_speculative: bool,
    mtp_gate: bool,
}

/// The qualified checkpoint is a multimodal wrapper, so `vision_config` does
/// not describe the target arithmetic. Pin the selected text-kernel geometry
/// and its canonical dense trunk identity instead.
fn is_dense_q27(context: ReferenceContext<'_>) -> bool {
    context.model_type == "qwen3_5" && context.hidden_size == 5120 && context.num_experts == 0
}

fn live_environment_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} is not valid Unicode")
        }
    }
}

fn admit_with(
    context: ReferenceContext<'_>,
    mut get: impl FnMut(&str) -> Result<Option<String>>,
) -> Result<Option<usize>> {
    let Some(control) = get(CONTROL_ENV)? else {
        return Ok(None);
    };
    anyhow::ensure!(
        control == QUALIFIED_ROWS.to_string(),
        "{CONTROL_ENV} must be unset or exactly {QUALIFIED_ROWS}; got {control:?}"
    );
    anyhow::ensure!(
        context.target_model == QUALIFIED_TARGET,
        "{CONTROL_ENV}={QUALIFIED_ROWS} is diagnostic-only for kernel target {QUALIFIED_TARGET}; got {:?}",
        context.target_model
    );
    anyhow::ensure!(
        is_dense_q27(context),
        "{CONTROL_ENV}={QUALIFIED_ROWS} requires the dense qwen3_5 27B language trunk (model_type=qwen3_5, hidden_size=5120, num_experts=0)"
    );
    anyhow::ensure!(
        !(context.speculative
            || context.dflash
            || context.self_speculative
            || context.ngram_speculative
            || context.mtp_gate),
        "{CONTROL_ENV}={QUALIFIED_ROWS} requires plain autoregressive scheduling: remove every speculative flag and --mtp-gate"
    );

    match get("ATLAS_DISABLE_SPECULATION")?.as_deref() {
        None | Some("0") => {}
        Some(value) => anyhow::bail!(
            "{CONTROL_ENV}={QUALIFIED_ROWS} requires ATLAS_DISABLE_SPECULATION to be unset or 0; got {value:?}"
        ),
    }
    match get("ATLAS_TARGET_LMHEAD_VOCAB")?.as_deref() {
        None | Some("0") => {}
        Some(value) => anyhow::bail!(
            "{CONTROL_ENV}={QUALIFIED_ROWS} requires a full target vocabulary; ATLAS_TARGET_LMHEAD_VOCAB must be unset or 0, got {value:?}"
        ),
    }
    for &(name, expected) in REQUIRED_ROUTE_ENV {
        let actual = get(name)?;
        anyhow::ensure!(
            actual.as_deref() == Some(expected),
            "{CONTROL_ENV}={QUALIFIED_ROWS} requires {name}={expected}; got {actual:?}"
        );
    }
    Ok(Some(QUALIFIED_ROWS))
}

pub(crate) fn admit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    target_model: &str,
) -> Result<Option<usize>> {
    admit_with(
        ReferenceContext {
            target_model,
            model_type: &config.model_type,
            hidden_size: config.hidden_size,
            num_experts: config.num_experts,
            speculative: args.speculative,
            dflash: args.dflash,
            self_speculative: args.self_speculative,
            ngram_speculative: args.ngram_speculative,
            mtp_gate: args.mtp_gate.is_some(),
        },
        live_environment_value,
    )
}

#[cfg(test)]
#[path = "tc_parity_ar_reference_tests.rs"]
mod tests;
