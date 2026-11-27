// SPDX-License-Identifier: AGPL-3.0-only

//! Universal DualPlaneNvfp4 self-speculation controller with fail-closed roofline guards.
//!
//! Enables zero-draft-model speculative decoding for checkpoints that lack an
//! external DFlash2 drafter (e.g., GLM-5.3, Gemma-4, Qwen4, custom fine-tunes)
//! using shared-scale [`DualPlaneNvfp4Weight`](crate::weight_map::DualPlaneNvfp4Weight):
//!
//! - **Dense models (no DFlash2)**: drafts via `W_hi` (`4.5 bpw`) combined with
//!   50% lowest-activity ($\mathcal{L}_A^\ell$) FFN-only layer skipping (keeping
//!   every Attention/GDN recurrence active so KV/SSM caches stay consistent),
//!   then verifies `gamma + 1` tokens in one batched `W_hi + 0.125*W_lo` pass.
//! - **MoE models (no DFlash2)**: drafts via `W_hi` + Top-1 expert routing and
//!   verifies with all active experts — **STRICTLY GATED** on `concurrency >= 8`
//!   AND grouped-by-expert verify dispatch (`MoeVerifyKernelKind::GroupedByExpert`).
//!   At `C = 1..7` (especially `C = 1` on fine-grained MoEs like Qwen3.8-80B
//!   `E=512, k=10`), union-of-experts verify reads nearly `(gamma + 1)x` as many
//!   unique expert matrices as single-token decode (an anti-speedup death trap),
//!   so the controller fails closed and refuses activation.
//!
//! Opt-in only via `ATLAS_DUAL_PLANE_SELF_SPEC` (default OFF). Also integrates
//! [`spark_runtime::autoematic::EntropyGateSignal`] for optional AutoFP6 verify
//! precision escalation.

use std::fmt;

use spark_runtime::autoematic::{
    AutoEmaticGateConfig, AutoEmaticGateError, AutoEmaticPrecisionLevel, EntropyGateSignal,
};

/// Environment variable enabling universal DualPlaneNvfp4 self-speculation.
///
/// Unset or empty disables self-speculation (default OFF).
pub const DUAL_PLANE_SELF_SPEC_ENV: &str = "ATLAS_DUAL_PLANE_SELF_SPEC";

/// Minimum concurrency required to admit MoE `W_hi + Top-1` self-speculation.
///
/// Below `C = 8`, fine-grained MoE verify reads expand linearly in `gamma + 1`
/// due to sparse expert collision (coupon-collector death trap), yielding a
/// net slowdown. At `C >= 8` under grouped-by-expert dispatch, expert loads
/// amortize across requests and flip to a net win.
pub const MOE_SELF_SPEC_MIN_CONCURRENCY: usize = 8;

/// Default fraction of lowest-$\mathcal{L}_A^\ell$ dense FFN blocks skipped
/// during `W_hi` self-drafting (all Attention/GDN layers remain active).
pub const DENSE_SELF_SPEC_FFN_SKIP_FRACTION: f32 = 0.50;

/// Default speculative draft depth `gamma` for dense `W_hi + FFN-skip`.
pub const DENSE_SELF_SPEC_DEFAULT_GAMMA: usize = 4;

/// Default speculative draft depth `gamma` for batched `C >= 8` MoE `W_hi + Top-1`.
pub const MOE_SELF_SPEC_DEFAULT_GAMMA: usize = 3;

/// Opt-in enablement state parsed from `ATLAS_DUAL_PLANE_SELF_SPEC`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelfSpecEnablement {
    /// Disabled (default): baseline decode/verify path untouched.
    #[default]
    Disabled,
    /// Opt-in enabled (still subject to all roofline & drafter admission guards).
    Enabled,
}

/// Whether an external trained drafter (e.g., DFlash2) is active on this model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalDrafterPresence {
    /// No external drafter is installed; model is eligible for self-speculation.
    Absent,
    /// External DFlash2 drafter is installed and strictly dominates self-spec.
    DFlash2Installed,
}

/// Whether DualPlaneNvfp4 (`W_hi + W_lo`) weights are resident for this model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualPlaneAvailability {
    /// Standard single-plane NVFP4/FP8 weights only; cannot self-draft on `W_hi`.
    Unavailable,
    /// Shared-scale `DualPlaneNvfp4Weight` (`8.5 bpw`) is loaded.
    Available,
}

/// Dispatch pattern used by the target MoE verify pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeVerifyKernelKind {
    /// Per-token ungrouped dispatch (re-reads experts per token; never amortizes).
    UngroupedPerToken,
    /// Grouped-by-expert GEMM dispatch (amortizes each loaded expert across batch).
    GroupedByExpert,
}

/// Target model topology evaluated by [`SelfSpecController::admit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelfSpecModelTopology {
    /// Dense transformer (all FFN weights active every token).
    Dense {
        /// Total number of transformer layers.
        num_layers: usize,
    },
    /// Mixture-of-Experts transformer.
    Moe {
        /// Total number of transformer layers.
        num_layers: usize,
        /// Total routed experts per MoE layer (`E`, e.g. 512 on Qwen3.8-80B).
        total_experts: usize,
        /// Active routed experts per token during full verify (`k`, e.g. 10).
        active_experts_per_token: usize,
        /// Target verify dispatch kernel architecture.
        verify_kernel: MoeVerifyKernelKind,
    },
}

/// Per-layer FFN execution choice during dense `W_hi` self-drafting.
///
/// Attention and GDN recurrence ALWAYS execute in every layer so KV and
/// recurrent states stay valid across draft tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnDraftExecution {
    /// Execute this layer's FFN using the `4.5 bpw` `W_hi` plane.
    ExecuteWHi,
    /// Skip this layer's FFN during drafting (lowest-$\mathcal{L}_A^\ell$ block).
    SkipFfn,
}

/// Per-layer relative residual activity profile
/// $\mathcal{L}_A^\ell = \|H_\ell - H_{\ell-1}\|_2 / (\|H_{\ell-1}\|_2 + \epsilon)$
/// used to pick the 50% lowest-activity FFN blocks for dense self-drafting.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerActivityProfile {
    layer_deltas: Vec<f32>,
}

impl LayerActivityProfile {
    /// Create a profile from per-layer relative residual deltas.
    ///
    /// # Errors
    /// Returns an error if `layer_deltas` is empty or contains non-finite values.
    pub fn try_new(layer_deltas: Vec<f32>) -> Result<Self, SelfSpecRefusal> {
        if layer_deltas.is_empty() {
            return Err(SelfSpecRefusal::InvalidTopology(
                "layer activity profile must be non-empty".to_owned(),
            ));
        }
        if layer_deltas.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(SelfSpecRefusal::InvalidTopology(
                "layer activity deltas must be finite and non-negative".to_owned(),
            ));
        }
        Ok(Self { layer_deltas })
    }

    /// Number of layers covered by this profile.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layer_deltas.len()
    }

    /// Whether the profile has zero layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layer_deltas.is_empty()
    }

    /// Deterministically select the lowest-$\mathcal{L}_A^\ell$ FFN layers to
    /// skip during `W_hi` self-drafting.
    ///
    /// Ties are broken deterministically by preferring deeper layers for skipping
    /// after layer 0 (protecting the input embedding boundary layer).
    #[must_use]
    pub fn build_ffn_skip_schedule(&self, skip_fraction: f32) -> Vec<FfnDraftExecution> {
        let num_layers = self.layer_deltas.len();
        let clamped = skip_fraction.clamp(0.0, 0.75);
        let target_skip = ((num_layers as f32) * clamped).floor() as usize;
        let mut order: Vec<usize> = (0..num_layers).collect();
        order.sort_by(|&a, &b| {
            self.layer_deltas[a]
                .total_cmp(&self.layer_deltas[b])
                .then_with(|| b.cmp(&a))
        });

        let mut schedule = vec![FfnDraftExecution::ExecuteWHi; num_layers];
        for &layer_idx in order.iter().take(target_skip) {
            schedule[layer_idx] = FfnDraftExecution::SkipFfn;
        }
        schedule
    }
}

/// Error returned when parsing `ATLAS_DUAL_PLANE_SELF_SPEC` or its companion
/// `ATLAS_AUTOEMATIC_GATE_ENTROPY` configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelfSpecConfigError {
    /// `ATLAS_DUAL_PLANE_SELF_SPEC` is not valid UTF-8.
    NonUtf8,
    /// `ATLAS_DUAL_PLANE_SELF_SPEC` is not a recognized boolean token.
    InvalidToken(String),
    /// Companion `ATLAS_AUTOEMATIC_GATE_ENTROPY` failed fail-closed validation.
    AutoEmaticGate(AutoEmaticGateError),
}

impl fmt::Display for SelfSpecConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => write!(f, "{DUAL_PLANE_SELF_SPEC_ENV} must contain valid UTF-8"),
            Self::InvalidToken(raw) => write!(
                f,
                "{DUAL_PLANE_SELF_SPEC_ENV} must be one of 0/1/false/true/off/on, got {raw:?}"
            ),
            Self::AutoEmaticGate(err) => write!(f, "invalid AutoEMatic entropy gate: {err}"),
        }
    }
}

impl std::error::Error for SelfSpecConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AutoEmaticGate(err) => Some(err),
            Self::NonUtf8 | Self::InvalidToken(_) => None,
        }
    }
}

impl From<AutoEmaticGateError> for SelfSpecConfigError {
    fn from(err: AutoEmaticGateError) -> Self {
        Self::AutoEmaticGate(err)
    }
}

/// Fail-closed refusal reason returned when self-speculation is not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelfSpecRefusal {
    /// `ATLAS_DUAL_PLANE_SELF_SPEC` is disabled (default OFF).
    DisabledByConfig,
    /// Model does not have `DualPlaneNvfp4Weight` loaded.
    DualPlaneWeightsUnavailable,
    /// An external DFlash2 drafter is installed and takes precedence.
    ExternalDrafterPresent,
    /// MoE concurrency is below the batch-amortization crossover (`C < 8`).
    MoeLowConcurrencyDeathTrap {
        concurrency: usize,
        min_required: usize,
    },
    /// MoE verify pass is not grouped-by-expert, so expert weight reads cannot
    /// amortize across requests.
    MoeVerifyNotGrouped,
    /// Model topology parameters are invalid or inconsistent.
    InvalidTopology(String),
}

impl fmt::Display for SelfSpecRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DisabledByConfig => {
                write!(
                    f,
                    "self-speculation disabled ({DUAL_PLANE_SELF_SPEC_ENV} unset/off)"
                )
            }
            Self::DualPlaneWeightsUnavailable => {
                write!(f, "DualPlaneNvfp4 weights are not loaded")
            }
            Self::ExternalDrafterPresent => {
                write!(
                    f,
                    "external DFlash2 drafter is installed; skipping self-spec"
                )
            }
            Self::MoeLowConcurrencyDeathTrap {
                concurrency,
                min_required,
            } => write!(
                f,
                "refusing MoE W_hi+Top-1 self-spec at concurrency={concurrency} < {min_required} \
                 (sparse union-of-experts verify anti-speedup death trap)"
            ),
            Self::MoeVerifyNotGrouped => write!(
                f,
                "refusing MoE W_hi+Top-1 self-spec without grouped-by-expert verify dispatch"
            ),
            Self::InvalidTopology(msg) => write!(f, "invalid self-spec topology: {msg}"),
        }
    }
}

impl std::error::Error for SelfSpecRefusal {}

/// Admitted self-speculation strategy chosen by [`SelfSpecController::admit`].
#[derive(Clone, Debug, PartialEq)]
pub enum SelfSpecDraftStrategy {
    /// Dense model: draft on `W_hi` (`4.5 bpw`) with 50% FFN-only layer skipping.
    DenseWHiWithFfnSkip {
        ffn_schedule: Vec<FfnDraftExecution>,
        skipped_ffn_layers: usize,
        gamma: usize,
    },
    /// Batched MoE (`C >= 8`, grouped verify): draft on `W_hi` + Top-1 expert.
    MoeWHiTop1Grouped {
        draft_top_k: usize,
        verify_top_k: usize,
        concurrency: usize,
        gamma: usize,
        expected_draft_unique_experts: f64,
        expected_verify_unique_experts: f64,
    },
}

/// Complete admitted self-speculation execution plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfSpecPlan {
    pub strategy: SelfSpecDraftStrategy,
    /// Estimated weight-read ratio `verify_bytes / draft_bytes` per forward pass.
    pub estimated_draft_bandwidth_reduction: f64,
}

/// Configuration for [`SelfSpecController`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfSpecConfig {
    pub enablement: SelfSpecEnablement,
    pub min_moe_concurrency: usize,
    pub dense_ffn_skip_fraction: f32,
    pub dense_gamma: usize,
    pub moe_gamma: usize,
    pub autoematic_gate: Option<AutoEmaticGateConfig>,
}

impl Default for SelfSpecConfig {
    fn default() -> Self {
        Self {
            enablement: SelfSpecEnablement::Disabled,
            min_moe_concurrency: MOE_SELF_SPEC_MIN_CONCURRENCY,
            dense_ffn_skip_fraction: DENSE_SELF_SPEC_FFN_SKIP_FRACTION,
            dense_gamma: DENSE_SELF_SPEC_DEFAULT_GAMMA,
            moe_gamma: MOE_SELF_SPEC_DEFAULT_GAMMA,
            autoematic_gate: None,
        }
    }
}

impl SelfSpecConfig {
    /// Load configuration from `ATLAS_DUAL_PLANE_SELF_SPEC` and optional
    /// `ATLAS_AUTOEMATIC_GATE_ENTROPY` (both default OFF).
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or invalid `ATLAS_DUAL_PLANE_SELF_SPEC` tokens
    /// and propagates any [`AutoEmaticGateError`] from `ATLAS_AUTOEMATIC_GATE_ENTROPY`.
    pub fn from_env() -> Result<Self, SelfSpecConfigError> {
        let enable_raw = std::env::var_os(DUAL_PLANE_SELF_SPEC_ENV);
        let enablement = Self::parse_enablement_os(enable_raw.as_deref())?;
        let autoematic_gate = AutoEmaticGateConfig::from_env()?;
        Ok(Self {
            enablement,
            autoematic_gate,
            ..Self::default()
        })
    }

    /// Construct from optional OS strings (used for fail-closed unit testing).
    ///
    /// # Errors
    /// Fails closed on non-UTF-8, invalid enablement tokens, or invalid entropy thresholds.
    pub fn from_os_values(
        enablement: Option<&std::ffi::OsStr>,
        gate_entropy: Option<&std::ffi::OsStr>,
    ) -> Result<Self, SelfSpecConfigError> {
        let enablement = Self::parse_enablement_os(enablement)?;
        let autoematic_gate = AutoEmaticGateConfig::parse_os(gate_entropy)?;
        Ok(Self {
            enablement,
            autoematic_gate,
            ..Self::default()
        })
    }

    /// Parse an optional OS string for `ATLAS_DUAL_PLANE_SELF_SPEC`.
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or unrecognized non-empty tokens.
    pub fn parse_enablement_os(
        raw: Option<&std::ffi::OsStr>,
    ) -> Result<SelfSpecEnablement, SelfSpecConfigError> {
        let Some(os_val) = raw else {
            return Ok(SelfSpecEnablement::Disabled);
        };
        let utf8 = os_val.to_str().ok_or(SelfSpecConfigError::NonUtf8)?;
        Self::parse_enablement(Some(utf8))
    }

    /// Parse an optional `ATLAS_DUAL_PLANE_SELF_SPEC` value.
    ///
    /// # Errors
    /// Fails closed on unrecognized non-empty tokens.
    pub fn parse_enablement(raw: Option<&str>) -> Result<SelfSpecEnablement, SelfSpecConfigError> {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(SelfSpecEnablement::Disabled);
        };
        match trimmed.to_ascii_lowercase().as_str() {
            "0" | "false" | "off" => Ok(SelfSpecEnablement::Disabled),
            "1" | "true" | "on" => Ok(SelfSpecEnablement::Enabled),
            _ => Err(SelfSpecConfigError::InvalidToken(trimmed.to_owned())),
        }
    }

    /// Construct an opt-in enabled configuration using canonical roofline floors.
    #[must_use]
    pub const fn enabled_canonical(autoematic_gate: Option<AutoEmaticGateConfig>) -> Self {
        Self {
            enablement: SelfSpecEnablement::Enabled,
            min_moe_concurrency: MOE_SELF_SPEC_MIN_CONCURRENCY,
            dense_ffn_skip_fraction: DENSE_SELF_SPEC_FFN_SKIP_FRACTION,
            dense_gamma: DENSE_SELF_SPEC_DEFAULT_GAMMA,
            moe_gamma: MOE_SELF_SPEC_DEFAULT_GAMMA,
            autoematic_gate,
        }
    }
}

/// Coupon-collector expected unique experts touched when `tokens` tokens each
/// route independently to `k_active` out of `total_experts`.
///
/// Exact formula: `E * (1 - (1 - k/E)^tokens)`.
#[must_use]
pub fn expected_unique_experts(total_experts: usize, k_active: usize, tokens: usize) -> f64 {
    if total_experts == 0 || k_active == 0 || tokens == 0 {
        return 0.0;
    }
    let e = total_experts as f64;
    let p_miss = (1.0 - (k_active.min(total_experts) as f64) / e).clamp(0.0, 1.0);
    e * (1.0 - p_miss.powf(tokens as f64))
}

/// Roofline-guarded DualPlaneNvfp4 self-speculation admission & verify controller.
#[derive(Clone, Debug)]
pub struct SelfSpecController {
    config: SelfSpecConfig,
}

impl SelfSpecController {
    /// Create a controller from a validated [`SelfSpecConfig`].
    ///
    /// Enforces `min_moe_concurrency >= MOE_SELF_SPEC_MIN_CONCURRENCY` (8) even
    /// if a caller manually lowered the struct field, so the `C = 1..7` MoE
    /// death trap can never be bypassed accidentally.
    #[must_use]
    pub fn new(mut config: SelfSpecConfig) -> Self {
        if config.min_moe_concurrency < MOE_SELF_SPEC_MIN_CONCURRENCY {
            config.min_moe_concurrency = MOE_SELF_SPEC_MIN_CONCURRENCY;
        }
        Self { config }
    }

    /// Return the active configuration.
    #[must_use]
    pub const fn config(&self) -> SelfSpecConfig {
        self.config
    }

    /// Admit or fail-closed refuse self-speculation for a serving configuration.
    ///
    /// # Errors
    /// Returns a specific [`SelfSpecRefusal`] explaining why self-speculation
    /// is refused (disabled, DFlash2 present, missing DualPlane weights,
    /// `C < 8` MoE death trap, ungrouped MoE verify, or invalid topology).
    pub fn admit(
        &self,
        topology: &SelfSpecModelTopology,
        drafter: ExternalDrafterPresence,
        dual_plane: DualPlaneAvailability,
        concurrency: usize,
        layer_activity: Option<&LayerActivityProfile>,
    ) -> Result<SelfSpecPlan, SelfSpecRefusal> {
        if self.config.enablement == SelfSpecEnablement::Disabled {
            return Err(SelfSpecRefusal::DisabledByConfig);
        }
        if drafter == ExternalDrafterPresence::DFlash2Installed {
            return Err(SelfSpecRefusal::ExternalDrafterPresent);
        }
        if dual_plane == DualPlaneAvailability::Unavailable {
            return Err(SelfSpecRefusal::DualPlaneWeightsUnavailable);
        }
        if concurrency == 0 {
            return Err(SelfSpecRefusal::InvalidTopology(
                "concurrency must be >= 1".to_owned(),
            ));
        }

        match *topology {
            SelfSpecModelTopology::Dense { num_layers } => {
                if num_layers == 0 {
                    return Err(SelfSpecRefusal::InvalidTopology(
                        "dense num_layers must be > 0".to_owned(),
                    ));
                }
                let ffn_schedule = match layer_activity {
                    Some(profile) => {
                        if profile.len() != num_layers {
                            return Err(SelfSpecRefusal::InvalidTopology(format!(
                                "layer activity profile length {} != num_layers {num_layers}",
                                profile.len()
                            )));
                        }
                        profile.build_ffn_skip_schedule(self.config.dense_ffn_skip_fraction)
                    }
                    None => default_middle_layer_skip_schedule(
                        num_layers,
                        self.config.dense_ffn_skip_fraction,
                    ),
                };
                let skipped_ffn_layers = ffn_schedule
                    .iter()
                    .filter(|&&mode| mode == FfnDraftExecution::SkipFfn)
                    .count();

                // Estimate draft vs verify weight read ratio:
                //   Verify reads 8.5 bpw across all layers (approx 75% FFN, 25% Attn/GDN).
                //   Draft reads 4.5 bpw on Attention/GDN + (1 - skip_frac) of FFN.
                let ffn_keep = 1.0 - (skipped_ffn_layers as f64) / (num_layers as f64);
                let draft_bpw_equiv = 4.5 * (0.25 + 0.75 * ffn_keep);
                let bw_reduction = 8.5 / draft_bpw_equiv;

                Ok(SelfSpecPlan {
                    strategy: SelfSpecDraftStrategy::DenseWHiWithFfnSkip {
                        ffn_schedule,
                        skipped_ffn_layers,
                        gamma: self.config.dense_gamma,
                    },
                    estimated_draft_bandwidth_reduction: bw_reduction,
                })
            }
            SelfSpecModelTopology::Moe {
                num_layers,
                total_experts,
                active_experts_per_token,
                verify_kernel,
            } => {
                if num_layers == 0
                    || total_experts < 2
                    || active_experts_per_token < 2
                    || active_experts_per_token > total_experts
                {
                    return Err(SelfSpecRefusal::InvalidTopology(format!(
                        "invalid MoE topology: layers={num_layers}, E={total_experts}, k={active_experts_per_token}"
                    )));
                }
                if verify_kernel != MoeVerifyKernelKind::GroupedByExpert {
                    return Err(SelfSpecRefusal::MoeVerifyNotGrouped);
                }
                if concurrency < self.config.min_moe_concurrency {
                    return Err(SelfSpecRefusal::MoeLowConcurrencyDeathTrap {
                        concurrency,
                        min_required: self.config.min_moe_concurrency,
                    });
                }

                let gamma = self.config.moe_gamma;
                let draft_unique = expected_unique_experts(total_experts, 1, concurrency);
                let verify_unique = expected_unique_experts(
                    total_experts,
                    active_experts_per_token,
                    concurrency * (gamma + 1),
                );
                let draft_bytes_rel = draft_unique * 4.5;
                let verify_bytes_rel = verify_unique * 8.5;
                let bw_reduction = if draft_bytes_rel > 0.0 {
                    verify_bytes_rel / draft_bytes_rel
                } else {
                    1.0
                };

                Ok(SelfSpecPlan {
                    strategy: SelfSpecDraftStrategy::MoeWHiTop1Grouped {
                        draft_top_k: 1,
                        verify_top_k: active_experts_per_token,
                        concurrency,
                        gamma,
                        expected_draft_unique_experts: draft_unique,
                        expected_verify_unique_experts: verify_unique,
                    },
                    estimated_draft_bandwidth_reduction: bw_reduction,
                })
            }
        }
    }

    /// Evaluate the verify precision level for a decode/verify token pivot
    /// using the decoupled [`EntropyGateSignal`] contract.
    ///
    /// Returns [`AutoEmaticPrecisionLevel::Nvfp4`] whenever the AutoEMatic
    /// entropy gate is unconfigured (default OFF).
    #[must_use]
    pub fn verify_precision_for(&self, signal: EntropyGateSignal) -> AutoEmaticPrecisionLevel {
        match self.config.autoematic_gate {
            Some(gate) => gate.evaluate(signal),
            None => AutoEmaticPrecisionLevel::Nvfp4,
        }
    }
}

/// Fallback schedule when no runtime $\mathcal{L}_A^\ell$ profile has been
/// collected yet: preserves boundary layers (0 and last) and skips alternating
/// middle layers up to `skip_fraction`.
fn default_middle_layer_skip_schedule(
    num_layers: usize,
    skip_fraction: f32,
) -> Vec<FfnDraftExecution> {
    let target_skip = ((num_layers as f32) * skip_fraction.clamp(0.0, 0.75)).floor() as usize;
    let mut schedule = vec![FfnDraftExecution::ExecuteWHi; num_layers];
    let mut skipped = 0_usize;
    for idx in (1..num_layers.saturating_sub(1)).step_by(2) {
        if skipped >= target_skip {
            break;
        }
        schedule[idx] = FfnDraftExecution::SkipFfn;
        skipped += 1;
    }
    for idx in (2..num_layers.saturating_sub(1)).step_by(2) {
        if skipped >= target_skip {
            break;
        }
        schedule[idx] = FfnDraftExecution::SkipFfn;
        skipped += 1;
    }
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen38_80b_moe(verify_kernel: MoeVerifyKernelKind) -> SelfSpecModelTopology {
        SelfSpecModelTopology::Moe {
            num_layers: 48,
            total_experts: 512,
            active_experts_per_token: 10,
            verify_kernel,
        }
    }

    #[test]
    fn defaults_off_and_fails_closed_on_invalid_env() {
        assert_eq!(
            SelfSpecConfig::parse_enablement(None).unwrap(),
            SelfSpecEnablement::Disabled
        );
        assert_eq!(
            SelfSpecConfig::parse_enablement(Some("")).unwrap(),
            SelfSpecEnablement::Disabled
        );
        assert_eq!(
            SelfSpecConfig::parse_enablement(Some("1")).unwrap(),
            SelfSpecEnablement::Enabled
        );
        assert!(SelfSpecConfig::parse_enablement(Some("auto")).is_err());

        // Companion ATLAS_AUTOEMATIC_GATE_ENTROPY errors must propagate fail-closed.
        let gate_err = SelfSpecConfig::from_os_values(
            Some(std::ffi::OsStr::new("1")),
            Some(std::ffi::OsStr::new("NaN")),
        )
        .unwrap_err();
        assert!(matches!(gate_err, SelfSpecConfigError::AutoEmaticGate(_)));
        assert!(std::error::Error::source(&gate_err).is_some());

        #[cfg(unix)]
        {
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt;
            let bad = OsStr::from_bytes(&[0xFF, 0xFE]);
            assert_eq!(
                SelfSpecConfig::from_os_values(Some(bad), None),
                Err(SelfSpecConfigError::NonUtf8)
            );
            assert_eq!(
                SelfSpecConfig::from_os_values(Some(OsStr::new("1")), Some(bad)),
                Err(SelfSpecConfigError::AutoEmaticGate(
                    AutoEmaticGateError::NonUtf8
                ))
            );
        }

        let ctrl = SelfSpecController::new(SelfSpecConfig::default());
        let err = ctrl
            .admit(
                &SelfSpecModelTopology::Dense { num_layers: 64 },
                ExternalDrafterPresence::Absent,
                DualPlaneAvailability::Available,
                1,
                None,
            )
            .unwrap_err();
        assert_eq!(err, SelfSpecRefusal::DisabledByConfig);
    }

    #[test]
    fn moe_self_spec_refuses_c1_through_c7_death_trap_and_ungrouped_verify() {
        let ctrl = SelfSpecController::new(SelfSpecConfig::enabled_canonical(None));
        let grouped = qwen38_80b_moe(MoeVerifyKernelKind::GroupedByExpert);

        // C=1 through C=7 MUST fail closed with MoeLowConcurrencyDeathTrap.
        for c in 1..MOE_SELF_SPEC_MIN_CONCURRENCY {
            let err = ctrl
                .admit(
                    &grouped,
                    ExternalDrafterPresence::Absent,
                    DualPlaneAvailability::Available,
                    c,
                    None,
                )
                .unwrap_err();
            assert_eq!(
                err,
                SelfSpecRefusal::MoeLowConcurrencyDeathTrap {
                    concurrency: c,
                    min_required: MOE_SELF_SPEC_MIN_CONCURRENCY,
                },
                "MoE self-spec must refuse C={c}"
            );
        }

        // Even if someone lowers min_moe_concurrency manually, constructor clamps it to >= 8.
        let mut unsafe_cfg = SelfSpecConfig::enabled_canonical(None);
        unsafe_cfg.min_moe_concurrency = 1;
        let clamped_ctrl = SelfSpecController::new(unsafe_cfg);
        assert_eq!(
            clamped_ctrl
                .admit(
                    &grouped,
                    ExternalDrafterPresence::Absent,
                    DualPlaneAvailability::Available,
                    1,
                    None,
                )
                .unwrap_err(),
            SelfSpecRefusal::MoeLowConcurrencyDeathTrap {
                concurrency: 1,
                min_required: MOE_SELF_SPEC_MIN_CONCURRENCY,
            }
        );

        // Even at C=16, ungrouped verify MUST fail closed.
        let ungrouped = qwen38_80b_moe(MoeVerifyKernelKind::UngroupedPerToken);
        let err = ctrl
            .admit(
                &ungrouped,
                ExternalDrafterPresence::Absent,
                DualPlaneAvailability::Available,
                16,
                None,
            )
            .unwrap_err();
        assert_eq!(err, SelfSpecRefusal::MoeVerifyNotGrouped);
    }

    #[test]
    fn moe_self_spec_admits_c8_and_c16_grouped_dispatch() {
        let ctrl = SelfSpecController::new(SelfSpecConfig::enabled_canonical(None));
        let grouped = qwen38_80b_moe(MoeVerifyKernelKind::GroupedByExpert);

        let plan = ctrl
            .admit(
                &grouped,
                ExternalDrafterPresence::Absent,
                DualPlaneAvailability::Available,
                8,
                None,
            )
            .expect("C=8 grouped MoE without DFlash2 must admit");

        match plan.strategy {
            SelfSpecDraftStrategy::MoeWHiTop1Grouped {
                draft_top_k,
                verify_top_k,
                concurrency,
                gamma,
                expected_draft_unique_experts,
                expected_verify_unique_experts,
            } => {
                assert_eq!(draft_top_k, 1);
                assert_eq!(verify_top_k, 10);
                assert_eq!(concurrency, 8);
                assert_eq!(gamma, MOE_SELF_SPEC_DEFAULT_GAMMA);
                // E=512, k=1, C=8 -> ~7.95 unique experts drafted.
                assert!((7.5..=8.0).contains(&expected_draft_unique_experts));
                // E=512, k=10, C*(gamma+1)=32 tokens -> ~239 unique experts verified.
                assert!(expected_verify_unique_experts > 200.0);
                assert!(plan.estimated_draft_bandwidth_reduction > 20.0);
            }
            other => panic!("expected MoeWHiTop1Grouped, got {other:?}"),
        }
    }

    #[test]
    fn dense_self_spec_skips_lowest_la_ffn_layers_and_respects_dflash2() {
        let ctrl = SelfSpecController::new(SelfSpecConfig::enabled_canonical(Some(
            AutoEmaticGateConfig::canonical(),
        )));
        let dense = SelfSpecModelTopology::Dense { num_layers: 8 };

        // External DFlash2 present -> fail closed.
        assert_eq!(
            ctrl.admit(
                &dense,
                ExternalDrafterPresence::DFlash2Installed,
                DualPlaneAvailability::Available,
                1,
                None,
            )
            .unwrap_err(),
            SelfSpecRefusal::ExternalDrafterPresent
        );

        // Missing DualPlane weights -> fail closed.
        assert_eq!(
            ctrl.admit(
                &dense,
                ExternalDrafterPresence::Absent,
                DualPlaneAvailability::Unavailable,
                1,
                None,
            )
            .unwrap_err(),
            SelfSpecRefusal::DualPlaneWeightsUnavailable
        );

        // Boundary layers (0, 1, 6, 7) have high L_A; middle layers (2, 3, 4, 5) have low L_A.
        let profile =
            LayerActivityProfile::try_new(vec![0.95, 0.70, 0.12, 0.08, 0.15, 0.10, 0.65, 0.90])
                .unwrap();
        let plan = ctrl
            .admit(
                &dense,
                ExternalDrafterPresence::Absent,
                DualPlaneAvailability::Available,
                1,
                Some(&profile),
            )
            .expect("dense without DFlash2 must admit");

        match plan.strategy {
            SelfSpecDraftStrategy::DenseWHiWithFfnSkip {
                ffn_schedule,
                skipped_ffn_layers,
                gamma,
            } => {
                assert_eq!(skipped_ffn_layers, 4);
                assert_eq!(gamma, DENSE_SELF_SPEC_DEFAULT_GAMMA);
                assert_eq!(ffn_schedule[0], FfnDraftExecution::ExecuteWHi);
                assert_eq!(ffn_schedule[1], FfnDraftExecution::ExecuteWHi);
                assert_eq!(ffn_schedule[2], FfnDraftExecution::SkipFfn);
                assert_eq!(ffn_schedule[3], FfnDraftExecution::SkipFfn);
                assert_eq!(ffn_schedule[4], FfnDraftExecution::SkipFfn);
                assert_eq!(ffn_schedule[5], FfnDraftExecution::SkipFfn);
                assert_eq!(ffn_schedule[6], FfnDraftExecution::ExecuteWHi);
                assert_eq!(ffn_schedule[7], FfnDraftExecution::ExecuteWHi);
            }
            other => panic!("expected DenseWHiWithFfnSkip, got {other:?}"),
        }
        // Draft reads 4.5*(0.25 + 0.75*0.5) = 2.8125 bpw vs 8.5 bpw verify -> ~3.02x reduction.
        assert!(plan.estimated_draft_bandwidth_reduction > 2.9);

        // Verify Seam 3 entropy gate escalation at 0.10.
        assert_eq!(
            ctrl.verify_precision_for(EntropyGateSignal {
                topk_entropy: 0.05,
                top1_margin: 0.92,
                layer_rel_delta: Some(0.15),
            }),
            AutoEmaticPrecisionLevel::Nvfp4
        );
        assert_eq!(
            ctrl.verify_precision_for(EntropyGateSignal {
                topk_entropy: 0.12,
                top1_margin: 0.45,
                layer_rel_delta: Some(0.55),
            }),
            AutoEmaticPrecisionLevel::AutoFp6
        );
    }
}
