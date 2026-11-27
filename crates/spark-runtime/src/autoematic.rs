// SPDX-License-Identifier: AGPL-3.0-only

//! AutoEMatic runtime entropy gate and layer-activity precision escalation policy.
//!
//! Exposes the decoupled [`EntropyGateSignal`] POD consumed across DFlash2,
//! DCOCO, and target verification to escalate NVFP4 layers to AutoFP6
//! (`+0.4 bpw` sparse refinement) only on high-entropy reasoning pivots
//! (arXiv:2611.10165 §3.1). Default-OFF when `ATLAS_AUTOEMATIC_GATE_ENTROPY`
//! is unset so the baseline decode/verify path remains bit-identical.

use std::fmt;

/// Default normalized top-K entropy threshold from AutoEMatic (§5.1, `tau = 0.10`).
pub const DEFAULT_AUTOEMATIC_GATE_ENTROPY: f32 = 0.10;

/// Environment variable configuring the AutoEMatic entropy escalation threshold.
///
/// Unset or empty disables entropy-gated escalation (default OFF).
pub const AUTOEMATIC_GATE_ENTROPY_ENV: &str = "ATLAS_AUTOEMATIC_GATE_ENTROPY";

/// Default decode weighting between local layer update activity (`L_A`) and
/// global generation entropy (`G_E`): `M_D = 0.7 * L_A + 0.3 * G_E`.
pub const AUTOEMATIC_DECODE_LAMBDA: f32 = 0.70;

/// Default consecutive low-activity layers required to lower precision on
/// high-entropy steps (`alpha = 4`, AutoEMatic Table 1).
pub const AUTOEMATIC_TREND_ALPHA: usize = 4;

/// Default consecutive high-activity layers required to raise precision on
/// high-entropy steps (`beta = 2`, AutoEMatic Table 1).
pub const AUTOEMATIC_TREND_BETA: usize = 2;

/// Decoupled cross-track signal carrying next-token uncertainty and optional
/// layer residual-update magnitude for AutoEMatic AutoFP6 escalation.
///
/// Designed as a `Copy` POD so DFlash2/DCOCO proposers and target verifiers
/// can exchange signals without coupling runtime state or allocating.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EntropyGateSignal {
    /// Normalized Shannon entropy of the top-K next-token distribution in `[0.0, 1.0]`.
    pub topk_entropy: f32,
    /// Probability gap `p_top1 - p_top2` in `[0.0, 1.0]`.
    pub top1_margin: f32,
    /// Optional relative residual update `||H_l - H_{l-1}||_2 / (||H_{l-1}||_2 + eps)`.
    pub layer_rel_delta: Option<f32>,
}

impl EntropyGateSignal {
    /// Construct a signal from raw (unnormalized) top-K logits using numerically
    /// stable max-shifted softmax normalized by `ln(K)`.
    ///
    /// Empty or single-element slices yield zero entropy and full margin (`1.0`).
    #[must_use]
    pub fn from_topk_logits(logits: &[f32], layer_rel_delta: Option<f32>) -> Self {
        let mut finite: Vec<f32> = logits.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() <= 1 {
            return Self {
                topk_entropy: 0.0,
                top1_margin: 1.0,
                layer_rel_delta,
            };
        }
        finite.sort_by(|a, b| b.total_cmp(a));
        let max_logit = finite[0];

        let mut probs = Vec::with_capacity(finite.len());
        let mut sum = 0.0_f64;
        for &logit in &finite {
            let weight = f64::from(logit - max_logit).exp();
            probs.push(weight);
            sum += weight;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return Self {
                topk_entropy: 0.0,
                top1_margin: 1.0,
                layer_rel_delta,
            };
        }

        let mut entropy = 0.0_f64;
        for weight in &mut probs {
            *weight /= sum;
            if *weight > 0.0 {
                entropy -= *weight * weight.ln();
            }
        }
        let max_entropy = (probs.len() as f64).ln();
        let norm_entropy = if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        let top1_margin = (probs[0] - probs[1]).clamp(0.0, 1.0) as f32;

        Self {
            topk_entropy: norm_entropy,
            top1_margin,
            layer_rel_delta,
        }
    }

    /// Construct a signal directly from already-normalized top-K probabilities.
    #[must_use]
    pub fn from_topk_probs(probs: &[f32], layer_rel_delta: Option<f32>) -> Self {
        let mut valid: Vec<f64> = probs
            .iter()
            .copied()
            .filter(|p| p.is_finite() && *p >= 0.0)
            .map(f64::from)
            .collect();
        if valid.len() <= 1 {
            return Self {
                topk_entropy: 0.0,
                top1_margin: 1.0,
                layer_rel_delta,
            };
        }
        let sum: f64 = valid.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Self {
                topk_entropy: 0.0,
                top1_margin: 1.0,
                layer_rel_delta,
            };
        }
        valid.sort_by(|a, b| b.total_cmp(a));
        let mut entropy = 0.0_f64;
        for p in &mut valid {
            *p /= sum;
            if *p > 0.0 {
                entropy -= *p * p.ln();
            }
        }
        let max_entropy = (valid.len() as f64).ln();
        let norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0) as f32;
        let top1_margin = (valid[0] - valid[1]).clamp(0.0, 1.0) as f32;
        Self {
            topk_entropy: norm_entropy,
            top1_margin,
            layer_rel_delta,
        }
    }

    /// Returns `true` when `topk_entropy >= threshold` (both finite), indicating
    /// this decode/verify pivot should escalate from NVFP4 to AutoFP6.
    #[must_use]
    pub fn should_escalate_autofp6(self, threshold: f32) -> bool {
        self.topk_entropy.is_finite() && threshold.is_finite() && self.topk_entropy >= threshold
    }

    /// Compute the combined AutoEMatic decode demand metric
    /// `M_D = lambda * L_A + (1 - lambda) * G_E`.
    ///
    /// Falls back to `topk_entropy` when `layer_rel_delta` is absent.
    #[must_use]
    pub fn decode_demand_metric(self, lambda: f32) -> f32 {
        match self.layer_rel_delta {
            Some(la) if la.is_finite() => {
                lambda * la.max(0.0) + (1.0 - lambda) * self.topk_entropy.clamp(0.0, 1.0)
            }
            _ => self.topk_entropy.clamp(0.0, 1.0),
        }
    }
}

/// Error returned when parsing or constructing [`AutoEmaticGateConfig`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutoEmaticGateError {
    /// Environment value is not a valid finite float.
    InvalidFloat(String),
    /// Threshold lies outside the valid normalized entropy range `[0.0, 1.0]`.
    ThresholdOutOfRange(String),
}

impl fmt::Display for AutoEmaticGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFloat(raw) => {
                write!(
                    f,
                    "{AUTOEMATIC_GATE_ENTROPY_ENV} must be a finite float, got {raw:?}"
                )
            }
            Self::ThresholdOutOfRange(raw) => {
                write!(
                    f,
                    "{AUTOEMATIC_GATE_ENTROPY_ENV} must lie within [0.0, 1.0], got {raw}"
                )
            }
        }
    }
}

impl std::error::Error for AutoEmaticGateError {}

/// Validated configuration for entropy-gated AutoEMatic AutoFP6 escalation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoEmaticGateConfig {
    threshold: f32,
}

impl AutoEmaticGateConfig {
    /// Construct a gate with an explicit normalized entropy threshold in `[0.0, 1.0]`.
    ///
    /// # Errors
    /// Returns [`AutoEmaticGateError::ThresholdOutOfRange`] if `threshold` is
    /// NaN, infinite, negative, or greater than `1.0`.
    pub fn try_new(threshold: f32) -> Result<Self, AutoEmaticGateError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(AutoEmaticGateError::ThresholdOutOfRange(
                threshold.to_string(),
            ));
        }
        Ok(Self { threshold })
    }

    /// Construct a gate at the canonical AutoEMatic threshold (`0.10`).
    #[must_use]
    pub const fn canonical() -> Self {
        Self {
            threshold: DEFAULT_AUTOEMATIC_GATE_ENTROPY,
        }
    }

    /// Parse `ATLAS_AUTOEMATIC_GATE_ENTROPY` from the process environment.
    ///
    /// Returns `Ok(None)` when unset or empty (default OFF).
    ///
    /// # Errors
    /// Fails closed on non-numeric, NaN, infinite, or out-of-range values.
    pub fn from_env() -> Result<Option<Self>, AutoEmaticGateError> {
        let raw = std::env::var(AUTOEMATIC_GATE_ENTROPY_ENV).ok();
        Self::parse_optional(raw.as_deref())
    }

    /// Parse an optional string value (used by both [`Self::from_env`] and tests).
    ///
    /// # Errors
    /// Fails closed on invalid floats or values outside `[0.0, 1.0]`.
    pub fn parse_optional(raw: Option<&str>) -> Result<Option<Self>, AutoEmaticGateError> {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        let parsed: f32 = trimmed
            .parse()
            .map_err(|_| AutoEmaticGateError::InvalidFloat(trimmed.to_owned()))?;
        Self::try_new(parsed).map(Some)
    }

    /// Configured normalized entropy escalation threshold.
    #[must_use]
    pub const fn threshold(self) -> f32 {
        self.threshold
    }

    /// Evaluate whether `signal` escalates to AutoFP6 under this gate.
    #[must_use]
    pub fn evaluate(self, signal: EntropyGateSignal) -> AutoEmaticPrecisionLevel {
        if signal.should_escalate_autofp6(self.threshold) {
            AutoEmaticPrecisionLevel::AutoFp6
        } else {
            AutoEmaticPrecisionLevel::Nvfp4
        }
    }
}

/// Active precision level selected by AutoEMatic for a layer execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoEmaticPrecisionLevel {
    /// Base 4.5 bpw NVFP4 execution (fast path).
    Nvfp4,
    /// Escalated 4.9 bpw AutoFP6 execution via Extended-K NVFP4 Tensor Cores.
    AutoFp6,
}

/// Layer-by-layer trend tracker implementing AutoEMatic's sustained-change rule
/// (Table 1 of arXiv:2611.10165) to filter isolated single-layer noise blips.
#[derive(Clone, Debug)]
pub struct AutoEmaticLayerTracker {
    gate: AutoEmaticGateConfig,
    current: AutoEmaticPrecisionLevel,
    running_mean: f64,
    samples: u64,
    above_streak: usize,
    below_streak: usize,
}

impl AutoEmaticLayerTracker {
    /// Create a fresh tracker starting layer 0 at [`AutoEmaticPrecisionLevel::Nvfp4`].
    #[must_use]
    pub const fn new(gate: AutoEmaticGateConfig) -> Self {
        Self {
            gate,
            current: AutoEmaticPrecisionLevel::Nvfp4,
            running_mean: 0.0,
            samples: 0,
            above_streak: 0,
            below_streak: 0,
        }
    }

    /// Reset per-step layer streaks before starting layer 0 of a new decode step.
    pub fn begin_decode_step(&mut self) {
        self.current = AutoEmaticPrecisionLevel::Nvfp4;
        self.above_streak = 0;
        self.below_streak = 0;
    }

    /// Current selected precision level.
    #[must_use]
    pub const fn current_level(&self) -> AutoEmaticPrecisionLevel {
        self.current
    }

    /// Observe one layer's [`EntropyGateSignal`] and return the precision level
    /// selected for the next layer.
    ///
    /// On high-entropy steps (`G_E >= tau`), requires `beta = 2` consecutive
    /// above-mean layers to escalate and `alpha = 4` below-mean layers to drop.
    /// On low-entropy steps, thresholds reverse (`alpha = 4` to raise, `beta = 2`
    /// to drop).
    pub fn observe_layer(&mut self, signal: EntropyGateSignal) -> AutoEmaticPrecisionLevel {
        let metric = f64::from(signal.decode_demand_metric(AUTOEMATIC_DECODE_LAMBDA));
        let prev_mean = if self.samples == 0 {
            metric
        } else {
            self.running_mean
        };
        self.samples += 1;
        self.running_mean += (metric - self.running_mean) / (self.samples as f64);

        let high_entropy = signal.should_escalate_autofp6(self.gate.threshold());
        let (raise_need, lower_need) = if high_entropy {
            (AUTOEMATIC_TREND_BETA, AUTOEMATIC_TREND_ALPHA)
        } else {
            (AUTOEMATIC_TREND_ALPHA, AUTOEMATIC_TREND_BETA)
        };

        if metric > prev_mean {
            self.above_streak += 1;
            self.below_streak = 0;
            if self.above_streak >= raise_need {
                self.current = AutoEmaticPrecisionLevel::AutoFp6;
            }
        } else if metric < prev_mean {
            self.below_streak += 1;
            self.above_streak = 0;
            if self.below_streak >= lower_need {
                self.current = AutoEmaticPrecisionLevel::Nvfp4;
            }
        }
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_signal_uniform_vs_peaked_logits() {
        let uniform = EntropyGateSignal::from_topk_logits(&[2.0; 10], None);
        assert!(
            (uniform.topk_entropy - 1.0).abs() < 1e-5,
            "uniform top-10 logits must yield normalized entropy 1.0, got {}",
            uniform.topk_entropy
        );
        assert!(uniform.top1_margin.abs() < 1e-5);
        assert!(uniform.should_escalate_autofp6(DEFAULT_AUTOEMATIC_GATE_ENTROPY));

        let mut peaked_logits = [-20.0_f32; 10];
        peaked_logits[0] = 20.0;
        let peaked = EntropyGateSignal::from_topk_logits(&peaked_logits, Some(0.05));
        assert!(
            peaked.topk_entropy < 1e-4,
            "peaked logits must yield near-zero entropy, got {}",
            peaked.topk_entropy
        );
        assert!(peaked.top1_margin > 0.9999);
        assert!(!peaked.should_escalate_autofp6(DEFAULT_AUTOEMATIC_GATE_ENTROPY));
    }

    #[test]
    fn entropy_gate_threshold_boundary_and_fail_closed_parsing() {
        assert_eq!(AutoEmaticGateConfig::parse_optional(None).unwrap(), None);
        assert_eq!(
            AutoEmaticGateConfig::parse_optional(Some("   ")).unwrap(),
            None
        );

        let cfg = AutoEmaticGateConfig::parse_optional(Some("0.10"))
            .unwrap()
            .expect("valid threshold must parse");
        assert!((cfg.threshold() - 0.10).abs() < 1e-7);

        let below = EntropyGateSignal {
            topk_entropy: 0.099,
            top1_margin: 0.95,
            layer_rel_delta: None,
        };
        let at_thresh = EntropyGateSignal {
            topk_entropy: 0.10,
            top1_margin: 0.85,
            layer_rel_delta: None,
        };
        assert_eq!(cfg.evaluate(below), AutoEmaticPrecisionLevel::Nvfp4);
        assert_eq!(cfg.evaluate(at_thresh), AutoEmaticPrecisionLevel::AutoFp6);

        assert!(AutoEmaticGateConfig::parse_optional(Some("NaN")).is_err());
        assert!(AutoEmaticGateConfig::parse_optional(Some("-0.01")).is_err());
        assert!(AutoEmaticGateConfig::parse_optional(Some("1.05")).is_err());
        assert!(AutoEmaticGateConfig::parse_optional(Some("not_a_float")).is_err());
    }

    #[test]
    fn trend_tracker_ignores_single_layer_blip_and_escalates_on_sustained_high_entropy() {
        let mut tracker = AutoEmaticLayerTracker::new(AutoEmaticGateConfig::canonical());
        // Seed baseline mean around 0.20 with low-entropy observations.
        for _ in 0..8 {
            tracker.observe_layer(EntropyGateSignal {
                topk_entropy: 0.05,
                top1_margin: 0.95,
                layer_rel_delta: Some(0.25),
            });
        }
        tracker.begin_decode_step();

        // Single high-activity blip on high-entropy step must not flip yet (beta = 2).
        let high = EntropyGateSignal {
            topk_entropy: 0.40,
            top1_margin: 0.30,
            layer_rel_delta: Some(0.60),
        };
        assert_eq!(tracker.observe_layer(high), AutoEmaticPrecisionLevel::Nvfp4);
        // Second consecutive high-activity layer escalates to AutoFP6.
        assert_eq!(
            tracker.observe_layer(high),
            AutoEmaticPrecisionLevel::AutoFp6
        );
    }
}
