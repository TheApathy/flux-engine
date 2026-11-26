// SPDX-License-Identifier: AGPL-3.0-only

//! `think_nudge.*`: the runtime face of the scheduler's thinking nudge.
//!
//! The nudge config is part of the request-defaults snapshot: a request
//! carries it to the scheduler (`InferenceRequest::think_nudge`), and each
//! sequence keeps the config it was admitted with.

use std::sync::Arc;

use crate::scheduler::think_nudge::{ThinkNudgeConfig, ThinkNudgePolicy};

use super::{SettingError, Source};

/// The nudge policy the settings describe: the launch policy plus the
/// runtime overrides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NudgePolicyView {
    /// The launch policy (`None`: off at launch).
    pub base: Option<ThinkNudgePolicy>,
    pub enabled: Option<bool>,
    pub rate: Option<f64>,
    pub min_tokens: Option<u32>,
    pub max_nudges: Option<u8>,
}

impl NudgePolicyView {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(self.base.is_some())
    }

    pub fn effective_policy(&self) -> ThinkNudgePolicy {
        let mut p = self.base.clone().unwrap_or_default();
        if let Some(rate) = self.rate {
            p.rate = rate;
        }
        if let Some(min_tokens) = self.min_tokens {
            p.min_tokens = min_tokens;
        }
        if let Some(max_nudges) = self.max_nudges {
            p.max_nudges = max_nudges;
        }
        p
    }

    fn has_overrides(&self) -> bool {
        self.enabled.is_some()
            || self.rate.is_some()
            || self.min_tokens.is_some()
            || self.max_nudges.is_some()
    }

    pub fn validate(&self) -> Result<(), SettingError> {
        self.effective_policy()
            .validate()
            .map_err(|e| SettingError::invalid("think_nudge", format!("{e:#}")))
    }
}

/// Owns the installed nudge config for one loaded model.
pub struct NudgeControl {
    /// Token tables of a supported model; `None`: not available here.
    tables: Option<Arc<ThinkNudgeConfig>>,
    startup: NudgePolicyView,
    set_at_startup: bool,
    current: parking_lot::Mutex<NudgePolicyView>,
}

impl NudgeControl {
    pub fn new(
        tables: Option<ThinkNudgeConfig>,
        startup_policy: Option<ThinkNudgePolicy>,
        set_at_startup: bool,
    ) -> Self {
        let startup = NudgePolicyView {
            base: startup_policy,
            ..Default::default()
        };
        Self {
            tables: tables.map(Arc::new),
            current: parking_lot::Mutex::new(startup.clone()),
            startup,
            set_at_startup,
        }
    }

    /// A model with no nudge support (tests, other models).
    pub fn unavailable() -> Self {
        Self::new(None, None, false)
    }

    pub fn startup(&self) -> NudgePolicyView {
        self.startup.clone()
    }

    pub fn effective(&self) -> NudgePolicyView {
        self.current.lock().clone()
    }

    pub fn startup_source(&self) -> Source {
        if self.set_at_startup {
            Source::Startup
        } else {
            Source::Default
        }
    }

    /// The launch config (`None`: off at launch or unsupported).
    pub fn startup_config(&self) -> Option<Arc<ThinkNudgeConfig>> {
        match self.config_for(&self.startup) {
            Ok(config) => config,
            Err(e) => {
                tracing::warn!(
                    reason = e.message(),
                    "Thinking nudge is OFF: the launch policy failed validation"
                );
                None
            }
        }
    }

    /// The config `view` describes, or `None` when it is off. Refuses any
    /// override on a model without the nudge.
    pub fn config_for(
        &self,
        view: &NudgePolicyView,
    ) -> Result<Option<Arc<ThinkNudgeConfig>>, SettingError> {
        let Some(tables) = &self.tables else {
            if view.has_overrides() {
                return Err(SettingError::invalid(
                    "think_nudge",
                    "the thinking nudge is not available for this model (qwen3.8-flash-next and qwen3.8-27b only)",
                ));
            }
            return Ok(None);
        };
        if !view.is_enabled() {
            return Ok(None);
        }
        tables
            .with_policy(view.effective_policy())
            .map(|c| Some(Arc::new(c)))
            .map_err(|e| SettingError::invalid("think_nudge", format!("{e:#}")))
    }

    /// Record the view a publication used (for GET and the next PATCH).
    pub fn set_current(&self, view: NudgePolicyView) {
        *self.current.lock() = view;
    }
}
