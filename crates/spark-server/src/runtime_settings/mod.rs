// SPDX-License-Identifier: AGPL-3.0-only

//! Live-adjustable server defaults (`GET`/`PATCH /v1/flux/settings`).
//!
//! The settings are request DEFAULTS: thinking on/off and budgets, the
//! effort→budget table, per-preset sampling, the default `max_tokens`, and
//! the thinking nudge. A request field always wins over them, through the
//! same `req.x.unwrap_or(default)` paths as before.
//!
//! Each loaded model owns one [`RuntimeSettings`] (an `AppState` field), so
//! a model swap starts from the new model's startup values. A request takes
//! ONE snapshot ([`RuntimeSettings::snapshot`], an `Arc` clone) at admission
//! and resolves everything from it, so a change never reaches a request that
//! is already in flight. With no change applied the snapshot carries exactly
//! the startup values, so serving is byte-identical to a build without this
//! module.
//!
//! Anything that needs a reload or a reallocation (KV dtype, context length,
//! concurrency, speculation) is restart-only and is refused by name.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

mod catalog;
pub mod handlers;
pub mod local_admin;
mod nudge;
mod persist;

use catalog::parse_key;
pub use catalog::{RESTART_ONLY, SettingError};
pub use nudge::NudgeControl;
pub use persist::SettingsFile;

/// Where an effective value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// MODEL.toml, generation_config or a built-in table.
    Default,
    /// A launch flag or environment variable.
    Startup,
    /// A `PATCH /v1/flux/settings` (or the persisted settings file).
    Runtime,
}

/// The request defaults one request resolves from. Built once per change,
/// shared by `Arc`, never mutated after publication.
#[derive(Debug, Clone)]
pub struct RequestDefaults {
    pub thinking_default: bool,
    /// The thinking default was set at runtime: it replaces the launch
    /// `--default-chat-template-kwargs` `enable_thinking`, which would
    /// otherwise be injected into every silent request and win.
    pub thinking_default_runtime: bool,
    pub max_thinking_budget: u32,
    pub thinking_answer_reserve: u32,
    /// Runtime effort→budget overrides (`thinking.effort_budget.<effort>`).
    /// Efforts not listed use the model's built-in table.
    pub effort_budgets: BTreeMap<&'static str, u32>,
    pub sampling_presets: atlas_kernels::SamplingPresets,
    pub default_max_tokens: Option<usize>,
    /// The thinking-nudge config new sequences get (`None`: off). It is
    /// published in the same snapshot as everything else, and the request
    /// carries it to the scheduler, so a request never mixes a new nudge
    /// with old defaults (or the reverse).
    pub think_nudge: Option<Arc<crate::scheduler::think_nudge::ThinkNudgeConfig>>,
    /// Keys a PATCH set (sorted), for the `Effective request:` log line.
    /// Empty when no runtime setting applies.
    pub runtime_keys: Vec<String>,
}

impl RequestDefaults {
    /// The effort budget for `effort`, a runtime override first, then the
    /// model's table.
    pub fn effort_budget(&self, effort: &str, table: fn(&str) -> Option<u32>) -> Option<u32> {
        self.effort_budgets
            .get(effort)
            .copied()
            .or_else(|| table(effort))
    }

    /// The model default and launch kwargs thinking resolves from: a runtime
    /// thinking default replaces both the model default and the launch
    /// kwargs' `enable_thinking`; otherwise both stand as before.
    pub fn thinking_inputs(
        &self,
        model_default: bool,
        launch_kwargs: Option<&crate::openai::ChatTemplateKwargs>,
    ) -> (bool, Option<crate::openai::ChatTemplateKwargs>) {
        let mut kwargs = launch_kwargs.cloned();
        if !self.thinking_default_runtime {
            return (model_default, kwargs);
        }
        if let Some(kw) = kwargs.as_mut() {
            kw.enable_thinking = None;
        }
        (self.thinking_default, kwargs)
    }

    /// ` runtime=[k1,k2]` for the request log line, or empty.
    pub fn log_suffix(&self) -> String {
        if self.runtime_keys.is_empty() {
            String::new()
        } else {
            format!(", runtime=[{}]", self.runtime_keys.join(","))
        }
    }
}

/// The startup values and where each came from.
#[derive(Debug, Clone)]
pub struct StartupDefaults {
    pub defaults: RequestDefaults,
    /// `--disable-thinking`: thinking is forced off and the thinking default
    /// cannot be changed at runtime.
    pub disable_thinking: bool,
    pub thinking_default_source: Source,
    pub max_thinking_budget_source: Source,
    pub default_max_tokens_source: Source,
    /// The context length, the upper bound of `max_tokens.default`.
    pub max_seq_len: usize,
    /// The built-in effort table of this model (for display).
    pub effort_table: fn(&str) -> Option<u32>,
}

impl StartupDefaults {
    /// Startup values from the loaded model's resolved state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        behavior: &atlas_kernels::ModelBehavior,
        cli_max_thinking_budget: bool,
        sampling_presets: atlas_kernels::SamplingPresets,
        default_chat_template_kwargs: Option<&crate::openai::ChatTemplateKwargs>,
        disable_thinking: bool,
        default_max_tokens: Option<usize>,
        max_seq_len: usize,
        effort_table: fn(&str) -> Option<u32>,
    ) -> Self {
        let startup_thinking = default_chat_template_kwargs.and_then(|kw| kw.enable_thinking);
        let source = |set: bool| {
            if set {
                Source::Startup
            } else {
                Source::Default
            }
        };
        Self {
            defaults: RequestDefaults {
                thinking_default: !disable_thinking
                    && startup_thinking.unwrap_or(behavior.thinking_default),
                thinking_default_runtime: false,
                max_thinking_budget: behavior.max_thinking_budget,
                thinking_answer_reserve: behavior.thinking_answer_reserve,
                effort_budgets: BTreeMap::new(),
                sampling_presets,
                default_max_tokens,
                think_nudge: None,
                runtime_keys: Vec::new(),
            },
            disable_thinking,
            thinking_default_source: source(startup_thinking.is_some() || disable_thinking),
            max_thinking_budget_source: source(cli_max_thinking_budget),
            default_max_tokens_source: source(default_max_tokens.is_some()),
            max_seq_len,
            effort_table,
        }
    }
}

/// One applied change, for the log and the PATCH response.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Change {
    pub key: String,
    pub old: Value,
    pub new: Value,
}

struct Inner {
    /// The validated runtime overlay: key → JSON value.
    overlay: BTreeMap<String, Value>,
}

/// The live settings of one loaded model.
pub struct RuntimeSettings {
    startup: StartupDefaults,
    published: parking_lot::RwLock<Arc<RequestDefaults>>,
    /// Serialises writers (PATCH, file load) so two concurrent changes can
    /// never lose one another. Readers never take it.
    writer: parking_lot::Mutex<Inner>,
    nudge: NudgeControl,
    persist: Option<SettingsFile>,
    model_name: String,
}

impl RuntimeSettings {
    pub fn new(
        mut startup: StartupDefaults,
        nudge: NudgeControl,
        persist: Option<SettingsFile>,
        model_name: String,
    ) -> Self {
        startup.defaults.think_nudge = nudge.startup_config();
        let published = parking_lot::RwLock::new(Arc::new(startup.defaults.clone()));
        Self {
            startup,
            published,
            writer: parking_lot::Mutex::new(Inner {
                overlay: BTreeMap::new(),
            }),
            nudge,
            persist,
            model_name,
        }
    }

    /// Startup values of a default `ModelBehavior` and presets, no nudge and
    /// no settings file: for tests that build an `AppState` by hand.
    #[cfg(test)]
    pub fn for_tests(max_seq_len: usize) -> Self {
        Self::new(
            StartupDefaults::new(
                &atlas_kernels::ModelBehavior::default(),
                false,
                atlas_kernels::SamplingPresets::default(),
                None,
                false,
                None,
                max_seq_len,
                crate::openai::generic_effort_thinking_budget,
            ),
            NudgeControl::unavailable(),
            None,
            String::new(),
        )
    }

    /// The defaults for one request. Take it once, at admission.
    pub fn snapshot(&self) -> Arc<RequestDefaults> {
        Arc::clone(&self.published.read())
    }

    /// Every adjustable key with its effective value and source.
    pub fn describe(&self) -> Value {
        let overlay = self.writer.lock().overlay.clone();
        let defaults = self.snapshot();
        let nudge = self.nudge.effective();
        let mut settings = serde_json::Map::new();
        for key in catalog::all_keys() {
            let name = key.name();
            let value = key.read(&defaults, &self.startup, &nudge);
            let source = if overlay.contains_key(&name) {
                Source::Runtime
            } else {
                key.startup_source(&self.startup, &self.nudge)
            };
            settings.insert(
                name,
                serde_json::json!({ "value": value, "source": source }),
            );
        }
        serde_json::json!({
            "object": "flux.settings",
            "model": self.model_name,
            "settings": settings,
            "restart_only": RESTART_ONLY,
            "persist_file": self.persist.as_ref().map(|p| p.path().display().to_string()),
        })
    }

    /// Validate and apply a PATCH body atomically: every key is checked
    /// before anything changes, and the new defaults are published in one
    /// swap. `null` resets a key to its startup value.
    pub fn apply_patch(&self, body: &Value) -> Result<Vec<Change>, SettingError> {
        let Some(entries) = body.as_object() else {
            return Err(SettingError::invalid(
                "",
                "the body must be a JSON object of setting keys",
            ));
        };
        if entries.is_empty() {
            return Err(SettingError::invalid("", "no settings given"));
        }
        let mut inner = self.writer.lock();
        let mut overlay = inner.overlay.clone();
        let mut parsed = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            let key = parse_key(name, &self.startup)?;
            if value.is_null() {
                overlay.remove(name);
            } else {
                key.validate(value, &self.startup)?;
                overlay.insert(name.clone(), value.clone());
            }
            parsed.push(key);
        }
        let before = self.snapshot();
        let nudge_before = self.nudge.effective();
        let (defaults, nudge_after) = self.build(&overlay)?;
        let changes: Vec<Change> = parsed
            .iter()
            .map(|key| Change {
                key: key.name(),
                old: key.read(&before, &self.startup, &nudge_before),
                new: key.read(&defaults, &self.startup, &nudge_after),
            })
            .filter(|c| c.old != c.new || overlay.contains_key(&c.key))
            .collect();
        // Persist before publishing: a change that cannot be recorded is not
        // applied.
        if let Some(file) = &self.persist {
            file.write(&self.model_name, &overlay)
                .map_err(|e| SettingError::internal(format!("settings file: {e:#}")))?;
        }
        // One publication: the nudge config travels inside `defaults`.
        *self.published.write() = Arc::new(defaults);
        self.nudge.set_current(nudge_after);
        inner.overlay = overlay;
        Ok(changes)
    }

    /// Reapply the persisted settings at startup (when a file is configured).
    pub fn load_persisted(&self) -> anyhow::Result<()> {
        let Some(file) = &self.persist else {
            return Ok(());
        };
        let Some(overlay) = file.read(&self.model_name)? else {
            return Ok(());
        };
        // A key this build no longer knows (renamed, removed, or restart-only
        // now) is skipped with a warning; an invalid VALUE fails the load.
        let overlay: BTreeMap<String, Value> = overlay
            .into_iter()
            .filter(|(name, _)| match parse_key(name, &self.startup) {
                Ok(_) => true,
                Err(e) => {
                    tracing::warn!(
                        key = %name,
                        file = %file.path().display(),
                        reason = e.message(),
                        "Settings file key skipped"
                    );
                    false
                }
            })
            .collect();
        if overlay.is_empty() {
            return Ok(());
        }
        let body = Value::Object(overlay.into_iter().collect());
        let changes = self
            .apply_patch(&body)
            .map_err(|e| anyhow::anyhow!("{}: {}", file.path().display(), e.message()))?;
        for c in &changes {
            tracing::info!(
                key = %c.key,
                old = %c.old,
                new = %c.new,
                file = %file.path().display(),
                "Runtime setting restored from the settings file"
            );
        }
        Ok(())
    }

    fn build(
        &self,
        overlay: &BTreeMap<String, Value>,
    ) -> Result<(RequestDefaults, nudge::NudgePolicyView), SettingError> {
        let mut defaults = self.startup.defaults.clone();
        let mut nudge = self.nudge.startup();
        for (name, value) in overlay {
            parse_key(name, &self.startup)?.apply(
                value,
                &mut defaults,
                &mut nudge,
                &self.startup,
            )?;
        }
        defaults.runtime_keys = overlay.keys().cloned().collect();
        nudge.validate()?;
        defaults.think_nudge = self.nudge.config_for(&nudge)?;
        Ok((defaults, nudge))
    }
}

/// This request's defaults snapshot: taken from `settings` the first time,
/// the same `Arc` on every later call. Call it at admission, before any
/// default is read.
pub fn pin(
    req: &mut crate::openai::ChatCompletionRequest,
    settings: &RuntimeSettings,
) -> Arc<RequestDefaults> {
    Arc::clone(
        req.runtime_defaults
            .get_or_insert_with(|| settings.snapshot()),
    )
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_admin;
#[cfg(test)]
mod tests_gate;
#[cfg(test)]
mod tests_nudge;
#[cfg(test)]
mod tests_persist;
