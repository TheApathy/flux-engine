// SPDX-License-Identifier: AGPL-3.0-only

//! The adjustable keys: names, validation, how each applies to the request
//! defaults, and how each is read back.

use serde_json::Value;

use super::nudge::NudgePolicyView;
use super::{RequestDefaults, Source, StartupDefaults};

/// Settings that need a reload or a reallocation. A PATCH naming one is
/// refused with `restart_required`.
pub const RESTART_ONLY: &[&str] = &[
    "model",
    "kv_cache_dtype",
    "max_seq_len",
    "context_length",
    "max_batch_size",
    "concurrency",
    "speculative",
    "speculation",
    "num_drafts",
    "dflash",
    "mtp_gate",
    "gpu_memory_utilization",
    "max_prefill_tokens",
    "tool_max_tokens",
    "thinking.disable",
];

/// The presets a chat request selects from (`thinking_coding` is never
/// selected, so it is not offered).
const PRESETS: [&str; 3] = ["thinking_text", "non_thinking", "tools"];
const SAMPLING_FIELDS: [&str; 7] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
];
pub(super) const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Upper bound for every token-count setting.
const MAX_TOKENS_SETTING: u64 = 1 << 20;

/// A refused setting change.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingError {
    /// Unknown key or invalid value (400).
    Invalid { key: String, message: String },
    /// Restart-only setting (409).
    RestartRequired { key: String, message: String },
    /// The change could not be recorded (500).
    Internal { message: String },
}

impl SettingError {
    pub(super) fn invalid(key: &str, message: impl Into<String>) -> Self {
        Self::Invalid {
            key: key.to_string(),
            message: message.into(),
        }
    }

    pub(super) fn restart(key: &str, message: impl Into<String>) -> Self {
        Self::RestartRequired {
            key: key.to_string(),
            message: message.into(),
        }
    }

    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Invalid { message, .. }
            | Self::RestartRequired { message, .. }
            | Self::Internal { message } => message,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SamplingField {
    Temperature,
    TopP,
    TopK,
    MinP,
    PresencePenalty,
    FrequencyPenalty,
    RepetitionPenalty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Key {
    ThinkingDefault,
    ThinkingMaxBudget,
    ThinkingAnswerReserve,
    EffortBudget(&'static str),
    Sampling(&'static str, SamplingField),
    MaxTokensDefault,
    NudgeEnabled,
    NudgeRate,
    NudgeMinTokens,
    NudgeMax,
}

/// Every adjustable key, in display order.
pub(super) fn all_keys() -> Vec<Key> {
    let mut keys = vec![
        Key::ThinkingDefault,
        Key::ThinkingMaxBudget,
        Key::ThinkingAnswerReserve,
    ];
    keys.extend(EFFORTS.iter().map(|e| Key::EffortBudget(e)));
    for preset in PRESETS {
        for field in SAMPLING_FIELDS {
            keys.push(Key::Sampling(
                preset,
                sampling_field(field).expect("known field"),
            ));
        }
    }
    keys.extend([
        Key::MaxTokensDefault,
        Key::NudgeEnabled,
        Key::NudgeRate,
        Key::NudgeMinTokens,
        Key::NudgeMax,
    ]);
    keys
}

fn sampling_field(name: &str) -> Option<SamplingField> {
    Some(match name {
        "temperature" => SamplingField::Temperature,
        "top_p" => SamplingField::TopP,
        "top_k" => SamplingField::TopK,
        "min_p" => SamplingField::MinP,
        "presence_penalty" => SamplingField::PresencePenalty,
        "frequency_penalty" => SamplingField::FrequencyPenalty,
        "repetition_penalty" => SamplingField::RepetitionPenalty,
        _ => return None,
    })
}

fn sampling_field_name(field: SamplingField) -> &'static str {
    match field {
        SamplingField::Temperature => "temperature",
        SamplingField::TopP => "top_p",
        SamplingField::TopK => "top_k",
        SamplingField::MinP => "min_p",
        SamplingField::PresencePenalty => "presence_penalty",
        SamplingField::FrequencyPenalty => "frequency_penalty",
        SamplingField::RepetitionPenalty => "repetition_penalty",
    }
}

/// Resolve a PATCH key, refusing restart-only and unknown names.
pub(super) fn parse_key(name: &str, startup: &StartupDefaults) -> Result<Key, SettingError> {
    let root = name.split('.').next().unwrap_or(name);
    if RESTART_ONLY.contains(&name) || RESTART_ONLY.contains(&root) {
        return Err(SettingError::restart(
            name,
            format!(
                "'{name}' needs a restart (reload or reallocation); it cannot change at runtime"
            ),
        ));
    }
    if name.starts_with("watchdog.") {
        return Err(SettingError::restart(
            name,
            "loop-watchdog thresholds are not runtime-adjustable yet; set them at launch",
        ));
    }
    let key = match name {
        "thinking.default" => Key::ThinkingDefault,
        "thinking.max_budget" => Key::ThinkingMaxBudget,
        "thinking.answer_reserve" => Key::ThinkingAnswerReserve,
        "max_tokens.default" => Key::MaxTokensDefault,
        "think_nudge.enabled" => Key::NudgeEnabled,
        "think_nudge.rate" => Key::NudgeRate,
        "think_nudge.min_tokens" => Key::NudgeMinTokens,
        "think_nudge.max" => Key::NudgeMax,
        _ => parse_compound(name).ok_or_else(|| {
            SettingError::invalid(
                name,
                format!("unknown setting '{name}' (GET /v1/flux/settings lists them)"),
            )
        })?,
    };
    if key == Key::ThinkingDefault && startup.disable_thinking {
        return Err(SettingError::restart(
            name,
            "the server was started with --disable-thinking; restart without it to change the thinking default",
        ));
    }
    Ok(key)
}

fn parse_compound(name: &str) -> Option<Key> {
    if let Some(effort) = name.strip_prefix("thinking.effort_budget.") {
        return EFFORTS
            .iter()
            .find(|e| **e == effort)
            .map(|e| Key::EffortBudget(e));
    }
    let rest = name.strip_prefix("sampling.")?;
    let (preset, field) = rest.split_once('.')?;
    let preset = PRESETS.iter().find(|p| **p == preset)?;
    Some(Key::Sampling(preset, sampling_field(field)?))
}

fn as_bool(key: &str, v: &Value) -> Result<bool, SettingError> {
    v.as_bool()
        .ok_or_else(|| SettingError::invalid(key, format!("'{key}' must be true or false")))
}

fn as_count(key: &str, v: &Value, min: u64, max: u64) -> Result<u64, SettingError> {
    match v.as_u64() {
        Some(n) if (min..=max).contains(&n) => Ok(n),
        _ => Err(SettingError::invalid(
            key,
            format!("'{key}' must be an integer in {min}..={max}"),
        )),
    }
}

fn as_f64(
    key: &str,
    v: &Value,
    ok: impl Fn(f64) -> bool,
    range: &str,
) -> Result<f64, SettingError> {
    match v.as_f64() {
        Some(x) if x.is_finite() && ok(x) => Ok(x),
        _ => Err(SettingError::invalid(
            key,
            format!("'{key}' must be a number in {range}"),
        )),
    }
}

fn as_float(
    key: &str,
    v: &Value,
    ok: impl Fn(f64) -> bool,
    range: &str,
) -> Result<f32, SettingError> {
    as_f64(key, v, ok, range).map(|x| x as f32)
}

impl Key {
    pub(super) fn name(self) -> String {
        match self {
            Key::ThinkingDefault => "thinking.default".into(),
            Key::ThinkingMaxBudget => "thinking.max_budget".into(),
            Key::ThinkingAnswerReserve => "thinking.answer_reserve".into(),
            Key::EffortBudget(e) => format!("thinking.effort_budget.{e}"),
            Key::Sampling(p, f) => format!("sampling.{p}.{}", sampling_field_name(f)),
            Key::MaxTokensDefault => "max_tokens.default".into(),
            Key::NudgeEnabled => "think_nudge.enabled".into(),
            Key::NudgeRate => "think_nudge.rate".into(),
            Key::NudgeMinTokens => "think_nudge.min_tokens".into(),
            Key::NudgeMax => "think_nudge.max".into(),
        }
    }

    /// Type and range check of a non-null value.
    pub(super) fn validate(self, v: &Value, startup: &StartupDefaults) -> Result<(), SettingError> {
        let mut defaults = startup.defaults.clone();
        let mut nudge = NudgePolicyView::default();
        self.apply(v, &mut defaults, &mut nudge, startup)
    }

    /// Apply a value to the defaults being built (validating it again).
    pub(super) fn apply(
        self,
        v: &Value,
        d: &mut RequestDefaults,
        nudge: &mut NudgePolicyView,
        startup: &StartupDefaults,
    ) -> Result<(), SettingError> {
        let name = self.name();
        let key = name.as_str();
        match self {
            Key::ThinkingDefault => {
                d.thinking_default = as_bool(key, v)?;
                d.thinking_default_runtime = true;
            }
            Key::ThinkingMaxBudget => {
                d.max_thinking_budget = as_count(key, v, 1, MAX_TOKENS_SETTING)? as u32;
            }
            Key::ThinkingAnswerReserve => {
                d.thinking_answer_reserve = as_count(key, v, 0, MAX_TOKENS_SETTING)? as u32;
            }
            Key::EffortBudget(effort) => {
                d.effort_budgets
                    .insert(effort, as_count(key, v, 0, MAX_TOKENS_SETTING)? as u32);
            }
            Key::MaxTokensDefault => {
                let max = (startup.max_seq_len as u64).min(MAX_TOKENS_SETTING);
                d.default_max_tokens = Some(as_count(key, v, 1, max)? as usize);
            }
            Key::Sampling(preset, field) => {
                let cat = preset_mut(&mut d.sampling_presets, preset);
                match field {
                    SamplingField::Temperature => {
                        cat.temperature = as_float(key, v, |x| (0.0..=2.0).contains(&x), "[0, 2]")?
                    }
                    SamplingField::TopP => {
                        cat.top_p = as_float(key, v, |x| x > 0.0 && x <= 1.0, "(0, 1]")?
                    }
                    SamplingField::TopK => cat.top_k = as_count(key, v, 0, 1 << 20)? as u32,
                    SamplingField::MinP => {
                        cat.min_p = Some(as_float(key, v, |x| (0.0..=1.0).contains(&x), "[0, 1]")?)
                    }
                    SamplingField::PresencePenalty => {
                        cat.presence_penalty =
                            as_float(key, v, |x| (-2.0..=2.0).contains(&x), "[-2, 2]")?
                    }
                    SamplingField::FrequencyPenalty => {
                        cat.frequency_penalty =
                            as_float(key, v, |x| (-2.0..=2.0).contains(&x), "[-2, 2]")?
                    }
                    SamplingField::RepetitionPenalty => {
                        cat.repetition_penalty =
                            as_float(key, v, |x| x > 0.0 && x <= 10.0, "(0, 10]")?
                    }
                }
            }
            Key::NudgeEnabled => nudge.enabled = Some(as_bool(key, v)?),
            Key::NudgeRate => nudge.rate = Some(as_f64(key, v, |x| x > 0.0 && x <= 1.0, "(0, 1]")?),
            Key::NudgeMinTokens => {
                nudge.min_tokens = Some(as_count(key, v, 0, MAX_TOKENS_SETTING)? as u32)
            }
            Key::NudgeMax => nudge.max_nudges = Some(as_count(key, v, 0, 2)? as u8),
        }
        Ok(())
    }

    /// The effective value as JSON (`null`: not set / the model's fallback).
    pub(super) fn read(
        self,
        d: &RequestDefaults,
        startup: &StartupDefaults,
        nudge: &NudgePolicyView,
    ) -> Value {
        match self {
            Key::ThinkingDefault => Value::from(d.thinking_default),
            Key::ThinkingMaxBudget => Value::from(d.max_thinking_budget),
            Key::ThinkingAnswerReserve => Value::from(d.thinking_answer_reserve),
            Key::EffortBudget(effort) => d
                .effort_budget(effort, startup.effort_table)
                .map_or(Value::Null, Value::from),
            Key::MaxTokensDefault => d.default_max_tokens.map_or(Value::Null, Value::from),
            Key::Sampling(preset, field) => {
                let cat = preset_ref(&d.sampling_presets, preset);
                let f = |x: f32| {
                    // Round-trip through the f32 text so 0.95 reads as 0.95.
                    serde_json::from_str::<Value>(&x.to_string()).unwrap_or(Value::Null)
                };
                match field {
                    SamplingField::Temperature => f(cat.temperature),
                    SamplingField::TopP => f(cat.top_p),
                    SamplingField::TopK => Value::from(cat.top_k),
                    SamplingField::MinP => cat.min_p.map_or(Value::Null, f),
                    SamplingField::PresencePenalty => f(cat.presence_penalty),
                    SamplingField::FrequencyPenalty => f(cat.frequency_penalty),
                    SamplingField::RepetitionPenalty => f(cat.repetition_penalty),
                }
            }
            Key::NudgeEnabled => Value::from(nudge.is_enabled()),
            Key::NudgeRate => Value::from(nudge.effective_policy().rate),
            Key::NudgeMinTokens => Value::from(nudge.effective_policy().min_tokens),
            Key::NudgeMax => Value::from(nudge.effective_policy().max_nudges),
        }
    }

    pub(super) fn startup_source(
        self,
        startup: &StartupDefaults,
        nudge: &super::nudge::NudgeControl,
    ) -> Source {
        match self {
            Key::ThinkingDefault => startup.thinking_default_source,
            Key::ThinkingMaxBudget => startup.max_thinking_budget_source,
            Key::MaxTokensDefault => startup.default_max_tokens_source,
            Key::NudgeEnabled | Key::NudgeRate | Key::NudgeMinTokens | Key::NudgeMax => {
                nudge.startup_source()
            }
            _ => Source::Default,
        }
    }
}

fn preset_mut<'a>(
    p: &'a mut atlas_kernels::SamplingPresets,
    preset: &str,
) -> &'a mut atlas_kernels::SamplingCategory {
    match preset {
        "thinking_text" => &mut p.thinking_text,
        "non_thinking" => &mut p.non_thinking,
        _ => &mut p.tools,
    }
}

fn preset_ref<'a>(
    p: &'a atlas_kernels::SamplingPresets,
    preset: &str,
) -> &'a atlas_kernels::SamplingCategory {
    match preset {
        "thinking_text" => &p.thinking_text,
        "non_thinking" => &p.non_thinking,
        _ => &p.tools,
    }
}
