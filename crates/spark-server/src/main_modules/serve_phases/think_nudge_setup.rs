// SPDX-License-Identifier: AGPL-3.0-only

//! Resolve the thinking nudge (`FLUX_THINK_NUDGE`, default off; opt-in via
//! `FLUX_THINK_NUDGE=on` for qualified Qwen3.8 targets) for the model being
//! loaded. The result seeds the runtime settings, whose request snapshots carry
//! the config to the scheduler, so a swap to an unsupported model, or a
//! relaunch with `FLUX_THINK_NUDGE=off`, turns it off.

use anyhow::Result;

use crate::scheduler::think_nudge::{self, NewlineClass, ThinkNudgeConfig, ThinkNudgePolicy};

/// What the load resolved, for the runtime settings (`think_nudge.*`).
pub(crate) struct ThinkNudgeSetup {
    /// The token tables, built for every supported model whether or not the
    /// nudge is on at launch, so it can be turned on at runtime. `None`: the
    /// model does not support the nudge.
    pub(crate) tables: Option<ThinkNudgeConfig>,
    /// The launch policy, `None` when `FLUX_THINK_NUDGE` is off.
    pub(crate) startup_policy: Option<ThinkNudgePolicy>,
    /// A launch setting named the nudge (on, or tuned).
    pub(crate) set_at_startup: bool,
}

/// `structural` are ids a nudge must never contain (EOS, think and tool-call
/// delimiters). A malformed setting is a startup error.
pub(crate) fn configure_think_nudge(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    target_model: &str,
    supports_thinking: bool,
    structural: &[u32],
) -> Result<ThinkNudgeSetup> {
    let set_at_startup = think_nudge::env_setting("THINK_NUDGE")?.is_some();
    let policy = think_nudge::parse_settings(think_nudge::env_setting)?;
    let qualified = supports_thinking && think_nudge::model_supported(target_model);
    if !qualified {
        if policy.is_some() {
            tracing::warn!(
                "FLUX_THINK_NUDGE=on is ignored for '{target_model}': the nudge is \
                 qualified for qwen3.8-flash-next and qwen3.8-27b only"
            );
        }
        return Ok(ThinkNudgeSetup {
            tables: None,
            startup_policy: None,
            set_at_startup,
        });
    }
    let vocab = tokenizer.inner().get_vocab_size(true);
    let mut newline = Vec::with_capacity(vocab);
    for id in 0..vocab as u32 {
        let class = tokenizer
            .decode(&[id])
            .map(|text| NewlineClass::of(&text))
            .unwrap_or(NewlineClass::Other);
        newline.push(class as u8);
    }
    let tables = ThinkNudgeConfig::build(
        policy.clone().unwrap_or_default(),
        |text| tokenizer.encode(text),
        newline,
        structural,
    )?;
    if let Some(policy) = &policy {
        tracing::info!(
            min_tokens = policy.min_tokens,
            window = policy.window,
            rate = policy.rate,
            max_per_turn = policy.max_nudges,
            nudge1_tokens = tables.tokens(0, false).len(),
            nudge2_tokens = tables.tokens(1, false).len(),
            "Thinking nudge ENABLED (FLUX_THINK_NUDGE=on): injects a commit sentence into \
             circling thinking; never closes thinking or stops generation"
        );
    }
    Ok(ThinkNudgeSetup {
        tables: Some(tables),
        startup_policy: policy,
        set_at_startup,
    })
}

/// Resolve and install the thinking-budget nudge (`FLUX_THINK_BUDGET_NUDGE`,
/// default on) for the model being loaded. Qualified for the same model
/// families as the repetition nudge; other models keep the bare forced
/// `</think>`. A malformed setting is a startup error.
pub(crate) fn configure_think_budget_nudge(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    target_model: &str,
    supports_thinking: bool,
    structural: &[u32],
) -> Result<()> {
    use crate::scheduler::think_budget_nudge::{self as budget, BUDGET_NUDGE_TEXT};
    let raw = think_nudge::env_setting("THINK_BUDGET_NUDGE")?;
    let enabled = budget::parse_enabled(raw.as_deref())?;
    let qualified = supports_thinking && think_nudge::model_supported(target_model);
    let tokens = if enabled && qualified {
        Some(budget::validate_tokens(
            tokenizer.encode(BUDGET_NUDGE_TEXT)?,
            structural,
        )?)
    } else {
        None
    };
    tracing::info!(
        enabled,
        qualified,
        nudge_tokens = tokens.as_ref().map_or(0, |t| t.len()),
        "Thinking-budget nudge: {}",
        if tokens.is_some() {
            "a budget-forced </think> is preceded by an answer-now sentence"
        } else {
            "off; a budget-forced </think> is injected bare"
        }
    );
    budget::install(tokens);
    Ok(())
}
