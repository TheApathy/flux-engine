// SPDX-License-Identifier: AGPL-3.0-only

//! Serve-flag and checkpoint facts, checked against the recipe.

use super::refusal::Refusal;
use super::schema::{Checkpoint, Limits, Speculation, SpeculationMode};
use std::collections::BTreeMap;

/// Serve flags that change numerics, KV precision or the rendered prompt
/// without touching any env var. Each is pinned by `[cli]` or must stay at
/// its built-in default.
pub const CLI_NUMERICS: &[&str] = &[
    "--rope-yarn-factor",
    "--rope-theta-override",
    "--kv-high-precision-layers",
    "--kv-high-precision-layer-set",
    "--fp8-kv-calibration-tokens",
    "--max-thinking-budget",
    "--disable-thinking",
    "--default-chat-template-kwargs",
    "--allow-special-tokens-in-content",
    "--kernel-target",
    "--mtp-vocab",
    "--mtp-quantization",
];

/// The serve flags the recipe constrains at startup, copied out of `ServeArgs`.
/// `kv_cache_dtype` and `num_drafts` are intentionally excluded here because
/// `MODEL.toml` can replace clap defaults; they are checked after resolution in
/// [`super::resolved`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeFacts {
    pub max_seq_len: u64,
    pub max_prefill_tokens: u64,
    pub max_batch_size: u64,
    pub qwen4_qsa: bool,
    pub mtp_speculative: bool,
    /// Any of `--self-speculative`, `--ngram-speculative`, `--dflash`.
    pub other_speculative: bool,
    /// `CLI_NUMERICS` flag to its current value (`unset` for an absent option).
    pub cli: BTreeMap<&'static str, String>,
    /// The same flags at their built-in defaults.
    pub cli_defaults: BTreeMap<&'static str, String>,
}

/// What the loader read from the model directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointFacts {
    /// `config.json` `model_type`; `None` when unreadable.
    pub model_type: Option<String>,
    /// Pinned file name to its sha256, or the read error.
    pub digests: BTreeMap<String, Result<String, String>>,
}

pub type Settings = Vec<(&'static str, String)>;

fn in_range(
    flag: &'static str,
    got: u64,
    [lo, hi]: [u64; 2],
    out: &mut Settings,
    refusals: &mut Vec<Refusal>,
) {
    if (lo..=hi).contains(&got) {
        out.push((flag, got.to_string()));
    } else {
        refusals.push(Refusal::FlagMismatch {
            flag,
            want: format!("[{lo}, {hi}]"),
            got: got.to_string(),
        });
    }
}

fn exact(
    flag: &'static str,
    want: String,
    got: String,
    out: &mut Settings,
    refusals: &mut Vec<Refusal>,
) {
    if want == got {
        out.push((flag, got));
    } else {
        refusals.push(Refusal::FlagMismatch { flag, want, got });
    }
}

pub fn check_flags(
    limits: &Limits,
    kv_cache_dtype: &str,
    spec: &Speculation,
    facts: &ServeFacts,
) -> Result<Settings, Vec<Refusal>> {
    let mut out = Vec::new();
    let mut refusals = Vec::new();
    in_range(
        "--max-seq-len",
        facts.max_seq_len,
        limits.max_seq_len,
        &mut out,
        &mut refusals,
    );
    in_range(
        "--max-prefill-tokens",
        facts.max_prefill_tokens,
        limits.max_prefill_tokens,
        &mut out,
        &mut refusals,
    );
    in_range(
        "--max-batch-size",
        facts.max_batch_size,
        limits.max_batch_size,
        &mut out,
        &mut refusals,
    );
    // KV dtype is checked after MODEL.toml resolution (`resolved::check_kv`);
    // the CLI value alone can be the clap default that MODEL.toml replaces.
    out.push((
        "--kv-cache-dtype",
        format!(
            "{} (checked after resolution)",
            kv_cache_dtype.to_ascii_lowercase()
        ),
    ));
    exact(
        "--qwen4-qsa",
        limits.qwen4_qsa.to_string(),
        facts.qwen4_qsa.to_string(),
        &mut out,
        &mut refusals,
    );
    let got_mode = match (facts.mtp_speculative, facts.other_speculative) {
        (false, false) => "off",
        (true, false) => "mtp",
        _ => "other",
    };
    let want_mode = match spec.mode {
        SpeculationMode::Off => "off",
        SpeculationMode::Mtp => "mtp",
    };
    exact(
        "speculation",
        want_mode.into(),
        got_mode.into(),
        &mut out,
        &mut refusals,
    );
    if refusals.is_empty() {
        Ok(out)
    } else {
        Err(refusals)
    }
}

/// Every `CLI_NUMERICS` flag must equal its `[cli]` pin, or its default.
pub fn check_cli(
    declared: &BTreeMap<String, String>,
    facts: &ServeFacts,
) -> Result<Settings, Vec<Refusal>> {
    let mut out = Vec::new();
    let mut refusals = Vec::new();
    for flag in CLI_NUMERICS {
        let got = facts
            .cli
            .get(flag)
            .cloned()
            .unwrap_or_else(|| "unset".into());
        let want = declared
            .get(*flag)
            .cloned()
            .or_else(|| facts.cli_defaults.get(flag).cloned())
            .unwrap_or_else(|| "unset".into());
        exact(flag, want, got, &mut out, &mut refusals);
    }
    if refusals.is_empty() {
        Ok(out)
    } else {
        Err(refusals)
    }
}

pub fn check_checkpoint(pins: &Checkpoint, facts: &CheckpointFacts) -> Result<(), Vec<Refusal>> {
    let mut refusals = Vec::new();
    match &facts.model_type {
        Some(mt) if *mt == pins.model_type => {}
        other => refusals.push(Refusal::CheckpointPin {
            file: "config.json".into(),
            detail: format!("model_type want {} got {other:?}", pins.model_type),
        }),
    }
    for (file, want) in &pins.sha256 {
        let detail = match facts.digests.get(file) {
            Some(Ok(got)) if got.eq_ignore_ascii_case(want) => continue,
            Some(Ok(got)) => format!("sha256 want {want} got {got}"),
            Some(Err(e)) => e.clone(),
            None => "not hashed".into(),
        };
        refusals.push(Refusal::CheckpointPin {
            file: file.clone(),
            detail,
        });
    }
    if refusals.is_empty() {
        Ok(())
    } else {
        Err(refusals)
    }
}
