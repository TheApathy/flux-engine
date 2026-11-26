// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed recipe contract (`recipe.toml`), parity-plan item 2.
//!
//! A recipe declares one checkpoint's identity, its per-family precision plan,
//! the NUMERICS and ROUTE env gates it was qualified with, sampling defaults
//! and ranges, serve limits, KV dtype, speculation mode and the operational env
//! it tolerates. With `--recipe PATH` or `FLUX_RECIPE=PATH`, startup refuses
//! with one `RECIPE_REFUSED ...` line on any mismatch and otherwise logs one
//! `RECIPE_RECEIPT {json}` line. Without either, nothing is checked and
//! `RECIPE_LEGACY_MODE` is logged once: behaviour is unchanged.
//!
//! Not to be confused with `crate::recipe`, the YAML serving-recipe index.
//! See `docs/adr/recipe-contract.md`.

pub mod catalog;
mod checkpoint;
pub mod contract;
pub mod env_check;
pub mod facts;
pub mod precision;
pub mod receipt;
pub mod refusal;
pub mod resolved;
pub mod schema;

#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod legacy_tests;
#[cfg(test)]
mod receipt_tests;
#[cfg(test)]
mod refusal_tests;
#[cfg(test)]
mod resolved_tests;
#[cfg(test)]
mod test_fixtures;
#[cfg(test)]
mod tier_tests;

use crate::cli::ServeArgs;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

pub const RECIPE_ENV: &str = "FLUX_RECIPE";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Legacy,
    Recipe(PathBuf),
}

/// The flag wins over the env; an empty env value is the same as unset.
pub fn mode(flag: Option<&Path>, env_value: Option<&OsStr>) -> Mode {
    match (flag, env_value.filter(|v| !v.is_empty())) {
        (Some(path), _) => Mode::Recipe(path.to_path_buf()),
        (None, Some(v)) => Mode::Recipe(PathBuf::from(v)),
        (None, None) => Mode::Legacy,
    }
}

static LEGACY_ONCE: std::sync::Once = std::sync::Once::new();

/// Called once before any weight load. Legacy mode returns `Ok(())` without
/// reading anything else.
pub fn enforce_at_startup(args: &ServeArgs) -> anyhow::Result<()> {
    match mode(
        args.recipe.as_deref(),
        std::env::var_os(RECIPE_ENV).as_deref(),
    ) {
        Mode::Legacy => {
            LEGACY_ONCE.call_once(|| tracing::info!("{}", receipt::LEGACY_TAG));
            Ok(())
        }
        Mode::Recipe(path) => enforce(&path, args),
    }
}

fn refuse(refusals: &[refusal::Refusal]) -> anyhow::Error {
    let line = refusal::render(refusals);
    tracing::error!("{line}");
    anyhow::anyhow!(line)
}

fn enforce(path: &Path, args: &ServeArgs) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        refuse(&[refusal::Refusal::Invalid(format!(
            "cannot read {}: {e}",
            path.display()
        ))])
    })?;
    let catalog =
        catalog::Catalog::builtin().map_err(|e| refuse(&[refusal::Refusal::Invalid(e)]))?;
    let loaded = contract::parse(&text, &catalog).map_err(|r| refuse(&[r]))?;
    let dir = checkpoint::model_dir(args.model_from_path.as_deref(), args.model.as_deref());
    let ckpt = checkpoint::read(dir.as_deref(), &loaded.recipe.checkpoint);
    let env = env_snapshot(std::env::vars_os()).map_err(|r| refuse(&r))?;
    let effective = contract::evaluate(&loaded, &env, &serve_facts(args), &ckpt, &catalog)
        .map_err(|r| refuse(&r))?;
    tracing::info!("{}", receipt::line(&effective));
    let r = &effective.loaded.recipe;
    resolved::arm(resolved::Expect {
        kv_cache_dtype: r.kv_cache_dtype.clone(),
        speculation: r.speculation.mode,
        num_drafts: r.speculation.num_drafts,
        max_prefill_tokens: r.limits.max_prefill_tokens,
    });
    Ok(())
}

fn governed(key: &str) -> bool {
    (key.starts_with("ATLAS_") || key.starts_with("FLUX_")) && key != RECIPE_ENV
}

/// Every `ATLAS_*`/`FLUX_*` variable except `FLUX_RECIPE` itself. A name or
/// value that is not UTF-8 refuses: the engine reads it with `var_os`, so
/// dropping it would let it past the check.
pub fn env_snapshot(
    vars: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
) -> Result<BTreeMap<String, String>, Vec<refusal::Refusal>> {
    let mut out = BTreeMap::new();
    let mut refusals = Vec::new();
    for (k, v) in vars {
        let lossy = k.to_string_lossy().into_owned();
        if !governed(&lossy) {
            continue;
        }
        match (k.into_string(), v.into_string()) {
            (Ok(k), Ok(v)) => {
                out.insert(k, v);
            }
            _ => refusals.push(refusal::Refusal::NonUtf8Env(lossy)),
        }
    }
    if refusals.is_empty() {
        Ok(out)
    } else {
        Err(refusals)
    }
}

fn opt<T: ToString>(v: Option<T>) -> String {
    v.map_or_else(|| "unset".to_string(), |x| x.to_string())
}

/// `facts::CLI_NUMERICS` values as the recipe's `[cli]` table spells them.
pub fn cli_values(a: &ServeArgs) -> BTreeMap<&'static str, String> {
    BTreeMap::from([
        ("--rope-yarn-factor", opt(a.rope_yarn_factor)),
        ("--rope-theta-override", opt(a.rope_theta_override)),
        (
            "--kv-high-precision-layers",
            a.kv_high_precision_layers.clone(),
        ),
        (
            "--kv-high-precision-layer-set",
            a.kv_high_precision_layer_set.clone(),
        ),
        (
            "--fp8-kv-calibration-tokens",
            a.fp8_kv_calibration_tokens.to_string(),
        ),
        ("--max-thinking-budget", opt(a.max_thinking_budget)),
        ("--disable-thinking", a.disable_thinking.to_string()),
        (
            "--default-chat-template-kwargs",
            opt(a.default_chat_template_kwargs.clone()),
        ),
        (
            "--allow-special-tokens-in-content",
            a.allow_special_tokens_in_content.to_string(),
        ),
        ("--kernel-target", opt(a.kernel_target.clone())),
        ("--mtp-vocab", a.mtp_vocab.to_string()),
        ("--mtp-quantization", a.mtp_quantization.clone()),
    ])
}

fn cli_defaults() -> BTreeMap<&'static str, String> {
    use clap::Parser;
    cli_values(&ServeArgs::parse_from(["spark", "defaults"]))
}

pub fn serve_facts(args: &ServeArgs) -> facts::ServeFacts {
    facts::ServeFacts {
        max_seq_len: args.max_seq_len as u64,
        max_prefill_tokens: args.max_prefill_tokens as u64,
        max_batch_size: args.max_batch_size as u64,
        qwen4_qsa: args.qwen4_qsa,
        mtp_speculative: args.speculative,
        other_speculative: args.self_speculative || args.ngram_speculative || args.dflash,
        cli: cli_values(args),
        cli_defaults: cli_defaults(),
    }
}
