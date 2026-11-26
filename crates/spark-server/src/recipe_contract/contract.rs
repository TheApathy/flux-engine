// SPDX-License-Identifier: AGPL-3.0-only

//! Pure parse and evaluate: no env reads, no file reads.

use super::catalog::Catalog;
use super::env_check::{self, Declared, EnvReport};
use super::facts::{self, CheckpointFacts, ServeFacts, Settings};
use super::precision::{self, FamilyFormat};
use super::refusal::Refusal;
use super::schema::{RecipeFile, SCHEMA_VERSION, SamplingRanges, SamplingValues};
use std::collections::BTreeMap;

/// A parsed, self-consistent recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub recipe: RecipeFile,
    /// sha256 of the recipe file bytes.
    pub sha256: String,
    pub declared: Declared,
}

/// Everything the receipt prints.
#[derive(Clone, Debug, PartialEq)]
pub struct Effective {
    pub loaded: Loaded,
    pub precision: Vec<FamilyFormat>,
    pub flags: Settings,
    pub env: EnvReport,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn parse(text: &str, catalog: &Catalog) -> Result<Loaded, Refusal> {
    let recipe: RecipeFile =
        toml::from_str(text).map_err(|e| Refusal::Invalid(e.to_string().trim().to_string()))?;
    if recipe.schema != SCHEMA_VERSION {
        return Err(Refusal::Invalid(format!(
            "schema {} (this build reads {SCHEMA_VERSION})",
            recipe.schema
        )));
    }
    if recipe.id.trim().is_empty() || recipe.checkpoint.model_type.trim().is_empty() {
        return Err(Refusal::Invalid(
            "id and checkpoint.model_type must be non-empty".into(),
        ));
    }
    if recipe.checkpoint.sha256.is_empty() {
        return Err(Refusal::Invalid(
            "checkpoint.sha256 must pin at least one file".into(),
        ));
    }
    if let Some((file, _)) = recipe.checkpoint.sha256.iter().find(|(_, v)| !is_sha256(v)) {
        return Err(Refusal::Invalid(format!(
            "checkpoint.sha256.{file:?} is not 64 hex chars"
        )));
    }
    for (name, [lo, hi]) in [
        ("max_seq_len", recipe.limits.max_seq_len),
        ("max_prefill_tokens", recipe.limits.max_prefill_tokens),
        ("max_batch_size", recipe.limits.max_batch_size),
    ] {
        if lo == 0 || lo > hi {
            return Err(Refusal::Invalid(format!(
                "limits.{name} = [{lo}, {hi}] is not a range"
            )));
        }
    }
    check_sampling(
        &recipe.sampling.defaults,
        &recipe.sampling.ranges,
        "defaults",
    )?;
    if let Some(thinking) = &recipe.sampling.thinking {
        check_sampling(thinking, &recipe.sampling.ranges, "thinking")?;
    }
    if let Some(flag) = recipe
        .cli
        .keys()
        .find(|k| !facts::CLI_NUMERICS.contains(&k.as_str()))
    {
        return Err(Refusal::Invalid(format!(
            "[cli] {flag} is not one of the numerics flags"
        )));
    }
    precision::validate_plan(&recipe.precision)?;
    let declared = env_check::validate_declarations(&recipe, catalog)?;
    Ok(Loaded {
        recipe,
        sha256: sha256_hex(text.as_bytes()),
        declared,
    })
}

fn check_sampling(v: &SamplingValues, r: &SamplingRanges, label: &str) -> Result<(), Refusal> {
    let float_ok = |x: f64, [lo, hi]: [f64; 2]| lo <= hi && (lo..=hi).contains(&x);
    let ok = float_ok(v.temperature, r.temperature)
        && float_ok(v.top_p, r.top_p)
        && float_ok(v.presence_penalty, r.presence_penalty)
        && r.top_k[0] <= r.top_k[1]
        && (r.top_k[0]..=r.top_k[1]).contains(&v.top_k);
    if ok {
        Ok(())
    } else {
        Err(Refusal::Invalid(format!(
            "sampling.{label} outside sampling.ranges"
        )))
    }
}

/// Evaluate a parsed recipe against one process's env, flags and checkpoint.
/// `raw_env` holds only the `ATLAS_*`/`FLUX_*` variables. All refusals are
/// collected so one startup reports every problem.
pub fn evaluate(
    loaded: &Loaded,
    raw_env: &BTreeMap<String, String>,
    serve: &ServeFacts,
    checkpoint: &CheckpointFacts,
    catalog: &Catalog,
) -> Result<Effective, Vec<Refusal>> {
    let recipe = &loaded.recipe;
    let env_check::Canonical { env, shadowed } = env_check::canonicalize(raw_env, catalog);
    let mut refusals = Vec::new();
    let env_report = env_check::check(&env, &loaded.declared, recipe.route_policy, catalog)
        .map(|report| EnvReport {
            alias_shadowed: shadowed,
            ..report
        })
        .map_err(|r| refusals.extend(r))
        .ok();
    let precision = precision::check(&recipe.precision, &env, catalog)
        .map_err(|r| refusals.extend(r))
        .ok();
    let flags = facts::check_flags(
        &recipe.limits,
        &recipe.kv_cache_dtype,
        &recipe.speculation,
        serve,
    )
    .map_err(|r| refusals.extend(r))
    .ok();
    let cli = facts::check_cli(&recipe.cli, serve)
        .map_err(|r| refusals.extend(r))
        .ok();
    let flags = flags.zip(cli).map(|(mut f, c)| {
        f.extend(c);
        f
    });
    if let Err(r) = facts::check_checkpoint(&recipe.checkpoint, checkpoint) {
        refusals.extend(r);
    }
    match (env_report, precision, flags) {
        (Some(env), Some(precision), Some(flags)) if refusals.is_empty() => Ok(Effective {
            loaded: loaded.clone(),
            precision,
            flags,
            env,
        }),
        _ => Err(refusals),
    }
}
