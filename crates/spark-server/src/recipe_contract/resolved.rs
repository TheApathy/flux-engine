// SPDX-License-Identifier: AGPL-3.0-only

//! Checks that run after the startup contract, once MODEL.toml defaults and
//! the prefill budget are resolved, plus the swap refusal. All are no-ops
//! unless a recipe passed the startup check in this process.

use super::refusal::{self, Refusal};
use super::schema::SpeculationMode;
use std::sync::OnceLock;

/// What the recipe requires of values that resolve after the startup check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expect {
    pub kv_cache_dtype: String,
    pub speculation: SpeculationMode,
    pub num_drafts: Option<u64>,
    pub max_prefill_tokens: [u64; 2],
}

static ACTIVE: OnceLock<Expect> = OnceLock::new();

pub(super) fn arm(expect: Expect) {
    // A second arm in one process cannot happen: swaps are refused below.
    let _ = ACTIVE.set(expect);
}

pub fn active() -> bool {
    ACTIVE.get().is_some()
}

fn refuse(r: Refusal) -> anyhow::Error {
    let line = refusal::render(&[r]);
    tracing::error!("{line}");
    anyhow::anyhow!(line)
}

/// Pure comparison, separate from the global for tests.
pub fn compare_kv(expect: &Expect, effective: &str) -> Result<(), Refusal> {
    if expect.kv_cache_dtype.eq_ignore_ascii_case(effective) {
        Ok(())
    } else {
        Err(Refusal::ResolvedMismatch {
            field: "kv_cache_dtype",
            want: expect.kv_cache_dtype.clone(),
            got: effective.to_string(),
        })
    }
}

pub fn compare_num_drafts(
    expect: &Expect,
    speculative: bool,
    num_drafts: u64,
) -> Result<(), Refusal> {
    match (expect.speculation, expect.num_drafts) {
        (SpeculationMode::Mtp, Some(want)) if speculative && want != num_drafts => {
            Err(Refusal::ResolvedMismatch {
                field: "num_drafts",
                want: want.to_string(),
                got: num_drafts.to_string(),
            })
        }
        _ => Ok(()),
    }
}

pub fn compare_prefill(expect: &Expect, budget: u64) -> Result<(), Refusal> {
    let [lo, hi] = expect.max_prefill_tokens;
    if (lo..=hi).contains(&budget) {
        Ok(())
    } else {
        Err(Refusal::ResolvedMismatch {
            field: "prefill_budget",
            want: format!("[{lo}, {hi}]"),
            got: budget.to_string(),
        })
    }
}

fn run(
    check: impl FnOnce(&Expect) -> Result<(), Refusal>,
    field: &str,
    value: &str,
) -> anyhow::Result<()> {
    let Some(expect) = ACTIVE.get() else {
        return Ok(());
    };
    check(expect).map_err(refuse)?;
    tracing::info!("RECIPE_RESOLVED {field}={value}");
    Ok(())
}

/// After `resolve_kv_cache_config`.
pub fn check_kv(effective: &str) -> anyhow::Result<()> {
    run(|e| compare_kv(e, effective), "kv_cache_dtype", effective)
}

/// After `apply_model_default_num_drafts`.
pub fn check_num_drafts(speculative: bool, num_drafts: usize) -> anyhow::Result<()> {
    let n = num_drafts as u64;
    run(
        |e| compare_num_drafts(e, speculative, n),
        "num_drafts",
        &n.to_string(),
    )
}

/// After the prefill budget is resolved.
pub fn check_prefill_budget(budget: usize) -> anyhow::Result<()> {
    let b = budget as u64;
    run(|e| compare_prefill(e, b), "prefill_budget", &b.to_string())
}

/// At the start of every model swap.
pub fn refuse_swap() -> anyhow::Result<()> {
    if active() {
        Err(refuse(Refusal::SwapUnderRecipe))
    } else {
        Ok(())
    }
}
