// SPDX-License-Identifier: AGPL-3.0-only

//! Post-resolution checks: MODEL.toml defaults cannot drift a recipe.

use super::refusal::Refusal;
use super::resolved::{self, Expect};
use super::schema::SpeculationMode;

fn expect() -> Expect {
    Expect {
        kv_cache_dtype: "bf16".into(),
        speculation: SpeculationMode::Mtp,
        num_drafts: Some(4),
        max_prefill_tokens: [1, 16384],
    }
}

#[test]
fn a_model_toml_kv_default_that_differs_refuses() {
    assert!(resolved::compare_kv(&expect(), "BF16").is_ok());
    assert!(matches!(
        resolved::compare_kv(&expect(), "fp8"),
        Err(Refusal::ResolvedMismatch {
            field: "kv_cache_dtype",
            ..
        })
    ));
}

#[test]
fn a_model_toml_num_drafts_default_that_differs_refuses() {
    assert!(resolved::compare_num_drafts(&expect(), true, 4).is_ok());
    assert!(resolved::compare_num_drafts(&expect(), true, 3).is_err());
    let off = Expect {
        speculation: SpeculationMode::Off,
        ..expect()
    };
    assert!(resolved::compare_num_drafts(&off, false, 3).is_ok());
}

#[test]
fn a_resolved_prefill_budget_outside_the_range_refuses() {
    assert!(resolved::compare_prefill(&expect(), 16384).is_ok());
    assert!(resolved::compare_prefill(&expect(), 32768).is_err());
}

#[test]
fn checks_are_no_ops_without_an_armed_recipe() {
    if resolved::active() {
        return;
    }
    assert!(resolved::check_kv("fp8").is_ok());
    assert!(resolved::check_num_drafts(true, 9).is_ok());
    assert!(resolved::refuse_swap().is_ok());
}
