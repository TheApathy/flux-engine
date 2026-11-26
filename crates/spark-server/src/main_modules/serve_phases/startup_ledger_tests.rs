// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    FloorDecision, LedgerRow, StartupBudgetLedger, assemble, merge_requested, weight_family,
    weight_family_rows,
};
use spark_runtime::startup_ledger::LedgerEntry;

const MIB: usize = 1024 * 1024;
const FLOOR: usize = 6656 * MIB;

fn row(pool: &str, requested: Option<usize>, actual: Option<i64>, detail: &str) -> LedgerRow {
    LedgerRow {
        pool: pool.to_string(),
        detail: detail.to_string(),
        requested_bytes: requested,
        actual_delta_bytes: actual,
    }
}

/// The r8 `095613Z` W16K start, in MiB, after the transpose pass moved.
fn r8_ledger(mem_available: Option<usize>, override_selected: bool) -> StartupBudgetLedger {
    StartupBudgetLedger {
        rows: vec![
            row(
                "weights/routed-experts",
                Some(61_440 * MIB),
                None,
                "147456 tensors (logical store bytes)",
            ),
            row(
                "moe-transpose-pass",
                None,
                Some(-10_409 * MIB as i64),
                "unified; originals freed",
            ),
            row(
                "kv-cache",
                Some(3_172 * MIB),
                Some(3_180 * MIB as i64),
                "8450 blocks x 16 tok x 12 layers",
            ),
            row(
                "fp8-weight-cache",
                None,
                None,
                "fills at first prefill until MemAvailable - weight < 8704 MiB",
            ),
        ],
        mem_available_bytes: mem_available,
        floor_bytes: FLOOR,
        override_selected,
    }
}

#[test]
fn weight_families_are_classified_by_name() {
    assert_eq!(
        weight_family("model.language_model.layers.3.mlp.experts.17.gate_proj.weight"),
        "weights/routed-experts"
    );
    assert_eq!(
        weight_family("model.language_model.layers.3.mlp.shared_expert.up_proj.weight"),
        "weights/shared-expert"
    );
    assert_eq!(
        weight_family("model.language_model.layers.0.linear_attn.in_proj_qkvz.weight"),
        "weights/ssm"
    );
    assert_eq!(
        weight_family("model.language_model.layers.3.self_attn.q_proj.weight"),
        "weights/attention"
    );
    assert_eq!(
        weight_family("mtp.layers.0.mlp.experts.0.down_proj.weight"),
        "weights/mtp"
    );
    assert_eq!(
        weight_family("model.language_model.layers.1.ple.key_proj.weight"),
        "weights/ple"
    );
    assert_eq!(
        weight_family("model.language_model.embed_tokens.weight"),
        "weights/embedding"
    );
    assert_eq!(weight_family("lm_head.weight"), "weights/lm-head");
    assert_eq!(
        weight_family("model.visual.blocks.0.attn.qkv.weight"),
        "weights/vision"
    );
    assert_eq!(
        weight_family("model.layers.2.mlp.gate_proj.weight"),
        "weights/dense-ffn"
    );
    assert_eq!(weight_family("model.norm.weight"), "weights/other");
}

#[test]
fn weight_family_rows_sum_bytes_and_count_tensors() {
    let rows = weight_family_rows(
        [
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                100,
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
                50,
            ),
            ("lm_head.weight", 7),
        ]
        .into_iter(),
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].pool, "weights/lm-head");
    assert_eq!(rows[0].requested_bytes, Some(7));
    assert_eq!(rows[1].pool, "weights/routed-experts");
    assert_eq!(rows[1].requested_bytes, Some(150));
    assert_eq!(rows[1].detail, "2 tensors (logical store bytes)");
    assert!(rows.iter().all(|row| row.actual_delta_bytes.is_none()));
}

#[test]
fn merge_requested_fills_the_measured_row_or_adds_one() {
    let mut rows = vec![row("kv-cache", None, Some(5), "measured")];
    merge_requested(&mut rows, "kv-cache", 4, "from preflight");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].requested_bytes, Some(4));
    assert_eq!(rows[0].detail, "measured", "a measured detail is kept");
    merge_requested(&mut rows, "cuda-headroom-reserve", 9, "reserve");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1].pool, "cuda-headroom-reserve");
    assert_eq!(rows[1].actual_delta_bytes, None);
    // A row whose requested bytes the factory already knows is never overwritten.
    merge_requested(&mut rows, "kv-cache", 99, "late");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].requested_bytes, Some(4));
}

#[test]
fn assemble_orders_weights_then_measured_then_reserves_then_extras() {
    let ledger = assemble(
        vec![row("weights/lm-head", Some(1), None, "")],
        &[
            ("kv-cache", 10, "preflight"),
            ("cuda-headroom-reserve", 20, "reserve"),
        ],
        vec![LedgerEntry {
            pool: "kv-cache".to_string(),
            detail: "measured".to_string(),
            requested_bytes: None,
            actual_delta_bytes: Some(11),
        }],
        vec![row("cuda-graphs", None, None, "off")],
        Some(FLOOR + MIB),
        FLOOR,
        false,
    );
    let pools: Vec<&str> = ledger.rows.iter().map(|row| row.pool.as_str()).collect();
    assert_eq!(
        pools,
        [
            "weights/lm-head",
            "kv-cache",
            "cuda-headroom-reserve",
            "cuda-graphs"
        ]
    );
    assert_eq!(ledger.rows[1].requested_bytes, Some(10));
    assert_eq!(ledger.rows[1].actual_delta_bytes, Some(11));
    assert_eq!(
        ledger.decision(),
        FloorDecision::Pass {
            headroom_bytes: MIB
        }
    );
}

#[test]
fn render_prints_one_table_with_unknowns_as_dashes_and_signed_actuals() {
    let text = r8_ledger(Some(13_412 * MIB), false).render();
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[0].starts_with("STARTUP_BUDGET_LEDGER (MiB;"),
        "{text}"
    );
    assert!(
        lines[1].contains("pool") && lines[1].contains("requested") && lines[1].contains("actual")
    );
    assert!(
        text.contains("weights/routed-experts                61440          -  147456 tensors"),
        "{text}"
    );
    assert!(
        text.contains("moe-transpose-pass                        -     -10409  unified"),
        "{text}"
    );
    assert!(
        text.contains("kv-cache                               3172       3180  8450 blocks"),
        "{text}"
    );
    assert!(
        text.contains(
            "fp8-weight-cache                          -          -  fills at first prefill"
        ),
        "{text}"
    );
    assert!(
        text.contains("MemAvailable now                      13412"),
        "{text}"
    );
    assert!(
        text.contains("floor                                  6656  ATLAS_MEMAVAILABLE_FLOOR_MIB"),
        "{text}"
    );
    assert!(
        text.ends_with("headroom above floor                   6756  PASS"),
        "{text}"
    );
    assert_eq!(
        text.matches("STARTUP_BUDGET_LEDGER").count(),
        1,
        "exactly one table"
    );
}

#[test]
fn floor_decision_passes_on_the_floor_and_refuses_one_byte_under() {
    assert_eq!(
        r8_ledger(Some(FLOOR), false).decision(),
        FloorDecision::Pass { headroom_bytes: 0 }
    );
    assert_eq!(
        r8_ledger(Some(FLOOR - 1), false).decision(),
        FloorDecision::Refused { shortfall_bytes: 1 }
    );
    assert_eq!(
        r8_ledger(Some(FLOOR - 1), true).decision(),
        FloorDecision::Overridden { shortfall_bytes: 1 }
    );
    assert_eq!(
        r8_ledger(None, false).decision(),
        FloorDecision::Unmeasurable
    );
}

#[test]
fn enforce_refuses_below_the_floor_and_names_both_selectors() {
    // r6 `run-r6-stage-a-w8k` bottomed at 5,998 MiB during D61.
    let error = r8_ledger(Some(5_998 * MIB), false)
        .enforce()
        .unwrap_err()
        .to_string();
    assert!(error.contains("5998 MiB MemAvailable"), "{error}");
    assert!(error.contains("658 MiB under"), "{error}");
    assert!(error.contains("ATLAS_MEMAVAILABLE_FLOOR_MIB"), "{error}");
    assert!(
        error.contains("ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1"),
        "{error}"
    );
    assert!(r8_ledger(Some(5_998 * MIB), true).enforce().is_ok());
    assert!(r8_ledger(Some(13_412 * MIB), false).enforce().is_ok());
}

#[test]
fn enforce_fails_closed_when_memavailable_is_unreadable_on_linux() {
    let result = r8_ledger(None, false).enforce();
    if cfg!(target_os = "linux") {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("/proc/meminfo"), "{error}");
    } else {
        assert!(result.is_ok());
    }
    assert!(r8_ledger(None, true).enforce().is_ok());
}

#[test]
fn render_shows_the_refusal_and_the_override_verdicts() {
    let refused = r8_ledger(Some(5_998 * MIB), false).render();
    assert!(refused.ends_with("REFUSED (short by 658 MiB)"), "{refused}");
    let overridden = r8_ledger(Some(5_998 * MIB), true).render();
    assert!(
        overridden.contains("OVERRIDDEN by ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1"),
        "{overridden}"
    );
    let unmeasurable = r8_ledger(None, false).render();
    assert!(
        unmeasurable.contains("MemAvailable now                          -"),
        "{unmeasurable}"
    );
    assert!(
        unmeasurable.ends_with("UNMEASURABLE (/proc/meminfo unreadable)"),
        "{unmeasurable}"
    );
}

#[test]
fn reduced_floor_never_produces_an_ordinary_pass() {
    for floor_bytes in [0, MIB, FLOOR - MIB] {
        let mut ledger = r8_ledger(Some(13_412 * MIB), false);
        ledger.floor_bytes = floor_bytes;
        assert_eq!(ledger.decision(), FloorDecision::InvalidFloor);
        assert!(ledger.enforce().is_err());
        assert!(ledger.render().contains("REFUSED"));
        ledger.override_selected = true;
        assert_eq!(
            ledger.decision(),
            FloorDecision::Overridden { shortfall_bytes: 0 }
        );
        assert!(ledger.enforce().is_ok());
        let text = ledger.render();
        assert!(text.contains("NONQUALIFYING"), "{text}");
        assert!(!text.contains("PASS"), "{text}");
    }
}

#[test]
fn explicit_override_is_nonqualifying_even_with_positive_headroom() {
    let ledger = r8_ledger(Some(FLOOR + MIB), true);
    assert_eq!(
        ledger.decision(),
        FloorDecision::Overridden { shortfall_bytes: 0 }
    );
    assert!(ledger.render().contains("NONQUALIFYING"));
    assert!(!ledger.render().contains("PASS"));
    let unmeasurable = r8_ledger(None, true).render();
    assert!(unmeasurable.contains("UNMEASURABLE"));
    assert!(unmeasurable.contains("OVERRIDDEN"));
    assert!(unmeasurable.contains("NONQUALIFYING"));
}
