// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn row(slot: usize, qsa_required: bool, host_seq_len: usize) -> RowFacts {
    RowFacts {
        slot,
        qsa_required,
        host_seq_len,
        identity_rotary: true,
    }
}

fn plan(max_batch_size: usize, native: bool, qsa_rows: bool) -> BTreeSet<GraphKey> {
    declared_keys(&PlanInput {
        max_batch_size,
        native_selected: native,
        qsa_enabled: true,
        qsa_graph_safe: true,
        qsa_rows_selected: qsa_rows,
        widths: default_widths(max_batch_size),
    })
}

#[test]
fn mode_parse_is_exact() {
    assert_eq!(parse_catalog_mode(None), Ok(CatalogMode::Off));
    assert_eq!(parse_catalog_mode(Some("off")), Ok(CatalogMode::Off));
    assert_eq!(
        parse_catalog_mode(Some("prepare")),
        Ok(CatalogMode::Prepare)
    );
    assert_eq!(parse_catalog_mode(Some("strict")), Ok(CatalogMode::Strict));
    for bad in ["", "1", "on", "Prepare", " strict", "lazy"] {
        assert!(parse_catalog_mode(Some(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(parse_qsa_rows(Some("1")), Ok(true));
    assert!(parse_qsa_rows(Some("true")).is_err());
    assert_eq!(parse_max_mib(None), Ok(DEFAULT_MAX_MIB));
    assert!(parse_max_mib(Some("0")).is_err());
    assert!(parse_max_mib(Some("-1")).is_err());
}

#[test]
fn regime_switches_at_the_first_sparse_step() {
    assert_eq!(row_regime(false, 2_046), Regime::Dense);
    assert_eq!(row_regime(true, 0), Regime::IndexedDense);
    assert_eq!(row_regime(true, 2_047), Regime::IndexedDense);
    assert_eq!(row_regime(true, 2_048), Regime::Sparse);
    assert_eq!(row_regime(true, 262_143), Regime::Sparse);
}

#[test]
fn default_widths_cover_every_width_to_eight_then_powers_of_two() {
    assert!(default_widths(1).is_empty());
    assert_eq!(default_widths(4), vec![2, 3, 4]);
    assert_eq!(default_widths(8), vec![2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(default_widths(12), vec![2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(default_widths(64), vec![2, 3, 4, 5, 6, 7, 8, 16, 32, 64]);
    assert_eq!(
        default_widths(128),
        vec![2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128]
    );
}

#[test]
fn explicit_widths_are_validated() {
    assert_eq!(parse_widths(Some("2,4,8"), 8), Ok(vec![2, 4, 8]));
    assert_eq!(parse_widths(Some(" 2, 4 "), 4), Ok(vec![2, 4]));
    for bad in ["", "1,2", "2,9", "4,2", "2,2", "2,x", "33", "129"] {
        assert!(parse_widths(Some(bad), 8).is_err(), "{bad:?}");
    }
}

#[test]
fn declared_set_counts() {
    assert_eq!(plan(1, false, false).len(), 1);
    assert_eq!(plan(8, false, false).len(), 8);
    assert_eq!(plan(8, true, false).len(), 8 + 7);
    assert_eq!(plan(8, true, true).len(), 8 * 3 + 7);
    let unsafe_qsa = declared_keys(&PlanInput {
        max_batch_size: 4,
        native_selected: true,
        qsa_enabled: true,
        qsa_graph_safe: false,
        qsa_rows_selected: true,
        widths: vec![2, 4],
    });
    assert!(unsafe_qsa.iter().all(|key| matches!(
        key,
        GraphKey::Row {
            regime: Regime::Dense,
            ..
        } | GraphKey::Native {
            regime: Regime::Dense,
            ..
        }
    )));
}

#[test]
fn native_frames_are_declared_dense_only() {
    for key in plan(8, true, true) {
        if let GraphKey::Native { regime, .. } = key {
            assert_eq!(regime, Regime::Dense);
        }
    }
}

#[test]
fn off_mode_is_lazy_for_every_frame() {
    let declared = plan(8, true, true);
    let r = row(0, true, 9_000);
    assert_eq!(
        decide(
            CatalogMode::Off,
            true,
            Frame::Row(&r),
            true,
            &declared,
            |_| EntryState::Absent
        ),
        Decision::Lazy
    );
}

#[test]
fn sealed_catalog_replays_or_names_the_miss() {
    let declared = plan(4, true, false);
    let ready = |_: &GraphKey| EntryState::Ready;
    let rows = [row(0, false, 10), row(3, false, 900)];
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Native(&rows),
            false,
            &declared,
            ready
        ),
        Decision::Replay(GraphKey::Native {
            width: 2,
            regime: Regime::Dense
        })
    );
    let qsa_rows = [row(0, true, 10), row(1, true, 20)];
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Native(&qsa_rows),
            false,
            &declared,
            ready
        ),
        Decision::Eager(OffCatalogReason::NativeQsaEager)
    );
    let mixed = [row(0, false, 10), row(1, true, 20)];
    assert_eq!(
        decide(
            CatalogMode::Strict,
            true,
            Frame::Native(&mixed),
            false,
            &declared,
            ready
        ),
        Decision::Refuse(OffCatalogReason::NativeQsaEager)
    );
    let wide = [row(0, false, 1); 5];
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Native(&wide),
            false,
            &declared,
            ready
        ),
        Decision::Eager(OffCatalogReason::WidthNotPrepared)
    );
    let far_slot = row(9, false, 1);
    assert_eq!(
        decide(
            CatalogMode::Strict,
            true,
            Frame::Row(&far_slot),
            false,
            &declared,
            ready
        ),
        Decision::Refuse(OffCatalogReason::SlotNotPrepared)
    );
    let sparse = row(1, true, 4_096);
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Row(&sparse),
            false,
            &declared,
            ready
        ),
        Decision::Eager(OffCatalogReason::RegimeNotPrepared)
    );
}

#[test]
fn capture_happens_only_before_the_seal() {
    let declared = plan(2, true, true);
    let absent = |_: &GraphKey| EntryState::Absent;
    let r = row(1, true, 2_048);
    let key = GraphKey::Row {
        slot: 1,
        regime: Regime::Sparse,
    };
    assert_eq!(
        decide(
            CatalogMode::Strict,
            false,
            Frame::Row(&r),
            false,
            &declared,
            absent
        ),
        Decision::Capture(key)
    );
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Row(&r),
            false,
            &declared,
            absent
        ),
        Decision::Eager(OffCatalogReason::EntryMissing)
    );
    assert_eq!(
        decide(
            CatalogMode::Strict,
            true,
            Frame::Row(&r),
            false,
            &declared,
            |_| EntryState::Poisoned
        ),
        Decision::Refuse(OffCatalogReason::EntryPoisoned)
    );
}

#[test]
fn image_and_diagnostic_frames_never_replay() {
    let declared = plan(2, true, false);
    let ready = |_: &GraphKey| EntryState::Ready;
    let image = RowFacts {
        identity_rotary: false,
        ..row(0, false, 3)
    };
    assert_eq!(
        decide(
            CatalogMode::Prepare,
            true,
            Frame::Row(&image),
            false,
            &declared,
            ready
        ),
        Decision::Eager(OffCatalogReason::ImageRotary)
    );
    let plain = row(0, false, 3);
    assert_eq!(
        decide(
            CatalogMode::Strict,
            true,
            Frame::Row(&plain),
            true,
            &declared,
            ready
        ),
        Decision::Refuse(OffCatalogReason::DiagnosticFrame)
    );
}

#[test]
fn load_refusal_reports_the_first_blocking_fact() {
    let clean = LoadFacts {
        qwen4: true,
        ..LoadFacts::default()
    };
    assert_eq!(load_refusal(&clean), None);
    assert_eq!(
        load_refusal(&LoadFacts::default()),
        Some(LoadRefusal::NotQwen4)
    );
    assert_eq!(
        load_refusal(&LoadFacts {
            speculative: true,
            seq_dependent_split_k: true,
            ..clean
        }),
        Some(LoadRefusal::Speculative)
    );
    assert_eq!(
        load_refusal(&LoadFacts {
            seq_dependent_split_k: true,
            ..clean
        })
        .map(LoadRefusal::as_str),
        Some("seq_dependent_split_k")
    );
}

#[test]
fn budget_fails_closed() {
    let floor = 8_192 * MIB;
    assert_eq!(
        budget_verdict(None, 512, Some(floor), floor),
        Err(BudgetRefusal::Unmeasured)
    );
    assert_eq!(
        budget_verdict(Some(1), 512, None, floor),
        Err(BudgetRefusal::Unmeasured)
    );
    assert_eq!(
        budget_verdict(Some(-4 * MIB as i64), 512, Some(floor), floor),
        Ok(())
    );
    assert_eq!(
        budget_verdict(Some((512 * MIB) as i64), 512, Some(floor), floor),
        Ok(())
    );
    assert_eq!(
        budget_verdict(Some((512 * MIB + 1) as i64), 512, Some(floor), floor),
        Err(BudgetRefusal::OverCap)
    );
    assert_eq!(
        budget_verdict(Some(MIB as i64), 512, Some(floor - 1), floor),
        Err(BudgetRefusal::UnderFloor)
    );
}

#[test]
fn native_qsa_frames_stay_eager_even_before_the_seal() {
    let declared = plan(4, true, true);
    let sparse = [row(0, true, 4_096), row(1, true, 9_000)];
    for sealed in [false, true] {
        assert_eq!(
            decide(
                CatalogMode::Prepare,
                sealed,
                Frame::Native(&sparse),
                false,
                &declared,
                |_| EntryState::Absent
            ),
            Decision::Eager(OffCatalogReason::NativeQsaEager)
        );
    }
}
