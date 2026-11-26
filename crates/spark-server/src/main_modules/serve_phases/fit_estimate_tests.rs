// SPDX-License-Identifier: AGPL-3.0-only

//! Calibration against measured GB10 launch peaks (Flash-Next and Qwen3.8-27B
//! with DFlash2). The modelled inputs below are the measured rows for those
//! terms, so these tests pin the measured constants and the sum; the modelled
//! functions themselves are covered by the preflight tests.

use super::*;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB_U: usize = 1024 * 1024;

fn gib(x: f64) -> usize {
    (x * GIB) as usize
}

/// Census sec. 2: FN auto, spec-on (native MTP), 135168 ctx, C1.
fn fn_facts() -> FitFacts {
    FitFacts {
        family: Family::Qwen4,
        checkpoint_bytes: gib(72.17),
        mtp_release_bytes: 0,
        fp8_weight_cache: true,
        construction_modelled_bytes: gib(1.0),
        qwen4_transient_bytes: gib(2.83),
        lm_head_elems: 248_320 * 2_560,
        pools_bytes: gib(1.92),
        arena_bytes: gib(3.48),
        kv_bytes: gib(3.15),
        qsa_bytes: gib(1.58),
        dflash: false,
    }
}

/// Census sec. 3/4: 27B auto with the DFlash drafter.
fn q27_facts() -> FitFacts {
    FitFacts {
        family: Family::HybridDense,
        checkpoint_bytes: gib(20.41),
        mtp_release_bytes: 0,
        fp8_weight_cache: false,
        construction_modelled_bytes: gib(38.50),
        qwen4_transient_bytes: 0,
        lm_head_elems: 248_320 * 5_120,
        pools_bytes: gib(5.08),
        arena_bytes: gib(2.98),
        kv_bytes: gib(2.25),
        qsa_bytes: 0,
        dflash: true,
    }
}

const FLOOR: usize = 12_288 * MIB_U;

#[test]
fn flash_next_estimate_covers_the_census_peak_with_bounded_slack() {
    let est = estimate(&fn_facts(), 119_679 * MIB_U, FLOOR);
    let need = est.need_bytes() as f64 / GIB;
    // Measured serving peak 92.99 GiB. The estimate keeps the transposed-slab
    // and MTP transient (load peak) on top of it: conservative by < 3 GiB.
    assert!(
        (92.99..92.99 + 3.0).contains(&need),
        "FN need {need:.2} GiB"
    );
    assert!(est.fits(), "{}", est.render());
}

#[test]
fn qwen27b_estimate_matches_the_census_peak() {
    let est = estimate(&q27_facts(), 119_920 * MIB_U, FLOOR);
    let need = est.need_bytes() as f64 / GIB;
    // Measured serving peak 92.52 GiB.
    assert!(
        (92.52 - 0.5..92.52 + 1.0).contains(&need),
        "27B need {need:.2} GiB"
    );
    assert!(est.fits());
}

#[test]
fn store_term_is_measured_and_rounding_tracks_the_family() {
    let fn_est = estimate(&fn_facts(), 0, 0);
    let fn_store = fn_est.term("weights-store").expect("row");
    assert_eq!(fn_store.source, TermSource::Measured);
    let store_gib = fn_store.bytes as f64 / GIB;
    assert!((store_gib - 74.25).abs() < 0.05, "{store_gib}");
    let q = estimate(&q27_facts(), 0, 0)
        .term("weights-store")
        .expect("row")
        .bytes as f64
        / GIB;
    assert!((q - 21.61).abs() < 0.05, "{q}");
}

#[test]
fn construction_applies_the_measured_correction() {
    let c = estimate(&q27_facts(), 0, 0)
        .term("construction")
        .expect("row")
        .bytes as f64
        / GIB;
    assert!((c - 45.52).abs() < 0.05, "27B construction {c}");
    // FN: the measured target-layers row (3.99 GiB) is the floor of the term.
    let f = estimate(&fn_facts(), 0, 0)
        .term("construction")
        .expect("row")
        .bytes as f64
        / GIB;
    assert!((f - 3.99).abs() < 0.05, "FN construction {f}");
}

#[test]
fn apps_running_on_the_host_refuse_with_exact_numbers() {
    // desktop host with user apps up: ~96 GiB available.
    let est = estimate(&fn_facts(), 98_304 * MIB_U, FLOOR);
    assert!(!est.fits());
    let line = est.refusal_line();
    assert!(line.starts_with("FIT_REFUSED need_mib="), "{line}");
    assert!(
        line.contains(" avail_mib=98304 floor_mib=12288 terms={"),
        "{line}"
    );
    assert!(line.contains("\"weights-store\":"), "{line}");
    assert!(line.contains("FLUX_FIT_CHECK=off"), "{line}");
}

#[test]
fn render_marks_measured_versus_modelled() {
    let text = estimate(&fn_facts(), 119_679 * MIB_U, FLOOR).render();
    assert!(
        text.contains("kv-cache") && text.contains("modelled"),
        "{text}"
    );
    assert!(
        text.contains("serve-growth") && text.contains("measured"),
        "{text}"
    );
}

#[test]
fn dflash_term_only_when_selected() {
    let with = estimate(&q27_facts(), 0, 0);
    assert!(with.term("dflash-drafter").is_some());
    let without = estimate(
        &FitFacts {
            dflash: false,
            ..q27_facts()
        },
        0,
        0,
    );
    assert!(without.term("dflash-drafter").is_none());
}

#[test]
fn qwen4_credits_the_released_bf16_lm_head() {
    let est = estimate(&fn_facts(), 0, 0);
    let credit = est.term("lm-head").expect("row");
    assert!(
        credit.bytes < 0,
        "FN releases the BF16 head: {}",
        credit.bytes
    );
}

#[test]
fn mode_defaults_to_refuse_and_parses_strictly() {
    // Warn by default until calibrated against two real launches.
    assert_eq!(parse_mode(None).expect("default"), FitMode::Warn);
    assert_eq!(parse_mode(Some("1")).expect("1"), FitMode::Refuse);
    assert_eq!(parse_mode(Some("off")).expect("off"), FitMode::Off);
    assert_eq!(parse_mode(Some("warn")).expect("warn"), FitMode::Warn);
    assert_eq!(parse_mode(Some("refuse")).expect("refuse"), FitMode::Refuse);
    assert!(parse_mode(Some("yes")).is_err());
}

#[test]
fn uncalibrated_families_warn_instead_of_refusing() {
    assert_eq!(
        effective_mode(FitMode::Refuse, Family::Other),
        FitMode::Warn
    );
    assert_eq!(
        effective_mode(FitMode::Refuse, Family::Qwen4),
        FitMode::Refuse
    );
    assert_eq!(effective_mode(FitMode::Off, Family::Qwen4), FitMode::Off);
}

#[test]
fn decide_refuses_only_in_refuse_mode() {
    let tight = estimate(&fn_facts(), 90_000 * MIB_U, FLOOR);
    assert!(decide(FitMode::Refuse, &tight).is_err());
    assert!(decide(FitMode::Warn, &tight).is_ok());
    assert!(decide(FitMode::Off, &tight).is_ok());
}

/// Gate run resource-contract arm B (2026-10-31): lean FN recipe, auto,
/// speculation off, 16384 ctx, FP8 weight cache off, PLE offloaded. The
/// first estimate summed on-disk shards (104331 MiB store) and refused; the
/// base arm with the same recipe measured ready ~78 / peak ~80 GiB.
fn lean_fn_facts() -> FitFacts {
    FitFacts {
        family: Family::Qwen4,
        // FAST_WEIGHT_ALLOCATION_AUDIT logical_store_bytes of the base arm.
        checkpoint_bytes: 77_493_858_296,
        // QWEN4_RELEASE_UNUSED_MTP_EARLY requested_bytes.
        mtp_release_bytes: 1_596_720_640,
        fp8_weight_cache: false,
        construction_modelled_bytes: 4_186 * MIB_U,
        qwen4_transient_bytes: 1_691 * MIB_U,
        lm_head_elems: 1_212 * MIB_U / 2,
        pools_bytes: 574 * MIB_U,
        arena_bytes: 3_519 * MIB_U,
        kv_bytes: 384 * MIB_U,
        qsa_bytes: 239 * MIB_U,
        dflash: false,
    }
}

#[test]
fn lean_flash_next_launch_is_not_refused() {
    let est = estimate(&lean_fn_facts(), 106_302 * MIB_U, FLOOR);
    let need = est.need_bytes() as f64 / GIB;
    assert!(
        (80.0..86.0).contains(&need),
        "lean FN need {need:.2} GiB\n{}",
        est.render()
    );
    assert!(est.fits(), "{}", est.render());
    assert!(est.term("fp8-weight-cache").is_none());
    assert!(est.term("mtp-release").is_some_and(|t| t.bytes < 0));
}

#[test]
fn warn_line_is_not_a_refusal() {
    let tight = estimate(&lean_fn_facts(), 90_000 * MIB_U, FLOOR);
    let line = tight.warn_line();
    assert!(line.starts_with("FIT_WARN need_mib="), "{line}");
    assert!(line.contains("starting anyway"), "{line}");
}

#[test]
fn header_bytes_skip_metadata_and_reject_bad_offsets() {
    let header = r#"{"__metadata__":{"format":"pt"},
        "a.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
        "mtp.fc.weight":{"dtype":"U8","shape":[6],"data_offsets":[4,10]}}"#;
    let mut rows = header_tensor_bytes(header).expect("parse");
    rows.sort();
    assert_eq!(
        rows,
        vec![
            ("a.weight".to_string(), 4),
            ("mtp.fc.weight".to_string(), 6)
        ]
    );
    assert!(header_tensor_bytes(r#"{"x":{"data_offsets":[9,1]}}"#).is_err());
    assert!(header_tensor_bytes("not json").is_err());
}

#[test]
fn resident_bytes_drop_the_offloaded_table_and_count_mtp() {
    let rows = vec![
        (
            "model.layers.1.ple.ple_embedding.ngram_embedding.weight".to_string(),
            1_000,
        ),
        ("model.layers.0.mlp.weight".to_string(), 10),
        ("mtp.fc.weight".to_string(), 5),
    ];
    let skip = |n: &str| n.contains("ngram_embedding");
    assert_eq!(
        resident_bytes(rows.clone(), &skip),
        ResidentBytes { store: 15, mtp: 5 }
    );
    assert_eq!(
        resident_bytes(rows, &|_| false),
        ResidentBytes {
            store: 1_015,
            mtp: 5
        }
    );
}
