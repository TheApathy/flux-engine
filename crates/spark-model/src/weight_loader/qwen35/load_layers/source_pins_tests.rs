// SPDX-License-Identifier: AGPL-3.0-only

//! Pairwise matrix: every memlean release primitive x every native-dense
//! reader selector, on MockGpuBackend. A pinned source must stay resident
//! (no "consumed during model construction" on lookup) exactly as in the
//! reader-only run, the skip must be reported with the pinning selector, and
//! an unpinned source must still be released. MXFP8 originals are pinned
//! only by the selectors that alias them (DENSE_FP8 / DENSE_MXFP8).

use std::collections::{BTreeSet, HashMap};

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{dense_dedup, mxfp8_release};
use crate::qwen4_native_selectors::plan_from;
use crate::source_pins::*;
use crate::weight_map::DenseWeight;

const LP: &str = "model.language_model.layers.3";

fn names() -> Vec<(String, WeightDtype)> {
    let mut v = Vec::new();
    for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        v.push((format!("{LP}.self_attn.{p}.weight"), WeightDtype::FP8E4M3));
    }
    for p in ["in_proj_qkv", "in_proj_z", "out_proj"] {
        v.push((format!("{LP}.linear_attn.{p}.weight"), WeightDtype::FP8E4M3));
    }
    for p in ["gate_proj", "up_proj", "down_proj"] {
        v.push((
            format!("{LP}.mlp.shared_expert.{p}.weight"),
            WeightDtype::FP8E4M3,
        ));
    }
    v.push((format!("{LP}.mlp.gate.weight"), WeightDtype::BF16));
    v.push((
        format!("{LP}.attn_hyper_connection.input_mix_weight_down.weight"),
        WeightDtype::BF16,
    ));
    v
}

fn store(gpu: &MockGpuBackend) -> WeightStore {
    WeightStore::from_map(
        names()
            .into_iter()
            .map(|(name, dtype)| {
                let ptr = gpu.alloc(64 * dtype.byte_size()).expect("alloc");
                (
                    name,
                    WeightTensor {
                        ptr,
                        shape: vec![64],
                        dtype,
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    )
}

/// The release primitives under test, each on its own family.
#[derive(Clone, Copy, Debug)]
enum Release {
    Mxfp8Attn,
    Mxfp8Gdn,
    Mxfp8Shared,
    DedupShared,
    DedupRouter,
    DedupHc,
}

const RELEASES: [Release; 6] = [
    Release::Mxfp8Attn,
    Release::Mxfp8Gdn,
    Release::Mxfp8Shared,
    Release::DedupShared,
    Release::DedupRouter,
    Release::DedupHc,
];

impl Release {
    fn family(self) -> PinFamily {
        match self {
            Self::Mxfp8Attn => PinFamily::Attn,
            Self::Mxfp8Gdn => PinFamily::Gdn,
            Self::Mxfp8Shared | Self::DedupShared => PinFamily::Shared,
            Self::DedupRouter => PinFamily::Router,
            Self::DedupHc => PinFamily::Hc,
        }
    }

    fn kind(self) -> PinKind {
        match self {
            Self::Mxfp8Attn | Self::Mxfp8Gdn | Self::Mxfp8Shared | Self::DedupShared => {
                PinKind::Mxfp8Source
            }
            Self::DedupRouter | Self::DedupHc => PinKind::Bf16Alias,
        }
    }

    fn run(self, store: &WeightStore, gpu: &MockGpuBackend) -> Result<usize> {
        match self {
            Self::Mxfp8Attn => {
                ["q_proj", "k_proj", "v_proj", "o_proj"]
                    .iter()
                    .try_fold(0, |n, p| {
                        Ok(n + mxfp8_release::release_attention_proj(
                            store,
                            &format!("{LP}.self_attn"),
                            p,
                            gpu,
                        )?)
                    })
            }
            Self::Mxfp8Gdn => {
                mxfp8_release::release_linear_attn(store, &format!("{LP}.linear_attn"), gpu)
            }
            Self::Mxfp8Shared => mxfp8_release::release_shared_expert(store, LP, gpu),
            Self::DedupShared => dense_dedup::release_shared_expert_sources(store, LP, gpu),
            Self::DedupRouter => {
                let ptr = store.get(&format!("{LP}.mlp.gate.weight"))?.ptr;
                let mut gate = DenseWeight { weight: ptr };
                dense_dedup::release_router_bf16(store, LP, &mut gate, true, gpu)
            }
            Self::DedupHc => {
                let prefix = format!("{LP}.attn_hyper_connection.input_mix_weight_down");
                let ptr = store.get(&format!("{prefix}.weight"))?.ptr;
                dense_dedup::release_bf16_sources(
                    store,
                    &[(prefix, DenseWeight { weight: ptr })],
                    gpu,
                )
            }
        }
    }
}

fn pins_for(env: &[(&str, &str)]) -> SourcePins {
    let lookup = |name: &str| {
        Ok(env
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.to_string()))
    };
    let plan = plan_from(lookup).expect("plan");
    SourcePins::from_plan(&plan, |name| {
        env.iter().any(|(k, v)| *k == name && *v == "1")
    })
}

const MXFP8_ALIASERS: [&str; 2] = ["ATLAS_QWEN4_DENSE_FP8", "ATLAS_QWEN4_DENSE_MXFP8"];

/// Each reader selector and the families it reads.
fn readers() -> Vec<(&'static str, Vec<PinFamily>)> {
    use PinFamily::*;
    vec![
        ("ATLAS_QWEN4_ROUTER_BF16", vec![Router]),
        ("ATLAS_QWEN4_HC_BF16", vec![Hc]),
        ("ATLAS_QWEN4_GDN_BF16", vec![Gdn]),
        ("ATLAS_QWEN4_ATTN_BF16", vec![Attn]),
        ("ATLAS_QWEN4_SHARED_BF16", vec![Shared]),
        ("ATLAS_QWEN4_LMHEAD_BF16", vec![LmHead]),
        (
            "ATLAS_QWEN4_PREFILL_NATIVE",
            vec![Attn, Gdn, Shared, Router, Hc],
        ),
        ("ATLAS_QWEN4_DENSE_FP8", vec![Attn, Gdn, Shared]),
        ("ATLAS_QWEN4_DENSE_MXFP8", vec![Attn, Gdn, Shared]),
    ]
}

fn resident(store: &WeightStore) -> BTreeSet<String> {
    names()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| store.contains(n) && store.get(n).is_ok())
        .collect()
}

fn family_names(family: PinFamily) -> BTreeSet<String> {
    names()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| PinFamily::of(n) == Some(family))
        .collect()
}

#[test]
fn family_classification_covers_every_release_target() {
    for (name, _) in names() {
        assert!(PinFamily::of(&name).is_some(), "{name}");
    }
    assert_eq!(PinFamily::of("lm_head.weight"), Some(PinFamily::LmHead));
    assert_eq!(
        PinFamily::of("model.hyper_connection_mixer.input_mix_weight_up"),
        Some(PinFamily::Hc)
    );
    assert_eq!(
        PinFamily::of(&format!("{LP}.mlp.experts.0.gate_proj.weight")),
        None
    );
    assert_eq!(
        PinFamily::of(&format!("{LP}.self_attn.q_norm.weight")),
        None
    );
    // ML-5: in_proj_a/b feed the interleaved BA copy; no reader holds them.
    for p in ["in_proj_a", "in_proj_b"] {
        assert_eq!(PinFamily::of(&format!("{LP}.linear_attn.{p}")), None, "{p}");
    }
}

#[test]
fn every_release_times_every_reader() {
    for (selector, families) in readers() {
        let pins = pins_for(&[(selector, "1")]);
        for release in RELEASES {
            let gpu = MockGpuBackend::new();
            let st = store(&gpu);
            let reader_only = resident(&st);
            let _scope = scoped(pins.clone());
            let released = release
                .run(&st, &gpu)
                .unwrap_or_else(|e| panic!("{selector} x {release:?}: {e:#}"));
            let target = family_names(release.family());
            let after = resident(&st);
            let lines = skipped_lines(LP, "test");
            let pinned = families.contains(&release.family())
                && (release.kind() == PinKind::Bf16Alias || MXFP8_ALIASERS.contains(&selector));
            if pinned {
                assert_eq!(
                    released, 0,
                    "{selector} x {release:?} released a pinned source"
                );
                assert_eq!(
                    after, reader_only,
                    "{selector} x {release:?}: resident set changed"
                );
                assert_eq!(lines.len(), 1, "{selector} x {release:?}: {lines:?}");
                assert!(
                    lines[0].starts_with(&format!("MXFP8_RELEASE_SKIPPED pinned_by={selector} ")),
                    "{lines:?}"
                );
            } else {
                assert!(
                    released > 0,
                    "{selector} x {release:?} pinned an unread source"
                );
                assert!(target.is_disjoint(&after), "{selector} x {release:?}");
                assert!(lines.is_empty(), "{selector} x {release:?}: {lines:?}");
            }
            // Nothing outside the release's own family moved.
            let others: BTreeSet<_> = reader_only.difference(&target).cloned().collect();
            assert!(others.is_subset(&after), "{selector} x {release:?}");
        }
    }
}

#[test]
fn the_umbrella_pins_what_prefill_pins() {
    let umbrella = pins_for(&[("ATLAS_QWEN4_DENSE_NATIVE", "1")]);
    for (name, _) in names() {
        assert!(
            umbrella.pinned_by(&name, PinKind::Bf16Alias).is_some(),
            "umbrella must pin {name}"
        );
        // ML-4: the umbrella has no MXFP8 aliaser, so the originals are released.
        assert_eq!(
            umbrella.pinned_by(&name, PinKind::Mxfp8Source),
            None,
            "{name}"
        );
    }
    // An explicit per-module 0 wins over the umbrella; prefill still pins HC.
    let no_hc = pins_for(&[
        ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
        ("ATLAS_QWEN4_HC_BF16", "0"),
    ]);
    let hc = format!("{LP}.attn_hyper_connection.input_mix_weight_down");
    assert_eq!(
        no_hc.pinned_by(&hc, PinKind::Bf16Alias),
        Some("ATLAS_QWEN4_PREFILL_NATIVE")
    );
}

#[test]
fn default_env_pins_nothing_and_releases_every_source() {
    let pins = pins_for(&[]);
    assert!(pins.is_empty());
    let gpu = MockGpuBackend::new();
    let st = store(&gpu);
    let _scope = scoped(pins);
    let released: usize = RELEASES
        .into_iter()
        .filter(|r| !matches!(r, Release::DedupShared))
        .map(|r| r.run(&st, &gpu).expect("release"))
        .sum();
    // 10 FP8 tensors x 64 B + 2 BF16 tensors x 128 B.
    assert_eq!(released, 10 * 64 + 2 * 128);
    assert!(resident(&st).is_empty());
    assert!(skipped_lines(LP, "test").is_empty());
    assert_eq!(take_total(), Skipped::default());
}

#[test]
fn skips_accumulate_into_the_load_total() {
    let gpu = MockGpuBackend::new();
    let st = store(&gpu);
    let _scope = scoped(pins_for(&[("ATLAS_QWEN4_DENSE_FP8", "1")]));
    for r in [Release::Mxfp8Attn, Release::Mxfp8Gdn, Release::Mxfp8Shared] {
        r.run(&st, &gpu).expect("release");
    }
    let lines = skipped_lines(LP, "layer");
    assert_eq!(
        lines,
        vec![format!(
            "MXFP8_RELEASE_SKIPPED pinned_by=ATLAS_QWEN4_DENSE_FP8 layer={LP} family=layer names=10 bytes=640"
        )]
    );
    assert_eq!(
        take_total(),
        Skipped {
            names: 10,
            bytes: 640
        }
    );
}

/// ML-4: ATTN_BF16 alone frees the attention MXFP8 originals and counts them
/// as reclaimed under a BF16 arm; DENSE_FP8 pins them.
#[test]
fn mxfp8_attention_originals_follow_the_aliasing_selector_only() {
    let gpu = MockGpuBackend::new();
    let st = store(&gpu);
    let _scope = scoped(pins_for(&[
        ("ATLAS_QWEN4_ATTN_BF16", "1"),
        ("ATLAS_QWEN4_DENSE_FP8", "0"),
    ]));
    assert_eq!(Release::Mxfp8Attn.run(&st, &gpu).expect("release"), 4 * 64);
    assert!(family_names(PinFamily::Attn).is_disjoint(&resident(&st)));
    assert_eq!(
        take_unpinned(),
        Skipped {
            names: 4,
            bytes: 256
        }
    );
    assert_eq!(
        total_lines(
            Skipped::default(),
            Skipped {
                names: 4,
                bytes: 256
            }
        ),
        vec!["MXFP8_RELEASE_UNPINNED_BF16_ARMS names=4 bytes=256 gib=0.00".to_string()]
    );
    drop(_scope);

    let gpu = MockGpuBackend::new();
    let st = store(&gpu);
    let _scope = scoped(pins_for(&[
        ("ATLAS_QWEN4_ATTN_BF16", "1"),
        ("ATLAS_QWEN4_DENSE_FP8", "1"),
    ]));
    assert_eq!(Release::Mxfp8Attn.run(&st, &gpu).expect("release"), 0);
    assert!(family_names(PinFamily::Attn).is_subset(&resident(&st)));
    assert_eq!(take_unpinned(), Skipped::default());
}

/// ML-1: a store-alias BF16 source passed to release_replaced_bf16_source is
/// kept when a reader pins it.
#[test]
fn replaced_bf16_store_alias_respects_the_pins() {
    let gpu = MockGpuBackend::new();
    let name = format!("{LP}.linear_attn.out_proj.weight");
    let ptr = gpu.alloc(128).expect("alloc");
    let st = WeightStore::from_map(HashMap::from([(
        name.clone(),
        WeightTensor {
            ptr,
            shape: vec![64],
            dtype: WeightDtype::BF16,
        },
    )]));
    let _scope = scoped(pins_for(&[("ATLAS_QWEN4_GDN_BF16", "1")]));
    let freed = super::linear_attn_arms::release_replaced_bf16_source(
        &st,
        &name,
        DenseWeight { weight: ptr },
        128,
        &gpu,
    )
    .expect("release");
    assert_eq!(freed, 0);
    assert!(st.get(&name).is_ok());
    assert_eq!(skipped_lines(LP, "linear_attn").len(), 1);
}

/// ML-2a: a pinned model-level mixer is counted in the load total, and the
/// total is emitted by the factory after the last release.
#[test]
fn pinned_final_mixer_is_counted_and_totals_print_after_the_last_release() {
    let gpu = MockGpuBackend::new();
    let prefix = "model.language_model.hyper_connection_mixer.input_mix_weight_down".to_string();
    let ptr = gpu.alloc(128).expect("alloc");
    let st = WeightStore::from_map(HashMap::from([(
        format!("{prefix}.weight"),
        WeightTensor {
            ptr,
            shape: vec![64],
            dtype: WeightDtype::BF16,
        },
    )]));
    let _scope = scoped(pins_for(&[("ATLAS_QWEN4_HC_BF16", "1")]));
    let released =
        dense_dedup::release_bf16_sources(&st, &[(prefix, DenseWeight { weight: ptr })], &gpu)
            .expect("release");
    assert_eq!(released, 0);
    assert_eq!(
        take_total(),
        Skipped {
            names: 1,
            bytes: 128
        }
    );

    let build = include_str!("../../../factory/build.rs");
    let emit = build
        .find("crate::source_pins::emit_totals();")
        .expect("factory emits totals");
    for call in [
        ".load_qwen4_final_mixer(",
        "release_unused_qwen4_bf16_lm_head(",
    ] {
        let last = build[..build.find("#[cfg(test)]\nmod ").unwrap_or(build.len())]
            .rfind(call)
            .expect(call);
        assert!(last < emit, "{call} must run before the totals print");
    }
    let layers = include_str!("../load_layers.rs");
    assert!(
        !layers.contains("source_pins::take_total()"),
        "load_layers must not drain the total early"
    );
}

/// ML-2b: the MTP mixer is never released by dense dedup.
#[test]
fn mtp_final_mixer_is_not_a_dedup_target() {
    assert!(!dense_dedup::is_target_layer_prefix(
        "mtp.hyper_connection_mixer"
    ));
    let layers = include_str!("../load_layers.rs");
    let body = &layers[layers
        .find("pub(super) fn release_final_mixer_sources(")
        .expect("fn")..];
    let body = &body[..body.find("\n}\n").expect("end")];
    assert!(
        body.contains("!dense_dedup::is_target_layer_prefix(prefix)"),
        "{body}"
    );
}

/// The umbrella never selects the FP8 arm; with ATLAS_QWEN4_DENSE_FP8=1 on
/// top (BF16 arms explicitly 0, the only admitted combination) the MXFP8
/// originals of attn/gdn/shared are pinned by DENSE_FP8, and released again
/// when it is off.
#[test]
fn umbrella_plus_dense_fp8_pins_the_mxfp8_originals() {
    let mxfp8: Vec<_> = names()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| {
            matches!(
                PinFamily::of(n),
                Some(PinFamily::Attn | PinFamily::Gdn | PinFamily::Shared)
            )
        })
        .collect();
    let zeros = [
        ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
        ("ATLAS_QWEN4_GDN_BF16", "0"),
        ("ATLAS_QWEN4_ATTN_BF16", "0"),
        ("ATLAS_QWEN4_SHARED_BF16", "0"),
    ];
    for env in [
        vec![
            ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
            ("ATLAS_QWEN4_DENSE_FP8", "1"),
        ],
        zeros
            .iter()
            .copied()
            .chain([("ATLAS_QWEN4_DENSE_FP8", "1")])
            .collect(),
    ] {
        let pins = pins_for(&env);
        for name in &mxfp8 {
            assert_eq!(
                pins.pinned_by(name, PinKind::Mxfp8Source),
                Some("ATLAS_QWEN4_DENSE_FP8"),
                "{env:?} {name}"
            );
        }
        let gpu = MockGpuBackend::new();
        let st = store(&gpu);
        let _scope = scoped(pins);
        for r in [Release::Mxfp8Attn, Release::Mxfp8Gdn, Release::Mxfp8Shared] {
            assert_eq!(r.run(&st, &gpu).expect("release"), 0, "{env:?} {r:?}");
        }
        assert_eq!(take_unpinned(), Skipped::default());
    }
    let off = pins_for(&zeros);
    for name in &mxfp8 {
        assert_eq!(off.pinned_by(name, PinKind::Mxfp8Source), None, "{name}");
    }
}
