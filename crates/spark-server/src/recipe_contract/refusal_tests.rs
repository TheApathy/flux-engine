// SPDX-License-Identifier: AGPL-3.0-only

//! Every refusal case, plus the accept paths next to them.

use super::catalog::Catalog;
use super::contract::{self, Effective};
use super::env_check::Source;
use super::facts::{CheckpointFacts, ServeFacts};
use super::refusal::{self, Refusal};
use super::test_fixtures::*;
use std::collections::BTreeMap;

fn run(env: &BTreeMap<String, String>) -> Result<Effective, Vec<Refusal>> {
    contract::evaluate(&loaded(), env, &serve(), &ckpt(), &catalog())
}

fn run_with(serve: &ServeFacts, ckpt: &CheckpointFacts) -> Vec<Refusal> {
    contract::evaluate(&loaded(), &good_env(), serve, ckpt, &catalog()).expect_err("must refuse")
}

fn plus(extra: &[(&str, &str)]) -> BTreeMap<String, String> {
    good_env().into_iter().chain(env(extra)).collect()
}

#[test]
fn the_declared_environment_is_accepted() {
    let e = run(&good_env()).expect("accepted");
    let gdn = e
        .env
        .numerics
        .iter()
        .find(|x| x.name == "ATLAS_QWEN4_PREFILL_GDN_FI")
        .expect("entry");
    assert_eq!(gdn.source, Source::Recipe);
    let attn = e
        .env
        .numerics
        .iter()
        .find(|x| x.name == "ATLAS_QWEN4_PREFILL_ATTN_FI")
        .expect("entry");
    assert_eq!(
        (attn.value.as_deref(), attn.source),
        (None, Source::Default)
    );
}

#[test]
fn an_undeclared_numerics_gate_refuses() {
    let r = run(&plus(&[("ATLAS_QWEN4_PREFILL_ATTN_FI", "1")])).expect_err("refused");
    assert_eq!(
        r,
        vec![Refusal::UnknownNumericsEnv(
            "ATLAS_QWEN4_PREFILL_ATTN_FI".into()
        )]
    );
    assert_eq!(
        refusal::render(&r),
        "RECIPE_REFUSED unknown_numerics_env=ATLAS_QWEN4_PREFILL_ATTN_FI"
    );
}

#[test]
fn a_declared_gate_with_another_value_refuses() {
    let r = run(&env(&[
        ("ATLAS_QWEN4_PREFILL_GDN_FI", "0"),
        ("ATLAS_MOE_T_LANES", "1"),
    ]))
    .expect_err("refused");
    assert!(
        matches!(&r[..], [Refusal::EnvMismatch { name, .. }] if name == "ATLAS_QWEN4_PREFILL_GDN_FI")
    );
}

#[test]
fn a_declared_gate_left_unset_refuses() {
    let r = run(&env(&[("ATLAS_MOE_T_LANES", "1")])).expect_err("refused");
    assert_eq!(
        r,
        vec![Refusal::EnvMissing("ATLAS_QWEN4_PREFILL_GDN_FI".into())]
    );
}

#[test]
fn a_flux_mirror_counts_as_its_atlas_reader() {
    let e = run(&env(&[
        ("FLUX_QWEN4_PREFILL_GDN_FI", "2"),
        ("ATLAS_MOE_T_LANES", "1"),
    ]))
    .expect("accepted");
    assert!(
        e.env
            .numerics
            .iter()
            .any(|x| x.name == "ATLAS_QWEN4_PREFILL_GDN_FI" && x.source == Source::Recipe)
    );
}

#[test]
fn the_atlas_spelling_wins_and_a_stale_flux_value_is_reported() {
    // main's alias sync never overwrites an ATLAS_ value, so the engine reads
    // GDN_FI=2 here; the stale FLUX_ value is listed, not refused.
    let e = run(&plus(&[("FLUX_QWEN4_PREFILL_GDN_FI", "1")])).expect("accepted");
    assert_eq!(
        e.env.alias_shadowed,
        vec!["FLUX_QWEN4_PREFILL_GDN_FI".to_string()]
    );
    let gdn = e
        .env
        .numerics
        .iter()
        .find(|x| x.name == "ATLAS_QWEN4_PREFILL_GDN_FI")
        .expect("entry");
    assert_eq!(gdn.value.as_deref(), Some("2"));
}

#[test]
fn a_flux_value_alone_is_checked_under_its_atlas_name() {
    let r = run(&env(&[
        ("FLUX_QWEN4_PREFILL_GDN_FI", "1"),
        ("ATLAS_MOE_T_LANES", "1"),
    ]))
    .expect_err("refused");
    assert!(
        matches!(&r[..], [Refusal::EnvMismatch { name, .. }] if name == "ATLAS_QWEN4_PREFILL_GDN_FI")
    );
}

#[test]
fn non_utf8_env_refuses_instead_of_vanishing() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let bad = OsString::from_vec(vec![0x66, 0xff]);
    let r = super::env_snapshot([
        (OsString::from("ATLAS_QWEN4_PREFILL_ATTN_FI"), bad),
        (OsString::from("PATH"), OsString::from_vec(vec![0xff])),
        (OsString::from("ATLAS_MOE_T_LANES"), OsString::from("1")),
    ])
    .expect_err("refused");
    assert_eq!(
        r,
        vec![Refusal::NonUtf8Env("ATLAS_QWEN4_PREFILL_ATTN_FI".into())]
    );
    let ok = super::env_snapshot([(OsString::from("FLUX_RECIPE"), OsString::from("/r.toml"))])
        .expect("ok");
    assert!(ok.is_empty(), "FLUX_RECIPE itself is not a gate");
}

#[test]
fn numerics_cli_flags_must_match_the_recipe_or_their_default() {
    let mut s = serve();
    s.cli.insert("--rope-yarn-factor", "4".into());
    let r = run_with(&s, &ckpt());
    assert!(
        matches!(
            &r[..],
            [Refusal::FlagMismatch {
                flag: "--rope-yarn-factor",
                ..
            }]
        ),
        "{r:?}"
    );
    let pinned = RECIPE.replace(
        "[sampling.defaults]",
        "[cli]\n\"--rope-yarn-factor\" = \"4\"\n\n[sampling.defaults]",
    );
    let l = loaded_from(&pinned, &catalog());
    assert!(contract::evaluate(&l, &good_env(), &s, &ckpt(), &catalog()).is_ok());
    let r = contract::evaluate(&l, &good_env(), &serve(), &ckpt(), &catalog())
        .expect_err("default now refused");
    assert!(matches!(
        &r[..],
        [Refusal::FlagMismatch {
            flag: "--rope-yarn-factor",
            ..
        }]
    ));
}

#[test]
fn unknown_cli_keys_are_invalid() {
    let text = RECIPE.replace(
        "[sampling.defaults]",
        "[cli]\n\"--max-seq-len\" = \"4\"\n\n[sampling.defaults]",
    );
    assert!(matches!(
        contract::parse(&text, &catalog()),
        Err(Refusal::Invalid(_))
    ));
}

#[test]
fn undeclared_route_gates_are_flagged_or_refused_by_policy() {
    let e = run(&plus(&[("ATLAS_QWEN4_PREFILL_WIDE", "8192")])).expect("flag policy accepts");
    assert_eq!(e.env.route_undeclared.len(), 1);
    let strict = RECIPE.replace("route_policy = \"flag\"", "route_policy = \"refuse\"");
    let l = loaded_from(&strict, &catalog());
    let r = contract::evaluate(
        &l,
        &plus(&[("ATLAS_QWEN4_PREFILL_WIDE", "8192")]),
        &serve(),
        &ckpt(),
        &catalog(),
    )
    .expect_err("refuse policy");
    assert_eq!(
        r,
        vec![Refusal::UnknownRouteEnv("ATLAS_QWEN4_PREFILL_WIDE".into())]
    );
}

#[test]
fn operational_env_is_range_checked() {
    assert!(run(&plus(&[("ATLAS_SCHED_BURST_GATHER_MS", "50")])).is_ok());
    let r = run(&plus(&[("ATLAS_SCHED_BURST_GATHER_MS", "500")])).expect_err("refused");
    assert!(matches!(&r[..], [Refusal::OperationalOutOfRange { .. }]));
    let e =
        run(&plus(&[("ATLAS_FULL_PROFILE", "1")])).expect("undeclared observability is allowed");
    assert!(
        e.env
            .operational
            .iter()
            .any(|x| x.name == "ATLAS_FULL_PROFILE" && x.source == Source::EnvAllowed)
    );
}

#[test]
fn names_without_a_reader_are_ignored_and_listed() {
    let e = run(&plus(&[("ATLAS_NO_SUCH_GATE", "1")])).expect("accepted");
    assert_eq!(e.env.ignored, vec!["ATLAS_NO_SUCH_GATE".to_string()]);
}

#[test]
fn a_precision_selector_that_contradicts_the_plan_refuses() {
    let r = run(&plus(&[("ATLAS_QWEN4_LMHEAD_BF16", "1")])).expect_err("refused");
    assert!(matches!(
        &r[..],
        [Refusal::PrecisionMismatch {
            family: "lm_head",
            ..
        }]
    ));
}

fn native_text() -> String {
    [
        "router",
        "hyper_connection",
        "gdn_proj",
        "attention_proj",
        "shared_expert",
        "lm_head",
    ]
    .iter()
    .fold(RECIPE.to_string(), |t, f| {
        t.replace(
            &format!("{f} = \"nvfp4_requant\""),
            &format!("{f} = \"bf16\""),
        )
    })
    .replace("prefill_dense = \"fast_fp8\"", "prefill_dense = \"native\"")
}

#[test]
fn a_plan_this_build_cannot_serve_refuses() {
    let l = loaded_from(&native_text(), &catalog());
    let r =
        contract::evaluate(&l, &good_env(), &serve(), &ckpt(), &catalog()).expect_err("refused");
    assert!(r.contains(&Refusal::PrecisionUnsupported {
        family: "router",
        selector: "ATLAS_QWEN4_ROUTER_BF16"
    }));
    assert!(r.contains(&Refusal::PrecisionUnsupported {
        family: "prefill_dense",
        selector: "ATLAS_QWEN4_PREFILL_NATIVE"
    }));
}

#[test]
fn the_umbrella_satisfies_the_native_plan_and_explicit_zero_beats_it() {
    let c = native_catalog();
    let l = loaded_from(&native_text(), &c);
    let on = plus(&[("ATLAS_QWEN4_DENSE_NATIVE", "1")]);
    let e = contract::evaluate(&l, &on, &serve(), &ckpt(), &c).expect("umbrella accepted");
    assert!(
        e.precision
            .iter()
            .any(|f| f.family == "router" && f.format == "bf16")
    );
    let off = plus(&[
        ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
        ("ATLAS_QWEN4_HC_BF16", "0"),
    ]);
    let r = contract::evaluate(&l, &off, &serve(), &ckpt(), &c).expect_err("refused");
    assert!(matches!(
        &r[..],
        [Refusal::PrecisionMismatch {
            family: "hyper_connection",
            ..
        }]
    ));
}

fn fp8_dense_text() -> String {
    RECIPE
        .replace("router = \"nvfp4_requant\"", "router = \"bf16\"")
        .replace(
            "hyper_connection = \"nvfp4_requant\"",
            "hyper_connection = \"bf16\"",
        )
        .replace("gdn_proj = \"nvfp4_requant\"", "gdn_proj = \"fp8_e4m3\"")
        .replace(
            "attention_proj = \"nvfp4_requant\"",
            "attention_proj = \"fp8_e4m3\"",
        )
        .replace(
            "shared_expert = \"nvfp4_requant\"",
            "shared_expert = \"fp8_e4m3\"",
        )
        .replace("lm_head = \"nvfp4_requant\"", "lm_head = \"bf16\"")
}

#[test]
fn fp8_dense_coexists_with_bf16_router_hc_lmhead_and_refuses_gdn_attn_shared_bf16_conflicts() {
    let c = Catalog::parse(&format!(
        "{CATALOG}{NATIVE_EXTRA}ATLAS_QWEN4_DENSE_FP8\tNUMERICS\n"
    ))
    .expect("catalog");
    let l = loaded_from(&fp8_dense_text(), &c);

    // 1) Explicit DENSE_FP8=1 + ROUTER/HC/LMHEAD_BF16=1 resolves router/hc/lm_head to bf16 and gdn/attn/shared to fp8_e4m3.
    let explicit = plus(&[
        ("ATLAS_QWEN4_DENSE_FP8", "1"),
        ("ATLAS_QWEN4_ROUTER_BF16", "1"),
        ("ATLAS_QWEN4_HC_BF16", "1"),
        ("ATLAS_QWEN4_LMHEAD_BF16", "1"),
    ]);
    let e = contract::evaluate(&l, &explicit, &serve(), &ckpt(), &c)
        .expect("explicit fp8+bf16 accepted");
    for fam in ["router", "hyper_connection", "lm_head"] {
        assert!(
            e.precision
                .iter()
                .any(|f| f.family == fam && f.format == "bf16"),
            "{fam}"
        );
    }
    for fam in ["gdn_proj", "attention_proj", "shared_expert"] {
        assert!(
            e.precision
                .iter()
                .any(|f| f.family == fam && f.format == "fp8_e4m3"),
            "{fam}"
        );
    }

    // 2) Umbrella DENSE_NATIVE=1 with GDN/ATTN/SHARED_BF16=0 and PREFILL_NATIVE=0 + DENSE_FP8=1 is accepted.
    let umbrella_override = plus(&[
        ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
        ("ATLAS_QWEN4_GDN_BF16", "0"),
        ("ATLAS_QWEN4_ATTN_BF16", "0"),
        ("ATLAS_QWEN4_SHARED_BF16", "0"),
        ("ATLAS_QWEN4_PREFILL_NATIVE", "0"),
        ("ATLAS_QWEN4_DENSE_FP8", "1"),
    ]);
    let e2 = contract::evaluate(&l, &umbrella_override, &serve(), &ckpt(), &c)
        .expect("umbrella override accepted");
    assert_eq!(e2.precision, e.precision);

    // 3) Genuine conflicts on GDN, ATTN, or SHARED with DENSE_FP8=1 refuse with got="bf16+fp8_e4m3".
    for (knob, fam) in [
        ("ATLAS_QWEN4_GDN_BF16", "gdn_proj"),
        ("ATLAS_QWEN4_ATTN_BF16", "attention_proj"),
        ("ATLAS_QWEN4_SHARED_BF16", "shared_expert"),
    ] {
        let mut conflict = explicit.clone();
        conflict.insert(knob.into(), "1".into());
        let r = contract::evaluate(&l, &conflict, &serve(), &ckpt(), &c)
            .expect_err("conflict must refuse");
        assert_eq!(
            r,
            vec![Refusal::PrecisionMismatch {
                family: fam,
                want: "fp8_e4m3".into(),
                got: "bf16+fp8_e4m3".into(),
            }],
            "{knob}"
        );
    }

    // 4) Unoverridden umbrella DENSE_NATIVE=1 + DENSE_FP8=1 refuses gdn_proj, attention_proj, shared_expert, and prefill_dense,
    //    but does NOT report false conflicts on router, hyper_connection, or lm_head.
    let umbrella_conflict = plus(&[
        ("ATLAS_QWEN4_DENSE_NATIVE", "1"),
        ("ATLAS_QWEN4_DENSE_FP8", "1"),
    ]);
    let r_umb = contract::evaluate(&l, &umbrella_conflict, &serve(), &ckpt(), &c)
        .expect_err("umbrella conflict must refuse");
    assert_eq!(
        r_umb,
        vec![
            Refusal::PrecisionMismatch {
                family: "gdn_proj",
                want: "fp8_e4m3".into(),
                got: "bf16+fp8_e4m3".into()
            },
            Refusal::PrecisionMismatch {
                family: "attention_proj",
                want: "fp8_e4m3".into(),
                got: "bf16+fp8_e4m3".into()
            },
            Refusal::PrecisionMismatch {
                family: "shared_expert",
                want: "fp8_e4m3".into(),
                got: "bf16+fp8_e4m3".into()
            },
            Refusal::PrecisionMismatch {
                family: "prefill_dense",
                want: "fast_fp8".into(),
                got: "native".into()
            },
        ]
    );
}

#[test]
fn serve_flags_outside_the_recipe_refuse() {
    let r = run_with(
        &ServeFacts {
            max_batch_size: 8,
            ..serve()
        },
        &ckpt(),
    );
    assert!(matches!(
        &r[..],
        [Refusal::FlagMismatch {
            flag: "--max-batch-size",
            ..
        }]
    ));
    let r = run_with(
        &ServeFacts {
            mtp_speculative: true,
            ..serve()
        },
        &ckpt(),
    );
    assert!(matches!(
        &r[..],
        [Refusal::FlagMismatch {
            flag: "speculation",
            ..
        }]
    ));
    let r = run_with(
        &ServeFacts {
            max_seq_len: 524_288,
            ..serve()
        },
        &ckpt(),
    );
    assert!(matches!(
        &r[..],
        [Refusal::FlagMismatch {
            flag: "--max-seq-len",
            ..
        }]
    ));
}

#[test]
fn checkpoint_pins_refuse_on_any_difference() {
    let other = "0".repeat(64);
    let bad_sha = CheckpointFacts {
        digests: [("config.json".to_string(), Ok(other))]
            .into_iter()
            .collect(),
        ..ckpt()
    };
    assert!(matches!(
        &run_with(&serve(), &bad_sha)[..],
        [Refusal::CheckpointPin { .. }]
    ));
    let missing = CheckpointFacts {
        digests: [(
            "config.json".to_string(),
            Err("open: not found".to_string()),
        )]
        .into_iter()
        .collect(),
        ..ckpt()
    };
    assert!(matches!(
        &run_with(&serve(), &missing)[..],
        [Refusal::CheckpointPin { .. }]
    ));
    let wrong_type = CheckpointFacts {
        model_type: Some("qwen3_5".into()),
        ..ckpt()
    };
    assert!(
        matches!(&run_with(&serve(), &wrong_type)[..], [Refusal::CheckpointPin { file, .. }] if file == "config.json")
    );
}

#[test]
fn all_refusals_are_reported_together() {
    let r = contract::evaluate(
        &loaded(),
        &env(&[("ATLAS_QWEN4_PREFILL_ATTN_FI", "1")]),
        &ServeFacts {
            max_batch_size: 8,
            ..serve()
        },
        &CheckpointFacts {
            model_type: None,
            ..ckpt()
        },
        &Catalog::parse(CATALOG).expect("catalog"),
    )
    .expect_err("refused");
    assert_eq!(r.len(), 5, "{r:?}");
    assert!(
        refusal::render(&r)
            .starts_with("RECIPE_REFUSED unknown_numerics_env=ATLAS_QWEN4_PREFILL_ATTN_FI ; ")
    );
}

#[test]
fn a_flux_only_numerics_gate_is_refused_not_ignored() {
    let c = Catalog::parse(&format!("{CATALOG}FLUX_TARGET_QUANT\tNUMERICS\n")).expect("catalog");
    let l = loaded_from(RECIPE, &c);
    let r = contract::evaluate(
        &l,
        &plus(&[("FLUX_TARGET_QUANT", "nvfp4")]),
        &serve(),
        &ckpt(),
        &c,
    )
    .expect_err("refused");
    assert_eq!(
        r,
        vec![Refusal::UnknownNumericsEnv("FLUX_TARGET_QUANT".into())]
    );
}

#[test]
fn malformed_umbrella_and_fp8_values_refuse() {
    let c = native_catalog();
    for (name, value) in [
        ("ATLAS_QWEN4_DENSE_NATIVE", "true"),
        ("ATLAS_QWEN4_DENSE_NATIVE", "2"),
    ] {
        let r = contract::evaluate(
            &loaded_from(RECIPE, &c),
            &plus(&[(name, value)]),
            &serve(),
            &ckpt(),
            &c,
        )
        .expect_err("refused");
        assert!(
            r.iter()
                .all(|x| matches!(x, Refusal::PrecisionMismatch { got, .. } if got.contains(name))),
            "{r:?}"
        );
    }
    let c =
        Catalog::parse(&format!("{CATALOG}ATLAS_QWEN4_DENSE_FP8\tNUMERICS\n")).expect("catalog");
    let r = contract::evaluate(
        &loaded_from(RECIPE, &c),
        &plus(&[("ATLAS_QWEN4_DENSE_FP8", "yes")]),
        &serve(),
        &ckpt(),
        &c,
    )
    .expect_err("refused");
    assert!(
        r.iter().any(|x| matches!(
            x,
            Refusal::PrecisionMismatch {
                family: "gdn_proj",
                ..
            }
        )),
        "{r:?}"
    );
}
