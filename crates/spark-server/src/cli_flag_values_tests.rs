// SPDX-License-Identifier: AGPL-3.0-only

use super::options_for_flag;

#[test]
fn kv_cache_dtype_and_dflash_methods_offer_their_authoritative_sets() {
    // Derived, not listed: this is the whole point of pointing at the enum.
    // A variant added to `KvCacheDtype` must appear here without anyone
    // remembering to edit this module.
    let got = options_for_flag("kv-cache-dtype").expect("kv-cache-dtype is enumerated");
    let want: Vec<String> = spark_runtime::kv_cache::KvCacheDtype::ALL
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(got, want);
    assert!(
        !got.is_empty(),
        "an empty picker would be indistinguishable from free text"
    );
    assert_eq!(
        options_for_flag("dflash-method"),
        Some(vec!["flat".to_string()]),
        "DDTree must not be offered until its runtime can fail closed"
    );
    assert_eq!(
        options_for_flag("dflash-draft-sample-method"),
        Some(vec!["greedy".to_string(), "probabilistic".to_string()]),
        "the picker and fail-closed validator must share one value set"
    );

    // And the negative arm, so this cannot pass by returning Some(..) always.
    assert!(
        options_for_flag("max-seq-len").is_none(),
        "numbers stay free text"
    );
    assert!(options_for_flag("not-a-flag").is_none());
}

#[test]
fn the_long_name_is_taken_without_leading_dashes() {
    // A caller passing "--kv-cache-dtype" gets None, which would silently turn
    // the picker into a text field. Pinned so the convention cannot drift.
    assert!(options_for_flag("--kv-cache-dtype").is_none());
}

#[test]
fn validation_refuses_a_bad_enum_and_names_every_problem_not_just_the_first() {
    use clap::Parser as _;
    let ok = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);
    assert!(
        crate::cli::validate_serve_args(&ok).is_ok(),
        "clap defaults are valid"
    );

    // One bad value: named, with the accepted set.
    let mut bad = ok.clone();
    bad.scheduling_policy = "nonsense".into();
    let e = crate::cli::validate_serve_args(&bad).expect_err("refused");
    assert!(e.contains("scheduling-policy"), "{e}");
    assert!(
        e.contains("fifo") && e.contains("slai"),
        "the set is shown: {e}"
    );

    // TWO bad values: both reported. An operator who fixes one and hits the
    // same wall again learned nothing the first message could not have said.
    let mut worse = bad.clone();
    worse.kv_cache_dtype = "not-a-dtype".into();
    let e2 = crate::cli::validate_serve_args(&worse).expect_err("refused");
    assert!(e2.contains("scheduling-policy"), "{e2}");
    assert!(e2.contains("kv-cache-dtype"), "{e2}");
}

#[test]
fn a_fraction_out_of_range_and_a_stranded_pair_are_both_caught() {
    use clap::Parser as _;
    let ok = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);

    // clap takes 9.0 as a valid f64; it is not a valid FRACTION.
    let mut over = ok.clone();
    over.gpu_memory_utilization = 9.0;
    let e = crate::cli::validate_serve_args(&over).expect_err("refused");
    assert!(
        e.contains("gpu-memory-utilization") && e.contains("fraction"),
        "{e}"
    );

    // The PAIR, not the field: num-drafts alone is meaningless.
    let mut stranded = ok.clone();
    stranded.num_drafts = 2;
    stranded.speculative = false;
    stranded.self_speculative = false;
    stranded.ngram_speculative = false;
    stranded.dflash = false;
    let e2 = crate::cli::validate_serve_args(&stranded).expect_err("refused");
    assert!(
        e2.contains("num-drafts") && e2.contains("speculative"),
        "{e2}"
    );

    // And the same num-drafts is fine once a method is on — otherwise this
    // check would just forbid the flag.
    let mut fine = stranded.clone();
    fine.speculative = true;
    assert!(crate::cli::validate_serve_args(&fine).is_ok());
}

#[test]
fn serving_widths_reject_zero_and_rows_above_c128_before_load() {
    use clap::Parser as _;
    let base = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);

    let mut zero_batch = base.clone();
    zero_batch.max_batch_size = 0;
    let error = crate::cli::validate_serve_args(&zero_batch).expect_err("C0 must fail");
    assert!(
        error.contains("--max-batch-size must be at least 1"),
        "{error}"
    );

    let mut too_wide = base.clone();
    let ceiling = spark_runtime::buffers::DECODE_META_MAX_ROWS;
    too_wide.max_batch_size = ceiling + 1;
    let error = crate::cli::validate_serve_args(&too_wide)
        .expect_err("batch above decode-metadata ceiling must fail");
    assert!(
        error.contains(&format!("decode-metadata ceiling of {ceiling}")),
        "{error}"
    );

    let mut zero_queue = base;
    zero_queue.max_num_seqs = 0;
    let error = crate::cli::validate_serve_args(&zero_queue).expect_err("zero channel must fail");
    assert!(
        error.contains("--max-num-seqs must be at least 1"),
        "{error}"
    );
}

#[test]
fn auxiliary_spec_modes_cannot_silently_shadow_another_mode() {
    use clap::Parser as _;
    let base = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);
    assert!(crate::cli::validate_serve_args(&base).is_ok());

    // Native target MTP is an intentional second proposer arm for DFlash.
    let mut dual_arm = base.clone();
    dual_arm.speculative = true;
    assert!(crate::cli::validate_serve_args(&dual_arm).is_ok());

    let mut self_spec = base.clone();
    self_spec.self_speculative = true;
    let self_error = crate::cli::validate_serve_args(&self_spec).expect_err("refused");
    assert!(self_error.contains("--self-speculative"), "{self_error}");
    assert!(self_error.contains("silently bypass"), "{self_error}");

    let mut ngram = base.clone();
    ngram.ngram_speculative = true;
    let ngram_error = crate::cli::validate_serve_args(&ngram).expect_err("refused");
    assert!(ngram_error.contains("--ngram-speculative"), "{ngram_error}");

    let mut both = self_spec;
    both.ngram_speculative = true;
    let both_error = crate::cli::validate_serve_args(&both).expect_err("refused");
    assert!(both_error.contains("--self-speculative"), "{both_error}");
    assert!(both_error.contains("--ngram-speculative"), "{both_error}");

    for flag in ["--self-speculative", "--ngram-speculative"] {
        let args = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--speculative", flag]);
        let error = crate::cli::validate_serve_args(&args).expect_err("shadowing refused");
        assert!(error.contains(flag), "{flag}: {error}");
        assert!(error.contains("sole supported dual-arm"), "{flag}: {error}");
    }
}

#[test]
fn dflash_rejects_unsafe_or_misreported_runtime_configuration() {
    use clap::Parser as _;
    let base = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);

    let mut bad_quant = base.clone();
    bad_quant.dflash_quantization = "nvfp44".into();
    let quant_error = crate::cli::validate_serve_args(&bad_quant).expect_err("refused");
    assert!(
        quant_error.contains("--dflash-quantization"),
        "{quant_error}"
    );
    assert!(
        quant_error.contains("bf16") && quant_error.contains("nvfp4"),
        "{quant_error}"
    );

    let mut cached = base;
    cached.enable_prefix_caching = true;
    let cache_error = crate::cli::validate_serve_args(&cached).expect_err("refused");
    assert!(
        cache_error.contains("--enable-prefix-caching"),
        "{cache_error}"
    );
    assert!(
        cache_error.contains("correctness regression"),
        "{cache_error}"
    );
}

#[test]
fn dflash_draft_sample_method_is_closed_and_non_greedy_cannot_be_inert() {
    use clap::Parser as _;
    let greedy = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);
    assert_eq!(greedy.dflash_draft_sample_method, "greedy");
    assert!(crate::cli::validate_serve_args(&greedy).is_ok());

    let sampled = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--dflash",
        "--dflash-draft-sample-method",
        "probabilistic",
    ]);
    let error = crate::cli::validate_serve_args(&sampled)
        .expect_err("probabilistic DFlash must remain default-off");
    assert!(
        error.contains("--experimental-dflash-probabilistic"),
        "{error}"
    );

    let qualified = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--dflash",
        "--dflash-draft-sample-method",
        "probabilistic",
        "--experimental-dflash-probabilistic",
    ]);
    crate::cli::validate_serve_args(&qualified)
        .expect("the exact default-off sampled-DFlash contract is admitted");

    let inert_qualification = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--dflash",
        "--experimental-dflash-probabilistic",
    ]);
    let error = crate::cli::validate_serve_args(&inert_qualification)
        .expect_err("qualification flag cannot be inert on greedy");
    assert!(error.contains("exact pair"), "{error}");

    let dual_arm = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--dflash",
        "--speculative",
        "--dflash-draft-sample-method",
        "probabilistic",
        "--experimental-dflash-probabilistic",
    ]);
    let error = crate::cli::validate_serve_args(&dual_arm)
        .expect_err("MTP cannot publish a sampled DFlash q frame");
    assert!(error.contains("MTP arm"), "{error}");

    let inert = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--dflash-draft-sample-method",
        "probabilistic",
    ]);
    let error = crate::cli::validate_serve_args(&inert).expect_err("inert mode refused");
    assert!(error.contains("requires --dflash"), "{error}");

    let mut invalid = greedy;
    invalid.dflash_draft_sample_method = "roulette".to_string();
    let error = crate::cli::validate_serve_args(&invalid).expect_err("unknown mode refused");
    assert!(error.contains("dflash-draft-sample-method"), "{error}");
    assert!(
        error.contains("greedy") && error.contains("probabilistic"),
        "{error}"
    );
}

#[test]
fn dflash_admits_single_rank_and_tp2_ep1_pp1_and_refuses_other_topologies() {
    use clap::Parser as _;
    let base = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);
    assert!(crate::cli::validate_serve_args(&base).is_ok());

    for argv in [
        &["spark", "org/m", "--dflash", "--tp-size", "2"][..],
        &[
            "spark",
            "org/m",
            "--dflash",
            "--world-size",
            "2",
            "--tp-size",
            "2",
            "--ep-size",
            "1",
            "--pp-size",
            "1",
        ][..],
        &[
            "spark",
            "org/m",
            "--dflash",
            "--world-size",
            "2",
            "--tp-size",
            "2",
            "--ep-size",
            "2",
            "--pp-size",
            "1",
        ][..],
        &["spark", "org/m", "--dflash", "--tp-size", "4"][..],
        &[
            "spark",
            "org/m",
            "--dflash",
            "--world-size",
            "4",
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--pp-size",
            "1",
        ][..],
    ] {
        let args = crate::cli::ServeArgs::parse_from(argv);
        assert!(
            crate::cli::validate_serve_args(&args).is_ok(),
            "TP=2/4 (EP=1 or EP=TP, PP=1) DFlash must be admitted: {argv:?}"
        );
    }

    for (argv, field) in [
        (
            &["spark", "org/m", "--dflash", "--world-size", "2"][..],
            "world_size",
        ),
        (
            &["spark", "org/m", "--dflash", "--ep-size", "2"][..],
            "ep_size",
        ),
        (
            &["spark", "org/m", "--dflash", "--pp-size", "2"][..],
            "pp_size",
        ),
        (
            &["spark", "org/m", "--dflash", "--tp-size", "3"][..],
            "tp_size",
        ),
        (
            &[
                "spark",
                "org/m",
                "--dflash",
                "--world-size",
                "4",
                "--tp-size",
                "4",
                "--ep-size",
                "2",
            ][..],
            "ep_size",
        ),
    ] {
        let args = crate::cli::ServeArgs::parse_from(argv);
        let error = crate::cli::validate_serve_args(&args)
            .expect_err("unqualified multi-rank DFlash refused");
        assert!(error.contains("--dflash"), "{argv:?}: {error}");
        assert!(error.contains("single-rank or TP=2/4"), "{argv:?}: {error}");
        assert!(error.contains(field), "{argv:?}: {error}");
        assert!(error.contains("not qualified"), "{argv:?}: {error}");
    }

    for flag in ["--world-size", "--tp-size", "--ep-size", "--pp-size"] {
        let plain = crate::cli::ServeArgs::parse_from(["spark", "org/m", flag, "2"]);
        assert!(
            crate::cli::validate_serve_args(&plain).is_ok(),
            "non-DFlash topology must remain available: {flag}"
        );
    }
}

#[test]
fn disagg_role_and_pp_size_validation_are_enforced_before_load() {
    use clap::Parser as _;
    assert_eq!(
        options_for_flag("role"),
        Some(vec![
            "unified".to_string(),
            "prefill".to_string(),
            "decode".to_string()
        ])
    );

    let prefill = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--role", "prefill"]);
    assert!(crate::cli::validate_serve_args(&prefill).is_ok());

    let decode = crate::cli::ServeArgs::parse_from([
        "spark",
        "org/m",
        "--role",
        "decode",
        "--disagg-peer",
        "http://10.0.0.1:8000",
    ]);
    assert!(crate::cli::validate_serve_args(&decode).is_ok());

    let mut bad_role = prefill.clone();
    bad_role.role = "worker".into();
    let err = crate::cli::validate_serve_args(&bad_role).expect_err("unknown role refused");
    assert!(err.contains("--role"), "{err}");
    assert!(
        err.contains("unified") && err.contains("prefill") && err.contains("decode"),
        "{err}"
    );

    let mut zero_pp = prefill.clone();
    zero_pp.pp_size = 0;
    let err = crate::cli::validate_serve_args(&zero_pp).expect_err("pp_size=0 refused");
    assert!(err.contains("--pp-size must be at least 1"), "{err}");

    let mut prefill_with_peer = prefill;
    prefill_with_peer.disagg_peer = Some("http://10.0.0.2:8000".into());
    let err =
        crate::cli::validate_serve_args(&prefill_with_peer).expect_err("prefill+peer refused");
    assert!(err.contains("--disagg-peer"), "{err}");
}

#[test]
fn auxiliary_speculation_is_single_rank_until_rollback_protocol_is_qualified() {
    use clap::Parser as _;
    for mode in ["--self-speculative", "--ngram-speculative"] {
        for flag in ["--world-size", "--tp-size", "--ep-size"] {
            let args = crate::cli::ServeArgs::parse_from(["spark", "org/m", mode, flag, "2"]);
            let error = crate::cli::validate_serve_args(&args).expect_err("multi-rank refused");
            assert!(error.contains(mode), "{mode} {flag}: {error}");
            assert!(error.contains("single-rank"), "{mode} {flag}: {error}");
            assert!(
                error.contains("checkpoint/rollback"),
                "{mode} {flag}: {error}"
            );
        }
    }
}

#[test]
fn dflash_method_and_unwired_ddtree_fields_fail_closed() {
    use clap::Parser as _;
    let flat = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);
    assert!(crate::cli::validate_serve_args(&flat).is_ok());

    let mut tree = flat.clone();
    tree.dflash_method = "ddtree".into();
    let method_error = crate::cli::validate_serve_args(&tree).expect_err("DDTree is unavailable");
    assert!(method_error.contains("--dflash-method"), "{method_error}");
    assert!(method_error.contains("flat"), "{method_error}");
    assert!(method_error.contains("fail closed"), "{method_error}");

    let mut fields = flat;
    fields.ddtree_budget = Some(12);
    fields.ddtree_top_k = 4;
    fields.ddtree_temperature = 0.75;
    fields.ddtree_chain_seed = false;
    let field_error = crate::cli::validate_serve_args(&fields).expect_err("fields are unwired");
    for flag in [
        "--ddtree-budget",
        "--ddtree-top-k",
        "--ddtree-temperature",
        "--ddtree-chain-seed",
    ] {
        assert!(field_error.contains(flag), "{flag}: {field_error}");
    }
}

#[test]
fn environment_cannot_override_the_flat_dflash_runtime_contract() {
    use clap::Parser as _;
    let args = crate::cli::ServeArgs::parse_from(["spark", "org/m", "--dflash"]);

    crate::cli::validate_dflash_runtime_environment_with(&args, |_| None)
        .expect("an empty environment preserves flat");
    crate::cli::validate_dflash_runtime_environment_with(&args, |name| {
        (name == "ATLAS_DFLASH_METHOD").then(|| "flat".to_string())
    })
    .expect("an explicit matching flat environment is harmless");

    let method_error = crate::cli::validate_dflash_runtime_environment_with(&args, |name| {
        (name == "ATLAS_DFLASH_METHOD").then(|| "ddtree".to_string())
    })
    .expect_err("the environment cannot override CLI authority");
    assert!(
        method_error.contains("ATLAS_DFLASH_METHOD"),
        "{method_error}"
    );
    assert!(
        method_error.contains("--dflash-method flat"),
        "{method_error}"
    );

    let mut unsupported_cli = args.clone();
    unsupported_cli.dflash_method = "ddtree".into();
    let cli_error =
        crate::cli::validate_dflash_runtime_environment_with(&unsupported_cli, |_| None)
            .expect_err(
                "the load gate must enforce the method even if a caller skipped validation",
            );
    assert!(cli_error.contains("--dflash-method"), "{cli_error}");
    assert!(cli_error.contains("silently execute"), "{cli_error}");

    let legacy_error = crate::cli::validate_dflash_runtime_environment_with(&args, |name| {
        (name == "DDTREE_TOP_K").then(|| "8".to_string())
    })
    .expect_err("legacy DDTree tuning must not be silently ignored");
    assert!(legacy_error.contains("DDTREE_TOP_K"), "{legacy_error}");

    // Batched flat-DFlash verify controls are opt-in ("0" / "1") and fail closed on other values.
    for name in [
        "ATLAS_DFLASH_BATCHED_VERIFY",
        "ATLAS_DFLASH_BATCHED_QKV",
        "ATLAS_DFLASH_BATCHED_VERIFY_LOG",
    ] {
        for ok_val in ["0", "1"] {
            crate::cli::validate_dflash_runtime_environment_with(&args, |candidate| {
                (candidate == name).then(|| ok_val.to_string())
            })
            .unwrap_or_else(|e| panic!("{name}={ok_val} should be admitted: {e}"));
        }
        let bad = crate::cli::validate_dflash_runtime_environment_with(&args, |candidate| {
            (candidate == name).then(|| "2".to_string())
        })
        .expect_err("non-boolean batched DFlash control must fail closed");
        assert!(bad.contains(name), "{name}: {bad}");
    }

    for name in ["ATLAS_DFLASH_RECYCLE", "ATLAS_DFLASH_DEEPLOOP"] {
        let error = crate::cli::validate_dflash_runtime_environment_with(&args, |candidate| {
            (candidate == name).then(|| "1".to_string())
        })
        .expect_err("a stale experimental direct-CLI override must fail before load");
        assert!(error.contains(name), "{name}: {error}");
    }
}
