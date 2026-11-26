// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// (base_s, explicit, max_tokens, prompt_len) -> (total_s, source).
/// Defaults: 8 tok/s decode floor, 200 tok/s prefill floor, 30 s slack,
/// 7200 s ceiling, stall off.
#[test]
fn timeout_derivation_table() {
    use TimeoutSource::*;
    let rows: &[(u32, Option<f32>, usize, usize, Option<f64>, TimeoutSource)] = &[
        // small request: base dominates
        (300, None, 1024, 500, Some(300.0), Env),
        // 2048/8=256 + 3 + 30 = 289 < 300
        (300, None, 2048, 500, Some(300.0), Env),
        // arm B shape: 16384/8=2048 + ceil(2000/200)=10 + 30 = 2088
        (300, None, 16384, 2000, Some(2088.0), Derived),
        // 32768/8=4096 + 82 (16384/200=81.92) + 30 = 4208
        (300, None, 32768, 16384, Some(4208.0), Derived),
        // ceiling: 131072/8 = 16384 -> capped at 7200
        (300, None, 131072, 1000, Some(7200.0), Derived),
        // env base above the ceiling is honoured (ceiling caps derived only)
        (9000, None, 131072, 1000, Some(9000.0), Env),
        // operator base above derived stays
        (1800, None, 8192, 1000, Some(1800.0), Env),
        // base 0 keeps "no total timeout"
        (0, None, 32768, 1000, None, Env),
        // explicit request timeout wins, even below derived
        (300, Some(60.0), 32768, 1000, Some(60.0), Request),
        // explicit 0 = none
        (300, Some(0.0), 32768, 1000, None, Request),
    ];
    for &(base, explicit, max_tokens, prompt, want_total, want_src) in rows {
        let cfg = TimeoutConfig::with_base(base);
        let p = derive_timeout_policy(&cfg, explicit, max_tokens, prompt);
        assert_eq!(
            (p.total_s, p.source),
            (want_total, want_src),
            "base={base} explicit={explicit:?} max_tokens={max_tokens} prompt={prompt}"
        );
        assert_eq!(p.stall_s, None, "stall timeout is off by default");
    }
}

#[test]
fn derived_total_respects_custom_floor_and_ceiling() {
    let cfg = TimeoutConfig {
        min_decode_toks_per_s: 31.0,
        max_s: 1000,
        ..TimeoutConfig::with_base(300)
    };
    // ceil(16384/31)=529 + 0 + 30
    assert_eq!(derived_total_s(&cfg, 16384, 0), 559);
    assert_eq!(derived_total_s(&cfg, 1_000_000, 0), 1000);
}

#[test]
fn stall_zero_disables_the_stall_limit() {
    let cfg = TimeoutConfig {
        stall_s: 0,
        ..TimeoutConfig::with_base(300)
    };
    assert_eq!(derive_timeout_policy(&cfg, None, 10, 10).stall_s, None);
}

#[test]
fn derived_never_cuts_a_request_decoding_at_the_floor_rate() {
    // At exactly the floor rate the request needs max_tokens/floor seconds;
    // the derived total must be strictly larger for every size below the cap.
    let cfg = TimeoutConfig::with_base(300);
    for max_tokens in [1usize, 255, 4096, 16384, 32768, 50000] {
        let needed = max_tokens as f64 / cfg.min_decode_toks_per_s;
        let p = derive_timeout_policy(&cfg, None, max_tokens, 0);
        assert!(
            p.total_s.expect("total") > needed,
            "max_tokens={max_tokens}"
        );
    }
}

/// Admission matrix: (config, prompt, max_tokens, max_seq_len) -> outcome.
#[test]
fn admission_matrix() {
    let off = AdmissionConfig::default();
    let on = AdmissionConfig { clamp: true, ..off };
    use OutputAdmission::*;
    let rows: &[(AdmissionConfig, usize, usize, usize, OutputAdmission)] = &[
        // fits exactly (inclusive bound), either mode
        (on, 16384, 16384, 32768, Fits),
        (off, 16384, 16384, 32768, Fits),
        // arm G repair: 17000 + 16384 > 32768 -> clamp to 32768-17000
        (
            on,
            17000,
            16384,
            32768,
            Clamped {
                from: 16384,
                to: 15768,
            },
        ),
        // one over: clamped to the largest admitted value, not one below it
        (
            on,
            16385,
            16384,
            32768,
            Clamped {
                from: 16384,
                to: 16383,
            },
        ),
        // prompt leaves 255 (< 256) -> structured refusal
        (on, 32513, 1024, 32768, TooLittleRoom { room: 255 }),
        // prompt leaves exactly 256 -> clamp to 256
        (
            on,
            32512,
            1024,
            32768,
            Clamped {
                from: 1024,
                to: 256,
            },
        ),
        // prompt at the context -> no room
        (on, 32768, 1, 32768, TooLittleRoom { room: 0 }),
        // clamp off (default): every over-long request is the pre-contract 400
        (off, 17000, 16384, 32768, Exceeds),
        (off, 32513, 1024, 32768, Exceeds),
        (off, 32768, 1, 32768, Exceeds),
    ];
    for (cfg, prompt, max_tokens, max_seq, want) in rows {
        assert_eq!(
            &admit_output(cfg, *prompt, *max_tokens, *max_seq),
            want,
            "prompt={prompt} max_tokens={max_tokens} max_seq_len={max_seq} clamp={}",
            cfg.clamp
        );
    }
}

/// Pre/post pin for the arm-G repair shape: default (pre-contract) behaviour
/// is the 400, opt-in clamp admits it at exactly the inclusive boundary.
#[test]
fn arm_g_shape_pre_and_post() {
    let pre = AdmissionConfig::default();
    assert!(!pre.clamp, "clamp must be opt-in (byte-identical defaults)");
    assert_eq!(
        admit_output(&pre, 17000, 16384, 32768),
        OutputAdmission::Exceeds
    );
    let post = AdmissionConfig { clamp: true, ..pre };
    let OutputAdmission::Clamped { to, .. } = admit_output(&post, 17000, 16384, 32768) else {
        panic!("expected clamp");
    };
    assert_eq!(to, 15768);
    assert_eq!(
        super::super::context_budget::admitted_total(17000, to, 32768),
        Some(32768)
    );
    assert_eq!(
        super::super::context_budget::admitted_total(17000, to + 1, 32768),
        None
    );
}

#[test]
fn refusal_message_names_prompt_seq_len_and_minimum() {
    let m = too_little_room_message(32513, 1024, 32768, 255, 256);
    for needle in [
        "prompt_tokens=32513",
        "max_seq_len=32768",
        "min_answer_tokens=256",
    ] {
        assert!(m.contains(needle), "{needle} missing from {m}");
    }
}

#[test]
fn stall_timeout_defaults_off() {
    assert_eq!(DEFAULT_STALL_TIMEOUT_S, 0);
}

#[test]
fn timeout_causes_produce_a_client_warning_and_others_do_not() {
    assert!(
        timeout_warning(Some(TIMEOUT_STALL_CAUSE))
            .is_some_and(|w| w.starts_with(TIMEOUT_STALL_CAUSE))
    );
    assert!(timeout_warning(Some(TIMEOUT_TOTAL_CAUSE)).is_some());
    assert_eq!(timeout_warning(Some("watchdog")), None);
    assert_eq!(timeout_warning(None), None);
}
