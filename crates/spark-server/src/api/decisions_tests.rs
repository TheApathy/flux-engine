// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use super::{
    first_tokens_are_disjoint, handle_eider_questions, handle_flare_questions, logprob_token_ids,
    score_boolean_predicate, score_candidate_groups, shannon_concentration, softmax,
};
use crate::tokenizer::ChatTokenizer;

fn fixture_tokenizer() -> ChatTokenizer {
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    let entries = [
        ("<unk>".to_string(), 0u32),
        ("true".to_string(), 1),
        ("false".to_string(), 2),
        ("yes".to_string(), 3),
        ("no".to_string(), 4),
        ("approve".to_string(), 5),
        ("reject".to_string(), 6),
        ("escalate".to_string(), 7),
        ("0".to_string(), 8),
        ("1".to_string(), 9),
        ("2".to_string(), 10),
    ];
    let wl = WordLevel::builder()
        .vocab(entries.into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .expect("wordlevel vocab");
    let mut tok = Tokenizer::new(wl);
    tok.with_pre_tokenizer(Some(WhitespaceSplit));

    let dir = tempfile::tempdir().expect("tempdir");
    tok.save(dir.path().join("tokenizer.json"), false)
        .expect("save tokenizer.json");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}assistant: "}"#,
    )
    .expect("save tokenizer_config.json");

    ChatTokenizer::from_model_dir(dir.path(), 0, false, "fixture", None)
        .expect("load ChatTokenizer")
}

#[test]
fn logprob_token_ids_resolves_choices_and_rejects_empty_inputs() {
    let tok = fixture_tokenizer();
    assert!(logprob_token_ids(&tok, &[]).is_err());
    assert!(logprob_token_ids(&tok, &[String::new()]).is_err());

    let choices = vec![
        "approve".to_string(),
        "reject".to_string(),
        "escalate".to_string(),
    ];
    let resolved = logprob_token_ids(&tok, &choices).expect("resolve choices");
    assert_eq!(resolved.len(), 3);
    assert_eq!(resolved[0].token_ids, vec![5]);
    assert_eq!(resolved[1].token_ids, vec![6]);
    assert_eq!(resolved[2].token_ids, vec![7]);
    assert!(first_tokens_are_disjoint(&resolved));
}

#[test]
fn candidate_scoring_computes_normalized_softmax_confidence_and_concentration() {
    let mut lps = BTreeMap::new();
    lps.insert(5u32, -0.2); // approve (~0.8187 vocab prob)
    lps.insert(6u32, -2.2); // reject
    lps.insert(7u32, -4.2); // escalate

    let groups = vec![vec![5], vec![6], vec![7]];
    let scored = score_candidate_groups(&lps, -15.0, &groups);
    assert_eq!(scored.best_index, 0);
    let sum: f64 = scored.probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    assert!(scored.probabilities[0] > 0.85);
    assert!((scored.confidence - (-0.2f64).exp()).abs() < 1e-9);
    assert!(scored.concentration > 0.5 && scored.concentration <= 1.0);

    // Uniform distribution has zero concentration.
    let uniform = softmax(&[-1.0, -1.0, -1.0]);
    assert!(shannon_concentration(&uniform) < 1e-9);
}

#[test]
fn boolean_predicate_scoring_separates_true_and_false_tokens() {
    let tok = fixture_tokenizer();
    let mut lps = BTreeMap::new();
    lps.insert(1u32, -0.1); // "true"
    lps.insert(2u32, -3.5); // "false"

    let p_true = score_boolean_predicate(&tok, &lps, -20.0);
    assert!(p_true > 0.95, "expected high p_true, got {p_true}");

    lps.insert(1u32, -4.0);
    lps.insert(2u32, -0.05);
    let p_false_case = score_boolean_predicate(&tok, &lps, -20.0);
    assert!(
        p_false_case < 0.05,
        "expected low p_true, got {p_false_case}"
    );
}

#[test]
fn flare_and_eider_decision_contracts_produce_exact_disjoint_wire_shapes() {
    let tok = fixture_tokenizer();
    let mut lps = BTreeMap::new();
    lps.insert(1u32, -0.15); // true
    lps.insert(2u32, -3.0); // false
    lps.insert(5u32, -0.3); // approve
    lps.insert(6u32, -2.5); // reject
    lps.insert(7u32, -4.0); // escalate
    lps.insert(8u32, -4.0); // 0
    lps.insert(9u32, -2.0); // 1
    lps.insert(10u32, -0.2); // 2
    let offline = (lps, -20.0);

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");

    // 1. Flare mode: `questions` is a JSON Array.
    let flare_questions = serde_json::json!([
        {"type": "predicate", "name": "is_safe", "question": "Is the patch safe?"},
        {
            "type": "choice",
            "name": "action",
            "choices": [
                {"value": "approve", "description": "Ship immediately"},
                {"value": "reject", "description": "Block PR"},
                {"value": "escalate"}
            ]
        },
        {
            "type": "score",
            "name": "quality",
            "levels": [{"label": "0"}, {"label": "1"}, {"label": "2"}]
        }
    ]);
    let flare_out = rt
        .block_on(handle_flare_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            flare_questions.as_array().expect("array"),
            Some(&offline),
        ))
        .expect("flare evaluation");
    let decisions = flare_out["decisions"].as_array().expect("flare array");
    assert_eq!(decisions.len(), 3);
    assert_eq!(decisions[0]["type"], "predicate");
    assert_eq!(decisions[0]["name"], "is_safe");
    assert!(decisions[0]["probability"].as_f64().expect("prob") > 0.9);
    assert_eq!(decisions[1]["type"], "choice");
    assert_eq!(decisions[1]["choice"], "approve");
    assert!(decisions[1]["confidence"].as_f64().expect("conf") > 0.7);
    assert_eq!(decisions[2]["type"], "score");
    assert!(decisions[2]["score"].as_f64().expect("score") > 1.7);

    // 2. Eider / Decisive Dory mode: `questions` is a JSON Object/Map.
    let eider_questions = serde_json::json!({
        "safe": {"type": "noul"},
        "route": {
            "type": "choice",
            "criteria": {
                "approve": "Ship immediately",
                "reject": "Block PR",
                "escalate": "Ask human"
            }
        },
        "severity": {
            "type": "score",
            "criteria": ["0", "1", "2"]
        }
    });
    let eider_out = rt
        .block_on(handle_eider_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            eider_questions.as_object().expect("map"),
            Some(&offline),
        ))
        .expect("eider evaluation");
    let ans = &eider_out["decisions"];
    assert_eq!(ans["safe"]["type"], "noul");
    assert!(ans["safe"]["noul"].as_f64().expect("noul") > 0.9);
    assert_eq!(ans["route"]["type"], "choice");
    assert_eq!(ans["route"]["choice"], "approve");
    assert!(
        ans["route"]["probabilities"]["approve"]
            .as_f64()
            .expect("p")
            > 0.8
    );
    assert!(ans["route"]["concentration"].as_f64().expect("conc") > 0.4);
    assert_eq!(ans["severity"]["type"], "score");
    assert!(ans["severity"]["score"].as_f64().expect("score") > 1.7);
    assert_eq!(ans["severity"]["legend"]["2"], "2");
}

#[test]
fn flare_and_eider_enforce_64_question_bound() {
    use axum::http::StatusCode;
    use serde_json::{Map, Value};

    let tok = fixture_tokenizer();
    let mut lps = BTreeMap::new();
    lps.insert(1u32, -0.1);
    lps.insert(2u32, -3.0);
    let offline = (lps, -20.0);

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");

    let flare_64: Vec<Value> = (0..64)
        .map(|i| serde_json::json!({"type": "predicate", "name": format!("q{i}")}))
        .collect();
    assert!(
        rt.block_on(handle_flare_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            &flare_64,
            Some(&offline),
        ))
        .is_ok(),
        "64 Flare questions must be accepted"
    );

    let mut flare_65 = flare_64;
    flare_65.push(serde_json::json!({"type": "predicate", "name": "q64"}));
    let err = rt
        .block_on(handle_flare_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            &flare_65,
            Some(&offline),
        ))
        .expect_err("65 Flare questions must be rejected");
    assert_eq!(err.status, StatusCode::BAD_REQUEST);
    assert!(err.message.contains("64"));

    let mut eider_64 = Map::new();
    for i in 0..64 {
        eider_64.insert(format!("q{i}"), serde_json::json!({"type": "noul"}));
    }
    assert!(
        rt.block_on(handle_eider_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            &eider_64,
            Some(&offline),
        ))
        .is_ok(),
        "64 Eider questions must be accepted"
    );

    let mut eider_65 = eider_64;
    eider_65.insert("q64".to_string(), serde_json::json!({"type": "noul"}));
    let err = rt
        .block_on(handle_eider_questions(
            None,
            &tok,
            "fixture-model",
            &[],
            &eider_65,
            Some(&offline),
        ))
        .expect_err("65 Eider questions must be rejected");
    assert_eq!(err.status, StatusCode::BAD_REQUEST);
    assert!(err.message.contains("64"));
}

#[test]
fn decisions_uses_bounded_scheduler_admission_and_preserves_retry_after() {
    use axum::http::{StatusCode, header};

    let source = include_str!("decisions.rs");
    assert!(source.contains("try_admit_scheduler(&state.request_tx, infer_req)"));
    assert!(!source.contains("request_tx\n        .send("));
    assert!(!source.contains("request_tx.send("));

    // Full queue -> 429 + Retry-After: 1
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    crate::api::inference_impl::try_admit_scheduler(&tx, 1u8).expect("first slot");
    let (status, msg) = crate::api::inference_impl::try_admit_scheduler(&tx, 2u8).unwrap_err();
    let err = super::DecisionError::from_status(status, msg);
    let resp = crate::api::compact::openai_error_response(err.status, err.message);
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        resp.headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("1")
    );

    // Closed queue -> 503 + Retry-After: 5
    let (tx_closed, rx_closed) = tokio::sync::mpsc::channel(1);
    drop(rx_closed);
    let (status, msg) =
        crate::api::inference_impl::try_admit_scheduler(&tx_closed, 1u8).unwrap_err();
    let err = super::DecisionError::from_status(status, msg);
    let resp = crate::api::compact::openai_error_response(err.status, err.message);
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("5")
    );
}
