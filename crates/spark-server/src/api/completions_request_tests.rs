// SPDX-License-Identifier: AGPL-3.0-only

//! Execute the real completion handlers with an in-memory scheduler channel.

use super::*;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::sync::mpsc;

fn state() -> (Arc<AppState>, mpsc::Receiver<InferenceRequest>) {
    let directory = tempfile::tempdir().unwrap();
    let vocabulary = ["[UNK]", "hello", "answer", "##<sto", "##p>", "##extra"]
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_decoder(Some(tokenizers::decoders::wordpiece::WordPiece::new(
        "##".into(),
        false,
    )));
    tokenizer
        .save(directory.path().join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        directory.path().join("tokenizer_config.json"),
        r#"{"chat_template":"{{ messages }}"}"#,
    )
    .unwrap();
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        directory.path(),
        0,
        false,
        "completion-flow-fixture",
        None,
    )
    .unwrap();
    assert_eq!(
        tokenizer.decode(&[2, 3, 4, 5]).unwrap(),
        "answer<stop>extra"
    );
    let (request_tx, request_rx) = mpsc::channel(4);
    let state = Arc::new(AppState {
        tokenizer,
        model_name: "completion-flow-fixture".into(),
        max_seq_len: 128,
        max_batch_size: 1,
        serving: serde_json::Value::Null,
        yarn_context: false,
        request_tx,
        vision_config: None,
        dsv41: false,
        dsv41_vision: None,
        default_temperature: 0.0,
        default_top_k: 0,
        default_top_p: 1.0,
        default_top_n_sigma: 0.0,
        default_min_p: 0.0,
        tool_call_parser: None,
        reasoning_parser: None,
        think_end_token_id: None,
        think_start_token_id: None,
        tool_max_tokens: 16,
        default_max_tokens: None,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        auto_compact_threshold: None,
        model_ready: Arc::new(AtomicBool::new(true)),
        request_timeout: 0,
        effective_context: 0,
        behavior: Default::default(),
        disable_thinking: false,
        disable_simhash_watchdog: true,
        default_chat_template_kwargs: None,
        runtime: crate::runtime_settings::RuntimeSettings::for_tests(128),
        response_store: crate::response_store::ResponseStore::with_config(
            4,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 0,
                burst_tpm: 0,
            },
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            4,
            Duration::from_secs(60),
        ),
        dump_writer: None,
        auth: None,
    });
    (state, request_rx)
}

fn request(report: bool, stream: bool) -> CompletionRequest {
    serde_json::from_value(json!({
        "prompt_token_ids": [1],
        "max_tokens": 4,
        "temperature": 0.0,
        "stop": "<stop>",
        "stream": stream,
        "return_token_ids": !stream,
        "report_interventions": report,
    }))
    .unwrap()
}

fn record_scheduler_event(ledger: Option<&crate::openai::SharedInterventionLedger>, report: bool) {
    assert_eq!(ledger.is_some(), report);
    crate::openai::record_shared_intervention(ledger, "content_loop_watchdog", "stop");
}

fn assert_ledger(value: &Value, report: bool, stop_action: &str) {
    if !report {
        assert!(value.get("interventions").is_none(), "{value}");
        return;
    }
    assert_eq!(value["interventions"]["schema"], "atlas.interventions.v1");
    assert_eq!(value["interventions"]["policy"], "report_only");
    assert_eq!(
        value["interventions"]["events"],
        json!([
            {"code": "content_loop_watchdog", "action": "stop", "count": 1},
            {"code": "completion_stop_sequence", "action": stop_action, "count": 1},
        ])
    );
}

#[tokio::test]
async fn blocking_completion_request_preserves_tokens_and_shared_scheduler_ledger() {
    for report in [false, true] {
        let (state, mut request_rx) = state();
        let handler = completions(CurrentModel(state), Ok(Json(request(report, false))));
        let scheduler = async move {
            let InferenceRequest::Blocking {
                prompt_tokens,
                intervention_ledger,
                response_tx,
                ..
            } = request_rx.recv().await.unwrap()
            else {
                panic!("blocking handler must admit a blocking request");
            };
            assert_eq!(prompt_tokens, [1]);
            record_scheduler_event(intervention_ledger.as_ref(), report);
            assert!(
                response_tx
                    .send(Ok(InferenceResponse {
                        output_tokens: vec![2, 3, 4, 5],
                        thinking_output_boundary: None,
                        finish_reason: "stop".into(),
                        time_to_first_token_ms: 5.0,
                        decode_time_ms: 30.0,
                        logprobs: vec![],
                        reasoning_tokens: 0,
                        cached_prompt_tokens: 0,
                        think_nudges: 0,
                        stop_cause: None,
                    }))
                    .is_ok()
            );
        };
        let (response, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(handler, scheduler)
        })
        .await
        .expect("completion handler and scheduler must make progress");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["choices"][0]["text"], "answer");
        assert_eq!(value["choices"][0]["token_ids"], json!([2, 3, 4, 5]));
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["prompt_tokens"], 1);
        assert_eq!(value["usage"]["completion_tokens"], 4);
        assert_eq!(value["usage"]["total_tokens"], 5);
        assert_ledger(&value, report, "strip_output");
    }
}

#[tokio::test]
async fn streaming_completion_request_reports_cross_delta_stop_and_cancels_scheduler() {
    for report in [false, true] {
        let (state, mut request_rx) = state();
        let response = completions(CurrentModel(state), Ok(Json(request(report, true)))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let InferenceRequest::Streaming {
            prompt_tokens,
            intervention_ledger,
            token_tx,
            cancel_flag,
            ..
        } = request_rx.recv().await.unwrap()
        else {
            panic!("streaming handler must admit a streaming request");
        };
        assert_eq!(prompt_tokens, [1]);
        record_scheduler_event(intervention_ledger.as_ref(), report);
        for token in [2, 3, 4, 5] {
            assert!(token_tx.send(StreamEvent::Token(token)).await.is_ok());
        }
        assert!(
            token_tx
                .send(StreamEvent::Done {
                    finish_reason: "stop".into(),
                    prompt_tokens: 1,
                    completion_token_ids: vec![2, 3, 4, 5],
                    completion_tokens: 4,
                    time_to_first_token_ms: 5.0,
                    decode_time_ms: 30.0,
                    reasoning_tokens: 0,
                    cached_prompt_tokens: 0,
                    think_nudges: 0,
                    stop_cause: None,
                })
                .await
                .is_ok()
        );
        drop(token_tx);
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let wire = std::str::from_utf8(&bytes).unwrap();
        let data: Vec<_> = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(data.last().copied(), Some("[DONE]"));
        let chunks: Vec<Value> = data[..data.len() - 1]
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let text: String = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(text, "answer");
        assert!(cancel_flag.load(Ordering::Acquire));
        let done = chunks.last().unwrap();
        assert_eq!(done["choices"][0]["finish_reason"], "stop");
        assert_eq!(done["usage"]["completion_tokens"], 4);
        assert_eq!(done["usage"]["total_tokens"], 5);
        assert_ledger(done, report, "stop_stream");
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.get("interventions").is_none());
        }
    }
}

#[tokio::test]
async fn dropped_completion_request_body_cancels_its_actual_scheduler_request() {
    let (state, mut request_rx) = state();
    let response = completions(CurrentModel(state), Ok(Json(request(false, true)))).await;
    let InferenceRequest::Streaming { cancel_flag, .. } = request_rx.recv().await.unwrap() else {
        panic!("expected streaming scheduler request");
    };
    assert!(!cancel_flag.load(Ordering::Acquire));
    drop(response);
    assert!(cancel_flag.load(Ordering::Acquire));
}

#[tokio::test]
async fn model_endpoints_report_the_served_max_model_len() {
    let (state, _request_rx) = state();
    let Json(list) = list_models(CurrentModel(state.clone())).await;
    let listed = serde_json::to_value(&list).unwrap();
    assert_eq!(listed["data"][0]["id"], "completion-flow-fixture");
    assert_eq!(listed["data"][0]["max_model_len"], 128);
    let response = get_model(
        CurrentModel(state),
        axum::extract::Path("completion-flow-fixture".to_string()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let single: Value = serde_json::from_slice(&body).unwrap();
    for key in ["id", "object", "owned_by", "max_model_len"] {
        assert_eq!(single[key], listed["data"][0][key], "{key}");
    }
}

/// Runtime defaults (`/v1/flux/settings`) reach a request that omits the
/// field; a request that names the field keeps its own value.
#[tokio::test]
async fn runtime_defaults_apply_to_silent_requests_and_request_fields_win() {
    for (body, want_max, want_rep) in [
        (
            json!({"prompt_token_ids": [1], "temperature": 0.0}),
            7usize,
            1.3f32,
        ),
        (
            json!({"prompt_token_ids": [1], "temperature": 0.0, "max_tokens": 3,
                   "repetition_penalty": 1.0}),
            3,
            1.0,
        ),
    ] {
        let (state, mut request_rx) = state();
        state
            .runtime
            .apply_patch(&json!({
                "max_tokens.default": 7,
                "sampling.non_thinking.repetition_penalty": 1.3,
            }))
            .unwrap();
        let req: CompletionRequest = serde_json::from_value(body).unwrap();
        let handler = completions(CurrentModel(state), Ok(Json(req)));
        let scheduler = async move {
            let InferenceRequest::Blocking {
                max_tokens,
                repetition_penalty,
                response_tx,
                ..
            } = request_rx.recv().await.unwrap()
            else {
                panic!("blocking handler must admit a blocking request");
            };
            assert_eq!(max_tokens, want_max);
            assert_eq!(repetition_penalty, want_rep);
            drop(response_tx);
        };
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(handler, scheduler)
        })
        .await
        .expect("completion handler and scheduler must make progress");
    }
}

fn state_with_timeout_and_context(
    request_timeout: u32,
    max_seq_len: usize,
    dsv41: bool,
) -> (Arc<AppState>, mpsc::Receiver<InferenceRequest>) {
    let (base_arc, _rx) = state();
    let Ok(base) = Arc::try_unwrap(base_arc) else {
        panic!("fresh test AppState must have unique ownership");
    };
    let (request_tx, request_rx) = mpsc::channel(4);
    let custom = Arc::new(AppState {
        tokenizer: base.tokenizer,
        model_name: base.model_name,
        max_seq_len,
        max_batch_size: base.max_batch_size,
        yarn_context: base.yarn_context,
        request_tx,
        vision_config: None,
        dsv41,
        dsv41_vision: None,
        default_temperature: base.default_temperature,
        default_top_k: base.default_top_k,
        default_top_p: base.default_top_p,
        default_top_n_sigma: base.default_top_n_sigma,
        default_min_p: base.default_min_p,
        tool_call_parser: None,
        reasoning_parser: None,
        think_end_token_id: None,
        think_start_token_id: None,
        tool_max_tokens: base.tool_max_tokens,
        default_max_tokens: None,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        auto_compact_threshold: None,
        model_ready: Arc::new(AtomicBool::new(true)),
        request_timeout,
        effective_context: 0,
        behavior: Default::default(),
        disable_thinking: false,
        disable_simhash_watchdog: true,
        default_chat_template_kwargs: None,
        runtime: crate::runtime_settings::RuntimeSettings::for_tests(max_seq_len),
        response_store: crate::response_store::ResponseStore::with_config(
            4,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 0,
                burst_tpm: 0,
            },
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            4,
            Duration::from_secs(60),
        ),
        dump_writer: None,
        auth: None,
        serving: base.serving,
    });
    (custom, request_rx)
}

#[tokio::test]
async fn blocking_and_streaming_completions_honor_per_request_timeout_and_zero_disable() {
    for stream in [false, true] {
        // Explicit timeout overrides server base (even below derived).
        let (st, mut rx) = state_with_timeout_and_context(300, 32_768, false);
        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt_token_ids": [1],
            "max_tokens": 16_384,
            "stream": stream,
            "timeout": 42.5
        }))
        .unwrap();
        let before = std::time::Instant::now();
        if stream {
            let response = completions(CurrentModel(st), Ok(Json(req))).await;
            assert_eq!(response.status(), StatusCode::OK);
            let InferenceRequest::Streaming {
                timeout_at,
                decode_timeout_at,
                ..
            } = rx.recv().await.unwrap()
            else {
                panic!("expected streaming request");
            };
            let span = timeout_at
                .expect("streaming request must receive timeout_at")
                .duration_since(before);
            assert!(
                (42.0..=45.0).contains(&span.as_secs_f64()),
                "unexpected streaming timeout span: {span:?}"
            );
            assert_eq!(
                decode_timeout_at, None,
                "explicit timeout wins over derived decode deadline"
            );
        } else {
            let handler = completions(CurrentModel(st), Ok(Json(req)));
            let scheduler = async move {
                let InferenceRequest::Blocking {
                    timeout_at,
                    decode_timeout_at,
                    response_tx,
                    ..
                } = rx.recv().await.unwrap()
                else {
                    panic!("expected blocking request");
                };
                let span = timeout_at
                    .expect("blocking request must receive timeout_at")
                    .duration_since(before);
                assert!(
                    (42.0..=45.0).contains(&span.as_secs_f64()),
                    "unexpected blocking timeout span: {span:?}"
                );
                assert_eq!(decode_timeout_at, None);
                drop(response_tx);
            };
            let _ = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(handler, scheduler)
            })
            .await
            .unwrap();
        }

        // Explicit timeout=0 disables server base deadline on both paths.
        let (st_zero, mut rx_zero) = state_with_timeout_and_context(300, 32_768, false);
        let req_zero: CompletionRequest = serde_json::from_value(json!({
            "prompt_token_ids": [1],
            "max_tokens": 16_384,
            "stream": stream,
            "timeout": 0
        }))
        .unwrap();
        if stream {
            let response = completions(CurrentModel(st_zero), Ok(Json(req_zero))).await;
            assert_eq!(response.status(), StatusCode::OK);
            let InferenceRequest::Streaming {
                timeout_at,
                decode_timeout_at,
                ..
            } = rx_zero.recv().await.unwrap()
            else {
                panic!("expected streaming request");
            };
            assert_eq!(timeout_at, None);
            assert_eq!(decode_timeout_at, None);
        } else {
            let handler = completions(CurrentModel(st_zero), Ok(Json(req_zero)));
            let scheduler = async move {
                let InferenceRequest::Blocking {
                    timeout_at,
                    decode_timeout_at,
                    response_tx,
                    ..
                } = rx_zero.recv().await.unwrap()
                else {
                    panic!("expected blocking request");
                };
                assert_eq!(timeout_at, None);
                assert_eq!(decode_timeout_at, None);
                drop(response_tx);
            };
            let _ = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(handler, scheduler)
            })
            .await
            .unwrap();
        }
    }
}

#[tokio::test]
async fn blocking_and_streaming_completions_derive_decode_timeout_from_server_base() {
    // Base = 300s, max_tokens = 16384, prompt_len = 1 -> derived = 2048 + 1 + 30 = 2079s > 300s.
    for stream in [false, true] {
        let (st, mut rx) = state_with_timeout_and_context(300, 32_768, false);
        let req: CompletionRequest = serde_json::from_value(json!({
            "prompt_token_ids": [1],
            "max_tokens": 16_384,
            "stream": stream
        }))
        .unwrap();
        let before = std::time::Instant::now();
        if stream {
            let response = completions(CurrentModel(st), Ok(Json(req))).await;
            assert_eq!(response.status(), StatusCode::OK);
            let InferenceRequest::Streaming {
                timeout_at,
                decode_timeout_at,
                ..
            } = rx.recv().await.unwrap()
            else {
                panic!("expected streaming request");
            };
            let prefill_s = timeout_at
                .expect("streaming server base timeout_at")
                .duration_since(before)
                .as_secs_f64();
            let decode_s = decode_timeout_at
                .expect("streaming derived decode_timeout_at")
                .duration_since(before)
                .as_secs_f64();
            assert!(
                (299.0..=305.0).contains(&prefill_s),
                "prefill_s={prefill_s}"
            );
            assert!((2078.0..=2085.0).contains(&decode_s), "decode_s={decode_s}");
        } else {
            let handler = completions(CurrentModel(st), Ok(Json(req)));
            let scheduler = async move {
                let InferenceRequest::Blocking {
                    timeout_at,
                    decode_timeout_at,
                    response_tx,
                    ..
                } = rx.recv().await.unwrap()
                else {
                    panic!("expected blocking request");
                };
                let prefill_s = timeout_at
                    .expect("blocking server base timeout_at")
                    .duration_since(before)
                    .as_secs_f64();
                let decode_s = decode_timeout_at
                    .expect("blocking derived decode_timeout_at")
                    .duration_since(before)
                    .as_secs_f64();
                assert!(
                    (299.0..=305.0).contains(&prefill_s),
                    "prefill_s={prefill_s}"
                );
                assert!((2078.0..=2085.0).contains(&decode_s), "decode_s={decode_s}");
                drop(response_tx);
            };
            let _ = tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(handler, scheduler)
            })
            .await
            .unwrap();
        }
    }
}

#[tokio::test]
async fn invalid_completion_timeouts_fail_with_bad_request_and_dsv41_respects_no_base_deadline() {
    for invalid in [
        -1.0f32,
        -0.001,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
    ] {
        for stream in [false, true] {
            let (st, mut rx) = state_with_timeout_and_context(300, 128, false);
            let mut req = request(false, stream);
            req.timeout = Some(invalid);
            let response = completions(CurrentModel(st), Ok(Json(req))).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "invalid={invalid} stream={stream}"
            );
            let body = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let parsed: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(parsed["error"]["param"], "timeout");
            assert_eq!(parsed["error"]["code"], "invalid_timeout");
            assert!(
                rx.try_recv().is_err(),
                "invalid request must not reach scheduler"
            );
        }
    }

    // deepseek_v41: no server-base deadline when omitted, but explicit request timeout applies.
    let (st_dsv, mut rx_dsv) = state_with_timeout_and_context(300, 32_768, true);
    let omitted: CompletionRequest = serde_json::from_value(json!({
        "prompt_token_ids": [1],
        "max_tokens": 16_384,
        "stream": true
    }))
    .unwrap();
    let resp = completions(CurrentModel(st_dsv), Ok(Json(omitted))).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let InferenceRequest::Streaming {
        timeout_at,
        decode_timeout_at,
        ..
    } = rx_dsv.recv().await.unwrap()
    else {
        panic!("expected streaming request");
    };
    assert_eq!(timeout_at, None);
    assert_eq!(decode_timeout_at, None);
}
