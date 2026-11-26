// SPDX-License-Identifier: AGPL-3.0-only

use super::ChatCompletionRequest;

fn request_with(stream_options: Option<serde_json::Value>) -> ChatCompletionRequest {
    let mut body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    if let Some(options) = stream_options {
        body["stream_options"] = options;
    }
    serde_json::from_value(body).expect("valid streaming request")
}

#[test]
fn continuous_usage_is_absent_and_false_by_default() {
    let absent = request_with(None);
    assert!(absent.stream_options.is_none());

    let include_only = request_with(Some(serde_json::json!({"include_usage": true})));
    let options = include_only.stream_options.expect("stream options");
    assert!(options.include_usage);
    assert!(!options.continuous_usage_stats);

    let explicit_false = request_with(Some(serde_json::json!({
        "include_usage": true,
        "continuous_usage_stats": false,
    })));
    assert!(
        !explicit_false
            .stream_options
            .expect("stream options")
            .continuous_usage_stats
    );
}

#[test]
fn continuous_usage_opt_in_parses() {
    let request = request_with(Some(serde_json::json!({
        "include_usage": true,
        "continuous_usage_stats": true,
    })));
    let options = request.stream_options.expect("stream options");
    assert!(options.include_usage);
    assert!(options.continuous_usage_stats);
}
