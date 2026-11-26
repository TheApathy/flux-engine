// SPDX-License-Identifier: AGPL-3.0-only

use super::super::translate::anthropic_to_chat_request_json;
use super::super::types::MessagesRequest;

#[test]
fn anthropic_output_config_lowers_to_response_format() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": false
    });
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3.8",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Return ok"}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }
    }))
    .unwrap();
    let chat = anthropic_to_chat_request_json(&req);
    assert_eq!(chat["response_format"]["type"], "json_schema");
    assert_eq!(
        chat["response_format"]["json_schema"]["strict"],
        serde_json::json!(true)
    );
    assert_eq!(
        chat["response_format"]["json_schema"]["schema"]["required"],
        serde_json::json!(["ok"])
    );

    // Legacy `output_format` alias also lowers cleanly.
    let req_legacy: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "qwen3.8",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Return JSON"}],
        "output_format": {"type": "json_object"}
    }))
    .unwrap();
    let chat_legacy = anthropic_to_chat_request_json(&req_legacy);
    assert_eq!(chat_legacy["response_format"]["type"], "json_object");
}
