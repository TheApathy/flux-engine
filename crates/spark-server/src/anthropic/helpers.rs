// SPDX-License-Identifier: AGPL-3.0-only

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

pub(super) fn anthropic_error(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

// ── Conversion helpers ──

/// Convert an OpenAI `finish_reason` to an Anthropic `stop_reason` string.
pub(super) fn convert_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}
