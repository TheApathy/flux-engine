// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    image_content::{content_value, text_part, tool_result_value},
    types::ContentBlock,
};
use serde_json::{Value, json};

pub(super) fn translate_blocks(role: &str, blocks: &[ContentBlock]) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut thinking_parts: Vec<&str> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } => parts.push(text_part(text)),
            ContentBlock::Image { source } => parts.push(source.chat_part()),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                "id":id,"type":"function",
                "function":{"name":name,"arguments":input.to_string()}
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                // Flush preceding user content before the result rather than
                // moving all user text (and screenshots) ahead of every tool.
                if role != "assistant" {
                    if !parts.is_empty() {
                        messages.push(json!({"role":role,
                            "content":content_value(std::mem::take(&mut parts), "")}));
                    }
                    messages.push(json!({"role":"tool", "tool_call_id":tool_use_id,
                        "content":tool_result_value(content.as_ref(), is_error.unwrap_or(false))}));
                }
            }
            ContentBlock::Thinking {
                thinking: Some(text),
            } if role == "assistant" && !text.is_empty() => {
                thinking_parts.push(text.as_str());
            }
            ContentBlock::Thinking { .. } | ContentBlock::Unknown => {}
        }
    }
    if role == "assistant" || !parts.is_empty() || messages.is_empty() {
        let mut message = json!({"role":role, "content":content_value(parts, "")});
        if role == "assistant" && !thinking_parts.is_empty() {
            message["reasoning_content"] = Value::String(thinking_parts.join("\n"));
        }
        if role == "assistant" && !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        messages.push(message);
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_thinking_blocks_replay_as_reasoning_content() {
        let blocks = [
            ContentBlock::Thinking {
                thinking: Some("Step 1: check inputs.".to_string()),
            },
            ContentBlock::Thinking {
                thinking: Some("Step 2: call tool.".to_string()),
            },
            ContentBlock::Text {
                text: "Running lookup.".to_string(),
            },
            ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "lookup".to_string(),
                input: json!({"q": "rust"}),
            },
        ];

        let lowered = translate_blocks("assistant", &blocks);
        assert_eq!(lowered.len(), 1);
        assert_eq!(
            lowered[0]["reasoning_content"],
            "Step 1: check inputs.\nStep 2: call tool."
        );
        assert_eq!(lowered[0]["content"], "Running lookup.");
        assert_eq!(lowered[0]["tool_calls"][0]["id"], "toolu_1");

        let incoming: crate::openai::IncomingMessage =
            serde_json::from_value(lowered[0].clone()).expect("valid IncomingMessage");
        assert_eq!(
            incoming.reasoning_content.as_deref(),
            Some("Step 1: check inputs.\nStep 2: call tool.")
        );
        assert_eq!(incoming.content.text, "Running lookup.");
        assert_eq!(incoming.tool_calls.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn assistant_thinking_only_message_preserves_reasoning_content() {
        let blocks = [ContentBlock::Thinking {
            thinking: Some("Silent reasoning.".to_string()),
        }];
        let lowered = translate_blocks("assistant", &blocks);
        assert_eq!(lowered.len(), 1);
        assert_eq!(lowered[0]["role"], "assistant");
        assert_eq!(lowered[0]["content"], "");
        assert_eq!(lowered[0]["reasoning_content"], "Silent reasoning.");
    }

    #[test]
    fn empty_or_user_thinking_blocks_do_not_set_reasoning_content() {
        let empty_assistant = translate_blocks(
            "assistant",
            &[
                ContentBlock::Thinking {
                    thinking: Some(String::new()),
                },
                ContentBlock::Thinking { thinking: None },
                ContentBlock::Text {
                    text: "Done.".to_string(),
                },
            ],
        );
        assert!(empty_assistant[0].get("reasoning_content").is_none());

        let user_thinking = translate_blocks(
            "user",
            &[
                ContentBlock::Thinking {
                    thinking: Some("should not leak onto user turn".to_string()),
                },
                ContentBlock::Text {
                    text: "Hello".to_string(),
                },
            ],
        );
        assert!(user_thinking[0].get("reasoning_content").is_none());
    }
}
