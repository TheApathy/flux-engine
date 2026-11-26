// SPDX-License-Identifier: AGPL-3.0-only

//! Tokenizer wrapper using HuggingFace tokenizers + minijinja chat template.
//!
//! Loads the model's official Jinja template from `tokenizer_config.json` and
//! renders it with minijinja for byte-exact alignment with the model's training
//! format. No fallback — if there's no Jinja template, the model is misconfigured.

use anyhow::Result;
use tokenizers::Tokenizer;

/// F76 (2026-04-29): pre-parse `tool_calls[*].function.arguments` from
/// OpenAI's wire format (JSON-encoded string) into the JSON value the
/// model's chat template expects. MiniMax M2.7's template iterates
/// `tool_call.function.arguments.items()` which crashes on a string.
/// We rebuild the message list with parsed arguments where present,
/// leaving every other field untouched. Returns a fresh Vec rather
/// than mutating the caller's slice.
fn normalize_tool_call_arguments(
    messages: &[serde_json::Value],
) -> std::borrow::Cow<'_, [serde_json::Value]> {
    let has_string_tool_args = messages.iter().any(|m| {
        m.get("tool_calls")
            .and_then(|v| v.as_array())
            .is_some_and(|tcs| {
                tcs.iter().any(|tc| {
                    tc.get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .is_some()
                })
            })
    });
    if !has_string_tool_args {
        return std::borrow::Cow::Borrowed(messages);
    }
    let mut total_parsed = 0usize;
    let mut total_seen = 0usize;
    let out: Vec<_> = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
                return msg;
            };
            for tc in tool_calls.iter_mut() {
                let Some(function) = tc.get_mut("function") else {
                    continue;
                };
                let Some(args) = function.get_mut("arguments") else {
                    continue;
                };
                total_seen += 1;
                let parsed_owned = if let Some(s) = args.as_str() {
                    serde_json::from_str::<serde_json::Value>(s).ok()
                } else {
                    None
                };
                if let Some(parsed) = parsed_owned {
                    *args = parsed;
                    total_parsed += 1;
                }
                // If parse fails or args wasn't a string, leave as-is —
                // template may handle via tojson, or surface the
                // original error for the operator.
            }
            msg
        })
        .collect();
    if total_seen > 0 {
        tracing::debug!(
            "F76 normalize: {}/{} tool_call arguments parsed string→dict",
            total_parsed,
            total_seen,
        );
    }
    std::borrow::Cow::Owned(out)
}

/// Wraps a HuggingFace tokenizer with Jinja chat template support.
mod chat_impl;
mod ids_only;
mod jinja_helpers;
mod large_encode;

/// The production environment for a `jinja-templates/` override, for tests
/// that render a template directly: the same Python-to-minijinja rewrite
/// (`.startswith` and friends) and the same method bridge `ChatTokenizer` uses.
// Used by the bin's `api::chat` tests; the lib test target compiles this
// module without them.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn override_template_env_for_tests(
    template: &str,
) -> anyhow::Result<minijinja::Environment<'static>> {
    jinja_helpers::build_jinja_env(&jinja_helpers::convert_python_jinja_to_minijinja(template))
}
pub(crate) mod special_literals;
#[cfg(test)]
mod special_literals_checkpoint_tests;
#[cfg(test)]
mod special_literals_tests;

#[cfg(test)]
mod glm_tool_contract_template_tests;

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    vocab_size: usize,
    eos_token_id: u32,
    supports_thinking: bool,
    /// Compiled Jinja chat template (from tokenizer_config.json).
    #[allow(dead_code)]
    chat_template: String,
    /// Precompiled minijinja environment (avoids re-creating + re-compiling each call).
    jinja_env: minijinja::Environment<'static>,
    /// OpenAI-variant template: gates historical `<think>` wrappers on enable_thinking.
    /// Falls back to jinja_env if no openai/ variant exists.
    openai_jinja_env: Option<minijinja::Environment<'static>>,
    /// The checkpoint template consumes Qwen3.8's three-level
    /// `reasoning_effort` vocabulary (`xhigh | medium | low`).  This is
    /// detected from the template contract rather than from `model_type`:
    /// Qwen3.5/3.6/3.8 dense checkpoints all report `qwen3_5`.
    qwen38_reasoning_effort: bool,
    /// True only for qwen4_exp checkpoints whose own Jinja template renders
    /// the native Qwen XML tool schema/instructions.  The API uses this to
    /// avoid adding a second full parser prompt while leaving every older
    /// qwen3_coder/Hermes prompt path unchanged.
    native_qwen_xml_tools: bool,
    /// `bos_token` from tokenizer_config.json, passed to the chat template.
    bos_token: String,
    /// `eos_token` from tokenizer_config.json (or decoded `eos_token_id`), passed to the chat template.
    eos_token: String,
    /// Keeps special-token text in request strings literal when rendering.
    special_literals: special_literals::SpecialLiterals,
    /// `--allow-special-tokens-in-content`: skip that escaping.
    special_tokens_in_content: bool,
    /// Default-on fast IDs-only encoding (`encode_fast(text, false)`), skipping
    /// unused per-token string and char-offset allocations.
    ids_only: bool,
    /// Concurrency gate bounding simultaneous `>= 1 MiB` encodes.
    large_encode_gate: std::sync::Arc<large_encode::Gate>,
}

/// Wrapper around tokenizers::DecodeStream that hides the generic parameters.
/// O(1) per step vs O(n) for full re-decode.
pub struct StreamingDecoder<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

impl StreamingDecoder<'_> {
    /// Feed one token. Returns Some(text) when valid UTF-8 is ready.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow::anyhow!("Streaming decode error: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spacing_tokenizer() -> ChatTokenizer {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let vocab = [
            ("ACK".to_string(), 0),
            ("Ġ".to_string(), 1),
            ("7741".to_string(), 2),
            ("-".to_string(), 3),
            ("C".to_string(), 4),
            ("Ġand".to_string(), 5),
            ("7".to_string(), 6),
            ("Ġsections".to_string(), 7),
            ("Ċ".to_string(), 8),
            // ByteLevel's reversible byte alphabet for UTF-8 E2 82 AC ("€").
            ("â".to_string(), 9),
            ("Ĥ".to_string(), 10),
            ("¬".to_string(), 11),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        let jinja_env = jinja_helpers::build_jinja_env("{{ messages }}").unwrap();
        ChatTokenizer {
            vocab_size: tokenizer.get_vocab_size(true),
            tokenizer,
            eos_token_id: 999,
            supports_thinking: false,
            chat_template: "{{ messages }}".into(),
            jinja_env,
            openai_jinja_env: None,
            qwen38_reasoning_effort: false,
            native_qwen_xml_tools: false,
            bos_token: String::new(),
            eos_token: String::new(),
            special_literals: special_literals::SpecialLiterals::from_tokenizer(&Tokenizer::new(
                BPE::default(),
            )),
            special_tokens_in_content: false,
            ids_only: ids_only::DEFAULT_IDS_ONLY,
            large_encode_gate: large_encode::Gate::default_gate(),
        }
    }

    #[test]
    fn incremental_full_decode_preserves_spacing_boundaries() {
        let tokenizer = spacing_tokenizer();
        // Standalone Ġ and Ċ tokens are the boundary shape that exposed the
        // tool-eval corruption: dropping whitespace-only deltas glues words,
        // numbers, and whole lines together.
        let ids = [0, 1, 2, 3, 4, 5, 1, 6, 8, 8, 7];
        let full = tokenizer.decode(&ids).unwrap();
        assert_eq!(full, "ACK 7741-C and 7\n\n sections");

        let mut prefix_offset = 0;
        let mut read_offset = 0;
        let mut streamed = String::new();
        let mut deltas = Vec::new();
        for end in 1..=ids.len() {
            let delta = tokenizer
                .incremental_decode(&ids[..end], &mut prefix_offset, &mut read_offset)
                .unwrap();
            streamed.push_str(&delta);
            deltas.push(delta);
        }

        assert_eq!(streamed, full);
        assert!(deltas.iter().any(|delta| delta == " "));
        assert_eq!(deltas.iter().filter(|delta| *delta == "\n").count(), 2);
    }

    #[test]
    fn decode_stream_comparison_fixture_matches_full_decode() {
        let tokenizer = spacing_tokenizer();
        let ids = [0, 1, 2, 5, 1, 6, 8, 7];
        let full = tokenizer.decode(&ids).unwrap();
        let mut legacy = tokenizer.streaming_decoder(true);
        let via_decode_stream: String = ids
            .iter()
            .filter_map(|&id| legacy.step(id).unwrap())
            .collect();

        // This synthetic control proves the fixture itself is well-formed.
        // Production nevertheless uses incremental_decode above, whose source
        // of truth is the same full-decode API as the blocking response path.
        assert_eq!(via_decode_stream, full);
    }

    #[test]
    fn incremental_decode_waits_for_and_then_emits_multitoken_utf8() {
        let tokenizer = spacing_tokenizer();
        let ids = [9, 10, 11, 1, 0];
        assert_eq!(tokenizer.decode(&ids).unwrap(), "€ ACK");

        let mut prefix_offset = 0;
        let mut read_offset = 0;
        let mut streamed = String::new();
        let mut deltas = Vec::new();
        for end in 1..=ids.len() {
            let delta = tokenizer
                .incremental_decode(&ids[..end], &mut prefix_offset, &mut read_offset)
                .unwrap();
            streamed.push_str(&delta);
            deltas.push(delta);
        }

        assert_eq!(deltas[0], "");
        assert_eq!(deltas[1], "");
        assert_eq!(deltas[2], "€");
        assert_eq!(streamed, tokenizer.decode(&ids).unwrap());
    }

    #[test]
    fn terminal_incremental_residual_matches_blocking_mid_utf8_cut() {
        let tokenizer = spacing_tokenizer();
        let ids = [0, 1, 9, 10];
        let full = tokenizer.decode(&ids).unwrap();
        assert_eq!(full, "ACK �");

        let mut prefix_offset = 0;
        let mut read_offset = 0;
        let mut streamed = String::new();
        for end in 1..=ids.len() {
            streamed.push_str(
                &tokenizer
                    .incremental_decode(&ids[..end], &mut prefix_offset, &mut read_offset)
                    .unwrap(),
            );
        }
        assert_eq!(streamed, "ACK ");
        streamed.push_str(
            &tokenizer
                .terminal_decode_residual(&ids, streamed.len())
                .unwrap(),
        );
        assert_eq!(streamed, full);
    }

    #[test]
    fn reasoning_phase_boundary_matches_blocking_mid_utf8_cut() {
        let tokenizer = spacing_tokenizer();
        // The next token would be </think> in the stream state and is excluded
        // from both blocking's reasoning slice and the phase residual decode.
        let reasoning_ids = [0, 1, 9, 10];
        let blocking_reasoning = tokenizer.decode(&reasoning_ids).unwrap();
        assert_eq!(blocking_reasoning, "ACK �");

        let mut prefix_offset = 0;
        let mut read_offset = 0;
        let mut streamed_reasoning = String::new();
        for end in 1..=reasoning_ids.len() {
            streamed_reasoning.push_str(
                &tokenizer
                    .incremental_decode(&reasoning_ids[..end], &mut prefix_offset, &mut read_offset)
                    .unwrap(),
            );
        }
        assert_eq!(streamed_reasoning, "ACK ");
        streamed_reasoning.push_str(
            &tokenizer
                .terminal_decode_residual(&reasoning_ids, streamed_reasoning.len())
                .unwrap(),
        );

        assert_eq!(streamed_reasoning, blocking_reasoning);
    }

    #[test]
    fn normalize_tool_call_arguments_parses_string_to_dict() {
        // The shape opencode sends back on the second turn: assistant
        // message with tool_calls whose function.arguments is a JSON
        // string. F76: must round-trip into a dict for MiniMax's
        // template `_args.items()` to work.
        let messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\":\"mkdir -p /tmp/x\",\"description\":\"make dir\"}"
                }
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        let args = &normalized[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "expected dict, got {args:?}");
        assert_eq!(args["command"], "mkdir -p /tmp/x");
        assert_eq!(args["description"], "make dir");
    }

    #[test]
    fn normalize_tool_call_arguments_leaves_non_tool_messages_alone() {
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(normalized, messages);
    }

    #[test]
    fn normalize_tool_call_arguments_passes_through_already_dict() {
        // Some clients send args pre-parsed as a dict — must not double-encode.
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": {"command": "ls"}}
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(
            normalized[0]["tool_calls"][0]["function"]["arguments"]["command"],
            "ls"
        );
    }

    /// Integration check: render a MiniMax chat template when configured via
    /// `FLUX_TEST_MINIMAX_TEMPLATE` (with string-encoded tool_call arguments
    /// on turn 2). Skips cleanly when the template path is not configured.
    #[test]
    fn render_minimax_template_with_string_tool_call_args() {
        let Ok(template_path) = std::env::var("FLUX_TEST_MINIMAX_TEMPLATE") else {
            return;
        };
        let Ok(template) = std::fs::read_to_string(&template_path) else {
            eprintln!("MiniMax template not on disk at {template_path}; skipping");
            return;
        };
        let env = super::jinja_helpers::build_jinja_env(&template).expect("template compiles");
        let tmpl = env.get_template("chat").unwrap();
        // The exact wire shape opencode sends back on turn 2.
        let messages = vec![
            json!({"role": "user", "content": "List /tmp"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{\"command\":\"ls -la /tmp\"}"
                    }
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_0", "content": "total 0"}),
            json!({"role": "user", "content": "Now uname -r"}),
        ];
        let normalized = normalize_tool_call_arguments(&messages);
        let messages_val = minijinja::Value::from_serialize(&normalized);
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => true,
            reasoning_effort => "high",
            disable_tool_steering => false,
            add_vision_id => false,
        };
        let rendered = tmpl
            .render(ctx)
            .expect("F76 must keep MiniMax template from raising on second-turn");
        // Sanity check: rendered output should contain the bash invoke
        // with command parameter — the items() iteration produced output.
        assert!(
            rendered.contains("<invoke name=\"bash\">"),
            "expected `<invoke name=\"bash\">` in render: {rendered}"
        );
        assert!(
            rendered.contains("<parameter name=\"command\">"),
            "expected `<parameter name=\"command\">` from .items() iteration: {rendered}"
        );
        assert!(
            rendered.contains("ls -la /tmp"),
            "expected the parsed command value in render: {rendered}"
        );
    }

    #[test]
    fn normalize_tool_call_arguments_invalid_json_string_left_alone() {
        // If args is a string but not valid JSON, leave as-is so the
        // template either coerces via tojson or the operator sees the
        // original error.
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": "not valid json {"}
            }]
        })];
        let normalized = normalize_tool_call_arguments(&messages);
        assert_eq!(
            normalized[0]["tool_calls"][0]["function"]["arguments"],
            "not valid json {"
        );
    }

    /// Regression: Gemma-4's bundled template calls `text.split('<channel|>')`
    /// inside its `strip_thinking` macro. minijinja has no `.split()` *method*
    /// on strings, so before the unknown-method bridge every assistant
    /// (model-role) turn raised `string has no method named split` and the
    /// whole chat request 400'd. A null-content tool message is part of the
    /// same conversation shape (coherence test "null content / tool role").
    #[test]
    fn render_gemma4_template_with_assistant_and_null_tool_content() {
        let template_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../jinja-templates/gemma4.jinja"
        );
        let raw = std::fs::read_to_string(template_path)
            .expect("bundled gemma4.jinja must be present in the repo");
        let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
        let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
        let tmpl = env.get_template("chat").unwrap();
        // The exact shape of the "null content / tool role" coherence case:
        // an assistant turn (exercises strip_thinking → .split) plus a
        // tool-role message whose content is null.
        let messages = vec![
            json!({"role": "user", "content": "What time is it?"}),
            json!({"role": "assistant", "content": "I'll check."}),
            json!({"role": "tool", "content": null}),
            json!({"role": "user", "content": "Thanks."}),
        ];
        let messages_val = minijinja::Value::from_serialize(&messages);
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => false,
            bos_token => "<bos>",
        };
        let rendered = tmpl
            .render(ctx)
            .expect("Gemma-4 template must render assistant + null-content tool message");
        // The assistant content survived strip_thinking's .split() round-trip.
        assert!(
            rendered.contains("I'll check."),
            "expected assistant content in render: {rendered}"
        );
    }
}
