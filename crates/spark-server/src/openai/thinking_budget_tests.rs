// SPDX-License-Identifier: AGPL-3.0-only

use super::ChatCompletionRequest;
use serde_json::{Value, json};

fn request(mut fields: Value) -> ChatCompletionRequest {
    fields["model"] = json!("test");
    fields["messages"] = json!([{"role": "user", "content": "hi"}]);
    serde_json::from_value(fields).expect("valid thinking request")
}

#[test]
fn enabled_without_budget_defers_to_model_policy_on_every_channel() {
    for fields in [
        json!({"thinking": {"type": "enabled"}}),
        json!({"thinking": {}}),
        json!({"chat_template_kwargs": {"enable_thinking": true}}),
        json!({"enable_thinking": true}),
    ] {
        let req = request(fields.clone());
        assert!(req.thinking_explicitly_requested(), "{fields}");
        for model_default in [false, true] {
            assert_eq!(
                req.resolve_thinking(model_default),
                (true, None),
                "{fields}"
            );
        }
    }
}

#[test]
fn explicit_budgets_disable_and_zero_semantics_are_unchanged() {
    for (fields, expected) in [
        (
            json!({"thinking": {"budget_tokens": 300}}),
            (true, Some(300)),
        ),
        (json!({"thinking_token_budget": 400}), (true, Some(400))),
        (
            json!({"chat_template_kwargs": {"thinking_budget": 500}}),
            (true, Some(500)),
        ),
        (
            json!({"thinking": {"type": "disabled", "budget_tokens": 300}}),
            (false, Some(0)),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            (false, Some(0)),
        ),
        (json!({"thinking_token_budget": 0}), (false, Some(0))),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 0}}),
            (false, Some(0)),
        ),
        // This existing Anthropic zero-budget distinction is not changed.
        (json!({"thinking": {"budget_tokens": 0}}), (true, Some(0))),
    ] {
        let req = request(fields.clone());
        assert!(req.thinking_explicitly_requested(), "{fields}");
        for model_default in [false, true] {
            assert_eq!(req.resolve_thinking(model_default), expected, "{fields}");
        }
    }
}

#[test]
fn known_and_unknown_effort_budgets_are_unchanged_on_both_spellings() {
    for (effort, budget) in [
        ("none", 0),
        ("minimal", 64),
        ("low", 128),
        ("medium", 256),
        ("high", 512),
        ("xhigh", 1024),
        ("max", 1024),
        ("turbo", 256),
        ("", 256),
        ("90", 256),
    ] {
        for fields in [
            json!({"reasoning": {"effort": effort}}),
            json!({"reasoning_effort": effort}),
        ] {
            let req = request(fields.clone());
            assert!(req.thinking_explicitly_requested(), "{fields}");
            for model_default in [false, true] {
                assert_eq!(
                    req.resolve_thinking(model_default),
                    (budget > 0, Some(budget)),
                    "{fields}",
                );
            }
        }
    }
}

#[test]
fn omitted_budget_does_not_fall_through_to_lower_priority_controls() {
    for (fields, expected) in [
        (
            json!({"thinking": {"type": "enabled"}, "thinking_token_budget": 700}),
            (true, None),
        ),
        (
            json!({"thinking_token_budget": 700, "reasoning_effort": "low"}),
            (true, Some(700)),
        ),
        (
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"thinking_budget": 300}}),
            (true, Some(128)),
        ),
        (
            json!({"reasoning": {"effort": "high"}, "reasoning_effort": "low"}),
            (true, Some(512)),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": false}, "enable_thinking": true}),
            (false, Some(0)),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true}, "enable_thinking": true}),
            (true, None),
        ),
        (
            json!({"thinking": {"type": "disabled"}, "thinking_token_budget": 700}),
            (false, Some(0)),
        ),
    ] {
        let req = request(fields.clone());
        for model_default in [false, true] {
            assert_eq!(req.resolve_thinking(model_default), expected, "{fields}");
        }
    }
}

#[test]
fn silent_and_legacy_false_still_use_the_model_default() {
    for fields in [json!({}), json!({"enable_thinking": false})] {
        let req = request(fields);
        assert!(!req.thinking_explicitly_requested());
        assert_eq!(req.resolve_thinking(true), (true, None));
        assert_eq!(req.resolve_thinking(false), (false, None));
    }
}

#[test]
fn qwen38_vocabulary_effort_budgets_follow_the_rendered_instruction() {
    use super::qwen38_effort_thinking_budget as qwen38;
    for (effort, expected) in [
        ("none", (false, Some(0))),
        // Rendered as "low: keep your thinking brief".
        ("minimal", (true, Some(1024))),
        ("low", (true, Some(1024))),
        // Rendered with no instruction.
        ("medium", (true, Some(8192))),
        // Rendered as the xhigh "think carefully" instruction, like an
        // omitted effort, so they take the model-policy budget.
        ("high", (true, None)),
        ("xhigh", (true, None)),
        ("max", (true, None)),
        ("turbo", (true, None)),
        ("90", (true, None)),
    ] {
        for fields in [
            json!({"reasoning": {"effort": effort}}),
            json!({"reasoning_effort": effort}),
        ] {
            let req = request(fields.clone());
            assert!(req.thinking_explicitly_requested(), "{fields}");
            for model_default in [false, true] {
                assert_eq!(
                    req.resolve_thinking_with_effort(model_default, &qwen38),
                    expected,
                    "{fields}",
                );
            }
        }
    }
}

#[test]
fn qwen38_vocabulary_leaves_explicit_budgets_and_priority_alone() {
    use super::qwen38_effort_thinking_budget as qwen38;
    for (fields, expected) in [
        (
            json!({"thinking": {"budget_tokens": 300}, "reasoning_effort": "high"}),
            (true, Some(300)),
        ),
        (
            json!({"thinking_token_budget": 700, "reasoning_effort": "low"}),
            (true, Some(700)),
        ),
        (
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"thinking_budget": 300}}),
            (true, Some(1024)),
        ),
        (
            json!({"reasoning": {"effort": "high"}, "reasoning_effort": "low"}),
            (true, None),
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            (true, None),
        ),
        (json!({}), (false, None)),
    ] {
        let req = request(fields.clone());
        assert_eq!(
            req.resolve_thinking_with_effort(false, &qwen38),
            expected,
            "{fields}"
        );
    }
}

#[test]
fn top_level_thinking_budget_and_reserve_fields_resolve_cleanly() {
    let req = request(json!({"thinking_budget": 1536, "thinking_reserve": 2048}));
    assert!(req.thinking_explicitly_requested());
    assert_eq!(req.thinking_reserve, Some(2048));
    assert_eq!(req.resolve_thinking(false), (true, Some(1536)));

    let zero = request(json!({"thinking_budget": 0}));
    assert!(zero.thinking_explicitly_requested());
    assert_eq!(zero.resolve_thinking(true), (false, Some(0)));
}

#[test]
fn integer_reasoning_effort_maps_proportionally_to_budget() {
    for (pct, expected) in [(1, 11), (25, 256), (50, 512), (75, 768), (100, 1024)] {
        for fields in [
            json!({"reasoning_effort": pct}),
            json!({"reasoning": {"effort": pct}}),
        ] {
            let req = request(fields.clone());
            assert!(req.thinking_explicitly_requested(), "{fields}");
            assert_eq!(req.requested_reasoning_effort_int(), Some(pct));
            assert_eq!(req.resolve_thinking(false), (true, Some(expected)));
        }
    }
}
