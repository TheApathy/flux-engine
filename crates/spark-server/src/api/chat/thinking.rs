// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single
// request. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. Request body (`reasoning_effort`, `thinking.budget_tokens`, …)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::super::failures::recent_message_is_tool_error;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    defaults: &crate::runtime_settings::RequestDefaults,
    req: &mut ChatCompletionRequest,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if state.disable_thinking {
        return (false, None);
    }
    // The thinking default, budgets and effort table come from the request's
    // admission snapshot (`/v1/flux/settings`); with no runtime change they
    // are exactly the startup values.
    let table = if state.tokenizer.renders_qwen38_reasoning_effort() {
        crate::openai::qwen38_effort_thinking_budget
    } else {
        crate::openai::generic_effort_thinking_budget
    };
    // A runtime thinking default replaces the model default; otherwise the
    // model default stands exactly as before (a launch
    // --default-chat-template-kwargs reaches this resolution as request kwargs).
    let model_default = if defaults.thinking_default_runtime {
        defaults.thinking_default
    } else {
        state.behavior.thinking_default
    };
    let (et, tb) =
        req.resolve_thinking_with_effort(model_default, &|e| defaults.effort_budget(e, table));
    let mt = u32::try_from(generation_max_tokens(
        req.max_tokens,
        tools_active,
        state.tool_max_tokens,
    ))
    .unwrap_or(u32::MAX);
    let max_budget = defaults.max_thinking_budget;
    let active_reserve = effective_thinking_reserve(req.thinking_reserve).filter(|&r| r > 0);
    let effective_mt = match active_reserve {
        Some(r) => mt.saturating_add(r),
        None => mt,
    };
    let tb = match req.requested_reasoning_effort_int() {
        Some(pct) => {
            let effort_ceiling = if max_budget > 0 {
                max_budget.min(effective_mt)
            } else {
                effective_mt
            };
            Some(crate::openai::proportional_effort_budget(
                pct,
                effort_ceiling,
            ))
        }
        None => tb,
    };
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    let et = if tools_active
        && !state.behavior.thinking_in_tools
        && !req.thinking_explicitly_requested()
    {
        false
    } else {
        et
    };
    // Opt-in (`FLUX_AGENT_GUARDS=1`) auto-disable thinking on turns following a tool error.
    let et = if et
        && super::super::chat_phases::agent_guards_enabled()
        && recent_message_is_tool_error(&req.messages)
    {
        tracing::info!(
            "post-error guard: disabling thinking on this turn (most recent message is tool error)"
        );
        req.record_intervention("F28", "disable_thinking_after_tool_error");
        false
    } else {
        et
    };
    let budget = if et {
        let answer_reserve = if active_reserve.is_some() {
            0
        } else {
            defaults.thinking_answer_reserve
        };
        let capped = cap_thinking_budget(
            tb,
            max_budget,
            effective_mt,
            tools_active,
            state.behavior.thinking_in_tools,
            answer_reserve,
        );
        Some(match active_reserve {
            Some(r) => capped.min(r),
            None => capped,
        })
    } else {
        None
    };
    (et, budget)
}

pub(super) fn cap_thinking_budget(
    requested: Option<u32>,
    model_max: u32,
    generation_max: u32,
    tools_active: bool,
    thinking_in_tools: bool,
    answer_reserve: u32,
) -> u32 {
    let safety_cap_pct = if tools_active && thinking_in_tools {
        7
    } else {
        9
    };
    let safety_max =
        ((generation_max as u64 * safety_cap_pct as u64) / 10).clamp(1, u32::MAX as u64) as u32;
    let model_ceiling = if model_max > 0 { model_max } else { u32::MAX };
    match requested {
        Some(budget) => budget.min(model_ceiling).min(safety_max),
        // A budget the client did not size also keeps answer room, so a large
        // model ceiling cannot spend nearly all of `max_tokens` on reasoning.
        None => model_ceiling
            .min(safety_max)
            .min(omitted_budget_answer_cap(generation_max, answer_reserve)),
    }
}

/// Largest proportional answer reserve: past it a big `max_tokens` keeps
/// its extra room for thinking instead of reserving 15% of it.
const MAX_PROPORTIONAL_ANSWER_RESERVE: u32 = 4096;

/// Thinking ceiling that leaves `max(min(15%, 4096), min(answer_reserve,
/// half))` of the generation allowance for the answer. `answer_reserve = 0`
/// disables it.
fn omitted_budget_answer_cap(generation_max: u32, answer_reserve: u32) -> u32 {
    if answer_reserve == 0 {
        return u32::MAX;
    }
    let proportional =
        ((generation_max as u64 * 15 / 100) as u32).min(MAX_PROPORTIONAL_ANSWER_RESERVE);
    let room = proportional.max(answer_reserve.min(generation_max / 2));
    generation_max.saturating_sub(room).max(1)
}

/// Re-apply the thinking safety cap after the generation allowance shrank
/// (a server-defaulted `max_tokens` clamped to the remaining context).
pub(super) fn recap_thinking_budget(
    budget: u32,
    generation_max: usize,
    tools_active: bool,
    thinking_in_tools: bool,
    answer_reserve: u32,
) -> u32 {
    let generation_max = u32::try_from(generation_max).unwrap_or(u32::MAX);
    // The resolved budget is the ceiling; the smaller allowance re-applies
    // both the safety cap and the model's answer reserve.
    cap_thinking_budget(
        None,
        budget,
        generation_max,
        tools_active,
        thinking_in_tools,
        answer_reserve,
    )
}

/// Generation allowance after the tools-active server cap. Thinking-budget
/// resolution and sampling must consume this same ceiling: resolving from the
/// raw client request first can grant reasoning more tokens than the whole
/// tool turn is allowed to emit.
pub(super) fn generation_max_tokens(
    max_tokens: usize,
    tools_active: bool,
    tool_max_tokens: usize,
) -> usize {
    if tools_active {
        max_tokens.min(tool_max_tokens)
    } else {
        max_tokens
    }
}

/// Sentinel indicating an unset optional `u32` server control in atomic storage.
const UNSET_U32: u64 = u64::MAX;

static SERVER_THINKING_RESERVE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(UNSET_U32);
static SERVER_THINKING_OFF_BUDGET: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(UNSET_U32);

/// Publish startup `--thinking-reserve` / `FLUX_THINKING_RESERVE` and
/// `--thinking-off-budget` / `FLUX_THINKING_OFF_BUDGET` controls once at boot.
pub(crate) fn set_server_thinking_controls(
    thinking_reserve: Option<u32>,
    thinking_off_budget: Option<u32>,
) {
    use std::sync::atomic::Ordering;
    SERVER_THINKING_RESERVE.store(
        thinking_reserve.map_or(UNSET_U32, u64::from),
        Ordering::Release,
    );
    SERVER_THINKING_OFF_BUDGET.store(
        thinking_off_budget.map_or(UNSET_U32, u64::from),
        Ordering::Release,
    );
}

fn read_env_u32(primary: &str, alias: &str) -> Option<u32> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(alias).ok())
        .and_then(|s| s.trim().parse::<u32>().ok())
}

/// Effective `thinking_reserve` for a request: per-request field wins, then
/// CLI `--thinking-reserve`, then `FLUX_THINKING_RESERVE` / `ATLAS_THINKING_RESERVE`.
pub(crate) fn effective_thinking_reserve(req_reserve: Option<u32>) -> Option<u32> {
    if let Some(r) = req_reserve {
        return Some(r);
    }
    let stored = SERVER_THINKING_RESERVE.load(std::sync::atomic::Ordering::Acquire);
    if stored != UNSET_U32 {
        return u32::try_from(stored).ok();
    }
    read_env_u32("FLUX_THINKING_RESERVE", "ATLAS_THINKING_RESERVE")
}

/// Effective `--thinking-off-budget` / `FLUX_THINKING_OFF_BUDGET` cap when
/// `enable_thinking == false` yet a template or spontaneous token opens `<think>`.
pub(crate) fn effective_thinking_off_budget() -> Option<u32> {
    let stored = SERVER_THINKING_OFF_BUDGET.load(std::sync::atomic::Ordering::Acquire);
    if stored != UNSET_U32 {
        return u32::try_from(stored).ok();
    }
    read_env_u32("FLUX_THINKING_OFF_BUDGET", "ATLAS_THINKING_OFF_BUDGET")
}

/// Cap spontaneous `<think>` budget when `--thinking-off-budget` is configured.
pub(crate) fn effective_spontaneous_think_budget(base: u32) -> u32 {
    match effective_thinking_off_budget() {
        Some(off_cap) => base.min(off_cap),
        None => base,
    }
}

/// Reserve `max_tokens` for visible output by extending total generation budget
/// by `reserve` (clamped safely to remaining context window `max_seq_len - prompt_len`)
/// and clamping `thinking_budget` to `min(thinking_budget, reserve, added_reserve)`.
#[must_use]
pub(crate) fn apply_thinking_reserve(
    enable_thinking: bool,
    req_reserve: Option<u32>,
    max_tokens: usize,
    thinking_budget: Option<u32>,
    prompt_len: usize,
    max_seq_len: usize,
) -> (usize, Option<u32>) {
    let Some(reserve) =
        effective_thinking_reserve(req_reserve).filter(|&r| enable_thinking && r > 0)
    else {
        return (max_tokens, thinking_budget);
    };
    let ctx_room = max_seq_len.saturating_sub(prompt_len.saturating_add(max_tokens));
    let added_reserve = (reserve as usize).min(ctx_room);
    let extended_max = max_tokens.saturating_add(added_reserve);
    let mut capped = thinking_budget.unwrap_or(reserve).min(reserve);
    if added_reserve > 0 {
        capped = capped.min(u32::try_from(added_reserve).unwrap_or(u32::MAX));
    } else {
        // Context window already full: cap thinking to at most half of max_tokens
        // so visible output is still never starved.
        capped = capped.min(u32::try_from(max_tokens / 2).unwrap_or(u32::MAX).max(1));
    }
    (extended_max, Some(capped))
}

#[cfg(test)]
mod tests {
    use super::{cap_thinking_budget, generation_max_tokens};

    #[test]
    fn tool_cap_is_the_thinking_and_generation_ceiling() {
        let ceiling = generation_max_tokens(2048, true, 256);
        assert_eq!(ceiling, 256);
        // The normal 90% thinking safety cap now leaves room in the actual
        // tool-turn allowance: 230, not 90% of the raw 2048 (=1843).
        assert_eq!(
            cap_thinking_budget(None, 2048, ceiling as u32, true, false, 0),
            230
        );
        assert_eq!(generation_max_tokens(128, true, 256), 128);
        assert_eq!(generation_max_tokens(2048, false, 256), 2048);
    }

    #[test]
    fn omitted_budget_resolution_consumes_the_existing_model_and_tool_policy() {
        use crate::openai::ChatCompletionRequest;
        use serde_json::json;

        for fields in [
            json!({"thinking": {"type": "enabled"}}),
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            json!({"enable_thinking": true}),
        ] {
            let mut body = fields;
            body["model"] = json!("test");
            body["messages"] = json!([{"role": "user", "content": "hi"}]);
            let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            let (enabled, requested) = req.resolve_thinking(false);
            assert!(enabled);
            assert_eq!(requested, None);
            for (output, tools, thinking_in_tools, tool_cap, expected) in [
                (8192, false, false, 256, 2048),
                (1000, false, false, 256, 900),
                (8192, true, true, 256, 179),
                (8192, true, false, 256, 230),
                (128, true, true, 256, 89),
                (1, false, false, 256, 1),
            ] {
                let maximum = generation_max_tokens(output, tools, tool_cap) as u32;
                assert_eq!(
                    cap_thinking_budget(requested, 2048, maximum, tools, thinking_in_tools, 0),
                    expected,
                );
            }
        }
    }

    #[test]
    fn explicit_budget_policy_is_unchanged_by_the_omission_fix() {
        assert_eq!(
            cap_thinking_budget(Some(256), 2048, 8192, false, false, 0),
            256
        );
        assert_eq!(cap_thinking_budget(Some(0), 2048, 8192, false, false, 0), 0);
        // Bug 2 fix: explicit budgets are now bounded by `model_max` when `model_max > 0`
        // (and unbounded by `model_max` when `model_max == 0`).
        assert_eq!(
            cap_thinking_budget(Some(5000), 2048, 8192, false, false, 0),
            2048
        );
        assert_eq!(
            cap_thinking_budget(Some(5000), 0, 8192, false, false, 0),
            5000
        );
        assert_eq!(
            cap_thinking_budget(Some(5000), 2048, 1000, false, false, 0),
            900
        );
    }

    #[test]
    fn omitted_budget_with_zero_model_max_uses_unbounded_ceiling() {
        // `model_max == 0` means no model-level cap; omitted budgets must fall
        // back to safety_max and answer_reserve rather than crushing to 0.
        assert_eq!(cap_thinking_budget(None, 0, 8192, false, false, 0), 7372);
        assert_eq!(cap_thinking_budget(None, 0, 4096, false, false, 1024), 3072);
    }

    #[test]
    fn apply_thinking_reserve_extends_generation_and_preserves_answer_quota() {
        use super::apply_thinking_reserve;

        // Plenty of context room: 512 visible + 2048 reserve = 2560 total, thinking capped at 2048.
        assert_eq!(
            apply_thinking_reserve(true, Some(2048), 512, Some(4096), 1000, 32768),
            (2560, Some(2048))
        );
        // Tight context room: prompt=7000 + answer=512 in max_seq_len=8192 leaves 680 extra tokens.
        // Total becomes 1192 and thinking_budget is clamped to 680 so all 512 answer tokens survive.
        assert_eq!(
            apply_thinking_reserve(true, Some(2048), 512, Some(2048), 7000, 8192),
            (1192, Some(680))
        );
        // Disabled thinking or zero reserve leaves budgets untouched.
        assert_eq!(
            apply_thinking_reserve(false, Some(2048), 512, None, 1000, 32768),
            (512, None)
        );
        assert_eq!(
            apply_thinking_reserve(true, Some(0), 512, Some(256), 1000, 32768),
            (512, Some(256))
        );
    }

    #[test]
    fn answer_reserve_keeps_room_for_omitted_budgets_only() {
        // Flash-Next profile: model ceiling 32768, answer reserve 1024.
        for (output, tools, thinking_in_tools, expected) in [
            // Default max_tokens: 1024 of 4096 stays for the answer.
            (4096, false, false, 3072),
            // 15% of a large allowance is more than the 1024 floor, but the
            // proportional reserve stops growing at 4096 (9.15.7).
            (32768, false, false, 28672),
            // The model ceiling still binds above it.
            (65536, false, false, 32768),
            // The 70% tool-turn cap is tighter than the reserve here.
            (8192, true, true, 5734),
            // Small allowances keep half for the answer.
            (1000, false, false, 500),
            (1, false, false, 1),
        ] {
            assert_eq!(
                cap_thinking_budget(None, 32768, output, tools, thinking_in_tools, 1024),
                expected,
                "max_tokens={output} tools={tools}",
            );
        }
        // Explicit budgets keep the existing 90% rule.
        assert_eq!(
            cap_thinking_budget(Some(5000), 32768, 4096, false, false, 1024),
            3686
        );
        assert_eq!(
            cap_thinking_budget(Some(2000), 32768, 4096, false, false, 1024),
            2000
        );
    }

    #[test]
    fn proportional_answer_reserve_is_capped_at_4096() {
        use super::omitted_budget_answer_cap;
        // (max_tokens, answer cap, resolved omitted budget with a model
        // ceiling above max_tokens). Reserve 1024 is Flash-Next's floor.
        for (output, cap, resolved) in [
            // The 1024 floor beats 15% (614).
            (4096, 3072, 3072),
            // 15% = 1228.
            (8192, 6964, 6964),
            // 15% = 2457: unchanged from 9.15.6.
            (16384, 13927, 13927),
            // 15% = 4915, capped at 4096 (was 27853).
            (32768, 28672, 28672),
            // 15% = 9830, capped at 4096 (was 55706); the unchanged 90%
            // safety cap (58982) then binds.
            (65536, 61440, 58982),
        ] {
            assert_eq!(omitted_budget_answer_cap(output, 1024), cap, "cap {output}");
            assert_eq!(
                cap_thinking_budget(None, u32::MAX, output, false, false, 1024),
                resolved,
                "resolved {output}",
            );
        }
        // A floor above the capped proportional term still wins.
        assert_eq!(omitted_budget_answer_cap(65536, 6000), 59536);
        // The tools 70% rule is unchanged.
        assert_eq!(
            cap_thinking_budget(None, u32::MAX, 32768, true, true, 1024),
            22937
        );
    }

    #[test]
    fn zero_answer_reserve_is_the_existing_policy() {
        for output in [1, 1000, 4096, 8192, 32768, 65536] {
            for (tools, thinking_in_tools) in [(false, false), (true, true), (true, false)] {
                let legacy = {
                    let pct = if tools && thinking_in_tools { 7 } else { 9 };
                    let safety = ((output as u64 * pct) / 10).clamp(1, u32::MAX as u64) as u32;
                    32768u32.min(safety)
                };
                assert_eq!(
                    cap_thinking_budget(None, 32768, output, tools, thinking_in_tools, 0),
                    legacy,
                );
            }
        }
    }

    #[test]
    fn recap_after_a_clamped_default_keeps_the_safety_cap() {
        use super::recap_thinking_budget;
        assert_eq!(recap_thinking_budget(29491, 2192, false, false, 0), 1972);
        assert_eq!(recap_thinking_budget(29491, 2192, true, true, 0), 1534);
        // A budget already inside the smaller allowance is kept.
        assert_eq!(recap_thinking_budget(512, 2192, false, false, 0), 512);
        // With Flash-Next's 1024-token answer reserve the clamped allowance
        // still leaves room for the answer: 2192 - 1024.
        assert_eq!(recap_thinking_budget(29491, 2192, false, false, 1024), 1168);
    }

    #[test]
    fn flash_next_high_effort_gets_more_than_medium() {
        // 9.15.5 capped high/xhigh/max at the old 2048 model ceiling while
        // medium had a fixed 8192, so high thought less than medium.
        use crate::openai::{ChatCompletionRequest, qwen38_effort_thinking_budget};
        use serde_json::json;

        let budget = |effort: &str, max_tokens: u32| {
            let req: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            }))
            .unwrap();
            let (enabled, requested) =
                req.resolve_thinking_with_effort(false, &qwen38_effort_thinking_budget);
            assert!(enabled, "{effort}");
            // Flash-Next profile: ceiling 32768, answer reserve 1024.
            cap_thinking_budget(requested, 32768, max_tokens, false, false, 1024)
        };
        for effort in ["high", "xhigh", "max"] {
            // 32768 - min(15%, 4096) = 28672 (27853 before 9.15.7).
            assert_eq!(budget(effort, 32768), 28672, "{effort}");
            assert!(budget(effort, 32768) > budget("medium", 32768), "{effort}");
            // At the 4096 default every effort is bounded by the allowance.
            assert_eq!(budget(effort, 4096), 3072, "{effort}");
        }
        assert_eq!(budget("medium", 32768), 8192);
        assert_eq!(budget("low", 32768), 1024);
    }
}
