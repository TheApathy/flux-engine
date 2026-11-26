// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed scheduler dispatch for a model-owned speculative decoder.

use super::*;

/// Internal speculation is admitted only when ordinary serving reduces to the
/// same raw, first-wins BF16 argmax used by `spec_verify`.
///
/// Sampled requests are deliberately held on serial decode.  The runtime has
/// the exact target distribution API, but this scheduler slice does not yet
/// consume the drafter's per-position q rows.
pub(super) fn internal_spec_eligible(
    a: &ActiveSeq,
    model: &dyn Model,
    adaptive_sampling: bool,
) -> bool {
    fast_path_seq_eligible(a)
        && !adaptive_sampling
        && a.grammar_state.is_none()
        && !a.inside_thinking
        && !a.think_ended
        && !a.think_just_ended
        && !a.force_end_thinking
        && !a.suppress_tool_call
        && !a.require_tool_call
        && !a.disable_mtp
        && a.remaining > 0
        && a.output_tokens.len() < a.max_output_tokens
        && !model.decode_logits_fp32()
        && !crate::dsv41::repetition::active()
}

fn fail_step(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    cache_on_finish: &mut Vec<bool>,
    stage: &str,
    error: anyhow::Error,
) {
    tracing::error!("internal DSpark {stage}: {error:#}");
    let mut a = active.remove(0);
    cache_on_finish.remove(0);
    send_error(
        model,
        &mut a,
        &format!("internal DSpark {stage}: {error:#}"),
    );
}

/// Execute one exact-greedy internal speculative step.
///
/// `Ok(None)` from `spec_verify` is the sole fallback edge: it means the
/// model made no mutation (normally because the verify frame would cross
/// max-seq), so one ordinary decode is safe.  Every error terminates and frees
/// the sequence; replaying after a possibly partial verify is forbidden.
#[allow(clippy::too_many_arguments)]
pub(super) fn step_internal_spec(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    cache_on_finish: &mut Vec<bool>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) {
    debug_assert_eq!(active.len(), 1);
    debug_assert_eq!(cache_on_finish.len(), 1);
    let t0 = std::time::Instant::now();
    let a = &mut active[0];

    if let Err(error) = model.ep_broadcast_cmd(a.last_token) {
        return fail_step(model, active, cache_on_finish, "broadcast", error);
    }
    let (drafts, target_argmax) = match model.spec_verify(a.last_token, &mut a.seq, None, None, 0) {
        Ok(Some(frame)) => frame,
        Ok(None) => {
            step_decode_only(
                model,
                active,
                cache_on_finish,
                think_end_token,
                think_start_token,
                code_fence_token,
                tool_call_start_token,
                tool_call_end_token,
                reflection_suppress_ids,
                adaptive_sampling,
            );
            return;
        }
        Err(error) => return fail_step(model, active, cache_on_finish, "verify", error),
    };

    if target_argmax.len() != drafts.len() + 1 {
        return fail_step(
            model,
            active,
            cache_on_finish,
            "verify contract",
            anyhow::anyhow!(
                "{} target rows for {} drafts (expected {})",
                target_argmax.len(),
                drafts.len(),
                drafts.len() + 1
            ),
        );
    }

    let a = &mut active[0];
    let structural = |token: u32| {
        a.eos_tokens.contains(&token)
            || [
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                im_start_hard_stop(),
            ]
            .contains(&Some(token))
    };
    let output_room = a
        .max_output_tokens
        .saturating_sub(a.output_tokens.len())
        .min(a.remaining);
    let accepted = drafts
        .iter()
        .zip(&target_argmax)
        .take(output_room)
        .take_while(|(draft, target)| draft == target && !structural(**draft))
        .count();

    if let Err(error) = model.spec_commit(&mut a.seq, accepted, 0) {
        return fail_step(model, active, cache_on_finish, "commit", error);
    }
    for &token in &drafts[..accepted] {
        emit_token(a, token, None);
        if a.finished {
            // Cancellation/backpressure can arrive after the atomic model
            // commit.  Teardown is safe, but this over-planned frame must not
            // enter the prefix cache.
            cache_on_finish[0] = false;
            return;
        }
        a.last_token = token;
    }

    let logits = model
        .logits_buffer_ptr()
        .offset(accepted * model.vocab_size() * 2);
    process_decode_logits(
        model,
        active,
        logits,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        reflection_suppress_ids,
        adaptive_sampling,
        honor_eos_inside_thinking(),
    );
    if active.is_empty() {
        cache_on_finish.clear();
    }
}

#[cfg(test)]
#[path = "internal_spec_step_tests.rs"]
mod tests;
