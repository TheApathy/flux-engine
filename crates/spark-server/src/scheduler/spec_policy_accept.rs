// SPDX-License-Identifier: AGPL-3.0-only

//! Full sampler-policy oracle for speculative content tokens.
//!
//! A speculative verifier produces raw target argmaxes, but normal serving
//! samples after grammar masks, think masks, penalties, logit bias, and the
//! adaptive sampler.  Those are not optional decorations: Qwen presets ship
//! with non-neutral presence/LZ/DRY penalties, so comparing drafts to raw
//! argmaxes changes the output from the very first bootstrap step.

use super::*;

/// Element layout of the resident target verify-logits buffer.
///
/// Keep this explicit instead of inferring the width from a row length: both
/// BF16 and FP32 rows are byte slices, and interpreting an FP32 row as BF16
/// changes the sampler oracle rather than producing a useful error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyLogitsFormat {
    Bf16,
    Fp32,
}

impl VerifyLogitsFormat {
    fn elem_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp32 => 4,
        }
    }

    fn is_fp32(self) -> bool {
        matches!(self, Self::Fp32)
    }
}

/// Whether raw greedy argmax is insufficient to reproduce the plain sampler.
pub(super) fn content_policy_required(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    ctx.adaptive_sampling
        || !fast_path_seq_eligible(a)
        || a.think_ended
        || a.think_just_ended
        || a.suppress_tool_call
        || a.grammar_state.is_some()
}

/// Whether a non-thinking DDTree may safely use the target's per-row greedy
/// argmaxes instead of the linear host policy walk.
///
/// `content_policy_required` is deliberately true for every post-thinking
/// sequence because think tags remain masked. Post-thinking trees stay
/// fail-closed even for neutral greedy requests: the current tree-row mask
/// cannot distinguish a safe no-op from FP32 logits, a disabled mask, D2H
/// failure, or an unresolved compact-row layout. `mtp_step` therefore drops
/// only the topology and preserves the top-1 spine for the BF16/FP32-capable
/// linear policy walk. Any other history-dependent or request-specific policy
/// remains fail-closed because its result depends on the chosen branch prefix.
pub(super) fn tree_content_raw_argmax_eligible(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    !ctx.adaptive_sampling
        && fast_path_seq_eligible(a)
        && !a.think_ended
        && !a.think_just_ended
        && !a.suppress_tool_call
        && !a.require_tool_call
        && a.output_tokens.len() >= a.min_tokens
        && a.grammar_state.is_none()
}

/// Whether a non-thinking sequence is eligible for cross-sequence batched flat
/// DFlash verify (`step_verify_dflash_batched`).
///
/// Mirrors [`accept_row_fast_path_ok`] so post-thinking (`a.think_ended`, e.g.
/// `--disable-thinking` or after `</think>`) neutral-greedy sequences can still
/// join the batched verify cohort. Whenever `a.think_ended` is true, the
/// batched commit path runs `dflash_think_mask_verified_at_row` against the
/// single-pass resident BF16 logits slab, keeping output bit-identical to the
/// per-sequence path while avoiding per-row logits D2H in the common case where
/// no `<think>`/`</think>` tag won argmax.
pub(super) fn batched_dflash_content_raw_argmax_eligible(
    a: &ActiveSeq,
    ctx: &ThinkSpecCtx<'_>,
    logits_fp32: bool,
) -> bool {
    !ctx.adaptive_sampling
        && fast_path_seq_eligible(a)
        && !a.inside_thinking
        && !a.think_just_ended
        && !a.suppress_tool_call
        && !a.require_tool_call
        && a.output_tokens.len() >= a.min_tokens
        && a.grammar_state.is_none()
        && !super::decode_logits_seq::logits_fnv_requires_row()
        && (!a.think_ended || !logits_fp32)
}

/// Sample and commit one non-thinking MTP bootstrap token through the same
/// host-side policy pipeline as `process_decode_logits`.
pub(super) fn bootstrap_content_token(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<u32> {
    debug_assert!(!a.inside_thinking);
    let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && a.temperature > 0.0)
        .then(|| {
            spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                super::stochastic_verify_walk::seq_abs_pos(a),
            )
        });
    let device = if model.device_topk_supported() {
        super::device_topk_spec::sample_spec_row(model, a, logits, model.decode_logits_fp32(), ctx)
    } else {
        super::device_topk_spec::DeviceRow::Host
    };
    let (tok, lp) = match device {
        super::device_topk_spec::DeviceRow::Sampled(sampled) => sampled,
        super::device_topk_spec::DeviceRow::Failed => {
            a.finished = true;
            return None;
        }
        super::device_topk_spec::DeviceRow::Host => bootstrap_content_host(model, a, logits, ctx)?,
    };
    let receipt = if super::logits_row_receipt::enabled() {
        super::logits_row_receipt::capture_device(
            model,
            logits,
            a.session_hash,
            &a.output_tokens,
            tok,
            super::logits_row_receipt::single_location("content_bootstrap", a, 0),
        )
    } else {
        None
    };
    let output_len = a.output_tokens.len();
    emit_token(a, tok, lp);
    if let Some(receipt) = receipt {
        super::logits_row_receipt::output_map(&receipt, &a.output_tokens);
    }
    // Plain single-token decode snapshots the recurrent state after a
    // non-EOS content boundary.  This bootstrap has advanced exactly one
    // input position, so taking the same snapshot is safe (unlike a
    // multi-row speculative verify, whose state is still over-extended).
    if a.output_tokens.len() > output_len && !a.inside_thinking && !a.eos_tokens.contains(&tok) {
        rollback::snapshot_boundary_if_ssm(a, model);
    }
    Some(tok)
}

/// The host half of [`bootstrap_content_token`]: full-row D2H, the optional
/// `ATLAS_SPEC_BOOTSTRAP_TRACE` line, and the plain path.
fn bootstrap_content_host(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<(u32, Option<crate::api::TokenLogprobs>)> {
    let vocab = model.vocab_size();
    let logits_fp32 = model.decode_logits_fp32();
    let elem_bytes = if logits_fp32 { 4 } else { 2 };
    let mut buf = vec![0u8; vocab * elem_bytes];
    if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
        tracing::error!("spec-policy bootstrap: logits D2H failed: {e:#}");
        a.finished = true;
        return None;
    }
    let trace = std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1");
    let raw_argmax = trace.then(|| {
        (0..vocab)
            .max_by(|&x, &y| {
                let read = |j: usize| {
                    if logits_fp32 {
                        let off = j * 4;
                        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
                    } else {
                        bf16_to_f32(buf[j * 2], buf[j * 2 + 1])
                    }
                };
                read(x).total_cmp(&read(y))
            })
            .unwrap_or(0) as u32
    });
    let (tok, lp) = process_seq_logits(
        a,
        &buf,
        0,
        vocab,
        elem_bytes,
        logits_fp32,
        a.think_end_token,
        a.think_start_token,
        a.tool_call_start_token,
        a.tool_call_end_token,
        ctx.reflection_suppress_ids,
        ctx.adaptive_sampling,
    );
    if trace {
        tracing::info!(
            raw_argmax = raw_argmax.unwrap_or(0),
            sampled = tok,
            think_ended = a.think_ended,
            think_start = ?a.think_start_token,
            think_end = ?a.think_end_token,
            "SPEC_POLICY_BOOTSTRAP"
        );
    }
    Some((tok, lp))
}

/// Environment kill-switch for the raw-BF16 accept fast path.
///
/// Default ON. Set `ATLAS_ACCEPT_FAST_ARGMAX=0` to force every row through
/// the full `process_seq_logits` walk (use only when debugging a drift the
/// fast path might have introduced — the path is intended to be
/// byte-identical by construction).
fn accept_fast_argmax_enabled() -> bool {
    std::env::var("ATLAS_ACCEPT_FAST_ARGMAX").as_deref() != Ok("0")
}

/// Greedy pick over a host-copied BF16 logits row, skipping `skip_toks`.
///
/// Must commit exactly what the full walk's greedy branch would: both go
/// through the shared first-wins rule
/// ([`spark_runtime::sampler::argmax_bf16_first_wins`]), which is also the
/// device argmax's rule.
fn argmax_bf16_skip(bytes: &[u8], skip_toks: &[u32], vocab: usize) -> Option<u32> {
    spark_runtime::sampler::argmax_bf16_first_wins(&bytes[..2 * vocab], |tok| {
        !skip_toks.contains(&tok)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifiedArgmaxRow {
    Direct(u32),
    Fetch,
    Invalid,
}

/// Decide whether a verifier argmax is itself the exact post-policy token.
///
/// `verified` is already the device's first-wins raw argmax. A host row is
/// needed only when a post-thinking mask removes that winner or qualification
/// instrumentation must hash the raw logits. An out-of-vocabulary verifier
/// result violates the model contract and must not be emitted or repaired
/// silently from a different source.
fn verified_argmax_row(
    verified: u32,
    skip_toks: &[u32],
    vocab: usize,
    logits_fnv_requires_row: bool,
) -> VerifiedArgmaxRow {
    if verified as usize >= vocab {
        VerifiedArgmaxRow::Invalid
    } else if logits_fnv_requires_row || skip_toks.contains(&verified) {
        VerifiedArgmaxRow::Fetch
    } else {
        VerifiedArgmaxRow::Direct(verified)
    }
}

/// Per-row gate for the raw-BF16 accept fast path.
///
/// When all of these hold, the full walk reduces to a masked argmax over
/// the raw BF16 row:
///
/// - `fast_path_seq_eligible(a)`: temp 0, all penalties neutral, no
///   logit_bias, no logprobs — the sampler's greedy bypass is a pure
///   post-intervention argmax.
/// - no adaptive sampling (entropy/zone side effects must stay on the full
///   path).
/// - not inside thinking: reflection suppression, think-efficiency wave,
///   F2 confidence early-stop, and tool-call hard masks all apply only
///   while thinking.
/// - `!suppress_tool_call` / `!require_tool_call`: the −12 tool bias and
///   the one-shot pin-to-tool-call-start would re-order the argmax.
/// - no grammar: the grammar bitmask would re-order logits.
/// - no DeepSeek-v41 host cycle breaker (`!dsv41::repetition::active()`).
/// - BF16 rows only: the FP32 path needs an f32 row, not a BF16 one.
///
/// The `</think>`/`<think>` masks applied by the full walk when
/// `think_ended` are reproduced here as a skip list; masking to −inf and
/// skipping are equivalent for argmax selection.
fn accept_row_fast_path_ok(
    a: &ActiveSeq,
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
) -> bool {
    accept_fast_argmax_enabled()
        && !logits_format.is_fp32()
        && fast_path_seq_eligible(a)
        && !ctx.adaptive_sampling
        && !crate::dsv41::repetition::active()
        && !a.inside_thinking
        && !a.suppress_tool_call
        && !a.require_tool_call
        && a.grammar_state.is_none()
}

/// Walk flat DFlash verify rows using the actual plain-serving sampler.
///
/// Each processed token is committed immediately so history penalties,
/// grammar state, adaptive state, and seeded sampling see exactly the same
/// prefix they would see during serial decode.  Acceptance stops at the first
/// draft mismatch; that target token is the bonus and has not yet been fed to
/// the model.  Phase-transition tokens also remain bonuses so the next step
/// feeds them under the new phase exactly once.
///
/// Fast path: when `accept_row_fast_path_ok` holds and the verifier's raw
/// winner is not one of the two post-thinking mask tokens, `verified[i]` is
/// committed directly. That skips both logits D2H and the CPU scan. A masked
/// winner (or an active logits-FNV receipt) fetches the row and uses the prior
/// raw-BF16 masked argmax path. Both are byte-identical by construction:
/// BF16→FP32 is a monotone injection, all other interventions that could
/// re-order the argmax are gated out, and fallback masking uses the same
/// first-wins tie rule.
pub(super) fn dflash_content_accept(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
    vocab: usize,
    fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> SpecAcceptOutcome {
    dflash_content_accept_routed(
        a,
        drafts,
        verified,
        ctx,
        logits_format,
        vocab,
        fetch_row,
        |_, _| super::device_topk_spec::DeviceRow::Host,
    )
}

/// [`dflash_content_accept`] with a device top-k row source: every row that
/// would run the full `process_seq_logits` walk first asks `device_row(i, a)`,
/// which may sample it on the device route (`ATLAS_SAMPLER_DEVICE_TOPK=1`).
#[allow(clippy::too_many_arguments)]
pub(super) fn dflash_content_accept_routed(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
    vocab: usize,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut device_row: impl FnMut(usize, &mut ActiveSeq) -> super::device_topk_spec::DeviceRow,
) -> SpecAcceptOutcome {
    debug_assert!(!a.inside_thinking);
    let mut row_buf = Vec::new();
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;
    let elem_bytes = logits_format.elem_bytes();
    let mut masked_fast_engaged_logged = false;
    let logits_fnv_requires_row = super::decode_logits_seq::logits_fnv_requires_row();
    let mut skip_storage = [0u32; 2];
    let mut skip_len = 0usize;
    if a.think_ended && !crate::dsv41::serving() {
        if let Some(t) = a.think_start_token {
            skip_storage[skip_len] = t;
            skip_len += 1;
        }
        if let Some(t) = a.think_end_token {
            skip_storage[skip_len] = t;
            skip_len += 1;
        }
    }
    let skip = &skip_storage[..skip_len];

    for i in 0..verified.len() {
        let fast_path_ok = accept_row_fast_path_ok(a, ctx, logits_format);
        let fast_row = fast_path_ok
            .then(|| verified_argmax_row(verified[i], skip, vocab, logits_fnv_requires_row));
        let (target, lp) = match fast_row {
            Some(VerifiedArgmaxRow::Direct(tok)) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_ACCEPT_FAST_ARGMAX: raw target argmax accepted without logits D2H"
                    );
                });
                (tok, None)
            }
            Some(VerifiedArgmaxRow::Invalid) => {
                tracing::error!(
                    row = i,
                    token = verified[i],
                    vocab,
                    "spec-policy verify: raw argmax is outside vocabulary; finishing seq"
                );
                a.finished = true;
                return SpecAcceptOutcome {
                    num_accepted,
                    bonus: None,
                    consumed_rows,
                };
            }
            Some(VerifiedArgmaxRow::Fetch) | None => {
                let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                    == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                    && a.temperature > 0.0)
                    .then(|| {
                        spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                            super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                            super::stochastic_verify_walk::seq_abs_pos(a),
                        )
                    });
                let routed = if fast_path_ok {
                    super::device_topk_spec::DeviceRow::Host
                } else {
                    device_row(i, a)
                };
                match routed {
                    super::device_topk_spec::DeviceRow::Sampled(sampled) => sampled,
                    super::device_topk_spec::DeviceRow::Failed => {
                        tracing::error!(
                            "spec-policy verify: device top-k row {i} failed; finishing seq"
                        );
                        a.finished = true;
                        return SpecAcceptOutcome {
                            num_accepted,
                            bonus: None,
                            consumed_rows,
                        };
                    }
                    super::device_topk_spec::DeviceRow::Host => {
                        if !fetch_row(i, &mut row_buf) {
                            tracing::error!(
                                "spec-policy verify: logits row {i} unavailable; finishing seq"
                            );
                            a.finished = true;
                            return SpecAcceptOutcome {
                                num_accepted,
                                bonus: None,
                                consumed_rows,
                            };
                        }
                        if row_buf.len() != vocab * elem_bytes {
                            tracing::error!(
                                row = i,
                                bytes = row_buf.len(),
                                expected_bytes = vocab * elem_bytes,
                                elem_bytes,
                                "spec-policy verify: malformed logits row; finishing seq"
                            );
                            a.finished = true;
                            return SpecAcceptOutcome {
                                num_accepted,
                                bonus: None,
                                consumed_rows,
                            };
                        }
                        if fast_path_ok {
                            if !masked_fast_engaged_logged {
                                masked_fast_engaged_logged = true;
                                tracing::info!(
                                    "spec-policy accept: raw-BF16 masked-argmax row fallback engaged"
                                );
                            }
                            super::decode_logits_seq::log_logits_fnv(
                                a.session_hash,
                                a.output_tokens.len(),
                                &row_buf,
                            );
                            (argmax_bf16_skip(&row_buf, skip, vocab).unwrap_or(0), None)
                        } else {
                            process_seq_logits(
                                a,
                                &row_buf,
                                0,
                                vocab,
                                elem_bytes,
                                logits_format.is_fp32(),
                                a.think_end_token,
                                a.think_start_token,
                                a.tool_call_start_token,
                                a.tool_call_end_token,
                                ctx.reflection_suppress_ids,
                                ctx.adaptive_sampling,
                            )
                        }
                    }
                }
            }
        };

        let was_inside_thinking = a.inside_thinking;
        let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
        let receipt = super::logits_row_receipt::current_verify_row(a, i, target);
        if !emit_token_for_verify(a, target, lp) {
            return SpecAcceptOutcome {
                num_accepted,
                bonus: None,
                consumed_rows,
            };
        }
        if let Some(receipt) = receipt {
            super::logits_row_receipt::output_map(&receipt, &a.output_tokens);
        }
        consumed_rows += 1;
        let phase_boundary = was_inside_thinking != a.inside_thinking
            || a.think_start_token == Some(target)
            || a.think_end_token == Some(target);

        if phase_boundary || !draft_match {
            if !a.finished {
                a.last_token = target;
            }
            return SpecAcceptOutcome {
                num_accepted,
                bonus: Some(target),
                consumed_rows,
            };
        }
        if a.finished {
            return SpecAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
                consumed_rows,
            };
        }
        num_accepted += 1;
    }

    debug_assert!(verified.is_empty());
    a.finished = true;
    SpecAcceptOutcome {
        num_accepted,
        bonus: None,
        consumed_rows,
    }
}

/// Bind the content-policy walk to the resident BF16 or FP32 verify-logits rows.
pub(super) fn run_dflash_content_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    run_dflash_content_accept_at_row(model, a, 0, drafts, verified, ctx)
}

pub(super) fn run_dflash_content_accept_for_frame(
    model: &dyn Model,
    a: &mut ActiveSeq,
    inputs: &[u32],
    pre_verify_len: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    let _receipt_guard =
        super::logits_row_receipt::enter_verify_frame(model, a, inputs, pre_verify_len);
    run_dflash_content_accept_at_row(model, a, 0, drafts, verified, ctx)
}

/// Emit the one-time engagement receipt when batched GPU target Gumbel-Max
/// verify serves its first verify frame or sequence.
pub(super) fn log_batched_gumbel_engaged_once(rows: usize, seqs: usize) {
    static ENGAGED_ONCE: std::sync::Once = std::sync::Once::new();
    ENGAGED_ONCE.call_once(|| {
        tracing::info!("SAMPLER_BATCHED_GUMBEL_VERIFY_ENGAGED rows={rows} seqs={seqs}");
    });
}

/// Strict 100%-lossless eligibility check for pre-sampling all `k_verify`
/// BF16 verify rows of `a` in a single batched GPU Gumbel-Max launch
/// (`Model::sample_gumbel_verify_rows_batched`).
///
/// Any request with active penalties, `min_p`, `top_n_sigma`, `logit_bias`,
/// active grammar, logprobs, adaptive sampling, reflection suppression,
/// unsupported `(top_k, top_p)`, imminent thinking-budget / answer-runway
/// exhaustion, F2 confidence early-stop window, or pending forced tokens is
/// excluded and keeps its existing per-row sampler path.
pub(super) fn batched_gumbel_seq_eligible(
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) -> bool {
    if k_verify == 0 {
        return false;
    }
    if think.adaptive_sampling
        || a.top_logprobs.is_some()
        || a.grammar_state.is_some()
        || super::decode_logits_seq::logits_fnv_requires_row()
        || super::logits_row_receipt::enabled()
        || crate::dsv41::serving()
        || crate::dsv41::repetition::active()
    {
        return false;
    }
    if a.temperature <= 0.0 || !a.temperature.is_finite() {
        return false;
    }
    if a.min_p != 0.0
        || a.top_n_sigma != 0.0
        || a.presence_penalty != 0.0
        || a.frequency_penalty != 0.0
        || a.repetition_penalty != 1.0
        || a.lz_penalty > 0.0
        || a.lz_penalty.is_nan()
        || a.dry_multiplier > 0.0
        || a.dry_multiplier.is_nan()
        || !a.logit_bias.is_empty()
    {
        return false;
    }
    // The GPU kernel treats `top_p <= 0` as disabled whereas the host sampler
    // truncates to 1 token; require strictly positive finite `top_p`, and either
    // `1 <= top_k <= DEVICE_TOPK_K_MAX` or full-vocab with nucleus disabled.
    let top_k_p_ok = a.top_p > 0.0
        && a.top_p.is_finite()
        && ((1..=spark_runtime::sampler::device_topk::DEVICE_TOPK_K_MAX)
            .contains(&(a.top_k as usize))
            || (a.top_k == 0 && a.top_p >= 1.0));
    if !top_k_p_ok {
        return false;
    }
    if super::think_nudge::forced_token(a).is_some()
        || super::think_budget_nudge::forced_token(a).is_some()
    {
        return false;
    }
    if a.inside_thinking {
        if !think.enabled
            || !think.reflection_suppress_ids.is_empty()
            || crate::scheduler::think_efficiency_config().is_active()
            || a.think_ended
            || a.force_end_thinking
            || a.budget_nudge.is_some()
            || super::think_budget_nudge::close_allowed(a)
            || super::think_spec_accept::f2_window_active(a)
        {
            return false;
        }
        let floor = crate::scheduler::helpers::min_thinking_floor();
        if floor > 0 && a.thinking_tokens < floor && a.think_end_token.is_some() {
            return false;
        }
        if let Some(budget) = a.thinking_budget
            && (a.thinking_tokens as usize).saturating_add(k_verify) > budget as usize
        {
            return false;
        }
        if a.thinking_budget.is_none()
            && a.max_output_tokens > 0
            && a.output_tokens.len()
                >= a.max_output_tokens
                    .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens))
        {
            return false;
        }
        if crate::scheduler::helpers::watchdog_params().confidence_early_stop
            && (a.thinking_tokens as usize).saturating_add(k_verify)
                > super::confidence::CONFIDENCE_EARLY_STOP_MIN_THINKING as usize
        {
            return false;
        }
        true
    } else {
        !a.think_just_ended
            && !a.suppress_tool_call
            && !a.require_tool_call
            && !a.inside_tool_body
            && a.output_tokens.len() >= a.min_tokens
    }
}

/// Per-sequence tail mode staged into the unified `target_gumbel_verify_rows_bf16`
/// verify kernel (`temp > 0`: Gumbel-Max, `temp == 0`: 0-ULP greedy argmax,
/// `temp < 0`: instant 0-cycle skip leaving resident BF16 logits untouched).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyTailRowMode {
    /// Exact keyed Gumbel-Max in-graph (`temp > 0.0`).
    Gumbel,
    /// Exact 1-pass 0-ULP greedy argmax in-graph (`temp == 0.0`).
    GreedyArgmax,
    /// Instant 0-cycle GPU skip (`temp = -1.0f32` -> `u32::MAX`), leaving
    /// resident BF16 logits untouched for host-side policy/nudge handling.
    HostPolicySkip,
}

/// Classify how one sequence's verify rows should run in the unified
/// `target_gumbel_verify_rows_bf16` suffix tail.
///
/// `HostPolicySkip` (`temp = -1.0f32`) is emitted ONLY when the sequence is
/// guaranteed to run a policy accept walk over resident BF16 logits and never
/// read raw `verified[i]` argmax token IDs.
pub(super) fn classify_verify_tail_row_mode(
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> VerifyTailRowMode {
    if policy == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && batched_gumbel_seq_eligible(a, k_verify, think)
    {
        return VerifyTailRowMode::Gumbel;
    }
    if !a.inside_thinking || think.enabled {
        let stochastic_policy = a.temperature > 0.0 && a.temperature.is_finite();
        let greedy_non_fast_policy = a.temperature == 0.0
            && (!fast_path_seq_eligible(a) || (!a.inside_thinking && a.grammar_state.is_some()));
        if stochastic_policy || greedy_non_fast_policy {
            return VerifyTailRowMode::HostPolicySkip;
        }
    }
    VerifyTailRowMode::GreedyArgmax
}

/// Append `k_verify` contiguous [`GumbelVerifyRowSpec`] entries starting at
/// `row_base` for sequence `a` according to `mode`.
pub(super) fn push_verify_tail_row_specs(
    reqs: &mut Vec<spark_runtime::sampler::device_topk::GumbelVerifyRowSpec>,
    a: &ActiveSeq,
    row_base: usize,
    k_verify: usize,
    mode: VerifyTailRowMode,
) {
    match mode {
        VerifyTailRowMode::Gumbel => {
            let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
            let base_pos = super::stochastic_verify_walk::seq_abs_pos(a);
            for r in 0..k_verify {
                reqs.push(spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
                    row_idx: row_base + r,
                    seed,
                    abs_pos: base_pos.saturating_add(r as u64),
                    temperature: a.temperature,
                    top_k: a.top_k,
                    top_p: a.top_p,
                });
            }
        }
        VerifyTailRowMode::GreedyArgmax | VerifyTailRowMode::HostPolicySkip => {
            let temperature = if mode == VerifyTailRowMode::GreedyArgmax {
                0.0f32
            } else {
                -1.0f32
            };
            for r in 0..k_verify {
                reqs.push(spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
                    row_idx: row_base + r,
                    seed: 0,
                    abs_pos: 0,
                    temperature,
                    top_k: 1,
                    top_p: 1.0,
                });
            }
        }
    }
}

/// Stage `k_verify` unified verify tail row specs (`row_idx = 0..k_verify`)
/// before a `C = 1` verify forward (`decode_verify_dflash`) so CUDA-graphed
/// verify folds `target_gumbel_verify_rows_bf16` directly into its tail with
/// zero extra GPU launch, host sync, or mid-run graph-key splits.
pub(super) fn stage_single_seq_gumbel_if_eligible(
    model: &dyn Model,
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) {
    stage_single_seq_gumbel_if_eligible_with_policy(
        model,
        a,
        k_verify,
        think,
        super::stochastic_verify_walk::spec_verify_policy(),
    );
}

pub(super) fn stage_single_seq_gumbel_if_eligible_with_policy(
    model: &dyn Model,
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) {
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        || k_verify == 0
        || k_verify > spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS
    {
        return;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return;
    }
    // For C=1 greedy (`T == 0.0`), leave un-staged so `verify_d` auto-stages
    // `temp = 0.0` with `c1_coalesced_lm_head = false` (strict 0-ULP greedy).
    // For C=1 stochastic (`T > 0.0`), stage either `Gumbel` (`temp > 0`) or
    // `HostPolicySkip` (`temp = -1.0`) when a policy walk is guaranteed, keeping
    // the C=1 CUDA graph key constant across budget-nudge / force-end-thinking.
    let mode = if batched_gumbel_seq_eligible(a, k_verify, think) {
        VerifyTailRowMode::Gumbel
    } else if a.temperature > 0.0
        && a.temperature.is_finite()
        && (!a.inside_thinking || think.enabled)
    {
        VerifyTailRowMode::HostPolicySkip
    } else {
        return;
    };
    let mut reqs = Vec::with_capacity(k_verify);
    push_verify_tail_row_specs(&mut reqs, a, 0, k_verify, mode);
    model.stage_next_verify_gumbel_rows(&reqs);
}

/// Pre-sample all `k_verify` BF16 verify rows of a single sequence (`C=1` or
/// non-batched fallback) in ONE GPU Gumbel-Max launch when strictly eligible,
/// or consume in-graph Gumbel tokens directly when already produced by verify.
pub(super) fn presample_single_seq_gumbel(
    model: &dyn Model,
    a: &ActiveSeq,
    row_base: usize,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) -> Result<Option<Vec<u32>>> {
    presample_single_seq_gumbel_with_policy(
        model,
        a,
        row_base,
        k_verify,
        think,
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

pub(super) fn presample_single_seq_gumbel_with_policy(
    model: &dyn Model,
    a: &ActiveSeq,
    row_base: usize,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> Result<Option<Vec<u32>>> {
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel {
        return Ok(None);
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return Ok(None);
    }
    if !batched_gumbel_seq_eligible(a, k_verify, think) {
        return Ok(None);
    }
    if row_base == 0
        && let Some(tokens) = model.take_in_graph_gumbel_verify_tokens()
        && tokens.len() == k_verify
    {
        log_batched_gumbel_engaged_once(k_verify, 1);
        return Ok(Some(tokens));
    }
    let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
    let base_pos = super::stochastic_verify_walk::seq_abs_pos(a);
    let mut reqs = Vec::with_capacity(k_verify);
    for r in 0..k_verify {
        reqs.push(spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
            row_idx: row_base + r,
            seed,
            abs_pos: base_pos.saturating_add(r as u64),
            temperature: a.temperature,
            top_k: a.top_k,
            top_p: a.top_p,
        });
    }
    let vocab = model.vocab_size();
    let Some(tokens) = model.sample_gumbel_verify_rows_batched(logits_base, vocab, &reqs)? else {
        return Ok(None);
    };
    anyhow::ensure!(
        tokens.len() == k_verify,
        "sample_gumbel_verify_rows_batched returned {} tokens for {} rows",
        tokens.len(),
        k_verify
    );
    log_batched_gumbel_engaged_once(k_verify, 1);
    Ok(Some(tokens))
}

/// Per-row dynamic lossless guard for pre-sampled GPU Gumbel-Max tokens in
/// content (non-thinking) spans. Any state transition or masked token
/// invalidates pre-sampled tokens for this row and all remaining rows of the
/// sequence so the walk falls back cleanly to the exact per-row sampler.
pub(super) fn presampled_content_row_ok(
    a: &ActiveSeq,
    tok: u32,
    vocab: usize,
    expected_abs_pos: u64,
    ctx: &ThinkSpecCtx<'_>,
) -> bool {
    if (tok as usize) >= vocab {
        return false;
    }
    if a.inside_thinking
        || a.think_just_ended
        || a.suppress_tool_call
        || a.require_tool_call
        || a.inside_tool_body
        || a.grammar_state.is_some()
        || a.top_logprobs.is_some()
        || ctx.adaptive_sampling
    {
        return false;
    }
    if a.output_tokens.len() < a.min_tokens {
        return false;
    }
    if super::stochastic_verify_walk::seq_abs_pos(a) != expected_abs_pos {
        return false;
    }
    if super::think_nudge::forced_token(a).is_some()
        || super::think_budget_nudge::forced_token(a).is_some()
    {
        return false;
    }
    if a.think_ended
        && !crate::dsv41::serving()
        && (a.think_start_token == Some(tok) || a.think_end_token == Some(tok))
    {
        return false;
    }
    true
}

/// Bind the content-policy walk to the resident BF16 or FP32 verify-logits rows
/// starting at `row_base`.
pub(super) fn run_dflash_content_accept_at_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    run_dflash_content_accept_at_row_with_presampled(
        model, a, row_base, drafts, verified, ctx, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_dflash_content_accept_at_row_with_presampled(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    presampled: Option<&[u32]>,
) -> Option<SpecAcceptOutcome> {
    if verified.is_empty() {
        tracing::error!("spec-policy verify: empty target result; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let logits_base = model.logits_buffer_ptr();
    let logits_fp32 = model.logits_ptr_is_fp32(logits_base);
    let local_presampled = if presampled.is_none() && !logits_fp32 {
        match presample_single_seq_gumbel(model, a, row_base, verified.len(), ctx) {
            Ok(tokens) => tokens,
            Err(error) => {
                fail_speculative_step(a, "batched gumbel content verify", error);
                return Some(SpecAcceptOutcome {
                    num_accepted: 0,
                    bonus: None,
                    consumed_rows: 0,
                });
            }
        }
    } else {
        None
    };
    let effective_presampled = presampled.or(local_presampled.as_deref());
    // The one dispatch point for FLUX_SPEC_VERIFY (default argmax: untouched).
    // When `ATLAS_SAMPLER_DEVICE_TOPK=1|auto` is enabled under `SpecVerifyPolicy::Gumbel`
    // and this sequence is eligible for device top-k, fall through to
    // `dflash_content_accept_routed` so `sample_spec_row` runs the 31-CTA device
    // top-k kernel per row under `GumbelPosGuard` instead of a 2.5 MB full-vocab
    // host logits download. Ineligible requests (`top_k == 0`, oversized `top_k`,
    // etc.) keep the single-bulk-D2H host Gumbel walk.
    let policy = super::stochastic_verify_walk::spec_verify_policy();
    let take_host_stochastic_walk = match policy {
        super::stochastic_verify_walk::SpecVerifyPolicy::Stochastic => true,
        super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel => {
            !super::device_topk_route::seq_uses_device_topk(
                model,
                a,
                logits_fp32,
                ctx.adaptive_sampling,
                ctx.reflection_suppress_ids,
            )
        }
        super::stochastic_verify_walk::SpecVerifyPolicy::Argmax
        | super::stochastic_verify_walk::SpecVerifyPolicy::Rejection => false,
    };
    if effective_presampled.is_none()
        && take_host_stochastic_walk
        && super::stochastic_verify_walk::stochastic_seq_eligible(a, ctx)
    {
        return Some(
            super::stochastic_verify_walk::run_stochastic_content_accept_at_row(
                model,
                a,
                row_base,
                drafts,
                verified.len(),
            ),
        );
    }
    let logits_format = if logits_fp32 {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let mut bulk: Option<Vec<u8>> = None;
    let mut bulk_rows: usize = 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "DFlash sampler-policy filter active (penalties, masks, grammar, adaptive state)"
        );
    });
    let vocab = model.vocab_size();
    let elem_bytes = logits_format.elem_bytes();
    let rows_total = verified.len();
    let row_bytes = vocab * elem_bytes;
    let seq_logits_base = logits_base.offset(row_base * row_bytes);
    let base_abs_pos = super::stochastic_verify_walk::seq_abs_pos(a);
    let mut presampled_active =
        !logits_fp32 && effective_presampled.is_some_and(|p| p.len() >= rows_total);
    let mut download_error = None;
    let outcome = dflash_content_accept_routed(
        a,
        drafts,
        verified,
        ctx,
        logits_format,
        vocab,
        // Lazy bulk D2H fallback: most exact-greedy rows commit `verified[i]`
        // without calling this closure. If one row needs masking or a logits
        // receipt, verify logits are contiguous, so one bulk copy supplies
        // that row and every later fallback without per-row copy launches.
        |i, buf| {
            if bulk.is_none() {
                let mut all = vec![0u8; rows_total * row_bytes];
                match model.copy_logits_to_host(seq_logits_base, &mut all) {
                    Ok(()) => {
                        bulk = Some(all);
                        bulk_rows = rows_total;
                    }
                    Err(e) => {
                        tracing::error!(
                            "spec-policy verify: bulk logits D2H failed ({} rows): {e:#}",
                            rows_total
                        );
                        download_error = Some(e);
                        return false;
                    }
                }
            }
            let Some(all) = bulk.as_ref() else {
                return false;
            };
            if i >= bulk_rows {
                return false;
            }
            let off = i * row_bytes;
            buf.clear();
            buf.extend_from_slice(&all[off..off + row_bytes]);
            true
        },
        |i, a: &mut ActiveSeq| {
            if presampled_active {
                if let Some(tokens) = effective_presampled
                    && let Some(&tok) = tokens.get(i)
                    && presampled_content_row_ok(
                        a,
                        tok,
                        vocab,
                        base_abs_pos.saturating_add(i as u64),
                        ctx,
                    )
                {
                    return super::device_topk_spec::DeviceRow::Sampled((tok, None));
                }
                presampled_active = false;
            }
            let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                && a.temperature > 0.0)
                .then(|| {
                    spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                        super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                        super::stochastic_verify_walk::seq_abs_pos(a),
                    )
                });
            if model.device_topk_supported() {
                super::device_topk_spec::sample_spec_row(
                    model,
                    a,
                    seq_logits_base.offset(i * row_bytes),
                    logits_format.is_fp32(),
                    ctx,
                )
            } else {
                super::device_topk_spec::DeviceRow::Host
            }
        },
    );
    if let Some(error) = download_error {
        fail_speculative_step(a, "content verify logits download", error);
    }
    super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
    Some(outcome)
}

/// Emit the one-time engagement receipt when GPU maximal-coupling rejection
/// verify serves its first verify frame or sequence.
pub(super) fn log_batched_rejection_engaged_once(rows: usize, seqs: usize) {
    static ENGAGED_ONCE: std::sync::Once = std::sync::Once::new();
    ENGAGED_ONCE.call_once(|| {
        tracing::info!("SAMPLER_BATCHED_REJECTION_VERIFY_ENGAGED rows={rows} seqs={seqs}");
    });
}

/// Strict 100%-lossless eligibility check for pre-verifying all `k_verify`
/// BF16 verify rows of `a` via `target_rejection_verify_rows_bf16`.
///
/// Unlike `target_gumbel_verify_rows_bf16`, the rejection kernel natively
/// handles arbitrary `top_k >= 0`, `top_p > 0.0`, `min_p >= 0.0`, and
/// post-thinking `(think_start, think_end)` masking in-graph with zero D2H.
/// Requests with penalties, `top_n_sigma`, `logit_bias`, active grammar,
/// logprobs, adaptive sampling, reflection suppression, imminent thinking
/// budget / runway / F2 exhaustion, or forced tokens are excluded and fall
/// back cleanly to exact host-side sampling.
pub(super) fn batched_rejection_seq_eligible(
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) -> bool {
    if k_verify == 0 {
        return false;
    }
    if think.adaptive_sampling
        || a.top_logprobs.is_some()
        || a.grammar_state.is_some()
        || super::decode_logits_seq::logits_fnv_requires_row()
        || super::logits_row_receipt::enabled()
        || crate::dsv41::serving()
        || crate::dsv41::repetition::active()
    {
        return false;
    }
    if a.temperature <= 0.0 || !a.temperature.is_finite() {
        return false;
    }
    if !a.top_p.is_finite() || a.top_p <= 0.0 || !a.min_p.is_finite() || a.min_p < 0.0 {
        return false;
    }
    if a.top_n_sigma != 0.0
        || a.presence_penalty != 0.0
        || a.frequency_penalty != 0.0
        || a.repetition_penalty != 1.0
        || a.lz_penalty > 0.0
        || a.lz_penalty.is_nan()
        || a.dry_multiplier > 0.0
        || a.dry_multiplier.is_nan()
        || !a.logit_bias.is_empty()
    {
        return false;
    }
    if super::think_nudge::forced_token(a).is_some()
        || super::think_budget_nudge::forced_token(a).is_some()
    {
        return false;
    }
    if a.inside_thinking {
        if !think.enabled
            || !think.reflection_suppress_ids.is_empty()
            || crate::scheduler::think_efficiency_config().is_active()
            || a.think_ended
            || a.force_end_thinking
            || a.budget_nudge.is_some()
            || super::think_budget_nudge::close_allowed(a)
            || super::think_spec_accept::f2_window_active(a)
        {
            return false;
        }
        let floor = crate::scheduler::helpers::min_thinking_floor();
        if floor > 0 && a.thinking_tokens < floor && a.think_end_token.is_some() {
            return false;
        }
        if let Some(budget) = a.thinking_budget
            && (a.thinking_tokens as usize).saturating_add(k_verify) > budget as usize
        {
            return false;
        }
        if a.thinking_budget.is_none()
            && a.max_output_tokens > 0
            && a.output_tokens.len()
                >= a.max_output_tokens
                    .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens))
        {
            return false;
        }
        if crate::scheduler::helpers::watchdog_params().confidence_early_stop
            && (a.thinking_tokens as usize).saturating_add(k_verify)
                > super::confidence::CONFIDENCE_EARLY_STOP_MIN_THINKING as usize
        {
            return false;
        }
        true
    } else {
        !a.think_just_ended
            && !a.suppress_tool_call
            && !a.require_tool_call
            && !a.inside_tool_body
            && a.output_tokens.len() >= a.min_tokens
    }
}

/// Per-sequence tail mode staged into `target_rejection_verify_rows_bf16`
/// (`temp > 0`: maximal-coupling draft / bonus, `temp == 0`: exact 1-pass
/// greedy argmax, `temp < 0`: instant 0-cycle skip emitting
/// [`spark_model::HOST_POLICY_FALLBACK_TOKEN`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RejectionTailRowMode {
    /// Exact GPU maximal-coupling rejection / bonus sampling (`temp > 0.0`).
    Rejection,
    /// Exact 1-pass 0-ULP greedy argmax (`temp == 0.0`).
    GreedyArgmax,
    /// Instant 0-cycle GPU skip (`temp = -1.0f32` -> `HOST_POLICY_FALLBACK_TOKEN`).
    HostPolicySkip,
}

/// Classify how one sequence's verify rows should run in
/// `target_rejection_verify_rows_bf16`.
pub(super) fn classify_rejection_tail_row_mode(
    a: &ActiveSeq,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> RejectionTailRowMode {
    if policy == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        && batched_rejection_seq_eligible(a, k_verify, think)
    {
        return RejectionTailRowMode::Rejection;
    }
    if !a.inside_thinking || think.enabled {
        let stochastic_policy = a.temperature > 0.0 && a.temperature.is_finite();
        let greedy_non_fast_policy = a.temperature == 0.0
            && (!fast_path_seq_eligible(a) || (!a.inside_thinking && a.grammar_state.is_some()));
        if stochastic_policy || greedy_non_fast_policy {
            return RejectionTailRowMode::HostPolicySkip;
        }
    }
    RejectionTailRowMode::GreedyArgmax
}

/// Token-suppression pair `(mask_tok0, mask_tok1)` passed to
/// `target_rejection_verify_rows_bf16`: masks `<think>` and `</think>` in-graph
/// on post-thinking content rows for 100% lossless zero-D2H verification.
#[inline]
#[must_use]
pub(super) fn rejection_row_masks(a: &ActiveSeq) -> (u32, u32) {
    if !a.inside_thinking && a.think_ended && !crate::dsv41::serving() {
        (
            a.think_start_token
                .unwrap_or(spark_model::REJECTION_NO_MASK_TOKEN),
            a.think_end_token
                .unwrap_or(spark_model::REJECTION_NO_MASK_TOKEN),
        )
    } else {
        (
            spark_model::REJECTION_NO_MASK_TOKEN,
            spark_model::REJECTION_NO_MASK_TOKEN,
        )
    }
}

/// Validate a sequence's GPU rejection `q` fence `(q_slot, q_pos, valid_len)`
/// against `(slot_idx, base_abs_pos)` and clamp to `draft_len`.
#[inline]
#[must_use]
pub(super) fn resolve_valid_rejection_q_len(
    q_fence: Option<(usize, u64, usize)>,
    slot_idx: usize,
    base_abs_pos: u64,
    draft_len: usize,
) -> usize {
    match q_fence {
        Some((q_slot, q_pos, valid_len)) if q_slot == slot_idx && q_pos == base_abs_pos => {
            valid_len.min(draft_len)
        }
        _ => 0,
    }
}

/// Append `k_verify` contiguous [`spark_model::RejectionVerifyRowSpec`] entries
/// starting at `row_base` using an explicit `q_fence` (pure testable core).
#[allow(clippy::too_many_arguments)]
pub(super) fn push_rejection_tail_row_specs_with_fence(
    reqs: &mut Vec<spark_model::RejectionVerifyRowSpec>,
    a: &ActiveSeq,
    drafts: &[u32],
    row_base: usize,
    k_verify: usize,
    mode: RejectionTailRowMode,
    q_fence: Option<(usize, u64, usize)>,
) {
    match mode {
        RejectionTailRowMode::Rejection => {
            let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
            let base_pos = super::stochastic_verify_walk::seq_abs_pos(a);
            let (mask_tok0, mask_tok1) = rejection_row_masks(a);
            let valid_q_len =
                resolve_valid_rejection_q_len(q_fence, a.seq.slot_idx, base_pos, drafts.len());
            for r in 0..k_verify {
                let q_idx = a.seq.slot_idx.checked_mul(8).and_then(|b| b.checked_add(r));
                let is_draft = r < drafts.len()
                    && r + 1 < k_verify
                    && r < valid_q_len
                    && q_idx.is_some_and(|idx| idx < spark_model::REJECTION_Q_ROW_CAP);
                let (draft_token, q_row_idx) = if is_draft {
                    (
                        drafts[r],
                        q_idx.expect("is_draft checks q_idx < REJECTION_Q_ROW_CAP") as u32,
                    )
                } else {
                    (0, spark_model::REJECTION_BONUS_Q_ROW_IDX)
                };
                reqs.push(spark_model::RejectionVerifyRowSpec {
                    row_idx: row_base + r,
                    seed,
                    abs_pos: base_pos.saturating_add(r as u64),
                    temperature: a.temperature,
                    top_k: a.top_k,
                    top_p: a.top_p,
                    min_p: a.min_p,
                    draft_token,
                    q_row_idx,
                    mask_tok0,
                    mask_tok1,
                });
            }
        }
        RejectionTailRowMode::GreedyArgmax => {
            for r in 0..k_verify {
                let q_idx = a.seq.slot_idx.checked_mul(8).and_then(|b| b.checked_add(r));
                let is_draft = r < drafts.len()
                    && r + 1 < k_verify
                    && q_idx
                        .is_some_and(|idx| idx < spark_model::REJECTION_BONUS_Q_ROW_IDX as usize);
                let (draft_token, q_row_idx) = if is_draft {
                    (
                        drafts[r],
                        q_idx.expect("is_draft checks q_idx < REJECTION_BONUS_Q_ROW_IDX") as u32,
                    )
                } else {
                    (0, spark_model::REJECTION_BONUS_Q_ROW_IDX)
                };
                reqs.push(spark_model::RejectionVerifyRowSpec {
                    row_idx: row_base + r,
                    seed: 0,
                    abs_pos: 0,
                    temperature: 0.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                    draft_token,
                    q_row_idx,
                    mask_tok0: spark_model::REJECTION_NO_MASK_TOKEN,
                    mask_tok1: spark_model::REJECTION_NO_MASK_TOKEN,
                });
            }
        }
        RejectionTailRowMode::HostPolicySkip => {
            for r in 0..k_verify {
                reqs.push(spark_model::RejectionVerifyRowSpec {
                    row_idx: row_base + r,
                    seed: 0,
                    abs_pos: 0,
                    temperature: -1.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                    draft_token: 0,
                    q_row_idx: spark_model::REJECTION_BONUS_Q_ROW_IDX,
                    mask_tok0: spark_model::REJECTION_NO_MASK_TOKEN,
                    mask_tok1: spark_model::REJECTION_NO_MASK_TOKEN,
                });
            }
        }
    }
}

/// Append `k_verify` contiguous [`spark_model::RejectionVerifyRowSpec`] entries
/// starting at `row_base`, reading `a`'s live `DflashProposerState` `q` fence.
pub(super) fn push_rejection_tail_row_specs(
    reqs: &mut Vec<spark_model::RejectionVerifyRowSpec>,
    a: &ActiveSeq,
    drafts: &[u32],
    row_base: usize,
    k_verify: usize,
    mode: RejectionTailRowMode,
) {
    let q_fence = super::stochastic_verify_walk::seq_rejection_q_fence(a);
    push_rejection_tail_row_specs_with_fence(reqs, a, drafts, row_base, k_verify, mode, q_fence);
}

/// Stage `k_verify` GPU maximal-coupling rejection verify row specs before a
/// `C = 1` verify forward (`decode_verify_dflash`), always clearing stale staged
/// rows with `&[]` when not in `SpecVerifyPolicy::Rejection` or ineligible.
pub(super) fn stage_single_seq_rejection_if_eligible(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) {
    stage_single_seq_rejection_if_eligible_with_policy(
        model,
        a,
        drafts,
        k_verify,
        think,
        super::stochastic_verify_walk::spec_verify_policy(),
    );
}

pub(super) fn stage_single_seq_rejection_if_eligible_with_policy(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) {
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        || !model.gpu_rejection_verify_available()
        || k_verify == 0
        || k_verify > spark_model::REJECTION_MAX_ROWS
    {
        model.stage_next_verify_rejection_rows(&[]);
        return;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        model.stage_next_verify_rejection_rows(&[]);
        return;
    }
    let mode = if batched_rejection_seq_eligible(a, k_verify, think) {
        RejectionTailRowMode::Rejection
    } else if a.temperature > 0.0
        && a.temperature.is_finite()
        && (!a.inside_thinking || think.enabled)
    {
        RejectionTailRowMode::HostPolicySkip
    } else {
        model.stage_next_verify_rejection_rows(&[]);
        return;
    };
    let mut reqs = Vec::with_capacity(k_verify);
    push_rejection_tail_row_specs(&mut reqs, a, drafts, 0, k_verify, mode);
    model.stage_next_verify_rejection_rows(&reqs);
}

/// Consume in-graph [`spark_model::PackedRejectionWord`] results for a single
/// sequence (`C = 1`), or fall back to one batched
/// [`Model::sample_rejection_verify_rows_batched`] launch when eager verify
/// bypassed in-graph capture.
pub(super) fn presample_single_seq_rejection(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    row_base: usize,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
) -> Result<Option<Vec<spark_model::PackedRejectionWord>>> {
    presample_single_seq_rejection_with_policy(
        model,
        a,
        drafts,
        row_base,
        k_verify,
        think,
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn presample_single_seq_rejection_with_policy(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    row_base: usize,
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> Result<Option<Vec<spark_model::PackedRejectionWord>>> {
    // Always drain any in-graph buffer published by the immediately preceding
    // verify forward so packed words can never leak across steps.
    let in_graph_words = model.take_in_graph_rejection_verify_words();
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        || !model.gpu_rejection_verify_available()
    {
        return Ok(None);
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return Ok(None);
    }
    if !batched_rejection_seq_eligible(a, k_verify, think) {
        return Ok(None);
    }
    let vocab = model.vocab_size();
    if row_base == 0
        && let Some(words) = in_graph_words
        && words.len() == k_verify
        && words
            .iter()
            .all(|w| w.is_host_fallback() || (w.token_id() as usize) < vocab)
    {
        log_batched_rejection_engaged_once(k_verify, 1);
        return Ok(Some(words));
    }
    let mut reqs = Vec::with_capacity(k_verify);
    push_rejection_tail_row_specs(
        &mut reqs,
        a,
        drafts,
        row_base,
        k_verify,
        RejectionTailRowMode::Rejection,
    );
    let Some(words) = model.sample_rejection_verify_rows_batched(logits_base, vocab, &reqs)? else {
        return Ok(None);
    };
    anyhow::ensure!(
        words.len() == k_verify,
        "sample_rejection_verify_rows_batched returned {} words for {} rows",
        words.len(),
        k_verify
    );
    for (idx, word) in words.iter().enumerate() {
        anyhow::ensure!(
            word.is_host_fallback() || (word.token_id() as usize) < vocab,
            "sample_rejection_verify_rows_batched returned out-of-vocab token {} (vocab={vocab}) at row {idx}",
            word.token_id()
        );
    }
    log_batched_rejection_engaged_once(k_verify, 1);
    Ok(Some(words))
}

/// Authoritative bit-31 GPU maximal-coupling rejection commit walk for content
/// (non-thinking) spans.
///
/// For each row `i in 0..verified.len()`:
/// - When `presampled_rejection` is active and passes the dynamic row guard
///   (`!word.is_host_fallback()` && [`presampled_content_row_ok`]), bit 31 is
///   authoritative (`accepted_by_kernel = i < drafts.len() && i + 1 < k_verify && word.is_accepted()`),
///   with a fail-closed assertion that `word.token_id() == drafts[i]` whenever
///   accepted.
/// - If the dynamic guard trips or a row carries `HOST_POLICY_FALLBACK_TOKEN`,
///   presampled rejection disables itself for row `i` onward and falls back
///   seamlessly to exact `device_row` / host `process_seq_logits`.
/// - Phase boundaries (`<think>`, `</think>`, EOS, or `inside_thinking` flip)
///   always end the walk as a bonus token (never accepted).
#[allow(clippy::too_many_arguments)]
pub(super) fn dflash_content_accept_rejection_routed(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
    ctx: &ThinkSpecCtx<'_>,
    logits_format: VerifyLogitsFormat,
    vocab: usize,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut device_row: impl FnMut(usize, &mut ActiveSeq) -> super::device_topk_spec::DeviceRow,
) -> Result<SpecAcceptOutcome> {
    debug_assert!(!a.inside_thinking);
    let mut row_buf = Vec::new();
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;
    let elem_bytes = logits_format.elem_bytes();
    let mut masked_fast_engaged_logged = false;
    let logits_fnv_requires_row = super::decode_logits_seq::logits_fnv_requires_row();
    let base_abs_pos = super::stochastic_verify_walk::seq_abs_pos(a);
    let mut presampled_active = !logits_format.is_fp32()
        && presampled_rejection.is_some_and(|words| words.len() >= verified.len());
    let mut skip_storage = [0u32; 2];
    let mut skip_len = 0usize;
    if a.think_ended && !crate::dsv41::serving() {
        if let Some(t) = a.think_start_token {
            skip_storage[skip_len] = t;
            skip_len += 1;
        }
        if let Some(t) = a.think_end_token {
            skip_storage[skip_len] = t;
            skip_len += 1;
        }
    }
    let skip = &skip_storage[..skip_len];

    for i in 0..verified.len() {
        let mut kernel_choice: Option<(u32, bool)> = None;
        if presampled_active {
            if let Some(words) = presampled_rejection
                && let Some(&word) = words.get(i)
                && !word.is_host_fallback()
                && presampled_content_row_ok(
                    a,
                    word.token_id(),
                    vocab,
                    base_abs_pos.saturating_add(i as u64),
                    ctx,
                )
            {
                let tok = word.token_id();
                let accepted_by_kernel =
                    i < drafts.len() && i + 1 < verified.len() && word.is_accepted();
                if accepted_by_kernel {
                    anyhow::ensure!(
                        tok == drafts[i],
                        "rejection kernel accepted bit mismatch: tok={tok} draft={}",
                        drafts[i]
                    );
                }
                kernel_choice = Some((tok, accepted_by_kernel));
            } else {
                presampled_active = false;
            }
        }

        let (target, lp, accepted) = if let Some((tok, accepted_by_kernel)) = kernel_choice {
            (tok, None, accepted_by_kernel)
        } else {
            let fast_path_ok = accept_row_fast_path_ok(a, ctx, logits_format);
            let fast_row = fast_path_ok
                .then(|| verified_argmax_row(verified[i], skip, vocab, logits_fnv_requires_row));
            let (target, lp) = match fast_row {
                Some(VerifiedArgmaxRow::Direct(tok)) => (tok, None),
                Some(VerifiedArgmaxRow::Invalid) => {
                    tracing::error!(
                        row = i,
                        token = verified[i],
                        vocab,
                        "spec-policy rejection fallback: raw argmax outside vocabulary; finishing seq"
                    );
                    a.finished = true;
                    return Ok(SpecAcceptOutcome {
                        num_accepted,
                        bonus: None,
                        consumed_rows,
                    });
                }
                Some(VerifiedArgmaxRow::Fetch) | None => {
                    let routed = if fast_path_ok {
                        super::device_topk_spec::DeviceRow::Host
                    } else {
                        device_row(i, a)
                    };
                    match routed {
                        super::device_topk_spec::DeviceRow::Sampled(sampled) => sampled,
                        super::device_topk_spec::DeviceRow::Failed => {
                            tracing::error!(
                                "spec-policy rejection fallback: device top-k row {i} failed; finishing seq"
                            );
                            a.finished = true;
                            return Ok(SpecAcceptOutcome {
                                num_accepted,
                                bonus: None,
                                consumed_rows,
                            });
                        }
                        super::device_topk_spec::DeviceRow::Host => {
                            if !fetch_row(i, &mut row_buf) {
                                tracing::error!(
                                    "spec-policy rejection fallback: logits row {i} unavailable; finishing seq"
                                );
                                a.finished = true;
                                return Ok(SpecAcceptOutcome {
                                    num_accepted,
                                    bonus: None,
                                    consumed_rows,
                                });
                            }
                            if row_buf.len() != vocab * elem_bytes {
                                tracing::error!(
                                    row = i,
                                    bytes = row_buf.len(),
                                    expected_bytes = vocab * elem_bytes,
                                    elem_bytes,
                                    "spec-policy rejection fallback: malformed logits row; finishing seq"
                                );
                                a.finished = true;
                                return Ok(SpecAcceptOutcome {
                                    num_accepted,
                                    bonus: None,
                                    consumed_rows,
                                });
                            }
                            if fast_path_ok {
                                if !masked_fast_engaged_logged {
                                    masked_fast_engaged_logged = true;
                                    tracing::info!(
                                        "spec-policy accept: raw-BF16 masked-argmax row fallback engaged"
                                    );
                                }
                                super::decode_logits_seq::log_logits_fnv(
                                    a.session_hash,
                                    a.output_tokens.len(),
                                    &row_buf,
                                );
                                (argmax_bf16_skip(&row_buf, skip, vocab).unwrap_or(0), None)
                            } else {
                                process_seq_logits(
                                    a,
                                    &row_buf,
                                    0,
                                    vocab,
                                    elem_bytes,
                                    logits_format.is_fp32(),
                                    a.think_end_token,
                                    a.think_start_token,
                                    a.tool_call_start_token,
                                    a.tool_call_end_token,
                                    ctx.reflection_suppress_ids,
                                    ctx.adaptive_sampling,
                                )
                            }
                        }
                    }
                }
            };
            let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
            (target, lp, draft_match)
        };

        let was_inside_thinking = a.inside_thinking;
        let receipt = super::logits_row_receipt::current_verify_row(a, i, target);
        if !emit_token_for_verify(a, target, lp) {
            return Ok(SpecAcceptOutcome {
                num_accepted,
                bonus: None,
                consumed_rows,
            });
        }
        if let Some(receipt) = receipt {
            super::logits_row_receipt::output_map(&receipt, &a.output_tokens);
        }
        consumed_rows += 1;
        let phase_boundary = was_inside_thinking != a.inside_thinking
            || a.think_start_token == Some(target)
            || a.think_end_token == Some(target)
            || a.eos_tokens.contains(&target);
        if phase_boundary || !accepted {
            if !a.finished {
                a.last_token = target;
            }
            return Ok(SpecAcceptOutcome {
                num_accepted,
                bonus: Some(target),
                consumed_rows,
            });
        }
        if a.finished {
            return Ok(SpecAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
                consumed_rows,
            });
        }
        num_accepted += 1;
    }

    debug_assert!(verified.is_empty());
    a.finished = true;
    Ok(SpecAcceptOutcome {
        num_accepted,
        bonus: None,
        consumed_rows,
    })
}

/// Authoritative bit-31 GPU maximal-coupling rejection commit walk for
/// `<think>` spans.
///
/// Mirrors [`super::think_spec_accept::dflash_thinking_accept_routed`] while
/// treating bit 31 of `presampled_rejection` as authoritative and failing closed
/// on any `accepted && tok != drafts[i]` mismatch. Mid-walk dynamic policy
/// transitions (budget nudge, runway exhaustion, F2 window, `<tool_call>` mask,
/// or `HOST_POLICY_FALLBACK_TOKEN`) fall back seamlessly to exact host
/// `process_seq_logits` / `deterministic_forced` for row `i` onward.
#[allow(clippy::too_many_arguments)]
pub(super) fn dflash_thinking_rejection_accept_routed(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
    ctx: &ThinkSpecCtx<'_>,
    vocab: usize,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut device_row: impl FnMut(usize, &mut ActiveSeq) -> super::device_topk_spec::DeviceRow,
    mut on_think_end: impl FnMut(&mut ActiveSeq),
) -> Result<SpecAcceptOutcome> {
    debug_assert!(a.inside_thinking);
    let seq_fast =
        fast_path_seq_eligible(a) && !ctx.adaptive_sampling && !crate::dsv41::repetition::active();
    let wave_active = think_efficiency_config().is_active();
    let logits_fnv_requires_row = super::decode_logits_seq::logits_fnv_requires_row();
    let base_abs_pos = super::stochastic_verify_walk::seq_abs_pos(a);
    let mut presampled_active =
        presampled_rejection.is_some_and(|words| words.len() >= verified.len());
    let mut row_buf: Vec<u8> = Vec::new();
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;

    for i in 0..verified.len() {
        let raw = verified[i];
        let min_runway = super::confidence::min_answer_runway(a.max_output_tokens);
        let runway_exhausted = a.max_output_tokens > 0
            && a.output_tokens.len() >= a.max_output_tokens.saturating_sub(min_runway);
        if a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_budget.is_none()
            && runway_exhausted
        {
            a.force_end_thinking = true;
        }

        let mut kernel_choice: Option<(u32, bool)> = None;
        if presampled_active {
            if let Some(words) = presampled_rejection
                && let Some(&word) = words.get(i)
                && !word.is_host_fallback()
                && a.top_logprobs.is_none()
                && super::think_spec_accept::presampled_thinking_row_ok(
                    a,
                    word.token_id(),
                    vocab,
                    base_abs_pos.saturating_add(i as u64),
                    ctx,
                )
            {
                let tok = word.token_id();
                let accepted_by_kernel =
                    i < drafts.len() && i + 1 < verified.len() && word.is_accepted();
                if accepted_by_kernel {
                    anyhow::ensure!(
                        tok == drafts[i],
                        "rejection kernel accepted bit mismatch: tok={tok} draft={}",
                        drafts[i]
                    );
                }
                kernel_choice = Some((tok, accepted_by_kernel));
            } else {
                presampled_active = false;
            }
        }

        let (target, logprobs, accepted) = if let Some((tok, accepted_by_kernel)) = kernel_choice {
            (tok, None, accepted_by_kernel)
        } else {
            let fast = seq_fast
                && a.thinking_tokens >= crate::scheduler::helpers::min_thinking_floor()
                && super::think_spec_accept::position_fast_path_ok(
                    raw,
                    ctx.reflection_suppress_ids,
                    a.tool_call_start_token,
                    wave_active,
                    super::think_spec_accept::f2_window_active(a),
                    logits_fnv_requires_row,
                );
            let deterministic_forced = if !logits_fnv_requires_row && a.top_logprobs.is_none() {
                match (
                    super::think_budget_nudge::forced_token(a),
                    super::think_budget_nudge::close_allowed(a),
                    a.think_end_token,
                ) {
                    (Some(nudge), _, _) => Some(nudge),
                    (None, true, Some(end)) => Some(end),
                    _ => super::think_nudge::forced_token(a),
                }
            } else {
                None
            };
            let (target, logprobs) = if let Some(forced_tok) = deterministic_forced {
                (forced_tok, None)
            } else if fast {
                (raw, None)
            } else {
                match device_row(i, a) {
                    super::device_topk_spec::DeviceRow::Sampled(sampled) => sampled,
                    super::device_topk_spec::DeviceRow::Failed => {
                        tracing::error!(
                            "think-spec rejection fallback: device top-k row {i} failed; finishing seq"
                        );
                        a.finished = true;
                        return Ok(SpecAcceptOutcome {
                            num_accepted,
                            bonus: None,
                            consumed_rows,
                        });
                    }
                    super::device_topk_spec::DeviceRow::Host => {
                        if !fetch_row(i, &mut row_buf) {
                            tracing::error!(
                                "think-spec rejection fallback: logits row {i} unavailable; finishing seq"
                            );
                            a.finished = true;
                            return Ok(SpecAcceptOutcome {
                                num_accepted,
                                bonus: None,
                                consumed_rows,
                            });
                        }
                        if row_buf.len() != vocab * 2 {
                            tracing::error!(
                                row = i,
                                bytes = row_buf.len(),
                                expected_bytes = vocab * 2,
                                "think-spec rejection fallback: malformed logits row; finishing seq"
                            );
                            a.finished = true;
                            return Ok(SpecAcceptOutcome {
                                num_accepted,
                                bonus: None,
                                consumed_rows,
                            });
                        }
                        process_seq_logits(
                            a,
                            &row_buf,
                            0,
                            vocab,
                            2,
                            false,
                            a.think_end_token,
                            a.think_start_token,
                            a.tool_call_start_token,
                            a.tool_call_end_token,
                            ctx.reflection_suppress_ids,
                            ctx.adaptive_sampling,
                        )
                    }
                }
            };
            let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
            (target, logprobs, draft_match)
        };

        let boundary = a.think_end_token == Some(target)
            || a.think_start_token == Some(target)
            || a.eos_tokens.contains(&target);
        if !super::think_spec_accept::commit_thinking_token(
            a,
            target,
            logprobs,
            ctx.code_fence_token,
            ctx.resolved_honor_eos_inside_thinking(),
            &mut on_think_end,
        ) {
            return Ok(SpecAcceptOutcome {
                num_accepted,
                bonus: None,
                consumed_rows,
            });
        }
        consumed_rows += 1;
        if boundary || !accepted {
            if !a.finished {
                a.last_token = target;
            }
            return Ok(SpecAcceptOutcome {
                num_accepted,
                bonus: Some(target),
                consumed_rows,
            });
        }
        if a.finished {
            return Ok(SpecAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
                consumed_rows,
            });
        }
        num_accepted += 1;
    }

    debug_assert!(
        verified.is_empty(),
        "think-spec rejection walk fell off a non-empty verified loop"
    );
    a.finished = true;
    Ok(SpecAcceptOutcome {
        num_accepted,
        bonus: None,
        consumed_rows,
    })
}

/// Bind the content rejection commit walk to resident BF16 verify-logits rows
/// starting at `row_base`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dflash_content_accept_at_row_with_rejection(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
) -> Option<SpecAcceptOutcome> {
    if verified.is_empty() {
        tracing::error!("spec-policy rejection verify: empty target result; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let logits_base = model.logits_buffer_ptr();
    let logits_fp32 = model.logits_ptr_is_fp32(logits_base);
    let logits_format = if logits_fp32 {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let vocab = model.vocab_size();
    let elem_bytes = logits_format.elem_bytes();
    let rows_total = verified.len();
    let row_bytes = vocab * elem_bytes;
    let seq_logits_base = logits_base.offset(row_base * row_bytes);
    let mut bulk: Option<Vec<u8>> = None;
    let mut bulk_rows: usize = 0;
    let mut download_error = None;
    let result = dflash_content_accept_rejection_routed(
        a,
        drafts,
        verified,
        presampled_rejection,
        ctx,
        logits_format,
        vocab,
        |i, buf| {
            if bulk.is_none() {
                let mut all = vec![0u8; rows_total * row_bytes];
                match model.copy_logits_to_host(seq_logits_base, &mut all) {
                    Ok(()) => {
                        bulk = Some(all);
                        bulk_rows = rows_total;
                    }
                    Err(e) => {
                        tracing::error!(
                            "spec-policy rejection verify: bulk logits D2H failed ({rows_total} rows): {e:#}"
                        );
                        download_error = Some(e);
                        return false;
                    }
                }
            }
            let Some(all) = bulk.as_ref() else {
                return false;
            };
            if i >= bulk_rows {
                return false;
            }
            let off = i * row_bytes;
            buf.clear();
            buf.extend_from_slice(&all[off..off + row_bytes]);
            true
        },
        |i, a: &mut ActiveSeq| {
            if model.device_topk_supported() {
                super::device_topk_spec::sample_spec_row(
                    model,
                    a,
                    seq_logits_base.offset(i * row_bytes),
                    logits_format.is_fp32(),
                    ctx,
                )
            } else {
                super::device_topk_spec::DeviceRow::Host
            }
        },
    );
    match result {
        Ok(outcome) => {
            if let Some(error) = download_error {
                fail_speculative_step(a, "content rejection verify logits download", error);
            }
            super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
            Some(outcome)
        }
        Err(error) => {
            fail_speculative_step(a, "content rejection verify accept", error);
            Some(SpecAcceptOutcome {
                num_accepted: 0,
                bonus: None,
                consumed_rows: 0,
            })
        }
    }
}

/// Frame-scoped content rejection commit wrapper (mirrors
/// [`run_dflash_content_accept_for_frame`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dflash_content_accept_for_frame_with_rejection(
    model: &dyn Model,
    a: &mut ActiveSeq,
    inputs: &[u32],
    pre_verify_len: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
) -> Option<SpecAcceptOutcome> {
    let _receipt_guard =
        super::logits_row_receipt::enter_verify_frame(model, a, inputs, pre_verify_len);
    run_dflash_content_accept_at_row_with_rejection(
        model,
        a,
        0,
        drafts,
        verified,
        ctx,
        presampled_rejection,
    )
}

/// Bind the thinking rejection commit walk to resident BF16 verify-logits rows
/// starting at `row_base`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dflash_thinking_rejection_accept_at_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
) -> Option<SpecAcceptOutcome> {
    if verified.is_empty() {
        tracing::error!("think-spec rejection verify: empty target result; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        tracing::error!(
            "think-spec rejection verify: FP32 verify logits despite BF16 gate; finishing seq"
        );
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let vocab = model.vocab_size();
    let mut download_error = None;
    let result = dflash_thinking_rejection_accept_routed(
        a,
        drafts,
        verified,
        presampled_rejection,
        ctx,
        vocab,
        |i, buf| {
            buf.resize(vocab * 2, 0);
            match model.copy_logits_to_host(logits_base.offset((row_base + i) * vocab * 2), buf) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!(
                        "think-spec rejection verify: logits D2H failed at row {i}: {e:#}"
                    );
                    download_error = Some(e);
                    false
                }
            }
        },
        |i, a: &mut ActiveSeq| {
            if model.device_topk_supported() {
                super::device_topk_spec::sample_spec_row(
                    model,
                    a,
                    logits_base.offset((row_base + i) * vocab * 2),
                    false,
                    ctx,
                )
            } else {
                super::device_topk_spec::DeviceRow::Host
            }
        },
        // Defer the SSM boundary snapshot until the caller commits the accepted
        // prefix and drains `sync_secondary()` (checked via `outcome_ended_thinking`):
        // live recurrent state still holds the over-extended verify tail here.
        |_| {},
    );
    match result {
        Ok(outcome) => {
            if let Some(error) = download_error {
                fail_speculative_step(a, "thinking rejection verify logits download", error);
            }
            super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
            Some(outcome)
        }
        Err(error) => {
            fail_speculative_step(a, "thinking rejection verify accept", error);
            Some(SpecAcceptOutcome {
                num_accepted: 0,
                bonus: None,
                consumed_rows: 0,
            })
        }
    }
}

#[cfg(test)]
mod fast_argmax_tests {
    use super::{VerifiedArgmaxRow, argmax_bf16_skip, verified_argmax_row};

    fn bf16(f: f32) -> u16 {
        // Round-to-nearest-even BF16 conversion: keep the top 16 bits and
        // round based on the low 16 bits of the f32 mantissa.
        let bits = f.to_bits();
        let lsb = (bits >> 16) & 1;
        let round = ((bits & 0xFFFF) > 0x8000) || ((bits & 0xFFFF) == 0x8000 && lsb == 1);
        ((bits >> 16) + round as u32) as u16
    }

    /// Reference: exactly what `process_seq_logits`'s greedy branch does —
    /// expand BF16→FP32, apply −inf masks, first-wins argmax.
    fn ref_walk(row: &[u8], skip: &[u32], vocab: usize) -> u32 {
        let f32s: Vec<f32> = (0..vocab)
            .map(|j| {
                let lo = row[2 * j];
                let hi = row[2 * j + 1];
                let u = u16::from_le_bytes([lo, hi]);
                let f = f32::from_bits((u as u32) << 16);
                if skip.contains(&(j as u32)) {
                    f32::NEG_INFINITY
                } else {
                    f
                }
            })
            .collect();
        spark_runtime::sampler::argmax_first_wins_f32(&f32s)
    }

    #[test]
    fn exact_tie_keeps_the_first_id_like_the_device_argmax() {
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [1.0f32, -2.0, 3.0, 3.0, 0.5, -1.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[], 6), Some(2));
    }

    #[test]
    fn verified_device_tie_winner_needs_no_host_row() {
        assert_eq!(
            verified_argmax_row(2, &[], 6, false),
            VerifiedArgmaxRow::Direct(2)
        );
    }

    #[test]
    fn masked_verified_winner_and_fnv_receipt_require_the_host_row() {
        assert_eq!(
            verified_argmax_row(4, &[4, 5], 6, false),
            VerifiedArgmaxRow::Fetch
        );
        assert_eq!(
            verified_argmax_row(2, &[4, 5], 6, true),
            VerifiedArgmaxRow::Fetch
        );
    }

    #[test]
    fn out_of_vocab_verified_winner_fails_closed() {
        assert_eq!(
            verified_argmax_row(6, &[], 6, false),
            VerifiedArgmaxRow::Invalid
        );
    }

    #[test]
    fn all_negative_row_orders_numerically() {
        let mut row = vec![0u8; 4 * 2];
        for (i, v) in [-8.0f32, -1.0, -2.0, -8.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[], 4), Some(1));
    }

    #[test]
    fn signed_zeros_tie_and_the_first_wins() {
        let mut row = vec![0u8; 4 * 2];
        for (i, v) in [-8.0f32, -0.0, 0.0, -8.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[], 4), Some(1));
    }

    #[test]
    fn negative_and_positive_values_ordered_correctly() {
        let mut row = vec![0u8; 6 * 2];
        // Largest magnitude positive must win; the most-negative value must
        // never be confused with a large unsigned pattern.
        for (i, v) in [0.0f32, -100.0, 5.5, -0.25, 1e3, -1e3].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[], 6), Some(4));
        assert_eq!(ref_walk(&row, &[], 6), 4);
    }

    #[test]
    fn masked_tokens_never_win() {
        // Token 2 is the raw max but is masked (think token); the runner-up
        // must win, identically in both paths.
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [1.0f32, 2.0, 9.0, 3.0, 0.5, 4.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[2], 6), Some(5));
        assert_eq!(ref_walk(&row, &[2], 6), 5);
    }

    #[test]
    fn masked_tie_breaks_toward_first_unmasked() {
        let mut row = vec![0u8; 6 * 2];
        for (i, v) in [7.0f32, 7.0, 7.0, 1.0, 2.0, 3.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        // Mask the first 7.0: the next 7.0 (index 1) wins in both paths.
        assert_eq!(argmax_bf16_skip(&row, &[0], 6), Some(1));
        assert_eq!(ref_walk(&row, &[0], 6), 1);
    }

    #[test]
    fn all_tokens_masked_falls_back_to_zero() {
        let mut row = vec![0u8; 3 * 2];
        for (i, v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            row[2 * i..2 * i + 2].copy_from_slice(&bf16(*v).to_le_bytes());
        }
        assert_eq!(argmax_bf16_skip(&row, &[0, 1, 2], 3), None);
    }

    #[test]
    fn random_rows_match_reference() {
        // Deterministic pseudo-random sweep: many rows, various skip sets.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as f32 / u64::MAX as f32
        };
        for trial in 0..500 {
            let vocab = 32 + (trial % 200);
            let mut row = vec![0u8; vocab * 2];
            for j in 0..vocab {
                // Mix of signs and magnitudes; includes exact ties via the
                // quantized pool.
                let pool = [(next() * 200.0 - 100.0), ((next() * 20.0) as i32 as f32)];
                let v = if trial % 3 == 0 { pool[0] } else { pool[1] };
                row[2 * j..2 * j + 2].copy_from_slice(&bf16(v).to_le_bytes());
            }
            let skip: Vec<u32> = (0..vocab)
                .filter(|_| next() < 0.1)
                .map(|j| j as u32)
                .collect();
            let fast = argmax_bf16_skip(&row, &skip, vocab);
            let reference = ref_walk(&row, &skip, vocab);
            if skip.len() == vocab {
                assert_eq!(fast, None, "trial {trial}: all-skipped must be None");
            } else {
                assert_eq!(fast, Some(reference), "trial {trial}: fast path diverged");
            }
        }
    }
}

#[cfg(test)]
mod rejection_verify_tests {
    use super::{
        RejectionTailRowMode, VerifyLogitsFormat, batched_rejection_seq_eligible,
        classify_rejection_tail_row_mode, dflash_content_accept_rejection_routed,
        dflash_thinking_rejection_accept_routed, push_rejection_tail_row_specs_with_fence,
        resolve_valid_rejection_q_len,
    };
    use crate::scheduler::device_topk_spec::DeviceRow;
    use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;
    use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
    use spark_model::{
        HOST_POLICY_FALLBACK_TOKEN, PackedRejectionWord, REJECTION_BONUS_Q_ROW_IDX,
        REJECTION_NO_MASK_TOKEN, REJECTION_Q_ROW_CAP,
    };

    const VOCAB: usize = 128;
    const THINK_START: u32 = 4;
    const THINK_END: u32 = 5;
    const TOOL_CALL: u32 = 6;
    const EOS_ID: u32 = 127;

    fn packed_accept(tok: u32) -> PackedRejectionWord {
        PackedRejectionWord((1u32 << 31) | tok)
    }

    fn packed_reject(tok: u32) -> PackedRejectionWord {
        PackedRejectionWord(tok & 0x7FFF_FFFF)
    }

    fn packed_fallback() -> PackedRejectionWord {
        PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN)
    }

    fn think_ctx(enabled: bool) -> ThinkSpecCtx<'static> {
        ThinkSpecCtx {
            enabled,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        }
    }

    fn make_seq(slot: usize, inside_thinking: bool) -> ActiveSeq {
        let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
        a.seq.prompt_len = 8;
        a.seq.seq_len = 8;
        a.output_tokens.clear();
        a.temperature = 0.8;
        a.top_k = 20;
        a.top_p = 0.95;
        a.min_p = 0.05;
        a.top_n_sigma = 0.0;
        a.repetition_penalty = 1.0;
        a.presence_penalty = 0.0;
        a.frequency_penalty = 0.0;
        a.lz_penalty = 0.0;
        a.dry_multiplier = 0.0;
        a.logit_bias.clear();
        a.seed = Some(0x5151_5151);
        a.max_output_tokens = 200;
        a.remaining = 200;
        a.eos_tokens = vec![EOS_ID];
        a.inside_thinking = inside_thinking;
        a.enable_thinking = inside_thinking;
        a.think_ended = !inside_thinking;
        a.think_just_ended = false;
        a.think_start_token = Some(THINK_START);
        a.think_end_token = Some(THINK_END);
        a.tool_call_start_token = Some(TOOL_CALL);
        a
    }

    fn bf16_row_peak(peak_tok: u32) -> Vec<u8> {
        let mut f32s = vec![-20.0f32; VOCAB];
        f32s[peak_tok as usize] = 40.0;
        f32s.into_iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn bf16_row_with_masked_leader(masked: u32, runner_up: u32) -> Vec<u8> {
        let mut f32s = vec![-20.0f32; VOCAB];
        f32s[masked as usize] = 50.0;
        f32s[runner_up as usize] = 40.0;
        f32s.into_iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn content_rejection_walk_honors_bit31_accept_reject_residual_and_bonus() {
        let ctx = think_ctx(true);

        // 1. Accept 2 drafts + residual reject at row 2 (zero logits fetches).
        {
            let mut a = make_seq(0, false);
            let drafts = [10u32, 11, 12, 13];
            let verified = [99u32; 5];
            let words = [
                packed_accept(10),
                packed_accept(11),
                packed_reject(42), // residual sample != draft[2]
                packed_accept(13),
                packed_reject(55),
            ];
            let mut fetches = 0usize;
            let out = dflash_content_accept_rejection_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VerifyLogitsFormat::Bf16,
                VOCAB,
                |_, _| {
                    fetches += 1;
                    false
                },
                |_, _| DeviceRow::Host,
            )
            .expect("valid rejection words must succeed");
            assert_eq!(fetches, 0, "accepted/rejected kernel rows need zero D2H");
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (2, Some(42), 3)
            );
            assert_eq!(a.output_tokens, vec![10, 11, 42]);
            assert_eq!(a.last_token, 42);
        }

        // 2. Bit 31 == 0 is authoritative even if payload token happens to equal drafts[i]
        //    (e.g. bonus/unpaired pure target draw after q-fence truncation).
        {
            let mut a = make_seq(0, false);
            let drafts = [10u32, 11];
            let verified = [99u32; 3];
            let words = [packed_accept(10), packed_reject(11), packed_reject(77)];
            let out = dflash_content_accept_rejection_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VerifyLogitsFormat::Bf16,
                VOCAB,
                |_, _| false,
                |_, _| DeviceRow::Host,
            )
            .expect("bit-31=0 row must cleanly end walk as bonus");
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (1, Some(11), 2)
            );
            assert_eq!(a.output_tokens, vec![10, 11]);
            assert_eq!(a.last_token, 11);
        }

        // 3. Full accept of all drafts + final bonus row (even if bonus word had bit 31 set,
        //    `i + 1 < k_verify` ensures bonus row is never treated as an accepted draft).
        {
            let mut a = make_seq(0, false);
            let drafts = [10u32, 11];
            let verified = [99u32; 3];
            let words = [packed_accept(10), packed_accept(11), packed_reject(88)];
            let out = dflash_content_accept_rejection_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VerifyLogitsFormat::Bf16,
                VOCAB,
                |_, _| false,
                |_, _| DeviceRow::Host,
            )
            .expect("full accept + bonus must succeed");
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (2, Some(88), 3)
            );
            assert_eq!(a.output_tokens, vec![10, 11, 88]);
        }

        // 4. Phase boundary (EOS) ALWAYS ends walk as bonus, never accepted.
        {
            let mut a = make_seq(0, false);
            let drafts = [10u32, EOS_ID, 12];
            let verified = [99u32; 4];
            let words = [
                packed_accept(10),
                packed_accept(EOS_ID),
                packed_accept(12),
                packed_reject(15),
            ];
            let out = dflash_content_accept_rejection_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VerifyLogitsFormat::Bf16,
                VOCAB,
                |_, _| false,
                |_, _| DeviceRow::Host,
            )
            .expect("EOS phase boundary must end cleanly as bonus");
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (1, Some(EOS_ID), 2)
            );
            assert!(a.finished);
        }
    }

    #[test]
    fn rejection_walk_fails_closed_on_accepted_bit_draft_token_mismatch() {
        let ctx = think_ctx(true);

        // Content mismatch: bit 31 set, but payload 77 != drafts[1] == 11.
        {
            let mut a = make_seq(0, false);
            let err = dflash_content_accept_rejection_routed(
                &mut a,
                &[10, 11],
                &[99, 99, 99],
                Some(&[packed_accept(10), packed_accept(77), packed_reject(20)]),
                &ctx,
                VerifyLogitsFormat::Bf16,
                VOCAB,
                |_, _| false,
                |_, _| DeviceRow::Host,
            )
            .err()
            .expect("corrupt content bit-31 mismatch must fail closed");
            assert!(
                err.to_string()
                    .contains("rejection kernel accepted bit mismatch"),
                "unexpected error: {err:#}"
            );
        }

        // Thinking mismatch: bit 31 set, but payload 66 != drafts[0] == 10.
        {
            let mut a = make_seq(0, true);
            let err = dflash_thinking_rejection_accept_routed(
                &mut a,
                &[10, 11],
                &[99, 99, 99],
                Some(&[packed_accept(66), packed_accept(11), packed_reject(20)]),
                &ctx,
                VOCAB,
                |_, _| false,
                |_, _| DeviceRow::Host,
                |_| {},
            )
            .err()
            .expect("corrupt thinking bit-31 mismatch must fail closed");
            assert!(
                err.to_string()
                    .contains("rejection kernel accepted bit mismatch"),
                "unexpected error: {err:#}"
            );
        }
    }

    #[test]
    fn thinking_rejection_walk_handles_think_end_boundary_and_midwalk_host_fallback() {
        let ctx = think_ctx(true);

        // 1. `</think>` with bit 31 set STILL ends walk as bonus (never accepted), 0 D2H.
        {
            let mut a = make_seq(0, true);
            let drafts = [10u32, THINK_END, 25];
            let verified = [99u32; 4];
            let words = [
                packed_accept(10),
                packed_accept(THINK_END),
                packed_accept(25),
                packed_reject(30),
            ];
            let mut fetches = 0usize;
            let mut think_ended_cb = 0usize;
            let out = dflash_thinking_rejection_accept_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VOCAB,
                |_, _| {
                    fetches += 1;
                    false
                },
                |_, _| DeviceRow::Host,
                |_| think_ended_cb += 1,
            )
            .expect("</think> boundary must succeed");
            assert_eq!(fetches, 0);
            assert_eq!(think_ended_cb, 1);
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (1, Some(THINK_END), 2)
            );
            assert!(!a.inside_thinking);
            assert!(a.think_ended);
            assert_eq!(a.output_tokens, vec![10, THINK_END]);
            assert_eq!(a.last_token, THINK_END);
        }

        // 2. Mid-walk HOST_POLICY_FALLBACK_TOKEN at row 1 seamlessly falls back to host
        //    `process_seq_logits` for row 1 onward while keeping row 0's kernel accept.
        {
            let mut a = make_seq(0, true);
            a.top_k = 1; // deterministic peak selection on host fallback
            let drafts = [10u32, 33, 35];
            let verified = [99u32; 4];
            let words = [
                packed_accept(10),
                packed_fallback(),
                packed_accept(99), // must be ignored once presampled_active flips false!
                packed_reject(99),
            ];
            let rows = [
                bf16_row_peak(10),
                bf16_row_peak(33), // matches draft[1] on host -> continues to row 2 on host!
                bf16_row_peak(44), // diverges from draft[2]=35 on host -> bonus 44!
                bf16_row_peak(55),
            ];
            let mut fetched_rows = Vec::new();
            let out = dflash_thinking_rejection_accept_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VOCAB,
                |i, buf| {
                    fetched_rows.push(i);
                    buf.clear();
                    buf.extend_from_slice(&rows[i]);
                    true
                },
                |_, _| DeviceRow::Host,
                |_| {},
            )
            .expect("mid-walk host fallback must succeed");
            assert_eq!(
                fetched_rows,
                vec![1, 2],
                "row 0 used kernel word; rows 1..2 fell back to exact host logits"
            );
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (2, Some(44), 3)
            );
            assert_eq!(a.output_tokens, vec![10, 33, 44]);
        }

        // 3. Masked `<tool_call>` returned by kernel inside thinking trips dynamic guard
        //    at row 1 and falls back cleanly to host `process_seq_logits` (which masks it).
        {
            let mut a = make_seq(0, true);
            a.top_k = 1;
            let drafts = [10u32, 15];
            let verified = [99u32; 3];
            let words = [
                packed_accept(10),
                packed_reject(TOOL_CALL),
                packed_reject(20),
            ];
            let rows = [
                bf16_row_with_masked_leader(TOOL_CALL, 31),
                bf16_row_with_masked_leader(TOOL_CALL, 37),
                bf16_row_with_masked_leader(TOOL_CALL, 39),
            ];
            let mut fetched_rows = Vec::new();
            let out = dflash_thinking_rejection_accept_routed(
                &mut a,
                &drafts,
                &verified,
                Some(&words),
                &ctx,
                VOCAB,
                |i, buf| {
                    fetched_rows.push(i);
                    buf.clear();
                    buf.extend_from_slice(&rows[i]);
                    true
                },
                |_, _| DeviceRow::Host,
                |_| {},
            )
            .expect("tool_call guard trip must fall back to host");
            assert_eq!(fetched_rows, vec![1]);
            assert_eq!(
                (out.num_accepted, out.bonus, out.consumed_rows),
                (1, Some(37), 2)
            );
            assert_eq!(a.output_tokens, vec![10, 37]);
        }
    }

    #[test]
    fn content_rejection_walk_falls_back_midwalk_on_host_fallback_word_or_masked_think_tag() {
        let ctx = think_ctx(true);
        let mut a = make_seq(0, false);
        a.top_k = 1; // deterministic peak selection on host fallback
        let drafts = [10u32, 20];
        let verified = [99u32; 3];
        // Row 0 accepts via kernel; row 1 trips `presampled_content_row_ok` on `THINK_END`
        // (post-thinking content), falling back seamlessly to host `process_seq_logits`.
        let words = [
            packed_accept(10),
            packed_reject(THINK_END),
            packed_reject(99),
        ];
        let rows = [
            bf16_row_with_masked_leader(THINK_END, 10),
            bf16_row_with_masked_leader(THINK_END, 51),
            bf16_row_with_masked_leader(THINK_END, 52),
        ];
        let mut fetched_rows = Vec::new();
        let out = dflash_content_accept_rejection_routed(
            &mut a,
            &drafts,
            &verified,
            Some(&words),
            &ctx,
            VerifyLogitsFormat::Bf16,
            VOCAB,
            |i, buf| {
                fetched_rows.push(i);
                buf.clear();
                buf.extend_from_slice(&rows[i]);
                true
            },
            |_, _| DeviceRow::Host,
        )
        .expect("post-think masked token guard trip must fall back to host");
        assert_eq!(fetched_rows, vec![1]);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (1, Some(51), 2)
        );
        assert_eq!(a.output_tokens, vec![10, 51]);
    }

    #[test]
    fn rejection_row_dispatch_matrix_enforces_fence_masks_greedy_and_host_fallback() {
        let ctx = think_ctx(true);

        // 1. Eligible post-thinking content seq (`slot=3`, `abs_pos=8`, `drafts=[10,11,12]`,
        //    truncated `valid_len=2` fence):
        //    - r=0,1: draft rows with `q_row_idx = 3*8 + r = 24, 25` and `(THINK_START, THINK_END)` masks
        //    - r=2 (beyond truncated valid_len): `REJECTION_BONUS_Q_ROW_IDX`, `draft_token = 0`
        //    - r=3 (bonus row): `REJECTION_BONUS_Q_ROW_IDX`, `draft_token = 0`
        let a = make_seq(3, false);
        assert!(batched_rejection_seq_eligible(&a, 4, &ctx));
        assert_eq!(
            classify_rejection_tail_row_mode(&a, 4, &ctx, SpecVerifyPolicy::Rejection),
            RejectionTailRowMode::Rejection
        );
        let mut reqs = Vec::new();
        push_rejection_tail_row_specs_with_fence(
            &mut reqs,
            &a,
            &[10, 11, 12],
            10,
            4,
            RejectionTailRowMode::Rejection,
            Some((3, 8, 2)),
        );
        assert_eq!(reqs.len(), 4);
        assert_eq!(reqs[0].row_idx, 10);
        assert_eq!(reqs[0].q_row_idx, 24);
        assert_eq!(reqs[0].draft_token, 10);
        assert_eq!(reqs[0].mask_tok0, THINK_START);
        assert_eq!(reqs[0].mask_tok1, THINK_END);
        assert_eq!(reqs[0].min_p, 0.05);
        assert_eq!(reqs[1].q_row_idx, 25);
        assert_eq!(reqs[1].draft_token, 11);
        assert_eq!(reqs[2].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
        assert_eq!(reqs[2].draft_token, 0);
        assert_eq!(reqs[3].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
        assert_eq!(reqs[3].draft_token, 0);

        // 2. Stale/mismatched `q` fence (wrong slot or wrong abs_pos) -> valid_len == 0!
        assert_eq!(resolve_valid_rejection_q_len(Some((2, 8, 4)), 3, 8, 3), 0);
        assert_eq!(resolve_valid_rejection_q_len(Some((3, 9, 4)), 3, 8, 3), 0);
        assert_eq!(resolve_valid_rejection_q_len(None, 3, 8, 3), 0);
        assert_eq!(resolve_valid_rejection_q_len(Some((3, 8, 5)), 3, 8, 3), 3);

        // 3. Slot exceeding `REJECTION_Q_ROW_CAP / 8` fails closed to `REJECTION_BONUS_Q_ROW_IDX`.
        let high_slot = REJECTION_Q_ROW_CAP / 8;
        let a_high = make_seq(high_slot, true);
        let mut reqs_high = Vec::new();
        push_rejection_tail_row_specs_with_fence(
            &mut reqs_high,
            &a_high,
            &[10],
            0,
            2,
            RejectionTailRowMode::Rejection,
            Some((high_slot, 8, 1)),
        );
        assert_eq!(reqs_high[0].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
        assert_eq!(reqs_high[0].mask_tok0, REJECTION_NO_MASK_TOKEN);
        assert_eq!(reqs_high[0].mask_tok1, REJECTION_NO_MASK_TOKEN);

        // 4. Greedy row (`T == 0.0`): temp=0.0, draft rows carry slot*8+r + draft_token,
        //    bonus row carries `REJECTION_BONUS_Q_ROW_IDX`.
        let mut greedy = make_seq(2, false);
        greedy.temperature = 0.0;
        assert_eq!(
            classify_rejection_tail_row_mode(&greedy, 3, &ctx, SpecVerifyPolicy::Rejection),
            RejectionTailRowMode::GreedyArgmax
        );
        let mut reqs_g = Vec::new();
        push_rejection_tail_row_specs_with_fence(
            &mut reqs_g,
            &greedy,
            &[10, 11],
            0,
            3,
            RejectionTailRowMode::GreedyArgmax,
            None,
        );
        assert_eq!(reqs_g[0].temperature, 0.0);
        assert_eq!(reqs_g[0].q_row_idx, 16);
        assert_eq!(reqs_g[0].draft_token, 10);
        assert_eq!(reqs_g[1].q_row_idx, 17);
        assert_eq!(reqs_g[1].draft_token, 11);
        assert_eq!(reqs_g[2].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
        assert_eq!(reqs_g[2].draft_token, 0);

        // 5. Host-policy fallback (e.g. non-zero presence penalty): temp=-1.0, bonus q_row_idx.
        let mut pen = make_seq(1, false);
        pen.presence_penalty = 0.3;
        assert!(!batched_rejection_seq_eligible(&pen, 3, &ctx));
        assert_eq!(
            classify_rejection_tail_row_mode(&pen, 3, &ctx, SpecVerifyPolicy::Rejection),
            RejectionTailRowMode::HostPolicySkip
        );
        let mut reqs_p = Vec::new();
        push_rejection_tail_row_specs_with_fence(
            &mut reqs_p,
            &pen,
            &[10, 11],
            0,
            3,
            RejectionTailRowMode::HostPolicySkip,
            Some((1, 8, 2)),
        );
        assert!(reqs_p.iter().all(|r| r.temperature == -1.0
            && r.q_row_idx == REJECTION_BONUS_Q_ROW_IDX
            && r.draft_token == 0));
    }
}
