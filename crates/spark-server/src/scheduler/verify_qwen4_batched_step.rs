// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched native-MTP verify step (`ATLAS_QWEN4_BATCHED_K5=1`).
//!
//! Every eligible speculating flat-chain sequence — including neutral greedy,
//! grammar-constrained, thinking-span (`think.enabled`), stochastic/Gumbel,
//! penalized, and `min_tokens` requests — is grouped by verify width K (or
//! packed into ragged mixed-K frames in `1..=8` when `batched_ragged_enabled()`
//! holds) into frames bounded by the active GPU's adaptive row cap (`128` rows
//! on GB10, `256` rows on `> 64`-SM GPUs, clamped to `QWEN4_CROSS_SEQ_MAX_ROWS
//! = 256` and `QWEN4_CROSS_SEQ_MAX_SEQS = 64` via
//! `spark_model::layer::Qwen4CrossSeqFrame::plan_frames_with_max_rows`). One
//! target forward verifies a whole frame (`Model::decode_verify_qwen4_batched`).
//! Each sequence then accepts its own exact prefix, emits, rolls back its host
//! extent and commits its own recurrent/QSA/PLE state from its own intermediates
//! (the accepted row is promoted, the rest discarded), all inside the verify
//! lane of its slot. A cohort owns every physical lane until all frames have
//! settled. Hidden handoffs for every sequence are saved before the cohort is
//! released and any proposer runs, because a proposal reuses the shared target
//! scratch.
//!
//! Sequences that are not eligible (trees, sampled-q frames, finished
//! sequences, thinking when `!think.enabled`, or `ATLAS_GRAMMAR_SERIAL_VERIFY=1`
//! grammar overrides), singleton widths, and frames the model refuses at
//! read-only admission fall back to the per-sequence verifier. After cohort
//! begin, every failure aborts the whole cohort without fallback.
//!
//! `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` (opt-in) narrows the verify width with
//! the number of live sequences so one frame fits inside the active row cap
//! (`qwen4_verify_lanes::adaptive_verify_width` /
//! `adaptive_verify_width_for_profile_and_max_rows`): under the 128-row default
//! profile, C1-C16 K8, C17-C18 K7, C19-C21 K6, C22-C25 K5, C26-C32 K4,
//! C33-C42 K3, C43-C128 K2 (`C129+` returns `None`); 256-row GPUs scale widths
//! upward within `K <= 8`. Narrowing only shortens a draft chain, exactly as
//! the MTP confidence gate does, so verify still decides every token.

mod depth_controller;
mod dflash_budget;
mod planning;
mod propose;
mod stochastic_staging;
mod verify_exec;

#[cfg(test)]
mod batched_gumbel_verify_tests;
#[cfg(test)]
mod batched_rejection_verify_tests;
#[cfg(test)]
mod dflash_budget_tests;
#[cfg(test)]
mod planning_tests;
#[cfg(test)]
mod strict_grammar_batch_tests;

pub(super) use depth_controller::{
    adaptive_selected, effective_num_drafts, narrow_pending_drafts, set_step_cohort_sampled,
};
pub(super) use dflash_budget::{apply_dflash_pcum_truncation, dflash_gumbel_num_drafts};
pub(super) use planning::{
    DraftDepthPolicy, clamp_pending_drafts_to_model_max_seq_len, harmonize_cohort_widths,
    plan_for_cohort,
};
pub(super) use propose::{
    finalize_collected_async_dflash_drafts, propose_next_batch, warm_ple_draft_ahead_for_seq,
};
pub(super) use verify_exec::{batch_sequences_mut, step_verify_qwen4_batched};

#[cfg(test)]
use depth_controller::{
    CohortAcceptEwma, adaptive_drafts, adaptive_drafts_with_max_rows, ladder_drafts, parse_ladder,
    resolve_cohort_width_capped, resolve_cohort_width_with_controller, resolve_effective_depth,
};
#[cfg(test)]
use dflash_budget::{
    apply_dflash_pcum_truncation_with_probs, cohort_live_and_all_non_greedy,
    dflash_pcum_default_for_cohort_with_policy, dflash_pcum_default_for_cohort_with_policy_for_sm,
    dflash_pcum_keep_len, dflash_pcum_keep_len_and_frontier_for_cohort_with_policy,
    dflash_pcum_keep_len_for_cohort, dflash_pcum_keep_len_for_cohort_with_policy,
    dflash_ragged_row_bucket_step, dflash_ragged_row_bucket_step_for_sm,
    parse_dflash_gumbel_gamma_override, parse_dflash_pcum_thresh_override,
    quantize_dflash_ragged_drafts_to_graph_bucket, quantize_ragged_lens_to_graph_bucket,
    quantize_ragged_lens_to_graph_bucket_with_remaining,
    quantize_ragged_lens_upward_to_graph_bucket_with_remaining, resolve_dflash_gumbel_num_drafts,
    resolve_dflash_gumbel_num_drafts_for_sm,
};
#[cfg(test)]
use planning::{
    BatchedK5Candidate, BatchedK5Plan, batched_k5_policy_required, batched_k5_width,
    batched_k5_width_for, batched_verify_proposer_eligible_with_policy, choose_harmonized_width,
    clamp_drafts_to_max_seq_len_count, clamp_pending_drafts_for_max_seq_len,
    parse_batched_k5_log_enabled, parse_grammar_serial_verify, plan_batched_k5,
    plan_batched_uniform, plan_batched_uniform_with_max_rows, plan_ragged_cohort,
    plan_ragged_cohort_with_max_rows,
};
#[cfg(test)]
use propose::{warm_ple_draft_ahead_for_seqs, warm_ple_verify_prefetch_for_seqs};
#[cfg(test)]
use stochastic_staging::{
    presample_batched_gumbel_frame_ragged_with_policy, presample_batched_gumbel_frame_with_policy,
    presample_batched_rejection_frame_ragged_with_policy,
    stage_batched_gumbel_frame_if_eligible_with_policy,
    stage_batched_rejection_frame_if_eligible_with_policy,
};
#[cfg(test)]
use verify_exec::verify_frame;
