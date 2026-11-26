// SPDX-License-Identifier: AGPL-3.0-only

//! Native Qwen3.8-Flash-Next multi-token predictor.

mod forward;
mod graph_cache;
mod graphed_chain;
mod gumbel_draft;
mod init;
mod policy;
mod proposer_impl;
mod ragged_propose;
mod state;

#[cfg(test)]
mod draft_confidence_tests;

pub use init::Qwen4MtpHead;
pub use policy::{
    BATCHED_RAGGED_MTP_ENV, DEFAULT_MTP_RAGGED_PCUM_THRESH, MTP_CHAIN_ALIAS,
    MTP_DEVICE_PROPOSE_SELECTOR, MTP_DRAFT_SELECTION_REUSE_SELECTOR, MTP_GRAPH_SELECTOR,
    MTP_QSA_SELECTOR, MTP_RAGGED_PCUM_THRESH_ENV, MTP_STEP0_DUAL_GEMV_SELECTOR, MtpQsaMode,
    RaggedMtpTargetDraftsGuard, batched_ragged_mtp_enabled, checked_mtp_reused_index_width,
    mtp_bounded_blocks_selected, mtp_device_propose_selected, mtp_qsa_mode, mtp_ragged_pcum_thresh,
    mtp_step0_dual_gemv_selected, parse_mtp_qsa_mode, resolve_mtp_qsa_mode,
};
pub use state::{Qwen4MtpState, try_ngram_propose};

#[cfg(test)]
use graph_cache::{MtpGraphCache, MtpGraphKey, MtpPinnedH2dRing};
#[cfg(test)]
use policy::{
    BatchGumbelMetaLayout, CHAIN_ARGMAX_OFFSET, CHAIN_META_OFFSET, CHAIN_PROB_OFFSET,
    CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET, ChainConfidenceGate, ChainConfidenceScope,
    DraftStepReuseAction, MAX_ROTARY_METADATA_ROWS, MTP_GRAPH_CACHE_CAP, MtpGraphPolicy,
    POSITION_H_OFFSET, POSITION_W_OFFSET, advance_ragged_mtp_step, can_gpu_chain_ragged_steps,
    confident_prefix, confident_prefix_batch, draft_step_qsa_plan, mtp_graph_bt_blocks_cap,
    parse_batched_ragged_mtp, parse_mtp_graph_policy, parse_mtp_lm_head_tc_pipe_env,
    parse_mtp_ragged_pcum_thresh, plan_hidden_compaction_copies, prepare_packed_meta_scratch,
    prompt_tail_start, quantize_ragged_drafts_to_graph_bucket, ragged_mtp_should_stop_stragglers,
    ragged_row_bucket_step, resolve_chain_confidence_gate, resolve_chain_confidence_gate_for_scope,
    resolve_chain_confidence_gate_with_cum, resolve_mtp_graph_enabled, rotary_axes,
    rotary_metadata_ptrs, should_sync_mid_chain, should_use_batched_argmax_prob,
    should_use_mtp_fc_coalesced, should_use_mtp_lm_head_coalesced, should_use_mtp_lm_head_pipe,
    validate_rotary_span, write_rotary_metadata,
};
