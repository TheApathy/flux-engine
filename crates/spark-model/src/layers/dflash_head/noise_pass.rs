// SPDX-License-Identifier: AGPL-3.0-only

//! One denoise pass of the DFlash γ-block forward: steps 2–5 of the
//! original `forward_block` body (noise-embedding build → 8 drafter
//! layers → final norm/lm_head → per-row argmax), extracted verbatim so
//! `forward_block` can iterate the pass for multi-step denoise drafting
//! (`ATLAS_DFLASH_DENOISE_STEPS=N`). Pass k+1 re-embeds pass k's argmax
//! at rows whose prediction was confident; the remaining masked rows are
//! then conditioned on partial predictions (DiffusionGemma-style
//! iterative refinement).
//!
//! With `committed` all-`None` (the single-pass default) this reproduces
//! the pre-extraction behavior bit-for-bit.

#[path = "lm_head_path.rs"]
mod lm_head_path;

mod eager;
mod graph_state;
mod policy;
mod stage1;
mod stage2;
mod stage2_layers;

#[cfg(test)]
mod debug_row_count_tests;
#[cfg(test)]
mod noise_graph_tests;

pub(super) use eager::NoisePassArgs;
pub(super) use graph_state::{
    NoiseGraphCaptureGuard, clear_noise_graph_head_state, drop_noise_graph_head_entry,
    gumbel_spec_fields, invalidate_dstate_ctx_kv_ranges,
};
pub(crate) use policy::CtxKvCacheRepr;
pub(super) use policy::{
    LM_HEAD_PATH_TAG_BF16, LmHeadNvfp4Mode, REJECTION_SLOT_STRIDE, dflash_noise_graph_enabled,
    dflash_split_kv_h128_workspace_bytes, noise_graph_attn_kgamma_disable_o,
    noise_graph_swa_enabled, resolve_dflash_split_kv_splits_for_sm, resolve_dflash2_selector_topk,
    resolve_lm_head_path_tag, resolve_rejection_seq_staging, resolve_rejection_shortlist_k,
    validate_selector_shortlist,
};

#[cfg(test)]
use super::DraftRowLayout;
#[cfg(test)]
use graph_state::{
    DflashLayerQuantMode, DflashNoiseGraphMeta, GumbelScratchPtrs, NoiseGraphHeadState,
    RingMetaLayout,
};
#[cfg(test)]
use policy::{
    CtxCacheInvalidationReason, DFLASH_SPLIT_KV_H128_MAX_SPLITS, DflashNoiseGraphKey,
    NoiseGraphRuntimeKey, Stage1RingSpan, Stage2CtxAttnMode, all_layers_cover_old_fc_prefix,
    bf16_proj_cublas_aligned, noise_graph_stochastic_draft_sampling_active,
    parse_dflash_noise_graph_env, parse_dflash_noise_graph_env_for_sm,
    parse_dflash_stage1_graph_enabled, plan_ctx_kv_cache_transition, plan_stage1_ring_append,
    resolve_c1_lm_head_nvfp4_mode_for_sm, resolve_stage2_ctx_attn_mode,
    resolve_stage2_ctx_attn_mode_for_sm, should_use_coalesced_nvfp4_lm_head,
};
