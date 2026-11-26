// SPDX-License-Identifier: AGPL-3.0-only

//! Model implementations and forward-pass orchestration.
//!
//! `TransformerModel` provides the architecture-agnostic model loop
//! (`embed -> layers -> norm -> lm_head`) delegating layer-specific work to
//! `TransformerLayer` implementations. Specialized model runtimes (`dsv41`,
//! `glm53`) live alongside it.
//!
//! Key submodules:
//! - `types`             `TransformerModel`, `PinnedMetaStaging`, graph cache keys
//! - `impl_a1/2/3`, `impl_a1_init`, `impl_b1/2/3`  inherent `TransformerModel` helpers
//! - `trait_impl`        `impl Model for TransformerModel` and dispatch submodules
//! - `drop`              `impl Drop for TransformerModel`
//! - `block_mgmt`        KV block allocation, HSS sliding window, prefix reuse
//! - `ssm_pool`, `ssm_pool_geometry`, `ssm_snapshot`  SSM state & snapshot pools
//! - `seqstate`          Qwen4 sequence-state serialization and tiered RAM/disk cache
//! - `dsv41`             DeepSeek-V4.1 model & DSpark speculative runtime
//! - `glm53`             GLM-5.3 GGUF & EXL3 target + DFlash2 runtime
//! - `vision_embeddings`, `early_exit`, `decode_stages`, `host_bucket`, `k8a_lastk_defer`,
//!   `k16_commit_parity`, `qwen4_*`, `rotary_*`, `verify_*`, `env_diag`, `final_norm`

#![allow(unused_imports, dead_code)]

pub(crate) mod block_mgmt;
pub(crate) mod control_engagement;
pub(crate) mod decode_stages;
pub(crate) mod drop;
pub mod dsv41;
pub(crate) mod early_exit;
pub(crate) mod env_diag;
mod final_norm;
pub(crate) mod host_bucket;
pub(crate) mod impl_a1;
pub(crate) mod impl_a1_init;
pub(crate) mod impl_a2;
pub(crate) mod impl_a3;
pub(crate) mod impl_b1;
pub(crate) mod impl_b2;
pub(crate) mod impl_b3;
pub(crate) mod k10a_host;
pub mod k16_commit_parity;
pub(crate) mod k16_route_receipt;
pub(crate) mod k1_stage_diag;
pub(crate) mod k8a_lastk_defer;
pub(crate) mod proposer_rotary;
pub mod qwen4_k5_cmulti;
pub(crate) mod qwen4_mtp_hidden;
pub(crate) mod qwen4_prefill_engagement;
pub mod qwen4_target_state_receipt;
pub(crate) mod rotary_graphs;
pub(crate) mod rotary_meta;
pub mod seqstate;
pub(crate) mod slot_zero_plan;
pub(crate) mod speculation_control;
pub(crate) mod ssm_pool;
pub mod ssm_pool_geometry;
pub(crate) mod ssm_snapshot;
pub(crate) mod trait_impl;
pub(crate) mod types;
mod verify_batch_frame;
mod verify_failure;
mod verify_frame;
pub(crate) mod verify_logits_receipt;
pub(crate) mod vision_embeddings;

// GLM-5.3 GGUF & EXL3 model runtimes and admission gates.
pub mod glm53;

pub use trait_impl::commit_cache::{
    BatchCommitEntry, CachedSlotTable, CommitRuntimeFlags, SlotCommitTable, SsmLayerCommitPointers,
};
pub use trait_impl::mixed_qwen4_piggyback::{
    MIXED_PIGGYBACK_ENV, MixedPiggybackMode, Qwen4MixedRound, Qwen4MixedRoundOutput,
};
pub use types::TransformerModel;
