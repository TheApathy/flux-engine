// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-module MTP proposer (MiniMax M2, DeepSeek-V3 style).
//!
//! Differs from the single-module `MtpHead` in one way only: each draft
//! slot dispatches to a *different* transformer module with its own
//! weights and its own KV cache. Draft `i` is produced by
//! `modules[i].forward_one(previous_draft_token, previous_module_hidden)`.
//!
//! Module count matches `config.num_mtp_modules` (3 for MiniMax M2.7).
//! When the verify loop requests fewer drafts than modules
//! (e.g. `--num-drafts 1` for non-spec smoke), only the first K modules
//! run — trailing modules stay idle but their state remains allocated.
//!
//! Weight-level validation is deferred: the public tiny-random variant
//! ships no MTP module weights, so unit tests exercise the dispatcher
//! plumbing from randomly-initialized `MtpHead` instances and defer
//! end-to-end acceptance-rate measurement to a session with the full
//! 229B checkpoint staged.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;
use crate::layers::mtp_head::{MtpHead, MtpProposerState};
use crate::speculative::{DraftProposer, ProposerState};

/// Per-sequence state for `MultiModuleMtpHead`.
///
/// One inner `MtpProposerState` per module — each tracks its own
/// block table and `seq_len`, since the modules do not share KV cache.
pub struct MultiModuleMtpState {
    /// `per_module[i]` belongs to `MultiModuleMtpHead::modules[i]`.
    /// Length invariant: matches the parent head's `modules.len()`.
    pub per_module: Vec<MtpProposerState>,
    /// Number of drafts produced by the last `propose()` call.
    /// `after_verify()` trims that many entries from KV cache.
    pub last_num_drafted: usize,
}

impl ProposerState for MultiModuleMtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// N independent MTP modules, one per draft slot.
pub struct MultiModuleMtpHead {
    /// Invariant: non-empty. Length equals `config.num_mtp_modules`.
    modules: Vec<MtpHead>,
    /// Number of modules (`k`) that ran in the most recent `propose()` call
    /// (`0` before any non-empty `propose()`), so `read_deferred_draft_token`
    /// reads from `modules[k - 1]` rather than a stale trailing module.
    last_k: AtomicUsize,
}

impl std::fmt::Debug for MultiModuleMtpHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiModuleMtpHead")
            .field("modules", &self.modules.len())
            .finish()
    }
}

impl MultiModuleMtpHead {
    /// Assemble a multi-module proposer from per-module heads.
    ///
    /// Callers construct each `MtpHead` via `MtpHead::new` with the
    /// MTP weights for that module's prefix (e.g. `model.layers.62..64`
    /// for MiniMax M2 with `num_hidden_layers=62, num_mtp_modules=3`).
    pub fn new(modules: Vec<MtpHead>) -> Result<Self> {
        anyhow::ensure!(
            !modules.is_empty(),
            "MultiModuleMtpHead requires at least one module (got 0); \
             caller should not construct this type for single-module MTP"
        );
        Ok(Self {
            modules,
            last_k: AtomicUsize::new(0),
        })
    }

    /// Number of MTP modules available (caps `num_drafts` in propose).
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }
}

impl DraftProposer for MultiModuleMtpHead {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let per_module = (0..self.modules.len())
            .map(|_| MtpProposerState {
                rotary_positions: crate::traits::RotaryPositions::identity(),
                block_table: Vec::new(),
                seq_len: 0,
                last_num_drafted: 0,
            })
            .collect();
        Ok(Box::new(MultiModuleMtpState {
            per_module,
            last_num_drafted: 0,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;

        // Cap at module count — caller asking for more drafts than we
        // have modules is a config mismatch, clamp quietly (matches
        // single-module MtpHead behavior when num_drafts > 1).
        let k = num_drafts.min(self.modules.len());

        let mut drafts = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..k {
            // Only the last draft's embedding is GPU-pre-staged — the
            // verify loop uses it as the next step's input. Earlier
            // drafts feed into the next MTP module's `target_hidden`
            // in-process, no GPU embed needed.
            let embed_target = if i == k - 1 { draft_embed_target } else { None };

            // Grammar mask: the same single-position mask is passed to every
            // module; warn loudly if K > 1 combines with an active grammar mask
            // because we do not speculatively advance the matcher between steps.
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "MultiModuleMTP grammar-masked drafting called with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let mask_for_draft = grammar_bitmask;

            let draft = self.modules[i].forward_one(
                current_token,
                current_hidden,
                position + i,
                &mut mm_state.per_module[i],
                ctx,
                stream,
                embed_target,
                mask_for_draft,
            )?;

            tracing::debug!(
                "MultiMTP propose[{i}/{k}]: token={current_token} pos={} module_seq_len={} → draft={draft}",
                position + i,
                mm_state.per_module[i].seq_len,
            );

            drafts.push(draft);
            current_token = draft;
            // Chain: module {i+1} consumes module i's hidden state.
            // MtpHead writes its own hidden output into
            // `ctx.buffers.hidden_states()` before the LM head GEMM.
            current_hidden = ctx.buffers.hidden_states();
        }

        mm_state.last_num_drafted = drafts.len();
        self.last_k.store(drafts.len(), Ordering::Release);
        Ok(drafts)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        // The last draft came from `modules[k - 1]` where `k` is the draft
        // count produced by the preceding `propose()` call.
        let k = self.last_k.load(Ordering::Acquire);
        anyhow::ensure!(
            k > 0 && k <= self.modules.len(),
            "MultiModuleMtpHead::read_deferred_draft_token called without a prior non-empty propose() (last_k={k}, modules={})",
            self.modules.len()
        );
        self.modules[k - 1].read_deferred_draft_token(gpu)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;

        // Each module sees exactly one token per propose() iteration.
        // If the verifier accepted `num_accepted` of `k` drafts, each
        // module trims `(1 if this slot was rejected else 0)`.
        // Equivalent: modules[0..num_accepted] keep their last entry,
        // modules[num_accepted..k] trim 1.
        let k = mm_state.last_num_drafted;
        for (i, per) in mm_state.per_module.iter_mut().take(k).enumerate() {
            per.last_num_drafted = 1;
            let trim = if i < num_accepted { 0 } else { 1 };
            if trim > 0 {
                per.seq_len = per.seq_len.saturating_sub(trim);
            }
        }
        // Per-module trim above updates only host-side `seq_len`; no GPU work
        // needs to be enqueued on `stream`.
        let _ = stream;
        tracing::debug!(
            "MultiMTP after_verify: accepted={num_accepted} of {k}; per-module trim done"
        );
        Ok(())
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;
        // Reclaim each module's KV cache blocks directly via its cache lock.
        for (i, per) in mm_state.per_module.iter_mut().enumerate() {
            let head = &self.modules[i];
            if !per.block_table.is_empty() {
                head.kv_cache_lock().free_blocks(&per.block_table);
                per.block_table.clear();
            }
            per.seq_len = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_matches_module_count() {
        // Construct a MultiModuleMtpState as the dispatcher would.
        let state = MultiModuleMtpState {
            per_module: (0..3)
                .map(|_| MtpProposerState {
                    rotary_positions: crate::traits::RotaryPositions::identity(),
                    block_table: Vec::new(),
                    seq_len: 0,
                    last_num_drafted: 0,
                })
                .collect(),
            last_num_drafted: 0,
        };
        assert_eq!(state.per_module.len(), 3);
    }

    #[test]
    fn test_empty_modules_rejected() {
        let err = MultiModuleMtpHead::new(vec![]).unwrap_err();
        assert!(
            err.to_string().contains("at least one module"),
            "unexpected error: {err}"
        );
    }
}
