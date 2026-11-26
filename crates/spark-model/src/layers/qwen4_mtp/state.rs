// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence state and n-gram fast-path drafting for Qwen4 native MTP.

use std::any::Any;

use anyhow::Result;
use parking_lot::Mutex;

use super::gumbel_draft;
use crate::layer::LayerState;
use crate::layers::qwen4_mtp_rows::{AfterVerifyPlan, RowPolicy};
use crate::speculative::ProposerState;

pub struct Qwen4MtpState {
    /// Sequence-owned target rotary coordinates. Cache addresses and lengths
    /// remain physical even when an image prompt uses three-axis MRoPE.
    pub(crate) rotary_positions: crate::traits::RotaryPositions,
    pub(super) layer_state: Box<dyn LayerState>,
    pub(super) block_table: Vec<u32>,
    pub(super) disk_block_ids: Vec<u32>,
    pub(super) disk_last_offloaded_per_layer: Vec<u32>,
    pub(super) seq_len: usize,
    pub(super) last_num_drafted: usize,
    /// Every draft the last chain computed (`d_1..d_c`), for the K7
    /// committed-row rewrite (`ATLAS_QWEN4_MTP_RECONDITION`).
    pub(super) last_drafts: Vec<u32>,
    /// Committed draft rows to rewrite from staged target hidden rows at the
    /// start of the next propose (set by `after_verify`).
    pub(super) pending_replay: Option<PendingReplay>,
    /// The sequence's exact verify generation, bound before after_verify.
    pub(super) verify_frame: Option<crate::traits::verify_owner::Ticket>,
    /// Owner-bound row publication; an unrelated request cannot satisfy it by count.
    pub(super) staged_frame: Mutex<crate::traits::verify_owner::StagedFrame>,
    /// Sequence-scoped CRN Gumbel draft spec (`Some(Some(spec))` = coupled Gumbel,
    /// `Some(None)` = explicit greedy/non-Gumbel, `None` = fallback to ring).
    pub(super) pending_gumbel_spec:
        Mutex<Option<Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>>>,
}

/// A planned committed-row rewrite: the plan plus the frame's accepted count,
/// which locates the root row relative to the next chain's position. When
/// `after_verify` runs before `save_hidden_for_mtp` (`verify_k2/k3/k4_step`
/// partial/reject paths), `staged_at_trim` is false and `propose` upgrades
/// the fallback `KeepLast` plan once the staged hidden rows are noted.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingReplay {
    pub(super) owner: Option<crate::traits::verify_owner::Ticket>,
    pub(super) row_policy: RowPolicy,
    pub(super) computed: usize,
    pub(super) staged_accepted: usize,
    pub(super) plan: AfterVerifyPlan,
    pub(super) accepted: usize,
    pub(super) staged_at_trim: bool,
}

impl Qwen4MtpState {
    pub(crate) fn has_unconsumed_verify_frame(&self) -> bool {
        self.pending_replay.is_some() || self.staged_frame.lock().pending()
    }

    pub(crate) fn bind_verify_frame(
        &mut self,
        owner: Option<crate::traits::verify_owner::Ticket>,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .bind(owner)
            .map_err(|error| anyhow::anyhow!("MTP staged owner binding failed: {error:?}"))?;
        self.verify_frame = owner;
        Ok(())
    }

    pub(crate) fn publish_staged_frame(
        &self,
        owner: crate::traits::verify_owner::Ticket,
        rows: usize,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .publish(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged publication failed: {error:?}"))
    }

    pub(crate) fn check_staged_frame(
        &self,
        owner: crate::traits::verify_owner::Ticket,
        rows: usize,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .check_publish(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged publication preflight failed: {error:?}"))
    }

    pub(super) fn take_staged_frame(
        &self,
        owner: Option<crate::traits::verify_owner::Ticket>,
        rows: usize,
    ) -> Result<bool> {
        self.staged_frame
            .lock()
            .take(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged consumption failed: {error:?}"))
    }

    pub(crate) fn discard_verify_frame(&mut self) {
        self.verify_frame = None;
        self.staged_frame.lock().clear();
        self.pending_replay = None;
        *self.pending_gumbel_spec.lock() = None;
    }

    /// Stamp this sequence's upcoming CRN Gumbel draft spec (`None` = greedy / uncoupled).
    pub fn set_gumbel_spec(&self, spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>) {
        *self.pending_gumbel_spec.lock() = Some(spec);
    }

    pub(super) fn resolve_gumbel_spec(
        &self,
        position: usize,
    ) -> Option<spark_runtime::sampler::gumbel::GumbelDraftSpec> {
        if !gumbel_draft::coupled_drafting_enabled() {
            return None;
        }
        match *self.pending_gumbel_spec.lock() {
            Some(published) => published.filter(|s| s.is_active()),
            None => gumbel_draft::active_spec_for(position),
        }
    }

    pub(super) fn clear_gumbel_spec(&self) {
        *self.pending_gumbel_spec.lock() = None;
    }

    pub(crate) fn snapshot_dedup_draft(&self) -> (usize, usize, Vec<u32>) {
        (
            self.seq_len,
            self.last_num_drafted,
            self.last_drafts.clone(),
        )
    }

    pub(crate) fn apply_dedup_draft(
        &mut self,
        seq_len: usize,
        last_num_drafted: usize,
        last_drafts: &[u32],
    ) {
        self.staged_frame.lock().clear();
        self.seq_len = seq_len;
        self.last_num_drafted = last_num_drafted;
        self.last_drafts.clear();
        self.last_drafts.extend_from_slice(last_drafts);
    }
}

impl ProposerState for Qwen4MtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Qwen4MtpState {
    /// Prepare state when an external zero-cost drafter (e.g. N-gram suffix
    /// copy-drafter) supplies the current cycle's drafts instead of launching
    /// the GPU MTP chain.
    pub fn notify_external_draft(&mut self) {
        if let Some(pending) = self.pending_replay.take()
            && pending.computed > 0
            && pending.staged_at_trim
        {
            let keep_last = (pending.accepted + 1).min(pending.computed);
            if keep_last > pending.plan.keep_rows {
                self.seq_len += keep_last - pending.plan.keep_rows;
            }
        }
        self.verify_frame = None;
        self.last_num_drafted = 0;
        self.last_drafts.clear();
        self.staged_frame.lock().clear();
        self.clear_gumbel_spec();
    }

    pub fn notify_external_draft_with_tokens(&mut self, drafts: &[u32]) {
        self.notify_external_draft();
        self.last_drafts.extend_from_slice(drafts);
    }

    /// Record suffix tokens grafted onto the tail of a partial neural MTP
    /// proposal without altering `last_num_drafted` (only the neural prefix
    /// wrote MTP KV cache rows).
    pub fn notify_grafted_suffix(&mut self, suffix: &[u32]) {
        self.last_drafts.extend_from_slice(suffix);
    }

    /// Synchronize `last_drafts` prefix when host grammar coalescence splices
    /// forced/legal tokens into a proposed chain before installation.
    pub fn sync_spliced_drafts(&mut self, drafts: &[u32]) {
        let n = drafts.len().min(self.last_drafts.len());
        self.last_drafts[..n].copy_from_slice(&drafts[..n]);
    }
}

/// Zero-cost CPU N-gram suffix copy-drafter (`ATLAS_QWEN4_MTP_NGRAM`, default
/// enabled when `ATLAS_QWEN4_VERIFY_FAST_HOST=1` unless `ATLAS_QWEN4_MTP_NGRAM=0`).
///
/// Searches the committed token history `(prefix, last_token)` in a single
/// backward pass over an 8K window for a `6..=12`-gram suffix match, or a
/// repeated `4..=5`-gram match in the last 128 tokens with agreeing 2-token
/// continuation. Returns `num_drafts` copied tokens in `< 2 us` on hit.
pub fn try_ngram_propose(prefix: &[u32], last_token: u32, num_drafts: usize) -> Option<Vec<u32>> {
    static NGRAM_MIN_N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let min_n = *NGRAM_MIN_N.get_or_init(|| {
        if let Ok(v) = std::env::var("ATLAS_QWEN4_MTP_NGRAM") {
            let s = v.trim();
            if s == "0" || s.eq_ignore_ascii_case("off") {
                return 0;
            }
            if let Ok(n) = s.parse::<usize>()
                && (3..=16).contains(&n)
            {
                tracing::info!("QWEN4_MTP_NGRAM_ENGAGED min_n={n}");
                return n;
            }
        }
        if crate::layers::qwen4_verify_fast_host_selected() {
            tracing::info!("QWEN4_MTP_NGRAM_ENGAGED min_n=6 (with local n=4 repeated fallback)");
            6
        } else {
            0
        }
    });
    if min_n == 0 || num_drafts == 0 {
        return None;
    }
    let plen = prefix.len();
    let pat_prev = min_n - 1;
    // 1. Single backward pass for n >= min_n (default 6..=12) across an 8192-token window.
    if plen > pat_prev + num_drafts {
        let pat_slice = &prefix[plen - pat_prev..plen];
        let pat_first = pat_slice[0];
        let max_end = plen - num_drafts - 1;
        let start_end = plen.saturating_sub(8192).max(pat_prev);
        let mut best_len = 0usize;
        let mut best_start = 0usize;
        for idx_end in (start_end..=max_end).rev() {
            if prefix[idx_end] == last_token
                && prefix[idx_end - pat_prev] == pat_first
                && &prefix[idx_end - pat_prev..idx_end] == pat_slice
            {
                let mut mlen = min_n;
                while mlen < 12
                    && idx_end >= mlen
                    && plen >= mlen
                    && prefix[idx_end - mlen] == prefix[plen - mlen]
                {
                    mlen += 1;
                }
                if mlen > best_len {
                    best_len = mlen;
                    best_start = idx_end + 1;
                    if best_len == 12 {
                        break;
                    }
                }
            }
        }
        if best_len >= min_n {
            return Some(prefix[best_start..best_start + num_drafts].to_vec());
        }
    }
    // 2. Local 128-token fallback for repeated 4-gram suffix with agreeing 2-token continuation.
    if min_n <= 6 && plen > 3 + num_drafts && num_drafts >= 2 {
        let pat3 = &prefix[plen - 3..plen];
        let p0 = pat3[0];
        let max_end = plen - num_drafts - 1;
        let start_end = plen.saturating_sub(128).max(3);
        let mut first_cand: Option<&[u32]> = None;
        for idx_end in (start_end..=max_end).rev() {
            if prefix[idx_end] == last_token
                && prefix[idx_end - 3] == p0
                && &prefix[idx_end - 3..idx_end] == pat3
            {
                let cand = &prefix[idx_end + 1..idx_end + 1 + num_drafts];
                if let Some(first) = first_cand {
                    if cand[0] == first[0] && cand[1] == first[1] {
                        return Some(first.to_vec());
                    }
                } else {
                    first_cand = Some(cand);
                }
            }
        }
    }
    None
}
