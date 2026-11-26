// SPDX-License-Identifier: AGPL-3.0-only

//! Paired scheduler ownership for speculative drafts and optional DDTree data.

use anyhow::Result;
use spark_model::{layers::DDTreePayload, traits::SequenceState};

use super::{ActiveSeq, Model};

#[path = "flat_batch_lifecycle.rs"]
mod flat_batch_lifecycle;
pub(crate) use flat_batch_lifecycle::{
    FixedBatchEligibility, FixedBatchWidth, FlatBatchPreflight, fixed_batch_decision,
    flat_batch_preflight_at, take_flat_batch_at,
};

/// Typed producer boundary used by the scheduler pairing authority. Tests can
/// supply a narrow source without implementing the full serving model trait.
pub(crate) trait ProposalTreeSource<State, Tree> {
    type Sampled;

    fn propose(
        &self,
        state: &mut State,
        token: u32,
        position: usize,
        num_drafts: usize,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>>;

    fn skip_propose(&self, _state: &mut State) {}

    fn invalidate_rejection_q(&self, _state: &mut State) {}

    fn is_dflash_proposer(&self) -> bool {
        false
    }

    fn take_tree(&self, state: &mut State) -> Option<Tree>;

    fn take_sampled(&self, state: &mut State) -> Option<Self::Sampled>;

    fn validate_proposal_aux(
        &self,
        _drafts: &[u32],
        _tree: Option<&Tree>,
        _sampled: Option<&Self::Sampled>,
    ) -> Result<()> {
        Ok(())
    }
}

impl<M: Model + ?Sized> ProposalTreeSource<SequenceState, DDTreePayload> for M {
    type Sampled = spark_model::dflash_sampling::DflashSampledProposalFrame;

    fn propose(
        &self,
        state: &mut SequenceState,
        token: u32,
        position: usize,
        num_drafts: usize,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi(
            token,
            position,
            num_drafts,
            state,
            self.default_stream(),
            grammar_bitmask,
        )
    }

    fn skip_propose(&self, state: &mut SequenceState) {
        state.retire_committed_verify_handoff();
        self.invalidate_rejection_q(state);
    }

    fn invalidate_rejection_q(&self, state: &mut SequenceState) {
        if let Some(dflash_state) = state.proposer_state.as_ref().and_then(|s| {
            s.as_any()
                .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
        }) {
            dflash_state.invalidate_rejection_q();
        }
    }

    fn is_dflash_proposer(&self) -> bool {
        self.proposer_is_dflash()
    }

    fn take_tree(&self, state: &mut SequenceState) -> Option<DDTreePayload> {
        self.take_pending_tree_payload(state)
    }

    fn take_sampled(&self, state: &mut SequenceState) -> Option<Self::Sampled> {
        self.take_pending_dflash_sampled_proposal(state)
    }

    fn validate_proposal_aux(
        &self,
        drafts: &[u32],
        tree: Option<&DDTreePayload>,
        sampled: Option<&Self::Sampled>,
    ) -> Result<()> {
        anyhow::ensure!(
            tree.is_none() || sampled.is_none(),
            "a proposal cannot carry both DDTree topology and sampled DFlash q"
        );
        if let Some(frame) = sampled {
            frame.validate(self.vocab_size())?;
            anyhow::ensure!(
                frame.proposal.tokens.as_slice() == drafts,
                "sampled DFlash q does not describe the final published drafts"
            );
        }
        Ok(())
    }
}

/// Mutable scheduler frame accepted by the production pairing authority.
pub(crate) trait SchedulerProposalFrame {
    type State;
    type Tree;
    type Sampled;

    fn position(&self) -> usize;
    fn parts(&mut self) -> ProposalParts<'_, Self::State, Self::Tree, Self::Sampled>;
}

/// `(state, drafts, tree, sampled)` borrowed from one proposal frame.
pub(crate) type ProposalParts<'a, S, T, P> = (
    &'a mut S,
    &'a mut Vec<u32>,
    &'a mut Option<T>,
    &'a mut Option<P>,
);

impl SchedulerProposalFrame for ActiveSeq {
    type State = SequenceState;
    type Tree = DDTreePayload;
    type Sampled = spark_model::dflash_sampling::DflashSampledProposalFrame;

    fn position(&self) -> usize {
        self.seq.seq_len
    }

    fn parts(
        &mut self,
    ) -> (
        &mut Self::State,
        &mut Vec<u32>,
        &mut Option<Self::Tree>,
        &mut Option<Self::Sampled>,
    ) {
        (
            &mut self.seq,
            &mut self.pending_drafts,
            &mut self.pending_tree_payload,
            &mut self.pending_dflash_sampled_proposal,
        )
    }
}

/// Replace the current scheduler proposal with one producer result. The
/// producer-side tree slot is drained exactly once even when the result is
/// empty or failed, so topology can never cross proposal-frame boundaries.
pub(crate) fn install<T, Q, E>(
    pending_drafts: &mut Vec<u32>,
    pending_tree: &mut Option<T>,
    pending_sampled: &mut Option<Q>,
    result: Result<Vec<u32>, E>,
    take_tree: impl FnOnce() -> Option<T>,
    take_sampled: impl FnOnce() -> Option<Q>,
) -> Result<bool, E> {
    let tree = take_tree();
    let sampled = take_sampled();
    match result {
        Ok(drafts) if !drafts.is_empty() => {
            *pending_drafts = drafts;
            *pending_tree = tree;
            *pending_sampled = sampled;
            Ok(true)
        }
        Ok(_) => {
            clear(pending_drafts, pending_tree, pending_sampled);
            Ok(false)
        }
        Err(error) => {
            clear(pending_drafts, pending_tree, pending_sampled);
            Err(error)
        }
    }
}

/// Publish an already-collected result and drain its producer-side tree once.
pub(crate) fn install_collected<S, F>(
    source: &S,
    frame: &mut F,
    proposal: Result<Vec<u32>>,
) -> Result<bool>
where
    S: ProposalTreeSource<F::State, F::Tree, Sampled = F::Sampled> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let (state, pending_drafts, pending_tree, pending_sampled) = frame.parts();
    let tree = source.take_tree(state);
    let sampled = source.take_sampled(state);
    let proposal = match proposal {
        Ok(drafts) if !drafts.is_empty() => source
            .validate_proposal_aux(&drafts, tree.as_ref(), sampled.as_ref())
            .map(|()| drafts),
        other => other,
    };
    let installed = install(
        pending_drafts,
        pending_tree,
        pending_sampled,
        proposal,
        || tree,
        || sampled,
    );
    if !matches!(installed, Ok(true)) {
        source.invalidate_rejection_q(state);
    }
    installed
}

/// Publish a flat draft chain that did NOT come from the model proposer.
///
/// Used by retrieval drafting (`crate::rest_store`), which pre-empts a
/// proposer call rather than following one. The producer-side tree slot is
/// still drained exactly once and then dropped, so topology left by an
/// earlier proposal can neither leak into this frame nor survive into a
/// later one; the installed frame is always flat.
///
/// Returns whether a non-empty chain was installed.
pub(crate) fn install_external_flat<S, F>(source: &S, frame: &mut F, drafts: Vec<u32>) -> bool
where
    S: ProposalTreeSource<F::State, F::Tree, Sampled = F::Sampled> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let (state, pending_drafts, pending_tree, pending_sampled) = frame.parts();
    // Drained and dropped: an external chain publishes a flat frame and
    // invalidates any prior neural GPU rejection-sampling `q` fence.
    let _stale = source.take_tree(state);
    let _stale_sampled = source.take_sampled(state);
    source.invalidate_rejection_q(state);
    let installed: Result<bool, std::convert::Infallible> = install(
        pending_drafts,
        pending_tree,
        pending_sampled,
        Ok(drafts),
        || None,
        || None,
    );
    installed.unwrap_or(false)
}

#[inline]
fn should_propose_after_adaptive_gate(
    should_propose: bool,
    proposer_is_dflash: bool,
    proposal_suppressed: bool,
) -> bool {
    should_propose && (proposer_is_dflash || !proposal_suppressed)
}

/// Run an optional proposal, preserve caller work at the proposal-complete
/// boundary, then atomically drain and publish its drafts/tree frame.
pub(crate) fn propose_and_install_with<S, F, T>(
    source: &S,
    frame: &mut F,
    token: u32,
    num_drafts: usize,
    grammar_bitmask: Option<&[i32]>,
    should_propose: bool,
    after_propose: impl FnOnce(&mut F, &mut Result<Vec<u32>>) -> T,
) -> (Result<bool>, T)
where
    S: ProposalTreeSource<F::State, F::Tree, Sampled = F::Sampled> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    let position = frame.position();
    // FLUX_SPEC_VERIFY=stochastic adaptive depth 0: no drafts this round, so
    // the next step runs plain decode. Always false when the flag is off.
    // DFlash emits an entire block in a single pass and owns its own fallback
    // gates, so MTP adaptive_depth suppression never silences a DFlash proposer.
    let proposer_is_dflash = source.is_dflash_proposer();
    let suppressed = !proposer_is_dflash && super::adaptive_depth::proposal_suppressed();
    let should_propose =
        should_propose_after_adaptive_gate(should_propose, proposer_is_dflash, suppressed);
    let mut proposal = if should_propose {
        let (state, _, _, _) = frame.parts();
        source.propose(state, token, position, num_drafts, grammar_bitmask)
    } else {
        let (state, _, _, _) = frame.parts();
        source.skip_propose(state);
        Ok(Vec::new())
    };
    let after = after_propose(frame, &mut proposal);
    (install_collected(source, frame, proposal), after)
}

/// Run one model proposal and atomically publish its drafts/tree frame.
pub(crate) fn propose_and_install<S, F>(
    source: &S,
    frame: &mut F,
    token: u32,
    num_drafts: usize,
    grammar_bitmask: Option<&[i32]>,
) -> Result<bool>
where
    S: ProposalTreeSource<F::State, F::Tree, Sampled = F::Sampled> + ?Sized,
    F: SchedulerProposalFrame + ?Sized,
{
    propose_and_install_with(
        source,
        frame,
        token,
        num_drafts,
        grammar_bitmask,
        true,
        |_, _| (),
    )
    .0
}

/// Clear all three parts (`drafts`, `tree`, `sampled`) of the current scheduler proposal frame.
pub(crate) fn clear<T, Q>(drafts: &mut Vec<u32>, tree: &mut Option<T>, sampled: &mut Option<Q>) {
    drafts.clear();
    *tree = None;
    *sampled = None;
}

/// Clear the scheduler proposal frame on `a` and invalidate any pending GPU
/// rejection-sampling `(q_cand_ids, q_probs)` fence on its proposer state.
pub(crate) fn clear_seq(a: &mut ActiveSeq) {
    clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
}

/// Drain a scheduler-owned proposal frame when its request was cancelled at a
/// transaction boundary. Cancellation is not a proposal error, but no draft,
/// tree topology, or sampled-q evidence may survive into retirement.
pub(crate) fn clear_if_cancelled<T, Q>(
    cancelled: bool,
    drafts: &mut Vec<u32>,
    tree: &mut Option<T>,
    sampled: &mut Option<Q>,
) -> bool {
    if cancelled {
        clear(drafts, tree, sampled);
    }
    cancelled
}

/// Clear topology that has no draft frame to identify and consume it.
pub(crate) fn clear_orphan_aux<T, Q>(
    drafts: &[u32],
    tree: &mut Option<T>,
    sampled: &mut Option<Q>,
) {
    if drafts.is_empty() {
        *tree = None;
        *sampled = None;
    }
}

/// Consume a frame only when it is flat (neither `tree` nor `sampled` is
/// attached). Tree-bearing and sampled-q frames must be routed to the generic
/// verifier.
pub(crate) fn take_flat<T, Q>(
    drafts: &mut Vec<u32>,
    tree: &Option<T>,
    sampled: &Option<Q>,
) -> Option<Vec<u32>> {
    if tree.is_some() || sampled.is_some() {
        return None;
    }
    Some(std::mem::take(drafts))
}

/// Apply a grammar-safe prefix to a proposal. If `keep > drafts.len()`, or if
/// `sampled.is_some()` and `keep != drafts.len()` (sampled-q frames cannot be
/// partially truncated), all three parts (`drafts`, `tree`, `sampled`) are
/// cleared and `false` is returned. Otherwise truncating a non-sampled frame
/// clears `tree` so the shortened draft slice becomes a flat chain; a
/// zero-length result clears all three parts.
pub(crate) fn retain_prefix<T, Q>(
    drafts: &mut Vec<u32>,
    tree: &mut Option<T>,
    sampled: &mut Option<Q>,
    keep: usize,
) -> bool {
    if keep > drafts.len() {
        clear(drafts, tree, sampled);
        return false;
    }
    if sampled.is_some() && keep != drafts.len() {
        clear(drafts, tree, sampled);
        return false;
    }
    if keep < drafts.len() {
        drafts.truncate(keep);
        *tree = None;
    }
    clear_orphan_aux(drafts, tree, sampled);
    !drafts.is_empty()
}

#[cfg(test)]
#[path = "proposal_lifecycle_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "proposal_lifecycle_model_tests.rs"]
mod model_tests;

#[cfg(test)]
#[path = "proposal_lifecycle_source_guard_tests.rs"]
mod source_guard_tests;

#[cfg(test)]
mod dflash_adaptive_bypass_tests {
    use super::should_propose_after_adaptive_gate;

    #[test]
    fn dflash_ignores_mtp_adaptive_depth_suppression_while_native_mtp_honors_it() {
        assert!(should_propose_after_adaptive_gate(true, true, true));
        assert!(should_propose_after_adaptive_gate(true, true, false));
        assert!(!should_propose_after_adaptive_gate(false, true, true));

        assert!(!should_propose_after_adaptive_gate(true, false, true));
        assert!(should_propose_after_adaptive_gate(true, false, false));
        assert!(!should_propose_after_adaptive_gate(false, false, false));
    }

    #[test]
    fn external_empty_and_failed_proposals_invalidate_rejection_q_while_valid_keeps_it() {
        use super::{
            ProposalParts, ProposalTreeSource, SchedulerProposalFrame, install_collected,
            install_external_flat,
        };
        use anyhow::Result;

        struct StubState {
            rejection_q_valid: bool,
        }
        struct StubFrame {
            state: StubState,
            drafts: Vec<u32>,
            tree: Option<()>,
            sampled: Option<()>,
        }
        impl SchedulerProposalFrame for StubFrame {
            type State = StubState;
            type Tree = ();
            type Sampled = ();
            fn position(&self) -> usize {
                0
            }
            fn parts(&mut self) -> ProposalParts<'_, Self::State, Self::Tree, Self::Sampled> {
                (
                    &mut self.state,
                    &mut self.drafts,
                    &mut self.tree,
                    &mut self.sampled,
                )
            }
        }
        struct StubSource;
        impl ProposalTreeSource<StubState, ()> for StubSource {
            type Sampled = ();
            fn propose(
                &self,
                _state: &mut StubState,
                _token: u32,
                _position: usize,
                _num_drafts: usize,
                _grammar_bitmask: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                Ok(Vec::new())
            }
            fn invalidate_rejection_q(&self, state: &mut StubState) {
                state.rejection_q_valid = false;
            }
            fn take_tree(&self, _state: &mut StubState) -> Option<()> {
                None
            }
            fn take_sampled(&self, _state: &mut StubState) -> Option<()> {
                None
            }
        }

        let source = StubSource;
        let mut frame = StubFrame {
            state: StubState {
                rejection_q_valid: true,
            },
            drafts: Vec::new(),
            tree: None,
            sampled: None,
        };

        assert!(install_collected(&source, &mut frame, Ok(vec![10, 11])).unwrap());
        assert!(
            frame.state.rejection_q_valid,
            "non-empty neural proposal must preserve its GPU rejection q fence"
        );

        assert!(install_external_flat(&source, &mut frame, vec![20, 21]));
        assert!(
            !frame.state.rejection_q_valid,
            "external retrieval chain must invalidate stale GPU rejection q"
        );

        frame.state.rejection_q_valid = true;
        assert!(!install_collected(&source, &mut frame, Ok(Vec::new())).unwrap());
        assert!(
            !frame.state.rejection_q_valid,
            "empty proposal must invalidate GPU rejection q"
        );

        frame.state.rejection_q_valid = true;
        assert!(install_collected(&source, &mut frame, Err(anyhow::anyhow!("boom"))).is_err());
        assert!(
            !frame.state.rejection_q_valid,
            "failed proposal must invalidate GPU rejection q"
        );
    }
}
