// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_MTP_LASTK_DEFER` (K8 item A): run the MTP last-K prompt replay after the
//! first token instead of at the end of `finalize_last`.
//!
//! Today (serve recipe, `prefill_b/finalize_last.rs` step 8b): lm_head -> the replay
//! (`impl_b3.rs` `mtp_lastk_prefill_after_finalize_enabled`: 64 serial single-token
//! MTP-layer forwards over the prompt tail, logits saved before and restored after) ->
//! return -> scheduler: prefill handoff, first-token sample, emit, TTFT stamp. The
//! replay costs 30.7-32.5 ms per request [M: v3g server.log `MTP last-K prefill ...
//! took`; 31.9 ms of the 37.3 ms `finalize` phase in the 12-e1 serve census] and the
//! first token does not read anything it writes.
//!
//! Deferred: lm_head -> save `scratch[0..16]` -> return -> handoff, sample, emit, TTFT
//! stamp -> [`TransformerModel::k8a_finish_deferred_lastk`] (scheduler hook
//! `Model::finish_deferred_prefill_tail`): restore `scratch[0..16]`, the same replay
//! function on the same stream with the same inputs, ordered against the default stream
//! by two events (K9 item X, below; K8 A ended with a host sync of the prefill stream),
//! and -- only when the first token was sampled on the device -- the same
//! `argmax_on_device` again.
//!
//! Why the device state at the first decode step is unchanged:
//! 1. The device work the scheduler does between the two points is enumerated: the
//!    prefill handoff (event wait), `normalize_ssm_states` after a multi-chunk prompt's
//!    last chunk (its own pointer buffer + this slot's SSM h-states; the replay's MTP
//!    layer is attention + MoE with its own KV cache and reads neither), and the
//!    first-token sample: `copy_logits_to_host` (a D2H) on every current call site --
//!    they pass the EOS ids as `suppress_ids` (`sample_step.rs`) -- or
//!    `argmax_on_device` (writes `scratch[0..4]`) for a caller passing none.
//! 2. The logits the sample reads are identical in both orders (the replay restores them).
//! 3. The scratch head is restored before the replay, so the replay reads exactly what it
//!    read in `finalize_last`; the argmax write, when there was one, is redone after it.
//! 4. Only one sequence holds an SSM slot (decision guard, re-checked here), so no other
//!    prefill or decode runs on the shared buffers in between.
//!
//! A request that ends on its first token (max_tokens <= 1, EOS) never needs the replay;
//! `free_sequence` drops the record. Not covered by 1-4 and therefore loud: a scheduler
//! path that reaches the first propose without calling the hook (`..._LATE_REPLAY`).
//!
//! K9 item X (race-free ordering, same selector). The replay is enqueued on the prefill
//! stream and its consumers run on the default stream; both are non-blocking, so only an
//! event or a host wait orders them. K8 A's hook relied on the first-token sample's host
//! read to order the replay's start and on a trailing host drain to order its end; its
//! late fallback had no ordering at the start at all (a bootstrap decode just enqueued on
//! the default stream could overlap the replay). Now every replay -- hook or fallback --
//! runs [`host_bucket::replay_order_plan`]: record `in` on the default stream, the
//! prefill stream waits for it, the replay, record `out` on the prefill stream, the
//! default stream waits for it. No host wait: later default-stream work (first decode
//! step, propose, K5 verify frames and graphs, `free_sequence`) and later prefill-stream
//! work (the next request) are ordered by the device. A sequence whose replay never runs
//! still takes and returns the MTP KV blocks it would have allocated, so the next
//! request's MTP cache addresses equal the undeferred order's.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::host_bucket::{self, DeferDecision, DeferInputs, DeferredLastK, Lever, OrderStep};
use super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    fn k8a_logits_bytes(&self) -> usize {
        self.config.vocab_size * if self.use_fp32_logits { 4 } else { 2 }
    }

    /// Bytes per captured row, as `mtp_lastk_prefill_after_finalize_enabled` computes them.
    fn k8a_lastk_row_bytes(&self) -> usize {
        if self.config.is_qwen4_exp() {
            self.config.residual_width() * 2
        } else {
            let element_bytes = if self.config.use_fp32_residual() {
                4
            } else {
                2
            };
            self.config.hidden_size * element_bytes
        }
    }

    /// The scratch-head save slot: past the captured rows and the replay's saved logits
    /// inside `mtp_lastk_buf` (allocated as `capacity * row_bytes + vocab * 4`, `impl_a1.rs`).
    fn k8a_scratch_save_slot(&self) -> Option<DevicePtr> {
        let buf = self.mtp_lastk_buf?;
        Some(
            buf.offset(
                self.mtp_lastk_capacity * self.k8a_lastk_row_bytes() + self.k8a_logits_bytes(),
            ),
        )
    }

    /// Called first in `mtp_lastk_prefill_after_finalize`. True = the replay was
    /// recorded for [`Self::k8a_finish_deferred_lastk`] and must not run now.
    pub(super) fn k8a_defer_lastk(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Result<bool> {
        let armed = host_bucket::armed(Lever::MtpLastkDefer).map_err(anyhow::Error::msg)?;
        if !armed {
            return Ok(false);
        }
        // The replay's own early-outs (`run_lastk_if_enabled` and the enabled body).
        let rank_zero = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
        let replay_would_run = !self.speculation_forced_disabled
            && self.mtp_lastk_capacity > 0
            && self.active_proposer().is_some()
            && self.mtp_lastk_buf.is_some()
            && rank_zero
            && seq.mtp_lastk_host_filled > 0
            && !seq.mtp_lastk_host_buf.is_empty();
        let spare_bytes = (self.config.vocab_size * 4).saturating_sub(self.k8a_logits_bytes());
        let decision = host_bucket::defer_decision(&DeferInputs {
            armed,
            replay_would_run,
            claimed_sequences: self.ssm_pool.claimed_slots(),
            spare_bytes,
        });
        match decision {
            DeferDecision::Off => Ok(false),
            DeferDecision::Refused(reason) => {
                if host_bucket::first_time(reason) {
                    tracing::warn!(
                        "MTP_LASTK_DEFER_REFUSED selector={} reason={reason} (replay runs in finalize_last)",
                        Lever::MtpLastkDefer.selector()
                    );
                }
                Ok(false)
            }
            DeferDecision::Defer => {
                let save = self
                    .k8a_scratch_save_slot()
                    .ok_or_else(|| anyhow::anyhow!("MTP last-K defer: capture buffer missing"))?;
                // Stream-ordered after lm_head: the scratch head exactly as the replay
                // would have found it.
                self.gpu.copy_d2d_async(
                    self.buffers.scratch(),
                    save,
                    host_bucket::SCRATCH_SAVE_BYTES,
                    stream,
                )?;
                let mut pending = self.k8a_lastk_deferred.lock();
                if let Some(stale) = pending.take() {
                    tracing::warn!(
                        "MTP_LASTK_DEFER_STALE slot={} dropped (its sequence never reached the hook)",
                        stale.slot_idx
                    );
                }
                *pending = Some(DeferredLastK {
                    slot_idx: seq.slot_idx,
                    tokens: tokens.to_vec(),
                    stream,
                    speculation_forced_disabled: self.speculation_forced_disabled,
                });
                if host_bucket::first_time("MTP_LASTK_DEFER_ENGAGED") {
                    tracing::info!(
                        "MTP_LASTK_DEFER_ENGAGED selector={} capacity={} filled={} (replay after the first token)",
                        Lever::MtpLastkDefer.selector(),
                        self.mtp_lastk_capacity,
                        seq.mtp_lastk_host_filled,
                    );
                }
                Ok(true)
            }
        }
    }

    /// Scheduler hook body (`Model::finish_deferred_prefill_tail`). No-op unless a
    /// replay was recorded for this sequence. `device_first_token` = the first token
    /// when `argmax_on_device` sampled it.
    pub(super) fn k8a_finish_deferred_lastk(
        &self,
        seq: &mut SequenceState,
        device_first_token: Option<u32>,
        suppress_ids: &[u32],
    ) -> Result<()> {
        let Some(d) = self.k8a_take_deferred(seq.slot_idx) else {
            return Ok(());
        };
        if self.ssm_pool.claimed_slots() != Some(1)
            && host_bucket::first_time("MTP_LASTK_DEFER_INEXACT")
        {
            tracing::warn!(
                "MTP_LASTK_DEFER_INEXACT reason=other_sequences slot={} (replay runs, order not proven)",
                d.slot_idx
            );
        }
        self.k9x_run_deferred_ordered(&d, seq, "hook")?;
        if let Some(first) = device_first_token {
            // Redo the sample's scratch write after the replay, as in the original order.
            // The default stream waits for the replay (the plan's `out` event), so this
            // argmax reads the restored logits.
            let again = <Self as Model>::argmax_on_device(self, self.decode_logits_ptr(), 0)?;
            match host_bucket::classify_replay_argmax(first, again, suppress_ids) {
                host_bucket::ReplayArgmax::Match => {}
                // The raw argmax was a masked stop id and the sampler's host fallback
                // picked `first` (sample_step.rs): the logits are restored, the scratch
                // head holds the same raw argmax the undeferred order left there.
                host_bucket::ReplayArgmax::MaskedFirstToken => tracing::info!(
                    "MTP_LASTK_DEFER_FIRST_TOKEN_MASKED first={first} raw_argmax={again} (stop id masked on the first token)"
                ),
                host_bucket::ReplayArgmax::Mismatch => tracing::error!(
                    "MTP_LASTK_DEFER_ARGMAX_MISMATCH first={first} after_replay={again} (logits not restored?)"
                ),
            }
        }
        Ok(())
    }

    /// Fallback at the first propose: a path that skipped the hook still gets the
    /// replay before the proposer state is read, with a WARN (a scheduler route that
    /// emits a first token without the hook is a bug; every current route calls it).
    /// Ordered like the hook, so it cannot overlap the bootstrap decode just enqueued on
    /// the default stream nor the propose after it.
    pub(super) fn k8a_late_replay_if_pending(&self, seq: &mut SequenceState) -> Result<()> {
        let Some(d) = self.k8a_take_deferred(seq.slot_idx) else {
            return Ok(());
        };
        tracing::warn!(
            "MTP_LASTK_DEFER_LATE_REPLAY slot={} (the scheduler path did not call finish_deferred_prefill_tail; replay ordered by events)",
            d.slot_idx
        );
        self.k9x_run_deferred_ordered(&d, seq, "late")
    }

    /// `free_sequence`: a sequence that ends on its first token (or is cancelled or fails
    /// before the hook) never runs its replay. Its MTP proposer state still takes, host
    /// side only, the KV blocks the replay would have allocated; `free_sequence` frees
    /// the proposer state after this, so the MTP cache's free list ends in the same order
    /// as when the replay ran in `finalize_last` (allocation pops, the free pushes back in
    /// table order). No device work.
    pub(super) fn k8a_drop_deferred_lastk(&self, seq: &mut SequenceState) {
        let Some(d) = self.k8a_take_deferred(seq.slot_idx) else {
            return;
        };
        if d.speculation_forced_disabled {
            return;
        }
        let rows = host_bucket::mtp_replay_rows(
            seq.mtp_lastk_host_filled,
            crate::speculative::mtp_replay_next_token(),
        );
        let Some(proposer) = self.active_proposer() else {
            return;
        };
        if let Some(ref mut pstate) = seq.proposer_state {
            match proposer.reserve_prompt_replay_blocks(rows, pstate.as_mut()) {
                Ok(blocks) => {
                    if host_bucket::first_time("MTP_LASTK_DEFER_DROPPED") {
                        tracing::info!(
                            "MTP_LASTK_DEFER_DROPPED slot={} rows={rows} blocks_reserved={blocks} (replay not needed; MTP cache allocation mirrored)",
                            d.slot_idx
                        );
                    }
                }
                Err(e) => tracing::warn!(
                    "MTP_LASTK_DEFER_DROPPED slot={} rows={rows}: MTP cache allocation not mirrored: {e:#}",
                    d.slot_idx
                ),
            }
        }
    }

    /// `alloc_sequence`: a new sequence while a replay is still pending means a scheduler
    /// route admitted a request between a `finalize_last` that deferred and its hook.
    /// None does today (the record needs a single live sequence and every first-token
    /// route runs the hook in the same scheduler call). Loud, not fatal: the replay stays
    /// ordered (same prefill stream, `out` event), but its exactness argument -- no other
    /// sequence's device work in between -- no longer holds.
    pub(super) fn k9x_note_alloc_while_pending(&self) {
        let pending = self.k8a_lastk_deferred.lock();
        if let Some(d) = pending.as_ref()
            && host_bucket::first_time("MTP_LASTK_DEFER_INEXACT_ALLOC")
        {
            tracing::warn!(
                "MTP_LASTK_DEFER_INEXACT reason=alloc_while_pending slot={} (a sequence was allocated before the deferred replay ran)",
                d.slot_idx
            );
        }
    }

    fn k8a_take_deferred(&self, slot_idx: usize) -> Option<DeferredLastK> {
        let mut pending = self.k8a_lastk_deferred.lock();
        if host_bucket::pending_matches(pending.as_ref(), slot_idx) {
            pending.take()
        } else {
            None
        }
    }

    /// The two events of [`host_bucket::replay_order_plan`], created on first use (after
    /// the first token, never on the TTFT path) and kept for the lifetime of the model
    /// (destroyed in `TransformerModel::drop`). `None` when the backend has none; the
    /// plan then orders with host drains.
    fn k9x_order_events(&self) -> Option<(u64, u64)> {
        *self.k9x_lastk_events.get_or_init(|| {
            let make = || self.gpu.create_event().ok().filter(|&e| e != 0);
            match (make(), make()) {
                (Some(ein), Some(eout)) if ein != eout => Some((ein, eout)),
                (a, b) => {
                    for e in [a, b].into_iter().flatten() {
                        let _ = self.gpu.destroy_event(e);
                    }
                    tracing::warn!(
                        "MTP_LASTK_DEFER_ORDER_FALLBACK reason=no_events (host drains order the deferred replay)"
                    );
                    None
                }
            }
        })
    }

    /// The replay body on `stream`: restore the scratch head saved in `finalize_last`,
    /// then the same replay function with the same inputs. Enqueue only; the caller orders it.
    fn k8a_replay_body(
        &self,
        d: &DeferredLastK,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        let save = self
            .k8a_scratch_save_slot()
            .ok_or_else(|| anyhow::anyhow!("MTP last-K defer: capture buffer missing"))?;
        self.gpu.copy_d2d_async(
            save,
            self.buffers.scratch(),
            host_bucket::SCRATCH_SAVE_BYTES,
            stream,
        )?;
        super::speculation_control::run_lastk_if_enabled(d.speculation_forced_disabled, || {
            self.mtp_lastk_prefill_after_finalize_enabled(&d.tokens, seq, 0, 0, 0, stream)
        })?;
        Ok(())
    }

    /// Enqueue the deferred replay on its recorded stream, after everything already
    /// enqueued on the default stream and before everything enqueued there later
    /// ([`host_bucket::replay_order_plan`]). No host wait on the event path. A failed
    /// event call falls back to host drains (same order); a failed replay drains its
    /// stream before the error returns, so the free path cannot overlap partial work.
    fn k9x_run_deferred_ordered(
        &self,
        d: &DeferredLastK,
        seq: &mut SequenceState,
        site: &'static str,
    ) -> Result<()> {
        let consumer = self.gpu.default_stream();
        let events = if d.stream == consumer {
            None
        } else {
            self.k9x_order_events()
        };
        let plan = host_bucket::replay_order_plan(d.stream, consumer, events);
        let mut replayed = false;
        for step in &plan {
            let result = match *step {
                OrderStep::Record { event, stream } => self.gpu.record_event(event, stream),
                OrderStep::Wait { stream, event } => self.gpu.stream_wait_event(stream, event),
                OrderStep::Drain { stream } => {
                    self.gpu.synchronize(stream)?;
                    Ok(())
                }
                OrderStep::Replay { stream } => {
                    if let Err(e) = self.k8a_replay_body(d, seq, stream) {
                        if let Err(drain) = self.gpu.synchronize(stream) {
                            tracing::error!(
                                "MTP last-K defer: drain after a failed replay also failed: {drain:#}"
                            );
                        }
                        return Err(e);
                    }
                    replayed = true;
                    Ok(())
                }
            };
            if let Err(e) = result {
                // An event call failed: host drains give the same order.
                tracing::warn!(
                    "MTP_LASTK_DEFER_ORDER_FALLBACK reason=event_call_failed site={site} step={step:?}: {e:#}"
                );
                if !replayed {
                    self.gpu.synchronize(consumer)?;
                    let replay = self.k8a_replay_body(d, seq, d.stream);
                    let drain = self.gpu.synchronize(d.stream);
                    replay?;
                    return drain;
                }
                return self.gpu.synchronize(d.stream);
            }
        }
        let receipt_key = match site {
            "hook" => "MTP_LASTK_DEFER_ORDERED_HOOK",
            _ => "MTP_LASTK_DEFER_ORDERED_LATE",
        };
        if host_bucket::first_time(receipt_key) {
            let label = host_bucket::plan_label(&plan);
            let wait_note = if label == "events" {
                " (events: no host wait)"
            } else {
                ""
            };
            tracing::info!(
                "MTP_LASTK_DEFER_ORDERED selector={} site={site} plan={label} replay_stream={:#x} consumer_stream={:#x}{wait_note}",
                Lever::MtpLastkDefer.selector(),
                d.stream,
                consumer,
            );
        }
        Ok(())
    }
}
