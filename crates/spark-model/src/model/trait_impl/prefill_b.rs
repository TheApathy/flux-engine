// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_dispatch` orchestrator.
//!
//! Refactor wave-4e split a 1000-LoC monolith into Pattern-B phase fns
//! (siblings under `prefill_b/`). The MutexGuard on `kv_cache` is
//! acquired here once and threaded through each phase as `&mut`.
//!
//! Phases (by section comment in original):
//!   1+1b → embed_chunk     (token embed + vision-pad overlay)
//!   2    → prefix_lookup   (prefix-cache hit + EP-sync + Marconi)
//!   2b   → proc_range      (recompute proc_start/count after skip; may early-return)
//!   3    → upload_meta     (positions + MRoPE + slots staging upload)
//!   3b   → upload_paged    (paged-prefill block_table + seq_len upload)
//!   4    → forward_layers  (per-layer prefill/decode + diagnostics)
//!   5-8  → finalize_last   (final norm + lm_head + snapshot save) — last chunk
//!   9    → save_intermediate_checkpoint — non-last chunk

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use super::ple_prefetch_slots;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod prefix_lookup;
#[cfg(test)]
mod prefix_lookup_tests;
mod proc_range;
mod qgate_dump;
mod save_checkpoint;
mod stage_batched;
mod upload_meta;
mod upload_paged;

/// Whether an SSM snapshot may seed this prefill.
///
/// An exact full-prompt snapshot contains recurrent state *after* the last
/// prompt token. Replaying that token to obtain logits would advance the SSM a
/// second time, so exact hits default to a full recurrent recompute. The
/// partial-prefix path remains safe and enabled. Keep the opt-in only for
/// controlled A/B comparisons.
pub(in crate::model) fn should_restore_ssm_snapshot(
    snapshot_tokens: usize,
    matched_tokens: usize,
    total_tokens: usize,
) -> bool {
    should_restore_ssm_snapshot_with_opt_in(
        snapshot_tokens,
        matched_tokens,
        total_tokens,
        std::env::var("ATLAS_MARCONI_EXACT").as_deref() == Ok("1"),
    )
}

fn should_restore_ssm_snapshot_with_opt_in(
    snapshot_tokens: usize,
    matched_tokens: usize,
    total_tokens: usize,
    exact_opt_in: bool,
) -> bool {
    snapshot_tokens != matched_tokens || matched_tokens != total_tokens || exact_opt_in
}

/// Prefix length whose model state is actually reusable after lookup.
///
/// KV blocks can match deeper than the surviving SSM checkpoint. Recurrent
/// execution must resume at the checkpoint, not at the KV match depth.
pub(in crate::model) fn restored_prefix_skip_tokens(
    has_ssm: bool,
    snapshot_tokens: usize,
    matched_tokens: usize,
) -> usize {
    if has_ssm {
        snapshot_tokens
    } else {
        matched_tokens
    }
}

/// F82 is an attention-only shortcut. Hybrid models may deliberately decline
/// an otherwise paired snapshot (for example the exact-full replay hazard),
/// but that must never turn into a KV-only skip with unrestored recurrent
/// state.
pub(in crate::model) fn attention_only_prefix_skip(
    matched_tokens: usize,
    snapshot_restored: bool,
    has_ssm: bool,
) -> bool {
    matched_tokens > 0 && !snapshot_restored && !has_ssm
}

/// Number of leading rows in the current processing slice whose KV entries
/// already belong to the prefix cache and must not be overwritten.
pub(in crate::model) fn cached_kv_rows_in_slice(
    cached_prefix_tokens: usize,
    process_start: usize,
    process_count: usize,
) -> usize {
    cached_prefix_tokens
        .saturating_sub(process_start)
        .min(process_count)
}

/// `ATLAS_QWEN4_PREFILL_SYNC_DIAG` (fn-decode diagnostic, default off): `entry` =
/// device-wide synchronize (`cuCtxSynchronize`, every stream incl. legacy and
/// helper streams) at the start of every prefill chunk; `layers` = that plus one
/// after every prefill layer. Isolates the fnd4/fnd5 short-prompt corruption
/// (2nd+ identical 64-token request prefills a wrong first token): fixed by
/// `entry` => work from the previous request is still in flight; fixed only by
/// `layers` => an intra-prefill cross-stream race; fixed by neither => stale state.
pub(crate) fn prefill_sync_diag() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = match std::env::var("ATLAS_QWEN4_PREFILL_SYNC_DIAG").as_deref() {
            Ok("entry") => 1,
            Ok("layers") => 2,
            _ => 0,
        };
        if level > 0 {
            tracing::warn!(
                "ENGAGED ATLAS_QWEN4_PREFILL_SYNC_DIAG level={level} (diagnostic: device-wide syncs in prefill)"
            );
        }
        level
    })
}

impl TransformerModel {
    /// Launch the following scheduler chunk's PLE read while the current
    /// chunk still has almost its entire transformer stack left to execute.
    /// Engages at `stream_level >= 2` (default-on level 2) whenever fast
    /// lineage is admitted and `chunk_start + chunk_len < total_tokens`,
    /// including short final tails.
    pub(super) fn begin_cross_chunk_ple_prefetch(
        &self,
        ple: &crate::layers::qwen4_ple::Qwen4PleLayer,
        tokens: &[u32],
        slot_idx: usize,
        chunk_start: usize,
        chunk_len: usize,
        fast_lineage_admitted: bool,
    ) -> Result<()> {
        if !should_begin_cross_chunk_ple_prefetch(
            crate::layers::qwen4_ple::prefill_stream_level(),
            fast_lineage_admitted,
            chunk_start,
            chunk_len,
            tokens.len(),
        ) {
            return Ok(());
        }
        let next_start = chunk_start.saturating_add(chunk_len);
        if next_start >= tokens.len() {
            return Ok(());
        }
        let next_len = chunk_len.min(tokens.len() - next_start);
        let next_end = next_start + next_len;
        let prior = &tokens[..next_start];
        let current = &tokens[next_start..next_end];

        {
            let pending = self.qwen4_ple_prefetch.lock();
            anyhow::ensure!(
                !pending.contains_key(&slot_idx),
                "Qwen4 PLE slot {slot_idx} already owns an unconsumed cross-chunk prefetch"
            );
        }
        let key = crate::layers::qwen4_ple::PlePrefetchKey::new(next_start, current, prior);
        let read = ple.begin_prefill_read(current, prior);
        let pending = crate::layers::qwen4_ple::PleChunkPrefetch::new(key, read);
        self.qwen4_ple_prefetch.lock().insert(slot_idx, pending);
        tracing::info!(
            target: "atlas::qwen4_ple",
            slot_idx,
            chunk_start = next_start,
            chunk_len = next_len,
            "QWEN4_PLE_CROSS_CHUNK_PREFETCH_STARTED"
        );
        Ok(())
    }

    /// `ATLAS_QWEN4_PLE_EARLY_ISSUE=1` (K9 item W, default off): start the
    /// first chunk's PLE row read at admission, before the scheduler allocates
    /// the sequence slot (SSM state zeroing + host syncs, ~10 ms per request),
    /// so the read overlaps it. Held under `QWEN4_PLE_ADMISSION_SLOT` in the
    /// prefetch map; chunk 0 consumes it only when its exact tokens match
    /// (the same rows are read either way). Same preconditions as chunk 0's
    /// own read: `ATLAS_QWEN4_PLE_PREFILL_BATCH=1` and stream level >= 2.
    pub(super) fn begin_ple_admission_prefetch(&self, tokens: &[u32], first_chunk_len: usize) {
        if !ple_early_issue_active() {
            return;
        }
        let Some((ple, current)) = self.ple_chunk_zero_tokens(tokens, first_chunk_len) else {
            return;
        };
        // A read nobody consumed (its request failed before chunk 0, e.g. in
        // alloc_sequence) is dropped, not joined: its reader thread finishes on
        // its own (a published block nobody receives is discarded by the read's
        // sink) and holds the reader lock only until then, so the scheduler
        // thread never waits on it here; the new read queues behind it inside
        // its own thread.
        let route =
            ple_prefetch_slots::route_admission(&mut self.qwen4_ple_prefetch.lock(), current);
        match route {
            ple_prefetch_slots::AdmissionRoute::AlreadyIssued => return,
            ple_prefetch_slots::AdmissionRoute::AdoptedQueued => {
                tracing::info!(
                    target: "atlas::qwen4_ple",
                    chunk_len = current.len(),
                    "QWEN4_PLE_QUEUE_PREFETCH_ADOPTED"
                );
                return;
            }
            ple_prefetch_slots::AdmissionRoute::Start { discarded } => {
                if discarded {
                    tracing::info!(
                        target: "atlas::qwen4_ple",
                        "QWEN4_PLE_ADMISSION_PREFETCH_DISCARDED reason=unconsumed"
                    );
                }
            }
        }
        let key = crate::layers::qwen4_ple::PlePrefetchKey::new(0, current, &[]);
        let read = ple.begin_prefill_read(current, &[]);
        self.qwen4_ple_prefetch.lock().insert(
            QWEN4_PLE_ADMISSION_SLOT,
            crate::layers::qwen4_ple::PleChunkPrefetch::new(key, read),
        );
        crate::layers::qwen4_ple_pipeline::log_once(format!(
            "QWEN4_PLE_EARLY_ISSUE_ENGAGED selector={PLE_EARLY_ISSUE_SELECTOR} point=admission_before_alloc_sequence"
        ));
        tracing::info!(
            target: "atlas::qwen4_ple",
            chunk_len = current.len(),
            "QWEN4_PLE_ADMISSION_PREFETCH_STARTED"
        );
    }

    /// `ATLAS_QWEN4_PLE_QUEUE_PREFETCH=1` (default off): the scheduler hands
    /// the pending-queue head's tokens here while another request runs, so
    /// chunk 0's PLE read overlaps that request instead of starting at this
    /// one's admission. The read lives in the single queue slot; admission
    /// adopts it only for the same tokens (`ple_prefetch_slots`). It yields
    /// the reader lock per block, so the running request's decode reads wait
    /// one block at most, and a replaced read stops at the block it is on.
    pub(super) fn begin_ple_queue_prefetch(&self, tokens: &[u32], first_chunk_len: usize) {
        if !ple_queue_prefetch_active() {
            return;
        }
        let Some((ple, current)) = self.ple_chunk_zero_tokens(tokens, first_chunk_len) else {
            return;
        };
        let route = ple_prefetch_slots::route_queue(&mut self.qwen4_ple_prefetch.lock(), current);
        match route {
            ple_prefetch_slots::QueueRoute::AlreadyIssued => return,
            ple_prefetch_slots::QueueRoute::Start { replaced } => {
                if replaced {
                    tracing::info!(
                        target: "atlas::qwen4_ple",
                        "QWEN4_PLE_QUEUE_PREFETCH_DISCARDED reason=head_changed"
                    );
                }
            }
        }
        let Some(read) = ple.begin_prefill_read_yielding(current, &[]) else {
            crate::layers::qwen4_ple_pipeline::log_once(format!(
                "QWEN4_PLE_QUEUE_PREFETCH_REFUSED selector={PLE_QUEUE_PREFETCH_SELECTOR} reason=needs_ATLAS_QWEN4_PLE_PREFILL_PIPELINE=1"
            ));
            return;
        };
        let key = crate::layers::qwen4_ple::PlePrefetchKey::new(0, current, &[]);
        self.qwen4_ple_prefetch.lock().insert(
            ple_prefetch_slots::QUEUE_SLOT,
            crate::layers::qwen4_ple::PleChunkPrefetch::new(key, read),
        );
        crate::layers::qwen4_ple_pipeline::log_once(format!(
            "QWEN4_PLE_QUEUE_PREFETCH_ENGAGED selector={PLE_QUEUE_PREFETCH_SELECTOR} point=pending_queue_head lock=per_block_fair"
        ));
        tracing::info!(
            target: "atlas::qwen4_ple",
            chunk_len = current.len(),
            "QWEN4_PLE_QUEUE_PREFETCH_STARTED"
        );
    }

    /// The PLE layer and chunk 0's tokens when an early read may start:
    /// `ATLAS_QWEN4_PLE_PREFILL_BATCH=1`, stream level >= 2 and a non-empty
    /// chunk (the same preconditions as chunk 0's own read).
    fn ple_chunk_zero_tokens<'t>(
        &self,
        tokens: &'t [u32],
        first_chunk_len: usize,
    ) -> Option<(&crate::layers::Qwen4PleLayer, &'t [u32])> {
        let ple = self.qwen4_ple.as_ref()?;
        if crate::layers::qwen4_ple::prefill_stream_level() < 2
            || std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
                .ok()
                .as_deref()
                != Some("1")
        {
            crate::layers::qwen4_ple_pipeline::log_once(format!(
                "QWEN4_PLE_EARLY_ISSUE_REFUSED selector={PLE_EARLY_ISSUE_SELECTOR} reason=needs_ATLAS_QWEN4_PLE_PREFILL_BATCH=1_and_stream_level>=2"
            ));
            return None;
        }
        let len = first_chunk_len.min(tokens.len());
        (len > 0).then(|| (ple, &tokens[..len]))
    }

    /// The admission read for chunk 0, if one is pending and was issued for
    /// exactly these tokens; a mismatching one is retired and `None` returned.
    fn take_ple_admission_prefetch(
        &self,
        current: &[u32],
        prior: &[u32],
    ) -> Option<crate::layers::qwen4_ple::PlePrefillRead> {
        if !ple_early_issue_selected() {
            return None;
        }
        let taken = if prior.is_empty() {
            ple_prefetch_slots::take_for_chunk_zero(&mut self.qwen4_ple_prefetch.lock(), current)?
        } else {
            Err(self
                .qwen4_ple_prefetch
                .lock()
                .remove(&QWEN4_PLE_ADMISSION_SLOT)?)
        };
        let pending = match taken {
            Ok(pending) => {
                tracing::info!(
                    target: "atlas::qwen4_ple",
                    chunk_len = current.len(),
                    "QWEN4_PLE_ADMISSION_PREFETCH_CONSUMED"
                );
                return Some(pending.into_read());
            }
            Err(stale) => stale,
        };
        if let Err(error) = pending.finish_discard() {
            tracing::warn!(
                target: "atlas::qwen4_ple",
                error = %format!("{error:#}"),
                "discarded mismatching Qwen4 PLE admission read with an I/O failure"
            );
        }
        tracing::warn!(
            target: "atlas::qwen4_ple",
            "QWEN4_PLE_ADMISSION_PREFETCH_DISCARDED reason=token_mismatch; using a fresh read"
        );
        None
    }

    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        if prefill_sync_diag() >= 1 {
            self.gpu.synchronize_device()?;
        }
        let total = tokens.len();
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .filter(|&end| end <= total)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "prefill chunk [{chunk_start}, +{chunk_len}) exceeds prompt length {total}"
                )
            })?;
        anyhow::ensure!(
            chunk_len > 0,
            "prefill chunk must contain at least one token"
        );
        self.validate_vision_prompt(tokens, chunk_start, chunk_len)?;
        let (block_size, high_speed_swap) = {
            let kv_cache = self.kv_cache.lock();
            (
                kv_cache.block_size(),
                kv_cache.config().cache_blocks_per_seq.is_some(),
            )
        };
        let _interleaved_2k_guard =
            crate::layers::qwen4_prefill_moe::ChunkRowsOverrideGuard::maybe_enter_interleaved_2k(
                chunk_start,
                chunk_len,
                is_last_chunk,
            );
        let has_vision = self.vision_prompt_present(tokens);
        let marconi_fast_cont =
            crate::layers::qwen4_prefill_moe::attn16::marconi_fast_cont_enabled();
        let lineage_selected = crate::layers::qwen4_prefill_moe::multi_chunk_selected()?;
        let surface_admission = match crate::layers::qwen4_prefill_moe::attn16::admit_or_suppress(
            has_vision,
            seq.seq_len,
            chunk_start,
            chunk_len,
            total,
            is_last_chunk,
            || {
                crate::layers::qwen4_prefill_moe::attn16::request_chain_matches(
                    tokens,
                    &seq.tokens,
                    seq.seq_len,
                    seq.prompt_len,
                    chunk_start,
                )
            },
            || {
                let page_meta = seq.chunked_prefill_meta.as_ref().map(|meta| {
                    (
                        meta.block_capacity,
                        meta.uploaded_blocks,
                        !high_speed_swap
                            && seq.hss_window_start() == 0
                            && !meta.block_table.is_null()
                            && !meta.seq_len.is_null(),
                    )
                });
                if marconi_fast_cont && seq.marconi_skip_to > 0 {
                    crate::layers::qwen4_prefill_moe::attn16::continuation_state_ready_for_marconi(
                        chunk_start,
                        total,
                        block_size,
                        seq.block_table.len(),
                        seq.marconi_skip_to,
                        page_meta,
                    )
                } else {
                    crate::layers::qwen4_prefill_moe::attn16::continuation_state_ready(
                        chunk_start,
                        total,
                        block_size,
                        seq.block_table.len(),
                        page_meta,
                    )
                }
            },
            high_speed_swap,
        ) {
            Ok(admission) => admission,
            Err(error) => {
                if lineage_selected {
                    crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                        &mut seq.qwen4_prefill_lineage,
                        total,
                    );
                }
                return Err(error);
            }
        };
        let mut lineage_ticket = lineage_selected
            .then(|| {
                crate::layers::qwen4_prefill_moe::attn16::begin_fast_lineage(
                    &mut seq.qwen4_prefill_lineage,
                    tokens,
                    chunk_start,
                    chunk_len,
                    surface_admission.admitted(),
                )
            })
            .flatten();
        let mut cross_chunk_ple_prefetch_admitted = chunk_len > 0 && tokens.len() > chunk_len;
        // A surface can be valid in isolation yet lack the persistent proof
        // that its prior chunk actually completed on the fast route. Suppress
        // before rotary, embedding, KV allocation, or recurrent-state work.
        let _lineage_suppressed = (lineage_selected
            && lineage_ticket.is_none()
            && matches!(
                &seq.qwen4_prefill_lineage,
                crate::traits::Qwen4PrefillLineage::Ordinary
            )
            && surface_admission.admitted())
        .then(crate::layers::qwen4_prefill_moe::attn16::suppress_family);
        self.admit_rotary_prompt(tokens, seq, chunk_start)?;
        if crate::layers::qwen4_prefill_moe::multi_chunk_selected()? {
            crate::layers::qwen4_prefill_moe::admit_request(&self.config, chunk_len, chunk_start)?;
        } else {
            crate::layers::qwen4_prefill_moe::admit_request(&self.config, total, 0)?;
        }
        if chunk_start == 0 && crate::layers::qwen4_prefill_moe::attn16::selected()? {
            let kv_cache = self.kv_cache.lock();
            for layer in &self.layers {
                layer.preflight_qwen4_attn16(&kv_cache, self.gpu.as_ref(), stream)?;
            }
        }
        // Guard: chunk_len must not exceed buffer arena capacity.
        // Exceeding this causes CUDA illegal memory access (error 700)
        // which permanently corrupts GPU state.
        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        // Use the caller-provided stream for compute-copy overlap,
        // unless multi-rank (EP or TP) is active (NCCL requires the default stream).
        let stream = if self.multi_rank_prefill_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // EP=2: zero ALL buffers on every chunk (NCCL defense-in-depth).
        // EP=1, first chunk (chunk_start==0): zero essentials (stale data from prior request).
        // EP=1, subsequent chunks: skip zeroing — buffers are overwritten by embedding
        // + layer forward before read. Saves 7 memsets × (chunks-1) per prefill.
        let mut skipped_fill: Option<[Option<u8>; 19]> = None;
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            use crate::model::host_bucket::{ZERO_ALL_SELECTOR, first_time, zero_all_plan_for};
            let default_skip_data = self.config.is_qwen4_exp()
                && crate::layers::qwen4_prefill_moe::selected()?
                && !crate::layers::qwen4_prefill_moe::family_suppressed()
                && seq.seq_len == 0
                && !has_vision
                && chunk_len >= 2
                && total >= 2;
            let names = &spark_runtime::buffers::BufferArena::ZERO_ALL_BUFFERS;
            match zero_all_plan_for(names, default_skip_data).map_err(anyhow::Error::msg)? {
                None => self.buffers.zero_all(self.gpu.as_ref(), stream)?,
                Some(plan) => {
                    let fill: [Option<u8>; 19] = plan
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("{ZERO_ALL_SELECTOR} plan length"))?;
                    if default_skip_data && fill.iter().any(Option::is_none) {
                        skipped_fill = Some(fill);
                    }
                    let (written, untouched) =
                        self.buffers
                            .zero_all_with(self.gpu.as_ref(), stream, &fill)?;
                    if first_time("PREFILL_ZERO_ALL_PLAN_ENGAGED") {
                        tracing::info!(
                            "PREFILL_ZERO_ALL_PLAN_ENGAGED selector={ZERO_ALL_SELECTOR} plan={} written_bytes={written} untouched_bytes={untouched}",
                            std::env::var(ZERO_ALL_SELECTOR)
                                .ok()
                                .filter(|s| !s.is_empty())
                                .unwrap_or_else(|| "default_skip_data".to_string()),
                        );
                    }
                }
            }
        }

        let mut kv_cache = self.kv_cache.lock();

        // Env-gated phase profiler: ATLAS_PREFILL_PHASE_PROFILE=1 prints
        // CPU-wall ms per prefill phase (host time between markers; GPU work
        // is async unless the phase syncs).
        let phase_on = crate::full_profile::phase_profile_enabled();
        let phase_t0 = std::time::Instant::now();
        let mut phase_ms: Vec<(String, f64)> = Vec::new();
        macro_rules! phase_mark {
            ($name:expr) => {
                if phase_on {
                    phase_ms.push(($name.to_string(), phase_t0.elapsed().as_secs_f64() * 1000.0));
                }
            };
        }

        // ── Phase 1: prefix-cache lookup + EP sync + Marconi snapshot restore ──
        // Run BEFORE PLE O_DIRECT row reads and chunk embedding so cached chunks
        // early-return with zero NVMe I/O or GPU work, and partially cached Qwen4
        // chunks read/embed only the uncached suffix on the ordinary continuation path.
        let (kv_write_start, marconi_skip) =
            self.prefill_b_prefix_lookup(tokens, seq, chunk_start, total, &mut kv_cache, stream)?;
        phase_mark!("prefix_lookup");

        // Allocate blocks needed through end of this chunk.
        let bs = kv_cache.block_size();
        let end_pos = chunk_end;
        let blocks_needed = (end_pos - 1) / bs + 1;
        super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;
        phase_mark!("blocks");

        let seq_used_marconi_skip = marconi_skip && kv_write_start > 0;
        if seq_used_marconi_skip && self.config.is_qwen4_exp() && chunk_start == 0 {
            // Invalidate QSA FP8 shadow continuity on Marconi skip so the resumed
            // suffix always refills [0, start) from the restored BF16 KV pages.
            crate::layers::qwen4_qsa::prefill_fp8::invalidate_shadow_continuity(&kv_cache)?;
        }
        let marconi_fast_cont_base = seq_used_marconi_skip
            && self.config.is_qwen4_exp()
            && marconi_fast_cont
            && surface_admission.admitted()
            && !crate::layers::qwen4_prefill_moe::family_suppressed()
            && !has_vision
            && !high_speed_swap
            && seq.hss_window_start() == 0
            && bs > 0
            && bs.is_multiple_of(4)
            && kv_write_start.is_multiple_of(bs);

        if let Some(prev_fill) = skipped_fill
            && (seq_used_marconi_skip || crate::layers::qwen4_prefill_moe::family_suppressed())
        {
            let backfill = prev_fill.map(|v| if v.is_none() { Some(0) } else { None });
            self.buffers
                .zero_all_with(self.gpu.as_ref(), stream, &backfill)?;
        }

        let chunk_has_marconi_skip = marconi_skip && kv_write_start > chunk_start;
        let ple_stream_batch = crate::layers::qwen4_ple::prefill_stream_level() >= 2
            && std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
                .ok()
                .as_deref()
                == Some("1");

        let (ple_prefetch, proc_start, proc_count, effective_seq_len_start) =
            if chunk_has_marconi_skip {
                if chunk_start == 0
                    && let Some(stale) = self
                        .qwen4_ple_prefetch
                        .lock()
                        .remove(&QWEN4_PLE_ADMISSION_SLOT)
                    && let Err(error) = stale.finish_discard()
                {
                    tracing::warn!(
                        target: "atlas::qwen4_ple",
                        error = %format!("{error:#}"),
                        "discarded Qwen4 PLE admission read after prefix-cache hit"
                    );
                }
                if let Some(stale) = self.qwen4_ple_prefetch.lock().remove(&seq.slot_idx)
                    && let Err(error) = stale.finish_discard()
                {
                    tracing::warn!(
                        target: "atlas::qwen4_ple",
                        slot_idx = seq.slot_idx,
                        error = %format!("{error:#}"),
                        "discarded Qwen4 PLE cross-chunk read after prefix-cache hit"
                    );
                }
                let (proc_start, proc_count, effective_seq_len_start) = match self
                    .prefill_b_proc_range(
                        tokens,
                        seq,
                        chunk_start,
                        chunk_len,
                        is_last_chunk,
                        kv_write_start,
                        marconi_skip,
                        stream,
                    )? {
                    proc_range::ProcRange::Compute {
                        proc_start,
                        proc_count,
                        effective_seq_len_start,
                    } => (proc_start, proc_count, effective_seq_len_start),
                    proc_range::ProcRange::EarlyReturn(ptr) => {
                        if marconi_fast_cont_base && let Some(ticket) = lineage_ticket.take() {
                            self.upload_marconi_prefix_blocks_for_early_return(
                                seq, total, bs, stream,
                            )?;
                            crate::layers::qwen4_prefill_moe::attn16::advance_fast_lineage_for_marconi_skip(
                                &mut seq.qwen4_prefill_lineage,
                                ticket,
                                tokens,
                                chunk_start,
                                chunk_end,
                            )?;
                        } else if seq_used_marconi_skip && self.config.is_qwen4_exp() {
                            crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                                &mut seq.qwen4_prefill_lineage,
                                total,
                            );
                        }
                        return Ok(ptr);
                    }
                };
                phase_mark!("proc_range");
                let ple_prefetch = match &self.qwen4_ple {
                    Some(ple) if ple_stream_batch && proc_count > 0 => {
                        let current =
                            &tokens[effective_seq_len_start..effective_seq_len_start + proc_count];
                        let prior = &tokens[..effective_seq_len_start];
                        Some((
                            effective_seq_len_start,
                            proc_count,
                            ple.begin_prefill_read(current, prior),
                        ))
                    }
                    _ => None,
                };
                (
                    ple_prefetch,
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                )
            } else {
                // PLE level 2: start this chunk's O_DIRECT row reads now, so they
                // overlap embedding, metadata setup and layer 0. Used only if the chunk
                // ends up processing exactly [chunk_start, chunk_start + chunk_len).
                let ple_prefetch = match &self.qwen4_ple {
                    Some(ple) if ple_stream_batch => {
                        let current = &tokens[chunk_start..chunk_start + chunk_len];
                        let prior = &tokens[..chunk_start];
                        let admitted = if chunk_start == 0 {
                            self.take_ple_admission_prefetch(current, prior)
                        } else {
                            None
                        };
                        let read = if let Some(read) = admitted {
                            if let Some(stale) =
                                self.qwen4_ple_prefetch.lock().remove(&seq.slot_idx)
                                && let Err(error) = stale.finish_discard()
                            {
                                tracing::warn!(
                                    target: "atlas::qwen4_ple",
                                    slot_idx = seq.slot_idx,
                                    error = %format!("{error:#}"),
                                    "discarded stale Qwen4 PLE cross-chunk read with an I/O failure"
                                );
                            }
                            read
                        } else {
                            match self.qwen4_ple_prefetch.lock().remove(&seq.slot_idx) {
                                Some(pending) if pending.matches(chunk_start, current, prior) => {
                                    tracing::info!(
                                        target: "atlas::qwen4_ple",
                                        slot_idx = seq.slot_idx,
                                        chunk_start,
                                        chunk_len,
                                        "QWEN4_PLE_CROSS_CHUNK_PREFETCH_CONSUMED"
                                    );
                                    pending.into_read()
                                }
                                Some(stale) => {
                                    if let Err(error) = stale.finish_discard() {
                                        tracing::warn!(
                                            target: "atlas::qwen4_ple",
                                            slot_idx = seq.slot_idx,
                                            error = %format!("{error:#}"),
                                            "discarded stale Qwen4 PLE cross-chunk read with an I/O failure"
                                        );
                                    }
                                    tracing::warn!(
                                        target: "atlas::qwen4_ple",
                                        slot_idx = seq.slot_idx,
                                        chunk_start,
                                        chunk_len,
                                        "Qwen4 PLE cross-chunk token identity mismatch; using a fresh read"
                                    );
                                    ple.begin_prefill_read(current, prior)
                                }
                                None => ple.begin_prefill_read(current, prior),
                            }
                        };
                        Some((chunk_start, chunk_len, read))
                    }
                    _ => None,
                };

                // ── Phase 1b: embed chunk + vision pad overlay ──
                self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;
                phase_mark!("embed");
                (ple_prefetch, chunk_start, chunk_len, chunk_start)
            };

        let marconi_fast_cont_admitted = if seq_used_marconi_skip && self.config.is_qwen4_exp() {
            marconi_fast_cont_base
                && proc_count >= 2
                && proc_start.is_multiple_of(bs)
                && effective_seq_len_start == proc_start
                && crate::layers::qwen4_prefill_moe::exact_chunk_window_admitted(
                    proc_count, proc_start,
                )?
        } else {
            false
        };
        if seq_used_marconi_skip && self.config.is_qwen4_exp() {
            if marconi_fast_cont_admitted {
                // Re-invalidate right before the first computed chunk after the
                // Marconi skip (while holding `kv_cache`) so no interleaved request
                // between an early-returned prefix chunk and this suffix chunk can
                // leave a matching `s.end` that would bypass `refill_shadow`.
                if effective_seq_len_start == kv_write_start {
                    crate::layers::qwen4_qsa::prefill_fp8::invalidate_shadow_continuity(&kv_cache)?;
                }
                static MARCONI_FAST_CONT_LOGGED: std::sync::Once = std::sync::Once::new();
                MARCONI_FAST_CONT_LOGGED.call_once(|| {
                    tracing::info!(
                        "QWEN4_MARCONI_FAST_CONT_ENGAGED selector={} chunk_start={chunk_start} \
                         kv_write_start={kv_write_start} proc_start={proc_start} \
                         proc_count={proc_count} total={total}",
                        crate::layers::qwen4_prefill_moe::attn16::MARCONI_FAST_CONT_SELECTOR,
                    );
                });
            } else {
                lineage_ticket = None;
                cross_chunk_ple_prefetch_admitted = false;
                crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                    &mut seq.qwen4_prefill_lineage,
                    total,
                );
            }
        }
        let _prefix_hit_suppressed =
            (seq_used_marconi_skip && self.config.is_qwen4_exp() && !marconi_fast_cont_admitted)
                .then(crate::layers::qwen4_prefill_moe::attn16::suppress_family);

        // ── Phase 3: upload positions + MRoPE + slot metadata ──
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;
        phase_mark!("upload_meta");

        // ── Phase 3b: paged metadata (block_table + seq_len) ──
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        phase_mark!("upload_paged");

        // Force H2D metadata copy to complete before layer forward.
        // On DGX Spark SM121, the DMA engine may not properly serialize
        // pinned H2D copy with subsequent compute on the same stream,
        // causing CUDA 700 at >9K tokens. `prefill_b_upload_paged` already
        // synchronizes `stream` inside `copy_h2d_group_on_stream`, so only
        // synchronize here when `needs_paged` is false and on chunk 0.
        if !needs_paged && chunk_start == 0 && !Self::fast_host_h2d_enabled() {
            self.gpu.synchronize(stream)?;
        }
        phase_mark!("sync_meta");

        // ── Phase 4: forward through all layers ──
        self.prefill_b_forward_layers(
            tokens,
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            stream,
            ple_prefetch,
            cross_chunk_ple_prefetch_admitted,
        )?;
        self.prefill_b_qgate_dump(tokens, effective_seq_len_start, proc_count, stream)?;
        phase_mark!("layers");

        // ── Phase 4b: MTP last-K cross-chunk capture ──
        // D2H the tail rows of this chunk's `hidden_states` into the per-seq
        // host ring buffer. Must run BEFORE finalize_last (which reads logits
        // off hidden_states[proc_count-1] and may clobber adjacent buffers),
        // and BEFORE the next chunk overwrites hidden_states. No-op when MTP
        // last-K prefill is disabled or no proposer is wired.
        //
        // Capture only when this chunk contributes to the trailing `mtp_lastk_capacity`
        // tokens: on the final chunk, or on any preceding chunk ending past `L - capacity`.
        // Intermediate chunks ending at or before `L - capacity` strictly bypass D2H copy
        // and stream sync because their outputs are completely overwritten by subsequent chunks.
        let window_start = tokens.len().saturating_sub(self.mtp_lastk_capacity);
        if is_last_chunk || chunk_start + chunk_len > window_start {
            self.mtp_lastk_capture_chunk(
                seq,
                chunk_start,
                chunk_len,
                proc_count,
                tokens.len(),
                stream,
            )?;
        }

        // ── Phase 5: update sequence state incrementally ──
        // Always add chunk tokens exactly once: `EarlyReturn` records skipped
        // fully-cached non-last chunks itself in `prefill_b_proc_range`, while
        // chunks that reach Phase 5 record their tokens here.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;

        let result = if is_last_chunk {
            // ── Phase 6+7+8: final norm, lm_head, prefix-cache + snapshot save ──
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )?
        } else {
            // ── Phase 9: intermediate Marconi checkpoint ──
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            DevicePtr::NULL
        };
        if let Some(ticket) = lineage_ticket {
            crate::layers::qwen4_prefill_moe::attn16::commit_fast_lineage(
                &mut seq.qwen4_prefill_lineage,
                ticket,
                tokens,
                chunk_start,
                chunk_end,
            )?;
        }
        phase_mark!("finalize");
        if phase_on {
            let total = phase_t0.elapsed().as_secs_f64() * 1000.0;
            let mut joined: Vec<String> = Vec::with_capacity(phase_ms.len());
            let mut prev = 0.0f64;
            for (name, t) in &phase_ms {
                joined.push(format!("{name}={:.1}", t - prev));
                prev = *t;
            }
            joined.push(format!("total={total:.1}"));
            tracing::info!("PREFILL_PHASES tok={} | {}", chunk_len, joined.join(" "));
        }
        Ok(result)
    }

    /// Allocate `seq.chunked_prefill_meta` and upload the matched prefix blocks
    /// before a Marconi early-return chunk exits, so subsequent uncached suffix
    /// chunks see live GPU block-table metadata and stay on the fast prefill path.
    fn upload_marconi_prefix_blocks_for_early_return(
        &self,
        seq: &mut SequenceState,
        total: usize,
        bs: usize,
        stream: u64,
    ) -> Result<()> {
        let current_blocks = seq.block_table.len();
        let (upload_start, block_capacity, block_table_base) = {
            let meta = self.ensure_chunked_prefill_meta(seq, total, bs)?;
            (meta.uploaded_blocks, meta.block_capacity, meta.block_table)
        };
        anyhow::ensure!(
            upload_start <= current_blocks && current_blocks <= block_capacity,
            "Marconi fast-continuation block table out of bounds: \
             uploaded={upload_start}, current={current_blocks}, capacity={block_capacity}"
        );
        if upload_start < current_blocks {
            let new_blocks = &seq.block_table[upload_start..current_blocks];
            // SAFETY: `new_blocks` is a live slice of initialized `u32` block IDs;
            // `copy_h2d_group_fast` copies all bytes into pinned staging (or finishes
            // synchronously) before returning.
            let bt_bytes = unsafe {
                std::slice::from_raw_parts(
                    new_blocks.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(new_blocks),
                )
            };
            let dst = block_table_base.offset(upload_start * std::mem::size_of::<u32>());
            self.copy_h2d_group_fast(&[HostToDeviceCopy::new(bt_bytes, dst)], stream)?;
            if let Some(meta) = seq.chunked_prefill_meta.as_mut() {
                meta.uploaded_blocks = current_blocks;
            }
        }
        Ok(())
    }
}

/// Prefetch-map key of the admission read (`ATLAS_QWEN4_PLE_EARLY_ISSUE`).
/// Sequence slots are `< max_batch_size`, so it never names a real slot.
const QWEN4_PLE_ADMISSION_SLOT: usize = ple_prefetch_slots::ADMISSION_SLOT;

const PLE_EARLY_ISSUE_SELECTOR: &str = "ATLAS_QWEN4_PLE_EARLY_ISSUE";
/// `ATLAS_QWEN4_PLE_QUEUE_PREFETCH` (default off): see `begin_ple_queue_prefetch`.
const PLE_QUEUE_PREFETCH_SELECTOR: &str = "ATLAS_QWEN4_PLE_QUEUE_PREFETCH";

/// `ATLAS_QWEN4_PLE_QUEUE_PREFETCH=1` (read once; anything else is off) and,
/// when `ATLAS_PLE_READ_TRIGGER` is set, the same per-request trigger-file
/// gate as the admission read.
fn ple_queue_prefetch_active() -> bool {
    static SELECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let selected = *SELECTED.get_or_init(|| {
        parse_ple_queue_prefetch(std::env::var(PLE_QUEUE_PREFETCH_SELECTOR).ok().as_deref())
    });
    selected
        && match std::env::var_os(PLE_READ_TRIGGER).filter(|path| !path.is_empty()) {
            None => true,
            Some(path) => early_issue_trigger_allows(std::fs::read_to_string(path).ok().as_deref()),
        }
}

fn parse_ple_queue_prefetch(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

fn parse_ple_early_issue(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{PLE_EARLY_ISSUE_SELECTOR} must be absent, 0 or 1; got {other:?} (treated as off)"
        )),
    }
}

/// `ATLAS_QWEN4_PLE_EARLY_ISSUE=1` (read once; an invalid value is off and
/// logged).
fn ple_early_issue_selected() -> bool {
    static SELECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SELECTED.get_or_init(|| {
        match parse_ple_early_issue(std::env::var(PLE_EARLY_ISSUE_SELECTOR).ok().as_deref()) {
            Ok(selected) => selected,
            Err(error) => {
                tracing::error!("QWEN4_PLE_EARLY_ISSUE_REFUSED reason=invalid_value: {error}");
                false
            }
        }
    })
}

/// The reader's A/B/A switch (spark-storage `ple_offload`): when set, the K9
/// item W levers apply only while this file exists.
const PLE_READ_TRIGGER: &str = "ATLAS_PLE_READ_TRIGGER";

/// Trigger-file content that keeps the admission read on: present, and its
/// first word is not one of the reader-only modes (`legacy`, `plan`, `sector`)
/// used for attribution (the reader parses the first word the same way).
fn early_issue_trigger_allows(content: Option<&str>) -> bool {
    content.is_some_and(|content| {
        !matches!(
            content.split_whitespace().next().unwrap_or(""),
            "legacy" | "plan" | "sector"
        )
    })
}

/// Whether this admission starts a read: `ATLAS_QWEN4_PLE_EARLY_ISSUE=1` and,
/// when `ATLAS_PLE_READ_TRIGGER` is set, [`early_issue_trigger_allows`] on the
/// file's current content (checked per request, so one load serves an A/B/A).
fn ple_early_issue_active() -> bool {
    if !ple_early_issue_selected() {
        return false;
    }
    match std::env::var_os(PLE_READ_TRIGGER).filter(|path| !path.is_empty()) {
        None => true,
        Some(path) => early_issue_trigger_allows(std::fs::read_to_string(path).ok().as_deref()),
    }
}

fn should_begin_cross_chunk_ple_prefetch(
    stream_level: u8,
    fast_lineage_admitted: bool,
    chunk_start: usize,
    chunk_len: usize,
    total_tokens: usize,
) -> bool {
    stream_level >= 2
        && fast_lineage_admitted
        && chunk_len > 0
        && chunk_start
            .checked_add(chunk_len)
            .is_some_and(|next_start| next_start < total_tokens)
}

#[cfg(test)]
mod queue_prefetch_tests {
    use super::parse_ple_queue_prefetch;

    #[test]
    fn queue_prefetch_selector_is_exact_and_default_off() {
        assert!(parse_ple_queue_prefetch(Some("1")));
        for off in [None, Some(""), Some("0"), Some("true"), Some(" 1")] {
            assert!(!parse_ple_queue_prefetch(off), "{off:?}");
        }
    }
}

#[cfg(test)]
mod snapshot_restore_tests {
    use super::{
        attention_only_prefix_skip, cached_kv_rows_in_slice, restored_prefix_skip_tokens,
        should_begin_cross_chunk_ple_prefetch, should_restore_ssm_snapshot_with_opt_in,
    };

    #[test]
    fn cross_chunk_ple_read_ahead_requires_fast_lineage_and_a_full_next_chunk() {
        assert!(should_begin_cross_chunk_ple_prefetch(
            3, true, 0, 8192, 16384
        ));
        assert!(should_begin_cross_chunk_ple_prefetch(
            2, true, 0, 4096, 8192
        ));
        assert!(should_begin_cross_chunk_ple_prefetch(
            2, true, 0, 8192, 8232
        ));
        assert!(!should_begin_cross_chunk_ple_prefetch(
            3, false, 0, 8192, 8232
        ));
        assert!(!should_begin_cross_chunk_ple_prefetch(
            1, true, 0, 8192, 16384
        ));
        assert!(!should_begin_cross_chunk_ple_prefetch(
            3, true, 0, 8192, 8192
        ));
        assert!(!should_begin_cross_chunk_ple_prefetch(3, true, 0, 0, 8192));
        assert!(!should_begin_cross_chunk_ple_prefetch(
            3,
            true,
            usize::MAX,
            1,
            usize::MAX,
        ));
    }

    #[test]
    fn ple_early_issue_selector_is_strict_and_its_key_is_no_slot() {
        assert_eq!(super::parse_ple_early_issue(None), Ok(false));
        assert_eq!(super::parse_ple_early_issue(Some("")), Ok(false));
        assert_eq!(super::parse_ple_early_issue(Some("0")), Ok(false));
        assert_eq!(super::parse_ple_early_issue(Some("1")), Ok(true));
        for bad in ["2", "on", "true", " 1"] {
            assert!(super::parse_ple_early_issue(Some(bad)).is_err(), "{bad}");
        }
        // Trigger file: absent = off; reader-only modes keep the admission read off.
        assert!(!super::early_issue_trigger_allows(None));
        for on in ["", "\n", "on", "all\n", "single"] {
            assert!(super::early_issue_trigger_allows(Some(on)), "{on:?}");
        }
        for off in [
            "legacy",
            "plan\n",
            " sector ",
            "plan single",
            "sector single\n",
        ] {
            assert!(!super::early_issue_trigger_allows(Some(off)), "{off:?}");
        }
        // Sequence slots are bounded by max_batch_size; the admission key can never be one.
        let admission_key = super::QWEN4_PLE_ADMISSION_SLOT;
        assert!((0..4096usize).all(|slot| slot != admission_key));
    }

    #[test]
    fn exact_full_prompt_restore_requires_explicit_opt_in() {
        assert!(!should_restore_ssm_snapshot_with_opt_in(64, 64, 64, false));
        assert!(should_restore_ssm_snapshot_with_opt_in(64, 64, 64, true));
        assert!(should_restore_ssm_snapshot_with_opt_in(32, 64, 64, false));
        assert!(should_restore_ssm_snapshot_with_opt_in(64, 64, 80, false));
    }

    #[test]
    fn intermediate_ssm_checkpoint_limits_full_prompt_skip() {
        assert_eq!(restored_prefix_skip_tokens(true, 32, 64), 32);
        assert_eq!(restored_prefix_skip_tokens(false, 0, 64), 64);
    }

    #[test]
    fn recurrent_replay_does_not_overwrite_deeper_cached_kv() {
        assert_eq!(cached_kv_rows_in_slice(64, 32, 48), 32);
        assert_eq!(cached_kv_rows_in_slice(64, 0, 64), 64);
        assert_eq!(cached_kv_rows_in_slice(64, 64, 16), 0);
    }

    #[test]
    fn exact_snapshot_bypass_cannot_take_attention_only_skip() {
        assert!(!attention_only_prefix_skip(64, false, true));
        assert!(attention_only_prefix_skip(64, false, false));
        assert!(!attention_only_prefix_skip(64, true, false));
    }
}
