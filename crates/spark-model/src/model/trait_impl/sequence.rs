// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::gpu_lifetime::{before_reuse, require_completion};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn cache_sequence_dispatch(&self, seq: &SequenceState) {
        let bs = self.kv_cache.lock().block_size();
        // Only cache if the sequence has block-aligned content worth caching.
        // Sequences shorter than one block have no reusable KV blocks.
        if seq.tokens.len() >= bs && !seq.block_table.is_empty() {
            // Prompt tokens were already inserted + ref-bumped by prefill.
            // Only generated tokens past `prompt_len` are "newly seq-owned"
            // at this point — pass prompt_len as matched_tokens so insert
            // skips re-bumping the prompt portion.
            //
            // Phase 6.3 sliding-window: when HSS has slid older blocks out,
            // `block_table` no longer parallels `tokens` from index 0 — the
            // physical IDs at the front of block_table now hold WRITES for
            // recent positions, not the historical prompt. Skip cache_sequence
            // insert in that case to avoid populating the radix tree with
            // mis-correlated entries. (Disk-side ref counting via
            // `apply_evicted_blocks` keeps the disk_block_ids alive
            // independently when the prefix cache later evicts.)
            // Skip when the prefix cache is a no-op (`--enable-prefix-caching`
            // off): the manual inc_ref below would never get a paired dec_ref
            // from cache eviction, leaking the seq's blocks every request.
            // Also skip when HSS sliding has occurred (front of block_table no
            // longer parallels tokens) and on vision prompts.
            if self.prefix_cache.is_active()
                && !self.tokens_have_vision_pad(&seq.tokens)
                && seq.hss_window_start() == 0
            {
                let noncausal_spans = seq.rotary_positions.noncausal_spans();
                let acquired = self.prefix_cache.insert_with_noncausal_spans(
                    &seq.tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.prompt_len,
                    &noncausal_spans,
                );
                if self.config.is_qwen4_exp() && self.config.qwen4_qsa && seq.qwen4_qsa_required {
                    self.prefix_cache
                        .mark_blocks_qsa_populated(&seq.tokens, &seq.block_table, bs);
                }
                super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                // Bump KV block ref_counts so the prefix cache "owns" a reference.
                // This keeps blocks alive after free_sequence drops the sequence's ref.
                // Eviction (return_evicted_block) releases these refs when nodes are removed.
                let mut kv = self.kv_cache.lock();
                let num_cached_blocks = (seq.tokens.len() / bs).min(seq.block_table.len());
                for &block_idx in &seq.block_table[..num_cached_blocks] {
                    kv.inc_ref(block_idx);
                }
            }
        }
    }

    /// Release one proposer state's DFlash per-sequence device buffers, if it
    /// is a DFlash state at all. Extracted from `free_sequence_dispatch` so
    /// both proposer arms can be freed by the same code — see the call sites.
    ///
    /// Takes `&mut Option<Box<dyn ProposerState>>` — the FIELD's own type —
    /// rather than the `Option<&mut dyn ProposerState>` that reads more
    /// naturally. That is a borrow-checker requirement, not a style choice.
    /// `Box<dyn ProposerState>` carries the default object lifetime bound
    /// `+ 'static`, while an elided `&'a mut dyn ProposerState` parameter
    /// means `&'a mut (dyn ProposerState + 'a)`. `&mut` is INVARIANT in its
    /// pointee, so handing it a `&'a mut (dyn ProposerState + 'static)`
    /// derived from `seq: &'1 mut SequenceState` forces `'1: 'static` — the
    /// borrow is then inferred to live forever and every later use of `seq`
    /// in this method conflicts with it. Passing the field by reference keeps
    /// both sides at `+ 'static` and never forms the mismatched `&mut dyn`.
    pub(super) fn free_dflash_state_buffers(
        &self,
        state: &mut Option<Box<dyn crate::speculative::ProposerState>>,
    ) {
        let Some(ctx_allocation_bytes) = state
            .as_deref()
            .and_then(|ps| {
                ps.as_any()
                    .downcast_ref::<crate::layers::DflashProposerState>()
            })
            .map(|ds| ds.ctx_allocation_bytes)
        else {
            return;
        };
        tracing::trace!(
            ctx_allocation_bytes,
            max_slots = self.ssm_pool.max_slots,
            "reclaiming DFlash per-sequence state buffers"
        );
        // Return the retired state (ctx_hidden_acc, ctx_fc_cache, per-layer
        // ctx_k_cache/ctx_v_cache, and pinned host staging) to the owning
        // head's bounded free-list so subsequent requests reuse all buffers
        // with zero cuMemAlloc_v2 / cuMemFree_v2 stalls. When pooling is
        // disabled (`ATLAS_QWEN4_SEQ_BUF_POOL=0`), the pool is full, or host
        // MemAvailable drops below the UMA floor, every device buffer is freed
        // immediately instead.
        crate::layers::dflash_head::reclaim_dflash_state_buffers(
            self.gpu.as_ref(),
            state,
            self.ssm_pool.max_slots,
        );
    }

    pub(in crate::model) fn free_sequence_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        let was_identity_rotary = seq.rotary_positions.is_identity();
        super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
        self.slot_commit_table.lock().invalidate_slot(seq.slot_idx);
        self.reset_gdn_keep_late_slot(seq.slot_idx);
        // ATLAS_MTP_LASTK_DEFER: a sequence that ends on its first token never runs
        // its deferred MTP replay (its MTP cache allocation is mirrored, host only; the
        // proposer state is freed below).
        self.k8a_drop_deferred_lastk(seq);
        seq.rotary_positions = crate::traits::RotaryPositions::identity();
        // A verify-state commit may still be writing this slot on the
        // secondary stream. Chain that event onto the default stream before
        // zero_slot and its synchronize below; otherwise a newly allocated
        // sequence can reuse the slot while the retired request is still
        // overwriting its recurrent state.
        require_completion(
            self.sync_secondary_dispatch(),
            "free_sequence secondary stream",
        );

        // ATLAS_DFLASH_ASYNC: an in-flight second-stream propose reads this
        // sequence's ctx accumulator + fc/K/V caches — resolve (sync +
        // discard) BEFORE those buffers are freed below (use-after-free
        // guard). No-op when nothing is in flight.
        //
        // Both arms are resolved and freed. A two-arm build allocated a
        // proposer state per arm in `alloc_sequence`, and only ONE of them is
        // in `seq.proposer_state` at any moment — freeing just that one would
        // leak the parked arm's ctx accumulator + fc/K/V caches on every
        // request, which is the same unbounded-UMA failure the single-arm
        // free below was written to fix.
        if let Some(proposer) = self.active_proposer() {
            let res = match seq.proposer_state.as_deref_mut() {
                Some(ps) => proposer.resolve_async_inflight(self.gpu.as_ref(), Some(ps)),
                None => proposer.resolve_async_inflight(self.gpu.as_ref(), None),
            };
            require_completion(res, "free_sequence active proposer stream");
        }
        if let Some(proposer) = self.inactive_proposer() {
            let res = match seq.proposer_state_alt.as_deref_mut() {
                Some(ps) => proposer.resolve_async_inflight(self.gpu.as_ref(), Some(ps)),
                None => proposer.resolve_async_inflight(self.gpu.as_ref(), None),
            };
            require_completion(res, "free_sequence parked proposer stream");
        }

        require_completion(
            self.gpu.synchronize_device(),
            "free_sequence before device-buffer reclamation",
        );

        // Free the DFlash proposer state's per-sequence device buffers.
        // free_state can't (no GpuBackend in scope there); without this,
        // every request leaked the ctx accumulator + fc/K/V caches
        // (hundreds of MB per sequence — unbounded UMA growth while
        // serving).
        self.free_dflash_state_buffers(&mut seq.proposer_state);
        self.free_dflash_state_buffers(&mut seq.proposer_state_alt);

        // A slot is recyclable only after every previous reader/writer has
        // completed. Sticky CUDA errors require a fresh process, not reuse of
        // a CPU-side slot whose device lifetime is no longer known.
        let slot_to_release = if seq.slot_idx < self.ssm_pool.max_slots {
            Some(seq.slot_idx)
        } else {
            None
        };
        if let Some(slot) = slot_to_release {
            let stream = self.gpu.default_stream();
            require_completion(
                self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream),
                "free_sequence zero recurrent slot",
            );
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if let Some(ple) = &self.qwen4_ple {
                require_completion(
                    ple.zero_slot(slot, self.gpu.as_ref(), stream),
                    "free_sequence zero PLE slot",
                );
            }
            before_reuse(
                || self.gpu.synchronize(stream),
                || {
                    if let Some((_, frames)) = &self.qwen4_mtp_hidden_frames {
                        require_completion(
                            frames.lock().retire(slot).map_err(anyhow::Error::msg),
                            "free_sequence native MTP hidden retirement",
                        );
                    }
                    self.ssm_pool.release_slot(slot);
                },
                "free_sequence slot release barrier",
            );
            if self
                .comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
            {
                let mut pending = self.pending_freed_slots.lock();
                if !pending.contains(&slot) {
                    pending.push_back(slot);
                }
            }
        }

        // Release prefix cache refs before freeing blocks.
        // dec_ref only frees blocks whose ref_count hits 0
        // (i.e., blocks not shared with the prefix cache).
        self.prefix_cache
            .release(&seq.tokens, self.kv_cache.lock().block_size());
        if !seq.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&seq.block_table);
            seq.block_table.clear();
        }

        // --high-speed-swap: release disk-side refs for every block this
        // sequence ever held (Phase 6.1.c). disk_block_ids are layer-
        // agnostic (each ID indexes a slot in *every* layer's file), so
        // one dec_disk_ref per ID covers all layers' data simultaneously.
        // The orchestrator's free list only reclaims an ID when its
        // refcount hits 0, so sequences sharing a prefix correctly keep
        // each other's disk blocks alive via ref-counting.
        if !seq.disk_block_ids.is_empty() {
            // with_local returns Option<Result>: None when HSS isn't engaged
            // (no-op, fine), Some(Err) when the closure failed (advisory).
            if let Some(Err(e)) = spark_storage::with_local(|hss| {
                for &disk_id in &seq.disk_block_ids {
                    hss.dec_disk_ref(disk_id);
                }
                Ok(())
            }) {
                tracing::error!("free_sequence: spark_storage dec_disk_ref batch: {e:#}");
            }
            seq.disk_block_ids.clear();
            for v in seq.disk_last_offloaded_per_layer.iter_mut() {
                *v = 0;
            }
        }

        // All SSM buffers (h_state, conv_state, checkpoints, intermediates) belong
        // to the pool — do NOT gpu.free() them. Just clear the references.
        for state in &mut seq.layer_states {
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.h_state = DevicePtr(0);
                ssm.conv_state = DevicePtr(0);
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }

        // Invalidate cached CUDA graphs that reference this sequence's slot
        // — the graph was captured with this slot's KV/SSM pointers baked in,
        // and replaying after the slot is freed would read stale data.
        // decode_graph is keyed by slot, so drop only this slot's entry.
        // (parking_lot::Mutex::lock() never poisons, so the previous `if let
        // Ok(...) = .lock()` graceful-recovery branch is unreachable.)
        self.decode_graph_sparse_regime.lock().remove(&seq.slot_idx);
        if let Some(graph) = self.decode_graph.lock().remove(&seq.slot_idx)
            && let Err(e) = self.gpu.destroy_graph(graph)
        {
            tracing::error!(
                "free_sequence: destroy_graph(decode_graph[{}]): {e:#}",
                seq.slot_idx
            );
        }
        // batch_decode_graphs is keyed by (sorted_slot_ids, padded_n) when
        // ATLAS_SSM_MULTI_SEQ_GRAPH is on, else by ((empty), padded_n). In
        // graph mode, drop only entries that include this freed slot — other
        // slot-set graphs remain valid and avoid unnecessary recapture. In the
        // legacy off mode the cache is empty (never inserted), so the drain
        // is a no-op.
        {
            let mut cache = self.batch_decode_graphs.lock();
            let stale_keys: Vec<(Vec<usize>, usize)> = cache
                .keys()
                .filter(|(slots, _)| slots.is_empty() || slots.contains(&seq.slot_idx))
                .cloned()
                .collect();
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(batch_decode_graphs[{:?}]): {e:#}",
                        key
                    );
                }
            }
        }
        // Verify graphs are slot-keyed and bake only fixed model/slot device
        // addresses (`h_state`, `conv_state`, checkpoints/intermediates,
        // `gdn_lazy_retain_pool`, `meta_base`, `kv_cache`, `dflash_hidden_save`,
        // scratch buffers) — zero session-specific pointers under resident
        // non-HSS single-GPU KV across both Qwen3.8-27B and Qwen4 models.
        // Retain them across request retirement on the same slot to eliminate
        // ~20-40 ms re-capture per request; `pre_verify_copy_async_dispatch`
        // invalidates on any QSA or MRoPE regime change before replay.
        let retain_verify_graphs = can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            env_enabled: verify_graph_retain_across_reqs_enabled(),
            single_gpu: self.comm.is_none(),
            identity_rotary: was_identity_rotary,
            slots_in_pool: seq.slot_idx < self.ssm_pool.max_slots,
            resident_kv: self.kv_cache.lock().config().cache_blocks_per_seq.is_none(),
            headroom_ok: super::verify_d::verify_kgamma_retain_headroom_ok(),
        });
        if !retain_verify_graphs {
            for graph_mutex in [
                &self.verify2_graph,
                &self.verify3_graph,
                &self.verify4_graph,
            ] {
                if let Some(graph) = graph_mutex.lock().remove(&seq.slot_idx)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(verify[{}]): {e:#}",
                        seq.slot_idx
                    );
                }
            }
        }
        {
            let mut cache = self.verify_kgamma_graph.lock();
            let stale_keys = if retain_verify_graphs {
                self.note_verify_graph_retained_qsa(seq.slot_idx, seq.qwen4_qsa_required);
                Vec::new()
            } else {
                self.clear_verify_graph_qsa_regime(seq.slot_idx);
                super::sequence_graph_cleanup::verify_kgamma_keys_for_slots(&cache, &[seq.slot_idx])
            };
            let destroyed = stale_keys.len();
            if destroyed > 0 {
                tracing::info!(
                    "{}",
                    super::verify_graph_receipts::released_line(
                        "free_sequence",
                        seq.session_hash,
                        seq.slot_idx,
                        destroyed,
                    )
                );
            }
            super::verify_graph_receipts::reset_slot(seq.slot_idx);
            if !stale_keys.is_empty() {
                let mut lru = self.verify_kgamma_lru.lock();
                for key in &stale_keys {
                    lru.remove(key);
                }
            }
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(verify_kgamma_graph[{key:?}]): {e:#}"
                    );
                }
            }
        }

        // Free MTP proposer state (KV cache blocks) — for both arms, each
        // through the proposer that allocated it.
        if let Some(proposer) = self.active_proposer()
            && let Some(ref mut pstate) = seq.proposer_state
        {
            proposer.free_state(pstate.as_mut())?;
        }
        if let Some(proposer) = self.inactive_proposer()
            && let Some(ref mut pstate) = seq.proposer_state_alt
        {
            proposer.free_state(pstate.as_mut())?;
        }

        self.free_chunked_prefill_meta(seq)?;

        Ok(())
    }

    pub(super) fn compact_sequence_dispatch(
        &self,
        seq: &mut SequenceState,
        new_slot: usize,
    ) -> Result<bool> {
        let old_slot = seq.slot_idx;
        if old_slot == new_slot {
            return Ok(true);
        }

        // Claim the NEW slot EXCLUSIVELY, BEFORE any copy (upstream
        // 11e83f5b). A target not on the pool free list is OWNED by a live
        // sequence (e.g. one still PREFILLING, which the scheduler's active
        // list does not see): refuse without touching its hidden binding.
        // Retained-hidden preflight is fallible BEFORE device work. Roll
        // back only that reservation on error; source ownership is intact.
        let Some((mut hidden_frames, hidden_move)) =
            self.ssm_pool.claim_specific_with_preflight(new_slot, || {
                let mut hidden_frames = self
                    .qwen4_mtp_hidden_frames
                    .as_ref()
                    .map(|(_, frames)| frames.lock());
                let hidden_move = hidden_frames
                    .as_mut()
                    .map(|frames| frames.begin_move(old_slot, new_slot, seq.session_hash))
                    .transpose()
                    .map_err(anyhow::Error::msg)?;
                Ok((hidden_frames, hidden_move))
            })?
        else {
            return Ok(false);
        };

        // Once copying starts, every GPU failure is fail-stop; host pointers
        // and source-slot release still commit only after the copy barrier.
        let stream = self.gpu.default_stream();
        // Materialize any deferred KEEP_LATE DeltaNet rows on `old_slot` before
        // `ssm_pool.copy_slot` copies `h_state`/`conv_state` to `new_slot`.
        if self.has_gdn_keep_late_pending(old_slot) {
            require_completion(
                self.sync_secondary_dispatch(),
                "compact_sequence secondary stream before KEEP_LATE flush",
            );
            require_completion(
                self.flush_gdn_keep_late_slot(seq, true, stream),
                "compact_sequence KEEP_LATE flush",
            );
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = &self.qwen4_ple {
            require_completion(
                ple.copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream),
                "compact_sequence PLE copy",
            );
        }
        require_completion(
            self.ssm_pool
                .copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream),
            "compact_sequence recurrent copy",
        );
        if let Some((storage, _)) = &self.qwen4_mtp_hidden_frames
            && let Some((source, destination)) = hidden_move.and_then(|movement| movement.copy)
        {
            require_completion(
                self.gpu.copy_d2d_async(
                    storage.offset(source.offset),
                    storage.offset(destination.offset),
                    source.bytes,
                    stream,
                ),
                "compact_sequence native MTP hidden copy",
            );
        }

        // copy_slot mutates the destination asynchronously. Commit no host
        // pointers or ownership until that write has completed. A failed
        // barrier cannot safely return to the caller's destination cleanup.
        before_reuse(
            || self.gpu.synchronize(stream),
            || {
                if let Some(frames) = hidden_frames.as_mut()
                    && let Some(movement) = hidden_move
                {
                    require_completion(
                        frames.finish_move(movement).map_err(anyhow::Error::msg),
                        "compact_sequence native MTP hidden ownership",
                    );
                }
                // Update ALL SsmLayerState pool pointers to point at the new slot.
                // BUG FIX: previously only h_state and conv_state were repointed, leaving
                // the MTP checkpoint and intermediate pointers aimed at the OLD slot.
                // After release_slot, that old slot is reallocatable to a NEW sequence,
                // and any subsequent MTP save_hidden / start_checkpoint_async on this seq
                // would write into the new occupant's pool memory — cross-seq corruption.
                let has_mtp = self.ssm_pool.has_mtp;
                let num_intermediates = self.ssm_pool.num_intermediates;
                let mut ssm_layer_idx = 0usize;
                for (i, state) in seq.layer_states.iter_mut().enumerate() {
                    if self.config.layer_type(i) == LayerType::LinearAttention {
                        if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                            ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                            ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                            if has_mtp {
                                if ssm.h_state_checkpoint.is_some() {
                                    ssm.h_state_checkpoint =
                                        self.ssm_pool.h_checkpoint_opt(ssm_layer_idx, new_slot);
                                }
                                if ssm.conv_state_checkpoint.is_some() {
                                    ssm.conv_state_checkpoint = Some(
                                        self.ssm_pool.conv_checkpoint(ssm_layer_idx, new_slot),
                                    );
                                }
                                if !ssm.h_state_intermediates.is_empty() {
                                    ssm.h_state_intermediates.clear();
                                    for t in 0..num_intermediates {
                                        ssm.h_state_intermediates.push(
                                            self.ssm_pool.h_intermediate(
                                                ssm_layer_idx,
                                                new_slot,
                                                t,
                                            ),
                                        );
                                    }
                                }
                                if !ssm.conv_state_intermediates.is_empty() {
                                    ssm.conv_state_intermediates.clear();
                                    for t in 0..num_intermediates {
                                        ssm.conv_state_intermediates.push(
                                            self.ssm_pool.conv_intermediate(
                                                ssm_layer_idx,
                                                new_slot,
                                                t,
                                            ),
                                        );
                                    }
                                }
                                // WY17 LAZY-commit retention pointers follow the slot.
                                if ssm.wy17_kv_retain.is_some() {
                                    ssm.wy17_kv_retain =
                                        self.ssm_pool.wy17_kv_retain(ssm_layer_idx, new_slot);
                                }
                                if ssm.wy17_gate_retain.is_some() {
                                    ssm.wy17_gate_retain =
                                        self.ssm_pool.wy17_gate_retain(ssm_layer_idx, new_slot);
                                }
                            }
                        }
                        ssm_layer_idx += 1;
                    }
                }

                seq.slot_idx = new_slot;
                // GPU rejection shortlist rows are indexed by `slot_idx * 8`;
                // invalidate any pending or cached rejection-q slot metadata on
                // the moved DFlash proposer state(s) before `old_slot` returns
                // to the pool.
                clear_dflash_rejection_slot_metadata(&mut seq.proposer_state);
                clear_dflash_rejection_slot_metadata(&mut seq.proposer_state_alt);
                super::verify_qwen4_batched::reset_dedup_slot(old_slot);
                super::verify_qwen4_batched::reset_dedup_slot(new_slot);
                self.reset_gdn_keep_late_slot(old_slot);
                self.reset_gdn_keep_late_slot(new_slot);
                // The copy barrier above also makes old_slot safe to publish.
                self.ssm_pool.release_slot(old_slot);
            },
            "compact_sequence copy barrier",
        );

        // Invalidate decode graphs bound to either side of the slot swap.
        // Slot-keyed verify graphs bake only fixed pool/model addresses under
        // resident non-HSS single-GPU KV, so retain them across compaction when
        // pointer-invariant; otherwise destroy both sides cleanly.
        let retain_verify_graphs = can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            env_enabled: verify_graph_retain_across_reqs_enabled(),
            single_gpu: self.comm.is_none(),
            identity_rotary: seq.rotary_positions.is_identity(),
            slots_in_pool: old_slot < self.ssm_pool.max_slots && new_slot < self.ssm_pool.max_slots,
            resident_kv: self.kv_cache.lock().config().cache_blocks_per_seq.is_none(),
            headroom_ok: super::verify_d::verify_kgamma_retain_headroom_ok(),
        });
        for stale_slot in [old_slot, new_slot] {
            self.slot_commit_table.lock().invalidate_slot(stale_slot);
            self.decode_graph_sparse_regime.lock().remove(&stale_slot);
            if let Some(graph) = self.decode_graph.lock().remove(&stale_slot)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!(
                    "compact_sequence: destroy_graph(decode_graph[{stale_slot}]): {e:#}"
                );
            }
        }
        {
            let mut cache = self.batch_decode_graphs.lock();
            let stale_keys: Vec<(Vec<usize>, usize)> = cache
                .keys()
                .filter(|(slots, _)| {
                    slots.is_empty() || slots.contains(&old_slot) || slots.contains(&new_slot)
                })
                .cloned()
                .collect();
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "compact_sequence: destroy_graph(batch_decode_graphs[{:?}]): {e:#}",
                        key
                    );
                }
            }
        }
        if !retain_verify_graphs {
            for graph_mutex in [
                &self.verify2_graph,
                &self.verify3_graph,
                &self.verify4_graph,
            ] {
                for stale_slot in [old_slot, new_slot] {
                    if let Some(graph) = graph_mutex.lock().remove(&stale_slot)
                        && let Err(e) = self.gpu.destroy_graph(graph)
                    {
                        tracing::error!(
                            "compact_sequence: destroy_graph(verify[{stale_slot}]): {e:#}"
                        );
                    }
                }
            }
        }
        {
            let mut cache = self.verify_kgamma_graph.lock();
            let stale_keys = if retain_verify_graphs {
                // Record `old_slot`'s last active QSA regime; leave `new_slot`'s
                // already-recorded regime untouched so `pre_verify_copy_async_dispatch`
                // invalidates `new_slot` if the compacted sequence's QSA regime differs.
                self.note_verify_graph_retained_qsa(old_slot, seq.qwen4_qsa_required);
                Vec::new()
            } else {
                for stale_slot in [old_slot, new_slot] {
                    self.clear_verify_graph_qsa_regime(stale_slot);
                }
                super::sequence_graph_cleanup::verify_kgamma_keys_for_slots(
                    &cache,
                    &[old_slot, new_slot],
                )
            };
            let destroyed = stale_keys.len();
            if destroyed > 0 {
                tracing::info!(
                    "{}",
                    super::verify_graph_receipts::released_line(
                        "compact_sequence",
                        seq.session_hash,
                        old_slot,
                        destroyed,
                    )
                );
            }
            for stale_slot in [old_slot, new_slot] {
                super::verify_graph_receipts::reset_slot(stale_slot);
            }
            if !stale_keys.is_empty() {
                let mut lru = self.verify_kgamma_lru.lock();
                for key in &stale_keys {
                    lru.remove(key);
                }
            }
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "compact_sequence: destroy_graph(verify_kgamma_graph[{key:?}]): {e:#}"
                    );
                }
            }
        }
        // NOTE (ATLAS_MULTISEQ_GRAPHS): `piecewise_decode_graphs` is
        // intentionally NOT invalidated on compaction. Those segment graphs
        // are keyed by `(padded_n, seg_id)` and bake NO per-slot address —
        // the SSM h_state/conv_state pointers are read from the layer-stable
        // ptr scratch, which the piecewise dispatcher refreshes from each
        // sequence's (post-compaction) `layer_states` before every replay.
        // So a compaction that changes a seq's slot is transparent: the next
        // pre-replay `multiseq_refresh_ptr_table` uploads the new slot's
        // pointers and the same cached graph replays correctly.
        Ok(true)
    }

    pub(super) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        // Materialize any deferred KEEP_LATE DeltaNet rows and drain before
        // host-side D2H serialization reads `h_state`/`conv_state`.
        if self.has_gdn_keep_late_pending(seq.slot_idx) {
            let stream = self.gpu.default_stream();
            self.sync_secondary_dispatch()?;
            self.flush_gdn_keep_late_slot(seq, true, stream)?;
            self.gpu.synchronize(stream)?;
        }
        // ATLAS_QWEN4_SEQSTATE=1: the exact Qwen4 state blob (KV, QSA side
        // cache, GDN, PLE history, native-MTP state) replaces this format.
        if self.config.is_qwen4_exp() && crate::model::seqstate::enabled() {
            return self.save_qwen4_seqstate(seq, writer);
        }
        anyhow::ensure!(
            seq.rotary_positions.is_identity(),
            "image rotary state is not supported by the legacy disk-swap format"
        );
        let gpu = self.gpu.as_ref();

        // Phase 1: Copy all KV block data from GPU to host buffers under the lock.
        let kv_buffers = {
            let kv = self.kv_cache.lock();
            let mut bufs = Vec::with_capacity(seq.block_table.len() * kv.num_layers());
            for &block_idx in &seq.block_table {
                for layer_idx in 0..kv.num_layers() {
                    bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
                }
            }
            bufs
        }; // Lock released here.

        // Phase 2: Write KV data to disk (no lock held).
        for (k_data, v_data) in &kv_buffers {
            writer.write_all(k_data)?;
            writer.write_all(v_data)?;
        }

        // Phase 3: Copy SSM states from GPU to host, then write to disk.
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                // The swap FILE is always FP32, so a resumed sequence can be
                // read back without carrying a dtype tag. Under
                // ATLAS_SSM_H_FP16 a decoding slot holds FP16 packed into the
                // first half of h_bytes; serializing that raw would resume into
                // a freshly-allocated SequenceState whose h_is_f16 is false,
                // and the next decode step would narrow already-narrow bytes.
                // Widen here instead — the mirror of ssm_snapshot::save().
                if ssm.h_is_f16 {
                    let src = self.widen_h_to_f32_scratch(gpu, ssm.h_state)?;
                    gpu.copy_d2h(src, &mut h_buf)?;
                } else {
                    gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                }
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                writer.write_all(&h_buf)?;
                writer.write_all(&c_buf)?;
            }
        }

        // Phase 4: Serialize Qwen4 PLE convolution history when present.
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = &self.qwen4_ple {
            let ple_buf = ple.save_slot_host(seq.slot_idx, gpu)?;
            writer.write_all(&ple_buf)?;
        }

        // Phase 5: Serialize QSA physical-block side caches (`raw_ring` + `compressed_keys`)
        // so swapping back into freshly allocated physical KV blocks preserves sparse selector state.
        let qsa_layer_indices: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(idx, layer)| layer.has_qsa_side_cache().then_some(idx))
            .collect();
        if !qsa_layer_indices.is_empty() && !seq.block_table.is_empty() {
            let qsa_buffers = {
                let kv = self.kv_cache.lock();
                let mut bufs = Vec::with_capacity(seq.block_table.len() * qsa_layer_indices.len());
                for &block_idx in &seq.block_table {
                    for &layer_idx in &qsa_layer_indices {
                        bufs.push(self.layers[layer_idx].read_qsa_swap_block(block_idx, &kv, gpu)?);
                    }
                }
                bufs
            };
            for (raw_data, comp_data) in &qsa_buffers {
                writer.write_all(raw_data)?;
                writer.write_all(comp_data)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub(super) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.reset_gdn_keep_late_slot(seq.slot_idx);
        if self.config.is_qwen4_exp() && crate::model::seqstate::enabled() {
            return self.restore_qwen4_seqstate(seq, num_blocks, reader);
        }
        let gpu = self.gpu.as_ref();

        // Phase 1: Read all KV block data from disk into host buffers.
        let (num_layers, block_size, layer_strides) = {
            let kv = self.kv_cache.lock();
            let n = kv.num_layers();
            let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
            (n, kv.block_size(), strides)
        };

        let mut kv_buffers = Vec::with_capacity(num_blocks * num_layers);
        for _ in 0..num_blocks {
            for layer_idx in 0..num_layers {
                let stride = layer_strides[layer_idx];
                let mut k_data = vec![0u8; stride];
                let mut v_data = vec![0u8; stride];
                reader.read_exact(&mut k_data)?;
                reader.read_exact(&mut v_data)?;
                kv_buffers.push((k_data, v_data));
            }
        }

        // Phase 2: Allocate blocks and write data under the lock.
        {
            let mut kv = self.kv_cache.lock();
            let mut new_block_table = Vec::with_capacity(num_blocks);
            let mut buf_idx = 0;
            let write_res: Result<()> = (|| {
                for _ in 0..num_blocks {
                    let block_idx = kv.alloc_block()?;
                    new_block_table.push(block_idx);
                    for layer_idx in 0..num_layers {
                        let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                        kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                        buf_idx += 1;
                    }
                }
                Ok(())
            })();
            if let Err(err) = write_res {
                if !new_block_table.is_empty() {
                    kv.free_blocks(&new_block_table);
                }
                return Err(err);
            }
            seq.block_table = new_block_table;
        } // Lock released here.

        // Phase 3: Read SSM state data from disk and upload to GPU.
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                reader.read_exact(&mut h_buf)?;
                reader.read_exact(&mut c_buf)?;
                gpu.copy_h2d(&h_buf, ssm.h_state)?;
                gpu.copy_h2d(&c_buf, ssm.conv_state)?;
            }
        }

        // Phase 4: Restore Qwen4 PLE convolution history when present.
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = &self.qwen4_ple {
            let mut ple_buf = vec![0u8; ple.state_stride()];
            reader.read_exact(&mut ple_buf)?;
            ple.restore_slot_host(seq.slot_idx, &ple_buf, gpu)?;
        }

        // Phase 5: Restore QSA physical-block side caches into the newly allocated `seq.block_table`.
        let qsa_layer_indices: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(idx, layer)| layer.has_qsa_side_cache().then_some(idx))
            .collect();
        if !qsa_layer_indices.is_empty() && num_blocks > 0 {
            let (raw_bytes, comp_bytes) =
                crate::layers::Qwen4QsaIndexer::swap_block_byte_sizes(block_size)?;
            let mut qsa_buffers = Vec::with_capacity(num_blocks * qsa_layer_indices.len());
            for _ in 0..num_blocks {
                for _ in &qsa_layer_indices {
                    let mut raw_buf = vec![0u8; raw_bytes];
                    let mut comp_buf = vec![0u8; comp_bytes];
                    reader.read_exact(&mut raw_buf)?;
                    reader.read_exact(&mut comp_buf)?;
                    qsa_buffers.push((raw_buf, comp_buf));
                }
            }
            let kv = self.kv_cache.lock();
            let mut idx = 0;
            for &new_block_idx in &seq.block_table {
                for &layer_idx in &qsa_layer_indices {
                    let (ref raw_buf, ref comp_buf) = qsa_buffers[idx];
                    self.layers[layer_idx].write_qsa_swap_block(
                        new_block_idx,
                        raw_buf,
                        comp_buf,
                        &kv,
                        gpu,
                    )?;
                    idx += 1;
                }
            }
        }

        Ok(())
    }

    pub(super) fn num_free_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }

    pub(super) fn num_total_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_blocks()
    }
}

/// Clear staged and cached GPU rejection-sampling slot metadata on a DFlash
/// proposer state when its owning sequence compacts to a new physical slot.
fn clear_dflash_rejection_slot_metadata(
    state: &mut Option<Box<dyn crate::speculative::ProposerState>>,
) {
    if let Some(ds) = state.as_deref_mut().and_then(|ps| {
        ps.as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
    }) {
        ds.invalidate_rejection_slot_state();
    }
}

/// Keep slot-keyed verify CUDA graphs alive across request retirement and
/// slot compaction on the same physical slot under resident non-HSS single-GPU
/// KV (`ATLAS_VERIFY_GRAPH_RETAIN_ACROSS_REQS`, default ON unless `"0"`).
fn verify_graph_retain_across_reqs_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_VERIFY_GRAPH_RETAIN_ACROSS_REQS")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Pointer-invariance preconditions for retaining slot-keyed verify CUDA graphs
/// across request retirement or compaction.
#[derive(Debug, Clone, Copy)]
struct VerifyGraphRetainCheck {
    env_enabled: bool,
    single_gpu: bool,
    identity_rotary: bool,
    slots_in_pool: bool,
    resident_kv: bool,
    headroom_ok: bool,
}

fn can_retain_slot_verify_graphs(check: VerifyGraphRetainCheck) -> bool {
    check.env_enabled
        && check.single_gpu
        && check.identity_rotary
        && check.slots_in_pool
        && check.resident_kv
        && check.headroom_ok
}

#[cfg(test)]
mod compaction_rejection_tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::clear_dflash_rejection_slot_metadata;
    use crate::layers::DflashProposerState;
    use crate::speculative::ProposerState;

    #[test]
    fn compaction_clears_pending_and_cached_rejection_slot_metadata() {
        let gpu = MockGpuBackend::new();
        let ds = DflashProposerState::new_for_compaction_test(&gpu);
        ds.set_pending_rejection_slot(Some(3));
        ds.set_last_rejection_q_for_test(Some((3, 512, 4)));
        assert_eq!(ds.pending_rejection_slot_for_test(), Some(3));
        assert_eq!(ds.last_rejection_q(), Some((3, 512, 4)));

        let mut boxed: Option<Box<dyn ProposerState>> = Some(Box::new(ds));
        clear_dflash_rejection_slot_metadata(&mut boxed);

        let cleared = boxed
            .as_deref()
            .expect("state still present")
            .as_any()
            .downcast_ref::<DflashProposerState>()
            .expect("DflashProposerState");
        assert_eq!(cleared.pending_rejection_slot_for_test(), None);
        assert_eq!(cleared.last_rejection_q(), None);

        // None is a clean no-op.
        let mut empty: Option<Box<dyn ProposerState>> = None;
        clear_dflash_rejection_slot_metadata(&mut empty);
        assert!(empty.is_none());
    }

    #[test]
    fn compact_sequence_clears_both_proposer_arms_before_releasing_old_slot() {
        let source = include_str!("sequence.rs");
        let compact = source
            .split("pub(super) fn compact_sequence_dispatch(")
            .nth(1)
            .expect("compact_sequence_dispatch")
            .split("pub(super) fn save_sequence_state_dispatch(")
            .next()
            .expect("end of compact_sequence_dispatch");
        let clear_primary = compact
            .find("clear_dflash_rejection_slot_metadata(&mut seq.proposer_state);")
            .expect("clear primary proposer state");
        let clear_alt = compact
            .find("clear_dflash_rejection_slot_metadata(&mut seq.proposer_state_alt);")
            .expect("clear alt proposer state");
        let release = compact
            .find("self.ssm_pool.release_slot(old_slot);")
            .expect("release old_slot");
        assert!(clear_primary < release);
        assert!(clear_alt < release);
    }

    #[test]
    fn verify_graph_retention_requires_all_pointer_invariance_guards() {
        use super::{VerifyGraphRetainCheck, can_retain_slot_verify_graphs};

        let ok = VerifyGraphRetainCheck {
            env_enabled: true,
            single_gpu: true,
            identity_rotary: true,
            slots_in_pool: true,
            resident_kv: true,
            headroom_ok: true,
        };
        assert!(can_retain_slot_verify_graphs(ok));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            env_enabled: false,
            ..ok
        }));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            single_gpu: false,
            ..ok
        }));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            identity_rotary: false,
            ..ok
        }));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            slots_in_pool: false,
            ..ok
        }));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            resident_kv: false,
            ..ok
        }));
        assert!(!can_retain_slot_verify_graphs(VerifyGraphRetainCheck {
            headroom_ok: false,
            ..ok
        }));
    }
}
