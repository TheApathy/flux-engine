// SPDX-License-Identifier: AGPL-3.0-only

//! Model bridge for one generation across every native K-row cohort subframe
//! (K5 by default, or uniform/ragged `K_s in 1..=8` under
//! `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`).

use super::block_mgmt::ensure_blocks_through_decode;
use super::types::TransformerModel;
use crate::traits::verify_owner::batch::{BoundRange, FrameRow, LaneState, LaneTicket, Region};
use crate::traits::verify_owner::{Frame, Owner};
use crate::traits::{Model, SequenceState};
use anyhow::{Context, Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

fn range(base: DevicePtr, allocation_bytes: usize, span: DevicePtr, bytes: usize) -> BoundRange {
    BoundRange {
        allocation: Region {
            address: base.0,
            bytes: allocation_bytes,
        },
        span: Region {
            address: span.0,
            bytes,
        },
    }
}

fn lane_range(
    base: DevicePtr,
    allocation: usize,
    slot: usize,
    stride: usize,
) -> Result<BoundRange> {
    let offset = slot
        .checked_mul(stride)
        .context("batch persistent lane offset overflow")?;
    ensure!(
        base.0 != 0
            && stride > 0
            && offset
                .checked_add(stride)
                .is_some_and(|end| end <= allocation),
        "batch persistent lane exceeds its recorded allocation"
    );
    base.0
        .checked_add(u64::try_from(allocation)?)
        .context("batch allocation address overflow")?;
    Ok(range(base, allocation, base.offset(offset), stride))
}

impl TransformerModel {
    pub(super) fn begin_owned_verify_batch(
        &self,
        seqs: &mut [&mut SequenceState],
        k: usize,
    ) -> Result<()> {
        if let Some(ks) = crate::layers::current_ragged_cohort_widths() {
            return self.begin_owned_verify_batch_ragged(seqs, &ks);
        }
        let ks = vec![k; seqs.len()];
        self.begin_owned_verify_batch_ragged(seqs, &ks)
    }

    pub(super) fn begin_owned_verify_batch_ragged(
        &self,
        seqs: &mut [&mut SequenceState],
        ks: &[usize],
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        ensure!(
            self.qwen4_cross_seq_staging.is_some() && self.config.is_qwen4_exp(),
            "native K5 batch ownership requires provisioned model lanes"
        );
        ensure!(
            !ks.is_empty() && ks.len() == seqs.len(),
            "native batch width vector length {} differs from cohort size {}",
            ks.len(),
            seqs.len()
        );
        let adaptive = crate::layers::qwen4_verify_lanes::adaptive_k_selected()?;
        let mut min_k = usize::MAX;
        let mut max_k = 0usize;
        let mut total_rows = 0usize;
        for &k_s in ks {
            ensure!(
                crate::traits::verify_owner::batch::width_admitted(k_s, adaptive),
                "native batch cohort width K{k_s} is not admitted (adaptive={adaptive})"
            );
            min_k = min_k.min(k_s);
            max_k = max_k.max(k_s);
            total_rows = total_rows
                .checked_add(k_s)
                .context("native batch cohort row count overflow")?;
        }
        let is_ragged = min_k != max_k;
        if is_ragged || min_k == 1 {
            ensure!(
                adaptive && self.comm.is_none(),
                "ragged or K1 native batch cohort requires TP=1 with ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1"
            );
        }
        let is_dflash = self.active_proposer().is_some_and(|proposer| {
            proposer.kind() == crate::speculative::DraftProposerKind::DFlash
        });
        if is_dflash {
            ensure!(
                self.dflash_hidden_save.is_some(),
                "batched DFlash verify lacks allocated dflash_hidden_save capture buffer"
            );
        }
        for seq in seqs.iter() {
            self.ensure_verify_sequence_idle(seq)?;
            ensure!(
                seq.tokens.len() == seq.seq_len,
                "batch begin token extent differs from its cursor"
            );
            ensure!(
                seq.layer_states.len() == self.layers.len(),
                "batch begin lacks exact layer states"
            );
            for state in [&seq.proposer_state, &seq.proposer_state_alt] {
                if let Some(native) = state.as_ref().and_then(|state| {
                    state
                        .as_any()
                        .downcast_ref::<crate::layers::qwen4_mtp::Qwen4MtpState>()
                }) {
                    ensure!(
                        !native.has_unconsumed_verify_frame(),
                        "batch begin has an unconsumed native handoff"
                    );
                }
            }
        }
        let owners: Vec<_> = seqs
            .iter()
            .zip(ks.iter().copied())
            .map(|(seq, k_s)| Owner {
                slot: seq.slot_idx,
                session: seq.session_hash,
                cursor: seq.seq_len,
                rows: k_s,
            })
            .collect();
        let (reservation, tickets) = {
            let mut shared = self.verify_transactions.lock();
            self.verify_batch_transactions
                .lock()
                .reserve(
                    &mut shared,
                    &owners,
                    crate::layers::qwen4_verify_lanes::lanes(),
                )
                .map_err(|error| {
                    super::verify_frame::admission_error("batch reserve failed", error)
                })?
        };
        // Install every lane before any preparation can fail. Cleanup can then
        // reach the entire cohort, including lanes not yet forwarded.
        for (seq, &ticket) in seqs.iter_mut().zip(&tickets) {
            seq.verify_frame = Some(Frame::new_batch(ticket));
        }
        let mut bindings = Vec::with_capacity(seqs.len());
        let mut kv = self.kv_cache.lock();
        ensure!(
            kv.config().cache_blocks_per_seq.is_none(),
            "native K5 batch ownership requires resident non-HSS KV"
        );
        let block_size = kv.block_size();
        ensure!(block_size > 0, "native K5 batch has zero KV block size");
        for (seq, &k_s) in seqs.iter_mut().zip(ks.iter()) {
            let last = seq
                .seq_len
                .checked_add(k_s - 1)
                .context("batch KV lookahead overflow")?;
            for block in seq.seq_len / block_size..=last / block_size {
                ensure_blocks_through_decode(
                    seq,
                    block,
                    &mut kv,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    self.gpu.default_stream(),
                )?;
            }
        }
        for ((seq, &ticket), &k_s) in seqs.iter().zip(&tickets).zip(ks.iter()) {
            let last = seq
                .seq_len
                .checked_add(k_s - 1)
                .context("batch KV range overflow")?;
            let mut blocks = Vec::new();
            for logical in seq.seq_len / block_size..=last / block_size {
                let &physical = seq
                    .block_table
                    .get(logical)
                    .context("batch writable KV block is absent")?;
                ensure!(
                    (physical as usize) < kv.num_blocks() && kv.ref_count(physical) == 1,
                    "batch writable KV block is outside its pool or shared"
                );
                ensure!(!blocks.contains(&physical), "batch logical KV blocks alias");
                blocks.push(physical);
            }
            let mut ranges = self.batch_recurrent_ranges(seq, k_s)?;
            for layer in 0..kv.num_layers() {
                ensure!(
                    matches!(
                        kv.dtype_for_layer(layer),
                        KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4
                    ),
                    "native K5 batch requires BF16 or NVFP4 KV allocations"
                );
                for (base, stride, key) in [
                    (
                        kv.k_pool_ptr(layer),
                        kv.k_block_stride_bytes_for_layer(layer),
                        true,
                    ),
                    (
                        kv.v_pool_ptr(layer),
                        kv.v_block_stride_bytes_for_layer(layer),
                        false,
                    ),
                ] {
                    let bytes = kv
                        .num_blocks()
                        .checked_mul(stride)
                        .context("batch KV allocation overflow")?;
                    for &block in &blocks {
                        let actual = if key {
                            kv.k_cache_ptr(layer, block)
                        } else {
                            kv.v_cache_ptr(layer, block)
                        };
                        let checked = lane_range(base, bytes, block as usize, stride)?;
                        ensure!(
                            checked.span.address == actual.0,
                            "batch KV accessor allocation mismatch"
                        );
                        ranges.push(checked);
                    }
                }
            }
            let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
            for layer in &self.layers {
                for (base, bytes, span, span_bytes) in layer.qwen4_batch_verify_lane_ranges(
                    seq.slot_idx,
                    self.gpu.as_ref(),
                    &kv,
                    &self.config,
                    &blocks,
                )? {
                    ranges.push(range(base, bytes, span, span_bytes));
                }
            }
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if let Some(ple) = &self.qwen4_ple {
                for (base, bytes, span, span_bytes) in ple.batch_verify_lane_ranges(seq.slot_idx)? {
                    ranges.push(range(base, bytes, span, span_bytes));
                }
            }
            if !is_dflash {
                let (storage, hidden) = self
                    .qwen4_mtp_hidden_frames
                    .as_ref()
                    .context("native K5 batch lacks retained target hidden storage")?;
                let hidden = hidden.lock();
                let span = hidden
                    .slot_range(seq.slot_idx)
                    .map_err(anyhow::Error::msg)?;
                ranges.push(range(
                    *storage,
                    hidden.allocation_bytes(),
                    storage.offset(span.offset),
                    span.bytes,
                ));
            }
            bindings.push((ticket, ranges));
        }
        drop(kv);
        let mut shared = self.verify_transactions.lock();
        self.verify_batch_transactions
            .lock()
            .bind(&mut shared, reservation, &bindings)
            .map_err(|error| {
                super::verify_frame::protocol_error("batch address binding failed", error)
            })?;
        if std::env::var("ATLAS_QWEN4_BATCHED_K5_LOG").ok().as_deref() == Some("1") {
            tracing::info!(
                cohort_owner = ?reservation,
                cohort_sequences = seqs.len(),
                min_k,
                max_k,
                total_rows,
                ragged = is_ragged,
                "QWEN4_OWNED_BATCH_BOUND"
            );
        }
        Ok(())
    }

    fn batch_recurrent_ranges(&self, seq: &SequenceState, k: usize) -> Result<Vec<BoundRange>> {
        let pool = &self.ssm_pool;
        let geometry = pool.allocation_geometry;
        ensure!(
            pool.has_mtp
                && seq.slot_idx < pool.max_slots
                && seq.slot_idx < pool.spec_slots
                && pool.num_intermediates >= k
                && geometry.h_bytes == pool.h_bytes
                && geometry.conv_bytes == pool.conv_bytes
                && geometry.spec_slots == pool.spec_slots
                && geometry.compact_lazy_h == pool.compact_lazy_h,
            "native K5 batch has no recorded recurrent lane geometry"
        );
        for count in [
            pool.h_state_pools.len(),
            pool.conv_state_pools.len(),
            pool.conv_checkpoint_pools.len(),
            pool.conv_intermediate_pools.len(),
        ] {
            ensure!(
                count == pool.num_ssm_layers,
                "batch recurrent pool allocation count drifted"
            );
        }
        if pool.compact_lazy_h {
            ensure!(
                pool.h_checkpoint_pools.is_empty() && pool.h_intermediate_pools.is_empty(),
                "compact recurrent pool still holds per-layer H speculative pools"
            );
        } else {
            for count in [
                pool.h_checkpoint_pools.len(),
                pool.h_intermediate_pools.len(),
            ] {
                ensure!(
                    count == pool.num_ssm_layers,
                    "batch recurrent H pool allocation count drifted"
                );
            }
        }
        ensure!(
            pool.wy17_kv_retain_pools.len() == pool.wy17_gate_retain_pools.len()
                && (pool.wy17_kv_retain_pools.is_empty()
                    || pool.wy17_kv_retain_pools.len() == pool.num_ssm_layers),
            "batch recurrent retain allocation count drifted"
        );
        let slot = seq.slot_idx;
        let h_inter = pool
            .num_intermediates
            .checked_mul(pool.h_bytes)
            .context("batch H intermediate overflow")?;
        let conv_inter = pool
            .num_intermediates
            .checked_mul(pool.conv_bytes)
            .context("batch conv intermediate overflow")?;
        let mut out = Vec::new();
        if pool.compact_lazy_h && pool.num_ssm_layers > 0 {
            out.push(lane_range(
                pool.h_verify_scratch,
                geometry.h_verify_scratch_bytes,
                slot,
                pool.h_bytes,
            )?);
        }
        let mut ssm = 0;
        for (layer, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(layer) != LayerType::LinearAttention {
                continue;
            }
            ensure!(
                ssm < pool.num_ssm_layers,
                "batch recurrent schedule exceeds actual pool layers"
            );
            let state = state
                .as_any()
                .downcast_ref::<crate::layer::SsmLayerState>()
                .context("native K5 batch recurrent layer has foreign state")?;
            ensure!(
                state.h_state == pool.h_state(ssm, slot)
                    && state.conv_state == pool.conv_state(ssm, slot)
                    && state.h_state_checkpoint == pool.h_checkpoint_opt(ssm, slot)
                    && state.conv_state_checkpoint == Some(pool.conv_checkpoint(ssm, slot))
                    && state.h_state_intermediates.len() == pool.num_intermediates
                    && state.conv_state_intermediates.len() == pool.num_intermediates,
                "native K5 batch recurrent state is not owned by its physical slot"
            );
            ensure!(
                state.wy17_kv_retain == pool.wy17_kv_retain(ssm, slot)
                    && state.wy17_gate_retain == pool.wy17_gate_retain(ssm, slot),
                "native K5 batch recurrent retained inputs are not owned by their physical slot"
            );
            for row in 0..pool.num_intermediates {
                ensure!(
                    state.h_state_intermediates[row] == pool.h_intermediate(ssm, slot, row)
                        && state.conv_state_intermediates[row]
                            == pool.conv_intermediate(ssm, slot, row),
                    "native K5 batch recurrent intermediate points outside its lane"
                );
            }
            for (base, bytes, stride) in [
                (
                    pool.h_state_pools[ssm],
                    geometry.h_state_allocation_bytes,
                    pool.h_bytes,
                ),
                (
                    pool.conv_state_pools[ssm],
                    geometry.conv_state_allocation_bytes,
                    pool.conv_bytes,
                ),
                (
                    pool.conv_checkpoint_pools[ssm],
                    geometry.conv_checkpoint_allocation_bytes,
                    pool.conv_bytes,
                ),
                (
                    pool.conv_intermediate_pools[ssm],
                    geometry.conv_intermediate_allocation_bytes,
                    conv_inter,
                ),
            ] {
                out.push(lane_range(base, bytes, slot, stride)?);
            }
            if !pool.compact_lazy_h {
                for (base, bytes, stride) in [
                    (
                        pool.h_checkpoint_pools[ssm],
                        geometry.h_checkpoint_allocation_bytes,
                        pool.h_bytes,
                    ),
                    (
                        pool.h_intermediate_pools[ssm],
                        geometry.h_intermediate_allocation_bytes,
                        h_inter,
                    ),
                ] {
                    out.push(lane_range(base, bytes, slot, stride)?);
                }
            }
            if !pool.wy17_kv_retain_pools.is_empty() {
                out.push(lane_range(
                    pool.wy17_kv_retain_pools[ssm],
                    geometry.kv_retain_allocation_bytes,
                    slot,
                    pool.kv_retain_bytes,
                )?);
                out.push(lane_range(
                    pool.wy17_gate_retain_pools[ssm],
                    geometry.gate_retain_allocation_bytes,
                    slot,
                    pool.gate_retain_bytes,
                )?);
            }
            ssm += 1;
        }
        ensure!(
            ssm == pool.num_ssm_layers,
            "native K5 batch recurrent layer count drifted"
        );
        Ok(out)
    }

    pub(super) fn batch_lane(
        &self,
        seq: &SequenceState,
        rows: usize,
        cleanup: bool,
    ) -> Result<Option<LaneTicket>> {
        let Some(frame) = seq.verify_frame else {
            return Ok(None);
        };
        let Some(ticket) = frame.batch() else {
            return Ok(None);
        };
        ensure!(
            rows == ticket.owner().rows
                && frame.matches_sequence(seq.slot_idx, seq.session_hash, seq.seq_len, rows),
            "native batch ticket slot/session/cursor/width mismatch"
        );
        let shared = self.verify_transactions.lock();
        let batch = self.verify_batch_transactions.lock();
        ensure!(
            batch.owns_for_cleanup(&shared, ticket),
            "native K5 batch ticket lost its cohort reservation"
        );
        if !cleanup {
            crate::model::verify_failure::check()?;
            shared.check().map_err(|error| {
                super::verify_frame::protocol_error("batch lane owner failed", error)
            })?;
        }
        Ok(Some(ticket))
    }

    /// Width of the cohort that owns `seq`'s batch lane, if any.
    pub(super) fn batch_lane_rows(seq: &SequenceState) -> Option<usize> {
        seq.verify_frame
            .and_then(Frame::batch)
            .map(|ticket| ticket.owner().rows)
    }

    pub(super) fn begin_owned_batch_forward(
        &self,
        seqs: &[&mut SequenceState],
        k: usize,
    ) -> Result<()> {
        let ks = vec![k; seqs.len()];
        self.begin_owned_batch_forward_ragged(seqs, &ks)
    }

    pub(super) fn begin_owned_batch_forward_ragged(
        &self,
        seqs: &[&mut SequenceState],
        ks: &[usize],
    ) -> Result<()> {
        ensure!(
            ks.len() == seqs.len(),
            "batch forward width vector length {} differs from frame size {}",
            ks.len(),
            seqs.len()
        );
        let mut rows = Vec::with_capacity(seqs.len());
        let mut row_base = 0usize;
        for (seq, &k_s) in seqs.iter().zip(ks.iter()) {
            let lane = self
                .batch_lane(seq, k_s, false)?
                .context("batch forward has no generation-bearing lane")?;
            rows.push(FrameRow { lane, row_base });
            row_base = row_base
                .checked_add(k_s)
                .context("batch forward row offset overflow")?;
        }
        let reservation = rows
            .first()
            .context("empty native K5 batch forward")?
            .lane
            .reservation();
        let mut shared = self.verify_transactions.lock();
        self.verify_batch_transactions
            .lock()
            .begin_frame(&mut shared, reservation, &rows)
            .map_err(|error| {
                super::verify_frame::protocol_error("batch forward owner failed", error)
            })
    }

    pub(super) fn poison_owned_batch(&self, seqs: &[&mut SequenceState]) {
        if let Some(lane) = seqs
            .first()
            .and_then(|seq| seq.verify_frame)
            .and_then(Frame::batch)
        {
            let mut shared = self.verify_transactions.lock();
            let _ = self
                .verify_batch_transactions
                .lock()
                .fail_execution::<()>(&mut shared, lane.reservation());
        }
        crate::model::verify_failure::poison_owner();
    }

    pub(super) fn record_owned_batch_execution(&self, seqs: &[&mut SequenceState]) {
        if std::env::var("ATLAS_QWEN4_BATCHED_K5_LOG").ok().as_deref() != Some("1") {
            return;
        }
        if let Some(lane) = seqs
            .first()
            .and_then(|seq| seq.verify_frame)
            .and_then(Frame::batch)
        {
            let mut actual_frame_rows = 0usize;
            let mut min_k = usize::MAX;
            let mut max_k = 0usize;
            for seq in seqs {
                if let Some(k_s) = Self::batch_lane_rows(seq) {
                    actual_frame_rows += k_s;
                    min_k = min_k.min(k_s);
                    max_k = max_k.max(k_s);
                }
            }
            tracing::info!(
                cohort_owner = ?lane.reservation(),
                actual_frame_sequences = seqs.len(),
                actual_frame_rows,
                min_k,
                max_k,
                k = max_k,
                ragged = min_k != max_k,
                fallback = false,
                "QWEN4_OWNED_BATCH_EXECUTED"
            );
        }
    }

    pub(super) fn end_owned_verify_batch_frame(&self, seqs: &[&SequenceState]) -> Result<()> {
        let lane = seqs
            .first()
            .and_then(|seq| seq.verify_frame)
            .and_then(Frame::batch)
            .context("batch frame end has no cohort lane")?;
        let mut shared = self.verify_transactions.lock();
        self.verify_batch_transactions
            .lock()
            .end_frame(&mut shared, lane.reservation())
            .map_err(|error| super::verify_frame::protocol_error("batch frame end failed", error))
    }

    pub(super) fn finish_owned_verify_batch(&self, seqs: &[&SequenceState]) -> Result<()> {
        let lane = seqs
            .first()
            .and_then(|seq| seq.verify_frame)
            .and_then(Frame::batch)
            .context("batch finish has no cohort lane")?;
        let mut shared = self.verify_transactions.lock();
        self.verify_batch_transactions
            .lock()
            .finish(&mut shared, lane.reservation())
            .map_err(|error| super::verify_frame::protocol_error("batch finish failed", error))?;
        if std::env::var("ATLAS_QWEN4_BATCHED_K5_LOG").ok().as_deref() == Some("1") {
            tracing::info!(cohort_owner = ?lane.reservation(), cohort_sequences = seqs.len(),
                k = lane.owner().rows, "QWEN4_OWNED_BATCH_RELEASED");
        }
        Ok(())
    }

    pub(super) fn retire_owned_batch_terminal(&self, seq: &SequenceState) -> Result<()> {
        let rows = Self::batch_lane_rows(seq).context("terminal batch lane is absent")?;
        let lane = self
            .batch_lane(seq, rows, false)?
            .context("terminal batch lane is absent")?;
        let mut shared = self.verify_transactions.lock();
        self.verify_batch_transactions
            .lock()
            .retire_committed(&mut shared, lane)
            .map_err(|error| {
                super::verify_frame::protocol_error("terminal batch retirement failed", error)
            })
    }

    pub(super) fn commit_owned_batch_state(
        &self,
        seq: &mut SequenceState,
        lane: LaneTicket,
        accepted: usize,
        rows: usize,
        last_slot: usize,
    ) -> Result<()> {
        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
        let k = lane.owner().rows;
        ensure!(
            (1..=k).contains(&accepted) && rows == k && last_slot == accepted - 1,
            "native batch commit prefix is invalid"
        );
        ensure!(
            lane.owner().cursor.checked_add(accepted) == Some(seq.seq_len)
                && seq.tokens.len() == seq.seq_len,
            "native K5 batch commit host extent differs from its accepted prefix"
        );
        ensure!(
            self.verify_batch_transactions.lock().state(lane) == Some(LaneState::InFlight),
            "native K5 batch commit is not in flight"
        );
        let result = if super::trait_impl::verify_qwen4_batched::take_dedup_follower_commit_skip(
            seq.slot_idx,
        ) {
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)
        } else {
            if self.qwen4_tp_batched_verify_supported_dispatch()
                && self
                    .comm
                    .as_ref()
                    .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
            {
                self.ep_broadcast_u32(crate::traits::EP_CMD_COMMIT_VERIFY_SLOT)?;
                self.ep_broadcast_u32(seq.slot_idx as u32)?;
                self.ep_broadcast_u32(accepted as u32)?;
                self.ep_broadcast_u32(rows as u32)?;
                self.ep_broadcast_u32(last_slot as u32)?;
            }
            self.commit_verify_state_async_dispatch(seq, accepted, rows, last_slot)
        };
        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        if result.is_ok() {
            batch
                .commit_succeeded(&mut shared, lane, accepted)
                .map_err(|error| {
                    super::verify_frame::protocol_error("batch commit owner failed", error)
                })?;
            seq.verify_frame
                .as_mut()
                .context("batch commit lost sequence frame")?
                .committed(accepted);
        } else {
            let _ = batch.commit_failed(&mut shared, lane);
            crate::model::verify_failure::poison_owner();
        }
        result
    }

    /// Batched sibling of [`Self::commit_owned_batch_state`]: validates every
    /// lane ticket up front, executes one cached-pointer-table GPU commit via
    /// [`Self::commit_verify_batch_cached`], and transitions every lane's
    /// ownership ledger state cleanly. Falls back per entry for singletons,
    /// TP worker broadcasts, non-owned sequences, or disabled commit cache.
    pub(super) fn commit_owned_verify_batch_cached(
        &self,
        entries: &mut [(&mut SequenceState, usize, usize)],
        stream: u64,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        if entries.is_empty() {
            return Ok(());
        }
        let tp_broadcast = self.qwen4_tp_batched_verify_supported_dispatch()
            && self
                .comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() == 0);
        if entries.len() < 2 || tp_broadcast || !self.supports_verify_batch_cached() {
            for (seq, accepted, rows) in entries.iter_mut() {
                let last_slot = accepted.saturating_sub(1);
                self.commit_owned_verify_state(seq, *accepted, *rows, last_slot)?;
            }
            return Ok(());
        }

        // First validate whether every entry belongs to an owned batch lane.
        let mut lanes: Vec<(LaneTicket, usize)> = Vec::with_capacity(entries.len());
        let mut all_owned_batch = true;
        for (seq, accepted, rows) in entries.iter() {
            let Some(lane) = self.batch_lane(seq, *rows, false)? else {
                all_owned_batch = false;
                break;
            };
            let k = lane.owner().rows;
            ensure!(
                (1..=k).contains(accepted) && *rows == k,
                "native batch commit prefix is invalid"
            );
            ensure!(
                lane.owner().cursor.checked_add(*accepted) == Some(seq.seq_len)
                    && seq.tokens.len() == seq.seq_len,
                "native K5 batch commit host extent differs from its accepted prefix"
            );
            ensure!(
                self.verify_batch_transactions.lock().state(lane) == Some(LaneState::InFlight),
                "native K5 batch commit is not in flight"
            );
            lanes.push((lane, *accepted));
        }

        if !all_owned_batch {
            for (seq, accepted, rows) in entries.iter_mut() {
                let last_slot = accepted.saturating_sub(1);
                self.commit_owned_verify_state(seq, *accepted, *rows, last_slot)?;
            }
            return Ok(());
        }

        // Partition dedup followers (which skip GDN commit) from active slots
        // only after every lane validation above succeeded.
        let mut active: Vec<(&mut SequenceState, usize, usize)> = Vec::with_capacity(entries.len());
        let mut any_dedup_skip = false;
        for (seq, accepted, rows) in entries.iter_mut() {
            if super::trait_impl::verify_qwen4_batched::take_dedup_follower_commit_skip(
                seq.slot_idx,
            ) {
                any_dedup_skip = true;
            } else {
                active.push((&mut **seq, *accepted, *rows));
            }
        }

        let result = if !active.is_empty() {
            self.commit_verify_batch_cached(&mut active, stream)
        } else if any_dedup_skip {
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)
        } else {
            Ok(())
        };
        drop(active);

        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        if result.is_ok() {
            for ((seq, _, _), &(lane, accepted)) in entries.iter_mut().zip(&lanes) {
                batch
                    .commit_succeeded(&mut shared, lane, accepted)
                    .map_err(|error| {
                        super::verify_frame::protocol_error("batch commit owner failed", error)
                    })?;
                seq.verify_frame
                    .as_mut()
                    .context("batch commit lost sequence frame")?
                    .committed(accepted);
            }
        } else {
            for &(lane, _) in &lanes {
                let _ = batch.commit_failed(&mut shared, lane);
            }
            crate::model::verify_failure::poison_owner();
        }
        result
    }

    pub(super) fn abort_owned_batch_lane(
        &self,
        seq: &mut SequenceState,
        lane: LaneTicket,
    ) -> Result<()> {
        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
        let state = self
            .verify_batch_transactions
            .lock()
            .state(lane)
            .context("batch cleanup lost its lane state")?;
        let result = match state {
            LaneState::Reserved | LaneState::Bound | LaneState::Aborted => Ok(()),
            LaneState::InFlight => {
                if self.qwen4_tp_batched_verify_supported_dispatch()
                    && self
                        .comm
                        .as_ref()
                        .is_some_and(|c| c.rank() == 0 && c.world_size() > 1)
                {
                    self.ep_broadcast_u32(crate::traits::EP_CMD_ABORT_VERIFY_SLOT)?;
                    self.ep_broadcast_u32(seq.slot_idx as u32)?;
                    self.ep_broadcast_u32(lane.owner().rows as u32)?;
                    self.ep_broadcast_u32(lane.owner().cursor as u32)?;
                }
                self.abort_verify_state_dispatch(seq, lane.owner().rows)
            }
            LaneState::CommittedAwaitingHidden { .. }
            | LaneState::Published { .. }
            | LaneState::Retired { .. } => {
                let secondary = self.gpu.synchronize(self.secondary_stream);
                let primary = self.gpu.synchronize(self.gpu.default_stream());
                secondary.and(primary)
            }
        };
        self.discard_native_verify_frame(seq);
        if result.is_ok()
            && matches!(
                state,
                LaneState::Reserved | LaneState::Bound | LaneState::InFlight
            )
        {
            seq.seq_len = lane.owner().cursor;
            seq.tokens.truncate(lane.owner().cursor);
        }
        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        if result.is_ok() {
            batch.abort_succeeded(&mut shared, lane).map_err(|error| {
                super::verify_frame::protocol_error("batch abort owner failed", error)
            })?;
            // A terminal zero-prefix abort still identifies the cohort until
            // the scheduler has ended every subframe and released scratch.
            seq.verify_frame
                .as_mut()
                .context("batch abort lost sequence frame")?
                .committed(0);
        } else {
            let _ = batch.abort_failed(&mut shared, lane);
            crate::model::verify_failure::poison_owner();
        }
        result
    }

    pub(super) fn abort_owned_verify_batch(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        let lane = seqs
            .iter()
            .find_map(|seq| seq.verify_frame.and_then(Frame::batch))
            .context("batch cleanup has no cohort lane")?;
        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        let result = batch.cleanup_all(&mut shared, lane.reservation(), |ticket, state| {
            let Some(seq) = seqs.iter_mut().find(|seq| {
                seq.slot_idx == ticket.owner().slot
                    && seq.session_hash == ticket.owner().session
                    && seq.verify_frame.and_then(Frame::batch) == Some(ticket)
            }) else {
                return Err(());
            };
            if !seq.verify_frame.is_some_and(|frame| {
                frame.matches_sequence(
                    seq.slot_idx,
                    seq.session_hash,
                    seq.seq_len,
                    ticket.owner().rows,
                )
            }) {
                return Err(());
            }
            let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)
                .map_err(|_| ())?;
            let result = match state {
                LaneState::Reserved | LaneState::Bound | LaneState::Aborted => Ok(()),
                LaneState::InFlight => {
                    if self.qwen4_tp_batched_verify_supported_dispatch()
                        && self
                            .comm
                            .as_ref()
                            .is_some_and(|c| c.rank() == 0 && c.world_size() > 1)
                    {
                        self.ep_broadcast_u32(crate::traits::EP_CMD_ABORT_VERIFY_SLOT)
                            .map_err(|_| ())?;
                        self.ep_broadcast_u32(seq.slot_idx as u32).map_err(|_| ())?;
                        self.ep_broadcast_u32(ticket.owner().rows as u32)
                            .map_err(|_| ())?;
                        self.ep_broadcast_u32(ticket.owner().cursor as u32)
                            .map_err(|_| ())?;
                    }
                    self.abort_verify_state_dispatch(seq, ticket.owner().rows)
                }
                // These prefixes may already have been emitted. Drain their
                // pending work, but never roll them back to preframe state.
                LaneState::CommittedAwaitingHidden { .. }
                | LaneState::Published { .. }
                | LaneState::Retired { .. } => {
                    let secondary = self.gpu.synchronize(self.secondary_stream);
                    let primary = self.gpu.synchronize(self.gpu.default_stream());
                    secondary.and(primary)
                }
            };
            self.discard_native_verify_frame(seq);
            if result.is_ok()
                && matches!(
                    state,
                    LaneState::Reserved | LaneState::Bound | LaneState::InFlight
                )
            {
                seq.seq_len = ticket.owner().cursor;
                seq.tokens.truncate(ticket.owner().cursor);
            }
            if result.is_ok() {
                seq.verify_frame = None;
            }
            result.map_err(|_| ())
        });
        result
            .and_then(|()| batch.finish_cancelled(&mut shared, lane.reservation()))
            .map_err(|error| super::verify_frame::protocol_error("batch cleanup failed", error))
    }
}
