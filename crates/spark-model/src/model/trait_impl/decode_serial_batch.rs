// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness-first multi-sequence decode for models whose single-row path
//! owns state transitions that the generic batched path does not implement.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

fn serial_batch_suppress_counts() -> &'static Mutex<HashMap<usize, usize>> {
    static COUNTS: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

struct SerialBatchSuppressGuard<'a> {
    model: &'a TransformerModel,
    instance_key: usize,
    previous: bool,
}

impl<'a> SerialBatchSuppressGuard<'a> {
    fn enter(model: &'a TransformerModel) -> Self {
        let instance_key = model.instance_key();
        {
            let mut counts = serial_batch_suppress_counts().lock();
            *counts.entry(instance_key).or_insert(0) += 1;
        }
        let previous = model.suppress_graphs.swap(true, Ordering::Relaxed);
        Self {
            model,
            instance_key,
            previous,
        }
    }
}

impl Drop for SerialBatchSuppressGuard<'_> {
    fn drop(&mut self) {
        let mut counts = serial_batch_suppress_counts().lock();
        let remaining = match counts.get_mut(&self.instance_key) {
            Some(count) if *count > 1 => {
                *count -= 1;
                *count
            }
            Some(_) => {
                counts.remove(&self.instance_key);
                0
            }
            None => 0,
        };
        if remaining == 0 {
            self.model
                .suppress_graphs
                .store(self.previous, Ordering::Relaxed);
        }
    }
}

impl TransformerModel {
    pub(crate) fn serial_batch_suppress_active(&self) -> bool {
        serial_batch_suppress_counts()
            .lock()
            .get(&self.instance_key())
            .copied()
            .unwrap_or(0)
            > 0
    }

    pub(crate) fn clear_serial_batch_suppress(&self) {
        serial_batch_suppress_counts()
            .lock()
            .remove(&self.instance_key());
    }

    /// Run the proven single-sequence decoder once per row and assemble its
    /// row-zero logits into the ordinary `[batch, vocab]` output buffer.
    ///
    /// `decode()` launches on the backend default stream even when the
    /// scheduler passes stream zero to `decode_batch`.  Both staging
    /// directions therefore use that same backend stream; using the caller's
    /// stream can copy every row after the final decode and swap/duplicate
    /// outputs across requests.
    pub(super) fn decode_batch_serial_rows(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        debug_assert_eq!(n, seqs.len());
        anyhow::ensure!(
            n <= 1 || !self.decode_logits_fp32(),
            "serialized multi-sequence decode requires the BF16 batched logits arena"
        );
        let logits = self.decode_logits_ptr();
        let elem_bytes = if self.decode_logits_fp32() { 4 } else { 2 };
        let row_bytes = self
            .config
            .vocab_size
            .checked_mul(elem_bytes)
            .expect("logits row byte count overflow");
        let batch_bytes = n
            .checked_mul(row_bytes)
            .expect("batched logits byte count overflow");
        anyhow::ensure!(
            n <= self.buffers.decode_meta().rows(),
            "serialized decode batch {n} exceeds the {}-row serving layout",
            self.buffers.decode_meta().rows()
        );
        anyhow::ensure!(
            batch_bytes <= self.buffers.sizes().logits,
            "serialized decode logits need {batch_bytes} bytes, but the arena has {}",
            self.buffers.sizes().logits
        );
        let copy_stream = self.gpu.default_stream();

        // Graph caches are slot-keyed.  Capturing one row while another row
        // replays in this loop would bind persistent state from two slots into
        // one capture window.  The RAII guard restores `suppress_graphs` on
        // early error and unwind, and prevents FP8 auto-unsuppress from lifting
        // suppression while serial batching is active.
        let _suppress_guard = SerialBatchSuppressGuard::enter(self);
        let mut staged = vec![0u8; batch_bytes];
        for (row, (&token, seq)) in tokens.iter().zip(seqs.iter_mut()).enumerate() {
            if self
                .comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
            {
                self.ep_broadcast_cmd_dispatch(token)?;
            }
            if self.active_proposer().is_some_and(|p| p.is_dflash()) {
                self.reconcile_dflash_context_before_bootstrap_dispatch(seq, copy_stream)?;
            }
            self.decode(token, seq, copy_stream)?;
            if self.active_proposer().is_some_and(|p| p.is_dflash()) {
                self.capture_dflash_serial_decode_row(seq, copy_stream)?;
            }
            self.gpu.copy_d2h_on_stream(
                logits,
                &mut staged[row * row_bytes..(row + 1) * row_bytes],
                copy_stream,
            )?;
        }
        self.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&staged, logits)], copy_stream)?;
        Ok(logits)
    }
}
