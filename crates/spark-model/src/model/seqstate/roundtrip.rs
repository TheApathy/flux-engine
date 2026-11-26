// SPDX-License-Identifier: AGPL-3.0-only

//! Qualification hook for the resume-identity gate (seqstate-gates (a)).
//!
//! `ATLAS_QWEN4_SEQSTATE_ROUNDTRIP_AFTER=N` (with `ATLAS_QWEN4_SEQSTATE=1`):
//! once a sequence has decoded N tokens past its prompt, the next decode or
//! verify entry exports its blob, frees the sequence, allocates a fresh one
//! and imports the blob into it, then continues. Scheduler-side state
//! (pending drafts, sampler, grammar) is untouched, so the rest of the
//! response must be byte-identical to a run without the hook. Once per
//! sequence; a refused export (not quiescent) retries at the next entry.

use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use parking_lot::Mutex;

use crate::model::types::TransformerModel;
use crate::traits::{Model, SequenceState};

pub(crate) const ROUNDTRIP_SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE_ROUNDTRIP_AFTER";

fn roundtrip_after() -> Option<usize> {
    static AFTER: OnceLock<Option<usize>> = OnceLock::new();
    *AFTER.get_or_init(|| {
        let after = std::env::var(ROUNDTRIP_SELECTOR).ok()?.parse().ok()?;
        tracing::warn!(
            "QWEN4_SEQSTATE_ROUNDTRIP_ENGAGED selector={ROUNDTRIP_SELECTOR} after={after} \
             (qualification only)"
        );
        Some(after)
    })
}

/// Sequences already round-tripped, by (session, prompt length, first tokens).
fn done() -> &'static Mutex<HashSet<(u64, usize, Vec<u32>)>> {
    static DONE: OnceLock<Mutex<HashSet<(u64, usize, Vec<u32>)>>> = OnceLock::new();
    DONE.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Whether a sequence at `seq_len` with `prompt_len` is due.
pub(crate) fn due(after: usize, prompt_len: usize, seq_len: usize) -> bool {
    prompt_len > 0 && seq_len >= prompt_len.saturating_add(after)
}

impl TransformerModel {
    /// Called at the decode and verify entries; a no-op unless selected.
    pub(crate) fn qwen4_seqstate_roundtrip_if_due(&self, seq: &mut SequenceState) -> Result<()> {
        let Some(after) = roundtrip_after() else {
            return Ok(());
        };
        if !self.config.is_qwen4_exp()
            || !super::enabled()
            || !due(after, seq.prompt_len, seq.seq_len)
        {
            return Ok(());
        }
        let key = (
            seq.session_hash,
            seq.prompt_len,
            seq.tokens.iter().take(64).copied().collect::<Vec<_>>(),
        );
        if done().lock().contains(&key) {
            return Ok(());
        }
        let mut blob = Vec::new();
        if let Err(e) = self.save_qwen4_seqstate(seq, &mut blob) {
            tracing::info!(
                "QWEN4_SEQSTATE_ROUNDTRIP_DEFERRED seq_len={} reason={e:#}",
                seq.seq_len
            );
            return Ok(());
        }
        done().lock().insert(key);
        let num_blocks = seq.block_table.len();
        let cached_prefix_tokens = seq.cached_prefix_tokens;
        let old_slot = seq.slot_idx;
        self.free_sequence(seq)
            .context("seqstate roundtrip: free")?;
        let mut fresh = self.alloc_sequence().context("seqstate roundtrip: alloc")?;
        let mut reader: &[u8] = &blob;
        if let Err(e) = self.restore_qwen4_seqstate(&mut fresh, num_blocks, &mut reader) {
            let _ = self.free_sequence(&mut fresh);
            return Err(e.context("seqstate roundtrip: import (the sequence is lost)"));
        }
        fresh.cached_prefix_tokens = cached_prefix_tokens;
        tracing::warn!(
            "QWEN4_SEQSTATE_ROUNDTRIP_OK seq_len={} blocks={num_blocks} bytes={} slot {old_slot}->{}",
            fresh.seq_len,
            blob.len(),
            fresh.slot_idx
        );
        *seq = fresh;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::due;

    #[test]
    fn due_after_n_decoded_tokens_only() {
        assert!(!due(64, 0, 500), "no prompt recorded yet");
        assert!(!due(64, 100, 163));
        assert!(due(64, 100, 164));
        assert!(due(0, 100, 100));
        assert!(!due(usize::MAX, 100, usize::MAX - 1));
    }
}
