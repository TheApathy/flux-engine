// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation cache wiring (`ATLAS_QWEN4_SEQSTATE_CACHE=1`, with
//! `ATLAS_QWEN4_SEQSTATE=1`): a Resume-mode blob of every prompt of at least
//! `ATLAS_QWEN4_SEQSTATE_CACHE_MIN_TOKENS` (default 4096) is saved when its
//! first decode step starts, and a later prompt that extends a saved one
//! restores it before its first chunk and prefills only the suffix.
//!
//! The restored state at position n is byte-for-byte the state that prompt
//! had at n, so the resumed prefill is a cold prefill split at n. That is not
//! byte-identical to an unsplit cold prefill (chunk boundaries change kernel
//! shapes); the qualification compares it with exactly that split.
//!
//! Scope: one tenant (0) until request-level API-key identity reaches the
//! model; the server refuses the cache with more than one auth token.
//! RAM tier only unless `ATLAS_QWEN4_SEQSTATE_CACHE_DISK_DIR` is set.

use std::sync::OnceLock;

use anyhow::Result;

use super::format::DIGEST_BYTES;
use super::qwen4::SeqStateMode;
use super::store::{Payload, ProcMeminfo, Scope, StateStore, StoreConfig};
use crate::model::types::TransformerModel;
use crate::traits::SequenceState;

pub(crate) const CACHE_SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE_CACHE";
const MIN_TOKENS_SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE_CACHE_MIN_TOKENS";
const MARGIN_SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE_CACHE_RAM_MARGIN_GIB";
const DISK_SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE_CACHE_DISK_DIR";
const TENANT: u64 = 0;

/// Whether the conversation cache is selected (read by the server's startup
/// checks as well).
pub fn cache_selected() -> bool {
    std::env::var(super::SELECTOR).is_ok_and(|v| v == "1")
        && std::env::var(CACHE_SELECTOR).is_ok_and(|v| v == "1")
}

fn min_tokens() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var(MIN_TOKENS_SELECTOR)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

fn store() -> Option<&'static StateStore> {
    static STORE: OnceLock<Option<StateStore>> = OnceLock::new();
    STORE
        .get_or_init(|| {
            if !cache_selected() {
                return None;
            }
            let mut cfg = StoreConfig::default();
            if let Some(gib) = std::env::var(MARGIN_SELECTOR)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
            {
                cfg.ram_margin_bytes = gib.saturating_mul(1u64 << 30);
            }
            cfg.disk_dir = std::env::var(DISK_SELECTOR)
                .ok()
                .filter(|s| !s.is_empty())
                .map(Into::into);
            match StateStore::new(cfg, Box::new(ProcMeminfo)) {
                Ok(store) => {
                    tracing::info!(
                        "QWEN4_SEQSTATE_CACHE_ENGAGED selector={CACHE_SELECTOR} min_tokens={}",
                        min_tokens()
                    );
                    Some(store)
                }
                Err(e) => {
                    tracing::error!("QWEN4_SEQSTATE_CACHE disabled: {e:#}");
                    None
                }
            }
        })
        .as_ref()
}

fn scope(fingerprint: [u8; DIGEST_BYTES]) -> Scope {
    Scope {
        tenant: TENANT,
        fingerprint,
    }
}

/// A save is due when the prompt is long enough and the sequence stands at
/// its first decode step, before any generated token was forwarded.
pub(crate) fn save_due(min: usize, prompt_len: usize, seq_len: usize) -> bool {
    prompt_len >= min && seq_len == prompt_len
}

/// A saved prefix of `n` tokens resumes a prompt of `total` only when at least
/// one prompt token is left to prefill (the logits come from it).
pub(crate) fn resume_usable(n: usize, total: usize) -> bool {
    n > 0 && n < total
}

impl TransformerModel {
    /// Called at the decode and verify entries.
    pub(crate) fn qwen4_seqstate_cache_save_if_due(&self, seq: &SequenceState) {
        let Some(store) = store() else {
            return;
        };
        if !self.config.is_qwen4_exp() || !save_due(min_tokens(), seq.prompt_len, seq.seq_len) {
            return;
        }
        let mut blob = Vec::new();
        if let Err(e) = self.save_qwen4_seqstate_mode(seq, &mut blob, SeqStateMode::Resume) {
            tracing::info!(
                "QWEN4_SEQSTATE_CACHE_SAVE_SKIPPED seq_len={} reason={e:#}",
                seq.seq_len
            );
            return;
        }
        let fingerprint = self.qwen4_seqstate_fingerprint(&self.kv_cache.lock());
        let bytes = blob.len();
        match store.put(scope(fingerprint), seq.tokens.clone(), blob) {
            Ok(events) => {
                tracing::info!(
                    "QWEN4_SEQSTATE_CACHE_SAVED tokens={} bytes={bytes} events={events:?}",
                    seq.tokens.len()
                );
            }
            Err(e) => tracing::warn!("QWEN4_SEQSTATE_CACHE_SAVE_FAILED: {e:#}"),
        }
    }

    /// Before the first prefill chunk of `prompt` (scheduler-driven): restore
    /// the longest saved prefix into the fresh `seq` and return its length,
    /// or 0. The sequence then stands at that position exactly as the saving
    /// sequence did (`seq_len` = n, `tokens` = the prefix), and the scheduler
    /// prefills the rest from chunk start n.
    pub(crate) fn qwen4_seqstate_cache_resume(
        &self,
        prompt: &[u32],
        seq: &mut SequenceState,
    ) -> Result<usize> {
        let Some(store) = store() else {
            return Ok(0);
        };
        if !self.config.is_qwen4_exp()
            || !seq.block_table.is_empty()
            || seq.seq_len != 0
            || self.tokens_have_vision_pad(prompt)
        {
            return Ok(0);
        }
        let fingerprint = {
            let kv = self.kv_cache.lock();
            self.qwen4_seqstate_fingerprint(&kv)
        };
        let Some(hit) = store.lookup(&scope(fingerprint), prompt) else {
            return Ok(0);
        };
        if !resume_usable(hit.tokens, prompt.len()) {
            return Ok(0);
        }
        let blob = match hit.payload {
            Payload::Ram(blob) => blob,
            Payload::Disk(path) => std::sync::Arc::new(std::fs::read(&path)?),
        };
        // The saved table may hold more blocks than the prefix needs (blocks
        // reserved ahead of the first decode token): import exactly those.
        let num_blocks = {
            let mut peek: &[u8] = &blob;
            super::format::BlobReader::open(&mut peek)?
                .header()
                .num_blocks as usize
        };
        let session_hash = seq.session_hash;
        let mut reader: &[u8] = &blob;
        // A failed import returned its blocks but may have half-written the
        // slot's recurrent state: the caller fails this prefill (and frees
        // the slot) rather than continuing as a miss.
        let mut kv = self.kv_cache.lock();
        self.restore_qwen4_seqstate_in(seq, num_blocks, &mut reader, &mut kv, SeqStateMode::Resume)
            .map_err(|e| e.context("seqstate cache: resume import failed"))?;
        drop(kv);
        seq.prompt_len = prompt.len();
        seq.cached_prefix_tokens = hit.tokens;
        seq.session_hash = session_hash;
        tracing::info!(
            "QWEN4_SEQSTATE_CACHE_HIT tokens={} of {} blocks={num_blocks}",
            hit.tokens,
            prompt.len()
        );
        Ok(hit.tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{resume_usable, save_due};

    #[test]
    fn save_only_at_the_first_decode_step_of_a_long_prompt() {
        assert!(save_due(4096, 5000, 5000));
        assert!(
            !save_due(4096, 5000, 5001),
            "a generated token was forwarded"
        );
        assert!(!save_due(4096, 4095, 4095), "prompt too short");
        assert!(!save_due(4096, 5000, 4000), "still prefilling");
    }

    #[test]
    fn resume_needs_a_token_left_to_prefill() {
        assert!(resume_usable(100, 101));
        assert!(!resume_usable(101, 101), "nothing left to produce logits");
        assert!(!resume_usable(0, 50));
    }
}
