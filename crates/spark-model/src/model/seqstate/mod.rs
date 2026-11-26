// SPDX-License-Identifier: AGPL-3.0-only

//! `SeqStateBlob`: one Flash-Next sequence's exact resumable state.
//!
//! A blob holds everything a sequence carries across forwards at a quiescent
//! position n (no verify frame or MTP replay pending): the target K/V blocks
//! as stored, the QSA side cache of those blocks, every GDN layer's recurrent
//! and conv state, the PLE convolution history, and the native-MTP head's K/V
//! cache, draft bookkeeping and retained target-hidden frame. Importing it
//! into a fresh sequence reproduces those bytes exactly, so continuing from
//! the import is the same computation as continuing in place.
//!
//! Opt-in with `ATLAS_QWEN4_SEQSTATE=1` (Qwen4 / Flash-Next only); otherwise
//! the legacy KV + GDN swap format is unchanged.

pub mod cache;
pub mod engine;
pub mod format;
pub mod plan;
pub(crate) mod qwen4;
pub(crate) mod roundtrip;
pub(crate) mod slot_history;
pub mod store;

pub use engine::StateRegion;
pub use format::SectionKind;

/// Native-MTP proposer bookkeeping carried in the blob header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProposerSeqMeta {
    pub seq_len: usize,
    pub num_blocks: usize,
    pub last_num_drafted: usize,
    pub last_drafts: Vec<u32>,
}

/// A proposer's exportable per-sequence state: header fields plus the device
/// ranges that hold the rest.
#[derive(Clone, Debug)]
pub struct ProposerSeqState {
    pub meta: ProposerSeqMeta,
    pub regions: Vec<StateRegion>,
}

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_SEQSTATE";

/// `ATLAS_QWEN4_SEQSTATE=1` routes Qwen4 swap save/restore through the blob.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var(SELECTOR).is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QWEN4_SEQSTATE_ENGAGED selector={SELECTOR}");
        }
        on
    })
}
