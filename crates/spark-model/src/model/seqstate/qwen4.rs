// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 / Flash-Next adapter: which device bytes make up one sequence, and
//! the export (`save_qwen4_seqstate`) / import (`restore_qwen4_seqstate`)
//! entry points behind the swap dispatch.
//!
//! Section order (one function builds it for both directions):
//! target K/V per cache layer, QSA raw/compressed per attention layer, GDN
//! h/conv per linear-attention layer, PLE live/checkpoint, native-MTP K/V,
//! native-MTP retained hidden prefix.

use anyhow::{Context, Result, ensure};
use atlas_core::config::LayerType;
use ring::digest::{Context as DigestContext, SHA256};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::engine::{self, StateRegion, kv_block_regions};
use super::format::{BlobReader, DIGEST_BYTES, SectionKind, SeqStateHeader, flags};
use super::{ProposerSeqMeta, ProposerSeqState};
use crate::layer::SsmLayerState;
use crate::model::types::TransformerModel;
use crate::speculative::DraftProposerKind;
use crate::traits::{Model, SequenceState};

/// What decides whether a sequence's state can be captured right now. Kept
/// separate from the model so every refusal is unit-tested.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExportFacts {
    pub rotary_identity: bool,
    pub verify_frame_bound: bool,
    pub deferred_mtp_replay: bool,
    pub disk_blocks: bool,
    pub prefix_cache_active: bool,
    pub alt_proposer_state: bool,
    pub proposer_state: bool,
    pub native_mtp_proposer: bool,
    pub mixed_h_representation: bool,
    pub qsa_required: bool,
    pub qsa_cache_present: bool,
}

/// Refuses every state the blob cannot reproduce exactly. A refusal leaves
/// the sequence untouched; the caller keeps it resident instead of swapping.
pub(crate) fn check_exportable(f: &ExportFacts) -> Result<()> {
    ensure!(
        f.rotary_identity,
        "seqstate: image (MRoPE) rotary state is not exportable"
    );
    ensure!(
        !f.verify_frame_bound,
        "seqstate: a verify frame is bound to the sequence (not quiescent)"
    );
    ensure!(
        !f.deferred_mtp_replay,
        "seqstate: a deferred native-MTP prompt replay is pending (not quiescent)"
    );
    ensure!(
        !f.disk_blocks,
        "seqstate: high-speed-swap disk blocks are not exportable"
    );
    // free_sequence releases prefix-cache refs by token prefix; a restored
    // sequence on fresh blocks would release refs it never took and let the
    // radix tree evict blocks other sequences still read.
    ensure!(
        !f.prefix_cache_active,
        "seqstate: swap with an active prefix cache is not supported yet"
    );
    ensure!(
        !f.alt_proposer_state,
        "seqstate: a parked second proposer arm is not exportable"
    );
    ensure!(
        !f.proposer_state || f.native_mtp_proposer,
        "seqstate: only the native-MTP proposer state is exportable"
    );
    ensure!(
        !f.mixed_h_representation,
        "seqstate: GDN layers disagree on the FP16 h-state representation"
    );
    ensure!(
        !f.qsa_required || f.qsa_cache_present,
        "seqstate: the sequence requires QSA but no layer exposes its side cache"
    );
    Ok(())
}

/// `ATLAS_*` variables that cannot change kernel numerics or state layout:
/// paths, logging, diagnostics output, ports. Everything else counts, so an
/// unlisted selector invalidates blobs (a missed resume) rather than letting
/// a numerics change through (a wrong resume).
const NON_NUMERIC_ENV_PREFIXES: &[&str] = &[
    "ATLAS_HOME",
    "ATLAS_LOG",
    "ATLAS_TRACE",
    "ATLAS_DUMP",
    "ATLAS_PROFILE",
    "ATLAS_METRICS",
    "ATLAS_PORT",
    "ATLAS_HOST",
    "ATLAS_REQUEST_DUMP",
    "ATLAS_STARTUP_LEDGER",
];

/// The sorted `ATLAS_*` selectors that enter the fingerprint.
pub(crate) fn numerics_env(vars: impl Iterator<Item = (String, String)>) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = vars
        .filter(|(k, _)| {
            k.starts_with("ATLAS_")
                && !k.starts_with(super::SELECTOR)
                && !NON_NUMERIC_ENV_PREFIXES.iter().any(|p| k.starts_with(p))
        })
        .collect();
    env.sort();
    env
}

/// SHA-256 of the running executable, computed once: kernels, PTX and every
/// compiled-in default belong to the build that wrote a blob.
fn binary_digest() -> [u8; DIGEST_BYTES] {
    static DIGEST: std::sync::OnceLock<[u8; DIGEST_BYTES]> = std::sync::OnceLock::new();
    *DIGEST.get_or_init(|| {
        let mut ctx = DigestContext::new(&SHA256);
        let read = std::env::current_exe().and_then(|path| {
            use std::io::Read;
            let mut f = std::fs::File::open(path)?;
            let mut buf = vec![0u8; 8 << 20];
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 {
                    return Ok(());
                }
                ctx.update(&buf[..n]);
            }
        });
        let mut out = [0u8; DIGEST_BYTES];
        match read {
            Ok(()) => out.copy_from_slice(ctx.finish().as_ref()),
            Err(e) => {
                // Unknown build identity: a per-process random value makes
                // every blob from another process fail the fingerprint.
                tracing::warn!(
                    "seqstate: cannot hash the executable ({e}); blobs are process-local"
                );
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
                    ^ u128::from(std::process::id());
                out[..16].copy_from_slice(&nonce.to_le_bytes());
            }
        }
        out
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What a blob is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SeqStateMode {
    /// Swap and the qualification round trip: every byte the sequence carries,
    /// proposer included, so decoding continues exactly where it stopped.
    Exact,
    /// Conversation cache: target state only. The import feeds a prefill of the
    /// prompt suffix, and the native-MTP head starts fresh as on a cold prefill
    /// (its prompt replay covers only the last rows of any prompt).
    Resume,
}

/// Device ranges of the proposer and its retained hidden frame, resolved by
/// the caller for export or import.
struct ProposerParts {
    regions: Vec<StateRegion>,
    hidden: Option<(DevicePtr, usize)>,
}

impl TransformerModel {
    /// Model, cache geometry and numerics identity. Every `ATLAS_*` selector
    /// is included because many change kernel numerics; a blob therefore
    /// restores only into a process configured exactly like its writer.
    pub(crate) fn qwen4_seqstate_fingerprint(&self, kv: &PagedKvCache) -> [u8; DIGEST_BYTES] {
        let mut ctx = DigestContext::new(&SHA256);
        let mut put = |s: String| {
            ctx.update(&(s.len() as u64).to_le_bytes());
            ctx.update(s.as_bytes());
        };
        put("flux-seqstate-v1".into());
        let c = &self.config;
        put(format!(
            "layers={} hidden={} kv_heads={} head_dim={} vocab={}",
            c.num_hidden_layers, c.hidden_size, c.num_key_value_heads, c.head_dim, c.vocab_size
        ));
        put(format!(
            "types={:?}",
            (0..self.layers.len())
                .map(|i| c.layer_type(i))
                .collect::<Vec<_>>()
        ));
        put(format!(
            "block_size={} kv_layers={}",
            kv.block_size(),
            kv.num_layers()
        ));
        for layer in 0..kv.num_layers() {
            put(format!(
                "kv{layer}={}:{}:{}",
                kv.dtype_for_layer(layer),
                kv.k_block_stride_bytes_for_layer(layer),
                kv.v_block_stride_bytes_for_layer(layer)
            ));
        }
        put(format!(
            "ssm h={} conv={}",
            self.ssm_pool.h_bytes, self.ssm_pool.conv_bytes
        ));
        put(format!("ple={}", self.qwen4_ple.is_some()));
        put(format!(
            "proposer={:?}",
            self.active_proposer().map(|p| p.kind())
        ));
        put(format!(
            "transform={:?}",
            self.gpu.transform_cache_identity()
        ));
        // The resolved rotary schedule: any change makes saved K stale.
        put(format!(
            "rope theta={} partial={} rotary_dim={} max_pos={} mrope={:?}/{} \
             yarn={}:{}:{}:{}:{}:{}",
            c.rope_theta,
            c.partial_rotary_factor,
            c.rotary_dim,
            c.max_position_embeddings,
            c.mrope_section,
            c.mrope_interleaved,
            c.yarn_factor,
            c.yarn_beta_slow,
            c.yarn_beta_fast,
            c.yarn_original_max_position_embeddings,
            c.yarn_mscale,
            c.yarn_mscale_all_dim
        ));
        put(format!("binary={}", hex(&binary_digest())));
        for (k, v) in numerics_env(std::env::vars()) {
            put(format!("{k}={v}"));
        }
        let mut out = [0u8; DIGEST_BYTES];
        out.copy_from_slice(ctx.finish().as_ref());
        out
    }

    fn qwen4_seqstate_h_fp16(&self, seq: &SequenceState) -> Result<(bool, bool)> {
        let mut seen: Option<bool> = None;
        let mut mixed = false;
        for (i, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .with_context(|| format!("seqstate: expected SsmLayerState at layer {i}"))?;
            match seen {
                None => seen = Some(ssm.h_is_f16),
                Some(prev) if prev != ssm.h_is_f16 => mixed = true,
                Some(_) => {}
            }
        }
        Ok((seen.unwrap_or(false), mixed))
    }

    /// The ordered device ranges of `seq`, for either direction. `create_qsa`
    /// builds a lazy QSA cache on import.
    fn qwen4_seqstate_regions(
        &self,
        seq: &SequenceState,
        kv: &PagedKvCache,
        qsa_required: bool,
        create_qsa: bool,
        proposer: ProposerParts,
    ) -> Result<Vec<StateRegion>> {
        let gpu = self.gpu.as_ref();
        let mut regions =
            kv_block_regions(kv, &seq.block_table, SectionKind::KvK, SectionKind::KvV)?;

        if qsa_required {
            let mut attention = 0u16;
            for (i, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    continue;
                }
                let mut raw = StateRegion::new(SectionKind::QsaRaw, attention);
                let mut compressed = StateRegion::new(SectionKind::QsaCompressed, attention);
                let mut present = false;
                for &block in &seq.block_table {
                    match layer.qwen4_qsa_seqstate_block_regions(gpu, kv, block, create_qsa)? {
                        Some([(rp, rn), (cp, cn)]) => {
                            present = true;
                            raw.push(rp, rn);
                            compressed.push(cp, cn);
                        }
                        None => break,
                    }
                }
                ensure!(
                    present || seq.block_table.is_empty(),
                    "seqstate: attention layer {i} has no QSA side cache for a QSA sequence"
                );
                regions.push(raw);
                regions.push(compressed);
                attention += 1;
            }
        }

        for (i, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .with_context(|| format!("seqstate: expected SsmLayerState at layer {i}"))?;
            let index = u16::try_from(i)?;
            let mut h = StateRegion::new(SectionKind::GdnH, index);
            h.push(ssm.h_state, self.ssm_pool.h_bytes);
            let mut conv = StateRegion::new(SectionKind::GdnConv, index);
            conv.push(ssm.conv_state, self.ssm_pool.conv_bytes);
            regions.push(h);
            regions.push(conv);
        }

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = &self.qwen4_ple {
            let [(lp, ln), (cp, cn)] = ple.seqstate_slot_regions(seq.slot_idx)?;
            let mut live = StateRegion::new(SectionKind::PleLive, 0);
            live.push(lp, ln);
            let mut checkpoint = StateRegion::new(SectionKind::PleCheckpoint, 0);
            checkpoint.push(cp, cn);
            regions.push(live);
            regions.push(checkpoint);
        }

        regions.extend(proposer.regions);
        if let Some((ptr, bytes)) = proposer.hidden {
            let mut hidden = StateRegion::new(SectionKind::MtpHidden, 0);
            hidden.push(ptr, bytes);
            regions.push(hidden);
        }
        Ok(regions)
    }

    fn qwen4_seqstate_proposer_export(
        &self,
        seq: &SequenceState,
    ) -> Result<Option<(ProposerSeqState, Option<(usize, usize, DevicePtr, usize)>)>> {
        let Some(state) = seq.proposer_state.as_deref() else {
            return Ok(None);
        };
        let proposer = self
            .active_proposer()
            .context("seqstate: proposer state without an active proposer")?;
        let exported = proposer
            .seqstate_export(state)?
            .context("seqstate: the active proposer cannot export its state")?;
        let hidden = match &self.qwen4_mtp_hidden_frames {
            Some((storage, frames)) => frames
                .lock()
                .retained_for_export(seq.slot_idx, seq.session_hash)
                .map_err(anyhow::Error::msg)?
                .map(|(position, row, prefix)| {
                    (position, row, storage.offset(prefix.offset), prefix.bytes)
                }),
            None => None,
        };
        Ok(Some((exported, hidden)))
    }

    /// Writes `seq`'s exact state. Read-only on the sequence.
    pub(crate) fn save_qwen4_seqstate(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        self.save_qwen4_seqstate_mode(seq, writer, SeqStateMode::Exact)
    }

    pub(crate) fn save_qwen4_seqstate_mode(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
        mode: SeqStateMode,
    ) -> Result<()> {
        let exact = mode == SeqStateMode::Exact;
        let (h_fp16, mixed) = self.qwen4_seqstate_h_fp16(seq)?;
        let native_mtp = self
            .active_proposer()
            .is_some_and(|p| p.kind() == DraftProposerKind::Qwen4NativeMtp);
        let kv = self.kv_cache.lock();
        let qsa_cache_present = seq.block_table.first().is_some_and(|&b| {
            self.layers.iter().any(|layer| {
                layer
                    .qwen4_qsa_seqstate_block_regions(self.gpu.as_ref(), &kv, b, false)
                    .ok()
                    .flatten()
                    .is_some()
            })
        }) || seq.block_table.is_empty();
        check_exportable(&ExportFacts {
            rotary_identity: seq.rotary_positions.is_identity(),
            verify_frame_bound: seq.verify_frame.is_some(),
            // Resume carries no proposer state, so a pending MTP replay is
            // irrelevant to it.
            deferred_mtp_replay: exact
                && crate::model::host_bucket::pending_matches(
                    self.k8a_lastk_deferred.lock().as_ref(),
                    seq.slot_idx,
                ),
            disk_blocks: !seq.disk_block_ids.is_empty(),
            prefix_cache_active: self.prefix_cache.is_active(),
            alt_proposer_state: exact && seq.proposer_state_alt.is_some(),
            proposer_state: exact && seq.proposer_state.is_some(),
            native_mtp_proposer: native_mtp,
            mixed_h_representation: mixed,
            qsa_required: seq.qwen4_qsa_required,
            qsa_cache_present,
        })?;
        ensure!(
            seq.tokens.len() == seq.seq_len,
            "seqstate: {} tokens for seq_len {}",
            seq.tokens.len(),
            seq.seq_len
        );

        let proposer = if exact {
            self.qwen4_seqstate_proposer_export(seq)?
        } else {
            None
        };
        let mut header = SeqStateHeader {
            fingerprint: self.qwen4_seqstate_fingerprint(&kv),
            seq_len: seq.seq_len as u64,
            block_size: u32::try_from(kv.block_size())?,
            num_blocks: u32::try_from(seq.block_table.len())?,
            flags: 0,
            prompt_len: seq.prompt_len as u64,
            mtp_num_blocks: 0,
            mtp_hidden_selected_row: 0,
            mtp_seq_len: 0,
            session_hash: seq.session_hash,
            mtp_last_num_drafted: 0,
            mtp_hidden_position: 0,
            layout_digest: [0u8; DIGEST_BYTES],
            tokens: seq.tokens.clone(),
            mtp_last_drafts: Vec::new(),
        };
        if seq.qwen4_qsa_required {
            header.flags |= flags::QSA_REQUIRED;
        }
        if h_fp16 {
            header.flags |= flags::H_FP16;
        }
        let mut parts = ProposerParts {
            regions: Vec::new(),
            hidden: None,
        };
        if let Some((state, hidden)) = proposer {
            header.flags |= flags::MTP;
            header.mtp_num_blocks = u32::try_from(state.meta.num_blocks)?;
            header.mtp_seq_len = state.meta.seq_len as u64;
            header.mtp_last_num_drafted = u32::try_from(state.meta.last_num_drafted)?;
            header.mtp_last_drafts = state.meta.last_drafts;
            parts.regions = state.regions;
            if let Some((position, row, ptr, bytes)) = hidden {
                header.flags |= flags::MTP_HIDDEN;
                header.mtp_hidden_position = position as u64;
                header.mtp_hidden_selected_row = u32::try_from(row)?;
                parts.hidden = Some((ptr, bytes));
            }
        }
        let regions =
            self.qwen4_seqstate_regions(seq, &kv, seq.qwen4_qsa_required, false, parts)?;
        // The ranges stay valid without the lock: the sequence owns its blocks
        // and the side caches never move. Synchronizing under the lock could
        // wait on a secondary stream whose host side needs it.
        drop(kv);

        // Every producer of these bytes must have retired before the copies.
        self.sync_secondary()?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        engine::capture(self.gpu.as_ref(), &header, &regions, writer)?;
        tracing::info!(
            "QWEN4_SEQSTATE_SAVED slot={} seq_len={} blocks={} mtp={} hidden={} sections={}",
            seq.slot_idx,
            seq.seq_len,
            seq.block_table.len(),
            header.flags & flags::MTP != 0,
            header.flags & flags::MTP_HIDDEN != 0,
            regions.len()
        );
        Ok(())
    }

    /// Imports a blob into `seq`, a sequence fresh from `alloc_sequence`
    /// (slot claimed, nothing else). On error every block this call claimed
    /// is returned; the caller still owns (and must free) the slot.
    pub(crate) fn restore_qwen4_seqstate(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        let mut kv = self.kv_cache.lock();
        self.restore_qwen4_seqstate_in(seq, num_blocks, reader, &mut kv, SeqStateMode::Exact)
    }

    /// [`Self::restore_qwen4_seqstate`] for a caller that already holds the
    /// target cache (the prefill path).
    pub(crate) fn restore_qwen4_seqstate_in(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
        kv: &mut PagedKvCache,
        mode: SeqStateMode,
    ) -> Result<()> {
        let reader = BlobReader::open(reader)?;
        let header = reader.header().clone();
        ensure!(
            seq.block_table.is_empty() && seq.seq_len == 0 && seq.tokens.is_empty(),
            "seqstate: import needs a fresh sequence"
        );
        ensure!(
            header.num_blocks as usize == num_blocks,
            "seqstate: blob holds {} blocks, the caller expected {num_blocks}",
            header.num_blocks
        );
        ensure!(
            mode == SeqStateMode::Exact || header.flags & (flags::MTP | flags::MTP_HIDDEN) == 0,
            "seqstate: a resume import takes target-only blobs"
        );
        ensure!(
            header.fingerprint == self.qwen4_seqstate_fingerprint(kv),
            "seqstate: blob was written by a differently configured model"
        );
        ensure!(
            header.block_size as usize == kv.block_size(),
            "seqstate: block size {} != {}",
            header.block_size,
            kv.block_size()
        );
        let mut table = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match kv.alloc_block() {
                Ok(b) => table.push(b),
                Err(e) => {
                    kv.free_blocks(&table);
                    return Err(e.context("seqstate: target block allocation"));
                }
            }
        }
        seq.block_table = table;

        let result = self.qwen4_seqstate_import_claimed(seq, &header, reader, kv, mode);
        if result.is_err() {
            kv.free_blocks(&seq.block_table);
            seq.block_table.clear();
            if mode == SeqStateMode::Exact {
                if let (Some(proposer), Some(state)) =
                    (self.active_proposer(), seq.proposer_state.as_deref_mut())
                    && let Err(e) = proposer.free_state(state)
                {
                    tracing::error!(
                        "seqstate: releasing proposer blocks after a failed import: {e:#}"
                    );
                }
                if let Some((_, frames)) = &self.qwen4_mtp_hidden_frames
                    && let Err(e) = frames.lock().retire(seq.slot_idx)
                {
                    tracing::error!(
                        "seqstate: retiring the hidden frame after a failed import: {e}"
                    );
                }
            }
            return result;
        }

        seq.tokens = header.tokens.clone();
        seq.seq_len = header.seq_len as usize;
        seq.prompt_len = header.prompt_len as usize;
        seq.session_hash = header.session_hash;
        seq.qwen4_qsa_required = header.flags & flags::QSA_REQUIRED != 0;
        seq.rotary_positions = crate::traits::RotaryPositions::identity();
        let h_fp16 = header.flags & flags::H_FP16 != 0;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention
                && let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>()
            {
                ssm.h_is_f16 = h_fp16;
            }
        }
        // The slot's graphs and the QSA FP8 shadow were built for other bytes.
        self.invalidate_rotary_graphs(seq.slot_idx)?;
        crate::layers::qwen4_qsa::prefill_fp8::invalidate_shadow_ends(kv.cache_id())?;
        tracing::info!(
            "QWEN4_SEQSTATE_RESTORED mode={mode:?} slot={} seq_len={} blocks={} mtp={} hidden={}",
            seq.slot_idx,
            seq.seq_len,
            num_blocks,
            header.flags & flags::MTP != 0,
            header.flags & flags::MTP_HIDDEN != 0
        );
        Ok(())
    }

    /// Everything after the target blocks are claimed: proposer import,
    /// hidden-frame rebinding and the byte copies.
    fn qwen4_seqstate_import_claimed(
        &self,
        seq: &mut SequenceState,
        header: &SeqStateHeader,
        reader: BlobReader<'_>,
        kv: &PagedKvCache,
        mode: SeqStateMode,
    ) -> Result<()> {
        let mut parts = ProposerParts {
            regions: Vec::new(),
            hidden: None,
        };
        let mut hidden_write = None;
        if header.flags & flags::MTP != 0 {
            let proposer = self
                .active_proposer()
                .context("seqstate: blob carries MTP state but no proposer is active")?;
            ensure!(
                proposer.kind() == DraftProposerKind::Qwen4NativeMtp,
                "seqstate: blob carries native-MTP state for a different proposer"
            );
            let state = seq
                .proposer_state
                .as_deref_mut()
                .context("seqstate: blob carries MTP state but the sequence has none")?;
            let meta = ProposerSeqMeta {
                seq_len: header.mtp_seq_len as usize,
                num_blocks: header.mtp_num_blocks as usize,
                last_num_drafted: header.mtp_last_num_drafted as usize,
                last_drafts: header.mtp_last_drafts.clone(),
            };
            parts.regions = proposer
                .seqstate_import(state, &meta)?
                .context("seqstate: the active proposer cannot import state")?;
            if header.flags & flags::MTP_HIDDEN != 0 {
                let (storage, frames) = self
                    .qwen4_mtp_hidden_frames
                    .as_ref()
                    .context("seqstate: blob carries a hidden frame but no storage exists")?;
                let row = header.mtp_hidden_selected_row as usize;
                let write = frames
                    .lock()
                    .begin_write(
                        seq.slot_idx,
                        header.session_hash,
                        header.mtp_hidden_position as usize,
                        row,
                        usize::MAX,
                    )
                    .map_err(anyhow::Error::msg)?;
                parts.hidden = Some((
                    storage.offset(write.retained.prefix.offset),
                    write.retained.prefix.bytes,
                ));
                hidden_write = Some(write);
            }
        } else {
            // A resume leaves a fresh proposer state as allocated.
            ensure!(
                mode == SeqStateMode::Resume || seq.proposer_state.is_none(),
                "seqstate: blob has no proposer state but the destination sequence has one"
            );
        }

        let regions = self.qwen4_seqstate_regions(
            seq,
            kv,
            header.flags & flags::QSA_REQUIRED != 0,
            true,
            parts,
        )?;
        // Nothing may still be writing the destination ranges.
        self.sync_secondary()?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        engine::apply(self.gpu.as_ref(), reader, &regions)?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        if let (Some(write), Some((_, frames))) = (hidden_write, &self.qwen4_mtp_hidden_frames) {
            frames.lock().publish(write).map_err(anyhow::Error::msg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "qwen4_tests.rs"]
mod tests;
