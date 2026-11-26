// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched flat DFlash verification.
//!
//! Verifies `c` sequences' `K = gamma + 1` flat draft windows in a single
//! batched forward (`total_rows = c * K`), hoisting:
//! - 1-launch batched token embedding (`batched_embed`);
//! - 1-batch pre-layer-0 H2D upload of token IDs, disjoint per-sequence
//!   `AttnMetadataDev` slices, and (for `c >= 4`) all 48 SSM layers' 6-table
//!   pointer tables into a fixed-address device slab;
//! - 48 SSM/GDN layers across all `total_rows` (`in_proj_qkvz`, `in_proj_ba` +
//!   gates, multi-seq Conv1d + GDN lazyfinal at `c >= 4` or Split-V=4 at
//!   `c < 4`, per-slot `gdn_lazy_retain` lanes, `out_proj`, post-norm, FFN);
//! - 16 full-attention layers across all `total_rows` (hoisted QKV + QK-norm,
//!   per-sequence RoPE/KV-write/paged-decode directly into `attn_output` slices,
//!   hoisted `sigmoid_gate_mul_batched` + `o_proj` + post-norm + FFN);
//! - single-pass final norm + batched LM head (`LM_CHUNK` up to 128) +
//!   1-launch `argmax_bf16_rows`;
//! - full-step CUDA graph capture (`ATLAS_DFLASH_BATCHED_VERIFY_GRAPH`, default
//!   ON on wide discrete GPUs `sm_count > 64`, default OFF on `<= 64`-SM GPUs).

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy};

use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, FfnDefer, ForwardContext, LayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

/// High-bit tag in `TransformerModel::batch_decode_graphs` key `(slot_ids, meta)`
/// separating flat batched DFlash verify graphs from `decode_a2` (`padded_n`)
/// and Qwen4 batched verify graphs (bits 63/62) while inheriting automatic
/// invalidation in `free_sequence`, `compact_sequence`, `rotary_graphs`, and
/// `TransformerModel::drop`.
const DFLASH_BATCHED_VERIFY_GRAPH_TAG: usize = 1usize << (usize::BITS - 3);

/// Maximum cached flat batched DFlash verify graphs per model instance.
const DFLASH_BATCHED_VERIFY_GRAPH_CAP: usize = 24;

/// Maximum consecutive capture errors before disabling graph capture for this
/// model instance and staying cleanly on eager execution.
const DFLASH_BATCHED_CAPTURE_POISON_LIMIT: u32 = 4;

/// Fixed token-ID header capacity (in `u32` elements) inside the disjoint
/// verify slab so `SSM_PTR_SLAB_OFFSET` stays constant across `total_rows <= 256`.
const TOKEN_IDS_HEADER_MIN_SLOTS: usize = 256;

/// SM-count threshold above which `ATLAS_DFLASH_BATCHED_VERIFY_GRAPH` defaults ON.
const WIDE_SM_GRAPH_THRESHOLD: u32 = 64;

type ModelTrackerId = (usize, u64);
type DflashGraphKey = (Vec<usize>, usize);

#[derive(Default)]
struct DflashBatchedGraphTrackerState {
    last_slab_base: u64,
    warmed_shapes: HashMap<(usize, usize), ()>,
    buckets: HashMap<DflashGraphKey, Vec<u32>>,
    lru: HashMap<DflashGraphKey, u64>,
    tick: u64,
    poison_count: u32,
}

static DFLASH_BATCHED_GRAPH_TRACKER: OnceLock<
    Mutex<HashMap<ModelTrackerId, DflashBatchedGraphTrackerState>>,
> = OnceLock::new();

#[inline]
const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

#[inline]
const fn is_dflash_batched_graph_meta(meta: usize) -> bool {
    (meta & DFLASH_BATCHED_VERIFY_GRAPH_TAG) != 0
}

#[inline]
#[must_use]
fn hash_seq_len_buckets(buckets: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in buckets {
        for byte in b.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// Encode the exact paged-attention split-K regime for a sequence's K-token
/// verify window `[seq_len + 1 ..= seq_len + k]`:
/// - `max_len.div_ceil(512).min(64)` matches `SPLIT_TILE = 512` / `MAX_SPLITS_CAP = 64`
///   used by both `compute_kgamma_splitk_splits` and `compute_num_splits_for_sms`.
/// - `max_len >= 1024` tracks the SM120 `kgamma_ctx_ok` threshold where
///   short-context `run_paged_decode` switches to `paged_decode_attn_kgamma`.
#[inline]
#[must_use]
const fn dflash_seq_len_graph_bucket(seq_len: usize, k: usize) -> u32 {
    let max_len = seq_len.saturating_add(k) as u32;
    let raw_tile = max_len.div_ceil(512);
    let max_tile = if raw_tile > 64 { 64 } else { raw_tile };
    let kgamma_ge_1024 = if max_len >= 1024 { 1u32 } else { 0u32 };
    (max_tile << 1) | kgamma_ge_1024
}

#[inline]
#[must_use]
const fn window_straddles_512_split_tile(seq_len: usize, k: usize) -> bool {
    let min_len = seq_len.saturating_add(1) as u32;
    let max_len = seq_len.saturating_add(k) as u32;
    min_len.div_ceil(512) != max_len.div_ceil(512)
}

fn paged_decode_splitk_env_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_PAGED_DECODE_SPLITK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[inline]
#[must_use]
fn pack_dflash_batched_graph_meta(k: usize, c: usize, buckets: &[u32]) -> usize {
    let bucket_sig = (hash_seq_len_buckets(buckets) as usize) & 0x00FF_FFFF_FFFF;
    DFLASH_BATCHED_VERIFY_GRAPH_TAG | ((k & 0xFF) << 48) | ((c & 0xFF) << 40) | bucket_sig
}

#[inline]
#[must_use]
const fn resolve_dflash_batched_graph_flag(raw: Option<&str>, sm_count: u32) -> bool {
    match raw {
        Some(v) => {
            let b = v.as_bytes();
            matches!(b, b"1" | b"true" | b"TRUE" | b"on" | b"ON")
        }
        None => sm_count > WIDE_SM_GRAPH_THRESHOLD,
    }
}

/// Gate for full-step CUDA graph capture of `decode_verify_dflash_batched`.
/// Default ON on wide discrete GPUs (`sm_count > 64`), default OFF on `<= 64`-SM
/// GPUs (GB10). Set `ATLAS_DFLASH_BATCHED_VERIFY_GRAPH=0|1` to override.
fn dflash_batched_verify_graph_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BATCHED_VERIFY_GRAPH")
            .ok()
            .map(|s| s.trim().to_owned())
    });
    resolve_dflash_batched_graph_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[inline]
#[must_use]
const fn resolve_dflash_batched_qkv_flag(raw: Option<&str>) -> bool {
    match raw {
        Some(v) => {
            let b = v.as_bytes();
            matches!(b, b"1" | b"true" | b"TRUE" | b"on" | b"ON")
        }
        None => true,
    }
}

/// Cross-sequence batched-QKV gate. Default ON (delegates to the fast
/// `ms_phase_qkv` path). Set `ATLAS_DFLASH_BATCHED_QKV=0` to force the
/// per-sequence QKV fallback for A/B testing.
fn dflash_batched_qkv_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BATCHED_QKV")
            .ok()
            .map(|s| s.trim().to_owned())
    });
    resolve_dflash_batched_qkv_flag(raw.as_deref())
}

struct PreparedBatchedMetadata {
    token_ids_dev: DevicePtr,
    attn_metas: Vec<AttnMetadataDev>,
    ssm_ptr_tables: Vec<Option<DevicePtr>>,
    slab_base: DevicePtr,
}

impl TransformerModel {
    fn dflash_batched_tracker_id(&self) -> ModelTrackerId {
        (
            std::ptr::from_ref(self) as usize,
            self.buffers.hidden_states().0,
        )
    }

    fn invalidate_dflash_batched_graphs(&self, stream: u64) -> Result<()> {
        let stale: Vec<GraphHandle> = {
            let mut graphs = self.batch_decode_graphs.lock();
            let keys: Vec<DflashGraphKey> = graphs
                .keys()
                .filter(|(_, meta)| is_dflash_batched_graph_meta(*meta))
                .cloned()
                .collect();
            keys.into_iter()
                .filter_map(|k| graphs.remove(&k))
                .filter(|g| g.0 != 0)
                .collect()
        };
        if !stale.is_empty() {
            self.gpu.synchronize(stream)?;
        }
        for graph in stale {
            if let Err(err) = self.gpu.destroy_graph(graph) {
                tracing::error!(
                    "verify_dflash_batched: destroy_graph({}) failed: {err:#}",
                    graph.0
                );
            }
        }
        if let Some(tracker) = DFLASH_BATCHED_GRAPH_TRACKER.get()
            && let Some(state) = tracker.lock().get_mut(&self.dflash_batched_tracker_id())
        {
            state.buckets.clear();
            state.lru.clear();
        }
        Ok(())
    }

    fn evict_dflash_batched_graphs_if_full(&self, stream: u64) -> Result<()> {
        let model_id = self.dflash_batched_tracker_id();
        let evicted: Vec<GraphHandle> = {
            let mut graphs = self.batch_decode_graphs.lock();
            let tracker = DFLASH_BATCHED_GRAPH_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
            let mut map = tracker.lock();
            let state = map.entry(model_id).or_default();
            state.lru.retain(|k, _| graphs.contains_key(k));
            state.buckets.retain(|k, _| graphs.contains_key(k));
            let mut to_destroy = Vec::new();
            while state.lru.len() >= DFLASH_BATCHED_VERIFY_GRAPH_CAP {
                let Some(oldest_key) = state
                    .lru
                    .iter()
                    .min_by_key(|(_, tick)| **tick)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                state.lru.remove(&oldest_key);
                state.buckets.remove(&oldest_key);
                if let Some(handle) = graphs.remove(&oldest_key)
                    && handle.0 != 0
                {
                    to_destroy.push(handle);
                }
            }
            to_destroy
        };
        if !evicted.is_empty() {
            self.gpu.synchronize(stream)?;
            for handle in evicted {
                if let Err(err) = self.gpu.destroy_graph(handle) {
                    tracing::error!(
                        "verify_dflash_batched: evict destroy_graph({}) failed: {err:#}",
                        handle.0
                    );
                }
            }
        }
        Ok(())
    }

    /// Batched DFlash verify across `c` sequences. `tokens_per_seq[s]` is
    /// sequence `s`'s `[last_token, draft0, .., draft_{gamma-1}]` (all length `K`).
    /// Every sequence MUST have the same `K` (aligned by the scheduler).
    ///
    /// Returns `Vec<Vec<u32>>`: element `s` is sequence `s`'s per-position
    /// argmax (length `K`), identical layout to `decode_verify`'s return.
    ///
    /// **Side effects per seq (identical to `decode_verify`):** `seq_len += K`,
    /// all `K` tokens pushed to `seq.tokens`. Caller rolls back per-seq and
    /// calls `commit_verify_state_async` per-seq inside `LaneGuard::enter_for_slot`.
    pub(super) fn decode_verify_dflash_batched_dispatch(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let c = seqs.len();
        if c == 0 {
            return Ok(Vec::new());
        }
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_dflash_batched: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        let k = tokens_per_seq[0].len();
        if k == 0 {
            return Ok(vec![Vec::new(); c]);
        }
        for (s, t) in tokens_per_seq.iter().enumerate() {
            if t.len() != k {
                bail!(
                    "decode_verify_dflash_batched: seq {s} has K={} != K={} (unaligned windows)",
                    t.len(),
                    k
                );
            }
        }
        for seq in seqs.iter() {
            anyhow::ensure!(
                !seq.qwen4_qsa_required,
                "cross-sequence DFlash verify does not implement per-sequence QSA transactions"
            );
            seq.rotary_positions.tail_scalar(seq.seq_len)?;
            let last = seq
                .seq_len
                .checked_add(k - 1)
                .ok_or_else(|| anyhow::anyhow!("batched verify physical extent overflow"))?;
            seq.rotary_positions.tail_scalar(last)?;
        }
        // c==1 has no batching to do — defer to the single-seq K=γ verifier
        // (binding its physical slot's retain lane so commit reads the same lane)
        // so pre-verify SSM checkpoint copy, DFlash capture, HSS, and Qwen4 paths
        // all execute identically to standalone single-sequence verify.
        if c == 1 {
            let _lane =
                crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seqs[0].slot_idx)?;
            let out = self.decode_verify_graphed_kgamma_dispatch(&tokens_per_seq[0], seqs[0], 0)?;
            return Ok(vec![out]);
        }

        anyhow::ensure!(
            !self.config.is_qwen4_exp(),
            "cross-sequence flat DFlash verify does not support Qwen4 models; use decode_verify_qwen4_batched"
        );
        let total_rows = c
            .checked_mul(k)
            .ok_or_else(|| anyhow::anyhow!("decode_verify_dflash_batched: c*K overflow"))?;
        anyhow::ensure!(
            total_rows <= self.buffers.max_batch_tokens(),
            "decode_verify_dflash_batched: total_rows={total_rows} exceeds max_batch_tokens={}",
            self.buffers.max_batch_tokens()
        );
        if self.kv_cache.lock().config().cache_blocks_per_seq.is_some() {
            bail!(
                "cross-sequence flat DFlash verify does not support HSS disk-backed KV cache \
                 (use single-sequence verify)"
            );
        }

        let stream = self.gpu.default_stream();
        let serial = crate::model::env_diag::DflashSerialControls::current();
        let serial_family = serial.active_family()?;
        if let Some(family) = serial_family {
            static PROOF: std::sync::Once = std::sync::Once::new();
            PROOF.call_once(|| {
                tracing::warn!(family, c, k, total_rows, "DFLASH_K1_BISECT active");
            });
        }

        // Dual-buffer pre-verify copy per seq (checkpoint -> live state) and
        // eager pre-warming of per-slot `gdn_lazy_retain` pools outside capture.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let max_blocks = self.max_blocks_per_seq;

        // Ensure KV blocks for every seq's K positions up front.
        for seq in seqs.iter_mut() {
            let last_pos = seq.seq_len + k - 1;
            let blocks_needed = (last_pos / bs) + 1;
            ensure_blocks(
                self,
                seq,
                blocks_needed.saturating_sub(1),
                &mut kv_cache,
                stream,
            )?;
        }

        let slot_ids: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
        let mut flat_tokens: Vec<u32> = Vec::with_capacity(total_rows);
        for toks in tokens_per_seq {
            flat_tokens.extend_from_slice(toks);
        }
        let seq_lens_per_seq: Vec<Vec<usize>> = seqs
            .iter()
            .map(|seq| (0..k).map(|t| seq.seq_len + t).collect())
            .collect();

        // Hoist ALL per-step H2D uploads (token IDs + SSM pointer tables +
        // C disjoint AttnMetadataDev slices) into ONE fixed-address device slab
        // BEFORE layer 0 (and BEFORE CUDA graph capture/replay).
        let use_slab_batched_embed = self.batched_embed_kernel.0 != 0
            && !self.config.use_fp32_residual()
            && !self.config.is_qwen4_exp();
        let prepared = self.stage_dflash_batched_slab_and_upload(
            &flat_tokens,
            &slot_ids,
            seqs,
            k,
            bs,
            max_blocks,
            use_slab_batched_embed,
            stream,
        )?;

        let hidden = self.buffers.hidden_states();
        let batched_qkv = dflash_batched_qkv_enabled();
        let lm_chunk = self.dflash_batched_lm_chunk_rows();

        // P4: Full 64-layer + final-norm + LM-head + argmax CUDA graph capture.
        let seq_len_buckets: Vec<u32> = seqs
            .iter()
            .map(|seq| dflash_seq_len_graph_bucket(seq.seq_len, k))
            .collect();
        let any_split_boundary = paged_decode_splitk_env_enabled()
            && seqs
                .iter()
                .any(|seq| window_straddles_512_split_tile(seq.seq_len, k));
        let all_identity_rotary = seqs.iter().all(|seq| seq.rotary_positions.is_identity());
        let no_serial_controls = serial_family.is_none()
            && !serial.ffn
            && !serial.layer_norms
            && !serial.final_norm
            && !serial.lm_head
            && !crate::layers::attn_out_serial_enabled()
            && !crate::layers::attn_paged_serial_enabled()
            && !crate::layers::ssm_qkvz_serial_enabled()
            && !crate::layers::ssm_out_serial_enabled()
            && !crate::layers::ssm_recurrent_serial_enabled()
            && !crate::layers::ssm_ba_serial_enabled();
        let model_id = self.dflash_batched_tracker_id();
        let poison_ok = DFLASH_BATCHED_GRAPH_TRACKER
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .get(&model_id)
            .map(|s| s.poison_count < DFLASH_BATCHED_CAPTURE_POISON_LIMIT)
            .unwrap_or(true);

        let graph_eligible = dflash_batched_verify_graph_enabled()
            && !self.profile
            && !self.suppress_graphs.load(Ordering::Relaxed)
            && !self.use_fp32_logits
            && self.comm.is_none()
            && use_slab_batched_embed
            && batched_qkv
            && no_serial_controls
            && all_identity_rotary
            && !any_split_boundary
            && total_rows <= lm_chunk
            && total_rows <= TOKEN_IDS_HEADER_MIN_SLOTS
            && poison_ok;

        let all_argmax = if graph_eligible {
            let meta_key = pack_dflash_batched_graph_meta(k, c, &seq_len_buckets);
            let graph_key: DflashGraphKey = (slot_ids.clone(), meta_key);

            // Check cache hit with exact seq_len_buckets verification.
            let cached_graph = {
                let mut graphs = self.batch_decode_graphs.lock();
                let tracker =
                    DFLASH_BATCHED_GRAPH_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
                let mut map = tracker.lock();
                let state = map.entry(model_id).or_default();
                if let Some(&handle) = graphs.get(&graph_key) {
                    if state.buckets.get(&graph_key) == Some(&seq_len_buckets) && handle.0 != 0 {
                        state.tick = state.tick.saturating_add(1);
                        state.lru.insert(graph_key.clone(), state.tick);
                        Some(handle)
                    } else {
                        // Bucket hash collision or null handle: evict and recapture.
                        let stale = graphs.remove(&graph_key);
                        state.buckets.remove(&graph_key);
                        state.lru.remove(&graph_key);
                        drop(map);
                        drop(graphs);
                        if let Some(h) = stale
                            && h.0 != 0
                        {
                            self.gpu.synchronize(stream)?;
                            let _ = self.gpu.destroy_graph(h);
                        }
                        None
                    }
                } else {
                    None
                }
            };

            if let Some(graph) = cached_graph {
                self.gpu.launch_graph(graph, stream)?;
                self.gpu.synchronize(stream)?;
                self.dflash_copy_argmax(self.buffers.scratch(), total_rows)?
            } else {
                // First time seeing shape `(c, k)`: run 1 eager warmup step so
                // cuBLASLt weight scales, workspaces, and kernel caches are
                // populated outside capture.
                let warmed = {
                    let tracker =
                        DFLASH_BATCHED_GRAPH_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
                    tracker
                        .lock()
                        .get(&model_id)
                        .is_some_and(|s| s.warmed_shapes.contains_key(&(c, k)))
                };
                if !warmed {
                    self.execute_dflash_batched_body(
                        &flat_tokens,
                        &slot_ids,
                        seqs,
                        &seq_lens_per_seq,
                        &mut kv_cache,
                        &prepared,
                        k,
                        batched_qkv,
                        use_slab_batched_embed,
                        serial,
                        false,
                        stream,
                    )?;
                    self.dflash_batched_finalize_single_pass_gpu(hidden, total_rows, stream)?;
                    self.gpu.synchronize(stream)?;
                    let argmax = self.dflash_copy_argmax(self.buffers.scratch(), total_rows)?;
                    let tracker =
                        DFLASH_BATCHED_GRAPH_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
                    tracker
                        .lock()
                        .entry(model_id)
                        .or_default()
                        .warmed_shapes
                        .insert((c, k), ());
                    argmax
                } else {
                    self.evict_dflash_batched_graphs_if_full(stream)?;
                    let capture_res: Result<GraphHandle> = (|| {
                        let capture = super::capture_guard::CaptureGuard::begin(
                            self.gpu.as_ref(),
                            stream,
                            "verify_dflash_batched",
                        )?;
                        self.execute_dflash_batched_body(
                            &flat_tokens,
                            &slot_ids,
                            seqs,
                            &seq_lens_per_seq,
                            &mut kv_cache,
                            &prepared,
                            k,
                            batched_qkv,
                            use_slab_batched_embed,
                            serial,
                            true,
                            stream,
                        )?;
                        self.dflash_batched_finalize_single_pass_gpu(hidden, total_rows, stream)?;
                        capture.end()
                    })();
                    match capture_res {
                        Ok(graph) if graph.0 != 0 => {
                            {
                                let mut graphs = self.batch_decode_graphs.lock();
                                let tracker = DFLASH_BATCHED_GRAPH_TRACKER
                                    .get_or_init(|| Mutex::new(HashMap::new()));
                                let mut map = tracker.lock();
                                let state = map.entry(model_id).or_default();
                                state.tick = state.tick.saturating_add(1);
                                state.lru.insert(graph_key.clone(), state.tick);
                                state.buckets.insert(graph_key.clone(), seq_len_buckets);
                                graphs.insert(graph_key, graph);
                            }
                            static CAPTURED_LOG: std::sync::Once = std::sync::Once::new();
                            CAPTURED_LOG.call_once(|| {
                                tracing::info!(
                                    c,
                                    k,
                                    total_rows,
                                    slab_base = format_args!("{:#x}", prepared.slab_base.0),
                                    "DFLASH_BATCHED_VERIFY_GRAPH_CAPTURED"
                                );
                            });
                            self.gpu.launch_graph(graph, stream)?;
                            self.gpu.synchronize(stream)?;
                            self.dflash_copy_argmax(self.buffers.scratch(), total_rows)?
                        }
                        Ok(_) => {
                            // Null handle (e.g. mock backend): ops already executed during capture.
                            self.gpu.synchronize(stream)?;
                            self.dflash_copy_argmax(self.buffers.scratch(), total_rows)?
                        }
                        Err(err) => {
                            tracing::warn!(
                                c,
                                k,
                                total_rows,
                                "verify_dflash_batched graph capture failed ({err:#}); falling back eagerly"
                            );
                            let tracker = DFLASH_BATCHED_GRAPH_TRACKER
                                .get_or_init(|| Mutex::new(HashMap::new()));
                            tracker.lock().entry(model_id).or_default().poison_count += 1;
                            self.execute_dflash_batched_body(
                                &flat_tokens,
                                &slot_ids,
                                seqs,
                                &seq_lens_per_seq,
                                &mut kv_cache,
                                &prepared,
                                k,
                                batched_qkv,
                                use_slab_batched_embed,
                                serial,
                                false,
                                stream,
                            )?;
                            self.dflash_batched_finalize(
                                hidden,
                                total_rows,
                                serial.final_norm,
                                serial.lm_head,
                                stream,
                            )?
                        }
                    }
                }
            }
        } else {
            self.execute_dflash_batched_body(
                &flat_tokens,
                &slot_ids,
                seqs,
                &seq_lens_per_seq,
                &mut kv_cache,
                &prepared,
                k,
                batched_qkv,
                use_slab_batched_embed,
                serial,
                false,
                stream,
            )?;
            self.dflash_batched_finalize(
                hidden,
                total_rows,
                serial.final_norm,
                serial.lm_head,
                stream,
            )?
        };

        // Slice the flat per-row argmax back into per-seq K-length windows.
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(c);
        for s in 0..c {
            results.push(all_argmax[s * k..(s + 1) * k].to_vec());
        }

        // Advance per-seq state identically to decode_verify (push K, += K).
        for (seq, toks) in seqs.iter_mut().zip(tokens_per_seq.iter()) {
            for &token in toks {
                seq.tokens.push(token);
            }
            seq.seq_len += k;
        }

        Ok(results)
    }

    /// Stage and upload ALL per-step H2D metadata into ONE fixed-address
    /// disjoint device slab before layer 0 (and before CUDA graph replay):
    /// - `[0 .. token_header_bytes)`: flattened token IDs (`[c*K] u32`)
    /// - `[ssm_slab_off .. attn_slab_off)`: 6-table SSM pointer tables for all
    ///   SSM layers (uploaded in 1 H2D copy when `c >= 4`)
    /// - `[attn_slab_off ..)`: `c` disjoint `AttnMetadataDev` slices
    fn stage_dflash_batched_slab_and_upload(
        &self,
        flat_tokens: &[u32],
        slot_ids: &[usize],
        seqs: &[&mut SequenceState],
        k: usize,
        bs: usize,
        max_blocks: u32,
        upload_token_ids: bool,
        stream: u64,
    ) -> Result<PreparedBatchedMetadata> {
        let c = seqs.len();
        let total_rows = c * k;
        let mb = max_blocks as usize;

        let token_header_bytes = align_up(
            total_rows.max(TOKEN_IDS_HEADER_MIN_SLOTS) * std::mem::size_of::<u32>(),
            256,
        );
        let num_ssm_layers = self.config.num_ssm_layers();
        let ssm_table_stride = crate::layers::qwen3_ssm::DFLASH_SSM_PTR_TABLE_BYTES;
        let ssm_slab_bytes = num_ssm_layers * ssm_table_stride;
        let ssm_slab_off = token_header_bytes;
        let attn_slab_off = align_up(ssm_slab_off + ssm_slab_bytes, 256);

        let slots_off = align_up(k * 4, 256);
        let sl_off = slots_off + align_up(k * 8, 256);
        let bt_off = sl_off + align_up(k * 4, 256);
        let per_seq_meta_stride = align_up(bt_off + k * mb * 4, 256);
        let required_bytes = attn_slab_off + c * per_seq_meta_stride;

        let (slab_base, slab_cap) = self.ensure_disjoint_verify_attn_meta(required_bytes)?;

        // If the disjoint slab was freshly allocated or reallocated at a new
        // base address, zero it once (so trailing unread block-table entries
        // are deterministically 0) and invalidate any graphs that baked the old
        // base address.
        let model_id = self.dflash_batched_tracker_id();
        let slab_changed = {
            let tracker = DFLASH_BATCHED_GRAPH_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
            let mut map = tracker.lock();
            let state = map.entry(model_id).or_default();
            if state.last_slab_base != slab_base.0 {
                state.last_slab_base = slab_base.0;
                true
            } else {
                false
            }
        };
        if slab_changed {
            self.gpu.memset_async(slab_base, 0, slab_cap, stream)?;
            self.invalidate_dflash_batched_graphs(stream)?;
        }

        // Stage SSM pointer tables (1 contiguous host buffer -> 1 H2D copy).
        let mut ssm_host_buf: Vec<u8> = if num_ssm_layers > 0 && c >= 4 {
            vec![0u8; ssm_slab_bytes]
        } else {
            Vec::new()
        };
        let mut ssm_ptr_tables: Vec<Option<DevicePtr>> = Vec::with_capacity(num_ssm_layers);
        let mut any_ssm_table_staged = false;
        let stage_ctx = self.batched_verify_ctx_full(None, None, false, None);
        let mut state_refs: Vec<&(dyn LayerState + 'static)> = Vec::with_capacity(c);
        let mut ssm_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            state_refs.clear();
            for seq in seqs.iter() {
                state_refs.push(seq.layer_states[layer_idx].as_ref());
            }
            let dst_slice: &mut [u8] = if !ssm_host_buf.is_empty() {
                let start = ssm_idx * ssm_table_stride;
                &mut ssm_host_buf[start..start + ssm_table_stride]
            } else {
                &mut []
            };
            let staged = layer.stage_dflash_cross_seq_ssm_ptr_table(
                c,
                k,
                slot_ids,
                &state_refs,
                &stage_ctx,
                dst_slice,
            )?;
            if staged {
                any_ssm_table_staged = true;
                ssm_ptr_tables.push(Some(
                    slab_base.offset(ssm_slab_off + ssm_idx * ssm_table_stride),
                ));
            } else {
                ssm_ptr_tables.push(None);
            }
            ssm_idx += 1;
        }

        // Stage all C sequences' attention metadata headers (`positions | slots | seq_lens`)
        // into one contiguous host buffer and collect per-row active block-table slices.
        let mut attn_headers = vec![0u8; c * bt_off];
        let mut attn_metas = Vec::with_capacity(c);
        let depths: Vec<usize> = (0..k).collect();
        for (s, seq) in seqs.iter().enumerate() {
            let seq_meta_base = slab_base.offset(attn_slab_off + s * per_seq_meta_stride);
            let hdr = &mut attn_headers[s * bt_off..(s + 1) * bt_off];

            let positions = seq.rotary_positions.verify_tail(seq.seq_len, &depths)?;
            for (t, &pos) in positions.iter().enumerate() {
                hdr[t * 4..t * 4 + 4].copy_from_slice(&pos.to_le_bytes());
            }
            for t in 0..k {
                let pos = seq.seq_len + t;
                let block_idx = pos / bs;
                let block_offset = pos % bs;
                let physical_block = seq
                    .physical_block_for(block_idx)
                    .unwrap_or(self.dummy_kv_block);
                let slot = (physical_block as i64) * (bs as i64) + (block_offset as i64);
                hdr[slots_off + t * 8..slots_off + t * 8 + 8].copy_from_slice(&slot.to_le_bytes());
                let sl = (seq.seq_len + t + 1) as i32;
                hdr[sl_off + t * 4..sl_off + t * 4 + 4].copy_from_slice(&sl.to_le_bytes());
            }
            attn_metas.push(AttnMetadataDev {
                qwen4_qsa_required: seq.qwen4_qsa_required,
                positions: seq_meta_base,
                positions_h: seq_meta_base,
                positions_w: seq_meta_base,
                slot: seq_meta_base.offset(slots_off),
                seq_len: seq_meta_base.offset(sl_off),
                block_table: seq_meta_base.offset(bt_off),
                max_blocks_per_seq: max_blocks,
                num_seqs: k as u32,
            });
        }

        // Enqueue ALL H2D uploads in ONE batched fast-H2D submission.
        let mut copies: Vec<HostToDeviceCopy<'_>> = Vec::with_capacity(2 + c + c * k);
        if upload_token_ids {
            // SAFETY: `flat_tokens` is a contiguous `&[u32]` slice alive through `copy_h2d_group_fast`.
            let tok_bytes = unsafe {
                std::slice::from_raw_parts(
                    flat_tokens.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(flat_tokens),
                )
            };
            copies.push(HostToDeviceCopy::new(tok_bytes, slab_base));
        }
        if any_ssm_table_staged {
            copies.push(HostToDeviceCopy::new(
                &ssm_host_buf,
                slab_base.offset(ssm_slab_off),
            ));
        }
        for (s, seq) in seqs.iter().enumerate() {
            let seq_meta_base = slab_base.offset(attn_slab_off + s * per_seq_meta_stride);
            copies.push(HostToDeviceCopy::new(
                &attn_headers[s * bt_off..(s + 1) * bt_off],
                seq_meta_base,
            ));
            let bt_len = seq.block_table.len().min(mb);
            if bt_len > 0 {
                // SAFETY: `seq.block_table[..bt_len]` is a valid contiguous `&[u32]`
                // whose little-endian representation matches `i32` block IDs.
                let bt_row_bytes = unsafe {
                    std::slice::from_raw_parts(
                        seq.block_table.as_ptr().cast::<u8>(),
                        bt_len * std::mem::size_of::<u32>(),
                    )
                };
                for row in 0..k {
                    copies.push(HostToDeviceCopy::new(
                        bt_row_bytes,
                        seq_meta_base.offset(bt_off + row * mb * 4),
                    ));
                }
            }
        }
        self.copy_h2d_group_fast(&copies, stream)?;

        Ok(PreparedBatchedMetadata {
            token_ids_dev: slab_base,
            attn_metas,
            ssm_ptr_tables,
            slab_base,
        })
    }

    /// Execute embedding + all 64 transformer layers (shared identically
    /// between eager execution and P4 CUDA graph capture).
    fn execute_dflash_batched_body(
        &self,
        flat_tokens: &[u32],
        slot_ids: &[usize],
        seqs: &mut [&mut SequenceState],
        seq_lens_per_seq: &[Vec<usize>],
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        prepared: &PreparedBatchedMetadata,
        k: usize,
        batched_qkv: bool,
        use_slab_batched_embed: bool,
        serial: crate::model::env_diag::DflashSerialControls,
        graph_capture: bool,
        stream: u64,
    ) -> Result<()> {
        let c = seqs.len();
        let total_rows = c * k;
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // ── 1-launch batched token embedding ──
        if use_slab_batched_embed {
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                prepared.token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                total_rows as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, total_rows, stream)?;
        } else {
            self.embed_batch_and_zero_pad(flat_tokens, total_rows, hidden, stream)?;
        }

        let qkv_base = self.buffers.qkv_output();
        let attn_out_base = self.buffers.attn_output();
        let per_seq_qkv = self.dflash_per_seq_qkv_bytes();
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let attn_out_fits = total_rows * q_dim * bf16 <= self.buffers.sizes().attn_output;
        let mut ssm_layer_idx = 0usize;

        // ── Per-layer forward ──
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            if layer_type == LayerType::LinearAttention {
                let ptr_table_override = prepared
                    .ssm_ptr_tables
                    .get(ssm_layer_idx)
                    .copied()
                    .flatten();
                ssm_layer_idx += 1;

                let ssm_ctx =
                    self.batched_verify_ctx_full(None, None, graph_capture, ptr_table_override);
                let mut state_refs: Vec<&mut (dyn LayerState + 'static)> = seqs
                    .iter_mut()
                    .map(|seq| seq.layer_states[layer_idx].as_mut())
                    .collect();
                let hoisted = layer.decode_ssm_batched_cross_seq(
                    hidden,
                    residual,
                    c,
                    k,
                    slot_ids,
                    &mut state_refs,
                    &ssm_ctx,
                    stream,
                )?;
                if !hoisted {
                    anyhow::ensure!(
                        !graph_capture,
                        "verify_dflash_batched capture requires hoisted SSM at layer {layer_idx}"
                    );
                    let ffn_input = self.dflash_batched_ffn_input(total_rows)?;
                    for (s, seq) in seqs.iter_mut().enumerate() {
                        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(
                            seq.slot_idx,
                        )?;
                        let row_base = s * k;
                        let defer = FfnDefer {
                            dst_base: ffn_input,
                            row_offset: row_base,
                        };
                        let h_s = hidden.offset(row_base * h * fp32);
                        let r_s = residual.offset(row_base * h * fp32);
                        let ctx = self.batched_verify_ctx_full(None, Some(defer), false, None);
                        layer.decode_batched(
                            h_s,
                            r_s,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    }
                    self.run_batched_deferred_ffn(
                        layer.as_ref(),
                        ffn_input,
                        hidden,
                        total_rows,
                        serial.ffn,
                        stream,
                    )?;
                }
            } else {
                // FullAttention layer: hoist QKV across all c*K rows, run
                // per-seq RoPE/KV-write/paged-decode directly into disjoint
                // `attn_output` slices, then hoist gate-mul + o_proj +
                // residual_add_rms_norm + FFN across all c*K rows.
                let layer_qkv_batched = if batched_qkv {
                    let qkv_ctx = self.batched_verify_ctx_full(None, None, graph_capture, None);
                    match layer.decode_multi_seq_qkv_batched(
                        hidden, total_rows, qkv_base, &qkv_ctx, stream,
                    ) {
                        Ok(()) => true,
                        // Batched QKV weights/kernel or BF16 residual unavailable for this layer —
                        // fall back to per-seq QKV, but propagate any real GPU execution error.
                        Err(err) if is_batched_qkv_unavailable(&err) => false,
                        Err(err) => return Err(err),
                    }
                } else {
                    false
                };

                let hoist_attn_tail = layer_qkv_batched
                    && attn_out_fits
                    && !serial.ffn
                    && !serial.layer_norms
                    && !crate::layers::attn_out_serial_enabled();

                if hoist_attn_tail {
                    for s in 0..c {
                        let row_base = s * k;
                        let h_s = hidden.offset(row_base * h * fp32);
                        let r_s = residual.offset(row_base * h * fp32);
                        let qkv_s = qkv_base.offset(row_base * per_seq_qkv);
                        let attn_out_s = attn_out_base.offset(row_base * q_dim * bf16);
                        let core_ctx = self.batched_verify_ctx_full(
                            Some(prepared.attn_metas[s]),
                            None,
                            graph_capture,
                            None,
                        );
                        layer.decode_multi_seq_attn_core_from_qkv(
                            h_s,
                            r_s,
                            k,
                            kv_cache,
                            &seq_lens_per_seq[s],
                            &core_ctx,
                            stream,
                            qkv_s,
                            attn_out_s,
                        )?;
                    }
                    let tail_ctx = self.batched_verify_ctx_full(None, None, graph_capture, None);
                    layer.decode_multi_seq_o_proj_post_norm_ffn_batched(
                        hidden,
                        residual,
                        total_rows,
                        qkv_base,
                        attn_out_base,
                        &tail_ctx,
                        stream,
                    )?;
                } else {
                    anyhow::ensure!(
                        !graph_capture,
                        "verify_dflash_batched capture requires hoisted attention at layer {layer_idx}"
                    );
                    let ffn_input = self.dflash_batched_ffn_input(total_rows)?;
                    for (s, seq) in seqs.iter_mut().enumerate() {
                        let row_base = s * k;
                        let defer = FfnDefer {
                            dst_base: ffn_input,
                            row_offset: row_base,
                        };
                        let h_s = hidden.offset(row_base * h * fp32);
                        let r_s = residual.offset(row_base * h * fp32);
                        let ctx = self.batched_verify_ctx_full(
                            Some(prepared.attn_metas[s]),
                            Some(defer),
                            false,
                            None,
                        );
                        let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|st| st.as_mut()).collect();
                        if layer_qkv_batched {
                            let qkv_s = qkv_base.offset(row_base * per_seq_qkv);
                            layer.decode_multi_seq_attn_from_qkv(
                                h_s,
                                r_s,
                                k,
                                &mut refs,
                                kv_cache,
                                &seq_lens_per_seq[s],
                                &block_tables_vec,
                                &ctx,
                                stream,
                                qkv_s,
                            )?;
                        } else {
                            layer.decode_multi_seq(
                                h_s,
                                r_s,
                                k,
                                &mut refs,
                                kv_cache,
                                &seq_lens_per_seq[s],
                                &block_tables_vec,
                                &ctx,
                                stream,
                            )?;
                        }
                    }
                    self.run_batched_deferred_ffn(
                        layer.as_ref(),
                        ffn_input,
                        hidden,
                        total_rows,
                        serial.ffn,
                        stream,
                    )?;
                }
            }

            if total_rows <= self.dflash_hidden_save_rows {
                self.try_dflash_capture_rows(layer_idx, total_rows, stream)?;
            }
        }

        Ok(())
    }

    fn run_batched_deferred_ffn(
        &self,
        layer: &dyn crate::layer::TransformerLayer,
        ffn_input: DevicePtr,
        hidden: DevicePtr,
        total_rows: usize,
        serial_ffn: bool,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let ffn_ctx = self.batched_verify_ctx_full(None, None, false, None);
        if serial_ffn {
            for row in 0..total_rows {
                layer.run_deferred_ffn(
                    ffn_input.offset(row * h * bf16),
                    hidden.offset(row * h * fp32),
                    1,
                    &ffn_ctx,
                    stream,
                )?;
            }
        } else {
            layer.run_deferred_ffn(ffn_input, hidden, total_rows, &ffn_ctx, stream)?;
        }
        Ok(())
    }

    /// Per-row stride (bytes) of the concatenated `[Q | (Gate) | K | V]` layout
    /// in the shared `qkv_output` buffer.
    fn dflash_per_seq_qkv_bytes(&self) -> usize {
        let bf16 = 2usize;
        let q_heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let q_mul = if self.config.attn_gated { 2 } else { 1 };
        (q_heads * q_mul + 2 * kv_heads) * hd * bf16
    }

    /// Lazily allocate (and cache) the `[c*K, H]` BF16 FFN-input collection
    /// buffer (used only on non-hoisted fallback layers). Sized to
    /// `max_batch_tokens × H × bf16` on first use; rejects
    /// `total_rows > max_batch_tokens` so cached allocations never overflow.
    fn dflash_batched_ffn_input(&self, total_rows: usize) -> Result<DevicePtr> {
        let cap_rows = self.buffers.max_batch_tokens();
        anyhow::ensure!(
            total_rows <= cap_rows,
            "dflash_batched_ffn_input: total_rows={total_rows} exceeds cached capacity {cap_rows}"
        );
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let mut guard = self.dflash_batched_ffn_input.lock();
        if let Some(p) = *guard {
            return Ok(p);
        }
        let ptr = self.gpu.alloc(cap_rows * h * bf16)?;
        *guard = Some(ptr);
        Ok(ptr)
    }

    fn batched_verify_ctx_full(
        &self,
        attn_metadata: Option<AttnMetadataDev>,
        ffn_defer: Option<FfnDefer>,
        graph_capture: bool,
        ssm_multi_seq_ptr_table_override: Option<DevicePtr>,
    ) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata,
            profile: false,
            comm: self.comm_ref(),
            graph_capture,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override,
            self_spec_sparse_draft: None,
            ffn_defer,
        }
    }
}

/// Returns `true` only when `decode_multi_seq_qkv_batched` declined because
/// batched QKV weights/kernels or BF16 residual support are absent for this
/// layer (so per-seq QKV fallback is valid), never for GPU execution failures.
fn is_batched_qkv_unavailable(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("decode_multi_seq_qkv_batched not implemented")
        || msg.contains("decode_multi_seq_qkv_batched: transposed QKV weights")
        || msg.contains("decode_multi_seq_qkv_batched: FP32 residual")
}

/// Thin wrapper over `ensure_blocks_through_decode` for the batched verify.
fn ensure_blocks(
    model: &TransformerModel,
    seq: &mut SequenceState,
    max_block_idx: usize,
    kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
    stream: u64,
) -> Result<()> {
    super::super::block_mgmt::ensure_blocks_through_decode(
        seq,
        max_block_idx,
        kv_cache,
        model.prefix_cache.as_ref(),
        model.gpu.as_ref(),
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_qkv_defaults_on_and_honors_explicit_override() {
        assert!(resolve_dflash_batched_qkv_flag(None));
        assert!(resolve_dflash_batched_qkv_flag(Some("1")));
        assert!(resolve_dflash_batched_qkv_flag(Some("true")));
        assert!(!resolve_dflash_batched_qkv_flag(Some("0")));
        assert!(!resolve_dflash_batched_qkv_flag(Some("false")));
    }

    #[test]
    fn batched_verify_graph_defaults_on_wide_sm_and_off_on_gb10() {
        // GB10 (<= 64 SMs): default OFF, opt-in via "1".
        assert!(!resolve_dflash_batched_graph_flag(None, 48));
        assert!(!resolve_dflash_batched_graph_flag(None, 64));
        assert!(resolve_dflash_batched_graph_flag(Some("1"), 48));
        // SM120 / SM103 (> 64 SMs): default ON, opt-out via "0".
        assert!(resolve_dflash_batched_graph_flag(None, 65));
        assert!(resolve_dflash_batched_graph_flag(None, 188));
        assert!(!resolve_dflash_batched_graph_flag(Some("0"), 188));
        assert!(!resolve_dflash_batched_graph_flag(Some("false"), 188));
    }

    #[test]
    fn graph_meta_tag_is_disjoint_from_decode_and_qwen4_verify_tags() {
        let qwen4_tag = 1usize << (usize::BITS - 1);
        let qwen4_l0_tag = 1usize << (usize::BITS - 2);
        let m1 = pack_dflash_batched_graph_meta(8, 4, &[0, 0, 1, 2]);
        let m2 = pack_dflash_batched_graph_meta(8, 4, &[0, 0, 1, 3]);
        assert!(is_dflash_batched_graph_meta(m1));
        assert_eq!(m1 & qwen4_tag, 0);
        assert_eq!(m1 & qwen4_l0_tag, 0);
        assert_ne!(
            m1, m2,
            "distinct seq_len buckets must yield distinct meta keys"
        );
        assert!(!is_dflash_batched_graph_meta(64));
    }

    #[test]
    fn seq_len_graph_bucket_tracks_512_ceil_tiles_and_1024_kgamma_flip() {
        // Within tile 1 (max_len <= 512): constant bucket.
        assert_eq!(
            dflash_seq_len_graph_bucket(100, 8),
            dflash_seq_len_graph_bucket(504, 8)
        );
        // Crossing 512 -> 513 flips `div_ceil(512)` from 1 to 2.
        assert_ne!(
            dflash_seq_len_graph_bucket(504, 8),
            dflash_seq_len_graph_bucket(505, 8)
        );
        // Crossing 1023 -> 1024 flips `kgamma_ctx_ok` (`max_len >= 1024`) even
        // though `div_ceil(512)` is still 2.
        assert_ne!(
            dflash_seq_len_graph_bucket(1015, 8),
            dflash_seq_len_graph_bucket(1016, 8)
        );
        // Crossing 1024 -> 1025 flips `div_ceil(512)` from 2 to 3.
        assert_ne!(
            dflash_seq_len_graph_bucket(1016, 8),
            dflash_seq_len_graph_bucket(1017, 8)
        );
        // Straddle detection for 512-ceil boundaries.
        assert!(!window_straddles_512_split_tile(504, 8)); // [505..=512]
        assert!(window_straddles_512_split_tile(505, 8)); // [506..=513]
        assert!(!window_straddles_512_split_tile(512, 8)); // [513..=520]
    }
}
