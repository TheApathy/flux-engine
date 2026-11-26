// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! `TransformerModel::decode_batch_dispatch` — multi-sequence batched decode.
//!
//! Special-cases `n == 1`, native Qwen4 batching, piecewise multi-seq graphs,
//! and EP/serial fallback; otherwise pads to the nearest captured graph size,
//! builds a `ForwardContext`, dispatches `decode_multi_seq` for each layer,
//! and runs final norm + batched LM-head projection.

use std::sync::OnceLock;

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LmHeadTcOverride {
    Disabled,
    Enabled,
    Auto,
}

fn parse_lm_head_tc_override(raw: Option<&str>) -> LmHeadTcOverride {
    match raw.map(str::trim) {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
            LmHeadTcOverride::Disabled
        }
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => {
            LmHeadTcOverride::Enabled
        }
        _ => LmHeadTcOverride::Auto,
    }
}

fn raw_lm_head_tc_override() -> LmHeadTcOverride {
    static RAW: OnceLock<LmHeadTcOverride> = OnceLock::new();
    *RAW.get_or_init(|| {
        parse_lm_head_tc_override(std::env::var("ATLAS_LM_HEAD_TC").ok().as_deref())
    })
}

const fn resolve_lm_head_tc_for_sm(
    env_override: LmHeadTcOverride,
    sm_count: u32,
    rows: usize,
) -> bool {
    match env_override {
        LmHeadTcOverride::Disabled => false,
        LmHeadTcOverride::Enabled => true,
        LmHeadTcOverride::Auto => sm_count > 64 && rows >= 8,
    }
}

/// Partition `padded_n` LM-head rows into `<= 32`-row exact GEMV slices
/// (splitting a 33-row remainder into `17 + 16` so no singleton tail hits
/// serial GEMV).
fn plan_lm_head_exact_chunks(padded_n: usize) -> Vec<(usize, usize)> {
    if padded_n < 2 {
        return Vec::new();
    }
    let max_chunk = crate::layers::TC_VERIFY_MAX_ROWS;
    if padded_n <= max_chunk {
        return vec![(0, padded_n)];
    }
    let mut chunks = Vec::with_capacity(padded_n.div_ceil(max_chunk) + 1);
    let mut offset = 0;
    while offset < padded_n {
        let rem = padded_n - offset;
        let len = if rem <= max_chunk {
            rem
        } else if rem == max_chunk + 1 {
            max_chunk / 2 + 1
        } else {
            max_chunk
        };
        chunks.push((offset, len));
        offset += len;
    }
    chunks
}

/// RAII guard that returns taken per-sequence `layer_states` on every exit path
/// (`Ok`, `Err`, or panic unwind), dropping any appended dummy pad states.
struct DecodeBatchLayerStatesGuard<'a, 'b> {
    seqs: &'a mut [&'b mut SequenceState],
    states: Vec<Vec<Box<dyn LayerState>>>,
    n: usize,
}

impl<'a, 'b> DecodeBatchLayerStatesGuard<'a, 'b> {
    fn take(seqs: &'a mut [&'b mut SequenceState]) -> Self {
        let n = seqs.len();
        let states = seqs
            .iter_mut()
            .map(|s| std::mem::take(&mut s.layer_states))
            .collect();
        Self { seqs, states, n }
    }
}

impl Drop for DecodeBatchLayerStatesGuard<'_, '_> {
    fn drop(&mut self) {
        let restore_n = self.n.min(self.states.len());
        for (seq, ls) in self.seqs.iter_mut().zip(self.states.drain(..restore_n)) {
            seq.layer_states = ls;
        }
    }
}

impl TransformerModel {
    /// Embed `tokens` into `hidden[0..tokens.len())` with a single batched
    /// gather launch and asynchronously zero-pad `hidden[tokens.len()..padded_n)`.
    pub(super) fn embed_batch_and_zero_pad(
        &self,
        tokens: &[u32],
        padded_n: usize,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let n = tokens.len();
        let h = self.config.hidden_size;
        let elem_bytes = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        if n > 0 && self.batched_embed_kernel.0 != 0 && !self.config.is_qwen4_exp() {
            // SAFETY: `tokens` is a valid contiguous `&[u32]` slice of length `n`
            // (`size_of_val(tokens)` bytes, POD); `u8` has alignment 1 and
            // `copy_h2d_group_fast` finishes reading `token_ids_bytes` before returning.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    tokens.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(tokens),
                )
            };
            // `norm_output` is overwritten by layer 0's input RMS norm and is
            // sized `max_batch_tokens * h * 2 >> n * 4`.
            let token_ids_dev = self.buffers.norm_output();
            self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                stream,
            )?;
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                n as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, n, stream)?;
        } else {
            for (i, &tok) in tokens.iter().enumerate() {
                self.embed(tok, hidden.offset(i * h * elem_bytes), stream)?;
            }
        }
        if padded_n > n {
            self.gpu.memset_async(
                hidden.offset(n * h * elem_bytes),
                0,
                (padded_n - n) * h * elem_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Batched LM head for multi-sequence decode (`normed[padded_n, H] -> logits[padded_n, V]`).
    ///
    /// On SM120/SM103 (`sm_count > 64`) at `padded_n >= 8` (or explicit
    /// `ATLAS_LM_HEAD_TC=1`), prefers native block-scaled cuBLASLt NVFP4 or
    /// transposed TC `w4a16_gemm_t_m32_n64`. On GB10 (`sm_count <= 64`) when
    /// unset, preserves the bit-exact `w4a16_gemv_batch_logits_exact` path
    /// (chunked to `<= 32` rows when `padded_n > 32`) with serial fallback.
    pub(super) fn emit_batched_lm_head(
        &self,
        normed: DevicePtr,
        logits: DevicePtr,
        padded_n: usize,
        graph_capture: bool,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let sm_count = atlas_core::device::active_sm_count();
        let tc_override = raw_lm_head_tc_override();
        let tc_enabled = resolve_lm_head_tc_for_sm(tc_override, sm_count, padded_n);

        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            if tc_enabled {
                let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
                if padded_n <= sm120_max_m
                    && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                        self.gpu.as_ref(),
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        sm120_max_m,
                        stream,
                    )?
                {
                    return Ok(());
                }
                let has_tc_t =
                    self.lm_head_nvfp4_t.is_some() && self.w4a16_gemm_t_m32_n64_kernel.0 != 0;
                // Prefer cuBLASLt NVFP4 at `padded_n >= 24` or whenever the
                // optional transposed weight copy (`ATLAS_LM_HEAD_T=1`) was not
                // built (so SM120 gets TC LM head out of the box with 0 extra VRAM).
                if (padded_n >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M as usize
                    || !has_tc_t)
                    && ops::try_cublaslt_nvfp4_multiseq_single(
                        self.gpu.as_ref(),
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        v as u32,
                        graph_capture,
                        "lm_head_multi_seq",
                        stream,
                    )?
                {
                    return Ok(());
                }
                if let Some(ref nvfp4_t) = self.lm_head_nvfp4_t
                    && self.w4a16_gemm_t_m32_n64_kernel.0 != 0
                {
                    let ldb = (v.div_ceil(64) * 64) as u32;
                    return ops::w4a16_gemm_n64_m32_ldb(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_t_m32_n64_kernel,
                        normed,
                        nvfp4_t,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        ldb,
                        stream,
                    );
                }
            }

            let chunks = plan_lm_head_exact_chunks(padded_n);
            let all_chunks_exact = !chunks.is_empty()
                && chunks.iter().all(|&(_, len)| {
                    matches!(
                        self.w4a16_exact_lm_head_kernels.route_for_rows(len as u32),
                        Some(ops::ExactLmHeadRoute::Exact(_))
                    )
                });
            if all_chunks_exact {
                for (start, len) in chunks {
                    ops::w4a16_gemv_batch_logits_exact(
                        self.gpu.as_ref(),
                        self.w4a16_exact_lm_head_kernels,
                        normed.offset(start * h * bf16),
                        nvfp4,
                        logits.offset(start * v * bf16),
                        len as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
                return Ok(());
            }

            for i in 0..padded_n {
                let normed_i = normed.offset(i * h * bf16);
                let logits_i = logits.offset(i * v * bf16);
                ops::w4a16_decode_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    self.w4a16_gemv_sw_kernel,
                    self.gemv_sw,
                    normed_i,
                    nvfp4,
                    logits_i,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if sm_count > 64 && padded_n >= 4 {
            if graph_capture {
                return spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    normed.0,
                    self.lm_head_weight.weight.0,
                    logits.0,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                );
            }
            return ops::cublas_bf16_proj_dense(
                normed,
                &self.lm_head_weight,
                logits,
                padded_n as u32,
                v as u32,
                h as u32,
                stream,
            );
        }

        for i in 0..padded_n {
            let normed_i = normed.offset(i * h * bf16);
            let logits_i = logits.offset(i * v * bf16);
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                normed_i,
                &self.lm_head_weight,
                logits_i,
                v as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }
    pub(super) fn decode_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = tokens.len();
        assert_eq!(n, seqs.len(), "tokens.len() must equal seqs.len()");
        anyhow::ensure!(
            n <= 1 || seqs.iter().all(|seq| seq.rotary_positions.is_identity()),
            "image sequence decode currently requires C1"
        );

        // The complete native Qwen4 lane is the default eager `C2..=C256`
        // (`QWEN4_NATIVE_MAX_ROWS`) target-only route (`crate::qwen4_batch_policy`).
        // Select it before the generic FP16-state conversion because its
        // recurrent kernels deliberately retain FP32 request-owned state. Every
        // other C2+ frame takes the serialized path below with a typed
        // `SerialReason` (QSA-required rows, profiling,
        // `ATLAS_QWEN4_NATIVE_BATCH=0`, ...); the decision is made before any
        // mutation, so it is never a retry of failed native work. With native
        // batching enabled, widths above `QWEN4_NATIVE_MAX_ROWS` fail closed.
        let qwen4_route = if n > 1 && self.config.is_qwen4_exp() {
            Some(self.qwen4_frame_route(seqs)?)
        } else {
            None
        };
        if matches!(
            qwen4_route,
            Some(crate::qwen4_batch_policy::FrameRoute::Native)
        ) {
            debug_assert!(
                super::decode_qwen4_native_batch::qwen4_native_batch_request_eligible(seqs)
            );
            return self.decode_qwen4_native_batch_dispatch(tokens, seqs);
        }

        // ATLAS_SSM_H_FP16 stage 2: every sequence in the batch, before the
        // n==1 delegation below and before any graph work.
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }

        // Single-sequence: delegate to decode() which uses CUDA graphs.
        // decode_batch disables graphs for n≥2 (SSM state pointer staleness),
        // but n=1 is safe and benefits from graph replay (2x throughput).
        if n == 1 {
            self.reconcile_dflash_context_before_bootstrap_dispatch(seqs[0], stream)?;
            self.decode(tokens[0], seqs[0], stream)?;
            self.capture_dflash_serial_decode_row(seqs[0], stream)?;
            return Ok(self.decode_logits_ptr());
        }

        // MLA models: the batched `decode_multi_seq` path has a genuine MLA
        // branch (`ms_mla_decode` in `qwen3_attention/trait_impl/multi_seq/mla.rs`)
        // — the batched analogue of `attention_forward_mla`. It reads `self.mla`'s
        // projections (not the NULL `attn.q_proj` stub the Mistral loader
        // installs) and isolates each sequence's compressed latent-KV via
        // per-sequence metadata. Concurrent MLA decode therefore takes the
        // normal batched path below — no host round-trip, no cross-seq
        // contamination.
        //
        // The legacy per-sequence `decode()` fallback (host-staged logits +
        // CUDA-graph suppression) is retained ONLY behind the
        // `ATLAS_MLA_PERSEQ_FALLBACK` escape hatch, as a guarded safety net
        // should a regression surface in the batched MLA path. It does NOT
        // fully isolate concurrent sequences (each `decode()`'s
        // `Buffers::zero_all` wipes the shared `logits` buffer), so it is
        // not the default.
        let mla_perseq_fallback =
            self.is_mla_dispatch() && crate::model::env_diag::mla_perseq_fallback_enabled();
        // Qwen4/mHC C1 owns PLE injection, both hyperconnection boundaries,
        // per-token QSA ingest/select, and the final highway collapse.  The
        // generic multi-seq path implements none of those transitions yet.
        // Distributed (TP/EP/PP) mode likewise executes per-row lockstep decode
        // with worker ranks and stages each sequence's logits row into `[n, vocab]`.
        if self.comm.is_some()
            || mla_perseq_fallback
            || self.config.is_qwen4_exp()
            || self.config.hc_mult > 0
        {
            if let Some(crate::qwen4_batch_policy::FrameRoute::Serial { frame, .. }) = &qwen4_route
                && super::decode_qwen4_native_batch::qwen4_native_batch_receipt_enabled()?
            {
                let qwen4_native_selected =
                    !matches!(frame, crate::qwen4_batch_policy::SerialReason::SelectorOff);
                let sessions = seqs
                    .iter()
                    .map(|seq| format!("{:016x}", seq.session_hash))
                    .collect::<Vec<_>>()
                    .join(",");
                let slots = seqs
                    .iter()
                    .map(|seq| seq.slot_idx.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let pre_step_seq_lens = seqs
                    .iter()
                    .map(|seq| seq.seq_len.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let qsa_required = seqs.iter().any(|seq| seq.qwen4_qsa_required);
                tracing::info!(
                    "QWEN4_SERIAL_BATCH_FRAME rows={n} sessions={sessions} slots={slots} pre_step_seq_lens={pre_step_seq_lens} selector_on={qwen4_native_selected} qsa_required={qsa_required} graph=false"
                );
                // Separate line: the frame receipt grammar above is pinned by
                // the qualification probe and must stay byte-compatible.
                tracing::info!(
                    "QWEN4_SERIAL_BATCH_REASON rows={n} sessions={sessions} reason={}",
                    frame.label()
                );
            }
            return self.decode_batch_serial_rows(tokens, seqs);
        }

        // ── Piecewise CUDA-graph multi-seq decode (ATLAS_MULTISEQ_GRAPHS=1) ──
        //
        // Captures address-stable SSM/FFN layer runs + the norm/LM-head tail
        // into per-`padded_n` segment graphs and runs FullAttention layers
        // eagerly between replays. Requires the multi-seq SSM kernel path
        // (so per-seq state pointers are indirected via the layer-stable
        // ptr scratch rather than baked) and the non-EP / no-comm path
        // (graph capture is illegal under NCCL all-reduce). See
        // `crate::layers::multiseq_graphs_enabled` for the full audit of
        // which addresses are captured vs run eager.
        if crate::layers::multiseq_graphs_enabled()
            && crate::layers::ssm_multi_seq_kernel_enabled()
            && self.comm.is_none()
        {
            return self.decode_batch_dispatch_piecewise(tokens, seqs);
        }

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad to the shared C128 decode/capture ladder.
        let padded_n = crate::traits::padded_batch_n(n);

        // ── Phase 1: Pre-graph (runs every step, NOT captured) ──

        // 1a-1b. Single-launch batched embed for active tokens + async zero-pad
        // for hidden[n..padded_n).
        self.embed_batch_and_zero_pad(tokens, padded_n, hidden, stream)?;

        // 1c. Allocate KV blocks for active sequences
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for seq in seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1d. Upload metadata with fixed stride (active + padding)
        let metadata = self.upload_batch_metadata_fixed(seqs, padded_n, &mut kv_cache, stream)?;

        // CONC_HSD: per-seq hidden-state dump diagnostic (eager-only because
        // `synchronize` + `copy_d2h` are illegal inside CUDA graph capture).
        let conc_hsd =
            crate::model::env_diag::conc_hsd_enabled() && padded_n >= 2 && self.comm.is_none();

        // CUDA graphs DISABLED by default for multi-sequence decode: SSM state
        // pointers (h_state, conv_state) are baked into per-seq kernel args of
        // gdn_decode/conv1d_update at capture time. When batch composition
        // changes (sequences finish, swap_remove reorders), the graph
        // replays with stale pointers and corrupts SSM state across seqs.
        // Attention metadata uses fixed device addresses and is safe.
        // n==1 still uses self.decode()'s correct graph cache.
        //
        // Wire ATLAS_SSM_MULTI_SEQ_GRAPH=1 as an OPT-IN UNSAFE override (forced
        // off while `conc_hsd` is active so diagnostic host syncs stay legal).
        let use_graphs = crate::layers::ssm_multi_seq_graph_enabled() && !conc_hsd;
        if use_graphs {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "ATLAS_SSM_MULTI_SEQ_GRAPH=1 is set — multi-seq decode \
                     will use CUDA graph capture. EXPERIMENTAL: output will \
                     corrupt when batch composition changes (sequences \
                     finish, swap_remove reorders). Indirection-table fix \
                     is the proper unblocker; this is opt-in until then."
                );
            }
        }

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // ── Phase 2: CUDA graph lookup / capture ──
        let mut graphs = if use_graphs {
            Some(self.batch_decode_graphs.lock())
        } else {
            None
        };

        // `batch_decode_graphs` is keyed by `(sorted_slot_ids, padded_n)` (see
        // `TransformerModel::batch_decode_graphs` in `model/types.rs` and
        // invalidation in `trait_impl/sequence.rs`). Fires when `use_graphs` is
        // true (`ATLAS_SSM_MULTI_SEQ_GRAPH=1` without `conc_hsd`).
        //
        // NOTE: even with stable SSM pointer-table indirection, slot-specific
        // addresses can still leak into captured graphs via non-indirected
        // paths, so `sorted_slot_ids` remains part of the cache key.
        let graph_key: (Vec<usize>, usize) = {
            let mut slots: Vec<usize> = seqs.iter().map(|s| s.slot_idx).collect();
            slots.sort_unstable();
            (slots, padded_n)
        };

        if let Some(ref graphs) = graphs
            && let Some(&graph) = graphs.get(&graph_key)
        {
            // Refresh SSM pointer tables outside replay so the captured graph
            // (which omits H2D pointer-table nodes) always reads current slots.
            if crate::layers::ssm_multi_seq_kernel_enabled() {
                let mut layer_states_guard = DecodeBatchLayerStatesGuard::take(seqs);
                self.append_dummy_pad_states(&mut layer_states_guard.states, n, padded_n)?;
                for layer_idx in 0..self.layers.len() {
                    if self.config.layer_type(layer_idx) == LayerType::LinearAttention
                        && self.layers[layer_idx].multiseq_graph_safe(padded_n)
                    {
                        let mut refs =
                            extract_layer_refs(&mut layer_states_guard.states, layer_idx);
                        self.layers[layer_idx].multiseq_refresh_ptr_table(
                            &mut refs,
                            padded_n,
                            self.gpu.as_ref(),
                            stream,
                        )?;
                    }
                }
            }
            // Graph exists — replay (kernels use updated metadata + SSM pool addresses)
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }

            // ── Phase 3: Post-graph (update sequence state) ──
            for (i, seq) in seqs.iter_mut().enumerate() {
                seq.tokens.push(tokens[i]);
                seq.seq_len += 1;
            }
            // `emit_batched_lm_head` writes batched BF16 logits into `self.buffers.logits()`.
            return Ok(self.buffers.logits());
        }
        {
            // First time for this padded_n — capture a new graph (or run eagerly for EP).
            // Build layer states for all padded_n sequences (real + dummy padding).
            let seq_lens: Vec<usize> = (0..padded_n)
                .map(|i| if i < n { seqs[i].seq_len } else { 0 })
                .collect();
            let block_tables: Vec<Vec<u32>> = (0..padded_n)
                .map(|i| {
                    if i < n {
                        seqs[i].block_table.clone()
                    } else {
                        vec![self.dummy_kv_block]
                    }
                })
                .collect();

            // Extract real per-seq layer states behind a drop guard so any
            // error or unwind restores `seq.layer_states`; append dummy states
            // for padding positions using `dummy_slot()` so pad SSM kernel
            // writes can never collide with another claimed sequence's pool.
            let mut layer_states_guard = DecodeBatchLayerStatesGuard::take(seqs);
            self.append_dummy_pad_states(&mut layer_states_guard.states, n, padded_n)?;

            // When capturing, refresh SSM pointer tables BEFORE capture begins
            // (since `decode_multi_seq_inner` skips H2D uploads during capture).
            if use_graphs && crate::layers::ssm_multi_seq_kernel_enabled() {
                for layer_idx in 0..self.layers.len() {
                    if self.config.layer_type(layer_idx) == LayerType::LinearAttention
                        && self.layers[layer_idx].multiseq_graph_safe(padded_n)
                    {
                        let mut refs =
                            extract_layer_refs(&mut layer_states_guard.states, layer_idx);
                        self.layers[layer_idx].multiseq_refresh_ptr_table(
                            &mut refs,
                            padded_n,
                            self.gpu.as_ref(),
                            stream,
                        )?;
                    }
                }
            }

            // `CaptureGuard` closes the capture on any early `?` or unwind via
            // `Drop` instead of leaving the stream wedged (`trait_impl::capture_guard`).
            let mut capture = if use_graphs {
                Some(super::capture_guard::CaptureGuard::begin(
                    self.gpu.as_ref(),
                    stream,
                    "decode_a2_batched",
                )?)
            } else {
                None
            };

            let dump_hidden = |label: &str, stream: u64| -> Result<()> {
                if !conc_hsd {
                    return Ok(());
                }
                self.gpu.synchronize(stream)?;
                let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(padded_n);
                for i in 0..padded_n {
                    let mut buf = vec![0u8; 4 * fp32];
                    self.gpu
                        .copy_d2h_on_stream(hidden.offset(i * h * fp32), &mut buf, stream)?;
                    let vals: Vec<f32> = if fp32 == 4 {
                        buf.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    } else {
                        buf.chunks_exact(2)
                            .map(|c| {
                                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                            })
                            .collect()
                    };
                    bufs.push(vals);
                }
                let pretty: Vec<String> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("s{i}=[{:.4},{:.4},{:.4},{:.4}]", v[0], v[1], v[2], v[3]))
                    .collect();
                tracing::info!("CONC_HSD {label}: {}", pretty.join(" "));
                Ok(())
            };

            dump_hidden("post_embed", stream)?;

            // Layer loop for padded_n sequences
            self.pp_stage_enter(hidden, padded_n, stream)?;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let mut layer_state_refs =
                    extract_layer_refs(&mut layer_states_guard.states, layer_idx);
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &ctx,
                    stream,
                )?;
                if conc_hsd {
                    dump_hidden(&format!("after_L{:02}", layer_idx), stream)?;
                }
            }
            self.pp_stage_exit(hidden, padded_n, stream)?;

            // Final norm [padded_n, H] + batched LM head [padded_n, V]
            let normed = self.buffers.norm_output();
            self.final_norm_rows(hidden, normed, padded_n as u32, stream)?;
            let logits = self.buffers.logits();
            self.emit_batched_lm_head(normed, logits, padded_n, use_graphs, stream)?;

            if use_graphs {
                // `capture` is `Some` here: set exactly when `use_graphs` was,
                // untouched since.
                let graph = capture
                    .take()
                    .expect("use_graphs implies begin_capture ran above")
                    .end()?;
                if graph.0 != 0 {
                    tracing::info!("Captured CUDA graph for batch size {padded_n}");
                    if let Some(ref mut g) = graphs {
                        g.insert(graph_key.clone(), graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (update sequence state) ──
        for (i, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[i]);
            seq.seq_len += 1;
        }

        // `emit_batched_lm_head` writes batched BF16 logits into `self.buffers.logits()`.
        Ok(self.buffers.logits())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LmHeadTcOverride, parse_lm_head_tc_override, plan_lm_head_exact_chunks,
        resolve_lm_head_tc_for_sm,
    };

    #[test]
    fn lm_head_tc_policy_preserves_gb10_and_enables_sm120() {
        assert!(!resolve_lm_head_tc_for_sm(LmHeadTcOverride::Auto, 48, 32));
        assert!(!resolve_lm_head_tc_for_sm(LmHeadTcOverride::Auto, 64, 64));
        assert!(!resolve_lm_head_tc_for_sm(LmHeadTcOverride::Auto, 188, 4));
        assert!(resolve_lm_head_tc_for_sm(LmHeadTcOverride::Auto, 188, 8));
        assert!(resolve_lm_head_tc_for_sm(LmHeadTcOverride::Auto, 188, 64));
        assert!(!resolve_lm_head_tc_for_sm(
            LmHeadTcOverride::Disabled,
            188,
            64
        ));
        assert!(resolve_lm_head_tc_for_sm(LmHeadTcOverride::Enabled, 48, 2));
        assert_eq!(
            parse_lm_head_tc_override(Some("0")),
            LmHeadTcOverride::Disabled
        );
        assert_eq!(
            parse_lm_head_tc_override(Some("1")),
            LmHeadTcOverride::Enabled
        );
        assert_eq!(parse_lm_head_tc_override(None), LmHeadTcOverride::Auto);
    }

    #[test]
    fn lm_head_exact_chunk_planner_handles_c2_to_c128_without_singletons() {
        assert!(plan_lm_head_exact_chunks(1).is_empty());
        assert_eq!(plan_lm_head_exact_chunks(32), vec![(0, 32)]);
        assert_eq!(plan_lm_head_exact_chunks(33), vec![(0, 17), (17, 16)]);
        assert_eq!(plan_lm_head_exact_chunks(64), vec![(0, 32), (32, 32)]);
        for rows in 2usize..=128 {
            let chunks = plan_lm_head_exact_chunks(rows);
            let mut cursor = 0;
            for (start, len) in chunks {
                assert_eq!(start, cursor);
                assert!((2..=32).contains(&len));
                cursor += len;
            }
            assert_eq!(cursor, rows);
        }
    }
}
