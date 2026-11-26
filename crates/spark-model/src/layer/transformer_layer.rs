// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` trait — composable per-layer forward/decode hooks.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState, Qwen4CrossSeqFrame,
    Qwen4CrossSeqRows, Qwen4CrossSeqStaging,
};

/// One shared Qwen4 routed-expert transpose arena. Projection order is
/// gate/up/down; pointer-table arrays address the corresponding arena slices.
#[derive(Debug, Clone, Copy)]
pub struct MoeStreamTransposeScratch {
    pub(crate) packed: [DevicePtr; 3],
    pub(crate) scale: [DevicePtr; 3],
    pub(crate) packed_tables: [DevicePtr; 3],
    pub(crate) scale_tables: [DevicePtr; 3],
}

/// A single transformer layer performing the full per-layer computation.
///
/// Each layer encapsulates:
/// 1. Pre-norm -> attention/SSM -> residual add
/// 2. Post-norm -> FFN/MoE -> residual add
///
/// The generic model loop iterates `layers` without knowing whether
/// each is attention, SSM, MoE, or dense FFN.
pub trait TransformerLayer: Send + Sync {
    /// `&dyn Any` downcast hook for read-only layer walks (e.g. the DSpark
    /// γ-verify compressed-pool catch-up). Default `None`; layers that expose
    /// per-layer state override.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Qwen4 attention-side hyperconnection mixer (`layers[i].attn_hyper_connection`),
    /// used by `DflashCaptureMode::Qwen4LayerAttnHc` to contract layer `t`'s 4H
    /// residual through layer `t + 1`'s attention hyperconnection mixer.
    fn qwen4_attn_hyper(&self) -> Option<&crate::layers::qwen4_hyper::Qwen4HyperConnection> {
        None
    }

    /// Expert-major prefill (`ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR`): run this layer's MoE
    /// sublayer (MLP hyperconnection prepare, routed+shared experts, inject) over `rows` rows
    /// whose mixer already ran slice by slice. Default: unsupported.
    fn qwen4_prefill_finish_moe(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _rows: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("this layer does not support expert-major Qwen4 prefill")
    }

    /// Preflight the optional exact Qwen4 M16 attention path before request state changes.
    fn preflight_qwen4_attn16(
        &self,
        _kv_cache: &PagedKvCache,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Prepare the undo journal before row-serial QSA verification writes.
    /// Layers without a QSA indexer have no transaction to prepare.
    fn prepare_qwen4_qsa_verify(
        &self,
        _kv_cache: &PagedKvCache,
        _gpu: &dyn GpuBackend,
    ) -> Result<()> {
        Ok(())
    }

    /// `SeqStateBlob`: the QSA side-cache ranges of one physical KV block,
    /// `[raw ring rows, compressed keys]`. `None` for layers without QSA (or a
    /// QSA cache nothing has initialized while `create` is false).
    fn qwen4_qsa_seqstate_block_regions(
        &self,
        _gpu: &dyn GpuBackend,
        _kv_cache: &PagedKvCache,
        _block: u32,
        _create: bool,
    ) -> Result<Option<[(DevicePtr, usize); 2]>> {
        Ok(None)
    }

    /// Persistent mutable cache/journal/retain spans for one physical K5 lane.
    /// Tuples contain actual allocation base/bytes, then mutable span base/bytes.
    /// Shared execution scratch, KV, recurrent state and PLE are bound separately.
    fn qwen4_batch_verify_lane_ranges(
        &self,
        _slot_idx: usize,
        _gpu: &dyn GpuBackend,
        _kv_cache: &PagedKvCache,
        _config: &ModelConfig,
        _writable_blocks: &[u32],
    ) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
        anyhow::bail!("this layer cannot export actual persistent Qwen4 batch verify spans")
    }

    /// Save one physical QSA destination before the corresponding row writes.
    fn snapshot_qwen4_qsa_verify_row(
        &self,
        _kv_cache: &PagedKvCache,
        _slot: DevicePtr,
        _row: usize,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Finalize the per-layer QSA transaction for a flat physical K5 verify.
    /// Attention layers restore rejected cache rows; all other layer types are
    /// intentionally no-ops. The caller keeps this on the verify stream so the
    /// next proposal/forward observes the committed prefix only.
    fn commit_qwen4_qsa_k5(
        &self,
        _accepted: usize,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Commit the QSA journal of a `rows`-row verify frame (2..=5).
    fn commit_qwen4_qsa_rows(
        &self,
        accepted: usize,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if rows == 5 {
            self.commit_qwen4_qsa_k5(accepted, gpu, stream)
        } else {
            Ok(())
        }
    }

    /// Whether this layer owns a physical-block-indexed QSA side cache (`raw_ring` + `compressed_keys`).
    fn has_qsa_side_cache(&self) -> bool {
        false
    }

    /// Read one physical KV block's QSA side-cache slices (`raw_ring`, `compressed_keys`) for sequence swap-out.
    fn read_qsa_swap_block(
        &self,
        _block_idx: u32,
        _kv_cache: &PagedKvCache,
        _gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        anyhow::bail!("layer has no QSA side cache to read")
    }

    /// Write one physical KV block's QSA side-cache slices (`raw_ring`, `compressed_keys`) on sequence swap-in.
    fn write_qsa_swap_block(
        &self,
        _block_idx: u32,
        _raw_data: &[u8],
        _comp_data: &[u8],
        _kv_cache: &PagedKvCache,
        _gpu: &dyn GpuBackend,
    ) -> Result<()> {
        anyhow::bail!("layer has no QSA side cache to write")
    }

    /// Install Qwen4-Exp gated-residual mixers after the ordinary core layer
    /// has been assembled. Other architectures fail closed by default.
    fn set_qwen4_hyperconnections(
        &mut self,
        _attn: crate::layers::Qwen4HyperConnection,
        _mlp: crate::layers::Qwen4HyperConnection,
    ) -> Result<()> {
        anyhow::bail!("layer does not support Qwen4 hyperconnections")
    }

    /// Decode one token through this layer, modifying `hidden` in-place.
    ///
    /// # Arguments
    /// * `hidden` - [1, hidden_size] BF16, read and written
    /// * `residual` - [1, hidden_size] BF16, scratch for residual stream
    /// * `state` - Per-layer state (empty for attention, SSM state for recurrent)
    /// * `kv_cache` - Paged KV cache (may be mutated for block allocation)
    /// * `seq_len` - Current sequence length (for position encoding + cache)
    /// * `block_table` - Sequence's block table (may grow if new blocks needed)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        // `--high-speed-swap` disk-side IDs parallel to `block_table`.
        // Layer-agnostic: the same ID indexes a slot in every
        // layer's on-disk file. Empty when the feature is disabled.
        disk_block_ids: &mut Vec<u32>,
        // Per-layer offload progress. Layer L
        // reads/writes `disk_last_offloaded_per_layer[L]`. Each layer's
        // offload runs independently because each layer writes its own
        // K/V to a separate region of the on-disk file. Empty when HSS
        // is disabled; SSM/MoE layers ignore it.
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// Decode a speculative row without QSA side-stream writes escaping a
    /// failed transaction. Layers without a QSA side stream use ordinary decode.
    fn decode_qwen4_verified_row(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// Single-token KV-only decode for MTP replay rows whose hidden output is
    /// discarded (`!emit_draft`): updates the attention KV cache for `seq_len`
    /// without running Q projection, attention over KV, O projection, or FFN/MoE.
    /// Default falls back to `decode`.
    #[allow(clippy::too_many_arguments)]
    fn decode_kv_only(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// Qwen4 speculative verify: run `num_tokens` four-stream rows through
    /// this layer while batching the MoE sublayer. Architectures without
    /// Qwen4 hyperconnections fail closed.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_batched(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _h_intermediate: DevicePtr,
        _conv_intermediate: DevicePtr,
        _h_intermediate_stride: usize,
        _conv_intermediate_stride: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support batched Qwen4 verify")
    }

    /// Release device resources a layer lazily created for the
    /// cross-sequence Qwen4 verify (pointer tables, events). Called from the
    /// model's drop, after its streams are drained. Default: nothing owned.
    fn release_qwen4_cross_seq_resources(&self, _gpu: &dyn GpuBackend) -> Result<()> {
        Ok(())
    }

    /// Whether lazily created cross-sequence verify resources can serve
    /// another frame (false once a pointer table is poisoned). Read by the
    /// model's read-only admission. Default: nothing to refuse.
    fn qwen4_cross_seq_admits_frames(&self) -> bool {
        true
    }

    /// Read-only admission for one layer of a cross-sequence Qwen4 K-row
    /// verify frame (`ATLAS_QWEN4_BATCHED_K5`). Runs for every layer before
    /// any embedding, KV allocation or recurrent-state mutation, and must not
    /// touch device memory. Fail-closed by default.
    fn preflight_qwen4_cross_seq_verify(
        &self,
        _frame: Qwen4CrossSeqFrame,
        _qsa_required: &[bool],
        _ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::bail!("layer does not support cross-sequence Qwen4 verify")
    }

    /// Cross-sequence Qwen4 K-row verify for one complete layer (mixer and
    /// MLP halves). `seqs[s]` owns rows `frame.row_base(s)..frame.row_base(s) + frame.seq_k(s)`
    /// of `hidden`/`residual`. Row-parallel work runs once over every row on the
    /// exact multi-row tiers; recurrent/attention cores run per sequence with
    /// exactly the single-sequence K-row verifier's kernels and buffer
    /// footprint, inside that sequence's verify lane.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_cross_seq_verify(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _frame: Qwen4CrossSeqFrame,
        _seqs: &mut [Qwen4CrossSeqRows<'_>],
        _kv_cache: &mut PagedKvCache,
        _staging: Qwen4CrossSeqStaging,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support cross-sequence Qwen4 verify")
    }

    /// Read-only admission for the mixer-only half of a cross-sequence Qwen4
    /// K-row verify frame (omits `qwen4_cross_seq_mlp` preflight for callers
    /// that fuse MLP/MoE across mixed prefill+decode rows). Fail-closed by default.
    fn preflight_qwen4_cross_seq_mixer_only(
        &self,
        _frame: Qwen4CrossSeqFrame,
        _qsa_required: &[bool],
        _ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::bail!("layer does not support cross-sequence Qwen4 mixer-only verify")
    }

    /// Cross-sequence Qwen4 K-row verify mixer half only: stops immediately
    /// after `attn_hyper.inject_saved_batched` so a mixed prefill+decode caller
    /// can execute one unified MLP/MoE pass over all `P + D` rows.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_cross_seq_mixer_only(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _frame: Qwen4CrossSeqFrame,
        _seqs: &mut [Qwen4CrossSeqRows<'_>],
        _kv_cache: &mut PagedKvCache,
        _staging: Qwen4CrossSeqStaging,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support cross-sequence Qwen4 mixer-only verify")
    }

    /// Multi-row Qwen4 MTP forward: populates KV cache for all `num_tokens`
    /// rows, skipping Q projection, attention, O projection, and MoE on prefix
    /// replay rows (`0..num_tokens - 1` when `emit_last_row`, or all
    /// `0..num_tokens` when `!emit_last_row`). Default falls back to
    /// `decode_qwen4_batched`.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_batched_mtp(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        emit_last_row: bool,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = emit_last_row;
        self.decode_qwen4_batched(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            DevicePtr::NULL,
            DevicePtr::NULL,
            0,
            0,
            ctx,
            stream,
        )
    }

    /// Cross-sequence Qwen4 MTP forward for `seq_lens.len()` independent 1-row
    /// sequences: batches the attention hyperconnection, exact QKV projection,
    /// exact O projection, MLP hyperconnection, and exact worklist MoE across
    /// all sequences while running each sequence's RoPE, KV write, and paged
    /// attention with its own `seq_lens[s]` and `block_tables[s]`.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_cross_seq_mtp(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _seq_lens: &[usize],
        _block_tables: &mut [&mut Vec<u32>],
        _disk_block_ids: &mut [&mut Vec<u32>],
        _disk_last_offloaded_per_layer: &mut [&mut Vec<u32>],
        _kv_cache: &mut PagedKvCache,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support cross-sequence Qwen4 MTP")
    }

    /// Read-only admission for one layer of the target-only native Qwen4
    /// multi-sequence path (up to `QWEN4_CROSS_SEQ_MAX_ROWS`, 256 rows).
    ///
    /// The model walks every layer with the actual request-owned states before
    /// allocating KV blocks, embedding tokens, uploading metadata, or
    /// submitting any GPU work. Implementations must not mutate `states` or
    /// device storage. The default is deliberately fail-closed so a new layer
    /// kind cannot enter the native path merely by implementing an execution
    /// hook.
    fn preflight_qwen4_native_target(
        &self,
        _num_seqs: usize,
        _states: &[&(dyn LayerState + 'static)],
        _ctx: &ForwardContext,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 target preflight")
    }

    /// The state-independent part of [`Self::preflight_qwen4_native_target`]
    /// for a C`num_seqs` frame, as a typed gap. The model evaluates it once
    /// per width before routing, so a frame this layer cannot run serializes
    /// (or, with `ATLAS_QWEN4_NATIVE_BATCH=1`, refuses at load) instead of
    /// failing its rows. Read-only; fail-closed by default.
    fn qwen4_native_capability(
        &self,
        _num_seqs: usize,
        _ctx: &ForwardContext,
    ) -> std::result::Result<(), crate::qwen4_batch_policy::NativeGap> {
        Err(crate::qwen4_batch_policy::NativeGap::new(
            crate::qwen4_batch_policy::SerialReason::UnsupportedLayer,
            "layer does not support native Qwen4 target preflight",
        ))
    }

    /// Qwen4 target-only native cross-sequence SSM stage.
    ///
    /// This is deliberately separate from [`Self::decode_multi_seq`].  The
    /// generic multi-sequence hook is shared by established decode and verify
    /// routes, while this hook owns the native Qwen4 recurrent state lane
    /// (up to `QWEN4_CROSS_SEQ_MAX_ROWS`, 256 rows). Only the native Qwen4
    /// model loop may call it. All other layer types fail closed by default.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_native_target_ssm<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 target SSM")
    }

    /// Qwen4 target-only native cross-sequence ATTENTION stage.
    ///
    /// This hook intentionally stops after the attention hyperconnection has
    /// injected the O-projection result into each independent 4H residual row.
    /// It is not a complete transformer-layer forward: the caller must still
    /// run the matching MLP hyperconnection/MoE stage. Admits exact BF16
    /// multi-sequence frames (up to `QWEN4_CROSS_SEQ_MAX_ROWS`, 256 rows)
    /// below QSA's sparse boundary. All non-attention layers fail closed by
    /// default.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_native_target_attention(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 target attention")
    }

    /// Complete the MLP half of a native Qwen4 attention layer.
    ///
    /// The attention hook above intentionally stops after injecting its
    /// O-projection. A model loop that selects the native multi-sequence lane
    /// (up to `QWEN4_CROSS_SEQ_MAX_ROWS`, 256 rows) must call this hook
    /// immediately afterwards so the layer still has exactly one attention
    /// transition and one MLP transition for every active row. Non-attention
    /// layers fail closed by default.
    fn decode_qwen4_native_target_mlp(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 target MLP")
    }

    /// Read-only admission of a native Qwen4 frame's QSA rows
    /// (`ATLAS_QWEN4_NATIVE_BATCH_QSA=1`). The model calls it for every
    /// full-attention layer after KV ownership checks and before any effect.
    /// Fail-closed by default.
    fn preflight_qwen4_native_target_qsa(
        &self,
        _routes: &[crate::layers::qwen4_qsa::native_batch::NativeAttnRoute],
        _seq_lens: &[usize],
        _kv_cache: &PagedKvCache,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 QSA rows")
    }

    /// Native Qwen4 attention stage for a frame holding QSA rows. Identical
    /// to [`Self::decode_qwen4_native_target_attention`] except that each
    /// row's attention core follows `routes`. Fail-closed by default.
    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_native_target_attention_qsa(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _routes: &[crate::layers::qwen4_qsa::native_batch::NativeAttnRoute],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("layer does not support native Qwen4 QSA attention")
    }

    /// Prefill N tokens through this layer using GEMM-batched projections.
    ///
    /// Used during prompt processing: reads weight matrices once for all N
    /// tokens (GEMM M=N) instead of N separate GEMV calls. Attention uses
    /// Flash Attention on contiguous Q/K/V. SSM/GDN recurrence remains
    /// sequential per-token.
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, read and written
    /// * `residual` - [N, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (N)
    /// * `state` - Per-layer state (SSM state updated sequentially)
    /// * `kv_cache` - Paged KV cache (attention layers write K/V for all N)
    /// * `seq_len_start` - Sequence position of first token (usually 0)
    /// * `block_table` - Block table for KV cache (pre-allocated for N tokens)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    ///
    /// Default: falls back to sequential single-token decode calls.
    ///
    /// `kv_write_start`: number of tokens whose KV cache entries are already
    /// populated (prefix caching). Attention layers skip KV writes for
    /// positions `< kv_write_start`. SSM layers ignore this (recurrent).
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = t * h * 2; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset);
            let r_t = residual.offset(offset);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Two-phase SSM prefill — Phase 1: projections and GDN input staging.
    ///
    /// Runs RMS norm, QKVZ projection, BA+gates, conv1d, and L2 norm for a
    /// chunk of `num_tokens` tokens, then copies the GDN inputs (packed QKV,
    /// gate/beta, Z) into the full-sequence `gdn_bufs` at `token_offset`.
    ///
    /// Does NOT run the GDN recurrence — that happens in `prefill_gdn_full`
    /// after all chunks have staged their inputs.
    ///
    /// Attention layers: default falls back to full `prefill` (no phasing).
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase1(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Default: fall back to full prefill (attention layers, non-SSM layers)
        let _ = (gdn_bufs, token_offset);
        self.prefill(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    /// Two-phase SSM prefill — Phase 2: GDN recurrence on the full sequence.
    ///
    /// Runs the WY4-persistent GDN kernel over all `total_len` tokens in
    /// `gdn_bufs` in a single launch. The kernel reads packed QKV and
    /// gate/beta from the full-sequence buffers and writes the GDN output.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    fn prefill_gdn_full(
        &self,
        _state: &mut dyn LayerState,
        _gdn_bufs: &GdnPrefillBuffers,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Batched attention prefill across N stacked-input streams.
    ///
    /// Runs the full attention-layer prefill (rms_norm + residual, QKV proj,
    /// RoPE, KV-write, batched attention compute, O proj, post-attn norm,
    /// FFN, final residual) over `num_tokens = batch_size * chunk_len`
    /// stacked tokens, using `batched_meta` for per-stream metadata
    /// resolution.
    ///
    /// Default impl returns Err — only `Qwen3AttentionLayer` overrides.
    /// SSM/dense layers don't override (they have their own batched paths
    /// or work without batched metadata).
    ///
    /// Caller (model-level `prefill_attn_batched_layer`) is responsible for
    /// ensuring all streams share the same chunk_len, seq_len_start
    /// (q_offset), and that the layer is not MLA / not HDIM=512 / not HSS-
    /// engaged. The override bails Err if any unsupported case is detected.
    fn prefill_inner_batched_q12(
        &self,
        _hidden_stacked: DevicePtr,
        _residual_stacked: DevicePtr,
        _num_tokens: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _batched_meta: &BatchedAttnMetadata,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("prefill_inner_batched_q12: not implemented for this layer type")
    }

    /// Batched GDN recurrence across N streams.
    ///
    /// Runs the same WY32 / persistent / split4 GDN kernel as
    /// `prefill_gdn_full` but with `batch_size = batch_size` and
    /// `h_state_ptrs` pointing to a device array of N per-stream h_state
    /// pointers (staged by `TransformerModel::stage_h_state_ptrs`).
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked across N
    /// streams contiguously: each stream's data lives at
    /// `b * chunk_len * conv_dim` (BF16) within the buffer.
    ///
    /// Default impl returns `Err` — the SSM layer override implements the
    /// actual batched dispatch using its batched GDN kernel handles. Attention
    /// layers don't override (they don't have a GDN step).
    fn prefill_gdn_full_batched(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _chunk_len: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!(
            "prefill_gdn_full_batched: layer does not implement batched GDN \
             — caller should fall back to per-stream prefill_gdn_full"
        )
    }

    /// Two-phase SSM prefill — Phase 3: post-GDN processing.
    ///
    /// Reads GDN output and Z gate from `gdn_bufs` at `token_offset`,
    /// then runs gated RMS norm, output projection, residual add, and MoE
    /// for the chunk of `num_tokens` tokens.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase3(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _gdn_bufs: &GdnPrefillBuffers,
        _token_offset: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Returns true if this layer is an SSM layer (supports two-phase prefill).
    ///
    /// When true, the model loop can use `prefill_phase1` / `prefill_gdn_full` /
    /// `prefill_phase3` instead of the monolithic `prefill`.
    fn is_ssm_layer(&self) -> bool {
        false
    }

    /// Whether this layer preserves exact DDTree ancestor visibility when
    /// `ForwardContext::tree_aware_attn` is present.
    ///
    /// Non-attention layers return true: they do not consume the paged KV
    /// cache. Attention implementations must override this and return false
    /// for any KV dtype whose decode kernel ignores the tree indirection.
    /// `verify_d.rs` requires every layer to pass before it allows a branch
    /// commit; this deliberately fails closed as new KV formats are added.
    fn ddtree_ancestor_attention_exact(&self) -> bool {
        true
    }

    /// Per-attention-layer DDTree indirection certificate. Non-attention
    /// layers return `None`; attention layers report their cache dtype and
    /// whether the matching tree-aware decode handle is resolved.
    fn ddtree_attention_certificate(&self) -> Option<(KvCacheDtype, bool)> {
        None
    }

    /// Whether this layer can preserve DDTree's convolutional state along a
    /// non-flat parent chain. Non-SSM layers return true because they have no
    /// recurrent convolutional state; SSM implementations must fail closed
    /// when the tree re-root kernel is unavailable.
    fn ddtree_conv_state_exact(&self) -> bool {
        true
    }

    /// Allocate the transposed MoE expert weights used by the coalesced
    /// prefill GEMM kernels. Called as a post-load pass from `factory::build`
    /// after LM-head NVFP4 quantization has freed BF16 headroom, so
    /// memory-tight EP configurations (e.g. MiniMax M2.7-NVFP4 EP=2) can
    /// fit the transpose that layer-0 preflight would otherwise reject.
    ///
    /// Default: no-op (non-MoE layers, and MoE layers whose loader already
    /// called `MoeLayer::transpose_for_prefill` inline during construction).
    fn transpose_moe_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Like `transpose_moe_for_prefill` but only transposes the gate+up
    /// projections (skips the down projection), reducing the transpose cost
    /// from 3× to 2× per expert. Used as a memory-tight fallback by the
    /// MiniMax loader when full transpose doesn't fit.
    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Wire a shared per-prefill `down_proj` transpose scratch into this
    /// layer's MoE block. Used as a memory-tight alternative to the
    /// persistent down transpose: factory allocates one shared scratch,
    /// every MoE layer reuses it layer-by-layer during sequential
    /// prefill. No-op for non-MoE layers and MoE layers that already
    /// have a persistent transposed down.
    fn set_moe_down_transpose_scratch(
        &mut self,
        _scratch_packed: DevicePtr,
        _scratch_scale: DevicePtr,
        _packed_ptrs_t: DevicePtr,
        _scale_ptrs_t: DevicePtr,
    ) {
    }

    /// Wire the shared complete gate/up/down streaming-transpose arena.
    /// Originals remain authoritative for decode; prefill overwrites this
    /// scratch immediately before each layer consumes it.
    fn set_moe_stream_transpose_scratch(&mut self, _scratch: MoeStreamTransposeScratch) {}

    /// Phase 8a unified-layout MoE transpose: build persistent transposed
    /// gate/up/down for all experts and free the untransposed copies.
    /// Phased flow keeps memory budget tight enough for MiniMax M2.7 EP=2.
    /// After this call, the untransposed-layout decode kernels can no
    /// longer execute correctly — `MoeLayer::use_t_layout_for_decode()` must
    /// gate dispatch to the `_t` decode kernels. Default no-op.
    fn transpose_moe_for_prefill_unified(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Hybrid-layout MoE transpose: build persistent
    /// transposed gate/up/down alongside the untransposed originals (no
    /// frees). Doubles MoE-weight memory but recovers the ~15 % decode
    /// regression of pure unified mode — decode + MTP verify dispatch
    /// keeps using the warp-reduction kernels on the originals while
    /// prefill (forward_batched) routes through transposed kernels.
    /// Caller must verify enough free memory before invocation. Default
    /// no-op for non-MoE layers.
    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Decode K tokens through this layer using GEMM-batched projections.
    ///
    /// Used for speculative decode verification: processes multiple tokens
    /// per layer with GEMM for weight-heavy projections (amortizes bandwidth)
    /// and sequential ops for stateful/recurrent components.
    ///
    /// # Arguments
    /// * `hidden` - [K, hidden_size] BF16, read and written (K tokens contiguous)
    /// * `residual` - [K, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (K)
    /// * `state` - Per-layer state
    /// * `kv_cache` - Paged KV cache
    /// * `seq_len` - Starting sequence length (before these tokens)
    /// * `block_table` - Block table for KV cache
    /// * `ctx` - Shared context
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = (t * h * 2) as u64; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset as usize);
            let r_t = residual.offset(offset as usize);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Decode N sequences through this layer in a single batched call.
    ///
    /// Each sequence contributes 1 token. The weight matrices are loaded
    /// once and applied to all N sequences (amortizing memory bandwidth).
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, contiguous
    /// * `residual` - [N, hidden_size] BF16, contiguous
    /// * `num_seqs` - Number of sequences (N)
    /// * `states` - N per-layer states (one per sequence)
    /// * `kv_cache` - Shared paged KV cache
    /// * `ctx` - Forward context (attn_metadata contains N-sequence metadata)
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to N sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        for i in 0..num_seqs {
            let offset = i * h * 2;
            let h_i = hidden.offset(offset);
            let r_i = residual.offset(offset);
            let mut bt = block_tables[i].clone();
            // Per-seq disk_block_ids aren't threaded through this default impl;
            // pass empty stubs so the trait sig is satisfied (layers that need
            // disk IDs override decode_multi_seq).
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            self.decode(
                h_i,
                r_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut bt,
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Cross-sequence batched DFlash verify: project Q/K/V for `num_rows`
    /// contiguous rows of `hidden` (all `c*K` rows across every sequence) in
    /// ONE weight read per Q/K/V, writing the result into `qkv_out_base` in the
    /// per-seq-strided layout that [`Self::decode_multi_seq_attn_from_qkv`]
    /// consumes. RMS-norm (`input_norm`) is applied inline. `hidden` is only
    /// read (the residual stream is untouched).
    ///
    /// Default: unimplemented (only the attention layer overrides). Returns an
    /// error so the caller falls back to the per-seq v1 path.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_qkv_batched(
        &self,
        _hidden: DevicePtr,
        _num_rows: usize,
        _qkv_out_base: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("decode_multi_seq_qkv_batched not implemented for this layer type")
    }

    /// Cross-sequence batched DFlash verify: run attention phases 3-7 for
    /// ONE sequence's `num_seqs` (= K) rows, reading Q/K/V from `qkv_base` (a
    /// slice of the shared `qkv_output` populated by
    /// [`Self::decode_multi_seq_qkv_batched`]). Phases 1-2 are skipped.
    ///
    /// Default: unimplemented (only the attention layer overrides).
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_attn_from_qkv<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _ctx: &ForwardContext,
        _stream: u64,
        _qkv_base: DevicePtr,
    ) -> Result<()> {
        anyhow::bail!("decode_multi_seq_attn_from_qkv not implemented for this layer type")
    }

    /// Cross-sequence batched DFlash verify: run ONLY attention phases 3-5
    /// (RoPE, KV cache write, paged decode) for one sequence's `k` rows,
    /// reading pre-projected Q/K/V from `qkv_base` and writing paged-attention
    /// outputs directly into `attn_out_dst` (`[k, q_dim]` BF16) so the caller
    /// can hoist gate-mul + `o_proj` + post-norm + FFN across all `c * k` rows.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_attn_core_from_qkv(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _ctx: &ForwardContext,
        _stream: u64,
        _qkv_base: DevicePtr,
        _attn_out_dst: DevicePtr,
    ) -> Result<()> {
        anyhow::bail!("decode_multi_seq_attn_core_from_qkv not implemented for this layer type")
    }

    /// Cross-sequence batched DFlash verify: run hoisted phase 6 (gated
    /// sigmoid multiply + `o_proj` + TP all-reduce) and phase 7 (post-attn
    /// residual RMS norm + FFN + residual add) once over all `total_rows = c * k`
    /// rows whose paged-attention outputs were gathered in `attn_out_base`.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_o_proj_post_norm_ffn_batched(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _total_rows: usize,
        _qkv_base: DevicePtr,
        _attn_out_base: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!(
            "decode_multi_seq_o_proj_post_norm_ffn_batched not implemented for this layer type"
        )
    }

    /// Pre-allocate retain lanes (if needed) and serialize the 6-table
    /// cross-sequence SSM pointer table (`h`, `rows`, `gates|1`, `h_inter`,
    /// `conv`, `conv_inter`) for [`Self::decode_ssm_batched_cross_seq`] into
    /// `dst` (`DFLASH_SSM_PTR_TABLE_BYTES` bytes) before layer 0 / graph replay.
    ///
    /// Returns `Ok(true)` when this layer will consume the staged pointer table
    /// (`num_seqs >= 4` with lazy multi-seq kernels loaded), `Ok(false)` when
    /// no table upload is required.
    #[allow(clippy::too_many_arguments)]
    fn stage_dflash_cross_seq_ssm_ptr_table(
        &self,
        _num_seqs: usize,
        _k: usize,
        _slot_ids: &[usize],
        _states: &[&(dyn LayerState + 'static)],
        _ctx: &ForwardContext,
        _dst: &mut [u8],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Cross-sequence batched DFlash verify for SSM/GDN layers:
    /// runs input RMS norm, hoisted `in_proj_qkvz` + `in_proj_ba`/gates across
    /// all `c * k` rows, recurrent conv1d + GDN + gated RMS norm across all `c`
    /// sequences (binding each sequence to its own `slot_ids[s]` retain lane),
    /// hoisted `out_proj` (+ TP all-reduce), post-attn residual RMS norm into
    /// `norm_output`, and `run_deferred_ffn` directly into `hidden`.
    ///
    /// Returns `Ok(true)` when the hoisted SSM+FFN path executed completely, or
    /// `Ok(false)` when this layer is ineligible and the caller should fall
    /// back to the per-sequence `decode_batched` + `run_deferred_ffn` path.
    #[allow(clippy::too_many_arguments)]
    fn decode_ssm_batched_cross_seq<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_seqs: usize,
        _k: usize,
        _slot_ids: &[usize],
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Cross-sequence batched DFlash verify: run this layer's FFN ONCE over
    /// `total_rows` rows that were collected from every sequence's mixer output
    /// (via `ctx.ffn_defer`), then add the FFN output back into `hidden`.
    ///
    /// `ffn_input` is a contiguous `[total_rows, H]` BF16 buffer holding the
    /// post-mixer FFN input for every sequence's K rows (written by the mixer's
    /// deferred path). `hidden` is the same `[total_rows, H]` residual stream.
    /// This is the weight-amortizing step: the FFN weights are read ONCE for
    /// all `c×K` rows instead of once per sequence.
    ///
    /// Default: unimplemented (only the concrete SSM/attention layers with an
    /// FFN override it). Layers with no FFN return `Ok(())`.
    #[allow(clippy::too_many_arguments)]
    fn run_deferred_ffn(
        &self,
        _ffn_input: DevicePtr,
        _hidden: DevicePtr,
        _total_rows: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("run_deferred_ffn not implemented for this layer type")
    }

    /// Allocate per-sequence state for this layer.
    ///
    /// Called once when a new sequence is created. Returns:
    /// - `EmptyLayerState` for pure attention layers
    /// - `SsmLayerState` for SSM/recurrent layers
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>>;

    /// Piecewise-CUDA-graph hook: report whether, for a batch of `num_seqs`
    /// sequences, this layer's multi-seq decode takes the address-stable
    /// path (all per-slot device addresses indirected through fixed
    /// scratch), making it safe to include in a captured segment graph.
    ///
    /// Default `false` — a layer that returns `false` forces a segment
    /// boundary (the piecewise dispatcher runs it eagerly). Overridden by
    /// SSM layers that take the multi-seq kernel path.
    fn multiseq_graph_safe(&self, _num_seqs: usize) -> bool {
        false
    }

    /// Piecewise-CUDA-graph hook: gather-before-replay. Refresh any
    /// layer-local host→device pointer table (e.g. SSM h_state/conv_state
    /// indirection) with the current active sequences' addresses so a
    /// previously-captured segment graph replays against live state.
    ///
    /// Default no-op — layers with no per-slot indirection need nothing.
    /// Called by the piecewise dispatcher BEFORE replaying a segment that
    /// contains this layer, OUTSIDE the captured region. See
    /// `Qwen3SsmLayer::refresh_multi_seq_ptr_table`.
    #[allow(clippy::too_many_arguments)]
    fn multiseq_refresh_ptr_table<'a, 'b: 'a>(
        &self,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _num_seqs: usize,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// WY17 LAZY-commit hook: the `gated_delta_rule_wy17_replay` kernel handle
    /// for this layer, or a null handle (`KernelHandle(0)`) if this layer type
    /// / target doesn't provide it. The async-checkpoint commit path uses it to
    /// reconstruct a skipped (non-checkpoint) intermediate H slot on a partial
    /// accept. Default null — only SSM layers with the lazy kernel loaded
    /// return a live handle. See `Qwen3SsmLayer`.
    fn wy17_replay_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        spark_runtime::gpu::KernelHandle(0)
    }

    /// GDN LAZY-commit (ExactSequence path, `ATLAS_SSM_GDN_LAZY=1`): TRUE when
    /// a `num_tokens`-wide verify on this layer dispatched the `lazyfinal`
    /// kernel (no per-token H snapshots; h_state left at step-initial). Pure
    /// function of num_tokens + process-constant env + kernel handles, so the
    /// async commit can mirror the dispatch decision exactly (graph-safe, no
    /// per-step mutable flag). Default false — non-SSM layers.
    fn gdn_seq_lazy_engaged(&self, _num_tokens: usize) -> bool {
        false
    }

    /// The FP32 replay kernel for the lazy ExactSequence commit — the nosnap
    /// sequence kernel, which re-runs the identical recurrence over the
    /// retained inputs. Null when unavailable.
    fn gdn_seq_replay_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        spark_runtime::gpu::KernelHandle(0)
    }

    /// Layer-owned retention buffer for the lazy ExactSequence commit:
    /// [16 rows x qkvz_size] fp32 of this step's post-conv q/k/v(+z), with the
    /// [16 x 2*nv] fp32 gate/beta block at byte offset 16*qkvz_size*4.
    /// None until first lazy dispatch allocates it. Default None.
    fn gdn_seq_lazy_retain(&self) -> Option<spark_runtime::gpu::DevicePtr> {
        None
    }

    /// Whether this layer can participate in zero-replay `KEEP_LATE` DeltaNet
    /// commits (fused multi-seq lazyfinal kernel loaded with group-done counter).
    /// Default `false` — non-SSM layers.
    fn gdn_keep_late_supported(&self) -> bool {
        false
    }

    /// Whether `slot`'s most recent verify on this layer ran with
    /// `GDN_FUSED_FLAG_KEEP_LATE` (raw BF16 conv tail populated and live
    /// `conv_state` left at step-initial), making zero-replay commit deferral
    /// safe. Default `false` — non-SSM layers.
    fn gdn_keep_late_armed(&self, _slot: usize) -> bool {
        false
    }

    /// Notify this layer that a verify step for `slot` completed on GPU
    /// (covering CUDA-graph-replayed suffix layers whose Rust forward did not
    /// execute on host). If `slot` had a `GdnKeepLateCtrl` staged for this
    /// verify, marks prior pending rows materialized on GPU and arms
    /// `KEEP_LATE` commit deferral for this step. Default no-op.
    fn gdn_keep_late_note_verify_completed(&self, _slot: usize) {}

    /// Error-only abort reconciliation (called after synchronizing the default
    /// stream): checks whether the GPU actually executed the staged
    /// `GdnKeepLateCtrl` for `slot`, clearing `pending_rows` if already
    /// materialized so a subsequent flush never double-applies them. Default
    /// no-op.
    fn gdn_keep_late_reconcile_abort(&self, _gpu: &dyn GpuBackend, _slot: usize) -> Result<()> {
        Ok(())
    }

    /// Record `num_committed` pending DeltaNet commit rows (`1..=16`) for
    /// `slot`, flipping its ping-pong retain buffer so the next fused verify
    /// folds the commit directly in registers (0 replay launches, 0 conv D2D
    /// copies). Returns `true` when recorded, `false` if deferral is not armed.
    #[allow(clippy::too_many_arguments)]
    fn gdn_keep_late_record(
        &self,
        _slot: usize,
        _num_committed: usize,
        _h_state: DevicePtr,
        _conv_state: DevicePtr,
        _conv_inter_base: DevicePtr,
        _conv_inter_stride: usize,
        _h_scratch: DevicePtr,
    ) -> bool {
        false
    }

    /// Clear `slot`'s pending `KEEP_LATE` count and per-step staged flag after
    /// GPU execution has already materialized prior pending rows (e.g. a
    /// full-reject `num_accepted == 0` commit). Default no-op.
    fn gdn_keep_late_clear_pending(&self, _slot: usize) {}

    /// Reset all `KEEP_LATE` tracking state for `slot` (slot retirement,
    /// compaction, or state restore). Default no-op.
    fn gdn_keep_late_reset_slot(&self, _slot: usize) {}

    /// Whether `slot` currently holds unmaterialized `KEEP_LATE` commit rows
    /// on this layer. Default `false`.
    fn gdn_keep_late_has_pending(&self, _slot: usize) -> bool {
        false
    }

    /// Whether `slot` currently holds an in-place Scaled-FP16 `h_state` under
    /// `KEEP_LATE` that must be unpacked to FP32 before single-seq Split-V
    /// dispatch. Default `false`.
    fn gdn_keep_late_has_scaled_fp16(&self, _slot: usize) -> bool {
        false
    }

    /// Materialize any pending `KEEP_LATE` commit rows for `slot` into
    /// `h_state` and `conv_state` (plus optional `conv_ckpt`) using the exact
    /// FP32 replay kernel and conv intermediate copy, then reset
    /// `pending_rows = 0`. No-op when `pending_rows == 0` or on non-SSM layers.
    #[allow(clippy::too_many_arguments)]
    fn gdn_keep_late_flush_slot(
        &self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
        _slot: usize,
        _h_state: DevicePtr,
        _conv_state: DevicePtr,
        _conv_ckpt: Option<DevicePtr>,
        _conv_inter_base: DevicePtr,
        _conv_inter_stride: usize,
        _h_scratch: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Pre-allocate cross-sequence fused GDN descriptor and retain device
    /// buffers outside CUDA graph capture so zero `gpu.alloc` occurs inside
    /// capture. Default no-op (non-SSM layers).
    fn ensure_qwen4_cross_seq_fused_descs_ready(
        &self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
        _k: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Stage this layer's cross-sequence `GdnFusedMultiSeqDesc[0..frame.seqs()]`
    /// bytes and mark its persistent descriptor device buffer pre-staged for the
    /// upcoming frame. Callers upload all returned `(dest_dev_ptr, bytes)` pairs
    /// in one Phase-1 `copy_h2d_group_fast` call before `begin_capture` or graph
    /// replay so the layer loop issues zero H2D copies and zero events.
    /// Default `Ok(None)` (non-SSM layers).
    fn stage_qwen4_cross_seq_fused_desc_bytes(
        &self,
        _frame: Qwen4CrossSeqFrame,
        _seqs: &[Qwen4CrossSeqRows<'_>],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<Option<(DevicePtr, Vec<u8>)>> {
        Ok(None)
    }

    /// Stage this layer's 32-byte per-slot `GdnKeepLateCtrl` payload
    /// (`retain_qkv`, `retain_prev`, `late_pending_rows`, `executed_flag = 0`)
    /// so `pre_verify_copy_async_dispatch` can upload all SSM layers in one
    /// `copy_h2d_group_fast` call before graph capture or replay.
    /// Default `Ok(None)` (non-SSM layers).
    fn stage_qwen4_keep_late_ctrl_bytes(
        &self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
        _slot: usize,
        _h_state: DevicePtr,
    ) -> Result<Option<(DevicePtr, [u8; 32])>> {
        Ok(None)
    }

    /// Stage one 64-byte `GdnFusedMultiSeqDesc` for single-sequence (`C=1`)
    /// `KEEP_LATE` verify on `slot` and return `(dest_dev_ptr, bytes)` so
    /// `pre_verify_copy_async_dispatch` can upload all SSM layers in one
    /// `copy_h2d_group_fast` call before graph capture or replay.
    /// Default `Ok(None)` (non-SSM layers).
    #[allow(clippy::too_many_arguments)]
    fn stage_qwen4_single_seq_fused_desc_bytes(
        &self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
        _slot: usize,
        _k: usize,
        _h_state: DevicePtr,
        _conv_state: DevicePtr,
        _conv_inter_base: DevicePtr,
        _stream: u64,
    ) -> Result<Option<(DevicePtr, [u8; 64])>> {
        Ok(None)
    }

    /// TRUE when this layer's K=γ GDN dispatch would take the tree-aware
    /// kernel branch for a ForwardContext with `ddtree_parent_ids_dev` set
    /// (i.e. `gdn_tree_k` is loaded). Used by `verify_d.rs` to decide whether
    /// the graph-safe flat-chain parent injection actually reroutes the SSM
    /// to `gated_delta_rule_tree_wy` (which leaves `h_state` stale, so the
    /// commit path must know). Default false — non-SSM layers.
    fn gdn_tree_kernel_loaded(&self) -> bool {
        false
    }

    /// TRUE when a `num_tokens`-wide verify on this layer runs the LAZY wy17
    /// kernel (`gated_delta_rule_wy17_lazy`, sparse intermediate H writes).
    /// The async-checkpoint commit consults this before choosing the
    /// `gated_delta_rule_wy17_replay` path: replay is only bit-exact when the
    /// lazy kernel populated the retention buffers THIS verify — a K≠17
    /// (chunked) verify under global `ATLAS_WY17_LAZY` env gates must use the
    /// plain intermediate D2D copy instead.
    /// Default false — non-SSM layers. See `Qwen3SsmLayer`.
    fn wy17_lazy_engaged(&self, _num_tokens: usize) -> bool {
        false
    }
}
