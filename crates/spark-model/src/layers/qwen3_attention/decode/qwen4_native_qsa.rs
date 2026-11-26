// SPDX-License-Identifier: AGPL-3.0-only

//! Per-row QSA attention core for the native Qwen4 C2-C32 frame
//! (`ATLAS_QWEN4_NATIVE_BATCH_QSA=1`).
//!
//! The native seam has already run the batched hyperconnection, exact QKV,
//! RoPE and KV writes for every row. Each QSA row then replays the C1 order
//! (`attention_forward.rs`: KV write, QSA index, attention core) through the
//! unchanged C1 functions on a one-row metadata view:
//! `qwen4_qsa_indices` maintains the side index (and selects top-512),
//! `sparse_attention_bf16` is the sparse core, `run_paged_decode` with one
//! row is the dense core. Rows run strictly in order on one stream because
//! the QSA scratch (projected QK, pooled key, logits, selected tokens,
//! sparse partials) holds a single row per layer.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::qwen4_qsa::native_batch::{NativeAttnRoute, row_metadata_view};

/// Frame geometry the native attention seam hands to the QSA row loop.
#[derive(Clone, Copy, Debug)]
pub(in crate::layers::qwen3_attention) struct NativeQsaFrame {
    pub rows: usize,
    /// `[rows, hidden]` BF16 HC-mixed attention input (the QKV input).
    pub normed: DevicePtr,
    /// `[rows, per_seq_qkv]` BF16 post-RoPE projections, Q first.
    pub qkv: DevicePtr,
    pub per_seq_qkv: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: u32,
}

impl Qwen3AttentionLayer {
    /// Read-only admission of this layer for a native frame's QSA rows.
    pub(in crate::layers::qwen3_attention) fn preflight_qwen4_native_qsa_rows(
        &self,
        routes: &[NativeAttnRoute],
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
    ) -> Result<()> {
        anyhow::ensure!(
            routes.len() == seq_lens.len(),
            "native Qwen4 QSA routes and sequence lengths differ"
        );
        if !routes.iter().any(|route| route.maintains_index()) {
            return Ok(());
        }
        let qsa = self.qwen4_qsa.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "native Qwen4 QSA rows need this layer's QSA indexer; serve with --qwen4-qsa"
            )
        })?;
        anyhow::ensure!(
            self.mla.is_none()
                && matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4)
                && matches!(
                    kv_cache.dtype_for_layer(self.attn_layer_idx),
                    KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4
                )
                && !self.high_speed_swap_engaged(kv_cache),
            "native Qwen4 QSA rows require resident BF16/NVFP4 KV without MLA or high-speed swap"
        );
        qsa.preflight_native_rows(kv_cache, routes, seq_lens)
    }

    /// Maintain every QSA row's side index and, when the frame holds a
    /// sparse row, run every row's attention core into `attn_output`.
    ///
    /// Returns `None` when no row is sparse: the caller then runs the
    /// unchanged batched dense paged decode over all rows.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn qwen4_native_qsa_rows(
        &self,
        frame: NativeQsaFrame,
        routes: &[NativeAttnRoute],
        seq_lens: &[usize],
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        anyhow::ensure!(
            routes.len() == frame.rows
                && seq_lens.len() == frame.rows
                && meta.num_seqs as usize == frame.rows,
            "native Qwen4 QSA frame needs one route, length and metadata row per active row"
        );
        self.preflight_qwen4_native_qsa_rows(routes, seq_lens, kv_cache)?;
        let attend = routes.iter().any(|route| route.is_sparse());
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let core_row_bytes = h * 2;
        let q_dim = frame.nq * frame.hd;
        let q_row_bytes = q_dim as usize * 2;
        let q_rows = ctx.buffers.ssm_qkvz();
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(frame.hd);
        if attend {
            // Same contiguous query rows `ms_phase_paged_decode` builds.
            for row in 0..frame.rows {
                ctx.gpu.copy_d2d_async(
                    frame.qkv.offset(row * frame.per_seq_qkv),
                    q_rows.offset(row * q_row_bytes),
                    q_row_bytes,
                    stream,
                )?;
            }
        }
        for (row, (&route, &seq_len)) in routes.iter().zip(seq_lens).enumerate() {
            let row_meta = row_metadata_view(meta, row, route.maintains_index())?;
            let indices = if route.maintains_index() {
                let row_ctx = ForwardContext {
                    attn_metadata: Some(row_meta),
                    ..*ctx
                };
                self.qwen4_qsa_indices(
                    frame.normed.offset(row * core_row_bytes),
                    Some(seq_len),
                    None,
                    kv_cache,
                    &row_ctx,
                    h as u32,
                    eps,
                    stream,
                )?
            } else {
                None
            };
            if !attend {
                continue;
            }
            let q_row = q_rows.offset(row * q_row_bytes);
            let out_row = attn_out.offset(row * q_row_bytes);
            match route {
                NativeAttnRoute::Sparse => {
                    let qsa = self
                        .qwen4_qsa
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Qwen4 context beyond 2048 requires QSA"))?;
                    let indices = indices.ok_or_else(|| {
                        anyhow::anyhow!("native Qwen4 QSA row {row} produced no token indices")
                    })?;
                    qsa.sparse_attention_bf16(
                        q_row,
                        indices,
                        out_row,
                        kv_cache,
                        row_meta,
                        self.attn_layer_idx,
                        frame.nq,
                        frame.nkv,
                        frame.hd,
                        inv_sqrt_d,
                        ctx.gpu,
                        stream,
                    )?;
                }
                NativeAttnRoute::Dense | NativeAttnRoute::IndexedDense => {
                    let max_seq_len_host = seq_len
                        .checked_add(1)
                        .and_then(|len| u32::try_from(len).ok())
                        .ok_or_else(|| anyhow::anyhow!("attention sequence length overflow"))?;
                    self.run_paged_decode(
                        ctx.gpu,
                        q_row,
                        kv_cache,
                        out_row,
                        row_meta.block_table,
                        row_meta.seq_len,
                        row_meta.max_blocks_per_seq,
                        1,
                        frame.nq,
                        frame.nkv,
                        frame.hd,
                        frame.bs,
                        inv_sqrt_d,
                        q_dim,
                        ctx.buffers.splitk_workspace(),
                        DevicePtr::NULL,
                        DevicePtr::NULL,
                        0,
                        max_seq_len_host,
                        stream,
                    )?;
                }
            }
        }
        Ok(attend.then_some(attn_out))
    }
}
