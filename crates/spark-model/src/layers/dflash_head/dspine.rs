// SPDX-License-Identifier: AGPL-3.0-only

//! Native DSpine (`arXiv:2609.36173`) Adjacent Causal Injection drafter runtime
//! and Pass-2 exact-sender `serial_table` refinement.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::device_allocations::DeviceAllocations;
use crate::layers::ops;
use crate::weight_loader::dflash_loader::{DspineBridgeWeights, DspineSpec};
use crate::weight_map::DenseWeight;

use super::{BlockDiffusionDraftHead, DflashLayerQuantWeights};

const DSP_THREADS: u32 = 256;
const DSP_TAIL_THREADS: u32 = 512;
const TABLE_BUILD_CHUNK: usize = 4096;

/// Resolved CUDA kernel handles from `kernels/gb10/common/dspine_bridge.cu`.
pub struct DspineKernels {
    pub read: KernelHandle,
    pub code: KernelHandle,
    pub message: KernelHandle,
    pub write_norm: KernelHandle,
    pub serial_table: KernelHandle,
    pub serial_walk: KernelHandle,
}

/// Prepared device weights, precomputed `[vocab, R]` / `[vocab, 2*R]` serial
/// tables, and per-propose scratch for native DSpine execution.
pub struct DspineCodeBridge {
    pub spec: DspineSpec,
    pub codebook: DenseWeight,
    pub read_weight: DenseWeight,
    pub early_read_weights: Option<DenseWeight>,
    pub feature_write_weights: DenseWeight,
    pub gate_bias: DenseWeight,
    /// Contiguous `[3 * rank, code_size]` BF16 `[gate; predecessor_gate; prev]`.
    pub packed_weight: DenseWeight,
    /// Precomputed `TG [lm_vocab, rank]` BF16: `bf16(lm_head * norm_w) @ W_last`.
    pub tg: DevicePtr,
    /// Precomputed `TSV [lm_vocab, 2 * rank]` BF16:
    /// `[predecessor_gate; prev] @ bf16(sqrt(C) * codebook)` (+ folded terminal residual).
    pub tsv: DevicePtr,
    pub lm_vocab: usize,
    // ── Scratch sized for `max_block_rows = gamma + 1` ──
    pub raw: DevicePtr,
    pub normalized: DevicePtr,
    pub rms: DevicePtr,
    pub projected: DevicePtr,
    pub codes: DevicePtr,
    pub packed: DevicePtr,
    pub message: DevicePtr,
    pub delta: DevicePtr,
    pub cand: DevicePtr,
    pub idx: DevicePtr,
    pub kernels: DspineKernels,
}

impl DspineCodeBridge {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        weights: DspineBridgeWeights,
        lm_head_shared: DevicePtr,
        norm: &DenseWeight,
        num_layers: usize,
        hidden_size: usize,
        lm_vocab: usize,
        max_block_rows: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            !lm_head_shared.is_null(),
            "DSpine requires a non-null shared BF16 target lm_head pointer"
        );
        anyhow::ensure!(num_layers >= 1, "DSpine requires num_layers >= 1");
        let spec = weights.spec;
        let code_size = spec.code_size;
        let rank = spec.rank;
        let hidden_u32 = u32::try_from(hidden_size).context("DSpine hidden_size exceeds u32")?;
        let code_u32 = u32::try_from(code_size).context("DSpine code_size exceeds u32")?;
        let rank_u32 = u32::try_from(rank).context("DSpine rank exceeds u32")?;
        let rank2_u32 = rank_u32
            .checked_mul(2)
            .context("DSpine 2*rank exceeds u32")?;
        let _ = rank_u32
            .checked_mul(3)
            .context("DSpine 3*rank exceeds u32")?;
        let _ = u32::try_from(lm_vocab).context("DSpine lm_vocab exceeds u32")?;
        let bf16 = 2usize;

        let kernels = DspineKernels {
            read: gpu.kernel("dspine_bridge", "dspine_bridge_read")?,
            code: gpu.kernel("dspine_bridge", "dspine_bridge_code")?,
            message: gpu.kernel("dspine_bridge", "dspine_bridge_message")?,
            write_norm: gpu.kernel("dspine_bridge", "dspine_bridge_write_norm")?,
            serial_table: gpu.kernel("dspine_bridge", "dspine_serial_table")?,
            serial_walk: gpu.kernel("dspine_bridge", "dspine_serial_walk")?,
        };
        let scale_kernel = gpu.kernel("dspine_bridge", "dspine_scale")?;
        let fold_tr_kernel = gpu.kernel("dspine_bridge", "dspine_fold_terminal_tsv")?;
        let dense_gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;

        let mut owned = DeviceAllocations::new(gpu);
        let stream = gpu.default_stream();

        // 1. Pack `[gate; predecessor_gate; prev]` into `[3 * rank, code_size]` BF16.
        let slice_bytes = rank * code_size * bf16;
        let packed_ptr = owned.alloc(3 * slice_bytes)?;
        gpu.copy_d2d(weights.gate.weight, packed_ptr, slice_bytes)?;
        gpu.copy_d2d(
            weights.predecessor_gate.weight,
            packed_ptr.offset(slice_bytes),
            slice_bytes,
        )?;
        gpu.copy_d2d(
            weights.prev.weight,
            packed_ptr.offset(2 * slice_bytes),
            slice_bytes,
        )?;
        let packed_weight = DenseWeight { weight: packed_ptr };

        // 2. Precompute `TG [lm_vocab, rank]` and `TSV [lm_vocab, 2 * rank]`.
        let tg = owned.alloc(lm_vocab * rank * bf16)?;
        let tsv = owned.alloc(lm_vocab * 2 * rank * bf16)?;

        let mut temp = DeviceAllocations::new(gpu);
        // Transpose `feature_write_weights[num_layers - 1]` from `[H, R]` to `[R, H]`
        // so `ops::dense_gemm` (`A [n, H] @ B [R, H]^T`) computes `A @ W_last`.
        let last_write_off = (num_layers - 1) * hidden_size * rank * bf16;
        let mut w_hr = vec![0u8; hidden_size * rank * bf16];
        gpu.copy_d2h(
            weights.feature_write_weights.weight.offset(last_write_off),
            &mut w_hr,
        )
        .context("D2H read of final DSpine feature_write_weights slice")?;
        let mut w_rh = vec![0u8; rank * hidden_size * bf16];
        for h_idx in 0..hidden_size {
            for r_idx in 0..rank {
                let src = (h_idx * rank + r_idx) * 2;
                let dst = (r_idx * hidden_size + h_idx) * 2;
                w_rh[dst..dst + 2].copy_from_slice(&w_hr[src..src + 2]);
            }
        }
        let write_last_t = temp.alloc(w_rh.len())?;
        gpu.copy_h2d(&w_rh, write_last_t)?;
        let write_last_t_dw = DenseWeight {
            weight: write_last_t,
        };

        let chunk_cols = hidden_size.max(code_size).max(rank);
        let chunk_buf = temp.alloc(TABLE_BUILD_CHUNK * chunk_cols * bf16)?;
        let pv_dw = DenseWeight {
            weight: packed_ptr.offset(slice_bytes),
        };
        let sqrt_c = (code_size as f32).sqrt();

        let mut row = 0usize;
        while row < lm_vocab {
            let n = (lm_vocab - row).min(TABLE_BUILD_CHUNK);
            let n_u32 = u32::try_from(n).context("DSpine chunk rows exceed u32")?;

            // TG[row..row+n] = bf16(lm_head[row..row+n] * norm_w) @ W_last
            let head_elems = (n * hidden_size) as u64;
            let grid_h = div_ceil(u32::try_from(head_elems).unwrap_or(u32::MAX), DSP_THREADS);
            KernelLaunch::new(gpu, scale_kernel)
                .grid([grid_h.min(65_535), 1, 1])
                .block([DSP_THREADS, 1, 1])
                .arg_ptr(lm_head_shared.offset(row * hidden_size * bf16))
                .arg_ptr(chunk_buf)
                .arg_u64(head_elems)
                .arg_u32(hidden_u32)
                .arg_f32(1.0)
                .arg_ptr(norm.weight)
                .launch(stream)?;
            ops::dense_gemm(
                gpu,
                dense_gemm,
                chunk_buf,
                &write_last_t_dw,
                tg.offset(row * rank * bf16),
                n_u32,
                rank_u32,
                hidden_u32,
                stream,
            )?;

            // TSV[row..row+n] = bf16(sqrt(C) * codebook[row..row+n]) @ [predecessor_gate; prev]^T
            let code_elems = (n * code_size) as u64;
            let grid_c = div_ceil(u32::try_from(code_elems).unwrap_or(u32::MAX), DSP_THREADS);
            KernelLaunch::new(gpu, scale_kernel)
                .grid([grid_c.min(65_535), 1, 1])
                .block([DSP_THREADS, 1, 1])
                .arg_ptr(weights.codebook.weight.offset(row * code_size * bf16))
                .arg_ptr(chunk_buf)
                .arg_u64(code_elems)
                .arg_u32(code_u32)
                .arg_f32(sqrt_c)
                .arg_ptr(DevicePtr::NULL)
                .launch(stream)?;
            ops::dense_gemm(
                gpu,
                dense_gemm,
                chunk_buf,
                &pv_dw,
                tsv.offset(row * 2 * rank * bf16),
                n_u32,
                rank2_u32,
                code_u32,
                stream,
            )?;

            // Fold optional terminal token residual into `TSV[row..row+n, R..2R]`.
            if spec.terminal_residual_rank > 0 {
                let rt = spec.terminal_residual_rank;
                let rt_u32 =
                    u32::try_from(rt).context("DSpine terminal_residual_rank exceeds u32")?;
                let tr_emb = weights
                    .terminal_residual_embedding
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("DSpine missing terminal_residual_embedding"))?;
                let tr_proj = weights
                    .terminal_residual_projection
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!("DSpine missing terminal_residual_projection")
                    })?;
                ops::dense_gemm(
                    gpu,
                    dense_gemm,
                    tr_emb.weight.offset(row * rt * bf16),
                    tr_proj,
                    chunk_buf,
                    n_u32,
                    rank_u32,
                    rt_u32,
                    stream,
                )?;
                let tr_elems = (n * rank) as u64;
                let grid_tr = div_ceil(u32::try_from(tr_elems).unwrap_or(u32::MAX), DSP_THREADS);
                KernelLaunch::new(gpu, fold_tr_kernel)
                    .grid([grid_tr.min(65_535), 1, 1])
                    .block([DSP_THREADS, 1, 1])
                    .arg_ptr(tsv.offset(row * 2 * rank * bf16))
                    .arg_ptr(chunk_buf)
                    .arg_u32(n_u32)
                    .arg_u32(rank_u32)
                    .launch(stream)?;
            }

            row += n;
        }
        gpu.synchronize(stream)?;
        drop(temp);

        // 3. Allocate per-propose scratch for `max_block_rows` (`gamma + 1`).
        let rows = max_block_rows.max(2);
        let draft_rows = rows - 1;
        let raw = owned.alloc(rows * hidden_size * bf16)?;
        let normalized = owned.alloc(rows * hidden_size * bf16)?;
        let rms = owned.alloc(rows * 4)?;
        let projected = owned.alloc(rows * code_size * bf16)?;
        let codes = owned.alloc(rows * code_size * bf16)?;
        let packed = owned.alloc(rows * 3 * rank * bf16)?;
        let message = owned.alloc(rows * rank * bf16)?;
        let delta = owned.alloc(rows * hidden_size * bf16)?;
        let cand = owned.alloc(draft_rows * spec.head_topk * 4)?;
        let idx = owned.alloc(draft_rows * spec.head_topk * 4)?;
        owned.commit();

        Ok(Self {
            spec,
            codebook: weights.codebook,
            read_weight: weights.read_weight,
            early_read_weights: weights.early_read_weights,
            feature_write_weights: weights.feature_write_weights,
            gate_bias: weights.gate_bias,
            packed_weight,
            tg,
            tsv,
            lm_vocab,
            raw,
            normalized,
            rms,
            projected,
            codes,
            packed,
            message,
            delta,
            cand,
            idx,
            kernels,
        })
    }
}

impl BlockDiffusionDraftHead {
    /// True when layer `layer_idx`'s `norm_buf` noise slice was already produced
    /// by the preceding layer's fused `dspine_bridge_write_norm`.
    #[inline]
    pub(super) fn dspine_fused_prev_input_norm(&self, layer_idx: usize) -> bool {
        layer_idx > 0
            && self
                .dspine
                .as_ref()
                .is_some_and(|d| d.spec.is_layer_active(layer_idx - 1))
    }

    /// Pass-1 DSpine Adjacent Causal Injection after `layer_idx`'s MLP residual
    /// add. Operates on the `[eff_ctx .. n_attn)` noise block (`block_rows` rows,
    /// where row 0 is the verified anchor token at `draft_tokens_dev[eff_ctx]`).
    pub(super) fn dspine_layer_bridge(
        &self,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
        eff_ctx: usize,
        n_attn: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(dspine) = &self.dspine else {
            return Ok(());
        };
        if !dspine.spec.is_layer_active(layer_idx) {
            return Ok(());
        }
        let eff_ctx_u32 = u32::try_from(eff_ctx).context("DSpine eff_ctx exceeds u32")?;
        let block_rows = n_attn.saturating_sub(eff_ctx_u32);
        if block_rows == 0 {
            return Ok(());
        }

        let bf16 = 2usize;
        let h = self.hidden_size;
        let code_size = dspine.spec.code_size;
        let rank = dspine.spec.rank;
        let h_u32 = u32::try_from(h).context("DSpine hidden_size exceeds u32")?;
        let code_u32 = u32::try_from(code_size).context("DSpine code_size exceeds u32")?;
        let rank_u32 = u32::try_from(rank).context("DSpine rank exceeds u32")?;
        let rank3_u32 = rank_u32
            .checked_mul(3)
            .context("DSpine 3*rank exceeds u32")?;
        let vocab_u32 = u32::try_from(self.vocab_size).context("DSpine vocab_size exceeds u32")?;
        let is_last = layer_idx + 1 == self.num_layers;

        let noise_byte_off = eff_ctx * h * bf16;
        let x_noise = self.scratch.stream_buf.offset(noise_byte_off);
        let norm_noise = self.scratch.norm_buf.offset(noise_byte_off);
        let anchors_dev = self.scratch.draft_tokens_dev.offset(eff_ctx * 4);

        // 1. Read & RMS-normalize pre-bridge hidden state.
        KernelLaunch::new(gpu, dspine.kernels.read)
            .grid([block_rows, 1, 1])
            .block([DSP_THREADS, 1, 1])
            .arg_ptr(x_noise)
            .arg_ptr(dspine.normalized)
            .arg_ptr(dspine.rms)
            .arg_u32(h_u32)
            .arg_f32(dspine.spec.eps)
            .launch(stream)?;

        // 2. Project normalized hidden state to code space `[block_rows, code_size]`.
        let read_w = if is_last {
            dspine.read_weight
        } else {
            let early = dspine.early_read_weights.as_ref().ok_or_else(|| {
                anyhow::anyhow!("DSpine missing early_read_weights for layer {layer_idx}")
            })?;
            DenseWeight {
                weight: early.weight.offset(layer_idx * code_size * h * bf16),
            }
        };
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            dspine.normalized,
            &read_w,
            dspine.projected,
            block_rows,
            code_u32,
            h_u32,
            stream,
        )?;

        // 3. Build spherical token codes (exact anchor codebook on row 0, L2-normed projection on 1..).
        KernelLaunch::new(gpu, dspine.kernels.code)
            .grid([block_rows, 1, 1])
            .block([DSP_THREADS, 1, 1])
            .arg_ptr(dspine.projected)
            .arg_ptr(dspine.codebook.weight)
            .arg_ptr(anchors_dev)
            .arg_u32(1)
            .arg_ptr(dspine.codes)
            .arg_u32(block_rows)
            .arg_u32(code_u32)
            .arg_u32(vocab_u32)
            .arg_f32(dspine.spec.eps)
            .launch(stream)?;

        // 4. Compute `[q | s | v]` (`[block_rows, 3 * rank]`).
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            dspine.codes,
            &dspine.packed_weight,
            dspine.packed,
            block_rows,
            rank3_u32,
            code_u32,
            stream,
        )?;

        // 5. Adjacent causal gated message `m_t = sigmoid(q_t + s_{t-1} + b) * v_{t-1}`.
        KernelLaunch::new(gpu, dspine.kernels.message)
            .grid([block_rows, 1, 1])
            .block([DSP_THREADS, 1, 1])
            .arg_ptr(dspine.packed)
            .arg_ptr(dspine.gate_bias.weight)
            .arg_ptr(dspine.message)
            .arg_u32(block_rows)
            .arg_u32(rank_u32)
            .launch(stream)?;

        // 6. Feature write projection `delta = message @ W_layer^T` (`[block_rows, H]`).
        let write_w = DenseWeight {
            weight: dspine
                .feature_write_weights
                .weight
                .offset(layer_idx * h * rank * bf16),
        };
        ops::dense_gemm(
            gpu,
            self.kernels.dense_gemm,
            dspine.message,
            &write_w,
            dspine.delta,
            block_rows,
            h_u32,
            rank_u32,
            stream,
        )?;

        // 7. Residual write + optional pre-write stash (final layer) + fused next-layer RMSNorm.
        let next_norm_w = if is_last {
            self.norm.weight
        } else {
            match &self.layers[layer_idx + 1] {
                DflashLayerQuantWeights::Bf16(next) => next.input_layernorm.weight,
                DflashLayerQuantWeights::Fp8(next) => next.input_layernorm.weight,
                DflashLayerQuantWeights::Nvfp4(next) => next.input_layernorm.weight,
            }
        };
        let stash_ptr = if is_last { dspine.raw } else { DevicePtr::NULL };
        KernelLaunch::new(gpu, dspine.kernels.write_norm)
            .grid([block_rows, 1, 1])
            .block([DSP_THREADS, 1, 1])
            .arg_ptr(x_noise)
            .arg_ptr(dspine.rms)
            .arg_ptr(dspine.delta)
            .arg_ptr(stash_ptr)
            .arg_ptr(next_norm_w)
            .arg_ptr(norm_noise)
            .arg_u32(block_rows)
            .arg_u32(h_u32)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)
    }

    /// Pass-2 DSpine exact-sender `serial_table` refinement + greedy chain walk,
    /// writing `gamma_eff` refined draft token IDs into `self.scratch.draft_tokens_dev`.
    pub(super) fn dspine_refine_tail(
        &self,
        gpu: &dyn GpuBackend,
        dspine: &DspineCodeBridge,
        eff_ctx: usize,
        gamma_eff: usize,
        lm_vocab: usize,
        stream: u64,
    ) -> Result<()> {
        let topk_k = dspine.spec.head_topk;
        let layout =
            super::forward_block::checked_topk_layout(gamma_eff, self.gamma, lm_vocab, topk_k)?;
        let k_u32 = u32::try_from(layout.k).context("DSpine topk k exceeds u32")?;
        let gamma_u32 = u32::try_from(gamma_eff).context("DSpine gamma_eff exceeds u32")?;
        let topk_u32 = u32::try_from(topk_k).context("DSpine head_topk exceeds u32")?;
        let rank_u32 = u32::try_from(dspine.spec.rank).context("DSpine rank exceeds u32")?;
        let h_u32 = u32::try_from(self.hidden_size).context("DSpine hidden_size exceeds u32")?;

        // 1. Extract top-K candidate IDs per draft row from Pass-1 logits.
        ops::topk_bf16(
            gpu,
            self.kernels.topk,
            self.scratch.logits,
            self.scratch.topk_tokens_dev,
            self.scratch.topk_logits_dev,
            layout.num_rows,
            layout.vocab,
            k_u32,
            stream,
        )?;

        // 2. Score `[K_sender, K_cand]` exact-sender table for all `gamma_eff` draft rows.
        let anchors_dev = self.scratch.draft_tokens_dev.offset(eff_ctx * 4);
        let table_vocab = u32::try_from(lm_vocab.min(dspine.lm_vocab))
            .context("DSpine table_vocab exceeds u32")?;
        KernelLaunch::new(gpu, dspine.kernels.serial_table)
            .grid([gamma_u32, 1, 1])
            .block([DSP_TAIL_THREADS, 1, 1])
            .arg_ptr(self.scratch.topk_tokens_dev)
            .arg_ptr(dspine.raw)
            .arg_ptr(dspine.rms)
            .arg_ptr(dspine.packed)
            .arg_ptr(self.norm.weight)
            .arg_ptr(dspine.gate_bias.weight)
            .arg_ptr(dspine.tg)
            .arg_ptr(dspine.tsv)
            .arg_ptr(anchors_dev)
            .arg_u32(1)
            .arg_ptr(self.lm_head_shared)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_f32(1.0)
            .arg_u32(h_u32)
            .arg_ptr(dspine.cand)
            .arg_ptr(dspine.idx)
            .arg_u32(gamma_u32)
            .arg_u32(topk_u32)
            .arg_u32(rank_u32)
            .arg_u32(table_vocab)
            .launch(stream)?;

        // 3. Walk greedy chain starting from anchor (`j = 0`) into `draft_tokens_dev[0..gamma_eff]`.
        KernelLaunch::new(gpu, dspine.kernels.serial_walk)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(dspine.cand)
            .arg_ptr(dspine.idx)
            .arg_ptr(self.scratch.draft_tokens_dev)
            .arg_u32(gamma_u32)
            .arg_u32(topk_u32)
            .launch(stream)
    }
}

#[cfg(test)]
#[path = "dspine_tests.rs"]
mod tests;
