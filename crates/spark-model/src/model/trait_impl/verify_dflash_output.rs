// SPDX-License-Identifier: AGPL-3.0-only

//! Final norm and LM-head dispatch for DFlash verification.
//!
//! The normal path keeps its wide/chunked kernels. Diagnostic controls can
//! instead run one row at a time through the exact K=1 target family, making
//! a parity-hash restoration identify the batched family at fault.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layers::ops;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DflashK1LmHeadProof {
    requested_rows: usize,
    engaged_rows: usize,
    vocab: usize,
}

impl DflashK1LmHeadProof {
    fn begin(
        requested_rows: usize,
        verify_vocab: usize,
        model_vocab: usize,
        fp32_logits: bool,
    ) -> Result<Self> {
        if requested_rows == 0 {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=requested_rows must be positive"
            );
        }
        if verify_vocab != model_vocab {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=full vocabulary required verify_vocab={verify_vocab} \
                 model_vocab={model_vocab}"
            );
        }
        if fp32_logits {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=BF16 logits required"
            );
        }
        Ok(Self {
            requested_rows,
            engaged_rows: 0,
            vocab: model_vocab,
        })
    }

    fn engage(&mut self) -> Result<()> {
        if self.engaged_rows == self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=engaged_rows exceeded requested_rows"
            );
        }
        self.engaged_rows += 1;
        Ok(())
    }

    fn finish(self) -> Result<Self> {
        if self.engaged_rows != self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=engaged_rows does not equal requested_rows \
                 requested_rows={} engaged_rows={}",
                self.requested_rows,
                self.engaged_rows
            );
        }
        Ok(self)
    }

    pub(super) fn proof_line(
        &self,
        family: &str,
        pre_verify_len: usize,
        tokens: &[u32],
    ) -> Result<String> {
        if tokens.len() != self.requested_rows {
            bail!(
                "DFLASH_K1_LM_HEAD_PATH_PROOF requested=true engaged=false \
                 requirement=token vector length does not equal requested_rows"
            );
        }
        Ok(format!(
            "DFLASH_K1_LM_HEAD_PATH_PROOF family=\"{family}\" \
             pre_verify_len={pre_verify_len} tokens={tokens:?} requested=true engaged=true \
             requested_rows={} engaged_rows={} full_vocab=true vocab={} dtype=\"bf16\"",
            self.requested_rows, self.engaged_rows, self.vocab
        ))
    }
}

impl TransformerModel {
    pub(super) fn dflash_k1_final_norm(
        &self,
        hidden: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let residual_elem = if self.config.use_fp32_residual() {
            4usize
        } else {
            bf16
        };
        let normed = self.buffers.norm_output();
        for row in 0..rows {
            self.final_norm_rows(
                hidden.offset(row * h * residual_elem),
                normed.offset(row * h * bf16),
                1,
                stream,
            )?;
        }
        Ok(normed)
    }

    /// Run each row through the ordinary one-token LM head and write its
    /// BF16 argmax to the shared scratch output `[rows]`.
    pub(super) fn dflash_k1_lm_head_argmax(
        &self,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DflashK1LmHeadProof)> {
        let mut proof = DflashK1LmHeadProof::begin(
            rows,
            self.verify_lmhead_vocab() as usize,
            self.config.vocab_size,
            self.use_fp32_logits,
        )?;

        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let argmax_out = self.buffers.scratch();
        for row in 0..rows {
            let logits = self.lm_head(normed.offset(row * h * bf16), stream)?;
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits,
                argmax_out.offset(row * 4),
                self.config.vocab_size as u32,
                stream,
            )?;
            proof.engage()?;
        }
        Ok((argmax_out, proof.finish()?))
    }

    /// Maximum rows per batched LM-head sweep: sized to the logits buffer
    /// capacity (128 when `ATLAS_DFLASH_BATCHED_VERIFY=1`, 32 default), capped
    /// at 128 so `C=8 K=8` (64 rows) and `C=16 K=8` (128 rows) sweep the LM
    /// head ONCE with ONE sync.
    pub(super) fn dflash_batched_lm_chunk_rows(&self) -> usize {
        let vocab = self.verify_lmhead_vocab() as usize;
        resolve_dflash_batched_lm_chunk(self.buffers.sizes().logits, vocab)
    }

    /// Launch 1-launch row-parallel `argmax_bf16_rows` (or per-row fallback)
    /// over `rows` contiguous BF16 logit rows starting at `self.buffers.logits()`.
    pub(super) fn dflash_argmax_logits_rows(
        &self,
        argmax_out: DevicePtr,
        rows: usize,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        if rows > 1 && self.argmax_rows_kernel.0 != 0 {
            ops::argmax_bf16_rows(
                self.gpu.as_ref(),
                self.argmax_rows_kernel,
                self.buffers.logits(),
                argmax_out,
                vocab as u32,
                vocab as u32,
                rows as u32,
                stream,
            )
        } else {
            for row in 0..rows {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(row * vocab * bf16),
                    argmax_out.offset(row * 4),
                    vocab as u32,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// Graph-capturable single-pass final norm + LM head + 1-launch argmax
    /// into `self.buffers.scratch()[0..total_rows]` (requires
    /// `total_rows <= dflash_batched_lm_chunk_rows()`, no sync or D2H).
    pub(super) fn dflash_batched_finalize_single_pass_gpu(
        &self,
        hidden: DevicePtr,
        total_rows: usize,
        stream: u64,
    ) -> Result<()> {
        let normed = self.buffers.norm_output();
        let vocab = self.verify_lmhead_vocab() as usize;
        self.final_norm_rows(hidden, normed, total_rows as u32, stream)?;
        self.lm_head_batched(normed, total_rows as u32, stream)?;
        self.dflash_argmax_logits_rows(self.buffers.scratch(), total_rows, vocab, stream)
    }

    pub(super) fn dflash_batched_finalize(
        &self,
        hidden: DevicePtr,
        total_rows: usize,
        serial_final_norm: bool,
        serial_lm_head: bool,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let normed = self.buffers.norm_output();

        if serial_final_norm {
            self.dflash_k1_final_norm(hidden, total_rows, stream)?;
        } else {
            self.final_norm_rows(hidden, normed, total_rows as u32, stream)?;
        }

        if serial_lm_head {
            let (argmax_out, _) = self.dflash_k1_lm_head_argmax(normed, total_rows, stream)?;
            self.gpu.synchronize(stream)?;
            return self.dflash_copy_argmax(argmax_out, total_rows);
        }

        // Chunk the batched LM head by the shared BF16 logits arena capacity
        // (up to 128 rows per chunk) when the serial diagnostic is disabled.
        anyhow::ensure!(
            !self.use_fp32_logits,
            "dflash_batched_finalize requires BF16 logits (use_fp32_logits=true unsupported)"
        );
        let vocab = self.verify_lmhead_vocab() as usize;
        let row_bytes = vocab
            .checked_mul(bf16)
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| {
                anyhow::anyhow!("dflash_batched_finalize: invalid vocab size {vocab}")
            })?;
        let logits_arena_bytes = self.buffers.sizes().logits;
        anyhow::ensure!(
            logits_arena_bytes >= row_bytes,
            "dflash_batched_finalize: logits arena ({logits_arena_bytes} B) smaller than one BF16 row ({row_bytes} B)"
        );
        let lm_chunk = self.dflash_batched_lm_chunk_rows();
        let mut all_argmax = vec![0u32; total_rows];
        let mut chunk_start = 0usize;
        while chunk_start < total_rows {
            let rows = (total_rows - chunk_start).min(lm_chunk);
            let normed_chunk = normed.offset(chunk_start * h * bf16);
            self.lm_head_batched(normed_chunk, rows as u32, stream)?;
            let argmax_out = self.buffers.scratch();
            self.dflash_argmax_logits_rows(argmax_out, rows, vocab, stream)?;
            self.gpu.synchronize(stream)?;
            let chunk = self.dflash_copy_argmax(argmax_out, rows)?;
            all_argmax[chunk_start..chunk_start + rows].copy_from_slice(&chunk);
            chunk_start += rows;
        }
        Ok(all_argmax)
    }

    pub(super) fn dflash_copy_argmax(
        &self,
        argmax_out: DevicePtr,
        rows: usize,
    ) -> Result<Vec<u32>> {
        let mut bytes = vec![0u8; rows * 4];
        self.gpu.copy_d2h(argmax_out, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

/// Pure chunk-row cap for `dflash_batched_finalize`: `logits_cap_rows.min(128)`,
/// floored at `1`.
pub(super) fn resolve_dflash_batched_lm_chunk(logits_bytes: usize, vocab: usize) -> usize {
    const MAX_LM_CHUNK: usize = 128;
    const BF16: usize = 2;
    let row_bytes = vocab.saturating_mul(BF16);
    if row_bytes == 0 {
        return 1;
    }
    (logits_bytes / row_bytes).clamp(1, MAX_LM_CHUNK)
}

#[cfg(test)]
#[path = "verify_dflash_output_tests.rs"]
mod tests;

#[cfg(test)]
mod lm_chunk_tests {
    use super::resolve_dflash_batched_lm_chunk;

    #[test]
    fn lm_chunk_scales_to_128_when_logits_arena_has_capacity() {
        let vocab = 248_320usize;
        let bf16 = 2usize;
        // Default 32-row logits buffer stays capped at 32.
        assert_eq!(
            resolve_dflash_batched_lm_chunk(32 * vocab * bf16, vocab),
            32
        );
        // Batched DFlash 128-row logits buffer sweeps C=8 K=8 (64) & C=16 K=8 (128) in 1 pass.
        assert_eq!(
            resolve_dflash_batched_lm_chunk(128 * vocab * bf16, vocab),
            128
        );
        // Larger buffers still cap at 128 (matching lm_head_batched 1..=128 contract).
        assert_eq!(
            resolve_dflash_batched_lm_chunk(256 * vocab * bf16, vocab),
            128
        );
        // Zero/degenerate safely floors at 1.
        assert_eq!(resolve_dflash_batched_lm_chunk(0, vocab), 1);
        assert_eq!(resolve_dflash_batched_lm_chunk(1024, 0), 1);
    }
}
