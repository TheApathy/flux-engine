// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Final norm + LM head pass for `mixed_forward_dispatch`.
//!
//! Hoisted from `decode_b.rs` to keep that file under the 500 LoC cap.
//! Single helper `mixed_final_norm_lm_head` runs the post-layer-loop
//! reductions: RMS norm + per-position LM-head GEMV for the decode batch,
//! plus a single GEMV for the prefill last-token logits when `is_last`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layers::ops;

/// Returns `(decode_logits, prefill_logits)`. `prefill_logits` is
/// `DevicePtr::NULL` when `prefill_is_last == false`.
pub(super) struct MixedHeadOut {
    pub decode_logits: DevicePtr,
    pub prefill_logits: DevicePtr,
}

impl TransformerModel {
    pub(super) fn mixed_final_norm_lm_head(
        &self,
        hidden: DevicePtr,
        prefill_hidden: DevicePtr,
        padded_n: usize,
        proc_count: usize,
        prefill_is_last: bool,
        h: usize,
        bf16: usize,
        fp32: usize,
        stream: u64,
    ) -> Result<MixedHeadOut> {
        // ── 7. Final norm + LM head ──
        //
        // Decode: RMS norm on N tokens → N logits
        // Prefill: RMS norm on last token only (if is_last) → 1 logit
        let normed = self.buffers.norm_output();

        // 7a. Decode logits: norm [padded_n, H] → GEMV × padded_n
        self.final_norm_rows(hidden, normed, padded_n as u32, stream)?;

        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            if padded_n > 1
                && matches!(
                    self.w4a16_exact_lm_head_kernels
                        .route_for_rows(padded_n as u32),
                    Some(ops::ExactLmHeadRoute::Exact(_))
                )
            {
                ops::w4a16_gemv_batch_logits_exact(
                    self.gpu.as_ref(),
                    self.w4a16_exact_lm_head_kernels,
                    normed,
                    nvfp4,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
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
            }
        } else {
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
        }
        let decode_logits = logits;

        // 7b. Prefill logits: norm last token → 1 GEMV (if is_last)
        let prefill_logits = if prefill_is_last {
            anyhow::ensure!(
                proc_count > 0,
                "mixed_final_norm_lm_head: prefill_is_last requires proc_count > 0"
            );
            let last_hidden = prefill_hidden.offset((proc_count - 1) * h * fp32);
            // Place prefill normed output after decode normed to avoid overlap
            let prefill_normed = normed.offset(padded_n * h * bf16);
            self.final_norm_rows(last_hidden, prefill_normed, 1, stream)?;

            // Place prefill logits after decode logits
            let prefill_logits_ptr = logits.offset(padded_n * v * bf16);
            if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_decode_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    self.w4a16_gemv_sw_kernel,
                    self.gemv_sw,
                    prefill_normed,
                    nvfp4,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    prefill_normed,
                    &self.lm_head_weight,
                    prefill_logits_ptr,
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
            prefill_logits_ptr
        } else {
            DevicePtr::NULL
        };

        Ok(MixedHeadOut {
            decode_logits,
            prefill_logits,
        })
    }
}
