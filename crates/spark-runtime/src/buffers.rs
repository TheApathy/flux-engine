// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-allocated GPU buffer arena for intermediate tensors.
//!
//! All buffer sizes derive from [`ModelConfig`] (SSOT). The arena is
//! allocated once during initialization and reused across decode steps.

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;
use atlas_core::config::ModelConfig;

mod checked_math;
mod decode_meta;
mod sizes;
pub use decode_meta::{DECODE_META_MAX_ROWS, DecodeMetaLayout, padded_decode_rows};
pub use sizes::BufferSizes;

/// Pre-allocated GPU buffers for a single forward pass.
///
/// Each buffer is sized for `max_batch_tokens` tokens through the model.
/// Buffers are reused across steps — no per-step allocation.
///
/// Expert output buffers are sized for max(k_max, max_batch_tokens) to
/// support both speculative decode (K=3) and batched MoE prefill. At N=512,
/// this adds ~31 MB (vs the old grouped-GEMM approach that needed 260 MB
/// and caused a 15% decode regression). The GEMV-based prefill kernels
/// only touch k_max slots during decode, so the extra pages don't affect
/// decode bandwidth on unified memory.
pub struct BufferArena {
    /// Persistent hidden states: [M, residual_width] in BF16 (or FP32 for
    /// models that request it). `residual_width == hidden_size` except for
    /// multi-stream hyperconnection architectures such as Qwen4-Exp.
    hidden_states: DevicePtr,
    /// Residual scratch: [M, residual_width], matching `hidden_states`.
    residual: DevicePtr,
    /// Post-norm output: [M, hidden_size] in BF16.
    norm_output: DevicePtr,
    /// QKV projection output for full attention: `[M, (Hq * q_proj_mul + 2 * Hkv) * D]`
    /// in BF16 (where `q_proj_mul` is 2 for gated Q, else 1).
    qkv_output: DevicePtr,
    /// Attention output: [M, Hq * D] in BF16.
    attn_output: DevicePtr,
    /// MoE gate logits: [M, num_experts] in BF16.
    gate_logits: DevicePtr,
    /// MoE output: [M, hidden_size] in BF16.
    moe_output: DevicePtr,
    /// Logits: [M, vocab_size] in BF16.
    logits: DevicePtr,
    /// SSM QKVZ projection: [M, ssm_qkvz_size] in BF16.
    ssm_qkvz: DevicePtr,
    /// SSM beta-alpha projection: [M, ssm_ba_size] in BF16.
    ssm_ba: DevicePtr,
    /// SSM deinterleaved QKVZ: [M, ssm_qkvz_size] in BF16 (sequential [Q|K|V|Z]).
    ssm_deinterleaved: DevicePtr,
    /// SSM FP32 gates: `[mm, num_v_heads * 2]` as FP32 (gate + beta for GDN,
    /// where `mm = max_batch_tokens`).
    ssm_gates: DevicePtr,
    /// SSM conv1d output in FP32: [M, conv_dim] as FP32.
    /// Prevents BF16 truncation in the SSM recurrent path (conv → GDN).
    /// Without this, ~7 bits of precision are lost every token, causing
    /// coherence degradation after 8k+ tokens.
    ssm_conv_out_f32: DevicePtr,
    /// Shared exact-Qwen3.8 FlashInfer merged gate/up output in BF16.
    ffn_gate_up_bf16: DevicePtr,
    /// Scratch space for kernel metadata (positions, slot_mapping, block_tables).
    scratch: DevicePtr,
    /// Expert gate projection output: [k2 * top_k, moe_intermediate_size] BF16.
    expert_gate_out: DevicePtr,
    /// Expert up projection output: [k2 * top_k, moe_intermediate_size] BF16.
    expert_up_out: DevicePtr,
    /// Expert down projection output: [k2 * top_k, hidden_size] BF16.
    expert_down_out: DevicePtr,
    /// Persistent compact ordinary-NVFP4 MoE tile work-list.
    moe_worklist: DevicePtr,
    /// Device-written number of valid compact work-list items.
    moe_worklist_total: DevicePtr,
    /// Split-K decode attention workspace: partials from split CTAs (F32).
    splitk_workspace: DevicePtr,
    /// Flash-Next prefill gatepre gate scalars (`BufferSizes::moe_gate_scalars`);
    /// NULL when that size is zero.
    moe_gate_scalars: DevicePtr,
    /// Maximum batch tokens this arena was sized for.
    max_batch_tokens: usize,
    /// Fixed-stride decode metadata sized from the serving batch ceiling.
    decode_meta: DecodeMetaLayout,
    /// Sizes in bytes for each buffer (for debug/logging).
    sizes: BufferSizes,
}

impl BufferArena {
    /// Allocate all intermediate buffers on the GPU.
    pub fn new(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        max_batch_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_meta = DecodeMetaLayout::for_max_batch_size(max_batch_size);
        let sizes = BufferSizes::from_config(
            config,
            max_batch_tokens,
            max_seq_len,
            kv_block_size,
            max_batch_size,
        )?;
        let total_bytes = sizes.total_bytes()?;

        let hidden_states = gpu.alloc(sizes.hidden_states)?;
        let residual = gpu.alloc(sizes.residual)?;
        let norm_output = gpu.alloc(sizes.norm_output)?;
        let qkv_output = gpu.alloc(sizes.qkv_output)?;
        let attn_output = gpu.alloc(sizes.attn_output)?;
        let gate_logits = gpu.alloc(sizes.gate_logits)?;
        let moe_output = gpu.alloc(sizes.moe_output)?;
        let logits = gpu.alloc(sizes.logits)?;
        let ssm_qkvz = gpu.alloc(sizes.ssm_qkvz)?;
        let ssm_ba = gpu.alloc(sizes.ssm_ba)?;
        let ssm_deinterleaved = gpu.alloc(sizes.ssm_deinterleaved)?;
        let ssm_gates = gpu.alloc(sizes.ssm_gates)?;
        let ssm_conv_out_f32 = gpu.alloc(sizes.ssm_conv_out_f32)?;
        let ffn_gate_up_bf16 = gpu.alloc(sizes.ffn_gate_up_bf16)?;
        let scratch = gpu.alloc(sizes.scratch)?;
        let expert_gate_out = gpu.alloc(sizes.expert_gate_out)?;
        let expert_up_out = gpu.alloc(sizes.expert_up_out)?;
        let expert_down_out = gpu.alloc(sizes.expert_down_out)?;
        let moe_worklist = gpu.alloc(sizes.moe_worklist)?;
        let moe_worklist_total = gpu.alloc(sizes.moe_worklist_total)?;
        let splitk_workspace = gpu.alloc(sizes.splitk_workspace)?;
        let moe_gate_scalars = if sizes.moe_gate_scalars > 0 {
            gpu.alloc(sizes.moe_gate_scalars)?
        } else {
            DevicePtr::NULL
        };
        // ATLAS_PREFILL_SCRATCH_INIT=poison|arena-zero: fill every arena buffer once at load, so a
        // kernel reading rows it did not write in the same request is visible (poison) or deterministic.
        let mut initialised = 0usize;
        for (ptr, bytes) in [
            (hidden_states, sizes.hidden_states),
            (residual, sizes.residual),
            (norm_output, sizes.norm_output),
            (qkv_output, sizes.qkv_output),
            (attn_output, sizes.attn_output),
            (gate_logits, sizes.gate_logits),
            (moe_output, sizes.moe_output),
            (logits, sizes.logits),
            (ssm_qkvz, sizes.ssm_qkvz),
            (ssm_ba, sizes.ssm_ba),
            (ssm_deinterleaved, sizes.ssm_deinterleaved),
            (ssm_gates, sizes.ssm_gates),
            (ssm_conv_out_f32, sizes.ssm_conv_out_f32),
            (ffn_gate_up_bf16, sizes.ffn_gate_up_bf16),
            (scratch, sizes.scratch),
            (expert_gate_out, sizes.expert_gate_out),
            (expert_up_out, sizes.expert_up_out),
            (expert_down_out, sizes.expert_down_out),
            (moe_worklist, sizes.moe_worklist),
            (moe_worklist_total, sizes.moe_worklist_total),
            (splitk_workspace, sizes.splitk_workspace),
            (moe_gate_scalars, sizes.moe_gate_scalars),
        ] {
            if bytes > 0 && crate::scratch_init::init_arena(gpu, ptr, bytes)? {
                initialised += bytes;
            }
        }
        if initialised > 0 {
            tracing::info!(
                "PREFILL_SCRATCH_INIT selector={} site=arena bytes={initialised}",
                crate::scratch_init::SELECTOR
            );
        }

        tracing::info!(
            "Buffer arena: {} tokens × {:.1} MB total (attn_out={:.1}MB, ssm_deint={:.1}MB, kv_lora_rank={})",
            max_batch_tokens,
            total_bytes as f64 / (1024.0 * 1024.0),
            sizes.attn_output as f64 / (1024.0 * 1024.0),
            sizes.ssm_deinterleaved as f64 / (1024.0 * 1024.0),
            config.kv_lora_rank,
        );

        Ok(Self {
            hidden_states,
            residual,
            norm_output,
            qkv_output,
            attn_output,
            gate_logits,
            moe_output,
            logits,
            ssm_qkvz,
            ssm_ba,
            ssm_deinterleaved,
            ssm_gates,
            ssm_conv_out_f32,
            ffn_gate_up_bf16,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            moe_worklist,
            moe_worklist_total,
            splitk_workspace,
            moe_gate_scalars,
            max_batch_tokens,
            decode_meta,
            sizes,
        })
    }

    pub fn hidden_states(&self) -> DevicePtr {
        self.hidden_states
    }
    pub fn residual(&self) -> DevicePtr {
        self.residual
    }
    pub fn norm_output(&self) -> DevicePtr {
        self.norm_output
    }
    pub fn qkv_output(&self) -> DevicePtr {
        self.qkv_output
    }
    pub fn attn_output(&self) -> DevicePtr {
        self.attn_output
    }
    pub fn gate_logits(&self) -> DevicePtr {
        self.gate_logits
    }
    pub fn moe_output(&self) -> DevicePtr {
        self.moe_output
    }
    pub fn logits(&self) -> DevicePtr {
        self.logits
    }
    pub fn ssm_qkvz(&self) -> DevicePtr {
        self.ssm_qkvz
    }
    pub fn ssm_ba(&self) -> DevicePtr {
        self.ssm_ba
    }
    /// Sequential [Q|K|V|Z] after deinterleaving.
    pub fn ssm_deinterleaved(&self) -> DevicePtr {
        self.ssm_deinterleaved
    }
    /// FP32 `[gate, beta]` for GDN (`[mm, num_v_heads * 2]` floats, where `mm = max_batch_tokens`).
    pub fn ssm_gates(&self) -> DevicePtr {
        self.ssm_gates
    }
    /// FP32 conv1d output for SSM recurrent path (prevents BF16 precision drift).
    pub fn ssm_conv_out_f32(&self) -> DevicePtr {
        self.ssm_conv_out_f32
    }
    /// Shared row-major BF16 output for exact merged gate/up prefill.
    pub fn ffn_gate_up_bf16(&self) -> DevicePtr {
        self.ffn_gate_up_bf16
    }
    /// Scratch buffer for MoE routing + kernel metadata uploads.
    pub fn scratch(&self) -> DevicePtr {
        self.scratch
    }
    /// Batched expert gate projection output.
    pub fn expert_gate_out(&self) -> DevicePtr {
        self.expert_gate_out
    }
    /// Batched expert up projection output.
    pub fn expert_up_out(&self) -> DevicePtr {
        self.expert_up_out
    }
    /// Batched expert down projection output.
    pub fn expert_down_out(&self) -> DevicePtr {
        self.expert_down_out
    }
    /// Persistent compact ordinary-NVFP4 MoE tile work-list.
    pub fn moe_worklist(&self) -> DevicePtr {
        self.moe_worklist
    }
    /// Device-written number of valid compact work-list items.
    pub fn moe_worklist_total(&self) -> DevicePtr {
        self.moe_worklist_total
    }
    /// Split-K decode attention workspace (F32 partials).
    pub fn splitk_workspace(&self) -> DevicePtr {
        self.splitk_workspace
    }
    pub fn moe_gate_scalars(&self) -> DevicePtr {
        self.moe_gate_scalars
    }
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }
    pub fn decode_meta(&self) -> DecodeMetaLayout {
        self.decode_meta
    }
    pub fn sizes(&self) -> &BufferSizes {
        &self.sizes
    }

    /// Zero only the 7 buffers that can carry residual state between requests:
    /// `hidden_states`, `residual`, `gate_logits`, `expert_gate_out`,
    /// `expert_up_out`, `expert_down_out`, and `moe_output`.
    ///
    /// During prefill, the remaining 12 buffers in [`Self::ZERO_ALL_BUFFERS`] are
    /// overwritten before being read within the layer loop:
    /// - `norm_output`, `qkv_output`, `attn_output`: written by each layer's projection
    /// - `ssm_*` (`ssm_qkvz`, `ssm_ba`, `ssm_deinterleaved`, `ssm_gates`, `ssm_conv_out_f32`): written by SSM projection
    /// - `ffn_gate_up_bf16`: written by merged dense FFN projection
    /// - `logits`: written by LM head on last token
    /// - `scratch`: overwritten by metadata upload and MoE routing
    /// - `splitk_workspace`: written by attention kernel
    ///
    /// This reduces per-chunk memsets from 19 calls (`zero_all`) to 7, saving
    /// 12 memset launches × bandwidth on the LPDDR5X bus per prefill chunk.
    pub fn zero_prefill_essentials(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        gpu.memset_async(self.hidden_states, 0, self.sizes.hidden_states, stream)?;
        gpu.memset_async(self.residual, 0, self.sizes.residual, stream)?;
        // MoE buffers: gate_logits may carry stale expert indices from a prior
        // request with different token count, causing out-of-bounds expert access
        // (CUDA error 700 at layer 38+ on 122B). Zero to prevent.
        gpu.memset_async(self.gate_logits, 0, self.sizes.gate_logits, stream)?;
        gpu.memset_async(self.expert_gate_out, 0, self.sizes.expert_gate_out, stream)?;
        gpu.memset_async(self.expert_up_out, 0, self.sizes.expert_up_out, stream)?;
        gpu.memset_async(self.expert_down_out, 0, self.sizes.expert_down_out, stream)?;
        gpu.memset_async(self.moe_output, 0, self.sizes.moe_output, stream)?;
        Ok(())
    }

    /// The buffers [`Self::zero_all`] clears, in its order: the names
    /// `ATLAS_PREFILL_ZERO_ALL` accepts (K8 item A).
    pub const ZERO_ALL_BUFFERS: [&str; 19] = [
        "hidden_states",
        "residual",
        "norm_output",
        "qkv_output",
        "attn_output",
        "gate_logits",
        "moe_output",
        "ssm_qkvz",
        "ssm_ba",
        "ssm_deinterleaved",
        "ssm_gates",
        "ssm_conv_out_f32",
        "ffn_gate_up_bf16",
        "splitk_workspace",
        "expert_gate_out",
        "expert_up_out",
        "expert_down_out",
        "logits",
        "scratch",
    ];

    fn zero_all_spans(&self) -> [(DevicePtr, usize); 19] {
        let s = &self.sizes;
        [
            (self.hidden_states, s.hidden_states),
            (self.residual, s.residual),
            (self.norm_output, s.norm_output),
            (self.qkv_output, s.qkv_output),
            (self.attn_output, s.attn_output),
            (self.gate_logits, s.gate_logits),
            (self.moe_output, s.moe_output),
            (self.ssm_qkvz, s.ssm_qkvz),
            (self.ssm_ba, s.ssm_ba),
            (self.ssm_deinterleaved, s.ssm_deinterleaved),
            (self.ssm_gates, s.ssm_gates),
            (self.ssm_conv_out_f32, s.ssm_conv_out_f32),
            (self.ffn_gate_up_bf16, s.ffn_gate_up_bf16),
            (self.splitk_workspace, s.splitk_workspace),
            (self.expert_gate_out, s.expert_gate_out),
            (self.expert_up_out, s.expert_up_out),
            (self.expert_down_out, s.expert_down_out),
            (self.logits, s.logits),
            (self.scratch, s.scratch),
        ]
    }

    /// [`Self::zero_all`] with a per-buffer fill: `fill[i]` is the byte written over
    /// `ZERO_ALL_BUFFERS[i]`, `None` leaves it untouched; `[Some(0); 19]` issues exactly
    /// `zero_all`'s memsets. Returns (bytes written, bytes left untouched). Measurement
    /// only (`ATLAS_PREFILL_ZERO_ALL`, K8 item A): a non-zero fill or a skip is not
    /// known to be exact.
    pub fn zero_all_with(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        fill: &[Option<u8>; 19],
    ) -> anyhow::Result<(usize, usize)> {
        let mut written = 0usize;
        let mut untouched = 0usize;
        for ((ptr, bytes), value) in self.zero_all_spans().into_iter().zip(fill.iter()) {
            match value {
                Some(v) => {
                    gpu.memset_async(ptr, *v, bytes, stream)?;
                    written += bytes;
                }
                None => untouched += bytes,
            }
        }
        Ok((written, untouched))
    }

    /// Zero all reusable buffers to eliminate stale data between requests.
    /// Ensures deterministic computation regardless of request history.
    pub fn zero_all(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        gpu.memset_async(self.hidden_states, 0, self.sizes.hidden_states, stream)?;
        gpu.memset_async(self.residual, 0, self.sizes.residual, stream)?;
        gpu.memset_async(self.norm_output, 0, self.sizes.norm_output, stream)?;
        gpu.memset_async(self.qkv_output, 0, self.sizes.qkv_output, stream)?;
        gpu.memset_async(self.attn_output, 0, self.sizes.attn_output, stream)?;
        gpu.memset_async(self.gate_logits, 0, self.sizes.gate_logits, stream)?;
        gpu.memset_async(self.moe_output, 0, self.sizes.moe_output, stream)?;
        gpu.memset_async(self.ssm_qkvz, 0, self.sizes.ssm_qkvz, stream)?;
        gpu.memset_async(self.ssm_ba, 0, self.sizes.ssm_ba, stream)?;
        gpu.memset_async(
            self.ssm_deinterleaved,
            0,
            self.sizes.ssm_deinterleaved,
            stream,
        )?;
        gpu.memset_async(self.ssm_gates, 0, self.sizes.ssm_gates, stream)?;
        gpu.memset_async(
            self.ssm_conv_out_f32,
            0,
            self.sizes.ssm_conv_out_f32,
            stream,
        )?;
        gpu.memset_async(
            self.ffn_gate_up_bf16,
            0,
            self.sizes.ffn_gate_up_bf16,
            stream,
        )?;
        gpu.memset_async(
            self.splitk_workspace,
            0,
            self.sizes.splitk_workspace,
            stream,
        )?;
        gpu.memset_async(self.expert_gate_out, 0, self.sizes.expert_gate_out, stream)?;
        gpu.memset_async(self.expert_up_out, 0, self.sizes.expert_up_out, stream)?;
        gpu.memset_async(self.expert_down_out, 0, self.sizes.expert_down_out, stream)?;
        gpu.memset_async(self.logits, 0, self.sizes.logits, stream)?;
        gpu.memset_async(self.scratch, 0, self.sizes.scratch, stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
