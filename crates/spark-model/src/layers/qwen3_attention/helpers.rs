// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` setters and small per-layer compute helpers
//! (`apply_layer_scalar`, `effective_attn_scale`).

use super::types::{MlaWeights, Qwen3AttentionLayer};
use crate::layers::FfnComponent;
use crate::weight_map::DenseWeight;

impl Qwen3AttentionLayer {
    pub fn set_qwen4_hyperconnections(
        &mut self,
        attn: crate::layers::Qwen4HyperConnection,
        mlp: crate::layers::Qwen4HyperConnection,
    ) {
        self.qwen4_attn_hyper = Some(attn);
        self.qwen4_mlp_hyper = Some(mlp);
    }

    pub fn set_qwen4_qsa(&mut self, qsa: crate::layers::Qwen4QsaIndexer) {
        self.qwen4_qsa = Some(qsa);
    }

    /// Set MLA weights for 2-step latent decode. When set, decode uses
    /// latent→norm→expand instead of single-step GEMV.
    pub fn set_mla_weights(&mut self, mla: MlaWeights) {
        self.mla = Some(mla);
    }

    /// Set per-layer dimension overrides for heterogeneous models (Gemma-4).
    /// Full-attention layers have different Q/KV head counts and head_dim
    /// than sliding layers.
    pub fn set_dimension_overrides(
        &mut self,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) {
        self.head_dim_override = Some(head_dim);
        self.num_q_heads_override = Some(num_q_heads);
        self.num_kv_heads_override = Some(num_kv_heads);
    }

    /// Set per-layer sliding-window size (Gemma-4 hybrid attention).
    /// Call with `Some(window_size)` on sliding layers, `None` on
    /// full-attention layers. Non-Gemma-4 models never call this.
    pub fn set_sliding_window(&mut self, window: Option<u32>) {
        self.sliding_window = window;
    }

    /// Set per-layer RoPE overrides (theta, rotary_dim) for dual-RoPE
    /// models (Gemma-4).
    pub fn set_rope_overrides(&mut self, theta: f32, rotary_dim: u32) {
        self.rope_theta_override = Some(theta);
        self.rotary_dim_override = Some(rotary_dim);
    }

    /// Enable proportional RoPE (Gemma-4 full-attention layers). Must be
    /// called AFTER `set_rope_overrides`; the `rotary_dim` set there is
    /// reinterpreted as the number of non-zero rotation pairs.
    pub fn set_rope_proportional(&mut self, enable: bool) {
        self.rope_proportional = enable;
    }

    /// Set per-layer attention scale override. Gemma-4 uses QK-norm, so
    /// attention scale should be 1.0 (not 1/sqrt(head_dim)).
    pub fn set_attn_scale_override(&mut self, scale: f32) {
        self.attn_scale_override = Some(scale);
    }

    /// Set K=V mode (Gemma-4 full-attention layers).
    ///
    /// `v_norm_weight` is a BF16 weight buffer of size `[head_dim]`. For
    /// Gemma-4 it's ones-filled because Gemma-4's rms_norm kernel uses
    /// the absolute convention `out = x * rms * weight`, and `weight =
    /// 1.0` gives pure RMSNorm (matching HF
    /// `Gemma4RMSNorm(with_scale=False)`).
    pub fn set_k_eq_v(&mut self, v_norm_weight: DenseWeight) {
        self.k_eq_v = true;
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// Install a pure-RMSNorm v_norm WITHOUT enabling K=V aliasing. Used
    /// for Gemma-4 sliding-attention layers where V_proj exists on disk
    /// but HF `Gemma4TextAttention.forward()` still applies
    /// `value_states = self.v_norm(value_states)` with
    /// `Gemma4RMSNorm(with_scale=False)` — pure `x * rms`.
    pub fn set_v_norm(&mut self, v_norm_weight: DenseWeight) {
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// Install a BF16 dense fallback for the output projection. When
    /// set, decode + prefill skip the NVFP4 `attn.o_proj` path and use
    /// this BF16 dense_gemv / dense_gemm instead. Required for Gemma-4
    /// dense (Nvidia ModelOpt's official ignore list keeps ALL
    /// self_attn projections at BF16).
    pub fn set_o_dense_bf16(&mut self, o_dense: DenseWeight) {
        self.o_dense_bf16 = Some(o_dense);
    }

    /// `ATLAS_QWEN4_ATTN_BF16=1`: install the decode-only BF16 Q/K/V/O
    /// fallbacks. Separate from `set_o_dense_bf16` / the NVFP4
    /// `q_weight`/`k_weight`/`v_weight`/`o_weight` fields — see the
    /// field docs in `types.rs` for why prefill must not read these.
    pub(crate) fn set_qkvo_dense_bf16_decode_only(
        &mut self,
        q: DenseWeight,
        k: DenseWeight,
        v: DenseWeight,
        o: DenseWeight,
    ) {
        self.q_dense_bf16_decode = Some(q);
        self.k_dense_bf16_decode = Some(k);
        self.v_dense_bf16_decode = Some(v);
        self.o_dense_bf16_decode = Some(o);
    }

    /// Set post-sublayer norms (Gemma-4: 4-norm residual structure).
    pub fn set_post_sublayer_norms(
        &mut self,
        post_attn_out: DenseWeight,
        post_ffn_out: DenseWeight,
    ) {
        self.post_attn_out_norm = Some(post_attn_out);
        self.post_ffn_out_norm = Some(post_ffn_out);
    }

    /// Set per-layer scalar (Gemma-4: hidden_states *= scalar at end of
    /// layer).
    pub fn set_layer_scalar(&mut self, scalar: f32) {
        self.layer_scalar = Some(scalar);
    }

    /// Set secondary MoE FFN (Gemma-4 26B dual-FFN: dense + MoE per
    /// layer).
    pub fn set_moe_ffn(
        &mut self,
        ffn: FfnComponent,
        pre_norm: DenseWeight,
        post_norm: DenseWeight,
        post_dense_norm: DenseWeight,
    ) {
        self.moe_ffn = Some(ffn);
        self.pre_moe_norm = Some(pre_norm);
        self.post_moe_out_norm = Some(post_norm);
        self.post_dense_ffn_norm = Some(post_dense_norm);
    }

    /// Apply layer_scalar in-place: `hidden *= scalar`. Uses
    /// `bf16_scale_inplace` for BF16 buffers, or `f32_scale_inplace`
    /// when the FP32 residual path is active (Gemma-4). Gemma-4's
    /// layer_scalar values are small (L0=0.09, L1=0.065) so the BF16
    /// variant compounds underflow across layers — the FP32 path is a
    /// correctness requirement for 31B.
    pub(crate) fn apply_layer_scalar(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        hidden: spark_runtime::gpu::DevicePtr,
        hidden_size: usize,
        scalar: f32,
        stream: u64,
        fp32_residual: bool,
    ) -> anyhow::Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let scale_k = if fp32_residual {
            gpu.kernel("embed_scale", "f32_scale_inplace")?
        } else {
            gpu.kernel("embed_scale", "bf16_scale_inplace")?
        };
        let n = hidden_size as u32;
        KernelLaunch::new(gpu, scale_k)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden)
            .arg_u32(n)
            .arg_f32(scalar)
            .launch(stream)
    }

    /// Compute effective attention scale: override if set, else
    /// `1/sqrt(head_dim)`.
    pub(crate) fn effective_attn_scale(&self, head_dim: u32) -> f32 {
        self.attn_scale_override
            .unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt())
    }

    /// Apply TurboQuant WHT(Q) and post-WHT InnerQ `scale_inv` to contiguous
    /// `[total_q_heads, hd]` BF16 queries when K is stored in a rotated Turbo format.
    ///
    /// Both `wht_bf16_k` and `innerq_apply_q_k` run in the same post-WHT basis as
    /// `write_kv_cache` (`wht_bf16_k` followed by `innerq_apply_k_k` on K), preserving
    /// `<S^-1(H q), S(H k)> = <H q, H k> = <q, k>`. Both are bypassed when
    /// `TQ_PLUS_WEIGHT_ROTATION=1` (weights already pre-rotated into WHT space).
    pub(in crate::layers::qwen3_attention) fn apply_turbo_q_wht_and_innerq(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        q_ptr: spark_runtime::gpu::DevicePtr,
        total_q_heads: u32,
        hd: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        let (k_dtype, _) = self.kv_dtype.kv_pair();
        if !kv_dtype_is_turbo(k_dtype)
            || tq_plus_weight_rotation_enabled()
            || self.wht_bf16_k.0 == 0
            || !(hd == 128 || hd == 256 || hd == 512)
        {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        KernelLaunch::new(gpu, self.wht_bf16_k)
            .grid([total_q_heads, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(q_ptr)
            .arg_u32(hd)
            .launch(stream)?;
        // InnerQ post-WHT scale_inv on Q (no-op when d_innerq_active == 0 on device).
        if self.innerq_apply_q_k.0 != 0 && hd == 128 {
            KernelLaunch::new(gpu, self.innerq_apply_q_k)
                .grid([total_q_heads, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_ptr)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(())
    }

    /// Whether attention outputs need inverse WHT (`iWHT`) to leave the rotated-V basis.
    pub(in crate::layers::qwen3_attention) fn needs_turbo_out_iwht(&self, hd: u32) -> bool {
        let (_, v_dtype) = self.kv_dtype.kv_pair();
        kv_dtype_is_turbo(v_dtype)
            && !tq_plus_weight_rotation_enabled()
            && self.wht_bf16_k_inv.0 != 0
            && (hd == 128 || hd == 256 || hd == 512)
    }

    /// Apply inverse WHT (`iWHT`) to contiguous `[total_q_heads, hd]` BF16 attention
    /// outputs when V is stored in a rotated Turbo format.
    pub(in crate::layers::qwen3_attention) fn apply_turbo_out_iwht(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        attn_out: spark_runtime::gpu::DevicePtr,
        total_q_heads: u32,
        hd: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        if !self.needs_turbo_out_iwht(hd) {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        KernelLaunch::new(gpu, self.wht_bf16_k_inv)
            .grid([total_q_heads, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(attn_out)
            .arg_u32(hd)
            .launch(stream)
    }
}

/// Whether a single KV side (`K` or `V` from `KvCacheDtype::kv_pair()`) uses
/// Walsh-Hadamard-rotated TurboQuant storage.
pub(in crate::layers::qwen3_attention) fn kv_dtype_is_turbo(
    dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> bool {
    matches!(
        dtype,
        spark_runtime::kv_cache::KvCacheDtype::Turbo2
            | spark_runtime::kv_cache::KvCacheDtype::Turbo3
            | spark_runtime::kv_cache::KvCacheDtype::Turbo4
            | spark_runtime::kv_cache::KvCacheDtype::Turbo8
    )
}

/// Process-cached check for `TQ_PLUS_WEIGHT_ROTATION=1|true` (weights pre-rotated
/// into WHT space at load time, bypassing runtime WHT/iWHT/InnerQ bookends).
pub(in crate::layers::qwen3_attention) fn tq_plus_weight_rotation_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}
