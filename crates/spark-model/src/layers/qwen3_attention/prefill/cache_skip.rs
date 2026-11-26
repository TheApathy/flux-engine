// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_attention_with_cache_skip` — prefix-cache-aware prefill.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

use super::{
    Qwen38PrefillAttnGateRoute, Qwen38QkNormRopeRoute, mark_qwen38_prefill_attn_gate_engaged,
    mark_qwen38_qknorm_rope_cache_skip_engaged, qwen38_prefill_attn_gate_requested,
    qwen38_prefill_attn_gate_route, qwen38_qknorm_rope_route,
};

const PREFILL_ATTN_FI_SM_THRESHOLD: u32 = 64;
const PREFILL_ATTN_FI_MIN_TOKENS: u32 = 64;
const PREFILL_QKNORM_ROPE_SM_THRESHOLD: u32 = 64;

/// Parse explicit `ATLAS_PREFILL_QKNORM_ROPE`:
/// - absent -> `Ok(None)` (auto-policy decides by SM count)
/// - `"0"` -> `Ok(Some(false))` (explicit opt-out even on high-SM GPUs)
/// - `"1"` -> `Ok(Some(true))` (explicit fail-closed request)
/// - anything else -> `Err(...)`
pub(super) fn parse_cache_skip_qknorm_rope_explicit(
    raw: Option<&str>,
) -> std::result::Result<Option<bool>, &'static str> {
    match raw {
        None => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(_) => Err("ATLAS_PREFILL_QKNORM_ROPE must be exactly 0 or 1"),
    }
}

/// Resolve whether cache-skip prefill should request the fused QK-norm + RoPE
/// kernel:
/// - Explicit `"1"` always requests (and fails closed if YaRN or missing).
/// - Explicit `"0"` always disables.
/// - Unset auto-enables on wide-SM GPUs (`sm_count > 64`) when YaRN is absent
///   and the kernel handles are loaded, while staying default-OFF on GB10 (`<= 64`).
pub(super) const fn resolve_cache_skip_qknorm_rope(
    explicit: Option<bool>,
    sm_count: u32,
    no_yarn: bool,
    kernels_ready: bool,
) -> bool {
    match explicit {
        Some(flag) => flag,
        None => sm_count > PREFILL_QKNORM_ROPE_SM_THRESHOLD && no_yarn && kernels_ready,
    }
}

fn cache_skip_qknorm_rope_explicit() -> Result<Option<bool>> {
    static GATE: std::sync::OnceLock<std::result::Result<Option<bool>, &'static str>> =
        std::sync::OnceLock::new();
    (*GATE.get_or_init(|| match std::env::var("ATLAS_PREFILL_QKNORM_ROPE") {
        Ok(value) => parse_cache_skip_qknorm_rope_explicit(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_cache_skip_qknorm_rope_explicit(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_PREFILL_QKNORM_ROPE must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

/// Resolve `ATLAS_PREFILL_ATTN_FI`: explicit `"1"`/`"true"` enables,
/// explicit `"0"`/`"false"` disables, and unset auto-enables on discrete
/// high-SM GPUs (`sm_count > 64`) while staying default-OFF on GB10 (`<= 64`).
pub(super) fn resolve_prefill_attn_fi(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(_) => false,
        None => sm_count > PREFILL_ATTN_FI_SM_THRESHOLD,
    }
}

pub(super) fn prefill_attn_fi_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let raw = std::env::var("ATLAS_PREFILL_ATTN_FI").ok();
        resolve_prefill_attn_fi(raw.as_deref(), atlas_core::device::active_sm_count())
    })
}

/// Eligibility predicate for FlashInfer FA2 ragged prefill (`qwen4_fi_attn`):
/// requires `atlas_qwen4_fi_attn` compiled in, `ATLAS_PREFILL_ATTN_FI` enabled,
/// single-sequence non-MLA causal attention with `head_dim == 256`, no sliding
/// window, GQA-aligned head counts, and `n >= 64`.
#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every FlashInfer FA2 eligibility check"
)]
pub(super) const fn prefill_attn_fi_eligible(
    compiled_fi_attn: bool,
    enabled: bool,
    single_seq: bool,
    no_mla: bool,
    sliding_window: u32,
    n: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
) -> bool {
    compiled_fi_attn
        && enabled
        && single_seq
        && no_mla
        && sliding_window == 0
        && n >= PREFILL_ATTN_FI_MIN_TOKENS
        && hd == 256
        && nq > 0
        && nkv > 0
        && nq.is_multiple_of(nkv)
}

impl Qwen3AttentionLayer {
    /// Prefill attention with optional KV cache write skip for prefix caching.
    ///
    /// `kv_write_start`: number of token positions whose KV entries are already
    /// in the cache. `reshape_and_cache` is only called for positions >= this value.
    #[allow(unreachable_code, unused_variables, unused_assignments)]
    pub(in crate::layers::qwen3_attention) fn prefill_attention_with_cache_skip(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        kv_write_start: usize,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;
        let meta = ctx
            .attn_metadata
            .expect("attention prefill requires metadata");
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && ctx.config.hidden_size == 5120
            && ctx.config.tp_world_size.max(1) == 1
            && ctx.config.mrope_section == [11, 11, 10];
        let qknorm_rope_explicit = cache_skip_qknorm_rope_explicit()?;
        if self.yarn.is_some() && qknorm_rope_explicit == Some(true) {
            anyhow::bail!(
                "ATLAS_PREFILL_QKNORM_ROPE=1 is incompatible with YaRN; \
                 use rope_forward_mrope_interleaved_yarn"
            );
        }
        let qknorm_rope_requested = resolve_cache_skip_qknorm_rope(
            qknorm_rope_explicit,
            atlas_core::device::active_sm_count(),
            self.yarn.is_none(),
            self.rope_k.0 != 0 && self.qwen38_prefill_qknorm_rope_k.0 != 0,
        );
        // Parse and validate the complete route before Q/K/V projection or KV
        // cache mutation. An explicit malformed request or stale bundle must
        // be transactional: it cannot leave a partially populated cache.
        let gate_fused_route = qwen38_prefill_attn_gate_route(
            qwen38_prefill_attn_gate_requested()?,
            exact_qwen38_dense,
            meta.num_seqs == 1,
            self.mla.is_none(),
            self.gated,
            n,
            nq,
            nkv,
            hd,
            q_proj_dim as u32,
            self.prefill_attn_64_k.0 != 0,
            self.sigmoid_gate_mul_batched_k.0 != 0,
            self.prefill_attn_64_gate_fused_k.0 != 0,
        );
        if gate_fused_route == Qwen38PrefillAttnGateRoute::Missing {
            anyhow::bail!(
                "ATLAS_PREFILL_ATTN_GATE_FUSED=1 requires the BR64 parent, batched gate parent, and inferspark_prefill_64_gate_fused"
            );
        }

        // Pre-declare output buffers (used by both MLA and standard paths)
        let _qg_out = ctx.buffers.qkv_output();
        let _k_contiguous = ctx.buffers.ssm_qkvz();
        let _v_contiguous = ctx.buffers.attn_output();

        // Profiling helper
        macro_rules! aprof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  ATTN prefill [{}] N={}: {}µs", $label, n, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 0. Convert activations BF16 → FP8 once for all Q/K/V projections ──
        // FP8×FP8 GEMM is ~10% faster than BF16×FP8 for Q proj, more than
        // compensating for the 7.6ms conversion cost.
        let use_fp8_act = self.q_fp8.is_some();
        let normed_fp8 = if use_fp8_act {
            let act_fp8 = ctx.buffers.attn_output();
            ops::bf16_to_fp8(ctx.gpu, self.bf16_to_fp8_k, normed, act_fp8, n * h, stream)?;
            act_fp8
        } else {
            DevicePtr::NULL
        };
        aprof!("bf16_to_fp8", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── MLA 2-step prefill ── (extracted to cache_skip_mla.rs)
        if self.mla.is_some() {
            let args = super::cache_skip_mla::CacheSkipMlaArgs {
                normed,
                n,
                h,
                nq,
                hd,
                eps,
                stream,
                kv_write_start,
            };
            return self.prefill_attention_cache_skip_mla(kv_cache, ctx, &args);
        }

        // ── Standard Q/K/V projection (non-MLA models) ──
        if self.mla.is_none() {
            self.prefill_attention_cache_skip_qkv(
                normed, normed_fp8, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx,
                stream,
            )?;
        } // end if self.mla.is_none() (standard projection path)

        // ── 4+5. Deinterleave Q/Gate + per-head Q/K RMS norms ──
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let q_contiguous = ctx.buffers.ssm_deinterleaved();
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let rope_theta = self
            .rope_theta_override
            .unwrap_or(ctx.config.rope_theta as f32);
        let qknorm_rope_route = qwen38_qknorm_rope_route(
            qknorm_rope_requested,
            exact_qwen38_dense && self.yarn.is_none(),
            meta.num_seqs == 1,
            self.mla.is_none(),
            self.gated,
            !self.attn.q_norm.weight.is_null(),
            !self.attn.k_norm.weight.is_null(),
            self.attn.q_norm_full.is_none() && self.attn.k_norm_full.is_none(),
            self.v_norm_weight.is_none(),
            self.mrope_interleaved,
            self.rope_proportional,
            n,
            nq,
            nkv,
            hd,
            q_proj_dim as u32,
            rotary_dim,
            rope_theta.is_finite() && rope_theta > 0.0,
            self.rope_mrope_interleaved_k.0 != 0,
            self.qwen38_prefill_qknorm_rope_k.0 != 0,
        );
        let qknorm_rope_fused = match qknorm_rope_route {
            Qwen38QkNormRopeRoute::Complete => {
                // Pass all 3D MRoPE position axes (T, H, W) matching the
                // interleaved MRoPE parent path below.
                ops::qwen38_prefill_qknorm_rope(
                    ctx.gpu,
                    self.qwen38_prefill_qknorm_rope_k,
                    qg_out,
                    q_contiguous,
                    k_contiguous,
                    self.attn.q_norm.weight,
                    self.attn.k_norm.weight,
                    meta.positions,
                    meta.positions_h,
                    meta.positions_w,
                    n,
                    nq,
                    nkv,
                    hd,
                    q_proj_dim as u32,
                    rotary_dim,
                    eps,
                    rope_theta,
                    stream,
                )?;
                mark_qwen38_qknorm_rope_cache_skip_engaged();
                true
            }
            Qwen38QkNormRopeRoute::Missing => {
                anyhow::bail!(
                    "ATLAS_PREFILL_QKNORM_ROPE=1 selected eligible Qwen3.8 cache-skip prefill, \
                     but the interleaved-MRoPE parent or qwen38_prefill_qknorm_rope is missing; \
                     rebuild the kernel bundle before measuring"
                )
            }
            Qwen38QkNormRopeRoute::Disabled | Qwen38QkNormRopeRoute::Ineligible => false,
        };
        if !qknorm_rope_fused {
            if self.gated && !self.attn.q_norm.weight.is_null() {
                // Fused deinterleave + Q norm: eliminates Q global memory round-trip
                ops::deinterleave_qg_split_qnorm(
                    ctx.gpu,
                    self.deinterleave_qg_split_qnorm_k,
                    qg_out,
                    q_contiguous,
                    self.attn.q_norm.weight,
                    n,
                    nq,
                    hd,
                    q_proj_dim as u32,
                    eps,
                    stream,
                )?;
            } else if self.gated {
                ops::deinterleave_qg_split(
                    ctx.gpu,
                    self.deinterleave_qg_split_k,
                    qg_out,
                    q_contiguous,
                    n,
                    nq,
                    hd,
                    q_proj_dim as u32,
                    stream,
                )?;
            } else {
                ctx.gpu
                    .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)
                    .map_err(|e| anyhow::anyhow!("Q copy d2d failed: {e}"))?;
                if let Some(ref q_norm_full) = self.attn.q_norm_full {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        q_contiguous,
                        q_norm_full,
                        q_contiguous,
                        n,
                        nq * hd,
                        eps,
                        stream,
                    )?;
                } else if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        q_contiguous,
                        &self.attn.q_norm,
                        q_contiguous,
                        nq * n,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        if !qknorm_rope_fused {
            if let Some(ref k_norm_full) = self.attn.k_norm_full {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    k_contiguous,
                    k_norm_full,
                    k_contiguous,
                    n,
                    nkv * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    k_contiguous,
                    &self.attn.k_norm,
                    k_contiguous,
                    nkv * n,
                    hd,
                    eps,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("k_norm rms_norm failed: nkv={nkv} n={n} hd={hd}: {e}")
                })?;
            }
        }

        // Gemma-4 v_norm — applied at EVERY layer in HF reference
        // (modeling_gemma4.py:1220 `value_states = self.v_norm(value_states)`
        // with `Gemma4RMSNorm(with_scale=False)`). For full-attention K=V
        // layers, v_contiguous holds raw K (aliased v_proj). For sliding
        // layers, v_contiguous holds V_proj output. Either way normalize
        // with pure RMSNorm via the ones-buffer (Gemma-4's absolute-
        // formula rms_norm kernel: `x * rms * 1.0 = x * rms`). V does NOT
        // receive RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        aprof!("deinterleave+norms", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. RoPE for N tokens ──
        if qknorm_rope_fused {
            // Q/K normalization and scalar RoPE already completed in the
            // exact C=1 fusion above.
        } else if self.mla.is_some() {
            // MLA: RoPE already applied inside the MLA block to rope portions only.
            // Skip shared RoPE to avoid double-rotation.
        } else if !self.qwen4_yarn_inv_freq.is_null() {
            ops::rope_yarn_scaled(
                ctx.gpu,
                self.rope_yarn_scaled_k,
                q_contiguous,
                k_contiguous,
                meta.positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.qwen4_yarn_inv_freq,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                self.qwen4_yarn_attention_factor,
                stream,
            )?;
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                meta.positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_proportional failed: {e}"))?;
        } else if self.mrope_interleaved {
            self.apply_mrope(
                ctx.gpu,
                q_contiguous,
                k_contiguous,
                meta.positions,
                meta.positions_h,
                meta.positions_w,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            anyhow::ensure!(
                self.yarn.is_none(),
                "YaRN RoPE is not supported on non-MRoPE layers without precomputed qwen4_yarn_inv_freq"
            );
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                meta.positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope failed: {e}"))?;
        }
        aprof!("rope", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 7. Flash Attention on contiguous Q/K/V (BR=64 for long sequences) ──
        // Runs BEFORE `write_kv_cache` because TurboQuant dtypes apply WHT/InnerQ
        // in-place to `k_contiguous`/`v_contiguous` inside `write_kv_cache`.
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let gate_base = qg_out.offset(q_dim * bf16);
        let gate_fused = match gate_fused_route {
            Qwen38PrefillAttnGateRoute::Complete => {
                ops::prefill_attention_64_gate_fused(
                    ctx.gpu,
                    self.prefill_attn_64_gate_fused_k,
                    q_contiguous,
                    k_contiguous,
                    v_contiguous,
                    attn_out,
                    gate_base,
                    n,
                    1,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    true,
                    self.sliding_window.unwrap_or(0),
                    q_proj_dim as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "fused flash_attn_64+gate failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}"
                    )
                })?;
                mark_qwen38_prefill_attn_gate_engaged();
                true
            }
            Qwen38PrefillAttnGateRoute::Missing => unreachable!(
                "missing fused attention dependencies were rejected before cache mutation"
            ),
            Qwen38PrefillAttnGateRoute::Disabled | Qwen38PrefillAttnGateRoute::Ineligible => false,
        };
        let mut attn_label = if gate_fused {
            "flash_attn_64+sigmoid_gate"
        } else {
            "flash_attn_64"
        };
        if !gate_fused {
            if self.has_wide_prefill_attention(hd) {
                // HDIM=512 (Gemma-4 full attention): scalar reference or, under
                // ATLAS_GEMMA_PREFILL_512_TC=1, the tensor-core instantiation.
                // Full-attention layers (this path) always pass sliding_window=0.
                attn_label = self.prefill_attention_wide(
                    ctx,
                    q_contiguous,
                    k_contiguous,
                    v_contiguous,
                    attn_out,
                    n,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    0,
                    stream,
                )?;
            } else if prefill_attn_fi_eligible(
                cfg!(atlas_qwen4_fi_attn),
                prefill_attn_fi_enabled(),
                meta.num_seqs == 1,
                self.mla.is_none(),
                self.sliding_window.unwrap_or(0),
                n,
                nq,
                nkv,
                hd,
            ) {
                let call = crate::layers::qwen4_fi_attn::Call {
                    q: q_contiguous,
                    k: k_contiguous,
                    v: v_contiguous,
                    k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
                    v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
                    block_table: DevicePtr::NULL,
                    block_size: bs,
                    start: 0,
                    rows: num_tokens,
                    num_q_heads: nq as usize,
                    num_kv_heads: nkv as usize,
                    head_dim: hd as usize,
                    sm_scale: inv_sqrt_d,
                    output: attn_out,
                };
                crate::layers::qwen4_fi_attn::run(&call, stream)?;
                attn_label = "flashinfer_fa2_ragged_hd256";
            } else {
                ops::prefill_attention_64(
                    ctx.gpu,
                    self.prefill_attn_64_k,
                    q_contiguous,
                    k_contiguous,
                    v_contiguous,
                    attn_out,
                    n,
                    1,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    true,
                    self.sliding_window.unwrap_or(0),
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("flash_attn_64 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
                })?;
            }
        }
        // Name the kernel that ran: the wide-head branch is not flash_attn_64.
        aprof!(attn_label, t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 8. Write K/V to paged cache (after contiguous Flash Attention) ──
        let write_start = kv_write_start;
        let write_count = num_tokens.saturating_sub(write_start);
        if write_count > 0 {
            let k_offset = write_start * kv_dim * bf16;
            let v_offset = write_start * kv_dim * bf16;
            let slot_offset = write_start * 8;
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(k_offset),
                v_contiguous.offset(v_offset),
                kv_cache,
                meta.slot.offset(slot_offset),
                write_count as u32,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
        }
        aprof!("kv_cache_write", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 9. Sigmoid gate × attn_out (gated only) — single batched kernel ──
        if self.gated && !gate_fused {
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                nq * hd,
                q_proj_dim as u32,
                n,
                stream,
            )?;
        }
        if !gate_fused {
            aprof!("sigmoid_gate", t0);
            t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
        }

        // ── 10. O projection GEMM ── (extracted to paged_oproj.rs)
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;
        aprof!("o_proj", t0);
        Ok(o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_cache_skip_qknorm_rope_explicit, prefill_attn_fi_eligible,
        resolve_cache_skip_qknorm_rope, resolve_prefill_attn_fi,
    };

    #[test]
    fn cache_skip_qknorm_rope_auto_enables_above_64_sms_and_honors_opt_out() {
        assert_eq!(parse_cache_skip_qknorm_rope_explicit(None), Ok(None));
        assert_eq!(
            parse_cache_skip_qknorm_rope_explicit(Some("0")),
            Ok(Some(false))
        );
        assert_eq!(
            parse_cache_skip_qknorm_rope_explicit(Some("1")),
            Ok(Some(true))
        );
        for invalid in ["", "true", "2", " 1"] {
            assert!(
                parse_cache_skip_qknorm_rope_explicit(Some(invalid)).is_err(),
                "invalid={invalid:?}"
            );
        }

        // GB10 (<= 64 SMs): default-OFF when unset; explicit "1" still enables.
        for sm in [48u32, 64] {
            assert!(!resolve_cache_skip_qknorm_rope(None, sm, true, true));
            assert!(resolve_cache_skip_qknorm_rope(Some(true), sm, true, true));
            assert!(!resolve_cache_skip_qknorm_rope(Some(false), sm, true, true));
        }

        // Wide-SM GPUs (> 64 SMs, e.g. SM120): auto-ON when non-YaRN and kernels ready;
        // explicit "0" opts out; YaRN or missing kernels stay safely OFF under auto.
        for sm in [65u32, 110, 170, 188] {
            assert!(resolve_cache_skip_qknorm_rope(None, sm, true, true));
            assert!(!resolve_cache_skip_qknorm_rope(None, sm, false, true));
            assert!(!resolve_cache_skip_qknorm_rope(None, sm, true, false));
            assert!(!resolve_cache_skip_qknorm_rope(Some(false), sm, true, true));
            // Explicit "1" still requests even when kernel is missing so route fails closed.
            assert!(resolve_cache_skip_qknorm_rope(Some(true), sm, true, false));
        }
    }

    #[test]
    fn prefill_attn_fi_auto_enables_above_64_sms_and_honors_opt_out() {
        assert!(!resolve_prefill_attn_fi(None, 48));
        assert!(!resolve_prefill_attn_fi(None, 64));
        assert!(resolve_prefill_attn_fi(Some("1"), 48));
        assert!(resolve_prefill_attn_fi(Some("true"), 48));

        for sm in [65u32, 110, 170, 188] {
            assert!(resolve_prefill_attn_fi(None, sm), "sm={sm}");
            assert!(!resolve_prefill_attn_fi(Some("0"), sm), "sm={sm}");
            assert!(!resolve_prefill_attn_fi(Some("false"), sm), "sm={sm}");
        }
    }

    #[test]
    fn prefill_attn_fi_eligibility_matches_qwen38_27b_and_guards_edges() {
        // Qwen3.8-27B geometry: nq=24, nkv=4, hd=256, causal, no sliding window.
        assert!(prefill_attn_fi_eligible(
            true, true, true, true, 0, 8192, 24, 4, 256
        ));
        assert!(prefill_attn_fi_eligible(
            true, true, true, true, 0, 64, 24, 4, 256
        ));
        // Build without FLASHINFER_HOME never engages.
        assert!(!prefill_attn_fi_eligible(
            false, true, true, true, 0, 8192, 24, 4, 256
        ));
        // Short prompts, non-256 head_dim, sliding window, MLA, multi-seq stay on BR64.
        assert!(!prefill_attn_fi_eligible(
            true, true, true, true, 0, 63, 24, 4, 256
        ));
        assert!(!prefill_attn_fi_eligible(
            true, true, true, true, 4096, 8192, 24, 4, 256
        ));
        assert!(!prefill_attn_fi_eligible(
            true, true, true, true, 0, 8192, 24, 4, 128
        ));
        assert!(!prefill_attn_fi_eligible(
            true, true, false, true, 0, 8192, 24, 4, 256
        ));
        assert!(!prefill_attn_fi_eligible(
            true, true, true, false, 0, 8192, 24, 4, 256
        ));
    }
}
