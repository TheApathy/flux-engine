// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter constructor + target-config validation.
//!
//! Split out of `dflash_head.rs` for file-size budget. Contains
//! [`BlockDiffusionDraftHead::from_weights`] (kernel resolution + circular
//! context-cache setup) and [`BlockDiffusionDraftHead::validate_against_target`].

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::OnceLock;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    BlockDiffusionDraftHead, DflashFp8LmHead, DflashKernels, DflashLayer, DflashLayerFp8Weights,
    DflashLayerNvfp4, DflashLayerQuantWeights, DflashQuantization, DflashScratch,
};
use crate::weight_loader::{
    DflashConfig, DflashLayerProjections, DflashLayerWeights, DflashSelectorWeights, DflashWeights,
    DsparkVerifyMode, MarkovWeights,
};
use crate::weight_map::{DenseWeight, QuantizedWeight, quantize_to_fp8, quantize_to_nvfp4};

/// Process-lifetime sidecar registry mapping an `fc_nvfp4` base weight pointer
/// (`QuantizedWeight::weight.0`) to its pre-transposed `[K/2, N_pad]` copy used
/// by `w4a16_gemm_t_m32_n64(_splitk)` / `w4a16_gemm_t_m16`.
///
/// Keying by the immutable GPU device address avoids expanding the root
/// `BlockDiffusionDraftHead` struct while keeping lookup O(1) and lock-light.
static FC_NVFP4_T_REGISTRY: OnceLock<Mutex<HashMap<u64, QuantizedWeight>>> = OnceLock::new();

pub(super) fn register_fc_nvfp4_t(fc_weight_ptr: DevicePtr, fc_t: QuantizedWeight) {
    if fc_weight_ptr == DevicePtr::NULL || fc_t.weight == DevicePtr::NULL {
        return;
    }
    let map = FC_NVFP4_T_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    map.lock().insert(fc_weight_ptr.0, fc_t);
}

pub(super) fn lookup_fc_nvfp4_t(fc_weight_ptr: DevicePtr) -> Option<QuantizedWeight> {
    if fc_weight_ptr == DevicePtr::NULL {
        return None;
    }
    FC_NVFP4_T_REGISTRY
        .get()
        .and_then(|map| map.lock().get(&fc_weight_ptr.0).copied())
}

/// Device/kernel capability snapshot for deciding whether to materialize the
/// transposed `fc_nvfp4_t` weight (`[K/2, N_pad]`, ~73.7 MB at N=5120, K=25600).
#[derive(Debug, Clone, Copy)]
pub(super) struct FcTransposePolicyInput {
    pub want_ffn_kgamma_t: bool,
    pub want_attn_kgamma_t: bool,
    pub sm_count: u32,
    pub has_m32_n64: bool,
    pub has_m16_n128: bool,
    pub has_coalesced_bf16: bool,
    pub hidden_size: usize,
    pub target_hidden_dim: usize,
}

/// Decide whether `fc_nvfp4` should also be transposed into `fc_nvfp4_t`.
///
/// Un-normalized target hidden states feeding `fc` can contain outlier
/// activations exceeding unscaled FP8 E4M3's `±448` ceiling (regressing
/// arithmetic acceptance when routed through `w4a16_gemm_t_m32_n64*`).
/// Never materialize `fc_nvfp4_t` on wide discrete GPUs (`sm_count > 64`) or
/// whenever unclipped BF16×BF16 MMA `w4a16_gemm_coalesced*` kernels are
/// available — saving `73.7 MB` VRAM and preserving full BF16 precision.
pub(super) fn should_build_fc_nvfp4_t(input: FcTransposePolicyInput) -> bool {
    const WIDE_DISCRETE_SM_THRESHOLD: u32 = 64;
    if input.has_coalesced_bf16
        || input.sm_count > WIDE_DISCRETE_SM_THRESHOLD
        || input.hidden_size == 0
        || input.target_hidden_dim == 0
        || !input.target_hidden_dim.is_multiple_of(32)
    {
        return false;
    }
    let kernel_shape_ok = (input.has_m32_n64 && input.hidden_size.is_multiple_of(64))
        || (input.has_m16_n128 && input.hidden_size.is_multiple_of(128));
    if !kernel_shape_ok {
        return false;
    }
    input.want_ffn_kgamma_t || input.want_attn_kgamma_t
}

#[path = "dflash_context_window.rs"]
mod dflash_context_window;

/// Optional environment override for drafter weight precision so online FP8
/// quantization (`ATLAS_DFLASH_QUANTIZATION=fp8` / `FLUX_DFLASH_QUANTIZATION=fp8`)
/// can be selected without altering CLI flags.
fn resolve_dflash_quantization_env_override(
    requested: DflashQuantization,
) -> Result<DflashQuantization> {
    let raw = match std::env::var("ATLAS_DFLASH_QUANTIZATION") {
        Ok(value) => Some(("ATLAS_DFLASH_QUANTIZATION", value)),
        Err(std::env::VarError::NotPresent) => match std::env::var("FLUX_DFLASH_QUANTIZATION") {
            Ok(value) => Some(("FLUX_DFLASH_QUANTIZATION", value)),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                anyhow::bail!("FLUX_DFLASH_QUANTIZATION is not valid Unicode")
            }
        },
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ATLAS_DFLASH_QUANTIZATION is not valid Unicode")
        }
    };
    let Some((var_name, value)) = raw else {
        return Ok(requested);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "" => Ok(requested),
        "bf16" => Ok(DflashQuantization::Bf16),
        "fp8" => Ok(DflashQuantization::Fp8),
        "nvfp4" => Ok(DflashQuantization::Nvfp4),
        other => anyhow::bail!("unknown {var_name}={other:?}; expected `bf16`, `fp8`, or `nvfp4`"),
    }
}

/// Resolve the two Qwen config dialects (`rope_scaling` and Transformers 5's
/// `rope_parameters`) into the exact frequency table and amplitude multiplier
/// consumed by the DFlash kernel.
fn dflash_rope_table(config: &DflashConfig, rotary_dim: usize) -> (f32, f32, Vec<f32>) {
    let rope_theta = config
        .rope_scaling
        .as_ref()
        .and_then(|s| s.rope_theta)
        .unwrap_or(config.rope_theta);
    let dim_f = rotary_dim as f32;
    let n_pairs = rotary_dim / 2;
    let mut inv_freq_table = vec![0.0f32; n_pairs];
    let mut attention_factor = 1.0f32;

    if let Some(scaling) = config.rope_scaling.as_ref()
        && scaling.rope_type.as_deref() == Some("yarn")
    {
        let factor = scaling.factor.unwrap_or(64.0);
        let beta_fast = scaling.beta_fast.unwrap_or(32.0);
        let beta_slow = scaling.beta_slow.unwrap_or(1.0);
        let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(4096) as f32;
        let is_qwen4_flash_next_drafter =
            config.num_target_layers == 48 && config.hidden_size == 2560;
        attention_factor = scaling.attention_factor.unwrap_or_else(|| {
            if is_qwen4_flash_next_drafter || factor <= 1.0 {
                1.0
            } else {
                1.0 + 0.1 * factor.ln()
            }
        });
        let find_correction_dim = |num_rot: f32| -> f32 {
            (dim_f * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * rope_theta.ln())
        };
        let low = find_correction_dim(beta_fast).floor().max(0.0);
        let high = find_correction_dim(beta_slow)
            .ceil()
            .min((rotary_dim - 1) as f32);
        let ramp_denom = if (high - low).abs() < 1e-6 {
            high - low + 0.001
        } else {
            high - low
        };
        for (j, freq) in inv_freq_table.iter_mut().enumerate() {
            let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
            let inv_freq_extrap = 1.0 / pos_freq;
            let inv_freq_interp = 1.0 / (factor * pos_freq);
            let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            *freq = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
        }
    } else {
        for (j, freq) in inv_freq_table.iter_mut().enumerate() {
            let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
            *freq = 1.0 / pos_freq;
        }
    }

    (rope_theta, attention_factor, inv_freq_table)
}

/// Compute byte capacity for `DflashScratch::{topk_tokens_dev, topk_logits_dev}`.
///
/// Covers both C=1 (`gamma * 32 * 4`) and cross-sequence batched proposals
/// (`DFLASH_BATCH_MAX_SEL_ROWS * 32 * 4`), widening shortlist headroom to
/// `DFLASH2_GUMBEL_SHORTLIST_K` (32) without altering `DDTREE_TOP_K_MAX` (16).
#[inline]
#[must_use]
fn selector_topk_scratch_bytes(gamma: usize) -> usize {
    let max_rows = gamma.max(super::DFLASH_BATCH_MAX_SEL_ROWS);
    let max_k = super::DDTREE_TOP_K_MAX.max(super::DFLASH2_GUMBEL_SHORTLIST_K);
    max_rows * max_k * size_of::<u32>()
}

/// Compute byte capacity for `DflashScratch::batch_out_probs`.
///
/// Allocates at least `64 * 8 * 4 = 2048` B (shared by C=1 and batched C=2..64).
#[inline]
#[must_use]
fn batch_out_probs_scratch_bytes(gamma: usize) -> usize {
    gamma
        .max(super::DFLASH_BATCH_MAX_SEL_ROWS)
        .max(super::DFLASH_BATCH_OUT_PROBS_MIN_SLOTS)
        * size_of::<f32>()
}

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        target_hidden_size: usize,
        target_vocab_size: usize,
        gamma: Option<usize>,
        physical_verify_k: usize,
        verify_mode: DsparkVerifyMode,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        quantization: DflashQuantization,
        draft_sample_method: crate::dflash_sampling::DflashDraftSampleMethod,
        probabilistic_qualification_enabled: bool,
    ) -> Result<Self> {
        // Drafter's `fc` is `[draft_hidden, len(target_layer_ids) * target_hidden]`.
        // We rely on the drafter config's `hidden_size` and the parsed
        // `target_layer_ids` to derive the expected target_hidden, then
        // validate it matches what the caller provided.
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        let expected_fc_shape = [
            hidden_size,
            target_layer_ids.len().saturating_mul(target_hidden_size),
        ];
        anyhow::ensure!(
            weights.fc_shape == expected_fc_shape,
            "DFlash drafter fc.weight shape {:?} is incompatible with target intermediate width {} and {} capture layers; expected {:?}",
            weights.fc_shape,
            target_hidden_size,
            target_layer_ids.len(),
            expected_fc_shape,
        );
        weights.config.validate_verify_mode(verify_mode)?;
        let checkpoint_family = weights.config.checkpoint_family()?;
        crate::dflash_sampling::validate_dflash_runtime_admission(
            draft_sample_method,
            checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dflash
                && weights.config.is_dflash2()
                && weights.selector.is_some(),
            probabilistic_qualification_enabled,
        )?;
        let native_flash_next = dflash_context_window::is_native_flash_next(
            &weights.config.architectures,
            weights.config.model_type.as_deref(),
            hidden_size,
            intermediate_size,
            num_layers,
            weights.config.num_target_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            &target_layer_ids,
        );
        // Resolve and reject an invalid runtime context contract before any
        // backend kernel lookup or device allocation has an observable effect.
        let all_layers_sliding_window = dflash_context_window::effective_all_layers_sliding_window(
            weights.config.layer_types.as_deref(),
            weights.config.sliding_window,
            num_layers,
        );
        let ctx_window = match all_layers_sliding_window {
            Some(sw) => dflash_context_window::resolve_context_window_with_sliding(
                window_size,
                max_seq_len,
                native_flash_next,
                Some(sw),
            ),
            None => dflash_context_window::resolve_context_window(
                window_size,
                max_seq_len,
                native_flash_next,
            ),
        }
        .map_err(|error| anyhow::anyhow!(error))?;
        tracing::info!(
            "DFlash resident context window = {} (absolute max_seq_len={}, native_flash_next={}, all_layers_sliding_window={:?})",
            ctx_window,
            max_seq_len,
            native_flash_next,
            all_layers_sliding_window,
        );
        let block_size = weights.config.resolved_block_size();
        let gamma_val = weights.config.resolve_draft_count(gamma)?;
        super::draft_budget::DflashDraftBudget::validate_head(gamma_val, physical_verify_k)?;
        let quantization = resolve_dflash_quantization_env_override(quantization)?;
        let w4a4_requested = super::w4a4_admission::w4a4_requested()?;
        let exact_rtncal_schema = crate::weight_loader::exact_dflash2_profile(&weights.config)
            .is_ok_and(|profile| profile.allow_rtncal_w4a4)
            && weights
                .config
                .quantization_config
                .as_ref()
                .is_some_and(|config| config.is_exact_dflash2_rtncal());
        let w4a4_enabled = super::w4a4_admission::admit_w4a4_shadow(
            w4a4_requested,
            exact_rtncal_schema,
            quantization,
            gamma_val,
            physical_verify_k,
            super::forward_block_layer::dflash_noise_only_enabled(),
        )?;
        if checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dspark
            && gamma_val > block_size
        {
            tracing::warn!(
                "DSpark experimental wider single-pass query block: checkpoint B={} requested \
                 k={}; layout is anchor + {} masks followed by one k-token Markov chain; \
                 target verification remains static K={}",
                block_size,
                gamma_val,
                gamma_val.saturating_sub(1),
                gamma_val + 1,
            );
        }
        if let Some(confidence) = weights.confidence.as_ref() {
            tracing::info!(
                "DSpark verify planner: static verify-all (gamma+1 rows); confidence projection \
                 validated/loaded but intentionally not consumed, matching SGLang PR 34966 \
                 static semantics (input_dim={}, with_markov={})",
                confidence.input_dim,
                confidence.with_markov,
            );
        } else if checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dspark {
            tracing::info!(
                "DSpark verify planner: static verify-all (gamma+1 rows); checkpoint has no \
                 learned confidence head"
            );
        }

        // Resolve only kernels reachable from the active forward. An earlier
        // scaffold also allocated a full FP8 paged KV cache and resolved its
        // reshape/paged-attention handles, but neither the cache nor handles
        // had a read or write call site. Keeping them implied PR-34966 parity
        // while consuming up to gigabytes without affecting a single token.
        let has_dflash2_conv = weights.selector.is_some()
            || weights
                .layers
                .iter()
                .any(|l| l.attention_conv.is_some() || l.mlp_conv.is_some());
        let kernels = DflashKernels {
            // DFlash drafter uses HF's vanilla RMSNorm convention
            // (`out = x * w / RMS(x)`), NOT Atlas's default offset-from-1
            // form (`out = x * (1 + w) / RMS(x)`). Atlas's standard
            // `rms_norm` kernel includes the `+1` for Qwen3-Next-style
            // checkpoints; we must use `rms_norm_vanilla` for the drafter
            // to match the drafter's HF-trained weights exactly.
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            // Dedicated scaled YaRN kernel. Unlike the legacy shared kernel,
            // this applies HF's `attention_factor` multiplier to cos/sin.
            rope_qwen3: gpu.kernel("rope", "rope_forward_yarn_scaled")?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            scaled_add: gpu.kernel("residual_add", "bf16_scaled_add")?,
            token_recommit: gpu.kernel("residual_add", "dflash_token_recommit")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            argmax_rows: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_rows"),
            // DFlash 2 conv kernels are resolved whenever `has_dflash2_conv`
            // holds (selector or any layer conv present); selector kernels are
            // resolved only when `weights.selector.is_some()`. Plain DFlash /
            // DSpark drafters keep sentinel-zero handles and never load them.
            dflash2_conv_prepare: if has_dflash2_conv {
                gpu.kernel("dflash2_conv", "dflash2_conv_prepare")?
            } else {
                KernelHandle(0)
            },
            dflash2_conv_finish: if has_dflash2_conv {
                gpu.kernel("dflash2_conv", "dflash2_conv_finish")?
            } else {
                KernelHandle(0)
            },
            dflash2_conv_prepare_batched: if has_dflash2_conv {
                crate::layers::try_kernel(gpu, "dflash2_conv", "dflash2_conv_prepare_batched")
            } else {
                KernelHandle(0)
            },
            dflash2_conv_finish_batched: if has_dflash2_conv {
                crate::layers::try_kernel(gpu, "dflash2_conv", "dflash2_conv_finish_batched")
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk: if weights.selector.is_some() {
                gpu.kernel("dflash2_selector", "dflash2_selector_walk")?
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_batched: if weights.selector.is_some() {
                crate::layers::try_kernel(gpu, "dflash2_selector", "dflash2_selector_walk_batched")
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_gumbel_batched: if weights.selector.is_some() {
                crate::layers::try_kernel(
                    gpu,
                    "dflash2_selector",
                    "dflash2_selector_walk_gumbel_batched",
                )
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_gumbel_batched_k32: if weights.selector.is_some() {
                crate::layers::try_kernel(
                    gpu,
                    "dflash2_selector",
                    "dflash2_selector_walk_gumbel_batched_k32",
                )
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_probabilistic: if weights.selector.is_some() {
                crate::layers::try_kernel(
                    gpu,
                    "dflash2_selector",
                    "dflash2_selector_walk_probabilistic",
                )
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_probabilistic_batched: if weights.selector.is_some() {
                crate::layers::try_kernel(
                    gpu,
                    "dflash2_selector",
                    "dflash2_selector_walk_probabilistic_batched",
                )
            } else {
                KernelHandle(0)
            },
            dflash2_selector_walk_ngram_batched: if weights.selector.is_some() {
                crate::layers::try_kernel(
                    gpu,
                    "dflash2_selector",
                    "dflash2_selector_walk_ngram_batched",
                )
            } else {
                KernelHandle(0)
            },
            split_ctx_h128_batched: if head_dim == 128 {
                crate::layers::try_kernel(
                    gpu,
                    "inferspark_prefill_h128",
                    "inferspark_prefill_split_ctx_h128_batched",
                )
            } else {
                KernelHandle(0)
            },
            // Algorithm-specific: generic SpecForge DFlash never needs this
            // symbol. A DSpark/Markov checkpoint resolves it fail-closed.
            argmax_add: if weights.markov.is_some() {
                gpu.kernel("argmax", "argmax_add_bf16")?
            } else {
                KernelHandle(0)
            },
            // Top-K and Markov-add argmax kernels live in the same `argmax`
            // module (kernels/gb10/common/argmax_bf16.cu). Resolved unconditionally
            // so the handle table is shape-stable. Only invoked when the
            // DDTree propose path opts into real top-k tree construction
            // (chain-only mode skips it entirely).
            topk: gpu.kernel("argmax", "topk_bf16")?,
            // Fast radix top-K (`topk_bf16_radix`, K<=32). GB10 KERNEL.toml maps
            // `argmax_bf16.cu` to module `"argmax"`; fall back to `"argmax_bf16"`
            // when a backend registers the file stem directly.
            topk_bf16_radix: {
                let h = crate::layers::try_kernel(gpu, "argmax", "topk_bf16_radix");
                if h.0 != 0 {
                    h
                } else {
                    crate::layers::try_kernel(gpu, "argmax_bf16", "topk_bf16_radix")
                }
            },
            strided_copy_rows: crate::layers::try_kernel(
                gpu,
                "strided_copy_rows",
                "strided_copy_rows_16",
            ),
            batched_embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
            // Dispatch prefill attention by drafter head_dim: HDIM=128 drafters
            // (Qwen3.6/3.8 DFlash/DFlash2) use `inferspark_prefill_h128`, while
            // HDIM=256 drafters (e.g. PixelML Qwen3DSparkModel) use the native
            // HDIM=256 `inferspark_prefill` kernel. Fail closed on any other
            // head_dim rather than silently reading half-valid SMEM tiles.
            prefill_attn: match head_dim {
                128 => gpu.kernel("inferspark_prefill_h128", "inferspark_prefill_h128")?,
                256 => gpu.kernel("inferspark_prefill", "inferspark_prefill")?,
                other => anyhow::bail!(
                    "DFlash drafter head_dim={other} is unsupported for prefill attention; expected 128 or 256"
                ),
            },
            // NVFP4 kernels: resolved unconditionally so the kernel-handle
            // table is shape-stable across both quantization paths. Under
            // `DflashQuantization::Bf16` these handles are never invoked
            // (forward_block_layer's match-dispatch on the layer variant
            // routes BF16 layers through `dense_gemm` / `dense_gemv`).
            // Naming matches the target-side resolutions in
            // `qwen3_attention/init.rs` and `dense_ffn.rs`.
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            // Optional while the policy is off. An explicit eligible bulk
            // route validates this handle and fails closed at dispatch.
            w4a16_gemm_pipe: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_pipe_dual: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_dual"),
            // M_TILE=16 specialization for the drafter FFN K=γ batched
            // path (ATLAS_DFLASH_FFN_KGAMMA=1). Resolved via try_kernel so
            // older built kernel caches that pre-date this symbol still
            // link; the dispatch in forward_block_layer_nvfp4 checks for
            // `KernelHandle(0)` and falls back to the M_TILE=64 path.
            w4a16_gemm_t_m16: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            w4a16_gemm_t_m32_n64: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // FUSED gate_proj + up_proj + SiLU·mul — the drafter FFN routes
            // through it in place of two m32_n64 GEMMs + silu_mul when the
            // transposed gate/up weights are present (see the 3i branch).
            w4a16_gemm_t_m32_n64_gateup_silu: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu",
            ),
            w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64",
            ),
            // Split-K variant of `w4a16_gemm_t_m32_n64` + its FP32 band
            // reducer. Both optional: an older kernel cache without these
            // symbols leaves KernelHandle(0) and `draft_gemm_t` falls back to
            // the single-slice path regardless of the env gate.
            w4a16_gemm_t_m32_n64_splitk: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: crate::layers::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            fp8_gemm_t: crate::layers::try_kernel(gpu, "w4a16", "fp8_gemm_t"),
            dense_gemv_fp8w: if quantization == DflashQuantization::Fp8 {
                gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?
            } else {
                crate::layers::try_kernel(gpu, "gemv_fp8w", "dense_gemv_fp8w")
            },
            fp8_gemm_dflash: if quantization == DflashQuantization::Fp8 {
                gpu.kernel("fp8_gemm_dflash", "fp8_gemm_t_m16_n64_rowscale")?
            } else {
                crate::layers::try_kernel(gpu, "fp8_gemm_dflash", "fp8_gemm_t_m16_n64_rowscale")
            },
        };
        if weights.markov.is_some() {
            tracing::info!(
                "DSpark Markov dispatch: device-resident sequential W1 gather + W2 GEMV + \
                 full-vocabulary BF16 argmax(base+bias); lowest-token tie-break; one final \
                 gamma*u32 D2H and no per-position host synchronization"
            );
        }

        // Per-step scratch buffers. BF16 = 2 bytes/element.
        //
        // Sized for `n_attn_slots = ctx_window + γ` rows in the attention
        // path. The first `ctx_window` slots hold projected target ctx
        // (K/V only — Q is zero-padded so its attention output is
        // discarded). The next γ slots hold the noise tokens. lm_head +
        // logits + argmax tail still operates on γ rows (offset past ctx).
        let bf16 = 2usize;
        let g = gamma_val;
        // The startup-resolved value controls scratch, cache, and runtime
        // attention. Native Flash-Next keeps this 4096-token tail resident
        // even when absolute positions extend to 1M.
        let kv_dim = num_kv_heads * head_dim;
        let circular_kv_bytes = ctx_window * kv_dim * bf16 * 2 * num_layers;
        let circular_fc_bytes = ctx_window * hidden_size * bf16;
        tracing::info!(
            "DFlash draft context storage: active=BF16 circular, window={}, \
             per-sequence K/V={:.1} MiB + FC={:.1} MiB; paged FP8=unsupported/not allocated",
            ctx_window,
            circular_kv_bytes as f64 / (1024.0 * 1024.0),
            circular_fc_bytes as f64 / (1024.0 * 1024.0),
        );
        if checkpoint_family == crate::weight_loader::DrafterCheckpointFamily::Dspark {
            tracing::warn!(
                "DSpark KV parity boundary: Flux uses a BF16 circular window; SGLang PR 34966 \
                 uses a paged FP8-E4M3 draft pool and full context when no draft window is set. \
                 Do not label this run PR-equivalent."
            );
        }
        // Family-specific query geometry. Generic DFlash uses anchor + γ
        // MASK rows and drops the anchor output. DSpark uses exactly γ rows
        // (anchor + γ-1 MASKs) and samples all γ outputs, matching the
        // SGLang DSpark path rather than the generic DFlash path.
        let row_layout = super::DraftRowLayout::for_family(checkpoint_family, g);
        let n_attn = row_layout.query_rows + ctx_window;
        let q_dim = num_q_heads * head_dim;
        let mut alloc_tx = crate::device_allocations::DeviceAllocations::new(gpu);
        let w4a4_shadow = if w4a4_enabled {
            let shadow = super::w4a4_shadow::DflashW4a4Shadow::new(
                gpu,
                row_layout.query_rows,
                intermediate_size.max(hidden_size).max(q_dim),
            )?;
            for ptr in shadow.device_ptrs() {
                alloc_tx.adopt(ptr);
            }
            Some(shadow)
        } else {
            None
        };
        let logits_layout =
            super::logits_layout::LogitsLayout::new(g, target_vocab_size, vocab_size)?;
        let max_sel_rows = g.max(super::DFLASH_BATCH_MAX_SEL_ROWS);
        let topk_bytes = selector_topk_scratch_bytes(g);
        let batch_out_probs_bytes = batch_out_probs_scratch_bytes(g);
        let early_stage1_fc_bytes = if super::async_propose::dflash_early_stage1_enabled()
            || super::async_propose::dflash_opt_branch0_enabled()
        {
            physical_verify_k.max(g + 1) * hidden_size * bf16
        } else {
            0
        };
        let scratch = DflashScratch {
            stream_buf: alloc_tx.alloc(n_attn * hidden_size * bf16)?,
            norm_buf: alloc_tx.alloc(n_attn * hidden_size * bf16)?,
            q_buf: alloc_tx.alloc(n_attn * q_dim * bf16)?,
            k_buf: alloc_tx.alloc(n_attn * kv_dim * bf16)?,
            v_buf: alloc_tx.alloc(n_attn * kv_dim * bf16)?,
            attn_out: alloc_tx.alloc(n_attn * q_dim * bf16)?,
            mlp_intermediate: alloc_tx.alloc(n_attn * intermediate_size * bf16)?,
            mlp_up: alloc_tx.alloc(n_attn * intermediate_size * bf16)?,
            stream_acc: alloc_tx.alloc(n_attn * hidden_size * bf16)?,
            fc_proj: alloc_tx.adopt(gpu.alloc(ctx_window * hidden_size * bf16)?),
            logits: alloc_tx.alloc(logits_layout.allocation_bytes())?,
            draft_tokens_dev: alloc_tx.alloc(n_attn.max(max_sel_rows) * 4)?,
            draft_tokens_dev_alt: alloc_tx.alloc(n_attn.max(max_sel_rows) * 4)?,
            position_ids: alloc_tx.alloc(n_attn * 4)?,
            // DDTree M4B v2 / DFlash2 selector top-K scratch. Sized for
            // `max(γ, 128)` rows × `DFLASH2_GUMBEL_SHORTLIST_K` (32) so both
            // single-seq C=1 and cross-sequence batched DFlash2 fit K=32 cleanly.
            topk_tokens_dev: alloc_tx.alloc(topk_bytes)?,
            topk_logits_dev: alloc_tx.alloc(topk_bytes)?,
            // EAGLE-3.1 per-layer FC-normalization scratch
            // (ATLAS_DFLASH_FC_LAYERNORM=1). `fc_norm_in` holds one fc-input
            // slot [n_target_layers * target_hidden] BF16; `fc_norm_ones_w` is
            // an all-ones BF16 (`0x3F80`) weight of length target_hidden so
            // `rms_norm_vanilla` (`x * w / RMS(x)`) reduces to unit-variance
            // `x / RMS(x)` per slice. Both are allocated unconditionally (a few
            // KB) so the scratch shape is independent of the runtime flag; the
            // unit-scale weight is populated via H2D below.
            fc_norm_in: alloc_tx.alloc(target_layer_ids.len() * target_hidden_size * bf16)?,
            fc_norm_ones_w: alloc_tx.alloc(target_hidden_size * bf16)?,
            // DSpark Markov head scratch. Allocated only when the checkpoint
            // ships the head (`weights.markov.is_some()`), else NULL — the
            // propose path only touches these on the Some branch.
            markov_w1_row: match weights.markov.as_ref() {
                Some(m) => alloc_tx.alloc(m.rank * bf16)?,
                None => DevicePtr::NULL,
            },
            markov_bias: match weights.markov.as_ref() {
                Some(_) => alloc_tx.alloc(vocab_size * bf16)?,
                None => DevicePtr::NULL,
            },
            markov_prev_dev: match weights.markov.as_ref() {
                Some(_) => alloc_tx.alloc(4)?,
                None => DevicePtr::NULL,
            },
            // DFlash 2 conv + selector scratch. Conv buffers are allocated
            // whenever `has_dflash2_conv` holds (selector or any layer conv
            // present); selector buffers only when `weights.selector.is_some()`.
            // Sized for `n_attn` rows so the noise-slice pointers can share one
            // allocation with ctx rows unused. `conv_dyn` =
            // [n_attn, 2*kernel*groups], `conv_dyn1` = [n_attn, kernel*groups],
            // `selector_hidden` = [gamma, rank].
            conv_dyn: if has_dflash2_conv {
                // [n_attn, 2*kernel*groups] — sized with `hidden_size` per
                // row (4× headroom over the released 2·2·(5120/16)=1280).
                alloc_tx.alloc(n_attn * hidden_size * 2 * bf16)?
            } else {
                DevicePtr::NULL
            },
            conv_dyn1: if has_dflash2_conv {
                // [n_attn, kernel*groups] — half the dynamic width above.
                alloc_tx.alloc(n_attn * hidden_size * bf16)?
            } else {
                DevicePtr::NULL
            },
            conv_out: if has_dflash2_conv {
                // [n_attn, hidden] — non-aliased conv output (see field doc).
                alloc_tx.alloc(n_attn * hidden_size * bf16)?
            } else {
                DevicePtr::NULL
            },
            selector_hidden: match weights.selector.as_ref() {
                Some(sel) => alloc_tx.alloc(max_sel_rows * sel.rank * bf16)?,
                None => DevicePtr::NULL,
            },
            selector_uniforms: match (weights.selector.as_ref(), draft_sample_method) {
                (Some(_), crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic) => {
                    alloc_tx.alloc(g * size_of::<f32>())?
                }
                _ => DevicePtr::NULL,
            },
            selector_q_modifiers: match (weights.selector.as_ref(), draft_sample_method) {
                (Some(_), crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic) => {
                    alloc_tx.alloc(super::selector_policy::SELECTOR_Q_MODIFIER_BYTES)?
                }
                _ => DevicePtr::NULL,
            },
            selector_q_probabilities: match (weights.selector.as_ref(), draft_sample_method) {
                (Some(_), crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic) => {
                    alloc_tx.alloc(g * super::DDTREE_TOP_K_MAX * size_of::<f32>())?
                }
                _ => DevicePtr::NULL,
            },
            selector_q_candidates: match (weights.selector.as_ref(), draft_sample_method) {
                (Some(_), crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic) => {
                    alloc_tx.alloc(g * super::DDTREE_TOP_K_MAX * size_of::<u32>())?
                }
                _ => DevicePtr::NULL,
            },
            // Split-K FP32 partials `[k_splits, 64, max(hidden, q_dim)]`.
            // 8 = the slice clamp in `draft_splitk()`, 64 = 2 M-tiles of
            // M_TILE=32 so `C=4 q_proj` and `C=8 o/down/q_proj` fit full 8-way
            // Split-K without overrunning. Allocated here because `gpu.alloc`
            // is illegal during CUDA graph capture, and skipped entirely when
            // the gate is off so the default path holds no extra device memory.
            splitk_ws: if super::draft_splitk::draft_splitk() >= 2 {
                alloc_tx.alloc(8 * 64 * hidden_size.max(q_dim) * 4)?
            } else {
                DevicePtr::NULL
            },
            batch_ctx_k_ptrs: alloc_tx
                .alloc(num_layers.max(1) * super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>())?,
            batch_ctx_v_ptrs: alloc_tx
                .alloc(num_layers.max(1) * super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>())?,
            batch_eff_ctx: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>())?,
            batch_ring_start: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>())?,
            batch_anchor_ids: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>())?,
            batch_gumbel_seeds: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>())?,
            batch_gumbel_step0_pos: alloc_tx
                .alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>())?,
            batch_gumbel_temps: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>())?,
            batch_gumbel_top_ks: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>())?,
            batch_gumbel_top_ps: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>())?,
            batch_gumbel_min_ps: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>())?,
            batch_out_probs: alloc_tx.alloc(batch_out_probs_bytes)?,
            batch_out_probs_alt: alloc_tx.alloc(batch_out_probs_bytes)?,
            early_stage1_fc: if early_stage1_fc_bytes > 0 {
                alloc_tx.alloc(early_stage1_fc_bytes)?
            } else {
                DevicePtr::NULL
            },
            q_row_bases: alloc_tx.alloc(super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>())?,
            shared_q_ids: DevicePtr::NULL,
            shared_q_probs: DevicePtr::NULL,
            shared_q_row_cap: 0,
            shared_q_stride: 0,
            ngram_tables_dev: if weights.selector.is_some()
                && super::ngram_bonus::ngram_bonus_enabled()
            {
                alloc_tx
                    .alloc(super::DFLASH_BATCH_MAX_SEQS * super::ngram_bonus::NGRAM_TABLE_BYTES)?
            } else {
                DevicePtr::NULL
            },
            ngram_prev3_dev: if weights.selector.is_some()
                && super::ngram_bonus::ngram_bonus_enabled()
            {
                alloc_tx
                    .alloc(super::DFLASH_BATCH_MAX_SEQS * super::ngram_bonus::NGRAM_PREV3_BYTES)?
            } else {
                DevicePtr::NULL
            },
        };
        // Initialize the FC-layernorm weight buffer to BF16 `1.0` (`0x3F80`) so
        // `rms_norm_vanilla` (`x * w / RMS(x)`) computes plain unit-variance
        // normalization (`x / RMS(x)`, variant a).
        let fc_norm_ones = 0x3F80_u16.to_le_bytes().repeat(target_hidden_size);
        gpu.copy_h2d(&fc_norm_ones, scratch.fc_norm_ones_w)?;
        for (ptr, bytes) in [
            (
                scratch.batch_gumbel_seeds,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>(),
            ),
            (
                scratch.batch_gumbel_step0_pos,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<u64>(),
            ),
            (
                scratch.batch_gumbel_temps,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>(),
            ),
            (
                scratch.batch_gumbel_top_ks,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>(),
            ),
            (
                scratch.batch_gumbel_top_ps,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>(),
            ),
            (
                scratch.batch_gumbel_min_ps,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<f32>(),
            ),
            (scratch.batch_out_probs, batch_out_probs_bytes),
            (scratch.batch_out_probs_alt, batch_out_probs_bytes),
            (
                scratch.q_row_bases,
                super::DFLASH_BATCH_MAX_SEQS * size_of::<u32>(),
            ),
        ] {
            gpu.memset(ptr, 0, bytes)?;
        }
        if !scratch.ngram_tables_dev.is_null() {
            gpu.memset(
                scratch.ngram_tables_dev,
                0,
                super::DFLASH_BATCH_MAX_SEQS * super::ngram_bonus::NGRAM_TABLE_BYTES,
            )?;
        }
        if !scratch.ngram_prev3_dev.is_null() {
            gpu.memset(
                scratch.ngram_prev3_dev,
                0,
                super::DFLASH_BATCH_MAX_SEQS * super::ngram_bonus::NGRAM_PREV3_BYTES,
            )?;
        }

        // Capture the DSpark Markov head weights before `weights.layers` is
        // consumed below. Weights stay BF16 on device even under NVFP4 drafter
        // quantization — the head operates directly on the shared-vocab logit
        // space and the per-position GEMV (K=rank=256) is tiny, so quantizing
        // it would add complexity for no measurable win.
        let markov_weights: Option<MarkovWeights> = weights.markov;

        // Capture the DFlash 2 candidate-selector weights before
        // `weights.layers` is consumed below. All three tensors stay BF16 on
        // device (embedding codebooks + a small [rank, hidden] projection).
        let selector_weights: Option<DflashSelectorWeights> = weights.selector;

        // Pre-compute RoPE inv_freq table. Two paths based on the drafter's
        // `rope_scaling` config:
        //   - `Some(yarn)` → YaRN ramp interpolation (35B drafter)
        //   - `None`       → vanilla RoPE inv_freq (27B drafter — verified
        //                    via `z-lab/Qwen3.6-27B-DFlash/config.json` —
        //                    `rope_scaling: None`, `rope_theta: 10_000_000`)
        // Without this branch, the 27B drafter sees YaRN-rotated K/V and
        // attention scores diverge → ~3% accept rate (verified empirically).
        let rotary_dim = head_dim; // Qwen3.6-DFlash applies rope to full head_dim
        let n_pairs = rotary_dim / 2;
        let (rope_theta, rope_attention_factor, inv_freq_table) =
            dflash_rope_table(&weights.config, rotary_dim);
        if let Some(scaling) = weights.config.rope_scaling.as_ref()
            && scaling.rope_type.as_deref() == Some("yarn")
        {
            let factor = scaling.factor.unwrap_or(64.0);
            let beta_fast = scaling.beta_fast.unwrap_or(32.0);
            let beta_slow = scaling.beta_slow.unwrap_or(1.0);
            let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(4096) as f32;
            let find_correction_dim = |num_rot: f32| -> f32 {
                (rotary_dim as f32 * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                    / (2.0 * rope_theta.ln())
            };
            let low = find_correction_dim(beta_fast).floor().max(0.0);
            let high = find_correction_dim(beta_slow)
                .ceil()
                .min((rotary_dim - 1) as f32);
            tracing::info!(
                "DFlash YaRN inv_freq: {n_pairs} pairs, factor={factor}, attention_factor={rope_attention_factor}, \
                 beta_fast={beta_fast}, beta_slow={beta_slow}, \
                 max_pos={orig_max_pos}, low_dim={low:.1}, high_dim={high:.1}"
            );
        } else {
            tracing::info!(
                "DFlash vanilla RoPE inv_freq: {n_pairs} pairs, theta={rope_theta} \
                 (drafter has no rope_scaling)"
            );
        }
        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = alloc_tx.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;

        // Per-layer SWA window + causal flag (vLLM PR #40898 — Qwen3.6-27B-DFlash
        // has 4 sliding_attention + 1 full_attention layer).
        let (layer_window_sizes, layer_causal): (Vec<u32>, Vec<bool>) = match (
            weights.config.layer_types.as_ref(),
            weights.config.sliding_window,
        ) {
            (Some(types), Some(sw)) if types.len() == num_layers => {
                let mut sliding_count = 0usize;
                let mut windows = Vec::with_capacity(num_layers);
                let mut causals = Vec::with_capacity(num_layers);
                for t in types {
                    let is_sliding = t == "sliding_attention";
                    if is_sliding {
                        sliding_count += 1;
                        windows.push(sw as u32);
                    } else {
                        windows.push(0);
                    }
                    // Per-layer causal mask matching vLLM dflash.py:424-433.
                    // Sliding layers run causal + SWA (the drafter was
                    // trained with this mask); full-attention layers run
                    // non-causal across the [ctx, noise] block (bidirectional).
                    // Atlas's `inferspark_prefill_h128` kernel implements
                    // `(causal=true, sliding_window=W)` as the lower-triangular
                    // band `k >= q-W+1 AND k <= q`, matching flash-attn's
                    // `window_size=(W-1, 0)` semantics exactly — see
                    // `kernels/gb10/common/inferspark_prefill_h128.cu` and vLLM's
                    // flash-attn `(sliding_window - 1, 0)` window convention.
                    causals.push(is_sliding);
                }
                // A checkpoint that declares `"is_causal": false` at the top
                // level is telling us its draft block is attended
                // BIDIRECTIONALLY, and that declaration outranks the
                // `layer_types` heuristic above. DFlash2 ships exactly this,
                // with five `sliding_attention` layers and no
                // `full_attention` — so the heuristic makes every layer causal,
                // which is the opposite of how it was trained.
                //
                // An explicit checkpoint declaration is authoritative. Only
                // `Some(false)` changes the legacy layer-type heuristic, so a
                // checkpoint that omits the field keeps historical behavior.
                if weights.config.is_causal == Some(false) {
                    let was_causal = causals.iter().filter(|c| **c).count();
                    causals.iter_mut().for_each(|causal| *causal = false);
                    tracing::info!(
                        "DFlash checkpoint declares is_causal=false: forcing all \
                         {num_layers} drafter layers non-causal (was {was_causal} \
                         causal); per-layer SWA windows are unchanged"
                    );
                } else if weights.config.is_draft_block_causal() {
                    causals.iter_mut().for_each(|causal| *causal = true);
                }
                let causal_count = causals.iter().filter(|causal| **causal).count();
                let causal_mode = match causal_count {
                    0 => "noncausal",
                    count if count == num_layers => "causal",
                    _ => "mixed",
                };
                tracing::info!(
                    "DFlash per-layer SWA: {sliding_count}/{num_layers} layers \
                     use sliding_window={sw}; causal_layers={causal_count}/{num_layers} \
                     mode={causal_mode} causal={causals:?} checkpoint is_causal={:?}",
                    weights.config.is_causal,
                );
                (windows, causals)
            }
            (Some(types), None) if types.iter().any(|t| t == "sliding_attention") => {
                anyhow::bail!(
                    "DFlash drafter has sliding_attention layers but no \
                     `sliding_window` in config — refusing to silently treat \
                     as full-attention (would break drafts)"
                );
            }
            _ if weights.config.is_draft_block_causal() => {
                tracing::info!(
                    "DFlash checkpoint declares causal draft block attention across all \
                     {num_layers} full-attention layers"
                );
                (vec![0; num_layers], vec![true; num_layers])
            }
            _ => (Vec::new(), Vec::new()),
        };

        // Per-layer dimensions used for both the BF16 → NVFP4 quantization
        // (when requested) and downstream forward-pass kernel dispatch.
        let q_out = num_q_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        // Build the per-layer runtime variants. BF16 payloads are rewrapped
        // without GPU work. Under NVFP4, BF16 projections use the established
        // load-time quantizer while exact ModelOpt payloads pass through their
        // already-packed weights/scales directly. The top-level BF16 `fc`
        // projection is runtime-quantized in both NVFP4 cases.
        let (layers, fc_nvfp4, fc_fp8, fc_after, quantized_bf16_sources) = match quantization {
            DflashQuantization::Bf16 => {
                let layers: Vec<DflashLayerQuantWeights> = weights
                    .layers
                    .into_iter()
                    .map(|l| -> Result<_> {
                        let DflashLayerWeights {
                            input_layernorm,
                            post_attention_layernorm,
                            q_norm,
                            k_norm,
                            projections,
                            attention_conv,
                            mlp_conv,
                        } = l;
                        let DflashLayerProjections::Bf16 {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            gate_proj,
                            up_proj,
                            down_proj,
                        } = projections
                        else {
                            anyhow::bail!(
                                "pre-quantized ModelOpt DFlash2 projections require \
                                 --dflash-quantization nvfp4"
                            );
                        };
                        Ok(DflashLayerQuantWeights::Bf16(DflashLayer {
                            input_layernorm,
                            post_attention_layernorm,
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                            gate_proj,
                            up_proj,
                            down_proj,
                            attention_conv,
                            mlp_conv,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                (layers, None, None, weights.fc, Vec::new())
            }
            DflashQuantization::Fp8 => {
                let quantize_fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
                let qstream = gpu.default_stream();
                let target_hidden_dim = target_layer_ids.len() * target_hidden_size;

                let fc_q = quantize_to_fp8(
                    &weights.fc,
                    hidden_size,
                    target_hidden_dim,
                    gpu,
                    quantize_fp8_k,
                    qstream,
                )?;
                let mut quantized_bf16_sources = vec![weights.fc.weight];
                let fc_after = DenseWeight {
                    weight: DevicePtr::NULL,
                };

                let mut layers: Vec<DflashLayerQuantWeights> =
                    Vec::with_capacity(weights.layers.len());
                for l in weights.layers {
                    let DflashLayerWeights {
                        input_layernorm,
                        post_attention_layernorm,
                        q_norm,
                        k_norm,
                        projections,
                        attention_conv,
                        mlp_conv,
                    } = l;
                    let DflashLayerProjections::Bf16 {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        gate_proj,
                        up_proj,
                        down_proj,
                    } = projections
                    else {
                        anyhow::bail!(
                            "pre-quantized ModelOpt DFlash2 projections require \
                             --dflash-quantization nvfp4 (cannot re-quantize to fp8)"
                        );
                    };
                    let q =
                        quantize_to_fp8(&q_proj, q_out, hidden_size, gpu, quantize_fp8_k, qstream)?;
                    let k = quantize_to_fp8(
                        &k_proj,
                        kv_out,
                        hidden_size,
                        gpu,
                        quantize_fp8_k,
                        qstream,
                    )?;
                    let v = quantize_to_fp8(
                        &v_proj,
                        kv_out,
                        hidden_size,
                        gpu,
                        quantize_fp8_k,
                        qstream,
                    )?;
                    let o =
                        quantize_to_fp8(&o_proj, hidden_size, q_out, gpu, quantize_fp8_k, qstream)?;
                    let gate = quantize_to_fp8(
                        &gate_proj,
                        intermediate_size,
                        hidden_size,
                        gpu,
                        quantize_fp8_k,
                        qstream,
                    )?;
                    let up = quantize_to_fp8(
                        &up_proj,
                        intermediate_size,
                        hidden_size,
                        gpu,
                        quantize_fp8_k,
                        qstream,
                    )?;
                    let down = quantize_to_fp8(
                        &down_proj,
                        hidden_size,
                        intermediate_size,
                        gpu,
                        quantize_fp8_k,
                        qstream,
                    )?;
                    quantized_bf16_sources.extend([
                        q_proj.weight,
                        k_proj.weight,
                        v_proj.weight,
                        o_proj.weight,
                        gate_proj.weight,
                        up_proj.weight,
                        down_proj.weight,
                    ]);
                    layers.push(DflashLayerQuantWeights::Fp8(DflashLayerFp8Weights {
                        input_layernorm,
                        post_attention_layernorm,
                        q_norm,
                        k_norm,
                        q_proj: q,
                        k_proj: k,
                        v_proj: v,
                        o_proj: o,
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                        attention_conv,
                        mlp_conv,
                    }));
                }
                tracing::info!(
                    "DFlash drafter ready in online-quantized FP8 ({} layers x 7 projections + fc)",
                    layers.len()
                );
                (layers, None, Some(fc_q), fc_after, quantized_bf16_sources)
            }
            DflashQuantization::Nvfp4 => {
                // Resolve the two quantize kernels once — same module/symbol
                // names the target-model NVFP4 loaders use.
                let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
                let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
                let qstream = gpu.default_stream();
                let target_hidden_dim = target_layer_ids.len() * target_hidden_size;

                // ATLAS_DFLASH_FFN_KGAMMA=1: also build transposed (nvfp4_t)
                // copies of the per-layer FFN projections so the per-layer
                // forward can route gate/up/down through the M_TILE=16
                // specialization (w4a16_gemm_n128_m16). The transposed
                // weights are *additional* allocations alongside the
                // standard nvfp4 layout — the existing M_TILE=64 path
                // remains the fallback when the gate is off OR the M=16
                // kernel handle failed to resolve. One-time host-side
                // transpose cost at model build (per QuantizedWeight ~89MB
                // H2D+D2H round trip; drafter is 5 layers × 3 FFN projs ~=
                // 1.3 GB total). Disabled when ffn_kgamma kernel handle
                // failed to resolve (older cached PTX).
                let want_ffn_kgamma_t =
                    crate::layers::dflash_ffn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 != 0;
                if crate::layers::dflash_ffn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 == 0 {
                    tracing::warn!(
                        "ATLAS_DFLASH_FFN_KGAMMA=1 set but w4a16_gemm_t_m16 \
                         kernel symbol missing — drafter FFN will use the \
                         M_TILE=64 (w4a16_gemm) fallback. Rebuild the kernel \
                         cache to enable the M_TILE=16 specialization."
                    );
                }

                // ATLAS_DFLASH_ATTN_KGAMMA=1: same as the FFN kgamma toggle
                // above, but applied to the per-layer attention projections
                // (q/k/v/o). Drafter Q-proj observed ~5.5ms × 5 layers and
                // o_proj ~4-6ms × 5 layers under the M_TILE=64 path because
                // verify-row count (M ≈ 17) leaves 73% of accumulator slots
                // unused. Routing through M_TILE=16 reclaims that time. Adds
                // 4 additional `transpose_for_gemm` H↔D round trips per layer
                // at model build (~1.5 GB × 5 layers ≈ 7.5 GB transient host
                // memory, freed after each transpose). Disabled when the
                // kgamma kernel handle failed to resolve.
                let want_attn_kgamma_t =
                    crate::layers::dflash_attn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 != 0;
                if crate::layers::dflash_attn_kgamma_enabled() && kernels.w4a16_gemm_t_m16.0 == 0 {
                    tracing::warn!(
                        "ATLAS_DFLASH_ATTN_KGAMMA=1 set but w4a16_gemm_t_m16 \
                         kernel symbol missing — drafter attention will use \
                         the M_TILE=64 (w4a16_gemm) fallback. Rebuild the \
                         kernel cache to enable the M_TILE=16 specialization."
                    );
                }

                // Quantize the top-level `fc` projection first
                // (`[H, target_layer_ids.len() * target_hidden]`). N=H,
                // K=target_hidden_dim.
                let fc_q = quantize_to_nvfp4(
                    &weights.fc,
                    hidden_size,
                    target_hidden_dim,
                    gpu,
                    absmax_k,
                    quantize_k,
                    qstream,
                )?;
                let sm_count = gpu
                    .sm_count()
                    .filter(|&s| s > 0)
                    .unwrap_or_else(atlas_core::device::active_sm_count);
                if sm_count > 64 {
                    crate::layers::dense_ffn::warm_coalesced_nvfp4_sm120_kernels(gpu);
                }
                let has_coalesced_bf16 =
                    crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_coalesced_n16_sk4_m16").0
                        != 0
                        || crate::layers::try_kernel(
                            gpu,
                            "w4a16",
                            "w4a16_gemm_coalesced_v2_n32_m16",
                        )
                        .0 != 0;
                let want_fc_kgamma_t = should_build_fc_nvfp4_t(FcTransposePolicyInput {
                    want_ffn_kgamma_t,
                    want_attn_kgamma_t,
                    sm_count,
                    has_m32_n64: kernels.w4a16_gemm_t_m32_n64.0 != 0,
                    has_m16_n128: kernels.w4a16_gemm_t_m16.0 != 0,
                    has_coalesced_bf16,
                    hidden_size,
                    target_hidden_dim,
                });
                if want_fc_kgamma_t {
                    let fc_t = fc_q.transpose_for_gemm(gpu, hidden_size, target_hidden_dim)?;
                    register_fc_nvfp4_t(fc_q.weight, fc_t);
                }
                // BUG #29: skip free; leave a placeholder so callers can't
                // accidentally dispatch the BF16 path. The BF16 sources are
                // reported in `quantized_bf16_sources` so the factory can
                // release them in its quiescent window
                // (`ATLAS_MEMLEAN_DRAFTER_BF16`).
                let mut quantized_bf16_sources = vec![weights.fc.weight];
                let fc_after = DenseWeight {
                    weight: DevicePtr::NULL,
                };

                // Quantize each per-layer dense projection.
                let mut layers: Vec<DflashLayerQuantWeights> =
                    Vec::with_capacity(weights.layers.len());
                let mut direct_modelopt_layers = 0usize;
                for (layer_idx, l) in weights.layers.into_iter().enumerate() {
                    let DflashLayerWeights {
                        input_layernorm,
                        post_attention_layernorm,
                        q_norm,
                        k_norm,
                        projections,
                        attention_conv,
                        mlp_conv,
                    } = l;
                    let (q, k, v, o, gate, up, down, input_scales) = match projections {
                        DflashLayerProjections::Bf16 {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            gate_proj,
                            up_proj,
                            down_proj,
                        } => {
                            let q = quantize_to_nvfp4(
                                &q_proj,
                                q_out,
                                hidden_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let k = quantize_to_nvfp4(
                                &k_proj,
                                kv_out,
                                hidden_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let v = quantize_to_nvfp4(
                                &v_proj,
                                kv_out,
                                hidden_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let o = quantize_to_nvfp4(
                                &o_proj,
                                hidden_size,
                                q_out,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let gate = quantize_to_nvfp4(
                                &gate_proj,
                                intermediate_size,
                                hidden_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let up = quantize_to_nvfp4(
                                &up_proj,
                                intermediate_size,
                                hidden_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            let down = quantize_to_nvfp4(
                                &down_proj,
                                hidden_size,
                                intermediate_size,
                                gpu,
                                absmax_k,
                                quantize_k,
                                qstream,
                            )?;
                            // DON'T free BF16 source buffers here — BUG #29 (see
                            // weight_loader/nemotron.rs): GB10 UVM free can post
                            // in-band TLB invalidations that corrupt neighbours.
                            // The factory may release them after a device sync.
                            quantized_bf16_sources.extend([
                                q_proj.weight,
                                k_proj.weight,
                                v_proj.weight,
                                o_proj.weight,
                                gate_proj.weight,
                                up_proj.weight,
                                down_proj.weight,
                            ]);
                            (q, k, v, o, gate, up, down, None)
                        }
                        DflashLayerProjections::ModelOptNvfp4 {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            gate_proj,
                            up_proj,
                            down_proj,
                        } => {
                            direct_modelopt_layers += 1;
                            let input_scales = if w4a4_enabled {
                                let scales = super::w4a4_admission::DflashNvfp4InputScales::read(
                                    gpu, layer_idx, &q_proj, &k_proj, &v_proj, &o_proj, &gate_proj,
                                    &up_proj, &down_proj,
                                )?;
                                tracing::info!(
                                    "DFLASH2_W4A4_SCALE_RECEIPT layer={} input_scale_bits={:08x?}",
                                    layer_idx,
                                    scales.receipt_bits(),
                                );
                                Some(scales)
                            } else {
                                None
                            };
                            (
                                q_proj,
                                k_proj,
                                v_proj,
                                o_proj,
                                gate_proj,
                                up_proj,
                                down_proj,
                                input_scales,
                            )
                        }
                    };

                    // Build transposed FFN weights when ATLAS_DFLASH_FFN_KGAMMA=1.
                    // For each projection: N = output dim, K = input dim
                    // (matches the (n, k) parameters of transpose_for_gemm:
                    // see qwen35_dense.rs:232/244 for the established pattern).
                    let (gate_t, up_t, down_t) = if want_ffn_kgamma_t {
                        // gate: [intermediate, hidden]; up: [intermediate, hidden]; down: [hidden, intermediate].
                        let gt = gate.transpose_for_gemm(gpu, intermediate_size, hidden_size)?;
                        let ut = up.transpose_for_gemm(gpu, intermediate_size, hidden_size)?;
                        let dt = down.transpose_for_gemm(gpu, hidden_size, intermediate_size)?;
                        (Some(gt), Some(ut), Some(dt))
                    } else {
                        (None, None, None)
                    };

                    // Build transposed attention weights when
                    // ATLAS_DFLASH_ATTN_KGAMMA=1. Same (n=output, k=input)
                    // convention as the FFN block above.
                    // q: [q_out, hidden]; k: [kv_out, hidden]; v: [kv_out, hidden];
                    // o: [hidden, q_out].
                    let (q_t, k_t, v_t, o_t) = if want_attn_kgamma_t {
                        let qt = q.transpose_for_gemm(gpu, q_out, hidden_size)?;
                        let kt = k.transpose_for_gemm(gpu, kv_out, hidden_size)?;
                        let vt = v.transpose_for_gemm(gpu, kv_out, hidden_size)?;
                        let ot = o.transpose_for_gemm(gpu, hidden_size, q_out)?;
                        (Some(qt), Some(kt), Some(vt), Some(ot))
                    } else {
                        (None, None, None, None)
                    };

                    layers.push(DflashLayerQuantWeights::Nvfp4(DflashLayerNvfp4 {
                        input_layernorm,
                        post_attention_layernorm,
                        q_norm,
                        k_norm,
                        q_proj: q,
                        k_proj: k,
                        v_proj: v,
                        o_proj: o,
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                        input_scales,
                        gate_proj_t: gate_t,
                        up_proj_t: up_t,
                        down_proj_t: down_t,
                        attention_conv,
                        mlp_conv,
                        q_proj_t: q_t,
                        k_proj_t: k_t,
                        v_proj_t: v_t,
                        o_proj_t: o_t,
                    }));
                }

                super::w4a4_admission::validate_complete_layers(
                    w4a4_enabled,
                    direct_modelopt_layers,
                    num_layers,
                )?;

                tracing::info!(
                    "DFlash drafter ready in NVFP4 ({} layers × 7 projections + runtime-quantized \
                     BF16 fc); direct_modelopt_layers={direct_modelopt_layers}, \
                     runtime_quantized_layers={}; calibrated_w4a4_shadow={}; \
                     ffn_kgamma_t={} attn_kgamma_t={} fc_kgamma_t={}",
                    layers.len(),
                    layers.len() - direct_modelopt_layers,
                    w4a4_enabled,
                    want_ffn_kgamma_t,
                    want_attn_kgamma_t,
                    want_fc_kgamma_t,
                );
                (layers, Some(fc_q), None, fc_after, quantized_bf16_sources)
            }
        };

        // DSpark VanillaMarkov head (runtime form). Weights are BF16 on device
        // (loaded by the drafter weight loader); we just carry the pointers +
        // rank. The scratch buffers above are already sized for `rank`/`vocab`.
        let markov = markov_weights.map(|m: MarkovWeights| super::MarkovHead {
            w1: m.w1,
            w2: m.w2,
            rank: m.rank,
        });

        // ── ATLAS_DFLASH_LM_HEAD_FP8=1: pre-scaled E4M3 copy of the propose
        // lm_head slice (see the field doc on `lm_head_shared_fp8`). One-time
        // load cost: absmax scan + scaled BF16 copy + FP8 cast over
        // lm_vocab × K (~0.5 GB result, ~1 GB transient). The compensating
        // 1/s stays paired with the FP8 head. The canonical final `norm` must
        // remain unscaled because the independently selected NVFP4/BF16 heads
        // and the DFlash2 candidate selector consume its activations too.
        let mut lm_head_shared_fp8 = None;
        let norm = weights.norm;
        if std::env::var("ATLAS_DFLASH_LM_HEAD_FP8").ok().as_deref() == Some("1") {
            let absmax_k = crate::layers::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax");
            let tofp8_k = crate::layers::try_kernel(gpu, "w4a16", "bf16_to_fp8");
            if kernels.fp8_gemm_t.0 == 0 || absmax_k.0 == 0 || tofp8_k.0 == 0 {
                tracing::warn!(
                    "ATLAS_DFLASH_LM_HEAD_FP8=1 but fp8_gemm_t/absmax/bf16_to_fp8 \
                     kernel symbols missing — propose lm_head stays BF16"
                );
            } else {
                let n = target_vocab_size.min(vocab_size);
                let k = target_hidden_size;
                let total = (n * k) as u32;
                let qstream = gpu.default_stream();
                let absmax = {
                    let mut gmax_alloc = crate::device_allocations::DeviceAllocations::new(gpu);
                    let gmax = gmax_alloc.alloc(4)?;
                    gpu.memset(gmax, 0, 4)?;
                    crate::layers::ops::nvfp4_global_absmax(
                        gpu,
                        absmax_k,
                        lm_head_shared,
                        gmax,
                        total,
                        qstream,
                    )?;
                    gpu.synchronize(qstream)?;
                    let mut b4 = [0u8; 4];
                    gpu.copy_d2h(gmax, &mut b4)?;
                    gpu.free(gmax)?;
                    gmax_alloc.commit();
                    f32::from_le_bytes(b4)
                };
                // Power-of-2 scale targeting absmax·s ≈ 256 (E4M3 max 448):
                // exact in BF16 and in the folded norm, so the only lossy
                // step is the E4M3 cast itself.
                let s = if absmax > 0.0 {
                    (256.0f32 / absmax).log2().floor().exp2()
                } else {
                    1.0
                };
                let fp8 = {
                    let mut tmp_alloc = crate::device_allocations::DeviceAllocations::new(gpu);
                    let tmp = tmp_alloc.alloc(n * k * 2)?;
                    gpu.memset(tmp, 0, n * k * 2)?;
                    crate::layers::ops::scaled_add(
                        gpu,
                        kernels.scaled_add,
                        tmp,
                        lm_head_shared,
                        s,
                        total,
                        qstream,
                    )?;
                    let fp8 = alloc_tx.alloc(n * k)?;
                    crate::layers::ops::bf16_to_fp8(gpu, tofp8_k, tmp, fp8, total, qstream)?;
                    gpu.synchronize(qstream)?;
                    gpu.free(tmp)?;
                    tmp_alloc.commit();
                    fp8
                };
                let norm_scaled = alloc_tx.alloc(hidden_size * 2)?;
                gpu.memset(norm_scaled, 0, hidden_size * 2)?;
                crate::layers::ops::scaled_add(
                    gpu,
                    kernels.scaled_add,
                    norm_scaled,
                    norm.weight,
                    1.0 / s,
                    hidden_size as u32,
                    qstream,
                )?;
                gpu.synchronize(qstream)?;
                lm_head_shared_fp8 = Some(DflashFp8LmHead {
                    weight: fp8,
                    input_norm: DenseWeight {
                        weight: norm_scaled,
                    },
                });
                tracing::info!(
                    "DFlash propose lm_head FP8: [{n} x {k}] absmax={absmax:.4} \
                     scale=2^{} (1/s bound to FP8 input norm)",
                    s.log2() as i32,
                );
            }
        }

        let dspine = if let Some(dspine_weights) = weights.dspine {
            let lm_vocab = target_vocab_size.min(vocab_size);
            Some(super::dspine::DspineCodeBridge::build(
                dspine_weights,
                lm_head_shared,
                &norm,
                num_layers,
                hidden_size,
                lm_vocab,
                gamma_val + 1,
                gpu,
            )?)
        } else {
            None
        };

        // Batched DFlash2 exclusively reads/writes per-layer PostRope
        // `ctx_{k,v}_cache` and never touches per-sequence `ctx_fc_cache`
        // (saving 10-20 MiB/seq across concurrent cohorts).
        let skip_ctx_fc_cache = selector_weights.is_some()
            && crate::layers::qwen4_verify_lanes::batched_k5_selected().unwrap_or(false);

        let head = Self {
            checkpoint_family,
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            draft_sample_method,
            physical_verify_k,
            mask_token_id,
            window_size,
            layer_window_sizes,
            layer_causal,
            target_layer_ids,
            target_hidden_size,
            target_vocab_size,

            embed_tokens_shared,
            lm_head_shared,
            lm_head_shared_t: None,
            lm_head_shared_t_ldb: 0,
            lm_head_shared_nvfp4_exact: None,
            lm_head_exact_kernels: None,
            lm_head_shared_fp8,
            hidden_norm: weights.hidden_norm,
            norm,
            fc: fc_after,
            fc_nvfp4,
            fc_fp8,
            quantized_bf16_sources,
            draft_id_to_target_id: None,
            markov,
            selector: selector_weights,
            dspine,
            layers,
            w4a4_shadow,
            scratch,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_attention_factor,
            rope_theta,
            rotary_dim,
            rms_norm_eps: weights.config.resolved_rms_norm_eps(),
            ctx_window,
            proposal_policy: std::sync::OnceLock::new(),
            quant: quantization,
            async_inflight: Mutex::new(None),
            async_batched_inflight: Mutex::new(None),
            async_batch_slot_toggle: std::sync::atomic::AtomicU8::new(0),
            early_stage1_staged: Mutex::new(None),
            opt_branch0_staged: Mutex::new(None),
            async_propose_stream: std::sync::OnceLock::new(),
            async_order_event: std::sync::atomic::AtomicU64::new(0),
            fused_event_armed: std::sync::atomic::AtomicBool::new(false),
            batched_noise_graph: Mutex::new(super::batched_propose::BatchedNoiseGraphCache::new()),
            q_row_bases_released: std::sync::atomic::AtomicBool::new(false),
            skip_ctx_fc_cache,
            target_is_moe: None,
        };

        tracing::info!(
            "BlockDiffusionDraftHead loaded: family={}, checkpoint_block_size={}, verify_width={}, \
             {} layers, hidden={}, intermediate={}, GQA {}/{}, head_dim={}, γ={}, vocab={}, \
             mask_token_id={}, target_layers={:?}",
            checkpoint_family,
            block_size,
            head.gamma + 1,
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.target_layer_ids,
        );

        alloc_tx.commit();
        Ok(head)
    }

    /// Borrow-validate the drafter dimensions against the target's hidden_size
    /// at construction time. Mismatch is a hard error — the `fc` projection
    /// width is baked from `target_hidden_size` and a runtime mismatch would
    /// produce silent garbage (vLLM's loader hits this same check).
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }

    /// Synchronize the drafter's pre-allocated `yarn_inv_freq` table and
    /// `rope_attention_factor` when attached to a `qwen4_exp` target serving
    /// extended YaRN context (`--rope-yarn-factor > 0`).
    ///
    /// Qwen3.8-Flash-Next DFlash/DFlash2 checkpoints serialize `"rope_type": "default"`
    /// at the 262K native horizon; without propagating the target's runtime YaRN
    /// ramp and frequency-only `attention_factor = 1.0`, the drafter rotates K/Q
    /// with unscaled frequencies while target verification uses YaRN.
    pub(crate) fn sync_target_qwen4_yarn(
        &mut self,
        target_config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if !target_config.is_qwen4_exp() || target_config.yarn_factor <= 0.0 {
            return Ok(());
        }
        let inv_freq_table = crate::layers::qwen4_qsa::qwen4_yarn_inv_freq_values_with_theta(
            target_config,
            self.rope_theta,
            self.rotary_dim,
        )?;
        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        gpu.copy_h2d(&inv_freq_bytes, self.yarn_inv_freq)?;
        self.rope_attention_factor = 1.0;
        tracing::info!(
            "DFlash synced target Qwen4 YaRN inv_freq: {} pairs, factor={}, attention_factor=1.0",
            inv_freq_table.len(),
            target_config.yarn_factor,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::dflash_rope_table;
    use crate::weight_loader::dflash_loader::parse_dflash_config;
    use atlas_core::config::ModelConfig;

    #[test]
    fn qwen38_rope_parameters_match_transformers_yarn_fixture() {
        // Frozen from qwen38/dspark-drafter/config.json. Reference values were
        // produced by Transformers 5.5 Qwen3RotaryEmbedding on 2026-08-15.
        let json = r#"{
            "hidden_size": 5120,
            "num_hidden_layers": 5,
            "intermediate_size": 10240,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 10000000
            }
        }"#;
        let config = parse_dflash_config(json).expect("parse Qwen3.8 config dialect");
        assert!(
            config.rope_scaling.is_some(),
            "rope_parameters alias was ignored"
        );

        let (theta, attention_factor, inv) = dflash_rope_table(&config, 128);
        assert_eq!(theta, 10_000_000.0);
        assert!((attention_factor - 1.346_573_6).abs() < 1e-6);
        assert!((inv[0] - 1.0).abs() < 1e-7);
        assert!((inv[16] - 0.015_485_85).abs() < 1e-8);
        assert!((inv[24] - 0.000_839_861_5).abs() < 1e-10);
        assert!((inv[32] - 0.000_009_882_118).abs() < 1e-11);
        assert!((inv[63] - 0.000_000_004_019_990_6).abs() < 1e-14);
    }

    #[test]
    fn qwen4_flash_next_dflash_yarn_uses_unit_attention_factor_and_exact_target_ramp() {
        let json = r#"{
            "architectures": ["Qwen3DSparkModel"],
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 5,
            "num_target_layers": 48,
            "intermediate_size": 7680,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "block_size": 7,
            "mask_token_id": 248077,
            "target_layer_ids": [3, 15, 23, 35, 43],
            "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
            ],
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 262144,
                "rope_theta": 10000000
            }
        }"#;
        let dflash_cfg = parse_dflash_config(json).expect("parse Qwen4 DFlash config");
        let (theta, attention_factor, dflash_inv) = dflash_rope_table(&dflash_cfg, 256);
        assert_eq!(theta, 10_000_000.0);
        assert_eq!(attention_factor, 1.0);

        let mut target_cfg = ModelConfig::qwen3_next_80b_nvfp4();
        target_cfg.model_type = "qwen4_exp".into();
        target_cfg.rope_theta = 10_000_000.0;
        target_cfg.yarn_factor = 4.0;
        target_cfg.yarn_beta_fast = 32.0;
        target_cfg.yarn_beta_slow = 1.0;
        target_cfg.yarn_original_max_position_embeddings = 262_144;
        let synced_inv = crate::layers::qwen4_qsa::qwen4_yarn_inv_freq_values_with_theta(
            &target_cfg,
            theta,
            256,
        )
        .unwrap();
        assert_eq!(dflash_inv.len(), synced_inv.len());
        for (a, b) in dflash_inv.iter().zip(synced_inv.iter()) {
            assert!((a - b).abs() <= 1e-9, "mismatch {a} vs {b}");
        }
    }

    #[test]
    fn k32_selector_and_out_probs_scratch_sizing_cover_c1_and_batched() {
        use super::{batch_out_probs_scratch_bytes, selector_topk_scratch_bytes};

        // C=1 (gamma=7 or 15) and batched (up to 256 output rows: C=32 x K=8) both fit K=32 (4 B/elem).
        assert_eq!(selector_topk_scratch_bytes(7), 256 * 32 * 4);
        assert_eq!(selector_topk_scratch_bytes(15), 256 * 32 * 4);
        assert_eq!(selector_topk_scratch_bytes(300), 300 * 32 * 4);

        // `batch_out_probs` has at least `64 * 8 * 4 = 2048` B for C=1..64.
        assert_eq!(batch_out_probs_scratch_bytes(7), 64 * 8 * 4);
        assert_eq!(batch_out_probs_scratch_bytes(15), 64 * 8 * 4);
        assert_eq!(batch_out_probs_scratch_bytes(600), 600 * 4);
    }

    #[test]
    fn all_layers_sliding_window_honors_2048_for_incoai_dflash2_and_preserves_full_attention_4096()
    {
        use super::dflash_context_window::{
            effective_all_layers_sliding_window, resolve_context_window_with_sliding,
        };

        let all_sliding: Vec<String> = vec!["sliding_attention".to_owned(); 5];
        let mixed: Vec<String> = vec![
            "sliding_attention".to_owned(),
            "sliding_attention".to_owned(),
            "sliding_attention".to_owned(),
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
        ];

        assert_eq!(
            effective_all_layers_sliding_window(Some(&all_sliding), Some(2048), 5),
            Some(2048)
        );
        assert_eq!(
            effective_all_layers_sliding_window(Some(&mixed), Some(2048), 5),
            None
        );
        assert_eq!(
            effective_all_layers_sliding_window(Some(&all_sliding), None, 5),
            None
        );

        // incoai-dflash2-b8 (native_flash_next=true, all-sliding 2048):
        // default CLI 4096 and explicit 2048 resolve to 2048; bounded
        // power-of-two sub-windows (256/512/1024) are admitted for wide
        // discrete cohorts (C=48..64) while keeping verify lossless.
        assert_eq!(
            resolve_context_window_with_sliding(Some(4096), 262_144, true, Some(2048)),
            Ok(2048)
        );
        assert_eq!(
            resolve_context_window_with_sliding(Some(2048), 262_144, true, Some(2048)),
            Ok(2048)
        );
        assert_eq!(
            resolve_context_window_with_sliding(Some(1024), 4096, true, Some(2048)),
            Ok(1024)
        );
        assert_eq!(
            resolve_context_window_with_sliding(Some(512), 2048, true, Some(2048)),
            Ok(512)
        );
        assert_eq!(
            resolve_context_window_with_sliding(Some(256), 2048, true, Some(2048)),
            Ok(256)
        );
        // Arbitrary mismatched windows still fail closed.
        assert!(resolve_context_window_with_sliding(Some(300), 262_144, true, Some(2048)).is_err());
        assert!(
            resolve_context_window_with_sliding(Some(1536), 262_144, true, Some(2048)).is_err()
        );
        // Full-attention native Flash-Next (all_layers_sliding_window=None) still requires 4096.
        assert_eq!(
            resolve_context_window_with_sliding(Some(4096), 262_144, true, None),
            Ok(4096)
        );
        assert!(resolve_context_window_with_sliding(Some(2048), 262_144, true, None).is_err());
    }

    #[test]
    fn fc_nvfp4_t_policy_avoids_e4m3_clipping_on_sm120_and_coalesced_bf16() {
        use super::{
            FcTransposePolicyInput, lookup_fc_nvfp4_t, register_fc_nvfp4_t, should_build_fc_nvfp4_t,
        };
        use crate::weight_map::QuantizedWeight;
        use spark_runtime::gpu::DevicePtr;

        // GB10 (48 SMs) with kgamma flags unset -> false (zero regression).
        assert!(!should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: false,
            want_attn_kgamma_t: false,
            sm_count: 48,
            has_m32_n64: true,
            has_m16_n128: true,
            has_coalesced_bf16: false,
            hidden_size: 5120,
            target_hidden_dim: 25600,
        }));

        // GB10 with explicit kgamma opt-in and no coalesced BF16 kernels -> true.
        assert!(should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: true,
            want_attn_kgamma_t: false,
            sm_count: 48,
            has_m32_n64: true,
            has_m16_n128: true,
            has_coalesced_bf16: false,
            hidden_size: 5120,
            target_hidden_dim: 25600,
        }));

        // Coalesced BF16xBF16 MMA available -> never build E4M3-clamped fc_nvfp4_t.
        assert!(!should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: true,
            want_attn_kgamma_t: true,
            sm_count: 48,
            has_m32_n64: true,
            has_m16_n128: true,
            has_coalesced_bf16: true,
            hidden_size: 5120,
            target_hidden_dim: 25600,
        }));

        // SM120 (188 SMs) -> false so un-normalized fc stays on unclipped BF16 MMA
        // and saves 73.7 MB VRAM.
        assert!(!should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: true,
            want_attn_kgamma_t: true,
            sm_count: 188,
            has_m32_n64: true,
            has_m16_n128: false,
            has_coalesced_bf16: false,
            hidden_size: 5120,
            target_hidden_dim: 25600,
        }));

        // Misaligned N or K fails closed.
        assert!(!should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: true,
            want_attn_kgamma_t: true,
            sm_count: 48,
            has_m32_n64: true,
            has_m16_n128: true,
            has_coalesced_bf16: false,
            hidden_size: 5120,
            target_hidden_dim: 25616,
        }));
        assert!(!should_build_fc_nvfp4_t(FcTransposePolicyInput {
            want_ffn_kgamma_t: true,
            want_attn_kgamma_t: true,
            sm_count: 48,
            has_m32_n64: false,
            has_m16_n128: true,
            has_coalesced_bf16: false,
            hidden_size: 5184, // divisible by 64 but not 128
            target_hidden_dim: 25600,
        }));

        // Sidecar registry round-trip (synthetic non-zero device pointers).
        let key = DevicePtr(0xDEAD_BEEF_0000_1000);
        let qw = QuantizedWeight {
            weight: DevicePtr(0xCAFE_BABE_0000_2000),
            weight_scale: DevicePtr(0xCAFE_BABE_0000_3000),
            weight_scale_2: 0.25,
            input_scale: DevicePtr::NULL,
        };
        register_fc_nvfp4_t(key, qw);
        let found = lookup_fc_nvfp4_t(key).expect("registered fc_nvfp4_t must be found");
        assert_eq!(found.weight, qw.weight);
        assert_eq!(found.weight_scale, qw.weight_scale);
        assert_eq!(found.weight_scale_2.to_bits(), qw.weight_scale_2.to_bits());
        assert!(lookup_fc_nvfp4_t(DevicePtr::NULL).is_none());
    }

    #[test]
    fn from_weights_honors_config_rms_norm_eps_bf16_ones_fc_norm_and_conv_without_selector() {
        use crate::dflash_sampling::DflashDraftSampleMethod;
        use crate::layers::dflash_head::{BlockDiffusionDraftHead, DflashQuantization};
        use crate::weight_loader::{
            DflashConvWeights, DflashLayerProjections, DflashLayerWeights, DflashWeights,
            DsparkVerifyMode,
        };
        use crate::weight_map::DenseWeight;
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};

        let json = r#"{
            "architectures": ["DFlashDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_target_layers": 4,
            "intermediate_size": 512,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": 64,
            "block_size": 4,
            "rms_norm_eps": 1e-5,
            "dflash_config": {
                "target_layer_ids": [1],
                "mask_token_id": 63
            }
        }"#;
        let config = parse_dflash_config(json).expect("parse minimal DFlash config");
        let w = DenseWeight {
            weight: DevicePtr(0x1000),
        };
        let layer = DflashLayerWeights {
            input_layernorm: w,
            post_attention_layernorm: w,
            q_norm: w,
            k_norm: w,
            projections: DflashLayerProjections::Bf16 {
                q_proj: w,
                k_proj: w,
                v_proj: w,
                o_proj: w,
                gate_proj: w,
                up_proj: w,
                down_proj: w,
            },
            attention_conv: Some(DflashConvWeights {
                base_kernel: w,
                kernel_projection: w,
                kernel_size: 2,
                groups: 4,
                group_size: 64,
            }),
            mlp_conv: None,
        };
        let weights = DflashWeights {
            config,
            fc: w,
            fc_shape: [256, 256],
            hidden_norm: w,
            norm: w,
            layers: vec![layer],
            draft_id_to_target_id: None,
            markov: None,
            confidence: None,
            selector: None,
            dspine: None,
        };
        let gpu = MockGpuBackend::new();
        let head = BlockDiffusionDraftHead::from_weights(
            weights,
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            256,
            64,
            None,
            4,
            DsparkVerifyMode::Static,
            Some(4),
            &gpu,
            16,
            DflashQuantization::Bf16,
            DflashDraftSampleMethod::Greedy,
            false,
        )
        .expect("construct head from weights");

        // H-035: checkpoint rms_norm_eps is preserved instead of hardcoded 1e-6.
        assert_eq!(head.rms_norm_eps, 1e-5);

        // H-028: fc_norm_ones_w holds BF16 1.0 (0x3F80) for rms_norm_vanilla.
        let mut fc_w = vec![0u8; 256 * 2];
        gpu.copy_d2h(head.scratch.fc_norm_ones_w, &mut fc_w)
            .expect("read fc_norm_ones_w");
        for chunk in fc_w.chunks_exact(2) {
            assert_eq!(u16::from_le_bytes([chunk[0], chunk[1]]), 0x3F80);
        }

        // H-038: conv kernels and scratch resolve whenever any layer has conv,
        // even when `selector` is None.
        assert_ne!(head.kernels.dflash2_conv_prepare.0, 0);
        assert_ne!(head.kernels.dflash2_conv_finish.0, 0);
        assert!(!head.scratch.conv_dyn.is_null());
        assert!(!head.scratch.conv_dyn1.is_null());
        assert!(!head.scratch.conv_out.is_null());
    }
}
