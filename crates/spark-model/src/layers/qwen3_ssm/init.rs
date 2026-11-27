// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3SsmLayer constructors + setters.

use super::*;

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_scale_fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_ssm_projection(
    gpu: &dyn GpuBackend,
    label: &str,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<FlashinferSsmProjection> {
    use crate::weight_map::cutlass_scale_layout::{
        NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
    };
    use anyhow::Context as _;

    ensure!(
        !weight.weight.is_null() && weight.weight.0.is_multiple_of(16),
        "FlashInfer SSM {label} packed weight is null or misaligned"
    );
    ensure!(
        !weight.weight_scale.is_null(),
        "FlashInfer SSM {label} logical weight scales are null"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer SSM {label} has invalid N/K geometry"
    );
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "FlashInfer SSM {label} weight_scale_2 must be finite and positive"
    );

    let groups = k / NVFP4_GROUP_SIZE;
    let scale_len = n
        .checked_mul(groups)
        .context("FlashInfer SSM logical weight-scale length overflow")?;
    let mut logical_scales = vec![0_u8; scale_len];
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .with_context(|| format!("read FlashInfer SSM {label} logical weight scales"))?;
    let physical_scales =
        interleave_nvfp4_scales_128x4(&logical_scales, &[n, groups], NVFP4_GROUP_SIZE)?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(&physical_scales, &[n, groups], NVFP4_GROUP_SIZE,)?
            == logical_scales,
        "FlashInfer SSM {label} physical weight scales failed exact round trip"
    );

    let weight_scales_128x4 = gpu
        .alloc(physical_scales.len())
        .with_context(|| format!("allocate FlashInfer SSM {label} physical weight scales"))?;
    if let Err(error) = gpu.copy_h2d(&physical_scales, weight_scales_128x4) {
        let _ = gpu.free(weight_scales_128x4);
        return Err(error)
            .with_context(|| format!("upload FlashInfer SSM {label} physical weight scales"));
    }

    Ok(FlashinferSsmProjection {
        weight: weight.weight,
        weight_scales_128x4,
        weight_scales_hash: flashinfer_scale_fingerprint(&physical_scales),
        weight_scale_2: weight.weight_scale_2,
        n,
        k,
    })
}

impl Qwen3SsmLayer {
    pub fn new(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let nv = config.linear_num_value_heads;
        let vd = config.linear_value_head_dim;
        let nk = config.linear_num_key_heads;
        let kd = config.linear_key_head_dim;
        let d_conv = config.linear_conv_kernel_dim;

        // conv_dim = Q_flat + K_flat + V_flat = 2*key_dim + value_dim = 8192
        let conv_dim = nk * kd * 2 + nv * vd;
        // Allocate fused K=1 / verify completion counters for Qwen4 and for
        // matching sigmoid-gated architectures eligible on high-SM GPUs
        // (mirrors `fused_k1_model_eligible` in `ssm_forward.rs`).
        let alloc_fused_done = config.is_qwen4_exp()
            || (config.output_gate_type == "sigmoid"
                && nk == 16
                && nv == 48
                && kd == 128
                && vd == 128
                && d_conv == 4);
        let gdn_fused_split_done_stride = nv.max(64);
        let (gdn_fused_group_done, gdn_fused_split_done) = if alloc_fused_done {
            let group_bytes =
                2 * crate::layers::qwen4_verify_lanes::MAX_LANES * 32 * std::mem::size_of::<u32>();
            let group_ptr = gpu.alloc(group_bytes)?;
            if let Err(err) = gpu.memset(group_ptr, 0, group_bytes) {
                let _ = gpu.free(group_ptr);
                return Err(err);
            }
            let split_bytes = 2
                * crate::layers::qwen4_verify_lanes::MAX_LANES
                * gdn_fused_split_done_stride
                * std::mem::size_of::<u32>();
            let split_ptr = match gpu.alloc(split_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(group_ptr);
                    return Err(err);
                }
            };
            if let Err(err) = gpu.memset(split_ptr, 0, split_bytes) {
                let _ = gpu.free(split_ptr);
                let _ = gpu.free(group_ptr);
                return Err(err);
            }
            (group_ptr, split_ptr)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        Ok(Self {
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qwen4_attn_hyper: None,
            qwen4_mlp_hyper: None,
            qkvz_nvfp4,
            qkvz_nvfp4_t: None,
            out_proj_nvfp4_t: None,
            out_proj_dense: None,
            out_proj_dense_decode_only: None,
            qkvz_dense_decode_only: None,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            flashinfer_ssm_prefill: None,
            qkvz_fp8w: None,
            out_proj_fp8w: None,
            qwen4_dense_fp8: None,
            sequential_qkvz: false,
            rms_norm_residual_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "rms_norm_residual_f32")
                    .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            gated_rms_norm_k: if config.output_gate_type == "sigmoid" {
                gpu.kernel("qwen4_hyper", "qwen4_gated_rms_norm_sigmoid")?
            } else {
                gpu.kernel("norm", "gated_rms_norm")?
            },
            gated_rms_norm_f32_k: if config.output_gate_type == "sigmoid" {
                super::super::try_kernel(gpu, "qwen4_hyper", "qwen4_gated_rms_norm_sigmoid_f32")
            } else {
                super::super::try_kernel(gpu, "norm", "gated_rms_norm_f32_input")
            },
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            // Optional K=3 batched BA-proj GEMV. Built into the common
            // nvfp4/dense_gemv_bf16.cu so every target picks it up, but
            // use `try_kernel` for safety — older PTX bundles or future
            // model dirs that shadow `gemv` may not include it.
            dense_gemv_batch3_k: super::super::try_kernel(gpu, "gemv", "dense_gemv_bf16_batch3"),
            // General batched BA-proj GEMV (grid.y = token). Bit-identical
            // to per-token dense_gemv_bf16. NULL on older PTX bundles.
            dense_gemv_batchn_k: super::super::try_kernel(gpu, "gemv", "dense_gemv_bf16_batchn"),
            ba_gates_batchn_exact_k: super::super::try_kernel(
                gpu,
                "ssm_preprocess",
                "dense_gemv_ba_gates_batchn",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_exact_projection_kernels: ops::W4a16ExactLmHeadKernels::new(
                super::super::try_kernel(gpu, "w4a16_gemv", ops::ExactLmHeadTier::M4.symbol()),
                super::super::try_kernel(gpu, "w4a16_gemv", ops::ExactLmHeadTier::M8.symbol()),
                super::super::try_kernel(gpu, "w4a16_gemv", ops::ExactLmHeadTier::M17.symbol()),
                super::super::try_kernel(gpu, "w4a16_gemv", ops::ExactLmHeadTier::M32.symbol()),
            )
            .with_rt2(
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv_rt",
                    ops::ExactLmHeadTier::M4.symbol_rt2(),
                ),
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv_rt",
                    ops::ExactLmHeadTier::M8.symbol_rt2(),
                ),
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv_rt",
                    ops::ExactLmHeadTier::M17.symbol_rt2(),
                ),
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv_rt",
                    ops::ExactLmHeadTier::M32.symbol_rt2(),
                ),
            )
            .with_rt2_m32_grid(super::super::try_kernel(
                gpu,
                "w4a16_gemv_rt",
                "w4a16_gemv_batch_logits_exact_rt2_m32_grid",
            ))
            .with_wide_n_sw4(super::super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m8_o4w4",
            ))
            .with_wide_n_sw4_m10(super::super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m10_o4w4",
            ))
            .with_wide_n_sw4_m15_m25(
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_sw_m15_o2w4",
                ),
                super::super::try_kernel(
                    gpu,
                    "w4a16_gemv",
                    "w4a16_gemv_batch_logits_exact_sw_m25_o1w4",
                ),
            ),
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            gemv_sw: crate::layers::ops::gemv_sw_enabled(),
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
            w4a16_gemv_qkvz_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qkvz")?,
            fused_rms_qkvz_k: super::super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "rms_norm_residual_w4a16_gemv",
            ),
            fused_rms_qkvz_batch3_k: super::super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "rms_norm_residual_w4a16_gemv_batch3",
            ),
            deinterleave_k: gpu.kernel("ssm_preprocess", "deinterleave_qkvz")?,
            conv1d_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            conv1d_l2norm_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            // FP32 conv1d output prevents BF16 truncation in the recurrent
            // path from compounding past ~8k tokens. The Metal backend
            // (kernels/metal/common/causal_conv1d_update_l2norm.metal) only
            // ships the BF16 variant; on those targets we fall back to the
            // BF16 kernel via the `.0 != 0` gate at the use site
            // (ssm_forward.rs). Warn instead of error: missing-on-Metal is
            // expected, and a startup `error!` would page on benign cases.
            conv1d_l2norm_f32_k: {
                let h = super::super::try_kernel(
                    gpu,
                    "causal_conv1d",
                    "causal_conv1d_update_l2norm_f32",
                );
                if h.0 == 0 {
                    tracing::warn!(
                        "FP32 conv1d kernel not loaded; SSM uses BF16 conv \
                         output. Expect long-context coherence drift past ~8k \
                         tokens on this backend."
                    );
                }
                h
            },
            conv1d_l2norm_f32_sequence_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32_sequence",
            ),
            conv1d_l2norm_bf16_sequence_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_bf16_sequence",
            ),
            conv1d_l2norm_bf16_tree_sequence_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_bf16_tree_sequence",
            ),
            gdn_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_decode")?,
            gdn_f32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32",
            ),
            gdn_f32_sequence_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence",
            ),
            gdn_f32_sequence_persistent_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_persistent",
            ),
            gdn_f32_sequence_nosnap_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_nosnap",
            ),
            gdn_f32_sequence_lazyfinal_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal",
            ),
            gdn_f32_sequence_persistent_splitv4_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_persistent_splitv4",
            ),
            gdn_f32_sequence_lazyfinal_splitv4_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4",
            ),
            qwen4_gdn_decode_fused_k1_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_decode_fused_k1",
            ),
            qwen4_gdn_verify_fused_lazyfinal_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal",
            ),
            qwen4_gdn_verify_fused_persistent_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_persistent",
            ),
            qwen4_gdn_decode_fused_k1_multi_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_decode_fused_k1_multi_seq",
            ),
            qwen4_gdn_verify_fused_lazyfinal_multi_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq",
            ),
            qwen4_gdn_verify_fused_persistent_multi_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_persistent_multi_seq",
            ),
            qwen4_gdn_decode_fused_k1_splitv4_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_decode_fused_k1_splitv4",
            ),
            qwen4_gdn_verify_fused_lazyfinal_splitv4_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_splitv4",
            ),
            qwen4_gdn_verify_fused_persistent_splitv4_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_persistent_splitv4",
            ),
            qwen4_gdn_verify_fused_lazyfinal_multi_splitv4_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4",
            ),
            qwen4_gdn_verify_fused_persistent_multi_splitv4_k: super::super::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_persistent_multi_seq_splitv4",
            ),
            gdn_fused_group_done,
            gdn_fused_split_done_stride,
            gdn_fused_split_done,
            gdn_lazy_retain: std::sync::OnceLock::new(),
            gdn_1d_replay_k: ops::GdnCorrectionReplayKernels::load_optional(gpu),
            gdn_1d_replay_arena: std::sync::Mutex::new(None),
            ba_gates_k: gpu.kernel("ssm_preprocess", "dense_gemv_ba_gates")?,
            residual_add_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            l2_norm_k: gpu.kernel("norm", "l2_norm_bf16")?,
            residual_add_rms_norm_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "residual_add_rms_norm_f32")
                    .or_else(|_| gpu.kernel("norm", "residual_add_rms_norm"))?
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            gated_rms_norm_prefill_k: gpu.kernel("norm", "gated_rms_norm_prefill")?,
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_pipe_k: super::super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_t_k: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            w4a16_gemm_t_k64_k: gpu.kernel("w4a16", "w4a16_gemm_t_k64")?,
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            w4a16_gemm_t_w8_k: super::super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128n128_w8"),
            fp8_gemm_t_w8_k: super::super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128n128_w8"),
            // Optional small-M variant (qwen3.6-27b only). Use try_kernel so
            // generic builds without the kernel still link cleanly.
            w4a16_gemm_t_m16_k: super::super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            w4a16_gemm_t_m32_n64_k: super::super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // Split-K route for K=γ verify out_proj/qkvz (ATLAS_SSM_OUT_SPLITK /
            // ATLAS_SSM_QKVZ_SPLITK). try_kernel: absent on non-27b targets.
            w4a16_gemm_t_m32_n64_splitk_k: super::super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: super::super::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            ssm_splitk_workspace: std::sync::Mutex::new(None),
            ssm_act_e4m3_scratch: Default::default(),
            ssm_qkvz_e4m3: std::sync::Mutex::new(None),
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gdn_prefill_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_prefill")?,
            gdn_prefill_split_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split")?,
            gdn_prefill_split4_k: gpu
                .kernel("gated_delta_rule", "gated_delta_rule_prefill_split4")?,
            gdn_prefill_persistent_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent",
            ),
            gdn_prefill_persistent_wy4_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4",
            ),
            gdn_prefill_wy32_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64",
            ),
            gdn_prefill_wy32_gatecache_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy32_gatecache",
                "gated_delta_rule_prefill_wy32_gatecache",
            ),
            cast_bf16_to_e4m3_k: super::super::try_kernel(gpu, "w4a16_v2", "cast_bf16_to_e4m3"),
            dequant_nvfp4_to_e4m3_k: super::super::try_kernel(
                gpu,
                "w4a16_v2",
                "dequant_nvfp4_to_e4m3",
            ),
            gdn_prefill_wy32_gatecache_v2_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy32_gatecache_v2",
                "gated_delta_rule_prefill_wy32_gatecache_v2",
            ),
            // Optional Flash-Next warp-parallel WY32 kernel; handle is 0 on
            // targets without it and guarded by `.0 != 0` at dispatch.
            gdn_prefill_wy32_warp_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy32_warp",
                "gated_delta_rule_prefill_wy32_warp",
            ),
            // ── Batched GDN prefill kernel handles ──
            gdn_prefill_wy32_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy64_prefill",
                "gated_delta_rule_prefill_wy64_batched",
            ),
            gdn_prefill_persistent_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_batched",
            ),
            gdn_prefill_persistent_wy4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent_wy4_batched",
            ),
            gdn_prefill_split4_batched_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_prefill_split4_batched",
            ),
            compute_gdn_gates_k: gpu.kernel("ssm_preprocess", "compute_gdn_gates")?,
            // ── Multi-seq state-advance kernels (ATLAS_SSM_MULTI_SEQ_KERNEL=1) ──
            // Multi-seq kernels collapse the per-seq SSM state loop into ONE
            // launch each, advancing c independent sequences in parallel and
            // saturating the 48-SM GB10 at c >= 2 (vs single-seq launches
            // that touch 32 SMs and serialize across c). All are `try_kernel`
            // so older PTX bundles cleanly fall back to the per-seq path.
            conv1d_multi_seq_k: super::super::try_kernel(
                gpu,
                "causal_conv1d_multi_seq",
                "causal_conv1d_update_multi_seq",
            ),
            conv1d_l2norm_multi_seq_k: super::super::try_kernel(
                gpu,
                "causal_conv1d_multi_seq",
                "causal_conv1d_update_l2norm_multi_seq",
            ),
            gdn_decode_multi_seq_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_multi_seq",
                "gated_delta_rule_decode_multi_seq",
            ),
            compute_gdn_gates_multi_seq_k: super::super::try_kernel(
                gpu,
                "ssm_preprocess_multi_seq",
                "compute_gdn_gates_multi_seq",
            ),
            // FP32-output variants — production-precision multi-seq path.
            conv1d_l2norm_f32_multi_seq_k: super::super::try_kernel(
                gpu,
                "causal_conv1d_f32_multi_seq",
                "causal_conv1d_update_l2norm_f32_multi_seq",
            ),
            gdn_decode_f32_multi_seq_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_f32_multi_seq",
                "gated_delta_rule_decode_f32_multi_seq",
            ),
            gdn_decode_f16_multi_seq_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_f32_multi_seq",
                "gated_delta_rule_decode_f16_multi_seq",
            ),
            gated_rms_norm_f32_multi_seq_k: if config.output_gate_type == "sigmoid" {
                super::super::try_kernel(
                    gpu,
                    "qwen4_hyper",
                    "qwen4_gated_rms_norm_sigmoid_f32_multi_seq",
                )
            } else {
                super::super::try_kernel(
                    gpu,
                    "gated_rms_norm_f32_multi_seq",
                    "gated_rms_norm_f32_multi_seq",
                )
            },
            conv1d_l2norm_chunk3_k: super::super::try_kernel(
                gpu,
                "causal_conv1d_chunk3_l2norm",
                "causal_conv1d_update_l2norm_chunk3",
            ),
            // Per-layer scratch for the 2 × c × 8-byte per-seq state ptr
            // arrays uploaded before each multi-seq launch. Provisioned to
            // `QWEN4_NATIVE_MAX_ROWS` (256 seqs = 4096 B/layer).
            ssm_multi_seq_ptr_scratch: {
                const MAX_C: usize = crate::layers::QWEN4_NATIVE_MAX_ROWS;
                let bytes = 2 * MAX_C * std::mem::size_of::<u64>();
                gpu.alloc(bytes)?
            },
            ssm_multi_seq_ptr_max: crate::layers::QWEN4_NATIVE_MAX_ROWS,
            // Page-locked host staging gives a stable DMA source address
            // across CUDA graph replays (64 u64s = 512 B for <= 32 seqs).
            multi_seq_ptr_host: parking_lot::Mutex::new(
                gpu.alloc_host_pinned(64 * std::mem::size_of::<u64>())?,
            ),
            multi_seq_ptr_wide_host: parking_lot::Mutex::new(gpu.alloc_host_pinned(
                2 * crate::layers::QWEN4_NATIVE_MAX_ROWS * std::mem::size_of::<u64>(),
            )?),
            multi_seq_ptr_pinned_ring: std::sync::Mutex::new(None),
            multi_seq_ptr_uploaded_rows: std::sync::atomic::AtomicUsize::new(0),
            gdn_f32_sequence_persistent_multi_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_persistent_multi",
            ),
            gdn_f32_sequence_lazyfinal_multi_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal_multi",
            ),
            conv1d_l2norm_f32_sequence_multi_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_l2norm_f32_sequence_multi",
            ),
            cross_seq_ptr_table: std::sync::Mutex::new(None),
            ba_gates_prefill_k: gpu.kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?,
            conv1d_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv1d_prefill_zcopy_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_zcopy",
            ),
            conv1d_prefill_l2norm_zcopy_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_l2norm_zcopy",
            ),
            gdn_chunk2_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk2")?,
            conv1d_chunk2_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_chunk2")?,
            gdn_chunk3_k: gpu.kernel("gated_delta_rule", "gated_delta_rule_chunk3")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            gdn_wy2_k: gpu.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            gdn_wy3_k: gpu.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            gdn_wy4_k: gpu.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
            // ATLAS_SSM_H_FP16 twins. try_kernel, not kernel: these are absent
            // from some targets' PTX and their absence must not break a normal
            // FP32 launch. The .cu sources name the wy2 twin
            // `gated_delta_rule_wy_f16` (module `gated_delta_rule_wy_f16`),
            // mirroring how the FP32 wy2 lives in the `gated_delta_rule_wy`
            // module above.
            gdn_wy2_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy_f16",
                "gated_delta_rule_wy2_f16",
            ),
            gdn_wy3_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy3_f16",
                "gated_delta_rule_wy3_f16",
            ),
            gdn_wy4_f16_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy4_f16",
                "gated_delta_rule_wy4_f16",
            ),
            // wy17 only present in qwen3.6-35b-a3b's PTX module set; NULL on other targets.
            // decode_batched(K=17) checks for non-NULL before dispatching the fused path.
            gdn_wy17_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17",
            ),
            // V-dim-split occupancy variant of wy17. NULL on targets that
            // haven't compiled gated_delta_rule_wy17_vsplit.cu → dispatch
            // falls back to gdn_wy17_k (single CTA/head).
            gdn_wy17_vsplit_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17_vsplit",
                "gated_delta_rule_wy17_vsplit",
            ),
            // LAZY Hi-writes wy17 + its companion replay kernel. Both live in
            // the SAME PTX module as gdn_wy17_k. NULL on targets whose wy17
            // module predates the lazy build → dispatch uses gdn_wy17_k.
            gdn_wy17_lazy_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17_lazy",
            ),
            gdn_wy17_replay_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17",
                "gated_delta_rule_wy17_replay",
            ),
            // Combined LAZY + V-DIM SPLIT wy17: lives in the vsplit PTX module.
            // NULL when the vsplit module predates this kernel symbol.
            gdn_wy17_lazy_vsplit_k: super::super::try_kernel(
                gpu,
                "gated_delta_rule_wy17_vsplit",
                "gated_delta_rule_wy17_lazy_vsplit",
            ),
            // DDTree GDN verify: tree-aware GDN kernel (gated_delta_rule_tree.cu).
            // Sequential per-token loop with parent_ids state load — enables
            // non-flat DDTree branches. NULL on targets that haven't compiled
            // the kernel (gated_delta_rule_tree.cu must exist in their nvfp4/).
            gdn_tree_k: {
                let k =
                    super::super::try_kernel(gpu, "gated_delta_rule_tree", "gated_delta_rule_tree");
                tracing::debug!(
                    "DDTree GDN verify: gdn_tree_k handle = {} (0 = NOT LOADED)",
                    k.0
                );
                k
            },
            // DDTree GDN verify (WY-fused): tree-aware WY-fused kernel.
            // Bit-equivalent to wy17 on flat-chain payloads (same H_root reads,
            // kd_flat reductions, WY correction algebra). For tree branches,
            // walks ancestor chain per token. Preferred path when present.
            gdn_tree_wy_k: {
                let k = super::super::try_kernel(
                    gpu,
                    "gated_delta_rule_tree_wy",
                    "gated_delta_rule_tree_wy",
                );
                tracing::debug!(
                    "DDTree GDN verify (WY): gdn_tree_wy_k handle = {} (0 = NOT LOADED)",
                    k.0
                );
                k
            },
            conv1d_tree_reroot_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_tree_reroot",
            ),
            h_state_bytes: nv * vd * kd * 4, // FP32 [nv, kd, vd] transposed for coalescing
            conv_state_bytes: conv_dim * d_conv * 4, // FP32 [conv_dim, d_conv]
            qkvz_fp8: None,
            out_proj_fp8: None,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            fp8_gemm_t_m128_k: gpu.kernel("w4a16", "fp8_gemm_t_m128")?,
        })
    }

    /// Construct an SSM layer where QKVZ projection output is already sequential.
    ///
    /// Used by Qwen3.5 where separate QKV and Z weights are concatenated at load
    /// time into `[Q|K|V|Z]` row order. The `deinterleave_qkvz` kernel is skipped
    /// and plain `w4a16_gemv` writes directly to the deinterleaved buffer.
    pub fn new_sequential(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        qkvz_nvfp4_t: Option<QuantizedWeight>,
        out_proj_nvfp4_t: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::new_sequential_with_splitk(
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            qkvz_nvfp4_t,
            out_proj_nvfp4_t,
            config,
            gpu,
            None,
        )
    }

    /// [`Self::new_sequential`] whose split-K workspace, when the layer
    /// needs one, is leased from `shared` (`ATLAS_MEMLEAN_SHARED_SCRATCH`)
    /// instead of allocated per layer.
    pub fn new_sequential_with_splitk(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        qkvz_nvfp4_t: Option<QuantizedWeight>,
        out_proj_nvfp4_t: Option<QuantizedWeight>,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        shared_splitk: Option<&crate::layers::shared_splitk::SharedSplitkWorkspace>,
    ) -> Result<Self> {
        let mut layer = Self::new(
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            config,
            gpu,
        )?;
        layer.sequential_qkvz = true;
        layer.qkvz_nvfp4_t = qkvz_nvfp4_t;
        layer.out_proj_nvfp4_t = out_proj_nvfp4_t;
        layer.alloc_ssm_splitk_ws(config, gpu, shared_splitk)?;
        Ok(layer)
    }

    /// Eagerly allocate the FP32 split-K workspace for the K=γ verify
    /// out_proj / qkvz split-K routes (`ATLAS_SSM_OUT_SPLITK` /
    /// `ATLAS_SSM_QKVZ_SPLITK`). Sized for the larger of the two enabled
    /// routes: `[k_splits, 32, N]` FP32 where N = hidden (out_proj) or
    /// qkvz_size (qkvz). Allocated at load time because `gpu.alloc` is
    /// illegal during CUDA graph capture. No-op when both gates are off
    /// or the split-K kernel symbols are missing from the PTX bundle.
    fn alloc_ssm_splitk_ws(
        &self,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        shared: Option<&crate::layers::shared_splitk::SharedSplitkWorkspace>,
    ) -> Result<()> {
        let out_splits = super::super::ssm_out_splitk() as usize;
        let qkvz_splits = super::super::ssm_qkvz_splitk() as usize;
        if (out_splits == 0 && qkvz_splits == 0)
            || self.w4a16_gemm_t_m32_n64_splitk_k.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let bytes = std::cmp::max(
            out_splits * config.hidden_size,
            qkvz_splits * config.ssm_qkvz_size(),
        ) * 32
            * 4;
        let mut slot = self
            .ssm_splitk_workspace
            .lock()
            .map_err(|_| anyhow::anyhow!("ssm_splitk_workspace lock poisoned"))?;
        if slot.is_none() {
            *slot = Some(match shared {
                Some(shared) => shared.lease(gpu, bytes)?,
                None => gpu.alloc(bytes)?,
            });
        }
        Ok(())
    }

    /// Set native FP8 checkpoint weights for w8a16_gemv decode path.
    /// Also sets the raw FP8 DevicePtr fields for prefill GEMM (fp8_gemm_t).
    pub fn set_fp8_weights(&mut self, qkvz: Option<Fp8Weight>, out_proj: Option<Fp8Weight>) {
        // Set raw FP8 DevicePtr for prefill GEMM (fp8_gemm_t, no per-row scale needed)
        self.qkvz_fp8 = qkvz.as_ref().map(|w| w.weight);
        self.out_proj_fp8 = out_proj.as_ref().map(|w| w.weight);
        // Set Fp8Weight for decode GEMV (w8a16_gemv, needs per-row scale)
        self.qkvz_fp8w = qkvz;
        self.out_proj_fp8w = out_proj;
    }

    /// Build the exact runtime-generated NVFP4 weight operands for the
    /// default-off FlashInfer SSM prefill route. This must be called by the
    /// loader after all SSM requantization is complete. The layer is updated
    /// only after both projections and all four zero-workspace tactics pass.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn prepare_flashinfer_ssm_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        layer: usize,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        use anyhow::Context as _;
        use ops::flashinfer_sm121::{FlashInferSm121, FlashInferSm121Shape};

        ensure!(
            crate::layers::prefill_ssm_flashinfer_enabled()?,
            "prepare_flashinfer_ssm_prefill called while its route is disabled"
        );
        ensure!(
            self.flashinfer_ssm_prefill.is_none(),
            "FlashInfer SSM prefill operands were already prepared"
        );
        let value_dim = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .context("FlashInfer SSM value dimension overflow")?;
        ensure!(
            qwen38_ssm_flashinfer_geometry(
                2_079,
                config.hidden_size,
                config.ssm_qkvz_size(),
                value_dim,
            ),
            "FlashInfer SSM route requires exact Qwen3.8 H={QWEN38_FLASHINFER_HIDDEN}, QKVZ={QWEN38_FLASHINFER_SSM_QKVZ}, value={QWEN38_FLASHINFER_SSM_VALUE}"
        );
        ensure!(
            self.sequential_qkvz,
            "FlashInfer SSM route requires the Qwen3.8 sequential QKVZ layout"
        );
        ensure!(
            self.out_proj_dense.is_none(),
            "FlashInfer SSM route requires runtime-generated NVFP4 output weights"
        );
        let qkvz_weight = self
            .qkvz_nvfp4
            .as_ref()
            .context("FlashInfer SSM route requires the original-layout QKVZ weight")?;

        let library_path = std::env::var_os("ATLAS_FLASHINFER_SM121_LIB").ok_or_else(|| {
            anyhow::anyhow!("ATLAS_PREFILL_SSM_FLASHINFER=1 requires ATLAS_FLASHINFER_SM121_LIB")
        })?;
        let library = FlashInferSm121::open_qualified(std::path::Path::new(&library_path))?;
        let dynamic_scale_kernels = ops::nvfp4_dynamic_scale::Nvfp4DynamicScaleKernels::load(gpu)?;

        // Freeze every tactic/workspace result before allocating or retaining
        // either weight-scale view. Runtime uses this same sealed library.
        let preparation_stream = gpu.default_stream();
        for rows in [2_079, 8_192] {
            let (qkvz_tactic, output_tactic) = qwen38_ssm_flashinfer_tactics(rows)
                .context("missing qualified FlashInfer SSM tactic")?;
            for (tactic, n, k) in [
                (
                    qkvz_tactic,
                    QWEN38_FLASHINFER_SSM_QKVZ,
                    QWEN38_FLASHINFER_HIDDEN,
                ),
                (
                    output_tactic,
                    QWEN38_FLASHINFER_HIDDEN,
                    QWEN38_FLASHINFER_SSM_VALUE,
                ),
            ] {
                let shape = FlashInferSm121Shape::new(tactic, rows, n, k, 1)?;
                let prepared =
                    library.prepare_borrowed_zero_workspace(gpu, shape, preparation_stream)?;
                ensure!(
                    prepared.shape() == shape && prepared.stream() == preparation_stream,
                    "FlashInfer SSM preparation changed its frozen shape or stream"
                );
            }
        }

        let qkvz = build_flashinfer_ssm_projection(
            gpu,
            "qkvz",
            qkvz_weight,
            QWEN38_FLASHINFER_SSM_QKVZ,
            QWEN38_FLASHINFER_HIDDEN,
        )?;
        let output = match build_flashinfer_ssm_projection(
            gpu,
            "output",
            &self.ssm.out_proj,
            QWEN38_FLASHINFER_HIDDEN,
            QWEN38_FLASHINFER_SSM_VALUE,
        ) {
            Ok(output) => output,
            Err(error) => {
                let _ = gpu.free(qkvz.weight_scales_128x4);
                return Err(error);
            }
        };

        let (library_device, library_inode) = library.file_identity();
        tracing::info!(
            layer,
            library = %library.path().display(),
            library_sha256 = %library.sha256_hex(),
            library_device,
            library_inode,
            qkvz_weight = format_args!("{:#x}", qkvz.weight.0),
            output_weight = format_args!("{:#x}", output.weight.0),
            qkvz_scale_hash = format_args!("{:016x}", qkvz.weight_scales_hash),
            output_scale_hash = format_args!("{:016x}", output.weight_scales_hash),
            "admitted FlashInfer SM121 SSM prefill operands"
        );
        self.flashinfer_ssm_prefill = Some(FlashinferSsmPrefill {
            layer,
            library,
            dynamic_scale_kernels,
            qkvz,
            output,
        });
        Ok(())
    }

    pub fn set_qwen4_hyperconnections(
        &mut self,
        attn: crate::layers::Qwen4HyperConnection,
        mlp: crate::layers::Qwen4HyperConnection,
    ) {
        self.qwen4_attn_hyper = Some(attn);
        self.qwen4_mlp_hyper = Some(mlp);
    }

    /// Set raw FP8 DevicePtrs for the prefill GEMM path ONLY (no decode GEMV
    /// scale fields). Used by the Qwen3.6-27B-FP8 native-FP8 SSM prefill path:
    /// the FP8 buffer here is a single-scale FP8 (BF16 → FP8 truncation; values
    /// already in FP8 range) suitable for `fp8_gemm_n128`. Decode falls back to
    /// the NVFP4/BF16 paths via the existing `qkvz_nvfp4*` fields. Caller
    /// decides whether to install — never set implicitly.
    pub fn set_fp8_prefill_only_weights(
        &mut self,
        qkvz_fp8: Option<DevicePtr>,
        out_proj_fp8: Option<DevicePtr>,
    ) {
        if qkvz_fp8.is_some() {
            self.qkvz_fp8 = qkvz_fp8;
        }
        if out_proj_fp8.is_some() {
            self.out_proj_fp8 = out_proj_fp8;
        }
    }

    /// Pre-dequant non-transposed NVFP4 → FP8 for `out_proj` (QKVZ is skipped as
    /// bandwidth-bound). Eliminates per-inference dequant overhead in prefill GEMMs.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let qkvz_size = config.ssm_qkvz_size();
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

        // QKVZ FP8 predequant: tested at ISL=1019, FP8 is ~50% slower (1900µs vs 1228µs)
        // because weight matrix [12288, 2048] is bandwidth-dominated at M=1024 — the 2×
        // larger FP8 weights (25 MB vs 12.6 MB NVFP4) cost more than the dequant saves.
        let _ = qkvz_size; // suppress unused warning
        // Use NON-transposed out_proj (ssm.out_proj is [N, K/2] layout).
        // predequant_nvfp4_to_fp8 assumes [N, K/2] input layout.
        // Skip eager 1.51 GiB FP8 out_proj copy when cuBLASLt NVFP4 replaces FP8 on SM120+.
        if self.out_proj_nvfp4_t.is_some()
            && !crate::layers::dense_ffn::proj_27b_cublaslt_nvfp4_can_replace_fp8(
                gpu,
                Some(&self.ssm.out_proj),
                h as u32,
                value_dim as u32,
            )
        {
            self.out_proj_fp8 = Some(self.ssm.out_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                value_dim,
                stream,
            )?);
        }
        Ok(())
    }
}
