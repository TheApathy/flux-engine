// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::new constructor.

use super::*;

fn strict_binary_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(value) => anyhow::bail!("{name} must be exactly 0 or 1, got {value:?}"),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} must contain valid UTF-8 and be exactly 0 or 1")
        }
    }
}

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        // Sanity-check the routing config: top-k outside 1..=num_experts
        // would index OOB or produce empty routing in the topk kernel.
        // Catch the misconfiguration at load time.
        anyhow::ensure!(
            config.num_experts_per_tok >= 1
                && config.num_experts_per_tok <= num_experts
                && num_experts > 0,
            "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
            config.num_experts_per_tok,
            num_experts,
        );
        let qwen4_compact = qwen4_prefill_compact::load(gpu, config)?;
        let gate_ptrs = build_ptr_table(&weights.experts, |e| &e.gate_proj, gpu)?;
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;

        // Extract the optional correction-bias device pointer before the
        // struct literal below moves `weights`. `.map(|dw| dw.weight)` turns
        // an `Option<DenseWeight>` into an `Option<DevicePtr>` for the
        // `moe_topk_sigmoid` kernel's bias arg.
        let weights_correction_bias: Option<DevicePtr> =
            weights.correction_bias.map(|dw| dw.weight);

        let nvfp4_gate_up_m128 = std::env::var("ATLAS_NVFP4_GATE_UP_M128")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let nvfp4_moe_worklist = strict_binary_env("ATLAS_NVFP4_MOE_WORKLIST")?;
        if nvfp4_moe_worklist {
            anyhow::ensure!(
                !strict_binary_env("ATLAS_MOE_EXACT_PREFILL_GRID")?,
                "ATLAS_NVFP4_MOE_WORKLIST=1 conflicts with diagnostic ATLAS_MOE_EXACT_PREFILL_GRID=1"
            );
            anyhow::ensure!(
                !nvfp4_gate_up_m128,
                "ATLAS_NVFP4_MOE_WORKLIST=1 uses the compilable M64 parent and rejects unavailable ATLAS_NVFP4_GATE_UP_M128=1"
            );
        }
        let moe_build_nvfp4_worklist_k =
            super::super::try_kernel(gpu, "moe_w4a16", "moe_w4a16_build_tile_worklist");
        let moe_fused_gate_up_t_k64_worklist_k =
            super::super::try_kernel(gpu, "moe_w4a16", "moe_w4a16_fused_gate_up_t_k64_worklist");
        let moe_grouped_gemm_t_k64_worklist_k = super::super::try_kernel(
            gpu,
            "moe_w4a16",
            "moe_w4a16_grouped_gemm_ptrtable_t_k64_worklist",
        );
        if nvfp4_moe_worklist {
            anyhow::ensure!(
                moe_build_nvfp4_worklist_k.0 != 0
                    && moe_fused_gate_up_t_k64_worklist_k.0 != 0
                    && moe_grouped_gemm_t_k64_worklist_k.0 != 0,
                "ATLAS_NVFP4_MOE_WORKLIST=1 requires the complete moe_w4a16 compact-worklist kernel bundle"
            );
        }

        let _ = num_experts;
        let rms_norm_k = gpu.kernel("norm", "rms_norm")?;
        let gu_t8_r3 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_gate_up_shared_t_lanes_rows_t8_r3_k160",
        );
        let gu_t8_r5 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_gate_up_shared_t_lanes_rows_t8_r5_k160",
        );
        let gu_t8_r8 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_gate_up_shared_t_lanes_rows_t8_r8_k160",
        );
        let dn_t8_r3 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_silu_down_shared_t_lanes_rows_t8_r3_k40",
        );
        let dn_t8_r5 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_silu_down_shared_t_lanes_rows_t8_r5_k40",
        );
        let dn_t8_r8 = crate::layers::try_kernel(
            gpu,
            "moe_shared_expert_fused_t",
            "moe_expert_silu_down_shared_t_lanes_rows_t8_r8_k40",
        );
        Ok(Self {
            weights,
            routed_scale_layout: RoutedScaleLayout::FlatE4m3,
            gate_nvfp4,
            pre_expert_norm: None,
            pre_expert_norm_k: rms_norm_k,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_rows: gpu
                .kernel("w4a16_gemv", "w4a16_gemv_rows")
                .unwrap_or(KernelHandle(0)),
            w4a16_gemv_exact_m4: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_m4",
            ),
            w4a16_gemv_exact_m8: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_m8",
            ),
            w4a16_gemv_exact_m17: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_m17",
            ),
            w4a16_gemv_exact_m32: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_m32",
            ),
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            router_bf16_tk64: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_tk64"),
            moe_expert_gate_up_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")?,
            moe_expert_silu_down_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")?,
            moe_topk: gpu.kernel("moe_topk", "moe_topk_softmax")?,
            moe_weighted_sum_blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
            residual_add: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            moe_topk_batched: gpu.kernel("moe_topk", "moe_topk_softmax_batched")?,
            moe_expert_gate_up_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?,
            moe_expert_silu_down_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_silu_down_shared_batch2")?,
            moe_weighted_sum_blend_batch2: gpu
                .kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            moe_expert_gate_up_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_gate_up_shared_batch3")?,
            moe_expert_silu_down_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_silu_down_shared_batch3")?,
            moe_expert_gate_up_shared_dedup: [
                gpu.kernel("moe_fused_batch3", "moe_expert_gate_up_shared_dedup_r8")
                    .unwrap_or(KernelHandle(0)),
                gpu.kernel("moe_fused_batch3", "moe_expert_gate_up_shared_dedup_r16")
                    .unwrap_or(KernelHandle(0)),
            ],
            moe_expert_silu_down_shared_dedup: [
                gpu.kernel("moe_fused_batch3", "moe_expert_silu_down_shared_dedup_r8")
                    .unwrap_or(KernelHandle(0)),
                gpu.kernel("moe_fused_batch3", "moe_expert_silu_down_shared_dedup_r16")
                    .unwrap_or(KernelHandle(0)),
            ],
            moe_dedup_act_rows_k: gpu
                .kernel("moe_fused_batch3", "moe_dedup_act_rows")
                .unwrap_or(KernelHandle(0)),
            moe_expert_silu_down_shared_dedup_act: [
                (
                    gpu.kernel(
                        "moe_fused_batch3",
                        "moe_expert_silu_down_shared_dedup_act_r8_o4",
                    )
                    .unwrap_or(KernelHandle(0)),
                    4,
                ),
                (
                    gpu.kernel(
                        "moe_fused_batch3",
                        "moe_expert_silu_down_shared_dedup_act_r16_o1",
                    )
                    .unwrap_or(KernelHandle(0)),
                    1,
                ),
            ],
            moe_down_act_buf: std::sync::OnceLock::new(),
            moe_weighted_sum_blend_batch3: gpu
                .kernel("moe_fused_batch3", "moe_weighted_sum_blend_batch3")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            moe_sort_by_expert: gpu.kernel("moe", "moe_sort_by_expert")?,
            moe_grouped_gemm: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            moe_grouped_gemm_t_k64: gpu
                .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t_k64")?,
            moe_fused_gate_up_t_k64: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64")?,
            // M=128 variant only present in models where Block D #3 has
            // been ported (currently minimax-m2-229b). Other models keep
            // KernelHandle(0) and dispatch falls through to M=64.
            moe_fused_gate_up_t_k64_m128: {
                let k = super::super::try_kernel(
                    gpu,
                    "moe_w4a16",
                    "moe_w4a16_fused_gate_up_t_k64_m128",
                );
                if k.0 != 0 {
                    gpu.set_kernel_max_dynamic_shared_memory(k, 58_048)?;
                }
                k
            },
            nvfp4_moe_worklist,
            qwen4_compact,
            moe_build_nvfp4_worklist_k,
            moe_fused_gate_up_t_k64_worklist_k,
            moe_grouped_gemm_t_k64_worklist_k,
            moe_fp8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_fp8_grouped_gemm",
                "moe_fp8_grouped_gemm",
            ),
            moe_fp8_grouped_gemm_v2_k: super::super::try_kernel(
                gpu,
                "moe_fp8_grouped_gemm",
                "moe_fp8_grouped_gemm_v2",
            ),
            // 2026-05-20: default-ON. v1 had documented coalescing-perf bug;
            // verified that v1 also has *numerical* bug for some
            // (token, expert) tile combinations — chunk-4 last-token
            // expert-200 up_proj output 2.4× too large vs v2 on
            // Qwen3.6-35B-A3B-FP8 (matches HF reference). Drives long-context
            // residual amplification at the prefill-chunk-4 boundary.
            // Override via ATLAS_FP8_MOE_COALESCED=0 to restore v1.
            fp8_moe_coalesced_enabled: std::env::var("ATLAS_FP8_MOE_COALESCED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w4a16_gemm_t: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            moe_silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            moe_act_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?, // default: SiLU
            gelu_activation: false,
            moe_unpermute_reduce: gpu.kernel("moe", "moe_unpermute_reduce_indexed")?,
            moe_batched_blend: gpu.kernel("moe", "moe_batched_blend")?,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_ptrs_t: None,
            up_ptrs_t: None,
            down_ptrs_t: None,
            routed_gate_up_transpose_slab: None,
            routed_down_transpose_slab: None,
            routed_ts4e_scale_slabs: Vec::new(),
            routed_bounded_transpose_owners: Vec::new(),
            routed_original_groups: None,
            down_t_scratch_packed: None,
            down_t_scratch_scale: None,
            stream_t_scratch: None,
            moe_transpose_u8_batched_k: gpu
                .kernel("moe_transpose_batched", "moe_transpose_u8_batched")?,
            // ── Phase 8a transposed-layout decode kernels ──
            // Module name = file stem (default convention in atlas-kernels).
            moe_expert_gate_up_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t")?,
            moe_expert_silu_down_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t")?,
            moe_expert_gate_up_shared_t_lanes_k: gpu
                .kernel(
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_shared_t_lanes_o4t16u1",
                )
                .unwrap_or(KernelHandle(0)),
            moe_expert_silu_down_shared_t_lanes_k: gpu
                .kernel(
                    "moe_shared_expert_fused_t",
                    "moe_expert_silu_down_shared_t_lanes_o4t16u1",
                )
                .unwrap_or(KernelHandle(0)),
            moe_expert_gate_up_shared_t_lanes_t8_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_shared_t_lanes_t8_k160",
            ),
            moe_expert_silu_down_shared_t_lanes_t8_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_silu_down_shared_t_lanes_t8_k40",
            ),
            moe_expert_gate_up_shared_t_lanes_rows_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_shared_t_lanes_rows_o4t16",
            ),
            moe_expert_silu_down_shared_t_lanes_rows_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_silu_down_shared_t_lanes_rows_o4t16",
            ),
            t_lanes_rows_spec: [2u32, 3, 4, 5, 9]
                .iter()
                .filter_map(|&cap| {
                    let gu = crate::layers::try_kernel(
                        gpu,
                        "moe_shared_expert_fused_t",
                        &format!("moe_expert_gate_up_shared_t_lanes_rows_r{cap}"),
                    );
                    let dn = crate::layers::try_kernel(
                        gpu,
                        "moe_shared_expert_fused_t",
                        &format!("moe_expert_silu_down_shared_t_lanes_rows_r{cap}"),
                    );
                    if gu.0 == 0 || dn.0 == 0 {
                        return None;
                    }
                    // gate/up stages cap rows of BF16 input: opt in when that plus the
                    // kernel's static smem passes 48 KiB (cap 9 at 8+ rows otherwise
                    // fails to launch).
                    let smem = cap * config.hidden_size as u32 * 2;
                    if smem + ops::T_LANES_ROWS_STATIC_SMEM > 48 * 1024
                        && gpu.set_kernel_max_dynamic_shared_memory(gu, smem).is_err()
                    {
                        return None;
                    }
                    Some((cap, gu, dn))
                })
                .collect(),
            moe_expert_gate_up_shared_t_lanes_rows_t8_tiers: [
                gu_t8_r3,
                gu_t8_r5,
                gu_t8_r8,
                KernelHandle(0),
            ],
            moe_expert_silu_down_shared_t_lanes_rows_t8_tiers: [
                dn_t8_r3,
                dn_t8_r5,
                dn_t8_r8,
                KernelHandle(0),
            ],
            moe_build_unique_worklist_r8_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_build_unique_worklist_r8",
            ),
            moe_expert_gate_up_worklist_t8_tiers: [
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t8_r3_k160",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t8_r5_k160",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t8_r8_k160",
                ),
                KernelHandle(0),
            ],
            moe_expert_gate_up_worklist_t_lanes_rows_r2_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_worklist_t_lanes_rows_r2",
            ),
            moe_expert_gate_up_worklist_t_lanes_rows_r4_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_worklist_t_lanes_rows_r4",
            ),
            moe_expert_gate_up_worklist_t_lanes_rows_tiers: [
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t_lanes_rows_r3",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t_lanes_rows_r5",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_gate_up_worklist_t_lanes_rows_r8",
                ),
                {
                    let gu_r10 = crate::layers::try_kernel(
                        gpu,
                        "moe_shared_expert_fused_t",
                        "moe_expert_gate_up_worklist_t_lanes_rows_r10",
                    );
                    // Matches `SMEM_RMAX = min(RMAX, 8)` + `16 * 65 * 4` reduction scratch
                    // in `moe_shared_expert_fused_t.cu`.
                    let smem = 8 * config.hidden_size as u32 * 2 + 16 * 65 * 4;
                    if gu_r10.0 != 0
                        && gpu
                            .set_kernel_max_dynamic_shared_memory(gu_r10, smem)
                            .is_err()
                    {
                        KernelHandle(0)
                    } else {
                        gu_r10
                    }
                },
            ],
            moe_silu_mul_rows_f32_k: crate::layers::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_silu_mul_rows_f32",
            ),
            moe_expert_down_act_shared_t_lanes_rows_t8_tiers: [
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_down_act_shared_t_lanes_rows_t8_r3_k40",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_down_act_shared_t_lanes_rows_t8_r5_k40",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_down_act_shared_t_lanes_rows_t8_r8_k40",
                ),
                crate::layers::try_kernel(
                    gpu,
                    "moe_shared_expert_fused_t",
                    "moe_expert_down_act_shared_t_lanes_rows_t8_r10_k40",
                ),
            ],
            moe_t8_worklist_buf: std::sync::OnceLock::new(),
            moe_t8_act_buf: std::sync::OnceLock::new(),
            moe_expert_gate_up_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_gate_up_shared_batch2_t",
            )?,
            moe_expert_silu_down_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_silu_down_shared_batch2_t",
            )?,
            moe_expert_gate_up_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_gate_up_shared_batch3_t",
            )?,
            moe_expert_silu_down_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_silu_down_shared_batch3_t",
            )?,
            unified_layout: std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or_else(|_| config.is_qwen4_exp()),
            hybrid_layout: std::env::var("ATLAS_HYBRID_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_gate_up_m128,
            k3_fused_gate_up: std::env::var("ATLAS_MOE_K3_FUSED_GATE_UP")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            shared_gate_t: None,
            qwen4_shared_fp8: None,
            shared_up_t: None,
            shared_down_t: None,
            gate_fp8: None,
            shared_gate_fp8: None,
            shared_up_fp8: None,
            shared_down_fp8: None,
            prefill_stream: gpu.create_stream()?,
            event_a: gpu.create_event()?,
            event_b: gpu.create_event()?,
            moe_expert_gate_up_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_gate_up_shared_fp8",
            )?,
            moe_expert_silu_down_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_silu_down_shared_fp8",
            )?,
            // FP8 batch2/3 kernels for MTP verify
            moe_expert_gate_up_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_gate_up_shared_fp8_batch2",
            )?,
            moe_expert_silu_down_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_silu_down_shared_fp8_batch2",
            )?,
            moe_weighted_sum_blend_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_weighted_sum_blend_fp8_batch2",
            )?,
            moe_expert_gate_up_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_gate_up_shared_fp8_batch3",
            )?,
            moe_expert_silu_down_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_silu_down_shared_fp8_batch3",
            )?,
            moe_weighted_sum_blend_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_weighted_sum_blend_fp8_batch3",
            )?,
            fp8_gate_weight_ptrs: None,
            fp8_up_weight_ptrs: None,
            fp8_down_weight_ptrs: None,
            fp8_shared_expert: None,
            // Phase 2.7 Tier C — set by loader after construction (qwen35.rs).
            is_dflash_capture_layer: false,
            correction_bias_dev: weights_correction_bias,
            // `moe_topk_sig` is only registered for sigmoid-gated MoE models
            // (MiniMax-M2, Nemotron-Nano, Nemotron-Super). Softmax-gated MoEs
            // (Qwen3.5, Qwen3-Next, Gemma-4, Mistral) never hit the sigmoid
            // dispatch path, so a missing kernel is fine — fail at call time
            // via the KernelHandle(0) check in ops::moe_topk_sigmoid rather
            // than at MoeLayer::new(), which would otherwise block all
            // softmax-MoE model startup (observed on Qwen3.5-35B-A3B-FP8 in
            // alpha-2.43: "Module 'moe_topk_sig' not loaded" during model
            // build).
            moe_topk_sigmoid_k: super::super::try_kernel(gpu, "moe_topk_sig", "moe_topk_sigmoid"),
            moe_topk_sigmoid_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sig",
                "moe_topk_sigmoid_batched",
            ),
        })
    }
}
