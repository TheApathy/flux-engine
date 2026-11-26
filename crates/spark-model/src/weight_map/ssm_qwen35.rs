// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Qwen3.5 SSM weights with separate projections.
pub struct SsmWeightsQwen35 {
    /// QKV projection: [qkv_size, hidden_size] BF16 (Q+K+V, no Z).
    pub in_proj_qkv: DenseWeight,
    /// Z gate projection: [z_size, hidden_size] BF16.
    pub in_proj_z: DenseWeight,
    /// Alpha projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_a: DenseWeight,
    /// Beta projection: [num_value_heads, hidden_size] BF16.
    pub in_proj_b: DenseWeight,
    /// Conv1d weight: [d_inner, 1, d_conv] BF16.
    pub conv1d: DenseWeight,
    /// A_log parameter: `[num_v_heads]` FP32.
    pub a_log: DenseWeight,
    /// dt_bias parameter: `[num_v_heads]` FP32.
    pub dt_bias: DenseWeight,
    /// Gate norm weight: `[linear_value_head_dim]` (per-head `dv`) BF16.
    pub norm: DenseWeight,
    /// Output projection: `[hidden_size, value_dim]` BF16 (NOT NVFP4 — quantizer skipped these).
    pub out_proj: DenseWeight,
}

/// Load SSM weights for Qwen3.5 (separate projections, BF16 out_proj).
///
/// Handles three quant layouts for the linear-attention projections:
///   * Raw BF16 (Bf16Raw) — pointer alias.
///   * FP8 block-scaled (Fp8Dequanted) — dequant via `dense_auto`.
///   * NVFP4 packed (Standard / CompressedTensors) — dequant to BF16 via
///     `dequant_nvfp4_to_bf16`, which auto-detects modelopt naming
///     (`.weight` uint8 + `weight_scale_2`) vs compressed-tensors naming
///     (`.weight_packed` + `weight_global_scale`).
///
/// Returns BF16 dense weights regardless of source format so the
/// downstream concat / re-quantize pipeline in
/// `qwen35::load_layers::linear_attn_arms` works uniformly.
pub(crate) fn load_ssm_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &atlas_core::config::ModelConfig,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.linear_attn");
    let h = config.hidden_size;

    // Detect NVFP4 packed format under either naming convention.
    // Modelopt: packed bytes live in `.weight` (uint8) + `weight_scale_2` (FP32 scalar).
    // Compressed-tensors: packed bytes live in `.weight_packed` + `weight_global_scale`.
    let is_packed = |proj: &str| -> bool {
        let prefix = format!("{p}.{proj}");
        if store.contains(&format!("{prefix}.weight_packed")) {
            return true;
        }
        if let Ok(w) = store.get(&format!("{prefix}.weight"))
            && w.dtype == WeightDtype::UInt8
            && store.contains(&format!("{prefix}.weight_scale_2"))
        {
            return true;
        }
        false
    };

    // Load a projection as BF16 dense, dequantizing if the checkpoint
    // ships NVFP4 packed bytes. `n`, `k` are the logical dims of the
    // unpacked weight (n rows × k cols of BF16).
    let load_proj_bf16 = |proj: &str, n: usize, k: usize| -> Result<DenseWeight> {
        let prefix = format!("{p}.{proj}");
        if config
            .quantization_config
            .as_ref()
            .and_then(|quant| quant.modelopt_weight_format_for(&prefix))
            .is_some()
        {
            return dense_modelopt_mixed_or_fp8_or_bf16(store, &prefix, config, gpu);
        }
        if is_packed(proj) {
            return dequant_nvfp4_to_bf16(store, &prefix, n, k, gpu);
        }
        // FP8 SSM projections must become BF16 before concat/re-quantize.
        // Keep the established scalar-scale path, but route compressed-tensors
        // per-row `[N,1]` scales (and native block grids) through `dense_auto`;
        // treating the FP8 bytes as BF16 would read beyond the allocation.
        if let Ok(w) = store.get(&format!("{prefix}.weight"))
            && w.dtype == WeightDtype::FP8E4M3
        {
            let scalar_weight_scale = store
                .get(&format!("{prefix}.weight_scale"))
                .map(|scale| scale.num_elements() == 1)
                .unwrap_or(false);
            return if scalar_weight_scale {
                dequant_fp8_to_bf16(store, &prefix, gpu)
            } else {
                dense_auto(store, &format!("{prefix}.weight"), gpu)
            };
        }
        match variant {
            Nvfp4Variant::Fp8Dequanted => dense_auto(store, &format!("{prefix}.weight"), gpu),
            _ => dense(store, &format!("{prefix}.weight")),
        }
    };

    let tp_size = config.tp_world_size.max(1);
    let tp_rank = config.tp_rank;
    ensure!(
        tp_rank < tp_size,
        "SSM TP rank {tp_rank} >= tp_size {tp_size}"
    );
    let local_nk = config.linear_num_key_heads;
    let local_nv = config.linear_num_value_heads;
    let kd = config.linear_key_head_dim;
    let vd = config.linear_value_head_dim;
    let d_conv = config.linear_conv_kernel_dim;
    let local_kd = local_nk * kd;
    let local_vd = local_nv * vd;
    let local_qkv_size = 2 * local_kd + local_vd;
    let local_z_size = local_vd;

    let full_nk = local_nk * tp_size;
    let full_nv = local_nv * tp_size;
    let full_kd = full_nk * kd;
    let full_vd = full_nv * vd;
    let full_qkv_size = 2 * full_kd + full_vd;
    let full_z_size = full_vd;

    let full_qkv = load_proj_bf16("in_proj_qkv", full_qkv_size, h)?;
    let full_z = load_proj_bf16("in_proj_z", full_z_size, h)?;
    let full_a = load_proj_bf16("in_proj_a", full_nv, h)?;
    let full_b = load_proj_bf16("in_proj_b", full_nv, h)?;
    let full_conv1d = dense(store, &format!("{p}.conv1d.weight"))?;
    // A_log and dt_bias MUST be FP32 — BF16 precision causes exponential
    // error amplification in the GDR decay gate at 8k+ tokens.
    let full_a_log = dense_keep_f32(store, &format!("{p}.A_log"), gpu)?;
    let full_dt_bias = dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?;
    // norm.weight is [vd] BF16 (shared across heads; replicated across TP ranks).
    let norm = dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?;
    let full_out_proj = load_proj_bf16("out_proj", h, full_vd)?;

    if tp_size == 1 {
        return Ok(SsmWeightsQwen35 {
            in_proj_qkv: full_qkv,
            in_proj_z: full_z,
            in_proj_a: full_a,
            in_proj_b: full_b,
            conv1d: full_conv1d,
            a_log: full_a_log,
            dt_bias: full_dt_bias,
            norm,
            out_proj: full_out_proj,
        });
    }

    let free_if_not_store_alias = |name: &str, ptr: DevicePtr| -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        if store.get(name).map(|w| w.ptr == ptr).unwrap_or(false) {
            return Ok(());
        }
        gpu.free(ptr)
    };

    // Slice planar [Q(full_kd) | K(full_kd) | V(full_vd)] rows by TP rank.
    let slice_qkv_planar = |src: DevicePtr, row_bytes: usize| -> Result<DenseWeight> {
        let dst = gpu.alloc(local_qkv_size * row_bytes)?;
        let q_bytes = local_kd * row_bytes;
        let v_bytes = local_vd * row_bytes;
        gpu.copy_d2d(src.offset(tp_rank * q_bytes), dst, q_bytes)?;
        gpu.copy_d2d(
            src.offset(full_kd * row_bytes + tp_rank * q_bytes),
            dst.offset(q_bytes),
            q_bytes,
        )?;
        gpu.copy_d2d(
            src.offset(2 * full_kd * row_bytes + tp_rank * v_bytes),
            dst.offset(2 * q_bytes),
            v_bytes,
        )?;
        Ok(DenseWeight { weight: dst })
    };

    let slice_f32_1d = |src: DevicePtr| -> Result<DenseWeight> {
        let bytes = local_nv * 4;
        let dst = gpu.alloc(bytes)?;
        gpu.copy_d2d(src.offset(tp_rank * bytes), dst, bytes)?;
        Ok(DenseWeight { weight: dst })
    };

    use crate::tp_shard::{TpShardKind, shard_dense_weight};
    let in_proj_qkv = slice_qkv_planar(full_qkv.weight, h * 2)?;
    free_if_not_store_alias(&format!("{p}.in_proj_qkv.weight"), full_qkv.weight)?;

    let (in_proj_z, z_n, _) = shard_dense_weight(
        &full_z,
        full_z_size,
        h,
        TpShardKind::ColumnParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    debug_assert_eq!(z_n, local_z_size);
    free_if_not_store_alias(&format!("{p}.in_proj_z.weight"), full_z.weight)?;

    let (in_proj_a, _, _) = shard_dense_weight(
        &full_a,
        full_nv,
        h,
        TpShardKind::ColumnParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    free_if_not_store_alias(&format!("{p}.in_proj_a.weight"), full_a.weight)?;

    let (in_proj_b, _, _) = shard_dense_weight(
        &full_b,
        full_nv,
        h,
        TpShardKind::ColumnParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    free_if_not_store_alias(&format!("{p}.in_proj_b.weight"), full_b.weight)?;

    let conv1d = slice_qkv_planar(full_conv1d.weight, d_conv * 2)?;
    free_if_not_store_alias(&format!("{p}.conv1d.weight"), full_conv1d.weight)?;

    let a_log = slice_f32_1d(full_a_log.weight)?;
    free_if_not_store_alias(&format!("{p}.A_log"), full_a_log.weight)?;

    let dt_bias = slice_f32_1d(full_dt_bias.weight)?;
    free_if_not_store_alias(&format!("{p}.dt_bias"), full_dt_bias.weight)?;

    let (out_proj, _, _) = shard_dense_weight(
        &full_out_proj,
        h,
        full_vd,
        TpShardKind::RowParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    free_if_not_store_alias(&format!("{p}.out_proj.weight"), full_out_proj.weight)?;

    Ok(SsmWeightsQwen35 {
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        conv1d,
        a_log,
        dt_bias,
        norm,
        out_proj,
    })
}

/// Load MoE weights for Qwen3.5, auto-selecting NVFP4 naming convention.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
/// `skip_routed_experts`: when true, routed experts get NULL weights (saves memory
/// when native FP8 MoE dispatch handles them). Shared expert is always loaded.
pub(crate) fn load_moe_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    skip_routed_experts: bool,
    shared_bf16_requested: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;
    let shared_expert_gate = dense(store, &format!("{p}.shared_expert_gate.weight"))?;

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let load_bf16_then_nvfp4 = |full_prefix: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let bf16 = dense(store, &format!("{full_prefix}.weight"))?;
        quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)
    };

    // Qwen3.6-35B-A3B BF16 release ships a FUSED MoE layout: one
    // `experts.gate_up_proj: [num_experts, 2*inter, hidden]` and one
    // `experts.down_proj: [num_experts, hidden, inter]` per layer. Slice
    // each expert at load time and runtime-quantize to NVFP4.
    let fused_gate_up_key = format!("{p}.experts.gate_up_proj");
    let fused_down_key = format!("{p}.experts.down_proj");
    let numbered_expert_key = format!("{p}.experts.0.gate_proj.weight");
    let has_numbered_experts = store.contains(&numbered_expert_key);
    let has_fused_gate_up = store.contains(&fused_gate_up_key);
    let has_fused_down = store.contains(&fused_down_key);
    if !has_numbered_experts {
        ensure!(
            has_fused_gate_up == has_fused_down,
            "packed BF16 expert bank for {p} is partial"
        );
    }
    let is_fused_bf16 = !has_numbered_experts && has_fused_gate_up;
    let cache_official_packed_mtp = config.is_qwen4_exp() && p == "mtp.layers.0.mlp";
    if is_fused_bf16 {
        ensure!(num_experts > 0, "packed BF16 expert bank cannot be empty");
        let gate_up = store.get(&fused_gate_up_key)?;
        let down = store.get(&fused_down_key)?;
        let gate_up_rows = inter
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("packed BF16 gate/up shape overflow"))?;
        ensure!(
            gate_up.dtype == WeightDtype::BF16 && gate_up.shape == [num_experts, gate_up_rows, h],
            "{fused_gate_up_key} must be BF16 [{num_experts}, {gate_up_rows}, {h}]"
        );
        ensure!(
            down.dtype == WeightDtype::BF16 && down.shape == [num_experts, h, inter],
            "{fused_down_key} must be BF16 [{num_experts}, {h}, {inter}]"
        );
        let _ = crate::weight_loader::qwen4_mtp::packed_bf16_expert_offsets(
            num_experts - 1,
            num_experts,
            inter,
            h,
        )?;
        tracing::info!(
            experts = num_experts,
            "Loading exact packed BF16 expert bank with runtime NVFP4 quantization"
        );
    }

    let load_expert_fused = |expert_idx: usize| -> Result<ExpertWeight> {
        // gate_up: [num_experts, 2*inter, hidden] BF16
        let fused_gu = store.get(&fused_gate_up_key)?;
        // down: [num_experts, hidden, inter] BF16
        let fused_d = store.get(&fused_down_key)?;
        let offsets = crate::weight_loader::qwen4_mtp::packed_bf16_expert_offsets(
            expert_idx,
            num_experts,
            inter,
            h,
        )?;
        Ok(ExpertWeight {
            gate_proj: quantize_packed_mtp_slice(
                cache_official_packed_mtp,
                &fused_gate_up_key,
                fused_gu,
                expert_idx,
                PackedMtpProjection::Gate,
                offsets.gate,
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?,
            up_proj: quantize_packed_mtp_slice(
                cache_official_packed_mtp,
                &fused_gate_up_key,
                fused_gu,
                expert_idx,
                PackedMtpProjection::Up,
                offsets.up,
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?,
            down_proj: quantize_packed_mtp_slice(
                cache_official_packed_mtp,
                &fused_down_key,
                fused_d,
                expert_idx,
                PackedMtpProjection::Down,
                offsets.down,
                h,
                inter,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?,
        })
    };

    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        // Quantization recipes may exclude the shared expert while keeping the
        // routed experts in the checkpoint's native format.  Flash-Next NVFP4
        // does exactly that, so the model-wide variant is not sufficient to
        // choose the loader here.  Inspect the projection itself and perform
        // the established load-time BF16 -> NVFP4 transform when necessary.
        let weight_is_bf16 = |projection: &str| -> bool {
            store
                .get(&format!("{prefix}.{projection}.weight"))
                .map(|weight| weight.dtype == WeightDtype::BF16)
                .unwrap_or(false)
        };
        if weight_is_bf16("gate_proj") && weight_is_bf16("up_proj") && weight_is_bf16("down_proj") {
            return Ok(ExpertWeight {
                gate_proj: load_bf16_then_nvfp4(&format!("{prefix}.gate_proj"), inter, h)?,
                up_proj: load_bf16_then_nvfp4(&format!("{prefix}.up_proj"), inter, h)?,
                down_proj: load_bf16_then_nvfp4(&format!("{prefix}.down_proj"), h, inter)?,
            });
        }
        let declared = |projection: &str| {
            config.quantization_config.as_ref().and_then(|quant| {
                quant.modelopt_weight_format_for(&format!("{prefix}.{projection}"))
            })
        };
        if declared("gate_proj").is_some()
            || declared("up_proj").is_some()
            || declared("down_proj").is_some()
        {
            ensure!(
                declared("gate_proj").is_some()
                    && declared("up_proj").is_some()
                    && declared("down_proj").is_some(),
                "mixed ModelOpt expert {prefix} has a partial projection declaration"
            );
            return Ok(ExpertWeight {
                gate_proj: quantized_modelopt_mixed(
                    store,
                    &format!("{prefix}.gate_proj"),
                    inter,
                    h,
                    config,
                    variant,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                up_proj: quantized_modelopt_mixed(
                    store,
                    &format!("{prefix}.up_proj"),
                    inter,
                    h,
                    config,
                    variant,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                down_proj: quantized_modelopt_mixed(
                    store,
                    &format!("{prefix}.down_proj"),
                    h,
                    inter,
                    config,
                    variant,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
            });
        }
        match variant {
            Nvfp4Variant::Bf16Raw => Ok(ExpertWeight {
                gate_proj: load_bf16_then_nvfp4(&format!("{prefix}.gate_proj"), inter, h)?,
                up_proj: load_bf16_then_nvfp4(&format!("{prefix}.up_proj"), inter, h)?,
                down_proj: load_bf16_then_nvfp4(&format!("{prefix}.down_proj"), h, inter)?,
            }),
            Nvfp4Variant::Fp8Dequanted => Ok(ExpertWeight {
                gate_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                up_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.up_proj"),
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
                down_proj: quantized_from_fp8(
                    store,
                    &format!("{prefix}.down_proj"),
                    h,
                    inter,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?,
            }),
            _ => Ok(ExpertWeight {
                gate_proj: quantized_auto(store, &format!("{prefix}.gate_proj"), gpu, variant)?,
                up_proj: quantized_auto(store, &format!("{prefix}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(store, &format!("{prefix}.down_proj"), gpu, variant)?,
            }),
        }
    };

    // `ATLAS_QWEN4_SHARED_BF16=1`: keep the shared expert's exact BF16
    // dequant ALONGSIDE (not instead of) the usual NVFP4 re-quantization.
    // `shared_expert` is always built via `load_expert`, exactly as
    // without this flag, so prefill (`forward_prefill_phase.rs`,
    // `qwen4_shared_v7.rs`) keeps reading a real NVFP4 shared expert —
    // this control has no prefill arm, decode-only
    // (`layers/moe/qwen4_shared_bf16.rs`). The control is purely additive:
    // `shared_bf16` is an extra resident copy the decode dispatch
    // (`forward_phase.rs`) overwrites `shared_out` with, after the fused
    // NVFP4 kernels already computed (and discarded) their own answer.
    let shared_expert = load_expert(&format!("{p}.shared_expert"))?;
    let shared_bf16 = if shared_bf16_requested {
        let prefix = format!("{p}.shared_expert");
        Some(SharedExpertBf16 {
            gate: modelopt_mixed_to_bf16(
                store,
                &format!("{prefix}.gate_proj"),
                inter,
                h,
                config,
                gpu,
            )?,
            up: modelopt_mixed_to_bf16(store, &format!("{prefix}.up_proj"), inter, h, config, gpu)?,
            down: modelopt_mixed_to_bf16(
                store,
                &format!("{prefix}.down_proj"),
                h,
                inter,
                config,
                gpu,
            )?,
        })
    } else {
        None
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else if is_fused_bf16 {
            experts.push(load_expert_fused(e)?);
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_bf16,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// Load routed MoE experts as native FP8 weights (no NVFP4 conversion).
///
/// Returns a `Vec<Fp8ExpertWeight>` (with NULL placeholders for non-local EP
/// experts) for native FP8 dispatch; callers load the shared expert separately.
pub(crate) fn load_moe_qwen35_fp8_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<Vec<Fp8ExpertWeight>> {
    let p = format!("{layer_prefix}.mlp");
    let mut fp8_experts = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.gate_proj"),
                    gpu,
                )?,
                up_proj: load_fp8_block_scaled_as_fp8weight(store, &format!("{ep}.up_proj"), gpu)?,
                down_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.down_proj"),
                    gpu,
                )?,
            });
        } else {
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                up_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
                down_proj: Fp8Weight {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                },
            });
        }
    }

    Ok(fp8_experts)
}

/// Load MoE weights for models without shared experts (e.g. Qwen3-VL).
///
/// Creates zero-filled dummy shared expert weights so the fused MoE kernels
/// (which always launch top_k+1 blocks) produce zero contribution from the
/// shared expert slot. `weight_scale_2 = 0.0` ensures dequant → 0.
pub(crate) fn load_moe_no_shared(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;

    // Allocate correctly-sized zero-filled GPU buffers for dummy shared expert.
    // The fused kernel always runs a shared expert block (blockIdx.y == top_k),
    // which reads full expert-sized weight matrices. Buffers must match real
    // expert dimensions or the kernel will read out of bounds (CUDA error 900).
    // weight_scale_2 = 0.0 ensures dequant → 0 regardless of packed contents.
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize; // NVFP4 quantization group size (matches kernel GROUP_SIZE)

    // gate_proj/up_proj: [inter, h] → packed = inter * h / 2, scale = inter * (h / group_size)
    let gu_packed_bytes = inter * h / 2;
    let gu_scale_bytes = inter * (h / group_size);
    // down_proj: [h, inter] → packed = h * inter / 2, scale = h * (inter / group_size)
    let d_packed_bytes = h * inter / 2;
    let d_scale_bytes = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };

    let mk_zero_quant = |packed_sz: usize, scale_sz: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed_sz)?,
            weight_scale: alloc_zero(scale_sz)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        })
    };

    let shared_expert = ExpertWeight {
        gate_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        up_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        down_proj: mk_zero_quant(d_packed_bytes, d_scale_bytes)?,
    };
    // Gate weight for shared expert: zero BF16 [hidden_size] → sigmoid(0)=0.5.
    // Doesn't matter since shared_out is all zeros (0.5 * 0 = 0).
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    gpu,
                    variant,
                )?,
                up_proj: quantized_auto(store, &format!("{p}.experts.{e}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    gpu,
                    variant,
                )?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_bf16: None,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::{KernelHandle, mock::MockGpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::*;

    struct ByteCopyingGpu(MockGpuBackend);

    impl GpuBackend for ByteCopyingGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.0.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.0.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.0.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.0.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.0.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            let mut tmp = vec![0u8; bytes];
            self.0.copy_d2h(src, &mut tmp)?;
            self.0.copy_h2d(&tmp, dst)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.0.launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.0.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.0.default_stream()
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.0.kernel(module, func_name)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.0.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.0.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.0.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.0.free_memory()
        }
    }

    #[test]
    fn load_ssm_qwen35_tp2_slices_planar_qkv_and_conv1d_without_mixing_planes() {
        let gpu = ByteCopyingGpu(MockGpuBackend::new());
        let lp = "model.layers.0";
        let p = "model.layers.0.linear_attn";
        let h = 16usize;
        let d_conv = 4usize;
        let full_nk = 4usize;
        let full_nv = 8usize;
        let dk = 4usize;
        let dv = 4usize;
        let full_kd = full_nk * dk; // 16
        let full_vd = full_nv * dv; // 32
        let full_qkv_rows = full_kd * 2 + full_vd; // 64

        let mut map = HashMap::new();
        // Fill planar [Q(16) | K(16) | V(32)] with distinct u16 markers per plane & half:
        // Q0 = 0x10, Q1 = 0x11, K0 = 0x20, K1 = 0x21, V0 = 0x30, V1 = 0x31.
        let mut qkv_host = Vec::with_capacity(full_qkv_rows * h * 2);
        for r in 0..full_qkv_rows {
            let tag: u8 = if r < 8 {
                0x10
            } else if r < 16 {
                0x11
            } else if r < 24 {
                0x20
            } else if r < 32 {
                0x21
            } else if r < 48 {
                0x30
            } else {
                0x31
            };
            qkv_host.extend(std::iter::repeat_n(tag, h * 2));
        }
        let qkv_ptr = gpu.alloc(qkv_host.len()).unwrap();
        gpu.copy_h2d(&qkv_host, qkv_ptr).unwrap();
        map.insert(
            format!("{p}.in_proj_qkv.weight"),
            WeightTensor {
                ptr: qkv_ptr,
                shape: vec![full_qkv_rows, h],
                dtype: WeightDtype::BF16,
            },
        );

        let mut conv_host = Vec::with_capacity(full_qkv_rows * d_conv * 2);
        for r in 0..full_qkv_rows {
            let tag: u8 = if r < 8 {
                0xA0
            } else if r < 16 {
                0xA1
            } else if r < 24 {
                0xB0
            } else if r < 32 {
                0xB1
            } else if r < 48 {
                0xC0
            } else {
                0xC1
            };
            conv_host.extend(std::iter::repeat_n(tag, d_conv * 2));
        }
        let conv_ptr = gpu.alloc(conv_host.len()).unwrap();
        gpu.copy_h2d(&conv_host, conv_ptr).unwrap();
        map.insert(
            format!("{p}.conv1d.weight"),
            WeightTensor {
                ptr: conv_ptr,
                shape: vec![full_qkv_rows, 1, d_conv],
                dtype: WeightDtype::BF16,
            },
        );

        let mut add_bf16 = |suffix: &str, rows: usize, cols: usize, fill: u8| {
            let bytes = vec![fill; rows * cols * 2];
            let ptr = gpu.alloc(bytes.len()).unwrap();
            gpu.copy_h2d(&bytes, ptr).unwrap();
            map.insert(
                format!("{p}.{suffix}"),
                WeightTensor {
                    ptr,
                    shape: vec![rows, cols],
                    dtype: WeightDtype::BF16,
                },
            );
        };
        add_bf16("in_proj_z.weight", full_vd, h, 0x44);
        add_bf16("in_proj_a.weight", full_nv, h, 0x55);
        add_bf16("in_proj_b.weight", full_nv, h, 0x66);
        add_bf16("norm.weight", dv, 1, 0x77);
        add_bf16("out_proj.weight", h, full_vd, 0x88);

        for suffix in ["A_log", "dt_bias"] {
            let mut f32_bytes = Vec::with_capacity(full_nv * 4);
            for i in 0..full_nv {
                f32_bytes.extend_from_slice(&((i as f32) + 1.0).to_le_bytes());
            }
            let ptr = gpu.alloc(f32_bytes.len()).unwrap();
            gpu.copy_h2d(&f32_bytes, ptr).unwrap();
            map.insert(
                format!("{p}.{suffix}"),
                WeightTensor {
                    ptr,
                    shape: vec![full_nv],
                    dtype: WeightDtype::FP32,
                },
            );
        }

        let store = WeightStore::from_map(map);
        for (rank, q_tag, k_tag, v_tag, cq, ck, cv) in [
            (0usize, 0x10u8, 0x20u8, 0x30u8, 0xA0u8, 0xB0u8, 0xC0u8),
            (1usize, 0x11u8, 0x21u8, 0x31u8, 0xA1u8, 0xB1u8, 0xC1u8),
        ] {
            let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
            cfg.model_type = "qwen4_exp_moe".to_string();
            cfg.num_hidden_layers = 4;
            cfg.hidden_size = h;
            cfg.vocab_size = 64;
            cfg.num_attention_heads = 4;
            cfg.num_key_value_heads = 2;
            cfg.head_dim = 4;
            cfg.tp_world_size = 2;
            cfg.tp_rank = rank;
            cfg.linear_num_key_heads = full_nk / 2;
            cfg.linear_num_value_heads = full_nv / 2;
            cfg.linear_key_head_dim = dk;
            cfg.linear_value_head_dim = dv;
            cfg.linear_conv_kernel_dim = d_conv;

            let loaded = load_ssm_qwen35(&store, lp, &gpu, Nvfp4Variant::Standard, &cfg).unwrap();
            let local_qkv_rows = full_qkv_rows / 2; // 32 = 8 Q + 8 K + 16 V
            let mut got_qkv = vec![0u8; local_qkv_rows * h * 2];
            gpu.copy_d2h(loaded.in_proj_qkv.weight, &mut got_qkv)
                .unwrap();
            assert!(got_qkv[..8 * h * 2].iter().all(|&b| b == q_tag));
            assert!(got_qkv[8 * h * 2..16 * h * 2].iter().all(|&b| b == k_tag));
            assert!(got_qkv[16 * h * 2..].iter().all(|&b| b == v_tag));

            let mut got_conv = vec![0u8; local_qkv_rows * d_conv * 2];
            gpu.copy_d2h(loaded.conv1d.weight, &mut got_conv).unwrap();
            assert!(got_conv[..8 * d_conv * 2].iter().all(|&b| b == cq));
            assert!(
                got_conv[8 * d_conv * 2..16 * d_conv * 2]
                    .iter()
                    .all(|&b| b == ck)
            );
            assert!(got_conv[16 * d_conv * 2..].iter().all(|&b| b == cv));

            let mut got_alog = vec![0u8; (full_nv / 2) * 4];
            gpu.copy_d2h(loaded.a_log.weight, &mut got_alog).unwrap();
            let first_val = f32::from_le_bytes(got_alog[0..4].try_into().unwrap());
            assert_eq!(first_val, (rank * (full_nv / 2) + 1) as f32);
        }
    }
}
