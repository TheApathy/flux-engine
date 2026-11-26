// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Load an FP8 E4M3 block-scaled checkpoint weight as a native [`Fp8Weight`].
///
/// The FP8 checkpoint stores:
///   - `{prefix}.weight`: FP8E4M3 tensor [N, K]
///   - `{prefix}.weight_scale_inv`: BF16 tensor [N/block, K/block]
///
/// The `w8a16_gemv` kernel uses 2D block scales directly:
///   `dequant[i,j] = E4M3_LUT[fp8[i,j]] * block_scale[i/BS, j/BS]`
/// No per-row max reduction needed — the kernel loads the correct block
/// scale for each 128-element K chunk.
pub fn load_fp8_block_scaled_as_fp8weight(
    store: &WeightStore,
    prefix: &str,
    _gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let weight_ptr = w.ptr;

    // Load block scale_inv: BF16 [N/BS, K/BS] — already on GPU from safetensors
    let scale_key = format!("{prefix}.weight_scale_inv");
    let s = store.get(&scale_key)?;
    ensure!(
        s.dtype == WeightDtype::BF16,
        "Expected BF16 for {scale_key}, got {:?}",
        s.dtype,
    );
    let expected_scale_shape = [n.div_ceil(128), k.div_ceil(128)];
    ensure!(
        s.shape == expected_scale_shape,
        "Expected 128x128 block scale shape {:?} for {scale_key}, got {:?}",
        expected_scale_shape,
        s.shape,
    );

    tracing::debug!(
        "FP8 block scales: {prefix} [{n},{k}] scale=[{},{}]",
        s.shape[0],
        s.shape[1],
    );

    // Pass block scales directly — the w8a16_gemv kernel handles 2D indexing
    Ok(Fp8Weight {
        weight: weight_ptr,
        row_scale: s.ptr, // BF16 [N/BS, K/BS] block scales on GPU
        n: n as u32,
        k: k as u32,
    })
}

/// Quantize a BF16 dense weight to NVFP4 on GPU.
///
/// Two-phase: (1) find global max, (2) per-group E2M1 quantization.
/// Halves weight bandwidth vs FP8 (0.5 bytes/weight + group scales vs 1 byte/weight).
/// Called once at model load time (not on the hot path).
pub(crate) fn quantize_to_nvfp4(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_kernel: spark_runtime::gpu::KernelHandle,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    use spark_runtime::kernel_args::KernelLaunch;

    let total = n * k;

    // Phase 1: Find global absolute max
    let max_buf = gpu.alloc(4)?;
    gpu.memset(max_buf, 0, 4)?;

    let grid1 = (total / 256).clamp(1, 1024) as u32;
    KernelLaunch::new(gpu, absmax_kernel)
        .grid([grid1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(max_buf)
        .arg_u32(total as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;
    let mut max_bytes = [0u8; 4];
    gpu.copy_d2h(max_buf, &mut max_bytes)?;
    // The absmax pass is complete (synchronized above) and the value is on the
    // host: free the 4-byte slot instead of leaking one allocation per
    // quantized weight (~600 on Flash-Next).
    gpu.free(max_buf)?;
    let global_max = f32::from_le_bytes(max_bytes);

    // scale2 = global_max / (6.0 * 448.0)  [FP8 E4M3 max = 448]
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };

    // Diagnostic: log absmax result for first few quantizations
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static QUANT_DIAG: AtomicUsize = AtomicUsize::new(0);
        if QUANT_DIAG.fetch_add(1, Ordering::Relaxed) < 5 {
            tracing::info!(
                "quantize_to_nvfp4: n={n} k={k} total={total} global_max={global_max:.6} scale2={scale2:.8} grid1={grid1}",
            );
        }
    }

    // Phase 2: Quantize
    let packed_buf = gpu.alloc(n * k / 2)?;
    let scale_buf = gpu.alloc(n * k / 16)?;

    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(packed_buf)
        .arg_ptr(scale_buf)
        .arg_f32(scale2)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;

    Ok(QuantizedWeight {
        weight: packed_buf,
        weight_scale: scale_buf,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    })
}

/// Content-keyed variant of [`quantize_to_nvfp4`] used by Qwen4 runtime
/// transforms. It caches the packed payload, group scales, and scalar scale2
/// together; a cache hit therefore skips both the global-absmax pass and the
/// quantization kernel without weakening the cache's provenance contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_to_nvfp4_cached(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_kernel: spark_runtime::gpu::KernelHandle,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
    slot: &str,
) -> Result<QuantizedWeight> {
    use crate::weight_loader::transform_cache;

    let Some(cache) = transform_cache::get() else {
        return quantize_to_nvfp4(
            bf16_weight,
            n,
            k,
            gpu,
            absmax_kernel,
            quantize_kernel,
            stream,
        );
    };
    let lens = [n * k / 2, n * k / 16, 4];
    if let Some(parts) = cache.load_parts(slot, gpu, &lens) {
        let mut scale_bytes = [0u8; 4];
        if let Err(err) = gpu.copy_d2h(parts[2], &mut scale_bytes) {
            let _ = gpu.free(parts[0]);
            let _ = gpu.free(parts[1]);
            let _ = gpu.free(parts[2]);
            return Err(err);
        }
        let hit = QuantizedWeight {
            weight: parts[0],
            weight_scale: parts[1],
            weight_scale_2: f32::from_le_bytes(scale_bytes),
            input_scale: DevicePtr::NULL,
        };
        if !cache.verify_enabled() {
            if let Err(err) = gpu.free(parts[2]) {
                let _ = gpu.free(hit.weight);
                let _ = gpu.free(hit.weight_scale);
                return Err(err);
            }
            return Ok(hit);
        }
        let fresh = match quantize_to_nvfp4(
            bf16_weight,
            n,
            k,
            gpu,
            absmax_kernel,
            quantize_kernel,
            stream,
        ) {
            Ok(fresh) => fresh,
            Err(err) => {
                let _ = gpu.free(parts[0]);
                let _ = gpu.free(parts[1]);
                let _ = gpu.free(parts[2]);
                return Err(err);
            }
        };
        let fresh_scale_dev = match gpu.alloc(4) {
            Ok(ptr) => ptr,
            Err(err) => {
                let _ = gpu.free(parts[0]);
                let _ = gpu.free(parts[1]);
                let _ = gpu.free(parts[2]);
                let _ = gpu.free(fresh.weight);
                let _ = gpu.free(fresh.weight_scale);
                return Err(err);
            }
        };
        if let Err(err) = gpu.copy_h2d(&fresh.weight_scale_2.to_le_bytes(), fresh_scale_dev) {
            let _ = gpu.free(fresh_scale_dev);
            let _ = gpu.free(parts[0]);
            let _ = gpu.free(parts[1]);
            let _ = gpu.free(parts[2]);
            let _ = gpu.free(fresh.weight);
            let _ = gpu.free(fresh.weight_scale);
            return Err(err);
        }
        let pair_valid = cache.verify_pair(
            slot,
            gpu,
            &[
                (hit.weight, lens[0]),
                (hit.weight_scale, lens[1]),
                (parts[2], lens[2]),
            ],
            &[
                (fresh.weight, lens[0]),
                (fresh.weight_scale, lens[1]),
                (fresh_scale_dev, lens[2]),
            ],
        );
        let scale_match = fresh.weight_scale_2.to_bits() == hit.weight_scale_2.to_bits();
        if !scale_match {
            tracing::error!(
                "weight cache VERIFY FAIL: {slot}[2] weight_scale_2 mismatch: cached={} (0x{:08x}) vs fresh={} (0x{:08x})",
                hit.weight_scale_2,
                hit.weight_scale_2.to_bits(),
                fresh.weight_scale_2,
                fresh.weight_scale_2.to_bits(),
            );
        }
        let valid = pair_valid && scale_match;
        let _ = gpu.free(fresh_scale_dev);
        let _ = gpu.free(parts[2]);
        if valid {
            gpu.free(fresh.weight)?;
            gpu.free(fresh.weight_scale)?;
            return Ok(hit);
        }
        gpu.free(hit.weight)?;
        gpu.free(hit.weight_scale)?;
        return Ok(fresh);
    }

    let fresh = quantize_to_nvfp4(
        bf16_weight,
        n,
        k,
        gpu,
        absmax_kernel,
        quantize_kernel,
        stream,
    )?;
    let scale_dev = match gpu.alloc(4) {
        Ok(ptr) => ptr,
        Err(err) => {
            let _ = gpu.free(fresh.weight);
            let _ = gpu.free(fresh.weight_scale);
            return Err(err);
        }
    };
    if let Err(err) = gpu.copy_h2d(&fresh.weight_scale_2.to_le_bytes(), scale_dev) {
        let _ = gpu.free(scale_dev);
        let _ = gpu.free(fresh.weight);
        let _ = gpu.free(fresh.weight_scale);
        return Err(err);
    }
    cache.store_parts(
        slot,
        gpu,
        &[
            (fresh.weight, lens[0]),
            (fresh.weight_scale, lens[1]),
            (scale_dev, lens[2]),
        ],
    );
    gpu.free(scale_dev)?;
    Ok(fresh)
}

/// Load attention weights for a full_attention layer.
pub(crate) fn load_attention(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &atlas_core::config::ModelConfig,
) -> Result<AttentionWeights> {
    let p = format!("{layer_prefix}.self_attn");
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let h = config.hidden_size;
    // Use full pre-shard Q/O width because `WeightStore` holds unsharded tensors;
    // TP loaders (e.g. `weight_loader::qwen3`) shard `o_proj` after `load_attention`.
    let full_qkv_out = config.num_attention_heads * config.tp_world_size.max(1) * config.head_dim;
    let _kv_out = config.num_key_value_heads * config.head_dim;
    Ok(AttentionWeights {
        q_proj: dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?,
        k_proj: dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?,
        v_proj: dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?,
        o_proj: quantized_any(
            store,
            &format!("{p}.o_proj"),
            h,
            full_qkv_out,
            gpu,
            variant,
            qctx,
        )?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    })
}

/// Load SSM weights for a linear_attention layer.
pub(crate) fn load_ssm(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &atlas_core::config::ModelConfig,
) -> Result<SsmWeights> {
    let p = format!("{layer_prefix}.linear_attn");
    let h = config.hidden_size;
    // out_proj is [hidden_size, d_inner] where d_inner = linear_value_head_dim * linear_num_value_heads
    let d_inner = config.linear_value_head_dim * config.linear_num_value_heads;
    Ok(SsmWeights {
        in_proj_qkvz: dense_auto(store, &format!("{p}.in_proj_qkvz.weight"), gpu)?,
        in_proj_ba: dense_auto(store, &format!("{p}.in_proj_ba.weight"), gpu)?,
        conv1d: dense(store, &format!("{p}.conv1d.weight"))?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense(store, &format!("{p}.norm.weight"))?,
        out_proj: quantized_any(
            store,
            &format!("{p}.out_proj"),
            h,
            d_inner,
            gpu,
            variant,
            qctx,
        )?,
    })
}

/// Load MoE weights for a layer.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
pub(crate) fn load_moe(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        false,
    )
}

/// Load MoE with option to skip routed experts (native FP8 loads them separately).
pub(crate) fn load_moe_skip_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        true,
    )
}

#[cfg(test)]
mod quantize_allocation_tests {
    use spark_runtime::gpu::{GpuBackend, KernelHandle, mock::MockGpuBackend};

    use super::quantize_to_nvfp4;
    use crate::weight_map::DenseWeight;

    /// One quantization leaves exactly its packed payload and group scales
    /// allocated: the absmax slot is scratch, not part of the weight.
    #[test]
    fn quantize_to_nvfp4_keeps_only_payload_and_scales() {
        let gpu = MockGpuBackend::new();
        let (n, k) = (64usize, 256usize);
        let bf16 = DenseWeight {
            weight: gpu.alloc(n * k * 2).unwrap(),
        };
        assert_eq!(gpu.alloc_count(), 1);
        let q = quantize_to_nvfp4(
            &bf16,
            n,
            k,
            &gpu,
            KernelHandle(1),
            KernelHandle(2),
            gpu.default_stream(),
        )
        .unwrap();
        assert_eq!(
            gpu.alloc_count(),
            3,
            "source + payload + scales, no absmax slot"
        );
        assert!(
            gpu.read_alloc(q.weight)
                .is_some_and(|b| b.len() == n * k / 2)
        );
        assert!(
            gpu.read_alloc(q.weight_scale)
                .is_some_and(|b| b.len() == n * k / 16)
        );
    }
}
