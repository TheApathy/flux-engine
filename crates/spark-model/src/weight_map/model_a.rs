// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// All model weights organized by layer.
pub struct ModelWeights {
    /// Embedding table: [vocab_size, hidden_size] BF16.
    pub embed_tokens: DenseWeight,
    /// Final RMS norm: `[hidden_size]` BF16.
    pub final_norm: DenseWeight,
    /// LM head: [vocab_size, hidden_size] BF16.
    pub lm_head: DenseWeight,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
}

/// Extract a DevicePtr from the weight store by name.
pub(crate) fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store.get(name)?.ptr)
}

/// Extract a scalar f32 from a single-element weight tensor via D2H copy.
pub(crate) fn scalar_f32(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    ensure!(
        w.num_elements() == 1,
        "Expected scalar for {name}, got {} elements",
        w.num_elements()
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(w.ptr, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Load FP8 KV cache quantization scales from a checkpoint.
///
/// Searches for `{attn_prefix}.k_proj.k_scale` and `{attn_prefix}.v_proj.v_scale`
/// tensors in the weight store. When found, returns the scalar f32 values from the
/// checkpoint (calibrated per-tensor scales for FP8 KV cache quantization).
///
/// When not found, returns (1.0, 1.0) with a debug-level log. This fallback is
/// correct for BF16 KV cache (scales are unused) and acceptable as uncalibrated
/// default for FP8 KV cache (Workstream 4C will add proper calibration).
pub(crate) fn load_kv_scales(
    store: &WeightStore,
    attn_prefix: &str,
    gpu: &dyn GpuBackend,
) -> (f32, f32) {
    let k_key = format!("{attn_prefix}.k_proj.k_scale");
    let v_key = format!("{attn_prefix}.v_proj.v_scale");

    let k_scale = if store.contains(&k_key) {
        match scalar_f32(store, &k_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded k_scale={v:.6} from {k_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {k_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {k_key} in checkpoint, using k_scale=1.0");
        1.0
    };

    let v_scale = if store.contains(&v_key) {
        match scalar_f32(store, &v_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded v_scale={v:.6} from {v_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {v_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {v_key} in checkpoint, using v_scale=1.0");
        1.0
    };

    (k_scale, v_scale)
}

/// Build a QuantizedWeight from the store using the standard NVFP4 naming.
///
/// `weight_scale_2` is a single FP32 scalar — extracted from GPU via D2H copy.
pub(crate) fn quantized(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let input_scale_key = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&input_scale_key) {
        let tensor = store.get(&input_scale_key)?;
        validate_modelopt_input_scale(&input_scale_key, tensor.dtype, &tensor.shape)?;
        tensor.ptr
    } else {
        DevicePtr::NULL
    };
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?,
        input_scale,
    })
}

/// Load a ModelOpt NVFP4 projection while retaining the caller's exact
/// logical geometry as an admission boundary.
///
/// The ordinary pointer-only helper predates consumers whose native ABI reads
/// the complete packed and scale matrices directly.  Such consumers must not
/// infer allocation lengths from configuration after accepting a merely
/// one-element scalar, so this path validates every checkpoint tensor rank,
/// dtype, and extent before exposing its pointers.
pub(crate) fn quantized_modelopt_with_shape(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    ensure!(n > 0, "ModelOpt NVFP4 output dimension must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "ModelOpt NVFP4 input dimension must be a positive multiple of 16"
    );

    let weight_name = format!("{prefix}.weight");
    let weight = store.get(&weight_name)?;
    ensure!(
        weight.dtype == WeightDtype::UInt8,
        "Expected packed UInt8 ModelOpt weight for {weight_name}, got {:?}",
        weight.dtype
    );
    ensure!(
        weight.shape.as_slice() == [n, k / 2],
        "Expected ModelOpt weight {weight_name} shape [{n}, {}], got {:?}",
        k / 2,
        weight.shape
    );

    let scale_name = format!("{prefix}.weight_scale");
    let weight_scale = store.get(&scale_name)?;
    ensure!(
        weight_scale.dtype == WeightDtype::FP8E4M3,
        "Expected FP8 E4M3 ModelOpt block scales for {scale_name}, got {:?}",
        weight_scale.dtype
    );
    ensure!(
        weight_scale.shape.as_slice() == [n, k / 16],
        "Expected ModelOpt scale {scale_name} shape [{n}, {}], got {:?}",
        k / 16,
        weight_scale.shape
    );

    let scale2_name = format!("{prefix}.weight_scale_2");
    let scale2 = store.get(&scale2_name)?;
    ensure!(
        scale2.dtype == WeightDtype::FP32 && scale2.shape.is_empty(),
        "Expected rank-0 FP32 ModelOpt second-level scale for {scale2_name}, got {:?} {:?}",
        scale2.dtype,
        scale2.shape
    );

    let input_scale_name = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&input_scale_name) {
        let tensor = store.get(&input_scale_name)?;
        validate_modelopt_input_scale(&input_scale_name, tensor.dtype, &tensor.shape)?;
        tensor.ptr
    } else {
        DevicePtr::NULL
    };

    Ok(QuantizedWeight {
        weight: weight.ptr,
        weight_scale: weight_scale.ptr,
        weight_scale_2: scalar_f32(store, &scale2_name, gpu)?,
        input_scale,
    })
}

fn validate_modelopt_input_scale(name: &str, dtype: WeightDtype, shape: &[usize]) -> Result<()> {
    ensure!(
        dtype == WeightDtype::FP32,
        "Expected FP32 ModelOpt input scale for {name}, got {dtype:?}"
    );
    ensure!(
        shape.is_empty(),
        "Expected rank-0 ModelOpt input scale for {name}, got shape {shape:?}"
    );
    Ok(())
}

pub(crate) fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::BF16,
        "dense({name}) requires BF16 tensor, got {:?}; use dense_f32_safe for FP32 norms",
        w.dtype
    );
    Ok(DenseWeight { weight: w.ptr })
}

/// Load a weight, auto-dequanting FP8 block-scaled to BF16 when needed.
///
/// Used for models with mixed-precision layers — Qwen3.6's ViT, for
/// example, keeps the first four blocks in BF16 but stores the rest as
/// FP8. Callers pass a prefix (e.g. `"model.visual.blocks.5.attn.qkv"`)
/// and get back a BF16 GPU buffer either way.
pub(crate) fn dense_auto_fp8_or_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP8E4M3 => dequant_fp8_blockscaled_to_bf16(store, prefix, gpu),
        other => anyhow::bail!(
            "dense_auto_fp8_or_bf16: unsupported dtype {:?} for {prefix}.weight",
            other
        ),
    }
}

/// Load a dense weight, converting FP32 → BF16 if needed. Used for norm weights
/// that may be FP32 in some checkpoints (e.g. Qwen FP8 `linear_attn.norm.weight`
/// or ModelOpt QAD `hc_norm.weight` / `ple.norm_*.weight`).
pub(crate) fn dense_f32_safe(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP32 {
        let n = w.num_elements();
        let bf16_bytes = n
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("dense_f32_safe({name}): BF16 byte size overflow"))?;
        if n == 0 {
            let ptr = gpu.alloc(0)?;
            return Ok(DenseWeight { weight: ptr });
        }
        // On real CUDA backends (`transform_cache_identity` or `sm_count`
        // populated), run the RNE conversion asynchronously on the default
        // stream via `quantize_nvfp4::f32_to_bf16_rne`, avoiding synchronous
        // D2H/H2D stalls across hundreds of cold-load norm weights. Mock and
        // non-CUDA backends (`MockGpuBackend`, Metal) fall back to the host
        // RNE path so simulated memory holds exact converted BF16 bytes.
        if (gpu.transform_cache_identity().is_some() || gpu.sm_count().is_some())
            && let Ok(n_u32) = u32::try_from(n)
            && let Ok(kernel) = gpu.kernel("quantize_nvfp4", "f32_to_bf16_rne")
        {
            tracing::debug!(
                "dense_f32_safe: on-device RNE converting {name} from FP32 to BF16 ({:?})",
                w.shape
            );
            let ptr = gpu.alloc(bf16_bytes)?;
            let blocks = n_u32.div_ceil(256).max(1);
            if let Err(error) = spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
                .grid([blocks, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(w.ptr)
                .arg_ptr(ptr)
                .arg_u32(n_u32)
                .launch(gpu.default_stream())
            {
                let _ = gpu.free(ptr);
                return Err(error).with_context(|| {
                    format!(
                        "dense_f32_safe: GPU f32_to_bf16_rne launch failed for {name} ({:?})",
                        w.shape
                    )
                });
            }
            return Ok(DenseWeight { weight: ptr });
        }

        tracing::info!(
            "dense_f32_safe: converting {name} from FP32 to BF16 ({:?})",
            w.shape
        );
        let f32_bytes = n
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("dense_f32_safe({name}): FP32 byte size overflow"))?;
        let mut f32_buf = vec![0u8; f32_bytes];
        gpu.copy_d2h(w.ptr, &mut f32_buf)?;
        let bf16_buf: Vec<u8> = f32_buf
            .chunks_exact(4)
            .flat_map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                half::bf16::from_f32(v).to_le_bytes()
            })
            .collect();
        let ptr = gpu.alloc(bf16_buf.len())?;
        gpu.copy_h2d(&bf16_buf, ptr)?;
        Ok(DenseWeight { weight: ptr })
    } else {
        ensure!(
            w.dtype == WeightDtype::BF16,
            "dense_f32_safe({name}): expected BF16/FP32, got {:?}",
            w.dtype
        );
        Ok(DenseWeight { weight: w.ptr })
    }
}

/// Load a weight and ensure it's FP32 on GPU, regardless of source dtype.
///
/// Used for SSM gate parameters (A_log, dt_bias) where BF16 precision
/// causes exponential error amplification in the recurrent state at
/// long context (8k+ tokens). A 1-ULP BF16 error in the decay gate
/// produces (g_correct/g_error)^8000 ≈ 3000x magnitude divergence.
///
/// - FP32 in safetensors: keep as-is (no conversion)
/// - BF16 in safetensors: convert BF16 → FP32 via zero-extension
pub(crate) fn dense_keep_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::FP32 => {
            // Already FP32 — use directly, no conversion needed
            Ok(DenseWeight { weight: w.ptr })
        }
        WeightDtype::BF16 => {
            // Convert BF16 → FP32 to preserve precision
            tracing::info!(
                "dense_keep_f32: promoting {name} from BF16 to FP32 ({:?})",
                w.shape
            );
            let n = w.num_elements();
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
            let f32_buf: Vec<u8> = bf16_buf
                .chunks_exact(2)
                .flat_map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f32_bits = (bits as u32) << 16;
                    f32_bits.to_le_bytes()
                })
                .collect();
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(DenseWeight { weight: ptr })
        }
        other => {
            bail!("dense_keep_f32: unsupported dtype {:?} for {name}", other);
        }
    }
}

/// Load a BF16 tensor and convert to F32 on-device via CPU roundtrip.
///
/// Used for Nemotron-H SSM parameters (A_log, D, dt_bias, conv1d.bias)
/// which are stored as BF16 in safetensors but consumed as F32 by CUDA kernels.
pub(crate) fn dense_bf16_as_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::BF16,
        "Expected BF16 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut bf16_buf = vec![0u8; n * 2];
    gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
    let f32_buf: Vec<u8> = bf16_buf
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let f32_bits = (bits as u32) << 16;
            f32_bits.to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(f32_buf.len())?;
    gpu.copy_h2d(&f32_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Load an F32 tensor and convert to BF16 on-device via CPU roundtrip.
///
/// Used for Nemotron-H gate weights (F32 in safetensors, consumed as BF16).
pub(crate) fn dense_f32_as_bf16(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut f32_buf = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut f32_buf)?;
    let bf16_buf: Vec<u8> = f32_buf
        .chunks_exact(4)
        .flat_map(|c| {
            let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            f32_to_bf16(val).to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Dequantize FP8 E4M3 + per-tensor scale → BF16, returning a DenseWeight.
///
/// Allocates a new GPU buffer for the result. Use `dequant_fp8_to_bf16_into` to
/// write into a pre-allocated scratch buffer instead (avoids gpu.free on GB10 UVM).
pub(crate) fn dequant_fp8_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype
    );
    let n_bytes = w.num_elements();
    let mut fp8_buf = vec![0u8; n_bytes];
    gpu.copy_d2h(w.ptr, &mut fp8_buf)?;
    let scale = scalar_f32(store, &format!("{prefix}.weight_scale"), gpu)?;
    let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
    let ptr = gpu.alloc(bf16_buf.len())?;
    if let Err(err) = gpu.copy_h2d(&bf16_buf, ptr) {
        let _ = gpu.free(ptr);
        return Err(err);
    }
    Ok(DenseWeight { weight: ptr })
}

/// Dequantize FP8 E4M3 → BF16, writing into a pre-allocated destination buffer.
///
/// Avoids gpu.alloc/free for the intermediate BF16 data. The caller provides a
/// reusable scratch buffer that is overwritten each call. Safe to reuse after
/// `quantize_to_nvfp4` returns (it synchronizes internally).
pub(crate) fn dequant_fp8_to_bf16_into(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    dest: DevicePtr,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype
    );
    let n_bytes = w.num_elements();
    let mut fp8_buf = vec![0u8; n_bytes];
    gpu.copy_d2h(w.ptr, &mut fp8_buf)?;
    let scale = scalar_f32(store, &format!("{prefix}.weight_scale"), gpu)?;
    let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
    gpu.copy_h2d(&bf16_buf, dest)?;
    Ok(DenseWeight { weight: dest })
}

#[cfg(test)]
mod modelopt_input_scale_tests {
    use super::validate_modelopt_input_scale;
    use spark_runtime::weights::WeightDtype;

    #[test]
    fn requires_one_fp32_modelopt_input_scale() {
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[]).is_ok()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::BF16, &[])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[0])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[1])
                .is_err()
        );
        assert!(
            validate_modelopt_input_scale("projection.input_scale", WeightDtype::FP32, &[1, 1])
                .is_err()
        );
    }
}

#[cfg(test)]
mod rne_norm_and_weight_quant_tests {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{Result, bail};
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::dense_f32_safe;

    struct SimulatedCudaBackend {
        inner: MockGpuBackend,
        has_rne_kernel: bool,
        fail_launch: bool,
        d2h_calls: AtomicUsize,
        h2d_calls: AtomicUsize,
        sync_calls: AtomicUsize,
        launched: std::sync::Mutex<Vec<(u64, [u32; 3], [u32; 3], u64, Vec<u64>)>>,
    }

    impl SimulatedCudaBackend {
        fn new(has_rne_kernel: bool, fail_launch: bool) -> Self {
            Self {
                inner: MockGpuBackend::new(),
                has_rne_kernel,
                fail_launch,
                d2h_calls: AtomicUsize::new(0),
                h2d_calls: AtomicUsize::new(0),
                sync_calls: AtomicUsize::new(0),
                launched: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl GpuBackend for SimulatedCudaBackend {
        fn sm_count(&self) -> Option<u32> {
            Some(48)
        }

        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }

        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }

        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }

        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.h2d_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.copy_h2d(src, dst)
        }

        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.d2h_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.copy_d2h(src, dst)
        }

        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }

        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            _shared_mem: u32,
            stream: u64,
            _params: &mut [*mut c_void],
        ) -> Result<()> {
            if self.fail_launch {
                bail!("simulated launch failure");
            }
            self.launched.lock().expect("launch log lock").push((
                func.0,
                grid,
                block,
                stream,
                Vec::new(),
            ));
            Ok(())
        }

        fn launch_typed(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            _shared_mem: u32,
            stream: u64,
            args: &[KernelArg<'_>],
        ) -> Result<()> {
            if self.fail_launch {
                bail!("simulated launch failure");
            }
            let unpacked = args
                .iter()
                .map(|arg| match arg {
                    KernelArg::Buffer(p) => p.0,
                    KernelArg::Bytes(b) => {
                        let mut slot = [0u8; 8];
                        let len = b.len().min(8);
                        slot[..len].copy_from_slice(&b[..len]);
                        u64::from_le_bytes(slot)
                    }
                })
                .collect();
            self.launched
                .lock()
                .expect("launch log lock")
                .push((func.0, grid, block, stream, unpacked));
            Ok(())
        }

        fn synchronize(&self, _stream: u64) -> Result<()> {
            self.sync_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn default_stream(&self) -> u64 {
            0x42
        }

        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            match (module, func_name) {
                ("quantize_nvfp4", "f32_to_bf16_rne") if self.has_rne_kernel => {
                    Ok(KernelHandle(0xBEEF_0001))
                }
                _ => bail!("missing {module}::{func_name}"),
            }
        }

        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }

        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }

        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }

        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    #[test]
    fn dense_f32_safe_mock_backend_preserves_exact_rne_semantics_without_kernel_launch() {
        let gpu = MockGpuBackend::new();
        // Witness values: tie-to-even down (0x3f80_8000 -> 0x3f80),
        // tie-to-even up (0x3f81_8000 -> 0x3f82), above-midpoint (0x3f80_8001 -> 0x3f81),
        // negative zero (0x8000_0000 -> 0x8000), FP32 subnormal (0x007f_0000 -> 0x007f).
        let f32_bits: [u32; 5] = [
            0x3f80_8000,
            0x3f81_8000,
            0x3f80_8001,
            0x8000_0000,
            0x007f_0000,
        ];
        let src_bytes: Vec<u8> = f32_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let src_ptr = gpu.alloc(src_bytes.len()).expect("alloc src");
        gpu.copy_h2d(&src_bytes, src_ptr).expect("upload src");

        let store = WeightStore::from_map(HashMap::from([(
            "norm.weight".to_string(),
            WeightTensor {
                ptr: src_ptr,
                shape: vec![f32_bits.len()],
                dtype: WeightDtype::FP32,
            },
        )]));

        let loaded = dense_f32_safe(&store, "norm.weight", &gpu).expect("load f32 norm");
        assert_eq!(gpu.launch_count(), 0, "mock backend must use host RNE path");
        let raw = gpu.read_alloc(loaded.weight).expect("allocated bf16 norm");
        let got: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, vec![0x3f80, 0x3f82, 0x3f81, 0x8000, 0x007f]);
    }

    #[test]
    fn dense_f32_safe_cuda_backend_dispatches_async_f32_to_bf16_rne_without_host_stalls() {
        let gpu = SimulatedCudaBackend::new(true, false);
        let n = 513usize; // exercises multi-block div_ceil(513, 256) = 3
        let src_ptr = gpu.inner.alloc(n * 4).expect("alloc f32 source");
        let store = WeightStore::from_map(HashMap::from([(
            "hc_norm.weight".to_string(),
            WeightTensor {
                ptr: src_ptr,
                shape: vec![n],
                dtype: WeightDtype::FP32,
            },
        )]));

        let loaded = dense_f32_safe(&store, "hc_norm.weight", &gpu).expect("async rne load");
        assert_eq!(gpu.d2h_calls.load(Ordering::SeqCst), 0);
        assert_eq!(gpu.h2d_calls.load(Ordering::SeqCst), 0);
        assert_eq!(gpu.sync_calls.load(Ordering::SeqCst), 0);

        let launches = gpu.launched.lock().expect("launches");
        assert_eq!(launches.len(), 1);
        let (handle, grid, block, stream, ref args) = launches[0];
        assert_eq!(handle, 0xBEEF_0001);
        assert_eq!(grid, [3, 1, 1]);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(stream, 0x42);
        assert_eq!(args.as_slice(), &[src_ptr.0, loaded.weight.0, n as u64]);
        assert_eq!(
            gpu.inner
                .read_alloc(loaded.weight)
                .expect("dest alloc")
                .len(),
            n * 2
        );
    }

    #[test]
    fn dense_f32_safe_cuda_backend_falls_back_on_missing_kernel_and_fails_closed_on_launch_err() {
        // 1. Missing f32_to_bf16_rne symbol falls back cleanly to host RNE.
        let fallback_gpu = SimulatedCudaBackend::new(false, false);
        let src = fallback_gpu.inner.alloc(8).expect("alloc");
        let payload: Vec<u8> = [0x3f80_8000u32, 0x3f81_8000u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        fallback_gpu.inner.copy_h2d(&payload, src).expect("seed");
        let store = WeightStore::from_map(HashMap::from([(
            "ple.norm_key.weight".to_string(),
            WeightTensor {
                ptr: src,
                shape: vec![2],
                dtype: WeightDtype::FP32,
            },
        )]));
        let out = dense_f32_safe(&store, "ple.norm_key.weight", &fallback_gpu)
            .expect("host fallback succeeds");
        assert_eq!(fallback_gpu.d2h_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback_gpu.h2d_calls.load(Ordering::SeqCst), 1);
        let raw = fallback_gpu.inner.read_alloc(out.weight).expect("bf16 out");
        assert_eq!(raw, [0x80, 0x3f, 0x82, 0x3f]);

        // 2. Resolved kernel whose launch fails frees its destination buffer and propagates Err.
        let failing_gpu = SimulatedCudaBackend::new(true, true);
        let src_fail = failing_gpu.inner.alloc(8).expect("alloc");
        let store_fail = WeightStore::from_map(HashMap::from([(
            "ple.norm_key.weight".to_string(),
            WeightTensor {
                ptr: src_fail,
                shape: vec![2],
                dtype: WeightDtype::FP32,
            },
        )]));
        assert_eq!(failing_gpu.inner.alloc_count(), 1);
        assert!(dense_f32_safe(&store_fail, "ple.norm_key.weight", &failing_gpu).is_err());
        assert_eq!(
            failing_gpu.inner.alloc_count(),
            1,
            "failed launch must release its fresh BF16 allocation"
        );
    }

    #[test]
    fn dense_f32_safe_handles_bf16_alias_empty_f32_and_rejects_bad_dtypes() {
        let gpu = MockGpuBackend::new();
        let bf16_ptr = gpu.alloc(16).expect("bf16 alloc");
        let empty_ptr = gpu.alloc(0).expect("empty alloc");
        let bad_ptr = gpu.alloc(16).expect("bad alloc");
        let store = WeightStore::from_map(HashMap::from([
            (
                "bf16.weight".to_string(),
                WeightTensor {
                    ptr: bf16_ptr,
                    shape: vec![8],
                    dtype: WeightDtype::BF16,
                },
            ),
            (
                "empty_f32.weight".to_string(),
                WeightTensor {
                    ptr: empty_ptr,
                    shape: vec![0],
                    dtype: WeightDtype::FP32,
                },
            ),
            (
                "fp8.weight".to_string(),
                WeightTensor {
                    ptr: bad_ptr,
                    shape: vec![16],
                    dtype: WeightDtype::FP8E4M3,
                },
            ),
        ]));
        assert_eq!(
            dense_f32_safe(&store, "bf16.weight", &gpu).unwrap().weight,
            bf16_ptr
        );
        let empty_loaded = dense_f32_safe(&store, "empty_f32.weight", &gpu).unwrap();
        assert_eq!(gpu.read_alloc(empty_loaded.weight).unwrap().len(), 0);
        assert!(dense_f32_safe(&store, "fp8.weight", &gpu).is_err());
    }

    #[test]
    fn rne_nvfp4_weight_quantizer_and_loader_call_sites_are_source_bound() {
        let quant_cu = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
        assert!(quant_cu.contains("void quantize_bf16_to_nvfp4("));
        assert!(quant_cu.contains("void quantize_bf16_to_nvfp4_weight_rne("));
        assert!(quant_cu.contains("__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest)"));
        assert!(quant_cu.contains("void f32_to_bf16_rne("));
        assert!(quant_cu.contains("__float2bfloat16(__uint_as_float(in[i]))"));

        for (label, src, expected) in [
            (
                "mtp_head",
                include_str!("../layers/mtp_head/new.rs"),
                2usize,
            ),
            ("qwen35", include_str!("../weight_loader/qwen35.rs"), 1),
            (
                "qwen35_layers",
                include_str!("../weight_loader/qwen35/load_layers.rs"),
                1,
            ),
            (
                "qwen35_dense",
                include_str!("../weight_loader/qwen35_dense.rs"),
                1,
            ),
            ("qwen3", include_str!("../weight_loader/qwen3.rs"), 1),
            ("qwen3_vl", include_str!("../weight_loader/qwen3_vl.rs"), 1),
            (
                "gemma4",
                include_str!("../weight_loader/gemma4/loader_a.rs"),
                1,
            ),
            ("minimax", include_str!("../weight_loader/minimax.rs"), 1),
            ("nemotron", include_str!("../weight_loader/nemotron.rs"), 1),
        ] {
            assert_eq!(
                src.matches("quantize_bf16_to_nvfp4_weight_rne").count(),
                expected,
                "{label} must prefer quantize_bf16_to_nvfp4_weight_rne"
            );
        }
    }
}
