// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

fn transpose_u8_rows_padded(
    gpu: &dyn GpuBackend,
    source: DevicePtr,
    rows: usize,
    columns: usize,
    padded_rows: usize,
) -> Result<Vec<u8>> {
    let source_len = rows
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("quantized transpose source size overflow"))?;
    let mut source_bytes = vec![0u8; source_len];
    gpu.copy_d2h(source, &mut source_bytes)?;

    let destination_len = columns
        .checked_mul(padded_rows)
        .ok_or_else(|| anyhow::anyhow!("quantized transpose destination size overflow"))?;
    let mut destination = vec![0u8; destination_len];
    for row in 0..rows {
        for column in 0..columns {
            destination[column * padded_rows + row] = source_bytes[row * columns + column];
        }
    }
    Ok(destination)
}

/// NVFP4 quantized weight: packed E2M1 data + FP8 block scales + FP32 per-tensor scale.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedWeight {
    /// Packed E2M1 weights (2 values per byte).
    pub weight: DevicePtr,
    /// Per-group FP8 block scales.
    pub weight_scale: DevicePtr,
    /// Per-tensor FP32 scale factor (extracted from GPU via D2H copy at load time).
    pub weight_scale_2: f32,
    /// Input activation scale (FP32 on device, for FP8 activation path).
    pub input_scale: DevicePtr,
}

impl QuantizedWeight {
    /// Null weight (all pointers NULL). Used for remote experts under EP.
    pub fn null() -> Self {
        Self {
            weight: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        }
    }

    /// Whether this weight points to NULL (remote expert placeholder).
    pub fn is_null(&self) -> bool {
        self.weight == DevicePtr::NULL
    }

    /// [`Self::transpose_for_gemm`] with the post-transform buffers served
    /// from (or recorded into) the on-disk weight cache.
    ///
    /// The transpose is a D2H, a scalar byte-wise host loop, and an H2D —
    /// the single biggest per-layer cost of a cold start. `slot` must be a
    /// stable, unique name for this tensor across restarts (e.g.
    /// `"L12.self_attn.q_proj.t"`).
    ///
    /// With `ATLAS_WEIGHT_CACHE` unset this is exactly `transpose_for_gemm`.
    /// `weight_scale_2` and `input_scale` are carried over from `self` rather
    /// than cached — the transpose copies them unchanged.
    pub fn transpose_for_gemm_cached(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
        slot: &str,
    ) -> Result<QuantizedWeight> {
        use crate::weight_loader::transform_cache;

        let Some(cache) = transform_cache::get() else {
            return self.transpose_for_gemm(gpu, n, k);
        };
        let (w_len, s_len) = transform_cache::transposed_lens(n, k);

        if let Some(ptrs) = cache.load_parts(slot, gpu, &[w_len, s_len]) {
            let hit = QuantizedWeight {
                weight: ptrs[0],
                weight_scale: ptrs[1],
                weight_scale_2: self.weight_scale_2,
                input_scale: self.input_scale,
            };
            if !cache.verify_enabled() {
                return Ok(hit);
            }
            let fresh = self.transpose_for_gemm(gpu, n, k)?;
            let ok = cache.verify_pair(
                slot,
                gpu,
                &[(hit.weight, w_len), (hit.weight_scale, s_len)],
                &[(fresh.weight, w_len), (fresh.weight_scale, s_len)],
            );
            if ok {
                gpu.free(fresh.weight)?;
                gpu.free(fresh.weight_scale)?;
                return Ok(hit);
            }
            // Serve the recomputed buffers — never the ones that failed.
            gpu.free(hit.weight)?;
            gpu.free(hit.weight_scale)?;
            return Ok(fresh);
        }

        let fresh = self.transpose_for_gemm(gpu, n, k)?;
        cache.store_parts(
            slot,
            gpu,
            &[(fresh.weight, w_len), (fresh.weight_scale, s_len)],
        );
        Ok(fresh)
    }

    /// Transpose weight layout from `[N, K/2]` to `[K/2, N_pad]` for coalesced GEMM reads,
    /// where `N_pad = ceil(N / 64) * 64` (zero-padded tail columns when `N % 64 != 0`).
    ///
    /// Also transposes scale from `[N, K/GROUP_SIZE]` to `[K/GROUP_SIZE, N_pad]`.
    /// Returns a NEW `QuantizedWeight` with freshly allocated GPU buffers,
    /// leaving the original untouched (needed for decode kernels).
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
    ) -> Result<QuantizedWeight> {
        const GROUP_SIZE: usize = 16;
        let half_k = k / 2;
        // Row stride padded to 64 so odd-N weights (248077-vocab lm_head)
        // keep cp.async 16B alignment in w4a16_gemm_t_m32_n64. Tail columns
        // are zero; the kernel's C stores are guarded by the logical N.
        // For N already divisible by 64 (all FFN/attention weights) this is
        // identical to the previous tight layout.
        let n_pad = n.div_ceil(64) * 64;

        // Transpose B_packed: [N, K/2] → [K/2, N_pad] into a NEW GPU allocation.
        let t_buf = transpose_u8_rows_padded(gpu, self.weight, n, half_k, n_pad)?;
        let new_weight = gpu.alloc(t_buf.len())?;
        gpu.copy_h2d(&t_buf, new_weight)?;

        // Transpose B_scale: [N, K/GROUP_SIZE] → [K/GROUP_SIZE, N_pad].
        let num_groups = k / GROUP_SIZE;
        let st_buf = transpose_u8_rows_padded(gpu, self.weight_scale, n, num_groups, n_pad)?;
        let new_scale = gpu.alloc(st_buf.len())?;
        gpu.copy_h2d(&st_buf, new_scale)?;

        Ok(QuantizedWeight {
            weight: new_weight,
            weight_scale: new_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
        })
    }

    /// [`Self::transpose_for_gemm`] into caller-owned device ranges.
    ///
    /// The destinations must be large enough for `[K/2, N_pad]` packed data
    /// and `[K/16, N_pad]` block scales, respectively. This is used only by
    /// the canonical Qwen4 unified-layout pass, which sub-allocates both
    /// destinations from model-lifetime routed-expert phase slabs.
    pub(crate) fn transpose_for_gemm_into(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
        weight_destination: DevicePtr,
        scale_destination: DevicePtr,
    ) -> Result<QuantizedWeight> {
        const GROUP_SIZE: usize = 16;
        ensure!(
            !weight_destination.is_null() && !scale_destination.is_null(),
            "quantized transpose destinations must be non-null"
        );
        let n_pad = n.div_ceil(64) * 64;
        let packed = transpose_u8_rows_padded(gpu, self.weight, n, k / 2, n_pad)?;
        gpu.copy_h2d(&packed, weight_destination)?;
        let scales = transpose_u8_rows_padded(gpu, self.weight_scale, n, k / GROUP_SIZE, n_pad)?;
        gpu.copy_h2d(&scales, scale_destination)?;

        Ok(QuantizedWeight {
            weight: weight_destination,
            weight_scale: scale_destination,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
        })
    }

    /// Pre-dequant NVFP4 → FP8 E4M3 for zero-overhead prefill GEMMs.
    ///
    /// Reads B_packed[N, K/2] + B_scale[N, K/GROUP_SIZE] + scale2 and produces
    /// B_fp8[N, K] on GPU.  The resulting DevicePtr can be used with `fp8_gemm_t`
    /// which eliminates the per-inference dequant phase entirely.
    pub fn predequant_to_fp8(
        &self,
        gpu: &dyn GpuBackend,
        predequant_kernel: spark_runtime::gpu::KernelHandle,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let fp8_buf = gpu.alloc(n * k)?;
        crate::layers::ops::predequant_nvfp4_to_fp8(
            gpu,
            predequant_kernel,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            fp8_buf,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok(fp8_buf)
    }
}

/// BF16 dense weight (no quantization).
#[derive(Debug, Clone, Copy)]
pub struct DenseWeight {
    pub weight: DevicePtr,
}

/// FP8 E4M3 dense weight (runtime-quantized from BF16).
///
/// Halves weight bandwidth vs BF16. Per-row f32 scale preserves accuracy.
/// Created at model load time via GPU-side quantization kernel.
#[derive(Debug, Clone, Copy)]
pub struct Fp8DenseWeight {
    /// FP8 E4M3 weight data: [N, K] bytes.
    pub weight: DevicePtr,
    /// Per-row dequant scale: `[N]` f32.
    pub row_scale: DevicePtr,
}

/// FP8 E4M3 block-scaled checkpoint weight loaded directly from safetensors.
///
/// Unlike [`Fp8DenseWeight`] (runtime-quantized from BF16 with `[N]` f32
/// per-row scales), this struct represents an on-disk FP8 weight with 2D
/// 128×128 BF16 block scales (`[ceil(N/128), ceil(K/128)]`). Used by the
/// `w8a16_gemv` / `w8a16_gemm_t` kernels for native FP8 serving without
/// converting to NVFP4.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight {
    /// `[N, K]` FP8 E4M3 weight bytes on GPU.
    pub weight: DevicePtr,
    /// `[ceil(N/128), ceil(K/128)]` BF16 128×128 block dequant scales on GPU
    /// (field name retained for compatibility across kernel dispatch sites).
    pub row_scale: DevicePtr,
    /// Output dimension (rows).
    pub n: u32,
    /// Input dimension (columns).
    pub k: u32,
}

/// FP8 E4M3 weight with transposed layout for coalesced prefill GEMM.
///
/// B_t: [K, N] — transposed from checkpoint's B[N, K].
/// block_scale_t: [K/128, N/128] — transposed from [N/128, K/128].
/// Enables ~14x faster prefill via w8a16_gemm_t kernel.
#[derive(Debug, Clone, Copy)]
pub struct Fp8WeightTransposed {
    /// [K, N] FP8 E4M3 transposed weight on GPU.
    pub weight_t: DevicePtr,
    /// [K/128, N/128] BF16 transposed block scales on GPU.
    pub scale_t: DevicePtr,
    pub n: u32,
    pub k: u32,
}

impl Fp8Weight {
    /// Transpose this FP8 weight for coalesced prefill GEMM.
    /// Allocates new GPU buffers for `B_t[K,N]` and `scale_t[K/128, N/128]`.
    pub fn transpose_for_gemm(
        &self,
        gpu: &dyn GpuBackend,
        transpose_k: spark_runtime::gpu::KernelHandle,
        transpose_scale_k: spark_runtime::gpu::KernelHandle,
        stream: u64,
    ) -> anyhow::Result<Fp8WeightTransposed> {
        let n = self.n as usize;
        let k = self.k as usize;

        // Allocate transposed weight: [K, N] bytes
        let weight_t = gpu.alloc(k * n)?;
        crate::layers::ops::transpose_fp8(
            gpu,
            transpose_k,
            self.weight,
            weight_t,
            self.n,
            self.k,
            stream,
        )?;

        // Allocate transposed scale: [K/128, N/128] × 2 bytes (BF16)
        let n_blocks = n.div_ceil(128);
        let k_blocks = k.div_ceil(128);
        let scale_t = gpu.alloc(k_blocks * n_blocks * 2)?;
        crate::layers::ops::transpose_block_scale(
            gpu,
            transpose_scale_k,
            self.row_scale,
            scale_t,
            n_blocks as u32,
            k_blocks as u32,
            stream,
        )?;

        gpu.synchronize(stream)?;

        Ok(Fp8WeightTransposed {
            weight_t,
            scale_t,
            n: self.n,
            k: self.k,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};

    #[test]
    fn transpose_for_gemm_into_uses_only_the_caller_allocation() -> Result<()> {
        let gpu = MockGpuBackend::new();
        let packed_source = gpu.alloc(3 * 8)?;
        let scale_source = gpu.alloc(3)?;
        let packed_bytes: Vec<u8> = (0..24).collect();
        let scale_bytes = [41, 42, 43];
        gpu.copy_h2d(&packed_bytes, packed_source)?;
        gpu.copy_h2d(&scale_bytes, scale_source)?;

        let source = QuantizedWeight {
            weight: packed_source,
            weight_scale: scale_source,
            weight_scale_2: f32::from_bits(0x3f81_2345),
            input_scale: DevicePtr(0xfeed_beef),
        };
        let baseline = source.transpose_for_gemm(&gpu, 3, 16)?;
        let baseline_packed = gpu
            .read_alloc(baseline.weight)
            .expect("baseline packed allocation");
        let baseline_scale = gpu
            .read_alloc(baseline.weight_scale)
            .expect("baseline scale allocation");

        let allocations_before_slab = gpu.alloc_count();
        let slab = gpu.alloc(256 + 512 + 64)?;
        let weight_destination = slab.offset(256);
        let scale_destination = slab.offset(256 + 512);
        let transposed =
            source.transpose_for_gemm_into(&gpu, 3, 16, weight_destination, scale_destination)?;
        assert_eq!(gpu.alloc_count(), allocations_before_slab + 1);
        assert_eq!(transposed.weight, weight_destination);
        assert_eq!(transposed.weight_scale, scale_destination);
        assert_eq!(
            transposed.weight_scale_2.to_bits(),
            source.weight_scale_2.to_bits()
        );
        assert_eq!(transposed.input_scale, source.input_scale);

        let slab_bytes = gpu.read_alloc(slab).expect("slab allocation");
        let packed_transpose = &slab_bytes[256..256 + 512];
        assert_eq!(packed_transpose, baseline_packed);
        assert_eq!(&slab_bytes[256 + 512..], baseline_scale);
        for column in 0..8 {
            assert_eq!(
                &packed_transpose[column * 64..column * 64 + 3],
                &[
                    packed_bytes[column],
                    packed_bytes[8 + column],
                    packed_bytes[16 + column],
                ]
            );
            assert!(
                packed_transpose[column * 64 + 3..(column + 1) * 64]
                    .iter()
                    .all(|byte| *byte == 0)
            );
        }
        assert_eq!(&slab_bytes[256 + 512..256 + 512 + 3], &scale_bytes);
        assert!(slab_bytes[256 + 512 + 3..].iter().all(|byte| *byte == 0));
        Ok(())
    }
}
