// SPDX-License-Identifier: AGPL-3.0-only

//! Patch-embed step: f32 pixels → BF16 → patch_embed GEMM → +pos_embed.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, HostToDeviceCopy};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Upload f32 pixels → convert to BF16 → patch embed GEMM → add pos_embed.
    pub(super) fn patch_embed(
        &self,
        pixels: &[f32],
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let n_f32 = p
            .checked_mul(1536)
            .ok_or_else(|| anyhow::anyhow!("vision patch_embed element count overflow (p={p})"))?;
        anyhow::ensure!(
            p <= self.p_max,
            "vision patch_embed p={p} exceeds scratch capacity p_max={}",
            self.p_max
        );
        anyhow::ensure!(
            pixels.len() >= n_f32,
            "vision patch_embed pixels len {} < required {n_f32} (p={p} * 1536)",
            pixels.len()
        );
        let n_bytes = n_f32.checked_mul(4).ok_or_else(|| {
            anyhow::anyhow!("vision patch_embed byte size overflow (n_f32={n_f32})")
        })?;
        // SAFETY: `pixels.len() >= n_f32` was just checked above and `n_bytes == n_f32 * 4`
        // did not overflow; `f32` has size 4 with no padding bytes, `u8` has alignment 1,
        // and `f32_bytes` borrows `pixels` only for the synchronous copy call below.
        let f32_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, n_bytes) };
        gpu.copy_h2d_group_on_stream(&[HostToDeviceCopy::new(f32_bytes, self.buf_f32)], stream)?;
        // f32 → bf16 (result in buf_wide[0..p*1536])
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil(n_f32 as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_f32)
            .arg_ptr(self.buf_wide)
            .arg_u32(n_f32 as u32)
            .launch(stream)?;
        // patch_embed GEMM: buf_wide[p,1536] @ patch_embed_w[1152,1536]^T + b → buf_h1[p,1152]
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([
                div_ceil(self.hidden_size as u32, 32),
                div_ceil(p as u32, 32),
                1,
            ])
            .block([32, 32, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.patch_embed_w)
            .arg_ptr(self.patch_embed_b)
            .arg_ptr(self.buf_h1)
            .arg_u32(p as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(1536)
            .launch(stream)?;
        // add the image-specific bilinear-interpolated pos_embed to buf_h1.
        // (Source was prepared by `resample_pos_embed()` in forward().)
        let n_pe = p * self.hidden_size;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_pos_resampled)
            .arg_u32(n_pe as u32)
            .launch(stream)
    }
}
