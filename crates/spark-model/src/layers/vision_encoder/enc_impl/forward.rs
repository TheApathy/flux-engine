// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level `VisionEncoder::forward`: drives the full image → token
//! pipeline (pos_embed → RoPE → patch embed → 27 ViT blocks with
//! deepstack merger taps → final merger).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Full forward pass: pixels → `[total_rows, out_hidden_size]` BF16 in `buf_out`.
    ///
    /// Returns `total_rows = (1 + self.deepstack_indexes.len()) * merged_p`,
    /// where `merged_p = (grid_h * grid_w) / (spatial_merge_size * spatial_merge_size)`.
    /// The first `merged_p` rows hold the final merger output, followed by each
    /// deepstack merger's `merged_p` rows.
    pub fn forward(
        &self,
        pixels: &[f32], // [P, C*T*Hp*Wp = 1536]
        grid_h: usize,
        grid_w: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<usize> {
        let ms = self.spatial_merge_size;
        anyhow::ensure!(
            ms > 0 && grid_h.is_multiple_of(ms) && grid_w.is_multiple_of(ms),
            "vision grid ({grid_h}x{grid_w}) must be divisible by spatial_merge_size={ms}"
        );
        let p = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| anyhow::anyhow!("vision patch count overflow ({grid_h} * {grid_w})"))?;
        anyhow::ensure!(
            p <= self.p_max,
            "vision patch count {p} ({grid_h}x{grid_w}) exceeds p_max={}",
            self.p_max
        );
        let required_pixels = p
            .checked_mul(1536)
            .ok_or_else(|| anyhow::anyhow!("vision pixel count overflow (p={p})"))?;
        anyhow::ensure!(
            pixels.len() >= required_pixels,
            "vision pixels len {} < required {required_pixels} (p={p} * 1536)",
            pixels.len()
        );
        let merged_p = p / (ms * ms);
        let total_rows = (1 + self.deepstack_indexes.len())
            .checked_mul(merged_p)
            .ok_or_else(|| anyhow::anyhow!("vision output row count overflow"))?;
        anyhow::ensure!(
            total_rows <= self.p_max,
            "vision total_rows={total_rows} exceeds buf_out capacity p_max={}",
            self.p_max
        );

        // Per-image prep: interpolate learned pos_embed grid + build 2D
        // rotary cos/sin for attention. A/B toggles:
        //   ATLAS_VISION_POSINTERP=0  → copy raw pos_embed[0..p*h] instead
        //   ATLAS_VISION_ROPE=0       → upload cos=1, sin=0 (identity)
        let pos_interp_on = std::env::var("ATLAS_VISION_POSINTERP")
            .map(|v| v != "0")
            .unwrap_or(true);
        if pos_interp_on {
            self.resample_pos_embed(grid_h, grid_w, gpu, stream)?;
        } else {
            // Legacy path: copy first p*hidden_size entries of raw pos_embed.
            let raw_pos_rows = self.num_grid_per_side * self.num_grid_per_side;
            anyhow::ensure!(
                p <= raw_pos_rows,
                "ATLAS_VISION_POSINTERP=0 requires p={p} <= num_position_embeddings={raw_pos_rows}"
            );
            self.gpu_copy_bf16(
                gpu,
                self.pos_embed,
                self.buf_pos_resampled,
                p * self.hidden_size * 2,
                stream,
            )?;
        }
        self.build_rope_cossin(grid_h, grid_w, gpu, stream)?;

        self.patch_embed(pixels, p, gpu, stream)?;
        Self::maybe_dump_buf(
            gpu,
            self.buf_h1,
            p * self.hidden_size,
            "patch_embed",
            stream,
        )?;

        // Output layout in buf_out:
        //   rows [0 .. merged_p)                                = final merger (after last block)
        //   rows [(ds_idx + 1) * merged_p .. (ds_idx + 2) * merged_p) = deepstack merger `ds_idx`
        // The splicer reads the FIRST `merged_p` rows per image (replacing
        // `<|image_pad|>` in the LLM stream). Deepstack features are written
        // immediately after and are not consumed by the current LLM splicer.
        let n_h_bytes = p * self.hidden_size * 2;
        let mut deepstack_iter = self.deepstack_indexes.iter().enumerate();
        let mut next_ds = deepstack_iter.next(); // (merger_idx, &block_1indexed)

        for (block_idx, blk) in self.blocks.iter().enumerate() {
            self.vit_block(blk, p, gpu, stream)?;
            Self::maybe_dump_buf(
                gpu,
                self.buf_h1,
                p * self.hidden_size,
                &format!("block{block_idx:02}"),
                stream,
            )?;

            let block_1indexed = block_idx + 1;
            // Deepstack extraction point: snapshot buf_h1 → buf_h2 first,
            // then apply the merger out-of-place so the residual stream
            // into the next ViT block stays unperturbed.
            if let Some((ds_idx, &ds_block)) = next_ds
                && block_1indexed == ds_block
            {
                self.gpu_copy_bf16(gpu, self.buf_h1, self.buf_h2, n_h_bytes, stream)?;
                let offset_rows = (ds_idx + 1) * merged_p;
                let out_slice = self.buf_out.offset(offset_rows * self.out_hidden_size * 2);
                self.apply_merger(
                    &self.deepstack[ds_idx],
                    p,
                    grid_h,
                    grid_w,
                    self.buf_h2,
                    out_slice,
                    gpu,
                    stream,
                )?;
                next_ds = deepstack_iter.next();
            }
        }

        // Final merger runs on buf_h1 in-place (no subsequent block to protect).
        let out_slice = self.buf_out.offset(0);
        self.apply_merger(
            &self.merger,
            p,
            grid_h,
            grid_w,
            self.buf_h1,
            out_slice,
            gpu,
            stream,
        )?;
        Self::maybe_dump_buf(
            gpu,
            self.buf_out,
            total_rows * self.out_hidden_size,
            "final",
            stream,
        )?;

        Ok(total_rows)
    }
}
