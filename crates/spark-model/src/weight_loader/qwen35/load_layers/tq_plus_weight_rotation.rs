// SPDX-License-Identifier: AGPL-3.0-only

//! TurboQuant+ weight pre-rotation.
//!
//! Attribution: Canonical two-sided Rademacher WHT sign-mask rotation scheme
//! adapted from TurboQuant (`TheTom/llama-cpp-turboquant`, MIT) and
//! TurboQuant+ (`TheTom/turboquant_plus`, Apache-2.0); see root
//! `THIRD_PARTY_NOTICES.md` and `LICENSES/`.
//!
//! Folds the canonical forward rotation `R = S2·H·S1 / sqrt(d)` into Q/K/V
//! projection weights at load time so the per-token `wht_bf16_inplace`
//! dispatch on Q, K, and V can be skipped.
//!
//! Per-token runtime savings (Qwen3.6-35B-A3B has 40 total layers with
//! `full_attention_interval = 4`, i.e., 10 FullAttention layers and 30
//! LinearAttention/GDN layers):
//!   - Skip 1 `wht_bf16_inplace` on Q
//!   - Skip 1 `wht_bf16_inplace` on K
//!   - Skip 1 `wht_bf16_inplace` on V
//!
//! = at most 3 × 10 = 30 kernel launches saved per token. O-projection
//! inverse rotation is not yet implemented or wired.
//!
//! Weight layouts in `WeightStore` and `shard_dense_bf16` are row-major
//! `[out_dim, in_dim]`:
//!   - Ungated Q/K/V projections: `[n_heads * head_dim, hidden]`.
//!   - Gated Q projection (`config.attn_gated`): `[2 * n_heads * head_dim, hidden]`,
//!     interleaved per head as `[Q_h0(head_dim), Gate_h0(head_dim), ...]`;
//!     only the Q rows are rotated while the gate rows remain untouched.
//!   - O projection: `[hidden, n_heads * head_dim]`. Folding the inverse
//!     rotation `R^T = S1·H·S2 / sqrt(d)` onto its input head axis requires
//!     `wht_bf16_inplace_inv` (because `S1 != S2`, so `R` is not self-inverse)
//!     and is not yet implemented.
//!
//! Because `wht_bf16_inplace` rotates contiguous `head_dim`-element chunks
//! (`data + blockIdx.x * head_dim`), [`rotate_row_major_qkv_bf16`] transposes
//! the selected Q/K/V head rows from row-major `[out_dim, in_dim]` into
//! contiguous `[in_dim, n_heads, head_dim]` chunks before launching the kernel
//! and scatters the rotated weights back into a freshly allocated row-major
//! buffer (never mutating a `WeightStore` tensor in place).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;

/// Row layout of a row-major `[out_dim, in_dim]` Q/K/V projection weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjHeadLayout {
    /// Contiguous heads `[h0(head_dim), h1(head_dim), ...]`.
    Standard,
    /// Per-head interleaved Q and sigmoid gate
    /// `[Q_h0(head_dim), Gate_h0(head_dim), ...]`; only Q rows are rotated.
    GatedInterleaved,
}

impl ProjHeadLayout {
    fn head_row_stride(self, head_dim: usize) -> Result<usize> {
        ensure!(head_dim > 0, "TQ+ weight rotation requires head_dim > 0");
        match self {
            Self::Standard => Ok(head_dim),
            Self::GatedInterleaved => head_dim
                .checked_mul(2)
                .context("gated head row stride overflow"),
        }
    }

    pub(super) fn num_rotated_heads(self, out_dim: usize, head_dim: usize) -> Result<usize> {
        let stride = self.head_row_stride(head_dim)?;
        ensure!(
            out_dim > 0 && out_dim.is_multiple_of(stride),
            "TQ+ weight rotation: out_dim {out_dim} is not a positive multiple of head row stride {stride} (layout={self:?}, head_dim={head_dim})"
        );
        Ok(out_dim / stride)
    }
}

/// Geometry of a row-major `[out_dim, in_dim]` BF16 Q/K/V weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RowMajorProjGeometry {
    pub(super) out_dim: usize,
    pub(super) in_dim: usize,
    pub(super) head_dim: usize,
    pub(super) layout: ProjHeadLayout,
}

impl RowMajorProjGeometry {
    fn total_bf16_bytes(self) -> Result<usize> {
        self.out_dim
            .checked_mul(self.in_dim)
            .and_then(|elems| elems.checked_mul(2))
            .context("row-major BF16 weight byte size overflow")
    }

    fn rotated_bf16_bytes(self, n_heads: usize) -> Result<usize> {
        self.in_dim
            .checked_mul(n_heads)
            .and_then(|v| v.checked_mul(self.head_dim))
            .and_then(|elems| elems.checked_mul(2))
            .context("transposed BF16 head-chunk byte size overflow")
    }
}

/// Extract the Q/K/V head rows from row-major `[out_dim, in_dim]` BF16 bytes
/// into contiguous `[in_dim, n_heads, head_dim]` BF16 bytes so each
/// `head_dim` slice lies along the output head axis.
fn extract_heads_transposed_bf16(src_bytes: &[u8], geom: RowMajorProjGeometry) -> Result<Vec<u8>> {
    ensure!(geom.in_dim > 0, "TQ+ weight rotation requires in_dim > 0");
    let expected_src = geom.total_bf16_bytes()?;
    ensure!(
        src_bytes.len() == expected_src,
        "TQ+ extract: expected {expected_src} bytes for [{}, {}], got {}",
        geom.out_dim,
        geom.in_dim,
        src_bytes.len()
    );
    let n_heads = geom.layout.num_rotated_heads(geom.out_dim, geom.head_dim)?;
    let head_stride = geom.layout.head_row_stride(geom.head_dim)?;
    let mut transposed = vec![0u8; geom.rotated_bf16_bytes(n_heads)?];

    for head in 0..n_heads {
        let row_base = head * head_stride;
        for d in 0..geom.head_dim {
            let src_row_offset = (row_base + d) * geom.in_dim;
            for k in 0..geom.in_dim {
                let src_idx = (src_row_offset + k) * 2;
                let dst_idx = ((k * n_heads + head) * geom.head_dim + d) * 2;
                transposed[dst_idx] = src_bytes[src_idx];
                transposed[dst_idx + 1] = src_bytes[src_idx + 1];
            }
        }
    }
    Ok(transposed)
}

/// Scatter rotated `[in_dim, n_heads, head_dim]` BF16 bytes back into the
/// Q/K/V head rows of a row-major `[out_dim, in_dim]` BF16 buffer, leaving any
/// interleaved gate rows untouched.
fn scatter_heads_transposed_bf16(
    dst_bytes: &mut [u8],
    transposed_bytes: &[u8],
    geom: RowMajorProjGeometry,
) -> Result<()> {
    let expected_dst = geom.total_bf16_bytes()?;
    ensure!(
        dst_bytes.len() == expected_dst,
        "TQ+ scatter: expected {expected_dst} destination bytes, got {}",
        dst_bytes.len()
    );
    let n_heads = geom.layout.num_rotated_heads(geom.out_dim, geom.head_dim)?;
    let expected_transposed = geom.rotated_bf16_bytes(n_heads)?;
    ensure!(
        transposed_bytes.len() == expected_transposed,
        "TQ+ scatter: expected {expected_transposed} transposed bytes, got {}",
        transposed_bytes.len()
    );
    let head_stride = geom.layout.head_row_stride(geom.head_dim)?;

    for head in 0..n_heads {
        let row_base = head * head_stride;
        for d in 0..geom.head_dim {
            let dst_row_offset = (row_base + d) * geom.in_dim;
            for k in 0..geom.in_dim {
                let src_idx = ((k * n_heads + head) * geom.head_dim + d) * 2;
                let dst_idx = (dst_row_offset + k) * 2;
                dst_bytes[dst_idx] = transposed_bytes[src_idx];
                dst_bytes[dst_idx + 1] = transposed_bytes[src_idx + 1];
            }
        }
    }
    Ok(())
}

/// Apply the canonical forward two-sided Rademacher WHT rotation in-place to
/// BF16 storage already laid out as contiguous `[outer, n_heads, head_dim]`
/// chunks (private so callers cannot pass untransposed row-major
/// `[n_heads * head_dim, hidden]` weights directly).
fn apply_canonical_rotation_contiguous(
    gpu: &dyn GpuBackend,
    weight_bf16: DevicePtr,
    outer: usize,
    n_heads: usize,
    head_dim: usize,
    stream: u64,
) -> Result<()> {
    let total_heads = outer
        .checked_mul(n_heads)
        .context("outer * n_heads overflow")?;
    if total_heads == 0 {
        return Ok(());
    }
    // `wht_bf16_inplace` applies two-sided TQ+ Rademacher signs
    // (`tq_plus_signs.cuh`) for all three supported head dimensions
    // (128, 256, and 512).
    if !matches!(head_dim, 128 | 256 | 512) {
        bail!(
            "apply_canonical_rotation_contiguous: unsupported head_dim {head_dim} (need 128, 256, or 512)"
        );
    }

    let grid_x = u32::try_from(total_heads).context("total_heads exceeds u32::MAX")?;
    let head_dim_u32 = u32::try_from(head_dim).context("head_dim exceeds u32::MAX")?;

    let wht_kernel = try_kernel(gpu, "wht_bf16", "wht_bf16_inplace");
    if wht_kernel.0 == 0 {
        bail!("wht_bf16_inplace kernel handle not available");
    }

    KernelLaunch::new(gpu, wht_kernel)
        .grid([grid_x, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(weight_bf16)
        .arg_u32(head_dim_u32)
        .launch(stream)?;

    Ok(())
}

/// Return a freshly allocated row-major `[out_dim, in_dim]` BF16 buffer with
/// canonical TQ+ forward rotation `S2·H·S1 / sqrt(d)` folded along each head's
/// output axis.
///
/// The input `src_bf16` is never modified in place, so `WeightStore` aliases
/// remain immutable. Caller owns the returned [`DevicePtr`] and must free it
/// via `gpu.free(...)` after downstream quantization.
pub(super) fn rotate_row_major_qkv_bf16(
    gpu: &dyn GpuBackend,
    src_bf16: DevicePtr,
    geom: RowMajorProjGeometry,
    stream: u64,
) -> Result<DevicePtr> {
    if !matches!(geom.head_dim, 128 | 256 | 512) {
        bail!(
            "TQ+ weight pre-rotation: unsupported head_dim {} (need 128, 256, or 512)",
            geom.head_dim
        );
    }
    let n_heads = geom.layout.num_rotated_heads(geom.out_dim, geom.head_dim)?;
    let total_bytes = geom.total_bf16_bytes()?;

    gpu.synchronize(stream)?;
    let mut host_matrix = vec![0u8; total_bytes];
    gpu.copy_d2h(src_bf16, &mut host_matrix)?;

    let mut transposed = extract_heads_transposed_bf16(&host_matrix, geom)?;
    let t_dev = gpu.alloc(transposed.len())?;
    let rotated = gpu
        .copy_h2d(&transposed, t_dev)
        .and_then(|()| {
            apply_canonical_rotation_contiguous(
                gpu,
                t_dev,
                geom.in_dim,
                n_heads,
                geom.head_dim,
                stream,
            )
        })
        .and_then(|()| gpu.synchronize(stream))
        .and_then(|()| gpu.copy_d2h(t_dev, &mut transposed));
    if let Err(err) = rotated {
        let _ = gpu.free(t_dev);
        return Err(err);
    }
    gpu.free(t_dev)?;

    scatter_heads_transposed_bf16(&mut host_matrix, &transposed, geom)?;

    let dst_dev = gpu.alloc(total_bytes)?;
    if let Err(err) = gpu.copy_h2d(&host_matrix, dst_dev) {
        let _ = gpu.free(dst_dev);
        return Err(err);
    }
    Ok(dst_dev)
}

/// Should weight pre-rotation be active for this build? Opt-in via
/// `TQ_PLUS_WEIGHT_ROTATION=1` until correctness is validated.
pub(super) fn weight_rotation_enabled() -> bool {
    std::env::var("TQ_PLUS_WEIGHT_ROTATION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::{
        ProjHeadLayout, RowMajorProjGeometry, apply_canonical_rotation_contiguous,
        extract_heads_transposed_bf16, scatter_heads_transposed_bf16,
    };

    fn bf16_bytes(vals: &[u16]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn from_bf16_bytes(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn standard_extract_and_scatter_align_output_head_axis() {
        // 2 heads × head_dim=2, in_dim=3 => row-major [4, 3].
        let geom = RowMajorProjGeometry {
            out_dim: 4,
            in_dim: 3,
            head_dim: 2,
            layout: ProjHeadLayout::Standard,
        };
        let src_vals: Vec<u16> = (1..=12).collect();
        let src = bf16_bytes(&src_vals);
        let extracted = extract_heads_transposed_bf16(&src, geom).unwrap();
        // Transposed layout [in_dim=3, n_heads=2, head_dim=2]:
        // k=0: h0=[1,4], h1=[7,10]
        // k=1: h0=[2,5], h1=[8,11]
        // k=2: h0=[3,6], h1=[9,12]
        assert_eq!(
            from_bf16_bytes(&extracted),
            vec![1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]
        );

        let mut dst = vec![0u8; src.len()];
        scatter_heads_transposed_bf16(&mut dst, &extracted, geom).unwrap();
        assert_eq!(from_bf16_bytes(&dst), src_vals);
    }

    #[test]
    fn gated_interleaved_extract_and_scatter_preserves_gate_rows() {
        // 2 heads × head_dim=2, gated => out_dim=8, in_dim=2:
        // rows 0..2: Q_h0, rows 2..4: Gate_h0, rows 4..6: Q_h1, rows 6..8: Gate_h1.
        let geom = RowMajorProjGeometry {
            out_dim: 8,
            in_dim: 2,
            head_dim: 2,
            layout: ProjHeadLayout::GatedInterleaved,
        };
        let src_vals: Vec<u16> = vec![
            10, 11, // Q_h0 d=0
            12, 13, // Q_h0 d=1
            90, 91, // Gate_h0 d=0
            92, 93, // Gate_h0 d=1
            20, 21, // Q_h1 d=0
            22, 23, // Q_h1 d=1
            94, 95, // Gate_h1 d=0
            96, 97, // Gate_h1 d=1
        ];
        let mut matrix = bf16_bytes(&src_vals);
        let extracted = extract_heads_transposed_bf16(&matrix, geom).unwrap();
        assert_eq!(
            from_bf16_bytes(&extracted),
            vec![10, 12, 20, 22, 11, 13, 21, 23]
        );

        // Mutate the extracted Q entries and scatter back; gate rows must be untouched.
        let rotated_vals: Vec<u16> = from_bf16_bytes(&extracted)
            .into_iter()
            .map(|v| v + 100)
            .collect();
        scatter_heads_transposed_bf16(&mut matrix, &bf16_bytes(&rotated_vals), geom).unwrap();
        assert_eq!(
            from_bf16_bytes(&matrix),
            vec![
                110, 111, // Q_h0 rotated
                112, 113, // Q_h0 rotated
                90, 91, // Gate_h0 preserved
                92, 93, // Gate_h0 preserved
                120, 121, // Q_h1 rotated
                122, 123, // Q_h1 rotated
                94, 95, // Gate_h1 preserved
                96, 97, // Gate_h1 preserved
            ]
        );
    }

    #[test]
    fn contiguous_rotation_errors_on_overflow_and_unsupported_head_dim() {
        let gpu = MockGpuBackend::new();
        assert!(
            apply_canonical_rotation_contiguous(&gpu, DevicePtr(1), usize::MAX, 2, 128, 0).is_err()
        );
        assert!(
            apply_canonical_rotation_contiguous(
                &gpu,
                DevicePtr(1),
                (u32::MAX as usize) + 1,
                1,
                128,
                0
            )
            .is_err()
        );
        assert!(apply_canonical_rotation_contiguous(&gpu, DevicePtr(1), 4, 2, 64, 0).is_err());
    }
}
