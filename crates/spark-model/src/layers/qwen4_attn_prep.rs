// SPDX-License-Identifier: AGPL-3.0-only
//! Exact attention-prep kernels for Flash-Next prefill
//! (`ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED=1`, `qwen4_attn_prep_fused.cu`).
//!
//! The Atlas RoPE kernels compute each frequency with an FP64 `pow` per
//! (row, head, pair). These kernels read a per-process table built once on
//! the device by the same expression, so their outputs are bit-identical:
//! - `qwen4_attn_prep_fused`: Q/K norms + interleaved MRoPE + paged BF16 K/V
//!   write in one launch (the full-row attention prep);
//! - `qwen4_rope_table`: `rope_forward` for one tensor (the QSA indexer
//!   queries of the sparse prefill core).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED";
const MODULE: &str = "qwen4_attn_prep_fused";
pub(crate) const HEAD_DIM: usize = 256;

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{SELECTOR} must be absent, 0, or 1; got {other:?}"),
    }
}

/// Whether the fused prep is selected (read once per process).
pub fn selected() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow::anyhow!(e))
}

pub(crate) struct Kernels {
    pub table: KernelHandle,
    pub fused: KernelHandle,
    pub fused_yarn: KernelHandle,
    pub fused_nvfp4: KernelHandle,
    pub rope: KernelHandle,
}

pub(crate) fn kernels(gpu: &dyn GpuBackend) -> Result<&'static Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |f: &str| {
            gpu.kernel(MODULE, f)
                .map_err(|e| format!("{MODULE}::{f}: {e}"))
        };
        Ok(Kernels {
            table: load("qwen4_rope_freq_table")?,
            fused: load("qwen4_attn_prep_fused")?,
            fused_yarn: crate::layers::try_kernel(gpu, MODULE, "qwen4_attn_prep_fused_yarn"),
            fused_nvfp4: crate::layers::try_kernel(gpu, MODULE, "qwen4_attn_prep_fused_nvfp4"),
            rope: load("qwen4_rope_table")?,
        })
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernels unavailable: {e}"))
}

/// RoPE frequencies for (rotary_dim, theta), built once on `stream` per
/// geometry and kept for the process.
pub(crate) fn freq_table(
    gpu: &dyn GpuBackend,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<DevicePtr> {
    static T: Mutex<Vec<(u32, u32, DevicePtr)>> = Mutex::new(Vec::new());
    let mut tables = T
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: table lock poisoned"))?;
    if let Some(&(_, _, ptr)) = tables
        .iter()
        .find(|&&(dim, bits, _)| dim == rotary_dim && bits == theta.to_bits())
    {
        return Ok(ptr);
    }
    ensure!(
        rotary_dim >= 2 && rotary_dim <= HEAD_DIM as u32 && rotary_dim.is_multiple_of(2),
        "{SELECTOR}: rotary dimension {rotary_dim} is invalid"
    );
    ensure!(
        theta.is_finite() && theta > 0.0,
        "{SELECTOR}: theta {theta} is invalid"
    );
    let ptr = gpu
        .alloc(HEAD_DIM / 2 * 4)
        .with_context(|| format!("{SELECTOR}: alloc frequency table"))?;
    KernelLaunch::new(gpu, kernels(gpu)?.table)
        .grid([1, 1, 1])
        .block([rotary_dim / 2, 1, 1])
        .arg_ptr(ptr)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)?;
    tables.push((rotary_dim, theta.to_bits(), ptr));
    Ok(ptr)
}

/// `rope_forward` over `x` = `[seq_len][heads][head_dim]` bf16 in place
/// (NeoX pairs (i, i + rotary_dim/2)), frequencies from `freq`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_with_freq(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    positions: DevicePtr,
    freq: DevicePtr,
    seq_len: usize,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rotary_dim >= 2
            && rotary_dim <= head_dim
            && rotary_dim <= 256
            && rotary_dim.is_multiple_of(2),
        "{SELECTOR}: rotary dimension {rotary_dim} must be an even value in 2..=256 and <= head_dim {head_dim}"
    );
    let pos_per_block = 128 / (rotary_dim / 2);
    KernelLaunch::new(gpu, kernels(gpu)?.rope)
        .grid([heads, u32::try_from(seq_len)?.div_ceil(pos_per_block), 1])
        .block([128, 1, 1])
        .arg_ptr(x)
        .arg_ptr(positions)
        .arg_ptr(freq)
        .arg_u32(u32::try_from(seq_len)?)
        .arg_u32(heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .launch(stream)
}

/// `rope_forward` over `x` = `[seq_len][heads][head_dim]` bf16 in place
/// (NeoX pairs (i, i + rotary_dim/2)), frequencies from the table.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn rope(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    positions: DevicePtr,
    seq_len: usize,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    let freq = freq_table(gpu, rotary_dim, theta, stream)?;
    rope_with_freq(
        gpu, x, positions, freq, seq_len, heads, head_dim, rotary_dim, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_is_strict() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        assert!(parse(Some("yes")).is_err());
    }
}
