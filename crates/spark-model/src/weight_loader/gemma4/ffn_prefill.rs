// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in transposed NVFP4 FFN copies for Gemma-4 dense prefill
//! (`ATLAS_GEMMA4_FFN_TRANSPOSE=1`).
//!
//! Without them every Gemma-4 dense FFN prefill lands on the untransposed
//! `w4a16_gemm` fallback: the M>=128 `w4a16_gemm_t_m128` route is guarded on
//! the `[K/2, N]` copies, and this loader never built them. Upstream measured
//! that fallback on Gemma-4-31B at a 4096-token prompt as 23.7 s of a 28.2 s
//! prefill (~7 TFLOP/s).
//!
//! COST: 3 x hidden x intermediate x (1/2 + 1/16) bytes per layer, ~11.7 GB
//! (10.9 GiB) on the 31B, on top of the originals (decode keeps the HF layout). Decided once
//! before the layer loop: `free_memory()` shrinks as layers load, so a
//! per-layer probe would transpose early layers and skip late ones.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::layers::DenseFfnLayer;

/// Free memory left over after the transposed copies.
pub(super) const FFN_TRANSPOSE_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub(super) fn parse_ffn_transpose(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_GEMMA4_FFN_TRANSPOSE must be exactly 0 or 1"),
    }
}

fn ffn_transpose_requested() -> Result<bool> {
    match std::env::var("ATLAS_GEMMA4_FFN_TRANSPOSE") {
        Ok(value) => parse_ffn_transpose(Some(value.as_str())).map_err(anyhow::Error::msg),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("ATLAS_GEMMA4_FFN_TRANSPOSE must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

/// Bytes of the transposed gate/up/down copies across `layers` dense layers.
/// NVFP4: half a byte per weight plus one ue4m3 scale per 16, rows padded to
/// 64 columns the way `QuantizedWeight::transpose_for_gemm` lays them out.
pub(super) const fn ffn_transpose_bytes(
    hidden: usize,
    intermediate: usize,
    layers: usize,
) -> usize {
    // gate/up: N = intermediate, K = hidden. down: N = hidden, K = intermediate.
    let gate_or_up = transposed_bytes(intermediate, hidden);
    let down = transposed_bytes(hidden, intermediate);
    (2 * gate_or_up + down) * layers
}

/// `[K/2, N_pad]` packed data plus `[K/16, N_pad]` block scales.
const fn transposed_bytes(n: usize, k: usize) -> usize {
    (k / 2 + k / 16) * (n.div_ceil(64) * 64)
}

pub(super) const fn ffn_transpose_fits(needed: usize, free: usize) -> bool {
    needed <= free.saturating_sub(FFN_TRANSPOSE_HEADROOM_BYTES)
}

/// Whether to build the copies for every dense layer. An explicit request
/// that does not fit fails the load instead of silently serving the slow
/// fallback that the request was meant to replace. When `bf16_mlp` is active,
/// BF16 MLP weights supersede the NVFP4 prefill path, so the request is
/// skipped before probing free memory.
pub(super) fn plan_ffn_prefill_transpose(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    bf16_mlp: bool,
) -> Result<bool> {
    if !ffn_transpose_requested()? {
        return Ok(false);
    }
    if bf16_mlp {
        tracing::info!(
            "ATLAS_GEMMA4_FFN_TRANSPOSE=1 skipped because ATLAS_GEMMA4_BF16_MLP=1 \
             serves BF16 MLP weights instead of NVFP4 transposed prefill copies"
        );
        return Ok(false);
    }
    let needed = ffn_transpose_bytes(
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
    );
    let free = gpu.free_memory()?;
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    anyhow::ensure!(
        ffn_transpose_fits(needed, free),
        "ATLAS_GEMMA4_FFN_TRANSPOSE=1 needs {:.1} GiB for transposed FFN copies plus \
         {:.1} GiB headroom, but only {:.1} GiB is free before the layer loop",
        gib(needed),
        gib(FFN_TRANSPOSE_HEADROOM_BYTES),
        gib(free),
    );
    tracing::info!(
        "ATLAS_GEMMA4_FFN_TRANSPOSE=1: building transposed NVFP4 FFN copies \
         ({:.1} GiB, {:.1} GiB free) for the w4a16_gemm_t_m128 prefill route",
        gib(needed),
        gib(free),
    );
    Ok(true)
}

/// Build and install one dense layer's transposed gate/up/down copies.
pub(super) fn install_ffn_prefill_transpose(
    ffn_layer: &mut DenseFfnLayer,
    gpu: &dyn GpuBackend,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    let gate_t = ffn_layer
        .weights
        .gate_proj
        .transpose_for_gemm(gpu, intermediate, hidden)?;
    let up_t = ffn_layer
        .weights
        .up_proj
        .transpose_for_gemm(gpu, intermediate, hidden)?;
    let down_t = ffn_layer
        .weights
        .down_proj
        .transpose_for_gemm(gpu, hidden, intermediate)?;
    ffn_layer.set_prefill_transposed_weights(gate_t, up_t, down_t);
    Ok(())
}
