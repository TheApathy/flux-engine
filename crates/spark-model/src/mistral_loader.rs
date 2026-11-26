// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for Mistral Small 4 (absorbed MLA + MoE architecture).
//!
//! Loads LoRA Q/KV projections, precomputes absorbed MLA views (`W_UK_T`,
//! `W_UV`, `W_QK_absorbed`, and block-diagonal views) for compressed latent KV
//! caching, and attaches the per-layer MoE FFN.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub struct MistralWeightLoader;

/// Allocate GPU memory, falling back to managed (UVM) if device alloc fails.
/// Uses a static flag to avoid retrying device alloc after the first failure
/// (which wastes time and fragments memory).
fn gpu_alloc_or_managed(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static USE_MANAGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if USE_MANAGED.load(std::sync::atomic::Ordering::Relaxed) {
        return gpu.alloc_managed(bytes);
    }
    match gpu.alloc(bytes) {
        Ok(p) => Ok(p),
        Err(_) => {
            tracing::warn!(
                "GPU alloc failed ({bytes} bytes) — switching to managed for remaining allocations"
            );
            USE_MANAGED.store(true, std::sync::atomic::Ordering::Relaxed);
            gpu.alloc_managed(bytes)
        }
    }
}

mod loader_impl;
