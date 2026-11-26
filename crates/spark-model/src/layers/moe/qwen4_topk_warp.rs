// SPDX-License-Identifier: AGPL-3.0-only

//! Launch side of `ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP` (rule and contract in
//! [`super::qwen4_topk_warp_rule`]): one receipt line, ENGAGED once or REFUSED once per reason.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4_topk_warp_rule as rule;

/// Single-warp-per-CTA twin in `moe_topk.cu` for small-row SM occupancy on high-SM GPUs.
const SINGLE_WARP_KERNEL: &str = "moe_topk_softmax_batched_w32_1w";
/// Maximum token count for 1-warp/CTA dispatch (above 256 rows, 8-warp/CTA already fills SMs).
const SINGLE_WARP_MAX_TOKENS: u32 = 256;
/// SM count threshold above which small-row 1-warp/CTA dispatch engages (GB10 <= 64 stays 8-warp).
const SINGLE_WARP_HIGH_SM_THRESHOLD: u32 = 64;
/// Threads per CTA for the 1-warp/CTA twin (one token per CTA).
const SINGLE_WARP_BLOCK_THREADS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopkWarpVariant {
    EightWarpsPerCta,
    SingleWarpPerCta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TopkWarpLaunch {
    variant: TopkWarpVariant,
    grid_x: u32,
    block_x: u32,
}

#[derive(Clone, Copy)]
struct ResolvedTopkLaunch {
    handle: KernelHandle,
    kernel_name: &'static str,
    grid_x: u32,
    block_x: u32,
}

#[must_use]
fn select_topk_warp_launch(num_tokens: u32, sm_count: u32) -> TopkWarpLaunch {
    if sm_count > SINGLE_WARP_HIGH_SM_THRESHOLD && num_tokens <= SINGLE_WARP_MAX_TOKENS {
        TopkWarpLaunch {
            variant: TopkWarpVariant::SingleWarpPerCta,
            grid_x: num_tokens,
            block_x: SINGLE_WARP_BLOCK_THREADS,
        }
    } else {
        TopkWarpLaunch {
            variant: TopkWarpVariant::EightWarpsPerCta,
            grid_x: rule::grid(num_tokens),
            block_x: rule::BLOCK_THREADS,
        }
    }
}

pub(super) fn selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| rule::parse(std::env::var(rule::SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("moe_topk", rule::KERNEL)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| {
        anyhow::anyhow!(
            "{}: kernel {} unavailable: {e}",
            rule::SELECTOR,
            rule::KERNEL
        )
    })
}

fn kernel_single_warp(gpu: &dyn GpuBackend) -> Option<KernelHandle> {
    static K1W: OnceLock<Option<KernelHandle>> = OnceLock::new();
    *K1W.get_or_init(|| gpu.kernel("moe_topk", SINGLE_WARP_KERNEL).ok())
}

/// Resolve every fallible selector/kernel decision without launching work.
/// Refused geometries deliberately remain valid because execution will use the
/// already-preflighted softmax fallback.
pub(super) fn preflight(
    gpu: &dyn GpuBackend,
    gate_logits: DevicePtr,
    num_experts: u32,
    top_k: u32,
    num_tokens: u32,
) -> Result<()> {
    if selected()? && rule::refusal(num_experts, top_k, num_tokens, gate_logits.0).is_none() {
        kernel(gpu)?;
        if atlas_core::device::active_sm_count() > SINGLE_WARP_HIGH_SM_THRESHOLD {
            let _ = kernel_single_warp(gpu);
        }
    }
    Ok(())
}

fn log_refusal(reason: &'static str, num_experts: u32, top_k: u32, rows: u32) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(
            "MOE_TOPK_WARP_REFUSED selector={} reason={reason} experts={num_experts} top_k={top_k} rows={rows} fallback=moe_topk_softmax_batched",
            rule::SELECTOR
        );
    }
}

/// The softmax router's top-k through the warp twin when selected and admissible. `Ok(false)`:
/// the caller runs `moe_topk_softmax_batched` (selector off, or this dispatch refused).
#[allow(clippy::too_many_arguments)]
pub(super) fn try_launch(
    gpu: &dyn GpuBackend,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    num_tokens: u32,
    stream: u64,
) -> Result<bool> {
    if !selected()? {
        return Ok(false);
    }
    if let Some(reason) = rule::refusal(num_experts, top_k, num_tokens, gate_logits.0) {
        log_refusal(reason, num_experts, top_k, num_tokens);
        return Ok(false);
    }
    let launch = select_topk_warp_launch(num_tokens, atlas_core::device::active_sm_count());
    let resolved = match launch.variant {
        TopkWarpVariant::SingleWarpPerCta => match kernel_single_warp(gpu) {
            Some(k1w) => ResolvedTopkLaunch {
                handle: k1w,
                kernel_name: SINGLE_WARP_KERNEL,
                grid_x: launch.grid_x,
                block_x: launch.block_x,
            },
            None => ResolvedTopkLaunch {
                handle: kernel(gpu)?,
                kernel_name: rule::KERNEL,
                grid_x: rule::grid(num_tokens),
                block_x: rule::BLOCK_THREADS,
            },
        },
        TopkWarpVariant::EightWarpsPerCta => ResolvedTopkLaunch {
            handle: kernel(gpu)?,
            kernel_name: rule::KERNEL,
            grid_x: launch.grid_x,
            block_x: launch.block_x,
        },
    };
    KernelLaunch::new(gpu, resolved.handle)
        .grid([resolved.grid_x, 1, 1])
        .block([resolved.block_x, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_tokens)
        .arg_u32(top_k)
        .arg_u32(u32::from(normalize))
        .launch(stream)?;
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            "MOE_TOPK_WARP_ENGAGED selector={} kernel={} rows={num_tokens} experts={num_experts} top_k={top_k} normalize={normalize} bit_identical_to=moe_topk_softmax_batched",
            rule::SELECTOR,
            resolved.kernel_name
        );
    });
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_row_topk_uses_single_warp_on_high_sm_and_preserves_gb10() {
        for sm in [0u32, 48, 64] {
            for rows in [1u32, 8, 128, 256, 257, 8192] {
                let got = select_topk_warp_launch(rows, sm);
                assert_eq!(
                    got,
                    TopkWarpLaunch {
                        variant: TopkWarpVariant::EightWarpsPerCta,
                        grid_x: rule::grid(rows),
                        block_x: rule::BLOCK_THREADS,
                    },
                    "sm={sm} rows={rows} must keep 8-warp/CTA GB10 dispatch"
                );
            }
        }

        for sm in [65u32, 110, 148, 188] {
            for rows in [1u32, 8, 64, 128, 256] {
                let got = select_topk_warp_launch(rows, sm);
                assert_eq!(
                    got,
                    TopkWarpLaunch {
                        variant: TopkWarpVariant::SingleWarpPerCta,
                        grid_x: rows,
                        block_x: 32,
                    },
                    "sm={sm} rows={rows} must expand to 1-warp/CTA"
                );
            }
            for rows in [257u32, 512, 8192, 16_384] {
                let got = select_topk_warp_launch(rows, sm);
                assert_eq!(
                    got,
                    TopkWarpLaunch {
                        variant: TopkWarpVariant::EightWarpsPerCta,
                        grid_x: rule::grid(rows),
                        block_x: rule::BLOCK_THREADS,
                    },
                    "sm={sm} rows={rows} above 256 should return to 8-warp/CTA"
                );
            }
        }
    }
}
