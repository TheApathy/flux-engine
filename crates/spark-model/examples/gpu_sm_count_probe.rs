// SPDX-License-Identifier: AGPL-3.0-only

//! Print the device's real SM count and the drafter split-K slice counts it
//! implies.
//!
//! Every occupancy decision in this tree used to rest on a hardcoded SM
//! literal, and the literals disagreed: 48 in `layers/mod.rs` and
//! `qwen3_ssm/mod.rs`, 110 in `dflash_head/draft_splitk.rs` and
//! `ops/gemm_dense.rs`. Nothing had ever asked the device. This asks.
//!
//! SAFE TO RUN ALONGSIDE A BENCHMARK: the SM count comes from
//! `cuDeviceGetAttribute`, which reads a device property. No kernel is
//! launched and no device memory is allocated by this probe. It does create a
//! CUDA context (unavoidable through the backend), so it is not free — but it
//! does no GPU work.
//!
//! ```text
//! ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run --release -p spark-model --features cuda \
//!   --example gpu_sm_count_probe
//! ```

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

/// Mirrors `draft_splitk::STARVED_CTA_LIMIT` (48-SM GB10 floor; private to that module).
/// Production scales the active limit via `sm_count.saturating_mul(4).max(STARVED_CTA_LIMIT)`.
const SHIPPED_STARVED_CTA_LIMIT: u32 = 256;

/// Drafter projection widths under the production flag set.
const DRAFTER_SHAPES: [(&str, u32); 5] = [
    ("kv_noise", 1_024),
    ("q_proj", 4_096),
    ("o_proj", 5_120),
    ("down_proj", 5_120),
    ("gate_up", 17_408),
];

fn cta_count(n: u32) -> u32 {
    n.div_ceil(64)
}

fn slices_for(n: u32, budget: u32, limit: u32) -> u32 {
    let ctas = cta_count(n);
    if ctas >= limit {
        return 0;
    }
    limit.div_ceil(ctas.max(1)).min(budget)
}

fn print_table(label: &str, limit: u32, budget: u32) {
    println!("\n  {label} (STARVED_CTA_LIMIT={limit}, budget={budget})");
    for (name, n) in DRAFTER_SHAPES {
        println!(
            "    {name:<10} N={n:<6} CTAs={:<4} slices={}",
            cta_count(n),
            slices_for(n, budget, limit)
        );
    }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("initialize CUDA backend")?;
    let gpu: &dyn GpuBackend = &backend;

    match gpu.sm_count() {
        Some(sms) => {
            let active_limit = sms.saturating_mul(4).max(SHIPPED_STARVED_CTA_LIMIT);
            println!("device multiprocessor count: {sms}");
            println!(
                "  shipped starved_cta_limit_for_sm({sms})={active_limit} (floor={SHIPPED_STARVED_CTA_LIMIT}) = {:.2} CTAs/SM",
                f64::from(active_limit) / f64::from(sms.max(1))
            );
            println!(
                "  the drafter table's saturated shape (gate_up, 272 CTAs) = {:.2} CTAs/SM",
                272.0 / f64::from(sms.max(1))
            );
            println!(
                "  the drafter table's starved shapes (o/down, 80 CTAs)    = {:.2} CTAs/SM",
                80.0 / f64::from(sms.max(1))
            );
            print_table("shipped, SM-scaled (max(4 x SMs, 256))", active_limit, 8);
            print_table("GB10 baseline floor", SHIPPED_STARVED_CTA_LIMIT, 8);
            print_table("if re-derived as 2.5 x SMs", (sms as f64 * 2.5) as u32, 8);
            println!(
                "\n`STARVED_CTA_LIMIT = {SHIPPED_STARVED_CTA_LIMIT}` is the 48-SM GB10 baseline floor.\n\
                 Production `draft_splitk::starved_cta_limit_for_sm` scales the active threshold to\n\
                 `max(sm_count * 4, {SHIPPED_STARVED_CTA_LIMIT})` on wider GPUs (SM120/SM103) so high-SM\n\
                 devices stay saturated, while keeping 256 on 48-SM GB10."
            );
        }
        None => println!(
            "device multiprocessor count: UNAVAILABLE (driver query failed) — \
             occupancy sizing falls back to its compiled-in literal"
        ),
    }
    Ok(())
}
