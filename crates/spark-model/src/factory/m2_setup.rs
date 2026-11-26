// SPDX-License-Identifier: AGPL-3.0-only

//! MiniMax M2 layout-mode memory audit (extracted from `build_model`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::startup_ledger;

use crate::layer::TransformerLayer;

fn signed_byte_delta(after: usize, before: usize) -> Result<i64> {
    if after >= before {
        i64::try_from(after - before)
            .map_err(|_| anyhow::anyhow!("positive GPU free-memory delta exceeds i64"))
    } else {
        i64::try_from(before - after)
            .map_err(|_| anyhow::anyhow!("negative GPU free-memory delta exceeds i64"))?
            .checked_neg()
            .ok_or_else(|| anyhow::anyhow!("negative GPU free-memory delta overflow"))
    }
}

fn checked_transpose_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| anyhow::anyhow!("MoE transpose memory product overflow"))
    })
}

fn checked_transpose_sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| anyhow::anyhow!("MoE transpose memory sum overflow"))
    })
}

fn transposed_nvfp4_bytes(rows: usize, columns: usize) -> Result<(usize, usize)> {
    // Match QuantizedWeight::transpose_for_gemm's padded rows and group size.
    let padded_rows = rows
        .checked_add(63)
        .map(|rows| rows / 64 * 64)
        .ok_or_else(|| anyhow::anyhow!("MoE transpose padded rows overflow"))?;
    let packed = checked_transpose_product(&[padded_rows, columns / 2])?;
    let scale = checked_transpose_product(&[padded_rows, columns / 16])?;
    Ok((checked_transpose_sum(&[packed, scale])?, packed))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetainedTransposeBudget {
    full_pass_bytes: usize,
    full_layer_bytes: usize,
    unified_layer_bytes: usize,
    gate_up_pass_bytes: usize,
    gate_up_layer_bytes: usize,
    down_scratch_bytes: usize,
}

fn retained_transpose_budget(
    config: &ModelConfig,
    local_experts: usize,
    layer_count: usize,
) -> Result<RetainedTransposeBudget> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let shared = config.shared_expert_intermediate_size;
    let (gate, gate_packed) = transposed_nvfp4_bytes(inter, h)?;
    let (down, down_packed) = transposed_nvfp4_bytes(h, inter)?;
    let (shared_gate, shared_gate_packed) = transposed_nvfp4_bytes(shared, h)?;
    let (shared_down, shared_down_packed) = if shared > 0 {
        transposed_nvfp4_bytes(h, shared)?
    } else {
        (0, 0)
    };
    // Each projection table has packed/scale pointers plus f32 scale2 values.
    let table = checked_transpose_product(&[config.num_experts, 8 + 8 + 4])?;
    let gate_up_layer = checked_transpose_sum(&[
        checked_transpose_product(&[2, local_experts, gate])?,
        checked_transpose_product(&[2, shared_gate])?,
        checked_transpose_product(&[2, table])?,
    ])?;
    let full_layer = checked_transpose_sum(&[
        gate_up_layer,
        checked_transpose_product(&[local_experts, down])?,
        shared_down,
        table,
    ])?;
    // The CPU transpose holds source and destination bytes beside device data.
    let host_transient = checked_transpose_product(&[
        2,
        gate_packed
            .max(down_packed)
            .max(shared_gate_packed)
            .max(shared_down_packed),
    ])?;
    let down_scratch_bytes = checked_transpose_sum(&[
        checked_transpose_product(&[local_experts, checked_transpose_product(&[h, inter])? / 2])?,
        checked_transpose_product(&[local_experts, checked_transpose_product(&[h, inter])? / 16])?,
        checked_transpose_product(&[config.num_experts, 8 + 8])?,
    ])?;
    Ok(RetainedTransposeBudget {
        full_pass_bytes: checked_transpose_sum(&[
            checked_transpose_product(&[full_layer, layer_count])?,
            host_transient,
        ])?,
        full_layer_bytes: checked_transpose_sum(&[full_layer, host_transient])?,
        // Both canonical unified slabs have an unexposed 256-byte front guard.
        // Keep the full-layer bound and take no credit for Phase-B frees.
        unified_layer_bytes: checked_transpose_sum(&[full_layer, host_transient, 512])?,
        gate_up_pass_bytes: checked_transpose_sum(&[
            checked_transpose_product(&[gate_up_layer, layer_count])?,
            down_scratch_bytes,
            host_transient,
        ])?,
        gate_up_layer_bytes: checked_transpose_sum(&[
            gate_up_layer,
            down_scratch_bytes,
            host_transient,
        ])?,
        down_scratch_bytes,
    })
}

/// Pre-flight memory audit + MoE-transpose pass for unified / hybrid layout
/// modes.
///
/// Originally MiniMax-M2 only. Also admits Qwen4-exp (Qwen3.8-Flash-Next),
/// which has the identical problem and could never solve it inline: the
/// qwen35 loader sizes a full second copy of every expert's gate/up/down
/// (`load_layers.rs`, 63.3 GB for Flash-Next) during *layer construction*,
/// finds ~16 GB free, and permanently falls back to the grouped GEMM — so
/// prefill pays a per-token weight read and lands at ~40 tok/s, statistically
/// identical to decode. It is not tunable: the shortfall is ~46 GB.
///
/// Two things make this hook the right place. It runs POST-load, where more
/// memory is free than during layer construction; and the unified pass frees
/// each untransposed copy as it goes, so it is memory-NEUTRAL rather than
/// needing the 63.3 GB up front. `use_t_layout_for_{prefill,decode}` and the
/// transpose entry points are already model-agnostic — only this gate was
/// MiniMax-specific.
///
/// Cost model note: `per_expert_one` below is `inter * hidden * 9/16`, the
/// same packed+scale arithmetic the qwen35 loader uses, so the two agree on
/// what "full" costs.
pub(super) fn maybe_run_minimax_m2_moe_transpose(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
) -> Result<()> {
    let unified_layout = std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or_else(|_| config.is_qwen4_exp());
    let hybrid_layout = std::env::var("ATLAS_HYBRID_MOE_LAYOUT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let bounded =
        crate::layers::moe::bounded_transpose::selected(config, !unified_layout || hybrid_layout)?;
    if config.model_type != "minimax_m2" && !config.is_qwen4_exp() {
        return Ok(());
    }
    let local_experts: usize = (0..config.num_experts)
        .filter(|e| config.is_local_expert(*e))
        .count();
    // Per expert weight = packed (n*k/2 bytes) + scale (n*k/16 bytes) =
    // n*k * 9/16 bytes, where n*k = inter*hidden.
    let per_expert_one =
        checked_transpose_product(&[config.moe_intermediate_size, config.hidden_size, 9])? / 16;
    let cost_full =
        checked_transpose_product(&[local_experts, 3, per_expert_one, config.num_hidden_layers])?;
    let cost_gate_up =
        checked_transpose_product(&[local_experts, 2, per_expert_one, config.num_hidden_layers])?;
    let safety: usize = 2 * 1024 * 1024 * 1024; // 2 GB
    let hybrid_backend_need =
        checked_transpose_sum(&[checked_transpose_product(&[2, cost_full])?, safety])?;
    let full_backend_need = checked_transpose_sum(&[cost_full, safety])?;
    let gate_up_backend_need = checked_transpose_sum(&[cost_gate_up, safety])?;
    let mut retained_budget = retained_transpose_budget(config, local_experts, layers.len())?;
    if let Some(owner_cap) = bounded {
        retained_budget.unified_layer_bytes = checked_transpose_sum(&[
            retained_budget.unified_layer_bytes,
            crate::layers::moe::bounded_transpose::admission_guard_delta(config, owner_cap)?,
        ])?;
    }
    let free = gpu.free_memory()?;
    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    // Hybrid mode pre-flight: Block C Path 2 needs ~2× the cost_full
    // budget — keeps originals AND builds transposed copies. Fall back
    // to unified if it doesn't fit (defends against KV-cache-heavy
    // configurations exceeding the 122 GB GB10 budget).
    let hybrid_fits = hybrid_layout && free >= hybrid_backend_need;
    if hybrid_layout && !hybrid_fits {
        tracing::warn!(
            "MoE transpose pass (hybrid layout): ATLAS_HYBRID_MOE_LAYOUT=1 \
             requested but doesn't fit (need {:.1} GB, free {:.1} GB) — \
             falling back to unified-layout (decode regression).",
            gb(hybrid_backend_need),
            gb(free),
        );
    }
    if hybrid_fits {
        // Block C Path 2 hybrid-layout: keep originals + add transposed.
        // Decode + MTP verify route through originals (warp-reduction
        // 35 tok/s preserved); prefill (forward_batched) routes through
        // transposed (Phase 8a coalesced TTFT win retained).
        // Originals are already deducted from MemAvailable; charge only the
        // new retained set here, while preserving the legacy backend gate.
        ensure_retained_pass_above_floor(gpu, "hybrid", retained_budget.full_pass_bytes, safety)?;
        tracing::info!(
            "MoE transpose pass (hybrid layout): ATLAS_HYBRID_MOE_LAYOUT=1, \
             dual-layout (cost {:.1} GB), free pre-pass {:.1} GB → RUNNING",
            gb(hybrid_backend_need - safety),
            gb(free),
        );
        for layer in layers.iter_mut() {
            ensure_retained_pass_above_floor(
                gpu,
                "hybrid layer",
                retained_budget.full_layer_bytes,
                safety,
            )?;
            layer.transpose_moe_for_prefill_hybrid(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass (hybrid layout): done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if unified_layout {
        // Phase 8a unified-layout: phased transpose with originals
        // freed between phases. Fits in tight budgets that the
        // straight `transpose_moe_for_prefill` would reject.
        //
        // Floor guard: the pass holds one layer's transposed set beside that
        // layer's originals (1.35 GiB on Flash-Next). This path had no budget
        // check at all, and the qwen4 branch of the `load_layers.rs` check is
        // skipped (`defer_to_m2_setup`), so the guard there read as satisfied.
        ensure_unified_pass_above_floor(gpu, retained_budget.unified_layer_bytes)?;
        tracing::info!(
            "MoE transpose pass (unified layout): ATLAS_UNIFIED_MOE_LAYOUT=1, \
             phased transpose with frees, free pre-pass {:.1} GB → RUNNING",
            gb(free),
        );
        for (layer_index, layer) in layers.iter_mut().enumerate() {
            ensure_unified_pass_above_floor(gpu, retained_budget.unified_layer_bytes)?;
            let span = bounded.map(|_| {
                tracing::debug_span!(
                    target: "atlas_bounded_transpose_allocation",
                    "bounded_transpose_layer", layer_index, layer_ordinal = layer_index + 1,
                )
            });
            let _entered = span.as_ref().map(tracing::Span::enter);
            if bounded.is_some() {
                tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "layer_begin");
            }
            let result = layer.transpose_moe_for_prefill_unified(gpu, config);
            if bounded.is_some() {
                match &result {
                    Ok(_) => {
                        tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "layer_end")
                    }
                    Err(_) => {
                        tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "layer_error")
                    }
                }
            }
            result?;
        }
        let free_post = gpu.free_memory()?;
        let free_delta = signed_byte_delta(free_post, free)?;
        tracing::info!(
            "MoE transpose pass (unified layout): done, {:.1} GB free",
            gb(free_post)
        );
        tracing::info!(
            target: "atlas_transposed_expert_slab",
            schema = "atlas.moe-transpose-pass-memory.v1",
            layout = "unified",
            free_pre_bytes = free,
            free_post_bytes = free_post,
            free_delta_bytes = free_delta,
            "MOE_TRANSPOSE_PASS_MEMORY"
        );
    } else if free >= full_backend_need {
        ensure_retained_pass_above_floor(gpu, "full", retained_budget.full_pass_bytes, safety)?;
        tracing::info!(
            "MoE transpose pass: cost {:.1} GB (full gate+up+down), \
             free {:.1} GB → RUNNING",
            gb(cost_full),
            gb(free),
        );
        for layer in layers.iter_mut() {
            ensure_retained_pass_above_floor(
                gpu,
                "full layer",
                retained_budget.full_layer_bytes,
                safety,
            )?;
            layer.transpose_moe_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: done, {:.1} GB free",
            gb(gpu.free_memory()?)
        );
    } else if free >= gate_up_backend_need {
        ensure_retained_pass_above_floor(
            gpu,
            "gate+up",
            retained_budget.gate_up_pass_bytes,
            safety,
        )?;
        tracing::info!(
            "MoE transpose pass: full cost {:.1} GB > free {:.1} GB; \
             falling back to gate+up only (cost {:.1} GB) → RUNNING",
            gb(cost_full),
            gb(free),
            gb(cost_gate_up),
        );
        for layer in layers.iter_mut() {
            ensure_retained_pass_above_floor(
                gpu,
                "gate+up layer",
                retained_budget.gate_up_layer_bytes,
                safety,
            )?;
            layer.transpose_moe_gate_up_for_prefill(gpu, config)?;
        }
        tracing::info!(
            "MoE transpose pass: gate+up done, {:.1} GB free \
             (down: per-prefill scratch transpose)",
            gb(gpu.free_memory()?),
        );

        // ── Lazy down_proj transpose scratch ──
        //
        // The persistent transpose pass couldn't fit down (would have
        // needed another `cost_full - cost_gate_up` GB). Instead we
        // allocate a SINGLE shared scratch sized for ONE layer's down
        // weights, plus per-expert pointer tables that select into the
        // scratch. Every MoE layer reuses the same scratch — at the
        // start of each layer's prefill, a batched transpose kernel
        // populates the scratch from the layer's untransposed source
        // (`down_ptrs`). Decode unaffected — it still reads
        // `down_ptrs` directly.
        //
        // Memory: one layer's down packed = num_experts × N × K/2;
        // scale = num_experts × N × K/16. For MiniMax M2.7 NVFP4 EP=2
        // (128 local experts × 3072 × 1024/2 packed + same shape for
        // scale × 1/8 group): ~200 MB packed + ~24 MB scale ≈ 230 MB.
        // Fits comfortably in the 7 GB free post-gate+up.
        let n_per_expert_packed =
            checked_transpose_product(&[config.hidden_size, config.moe_intermediate_size])? / 2;
        let n_per_expert_scale =
            checked_transpose_product(&[config.hidden_size, config.moe_intermediate_size])? / 16;
        let scratch_packed_bytes =
            checked_transpose_product(&[local_experts, n_per_expert_packed])?;
        let scratch_scale_bytes = checked_transpose_product(&[local_experts, n_per_expert_scale])?;
        let scratch_pointer_table_bytes = checked_transpose_product(&[config.num_experts, 8])?;
        let scratch_allocation_bytes = checked_transpose_sum(&[
            scratch_packed_bytes,
            scratch_scale_bytes,
            checked_transpose_product(&[2, scratch_pointer_table_bytes])?,
        ])?;
        anyhow::ensure!(
            scratch_allocation_bytes == retained_budget.down_scratch_bytes,
            "MoE down scratch budget disagrees with requested allocation bytes"
        );
        ensure_retained_pass_above_floor(
            gpu,
            "gate+up down scratch",
            scratch_allocation_bytes,
            safety,
        )?;
        let scratch_packed = gpu.alloc(scratch_packed_bytes)?;
        let scratch_scale = gpu.alloc(scratch_scale_bytes)?;

        // Build per-expert ptr tables pointing into the shared scratch.
        // Local experts get real offsets; remote experts get NULL so
        // the prefill GEMM kernel skips them (existing behaviour).
        let mut packed_ptrs_host = Vec::<u8>::with_capacity(scratch_pointer_table_bytes);
        let mut scale_ptrs_host = Vec::<u8>::with_capacity(scratch_pointer_table_bytes);
        let mut local_idx = 0usize;
        for e in 0..config.num_experts {
            let (p_ptr, s_ptr) = if config.is_local_expert(e) {
                let p = scratch_packed.0 + (local_idx * n_per_expert_packed) as u64;
                let s = scratch_scale.0 + (local_idx * n_per_expert_scale) as u64;
                local_idx += 1;
                (p, s)
            } else {
                (0u64, 0u64)
            };
            packed_ptrs_host.extend_from_slice(&p_ptr.to_le_bytes());
            scale_ptrs_host.extend_from_slice(&s_ptr.to_le_bytes());
        }
        let packed_ptrs_t = gpu.alloc(scratch_pointer_table_bytes)?;
        gpu.copy_h2d(&packed_ptrs_host, packed_ptrs_t)?;
        let scale_ptrs_t = gpu.alloc(scratch_pointer_table_bytes)?;
        gpu.copy_h2d(&scale_ptrs_host, scale_ptrs_t)?;
        for layer in layers.iter_mut() {
            layer.set_moe_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        tracing::info!(
            "MoE down scratch: {:.0} MB packed + {:.0} MB scale, shared across {} layers",
            scratch_packed_bytes as f64 / (1024.0 * 1024.0),
            scratch_scale_bytes as f64 / (1024.0 * 1024.0),
            config.num_hidden_layers,
        );
    } else {
        tracing::warn!(
            "MoE transpose pass: cost {:.1} GB (full) / {:.1} GB (gate+up), \
             free {:.1} GB → SKIP (insufficient memory; prefill uses \
             uncoalesced fallback, TTFT will be ~2× slower)",
            gb(cost_full),
            gb(cost_gate_up),
            gb(free),
        );
    }
    Ok(())
}

/// Minimum discrete GPU VRAM headroom reserved during MoE transpose passes (512 MiB).
const DISCRETE_TRANSPOSE_VRAM_MARGIN_BYTES: usize =
    startup_ledger::DEFAULT_DISCRETE_VRAM_FLOOR_MIB * startup_ledger::MIB;

/// Refuse the unified pass when its per-layer transient would push the active
/// memory pool under its floor:
/// - On UMA (`gpu.is_unified_memory()`): checks host `MemAvailable` against the
///   declared GB10 floor (6656 MiB).
/// - On discrete GPUs (`!gpu.is_unified_memory()`): checks `gpu.free_memory()`
///   against `per_layer_transient + 512 MiB` VRAM margin.
fn ensure_unified_pass_above_floor(gpu: &dyn GpuBackend, per_layer_transient: usize) -> Result<()> {
    if !gpu.is_unified_memory() {
        return discrete_transpose_vram_decision(
            "unified per-layer transient",
            gpu.free_memory().ok(),
            per_layer_transient,
            DISCRETE_TRANSPOSE_VRAM_MARGIN_BYTES,
            startup_ledger::floor_override_selected(),
        );
    }
    let floor = startup_ledger::memavailable_floor_bytes()?;
    unified_pass_telemetry_decision(
        startup_ledger::mem_available_bytes(),
        per_layer_transient,
        floor,
        startup_ledger::floor_override_selected(),
        cfg!(target_os = "linux"),
    )
}

fn ensure_retained_pass_above_floor(
    gpu: &dyn GpuBackend,
    layout: &str,
    allocation_bytes: usize,
    safety: usize,
) -> Result<()> {
    if !gpu.is_unified_memory() {
        return discrete_transpose_vram_decision(
            layout,
            gpu.free_memory().ok(),
            allocation_bytes,
            DISCRETE_TRANSPOSE_VRAM_MARGIN_BYTES,
            startup_ledger::floor_override_selected(),
        );
    }
    let floor = startup_ledger::memavailable_floor_bytes()?;
    let allocation_budget = checked_transpose_sum(&[allocation_bytes, safety])?;
    transpose_pass_telemetry_decision(
        layout,
        startup_ledger::mem_available_bytes(),
        allocation_budget,
        floor,
        startup_ledger::floor_override_selected(),
        cfg!(target_os = "linux"),
    )
}

fn discrete_transpose_vram_decision(
    layout: &str,
    vram_free: Option<usize>,
    allocation_bytes: usize,
    vram_margin: usize,
    override_selected: bool,
) -> Result<()> {
    let mib = |b: usize| b / startup_ledger::MIB;
    let Some(available) = vram_free else {
        if override_selected {
            tracing::warn!(
                "MoE transpose pass ({layout}): discrete GPU free VRAM unreadable; \
                 margin not enforced, NONQUALIFYING run"
            );
            return Ok(());
        }
        anyhow::bail!(
            "MoE transpose pass ({layout}) refused: discrete GPU free VRAM unreadable; \
             set {}=1 only for an explicitly NONQUALIFYING run.",
            startup_ledger::FLOOR_OVERRIDE_SELECTOR
        );
    };
    let need = allocation_bytes
        .checked_add(vram_margin)
        .ok_or_else(|| anyhow::anyhow!("MoE transpose pass discrete VRAM margin overflow"))?;
    if available >= need {
        return Ok(());
    }
    if override_selected {
        tracing::warn!(
            "MoE transpose pass ({layout}): discrete GPU free VRAM {} MiB is under the {} MiB \
             allocation + {} MiB VRAM margin; NONQUALIFYING run because {}=1",
            mib(available),
            mib(allocation_bytes),
            mib(vram_margin),
            startup_ledger::FLOOR_OVERRIDE_SELECTOR,
        );
        return Ok(());
    }
    anyhow::bail!(
        "MoE transpose pass ({layout}) refused: discrete GPU has {} MiB free VRAM, but the \
         modeled allocation ({} MiB) plus the {} MiB discrete VRAM margin needs {} MiB. \
         Lower --max-seq-len or set {}=1 only with a measured receipt.",
        mib(available),
        mib(allocation_bytes),
        mib(vram_margin),
        mib(need),
        startup_ledger::FLOOR_OVERRIDE_SELECTOR,
    )
}

fn unified_pass_telemetry_decision(
    mem_available: Option<usize>,
    per_layer_transient: usize,
    floor: usize,
    override_selected: bool,
    is_linux: bool,
) -> Result<()> {
    transpose_pass_telemetry_decision(
        "unified per-layer transient",
        mem_available,
        per_layer_transient,
        floor,
        override_selected,
        is_linux,
    )
}

fn transpose_pass_telemetry_decision(
    layout: &str,
    mem_available: Option<usize>,
    allocation_budget: usize,
    floor: usize,
    override_selected: bool,
    is_linux: bool,
) -> Result<()> {
    match mem_available {
        Some(available) => transpose_pass_floor_decision(
            layout,
            available,
            allocation_budget,
            floor,
            override_selected,
        ),
        None if is_linux && !override_selected => anyhow::bail!(
            "MoE transpose pass ({layout}) refused: /proc/meminfo has no readable MemAvailable; \
             set {}=1 only for an explicitly NONQUALIFYING run.",
            startup_ledger::FLOOR_OVERRIDE_SELECTOR
        ),
        None => {
            tracing::warn!(
                "MoE transpose pass ({layout}): MemAvailable unreadable; floor not enforced, NONQUALIFYING run"
            );
            Ok(())
        }
    }
}

fn transpose_pass_floor_decision(
    layout: &str,
    mem_available: usize,
    allocation_budget: usize,
    floor: usize,
    override_selected: bool,
) -> Result<()> {
    let mib = |b: usize| b / startup_ledger::MIB;
    let need = allocation_budget
        .checked_add(floor)
        .ok_or_else(|| anyhow::anyhow!("MoE transpose pass floor arithmetic overflow"))?;
    if mem_available >= need {
        return Ok(());
    }
    if override_selected {
        tracing::warn!(
            "MoE transpose pass ({layout}): MemAvailable {} MiB is under the {} MiB \
             allocation budget + {} MiB floor; NONQUALIFYING run because {}=1",
            mib(mem_available),
            mib(allocation_budget),
            mib(floor),
            startup_ledger::FLOOR_OVERRIDE_SELECTOR,
        );
        return Ok(());
    }
    anyhow::bail!(
        "MoE transpose pass ({layout}) refused: MemAvailable {} MiB, but the modeled allocation \
         budget ({} MiB) plus the {} MiB floor ({}) needs {} MiB. \
         Free host memory or lower --max-seq-len; set {}=1 only with a measured receipt.",
        mib(mem_available),
        mib(allocation_budget),
        mib(floor),
        startup_ledger::FLOOR_SELECTOR,
        mib(need),
        startup_ledger::FLOOR_OVERRIDE_SELECTOR,
    )
}

#[cfg(test)]
fn unified_pass_floor_decision(
    mem_available: usize,
    per_layer_transient: usize,
    floor: usize,
    override_selected: bool,
) -> Result<()> {
    transpose_pass_floor_decision(
        "unified per-layer transient",
        mem_available,
        per_layer_transient,
        floor,
        override_selected,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        RetainedTransposeBudget, checked_transpose_product, checked_transpose_sum,
        retained_transpose_budget, signed_byte_delta, transpose_pass_floor_decision,
        transpose_pass_telemetry_decision, unified_pass_floor_decision,
        unified_pass_telemetry_decision,
    };
    use atlas_core::config::ModelConfig;

    const MIB: usize = 1024 * 1024;

    #[test]
    fn unified_pass_floor_passes_with_the_transient_above_the_floor() {
        // r8 095613Z after the transpose moved: ~19.5 GiB free post-load.
        assert!(unified_pass_floor_decision(19_968 * MIB, 1_350 * MIB, 6_656 * MIB, false).is_ok());
        assert!(unified_pass_floor_decision(8_006 * MIB, 1_350 * MIB, 6_656 * MIB, false).is_ok());
    }

    #[test]
    fn unified_pass_floor_refuses_one_mib_short_unless_overridden() {
        let error = unified_pass_floor_decision(8_005 * MIB, 1_350 * MIB, 6_656 * MIB, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ATLAS_MEMAVAILABLE_FLOOR_MIB"), "{error}");
        assert!(error.contains("8006 MiB"), "{error}");
        assert!(unified_pass_floor_decision(8_005 * MIB, 1_350 * MIB, 6_656 * MIB, true).is_ok());
    }

    #[test]
    fn unified_pass_refuses_missing_linux_telemetry_before_allocating() {
        let error = unified_pass_telemetry_decision(None, MIB, 6_656 * MIB, false, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("/proc/meminfo"), "{error}");
        assert!(
            error.contains("ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1"),
            "{error}"
        );
        assert!(unified_pass_telemetry_decision(None, MIB, 6_656 * MIB, true, true).is_ok());
        assert!(unified_pass_telemetry_decision(None, MIB, 6_656 * MIB, false, false).is_ok());
        assert!(
            unified_pass_telemetry_decision(Some(6_657 * MIB), MIB, 6_656 * MIB, false, true)
                .is_ok()
        );
    }

    #[test]
    fn unified_pass_telemetry_guard_precedes_the_transpose_loop() {
        let source = include_str!("m2_setup.rs");
        let start = source
            .find("} else if unified_layout {")
            .expect("unified branch");
        let branch = &source[start..];
        let guard = branch
            .find("ensure_unified_pass_above_floor(")
            .expect("host floor guard");
        let allocation = branch
            .find("layer.transpose_moe_for_prefill_unified(")
            .expect("transpose call");
        assert!(guard < allocation);
    }

    fn small_transpose_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = 64;
        config.moe_intermediate_size = 32;
        config.shared_expert_intermediate_size = 16;
        config.num_experts = 4;
        config.num_hidden_layers = 3;
        config
    }

    #[test]
    fn retained_budget_counts_padded_shared_weights_tables_and_down_scratch() {
        // Two local experts, but each projection table still has four rows.
        let budget = retained_transpose_budget(&small_transpose_config(), 2, 3).unwrap();
        assert_eq!(
            budget,
            RetainedTransposeBudget {
                full_pass_bytes: 54_928,
                full_layer_bytes: 21_040,
                unified_layer_bytes: 21_552,
                gate_up_pass_bytes: 48_416,
                gate_up_layer_bytes: 20_448,
                down_scratch_bytes: 2_368,
            }
        );
        let mut no_shared = small_transpose_config();
        no_shared.shared_expert_intermediate_size = 0;
        let without_shared = retained_transpose_budget(&no_shared, 2, 3).unwrap();
        assert_eq!(without_shared.full_pass_bytes, 39_376);
        assert_eq!(without_shared.gate_up_pass_bytes, 34_592);
        assert!(budget.full_pass_bytes > budget.full_layer_bytes);
    }

    #[test]
    fn bounded_owner_admission_adds_only_the_exact_guard_delta() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.num_hidden_layers = 48;
        config.num_experts = 512;
        config.hidden_size = 2560;
        config.moe_intermediate_size = 640;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        let original = retained_transpose_budget(&config, 512, 48).unwrap();
        let delta = crate::layers::moe::bounded_transpose::admission_guard_delta(
            &config,
            crate::layers::moe::bounded_transpose::OwnerCap::DEFAULT,
        )
        .unwrap();
        let mut enabled = original;
        enabled.unified_layer_bytes =
            checked_transpose_sum(&[original.unified_layer_bytes, delta]).unwrap();
        assert_eq!(delta, 5_632);
        assert_eq!(
            enabled.unified_layer_bytes - original.unified_layer_bytes,
            5_632
        );
        enabled.unified_layer_bytes = original.unified_layer_bytes;
        assert_eq!(enabled, original);
    }

    #[test]
    fn transpose_budget_rejects_size_padding_and_aggregate_overflow() {
        assert!(checked_transpose_product(&[usize::MAX, 2]).is_err());
        assert!(checked_transpose_sum(&[usize::MAX, 1]).is_err());
        let mut config = small_transpose_config();
        config.hidden_size = usize::MAX;
        assert!(retained_transpose_budget(&config, 2, 3).is_err());
        config = small_transpose_config();
        config.num_experts = usize::MAX;
        assert!(retained_transpose_budget(&config, 2, 3).is_err());
        assert!(retained_transpose_budget(&small_transpose_config(), 2, usize::MAX).is_err());
    }

    #[test]
    fn every_layout_refuses_missing_linux_telemetry_unless_explicitly_overridden() {
        for layout in ["hybrid", "full", "gate+up", "gate+up down scratch"] {
            let error =
                transpose_pass_telemetry_decision(layout, None, MIB, 6_656 * MIB, false, true)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains(layout), "{error}");
            assert!(error.contains("NONQUALIFYING"), "{error}");
            assert!(
                transpose_pass_telemetry_decision(layout, None, MIB, 6_656 * MIB, true, true)
                    .is_ok()
            );
            assert!(
                transpose_pass_telemetry_decision(layout, None, MIB, 6_656 * MIB, false, false)
                    .is_ok()
            );
            let need = 8_007 * MIB;
            assert!(
                transpose_pass_floor_decision(layout, need, 1_351 * MIB, 6_656 * MIB, false)
                    .is_ok()
            );
            assert!(
                transpose_pass_floor_decision(layout, need - 1, 1_351 * MIB, 6_656 * MIB, false)
                    .is_err()
            );
            assert!(
                transpose_pass_floor_decision(layout, need - 1, 1_351 * MIB, 6_656 * MIB, true)
                    .is_ok()
            );
        }
        assert!(transpose_pass_floor_decision("full", usize::MAX, usize::MAX, 1, false).is_err());
    }

    #[test]
    fn each_retained_branch_and_down_scratch_has_a_preallocation_guard() {
        let source = include_str!("m2_setup.rs");
        for (start, end, guard, allocation) in [
            (
                "if hybrid_fits {",
                "} else if unified_layout {",
                "retained_budget.full_pass_bytes",
                "layer.transpose_moe_for_prefill_hybrid(",
            ),
            (
                "} else if free >= full_backend_need {",
                "} else if free >= gate_up_backend_need {",
                "retained_budget.full_pass_bytes",
                "layer.transpose_moe_for_prefill(",
            ),
            (
                "} else if free >= gate_up_backend_need {",
                "// ── Lazy down_proj transpose scratch",
                "retained_budget.gate_up_pass_bytes",
                "layer.transpose_moe_gate_up_for_prefill(",
            ),
        ] {
            let branch = source
                .split_once(start)
                .unwrap()
                .1
                .split_once(end)
                .unwrap()
                .0;
            assert!(
                branch.find(guard).unwrap() < branch.find(allocation).unwrap(),
                "{start}"
            );
            assert!(
                branch.matches("ensure_retained_pass_above_floor(").count() >= 2,
                "{start}"
            );
        }
        let scratch = source.split_once("let scratch_scale_bytes =").unwrap().1;
        assert!(
            scratch.find("ensure_retained_pass_above_floor(").unwrap()
                < scratch.find("gpu.alloc(scratch_packed_bytes)").unwrap()
        );
    }

    #[test]
    fn free_memory_delta_preserves_direction_and_exact_bytes() {
        assert_eq!(
            signed_byte_delta(27_400_000_000, 17_000_000_000).unwrap(),
            10_400_000_000
        );
        assert_eq!(
            signed_byte_delta(17_000_000_000, 27_400_000_000).unwrap(),
            -10_400_000_000
        );
        assert_eq!(signed_byte_delta(42, 42).unwrap(), 0);
    }

    #[test]
    fn discrete_gpu_transpose_checks_vram_margin_instead_of_uma_host_floor() {
        use super::{
            DISCRETE_TRANSPOSE_VRAM_MARGIN_BYTES, discrete_transpose_vram_decision,
            ensure_retained_pass_above_floor, ensure_unified_pass_above_floor,
        };
        use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

        struct StubDiscreteGpu {
            free_bytes: usize,
        }

        impl GpuBackend for StubDiscreteGpu {
            fn alloc(&self, _bytes: usize) -> anyhow::Result<DevicePtr> {
                Ok(DevicePtr(0x1000))
            }
            fn alloc_managed(&self, _bytes: usize) -> anyhow::Result<DevicePtr> {
                Ok(DevicePtr(0x1000))
            }
            fn free(&self, _ptr: DevicePtr) -> anyhow::Result<()> {
                Ok(())
            }
            fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> anyhow::Result<()> {
                Ok(())
            }
            fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> anyhow::Result<()> {
                Ok(())
            }
            fn copy_d2d(
                &self,
                _src: DevicePtr,
                _dst: DevicePtr,
                _bytes: usize,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn launch(
                &self,
                _func: KernelHandle,
                _grid: [u32; 3],
                _block: [u32; 3],
                _shared_mem: u32,
                _stream: u64,
                _params: &mut [*mut std::ffi::c_void],
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn synchronize(&self, _stream: u64) -> anyhow::Result<()> {
                Ok(())
            }
            fn default_stream(&self) -> u64 {
                0
            }
            fn kernel(&self, _module: &str, _func_name: &str) -> anyhow::Result<KernelHandle> {
                Ok(KernelHandle(1))
            }
            fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> anyhow::Result<()> {
                Ok(())
            }
            fn memset_async(
                &self,
                _ptr: DevicePtr,
                _value: u8,
                _bytes: usize,
                _stream: u64,
            ) -> anyhow::Result<()> {
                Ok(())
            }
            fn total_memory(&self) -> anyhow::Result<usize> {
                Ok(32 * 1024 * MIB)
            }
            fn free_memory(&self) -> anyhow::Result<usize> {
                Ok(self.free_bytes)
            }
            fn is_unified_memory(&self) -> bool {
                false
            }
        }

        // Exact VRAM margin boundary: 1350 MiB unified transient + 512 MiB margin = 1862 MiB.
        assert_eq!(DISCRETE_TRANSPOSE_VRAM_MARGIN_BYTES, 512 * MIB);
        let ok_gpu = StubDiscreteGpu {
            free_bytes: 1_862 * MIB,
        };
        assert!(ensure_unified_pass_above_floor(&ok_gpu, 1_350 * MIB).is_ok());
        assert!(
            ensure_retained_pass_above_floor(&ok_gpu, "full", 1_350 * MIB, 2 * 1024 * MIB).is_ok()
        );

        // One byte under discrete VRAM margin refuses with discrete VRAM message.
        let tight_gpu = StubDiscreteGpu {
            free_bytes: 1_862 * MIB - 1,
        };
        let err = ensure_unified_pass_above_floor(&tight_gpu, 1_350 * MIB)
            .unwrap_err()
            .to_string();
        assert!(err.contains("free VRAM"), "{err}");
        assert!(err.contains("512 MiB discrete VRAM margin"), "{err}");

        // Pure helper: override permits tight or missing VRAM telemetry; overflow fails closed.
        assert!(
            discrete_transpose_vram_decision(
                "full",
                Some(1_000 * MIB),
                1_350 * MIB,
                512 * MIB,
                true
            )
            .is_ok()
        );
        assert!(discrete_transpose_vram_decision("full", None, MIB, 512 * MIB, false).is_err());
        assert!(discrete_transpose_vram_decision("full", None, MIB, 512 * MIB, true).is_ok());
        assert!(
            discrete_transpose_vram_decision("full", Some(usize::MAX), usize::MAX, 1, false)
                .is_err()
        );
    }
}
