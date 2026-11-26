// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;
use spark_model::model::ssm_pool_geometry::{
    SsmPoolGeometryInput, checked_ssm_pool_geometry, checked_ssm_speculative_geometry,
};

use crate::cli;
use spark_runtime::startup_ledger::{self, MIB};

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) prefill_budget: usize,
    pub(crate) max_batch_tokens: usize,
    pub(crate) breakdown: ReserveBreakdown,
}

/// The per-pool bytes behind `inference_reserve`, kept for the startup
/// budget ledger so every reserve term is a named row with its requested
/// size, not one opaque sum.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReserveBreakdown {
    pub(crate) ssm_pool_bytes: usize,
    pub(crate) ssm_snapshot_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ple_history_bytes: usize,
    pub(crate) cuda_headroom_bytes: usize,
    pub(crate) qsa_side_cache_bytes: usize,
    pub(crate) qsa_fp8_shadow_bytes: usize,
    pub(crate) batched_k5_lane_bytes: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SsmPreflightCapacity {
    ssm_pool_bytes: usize,
    ssm_snapshot_bytes: usize,
}

fn checked_preflight_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("SSM preflight capacity overflow: {label} ({left} * {right})"))
}

fn checked_preflight_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .with_context(|| format!("SSM preflight capacity overflow: {label} ({left} + {right})"))
}

fn checked_gdn_two_phase_bytes(config: &ModelConfig, rows: usize) -> Result<usize> {
    if config.num_ssm_layers() == 0 {
        return Ok(0);
    }
    let key_dim = checked_preflight_mul(
        config.linear_num_key_heads,
        config.linear_key_head_dim,
        "GDN key width",
    )?;
    let value_dim = checked_preflight_mul(
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        "GDN value width",
    )?;
    let conv_dim = checked_preflight_add(
        checked_preflight_mul(key_dim, 2, "GDN QK width")?,
        value_dim,
        "GDN conv width",
    )?;
    if conv_dim == 0 {
        return Ok(0);
    }
    let conv = checked_preflight_mul(conv_dim, 2, "GDN conv row bytes")?;
    let gates = checked_preflight_mul(config.linear_num_value_heads, 8, "GDN gates row bytes")?;
    let value = checked_preflight_mul(value_dim, 2, "GDN value row bytes")?;
    let row_bytes = checked_preflight_add(
        checked_preflight_add(conv, gates, "GDN conv + gates")?,
        checked_preflight_mul(value, 2, "GDN two value buffers")?,
        "GDN total row bytes",
    )?;
    checked_preflight_mul(rows, row_bytes, "GDN two-phase total bytes")
}

fn checked_ssm_preflight_capacity(
    ssm_pool_bytes: usize,
    max_batch_size: usize,
    num_ssm_layers: usize,
    h_bytes: usize,
    conv_bytes: usize,
    ssm_cache_slots: usize,
    decode_ring_slots: usize,
) -> Result<SsmPreflightCapacity> {
    if num_ssm_layers == 0 {
        return Ok(SsmPreflightCapacity::default());
    }
    let snapshot_enabled = ssm_cache_slots > 0 || (max_batch_size > 0 && decode_ring_slots > 0);
    let ssm_snapshot_bytes = if snapshot_enabled {
        if h_bytes == 0 {
            anyhow::bail!("SSM preflight capacity: enabled snapshot h bytes must be positive");
        }
        if conv_bytes == 0 {
            anyhow::bail!("SSM preflight capacity: enabled snapshot conv bytes must be positive");
        }
        let bytes_per_layer = checked_preflight_add(h_bytes, conv_bytes, "SSM bytes per layer")?;
        let decode_region =
            checked_preflight_mul(max_batch_size, decode_ring_slots, "decode region slots")?;
        let snapshot_slots =
            checked_preflight_add(ssm_cache_slots, decode_region, "snapshot slots")?;
        let layer_bytes =
            checked_preflight_mul(num_ssm_layers, bytes_per_layer, "snapshot layer bytes")?;
        checked_preflight_mul(snapshot_slots, layer_bytes, "snapshot total bytes")?
    } else {
        0
    };

    Ok(SsmPreflightCapacity {
        ssm_pool_bytes,
        ssm_snapshot_bytes,
    })
}

fn checked_reserve_totals(
    ssm_pool_bytes: usize,
    ssm_snapshot_bytes: usize,
    gdn_two_phase_bytes: usize,
    ple_allocation_bytes: usize,
    cuda_headroom: usize,
    buffer_arena_bytes: usize,
) -> Result<(usize, usize)> {
    let ssm_total = checked_preflight_add(
        ssm_pool_bytes,
        ssm_snapshot_bytes,
        "reserve SSM pool + snapshot",
    )?;
    let with_gdn = checked_preflight_add(ssm_total, gdn_two_phase_bytes, "reserve + GDN")?;
    let with_ple = checked_preflight_add(with_gdn, ple_allocation_bytes, "reserve + PLE")?;
    let inference_reserve =
        checked_preflight_add(with_ple, cuda_headroom, "reserve + CUDA headroom")?;
    let total_reserve = checked_preflight_add(
        inference_reserve,
        buffer_arena_bytes,
        "reserve + buffer arena",
    )?;
    Ok((inference_reserve, total_reserve))
}

const DISCRETE_SMALL_VRAM_THRESHOLD_MB: usize = 48 * 1024;
const DISCRETE_RESERVE_MIN_MB: usize = 512;

/// Cap default-sized memory reserves (`oom_guard_mb` / `speculative_cuda_headroom_mb`)
/// on discrete GPUs with `<= 48 GiB` VRAM to `min(configured, max(512, total_vram_mb / 8))`
/// so 16–32 GiB cards are not starved of KV cache.
///
/// On UMA devices (GB10) and large discrete GPUs (`> 48 GiB`), returns `configured_mb`
/// unchanged. Explicit smaller operator overrides are never raised.
pub(crate) fn effective_discrete_reserve_mb(
    configured_mb: usize,
    total_vram_bytes: Option<usize>,
    is_discrete: bool,
) -> usize {
    if !is_discrete {
        return configured_mb;
    }
    let Some(total_bytes) = total_vram_bytes.filter(|&bytes| bytes > 0) else {
        return configured_mb;
    };
    let total_vram_mb = total_bytes / MIB;
    if total_vram_mb > DISCRETE_SMALL_VRAM_THRESHOLD_MB {
        return configured_mb;
    }
    let cap_mb = (total_vram_mb / 8).max(DISCRETE_RESERVE_MIN_MB);
    configured_mb.min(cap_mb)
}

fn speculative_cuda_headroom(
    speculative_drafts: Option<usize>,
    configured_headroom_mb: usize,
) -> usize {
    match speculative_drafts {
        Some(_) => configured_headroom_mb.saturating_mul(1024 * 1024),
        None => 512 * 1024 * 1024,
    }
}

fn qwen4_qsa_physical_blocks(args: &cli::ServeArgs) -> Result<usize> {
    anyhow::ensure!(
        args.block_size >= 4 && args.block_size.is_multiple_of(4),
        "Qwen4 QSA requires --block-size divisible by 4"
    );
    let physical_blocks = match args.kv_pool_tokens {
        // The side cache follows the physical pool, so a shared pool
        // (`--kv-pool-tokens`) reserves for the pool, not every slot.
        Some(tokens) => spark_model::factory::kv_pool::requested_pool_blocks_with_prefix_cache(
            tokens,
            args.max_seq_len,
            args.max_batch_size,
            args.block_size,
            args.enable_prefix_caching,
        ),
        None => {
            let blocks_per_seq = args.max_seq_len.div_ceil(args.block_size);
            let blocks_with_spare = blocks_per_seq
                .checked_add(1)
                .context("QSA blocks-per-sequence reserve overflow")?;
            args.max_batch_size
                .checked_mul(blocks_with_spare)
                .and_then(|n| n.checked_add(1))
                .context("QSA physical block reserve overflow")?
        }
    };
    anyhow::ensure!(
        u32::try_from(physical_blocks).is_ok(),
        "QSA physical block reserve {physical_blocks} exceeds the paged-KV block-id ABI"
    );
    Ok(physical_blocks)
}

fn qwen4_qsa_side_cache_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> Result<usize> {
    if !config.is_qwen4_exp() || !config.qwen4_qsa {
        return Ok(0);
    }
    let physical_blocks = qwen4_qsa_physical_blocks(args)?;
    let bytes_per_layer =
        spark_model::layers::qwen4_qsa::checked_side_cache_bytes(physical_blocks, args.block_size)?;
    bytes_per_layer
        .checked_mul(config.num_attention_layers())
        .context("QSA side-cache layer reserve overflow")
}

/// Whether QSA E4M3 prefill shadows (`ATLAS_QWEN4_PREFILL_QSA_FP8`) are
/// eligible for preflight byte reservation.
///
/// E4M3 shadows materialize exclusively over all-BF16 KV caches; NVFP4 / FP8 KV
/// cannot host E4M3 shadows (`prefill_fp8::validate_target_shadow_geometry`)
/// and must never reserve shadow VRAM even if an outer environment inherited
/// `ATLAS_QWEN4_PREFILL_QSA_FP8=qkpv`.
#[must_use]
fn qwen4_qsa_fp8_shadow_eligible(config: &ModelConfig, kv_cache_dtype: &str) -> bool {
    config.is_qwen4_exp() && config.qwen4_qsa && kv_cache_dtype.trim().eq_ignore_ascii_case("bf16")
}

fn qwen4_qsa_fp8_shadow_reserve_for_mode(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    fp8_mode: Option<spark_model::layers::qwen4_qsa::prefill_fp8::Mode>,
) -> Result<usize> {
    let Some(mode) = fp8_mode else {
        return Ok(0);
    };
    if !config.is_qwen4_exp() || !config.qwen4_qsa {
        return Ok(0);
    }
    let physical_blocks = qwen4_qsa_physical_blocks(args)?;
    spark_model::layers::qwen4_qsa::prefill_fp8::checked_shadow_reserve_bytes_for_mode(
        mode,
        args.block_size,
        physical_blocks,
        config.num_attention_layers(),
    )
}

#[cfg(test)]
fn qwen4_qsa_fp8_shadow_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    fp8_selected: bool,
) -> Result<usize> {
    let Some(mode) =
        fp8_selected.then_some(spark_model::layers::qwen4_qsa::prefill_fp8::Mode::QkPv)
    else {
        return Ok(0);
    };
    if !config.is_qwen4_exp() || !config.qwen4_qsa {
        return Ok(0);
    }
    let physical_blocks = qwen4_qsa_physical_blocks(args)?;
    spark_model::layers::qwen4_qsa::prefill_fp8::checked_shadow_reserve_bytes_for_layout(
        mode,
        spark_model::layers::qwen4_qsa::prefill_fp8::V8Layout::PerLayer,
        args.block_size,
        physical_blocks,
        config.num_attention_layers(),
    )
}

fn qwen4_ple_allocation_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> Result<usize> {
    if !config.is_qwen4_exp() {
        return Ok(0);
    }
    Ok(
        spark_model::layers::qwen4_ple::checked_ple_allocation_bytes(
            args.max_batch_size,
            config.residual_width(),
        )?
        .total,
    )
}

/// Extra device bytes for ATLAS_QWEN4_BATCHED_K5 verify lanes: QSA undo
/// journals and GDN lazy-commit retain buffers owned per physical slot.
fn qwen4_batched_k5_lane_reserve(args: &cli::ServeArgs, config: &ModelConfig) -> Result<usize> {
    if !config.is_qwen4_exp() {
        return Ok(0);
    }
    let batched = spark_model::layers::qwen4_verify_lanes::batched_k5_selected()?;
    let batched_spec =
        (args.speculative || args.dflash) && !args.self_speculative && !args.ngram_speculative;
    let lanes = spark_model::layers::qwen4_verify_lanes::lanes_for(
        batched && batched_spec,
        args.max_batch_size,
    )?;
    let qsa_lane_bytes = if config.qwen4_qsa {
        spark_model::layers::qwen4_qsa::verify_journal_lane_bytes()?
    } else {
        0
    };
    let gdn_lane_bytes = if spark_model::layers::qwen3_ssm::gdn_seq_lazy_enabled() {
        Some(
            spark_model::layers::qwen3_ssm::gdn_lazy_retain_effective_lane_bytes(
                config.ssm_qkvz_size(),
                config.linear_num_value_heads,
            )?,
        )
    } else {
        None
    };
    let reserve = spark_model::layers::qwen4_verify_lanes::checked_lane_reserve(
        lanes,
        config.num_attention_layers(),
        qsa_lane_bytes,
        config.num_ssm_layers(),
        gdn_lane_bytes,
    )?;
    // The frame's row staging is allocated once at model construction
    // whenever lanes are configured.
    let staging_bytes = if lanes > 1 {
        spark_model::layer::Qwen4CrossSeqStaging::total_bytes_for(config)?
    } else {
        0
    };
    let total = checked_preflight_add(reserve.total()?, staging_bytes, "batched-K5 staging")?;
    if total > 0 {
        tracing::info!(
            lanes,
            qsa_journal_mb = reserve.qsa_journal_bytes as f64 / (1024.0 * 1024.0),
            gdn_retain_mb = reserve.gdn_retain_bytes as f64 / (1024.0 * 1024.0),
            staging_mb = staging_bytes as f64 / (1024.0 * 1024.0),
            "Qwen4 batched-K5 verify lane reserve"
        );
    }
    Ok(total)
}

/// Pure checked estimate of per-sequence `DflashProposerState` device buffers
/// (`ctx_hidden_acc` ring + `ctx_fc_cache` + per-layer K/V caches) when serving
/// concurrent DFlash2 (`dflash && max_batch_size > 1`).
///
/// At a 4096-token resident window on Qwen4 (`hidden_size = 2560`, 8 taps),
/// each active sequence allocates ~200 MiB (`160 MiB` hidden ring + `20 MiB`
/// FC cache + per-layer KV caches); smaller windows scale proportionally.
fn dflash_batched_proposer_state_bytes(
    dflash: bool,
    max_batch_size: usize,
    dflash_window_size: usize,
    max_seq_len: usize,
) -> Result<usize> {
    if !dflash || max_batch_size <= 1 {
        return Ok(0);
    }
    const DEFAULT_WINDOW: usize = 4096;
    const BYTES_PER_SEQ_AT_4096: usize = 200 * MIB;
    let effective_window = if dflash_window_size == 0 {
        max_seq_len.min(DEFAULT_WINDOW)
    } else {
        dflash_window_size.min(DEFAULT_WINDOW)
    }
    .max(1);
    let per_seq_bytes = checked_preflight_mul(
        BYTES_PER_SEQ_AT_4096,
        effective_window,
        "DFlash per-seq proposer state window",
    )?
    .div_ceil(DEFAULT_WINDOW);
    checked_preflight_mul(
        max_batch_size,
        per_seq_bytes,
        "DFlash batched proposer state reserve",
    )
}

fn dflash_batched_proposer_state_reserve(args: &cli::ServeArgs) -> Result<usize> {
    dflash_batched_proposer_state_bytes(
        args.dflash,
        args.max_batch_size,
        args.dflash_window_size,
        args.max_seq_len,
    )
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    let num_ssm_layers = config.num_ssm_layers();
    let speculative_drafts = super::speculative_draft_count(args);
    let has_mtp = speculative_drafts.is_some();
    let num_drafts = speculative_drafts.unwrap_or(0);
    let requested_ddtree_capacity = std::env::var("ATLAS_DDTREE_MAX_NODES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    let speculative_geometry = checked_ssm_speculative_geometry(
        has_mtp,
        args.dflash,
        num_drafts,
        requested_ddtree_capacity,
    )?;
    let qsa_side_cache_bytes = qwen4_qsa_side_cache_reserve(args, config)?;
    let qsa_fp8_mode = if qwen4_qsa_fp8_shadow_eligible(config, &args.kv_cache_dtype) {
        spark_model::layers::qwen4_qsa::prefill_fp8::mode()?
    } else {
        None
    };
    let qsa_fp8_shadow_bytes = qwen4_qsa_fp8_shadow_reserve_for_mode(args, config, qsa_fp8_mode)?;
    let qsa_total_bytes = checked_preflight_add(
        qsa_side_cache_bytes,
        qsa_fp8_shadow_bytes,
        "QSA side cache + FP8 shadow",
    )?;
    let batched_k5_lane_bytes = qwen4_batched_k5_lane_reserve(args, config)?;
    let qsa_total_bytes = checked_preflight_add(
        qsa_total_bytes,
        batched_k5_lane_bytes,
        "QSA reserves + batched-K5 verify lanes",
    )?;
    let dflash_proposer_state_bytes = dflash_batched_proposer_state_reserve(args)?;
    let ple_allocation_bytes = qwen4_ple_allocation_reserve(args, config)?;
    let (is_discrete, total_vram_bytes) = match atlas_core::device::active_device_props() {
        Some(props) => (!props.is_integrated, Some(props.total_mem_bytes)),
        None => (startup_ledger::is_discrete_device(), None),
    };
    let effective_speculative_headroom_mb = effective_discrete_reserve_mb(
        args.speculative_cuda_headroom_mb,
        total_vram_bytes,
        is_discrete,
    );
    if speculative_drafts.is_some()
        && effective_speculative_headroom_mb != args.speculative_cuda_headroom_mb
    {
        tracing::info!(
            "Speculative CUDA headroom capped on discrete GPU: {} MB (configured: {} MB)",
            effective_speculative_headroom_mb,
            args.speculative_cuda_headroom_mb,
        );
    }
    let base_cuda_headroom = checked_preflight_add(
        speculative_cuda_headroom(speculative_drafts, effective_speculative_headroom_mb),
        dflash_proposer_state_bytes,
        "CUDA headroom + DFlash batched proposer state",
    )?;
    let cuda_headroom = checked_preflight_add(
        base_cuda_headroom,
        qsa_total_bytes,
        "CUDA headroom + QSA reserves",
    )?;
    let decode_ring_slots = if num_ssm_layers > 0 {
        (atlas_kernels::ROLLBACK_RESTEER_CAP as usize)
            .checked_add(1)
            .context("SSM preflight decode ring slot count overflow")?
    } else {
        0
    };
    let decode_ring_slots =
        if spark_model::model::ssm_pool_geometry::ssm_decode_ring_enabled_for_model(
            config.is_qwen4_exp(),
            args.max_batch_size,
        )? {
            decode_ring_slots
        } else {
            0
        };
    let qwen4_compact_spec = (args.speculative
        || (args.dflash
            && speculative_geometry.num_intermediates <= 10
            && std::env::var("ATLAS_QWEN4_HYBRID_ALL_K").ok().as_deref() == Some("1")))
        && !args.self_speculative
        && !args.ngram_speculative;
    let pool_geometry = checked_ssm_pool_geometry(
        SsmPoolGeometryInput::from_config(
            config,
            args.max_batch_size,
            has_mtp,
            speculative_geometry.num_intermediates,
            spark_model::layers::wy17_lazy_commit(),
        )?
        .with_qwen4_native_mtp(config, qwen4_compact_spec),
    )?;
    let ssm_capacity = checked_ssm_preflight_capacity(
        pool_geometry.total_bytes,
        args.max_batch_size,
        num_ssm_layers,
        pool_geometry.h_bytes,
        pool_geometry.conv_bytes,
        args.ssm_cache_slots,
        decode_ring_slots,
    )?;
    let ssm_pool_bytes = ssm_capacity.ssm_pool_bytes;
    let ssm_snapshot_bytes = ssm_capacity.ssm_snapshot_bytes;
    let ssm_prefill_chunk: usize = if num_ssm_layers > 0 {
        args.max_seq_len.min(8192)
    } else {
        0
    };
    // Use the same HSS clamp, additive decode-row allowance, speculative
    // width, and ATLAS_MAX_BATCH_TOKENS override as the runtime allocation.
    // A smaller preflight estimate can pass startup and OOM during the actual
    // arena allocation, especially at C128.
    let resolved_prefill = super::resolve_prefill_budget(args, ssm_prefill_chunk)?;
    super::validate_qwen4_prefill_geometry(config, resolved_prefill.prefill_budget)?;
    let prefill_budget = resolved_prefill.prefill_budget;
    let max_batch_tokens = resolved_prefill.max_batch_tokens;
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens,
        args.max_seq_len,
        args.block_size,
        args.max_batch_size,
    )?
    .total_bytes()?;
    let gdn_two_phase_bytes: usize = checked_gdn_two_phase_bytes(config, max_batch_tokens)?;
    let (inference_reserve, total_reserve) = checked_reserve_totals(
        ssm_pool_bytes,
        ssm_snapshot_bytes,
        gdn_two_phase_bytes,
        ple_allocation_bytes,
        cuda_headroom,
        buffer_arena_bytes,
    )?;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_snapshot_bytes + ple_allocation_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = checked_gdn_two_phase_bytes(config, 1)?;
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}; \
             PLE history scales with --max-batch-size.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    if qsa_side_cache_bytes > 0 {
        tracing::info!(
            "Qwen4 QSA side-cache reserve: {} MB for max_seq_len={} batch={}",
            qsa_side_cache_bytes / (1024 * 1024),
            args.max_seq_len,
            args.max_batch_size,
        );
    }
    if qsa_fp8_shadow_bytes > 0 {
        let v8_layout = spark_model::layers::qwen4_qsa::prefill_fp8::v8_layout()?;
        let total_planes = qsa_fp8_mode
            .map(|mode| {
                spark_model::layers::qwen4_qsa::prefill_fp8::checked_total_shadow_planes(
                    mode,
                    v8_layout,
                    config.num_attention_layers(),
                )
            })
            .transpose()?
            .unwrap_or(0);
        tracing::info!(
            "Qwen4 QSA FP8 shadow reserve: {} bytes ({} MB) for max_seq_len={} batch={} layers={} planes={total_planes} v8_layout={v8_layout:?}",
            qsa_fp8_shadow_bytes,
            qsa_fp8_shadow_bytes / (1024 * 1024),
            args.max_seq_len,
            args.max_batch_size,
            config.num_attention_layers(),
        );
    }
    if ple_allocation_bytes > 0 {
        tracing::info!(
            "Qwen4 PLE history reserve: {} bytes ({} MB) for batch={} (live + checkpoint + intermediate histories)",
            ple_allocation_bytes,
            ple_allocation_bytes / (1024 * 1024),
            args.max_batch_size,
        );
    }
    if dflash_proposer_state_bytes > 0 {
        tracing::info!(
            "DFlash batched proposer state reserve: {} bytes ({} MB) for batch={} window={}",
            dflash_proposer_state_bytes,
            dflash_proposer_state_bytes / MIB,
            args.max_batch_size,
            args.dflash_window_size,
        );
    }
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = speculative_drafts.is_some();
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({} dummy-inclusive slots × {} layers × {} state copies), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         ple_history={} MB ({} slots), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        pool_geometry.total_slots,
        config.num_ssm_layers(),
        pool_geometry.state_copies,
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens,
        ple_allocation_bytes / (1024 * 1024),
        args.max_batch_size,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        speculative_drafts.map_or(-1, |drafts| drafts as i64),
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        prefill_budget,
        max_batch_tokens,
        breakdown: ReserveBreakdown {
            ssm_pool_bytes,
            ssm_snapshot_bytes,
            gdn_two_phase_bytes,
            ple_history_bytes: ple_allocation_bytes,
            // `cuda_headroom` already folds the QSA + batched-K5 reserves in;
            // report the speculative/CUDA headroom on its own so no term appears twice.
            cuda_headroom_bytes: cuda_headroom.saturating_sub(qsa_total_bytes),
            qsa_side_cache_bytes,
            qsa_fp8_shadow_bytes,
            batched_k5_lane_bytes,
        },
    })
}

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `atlas_kernels::metallib_modules()`. The `ptx_set` argument is
///   accepted (for ABI symmetry with the cuda variant) but ignored;
///   metal kernels live in a parallel registry.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(std::sync::Arc<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // Allocation receipt for backend init (context, kernel modules); no ledger row.
    let _receipt = spark_runtime::startup_ledger::scope_guard("gpu-init");
    let gpu: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> = std::sync::Arc::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    _ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(std::sync::Arc<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let modules = atlas_kernels::metallib_modules();
    let gpu: std::sync::Arc<dyn spark_runtime::gpu::GpuBackend> = std::sync::Arc::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

/// Bytes per KV block across every attention layer (K + V) at 2 bytes per
/// element (BF16 upper bound). MLA-absorbed caches store one latent head
/// (`build.rs` "MLA absorbed KV cache").
pub(crate) fn kv_block_bytes_all_layers(config: &ModelConfig, block_size: usize) -> Result<usize> {
    kv_block_bytes_all_layers_for_dtype(config, block_size, "bf16")
}

/// Bytes per KV block for one attention layer (K + V) at `dtype`.
fn kv_block_bytes_per_layer(
    config: &ModelConfig,
    block_size: usize,
    dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> Result<usize> {
    use spark_runtime::kv_cache::KvCacheDtype;
    /// Elements per scale group in NVFP4 and TurboQuant KV layouts
    /// (matches `spark_runtime::kv_cache::NVFP4_GROUP_SIZE`).
    const KV_QUANT_GROUP_SIZE: usize = 16;

    let (heads, dim) = if config.kv_lora_rank > 0 {
        (1, config.kv_lora_rank + config.qk_rope_head_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let elems_per_side = heads
        .checked_mul(dim)
        .and_then(|b| b.checked_mul(block_size))
        .context("KV block element count overflow")?;
    let side_bytes = |dt: KvCacheDtype| -> Result<usize> {
        match dt {
            KvCacheDtype::Bf16
            | KvCacheDtype::Bf16KTurbo4V
            | KvCacheDtype::Bf16KTurbo3V
            | KvCacheDtype::Bf16KTurbo2V => elems_per_side
                .checked_mul(2)
                .context("KV block byte overflow"),
            KvCacheDtype::Fp8
            | KvCacheDtype::Fp8KTurbo4V
            | KvCacheDtype::Fp8KTurbo3V
            | KvCacheDtype::Fp8KTurbo2V => Ok(elems_per_side),
            KvCacheDtype::Nvfp4
            | KvCacheDtype::Turbo4
            | KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V => (elems_per_side / 2)
                .checked_add(elems_per_side / KV_QUANT_GROUP_SIZE)
                .context("KV block byte overflow"),
            KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V => elems_per_side
                .checked_mul(3)
                .map(|n| n / 8)
                .and_then(|d| d.checked_add(elems_per_side / KV_QUANT_GROUP_SIZE))
                .context("KV block byte overflow"),
            KvCacheDtype::Turbo2 => (elems_per_side / 4)
                .checked_add(elems_per_side / KV_QUANT_GROUP_SIZE)
                .context("KV block byte overflow"),
            KvCacheDtype::Turbo8 => (elems_per_side / KV_QUANT_GROUP_SIZE)
                .checked_mul(2)
                .and_then(|s| elems_per_side.checked_add(s))
                .context("KV block byte overflow"),
        }
    };
    let (k_dt, v_dt) = dtype.kv_pair();
    side_bytes(k_dt)?
        .checked_add(side_bytes(v_dt)?)
        .context("KV block K+V byte overflow")
}

/// Bytes per KV block across every attention layer (K + V) for the requested
/// `--kv-cache-dtype` (`bf16`, `fp8`, `nvfp4`, `turbo*`, or asymmetric
/// `*_turbo*v`), matching [`spark_runtime::kv_cache::KvCacheConfig::block_bytes_kv_all_layers`].
/// Unrecognized dtype strings fall back conservatively to BF16 (`2 B/elem`).
pub(crate) fn kv_block_bytes_all_layers_for_dtype(
    config: &ModelConfig,
    block_size: usize,
    kv_cache_dtype: &str,
) -> Result<usize> {
    let dtype = kv_cache_dtype
        .trim()
        .to_ascii_lowercase()
        .parse::<spark_runtime::kv_cache::KvCacheDtype>()
        .unwrap_or(spark_runtime::kv_cache::KvCacheDtype::Bf16);
    kv_block_bytes_per_layer(config, block_size, dtype)?
        .checked_mul(config.num_attention_layers())
        .context("KV block byte overflow")
}

/// Bytes per KV block across every attention layer (K + V) honoring both
/// `--kv-cache-dtype` and `--kv-high-precision-layers` /
/// `--kv-high-precision-layer-set` (matching `serve_phases::resolve_kv_cache_config`
/// and `kv_dtypes`).
pub(crate) fn kv_block_bytes_all_layers_with_hp(
    config: &ModelConfig,
    block_size: usize,
    kv_cache_dtype: &str,
    kv_high_precision_layers: &str,
    kv_high_precision_layer_set: &str,
) -> Result<usize> {
    use spark_runtime::kv_cache::KvCacheDtype;

    let num_attn_layers = config.num_attention_layers();
    if num_attn_layers == 0 {
        return Ok(0);
    }
    let base_dtype = kv_cache_dtype
        .trim()
        .to_ascii_lowercase()
        .parse::<KvCacheDtype>()
        .unwrap_or(KvCacheDtype::Bf16);
    let bf16_per_layer = kv_block_bytes_per_layer(config, block_size, KvCacheDtype::Bf16)?;
    if base_dtype == KvCacheDtype::Bf16 {
        return bf16_per_layer
            .checked_mul(num_attn_layers)
            .context("KV block byte overflow");
    }

    let hp_layer_set: std::collections::BTreeSet<usize> = kv_high_precision_layer_set
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&idx| idx < num_attn_layers)
        .collect();
    let bf16_layers = if !hp_layer_set.is_empty() {
        hp_layer_set.len()
    } else {
        let mut hp_each_end: usize = match kv_high_precision_layers
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "max" | "all" => num_attn_layers,
            "auto" => 2,
            s => s.parse().unwrap_or(0),
        };
        if hp_each_end == 0
            && matches!(
                base_dtype,
                KvCacheDtype::Turbo3
                    | KvCacheDtype::Turbo4
                    | KvCacheDtype::Turbo8
                    | KvCacheDtype::Turbo4KTurbo3V
                    | KvCacheDtype::Turbo4KTurbo8V
                    | KvCacheDtype::Turbo3KTurbo8V
                    | KvCacheDtype::Bf16KTurbo4V
                    | KvCacheDtype::Bf16KTurbo3V
                    | KvCacheDtype::Fp8KTurbo4V
                    | KvCacheDtype::Fp8KTurbo3V
                    | KvCacheDtype::Bf16KTurbo2V
                    | KvCacheDtype::Fp8KTurbo2V
            )
        {
            hp_each_end = ((num_attn_layers as f32 / 3.0).ceil() as usize).max(2);
        }
        hp_each_end.saturating_mul(2).min(num_attn_layers)
    };
    let quant_layers = num_attn_layers.saturating_sub(bf16_layers);
    let quant_per_layer = kv_block_bytes_per_layer(config, block_size, base_dtype)?;
    let bf16_total = bf16_per_layer
        .checked_mul(bf16_layers)
        .context("KV high-precision layer byte overflow")?;
    let quant_total = quant_per_layer
        .checked_mul(quant_layers)
        .context("KV quantized layer byte overflow")?;
    bf16_total
        .checked_add(quant_total)
        .context("KV mixed-precision block byte overflow")
}

/// Bytes per KV block across every attention layer for the active CLI flags.
pub(crate) fn kv_block_bytes_all_layers_for_args(
    config: &ModelConfig,
    args: &cli::ServeArgs,
) -> Result<usize> {
    kv_block_bytes_all_layers_with_hp(
        config,
        args.block_size,
        &args.kv_cache_dtype,
        &args.kv_high_precision_layers,
        &args.kv_high_precision_layer_set,
    )
}

/// The addressable cap the factory applies to the KV pool: `batch x
/// (blocks/seq + 1 spare) + 1 dummy` blocks (`build.rs` "KV cache capped to
/// addressable size"). An upper bound only: at batch 8 / 135168 it is ~25 GiB
/// and at C128 / 262144 ~800 GiB, far above what the factory allocates.
pub(crate) fn kv_cache_addressable_bytes(
    max_batch_size: usize,
    max_seq_len: usize,
    block_size: usize,
    block_bytes_all_layers: usize,
) -> Result<usize> {
    let blocks_per_seq = max_seq_len.div_ceil(block_size.max(1));
    max_batch_size
        .checked_mul(
            blocks_per_seq
                .checked_add(1)
                .context("KV blocks/seq overflow")?,
        )
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(block_bytes_all_layers))
        .context("KV pool byte overflow")
}

/// What the factory will actually allocate: `min(addressable, budget)` with
/// `budget = (free - inference_reserve) x gpu_memory_utilization`
/// (`build.rs` "Size KV cache from actual free memory"). Mirrors the factory
/// so a projection never refuses a multi-sequence recipe the factory serves
/// by capping the pool to its budget.
pub(crate) fn kv_cache_projected_bytes(
    addressable_bytes: usize,
    mem_available: usize,
    inference_reserve: usize,
    gpu_memory_utilization: f64,
) -> usize {
    let allocatable = mem_available.saturating_sub(inference_reserve);
    let budget = (allocatable as f64 * gpu_memory_utilization.clamp(0.0, 1.0)) as usize;
    addressable_bytes.min(budget)
}

/// Bytes the Qwen4 builder allocates AFTER the target layers and BEFORE the
/// arena/KV sizing: the conservative sum of both routed transposed-expert
/// slabs, including their two 256-byte front guards. Down is allocated only
/// after gate/up originals are freed; this estimate takes no reclaim credit.
/// Also includes the native-MTP layer net of its own
/// transpose (r8 `QWEN4_MTP_UNIFIED_T_ENGAGED net_alloc_mib=1033` + 0.26 GiB
/// MTP KV, rounded up to 1.2 GiB) and the NVFP4 LM head (vocab x hidden x
/// 9/16). Zero for every other family.
pub(crate) fn qwen4_construction_transient_bytes(
    config: &ModelConfig,
    native_mtp: bool,
) -> Result<usize> {
    if !config.is_qwen4_exp() {
        return Ok(0);
    }
    const ROUTED_TRANSPOSED_SLAB_BYTES: usize = 1_415_578_112;
    const NATIVE_MTP_LAYER_NET_BYTES: usize = 1_200 * MIB;
    let lm_head_nvfp4 = config
        .vocab_size
        .checked_mul(config.hidden_size)
        .and_then(|bytes| bytes.checked_mul(9))
        .map(|bytes| bytes / 16)
        .context("Qwen4 NVFP4 LM head byte estimate overflow")?;
    let mtp = if native_mtp {
        NATIVE_MTP_LAYER_NET_BYTES
    } else {
        0
    };
    let shard_divisor = config
        .ep_world_size
        .max(1)
        .saturating_mul(config.pp_world_size.max(1));
    let routed_slab = ROUTED_TRANSPOSED_SLAB_BYTES / shard_divisor;
    routed_slab
        .checked_add(mtp)
        .and_then(|bytes| bytes.checked_add(lm_head_nvfp4))
        .context("Qwen4 construction transient overflow")
}

/// Partial pre-construction projection against real `MemAvailable`. Eager QSA
/// shadows use their addressable upper-bound reserve; additional prefill
/// scratch and lazy request caches are not modeled. This is not a peak bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartupProjection {
    pub(crate) mem_available_bytes: usize,
    pub(crate) target_construction_bytes: usize,
    pub(crate) qwen4_transient_bytes: usize,
    pub(crate) pools_bytes: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) kv_projected_bytes: usize,
    pub(crate) qsa_fp8_shadow_bytes: usize,
    pub(crate) floor_bytes: usize,
    pub(crate) is_discrete: bool,
}

impl StartupProjection {
    /// Sum of the modeled allocations, not the complete startup peak.
    pub(crate) fn need_bytes(&self) -> usize {
        self.target_construction_bytes
            .saturating_add(self.qwen4_transient_bytes)
            .saturating_add(self.pools_bytes)
            .saturating_add(self.buffer_arena_bytes)
            .saturating_add(self.kv_projected_bytes)
            .saturating_add(self.qsa_fp8_shadow_bytes)
    }

    /// Remaining memory after modeled allocations. Reclaims from unified MoE
    /// transpose and the BF16 lm_head are omitted, as are additional scratch
    /// allocations, so this value is neither a lower nor an upper bound. The
    /// end-of-load ledger, not this projection, enforces the measured floor.
    pub(crate) fn projected_post_start_bytes(&self) -> usize {
        self.mem_available_bytes.saturating_sub(self.need_bytes())
    }

    pub(crate) fn summary(&self) -> String {
        let avail_label = if self.is_discrete {
            "GPU VRAM"
        } else {
            "MemAvailable"
        };
        format!(
            "target construction {} + qwen4 transient {} + pools {} + arena {} + KV (factory-capped) {} + eager QSA shadows (addressable upper bound) {} = modeled need {} MiB; projected post-start {avail_label} {} MiB vs floor {} MiB (partial estimate, not a peak bound: excludes additional prefill scratch, lazy request caches, and transpose/lm_head reclaim)",
            self.target_construction_bytes / MIB,
            self.qwen4_transient_bytes / MIB,
            self.pools_bytes / MIB,
            self.buffer_arena_bytes / MIB,
            self.kv_projected_bytes / MIB,
            self.qsa_fp8_shadow_bytes / MIB,
            self.need_bytes() / MIB,
            self.projected_post_start_bytes() / MIB,
            self.floor_bytes / MIB,
        )
    }
}

/// Build the projection from the CLI, config, and pre-load preflight terms
/// against an explicit available-memory reading and floor (host `MemAvailable`
/// on UMA, GPU VRAM `available_free` on discrete GPUs).
#[allow(clippy::too_many_arguments)]
pub(crate) fn startup_projection(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    breakdown: &ReserveBreakdown,
    buffer_arena_bytes: usize,
    inference_reserve: usize,
    available_bytes: usize,
    floor_bytes: usize,
    is_discrete: bool,
) -> Result<StartupProjection> {
    let block_bytes = kv_block_bytes_all_layers_for_args(config, args)?;
    let addressable = match args.kv_pool_tokens {
        Some(tokens) => spark_model::factory::kv_pool::requested_pool_blocks_with_prefix_cache(
            tokens,
            args.max_seq_len,
            args.max_batch_size,
            args.block_size,
            args.enable_prefix_caching,
        )
        .checked_mul(block_bytes)
        .context("KV pool byte overflow")?,
        None => kv_cache_addressable_bytes(
            args.max_batch_size,
            args.max_seq_len,
            args.block_size,
            block_bytes,
        )?,
    };
    let kv_projected = kv_cache_projected_bytes(
        addressable,
        available_bytes,
        inference_reserve,
        args.gpu_memory_utilization,
    );
    Ok(StartupProjection {
        mem_available_bytes: available_bytes,
        target_construction_bytes: super::weights::construction_overhead_bytes(config).total(),
        qwen4_transient_bytes: qwen4_construction_transient_bytes(
            config,
            super::speculative_draft_count(args).is_some(),
        )?,
        pools_bytes: breakdown
            .ssm_pool_bytes
            .saturating_add(breakdown.ssm_snapshot_bytes)
            .saturating_add(breakdown.ple_history_bytes),
        buffer_arena_bytes,
        kv_projected_bytes: kv_projected,
        qsa_fp8_shadow_bytes: breakdown.qsa_fp8_shadow_bytes,
        floor_bytes,
        is_discrete,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
    breakdown: &ReserveBreakdown,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    // Pre-construction projection: on UMA (GB10), project against host
    // `MemAvailable` and the UMA qualification floor; on discrete GPUs, project
    // KV cache and headroom against VRAM `available_free` and the discrete VRAM
    // floor so host RAM neither inflates nor falsely constrains discrete VRAM.
    let is_discrete = !gpu.is_unified_memory() || startup_ledger::is_discrete_device();
    let projection_inputs = if is_discrete {
        let total_vram = gpu
            .total_memory()
            .ok()
            .filter(|&b| b > 0)
            .or_else(|| atlas_core::device::active_device_props().map(|p| p.total_mem_bytes));
        let floor = super::startup_ledger::discrete_vram_floor_bytes(total_vram);
        Some((available_free, floor))
    } else {
        startup_ledger::mem_available_bytes()
            .map(|mem| Ok::<_, anyhow::Error>((mem, startup_ledger::memavailable_floor_bytes()?)))
            .transpose()?
    };
    if let Some((mem_available, floor_bytes)) = projection_inputs {
        let projection = startup_projection(
            args,
            config,
            breakdown,
            total_reserve.saturating_sub(inference_reserve),
            inference_reserve,
            mem_available,
            floor_bytes,
            is_discrete,
        )?;
        tracing::info!(
            target: "atlas_startup_budget",
            mem_available_mib = mem_available / MIB,
            need_mib = projection.need_bytes() / MIB,
            kv_projected_mib = projection.kv_projected_bytes / MIB,
            projected_post_start_mib = projection.projected_post_start_bytes() / MIB,
            floor_mib = projection.floor_bytes / MIB,
            "STARTUP_BUDGET_PROJECTION {}",
            projection.summary()
        );
        startup_ledger::record(
            "projection/pre-construction",
            projection.summary(),
            Some(projection.need_bytes()),
            None,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}

/// Initialize the REST retrieval draft store against the tokenizer the
/// server actually loaded.
///
/// Call once, after the tokenizer is available: the store carries the
/// fingerprint of the `tokenizer.json` it was built from, and a mismatch
/// is FATAL. Token ids minted by a different tokenizer are not
/// lower-quality drafts, they are noise — every one of them would be
/// rejected at verify while still costing a verify slot.
///
/// A no-op returning `Ok(())` when `ATLAS_REST_STORE` is unset, which is
/// the default.
pub(crate) fn init_rest_store(model_dir: &std::path::Path) -> Result<()> {
    if std::env::var_os("ATLAS_REST_STORE").is_none() {
        return Ok(());
    }
    // The same file `ChatTokenizer::from_model_dir` loads, so the
    // fingerprint is taken over the bytes actually in use.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_json = std::fs::read(&tokenizer_path).with_context(|| {
        format!(
            "ATLAS_REST_STORE is set but {} could not be read to validate the store",
            tokenizer_path.display()
        )
    })?;
    crate::rest_store::init(&tokenizer_json).context("REST draft store rejected at startup")?;
    Ok(())
}

#[cfg(test)]
mod dflash_preflight_shape_tests {
    use super::{MIB, dflash_batched_proposer_state_bytes, speculative_cuda_headroom};

    #[test]
    fn dflash_block16_reserves_for_fifteen_drafts() {
        assert_eq!(
            speculative_cuda_headroom(Some(15), 4096),
            4 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn plain_decode_keeps_the_small_reserve() {
        assert_eq!(speculative_cuda_headroom(None, 4096), 512 * 1024 * 1024);
    }

    #[test]
    fn dflash_batched_proposer_state_scales_with_batch_and_window() {
        // C=1 or non-DFlash adds zero extra batched proposer headroom.
        assert_eq!(
            dflash_batched_proposer_state_bytes(true, 1, 4096, 262_144).unwrap(),
            0
        );
        assert_eq!(
            dflash_batched_proposer_state_bytes(false, 16, 4096, 262_144).unwrap(),
            0
        );
        // C=16 at 4096 window reserves 16 * 200 MiB = 3200 MiB.
        assert_eq!(
            dflash_batched_proposer_state_bytes(true, 16, 4096, 262_144).unwrap(),
            16 * 200 * MIB
        );
        // Half window (2048) scales proportionally to 100 MiB/seq.
        assert_eq!(
            dflash_batched_proposer_state_bytes(true, 16, 2048, 262_144).unwrap(),
            16 * 100 * MIB
        );
        // Window=0 (full attention) caps at the 4096-slot resident ring.
        assert_eq!(
            dflash_batched_proposer_state_bytes(true, 8, 0, 262_144).unwrap(),
            8 * 200 * MIB
        );
        // Overflow fails closed.
        assert!(dflash_batched_proposer_state_bytes(true, usize::MAX, 4096, 262_144).is_err());
    }
}

#[cfg(test)]
mod qsa_fp8_shared_v8_preflight_tests {
    use super::{
        ModelConfig, cli, qwen4_qsa_fp8_shadow_reserve, qwen4_qsa_fp8_shadow_reserve_for_mode,
    };
    use crate::cli::{Cli, Command};
    use clap::Parser;
    use spark_model::layers::qwen4_qsa::prefill_fp8::Mode;

    fn serve_args() -> cli::ServeArgs {
        let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]).unwrap();
        match cli.command {
            Command::Serve(args) => args,
        }
    }

    fn qwen4_qsa_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.qwen4_qsa = true;
        config
    }

    #[test]
    fn shared_v8_mode_reserves_thirteen_planes_and_reclaims_eleven_planes_at_262k() {
        let config = qwen4_qsa_config();
        let mut args = serve_args();
        args.block_size = 16;
        args.max_batch_size = 1;

        args.max_seq_len = 32_768;
        assert_eq!(
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, Some(Mode::QkPv)).unwrap(),
            218_316_800
        );
        assert_eq!(
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, Some(Mode::Qk)).unwrap(),
            201_523_200
        );

        args.max_seq_len = 135_168;
        assert_eq!(
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, Some(Mode::QkPv)).unwrap(),
            899_891_200
        );

        args.max_seq_len = 262_144;
        let shared =
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, Some(Mode::QkPv)).unwrap();
        let per_layer = qwen4_qsa_fp8_shadow_reserve(&args, &config, true).unwrap();
        assert_eq!(shared, 1_745_043_456);
        assert_eq!(per_layer, 3_221_618_688);
        assert_eq!(per_layer - shared, 1_476_575_232);
        assert_eq!(
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, Some(Mode::Qk)).unwrap(),
            1_610_809_344
        );
        assert_eq!(
            qwen4_qsa_fp8_shadow_reserve_for_mode(&args, &config, None).unwrap(),
            0
        );
    }
}

#[cfg(test)]
#[path = "preflight_ssm_capacity_tests.rs"]
mod ssm_capacity_tests;

#[cfg(test)]
#[path = "preflight_arena_tests.rs"]
mod arena_tests;

#[cfg(test)]
#[path = "preflight_projection_tests.rs"]
mod projection_tests;

#[cfg(test)]
mod discrete_preflight_tests {
    use super::{
        MIB, ModelConfig, ReserveBreakdown, cli, effective_discrete_reserve_mb,
        qwen4_construction_transient_bytes, startup_projection,
    };
    use crate::cli::{Cli, Command};
    use clap::Parser;

    const GIB: usize = 1024 * MIB;

    fn serve_args() -> cli::ServeArgs {
        let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]).unwrap();
        match cli.command {
            Command::Serve(args) => args,
        }
    }

    #[test]
    fn discrete_reserve_cap_scales_small_gpus_and_leaves_uma_and_large_gpus_intact() {
        // UMA GB10 is always untouched.
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(16 * GIB), false),
            4096
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(128 * GIB), false),
            4096
        );

        // Discrete GPUs <= 48 GiB cap default 4096 MB to min(configured, max(512, total_mb / 8)).
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(2 * GIB), true),
            512
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(4 * GIB), true),
            512
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(8 * GIB), true),
            1024
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(16 * GIB), true),
            2048
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(24 * GIB), true),
            3072
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(32 * GIB), true),
            4096
        );
        // Real 32 GB RTX 5090 reports ~32,607 MiB -> 4075 MiB.
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(32_607 * MIB), true),
            4075
        );

        // Explicit smaller operator settings are never raised.
        assert_eq!(
            effective_discrete_reserve_mb(256, Some(16 * GIB), true),
            256
        );
        assert_eq!(
            effective_discrete_reserve_mb(1024, Some(16 * GIB), true),
            1024
        );
        assert_eq!(effective_discrete_reserve_mb(0, Some(16 * GIB), true), 0);

        // Large discrete GPUs (> 48 GiB: 72 / 96 / 288 GiB) leave configured values intact.
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(72 * GIB), true),
            4096
        );
        assert_eq!(
            effective_discrete_reserve_mb(8192, Some(96 * GIB), true),
            8192
        );
        assert_eq!(
            effective_discrete_reserve_mb(4096, Some(288 * GIB), true),
            4096
        );
    }

    #[test]
    fn qwen4_routed_transposed_slab_scales_by_ep_times_pp() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        let lm_head = config.vocab_size * config.hidden_size * 9 / 16;

        config.ep_world_size = 1;
        config.pp_world_size = 1;
        assert_eq!(
            qwen4_construction_transient_bytes(&config, false).unwrap(),
            1_415_578_112 + lm_head
        );

        config.ep_world_size = 2;
        config.pp_world_size = 1;
        assert_eq!(
            qwen4_construction_transient_bytes(&config, false).unwrap(),
            1_415_578_112 / 2 + lm_head
        );

        config.ep_world_size = 2;
        config.pp_world_size = 2;
        assert_eq!(
            qwen4_construction_transient_bytes(&config, false).unwrap(),
            1_415_578_112 / 4 + lm_head
        );
    }

    #[test]
    fn discrete_projection_sizes_kv_from_vram_free_and_discrete_floor() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        let mut args = serve_args();
        args.max_batch_size = 8;
        args.max_seq_len = 135_168;
        args.block_size = 16;
        args.gpu_memory_utilization = 0.90;

        // Discrete 16 GB GPU with 6 GiB VRAM left after weights and 1 GiB inference reserve:
        // projected KV must be bounded by (6 - 1) * 0.90 = 4.5 GiB, not host RAM.
        let proj = startup_projection(
            &args,
            &config,
            &ReserveBreakdown::default(),
            GIB,
            GIB,
            6 * GIB,
            256 * MIB,
            true,
        )
        .unwrap();
        assert_eq!(proj.mem_available_bytes, 6 * GIB);
        assert_eq!(proj.floor_bytes, 256 * MIB);
        assert_eq!(proj.kv_projected_bytes, ((5 * GIB) as f64 * 0.90) as usize);
        assert!(proj.summary().contains("projected post-start GPU VRAM"));
    }

    #[test]
    fn kv_block_bytes_respect_actual_kv_cache_dtype_and_c64_gdn_retain_matches_runtime() {
        use super::{kv_block_bytes_all_layers, kv_block_bytes_all_layers_for_dtype};

        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();

        // Flash-Next: 12 attention layers x 2 KV heads x 256 head_dim x 16 tokens/block.
        // BF16 = 393,216 B, FP8 = 196,608 B (1/2), NVFP4 = 110,592 B (9/32 = 4096 data + 512 scales per side).
        let bf16 = kv_block_bytes_all_layers(&config, 16).unwrap();
        assert_eq!(bf16, 12 * 32_768);
        assert_eq!(
            kv_block_bytes_all_layers_for_dtype(&config, 16, "bf16").unwrap(),
            bf16
        );
        assert_eq!(
            kv_block_bytes_all_layers_for_dtype(&config, 16, "fp8").unwrap(),
            bf16 / 2
        );
        let nvfp4 = kv_block_bytes_all_layers_for_dtype(&config, 16, "nvfp4").unwrap();
        assert_eq!(nvfp4, 12 * 2 * (4_096 + 512));
        assert_eq!(nvfp4, 110_592);
        // At block_size=32, matches factory NVFP4 12-layer block size (221,184 B) byte-for-byte.
        assert_eq!(
            kv_block_bytes_all_layers_for_dtype(&config, 32, "nvfp4").unwrap(),
            221_184
        );
        // Asymmetric BF16-K / Turbo4-V: 16,384 (K) + 4,608 (V) = 20,992 B/layer.
        assert_eq!(
            kv_block_bytes_all_layers_for_dtype(&config, 16, "bf16k_turbo4v").unwrap(),
            12 * (16_384 + 4_608)
        );
        // Unknown dtype falls back conservatively to BF16.
        assert_eq!(
            kv_block_bytes_all_layers_for_dtype(&config, 16, "unknown").unwrap(),
            bf16
        );

        // `startup_projection` uses `args.kv_cache_dtype` instead of hardcoding BF16:
        // for batch=1 / 135,168 tokens with ample headroom, nvfp4 < fp8 < bf16 exactly.
        let mut args = serve_args();
        args.max_batch_size = 1;
        args.max_seq_len = 135_168;
        args.block_size = 16;
        args.gpu_memory_utilization = 0.90;
        let run_proj = |dt: &str| {
            let mut a = args.clone();
            a.kv_cache_dtype = dt.to_string();
            startup_projection(
                &a,
                &config,
                &ReserveBreakdown::default(),
                GIB,
                GIB,
                32 * GIB,
                256 * MIB,
                false,
            )
            .unwrap()
            .kv_projected_bytes
        };
        let proj_bf16 = run_proj("bf16");
        let proj_fp8 = run_proj("fp8");
        let proj_nvfp4 = run_proj("nvfp4");
        assert_eq!(proj_fp8 * 2, proj_bf16);
        assert_eq!(proj_nvfp4 * 32, proj_bf16 * 9);

        // C=64 GDN lazy retain reserve matches exact runtime compact ring stride
        // for both Qwen3-Next (nv=32 -> 1,064,960 B/lane) and Flash-Next (nv=48 ->
        // 1,335,296 B/lane x 64 lanes x 36 SSM layers = 3,076,521,984 B = 2.865 GiB).
        let lane_q3n = spark_model::layers::qwen3_ssm::gdn_lazy_retain_lane_stride_bytes(
            config.ssm_qkvz_size(),
            config.linear_num_value_heads,
            true,
        )
        .unwrap();
        assert_eq!(lane_q3n, 64 * 16_640);

        let mut fn_config = config.clone();
        fn_config.linear_num_key_heads = 16;
        fn_config.linear_key_head_dim = 128;
        fn_config.linear_num_value_heads = 48;
        fn_config.linear_value_head_dim = 128;
        let lane_fn = spark_model::layers::qwen3_ssm::gdn_lazy_retain_lane_stride_bytes(
            fn_config.ssm_qkvz_size(),
            fn_config.linear_num_value_heads,
            true,
        )
        .unwrap();
        let reserve_c64 = spark_model::layers::qwen4_verify_lanes::checked_lane_reserve(
            64,
            fn_config.num_attention_layers(),
            0,
            fn_config.num_ssm_layers(),
            Some(lane_fn),
        )
        .unwrap();
        assert_eq!(lane_fn, 1_335_296);
        assert_eq!(reserve_c64.gdn_retain_bytes, 64 * 36 * 1_335_296);
    }

    #[test]
    fn mixed_nvfp4_kv_block_bytes_and_qsa_fp8_shadow_guard_match_1x_tp2_tp4() {
        use super::{
            kv_block_bytes_all_layers_for_args, kv_block_bytes_all_layers_with_hp,
            qwen4_qsa_fp8_shadow_eligible,
        };

        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.qwen4_qsa = true;
        assert_eq!(config.num_attention_layers(), 12);
        assert_eq!(config.num_key_value_heads, 2);

        // QSA FP8 E4M3 shadows are ONLY eligible on BF16 KV caches; NVFP4/FP8
        // never materialize shadows and never reserve 6.5 GiB.
        assert!(qwen4_qsa_fp8_shadow_eligible(&config, "bf16"));
        assert!(qwen4_qsa_fp8_shadow_eligible(&config, " BF16 "));
        assert!(!qwen4_qsa_fp8_shadow_eligible(&config, "nvfp4"));
        assert!(!qwen4_qsa_fp8_shadow_eligible(&config, "fp8"));

        // 1x Flash-Next (12 attn layers x 2 KV heads x 256 head_dim x 16 tok/block):
        //   BF16 layer = 32,768 B, NVFP4 layer = 9,216 B.
        //   hp=0    (0 BF16 + 12 NVFP4) = 110,592 B (6.75 KiB/tok)
        //   hp=1    (2 BF16 + 10 NVFP4) = 157,696 B (9.625 KiB/tok)
        //   hp=auto (4 BF16 +  8 NVFP4) = 204,800 B (12.50 KiB/tok)
        //   hp=max  (12 BF16)           = 393,216 B (24.00 KiB/tok)
        assert_eq!(
            kv_block_bytes_all_layers_with_hp(&config, 16, "nvfp4", "0", "").unwrap(),
            12 * 9_216
        );
        assert_eq!(
            kv_block_bytes_all_layers_with_hp(&config, 16, "nvfp4", "1", "").unwrap(),
            2 * 32_768 + 10 * 9_216
        );
        assert_eq!(
            kv_block_bytes_all_layers_with_hp(&config, 16, "nvfp4", "auto", "").unwrap(),
            4 * 32_768 + 8 * 9_216
        );
        assert_eq!(
            kv_block_bytes_all_layers_with_hp(&config, 16, "nvfp4", "max", "").unwrap(),
            12 * 32_768
        );
        // Explicit measured layer set overrides positional hp count and deduplicates.
        assert_eq!(
            kv_block_bytes_all_layers_with_hp(&config, 16, "nvfp4", "auto", "0,5,5,11,99").unwrap(),
            3 * 32_768 + 9 * 9_216
        );

        // TP=2 / TP=4 shards Flash-Next KV heads from 2 -> 1 per rank (halving block bytes).
        let mut tp2_config = config.clone();
        tp2_config.tp_world_size = 2;
        tp2_config.num_key_value_heads = 1;
        let mut args = serve_args();
        args.block_size = 16;
        args.kv_cache_dtype = "nvfp4".to_string();
        args.kv_high_precision_layers = "auto".to_string();
        let bytes_1x = kv_block_bytes_all_layers_for_args(&config, &args).unwrap();
        let bytes_tp2 = kv_block_bytes_all_layers_for_args(&tp2_config, &args).unwrap();
        assert_eq!(bytes_1x, 204_800);
        assert_eq!(bytes_tp2, 102_400);

        // TP=4 replicates 2 KV heads across 4 ranks (local_kv_heads = 1 -> same per-rank KV as TP=2).
        let mut tp4_config = tp2_config;
        tp4_config.tp_world_size = 4;
        tp4_config.tp_kv_replicate = 2;
        assert_eq!(
            kv_block_bytes_all_layers_for_args(&tp4_config, &args).unwrap(),
            102_400
        );
    }
}
