// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer KV cache dtype vector construction.

/// Returns `true` when the positional first-N/last-N boundary policy silently
/// promotes **100%** of attention layers from a quantized `kv_dtype` to
/// `boundary_dtype` (i.e. `2 * high_precision_layers >= num_attention_layers`),
/// eliminating all KV cache compression.
///
/// Example: Qwen3.8-27B has `num_attention_layers = 16`; setting
/// `--kv-high-precision-layers 8` sets `0..8` and `8..16` to BF16 (`16/16` = 100% BF16).
#[must_use]
pub(crate) fn is_full_boundary_override(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layers: usize,
    boundary_dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> bool {
    num_attention_layers > 0
        && kv_dtype != boundary_dtype
        && high_precision_layers > 0
        && high_precision_layers.saturating_mul(2) >= num_attention_layers
}

/// Build per-attention-layer KV cache dtype vector.
///
/// When `high_precision_layers` is 0, returns an empty vec (all layers use uniform dtype).
/// When non-zero, the first N and last N attention layers use `boundary_dtype` (default
/// BF16); middle layers use the base `kv_dtype`. Per TurboQuant+ LA-V7 Mode 7: a
/// flexible boundary policy lets you mix e.g. middle=Turbo2 + boundary=Fp8 instead of
/// the rigid middle=Turbo2 + boundary=BF16. Returns empty vec if `boundary_dtype` ==
/// `kv_dtype` (no benefit) or `high_precision_layers` == 0.
pub(crate) fn build_layer_kv_dtypes(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layers: usize,
    boundary_dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> Vec<spark_runtime::kv_cache::KvCacheDtype> {
    if high_precision_layers == 0 || kv_dtype == boundary_dtype {
        return vec![];
    }

    let hp = high_precision_layers.min(num_attention_layers);
    let mut dtypes = vec![kv_dtype; num_attention_layers];

    for i in 0..hp.min(num_attention_layers) {
        dtypes[i] = boundary_dtype;
    }
    for i in num_attention_layers.saturating_sub(hp)..num_attention_layers {
        dtypes[i] = boundary_dtype;
    }

    let hp_count = dtypes.iter().filter(|d| **d == boundary_dtype).count();
    if is_full_boundary_override(
        kv_dtype,
        num_attention_layers,
        high_precision_layers,
        boundary_dtype,
    ) {
        tracing::warn!(
            "⚠ Selective boundary KV cache: --kv-high-precision-layers {} protects first {} \
             and last {} of {} attention layers, silently promoting 100% ({}/{}) of layers from \
             {} to {} (0% KV compression!). Pass --kv-high-precision-layers 0 (with \
             ATLAS_BITFP4_KV=1 for native Blackwell FP4 KV) or --kv-high-precision-layer-set \
             for partial protection.",
            high_precision_layers,
            hp,
            hp,
            num_attention_layers,
            hp_count,
            num_attention_layers,
            kv_dtype,
            boundary_dtype,
        );
    } else {
        tracing::info!(
            "Selective boundary KV cache: {}/{} attention layers at {}, rest at {}",
            hp_count,
            num_attention_layers,
            boundary_dtype,
            kv_dtype,
        );
    }

    dtypes
}

/// Build a per-attention-layer KV cache dtype vector from an EXPLICIT set of
/// attention-layer indices to keep at `boundary_dtype`.
///
/// This is the measured-ordering analog of [`build_layer_kv_dtypes`]: instead of
/// the positional first-N/last-N heuristic, the caller supplies the exact layers
/// (attention-layer-local indices, i.e. 0..num_attention_layers, NOT global model
/// layer ids) that a per-layer sensitivity sweep found most sensitive to KV
/// quantization. Those layers get `boundary_dtype`; the rest get `kv_dtype`.
///
/// Rationale (buun-llama-cpp / VBR): KV quantization sensitivity is not uniformly
/// positional — some middle layers are far more sensitive than the boundary ones
/// the positional heuristic protects. Spending the same BF16 budget on the
/// *measured* most-sensitive layers gives better long-context coherence at
/// identical memory.
///
/// Out-of-range indices are ignored (clamped by bounds check) so a stale set from
/// a differently-sized model can't panic the loader. Returns an empty vec (uniform
/// dtype) if `kv_dtype == boundary_dtype` or the set is empty — matching the
/// no-benefit semantics of [`build_layer_kv_dtypes`].
pub(crate) fn build_layer_kv_dtypes_from_set(
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layer_set: &[usize],
    boundary_dtype: spark_runtime::kv_cache::KvCacheDtype,
) -> Vec<spark_runtime::kv_cache::KvCacheDtype> {
    if kv_dtype == boundary_dtype || high_precision_layer_set.is_empty() {
        return vec![];
    }

    let mut dtypes = vec![kv_dtype; num_attention_layers];
    let mut applied: Vec<usize> = Vec::new();
    for &idx in high_precision_layer_set {
        if idx < num_attention_layers {
            dtypes[idx] = boundary_dtype;
            applied.push(idx);
        } else {
            tracing::warn!(
                "kv-high-precision-layer-set: attention-layer index {} out of range \
                 (model has {} attention layers) — ignoring",
                idx,
                num_attention_layers,
            );
        }
    }

    let hp_count = dtypes.iter().filter(|d| **d == boundary_dtype).count();
    tracing::info!(
        "Measured-ordering KV cache: {}/{} attention layers at {} (set {:?}), rest at {}",
        hp_count,
        num_attention_layers,
        boundary_dtype,
        applied,
        kv_dtype,
    );

    dtypes
}
