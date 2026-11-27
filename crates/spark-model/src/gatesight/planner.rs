// SPDX-License-Identifier: AGPL-3.0-only

//! 2-zone prefix-safe compaction planner, Flash-Next QSA safety floor, cross-layer budget
//! allocator, and exact streaming Top-K reservoir.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{
    DENSE_GQA_MIN_RETENTION, FLASH_NEXT_QSA_MIN_RETENTION, GATESIGHT_BLOCK_SIZE, GateSightError,
    GateSightMode, Result,
};

/// Architecture-specific retention safety policy preventing accuracy cliffs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelFamilyPolicy {
    /// Dense-GQA hybrid (e.g. `Qwen3.8-27B`): allows aggressive eviction down to `p = 0.10`.
    Qwen38Dense27B,
    /// QSA sparse-attention hybrid (e.g. `Qwen3.8-Flash-Next`): hard safety floor `p >= 0.60`
    /// to avoid double-pruning collapse on top of QSA's native 2,051-token sparse window.
    Qwen38FlashNextQsa,
}

impl ModelFamilyPolicy {
    /// Minimum permissible Zone-2 retention ratio for this model family.
    #[must_use]
    pub const fn min_retention_floor(self) -> f32 {
        match self {
            Self::Qwen38Dense27B => DENSE_GQA_MIN_RETENTION,
            Self::Qwen38FlashNextQsa => FLASH_NEXT_QSA_MIN_RETENTION,
        }
    }

    /// Clamp a requested retention ratio against the family safety floor, returning
    /// `(effective_retention, was_clamped)`.
    #[must_use]
    pub fn clamp_retention(self, requested: f32) -> (f32, bool) {
        let floor = self.min_retention_floor();
        if requested < floor {
            (floor, true)
        } else {
            (requested.min(1.0), false)
        }
    }
}

/// One scored Zone-2 item (either a 16-token physical block in Path A or a token position in Path B).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredItem {
    /// Monotonic logical index (block index for Path A, token position for Path B).
    pub index: usize,
    /// Importance score (higher is more important).
    pub score: f32,
}

impl ScoredItem {
    /// Construct a scored item.
    #[must_use]
    pub const fn new(index: usize, score: f32) -> Self {
        Self { index, score }
    }
}

/// Min-heap wrapper ordering lowest-importance items at the top so the reservoir can evict in
/// `O(log K)`. Ties break deterministically by **higher index kept** (smaller index evicted first),
/// matching global post-prefill Top-K selection 100%.
#[derive(Clone, Copy, Debug)]
struct MinScoreEntry(ScoredItem);

impl PartialEq for MinScoreEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.to_bits() == other.0.score.to_bits() && self.0.index == other.0.index
    }
}

impl Eq for MinScoreEntry {}

impl PartialOrd for MinScoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScoreEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for BinaryHeap (max-heap -> min-heap by `(score, index)`).
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| other.0.index.cmp(&self.0.index))
    }
}

/// Bounded streaming Top-K reservoir guaranteeing exact parity with global post-prefill Top-K.
///
/// Because Flux knows `total_prompt_tokens` at prefill entry (`prefill_b.rs:378`), GateSight's
/// relative age features are exact mid-prefill. Pushing each chunk's Zone-2 items into this
/// reservoir bounds peak KV memory during 1M-token prefills while keeping the exact same
/// surviving set as post-prefill global compaction.
#[derive(Clone, Debug)]
pub struct StreamingTopKReservoir {
    capacity: usize,
    heap: BinaryHeap<MinScoreEntry>,
    evicted_indices: Vec<usize>,
}

impl StreamingTopKReservoir {
    /// Create a reservoir retaining at most `capacity` highest-scoring items.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity.saturating_add(1)),
            evicted_indices: Vec::new(),
        }
    }

    /// Maximum number of retained items.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of retained items in the reservoir.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// True when the reservoir holds zero items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Push one scored item, returning `Some(evicted_index)` if an item was displaced.
    pub fn push(&mut self, item: ScoredItem) -> Option<usize> {
        if self.capacity == 0 {
            self.evicted_indices.push(item.index);
            return Some(item.index);
        }
        if self.heap.len() < self.capacity {
            self.heap.push(MinScoreEntry(item));
            return None;
        }
        let Some(weakest) = self.heap.peek().copied().map(|entry| entry.0) else {
            self.heap.push(MinScoreEntry(item));
            return None;
        };
        let incoming_beats_weakest = match item.score.total_cmp(&weakest.score) {
            Ordering::Greater => true,
            Ordering::Equal => item.index > weakest.index,
            Ordering::Less => false,
        };
        if incoming_beats_weakest {
            let displaced = self.heap.pop().map(|entry| entry.0.index);
            self.heap.push(MinScoreEntry(item));
            if let Some(idx) = displaced {
                self.evicted_indices.push(idx);
            }
            displaced
        } else {
            self.evicted_indices.push(item.index);
            Some(item.index)
        }
    }

    /// Push a chunk of scored items and return indices evicted during this chunk.
    pub fn push_chunk(&mut self, items: impl IntoIterator<Item = ScoredItem>) -> Vec<usize> {
        let mut chunk_evicted = Vec::new();
        for item in items {
            if let Some(idx) = self.push(item) {
                chunk_evicted.push(idx);
            }
        }
        chunk_evicted
    }

    /// Consume the reservoir and return `(kept_sorted_indices, evicted_sorted_indices)`.
    #[must_use]
    pub fn finish(mut self) -> (Vec<usize>, Vec<usize>) {
        let mut kept: Vec<usize> = self.heap.into_iter().map(|e| e.0.index).collect();
        kept.sort_unstable();
        self.evicted_indices.sort_unstable();
        (kept, self.evicted_indices)
    }
}

/// Inputs describing a 2-zone compaction decision for one attention layer.
#[derive(Clone, Debug)]
pub struct CompactionRequest<'a> {
    /// Requested eviction mode (`PathABlock16` or `PathBMlp`).
    pub mode: GateSightMode,
    /// Architecture safety policy (`Qwen38Dense27B` vs `Qwen38FlashNextQsa`).
    pub policy: ModelFamilyPolicy,
    /// Total tokens in the prompt.
    pub total_tokens: usize,
    /// Shared prefix length in tokens (Zone 1 `pos < shared_prefix_len`, protected for `RadixTree`).
    pub shared_prefix_len: usize,
    /// Physical KV block size (`16` for BitDecoding-FP4).
    pub block_size: usize,
    /// Minimum leading sink blocks always protected in Zone 1 (at least 1 block = 16 sinks).
    pub sink_blocks: usize,
    /// Trailing local-window tokens always protected at the end of Zone 2.
    pub recent_window_tokens: usize,
    /// Desired Zone-2 retention fraction `p in (0, 1]`.
    pub target_retention: f32,
    /// Per-block scores (for `PathABlock16`, length `ceil(total_tokens / block_size)`) or
    /// per-token scores (for `PathBMlp`, length `total_tokens`).
    pub scores: &'a [f32],
}

/// Seam 1 compaction output consumed by Flux's KV allocator and BitDecoding-FP4 kernels.
#[derive(Clone, Debug, PartialEq)]
pub enum CompactionPlan {
    /// GateSight disabled: all blocks/tokens retained untouched.
    Disabled,
    /// Path A (`block_size = 16`): zero re-quantization for both token-wise `K` and transposed `V^T`.
    Block16PathA {
        /// Zone 1 `[0..zone1_protected_blocks)` stays immutable in `RadixTree` (subsumes block-0 sinks).
        zone1_protected_blocks: usize,
        /// Sorted physical block indices retained in Zone 2 (includes protected recent tail blocks).
        kept_zone2_blocks: Vec<usize>,
        /// Sorted physical block indices safe to free immediately.
        evicted_zone2_blocks: Vec<usize>,
        /// Effective retention fraction after safety-floor clamping.
        effective_retention: f32,
        /// True when `ModelFamilyPolicy` raised `target_retention` to its safety floor.
        was_clamped: bool,
    },
    /// Path B (per-token): raw 144 B/tok memcpy for `K` + 1-shot `V^T` repack on kept Zone-2 tokens.
    TokenLevelPathB {
        /// Zone 1 `[0..zone1_protected_tokens)` (aligned to `block_size = 16`) stays immutable in `RadixTree`.
        zone1_protected_tokens: usize,
        /// Sorted token positions retained in Zone 2 (includes protected recent tail tokens).
        kept_zone2_tokens: Vec<usize>,
        /// Sorted token positions evicted from Zone 2.
        evicted_zone2_tokens: Vec<usize>,
        /// True whenever any Zone-2 token was evicted and transposed `V^T` requires 1-shot repack.
        requires_v_transpose_repack: bool,
        /// Effective retention fraction after safety-floor clamping.
        effective_retention: f32,
        /// True when `ModelFamilyPolicy` raised `target_retention` to its safety floor.
        was_clamped: bool,
    },
}

/// Compute a 2-zone compaction plan respecting `RadixTree` prefix immutability (Zone 1),
/// attention sinks (block 0), recent local tail, and the Flash-Next QSA `p >= 0.60` safety floor.
///
/// # Errors
/// Returns [`GateSightError::InvalidConfig`] or [`GateSightError::DimensionMismatch`] on invalid
/// parameters or mismatched score slice length.
pub fn plan_zone2_compaction(req: &CompactionRequest<'_>) -> Result<CompactionPlan> {
    if req.mode == GateSightMode::Disabled {
        return Ok(CompactionPlan::Disabled);
    }
    if req.block_size == 0 {
        return Err(GateSightError::InvalidConfig(
            "block_size must be > 0".to_string(),
        ));
    }
    if !req.target_retention.is_finite()
        || req.target_retention <= 0.0
        || req.target_retention > 1.0
    {
        return Err(GateSightError::InvalidConfig(format!(
            "target_retention={} must be in (0.0, 1.0]",
            req.target_retention
        )));
    }

    let (effective_retention, was_clamped) = req.policy.clamp_retention(req.target_retention);
    let total_blocks = req.total_tokens.div_ceil(req.block_size);
    let min_sink_tokens = req.sink_blocks.max(1).saturating_mul(req.block_size);
    let zone1_blocks = req
        .shared_prefix_len
        .max(min_sink_tokens)
        .div_ceil(req.block_size)
        .min(total_blocks);

    match req.mode {
        GateSightMode::Disabled => Ok(CompactionPlan::Disabled),
        GateSightMode::PathABlock16 => {
            if req.scores.len() != total_blocks {
                return Err(GateSightError::DimensionMismatch {
                    context: "PathABlock16 block scores",
                    expected: total_blocks,
                    actual: req.scores.len(),
                });
            }
            let recent_blocks = req.recent_window_tokens.div_ceil(req.block_size);
            let zone2_count = total_blocks.saturating_sub(zone1_blocks);
            if zone2_count == 0 {
                return Ok(CompactionPlan::Block16PathA {
                    zone1_protected_blocks: zone1_blocks,
                    kept_zone2_blocks: Vec::new(),
                    evicted_zone2_blocks: Vec::new(),
                    effective_retention,
                    was_clamped,
                });
            }
            let tail_start = total_blocks.saturating_sub(recent_blocks).max(zone1_blocks);
            let evictable_count = tail_start.saturating_sub(zone1_blocks);
            let target_zone2_keep = ((zone2_count as f32) * effective_retention).ceil() as usize;
            let tail_count = total_blocks - tail_start;
            let reservoir_cap = target_zone2_keep
                .saturating_sub(tail_count)
                .min(evictable_count);

            let mut reservoir = StreamingTopKReservoir::new(reservoir_cap);
            for block_idx in zone1_blocks..tail_start {
                reservoir.push(ScoredItem::new(block_idx, req.scores[block_idx]));
            }
            let (mut kept_zone2, evicted_zone2) = reservoir.finish();
            kept_zone2.extend(tail_start..total_blocks);

            Ok(CompactionPlan::Block16PathA {
                zone1_protected_blocks: zone1_blocks,
                kept_zone2_blocks: kept_zone2,
                evicted_zone2_blocks: evicted_zone2,
                effective_retention,
                was_clamped,
            })
        }
        GateSightMode::PathBMlp => {
            if req.scores.len() != req.total_tokens {
                return Err(GateSightError::DimensionMismatch {
                    context: "PathBMlp token scores",
                    expected: req.total_tokens,
                    actual: req.scores.len(),
                });
            }
            // Align Zone-1 boundary up to block_size = 16 so RadixTree shared blocks remain untouched.
            let zone1_tokens = (zone1_blocks * req.block_size).min(req.total_tokens);
            let zone2_count = req.total_tokens.saturating_sub(zone1_tokens);
            if zone2_count == 0 {
                return Ok(CompactionPlan::TokenLevelPathB {
                    zone1_protected_tokens: zone1_tokens,
                    kept_zone2_tokens: Vec::new(),
                    evicted_zone2_tokens: Vec::new(),
                    requires_v_transpose_repack: false,
                    effective_retention,
                    was_clamped,
                });
            }
            let tail_start = req
                .total_tokens
                .saturating_sub(req.recent_window_tokens)
                .max(zone1_tokens);
            let evictable_count = tail_start.saturating_sub(zone1_tokens);
            let target_zone2_keep = ((zone2_count as f32) * effective_retention).ceil() as usize;
            let tail_count = req.total_tokens - tail_start;
            let reservoir_cap = target_zone2_keep
                .saturating_sub(tail_count)
                .min(evictable_count);

            let mut reservoir = StreamingTopKReservoir::new(reservoir_cap);
            for pos in zone1_tokens..tail_start {
                reservoir.push(ScoredItem::new(pos, req.scores[pos]));
            }
            let (mut kept_zone2, evicted_zone2) = reservoir.finish();
            kept_zone2.extend(tail_start..req.total_tokens);
            let requires_v_transpose_repack = !evicted_zone2.is_empty();

            Ok(CompactionPlan::TokenLevelPathB {
                zone1_protected_tokens: zone1_tokens,
                kept_zone2_tokens: kept_zone2,
                evicted_zone2_tokens: evicted_zone2,
                requires_v_transpose_repack,
                effective_retention,
                was_clamped,
            })
        }
    }
}

/// Allocated KV retention budget for one FullAttention layer from GateSight §4.2 (`K_l proportional w_l`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerBudget {
    pub attn_layer: usize,
    pub gate_weight: f32,
    pub kept_items: usize,
}

/// Allocate `total_budget` Zone-2 items across FullAttention layers proportionally to their
/// attention output-gate weights `w_l` (GateSight §4.2, Table 4), clamped to `[min_per_layer, max_per_layer]`.
///
/// # Errors
/// Returns [`GateSightError::InvalidConfig`] if `layer_gate_weights` is empty, bounds are
/// inverted, or any weight is negative/non-finite.
pub fn allocate_layer_budgets(
    layer_gate_weights: &[f32],
    total_budget: usize,
    min_per_layer: usize,
    max_per_layer: usize,
) -> Result<Vec<LayerBudget>> {
    if layer_gate_weights.is_empty() {
        return Err(GateSightError::InvalidConfig(
            "layer_gate_weights must not be empty".to_string(),
        ));
    }
    if min_per_layer > max_per_layer {
        return Err(GateSightError::InvalidConfig(format!(
            "min_per_layer ({min_per_layer}) > max_per_layer ({max_per_layer})"
        )));
    }
    for (idx, &w) in layer_gate_weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(GateSightError::InvalidConfig(format!(
                "layer_gate_weights[{idx}]={w} must be finite and >= 0"
            )));
        }
    }

    let n = layer_gate_weights.len();
    let min_total = min_per_layer.checked_mul(n).ok_or_else(|| {
        GateSightError::InvalidConfig(format!(
            "min_per_layer ({min_per_layer}) * num_layers ({n}) overflows usize"
        ))
    })?;
    let max_total = max_per_layer.saturating_mul(n);
    let clamped_total = total_budget.clamp(min_total, max_total);
    let mut alloc = vec![min_per_layer; n];
    let mut remaining = clamped_total - min_total;

    // Iteratively distribute remaining budget proportionally among layers below `max_per_layer`.
    while remaining > 0 {
        let active: Vec<usize> = (0..n).filter(|&i| alloc[i] < max_per_layer).collect();
        if active.is_empty() {
            break;
        }
        let weight_sum: f64 = active
            .iter()
            .map(|&i| f64::from(layer_gate_weights[i]))
            .sum();
        let use_uniform = weight_sum <= 0.0;

        let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(active.len());
        let mut distributed = 0usize;
        for &i in &active {
            let share = if use_uniform {
                (remaining as f64) / (active.len() as f64)
            } else {
                (remaining as f64) * (f64::from(layer_gate_weights[i]) / weight_sum)
            };
            let headroom = max_per_layer - alloc[i];
            let whole = (share.floor() as usize).min(headroom);
            alloc[i] += whole;
            distributed += whole;
            if alloc[i] < max_per_layer {
                remainders.push((i, share - (whole as f64)));
            }
        }
        remaining -= distributed;

        if remaining == 0 {
            break;
        }
        if distributed == 0 {
            // Largest-remainder tie-break (descending fractional part, then descending weight, then layer idx).
            remainders.sort_by(|&(ia, ra), &(ib, rb)| {
                rb.total_cmp(&ra)
                    .then_with(|| layer_gate_weights[ib].total_cmp(&layer_gate_weights[ia]))
                    .then_with(|| ia.cmp(&ib))
            });
            for (i, _) in remainders {
                if remaining == 0 {
                    break;
                }
                if alloc[i] < max_per_layer {
                    alloc[i] += 1;
                    remaining -= 1;
                }
            }
        }
    }

    Ok(layer_gate_weights
        .iter()
        .copied()
        .enumerate()
        .map(|(attn_layer, gate_weight)| LayerBudget {
            attn_layer,
            gate_weight,
            kept_items: alloc[attn_layer],
        })
        .collect())
}

/// Validate block alignment invariant against the shared Seam 1 BitDecoding-FP4 constant.
#[must_use]
pub const fn is_bitfp4_block_aligned(tokens: usize) -> bool {
    tokens.is_multiple_of(GATESIGHT_BLOCK_SIZE)
}
