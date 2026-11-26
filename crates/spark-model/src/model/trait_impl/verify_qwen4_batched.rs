// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence native-MTP K-row verify (`ATLAS_QWEN4_BATCHED_K5=1`).
//!
//! C sequences x K verify rows run through ONE target forward, packed
//! sequence-major (`Qwen4CrossSeqFrame`, at most 256 rows). Every row-parallel
//! stage reads its weights once for the whole frame on the exact multi-row
//! tiers (hyperconnections, attention QKV-free projections, SSM QKVZ/BA/out,
//! O projection, MoE `_t_lanes` tiles, final mixer, LM head, argmax). The
//! recurrent conv/GDN cores and the attention cores (QSA journal, index,
//! select, row-ordered attention) run per sequence with the single-sequence
//! K-row verifier's kernels, row count and buffer footprint, inside the verify
//! lane of the sequence's physical slot, so every sequence's transaction
//! state survives until its own commit.
//!
//! Each sequence's per-row argmax is therefore identical to its own
//! single-sequence `decode_verify_dflash` result: the only difference is the
//! M extent of row-exact launches. The scheduler then accepts, commits and
//! rolls back each sequence independently (the SGLang "promote the last
//! accepted slot" rule is the existing per-slot intermediate commit).
//!
//! Flat-chain cross-sequence verify with both eager execution and cached CUDA
//! graph capture/replay (keyed by slot/row geometry and feature bits). Admission
//! is complete and read-only before the first effect; a failure after that point
//! leaves each sequence's frame for the scheduler to abort under its lane.

use anyhow::{Context, Result, ensure};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GraphHandle, HostToDeviceCopy};
use std::collections::{HashMap, VecDeque};

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext, Qwen4CrossSeqFrame, Qwen4CrossSeqRows};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

const FLAT_VERIFY_DEPTHS: [usize; crate::layer::QWEN4_CROSS_SEQ_MAX_K] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8];

/// High-bit tag in `TransformerModel::batch_decode_graphs` key `(slots, meta)`
/// separating batched-verify graphs from `decode_a2`'s `padded_n` entries
/// while inheriting automatic invalidation in `free_sequence`, `compact_sequence`,
/// `rotary_graphs`, and `TransformerModel::drop`.
const QWEN4_BATCHED_VERIFY_GRAPH_TAG: usize = 1usize << (usize::BITS - 1);
/// Secondary tag bit distinguishing Layer-0 prefix (`embed -> Layer 0`) graphs
/// from Layer-1.. suffix graphs inside `batch_decode_graphs` while still
/// matching `is_batched_verify_graph_meta` for shared invalidation.
const QWEN4_BATCHED_VERIFY_LAYER0_GRAPH_TAG: usize = 1usize << (usize::BITS - 2);
/// Bit 17 in batched-verify graph metadata separating QSA pre-2048 dense-prefix
/// graphs (which run `project_index_qk_rows` + `verify_rows_stage_pool_rope_store_multi_seq`
/// alongside exact dense attention) from pure non-QSA dense graphs.
const QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT: usize = 1usize << 17;
const QWEN4_BATCHED_VERIFY_GRAPH_CAP: usize = 48;
/// Base graph cache cap on wide discrete GPUs (`> 64` SMs) covering the
/// extended 8-bucket superset arity ladder (`[2, 4, 8, 16, 24, 32, 48, 64]`);
/// still bounded dynamically by [`effective_batched_verify_graph_cap`].
const QWEN4_BATCHED_VERIFY_GRAPH_CAP_WIDE_SM: usize = 64;
const QWEN4_VERIFY_WIDE_SM_THRESHOLD: u32 = 64;
const QWEN4_BATCHED_VERIFY_CAPTURE_POISON_LIMIT: u32 = 8;

#[inline]
#[must_use]
const fn batched_verify_graph_base_cap_for_sm(sm_count: u32) -> usize {
    if sm_count > QWEN4_VERIFY_WIDE_SM_THRESHOLD {
        QWEN4_BATCHED_VERIFY_GRAPH_CAP_WIDE_SM
    } else {
        QWEN4_BATCHED_VERIFY_GRAPH_CAP
    }
}

/// Minimum UMA headroom above the effective memory floor (`3 GiB`) to keep the
/// full base graph cap (`48`, or `64` on wide-SM `> 64` SM GPUs) with zero
/// extra evictions.
const QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_MARGIN_BYTES: usize =
    3072 * spark_runtime::startup_ledger::MIB;
/// Tight UMA headroom threshold (`1 GiB` above the effective memory floor)
/// below which the batched-verify graph cache shrinks aggressively.
const QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_MARGIN_BYTES: usize =
    1024 * spark_runtime::startup_ledger::MIB;
/// Severe UMA headroom threshold (`512 MiB` above the effective memory floor)
/// below which only the active Layer-0 prefix + suffix pair stays resident.
const QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_MARGIN_BYTES: usize =
    512 * spark_runtime::startup_ledger::MIB;

/// Cache cap under moderate UMA pressure (`1024..3072 MiB` headroom): reclaims
/// idle bucket graphs while keeping hot cohort shapes warm.
const QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_CAP: usize = 16;
/// Cache cap under critical UMA pressure (`512..1024 MiB` headroom).
const QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_CAP: usize = 8;
/// Cache cap under emergency UMA pressure (`< 512 MiB` headroom): retains the
/// active Layer-0 prefix + suffix pair.
const QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_CAP: usize = 2;

/// Resolve the effective batched-verify graph cache cap for the current host
/// `MemAvailable` headroom above the active UMA floor.
///
/// Mirrors `verify_d::effective_verify_kgamma_graph_cap`:
/// - unknown telemetry (`None`) or headroom `>= 3072 MiB`: `base_cap` (`48`, or `64` on wide-SM `> 64` SM GPUs)
/// - `1024..3072 MiB`: `base_cap.min(16)`
/// - `512..1024 MiB`: `base_cap.min(8)`
/// - `< 512 MiB`: `base_cap.min(2)` (keeps the active Layer-0 + suffix pair)
#[inline]
#[must_use]
fn effective_batched_verify_graph_cap(
    base_cap: usize,
    mem_available_bytes: Option<usize>,
    floor_bytes: Option<usize>,
) -> usize {
    let base = base_cap.max(1);
    let (Some(avail), Some(floor)) = (mem_available_bytes, floor_bytes) else {
        return base;
    };
    let headroom = avail.saturating_sub(floor);
    if headroom < QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_MARGIN_BYTES {
        base.min(QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_CAP)
    } else if headroom < QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_MARGIN_BYTES {
        base.min(QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_CAP)
    } else if headroom < QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_MARGIN_BYTES {
        base.min(QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_CAP)
    } else {
        base
    }
}

fn batched_verify_mem_floor_bytes() -> Option<usize> {
    let read_live = || {
        spark_runtime::startup_ledger::effective_mem_floor_bytes()
            .ok()
            .filter(|&bytes| bytes > 0)
    };
    #[cfg(not(test))]
    {
        static FLOOR_BYTES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *FLOOR_BYTES.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchedVerifyGraphStage {
    Layer0Prefix,
    Suffix,
}

impl BatchedVerifyGraphStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Layer0Prefix => "layer0_prefix",
            Self::Suffix => "suffix",
        }
    }
}

type BatchedVerifyGraphKey = (Vec<usize>, usize);
type BatchedVerifyTrackerId = (usize, u64);

fn parse_batched_verify_graph_env(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

fn batched_verify_graph_enabled() -> bool {
    let read_live = || {
        parse_batched_verify_graph_env(
            std::env::var("ATLAS_QWEN4_BATCHED_VERIFY_GRAPH")
                .ok()
                .as_deref(),
        )
    };
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// `ATLAS_QWEN4_LAYER0_PREFIX_GRAPH=0` disables Layer-0 prefix (`embed -> Layer 0`)
/// CUDA graph capture while keeping Layer-1.. suffix graphs active. Default ON.
pub(super) fn qwen4_layer0_prefix_graph_enabled() -> bool {
    let read_live = || {
        parse_batched_verify_graph_env(
            std::env::var("ATLAS_QWEN4_LAYER0_PREFIX_GRAPH")
                .ok()
                .as_deref(),
        )
    };
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Pre-resolved GPU kernel handles for device-only Qwen4 batched embedding
/// (`token_ids_dev` already uploaded to `buffers.scratch()` outside capture).
#[derive(Clone, Copy)]
pub(super) enum Qwen4BatchedEmbedKernels {
    Fused(spark_runtime::gpu::KernelHandle),
    Split {
        gather: spark_runtime::gpu::KernelHandle,
        expand: spark_runtime::gpu::KernelHandle,
    },
}

/// Sentinel base encoding a slot-agnostic batched-verify suffix graph key
/// (`[QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL - num_seqs]`) used when Phase-1 stages
/// all suffix SSM descriptors (`all_suffix_ssm_staged == true`) and attention
/// uses fixed-address dense metadata (`all_dense_attn == true`). Placing the
/// arity strictly in the high sentinel range ensures `slots.contains(&slot)`
/// in `free_sequence`/`compact_sequence`/`invalidate_rotary_graphs` NEVER
/// collides with any real `slot_idx` (`0..64`), so warm cohort graphs survive
/// request retirement intact.
const QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL: usize = usize::MAX;

#[inline]
fn batched_verify_graph_key_slots(
    leader_slots: Vec<usize>,
    all_suffix_ssm_staged: bool,
    all_dense_attn: bool,
) -> Vec<usize> {
    if all_suffix_ssm_staged && all_dense_attn {
        vec![QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL.saturating_sub(leader_slots.len())]
    } else {
        leader_slots
    }
}

/// Maximum sequence arity bucketed for zero-padded superset graph replay on
/// `<= 64`-SM GPUs (GB10).
const QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS: usize = 16;
/// Maximum sequence arity bucketed for zero-padded superset graph replay on
/// wide discrete GPUs (`> 64` SMs: SM120 / SM103), matching
/// `CROSS_SEQ_TABLE_BYTES` (`64 * 64 B`) and `VERIFY_META_BYTES` (`64 * 32 B`).
const QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS_WIDE_SM: usize = 64;

/// Power-of-2 sequence arity buckets (`{2, 4, 8, 16}`) probed during
/// slot-agnostic verify graph lookup on `<= 64`-SM GPUs (GB10).
///
/// Newly captured graphs are always keyed by their exact captured arity
/// `[QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL - N_cap]` so `grid.y == N_cap` is an
/// invariant of every cached graph. When a smaller or draining cohort
/// (`2 <= actual_seqs < N_cap <= max_bucket`) has all SSM descriptors pre-staged
/// and zero-padded through the max bucket (`num_tokens = 0, flags = 0,
/// h_state_inter = 0`), every trailing GDN / QSA CTA returns immediately before
/// any global read/write/atomic, while every row-parallel, dense-attention,
/// MoE, LM-head, and tail kernel depends solely on `total_rows` and
/// `bt_stride_bucket`.
const QWEN4_VERIFY_GRAPH_SEQ_BUCKETS: [usize; 4] = [2, 4, 8, 16];
/// Extended high-concurrency superset buckets on `> 64`-SM GPUs (`SM120` / `SM103`).
const QWEN4_VERIFY_GRAPH_SEQ_BUCKETS_WIDE_SM: [usize; 8] = [2, 4, 8, 16, 24, 32, 48, 64];

#[inline]
#[must_use]
const fn batched_verify_seq_buckets_for_sm(sm_count: u32) -> &'static [usize] {
    if sm_count > QWEN4_VERIFY_WIDE_SM_THRESHOLD {
        &QWEN4_VERIFY_GRAPH_SEQ_BUCKETS_WIDE_SM
    } else {
        &QWEN4_VERIFY_GRAPH_SEQ_BUCKETS
    }
}

#[inline]
#[must_use]
const fn batched_verify_bucket_max_seqs_for_sm(sm_count: u32) -> usize {
    if sm_count > QWEN4_VERIFY_WIDE_SM_THRESHOLD {
        QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS_WIDE_SM
    } else {
        QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS
    }
}

/// Number of `GdnFusedMultiSeqDesc` / `QsaBatchSeqDesc` slots to upload in
/// Phase-1 staging so any cached superset-arity verify graph for `sm_count`
/// sees zeroed no-op descriptors in trailing slots.
#[inline]
#[must_use]
const fn batched_verify_ssm_desc_pad_seqs_for_sm(actual_seqs: usize, sm_count: u32) -> usize {
    let max_bucket = batched_verify_bucket_max_seqs_for_sm(sm_count);
    if actual_seqs >= 2 && actual_seqs < max_bucket {
        max_bucket
    } else {
        actual_seqs
    }
}

/// Number of `GdnFusedMultiSeqDesc` slots to upload in Phase-1 staging on
/// `<= 64`-SM GPUs (`{2, 4, 8, 16}`).
#[cfg(test)]
#[inline]
#[must_use]
const fn batched_verify_ssm_desc_pad_seqs(actual_seqs: usize) -> usize {
    batched_verify_ssm_desc_pad_seqs_for_sm(actual_seqs, 0)
}

/// Zero-extend every staged SSM layer's `GdnFusedMultiSeqDesc` payload from
/// `actual_seqs` up to `batched_verify_ssm_desc_pad_seqs_for_sm(actual_seqs, sm_count)`
/// slots (`<= CROSS_SEQ_TABLE_BYTES`), explicitly overwriting stale trailing
/// slots in both the reused host slab scratch and GPU descriptor buffers.
fn pad_ssm_fused_desc_payloads_for_sm(
    staged: &mut [(DevicePtr, Vec<u8>)],
    actual_seqs: usize,
    sm_count: u32,
) {
    let target_seqs = batched_verify_ssm_desc_pad_seqs_for_sm(actual_seqs, sm_count);
    if target_seqs <= actual_seqs {
        return;
    }
    let target_bytes = target_seqs.saturating_mul(ops::GdnFusedMultiSeqDesc::SIZE_BYTES);
    if target_bytes > crate::layers::qwen3_ssm::CROSS_SEQ_TABLE_BYTES {
        return;
    }
    for (_, bytes) in staged.iter_mut() {
        if bytes.len() < target_bytes {
            bytes.resize(target_bytes, 0);
        }
    }
}

#[cfg(test)]
fn pad_ssm_fused_desc_payloads(staged: &mut [(DevicePtr, Vec<u8>)], actual_seqs: usize) {
    pad_ssm_fused_desc_payloads_for_sm(staged, actual_seqs, 0);
}

/// Look up a cached batched-verify CUDA graph (Layer-0 prefix or Layer-1..
/// suffix), preferring an exact `(slots, meta)` match and falling back to the
/// smallest cached superset arity bucket for `sm_count` when
/// `allow_superset_arity` holds.
fn lookup_cached_batched_verify_graph_for_sm(
    graphs: &HashMap<BatchedVerifyGraphKey, GraphHandle>,
    primary_key: &BatchedVerifyGraphKey,
    actual_seqs: usize,
    allow_superset_arity: bool,
    sm_count: u32,
) -> Option<(BatchedVerifyGraphKey, GraphHandle)> {
    if let Some(graph) = graphs.get(primary_key).copied().filter(|g| g.0 != 0) {
        return Some((primary_key.clone(), graph));
    }
    let max_bucket = batched_verify_bucket_max_seqs_for_sm(sm_count);
    if !allow_superset_arity || !(2..max_bucket).contains(&actual_seqs) {
        return None;
    }
    let mut candidate_key: BatchedVerifyGraphKey = (vec![0usize], primary_key.1);
    for &bucket_seqs in batched_verify_seq_buckets_for_sm(sm_count) {
        if bucket_seqs <= actual_seqs {
            continue;
        }
        candidate_key.0[0] = QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL.saturating_sub(bucket_seqs);
        if let Some(graph) = graphs.get(&candidate_key).copied().filter(|g| g.0 != 0) {
            return Some((candidate_key, graph));
        }
    }
    None
}

#[cfg(test)]
fn lookup_cached_batched_verify_graph(
    graphs: &HashMap<BatchedVerifyGraphKey, GraphHandle>,
    primary_key: &BatchedVerifyGraphKey,
    actual_seqs: usize,
    allow_superset_arity: bool,
) -> Option<(BatchedVerifyGraphKey, GraphHandle)> {
    lookup_cached_batched_verify_graph_for_sm(
        graphs,
        primary_key,
        actual_seqs,
        allow_superset_arity,
        0,
    )
}

/// Whether `BatchedVerifyTcDenseGuard` should arm for a cross-sequence verify
/// frame of `total_rows`.
///
/// Decouples W4A8 compact TC MoE (`DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS = 4`,
/// e.g. `C=2, K=2` with `r=4`) from the `LmHead` dense TC threshold
/// (`min_rows = 4` by default, or higher when overridden): every dense TC
/// consumer (`LmHead`, `PleProj`, `HcPrepare`, `FinalMixer`, `Router`,
/// `Fp8Dense`, `coalesced_nvfp4`) enforces its own component floor inside the
/// scope, while `TC_MOE` can engage at `r >= 4` whenever TC dense is not
/// globally disabled.
#[inline]
#[must_use]
const fn batched_verify_tc_scope_active(
    tc_dense_active: bool,
    tc_dense_enabled: bool,
    total_rows: usize,
) -> bool {
    tc_dense_active
        || (tc_dense_enabled
            && total_rows
                >= crate::layers::moe::qwen4_prefill_compact::DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS)
}

/// Tail sampling mode baked into a batched verify suffix CUDA graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchedVerifyTailMode {
    Argmax,
    Gumbel,
    Rejection,
}

/// Coarse power-of-2 block-table stride buckets (`32 / 128 / 512 / full`).
///
/// Bucket 0 (`32` blocks) covers `0..=2048` tokens at `block_size = 64` (the
/// entire Qwen4 dense-attention CUDA-graph regime) in a single fixed bucket
/// with zero graph recaptures while cutting `rows = 80` per-round block-table
/// uploads from `655 KB` (`mb = 2049`) down to `10 KB` (`mb = 32`).
const QWEN4_BT_STRIDE_BUCKETS: [usize; 3] = [32, 128, 512];

/// Select `(effective_mb, bucket_id)` (`bucket_id in 0..=3`, 2 bits) for a
/// cross-sequence verify frame's active block-table width.
#[inline]
#[must_use]
const fn select_block_table_stride_bucket(
    max_active_blocks: usize,
    max_blocks_per_seq: usize,
) -> (usize, usize) {
    if max_blocks_per_seq == 0 {
        return (0, 3);
    }
    let needed = if max_active_blocks == 0 {
        1
    } else {
        max_active_blocks
    };
    if needed <= QWEN4_BT_STRIDE_BUCKETS[0] && QWEN4_BT_STRIDE_BUCKETS[0] < max_blocks_per_seq {
        (QWEN4_BT_STRIDE_BUCKETS[0], 0)
    } else if needed <= QWEN4_BT_STRIDE_BUCKETS[1]
        && QWEN4_BT_STRIDE_BUCKETS[1] < max_blocks_per_seq
    {
        (QWEN4_BT_STRIDE_BUCKETS[1], 1)
    } else if needed <= QWEN4_BT_STRIDE_BUCKETS[2]
        && QWEN4_BT_STRIDE_BUCKETS[2] < max_blocks_per_seq
    {
        (QWEN4_BT_STRIDE_BUCKETS[2], 2)
    } else {
        (max_blocks_per_seq, 3)
    }
}

/// Key cached suffix graphs by `(slots_or_arity, total_rows, all_dense_attn,
/// tc_dense_active, argmax_rows_active, unified_tail_active,
/// rejection_tail_active, bt_stride_bucket)` rather than per-sequence `k_s`,
/// so ONE cached CUDA graph covers every ragged partition, every slot
/// permutation, and every per-row sampling mode summing to the same
/// `(num_seqs, total_rows)` within each block-table stride bucket.
///
/// Bit layout of `meta` (`usize`):
/// - bits `0..=9`:   `total_rows & 0x3FF` (10 bits: `0..=1023`, so `256` never aliases `0`)
/// - bit  `10`:      `all_dense_attn`
/// - bit  `11`:      `tc_dense_active`
/// - bit  `12`:      `argmax_rows_active` (suffix) / `embed_fused` (layer-0 prefix)
/// - bit  `13`:      `unified_tail_active` (Gumbel)
/// - bit  `14`:      `rejection_tail_active` (Wave-15 rejection)
/// - bits `15..=16`: `bt_stride_bucket & 0x3` (`0..=3`)
/// - bit  `17`:      `QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT`
/// - bit  `62`:      `QWEN4_BATCHED_VERIFY_LAYER0_GRAPH_TAG` (prefix only)
/// - bit  `63`:      `QWEN4_BATCHED_VERIFY_GRAPH_TAG`
#[allow(clippy::too_many_arguments)]
#[inline]
fn batched_verify_graph_key_with_tail_and_bt_bucket(
    slots: Vec<usize>,
    total_rows: usize,
    all_dense_attn: bool,
    tc_dense_active: bool,
    argmax_rows_active: bool,
    unified_tail_active: bool,
    rejection_tail_active: bool,
    bt_stride_bucket: usize,
) -> BatchedVerifyGraphKey {
    let meta = QWEN4_BATCHED_VERIFY_GRAPH_TAG
        | (total_rows & 0x3FF)
        | (usize::from(all_dense_attn) << 10)
        | (usize::from(tc_dense_active) << 11)
        | (usize::from(argmax_rows_active) << 12)
        | (usize::from(unified_tail_active) << 13)
        | (usize::from(rejection_tail_active) << 14)
        | ((bt_stride_bucket & 0x3) << 15);
    (slots, meta)
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn batched_verify_graph_key_with_tail(
    slots: Vec<usize>,
    total_rows: usize,
    all_dense_attn: bool,
    tc_dense_active: bool,
    argmax_rows_active: bool,
    unified_tail_active: bool,
    rejection_tail_active: bool,
) -> BatchedVerifyGraphKey {
    batched_verify_graph_key_with_tail_and_bt_bucket(
        slots,
        total_rows,
        all_dense_attn,
        tc_dense_active,
        argmax_rows_active,
        unified_tail_active,
        rejection_tail_active,
        0,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn batched_verify_graph_key(
    slots: Vec<usize>,
    total_rows: usize,
    all_dense_attn: bool,
    tc_dense_active: bool,
    argmax_rows_active: bool,
    unified_tail_active: bool,
) -> BatchedVerifyGraphKey {
    batched_verify_graph_key_with_tail(
        slots,
        total_rows,
        all_dense_attn,
        tc_dense_active,
        argmax_rows_active,
        unified_tail_active,
        false,
    )
}

/// Key cached Layer-0 prefix (`embed -> Layer 0`) graphs by
/// `(slots_or_arity, total_rows, all_dense_attn, tc_dense_active, embed_fused,
/// bt_stride_bucket)` with both `QWEN4_BATCHED_VERIFY_GRAPH_TAG` and
/// `QWEN4_BATCHED_VERIFY_LAYER0_GRAPH_TAG` set so prefix graphs never alias
/// suffix graphs or `decode_a2` entries yet inherit all shared invalidation.
#[inline]
fn batched_verify_layer0_graph_key_with_bt_bucket(
    slots: Vec<usize>,
    total_rows: usize,
    all_dense_attn: bool,
    tc_dense_active: bool,
    embed_fused: bool,
    bt_stride_bucket: usize,
) -> BatchedVerifyGraphKey {
    let meta = QWEN4_BATCHED_VERIFY_GRAPH_TAG
        | QWEN4_BATCHED_VERIFY_LAYER0_GRAPH_TAG
        | (total_rows & 0x3FF)
        | (usize::from(all_dense_attn) << 10)
        | (usize::from(tc_dense_active) << 11)
        | (usize::from(embed_fused) << 12)
        | ((bt_stride_bucket & 0x3) << 15);
    (slots, meta)
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn batched_verify_layer0_graph_key(
    slots: Vec<usize>,
    total_rows: usize,
    all_dense_attn: bool,
    tc_dense_active: bool,
    embed_fused: bool,
) -> BatchedVerifyGraphKey {
    batched_verify_layer0_graph_key_with_bt_bucket(
        slots,
        total_rows,
        all_dense_attn,
        tc_dense_active,
        embed_fused,
        0,
    )
}

/// Template spec for default 1-pass 0-ULP greedy argmax (`temp == 0.0`) in the
/// unified `target_gumbel_verify_rows_bf16` suffix tail.
const GREEDY_VERIFY_TAIL_ROW_TEMPLATE: spark_runtime::sampler::device_topk::GumbelVerifyRowSpec =
    spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
        row_idx: 0,
        seed: 0,
        abs_pos: 0,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
    };

/// Fill `buf[..num_rows]` with contiguous `temp == 0.0` greedy argmax specs so
/// un-staged or deduped verify frames execute inside the exact same unified
/// `target_gumbel_verify_rows_bf16` suffix graph with zero heap allocation.
#[inline]
fn fill_greedy_verify_tail_specs(
    buf: &mut [spark_runtime::sampler::device_topk::GumbelVerifyRowSpec;
             spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS],
    num_rows: usize,
) -> &[spark_runtime::sampler::device_topk::GumbelVerifyRowSpec] {
    for (row_idx, slot) in buf.iter_mut().take(num_rows).enumerate() {
        *slot = spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
            row_idx,
            ..GREEDY_VERIFY_TAIL_ROW_TEMPLATE
        };
    }
    &buf[..num_rows]
}

/// Template spec for default 1-pass 0-ULP greedy argmax (`temp == 0.0`) in the
/// unified `target_rejection_verify_rows_bf16` suffix tail.
const GREEDY_REJECTION_TAIL_ROW_TEMPLATE: super::device_topk::RejectionVerifyRowSpec =
    super::device_topk::RejectionVerifyRowSpec {
        row_idx: 0,
        seed: 0,
        abs_pos: 0,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
        draft_token: 0,
        q_row_idx: super::device_topk::REJECTION_BONUS_Q_ROW_IDX,
        mask_tok0: super::device_topk::REJECTION_NO_MASK_TOKEN,
        mask_tok1: super::device_topk::REJECTION_NO_MASK_TOKEN,
    };

/// Fill `buf[..num_rows]` with contiguous `temp == 0.0` greedy rejection specs
/// so unstaged verify frames share the exact same rejection suffix CUDA graph
/// with zero heap allocation.
#[inline]
fn fill_greedy_rejection_tail_specs(
    buf: &mut [super::device_topk::RejectionVerifyRowSpec; super::device_topk::REJECTION_MAX_ROWS],
    num_rows: usize,
) -> &[super::device_topk::RejectionVerifyRowSpec] {
    for (row_idx, slot) in buf.iter_mut().take(num_rows).enumerate() {
        *slot = super::device_topk::RejectionVerifyRowSpec {
            row_idx,
            ..GREEDY_REJECTION_TAIL_ROW_TEMPLATE
        };
    }
    &buf[..num_rows]
}

/// Eager warmup hits required before paying `cudaGraphInstantiate` for
/// transient non-power-of-2 draining cohorts (`seqs` outside `{1, 2, 4, 8, 16}`).
const QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS: u32 = 3;

/// Maximum distinct key entries retained in `warmed_hits` before resetting
/// stale non-slot-agnostic counters.
const QWEN4_BATCHED_VERIFY_WARMED_HITS_CAP: usize = 1024;

#[inline]
#[must_use]
fn with_qsa_dense_prefix_graph_bit(
    mut key: BatchedVerifyGraphKey,
    qsa_dense_prefix: bool,
) -> BatchedVerifyGraphKey {
    if qsa_dense_prefix {
        key.1 |= QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT;
    }
    key
}

/// True when a cohort with at least one `qwen4_qsa_required` sequence stays
/// strictly inside the pre-2048 dense-attention prefix (`seq_len + k <= 2048`)
/// with uniform QSA journaling (`verify_journal_rows(k)`) and pre-allocated
/// multi-seq QSA staging, mirroring `verify_d::qwen4_qsa_dense_prefix_graph_eligible`.
fn batched_verify_qsa_dense_prefix_eligible<I>(
    num_leaders: usize,
    has_verify_meta_dev: bool,
    has_qsa_scratch: bool,
    leaders: I,
) -> bool
where
    I: IntoIterator<Item = (bool, usize, usize)>,
{
    if num_leaders == 0
        || num_leaders > crate::layers::qwen4_qsa::QSA_CROSS_SEQ_MAX_SEQS
        || !has_verify_meta_dev
        || !has_qsa_scratch
    {
        return false;
    }
    let mut any_qsa = false;
    for (qsa_required, seq_len, k) in leaders {
        if !qsa_required {
            continue;
        }
        any_qsa = true;
        if !crate::layers::qwen4_qsa::verify_journal_rows(k) {
            return false;
        }
        if seq_len
            .checked_add(k)
            .is_none_or(|end| end > crate::layers::qwen4_qsa::VERIFY_ROWS_DENSE_ENDPOINT)
        {
            return false;
        }
    }
    any_qsa
}

/// Minimum eager warmup passes required before capturing a suffix CUDA graph
/// for `num_seqs` sequences. Steady-state power-of-2 cohorts (`1, 2, 4, 8, 16`)
/// capture on their existing threshold (after 1 warmup hit); transient draining
/// cohort sizes (`3, 5..=7, 9..=15, >16`) require 3 warmup hits so 1-2 step
/// drain shapes run eager (~1 ms) instead of paying ~100 ms graph instantiation.
#[inline]
#[must_use]
pub(super) const fn batched_verify_graph_required_warmup_hits(num_seqs: usize) -> u32 {
    match num_seqs {
        1 | 2 | 4 | 8 | 16 => 1,
        _ => QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS,
    }
}

/// SM-aware warmup hit count before CUDA graph capture: wide GPUs (`> 64` SMs)
/// capture after 1 warmup hit for all steady cohort arities
/// (`1, 2, 4, 8, 16, 24, 32, 48, 64`) while keeping the 3-hit transient drain
/// threshold for odd draining sizes (`23, 21, 17, 29, ...`) so draining tails
/// do not pay wasted `cuGraphInstantiate` latency; `<= 64`-SM GPUs (GB10) stay
/// unchanged.
#[inline]
#[must_use]
const fn batched_verify_graph_required_warmup_hits_for_sm(num_seqs: usize, sm_count: u32) -> u32 {
    if sm_count > QWEN4_VERIFY_WIDE_SM_THRESHOLD {
        match num_seqs {
            1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 => 1,
            _ => QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS,
        }
    } else {
        batched_verify_graph_required_warmup_hits(num_seqs)
    }
}

#[inline]
fn is_batched_verify_graph_meta(meta: usize) -> bool {
    (meta & QWEN4_BATCHED_VERIFY_GRAPH_TAG) != 0
}

/// True when a frame's `(is_ragged, num_seqs, total_rows)` should be admitted
/// to CUDA graph capture. Uniform frames always admit; ragged frames admit
/// bucket-aligned row counts (`step = 2/4/8/16`, matching
/// `qwen4_mtp::ragged_row_bucket_step`) plus the `K=2` floor (`2 * num_seqs`),
/// skipping one-shot tail-clamped row counts near `max_tokens` so they run
/// eager without wasting `cudaGraphInstantiate` latency or evicting hot graphs.
#[inline]
#[must_use]
const fn batched_verify_graph_capture_row_admitted(
    is_ragged: bool,
    num_seqs: usize,
    total_rows: usize,
) -> bool {
    if !is_ragged {
        return true;
    }
    let step = match num_seqs {
        0..=1 => 1,
        2 => 2,
        3..=5 => 4,
        6..=16 => 8,
        _ => 16,
    };
    total_rows.is_multiple_of(step) || total_rows == num_seqs.saturating_mul(2)
}

#[derive(Default)]
struct BatchedVerifyGraphModelState {
    /// Cohort key whose SSM `CrossSeqPtrTable` descriptors were most recently
    /// warmed in eager mode (used when a layer does not support Phase-1 staging).
    last_warm_key: Option<BatchedVerifyGraphKey>,
    /// Eager warmup hit counts per graph key (pre-allocating arenas, cuBLASLt
    /// plans, and SSM descriptor tables, while gating transient draining cohort
    /// sizes behind `QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS`).
    warmed_hits: HashMap<BatchedVerifyGraphKey, u32>,
    /// LRU order of captured batched-verify graph keys (oldest at front).
    lru: VecDeque<BatchedVerifyGraphKey>,
    /// Consecutive capture failures; poisoning engages only after repeated
    /// failures (`QWEN4_BATCHED_VERIFY_CAPTURE_POISON_LIMIT`) so a single
    /// isolated shape rejection cannot disable graphs for the entire process.
    consecutive_capture_failures: u32,
    /// Permanently disables capture for this model instance after repeated
    /// capture failures, guaranteeing graceful eager fallback without churn.
    capture_poisoned: bool,
}

impl BatchedVerifyGraphModelState {
    fn is_warm(
        &self,
        key: &BatchedVerifyGraphKey,
        all_suffix_ssm_staged: bool,
        required_hits: u32,
    ) -> bool {
        let hits = self.warmed_hits.get(key).copied().unwrap_or(0);
        if all_suffix_ssm_staged {
            hits >= required_hits
        } else {
            self.last_warm_key.as_ref() == Some(key) && hits >= required_hits
        }
    }

    fn record_warmup_hit(&mut self, key: BatchedVerifyGraphKey) {
        if !self.warmed_hits.contains_key(&key)
            && self.warmed_hits.len() >= QWEN4_BATCHED_VERIFY_WARMED_HITS_CAP
        {
            self.warmed_hits.clear();
        }
        let hits = self.warmed_hits.entry(key.clone()).or_insert(0);
        *hits = hits.saturating_add(1);
        self.last_warm_key = Some(key);
    }

    fn record_layer0_warmup_hit(&mut self, key: BatchedVerifyGraphKey) {
        if !self.warmed_hits.contains_key(&key)
            && self.warmed_hits.len() >= QWEN4_BATCHED_VERIFY_WARMED_HITS_CAP
        {
            self.warmed_hits.clear();
        }
        let hits = self.warmed_hits.entry(key).or_insert(0);
        *hits = hits.saturating_add(1);
    }

    fn record_captured(&mut self, key: BatchedVerifyGraphKey, required_hits: u32) {
        self.touch(&key);
        let hits = self.warmed_hits.entry(key.clone()).or_insert(0);
        *hits = (*hits).max(required_hits);
        self.last_warm_key = Some(key);
        self.consecutive_capture_failures = 0;
    }

    fn record_layer0_captured(&mut self, key: BatchedVerifyGraphKey, required_hits: u32) {
        self.touch(&key);
        let hits = self.warmed_hits.entry(key).or_insert(0);
        *hits = (*hits).max(required_hits);
        self.consecutive_capture_failures = 0;
    }

    fn touch(&mut self, key: &BatchedVerifyGraphKey) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    /// Prune keys already removed by `free_sequence`/`compact_sequence` and
    /// return the oldest key(s) to evict so inserting a new entry stays within
    /// `cap`, never evicting `preserve_key` (e.g. the Layer-0 prefix graph
    /// launched or captured earlier in the same verify step).
    fn collect_evictions_preserving(
        &mut self,
        live_keys: &HashMap<BatchedVerifyGraphKey, GraphHandle>,
        cap: usize,
        preserve_key: Option<&BatchedVerifyGraphKey>,
    ) -> Vec<BatchedVerifyGraphKey> {
        self.lru.retain(|k| live_keys.contains_key(k));
        let mut evict = Vec::new();
        let target = cap.saturating_sub(1);
        let mut idx = 0;
        while self.lru.len() > target && idx < self.lru.len() {
            if preserve_key.is_some_and(|pk| &self.lru[idx] == pk) {
                idx += 1;
                continue;
            }
            if let Some(oldest) = self.lru.remove(idx) {
                evict.push(oldest);
            }
        }
        evict
    }

    /// Prune keys already removed by `free_sequence`/`compact_sequence` and
    /// return the oldest key(s) to evict so inserting a new entry stays within `cap`.
    #[cfg(test)]
    fn collect_evictions(
        &mut self,
        live_keys: &HashMap<BatchedVerifyGraphKey, GraphHandle>,
        cap: usize,
    ) -> Vec<BatchedVerifyGraphKey> {
        self.collect_evictions_preserving(live_keys, cap, None)
    }
}

static BATCHED_VERIFY_GRAPH_TRACKER: std::sync::OnceLock<
    parking_lot::Mutex<HashMap<BatchedVerifyTrackerId, BatchedVerifyGraphModelState>>,
> = std::sync::OnceLock::new();

/// Recycle an empty `Vec<Qwen4CrossSeqRows<'a>>` heap allocation across layer
/// iterations with distinct borrow lifetimes, eliminating 48 per-layer heap
/// allocations per verify frame with zero `unsafe`.
fn recycle_views_vec<'a, 'b>(mut v: Vec<Qwen4CrossSeqRows<'a>>) -> Vec<Qwen4CrossSeqRows<'b>> {
    v.clear();
    v.into_iter()
        .map(|_| unreachable!("cleared Vec is empty"))
        .collect()
}

const DEDUP_NO_LEADER: u8 = 0xFF;
const DEDUP_MAX_SLOTS: usize = 255;
static DEDUP_LEADER_OF_SLOT: [AtomicU8; DEDUP_MAX_SLOTS] =
    [const { AtomicU8::new(DEDUP_NO_LEADER) }; DEDUP_MAX_SLOTS];
static DEDUP_FOLLOWER_COMMIT_SKIP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub(in crate::model) fn reset_dedup_slot(slot_idx: usize) {
    if let Some(entry) = DEDUP_LEADER_OF_SLOT.get(slot_idx) {
        entry.store(DEDUP_NO_LEADER, Ordering::Release);
    }
    if slot_idx < DEDUP_MAX_SLOTS {
        DEDUP_FOLLOWER_COMMIT_SKIP[slot_idx / 64]
            .fetch_and(!(1u64 << (slot_idx % 64)), Ordering::Release);
    }
}

pub(in crate::model) fn dedup_leader_of_slot(slot_idx: usize) -> Option<usize> {
    let val = DEDUP_LEADER_OF_SLOT.get(slot_idx)?.load(Ordering::Acquire);
    (val != DEDUP_NO_LEADER).then_some(val as usize)
}

fn set_dedup_follower(slot_idx: usize, leader_slot: usize) {
    if let (Some(entry), true) = (
        DEDUP_LEADER_OF_SLOT.get(slot_idx),
        leader_slot < DEDUP_MAX_SLOTS && slot_idx < DEDUP_MAX_SLOTS,
    ) {
        entry.store(leader_slot as u8, Ordering::Release);
        DEDUP_FOLLOWER_COMMIT_SKIP[slot_idx / 64]
            .fetch_or(1u64 << (slot_idx % 64), Ordering::Release);
    }
}

pub(in crate::model) fn take_dedup_follower_commit_skip(slot_idx: usize) -> bool {
    if slot_idx >= DEDUP_MAX_SLOTS {
        return false;
    }
    let mask = 1u64 << (slot_idx % 64);
    (DEDUP_FOLLOWER_COMMIT_SKIP[slot_idx / 64].fetch_and(!mask, Ordering::AcqRel) & mask) != 0
}

fn batched_k5_dedup_enabled() -> bool {
    let read_live = || {
        std::env::var("ATLAS_QWEN4_BATCHED_K5_DEDUP")
            .ok()
            .as_deref()
            == Some("1")
    };
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Cached wrapper around [`super::verify_d::k5_hybrid_selected`] (avoids
/// per-admission `std::env::var` syscalls outside unit tests).
#[inline]
fn k5_hybrid_selected_cached() -> bool {
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(super::verify_d::k5_hybrid_selected)
    }
    #[cfg(test)]
    {
        super::verify_d::k5_hybrid_selected()
    }
}

/// Cached wrapper around [`crate::layers::qwen3_ssm::ssm_h_fp16_enabled`]
/// (avoids per-admission `std::env::var` syscalls outside unit tests).
#[inline]
fn ssm_h_fp16_enabled_cached() -> bool {
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(crate::layers::qwen3_ssm::ssm_h_fp16_enabled)
    }
    #[cfg(test)]
    {
        crate::layers::qwen3_ssm::ssm_h_fp16_enabled()
    }
}

/// Cached check that `ATLAS_KGAMMA_DEBUG_DUMP` is unset (avoids per-admission
/// `std::env::var` syscalls outside unit tests).
#[inline]
fn kgamma_debug_dump_absent() -> bool {
    let read_live = || std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_err();
    #[cfg(not(test))]
    {
        static ABSENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ABSENT.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Cached wrapper around [`crate::model::verify_logits_receipt::selected`]
/// (preserves fail-closed parse errors while avoiding per-round syscalls).
#[inline]
fn verify_logits_receipt_selected_cached() -> Result<bool> {
    let read_live =
        || crate::model::verify_logits_receipt::selected().map_err(|e| format!("{e:#}"));
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_live().map_err(anyhow::Error::msg)
    }
}

/// Reasons a frame is refused before any effect. Pure, host-only checks over
/// the per-sequence facts the model can see. Admits both uniform (`K2..=K9`)
/// and ragged mixed-`K` (`K_s in 1..=9`, `sum(K_s) <= 256`, `seqs in 2..=64`)
/// frames.
pub(crate) fn validate_frame_inputs(
    tokens_per_seq: &[Vec<u32>],
    slots: &[usize],
    lanes: usize,
    vocab: usize,
) -> std::result::Result<Qwen4CrossSeqFrame, &'static str> {
    if tokens_per_seq.len() != slots.len() {
        return Err("cross-sequence verify token windows and sequences differ in count");
    }
    if slots.len() < 2 {
        return Err("cross-sequence verify needs at least two sequences");
    }
    let mut ks_stack = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let ks_heap;
    let ks: &[usize] = if tokens_per_seq.len() <= ks_stack.len() {
        for (dst, window) in ks_stack.iter_mut().zip(tokens_per_seq.iter()) {
            *dst = window.len();
        }
        &ks_stack[..tokens_per_seq.len()]
    } else {
        ks_heap = tokens_per_seq.iter().map(Vec::len).collect::<Vec<_>>();
        &ks_heap
    };
    let frame = Qwen4CrossSeqFrame::new_ragged(ks)?;
    if tokens_per_seq
        .iter()
        .flatten()
        .any(|&token| token as usize >= vocab)
    {
        return Err("cross-sequence verify token is outside the embedding vocabulary");
    }
    let mut seen_bits = [0u64; 4];
    let mut seen_overflow: Option<std::collections::HashSet<usize>> = None;
    for &slot in slots {
        let is_new = if slot < 256 {
            let word = &mut seen_bits[slot >> 6];
            let mask = 1u64 << (slot & 63);
            let fresh = (*word & mask) == 0;
            *word |= mask;
            fresh
        } else {
            seen_overflow
                .get_or_insert_with(|| std::collections::HashSet::with_capacity(slots.len()))
                .insert(slot)
        };
        if !is_new {
            return Err("cross-sequence verify requires one distinct SSM slot per sequence");
        }
    }
    if lanes <= 1 {
        return Err("cross-sequence verify requires per-slot verify lanes");
    }
    if slots.iter().any(|&slot| slot >= lanes) {
        return Err("cross-sequence verify slot has no verify lane");
    }
    Ok(frame)
}

/// Metadata bytes the frame uploads into its dedicated disjoint verify-metadata
/// staging buffer: the fixed 256-byte-spaced (`rows <= 32`), 1024-byte-spaced
/// (`33..=128`), or 2048-byte-spaced (`129..=256`) positions/slots/seq_lens
/// ABI plus one block-table row per verify row.
pub(crate) fn frame_metadata_bytes(rows: usize, max_blocks: usize) -> Option<usize> {
    let header = if rows.checked_mul(8)? <= 256 {
        768
    } else if rows.checked_mul(8)? <= 1024 {
        3072
    } else if rows.checked_mul(8)? <= 2048 {
        6144
    } else {
        return None;
    };
    rows.checked_mul(max_blocks)?
        .checked_mul(4)?
        .checked_add(header)
}

/// Prefix of `buffers.scratch()` written by `ffn.forward_prefill` MoE routing
/// for `total_rows = P + D` rows (`[0 .. 2 * total_rows * top_k * 4)` bytes).
pub(crate) fn moe_routing_scratch_bytes(total_rows: usize, top_k: usize) -> Option<usize> {
    total_rows.checked_mul(top_k)?.checked_mul(8)
}

/// Returns `true` iff `[a_base, a_base + a_bytes)` and `[b_base, b_base + b_bytes)`
/// do not overlap and neither interval overflows `u64`.
pub(crate) fn ranges_disjoint(a_base: u64, a_bytes: usize, b_base: u64, b_bytes: usize) -> bool {
    if a_bytes == 0 || b_bytes == 0 {
        return true;
    }
    let Some(a_end) = a_base.checked_add(a_bytes as u64) else {
        return false;
    };
    let Some(b_end) = b_base.checked_add(b_bytes as u64) else {
        return false;
    };
    a_end <= b_base || b_end <= a_base
}

/// Verify that a decode verify-metadata buffer `[meta_base, meta_base + meta_bytes)`
/// is non-null (when non-empty) and strictly disjoint from `scratch`.
pub(crate) fn ensure_verify_meta_disjoint_from_scratch(
    meta_base: DevicePtr,
    meta_bytes: usize,
    scratch_base: DevicePtr,
    scratch_bytes: usize,
    label: &str,
) -> Result<()> {
    if meta_bytes == 0 {
        return Ok(());
    }
    ensure!(
        !meta_base.is_null(),
        "{label} verify metadata device pointer is null for {meta_bytes} B"
    );
    let disjoint = ranges_disjoint(meta_base.0, meta_bytes, scratch_base.0, scratch_bytes);
    ensure!(
        disjoint,
        "{label} verify metadata [0x{:x}..+0x{:x}) aliases buffers.scratch() [0x{:x}..+0x{:x})",
        meta_base.0,
        meta_bytes,
        scratch_base.0,
        scratch_bytes
    );
    Ok(())
}

/// Persistent host staging buffers reused across `qwen4_cross_seq_upload_metadata`
/// calls so wide frames (`rows * max_blocks_per_seq * 4` ≈ 5 MiB) avoid per-step
/// `mmap`/`munmap` heap churn and full-buffer page-fault zeroing.
#[derive(Default)]
struct VerifyMetaHostScratch {
    positions: Vec<u32>,
    slots: Vec<i64>,
    seq_lens: Vec<i32>,
    block_tables: Vec<i32>,
    /// Per-row high-water column count written since the last stride reset;
    /// every entry at `row * last_mb + col` with `col >= dirty_cols_per_row[row]`
    /// is guaranteed to be `0`.
    dirty_cols_per_row: Vec<usize>,
    last_mb: usize,
    qsa_desc_bytes: Vec<u8>,
}

impl VerifyMetaHostScratch {
    fn prepare(&mut self, rows: usize, mb: usize, qsa_desc_len: usize) -> Result<()> {
        let bt_elems = rows
            .checked_mul(mb)
            .context("cross-sequence verify block table extent overflow")?;
        self.positions.clear();
        self.positions.reserve(rows);
        self.slots.clear();
        self.slots.reserve(rows);
        self.seq_lens.clear();
        self.seq_lens.reserve(rows);
        if self.last_mb != mb {
            if self.block_tables.len() < bt_elems {
                self.block_tables.resize(bt_elems, 0);
            }
            self.block_tables.fill(0);
            self.dirty_cols_per_row.clear();
            self.last_mb = mb;
        } else if self.block_tables.len() < bt_elems {
            self.block_tables.resize(bt_elems, 0);
        }
        if self.dirty_cols_per_row.len() < rows {
            self.dirty_cols_per_row.resize(rows, 0);
        }
        self.qsa_desc_bytes.clear();
        self.qsa_desc_bytes.resize(qsa_desc_len, 0);
        Ok(())
    }

    fn write_block_table_row(&mut self, row: usize, mb: usize, blocks: &[u32]) {
        let row_base = row * mb;
        let written = blocks.len().min(mb);
        let dst = &mut self.block_tables[row_base..row_base + mb];
        for (j, &block) in blocks.iter().enumerate().take(written) {
            dst[j] = block as i32;
        }
        let prev_dirty = self.dirty_cols_per_row[row];
        if written < prev_dirty {
            dst[written..prev_dirty.min(mb)].fill(0);
        }
        self.dirty_cols_per_row[row] = written;
    }
}

static VERIFY_META_HOST_SCRATCH: std::sync::OnceLock<parking_lot::Mutex<VerifyMetaHostScratch>> =
    std::sync::OnceLock::new();

/// Persistent host staging buffer reused across `stage_all_ssm_fused_descs`
/// calls to pack all 36 SSM layers' `GdnFusedMultiSeqDesc` tables into ONE
/// contiguous slab H2D transfer (`36 -> 1` H2D).
#[derive(Default)]
struct SsmFusedDescHostScratch {
    packed: Vec<u8>,
    runs: Vec<(DevicePtr, usize, usize)>,
}

static SSM_FUSED_DESC_HOST_SCRATCH: std::sync::OnceLock<
    parking_lot::Mutex<SsmFusedDescHostScratch>,
> = std::sync::OnceLock::new();

/// Pack per-layer `(dest_dev_ptr, desc_bytes)` payloads into `packed` and
/// coalesce layers belonging to a contiguous `[N][slot_stride]` GPU slab into
/// ONE upload run (`36 × 1024 B` -> 1 contiguous H2D).
///
/// Consecutive sorted layers merge into a single run iff their device addresses
/// are spaced by exactly `slot_stride` (so skipped layers are never overwritten)
/// and each payload fits within `slot_stride`. Disjoint non-slab allocations
/// remain separate upload runs.
fn pack_ssm_fused_desc_slab_runs(
    staged: &mut [(DevicePtr, Vec<u8>)],
    slot_stride: usize,
    packed: &mut Vec<u8>,
    runs: &mut Vec<(DevicePtr, usize, usize)>,
) {
    runs.clear();
    if staged.is_empty() {
        return;
    }
    staged.sort_unstable_by_key(|(dev, _)| dev.0);
    let mut run_ranges: Vec<(usize, usize)> = Vec::new();
    let mut total_bytes = 0usize;
    let mut run_start_idx = 0usize;
    while run_start_idx < staged.len() {
        let mut run_end_idx = run_start_idx + 1;
        if slot_stride > 0 && staged[run_start_idx].1.len() <= slot_stride {
            while run_end_idx < staged.len() {
                let prev_dev = staged[run_end_idx - 1].0.0;
                let cur_dev = staged[run_end_idx].0.0;
                let cur_len = staged[run_end_idx].1.len();
                if cur_len <= slot_stride
                    && prev_dev.checked_add(slot_stride as u64) == Some(cur_dev)
                {
                    run_end_idx += 1;
                } else {
                    break;
                }
            }
        }
        let run_base = staged[run_start_idx].0;
        let last = &staged[run_end_idx - 1];
        let span = (last.0.0 - run_base.0) as usize + last.1.len();
        runs.push((run_base, total_bytes, span));
        run_ranges.push((run_start_idx, run_end_idx));
        total_bytes += span;
        run_start_idx = run_end_idx;
    }
    if packed.len() < total_bytes {
        packed.resize(total_bytes, 0);
    }
    for (&(run_base, buf_start, _span), &(start_idx, end_idx)) in runs.iter().zip(run_ranges.iter())
    {
        for (dev, bytes) in &staged[start_idx..end_idx] {
            let off = buf_start + (dev.0 - run_base.0) as usize;
            packed[off..off + bytes.len()].copy_from_slice(bytes);
        }
    }
}

/// Whether the active proposer + DFlash capture configuration is admitted on
/// `tp_rank` for cross-sequence batched Qwen4 verification.
#[inline]
#[must_use]
pub(crate) fn cross_seq_proposer_admitted(
    tp_rank: usize,
    proposer_kind: Option<crate::speculative::DraftProposerKind>,
    dflash_hidden_save_allocated: bool,
    dflash_sample_method: crate::dflash_sampling::DflashDraftSampleMethod,
) -> bool {
    if tp_rank > 0 {
        return true;
    }
    match proposer_kind {
        Some(crate::speculative::DraftProposerKind::Qwen4NativeMtp) => true,
        Some(crate::speculative::DraftProposerKind::DFlash) => {
            dflash_hidden_save_allocated
                && dflash_sample_method
                    != crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
        }
        _ => false,
    }
}

/// Whether the cross-sequence verify frame admits the active LM-head route:
/// wide frames (`tc_dense_frame == true`, `rows >= min_rows`) admit Tensor Core
/// LM-head execution including `ATLAS_LM_HEAD_TC=1`; small frames (`rows < 4`
/// by default, or `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE=0`) keep the exact
/// refusal of `ATLAS_LM_HEAD_TC=1`.
#[inline]
#[must_use]
pub(crate) const fn lm_head_route_admitted_for_frame(
    tc_dense_frame: bool,
    lm_head_tc_active: bool,
) -> bool {
    tc_dense_frame || !lm_head_tc_active
}

/// Predict the next verify step's row-0 `(bonus_token, committed_prior_len)`
/// for one sequence under greedy prefix acceptance so its offloaded PLE rows
/// can pre-warm asynchronously across commit and MTP drafting.
#[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
#[inline]
fn predict_next_bonus_ple_input(
    window: &[u32],
    verified: &[u32],
    pre_verify_len: usize,
) -> Option<(u32, usize)> {
    let k = window.len();
    if k == 0 || verified.len() < k || verified[..k].contains(&u32::MAX) {
        return None;
    }
    let accepted = window[1..]
        .iter()
        .zip(&verified[..k - 1])
        .take_while(|&(draft, target)| draft == target)
        .count();
    let bonus = *verified.get(accepted)?;
    let committed_len = pre_verify_len.checked_add(accepted + 1)?;
    Some((bonus, committed_len))
}

/// Predict the next verify step's row-0 `(bonus_token, committed_prior_len)`
/// for one sequence under maximal-coupling rejection verification by walking
/// packed accept bits (`words[i].is_accepted()`), returning `None` if any row
/// in the window requested host policy fallback.
#[cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]
#[inline]
fn predict_next_bonus_ple_input_rejection(
    window: &[u32],
    words: &[super::device_topk::PackedRejectionWord],
    reqs: &[super::device_topk::RejectionVerifyRowSpec],
    pre_verify_len: usize,
) -> Option<(u32, usize)> {
    let k = window.len();
    if k == 0
        || words.len() < k
        || reqs.len() < k
        || words[..k].iter().any(|w| w.is_host_fallback())
    {
        return None;
    }
    let accepted = (0..k - 1)
        .take_while(|&i| {
            if reqs[i].temperature == 0.0 {
                words[i].is_accepted() || words[i].token_id() == window[i + 1]
            } else {
                words[i].is_accepted()
            }
        })
        .count();
    let bonus = words.get(accepted)?.token_id();
    let committed_len = pre_verify_len.checked_add(accepted + 1)?;
    Some((bonus, committed_len))
}

/// Read-only KV block ceiling and pool-budget preflight for a cross-sequence
/// verify cohort.
///
/// Checks every sequence's final logical block `(seq_len + k_s - 1) / block_size`
/// against `max_blocks_per_seq` and verifies total new physical KV blocks
/// required across the cohort does not exceed `free_blocks + evictable_prefix_blocks`.
/// Returns `total_new_blocks` on success.
fn check_batched_verify_kv_ceiling(
    seq_lens: &[usize],
    block_table_lens: &[usize],
    ks: &[usize],
    block_size: usize,
    max_blocks_per_seq: usize,
    free_blocks: usize,
    evictable_prefix_blocks: usize,
) -> Result<usize> {
    ensure!(
        block_size > 0,
        "cross-sequence verify requires positive KV block_size"
    );
    ensure!(
        seq_lens.len() == block_table_lens.len() && seq_lens.len() == ks.len(),
        "cross-sequence verify KV ceiling check length mismatch"
    );
    let mut total_new_blocks = 0usize;
    for i in 0..seq_lens.len() {
        let k_s = ks[i];
        ensure!(
            k_s > 0,
            "cross-sequence verify sequence {i} has zero verify width"
        );
        let last_pos = seq_lens[i]
            .checked_add(k_s - 1)
            .context("cross-sequence verify last KV position overflow")?;
        let last_block = last_pos / block_size;
        ensure!(
            last_block < max_blocks_per_seq,
            "cross-sequence verify sequence {i} requires logical block {last_block} >= max_blocks_per_seq {max_blocks_per_seq}"
        );
        let needed_blocks = last_block + 1;
        let new_blocks = needed_blocks.saturating_sub(block_table_lens[i]);
        total_new_blocks = total_new_blocks
            .checked_add(new_blocks)
            .context("cross-sequence verify total new KV blocks overflow")?;
    }
    let available_blocks = free_blocks.saturating_add(evictable_prefix_blocks);
    ensure!(
        total_new_blocks <= available_blocks,
        "cross-sequence verify needs {total_new_blocks} new KV blocks \
         (free={free_blocks}, evictable_prefix={evictable_prefix_blocks})"
    );
    Ok(total_new_blocks)
}

/// Pure predicate for whether `layer_idx` has an active DFlash target-hidden
/// capture tap on this rank (requiring `hidden` to be fully materialized in
/// DRAM before `try_dflash_capture_rows` reads it).
#[inline]
#[must_use]
fn layer_has_dflash_capture_tap_raw(
    has_dflash_save: bool,
    is_rank0: bool,
    capture_layers: &[usize],
    layer_idx: usize,
) -> bool {
    has_dflash_save && is_rank0 && capture_layers.contains(&layer_idx)
}

/// Pure predicate for whether `layer_idx` may defer its trailing MLP
/// `inject_saved_batched` into the following layer's `attn_hyper` (or
/// `qwen4_final_mixer` after the last layer).
///
/// Layer 0 MUST always flush before exiting so the Layer-0 prefix CUDA graph
/// leaves `hidden` materialized in DRAM before Layer-0->Layer-1 PLE and the
/// Layer-1 suffix CUDA graph. Any DFlash capture tap layer MUST also flush so
/// `try_dflash_capture_rows` reads the post-MLP `hidden` state.
#[inline]
#[must_use]
fn layer_should_defer_mlp_tail(
    fuse_enabled: bool,
    layer_idx: usize,
    has_dflash_save: bool,
    is_rank0: bool,
    capture_layers: &[usize],
) -> bool {
    fuse_enabled
        && layer_idx > 0
        && !layer_has_dflash_capture_tap_raw(has_dflash_save, is_rank0, capture_layers, layer_idx)
}

impl TransformerModel {
    #[inline]
    fn layer_has_dflash_capture_tap(&self, layer_idx: usize) -> bool {
        layer_has_dflash_capture_tap_raw(
            self.dflash_hidden_save.is_some(),
            self.comm.as_ref().is_none_or(|c| c.rank() == 0),
            &self.dflash_capture_layers,
            layer_idx,
        )
    }

    /// Best-effort asynchronous pre-warming of offloaded PLE rows for a flat
    /// token slice (delegated from `Model::warm_ple_draft_ahead`).
    pub(super) fn warm_ple_draft_ahead_dispatch(&self, tokens: &[u32]) {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = self.qwen4_ple.as_ref() {
            ple.submit_draft_ahead_tokens(tokens);
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let _ = tokens;
    }

    /// Best-effort asynchronous pre-warming of offloaded PLE rows for
    /// per-sequence `(window_tokens, prior_tokens)` pairs (delegated from
    /// `Model::warm_ple_draft_ahead_seqs`).
    ///
    /// When batched verify PLE is enabled and `seqs` packs into a valid
    /// cross-sequence verify key, submits a reusable keyed `Verify` prefetch
    /// so the immediately following `decode_verify_qwen4_batched` adopts it
    /// directly without preemption or a second NVMe read.
    pub(super) fn warm_ple_draft_ahead_seqs_dispatch(&self, seqs: &[(&[u32], &[u32])]) {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = self.qwen4_ple.as_ref() {
            if crate::layers::qwen4_ple::ple_verify_batch_selected()
                && let Some(warm_seqs) = crate::layers::qwen4_ple::PleWarmSeq::pack_batch(seqs)
            {
                ple.submit_verify_prefetch(warm_seqs);
            } else {
                ple.submit_draft_ahead_seqs(seqs);
            }
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let _ = seqs;
    }

    /// Complete read-only admission (no device or host state changes).
    pub(super) fn qwen4_cross_seq_admission(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &[&SequenceState],
    ) -> Result<Qwen4CrossSeqFrame> {
        self.ensure_verify_healthy()?;
        crate::layers::qwen4_verify_lanes::require_fixed_k5_policy()?;
        ensure!(
            crate::layers::qwen4_verify_lanes::batched_k5_selected()?,
            "cross-sequence verify requires ATLAS_QWEN4_BATCHED_K5=1"
        );
        ensure!(
            self.qwen4_cross_seq_staging.is_some(),
            "cross-sequence verify lanes were not provisioned at model construction"
        );
        let mut slots_stack = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
        let slots_heap;
        let slots: &[usize] = if seqs.len() <= slots_stack.len() {
            for (dst, seq) in slots_stack.iter_mut().zip(seqs.iter()) {
                *dst = seq.slot_idx;
            }
            &slots_stack[..seqs.len()]
        } else {
            slots_heap = seqs.iter().map(|seq| seq.slot_idx).collect::<Vec<_>>();
            &slots_heap
        };
        let frame = validate_frame_inputs(
            tokens_per_seq,
            slots,
            crate::layers::qwen4_verify_lanes::lanes(),
            self.config.vocab_size,
        )
        .map_err(anyhow::Error::msg)?;
        let min_k = frame.min_k();
        let max_k = frame.max_k();
        let adaptive = crate::layers::qwen4_verify_lanes::adaptive_k_selected()?;
        if adaptive {
            crate::layers::qwen4_verify_lanes::require_adaptive_k_policy()?;
        }
        if frame.is_ragged() || min_k == 1 {
            ensure!(
                adaptive && self.comm.is_none(),
                "ragged or K1 cross-sequence verify requires TP=1 with ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1"
            );
            ensure!(
                crate::layers::qwen4_cross_seq_mlp::moe_tiling()?
                    != crate::layers::qwen4_cross_seq_mlp::MoeTiling::Sequence,
                "ragged or K1 cross-sequence verify requires frame MoE tiling \
                 (unset ATLAS_QWEN4_BATCHED_K5_MOE_TILE=seq)"
            );
        }
        for s in 0..frame.seqs() {
            let k_s = frame.seq_k(s);
            ensure!(
                crate::traits::verify_owner::batch::width_admitted(k_s, adaptive),
                "owned native batch requires fixed K5 (K1..K8 with ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1), got K{k_s}"
            );
        }
        ensure!(
            self.ssm_pool.num_ssm_layers == 0 || self.ssm_pool.num_intermediates >= max_k,
            "cross-sequence verify K{max_k} requires at least {max_k} SSM intermediate slots (pool has {})",
            self.ssm_pool.num_intermediates
        );
        ensure!(
            min_k > 1
                || self.ssm_pool.num_ssm_layers == 0
                || !self.ssm_pool.compact_lazy_h
                || crate::layers::qwen3_ssm::replay_ssm_enabled(),
            "cross-sequence verify K1 under compact_lazy_h requires ATLAS_QWEN4_REPLAY_SSM!=0"
        );
        for (s, seq) in seqs.iter().enumerate() {
            let k_s = frame.seq_k(s);
            ensure!(
                !seq.qwen4_qsa_required
                    || k_s == 1
                    || crate::layers::qwen4_qsa::verify_journal_rows(k_s),
                "cross-sequence verify K{k_s} with active QSA requires QSA journaling for K{k_s}"
            );
        }
        ensure!(
            self.config.is_qwen4_exp()
                && self.config.hc_count == 4
                && self.config.hidden_size == 2_560
                && self.config.residual_width() == 10_240
                && !self.config.use_fp32_residual(),
            "cross-sequence verify requires the canonical BF16 Flash-Next geometry"
        );
        ensure!(
            (self.config.tp_world_size <= 4
                && (self.config.ep_world_size <= 1
                    || self.config.ep_world_size == self.config.tp_world_size)
                && self.config.pp_world_size <= 1)
                || (self.config.tp_world_size == 1
                    && self.config.ep_world_size == 1
                    && self.config.pp_world_size >= 2),
            "cross-sequence verify supports TP1..TP4 (+ matching EP) with PP1, or PP>=2 with TP1/EP1"
        );
        let active_proposer = self.active_proposer();
        let active_proposer_kind = active_proposer.map(|proposer| proposer.kind());
        let dflash_sample_method = active_proposer.map_or(
            crate::dflash_sampling::DflashDraftSampleMethod::Greedy,
            |proposer| proposer.dflash_draft_sample_method(),
        );
        ensure!(
            cross_seq_proposer_admitted(
                self.config.tp_rank.max(self.config.pp_rank),
                active_proposer_kind,
                self.dflash_hidden_save.is_some(),
                dflash_sample_method,
            ),
            "cross-sequence verify pairs only with the native Qwen4 MTP or non-probabilistic DFlash proposer"
        );
        if self.config.tp_rank == 0
            && self.config.pp_rank == 0
            && active_proposer_kind == Some(crate::speculative::DraftProposerKind::DFlash)
        {
            ensure!(
                frame.rows() <= self.dflash_hidden_save_rows,
                "cross-sequence DFlash verify frame rows {} exceed dflash_hidden_save_rows {}",
                frame.rows(),
                self.dflash_hidden_save_rows
            );
            for seq in seqs {
                let dstate = seq
                    .proposer_state
                    .as_deref()
                    .and_then(|ps| {
                        ps.as_any()
                            .downcast_ref::<crate::layers::DflashProposerState>()
                    })
                    .context("cross-sequence DFlash verify sequence lacks DflashProposerState")?;
                ensure!(
                    dstate.ctx_len == seq.seq_len,
                    "cross-sequence DFlash verify sequence on slot {} has unreconciled ctx_len {} != seq_len {}",
                    seq.slot_idx,
                    dstate.ctx_len,
                    seq.seq_len
                );
            }
        }
        let hybrid_rows_ok = if frame.is_ragged() {
            let mut checked_ks = 0u16;
            let mut all_ok = true;
            for s in 0..frame.seqs() {
                let k_s = frame.seq_k(s);
                let k_bit = if k_s < 16 { 1u16 << k_s } else { 0 };
                if k_bit != 0 && (checked_ks & k_bit) != 0 {
                    continue;
                }
                if !crate::layers::qwen4_hybrid_cross_seq_rows(k_s) {
                    all_ok = false;
                    break;
                }
                checked_ks |= k_bit;
            }
            all_ok
        } else {
            crate::layers::qwen4_hybrid_cross_seq_rows(max_k)
        };
        ensure!(
            k5_hybrid_selected_cached() && hybrid_rows_ok,
            "cross-sequence verify K{min_k}..=K{max_k} requires the qualified exact hybrid route"
        );
        ensure!(
            !ssm_h_fp16_enabled_cached(),
            "cross-sequence verify requires FP32 recurrent state; unset ATLAS_SSM_H_FP16"
        );
        ensure!(
            !crate::layers::qwen4_mtp_rows::recondition_engaged(),
            "cross-sequence verify does not share the global K7 recondition note"
        );
        // Diagnostics that rely on the single-sequence frame stay there.
        let serial = crate::model::env_diag::DflashSerialControls::current();
        ensure!(
            serial.active_family()?.is_none()
                && !crate::layers::attn_paged_serial_enabled()
                && !crate::layers::attn_out_serial_enabled()
                && !crate::model::k1_stage_diag::enabled()
                && !super::verify_d::layer_resolution_probe_enabled()
                && !crate::full_profile::is_enabled()
                && !self.profile
                && kgamma_debug_dump_absent(),
            "cross-sequence verify refuses serial, stage, probe and profile diagnostics"
        );
        // Flat chain only: no tree payload, DFS order or ancestor indirection.
        ensure!(
            self.ddtree_parent_ids_dev.lock().is_none()
                && *self.ddtree_num_tree_tokens.lock() == 0
                && self.ddtree_parent_ids_host.lock().is_empty(),
            "cross-sequence verify requires a flat candidate chain"
        );
        // BF16 full-vocabulary logits on the NVFP4 LM head; small frames
        // (`rows < min_rows` or TC dense disabled) preserve exact scalar GEMV
        // and refuse `ATLAS_LM_HEAD_TC=1`, while frames meeting the `LmHead`
        // threshold (`rows >= 4` by default) admit Tensor Core LM-head execution.
        let tc_dense_frame = crate::layers::qwen4_hyper::batched_verify_tc_component_active(
            crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
            frame.rows(),
            false,
        );
        let lm_head_tc_active = super::super::impl_a3::lm_head_tc_enabled()
            && self.lm_head_nvfp4_t.is_some()
            && self.w4a16_gemm_t_m32_n64_kernel.0 != 0;
        ensure!(
            !self.use_fp32_logits
                && self.verify_lmhead_vocab() as usize == self.config.vocab_size
                && self.lm_head_nvfp4.is_some()
                && lm_head_route_admitted_for_frame(tc_dense_frame, lm_head_tc_active),
            "cross-sequence verify requires exact BF16 full-vocabulary NVFP4 logits \
             (ATLAS_LM_HEAD_TC=1 requires batched verify TC dense)"
        );
        let final_mixer = self
            .qwen4_final_mixer
            .as_ref()
            .context("cross-sequence verify is missing the final hyperconnection mixer")?;
        final_mixer.preflight_qwen4_native_batched(frame.rows(), false)?;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.qwen4_ple.is_some() {
            ensure!(
                self.layers.len() > 1 && self.config.ple_layer_ids.as_slice() == [2],
                "cross-sequence verify PLE requires the canonical layer-2 boundary"
            );
        }
        ensure!(
            self.layers.len() == self.config.num_hidden_layers
                && (0..self.layers.len()).all(|index| {
                    matches!(
                        self.config.layer_type(index),
                        LayerType::FullAttention | LayerType::LinearAttention
                    )
                }),
            "cross-sequence verify found an unsupported layer schedule"
        );
        for (s, seq) in seqs.iter().enumerate() {
            let k_s = frame.seq_k(s);
            ensure!(
                seq.layer_states.len() == self.layers.len(),
                "cross-sequence verify sequence on slot {} lacks layer state",
                seq.slot_idx
            );
            seq.rotary_positions.tail_scalar(seq.seq_len)?;
            let last = seq
                .seq_len
                .checked_add(k_s - 1)
                .context("cross-sequence verify physical extent overflow")?;
            seq.rotary_positions.tail_scalar(last)?;
        }
        let n = seqs.len();
        let max_blocks = self.max_blocks_per_seq as usize;
        {
            let kv_cache = self.kv_cache.lock();
            ensure!(
                kv_cache.config().cache_blocks_per_seq.is_none(),
                "cross-sequence verify requires resident KV without high-speed swap"
            );
            let mut seq_lens = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            let mut block_table_lens = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            let mut ks = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            for s in 0..n {
                seq_lens[s] = seqs[s].seq_len;
                block_table_lens[s] = seqs[s].block_table.len();
                ks[s] = frame.seq_k(s);
            }
            let evictable_prefix_blocks = if self.prefix_cache.is_active() {
                self.prefix_cache.stats().1
            } else {
                0
            };
            let _ = check_batched_verify_kv_ceiling(
                &seq_lens[..n],
                &block_table_lens[..n],
                &ks[..n],
                kv_cache.block_size(),
                max_blocks,
                kv_cache.num_free_blocks(),
                evictable_prefix_blocks,
            )?;
        }
        let _metadata_bytes = frame_metadata_bytes(frame.rows(), max_blocks)
            .context("cross-sequence verify metadata exceeds the 256-row ABI")?;
        let arenas = self.buffers.sizes();
        let hyper_bytes = frame.rows() * self.config.residual_width() * 2;
        ensure!(
            frame.rows() <= self.buffers.max_batch_tokens()
                && hyper_bytes <= arenas.hidden_states
                && hyper_bytes <= arenas.residual
                && frame.rows() * self.config.vocab_size * 2 <= arenas.logits,
            "cross-sequence verify rows exceed the serving arenas"
        );

        // Read-only walk of every layer before the first effect.
        let _ragged_scope =
            crate::layers::Qwen4CrossSeqRaggedScopeGuard::enter(frame.is_ragged() || min_k == 1);
        let mut qsa_required_buf = [false; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
        for (dst, seq) in qsa_required_buf.iter_mut().zip(seqs.iter()) {
            *dst = seq.qwen4_qsa_required;
        }
        let qsa_required = &qsa_required_buf[..n];
        let ctx = self.qwen4_cross_seq_ctx(None);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            ensure!(
                layer.qwen4_cross_seq_admits_frames(),
                "cross-sequence verify layer {layer_index} has a poisoned pointer table; \
                 verifying sequences one at a time"
            );
            layer
                .preflight_qwen4_cross_seq_verify(frame, qsa_required, &ctx)
                .with_context(|| {
                    format!("cross-sequence verify layer {layer_index} failed read-only preflight")
                })?;
        }
        Ok(frame)
    }

    pub(super) fn qwen4_cross_seq_ctx(
        &self,
        attn_metadata: Option<AttnMetadataDev>,
    ) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata,
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        }
    }

    /// Lazily allocate (or grow) the dedicated decode-verify attention metadata
    /// device buffer that stays strictly disjoint from `buffers.scratch()`.
    pub(super) fn ensure_disjoint_verify_attn_meta(
        &self,
        required_bytes: usize,
    ) -> Result<(DevicePtr, usize)> {
        let mut guard = self.qwen4_verify_attn_meta.lock();
        if let Some((ptr, cap)) = *guard
            && cap >= required_bytes
            && !ptr.is_null()
        {
            return Ok((ptr, cap));
        }
        let default_cap = frame_metadata_bytes(
            crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS,
            self.max_blocks_per_seq as usize,
        )
        .unwrap_or(required_bytes);
        let alloc_bytes = required_bytes.max(default_cap).max(4096);
        let new_ptr = self
            .gpu
            .alloc(alloc_bytes)
            .context("allocate disjoint Qwen4 verify attention metadata buffer")?;
        if let Some((old_ptr, _)) = guard.replace((new_ptr, alloc_bytes))
            && !old_ptr.is_null()
        {
            self.gpu.synchronize(self.gpu.default_stream())?;
            self.invalidate_batched_verify_graphs();
            let _ = self.gpu.free(old_ptr);
        }
        Ok((new_ptr, alloc_bytes))
    }

    fn batched_verify_tracker_id(&self) -> BatchedVerifyTrackerId {
        (
            std::ptr::from_ref(self) as usize,
            self.buffers.hidden_states().0,
        )
    }

    /// Destroy all cached batched-verify suffix graphs for this model (called
    /// if `qwen4_verify_attn_meta` ever reallocates after a prior capture).
    fn invalidate_batched_verify_graphs(&self) {
        let stale: Vec<GraphHandle> = {
            let mut graphs = self.batch_decode_graphs.lock();
            let keys: Vec<BatchedVerifyGraphKey> = graphs
                .keys()
                .filter(|(_, meta)| is_batched_verify_graph_meta(*meta))
                .cloned()
                .collect();
            keys.into_iter()
                .filter_map(|k| graphs.remove(&k))
                .filter(|g| g.0 != 0)
                .collect()
        };
        for graph in stale {
            if let Err(err) = self.gpu.destroy_graph(graph) {
                tracing::error!(
                    "verify_qwen4_batched: destroy_graph({}) on meta realloc failed: {err:#}",
                    graph.0
                );
            }
        }
        if let Some(tracker) = BATCHED_VERIFY_GRAPH_TRACKER.get()
            && let Some(state) = tracker.lock().get_mut(&self.batched_verify_tracker_id())
        {
            state.last_warm_key = None;
            state.warmed_hits.clear();
            state.lru.clear();
        }
    }

    /// Enforce the UMA-headroom-aware LRU cap on batched-verify graphs strictly
    /// on cache misses prior to `CaptureGuard::begin` (zero `/proc/meminfo`
    /// syscalls on hot hits).
    ///
    /// Never evicts `preserve_key` (e.g. the Layer-0 prefix graph launched or
    /// captured earlier in the same verify step) and synchronizes `stream` once
    /// before destroying any evicted handles.
    fn evict_batched_verify_graphs_for_capture(
        &self,
        model_id: BatchedVerifyTrackerId,
        preserve_key: Option<&BatchedVerifyGraphKey>,
        stage: BatchedVerifyGraphStage,
        stream: u64,
    ) -> Result<()> {
        let base_cap = batched_verify_graph_base_cap_for_sm(atlas_core::device::active_sm_count());
        let floor_bytes = batched_verify_mem_floor_bytes();
        let mem_available_bytes =
            floor_bytes.and_then(|_| spark_runtime::startup_ledger::mem_available_bytes());
        let effective_cap =
            effective_batched_verify_graph_cap(base_cap, mem_available_bytes, floor_bytes);

        let (evicted_graphs, remaining_tracked): (Vec<GraphHandle>, usize) = {
            let mut graphs = self.batch_decode_graphs.lock();
            let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
            let mut map = tracker.lock();
            let state = map.entry(model_id).or_default();
            let evict_keys =
                state.collect_evictions_preserving(&graphs, effective_cap, preserve_key);
            let handles = evict_keys
                .into_iter()
                .filter_map(|k_old| graphs.remove(&k_old))
                .filter(|g| g.0 != 0)
                .collect();
            (handles, state.lru.len())
        };

        let evicted_count = evicted_graphs.len();
        if evicted_count > 0 {
            self.gpu.synchronize(stream)?;
            for old_graph in evicted_graphs {
                if let Err(err) = self.gpu.destroy_graph(old_graph) {
                    tracing::error!(
                        "verify_qwen4_batched: destroy evicted graph {} failed: {err:#}",
                        old_graph.0
                    );
                }
            }
            if effective_cap < base_cap {
                tracing::info!(
                    stage = stage.as_str(),
                    evicted = evicted_count,
                    effective_cap,
                    base_cap,
                    remaining = remaining_tracked,
                    mem_available_mib =
                        ?mem_available_bytes.map(|b| b / spark_runtime::startup_ledger::MIB),
                    floor_mib = ?floor_bytes.map(|b| b / spark_runtime::startup_ledger::MIB),
                    "BATCHED_VERIFY_GRAPH_UMA_EVICTED"
                );
            }
        }
        Ok(())
    }

    /// Resolve the Qwen4 batched embedding kernel handle(s) strictly outside
    /// CUDA graph capture so Layer-0 prefix capture performs zero module
    /// lookups or allocations.
    pub(super) fn resolve_qwen4_batched_embed_kernels(&self) -> Result<Qwen4BatchedEmbedKernels> {
        if Self::qwen4_embed_expand_fused_selected() {
            let kernel = self
                .gpu
                .kernel("qwen4_hyper", "qwen4_batched_embed_expand")?;
            Ok(Qwen4BatchedEmbedKernels::Fused(kernel))
        } else {
            let gather = self.gpu.kernel("embed_from_argmax", "batched_embed")?;
            let expand = self
                .gpu
                .kernel("qwen4_hyper", "qwen4_hc_expand_embedding")?;
            Ok(Qwen4BatchedEmbedKernels::Split { gather, expand })
        }
    }

    /// Upload `tokens` (`u32` IDs) into `buffers.scratch()` strictly outside
    /// CUDA graph capture and return that fixed device pointer for
    /// [`Self::launch_qwen4_batched_embed_from_dev`].
    pub(super) fn upload_qwen4_batched_embed_tokens(
        &self,
        tokens: &[u32],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.preflight_embed_qwen4_batch(tokens.len())?;
        // SAFETY: `tokens` is an initialized `&[u32]` slice of `tokens.len() * 4`
        // bytes; `copy_h2d_group_fast` finishes reading the host slice into its
        // pinned ring (or synchronous H2D) before returning.
        let token_ids_bytes =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), tokens.len() * 4) };
        let token_ids_dev = self.buffers.scratch();
        self.copy_h2d_group_fast(
            &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
            stream,
        )?;
        Ok(token_ids_dev)
    }

    /// Graph-capture-safe device-only Qwen4 batched embedding gather + HC
    /// expansion reading pre-staged token IDs from `token_ids_dev` and writing
    /// bit-identical BF16 rows to `output`.
    pub(super) fn launch_qwen4_batched_embed_from_dev(
        &self,
        kernels: Qwen4BatchedEmbedKernels,
        token_ids_dev: DevicePtr,
        output: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        match kernels {
            Qwen4BatchedEmbedKernels::Fused(kernel) => {
                spark_runtime::kernel_args::KernelLaunch::new(self.gpu.as_ref(), kernel)
                    .grid([num_tokens as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(token_ids_dev)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(output)
                    .arg_u32(h as u32)
                    .arg_u32(self.config.hc_count as u32)
                    .launch(stream)
            }
            Qwen4BatchedEmbedKernels::Split { gather, expand } => {
                let staged = self.buffers.norm_output();
                ops::batched_embed(
                    self.gpu.as_ref(),
                    gather,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    staged,
                    num_tokens as u32,
                    h as u32,
                    stream,
                )?;
                spark_runtime::kernel_args::KernelLaunch::new(self.gpu.as_ref(), expand)
                    .grid([
                        num_tokens as u32,
                        spark_runtime::kernel_args::div_ceil(
                            self.config.residual_width() as u32,
                            256,
                        ),
                        1,
                    ])
                    .block([256, 1, 1])
                    .arg_ptr(staged)
                    .arg_ptr(output)
                    .arg_u32(h as u32)
                    .arg_u32(self.config.hc_count as u32)
                    .launch(stream)
            }
        }
    }

    /// Phase-1 descriptor staging: build every SSM layer's per-sequence
    /// `(row_base, num_tokens)` `GdnFusedMultiSeqDesc` payload for `u_frame`
    /// and upload all layers in ONE batched `copy_h2d_group_fast` before Layer 0
    /// or suffix graph capture/replay.
    ///
    /// Returns `Ok((layer0_ssm_staged, all_suffix_ssm_staged))` indicating
    /// whether Layer 0 (`layer_idx == 0`) and every suffix SSM layer
    /// (`layer_idx >= 1`) staged cleanly for slot-agnostic ragged graph replay.
    #[allow(clippy::too_many_arguments)]
    fn stage_all_ssm_fused_descs(
        &self,
        u_frame: Qwen4CrossSeqFrame,
        seqs: &mut [&mut SequenceState],
        unique_leaders: &[usize],
        leader_of: &[usize],
        metadata: &[AttnMetadataDev],
        ctx: &ForwardContext<'_>,
        views_pool: &mut Vec<Qwen4CrossSeqRows<'static>>,
        stream: u64,
    ) -> Result<(bool, bool)> {
        if self.ssm_pool.num_ssm_layers == 0 {
            return Ok((true, true));
        }
        let mut staged: Vec<(DevicePtr, Vec<u8>)> =
            Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut layer0_ssm_staged = true;
        let mut all_suffix_ssm_staged = true;
        let mut ssm_layer_idx = 0usize;
        for layer_idx in 0..self.layers.len() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            let mut views: Vec<Qwen4CrossSeqRows<'_>> =
                recycle_views_vec(std::mem::take(views_pool));
            for (seq, &meta) in seqs
                .iter_mut()
                .enumerate()
                .filter(|&(s, _)| unique_leaders[leader_of[s]] == s)
                .map(|(_, seq)| seq)
                .zip(metadata.iter())
            {
                let slot_idx = seq.slot_idx;
                let seq_len = seq.seq_len;
                let h_intermediate = self.ssm_pool.h_intermediate(ssm_layer_idx, slot_idx, 0);
                let conv_intermediate = self.ssm_pool.conv_intermediate(ssm_layer_idx, slot_idx, 0);
                let SequenceState {
                    layer_states,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    ..
                } = &mut **seq;
                views.push(Qwen4CrossSeqRows {
                    state: layer_states[layer_idx].as_mut(),
                    seq_len,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    h_intermediate,
                    conv_intermediate,
                    h_intermediate_stride: self.ssm_pool.h_intermediate_stride(),
                    conv_intermediate_stride: self.ssm_pool.conv_bytes,
                    metadata: meta,
                    slot_idx,
                });
            }
            let staged_layer = self.layers[layer_idx]
                .stage_qwen4_cross_seq_fused_desc_bytes(u_frame, &views, ctx, stream)
                .with_context(|| {
                    format!("phase-1 stage cross-sequence SSM descriptors layer {layer_idx}")
                });
            *views_pool = recycle_views_vec(views);
            match staged_layer? {
                Some(entry) => staged.push(entry),
                None => {
                    if layer_idx == 0 {
                        layer0_ssm_staged = false;
                    } else {
                        all_suffix_ssm_staged = false;
                    }
                }
            }
            ssm_layer_idx += 1;
        }
        if !staged.is_empty() {
            pad_ssm_fused_desc_payloads_for_sm(
                &mut staged,
                u_frame.seqs(),
                atlas_core::device::active_sm_count(),
            );
            let mut scratch = SSM_FUSED_DESC_HOST_SCRATCH
                .get_or_init(|| parking_lot::Mutex::new(SsmFusedDescHostScratch::default()))
                .lock();
            let SsmFusedDescHostScratch { packed, runs } = &mut *scratch;
            pack_ssm_fused_desc_slab_runs(
                &mut staged,
                crate::layers::qwen3_ssm::CROSS_SEQ_TABLE_BYTES,
                packed,
                runs,
            );
            let copies: Vec<HostToDeviceCopy<'_>> = runs
                .iter()
                .map(|&(dst, start, len)| HostToDeviceCopy::new(&packed[start..start + len], dst))
                .collect();
            self.copy_h2d_group_fast(&copies, stream)?;
        }
        Ok((layer0_ssm_staged, all_suffix_ssm_staged))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batched_verify_layer(
        &self,
        layer_idx: usize,
        ssm_layer_idx: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        u_frame: Qwen4CrossSeqFrame,
        seqs: &mut [&mut SequenceState],
        unique_leaders: &[usize],
        leader_of: &[usize],
        metadata: &[AttnMetadataDev],
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext<'_>,
        views_pool: &mut Vec<Qwen4CrossSeqRows<'static>>,
        decode_stages: bool,
        stream: u64,
    ) -> Result<bool> {
        let linear = self.config.layer_type(layer_idx) == LayerType::LinearAttention;
        if decode_stages {
            spark_runtime::cublaslt::stage_timer::set_group(if linear { "ssm" } else { "attn" });
        }
        let mut views: Vec<Qwen4CrossSeqRows<'_>> = recycle_views_vec(std::mem::take(views_pool));
        for (seq, &meta) in seqs
            .iter_mut()
            .enumerate()
            .filter(|&(s, _)| unique_leaders[leader_of[s]] == s)
            .map(|(_, seq)| seq)
            .zip(metadata.iter())
        {
            let slot_idx = seq.slot_idx;
            let seq_len = seq.seq_len;
            let (h_intermediate, conv_intermediate) = if linear {
                (
                    self.ssm_pool.h_intermediate(ssm_layer_idx, slot_idx, 0),
                    self.ssm_pool.conv_intermediate(ssm_layer_idx, slot_idx, 0),
                )
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            let SequenceState {
                layer_states,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ..
            } = &mut **seq;
            views.push(Qwen4CrossSeqRows {
                state: layer_states[layer_idx].as_mut(),
                seq_len,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                h_intermediate,
                conv_intermediate,
                h_intermediate_stride: self.ssm_pool.h_intermediate_stride(),
                conv_intermediate_stride: self.ssm_pool.conv_bytes,
                metadata: meta,
                slot_idx,
            });
        }
        let _l0_guard =
            (layer_idx == 0).then(crate::layers::qwen4_cross_seq_mlp::CrossSeqHcFuseScope::enter);
        let res = self.layers[layer_idx]
            .decode_qwen4_cross_seq_verify(
                hidden, residual, u_frame, &mut views, kv_cache, staging, ctx, stream,
            )
            .with_context(|| format!("cross-sequence verify layer {layer_idx}"));
        *views_pool = recycle_views_vec(views);
        res?;
        if self.layer_has_dflash_capture_tap(layer_idx)
            && let Some(pending) = crate::layers::qwen4_cross_seq_mlp::take_pending_inject()
        {
            let final_mixer = self.qwen4_final_mixer.as_ref().context(
                "cross-sequence verify flush before DFlash capture is missing final mixer",
            )?;
            final_mixer.inject_saved_batched(
                pending.hyper,
                pending.core,
                pending.residual,
                pending.rows,
                self.gpu.as_ref(),
                pending.stream,
            )?;
        }
        self.try_dflash_capture_rows(layer_idx, u_frame.rows(), stream)?;
        Ok(linear)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batched_verify_suffix(
        &self,
        start_ssm_layer_idx: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        u_frame: Qwen4CrossSeqFrame,
        rows: usize,
        seqs: &mut [&mut SequenceState],
        unique_leaders: &[usize],
        leader_of: &[usize],
        metadata: &[AttnMetadataDev],
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext<'_>,
        views_pool: &mut Vec<Qwen4CrossSeqRows<'static>>,
        decode_stages: bool,
        vocab: u32,
        argmax_out: DevicePtr,
        tail_mode: BatchedVerifyTailMode,
        stream: u64,
    ) -> Result<()> {
        let u_rows = u_frame.rows();
        let fuse_inject_gn =
            crate::layers::qwen4_cross_seq_mlp::batched_hc_fuse_inject_gn_enabled();
        let has_dflash_save = self.dflash_hidden_save.is_some();
        let is_rank0 = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
        let mut hc_fuse_scope = crate::layers::qwen4_cross_seq_mlp::CrossSeqHcFuseScope::enter();
        let mut ssm_layer_idx = start_ssm_layer_idx;
        for layer_idx in 1..self.layers.len() {
            let defer_tail = layer_should_defer_mlp_tail(
                fuse_inject_gn,
                layer_idx,
                has_dflash_save,
                is_rank0,
                &self.dflash_capture_layers,
            ) && !(self.config.pp_world_size > 1
                && layer_idx + 1 == self.layers.len());
            hc_fuse_scope.set_defer_mlp_tail(defer_tail);
            if self.run_batched_verify_layer(
                layer_idx,
                ssm_layer_idx,
                hidden,
                residual,
                u_frame,
                seqs,
                unique_leaders,
                leader_of,
                metadata,
                kv_cache,
                staging,
                ctx,
                views_pool,
                decode_stages,
                stream,
            )? {
                ssm_layer_idx += 1;
            }
        }
        hc_fuse_scope.set_defer_mlp_tail(false);
        if self.config.pp_world_size > 1 {
            if let Some(pending) = crate::layers::qwen4_cross_seq_mlp::take_pending_inject() {
                let final_mixer = self
                    .qwen4_final_mixer
                    .as_ref()
                    .context("cross-sequence verify is missing the final hyperconnection mixer")?;
                final_mixer.inject_saved_batched(
                    pending.hyper,
                    pending.core,
                    pending.residual,
                    pending.rows,
                    self.gpu.as_ref(),
                    pending.stream,
                )?;
            }
            self.pp_stage_exit(hidden, u_rows, stream)?;
            if self.config.pp_rank > 0 {
                drop(hc_fuse_scope);
                return Ok(());
            }
        }
        let final_mixer = self
            .qwen4_final_mixer
            .as_ref()
            .context("cross-sequence verify is missing the final hyperconnection mixer")?;
        let normed = match crate::layers::qwen4_cross_seq_mlp::take_pending_inject() {
            Some(pending) if pending.matches(hidden, residual, u_rows, stream) => {
                crate::layers::qwen4_cross_seq_mlp::log_batched_hc_fuse_inject_gn_engaged(u_rows);
                final_mixer.prepare_batched_after_inject(
                    hidden,
                    pending.core,
                    residual,
                    u_rows,
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?
            }
            other => {
                if let Some(pending) = other {
                    final_mixer.inject_saved_batched(
                        pending.hyper,
                        pending.core,
                        pending.residual,
                        pending.rows,
                        self.gpu.as_ref(),
                        pending.stream,
                    )?;
                }
                final_mixer.prepare_batched(
                    hidden,
                    residual,
                    u_rows,
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?
            }
        };
        drop(hc_fuse_scope);
        self.final_norm_rows(normed, normed, u_rows as u32, stream)?;
        if decode_stages {
            spark_runtime::cublaslt::stage_timer::set_group("");
            spark_runtime::cublaslt::stage_timer::mark("final_norm", stream);
        }
        self.lm_head_qwen4_batched_verify_rows(normed, u_rows as u32, stream)?;
        if decode_stages {
            spark_runtime::cublaslt::stage_timer::set_group("");
            spark_runtime::cublaslt::stage_timer::mark("lm_head", stream);
            crate::model::decode_stages::verify_done(rows, seqs[0].seq_len);
        }
        match tail_mode {
            BatchedVerifyTailMode::Rejection => {
                super::device_topk::launch_target_rejection_verify_single_run(
                    self.gpu.as_ref(),
                    self.target_rejection_verify_kernel,
                    self.buffers.logits(),
                    self.target_rejection_q_buf,
                    self.target_rejection_q_buf
                        .offset(super::device_topk::REJECTION_Q_PROBS_OFFSET),
                    self.target_rejection_verify_buf,
                    u_rows,
                    vocab,
                    vocab as usize,
                    super::device_topk::REJECTION_Q_ROW_CAP,
                    super::device_topk::REJECTION_Q_STRIDE,
                    stream,
                )?;
            }
            BatchedVerifyTailMode::Gumbel => {
                super::device_topk::launch_target_gumbel_verify_single_run(
                    self.gpu.as_ref(),
                    self.target_gumbel_verify_kernel,
                    self.buffers.logits(),
                    self.target_gumbel_verify_buf,
                    u_rows,
                    vocab,
                    vocab as usize,
                    stream,
                )?;
            }
            BatchedVerifyTailMode::Argmax => {
                if super::verify_d::verify_argmax_rows_selected() && self.argmax_rows_kernel.0 != 0
                {
                    ops::argmax_bf16_rows(
                        self.gpu.as_ref(),
                        self.argmax_rows_kernel,
                        self.buffers.logits(),
                        argmax_out,
                        vocab,
                        vocab,
                        u_rows as u32,
                        stream,
                    )?;
                } else {
                    for row in 0..u_rows {
                        ops::argmax_bf16(
                            self.gpu.as_ref(),
                            self.argmax_kernel,
                            self.buffers.logits().offset(row * vocab as usize * 2),
                            argmax_out.offset(row * 4),
                            vocab,
                            stream,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Upload every row's flat-chain metadata (identical per row to the
    /// single-sequence verifier) into the disjoint verify-metadata buffer plus
    /// `QsaBatchSeqDesc[0..C]` into `staging.verify_meta_dev` in one
    /// `copy_h2d_group_fast` call, and return one K-row view per sequence.
    pub(super) fn qwen4_cross_seq_upload_metadata(
        &self,
        frame: Qwen4CrossSeqFrame,
        seqs: &[&mut SequenceState],
        unique_leaders: &[usize],
        staging: crate::layer::Qwen4CrossSeqStaging,
        block_size: usize,
        stream: u64,
    ) -> Result<Vec<AttnMetadataDev>> {
        use crate::layers::qwen4_qsa::{
            QSA_DESC_FLAG_JOURNAL, QSA_DESC_FLAG_QSA_REQUIRED, QSA_DESC_FLAG_SPARSE,
            QsaBatchSeqDesc, SCORE_V2_GROUPS_PER_BLOCK, VERIFY_ROWS_DENSE_ENDPOINT,
            verify_journal_rows,
        };
        let rows = frame.rows();
        let mut max_active_blocks = 1usize;
        for (u, &s) in unique_leaders.iter().enumerate() {
            let seq = &seqs[s];
            let k_u = frame.seq_k(u);
            let needed_for_pos = if k_u > 0 && block_size > 0 {
                (seq.seq_len + k_u - 1) / block_size + 1
            } else {
                1
            };
            max_active_blocks = max_active_blocks
                .max(seq.block_table.len())
                .max(needed_for_pos);
        }
        let (mb, _bt_bucket) =
            select_block_table_stride_bucket(max_active_blocks, self.max_blocks_per_seq as usize);
        // Zero-pad QSA descriptors through the active superset bucket ceiling
        // (<= VERIFY_META_BYTES = 64 * 32 B) so trailing padded lanes in
        // superset-arity QSA dense-prefix graphs see `flags = 0` and return.
        let qsa_pad_seqs = batched_verify_ssm_desc_pad_seqs_for_sm(
            unique_leaders.len(),
            atlas_core::device::active_sm_count(),
        )
        .min(staging.verify_meta_bytes / QsaBatchSeqDesc::SIZE_BYTES);
        let qsa_desc_len = qsa_pad_seqs.max(unique_leaders.len()) * QsaBatchSeqDesc::SIZE_BYTES;
        let (slot_off, sl_off, bt_off) = if rows <= 32 {
            (256, 512, 768)
        } else if rows <= 128 {
            (1024, 2048, 3072)
        } else {
            (2048, 4096, 6144)
        };
        let required_meta_bytes = frame_metadata_bytes(rows, mb)
            .context("cross-sequence verify metadata exceeds ABI row cap")?;
        let (meta_base, _meta_cap) = self.ensure_disjoint_verify_attn_meta(required_meta_bytes)?;
        let scratch_base = self.buffers.scratch();
        let scratch_bytes = self.buffers.sizes().scratch;
        ensure_verify_meta_disjoint_from_scratch(
            meta_base,
            required_meta_bytes,
            scratch_base,
            scratch_bytes,
            "decode_attn_meta",
        )?;
        let upload_qsa_descs = !staging.verify_meta_dev.is_null()
            && qsa_desc_len > 0
            && qsa_desc_len <= staging.verify_meta_bytes;
        if upload_qsa_descs {
            ensure_verify_meta_disjoint_from_scratch(
                staging.verify_meta_dev,
                staging.verify_meta_bytes,
                scratch_base,
                scratch_bytes,
                "qsa_verify_meta",
            )?;
        }

        let mut scratch = VERIFY_META_HOST_SCRATCH
            .get_or_init(|| parking_lot::Mutex::new(VerifyMetaHostScratch::default()))
            .lock();
        scratch.prepare(rows, mb, qsa_desc_len)?;
        let mut logits_cursor = 0usize;
        let mut score_block_cursor = 0u32;
        for (u, &s) in unique_leaders.iter().enumerate() {
            let seq = &seqs[s];
            let k_u = frame.seq_k(u);
            let base_u = frame.row_base(u);
            let depths_u = &FLAT_VERIFY_DEPTHS[..k_u];
            seq.rotary_positions.tail_scalar(seq.seq_len)?;
            for &depth in depths_u {
                let pos = seq
                    .seq_len
                    .checked_add(depth)
                    .context("rotary depth overflow")?;
                scratch
                    .positions
                    .push(seq.rotary_positions.tail_scalar(pos)?);
            }
            for t in 0..k_u {
                let pos = seq.seq_len + t;
                let physical_block = seq.physical_block_for(pos / block_size).unwrap_or(0);
                scratch.slots.push(
                    (physical_block as i64) * (block_size as i64) + (pos % block_size) as i64,
                );
                scratch
                    .seq_lens
                    .push(i32::try_from(pos + 1).context("verify seq_len exceeds i32")?);
                let row = base_u + t;
                scratch.write_block_table_row(row, mb, &seq.block_table);
            }
            let sparse = seq.qwen4_qsa_required && seq.seq_len >= VERIFY_ROWS_DENSE_ENDPOINT;
            let mut flags = 0u32;
            if seq.qwen4_qsa_required {
                flags |= QSA_DESC_FLAG_QSA_REQUIRED;
                if verify_journal_rows(k_u) {
                    flags |= QSA_DESC_FLAG_JOURNAL;
                }
            }
            let (logits_offset, logits_stride, seq_blocks) = if sparse {
                flags |= QSA_DESC_FLAG_SPARSE;
                let max_vis = (seq.seq_len + k_u) / 4;
                let blocks = max_vis.div_ceil(SCORE_V2_GROUPS_PER_BLOCK) as u32;
                let off = logits_cursor;
                logits_cursor = logits_cursor.saturating_add(k_u.saturating_mul(max_vis));
                (off as u32, max_vis as u32, blocks)
            } else {
                (0u32, 0u32, 0u32)
            };
            score_block_cursor = score_block_cursor.saturating_add(seq_blocks);
            let desc = QsaBatchSeqDesc {
                row_base: base_u as u32,
                seq_len: seq.seq_len as u32,
                k: k_u as u32,
                journal_lane: seq.slot_idx as u32,
                flags,
                logits_offset,
                logits_stride,
                score_block_end: score_block_cursor,
            };
            let byte_off = u * QsaBatchSeqDesc::SIZE_BYTES;
            desc.write_le_bytes(
                &mut scratch.qsa_desc_bytes[byte_off..byte_off + QsaBatchSeqDesc::SIZE_BYTES],
            );
        }
        // SAFETY: each pointer/length pair below is a live, initialized
        // plain-old-data Vec viewed as its own bytes (`len` never exceeds the
        // Vec's byte length). `copy_h2d_group_fast` copies the bytes into its
        // pinned ring (or runs the ordinary group copy) before returning, and
        // `scratch` stays locked across that call so the views remain valid.
        let as_bytes = |ptr: *const u8, len: usize| unsafe { std::slice::from_raw_parts(ptr, len) };
        let pos_bytes = as_bytes(scratch.positions.as_ptr().cast(), rows * 4);
        let slot_bytes = as_bytes(scratch.slots.as_ptr().cast(), rows * 8);
        let sl_bytes = as_bytes(scratch.seq_lens.as_ptr().cast(), rows * 4);
        let bt_bytes = as_bytes(scratch.block_tables.as_ptr().cast(), rows * mb * 4);
        if upload_qsa_descs {
            self.copy_h2d_group_fast(
                &[
                    HostToDeviceCopy::new(pos_bytes, meta_base),
                    HostToDeviceCopy::new(slot_bytes, meta_base.offset(slot_off)),
                    HostToDeviceCopy::new(sl_bytes, meta_base.offset(sl_off)),
                    HostToDeviceCopy::new(bt_bytes, meta_base.offset(bt_off)),
                    HostToDeviceCopy::new(&scratch.qsa_desc_bytes, staging.verify_meta_dev),
                ],
                stream,
            )?;
        } else {
            self.copy_h2d_group_fast(
                &[
                    HostToDeviceCopy::new(pos_bytes, meta_base),
                    HostToDeviceCopy::new(slot_bytes, meta_base.offset(slot_off)),
                    HostToDeviceCopy::new(sl_bytes, meta_base.offset(sl_off)),
                    HostToDeviceCopy::new(bt_bytes, meta_base.offset(bt_off)),
                ],
                stream,
            )?;
        }
        drop(scratch);
        Ok(unique_leaders
            .iter()
            .enumerate()
            .map(|(u, &s)| {
                let seq = &seqs[s];
                let base = frame.row_base(u);
                AttnMetadataDev {
                    qwen4_qsa_required: seq.qwen4_qsa_required,
                    positions: meta_base.offset(base * 4),
                    positions_h: meta_base.offset(base * 4),
                    positions_w: meta_base.offset(base * 4),
                    slot: meta_base.offset(slot_off + base * 8),
                    seq_len: meta_base.offset(sl_off + base * 4),
                    block_table: meta_base.offset(bt_off + base * mb * 4),
                    max_blocks_per_seq: mb as u32,
                    num_seqs: frame.seq_k(u) as u32,
                }
            })
            .collect())
    }

    pub(super) fn decode_verify_qwen4_batched_dispatch(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let staged_gumbel_reqs = self.staged_verify_gumbel_rows.lock().take();
        *self.in_graph_gumbel_verify_tokens.lock() = None;
        let staged_rejection_reqs = self.staged_verify_rejection_rows.lock().take();
        *self.in_graph_rejection_verify_words.lock() = None;
        crate::model::verify_failure::check()?;
        let views: Vec<&SequenceState> = seqs.iter().map(|seq| &**seq).collect();
        let frame = self.qwen4_cross_seq_admission(tokens_per_seq, &views)?;
        // Admission accepted each lane's width `k_s`; reclaim any evictable
        // prefix-cache blocks and confirm the cohort's full KV demand is free
        // before mutating any sequence's verify frame or SSM state.
        let n = frame.seqs();
        let mut ks = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
        for s in 0..n {
            ks[s] = frame.seq_k(s);
        }
        {
            let mut kv_cache = self.kv_cache.lock();
            let mut seq_lens = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            let mut block_table_lens = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            for s in 0..n {
                seq_lens[s] = seqs[s].seq_len;
                block_table_lens[s] = seqs[s].block_table.len();
            }
            let total_new_blocks = check_batched_verify_kv_ceiling(
                &seq_lens[..n],
                &block_table_lens[..n],
                &ks[..n],
                kv_cache.block_size(),
                self.max_blocks_per_seq as usize,
                kv_cache.num_free_blocks(),
                if self.prefix_cache.is_active() {
                    self.prefix_cache.stats().1
                } else {
                    0
                },
            )?;
            if self.prefix_cache.is_active() && total_new_blocks > kv_cache.num_free_blocks() {
                let deficit = total_new_blocks - kv_cache.num_free_blocks();
                let evicted = self.prefix_cache.evict(deficit);
                super::super::block_mgmt::apply_evicted_blocks(evicted, &mut kv_cache);
            }
            ensure!(
                total_new_blocks <= kv_cache.num_free_blocks(),
                "cross-sequence verify needs {total_new_blocks} new KV blocks but only {} are free after prefix-cache eviction",
                kv_cache.num_free_blocks()
            );
        }
        self.begin_owned_batch_forward_ragged(seqs, &ks[..n])?;
        if self.qwen4_tp_batched_verify_supported_dispatch()
            && self
                .comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
        {
            let notify = (|| -> Result<()> {
                self.ep_broadcast_u32(crate::traits::EP_CMD_VERIFY_QWEN4_BATCHED)?;
                self.ep_broadcast_u32(seqs.len() as u32)?;
                self.ep_broadcast_u32(frame.k() as u32)?;
                let mut packed_tokens: Vec<u32> = Vec::with_capacity(seqs.len() * (4 + frame.k()));
                for (s, seq) in seqs.iter().enumerate() {
                    let tail_len = seq.tokens.len().min(4);
                    let k_s = frame.seq_k(s);
                    let meta = ((seq.slot_idx as u32) & 0xFFFF)
                        | (u32::from(seq.qwen4_qsa_required) << 16)
                        | (((k_s as u32) & 0x7F) << 17)
                        | ((tail_len as u32) << 24);
                    self.ep_broadcast_u32(meta)?;
                    self.ep_broadcast_u32(seq.seq_len as u32)?;
                    packed_tokens.extend_from_slice(&seq.tokens[seq.tokens.len() - tail_len..]);
                }
                for window in tokens_per_seq.iter() {
                    packed_tokens.extend_from_slice(window);
                }
                let _ = self.ep_broadcast_tokens(&packed_tokens)?;
                self.sync_secondary()
            })();
            if let Err(err) = notify {
                self.poison_owned_batch(seqs);
                return Err(err);
            }
        }
        let result = self.decode_verify_qwen4_batched_unowned_inner(
            frame,
            tokens_per_seq,
            seqs,
            staged_gumbel_reqs,
            staged_rejection_reqs,
        );
        if result.is_err() {
            self.poison_owned_batch(seqs);
        } else {
            self.record_owned_batch_execution(seqs);
        }
        result
    }

    pub(in crate::model) fn decode_verify_qwen4_batched_unowned(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let staged_gumbel_reqs = self.staged_verify_gumbel_rows.lock().take();
        *self.in_graph_gumbel_verify_tokens.lock() = None;
        let staged_rejection_reqs = self.staged_verify_rejection_rows.lock().take();
        *self.in_graph_rejection_verify_words.lock() = None;
        self.decode_verify_qwen4_batched_unowned_inner(
            frame,
            tokens_per_seq,
            seqs,
            staged_gumbel_reqs,
            staged_rejection_reqs,
        )
    }

    fn decode_verify_qwen4_batched_unowned_inner(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
        staged_gumbel_reqs: Option<Vec<spark_runtime::sampler::device_topk::GumbelVerifyRowSpec>>,
        staged_rejection_reqs: Option<Vec<super::device_topk::RejectionVerifyRowSpec>>,
    ) -> Result<Vec<Vec<u32>>> {
        let _ragged_scope = crate::layers::Qwen4CrossSeqRaggedScopeGuard::enter(
            frame.is_ragged() || frame.min_k() == 1,
        );
        let staging = self
            .qwen4_cross_seq_staging
            .context("cross-sequence verify staging disappeared")?;
        let rows = frame.rows();
        let stream = self.gpu.default_stream();
        let row_bytes = self.config.residual_width() * 2;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // ── Effects begin: live recurrent/PLE state from each checkpoint ──
        // The flat-tree route flag is single-sequence frame state; a frame
        // without graphs never injects the flat tree, so commit sees none.
        self.dflash_flat_tree_route
            .store(false, std::sync::atomic::Ordering::Release);
        self.ddtree_dfs_inv_perm.lock().clear();

        let receipt_on = verify_logits_receipt_selected_cached()?;
        let dedup_allowed = self.comm.is_none()
            && batched_k5_dedup_enabled()
            && !receipt_on
            && !frame.is_ragged()
            && frame.min_k() >= 2
            && self.dflash_hidden_save.is_none()
            && staged_rejection_reqs.is_none();
        let rotary_tails: Vec<Vec<u32>> = if dedup_allowed {
            let mut tails = Vec::with_capacity(seqs.len());
            for (s, seq) in seqs.iter().enumerate() {
                let depths_s = &FLAT_VERIFY_DEPTHS[..frame.seq_k(s)];
                tails.push(seq.rotary_positions.verify_tail(seq.seq_len, depths_s)?);
            }
            tails
        } else {
            Vec::new()
        };
        let mut unique_leaders: Vec<usize> = Vec::with_capacity(seqs.len());
        let mut leader_of: Vec<usize> = Vec::with_capacity(seqs.len());
        for s in 0..seqs.len() {
            let slot_s = seqs[s].slot_idx;
            let mut matched_u = None;
            if dedup_allowed {
                let prev_leader = dedup_leader_of_slot(slot_s);
                for (u, &l_idx) in unique_leaders.iter().enumerate() {
                    let slot_l = seqs[l_idx].slot_idx;
                    if prev_leader.is_none_or(|prev| prev == slot_l)
                        && seqs[s].seq_len == seqs[l_idx].seq_len
                        && seqs[s].qwen4_qsa_required == seqs[l_idx].qwen4_qsa_required
                        && tokens_per_seq[s] == tokens_per_seq[l_idx]
                        && seqs[s].tokens == seqs[l_idx].tokens
                        && rotary_tails[s] == rotary_tails[l_idx]
                    {
                        matched_u = Some((u, slot_l));
                        break;
                    }
                }
            }
            if let Some((u, slot_l)) = matched_u {
                leader_of.push(u);
                set_dedup_follower(slot_s, slot_l);
            } else {
                let u = unique_leaders.len();
                unique_leaders.push(s);
                leader_of.push(u);
                reset_dedup_slot(slot_s);
            }
        }
        let u_frame = if unique_leaders.len() == seqs.len() {
            frame
        } else {
            let mut u_ks = [0usize; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            for (dst, &s) in u_ks.iter_mut().zip(unique_leaders.iter()) {
                *dst = frame.seq_k(s);
            }
            Qwen4CrossSeqFrame::new_ragged(&u_ks[..unique_leaders.len()])
                .map_err(anyhow::Error::msg)?
        };
        let u_rows = u_frame.rows();
        let tc_dense_active =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(u_rows, false);
        let tc_scope_active = batched_verify_tc_scope_active(
            tc_dense_active,
            crate::layers::qwen4_hyper::batched_verify_tc_dense_enabled(),
            u_rows,
        );
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_scope_active);

        let decode_stages = crate::model::decode_stages::enabled();
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let mut ple_warm_ticket = if let Some(ple) = self.qwen4_ple.as_ref()
            && crate::layers::qwen4_ple::ple_verify_batch_selected()
        {
            let warm_seqs: Vec<crate::layers::qwen4_ple::PleWarmSeq> = unique_leaders
                .iter()
                .map(|&s| {
                    crate::layers::qwen4_ple::PleWarmSeq::from_slices(
                        &tokens_per_seq[s],
                        &seqs[s].tokens,
                    )
                })
                .collect::<Result<_>>()?;
            Some(ple.submit_warm_cross_seq_verify(warm_seqs)?)
        } else {
            None
        };
        let mut ssm_layer_idx = 0usize;
        let mut flat_tokens = Vec::with_capacity(u_rows);
        {
            let _keep_late_cohort =
                crate::layers::qwen3_ssm::KeepLateCohortSeqsGuard::enter(unique_leaders.len());
            self.pre_verify_copy_async_batch_dispatch(seqs, &unique_leaders)?;
            for &s in &unique_leaders {
                flat_tokens.extend_from_slice(&tokens_per_seq[s]);
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none(),
            "cross-sequence verify requires resident KV without high-speed swap"
        );
        let block_size = kv_cache.block_size();
        for (s, seq) in seqs.iter_mut().enumerate() {
            let k_s = frame.seq_k(s);
            if k_s > 0 {
                let last_block = (seq.seq_len + k_s - 1) / block_size;
                ensure_blocks_through_decode(
                    seq,
                    last_block,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        let metadata = self.qwen4_cross_seq_upload_metadata(
            u_frame,
            seqs,
            &unique_leaders,
            staging,
            block_size,
            stream,
        )?;
        let bt_stride_bucket = metadata.first().map_or(3, |m| {
            select_block_table_stride_bucket(
                m.max_blocks_per_seq as usize,
                self.max_blocks_per_seq as usize,
            )
            .1
        });
        let ctx = self.qwen4_cross_seq_ctx(None);
        let mut views_pool: Vec<Qwen4CrossSeqRows<'static>> = Vec::with_capacity(u_frame.seqs());

        // Phase-1 SSM descriptor staging: batch all 36 SSM layers' per-seq
        // `(row_base, num_tokens)` descriptors into ONE `copy_h2d_group_fast`
        // before Layer 0 and before suffix graph capture/replay.
        let (layer0_ssm_staged, all_suffix_ssm_staged) = self.stage_all_ssm_fused_descs(
            u_frame,
            seqs,
            &unique_leaders,
            &leader_of,
            &metadata,
            &ctx,
            &mut views_pool,
            stream,
        )?;

        if decode_stages {
            spark_runtime::cublaslt::stage_timer::begin(stream);
        }

        let vocab = self.verify_lmhead_vocab();
        let unified_tail_available = self.target_gumbel_verify_kernel.0 != 0
            && self.target_gumbel_verify_buf.0 != 0
            && !self.use_fp32_logits
            && !receipt_on
            && (vocab as usize) == self.config.vocab_size
            && (1..=spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS).contains(&u_rows);
        let staged_gumbel_reqs = staged_gumbel_reqs.filter(|reqs| {
            unified_tail_available
                && reqs.len() == u_rows
                && unique_leaders.len() == seqs.len()
                && super::device_topk::can_launch_single_gumbel_verify_run(reqs)
        });
        let rejection_tail_available = self.target_rejection_verify_kernel.0 != 0
            && !self.target_rejection_verify_buf.is_null()
            && !self.target_rejection_q_buf.is_null()
            && !self.use_fp32_logits
            && !receipt_on
            && (vocab as usize) == self.config.vocab_size
            && (1..=super::device_topk::REJECTION_MAX_ROWS).contains(&u_rows);
        let staged_rejection_reqs = staged_rejection_reqs.filter(|reqs| {
            rejection_tail_available
                && reqs.len() == u_rows
                && unique_leaders.len() == seqs.len()
                && super::device_topk::can_launch_single_rejection_verify_run(
                    reqs,
                    super::device_topk::REJECTION_Q_ROW_CAP,
                    super::device_topk::REJECTION_Q_STRIDE,
                )
        });
        let publish_in_graph_rejection = staged_rejection_reqs.is_some();
        let use_rejection_tail = staged_rejection_reqs.is_some()
            || (rejection_tail_available
                && staged_gumbel_reqs.is_none()
                && spark_runtime::sampler::gumbel::rejection_sampling_enabled());
        let publish_in_graph_gumbel = !use_rejection_tail
            && staged_gumbel_reqs
                .as_ref()
                .is_some_and(|reqs| reqs.iter().any(|r| r.temperature > 0.0));
        let tail_mode = if use_rejection_tail {
            BatchedVerifyTailMode::Rejection
        } else if unified_tail_available {
            BatchedVerifyTailMode::Gumbel
        } else {
            BatchedVerifyTailMode::Argmax
        };
        // Stage `flat_tokens` into `buffers.scratch()` (disjoint from
        // `qwen4_verify_attn_meta`, SSM descriptor tables, and tail staging
        // buffers) alongside the unified tail params in ONE
        // `copy_h2d_group_fast` call strictly outside Layer-0 graph capture.
        self.preflight_embed_qwen4_batch(u_rows)?;
        let token_ids_dev = self.buffers.scratch();
        // SAFETY: `flat_tokens` has length `u_rows` (`u_rows * 4` initialized
        // bytes); `copy_h2d_group_fast` finishes reading the slice into its
        // pinned ring (or synchronous H2D) before returning.
        let token_ids_bytes = unsafe {
            std::slice::from_raw_parts(flat_tokens.as_ptr().cast::<u8>(), flat_tokens.len() * 4)
        };
        match tail_mode {
            BatchedVerifyTailMode::Rejection => {
                let stage = if let Some(ref reqs) = staged_rejection_reqs {
                    super::device_topk::encode_rejection_verify_chunk(reqs)?
                } else {
                    let mut greedy_buf = [GREEDY_REJECTION_TAIL_ROW_TEMPLATE;
                        super::device_topk::REJECTION_MAX_ROWS];
                    let greedy_reqs = fill_greedy_rejection_tail_specs(&mut greedy_buf, u_rows);
                    super::device_topk::encode_rejection_verify_chunk(greedy_reqs)?
                };
                self.copy_h2d_group_fast(
                    &[
                        HostToDeviceCopy::new(&stage, self.target_rejection_verify_buf),
                        HostToDeviceCopy::new(token_ids_bytes, token_ids_dev),
                    ],
                    stream,
                )?;
            }
            BatchedVerifyTailMode::Gumbel => {
                let stage = if let Some(ref reqs) = staged_gumbel_reqs {
                    super::device_topk::encode_gumbel_verify_chunk(reqs)?
                } else {
                    let mut greedy_buf = [GREEDY_VERIFY_TAIL_ROW_TEMPLATE;
                        spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS];
                    let greedy_reqs = fill_greedy_verify_tail_specs(&mut greedy_buf, u_rows);
                    super::device_topk::encode_gumbel_verify_chunk(greedy_reqs)?
                };
                self.copy_h2d_group_fast(
                    &[
                        HostToDeviceCopy::new(&stage, self.target_gumbel_verify_buf),
                        HostToDeviceCopy::new(token_ids_bytes, token_ids_dev),
                    ],
                    stream,
                )?;
            }
            BatchedVerifyTailMode::Argmax => {
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                    stream,
                )?;
            }
        }
        let argmax_out = self.buffers.scratch();
        let argmax_rows_active =
            super::verify_d::verify_argmax_rows_selected() && self.argmax_rows_kernel.0 != 0;
        let all_non_qsa = unique_leaders.iter().all(|&s| !seqs[s].qwen4_qsa_required);
        let qsa_dense_prefix = !all_non_qsa
            && batched_verify_qsa_dense_prefix_eligible(
                unique_leaders.len(),
                !staging.verify_meta_dev.is_null(),
                staging.qsa_scratch.is_allocated(),
                unique_leaders.iter().enumerate().map(|(u, &s)| {
                    (
                        seqs[s].qwen4_qsa_required,
                        seqs[s].seq_len,
                        u_frame.seq_k(u),
                    )
                }),
            );
        let all_dense_attn = all_non_qsa || qsa_dense_prefix;
        let kv_bf16 = kv_cache.config().dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16;
        // K=1 slots do not stage KEEP_LATE ctrl_ptr; under compact_lazy_h + KEEP_LATE,
        // run eager so mixed K1/K>=2 frames flush pending KEEP_LATE rows cleanly.
        let keep_late_graph_safe = u_frame.min_k() >= 2
            || !self.ssm_pool.compact_lazy_h
            || !crate::layers::qwen3_ssm::gdn_keep_late_enabled();
        let graph_eligible = batched_verify_graph_enabled()
            && self.comm.is_none()
            && !self.suppress_graphs.load(Ordering::Relaxed)
            && !decode_stages
            && !receipt_on
            && self.layers.len() > 1
            && u_frame.seqs() >= 2
            && all_dense_attn
            && kv_bf16
            && keep_late_graph_safe
            && (!u_frame.is_ragged() || all_suffix_ssm_staged)
            && crate::layers::qwen4_qsa::batched_qsa_multi_enabled()
            && unique_leaders
                .iter()
                .all(|&s| seqs[s].disk_block_ids.is_empty());
        let sm_count = atlas_core::device::active_sm_count();
        let model_id = self.batched_verify_tracker_id();
        let required_warmup_hits =
            batched_verify_graph_required_warmup_hits_for_sm(u_frame.seqs(), sm_count);
        let capture_row_admitted = batched_verify_graph_capture_row_admitted(
            u_frame.is_ragged(),
            unique_leaders.len(),
            u_rows,
        );
        let leader_slots: Vec<usize> = if graph_eligible {
            unique_leaders.iter().map(|&s| seqs[s].slot_idx).collect()
        } else {
            Vec::new()
        };
        let layer0_graph_eligible =
            graph_eligible && qwen4_layer0_prefix_graph_enabled() && layer0_ssm_staged;
        let layer0_slots = if layer0_graph_eligible {
            Some(batched_verify_graph_key_slots(
                leader_slots.clone(),
                layer0_ssm_staged,
                all_dense_attn,
            ))
        } else {
            None
        };
        let suffix_graph_key: Option<BatchedVerifyGraphKey> = if graph_eligible {
            let graph_slots =
                batched_verify_graph_key_slots(leader_slots, all_suffix_ssm_staged, all_dense_attn);
            Some(with_qsa_dense_prefix_graph_bit(
                batched_verify_graph_key_with_tail_and_bt_bucket(
                    graph_slots,
                    u_rows,
                    all_dense_attn,
                    tc_scope_active,
                    argmax_rows_active && tail_mode == BatchedVerifyTailMode::Argmax,
                    tail_mode == BatchedVerifyTailMode::Gumbel,
                    tail_mode == BatchedVerifyTailMode::Rejection,
                    bt_stride_bucket,
                ),
                qsa_dense_prefix,
            ))
        } else {
            None
        };
        let mut active_layer0_graph_key: Option<BatchedVerifyGraphKey> = None;

        // Prefix (`embed -> Layer 0`) right before PLE wait/inject: replay cached
        // slot-agnostic Layer-0 prefix graph, warmup on initial hits (gated by
        // `required_warmup_hits` so transient drain cohorts never waste
        // `cudaGraphInstantiate`), or capture via `CaptureGuard::begin` with
        // graceful eager fallback.
        if self.layers.is_empty() {
            let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
            self.launch_qwen4_batched_embed_from_dev(
                embed_kernels,
                token_ids_dev,
                hidden,
                u_rows,
                stream,
            )?;
        } else {
            match layer0_slots {
                None => {
                    if self.config.pp_rank == 0 {
                        let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
                        self.launch_qwen4_batched_embed_from_dev(
                            embed_kernels,
                            token_ids_dev,
                            hidden,
                            u_rows,
                            stream,
                        )?;
                    } else {
                        self.pp_stage_enter(hidden, u_rows, stream)?;
                    }
                    let _ = self.run_batched_verify_layer(
                        0,
                        ssm_layer_idx,
                        hidden,
                        residual,
                        u_frame,
                        seqs,
                        &unique_leaders,
                        &leader_of,
                        &metadata,
                        &mut kv_cache,
                        staging,
                        &ctx,
                        &mut views_pool,
                        decode_stages,
                        stream,
                    )?;
                }
                Some(layer0_slots) => {
                    let embed_fused = Self::qwen4_embed_expand_fused_selected();
                    let layer0_bt_bucket = if self.config.layer_type(0) == LayerType::FullAttention
                    {
                        bt_stride_bucket
                    } else {
                        0
                    };
                    let layer0_key = with_qsa_dense_prefix_graph_bit(
                        batched_verify_layer0_graph_key_with_bt_bucket(
                            layer0_slots,
                            u_rows,
                            all_dense_attn,
                            tc_scope_active,
                            embed_fused,
                            layer0_bt_bucket,
                        ),
                        qsa_dense_prefix,
                    );
                    let (capture_poisoned, layer0_warm) = {
                        let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                            .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                        let mut map = tracker.lock();
                        let state = map.entry(model_id).or_default();
                        let warm = state.is_warm(&layer0_key, true, required_warmup_hits);
                        (state.capture_poisoned, warm)
                    };
                    let allow_layer0_superset = layer0_ssm_staged
                        && all_dense_attn
                        && u_frame.min_k() >= 2
                        && crate::layers::qwen3_ssm::replay_ssm_enabled();
                    let cached_layer0 = if !capture_poisoned {
                        let graphs = self.batch_decode_graphs.lock();
                        lookup_cached_batched_verify_graph_for_sm(
                            &graphs,
                            &layer0_key,
                            u_frame.seqs(),
                            allow_layer0_superset,
                            sm_count,
                        )
                    } else {
                        None
                    };
                    if let Some((matched_key, graph)) = cached_layer0 {
                        {
                            let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                            if let Some(state) = tracker.lock().get_mut(&model_id) {
                                state.touch(&matched_key);
                            }
                        }
                        self.gpu.launch_graph(graph, stream)?;
                        active_layer0_graph_key = Some(matched_key);
                    } else if capture_poisoned || !layer0_warm || !capture_row_admitted {
                        let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
                        self.launch_qwen4_batched_embed_from_dev(
                            embed_kernels,
                            token_ids_dev,
                            hidden,
                            u_rows,
                            stream,
                        )?;
                        let _ = self.run_batched_verify_layer(
                            0,
                            ssm_layer_idx,
                            hidden,
                            residual,
                            u_frame,
                            seqs,
                            &unique_leaders,
                            &leader_of,
                            &metadata,
                            &mut kv_cache,
                            staging,
                            &ctx,
                            &mut views_pool,
                            decode_stages,
                            stream,
                        )?;
                        if capture_row_admitted {
                            let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                            let mut map = tracker.lock();
                            let state = map.entry(model_id).or_default();
                            state.record_layer0_warmup_hit(layer0_key);
                        }
                    } else {
                        self.evict_batched_verify_graphs_for_capture(
                            model_id,
                            suffix_graph_key.as_ref(),
                            BatchedVerifyGraphStage::Layer0Prefix,
                            stream,
                        )?;

                        let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
                        let capture_ctx = ForwardContext {
                            graph_capture: true,
                            ..ctx
                        };
                        let capture_res: Result<GraphHandle> = (|| {
                            let capture = super::capture_guard::CaptureGuard::begin(
                                self.gpu.as_ref(),
                                stream,
                                "verify_qwen4_batched_layer0",
                            )?;
                            self.launch_qwen4_batched_embed_from_dev(
                                embed_kernels,
                                token_ids_dev,
                                hidden,
                                u_rows,
                                stream,
                            )?;
                            let _ = self.run_batched_verify_layer(
                                0,
                                ssm_layer_idx,
                                hidden,
                                residual,
                                u_frame,
                                seqs,
                                &unique_leaders,
                                &leader_of,
                                &metadata,
                                &mut kv_cache,
                                staging,
                                &capture_ctx,
                                &mut views_pool,
                                decode_stages,
                                stream,
                            )?;
                            capture.end()
                        })();

                        match capture_res {
                            Ok(graph) if graph.0 != 0 => {
                                tracing::info!(
                                    seqs = u_frame.seqs(),
                                    min_k = u_frame.min_k(),
                                    max_k = u_frame.max_k(),
                                    rows = u_rows,
                                    ragged = u_frame.is_ragged(),
                                    embed_fused,
                                    slots = ?layer0_key.0,
                                    "QWEN4_BATCHED_VERIFY_LAYER0_GRAPH_CAPTURED prefix embed+layer0"
                                );
                                let replaced = {
                                    let mut graphs = self.batch_decode_graphs.lock();
                                    let prev = graphs.insert(layer0_key.clone(), graph);
                                    let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                        .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                                    let mut map = tracker.lock();
                                    let state = map.entry(model_id).or_default();
                                    state.record_layer0_captured(
                                        layer0_key.clone(),
                                        required_warmup_hits,
                                    );
                                    prev
                                };
                                if let Some(old_graph) = replaced
                                    && old_graph.0 != 0
                                    && old_graph.0 != graph.0
                                {
                                    self.gpu.synchronize(stream)?;
                                    if let Err(err) = self.gpu.destroy_graph(old_graph) {
                                        tracing::error!(
                                            "verify_qwen4_batched: destroy replaced graph {} failed: {err:#}",
                                            old_graph.0
                                        );
                                    }
                                }
                                self.gpu.launch_graph(graph, stream)?;
                                active_layer0_graph_key = Some(layer0_key);
                            }
                            Ok(_) => {
                                // Mock/no-graph backend executed the prefix inline during capture.
                            }
                            Err(capture_err) => {
                                {
                                    let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                        .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                                    let mut map = tracker.lock();
                                    let state = map.entry(model_id).or_default();
                                    state.consecutive_capture_failures =
                                        state.consecutive_capture_failures.saturating_add(1);
                                    if state.consecutive_capture_failures
                                        >= QWEN4_BATCHED_VERIFY_CAPTURE_POISON_LIMIT
                                    {
                                        state.capture_poisoned = true;
                                    }
                                }
                                tracing::warn!(
                                    seqs = u_frame.seqs(),
                                    min_k = u_frame.min_k(),
                                    max_k = u_frame.max_k(),
                                    rows = u_rows,
                                    "ATLAS_QWEN4_LAYER0_PREFIX_GRAPH capture failed ({capture_err:#}); \
                                     falling back to eager execution"
                                );
                                self.launch_qwen4_batched_embed_from_dev(
                                    embed_kernels,
                                    token_ids_dev,
                                    hidden,
                                    u_rows,
                                    stream,
                                )?;
                                let _ = self.run_batched_verify_layer(
                                    0,
                                    ssm_layer_idx,
                                    hidden,
                                    residual,
                                    u_frame,
                                    seqs,
                                    &unique_leaders,
                                    &leader_of,
                                    &metadata,
                                    &mut kv_cache,
                                    staging,
                                    &ctx,
                                    &mut views_pool,
                                    decode_stages,
                                    stream,
                                )?;
                            }
                        }
                    }
                }
            }
            if self.config.layer_type(0) == LayerType::LinearAttention {
                ssm_layer_idx += 1;
            }
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.layers.len() > 1
            && let Some(checked_ple) = self.qwen4_ple.as_ref()
        {
            let pre_read = if let Some(ticket) = ple_warm_ticket.take() {
                Some(ticket.wait()?)
            } else {
                None
            };
            if crate::layers::qwen4_ple::ple_verify_batch_selected() {
                let ple = checked_ple;
                let ple_seqs: Vec<(&[u32], &[u32], usize)> = unique_leaders
                    .iter()
                    .map(|&s| {
                        (
                            tokens_per_seq[s].as_slice(),
                            seqs[s].tokens.as_slice(),
                            seqs[s].slot_idx,
                        )
                    })
                    .collect();
                let ple_prep_stream = (!decode_stages
                    && crate::layers::qwen4_ple::Qwen4PleLayer::verify_overlap_selected()
                    && self.secondary_stream != 0
                    && self.secondary_stream != stream)
                    .then_some(self.secondary_stream);
                ple.forward_cross_seq_verify_with_prep_stream(
                    &ple_seqs,
                    pre_read,
                    hidden,
                    row_bytes,
                    self.gpu.as_ref(),
                    stream,
                    ple_prep_stream,
                )?;
            } else {
                for (u, &s) in unique_leaders.iter().enumerate() {
                    self.qwen4_verify_ple(
                        &tokens_per_seq[s],
                        seqs[s],
                        hidden.offset(u_frame.row_base(u) * row_bytes),
                        row_bytes,
                        stream,
                    )?;
                }
            }
            if decode_stages {
                spark_runtime::cublaslt::stage_timer::set_group("");
                spark_runtime::cublaslt::stage_timer::mark("ple", stream);
            }
        }

        // Suffix (Layers 1.. + final_mixer + final_norm + lm_head + argmax/gumbel):
        // replay cached cohort graph, warmup on first cohort step, or capture
        // with graceful eager fallback on any capture error.
        match suffix_graph_key {
            None => {
                if let Some(tracker) = BATCHED_VERIFY_GRAPH_TRACKER.get()
                    && let Some(state) = tracker.lock().get_mut(&model_id)
                {
                    state.last_warm_key = None;
                }
                self.run_batched_verify_suffix(
                    ssm_layer_idx,
                    hidden,
                    residual,
                    u_frame,
                    rows,
                    seqs,
                    &unique_leaders,
                    &leader_of,
                    &metadata,
                    &mut kv_cache,
                    staging,
                    &ctx,
                    &mut views_pool,
                    decode_stages,
                    vocab,
                    argmax_out,
                    tail_mode,
                    stream,
                )?;
            }
            Some(graph_key) => {
                let (capture_poisoned, cohort_warm) = {
                    let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                        .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                    let mut map = tracker.lock();
                    let state = map.entry(model_id).or_default();
                    let warm =
                        state.is_warm(&graph_key, all_suffix_ssm_staged, required_warmup_hits);
                    (state.capture_poisoned, warm)
                };
                let allow_suffix_superset = all_suffix_ssm_staged
                    && all_dense_attn
                    && u_frame.min_k() >= 2
                    && crate::layers::qwen3_ssm::replay_ssm_enabled();
                let cached_graph = if !capture_poisoned && (all_suffix_ssm_staged || cohort_warm) {
                    let graphs = self.batch_decode_graphs.lock();
                    lookup_cached_batched_verify_graph_for_sm(
                        &graphs,
                        &graph_key,
                        u_frame.seqs(),
                        allow_suffix_superset,
                        sm_count,
                    )
                } else {
                    None
                };
                if let Some((matched_key, graph)) = cached_graph {
                    {
                        let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                            .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                        if let Some(state) = tracker.lock().get_mut(&model_id) {
                            state.touch(&matched_key);
                        }
                    }
                    self.gpu.launch_graph(graph, stream)?;
                } else if capture_poisoned || !cohort_warm || !capture_row_admitted {
                    self.run_batched_verify_suffix(
                        ssm_layer_idx,
                        hidden,
                        residual,
                        u_frame,
                        rows,
                        seqs,
                        &unique_leaders,
                        &leader_of,
                        &metadata,
                        &mut kv_cache,
                        staging,
                        &ctx,
                        &mut views_pool,
                        decode_stages,
                        vocab,
                        argmax_out,
                        tail_mode,
                        stream,
                    )?;
                    if capture_row_admitted {
                        let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                            .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                        let mut map = tracker.lock();
                        let state = map.entry(model_id).or_default();
                        state.record_warmup_hit(graph_key);
                    }
                } else {
                    self.evict_batched_verify_graphs_for_capture(
                        model_id,
                        active_layer0_graph_key.as_ref(),
                        BatchedVerifyGraphStage::Suffix,
                        stream,
                    )?;

                    let capture_ctx = ForwardContext {
                        graph_capture: true,
                        ..ctx
                    };
                    let capture_res: Result<GraphHandle> = (|| {
                        let capture = super::capture_guard::CaptureGuard::begin(
                            self.gpu.as_ref(),
                            stream,
                            "verify_qwen4_batched",
                        )?;
                        self.run_batched_verify_suffix(
                            ssm_layer_idx,
                            hidden,
                            residual,
                            u_frame,
                            rows,
                            seqs,
                            &unique_leaders,
                            &leader_of,
                            &metadata,
                            &mut kv_cache,
                            staging,
                            &capture_ctx,
                            &mut views_pool,
                            decode_stages,
                            vocab,
                            argmax_out,
                            tail_mode,
                            stream,
                        )?;
                        capture.end()
                    })();

                    match capture_res {
                        Ok(graph) if graph.0 != 0 => {
                            tracing::info!(
                                seqs = u_frame.seqs(),
                                min_k = u_frame.min_k(),
                                max_k = u_frame.max_k(),
                                rows = u_rows,
                                ragged = u_frame.is_ragged(),
                                tail_mode = ?tail_mode,
                                slots = ?graph_key.0,
                                "QWEN4_BATCHED_VERIFY_GRAPH_CAPTURED suffix layers=1.. head=1"
                            );
                            let replaced = {
                                let mut graphs = self.batch_decode_graphs.lock();
                                let prev = graphs.insert(graph_key.clone(), graph);
                                let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                    .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                                let mut map = tracker.lock();
                                let state = map.entry(model_id).or_default();
                                state.record_captured(graph_key, required_warmup_hits);
                                prev
                            };
                            if let Some(old_graph) = replaced
                                && old_graph.0 != 0
                                && old_graph.0 != graph.0
                            {
                                self.gpu.synchronize(stream)?;
                                if let Err(err) = self.gpu.destroy_graph(old_graph) {
                                    tracing::error!(
                                        "verify_qwen4_batched: destroy replaced graph {} failed: {err:#}",
                                        old_graph.0
                                    );
                                }
                            }
                            self.gpu.launch_graph(graph, stream)?;
                        }
                        Ok(_) => {
                            // Mock/no-graph backend executed the suffix inline during capture.
                        }
                        Err(capture_err) => {
                            {
                                let tracker = BATCHED_VERIFY_GRAPH_TRACKER
                                    .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
                                let mut map = tracker.lock();
                                let state = map.entry(model_id).or_default();
                                state.consecutive_capture_failures =
                                    state.consecutive_capture_failures.saturating_add(1);
                                if state.consecutive_capture_failures
                                    >= QWEN4_BATCHED_VERIFY_CAPTURE_POISON_LIMIT
                                {
                                    state.capture_poisoned = true;
                                }
                                state.last_warm_key = None;
                            }
                            tracing::warn!(
                                seqs = u_frame.seqs(),
                                min_k = u_frame.min_k(),
                                max_k = u_frame.max_k(),
                                rows = u_rows,
                                "ATLAS_QWEN4_BATCHED_VERIFY_GRAPH capture failed ({capture_err:#}); \
                                 falling back to eager execution"
                            );
                            self.run_batched_verify_suffix(
                                ssm_layer_idx,
                                hidden,
                                residual,
                                u_frame,
                                rows,
                                seqs,
                                &unique_leaders,
                                &leader_of,
                                &metadata,
                                &mut kv_cache,
                                staging,
                                &ctx,
                                &mut views_pool,
                                decode_stages,
                                vocab,
                                argmax_out,
                                tail_mode,
                                stream,
                            )?;
                        }
                    }
                }
            }
        }
        drop(kv_cache);
        if self.config.pp_rank > 0 {
            for (s, (seq, window)) in seqs.iter_mut().zip(tokens_per_seq).enumerate() {
                seq.tokens.extend_from_slice(window);
                seq.seq_len += frame.seq_k(s);
            }
            return Ok((0..frame.seqs())
                .map(|s| vec![0u32; frame.seq_k(s)])
                .collect());
        }
        if unique_leaders.len() < seqs.len() {
            let vocab_row_bytes = vocab as usize * 2;
            for s in (0..seqs.len()).rev() {
                let u = leader_of[s];
                if s != u {
                    let k_s = frame.seq_k(s);
                    let src_row = u_frame.row_base(u);
                    let dst_row = frame.row_base(s);
                    self.gpu.copy_d2d_async(
                        hidden.offset(src_row * row_bytes),
                        hidden.offset(dst_row * row_bytes),
                        k_s * row_bytes,
                        stream,
                    )?;
                    self.gpu.copy_d2d_async(
                        self.buffers.logits().offset(src_row * vocab_row_bytes),
                        self.buffers.logits().offset(dst_row * vocab_row_bytes),
                        k_s * vocab_row_bytes,
                        stream,
                    )?;
                }
            }
        }
        let (argmax_u, rejection_words_u): (
            Vec<u32>,
            Option<Vec<super::device_topk::PackedRejectionWord>>,
        ) = match tail_mode {
            BatchedVerifyTailMode::Rejection => {
                let mut host = [0u8; super::device_topk::REJECTION_MAX_ROWS * 4];
                let d2h_slice = &mut host[..u_rows * 4];
                self.gpu.copy_d2h_on_stream(
                    self.target_rejection_verify_buf
                        .offset(super::device_topk::REJECTION_OUT_OFFSET),
                    d2h_slice,
                    stream,
                )?;
                let words = super::device_topk::decode_rejection_verify_tokens(
                    d2h_slice,
                    u_rows,
                    self.config.vocab_size,
                )?;
                if let Some(ref reqs) = staged_rejection_reqs {
                    for (slot, (req, &word)) in reqs.iter().zip(words.iter()).enumerate() {
                        ensure!(
                            req.temperature < 0.0 || !word.is_host_fallback(),
                            "target_rejection_verify_rows_bf16 returned host fallback sentinel for active row slot {slot}"
                        );
                    }
                }
                if publish_in_graph_rejection {
                    *self.in_graph_rejection_verify_words.lock() = Some(words.clone());
                    static IN_GRAPH_REJECTION_ENGAGED: std::sync::Once = std::sync::Once::new();
                    IN_GRAPH_REJECTION_ENGAGED.call_once(|| {
                        tracing::info!(
                            seqs = frame.seqs(),
                            rows = u_rows,
                            "QWEN4_BATCHED_VERIFY_IN_GRAPH_REJECTION_ENGAGED"
                        );
                    });
                }
                let plain_tokens = words
                    .iter()
                    .map(|w| {
                        if w.is_host_fallback() {
                            u32::MAX
                        } else {
                            w.token_id()
                        }
                    })
                    .collect();
                (plain_tokens, publish_in_graph_rejection.then_some(words))
            }
            BatchedVerifyTailMode::Gumbel => {
                let mut host =
                    [0u8; spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS * 4];
                let d2h_slice = &mut host[..u_rows * 4];
                self.gpu.copy_d2h_on_stream(
                    self.target_gumbel_verify_buf
                        .offset(super::device_topk::GUMBEL_OUT_OFFSET),
                    d2h_slice,
                    stream,
                )?;
                let tokens = super::device_topk::decode_gumbel_verify_tokens(
                    d2h_slice,
                    u_rows,
                    self.config.vocab_size,
                )?;
                if publish_in_graph_gumbel {
                    *self.in_graph_gumbel_verify_tokens.lock() = Some(tokens.clone());
                    static IN_GRAPH_GUMBEL_ENGAGED: std::sync::Once = std::sync::Once::new();
                    IN_GRAPH_GUMBEL_ENGAGED.call_once(|| {
                        tracing::info!(
                            seqs = frame.seqs(),
                            rows = u_rows,
                            "QWEN4_BATCHED_VERIFY_IN_GRAPH_GUMBEL_ENGAGED"
                        );
                    });
                }
                (tokens, None)
            }
            BatchedVerifyTailMode::Argmax => {
                let mut host = vec![0u8; u_rows * 4];
                self.gpu.copy_d2h(argmax_out, &mut host)?;
                let tokens = host
                    .chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect();
                (tokens, None)
            }
        };
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let _ = &rejection_words_u;

        if receipt_on {
            let vocab_row_bytes = vocab as usize * 2;
            for (s, (seq, window)) in seqs.iter().zip(tokens_per_seq).enumerate() {
                let k_s = frame.seq_k(s);
                let base = frame.row_base(s);
                let u_base = u_frame.row_base(leader_of[s]);
                let mut logits = vec![0u8; k_s * vocab_row_bytes];
                self.gpu.copy_d2h(
                    self.buffers.logits().offset(base * vocab_row_bytes),
                    &mut logits,
                )?;
                tracing::info!(
                    "{}",
                    crate::model::verify_logits_receipt::line(
                        "batched",
                        &seq.tokens,
                        window,
                        &argmax_u[u_base..u_base + k_s],
                        &logits,
                    )
                );
            }
        }
        for (s, (seq, window)) in seqs.iter_mut().zip(tokens_per_seq).enumerate() {
            seq.tokens.extend_from_slice(window);
            seq.seq_len += frame.seq_k(s);
        }
        // Speculatively pre-warm each leader's predicted next row-0 (`bonus`) PLE
        // rows immediately while commit and MTP drafting run (~8-12 ms overlap).
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(ple) = self.qwen4_ple.as_ref()
            && crate::layers::qwen4_ple::ple_verify_batch_selected()
        {
            let mut bonus_tokens = [0u32; crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            let mut bonus_entries: Vec<(usize, usize)> = Vec::with_capacity(unique_leaders.len());
            for &s in &unique_leaders {
                let idx = bonus_entries.len();
                if idx >= bonus_tokens.len() {
                    break;
                }
                let k_s = frame.seq_k(s);
                let u_base = u_frame.row_base(leader_of[s]);
                let pre_verify_len = seqs[s].tokens.len().saturating_sub(k_s);
                let predicted = if let (Some(words), Some(reqs)) = (
                    rejection_words_u.as_deref(),
                    staged_rejection_reqs.as_deref(),
                ) {
                    predict_next_bonus_ple_input_rejection(
                        &tokens_per_seq[s],
                        &words[u_base..u_base + k_s],
                        &reqs[u_base..u_base + k_s],
                        pre_verify_len,
                    )
                } else {
                    predict_next_bonus_ple_input(
                        &tokens_per_seq[s],
                        &argmax_u[u_base..u_base + k_s],
                        pre_verify_len,
                    )
                };
                if let Some((bonus, committed_len)) = predicted
                    && committed_len <= seqs[s].tokens.len()
                {
                    bonus_tokens[idx] = bonus;
                    bonus_entries.push((s, committed_len));
                }
            }
            if !bonus_entries.is_empty() {
                let warm_pairs: Vec<(&[u32], &[u32])> = bonus_entries
                    .iter()
                    .enumerate()
                    .map(|(idx, &(s, committed_len))| {
                        (
                            std::slice::from_ref(&bonus_tokens[idx]),
                            &seqs[s].tokens[..committed_len],
                        )
                    })
                    .collect();
                ple.submit_draft_ahead_seqs(&warm_pairs);
            }
        }
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                seqs = frame.seqs(),
                min_k = frame.min_k(),
                max_k = frame.max_k(),
                rows,
                ragged = frame.is_ragged(),
                "QWEN4_BATCHED_K5_ENGAGED cross-sequence native-MTP verify frame"
            );
        });
        Ok((0..frame.seqs())
            .map(|s| {
                let k_s = frame.seq_k(s);
                let u_base = u_frame.row_base(leader_of[s]);
                argmax_u[u_base..u_base + k_s].to_vec()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT, ensure_verify_meta_disjoint_from_scratch,
        frame_metadata_bytes, lm_head_route_admitted_for_frame, moe_routing_scratch_bytes,
        ranges_disjoint, validate_frame_inputs,
    };
    use spark_runtime::gpu::DevicePtr;

    fn windows(c: usize, k: usize) -> Vec<Vec<u32>> {
        (0..c)
            .map(|s| (0..k as u32).map(|t| t + s as u32).collect())
            .collect()
    }

    #[test]
    fn admits_distinct_slots_with_lanes() {
        let frame = validate_frame_inputs(&windows(4, 5), &[0, 3, 1, 7], 8, 1_000).unwrap();
        assert_eq!((frame.seqs(), frame.k(), frame.rows()), (4, 5, 20));
        let k8_slots: Vec<usize> = (0..16).collect();
        let k8 = validate_frame_inputs(&windows(16, 8), &k8_slots, 16, 1_000).unwrap();
        assert_eq!((k8.seqs(), k8.k(), k8.rows()), (16, 8, 128));
        let k7_slots: Vec<usize> = (0..18).collect();
        let k7 = validate_frame_inputs(&windows(18, 7), &k7_slots, 32, 1_000).unwrap();
        assert_eq!((k7.seqs(), k7.k(), k7.rows()), (18, 7, 126));
        let k6_slots: Vec<usize> = (0..21).collect();
        let k6 = validate_frame_inputs(&windows(21, 6), &k6_slots, 32, 1_000).unwrap();
        assert_eq!((k6.seqs(), k6.k(), k6.rows()), (21, 6, 126));
        let wide_slots: Vec<usize> = (0..64).collect();
        let wide = validate_frame_inputs(&windows(64, 2), &wide_slots, 64, 1_000).unwrap();
        assert_eq!((wide.seqs(), wide.k(), wide.rows()), (64, 2, 128));
    }

    #[test]
    fn admits_ragged_windows_and_refuses_empty_or_shared_slots() {
        let ragged = vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8], vec![9]];
        let frame = validate_frame_inputs(&ragged, &[0, 1, 2], 8, 1_000).unwrap();
        assert!(frame.is_ragged());
        assert_eq!(frame.seqs(), 3);
        assert_eq!(frame.min_k(), 1);
        assert_eq!(frame.max_k(), 5);
        assert_eq!(frame.rows(), 9);
        assert_eq!(frame.row_base(0), 0);
        assert_eq!(frame.row_base(1), 5);
        assert_eq!(frame.row_base(2), 8);

        // Empty window, >9 window, shared slots, and mismatched counts fail closed.
        assert!(validate_frame_inputs(&[vec![1, 2], vec![]], &[0, 1], 8, 1_000).is_err());
        assert!(validate_frame_inputs(&[vec![1; 5], vec![2; 10]], &[0, 1], 8, 1_000).is_err());
        assert!(validate_frame_inputs(&windows(2, 5), &[2, 2], 8, 1_000).is_err());
        assert!(validate_frame_inputs(&windows(2, 5), &[0], 8, 1_000).is_err());
    }

    #[test]
    fn refuses_missing_lanes_wide_frames_and_bad_tokens() {
        assert!(validate_frame_inputs(&windows(2, 5), &[0, 1], 1, 1_000).is_err());
        assert!(validate_frame_inputs(&windows(2, 5), &[0, 8], 8, 1_000).is_err());
        // 51 x K5 = 255 fits; 52 x K5 = 260 > 256 rejects.
        let slots_51: Vec<usize> = (0..51).collect();
        assert!(validate_frame_inputs(&windows(51, 5), &slots_51, 64, 1_000).is_ok());
        let too_many_slots: Vec<usize> = (0..52).collect();
        assert!(validate_frame_inputs(&windows(52, 5), &too_many_slots, 64, 1_000).is_err());
        // 32 x K8 = 256 fits; 33 x K8 = 264 > 256 rejects.
        let slots_32: Vec<usize> = (0..32).collect();
        assert!(validate_frame_inputs(&windows(32, 8), &slots_32, 64, 1_000).is_ok());
        let too_many_k8: Vec<usize> = (0..33).collect();
        assert!(validate_frame_inputs(&windows(33, 8), &too_many_k8, 64, 1_000).is_err());
        // 64 x K4 = 256 fits.
        let slots_64: Vec<usize> = (0..64).collect();
        assert!(validate_frame_inputs(&windows(64, 4), &slots_64, 64, 1_000).is_ok());
        assert!(validate_frame_inputs(&windows(1, 5), &[0], 8, 1_000).is_err());
        assert!(validate_frame_inputs(&windows(2, 5), &[0, 1], 8, 4).is_err());
    }

    #[test]
    fn metadata_fits_the_32_128_and_256_row_abi_tiers() {
        assert_eq!(
            frame_metadata_bytes(30, 16_385),
            Some(768 + 30 * 16_385 * 4)
        );
        assert_eq!(frame_metadata_bytes(32, 1), Some(768 + 32 * 4));
        assert_eq!(frame_metadata_bytes(33, 1), Some(3072 + 33 * 4));
        assert_eq!(
            frame_metadata_bytes(128, 16_385),
            Some(3072 + 128 * 16_385 * 4)
        );
        assert_eq!(frame_metadata_bytes(129, 1), Some(6144 + 129 * 4));
        assert_eq!(
            frame_metadata_bytes(256, 16_385),
            Some(6144 + 256 * 16_385 * 4)
        );
        assert_eq!(
            frame_metadata_bytes(257, 1),
            None,
            "slots overflow 2048 bytes"
        );
    }

    #[test]
    fn mixed_moe_routing_scratch_detects_aliasing_and_accepts_disjoint_staging() {
        let scratch = DevicePtr(0x1000_0000);
        // P=4096 + D=128 = 4224 rows, top_k=10 -> 2 * 4224 * 10 * 4 = 337_920 B.
        let moe_bytes = moe_routing_scratch_bytes(4096 + 128, 10).unwrap();
        assert_eq!(moe_bytes, 337_920);
        assert!(moe_bytes > crate::layer::STEP_META_SCRATCH_OFFSET);

        // Legacy placement at scratch + STEP_META_SCRATCH_OFFSET aliases MoE routing scratch.
        let legacy_meta = scratch.offset(crate::layer::STEP_META_SCRATCH_OFFSET);
        let meta_bytes = frame_metadata_bytes(128, 1024).unwrap();
        assert!(!ranges_disjoint(
            legacy_meta.0,
            meta_bytes,
            scratch.0,
            moe_bytes
        ));
        assert!(
            ensure_verify_meta_disjoint_from_scratch(
                legacy_meta,
                meta_bytes,
                scratch,
                moe_bytes,
                "legacy_scratch_meta",
            )
            .is_err()
        );

        // Dedicated disjoint staging allocations pass both full-scratch and MoE-prefix checks.
        let disjoint_attn_meta = DevicePtr(0x2000_0000);
        let disjoint_qsa_meta = DevicePtr(0x3000_0000);
        ensure_verify_meta_disjoint_from_scratch(
            disjoint_attn_meta,
            meta_bytes,
            scratch,
            moe_bytes,
            "disjoint_attn_meta",
        )
        .unwrap();
        ensure_verify_meta_disjoint_from_scratch(
            disjoint_qsa_meta,
            2048,
            scratch,
            moe_bytes,
            "disjoint_qsa_meta",
        )
        .unwrap();
        // Null non-empty metadata must fail closed.
        assert!(
            ensure_verify_meta_disjoint_from_scratch(
                DevicePtr::NULL,
                64,
                scratch,
                moe_bytes,
                "null_meta",
            )
            .is_err()
        );
    }

    #[test]
    fn lm_head_route_admission_preserves_exact_below_4_and_admits_tc_above() {
        use crate::layers::qwen4_hyper::{
            BatchedVerifyTcComponent, batched_verify_tc_component_admits,
        };
        // Rows <= 3 stay exact: refuse ATLAS_LM_HEAD_TC=1 (C=1 stays 100% exact via
        // `BatchedVerifyTcDenseGuard::enter(false)`).
        for rows in [1, 2, 3] {
            let small = batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::LmHead,
                true,
                None,
                rows,
                false,
            );
            assert!(!small, "rows={rows}");
            assert!(lm_head_route_admitted_for_frame(small, false));
            assert!(!lm_head_route_admitted_for_frame(small, true));
        }

        // C=2 r=4..8 and C=4/8/16 (rows=4..=128) admit TC dense with or without ATLAS_LM_HEAD_TC=1.
        for rows in [4, 5, 6, 7, 8, 10, 16, 20, 40, 80, 112, 128] {
            let wide = batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::LmHead,
                true,
                None,
                rows,
                false,
            );
            assert!(wide, "rows={rows}");
            assert!(lm_head_route_admitted_for_frame(wide, false));
            assert!(lm_head_route_admitted_for_frame(wide, true));
        }

        // Disabled TC dense refuses ATLAS_LM_HEAD_TC=1 even on wide frames.
        let disabled = batched_verify_tc_component_admits(
            BatchedVerifyTcComponent::LmHead,
            false,
            None,
            80,
            false,
        );
        assert!(!disabled);
        assert!(lm_head_route_admitted_for_frame(disabled, false));
        assert!(!lm_head_route_admitted_for_frame(disabled, true));
    }

    #[test]
    fn verify_meta_host_scratch_reuses_allocation_and_zeroes_shrunk_tails_and_stride_changes() {
        use super::VerifyMetaHostScratch;

        let mut scratch = VerifyMetaHostScratch::default();
        // Step 1: 4 rows x mb=8, varying active block counts.
        scratch.prepare(4, 8, 64).unwrap();
        let initial_ptr = scratch.block_tables.as_ptr();
        scratch.write_block_table_row(0, 8, &[11, 12, 13, 14, 15]);
        scratch.write_block_table_row(1, 8, &[21, 22]);
        scratch.write_block_table_row(2, 8, &[31, 32, 33, 34]);
        scratch.write_block_table_row(3, 8, &[41, 42, 43, 44, 45, 46]);
        assert_eq!(&scratch.block_tables[0..8], &[11, 12, 13, 14, 15, 0, 0, 0]);
        assert_eq!(
            &scratch.block_tables[24..32],
            &[41, 42, 43, 44, 45, 46, 0, 0]
        );

        // Step 2: shrink to 2 rows x mb=8 with shorter block tables; buffer pointer must not reallocate.
        scratch.prepare(2, 8, 32).unwrap();
        assert_eq!(scratch.block_tables.as_ptr(), initial_ptr);
        scratch.write_block_table_row(0, 8, &[99]);
        scratch.write_block_table_row(1, 8, &[]);
        assert_eq!(&scratch.block_tables[0..8], &[99, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&scratch.block_tables[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);

        // Step 3: grow back to 4 rows x mb=8; untouched high rows (2..4) must still clear prior dirty tails!
        scratch.prepare(4, 8, 64).unwrap();
        assert_eq!(scratch.block_tables.as_ptr(), initial_ptr);
        scratch.write_block_table_row(0, 8, &[1]);
        scratch.write_block_table_row(1, 8, &[2]);
        scratch.write_block_table_row(2, 8, &[3]);
        scratch.write_block_table_row(3, 8, &[4, 5]);
        assert_eq!(&scratch.block_tables[16..24], &[3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&scratch.block_tables[24..32], &[4, 5, 0, 0, 0, 0, 0, 0]);

        // Step 4: changing max_blocks_per_seq resets all stale entries cleanly.
        scratch.prepare(3, 4, 48).unwrap();
        assert_eq!(scratch.block_tables.as_ptr(), initial_ptr);
        scratch.write_block_table_row(0, 4, &[7]);
        scratch.write_block_table_row(1, 4, &[8]);
        scratch.write_block_table_row(2, 4, &[9]);
        assert_eq!(
            &scratch.block_tables[0..12],
            &[7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0]
        );
    }

    #[test]
    fn batched_verify_graph_key_lru_and_views_recycler_behave_deterministically() {
        use super::{
            BatchedVerifyGraphModelState, QWEN4_BATCHED_VERIFY_GRAPH_CAP, batched_verify_graph_key,
            is_batched_verify_graph_meta, parse_batched_verify_graph_env, recycle_views_vec,
        };
        use spark_runtime::gpu::GraphHandle;
        use std::collections::HashMap;

        assert!(parse_batched_verify_graph_env(None));
        assert!(parse_batched_verify_graph_env(Some("1")));
        assert!(!parse_batched_verify_graph_env(Some("0")));

        // High-bit tag never collides with `decode_a2`'s `padded_n` (2..=64).
        // Keyed on `(leader_slots, total_rows, ...)`: distinct ragged partitions
        // summing to the same `total_rows` share ONE cached CUDA graph!
        let f_uniform = crate::layer::Qwen4CrossSeqFrame::new_ragged(&[5, 5]).unwrap();
        let f_ragged = crate::layer::Qwen4CrossSeqFrame::new_ragged(&[7, 3]).unwrap();
        let f_ragged_k1 = crate::layer::Qwen4CrossSeqFrame::new_ragged(&[9, 1]).unwrap();
        let k_a = batched_verify_graph_key(vec![0, 1], f_uniform.rows(), true, true, true, false);
        let k_a_ragged =
            batched_verify_graph_key(vec![0, 1], f_ragged.rows(), true, true, true, false);
        let k_a_k1 =
            batched_verify_graph_key(vec![0, 1], f_ragged_k1.rows(), true, true, true, false);
        let k_a_gumbel =
            batched_verify_graph_key(vec![0, 1], f_uniform.rows(), true, true, false, true);
        let k_b = batched_verify_graph_key(vec![1, 0], 10, true, true, true, false);
        let k_c = batched_verify_graph_key(vec![0, 1], 8, true, true, true, false);
        let k_d = batched_verify_graph_key(vec![0, 1], 14, true, true, true, false);
        let k_e = batched_verify_graph_key(vec![0, 1], 16, true, true, true, false);
        assert!(is_batched_verify_graph_meta(k_a.1));
        assert!(is_batched_verify_graph_meta(k_a_gumbel.1));
        assert!(is_batched_verify_graph_meta(k_e.1));
        assert!(!is_batched_verify_graph_meta(16));
        assert_eq!(
            k_a, k_a_ragged,
            "partitions [5,5] and [7,3] with equal total_rows=10 must share one cached graph"
        );
        assert_eq!(
            k_a, k_a_k1,
            "partitions [5,5] and [9,1] with equal total_rows=10 must share one cached graph"
        );
        assert_ne!(
            k_a, k_a_gumbel,
            "argmax tail vs in-graph Gumbel tail must never share a cached graph"
        );
        assert_ne!(
            batched_verify_graph_key(vec![0, 1], 10, true, true, false, false),
            k_a_gumbel,
            "gumbel_tail bit alone must distinguish graph keys even when argmax_rows_active=false"
        );
        assert_ne!(k_a, k_b, "row slot order must be preserved in graph key");
        assert_ne!(k_a, k_c, "total_rows=10 vs 8 must distinguish graph keys");
        assert_ne!(k_a, k_d, "total_rows=10 vs 14 must distinguish graph keys");
        assert_ne!(k_d, k_e, "total_rows=14 vs 16 must distinguish graph keys");

        // Bounded LRU evicts oldest live graph key and skips already-freed entries.
        assert_eq!(QWEN4_BATCHED_VERIFY_GRAPH_CAP, 48);
        let mut state = BatchedVerifyGraphModelState::default();
        let mut live = HashMap::new();
        for i in 0..QWEN4_BATCHED_VERIFY_GRAPH_CAP {
            let key = batched_verify_graph_key(vec![0, i + 1], 10, true, true, true, false);
            live.insert(key.clone(), GraphHandle((i + 1) as u64));
            state.touch(&key);
        }
        // Touch oldest entry `i == 0` so `i == 1` becomes the LRU victim.
        let touched = batched_verify_graph_key(vec![0, 1], 10, true, true, true, false);
        state.touch(&touched);
        let evicted = state.collect_evictions(&live, QWEN4_BATCHED_VERIFY_GRAPH_CAP);
        assert_eq!(
            evicted,
            vec![batched_verify_graph_key(
                vec![0, 2],
                10,
                true,
                true,
                true,
                false
            )]
        );

        // `recycle_views_vec` preserves the underlying Vec allocation across lifetime rebinds.
        let v: Vec<crate::layer::Qwen4CrossSeqRows<'static>> = Vec::with_capacity(16);
        let ptr = v.as_ptr();
        let cap = v.capacity();
        let v2 = recycle_views_vec(v);
        assert_eq!(v2.as_ptr(), ptr);
        assert_eq!(v2.capacity(), cap);
    }

    #[test]
    fn predict_next_bonus_ple_input_tracks_exact_committed_prior_across_accept_counts() {
        use super::predict_next_bonus_ple_input;

        // Pre-verify tokens `[10, 20]`, verify window `[30, 40, 50, 60, 70]`.
        let mut tokens = vec![10u32, 20];
        let window = [30u32, 40, 50, 60, 70];
        tokens.extend_from_slice(&window);
        let pre = 2usize;

        // 0 drafts accepted: bonus = verified[0] = 99, committed prefix ends with [20, 30].
        let (bonus, len) =
            predict_next_bonus_ple_input(&window, &[99, 50, 60, 70, 80], pre).expect("zero accept");
        assert_eq!(bonus, 99);
        assert_eq!(len, 3);
        assert_eq!(&tokens[len - 2..len], &[20, 30]);

        // 2 drafts accepted (40, 50): bonus = verified[2] = 77, committed prefix ends with [40, 50].
        let (bonus, len) = predict_next_bonus_ple_input(&window, &[40, 50, 77, 70, 80], pre)
            .expect("partial accept");
        assert_eq!(bonus, 77);
        assert_eq!(len, 5);
        assert_eq!(&tokens[len - 2..len], &[40, 50]);

        // Full accept (40, 50, 60, 70): bonus = verified[4] = 88, committed prefix ends with [60, 70].
        let (bonus, len) =
            predict_next_bonus_ple_input(&window, &[40, 50, 60, 70, 88], pre).expect("full accept");
        assert_eq!(bonus, 88);
        assert_eq!(len, 7);
        assert_eq!(&tokens[len - 2..len], &[60, 70]);

        // K=8 full accept (7 drafts + bonus): committed prefix ends with the 7th draft.
        let k8_window = [30u32, 40, 50, 60, 70, 80, 90, 100];
        let k8_verified = [40u32, 50, 60, 70, 80, 90, 100, 111];
        let (bonus_k8, len_k8) =
            predict_next_bonus_ple_input(&k8_window, &k8_verified, pre).expect("k8 full accept");
        assert_eq!(bonus_k8, 111);
        assert_eq!(len_k8, pre + 8);

        // K=1 anchor-only (0 drafts + 1 bonus): committed prefix advances by 1.
        let (bonus_k1, len_k1) =
            predict_next_bonus_ple_input(&[30], &[123], pre).expect("k1 anchor-only");
        assert_eq!(bonus_k1, 123);
        assert_eq!(len_k1, pre + 1);

        // Invalid / empty / skipped-sentinel (`u32::MAX`) inputs fail closed to None.
        assert_eq!(predict_next_bonus_ple_input(&[], &[], pre), None);
        assert_eq!(predict_next_bonus_ple_input(&window, &[40, 50], pre), None);
        assert_eq!(
            predict_next_bonus_ple_input(&window, &[u32::MAX; 5], pre),
            None
        );
        assert_eq!(
            predict_next_bonus_ple_input(&window, &[40, 50, 60, 70, 88], usize::MAX),
            None
        );
    }

    #[test]
    fn transient_drain_cohort_sizes_require_three_warmup_hits_before_capture() {
        use super::{
            BatchedVerifyGraphModelState, GREEDY_VERIFY_TAIL_ROW_TEMPLATE,
            QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS, batched_verify_graph_key,
            batched_verify_graph_required_warmup_hits, fill_greedy_verify_tail_specs,
        };

        assert_eq!(QWEN4_BATCHED_VERIFY_TRANSIENT_WARMUP_HITS, 3);
        for pow2 in [1usize, 2, 4, 8, 16] {
            assert_eq!(
                batched_verify_graph_required_warmup_hits(pow2),
                1,
                "power-of-2 cohort size {pow2} must keep 1-hit warmup threshold"
            );
        }
        for transient in [3usize, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 24] {
            assert_eq!(
                batched_verify_graph_required_warmup_hits(transient),
                3,
                "transient draining cohort size {transient} must require 3 warmup hits"
            );
        }

        let mut state = BatchedVerifyGraphModelState::default();
        let pow2_key = batched_verify_graph_key(vec![usize::MAX - 16], 80, true, true, false, true);
        let drain_key =
            batched_verify_graph_key(vec![usize::MAX - 15], 72, true, true, false, true);

        // Power-of-2 cohort becomes warm after 1 eager warmup pass.
        let pow2_req = batched_verify_graph_required_warmup_hits(16);
        assert!(!state.is_warm(&pow2_key, true, pow2_req));
        state.record_warmup_hit(pow2_key.clone());
        assert!(state.is_warm(&pow2_key, true, pow2_req));

        // Transient draining cohort (15 seqs) stays eager for hits 1..=3 and captures on hit 4.
        let drain_req = batched_verify_graph_required_warmup_hits(15);
        for hit in 1..3 {
            assert!(
                !state.is_warm(&drain_key, true, drain_req),
                "before hit {hit}"
            );
            state.record_warmup_hit(drain_key.clone());
            assert!(
                !state.is_warm(&drain_key, true, drain_req),
                "after hit {hit}"
            );
        }
        state.record_warmup_hit(drain_key.clone());
        assert!(state.is_warm(&drain_key, true, drain_req));

        // Un-staged greedy helper fills valid temp=0.0 specs for the unified tail.
        let mut buf = [GREEDY_VERIFY_TAIL_ROW_TEMPLATE;
            spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS];
        let greedy = fill_greedy_verify_tail_specs(&mut buf, 5);
        assert_eq!(greedy.len(), 5);
        for (i, spec) in greedy.iter().enumerate() {
            assert_eq!(spec.row_idx, i);
            assert_eq!(spec.seed, 0);
            assert_eq!(spec.abs_pos, 0);
            assert_eq!(spec.temperature, 0.0);
            assert_eq!(spec.top_k, 1);
            assert_eq!(spec.top_p, 1.0);
        }
    }

    #[test]
    fn ragged_graph_capture_skips_one_shot_tail_clamped_row_counts() {
        use super::batched_verify_graph_capture_row_admitted;
        // Uniform frames always admit graph capture.
        assert!(batched_verify_graph_capture_row_admitted(false, 16, 64));
        assert!(batched_verify_graph_capture_row_admitted(false, 5, 35));
        assert!(batched_verify_graph_capture_row_admitted(false, 3, 15));

        // Ragged frames admit bucket-aligned row counts and K=2 floor, skipping
        // one-shot tail-clamped counts (e.g. 55, 63, 70 at C=16).
        assert!(batched_verify_graph_capture_row_admitted(true, 2, 8));
        assert!(!batched_verify_graph_capture_row_admitted(true, 2, 9));
        assert!(batched_verify_graph_capture_row_admitted(true, 4, 20));
        assert!(!batched_verify_graph_capture_row_admitted(true, 4, 23));
        assert!(batched_verify_graph_capture_row_admitted(true, 5, 10));
        assert!(batched_verify_graph_capture_row_admitted(true, 16, 56));
        assert!(batched_verify_graph_capture_row_admitted(true, 16, 64));
        assert!(batched_verify_graph_capture_row_admitted(true, 16, 72));
        assert!(!batched_verify_graph_capture_row_admitted(true, 16, 55));
        assert!(!batched_verify_graph_capture_row_admitted(true, 16, 63));
        assert!(!batched_verify_graph_capture_row_admitted(true, 16, 70));
    }

    #[test]
    fn staged_dense_graph_key_is_slot_agnostic_across_retiring_cohorts() {
        use super::{
            QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL, batched_verify_graph_key,
            batched_verify_graph_key_slots,
        };

        // When Phase-1 stages all suffix SSM descriptors and attention is dense,
        // distinct slot permutations of the same arity collapse onto one warm graph key
        // that never collides with any real slot_idx (0..1024) during free_sequence.
        let s1 = batched_verify_graph_key_slots(vec![0, 1, 2, 3], true, true);
        let s2 = batched_verify_graph_key_slots(vec![4, 7, 11, 15], true, true);
        assert_eq!(s1, vec![QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL - 4]);
        assert_eq!(s1, s2);
        for slot in 0..1024usize {
            assert!(
                !s1.contains(&slot),
                "must never collide with real slot {slot}"
            );
        }
        assert_eq!(
            batched_verify_graph_key(s1.clone(), 16, true, true, true, false),
            batched_verify_graph_key(s2.clone(), 16, true, true, true, false)
        );
        assert_eq!(
            batched_verify_graph_key(s1.clone(), 16, true, true, false, true),
            batched_verify_graph_key(s2.clone(), 16, true, true, false, true)
        );
        assert_ne!(
            batched_verify_graph_key(s1, 16, true, true, true, false),
            batched_verify_graph_key(s2, 16, true, true, false, true)
        );
        // Different cohort arities keep distinct graph keys.
        let s3 = batched_verify_graph_key_slots(vec![0, 1, 2], true, true);
        assert_ne!(
            batched_verify_graph_key(s3, 12, true, true, true, false),
            batched_verify_graph_key(
                vec![QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL - 4],
                12,
                true,
                true,
                true,
                false
            )
        );
        // Unstaged or QSA cohorts retain exact leader_slots in the graph key.
        assert_eq!(
            batched_verify_graph_key_slots(vec![2, 5], false, true),
            vec![2, 5]
        );
        assert_eq!(
            batched_verify_graph_key_slots(vec![2, 5], true, false),
            vec![2, 5]
        );
    }

    #[test]
    fn cross_seq_proposer_admission_allows_native_mtp_and_non_probabilistic_dflash() {
        use super::cross_seq_proposer_admitted;
        use crate::dflash_sampling::DflashDraftSampleMethod as Method;
        use crate::speculative::DraftProposerKind as Kind;

        // Rank 0 admits native Qwen4 MTP regardless of DFlash capture buffer.
        assert!(cross_seq_proposer_admitted(
            0,
            Some(Kind::Qwen4NativeMtp),
            false,
            Method::Greedy,
        ));
        // Rank 0 admits greedy DFlash only when `dflash_hidden_save` is allocated.
        assert!(cross_seq_proposer_admitted(
            0,
            Some(Kind::DFlash),
            true,
            Method::Greedy,
        ));
        assert!(!cross_seq_proposer_admitted(
            0,
            Some(Kind::DFlash),
            false,
            Method::Greedy,
        ));
        // Probabilistic DFlash stays refused on rank 0.
        assert!(!cross_seq_proposer_admitted(
            0,
            Some(Kind::DFlash),
            true,
            Method::Probabilistic,
        ));
        // Other / absent proposers stay refused on rank 0, while worker ranks pass.
        assert!(!cross_seq_proposer_admitted(
            0,
            Some(Kind::Other),
            true,
            Method::Greedy,
        ));
        assert!(!cross_seq_proposer_admitted(0, None, true, Method::Greedy));
        assert!(cross_seq_proposer_admitted(1, None, false, Method::Greedy));
    }

    #[test]
    fn layer0_prefix_graph_key_is_disjoint_from_suffix_and_preserves_warmup_isolation() {
        use super::{
            BatchedVerifyGraphModelState, QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL,
            batched_verify_graph_key, batched_verify_graph_key_slots,
            batched_verify_graph_required_warmup_hits, batched_verify_layer0_graph_key,
            is_batched_verify_graph_meta,
        };

        let s1 = batched_verify_graph_key_slots(vec![0, 1, 2, 3], true, true);
        let s2 = batched_verify_graph_key_slots(vec![5, 7, 11, 15], true, true);
        assert_eq!(s1, vec![QWEN4_SLOT_AGNOSTIC_GRAPH_SENTINEL - 4]);
        assert_eq!(s1, s2);

        let l0_fused = batched_verify_layer0_graph_key(s1.clone(), 20, true, true, true);
        let l0_fused_s2 = batched_verify_layer0_graph_key(s2.clone(), 20, true, true, true);
        let l0_split = batched_verify_layer0_graph_key(s1.clone(), 20, true, true, false);
        let l0_diff_rows = batched_verify_layer0_graph_key(s1.clone(), 16, true, true, true);
        let l0_diff_tc = batched_verify_layer0_graph_key(s1.clone(), 20, true, false, true);
        let l0_diff_attn = batched_verify_layer0_graph_key(s1.clone(), 20, false, true, true);
        let suffix_unified = batched_verify_graph_key(s1.clone(), 20, true, true, false, true);
        let suffix_argmax = batched_verify_graph_key(s1, 20, true, true, true, false);

        // Layer-0 prefix key shares the batched-verify high-bit tag (so shared
        // invalidation clears both prefix and suffix) while never colliding with
        // any suffix key or decode_a2 meta.
        assert!(is_batched_verify_graph_meta(l0_fused.1));
        assert!(is_batched_verify_graph_meta(l0_split.1));
        assert_eq!(l0_fused, l0_fused_s2);
        assert_ne!(l0_fused, l0_split);
        assert_ne!(l0_fused, l0_diff_rows);
        assert_ne!(l0_fused, l0_diff_tc);
        assert_ne!(l0_fused, l0_diff_attn);
        assert_ne!(l0_fused, suffix_unified);
        assert_ne!(l0_fused, suffix_argmax);
        assert_ne!(l0_split, suffix_unified);
        assert_ne!(l0_split, suffix_argmax);

        // Layer-0 warmup/capture accounting enforces transient drain thresholds
        // and never clobbers suffix `last_warm_key` for unstaged cohorts.
        let mut state = BatchedVerifyGraphModelState::default();
        let unstaged_suffix = batched_verify_graph_key(vec![0, 1], 10, true, true, false, true);
        state.record_warmup_hit(unstaged_suffix.clone());
        assert!(state.is_warm(&unstaged_suffix, false, 1));

        let drain_slots = batched_verify_graph_key_slots((0..15).collect(), true, true);
        let l0_drain = batched_verify_layer0_graph_key(drain_slots, 72, true, true, true);
        let drain_req = batched_verify_graph_required_warmup_hits(15);
        assert_eq!(drain_req, 3);
        for _ in 0..2 {
            state.record_layer0_warmup_hit(l0_drain.clone());
            assert!(!state.is_warm(&l0_drain, true, drain_req));
        }
        state.record_layer0_warmup_hit(l0_drain.clone());
        assert!(state.is_warm(&l0_drain, true, drain_req));
        // Unstaged suffix warmup must remain intact.
        assert!(state.is_warm(&unstaged_suffix, false, 1));

        state.record_layer0_captured(l0_drain.clone(), drain_req);
        assert!(state.is_warm(&l0_drain, true, drain_req));
        assert!(state.is_warm(&unstaged_suffix, false, 1));
    }

    #[test]
    fn batched_verify_kv_ceiling_guard_enforces_per_seq_limit_and_cohort_budget() {
        use super::check_batched_verify_kv_ceiling;

        // Block size 16, max_blocks_per_seq 10 (max logical block index 9 -> pos 159).
        // Seq 0: len=30, k=5 -> last_pos=34, block 2 (needs 3, has 2 -> +1).
        // Seq 1: len=15, k=5 -> last_pos=19, block 1 (needs 2, has 1 -> +1).
        assert_eq!(
            check_batched_verify_kv_ceiling(&[30, 15], &[2, 1], &[5, 5], 16, 10, 2, 0).unwrap(),
            2
        );
        // Satisfied via free + evictable prefix blocks.
        assert_eq!(
            check_batched_verify_kv_ceiling(&[30, 15], &[2, 1], &[5, 5], 16, 10, 1, 1).unwrap(),
            2
        );
        // Cohort block exhaustion fails cleanly before any mutation.
        assert!(
            check_batched_verify_kv_ceiling(&[30, 15], &[2, 1], &[5, 5], 16, 10, 1, 0).is_err()
        );
        // Per-seq max_blocks_per_seq ceiling fails cleanly (`(156 + 4) / 16 = 10 >= 10`).
        assert!(
            check_batched_verify_kv_ceiling(&[156, 15], &[10, 1], &[5, 5], 16, 10, 100, 0).is_err()
        );
        // Exact boundary `(155 + 4) / 16 = 9 < 10` succeeds with 0 new blocks for seq 0.
        assert_eq!(
            check_batched_verify_kv_ceiling(&[155, 15], &[10, 1], &[5, 5], 16, 10, 1, 0).unwrap(),
            1
        );
    }

    #[test]
    fn rejection_tail_graph_key_and_bonus_ple_prediction_walk_accept_bits() {
        use super::super::device_topk::{
            HOST_POLICY_FALLBACK_TOKEN, PackedRejectionWord, REJECTION_BONUS_Q_ROW_IDX,
            REJECTION_NO_MASK_TOKEN, RejectionVerifyRowSpec,
        };
        use super::{
            batched_verify_graph_key_with_tail, batched_verify_layer0_graph_key,
            is_batched_verify_graph_meta, predict_next_bonus_ple_input_rejection,
        };

        let gumbel_key =
            batched_verify_graph_key_with_tail(vec![0, 1], 10, true, true, false, true, false);
        let rejection_key =
            batched_verify_graph_key_with_tail(vec![0, 1], 10, true, true, false, false, true);
        let argmax_key =
            batched_verify_graph_key_with_tail(vec![0, 1], 10, true, true, true, false, false);
        let l0_key = batched_verify_layer0_graph_key(vec![0, 1], 10, true, true, true);
        assert!(is_batched_verify_graph_meta(rejection_key.1));
        assert_ne!(rejection_key, gumbel_key);
        assert_ne!(rejection_key, argmax_key);
        assert_ne!(rejection_key, l0_key);

        let mk_req = |row_idx: usize, temperature: f32| RejectionVerifyRowSpec {
            row_idx,
            seed: 1,
            abs_pos: 50 + row_idx as u64,
            temperature,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            draft_token: 0,
            q_row_idx: if row_idx == 4 {
                REJECTION_BONUS_Q_ROW_IDX
            } else {
                row_idx as u32
            },
            mask_tok0: REJECTION_NO_MASK_TOKEN,
            mask_tok1: REJECTION_NO_MASK_TOKEN,
        };
        let reqs: Vec<_> = (0..5).map(|i| mk_req(i, 0.8)).collect();
        let window = [30, 40, 50, 60, 70];
        let pre = 100usize;

        // 2 accepted drafts (bit 31 set), row 2 rejected with residual token 77 -> bonus=77, len=103.
        let words = [
            PackedRejectionWord((1 << 31) | 40),
            PackedRejectionWord((1 << 31) | 50),
            PackedRejectionWord(77),
            PackedRejectionWord(70),
            PackedRejectionWord(80),
        ];
        assert_eq!(
            predict_next_bonus_ple_input_rejection(&window, &words, &reqs, pre),
            Some((77, 103))
        );

        // Full accept (4 accepted drafts) -> row 4 bonus token 88, len=105.
        let full_accept = [
            PackedRejectionWord((1 << 31) | 40),
            PackedRejectionWord((1 << 31) | 50),
            PackedRejectionWord((1 << 31) | 60),
            PackedRejectionWord((1 << 31) | 70),
            PackedRejectionWord(88),
        ];
        assert_eq!(
            predict_next_bonus_ple_input_rejection(&window, &full_accept, &reqs, pre),
            Some((88, 105))
        );

        // Any host fallback word returns None.
        let with_fallback = [
            PackedRejectionWord((1 << 31) | 40),
            PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
            PackedRejectionWord(77),
            PackedRejectionWord(70),
            PackedRejectionWord(80),
        ];
        assert_eq!(
            predict_next_bonus_ple_input_rejection(&window, &with_fallback, &reqs, pre),
            None
        );
    }

    #[test]
    fn block_table_stride_buckets_and_graph_keys_stay_tight_and_disjoint() {
        use super::{
            batched_verify_graph_key_with_tail_and_bt_bucket,
            batched_verify_layer0_graph_key_with_bt_bucket, is_batched_verify_graph_meta,
            select_block_table_stride_bucket,
        };

        // 131K context (`max_blocks_per_seq = 2049`, block_size = 64):
        // Bucket 0 (`32` blocks) covers the entire 0..=2048 dense regime in one bucket.
        assert_eq!(select_block_table_stride_bucket(0, 2049), (32, 0));
        assert_eq!(select_block_table_stride_bucket(1, 2049), (32, 0));
        assert_eq!(select_block_table_stride_bucket(4, 2049), (32, 0));
        assert_eq!(select_block_table_stride_bucket(32, 2049), (32, 0));
        // Bucket 1 (`128` blocks) covers 33..=128 blocks (up to 8192 tokens).
        assert_eq!(select_block_table_stride_bucket(33, 2049), (128, 1));
        assert_eq!(select_block_table_stride_bucket(128, 2049), (128, 1));
        // Bucket 2 (`512` blocks) covers 129..=512 blocks (up to 32768 tokens).
        assert_eq!(select_block_table_stride_bucket(129, 2049), (512, 2));
        assert_eq!(select_block_table_stride_bucket(512, 2049), (512, 2));
        // Bucket 3 falls back to full `max_blocks_per_seq`.
        assert_eq!(select_block_table_stride_bucket(513, 2049), (2049, 3));
        assert_eq!(select_block_table_stride_bucket(2049, 2049), (2049, 3));
        // Small model ceilings never round up past `max_blocks_per_seq`.
        assert_eq!(select_block_table_stride_bucket(4, 16), (16, 3));
        assert_eq!(select_block_table_stride_bucket(4, 32), (32, 3));
        assert_eq!(select_block_table_stride_bucket(4, 64), (32, 0));
        assert_eq!(select_block_table_stride_bucket(40, 64), (64, 3));

        // All 4 bucket IDs occupy bits 15..=16 and stay pairwise distinct from
        // every tail mode (including bit 14 rejection tail), QSA dense-prefix
        // bit 17, 10-bit total_rows (0..=1023, so 256 never aliases 0), and
        // Layer-0 prefix tag.
        let suffix_keys: Vec<_> = (0..4usize)
            .map(|b| {
                batched_verify_graph_key_with_tail_and_bt_bucket(
                    vec![0, 1],
                    256,
                    true,
                    true,
                    false,
                    false,
                    true,
                    b,
                )
            })
            .collect();
        let l0_keys: Vec<_> = (0..4usize)
            .map(|b| {
                batched_verify_layer0_graph_key_with_bt_bucket(vec![0, 1], 256, true, true, true, b)
            })
            .collect();
        let zero_row_suffix = batched_verify_graph_key_with_tail_and_bt_bucket(
            vec![0, 1],
            0,
            true,
            true,
            false,
            false,
            true,
            0,
        );
        let zero_row_l0 =
            batched_verify_layer0_graph_key_with_bt_bucket(vec![0, 1], 0, true, true, true, 0);
        assert_ne!(
            suffix_keys[0], zero_row_suffix,
            "256-row suffix key must not wrap/alias 0-row key"
        );
        assert_ne!(
            l0_keys[0], zero_row_l0,
            "256-row Layer-0 key must not wrap/alias 0-row key"
        );
        for i in 0..4 {
            assert!(is_batched_verify_graph_meta(suffix_keys[i].1));
            assert!(is_batched_verify_graph_meta(l0_keys[i].1));
            // Rejection tail bit 14 must remain intact alongside bucket bits 15..=16,
            // and QSA dense-prefix bit 17 must never collide with bucket bits.
            assert_ne!(suffix_keys[i].1 & (1 << 14), 0);
            assert_eq!(
                suffix_keys[i].1 & QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT,
                0
            );
            assert_eq!(l0_keys[i].1 & QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT, 0);
            assert_eq!(suffix_keys[i].1 & 0x3FF, 256);
            assert_eq!(l0_keys[i].1 & 0x3FF, 256);
            for j in 0..4 {
                assert_ne!(suffix_keys[i], l0_keys[j]);
                if i != j {
                    assert_ne!(suffix_keys[i], suffix_keys[j]);
                    assert_ne!(l0_keys[i], l0_keys[j]);
                }
            }
        }
    }

    #[test]
    fn ssm_fused_desc_slab_packer_collapses_36_layers_to_one_h2d_and_respects_skips() {
        use super::pack_ssm_fused_desc_slab_runs;

        let slab = DevicePtr(0x2_0000_0000);
        let stride = crate::layers::qwen3_ssm::CROSS_SEQ_TABLE_BYTES;
        assert_eq!(stride, 4096);

        // 1. All 36 SSM layers on one slab (`16 seqs × 64 B = 1024 B` each)
        //    presented in reverse order -> 1 single H2D spanning `35 * 4096 + 1024`.
        let mut staged: Vec<(DevicePtr, Vec<u8>)> = (0..36usize)
            .rev()
            .map(|layer| {
                let mut bytes = vec![0u8; 1024];
                bytes[0..8].copy_from_slice(&(layer as u64 + 1).to_le_bytes());
                bytes[1016..1024].copy_from_slice(&(0xAA00_0000u64 | layer as u64).to_le_bytes());
                (slab.offset(layer * stride), bytes)
            })
            .collect();
        let mut packed = Vec::new();
        let mut runs = Vec::new();
        pack_ssm_fused_desc_slab_runs(&mut staged, stride, &mut packed, &mut runs);
        assert_eq!(runs, vec![(slab, 0, 35 * stride + 1024)]);
        for layer in 0..36usize {
            let off = layer * stride;
            assert_eq!(
                u64::from_le_bytes(packed[off..off + 8].try_into().unwrap()),
                layer as u64 + 1
            );
            assert_eq!(
                u64::from_le_bytes(packed[off + 1016..off + 1024].try_into().unwrap()),
                0xAA00_0000u64 | layer as u64
            );
        }

        // 2. Skipped layer 17 splits cleanly into 2 runs (`0..17` and `18..36`)
        //    so layer 17's GPU slot is never overwritten.
        let mut with_skip: Vec<(DevicePtr, Vec<u8>)> = (0..36usize)
            .filter(|&l| l != 17)
            .map(|l| {
                (
                    slab.offset(l * stride),
                    vec![(l as u8).wrapping_add(1); 1024],
                )
            })
            .collect();
        pack_ssm_fused_desc_slab_runs(&mut with_skip, stride, &mut packed, &mut runs);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], (slab, 0, 16 * stride + 1024));
        assert_eq!(
            runs[1],
            (slab.offset(18 * stride), runs[0].2, 17 * stride + 1024)
        );

        // 3. Disjoint non-slab pointers stay separate runs.
        let mut disjoint = vec![
            (DevicePtr(0x10_0000), vec![1u8; 512]),
            (DevicePtr(0x50_0000), vec![2u8; 512]),
        ];
        pack_ssm_fused_desc_slab_runs(&mut disjoint, stride, &mut packed, &mut runs);
        assert_eq!(
            runs,
            vec![
                (DevicePtr(0x10_0000), 0, 512),
                (DevicePtr(0x50_0000), 512, 512),
            ]
        );
    }

    #[test]
    fn inter_layer_hc_defer_respects_layer0_ple_and_dflash_capture_boundaries() {
        use super::{layer_has_dflash_capture_tap_raw, layer_should_defer_mlp_tail};

        let taps = [1usize, 12, 23, 34, 45];
        // Layer 0 ALWAYS flushes (Layer-0 prefix graph / PLE / Layer-1 suffix split).
        assert!(!layer_should_defer_mlp_tail(true, 0, false, true, &[]));
        assert!(!layer_should_defer_mlp_tail(true, 0, true, true, &taps));

        // Disabled gate never defers any layer.
        for l in 0..48usize {
            assert!(!layer_should_defer_mlp_tail(false, l, false, true, &taps));
        }

        // Without DFlash hidden capture (e.g. native MTP): all 47 suffix layers
        // (1..48, including Layer 47 -> final_mixer) defer cleanly -> 48 intra + 47 inter = 95.
        let native_inter: usize = (0..48usize)
            .filter(|&l| layer_should_defer_mlp_tail(true, l, false, true, &taps))
            .count();
        assert_eq!(native_inter, 47);

        // With 5 DFlash capture taps on rank 0: each tap flushes before
        // `try_dflash_capture_rows`, leaving 42 inter-layer pairs (48 + 42 = 90).
        for &tap in &taps {
            assert!(layer_has_dflash_capture_tap_raw(true, true, &taps, tap));
            assert!(!layer_should_defer_mlp_tail(true, tap, true, true, &taps));
        }
        let dflash_inter: usize = (0..48usize)
            .filter(|&l| layer_should_defer_mlp_tail(true, l, true, true, &taps))
            .count();
        assert_eq!(dflash_inter, 42);
        // Layer 47 -> final_mixer still fuses when Layer 47 is not a capture tap,
        // and flushes cleanly if a checkpoint taps Layer 47.
        assert!(layer_should_defer_mlp_tail(true, 47, true, true, &taps));
        assert!(!layer_should_defer_mlp_tail(true, 47, true, true, &[47]));

        // Non-rank-0 TP workers do not execute DFlash capture, so all 47 suffix layers defer.
        let worker_inter: usize = (0..48usize)
            .filter(|&l| layer_should_defer_mlp_tail(true, l, true, false, &taps))
            .count();
        assert_eq!(worker_inter, 47);
    }

    #[test]
    fn cached_verify_qwen4_batched_env_helpers_track_live_env_and_fail_closed() {
        use super::{
            batched_k5_dedup_enabled, batched_verify_graph_enabled, k5_hybrid_selected_cached,
            kgamma_debug_dump_absent, qwen4_layer0_prefix_graph_enabled, ssm_h_fp16_enabled_cached,
            verify_logits_receipt_selected_cached,
        };
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        let _lock = test_env_lock();

        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_VERIFY_GRAPH");
            assert!(batched_verify_graph_enabled());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_BATCHED_VERIFY_GRAPH", "0");
            assert!(!batched_verify_graph_enabled());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_BATCHED_VERIFY_GRAPH", "1");
            assert!(batched_verify_graph_enabled());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_LAYER0_PREFIX_GRAPH");
            assert!(qwen4_layer0_prefix_graph_enabled());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_LAYER0_PREFIX_GRAPH", "0");
            assert!(!qwen4_layer0_prefix_graph_enabled());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_LAYER0_PREFIX_GRAPH", "1");
            assert!(qwen4_layer0_prefix_graph_enabled());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_K5_DEDUP");
            assert!(!batched_k5_dedup_enabled());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_BATCHED_K5_DEDUP", "1");
            assert!(batched_k5_dedup_enabled());
        }
        {
            let _g0 = ScopedEnvVar::remove("ATLAS_QWEN4_K5_HYBRID");
            let _g1 = ScopedEnvVar::remove("ATLAS_QWEN4_MTP_BATCH_VERIFY_ROWS5");
            assert!(!k5_hybrid_selected_cached());
            let _gon = ScopedEnvVar::set("ATLAS_QWEN4_K5_HYBRID", "1");
            assert!(k5_hybrid_selected_cached());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_SSM_H_FP16");
            assert!(!ssm_h_fp16_enabled_cached());
            let _g1 = ScopedEnvVar::set("ATLAS_SSM_H_FP16", "1");
            assert!(ssm_h_fp16_enabled_cached());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_KGAMMA_DEBUG_DUMP");
            assert!(kgamma_debug_dump_absent());
            let _g1 = ScopedEnvVar::set("ATLAS_KGAMMA_DEBUG_DUMP", "1");
            assert!(!kgamma_debug_dump_absent());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT");
            assert!(!verify_logits_receipt_selected_cached().unwrap());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT", "1");
            assert!(verify_logits_receipt_selected_cached().unwrap());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT", "0");
            assert!(!verify_logits_receipt_selected_cached().unwrap());
            let _gb = ScopedEnvVar::set("ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT", "bogus");
            assert!(verify_logits_receipt_selected_cached().is_err());
        }
    }

    #[test]
    fn validate_frame_inputs_stack_bitset_and_overflow_slots_preserve_exact_errors() {
        // Distinct slots >= 256 exercise the HashSet fallback cleanly while
        // still enforcing duplicate detection before lane bounds.
        assert!(
            validate_frame_inputs(&windows(2, 5), &[300, 300], 8, 1_000)
                .unwrap_err()
                .contains("distinct SSM slot")
        );
        assert!(
            validate_frame_inputs(&windows(2, 5), &[300, 301], 8, 1_000)
                .unwrap_err()
                .contains("no verify lane")
        );
        // Repeated calls with identical inputs return identical frames.
        let f1 = validate_frame_inputs(&windows(16, 5), &(0..16).collect::<Vec<_>>(), 16, 1_000)
            .unwrap();
        let f2 = validate_frame_inputs(&windows(16, 5), &(0..16).collect::<Vec<_>>(), 16, 1_000)
            .unwrap();
        assert_eq!(f1, f2);
    }

    #[test]
    fn qwen4_cross_seq_admission_is_repeatable_and_preserves_env_refusal_reasons() {
        use std::sync::Arc;

        use atlas_core::config::ModelConfig;
        use spark_runtime::buffers::BufferArena;
        use spark_runtime::gpu::GpuBackend;
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
        use spark_runtime::prefix_cache::NoPrefixCaching;

        use super::TransformerModel;
        use crate::layers::MtpQuantization;
        use crate::layers::qwen4_verify_lanes::{BATCHED_K5_ENV, ScopedEnvVar, test_env_lock};
        use crate::traits::Model;
        use crate::weight_map::DenseWeight;

        let _lock = test_env_lock();

        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "qwen3_next".to_string();
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 64;
        cfg.vocab_size = 128;
        cfg.num_hidden_layers = 0;
        cfg.layer_types = Vec::new();
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 16;
        cfg.hc_count = 0;
        cfg.hc_mult = 0;

        let gpu = Arc::new(MockGpuBackend::new());
        let embed = DenseWeight {
            weight: gpu.alloc(cfg.vocab_size * cfg.hidden_size * 2).unwrap(),
        };
        let final_norm = DenseWeight {
            weight: gpu.alloc(cfg.hidden_size * 2).unwrap(),
        };
        let lm_head = DenseWeight {
            weight: gpu.alloc(cfg.vocab_size * cfg.hidden_size * 2).unwrap(),
        };
        let buffers = BufferArena::new(&cfg, 16, 128, 16, 4, gpu.as_ref()).unwrap();
        let kv_cfg = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 16,
            block_size: 16,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        };
        let kv_cache = PagedKvCache::new(kv_cfg, 64, gpu.as_ref()).unwrap();
        let model = TransformerModel::new(
            cfg,
            embed,
            final_norm,
            None,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            None,
            lm_head,
            None,
            vec![],
            buffers,
            kv_cache,
            vec![],
            None,
            None,
            gpu as Arc<dyn GpuBackend>,
            128,
            4,
            MtpQuantization::Bf16,
            true,
            Box::new(NoPrefixCaching),
            0,
            None,
            true,
            4,
            None,
            0,
            0,
        )
        .unwrap();

        let mut s0 = model.alloc_sequence().unwrap();
        let mut s1 = model.alloc_sequence().unwrap();
        let toks = windows(2, 5);

        let _p0 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_CONF", "0");
        let _p1 = ScopedEnvVar::set("FLUX_QWEN4_MTP_CONF", "0");
        let _p2 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_CONF_EARLY", "0");
        let _p3 = ScopedEnvVar::set("FLUX_QWEN4_MTP_CONF_EARLY", "0");
        let _p4 = ScopedEnvVar::set("ATLAS_QWEN4_MTP_VOCAB_PRUNE", "0");
        let _p5 = ScopedEnvVar::set("FLUX_QWEN4_MTP_VOCAB_PRUNE", "0");
        let _p6 = ScopedEnvVar::remove("ATLAS_QWEN4_MTP_CONF_FILE");
        let _p7 = ScopedEnvVar::remove("FLUX_QWEN4_MTP_CONF_FILE");
        let _p8 = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_K5_LADDER");
        let _p9 = ScopedEnvVar::remove("FLUX_QWEN4_BATCHED_K5_LADDER");
        let _p10 = ScopedEnvVar::remove("ATLAS_QWEN4_BATCHED_K5_GDN_MULTI");
        let _p11 = ScopedEnvVar::remove("FLUX_QWEN4_BATCHED_K5_GDN_MULTI");

        {
            let views = [&s0, &s1];

            // Disabled selector produces the same refusal across repeated calls.
            {
                let _b = ScopedEnvVar::set(BATCHED_K5_ENV, "0");
                let e1 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                let e2 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                assert_eq!(e1, e2);
                assert!(e1.contains("ATLAS_QWEN4_BATCHED_K5=1"), "got {e1}");
            }

            // Invalid selector fails closed identically across repeated calls.
            {
                let _b = ScopedEnvVar::set(BATCHED_K5_ENV, "bogus");
                let e1 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                let e2 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                assert_eq!(e1, e2);
                assert!(e1.contains("exact value 0 or 1"), "got {e1}");
            }

            // Enabled selector + valid fixed-K5 policy advances identically to the
            // staging check across repeated calls.
            {
                let _b = ScopedEnvVar::set(BATCHED_K5_ENV, "1");
                let e1 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                let e2 = model
                    .qwen4_cross_seq_admission(&toks, &views)
                    .unwrap_err()
                    .to_string();
                assert_eq!(e1, e2);
                assert!(e1.contains("not provisioned"), "got {e1}");
            }
        }

        let _ = model.free_sequence(&mut s1);
        let _ = model.free_sequence(&mut s0);
    }

    #[test]
    fn batched_verify_tc_scope_decouples_tc_moe_min_rows_from_lm_head() {
        use super::batched_verify_tc_scope_active;

        // Even when `LmHead` `tc_dense_active == false` (e.g. if `LmHead`'s
        // threshold is overridden above 4), C=2 K=2 (r=4) and r=5 must still
        // arm the TC scope whenever TC dense is enabled so W4A8 TC MoE
        // (`DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS = 4`) can engage.
        assert!(!batched_verify_tc_scope_active(false, true, 2));
        assert!(!batched_verify_tc_scope_active(false, true, 3));
        assert!(batched_verify_tc_scope_active(false, true, 4));
        assert!(batched_verify_tc_scope_active(false, true, 5));
        assert!(batched_verify_tc_scope_active(true, true, 6));

        // Explicitly disabling TC dense (`enabled = false`) keeps TC scope off
        // for strict 0-ULP scalar verification unless `tc_dense_active` is true.
        assert!(!batched_verify_tc_scope_active(false, false, 4));
        assert!(!batched_verify_tc_scope_active(false, false, 16));
        assert!(batched_verify_tc_scope_active(true, false, 6));
    }

    #[test]
    fn pad_ssm_fused_desc_payloads_zero_extends_trailing_slots_and_preserves_single_slab_run() {
        use super::{
            QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS, batched_verify_ssm_desc_pad_seqs,
            pack_ssm_fused_desc_slab_runs, pad_ssm_fused_desc_payloads,
        };
        use crate::layers::ops::GdnFusedMultiSeqDesc;
        use crate::layers::qwen3_ssm::CROSS_SEQ_TABLE_BYTES;
        use spark_runtime::gpu::DevicePtr;

        assert_eq!(batched_verify_ssm_desc_pad_seqs(1), 1);
        for c in 2..QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS {
            assert_eq!(
                batched_verify_ssm_desc_pad_seqs(c),
                QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS
            );
        }
        assert_eq!(batched_verify_ssm_desc_pad_seqs(16), 16);
        assert_eq!(batched_verify_ssm_desc_pad_seqs(20), 20);

        let desc_bytes = GdnFusedMultiSeqDesc::SIZE_BYTES;
        let active_seqs = 13usize;
        let num_layers = 4usize;
        let base = 0x4000_0000u64;
        let mut staged: Vec<(DevicePtr, Vec<u8>)> = (0..num_layers)
            .map(|l| {
                (
                    DevicePtr(base + (l * CROSS_SEQ_TABLE_BYTES) as u64),
                    vec![0xA5u8; active_seqs * desc_bytes],
                )
            })
            .collect();

        pad_ssm_fused_desc_payloads(&mut staged, active_seqs);
        let padded_bytes = QWEN4_VERIFY_GRAPH_BUCKET_MAX_SEQS * desc_bytes;
        assert!(padded_bytes <= CROSS_SEQ_TABLE_BYTES);
        for (_, bytes) in &staged {
            assert_eq!(bytes.len(), padded_bytes);
            assert!(bytes[..active_seqs * desc_bytes].iter().all(|&b| b == 0xA5));
            assert!(
                bytes[active_seqs * desc_bytes..].iter().all(|&b| b == 0),
                "trailing padded slots must be all-zero (`num_tokens = 0, flags = 0`)"
            );
        }

        // Packing into a dirty reused host buffer must overwrite trailing slots
        // with zeros (including on the final layer) while preserving a single slab H2D.
        let mut packed = vec![0xFFu8; num_layers * CROSS_SEQ_TABLE_BYTES];
        let mut runs = Vec::new();
        pack_ssm_fused_desc_slab_runs(&mut staged, CROSS_SEQ_TABLE_BYTES, &mut packed, &mut runs);
        assert_eq!(runs.len(), 1, "padded 1024 B layers must stay one slab H2D");
        assert_eq!(
            runs[0].2,
            (num_layers - 1) * CROSS_SEQ_TABLE_BYTES + padded_bytes
        );
        for l in 0..num_layers {
            let off = l * CROSS_SEQ_TABLE_BYTES;
            assert!(
                packed[off..off + active_seqs * desc_bytes]
                    .iter()
                    .all(|&b| b == 0xA5)
            );
            assert!(
                packed[off + active_seqs * desc_bytes..off + padded_bytes]
                    .iter()
                    .all(|&b| b == 0)
            );
        }
    }

    #[test]
    fn lookup_cached_batched_verify_graph_replays_smallest_superset_bucket_safely() {
        use super::{
            batched_verify_graph_key_slots, batched_verify_graph_key_with_tail_and_bt_bucket,
            batched_verify_layer0_graph_key_with_bt_bucket, lookup_cached_batched_verify_graph,
        };
        use spark_runtime::gpu::GraphHandle;
        use std::collections::HashMap;

        let mk_suffix = |seqs: usize, rows: usize, tc: bool| {
            let slots = batched_verify_graph_key_slots((0..seqs).collect(), true, true);
            batched_verify_graph_key_with_tail_and_bt_bucket(
                slots, rows, true, tc, false, false, true, 0,
            )
        };
        let mk_layer0 = |seqs: usize, rows: usize, tc: bool| {
            let slots = batched_verify_graph_key_slots((0..seqs).collect(), true, true);
            batched_verify_layer0_graph_key_with_bt_bucket(slots, rows, true, tc, true, 0)
        };

        let mut graphs = HashMap::new();
        let k16_r48 = mk_suffix(16, 48, true);
        let k8_r48 = mk_suffix(8, 48, true);
        let k4_r16 = mk_suffix(4, 16, true);
        let k16_l0_r48 = mk_layer0(16, 48, true);
        graphs.insert(k16_r48.clone(), GraphHandle(160));
        graphs.insert(k8_r48.clone(), GraphHandle(80));
        graphs.insert(k4_r16.clone(), GraphHandle(40));
        graphs.insert(k16_l0_r48.clone(), GraphHandle(161));

        let lookup_id = |primary: &(Vec<usize>, usize), seqs: usize, allow: bool| {
            lookup_cached_batched_verify_graph(&graphs, primary, seqs, allow).map(|(k, g)| (k, g.0))
        };

        // Draining cohorts `9..=15` with `u_rows = 48` replay the cached `N=16`
        // graph immediately instead of running 3 eager warmup steps per cohort.
        for draining in 9..=15 {
            let primary = mk_suffix(draining, 48, true);
            assert_eq!(
                lookup_id(&primary, draining, true),
                Some((k16_r48.clone(), 160))
            );
            let l0_primary = mk_layer0(draining, 48, true);
            assert_eq!(
                lookup_id(&l0_primary, draining, true),
                Some((k16_l0_r48.clone(), 161))
            );
        }

        // Smaller cohorts (`5..=7` at `u_rows = 48`) pick the smallest superset
        // bucket (`N=8` over `N=16`) to minimize no-op GDN CTAs.
        for draining in 5..=7 {
            let primary = mk_suffix(draining, 48, true);
            assert_eq!(
                lookup_id(&primary, draining, true),
                Some((k8_r48.clone(), 80))
            );
        }

        // Exact match always wins over a larger superset bucket.
        let k8_exact = mk_suffix(8, 48, true);
        assert_eq!(lookup_id(&k8_exact, 8, true), Some((k8_r48.clone(), 80)));

        // `C=3` at `u_rows = 16` hits `N=4`; `C=5` NEVER hits `N=4` (strictly
        // forbids replaying a smaller captured grid over more active sequences).
        assert_eq!(
            lookup_id(&mk_suffix(3, 16, true), 3, true),
            Some((k4_r16.clone(), 40))
        );
        assert_eq!(lookup_id(&mk_suffix(5, 16, true), 5, true), None);

        // Different `meta` (`tc_scope_active`, `total_rows`, or Layer-0 vs Suffix)
        // and `allow_superset_arity == false` never cross-alias.
        assert_eq!(lookup_id(&mk_suffix(15, 48, false), 15, true), None);
        assert_eq!(lookup_id(&mk_suffix(15, 64, true), 15, true), None);
        assert_eq!(lookup_id(&mk_suffix(15, 48, true), 15, false), None);
    }

    #[test]
    fn effective_batched_verify_graph_cap_and_uma_eviction_preserve_active_layer0_pair() {
        use super::{
            BatchedVerifyGraphModelState, QWEN4_BATCHED_VERIFY_GRAPH_CAP,
            QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_CAP, QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_CAP,
            QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_CAP, batched_verify_graph_key_slots,
            batched_verify_graph_key_with_tail_and_bt_bucket,
            batched_verify_layer0_graph_key_with_bt_bucket, effective_batched_verify_graph_cap,
        };
        use spark_runtime::gpu::GraphHandle;
        use spark_runtime::startup_ledger::MIB;
        use std::collections::HashMap;

        let floor = 6_656 * MIB;
        let base = QWEN4_BATCHED_VERIFY_GRAPH_CAP;
        assert_eq!(base, 48);

        // Unknown `MemAvailable` or floor falls back cleanly to `base_cap`.
        assert_eq!(
            effective_batched_verify_graph_cap(base, None, Some(floor)),
            base
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor), None),
            base
        );
        assert_eq!(effective_batched_verify_graph_cap(base, None, None), base);

        // Tier 1: headroom >= 3072 MiB keeps full `base_cap` (zero extra evictions).
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 3_072 * MIB), Some(floor)),
            base
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 4_096 * MIB), Some(floor)),
            base
        );

        // Tier 2: 1024..3072 MiB headroom shrinks to `base_cap.min(16)`.
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 3_072 * MIB - 1), Some(floor)),
            QWEN4_BATCHED_VERIFY_GRAPH_PRESSURE_CAP
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 3_071 * MIB), Some(floor)),
            16
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 1_024 * MIB), Some(floor)),
            16
        );

        // Tier 3: 512..1024 MiB headroom shrinks to `base_cap.min(8)`.
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 1_024 * MIB - 1), Some(floor)),
            QWEN4_BATCHED_VERIFY_GRAPH_CRITICAL_CAP
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 1_023 * MIB), Some(floor)),
            8
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 512 * MIB), Some(floor)),
            8
        );

        // Tier 4: < 512 MiB headroom (including below floor) shrinks to `base_cap.min(2)`.
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 512 * MIB - 1), Some(floor)),
            QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_CAP
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor + 511 * MIB), Some(floor)),
            2
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor), Some(floor)),
            2
        );
        assert_eq!(
            effective_batched_verify_graph_cap(base, Some(floor - 42 * MIB), Some(floor)),
            2
        );

        // Smaller `base_cap` is never exceeded and clamps 0 to 1.
        assert_eq!(
            effective_batched_verify_graph_cap(6, Some(floor + 1_500 * MIB), Some(floor)),
            6
        );
        assert_eq!(
            effective_batched_verify_graph_cap(6, Some(floor + 700 * MIB), Some(floor)),
            6
        );
        assert_eq!(
            effective_batched_verify_graph_cap(6, Some(floor + 256 * MIB), Some(floor)),
            2
        );
        assert_eq!(
            effective_batched_verify_graph_cap(1, Some(floor + 256 * MIB), Some(floor)),
            1
        );
        assert_eq!(effective_batched_verify_graph_cap(0, None, Some(floor)), 1);

        // Under emergency pressure (`cap = 2`), shrinking a populated cache evicts
        // idle graphs while NEVER evicting the active Layer-0 prefix graph from
        // the same verify step — even if it sits at the front of the LRU deque.
        let active_l0_key = batched_verify_layer0_graph_key_with_bt_bucket(
            batched_verify_graph_key_slots(vec![0, 1, 2, 3], true, true),
            16,
            true,
            true,
            true,
            0,
        );
        let mut state = BatchedVerifyGraphModelState::default();
        let mut live: HashMap<(Vec<usize>, usize), GraphHandle> = HashMap::new();
        live.insert(active_l0_key.clone(), GraphHandle(1));
        state.touch(&active_l0_key);
        for idx in 1..30usize {
            let key = batched_verify_graph_key_with_tail_and_bt_bucket(
                batched_verify_graph_key_slots((0..4).collect(), true, true),
                idx,
                true,
                true,
                false,
                false,
                true,
                0,
            );
            live.insert(key.clone(), GraphHandle((idx + 1) as u64));
            state.touch(&key);
        }

        let evicted = state.collect_evictions_preserving(
            &live,
            QWEN4_BATCHED_VERIFY_GRAPH_EMERGENCY_CAP,
            Some(&active_l0_key),
        );
        assert_eq!(
            evicted.len(),
            29,
            "all 29 non-preserved suffix graphs should be evicted under emergency cap=2"
        );
        assert!(
            !evicted.contains(&active_l0_key),
            "active Layer-0 prefix graph launched in the same step must never be evicted"
        );
        for key in &evicted {
            live.remove(key);
        }
        assert_eq!(live.len(), 1);
        assert!(live.contains_key(&active_l0_key));

        // Even at `cap = 1`, `preserve_key` must still never be evicted and must terminate.
        let evicted_cap1 = state.collect_evictions_preserving(&live, 1, Some(&active_l0_key));
        assert!(evicted_cap1.is_empty());
        assert!(live.contains_key(&active_l0_key));
    }

    #[test]
    fn batched_verify_qsa_dense_prefix_and_wide_sm_buckets_and_warmup_rules() {
        use std::collections::HashMap;

        use spark_runtime::gpu::GraphHandle;

        use super::{
            QWEN4_BATCHED_VERIFY_GRAPH_CAP, QWEN4_BATCHED_VERIFY_GRAPH_CAP_WIDE_SM,
            QWEN4_VERIFY_GRAPH_SEQ_BUCKETS, QWEN4_VERIFY_GRAPH_SEQ_BUCKETS_WIDE_SM,
            batched_verify_graph_base_cap_for_sm, batched_verify_graph_key_slots,
            batched_verify_graph_key_with_tail_and_bt_bucket,
            batched_verify_graph_required_warmup_hits_for_sm,
            batched_verify_qsa_dense_prefix_eligible, batched_verify_seq_buckets_for_sm,
            batched_verify_ssm_desc_pad_seqs_for_sm, is_batched_verify_graph_meta,
            lookup_cached_batched_verify_graph_for_sm, with_qsa_dense_prefix_graph_bit,
        };

        // QSA dense-prefix admission mirrors verify_d (all QSA seqs seq_len + k <= 2048 & k >= 2).
        assert!(batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(true, 1024, 8), (true, 2040, 8)]
        ));
        assert!(batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(false, 4096, 8), (true, 2040, 8)]
        ));
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(false, 100, 8), (false, 200, 8)]
        ));
        // Straddling 2048 or post-2048 or K=1 (non-journal) must stay eager.
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(true, 2041, 8), (true, 100, 8)]
        ));
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(true, 2048, 2), (true, 100, 2)]
        ));
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            true,
            [(true, 100, 1), (true, 200, 8)]
        ));
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            false,
            true,
            [(true, 100, 8), (true, 200, 8)]
        ));
        assert!(!batched_verify_qsa_dense_prefix_eligible(
            2,
            true,
            false,
            [(true, 100, 8), (true, 200, 8)]
        ));

        // Graph key bit 17 (`QWEN4_BATCHED_VERIFY_QSA_DENSE_PREFIX_BIT`)
        // strictly isolates QSA dense-prefix from pure non-QSA graphs while
        // keeping the shared batched-verify invalidation tag intact.
        let non_qsa = batched_verify_graph_key_with_tail_and_bt_bucket(
            batched_verify_graph_key_slots(vec![0, 1], true, true),
            16,
            true,
            true,
            true,
            false,
            false,
            0,
        );
        let qsa_dp = with_qsa_dense_prefix_graph_bit(non_qsa.clone(), true);
        assert_ne!(non_qsa, qsa_dp);
        assert!(is_batched_verify_graph_meta(qsa_dp.1));

        // GB10 (<= 64 SMs): legacy buckets [2, 4, 8, 16], transient warmup=3, cap=48.
        assert_eq!(
            batched_verify_seq_buckets_for_sm(48),
            &QWEN4_VERIFY_GRAPH_SEQ_BUCKETS
        );
        assert_eq!(batched_verify_ssm_desc_pad_seqs_for_sm(3, 48), 16);
        assert_eq!(batched_verify_ssm_desc_pad_seqs_for_sm(20, 48), 20);
        assert_eq!(batched_verify_graph_required_warmup_hits_for_sm(3, 48), 3);
        assert_eq!(batched_verify_graph_required_warmup_hits_for_sm(24, 48), 3);
        assert_eq!(
            batched_verify_graph_base_cap_for_sm(48),
            QWEN4_BATCHED_VERIFY_GRAPH_CAP
        );

        // SM120 (> 64 SMs): extended [2, 4, 8, 16, 24, 32, 48, 64] buckets, warmup=1, cap=64.
        assert_eq!(
            batched_verify_seq_buckets_for_sm(188),
            &QWEN4_VERIFY_GRAPH_SEQ_BUCKETS_WIDE_SM
        );
        assert_eq!(batched_verify_ssm_desc_pad_seqs_for_sm(20, 188), 64);
        assert_eq!(batched_verify_ssm_desc_pad_seqs_for_sm(64, 188), 64);
        for c in [1usize, 2, 4, 8, 16, 24, 32, 48, 64] {
            assert_eq!(
                batched_verify_graph_required_warmup_hits_for_sm(c, 188),
                1,
                "sm=188 steady c={c}"
            );
        }
        for c in [3usize, 5, 15, 20, 29] {
            assert_eq!(
                batched_verify_graph_required_warmup_hits_for_sm(c, 188),
                3,
                "sm=188 transient draining c={c}"
            );
        }
        assert_eq!(
            batched_verify_graph_base_cap_for_sm(188),
            QWEN4_BATCHED_VERIFY_GRAPH_CAP_WIDE_SM
        );

        // Superset arity lookup on SM120 matches 20 -> 24, 30 -> 32, 40 -> 48, 50 -> 64.
        let mut graphs = HashMap::new();
        for &bucket in &[24usize, 32, 48, 64] {
            let k = batched_verify_graph_key_with_tail_and_bt_bucket(
                batched_verify_graph_key_slots((0..bucket).collect(), true, true),
                64,
                true,
                true,
                true,
                false,
                false,
                0,
            );
            graphs.insert(k, GraphHandle(bucket as u64));
        }
        for (actual, want_bucket) in [(20usize, 24u64), (30, 32), (40, 48), (50, 64)] {
            let primary = batched_verify_graph_key_with_tail_and_bt_bucket(
                batched_verify_graph_key_slots((0..actual).collect(), true, true),
                64,
                true,
                true,
                true,
                false,
                false,
                0,
            );
            assert_eq!(
                lookup_cached_batched_verify_graph_for_sm(&graphs, &primary, actual, true, 188)
                    .map(|(_, g)| g.0),
                Some(want_bucket),
                "actual={actual}"
            );
            // GB10 (<= 64 SMs) does not probe buckets > 16.
            assert!(
                lookup_cached_batched_verify_graph_for_sm(&graphs, &primary, actual, true, 48)
                    .is_none()
            );
        }
    }
}
