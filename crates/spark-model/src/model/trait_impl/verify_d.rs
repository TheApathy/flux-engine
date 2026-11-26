// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! Two kinds of `unsafe` blocks appear here, each with an inline
//! `// SAFETY:` comment:
//!
//! - `std::slice::from_raw_parts(...)` reinterprets initialized stack arrays /
//!   `Vec`s of POD integers (`u32`, `i32`, `i64`) as byte slices for H2D
//!   upload. The length is always the source's byte size and the source
//!   outlives the (synchronous or pinned-ring-staged) copy.
//! - `GpuBackend::copy_h2d_pinned_async(...)` enqueues an async DMA from a
//!   persistent pinned-host shadow buffer. The shadow is owned by the model
//!   (lives for its lifetime), and the caller holds its lock while staging,
//!   so the source stays valid until the stream-ordered copy completes.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{
    PinnedMetaStaging, TransformerModel, VerifyKgammaGraphKey, VerifyKgammaGraphLru,
};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

pub const MTP_BATCH_VERIFY_ROWS5_SELECTOR: &str = "ATLAS_QWEN4_MTP_BATCH_VERIFY_ROWS5";

pub fn k5_hybrid_selected() -> bool {
    std::env::var("ATLAS_QWEN4_K5_HYBRID").ok().as_deref() == Some("1")
        || std::env::var(MTP_BATCH_VERIFY_ROWS5_SELECTOR)
            .ok()
            .as_deref()
            == Some("1")
}

// ── ATLAS_LAYER_RESOLUTION_PROBE ────────────────────────────────────────────
/// `ATLAS_VERIFY_ARGMAX_ROWS=1`: the K=γ verify argmax runs as one
/// `argmax_bf16_rows` launch instead of one single-block `argmax_bf16` per row
/// (same index per row; verify-cost argmax_rows_bench).
pub(super) fn verify_argmax_rows_selected() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let on = std::env::var("ATLAS_VERIFY_ARGMAX_ROWS").ok().as_deref() == Some("1");
        if on {
            tracing::info!("ATLAS_VERIFY_ARGMAX_ROWS=1 requested: row-parallel verify argmax");
        }
        on
    })
}

/// Receipt for the first row-parallel argmax launch (the env read alone proves nothing).
static VERIFY_ARGMAX_ROWS_LAUNCHED: std::sync::Once = std::sync::Once::new();

// Measures at which layer the K=γ verify argmax stabilises.  Run with
// ATLAS_LAYER_RESOLUTION_PROBE=1 (forces eager mode).  Every
// ATLAS_LRP_DUMP_EVERY steps (default 100) logs per-checkpoint match rates.

pub(super) fn layer_resolution_probe_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_LAYER_RESOLUTION_PROBE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn lrp_dump_every() -> u64 {
    static VAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ATLAS_LRP_DUMP_EVERY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100)
    })
}

/// 0-indexed layer indices at which hidden state is snapshotted.
/// Covers the full 64-layer depth at ~8-layer resolution.
const LRP_PROBE_LAYERS: &[usize] = &[7, 15, 23, 31, 39, 47, 55];

struct LrpAccum {
    step: u64,
    /// matches[probe_idx][token_pos] = steps where probe argmax == final argmax.
    matches: Vec<Vec<u64>>,
    k: usize,
}

impl LrpAccum {
    fn new(k: usize) -> Self {
        Self {
            step: 0,
            matches: vec![vec![0u64; k]; LRP_PROBE_LAYERS.len()],
            k,
        }
    }

    fn accumulate(&mut self, final_am: &[u32], probe_ams: &[Vec<u32>]) {
        let k = self.k.min(final_am.len());
        self.step += 1;
        for (pi, pam) in probe_ams.iter().enumerate() {
            for t in 0..k {
                if pam.get(t).copied() == Some(final_am[t]) {
                    self.matches[pi][t] += 1;
                }
            }
        }
    }

    fn dump(&self, num_layers: usize) {
        let steps = self.step.max(1);
        let mut msg = format!("LRP step={} K={}:", self.step, self.k);
        for (pi, &layer_idx) in LRP_PROBE_LAYERS.iter().enumerate() {
            let matched: u64 = self.matches[pi].iter().sum();
            let total = steps * self.k as u64;
            let rate = matched as f64 / total as f64;
            msg.push_str(&format!(" L{}={:.3}", layer_idx + 1, rate));
        }
        // The final layer agrees with the final argmax by definition.
        msg.push_str(&format!(" L{num_layers}=1.000"));
        tracing::info!("{msg}");
    }
}

static LRP_ACCUM: std::sync::OnceLock<Mutex<LrpAccum>> = std::sync::OnceLock::new();
// ── end LRP ─────────────────────────────────────────────────────────────────

/// Reusable host staging buffers for `decode_verify_graphed_kgamma_dispatch`
/// metadata upload, avoiding per-step `k * max_blocks * 4` (~327 KiB) heap
/// allocation, page-fault zeroing, and `munmap` churn.
#[derive(Default)]
struct VerifyDMetaHostScratch {
    slots: Vec<i64>,
    seq_lens: Vec<i32>,
    block_tables: Vec<i32>,
    dirty_cols_per_row: Vec<usize>,
    last_mb: usize,
}

impl VerifyDMetaHostScratch {
    fn prepare(&mut self, k: usize, mb: usize) {
        let needed = k.saturating_mul(mb);
        self.slots.clear();
        self.slots.reserve(k);
        self.seq_lens.clear();
        self.seq_lens.reserve(k);
        if self.last_mb != mb {
            if self.block_tables.len() < needed {
                self.block_tables.resize(needed, 0);
            }
            self.block_tables.fill(0);
            self.dirty_cols_per_row.clear();
            self.last_mb = mb;
        } else if self.block_tables.len() < needed {
            self.block_tables.resize(needed, 0);
        }
        if self.dirty_cols_per_row.len() < k {
            self.dirty_cols_per_row.resize(k, 0);
        }
    }

    fn fill_block_table_rows(&mut self, k: usize, mb: usize, blocks: &[u32]) {
        let written = blocks.len().min(mb);
        for row in 0..k {
            let row_base = row * mb;
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
}

static VERIFY_D_META_SCRATCH: std::sync::OnceLock<Mutex<VerifyDMetaHostScratch>> =
    std::sync::OnceLock::new();

/// Static verify-layer-skip set parsed once from `ATLAS_VERIFY_SKIP_LAYERS`
/// (comma-separated layer indices). Empty when unset. See the layer loop in
/// `decode_verify_graphed_kgamma_dispatch` for semantics (identity pass-through,
/// graph-safe, pass@1-gated). The "past-the-dense-read" bandwidth lever.
fn verify_skip_layer(idx: usize) -> bool {
    use std::sync::OnceLock;
    static SET: OnceLock<Vec<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        std::env::var("ATLAS_VERIFY_SKIP_LAYERS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default()
    })
    .contains(&idx)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen4VerifyLayerRoute {
    RowSerialOracle,
    QualifiedBatched,
    ExperimentalK16Batched,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen4VerifyRouteIdentity {
    target_is_qwen4: bool,
    native_dflash_pairing: bool,
    active_proposer_is_dflash: bool,
    active_proposer_physical_k: Option<usize>,
}

const QWEN4_K16_BATCHED_VERIFY_ENV: &str = "ATLAS_QWEN4_K16_BATCHED_VERIFY";

/// Parse `ATLAS_QWEN4_C1_TC_VERIFY`: explicit `"1"`/`"true"` enables,
/// explicit `"0"`/`"false"` or unset stays disabled (preserving strict `C=1`
/// `0-ULP` bit-exact greedy parity by default).
#[must_use]
fn parse_qwen4_c1_tc_verify(raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        _ => false,
    }
}

/// Opt-in gate (`ATLAS_QWEN4_C1_TC_VERIFY=1`, default OFF) enabling non-0-ULP
/// batched-verify TC dense/MoE routing and greedy coalesced NVFP4 LM-head
/// routing inside single-sequence `verify_d`. Keeping this default OFF
/// preserves strict `C=1` `0-ULP` bit-exact greedy parity against scalar
/// reference decode (`probe_flat_c1`) while retaining Check-A pre-warm, QSA
/// dense-prefix graphing, PLE overlap, batched dense attention, and pinned
/// argmax readback.
fn qwen4_c1_tc_verify_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let raw = std::env::var("ATLAS_QWEN4_C1_TC_VERIFY").ok();
        parse_qwen4_c1_tc_verify(raw.as_deref())
    })
}

/// Pure eligibility predicate for C=1 coalesced NVFP4 LM head in `verify_d`.
///
/// Engages for non-greedy Gumbel/rejection rows (`staged_non_greedy == true`)
/// OR greedy `T == 0` when explicitly opted in via `ATLAS_QWEN4_C1_TC_VERIFY=1`
/// (preserving strict GB10 default 0-ULP greedy parity when unset/`"0"`).
#[must_use]
const fn c1_coalesced_lm_head_eligible(
    gumbel_tail_active: bool,
    rejection_tail_active: bool,
    staged_non_greedy: bool,
    c1_tc_verify: bool,
    serial_lm_head: bool,
    dfs_active: bool,
    stage_capture: bool,
    receipt_on: bool,
    k: usize,
    has_lm_head_nvfp4: bool,
    lm_head_tc_explicit: bool,
    high_precision_lm_head: bool,
    coalesced_dense_enabled: bool,
) -> bool {
    let mode_allows = ((gumbel_tail_active || rejection_tail_active) && staged_non_greedy)
        || (c1_tc_verify && !serial_lm_head && !dfs_active && !stage_capture && !receipt_on);
    mode_allows
        && k >= 4
        && has_lm_head_nvfp4
        && !lm_head_tc_explicit
        && !high_precision_lm_head
        && coalesced_dense_enabled
}

/// `ATLAS_QWEN4_VERIFY_GRAPH`: capture the Qwen4 verify (layers 1.. and the
/// head) in a CUDA graph behind an eager layer-0 + PLE prefix.
///
/// On wide discrete GPUs (`active_sm_count() > 64`, e.g. SM120/SM103), defaults
/// to enabled when unset/`"auto"` (matching `ATLAS_QWEN4_BATCHED_VERIFY_GRAPH`
/// and `ATLAS_QWEN4_MTP_GRAPH`); GB10 (`<= 64` SMs) keeps opt-in (`"1"`).
fn parse_qwen4_verify_graph_env(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "on" | "true" | "yes") => true,
        Some("0" | "off" | "false" | "no") => false,
        None | Some("" | "auto") => sm_count > 64,
        Some(_) => false,
    }
}

fn qwen4_verify_graph_enabled() -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = OVERRIDE.get_or_init(|| std::env::var("ATLAS_QWEN4_VERIFY_GRAPH").ok());
    parse_qwen4_verify_graph_env(raw.as_deref(), atlas_core::device::active_sm_count())
}

fn qwen4_qsa_dense_prefix_graph_eligible(
    qsa_required: bool,
    host_seq_len: usize,
    rows: usize,
    qsa_dense_prefix_fused: bool,
) -> bool {
    qsa_required
        && qsa_dense_prefix_fused
        && rows >= 2
        && host_seq_len
            .checked_add(rows)
            .is_some_and(|end| end <= crate::layers::qwen4_qsa::VERIFY_ROWS_DENSE_ENDPOINT)
}

fn qwen4_qsa_graph_route_safe(
    qsa_required: bool,
    host_seq_len: usize,
    rows: usize,
    qsa_dense_prefix_fused: bool,
    device_attention_topology: bool,
    graph_safe_qsa: bool,
) -> bool {
    !qsa_required
        || device_attention_topology
        || qwen4_qsa_dense_prefix_graph_eligible(
            qsa_required,
            host_seq_len,
            rows,
            qsa_dense_prefix_fused,
        )
        || (graph_safe_qsa && host_seq_len >= 2_048)
}

fn qwen4_qsa_verify_frame_required(target_is_qwen4: bool, qsa_required: bool, rows: usize) -> bool {
    target_is_qwen4 && qsa_required && rows > 1
}

fn qwen4_qsa_verify_frame_admission(
    journal_available: bool,
    tree_frame_active: bool,
) -> std::result::Result<(), &'static str> {
    if !journal_available {
        return Err("Qwen4 QSA verify requires undo journaling for its exact physical width");
    }
    if tree_frame_active {
        return Err("Qwen4 QSA verify transaction currently requires a flat candidate chain");
    }
    Ok(())
}

/// Pure predicate for whether single-sequence Qwen4 verify (`verify_d.rs`) may
/// defer layer `layer_idx`'s trailing MLP `inject_saved_batched` into the next
/// layer's `attn_hyper.prepare_batched_into_after_inject` (or `qwen4_final_mixer`
/// after the last layer).
///
/// Layer 0 MUST always flush because PLE reads/updates `hidden` between Layer 0
/// and Layer 1. DFlash capture tap layers and diagnostic dump layers MUST also
/// flush so `hidden` is materialized in DRAM before capture/dump reads it.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
fn qwen4_verify_d_layer_should_defer_mlp_tail(
    fuse_enabled: bool,
    batched_route: bool,
    hybrid_rows: bool,
    layer_idx: usize,
    is_last_layer: bool,
    can_fuse_into_final_mixer: bool,
    has_dflash_save: bool,
    is_rank0: bool,
    capture_layers: &[usize],
    diag_reads_hidden: bool,
) -> bool {
    let is_capture_tap = has_dflash_save && is_rank0 && capture_layers.contains(&layer_idx);
    fuse_enabled
        && batched_route
        && hybrid_rows
        && layer_idx > 0
        && !is_capture_tap
        && !diag_reads_hidden
        && (!is_last_layer || can_fuse_into_final_mixer)
}

const QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT: u32 = 1 << 3;
const QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT: u32 = 1 << 4;
const QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT: u32 = 1 << 5;
const QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT: u32 = 1 << 6;
const QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT: u32 = 1 << 7;
const QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT: u32 = 1 << 8;
const QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT: u32 = 1 << 9;
const QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT: u32 = 1 << 10;
/// Bit 11: Unified 48-layer C=1 verify CUDA graph (`embed -> Layer 0 -> PLE -> Layers 1..47 -> head`)
/// used when PLE is `mmap`-backed (`--ple-ram` / `ATLAS_PLE_MMAP=1`).
const QWEN4_UNIFIED_PLE_VERIFY_GRAPH_BIT: u32 = 1 << 11;
/// Bit 12: Unified 48-layer verify graph overlaps PLE `dequant + key_proj + value_proj`
/// on `secondary_stream` concurrently with `embed + Layer 0` via in-graph fork/join events.
const QWEN4_UNIFIED_PLE_OVERLAP_VERIFY_GRAPH_BIT: u32 = 1 << 12;

/// Encode unified 48-layer mmap PLE verify graph topology bits for `pack_key`.
#[inline]
#[must_use]
const fn verify_d_unified_ple_graph_bits(
    unified_ple: bool,
    embed_fused: bool,
    overlap: bool,
) -> u32 {
    if !unified_ple {
        return 0;
    }
    let mut bits = QWEN4_UNIFIED_PLE_VERIFY_GRAPH_BIT;
    if embed_fused {
        bits |= QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT;
    }
    if overlap {
        bits |= QWEN4_UNIFIED_PLE_OVERLAP_VERIFY_GRAPH_BIT;
    }
    bits
}

/// Pure admission predicate for collapsing C=1 Qwen4 verify into ONE unified
/// 48-layer CUDA graph (inline `mmap` PLE upload before Layer 0, in-graph PLE
/// `dequant + key_proj + value_proj + fuse` between Layer 0 and Layer 1).
#[inline]
#[must_use]
const fn qwen4_unified_ple_verify_graph_eligible(
    qwen4_layer0_prefix_graph: bool,
    num_layers: usize,
    skip_layer1: bool,
    pp_world_size: usize,
    kgamma_debug_dump: bool,
    ple_mmap_eligible: bool,
) -> bool {
    qwen4_layer0_prefix_graph
        && num_layers >= 2
        && !skip_layer1
        && pp_world_size <= 1
        && !kgamma_debug_dump
        && ple_mmap_eligible
}

/// Encode the Qwen4 QSA attention-topology bits for `pack_key`.
///
/// Separates all three C=1 attention topologies into pairwise-disjoint keys:
/// - `qsa_required == false`: `0` (dense attention without QSA side-cache updates)
/// - `qsa_required == true && qsa_dense_prefix == true`:
///   `QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT`
///   (pre-2048 dense-prefix attention with fused QSA staging)
/// - `qsa_required == true && qsa_dense_prefix == false`:
///   `QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT` (post-2048 sparse QSA top-k attention)
#[inline]
#[must_use]
const fn verify_d_qsa_graph_bits(qsa_required: bool, qsa_dense_prefix: bool) -> u32 {
    let mut bits = 0u32;
    if qsa_required {
        bits |= QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT;
        if qsa_dense_prefix {
            bits |= QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT;
        }
    }
    bits
}

#[inline]
#[must_use]
const fn verify_d_gumbel_tail_graph_bits(in_graph_gumbel: bool, coalesced_lm_head: bool) -> u32 {
    let mut bits = 0u32;
    if in_graph_gumbel {
        bits |= QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT;
    }
    if coalesced_lm_head {
        bits |= QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT;
    }
    bits
}

/// Build the C=1 Layer-0 prefix (`embed -> Layer 0`) `pack_key` stored in
/// `TransformerModel::verify_kgamma_graph` alongside suffix graphs (inheriting
/// automatic invalidation in `free_sequence`, `compact_sequence`,
/// `invalidate_verify_graphs_for_slot`, and `Drop`).
///
/// Strips suffix-only LM-head / Gumbel / rejection-tail bits (bits 5, 6, 9) so
/// greedy (`T == 0`), Gumbel (`T > 0`), and maximal-coupling rejection verifies
/// share ONE C=1 Layer-0 prefix graph while preserving route/tree/QSA topology
/// bits (bits 0..=4, 10) and distinguishing fused vs split embedding kernels (bit 8).
#[inline]
#[must_use]
const fn verify_d_layer0_prefix_pack_key(pack_key: u32, embed_fused: bool) -> u32 {
    let shared_topology = pack_key
        & !(QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT
            | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT
            | QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT);
    let embed_bit = if embed_fused {
        QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT
    } else {
        0
    };
    shared_topology | QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT | embed_bit
}

/// Mask selecting the Qwen4 QSA attention-regime bits in `pack_key`:
/// - `0`: non-QSA dense attention
/// - `QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT`:
///   pre-2048 QSA dense-prefix attention with fused QSA staging
/// - `QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT`: post-2048 sparse QSA top-k attention
///
/// Both suffix graphs and Layer-0 prefix graphs preserve these exact two bits.
const VERIFY_KGAMMA_QSA_REGIME_MASK: u32 =
    QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT;

/// Environment override for the per-tier global entry cap on
/// `TransformerModel::verify_kgamma_graph` (suffix tier and Layer-0 prefix tier
/// are each bounded to at most this many entries).
const VERIFY_KGAMMA_GRAPH_MAX_ENTRIES_ENV: &str = "ATLAS_QWEN4_VERIFY_GRAPH_MAX_ENTRIES";

/// Default maximum cached suffix graphs (and matching Layer-0 prefix graphs) in
/// `verify_kgamma_graph`. Holds 12 entries per tier (~1.5× a full slot's
/// `K=2..8` tiers) with zero evictions on `C=1` workloads while keeping
/// steady-state UMA graph memory well below the 6,656 MiB floor.
const VERIFY_KGAMMA_GRAPH_MAX_ENTRIES: usize = 12;

/// Minimum UMA headroom above the effective memory floor (`3 GiB`) to keep the
/// full verify-graph cache cap before shrinking to `PRESSURE_CAP`.
const VERIFY_KGAMMA_GRAPH_PRESSURE_MARGIN_BYTES: usize = 3072 * spark_runtime::startup_ledger::MIB;

/// Tight UMA headroom threshold (`1 GiB` above the effective memory floor) below
/// which the verify-graph cap shrinks aggressively before capturing new graphs.
const VERIFY_KGAMMA_GRAPH_CRITICAL_MARGIN_BYTES: usize = 1024 * spark_runtime::startup_ledger::MIB;

/// Severe UMA headroom threshold (`512 MiB` above the effective memory floor)
/// below which at most one verify graph per tier stays resident.
const VERIFY_KGAMMA_GRAPH_EMERGENCY_MARGIN_BYTES: usize = 512 * spark_runtime::startup_ledger::MIB;

/// Per-tier cap under moderate UMA pressure (`floor + 1 GiB <= MemAvailable < floor + 3 GiB`,
/// i.e. below `VERIFY_KGAMMA_GRAPH_PRESSURE_MARGIN_BYTES`): keeps one full
/// active slot (`K=2..8` + 1 spare) warm while freeing idle slots.
const VERIFY_KGAMMA_GRAPH_PRESSURE_CAP: usize = 8;

/// Per-tier cap under critical UMA pressure (`floor + 512 MiB <= MemAvailable < floor + 1 GiB`).
const VERIFY_KGAMMA_GRAPH_CRITICAL_CAP: usize = 2;

/// Per-tier cap under emergency UMA pressure (`MemAvailable < floor + 512 MiB`).
const VERIFY_KGAMMA_GRAPH_EMERGENCY_CAP: usize = 1;

#[inline]
#[must_use]
const fn verify_kgamma_is_layer0_prefix_key(key: &VerifyKgammaGraphKey) -> bool {
    (key.2 & QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT) != 0
}

#[inline]
#[must_use]
const fn verify_kgamma_qsa_regime_bits(pack_key: u32) -> u32 {
    pack_key & VERIFY_KGAMMA_QSA_REGIME_MASK
}

fn parse_verify_kgamma_graph_max_entries(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        None | Some("") => VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
        Some("0") => {
            tracing::warn!(
                "{VERIFY_KGAMMA_GRAPH_MAX_ENTRIES_ENV}=0 clamped to 1; \
                 set ATLAS_QWEN4_VERIFY_GRAPH=0 to disable verify graphs"
            );
            1
        }
        Some(value) => match value.parse::<usize>() {
            Ok(n) => n.max(1),
            Err(_) => {
                tracing::warn!(
                    "{VERIFY_KGAMMA_GRAPH_MAX_ENTRIES_ENV}={value:?} is not a valid \
                     entry count; using default {VERIFY_KGAMMA_GRAPH_MAX_ENTRIES}"
                );
                VERIFY_KGAMMA_GRAPH_MAX_ENTRIES
            }
        },
    }
}

fn verify_kgamma_graph_max_entries() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        parse_verify_kgamma_graph_max_entries(
            std::env::var(VERIFY_KGAMMA_GRAPH_MAX_ENTRIES_ENV)
                .ok()
                .as_deref(),
        )
    })
}

#[inline]
#[must_use]
fn effective_verify_kgamma_graph_cap(
    base_cap: usize,
    mem_available_bytes: Option<usize>,
    floor_bytes: Option<usize>,
) -> usize {
    let base = base_cap.max(1);
    let (Some(avail), Some(floor)) = (mem_available_bytes, floor_bytes) else {
        return base;
    };
    let headroom = avail.saturating_sub(floor);
    if headroom < VERIFY_KGAMMA_GRAPH_EMERGENCY_MARGIN_BYTES {
        base.min(VERIFY_KGAMMA_GRAPH_EMERGENCY_CAP)
    } else if headroom < VERIFY_KGAMMA_GRAPH_CRITICAL_MARGIN_BYTES {
        base.min(VERIFY_KGAMMA_GRAPH_CRITICAL_CAP)
    } else if headroom < VERIFY_KGAMMA_GRAPH_PRESSURE_MARGIN_BYTES {
        base.min(VERIFY_KGAMMA_GRAPH_PRESSURE_CAP)
    } else {
        base
    }
}

#[inline]
#[must_use]
fn verify_kgamma_retain_headroom_ok_with(
    mem_available_bytes: Option<usize>,
    floor_bytes: Option<usize>,
) -> bool {
    match (mem_available_bytes, floor_bytes) {
        (Some(avail), Some(floor)) => {
            avail.saturating_sub(floor) >= VERIFY_KGAMMA_GRAPH_CRITICAL_MARGIN_BYTES
        }
        _ => true,
    }
}

fn verify_kgamma_mem_floor_bytes() -> Option<usize> {
    static FLOOR_BYTES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FLOOR_BYTES.get_or_init(|| {
        spark_runtime::startup_ledger::effective_mem_floor_bytes()
            .ok()
            .filter(|&bytes| bytes > 0)
    })
}

/// True when host `MemAvailable` has at least `1 GiB` headroom above the active
/// UMA floor (matching `MEM_FLOOR_RECOVERED` hysteresis, or telemetry is
/// unavailable), allowing `free_sequence` to retain slot-keyed verify graphs
/// across short requests without forcing per-request CUDA graph recaptures.
pub(super) fn verify_kgamma_retain_headroom_ok() -> bool {
    let Some(floor) = verify_kgamma_mem_floor_bytes() else {
        return true;
    };
    let avail = spark_runtime::startup_ledger::mem_available_bytes();
    verify_kgamma_retain_headroom_ok_with(avail, Some(floor))
}

#[derive(Debug, Default, PartialEq, Eq)]
struct VerifyKgammaEvictionPlan {
    qsa_regime_keys: Vec<VerifyKgammaGraphKey>,
    lru_keys: Vec<VerifyKgammaGraphKey>,
}

impl VerifyKgammaEvictionPlan {
    fn is_empty(&self) -> bool {
        self.qsa_regime_keys.is_empty() && self.lru_keys.is_empty()
    }

    fn all_keys(&self) -> impl Iterator<Item = VerifyKgammaGraphKey> + '_ {
        self.qsa_regime_keys
            .iter()
            .chain(self.lru_keys.iter())
            .copied()
    }
}

/// Plan evictions from `verify_kgamma_graph` before capturing `new_key`:
/// 1. Evict every cached entry on `new_key.0` (both suffix and Layer-0 prefix)
///    whose QSA regime bits (`VERIFY_KGAMMA_QSA_REGIME_MASK`) differ from
///    `new_key.2`, freeing dead pre-2048 / post-2048 / non-QSA regime graphs.
/// 2. Enforce `max_entries_per_tier` on `new_key`'s tier (leaving room for
///    `new_key` itself) and trim the sibling tier if the cap shrank under UMA
///    pressure, evicting other slots first in oldest-LRU-tick order and never
///    evicting `preserve_other_tier_key`.
fn plan_verify_kgamma_evictions(
    cache: &HashMap<VerifyKgammaGraphKey, GraphHandle>,
    lru: &VerifyKgammaGraphLru,
    new_key: VerifyKgammaGraphKey,
    preserve_other_tier_key: Option<VerifyKgammaGraphKey>,
    max_entries_per_tier: usize,
) -> VerifyKgammaEvictionPlan {
    let cap = max_entries_per_tier.max(1);
    let new_slot = new_key.0;
    let new_qsa_bits = verify_kgamma_qsa_regime_bits(new_key.2);
    let new_is_layer0 = verify_kgamma_is_layer0_prefix_key(&new_key);

    let mut qsa_regime_keys = Vec::new();
    let mut same_tier_candidates = Vec::new();
    let mut other_tier_candidates = Vec::new();

    for &key in cache.keys() {
        if key == new_key {
            continue;
        }
        if key.0 == new_slot && verify_kgamma_qsa_regime_bits(key.2) != new_qsa_bits {
            qsa_regime_keys.push(key);
            continue;
        }
        if verify_kgamma_is_layer0_prefix_key(&key) == new_is_layer0 {
            same_tier_candidates.push(key);
        } else if preserve_other_tier_key != Some(key) {
            other_tier_candidates.push(key);
        }
    }
    qsa_regime_keys.sort_unstable_by_key(|key| (key.0, key.1, key.2));

    let mut lru_keys = Vec::new();
    let same_tier_limit = cap.saturating_sub(1);
    if same_tier_candidates.len() > same_tier_limit {
        same_tier_candidates.sort_unstable_by_key(|key| {
            (
                key.0 == new_slot,
                lru.last_used_tick(key),
                key.0,
                key.1,
                key.2,
            )
        });
        let drop_count = same_tier_candidates.len() - same_tier_limit;
        lru_keys.extend(same_tier_candidates.into_iter().take(drop_count));
    }

    let other_tier_preserved = usize::from(preserve_other_tier_key.is_some_and(|pk| {
        pk != new_key
            && cache.contains_key(&pk)
            && verify_kgamma_is_layer0_prefix_key(&pk) != new_is_layer0
            && !(pk.0 == new_slot && verify_kgamma_qsa_regime_bits(pk.2) != new_qsa_bits)
    }));
    let other_tier_total = other_tier_candidates.len() + other_tier_preserved;
    if other_tier_total > cap {
        other_tier_candidates.sort_unstable_by_key(|key| {
            (
                key.0 == new_slot,
                lru.last_used_tick(key),
                key.0,
                key.1,
                key.2,
            )
        });
        let drop_count = (other_tier_total - cap).min(other_tier_candidates.len());
        lru_keys.extend(other_tier_candidates.into_iter().take(drop_count));
    }

    VerifyKgammaEvictionPlan {
        qsa_regime_keys,
        lru_keys,
    }
}

type VerifyDLayer0WarmKey = (
    usize,
    u32,
    Option<crate::layers::qwen3_attention::Qwen4K5DeviceAttentionTopology>,
);

const QWEN4_C1_LAYER0_CAPTURE_POISON_LIMIT: u32 = 8;
const QWEN4_C1_LAYER0_WARM_KEYS_CAP: usize = 256;

#[derive(Default)]
struct VerifyDLayer0State {
    warmed_hits: HashMap<VerifyDLayer0WarmKey, u32>,
    consecutive_capture_failures: u32,
    capture_poisoned: bool,
}

impl VerifyDLayer0State {
    fn is_warm(&self, key: &VerifyDLayer0WarmKey, required_hits: u32) -> bool {
        self.warmed_hits.get(key).copied().unwrap_or(0) >= required_hits
    }

    fn record_warmup_hit(&mut self, key: VerifyDLayer0WarmKey) {
        if !self.warmed_hits.contains_key(&key)
            && self.warmed_hits.len() >= QWEN4_C1_LAYER0_WARM_KEYS_CAP
        {
            self.warmed_hits.clear();
        }
        let hits = self.warmed_hits.entry(key).or_insert(0);
        *hits = hits.saturating_add(1);
    }

    fn record_captured(&mut self, key: VerifyDLayer0WarmKey, required_hits: u32) {
        let hits = self.warmed_hits.entry(key).or_insert(0);
        *hits = (*hits).max(required_hits);
        self.consecutive_capture_failures = 0;
    }

    fn record_capture_failure(&mut self) {
        self.consecutive_capture_failures = self.consecutive_capture_failures.saturating_add(1);
        if self.consecutive_capture_failures >= QWEN4_C1_LAYER0_CAPTURE_POISON_LIMIT {
            self.capture_poisoned = true;
        }
    }
}

static VERIFY_D_LAYER0_TRACKER: std::sync::OnceLock<
    Mutex<HashMap<(usize, u64), VerifyDLayer0State>>,
> = std::sync::OnceLock::new();

fn parse_exact_bool_env_value(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be exactly 0 or 1, got {other:?}"),
    }
}

fn exact_bool_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_exact_bool_env_value(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_bool_env_value(name, None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

/// Select the Qwen4 target-layer implementation for a physical verify width.
///
/// K=2 and K=3 have end-to-end parity evidence. K=5 may use its separately
/// qualified hybrid only behind the existing explicit opt-in. Exact native
/// DFlash K16 may enter the unqualified batched path only behind its separate
/// research flag. Every other width fails closed to ordinary single-token
/// decode until that exact shape has stage and committed-state parity proof.
fn qwen4_verify_layer_route(
    identity: Qwen4VerifyRouteIdentity,
    frame_k: usize,
    k5_hybrid_enabled: bool,
    k4_hybrid_enabled: bool,
    all_k_hybrid_enabled: bool,
    k16_batched_requested: bool,
) -> std::result::Result<Option<Qwen4VerifyLayerRoute>, &'static str> {
    if k16_batched_requested {
        if !identity.target_is_qwen4 {
            return Err("ATLAS_QWEN4_K16_BATCHED_VERIFY requires a Qwen4 target");
        }
        if !identity.native_dflash_pairing {
            return Err(
                "ATLAS_QWEN4_K16_BATCHED_VERIFY requires the exact native Qwen4 DFlash pairing",
            );
        }
        if !identity.active_proposer_is_dflash {
            return Err(
                "ATLAS_QWEN4_K16_BATCHED_VERIFY requires the native DFlash proposer to be active",
            );
        }
        if identity.active_proposer_physical_k != Some(16) {
            return Err("ATLAS_QWEN4_K16_BATCHED_VERIFY requires DFlash physical verify K=16");
        }
    }
    if !identity.target_is_qwen4 {
        return Ok(None);
    }
    let route = match frame_k {
        2 | 3 => Qwen4VerifyLayerRoute::QualifiedBatched,
        5 if k5_hybrid_enabled => Qwen4VerifyLayerRoute::QualifiedBatched,
        4 if k5_hybrid_enabled && k4_hybrid_enabled => Qwen4VerifyLayerRoute::QualifiedBatched,
        // ATLAS_QWEN4_HYBRID_ALL_K: a confidence-gated chain verifies at any
        // width up to num_drafts + 1; 4 and 6..=9 join the exact hybrid rows.
        4 | 6..=9 if k5_hybrid_enabled && all_k_hybrid_enabled => {
            Qwen4VerifyLayerRoute::QualifiedBatched
        }
        16 if k16_batched_requested => Qwen4VerifyLayerRoute::ExperimentalK16Batched,
        _ => Qwen4VerifyLayerRoute::RowSerialOracle,
    };
    Ok(Some(route))
}

fn qwen4_legacy_k2_admission(
    target_is_qwen4: bool,
    rows: usize,
    legacy_selected: bool,
) -> std::result::Result<(), &'static str> {
    if target_is_qwen4 && rows == 2 && legacy_selected {
        return Err("ATLAS_QWEN4_VERIFY_LEGACY_K2 cannot re-enter the Qwen4 fixed-K2 wrapper");
    }
    Ok(())
}

fn is_native_qwen4_dflash_pairing(
    config: &ModelConfig,
    capture_width: usize,
    capture_offset: usize,
    capture_layers: &[usize],
    capture_mode: atlas_core::config::DflashCaptureMode,
) -> bool {
    if !config.is_qwen4_exp() || capture_offset != 0 || !capture_mode.is_qwen4_native() {
        return false;
    }
    atlas_kernels::drafter_profiles_for("qwen3.8-flash-next").any(|profile| {
        let mode_matches = matches!(
            (profile.capture_mode, capture_mode),
            (
                atlas_kernels::DrafterCaptureMode::Qwen4HyperProjected,
                atlas_core::config::DflashCaptureMode::Qwen4HyperProjected,
            ) | (
                atlas_kernels::DrafterCaptureMode::Qwen4LayerAttnHc,
                atlas_core::config::DflashCaptureMode::Qwen4LayerAttnHc,
            ) | (
                atlas_kernels::DrafterCaptureMode::Qwen4HcMean,
                atlas_core::config::DflashCaptureMode::Qwen4HcMean,
            )
        );
        mode_matches
            && profile.drafter.target_layer_ids == Some(capture_layers)
            && profile.target.hidden_size == Some(config.hidden_size)
            && profile.target.hidden_size == Some(capture_width)
            && profile.target.residual_width == Some(config.residual_width())
            && profile.target.num_layers == Some(config.num_hidden_layers)
            && profile
                .target
                .vocab_sizes
                .is_some_and(|vocabs| vocabs.contains(&config.vocab_size))
    })
}

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        super::super::final_norm::check_partial_head_admission(
            &self.config,
            super::super::final_norm::PartialHead::LayerResolutionProbe,
            layer_resolution_probe_enabled(),
        )?;
        let staged_gumbel_reqs = self.staged_verify_gumbel_rows.lock().take();
        *self.in_graph_gumbel_verify_tokens.lock() = None;
        let staged_rejection_reqs = self.staged_verify_rejection_rows.lock().take();
        *self.in_graph_rejection_verify_words.lock() = None;
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }

        // ATLAS_QWEN4_K5_CMULTI: serial K5 at C2-C8 reuses this C1
        // transaction once per request. A flat native-MTP chain never sets
        // the model-wide tree state, and no frame may start while another
        // request still owns the per-layer QSA journals.
        let serial_cmulti_owner = (self.config.is_qwen4_exp()
            && super::super::qwen4_k5_cmulti::k5_cmulti_enforced())
        .then_some(super::super::qwen4_k5_cmulti::QsaJournalOwner {
            slot: seq.slot_idx,
            session: seq.session_hash,
        });
        if let Some(owner) = serial_cmulti_owner {
            super::super::qwen4_k5_cmulti::check_tree_singletons_unset(
                super::super::qwen4_k5_cmulti::TreeSingletons {
                    tree_tokens: *self.ddtree_num_tree_tokens.lock(),
                    host_parents: !self.ddtree_parent_ids_host.lock().is_empty(),
                    device_parents: self.ddtree_parent_ids_dev.lock().is_some(),
                    flat_tree_route: self
                        .dflash_flat_tree_route
                        .load(std::sync::atomic::Ordering::Acquire),
                },
            )
            .map_err(anyhow::Error::msg)?;
            self.qwen4_qsa_journal_owner
                .lock()
                .admit_frame(owner)
                .map_err(anyhow::Error::msg)?;
        }

        // Prefix rollback cannot undo an unjournaled width or a tree frame.
        // Refuse before conversion, snapshots, embeddings or cache writes.
        if qwen4_qsa_verify_frame_required(
            self.config.is_qwen4_exp(),
            seq.qwen4_qsa_required,
            k, // physical verify width
        ) {
            let tree_tokens = *self.ddtree_num_tree_tokens.lock();
            let host_parents_present = !self.ddtree_parent_ids_host.lock().is_empty();
            let device_parents_present = self.ddtree_parent_ids_dev.lock().is_some();
            qwen4_qsa_verify_frame_admission(
                crate::layers::qwen4_qsa::verify_journal_rows(k),
                tree_tokens != 0 || host_parents_present || device_parents_present,
            )
            .map_err(anyhow::Error::msg)?;
        }
        // A frame inside that scope journals its QSA writes; it alone may
        // consume them.
        if let Some(owner) = serial_cmulti_owner
            && qwen4_qsa_verify_frame_required(
                self.config.is_qwen4_exp(),
                seq.qwen4_qsa_required,
                k,
            )
        {
            self.qwen4_qsa_journal_owner
                .lock()
                .arm(owner)
                .map_err(anyhow::Error::msg)?;
        }

        // Physical storage extent is a conservative bound for every tree depth.
        // Reject replay/overflow before snapshots, embeddings or cache mutation.
        seq.rotary_positions.tail_scalar(seq.seq_len)?;
        let last = seq
            .seq_len
            .checked_add(k - 1)
            .ok_or_else(|| anyhow::anyhow!("verify physical extent overflow"))?;
        seq.rotary_positions.tail_scalar(last)?;

        // Freeze the experimental route before the first frame effect. The
        // runtime-only capture mode is set only after exact native Qwen4
        // target/drafter pairing validation; repeat the target geometry here
        // so a forged or partially-constructed config cannot enter the path.
        let k16_batched_requested = exact_bool_env(QWEN4_K16_BATCHED_VERIFY_ENV)?;
        let active_proposer = self.active_proposer();
        let active_proposer_is_dflash = active_proposer.is_some_and(|p| p.is_dflash());
        let active_proposer_physical_k = active_proposer.and_then(|p| {
            if p.is_dflash() {
                p.physical_verify_k()
            } else {
                None
            }
        });
        let native_dflash_pairing = is_native_qwen4_dflash_pairing(
            &self.config,
            self.dflash_capture_width,
            self.dflash_capture_offset,
            self.dflash_capture_layers.as_slice(),
            self.dflash_capture_mode,
        );
        let k5_hybrid_enabled = k5_hybrid_selected();
        let qwen4_layer_route = qwen4_verify_layer_route(
            Qwen4VerifyRouteIdentity {
                target_is_qwen4: self.config.is_qwen4_exp(),
                native_dflash_pairing,
                active_proposer_is_dflash,
                active_proposer_physical_k,
            },
            k,
            k5_hybrid_enabled,
            std::env::var("ATLAS_QWEN4_K4_HYBRID").ok().as_deref() == Some("1"),
            crate::layers::qwen4_hybrid_all_k(),
            k16_batched_requested,
        )
        .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            !(self.config.is_qwen4_exp()
                && seq.qwen4_qsa_required
                && k == 5
                && qwen4_layer_route != Some(Qwen4VerifyLayerRoute::QualifiedBatched)),
            "Qwen4 QSA K5 requires the qualified batched transaction route"
        );
        if qwen4_layer_route == Some(Qwen4VerifyLayerRoute::ExperimentalK16Batched) {
            crate::model::k16_route_receipt::mark_batched_entry(seq.seq_len, tokens)?;
            static K16_BATCHED_ENGAGED: std::sync::Once = std::sync::Once::new();
            K16_BATCHED_ENGAGED.call_once(|| {
                tracing::warn!(
                    "ENGAGED ATLAS_QWEN4_K16_BATCHED_VERIFY: native_qwen4_dflash_physical_k16"
                );
            });
        }
        if qwen4_layer_route == Some(Qwen4VerifyLayerRoute::QualifiedBatched) && k == 5 {
            static MTP_BATCH_VERIFY_LOGGED: std::sync::Once = std::sync::Once::new();
            MTP_BATCH_VERIFY_LOGGED.call_once(|| {
                tracing::info!(
                    "verify_d: PLE_VERIFY_BATCH rows=5 HC_DUAL_ROWS_ENGAGED t8 K-row down launched"
                );
            });
        }
        // The fixed-K2 Qwen4 wrapper delegates here. Refuse the obsolete
        // back-edge instead of recursively re-entering that wrapper.
        qwen4_legacy_k2_admission(
            self.config.is_qwen4_exp(),
            k,
            std::env::var("ATLAS_QWEN4_VERIFY_LEGACY_K2")
                .ok()
                .as_deref()
                == Some("1"),
        )
        .map_err(anyhow::Error::msg)?;
        // ATLAS_SSM_H_FP16 stage 2. This entry point does NOT exist upstream —
        // the speculative verify is ours — and it is the one that matters most
        // here, because the WY kernels it dispatches are the h-state readers and
        // writers whenever --speculative is on. Without this hook the FP16 twins
        // selected in `wy_chunk_kernel` would run over an unconverted FP32 pool.
        // Outside the graph capture that begins further down.
        self.ssm_h_to_f16_dispatch(seq)?;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let persistent_width = self.config.residual_width();
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let serial = crate::model::env_diag::DflashSerialControls::current();
        let serial_family = serial.active_family()?;
        let control_requests = crate::model::control_engagement::ControlRequests {
            attn_paged: crate::layers::attn_paged_serial_enabled(),
            attn_out: crate::layers::attn_out_serial_enabled(),
            ffn: serial.ffn,
            layer_norms: serial.layer_norms,
        };
        let controlled_verify = k > 1 && control_requests.any();
        let stage_capture = crate::model::k1_stage_diag::requested_at(seq.seq_len, tokens)?;
        crate::model::k1_stage_diag::validate_serial_control_overlap(
            crate::model::k1_stage_diag::enabled(),
            serial_family,
        )?;
        if stage_capture {
            if self.use_fp32_logits || self.verify_lmhead_vocab() as usize != self.config.vocab_size
            {
                bail!("DFLASH_K1_STAGE_DIAG requires BF16 full-vocabulary logits");
            }
            crate::model::k1_stage_diag::begin_batch(seq.seq_len, tokens)?;
        }
        if let Some(family) = serial_family {
            static PROOF: std::sync::Once = std::sync::Once::new();
            PROOF.call_once(|| {
                tracing::warn!(family, c = 1, k, "DFLASH_K1_BISECT C1 active");
            });
        }
        let kgamma_debug_dump = std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_ok()
            && std::env::var("ATLAS_KGAMMA_DEBUG_SEQ_LEN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .is_none_or(|wanted| wanted == seq.seq_len);

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        // ATLAS_FULL_PROFILE=1: bump per-step counter so per-kernel kprof!
        // accumulators fire on K=γ verify. begin_step() early-returns when
        // env var is unset — see full_profile.rs:103-111 for the safety
        // contract (must NOT flip ACTIVE while CUDA graph capture is live).
        crate::full_profile::begin_step();

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──
        //
        // DFS reorder (option C, ATLAS_DDTREE_DFS_REORDER=1):
        // Permute the K kernel-frame tokens into DFS pre-order so each
        // ancestor chain is contiguous in slot order. The paged-decode
        // attention kernel iterates [0..seq_lens[t]) sequentially — DFS
        // reorder ensures that the deepest ancestor chain (the one
        // greedy_sample_ddtree will walk) reads its TRUE ancestors,
        // not sibling/cousin slots. Sibling root children at the end of
        // the DFS order get short seq_lens and only read the bonus +
        // already-visited subtree — wrong but acceptable because the
        // greedy walker only ever commits one chain.
        //
        // `dfs_perm[i] = j` means "DFS slot i contains kernel slot j".
        // `dfs_inv_perm[j] = i` means "kernel slot j is at DFS slot i".
        // Empty when DFS reorder is disabled or no tree payload is active.
        let dfs_enabled = std::env::var("ATLAS_DDTREE_DFS_REORDER").ok().as_deref() == Some("1");
        let (dfs_perm, dfs_inv_perm, dfs_depths): (Vec<usize>, Vec<usize>, Vec<usize>) = {
            let host_parents_lock = self.ddtree_parent_ids_host.lock();
            // Allow payload shorter than k (we pad with linear-chain tail to
            // match k below). Only require non-empty payload.
            if dfs_enabled && !host_parents_lock.is_empty() && host_parents_lock.len() <= k {
                use crate::layers::dflash_head::ddtree::dfs_reorder;
                // Pad up to k if payload shorter (mirror the linear-chain
                // padding logic from the existing depth derivation).
                let mut hp = host_parents_lock.clone();
                drop(host_parents_lock);
                while hp.len() < k {
                    let next_chain_parent = hp.len() as i32 - 1;
                    hp.push(next_chain_parent);
                }
                let (perm, inv, depths) = dfs_reorder(&hp);
                // If perm is identity (flat chain or trivially-flat payload),
                // skip the reorder machinery — it's a no-op and the existing
                // chain path is already optimal.
                let is_identity = perm.iter().enumerate().all(|(i, &j)| i == j);
                if is_identity {
                    (Vec::new(), Vec::new(), Vec::new())
                } else {
                    static DFS_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !DFS_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!(
                            "ATLAS_DDTREE_DFS_REORDER: k={k} perm={:?} depths={:?}",
                            perm,
                            depths
                        );
                    }
                    (perm, inv, depths)
                }
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            }
        };
        let dfs_active = !dfs_perm.is_empty();

        // ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY=1: skip the token permutation
        // (embedding stays in linear-drafts order) and only permute attention
        // metadata + parent_ids. Use case: when the verify input `tokens`
        // is the LINEAR top-1 drafts (current Atlas plumbing), not the
        // tree's tree_token_ids. Permuting linear drafts into DFS slots
        // places semantically wrong tokens (e.g. drafts[5] at the chain-child
        // slot of drafts[0]) and the model output becomes incoherent.
        //
        // With METADATA_ONLY=1 we keep token-at-slot consistency for
        // attention (preserves the existing chain accept behavior) while
        // letting depth-based RoPE positions still fire. This is a
        // research-only mode used to A/B the metadata permutation alone.
        let dfs_permute_tokens = std::env::var("ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY")
            .ok()
            .as_deref()
            != Some("1");

        // Defer Qwen4 K-row embedding when Layer-0 prefix graph capture may
        // engage so `embed -> Layer 0` runs inside ONE CUDA graph (falling back
        // to the exact eager `self.embed` loop below whenever Layer-0 prefix
        // graphing is inactive). Nothing between here and Layer 0 reads
        // `hidden`.
        let defer_qwen4_embed = self.config.is_qwen4_exp()
            && !stage_capture
            && !crate::model::k1_stage_diag::enabled()
            && !dfs_active
            && !self.config.use_fp32_residual();
        if !defer_qwen4_embed && self.config.pp_rank == 0 {
            // 1a. Embed K tokens — in DFS order if reorder active AND
            // METADATA_ONLY=0 (default). Use single batched gather launch when
            // available (non-Qwen4 BF16 residual) to eliminate K-1 host copies.
            crate::kprof!(self.gpu.as_ref(), stream, "embed", {
                let can_batched_embed = k > 1
                    && !self.config.is_qwen4_exp()
                    && self.bf16_to_f32_kernel.0 == 0
                    && persistent_width == h
                    && self.batched_embed_kernel.0 != 0
                    && k.saturating_mul(4) <= self.buffers.sizes().norm_output
                    && (!dfs_active || !dfs_permute_tokens || k <= 64);
                if can_batched_embed {
                    let mut dfs_tokens = [0u32; 64];
                    let embed_ids: &[u32] = if dfs_active && dfs_permute_tokens {
                        for t in 0..k {
                            dfs_tokens[t] = tokens[dfs_perm[t]];
                        }
                        &dfs_tokens[..k]
                    } else {
                        &tokens[..k]
                    };
                    // SAFETY: `embed_ids` is a valid initialized `&[u32]` slice
                    // of `k * 4` bytes; `copy_h2d_group_fast` completes reading
                    // it before returning.
                    let token_ids_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            embed_ids.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(embed_ids),
                        )
                    };
                    // `norm_output` is overwritten by layer 0's input RMS norm
                    // and verified above to hold at least `k * 4` bytes.
                    let token_ids_dev = self.buffers.norm_output();
                    self.copy_h2d_group_fast(
                        &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                        stream,
                    )?;
                    ops::batched_embed(
                        self.gpu.as_ref(),
                        self.batched_embed_kernel,
                        token_ids_dev,
                        self.embed_tokens.weight,
                        hidden,
                        k as u32,
                        h as u32,
                        stream,
                    )?;
                    self.scale_embeddings(hidden, k, stream)?;
                } else {
                    for t in 0..k {
                        let src_token = if dfs_active && dfs_permute_tokens {
                            tokens[dfs_perm[t]]
                        } else {
                            tokens[t]
                        };
                        self.embed(
                            src_token,
                            hidden.offset(t * persistent_width * fp32),
                            stream,
                        )?;
                    }
                }
                anyhow::Result::<()>::Ok(())
            })?;
            self.capture_k1_stage("embed", hidden, k, persistent_width * fp32, stream)?;
        }

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 1c. Upload K-entry attention metadata. Layout in scratch at
        // STEP_META_SCRATCH_OFFSET (the same base the MTP draft step and chain
        // use; there is no separate MTP reservation, so this frame re-uploads
        // everything below before any launch reads it): positions[K*4] |
        // slots[K*8] | seq_lens[K*4] | block_table[K*max_blocks*4]. Need
        // K*16 + K*max_blocks*4 bytes per call — ~36 KB at K=17 max_blocks=512.
        let meta_base = self
            .buffers
            .scratch()
            .offset(crate::layer::STEP_META_SCRATCH_OFFSET);
        let max_blocks = self.max_blocks_per_seq;

        // Tree-aware metadata (ATLAS_DDTREE_TREE_AWARE_VERIFY=1):
        //
        // The chain-mode meta below treats the γ+1 verify tokens as a FLAT
        // chain (positions = seq.seq_len + t, seq_lens = seq.seq_len + t + 1).
        // That is wrong for the real top-K tree topology produced by M4B v2:
        //
        //   - Sibling root children (e.g. compact 0..4 with parent=-1) get
        //     DIFFERENT RoPE positions (1, 2, 3, 4, 5) when they are all at
        //     tree depth 1 and should share position 1. RoPE mis-alignment
        //     skews query directions per sibling and degrades the per-position
        //     argmax accuracy.
        //   - The per-row seq_lens upper bound also grows with compact index,
        //     so a sibling at compact index 4 reads positions [seq.seq_len ..
        //     seq.seq_len + 4] = bonus + 4 sibling KVs (not its ancestor chain).
        //
        // The full fix (per-row KV indirection so each query attends only to
        // its ancestors) requires modifying the paged-decode attention kernel
        // — see paged_decode_attn_nvfp4.cu / paged_decode_attn_fp8.cu, which
        // today iterate positions [0..seq_lens[t]) and read in-block offset
        // `pos % bs` with no slot-indirection knob. Until that lands,
        // ATLAS_DDTREE_TREE_AWARE_VERIFY=1 implements a *partial* fix that
        // produces coherent output without changing tok/s vs flat chain:
        //
        //   - positions[t] = seq.seq_len + depth[t]   (depth-based RoPE so
        //     sibling root children share RoPE position 1 instead of 1..N).
        //   - seq_lens[t]  unchanged (still compact-index-based). Setting it
        //     to depth-based caused deep chain queries to skip legitimate
        //     prior chain context and drop accept rate even further.
        //
        // Slots are still written at compact-index physical positions so the
        // KV cache stays consistent for the post-verify commit path. Block-
        // tables are unchanged (the same full sequence row replicated per t).
        //
        // RESULT: B (tree-aware) text == A (flat chain) text on essay prompt
        //         but tok/s does NOT clear A — needs the kernel-level KV
        //         indirection to land for a real win on non-flat topologies.
        //
        // depth[t] is derived from the kernel-frame parent_ids stashed by
        // set_ddtree_parent_ids — index 0 is the bonus (depth 0), index i+1
        // is draft i. parent[i] = -1 means "child of pre-tree state" (= bonus,
        // depth 1); parent[i] = k means depth[t] = 1 + depth[k].
        let tree_aware_enabled = std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY")
            .ok()
            .as_deref()
            == Some("1");
        let tree_depths: Option<Vec<usize>> = if tree_aware_enabled {
            // host_parents may be shorter than k when the tree payload has
            // fewer nodes than γ_eff (e.g. budget < γ_eff caps the tree at
            // budget). The persistent device buffer is padded by
            // clear_ddtree_parent_ids' linear-chain default — replicate the
            // same padding here so depth lookups don't OOB.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            // DDTree budgets MAX_NODES (e.g. 32) but SSM verify window is k
            // (e.g. 17). When len > k the == k guard below was falling through
            // to None, keeping tree_depths always None and preventing the
            // causal_conv1d_tree_reroot kernel from ever firing. Truncate first.
            if host_parents.len() > k {
                host_parents.truncate(k);
            }
            if !host_parents.is_empty() && host_parents.len() < k {
                let last_payload_idx = host_parents.len() as i32 - 1;
                while host_parents.len() < k {
                    let next_chain_parent = host_parents.len() as i32 - 1;
                    let _ = last_payload_idx;
                    host_parents.push(next_chain_parent);
                }
            }
            if host_parents.len() == k && !host_parents.is_empty() {
                // Build kernel-frame depths. Bonus (index 0) is depth 0.
                let mut depths = vec![0usize; k];
                // For index i >= 1: parent < 0 means "attaches to pre-tree
                // state" — treat as a child of the bonus (depth 1). Otherwise
                // parent kernel slot p (0 = bonus) means depth = 1 + depth[p].
                for i in 1..k {
                    let p = host_parents[i];
                    if p < 0 {
                        depths[i] = 1;
                    } else {
                        let pi = p as usize;
                        if pi >= i {
                            // Defensive: malformed payload → fall back to
                            // chain depth so we don't panic. Set to compact
                            // index so downstream stays self-consistent.
                            depths[i] = i;
                        } else {
                            depths[i] = depths[pi].saturating_add(1);
                        }
                    }
                }
                // Refuse to apply when the payload is degenerate (flat chain)
                // — chain mode is already optimal.
                let is_flat_chain = (1..k).all(|i| depths[i] == i);
                if is_flat_chain {
                    None
                } else {
                    static TREE_AWARE_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !TREE_AWARE_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!("ATLAS_DDTREE_TREE_AWARE_VERIFY: k={k} depths={:?}", depths);
                    }
                    Some(depths)
                }
            } else {
                None
            }
        } else {
            None
        };

        // ATLAS_TREE_AWARE_ATTN: full per-row KV indirection. Activated
        // when the env var is set AND tree mode is producing non-flat
        // depths AND DFS reorder is OFF (DFS already provides contiguous
        // ancestor reads, so the two should not be combined).
        //
        // STATUS (2026-05-21): correctness verified end-to-end (coherent
        // outputs, ancestor chains built correctly per log) but the kernel
        // single-position fallback in the tree window is ~3.6x slower than
        // the batched chain-mode path on the qwen3.6-27b config (K=γ=17,
        // depth up to 14, 16 q_heads × 17 rows = 272 CTAs each walking
        // 1-14 single-position iterations). The original chain path uses
        // BC=4 batched loads which dominate even though tree mode reads
        // fewer total KV positions.
        //
        // Default OFF until the kernel single-position path is batched
        // (e.g. by re-introducing BC=4 over indirected ancestors, which
        // requires same-block packing — non-trivial because ancestor slots
        // are scattered across blocks for deep trees).
        let tree_aware_attn_enabled =
            std::env::var("ATLAS_TREE_AWARE_ATTN").ok().as_deref() == Some("1");
        // A selective-boundary cache may mix attention KV dtypes. Do not pay
        // the indirection cost on the supported subset when any attention
        // layer would ignore the map: that verify cannot safely commit a
        // branch anyway. This also keeps the fallback on the established
        // batched chain kernel instead of running a strictly slower partial
        // tree computation whose result must be discarded.
        let all_attention_layers_exact = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_ancestor_attention_exact());
        let tree_kv_active = tree_aware_attn_enabled
            && !dfs_active
            && tree_depths.is_some()
            && all_attention_layers_exact
            && self.tree_kv_indir_stride > 0
            && self.tree_kv_indir_persistent.0 != 0
            && k <= self.tree_kv_indir_stride;

        // ── Ancestor-exact attention flag (2026-07-08 deep-branch root cause) ──
        //
        // Record whether THIS verify gives EVERY tree node ancestor-exact
        // attention (each row attends to exactly the pre-context + its true
        // ancestors + itself):
        //   * flat/chain payload (or none)  → exact: prefix reads ARE the
        //     conditioning (byte-identical baseline path);
        //   * per-row KV indirection active → exact: the kernel remaps the
        //     tree window through each row's true ancestor chain;
        //   * NON-flat payload under prefix metadata (chain-mode or DFS
        //     depth-prefix reads) → NOT exact. DFS pre-order lays only the
        //     LEFTMOST path at slot == depth; any branch node at DFS slot
        //     s > depth reads DFS slots [0..depth] — its spine SIBLING at
        //     slot `depth` instead of its own key at slot `s`
        //     (`ddtree::dfs_prefix_reads_are_ancestor_exact`). Branch rows'
        //     logits are then wrong, and the deep tree-commit walker would
        //     consume them for child-acceptance and the bonus → committed
        //     tokens diverge from the greedy oracle (VALIDATION-36 TEST 1).
        //
        // The scheduler reads this via `dflash_tree_ancestor_attn_exact()`
        // and degrades the FULL tree-commit walker to the flat-safe walker
        // when not exact — turning that silent-corruption class into a safe
        // (spine-only) accept.
        let payload_non_flat = {
            let hp = self.ddtree_parent_ids_host.lock();
            hp.iter()
                .enumerate()
                .skip(1)
                .any(|(i, &p)| p != i as i32 - 1)
        };
        // Conv-exact assertion (ATLAS_DDTREE_TREE_CONV_EXACT=1 + kernel loaded).
        // The causal_conv1d_tree_reroot kernel re-roots each branch token's conv
        // shift-register from its true ancestor's intermediate, making the conv
        // output oracle-correct for ALL tree topologies. Branch commits without
        // this kernel are WRONG (BUG 1 from freeslots-rootcause.md): the conv
        // window propagates compact-predecessor tokens instead of true ancestors,
        // corrupting all 48 GDN layer outputs for branch rows.
        // Gating branch commits on conv_exact prevents silent non-oracle commits.
        // ATLAS_DDTREE_ASSUME_CONV_EXACT=1 is the research escape hatch (repro only).
        let tree_conv_kernel_available = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_conv_state_exact());
        let tree_conv_exact_on = (tree_conv_kernel_available
            && std::env::var("ATLAS_DDTREE_TREE_CONV_EXACT")
                .ok()
                .as_deref()
                == Some("1"))
            || std::env::var("ATLAS_DDTREE_ASSUME_CONV_EXACT")
                .ok()
                .as_deref()
                == Some("1");
        // For a non-flat tree payload, BOTH attention indirection AND conv reroot
        // must be active to guarantee oracle-correct branch logits. A flat payload
        // (or no payload) is always exact by construction.
        // Indirection support is a per-attention-layer property. Selective
        // boundary caches can mix BF16 and FP8 layers; treating a live FP8
        // indirection buffer as a model-wide capability incorrectly certified
        // the BF16 layers even though their decode kernel dropped the tree
        // metadata. Require every layer to attest support and fail closed.
        let ancestor_attn_exact = !payload_non_flat
            || (tree_kv_active && tree_conv_exact_on && all_attention_layers_exact);
        self.dflash_tree_ancestor_attn
            .store(ancestor_attn_exact, std::sync::atomic::Ordering::Release);
        if payload_non_flat && ancestor_attn_exact {
            static EXACT_CERT_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = EXACT_CERT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 {
                use spark_runtime::kv_cache::KvCacheDtype;
                let attention_certificates: Vec<_> = self
                    .layers
                    .iter()
                    .filter_map(|layer| layer.ddtree_attention_certificate())
                    .collect();
                let count = |dtype| {
                    attention_certificates
                        .iter()
                        .filter(|(actual, _)| *actual == dtype)
                        .count()
                };
                let exact = |dtype| {
                    attention_certificates
                        .iter()
                        .filter(|(actual, supported)| *actual == dtype && *supported)
                        .count()
                };
                let bf16 = count(KvCacheDtype::Bf16);
                let fp8 = count(KvCacheDtype::Fp8);
                let other = attention_certificates.len().saturating_sub(bf16 + fp8);
                let conv_total = self
                    .layers
                    .iter()
                    .filter(|layer| layer.is_ssm_layer())
                    .count();
                let conv_exact = self
                    .layers
                    .iter()
                    .filter(|layer| layer.is_ssm_layer() && layer.ddtree_conv_state_exact())
                    .count();
                tracing::info!(
                    "DDTREE_EXACT_CERT #{n}: k={k} attention_layers={} \
                     bf16_tree_handles={}/{} fp8_indirection={}/{} \
                     other_attention={} conv_reroot={}/{}",
                    attention_certificates.len(),
                    exact(KvCacheDtype::Bf16),
                    bf16,
                    exact(KvCacheDtype::Fp8),
                    fp8,
                    other,
                    conv_exact,
                    conv_total,
                );
            }
        }
        if payload_non_flat && !ancestor_attn_exact {
            static NONEXACT_DBG: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let n = NONEXACT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 16 {
                tracing::info!(
                    "K=γ verify: non-flat tree payload — branch commits degraded to flat-safe \
                     (k={k}, dfs_active={dfs_active}, tree_aware_attn={tree_aware_attn_enabled}, \
                     tree_depths_some={}, indir_stride={}, indir_ptr_ok={}, all_attn_exact={}, \
                     tree_conv_kernel_ok={tree_conv_kernel_available}, \
                     tree_conv_exact_on={tree_conv_exact_on}) — \
                     Enable ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1 + \
                     ATLAS_DDTREE_TREE_CONV_EXACT=1 for lossless deep-branch commits.",
                    tree_depths.is_some(),
                    self.tree_kv_indir_stride,
                    self.tree_kv_indir_persistent.0 != 0,
                    all_attention_layers_exact,
                );
            }
        }

        let rotary_depths: Vec<usize> = if dfs_active {
            // DFS slot i contains kernel slot dfs_perm[i]; its tree depth
            // (kernel frame) is dfs_depths[dfs_perm[i]]. RoPE position is
            // seq.seq_len + depth so siblings at the same depth share
            // a position.
            (0..k).map(|t| dfs_depths[dfs_perm[t]]).collect()
        } else if let Some(ref depths) = tree_depths {
            // Kernel slot 0 (bonus) is depth 0 → position seq.seq_len (the
            // last_token's slot). For drafts at depth d, position is
            // seq.seq_len + d.
            (0..k).map(|t| depths[t]).collect()
        } else {
            (0..k).collect()
        };
        let positions = seq
            .rotary_positions
            .verify_tail(seq.seq_len, &rotary_depths)?;
        // SAFETY: `positions` is an initialized `Vec<u32>` of `k` elements
        // (`verify_tail` returns one position per row) that outlives the
        // `copy_h2d_group_fast` call below, which finishes reading it before
        // returning.
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(
                positions.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&positions[..k]),
            )
        };
        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut meta_scratch = VERIFY_D_META_SCRATCH
            .get_or_init(|| Mutex::new(VerifyDMetaHostScratch::default()))
            .lock();
        meta_scratch.prepare(k, mb);
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            meta_scratch
                .slots
                .push((physical_block as i64) * (bs as i64) + (block_offset as i64));
        }
        // When neither DFS reorder nor tree-aware KV indirection
        // (`tree_kv_active`, handled in the branch below) is active,
        // `seq_lens` stays compact-index-based: setting `seq_lens` to depth+1
        // makes a depth-d query read positions `[0..seq.seq_len+d]` which are
        // compact-ordered slots (sibling/cousin slots, not actual ancestors,
        // for any non-flat tree). Without per-row indirection enabled, the
        // chain-based `seq_len` is no worse than depth-based and avoids
        // dropping legitimate chain context inside ancestor-only subtrees.
        //
        // DFS reorder changes this: in DFS pre-order, when we visit a node at
        // depth d, the immediate ancestor chain (root → … → us) occupies the
        // d slots immediately preceding the current one (their stack frames).
        // So `seq_lens[t] = seq.seq_len + depth[dfs_perm[t]] + 1` gives a
        // depth-d query exactly its ancestor chain (the bonus + d ancestors),
        // *contiguously* in slot order. Sibling subtrees that were already
        // visited (and would otherwise contaminate the read) are skipped
        // because seq_len cuts off the read before reaching them.
        if dfs_active {
            meta_scratch
                .seq_lens
                .extend((0..k).map(|t| (seq.seq_len + dfs_depths[dfs_perm[t]] + 1) as i32));
        } else if tree_kv_active {
            // ATLAS_TREE_AWARE_ATTN: kernel only walks the ancestor chain
            // for the tree window, so seq_len = prior_context + depth[t] + 1.
            // The kernel remaps positions ≥ seq.seq_len via the indirection
            // table so they read the right ancestors, not sibling slots.
            let depths = tree_depths.as_ref().unwrap();
            meta_scratch
                .seq_lens
                .extend((0..k).map(|t| (seq.seq_len + depths[t] + 1) as i32));
        } else {
            meta_scratch
                .seq_lens
                .extend((0..k).map(|t| (seq.seq_len + t + 1) as i32));
        }
        meta_scratch.fill_block_table_rows(k, mb, &seq.block_table);
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        // SAFETY: `meta_scratch` stays locked across `copy_h2d_group_fast`, and
        // `meta_scratch.slots` holds `k` initialized `i64` elements (`k * 8` bytes).
        let slot_bytes = unsafe {
            std::slice::from_raw_parts(
                meta_scratch.slots.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&meta_scratch.slots[..k]),
            )
        };
        // SAFETY: `meta_scratch` stays locked across `copy_h2d_group_fast`, and
        // `meta_scratch.seq_lens` holds `k` initialized `i32` elements (`k * 4` bytes).
        let sl_bytes = unsafe {
            std::slice::from_raw_parts(
                meta_scratch.seq_lens.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&meta_scratch.seq_lens[..k]),
            )
        };
        // SAFETY: `meta_scratch` stays locked across `copy_h2d_group_fast`, and
        // `meta_scratch.block_tables` holds at least `needed` initialized `i32` elements.
        let bt_bytes = unsafe {
            std::slice::from_raw_parts(
                meta_scratch.block_tables.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&meta_scratch.block_tables[..needed]),
            )
        };
        let metadata_copies = [
            HostToDeviceCopy::new(pos_bytes, meta_base),
            HostToDeviceCopy::new(slot_bytes, meta_base.offset(256)),
            HostToDeviceCopy::new(sl_bytes, meta_base.offset(512)),
            HostToDeviceCopy::new(bt_bytes, meta_base.offset(768)),
        ];
        self.copy_h2d_group_fast(&metadata_copies, stream)?;
        drop(meta_scratch);

        // ATLAS_TREE_AWARE_ATTN CUDA graph fix: upload `seq.seq_len as i32`
        // into the persistent 1×i32 device buffer that backs the kernel's
        // `kv_indir_base_ptr` / `abs_base_ptr` args. The captured K=γ verify
        // graph reads this via pointer indirection so the fresh value lands
        // on every replay (previously the scalar was baked into the kernel-
        // launch node at capture time and went stale → wrong `local_off`).
        // Upload happens BEFORE `begin_capture` (same pattern as `seq_lens`),
        // so it stays outside the graph.
        // Stamp the per-step base value into the persistent pinned-host
        // shadow then fire the async H2D copy. The kernels read this via
        // `kv_indir_base_ptr` / `abs_base_ptr` so captured CUDA graphs
        // pick up the fresh value on each replay (the prior scalar arg was
        // baked into the kernel-launch node at capture time).
        if tree_kv_active && !self.tree_kv_indir_base_persistent.is_null() {
            let mut pinned_guard = self.tree_kv_indir_base_host_pinned.lock();
            let pinned = pinned_guard
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("tree KV base pinned host allocation is missing"))?;
            // Write the new base into the persistent pinned-host shadow
            // (in-place, host-side, no GPU involved) THEN fire the async
            // H2D copy from the pinned shadow. Pinned-host sources establish
            // a proper stream-ordered dependency for captured graph kernels;
            // pageable Vec sources do not (small pageable copies can race
            // ahead of subsequent graph launches → stale `kv_indir_base`).
            pinned
                .as_mut_slice()
                .copy_from_slice(&(seq.seq_len as i32).to_ne_bytes());
            let base_bytes = pinned.pinned_slice(std::mem::size_of::<i32>())?;
            // SAFETY: `base_bytes` views `size_of::<i32>()` initialized bytes
            // in the persistent pinned-host shadow owned by `self` (serialized
            // by `pinned_guard`), and `self.tree_kv_indir_base_persistent` is a
            // non-null device allocation of at least `size_of::<i32>()` bytes.
            unsafe {
                self.gpu.copy_h2d_pinned_async(
                    base_bytes,
                    self.tree_kv_indir_base_persistent,
                    stream,
                )?;
            }
        }

        // ATLAS_TREE_AWARE_ATTN: build + upload per-row KV indirection table.
        // Row t = ancestor chain of compact slot t in top-down (bonus → t)
        // order, indexed by depth in [0..depth[t]+1). Kernel reads
        // indirection[seq_idx][i] for i in [0..seq_lens[t]-kv_indir_base).
        if tree_kv_active {
            let stride = self.tree_kv_indir_stride;
            let depths_ref = tree_depths.as_ref().unwrap();
            let host_parents = self.ddtree_parent_ids_host.lock().clone();
            let mut indir = vec![0i32; stride * stride];
            for t in 0..k {
                // True ancestor chain [root(0), …, t] — unit-tested against
                // the free-slots branch shapes in ddtree.rs
                // (`ancestor_chain_matches_true_ancestors_for_free_slots_branch`).
                let mut chain: Vec<usize> =
                    crate::layers::dflash_head::ddtree::ancestor_chain_topdown(&host_parents, t, k);
                let want = depths_ref[t] + 1;
                while chain.len() < want {
                    chain.push(*chain.last().unwrap_or(&t));
                }
                for (j, &anc) in chain.iter().enumerate().take(stride) {
                    indir[t * stride + j] = anc as i32;
                }
                let last_anc = *chain.last().unwrap_or(&t) as i32;
                for j in chain.len()..stride {
                    indir[t * stride + j] = last_anc;
                }
            }
            // SAFETY: `indir` is an initialized `Vec<i32>` of `stride * stride`
            // POD elements, so viewing it as `size_of_val(indir.as_slice())`
            // bytes is valid for `u8`; `copy_h2d_group_fast` finishes reading
            // `indir_bytes_view` before `indir` is dropped.
            let indir_bytes_view = unsafe {
                std::slice::from_raw_parts(
                    indir.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(indir.as_slice()),
                )
            };
            self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(
                    indir_bytes_view,
                    self.tree_kv_indir_persistent,
                )],
                stream,
            )?;
            static TREE_KV_DBG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !TREE_KV_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let preview: Vec<Vec<i32>> = (0..k)
                    .map(|t| {
                        let d = (depths_ref[t] + 1).min(stride);
                        indir[t * stride..t * stride + d].to_vec()
                    })
                    .collect();
                tracing::info!(
                    "ATLAS_TREE_AWARE_ATTN: k={k} indir_stride={stride} chains={:?}",
                    preview
                );
            }
        }

        let metadata = AttnMetadataDev {
            qwen4_qsa_required: seq.qwen4_qsa_required,
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
        };
        // The device-attention K5 route still makes host-selected launch
        // decisions (pool endpoints, score grids, dense-vs-QSA route). Those
        // decisions are CUDA graph topology, not frame data, and therefore
        // must participate in the graph key. Omitting them replays stale
        // launches when seq_len crosses a four-token/group or sparse boundary.
        let qwen4_k5_attention_topology =
            crate::layers::qwen3_attention::Qwen4K5DeviceAttentionPlan::from_host(
                &self.config,
                k,
                seq.seq_len,
                metadata,
            )?
            .map(|plan| plan.topology());

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        let force_eager = std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1")
            || kgamma_debug_dump;
        // Mirror verify_b.rs (K=2) auto-unsuppress logic. Without this the
        // K=γ verify path stays EAGER forever once FP8/turbo KV calibration
        // sets suppress_graphs=true at startup, and never gets recaptured
        // into a CUDA graph. Eager K=γ verify is ~5x slower than graphed
        // (observed: K=9 eager 600ms/step vs K=4 graphed 80ms/step). Once
        // seq_len passes the calibration window, FP8 scales are frozen and
        // graph capture is safe — provided no permanent or scoped suppressor
        // (`ATLAS_DIAG_GEMMA4`, `ATLAS_DUMP_HIDDEN`, or `decode_serial_batch`)
        // owns `suppress_graphs`.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
            && !crate::model::env_diag::diag_gemma4_enabled()
            && std::env::var("ATLAS_DUMP_HIDDEN").is_err()
            && !self.serial_batch_suppress_active()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (K=γ verify)");
        }
        // ATLAS_FULL_PROFILE=1 disables graph capture so per-kernel sync is
        // legal (CUDA graph capture forbids host syncs / D2H copies inside).
        let full_profile = crate::full_profile::is_enabled();
        let qsa_stage_pool_fused = std::env::var("ATLAS_QWEN4_QSA_VERIFY_STAGE_POOL_FUSED")
            .ok()
            .as_deref()
            != Some("0");
        let qsa_dense_prefix_fused = qwen4_layer_route
            == Some(Qwen4VerifyLayerRoute::QualifiedBatched)
            && (crate::layers::qwen4_qsa::graph_safe_enabled()
                || (crate::layers::qwen4_qsa::verify_journal_rows(k) && qsa_stage_pool_fused));
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !full_profile
            && !crate::model::decode_stages::enabled()
            && !layer_resolution_probe_enabled()
            && serial_family.is_none()
            && crate::model::k1_stage_diag::graph_capture_allowed()
            && !controlled_verify
            // Host-selected QSA attention changes launch topology across the
            // 2048 dense->sparse boundary. Admit the pure-dense prefix
            // (`seq_len + k <= 2048` with constant-topology fused QSA staging,
            // keyed separately by `QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT`), the
            // bounded device-attention topology, or the graph-safe QSA route
            // once every row is permanently sparse (`seq_len >= 2048`).
            && qwen4_qsa_graph_route_safe(
                seq.qwen4_qsa_required,
                seq.seq_len,
                k,
                qsa_dense_prefix_fused,
                qwen4_k5_attention_topology.is_some(),
                crate::layers::qwen4_qsa::graph_safe_enabled(),
            )
            // Qwen4 PLE performs host-backed sparse row fetches, so a Qwen4
            // verify graph is segmented: layer 0 and the PLE rows run eagerly
            // before the graph, which covers layers 1.. and the head (the same
            // split as the plain-decode suffix graph). Opt-in until gated.
            && (!self.config.is_qwen4_exp() || qwen4_verify_graph_enabled());
        let qsa_dense_prefix_graph = qwen4_qsa_dense_prefix_graph_eligible(
            seq.qwen4_qsa_required,
            seq.seq_len,
            k,
            qsa_dense_prefix_fused,
        ) && qwen4_k5_attention_topology.is_none();
        let qwen4_graph_prefix = use_graphs && self.config.is_qwen4_exp();
        if use_graphs && super::verify_graph_receipts::note_engaged(seq.slot_idx) {
            tracing::info!(
                "{}",
                super::verify_graph_receipts::engaged_line(
                    seq.session_hash,
                    seq.slot_idx,
                    seq.seq_len,
                    super::verify_graph_receipts::engaged_reason(
                        seq.qwen4_qsa_required,
                        qwen4_k5_attention_topology.is_some(),
                        qsa_dense_prefix_graph,
                    ),
                )
            );
        }

        // M8A: upload DDTree parent_ids (when stashed) so the GDN dispatch
        // can fire the tree-aware kernel. Stash lives on the model so we
        // don't have to thread it through every call site here.
        // verify_dflash_step.rs sets ddtree_parent_ids_dev before calling
        // this verify entry (via model.set_ddtree_parent_ids) so we just
        // read here.
        //
        // CUDA-graph-safe path: when graphs are enabled and the verify width K
        // matches `ddtree_parent_ids_capacity` with no explicit tree payload,
        // substitute the persistent linear-chain parent buffer so the captured
        // graph references a constant kernel and device pointer. Setting
        // `ATLAS_DISABLE_TREE_WY=1` (or engaging lazy sequential GDN) skips
        // this injection and runs the standard flat `wy_k` path directly,
        // avoiding numerical drift between the tree-WY kernel and `wy17` on
        // flat chains.
        let mut ddtree_parent_ids_dev = *self.ddtree_parent_ids_dev.lock();
        let disable_tree_wy = std::env::var("ATLAS_DISABLE_TREE_WY").ok().as_deref() == Some("1")
            || self.layers.iter().any(|l| l.gdn_seq_lazy_engaged(k));
        let flat_injected = super::commit_plan::flat_tree_wy_injection_applies(
            disable_tree_wy,
            use_graphs,
            ddtree_parent_ids_dev.is_some(),
            self.ddtree_parent_ids_capacity,
            k,
        );
        if flat_injected {
            ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
        }
        // The injected flat-chain verify runs `gated_delta_rule_tree_wy`, which
        // writes per-token states into the intermediate pool without updating
        // `h_state` in place. Record the injection on the model so the commit
        // path (`async_chkpt.rs`) routes full accepts through
        // `h_intermediate[K-1]` rather than keeping the pre-verify `h_state`.
        //
        // The flag is only honored when the SSM dispatch would actually take
        // the tree branch (`gdn_tree_k` loaded and `ATLAS_FORCE_WY17` unset —
        // mirrors `trait_decode_batched_conv_gdn.rs`); otherwise the flat
        // wy17/chunked kernels run, which DO write `h_state`.
        // `any` assumes GDN kernel homogeneity: every SSM layer is built from
        // the same PTX bundle, so either all have the tree kernel or none do
        // (non-SSM layers return false). A mixed target would need a
        // filtered `all` here — the per-layer dispatch checks its OWN handle,
        // and a layer falling back to wy17/chunked writes h_state live,
        // making the InterSlot commit source wrong for that layer.
        let injection_routes_tree = flat_injected
            && std::env::var("ATLAS_FORCE_WY17").ok().as_deref() != Some("1")
            && self.layers.iter().any(|l| l.gdn_tree_kernel_loaded());
        // Release/Acquire pairing with the commit-side swap: the scheduler
        // calls verify and commit sequentially on one thread today, but the
        // flag guards h_state commit-source selection (silent corruption if
        // desynced), so pay the fence and stay correct under any future
        // cross-thread step orchestration.
        self.dflash_flat_tree_route
            .store(injection_routes_tree, std::sync::atomic::Ordering::Release);
        // If a prior `seq_lazy` `no_mirror` commit left `h_state_checkpoint`
        // stale, refresh it from canonical live `ssm.h_state` whenever this
        // verify frame will NOT run flat-chain `seq_lazy` (e.g. tree mode or
        // non-lazy `k`), before Phase 2 mutates or reads checkpoint state.
        self.ensure_h_ckpt_fresh_if_non_lazy(seq, k, injection_routes_tree, stream)?;

        // DFS reorder: re-stamp parent_ids in DFS frame so the SSM (GDN
        // tree-WY) kernel processes tokens in DFS order, matching the
        // attention KV layout. After this, h_state_inter[i] holds the SSM
        // state AFTER DFS slot i. The commit path must map original-compact
        // indices → DFS slots via dfs_inv_perm to read the right inter slot.
        if dfs_active && k <= self.ddtree_parent_ids_capacity {
            use crate::layers::dflash_head::ddtree::permute_parent_ids;
            // host_parents is the current kernel-frame parent_ids (pre-DFS).
            // Pad with linear-chain tail to match k, mirroring the depth
            // derivation logic above.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            while host_parents.len() < k {
                let next_chain_parent = host_parents.len() as i32 - 1;
                host_parents.push(next_chain_parent);
            }
            if host_parents.len() == k {
                let permuted = permute_parent_ids(&host_parents, &dfs_perm, &dfs_inv_perm);
                let bytes: Vec<u8> = permuted.iter().flat_map(|p| p.to_le_bytes()).collect();
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(
                        &bytes,
                        self.ddtree_parent_ids_persistent,
                    )],
                    stream,
                )?;
                // Make sure dispatch uses persistent buffer (which now holds
                // DFS-frame parents).
                ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
            }
        }

        // Stash dfs_inv_perm so commit_verify_state_async_with_slot can map
        // original-compact `last_inter_slot` (from greedy_sample_ddtree) →
        // DFS slot (where the SSM kernel actually wrote the canonical state).
        // Cleared by clear_ddtree_parent_ids() after commit finishes reading.
        //
        // METADATA_ONLY=1: SSM still writes h_state_inter in kernel-original
        // slot order (because parent_ids re-stamping happens above either
        // way), so we still need the inv_perm mapping for commit.
        if dfs_active {
            *self.ddtree_dfs_inv_perm.lock() = dfs_inv_perm.clone();
        } else {
            self.ddtree_dfs_inv_perm.lock().clear();
        }

        // ATLAS_TREE_AWARE_ATTN: wire the per-row KV indirection plumbing
        // into the forward context so paged-decode attention can remap
        // tree-window positions to true ancestors. None when the env var
        // is off or DFS reorder is active.
        //
        // ATLAS_TREE_KV_PACK (additional): also pack ancestor KV into a
        // contiguous per-layer scratch pool and upload `seq_lens =
        // depth[t]+1` so the consumer kernel can run with NULL indirection
        // over the scratch — restores the fast BC=4 batched attention path
        // that the per-position fallback gives up.
        let tree_aware_attn = if tree_kv_active {
            let pack = if self.tree_kv_pack_active
                && !self.tree_kv_pack_scratch_k.is_empty()
                && !self.tree_kv_pack_scratch_v.is_empty()
            {
                // Upload per-step `seq_lens = depth[t]+1` into the dedicated
                // packed-KV seq_lens buffer (separate from the main attn
                // metadata `seq_len` which still holds `seq.seq_len + d + 1`
                // for the unpacked path / fallback).
                let depths_ref = tree_depths.as_ref().unwrap();
                let mut packed_seq_lens = vec![0i32; self.tree_kv_indir_stride];
                for t in 0..k {
                    packed_seq_lens[t] = (depths_ref[t] + 1) as i32;
                }
                // SAFETY: `packed_seq_lens` is an initialized `Vec<i32>` of
                // `tree_kv_indir_stride` POD elements, so viewing it as
                // `size_of_val(packed_seq_lens.as_slice())` bytes is valid for
                // `u8`; `copy_h2d_group_fast` finishes reading `sl_bytes_view`
                // before `packed_seq_lens` is dropped.
                let sl_bytes_view = unsafe {
                    std::slice::from_raw_parts(
                        packed_seq_lens.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(packed_seq_lens.as_slice()),
                    )
                };
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(
                        sl_bytes_view,
                        self.tree_kv_pack_seq_lens,
                    )],
                    stream,
                )?;
                // NOTE: kv_cache mutex is already held earlier in this fn
                // (`let mut kv_cache = self.kv_cache.lock();`), so we must
                // NOT re-lock it. Read block_size via the already-held
                // borrow (`kv_cache.block_size()`).
                let cache_block_size = kv_cache.block_size() as u32;
                static PACK_DBG_DONE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !PACK_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        "ATLAS_TREE_KV_PACK: k={k} chain_lens={:?} stride={} cache_bs={}",
                        &packed_seq_lens[..k],
                        self.tree_kv_indir_stride,
                        cache_block_size
                    );
                }
                Some(crate::layer::TreeKvPack {
                    num_attn_layers: self.tree_kv_pack_scratch_k.len() as u32,
                    scratch_k_ptrs: self.tree_kv_pack_scratch_k.as_ptr(),
                    scratch_v_ptrs: self.tree_kv_pack_scratch_v.as_ptr(),
                    identity_block_table: self.tree_kv_pack_block_table,
                    seq_lens: self.tree_kv_pack_seq_lens,
                    block_stride_bytes: self.tree_kv_pack_block_stride_bytes,
                    data_section_bytes: self.tree_kv_pack_data_section_bytes,
                    block_size: self.tree_kv_indir_stride as u32,
                    scatter_fp8_kernel: self.tree_kv_pack_scatter_fp8_kernel,
                    scatter_nvfp4_kernel: self.tree_kv_pack_scatter_nvfp4_kernel,
                    cache_block_size,
                    cache_max_blocks_per_seq: max_blocks,
                    // CUDA graph fix: read abs_base via device buffer so a
                    // captured graph picks up the fresh `seq.seq_len` on
                    // each replay. Buffer is updated above (before
                    // begin_capture).
                    abs_base_ptr: self.tree_kv_indir_base_persistent,
                })
            } else {
                None
            };
            Some(crate::layer::TreeAwareAttn {
                kv_indir: self.tree_kv_indir_persistent,
                // CUDA graph fix: see `abs_base_ptr` above.
                kv_indir_base_ptr: self.tree_kv_indir_base_persistent,
                kv_indir_stride: self.tree_kv_indir_stride as u32,
                pack,
            })
        } else {
            None
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: full_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev,
            tree_aware_attn,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        let receipt_on = self.config.is_qwen4_exp()
            && !dfs_active
            && crate::model::verify_logits_receipt::selected()?;
        let unified_tail_available = self.target_gumbel_verify_kernel.0 != 0
            && self.target_gumbel_verify_buf.0 != 0
            && !self.use_fp32_logits
            && (self.verify_lmhead_vocab() as usize) == self.config.vocab_size
            && !serial.lm_head
            && !dfs_active
            && !stage_capture
            && !layer_resolution_probe_enabled()
            && !receipt_on
            && self.config.pp_rank == 0
            && (1..=spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS).contains(&k);
        let staged_gumbel_reqs = staged_gumbel_reqs.filter(|reqs| {
            unified_tail_available
                && reqs.len() == k
                && super::device_topk::can_launch_single_gumbel_verify_run(reqs)
        });
        let rejection_tail_available = self.target_rejection_verify_kernel.0 != 0
            && !self.target_rejection_verify_buf.is_null()
            && !self.target_rejection_q_buf.is_null()
            && !self.use_fp32_logits
            && (self.verify_lmhead_vocab() as usize) == self.config.vocab_size
            && !serial.lm_head
            && !dfs_active
            && !stage_capture
            && !layer_resolution_probe_enabled()
            && !receipt_on
            && self.config.pp_rank == 0
            && (1..=super::device_topk::REJECTION_MAX_ROWS).contains(&k);
        let staged_rejection_reqs = staged_rejection_reqs.filter(|reqs| {
            rejection_tail_available
                && reqs.len() == k
                && super::device_topk::can_launch_single_rejection_verify_run(
                    reqs,
                    super::device_topk::REJECTION_Q_ROW_CAP,
                    super::device_topk::REJECTION_Q_STRIDE,
                )
        });
        let publish_in_graph_rejection = staged_rejection_reqs.is_some();
        let rejection_tail_active = staged_rejection_reqs.is_some()
            || (rejection_tail_available
                && staged_gumbel_reqs.is_none()
                && spark_runtime::sampler::gumbel::rejection_sampling_enabled());
        let gumbel_tail_active = !rejection_tail_active && unified_tail_available;
        let publish_in_graph_gumbel = gumbel_tail_active
            && staged_gumbel_reqs
                .as_ref()
                .is_some_and(|reqs| reqs.iter().any(|r| r.temperature > 0.0));
        let staged_non_greedy = staged_gumbel_reqs
            .as_ref()
            .is_some_and(|reqs| reqs.iter().any(|r| r.temperature != 0.0))
            || staged_rejection_reqs
                .as_ref()
                .is_some_and(|reqs| reqs.iter().any(|r| r.temperature != 0.0));
        let lm_head_tc_explicit = super::super::impl_a3::lm_head_tc_enabled()
            && self.lm_head_nvfp4_t.is_some()
            && self.w4a16_gemm_t_m32_n64_kernel.0 != 0;
        // Preserve strict C=1 0-ULP bit-exact greedy parity (`staged_non_greedy == false`)
        // by default, while allowing coalesced NVFP4 LM head for T>0 Gumbel/rejection,
        // host-policy/budget-nudge skip rows (`temp < 0`), or explicit greedy
        // `ATLAS_QWEN4_C1_TC_VERIFY=1` across both Flash-Next and dense 27B models.
        let c1_coalesced_lm_head = c1_coalesced_lm_head_eligible(
            gumbel_tail_active,
            rejection_tail_active,
            staged_non_greedy,
            qwen4_c1_tc_verify_enabled(),
            serial.lm_head,
            dfs_active,
            stage_capture,
            receipt_on,
            k,
            self.lm_head_nvfp4.is_some(),
            lm_head_tc_explicit,
            self.use_target_high_precision_lm_head(),
            crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled(),
        );
        let qwen4_layer0_prefix_graph = defer_qwen4_embed
            && qwen4_graph_prefix
            && super::verify_qwen4_batched::qwen4_layer0_prefix_graph_enabled()
            && !self.layers.is_empty()
            && !verify_skip_layer(0)
            && !receipt_on
            && (1..=spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS).contains(&k)
            && k * 4 <= crate::layer::STEP_META_SCRATCH_OFFSET
            && qwen4_layer_route != Some(Qwen4VerifyLayerRoute::RowSerialOracle)
            && self.preflight_embed_qwen4_batch(k).is_ok();
        let layer0_token_ids_dev = if qwen4_layer0_prefix_graph {
            Some(self.buffers.scratch())
        } else {
            None
        };
        if rejection_tail_active {
            let stage = if let Some(ref reqs) = staged_rejection_reqs {
                super::device_topk::encode_rejection_verify_chunk(reqs)?
            } else {
                const GREEDY_REJECTION_ROW_TEMPLATE: super::device_topk::RejectionVerifyRowSpec =
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
                let mut greedy_buf =
                    [GREEDY_REJECTION_ROW_TEMPLATE; super::device_topk::REJECTION_MAX_ROWS];
                for (row_idx, slot) in greedy_buf.iter_mut().take(k).enumerate() {
                    slot.row_idx = row_idx;
                }
                super::device_topk::encode_rejection_verify_chunk(&greedy_buf[..k])?
            };
            if let Some(token_ids_dev) = layer0_token_ids_dev {
                // SAFETY: `tokens` is an initialized `&[u32]` slice of `k * 4`
                // bytes; `copy_h2d_group_fast` finishes reading both slices into
                // its pinned ring (or synchronous H2D) before returning.
                let token_ids_bytes = unsafe {
                    std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), tokens.len() * 4)
                };
                self.copy_h2d_group_fast(
                    &[
                        HostToDeviceCopy::new(&stage, self.target_rejection_verify_buf),
                        HostToDeviceCopy::new(token_ids_bytes, token_ids_dev),
                    ],
                    stream,
                )?;
            } else {
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(
                        &stage,
                        self.target_rejection_verify_buf,
                    )],
                    stream,
                )?;
            }
        } else if gumbel_tail_active {
            let stage = if let Some(ref reqs) = staged_gumbel_reqs {
                super::device_topk::encode_gumbel_verify_chunk(reqs)?
            } else {
                const GREEDY_ROW_TEMPLATE:
                    spark_runtime::sampler::device_topk::GumbelVerifyRowSpec =
                    spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
                        row_idx: 0,
                        seed: 0,
                        abs_pos: 0,
                        temperature: 0.0,
                        top_k: 1,
                        top_p: 1.0,
                    };
                let mut greedy_buf = [GREEDY_ROW_TEMPLATE;
                    spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS];
                for (row_idx, slot) in greedy_buf.iter_mut().take(k).enumerate() {
                    slot.row_idx = row_idx;
                }
                super::device_topk::encode_gumbel_verify_chunk(&greedy_buf[..k])?
            };
            if let Some(token_ids_dev) = layer0_token_ids_dev {
                // SAFETY: `tokens` is an initialized `&[u32]` slice of `k * 4`
                // bytes; `copy_h2d_group_fast` finishes reading both slices into
                // its pinned ring (or synchronous H2D) before returning.
                let token_ids_bytes = unsafe {
                    std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), tokens.len() * 4)
                };
                self.copy_h2d_group_fast(
                    &[
                        HostToDeviceCopy::new(&stage, self.target_gumbel_verify_buf),
                        HostToDeviceCopy::new(token_ids_bytes, token_ids_dev),
                    ],
                    stream,
                )?;
            } else {
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(&stage, self.target_gumbel_verify_buf)],
                    stream,
                )?;
            }
        } else if layer0_token_ids_dev.is_some() {
            let _ = self.upload_qwen4_batched_embed_tokens(tokens, stream)?;
        }
        if defer_qwen4_embed && !qwen4_layer0_prefix_graph && self.config.pp_rank == 0 {
            crate::kprof!(self.gpu.as_ref(), stream, "embed", {
                for (t, &src_token) in tokens.iter().enumerate() {
                    self.embed(
                        src_token,
                        hidden.offset(t * persistent_width * fp32),
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;
        }

        // ── Phase 2: CUDA graph capture / replay ──

        // Cache key includes (slot, k, graph-shape flags, Qwen4 K5 attention
        // topology). The pack flag is
        // critical: when ATLAS_TREE_KV_PACK is enabled, the first K=γ verify
        // (often pre-tree, flat-chain) captures a graph WITHOUT pack-pool
        // kernel arguments. Later verifies with non-flat trees can't reuse
        // that graph — its kernel sequence has stale arg pointers. Treating
        // pack as a separate cache key forces a fresh capture for the
        // pack-active path.
        // Bit 1: tree-aware indirection active. A graph captured WITHOUT the
        // indirection kernel arguments (flat step, kv_indirection=NULL) must
        // never be replayed for a tree-indirected step of the same K (and
        // vice versa) — the kernel argument sets differ.
        // Bit 2: a real/synthetic SSM tree parent pointer is active. Flat K4
        // uses exact FP32 sequence recurrence, whereas a genuine DDTree K4
        // uses the tree kernel family; those graph topologies cannot alias.
        // Bit 3: the experimental native Qwen4 K16 batched layer path. It has
        // a different launch topology from the released row-serial oracle.
        // Bit 4: Qwen4 QSA pure-dense prefix (`seq_len + k <= 2048`). Its
        // graph skips QSA score/select and executes dense paged attention with
        // fused QSA staging, so it must never alias a post-2048 sparse QSA graph.
        // Bit 5: unified in-graph `target_gumbel_verify_rows_bf16` tail active
        // (covers `temp > 0` Gumbel-Max, `temp == 0` 0-ULP greedy argmax, and
        // `temp < 0` host-policy skip without splitting the graph cache key).
        // Bit 6: C=1 coalesced NVFP4 LM head active (`k >= 4`; enabled for
        // non-greedy frames by default and for greedy frames under
        // `ATLAS_QWEN4_C1_TC_VERIFY=1`).
        // Bit 7: C=1 Qwen4 Layer-0 prefix (`embed -> Layer 0`) CUDA graph.
        // Bit 8: fused vs split Qwen4 batched embedding kernel in Layer-0 prefix.
        // Bit 9: unified in-graph `target_rejection_verify_rows_bf16` tail active.
        // Bit 10: `seq.qwen4_qsa_required` active. Prevents a cached non-QSA
        // dense graph (bit 4 = 0, bit 10 = 0) from aliasing a post-2048 sparse
        // QSA graph (bit 4 = 0, bit 10 = 1) on the same slot.
        // Opt-in (`ATLAS_QWEN4_C1_TC_VERIFY=1`) scoped TC dense/MoE routing for
        // single-sequence Qwen4 verify (`k >= 6`); default OFF preserves strict
        // C=1 0-ULP bit-exact greedy parity across K=5..8.
        let qwen4_tc_dense_active = self.config.is_qwen4_exp()
            && qwen4_c1_tc_verify_enabled()
            && k >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS
            && crate::layers::qwen4_hyper::batched_verify_tc_dense_active(k, use_graphs);
        let tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(qwen4_tc_dense_active);
        // When PLE is mmap-backed (`--ple-ram` / `ATLAS_PLE_MMAP=1`), eliminate the
        // eager host-GPU gap between Layer 0 and Layer 1 by uploading the <=13.5 KiB
        // PLE records inline before Layer 0 and capturing `embed -> Layer 0 -> PLE ->
        // Layers 1..47 -> head` inside ONE unified 48-layer CUDA graph (overlapping
        // `dequant + key_proj + value_proj` on `secondary_stream` concurrently with
        // Layer 0 on high-SM GPUs). Disk-backed PLE stays untouched on the existing
        // `PleWarmWorker` + 2-graph split path.
        let ple_mmap_eligible = self.qwen4_ple.as_ref().is_some_and(|ple| {
            ple.unified_graph_eligible(k, persistent_width * fp32, seq.slot_idx)
        });
        let qwen4_unified_ple_graph = qwen4_unified_ple_verify_graph_eligible(
            qwen4_layer0_prefix_graph && layer0_token_ids_dev.is_some(),
            self.layers.len(),
            verify_skip_layer(1),
            self.config.pp_world_size,
            kgamma_debug_dump,
            ple_mmap_eligible,
        );
        let unified_embed_fused = Self::qwen4_embed_expand_fused_selected();
        let qwen4_unified_ple_overlap = if qwen4_unified_ple_graph {
            let ple = self
                .qwen4_ple
                .as_ref()
                .expect("unified PLE graph requires qwen4_ple");
            let prep_stream = (self.secondary_stream != 0 && self.secondary_stream != stream)
                .then_some(self.secondary_stream);
            ple.prepare_unified_verify_graph(k, prep_stream, stream)?
        } else {
            false
        };
        let qwen4_split_graph_prefix = qwen4_graph_prefix && !qwen4_unified_ple_graph;
        let pack_key =
            super::commit_plan::verify_graph_shape_key(
                ctx.tree_aware_attn.and_then(|t| t.pack).is_some(),
                ctx.tree_aware_attn.is_some(),
                ctx.ddtree_parent_ids_dev.is_some(),
            ) | if qwen4_layer_route == Some(Qwen4VerifyLayerRoute::ExperimentalK16Batched) {
                QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT
            } else {
                0
            } | verify_d_qsa_graph_bits(seq.qwen4_qsa_required, qsa_dense_prefix_graph)
                | verify_d_gumbel_tail_graph_bits(gumbel_tail_active, c1_coalesced_lm_head)
                | if rejection_tail_active {
                    QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT
                } else {
                    0
                }
                | verify_d_unified_ple_graph_bits(
                    qwen4_unified_ple_graph,
                    unified_embed_fused,
                    qwen4_unified_ple_overlap,
                );
        let cache_key = (seq.slot_idx, k, pack_key, qwen4_k5_attention_topology);
        // Submit C=1 PLE warm-up before Layer 0 on disk-backed PLE so
        // ParkedPleCache misses / NVMe reads overlap with Layer 0 GPU execution.
        // For mmap-backed unified graphs, pack and upload inline directly before
        // the single 48-layer graph launch with zero background thread wakeup.
        let mut ple_warm_ticket = if !qwen4_unified_ple_graph
            && self.config.is_qwen4_exp()
            && (1..=crate::layers::qwen4_ple::PleWarmSeq::MAX_DRAFT_LEN).contains(&tokens.len())
            && let Some(ple) = self.qwen4_ple.as_ref()
            && crate::layers::qwen4_ple::ple_verify_batch_selected()
        {
            let warm_seq = crate::layers::qwen4_ple::PleWarmSeq::from_slices(tokens, &seq.tokens)?;
            Some(ple.submit_warm_cross_seq_verify(vec![warm_seq])?)
        } else {
            None
        };
        if qwen4_unified_ple_graph {
            let ple = self
                .qwen4_ple
                .as_ref()
                .expect("unified PLE graph requires qwen4_ple");
            ple.upload_verify_records_inline_mmap(tokens, &seq.tokens, self.gpu.as_ref(), stream)?;
        }
        let mut active_layer0_cache_key: Option<VerifyKgammaGraphKey> = None;
        if qwen4_split_graph_prefix {
            if let Some(token_ids_dev) = layer0_token_ids_dev {
                let embed_fused = Self::qwen4_embed_expand_fused_selected();
                let layer0_pack_key = verify_d_layer0_prefix_pack_key(pack_key, embed_fused);
                let layer0_cache_key = (
                    seq.slot_idx,
                    k,
                    layer0_pack_key,
                    qwen4_k5_attention_topology,
                );
                active_layer0_cache_key = Some(layer0_cache_key);
                let layer0_warm_key: VerifyDLayer0WarmKey =
                    (k, layer0_pack_key, qwen4_k5_attention_topology);
                let model_id = (
                    std::ptr::from_ref(self) as usize,
                    self.buffers.hidden_states().0,
                );
                let required_warmup_hits =
                    super::verify_qwen4_batched::batched_verify_graph_required_warmup_hits(1);
                let (capture_poisoned, layer0_warm) = {
                    let tracker =
                        VERIFY_D_LAYER0_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
                    let mut map = tracker.lock();
                    let state = map.entry(model_id).or_default();
                    (
                        state.capture_poisoned,
                        state.is_warm(&layer0_warm_key, required_warmup_hits),
                    )
                };
                let cached_layer0 = if !capture_poisoned {
                    self.lookup_verify_kgamma_graph(&layer0_cache_key)
                } else {
                    None
                };
                let eager_ctx = ForwardContext {
                    graph_capture: false,
                    ..ctx
                };
                if let Some(graph) = cached_layer0 {
                    self.gpu.launch_graph(graph, stream)?;
                } else if capture_poisoned || !layer0_warm {
                    let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
                    self.launch_qwen4_batched_embed_from_dev(
                        embed_kernels,
                        token_ids_dev,
                        hidden,
                        k,
                        stream,
                    )?;
                    self.qwen4_verify_layer(
                        0,
                        self.layers[0].as_ref(),
                        0,
                        qwen4_layer_route,
                        k,
                        seq,
                        &mut kv_cache,
                        &eager_ctx,
                        mb,
                        hidden,
                        residual,
                        persistent_width * fp32,
                        stream,
                    )?;
                    self.try_dflash_capture_rows(0, k, stream)?;
                    if !capture_poisoned {
                        let tracker =
                            VERIFY_D_LAYER0_TRACKER.get_or_init(|| Mutex::new(HashMap::new()));
                        let mut map = tracker.lock();
                        map.entry(model_id)
                            .or_default()
                            .record_warmup_hit(layer0_warm_key);
                    }
                } else {
                    // Evict stale QSA-regime graphs for this slot and enforce
                    // bounded per-tier LRU / UMA-headroom caps strictly outside
                    // capture before pre-warming and instantiating a new graph.
                    self.prepare_verify_kgamma_graph_slot(
                        layer0_cache_key,
                        Some(cache_key),
                        stream,
                    )?;
                    // Pre-warm Layer-0 lazy buffers strictly outside capture so
                    // CaptureGuard never allocates or memsets (even when a new
                    // slot captures before the suffix loop pre-warms it).
                    if k >= 2 {
                        let prewarm_qsa = seq.qwen4_qsa_required
                            && crate::layers::qwen4_qsa::verify_journal_rows(k);
                        match self.config.layer_type(0) {
                            LayerType::LinearAttention => {
                                self.layers[0].ensure_qwen4_cross_seq_fused_descs_ready(
                                    self.gpu.as_ref(),
                                    &self.config,
                                    k,
                                )?;
                            }
                            LayerType::FullAttention if prewarm_qsa => {
                                self.layers[0]
                                    .prepare_qwen4_qsa_verify(&kv_cache, self.gpu.as_ref())?;
                            }
                            _ => {}
                        }
                    }
                    let embed_kernels = self.resolve_qwen4_batched_embed_kernels()?;
                    let capture_res: Result<GraphHandle> = (|| {
                        let capture = super::capture_guard::CaptureGuard::begin(
                            self.gpu.as_ref(),
                            stream,
                            "verify_d_layer0",
                        )?;
                        self.launch_qwen4_batched_embed_from_dev(
                            embed_kernels,
                            token_ids_dev,
                            hidden,
                            k,
                            stream,
                        )?;
                        self.qwen4_verify_layer(
                            0,
                            self.layers[0].as_ref(),
                            0,
                            qwen4_layer_route,
                            k,
                            &mut *seq,
                            &mut kv_cache,
                            &ctx,
                            mb,
                            hidden,
                            residual,
                            persistent_width * fp32,
                            stream,
                        )?;
                        self.try_dflash_capture_rows(0, k, stream)?;
                        capture.end()
                    })();
                    match capture_res {
                        Ok(graph) if graph.0 != 0 => {
                            tracing::info!(
                                session_hash = seq.session_hash,
                                slot = seq.slot_idx,
                                k,
                                seq_len = seq.seq_len,
                                pack_key = layer0_pack_key,
                                embed_fused,
                                "QWEN4_C1_VERIFY_LAYER0_GRAPH_CAPTURED prefix embed+layer0"
                            );
                            self.store_verify_kgamma_graph(layer0_cache_key, graph, stream)?;
                            {
                                let tracker = VERIFY_D_LAYER0_TRACKER
                                    .get_or_init(|| Mutex::new(HashMap::new()));
                                let mut map = tracker.lock();
                                map.entry(model_id)
                                    .or_default()
                                    .record_captured(layer0_warm_key, required_warmup_hits);
                            }
                            self.gpu.launch_graph(graph, stream)?;
                        }
                        Ok(_) => {
                            // Mock/no-graph backend executed the prefix inline during capture.
                        }
                        Err(capture_err) => {
                            {
                                let tracker = VERIFY_D_LAYER0_TRACKER
                                    .get_or_init(|| Mutex::new(HashMap::new()));
                                let mut map = tracker.lock();
                                map.entry(model_id).or_default().record_capture_failure();
                            }
                            tracing::warn!(
                                slot = seq.slot_idx,
                                k,
                                seq_len = seq.seq_len,
                                "ATLAS_QWEN4_LAYER0_PREFIX_GRAPH C=1 capture failed ({capture_err:#}); \
                                 falling back to eager execution"
                            );
                            self.launch_qwen4_batched_embed_from_dev(
                                embed_kernels,
                                token_ids_dev,
                                hidden,
                                k,
                                stream,
                            )?;
                            self.qwen4_verify_layer(
                                0,
                                self.layers[0].as_ref(),
                                0,
                                qwen4_layer_route,
                                k,
                                seq,
                                &mut kv_cache,
                                &eager_ctx,
                                mb,
                                hidden,
                                residual,
                                persistent_width * fp32,
                                stream,
                            )?;
                            self.try_dflash_capture_rows(0, k, stream)?;
                        }
                    }
                }
            } else if !self.layers.is_empty() && !verify_skip_layer(0) {
                let eager_ctx = ForwardContext {
                    graph_capture: false,
                    ..ctx
                };
                self.qwen4_verify_layer(
                    0,
                    self.layers[0].as_ref(),
                    0,
                    qwen4_layer_route,
                    k,
                    seq,
                    &mut kv_cache,
                    &eager_ctx,
                    mb,
                    hidden,
                    residual,
                    persistent_width * fp32,
                    stream,
                )?;
                self.capture_k1_stage("layer_00", hidden, k, persistent_width * fp32, stream)?;
                self.try_dflash_capture_rows(0, k, stream)?;
            }
            self.qwen4_verify_ple_with_ticket(
                tokens,
                seq,
                ple_warm_ticket.take(),
                hidden,
                persistent_width * fp32,
                stream,
            )?;
        }
        let cached_for_slot = if use_graphs {
            self.lookup_verify_kgamma_graph(&cache_key)
        } else {
            None
        };
        if let Some(graph) = cached_for_slot {
            self.gpu.launch_graph(graph, stream)?;
        }
        let mut lrp_snapshots: Vec<Vec<u8>> = Vec::new();
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            let control_guard = if controlled_verify {
                let counts = self.layers.iter().enumerate().fold(
                    crate::model::control_engagement::LayerCounts::default(),
                    |mut counts, (layer_idx, _)| {
                        if !verify_skip_layer(layer_idx) {
                            if self.config.layer_type(layer_idx) == LayerType::FullAttention {
                                counts.attention += 1;
                            } else {
                                counts.ssm += 1;
                            }
                        }
                        counts
                    },
                );
                if hss_engaged && counts.attention > 0 && control_requests.attention_requested() {
                    bail!(
                        "DFLASH_CONTROL_PATH_PROOF requested=true engaged=false \
                         requirement=attention controls require the multi-sequence HBM path"
                    );
                }
                crate::model::control_engagement::begin(control_requests, counts)?
            } else {
                None
            };

            // Evict stale QSA-regime graphs for this slot and enforce bounded
            // per-tier LRU / UMA-headroom caps strictly outside capture before
            // pre-warming lazy layer buffers and instantiating a new graph.
            if use_graphs {
                self.prepare_verify_kgamma_graph_slot(cache_key, active_layer0_cache_key, stream)?;
            }

            // Pre-warm lazy per-layer buffers (`gdn_lazy_retain_pool` for every
            // SSM layer on both Qwen4 and Qwen3.8-27B, plus QSA verify side cache
            // and undo journal for Qwen4 attention layers) strictly outside CUDA
            // graph capture so Layers 1.. never allocate or memset inside `CaptureGuard`.
            if use_graphs && k >= 2 {
                let prewarm_qsa = self.config.is_qwen4_exp()
                    && seq.qwen4_qsa_required
                    && crate::layers::qwen4_qsa::verify_journal_rows(k);
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    if verify_skip_layer(layer_idx) {
                        continue;
                    }
                    match self.config.layer_type(layer_idx) {
                        LayerType::LinearAttention => {
                            layer.ensure_qwen4_cross_seq_fused_descs_ready(
                                self.gpu.as_ref(),
                                &self.config,
                                k,
                            )?;
                        }
                        LayerType::FullAttention if prewarm_qsa => {
                            layer.prepare_qwen4_qsa_verify(&kv_cache, self.gpu.as_ref())?;
                        }
                        _ => {}
                    }
                }
            }

            if c1_coalesced_lm_head || qwen4_tc_dense_active {
                crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(self.gpu.as_ref());
            }

            let unified_embed_kernels = if qwen4_unified_ple_graph {
                Some(self.resolve_qwen4_batched_embed_kernels()?)
            } else {
                None
            };

            // Was `verify_d.rs:1490,1878`'s bare `begin_capture(stream)?` ..
            // `end_capture(stream)?`. See `trait_impl::capture_guard` /
            // `DEBUG_SYNC_AUDIT.md`.
            let mut capture = if use_graphs {
                Some(super::capture_guard::CaptureGuard::begin(
                    self.gpu.as_ref(),
                    stream,
                    "verify_d",
                )?)
            } else {
                None
            };

            struct PlePrepForkGuard<'a> {
                ple: &'a crate::layers::qwen4_ple::Qwen4PleLayer,
                gpu: &'a dyn GpuBackend,
                stream: u64,
                armed: bool,
            }
            impl Drop for PlePrepForkGuard<'_> {
                fn drop(&mut self) {
                    if self.armed {
                        let _ = self.ple.join_verify_prep_in_graph(self.gpu, self.stream);
                    }
                }
            }
            let mut ple_prep_fork_guard: Option<PlePrepForkGuard<'_>> = None;
            if qwen4_unified_ple_graph {
                let ple = self
                    .qwen4_ple
                    .as_ref()
                    .expect("unified PLE graph requires qwen4_ple");
                if qwen4_unified_ple_overlap {
                    let forked = ple.launch_verify_prep_fork_in_graph(
                        k,
                        self.gpu.as_ref(),
                        stream,
                        self.secondary_stream,
                    )?;
                    if forked {
                        ple_prep_fork_guard = Some(PlePrepForkGuard {
                            ple,
                            gpu: self.gpu.as_ref(),
                            stream,
                            armed: true,
                        });
                    }
                }
                let embed_kernels =
                    unified_embed_kernels.expect("unified PLE graph resolves embed kernels");
                let token_ids_dev =
                    layer0_token_ids_dev.expect("unified PLE graph requires layer0_token_ids_dev");
                self.launch_qwen4_batched_embed_from_dev(
                    embed_kernels,
                    token_ids_dev,
                    hidden,
                    k,
                    stream,
                )?;
            }

            let verify_stages =
                crate::model::decode_stages::enabled() && self.config.is_qwen4_exp();
            if verify_stages {
                spark_runtime::cublaslt::stage_timer::begin(stream);
            }
            let mut qwen4_ssm_layer_idx = 0usize;
            let qwen4_batch_hyper =
                std::env::var("ATLAS_QWEN4_K5_BATCH_HYPER").ok().as_deref() != Some("0");
            let qwen4_fuse_inject_gn = self.config.is_qwen4_exp()
                && crate::layers::qwen4_hyper::verify_hc_fuse_inject_gn_enabled();
            let qwen4_has_dflash_save = self.dflash_hidden_save.is_some();
            let qwen4_is_rank0 = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
            let qwen4_k1_diag = crate::model::k1_stage_diag::serial_active()
                || crate::model::k1_stage_diag::batch_active();
            let qwen4_lrp = layer_resolution_probe_enabled();
            let qwen4_batched_final_mixer = !serial.final_norm
                && (2..=32).contains(&k)
                && qwen4_batch_hyper
                && self.config.pp_world_size <= 1
                && self.qwen4_final_mixer.is_some();
            let mut verify_hc_fuse_scope = crate::layers::qwen4_hyper::VerifyHcFuseScope::enter();
            self.pp_stage_enter(hidden, k, stream)?;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);
                if verify_stages {
                    spark_runtime::cublaslt::stage_timer::set_group(if layer_idx == 0 {
                        "l0"
                    } else if layer_type == atlas_core::config::LayerType::FullAttention {
                        "attn"
                    } else {
                        "ssm"
                    });
                }

                // ATLAS_VERIFY_SKIP_LAYERS="2,3,7,...": static layer-skip in the
                // single-stream γ verify. A skipped layer is identity (residual
                // passes through unchanged) at EVERY step, so its SSM/KV state is
                // never written and never read — consistent, graph-capture-safe.
                // Lossy vs md5 but pass@1-gated (measured: skip-16 = 25% verify
                // bandwidth cut at ~pass@1-preserving on HumanEval-style coding).
                // Capture layers [1,10,18,27,35,44,52,61] must NOT be in the set.
                if verify_skip_layer(layer_idx) {
                    continue;
                }

                if self.config.is_qwen4_exp() {
                    if qwen4_split_graph_prefix && layer_idx == 0 {
                        // Ran eagerly (or via separate Layer-0 graph) before the suffix graph.
                        if layer_type == LayerType::LinearAttention {
                            qwen4_ssm_layer_idx += 1;
                        }
                        continue;
                    }
                    if layer_idx == 1 && !qwen4_split_graph_prefix {
                        if verify_stages {
                            spark_runtime::cublaslt::stage_timer::set_group("");
                        }
                        if qwen4_unified_ple_graph {
                            let ple = self
                                .qwen4_ple
                                .as_ref()
                                .expect("unified PLE graph requires qwen4_ple");
                            let prep_forked = ple_prep_fork_guard.as_mut().is_some_and(|g| {
                                let was_armed = g.armed;
                                g.armed = false;
                                was_armed
                            });
                            ple.launch_verify_fuse_in_graph(
                                k,
                                hidden,
                                persistent_width * fp32,
                                seq.slot_idx,
                                prep_forked,
                                self.gpu.as_ref(),
                                stream,
                            )?;
                        } else {
                            self.qwen4_verify_ple_with_ticket(
                                tokens,
                                seq,
                                ple_warm_ticket.take(),
                                hidden,
                                persistent_width * fp32,
                                stream,
                            )?;
                        }
                        if verify_stages {
                            spark_runtime::cublaslt::stage_timer::mark("ple", stream);
                            spark_runtime::cublaslt::stage_timer::set_group(
                                if layer_type == LayerType::FullAttention {
                                    "attn"
                                } else {
                                    "ssm"
                                },
                            );
                        }
                    }

                    let defer_tail = qwen4_verify_d_layer_should_defer_mlp_tail(
                        qwen4_fuse_inject_gn,
                        qwen4_layer_route != Some(Qwen4VerifyLayerRoute::RowSerialOracle),
                        crate::layers::qwen4_hybrid_rows(k),
                        layer_idx,
                        layer_idx + 1 == self.layers.len(),
                        qwen4_batched_final_mixer,
                        qwen4_has_dflash_save,
                        qwen4_is_rank0,
                        &self.dflash_capture_layers,
                        qwen4_k1_diag
                            || (qwen4_lrp && LRP_PROBE_LAYERS.contains(&layer_idx))
                            || (kgamma_debug_dump && !use_graphs),
                    );
                    verify_hc_fuse_scope.set_defer_mlp_tail(defer_tail);
                    self.qwen4_verify_layer(
                        layer_idx,
                        layer.as_ref(),
                        qwen4_ssm_layer_idx,
                        qwen4_layer_route,
                        k,
                        seq,
                        &mut kv_cache,
                        &ctx,
                        mb,
                        hidden,
                        residual,
                        persistent_width * fp32,
                        stream,
                    )?;
                    verify_hc_fuse_scope.set_defer_mlp_tail(false);
                    if layer_type == LayerType::LinearAttention {
                        qwen4_ssm_layer_idx += 1;
                    }
                } else if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // HSS path: decode_multi_seq's paged-decode kernel
                        // reads K/V from HBM only, missing the long-context
                        // history on disk. Fall back to decode_batched
                        // (sequential single-token decodes via the HSS
                        // orchestrator). See verify_b.rs for full rationale.
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        crate::kprof!(self.gpu.as_ref(), stream, "verify_attn", {
                            layer.decode_multi_seq(
                                hidden,
                                residual,
                                k,
                                &mut refs,
                                &mut kv_cache,
                                &seq_lens_vec,
                                &block_tables_vec,
                                &ctx,
                                stream,
                            )
                        })?;
                    }
                } else {
                    crate::kprof!(self.gpu.as_ref(), stream, "verify_ssm", {
                        layer.decode_batched(
                            hidden,
                            residual,
                            k,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )
                    })?;
                }
                self.capture_k1_stage(
                    &format!("layer_{layer_idx:02}"),
                    hidden,
                    k,
                    persistent_width * fp32,
                    stream,
                )?;
                // DFlash hidden capture for ctx conditioning. Save ALL k
                // tokens so the scheduler can pick the correct one
                // (num_accepted) after verify. Layout:
                // [token_idx, capture_layer, hidden] in dflash_hidden_save.
                self.try_dflash_capture_rows(layer_idx, k, stream)?;
                // ATLAS_LAYER_RESOLUTION_PROBE: snapshot `[K, persistent_width]`
                // hidden state at checkpoint layers. GPU sync required (probe is
                // eager-only).
                if layer_resolution_probe_enabled() && LRP_PROBE_LAYERS.contains(&layer_idx) {
                    let row_bytes = persistent_width * fp32;
                    let mut snap = vec![0u8; k * row_bytes];
                    self.gpu.synchronize(stream)?;
                    self.gpu.copy_d2h(hidden, &mut snap)?;
                    lrp_snapshots.push(snap);
                }
                // ATLAS_KGAMMA_DEBUG_DUMP=1: dump per-position hidden state
                // for K=γ verify so we can diff against HF reference
                // (modelforge inspect-batched). Writes
                // /tmp/atlas_kgamma_layer{L}_pos{t}.bin one-shot per pair.
                // Dump only when NOT capturing a CUDA graph (sync ops illegal
                // during capture). Set ATLAS_DFLASH_DEBUG_NO_GRAPH=1 too.
                if kgamma_debug_dump && !use_graphs {
                    let row_bytes = persistent_width * fp32;
                    for t in 0..k {
                        let path = format!("/tmp/atlas_kgamma_layer{}_pos{}.bin", layer_idx, t);
                        if !std::path::Path::new(&path).exists() {
                            let mut buf = vec![0u8; row_bytes];
                            self.gpu.synchronize(stream)?;
                            self.gpu.copy_d2h(hidden.offset(t * row_bytes), &mut buf)?;
                            let _ = std::fs::write(&path, &buf);
                        }
                    }
                }
            }
            verify_hc_fuse_scope.set_defer_mlp_tail(false);
            if self.config.pp_world_size > 1
                && let Some(pending) = crate::layers::qwen4_hyper::take_pending_verify_hc_inject()
            {
                let mixer = self
                    .qwen4_final_mixer
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("qwen4_exp verify missing final mixer"))?;
                mixer.inject_saved_batched(
                    pending.hyper,
                    pending.core,
                    pending.residual,
                    pending.rows,
                    self.gpu.as_ref(),
                    pending.stream,
                )?;
            }
            self.pp_stage_exit(hidden, k, stream)?;

            if let Some(guard) = control_guard {
                guard.finish()?;
            }
            if !use_graphs && self.config.pp_rank > 0 {
                drop(verify_hc_fuse_scope);
                for &t in tokens {
                    seq.tokens.push(t);
                }
                seq.seq_len += k;
                return Ok(vec![0u32; k]);
            }

            let normed = self.buffers.norm_output();
            if self.config.is_qwen4_exp() {
                if !serial.final_norm && (2..=32).contains(&k) && qwen4_batch_hyper {
                    let mixer = self
                        .qwen4_final_mixer
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("qwen4_exp verify missing final mixer"))?;
                    match crate::layers::qwen4_hyper::take_pending_verify_hc_inject() {
                        Some(pending) if pending.matches(hidden, residual, k, stream) => {
                            mixer.prepare_batched_after_inject(
                                hidden,
                                pending.core,
                                residual,
                                k,
                                &self.buffers,
                                self.gpu.as_ref(),
                                self.config.rms_norm_eps as f32,
                                stream,
                            )?;
                        }
                        other => {
                            if let Some(pending) = other {
                                mixer.inject_saved_batched(
                                    pending.hyper,
                                    pending.core,
                                    pending.residual,
                                    pending.rows,
                                    self.gpu.as_ref(),
                                    pending.stream,
                                )?;
                            }
                            mixer.prepare_batched(
                                hidden,
                                residual,
                                k,
                                &self.buffers,
                                self.gpu.as_ref(),
                                self.config.rms_norm_eps as f32,
                                stream,
                            )?;
                        }
                    }
                } else {
                    if let Some(pending) =
                        crate::layers::qwen4_hyper::take_pending_verify_hc_inject()
                    {
                        let mixer = self.qwen4_final_mixer.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("qwen4_exp verify missing final mixer")
                        })?;
                        mixer.inject_saved_batched(
                            pending.hyper,
                            pending.core,
                            pending.residual,
                            pending.rows,
                            self.gpu.as_ref(),
                            pending.stream,
                        )?;
                    }
                    let saved_row0 = self.buffers.attn_output();
                    for t in 0..k {
                        let row_offset = t * persistent_width * fp32;
                        self.qwen4_final_hidden_nvfp4(
                            hidden.offset(row_offset),
                            residual.offset(row_offset),
                            stream,
                        )?
                        .ok_or_else(|| anyhow::anyhow!("qwen4_exp verify missing final mixer"))?;
                        if t == 0 {
                            self.gpu
                                .copy_d2d_async(normed, saved_row0, h * bf16, stream)?;
                        } else {
                            self.gpu.copy_d2d_async(
                                normed,
                                normed.offset(t * h * bf16),
                                h * bf16,
                                stream,
                            )?;
                        }
                    }
                    self.gpu
                        .copy_d2d_async(saved_row0, normed, h * bf16, stream)?;
                }
                self.final_norm_rows(normed, normed, k as u32, stream)?;
            } else if serial.final_norm {
                self.dflash_k1_final_norm(hidden, k, stream)?;
            } else {
                crate::kprof!(self.gpu.as_ref(), stream, "final_norm", {
                    self.final_norm_rows(hidden, normed, k as u32, stream)?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            drop(verify_hc_fuse_scope);
            self.capture_k1_stage("final_norm", normed, k, h * bf16, stream)?;

            // DEBUG: dump post-final-norm hidden states (K positions)
            if kgamma_debug_dump && !use_graphs {
                let h_bytes = h * 2;
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_final_norm_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; h_bytes];
                        self.gpu.synchronize(stream)?;
                        self.gpu.copy_d2h(normed.offset(t * h_bytes), &mut buf)?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            let lm_head_proof = if serial.lm_head {
                let (_, proof) = self.dflash_k1_lm_head_argmax(normed, k, stream)?;
                Some(proof)
            } else {
                if verify_stages {
                    spark_runtime::cublaslt::stage_timer::set_group("");
                    spark_runtime::cublaslt::stage_timer::mark("pre_head", stream);
                }
                crate::kprof!(self.gpu.as_ref(), stream, "lm_head", {
                    let mut dispatched_coalesced = false;
                    if c1_coalesced_lm_head && let Some(ref nvfp4) = self.lm_head_nvfp4 {
                        let v = self.verify_lmhead_vocab();
                        let logits = self.buffers.logits();
                        dispatched_coalesced =
                            crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                                self.gpu.as_ref(),
                                normed,
                                nvfp4,
                                logits,
                                k,
                                v as usize,
                                h,
                                stream,
                            )?;
                        if dispatched_coalesced && self.logit_softcap_kernel.0 != 0 {
                            let cap = self.config.final_logit_softcapping;
                            self.apply_logit_softcap(logits, k as u32 * v, cap, stream)?;
                        }
                    }
                    if !dispatched_coalesced {
                        self.lm_head_batched(normed, k as u32, stream)?;
                    }
                    anyhow::Result::<()>::Ok(())
                })?;
                None
            };
            if verify_stages {
                spark_runtime::cublaslt::stage_timer::set_group("");
                spark_runtime::cublaslt::stage_timer::mark("head", stream);
                crate::model::decode_stages::verify_done(k, seq.seq_len);
            }
            self.capture_k1_stage(
                "logits",
                self.buffers.logits(),
                k,
                self.config.vocab_size * bf16,
                stream,
            )?;

            if stage_capture {
                if let Some(proof) = lm_head_proof {
                    let family = serial_family.ok_or_else(|| {
                        anyhow::anyhow!("DFLASH_K1_LM_HEAD_PATH_PROOF lacks a named serial family")
                    })?;
                    let proof_line = proof.proof_line(family, seq.seq_len, tokens)?;
                    tracing::warn!("{proof_line}");
                }
                let report = crate::model::k1_stage_diag::finish_batch()?;
                if let Some(first) = report.first {
                    tracing::warn!(
                        run_id = report.manifest.run_id,
                        verify_step = report.manifest.verify_step,
                        pre_verify_len = report.manifest.pre_verify_len,
                        tokens = ?report.manifest.tokens,
                        absolute_seq_lens = ?report.manifest.absolute_seq_lens,
                        family = report.manifest.family,
                        stage = first.stage,
                        row = first.row,
                        mismatch_rows = ?first.mismatch_rows,
                        first_byte = first.first_byte,
                        serial_hash = format_args!("{:016x}", first.serial_hash),
                        batch_hash = format_args!("{:016x}", first.batch_hash),
                        stages = report.stages,
                        terminal_stage = report.terminal_stage,
                        logits_compared = report.logits_compared,
                        "DFLASH_K1_STAGE_FIRST_DIVERGENCE in_process_exact"
                    );
                } else {
                    tracing::info!(
                        run_id = report.manifest.run_id,
                        verify_step = report.manifest.verify_step,
                        pre_verify_len = report.manifest.pre_verify_len,
                        tokens = ?report.manifest.tokens,
                        absolute_seq_lens = ?report.manifest.absolute_seq_lens,
                        family = report.manifest.family,
                        stages = report.stages,
                        terminal_stage = report.terminal_stage,
                        logits_compared = report.logits_compared,
                        "DFLASH_K1_STAGE_MATCH in_process_exact"
                    );
                }
            }

            // DEBUG: dump logits for first 100 vocab entries per position
            if kgamma_debug_dump && !use_graphs {
                let vocab = self.verify_lmhead_vocab() as usize;
                let bf16 = 2usize;
                let dump_n = 100.min(vocab);
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_logits_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; dump_n * bf16];
                        self.gpu.synchronize(stream)?;
                        self.gpu
                            .copy_d2h(self.buffers.logits().offset(t * vocab * bf16), &mut buf)?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            // Argmax inside graph (fixed scratch addresses — graph-safe).
            // Use the verify-side (possibly truncated) vocab so the per-row
            // logits stride AND the argmax range match exactly what
            // `lm_head_batched` wrote for the K=γ transposed GEMM — full vocab
            // when `ATLAS_TARGET_LMHEAD_VOCAB` truncation is off.
            if !serial.lm_head {
                let vocab = self.verify_lmhead_vocab() as usize;
                let argmax_out = self.buffers.scratch();
                if rejection_tail_active {
                    crate::kprof!(self.gpu.as_ref(), stream, "rejection_verify", {
                        super::device_topk::launch_target_rejection_verify_single_run(
                            self.gpu.as_ref(),
                            self.target_rejection_verify_kernel,
                            self.buffers.logits(),
                            self.target_rejection_q_buf,
                            self.target_rejection_q_buf
                                .offset(super::device_topk::REJECTION_Q_PROBS_OFFSET),
                            self.target_rejection_verify_buf,
                            k,
                            vocab as u32,
                            vocab,
                            super::device_topk::REJECTION_Q_ROW_CAP,
                            super::device_topk::REJECTION_Q_STRIDE,
                            stream,
                        )
                    })?;
                } else if gumbel_tail_active {
                    crate::kprof!(self.gpu.as_ref(), stream, "gumbel_verify", {
                        super::device_topk::launch_target_gumbel_verify_single_run(
                            self.gpu.as_ref(),
                            self.target_gumbel_verify_kernel,
                            self.buffers.logits(),
                            self.target_gumbel_verify_buf,
                            k,
                            vocab as u32,
                            vocab,
                            stream,
                        )
                    })?;
                } else {
                    crate::kprof!(self.gpu.as_ref(), stream, "argmax", {
                        // ATLAS_VERIFY_ARGMAX_ROWS=1: one launch, one block per row,
                        // the same index per row as the per-row argmax_bf16 loop.
                        if verify_argmax_rows_selected() && self.argmax_rows_kernel.0 != 0 {
                            VERIFY_ARGMAX_ROWS_LAUNCHED.call_once(|| {
                                tracing::info!(
                                    "ENGAGED ATLAS_VERIFY_ARGMAX_ROWS: row-parallel verify argmax launched (rows={k})"
                                );
                            });
                            ops::argmax_bf16_rows(
                                self.gpu.as_ref(),
                                self.argmax_rows_kernel,
                                self.buffers.logits(),
                                argmax_out,
                                vocab as u32,
                                vocab as u32,
                                k as u32,
                                stream,
                            )?;
                        } else {
                            for t in 0..k {
                                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                                let out_t = argmax_out.offset(t * 4);
                                ops::argmax_bf16(
                                    self.gpu.as_ref(),
                                    self.argmax_kernel,
                                    logits_t,
                                    out_t,
                                    vocab as u32,
                                    stream,
                                )?;
                            }
                        }
                        anyhow::Result::<()>::Ok(())
                    })?;
                }
            }

            if use_graphs {
                let graph = capture
                    .take()
                    .expect("use_graphs implies begin_capture ran above")
                    .end()?;
                if graph.0 != 0 {
                    tracing::info!(
                        "{}",
                        super::verify_graph_receipts::captured_line(
                            seq.session_hash,
                            seq.slot_idx,
                            k,
                            seq.seq_len,
                            pack_key,
                        )
                    );
                    self.store_verify_kgamma_graph(cache_key, graph, stream)?;
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        drop(tc_dense_guard);
        if qwen4_unified_ple_graph && let Some(ple) = self.qwen4_ple.as_ref() {
            ple.note_verify_graph_submitted(self.gpu.as_ref(), stream)?;
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        // ATLAS_DUMP_HIDDEN: flush captured layer hiddens to file. See
        // verify_b.rs for the eager-mode safety contract. On the DFlash
        // path `try_dflash_capture_rows` is already called per layer above,
        // so by Phase 3 `dflash_hidden_save` holds k tokens' worth of
        // `(n_capture, hidden)` records ready to flush.
        self.flush_hidden_dump(k)?;

        let mut out = if rejection_tail_active {
            let out_ptr = self
                .target_rejection_verify_buf
                .offset(super::device_topk::REJECTION_OUT_OFFSET);
            let bytes = k * 4;
            let words = {
                let mut stg = self.pinned_staging.lock();
                if let Some(offset) = stg.reserve_h2d_range(bytes) {
                    let dst = &mut stg.buffer.as_mut_slice()[offset..offset + bytes];
                    self.gpu.copy_d2h_on_stream(out_ptr, dst, stream)?;
                    super::device_topk::decode_rejection_verify_tokens(
                        dst,
                        k,
                        self.config.vocab_size,
                    )?
                } else {
                    drop(stg);
                    let mut buf = [0u8; super::device_topk::REJECTION_MAX_ROWS * 4];
                    let d2h_slice = &mut buf[..bytes];
                    self.gpu.copy_d2h_on_stream(out_ptr, d2h_slice, stream)?;
                    super::device_topk::decode_rejection_verify_tokens(
                        d2h_slice,
                        k,
                        self.config.vocab_size,
                    )?
                }
            };
            if let Some(ref reqs) = staged_rejection_reqs {
                for (slot, (req, &word)) in reqs.iter().zip(words.iter()).enumerate() {
                    anyhow::ensure!(
                        req.temperature < 0.0 || !word.is_host_fallback(),
                        "target_rejection_verify_rows_bf16 returned host fallback sentinel for active row slot {slot}"
                    );
                }
            }
            if publish_in_graph_rejection {
                *self.in_graph_rejection_verify_words.lock() = Some(words.clone());
                static C1_IN_GRAPH_REJECTION_ENGAGED: std::sync::Once = std::sync::Once::new();
                C1_IN_GRAPH_REJECTION_ENGAGED.call_once(|| {
                    tracing::info!(
                        k,
                        coalesced_lm_head = c1_coalesced_lm_head,
                        use_graphs,
                        "QWEN4_C1_VERIFY_IN_GRAPH_REJECTION_ENGAGED"
                    );
                });
            }
            words
                .iter()
                .map(|w| {
                    if w.is_host_fallback() {
                        u32::MAX
                    } else {
                        w.token_id()
                    }
                })
                .collect()
        } else if gumbel_tail_active {
            let out_ptr = self
                .target_gumbel_verify_buf
                .offset(super::device_topk::GUMBEL_OUT_OFFSET);
            let bytes = k * 4;
            let tokens = {
                let mut stg = self.pinned_staging.lock();
                if let Some(offset) = stg.reserve_h2d_range(bytes) {
                    let dst = &mut stg.buffer.as_mut_slice()[offset..offset + bytes];
                    self.gpu.copy_d2h_on_stream(out_ptr, dst, stream)?;
                    super::device_topk::decode_gumbel_verify_tokens(dst, k, self.config.vocab_size)?
                } else {
                    drop(stg);
                    let mut buf =
                        [0u8; spark_runtime::sampler::device_topk::GUMBEL_VERIFY_MAX_ROWS * 4];
                    let d2h_slice = &mut buf[..bytes];
                    self.gpu.copy_d2h_on_stream(out_ptr, d2h_slice, stream)?;
                    super::device_topk::decode_gumbel_verify_tokens(
                        d2h_slice,
                        k,
                        self.config.vocab_size,
                    )?
                }
            };
            if publish_in_graph_gumbel {
                *self.in_graph_gumbel_verify_tokens.lock() = Some(tokens.clone());
                static C1_IN_GRAPH_GUMBEL_ENGAGED: std::sync::Once = std::sync::Once::new();
                C1_IN_GRAPH_GUMBEL_ENGAGED.call_once(|| {
                    tracing::info!(
                        k,
                        coalesced_lm_head = c1_coalesced_lm_head,
                        use_graphs,
                        "QWEN4_C1_VERIFY_IN_GRAPH_GUMBEL_ENGAGED"
                    );
                });
            }
            tokens
        } else {
            let out_ptr = self.buffers.scratch();
            let bytes = k * 4;
            let mut out = Vec::with_capacity(k);
            let mut stg = self.pinned_staging.lock();
            if let Some(offset) = stg.reserve_h2d_range(bytes) {
                let dst = &mut stg.buffer.as_mut_slice()[offset..offset + bytes];
                self.gpu.copy_d2h_on_stream(out_ptr, dst, stream)?;
                for chunk in dst.chunks_exact(4) {
                    out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            } else {
                drop(stg);
                let mut buf = vec![0u8; bytes];
                self.gpu.copy_d2h_on_stream(out_ptr, &mut buf, stream)?;
                for chunk in buf.chunks_exact(4) {
                    out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            out
        };

        // ATLAS_LAYER_RESOLUTION_PROBE: for each captured snapshot, run
        // norm+lm_head+argmax on GPU and compare to final argmax.  Accumulate
        // match rates into LRP_ACCUM, dump every lrp_dump_every() steps.
        // Post-verify callers still read `hidden` (`save_hidden_for_mtp`) and
        // `buffers.logits()` (host sampling/verification), so snapshot both
        // arenas before the probe loop and restore them unconditionally after.
        if layer_resolution_probe_enabled() && !lrp_snapshots.is_empty() {
            let normed = self.buffers.norm_output();
            let vocab = self.verify_lmhead_vocab() as usize;
            let argmax_out = self.buffers.scratch();
            let hidden_bytes = k * persistent_width * fp32;
            let logits_bytes = k * vocab * bf16;
            self.gpu.synchronize(stream)?;
            let mut saved_hidden = vec![0u8; hidden_bytes];
            self.gpu.copy_d2h(hidden, &mut saved_hidden)?;
            let mut saved_logits = vec![0u8; logits_bytes];
            self.gpu
                .copy_d2h(self.buffers.logits(), &mut saved_logits)?;
            let probe_res: Result<Vec<Vec<u32>>> = (|| {
                let mut probe_ams: Vec<Vec<u32>> = Vec::with_capacity(lrp_snapshots.len());
                for snap in &lrp_snapshots {
                    self.gpu.copy_h2d(snap, hidden)?;
                    self.final_norm_rows(hidden, normed, k as u32, stream)?;
                    self.lm_head_batched(normed, k as u32, stream)?;
                    for t in 0..k {
                        let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                        let out_t = argmax_out.offset(t * 4);
                        ops::argmax_bf16(
                            self.gpu.as_ref(),
                            self.argmax_kernel,
                            logits_t,
                            out_t,
                            vocab as u32,
                            stream,
                        )?;
                    }
                    self.gpu.synchronize(stream)?;
                    let mut pbuf = vec![0u8; k * 4];
                    self.gpu.copy_d2h(argmax_out, &mut pbuf)?;
                    let pam: Vec<u32> = (0..k)
                        .map(|t| {
                            let o = t * 4;
                            u32::from_le_bytes([pbuf[o], pbuf[o + 1], pbuf[o + 2], pbuf[o + 3]])
                        })
                        .collect();
                    probe_ams.push(pam);
                }
                Ok(probe_ams)
            })();
            let restore_hidden = self.gpu.copy_h2d(&saved_hidden, hidden);
            let restore_logits = self.gpu.copy_h2d(&saved_logits, self.buffers.logits());
            let probe_ams = probe_res?;
            restore_hidden?;
            restore_logits?;
            let accum = LRP_ACCUM.get_or_init(|| Mutex::new(LrpAccum::new(k)));
            let mut g = accum.lock();
            g.accumulate(&out, &probe_ams);
            if g.step.is_multiple_of(lrp_dump_every()) {
                g.dump(self.layers.len());
            }
        }

        // DFS reorder: un-permute verified outputs back to original-compact
        // frame so the caller (verify_dflash_step.rs::greedy_sample_ddtree)
        // sees outputs indexed by the ORIGINAL compact (kernel) frame.
        //
        // `out[i]` (DFS frame) is the argmax-after-`tokens[dfs_perm[i]]`.
        // We want `out_orig[j]` = argmax-after-`tokens[j]` = `out[dfs_inv_perm[j]]`.
        //
        // Un-permute is only needed when token permutation was actually
        // applied (METADATA_ONLY=0). With METADATA_ONLY=1, tokens stayed
        // in linear order so out[i] already corresponds to drafts[i-1]'s
        // next-token prediction in the same kernel slot.
        if dfs_active && dfs_permute_tokens {
            let mut out_orig = vec![0u32; k];
            for j in 0..k {
                out_orig[j] = out[dfs_inv_perm[j]];
            }
            out = out_orig;
        }

        // Qualification-only logits receipt, identical in shape to the
        // cross-sequence frame's (verify_qwen4_batched.rs).
        if receipt_on {
            let vocab = self.verify_lmhead_vocab() as usize;
            let mut logits = vec![0u8; k * vocab * bf16];
            self.gpu.copy_d2h(self.buffers.logits(), &mut logits)?;
            tracing::info!(
                "{}",
                crate::model::verify_logits_receipt::line("c1", &seq.tokens, tokens, &out, &logits)
            );
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }

    /// FIX 2 — KV compaction for sparse (tree-fork) accepts.
    ///
    /// During K=γ verify, compact slot `t` writes its attention K/V to the
    /// linear KV position `pre_verify_len + t` (see the `slots` build above).
    /// When the target's greedy walk crosses a sibling fork, the accepted
    /// path's compact indices are NON-contiguous (e.g. `[1, 2, 3, 7]`), so the
    /// accepted tokens' KV is scattered across positions `pre+1, pre+2, pre+3,
    /// pre+7`. The next decode reads a CONTIGUOUS sequence `pre+1 .. pre+N`, so
    /// the scattered entries must be GATHERED down to the contiguous run that
    /// mirrors how the flat chain lays KV.
    ///
    /// `accepted_compact` is `greedy_sample_ddtree_full`'s path (compact
    /// indices, in walk order). `pre_verify_len` is the sequence length BEFORE
    /// this verify advanced it. For each accepted step `j` (1-based), the
    /// token lives at compact index `accepted_compact[j-1]` (KV pos
    /// `pre + that`); its contiguous home is `pre + j`. A copy fires only when
    /// source != destination (a flat prefix is already in place → no-op).
    ///
    /// LOSSLESS: this only relocates already-committed K/V bytes; it never
    /// changes a value. Bonus row needs no KV (it is fed as verify input next
    /// step). Supported KV dtypes: bf16 / fp8 (contiguous per-position element
    /// layout). Quantized cache dtypes (NVFP4/Turbo*) have split data+scale
    /// sections and are refused so the feature never silently corrupts.
    pub(super) fn compact_verify_kv_dispatch(
        &self,
        seq: &SequenceState,
        accepted_compact: &[usize],
        pre_verify_len: usize,
    ) -> Result<()> {
        // TASK #29 root-cause fix. Two things this must get right:
        //
        //   1. FRAME. The accepted path is in the COMPACT frame, but the KV was
        //      written by the kernel in the KERNEL (DFS-reordered) frame. When
        //      ATLAS_DDTREE_DFS_REORDER=1 (the portfolio / deep-DDTree commit
        //      config), compact index `c`'s KV lives at KV position
        //      `pre + dfs_inv_perm[c]`, NOT `pre + c`. The pre-fix code gathered
        //      from `pre + compact` → wrong bytes → counting md5 corruption on a
        //      forked accept. We map through the SAME `ddtree_dfs_inv_perm` the
        //      SSM commit uses (`commit_verify_state_async_with_slot`).
        //   2. OVERLAP. Once sources are permuted kernel slots, a source may be
        //      another move's destination; a naive in-place copy clobbers data a
        //      later move still needs. `plan_kv_compaction_moves` evacuates such
        //      conflicts through free tail verify slots first, so the whole
        //      relocation is a lossless permutation.
        //
        // Empty `inv_perm` (chain mode / DFS off) ⇒ identity map ⇒ this reduces
        // to the legacy ascending-dst plan (and is a no-op for a contiguous
        // path), so the flat path is byte-for-byte unchanged.
        use crate::layers::dflash_head::ddtree::plan_kv_compaction_moves;
        let inv_perm = self.ddtree_dfs_inv_perm.lock().clone();
        // Verify width (kernel-slot count). Under DFS reorder inv_perm has
        // exactly k entries; otherwise the general (scratch) path is never
        // taken, so any bound >= accepted_compact.len()+1 is safe. Use the
        // accepted-path max compact index as a floor so tail scratch exists.
        let k = if !inv_perm.is_empty() {
            inv_perm.len()
        } else {
            accepted_compact.iter().copied().max().unwrap_or(0) + 1
        };
        let planned = plan_kv_compaction_moves(accepted_compact, &inv_perm, pre_verify_len, k);
        let moves: Vec<(usize, usize)> = planned.iter().map(|m| (m.src_pos, m.dst_pos)).collect();
        if moves.is_empty() {
            return Ok(());
        }

        let stream = self.gpu.default_stream();
        let kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let num_layers = kv_cache.num_layers();

        for layer_idx in 0..num_layers {
            let dtype = kv_cache.dtype_for_layer(layer_idx);
            let (nkv, hd) = kv_cache.config().dims_for_layer(layer_idx);
            let elem_per_pos = nkv * hd;
            // Per-position byte stride for the contiguous element dtypes.
            let pos_bytes = match dtype {
                spark_runtime::kv_cache::KvCacheDtype::Bf16 => elem_per_pos * 2,
                spark_runtime::kv_cache::KvCacheDtype::Fp8 => elem_per_pos,
                other => {
                    bail!(
                        "compact_verify_kv: unsupported KV dtype {other:?} for layer {layer_idx} \
                         (tree-fork KV compaction only supports contiguous bf16/fp8); \
                         disable ATLAS_DFLASH_TREE_COMMIT or use --kv-cache-dtype fp8"
                    );
                }
            };

            for &(src_pos, dst_pos) in &moves {
                let src_block = seq
                    .physical_block_for(src_pos / bs)
                    .ok_or_else(|| anyhow::anyhow!("compact_verify_kv: src block evicted"))?;
                let dst_block = seq
                    .physical_block_for(dst_pos / bs)
                    .ok_or_else(|| anyhow::anyhow!("compact_verify_kv: dst block evicted"))?;
                let src_off = (src_pos % bs) * pos_bytes;
                let dst_off = (dst_pos % bs) * pos_bytes;

                let k_src = kv_cache.k_cache_ptr(layer_idx, src_block).offset(src_off);
                let k_dst = kv_cache.k_cache_ptr(layer_idx, dst_block).offset(dst_off);
                let v_src = kv_cache.v_cache_ptr(layer_idx, src_block).offset(src_off);
                let v_dst = kv_cache.v_cache_ptr(layer_idx, dst_block).offset(dst_off);

                self.gpu.copy_d2d_async(k_src, k_dst, pos_bytes, stream)?;
                self.gpu.copy_d2d_async(v_src, v_dst, pos_bytes, stream)?;
            }
        }
        drop(kv_cache);

        static COMPACT_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COMPACT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 8 {
            tracing::info!(
                "compact_verify_kv #{n}: pre_len={pre_verify_len} accepted_compact={:?} moves={:?}",
                accepted_compact,
                moves,
            );
        }
        Ok(())
    }
}

impl TransformerModel {
    fn lookup_verify_kgamma_graph(&self, key: &VerifyKgammaGraphKey) -> Option<GraphHandle> {
        let graph = {
            let cache = self.verify_kgamma_graph.lock();
            cache.get(key).copied().filter(|g| g.0 != 0)
        };
        if graph.is_some() {
            self.verify_kgamma_lru.lock().touch(*key);
        }
        graph
    }

    /// Evict stale QSA-regime graphs for `new_key.0` and enforce the bounded
    /// per-tier LRU cap (tightening under UMA memory pressure) strictly before
    /// `CaptureGuard::begin` instantiates a new CUDA graph.
    ///
    /// Synchronizes `stream` before calling `destroy_graph` whenever any graph
    /// is evicted so no in-flight graph execution on `stream` is destroyed.
    fn prepare_verify_kgamma_graph_slot(
        &self,
        new_key: VerifyKgammaGraphKey,
        preserve_other_tier_key: Option<VerifyKgammaGraphKey>,
        stream: u64,
    ) -> Result<()> {
        let floor_bytes = verify_kgamma_mem_floor_bytes();
        let mem_available_bytes =
            floor_bytes.and_then(|_| spark_runtime::startup_ledger::mem_available_bytes());
        let effective_cap = effective_verify_kgamma_graph_cap(
            verify_kgamma_graph_max_entries(),
            mem_available_bytes,
            floor_bytes,
        );

        let (evicted_handles, evicted_qsa_regime, evicted_lru, remaining_suffix, remaining_layer0) = {
            let mut cache = self.verify_kgamma_graph.lock();
            let mut lru = self.verify_kgamma_lru.lock();
            lru.retain_live(&cache);

            let plan = plan_verify_kgamma_evictions(
                &cache,
                &lru,
                new_key,
                preserve_other_tier_key,
                effective_cap,
            );
            if plan.is_empty() {
                return Ok(());
            }

            let evicted_qsa_regime = plan.qsa_regime_keys.len();
            let evicted_lru = plan.lru_keys.len();
            let mut handles = Vec::with_capacity(evicted_qsa_regime + evicted_lru);
            for key in plan.all_keys() {
                lru.remove(&key);
                if let Some(graph) = cache.remove(&key)
                    && graph.0 != 0
                {
                    handles.push((key, graph));
                }
            }
            let remaining_layer0 = cache
                .keys()
                .filter(|k| verify_kgamma_is_layer0_prefix_key(k))
                .count();
            let remaining_suffix = cache.len().saturating_sub(remaining_layer0);
            (
                handles,
                evicted_qsa_regime,
                evicted_lru,
                remaining_suffix,
                remaining_layer0,
            )
        };

        if !evicted_handles.is_empty() {
            self.gpu.synchronize(stream)?;
            for (evicted_key, old_graph) in evicted_handles {
                if let Err(err) = self.gpu.destroy_graph(old_graph) {
                    tracing::error!(
                        "verify_d: destroy_graph(verify_kgamma_graph[{evicted_key:?}], {}) failed: {err:#}",
                        old_graph.0
                    );
                }
            }
            tracing::info!(
                slot = new_key.0,
                k = new_key.1,
                pack_key = new_key.2,
                layer0_prefix = verify_kgamma_is_layer0_prefix_key(&new_key),
                evicted_qsa_regime,
                evicted_lru,
                effective_cap,
                remaining_suffix,
                remaining_layer0,
                mem_available_mib =
                    ?mem_available_bytes.map(|b| b / spark_runtime::startup_ledger::MIB),
                "VERIFY_KGAMMA_GRAPH_EVICTED"
            );
        }
        Ok(())
    }

    fn store_verify_kgamma_graph(
        &self,
        key: VerifyKgammaGraphKey,
        graph: GraphHandle,
        stream: u64,
    ) -> Result<()> {
        let replaced = {
            let mut cache = self.verify_kgamma_graph.lock();
            let mut lru = self.verify_kgamma_lru.lock();
            let prev = cache.insert(key, graph);
            lru.touch(key);
            prev
        };
        if let Some(old_graph) = replaced
            && old_graph.0 != 0
            && old_graph.0 != graph.0
        {
            self.gpu.synchronize(stream)?;
            if let Err(err) = self.gpu.destroy_graph(old_graph) {
                tracing::error!(
                    "verify_d: destroy_graph(replaced verify_kgamma_graph[{key:?}], {}) failed: {err:#}",
                    old_graph.0
                );
            }
        }
        Ok(())
    }

    /// One Qwen4 target layer of the K-row verify: row-serial oracle or the
    /// qualified batched route, with per-row recurrent intermediates.
    #[allow(clippy::too_many_arguments)]
    fn qwen4_verify_layer(
        &self,
        layer_idx: usize,
        layer: &dyn TransformerLayer,
        qwen4_ssm_layer_idx: usize,
        qwen4_layer_route: Option<Qwen4VerifyLayerRoute>,
        k: usize,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        mb: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        row_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let layer_type = self.config.layer_type(layer_idx);
        let (h_inter, conv_inter) = if layer_type == LayerType::LinearAttention {
            (
                self.ssm_pool
                    .h_intermediate(qwen4_ssm_layer_idx, seq.slot_idx, 0),
                self.ssm_pool
                    .conv_intermediate(qwen4_ssm_layer_idx, seq.slot_idx, 0),
            )
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        // A batched route with no proof for this exact physical K
        // can corrupt recurrent state even when its logits happen
        // to agree. Keep gamma15/K16 and shortened, unqualified
        // widths on ordinary row-serial decode. K=2/K=3, the
        // separately qualified K=5 hybrid, and the explicit
        // research-only K16 selector may enter
        // `decode_qwen4_batched`.
        let verify_route = qwen4_layer_route
            .ok_or_else(|| anyhow::anyhow!("Qwen4 verify layer route was not resolved (K={k})"))?;
        if verify_route == Qwen4VerifyLayerRoute::RowSerialOracle {
            anyhow::ensure!(
                !self.ssm_pool.compact_lazy_h,
                "Qwen4 row-serial verify (K={k}) requires per-layer H intermediates; \
                 set ATLAS_QWEN4_HYBRID_ALL_K=1 or ATLAS_QWEN4_REPLAY_SSM=0"
            );
            let metadata = ctx.attn_metadata.ok_or_else(|| {
                anyhow::anyhow!("Qwen4 row-serial verify requires attention metadata (K={k})")
            })?;
            let qsa_journal =
                metadata.qwen4_qsa_required && crate::layers::qwen4_qsa::verify_journal_rows(k);
            if qsa_journal {
                layer.prepare_qwen4_qsa_verify(kv_cache, self.gpu.as_ref())?;
            }
            for row in 0..k {
                let token_metadata = AttnMetadataDev {
                    positions: metadata.positions.offset(row * 4),
                    positions_h: metadata.positions_h.offset(row * 4),
                    positions_w: metadata.positions_w.offset(row * 4),
                    slot: metadata.slot.offset(row * 8),
                    seq_len: metadata.seq_len.offset(row * 4),
                    block_table: metadata.block_table.offset(row * mb * 4),
                    num_seqs: 1,
                    ..metadata
                };
                let token_ctx = ForwardContext {
                    attn_metadata: Some(token_metadata),
                    ..*ctx
                };
                if qsa_journal {
                    // The journal stores undo data, so snapshot before decode
                    // can overwrite the raw ring or a compressed key.
                    layer.snapshot_qwen4_qsa_verify_row(
                        kv_cache,
                        token_metadata.slot,
                        row,
                        self.gpu.as_ref(),
                        stream,
                    )?;
                }
                layer.decode_qwen4_verified_row(
                    hidden.offset(row * row_stride),
                    residual.offset(row * row_stride),
                    seq.layer_states[layer_idx].as_mut(),
                    kv_cache,
                    seq.seq_len + row,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &token_ctx,
                    stream,
                )?;
                if layer_type == LayerType::LinearAttention {
                    let ssm = seq.layer_states[layer_idx]
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Qwen4 row-serial verify expected SSM state at layer \
                                 {layer_idx} (K={k})"
                            )
                        })?;
                    self.gpu.copy_d2d_async(
                        ssm.h_state,
                        h_inter.offset(row * self.ssm_pool.h_bytes),
                        self.ssm_pool.h_bytes,
                        stream,
                    )?;
                    self.gpu.copy_d2d_async(
                        ssm.conv_state,
                        conv_inter.offset(row * self.ssm_pool.conv_bytes),
                        self.ssm_pool.conv_bytes,
                        stream,
                    )?;
                }
            }
        } else {
            layer.decode_qwen4_batched(
                hidden,
                residual,
                k,
                seq.layer_states[layer_idx].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                h_inter,
                conv_inter,
                self.ssm_pool.h_intermediate_stride(),
                self.ssm_pool.conv_bytes,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Qwen4 sparse PLE rows consuming a pre-submitted warm ticket overlapped
    /// with Layer 0 when available, falling back cleanly to [`Self::qwen4_verify_ple`].
    fn qwen4_verify_ple_with_ticket(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        ple_warm_ticket: Option<crate::layers::qwen4_ple::PleWarmTicket<'_>>,
        hidden: DevicePtr,
        row_stride: usize,
        stream: u64,
    ) -> Result<()> {
        if let Some(ticket) = ple_warm_ticket
            && let Some(ple) = self.qwen4_ple.as_ref()
            && crate::layers::qwen4_ple::ple_verify_batch_selected()
        {
            let pre_read = Some(ticket.wait()?);
            return ple.forward_cross_seq_verify(
                &[(tokens, seq.tokens.as_slice(), seq.slot_idx)],
                pre_read,
                hidden,
                row_stride,
                self.gpu.as_ref(),
                stream,
            );
        }
        self.qwen4_verify_ple(tokens, seq, hidden, row_stride, stream)
    }

    /// Qwen4 sparse PLE rows for every verify row. Host/NVMe row I/O, so it
    /// runs eagerly and never inside a captured graph.
    pub(super) fn qwen4_verify_ple(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        hidden: DevicePtr,
        row_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(ple) = &self.qwen4_ple else {
            return Ok(());
        };
        // ATLAS_QWEN4_PLE_VERIFY_BATCH=1: one PLE read + one pinned upload for
        // all rows, then the identical per-row kernels (fn-mtp).
        if crate::layers::qwen4_ple::ple_verify_batch_selected() {
            return ple.forward_tokens_verify(
                tokens,
                &seq.tokens,
                hidden,
                row_stride,
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
            );
        }
        for (row, &token) in tokens.iter().enumerate() {
            let mut prior = seq.tokens.clone();
            prior.extend_from_slice(&tokens[..row]);
            ple.forward_token(
                token,
                &prior,
                hidden.offset(row * row_stride),
                seq.slot_idx,
                false,
                self.gpu.as_ref(),
                stream,
            )?;
            ple.save_intermediate(seq.slot_idx, row, self.gpu.as_ref(), stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod qwen4_verify_route_tests {
    use super::{
        QWEN4_K16_BATCHED_VERIFY_ENV, QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
        QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, Qwen4VerifyLayerRoute, Qwen4VerifyRouteIdentity,
        is_native_qwen4_dflash_pairing, parse_exact_bool_env_value, qwen4_legacy_k2_admission,
        qwen4_qsa_dense_prefix_graph_eligible, qwen4_qsa_graph_route_safe,
        qwen4_qsa_verify_frame_admission, qwen4_qsa_verify_frame_required,
        qwen4_verify_layer_route,
    };

    const NATIVE_K16: Qwen4VerifyRouteIdentity = Qwen4VerifyRouteIdentity {
        target_is_qwen4: true,
        native_dflash_pairing: true,
        active_proposer_is_dflash: true,
        active_proposer_physical_k: Some(16),
    };

    fn route(
        frame_k: usize,
        k5_hybrid_enabled: bool,
        k16_batched_requested: bool,
    ) -> Qwen4VerifyLayerRoute {
        qwen4_verify_layer_route(
            NATIVE_K16,
            frame_k,
            k5_hybrid_enabled,
            false,
            false,
            k16_batched_requested,
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn qsa_graph_admission_requires_a_topology_safe_route_for_every_width() {
        assert!(!qwen4_qsa_graph_route_safe(
            true, 8_192, 7, true, false, false
        ));
        assert!(qwen4_qsa_graph_route_safe(true, 0, 5, false, true, false));
        assert!(qwen4_qsa_graph_route_safe(
            true, 2_048, 7, true, false, true
        ));
        assert!(qwen4_qsa_graph_route_safe(true, 8_192, 7, true, true, true));

        // Pure-dense prefix frames (`seq_len + rows <= 2048`) with fused QSA
        // staging have constant launch topology and admit CUDA graphs.
        for rows in [2usize, 5, 7, 8, 9] {
            assert!(
                qwen4_qsa_graph_route_safe(true, 0, rows, true, false, false),
                "rows={rows} seq_len=0 must admit dense-prefix graph"
            );
            assert!(
                qwen4_qsa_graph_route_safe(true, 2_048 - rows, rows, true, false, false),
                "rows={rows} endpoint=2048 must admit dense-prefix graph"
            );
            // Frames straddling the 2048 dense->sparse boundary stay eager.
            assert!(
                !qwen4_qsa_graph_route_safe(true, 2_048 - rows + 1, rows, true, false, true),
                "rows={rows} straddling 2048 must stay eager"
            );
            // Without fused QSA staging, pre-2048 QSA stays eager.
            assert!(
                !qwen4_qsa_graph_route_safe(true, 0, rows, false, false, false),
                "rows={rows} without fused QSA staging must stay eager"
            );
        }
        // Single-row QSA decode (`rows == 1`) does not use batched verify QSA staging.
        assert!(!qwen4_qsa_graph_route_safe(true, 0, 1, true, false, false));

        // A frame starting at 2047 straddles 2048 for any multi-row width.
        assert!(!qwen4_qsa_graph_route_safe(
            true, 2_047, 5, true, false, true
        ));
        assert!(!qwen4_qsa_graph_route_safe(
            true, 2_047, 5, false, false, true
        ));

        // Non-QSA requests always admit graphs.
        assert!(qwen4_qsa_graph_route_safe(false, 0, 7, false, false, false));

        // Dense-prefix graph cache bit must never alias existing bits or post-2048 sparse graphs.
        assert_eq!(
            QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT & (0b111 | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT),
            0
        );
        assert!(qwen4_qsa_dense_prefix_graph_eligible(true, 2_041, 7, true));
        assert!(!qwen4_qsa_dense_prefix_graph_eligible(true, 2_042, 7, true));
        assert!(!qwen4_qsa_dense_prefix_graph_eligible(true, 2_048, 7, true));
        assert!(!qwen4_qsa_dense_prefix_graph_eligible(false, 0, 7, true));
    }

    #[test]
    fn qsa_transaction_accepts_only_journaled_flat_frames() {
        assert_eq!(qwen4_qsa_verify_frame_admission(true, false), Ok(()));
        assert!(qwen4_qsa_verify_frame_admission(false, false).is_err());
        assert!(qwen4_qsa_verify_frame_admission(true, true).is_err());
        assert!(qwen4_qsa_verify_frame_admission(false, true).is_err());
    }

    #[test]
    fn qsa_transaction_scope_leaves_nonqwen_nonqsa_and_single_rows_unchanged() {
        for physical_k in [0, 1, 2, 3, 4, 5, 9, 10, 16, 128, usize::MAX] {
            assert!(!qwen4_qsa_verify_frame_required(false, false, physical_k));
            assert!(!qwen4_qsa_verify_frame_required(false, true, physical_k));
            assert!(!qwen4_qsa_verify_frame_required(true, false, physical_k));
            assert_eq!(
                qwen4_qsa_verify_frame_required(true, true, physical_k),
                physical_k > 1,
            );
        }
    }

    #[test]
    fn qsa_gamma15_and_wider_frames_have_no_undo_journal_admission() {
        for physical_k in [10, 11, 15, 16, 17, 32, 128, usize::MAX] {
            let journal_available = crate::layers::qwen4_qsa::verify_journal_rows(physical_k);
            assert!(!journal_available, "unexpected journal width {physical_k}");
            assert!(qwen4_qsa_verify_frame_admission(journal_available, false).is_err());
            assert!(qwen4_qsa_verify_frame_admission(journal_available, true).is_err());
        }
    }

    #[test]
    fn qsa_k5_still_requires_flat_frames_when_always_journaled() {
        let journal_available = crate::layers::qwen4_qsa::verify_journal_rows(5);
        assert!(journal_available);
        assert_eq!(
            qwen4_qsa_verify_frame_admission(journal_available, false),
            Ok(()),
        );
        assert!(qwen4_qsa_verify_frame_admission(journal_available, true).is_err());
    }

    #[test]
    fn qsa_shortened_and_wider_journal_frames_all_refuse_tree_metadata() {
        for physical_k in 2..=9 {
            assert!(
                qwen4_qsa_verify_frame_admission(true, true).is_err(),
                "K={physical_k}",
            );
            assert!(
                qwen4_qsa_verify_frame_admission(false, true).is_err(),
                "K={physical_k}",
            );
        }
    }

    #[test]
    fn qsa_disabled_journal_never_admits_a_rejectable_suffix() {
        for physical_k in [2, 3, 4, 6, 7, 8, 9] {
            assert!(
                qwen4_qsa_verify_frame_admission(false, false).is_err(),
                "K={physical_k}",
            );
        }
    }

    #[test]
    fn qwen4_fixed_k2_refuses_the_recursive_legacy_selector() {
        assert_eq!(qwen4_legacy_k2_admission(true, 2, false), Ok(()));
        assert!(qwen4_legacy_k2_admission(true, 2, true).is_err());
    }

    #[test]
    fn fixed_k2_legacy_selector_leaves_other_targets_and_widths_unchanged() {
        for rows in [0, 1, 2, 3, 4, 5, 9, 16, 128, usize::MAX] {
            for selected in [false, true] {
                assert_eq!(qwen4_legacy_k2_admission(false, rows, selected), Ok(()));
                if rows != 2 {
                    assert_eq!(qwen4_legacy_k2_admission(true, rows, selected), Ok(()));
                }
            }
        }
    }

    #[test]
    fn gamma15_physical_k16_defaults_to_row_serial_oracle() {
        let gamma = 15usize;
        let physical_k = gamma + 1;

        assert_eq!(physical_k, 16);
        assert_eq!(
            route(physical_k, false, false),
            Qwen4VerifyLayerRoute::RowSerialOracle
        );
        assert_eq!(
            route(physical_k, true, false),
            Qwen4VerifyLayerRoute::RowSerialOracle,
            "the K5 opt-in must not widen to K16"
        );
    }

    #[test]
    fn exact_native_gamma15_k16_can_select_experimental_batching() {
        assert_eq!(
            route(16, false, true),
            Qwen4VerifyLayerRoute::ExperimentalK16Batched
        );
        assert_eq!(
            QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT & 0b111,
            0,
            "the K16 route bit must not alias existing graph-shape bits"
        );
    }

    #[test]
    fn shortened_width_dispatch_table_does_not_leak_k16_opt_in() {
        let expected = [
            (1, Qwen4VerifyLayerRoute::RowSerialOracle),
            (2, Qwen4VerifyLayerRoute::QualifiedBatched),
            (3, Qwen4VerifyLayerRoute::QualifiedBatched),
            (4, Qwen4VerifyLayerRoute::RowSerialOracle),
            (5, Qwen4VerifyLayerRoute::RowSerialOracle),
            (6, Qwen4VerifyLayerRoute::RowSerialOracle),
            (7, Qwen4VerifyLayerRoute::RowSerialOracle),
            (8, Qwen4VerifyLayerRoute::RowSerialOracle),
            (9, Qwen4VerifyLayerRoute::RowSerialOracle),
            (10, Qwen4VerifyLayerRoute::RowSerialOracle),
            (11, Qwen4VerifyLayerRoute::RowSerialOracle),
            (12, Qwen4VerifyLayerRoute::RowSerialOracle),
            (13, Qwen4VerifyLayerRoute::RowSerialOracle),
            (14, Qwen4VerifyLayerRoute::RowSerialOracle),
            (15, Qwen4VerifyLayerRoute::RowSerialOracle),
        ];

        for (physical_k, route) in expected {
            assert_eq!(
                self::route(physical_k, false, false),
                route,
                "unexpected Qwen4 route for shortened physical K={physical_k}"
            );
            for k5_hybrid_enabled in [false, true] {
                assert_eq!(
                    self::route(physical_k, k5_hybrid_enabled, true),
                    self::route(physical_k, k5_hybrid_enabled, false),
                    "K16 opt-in leaked to shortened frame K={physical_k} with K5 hybrid={k5_hybrid_enabled}"
                );
            }
        }
    }

    #[test]
    fn k5_hybrid_opt_in_is_shape_exact() {
        assert_eq!(
            route(5, true, false),
            Qwen4VerifyLayerRoute::QualifiedBatched
        );
        for physical_k in [1, 4, 6, 15, 16, 17] {
            assert_eq!(
                route(physical_k, true, false),
                Qwen4VerifyLayerRoute::RowSerialOracle,
                "K5 opt-in leaked to physical K={physical_k}"
            );
        }
    }

    #[test]
    fn k4_hybrid_needs_both_opt_ins_and_is_shape_exact() {
        let k4 = |frame_k, k5, k4| {
            qwen4_verify_layer_route(NATIVE_K16, frame_k, k5, k4, false, false)
                .unwrap()
                .unwrap()
        };
        assert_eq!(k4(4, true, true), Qwen4VerifyLayerRoute::QualifiedBatched);
        assert_eq!(k4(4, true, false), Qwen4VerifyLayerRoute::RowSerialOracle);
        assert_eq!(k4(4, false, true), Qwen4VerifyLayerRoute::RowSerialOracle);
        for physical_k in [1, 6, 7, 8, 15, 16, 17] {
            assert_eq!(
                k4(physical_k, true, true),
                Qwen4VerifyLayerRoute::RowSerialOracle,
                "K4 opt-in leaked to physical K={physical_k}"
            );
        }
    }

    #[test]
    fn all_k_hybrid_admits_two_through_nine_only_with_k5() {
        let r = |frame_k, k5, all_k| {
            qwen4_verify_layer_route(NATIVE_K16, frame_k, k5, false, all_k, false)
                .unwrap()
                .unwrap()
        };
        for frame_k in 2..=9 {
            assert_eq!(
                r(frame_k, true, true),
                Qwen4VerifyLayerRoute::QualifiedBatched,
                "K={frame_k}"
            );
        }
        for frame_k in [1, 10, 15, 16, 17] {
            assert_eq!(
                r(frame_k, true, true),
                Qwen4VerifyLayerRoute::RowSerialOracle,
                "all-K leaked to K={frame_k}"
            );
        }
        for frame_k in [4, 6, 7, 8] {
            assert_eq!(
                r(frame_k, false, true),
                Qwen4VerifyLayerRoute::RowSerialOracle
            );
            assert_eq!(
                r(frame_k, true, false),
                Qwen4VerifyLayerRoute::RowSerialOracle
            );
        }
    }

    #[test]
    fn k16_request_rejects_every_wrong_runtime_identity() {
        let invalid = [
            Qwen4VerifyRouteIdentity {
                target_is_qwen4: false,
                ..NATIVE_K16
            },
            Qwen4VerifyRouteIdentity {
                native_dflash_pairing: false,
                ..NATIVE_K16
            },
            Qwen4VerifyRouteIdentity {
                active_proposer_is_dflash: false,
                ..NATIVE_K16
            },
            Qwen4VerifyRouteIdentity {
                active_proposer_physical_k: None,
                ..NATIVE_K16
            },
            Qwen4VerifyRouteIdentity {
                active_proposer_physical_k: Some(5),
                ..NATIVE_K16
            },
            Qwen4VerifyRouteIdentity {
                active_proposer_physical_k: Some(17),
                ..NATIVE_K16
            },
        ];

        for identity in invalid {
            assert!(
                qwen4_verify_layer_route(identity, 16, false, false, false, true).is_err(),
                "invalid identity unexpectedly admitted: {identity:?}"
            );
        }
    }

    #[test]
    fn disabled_flag_does_not_create_generic_or_dense_routes() {
        let generic = Qwen4VerifyRouteIdentity {
            target_is_qwen4: false,
            native_dflash_pairing: false,
            active_proposer_is_dflash: false,
            active_proposer_physical_k: None,
        };
        for frame_k in 1..=17 {
            assert_eq!(
                qwen4_verify_layer_route(generic, frame_k, false, false, false, false).unwrap(),
                None
            );
        }

        let dense_donor_qwen4 = Qwen4VerifyRouteIdentity {
            target_is_qwen4: true,
            native_dflash_pairing: false,
            active_proposer_is_dflash: true,
            active_proposer_physical_k: Some(16),
        };
        assert_eq!(
            qwen4_verify_layer_route(dense_donor_qwen4, 16, false, false, false, false).unwrap(),
            Some(Qwen4VerifyLayerRoute::RowSerialOracle)
        );
        assert!(
            qwen4_verify_layer_route(dense_donor_qwen4, 16, false, false, false, true).is_err()
        );
    }

    #[test]
    fn k16_flag_parser_is_exact() {
        assert!(!parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, None).unwrap());
        assert!(!parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, Some("0")).unwrap());
        assert!(parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, Some("1")).unwrap());
        for invalid in ["", "true", "01", " 1", "1 ", "2", "-1"] {
            assert!(
                parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, Some(invalid)).is_err(),
                "invalid flag value accepted: {invalid:?}"
            );
        }
    }

    #[test]
    fn native_dflash_pairing_is_profile_driven_for_v3_dspark_and_incoai() {
        use atlas_core::config::{DflashCaptureMode, ModelConfig};

        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "qwen4_exp".to_string();
        cfg.hidden_size = 2_560;
        cfg.hc_count = 4;
        cfg.vocab_size = 248_320;
        cfg.num_hidden_layers = 48;

        // native-v3 / native-dflash2
        assert!(is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[1, 7, 13, 20, 26, 33, 39, 46],
            DflashCaptureMode::Qwen4HyperProjected,
        ));
        // pixelml-qwen3dspark
        assert!(is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));
        // incoai-dflash2-b8
        assert!(is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[5, 15, 25, 35, 45],
            DflashCaptureMode::Qwen4HcMean,
        ));
        // Logical vocab 248077 is equally native.
        cfg.vocab_size = 248_077;
        assert!(is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));

        // Mismatched capture mode / mutated taps / offset / width / donor mode all fail closed.
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4HyperProjected,
        ));
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 44],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::ResidualSlice,
        ));
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            1,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_048,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));
        cfg.model_type = "qwen3_next".to_string();
        assert!(!is_native_qwen4_dflash_pairing(
            &cfg,
            2_560,
            0,
            &[3, 15, 23, 35, 43],
            DflashCaptureMode::Qwen4LayerAttnHc,
        ));
    }

    #[test]
    fn verify_d_meta_scratch_reuses_allocation_and_zeroes_shrunk_tails() {
        let mut scratch = super::VerifyDMetaHostScratch::default();
        scratch.prepare(4, 8);
        let ptr = scratch.block_tables.as_ptr();
        scratch.fill_block_table_rows(4, 8, &[10, 20, 30, 40, 50]);
        assert_eq!(&scratch.block_tables[0..8], &[10, 20, 30, 40, 50, 0, 0, 0]);
        assert_eq!(
            &scratch.block_tables[24..32],
            &[10, 20, 30, 40, 50, 0, 0, 0]
        );

        // Shrink active blocks: same allocation reused, stale tail columns zeroed.
        scratch.prepare(2, 8);
        assert_eq!(scratch.block_tables.as_ptr(), ptr);
        scratch.fill_block_table_rows(2, 8, &[7, 8]);
        assert_eq!(&scratch.block_tables[0..8], &[7, 8, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&scratch.block_tables[8..16], &[7, 8, 0, 0, 0, 0, 0, 0]);

        // Grow back to 4 rows: rows 2..4 also zero prior dirty tails cleanly.
        scratch.prepare(4, 8);
        assert_eq!(scratch.block_tables.as_ptr(), ptr);
        scratch.fill_block_table_rows(4, 8, &[99]);
        assert_eq!(&scratch.block_tables[16..24], &[99, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&scratch.block_tables[24..32], &[99, 0, 0, 0, 0, 0, 0, 0]);

        // Stride change resets all stale entries cleanly.
        scratch.prepare(3, 4);
        assert_eq!(scratch.block_tables.as_ptr(), ptr);
        scratch.fill_block_table_rows(3, 4, &[3, 4]);
        assert_eq!(
            &scratch.block_tables[0..12],
            &[3, 4, 0, 0, 3, 4, 0, 0, 3, 4, 0, 0]
        );
    }

    #[test]
    fn verify_d_in_graph_gumbel_graph_bits_are_disjoint_and_separate_keys() {
        use super::{
            QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT, QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT,
            verify_d_gumbel_tail_graph_bits,
        };
        let prior_bits =
            0b111 | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT;
        assert_eq!(QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(
            QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT & QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT,
            0
        );

        let greedy = verify_d_gumbel_tail_graph_bits(false, false);
        let gumbel_exact_head = verify_d_gumbel_tail_graph_bits(true, false);
        let gumbel_coalesced_head = verify_d_gumbel_tail_graph_bits(true, true);
        assert_eq!(greedy, 0);
        assert_ne!(greedy, gumbel_exact_head);
        assert_ne!(greedy, gumbel_coalesced_head);
        assert_ne!(gumbel_exact_head, gumbel_coalesced_head);
    }

    #[test]
    fn verify_d_layer0_prefix_pack_key_shares_greedy_and_gumbel_and_never_aliases_suffix() {
        use super::{
            QWEN4_C1_LAYER0_CAPTURE_POISON_LIMIT, QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT,
            QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT, QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
            QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT, QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT,
            QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, VerifyDLayer0State, verify_d_gumbel_tail_graph_bits,
            verify_d_layer0_prefix_pack_key,
        };

        let prior_bits = 0b111
            | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT
            | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT
            | QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT
            | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT;
        assert_eq!(QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(
            QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT & QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT,
            0
        );

        let greedy_suffix = verify_d_gumbel_tail_graph_bits(false, false);
        let gumbel_suffix = verify_d_gumbel_tail_graph_bits(true, false);
        let coalesced_suffix = verify_d_gumbel_tail_graph_bits(true, true);

        let l0_greedy = verify_d_layer0_prefix_pack_key(greedy_suffix, true);
        let l0_gumbel = verify_d_layer0_prefix_pack_key(gumbel_suffix, true);
        let l0_coalesced = verify_d_layer0_prefix_pack_key(coalesced_suffix, true);
        let l0_split_embed = verify_d_layer0_prefix_pack_key(gumbel_suffix, false);
        let l0_qsa =
            verify_d_layer0_prefix_pack_key(gumbel_suffix | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, true);
        let l0_k16 = verify_d_layer0_prefix_pack_key(
            gumbel_suffix | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
            true,
        );

        // Greedy, Gumbel, and coalesced-LM-head suffixes share ONE Layer-0 prefix key.
        assert_eq!(l0_greedy, l0_gumbel);
        assert_eq!(l0_greedy, l0_coalesced);
        // Never aliases any suffix pack_key.
        for suffix in [greedy_suffix, gumbel_suffix, coalesced_suffix] {
            assert_ne!(l0_greedy, suffix);
            assert_ne!(l0_split_embed, suffix);
        }
        // Distinct Layer-0 topologies keep distinct keys.
        assert_ne!(l0_greedy, l0_split_embed);
        assert_ne!(l0_greedy, l0_qsa);
        assert_ne!(l0_greedy, l0_k16);
        assert_ne!(l0_qsa, l0_k16);

        // Warmup requires 1 hit for C=1, resets failure count on capture, and poisons at limit.
        let mut state = VerifyDLayer0State::default();
        let req = super::super::verify_qwen4_batched::batched_verify_graph_required_warmup_hits(1);
        assert_eq!(req, 1);
        let warm_key = (5usize, l0_greedy, None);
        assert!(!state.is_warm(&warm_key, req));
        state.record_warmup_hit(warm_key);
        assert!(state.is_warm(&warm_key, req));
        state.record_capture_failure();
        assert!(!state.capture_poisoned);
        state.record_captured(warm_key, req);
        assert_eq!(state.consecutive_capture_failures, 0);
        for _ in 0..QWEN4_C1_LAYER0_CAPTURE_POISON_LIMIT - 1 {
            state.record_capture_failure();
            assert!(!state.capture_poisoned);
        }
        state.record_capture_failure();
        assert!(state.capture_poisoned);
    }

    #[test]
    fn verify_d_rejection_tail_bit_is_disjoint_and_shares_layer0_prefix() {
        use super::{
            QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT, QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT,
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
            QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT, QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT,
            QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, verify_d_gumbel_tail_graph_bits,
            verify_d_layer0_prefix_pack_key,
        };

        let prior_bits = 0b111
            | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT
            | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT
            | QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT
            | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT;
        assert_eq!(QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT & prior_bits, 0);

        let greedy_suffix = verify_d_gumbel_tail_graph_bits(false, false);
        let gumbel_suffix = verify_d_gumbel_tail_graph_bits(true, false);
        let rejection_suffix =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT;
        assert_ne!(rejection_suffix, greedy_suffix);
        assert_ne!(rejection_suffix, gumbel_suffix);
        assert_eq!(
            verify_d_layer0_prefix_pack_key(rejection_suffix, true),
            verify_d_layer0_prefix_pack_key(greedy_suffix, true)
        );
        assert_eq!(
            verify_d_layer0_prefix_pack_key(rejection_suffix, true),
            verify_d_layer0_prefix_pack_key(gumbel_suffix, true)
        );
    }

    #[test]
    fn verify_d_qsa_graph_bits_separate_non_qsa_dense_prefix_and_sparse_qsa() {
        use super::{
            QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT, QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT,
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
            QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT, QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT,
            QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT,
            qwen4_qsa_dense_prefix_graph_eligible, verify_d_gumbel_tail_graph_bits,
            verify_d_layer0_prefix_pack_key, verify_d_qsa_graph_bits,
        };

        let prior_bits = 0b111
            | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT
            | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT
            | QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT
            | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT
            | QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT;
        assert_eq!(prior_bits, 0x3ff, "bits 0..=9 must all be accounted for");
        assert_eq!(
            QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT & prior_bits,
            0,
            "QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT must not overlap pack_key bits 0..=9"
        );
        assert_eq!(
            verify_d_qsa_graph_bits(false, true),
            0,
            "dense-prefix bit cannot fire when qsa_required is false"
        );

        for k in [2usize, 5, 7, 8, 9] {
            let short_non_qsa = verify_d_qsa_graph_bits(
                false,
                qwen4_qsa_dense_prefix_graph_eligible(false, 62, k, true),
            );
            let qsa_dense_prefix = verify_d_qsa_graph_bits(
                true,
                qwen4_qsa_dense_prefix_graph_eligible(true, 120, k, true),
            );
            let qsa_sparse = verify_d_qsa_graph_bits(
                true,
                qwen4_qsa_dense_prefix_graph_eligible(true, 16_176, k, true),
            );
            let qsa_sparse_128k = verify_d_qsa_graph_bits(
                true,
                qwen4_qsa_dense_prefix_graph_eligible(true, 131_072, k, true),
            );

            assert_eq!(
                short_non_qsa, 0,
                "k={k} non-QSA dense must have zero QSA bits"
            );
            assert_eq!(
                qsa_dense_prefix,
                QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT,
                "k={k} pre-2048 QSA dense prefix must set both QSA bits"
            );
            assert_eq!(
                qsa_sparse, QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT,
                "k={k} post-2048 sparse QSA must set only QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT"
            );
            assert_eq!(
                qsa_sparse, qsa_sparse_128k,
                "k={k} 16K and 128K sparse QSA share the same post-2048 sparse QSA bits"
            );

            // All three attention topologies must produce strictly disjoint keys.
            assert_ne!(short_non_qsa, qsa_dense_prefix, "k={k}");
            assert_ne!(short_non_qsa, qsa_sparse, "k={k}");
            assert_ne!(qsa_dense_prefix, qsa_sparse, "k={k}");

            // Suffix tail combinations preserve disjointness across QSA topologies,
            // and Layer-0 prefix preserves QSA bits while sharing suffix tails.
            let tails = [
                verify_d_gumbel_tail_graph_bits(false, false),
                verify_d_gumbel_tail_graph_bits(true, false),
                verify_d_gumbel_tail_graph_bits(true, true),
                QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT,
                QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT,
            ];
            let l0_non_qsa = verify_d_layer0_prefix_pack_key(short_non_qsa, true);
            let l0_dense_prefix = verify_d_layer0_prefix_pack_key(qsa_dense_prefix, true);
            let l0_sparse = verify_d_layer0_prefix_pack_key(qsa_sparse, true);
            assert_ne!(l0_non_qsa, l0_dense_prefix, "k={k}");
            assert_ne!(l0_non_qsa, l0_sparse, "k={k}");
            assert_ne!(l0_dense_prefix, l0_sparse, "k={k}");

            for tail in tails {
                let s_non_qsa = tail | short_non_qsa;
                let s_dense_prefix = tail | qsa_dense_prefix;
                let s_sparse = tail | qsa_sparse;
                assert_ne!(s_non_qsa, s_dense_prefix, "k={k} tail={tail}");
                assert_ne!(s_non_qsa, s_sparse, "k={k} tail={tail}");
                assert_ne!(s_dense_prefix, s_sparse, "k={k} tail={tail}");

                assert_eq!(verify_d_layer0_prefix_pack_key(s_non_qsa, true), l0_non_qsa);
                assert_eq!(
                    verify_d_layer0_prefix_pack_key(s_dense_prefix, true),
                    l0_dense_prefix
                );
                assert_eq!(verify_d_layer0_prefix_pack_key(s_sparse, true), l0_sparse);

                assert_ne!(l0_non_qsa, s_non_qsa);
                assert_ne!(l0_dense_prefix, s_dense_prefix);
                assert_ne!(l0_sparse, s_sparse);
            }
        }
    }

    #[test]
    fn verify_kgamma_qsa_regime_transition_evicts_only_stale_regime_on_same_slot() {
        use std::collections::HashMap;

        use spark_runtime::gpu::GraphHandle;

        use super::{
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
            VerifyKgammaGraphKey, VerifyKgammaGraphLru, plan_verify_kgamma_evictions,
            verify_d_layer0_prefix_pack_key, verify_d_qsa_graph_bits,
            verify_kgamma_is_layer0_prefix_key, verify_kgamma_qsa_regime_bits,
        };

        let non_qsa_suffix =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | verify_d_qsa_graph_bits(false, false);
        let dense_prefix_suffix =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | verify_d_qsa_graph_bits(true, true);
        let sparse_suffix =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | verify_d_qsa_graph_bits(true, false);

        let dense_prefix_l0 = verify_d_layer0_prefix_pack_key(dense_prefix_suffix, true);
        let sparse_l0 = verify_d_layer0_prefix_pack_key(sparse_suffix, true);
        assert_eq!(
            verify_kgamma_qsa_regime_bits(dense_prefix_suffix),
            verify_kgamma_qsa_regime_bits(dense_prefix_l0)
        );
        assert_eq!(
            verify_kgamma_qsa_regime_bits(sparse_suffix),
            verify_kgamma_qsa_regime_bits(sparse_l0)
        );
        assert_ne!(
            verify_kgamma_qsa_regime_bits(dense_prefix_suffix),
            verify_kgamma_qsa_regime_bits(sparse_suffix)
        );

        let mut cache: HashMap<VerifyKgammaGraphKey, GraphHandle> = HashMap::new();
        let mut lru = VerifyKgammaGraphLru::default();
        let mut next_id = 1u64;
        for slot in [0usize, 1] {
            for k in 2..=8usize {
                for pack in [dense_prefix_suffix, dense_prefix_l0] {
                    let key: VerifyKgammaGraphKey = (slot, k, pack, None);
                    cache.insert(key, GraphHandle(next_id));
                    lru.touch(key);
                    next_id += 1;
                }
            }
        }

        // Slot 0 crosses seq_len >= 2048 (`dense_prefix` -> `sparse`): all 14
        // stale dense_prefix graphs on slot 0 (7 suffix + 7 Layer-0) must be
        // evicted immediately, while slot 1's 14 graphs remain untouched.
        let new_sparse_key: VerifyKgammaGraphKey = (0, 7, sparse_suffix, None);
        let plan = plan_verify_kgamma_evictions(
            &cache,
            &lru,
            new_sparse_key,
            None,
            VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
        );
        assert_eq!(plan.qsa_regime_keys.len(), 14);
        assert!(plan.lru_keys.is_empty());
        assert!(plan.qsa_regime_keys.iter().all(|k| k.0 == 0));
        let evicted_l0 = plan
            .qsa_regime_keys
            .iter()
            .filter(|k| verify_kgamma_is_layer0_prefix_key(k))
            .count();
        assert_eq!(evicted_l0, 7);

        for key in plan.all_keys() {
            cache.remove(&key);
            lru.remove(&key);
        }
        cache.insert(new_sparse_key, GraphHandle(next_id));
        lru.touch(new_sparse_key);
        next_id += 1;
        let new_sparse_l0_key: VerifyKgammaGraphKey = (0, 7, sparse_l0, None);
        cache.insert(new_sparse_l0_key, GraphHandle(next_id));
        lru.touch(new_sparse_l0_key);

        // Slot 0 now reuses its slot for a new non-QSA request (`sparse` -> `non-QSA`):
        // both sparse suffix and sparse Layer-0 on slot 0 evict cleanly, leaving slot 1 intact.
        let non_qsa_key: VerifyKgammaGraphKey = (0, 5, non_qsa_suffix, None);
        let plan_non_qsa = plan_verify_kgamma_evictions(
            &cache,
            &lru,
            non_qsa_key,
            None,
            VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
        );
        assert_eq!(plan_non_qsa.qsa_regime_keys.len(), 2);
        assert!(plan_non_qsa.qsa_regime_keys.contains(&new_sparse_key));
        assert!(plan_non_qsa.qsa_regime_keys.contains(&new_sparse_l0_key));
        assert!(plan_non_qsa.lru_keys.is_empty());
        assert_eq!(cache.keys().filter(|k| k.0 == 1).count(), 14);
    }

    #[test]
    fn verify_kgamma_global_lru_cap_evicts_oldest_other_slot_and_honors_hits() {
        use std::collections::HashMap;

        use spark_runtime::gpu::GraphHandle;

        use super::{
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
            VerifyKgammaGraphKey, VerifyKgammaGraphLru, plan_verify_kgamma_evictions,
            verify_d_layer0_prefix_pack_key, verify_d_qsa_graph_bits,
            verify_kgamma_is_layer0_prefix_key,
        };

        let suffix_pack =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | verify_d_qsa_graph_bits(true, false);
        let l0_pack = verify_d_layer0_prefix_pack_key(suffix_pack, true);
        let cap = VERIFY_KGAMMA_GRAPH_MAX_ENTRIES;

        let mut cache: HashMap<VerifyKgammaGraphKey, GraphHandle> = HashMap::new();
        let mut lru = VerifyKgammaGraphLru::default();
        let mut next_id = 1u64;

        let insert_pair = |cache: &mut HashMap<VerifyKgammaGraphKey, GraphHandle>,
                           lru: &mut VerifyKgammaGraphLru,
                           next_id: &mut u64,
                           slot: usize,
                           k: usize| {
            let l0_key: VerifyKgammaGraphKey = (slot, k, l0_pack, None);
            let s_key: VerifyKgammaGraphKey = (slot, k, suffix_pack, None);

            let l0_plan = plan_verify_kgamma_evictions(cache, lru, l0_key, Some(s_key), cap);
            for evicted in l0_plan.all_keys() {
                cache.remove(&evicted);
                lru.remove(&evicted);
            }
            cache.insert(l0_key, GraphHandle(*next_id));
            lru.touch(l0_key);
            *next_id += 1;

            let s_plan = plan_verify_kgamma_evictions(cache, lru, s_key, Some(l0_key), cap);
            for evicted in s_plan.all_keys() {
                cache.remove(&evicted);
                lru.remove(&evicted);
            }
            cache.insert(s_key, GraphHandle(*next_id));
            lru.touch(s_key);
            *next_id += 1;

            (l0_plan, s_plan)
        };

        // C=1 single slot across K=2..=8 (7 suffix + 7 Layer-0): ZERO evictions.
        for k in 2..=8usize {
            let (l0_plan, s_plan) = insert_pair(&mut cache, &mut lru, &mut next_id, 0, k);
            assert!(l0_plan.is_empty(), "C=1 K={k} Layer-0 unexpectedly evicted");
            assert!(s_plan.is_empty(), "C=1 K={k} suffix unexpectedly evicted");
        }

        // Fill slot 1 up to exact cap (12 entries per tier = 7 + 5): still ZERO evictions.
        for k in 2..=6usize {
            let (l0_plan, s_plan) = insert_pair(&mut cache, &mut lru, &mut next_id, 1, k);
            assert!(l0_plan.is_empty() && s_plan.is_empty());
        }
        assert_eq!(
            cache
                .keys()
                .filter(|k| !verify_kgamma_is_layer0_prefix_key(k))
                .count(),
            cap
        );
        assert_eq!(
            cache
                .keys()
                .filter(|k| verify_kgamma_is_layer0_prefix_key(k))
                .count(),
            cap
        );

        // Touch slot 0's K=2 (originally oldest) so slot 0's K=3 becomes the LRU victim.
        let hot_l0: VerifyKgammaGraphKey = (0, 2, l0_pack, None);
        let hot_suffix: VerifyKgammaGraphKey = (0, 2, suffix_pack, None);
        lru.touch(hot_l0);
        lru.touch(hot_suffix);

        // Inserting slot 1's K=7 (13th entry) must evict slot 0's K=3, NOT slot 0's K=2.
        let (l0_plan, s_plan) = insert_pair(&mut cache, &mut lru, &mut next_id, 1, 7);
        assert_eq!(l0_plan.lru_keys, vec![(0, 3, l0_pack, None)]);
        assert_eq!(s_plan.lru_keys, vec![(0, 3, suffix_pack, None)]);
        assert!(cache.contains_key(&hot_l0));
        assert!(cache.contains_key(&hot_suffix));

        // Slot 3 across K=2..=8 keeps both tiers strictly <= cap (`12`).
        for k in 2..=8usize {
            let _ = insert_pair(&mut cache, &mut lru, &mut next_id, 3, k);
            let l0_count = cache
                .keys()
                .filter(|k| verify_kgamma_is_layer0_prefix_key(k))
                .count();
            let suffix_count = cache.len() - l0_count;
            assert!(suffix_count <= cap, "suffix_count={suffix_count} > {cap}");
            assert!(l0_count <= cap, "l0_count={l0_count} > {cap}");
        }

        // External removal (e.g. `async_chkpt::invalidate_verify_graphs_for_slot`)
        // leaves `lru` cleanly pruned by `retain_live`.
        let external_key: VerifyKgammaGraphKey = (3, 8, suffix_pack, None);
        cache.remove(&external_key);
        lru.retain_live(&cache);
        assert_eq!(lru.last_used_tick(&external_key), 0);
    }

    #[test]
    fn verify_kgamma_mem_headroom_shrinks_cap_preserves_active_sibling_and_gates_retention() {
        use std::collections::HashMap;

        use spark_runtime::gpu::GraphHandle;
        use spark_runtime::startup_ledger::MIB;

        use super::{
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, VERIFY_KGAMMA_GRAPH_CRITICAL_CAP,
            VERIFY_KGAMMA_GRAPH_EMERGENCY_CAP, VERIFY_KGAMMA_GRAPH_MAX_ENTRIES,
            VERIFY_KGAMMA_GRAPH_PRESSURE_CAP, VerifyKgammaGraphKey, VerifyKgammaGraphLru,
            effective_verify_kgamma_graph_cap, parse_verify_kgamma_graph_max_entries,
            plan_verify_kgamma_evictions, verify_d_layer0_prefix_pack_key, verify_d_qsa_graph_bits,
            verify_kgamma_is_layer0_prefix_key, verify_kgamma_retain_headroom_ok_with,
        };

        assert_eq!(
            parse_verify_kgamma_graph_max_entries(None),
            VERIFY_KGAMMA_GRAPH_MAX_ENTRIES
        );
        assert_eq!(
            parse_verify_kgamma_graph_max_entries(Some("")),
            VERIFY_KGAMMA_GRAPH_MAX_ENTRIES
        );
        assert_eq!(parse_verify_kgamma_graph_max_entries(Some("0")), 1);
        assert_eq!(parse_verify_kgamma_graph_max_entries(Some(" 12 ")), 12);
        assert_eq!(
            parse_verify_kgamma_graph_max_entries(Some("bogus")),
            VERIFY_KGAMMA_GRAPH_MAX_ENTRIES
        );

        let floor = 6_656 * MIB;
        assert_eq!(effective_verify_kgamma_graph_cap(12, None, Some(floor)), 12);
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 3_072 * MIB), Some(floor)),
            12
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 3_071 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_PRESSURE_CAP
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 1_024 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_PRESSURE_CAP
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 1_023 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_CRITICAL_CAP
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 512 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_CRITICAL_CAP
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor + 511 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_EMERGENCY_CAP
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(12, Some(floor - 21 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_EMERGENCY_CAP
        );
        // Never exceeds an explicitly smaller `base_cap`.
        assert_eq!(
            effective_verify_kgamma_graph_cap(4, Some(floor + 1_500 * MIB), Some(floor)),
            4
        );
        assert_eq!(
            effective_verify_kgamma_graph_cap(4, Some(floor + 700 * MIB), Some(floor)),
            VERIFY_KGAMMA_GRAPH_CRITICAL_CAP
        );

        assert!(verify_kgamma_retain_headroom_ok_with(None, Some(floor)));
        assert!(verify_kgamma_retain_headroom_ok_with(
            Some(floor + 1_024 * MIB),
            Some(floor)
        ));
        assert!(!verify_kgamma_retain_headroom_ok_with(
            Some(floor + 1_023 * MIB),
            Some(floor)
        ));

        // Under emergency pressure (`cap = 1`), capturing suffix for `(3, 7)`
        // trims both tiers down to 1 while preserving this step's Layer-0 prefix.
        let suffix_pack =
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT | verify_d_qsa_graph_bits(true, false);
        let l0_pack = verify_d_layer0_prefix_pack_key(suffix_pack, true);
        let mut cache: HashMap<VerifyKgammaGraphKey, GraphHandle> = HashMap::new();
        let mut lru = VerifyKgammaGraphLru::default();
        let mut next_id = 1u64;
        for slot in [0usize, 1, 2] {
            for k in 2..=4usize {
                for pack in [suffix_pack, l0_pack] {
                    let key: VerifyKgammaGraphKey = (slot, k, pack, None);
                    cache.insert(key, GraphHandle(next_id));
                    lru.touch(key);
                    next_id += 1;
                }
            }
        }
        let active_l0: VerifyKgammaGraphKey = (3, 7, l0_pack, None);
        cache.insert(active_l0, GraphHandle(next_id));
        lru.touch(active_l0);
        let new_suffix: VerifyKgammaGraphKey = (3, 7, suffix_pack, None);

        let plan = plan_verify_kgamma_evictions(&cache, &lru, new_suffix, Some(active_l0), 1);
        assert!(plan.qsa_regime_keys.is_empty());
        assert!(!plan.lru_keys.contains(&active_l0));
        for evicted in plan.all_keys() {
            cache.remove(&evicted);
            lru.remove(&evicted);
        }
        cache.insert(new_suffix, GraphHandle(next_id + 1));
        lru.touch(new_suffix);

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&active_l0));
        assert!(cache.contains_key(&new_suffix));
        assert_eq!(
            cache
                .keys()
                .filter(|k| verify_kgamma_is_layer0_prefix_key(k))
                .count(),
            1
        );
        assert_eq!(
            cache
                .keys()
                .filter(|k| !verify_kgamma_is_layer0_prefix_key(k))
                .count(),
            1
        );
    }

    #[test]
    fn verify_kgamma_eviction_synchronizes_stream_before_destroy_graph() {
        let source = include_str!("verify_d.rs");
        for fn_name in [
            "fn prepare_verify_kgamma_graph_slot(",
            "fn store_verify_kgamma_graph(",
        ] {
            let body = source
                .split(fn_name)
                .nth(1)
                .unwrap_or_else(|| panic!("missing {fn_name}"))
                .split("\n    fn ")
                .next()
                .expect("function body");
            let sync = body
                .find("self.gpu.synchronize(stream)?;")
                .unwrap_or_else(|| panic!("{fn_name} must synchronize stream before destroy"));
            let destroy = body
                .find("self.gpu.destroy_graph(old_graph)")
                .unwrap_or_else(|| panic!("{fn_name} must destroy evicted graph"));
            assert!(
                sync < destroy,
                "{fn_name}: synchronize must precede destroy"
            );
        }
    }

    #[test]
    fn c1_coalesced_lm_head_allows_greedy_only_under_c1_tc_verify_opt_in() {
        use super::{c1_coalesced_lm_head_eligible, parse_qwen4_c1_tc_verify};

        assert!(!parse_qwen4_c1_tc_verify(None));
        assert!(!parse_qwen4_c1_tc_verify(Some("0")));
        assert!(!parse_qwen4_c1_tc_verify(Some("false")));
        assert!(parse_qwen4_c1_tc_verify(Some("1")));
        assert!(parse_qwen4_c1_tc_verify(Some("true")));

        // Default (ATLAS_QWEN4_C1_TC_VERIFY off): greedy stays OFF (0-ULP exact);
        // non-greedy Gumbel/rejection engages for k >= 4.
        assert!(!c1_coalesced_lm_head_eligible(
            true, false, false, false, false, false, false, false, 8, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, false, false, false, false, false, 8, true, false, false, true,
        ));
        assert!(c1_coalesced_lm_head_eligible(
            true, false, true, false, false, false, false, false, 8, true, false, false, true,
        ));
        assert!(c1_coalesced_lm_head_eligible(
            false, true, true, false, false, false, false, false, 4, true, false, false, true,
        ));

        // Explicit ATLAS_QWEN4_C1_TC_VERIFY=1: greedy T=0 engages on both Flash-Next
        // (with Gumbel/rejection tail) and dense 27B (plain argmax tail) for k >= 4.
        assert!(c1_coalesced_lm_head_eligible(
            true, false, false, true, false, false, false, false, 8, true, false, false, true,
        ));
        assert!(c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 8, true, false, false, true,
        ));

        // Diagnostic/high-precision/small-k/explicit-TC guards remain fail-closed.
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 3, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, true, false, false, false, 8, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, true, false, false, 8, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, true, false, 8, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, true, 8, true, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 8, false, false, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 8, true, true, false, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 8, true, false, true, true,
        ));
        assert!(!c1_coalesced_lm_head_eligible(
            false, false, false, true, false, false, false, false, 8, true, false, false, false,
        ));
    }

    #[test]
    fn verify_d_inter_layer_hc_defer_protects_layer0_taps_diag_and_final_mixer() {
        use super::qwen4_verify_d_layer_should_defer_mlp_tail;

        let taps = [1usize, 10, 18, 27, 35];
        // Layer 0 NEVER defers (PLE runs between Layer 0 and Layer 1).
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true,
            true,
            true,
            0,
            false,
            true,
            false,
            true,
            &[],
            false
        ));
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, true, 0, false, true, true, true, &taps, false
        ));
        // Non-tap layers 1..=46 defer when enabled on batched hybrid route.
        assert!(qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, true, 2, false, true, true, true, &taps, false
        ));
        // DFlash capture tap layers flush before `try_dflash_capture_rows`.
        for &tap in &taps {
            assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
                true, true, true, tap, false, true, true, true, &taps, false
            ));
        }
        // Layer 47 defers only when `final_mixer` can fuse it.
        assert!(qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, true, 47, true, true, true, true, &taps, false
        ));
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, true, 47, true, false, true, true, &taps, false
        ));
        // Non-batched / non-hybrid / diagnostic / disabled fail closed.
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            false, true, true, 2, false, true, true, true, &taps, false
        ));
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true, false, true, 2, false, true, true, true, &taps, false
        ));
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, false, 2, false, true, true, true, &taps, false
        ));
        assert!(!qwen4_verify_d_layer_should_defer_mlp_tail(
            true, true, true, 2, false, true, true, true, &taps, true
        ));
    }

    #[test]
    fn verify_d_unified_ple_graph_bits_are_disjoint_and_admission_guards_disk_ple() {
        use super::{
            QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT, QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT,
            QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT, QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT,
            QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT, QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT,
            QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT, QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT,
            QWEN4_UNIFIED_PLE_OVERLAP_VERIFY_GRAPH_BIT, QWEN4_UNIFIED_PLE_VERIFY_GRAPH_BIT,
            qwen4_unified_ple_verify_graph_eligible, verify_d_unified_ple_graph_bits,
        };

        let prior_bits = 0b111
            | QWEN4_K16_BATCHED_VERIFY_GRAPH_BIT
            | QWEN4_QSA_DENSE_PREFIX_GRAPH_BIT
            | QWEN4_IN_GRAPH_GUMBEL_VERIFY_GRAPH_BIT
            | QWEN4_COALESCED_LMHEAD_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT
            | QWEN4_LAYER0_EMBED_FUSED_VERIFY_GRAPH_BIT
            | QWEN4_IN_GRAPH_REJECTION_VERIFY_GRAPH_BIT
            | QWEN4_QSA_REQUIRED_VERIFY_GRAPH_BIT;
        assert_eq!(QWEN4_UNIFIED_PLE_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(QWEN4_UNIFIED_PLE_OVERLAP_VERIFY_GRAPH_BIT & prior_bits, 0);
        assert_eq!(
            QWEN4_UNIFIED_PLE_VERIFY_GRAPH_BIT & QWEN4_UNIFIED_PLE_OVERLAP_VERIFY_GRAPH_BIT,
            0
        );

        assert_eq!(verify_d_unified_ple_graph_bits(false, true, true), 0);
        let serial_split = verify_d_unified_ple_graph_bits(true, false, false);
        let serial_fused = verify_d_unified_ple_graph_bits(true, true, false);
        let overlap_fused = verify_d_unified_ple_graph_bits(true, true, true);
        assert_ne!(serial_split, 0);
        assert_ne!(serial_split, serial_fused);
        assert_ne!(serial_fused, overlap_fused);
        // Unified graphs never set QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT (they live in the main tier).
        assert_eq!(overlap_fused & QWEN4_LAYER0_PREFIX_VERIFY_GRAPH_BIT, 0);

        // Admission requires mmap PLE, >= 2 layers, unskipped layer 1, single-PP, no debug dump.
        assert!(qwen4_unified_ple_verify_graph_eligible(
            true, 48, false, 1, false, true
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            true, 48, false, 1, false, false
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            false, 48, false, 1, false, true
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            true, 1, false, 1, false, true
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            true, 48, true, 1, false, true
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            true, 48, false, 2, false, true
        ));
        assert!(!qwen4_unified_ple_verify_graph_eligible(
            true, 48, false, 1, true, true
        ));
    }
}
