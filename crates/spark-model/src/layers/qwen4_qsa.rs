// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 sparse-attention index side branch.
//!
//! The first implementation is fail-closed: C=1, BF16 main KV, compression
//! ratio four, and the released 4x128/1x128 indexer geometry. Side-cache rows
//! are keyed by main paged-KV physical blocks, preserving block ownership.

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

mod device;
pub mod native_batch;
mod position_contract;
mod prefill_exact;
pub mod prefill_fp8;
pub mod prefill_fused_topk;
pub mod prefill_sparse;
mod prefill_sparse_dump;
mod prefill_sparse_run;

use position_contract::{prefill_index_batch_is_safe, valid_physical_query};

const INDEX_HEADS: u32 = 4;
const INDEX_DIM: u32 = 128;
const INDEX_WIDTH: u32 = 640;
const COMPRESS_RATIO: usize = 4;
const VERIFY_K5_ROWS: usize = 5;
/// Widest journaled verify frame: every width the all-K hybrid admits
/// (kernels/gb10/common/qwen4_qsa.cu VERIFY_MAX_ROWS must match).
const VERIFY_MAX_ROWS: usize = crate::layers::QWEN4_HYBRID_MAX_ROWS;
const OUTPUT_WIDTH: usize = 2051;
/// Base QSA top-k token index width (`512 * 4 + 3 = 2051`).
pub const QSA_OUTPUT_WIDTH: usize = OUTPUT_WIDTH;
/// Maximum speculative draft slots reserved after `OUTPUT_WIDTH` in
/// `QsaCache::reused_draft_indices` (`(2051 + 16) * 4 = 8,268 B`).
pub const QSA_REUSE_MAX_DRAFTS: usize = 16;
const QUERY_HEADS: usize = 24;
const SPARSE_SPLITS: usize = 8;
/// Block size of the parallel select kernels (one CTA per row).
pub(crate) const SELECT_PAR_THREADS: u32 = 1024;

/// Validate and compute `OUTPUT_WIDTH + step_offset` for `DraftSelectionReuse`.
pub fn checked_reused_index_width(step_offset: usize, max_drafts: usize) -> Result<usize> {
    ensure!(
        max_drafts <= QSA_REUSE_MAX_DRAFTS,
        "QSA draft-selection reuse max_drafts={max_drafts} exceeds capacity {QSA_REUSE_MAX_DRAFTS}"
    );
    ensure!(
        step_offset <= max_drafts,
        "QSA draft-selection reuse step_offset={step_offset} exceeds max_drafts={max_drafts}"
    );
    OUTPUT_WIDTH
        .checked_add(step_offset)
        .context("QSA reused index width overflow")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftReuseState {
    SnapshotStep0 {
        snapshotted_in_update: bool,
        from_verify_rows: bool,
    },
    ReuseStep {
        base_seq_len: usize,
        step_offset: usize,
        max_drafts: usize,
    },
}

std::thread_local! {
    static DRAFT_SELECTION_REUSE_STATE: std::cell::Cell<Option<DraftReuseState>> =
        const { std::cell::Cell::new(None) };
}

/// Thread-local RAII guard coordinating MTP `DraftSelectionReuse` between
/// `qwen4_mtp` and `Qwen4QsaIndexer` without widening the `TransformerLayer`
/// trait boundary.
pub(crate) struct DraftSelectionReuseGuard {
    prev: Option<DraftReuseState>,
}

impl DraftSelectionReuseGuard {
    /// Arm step-0 snapshot: runs full QSA selection and copies the resulting
    /// `OUTPUT_WIDTH` indices into `QsaCache::reused_draft_indices`.
    pub(crate) fn snapshot_step0() -> Self {
        let prev = DRAFT_SELECTION_REUSE_STATE.with(|cell| {
            cell.replace(Some(DraftReuseState::SnapshotStep0 {
                snapshotted_in_update: false,
                from_verify_rows: false,
            }))
        });
        Self { prev }
    }

    /// Arm draft-step reuse (`step_offset` in `1..=max_drafts`): skips QSA
    /// side-cache update/scoring/selection and attends over
    /// `reused_draft_indices[0..OUTPUT_WIDTH + step_offset]` after appending
    /// the speculative draft position(s).
    pub(crate) fn reuse_step(base_seq_len: usize, step_offset: usize, max_drafts: usize) -> Self {
        let prev = DRAFT_SELECTION_REUSE_STATE.with(|cell| {
            cell.replace(Some(DraftReuseState::ReuseStep {
                base_seq_len,
                step_offset,
                max_drafts,
            }))
        });
        Self { prev }
    }
}

impl Drop for DraftSelectionReuseGuard {
    fn drop(&mut self) {
        DRAFT_SELECTION_REUSE_STATE.with(|cell| cell.set(self.prev));
    }
}

#[derive(Clone, Copy, Debug)]
struct QsaCacheLayout {
    raw_ring: usize,
    compressed_keys: usize,
    verify_undo: usize,
    verify_offsets: usize,
    verify_valid_mask: usize,
    projected_qk: usize,
    pooled_key: usize,
    first_position: usize,
    logits: usize,
    token_indices: usize,
    reused_draft_indices: usize,
    sparse_partial_output: usize,
    sparse_partial_stat: usize,
    max_compressed_groups: usize,
}

fn checked_qsa_product(factors: &[usize], label: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |total, factor| {
        total
            .checked_mul(*factor)
            .with_context(|| format!("QSA cache size overflow: {label}"))
    })
}

impl QsaCacheLayout {
    fn checked(num_main_blocks: usize, block_size: usize) -> Result<Self> {
        ensure!(
            block_size >= COMPRESS_RATIO && block_size.is_multiple_of(COMPRESS_RATIO),
            "QSA requires KV block size divisible by {COMPRESS_RATIO}, got {block_size}"
        );
        ensure!(num_main_blocks > 0, "empty QSA cache geometry");
        let groups_per_page = block_size / COMPRESS_RATIO;
        let max_compressed_groups = checked_qsa_product(
            &[num_main_blocks, groups_per_page],
            "compressed group count",
        )?;
        let sparse_parts = checked_qsa_product(
            &[VERIFY_MAX_ROWS, QUERY_HEADS, SPARSE_SPLITS],
            "sparse partial count",
        )?;
        Ok(Self {
            raw_ring: checked_qsa_product(
                &[num_main_blocks, COMPRESS_RATIO, INDEX_DIM as usize, 2],
                "raw ring",
            )?,
            compressed_keys: checked_qsa_product(
                &[num_main_blocks, groups_per_page, INDEX_DIM as usize, 2],
                "compressed keys",
            )?,
            verify_undo: checked_qsa_product(
                &[VERIFY_MAX_ROWS, INDEX_DIM as usize, 2],
                "verify undo rows",
            )?,
            verify_offsets: checked_qsa_product(
                &[VERIFY_MAX_ROWS, std::mem::size_of::<u64>()],
                "verify offsets",
            )?,
            verify_valid_mask: std::mem::size_of::<u32>(),
            projected_qk: checked_qsa_product(
                &[
                    crate::layers::QWEN4_NATIVE_MAX_ROWS,
                    INDEX_WIDTH as usize,
                    2,
                ],
                "projected QK rows",
            )?,
            pooled_key: checked_qsa_product(&[INDEX_DIM as usize, 2], "pooled key")?,
            first_position: std::mem::size_of::<u32>(),
            logits: checked_qsa_product(
                &[
                    VERIFY_MAX_ROWS,
                    max_compressed_groups,
                    std::mem::size_of::<f32>(),
                ],
                "verify logits",
            )?,
            token_indices: checked_qsa_product(
                &[VERIFY_MAX_ROWS, OUTPUT_WIDTH, std::mem::size_of::<i32>()],
                "token indices",
            )?,
            reused_draft_indices: checked_qsa_product(
                &[
                    OUTPUT_WIDTH + QSA_REUSE_MAX_DRAFTS,
                    std::mem::size_of::<i32>(),
                ],
                "reused draft indices",
            )?,
            sparse_partial_output: checked_qsa_product(
                &[
                    sparse_parts,
                    INDEX_DIM as usize,
                    2,
                    std::mem::size_of::<f32>(),
                ],
                "sparse partial output",
            )?,
            sparse_partial_stat: checked_qsa_product(
                &[sparse_parts, std::mem::size_of::<f32>()],
                "sparse partial statistics",
            )?,
            max_compressed_groups,
        })
    }

    fn total_bytes(self) -> Result<usize> {
        [
            self.raw_ring,
            self.compressed_keys,
            self.verify_undo,
            self.verify_undo,
            self.verify_offsets,
            self.verify_offsets,
            self.verify_valid_mask,
            self.projected_qk,
            self.pooled_key,
            self.first_position,
            self.logits,
            self.token_indices,
            self.reused_draft_indices,
            self.sparse_partial_output,
            self.sparse_partial_stat,
            self.sparse_partial_stat,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .context("QSA cache total byte count overflow")
        })
    }
}

/// Exact bytes allocated by one attention layer's lazy QSA side cache.
/// Startup preflight calls the same checked geometry as the allocator so a
/// widened concurrency profile cannot hide verifier-row scratch allocations.
pub fn checked_side_cache_bytes(num_main_blocks: usize, block_size: usize) -> Result<usize> {
    QsaCacheLayout::checked(num_main_blocks, block_size)?.total_bytes()
}

/// ATLAS_QWEN4_QSA_SELECT_PAR=1 swaps the single-thread top-512 decode select
/// for the parallel kernel. Same group set, same tie rule, same output order:
/// the token_indices buffer is byte-identical (fn-decode qsa_select_test).
/// Verify widths whose QSA writes are journaled and committed: K5 always;
/// every other exact frame (2..=4 rows from confidence-gated drafts, 6..=9
/// rows from wider drafts) unless `ATLAS_QWEN4_QSA_JOURNAL_ALL_K=0` (A/B
/// control only: without the journal their rejected suffix can corrupt a
/// pooled key, see qwen4_qsa_commit_verify_prefix_rows).
pub(crate) fn verify_journal_rows(rows: usize) -> bool {
    static ALL_K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let all_k = *ALL_K.get_or_init(|| {
        let enabled = std::env::var("ATLAS_QWEN4_QSA_JOURNAL_ALL_K").as_deref() != Ok("0");
        if enabled {
            tracing::info!(
                "QSA_JOURNAL_ALL_K_ENGAGED selector=ATLAS_QWEN4_QSA_JOURNAL_ALL_K max_k={VERIFY_MAX_ROWS} tracked_slots={VERIFY_MAX_ROWS}"
            );
        }
        enabled
    });
    rows == VERIFY_K5_ROWS || ((2..=VERIFY_MAX_ROWS).contains(&rows) && all_k)
}

pub(crate) fn graph_safe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_GRAPH_SAFE").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QSA_GRAPH_SAFE_ENGAGED selector=ATLAS_QWEN4_QSA_GRAPH_SAFE");
        }
        on
    })
}

/// ATLAS_QWEN4_QSA_ATTN_V2=1: byte-identical sparse-attention decode partial
/// (per-warp exact score tree, sequential softmax scalars, pipelined V).
/// Groups per 256-thread block of the v2 score kernels (8 warps x 4 groups).
pub(crate) const SCORE_V2_GROUPS_PER_BLOCK: usize = 32;

/// ATLAS_QWEN4_QSA_SCORE_V2=1: byte-identical QSA index score, 4 groups per
/// warp with all loads in flight and the query in registers.
pub(crate) fn score_v2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_SCORE_V2").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QSA_SCORE_V2_ENGAGED selector=ATLAS_QWEN4_QSA_SCORE_V2");
        }
        on
    })
}

pub(crate) fn attn_v2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_ATTN_V2").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QSA_ATTN_V2_ENGAGED selector=ATLAS_QWEN4_QSA_ATTN_V2");
        }
        on
    })
}

pub(crate) fn select_par_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_SELECT_PAR").is_ok_and(|v| v == "1");
        if on {
            tracing::info!(
                "QSA_SELECT_PAR_ENGAGED selector=ATLAS_QWEN4_QSA_SELECT_PAR threads=1024"
            );
        }
        on
    })
}

/// ATLAS_QWEN4_QSA_OVERLAP=1: run the plain-decode indexer chain (index
/// projection, norms, pool, RoPE, score, select) on a side stream while the
/// main stream runs the QKV projection and KV write; the main stream joins
/// before sparse attention. Same kernels, same arguments, disjoint buffers
/// (the chain only reads the attention input and writes the QSA cache), so
/// token_indices and every downstream byte are unchanged.
pub(crate) fn overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_OVERLAP").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QSA_OVERLAP_ENGAGED selector=ATLAS_QWEN4_QSA_OVERLAP");
        }
        on
    })
}

/// Side stream + fork/join events of one QSA layer (ATLAS_QWEN4_QSA_OVERLAP=1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaOverlap {
    pub stream: u64,
    pub fork: u64,
    pub join: u64,
}

/// ATLAS_QWEN4_QSA_SELECT_V3=1: byte-identical register-resident select (keys
/// loaded once, per-(j, warp) ballot compaction); implies the parallel route.
/// At <= 3072 groups or > 33792 the kernel runs the SELECT_PAR code itself.
pub(crate) fn select_v3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_QSA_SELECT_V3").is_ok_and(|v| v == "1");
        if on {
            tracing::info!("QSA_SELECT_V3_ENGAGED selector=ATLAS_QWEN4_QSA_SELECT_V3 threads=1024");
        }
        on
    })
}

/// Multi-row QSA verify (K11-D2): runs ring stage/pool/store for rows 0..k-1,
/// then scores all k rows in ONE pass over `compressed_keys` and selects top-512
/// in ONE parallel `grid = [1, k, 1]` launch. Enabled when `ATLAS_QWEN4_QSA_VERIFY_ROWS=1`
/// or by default when `score_v2_enabled()` and `(select_v3_enabled() || select_par_enabled())`
/// (unless `ATLAS_QWEN4_QSA_VERIFY_ROWS=0`).
pub(crate) fn verify_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let explicit = std::env::var("ATLAS_QWEN4_QSA_VERIFY_ROWS").ok();
        let on = match explicit.as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => score_v2_enabled() && (select_v3_enabled() || select_par_enabled()),
        };
        if on {
            tracing::info!("QSA_VERIFY_ROWS_ENGAGED selector=ATLAS_QWEN4_QSA_VERIFY_ROWS");
        }
        on
    })
}

/// `ATLAS_QWEN4_QSA_VERIFY_STAGE_POOL_FUSED` (default on unless `"0"`):
/// fuse QSA verify-row stage/pool/RoPE/store into a single kernel launch.
fn verify_stage_pool_fused_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_QSA_VERIFY_STAGE_POOL_FUSED")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Qwen4 switches from dense attention to sparse QSA at position 2048.
/// Batched multi-row preparation and dispatch are valid when every row in the
/// frame is dense (`endpoint <= 2048`) or every row in the frame is sparse
/// (`seq_len >= 2048`). Only a frame that straddles the 2048 boundary
/// (`seq_len < 2048 && endpoint > 2048`) falls back to row-by-row execution.
pub(crate) const VERIFY_ROWS_DENSE_ENDPOINT: usize = 2_048;

/// `ATLAS_QWEN4_BATCHED_QSA_MULTI` (default on unless `"0"`): run cross-sequence
/// verify QSA stage/pool/RoPE/store, score, select-expand, and sparse attention
/// as batched multi-sequence launches across all active sequences in the frame.
pub(crate) fn batched_qsa_multi_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_BATCHED_QSA_MULTI")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Flag bits on [`QsaBatchSeqDesc::flags`], matching `qwen4_qsa.cu` and `qwen4_qsa_wave2.cu`.
pub const QSA_DESC_FLAG_QSA_REQUIRED: u32 = 1 << 0;
pub const QSA_DESC_FLAG_SPARSE: u32 = 1 << 1;
pub const QSA_DESC_FLAG_JOURNAL: u32 = 1 << 2;

/// Maximum sequences addressed by one cross-sequence QSA descriptor upload (`128 / K2 = 64`).
pub const QSA_CROSS_SEQ_MAX_SEQS: usize = 64;

/// Environment selector for the SM120 token-aligned `TILE_ROWS = 4`, `kBc = 64`
/// E4M3 QSA prefill core (`qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4` +
/// `qwen4_qsa_prefill_plan_fused_sort_desc_t4_b128`).
///
/// - Unset / `"auto"`: enabled on GPUs with `active_sm_count() > 64` (SM120/SM103);
///   remains disabled on GB10 (`active_sm_count() <= 64`) to keep GB10 defaults 100% unchanged.
/// - `"0"` / `"off"` / `"false"`: explicitly opt out back to `pipe_2cta` / `pipe`.
/// - `"1"` / `"on"` / `"true"`: explicitly force on (e.g. for GB10 A/B parity).
pub const QSA_PREFILL_KBC64_SELECTOR: &str = "ATLAS_QWEN4_QSA_PREFILL_KBC64";
/// SM threshold above which [`QSA_PREFILL_KBC64_SELECTOR`] auto-engages.
pub const QSA_PREFILL_KBC64_SM_THRESHOLD: u32 = 64;
/// Query rows per CTA for the token-aligned `kBc = 64` QSA prefill core
/// (`4` tokens x `12` Q heads = `48` live rows -> 3 x 16-row MMA warp-pairs,
/// 100% M-tile fill with 0 padded rows).
pub const QSA_PREFILL_T4_TILE_ROWS: usize = 4;
/// Maximum union entries per `TILE_ROWS = 4` tile (`ceil(4 * 513 / 4) * 4 = 2052`).
pub const QSA_PREFILL_T4_CAP: usize = (QSA_PREFILL_T4_TILE_ROWS * 513).div_ceil(4) * 4;
/// Dynamic shared memory for `qwen4_qsa_prefill_plan_fused_sort_desc_t4_b128`
/// (`(16 + 2052) * 4 = 8,272 B`).
pub const QSA_PREFILL_T4_PLAN_SMEM: usize = (16 + QSA_PREFILL_T4_CAP) * 4;
/// Threads per CTA (`6` warps = `3` warp-pairs = `48` exact M-rows) for
/// `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4` (all 256 Q cols in `qf[8][4]` regs!).
pub const QSA_PREFILL_KBC64_T4_THREADS: u32 = 192;
/// Dynamic shared memory for `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4`:
/// `16,384` B (`buf_k`) + `16,384` B (`buf_v`) + `384` B (`s_mx`) + `3,072` B (`s_a`)
/// + `128` B (`s_ent[2][16]`) = `36,352 B <= 49,152 B` (2 CTAs/SM, 0 Q SMEM reloads).
pub const QSA_PREFILL_KBC64_T4_SMEM: usize = 16_384 + 16_384 + 384 + 3_072 + 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaPrefillKbc64Policy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
pub fn parse_qsa_prefill_kbc64_policy(raw: Option<&str>) -> QsaPrefillKbc64Policy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") => {
            QsaPrefillKbc64Policy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on") => {
            QsaPrefillKbc64Policy::Enabled
        }
        _ => QsaPrefillKbc64Policy::Auto,
    }
}

#[inline]
#[must_use]
pub const fn resolve_qsa_prefill_kbc64_enabled(
    policy: QsaPrefillKbc64Policy,
    sm_count: u32,
) -> bool {
    match policy {
        QsaPrefillKbc64Policy::Disabled => false,
        QsaPrefillKbc64Policy::Auto => sm_count > QSA_PREFILL_KBC64_SM_THRESHOLD,
        QsaPrefillKbc64Policy::Enabled => true,
    }
}

/// Process-cached gate for the SM120 `TILE_ROWS = 4`, `kBc = 64` QSA prefill core.
#[must_use]
pub fn qsa_prefill_kbc64_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var(QSA_PREFILL_KBC64_SELECTOR).ok();
        let policy = parse_qsa_prefill_kbc64_policy(raw.as_deref());
        let sm_count = atlas_core::device::active_sm_count();
        let active = resolve_qsa_prefill_kbc64_enabled(policy, sm_count);
        if active {
            tracing::info!(
                "QSA_PREFILL_KBC64_ENGAGED selector={} sm_count={} tile_rows={} kbc=64 smem_bytes={}",
                QSA_PREFILL_KBC64_SELECTOR,
                sm_count,
                QSA_PREFILL_T4_TILE_ROWS,
                QSA_PREFILL_KBC64_T4_SMEM,
            );
        }
        active
    })
}

/// Loaded kernel handles for the SM120 `TILE_ROWS = 4`, `kBc = 64` QSA prefill core.
#[derive(Clone, Copy, Debug)]
pub struct QsaPrefillKbc64T4Kernels {
    pub plan_t4_b128: KernelHandle,
    pub attn_kbc64_t4: KernelHandle,
}

/// Resolve both `TILE_ROWS = 4` / `kBc = 64` kernels from the `qwen4_qsa_prefill_attn_fp8` module.
/// Returns `None` if the active GPU bundle does not contain both symbols.
pub fn qsa_prefill_kbc64_t4_kernels(gpu: &dyn GpuBackend) -> Option<QsaPrefillKbc64T4Kernels> {
    let plan_t4_b128 = crate::layers::try_kernel(
        gpu,
        "qwen4_qsa_prefill_attn_fp8",
        "qwen4_qsa_prefill_plan_fused_sort_desc_t4_b128",
    );
    let attn_kbc64_t4 = crate::layers::try_kernel(
        gpu,
        "qwen4_qsa_prefill_attn_fp8",
        "qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4",
    );
    if plan_t4_b128.0 != 0 && attn_kbc64_t4.0 != 0 {
        Some(QsaPrefillKbc64T4Kernels {
            plan_t4_b128,
            attn_kbc64_t4,
        })
    } else {
        None
    }
}

/// Launch `qwen4_qsa_prefill_plan_fused_sort_desc_t4_b128` for `rows` queries (`TILE_ROWS = 4`).
#[allow(clippy::too_many_arguments)]
pub fn launch_qsa_prefill_plan_t4_b128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    masks: DevicePtr,
    entries: DevicePtr,
    counts: DevicePtr,
    rows: usize,
    first_pos: usize,
    mask_words: usize,
    cap: usize,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "QSA T=4 plan kernel handle is null");
    ensure!(rows > 0, "QSA T=4 plan requires non-zero rows");
    ensure!(
        cap >= QSA_PREFILL_T4_CAP,
        "QSA T=4 plan requires cap >= {QSA_PREFILL_T4_CAP}, got {cap}"
    );
    let tiles = rows.div_ceil(QSA_PREFILL_T4_TILE_ROWS);
    let smem_cap = cap.min(QSA_PREFILL_T4_CAP);
    let smem_bytes = (16 + smem_cap) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([tiles as u32, 1, 1])
        .block([128, 1, 1])
        .shared_mem(smem_bytes as u32)
        .arg_ptr(masks)
        .arg_ptr(entries)
        .arg_ptr(counts)
        .arg_u32(rows as u32)
        .arg_u32(first_pos as u32)
        .arg_u32(mask_words as u32)
        .arg_u32(QSA_PREFILL_T4_TILE_ROWS as u32)
        .arg_u32(cap as u32)
        .arg_u32(tiles as u32)
        .launch(stream)
}

/// Parameters for launching `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4`.
#[derive(Clone, Copy, Debug)]
pub struct QsaPrefillKbc64T4Args {
    pub q8: DevicePtr,
    pub kc8: DevicePtr,
    pub vc8: DevicePtr,
    pub block_table: DevicePtr,
    pub entries: DevicePtr,
    pub counts: DevicePtr,
    pub out_e4m3: DevicePtr,
    pub gate: DevicePtr,
    pub rows: u32,
    pub first_pos: u32,
    pub q_stride: u32,
    pub o_stride: u32,
    pub gate_stride: u32,
    pub page_size: u32,
    pub k_page_stride: u64,
    pub k_tok_stride: u32,
    pub v_page_stride: u64,
    pub v_tok_stride: u32,
    pub scale_log2_h0: f32,
    pub scale_log2_h1: f32,
    pub out_scale_h0: f32,
    pub out_scale_h1: f32,
    pub a_scale_out: DevicePtr,
    pub a_scale_val: f32,
    pub local_kv_heads: u32,
    pub cap: u32,
}

/// Launch `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_kbc64_t4` with 2-CTA/SM dynamic SMEM
/// (`QSA_PREFILL_KBC64_T4_SMEM` = `36,352 B`).
pub fn launch_qsa_prefill_attn_kbc64_t4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &QsaPrefillKbc64T4Args,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "QSA kBc=64 T=4 attention kernel handle is null"
    );
    ensure!(args.rows > 0, "QSA kBc=64 T=4 attention requires rows > 0");
    ensure!(
        args.page_size.is_power_of_two() && args.page_size >= COMPRESS_RATIO as u32,
        "QSA kBc=64 T=4 requires power-of-two page_size >= {COMPRESS_RATIO}, got {}",
        args.page_size
    );
    ensure!(
        args.local_kv_heads == 1 || args.local_kv_heads == 2,
        "QSA kBc=64 T=4 requires 1 or 2 local KV heads, got {}",
        args.local_kv_heads
    );
    let tiles = (args.rows as usize).div_ceil(QSA_PREFILL_T4_TILE_ROWS) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([tiles, args.local_kv_heads, 1])
        .block([QSA_PREFILL_KBC64_T4_THREADS, 1, 1])
        .shared_mem(QSA_PREFILL_KBC64_T4_SMEM as u32)
        .arg_ptr(args.q8)
        .arg_ptr(args.kc8)
        .arg_ptr(args.vc8)
        .arg_ptr(args.block_table)
        .arg_ptr(args.entries)
        .arg_ptr(args.counts)
        .arg_ptr(args.out_e4m3)
        .arg_ptr(args.gate)
        .arg_u32(args.rows)
        .arg_u32(args.first_pos)
        .arg_u32(args.q_stride)
        .arg_u32(args.o_stride)
        .arg_u32(args.gate_stride)
        .arg_u32(args.page_size)
        .arg_u64(args.k_page_stride)
        .arg_u32(args.k_tok_stride)
        .arg_u64(args.v_page_stride)
        .arg_u32(args.v_tok_stride)
        .arg_f32(args.scale_log2_h0)
        .arg_f32(args.scale_log2_h1)
        .arg_f32(args.out_scale_h0)
        .arg_f32(args.out_scale_h1)
        .arg_ptr(args.a_scale_out)
        .arg_f32(args.a_scale_val)
        .arg_u32(args.cap)
        .launch(stream)
}

/// Per-sequence descriptor (`32 B`, 16-byte aligned) matching
/// `struct alignas(16) QsaBatchSeqDesc` in `qwen4_qsa.cu` and `qwen4_qsa_wave2.cu`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QsaBatchSeqDesc {
    pub row_base: u32,
    pub seq_len: u32,
    pub k: u32,
    pub journal_lane: u32,
    pub flags: u32,
    pub logits_offset: u32,
    pub logits_stride: u32,
    pub score_block_end: u32,
}

impl QsaBatchSeqDesc {
    /// Exact byte size matching CUDA `sizeof(QsaBatchSeqDesc)`.
    pub const SIZE_BYTES: usize = 32;

    /// Serialize this descriptor in exact little-endian ABI layout into `dst`.
    pub fn write_le_bytes(&self, dst: &mut [u8]) {
        debug_assert!(dst.len() >= Self::SIZE_BYTES);
        dst[0..4].copy_from_slice(&self.row_base.to_le_bytes());
        dst[4..8].copy_from_slice(&self.seq_len.to_le_bytes());
        dst[8..12].copy_from_slice(&self.k.to_le_bytes());
        dst[12..16].copy_from_slice(&self.journal_lane.to_le_bytes());
        dst[16..20].copy_from_slice(&self.flags.to_le_bytes());
        dst[20..24].copy_from_slice(&self.logits_offset.to_le_bytes());
        dst[24..28].copy_from_slice(&self.logits_stride.to_le_bytes());
        dst[28..32].copy_from_slice(&self.score_block_end.to_le_bytes());
    }
}

/// Model-wide shared scratch arena for cross-sequence batched QSA verify
/// (`~57.2 MiB`), allocated once on [`crate::layer::Qwen4CrossSeqStaging`]
/// and reused sequentially across every full-attention layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4CrossSeqQsaScratch {
    pub batch_token_indices: DevicePtr,
    pub batch_partial_output: DevicePtr,
    pub batch_partial_max: DevicePtr,
    pub batch_partial_sum: DevicePtr,
    pub batch_logits: DevicePtr,
    pub batch_logits_floats: usize,
}

/// Partition `1..=256` sparse-attention rows for the 128-bit masked CUDA ABI.
///
/// When `needs_128_chunking` is `false` (unmasked partial + reduce kernels),
/// emits a single pass covering all `num_rows`. When `true`, splits across
/// `0..min(num_rows, 128)` (`mask[0..2]`) and `128..num_rows` (`mask[2..4]`),
/// skipping any 128-row half whose mask is all zeros.
#[inline]
#[must_use]
fn plan_qsa_sparse_row_chunks(
    num_rows: usize,
    row_mask: Option<[u64; 4]>,
    needs_128_chunking: bool,
) -> ([(usize, usize, u64, u64); 2], usize) {
    if !needs_128_chunking {
        return ([(0, num_rows, u64::MAX, u64::MAX), (0, 0, 0, 0)], 1);
    }
    let mask = row_mask.unwrap_or([u64::MAX; 4]);
    let mut chunks = [(0usize, 0usize, 0u64, 0u64); 2];
    let mut count = 0usize;
    let lo_rows = num_rows.min(128);
    if lo_rows > 0 && (mask[0] != 0 || mask[1] != 0) {
        chunks[count] = (0, lo_rows, mask[0], mask[1]);
        count += 1;
    }
    if num_rows > 128 && (mask[2] != 0 || mask[3] != 0) {
        chunks[count] = (128, num_rows - 128, mask[2], mask[3]);
        count += 1;
    }
    (chunks, count)
}

impl Qwen4CrossSeqQsaScratch {
    /// Capacity of `batch_logits` in FP32 elements (`32 MiB = 8,388,608` floats).
    pub const BATCH_LOGITS_FLOATS: usize = 8 * 1024 * 1024;
    /// Maximum rows covered by this scratch (`QWEN4_CROSS_SEQ_MAX_ROWS = 256`).
    pub const MAX_ROWS: usize = crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS;
    /// Null placeholder when QSA is disabled.
    pub const NULL: Self = Self {
        batch_token_indices: DevicePtr::NULL,
        batch_partial_output: DevicePtr::NULL,
        batch_partial_max: DevicePtr::NULL,
        batch_partial_sum: DevicePtr::NULL,
        batch_logits: DevicePtr::NULL,
        batch_logits_floats: 0,
    };

    pub fn is_allocated(&self) -> bool {
        !self.batch_token_indices.is_null()
            && !self.batch_partial_output.is_null()
            && !self.batch_partial_max.is_null()
            && !self.batch_partial_sum.is_null()
            && !self.batch_logits.is_null()
            && self.batch_logits_floats > 0
    }

    /// Checked byte sizes for `(token_indices, partial_output, partial_max, partial_sum, logits)`.
    pub fn component_bytes() -> Result<[usize; 5]> {
        let parts = checked_qsa_product(
            &[Self::MAX_ROWS, QUERY_HEADS, SPARSE_SPLITS],
            "cross-seq QSA partial count",
        )?;
        let token_indices = checked_qsa_product(
            &[Self::MAX_ROWS, OUTPUT_WIDTH, std::mem::size_of::<i32>()],
            "cross-seq QSA token indices",
        )?;
        let partial_output = checked_qsa_product(
            &[parts, INDEX_DIM as usize, 2, std::mem::size_of::<f32>()],
            "cross-seq QSA partial output",
        )?;
        let partial_stat = checked_qsa_product(
            &[parts, std::mem::size_of::<f32>()],
            "cross-seq QSA partial stat",
        )?;
        let logits = checked_qsa_product(
            &[Self::BATCH_LOGITS_FLOATS, std::mem::size_of::<f32>()],
            "cross-seq QSA batch logits",
        )?;
        Ok([
            token_indices,
            partial_output,
            partial_stat,
            partial_stat,
            logits,
        ])
    }

    /// Total device bytes of the model-wide cross-sequence QSA scratch arena.
    pub fn total_bytes() -> Result<usize> {
        Self::component_bytes()?
            .into_iter()
            .try_fold(0usize, |acc, b| {
                acc.checked_add(b)
                    .context("cross-seq QSA scratch total byte count overflow")
            })
    }

    /// Allocate the model-wide cross-sequence QSA scratch arena once.
    pub fn allocate(gpu: &dyn GpuBackend) -> Result<Self> {
        let [idx_b, po_b, pm_b, ps_b, log_b] = Self::component_bytes()?;
        Ok(Self {
            batch_token_indices: gpu.alloc(idx_b)?,
            batch_partial_output: gpu.alloc(po_b)?,
            batch_partial_max: gpu.alloc(pm_b)?,
            batch_partial_sum: gpu.alloc(ps_b)?,
            batch_logits: gpu.alloc(log_b)?,
            batch_logits_floats: Self::BATCH_LOGITS_FLOATS,
        })
    }
}

pub(crate) fn verify_rows_context_safe(seq_len: usize, num_tokens: usize) -> bool {
    seq_len.checked_add(num_tokens).is_some_and(|endpoint| {
        u32::try_from(endpoint).is_ok()
            && (endpoint <= VERIFY_ROWS_DENSE_ENDPOINT || seq_len >= VERIFY_ROWS_DENSE_ENDPOINT)
    })
}

const VERIFY_ROUTE_TELEMETRY_ENV: &str = "ATLAS_QWEN4_QSA_ROUTE_TELEMETRY";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct VerifyRouteTotals {
    row_serial_frames: u64,
    dense_paged_rows: u64,
    sparse_qsa_rows: u64,
    high_speed_swap_rows: u64,
    mla_rows: u64,
    sparse_rows_legacy_frames: u64,
    sparse_rows_v2_frames: u64,
}

/// The core that actually completed for one serial verifier row. The
/// attention dispatch site returns this value; callers must not reconstruct
/// it from sequence geometry because boundary frames can mix routes and HSS
/// supersedes both ordinary routes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VerifyRowAttentionRoute {
    DensePaged,
    SparseQsa,
    HighSpeedSwap,
    Mla,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VerifyRowRouteCounts {
    dense_paged: u64,
    sparse_qsa: u64,
    high_speed_swap: u64,
    mla: u64,
}

impl VerifyRowRouteCounts {
    pub(crate) fn record(&mut self, route: VerifyRowAttentionRoute) {
        match route {
            VerifyRowAttentionRoute::DensePaged => self.dense_paged += 1,
            VerifyRowAttentionRoute::SparseQsa => self.sparse_qsa += 1,
            VerifyRowAttentionRoute::HighSpeedSwap => self.high_speed_swap += 1,
            VerifyRowAttentionRoute::Mla => self.mla += 1,
        }
    }

    fn total(self) -> u64 {
        self.dense_paged + self.sparse_qsa + self.high_speed_swap + self.mla
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VerifyAttentionRoute {
    RowSerial(VerifyRowRouteCounts),
    SparseRowsLegacy,
    SparseRowsV2,
}

impl VerifyAttentionRoute {
    fn label(self) -> &'static str {
        match self {
            Self::RowSerial(_) => "row-serial",
            Self::SparseRowsLegacy => "sparse-rows-legacy",
            Self::SparseRowsV2 => "sparse-rows-v2",
        }
    }

    fn row_counts(self) -> VerifyRowRouteCounts {
        match self {
            Self::RowSerial(counts) => counts,
            Self::SparseRowsLegacy | Self::SparseRowsV2 => VerifyRowRouteCounts::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparseRowsKernel {
    Legacy,
    V2,
}

/// Process-wide launch counters used only by the opt-in qualification trace.
///
/// The counters are deliberately outside the QSA cache: a qualification run
/// needs one monotone sequence across requests, while the cache is per model
/// layer. Normal serving pays only one cached boolean branch per QSA verify
/// layer and performs no atomic operation when telemetry is disabled.
struct VerifyRouteTally {
    row_serial_frames: std::sync::atomic::AtomicU64,
    dense_paged_rows: std::sync::atomic::AtomicU64,
    sparse_qsa_rows: std::sync::atomic::AtomicU64,
    high_speed_swap_rows: std::sync::atomic::AtomicU64,
    mla_rows: std::sync::atomic::AtomicU64,
    sparse_rows_legacy_frames: std::sync::atomic::AtomicU64,
    sparse_rows_v2_frames: std::sync::atomic::AtomicU64,
}

impl VerifyRouteTally {
    const fn new() -> Self {
        Self {
            row_serial_frames: std::sync::atomic::AtomicU64::new(0),
            dense_paged_rows: std::sync::atomic::AtomicU64::new(0),
            sparse_qsa_rows: std::sync::atomic::AtomicU64::new(0),
            high_speed_swap_rows: std::sync::atomic::AtomicU64::new(0),
            mla_rows: std::sync::atomic::AtomicU64::new(0),
            sparse_rows_legacy_frames: std::sync::atomic::AtomicU64::new(0),
            sparse_rows_v2_frames: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn record(&self, route: VerifyAttentionRoute) -> VerifyRouteTotals {
        use std::sync::atomic::Ordering;

        match route {
            VerifyAttentionRoute::RowSerial(counts) => {
                self.row_serial_frames.fetch_add(1, Ordering::Relaxed);
                self.dense_paged_rows
                    .fetch_add(counts.dense_paged, Ordering::Relaxed);
                self.sparse_qsa_rows
                    .fetch_add(counts.sparse_qsa, Ordering::Relaxed);
                self.high_speed_swap_rows
                    .fetch_add(counts.high_speed_swap, Ordering::Relaxed);
                self.mla_rows.fetch_add(counts.mla, Ordering::Relaxed);
            }
            VerifyAttentionRoute::SparseRowsLegacy => {
                self.sparse_rows_legacy_frames
                    .fetch_add(1, Ordering::Relaxed);
            }
            VerifyAttentionRoute::SparseRowsV2 => {
                self.sparse_rows_v2_frames.fetch_add(1, Ordering::Relaxed);
            }
        }
        VerifyRouteTotals {
            row_serial_frames: self.row_serial_frames.load(Ordering::Relaxed),
            dense_paged_rows: self.dense_paged_rows.load(Ordering::Relaxed),
            sparse_qsa_rows: self.sparse_qsa_rows.load(Ordering::Relaxed),
            high_speed_swap_rows: self.high_speed_swap_rows.load(Ordering::Relaxed),
            mla_rows: self.mla_rows.load(Ordering::Relaxed),
            sparse_rows_legacy_frames: self.sparse_rows_legacy_frames.load(Ordering::Relaxed),
            sparse_rows_v2_frames: self.sparse_rows_v2_frames.load(Ordering::Relaxed),
        }
    }
}

static VERIFY_ROUTE_TALLY: VerifyRouteTally = VerifyRouteTally::new();
static VERIFY_ROUTE_FRAME_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn verify_route_telemetry_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(VERIFY_ROUTE_TELEMETRY_ENV) {
        Ok(value) if value == "1" => {
            tracing::info!(
                "QSA_VERIFY_ROUTE_TELEMETRY_ENGAGED selector={VERIFY_ROUTE_TELEMETRY_ENV}"
            );
            true
        }
        Ok(value) if value == "0" => false,
        Ok(value) => {
            tracing::warn!(
                "{VERIFY_ROUTE_TELEMETRY_ENV} must be exactly 0 or 1; got {value:?}; telemetry disabled"
            );
            false
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "{VERIFY_ROUTE_TELEMETRY_ENV} must be valid UTF-8; telemetry disabled"
            );
            false
        }
    })
}

/// Record only after every host dispatch for this layer returned successfully.
/// Row-serial receipts contain exact per-row dispatch counts, so mixed
/// dense/sparse boundary frames and HSS cannot be mislabeled by geometry.
/// Group the monotone `frame_id` inside one request-scoped log delta, verify
/// its `(seq_len, rows)`, and require the exact QSA layer set. This is enqueue
/// evidence, not GPU completion, correctness, latency, or throughput evidence.
pub(crate) fn record_verify_route(
    route: VerifyAttentionRoute,
    num_tokens: usize,
    seq_len: usize,
    layer_idx: usize,
) {
    if !verify_route_telemetry_enabled() {
        return;
    }
    use std::sync::atomic::Ordering;

    // Qwen4 full-attention ordinal zero starts each eager verifier frame. The
    // fresh-process C=1/TP=1 qualification contract makes this monotone ID
    // unambiguous; missing layer zero leaves ID 0 or a stale ID and the
    // consumer rejects the incomplete layer set.
    let frame_id = if layer_idx == 0 {
        VERIFY_ROUTE_FRAME_ID.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        VERIFY_ROUTE_FRAME_ID.load(Ordering::Relaxed)
    };
    let totals = VERIFY_ROUTE_TALLY.record(route);
    let row_counts = route.row_counts();
    debug_assert_eq!(
        row_counts.total(),
        if matches!(route, VerifyAttentionRoute::RowSerial(_)) {
            num_tokens as u64
        } else {
            0
        }
    );
    let route_label = route.label();
    tracing::info!(
        "QSA_VERIFY_ROUTE_RECEIPT route={route_label} frame_id={frame_id} \
         rows={num_tokens} seq_len={seq_len} layer={layer_idx} \
         dense_paged_rows={} sparse_qsa_rows={} high_speed_swap_rows={} mla_rows={} \
         row_serial_frame_total={} dense_paged_row_total={} sparse_qsa_row_total={} \
         high_speed_swap_row_total={} mla_row_total={} sparse_rows_legacy_frame_total={} \
         sparse_rows_v2_frame_total={}",
        row_counts.dense_paged,
        row_counts.sparse_qsa,
        row_counts.high_speed_swap,
        row_counts.mla,
        totals.row_serial_frames,
        totals.dense_paged_rows,
        totals.sparse_qsa_rows,
        totals.high_speed_swap_rows,
        totals.mla_rows,
        totals.sparse_rows_legacy_frames,
        totals.sparse_rows_v2_frames,
    );
}

pub(crate) fn qwen4_yarn_inv_freq_values_with_theta(
    config: &atlas_core::config::ModelConfig,
    theta: f32,
    rotary_dim: usize,
) -> Result<Vec<f32>> {
    ensure!(config.yarn_factor > 0.0, "YaRN factor must be positive");
    let factor = config.yarn_factor;
    let beta_fast = if config.yarn_beta_fast > 0.0 {
        config.yarn_beta_fast
    } else {
        32.0
    };
    let beta_slow = if config.yarn_beta_slow > 0.0 {
        config.yarn_beta_slow
    } else {
        1.0
    };
    let original = config.yarn_original_max_position_embeddings as f32;
    ensure!(
        original > 0.0 && rotary_dim >= 2 && theta.is_finite() && theta > 1.0,
        "invalid Qwen4 YaRN geometry"
    );
    let dim = rotary_dim as f32;
    let correction = |rotations: f32| {
        (dim * (original / (rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * theta.ln())
    };
    let low = correction(beta_fast).floor().max(0.0);
    let high = correction(beta_slow).ceil().min((rotary_dim - 1) as f32);
    let denom = if (high - low).abs() < 1e-6 {
        0.001
    } else {
        high - low
    };
    let mut table = Vec::with_capacity(rotary_dim / 2);
    for j in 0..rotary_dim / 2 {
        let base = theta.powf((2 * j) as f32 / dim);
        let ramp = ((j as f32 - low) / denom).clamp(0.0, 1.0);
        table.push((1.0 / base) * (1.0 - ramp) + (1.0 / (factor * base)) * ramp);
    }
    Ok(table)
}

pub fn compute_yarn_inv_freq(
    config: &atlas_core::config::ModelConfig,
    rotary_dim: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let table =
        qwen4_yarn_inv_freq_values_with_theta(config, config.rope_theta as f32, rotary_dim)?;
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

struct QsaCache {
    raw_ring: DevicePtr,
    compressed_keys: DevicePtr,
    /// Exact byte counts passed to the physical raw/compressed allocations.
    side_cache_allocation_bytes: [usize; 2],
    verify_undo_raw: DevicePtr,
    verify_undo_compressed: DevicePtr,
    verify_undo_raw_offsets: DevicePtr,
    verify_undo_compressed_offsets: DevicePtr,
    verify_valid_mask: DevicePtr,
    /// Lane-major journal geometry (`qwen4_verify_lanes`): each of the five
    /// journal arrays above holds `journal_lanes` consecutive lanes.
    journal_lanes: usize,
    journal_undo_lane_bytes: usize,
    journal_offsets_lane_bytes: usize,
    journal_mask_lane_bytes: usize,
    /// Exact byte counts passed to the five journal allocations above.
    journal_allocation_bytes: [usize; 5],
    projected_qk: DevicePtr,
    pooled_key: DevicePtr,
    first_position: DevicePtr,
    logits: DevicePtr,
    token_indices: DevicePtr,
    /// Persistent snapshot of step-0 `token_indices[0..OUTPUT_WIDTH]` plus up to
    /// `QSA_REUSE_MAX_DRAFTS` appended speculative draft positions (`8,268 B`).
    reused_draft_indices: DevicePtr,
    sparse_partial_output: DevicePtr,
    sparse_partial_max: DevicePtr,
    sparse_partial_sum: DevicePtr,
    max_compressed_groups: usize,
    num_main_blocks: usize,
    block_size: usize,
    precomputed_next_row: std::sync::atomic::AtomicUsize,
    precomputed_num_rows: std::sync::atomic::AtomicUsize,
    reused_snapshot_valid: std::sync::atomic::AtomicBool,
    reused_last_appended_step: std::sync::atomic::AtomicUsize,
}

/// One lane of the verify undo journal (see `qwen4_verify_lanes`).
#[derive(Clone, Copy, Debug)]
struct QsaJournalPtrs {
    undo_raw: DevicePtr,
    undo_compressed: DevicePtr,
    undo_raw_offsets: DevicePtr,
    undo_compressed_offsets: DevicePtr,
    valid_mask: DevicePtr,
}

/// Byte offsets of `lane` in the five lane-major journal arrays.
fn journal_lane_offsets(
    lane: usize,
    lanes: usize,
    undo_lane_bytes: usize,
    offsets_lane_bytes: usize,
    mask_lane_bytes: usize,
) -> Result<(usize, usize, usize)> {
    ensure!(
        lane < lanes,
        "QSA verify journal lane {lane} outside the allocated {lanes} lanes"
    );
    use crate::layers::qwen4_verify_lanes::lane_offset;
    Ok((
        lane_offset(lane, undo_lane_bytes)?,
        lane_offset(lane, offsets_lane_bytes)?,
        lane_offset(lane, mask_lane_bytes)?,
    ))
}

/// Export a lane only from a real allocation and its recorded exact extent.
pub(crate) fn checked_batch_verify_lane_range(
    allocation: DevicePtr,
    allocation_bytes: usize,
    lane_bytes: usize,
    lanes: usize,
    lane: usize,
) -> Result<(DevicePtr, usize, DevicePtr, usize)> {
    ensure!(
        !allocation.is_null() && lane_bytes > 0 && lanes > 0 && lane < lanes,
        "verify lane has no nonempty actual allocation"
    );
    let total = lane_bytes
        .checked_mul(lanes)
        .context("verify allocation extent overflow")?;
    ensure!(
        total == allocation_bytes,
        "recorded verify allocation extent differs from lane layout"
    );
    let offset = lane
        .checked_mul(lane_bytes)
        .context("verify lane offset overflow")?;
    let allocation_end = allocation
        .0
        .checked_add(u64::try_from(allocation_bytes)?)
        .context("verify allocation address overflow")?;
    let span = allocation
        .0
        .checked_add(u64::try_from(offset)?)
        .context("verify span address overflow")?;
    let span_end = span
        .checked_add(u64::try_from(lane_bytes)?)
        .context("verify span endpoint overflow")?;
    ensure!(
        span >= allocation.0 && span_end <= allocation_end,
        "verify lane span escapes the recorded actual allocation"
    );
    Ok((allocation, allocation_bytes, DevicePtr(span), lane_bytes))
}

fn distinct_batch_verify_lane_ranges(
    ranges: Vec<(DevicePtr, usize, DevicePtr, usize)>,
) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
    for (index, (_, _, span, bytes)) in ranges.iter().enumerate() {
        let end = span
            .0
            .checked_add(u64::try_from(*bytes)?)
            .context("verify span endpoint overflow")?;
        for (_, _, previous, previous_bytes) in &ranges[..index] {
            let previous_end = previous
                .0
                .checked_add(u64::try_from(*previous_bytes)?)
                .context("verify span endpoint overflow")?;
            ensure!(
                span.0 >= previous_end || previous.0 >= end,
                "persistent verify lane spans alias"
            );
        }
    }
    Ok(ranges)
}

impl QsaCache {
    /// Journal lane selected by the calling thread's verify lane guard.
    fn journal(&self) -> Result<QsaJournalPtrs> {
        let lane = crate::layers::qwen4_verify_lanes::current_lane();
        let (undo, offsets, mask) = journal_lane_offsets(
            lane,
            self.journal_lanes,
            self.journal_undo_lane_bytes,
            self.journal_offsets_lane_bytes,
            self.journal_mask_lane_bytes,
        )?;
        Ok(QsaJournalPtrs {
            undo_raw: self.verify_undo_raw.offset(undo),
            undo_compressed: self.verify_undo_compressed.offset(undo),
            undo_raw_offsets: self.verify_undo_raw_offsets.offset(offsets),
            undo_compressed_offsets: self.verify_undo_compressed_offsets.offset(offsets),
            valid_mask: self.verify_valid_mask.offset(mask),
        })
    }
}

/// Device bytes of one verify-journal lane for one attention layer: two undo
/// row arrays, two offset arrays and the valid mask.
pub fn verify_journal_lane_bytes() -> Result<usize> {
    // Journal extents do not depend on cache geometry; any valid layout works.
    let layout = QsaCacheLayout::checked(1, COMPRESS_RATIO)?;
    [
        layout.verify_undo,
        layout.verify_undo,
        layout.verify_offsets,
        layout.verify_offsets,
        layout.verify_valid_mask,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .context("QSA journal lane byte count overflow")
    })
}

pub struct Qwen4QsaIndexer {
    index_qk_proj: DenseWeight,
    q_norm: DenseWeight,
    k_norm: DenseWeight,
    dense_gemv: KernelHandle,
    dense_gemv_batchn: KernelHandle,
    dense_gemm: KernelHandle,
    rms_norm: KernelHandle,
    rope: KernelHandle,
    rope_yarn_scaled: KernelHandle,
    yarn_inv_freq: DevicePtr,
    yarn_attention_factor: f32,
    stage_pool: KernelHandle,
    store_compressed: KernelHandle,
    snapshot_verify_k5_row: KernelHandle,
    commit_verify_k5_prefix: KernelHandle,
    commit_verify_prefix_rows: KernelHandle,
    stage_prefill_raw: KernelHandle,
    pool_prefill: KernelHandle,
    store_prefill_compressed: KernelHandle,
    score: KernelHandle,
    select_expand: KernelHandle,
    score_device: KernelHandle,
    select_expand_device: KernelHandle,
    /// Parallel bit-exact select (ATLAS_QWEN4_QSA_SELECT_PAR=1); NULL when off or absent.
    select_expand_par: KernelHandle,
    select_expand_device_par: KernelHandle,
    sparse_attention_bf16_partial: KernelHandle,
    sparse_attention_nvfp4_partial: KernelHandle,
    /// Byte-identical 4-groups-per-warp score (ATLAS_QWEN4_QSA_SCORE_V2=1); NULL when off.
    score_v2: KernelHandle,
    score_device_v2: KernelHandle,
    /// Multi-row verify score and select-expand (ATLAS_QWEN4_QSA_VERIFY_ROWS); NULL when off.
    score_rows_v2: KernelHandle,
    score_device_rows_v2: KernelHandle,
    select_expand_rows_par: KernelHandle,
    select_expand_device_rows_par: KernelHandle,
    verify_rows_stage_pool_rope_store: KernelHandle,
    sparse_attention_bf16_partial_rows: KernelHandle,
    sparse_attention_bf16_partial_rows_v2: KernelHandle,
    sparse_attention_nvfp4_partial_rows: KernelHandle,
    sparse_attention_nvfp4_partial_rows_v2: KernelHandle,
    sparse_attention_bf16_reduce_rows: KernelHandle,
    sparse_attention_bf16_reduce_gated_rows: KernelHandle,
    /// Cross-sequence batched multi-seq QSA verify kernels (`ATLAS_QWEN4_BATCHED_QSA_MULTI`).
    verify_rows_stage_pool_rope_store_multi_seq: KernelHandle,
    score_multi_seq_rows_v2: KernelHandle,
    select_expand_multi_seq_rows: KernelHandle,
    sparse_attention_bf16_partial_rows_v2_masked: KernelHandle,
    sparse_attention_nvfp4_partial_rows_v2_masked: KernelHandle,
    sparse_attention_bf16_reduce_gated_rows_masked: KernelHandle,
    sparse_attention_bf16_reduce_rows_masked: KernelHandle,
    commit_verify_prefix_multi_seq: KernelHandle,
    /// Byte-identical faster BF16 partial (ATLAS_QWEN4_QSA_ATTN_V2=1); NULL when off.
    sparse_attention_bf16_partial_v2: KernelHandle,
    /// MTP `DraftSelectionReuse` kernels (`qwen4_qsa_append_draft_pos` and
    /// parameterized-width `*_partial_reused` variants); NULL if absent.
    append_draft_pos: KernelHandle,
    sparse_attention_bf16_partial_reused: KernelHandle,
    sparse_attention_bf16_partial_v2_reused: KernelHandle,
    sparse_attention_nvfp4_partial_reused: KernelHandle,
    sparse_attention_bf16_reduce: KernelHandle,
    sparse_attention_bf16_reduce_gated: KernelHandle,
    /// Side stream + events for the decode indexer overlap; None when off.
    overlap: Option<QsaOverlap>,
    cache: OnceLock<QsaCache>,
}

pub fn qwen4_qsa_fused_gate_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_QWEN4_QSA_FUSED_GATE").ok().as_deref() != Some("0"))
}

impl Qwen4QsaIndexer {
    pub(crate) fn validate_prefill_index_batch(
        num_tokens: usize,
        seq_len_start: usize,
    ) -> Result<()> {
        ensure!(
            prefill_index_batch_is_safe(num_tokens, seq_len_start),
            "ATLAS_QWEN4_QSA_PREFILL_GEMM requires 1..4 rows within one physical compression group; wider or crossing batches can overwrite live QSA raw rows"
        );
        Ok(())
    }

    pub fn new(
        index_qk_proj: DenseWeight,
        q_norm: DenseWeight,
        k_norm: DenseWeight,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let yarn_inv_freq = if config.yarn_factor > 0.0 {
            compute_yarn_inv_freq(config, config.rotary_dim(), gpu)?
        } else {
            DevicePtr::NULL
        };
        let try_verify_rows_kernel = |name: &str| {
            let h = crate::layers::try_kernel(gpu, "qwen4_qsa", name);
            if h.0 != 0 {
                h
            } else {
                crate::layers::try_kernel(gpu, "qwen4_qsa_prefill_attn_fp8", name)
            }
        };
        let rows_on = verify_rows_enabled();
        let rows_use_v3 = select_v3_enabled()
            || (!select_par_enabled()
                && std::env::var("ATLAS_QWEN4_QSA_VERIFY_ROWS").as_deref() == Ok("1"));
        Ok(Self {
            index_qk_proj,
            q_norm,
            k_norm,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchn: crate::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16_batchn"),
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            rms_norm: gpu.kernel("norm", "rms_norm")?,
            rope: gpu.kernel("rope", "rope_forward")?,
            rope_yarn_scaled: gpu.kernel("rope", "rope_forward_yarn_scaled")?,
            yarn_inv_freq,
            yarn_attention_factor: if config.yarn_factor > 0.0 && !config.is_qwen4_exp() {
                1.0 + 0.1 * config.yarn_factor.ln()
            } else {
                1.0
            },
            stage_pool: gpu.kernel("qwen4_qsa", "qwen4_qsa_stage_pool")?,
            store_compressed: gpu.kernel("qwen4_qsa", "qwen4_qsa_store_compressed")?,
            snapshot_verify_k5_row: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_snapshot_verify_k5_row",
            ),
            commit_verify_k5_prefix: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_commit_verify_k5_prefix",
            ),
            commit_verify_prefix_rows: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_commit_verify_prefix_rows",
            ),
            verify_rows_stage_pool_rope_store: {
                let h = crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa_wave2",
                    "qwen4_qsa_verify_rows_stage_pool_rope_store",
                );
                if h.0 != 0 {
                    h
                } else {
                    try_verify_rows_kernel("qwen4_qsa_verify_rows_stage_pool_rope_store")
                }
            },
            stage_prefill_raw: gpu.kernel("qwen4_qsa", "qwen4_qsa_stage_prefill_raw")?,
            pool_prefill: gpu.kernel("qwen4_qsa", "qwen4_qsa_pool_prefill")?,
            store_prefill_compressed: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_store_prefill_compressed")?,
            score: gpu.kernel("qwen4_qsa", "qwen4_qsa_score")?,
            select_expand: gpu.kernel("qwen4_qsa", "qwen4_qsa_select_expand")?,
            score_device: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_score_device_meta",
            ),
            select_expand_device: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_select_expand_device_meta",
            ),
            select_expand_par: if select_v3_enabled() {
                crate::layers::try_kernel(gpu, "qwen4_qsa", "qwen4_qsa_select_expand_par_v3")
            } else if select_par_enabled() {
                crate::layers::try_kernel(gpu, "qwen4_qsa", "qwen4_qsa_select_expand_par")
            } else {
                KernelHandle(0)
            },
            select_expand_device_par: if select_v3_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_select_expand_device_meta_par_v3",
                )
            } else if select_par_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_select_expand_device_meta_par",
                )
            } else {
                KernelHandle(0)
            },
            sparse_attention_bf16_partial: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_partial")?,
            sparse_attention_nvfp4_partial: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_nvfp4_partial")?,
            score_v2: if score_v2_enabled() {
                crate::layers::try_kernel(gpu, "qwen4_qsa", "qwen4_qsa_score_v2")
            } else {
                KernelHandle(0)
            },
            score_device_v2: if score_v2_enabled() {
                crate::layers::try_kernel(gpu, "qwen4_qsa", "qwen4_qsa_score_device_meta_v2")
            } else {
                KernelHandle(0)
            },
            score_rows_v2: if rows_on {
                try_verify_rows_kernel("qwen4_qsa_score_rows_v2")
            } else {
                KernelHandle(0)
            },
            score_device_rows_v2: if rows_on {
                try_verify_rows_kernel("qwen4_qsa_score_device_meta_rows_v2")
            } else {
                KernelHandle(0)
            },
            select_expand_rows_par: if rows_on {
                if rows_use_v3 {
                    try_verify_rows_kernel("qwen4_qsa_select_expand_rows_v3")
                } else {
                    try_verify_rows_kernel("qwen4_qsa_select_expand_rows_par")
                }
            } else {
                KernelHandle(0)
            },
            select_expand_device_rows_par: if rows_on {
                if rows_use_v3 {
                    try_verify_rows_kernel("qwen4_qsa_select_expand_device_meta_rows_par_v3")
                } else {
                    try_verify_rows_kernel("qwen4_qsa_select_expand_device_meta_rows_par")
                }
            } else {
                KernelHandle(0)
            },
            sparse_attention_bf16_partial_rows: if rows_on {
                try_verify_rows_kernel("qwen4_qsa_sparse_attention_bf16_partial_rows")
            } else {
                KernelHandle(0)
            },
            sparse_attention_bf16_partial_rows_v2: if rows_on && attn_v2_enabled() {
                try_verify_rows_kernel("qwen4_qsa_sparse_attention_bf16_partial_rows_v2")
            } else {
                KernelHandle(0)
            },
            sparse_attention_nvfp4_partial_rows: if rows_on {
                try_verify_rows_kernel("qwen4_qsa_sparse_attention_nvfp4_partial_rows")
            } else {
                KernelHandle(0)
            },
            sparse_attention_nvfp4_partial_rows_v2: if rows_on && attn_v2_enabled() {
                try_verify_rows_kernel("qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2")
            } else {
                KernelHandle(0)
            },
            sparse_attention_bf16_reduce_rows: if rows_on {
                try_verify_rows_kernel("qwen4_qsa_sparse_attention_bf16_reduce_rows")
            } else {
                KernelHandle(0)
            },
            sparse_attention_bf16_reduce_gated_rows: if rows_on {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_sparse_attention_bf16_reduce_gated_rows",
                )
            } else {
                KernelHandle(0)
            },
            verify_rows_stage_pool_rope_store_multi_seq: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa_wave2",
                "qwen4_qsa_verify_rows_stage_pool_rope_store_multi_seq",
            ),
            score_multi_seq_rows_v2: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_score_multi_seq_rows_v2",
            ),
            select_expand_multi_seq_rows: if select_par_enabled() && !select_v3_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_select_expand_multi_seq_rows_par",
                )
            } else {
                let h = crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_select_expand_multi_seq_rows_v3",
                );
                if h.0 != 0 {
                    h
                } else {
                    crate::layers::try_kernel(
                        gpu,
                        "qwen4_qsa",
                        "qwen4_qsa_select_expand_multi_seq_rows_par",
                    )
                }
            },
            sparse_attention_bf16_partial_rows_v2_masked: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_bf16_partial_rows_v2_masked",
            ),
            sparse_attention_nvfp4_partial_rows_v2_masked: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_nvfp4_partial_rows_v2_masked",
            ),
            sparse_attention_bf16_reduce_gated_rows_masked: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_bf16_reduce_gated_rows_masked",
            ),
            sparse_attention_bf16_reduce_rows_masked: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_bf16_reduce_rows_masked",
            ),
            commit_verify_prefix_multi_seq: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_commit_verify_prefix_multi_seq",
            ),
            sparse_attention_bf16_partial_v2: if attn_v2_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_sparse_attention_bf16_partial_v2",
                )
            } else {
                KernelHandle(0)
            },
            append_draft_pos: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_append_draft_pos",
            ),
            sparse_attention_bf16_partial_reused: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_bf16_partial_reused",
            ),
            sparse_attention_bf16_partial_v2_reused: if attn_v2_enabled() {
                crate::layers::try_kernel(
                    gpu,
                    "qwen4_qsa",
                    "qwen4_qsa_sparse_attention_bf16_partial_v2_reused",
                )
            } else {
                KernelHandle(0)
            },
            sparse_attention_nvfp4_partial_reused: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_nvfp4_partial_reused",
            ),
            sparse_attention_bf16_reduce: gpu
                .kernel("qwen4_qsa", "qwen4_qsa_sparse_attention_bf16_reduce")?,
            sparse_attention_bf16_reduce_gated: crate::layers::try_kernel(
                gpu,
                "qwen4_qsa",
                "qwen4_qsa_sparse_attention_bf16_reduce_gated",
            ),
            overlap: if overlap_enabled() {
                Some(QsaOverlap {
                    stream: gpu.create_stream()?,
                    fork: gpu.create_event()?,
                    join: gpu.create_event()?,
                })
            } else {
                None
            },
            cache: OnceLock::new(),
        })
    }

    /// Side stream + fork/join events when ATLAS_QWEN4_QSA_OVERLAP=1.
    pub(crate) fn overlap(&self) -> Option<QsaOverlap> {
        self.overlap
    }

    /// `SeqStateBlob` ranges of one physical block's side cache: its raw
    /// index-key ring rows and its compressed group keys. `create` builds the
    /// lazy cache (import into a fresh model); without it an uninitialized
    /// cache reports `None` (no request has written QSA state yet).
    pub(crate) fn seqstate_block_regions(
        &self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        block: u32,
        create: bool,
    ) -> Result<Option<[(DevicePtr, usize); 2]>> {
        let cache = if create {
            self.cache(gpu, kv_cache)?
        } else {
            match self.cache.get() {
                Some(cache) => cache,
                None => return Ok(None),
            }
        };
        let block = block as usize;
        ensure!(
            block < cache.num_main_blocks,
            "QSA seqstate block {block} outside {} cache blocks",
            cache.num_main_blocks
        );
        let raw_block = COMPRESS_RATIO * INDEX_DIM as usize * 2;
        let compressed_block = cache.block_size / COMPRESS_RATIO * INDEX_DIM as usize * 2;
        Ok(Some([
            (cache.raw_ring.offset(block * raw_block), raw_block),
            (
                cache.compressed_keys.offset(block * compressed_block),
                compressed_block,
            ),
        ]))
    }

    fn cache<'a>(&'a self, gpu: &dyn GpuBackend, kv_cache: &PagedKvCache) -> Result<&'a QsaCache> {
        if let Some(cache) = self.cache.get() {
            ensure!(
                cache.num_main_blocks == kv_cache.num_blocks()
                    && cache.block_size == kv_cache.block_size(),
                "QSA main-cache geometry changed after initialization"
            );
            return Ok(cache);
        }
        let block_size = kv_cache.block_size();
        ensure!(
            block_size >= COMPRESS_RATIO && block_size.is_multiple_of(COMPRESS_RATIO),
            "QSA requires KV block size divisible by {COMPRESS_RATIO}, got {block_size}"
        );
        let num_main_blocks = kv_cache.num_blocks();
        // Size from physical ownership, not the first request's logical block
        // table. A short first request must not permanently cap this OnceLock
        // and reject a later long-context request. Keep this shared with the
        // startup preflight; every individual allocation below comes from it.
        let layout = QsaCacheLayout::checked(num_main_blocks, block_size)?;
        // One journal lane per physical slot under ATLAS_QWEN4_BATCHED_K5;
        // exactly the historical single-lane extents otherwise.
        let journal_lanes = crate::layers::qwen4_verify_lanes::lanes();
        let lanes_of = |bytes: usize| {
            bytes
                .checked_mul(journal_lanes)
                .context("QSA journal lane extent overflow")
        };
        let journal_allocation_bytes = [
            lanes_of(layout.verify_undo)?,
            lanes_of(layout.verify_undo)?,
            lanes_of(layout.verify_offsets)?,
            lanes_of(layout.verify_offsets)?,
            lanes_of(layout.verify_valid_mask)?,
        ];
        let side_cache_allocation_bytes = [layout.raw_ring, layout.compressed_keys];
        let cache = QsaCache {
            raw_ring: gpu.alloc(side_cache_allocation_bytes[0])?,
            compressed_keys: gpu.alloc(side_cache_allocation_bytes[1])?,
            side_cache_allocation_bytes,
            verify_undo_raw: gpu.alloc(journal_allocation_bytes[0])?,
            verify_undo_compressed: gpu.alloc(journal_allocation_bytes[1])?,
            verify_undo_raw_offsets: gpu.alloc(journal_allocation_bytes[2])?,
            verify_undo_compressed_offsets: gpu.alloc(journal_allocation_bytes[3])?,
            verify_valid_mask: gpu.alloc(journal_allocation_bytes[4])?,
            journal_lanes,
            journal_undo_lane_bytes: layout.verify_undo,
            journal_offsets_lane_bytes: layout.verify_offsets,
            journal_mask_lane_bytes: layout.verify_valid_mask,
            journal_allocation_bytes,
            projected_qk: gpu.alloc(layout.projected_qk)?,
            pooled_key: gpu.alloc(layout.pooled_key)?,
            first_position: gpu.alloc(layout.first_position)?,
            logits: gpu.alloc(layout.logits)?,
            token_indices: gpu.alloc(layout.token_indices)?,
            reused_draft_indices: gpu.alloc(layout.reused_draft_indices)?,
            sparse_partial_output: gpu.alloc(layout.sparse_partial_output)?,
            sparse_partial_max: gpu.alloc(layout.sparse_partial_stat)?,
            sparse_partial_sum: gpu.alloc(layout.sparse_partial_stat)?,
            max_compressed_groups: layout.max_compressed_groups,
            num_main_blocks,
            block_size,
            precomputed_next_row: std::sync::atomic::AtomicUsize::new(0),
            precomputed_num_rows: std::sync::atomic::AtomicUsize::new(0),
            reused_snapshot_valid: std::sync::atomic::AtomicBool::new(false),
            reused_last_appended_step: std::sync::atomic::AtomicUsize::new(0),
        };
        // ATLAS_PREFILL_SCRATCH_INIT=poison fills the never-written groups with 0xFF so
        // a reader of a group nothing stored shows up as NaN; every other mode keeps zeros.
        let fill = spark_runtime::scratch_init::mode()?
            .arena_byte()
            .unwrap_or(0);
        gpu.memset_async(cache.raw_ring, fill, layout.raw_ring, gpu.default_stream())?;
        gpu.memset_async(
            cache.compressed_keys,
            fill,
            layout.compressed_keys,
            gpu.default_stream(),
        )?;
        gpu.memset_async(
            cache.verify_valid_mask,
            0,
            layout.verify_valid_mask * journal_lanes,
            gpu.default_stream(),
        )?;
        self.cache
            .set(cache)
            .map_err(|_| anyhow::anyhow!("QSA cache initialized concurrently"))?;
        tracing::info!(
            layer_cache_mib = (layout.raw_ring + layout.compressed_keys) as f64 / (1024.0 * 1024.0),
            max_compressed_groups = layout.max_compressed_groups,
            "Qwen4 QSA physical side cache initialized"
        );
        Ok(self.cache.get().expect("QSA cache just initialized"))
    }

    /// Journal the physical QSA destinations about to be overwritten by one
    /// row of an exact flat K5 verification frame. Row zero clears the mask in
    /// the same stream/graph as the snapshots, making captured replay safe.
    pub(crate) fn prepare_verify_k5_transaction(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        ensure!(
            self.snapshot_verify_k5_row.0 != 0
                && self.commit_verify_k5_prefix.0 != 0
                && self.commit_verify_prefix_rows.0 != 0,
            "QSA verify transaction kernels are unavailable"
        );
        self.cache(gpu, kv_cache)?;
        Ok(())
    }

    pub(crate) fn batch_verify_lane_ranges(
        &self,
        slot_idx: usize,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        writable_blocks: &[u32],
    ) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
        let configured_lanes = crate::layers::qwen4_verify_lanes::lanes();
        ensure!(
            configured_lanes > 1 && slot_idx < configured_lanes,
            "batch verify physical slot has no distinct configured journal lane"
        );
        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)?;
        self.prepare_verify_k5_transaction(kv_cache, gpu)?;
        let cache = self.cache(gpu, kv_cache)?;
        ensure!(
            cache.journal_lanes == configured_lanes,
            "QSA journal allocation belongs to a different lane registration"
        );
        let layout = QsaCacheLayout::checked(cache.num_main_blocks, cache.block_size)?;
        ensure!(
            cache.journal_undo_lane_bytes == layout.verify_undo
                && cache.journal_offsets_lane_bytes == layout.verify_offsets
                && cache.journal_mask_lane_bytes == layout.verify_valid_mask,
            "QSA journal allocated lane geometry differs from the physical cache layout"
        );
        ensure!(
            !writable_blocks.is_empty() && writable_blocks.len() <= VERIFY_MAX_ROWS,
            "QSA exact verify export requires one to {VERIFY_MAX_ROWS} physical writable blocks"
        );
        let block_bytes = [
            checked_qsa_product(&[COMPRESS_RATIO, INDEX_DIM as usize, 2], "raw block span")?,
            checked_qsa_product(
                &[cache.block_size / COMPRESS_RATIO, INDEX_DIM as usize, 2],
                "compressed block span",
            )?,
        ];
        let mut ranges = Vec::with_capacity(5 + 2 * writable_blocks.len());
        // The adapter proves these pages cover the five writes and are owned
        // exclusively. Export full physical blocks to include ring/pool writes.
        for (index, &block) in writable_blocks.iter().enumerate() {
            ensure!(
                !writable_blocks[..index].contains(&block),
                "QSA writable physical blocks repeat"
            );
            let block = usize::try_from(block)?;
            for (allocation_index, base) in [cache.raw_ring, cache.compressed_keys]
                .into_iter()
                .enumerate()
            {
                ranges.push(checked_batch_verify_lane_range(
                    base,
                    cache.side_cache_allocation_bytes[allocation_index],
                    block_bytes[allocation_index],
                    cache.num_main_blocks,
                    block,
                )?);
            }
        }
        let journals = [
            (cache.verify_undo_raw, cache.journal_undo_lane_bytes),
            (cache.verify_undo_compressed, cache.journal_undo_lane_bytes),
            (
                cache.verify_undo_raw_offsets,
                cache.journal_offsets_lane_bytes,
            ),
            (
                cache.verify_undo_compressed_offsets,
                cache.journal_offsets_lane_bytes,
            ),
            (cache.verify_valid_mask, cache.journal_mask_lane_bytes),
        ];
        let journal_ranges = journals
            .into_iter()
            .zip(cache.journal_allocation_bytes)
            .map(|((base, bytes), allocated)| {
                checked_batch_verify_lane_range(
                    base,
                    allocated,
                    bytes,
                    cache.journal_lanes,
                    slot_idx,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        ranges.extend(journal_ranges);
        distinct_batch_verify_lane_ranges(ranges)
    }

    /// Exact `(raw_ring_bytes, compressed_keys_bytes)` per physical KV block of `block_size`.
    pub fn swap_block_byte_sizes(block_size: usize) -> Result<(usize, usize)> {
        ensure!(
            block_size >= COMPRESS_RATIO && block_size.is_multiple_of(COMPRESS_RATIO),
            "QSA requires KV block size divisible by {COMPRESS_RATIO}, got {block_size}"
        );
        let raw_bytes =
            checked_qsa_product(&[COMPRESS_RATIO, INDEX_DIM as usize, 2], "raw swap span")?;
        let comp_bytes = checked_qsa_product(
            &[block_size / COMPRESS_RATIO, INDEX_DIM as usize, 2],
            "compressed swap span",
        )?;
        Ok((raw_bytes, comp_bytes))
    }

    /// Copy one physical block's QSA `raw_ring` and `compressed_keys` slices to host memory.
    pub fn read_swap_block(
        &self,
        block_idx: u32,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let cache = self.cache(gpu, kv_cache)?;
        let block = usize::try_from(block_idx)?;
        ensure!(
            block < cache.num_main_blocks,
            "QSA swap read block {block} out of range ({})",
            cache.num_main_blocks
        );
        let (raw_bytes, comp_bytes) = Self::swap_block_byte_sizes(cache.block_size)?;
        let mut raw_buf = vec![0u8; raw_bytes];
        let mut comp_buf = vec![0u8; comp_bytes];
        gpu.copy_d2h(cache.raw_ring.offset(block * raw_bytes), &mut raw_buf)?;
        gpu.copy_d2h(
            cache.compressed_keys.offset(block * comp_bytes),
            &mut comp_buf,
        )?;
        Ok((raw_buf, comp_buf))
    }

    /// Restore one physical block's QSA `raw_ring` and `compressed_keys` slices from host memory.
    pub fn write_swap_block(
        &self,
        block_idx: u32,
        raw_data: &[u8],
        comp_data: &[u8],
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let cache = self.cache(gpu, kv_cache)?;
        let block = usize::try_from(block_idx)?;
        ensure!(
            block < cache.num_main_blocks,
            "QSA swap write block {block} out of range ({})",
            cache.num_main_blocks
        );
        let (raw_bytes, comp_bytes) = Self::swap_block_byte_sizes(cache.block_size)?;
        ensure!(
            raw_data.len() == raw_bytes && comp_data.len() == comp_bytes,
            "QSA swap write byte length mismatch: got ({}, {}), expected ({raw_bytes}, {comp_bytes})",
            raw_data.len(),
            comp_data.len()
        );
        gpu.copy_h2d(raw_data, cache.raw_ring.offset(block * raw_bytes))?;
        gpu.copy_h2d(comp_data, cache.compressed_keys.offset(block * comp_bytes))?;
        Ok(())
    }

    pub(crate) fn snapshot_verify_k5_row(
        &self,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        row: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            row < VERIFY_MAX_ROWS,
            "QSA verify journal row {row} is out of range"
        );
        ensure!(
            self.snapshot_verify_k5_row.0 != 0,
            "QSA K5 transaction snapshot kernel is unavailable"
        );
        let cache = self.cache(gpu, kv_cache)?;
        let journal = cache.journal()?;
        KernelLaunch::new(gpu, self.snapshot_verify_k5_row)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(slot)
            .arg_ptr(journal.undo_raw)
            .arg_ptr(journal.undo_compressed)
            .arg_ptr(journal.undo_raw_offsets)
            .arg_ptr(journal.undo_compressed_offsets)
            .arg_ptr(journal.valid_mask)
            .arg_u32(row as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)
    }

    /// Commit the accepted prefix of a journaled `rows`-row frame (2..=9).
    ///
    /// When a [`QsaMultiSeqCommitGuard`] is active on the calling thread,
    /// dispatches ONE `qwen4_qsa_commit_verify_prefix_multi_seq` launch across
    /// all batched lanes (falling back to the per-lane loop if the multi-seq
    /// kernel handle is absent or a lane exceeds `cache.journal_lanes`).
    pub(crate) fn commit_verify_prefix(
        &self,
        accepted: usize,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let fast_batch = QSA_MULTI_SEQ_COMMIT_BATCH.with(|cell| {
            cell.borrow().as_ref().map(|b| {
                (
                    b.lanes_dev,
                    b.accepted_counts_dev,
                    b.row_counts_dev,
                    b.num_seqs,
                    b.max_lane,
                )
            })
        });
        if let Some((lanes_dev, accepted_counts_dev, row_counts_dev, num_seqs, max_lane)) =
            fast_batch
        {
            if num_seqs == 0 {
                return Ok(());
            }
            let Some(cache) = self.cache.get() else {
                return Ok(());
            };
            if self.commit_verify_prefix_multi_seq.0 != 0 && max_lane < cache.journal_lanes {
                return self.commit_verify_prefix_multi_seq(
                    lanes_dev,
                    accepted_counts_dev,
                    row_counts_dev,
                    num_seqs,
                    gpu,
                    stream,
                );
            }
            let suspend = QsaMultiSeqCommitSuspendGuard {
                saved: QSA_MULTI_SEQ_COMMIT_BATCH.with(|cell| cell.borrow_mut().take()),
            };
            if let Some(ref batch) = suspend.saved {
                for entry in &batch.fallback_entries {
                    let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(
                        entry.lane as usize,
                    )?;
                    self.commit_verify_prefix(
                        entry.accepted as usize,
                        entry.rows as usize,
                        gpu,
                        stream,
                    )?;
                }
            }
            return Ok(());
        }
        if rows == VERIFY_K5_ROWS {
            return self.commit_verify_k5_prefix(accepted, gpu, stream);
        }
        ensure!(
            (2..=VERIFY_MAX_ROWS).contains(&rows) && accepted <= rows,
            "QSA verify commit of {accepted} rows from a {rows}-row frame"
        );
        let Some(cache) = self.cache.get() else {
            return Ok(());
        };
        ensure!(
            self.commit_verify_prefix_rows.0 != 0,
            "QSA K-row transaction commit kernel is unavailable"
        );
        let journal = cache.journal()?;
        KernelLaunch::new(gpu, self.commit_verify_prefix_rows)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(journal.undo_raw)
            .arg_ptr(journal.undo_compressed)
            .arg_ptr(journal.undo_raw_offsets)
            .arg_ptr(journal.undo_compressed_offsets)
            .arg_ptr(journal.valid_mask)
            .arg_u32(accepted as u32)
            .arg_u32(rows as u32)
            .launch(stream)
    }

    /// Commit an accepted K5 prefix by restoring the rejected physical-cache
    /// suffix in reverse row order. A full accept still consumes/clears the
    /// journal. No initialized cache means this layer did not run QSA.
    pub(crate) fn commit_verify_k5_prefix(
        &self,
        accepted: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            accepted <= VERIFY_K5_ROWS,
            "QSA K5 accepted prefix {accepted} exceeds {VERIFY_K5_ROWS} rows"
        );
        let Some(cache) = self.cache.get() else {
            return Ok(());
        };
        ensure!(
            self.commit_verify_k5_prefix.0 != 0,
            "QSA K5 transaction commit kernel is unavailable"
        );
        let journal = cache.journal()?;
        KernelLaunch::new(gpu, self.commit_verify_k5_prefix)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(journal.undo_raw)
            .arg_ptr(journal.undo_compressed)
            .arg_ptr(journal.undo_raw_offsets)
            .arg_ptr(journal.undo_compressed_offsets)
            .arg_ptr(journal.valid_mask)
            .arg_u32(accepted as u32)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_and_select(
        &self,
        hidden: DevicePtr,
        position: usize,
        sequence_length: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            meta.num_seqs == 1,
            "QSA correctness path currently supports C=1"
        );
        ensure!(
            valid_physical_query(position, sequence_length),
            "QSA requires the current physical query position and nonzero u32 sequence length"
        );
        let cache = self.cache(gpu, kv_cache)?;
        let visible_groups = sequence_length / COMPRESS_RATIO;
        ensure!(
            visible_groups <= cache.max_compressed_groups,
            "QSA context requires {visible_groups} groups but cache holds {}",
            cache.max_compressed_groups
        );

        ops::dense_gemv(
            gpu,
            self.dense_gemv,
            hidden,
            &self.index_qk_proj,
            cache.projected_qk,
            INDEX_WIDTH,
            hidden_size,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            cache.projected_qk,
            &self.q_norm,
            cache.projected_qk,
            INDEX_HEADS,
            INDEX_DIM,
            eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_pool)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        self.apply_rope(
            cache.projected_qk,
            meta.positions,
            INDEX_HEADS,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        if (position + 1).is_multiple_of(COMPRESS_RATIO) {
            ops::rms_norm(
                gpu,
                self.rms_norm,
                cache.pooled_key,
                &self.k_norm,
                cache.pooled_key,
                1,
                INDEX_DIM,
                eps,
                stream,
            )?;
            self.apply_rope(
                cache.pooled_key,
                cache.first_position,
                1,
                rotary_dim,
                rope_theta,
                gpu,
                stream,
            )?;
            KernelLaunch::new(gpu, self.store_compressed)
                .grid([1, 1, 1])
                .block([INDEX_DIM, 1, 1])
                .arg_ptr(cache.pooled_key)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.slot)
                .arg_ptr(meta.positions)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        if visible_groups > 512 {
            let (score_kernel, groups_per_block) = if self.score_v2.0 != 0 {
                (self.score_v2, SCORE_V2_GROUPS_PER_BLOCK)
            } else {
                (self.score, 8)
            };
            KernelLaunch::new(gpu, score_kernel)
                .grid([visible_groups.div_ceil(groups_per_block) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(cache.projected_qk)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.block_table)
                .arg_ptr(cache.logits)
                .arg_u32(visible_groups as u32)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        let (select_kernel, select_threads) = if self.select_expand_par.0 != 0 {
            (self.select_expand_par, SELECT_PAR_THREADS)
        } else {
            (self.select_expand, 1)
        };
        KernelLaunch::new(gpu, select_kernel)
            .grid([1, 1, 1])
            .block([select_threads, 1, 1])
            .arg_ptr(cache.logits)
            .arg_ptr(cache.token_indices)
            .arg_u32(visible_groups as u32)
            .arg_u32(position as u32)
            .arg_u32(sequence_length as u32)
            .launch(stream)?;
        if matches!(
            DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
            Some(DraftReuseState::SnapshotStep0 { .. })
        ) {
            self.snapshot_indices_for_reuse(cache.token_indices, cache, gpu, stream)?;
            DRAFT_SELECTION_REUSE_STATE.with(|cell| {
                cell.set(Some(DraftReuseState::SnapshotStep0 {
                    snapshotted_in_update: true,
                    from_verify_rows: false,
                }));
            });
        }
        Ok(cache.token_indices)
    }

    /// Build the persistent QSA key-index history for a whole prefill chunk.
    ///
    /// Dense prefill does not consume the four query-index heads or selected
    /// token list. Projecting only the fifth (key) head with one GEMM avoids
    /// the per-token GEMV/norm/RoPE/select launch train while preserving the
    /// exact physical-page ownership used by decode.
    #[allow(clippy::too_many_arguments)]
    pub fn update_prefill_index(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        raw_keys: DevicePtr,
        pooled_keys: DevicePtr,
        first_positions: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(num_tokens > 0, "empty QSA prefill index update");
        ensure!(meta.num_seqs == 1, "QSA prefill index requires C=1");
        Self::validate_prefill_index_batch(num_tokens, seq_len_start)?;
        let cache = self.cache(gpu, kv_cache)?;
        let final_len = seq_len_start
            .checked_add(num_tokens)
            .ok_or_else(|| anyhow::anyhow!("QSA prefill length overflow"))?;
        ensure!(
            final_len / COMPRESS_RATIO <= cache.max_compressed_groups,
            "QSA prefill context exceeds side-cache capacity"
        );

        // index_qk_proj is [4 query heads | 1 key head, hidden_size].
        let key_weight = DenseWeight {
            weight: self
                .index_qk_proj
                .weight
                .offset(INDEX_HEADS as usize * INDEX_DIM as usize * hidden_size as usize * 2),
        };
        ops::dense_gemm(
            gpu,
            self.dense_gemm,
            hidden,
            &key_weight,
            raw_keys,
            num_tokens as u32,
            INDEX_DIM,
            hidden_size,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_prefill_raw)
            .grid([num_tokens as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(raw_keys)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(num_tokens as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;

        let first_endpoint =
            (COMPRESS_RATIO - 1 + COMPRESS_RATIO - seq_len_start % COMPRESS_RATIO) % COMPRESS_RATIO;
        if first_endpoint >= num_tokens {
            return Ok(());
        }
        let num_groups = 1 + (num_tokens - 1 - first_endpoint) / COMPRESS_RATIO;
        KernelLaunch::new(gpu, self.pool_prefill)
            .grid([num_groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(pooled_keys)
            .arg_ptr(first_positions)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(first_endpoint as u32)
            .arg_u32(num_groups as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            pooled_keys,
            &self.k_norm,
            pooled_keys,
            num_groups as u32,
            INDEX_DIM,
            eps,
            stream,
        )?;
        self.apply_rope_n(
            pooled_keys,
            first_positions,
            num_groups,
            1,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        KernelLaunch::new(gpu, self.store_prefill_compressed)
            .grid([num_groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(pooled_keys)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_u32(first_endpoint as u32)
            .arg_u32(num_groups as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rope(
        &self,
        tensor: DevicePtr,
        positions: DevicePtr,
        heads: u32,
        rotary_dim: u32,
        rope_theta: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.apply_rope_n(
            tensor, positions, 1, heads, rotary_dim, rope_theta, gpu, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rope_n(
        &self,
        tensor: DevicePtr,
        positions: DevicePtr,
        num_tokens: usize,
        heads: u32,
        rotary_dim: u32,
        rope_theta: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.yarn_inv_freq.is_null() {
            ops::rope(
                gpu,
                self.rope,
                tensor,
                DevicePtr::NULL,
                positions,
                num_tokens as u32,
                heads,
                0,
                INDEX_DIM,
                rotary_dim,
                rope_theta,
                stream,
            )
        } else {
            ops::rope_yarn_scaled(
                gpu,
                self.rope_yarn_scaled,
                tensor,
                DevicePtr::NULL,
                positions,
                num_tokens as u32,
                heads,
                0,
                INDEX_DIM,
                rotary_dim,
                self.yarn_inv_freq,
                rope_theta,
                self.yarn_attention_factor,
                stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_bf16(
        &self,
        query: DevicePtr,
        indices: DevicePtr,
        output: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.sparse_attention_bf16_gated(
            query,
            indices,
            output,
            kv_cache,
            meta,
            attn_layer_idx,
            num_query_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            None,
            gpu,
            stream,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_bf16_gated(
        &self,
        query: DevicePtr,
        indices: DevicePtr,
        output: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        gate: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let kv_dtype = kv_cache.dtype_for_layer(attn_layer_idx);
        ensure!(
            matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            "QSA supports BF16 or NVFP4 main KV cache, got {kv_dtype}"
        );
        ensure!(head_dim == 256, "released Qwen4 QSA expects head_dim=256");
        ensure!(
            num_query_heads > 0
                && (num_query_heads as usize) <= QUERY_HEADS
                && num_kv_heads > 0
                && num_query_heads.is_multiple_of(num_kv_heads),
            "Qwen4 QSA expects 1..={QUERY_HEADS} query heads divisible by kv heads ({num_kv_heads}), got {num_query_heads}"
        );
        let cache = self.cache(gpu, kv_cache)?;
        match DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get) {
            Some(DraftReuseState::SnapshotStep0 {
                snapshotted_in_update,
                from_verify_rows,
            }) => {
                if !from_verify_rows {
                    if !snapshotted_in_update && indices != cache.reused_draft_indices {
                        self.snapshot_indices_for_reuse(indices, cache, gpu, stream)?;
                    }
                    DRAFT_SELECTION_REUSE_STATE.with(|cell| {
                        cell.set(Some(DraftReuseState::SnapshotStep0 {
                            snapshotted_in_update: false,
                            from_verify_rows: false,
                        }));
                    });
                }
            }
            Some(DraftReuseState::ReuseStep {
                base_seq_len,
                step_offset,
                max_drafts,
            }) if indices == cache.reused_draft_indices => {
                let last_appended = cache
                    .reused_last_appended_step
                    .load(std::sync::atomic::Ordering::Acquire);
                for s in (last_appended + 1)..=step_offset {
                    let pos = base_seq_len
                        .checked_add(s)
                        .context("QSA reused draft position overflow")?;
                    let pos_scalar =
                        i32::try_from(pos).context("QSA reused draft position exceeds i32")?;
                    self.append_reused_draft_pos(
                        kv_cache, None, pos_scalar, s, max_drafts, gpu, stream,
                    )?;
                }
                let index_width = checked_reused_index_width(step_offset, max_drafts)?;
                return self.sparse_attention_reused_selection(
                    query,
                    output,
                    kv_cache,
                    meta,
                    attn_layer_idx,
                    num_query_heads,
                    num_kv_heads,
                    head_dim,
                    softmax_scale,
                    index_width,
                    gate,
                    gpu,
                    stream,
                );
            }
            _ => {}
        }
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(attn_layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(attn_layer_idx);
        let partial_kernel = match kv_dtype {
            KvCacheDtype::Bf16 if self.sparse_attention_bf16_partial_v2.0 != 0 => {
                self.sparse_attention_bf16_partial_v2
            }
            KvCacheDtype::Bf16 => self.sparse_attention_bf16_partial,
            KvCacheDtype::Nvfp4 => self.sparse_attention_nvfp4_partial,
            _ => unreachable!("dtype checked above"),
        };
        let mut launch = KernelLaunch::new(gpu, partial_kernel)
            .grid([num_query_heads, SPARSE_SPLITS as u32, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(query)
            .arg_ptr(kv_cache.k_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(kv_cache.v_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(indices)
            .arg_ptr(meta.block_table)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (k_stride / 2) as u64
            } else {
                k_stride as u64
            })
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (v_stride / 2) as u64
            } else {
                v_stride as u64
            });
        if kv_dtype == KvCacheDtype::Nvfp4 {
            launch = launch.arg_u64(kv_cache.nvfp4_data_bytes() as u64);
        }
        launch
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u32(num_query_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(head_dim)
            .arg_f32(softmax_scale)
            .arg_u32(SPARSE_SPLITS as u32)
            .launch(stream)?;
        if let Some(gate_ptr) = gate
            && qwen4_qsa_fused_gate_enabled()
            && self.sparse_attention_bf16_reduce_gated.0 != 0
        {
            KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce_gated)
                .grid([num_query_heads, 1, 1])
                .block([head_dim, 1, 1])
                .pdl(true)
                .arg_ptr(cache.sparse_partial_output)
                .arg_ptr(cache.sparse_partial_max)
                .arg_ptr(cache.sparse_partial_sum)
                .arg_ptr(gate_ptr)
                .arg_ptr(output)
                .arg_u32(num_query_heads)
                .arg_u32(head_dim)
                .arg_u32(SPARSE_SPLITS as u32)
                .launch(stream)?;
            return Ok(true);
        }
        KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce)
            .grid([num_query_heads, 1, 1])
            .block([head_dim, 1, 1])
            .pdl(true)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_ptr(output)
            .arg_u32(num_query_heads)
            .arg_u32(head_dim)
            .arg_u32(SPARSE_SPLITS as u32)
            .launch(stream)?;
        Ok(false)
    }

    fn has_reused_kernels(&self) -> bool {
        self.append_draft_pos.0 != 0
            && self.sparse_attention_bf16_partial_reused.0 != 0
            && self.sparse_attention_nvfp4_partial_reused.0 != 0
    }

    fn draft_selection_reuse_ready(&self) -> bool {
        if !self.has_reused_kernels() {
            return false;
        }
        if !matches!(
            DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
            Some(DraftReuseState::ReuseStep { .. })
        ) {
            return false;
        }
        self.cache.get().is_some_and(|cache| {
            cache
                .reused_snapshot_valid
                .load(std::sync::atomic::Ordering::Acquire)
        })
    }

    fn snapshot_indices_for_reuse(
        &self,
        src_indices: DevicePtr,
        cache: &QsaCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let base_bytes = OUTPUT_WIDTH * std::mem::size_of::<i32>();
        gpu.copy_d2d_async(src_indices, cache.reused_draft_indices, base_bytes, stream)?;
        if self.append_draft_pos.0 != 0 {
            KernelLaunch::new(gpu, self.append_draft_pos)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(cache.reused_draft_indices)
                .arg_ptr(DevicePtr::NULL)
                .arg_i32(-1)
                .arg_i32(0)
                .arg_i32(OUTPUT_WIDTH as i32)
                .arg_i32(0)
                .arg_i32(QSA_REUSE_MAX_DRAFTS as i32)
                .launch(stream)?;
        } else {
            gpu.memset_u32_async(
                cache.reused_draft_indices.offset(base_bytes),
                u32::MAX,
                QSA_REUSE_MAX_DRAFTS,
                stream,
            )?;
        }
        cache
            .reused_last_appended_step
            .store(0, std::sync::atomic::Ordering::Release);
        cache
            .reused_snapshot_valid
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Snapshot `token_indices[0..OUTPUT_WIDTH]` into `reused_draft_indices[0..OUTPUT_WIDTH]`
    /// via an async D2D copy and initialize tail slots `[OUTPUT_WIDTH..OUTPUT_WIDTH+16)` to `-1`.
    pub fn snapshot_selection_for_reuse(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let cache = self.cache(gpu, kv_cache)?;
        self.snapshot_indices_for_reuse(cache.token_indices, cache, gpu, stream)
    }

    /// Append a speculative draft token position at fixed slot
    /// `OUTPUT_WIDTH + (step_offset - 1)` in `reused_draft_indices` and set
    /// `[OUTPUT_WIDTH + step_offset .. OUTPUT_WIDTH + max_drafts)` to `-1`.
    /// Graph-safe (zero host syncs); reads `*pos_dev` when `pos_dev` is `Some`,
    /// otherwise uses `pos_scalar`.
    #[allow(clippy::too_many_arguments)]
    pub fn append_reused_draft_pos(
        &self,
        kv_cache: &PagedKvCache,
        pos_dev: Option<DevicePtr>,
        pos_scalar: i32,
        step_offset: usize,
        max_drafts: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            self.append_draft_pos.0 != 0,
            "QSA draft-selection reuse requires kernel qwen4_qsa_append_draft_pos"
        );
        ensure!(
            (1..=QSA_REUSE_MAX_DRAFTS).contains(&max_drafts),
            "QSA append_reused_draft_pos expects 1 <= max_drafts <= {QSA_REUSE_MAX_DRAFTS}, got {max_drafts}"
        );
        ensure!(
            (1..=max_drafts).contains(&step_offset),
            "QSA append_reused_draft_pos expects 1 <= step_offset <= {max_drafts}, got {step_offset}"
        );
        let cache = self.cache(gpu, kv_cache)?;
        KernelLaunch::new(gpu, self.append_draft_pos)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(cache.reused_draft_indices)
            .arg_ptr(pos_dev.unwrap_or(DevicePtr::NULL))
            .arg_i32(pos_scalar)
            .arg_i32(i32::from(pos_dev.is_some()))
            .arg_i32(OUTPUT_WIDTH as i32)
            .arg_i32(step_offset as i32)
            .arg_i32(max_drafts as i32)
            .launch(stream)?;
        cache
            .reused_last_appended_step
            .store(step_offset, std::sync::atomic::Ordering::Release);
        Ok(cache.reused_draft_indices)
    }

    /// Run split-token sparse attention over `reused_draft_indices[0..index_width]`
    /// (`OUTPUT_WIDTH <= index_width <= OUTPUT_WIDTH + QSA_REUSE_MAX_DRAFTS`) for
    /// MTP draft steps `1..K-1`, skipping `-1` holes between base selection tail
    /// and appended draft positions.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_reused_selection(
        &self,
        query: DevicePtr,
        output: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        index_width: usize,
        gate: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let kv_dtype = kv_cache.dtype_for_layer(attn_layer_idx);
        ensure!(
            matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            "QSA supports BF16 or NVFP4 main KV cache, got {kv_dtype}"
        );
        ensure!(head_dim == 256, "released Qwen4 QSA expects head_dim=256");
        ensure!(
            num_query_heads > 0
                && (num_query_heads as usize) <= QUERY_HEADS
                && num_kv_heads > 0
                && num_query_heads.is_multiple_of(num_kv_heads),
            "Qwen4 QSA expects 1..={QUERY_HEADS} query heads divisible by kv heads ({num_kv_heads}), got {num_query_heads}"
        );
        ensure!(
            (OUTPUT_WIDTH..=OUTPUT_WIDTH + QSA_REUSE_MAX_DRAFTS).contains(&index_width),
            "QSA reused sparse attention expects index_width in {OUTPUT_WIDTH}..={}, got {index_width}",
            OUTPUT_WIDTH + QSA_REUSE_MAX_DRAFTS
        );
        let cache = self.cache(gpu, kv_cache)?;
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(attn_layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(attn_layer_idx);
        let partial_kernel = match kv_dtype {
            KvCacheDtype::Bf16 if self.sparse_attention_bf16_partial_v2_reused.0 != 0 => {
                self.sparse_attention_bf16_partial_v2_reused
            }
            KvCacheDtype::Bf16 => {
                ensure!(
                    self.sparse_attention_bf16_partial_reused.0 != 0,
                    "QSA reused sparse attention requires kernel qwen4_qsa_sparse_attention_bf16_partial_reused"
                );
                self.sparse_attention_bf16_partial_reused
            }
            KvCacheDtype::Nvfp4 => {
                ensure!(
                    self.sparse_attention_nvfp4_partial_reused.0 != 0,
                    "QSA reused sparse attention requires kernel qwen4_qsa_sparse_attention_nvfp4_partial_reused"
                );
                self.sparse_attention_nvfp4_partial_reused
            }
            _ => unreachable!("dtype checked above"),
        };
        let mut launch = KernelLaunch::new(gpu, partial_kernel)
            .grid([num_query_heads, SPARSE_SPLITS as u32, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(query)
            .arg_ptr(kv_cache.k_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(kv_cache.v_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(cache.reused_draft_indices)
            .arg_ptr(meta.block_table)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (k_stride / 2) as u64
            } else {
                k_stride as u64
            })
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (v_stride / 2) as u64
            } else {
                v_stride as u64
            });
        if kv_dtype == KvCacheDtype::Nvfp4 {
            launch = launch.arg_u64(kv_cache.nvfp4_data_bytes() as u64);
        }
        launch
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u32(num_query_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(head_dim)
            .arg_f32(softmax_scale)
            .arg_u32(SPARSE_SPLITS as u32)
            .arg_i32(index_width as i32)
            .launch(stream)?;
        if let Some(gate_ptr) = gate
            && qwen4_qsa_fused_gate_enabled()
            && self.sparse_attention_bf16_reduce_gated.0 != 0
        {
            KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce_gated)
                .grid([num_query_heads, 1, 1])
                .block([head_dim, 1, 1])
                .pdl(true)
                .arg_ptr(cache.sparse_partial_output)
                .arg_ptr(cache.sparse_partial_max)
                .arg_ptr(cache.sparse_partial_sum)
                .arg_ptr(gate_ptr)
                .arg_ptr(output)
                .arg_u32(num_query_heads)
                .arg_u32(head_dim)
                .arg_u32(SPARSE_SPLITS as u32)
                .launch(stream)?;
            return Ok(true);
        }
        KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce)
            .grid([num_query_heads, 1, 1])
            .block([head_dim, 1, 1])
            .pdl(true)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_ptr(output)
            .arg_u32(num_query_heads)
            .arg_u32(head_dim)
            .arg_u32(SPARSE_SPLITS as u32)
            .launch(stream)?;
        Ok(false)
    }

    pub(crate) fn clear_precomputed_verify_rows(&self) {
        if let Some(cache) = self.cache.get() {
            cache
                .precomputed_next_row
                .store(0, std::sync::atomic::Ordering::Release);
            cache
                .precomputed_num_rows
                .store(0, std::sync::atomic::Ordering::Release);
        }
    }

    pub(crate) fn has_precomputed_verify_rows(&self) -> bool {
        let has_verify = self.cache.get().is_some_and(|cache| {
            cache
                .precomputed_next_row
                .load(std::sync::atomic::Ordering::Acquire)
                < cache
                    .precomputed_num_rows
                    .load(std::sync::atomic::Ordering::Acquire)
        });
        has_verify || self.draft_selection_reuse_ready()
    }

    pub(crate) fn take_precomputed_verify_row(&self) -> Option<DevicePtr> {
        let cache = self.cache.get()?;
        let num = cache
            .precomputed_num_rows
            .load(std::sync::atomic::Ordering::Acquire);
        if num == 0 {
            return self
                .draft_selection_reuse_ready()
                .then_some(cache.reused_draft_indices);
        }
        let row = cache
            .precomputed_next_row
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if row >= num {
            cache
                .precomputed_num_rows
                .store(0, std::sync::atomic::Ordering::Release);
            return None;
        }
        if row + 1 == num {
            cache
                .precomputed_num_rows
                .store(0, std::sync::atomic::Ordering::Release);
        }
        Some(
            cache
                .token_indices
                .offset(row * OUTPUT_WIDTH * std::mem::size_of::<i32>()),
        )
    }

    /// Run Stage A (journal snapshot + index QK projection + norm + ring stage/pool +
    /// RoPE + optional compressed key store) in row order `0..num_tokens`, then
    /// score all `num_tokens` rows in ONE pass over `compressed_keys` and select
    /// top-512 in ONE parallel `grid = [1, num_tokens, 1]` launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_and_select_verify_rows(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_len: usize,
        qsa_journal: bool,
        graph_capture: bool,
        device_plan: Option<crate::layers::qwen3_attention::Qwen4K5DeviceAttentionPlan>,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        self.clear_precomputed_verify_rows();
        if !(2..=VERIFY_MAX_ROWS).contains(&num_tokens)
            || !meta.qwen4_qsa_required
            || !verify_rows_enabled()
            || !verify_rows_context_safe(seq_len, num_tokens)
        {
            return Ok(false);
        }
        // Only use the fixed 262K max_compressed_groups graph-safe grid when
        // actually capturing a CUDA graph. On eager paths (including C>=2
        // cross_seq_verify), exact max_visible_groups avoids launching 4-6.5x
        // out-of-range CTAs per layer.
        let use_graph_safe = graph_capture && graph_safe_enabled();
        let use_device_meta = device_plan.is_some() || use_graph_safe;
        if use_device_meta {
            if self.score_device_rows_v2.0 == 0 || self.select_expand_device_rows_par.0 == 0 {
                return Ok(false);
            }
        } else if self.score_rows_v2.0 == 0 || self.select_expand_rows_par.0 == 0 {
            return Ok(false);
        }

        let cache = self.cache(gpu, kv_cache)?;
        if !use_device_meta {
            for row in 0..num_tokens {
                let pos = seq_len + row;
                let cur_len = pos + 1;
                ensure!(
                    valid_physical_query(pos, cur_len),
                    "QSA requires the current physical query position and nonzero u32 sequence length"
                );
                let vis = cur_len / COMPRESS_RATIO;
                ensure!(
                    vis <= cache.max_compressed_groups,
                    "QSA context requires {vis} groups but cache holds {}",
                    cache.max_compressed_groups
                );
            }
        }

        let core_bytes = hidden_size as usize * 2;
        let qk_row_bytes = INDEX_WIDTH as usize * 2;
        let projected_qk = if hidden_size == 0 {
            hidden
        } else {
            if self.dense_gemv_batchn.0 != 0 {
                ops::dense_gemv_batchn(
                    gpu,
                    self.dense_gemv_batchn,
                    hidden,
                    &self.index_qk_proj,
                    cache.projected_qk,
                    num_tokens as u32,
                    INDEX_WIDTH,
                    hidden_size,
                    stream,
                )?;
            }
            cache.projected_qk
        };
        if self.verify_rows_stage_pool_rope_store.0 != 0
            && (self.dense_gemv_batchn.0 != 0 || hidden_size == 0)
            && qsa_journal
            && verify_stage_pool_fused_enabled()
        {
            let journal = cache.journal()?;
            KernelLaunch::new(gpu, self.verify_rows_stage_pool_rope_store)
                .grid([1, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(projected_qk)
                .arg_ptr(cache.raw_ring)
                .arg_ptr(cache.pooled_key)
                .arg_ptr(cache.first_position)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(self.q_norm.weight)
                .arg_ptr(self.k_norm.weight)
                .arg_ptr(meta.slot)
                .arg_ptr(meta.positions)
                .arg_ptr(journal.undo_raw)
                .arg_ptr(journal.undo_compressed)
                .arg_ptr(journal.undo_raw_offsets)
                .arg_ptr(journal.undo_compressed_offsets)
                .arg_ptr(journal.valid_mask)
                .arg_ptr(self.yarn_inv_freq)
                .arg_f32(rope_theta)
                .arg_f32(self.yarn_attention_factor)
                .arg_f32(eps)
                .arg_u32(rotary_dim)
                .arg_u32(cache.block_size as u32)
                .arg_u32(num_tokens as u32)
                .launch(stream)?;
        } else {
            for row in 0..num_tokens {
                let row_slot = meta.slot.offset(row * 8);
                let row_pos = meta.positions.offset(row * 4);
                if qsa_journal {
                    self.snapshot_verify_k5_row(kv_cache, row_slot, row, gpu, stream)?;
                }
                let row_hidden = hidden.offset(row * core_bytes);
                let row_qk = projected_qk.offset(row * qk_row_bytes);
                if self.dense_gemv_batchn.0 == 0 && hidden_size != 0 {
                    ops::dense_gemv(
                        gpu,
                        self.dense_gemv,
                        row_hidden,
                        &self.index_qk_proj,
                        row_qk,
                        INDEX_WIDTH,
                        hidden_size,
                        stream,
                    )?;
                }
                ops::rms_norm(
                    gpu,
                    self.rms_norm,
                    row_qk,
                    &self.q_norm,
                    row_qk,
                    INDEX_HEADS,
                    INDEX_DIM,
                    eps,
                    stream,
                )?;
                KernelLaunch::new(gpu, self.stage_pool)
                    .grid([1, 1, 1])
                    .block([INDEX_DIM, 1, 1])
                    .arg_ptr(row_qk)
                    .arg_ptr(cache.raw_ring)
                    .arg_ptr(cache.pooled_key)
                    .arg_ptr(cache.first_position)
                    .arg_ptr(row_slot)
                    .arg_ptr(row_pos)
                    .arg_u32(cache.block_size as u32)
                    .launch(stream)?;
                self.apply_rope(
                    row_qk,
                    row_pos,
                    INDEX_HEADS,
                    rotary_dim,
                    rope_theta,
                    gpu,
                    stream,
                )?;
                let pool_endpoint = if let Some(plan) = device_plan {
                    plan.row(row)?.qsa_pool_endpoint
                } else if use_graph_safe {
                    true
                } else {
                    (seq_len + row + 1).is_multiple_of(COMPRESS_RATIO)
                };
                if pool_endpoint {
                    ops::rms_norm(
                        gpu,
                        self.rms_norm,
                        cache.pooled_key,
                        &self.k_norm,
                        cache.pooled_key,
                        1,
                        INDEX_DIM,
                        eps,
                        stream,
                    )?;
                    self.apply_rope(
                        cache.pooled_key,
                        cache.first_position,
                        1,
                        rotary_dim,
                        rope_theta,
                        gpu,
                        stream,
                    )?;
                    KernelLaunch::new(gpu, self.store_compressed)
                        .grid([1, 1, 1])
                        .block([INDEX_DIM, 1, 1])
                        .arg_ptr(cache.pooled_key)
                        .arg_ptr(cache.compressed_keys)
                        .arg_ptr(row_slot)
                        .arg_ptr(row_pos)
                        .arg_u32(cache.block_size as u32)
                        .launch(stream)?;
                }
            }
        }

        let logits_row_stride = cache.max_compressed_groups as u64;
        if let Some(plan) = device_plan {
            let mut max_grid_x = 0u32;
            for r in 0..num_tokens {
                max_grid_x = max_grid_x.max(plan.row(r)?.qsa_score_grid_x);
            }
            if max_grid_x != 0 {
                let grid_x = max_grid_x.div_ceil(4).max(1);
                KernelLaunch::new(gpu, self.score_device_rows_v2)
                    .grid([grid_x, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(projected_qk)
                    .arg_u32(INDEX_WIDTH)
                    .arg_ptr(cache.compressed_keys)
                    .arg_ptr(meta.block_table)
                    .arg_ptr(cache.logits)
                    .arg_u64(logits_row_stride)
                    .arg_ptr(meta.seq_len)
                    .arg_u32(num_tokens as u32)
                    .arg_u32(cache.block_size as u32)
                    .launch(stream)?;
            }
            KernelLaunch::new(gpu, self.select_expand_device_rows_par)
                .grid([1, num_tokens as u32, 1])
                .block([SELECT_PAR_THREADS, 1, 1])
                .arg_ptr(cache.logits)
                .arg_u64(logits_row_stride)
                .arg_ptr(cache.token_indices)
                .arg_ptr(meta.positions)
                .arg_ptr(meta.seq_len)
                .launch(stream)?;
        } else if use_graph_safe
            && (seq_len >= VERIFY_ROWS_DENSE_ENDPOINT
                || (seq_len + num_tokens) / COMPRESS_RATIO > 512)
        {
            let score_grid = u32::try_from(
                cache
                    .max_compressed_groups
                    .div_ceil(SCORE_V2_GROUPS_PER_BLOCK),
            )?;
            KernelLaunch::new(gpu, self.score_device_rows_v2)
                .grid([score_grid.max(1), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(projected_qk)
                .arg_u32(INDEX_WIDTH)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.block_table)
                .arg_ptr(cache.logits)
                .arg_u64(logits_row_stride)
                .arg_ptr(meta.seq_len)
                .arg_u32(num_tokens as u32)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.select_expand_device_rows_par)
                .grid([1, num_tokens as u32, 1])
                .block([SELECT_PAR_THREADS, 1, 1])
                .arg_ptr(cache.logits)
                .arg_u64(logits_row_stride)
                .arg_ptr(cache.token_indices)
                .arg_ptr(meta.positions)
                .arg_ptr(meta.seq_len)
                .launch(stream)?;
        } else {
            let base_seq_len = seq_len + 1;
            let max_visible_groups = (seq_len + num_tokens) / COMPRESS_RATIO;
            if seq_len >= VERIFY_ROWS_DENSE_ENDPOINT || max_visible_groups > 512 {
                KernelLaunch::new(gpu, self.score_rows_v2)
                    .grid([
                        max_visible_groups.div_ceil(SCORE_V2_GROUPS_PER_BLOCK) as u32,
                        1,
                        1,
                    ])
                    .block([256, 1, 1])
                    .arg_ptr(projected_qk)
                    .arg_u32(INDEX_WIDTH)
                    .arg_ptr(cache.compressed_keys)
                    .arg_ptr(meta.block_table)
                    .arg_ptr(cache.logits)
                    .arg_u64(logits_row_stride)
                    .arg_u32(base_seq_len as u32)
                    .arg_u32(num_tokens as u32)
                    .arg_u32(cache.block_size as u32)
                    .launch(stream)?;
                KernelLaunch::new(gpu, self.select_expand_rows_par)
                    .grid([1, num_tokens as u32, 1])
                    .block([SELECT_PAR_THREADS, 1, 1])
                    .arg_ptr(cache.logits)
                    .arg_u64(logits_row_stride)
                    .arg_ptr(cache.token_indices)
                    .arg_u32(seq_len as u32)
                    .arg_u32(base_seq_len as u32)
                    .launch(stream)?;
            }
        }

        if matches!(
            DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
            Some(DraftReuseState::SnapshotStep0 { .. })
        ) {
            let last_row = num_tokens.saturating_sub(1);
            let src = cache
                .token_indices
                .offset(last_row * OUTPUT_WIDTH * std::mem::size_of::<i32>());
            self.snapshot_indices_for_reuse(src, cache, gpu, stream)?;
            DRAFT_SELECTION_REUSE_STATE.with(|cell| {
                cell.set(Some(DraftReuseState::SnapshotStep0 {
                    snapshotted_in_update: true,
                    from_verify_rows: true,
                }));
            });
        }

        cache
            .precomputed_next_row
            .store(0, std::sync::atomic::Ordering::Release);
        cache
            .precomputed_num_rows
            .store(num_tokens, std::sync::atomic::Ordering::Release);
        Ok(true)
    }

    /// Project `rows` contiguous attention inputs through `index_qk_proj` in one
    /// single-pass `dense_gemv_batchn` launch, returning `(output, row_bytes)`.
    #[allow(dead_code)]
    pub(crate) fn project_index_qk_rows(
        &self,
        hidden: DevicePtr,
        output: DevicePtr,
        rows: usize,
        hidden_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        if self.dense_gemv_batchn.0 == 0 || rows == 0 {
            return Ok(None);
        }
        ops::dense_gemv_batchn(
            gpu,
            self.dense_gemv_batchn,
            hidden,
            &self.index_qk_proj,
            output,
            rows as u32,
            INDEX_WIDTH,
            hidden_size,
            stream,
        )?;
        Ok(Some((output, INDEX_WIDTH as usize * 2)))
    }

    pub(crate) fn has_sparse_attention_rows_for_dtype(&self, kv_dtype: KvCacheDtype) -> bool {
        let has_partial = match kv_dtype {
            KvCacheDtype::Bf16 => {
                self.sparse_attention_bf16_partial_rows_v2.0 != 0
                    || self.sparse_attention_bf16_partial_rows.0 != 0
            }
            KvCacheDtype::Nvfp4 => {
                self.sparse_attention_nvfp4_partial_rows_v2.0 != 0
                    || self.sparse_attention_nvfp4_partial_rows.0 != 0
            }
            _ => false,
        };
        has_partial && self.sparse_attention_bf16_reduce_rows.0 != 0
    }

    pub(crate) fn can_fuse_sparse_rows_gate(&self) -> bool {
        qwen4_qsa_fused_gate_enabled() && self.sparse_attention_bf16_reduce_gated_rows.0 != 0
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_bf16_rows(
        &self,
        query: DevicePtr,
        query_row_stride: u32,
        output: DevicePtr,
        num_rows: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        gate: Option<(DevicePtr, u32)>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<SparseRowsKernel> {
        let kv_dtype = kv_cache.dtype_for_layer(attn_layer_idx);
        ensure!(
            matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            "QSA multi-row sparse attention requires BF16 or NVFP4 main KV cache, got {kv_dtype}"
        );
        ensure!(
            (1..=VERIFY_MAX_ROWS).contains(&num_rows),
            "QSA multi-row sparse attention expects 1..={VERIFY_MAX_ROWS} rows, got {num_rows}"
        );
        ensure!(head_dim == 256, "released Qwen4 QSA expects head_dim=256");
        ensure!(
            num_query_heads > 0
                && (num_query_heads as usize) <= QUERY_HEADS
                && num_kv_heads > 0
                && num_query_heads.is_multiple_of(num_kv_heads),
            "Qwen4 QSA expects 1..={QUERY_HEADS} query heads divisible by kv heads ({num_kv_heads}), got {num_query_heads}"
        );
        ensure!(
            self.has_sparse_attention_rows_for_dtype(kv_dtype),
            "QSA multi-row sparse attention kernels are unavailable for {kv_dtype}"
        );
        let cache = self.cache(gpu, kv_cache)?;
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(attn_layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(attn_layer_idx);
        let (partial_kernel, selected_kernel) = match kv_dtype {
            KvCacheDtype::Bf16 if self.sparse_attention_bf16_partial_rows_v2.0 != 0 => (
                self.sparse_attention_bf16_partial_rows_v2,
                SparseRowsKernel::V2,
            ),
            KvCacheDtype::Bf16 => (
                self.sparse_attention_bf16_partial_rows,
                SparseRowsKernel::Legacy,
            ),
            KvCacheDtype::Nvfp4 if self.sparse_attention_nvfp4_partial_rows_v2.0 != 0 => (
                self.sparse_attention_nvfp4_partial_rows_v2,
                SparseRowsKernel::V2,
            ),
            KvCacheDtype::Nvfp4 => (
                self.sparse_attention_nvfp4_partial_rows,
                SparseRowsKernel::Legacy,
            ),
            _ => unreachable!("dtype checked above"),
        };
        let mut launch = KernelLaunch::new(gpu, partial_kernel)
            .grid([num_query_heads, SPARSE_SPLITS as u32, num_rows as u32])
            .block([head_dim, 1, 1])
            .arg_ptr(query)
            .arg_ptr(kv_cache.k_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(kv_cache.v_cache_ptr(attn_layer_idx, 0))
            .arg_ptr(cache.token_indices)
            .arg_ptr(meta.block_table)
            .arg_ptr(cache.sparse_partial_output)
            .arg_ptr(cache.sparse_partial_max)
            .arg_ptr(cache.sparse_partial_sum)
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (k_stride / 2) as u64
            } else {
                k_stride as u64
            })
            .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                (v_stride / 2) as u64
            } else {
                v_stride as u64
            });
        if kv_dtype == KvCacheDtype::Nvfp4 {
            launch = launch.arg_u64(kv_cache.nvfp4_data_bytes() as u64);
        }
        launch
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u32(num_query_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(head_dim)
            .arg_f32(softmax_scale)
            .arg_u32(SPARSE_SPLITS as u32)
            .arg_u32(query_row_stride)
            .arg_u32(meta.max_blocks_per_seq)
            .launch(stream)?;
        if let Some((gate_ptr, gate_stride)) = gate
            && self.can_fuse_sparse_rows_gate()
        {
            KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce_gated_rows)
                .grid([num_query_heads, num_rows as u32, 1])
                .block([head_dim, 1, 1])
                .pdl(num_rows <= 64)
                .arg_ptr(cache.sparse_partial_output)
                .arg_ptr(cache.sparse_partial_max)
                .arg_ptr(cache.sparse_partial_sum)
                .arg_ptr(gate_ptr)
                .arg_ptr(output)
                .arg_u32(num_query_heads)
                .arg_u32(head_dim)
                .arg_u32(SPARSE_SPLITS as u32)
                .arg_u32(gate_stride)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, self.sparse_attention_bf16_reduce_rows)
                .grid([num_query_heads, num_rows as u32, 1])
                .block([head_dim, 1, 1])
                .pdl(num_rows <= 64)
                .arg_ptr(cache.sparse_partial_output)
                .arg_ptr(cache.sparse_partial_max)
                .arg_ptr(cache.sparse_partial_sum)
                .arg_ptr(output)
                .arg_u32(num_query_heads)
                .arg_u32(head_dim)
                .arg_u32(SPARSE_SPLITS as u32)
                .launch(stream)?;
        }
        Ok(selected_kernel)
    }

    /// Whether all multi-sequence QSA verify kernels needed for `kv_dtype`
    /// (and `masked` when the frame mixes dense and sparse sequences) are loaded.
    pub(crate) fn has_multi_seq_verify_kernels(
        &self,
        kv_dtype: KvCacheDtype,
        any_sparse: bool,
        masked: bool,
    ) -> bool {
        if self.verify_rows_stage_pool_rope_store_multi_seq.0 == 0 {
            return false;
        }
        if !any_sparse {
            return true;
        }
        if self.score_multi_seq_rows_v2.0 == 0 || self.select_expand_multi_seq_rows.0 == 0 {
            return false;
        }
        if masked {
            let has_partial = match kv_dtype {
                KvCacheDtype::Bf16 => self.sparse_attention_bf16_partial_rows_v2_masked.0 != 0,
                KvCacheDtype::Nvfp4 => self.sparse_attention_nvfp4_partial_rows_v2_masked.0 != 0,
                _ => false,
            };
            has_partial
                && (self.sparse_attention_bf16_reduce_gated_rows_masked.0 != 0
                    || self.sparse_attention_bf16_reduce_rows_masked.0 != 0)
        } else {
            let has_partial = match kv_dtype {
                KvCacheDtype::Bf16 => {
                    self.sparse_attention_bf16_partial_rows_v2.0 != 0
                        || self.sparse_attention_bf16_partial_rows_v2_masked.0 != 0
                        || self.sparse_attention_bf16_partial_rows.0 != 0
                }
                KvCacheDtype::Nvfp4 => {
                    self.sparse_attention_nvfp4_partial_rows_v2.0 != 0
                        || self.sparse_attention_nvfp4_partial_rows_v2_masked.0 != 0
                        || self.sparse_attention_nvfp4_partial_rows.0 != 0
                }
                _ => false,
            };
            has_partial
                && (self.sparse_attention_bf16_reduce_gated_rows.0 != 0
                    || self.sparse_attention_bf16_reduce_rows.0 != 0
                    || self.sparse_attention_bf16_reduce_gated_rows_masked.0 != 0
                    || self.sparse_attention_bf16_reduce_rows_masked.0 != 0)
        }
    }

    /// Run multi-sequence QSA stage/pool/RoPE/store (1 launch across `num_seqs`),
    /// packed 1D multi-sequence scoring (`total_score_blocks` CTAs), and parallel
    /// top-512 select-expand (`grid=[max_k, num_seqs, 1]`) into `scratch`.
    /// Supports both uniform and ragged mixed-`K` frames (`1..=VERIFY_MAX_ROWS`,
    /// `total_rows <= 128`); each sequence's actual `k_s` and `row_base` live in
    /// `seq_descs_dev[s]`, and `k_s == 1` slots carry `QSA_DESC_FLAG_JOURNAL = 0`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_and_select_verify_multi_seq(
        &self,
        projected_qk: DevicePtr,
        num_seqs: usize,
        max_k: usize,
        total_rows: usize,
        total_score_blocks: u32,
        max_visible_groups: usize,
        any_journal: bool,
        max_journal_lane: usize,
        seq_descs_dev: DevicePtr,
        scratch: Qwen4CrossSeqQsaScratch,
        kv_cache: &PagedKvCache,
        frame_meta: AttnMetadataDev,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=QSA_CROSS_SEQ_MAX_SEQS).contains(&num_seqs),
            "multi-seq QSA verify expects 1..={QSA_CROSS_SEQ_MAX_SEQS} sequences, got {num_seqs}"
        );
        ensure!(
            (1..=VERIFY_MAX_ROWS).contains(&max_k),
            "multi-seq QSA verify expects K1..=K{VERIFY_MAX_ROWS}, got K{max_k}"
        );
        ensure!(
            total_rows >= num_seqs && total_rows <= crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS,
            "multi-seq QSA verify total_rows {total_rows} outside [{num_seqs}, {}]",
            crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS
        );
        ensure!(
            !seq_descs_dev.is_null() && scratch.is_allocated(),
            "multi-seq QSA verify requires allocated descriptor and scratch buffers"
        );
        ensure!(
            self.verify_rows_stage_pool_rope_store_multi_seq.0 != 0,
            "qwen4_qsa_verify_rows_stage_pool_rope_store_multi_seq kernel is unavailable"
        );
        let cache = self.cache(gpu, kv_cache)?;
        if total_score_blocks > 0 {
            ensure!(
                max_visible_groups <= cache.max_compressed_groups,
                "multi-seq QSA requires {max_visible_groups} compressed groups but cache holds {}",
                cache.max_compressed_groups
            );
        }
        if any_journal {
            ensure!(
                max_journal_lane < cache.journal_lanes,
                "multi-seq QSA journal lane {max_journal_lane} exceeds allocated lanes {}",
                cache.journal_lanes
            );
            ensure!(
                cache.journal_undo_lane_bytes == VERIFY_MAX_ROWS * INDEX_DIM as usize * 2
                    && cache.journal_offsets_lane_bytes
                        == VERIFY_MAX_ROWS * std::mem::size_of::<u64>()
                    && cache.journal_mask_lane_bytes == std::mem::size_of::<u32>(),
                "multi-seq QSA journal lane layout mismatch"
            );
        }

        let (undo_raw, undo_comp, undo_raw_off, undo_comp_off, valid_mask) = if any_journal {
            (
                cache.verify_undo_raw,
                cache.verify_undo_compressed,
                cache.verify_undo_raw_offsets,
                cache.verify_undo_compressed_offsets,
                cache.verify_valid_mask,
            )
        } else {
            (
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
                DevicePtr::NULL,
            )
        };
        KernelLaunch::new(gpu, self.verify_rows_stage_pool_rope_store_multi_seq)
            .grid([num_seqs as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(self.q_norm.weight)
            .arg_ptr(self.k_norm.weight)
            .arg_ptr(frame_meta.slot)
            .arg_ptr(frame_meta.positions)
            .arg_ptr(undo_raw)
            .arg_ptr(undo_comp)
            .arg_ptr(undo_raw_off)
            .arg_ptr(undo_comp_off)
            .arg_ptr(valid_mask)
            .arg_ptr(seq_descs_dev)
            .arg_ptr(self.yarn_inv_freq)
            .arg_f32(rope_theta)
            .arg_f32(self.yarn_attention_factor)
            .arg_f32(eps)
            .arg_u32(rotary_dim)
            .arg_u32(cache.block_size as u32)
            .arg_u32(num_seqs as u32)
            .launch(stream)?;

        if total_score_blocks > 0 {
            ensure!(
                self.score_multi_seq_rows_v2.0 != 0 && self.select_expand_multi_seq_rows.0 != 0,
                "multi-seq QSA score/select kernels are unavailable"
            );
            KernelLaunch::new(gpu, self.score_multi_seq_rows_v2)
                .grid([total_score_blocks, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(projected_qk)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(frame_meta.block_table)
                .arg_ptr(scratch.batch_logits)
                .arg_ptr(seq_descs_dev)
                .arg_u32(num_seqs as u32)
                .arg_u32(frame_meta.max_blocks_per_seq)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.select_expand_multi_seq_rows)
                .grid([max_k as u32, num_seqs as u32, 1])
                .block([SELECT_PAR_THREADS, 1, 1])
                .arg_ptr(scratch.batch_logits)
                .arg_ptr(scratch.batch_token_indices)
                .arg_ptr(seq_descs_dev)
                .arg_u32(num_seqs as u32)
                .launch(stream)?;
        }
        Ok(())
    }

    /// Multi-sequence batched QSA sparse attention over up to
    /// `QWEN4_CROSS_SEQ_MAX_ROWS` (`256`) rows using the model-wide shared
    /// [`Qwen4CrossSeqQsaScratch`] arena.
    ///
    /// When `row_mask` is `Some([m0, m1, m2, m3])` (or when falling back to a
    /// masked 128-bit CUDA kernel for `num_rows > 128`), launches are chunked
    /// cleanly across `0..min(num_rows, 128)` and `128..num_rows`, skipping any
    /// half whose 128-bit mask is all zeros.
    ///
    /// Returns `Ok(true)` if output sigmoid gating was fused inside the reduce
    /// kernel, or `Ok(false)` if the caller must apply output gating separately.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attention_multi_seq_rows(
        &self,
        query: DevicePtr,
        query_row_stride: u32,
        output: DevicePtr,
        num_rows: usize,
        row_mask: Option<[u64; 4]>,
        scratch: Qwen4CrossSeqQsaScratch,
        kv_cache: &PagedKvCache,
        frame_meta: AttnMetadataDev,
        attn_layer_idx: usize,
        num_query_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        softmax_scale: f32,
        gate: Option<(DevicePtr, u32)>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        let kv_dtype = kv_cache.dtype_for_layer(attn_layer_idx);
        ensure!(
            matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            "multi-seq QSA sparse attention requires BF16 or NVFP4 KV cache, got {kv_dtype}"
        );
        ensure!(
            (1..=crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS).contains(&num_rows),
            "multi-seq QSA sparse attention expects 1..={} rows, got {num_rows}",
            crate::layer::QWEN4_CROSS_SEQ_MAX_ROWS
        );
        ensure!(
            scratch.is_allocated(),
            "multi-seq QSA sparse attention requires allocated cross-seq QSA scratch"
        );
        ensure!(head_dim == 256, "released Qwen4 QSA expects head_dim=256");
        ensure!(
            num_query_heads > 0
                && (num_query_heads as usize) <= QUERY_HEADS
                && num_kv_heads > 0
                && num_query_heads.is_multiple_of(num_kv_heads),
            "Qwen4 QSA expects 1..={QUERY_HEADS} query heads divisible by kv heads ({num_kv_heads}), got {num_query_heads}"
        );

        let k_stride = kv_cache.k_block_stride_bytes_for_layer(attn_layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(attn_layer_idx);
        let (partial_kernel, partial_masked) = match (kv_dtype, row_mask.is_some()) {
            (KvCacheDtype::Bf16, true) => (self.sparse_attention_bf16_partial_rows_v2_masked, true),
            (KvCacheDtype::Nvfp4, true) => {
                (self.sparse_attention_nvfp4_partial_rows_v2_masked, true)
            }
            (KvCacheDtype::Bf16, false) if self.sparse_attention_bf16_partial_rows_v2.0 != 0 => {
                (self.sparse_attention_bf16_partial_rows_v2, false)
            }
            (KvCacheDtype::Bf16, false)
                if self.sparse_attention_bf16_partial_rows_v2_masked.0 != 0 =>
            {
                (self.sparse_attention_bf16_partial_rows_v2_masked, true)
            }
            (KvCacheDtype::Bf16, false) => (self.sparse_attention_bf16_partial_rows, false),
            (KvCacheDtype::Nvfp4, false) if self.sparse_attention_nvfp4_partial_rows_v2.0 != 0 => {
                (self.sparse_attention_nvfp4_partial_rows_v2, false)
            }
            (KvCacheDtype::Nvfp4, false)
                if self.sparse_attention_nvfp4_partial_rows_v2_masked.0 != 0 =>
            {
                (self.sparse_attention_nvfp4_partial_rows_v2_masked, true)
            }
            (KvCacheDtype::Nvfp4, false) => (self.sparse_attention_nvfp4_partial_rows, false),
            _ => unreachable!("dtype checked above"),
        };
        ensure!(
            partial_kernel.0 != 0,
            "multi-seq QSA sparse attention partial kernel unavailable for {kv_dtype}"
        );

        let (reduce_k, reduce_masked, fused_gate) =
            if gate.is_some() && qwen4_qsa_fused_gate_enabled() {
                let (gated_k, gated_masked) = if row_mask.is_some() {
                    (self.sparse_attention_bf16_reduce_gated_rows_masked, true)
                } else if self.sparse_attention_bf16_reduce_gated_rows.0 != 0 {
                    (self.sparse_attention_bf16_reduce_gated_rows, false)
                } else {
                    (self.sparse_attention_bf16_reduce_gated_rows_masked, true)
                };
                if gated_k.0 != 0 {
                    (gated_k, gated_masked, true)
                } else {
                    let (plain_k, plain_masked) = if row_mask.is_some() {
                        (self.sparse_attention_bf16_reduce_rows_masked, true)
                    } else if self.sparse_attention_bf16_reduce_rows.0 != 0 {
                        (self.sparse_attention_bf16_reduce_rows, false)
                    } else {
                        (self.sparse_attention_bf16_reduce_rows_masked, true)
                    };
                    (plain_k, plain_masked, false)
                }
            } else {
                let (plain_k, plain_masked) = if row_mask.is_some() {
                    (self.sparse_attention_bf16_reduce_rows_masked, true)
                } else if self.sparse_attention_bf16_reduce_rows.0 != 0 {
                    (self.sparse_attention_bf16_reduce_rows, false)
                } else {
                    (self.sparse_attention_bf16_reduce_rows_masked, true)
                };
                (plain_k, plain_masked, false)
            };
        ensure!(
            reduce_k.0 != 0,
            "multi-seq QSA sparse attention reduce kernel unavailable"
        );

        let (chunks, n_chunks) =
            plan_qsa_sparse_row_chunks(num_rows, row_mask, partial_masked || reduce_masked);
        let q_row_bytes = (query_row_stride as usize) * 2;
        let out_row_bytes = (num_query_heads as usize) * (head_dim as usize) * 2;
        let idx_row_bytes = OUTPUT_WIDTH * std::mem::size_of::<i32>();
        let bt_row_bytes = (frame_meta.max_blocks_per_seq as usize) * std::mem::size_of::<i32>();
        let parts_per_row = (num_query_heads as usize) * SPARSE_SPLITS;
        for &(chunk_base, chunk_rows, mask_lo, mask_hi) in &chunks[..n_chunks] {
            let row_parts = chunk_base * parts_per_row;
            let chunk_q = query.offset(chunk_base * q_row_bytes);
            let chunk_out = output.offset(chunk_base * out_row_bytes);
            let chunk_idx = scratch
                .batch_token_indices
                .offset(chunk_base * idx_row_bytes);
            let chunk_bt = frame_meta.block_table.offset(chunk_base * bt_row_bytes);
            let chunk_po = scratch
                .batch_partial_output
                .offset(row_parts * (head_dim as usize) * std::mem::size_of::<f32>());
            let chunk_pm = scratch
                .batch_partial_max
                .offset(row_parts * std::mem::size_of::<f32>());
            let chunk_ps = scratch
                .batch_partial_sum
                .offset(row_parts * std::mem::size_of::<f32>());

            let mut launch = KernelLaunch::new(gpu, partial_kernel)
                .grid([num_query_heads, SPARSE_SPLITS as u32, chunk_rows as u32])
                .block([head_dim, 1, 1])
                .arg_ptr(chunk_q)
                .arg_ptr(kv_cache.k_cache_ptr(attn_layer_idx, 0))
                .arg_ptr(kv_cache.v_cache_ptr(attn_layer_idx, 0))
                .arg_ptr(chunk_idx)
                .arg_ptr(chunk_bt)
                .arg_ptr(chunk_po)
                .arg_ptr(chunk_pm)
                .arg_ptr(chunk_ps)
                .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                    (k_stride / 2) as u64
                } else {
                    k_stride as u64
                })
                .arg_u64(if kv_dtype == KvCacheDtype::Bf16 {
                    (v_stride / 2) as u64
                } else {
                    v_stride as u64
                });
            if kv_dtype == KvCacheDtype::Nvfp4 {
                launch = launch.arg_u64(kv_cache.nvfp4_data_bytes() as u64);
            }
            launch = launch
                .arg_u32(kv_cache.block_size() as u32)
                .arg_u32(num_query_heads)
                .arg_u32(num_kv_heads)
                .arg_u32(head_dim)
                .arg_f32(softmax_scale)
                .arg_u32(SPARSE_SPLITS as u32)
                .arg_u32(query_row_stride)
                .arg_u32(frame_meta.max_blocks_per_seq);
            if partial_masked {
                launch = launch.arg_u64(mask_lo).arg_u64(mask_hi);
            }
            launch.launch(stream)?;

            if fused_gate {
                let (gate_ptr, gate_stride) = gate.expect("fused_gate requires gate");
                let chunk_gate = gate_ptr.offset(chunk_base * (gate_stride as usize) * 2);
                let mut rlaunch = KernelLaunch::new(gpu, reduce_k)
                    .grid([num_query_heads, chunk_rows as u32, 1])
                    .block([head_dim, 1, 1])
                    .pdl(chunk_rows <= 64)
                    .arg_ptr(chunk_po)
                    .arg_ptr(chunk_pm)
                    .arg_ptr(chunk_ps)
                    .arg_ptr(chunk_gate)
                    .arg_ptr(chunk_out)
                    .arg_u32(num_query_heads)
                    .arg_u32(head_dim)
                    .arg_u32(SPARSE_SPLITS as u32)
                    .arg_u32(gate_stride);
                if reduce_masked {
                    rlaunch = rlaunch.arg_u64(mask_lo).arg_u64(mask_hi);
                }
                rlaunch.launch(stream)?;
            } else {
                let mut rlaunch = KernelLaunch::new(gpu, reduce_k)
                    .grid([num_query_heads, chunk_rows as u32, 1])
                    .block([head_dim, 1, 1])
                    .pdl(chunk_rows <= 64)
                    .arg_ptr(chunk_po)
                    .arg_ptr(chunk_pm)
                    .arg_ptr(chunk_ps)
                    .arg_ptr(chunk_out)
                    .arg_u32(num_query_heads)
                    .arg_u32(head_dim)
                    .arg_u32(SPARSE_SPLITS as u32);
                if reduce_masked {
                    rlaunch = rlaunch.arg_u64(mask_lo).arg_u64(mask_hi);
                }
                rlaunch.launch(stream)?;
            }
        }
        Ok(fused_gate)
    }

    /// Multi-sequence batched QSA verify prefix commit across `num_seqs` lanes
    /// in one `grid=[num_seqs, 1, 1]` launch.
    ///
    /// Matches [`Self::commit_verify_prefix`] semantics: if this layer's QSA
    /// side cache has not been initialized yet, returns `Ok(())` without
    /// allocating the cache or launching a kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_verify_prefix_multi_seq(
        &self,
        lanes_dev: DevicePtr,
        accepted_counts_dev: DevicePtr,
        row_counts_dev: DevicePtr,
        num_seqs: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if num_seqs == 0 {
            return Ok(());
        }
        let Some(cache) = self.cache.get() else {
            return Ok(());
        };
        ensure!(
            self.commit_verify_prefix_multi_seq.0 != 0,
            "qwen4_qsa_commit_verify_prefix_multi_seq kernel is unavailable"
        );
        KernelLaunch::new(gpu, self.commit_verify_prefix_multi_seq)
            .grid([num_seqs as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(cache.verify_undo_raw)
            .arg_ptr(cache.verify_undo_compressed)
            .arg_ptr(cache.verify_undo_raw_offsets)
            .arg_ptr(cache.verify_undo_compressed_offsets)
            .arg_ptr(cache.verify_valid_mask)
            .arg_ptr(lanes_dev)
            .arg_ptr(accepted_counts_dev)
            .arg_ptr(row_counts_dev)
            .arg_u32(num_seqs as u32)
            .launch(stream)
    }
}

/// One sequence's QSA verify prefix commit descriptor for batched
/// multi-sequence commit (`qwen4_qsa_commit_verify_prefix_multi_seq`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QsaMultiSeqCommitEntry {
    pub lane: u32,
    pub accepted: u32,
    pub rows: u32,
}

/// Packed little-endian host payload `[lanes[N] | accepted_counts[N] | row_counts[N]]`
/// (`3 * N * 4` bytes) for [`Qwen4QsaIndexer::commit_verify_prefix_multi_seq`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QsaMultiSeqCommitPack {
    pub bytes: Vec<u8>,
    pub num_seqs: usize,
    pub accepted_offset: usize,
    pub rows_offset: usize,
    pub max_lane: usize,
    pub fallback_entries: Vec<QsaMultiSeqCommitEntry>,
}

/// Validate and pack ragged multi-sequence QSA verify commit entries
/// (`(slot_lane, accepted, rows)`).
///
/// Filters out non-journaled widths (`!verify_journal_rows(rows)`, e.g. `K=1`),
/// validates `accepted <= rows`, rejects duplicate lanes by returning `Ok(None)`
/// so callers fall back cleanly to sequential per-lane commits, and serializes
/// three contiguous `u32` arrays `[lanes | accepted_counts | row_counts]`.
pub(crate) fn pack_qsa_multi_seq_commit_entries(
    entries: &[(usize, usize, usize)],
) -> Result<Option<QsaMultiSeqCommitPack>> {
    let mut filtered: Vec<QsaMultiSeqCommitEntry> = Vec::with_capacity(entries.len());
    let mut max_lane = 0usize;
    for &(lane, accepted, rows) in entries {
        ensure!(
            accepted <= rows,
            "QSA verify commit accepted {accepted} rows from a K={rows} frame"
        );
        if !verify_journal_rows(rows) {
            continue;
        }
        ensure!(
            (2..=VERIFY_MAX_ROWS).contains(&rows),
            "QSA multi-seq verify commit row count {rows} outside 2..={VERIFY_MAX_ROWS}"
        );
        let lane_u32 = u32::try_from(lane).context("QSA multi-seq commit lane exceeds u32")?;
        if filtered.iter().any(|prev| prev.lane == lane_u32) {
            return Ok(None);
        }
        max_lane = max_lane.max(lane);
        filtered.push(QsaMultiSeqCommitEntry {
            lane: lane_u32,
            accepted: accepted as u32,
            rows: rows as u32,
        });
    }
    let num_seqs = filtered.len();
    if num_seqs == 0 {
        return Ok(Some(QsaMultiSeqCommitPack {
            bytes: Vec::new(),
            num_seqs: 0,
            accepted_offset: 0,
            rows_offset: 0,
            max_lane: 0,
            fallback_entries: filtered,
        }));
    }
    let section_bytes = num_seqs
        .checked_mul(std::mem::size_of::<u32>())
        .context("QSA multi-seq commit section byte overflow")?;
    let total_bytes = section_bytes
        .checked_mul(3)
        .context("QSA multi-seq commit total byte overflow")?;
    let mut bytes = vec![0u8; total_bytes];
    let accepted_offset = section_bytes;
    let rows_offset = section_bytes * 2;
    for (i, entry) in filtered.iter().enumerate() {
        let off = i * 4;
        bytes[off..off + 4].copy_from_slice(&entry.lane.to_le_bytes());
        bytes[accepted_offset + off..accepted_offset + off + 4]
            .copy_from_slice(&entry.accepted.to_le_bytes());
        bytes[rows_offset + off..rows_offset + off + 4].copy_from_slice(&entry.rows.to_le_bytes());
    }
    Ok(Some(QsaMultiSeqCommitPack {
        bytes,
        num_seqs,
        accepted_offset,
        rows_offset,
        max_lane,
        fallback_entries: filtered,
    }))
}

struct QsaMultiSeqCommitBatch {
    lanes_dev: DevicePtr,
    accepted_counts_dev: DevicePtr,
    row_counts_dev: DevicePtr,
    num_seqs: usize,
    max_lane: usize,
    fallback_entries: Vec<QsaMultiSeqCommitEntry>,
}

thread_local! {
    static QSA_MULTI_SEQ_COMMIT_BATCH: std::cell::RefCell<Option<QsaMultiSeqCommitBatch>> =
        const { std::cell::RefCell::new(None) };
}

struct QsaMultiSeqCommitSuspendGuard {
    saved: Option<QsaMultiSeqCommitBatch>,
}

impl Drop for QsaMultiSeqCommitSuspendGuard {
    fn drop(&mut self) {
        let saved = self.saved.take();
        QSA_MULTI_SEQ_COMMIT_BATCH.with(|cell| {
            *cell.borrow_mut() = saved;
        });
    }
}

/// Thread-local RAII scope binding pre-uploaded multi-sequence QSA commit
/// arrays so [`Qwen4QsaIndexer::commit_verify_prefix`] collapses `C * 12`
/// per-sequence launches into 12 `grid=[C, 1, 1]` launches without widening
/// `TransformerLayer`.
pub(crate) struct QsaMultiSeqCommitGuard {
    prev: Option<QsaMultiSeqCommitBatch>,
}

impl QsaMultiSeqCommitGuard {
    pub(crate) fn enter(
        lanes_dev: DevicePtr,
        accepted_counts_dev: DevicePtr,
        row_counts_dev: DevicePtr,
        num_seqs: usize,
        max_lane: usize,
        fallback_entries: Vec<QsaMultiSeqCommitEntry>,
    ) -> Self {
        let batch = QsaMultiSeqCommitBatch {
            lanes_dev,
            accepted_counts_dev,
            row_counts_dev,
            num_seqs,
            max_lane,
            fallback_entries,
        };
        let prev = QSA_MULTI_SEQ_COMMIT_BATCH.with(|cell| cell.borrow_mut().replace(batch));
        Self { prev }
    }
}

impl Drop for QsaMultiSeqCommitGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        QSA_MULTI_SEQ_COMMIT_BATCH.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

#[cfg(test)]
mod long_context_tests {
    use super::{
        QSA_DESC_FLAG_JOURNAL, QSA_DESC_FLAG_QSA_REQUIRED, QSA_DESC_FLAG_SPARSE, QsaBatchSeqDesc,
        QsaCacheLayout, Qwen4CrossSeqQsaScratch, VERIFY_MAX_ROWS, VERIFY_ROWS_DENSE_ENDPOINT,
        VerifyAttentionRoute, VerifyRouteTally, VerifyRowAttentionRoute, VerifyRowRouteCounts,
        checked_side_cache_bytes, verify_rows_context_safe,
    };

    #[test]
    fn qsa_batch_seq_desc_abi_and_scratch_bytes() {
        assert_eq!(std::mem::size_of::<QsaBatchSeqDesc>(), 32);
        assert_eq!(std::mem::align_of::<QsaBatchSeqDesc>(), 16);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, row_base), 0);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, seq_len), 4);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, k), 8);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, journal_lane), 12);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, flags), 16);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, logits_offset), 20);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, logits_stride), 24);
        assert_eq!(std::mem::offset_of!(QsaBatchSeqDesc, score_block_end), 28);

        let desc = QsaBatchSeqDesc {
            row_base: 10,
            seq_len: 4096,
            k: 5,
            journal_lane: 3,
            flags: QSA_DESC_FLAG_QSA_REQUIRED | QSA_DESC_FLAG_SPARSE | QSA_DESC_FLAG_JOURNAL,
            logits_offset: 10_240,
            logits_stride: 1_025,
            score_block_end: 66,
        };
        let mut bytes = [0u8; QsaBatchSeqDesc::SIZE_BYTES];
        desc.write_le_bytes(&mut bytes);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 10);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4096);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 7);
        assert_eq!(
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            10_240
        );
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 1_025);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 66);

        let total = Qwen4CrossSeqQsaScratch::total_bytes().unwrap();
        // 256*2051*4 + 256*24*8*256*4 + 2*(256*24*8*4) + 32 MiB = 86,382,592 B (~82.38 MiB)
        assert_eq!(Qwen4CrossSeqQsaScratch::MAX_ROWS, 256);
        assert_eq!(
            total,
            256 * 2051 * 4 + 256 * 24 * 8 * 256 * 4 + 2 * (256 * 24 * 8 * 4) + 32 * 1024 * 1024
        );

        // Verify 256-row masked chunking splits cleanly across the 128-bit CUDA ABI
        // and skips all-zero halves while keeping unmasked launches single-pass.
        assert_eq!(
            super::plan_qsa_sparse_row_chunks(256, None, false),
            ([(0, 256, u64::MAX, u64::MAX), (0, 0, 0, 0)], 1)
        );
        assert_eq!(
            super::plan_qsa_sparse_row_chunks(256, None, true),
            (
                [(0, 128, u64::MAX, u64::MAX), (128, 128, u64::MAX, u64::MAX)],
                2
            )
        );
        assert_eq!(
            super::plan_qsa_sparse_row_chunks(192, Some([0, 0, 0xF, 0]), true),
            ([(128, 64, 0xF, 0), (0, 0, 0, 0)], 1)
        );
        assert_eq!(
            super::plan_qsa_sparse_row_chunks(192, Some([1, 0, 0, 0]), true),
            ([(0, 128, 1, 0), (0, 0, 0, 0)], 1)
        );
        assert_eq!(
            super::plan_qsa_sparse_row_chunks(200, Some([1, 2, 3, 4]), true),
            ([(0, 128, 1, 2), (128, 72, 3, 4)], 2)
        );
    }

    #[test]
    fn multi_row_verify_falls_back_only_when_straddling_2048_or_overflowing() {
        assert!(verify_rows_context_safe(2_043, 5));
        assert!(!verify_rows_context_safe(2_044, 5));
        assert!(!verify_rows_context_safe(2_047, 5));
        assert!(verify_rows_context_safe(2_048, 5));
        assert!(verify_rows_context_safe(131_067, 5));
        assert!(verify_rows_context_safe(262_139, 5));
        assert!(!verify_rows_context_safe(u32::MAX as usize, 5));
        assert!(!verify_rows_context_safe(usize::MAX, 5));
    }

    #[test]
    fn route_tally_counts_only_the_actual_selected_route() {
        let tally = VerifyRouteTally::new();
        let mut rows = VerifyRowRouteCounts::default();
        rows.record(VerifyRowAttentionRoute::DensePaged);
        rows.record(VerifyRowAttentionRoute::SparseQsa);
        rows.record(VerifyRowAttentionRoute::HighSpeedSwap);
        rows.record(VerifyRowAttentionRoute::Mla);
        tally.record(VerifyAttentionRoute::RowSerial(rows));
        tally.record(VerifyAttentionRoute::SparseRowsLegacy);
        let totals = tally.record(VerifyAttentionRoute::SparseRowsV2);
        assert_eq!(totals.row_serial_frames, 1);
        assert_eq!(totals.dense_paged_rows, 1);
        assert_eq!(totals.sparse_qsa_rows, 1);
        assert_eq!(totals.high_speed_swap_rows, 1);
        assert_eq!(totals.mla_rows, 1);
        assert_eq!(totals.sparse_rows_legacy_frames, 1);
        assert_eq!(totals.sparse_rows_v2_frames, 1);
    }

    #[test]
    fn row_route_counts_preserve_mixed_boundary_and_hss_dispatches() {
        let mut rows = VerifyRowRouteCounts::default();
        rows.record(VerifyRowAttentionRoute::DensePaged);
        rows.record(VerifyRowAttentionRoute::DensePaged);
        rows.record(VerifyRowAttentionRoute::SparseQsa);
        rows.record(VerifyRowAttentionRoute::HighSpeedSwap);
        rows.record(VerifyRowAttentionRoute::Mla);
        assert_eq!(rows.dense_paged, 2);
        assert_eq!(rows.sparse_qsa, 1);
        assert_eq!(rows.high_speed_swap, 1);
        assert_eq!(rows.mla, 1);
        assert_eq!(rows.total(), 5);
    }

    #[test]
    fn boundary_straddling_frames_reject_while_pure_dense_and_pure_sparse_admit() {
        for rows in 2..=VERIFY_MAX_ROWS {
            let last_dense_start = VERIFY_ROWS_DENSE_ENDPOINT - rows;
            assert!(verify_rows_context_safe(last_dense_start, rows));
            for seq_len in (last_dense_start + 1)..VERIFY_ROWS_DENSE_ENDPOINT {
                assert!(
                    !verify_rows_context_safe(seq_len, rows),
                    "straddling frame admitted at seq_len={seq_len}, rows={rows}"
                );
            }
            assert!(verify_rows_context_safe(VERIFY_ROWS_DENSE_ENDPOINT, rows));
            assert!(verify_rows_context_safe(
                VERIFY_ROWS_DENSE_ENDPOINT + 1,
                rows
            ));
        }
    }

    #[test]
    fn side_cache_accounting_includes_every_verify_row() {
        let blocks = 16_385;
        let layout = QsaCacheLayout::checked(blocks, 16).unwrap();
        assert_eq!(layout.max_compressed_groups, blocks * 4);
        assert_eq!(
            layout.logits,
            VERIFY_MAX_ROWS * layout.max_compressed_groups * std::mem::size_of::<f32>()
        );
        assert_eq!(
            checked_side_cache_bytes(blocks, 16).unwrap(),
            layout.total_bytes().unwrap()
        );
    }

    #[test]
    fn side_cache_accounting_rejects_invalid_or_overflowing_geometry() {
        assert!(checked_side_cache_bytes(1, 2).is_err());
        assert!(checked_side_cache_bytes(usize::MAX, 16).is_err());
    }

    #[test]
    fn reused_draft_indices_layout_and_width_bounds() {
        let layout = QsaCacheLayout::checked(16, 16).unwrap();
        assert_eq!(
            layout.reused_draft_indices,
            (super::QSA_OUTPUT_WIDTH + super::QSA_REUSE_MAX_DRAFTS) * std::mem::size_of::<i32>()
        );
        assert_eq!(layout.reused_draft_indices, 8_268);
        assert_eq!(super::checked_reused_index_width(0, 8).unwrap(), 2_051);
        assert_eq!(super::checked_reused_index_width(8, 8).unwrap(), 2_059);
        assert_eq!(super::checked_reused_index_width(16, 16).unwrap(), 2_067);
        assert!(super::checked_reused_index_width(9, 8).is_err());
        assert!(super::checked_reused_index_width(1, 17).is_err());
    }

    #[test]
    fn draft_selection_reuse_guard_restores_previous_state_on_drop() {
        assert!(
            super::DRAFT_SELECTION_REUSE_STATE
                .with(std::cell::Cell::get)
                .is_none()
        );
        {
            let _outer = super::DraftSelectionReuseGuard::snapshot_step0();
            assert!(matches!(
                super::DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
                Some(super::DraftReuseState::SnapshotStep0 {
                    snapshotted_in_update: false,
                    from_verify_rows: false,
                })
            ));
            {
                let _inner = super::DraftSelectionReuseGuard::reuse_step(4096, 3, 8);
                assert_eq!(
                    super::DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
                    Some(super::DraftReuseState::ReuseStep {
                        base_seq_len: 4096,
                        step_offset: 3,
                        max_drafts: 8,
                    })
                );
            }
            assert!(matches!(
                super::DRAFT_SELECTION_REUSE_STATE.with(std::cell::Cell::get),
                Some(super::DraftReuseState::SnapshotStep0 { .. })
            ));
        }
        assert!(
            super::DRAFT_SELECTION_REUSE_STATE
                .with(std::cell::Cell::get)
                .is_none()
        );
    }

    #[test]
    fn yarn_inv_freq_respects_positive_beta_fast_below_32_and_defaults_non_positive() {
        let mut cfg_default = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        cfg_default.yarn_factor = 4.0;
        cfg_default.yarn_beta_fast = 0.0;
        cfg_default.yarn_beta_slow = 1.0;
        cfg_default.yarn_original_max_position_embeddings = 32_768;

        let mut cfg_32 = cfg_default.clone();
        cfg_32.yarn_beta_fast = 32.0;

        let mut cfg_16 = cfg_default.clone();
        cfg_16.yarn_beta_fast = 16.0;

        let default_inv =
            super::qwen4_yarn_inv_freq_values_with_theta(&cfg_default, 10_000.0, 64).unwrap();
        let inv_32 = super::qwen4_yarn_inv_freq_values_with_theta(&cfg_32, 10_000.0, 64).unwrap();
        let inv_16 = super::qwen4_yarn_inv_freq_values_with_theta(&cfg_16, 10_000.0, 64).unwrap();

        assert_eq!(default_inv, inv_32);
        assert_ne!(
            inv_16, inv_32,
            "positive yarn_beta_fast < 32.0 must not be clamped to 32.0"
        );
    }
}

#[cfg(test)]
mod verify_journal_lane_tests {
    use super::{journal_lane_offsets, verify_journal_lane_bytes};

    #[test]
    fn one_lane_is_the_historical_single_journal() {
        // 2 x (9 rows x 128 x BF16) undo + 2 x (9 x u64) offsets + u32 mask.
        assert_eq!(verify_journal_lane_bytes().unwrap(), 4_756);
        assert_eq!(journal_lane_offsets(0, 1, 2_304, 72, 4).unwrap(), (0, 0, 0));
        assert!(journal_lane_offsets(1, 1, 2_304, 72, 4).is_err());
    }

    #[test]
    fn lanes_are_disjoint_and_lane_major() {
        for lane in 0..8 {
            assert_eq!(
                journal_lane_offsets(lane, 8, 2_304, 72, 4).unwrap(),
                (lane * 2_304, lane * 72, lane * 4)
            );
        }
        assert!(journal_lane_offsets(8, 8, 2_304, 72, 4).is_err());
    }
}

#[cfg(test)]
mod batch_verify_lane_range_tests {
    use super::{checked_batch_verify_lane_range, distinct_batch_verify_lane_ranges};
    use spark_runtime::gpu::DevicePtr;

    #[test]
    fn actual_allocation_spans_are_contained_and_disjoint_for_all_eight_lanes() {
        let base = DevicePtr(0x1000);
        let ranges = (0..8)
            .map(|lane| checked_batch_verify_lane_range(base, 8 * 2304, 2304, 8, lane).unwrap())
            .collect::<Vec<_>>();
        for (lane, range) in ranges.iter().enumerate() {
            assert_eq!(
                *range,
                (base, 18432, DevicePtr(0x1000 + lane as u64 * 2304), 2304)
            );
        }
        assert_eq!(
            distinct_batch_verify_lane_ranges(ranges.clone()).unwrap(),
            ranges
        );
    }

    #[test]
    fn missing_or_mismatched_actual_allocation_and_unavailable_lane_are_rejected() {
        for (base, allocated, bytes, lanes, lane) in [
            (DevicePtr::NULL, 128, 16, 8, 0),
            (DevicePtr(0x1000), 0, 16, 8, 0),
            (DevicePtr(0x1000), 128, 0, 8, 0),
            (DevicePtr(0x1000), 128, 16, 0, 0),
            (DevicePtr(0x1000), 128, 16, 8, 8),
            (DevicePtr(0x1000), 127, 16, 8, 0),
            (DevicePtr(0x1000), 129, 16, 8, 0),
        ] {
            assert!(checked_batch_verify_lane_range(base, allocated, bytes, lanes, lane).is_err());
        }
    }

    #[test]
    fn overflowing_allocation_or_pointer_endpoints_are_rejected() {
        assert!(
            checked_batch_verify_lane_range(DevicePtr(1), usize::MAX, usize::MAX, 2, 0).is_err()
        );
        assert!(checked_batch_verify_lane_range(DevicePtr(u64::MAX - 15), 128, 16, 8, 0).is_err());
    }

    #[test]
    fn physical_side_cache_block_spans_include_the_final_block_and_reject_past_end() {
        let raw = DevicePtr(0x10000);
        let compressed = DevicePtr(0x20000);
        let raw_bytes = 4 * 128 * 2;
        let compressed_bytes = (256 / 4) * 128 * 2;
        let ranges = vec![
            checked_batch_verify_lane_range(raw, raw_bytes * 3, raw_bytes, 3, 2).unwrap(),
            checked_batch_verify_lane_range(
                compressed,
                compressed_bytes * 3,
                compressed_bytes,
                3,
                2,
            )
            .unwrap(),
        ];
        assert_eq!(ranges[0].2, raw.offset(2 * raw_bytes));
        assert_eq!(ranges[1].2, compressed.offset(2 * compressed_bytes));
        assert!(distinct_batch_verify_lane_ranges(ranges).is_ok());
        assert!(checked_batch_verify_lane_range(raw, raw_bytes * 3, raw_bytes, 3, 3).is_err());
    }

    #[test]
    fn aliased_journal_spans_are_rejected_but_adjacent_spans_are_accepted() {
        let first = checked_batch_verify_lane_range(DevicePtr(0x1000), 128, 16, 8, 0).unwrap();
        let adjacent = checked_batch_verify_lane_range(DevicePtr(0x1000), 128, 16, 8, 1).unwrap();
        assert!(distinct_batch_verify_lane_ranges(vec![first, first]).is_err());
        assert!(distinct_batch_verify_lane_ranges(vec![first, adjacent]).is_ok());
    }

    #[test]
    fn qsa_multi_seq_commit_pack_handles_ragged_c2_to_c16_and_validation() {
        use super::{QsaMultiSeqCommitEntry, pack_qsa_multi_seq_commit_entries};

        // Ragged C=16 cohort mixing K=2..=9 (plus one K=1 anchor-only slot
        // that has no journal and must be filtered out) and accepted=0..=K.
        let mut entries: Vec<(usize, usize, usize)> = Vec::new();
        let mut expected: Vec<QsaMultiSeqCommitEntry> = Vec::new();
        for i in 0..16usize {
            let lane = i * 2 + 1; // non-contiguous lanes 1, 3, ..., 31
            let rows = match i {
                0 => 1, // K=1 anchor-only: filtered out
                1 => 2,
                2 => 3,
                3 => 4,
                4..=9 => 5,
                10 => 6,
                11 => 7,
                12 => 8,
                _ => 9,
            };
            let accepted = (i + 1).min(rows);
            entries.push((lane, accepted, rows));
            if rows >= 2 {
                expected.push(QsaMultiSeqCommitEntry {
                    lane: lane as u32,
                    accepted: accepted as u32,
                    rows: rows as u32,
                });
            }
        }
        let pack = pack_qsa_multi_seq_commit_entries(&entries)
            .expect("valid ragged cohort")
            .expect("distinct lanes must pack");
        assert_eq!(pack.num_seqs, 15);
        assert_eq!(pack.accepted_offset, 15 * 4);
        assert_eq!(pack.rows_offset, 30 * 4);
        assert_eq!(pack.bytes.len(), 45 * 4);
        assert_eq!(pack.max_lane, 31);
        assert_eq!(pack.fallback_entries, expected);
        for (i, want) in expected.iter().enumerate() {
            let lane = u32::from_le_bytes(pack.bytes[i * 4..i * 4 + 4].try_into().unwrap());
            let acc = u32::from_le_bytes(
                pack.bytes[pack.accepted_offset + i * 4..pack.accepted_offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let rows = u32::from_le_bytes(
                pack.bytes[pack.rows_offset + i * 4..pack.rows_offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!((lane, acc, rows), (want.lane, want.accepted, want.rows));
        }

        // C=2 minimum multi-seq cohort with accepted=0 (full reject) and full accept.
        let c2 = pack_qsa_multi_seq_commit_entries(&[(4, 0, 5), (9, 8, 8)])
            .unwrap()
            .unwrap();
        assert_eq!(c2.num_seqs, 2);
        assert_eq!(c2.bytes.len(), 24);
        assert_eq!(c2.max_lane, 9);

        // Pure K=1 cohort packs to num_seqs == 0 (no-op).
        let k1_only = pack_qsa_multi_seq_commit_entries(&[(0, 1, 1), (1, 1, 1)])
            .unwrap()
            .unwrap();
        assert_eq!(k1_only.num_seqs, 0);
        assert!(k1_only.bytes.is_empty());

        // Duplicate lane returns Ok(None) so caller falls back cleanly to per-seq commits.
        assert_eq!(
            pack_qsa_multi_seq_commit_entries(&[(3, 2, 5), (3, 4, 5)]).unwrap(),
            None
        );

        // accepted > rows fails closed with Err.
        assert!(pack_qsa_multi_seq_commit_entries(&[(0, 6, 5)]).is_err());
    }

    #[test]
    fn qsa_prefill_kbc64_t4_policy_auto_enables_above_64_sms_and_fits_2cta_smem() {
        use super::{
            QSA_PREFILL_KBC64_SELECTOR, QSA_PREFILL_KBC64_SM_THRESHOLD, QSA_PREFILL_KBC64_T4_SMEM,
            QSA_PREFILL_KBC64_T4_THREADS, QSA_PREFILL_T4_CAP, QSA_PREFILL_T4_PLAN_SMEM,
            QSA_PREFILL_T4_TILE_ROWS, QsaPrefillKbc64Policy, parse_qsa_prefill_kbc64_policy,
            resolve_qsa_prefill_kbc64_enabled,
        };

        assert_eq!(QSA_PREFILL_KBC64_SELECTOR, "ATLAS_QWEN4_QSA_PREFILL_KBC64");
        assert_eq!(QSA_PREFILL_KBC64_SM_THRESHOLD, 64);
        assert_eq!(QSA_PREFILL_T4_TILE_ROWS, 4);
        assert_eq!(QSA_PREFILL_T4_CAP, 2052);
        assert_eq!(QSA_PREFILL_T4_PLAN_SMEM, 8_272);
        assert_eq!(QSA_PREFILL_KBC64_T4_THREADS, 192);
        assert_eq!(QSA_PREFILL_KBC64_T4_SMEM, 36_352);
        const {
            assert!(
                QSA_PREFILL_KBC64_T4_SMEM <= 49_152,
                "kBc=64 T=4 dynamic SMEM must fit 2 CTAs/SM (<= 49,152 B)"
            );
        }

        assert_eq!(
            parse_qsa_prefill_kbc64_policy(None),
            QsaPrefillKbc64Policy::Auto
        );
        assert_eq!(
            parse_qsa_prefill_kbc64_policy(Some("auto")),
            QsaPrefillKbc64Policy::Auto
        );
        assert_eq!(
            parse_qsa_prefill_kbc64_policy(Some("1")),
            QsaPrefillKbc64Policy::Enabled
        );
        assert_eq!(
            parse_qsa_prefill_kbc64_policy(Some("TRUE")),
            QsaPrefillKbc64Policy::Enabled
        );
        assert_eq!(
            parse_qsa_prefill_kbc64_policy(Some("0")),
            QsaPrefillKbc64Policy::Disabled
        );
        assert_eq!(
            parse_qsa_prefill_kbc64_policy(Some("off")),
            QsaPrefillKbc64Policy::Disabled
        );

        // Auto: GB10 (<= 64 SMs) stays OFF; SM120+ (> 64 SMs) turns ON.
        assert!(!resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Auto,
            48
        ));
        assert!(!resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Auto,
            64
        ));
        assert!(resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Auto,
            65
        ));
        assert!(resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Auto,
            188
        ));

        // Explicit overrides win on any SM count.
        assert!(!resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Disabled,
            188
        ));
        assert!(resolve_qsa_prefill_kbc64_enabled(
            QsaPrefillKbc64Policy::Enabled,
            48
        ));
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "qwen4_qsa/gpu_parity_tests.rs"]
mod gpu_parity_tests;
