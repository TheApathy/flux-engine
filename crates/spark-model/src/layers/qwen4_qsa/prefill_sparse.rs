// SPDX-License-Identifier: AGPL-3.0-only

//! QSA-sparse prefill attention core (`ATLAS_QWEN4_PREFILL_QSA=1`).
//!
//! Past position 2050 a Flash-Next query attends only to the top-512 of its
//! complete 4-token blocks plus its open 0-3 token tail (HF
//! `Qwen4ExpTextQSAIndexer`; Atlas decode `update_and_select`). The dense
//! prefill cores compute a different function there, at O(N^2). This stage
//! computes the model's function for a whole prefill chunk:
//!   1. indexer queries for every row: one cuBLASLt projection, then the
//!      same q_norm and RoPE ops and weights decode uses;
//!   2. block scores against the paged compressed-key side cache (the chunk
//!      prep has already stored this chunk's groups), top-512 per row into a
//!      block bitmask (rows inside the dense window take every block);
//!   3. a union plan per tile of consecutive rows and the sparse attention
//!      kernel over the BF16 paged K/V.
//!
//! It writes the same `[rows][nq*hd]` output as the dense core; the sigmoid
//! gate and o_proj that follow are unchanged. Not bit-exact with anything:
//! the reduction order is its own.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::prefill_fused_topk as fused_topk;
use super::{COMPRESS_RATIO, INDEX_DIM, INDEX_HEADS, QUERY_HEADS};
pub use fused_topk::FUSED_RADIX_ALIAS;

pub const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA";
const BLOCK_TOPK: usize = 512;
/// Last absolute position whose query sees at most 512 complete blocks; every
/// row up to here is dense under QSA as well.
pub const DENSE_LAST_POS: usize = COMPRESS_RATIO * (BLOCK_TOPK + 1) - 2;
pub(super) const KV_HEADS: usize = 2;
pub(super) const HEAD_DIM: usize = 256;
pub(super) const TILE_ROWS: usize = 10;
/// Union bound per tile: 512 selected + 1 tail block per row (kernel kCap).
pub(super) const ATTN_CAP: usize = (TILE_ROWS * (BLOCK_TOPK + 1)).div_ceil(4) * 4;
pub(super) const ATTN_THREADS: usize = (QUERY_HEADS / KV_HEADS * TILE_ROWS).div_ceil(16) * 32;
pub(super) const ATTN_THREADS_G6: usize = (6 * TILE_ROWS).div_ceil(16) * 32;
pub(super) const ATTN_SMEM: usize = 4 * 32 * HEAD_DIM * 2 + ATTN_CAP * 4;
pub(super) const SCORE_BATCH_ROWS: usize = 2048;
/// fp32 score scratch per batch: long contexts score fewer rows per batch
/// (2048 rows up to 32K tokens, 512 at 128K).
const SCORE_SCRATCH_BYTES: usize = 64 << 20;
/// `ATLAS_QWEN4_PREFILL_QSA_V2=1`: an L2-sized score batch, so the fp32 scores
/// the top-k reads back stay in the 24 MB L2 instead of round-tripping DRAM,
/// while doubling score_batch at 32K..128K within the same 192 MiB scratch step.
const V2_SCORE_SCRATCH_BYTES: usize = 32 << 20;
pub const V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_V2";
pub const SCORE_V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_SCORE_V2";
pub const PREP_FUSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_PREP_FUSED";
pub(crate) use super::prefill_sparse_dump::dump_enabled;
/// GB10 (sm_121) MaxSharedMemoryPerBlockOptin: a kernel's static plus its
/// requested dynamic shared memory must fit, or cuFuncSetAttribute fails.
const SMEM_OPTIN_MAX: usize = 101_376;
/// Static shared memory of `qwen4_qsa_prefill_topk` as ptxas reports it:
/// hist[2048] + s_warp[8] + 3 scalars = 8,236 B, aligned to 8,240.
const TOPK_STATIC_SMEM: usize = 8_240;
/// The staged top-k kernel holds one row of fp32 scores in dynamic shared
/// memory (rows up to 22528 blocks = 90112 tokens); longer rows use
/// `qwen4_qsa_prefill_topk_global`, which reads the row from global memory.
pub(super) const MAX_TOPK_SMEM: usize = 88 * 1024;
// Shared-memory budgets checked at compile time: an over-budget request makes
// cuFuncSetAttribute fail at the first QSA chunk in the server (spark-w6).
const _: () = assert!(TOPK_STATIC_SMEM + MAX_TOPK_SMEM <= SMEM_OPTIN_MAX);
const _: () = assert!(ATTN_SMEM <= SMEM_OPTIN_MAX);
const _: () = assert!(PLAN_SORT_SMEM <= 48 * 1024);

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{SELECTOR} must be absent, 0, or 1; got {other:?}"),
    }
}

/// Multi-tile `ldmatrix.x4` QSA score + top-k v2 mode (`ATLAS_QWEN4_PREFILL_QSA_SCORE_V2`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreV2Mode {
    Auto,
    Tiles2,
    Tiles4,
    Tiles8,
    Tiles16,
}

impl ScoreV2Mode {
    pub const fn blocks_per_cta(self, n_max: usize) -> usize {
        match self {
            Self::Auto => {
                if n_max >= 16384 {
                    1024
                } else if n_max >= 8192 {
                    512
                } else if n_max >= 4096 {
                    256
                } else {
                    128
                }
            }
            Self::Tiles2 => 128,
            Self::Tiles4 => 256,
            Self::Tiles8 => 512,
            Self::Tiles16 => 1024,
        }
    }
}

fn parse_score_v2(value: Option<&str>) -> Result<Option<ScoreV2Mode>> {
    match value {
        None | Some("0") => Ok(None),
        Some("1") => Ok(Some(ScoreV2Mode::Auto)),
        Some("2") => Ok(Some(ScoreV2Mode::Tiles2)),
        Some("4") => Ok(Some(ScoreV2Mode::Tiles4)),
        Some("8") => Ok(Some(ScoreV2Mode::Tiles8)),
        Some("16") => Ok(Some(ScoreV2Mode::Tiles16)),
        Some(other) => {
            anyhow::bail!("{SCORE_V2_SELECTOR} must be absent, 0, 1, 2, 4, 8, or 16; got {other:?}")
        }
    }
}

/// Whether the sparse core is selected (read once per process).
pub fn selected() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow::anyhow!(e))
}

/// `ATLAS_QWEN4_PREFILL_QSA_V2=1` (with the QSA core): L2-sized score batches,
/// and tile plans sorted by need mask for the warp-skipping attention kernel.
/// Same function; the attention sums its blocks in a different order.
pub fn v2_selected() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse(std::env::var(V2_SELECTOR).ok().as_deref())
            .map_err(|e| e.to_string().replace(SELECTOR, V2_SELECTOR))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// `ATLAS_QWEN4_PREFILL_QSA_SCORE_V2` (`1` = auto 2/4/8/16 tiles per CTA, `2`/`4`/`8`/`16` =
/// fixed tiles per CTA, `0` = off). Defaults to `Auto` when unset and `plan_fused()` is enabled.
pub fn score_v2_mode() -> Result<Option<ScoreV2Mode>> {
    static S: OnceLock<Result<Option<ScoreV2Mode>, String>> = OnceLock::new();
    S.get_or_init(|| match std::env::var(SCORE_V2_SELECTOR).ok().as_deref() {
        None if super::prefill_fp8::plan_fused().unwrap_or(false) => Ok(Some(ScoreV2Mode::Auto)),
        raw => parse_score_v2(raw).map_err(|e| e.to_string()),
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// `ATLAS_QWEN4_PREFILL_QSA_PREP_FUSED=1`: fused QSA key pool+norm+RoPE+store,
/// fused indexer query norm+RoPE, and vectorized V / omitted dead BF16 Q/K writeback in FP8 prep.
pub fn prep_fused_selected() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse(std::env::var(PREP_FUSED_SELECTOR).ok().as_deref())
            .map_err(|e| e.to_string().replace(SELECTOR, PREP_FUSED_SELECTOR))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// True when a chunk holds a row past the dense window, i.e. when QSA and
/// dense causal attention differ for it.
pub fn chunk_is_sparse(start: usize, rows: usize) -> bool {
    start.saturating_add(rows) > DENSE_LAST_POS + 1
}

/// Geometry and scratch layout of one chunk (all offsets in bytes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SparsePlan {
    pub rows: usize,
    pub start: usize,
    /// Complete blocks visible to the chunk's last row.
    pub blocks: usize,
    pub ld: usize,
    pub mask_words: usize,
    pub sparse_row0: usize,
    pub tiles: usize,
    pub iq_off: usize,
    pub mask_off: usize,
    pub entry_off: usize,
    pub count_off: usize,
    pub score_off: usize,
    /// Sparse rows scored (and selected) per batch.
    pub score_batch: usize,
    /// `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK`: the score batch's group maxima,
    /// `[score_batch][ldg]` u32 after the scores (`ldg = ld / 8`); 0 / 0 when
    /// the fused select is off (the layout is then exactly the unfused one).
    pub gmax_off: usize,
    pub ldg: usize,
    pub bytes: usize,
}

impl SparsePlan {
    // Test-only since the fused-select change routes every caller through `for_chunk` (deny(dead_code) otherwise).
    #[cfg(test)]
    pub fn new(rows: usize, start: usize) -> Result<Self> {
        Self::with_score_budget(rows, start, SCORE_SCRATCH_BYTES, false)
    }

    /// The V2 plan: score batches sized for L2.
    #[cfg(test)]
    pub fn v2(rows: usize, start: usize) -> Result<Self> {
        Self::with_score_budget(rows, start, V2_SCORE_SCRATCH_BYTES, false)
    }

    /// The plan a chunk runs with: V1 or V2 score batches, plus the group-maxima
    /// region when the fused select is on (same batches, same other offsets).
    pub fn for_chunk(rows: usize, start: usize, v2: bool, fused: bool) -> Result<Self> {
        let budget = if v2 {
            V2_SCORE_SCRATCH_BYTES
        } else {
            SCORE_SCRATCH_BYTES
        };
        Self::with_score_budget(rows, start, budget, fused)
    }

    fn with_score_budget(
        rows: usize,
        start: usize,
        score_bytes: usize,
        fused: bool,
    ) -> Result<Self> {
        ensure!(rows > 0, "{SELECTOR}: empty chunk");
        let end = start.checked_add(rows).context("chunk end overflows")?;
        ensure!(end <= u32::MAX as usize / 2, "{SELECTOR}: context too long");
        let blocks = end / COMPRESS_RATIO;
        let ld = blocks.next_multiple_of(64);
        let mask_words = (blocks + 1).div_ceil(32);
        let sparse_row0 = (DENSE_LAST_POS + 1).saturating_sub(start).min(rows);
        let tiles = rows.div_ceil(TILE_ROWS);
        let align = |x: usize| x.next_multiple_of(256);
        let iq_off = 0;
        let mask_off = align(iq_off + rows * INDEX_HEADS as usize * INDEX_DIM as usize * 2);
        let entry_off = align(mask_off + rows * mask_words * 4);
        let count_off = align(entry_off + tiles * ATTN_CAP * 4);
        let score_off = align(count_off + tiles * 4);
        let score_batch = (score_bytes / (ld * 4)).clamp(64, SCORE_BATCH_ROWS) / 64 * 64;
        let score_rows = (rows - sparse_row0).min(score_batch);
        let scores_end = align(score_off + score_rows * ld * 4);
        let (gmax_off, ldg, bytes) = if fused {
            let ldg = fused_topk::gmax_ld(ld);
            (scores_end, ldg, align(scores_end + score_rows * ldg * 4))
        } else {
            (0, 0, scores_end)
        };
        Ok(Self {
            rows,
            start,
            blocks,
            ld,
            mask_words,
            sparse_row0,
            tiles,
            iq_off,
            mask_off,
            entry_off,
            count_off,
            score_off,
            score_batch,
            gmax_off,
            ldg,
            bytes,
        })
    }
}

pub(super) struct Kernels {
    pub score: KernelHandle,
    pub topk: KernelHandle,
    pub topk_global: KernelHandle,
    pub plan: KernelHandle,
    pub attn: KernelHandle,
    pub attn_nvfp4: KernelHandle,
    pub plan_sort: KernelHandle,
    pub attn_skip: KernelHandle,
    pub attn_skip_nvfp4: KernelHandle,
    pub attn_g6: KernelHandle,
    pub attn_nvfp4_g6: KernelHandle,
    pub attn_skip_g6: KernelHandle,
    pub attn_skip_nvfp4_g6: KernelHandle,
}

/// Shared-memory tile plan capacity (10 rows share the 4 sink blocks, so union <= 5094 < 5120;
/// 1024 bins + 5120 entries = 24,576 B + 1,024 B CTA overhead = 25,600 B = 102,400 / 4 -> 4 CTAs/SM).
pub(super) const PLAN_SORT_CAP: usize = 5120;
/// Dynamic shared memory of `qwen4_qsa_prefill_plan_sort`: 1024 bins + one tile plan.
pub(super) const PLAN_SORT_SMEM: usize = (1024 + PLAN_SORT_CAP) * 4;

pub(super) fn kernels(gpu: &dyn GpuBackend) -> Result<&'static Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |m: &str, f: &str| gpu.kernel(m, f).map_err(|e| format!("{m}::{f}: {e}"));
        let sel = "qwen4_qsa_prefill_select";
        let attn_nvfp4 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_nvfp4_t10",
        );
        let attn_skip_nvfp4 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_nvfp4_t10_skip",
        );
        let attn_g6 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_t10_g6",
        );
        let attn_nvfp4_g6 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_nvfp4_t10_g6",
        );
        let attn_skip_g6 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_t10_skip_g6",
        );
        let attn_skip_nvfp4_g6 = crate::layers::try_kernel(
            gpu,
            "qwen4_qsa_prefill_attn",
            "qwen4_qsa_prefill_attn_nvfp4_t10_skip_g6",
        );
        let k = Kernels {
            score: load(sel, "qwen4_qsa_prefill_score")?,
            topk: load(sel, "qwen4_qsa_prefill_topk_v2")?,
            topk_global: load(sel, "qwen4_qsa_prefill_topk_global_v2")?,
            plan: load(sel, "qwen4_qsa_prefill_plan")?,
            attn: load("qwen4_qsa_prefill_attn", "qwen4_qsa_prefill_attn_t10")?,
            attn_nvfp4,
            plan_sort: load(sel, "qwen4_qsa_prefill_plan_sort")?,
            attn_skip: load("qwen4_qsa_prefill_attn", "qwen4_qsa_prefill_attn_t10_skip")?,
            attn_skip_nvfp4,
            attn_g6,
            attn_nvfp4_g6,
            attn_skip_g6,
            attn_skip_nvfp4_g6,
        };
        for handle in [
            k.attn_nvfp4,
            k.attn_skip_nvfp4,
            k.attn_g6,
            k.attn_nvfp4_g6,
            k.attn_skip_g6,
            k.attn_skip_nvfp4_g6,
        ] {
            if handle.0 != 0 {
                gpu.set_kernel_max_dynamic_shared_memory(handle, ATTN_SMEM as u32)
                    .map_err(|e| e.to_string())?;
            }
        }
        gpu.set_kernel_max_dynamic_shared_memory(k.attn, ATTN_SMEM as u32)
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.attn_skip, ATTN_SMEM as u32))
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.topk, MAX_TOPK_SMEM as u32))
            .and_then(|()| {
                gpu.set_kernel_max_dynamic_shared_memory(k.topk_global, MAX_TOPK_SMEM as u32)
            })
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernels unavailable: {e}"))
}

/// Whether a top-k launch whose longest row has `blocks` complete blocks fits
/// the staged (shared-memory) kernel; longer rows take the global-memory one.
pub(super) fn topk_staged(blocks: usize) -> bool {
    blocks * 4 <= MAX_TOPK_SMEM
}

/// GB10 SM count and the L2 share the global top-k's rows in flight may use:
/// each of the four passes re-reads its row, so the rows must stay L2-resident.
const GB10_SMS: usize = 48;
const TOPK_GLOBAL_L2_BYTES: usize = 12 << 20;

/// Top-k kernel and dynamic shared memory for a launch whose longest row has
/// `blocks` complete blocks. The global kernel uses no dynamic shared memory;
/// it requests some anyway to cap its CTAs per SM (2 up to 128K tokens, 1
/// beyond) so the rows in flight fit L2: 18.5 -> 6.4 ms at 128K, 44 -> 17 ms
/// at 256K per 8192 rows. Scheduling only; the selection is unchanged.
pub(super) fn topk_variant(k: &Kernels, blocks: usize) -> (KernelHandle, u32) {
    if blocks <= 4096 {
        (k.topk, 0)
    } else if topk_staged(blocks) {
        (k.topk, ((blocks - 4096) * 4) as u32)
    } else {
        (k.topk_global, topk_global_occupancy_smem(blocks) as u32)
    }
}

pub(super) fn topk_global_occupancy_smem(blocks: usize) -> usize {
    if TOPK_GLOBAL_L2_BYTES / (GB10_SMS * blocks * 4) >= 2 {
        40 * 1024
    } else {
        MAX_TOPK_SMEM
    }
}

pub(super) struct ScoreV2Kernels {
    pub score: KernelHandle,
    pub topk: KernelHandle,
    pub topk_global: KernelHandle,
}

pub(super) fn score_v2_kernels(
    gpu: &dyn GpuBackend,
) -> Result<Option<(&'static ScoreV2Kernels, ScoreV2Mode)>> {
    let Some(mode) = score_v2_mode()? else {
        return Ok(None);
    };
    static K: OnceLock<Result<ScoreV2Kernels, String>> = OnceLock::new();
    let k = K
        .get_or_init(|| {
            let load = |m: &str, f: &str| gpu.kernel(m, f).map_err(|e| format!("{m}::{f}: {e}"));
            let sel = "qwen4_qsa_prefill_select";
            let k = ScoreV2Kernels {
                score: load(sel, "qwen4_qsa_prefill_score_v2")?,
                topk: load(sel, "qwen4_qsa_prefill_topk_v2")?,
                topk_global: load(sel, "qwen4_qsa_prefill_topk_global_v2")?,
            };
            gpu.set_kernel_max_dynamic_shared_memory(k.topk, MAX_TOPK_SMEM as u32)
                .and_then(|()| {
                    gpu.set_kernel_max_dynamic_shared_memory(k.topk_global, MAX_TOPK_SMEM as u32)
                })
                .map_err(|e| e.to_string())?;
            Ok(k)
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{SCORE_V2_SELECTOR}: kernels unavailable: {e}"))?;
    Ok(Some((k, mode)))
}

pub(super) fn topk_v2_variant(k: &ScoreV2Kernels, blocks: usize) -> (KernelHandle, u32) {
    if blocks <= 4096 {
        (k.topk, 0)
    } else if topk_staged(blocks) {
        (k.topk, ((blocks - 4096) * 4) as u32)
    } else {
        (k.topk_global, topk_global_occupancy_smem(blocks) as u32)
    }
}

pub(super) fn iq_norm_rope_fused_kernel(gpu: &dyn GpuBackend) -> Result<Option<KernelHandle>> {
    if !prep_fused_selected()? {
        return Ok(None);
    }
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    let k = K
        .get_or_init(|| {
            gpu.kernel("qwen4_qsa_tile", "qwen4_qsa_iq_norm_rope_fused")
                .map_err(|e| format!("qwen4_qsa_tile::qwen4_qsa_iq_norm_rope_fused: {e}"))
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{PREP_FUSED_SELECTOR}: kernel unavailable: {e}"))?;
    Ok(Some(*k))
}

/// The two kernels of `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK` (same module): the score kernel with
/// group maxima and the compact select `qwen4_qsa_prefill_topk_fused_c3` (three CTAs per SM;
/// micro-queue 121: 3.57 vs 4.73 ms for the staged select at the 128K last chunk, same masks).
pub(super) struct FusedKernels {
    pub score_gmax: KernelHandle,
    pub topk_fused: KernelHandle,
}

/// Whether `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK=1` (strict; read once).
pub fn fused_topk_selected() -> Result<bool> {
    fused_topk::selected().map_err(|e| anyhow::anyhow!(e))
}

/// The fused-select kernels when the selector is on, loaded once. A build
/// without them logs one `QWEN4_PREFILL_QSA_FUSED_TOPK_REFUSED` line and the
/// chunk keeps the unfused select (same masks either way).
pub(super) fn fused_kernels(gpu: &dyn GpuBackend) -> Result<Option<&'static FusedKernels>> {
    if !fused_topk_selected()? {
        return Ok(None);
    }
    fused_topk::min_blocks().map_err(|e| anyhow::anyhow!(e))?;
    let score_v2 = score_v2_mode()?.is_some();
    static K: OnceLock<Option<FusedKernels>> = OnceLock::new();
    Ok(K.get_or_init(|| {
        let sel = "qwen4_qsa_prefill_select";
        let score_gmax_name = if score_v2 {
            "qwen4_qsa_prefill_score_gmax_v2"
        } else {
            "qwen4_qsa_prefill_score_gmax"
        };
        let topk_fused_name = "qwen4_qsa_prefill_topk_fused_c3_v2";
        let loaded = (|| -> Result<FusedKernels> {
            let score_gmax = gpu.kernel(sel, score_gmax_name)?;
            let topk_fused = gpu.kernel(sel, topk_fused_name)?;
            gpu.set_kernel_max_dynamic_shared_memory(
                topk_fused,
                fused_topk::MAX_DYN_SMEM as u32,
            )?;
            Ok(FusedKernels {
                score_gmax,
                topk_fused,
            })
        })();
        match loaded {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!(
                    "QWEN4_PREFILL_QSA_FUSED_TOPK_REFUSED selector={} reason=kernels_unavailable err={e:#}",
                    fused_topk::SELECTOR
                );
                None
            }
        }
    })
    .as_ref())
}

/// Growth step of the scratch: a long request grows it a few times, not once
/// per chunk (the mask region grows ~2 MB per 8192-row chunk).
const SCRATCH_STEP: usize = 32 << 20;

/// Grow-only scratch shared by every attention layer; all of them run on the
/// one prefill stream, the only stream that touches it. Never freed: a free
/// here would run mid-request (the GB10 free-under-traffic class), so growing
/// retires the old block (kept, not freed). [`reserve`] sizes it at load for
/// the largest chunk the engine admits, so a request never grows it.
pub(super) fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scratch lock poisoned"))?;
    let retired = match *guard {
        Some((ptr, cap)) if cap >= bytes => return Ok(ptr),
        Some((_, cap)) => cap,
        None => 0,
    };
    let cap = scratch_capacity(bytes);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "qsa_prefill_sparse")?;
    if retired > 0 {
        tracing::warn!(
            "QWEN4_PREFILL_QSA scratch grown bytes={cap} retired={retired} freed=0 (past the load-time reserve)"
        );
    } else {
        tracing::info!("QWEN4_PREFILL_QSA scratch allocated bytes={cap}");
    }
    *guard = Some((ptr, cap));
    Ok(ptr)
}

/// Scratch bytes covering every sparse chunk of at most `max_rows` rows that
/// ends at or before `max_seq_len`. For a fixed chunk end the plan only grows
/// with its rows, so the widest chunk per end is swept over every end (the
/// last chunk is not always the largest: the score batch shrinks as ld grows).
pub(super) fn reserve_bytes(
    max_rows: usize,
    max_seq_len: usize,
    v2: bool,
    fused: bool,
) -> Result<usize> {
    ensure!(max_rows > 0, "{SELECTOR}: empty reserve");
    let first_sparse_end = DENSE_LAST_POS + 2;
    let mut bytes = 0;
    let ends = (first_sparse_end..=max_seq_len)
        .step_by(32 * COMPRESS_RATIO)
        .chain((max_seq_len >= first_sparse_end).then_some(max_seq_len));
    for end in ends {
        let rows = max_rows.min(end);
        let plan = SparsePlan::for_chunk(rows, end - rows, v2, fused)?;
        bytes = bytes.max(plan.bytes);
    }
    Ok(bytes)
}

/// Load-time sizing of the sparse-core scratch for chunks of up to `max_rows`
/// rows within `max_seq_len` (no-op unless the sparse core is selected), so no
/// request allocates, grows or frees it.
pub fn reserve(gpu: &dyn GpuBackend, max_rows: usize, max_seq_len: usize) -> Result<()> {
    let fused = fused_topk_selected()?;
    if !selected()? {
        if fused {
            tracing::warn!(
                "QWEN4_PREFILL_QSA_FUSED_TOPK_REFUSED selector={} reason=qsa_sparse_off (needs {SELECTOR}=1)",
                fused_topk::SELECTOR
            );
        }
        return Ok(());
    }
    let bytes = reserve_bytes(max_rows, max_seq_len, v2_selected()?, fused)?;
    if bytes == 0 {
        return Ok(());
    }
    let cap_bytes = if fused && max_seq_len >= 65_536 {
        bytes.max(160 << 20)
    } else {
        bytes
    };
    scratch(gpu, cap_bytes)?;
    tracing::info!(
        "QWEN4_PREFILL_SCRATCH_RESERVED site=qsa_prefill_sparse rows={max_rows} max_seq_len={max_seq_len} plan_bytes={bytes} bytes={} never_freed=1",
        scratch_capacity(cap_bytes)
    );
    Ok(())
}

pub(super) fn scratch_capacity(bytes: usize) -> usize {
    bytes.next_multiple_of(SCRATCH_STEP)
}

/// Device inputs of one chunk's sparse core.
#[derive(Clone, Copy, Debug)]
pub struct SparseChunk {
    /// Attention input rows `[rows][hidden]` bf16 (what QKV and QSA project).
    pub hidden: DevicePtr,
    /// Post-norm, post-RoPE queries: row r, head h at `q + (r * q_stride + h * 256)`
    /// bf16 elements (the strided fused-QKV rows; no compaction needed).
    pub q: DevicePtr,
    pub q_stride: u32,
    /// Output `[rows][24*256]` bf16, contiguous.
    pub out: DevicePtr,
    /// Sigmoid output gate `[rows]` at `gate_stride` elements (24*256 wide),
    /// fused into the core's epilogue: out = bf16(bf16(o) * sigmoid(g)).
    pub gate: DevicePtr,
    pub gate_stride: u32,
    pub rows: usize,
    pub start: usize,
    pub attn_layer_idx: usize,
    pub hidden_size: u32,
    pub eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub softmax_scale: f32,
    /// The fused prep wrote this chunk's E4M3 Q rows and K/V shadow
    /// (`ATLAS_QWEN4_PREFILL_QSA_FP8`): run the E4M3 core.
    pub e4m3: bool,
    /// Local query head count (`24` for TP=1, `12` for TP=2, `6` for TP=4).
    pub num_query_heads: usize,
}

#[cfg(test)]
#[path = "prefill_sparse_tests.rs"]
mod tests;

#[cfg(test)]
mod compaction_tests {
    use super::*;

    #[test]
    fn qsa_160mib_compaction_step_and_serve_ceiling() {
        assert_eq!(SCRATCH_STEP, 32 << 20);
        assert_eq!(scratch_capacity(160 << 20), 160 << 20);
        let (max_rows, max_seq_len) = (16_384, 135_168);
        let fused = reserve_bytes(max_rows, max_seq_len, true, true).unwrap();
        assert!(
            fused <= 160 << 20,
            "fused V2 reserve {fused} exceeded 160 MiB"
        );
        assert_eq!(scratch_capacity(fused.max(160 << 20)), 160 << 20);
    }

    #[test]
    fn paged_sliding_window_and_noncausal_block_bounds_are_exact() {
        const COMPUTE_CUH: &str =
            include_str!("../../../../../kernels/gb10/common/prefill_paged_compute.cuh");
        const BR128_CUH: &str =
            include_str!("../../../../../kernels/gb10/common/prefill_paged_nvfp4_br128.cuh");

        assert_eq!(COMPUTE_CUH.matches("if (causal_mask_enabled) {").count(), 2);
        assert_eq!(
            COMPUTE_CUH
                .matches("first_kv_block = (min_q - sliding_window + 1) / BC;")
                .count(),
            2
        );
        assert!(BR128_CUH.contains("if (causal_mask_enabled) {"));

        for bc in [16u32, 32, 64] {
            for w in [1u32, 16, 32, 64, 512, 4096] {
                for min_q in [0u32, w / 2, w.saturating_sub(1), w, w + 1, w + bc, 16_384] {
                    let kv_len = min_q + 128;
                    let all_blocks = kv_len.div_ceil(bc);
                    let first_kv_block = if min_q >= w { (min_q - w + 1) / bc } else { 0 };
                    assert!(first_kv_block <= all_blocks);
                    for skipped in 0..first_kv_block {
                        for k in (skipped * bc)..((skipped + 1) * bc).min(kv_len) {
                            assert!(min_q >= k && min_q - k >= w);
                        }
                    }
                    if first_kv_block > 0 {
                        assert_eq!(first_kv_block, (min_q - w + 1) / bc);
                    }
                    // Non-causal (causal_mask_enabled = false) must keep all_blocks even when q_offset=0, q_tile_end=16
                    let noncausal_blocks = all_blocks;
                    let causal_blocks = all_blocks.min(15 / bc + 1);
                    assert_eq!(noncausal_blocks, all_blocks);
                    assert!(causal_blocks <= noncausal_blocks);
                }
            }
        }
    }
}
