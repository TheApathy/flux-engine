// SPDX-License-Identifier: AGPL-3.0-only

//! Device top-k candidate selection for stochastic sampling
//! (`ATLAS_SAMPLER_DEVICE_TOPK=1`, SPEED_PLAN W1). Off by default: no kernel
//! lookup and no allocation, so the default path is untouched.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use spark_runtime::gpu::HostToDeviceCopy;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::sampler::device_topk::{
    DEVICE_TOPK_K_MAX, DEVICE_TOPK_M_MAX, DEVICE_TOPK_SLICE_M_MAX, DEVICE_TOPK_SLICE_MAX_CTAS,
    DEVICE_TOPK_SLICE_WORKSPACE_BYTES, GUMBEL_VERIFY_MAX_ROWS, GumbelVerifyRowSpec,
    device_topk_enabled, selection_key, topk_slice_grid, use_sliced_topk,
};
use spark_runtime::sampler::f2_certify::{F2_EDITS_MAX, PreF2Edit, SumExpStats};

use super::super::types::TransformerModel;
use spark_runtime::kernel_args::KernelLaunch;

/// Result buffer layout: `[count u32, pad]` (16 bytes), then `M_MAX` ids,
/// then `M_MAX` raw f32 values.
const COUNT_BYTES: usize = 16;
const IDS_OFFSET: usize = COUNT_BYTES;
const VALS_OFFSET: usize = IDS_OFFSET + DEVICE_TOPK_M_MAX * 4;
const TOPK_BYTES: usize = VALS_OFFSET + DEVICE_TOPK_M_MAX * 4;
/// F2 region: edit ids / kinds / values (`F2_EDITS_MAX` each), then the
/// 32-byte stats block written by `edited_sum_exp_bf16`.
const F2_IDS_OFFSET: usize = TOPK_BYTES;
const F2_KIND_OFFSET: usize = F2_IDS_OFFSET + F2_EDITS_MAX * 4;
const F2_VAL_OFFSET: usize = F2_KIND_OFFSET + F2_EDITS_MAX * 4;
const F2_EDITS_BYTES: usize = F2_EDITS_MAX * 12;
const F2_OUT_OFFSET: usize = F2_VAL_OFFSET + F2_EDITS_MAX * 4;
const F2_OUT_BYTES: usize = 32;
/// 16-byte-aligned scratch region for 2-stage multi-CTA sliced top-k
/// (`topk_slice_stage1_bf16` + `topk_slice_stage2_merge`):
/// `[MAX_CTAS * SLICE_M_MAX]` u64 keys, `[MAX_CTAS * SLICE_M_MAX]` f32 raw
/// values, and `[MAX_CTAS]` u32 per-slice counts.
const SLICE_KEYS_OFFSET: usize = (F2_OUT_OFFSET + F2_OUT_BYTES + 15) & !15;
const SLICE_VALS_OFFSET: usize =
    SLICE_KEYS_OFFSET + DEVICE_TOPK_SLICE_MAX_CTAS * DEVICE_TOPK_SLICE_M_MAX * 8;
const SLICE_COUNTS_OFFSET: usize =
    SLICE_VALS_OFFSET + DEVICE_TOPK_SLICE_MAX_CTAS * DEVICE_TOPK_SLICE_M_MAX * 4;
pub(crate) const DEVICE_TOPK_BUF_BYTES: usize =
    SLICE_KEYS_OFFSET + DEVICE_TOPK_SLICE_WORKSPACE_BYTES;

/// Reusable 8 KiB device staging buffer layout for batched GPU target
/// Gumbel-Max verify (`sampler_topk_candidates::target_gumbel_verify_rows_bf16`):
///   - `[0..2048)`    `seeds`:        `[256] u64`
///   - `[2048..4096)` `positions`:    `[256] u64`
///   - `[4096..5120)` `temperatures`: `[256] f32`
///   - `[5120..6144)` `top_ks`:       `[256] u32`
///   - `[6144..7168)` `top_ps`:       `[256] f32`
///   - `[7168..8192)` `out_tokens`:   `[256] u32`
const GUMBEL_SEEDS_OFFSET: usize = 0;
const GUMBEL_POSITIONS_OFFSET: usize = GUMBEL_SEEDS_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 8;
const GUMBEL_TEMPS_OFFSET: usize = GUMBEL_POSITIONS_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 8;
const GUMBEL_TOPKS_OFFSET: usize = GUMBEL_TEMPS_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 4;
const GUMBEL_TOPPS_OFFSET: usize = GUMBEL_TOPKS_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 4;
const GUMBEL_H2D_BYTES: usize = GUMBEL_TOPPS_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 4;
pub(super) const GUMBEL_OUT_OFFSET: usize = GUMBEL_H2D_BYTES;
pub(crate) const GUMBEL_VERIFY_STAGE_BYTES: usize = GUMBEL_OUT_OFFSET + GUMBEL_VERIFY_MAX_ROWS * 4;
/// Threads per CTA for `sampler_topk_candidates::target_gumbel_verify_rows_bf16`
/// (16 warps, matching `__launch_bounds__(512, 3)` in `sampler_topk_candidates.cu`).
const GUMBEL_VERIFY_BLOCK_THREADS: u32 = 512;
/// Output sentinel written by `target_gumbel_verify_rows_bf16` when a row is
/// skipped via `temperature < 0.0` (0 logits reads; caller falls back on host).
pub(crate) const GUMBEL_VERIFY_SKIP_SENTINEL: u32 = u32::MAX;

/// Packed output word returned by `target_rejection_verify_rows_bf16` when a
/// row is skipped via `temperature < 0.0` (bit 31 = 0, payload = `0x7FFF_FFFF`).
#[allow(dead_code)]
pub const HOST_POLICY_FALLBACK_TOKEN: u32 = 0x7FFF_FFFF;

/// Sentinel `q_row_idx` marking a bonus or greedy row with no draft distribution
/// (`q_cand_ids` / `q_probs` row is not read).
#[allow(dead_code)]
pub const REJECTION_BONUS_Q_ROW_IDX: u32 = 0xFFFF;

/// Sentinel mask token ID (`u32::MAX`) indicating an unused token-suppression slot.
#[allow(dead_code)]
pub const REJECTION_NO_MASK_TOKEN: u32 = u32::MAX;

/// Maximum verify rows staged per `target_rejection_verify_rows_bf16` launch.
#[allow(dead_code)]
pub const REJECTION_MAX_ROWS: usize = 256;

/// Default draft-distribution candidate stride (`q_cand_ids` / `q_probs` columns).
#[allow(dead_code)]
pub const REJECTION_Q_STRIDE: usize = 20;

/// Maximum draft-distribution rows provisioned in GPU `(q_cand_ids, q_probs)`.
#[allow(dead_code)]
pub const REJECTION_Q_ROW_CAP: usize = 512;

/// Byte size of the `q_cand_ids: [REJECTION_Q_ROW_CAP, REJECTION_Q_STRIDE] u32`
/// slab at offset `0` of `target_rejection_q_buf`.
pub(crate) const REJECTION_Q_IDS_BYTES: usize = REJECTION_Q_ROW_CAP * REJECTION_Q_STRIDE * 4;
/// Byte offset of `q_probs: [REJECTION_Q_ROW_CAP, REJECTION_Q_STRIDE] f32`
/// inside `target_rejection_q_buf`.
pub(crate) const REJECTION_Q_PROBS_OFFSET: usize = REJECTION_Q_IDS_BYTES;
/// Total byte size of the shared `(q_cand_ids, q_probs)` GPU scratch buffer
/// (`512 * 20 * (4 + 4) = 81_920` bytes = 80 KiB).
pub(crate) const REJECTION_Q_BUF_BYTES: usize = REJECTION_Q_ROW_CAP * REJECTION_Q_STRIDE * 8;

/// Threads per CTA for `sampler_topk_candidates::target_rejection_verify_rows_bf16`
/// (16 warps, matching `__launch_bounds__(512, 3)` in `sampler_topk_candidates.cu`).
const REJECTION_VERIFY_BLOCK_THREADS: u32 = 512;

/// Reusable 13 KiB device staging buffer layout for batched GPU maximal-coupling
/// rejection verify (`sampler_topk_candidates::target_rejection_verify_rows_bf16`):
///   - `[0..2048)`      `seeds`:         `[256] u64`
///   - `[2048..4096)`   `abs_pos`:       `[256] u64`
///   - `[4096..5120)`   `temperatures`:  `[256] f32`
///   - `[5120..6144)`   `top_ks`:        `[256] u32`
///   - `[6144..7168)`   `top_ps`:        `[256] f32`
///   - `[7168..8192)`   `min_ps`:        `[256] f32`
///   - `[8192..9216)`   `draft_tokens`:  `[256] u32`
///   - `[9216..10240)`  `q_row_indices`: `[256] u32`
///   - `[10240..11264)` `mask_tok0`:     `[256] u32`
///   - `[11264..12288)` `mask_tok1`:     `[256] u32`
///   - `[12288..13312)` `out_words`:     `[256] u32`
#[allow(dead_code)]
const REJECTION_SEEDS_OFFSET: usize = 0;
#[allow(dead_code)]
const REJECTION_ABS_POS_OFFSET: usize = REJECTION_SEEDS_OFFSET + REJECTION_MAX_ROWS * 8;
#[allow(dead_code)]
const REJECTION_TEMPS_OFFSET: usize = REJECTION_ABS_POS_OFFSET + REJECTION_MAX_ROWS * 8;
#[allow(dead_code)]
const REJECTION_TOPKS_OFFSET: usize = REJECTION_TEMPS_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_TOPPS_OFFSET: usize = REJECTION_TOPKS_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_MINPS_OFFSET: usize = REJECTION_TOPPS_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_DRAFT_TOKS_OFFSET: usize = REJECTION_MINPS_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_Q_ROW_IDX_OFFSET: usize = REJECTION_DRAFT_TOKS_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_MASK0_OFFSET: usize = REJECTION_Q_ROW_IDX_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
const REJECTION_MASK1_OFFSET: usize = REJECTION_MASK0_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
pub const REJECTION_H2D_BYTES: usize = REJECTION_MASK1_OFFSET + REJECTION_MAX_ROWS * 4;
#[allow(dead_code)]
pub const REJECTION_OUT_OFFSET: usize = REJECTION_H2D_BYTES;
#[allow(dead_code)]
pub const REJECTION_VERIFY_STAGE_BYTES: usize = REJECTION_OUT_OFFSET + REJECTION_MAX_ROWS * 4;

/// Packed 32-bit result word written by `target_rejection_verify_rows_bf16`:
/// - Bit 31: `1` if the draft token was accepted, `0` otherwise.
/// - Bits 30..0: token ID, or [`HOST_POLICY_FALLBACK_TOKEN`] (`0x7FFF_FFFF`)
///   when the row was skipped via `temperature < 0.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub struct PackedRejectionWord(pub u32);

#[allow(dead_code)]
impl PackedRejectionWord {
    /// Returns `true` iff bit 31 is set (draft token accepted).
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        (self.0 >> 31) != 0
    }

    /// Returns the 31-bit token payload (bits 30..0), or
    /// [`HOST_POLICY_FALLBACK_TOKEN`] when [`Self::is_host_fallback`] is `true`.
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.0 & 0x7FFF_FFFF
    }

    /// Returns `true` iff the row was skipped via `temperature < 0.0` for
    /// host-side policy fallback (`word == HOST_POLICY_FALLBACK_TOKEN`).
    #[must_use]
    pub const fn is_host_fallback(self) -> bool {
        self.0 == HOST_POLICY_FALLBACK_TOKEN
    }
}

/// Per-row parameters for `sampler_topk_candidates::target_rejection_verify_rows_bf16`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub struct RejectionVerifyRowSpec {
    /// Row index in the `[num_rows, vocab_size]` BF16 verify logits buffer.
    pub row_idx: usize,
    /// Sequence base RNG seed.
    pub seed: u64,
    /// Absolute token position for this verify row.
    pub abs_pos: u64,
    /// Sampling temperature (`< 0.0` skip sentinel, `== 0.0` greedy, `> 0.0` maximal coupling).
    pub temperature: f32,
    /// Request `top_k` (`0` = full vocabulary).
    pub top_k: u32,
    /// Request `top_p` (`(0.0, 1.0]` finite; `>= 1.0` disables nucleus truncation).
    pub top_p: f32,
    /// Request `min_p` (`<= 0.0` disables min-p truncation).
    pub min_p: f32,
    /// Drafted token ID for draft rows (ignored on bonus rows).
    pub draft_token: u32,
    /// Row index in `(q_cand_ids, q_probs)`, or [`REJECTION_BONUS_Q_ROW_IDX`] for bonus/greedy.
    pub q_row_idx: u32,
    /// First suppressed token ID (`u32::MAX` = none).
    pub mask_tok0: u32,
    /// Second suppressed token ID (`u32::MAX` = none).
    pub mask_tok1: u32,
}

type SlicedTopkKernels = Option<(KernelHandle, KernelHandle)>;

/// Per-buffer cache of `(topk_slice_stage1_bf16, topk_slice_stage2_merge)`
/// handles resolved once per model instance so `topk_candidates_dispatch`
/// never repeats `cuModuleGetFunction` on the decode hot path.
static SLICED_TOPK_CACHE: Mutex<Vec<(u64, SlicedTopkKernels)>> = Mutex::new(Vec::new());

fn resolve_sliced_topk_kernels(gpu: &dyn GpuBackend) -> SlicedTopkKernels {
    let stage1 = gpu
        .kernel("sampler_topk_candidates", "topk_slice_stage1_bf16")
        .ok()
        .filter(|h| h.0 != 0)?;
    let stage2 = gpu
        .kernel("sampler_topk_candidates", "topk_slice_stage2_merge")
        .ok()
        .filter(|h| h.0 != 0)?;
    Some((stage1, stage2))
}

fn cached_sliced_topk_kernels(gpu: &dyn GpuBackend, buf: DevicePtr) -> SlicedTopkKernels {
    if let Some(&(_, cached)) = SLICED_TOPK_CACHE
        .lock()
        .iter()
        .find(|&&(key, _)| key == buf.0)
    {
        return cached;
    }
    let resolved = resolve_sliced_topk_kernels(gpu);
    let mut cache = SLICED_TOPK_CACHE.lock();
    if let Some(slot) = cache.iter_mut().find(|(key, _)| *key == buf.0) {
        slot.1 = resolved;
    } else {
        cache.push((buf.0, resolved));
    }
    resolved
}

/// Load the kernel and its result buffer when the route is on. A missing
/// kernel is a load error (refusal), never a silent host fallback.
pub(crate) fn load_device_topk(
    gpu: &dyn GpuBackend,
) -> Result<(KernelHandle, KernelHandle, DevicePtr)> {
    if !device_topk_enabled() {
        return Ok((KernelHandle(0), KernelHandle(0), DevicePtr(0)));
    }
    let kernel = gpu
        .kernel("sampler_topk_candidates", "topk_candidates_bf16")
        .context("ATLAS_SAMPLER_DEVICE_TOPK=1: kernel sampler_topk_candidates::topk_candidates_bf16 missing")?;
    let f2_kernel = gpu
        .kernel("sampler_topk_candidates", "edited_sum_exp_bf16")
        .context("ATLAS_SAMPLER_DEVICE_TOPK=1: kernel sampler_topk_candidates::edited_sum_exp_bf16 missing")?;
    let buf = gpu
        .alloc(DEVICE_TOPK_BUF_BYTES)
        .context("ATLAS_SAMPLER_DEVICE_TOPK=1: result buffer allocation failed")?;
    let sliced = resolve_sliced_topk_kernels(gpu);
    let mut cache = SLICED_TOPK_CACHE.lock();
    if let Some(slot) = cache.iter_mut().find(|(key, _)| *key == buf.0) {
        slot.1 = sliced;
    } else {
        cache.push((buf.0, sliced));
    }
    Ok((kernel, f2_kernel, buf))
}

/// Optionally load `sampler_topk_candidates::target_gumbel_verify_rows_bf16`
/// and allocate its reusable 8 KiB staging buffer once at model construction.
///
/// Missing kernel on non-GB10 / universal targets returns
/// `(KernelHandle(0), DevicePtr::NULL)` with zero device allocation so callers
/// fall back cleanly to their per-row sampler (`Ok(None)`).
pub(crate) fn load_batched_gumbel_verify(
    gpu: &dyn GpuBackend,
) -> Result<(KernelHandle, DevicePtr)> {
    let kernel = match gpu.kernel("sampler_topk_candidates", "target_gumbel_verify_rows_bf16") {
        Ok(k) if k.0 != 0 => k,
        _ => return Ok((KernelHandle(0), DevicePtr::NULL)),
    };
    let buf = gpu
        .alloc(GUMBEL_VERIFY_STAGE_BYTES)
        .context("target_gumbel_verify_rows_bf16: staging buffer allocation failed")?;
    Ok((kernel, buf))
}

/// Optionally load `sampler_topk_candidates::target_rejection_verify_rows_bf16`
/// and allocate its reusable 13 KiB staging buffer plus the shared 80 KiB
/// `(q_cand_ids, q_probs)` draft-distribution buffer once at model construction.
///
/// Missing kernel on non-GB10 / CPU mock targets returns
/// `(KernelHandle(0), DevicePtr::NULL, DevicePtr::NULL)` with zero device
/// allocations so callers fall back cleanly.
pub(crate) fn load_batched_rejection_verify(
    gpu: &dyn GpuBackend,
) -> Result<(KernelHandle, DevicePtr, DevicePtr)> {
    let kernel = match gpu.kernel(
        "sampler_topk_candidates",
        "target_rejection_verify_rows_bf16",
    ) {
        Ok(k) if k.0 != 0 => k,
        _ => return Ok((KernelHandle(0), DevicePtr::NULL, DevicePtr::NULL)),
    };
    let stage_buf = gpu
        .alloc(REJECTION_VERIFY_STAGE_BYTES)
        .context("target_rejection_verify_rows_bf16: staging buffer allocation failed")?;
    let q_buf = match gpu
        .alloc(REJECTION_Q_BUF_BYTES)
        .context("target_rejection_verify_rows_bf16: shared q buffer allocation failed")
    {
        Ok(ptr) => ptr,
        Err(err) => {
            let _ = gpu.free(stage_buf);
            return Err(err);
        }
    };
    if let Err(err) = gpu
        .memset(q_buf, 0, REJECTION_Q_BUF_BYTES)
        .context("target_rejection_verify_rows_bf16: shared q buffer zero-init failed")
    {
        let _ = gpu.free(q_buf);
        let _ = gpu.free(stage_buf);
        return Err(err);
    }
    Ok((kernel, stage_buf, q_buf))
}

/// Validate one row's unified 3-way sampling parameters for
/// `target_gumbel_verify_rows_bf16`:
/// - `temperature < 0.0` (finite): instant skip sentinel ([`GUMBEL_VERIFY_SKIP_SENTINEL`])
/// - `temperature == 0.0`: 1-pass exact greedy argmax (`top_k` / `top_p` unused)
/// - `temperature > 0.0` (finite): 2-pass radix `top_k` + `top_p` + keyed Gumbel-Max
fn validate_gumbel_verify_row(req: &GumbelVerifyRowSpec, slot: usize) -> Result<()> {
    anyhow::ensure!(
        req.temperature.is_finite(),
        "target_gumbel_verify_rows_bf16 invalid temperature={} at slot {slot}",
        req.temperature
    );
    if req.temperature <= 0.0 {
        // `< 0.0` skips the row with `GUMBEL_VERIFY_SKIP_SENTINEL`;
        // `== 0.0` runs exact 1-pass greedy argmax ignoring `top_k` / `top_p`.
        return Ok(());
    }
    // Host treats `top_p <= 0.0` as top-1 truncation whereas the device
    // kernel treats `top_p <= 0.0` as disabled; refuse non-positive `top_p`
    // so both paths remain 100% lossless.
    anyhow::ensure!(
        req.top_p > 0.0 && req.top_p.is_finite(),
        "target_gumbel_verify_rows_bf16 invalid top_p={} at slot {slot}",
        req.top_p
    );
    let top_k_ok = (1..=DEVICE_TOPK_K_MAX).contains(&(req.top_k as usize))
        || (req.top_k == 0 && req.top_p >= 1.0);
    anyhow::ensure!(
        top_k_ok,
        "target_gumbel_verify_rows_bf16 unsupported (top_k={}, top_p={}) at slot {slot}",
        req.top_k,
        req.top_p
    );
    Ok(())
}

/// Validate and pack one chunk (`<= GUMBEL_VERIFY_MAX_ROWS`) of per-row
/// Gumbel-Max verify parameters into the single contiguous H2D staging block.
pub(crate) fn encode_gumbel_verify_chunk(
    reqs: &[GumbelVerifyRowSpec],
) -> Result<[u8; GUMBEL_H2D_BYTES]> {
    anyhow::ensure!(
        reqs.len() <= GUMBEL_VERIFY_MAX_ROWS,
        "target_gumbel_verify_rows_bf16 chunk size {} exceeds max {GUMBEL_VERIFY_MAX_ROWS}",
        reqs.len()
    );
    let mut stage = [0u8; GUMBEL_H2D_BYTES];
    for (slot, req) in reqs.iter().enumerate() {
        validate_gumbel_verify_row(req, slot)?;
        stage[GUMBEL_SEEDS_OFFSET + slot * 8..][..8].copy_from_slice(&req.seed.to_le_bytes());
        stage[GUMBEL_POSITIONS_OFFSET + slot * 8..][..8]
            .copy_from_slice(&req.abs_pos.to_le_bytes());
        stage[GUMBEL_TEMPS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.temperature.to_bits().to_le_bytes());
        stage[GUMBEL_TOPKS_OFFSET + slot * 4..][..4].copy_from_slice(&req.top_k.to_le_bytes());
        stage[GUMBEL_TOPPS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.top_p.to_bits().to_le_bytes());
    }
    Ok(stage)
}

/// Launch `target_gumbel_verify_rows_bf16` across maximal contiguous `row_idx`
/// runs in `chunk` (1 launch for a homogeneous frame, e.g. `16 × 5 = 80` rows
/// with `.grid([80, 1, 1]).block([512, 1, 1])`). Metadata must already be
/// staged in `buf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_target_gumbel_verify_runs(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_base: DevicePtr,
    buf: DevicePtr,
    chunk: &[GumbelVerifyRowSpec],
    vocab: u32,
    vocab_stride: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(vocab > 0, "target_gumbel_verify_rows_bf16 vocab=0");
    anyhow::ensure!(
        vocab_stride >= vocab as usize,
        "target_gumbel_verify_rows_bf16 vocab_stride={vocab_stride} < vocab={vocab}"
    );
    let vocab_stride_u32 = u32::try_from(vocab_stride).with_context(|| {
        format!("target_gumbel_verify_rows_bf16 vocab_stride={vocab_stride} exceeds u32::MAX")
    })?;
    anyhow::ensure!(
        chunk.len() <= GUMBEL_VERIFY_MAX_ROWS,
        "target_gumbel_verify_rows_bf16 chunk size {} exceeds max {GUMBEL_VERIFY_MAX_ROWS}",
        chunk.len()
    );
    // Pre-check row byte offsets before issuing any launch.
    for req in chunk {
        req.row_idx
            .checked_mul(vocab_stride)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("target_gumbel_verify_rows_bf16 row offset overflow"))?;
    }

    let mut run_start = 0usize;
    while run_start < chunk.len() {
        let base_row = chunk[run_start].row_idx;
        let mut run_len = 1usize;
        while run_start + run_len < chunk.len()
            && chunk[run_start + run_len].row_idx == base_row.saturating_add(run_len)
        {
            run_len += 1;
        }
        let row_byte_off = base_row * vocab_stride * 2;
        KernelLaunch::new(gpu, kernel)
            .grid([run_len as u32, 1, 1])
            .block([GUMBEL_VERIFY_BLOCK_THREADS, 1, 1])
            .arg_ptr(logits_base.offset(row_byte_off))
            .arg_ptr(buf.offset(GUMBEL_SEEDS_OFFSET + run_start * 8))
            .arg_ptr(buf.offset(GUMBEL_POSITIONS_OFFSET + run_start * 8))
            .arg_ptr(buf.offset(GUMBEL_TEMPS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(GUMBEL_TOPKS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(GUMBEL_TOPPS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(GUMBEL_OUT_OFFSET + run_start * 4))
            .arg_u32(vocab)
            .arg_u32(vocab_stride_u32)
            .launch(stream)?;
        run_start += run_len;
    }
    Ok(())
}

/// Returns `true` iff `chunk` can execute as ONE single deterministic
/// `target_gumbel_verify_rows_bf16` kernel launch over contiguous rows
/// `0..chunk.len()` inside a captured verify CUDA graph:
/// - `1 <= chunk.len() <= GUMBEL_VERIFY_MAX_ROWS` (256),
/// - `chunk[i].row_idx == i` for every `i` (so `logits_base` and `buf` start at
///   offset `0` with fixed device pointers and `grid = [chunk.len(), 1, 1]`),
/// - every row's `(temperature, top_k, top_p)` passes the exact unified 3-way
///   validation enforced by [`encode_gumbel_verify_chunk`] (`temp < 0.0` skip
///   sentinel, `temp == 0.0` exact greedy argmax, or `temp > 0.0` lossless
///   Gumbel-Max).
///
/// Because `target_gumbel_verify_rows_bf16` reads per-row `seeds`, `positions`,
/// `temperatures`, `top_ks`, and `top_ps` from device staging memory (`buf`)
/// rather than scalar kernel arguments, per-row sampling parameters may vary
/// freely across rows and across steps without recapturing the CUDA graph.
#[must_use]
pub(super) fn can_launch_single_gumbel_verify_run(chunk: &[GumbelVerifyRowSpec]) -> bool {
    if chunk.is_empty() || chunk.len() > GUMBEL_VERIFY_MAX_ROWS {
        return false;
    }
    chunk.iter().enumerate().all(|(i, req)| {
        req.row_idx == i
            && req.temperature.is_finite()
            && (req.temperature <= 0.0
                || (req.top_p > 0.0
                    && req.top_p.is_finite()
                    && ((1..=DEVICE_TOPK_K_MAX).contains(&(req.top_k as usize))
                        || (req.top_k == 0 && req.top_p >= 1.0))))
    })
}

/// Launch ONE single CUDA-graph-safe `target_gumbel_verify_rows_bf16` kernel
/// over contiguous rows `0..num_rows` (`1 <= num_rows <= GUMBEL_VERIFY_MAX_ROWS`)
/// using fixed device pointers (`logits_base` and `buf`).
///
/// Per-row `seeds`, `positions`, `temperatures`, `top_ks`, and `top_ps` must
/// already be staged in `buf` via [`encode_gumbel_verify_chunk`] before graph
/// launch or capture.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_target_gumbel_verify_single_run(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_base: DevicePtr,
    buf: DevicePtr,
    num_rows: usize,
    vocab: u32,
    vocab_stride: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=GUMBEL_VERIFY_MAX_ROWS).contains(&num_rows),
        "target_gumbel_verify_rows_bf16 single-run num_rows={num_rows} outside 1..={GUMBEL_VERIFY_MAX_ROWS}"
    );
    anyhow::ensure!(
        kernel.0 != 0 && buf.0 != 0 && logits_base.0 != 0,
        "target_gumbel_verify_rows_bf16 single-run requires non-null kernel, buf, and logits"
    );
    anyhow::ensure!(vocab > 0, "target_gumbel_verify_rows_bf16 vocab=0");
    anyhow::ensure!(
        vocab_stride >= vocab as usize,
        "target_gumbel_verify_rows_bf16 vocab_stride={vocab_stride} < vocab={vocab}"
    );
    let vocab_stride_u32 = u32::try_from(vocab_stride).with_context(|| {
        format!("target_gumbel_verify_rows_bf16 vocab_stride={vocab_stride} exceeds u32::MAX")
    })?;
    KernelLaunch::new(gpu, kernel)
        .grid([num_rows as u32, 1, 1])
        .block([GUMBEL_VERIFY_BLOCK_THREADS, 1, 1])
        .arg_ptr(logits_base)
        .arg_ptr(buf.offset(GUMBEL_SEEDS_OFFSET))
        .arg_ptr(buf.offset(GUMBEL_POSITIONS_OFFSET))
        .arg_ptr(buf.offset(GUMBEL_TEMPS_OFFSET))
        .arg_ptr(buf.offset(GUMBEL_TOPKS_OFFSET))
        .arg_ptr(buf.offset(GUMBEL_TOPPS_OFFSET))
        .arg_ptr(buf.offset(GUMBEL_OUT_OFFSET))
        .arg_u32(vocab)
        .arg_u32(vocab_stride_u32)
        .launch(stream)
}

/// Validate and decode `num_rows` little-endian `u32` token IDs read back from
/// `buf.offset(GUMBEL_OUT_OFFSET)`.
///
/// Rows skipped via `temperature < 0.0` yield [`GUMBEL_VERIFY_SKIP_SENTINEL`]
/// (`u32::MAX`) and are preserved verbatim so mixed batches can route skipped
/// rows to host-side sampling while consuming in-graph tokens for active rows.
pub(super) fn decode_gumbel_verify_tokens(
    bytes: &[u8],
    num_rows: usize,
    vocab_size: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        bytes.len() == num_rows * 4,
        "target_gumbel_verify_rows_bf16 D2H byte len {} != {}",
        bytes.len(),
        num_rows * 4
    );
    let mut out = Vec::with_capacity(num_rows);
    for word in bytes.chunks_exact(4) {
        let tok = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        anyhow::ensure!(
            (tok as usize) < vocab_size || tok == GUMBEL_VERIFY_SKIP_SENTINEL,
            "target_gumbel_verify_rows_bf16 returned out-of-range token {tok} >= vocab {vocab_size}"
        );
        out.push(tok);
    }
    Ok(out)
}

/// Validate one row's unified 3-way parameters for
/// `target_rejection_verify_rows_bf16`.
#[allow(dead_code)]
fn validate_rejection_verify_row(req: &RejectionVerifyRowSpec, slot: usize) -> Result<()> {
    anyhow::ensure!(
        req.temperature.is_finite(),
        "target_rejection_verify_rows_bf16 invalid temperature={} at slot {slot}",
        req.temperature
    );
    if req.temperature <= 0.0 {
        // `< 0.0` skips the row with `HOST_POLICY_FALLBACK_TOKEN`;
        // `== 0.0` runs exact 1-pass greedy argmax ignoring `top_k`/`top_p`/`min_p`.
        return Ok(());
    }
    anyhow::ensure!(
        req.top_p > 0.0 && req.top_p.is_finite(),
        "target_rejection_verify_rows_bf16 invalid top_p={} at slot {slot}",
        req.top_p
    );
    anyhow::ensure!(
        req.min_p >= 0.0 && req.min_p.is_finite(),
        "target_rejection_verify_rows_bf16 invalid min_p={} at slot {slot}",
        req.min_p
    );
    Ok(())
}

/// Validate and pack one chunk (`<= REJECTION_MAX_ROWS`) of per-row
/// maximal-coupling rejection verify parameters into the single contiguous H2D
/// staging block.
#[allow(dead_code)]
pub(crate) fn encode_rejection_verify_chunk(
    reqs: &[RejectionVerifyRowSpec],
) -> Result<[u8; REJECTION_H2D_BYTES]> {
    anyhow::ensure!(
        reqs.len() <= REJECTION_MAX_ROWS,
        "target_rejection_verify_rows_bf16 chunk size {} exceeds max {REJECTION_MAX_ROWS}",
        reqs.len()
    );
    let mut stage = [0u8; REJECTION_H2D_BYTES];
    for (slot, req) in reqs.iter().enumerate() {
        validate_rejection_verify_row(req, slot)?;
        stage[REJECTION_SEEDS_OFFSET + slot * 8..][..8].copy_from_slice(&req.seed.to_le_bytes());
        stage[REJECTION_ABS_POS_OFFSET + slot * 8..][..8]
            .copy_from_slice(&req.abs_pos.to_le_bytes());
        stage[REJECTION_TEMPS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.temperature.to_bits().to_le_bytes());
        stage[REJECTION_TOPKS_OFFSET + slot * 4..][..4].copy_from_slice(&req.top_k.to_le_bytes());
        stage[REJECTION_TOPPS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.top_p.to_bits().to_le_bytes());
        stage[REJECTION_MINPS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.min_p.to_bits().to_le_bytes());
        stage[REJECTION_DRAFT_TOKS_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.draft_token.to_le_bytes());
        stage[REJECTION_Q_ROW_IDX_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.q_row_idx.to_le_bytes());
        stage[REJECTION_MASK0_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.mask_tok0.to_le_bytes());
        stage[REJECTION_MASK1_OFFSET + slot * 4..][..4]
            .copy_from_slice(&req.mask_tok1.to_le_bytes());
    }
    Ok(stage)
}

/// Launch `target_rejection_verify_rows_bf16` across maximal contiguous `row_idx`
/// runs in `chunk` (1 launch for a homogeneous frame, e.g. `16 × 5 = 80` rows
/// with `.grid([80, 1, 1]).block([512, 1, 1])`). Metadata must already be
/// staged in `buf`.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_target_rejection_verify_runs(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_base: DevicePtr,
    q_cand_ids: DevicePtr,
    q_probs: DevicePtr,
    buf: DevicePtr,
    chunk: &[RejectionVerifyRowSpec],
    vocab: u32,
    q_row_cap: usize,
    q_stride: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(vocab > 0, "target_rejection_verify_rows_bf16 vocab=0");
    let vocab_i32 = i32::try_from(vocab).with_context(|| {
        format!("target_rejection_verify_rows_bf16 vocab={vocab} exceeds i32::MAX")
    })?;
    let q_row_cap_i32 = i32::try_from(q_row_cap).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_row_cap={q_row_cap} exceeds i32::MAX")
    })?;
    let q_stride_i32 = i32::try_from(q_stride).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_stride={q_stride} exceeds i32::MAX")
    })?;
    anyhow::ensure!(
        chunk.len() <= REJECTION_MAX_ROWS,
        "target_rejection_verify_rows_bf16 chunk size {} exceeds max {REJECTION_MAX_ROWS}",
        chunk.len()
    );
    // Pre-check row byte offsets and active draft-row `q_row_idx` bounds before issuing any launch.
    let vocab_usize = vocab as usize;
    for (slot, req) in chunk.iter().enumerate() {
        req.row_idx
            .checked_mul(vocab_usize)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                anyhow::anyhow!("target_rejection_verify_rows_bf16 row offset overflow")
            })?;
        if req.temperature > 0.0 && req.q_row_idx != REJECTION_BONUS_Q_ROW_IDX {
            anyhow::ensure!(
                (req.q_row_idx as usize) < q_row_cap && q_stride > 0,
                "target_rejection_verify_rows_bf16 invalid q_row_idx={} (q_row_cap={q_row_cap}, q_stride={q_stride}) at slot {slot}",
                req.q_row_idx
            );
        }
    }

    let mut run_start = 0usize;
    while run_start < chunk.len() {
        let base_row = chunk[run_start].row_idx;
        let mut run_len = 1usize;
        while run_start + run_len < chunk.len()
            && chunk[run_start + run_len].row_idx == base_row.saturating_add(run_len)
        {
            run_len += 1;
        }
        let row_byte_off = base_row * vocab_usize * 2;
        KernelLaunch::new(gpu, kernel)
            .grid([run_len as u32, 1, 1])
            .block([REJECTION_VERIFY_BLOCK_THREADS, 1, 1])
            .arg_ptr(logits_base.offset(row_byte_off))
            .arg_ptr(q_cand_ids)
            .arg_ptr(q_probs)
            .arg_ptr(buf.offset(REJECTION_SEEDS_OFFSET + run_start * 8))
            .arg_ptr(buf.offset(REJECTION_ABS_POS_OFFSET + run_start * 8))
            .arg_ptr(buf.offset(REJECTION_TEMPS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_TOPKS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_TOPPS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_MINPS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_DRAFT_TOKS_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_Q_ROW_IDX_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_MASK0_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_MASK1_OFFSET + run_start * 4))
            .arg_ptr(buf.offset(REJECTION_OUT_OFFSET + run_start * 4))
            .arg_i32(run_len as i32)
            .arg_i32(vocab_i32)
            .arg_i32(q_row_cap_i32)
            .arg_i32(q_stride_i32)
            .launch(stream)?;
        run_start += run_len;
    }
    Ok(())
}

/// Returns `true` iff `chunk` can execute as ONE single deterministic
/// `target_rejection_verify_rows_bf16` kernel launch over contiguous rows
/// `0..chunk.len()` inside a captured verify CUDA graph:
/// - `1 <= chunk.len() <= REJECTION_MAX_ROWS` (256),
/// - `chunk[i].row_idx == i` for every `i`,
/// - every row passes [`validate_rejection_verify_row`], and every stochastic
///   non-bonus row has `(q_row_idx as usize) < q_row_cap && q_stride > 0`.
#[must_use]
pub(super) fn can_launch_single_rejection_verify_run(
    chunk: &[RejectionVerifyRowSpec],
    q_row_cap: usize,
    q_stride: usize,
) -> bool {
    if chunk.is_empty() || chunk.len() > REJECTION_MAX_ROWS {
        return false;
    }
    chunk.iter().enumerate().all(|(i, req)| {
        req.row_idx == i
            && validate_rejection_verify_row(req, i).is_ok()
            && (req.temperature <= 0.0
                || req.q_row_idx == REJECTION_BONUS_Q_ROW_IDX
                || ((req.q_row_idx as usize) < q_row_cap && q_stride > 0))
    })
}

/// Launch ONE single CUDA-graph-safe `target_rejection_verify_rows_bf16` kernel
/// over contiguous rows `0..num_rows` (`1 <= num_rows <= REJECTION_MAX_ROWS`)
/// using fixed device pointers (`logits_base`, `q_cand_ids`, `q_probs`, `buf`).
///
/// Per-row metadata must already be staged in `buf` via
/// [`encode_rejection_verify_chunk`] before graph launch or capture.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_target_rejection_verify_single_run(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_base: DevicePtr,
    q_cand_ids: DevicePtr,
    q_probs: DevicePtr,
    buf: DevicePtr,
    num_rows: usize,
    vocab: u32,
    vocab_stride: usize,
    q_row_cap: usize,
    q_stride: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=REJECTION_MAX_ROWS).contains(&num_rows),
        "target_rejection_verify_rows_bf16 single-run num_rows={num_rows} outside 1..={REJECTION_MAX_ROWS}"
    );
    anyhow::ensure!(
        kernel.0 != 0 && buf.0 != 0 && logits_base.0 != 0 && q_cand_ids.0 != 0 && q_probs.0 != 0,
        "target_rejection_verify_rows_bf16 single-run requires non-null kernel, buf, q buffers, and logits"
    );
    anyhow::ensure!(vocab > 0, "target_rejection_verify_rows_bf16 vocab=0");
    anyhow::ensure!(
        vocab_stride == vocab as usize || (num_rows == 1 && vocab_stride >= vocab as usize),
        "target_rejection_verify_rows_bf16 single-run requires vocab_stride==vocab (got vocab_stride={vocab_stride}, vocab={vocab}, num_rows={num_rows})"
    );
    anyhow::ensure!(
        q_row_cap > 0 && q_stride > 0,
        "target_rejection_verify_rows_bf16 single-run invalid q geometry (q_row_cap={q_row_cap}, q_stride={q_stride})"
    );
    let vocab_i32 = i32::try_from(vocab).with_context(|| {
        format!("target_rejection_verify_rows_bf16 vocab={vocab} exceeds i32::MAX")
    })?;
    let q_row_cap_i32 = i32::try_from(q_row_cap).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_row_cap={q_row_cap} exceeds i32::MAX")
    })?;
    let q_stride_i32 = i32::try_from(q_stride).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_stride={q_stride} exceeds i32::MAX")
    })?;
    KernelLaunch::new(gpu, kernel)
        .grid([num_rows as u32, 1, 1])
        .block([REJECTION_VERIFY_BLOCK_THREADS, 1, 1])
        .arg_ptr(logits_base)
        .arg_ptr(q_cand_ids)
        .arg_ptr(q_probs)
        .arg_ptr(buf.offset(REJECTION_SEEDS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_ABS_POS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_TEMPS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_TOPKS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_TOPPS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_MINPS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_DRAFT_TOKS_OFFSET))
        .arg_ptr(buf.offset(REJECTION_Q_ROW_IDX_OFFSET))
        .arg_ptr(buf.offset(REJECTION_MASK0_OFFSET))
        .arg_ptr(buf.offset(REJECTION_MASK1_OFFSET))
        .arg_ptr(buf.offset(REJECTION_OUT_OFFSET))
        .arg_i32(num_rows as i32)
        .arg_i32(vocab_i32)
        .arg_i32(q_row_cap_i32)
        .arg_i32(q_stride_i32)
        .launch(stream)
}

/// Launch `target_rejection_verify_rows_bf16` supporting arbitrary
/// `vocab_stride >= vocab`: coalesces contiguous runs when `vocab_stride == vocab`
/// and splits into per-row launches when `vocab_stride > vocab` because the
/// CUDA kernel indexes row `r` at `logits + r * vocab`.
#[allow(clippy::too_many_arguments)]
fn launch_target_rejection_verify_runs_with_stride(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_base: DevicePtr,
    q_cand_ids: DevicePtr,
    q_probs: DevicePtr,
    buf: DevicePtr,
    chunk: &[RejectionVerifyRowSpec],
    vocab: u32,
    vocab_stride: usize,
    q_row_cap: usize,
    q_stride: usize,
    stream: u64,
) -> Result<()> {
    let vocab_usize = vocab as usize;
    anyhow::ensure!(
        vocab_stride >= vocab_usize,
        "target_rejection_verify_rows_bf16 vocab_stride={vocab_stride} < vocab={vocab}"
    );
    if vocab_stride == vocab_usize {
        return launch_target_rejection_verify_runs(
            gpu,
            kernel,
            logits_base,
            q_cand_ids,
            q_probs,
            buf,
            chunk,
            vocab,
            q_row_cap,
            q_stride,
            stream,
        );
    }
    anyhow::ensure!(vocab > 0, "target_rejection_verify_rows_bf16 vocab=0");
    let vocab_i32 = i32::try_from(vocab).with_context(|| {
        format!("target_rejection_verify_rows_bf16 vocab={vocab} exceeds i32::MAX")
    })?;
    let q_row_cap_i32 = i32::try_from(q_row_cap).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_row_cap={q_row_cap} exceeds i32::MAX")
    })?;
    let q_stride_i32 = i32::try_from(q_stride).with_context(|| {
        format!("target_rejection_verify_rows_bf16 q_stride={q_stride} exceeds i32::MAX")
    })?;
    anyhow::ensure!(
        chunk.len() <= REJECTION_MAX_ROWS,
        "target_rejection_verify_rows_bf16 chunk size {} exceeds max {REJECTION_MAX_ROWS}",
        chunk.len()
    );
    for (slot, req) in chunk.iter().enumerate() {
        req.row_idx
            .checked_mul(vocab_stride)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                anyhow::anyhow!("target_rejection_verify_rows_bf16 row offset overflow")
            })?;
        if req.temperature > 0.0 && req.q_row_idx != REJECTION_BONUS_Q_ROW_IDX {
            anyhow::ensure!(
                (req.q_row_idx as usize) < q_row_cap && q_stride > 0,
                "target_rejection_verify_rows_bf16 invalid q_row_idx={} (q_row_cap={q_row_cap}, q_stride={q_stride}) at slot {slot}",
                req.q_row_idx
            );
        }
    }
    for (slot, req) in chunk.iter().enumerate() {
        let row_byte_off = req.row_idx * vocab_stride * 2;
        KernelLaunch::new(gpu, kernel)
            .grid([1, 1, 1])
            .block([REJECTION_VERIFY_BLOCK_THREADS, 1, 1])
            .arg_ptr(logits_base.offset(row_byte_off))
            .arg_ptr(q_cand_ids)
            .arg_ptr(q_probs)
            .arg_ptr(buf.offset(REJECTION_SEEDS_OFFSET + slot * 8))
            .arg_ptr(buf.offset(REJECTION_ABS_POS_OFFSET + slot * 8))
            .arg_ptr(buf.offset(REJECTION_TEMPS_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_TOPKS_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_TOPPS_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_MINPS_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_DRAFT_TOKS_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_Q_ROW_IDX_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_MASK0_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_MASK1_OFFSET + slot * 4))
            .arg_ptr(buf.offset(REJECTION_OUT_OFFSET + slot * 4))
            .arg_i32(1)
            .arg_i32(vocab_i32)
            .arg_i32(q_row_cap_i32)
            .arg_i32(q_stride_i32)
            .launch(stream)?;
    }
    Ok(())
}

/// Validate and decode `num_rows` packed little-endian `u32` words read back
/// from `buf.offset(REJECTION_OUT_OFFSET)`.
///
/// Each word is `(accepted << 31) | token_id`, or [`HOST_POLICY_FALLBACK_TOKEN`]
/// (`0x7FFF_FFFF`, bit 31 = 0) for rows skipped via `temperature < 0.0`.
/// Any non-fallback word whose 31-bit `token_id` is `>= vocab_size` fails closed.
#[allow(dead_code)]
pub(crate) fn decode_rejection_verify_tokens(
    bytes: &[u8],
    num_rows: usize,
    vocab_size: usize,
) -> Result<Vec<PackedRejectionWord>> {
    anyhow::ensure!(
        bytes.len() == num_rows * 4,
        "target_rejection_verify_rows_bf16 D2H byte len {} != {}",
        bytes.len(),
        num_rows * 4
    );
    let mut out = Vec::with_capacity(num_rows);
    for raw_bytes in bytes.chunks_exact(4) {
        let word = PackedRejectionWord(u32::from_le_bytes([
            raw_bytes[0],
            raw_bytes[1],
            raw_bytes[2],
            raw_bytes[3],
        ]));
        anyhow::ensure!(
            word.is_host_fallback() || (word.token_id() as usize) < vocab_size,
            "target_rejection_verify_rows_bf16 returned out-of-range word 0x{:08x} (token {} >= vocab {vocab_size})",
            word.0,
            word.token_id()
        );
        out.push(word);
    }
    Ok(out)
}

/// Decode the result buffer into `(id, raw)` pairs in selection-key order.
pub(crate) fn decode_topk_result(
    bytes: &[u8],
    m: usize,
    temperature: f32,
) -> Result<Vec<(u32, f32)>> {
    anyhow::ensure!(
        bytes.len() >= 4,
        "topk_candidates_bf16 result buffer too short: {} bytes (need at least 4)",
        bytes.len()
    );
    let word = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let count = word(0) as usize;
    anyhow::ensure!(
        count <= m && count <= DEVICE_TOPK_M_MAX,
        "topk_candidates_bf16 reported {count} candidates for m={m} (max {DEVICE_TOPK_M_MAX})"
    );
    let needed = if count == 0 {
        4
    } else {
        VALS_OFFSET + count * 4
    };
    anyhow::ensure!(
        bytes.len() >= needed,
        "topk_candidates_bf16 result buffer too short: {} bytes (need at least {needed} for count={count})",
        bytes.len()
    );
    let mut out: Vec<(u64, u32, f32)> = Vec::with_capacity(count);
    for slot in 0..count {
        let id = word(IDS_OFFSET + slot * 4);
        let raw = f32::from_bits(word(VALS_OFFSET + slot * 4));
        let key = selection_key(id, raw, temperature).with_context(|| {
            format!("topk_candidates_bf16 returned a non-finite value for id {id}")
        })?;
        out.push((key, id, raw));
    }
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    anyhow::ensure!(
        out.windows(2).all(|w| w[0].1 != w[1].1),
        "topk_candidates_bf16 returned a duplicate id"
    );
    Ok(out.into_iter().map(|(_, id, raw)| (id, raw)).collect())
}

/// Launch `topk_candidates_bf16` for one BF16 row into the result buffer
/// `buf` (layout above). Shape errors fail before launch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_topk_candidates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_row: DevicePtr,
    buf: DevicePtr,
    vocab: u32,
    m: usize,
    temperature: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=DEVICE_TOPK_M_MAX).contains(&m),
        "device top-k m={m} outside 1..={DEVICE_TOPK_M_MAX}"
    );
    anyhow::ensure!(
        temperature > 0.0 && temperature.is_finite(),
        "device top-k temperature={temperature}"
    );
    anyhow::ensure!(vocab > 0, "device top-k vocab=0");
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits_row)
        .arg_ptr(buf.offset(IDS_OFFSET))
        .arg_ptr(buf.offset(VALS_OFFSET))
        .arg_ptr(buf)
        .arg_u32(vocab)
        .arg_u32(m as u32)
        .arg_f32(temperature)
        .launch(stream)
}

/// Launch the 2-stage multi-CTA sliced top-k pipeline (`topk_slice_stage1_bf16`
/// across `num_slices = ceil(vocab / 8192)` CTAs followed by single-CTA
/// `topk_slice_stage2_merge`) into the shared result buffer `buf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_sliced_topk_candidates(
    gpu: &dyn GpuBackend,
    stage1: KernelHandle,
    stage2: KernelHandle,
    logits_row: DevicePtr,
    buf: DevicePtr,
    vocab: u32,
    m: usize,
    temperature: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        use_sliced_topk(vocab as usize, m),
        "sliced device top-k requires use_sliced_topk(vocab={vocab}, m={m})"
    );
    anyhow::ensure!(
        temperature > 0.0 && temperature.is_finite(),
        "device top-k temperature={temperature}"
    );
    let num_slices = topk_slice_grid(vocab as usize);
    let slice_keys = buf.offset(SLICE_KEYS_OFFSET);
    let slice_vals = buf.offset(SLICE_VALS_OFFSET);
    let slice_counts = buf.offset(SLICE_COUNTS_OFFSET);
    KernelLaunch::new(gpu, stage1)
        .grid([num_slices, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits_row)
        .arg_ptr(slice_keys)
        .arg_ptr(slice_vals)
        .arg_ptr(slice_counts)
        .arg_u32(vocab)
        .arg_u32(m as u32)
        .arg_f32(temperature)
        .launch(stream)?;
    KernelLaunch::new(gpu, stage2)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slice_keys)
        .arg_ptr(slice_vals)
        .arg_ptr(slice_counts)
        .arg_ptr(buf.offset(IDS_OFFSET))
        .arg_ptr(buf.offset(VALS_OFFSET))
        .arg_ptr(buf)
        .arg_u32(num_slices)
        .arg_u32(m as u32)
        .launch(stream)
}

/// Dispatch 2-stage sliced GPU top-k when `(vocab, m)` fits the sliced
/// envelope and both sliced stage kernels resolved on `gpu`, falling back
/// seamlessly to single-CTA `topk_candidates_bf16` otherwise.
///
/// Returns `Ok(true)` when the 2-stage sliced pipeline ran and `Ok(false)`
/// when single-CTA `topk_candidates_bf16` ran.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_topk_candidates(
    gpu: &dyn GpuBackend,
    single_cta_kernel: KernelHandle,
    logits_row: DevicePtr,
    buf: DevicePtr,
    vocab: u32,
    m: usize,
    temperature: f32,
    stream: u64,
) -> Result<bool> {
    anyhow::ensure!(
        (1..=DEVICE_TOPK_M_MAX).contains(&m),
        "device top-k m={m} outside 1..={DEVICE_TOPK_M_MAX}"
    );
    anyhow::ensure!(
        temperature > 0.0 && temperature.is_finite(),
        "device top-k temperature={temperature}"
    );
    anyhow::ensure!(vocab > 0, "device top-k vocab=0");
    if use_sliced_topk(vocab as usize, m)
        && let Some((stage1, stage2)) = cached_sliced_topk_kernels(gpu, buf)
    {
        launch_sliced_topk_candidates(
            gpu,
            stage1,
            stage2,
            logits_row,
            buf,
            vocab,
            m,
            temperature,
            stream,
        )?;
        return Ok(true);
    }
    launch_topk_candidates(
        gpu,
        single_cta_kernel,
        logits_row,
        buf,
        vocab,
        m,
        temperature,
        stream,
    )?;
    Ok(false)
}

/// Encode `edits` as the kernel's id / kind / value arrays.
pub(crate) fn encode_f2_edits(edits: &[(u32, PreF2Edit)]) -> Result<Vec<u8>> {
    anyhow::ensure!(
        edits.len() <= F2_EDITS_MAX,
        "F2 certification takes at most {F2_EDITS_MAX} edits, got {}",
        edits.len()
    );
    let mut b = vec![0u8; F2_EDITS_BYTES];
    for (slot, &(id, edit)) in edits.iter().enumerate() {
        let (kind, val) = edit.encode();
        b[slot * 4..][..4].copy_from_slice(&id.to_le_bytes());
        b[F2_EDITS_MAX * 4 + slot * 4..][..4].copy_from_slice(&kind.to_le_bytes());
        b[F2_EDITS_MAX * 8 + slot * 4..][..4].copy_from_slice(&val.to_bits().to_le_bytes());
    }
    Ok(b)
}

/// Decode the kernel's 32-byte stats block.
pub(crate) fn decode_f2_stats(b: &[u8], n: u32) -> SumExpStats {
    let f64_at = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    SumExpStats {
        sum: f64_at(0),
        weighted: f64_at(8),
        term_err: f64_at(16),
        max: f32::from_le_bytes(b[24..28].try_into().unwrap()),
        nonfinite: u32::from_le_bytes(b[28..32].try_into().unwrap()) != 0,
        n,
    }
}

/// Launch `edited_sum_exp_bf16` for one row; edits must already be in `buf`.
pub(crate) fn launch_edited_sum_exp(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits_row: DevicePtr,
    buf: DevicePtr,
    n_edits: usize,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(n_edits <= F2_EDITS_MAX && vocab > 0, "F2 launch shape");
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits_row)
        .arg_ptr(buf.offset(F2_IDS_OFFSET))
        .arg_ptr(buf.offset(F2_KIND_OFFSET))
        .arg_ptr(buf.offset(F2_VAL_OFFSET))
        .arg_u32(n_edits as u32)
        .arg_u32(vocab)
        .arg_ptr(buf.offset(F2_OUT_OFFSET))
        .launch(stream)
}

impl TransformerModel {
    pub(super) fn topk_candidates_dispatch(
        &self,
        logits_row: DevicePtr,
        temperature: f32,
        m: usize,
    ) -> Result<Vec<(u32, f32)>> {
        anyhow::ensure!(
            self.topk_candidates_kernel.0 != 0 && self.topk_candidates_buf.0 != 0,
            "device top-k requested but ATLAS_SAMPLER_DEVICE_TOPK=1 was not set at load"
        );
        anyhow::ensure!(
            !(self.use_fp32_logits && logits_row.0 == self.logits_fp32_buf.0),
            "device top-k reads BF16 logits; this row is FP32"
        );
        let stream = self.gpu.default_stream();
        let buf = self.topk_candidates_buf;
        let _used_sliced = dispatch_topk_candidates(
            self.gpu.as_ref(),
            self.topk_candidates_kernel,
            logits_row,
            buf,
            self.config.vocab_size as u32,
            m,
            temperature,
            stream,
        )?;
        let mut host = vec![0u8; TOPK_BYTES];
        self.gpu.copy_d2h_on_stream(buf, &mut host, stream)?;
        decode_topk_result(&host[..TOPK_BYTES], m, temperature)
    }

    pub(super) fn edited_sum_exp_dispatch(
        &self,
        logits_row: DevicePtr,
        edits: &[(u32, PreF2Edit)],
    ) -> Result<SumExpStats> {
        anyhow::ensure!(
            self.f2_sum_kernel.0 != 0 && self.topk_candidates_buf.0 != 0,
            "F2 certification requested but ATLAS_SAMPLER_DEVICE_TOPK=1 was not set at load"
        );
        anyhow::ensure!(
            !(self.use_fp32_logits && logits_row.0 == self.logits_fp32_buf.0),
            "F2 certification reads BF16 logits; this row is FP32"
        );
        let stream = self.gpu.default_stream();
        let buf = self.topk_candidates_buf;
        let encoded = encode_f2_edits(edits)?;
        self.gpu.copy_h2d_group_on_stream(
            &[HostToDeviceCopy::new(&encoded, buf.offset(F2_IDS_OFFSET))],
            stream,
        )?;
        let vocab = self.config.vocab_size as u32;
        launch_edited_sum_exp(
            self.gpu.as_ref(),
            self.f2_sum_kernel,
            logits_row,
            buf,
            edits.len(),
            vocab,
            stream,
        )?;
        let mut out = [0u8; F2_OUT_BYTES];
        self.gpu
            .copy_d2h_on_stream(buf.offset(F2_OUT_OFFSET), &mut out, stream)?;
        Ok(decode_f2_stats(&out, vocab))
    }

    /// Batched GPU target Gumbel-Max verify over BF16 verify logits rows.
    ///
    /// Uses the pre-allocated 8 KiB staging buffer (`self.target_gumbel_verify_buf`)
    /// so there are zero per-step `cuMemAlloc` calls and zero intermediate heap
    /// allocations. Homogeneous verify frames (`<= 256` rows) execute in 1 fast
    /// pinned-ring H2D copy + 1 kernel launch + 1 small (`num_rows * 4` B) D2H copy.
    pub(super) fn sample_gumbel_verify_rows_batched_dispatch(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[GumbelVerifyRowSpec],
    ) -> Result<Option<Vec<u32>>> {
        if self.target_gumbel_verify_kernel.0 == 0 || self.target_gumbel_verify_buf.0 == 0 {
            return Ok(None);
        }
        if self.use_fp32_logits && logits_base.0 == self.logits_fp32_buf.0 {
            return Ok(None);
        }
        if reqs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        anyhow::ensure!(
            logits_base.0 != 0,
            "target_gumbel_verify_rows_bf16 called with null logits_base"
        );
        let vocab_usize = self.config.vocab_size;
        let vocab = u32::try_from(vocab_usize)
            .with_context(|| format!("vocab_size={vocab_usize} exceeds u32::MAX"))?;
        let stream = self.gpu.default_stream();
        let buf = self.target_gumbel_verify_buf;
        let mut all_tokens = Vec::with_capacity(reqs.len());
        let mut out_bytes = [0u8; GUMBEL_VERIFY_MAX_ROWS * 4];

        for chunk in reqs.chunks(GUMBEL_VERIFY_MAX_ROWS) {
            let stage = encode_gumbel_verify_chunk(chunk)?;
            self.copy_h2d_group_fast(&[HostToDeviceCopy::new(&stage, buf)], stream)?;
            launch_target_gumbel_verify_runs(
                self.gpu.as_ref(),
                self.target_gumbel_verify_kernel,
                logits_base,
                buf,
                chunk,
                vocab,
                vocab_stride,
                stream,
            )?;
            let d2h_slice = &mut out_bytes[..chunk.len() * 4];
            self.gpu
                .copy_d2h_on_stream(buf.offset(GUMBEL_OUT_OFFSET), d2h_slice, stream)?;
            let decoded = decode_gumbel_verify_tokens(d2h_slice, chunk.len(), vocab_usize)?;
            for (slot, (req, &tok)) in chunk.iter().zip(decoded.iter()).enumerate() {
                anyhow::ensure!(
                    req.temperature < 0.0 || tok != GUMBEL_VERIFY_SKIP_SENTINEL,
                    "target_gumbel_verify_rows_bf16 returned skip sentinel for active row slot {slot}"
                );
            }
            all_tokens.extend(decoded);
        }
        Ok(Some(all_tokens))
    }

    /// Batched GPU maximal-coupling rejection verify over BF16 verify logits rows.
    ///
    /// Uses the pre-allocated 13 KiB staging buffer (`self.target_rejection_verify_buf`)
    /// and 80 KiB shared draft-distribution buffer (`self.target_rejection_q_buf`)
    /// with zero per-step device allocations.
    pub(super) fn sample_rejection_verify_rows_batched_dispatch(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[RejectionVerifyRowSpec],
    ) -> Result<Option<Vec<PackedRejectionWord>>> {
        if self.target_rejection_verify_kernel.0 == 0
            || self.target_rejection_verify_buf.is_null()
            || self.target_rejection_q_buf.is_null()
        {
            return Ok(None);
        }
        if self.use_fp32_logits && logits_base.0 == self.logits_fp32_buf.0 {
            return Ok(None);
        }
        if reqs.is_empty() {
            return Ok(Some(Vec::new()));
        }
        anyhow::ensure!(
            !logits_base.is_null(),
            "target_rejection_verify_rows_bf16 called with null logits_base"
        );
        let vocab_usize = self.config.vocab_size;
        let vocab = u32::try_from(vocab_usize)
            .with_context(|| format!("vocab_size={vocab_usize} exceeds u32::MAX"))?;
        let stream = self.gpu.default_stream();
        let buf = self.target_rejection_verify_buf;
        let q_ids = self.target_rejection_q_buf;
        let q_probs = self.target_rejection_q_buf.offset(REJECTION_Q_PROBS_OFFSET);
        let mut all_words = Vec::with_capacity(reqs.len());
        let mut out_bytes = [0u8; REJECTION_MAX_ROWS * 4];

        for chunk in reqs.chunks(REJECTION_MAX_ROWS) {
            let stage = encode_rejection_verify_chunk(chunk)?;
            self.copy_h2d_group_fast(&[HostToDeviceCopy::new(&stage, buf)], stream)?;
            launch_target_rejection_verify_runs_with_stride(
                self.gpu.as_ref(),
                self.target_rejection_verify_kernel,
                logits_base,
                q_ids,
                q_probs,
                buf,
                chunk,
                vocab,
                vocab_stride,
                REJECTION_Q_ROW_CAP,
                REJECTION_Q_STRIDE,
                stream,
            )?;
            let d2h_slice = &mut out_bytes[..chunk.len() * 4];
            self.gpu
                .copy_d2h_on_stream(buf.offset(REJECTION_OUT_OFFSET), d2h_slice, stream)?;
            let decoded = decode_rejection_verify_tokens(d2h_slice, chunk.len(), vocab_usize)?;
            for (slot, (req, &word)) in chunk.iter().zip(decoded.iter()).enumerate() {
                anyhow::ensure!(
                    req.temperature < 0.0 || !word.is_host_fallback(),
                    "target_rejection_verify_rows_bf16 returned host fallback sentinel for active row slot {slot}"
                );
            }
            all_words.extend(decoded);
        }
        Ok(Some(all_words))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(pairs: &[(u32, f32)], count: u32) -> Vec<u8> {
        let mut b = vec![0u8; TOPK_BYTES];
        b[..4].copy_from_slice(&count.to_le_bytes());
        for (slot, (id, v)) in pairs.iter().enumerate() {
            b[IDS_OFFSET + slot * 4..][..4].copy_from_slice(&id.to_le_bytes());
            b[VALS_OFFSET + slot * 4..][..4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        b
    }

    #[test]
    fn decode_orders_unordered_kernel_output_by_value_then_lower_id() {
        let b = encode(&[(9, 1.0), (4, 2.0), (3, 1.0)], 3);
        assert_eq!(
            decode_topk_result(&b, 4, 1.0).unwrap(),
            vec![(4, 2.0), (3, 1.0), (9, 1.0)]
        );
    }

    #[test]
    fn decode_refuses_overflow_duplicates_and_non_finite() {
        assert!(decode_topk_result(&encode(&[(1, 1.0), (2, 1.0)], 2), 1, 1.0).is_err());
        assert!(decode_topk_result(&encode(&[(1, 1.0), (1, 1.0)], 2), 4, 1.0).is_err());
        assert!(decode_topk_result(&encode(&[(1, f32::NAN)], 1), 4, 1.0).is_err());
    }

    #[test]
    fn launch_checks_shape_before_launching_one_block() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        let gpu = MockGpuBackend::new();
        let go = |m: usize, t: f32| {
            launch_topk_candidates(
                &gpu,
                KernelHandle(7),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                248_320,
                m,
                t,
                0,
            )
        };
        assert!(go(0, 1.0).is_err());
        assert!(go(DEVICE_TOPK_M_MAX + 1, 1.0).is_err());
        assert!(go(20, 0.0).is_err());
        assert!(go(20, f32::NAN).is_err());
        assert_eq!(gpu.launch_count(), 0);
        go(27, 0.7).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn f2_edits_encode_in_the_kernel_layout_and_refuse_overflow() {
        let b = encode_f2_edits(&[(7, PreF2Edit::Sub(10.0)), (9, PreF2Edit::NegInf)]).unwrap();
        let word = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!((word(0), word(4)), (7, 9));
        assert_eq!((word(F2_EDITS_MAX * 4), word(F2_EDITS_MAX * 4 + 4)), (0, 1));
        assert_eq!(f32::from_bits(word(F2_EDITS_MAX * 8)), 10.0);
        let too_many = vec![(0u32, PreF2Edit::NegInf); F2_EDITS_MAX + 1];
        assert!(encode_f2_edits(&too_many).is_err());
    }

    #[test]
    fn f2_stats_decode_round_trips_the_kernel_block() {
        let mut b = [0u8; F2_OUT_BYTES];
        b[0..8].copy_from_slice(&1.25f64.to_le_bytes());
        b[8..16].copy_from_slice(&300.0f64.to_le_bytes());
        b[16..24].copy_from_slice(&9.0f64.to_le_bytes());
        b[24..28].copy_from_slice(&4.5f32.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        let s = decode_f2_stats(&b, 1000);
        assert_eq!(
            (s.sum, s.weighted, s.term_err, s.max, s.nonfinite, s.n),
            (1.25, 300.0, 9.0, 4.5, true, 1000)
        );
    }

    #[test]
    fn f2_launch_is_one_block_and_checks_shape() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        let gpu = MockGpuBackend::new();
        let go = |e: usize| {
            launch_edited_sum_exp(
                &gpu,
                KernelHandle(3),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                e,
                248_320,
                0,
            )
        };
        assert!(go(F2_EDITS_MAX + 1).is_err());
        assert_eq!(gpu.launch_count(), 0);
        go(5).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn route_off_loads_nothing() {
        if device_topk_enabled() {
            return;
        }
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let (k, f2, b) = load_device_topk(&gpu).unwrap();
        assert_eq!((k.0, f2.0, b.0), (0, 0, 0));
    }

    struct SelectiveTopkGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        has_stage1: bool,
        has_stage2: bool,
        kernel_lookups: Mutex<usize>,
        recorded: Mutex<Vec<(u64, [u32; 3], [u32; 3])>>,
    }

    impl SelectiveTopkGpu {
        fn new(has_stage1: bool, has_stage2: bool) -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                has_stage1,
                has_stage2,
                kernel_lookups: Mutex::new(0),
                recorded: Mutex::new(Vec::new()),
            }
        }
    }

    impl GpuBackend for SelectiveTopkGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.recorded.lock().push((func.0, grid, block));
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            0
        }
        #[track_caller]
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            *self.kernel_lookups.lock() += 1;
            match (module, func_name) {
                ("sampler_topk_candidates", "topk_slice_stage1_bf16") if self.has_stage1 => {
                    Ok(KernelHandle(0x51A1))
                }
                ("sampler_topk_candidates", "topk_slice_stage2_merge") if self.has_stage2 => {
                    Ok(KernelHandle(0x51A2))
                }
                ("sampler_topk_candidates", "topk_candidates_bf16") => Ok(KernelHandle(0x1C7A)),
                _ => anyhow::bail!("missing mock kernel {module}::{func_name}"),
            }
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    #[test]
    fn sliced_topk_dispatches_two_stages_with_31_ctas_and_caches_kernel_handles() {
        static NEXT_TEST_BUF: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0xA000_0000);
        let gpu = SelectiveTopkGpu::new(true, true);
        let buf =
            DevicePtr(NEXT_TEST_BUF.fetch_add(0x10_0000, std::sync::atomic::Ordering::Relaxed));
        const {
            assert!(DEVICE_TOPK_BUF_BYTES >= SLICE_COUNTS_OFFSET + DEVICE_TOPK_SLICE_MAX_CTAS * 4);
        }

        let used_first = dispatch_topk_candidates(
            &gpu,
            KernelHandle(0x1C7A),
            DevicePtr(0x1000),
            buf,
            248_320,
            24,
            0.7,
            0,
        )
        .unwrap();
        assert!(used_first);
        let lookups_after_first = *gpu.kernel_lookups.lock();
        assert_eq!(lookups_after_first, 2);

        let used_second = dispatch_topk_candidates(
            &gpu,
            KernelHandle(0x1C7A),
            DevicePtr(0x1000),
            buf,
            248_320,
            64,
            1.0,
            0,
        )
        .unwrap();
        assert!(used_second);
        assert_eq!(
            *gpu.kernel_lookups.lock(),
            lookups_after_first,
            "second dispatch must reuse cached stage1/stage2 handles"
        );

        let launches = gpu.recorded.lock().clone();
        assert_eq!(
            launches,
            vec![
                (0x51A1, [31, 1, 1], [256, 1, 1]),
                (0x51A2, [1, 1, 1], [256, 1, 1]),
                (0x51A1, [31, 1, 1], [256, 1, 1]),
                (0x51A2, [1, 1, 1], [256, 1, 1]),
            ]
        );
    }

    #[test]
    fn sliced_topk_falls_back_to_single_cta_for_wide_m_small_vocab_or_missing_kernels() {
        static NEXT_FALLBACK_BUF: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0xB000_0000);
        let next_buf = || {
            DevicePtr(NEXT_FALLBACK_BUF.fetch_add(0x10_0000, std::sync::atomic::Ordering::Relaxed))
        };

        // Case 1: kernels exist, but m > 64 or vocab <= 8192 -> single-CTA.
        let gpu = SelectiveTopkGpu::new(true, true);
        let buf = next_buf();
        assert!(
            !dispatch_topk_candidates(
                &gpu,
                KernelHandle(0x1C7A),
                DevicePtr(0x1000),
                buf,
                248_320,
                65,
                0.8,
                0,
            )
            .unwrap()
        );
        assert!(
            !dispatch_topk_candidates(
                &gpu,
                KernelHandle(0x1C7A),
                DevicePtr(0x1000),
                buf,
                8192,
                20,
                0.8,
                0,
            )
            .unwrap()
        );
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![
                (0x1C7A, [1, 1, 1], [1024, 1, 1]),
                (0x1C7A, [1, 1, 1], [1024, 1, 1]),
            ]
        );

        // Case 2: (vocab, m) is slice-eligible, but stage1 or stage2 is absent on backend.
        for (has_s1, has_s2) in [(false, false), (true, false), (false, true)] {
            let missing_gpu = SelectiveTopkGpu::new(has_s1, has_s2);
            let missing_buf = next_buf();
            let used = dispatch_topk_candidates(
                &missing_gpu,
                KernelHandle(0x1C7A),
                DevicePtr(0x1000),
                missing_buf,
                248_320,
                24,
                0.7,
                0,
            )
            .unwrap();
            assert!(!used, "has_s1={has_s1} has_s2={has_s2}");
            assert_eq!(
                missing_gpu.recorded.lock().clone(),
                vec![(0x1C7A, [1, 1, 1], [1024, 1, 1])]
            );
        }
    }

    struct GumbelVerifyMockGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        has_kernel: bool,
        alloc_sizes: Mutex<Vec<usize>>,
        recorded: Mutex<Vec<(u64, [u32; 3], [u32; 3])>>,
    }

    impl GumbelVerifyMockGpu {
        fn new(has_kernel: bool) -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                has_kernel,
                alloc_sizes: Mutex::new(Vec::new()),
                recorded: Mutex::new(Vec::new()),
            }
        }
    }

    impl GpuBackend for GumbelVerifyMockGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.alloc_sizes.lock().push(bytes);
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.recorded.lock().push((func.0, grid, block));
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            match (module, func_name) {
                ("sampler_topk_candidates", "target_gumbel_verify_rows_bf16")
                    if self.has_kernel =>
                {
                    Ok(KernelHandle(0x600B))
                }
                ("sampler_topk_candidates", "target_rejection_verify_rows_bf16")
                    if self.has_kernel =>
                {
                    Ok(KernelHandle(0x600C))
                }
                _ => anyhow::bail!("missing mock kernel {module}::{func_name}"),
            }
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    #[test]
    fn load_batched_gumbel_verify_is_optional_and_allocates_small_reusable_stage() {
        assert_eq!(GUMBEL_VERIFY_STAGE_BYTES, 8192);

        let absent = GumbelVerifyMockGpu::new(false);
        let (k, b) = load_batched_gumbel_verify(&absent).unwrap();
        assert_eq!((k.0, b.0), (0, 0));
        assert!(absent.alloc_sizes.lock().is_empty());

        let present = GumbelVerifyMockGpu::new(true);
        let (k, b) = load_batched_gumbel_verify(&present).unwrap();
        assert_eq!(k.0, 0x600B);
        assert_ne!(b.0, 0);
        assert_eq!(*present.alloc_sizes.lock(), vec![GUMBEL_VERIFY_STAGE_BYTES]);
    }

    #[test]
    fn encode_gumbel_verify_chunk_packs_offsets_and_validates_inputs() {
        let reqs = [
            GumbelVerifyRowSpec {
                row_idx: 0,
                seed: 0x1122_3344_5566_7788,
                abs_pos: 99,
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
            },
            GumbelVerifyRowSpec {
                row_idx: 1,
                seed: 42,
                abs_pos: 100,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
            },
            GumbelVerifyRowSpec {
                row_idx: 2,
                seed: 0,
                abs_pos: 101,
                temperature: 0.0,
                top_k: 0,
                top_p: 0.0,
            },
            GumbelVerifyRowSpec {
                row_idx: 3,
                seed: 0,
                abs_pos: 102,
                temperature: -1.0,
                top_k: (DEVICE_TOPK_K_MAX + 1) as u32,
                top_p: 0.0,
            },
        ];
        let stage = encode_gumbel_verify_chunk(&reqs).unwrap();
        assert_eq!(
            u64::from_le_bytes(stage[GUMBEL_SEEDS_OFFSET..][..8].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        assert_eq!(
            u64::from_le_bytes(
                stage[GUMBEL_POSITIONS_OFFSET + 8..][..8]
                    .try_into()
                    .unwrap()
            ),
            100
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[GUMBEL_TEMPS_OFFSET..][..4].try_into().unwrap()
            )),
            0.7
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[GUMBEL_TEMPS_OFFSET + 8..][..4].try_into().unwrap()
            )),
            0.0
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[GUMBEL_TEMPS_OFFSET + 12..][..4].try_into().unwrap()
            )),
            -1.0
        );
        assert_eq!(
            u32::from_le_bytes(stage[GUMBEL_TOPKS_OFFSET..][..4].try_into().unwrap()),
            20
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[GUMBEL_TOPPS_OFFSET..][..4].try_into().unwrap()
            )),
            0.95
        );

        let bad_cases = [
            GumbelVerifyRowSpec {
                temperature: f32::NAN,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                temperature: f32::INFINITY,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                temperature: f32::NEG_INFINITY,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                top_p: 0.0,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                top_p: -0.5,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                top_k: 0,
                top_p: 0.9,
                ..reqs[0]
            },
            GumbelVerifyRowSpec {
                top_k: (DEVICE_TOPK_K_MAX + 1) as u32,
                ..reqs[0]
            },
        ];
        for bad in bad_cases {
            assert!(encode_gumbel_verify_chunk(&[bad]).is_err(), "{bad:?}");
        }
        let oversized = vec![reqs[0]; GUMBEL_VERIFY_MAX_ROWS + 1];
        assert!(encode_gumbel_verify_chunk(&oversized).is_err());
    }

    #[test]
    fn launch_target_gumbel_verify_uses_one_launch_for_contiguous_and_splits_gaps() {
        let gpu = GumbelVerifyMockGpu::new(true);
        let mk = |row_idx: usize| GumbelVerifyRowSpec {
            row_idx,
            seed: 7,
            abs_pos: 10 + row_idx as u64,
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
        };

        // Invalid shape fails before any launch.
        assert!(
            launch_target_gumbel_verify_runs(
                &gpu,
                KernelHandle(0x600B),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                &[mk(0)],
                248_320,
                248_000,
                0,
            )
            .is_err()
        );
        assert!(gpu.recorded.lock().is_empty());

        // Homogeneous C=16, K=5 frame (80 contiguous rows) -> 1 launch [80, 1, 1] x [512, 1, 1].
        let frame80: Vec<GumbelVerifyRowSpec> = (0..80).map(mk).collect();
        launch_target_gumbel_verify_runs(
            &gpu,
            KernelHandle(0x600B),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            &frame80,
            248_320,
            248_320,
            0,
        )
        .unwrap();
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![(0x600B, [80, 1, 1], [512, 1, 1])]
        );
        gpu.recorded.lock().clear();

        // Mixed frame with gaps -> one launch per maximal contiguous run.
        let mixed: Vec<GumbelVerifyRowSpec> = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 25, 26, 27]
            .into_iter()
            .map(mk)
            .collect();
        launch_target_gumbel_verify_runs(
            &gpu,
            KernelHandle(0x600B),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            &mixed,
            248_320,
            248_320,
            0,
        )
        .unwrap();
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![
                (0x600B, [5, 1, 1], [512, 1, 1]),
                (0x600B, [5, 1, 1], [512, 1, 1]),
                (0x600B, [3, 1, 1], [512, 1, 1]),
            ]
        );
    }

    #[test]
    fn can_launch_single_gumbel_verify_run_validates_bounds_order_and_params() {
        let mk = |row_idx: usize, temperature: f32, top_k: u32, top_p: f32| GumbelVerifyRowSpec {
            row_idx,
            seed: 0xABCD_EF01 + row_idx as u64,
            abs_pos: 100 + row_idx as u64,
            temperature,
            top_k,
            top_p,
        };

        // Empty and >256 chunks are refused.
        assert!(!can_launch_single_gumbel_verify_run(&[]));
        let max_valid: Vec<GumbelVerifyRowSpec> = (0..GUMBEL_VERIFY_MAX_ROWS)
            .map(|i| mk(i, 1.0, 20, 0.95))
            .collect();
        assert!(can_launch_single_gumbel_verify_run(&max_valid));
        let mut oversized = max_valid;
        oversized.push(mk(GUMBEL_VERIFY_MAX_ROWS, 1.0, 20, 0.95));
        assert!(!can_launch_single_gumbel_verify_run(&oversized));

        // Homogeneous contiguous 0..k succeeds.
        let homogeneous: Vec<GumbelVerifyRowSpec> = (0..5).map(|i| mk(i, 0.8, 20, 0.95)).collect();
        assert!(can_launch_single_gumbel_verify_run(&homogeneous));

        // Mixed 3-way per-row parameters (`> 0` stochastic, `== 0` greedy, `< 0` skip)
        // also succeed in one launch because parameters are read from device memory.
        let mixed_valid = [
            mk(0, 1.0, 20, 0.95),
            mk(1, 0.7, 0, 1.0),
            mk(2, 0.6, 40, 0.9),
            mk(3, 0.0, 0, 0.0),
            mk(4, -1.0, (DEVICE_TOPK_K_MAX + 1) as u32, 0.0),
        ];
        assert!(can_launch_single_gumbel_verify_run(&mixed_valid));

        // Out-of-order, gapped, or non-zero-based row indices are refused.
        assert!(!can_launch_single_gumbel_verify_run(&[
            mk(1, 1.0, 20, 0.95),
            mk(0, 1.0, 20, 0.95),
        ]));
        assert!(!can_launch_single_gumbel_verify_run(&[
            mk(0, 1.0, 20, 0.95),
            mk(2, 1.0, 20, 0.95),
        ]));
        assert!(!can_launch_single_gumbel_verify_run(&[
            mk(1, 1.0, 20, 0.95),
            mk(2, 1.0, 20, 0.95),
        ]));

        // Invalid per-row sampling parameters are refused cleanly before H2D.
        assert!(!can_launch_single_gumbel_verify_run(&[mk(
            0,
            f32::NAN,
            20,
            0.95
        )]));
        assert!(!can_launch_single_gumbel_verify_run(&[mk(
            0,
            f32::INFINITY,
            20,
            0.95
        )]));
        assert!(!can_launch_single_gumbel_verify_run(&[mk(
            0,
            f32::NEG_INFINITY,
            0,
            1.0
        )]));
        assert!(!can_launch_single_gumbel_verify_run(&[mk(0, 1.0, 20, 0.0)]));
        assert!(!can_launch_single_gumbel_verify_run(&[mk(0, 1.0, 0, 0.95)]));
        assert!(!can_launch_single_gumbel_verify_run(&[mk(
            0,
            1.0,
            (DEVICE_TOPK_K_MAX + 1) as u32,
            0.95
        )]));
    }

    #[test]
    fn single_run_gumbel_launch_and_decode_validate_shapes_and_bounds() {
        let gpu = GumbelVerifyMockGpu::new(true);
        assert!(
            launch_target_gumbel_verify_single_run(
                &gpu,
                KernelHandle(0x600B),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                0,
                248_320,
                248_320,
                0,
            )
            .is_err()
        );
        assert!(
            launch_target_gumbel_verify_single_run(
                &gpu,
                KernelHandle(0x600B),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                GUMBEL_VERIFY_MAX_ROWS + 1,
                248_320,
                248_320,
                0,
            )
            .is_err()
        );
        assert!(
            launch_target_gumbel_verify_single_run(
                &gpu,
                KernelHandle(0),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                5,
                248_320,
                248_320,
                0,
            )
            .is_err()
        );
        assert!(gpu.recorded.lock().is_empty());

        launch_target_gumbel_verify_single_run(
            &gpu,
            KernelHandle(0x600B),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            80,
            248_320,
            248_320,
            0,
        )
        .unwrap();
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![(0x600B, [80, 1, 1], [512, 1, 1])]
        );

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&42u32.to_le_bytes());
        bytes.extend_from_slice(&248_319u32.to_le_bytes());
        bytes.extend_from_slice(&GUMBEL_VERIFY_SKIP_SENTINEL.to_le_bytes());
        assert_eq!(
            decode_gumbel_verify_tokens(&bytes, 3, 248_320).unwrap(),
            vec![42, 248_319, GUMBEL_VERIFY_SKIP_SENTINEL]
        );
        assert!(decode_gumbel_verify_tokens(&bytes, 2, 248_320).is_err());
        let bad = 248_320u32.to_le_bytes();
        assert!(decode_gumbel_verify_tokens(&bad, 1, 248_320).is_err());
    }

    #[test]
    fn packed_rejection_word_decodes_accept_reject_and_host_fallback() {
        let accepted = PackedRejectionWord((1u32 << 31) | 42);
        assert!(accepted.is_accepted());
        assert_eq!(accepted.token_id(), 42);
        assert!(!accepted.is_host_fallback());

        let rejected = PackedRejectionWord(12_345);
        assert!(!rejected.is_accepted());
        assert_eq!(rejected.token_id(), 12_345);
        assert!(!rejected.is_host_fallback());

        let fallback = PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN);
        assert!(!fallback.is_accepted());
        assert_eq!(fallback.token_id(), HOST_POLICY_FALLBACK_TOKEN);
        assert!(fallback.is_host_fallback());

        // Corrupt word with bit 31 = 1 and fallback payload is NOT a valid host fallback.
        let corrupt = PackedRejectionWord(u32::MAX);
        assert!(corrupt.is_accepted());
        assert!(!corrupt.is_host_fallback());
    }

    #[test]
    fn encode_rejection_verify_chunk_packs_layout_and_validates_inputs() {
        assert_eq!(REJECTION_MAX_ROWS, 256);
        assert_eq!(REJECTION_Q_STRIDE, 20);
        assert_eq!(REJECTION_Q_ROW_CAP, 512);
        assert_eq!(REJECTION_H2D_BYTES, 12_288);
        assert_eq!(REJECTION_VERIFY_STAGE_BYTES, 13_312);

        let reqs = [
            RejectionVerifyRowSpec {
                row_idx: 0,
                seed: 0xDEAD_BEEF_CAFE_BABE,
                abs_pos: 512,
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.05,
                draft_token: 777,
                q_row_idx: 3,
                mask_tok0: 151_668,
                mask_tok1: REJECTION_NO_MASK_TOKEN,
            },
            RejectionVerifyRowSpec {
                row_idx: 1,
                seed: 1,
                abs_pos: 513,
                temperature: 0.0,
                top_k: 0,
                top_p: 0.0,
                min_p: -1.0,
                draft_token: 888,
                q_row_idx: REJECTION_BONUS_Q_ROW_IDX,
                mask_tok0: REJECTION_NO_MASK_TOKEN,
                mask_tok1: REJECTION_NO_MASK_TOKEN,
            },
            RejectionVerifyRowSpec {
                row_idx: 2,
                seed: 2,
                abs_pos: 514,
                temperature: -1.0,
                top_k: 0,
                top_p: 0.0,
                min_p: 0.0,
                draft_token: 0,
                q_row_idx: REJECTION_BONUS_Q_ROW_IDX,
                mask_tok0: REJECTION_NO_MASK_TOKEN,
                mask_tok1: REJECTION_NO_MASK_TOKEN,
            },
        ];
        let stage = encode_rejection_verify_chunk(&reqs).unwrap();
        assert_eq!(
            u64::from_le_bytes(stage[REJECTION_SEEDS_OFFSET..][..8].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_BABE
        );
        assert_eq!(
            u64::from_le_bytes(stage[REJECTION_ABS_POS_OFFSET..][..8].try_into().unwrap()),
            512
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[REJECTION_TEMPS_OFFSET..][..4].try_into().unwrap()
            )),
            0.7
        );
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(
                stage[REJECTION_MINPS_OFFSET..][..4].try_into().unwrap()
            )),
            0.05
        );
        assert_eq!(
            u32::from_le_bytes(
                stage[REJECTION_DRAFT_TOKS_OFFSET..][..4]
                    .try_into()
                    .unwrap()
            ),
            777
        );
        assert_eq!(
            u32::from_le_bytes(stage[REJECTION_Q_ROW_IDX_OFFSET..][..4].try_into().unwrap()),
            3
        );
        assert_eq!(
            u32::from_le_bytes(stage[REJECTION_MASK0_OFFSET..][..4].try_into().unwrap()),
            151_668
        );
        assert_eq!(
            u32::from_le_bytes(stage[REJECTION_MASK1_OFFSET..][..4].try_into().unwrap()),
            REJECTION_NO_MASK_TOKEN
        );

        let bad_cases = [
            RejectionVerifyRowSpec {
                temperature: f32::NAN,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                temperature: f32::INFINITY,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                temperature: f32::NEG_INFINITY,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                top_p: 0.0,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                top_p: f32::NAN,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                min_p: -0.01,
                ..reqs[0]
            },
            RejectionVerifyRowSpec {
                min_p: f32::NAN,
                ..reqs[0]
            },
        ];
        for bad in bad_cases {
            assert!(encode_rejection_verify_chunk(&[bad]).is_err(), "{bad:?}");
        }
        let oversized = vec![reqs[0]; REJECTION_MAX_ROWS + 1];
        assert!(encode_rejection_verify_chunk(&oversized).is_err());
    }

    #[test]
    fn launch_and_decode_rejection_verify_validate_bounds_and_packed_words() {
        let gpu = GumbelVerifyMockGpu::new(true);
        let mk = |row_idx: usize, q_row_idx: u32| RejectionVerifyRowSpec {
            row_idx,
            seed: 9,
            abs_pos: 20 + row_idx as u64,
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            draft_token: 42,
            q_row_idx,
            mask_tok0: REJECTION_NO_MASK_TOKEN,
            mask_tok1: REJECTION_NO_MASK_TOKEN,
        };

        // Invalid q_row_idx fails before any launch.
        assert!(
            launch_target_rejection_verify_runs(
                &gpu,
                KernelHandle(0x600C),
                DevicePtr(0x1000),
                DevicePtr(0x2000),
                DevicePtr(0x3000),
                DevicePtr(0x4000),
                &[mk(0, REJECTION_Q_ROW_CAP as u32)],
                248_320,
                REJECTION_Q_ROW_CAP,
                REJECTION_Q_STRIDE,
                0,
            )
            .is_err()
        );
        assert!(gpu.recorded.lock().is_empty());

        // Contiguous 80-row frame -> 1 launch [80, 1, 1] x [512, 1, 1].
        let frame80: Vec<RejectionVerifyRowSpec> = (0..80)
            .map(|r| {
                mk(
                    r,
                    if r % 5 == 4 {
                        REJECTION_BONUS_Q_ROW_IDX
                    } else {
                        r as u32
                    },
                )
            })
            .collect();
        launch_target_rejection_verify_runs(
            &gpu,
            KernelHandle(0x600C),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            DevicePtr(0x4000),
            &frame80,
            248_320,
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE,
            0,
        )
        .unwrap();
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![(0x600C, [80, 1, 1], [512, 1, 1])]
        );
        gpu.recorded.lock().clear();

        // Mixed frame with gaps -> one launch per maximal contiguous run.
        let mixed: Vec<RejectionVerifyRowSpec> = [0, 1, 2, 10, 11]
            .into_iter()
            .map(|r| mk(r, r as u32))
            .collect();
        launch_target_rejection_verify_runs(
            &gpu,
            KernelHandle(0x600C),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            DevicePtr(0x3000),
            DevicePtr(0x4000),
            &mixed,
            248_320,
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE,
            0,
        )
        .unwrap();
        assert_eq!(
            gpu.recorded.lock().clone(),
            vec![
                (0x600C, [3, 1, 1], [512, 1, 1]),
                (0x600C, [2, 1, 1], [512, 1, 1]),
            ]
        );

        // Decode accepted, rejected, and host-fallback words; reject out-of-range vocab.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&((1u32 << 31) | 42).to_le_bytes());
        bytes.extend_from_slice(&248_319u32.to_le_bytes());
        bytes.extend_from_slice(&HOST_POLICY_FALLBACK_TOKEN.to_le_bytes());
        let decoded = decode_rejection_verify_tokens(&bytes, 3, 248_320).unwrap();
        assert_eq!(
            decoded,
            vec![
                PackedRejectionWord((1u32 << 31) | 42),
                PackedRejectionWord(248_319),
                PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
            ]
        );
        assert!(decoded[0].is_accepted() && decoded[0].token_id() == 42);
        assert!(!decoded[1].is_accepted() && decoded[1].token_id() == 248_319);
        assert!(decoded[2].is_host_fallback() && !decoded[2].is_accepted());

        // Byte-length mismatch fails.
        assert!(decode_rejection_verify_tokens(&bytes, 2, 248_320).is_err());

        // Out-of-range rejected, out-of-range accepted, and corrupt 0xFFFF_FFFF all fail closed.
        assert!(decode_rejection_verify_tokens(&248_320u32.to_le_bytes(), 1, 248_320).is_err());
        assert!(
            decode_rejection_verify_tokens(&((1u32 << 31) | 248_320).to_le_bytes(), 1, 248_320)
                .is_err()
        );
        assert!(decode_rejection_verify_tokens(&u32::MAX.to_le_bytes(), 1, 248_320).is_err());
    }

    #[test]
    fn load_and_single_run_rejection_verify_validate_allocations_and_bounds() {
        assert_eq!(REJECTION_VERIFY_STAGE_BYTES, 13_312);
        assert_eq!(REJECTION_Q_PROBS_OFFSET, 40_960);
        assert_eq!(REJECTION_Q_BUF_BYTES, 81_920);

        let absent = GumbelVerifyMockGpu::new(false);
        let (k, stage, q) = load_batched_rejection_verify(&absent).unwrap();
        assert_eq!((k.0, stage.0, q.0), (0, 0, 0));
        assert!(absent.alloc_sizes.lock().is_empty());

        let present = GumbelVerifyMockGpu::new(true);
        let (k, stage, q) = load_batched_rejection_verify(&present).unwrap();
        assert_eq!(k.0, 0x600C);
        assert!(!stage.is_null());
        assert!(!q.is_null());
        assert_eq!(
            *present.alloc_sizes.lock(),
            vec![REJECTION_VERIFY_STAGE_BYTES, REJECTION_Q_BUF_BYTES]
        );

        let mk = |row_idx: usize, temperature: f32, q_row_idx: u32| RejectionVerifyRowSpec {
            row_idx,
            seed: 7,
            abs_pos: 10 + row_idx as u64,
            temperature,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            draft_token: 42,
            q_row_idx,
            mask_tok0: REJECTION_NO_MASK_TOKEN,
            mask_tok1: REJECTION_NO_MASK_TOKEN,
        };
        assert!(!can_launch_single_rejection_verify_run(
            &[],
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE
        ));
        assert!(can_launch_single_rejection_verify_run(
            &[
                mk(0, 0.8, 0),
                mk(1, 0.0, REJECTION_BONUS_Q_ROW_IDX),
                mk(2, -1.0, REJECTION_Q_ROW_CAP as u32),
            ],
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE
        ));
        assert!(!can_launch_single_rejection_verify_run(
            &[mk(0, 0.8, REJECTION_Q_ROW_CAP as u32)],
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE
        ));
        assert!(!can_launch_single_rejection_verify_run(
            &[mk(1, 0.8, 0)],
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE
        ));

        assert!(
            launch_target_rejection_verify_single_run(
                &present,
                KernelHandle(0x600C),
                DevicePtr(0x1000),
                q,
                q.offset(REJECTION_Q_PROBS_OFFSET),
                stage,
                2,
                248_320,
                248_328,
                REJECTION_Q_ROW_CAP,
                REJECTION_Q_STRIDE,
                0,
            )
            .is_err()
        );
        assert!(present.recorded.lock().is_empty());

        launch_target_rejection_verify_single_run(
            &present,
            KernelHandle(0x600C),
            DevicePtr(0x1000),
            q,
            q.offset(REJECTION_Q_PROBS_OFFSET),
            stage,
            5,
            248_320,
            248_320,
            REJECTION_Q_ROW_CAP,
            REJECTION_Q_STRIDE,
            0,
        )
        .unwrap();
        assert_eq!(
            present.recorded.lock().clone(),
            vec![(0x600C, [5, 1, 1], [512, 1, 1])]
        );
    }
}
