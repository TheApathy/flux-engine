// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const SELECTOR_RANK: u32 = 256;
const SELECTOR_TOP_K: u32 = 16;
const PHYSICAL_VOCAB: u32 = 154_880;
const LOGICAL_VOCAB: u32 = PHYSICAL_VOCAB;
const MASK_TOKEN_ID: u32 = 154_856;
const MAX_BLOCK_TOKENS: u32 = 8;
const MAX_GRID_X: u32 = i32::MAX as u32;
const THREADS: u32 = 32;
/// CTA width for Stage-1 BF16 context gather and fused RoPE+KV-ring scatter.
const STAGE1_BLOCK_THREADS: u32 = 256;
/// Maximum candidate shortlist width for `dflash2_selector_walk_probabilistic_batched`.
const PROBABILISTIC_SELECTOR_MAX_SHORTLIST: usize = 32;
/// Single-warp CTA width for `dflash2_selector_walk_probabilistic_batched`.
const PROBABILISTIC_SELECTOR_WARP_THREADS: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2SelectorPlan {
    pub batch: u32,
    pub tokens: u32,
    pub blocks: u32,
    pub unary_bytes: usize,
    pub candidate_bytes: usize,
    pub hidden_bytes: usize,
    pub codebook_bytes: usize,
    pub anchor_bytes: usize,
    pub producer_status_bytes: usize,
    pub path_bytes: usize,
    pub status_bytes: usize,
}

impl Glm53Dflash2SelectorPlan {
    pub fn new(
        batch: u32,
        tokens: u32,
        selector_rank: u32,
        selector_top_k: u32,
        physical_vocab: u32,
        logical_vocab: u32,
        mask_token_id: u32,
    ) -> Result<Self> {
        if batch == 0 || batch > MAX_GRID_X || tokens == 0 || tokens > MAX_BLOCK_TOKENS {
            bail!("GLM DFlash2 selector requires batch>0 and 1..=8 tokens");
        }
        if selector_rank != SELECTOR_RANK || selector_top_k != SELECTOR_TOP_K {
            bail!("GLM DFlash2 selector requires exact rank256/top16 geometry");
        }
        if physical_vocab != PHYSICAL_VOCAB
            || logical_vocab != LOGICAL_VOCAB
            || mask_token_id != MASK_TOKEN_ID
            || mask_token_id >= logical_vocab
        {
            bail!("GLM DFlash2 selector vocabulary or mask-token contract drift");
        }
        let rows = usize::try_from(batch)?
            .checked_mul(usize::try_from(tokens)?)
            .context("GLM DFlash2 selector row overflow")?;
        let row_bytes = |columns: u32, width: usize, label: &str| -> Result<usize> {
            rows.checked_mul(usize::try_from(columns)?)
                .and_then(|count| count.checked_mul(width))
                .with_context(|| format!("GLM DFlash2 selector {label} byte overflow"))
        };
        let batch_bytes = |columns: u32, width: usize, label: &str| -> Result<usize> {
            usize::try_from(batch)?
                .checked_mul(usize::try_from(columns)?)
                .and_then(|count| count.checked_mul(width))
                .with_context(|| format!("GLM DFlash2 selector {label} byte overflow"))
        };
        let codebook_bytes = usize::try_from(physical_vocab)?
            .checked_mul(usize::try_from(selector_rank)?)
            .and_then(|count| count.checked_mul(2))
            .context("GLM DFlash2 selector codebook byte overflow")?;
        Ok(Self {
            batch,
            tokens,
            blocks: batch,
            unary_bytes: row_bytes(selector_top_k, 4, "unary")?,
            candidate_bytes: row_bytes(selector_top_k, 4, "candidate")?,
            hidden_bytes: row_bytes(selector_rank, 2, "hidden")?,
            codebook_bytes,
            anchor_bytes: batch_bytes(1, 4, "anchor")?,
            producer_status_bytes: row_bytes(1, 4, "producer status")?,
            path_bytes: row_bytes(1, 4, "path")?,
            status_bytes: batch_bytes(1, 4, "status")?,
        })
    }

    fn validate(self) -> Result<()> {
        let expected = Self::new(
            self.batch,
            self.tokens,
            SELECTOR_RANK,
            SELECTOR_TOP_K,
            PHYSICAL_VOCAB,
            LOGICAL_VOCAB,
            MASK_TOKEN_ID,
        )?;
        if self != expected {
            bail!("GLM DFlash2 selector plan was modified after admission");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2SelectorBuffers {
    /// Exact BF16 logits widened to F32 by the top-k producer.
    pub unary_f32: GgmlIqBuffer,
    pub candidates_u32: GgmlIqBuffer,
    pub hidden_bf16: GgmlIqBuffer,
    pub predecessor_bf16: GgmlIqBuffer,
    pub successor_bf16: GgmlIqBuffer,
    pub anchors_u32: GgmlIqBuffer,
    /// Per-row status from the top-k producer; every value must be zero.
    pub producer_status_u32: GgmlIqBuffer,
    pub path_u32: GgmlIqBuffer,
    /// Per-batch zero on success and one on rejected device data.
    pub status_u32: GgmlIqBuffer,
}

pub struct Glm53Dflash2SelectorKernel {
    greedy_walk: KernelHandle,
}

impl Glm53Dflash2SelectorKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            greedy_walk: gpu.kernel(
                "glm53_dflash2_selector",
                "atlas_glm53_dflash2_selector_walk",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2SelectorPlan,
        buffers: Glm53Dflash2SelectorBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.greedy_walk)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.unary_f32.ptr)
            .arg_ptr(buffers.candidates_u32.ptr)
            .arg_ptr(buffers.hidden_bf16.ptr)
            .arg_ptr(buffers.predecessor_bf16.ptr)
            .arg_ptr(buffers.successor_bf16.ptr)
            .arg_ptr(buffers.anchors_u32.ptr)
            .arg_ptr(buffers.producer_status_u32.ptr)
            .arg_ptr(buffers.path_u32.ptr)
            .arg_ptr(buffers.status_u32.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(SELECTOR_RANK)
            .arg_u32(SELECTOR_TOP_K)
            .arg_u32(PHYSICAL_VOCAB)
            .arg_u32(LOGICAL_VOCAB)
            .arg_u32(MASK_TOKEN_ID)
            .launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53Dflash2SelectorPlan,
    buffers: Glm53Dflash2SelectorBuffers,
) -> Result<()> {
    let named = [
        ("unary", buffers.unary_f32, plan.unary_bytes),
        ("candidates", buffers.candidates_u32, plan.candidate_bytes),
        ("hidden", buffers.hidden_bf16, plan.hidden_bytes),
        (
            "predecessor codebook",
            buffers.predecessor_bf16,
            plan.codebook_bytes,
        ),
        (
            "successor codebook",
            buffers.successor_bf16,
            plan.codebook_bytes,
        ),
        ("anchors", buffers.anchors_u32, plan.anchor_bytes),
        (
            "producer status",
            buffers.producer_status_u32,
            plan.producer_status_bytes,
        ),
        ("path", buffers.path_u32, plan.path_bytes),
        ("status", buffers.status_u32, plan.status_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM DFlash2 selector {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM DFlash2 selector {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DFlash2 selector device buffers overlap");
            }
        }
    }
    Ok(())
}

/// Coalesced BF16 context-row gather replacing Stage-1 per-sequence ring D2Ds.
///
/// Launches `dflash_stage1_gather_ctx_bf16` with `grid = [total_rows, 1, 1]` and
/// `block = [256, 1, 1]`. Each CTA `r` copies `row_elems` BF16 elements from
/// `src_ptrs[r] + src_row_idx[r] * row_elems` into `dst[r * row_elems ..]`.
///
/// # Errors
///
/// Returns an error if `kernel` is null (`KernelHandle(0)`), `total_rows` or
/// `row_elems` is zero or exceeds `i32::MAX`, any device pointer is null, or the
/// GPU backend launch fails.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_stage1_gather_ctx_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    dst: DevicePtr,
    src_ptrs: DevicePtr,
    src_row_idx: DevicePtr,
    total_rows: usize,
    row_elems: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dflash_stage1_gather_ctx_bf16 requires a non-zero kernel handle"
    );
    ensure!(
        total_rows > 0 && row_elems > 0,
        "dflash_stage1_gather_ctx_bf16 requires positive total_rows/row_elems (got total_rows={total_rows} row_elems={row_elems})"
    );
    for (name, ptr) in [
        ("dst", dst),
        ("src_ptrs", src_ptrs),
        ("src_row_idx", src_row_idx),
    ] {
        ensure!(
            !ptr.is_null(),
            "dflash_stage1_gather_ctx_bf16 {name} pointer is null"
        );
    }
    let total_rows_i32 = i32::try_from(total_rows).with_context(|| {
        format!("dflash_stage1_gather_ctx_bf16 total_rows {total_rows} exceeds i32")
    })?;
    let row_elems_i32 = i32::try_from(row_elems).with_context(|| {
        format!("dflash_stage1_gather_ctx_bf16 row_elems {row_elems} exceeds i32")
    })?;
    let grid_x = u32::try_from(total_rows_i32)
        .context("dflash_stage1_gather_ctx_bf16 grid_x conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([STAGE1_BLOCK_THREADS, 1, 1])
        .arg_ptr(dst)
        .arg_ptr(src_ptrs)
        .arg_ptr(src_row_idx)
        .arg_i32(total_rows_i32)
        .arg_i32(row_elems_i32)
        .launch(stream)
}

/// Fused YaRN-scaled RoPE and circular KV-ring scatter for Stage-1 context append.
///
/// Launches `dflash_rope_yarn_scaled_scatter_ring_bf16` with `grid = [total_rows, 1, 1]`
/// and `block = [256, 1, 1]`. Each CTA `r` applies bit-exact YaRN-scaled rotary
/// embedding (matching `rope_yarn_scaled`) to `k_in[r]` at `row_pos[r]`, writes the
/// rotated K (plus any unrotated `rotary_dim..head_dim` tail) directly to
/// `dst_k_ptrs[r] + dst_slot[r] * (num_kv_heads * head_dim)`, and copies `v_in[r]`
/// directly to `dst_v_ptrs[r] + dst_slot[r] * (num_kv_heads * head_dim)`.
///
/// # Errors
///
/// Returns an error if `kernel` is null, any dimension is zero or exceeds `i32::MAX`,
/// `rotary_dim` is odd or exceeds `head_dim`, `num_kv_heads * head_dim` overflows
/// `i32`, `attention_factor` is non-finite, any device pointer is null, or the GPU
/// backend launch fails.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn dflash_rope_yarn_scaled_scatter_ring_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    v_in: DevicePtr,
    dst_k_ptrs: DevicePtr,
    dst_v_ptrs: DevicePtr,
    dst_slot: DevicePtr,
    row_pos: DevicePtr,
    inv_freq: DevicePtr,
    attention_factor: f32,
    total_rows: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dflash_rope_yarn_scaled_scatter_ring_bf16 requires a non-zero kernel handle"
    );
    ensure!(
        total_rows > 0 && num_kv_heads > 0 && head_dim > 0,
        "dflash_rope_yarn_scaled_scatter_ring_bf16 requires positive total_rows/num_kv_heads/head_dim (got total_rows={total_rows} num_kv_heads={num_kv_heads} head_dim={head_dim})"
    );
    ensure!(
        rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim.is_multiple_of(2),
        "dflash_rope_yarn_scaled_scatter_ring_bf16 requires even rotary_dim in 1..={head_dim}, got {rotary_dim}"
    );
    ensure!(
        attention_factor.is_finite(),
        "dflash_rope_yarn_scaled_scatter_ring_bf16 requires finite attention_factor, got {attention_factor}"
    );
    for (name, ptr) in [
        ("k_in", k_in),
        ("v_in", v_in),
        ("dst_k_ptrs", dst_k_ptrs),
        ("dst_v_ptrs", dst_v_ptrs),
        ("dst_slot", dst_slot),
        ("row_pos", row_pos),
        ("inv_freq", inv_freq),
    ] {
        ensure!(
            !ptr.is_null(),
            "dflash_rope_yarn_scaled_scatter_ring_bf16 {name} pointer is null"
        );
    }
    let total_rows_i32 = i32::try_from(total_rows).with_context(|| {
        format!("dflash_rope_yarn_scaled_scatter_ring_bf16 total_rows {total_rows} exceeds i32")
    })?;
    let num_kv_heads_i32 = i32::try_from(num_kv_heads).with_context(|| {
        format!("dflash_rope_yarn_scaled_scatter_ring_bf16 num_kv_heads {num_kv_heads} exceeds i32")
    })?;
    let head_dim_i32 = i32::try_from(head_dim).with_context(|| {
        format!("dflash_rope_yarn_scaled_scatter_ring_bf16 head_dim {head_dim} exceeds i32")
    })?;
    let rotary_dim_i32 = i32::try_from(rotary_dim).with_context(|| {
        format!("dflash_rope_yarn_scaled_scatter_ring_bf16 rotary_dim {rotary_dim} exceeds i32")
    })?;
    let _ = num_kv_heads_i32
        .checked_mul(head_dim_i32)
        .context("dflash_rope_yarn_scaled_scatter_ring_bf16 kv_stride exceeds i32")?;
    let grid_x = u32::try_from(total_rows_i32)
        .context("dflash_rope_yarn_scaled_scatter_ring_bf16 grid_x conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([STAGE1_BLOCK_THREADS, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(v_in)
        .arg_ptr(dst_k_ptrs)
        .arg_ptr(dst_v_ptrs)
        .arg_ptr(dst_slot)
        .arg_ptr(row_pos)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .arg_i32(total_rows_i32)
        .arg_i32(num_kv_heads_i32)
        .arg_i32(head_dim_i32)
        .arg_i32(rotary_dim_i32)
        .launch(stream)
}

/// Batched probabilistic DFlash2 selector walk with GPU-resident post-policy `q` export.
///
/// Launches `dflash2_selector_walk_probabilistic_batched` with one 32-thread warp CTA
/// per sequence (`grid = [batch_size, 1, 1]`, `block = [32, 1, 1]`) for
/// `1 <= top_k_shortlist <= 32` and `out_q_stride >= 1`.
///
/// When `out_q_row_bases` is `Some(ptr)`, CTA `b` writes `out_q_ids` and `out_q_probs`
/// starting at row `out_q_row_bases[b]` (e.g. `slot_idx * 8`) for slot-isolated
/// addressing across concurrent Phase-A/Phase-B batches. When `None`, passes null so
/// the kernel falls back to contiguous row base `b * gamma`.
///
/// Sequences with `temperatures[b] <= 0.0` or non-finite temperature execute exact
/// greedy argmax (`out_max_prob = 1.0`) and write 1-hot `(best_id, 1.0)` in slot 0 with
/// `(0xFFFF_FFFF, 0.0)` padding. Sequences with positive finite temperature compute the
/// descending-sorted post-policy (`top_k`, `min_p`, nucleus `top_p`) renormalized draft
/// distribution `q[0..kept-1]`, sample `x_t` via exact multinomial CDF draw using
/// stateless `SplitMix64(seed, abs_pos, 0xDF2D5A31)`, write `out_max_prob = max_z q_t(z)`
/// (prefix-only confidence independent of sampled `x_t` for lossless `p_cum` stopping), and
/// export `(out_q_ids, out_q_probs)` directly to GPU memory for Wave-15 maximal-coupling
/// verification without host round-trips.
///
/// # Errors
///
/// Returns an error if `kernel` is null, `top_k_shortlist` is outside `1..=32`,
/// `out_q_stride` is zero, any dimension (`batch_size`, `gamma`, `rank`) is zero or
/// exceeds `i32::MAX`, any required device pointer (or `Some(DevicePtr::NULL)` for
/// `out_q_row_bases`) is null, or the GPU backend launch fails.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk_probabilistic_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden: DevicePtr,
    pred: DevicePtr,
    succ: DevicePtr,
    out_tokens: DevicePtr,
    out_max_prob: DevicePtr,
    out_q_ids: DevicePtr,
    out_q_probs: DevicePtr,
    out_q_row_bases: Option<DevicePtr>,
    anchor_ids: DevicePtr,
    seeds: DevicePtr,
    step0_abs_pos: DevicePtr,
    temperatures: DevicePtr,
    top_ks: DevicePtr,
    top_ps: DevicePtr,
    min_ps: DevicePtr,
    top_k_shortlist: usize,
    out_q_stride: usize,
    gamma: usize,
    rank: usize,
    batch_size: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dflash2_selector_walk_probabilistic_batched requires a non-zero kernel handle"
    );
    let max_shortlist = PROBABILISTIC_SELECTOR_MAX_SHORTLIST;
    ensure!(
        (1..=max_shortlist).contains(&top_k_shortlist),
        "dflash2_selector_walk_probabilistic_batched requires 1 <= top_k_shortlist <= {max_shortlist}, got {top_k_shortlist}"
    );
    ensure!(
        out_q_stride > 0,
        "dflash2_selector_walk_probabilistic_batched requires positive out_q_stride, got {out_q_stride}"
    );
    ensure!(
        batch_size > 0 && gamma > 0 && rank > 0,
        "dflash2_selector_walk_probabilistic_batched requires positive batch_size/gamma/rank (got batch_size={batch_size} gamma={gamma} rank={rank})"
    );
    for (name, ptr) in [
        ("unary", unary),
        ("candidates", candidates),
        ("hidden", hidden),
        ("pred", pred),
        ("succ", succ),
        ("out_tokens", out_tokens),
        ("out_max_prob", out_max_prob),
        ("out_q_ids", out_q_ids),
        ("out_q_probs", out_q_probs),
        ("anchor_ids", anchor_ids),
        ("seeds", seeds),
        ("step0_abs_pos", step0_abs_pos),
        ("temperatures", temperatures),
        ("top_ks", top_ks),
        ("top_ps", top_ps),
        ("min_ps", min_ps),
    ] {
        ensure!(
            !ptr.is_null(),
            "dflash2_selector_walk_probabilistic_batched {name} pointer is null"
        );
    }
    let out_q_row_bases_ptr = match out_q_row_bases {
        Some(ptr) => {
            ensure!(
                !ptr.is_null(),
                "dflash2_selector_walk_probabilistic_batched out_q_row_bases pointer is null (use None for default b * gamma)"
            );
            ptr
        }
        None => DevicePtr::NULL,
    };
    let top_k_shortlist_i32 = i32::try_from(top_k_shortlist).with_context(|| {
        format!(
            "dflash2_selector_walk_probabilistic_batched top_k_shortlist {top_k_shortlist} exceeds i32"
        )
    })?;
    let out_q_stride_i32 = i32::try_from(out_q_stride).with_context(|| {
        format!(
            "dflash2_selector_walk_probabilistic_batched out_q_stride {out_q_stride} exceeds i32"
        )
    })?;
    let gamma_i32 = i32::try_from(gamma).with_context(|| {
        format!("dflash2_selector_walk_probabilistic_batched gamma {gamma} exceeds i32")
    })?;
    let rank_i32 = i32::try_from(rank).with_context(|| {
        format!("dflash2_selector_walk_probabilistic_batched rank {rank} exceeds i32")
    })?;
    let batch_size_i32 = i32::try_from(batch_size).with_context(|| {
        format!("dflash2_selector_walk_probabilistic_batched batch_size {batch_size} exceeds i32")
    })?;
    let grid_x = u32::try_from(batch_size_i32)
        .context("dflash2_selector_walk_probabilistic_batched grid_x conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([PROBABILISTIC_SELECTOR_WARP_THREADS, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden)
        .arg_ptr(pred)
        .arg_ptr(succ)
        .arg_ptr(out_tokens)
        .arg_ptr(out_max_prob)
        .arg_ptr(out_q_ids)
        .arg_ptr(out_q_probs)
        .arg_ptr(out_q_row_bases_ptr)
        .arg_ptr(anchor_ids)
        .arg_ptr(seeds)
        .arg_ptr(step0_abs_pos)
        .arg_ptr(temperatures)
        .arg_ptr(top_ks)
        .arg_ptr(top_ps)
        .arg_ptr(min_ps)
        .arg_i32(top_k_shortlist_i32)
        .arg_i32(out_q_stride_i32)
        .arg_i32(gamma_i32)
        .arg_i32(rank_i32)
        .arg_i32(batch_size_i32)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn rounded(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn scheduled_score(unary: f32, pred: &[f32], hidden: &[f32], succ: &[f32]) -> f32 {
        let mut pair = 0.0f32;
        for ((&pred, &hidden), &succ) in pred.iter().zip(hidden).zip(succ) {
            let gate = rounded(pred * hidden);
            pair = gate.mul_add(succ, pair);
        }
        rounded(rounded(unary) + rounded(pair))
    }

    fn walk(
        unary: &[f32],
        candidates: &[u32],
        hidden: &[f32],
        pred: &[f32],
        succ: &[f32],
        tokens: usize,
        top_k: usize,
        rank: usize,
        producer_status: &[u32],
        mut predecessor: u32,
    ) -> Option<Vec<u32>> {
        if producer_status.len() != tokens || producer_status.iter().any(|&status| status != 0) {
            return None;
        }
        let mut path = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let mut best_slot = 0;
            let mut best = f32::NEG_INFINITY;
            for slot in 0..top_k {
                let candidate = candidates[token * top_k + slot] as usize;
                let score = scheduled_score(
                    unary[token * top_k + slot],
                    &pred[predecessor as usize * rank..][..rank],
                    &hidden[token * rank..][..rank],
                    &succ[candidate * rank..][..rank],
                );
                if score > best {
                    best = score;
                    best_slot = slot;
                }
            }
            predecessor = candidates[token * top_k + best_slot];
            path.push(predecessor);
        }
        Some(path)
    }

    #[test]
    fn exact_geometry_extents_and_bf16_schedule_are_pinned() {
        let plan = Glm53Dflash2SelectorPlan::new(2, 8, 256, 16, 154_880, 154_880, 154_856).unwrap();
        assert_eq!(plan.blocks, 2);
        assert_eq!(plan.unary_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.candidate_bytes, 2 * 8 * 16 * 4);
        assert_eq!(plan.hidden_bytes, 2 * 8 * 256 * 2);
        assert_eq!(plan.codebook_bytes, 154_880 * 256 * 2);
        assert_eq!(plan.anchor_bytes, 2 * 4);
        assert_eq!(plan.producer_status_bytes, 2 * 8 * 4);
        assert_eq!(plan.path_bytes, 2 * 8 * 4);
        assert_eq!(plan.status_bytes, 2 * 4);
        assert!(Glm53Dflash2SelectorPlan::new(0, 8, 256, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 9, 256, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 8, 255, 16, 154_880, 154_880, 154_856).is_err());
        assert!(Glm53Dflash2SelectorPlan::new(1, 8, 256, 16, 154_880, 154_880, 154_855).is_err());

        let mut found_rounding_difference = false;
        for seed in 1..4096usize {
            let sample = |factor: usize| rounded(((seed * factor) % 257) as f32 / 31.0 - 4.0);
            let pred = [sample(17), sample(29), sample(43)];
            let hidden = [sample(61), sample(73), sample(89)];
            let succ = [sample(101), sample(113), sample(127)];
            let exact = scheduled_score(sample(137), &pred, &hidden, &succ);
            let fused = rounded(
                sample(137)
                    + pred
                        .iter()
                        .zip(hidden)
                        .zip(succ)
                        .map(|((&p, h), s)| p * h * s)
                        .sum::<f32>(),
            );
            if exact.to_bits() != fused.to_bits() {
                found_rounding_difference = true;
                break;
            }
        }
        assert!(found_rounding_difference);
    }

    #[test]
    fn predecessor_is_the_prior_selected_token_and_ties_use_first_slot() {
        let unary = [0.5, 0.0, 0.0, 0.0];
        let candidates = [1, 2, 3, 4];
        let hidden = [1.0, 1.0, 1.0, 1.0];
        let pred = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let succ = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            walk(
                &unary,
                &candidates,
                &hidden,
                &pred,
                &succ,
                2,
                2,
                2,
                &[0, 0],
                0,
            )
            .unwrap(),
            [1, 3]
        );

        let tied = walk(
            &[0.0, 0.0],
            &[7, 6],
            &[0.0],
            &[0.0; 8],
            &[0.0; 8],
            1,
            2,
            1,
            &[0],
            0,
        )
        .unwrap();
        assert_eq!(tied, [7]);

        // Stale candidate/unary rows can be structurally valid. A failed
        // producer status must still prevent a selector path from existing.
        assert!(
            walk(
                &unary,
                &candidates,
                &hidden,
                &pred,
                &succ,
                2,
                2,
                2,
                &[0, 1],
                0,
            )
            .is_none()
        );
    }

    #[test]
    fn invalid_buffers_fail_before_launch() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53Dflash2SelectorKernel::load(&gpu).unwrap();
        let plan = Glm53Dflash2SelectorPlan::new(2, 8, 256, 16, 154_880, 154_880, 154_856).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53Dflash2SelectorBuffers {
            unary_f32: at(0x10_0000, plan.unary_bytes),
            candidates_u32: at(0x20_0000, plan.candidate_bytes),
            hidden_bf16: at(0x30_0000, plan.hidden_bytes),
            predecessor_bf16: at(0x1000_0000, plan.codebook_bytes),
            successor_bf16: at(0x2000_0000, plan.codebook_bytes),
            anchors_u32: at(0x3000_0000, plan.anchor_bytes),
            producer_status_u32: at(0x3001_0000, plan.producer_status_bytes),
            path_u32: at(0x3100_0000, plan.path_bytes),
            status_u32: at(0x3200_0000, plan.status_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53Dflash2SelectorBuffers {
                        path_u32: valid.producer_status_u32,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        let mut forged = plan;
        forged.tokens = 7;
        assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn stage1_gather_validates_args_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x1000);
        let k = KernelHandle(7);
        assert!(
            dflash_stage1_gather_ctx_bf16(&gpu, KernelHandle(0), p, p, p, 4, 20480, 0).is_err()
        );
        assert!(dflash_stage1_gather_ctx_bf16(&gpu, k, p, p, p, 0, 20480, 0).is_err());
        assert!(dflash_stage1_gather_ctx_bf16(&gpu, k, p, p, p, 4, 0, 0).is_err());
        assert!(
            dflash_stage1_gather_ctx_bf16(&gpu, k, DevicePtr::NULL, p, p, 4, 20480, 0).is_err()
        );
        assert!(
            dflash_stage1_gather_ctx_bf16(&gpu, k, p, DevicePtr::NULL, p, 4, 20480, 0).is_err()
        );
        assert!(
            dflash_stage1_gather_ctx_bf16(&gpu, k, p, p, DevicePtr::NULL, 4, 20480, 0).is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dflash_stage1_gather_ctx_bf16(&gpu, k, p, p, p, 16, 20480, 0)
            .expect("valid stage1 gather launch");
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn rope_yarn_scatter_ring_validates_args_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x1000);
        let k = KernelHandle(8);
        assert!(
            dflash_rope_yarn_scaled_scatter_ring_bf16(
                &gpu,
                KernelHandle(0),
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                1.0,
                8,
                4,
                128,
                128,
                0,
            )
            .is_err()
        );
        for (rows, kv_heads, hd, rd, attn_factor) in [
            (0, 4, 128, 128, 1.0),
            (8, 0, 128, 128, 1.0),
            (8, 4, 0, 0, 1.0),
            (8, 4, 128, 0, 1.0),
            (8, 4, 128, 63, 1.0),
            (8, 4, 128, 130, 1.0),
            (8, 4, 128, 128, f32::NAN),
            (8, 4, 128, 128, f32::INFINITY),
        ] {
            assert!(
                dflash_rope_yarn_scaled_scatter_ring_bf16(
                    &gpu,
                    k,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    attn_factor,
                    rows,
                    kv_heads,
                    hd,
                    rd,
                    0,
                )
                .is_err()
            );
        }
        assert!(
            dflash_rope_yarn_scaled_scatter_ring_bf16(
                &gpu,
                k,
                p,
                p,
                p,
                p,
                p,
                p,
                DevicePtr::NULL,
                1.0,
                8,
                4,
                128,
                128,
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dflash_rope_yarn_scaled_scatter_ring_bf16(
            &gpu, k, p, p, p, p, p, p, p, 1.1, 16, 4, 128, 64, 0,
        )
        .expect("valid fused rope scatter launch");
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn probabilistic_batched_selector_validates_args_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x1000);
        let k = KernelHandle(9);
        assert!(
            dflash2_selector_walk_probabilistic_batched(
                &gpu,
                KernelHandle(0),
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                None,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                20,
                20,
                8,
                256,
                4,
                0,
            )
            .is_err()
        );
        for (shortlist, q_stride, gamma, rank, batch_size) in [
            (0, 20, 8, 256, 4),
            (33, 32, 8, 256, 4),
            (20, 0, 8, 256, 4),
            (20, 20, 0, 256, 4),
            (20, 20, 8, 0, 4),
            (20, 20, 8, 256, 0),
        ] {
            assert!(
                dflash2_selector_walk_probabilistic_batched(
                    &gpu, k, p, p, p, p, p, p, p, p, p, None, p, p, p, p, p, p, p, shortlist,
                    q_stride, gamma, rank, batch_size, 0,
                )
                .is_err()
            );
        }
        assert!(
            dflash2_selector_walk_probabilistic_batched(
                &gpu,
                k,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                DevicePtr::NULL,
                p,
                None,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                20,
                32,
                8,
                256,
                4,
                0,
            )
            .is_err()
        );
        assert!(
            dflash2_selector_walk_probabilistic_batched(
                &gpu,
                k,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                Some(DevicePtr::NULL),
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                20,
                32,
                8,
                256,
                4,
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dflash2_selector_walk_probabilistic_batched(
            &gpu, k, p, p, p, p, p, p, p, p, p, None, p, p, p, p, p, p, p, 16, 20, 8, 256, 4, 0,
        )
        .expect("valid probabilistic batched selector launch (None row bases, k=16, stride=20)");
        dflash2_selector_walk_probabilistic_batched(
            &gpu,
            k,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            Some(p),
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            20,
            32,
            8,
            256,
            4,
            0,
        )
        .expect("valid probabilistic batched selector launch (Some row bases, k=20, stride=32)");
        assert_eq!(gpu.launch_count(), 2);
    }
}
