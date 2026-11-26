// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash 2 `GroupedDynamicCausalConv` + candidate-selector ops.
//!
//! Conv math (reference: z-lab/dflash `dflash/model.py`):
//!   `out[b,l,g,s] = sum_offset (base[stage][offset][g,s] + dyn[b,l,stage,offset,g]) * x[b,l-offset,g,s]`
//! with causal zero padding. `prepare` runs stage 0 and exports the stage-1
//! dynamic rows for `finish` (which runs stage 1 on the sublayer output).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// `dflash2_conv_prepare` — stage 0 conv of `hidden` plus export of the
/// stage-1 dynamic rows.
///
/// Args (device BF16 unless noted): hidden `[n_attn, hidden]`, dynamic
/// `[n_attn, 2*kernel*groups]`, base `[2, kernel, hidden]`, out
/// `[n_attn, hidden]`, dyn1_out `[n_attn, kernel*groups]`, n_attn, groups.
pub fn dflash2_conv_prepare(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    dyn1_out: DevicePtr,
    n_attn: u32,
    groups: u32,
    stream: u64,
) -> Result<()> {
    let elems = n_attn as u64 * groups as u64 * 16; // GROUP_SIZE = 16
    let grid = div_ceil(elems as u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_ptr(dyn1_out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .launch(stream)
}

/// `dflash2_conv_finish` — stage 1 conv of the sublayer output.
pub fn dflash2_conv_finish(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    n_attn: u32,
    groups: u32,
    stream: u64,
) -> Result<()> {
    let elems = n_attn as u64 * groups as u64 * 16;
    let grid = div_ceil(elems as u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .launch(stream)
}

/// `dflash2_conv_prepare_batched` — cross-sequence batched stage-0 conv with
/// causal padding reset every `block_len` rows.
///
/// Exact ABI: same arguments as [`dflash2_conv_prepare`] plus final
/// `block_len: u32`. `n_attn` is the packed row count (`batch_size * block_len`).
#[allow(clippy::too_many_arguments)]
pub fn dflash2_conv_prepare_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    dyn1_out: DevicePtr,
    n_attn: u32,
    groups: u32,
    block_len: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        n_attn > 0 && groups > 0 && block_len > 0 && n_attn.is_multiple_of(block_len),
        "dflash2_conv_prepare_batched invalid geometry: n_attn={n_attn} groups={groups} block_len={block_len}"
    );
    let elems = u64::from(n_attn)
        .checked_mul(u64::from(groups))
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| anyhow::anyhow!("dflash2_conv_prepare_batched element count overflow"))?;
    let elems_u32 = u32::try_from(elems)
        .map_err(|_| anyhow::anyhow!("dflash2_conv_prepare_batched elements {elems} exceed u32"))?;
    let grid = div_ceil(elems_u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_ptr(dyn1_out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .arg_u32(block_len)
        .launch(stream)
}

/// `dflash2_conv_finish_batched` — cross-sequence batched stage-1 conv with
/// causal padding reset every `block_len` rows.
///
/// Exact ABI: same arguments as [`dflash2_conv_finish`] plus final
/// `block_len: u32`. `n_attn` is the packed row count (`batch_size * block_len`).
#[allow(clippy::too_many_arguments)]
pub fn dflash2_conv_finish_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    dynamic: DevicePtr,
    base: DevicePtr,
    out: DevicePtr,
    n_attn: u32,
    groups: u32,
    block_len: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        n_attn > 0 && groups > 0 && block_len > 0 && n_attn.is_multiple_of(block_len),
        "dflash2_conv_finish_batched invalid geometry: n_attn={n_attn} groups={groups} block_len={block_len}"
    );
    let elems = u64::from(n_attn)
        .checked_mul(u64::from(groups))
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| anyhow::anyhow!("dflash2_conv_finish_batched element count overflow"))?;
    let elems_u32 = u32::try_from(elems)
        .map_err(|_| anyhow::anyhow!("dflash2_conv_finish_batched elements {elems} exceed u32"))?;
    let grid = div_ceil(elems_u32, 256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(dynamic)
        .arg_ptr(base)
        .arg_ptr(out)
        .arg_u32(n_attn)
        .arg_u32(groups)
        .arg_u32(block_len)
        .launch(stream)
}

/// `dflash2_selector_walk` — DFlash 2 candidate-selector greedy walk (T=0).
///
/// Reference: `CandidateSelector.select` (z-lab/dflash `dflash/model.py`),
/// greedy branch: per position, `scores[k] = unary[k] + (pred*hidden)·succ[k]`,
/// `index = argmax_k`, `predecessor = candidates[index]`.
///
/// Args:
///   unary      `[γ, TOP_K]` f32 (topk logits)
///   candidates `[γ, TOP_K]` u32 (topk token ids)
///   hidden_proj `[γ, rank]` BF16 (hidden_projection(draft hidden))
///   pred_codebook `[V, rank]` BF16
///   succ_codebook `[V, rank]` BF16
///   path       `[γ]` u32 out (selected draft token ids)
///   anchor_id  u32 — the last verified token (row 0's predecessor)
///   gamma      i32
///   rank       i32
///
/// One block of 16 threads (TOP_K), serialized across positions.
pub fn dflash2_selector_walk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden_proj: DevicePtr,
    pred_codebook: DevicePtr,
    succ_codebook: DevicePtr,
    path: DevicePtr,
    anchor_id: u32,
    gamma: i32,
    rank: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([16, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden_proj)
        .arg_ptr(pred_codebook)
        .arg_ptr(succ_codebook)
        .arg_ptr(path)
        .arg_u32(anchor_id)
        .arg_i32(gamma)
        .arg_i32(rank)
        .launch(stream)
}

/// `dflash2_selector_walk_batched` — cross-sequence batched DFlash 2
/// candidate-selector greedy walk (T=0).
///
/// Exact ABI:
/// `(unary, candidates, hidden, pred, succ, path, anchor_ids, gamma, rank, batch_size)`
/// Grid `[batch_size, 1, 1]`, Block `[16, 1, 1]`. Each CTA `b` reads
/// `anchor_ids[b]` from device memory and walks `[b*gamma .. (b+1)*gamma)`.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden_proj: DevicePtr,
    pred_codebook: DevicePtr,
    succ_codebook: DevicePtr,
    path: DevicePtr,
    anchor_ids: DevicePtr,
    gamma: i32,
    rank: i32,
    batch_size: i32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        batch_size > 0 && gamma > 0 && rank > 0,
        "dflash2_selector_walk_batched requires positive batch_size/gamma/rank (got batch_size={batch_size} gamma={gamma} rank={rank})"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([batch_size as u32, 1, 1])
        .block([16, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden_proj)
        .arg_ptr(pred_codebook)
        .arg_ptr(succ_codebook)
        .arg_ptr(path)
        .arg_ptr(anchor_ids)
        .arg_i32(gamma)
        .arg_i32(rank)
        .arg_i32(batch_size)
        .launch(stream)
}

/// Sampled DFlash2 selector walk. This is intentionally a different CUDA
/// symbol and Rust wrapper from [`dflash2_selector_walk`], preserving the
/// released greedy ABI byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk_probabilistic(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden_proj: DevicePtr,
    pred_codebook: DevicePtr,
    succ_codebook: DevicePtr,
    path: DevicePtr,
    uniforms: DevicePtr,
    q_modifier_token_ids: DevicePtr,
    q_modifier_values: DevicePtr,
    q_modifier_count: i32,
    temperature: f32,
    top_k: u32,
    min_p: f32,
    top_p: f32,
    q_probabilities: DevicePtr,
    q_candidates: DevicePtr,
    anchor_id: u32,
    gamma: i32,
    rank: i32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([16, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden_proj)
        .arg_ptr(pred_codebook)
        .arg_ptr(succ_codebook)
        .arg_ptr(path)
        .arg_ptr(uniforms)
        .arg_ptr(q_modifier_token_ids)
        .arg_ptr(q_modifier_values)
        .arg_i32(q_modifier_count)
        .arg_f32(temperature)
        .arg_u32(top_k)
        .arg_f32(min_p)
        .arg_f32(top_p)
        .arg_ptr(q_probabilities)
        .arg_ptr(q_candidates)
        .arg_u32(anchor_id)
        .arg_i32(gamma)
        .arg_i32(rank)
        .launch(stream)
}

/// Batched CRN noise-coupled Gumbel-Max DFlash2 selector walk.
///
/// Launches one 16-thread CTA per sequence in `0..batch_size` (`grid = [batch_size, 1, 1]`,
/// `block = [16, 1, 1]`). Sequences with `temperatures[b] <= 0.0` or non-finite temperature
/// execute the exact greedy argmax path of [`dflash2_selector_walk_batched`]; positive finite
/// temperatures apply temperature scaling, `top_k`, `min_p`, `top_p`, and position-coupled
/// SplitMix64 Gumbel-Max selection.
///
/// # Errors
///
/// Returns an error if `kernel` is null (`KernelHandle(0)`), any dimension (`batch_size`,
/// `gamma`, `rank`) is zero or exceeds `i32::MAX`, any device pointer is null, or the GPU
/// backend launch fails.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk_gumbel_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden: DevicePtr,
    pred: DevicePtr,
    succ: DevicePtr,
    path: DevicePtr,
    anchor_ids: DevicePtr,
    seeds: DevicePtr,
    step0_abs_pos: DevicePtr,
    temperatures: DevicePtr,
    top_ks: DevicePtr,
    top_ps: DevicePtr,
    min_ps: DevicePtr,
    gamma: usize,
    rank: usize,
    batch_size: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0,
        "dflash2_selector_walk_gumbel_batched requires a non-zero kernel handle"
    );
    anyhow::ensure!(
        batch_size > 0 && gamma > 0 && rank > 0,
        "dflash2_selector_walk_gumbel_batched requires positive batch_size/gamma/rank (got batch_size={batch_size} gamma={gamma} rank={rank})"
    );
    for (name, ptr) in [
        ("unary", unary),
        ("candidates", candidates),
        ("hidden", hidden),
        ("pred", pred),
        ("succ", succ),
        ("path", path),
        ("anchor_ids", anchor_ids),
        ("seeds", seeds),
        ("step0_abs_pos", step0_abs_pos),
        ("temperatures", temperatures),
        ("top_ks", top_ks),
        ("top_ps", top_ps),
        ("min_ps", min_ps),
    ] {
        anyhow::ensure!(
            !ptr.is_null(),
            "dflash2_selector_walk_gumbel_batched {name} pointer is null"
        );
    }
    let gamma_i32 = i32::try_from(gamma).map_err(|_| {
        anyhow::anyhow!("dflash2_selector_walk_gumbel_batched gamma {gamma} exceeds i32")
    })?;
    let rank_i32 = i32::try_from(rank).map_err(|_| {
        anyhow::anyhow!("dflash2_selector_walk_gumbel_batched rank {rank} exceeds i32")
    })?;
    let batch_size_i32 = i32::try_from(batch_size).map_err(|_| {
        anyhow::anyhow!("dflash2_selector_walk_gumbel_batched batch_size {batch_size} exceeds i32")
    })?;

    KernelLaunch::new(gpu, kernel)
        .grid([batch_size as u32, 1, 1])
        .block([16, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden)
        .arg_ptr(pred)
        .arg_ptr(succ)
        .arg_ptr(path)
        .arg_ptr(anchor_ids)
        .arg_ptr(seeds)
        .arg_ptr(step0_abs_pos)
        .arg_ptr(temperatures)
        .arg_ptr(top_ks)
        .arg_ptr(top_ps)
        .arg_ptr(min_ps)
        .arg_i32(gamma_i32)
        .arg_i32(rank_i32)
        .arg_i32(batch_size_i32)
        .launch(stream)
}

/// Widened (`1 <= top_k_shortlist <= 32`) batched CRN noise-coupled Gumbel-Max
/// DFlash2 selector walk with per-position confidence probabilities.
///
/// Launches `dflash2_selector_walk_gumbel_batched_k32` with one 32-thread warp CTA
/// per sequence (`grid = [batch_size, 1, 1]`, `block = [32, 1, 1]`). Sequences with
/// `temperatures[b] <= 0.0` or non-finite temperature execute the exact greedy argmax
/// path plus shortlist softmax confidence; positive finite temperatures apply
/// temperature scaling, `top_k`, `min_p`, `top_p`, and position-coupled SplitMix64
/// Gumbel-Max selection with post-policy confidence written to `probs_out`.
///
/// # Errors
///
/// Returns an error if `kernel` is null (`KernelHandle(0)`), `top_k_shortlist` is
/// outside `1..=32`, any dimension (`batch_size`, `gamma`, `rank`) is zero or
/// exceeds `i32::MAX`, any of the 14 device pointers is null, or the GPU backend
/// launch fails.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_walk_gumbel_batched_k32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    unary: DevicePtr,
    candidates: DevicePtr,
    hidden: DevicePtr,
    pred: DevicePtr,
    succ: DevicePtr,
    path_out: DevicePtr,
    probs_out: DevicePtr,
    anchor_ids: DevicePtr,
    seeds: DevicePtr,
    step0_abs_pos: DevicePtr,
    temperatures: DevicePtr,
    top_ks: DevicePtr,
    top_ps: DevicePtr,
    min_ps: DevicePtr,
    top_k_shortlist: usize,
    gamma: usize,
    rank: usize,
    batch_size: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        kernel.0 != 0,
        "dflash2_selector_walk_gumbel_batched_k32 requires a non-zero kernel handle"
    );
    anyhow::ensure!(
        (1..=32).contains(&top_k_shortlist),
        "dflash2_selector_walk_gumbel_batched_k32 requires 1 <= top_k_shortlist <= 32, got {top_k_shortlist}"
    );
    anyhow::ensure!(
        batch_size > 0 && gamma > 0 && rank > 0,
        "dflash2_selector_walk_gumbel_batched_k32 requires positive batch_size/gamma/rank (got batch_size={batch_size} gamma={gamma} rank={rank})"
    );
    for (name, ptr) in [
        ("unary", unary),
        ("candidates", candidates),
        ("hidden", hidden),
        ("pred", pred),
        ("succ", succ),
        ("path_out", path_out),
        ("probs_out", probs_out),
        ("anchor_ids", anchor_ids),
        ("seeds", seeds),
        ("step0_abs_pos", step0_abs_pos),
        ("temperatures", temperatures),
        ("top_ks", top_ks),
        ("top_ps", top_ps),
        ("min_ps", min_ps),
    ] {
        anyhow::ensure!(
            !ptr.is_null(),
            "dflash2_selector_walk_gumbel_batched_k32 {name} pointer is null"
        );
    }
    let top_k_shortlist_i32 = i32::try_from(top_k_shortlist).map_err(|_| {
        anyhow::anyhow!(
            "dflash2_selector_walk_gumbel_batched_k32 top_k_shortlist {top_k_shortlist} exceeds i32"
        )
    })?;
    let gamma_i32 = i32::try_from(gamma).map_err(|_| {
        anyhow::anyhow!("dflash2_selector_walk_gumbel_batched_k32 gamma {gamma} exceeds i32")
    })?;
    let rank_i32 = i32::try_from(rank).map_err(|_| {
        anyhow::anyhow!("dflash2_selector_walk_gumbel_batched_k32 rank {rank} exceeds i32")
    })?;
    let batch_size_i32 = i32::try_from(batch_size).map_err(|_| {
        anyhow::anyhow!(
            "dflash2_selector_walk_gumbel_batched_k32 batch_size {batch_size} exceeds i32"
        )
    })?;

    KernelLaunch::new(gpu, kernel)
        .grid([batch_size as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(unary)
        .arg_ptr(candidates)
        .arg_ptr(hidden)
        .arg_ptr(pred)
        .arg_ptr(succ)
        .arg_ptr(path_out)
        .arg_ptr(probs_out)
        .arg_ptr(anchor_ids)
        .arg_ptr(seeds)
        .arg_ptr(step0_abs_pos)
        .arg_ptr(temperatures)
        .arg_ptr(top_ks)
        .arg_ptr(top_ps)
        .arg_ptr(min_ps)
        .arg_i32(top_k_shortlist_i32)
        .arg_i32(gamma_i32)
        .arg_i32(rank_i32)
        .arg_i32(batch_size_i32)
        .launch(stream)
}

#[cfg(test)]
mod k32_selector_tests {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, KernelHandle};

    use super::dflash2_selector_walk_gumbel_batched_k32;

    #[test]
    fn validates_shortlist_dims_and_pointers_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x1000);
        assert!(
            dflash2_selector_walk_gumbel_batched_k32(
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
                p,
                p,
                p,
                p,
                p,
                32,
                8,
                256,
                4,
                0,
            )
            .is_err()
        );
        for (shortlist, gamma, rank, batch_size) in [
            (0, 8, 256, 4),
            (33, 8, 256, 4),
            (32, 0, 256, 4),
            (32, 8, 0, 4),
            (32, 8, 256, 0),
        ] {
            assert!(
                dflash2_selector_walk_gumbel_batched_k32(
                    &gpu,
                    KernelHandle(5),
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    p,
                    shortlist,
                    gamma,
                    rank,
                    batch_size,
                    0,
                )
                .is_err()
            );
        }
        assert!(
            dflash2_selector_walk_gumbel_batched_k32(
                &gpu,
                KernelHandle(5),
                p,
                p,
                p,
                p,
                p,
                p,
                DevicePtr(0),
                p,
                p,
                p,
                p,
                p,
                p,
                p,
                32,
                8,
                256,
                4,
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dflash2_selector_walk_gumbel_batched_k32(
            &gpu,
            KernelHandle(5),
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            32,
            8,
            256,
            4,
            0,
        )
        .expect("valid k32 selector launch");
        assert_eq!(gpu.launch_count(), 1);
    }
}
