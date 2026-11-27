// SPDX-License-Identifier: AGPL-3.0-only

//! Validated GPU kernel launch wrappers for DCOCO contextual draft correction
//! (`kernels/gb10/common/dcoco_selector.cu`) and the 4-gate safe in-context
//! n-gram DFlash2 selector walk (`dflash2_selector_walk_ngram_batched` in
//! `kernels/gb10/common/dflash2_selector.cu`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Maximum shortlist width supported by `dcoco_proj_fuse_bf16` and `dcoco_pair_mlp_bf16`.
pub const DCOCO_PAIR_MAX_K: usize = 16;

/// Maximum shortlist width supported by `dcoco_walk_and_export_q` and
/// `dflash2_selector_walk_ngram_batched`.
pub const DCOCO_WALK_MAX_K: usize = 32;

/// CTA width for `dcoco_proj_fuse_bf16` and `dcoco_pair_mlp_bf16` (8 warps).
const DCOCO_CTA_THREADS: u32 = 256;

/// Single-warp CTA width for `dcoco_walk_and_export_q` and
/// `dflash2_selector_walk_ngram_batched`.
const SELECTOR_WARP_THREADS: u32 = 32;

/// Execution mode for `dcoco_walk_and_export_q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DcocoWalkMode {
    /// Exact greedy argmax over precomputed transition scores `S[b, t, a, :]`.
    Greedy = 0,
    /// Common-random-numbers SplitMix64 Gumbel-Max sampling.
    GumbelMax = 1,
    /// Wave-15 maximal-coupling rejection sampling exporting conditional
    /// `q_t(. | k_p)` and $\mathcal{F}_t$-measurable `out_max_prob`.
    MaximalCoupling = 2,
}

impl DcocoWalkMode {
    /// Numeric ABI tag passed to `dcoco_walk_and_export_q`.
    #[inline]
    #[must_use]
    pub const fn as_abi_i32(self) -> i32 {
        self as i32
    }
}

/// Device buffers and geometry for `dcoco_proj_fuse_bf16`.
#[derive(Debug, Clone, Copy)]
pub struct DcocoProjFuseArgs {
    pub embed_tokens: DevicePtr,
    pub hidden_states: DevicePtr,
    pub anchor_ids: DevicePtr,
    pub candidates: DevicePtr,
    pub w_cand: DevicePtr,
    pub w_ctx: DevicePtr,
    pub w_fuse: DevicePtr,
    pub fuse_ln_w: DevicePtr,
    pub fuse_ln_b: DevicePtr,
    pub w_pair: DevicePtr,
    pub z_out: DevicePtr,
    pub s_out: DevicePtr,
    pub ln_eps: f32,
    pub batch_size: usize,
    pub gamma: usize,
    pub top_k: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
}

/// Launch `dcoco_proj_fuse_bf16` (`grid = [batch_size * (gamma + 1), 1, 1]`,
/// `block = [256, 1, 1]`).
///
/// Computes fused candidate embeddings `z_out[B, gamma + 1, top_k, 128]` and
/// adjacent pair states `s_out[B, gamma, 128]` directly on the active GPU stream.
///
/// # Errors
///
/// Returns an error if `kernel` is null, any dimension is zero or out of ABI bounds,
/// `ln_eps` is non-finite or non-positive, any device pointer is null, or the GPU
/// backend launch fails.
pub fn dcoco_proj_fuse_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &DcocoProjFuseArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dcoco_proj_fuse_bf16 requires a non-zero kernel handle"
    );
    ensure!(
        args.batch_size > 0 && args.gamma > 0 && args.hidden_dim > 0 && args.vocab_size > 0,
        "dcoco_proj_fuse_bf16 requires positive batch_size/gamma/hidden_dim/vocab_size \
         (got batch_size={} gamma={} hidden_dim={} vocab_size={})",
        args.batch_size,
        args.gamma,
        args.hidden_dim,
        args.vocab_size,
    );
    ensure!(
        (1..=DCOCO_PAIR_MAX_K).contains(&args.top_k),
        "dcoco_proj_fuse_bf16 requires 1 <= top_k <= {DCOCO_PAIR_MAX_K}, got {}",
        args.top_k,
    );
    ensure!(
        args.ln_eps.is_finite() && args.ln_eps > 0.0,
        "dcoco_proj_fuse_bf16 requires finite positive ln_eps, got {}",
        args.ln_eps,
    );
    for (name, ptr) in [
        ("embed_tokens", args.embed_tokens),
        ("hidden_states", args.hidden_states),
        ("anchor_ids", args.anchor_ids),
        ("candidates", args.candidates),
        ("w_cand", args.w_cand),
        ("w_ctx", args.w_ctx),
        ("w_fuse", args.w_fuse),
        ("fuse_ln_w", args.fuse_ln_w),
        ("fuse_ln_b", args.fuse_ln_b),
        ("w_pair", args.w_pair),
        ("z_out", args.z_out),
        ("s_out", args.s_out),
    ] {
        ensure!(
            !ptr.is_null(),
            "dcoco_proj_fuse_bf16 {name} pointer is null"
        );
    }

    let slots_per_seq = args
        .gamma
        .checked_add(1)
        .context("dcoco_proj_fuse_bf16 gamma + 1 overflow")?;
    let total_slots = args
        .batch_size
        .checked_mul(slots_per_seq)
        .context("dcoco_proj_fuse_bf16 grid_x overflow")?;
    let grid_x = u32::try_from(total_slots)
        .with_context(|| format!("dcoco_proj_fuse_bf16 grid_x {total_slots} exceeds u32"))?;
    let batch_i32 =
        i32::try_from(args.batch_size).context("dcoco_proj_fuse_bf16 batch exceeds i32")?;
    let gamma_i32 = i32::try_from(args.gamma).context("dcoco_proj_fuse_bf16 gamma exceeds i32")?;
    let top_k_i32 = i32::try_from(args.top_k).context("dcoco_proj_fuse_bf16 top_k exceeds i32")?;
    let hidden_i32 =
        i32::try_from(args.hidden_dim).context("dcoco_proj_fuse_bf16 hidden_dim exceeds i32")?;
    let vocab_i32 =
        i32::try_from(args.vocab_size).context("dcoco_proj_fuse_bf16 vocab_size exceeds i32")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([DCOCO_CTA_THREADS, 1, 1])
        .arg_ptr(args.embed_tokens)
        .arg_ptr(args.hidden_states)
        .arg_ptr(args.anchor_ids)
        .arg_ptr(args.candidates)
        .arg_ptr(args.w_cand)
        .arg_ptr(args.w_ctx)
        .arg_ptr(args.w_fuse)
        .arg_ptr(args.fuse_ln_w)
        .arg_ptr(args.fuse_ln_b)
        .arg_ptr(args.w_pair)
        .arg_ptr(args.z_out)
        .arg_ptr(args.s_out)
        .arg_f32(args.ln_eps)
        .arg_i32(batch_i32)
        .arg_i32(gamma_i32)
        .arg_i32(top_k_i32)
        .arg_i32(hidden_i32)
        .arg_i32(vocab_i32)
        .launch(stream)
}

/// Device buffers and geometry for `dcoco_pair_mlp_bf16`.
#[derive(Debug, Clone, Copy)]
pub struct DcocoPairMlpArgs {
    pub s_pair: DevicePtr,
    pub z_cand: DevicePtr,
    pub unary: DevicePtr,
    pub anchor_logits: Option<DevicePtr>,
    pub mlp_w1: DevicePtr,
    pub mlp_b1: DevicePtr,
    pub mlp_w2: DevicePtr,
    pub mlp_b2: DevicePtr,
    pub out_scores: DevicePtr,
    pub batch_size: usize,
    pub gamma: usize,
    pub top_k: usize,
}

/// Launch `dcoco_pair_mlp_bf16` (`grid = [gamma, batch_size, 1]`, `block = [256, 1, 1]`).
///
/// Computes all `K x K` transition scores `out_scores[B, gamma, top_k, top_k]`
/// in SMEM via factored BF16 Tensor-Core MMA without materializing any
/// `[B * gamma * K^2, 386]` global intermediate.
///
/// # Errors
///
/// Returns an error if `kernel` is null, any dimension is zero or out of bounds,
/// any required device pointer (or `Some(DevicePtr::NULL)` for `anchor_logits`) is
/// null, or the GPU backend launch fails.
pub fn dcoco_pair_mlp_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &DcocoPairMlpArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dcoco_pair_mlp_bf16 requires a non-zero kernel handle"
    );
    ensure!(
        args.batch_size > 0 && args.gamma > 0,
        "dcoco_pair_mlp_bf16 requires positive batch_size/gamma (got batch_size={} gamma={})",
        args.batch_size,
        args.gamma,
    );
    ensure!(
        (1..=DCOCO_PAIR_MAX_K).contains(&args.top_k),
        "dcoco_pair_mlp_bf16 requires 1 <= top_k <= {DCOCO_PAIR_MAX_K}, got {}",
        args.top_k,
    );
    for (name, ptr) in [
        ("s_pair", args.s_pair),
        ("z_cand", args.z_cand),
        ("unary", args.unary),
        ("mlp_w1", args.mlp_w1),
        ("mlp_b1", args.mlp_b1),
        ("mlp_w2", args.mlp_w2),
        ("mlp_b2", args.mlp_b2),
        ("out_scores", args.out_scores),
    ] {
        ensure!(!ptr.is_null(), "dcoco_pair_mlp_bf16 {name} pointer is null");
    }
    let anchor_logits_ptr =
        optional_non_null("dcoco_pair_mlp_bf16", "anchor_logits", args.anchor_logits)?;

    let batch_i32 =
        i32::try_from(args.batch_size).context("dcoco_pair_mlp_bf16 batch exceeds i32")?;
    let gamma_i32 = i32::try_from(args.gamma).context("dcoco_pair_mlp_bf16 gamma exceeds i32")?;
    let top_k_i32 = i32::try_from(args.top_k).context("dcoco_pair_mlp_bf16 top_k exceeds i32")?;
    let grid_x =
        u32::try_from(gamma_i32).context("dcoco_pair_mlp_bf16 grid_x conversion failed")?;
    let grid_y =
        u32::try_from(batch_i32).context("dcoco_pair_mlp_bf16 grid_y conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, grid_y, 1])
        .block([DCOCO_CTA_THREADS, 1, 1])
        .arg_ptr(args.s_pair)
        .arg_ptr(args.z_cand)
        .arg_ptr(args.unary)
        .arg_ptr(anchor_logits_ptr)
        .arg_ptr(args.mlp_w1)
        .arg_ptr(args.mlp_b1)
        .arg_ptr(args.mlp_w2)
        .arg_ptr(args.mlp_b2)
        .arg_ptr(args.out_scores)
        .arg_i32(batch_i32)
        .arg_i32(gamma_i32)
        .arg_i32(top_k_i32)
        .launch(stream)
}

/// Device buffers and geometry for `dcoco_walk_and_export_q`.
#[derive(Debug, Clone, Copy)]
pub struct DcocoWalkArgs {
    pub transition_scores: DevicePtr,
    pub candidates: DevicePtr,
    pub out_tokens: DevicePtr,
    pub out_max_prob: DevicePtr,
    pub out_q_ids: Option<DevicePtr>,
    pub out_q_probs: Option<DevicePtr>,
    pub out_q_row_bases: Option<DevicePtr>,
    pub seeds: Option<DevicePtr>,
    pub step0_abs_pos: Option<DevicePtr>,
    pub temperatures: Option<DevicePtr>,
    pub draft_temp_scales: Option<DevicePtr>,
    pub top_ks: Option<DevicePtr>,
    pub top_ps: Option<DevicePtr>,
    pub min_ps: Option<DevicePtr>,
    pub walk_mode: DcocoWalkMode,
    pub top_k_shortlist: usize,
    pub out_q_stride: usize,
    pub gamma: usize,
    pub batch_size: usize,
}

/// Launch `dcoco_walk_and_export_q` (`grid = [batch_size, 1, 1]`, `block = [32, 1, 1]`).
///
/// Walks `transition_scores[B, gamma, K, K]` in Greedy, CRN Gumbel-Max, or
/// Wave-15 Maximal-Coupling mode, exporting optional normalized conditional
/// `(out_q_ids, out_q_probs)` and $\mathcal{F}_t$-measurable `out_max_prob`.
///
/// # Errors
///
/// Returns an error if `kernel` is null, dimensions are out of bounds, sampling
/// buffers are missing in sampled modes, `out_q_ids`/`out_q_probs` are only half
/// supplied or paired with `out_q_stride == 0`, any pointer is `Some(NULL)`, or
/// the GPU backend launch fails.
pub fn dcoco_walk_and_export_q(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &DcocoWalkArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dcoco_walk_and_export_q requires a non-zero kernel handle"
    );
    ensure!(
        args.batch_size > 0 && args.gamma > 0,
        "dcoco_walk_and_export_q requires positive batch_size/gamma (got batch_size={} gamma={})",
        args.batch_size,
        args.gamma,
    );
    ensure!(
        (1..=DCOCO_WALK_MAX_K).contains(&args.top_k_shortlist),
        "dcoco_walk_and_export_q requires 1 <= top_k_shortlist <= {DCOCO_WALK_MAX_K}, got {}",
        args.top_k_shortlist,
    );
    for (name, ptr) in [
        ("transition_scores", args.transition_scores),
        ("candidates", args.candidates),
        ("out_tokens", args.out_tokens),
        ("out_max_prob", args.out_max_prob),
    ] {
        ensure!(
            !ptr.is_null(),
            "dcoco_walk_and_export_q {name} pointer is null"
        );
    }
    ensure!(
        args.out_q_ids.is_some() == args.out_q_probs.is_some(),
        "dcoco_walk_and_export_q requires out_q_ids and out_q_probs to both be Some or both None"
    );
    if args.out_q_ids.is_some() {
        ensure!(
            args.out_q_stride > 0,
            "dcoco_walk_and_export_q requires positive out_q_stride when q export pointers are provided"
        );
    }
    if args.walk_mode != DcocoWalkMode::Greedy {
        for (name, ptr) in [
            ("seeds", args.seeds),
            ("step0_abs_pos", args.step0_abs_pos),
            ("temperatures", args.temperatures),
            ("top_ks", args.top_ks),
            ("top_ps", args.top_ps),
            ("min_ps", args.min_ps),
        ] {
            ensure!(
                ptr.is_some(),
                "dcoco_walk_and_export_q mode {:?} requires non-None {name}",
                args.walk_mode,
            );
        }
    }

    let out_q_ids_ptr = optional_non_null("dcoco_walk_and_export_q", "out_q_ids", args.out_q_ids)?;
    let out_q_probs_ptr =
        optional_non_null("dcoco_walk_and_export_q", "out_q_probs", args.out_q_probs)?;
    let out_q_row_bases_ptr = optional_non_null(
        "dcoco_walk_and_export_q",
        "out_q_row_bases",
        args.out_q_row_bases,
    )?;
    let seeds_ptr = optional_non_null("dcoco_walk_and_export_q", "seeds", args.seeds)?;
    let step0_ptr = optional_non_null(
        "dcoco_walk_and_export_q",
        "step0_abs_pos",
        args.step0_abs_pos,
    )?;
    let temps_ptr =
        optional_non_null("dcoco_walk_and_export_q", "temperatures", args.temperatures)?;
    let scales_ptr = optional_non_null(
        "dcoco_walk_and_export_q",
        "draft_temp_scales",
        args.draft_temp_scales,
    )?;
    let top_ks_ptr = optional_non_null("dcoco_walk_and_export_q", "top_ks", args.top_ks)?;
    let top_ps_ptr = optional_non_null("dcoco_walk_and_export_q", "top_ps", args.top_ps)?;
    let min_ps_ptr = optional_non_null("dcoco_walk_and_export_q", "min_ps", args.min_ps)?;

    let top_k_i32 = i32::try_from(args.top_k_shortlist)
        .context("dcoco_walk_and_export_q top_k_shortlist exceeds i32")?;
    let out_q_stride_i32 = i32::try_from(args.out_q_stride)
        .context("dcoco_walk_and_export_q out_q_stride exceeds i32")?;
    let gamma_i32 =
        i32::try_from(args.gamma).context("dcoco_walk_and_export_q gamma exceeds i32")?;
    let batch_i32 =
        i32::try_from(args.batch_size).context("dcoco_walk_and_export_q batch_size exceeds i32")?;
    let grid_x =
        u32::try_from(batch_i32).context("dcoco_walk_and_export_q grid_x conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([SELECTOR_WARP_THREADS, 1, 1])
        .arg_ptr(args.transition_scores)
        .arg_ptr(args.candidates)
        .arg_ptr(args.out_tokens)
        .arg_ptr(args.out_max_prob)
        .arg_ptr(out_q_ids_ptr)
        .arg_ptr(out_q_probs_ptr)
        .arg_ptr(out_q_row_bases_ptr)
        .arg_ptr(seeds_ptr)
        .arg_ptr(step0_ptr)
        .arg_ptr(temps_ptr)
        .arg_ptr(scales_ptr)
        .arg_ptr(top_ks_ptr)
        .arg_ptr(top_ps_ptr)
        .arg_ptr(min_ps_ptr)
        .arg_i32(args.walk_mode.as_abi_i32())
        .arg_i32(top_k_i32)
        .arg_i32(out_q_stride_i32)
        .arg_i32(gamma_i32)
        .arg_i32(batch_i32)
        .launch(stream)
}

/// Device buffers and hyperparameters for `dflash2_selector_walk_ngram_batched`.
#[derive(Debug, Clone, Copy)]
pub struct Dflash2NgramWalkArgs {
    pub unary: DevicePtr,
    pub candidates: DevicePtr,
    pub hidden: DevicePtr,
    pub pred: DevicePtr,
    pub succ: DevicePtr,
    pub out_tokens: DevicePtr,
    pub out_probs: Option<DevicePtr>,
    pub anchor_ids: DevicePtr,
    pub prev3_tokens: Option<DevicePtr>,
    pub ngram_tables: Option<DevicePtr>,
    pub temperatures: Option<DevicePtr>,
    pub margin_thresh: f32,
    pub max_bonus: f32,
    pub top_k_shortlist: usize,
    pub gamma: usize,
    pub rank: usize,
    pub batch_size: usize,
}

/// Launch `dflash2_selector_walk_ngram_batched` (`grid = [batch_size, 1, 1]`,
/// `block = [32, 1, 1]`).
///
/// 4-gate safe in-context n-gram bonus walk (`T=0` ONLY, uncertainty margin gate,
/// top-4 verbatim 3/4-gram only, hard logit cap `<= 0.75`, anti-loop `<= 64`
/// exclusion enforced in host table construction).
///
/// # Errors
///
/// Returns an error if `kernel` is null, dimensions are out of bounds,
/// `margin_thresh` / `max_bonus` are non-finite or negative, any required pointer
/// (or `Some(DevicePtr::NULL)` for optional pointers) is null, or the GPU backend
/// launch fails.
pub fn dflash2_selector_walk_ngram_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Dflash2NgramWalkArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "dflash2_selector_walk_ngram_batched requires a non-zero kernel handle"
    );
    ensure!(
        (1..=DCOCO_WALK_MAX_K).contains(&args.top_k_shortlist),
        "dflash2_selector_walk_ngram_batched requires 1 <= top_k_shortlist <= {DCOCO_WALK_MAX_K}, got {}",
        args.top_k_shortlist,
    );
    ensure!(
        args.batch_size > 0 && args.gamma > 0 && args.rank > 0,
        "dflash2_selector_walk_ngram_batched requires positive batch_size/gamma/rank \
         (got batch_size={} gamma={} rank={})",
        args.batch_size,
        args.gamma,
        args.rank,
    );
    ensure!(
        args.margin_thresh.is_finite() && args.margin_thresh >= 0.0,
        "dflash2_selector_walk_ngram_batched requires finite non-negative margin_thresh, got {}",
        args.margin_thresh,
    );
    ensure!(
        args.max_bonus.is_finite() && (0.0..=0.75).contains(&args.max_bonus),
        "dflash2_selector_walk_ngram_batched requires max_bonus in 0.0..=0.75, got {}",
        args.max_bonus,
    );
    for (name, ptr) in [
        ("unary", args.unary),
        ("candidates", args.candidates),
        ("hidden", args.hidden),
        ("pred", args.pred),
        ("succ", args.succ),
        ("out_tokens", args.out_tokens),
        ("anchor_ids", args.anchor_ids),
    ] {
        ensure!(
            !ptr.is_null(),
            "dflash2_selector_walk_ngram_batched {name} pointer is null"
        );
    }

    let out_probs_ptr = optional_non_null(
        "dflash2_selector_walk_ngram_batched",
        "out_probs",
        args.out_probs,
    )?;
    let prev3_ptr = optional_non_null(
        "dflash2_selector_walk_ngram_batched",
        "prev3_tokens",
        args.prev3_tokens,
    )?;
    let tables_ptr = optional_non_null(
        "dflash2_selector_walk_ngram_batched",
        "ngram_tables",
        args.ngram_tables,
    )?;
    let temps_ptr = optional_non_null(
        "dflash2_selector_walk_ngram_batched",
        "temperatures",
        args.temperatures,
    )?;

    let top_k_i32 = i32::try_from(args.top_k_shortlist)
        .context("dflash2_selector_walk_ngram_batched top_k_shortlist exceeds i32")?;
    let gamma_i32 = i32::try_from(args.gamma)
        .context("dflash2_selector_walk_ngram_batched gamma exceeds i32")?;
    let rank_i32 =
        i32::try_from(args.rank).context("dflash2_selector_walk_ngram_batched rank exceeds i32")?;
    let batch_i32 = i32::try_from(args.batch_size)
        .context("dflash2_selector_walk_ngram_batched batch_size exceeds i32")?;
    let grid_x = u32::try_from(batch_i32)
        .context("dflash2_selector_walk_ngram_batched grid_x conversion failed")?;

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([SELECTOR_WARP_THREADS, 1, 1])
        .arg_ptr(args.unary)
        .arg_ptr(args.candidates)
        .arg_ptr(args.hidden)
        .arg_ptr(args.pred)
        .arg_ptr(args.succ)
        .arg_ptr(args.out_tokens)
        .arg_ptr(out_probs_ptr)
        .arg_ptr(args.anchor_ids)
        .arg_ptr(prev3_ptr)
        .arg_ptr(tables_ptr)
        .arg_ptr(temps_ptr)
        .arg_f32(args.margin_thresh)
        .arg_f32(args.max_bonus)
        .arg_i32(top_k_i32)
        .arg_i32(gamma_i32)
        .arg_i32(rank_i32)
        .arg_i32(batch_i32)
        .launch(stream)
}

fn optional_non_null(op: &str, name: &str, opt: Option<DevicePtr>) -> Result<DevicePtr> {
    match opt {
        Some(ptr) => {
            ensure!(
                !ptr.is_null(),
                "{op} {name} pointer is null (use None when absent)"
            );
            Ok(ptr)
        }
        None => Ok(DevicePtr::NULL),
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, KernelHandle};

    use super::*;

    #[test]
    fn dcoco_proj_fuse_validates_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x1000);
        let k = KernelHandle(11);
        let valid = DcocoProjFuseArgs {
            embed_tokens: p,
            hidden_states: p,
            anchor_ids: p,
            candidates: p,
            w_cand: p,
            w_ctx: p,
            w_fuse: p,
            fuse_ln_w: p,
            fuse_ln_b: p,
            w_pair: p,
            z_out: p,
            s_out: p,
            ln_eps: 1e-5,
            batch_size: 4,
            gamma: 7,
            top_k: 16,
            hidden_dim: 5120,
            vocab_size: 248_320,
        };

        assert!(dcoco_proj_fuse_bf16(&gpu, KernelHandle(0), &valid, 0).is_err());
        assert!(
            dcoco_proj_fuse_bf16(&gpu, k, &DcocoProjFuseArgs { top_k: 0, ..valid }, 0).is_err()
        );
        assert!(
            dcoco_proj_fuse_bf16(&gpu, k, &DcocoProjFuseArgs { top_k: 17, ..valid }, 0).is_err()
        );
        assert!(
            dcoco_proj_fuse_bf16(
                &gpu,
                k,
                &DcocoProjFuseArgs {
                    ln_eps: 0.0,
                    ..valid
                },
                0
            )
            .is_err()
        );
        assert!(
            dcoco_proj_fuse_bf16(
                &gpu,
                k,
                &DcocoProjFuseArgs {
                    w_pair: DevicePtr::NULL,
                    ..valid
                },
                0
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dcoco_proj_fuse_bf16(&gpu, k, &valid, 0).expect("valid dcoco_proj_fuse_bf16 launch");
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn dcoco_pair_mlp_validates_before_launch() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x2000);
        let k = KernelHandle(12);
        let valid = DcocoPairMlpArgs {
            s_pair: p,
            z_cand: p,
            unary: p,
            anchor_logits: None,
            mlp_w1: p,
            mlp_b1: p,
            mlp_w2: p,
            mlp_b2: p,
            out_scores: p,
            batch_size: 2,
            gamma: 7,
            top_k: 16,
        };

        assert!(dcoco_pair_mlp_bf16(&gpu, KernelHandle(0), &valid, 0).is_err());
        assert!(
            dcoco_pair_mlp_bf16(
                &gpu,
                k,
                &DcocoPairMlpArgs {
                    anchor_logits: Some(DevicePtr::NULL),
                    ..valid
                },
                0
            )
            .is_err()
        );
        assert!(dcoco_pair_mlp_bf16(&gpu, k, &DcocoPairMlpArgs { top_k: 17, ..valid }, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);

        dcoco_pair_mlp_bf16(&gpu, k, &valid, 0).expect("valid pair mlp launch with None anchor");
        dcoco_pair_mlp_bf16(
            &gpu,
            k,
            &DcocoPairMlpArgs {
                anchor_logits: Some(p),
                ..valid
            },
            0,
        )
        .expect("valid pair mlp launch with Some anchor");
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn dcoco_walk_validates_modes_and_optional_pointers() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x3000);
        let k = KernelHandle(13);
        let greedy = DcocoWalkArgs {
            transition_scores: p,
            candidates: p,
            out_tokens: p,
            out_max_prob: p,
            out_q_ids: None,
            out_q_probs: None,
            out_q_row_bases: None,
            seeds: None,
            step0_abs_pos: None,
            temperatures: None,
            draft_temp_scales: None,
            top_ks: None,
            top_ps: None,
            min_ps: None,
            walk_mode: DcocoWalkMode::Greedy,
            top_k_shortlist: 16,
            out_q_stride: 0,
            gamma: 7,
            batch_size: 2,
        };

        assert!(dcoco_walk_and_export_q(&gpu, KernelHandle(0), &greedy, 0).is_err());
        assert!(
            dcoco_walk_and_export_q(
                &gpu,
                k,
                &DcocoWalkArgs {
                    out_q_ids: Some(p),
                    ..greedy
                },
                0,
            )
            .is_err()
        );
        assert!(
            dcoco_walk_and_export_q(
                &gpu,
                k,
                &DcocoWalkArgs {
                    walk_mode: DcocoWalkMode::MaximalCoupling,
                    ..greedy
                },
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dcoco_walk_and_export_q(&gpu, k, &greedy, 0).expect("greedy walk launch");
        let coupled = DcocoWalkArgs {
            out_q_ids: Some(p),
            out_q_probs: Some(p),
            out_q_row_bases: Some(p),
            seeds: Some(p),
            step0_abs_pos: Some(p),
            temperatures: Some(p),
            draft_temp_scales: Some(p),
            top_ks: Some(p),
            top_ps: Some(p),
            min_ps: Some(p),
            walk_mode: DcocoWalkMode::MaximalCoupling,
            out_q_stride: 20,
            ..greedy
        };
        dcoco_walk_and_export_q(&gpu, k, &coupled, 0).expect("maximal coupling launch");
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn dflash2_ngram_walk_validates_cap_and_optional_pointers() {
        let gpu = MockGpuBackend::new();
        let p = DevicePtr(0x4000);
        let k = KernelHandle(14);
        let valid = Dflash2NgramWalkArgs {
            unary: p,
            candidates: p,
            hidden: p,
            pred: p,
            succ: p,
            out_tokens: p,
            out_probs: Some(p),
            anchor_ids: p,
            prev3_tokens: Some(p),
            ngram_tables: Some(p),
            temperatures: Some(p),
            margin_thresh: 1.0,
            max_bonus: 0.75,
            top_k_shortlist: 16,
            gamma: 7,
            rank: 256,
            batch_size: 4,
        };

        assert!(dflash2_selector_walk_ngram_batched(&gpu, KernelHandle(0), &valid, 0).is_err());
        assert!(
            dflash2_selector_walk_ngram_batched(
                &gpu,
                k,
                &Dflash2NgramWalkArgs {
                    max_bonus: 0.76,
                    ..valid
                },
                0,
            )
            .is_err()
        );
        assert!(
            dflash2_selector_walk_ngram_batched(
                &gpu,
                k,
                &Dflash2NgramWalkArgs {
                    ngram_tables: Some(DevicePtr::NULL),
                    ..valid
                },
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        dflash2_selector_walk_ngram_batched(&gpu, k, &valid, 0).expect("valid ngram walk launch");
        assert_eq!(gpu.launch_count(), 1);
    }
}
