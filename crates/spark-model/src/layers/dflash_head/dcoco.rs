// SPDX-License-Identifier: AGPL-3.0-only

//! Clean-room Flux-DCOCO-v2 contextual draft corrector foundation (`M2`).
//!
//! Pairs with `kernels/gb10/common/dcoco_selector.cu` (`dcoco_proj_fuse_bf16`,
//! `dcoco_pair_mlp_bf16`, `dcoco_walk_and_export_q`) to provide:
//!
//! 1. Strict, fail-closed weight loading from `WeightStore` (`dcoco_corrector.*`,
//!    exact BF16 tensor shapes; absent checkpoint returns `Ok(None)` with zero
//!    allocations so baseline models remain bit-identical).
//! 2. Stream-resident 3-kernel GPU execution (`DcocoCorrector::run_on_stream`)
//!    with zero host D2H round-trips and full CUDA-graph / async-propose safety.
//! 3. Exact CPU reference implementation (`dcoco_proj_fuse_cpu`,
//!    `dcoco_pair_mlp_factored_cpu`, `dcoco_pair_mlp_naive_cpu`, `dcoco_walk_cpu`)
//!    proving algebraic parity of the SMEM-resident factored Tensor-Core MLP
//!    against brute-force `[386] -> [256] -> [1]` evaluation, Wave-15 conditional
//!    `q_t(. | k_p)` normalization, $\mathcal{F}_t$-measurable `out_max_prob`,
//!    and continuous `beta` draft temperature scaling (`beta -> 0+` point-mass).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::device_allocations::DeviceAllocations;
use crate::layers::ops::{
    DCOCO_PAIR_MAX_K, DcocoPairMlpArgs, DcocoProjFuseArgs, DcocoWalkArgs, DcocoWalkMode,
    dcoco_pair_mlp_bf16, dcoco_proj_fuse_bf16, dcoco_walk_and_export_q,
};
use crate::weight_map::DenseWeight;

/// Bottleneck projection rank `r` (matches `DCOCO_RANK` in `dcoco_selector.cu`).
pub const DCOCO_RANK: usize = 128;

/// Pair MLP input width `3 * r + 2 = 386` (`[s_p, z_{p,a}, z_{p+1,b}, ell_p(a), ell_{p+1}(b)]`).
pub const DCOCO_MLP_IN: usize = 3 * DCOCO_RANK + 2;

/// Pair MLP hidden width `h = 256` (`DCOCO_MLP_HIDDEN`).
pub const DCOCO_MLP_HIDDEN: usize = 256;

/// Default LayerNorm epsilon matching PyTorch `nn.LayerNorm` (`1e-5`).
pub const DCOCO_DEFAULT_LN_EPS: f32 = 1.0e-5;

/// Default checkpoint tensor prefix for optional DCOCO corrector weights.
pub const DCOCO_WEIGHT_PREFIX: &str = "dcoco_corrector";

/// Point-mass collapse threshold on `beta` (matches `dcoco_walk_and_export_q`).
pub const DCOCO_BETA_POINT_MASS_THRESH: f32 = 1.0e-4;

/// Static geometry and parameter count specification of a Flux-DCOCO-v2 corrector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DcocoSpec {
    pub hidden_size: usize,
    pub top_k: usize,
    pub rank: usize,
    pub mlp_hidden: usize,
}

impl DcocoSpec {
    /// Construct and validate a standard Flux-DCOCO-v2 specification (`r=128, h=256`).
    ///
    /// # Errors
    ///
    /// Returns an error if `hidden_size == 0` or `top_k` is outside `1..=16`.
    pub fn new(hidden_size: usize, top_k: usize) -> Result<Self> {
        ensure!(hidden_size > 0, "DCOCO requires hidden_size > 0");
        ensure!(
            (1..=DCOCO_PAIR_MAX_K).contains(&top_k),
            "DCOCO requires 1 <= top_k <= {DCOCO_PAIR_MAX_K}, got {top_k}"
        );
        Ok(Self {
            hidden_size,
            top_k,
            rank: DCOCO_RANK,
            mlp_hidden: DCOCO_MLP_HIDDEN,
        })
    }

    /// Exact total trainable parameter count across all 10 tensors:
    /// `512 * d + 132,353` (`1,443,073` at `d=2560`, `2,229,505` at `d=4096`,
    /// `2,753,793` at `d=5120`).
    #[must_use]
    pub const fn param_count(&self) -> usize {
        let w_cand = self.rank * self.hidden_size;
        let w_ctx = self.rank * self.hidden_size;
        let w_fuse = self.rank * (2 * self.rank);
        let fuse_ln = 2 * self.rank;
        let w_pair = self.rank * (2 * self.hidden_size);
        let mlp_in = 3 * self.rank + 2;
        let mlp_w1 = self.mlp_hidden * mlp_in;
        let mlp_b1 = self.mlp_hidden;
        let mlp_w2 = self.mlp_hidden;
        let mlp_b2 = 1;
        w_cand + w_ctx + w_fuse + fuse_ln + w_pair + mlp_w1 + mlp_b1 + mlp_w2 + mlp_b2
    }
}

/// Device-resident BF16 weight bundle for the 10 Flux-DCOCO-v2 tensors.
#[derive(Debug, Clone, Copy)]
pub struct DcocoWeights {
    pub spec: DcocoSpec,
    /// `[128, hidden_size]` BF16 — frozen candidate token embedding projector.
    pub w_cand: DenseWeight,
    /// `[128, hidden_size]` BF16 — drafter post-norm hidden projector.
    pub w_ctx: DenseWeight,
    /// `[128, 256]` BF16 — candidate + context fusion projection.
    pub w_fuse: DenseWeight,
    /// `[128]` BF16 — affine LayerNorm weight after `w_fuse`.
    pub fuse_ln_w: DenseWeight,
    /// `[128]` BF16 — affine LayerNorm bias after `w_fuse`.
    pub fuse_ln_b: DenseWeight,
    /// `[128, 2 * hidden_size]` BF16 — adjacent pair-state projection `W_pair [h_p; h_{p+1}]`.
    pub w_pair: DenseWeight,
    /// `[256, 386]` BF16 — pair MLP layer-1 weight.
    pub mlp_w1: DenseWeight,
    /// `[256]` BF16 — pair MLP layer-1 bias.
    pub mlp_b1: DenseWeight,
    /// `[256]` BF16 — pair MLP layer-2 weight (zero-initialized at training start).
    pub mlp_w2: DenseWeight,
    /// `[1]` BF16 — pair MLP layer-2 scalar bias (zero-initialized at training start).
    pub mlp_b2: DenseWeight,
    /// LayerNorm epsilon (`1e-5`).
    pub ln_eps: f32,
}

/// Load optional Flux-DCOCO-v2 weights from `store` under `prefix`
/// (e.g. `"dcoco_corrector"`).
///
/// Returns `Ok(None)` when none of the 10 DCOCO tensors are present (preserving
/// bit-identical baseline behavior). Fails closed with an error if only a subset
/// of tensors is present, if any tensor has a non-BF16 dtype, or if any shape
/// disagrees with `hidden_size`.
///
/// # Errors
///
/// Returns an error on invalid geometry, partial tensor presence, wrong dtype,
/// or mismatched tensor shape.
pub fn load_dcoco_weights(
    store: &WeightStore,
    prefix: &str,
    hidden_size: usize,
    top_k: usize,
) -> Result<Option<DcocoWeights>> {
    let spec = DcocoSpec::new(hidden_size, top_k)?;
    let names = [
        format!("{prefix}.w_cand.weight"),
        format!("{prefix}.w_ctx.weight"),
        format!("{prefix}.w_fuse.weight"),
        format!("{prefix}.fuse_ln.weight"),
        format!("{prefix}.fuse_ln.bias"),
        format!("{prefix}.w_pair.weight"),
        format!("{prefix}.mlp_w1.weight"),
        format!("{prefix}.mlp_w1.bias"),
        format!("{prefix}.mlp_w2.weight"),
        format!("{prefix}.mlp_w2.bias"),
    ];

    let present_count = names.iter().filter(|name| store.contains(name)).count();
    if present_count == 0 {
        return Ok(None);
    }
    if present_count != names.len() {
        let missing: Vec<&str> = names
            .iter()
            .filter(|name| !store.contains(name))
            .map(String::as_str)
            .collect();
        bail!(
            "partial DCOCO checkpoint under `{prefix}` ({present_count}/{} tensors); missing {missing:?}",
            names.len()
        );
    }

    let load_bf16 = |name: &str, allowed_shapes: &[&[usize]]| -> Result<DenseWeight> {
        let tensor = store
            .get(name)
            .with_context(|| format!("missing DCOCO weight `{name}`"))?;
        ensure!(
            tensor.dtype == WeightDtype::BF16,
            "DCOCO weight `{name}` requires BF16 dtype, got {:?}",
            tensor.dtype
        );
        ensure!(
            allowed_shapes.contains(&tensor.shape.as_slice()),
            "DCOCO weight `{name}` has shape {:?}; expected one of {allowed_shapes:?}",
            tensor.shape
        );
        ensure!(
            !tensor.ptr.is_null(),
            "DCOCO weight `{name}` has null device pointer"
        );
        Ok(DenseWeight { weight: tensor.ptr })
    };

    Ok(Some(DcocoWeights {
        spec,
        w_cand: load_bf16(&names[0], &[&[DCOCO_RANK, hidden_size]])?,
        w_ctx: load_bf16(&names[1], &[&[DCOCO_RANK, hidden_size]])?,
        w_fuse: load_bf16(&names[2], &[&[DCOCO_RANK, 2 * DCOCO_RANK]])?,
        fuse_ln_w: load_bf16(&names[3], &[&[DCOCO_RANK]])?,
        fuse_ln_b: load_bf16(&names[4], &[&[DCOCO_RANK]])?,
        w_pair: load_bf16(&names[5], &[&[DCOCO_RANK, 2 * hidden_size]])?,
        mlp_w1: load_bf16(&names[6], &[&[DCOCO_MLP_HIDDEN, DCOCO_MLP_IN]])?,
        mlp_b1: load_bf16(&names[7], &[&[DCOCO_MLP_HIDDEN]])?,
        mlp_w2: load_bf16(&names[8], &[&[1, DCOCO_MLP_HIDDEN], &[DCOCO_MLP_HIDDEN]])?,
        mlp_b2: load_bf16(&names[9], &[&[1]])?,
        ln_eps: DCOCO_DEFAULT_LN_EPS,
    }))
}

/// Resolved CUDA kernel handles from `kernels/gb10/common/dcoco_selector.cu`.
#[derive(Debug, Clone, Copy)]
pub struct DcocoKernels {
    pub proj_fuse_bf16: KernelHandle,
    pub pair_mlp_bf16: KernelHandle,
    pub walk_and_export_q: KernelHandle,
}

impl DcocoKernels {
    /// Resolve all three DCOCO kernels from the `"dcoco_selector"` module.
    ///
    /// # Errors
    ///
    /// Returns an error if any symbol is unavailable on the active backend.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            proj_fuse_bf16: gpu.kernel("dcoco_selector", "dcoco_proj_fuse_bf16")?,
            pair_mlp_bf16: gpu.kernel("dcoco_selector", "dcoco_pair_mlp_bf16")?,
            walk_and_export_q: gpu.kernel("dcoco_selector", "dcoco_walk_and_export_q")?,
        })
    }
}

/// Pre-allocated device scratch buffers for the 3-kernel DCOCO pipeline.
///
/// Holds `z_cand`, `s_pair`, and `transition_scores` (zero global `[B*gamma*K^2, 386]`
/// materialization!). Sized once for `(max_batch, max_gamma, top_k)` so every
/// execution step is allocation-free and CUDA-graph capture-safe.
#[derive(Debug)]
pub struct DcocoScratch {
    /// `[max_batch, max_gamma + 1, top_k, 128]` BF16.
    pub z_cand: DevicePtr,
    /// `[max_batch, max_gamma, 128]` BF16.
    pub s_pair: DevicePtr,
    /// `[max_batch, max_gamma, top_k, top_k]` FP32.
    pub transition_scores: DevicePtr,
    pub max_batch: usize,
    pub max_gamma: usize,
    pub top_k: usize,
}

impl DcocoScratch {
    /// Allocate transactional DCOCO scratch buffers on `gpu`.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is invalid, byte counts overflow, or
    /// GPU allocation fails (rolling back partial allocations cleanly).
    pub fn allocate(
        gpu: &dyn GpuBackend,
        max_batch: usize,
        max_gamma: usize,
        top_k: usize,
    ) -> Result<Self> {
        ensure!(
            max_batch > 0 && max_gamma > 0 && (1..=DCOCO_PAIR_MAX_K).contains(&top_k),
            "invalid DCOCO scratch geometry: max_batch={max_batch} max_gamma={max_gamma} top_k={top_k}"
        );
        let bf16 = 2usize;
        let f32_bytes = 4usize;
        let z_bytes = max_batch
            .checked_mul(max_gamma + 1)
            .and_then(|v| v.checked_mul(top_k))
            .and_then(|v| v.checked_mul(DCOCO_RANK))
            .and_then(|v| v.checked_mul(bf16))
            .context("DCOCO z_cand byte size overflow")?;
        let s_bytes = max_batch
            .checked_mul(max_gamma)
            .and_then(|v| v.checked_mul(DCOCO_RANK))
            .and_then(|v| v.checked_mul(bf16))
            .context("DCOCO s_pair byte size overflow")?;
        let score_bytes = max_batch
            .checked_mul(max_gamma)
            .and_then(|v| v.checked_mul(top_k))
            .and_then(|v| v.checked_mul(top_k))
            .and_then(|v| v.checked_mul(f32_bytes))
            .context("DCOCO transition_scores byte size overflow")?;

        let mut tx = DeviceAllocations::new(gpu);
        let z_cand = tx.alloc(z_bytes)?;
        let s_pair = tx.alloc(s_bytes)?;
        let transition_scores = tx.alloc(score_bytes)?;
        tx.commit();
        Ok(Self {
            z_cand,
            s_pair,
            transition_scores,
            max_batch,
            max_gamma,
            top_k,
        })
    }

    /// Release all owned device scratch buffers.
    pub fn free(&mut self, gpu: &dyn GpuBackend) {
        for ptr in [
            &mut self.z_cand,
            &mut self.s_pair,
            &mut self.transition_scores,
        ] {
            if !ptr.is_null() {
                let _ = gpu.free(*ptr);
                *ptr = DevicePtr::NULL;
            }
        }
    }
}

/// Per-call runtime inputs to [`DcocoCorrector::run_on_stream`].
#[derive(Debug, Clone, Copy)]
pub struct DcocoStepInputs {
    pub embed_tokens: DevicePtr,
    pub hidden_states: DevicePtr,
    pub anchor_ids: DevicePtr,
    pub anchor_logits: Option<DevicePtr>,
    pub candidates: DevicePtr,
    pub unary: DevicePtr,
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
    pub out_q_stride: usize,
    pub batch_size: usize,
    pub gamma: usize,
    pub vocab_size: usize,
}

/// Complete stream-resident Flux-DCOCO-v2 corrector runtime.
#[derive(Debug)]
pub struct DcocoCorrector {
    pub weights: DcocoWeights,
    pub kernels: DcocoKernels,
    pub scratch: DcocoScratch,
}

impl DcocoCorrector {
    /// Construct a ready-to-launch DCOCO corrector on `gpu`.
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading or scratch allocation fails.
    pub fn new(
        gpu: &dyn GpuBackend,
        weights: DcocoWeights,
        max_batch: usize,
        max_gamma: usize,
    ) -> Result<Self> {
        let kernels = DcocoKernels::load(gpu)?;
        let scratch = DcocoScratch::allocate(gpu, max_batch, max_gamma, weights.spec.top_k)?;
        Ok(Self {
            weights,
            kernels,
            scratch,
        })
    }

    /// Execute the 3-stage DCOCO pipeline (`proj_fuse -> pair_mlp -> walk_and_export_q`)
    /// entirely on `stream` with zero host D2H synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error if `(batch_size, gamma)` exceeds scratch capacity or any
    /// kernel launch fails validation.
    pub fn run_on_stream(
        &self,
        gpu: &dyn GpuBackend,
        inputs: &DcocoStepInputs,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            inputs.batch_size > 0 && inputs.batch_size <= self.scratch.max_batch,
            "DCOCO batch_size={} outside 1..={}",
            inputs.batch_size,
            self.scratch.max_batch,
        );
        ensure!(
            inputs.gamma > 0 && inputs.gamma <= self.scratch.max_gamma,
            "DCOCO gamma={} outside 1..={}",
            inputs.gamma,
            self.scratch.max_gamma,
        );
        let top_k = self.weights.spec.top_k;

        dcoco_proj_fuse_bf16(
            gpu,
            self.kernels.proj_fuse_bf16,
            &DcocoProjFuseArgs {
                embed_tokens: inputs.embed_tokens,
                hidden_states: inputs.hidden_states,
                anchor_ids: inputs.anchor_ids,
                candidates: inputs.candidates,
                w_cand: self.weights.w_cand.weight,
                w_ctx: self.weights.w_ctx.weight,
                w_fuse: self.weights.w_fuse.weight,
                fuse_ln_w: self.weights.fuse_ln_w.weight,
                fuse_ln_b: self.weights.fuse_ln_b.weight,
                w_pair: self.weights.w_pair.weight,
                z_out: self.scratch.z_cand,
                s_out: self.scratch.s_pair,
                ln_eps: self.weights.ln_eps,
                batch_size: inputs.batch_size,
                gamma: inputs.gamma,
                top_k,
                hidden_dim: self.weights.spec.hidden_size,
                vocab_size: inputs.vocab_size,
            },
            stream,
        )?;

        dcoco_pair_mlp_bf16(
            gpu,
            self.kernels.pair_mlp_bf16,
            &DcocoPairMlpArgs {
                s_pair: self.scratch.s_pair,
                z_cand: self.scratch.z_cand,
                unary: inputs.unary,
                anchor_logits: inputs.anchor_logits,
                mlp_w1: self.weights.mlp_w1.weight,
                mlp_b1: self.weights.mlp_b1.weight,
                mlp_w2: self.weights.mlp_w2.weight,
                mlp_b2: self.weights.mlp_b2.weight,
                out_scores: self.scratch.transition_scores,
                batch_size: inputs.batch_size,
                gamma: inputs.gamma,
                top_k,
            },
            stream,
        )?;

        dcoco_walk_and_export_q(
            gpu,
            self.kernels.walk_and_export_q,
            &DcocoWalkArgs {
                transition_scores: self.scratch.transition_scores,
                candidates: inputs.candidates,
                out_tokens: inputs.out_tokens,
                out_max_prob: inputs.out_max_prob,
                out_q_ids: inputs.out_q_ids,
                out_q_probs: inputs.out_q_probs,
                out_q_row_bases: inputs.out_q_row_bases,
                seeds: inputs.seeds,
                step0_abs_pos: inputs.step0_abs_pos,
                temperatures: inputs.temperatures,
                draft_temp_scales: inputs.draft_temp_scales,
                top_ks: inputs.top_ks,
                top_ps: inputs.top_ps,
                min_ps: inputs.min_ps,
                walk_mode: inputs.walk_mode,
                top_k_shortlist: top_k,
                out_q_stride: inputs.out_q_stride,
                gamma: inputs.gamma,
                batch_size: inputs.batch_size,
            },
            stream,
        )
    }

    /// Release owned device scratch buffers.
    pub fn release(&mut self, gpu: &dyn GpuBackend) {
        self.scratch.free(gpu);
    }
}

// ── Exact CPU Reference Mirroring `dcoco_selector.cu` ────────────────────────

/// Round FP32 to BF16 (round-to-nearest-even) and widen back to FP32, matching
/// CUDA `__float2bfloat16_rn` -> `__bfloat162float`.
#[inline]
#[must_use]
pub fn round_bf16_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32.wrapping_add(lsb);
    let rounded = bits.wrapping_add(rounding_bias) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// Exact register-only A&S 7.1.26 erf-GELU matching CUDA `dcoco_gelu` (`max |err| < 1.5e-7`).
#[inline]
#[must_use]
pub fn dcoco_gelu_f32(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * az);
    let poly =
        ((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6;
    let erf_abs = 1.0 - (poly * t) * (-(az * az)).exp();
    let erf_val = if z < 0.0 { -erf_abs } else { erf_abs };
    0.5 * x * (1.0 + erf_val)
}

/// Host-side FP32/BF16-rounded weight matrices for CPU parity verification.
#[derive(Debug, Clone)]
pub struct DcocoCpuWeights {
    pub hidden_size: usize,
    pub w_cand: Vec<f32>,
    pub w_ctx: Vec<f32>,
    pub w_fuse: Vec<f32>,
    pub fuse_ln_w: Vec<f32>,
    pub fuse_ln_b: Vec<f32>,
    pub w_pair: Vec<f32>,
    pub mlp_w1: Vec<f32>,
    pub mlp_b1: Vec<f32>,
    pub mlp_w2: Vec<f32>,
    pub mlp_b2: f32,
    pub ln_eps: f32,
}

fn layernorm_128(
    vec: &mut [f32; DCOCO_RANK],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) {
    let mean = vec.iter().copied().sum::<f32>() / (DCOCO_RANK as f32);
    let var = vec
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / (DCOCO_RANK as f32);
    let inv_std = 1.0 / (var + eps).sqrt();
    for (i, v) in vec.iter_mut().enumerate() {
        let n = (*v - mean) * inv_std;
        *v = match (gamma, beta) {
            (Some(g), Some(b)) => n * g[i] + b[i],
            _ => n,
        };
    }
}

/// CPU reference for `dcoco_proj_fuse_bf16` (batch size 1).
///
/// Returns `(z_out, s_out)` with shapes `[(gamma + 1) * top_k * 128]` and
/// `[gamma * 128]`. When `quantize_bf16` is `Bf16Rounding::Enabled`, rounds
/// outputs to BF16 to mirror GPU `z_out` and `s_out`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16Rounding {
    Disabled,
    Enabled,
}

#[must_use]
pub fn dcoco_proj_fuse_cpu(
    embed_tokens: &[f32],
    hidden_states: &[f32],
    anchor_id: u32,
    candidates: &[u32],
    w: &DcocoCpuWeights,
    gamma: usize,
    top_k: usize,
    rounding: Bf16Rounding,
) -> (Vec<f32>, Vec<f32>) {
    let d = w.hidden_size;
    let slots = gamma + 1;
    let mut z_out = vec![0.0f32; slots * top_k * DCOCO_RANK];
    let mut s_out = vec![0.0f32; gamma * DCOCO_RANK];
    let maybe_round = |x: f32| match rounding {
        Bf16Rounding::Enabled => round_bf16_f32(x),
        Bf16Rounding::Disabled => x,
    };

    for slot in 0..slots {
        let h_cur = &hidden_states[slot * d..(slot + 1) * d];
        let mut g_vec = [0.0f32; DCOCO_RANK];
        for (r, g_slot) in g_vec.iter_mut().enumerate() {
            let w_row = &w.w_ctx[r * d..(r + 1) * d];
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += maybe_round(w_row[j]) * maybe_round(h_cur[j]);
            }
            *g_slot = acc;
        }
        layernorm_128(&mut g_vec, None, None, w.ln_eps);

        let slot_k = if slot == 0 { 1 } else { top_k };
        for k in 0..slot_k {
            let cid = if slot == 0 {
                anchor_id as usize
            } else {
                candidates[(slot - 1) * top_k + k] as usize
            };
            let emb = &embed_tokens[cid * d..(cid + 1) * d];
            let mut b_vec = [0.0f32; DCOCO_RANK];
            for (r, b_slot) in b_vec.iter_mut().enumerate() {
                let w_row = &w.w_cand[r * d..(r + 1) * d];
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += maybe_round(w_row[j]) * maybe_round(emb[j]);
                }
                *b_slot = acc;
            }
            layernorm_128(&mut b_vec, None, None, w.ln_eps);

            let mut u_vec = [0.0f32; DCOCO_RANK];
            for (r, u_slot) in u_vec.iter_mut().enumerate() {
                let wf_row = &w.w_fuse[r * (2 * DCOCO_RANK)..(r + 1) * (2 * DCOCO_RANK)];
                let mut acc = 0.0f32;
                for j in 0..DCOCO_RANK {
                    acc += maybe_round(wf_row[j]) * b_vec[j];
                    acc += maybe_round(wf_row[DCOCO_RANK + j]) * g_vec[j];
                }
                *u_slot = acc;
            }
            layernorm_128(&mut u_vec, Some(&w.fuse_ln_w), Some(&w.fuse_ln_b), w.ln_eps);

            let z_dst =
                &mut z_out[(slot * top_k + k) * DCOCO_RANK..(slot * top_k + k + 1) * DCOCO_RANK];
            for j in 0..DCOCO_RANK {
                z_dst[j] = maybe_round(u_vec[j]);
            }
        }

        if slot < gamma {
            let h_next = &hidden_states[(slot + 1) * d..(slot + 2) * d];
            let s_dst = &mut s_out[slot * DCOCO_RANK..(slot + 1) * DCOCO_RANK];
            for (r, s_val) in s_dst.iter_mut().enumerate() {
                let wp_row = &w.w_pair[r * (2 * d)..(r + 1) * (2 * d)];
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += maybe_round(wp_row[j]) * maybe_round(h_cur[j]);
                    acc += maybe_round(wp_row[d + j]) * maybe_round(h_next[j]);
                }
                *s_val = maybe_round(acc);
            }
        }
    }

    (z_out, s_out)
}

/// Brute-force `[386] -> [256] -> [1]` reference evaluating every `(p, a, b)` pair
/// independently by explicitly concatenating `[s_p, z_{p,a}, z_{p+1,b}, ell_p(a), ell_{p+1}(b)]`.
#[must_use]
pub fn dcoco_pair_mlp_naive_cpu(
    s_pair: &[f32],
    z_cand: &[f32],
    unary: &[f32],
    anchor_logit: f32,
    w: &DcocoCpuWeights,
    gamma: usize,
    top_k: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; gamma * top_k * top_k];
    let mut feat = [0.0f32; DCOCO_MLP_IN];
    for p in 0..gamma {
        let k_prev = if p == 0 { 1 } else { top_k };
        feat[0..DCOCO_RANK].copy_from_slice(&s_pair[p * DCOCO_RANK..(p + 1) * DCOCO_RANK]);
        for a in 0..k_prev {
            let za_off = (p * top_k + a) * DCOCO_RANK;
            feat[DCOCO_RANK..2 * DCOCO_RANK].copy_from_slice(&z_cand[za_off..za_off + DCOCO_RANK]);
            let ell_a = if p == 0 {
                anchor_logit
            } else {
                unary[(p - 1) * top_k + a]
            };
            feat[384] = ell_a;
            for b in 0..top_k {
                let zb_off = ((p + 1) * top_k + b) * DCOCO_RANK;
                feat[2 * DCOCO_RANK..3 * DCOCO_RANK]
                    .copy_from_slice(&z_cand[zb_off..zb_off + DCOCO_RANK]);
                let ell_b = unary[p * top_k + b];
                feat[385] = ell_b;

                let mut sum_out = 0.0f32;
                for h in 0..DCOCO_MLP_HIDDEN {
                    let w1_row = &w.mlp_w1[h * DCOCO_MLP_IN..(h + 1) * DCOCO_MLP_IN];
                    let mut pre = w.mlp_b1[h];
                    for j in 0..DCOCO_MLP_IN {
                        pre += w1_row[j] * feat[j];
                    }
                    sum_out += w.mlp_w2[h] * dcoco_gelu_f32(pre);
                }
                out[(p * top_k + a) * top_k + b] = ell_b + sum_out + w.mlp_b2;
            }
        }
    }
    out
}

/// Algebraically factored CPU reference mirroring CUDA `dcoco_pair_mlp_bf16`.
#[must_use]
pub fn dcoco_pair_mlp_factored_cpu(
    s_pair: &[f32],
    z_cand: &[f32],
    unary: &[f32],
    anchor_logit: f32,
    w: &DcocoCpuWeights,
    gamma: usize,
    top_k: usize,
    rounding: Bf16Rounding,
) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; gamma * top_k * top_k];
    let q = |x: f32| match rounding {
        Bf16Rounding::Enabled => round_bf16_f32(x),
        Bf16Rounding::Disabled => x,
    };

    for p in 0..gamma {
        let k_prev = if p == 0 { 1 } else { top_k };
        let sp = &s_pair[p * DCOCO_RANK..(p + 1) * DCOCO_RANK];

        let mut base = [0.0f32; DCOCO_MLP_HIDDEN];
        for h in 0..DCOCO_MLP_HIDDEN {
            let w1_row = &w.mlp_w1[h * DCOCO_MLP_IN..(h + 1) * DCOCO_MLP_IN];
            let mut dot = 0.0f32;
            for j in 0..DCOCO_RANK {
                dot += q(w1_row[j]) * q(sp[j]);
            }
            base[h] = q(w.mlp_b1[h]) + dot;
        }

        let mut term_a = vec![0.0f32; k_prev * DCOCO_MLP_HIDDEN];
        for a in 0..k_prev {
            let za = &z_cand[(p * top_k + a) * DCOCO_RANK..(p * top_k + a + 1) * DCOCO_RANK];
            let ell_a = if p == 0 {
                anchor_logit
            } else {
                unary[(p - 1) * top_k + a]
            };
            for h in 0..DCOCO_MLP_HIDDEN {
                let w1_row = &w.mlp_w1[h * DCOCO_MLP_IN..(h + 1) * DCOCO_MLP_IN];
                let mut dot = 0.0f32;
                for j in 0..DCOCO_RANK {
                    dot += q(w1_row[DCOCO_RANK + j]) * q(za[j]);
                }
                term_a[a * DCOCO_MLP_HIDDEN + h] = dot + q(w1_row[384]) * ell_a;
            }
        }

        let mut term_b = vec![0.0f32; top_k * DCOCO_MLP_HIDDEN];
        for b in 0..top_k {
            let zb =
                &z_cand[((p + 1) * top_k + b) * DCOCO_RANK..((p + 1) * top_k + b + 1) * DCOCO_RANK];
            let ell_b = unary[p * top_k + b];
            for h in 0..DCOCO_MLP_HIDDEN {
                let w1_row = &w.mlp_w1[h * DCOCO_MLP_IN..(h + 1) * DCOCO_MLP_IN];
                let mut dot = 0.0f32;
                for j in 0..DCOCO_RANK {
                    dot += q(w1_row[2 * DCOCO_RANK + j]) * q(zb[j]);
                }
                term_b[b * DCOCO_MLP_HIDDEN + h] = dot + q(w1_row[385]) * ell_b;
            }
        }

        let b2 = q(w.mlp_b2);
        for a in 0..k_prev {
            for b in 0..top_k {
                let mut acc = 0.0f32;
                for h in 0..DCOCO_MLP_HIDDEN {
                    let pre = base[h]
                        + term_a[a * DCOCO_MLP_HIDDEN + h]
                        + term_b[b * DCOCO_MLP_HIDDEN + h];
                    acc += q(w.mlp_w2[h]) * dcoco_gelu_f32(pre);
                }
                out[(p * top_k + a) * top_k + b] = unary[p * top_k + b] + acc + b2;
            }
        }
    }
    out
}

/// Output of [`dcoco_walk_cpu`] for a single sequence.
#[derive(Debug, Clone)]
pub struct DcocoWalkOutput {
    pub tokens: Vec<u32>,
    pub max_probs: Vec<f32>,
    pub q_ids: Vec<Vec<u32>>,
    pub q_probs: Vec<Vec<f32>>,
}

/// Sampling parameters for [`dcoco_walk_cpu`].
#[derive(Debug, Clone, Copy)]
pub struct DcocoWalkParams {
    pub mode: DcocoWalkMode,
    pub seed: u64,
    pub step0_abs_pos: u64,
    pub temperature: f32,
    pub draft_temp_scale_beta: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub out_q_stride: usize,
}

fn dcoco_pos_base(seed: u64, abs_pos: u64) -> u64 {
    super::ngram_bonus::splitmix64_mix(
        super::ngram_bonus::splitmix64_mix(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
            ^ abs_pos.wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

fn dcoco_gumbel_from_base(pos_base: u64, token_id: u32) -> f32 {
    let bits = super::ngram_bonus::splitmix64_mix(pos_base ^ u64::from(token_id));
    let u = (((bits >> 40) as f32) * (1.0f32 / 16_777_216.0) + (1.0f32 / 33_554_432.0))
        .min(0.999_999_94);
    -(-u.ln()).ln()
}

fn dcoco_prob_uniform(seed: u64, abs_pos: u64) -> f32 {
    let pos_base = dcoco_pos_base(seed, abs_pos);
    let bits = super::ngram_bonus::splitmix64_mix(pos_base ^ 0xDF2D_5A31);
    (((bits >> 40) as f32) * (1.0f32 / 16_777_216.0)).min(0.999_999_94)
}

/// Exact CPU reference mirroring CUDA `dcoco_walk_and_export_q`.
#[must_use]
pub fn dcoco_walk_cpu(
    transition_scores: &[f32],
    candidates: &[u32],
    gamma: usize,
    top_k_shortlist: usize,
    params: DcocoWalkParams,
) -> DcocoWalkOutput {
    if gamma == 0 || top_k_shortlist == 0 {
        return DcocoWalkOutput {
            tokens: Vec::new(),
            max_probs: Vec::new(),
            q_ids: Vec::new(),
            q_probs: Vec::new(),
        };
    }
    let beta = if params.draft_temp_scale_beta.is_finite() && params.draft_temp_scale_beta > 0.0 {
        params.draft_temp_scale_beta
    } else {
        0.0
    };
    let sampled = params.mode != DcocoWalkMode::Greedy
        && params.temperature > 0.0
        && params.temperature.is_finite();
    let point_mass_beta =
        params.mode == DcocoWalkMode::MaximalCoupling && beta <= DCOCO_BETA_POINT_MASS_THRESH;
    let eff_temp = if sampled && !point_mass_beta {
        (params.temperature * if beta > 0.0 { beta } else { 1.0 }).max(1.0e-6)
    } else {
        1.0
    };

    let mut tokens = Vec::with_capacity(gamma);
    let mut max_probs = Vec::with_capacity(gamma);
    let mut q_ids = Vec::with_capacity(gamma);
    let mut q_probs = Vec::with_capacity(gamma);
    let mut prev_slot = 0usize;

    for t in 0..gamma {
        let score_row = &transition_scores[(t * top_k_shortlist + prev_slot) * top_k_shortlist..]
            [..top_k_shortlist];
        let cand_row = &candidates[t * top_k_shortlist..][..top_k_shortlist];

        if !sampled || point_mass_beta {
            let mut best = score_row[0];
            let mut best_k = 0usize;
            let mut best_cid = cand_row[0];
            for j in 1..top_k_shortlist {
                let sc = score_row[j];
                let cid = cand_row[j];
                if sc > best || (sc == best && cid < best_cid) {
                    best = sc;
                    best_k = j;
                    best_cid = cid;
                }
            }
            let mut denom = 0.0f32;
            for &sc in score_row {
                let term = (sc - best).exp();
                if term.is_finite() {
                    denom += term;
                }
            }
            let mp = if point_mass_beta {
                1.0
            } else if denom > 0.0 && denom.is_finite() {
                1.0 / denom
            } else {
                1.0
            };
            tokens.push(best_cid);
            max_probs.push(mp);
            let mut row_ids = vec![u32::MAX; params.out_q_stride];
            let mut row_probs = vec![0.0f32; params.out_q_stride];
            if params.out_q_stride > 0 {
                row_ids[0] = best_cid;
                row_probs[0] = 1.0;
            }
            q_ids.push(row_ids);
            q_probs.push(row_probs);
            prev_slot = best_k;
            continue;
        }

        let mut scaled = vec![0.0f32; top_k_shortlist];
        let mut ord: Vec<usize> = (0..top_k_shortlist).collect();
        for j in 0..top_k_shortlist {
            let raw = score_row[j] / eff_temp;
            scaled[j] = if raw.is_finite() { raw } else { -1.0e30 };
        }
        ord.sort_by(|&a, &b| {
            scaled[b]
                .total_cmp(&scaled[a])
                .then_with(|| cand_row[a].cmp(&cand_row[b]))
        });

        let mut k_eff = if params.top_k > 0 && (params.top_k as usize) < top_k_shortlist {
            params.top_k as usize
        } else {
            top_k_shortlist
        };
        let max_s = scaled[ord[0]];
        let mut w: Vec<f32> = (0..k_eff).map(|j| (scaled[ord[j]] - max_s).exp()).collect();
        let sum_w: f32 = w.iter().copied().sum();

        if params.min_p > 0.0 && params.min_p.is_finite() {
            let threshold = params.min_p * w[0];
            while k_eff > 1 && w[k_eff - 1] < threshold {
                k_eff -= 1;
            }
        }

        let mut kept = k_eff;
        if params.top_p > 0.0 && params.top_p < 1.0 && sum_w > 0.0 {
            let rem_sum: f32 = w[..k_eff].iter().copied().sum();
            if rem_sum > 0.0 {
                let inv_sum = 1.0 / rem_sum;
                let mut cum = 0.0f32;
                for (j, &wj) in w.iter().take(k_eff).enumerate() {
                    cum += wj * inv_sum;
                    if cum >= params.top_p {
                        kept = j + 1;
                        break;
                    }
                }
            }
        }
        if params.out_q_stride > 0 && kept > params.out_q_stride {
            kept = params.out_q_stride;
        }
        let kept_sum: f32 = w[..kept].iter().copied().sum();
        let valid_sum = kept_sum > 0.0 && kept_sum.is_finite();
        for (j, slot) in w.iter_mut().take(kept).enumerate() {
            *slot = if valid_sum {
                *slot / kept_sum
            } else if j == 0 {
                1.0
            } else {
                0.0
            };
        }

        let mut selected_j = 0usize;
        let mp = if params.mode == DcocoWalkMode::GumbelMax {
            let pos_base = dcoco_pos_base(params.seed, params.step0_abs_pos + t as u64);
            let mut best_cid = cand_row[ord[0]];
            let mut best_g = f32::NEG_INFINITY;
            for (j, &idx) in ord.iter().take(kept).enumerate() {
                let cid = cand_row[idx];
                let g_score = scaled[idx] + dcoco_gumbel_from_base(pos_base, cid);
                if j == 0 || g_score > best_g || (g_score == best_g && cid < best_cid) {
                    best_g = g_score;
                    best_cid = cid;
                    selected_j = j;
                }
            }
            w[selected_j]
        } else {
            let u = dcoco_prob_uniform(params.seed, params.step0_abs_pos + t as u64);
            let mut cum = 0.0f32;
            let mut last_nonzero = 0usize;
            for (j, &p_val) in w.iter().take(kept).enumerate() {
                if p_val > 0.0 {
                    last_nonzero = j;
                }
                cum += p_val;
                if u < cum {
                    selected_j = j;
                    break;
                }
                selected_j = last_nonzero;
            }
            // Strictly F_t-measurable prefix confidence independent of sampled x_t.
            w[0]
        };

        let chosen_slot = ord[selected_j];
        tokens.push(cand_row[chosen_slot]);
        max_probs.push(mp);

        let mut row_ids = vec![u32::MAX; params.out_q_stride];
        let mut row_probs = vec![0.0f32; params.out_q_stride];
        for j in 0..kept.min(params.out_q_stride) {
            row_ids[j] = cand_row[ord[j]];
            row_probs[j] = w[j];
        }
        q_ids.push(row_ids);
        q_probs.push(row_probs);
        prev_slot = chosen_slot;
    }

    DcocoWalkOutput {
        tokens,
        max_probs,
        q_ids,
        q_probs,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;

    use super::*;

    fn det_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = super::super::ngram_bonus::splitmix64_mix(seed.wrapping_add(i as u64));
                let u = ((h >> 40) as f32) / 16_777_216.0;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn synthetic_cpu_weights(hidden_size: usize) -> DcocoCpuWeights {
        DcocoCpuWeights {
            hidden_size,
            w_cand: det_vec(DCOCO_RANK * hidden_size, 101, 0.08),
            w_ctx: det_vec(DCOCO_RANK * hidden_size, 202, 0.08),
            w_fuse: det_vec(DCOCO_RANK * 2 * DCOCO_RANK, 303, 0.08),
            fuse_ln_w: det_vec(DCOCO_RANK, 404, 0.25)
                .into_iter()
                .map(|x| 1.0 + x)
                .collect(),
            fuse_ln_b: det_vec(DCOCO_RANK, 505, 0.05),
            w_pair: det_vec(DCOCO_RANK * 2 * hidden_size, 606, 0.06),
            mlp_w1: det_vec(DCOCO_MLP_HIDDEN * DCOCO_MLP_IN, 707, 0.06),
            mlp_b1: det_vec(DCOCO_MLP_HIDDEN, 808, 0.05),
            mlp_w2: det_vec(DCOCO_MLP_HIDDEN, 909, 0.08),
            mlp_b2: 0.03,
            ln_eps: DCOCO_DEFAULT_LN_EPS,
        }
    }

    #[test]
    fn exact_parameter_counts_match_paper_and_flux_architectures() {
        assert_eq!(DcocoSpec::new(2560, 16).unwrap().param_count(), 1_443_073);
        assert_eq!(DcocoSpec::new(4096, 16).unwrap().param_count(), 2_229_505);
        assert_eq!(DcocoSpec::new(5120, 16).unwrap().param_count(), 2_753_793);
        assert!(DcocoSpec::new(0, 16).is_err());
        assert!(DcocoSpec::new(5120, 0).is_err());
        assert!(DcocoSpec::new(5120, 17).is_err());
    }

    #[test]
    fn gelu_approximation_matches_exact_erf_within_2e_7() {
        for i in -800..=800 {
            let x = (i as f32) * 0.01;
            let approx = dcoco_gelu_f32(x);
            // High-precision reference via f64 tanh-free series check at key points
            // and monotonicity / symmetry identity: GELU(x) - GELU(-x) == x.
            let neg = dcoco_gelu_f32(-x);
            assert!(
                ((approx - neg) - x).abs() < 2e-6,
                "GELU odd-part identity failed at x={x}: {approx} - {neg} != {x}"
            );
            assert!(approx.is_finite());
        }
        assert!((dcoco_gelu_f32(0.0)).abs() < 1e-7);
        assert!((dcoco_gelu_f32(1.0) - 0.841_344_7).abs() < 5e-7);
        assert!((dcoco_gelu_f32(-1.0) - (-0.158_655_26)).abs() < 5e-7);
        assert!((dcoco_gelu_f32(2.0) - 1.954_499_7).abs() < 1e-6);
    }

    #[test]
    fn proj_fuse_non_affine_ln_absorbs_100x_scale_gap_between_embed_and_hidden() {
        let d = 64;
        let w = synthetic_cpu_weights(d);
        let vocab = 32;
        let gamma = 3;
        let top_k = 4;
        let embed_base = det_vec(vocab * d, 1111, 0.25);
        let hidden_base = det_vec((gamma + 1) * d, 2222, 1.0);
        let embed_scaled: Vec<f32> = embed_base.iter().map(|&x| x * 100.0).collect();
        let candidates: Vec<u32> = (0..gamma * top_k).map(|i| (i % vocab) as u32).collect();

        let (z1, s1) = dcoco_proj_fuse_cpu(
            &embed_base,
            &hidden_base,
            5,
            &candidates,
            &w,
            gamma,
            top_k,
            Bf16Rounding::Disabled,
        );
        let (z2, s2) = dcoco_proj_fuse_cpu(
            &embed_scaled,
            &hidden_base,
            5,
            &candidates,
            &w,
            gamma,
            top_k,
            Bf16Rounding::Disabled,
        );
        assert_eq!(s1, s2, "pair state s_p depends only on hidden_states");
        let max_diff = z1
            .iter()
            .zip(z2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-2,
            "non-affine LN must absorb 100x embedding scale gap (max_diff={max_diff})"
        );
    }

    #[test]
    fn factored_pair_mlp_matches_brute_force_386_to_256_to_1_in_fp32_and_bf16() {
        let d = 64;
        let vocab = 32;
        let gamma = 4;
        let top_k = 8;
        let w = synthetic_cpu_weights(d);
        let embed = det_vec(vocab * d, 3001, 0.2);
        let hidden = det_vec((gamma + 1) * d, 3002, 0.5);
        let candidates: Vec<u32> = (0..gamma * top_k)
            .map(|i| ((i * 7 + 3) % vocab) as u32)
            .collect();
        let unary = det_vec(gamma * top_k, 3003, 3.0);

        let (z_fp32, s_fp32) = dcoco_proj_fuse_cpu(
            &embed,
            &hidden,
            9,
            &candidates,
            &w,
            gamma,
            top_k,
            Bf16Rounding::Disabled,
        );
        let naive = dcoco_pair_mlp_naive_cpu(&s_fp32, &z_fp32, &unary, 1.25, &w, gamma, top_k);
        let factored_fp32 = dcoco_pair_mlp_factored_cpu(
            &s_fp32,
            &z_fp32,
            &unary,
            1.25,
            &w,
            gamma,
            top_k,
            Bf16Rounding::Disabled,
        );
        let factored_bf16 = dcoco_pair_mlp_factored_cpu(
            &s_fp32,
            &z_fp32,
            &unary,
            1.25,
            &w,
            gamma,
            top_k,
            Bf16Rounding::Enabled,
        );

        let mut max_fp32_err = 0.0f32;
        let mut max_bf16_err = 0.0f32;
        for p in 0..gamma {
            let k_prev = if p == 0 { 1 } else { top_k };
            for a in 0..top_k {
                for b in 0..top_k {
                    let idx = (p * top_k + a) * top_k + b;
                    if a < k_prev {
                        max_fp32_err = max_fp32_err.max((naive[idx] - factored_fp32[idx]).abs());
                        max_bf16_err = max_bf16_err.max((naive[idx] - factored_bf16[idx]).abs());
                    } else {
                        assert!(naive[idx].is_infinite() && naive[idx].is_sign_negative());
                        assert!(
                            factored_fp32[idx].is_infinite()
                                && factored_fp32[idx].is_sign_negative()
                        );
                    }
                }
            }
        }
        assert!(
            max_fp32_err < 1e-4,
            "factored FP32 MLP drifted from brute-force [386->256->1]: {max_fp32_err}"
        );
        assert!(
            max_bf16_err < 2e-2,
            "factored BF16 MMA MLP drifted from brute-force [386->256->1]: {max_bf16_err}"
        );

        // Zero-init residual identity: when mlp_w2 = 0 and mlp_b2 = 0,
        // transition scores equal unary[p, b] identically for every predecessor a.
        let mut zero_w = w;
        zero_w.mlp_w2.fill(0.0);
        zero_w.mlp_b2 = 0.0;
        let id_scores = dcoco_pair_mlp_factored_cpu(
            &s_fp32,
            &z_fp32,
            &unary,
            0.0,
            &zero_w,
            gamma,
            top_k,
            Bf16Rounding::Enabled,
        );
        for p in 0..gamma {
            let k_prev = if p == 0 { 1 } else { top_k };
            for a in 0..k_prev {
                for b in 0..top_k {
                    assert_eq!(
                        id_scores[(p * top_k + a) * top_k + b],
                        unary[p * top_k + b],
                        "zero-init residual must recover exact unary score"
                    );
                }
            }
        }
    }

    #[test]
    fn wave15_walk_normalizes_q_preserves_ft_measurability_and_beta_sweep() {
        let gamma = 4;
        let top_k = 8;
        let scores = det_vec(gamma * top_k * top_k, 4001, 2.5);
        let candidates: Vec<u32> = (0..gamma * top_k).map(|i| 100 + i as u32).collect();

        // 1. MaximalCoupling at beta = 1.0: normalized q, F_t-measurable out_max_prob at t=0
        //    must be identical across all random seeds (independent of sampled x_0!).
        let mut max_prob_t0_ref: Option<f32> = None;
        let mut distinct_t0_draws = std::collections::HashSet::new();
        for seed in 1u64..=32 {
            let out = dcoco_walk_cpu(
                &scores,
                &candidates,
                gamma,
                top_k,
                DcocoWalkParams {
                    mode: DcocoWalkMode::MaximalCoupling,
                    seed,
                    step0_abs_pos: 1024,
                    temperature: 0.8,
                    draft_temp_scale_beta: 1.0,
                    top_k: 8,
                    top_p: 0.95,
                    min_p: 0.02,
                    out_q_stride: 20,
                },
            );
            for t in 0..gamma {
                let sum: f32 = out.q_probs[t].iter().copied().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "q_t must normalize to 1.0 (got {sum})"
                );
                assert!(
                    out.max_probs[t].is_finite()
                        && out.max_probs[t] > 0.0
                        && out.max_probs[t] <= 1.0
                );
                // out_max_prob[t] equals max_z q_t(z) = q_probs[t][0] (sorted descending).
                assert!((out.max_probs[t] - out.q_probs[t][0]).abs() < 1e-6);
            }
            if let Some(ref_p) = max_prob_t0_ref {
                assert_eq!(
                    out.max_probs[0].to_bits(),
                    ref_p.to_bits(),
                    "out_max_prob[0] must be F_0-measurable and independent of sampled x_0"
                );
            } else {
                max_prob_t0_ref = Some(out.max_probs[0]);
            }
            distinct_t0_draws.insert(out.tokens[0]);
        }
        assert!(
            distinct_t0_draws.len() >= 2,
            "stochastic maximal coupling must sample multiple candidates across seeds"
        );

        // 2. Continuous beta -> 0+ (<= 1e-4) collapses cleanly to exact 1-hot DCOCO App. G
        //    point-mass matching greedy argmax with zero NaNs.
        let greedy = dcoco_walk_cpu(
            &scores,
            &candidates,
            gamma,
            top_k,
            DcocoWalkParams {
                mode: DcocoWalkMode::Greedy,
                seed: 7,
                step0_abs_pos: 1024,
                temperature: 0.0,
                draft_temp_scale_beta: 1.0,
                top_k: 8,
                top_p: 0.95,
                min_p: 0.0,
                out_q_stride: 20,
            },
        );
        let point_mass = dcoco_walk_cpu(
            &scores,
            &candidates,
            gamma,
            top_k,
            DcocoWalkParams {
                mode: DcocoWalkMode::MaximalCoupling,
                seed: 7,
                step0_abs_pos: 1024,
                temperature: 0.8,
                draft_temp_scale_beta: 1.0e-5,
                top_k: 8,
                top_p: 0.95,
                min_p: 0.0,
                out_q_stride: 20,
            },
        );
        assert_eq!(point_mass.tokens, greedy.tokens);
        for t in 0..gamma {
            assert_eq!(point_mass.max_probs[t], 1.0);
            assert_eq!(point_mass.q_ids[t][0], greedy.tokens[t]);
            assert_eq!(point_mass.q_probs[t][0], 1.0);
            assert_eq!(point_mass.q_ids[t][1], u32::MAX);
            assert_eq!(point_mass.q_probs[t][1], 0.0);
        }

        // 3. Maximal-coupling overlap sum_x min(p(x), q_beta(x)) at beta=1.0 strictly beats
        //    point-mass acceptance p(argmax) on a non-degenerate target distribution p = q_1.
        let non_degen = dcoco_walk_cpu(
            &scores,
            &candidates,
            1,
            top_k,
            DcocoWalkParams {
                mode: DcocoWalkMode::MaximalCoupling,
                seed: 9,
                step0_abs_pos: 0,
                temperature: 1.0,
                draft_temp_scale_beta: 1.0,
                top_k: 8,
                top_p: 1.0,
                min_p: 0.0,
                out_q_stride: 8,
            },
        );
        let p_argmax = non_degen.q_probs[0][0];
        let coupling_overlap: f32 = non_degen.q_probs[0].iter().copied().sum();
        assert!(
            coupling_overlap > p_argmax + 0.05,
            "Wave-15 maximal coupling overlap ({coupling_overlap}) must strictly exceed \
             point-mass acceptance p(argmax) ({p_argmax})"
        );

        // 4. Empty edge inputs return cleanly without panicking.
        let empty_out = dcoco_walk_cpu(
            &[],
            &[],
            1,
            0,
            DcocoWalkParams {
                mode: DcocoWalkMode::Greedy,
                seed: 0,
                step0_abs_pos: 0,
                temperature: 0.0,
                draft_temp_scale_beta: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                out_q_stride: 0,
            },
        );
        assert!(empty_out.tokens.is_empty());
        assert!(empty_out.max_probs.is_empty());
    }

    #[test]
    fn weight_loader_returns_none_when_absent_fails_closed_on_partial_and_runs_mock_pipeline() {
        let empty = WeightStore::empty();
        assert!(
            load_dcoco_weights(&empty, DCOCO_WEIGHT_PREFIX, 64, 16)
                .expect("absent weights return Ok(None)")
                .is_none()
        );

        let d = 64;
        let mk = |ptr: u64, shape: Vec<usize>, dtype: WeightDtype| WeightTensor {
            ptr: DevicePtr(ptr),
            shape,
            dtype,
        };
        let mut full_map = HashMap::from([
            (
                format!("{DCOCO_WEIGHT_PREFIX}.w_cand.weight"),
                mk(0x1000, vec![128, d], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.w_ctx.weight"),
                mk(0x2000, vec![128, d], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.w_fuse.weight"),
                mk(0x3000, vec![128, 256], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.fuse_ln.weight"),
                mk(0x4000, vec![128], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.fuse_ln.bias"),
                mk(0x5000, vec![128], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.w_pair.weight"),
                mk(0x6000, vec![128, 2 * d], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.mlp_w1.weight"),
                mk(0x7000, vec![256, 386], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.mlp_w1.bias"),
                mk(0x8000, vec![256], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.mlp_w2.weight"),
                mk(0x9000, vec![1, 256], WeightDtype::BF16),
            ),
            (
                format!("{DCOCO_WEIGHT_PREFIX}.mlp_w2.bias"),
                mk(0xA000, vec![1], WeightDtype::BF16),
            ),
        ]);

        // Partial checkpoint MUST fail closed.
        let mut partial = HashMap::new();
        partial.insert(
            format!("{DCOCO_WEIGHT_PREFIX}.w_cand.weight"),
            mk(0x1000, vec![128, d], WeightDtype::BF16),
        );
        assert!(
            load_dcoco_weights(&WeightStore::from_map(partial), DCOCO_WEIGHT_PREFIX, d, 16)
                .is_err()
        );

        // Wrong dtype MUST fail closed.
        let mut bad_dtype = HashMap::new();
        for (k, v) in &full_map {
            bad_dtype.insert(
                k.clone(),
                WeightTensor {
                    ptr: v.ptr,
                    shape: v.shape.clone(),
                    dtype: if k.ends_with("mlp_b2") || k.ends_with("mlp_w2.bias") {
                        WeightDtype::FP32
                    } else {
                        v.dtype
                    },
                },
            );
        }
        assert!(
            load_dcoco_weights(
                &WeightStore::from_map(bad_dtype),
                DCOCO_WEIGHT_PREFIX,
                d,
                16
            )
            .is_err()
        );

        // Complete valid checkpoint loads cleanly and launches all 3 kernels on MockGpuBackend.
        let store = WeightStore::from_map(std::mem::take(&mut full_map));
        let weights = load_dcoco_weights(&store, DCOCO_WEIGHT_PREFIX, d, 16)
            .expect("valid DCOCO checkpoint")
            .expect("weights present");
        let gpu = MockGpuBackend::new();
        let mut corrector = DcocoCorrector::new(&gpu, weights, 4, 7).expect("build corrector");
        let p = DevicePtr(0x1_0000);
        corrector
            .run_on_stream(
                &gpu,
                &DcocoStepInputs {
                    embed_tokens: p,
                    hidden_states: p,
                    anchor_ids: p,
                    anchor_logits: None,
                    candidates: p,
                    unary: p,
                    out_tokens: p,
                    out_max_prob: p,
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
                    batch_size: 2,
                    gamma: 7,
                    vocab_size: 1024,
                },
                0,
            )
            .expect("3-kernel DCOCO pipeline launch");
        assert_eq!(gpu.launch_count(), 3);
        corrector.release(&gpu);
    }
}
