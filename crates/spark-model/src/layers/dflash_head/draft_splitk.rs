// SPDX-License-Identifier: AGPL-3.0-only

//! Split-K dispatch for the DFlash drafter's own NVFP4 projections.
//!
//! # Why the drafter needs this
//!
//! Under the production flag set (`ATLAS_DFLASH_NOISE_ONLY=1` +
//! `ATLAS_DFLASH_ATTN_KGAMMA=1` + `ATLAS_DFLASH_FFN_KGAMMA=1`) every drafter
//! projection runs at `M = γ+1 = 13` rows through `w4a16_gemm_t_m32_n64`,
//! whose grid is `(ceil(N/64), ceil(M/32), 1)`. At M ≤ 32 that is exactly
//! `ceil(N/64)` CTAs — the N dimension alone decides occupancy, and the whole
//! machine has to be filled from that one dimension. (An earlier draft of this
//! comment asserted "110 SMs"; nothing in the tree had ever queried the
//! device, so treat that as unverified — `GpuBackend::sm_count()` now reports
//! the real number, and the threshold below is anchored on measurement rather
//! than on any SM literal.)
//!
//! Measured achieved bandwidth per drafter kernel (6 layers, γ_eff=12, from
//! an internal kernel profile, not published), against the CTA count its N
//! implies:
//!
//! | kernel      |    N  | CTAs | weight bytes | measured | achieved  |
//! |-------------|-------|------|--------------|----------|-----------|
//! | kv_noise    |  1024 |   16 |    35.4 MB   | 1.16 ms  |  30 GB/s  |
//! | q_proj      |  4096 |   64 |    70.8 MB   | 0.93 ms  |  76 GB/s  |
//! | o_proj      |  5120 |   80 |    70.8 MB   | 0.78 ms  |  91 GB/s  |
//! | down_proj   |  5120 |   80 |   301.0 MB   | 3.10 ms  |  97 GB/s  |
//! | gate_up     | 17408 |  272 |   602.0 MB   | 3.10 ms  | 194 GB/s  |
//!
//! Achieved bandwidth is a monotone function of CTA count and saturates only
//! once the grid exceeds the SM count by ~2.5×. `gate_up` — the one drafter
//! GEMM with a wide N — already runs at 84% of the 232 GB/s achievable floor
//! and has nothing left to give. Every other drafter GEMM is occupancy-bound,
//! not bandwidth-bound, and that is the whole of the propose path's gap to its
//! memory-traffic floor.
//!
//! This is the identical diagnosis (and identical remedy) already applied on
//! the target side: see the `w4a16_gemm_n64_m32_splitk` doc comment in
//! `layers/ops/gemm_dense.rs`, which records the K=17 verify `down_proj`
//! running at "~91 GB/s vs gate/up ~163 GB/s on the same-size weight" and
//! fixes it by multiplying the CTA count through `gridDim.z`. Production
//! already sets `ATLAS_FFN_DOWN_SPLITK`, `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` and
//! `ATLAS_PAGED_DECODE_SPLITK`; the drafter's own layers are the last consumer
//! of the m32_n64 kernel that was never given the same treatment.
//!
//! # Numerical contract
//!
//! Split-K is *reassociated*, not bit-identical: the K-loop is sliced across
//! `gridDim.z`, each slice accumulates in FP32 into a workspace band, and
//! `reduce_splitk_f32_to_bf16` sums the bands. Every partial is FP32 (no
//! precision is dropped) but the summation ORDER changes, so results can
//! differ in the last ULP of the BF16 result.
//!
//! For the drafter this cannot corrupt output: speculation is verified by a
//! strict argmax match against the target, so a drafter logit that moves by an
//! ULP can only change WHICH token is proposed, never which is committed. The
//! honest risk is therefore acceptance, not correctness — hence the SM-gated
//! default below (OFF on `<= 64`-SM GPUs, `8` slices on `> 64`-SM GPUs) and
//! the A/B requirement.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

use super::BlockDiffusionDraftHead;

/// SM-count threshold above which `ATLAS_DFLASH_DRAFT_SPLITK` defaults to
/// enabled when unset (`<= 64` keeps 48-SM GB10 off by default).
const AUTO_DRAFT_SPLITK_SM_THRESHOLD: u32 = 64;
/// Default split-K slice budget when auto-enabled on >64-SM discrete GPUs,
/// matching the `8 * 64 * hidden_size.max(q_dim) * 4` workspace allocation in
/// `from_weights.rs`.
const AUTO_DRAFT_SPLITK_BUDGET: u32 = 8;

#[inline]
#[must_use]
fn resolve_draft_splitk(raw: Option<&str>, sm_count: u32) -> u32 {
    match raw {
        Some(s) => s
            .trim()
            .parse::<u32>()
            .ok()
            .map(|v| if v < 2 { 0 } else { v.min(8) })
            .unwrap_or(0),
        None if sm_count > AUTO_DRAFT_SPLITK_SM_THRESHOLD => AUTO_DRAFT_SPLITK_BUDGET,
        None => 0,
    }
}

/// `ATLAS_DFLASH_DRAFT_SPLITK` — number of K-slices for the drafter's own
/// occupancy-starved projections. `0` (the default on `<= 64`-SM GPUs like GB10)
/// disables split-K entirely and every dispatch below is byte-for-byte the
/// pre-existing kernel choice. On `> 64`-SM GPUs (`SM120` / `SM103`), defaults
/// to `8` when unset unless explicitly set to `0`. Values `1` and below are
/// treated as off; the kernel's workspace is sized for at most 8 slices,
/// matching `ffn_down_splitk`'s clamp.
pub(super) fn draft_splitk() -> u32 {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_DFLASH_DRAFT_SPLITK").ok());
    let v = resolve_draft_splitk(raw.as_deref(), atlas_core::device::active_sm_count());
    if v > 0 {
        LOGGED.call_once(|| {
            tracing::info!(
                "DFlash drafter split-K ENABLED (ATLAS_DFLASH_DRAFT_SPLITK={v}): \
                 q/kv/o/down projections slice K across gridDim.z. Reassociated \
                 FP32 partials — drafts may shift, commits cannot (target argmax \
                 verifies)."
            );
        });
    }
    v
}

/// Baseline CTA floor below which a grid is considered occupancy-starved on
/// 48-SM GB10 (256 CTAs sits just under `gate_up`'s saturated 272 CTAs).
/// On wider GPUs (`SM120` / `SM103`), [`starved_cta_limit_for_sm`] scales this
/// floor to `max(sm_count * 4, 256)` so 110–188 SM devices saturate GDDR7/HBM3e.
const STARVED_CTA_LIMIT: u32 = 256;

#[inline]
#[must_use]
fn starved_cta_limit_for_sm(sm_count: u32) -> u32 {
    sm_count.saturating_mul(4).max(STARVED_CTA_LIMIT)
}

/// Report the device's real SM count against the active CTA starvation floor,
/// once per process.
fn log_sm_count_assumption(gpu: &dyn GpuBackend) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| match gpu.sm_count() {
        Some(sms) => {
            let limit = starved_cta_limit_for_sm(sms);
            let ctas_per_sm = f64::from(limit) / f64::from(sms.max(1));
            tracing::info!(
                "drafter split-K: device reports {sms} SMs; starved_cta_limit={limit} \
                 ({ctas_per_sm:.1} CTAs/SM, floor={STARVED_CTA_LIMIT})."
            );
        }
        None => tracing::info!(
            "drafter split-K: backend cannot report an SM count; \
             using starved_cta_limit={STARVED_CTA_LIMIT}."
        ),
    });
}

/// CTAs the single-slice m32_n64 kernel would field for output width `n` at
/// `m <= 32` (one M-tile).
#[cfg(test)]
fn cta_count(n: u32) -> u32 {
    n.div_ceil(64)
}

/// CTAs the single-slice m32_n64 kernel fields for `[m, n]`.
#[inline]
#[must_use]
fn cta_count_2d(m: u32, n: u32) -> u32 {
    n.div_ceil(64).saturating_mul(m.div_ceil(32))
}

/// Slices needed to lift a `ctas`-wide grid past `starved_cta_limit_for_sm(sm_count)`,
/// clamped to the caller's configured budget. Returns 0 when already saturated.
#[inline]
#[must_use]
fn slices_for_sm_ctas(ctas: u32, budget: u32, sm_count: u32) -> u32 {
    let limit = starved_cta_limit_for_sm(sm_count);
    if ctas >= limit {
        return 0;
    }
    limit.div_ceil(ctas.max(1)).min(budget)
}

/// Slices needed to lift `n`'s 1-M-tile grid past `starved_cta_limit_for_sm(sm_count)`,
/// clamped to the caller's configured budget. Returns 0 when already saturated.
#[cfg(test)]
fn slices_for_sm(n: u32, budget: u32, sm_count: u32) -> u32 {
    slices_for_sm_ctas(cta_count(n), budget, sm_count)
}

#[cfg(test)]
fn slices_for(n: u32, budget: u32) -> u32 {
    slices_for_sm(n, budget, atlas_core::device::active_sm_count())
}

/// The kernel choice for one drafter projection, as a value.
///
/// Extracted from `draft_gemm_t` so the choice is testable WITHOUT a GPU or a
/// constructed drafter head. The regression this guards: a refactor that
/// funnels nine call sites through one helper must, with the gate off, pick
/// exactly the kernel each site picked before. See `plan_matches_legacy_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DispatchPlan {
    /// `ops::w4a16_gemm_n64_m32` with `kernels.w4a16_gemm_t_m32_n64`.
    M32,
    /// `ops::w4a16_gemm_n128_m16` with `kernels.w4a16_gemm_t_m16`.
    M16,
    /// `ops::w4a16_gemm_n64_m32_splitk` with `k_splits` K-slices.
    SplitK { splits: u32 },
}

/// Everything `plan_dispatch` needs that would otherwise require a live head.
#[derive(Debug, Clone, Copy)]
pub(super) struct DispatchCaps {
    pub has_m32: bool,
    pub has_splitk: bool,
    pub has_reduce: bool,
    pub has_workspace: bool,
    /// Maximum output width eligible for split-K (`hidden_size.max(q_dim).max(inter)`).
    pub workspace_n: u32,
    /// Exact FP32 element capacity of `splitk_ws` (`8 * 64 * hidden_size.max(q_dim)`
    /// elements, i.e. `* 4` bytes, in `from_weights.rs`). Enforces
    /// `splits * m * n <= workspace_elems` so wider `n` (`q_dim > hidden_size`)
    /// and 2-tile `m <= 64` never overrun.
    pub workspace_elems: u64,
    /// `ATLAS_DFLASH_DRAFT_SPLITK`; 0 disables split-K entirely.
    pub budget: u32,
    /// Active device SM count used to scale the CTA starvation threshold.
    pub sm_count: u32,
}

/// The pre-refactor dispatch, reproduced exactly: each of the nine NVFP4 call
/// sites used `if <site guard> && m32_handle != 0 { m32 } else { m16 }`.
/// Kept as its own function so the equivalence test compares against a written
/// specification rather than against the implementation under test.
pub(super) fn legacy_plan(has_m32: bool, allow_m32: bool) -> DispatchPlan {
    if allow_m32 && has_m32 {
        DispatchPlan::M32
    } else {
        DispatchPlan::M16
    }
}

/// Choose the kernel for `[m, k] x [k, n]`.
///
/// With `caps.budget == 0` this is required to equal [`legacy_plan`] for every
/// input — that is the "unset is byte-identical" contract, and it is asserted
/// exhaustively over the production shapes in this module's tests.
pub(super) fn plan_dispatch(caps: &DispatchCaps, m: u32, n: u32, allow_m32: bool) -> DispatchPlan {
    let legacy = legacy_plan(caps.has_m32, allow_m32);
    if legacy != DispatchPlan::M32 || m == 0 || n == 0 {
        // Split-K is a K-sliced variant of the m32 kernel, so it is only
        // reachable where m32 itself was chosen.
        return legacy;
    }
    let ctas = cta_count_2d(m, n);
    // Allow 1 M-tile (`m <= 32`) on all GPUs, and extend to 2 M-tiles (`m <= 64`)
    // on wide discrete GPUs (`> 64` SMs) when single-slice CTAs fall under
    // `2 * sm_count` (e.g. C=8 `k/v_proj` fielding only 32 CTAs on 188 SMs).
    let m_allowed = if m <= 32 {
        true
    } else if m <= 64 {
        caps.sm_count > AUTO_DRAFT_SPLITK_SM_THRESHOLD && ctas < caps.sm_count.saturating_mul(2)
    } else {
        false
    };
    if !m_allowed {
        return legacy;
    }
    if caps.budget < 2
        || !caps.has_splitk
        || !caps.has_reduce
        || !caps.has_workspace
        || n > caps.workspace_n
    {
        return legacy;
    }
    let mn = u64::from(m).saturating_mul(u64::from(n));
    let max_ws_splits = u32::try_from(caps.workspace_elems / mn).unwrap_or(u32::MAX);
    let effective_budget = caps.budget.min(max_ws_splits);
    if effective_budget < 2 {
        return legacy;
    }
    let splits = slices_for_sm_ctas(ctas, effective_budget, caps.sm_count);
    if splits >= 2 {
        DispatchPlan::SplitK { splits }
    } else {
        legacy
    }
}

impl BlockDiffusionDraftHead {
    /// Capability snapshot for [`plan_dispatch`].
    fn dispatch_caps(&self, gpu: &dyn GpuBackend) -> DispatchCaps {
        let q_dim = self.num_q_heads.saturating_mul(self.head_dim);
        let workspace_n = self.hidden_size.max(q_dim).max(self.intermediate_size) as u32;
        // Matches `gpu.alloc(8 * 64 * hidden_size.max(q_dim) * 4)` in `from_weights.rs`.
        let workspace_elems =
            u64::from(AUTO_DRAFT_SPLITK_BUDGET) * 64 * (self.hidden_size.max(q_dim) as u64);
        DispatchCaps {
            has_m32: self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            has_splitk: self.kernels.w4a16_gemm_t_m32_n64_splitk.0 != 0,
            has_reduce: self.kernels.reduce_splitk_k.0 != 0,
            has_workspace: self.scratch.splitk_ws != DevicePtr::NULL,
            workspace_n,
            workspace_elems,
            budget: draft_splitk(),
            sm_count: gpu
                .sm_count()
                .filter(|&s| s > 0)
                .unwrap_or_else(atlas_core::device::active_sm_count),
        }
    }

    /// Dispatch one drafter NVFP4 projection `[M, K] x [K, N] -> [M, N]`.
    ///
    /// `allow_m32` is the call site's pre-existing guard on whether the
    /// M_TILE=32 kernel is legal for this shape (the kv-context-new site
    /// requires `new_ctx_count <= 32`; the others pass `true`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_gemm_t(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        allow_m32: bool,
        stream: u64,
    ) -> Result<()> {
        // Observational only, `Once`-guarded: puts the device's real SM count
        // in the log next to the active starvation threshold.
        log_sm_count_assumption(gpu);
        match plan_dispatch(&self.dispatch_caps(gpu), m, n, allow_m32) {
            DispatchPlan::SplitK { splits } => ops::w4a16_gemm_n64_m32_splitk(
                gpu,
                self.kernels.w4a16_gemm_t_m32_n64_splitk,
                self.kernels.reduce_splitk_k,
                input,
                weight,
                output,
                self.scratch.splitk_ws,
                m,
                n,
                k,
                n, // ldb == N for tightly-packed T-weights
                splits,
                stream,
            ),
            DispatchPlan::M32 => ops::w4a16_gemm_n64_m32(
                gpu,
                self.kernels.w4a16_gemm_t_m32_n64,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            ),
            DispatchPlan::M16 => ops::w4a16_gemm_n128_m16(
                gpu,
                self.kernels.w4a16_gemm_t_m16,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DispatchCaps, DispatchPlan, STARVED_CTA_LIMIT, cta_count, legacy_plan, plan_dispatch,
        resolve_draft_splitk, slices_for, slices_for_sm, starved_cta_limit_for_sm,
    };

    const HIDDEN: u32 = 5120;
    const Q_DIM: u32 = 4096;
    const KV_DIM: u32 = 1024;
    const INTER: u32 = 17408;
    /// gamma+1 noise rows under the production GAMMA.
    const M_NOISE: u32 = 13;

    /// The nine NVFP4 GEMM call sites in `forward_block_layer_nvfp4`, as
    /// (label, m, n, k, allow_m32) — transcribed from the call sites.
    fn production_sites() -> Vec<(&'static str, u32, u32, u32, bool)> {
        let mut v = vec![
            ("q_proj", M_NOISE, Q_DIM, HIDDEN, true),
            ("kv_noise_k", M_NOISE, KV_DIM, HIDDEN, true),
            ("kv_noise_v", M_NOISE, KV_DIM, HIDDEN, true),
            ("o_proj", M_NOISE, HIDDEN, Q_DIM, true),
            ("gate", M_NOISE, INTER, HIDDEN, true),
            ("up", M_NOISE, INTER, HIDDEN, true),
            ("down_proj", M_NOISE, HIDDEN, INTER, true),
        ];
        // kv_ctx_new runs at `new_ctx_count`, which varies per step and CAN
        // exceed 32 (first propose after a prefill) — hence its site guard.
        for ctx in [1u32, 8, 13, 32, 33, 64, 512] {
            v.push(("kv_ctx_new_k", ctx, KV_DIM, HIDDEN, ctx <= 32));
            v.push(("kv_ctx_new_v", ctx, KV_DIM, HIDDEN, ctx <= 32));
        }
        v
    }

    fn caps(budget: u32, has_m32: bool) -> DispatchCaps {
        DispatchCaps {
            has_m32,
            has_splitk: true,
            has_reduce: true,
            has_workspace: true,
            workspace_n: HIDDEN,
            workspace_elems: 8 * 32 * u64::from(HIDDEN),
            budget,
            sm_count: 48,
        }
    }

    /// THE regression guard: gate unset must reproduce the pre-refactor
    /// ternary at every call site, with and without the m32 kernel present.
    #[test]
    fn gate_off_is_identical_to_the_pre_refactor_ternary() {
        for has_m32 in [true, false] {
            let c = caps(0, has_m32);
            for (label, m, n, k, allow) in production_sites() {
                let got = plan_dispatch(&c, m, n, allow);
                let want = legacy_plan(has_m32, allow);
                assert_eq!(
                    got, want,
                    "{label} [m={m} n={n} k={k} allow_m32={allow} has_m32={has_m32}]                      diverged from the legacy dispatch with the gate OFF"
                );
            }
        }
    }

    /// Missing kernels, a missing workspace, or an oversized N must all fall
    /// back to the legacy choice even with the gate on — a null handle or an
    /// unallocated workspace must never reach a launch.
    #[test]
    fn gate_on_still_falls_back_when_prerequisites_are_missing() {
        let sites = production_sites();
        let degradations: [(&str, DispatchCaps); 4] = [
            (
                "no splitk kernel",
                DispatchCaps {
                    has_splitk: false,
                    ..caps(8, true)
                },
            ),
            (
                "no reduce kernel",
                DispatchCaps {
                    has_reduce: false,
                    ..caps(8, true)
                },
            ),
            (
                "no workspace",
                DispatchCaps {
                    has_workspace: false,
                    ..caps(8, true)
                },
            ),
            (
                "workspace too small",
                DispatchCaps {
                    workspace_n: 0,
                    ..caps(8, true)
                },
            ),
        ];
        for (why, c) in degradations {
            for (label, m, n, _k, allow) in &sites {
                assert_eq!(
                    plan_dispatch(&c, *m, *n, *allow),
                    legacy_plan(c.has_m32, *allow),
                    "{label} did not fall back to the legacy dispatch when {why}"
                );
            }
        }
    }

    /// With the gate on, only the occupancy-starved shapes are split, only
    /// where the m32 kernel was already legal, and never above one M-tile.
    #[test]
    fn gate_on_splits_exactly_the_starved_shapes() {
        let c = caps(8, true);
        // gate/up (N=intermediate) is already saturated — never split.
        assert_eq!(plan_dispatch(&c, M_NOISE, INTER, true), DispatchPlan::M32);
        // The starved shapes are split.
        assert_eq!(
            plan_dispatch(&c, M_NOISE, KV_DIM, true),
            DispatchPlan::SplitK { splits: 8 }
        );
        assert_eq!(
            plan_dispatch(&c, M_NOISE, Q_DIM, true),
            DispatchPlan::SplitK { splits: 4 }
        );
        assert_eq!(
            plan_dispatch(&c, M_NOISE, HIDDEN, true),
            DispatchPlan::SplitK { splits: 4 }
        );
        // M above one M-tile, and sites whose guard forbids m32, stay legacy.
        assert_eq!(plan_dispatch(&c, 33, KV_DIM, false), DispatchPlan::M16);
        assert_eq!(plan_dispatch(&c, 64, KV_DIM, true), DispatchPlan::M32);
    }

    #[test]
    fn only_the_wide_n_shape_is_considered_saturated() {
        assert_eq!(cta_count(KV_DIM), 16);
        assert_eq!(cta_count(Q_DIM), 64);
        assert_eq!(cta_count(HIDDEN), 80);
        assert_eq!(cta_count(INTER), 272);
        assert!(cta_count(INTER) >= STARVED_CTA_LIMIT);
    }

    #[test]
    fn saturated_shapes_are_never_split() {
        assert_eq!(slices_for(INTER, 8), 0);
    }

    #[test]
    fn starved_shapes_are_lifted_toward_the_limit_within_budget() {
        assert_eq!(slices_for(KV_DIM, 8), 8);
        assert_eq!(slices_for(KV_DIM, 4), 4);
        assert_eq!(slices_for(Q_DIM, 8), 4);
        assert_eq!(slices_for(HIDDEN, 8), 4);
    }

    #[test]
    fn a_budget_below_two_yields_no_usable_split() {
        assert!(slices_for(HIDDEN, 1) < 2);
    }

    #[test]
    fn draft_splitk_auto_enables_and_scales_starved_limit_on_high_sm_gpus() {
        // <= 64 SMs (GB10): default OFF, floor stays 256.
        assert_eq!(resolve_draft_splitk(None, 48), 0);
        assert_eq!(resolve_draft_splitk(None, 64), 0);
        assert_eq!(starved_cta_limit_for_sm(48), 256);

        // > 64 SMs (SM120 / SM103): default ON (budget=8), explicit 0 disables.
        for (sm, want_limit) in [(110u32, 440u32), (170, 680), (188, 752)] {
            assert_eq!(resolve_draft_splitk(None, sm), 8, "sm={sm}");
            assert_eq!(resolve_draft_splitk(Some("0"), sm), 0, "sm={sm}");
            assert_eq!(resolve_draft_splitk(Some("4"), sm), 4, "sm={sm}");
            assert_eq!(starved_cta_limit_for_sm(sm), want_limit, "sm={sm}");
        }

        // On 170–188 SM GPUs, Q_DIM (64 CTAs) and HIDDEN (80 CTAs) lift to 8 splits
        // (512–640 CTAs) to saturate GDDR7 bandwidth while INTER stays safely on M32
        // (bounded by workspace_n = HIDDEN).
        assert_eq!(slices_for_sm(Q_DIM, 8, 110), 7);
        assert_eq!(slices_for_sm(HIDDEN, 8, 110), 6);
        assert_eq!(slices_for_sm(Q_DIM, 8, 170), 8);
        assert_eq!(slices_for_sm(HIDDEN, 8, 170), 8);
        assert_eq!(slices_for_sm(Q_DIM, 8, 188), 8);
        assert_eq!(slices_for_sm(HIDDEN, 8, 188), 8);

        let high_sm_caps = DispatchCaps {
            sm_count: 188,
            ..caps(8, true)
        };
        assert_eq!(
            plan_dispatch(&high_sm_caps, M_NOISE, INTER, true),
            DispatchPlan::M32
        );
        assert_eq!(
            plan_dispatch(&high_sm_caps, M_NOISE, HIDDEN, true),
            DispatchPlan::SplitK { splits: 8 }
        );
    }

    #[test]
    fn wide_sm_q_proj_and_c8_splitk_never_overrun_workspace() {
        // DFlash2 Flash-Next geometry: hidden=2560 < q_dim=4096 < inter=10240.
        const FN_HIDDEN: u32 = 2560;
        const FN_Q_DIM: u32 = 4096;
        const FN_KV_DIM: u32 = 1024;
        const FN_INTER: u32 = 10240;
        let ws_elems = 8 * 32 * u64::from(FN_HIDDEN);
        let sm120 = DispatchCaps {
            has_m32: true,
            has_splitk: true,
            has_reduce: true,
            has_workspace: true,
            workspace_n: FN_HIDDEN.max(FN_Q_DIM).max(FN_INTER),
            workspace_elems: ws_elems,
            budget: 8,
            sm_count: 188,
        };

        // C=1 (m=8) and C=2 (m=16): q_proj (n=4096 > hidden=2560) now splits to 8 (512 CTAs).
        assert_eq!(
            plan_dispatch(&sm120, 8, FN_Q_DIM, true),
            DispatchPlan::SplitK { splits: 8 }
        );
        assert_eq!(
            plan_dispatch(&sm120, 16, FN_Q_DIM, true),
            DispatchPlan::SplitK { splits: 8 }
        );
        // C=4 (m=32): q_proj clamps to 5 splits so 5 * 32 * 4096 == 8 * 32 * 2560 (0 overrun).
        assert_eq!(
            plan_dispatch(&sm120, 32, FN_Q_DIM, true),
            DispatchPlan::SplitK { splits: 5 }
        );
        // C=8 (m=64): k/v_proj (32 CTAs < 188*2) splits to 8 (256 CTAs); o/down clamps to 4;
        // q_proj clamps to 2; gate/up (n=10240) exceeds workspace for >=2 splits -> safe M32.
        assert_eq!(
            plan_dispatch(&sm120, 64, FN_KV_DIM, true),
            DispatchPlan::SplitK { splits: 8 }
        );
        assert_eq!(
            plan_dispatch(&sm120, 64, FN_HIDDEN, true),
            DispatchPlan::SplitK { splits: 4 }
        );
        assert_eq!(
            plan_dispatch(&sm120, 64, FN_Q_DIM, true),
            DispatchPlan::SplitK { splits: 2 }
        );
        assert_eq!(plan_dispatch(&sm120, 64, FN_INTER, true), DispatchPlan::M32);
        // Above m=64 stays single-slice M32.
        assert_eq!(
            plan_dispatch(&sm120, 65, FN_KV_DIM, true),
            DispatchPlan::M32
        );

        // GB10 (<= 64 SMs): m > 32 stays single-slice M32 even if budget is forced on.
        let gb10 = DispatchCaps {
            sm_count: 48,
            ..sm120
        };
        assert_eq!(plan_dispatch(&gb10, 64, FN_KV_DIM, true), DispatchPlan::M32);

        // Exhaustive workspace safety invariant across all shapes and SM counts.
        for sm in [48u32, 64, 110, 170, 188] {
            let c = DispatchCaps {
                sm_count: sm,
                ..sm120
            };
            for m in [1u32, 8, 13, 16, 24, 32, 33, 48, 64, 65, 128] {
                for n in [FN_KV_DIM, FN_HIDDEN, FN_Q_DIM, FN_INTER, INTER] {
                    if let DispatchPlan::SplitK { splits } = plan_dispatch(&c, m, n, true) {
                        assert!(splits >= 2 && splits <= c.budget, "sm={sm} m={m} n={n}");
                        assert!(
                            u64::from(splits) * u64::from(m) * u64::from(n) <= c.workspace_elems,
                            "workspace overrun: splits={splits} m={m} n={n} > ws={}",
                            c.workspace_elems
                        );
                    }
                }
            }
        }
    }
}
