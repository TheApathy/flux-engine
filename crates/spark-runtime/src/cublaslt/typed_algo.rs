// SPDX-License-Identifier: AGPL-3.0-only

//! Which cuBLASLt algorithm [`super::gemm_act_weight_t_typed_pinned`] runs.
//!
//! A caller that replaces that GEMM with its own kernel reproducing one
//! algorithm's accumulation order (bit for bit) must know which row counts
//! actually run that algorithm. This picks the algorithm exactly as the
//! pinned call does (same descriptors, same pin cache, same heuristic at the
//! reference M) without issuing a matmul, and reports its configuration.

use anyhow::{Result, bail};

use super::typed::{GemmDtype, PinKey, pinned_algos};
use super::*;

/// `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` (u32).
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;
/// `cublasLtMatmulAlgoConfigAttributes_t`: ALGO_ID, TILE_ID, SPLITK_NUM,
/// REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID (all 4 bytes).
const CONFIG_ATTRS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

/// Attributes of a pinned cuBLASLt algorithm for a specific matrix shape.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedAlgoSpec {
    pub name: &'static str,
    pub m_ref: u32,
    pub n: u32,
    pub k: u32,
    pub config: [u32; 7],
}

/// Catalog of reference pinned configurations for Blackwell GB10 SM121a:
/// `[ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID]`
/// Pinning splitK=1 and deterministic reduction eliminates runtime heuristic drift.
#[cfg(test)]
pub const PRODUCTION_PINNED_CATALOG: &[PinnedAlgoSpec] = &[
    // MoE gate_up: M=80, N=512, K=2048 (no split-K, 16 CTAs)
    PinnedAlgoSpec {
        name: "MoE_gate_up_80x2048x512",
        m_ref: 80,
        n: 512,
        k: 2048,
        config: [23, 15, 1, 0, 0, 0, 18],
    },
    // MoE down: M=80, N=2048, K=512 (no split-K, 64 CTAs saturating 48 SMs)
    PinnedAlgoSpec {
        name: "MoE_down_80x512x2048",
        m_ref: 80,
        n: 2048,
        k: 512,
        config: [23, 18, 1, 0, 0, 0, 15],
    },
    // Attn Q projection: M=16, N=256, K=2048 (pinned tile 16x32, split-K=1, deterministic reduction)
    PinnedAlgoSpec {
        name: "Attn_Q_16x2048x256",
        m_ref: 16,
        n: 256,
        k: 2048,
        config: [23, 20, 1, 0, 0, 0, 15],
    },
    // Medium reference: M=256, N=256, K=256
    PinnedAlgoSpec {
        name: "Medium_256x256x256",
        m_ref: 256,
        n: 256,
        k: 256,
        config: [23, 24, 1, 0, 0, 0, 9],
    },
];

/// Return the GB10 SM121a pinned algorithm spec for `(n, k, m_ref)` only when
/// running on SM121 (`Some((12, 1))`) or before device initialization (`None`,
/// e.g. unit tests). Discrete Blackwell (`SM120` `(12, 0)`) and Blackwell Ultra
/// (`SM103` `(10, 3)`) return `None` so callers query the cuBLASLt heuristic
/// at runtime for the actual GPU.
#[cfg(test)]
#[must_use]
fn lookup_pinned_algo_for_cc(
    n: u32,
    k: u32,
    m_ref: u32,
    cc: Option<(u32, u32)>,
) -> Option<&'static PinnedAlgoSpec> {
    if !matches!(cc, None | Some((12, 1))) {
        return None;
    }
    PRODUCTION_PINNED_CATALOG
        .iter()
        .find(|spec| spec.n == n && spec.k == k && spec.m_ref == m_ref)
}

#[cfg(test)]
#[must_use]
pub fn lookup_pinned_algo(n: u32, k: u32, m_ref: u32) -> Option<&'static PinnedAlgoSpec> {
    lookup_pinned_algo_for_cc(n, k, m_ref, atlas_core::device::active_compute_capability())
}

/// The configuration `[ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME,
/// CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID]` of the algorithm
/// [`super::gemm_act_weight_t_typed_pinned`] issues for these operands with
/// `ref_m`. Picks (and pins) it the way that call would if it ran first.
#[allow(clippy::too_many_arguments)]
pub fn pinned_algo_config(
    lda: u32,
    ldc: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    no_split_k: bool,
    ref_m: u32,
) -> Result<[u32; 7]> {
    if lda < k || ldc < n || ref_m == 0 || n == 0 || k == 0 {
        bail!("cuBLASLt pinned algo: bad shape lda {lda} ldc {ldc} n {n} k {k} ref_m {ref_m}");
    }
    let (ti, to) = (in_dtype.cuda(), out_dtype.cuda());
    let key: PinKey = (n, k, lda, ldc, ti, to, no_split_k, ref_m);
    let cached = pinned_algos()
        .lock()
        .ok()
        .and_then(|t| t.get(&key).copied());
    let heur = match cached {
        Some(h) => h,
        None => {
            let h = heuristic_at(lda, ldc, n, k, ti, to, no_split_k, ref_m)?;
            if let Ok(mut t) = pinned_algos().lock() {
                t.entry(key).or_insert(h);
            }
            h
        }
    };
    let mut config = [0u32; 7];
    for (slot, attr) in config.iter_mut().zip(CONFIG_ATTRS) {
        let mut written = 0usize;
        // SAFETY: `heur.algo` is an 8-byte-aligned `AlignedAlgo` initialized by
        // `cublasLtMatmulAlgoGetHeuristic`; `slot` is a valid 4-byte `u32` destination.
        chk(
            unsafe {
                cublasLtMatmulAlgoConfigGetAttribute(
                    heur.algo.as_ptr().cast(),
                    attr,
                    (slot as *mut u32).cast(),
                    std::mem::size_of::<u32>(),
                    &mut written,
                )
            },
            "AlgoConfigGetAttribute",
        )?;
    }
    Ok(config)
}

/// The heuristic's top algorithm at M = `ref_m`, with the descriptors and
/// preference `typed_impl` builds.
#[allow(clippy::too_many_arguments)]
fn heuristic_at(
    lda: u32,
    ldc: u32,
    n: u32,
    k: u32,
    ti: i32,
    to: i32,
    no_split_k: bool,
    ref_m: u32,
) -> Result<HeurResult> {
    let ctx = ctx()?;
    let mut result = HeurResult::ZERO;
    let mut returned: i32 = 0;
    // SAFETY: Every handle is created before use and owned by an RAII guard; `result` is an
    // 8-byte-aligned `HeurResult` matching `cublasLtMatmulHeuristicResult_t`.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let (ta, tb) = (CUBLAS_OP_T, CUBLAS_OP_N);
        chk(
            cublasLtMatmulDescSetAttribute(desc.0, DESC_TRANSA, (&ta as *const i32).cast(), 4),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(desc.0, DESC_TRANSB, (&tb as *const i32).cast(), 4),
            "TRANSB",
        )?;
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, ti, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut lb_raw, ti, k as u64, ref_m as u64, lda as i64),
            "LayoutB",
        )?;
        let lb = LayoutGuard(lb_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_raw, to, n as u64, ref_m as u64, ldc as i64),
            "LayoutD",
        )?;
        let ld_ = LayoutGuard(ld_raw);
        let mut pref_raw: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref_raw), "PrefCreate")?;
        let pref = PrefGuard(pref_raw);
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref.0,
                PREF_MAX_WORKSPACE_BYTES,
                (&ws_size as *const usize).cast(),
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        if no_split_k {
            let mask: u32 = 0;
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    pref.0,
                    PREF_REDUCTION_SCHEME_MASK,
                    (&mask as *const u32).cast(),
                    std::mem::size_of::<u32>(),
                ),
                "PrefReductionScheme",
            )?;
        }
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc.0,
                la.0,
                lb.0,
                ld_.0,
                ld_.0,
                pref.0,
                1,
                (&mut result as *mut HeurResult).cast(),
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
    }
    if returned < 1 {
        bail!("cuBLASLt pinned algo: no algorithm for n {n} k {k} at M {ref_m}");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_pinned_catalog_invariants() {
        assert_eq!(PRODUCTION_PINNED_CATALOG.len(), 4);
        for spec in PRODUCTION_PINNED_CATALOG {
            // Guarantee no split-K and deterministic reduction
            assert_eq!(spec.config[2], 1, "{}: splitK must be 1", spec.name);
            assert_eq!(
                spec.config[3], 0,
                "{}: reduction_scheme must be 0",
                spec.name
            );
            assert!(lookup_pinned_algo(spec.n, spec.k, spec.m_ref).is_some());
        }
    }

    #[test]
    fn test_lookup_pinned_algo_shapes() {
        let gate_up = lookup_pinned_algo(512, 2048, 80).expect("MoE gate_up");
        assert_eq!(gate_up.name, "MoE_gate_up_80x2048x512");
        assert_eq!(gate_up.config, [23, 15, 1, 0, 0, 0, 18]);

        let down = lookup_pinned_algo(2048, 512, 80).expect("MoE down");
        assert_eq!(down.name, "MoE_down_80x512x2048");
        assert_eq!(down.config, [23, 18, 1, 0, 0, 0, 15]);

        let attn_q = lookup_pinned_algo(256, 2048, 16).expect("Attn Q");
        assert_eq!(attn_q.name, "Attn_Q_16x2048x256");
        assert_eq!(attn_q.config, [23, 20, 1, 0, 0, 0, 15]);

        let medium = lookup_pinned_algo(256, 256, 256).expect("Medium");
        assert_eq!(medium.name, "Medium_256x256x256");
        assert_eq!(medium.config, [23, 24, 1, 0, 0, 0, 9]);

        assert!(lookup_pinned_algo(999, 999, 999).is_none());
    }

    #[test]
    fn test_lookup_pinned_algo_gates_non_sm121_compute_capabilities() {
        for spec in PRODUCTION_PINNED_CATALOG {
            // GB10 SM121 and unset unit-test context hit the pinned catalog.
            assert_eq!(
                lookup_pinned_algo_for_cc(spec.n, spec.k, spec.m_ref, Some((12, 1))),
                Some(spec),
                "{}: SM121 must hit catalog",
                spec.name
            );
            assert_eq!(
                lookup_pinned_algo_for_cc(spec.n, spec.k, spec.m_ref, None),
                Some(spec),
                "{}: unset CC (unit tests) must hit catalog",
                spec.name
            );
            // SM120 (RTX PRO 5000 / 5090 / PRO 6000), SM103 (B300 / GB300), and
            // other GPUs must return None so callers query cuBLASLt at runtime.
            for other_cc in [(12, 0), (10, 3), (10, 0), (9, 0)] {
                assert_eq!(
                    lookup_pinned_algo_for_cc(spec.n, spec.k, spec.m_ref, Some(other_cc)),
                    None,
                    "{}: CC {other_cc:?} must bypass SM121 pinned catalog",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn test_pinned_algo_config_bad_shape_rejection() {
        // Bad shapes (lda < k, ldc < n, or zero dimensions) must fail closed before cuBLASLt.
        assert!(
            pinned_algo_config(
                1024,
                512,
                512,
                2048,
                GemmDtype::Bf16,
                GemmDtype::Bf16,
                true,
                80
            )
            .is_err()
        );
        assert!(
            pinned_algo_config(
                2048,
                256,
                512,
                2048,
                GemmDtype::Bf16,
                GemmDtype::Bf16,
                true,
                80
            )
            .is_err()
        );
        assert!(
            pinned_algo_config(
                2048,
                512,
                512,
                2048,
                GemmDtype::Bf16,
                GemmDtype::Bf16,
                true,
                0
            )
            .is_err()
        );
    }
}
