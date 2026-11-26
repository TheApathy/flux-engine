// SPDX-License-Identifier: AGPL-3.0-only

//! Dtype- and stride-parameterised `out[M,N] = act[M,K] @ weight[N,K]ᵀ`.
//!
//! The BF16 entry points in the parent module fix the operand type. DeepSeek-V4.1 also needs
//! TRUE fp32 GEMMs — the router (`y.float() @ gate_w`), the ratio-2 compressor and the
//! mHC mixes all run in fp32 in the reference, and routing turns on ulps — so this adds one
//! path where the operand and output types are arguments. `CUBLAS_COMPUTE_32F` with
//! `CUDA_R_32F` operands is IEEE fp32 (no TF32: that is `CUBLAS_COMPUTE_32F_FAST_TF32`).

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::*;

/// Give `stream` its own cuBLASLt workspace (idempotent). Process-lifetime,
/// like the primary workspace. Must be called before CUDA graph capture on any
/// side stream that issues cuBLASLt GEMMs.
pub fn register_stream_workspace(stream: u64) -> Result<()> {
    ctx()?.register_stream_workspace(stream)
}

/// `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` (u32).
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;

/// Element type of a GEMM operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmDtype {
    Bf16,
    F32,
}

impl GemmDtype {
    pub(super) fn cuda(self) -> i32 {
        match self {
            GemmDtype::Bf16 => CUDA_R_16BF,
            GemmDtype::F32 => CUDA_R_32F,
        }
    }
}

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ` with explicit row strides (`lda` for `act`,
/// `ldc` for `out`, in elements; the weight is packed `[N,K]`), fp32 accumulate.
///
/// `in_dtype` applies to both `act` and `weight`; `out_dtype` to `out`. cuBLASLt's default
/// heuristic (split-K allowed). See [`gemm_act_weight_t_typed_ex`] for the no-split-K form.
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    stream: u64,
) -> Result<()> {
    gemm_act_weight_t_typed_ex(
        act, lda, weight, out, ldc, m, n, k, in_dtype, out_dtype, false, stream,
    )
}

/// [`gemm_act_weight_t_typed`] with a PER-CALL reduction policy. `no_split_k = true` restricts
/// the heuristic to algorithms that do not split K
/// (`CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 0`), so a row's sum is accumulated in one
/// order whatever M is.
///
/// Why per call and not global: this file is shared across models, and the only caller that
/// needs it measured it. dsv41-attention, on DeepSeek-V4.1's L20 compressor (16x512x5120,
/// runF_faithful): default heuristic (split-K chosen) ckv 70.7% bf16-exact, top-k 296/512 rows
/// wrong; split-K with fp32 reduction (mask 2) 97.75% / 53/512; mask 0 100.00% / 1/512.
/// (The integrate/all-models branch has a separate deterministic-SELECTION fix for the tuned
/// bf16 path in cublaslt.rs — tie-breaking between measured candidates. It does not restrict
/// the reduction scheme and does not cover this untuned typed path.)
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed_ex(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    no_split_k: bool,
    stream: u64,
) -> Result<()> {
    typed_impl(
        act, lda, weight, out, ldc, m, n, k, in_dtype, out_dtype, no_split_k, None, stream,
    )
}

/// [`gemm_act_weight_t_typed_ex`] with ONE algorithm per shape, independent of M: the heuristic
/// runs once per (n, k, lda, ldc, dtypes, no_split_k) at M = `ref_m`, and that algorithm is then
/// issued at the TRUE M of every call. With a fixed tile configuration and no split-K, an output
/// element's K-loop does not depend on how many rows share the call, so results become
/// M-invariant without padding M (the property DeepSeek-V4.1's chunk invariance needs,
/// verified empirically against split-chunk reference outputs).
#[allow(clippy::too_many_arguments)]
pub fn gemm_act_weight_t_typed_pinned(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    no_split_k: bool,
    ref_m: u32,
    stream: u64,
) -> Result<()> {
    typed_impl(
        act,
        lda,
        weight,
        out,
        ldc,
        m,
        n,
        k,
        in_dtype,
        out_dtype,
        no_split_k,
        Some(ref_m),
        stream,
    )
}

pub(super) type PinKey = (u32, u32, u32, u32, i32, i32, bool, u32);

pub(super) fn pinned_algos()
-> &'static std::sync::Mutex<std::collections::HashMap<PinKey, HeurResult>> {
    static P: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PinKey, HeurResult>>> =
        std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[allow(clippy::too_many_arguments)]
fn typed_impl(
    act: u64,
    lda: u32,
    weight: u64,
    out: u64,
    ldc: u32,
    m: u32,
    n: u32,
    k: u32,
    in_dtype: GemmDtype,
    out_dtype: GemmDtype,
    no_split_k: bool,
    pin_ref_m: Option<u32>,
    stream: u64,
) -> Result<()> {
    if lda < k || ldc < n {
        bail!("cuBLASLt typed: lda ({lda}) < k ({k}) or ldc ({ldc}) < n ({n})");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let ctx = ctx()?;
    let ws = ctx.workspace_for_stream(stream)?;
    // SAFETY: All cuBLASLt descriptors/layouts/preferences are initialized before use and wrapped
    // in RAII guards so early error returns never leak handles. `result` is an 8-byte-aligned
    // `HeurResult` matching `cublasLtMatmulHeuristicResult_t`; `ws` is a dedicated per-stream
    // device workspace of `ctx.ws_size` bytes; caller supplies valid device pointers and stream.
    unsafe {
        let mut desc_raw: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc_raw, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let desc = DescGuard(desc_raw);
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc.0,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        let (ti, to) = (in_dtype.cuda(), out_dtype.cuda());
        let mut la_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la_raw, ti, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        let la = LayoutGuard(la_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut lb_raw, ti, k as u64, m as u64, lda as i64),
            "LayoutB",
        )?;
        let lb = LayoutGuard(lb_raw);
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_raw, to, n as u64, m as u64, ldc as i64),
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
                &ws_size as *const usize as *const c_void,
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
                    &mask as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
                "PrefReductionScheme",
            )?;
        }
        let mut result = HeurResult::ZERO;
        let mut returned: i32 = 0;
        let pin_key: Option<PinKey> = pin_ref_m.map(|r| (n, k, lda, ldc, ti, to, no_split_k, r));
        let cached = pin_key.and_then(|key| pinned_algos().lock().ok()?.get(&key).copied());
        let heur = if let Some(c) = cached {
            result = c;
            returned = 1;
            0
        } else if let Some(r) = pin_ref_m {
            // Choose the algorithm at the REFERENCE M, then reuse it at every M.
            let mut lb_r_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            let mut ld_r_raw: cublasLtMatrixLayout_t = std::ptr::null_mut();
            chk(
                cublasLtMatrixLayoutCreate(&mut lb_r_raw, ti, k as u64, r as u64, lda as i64),
                "LayoutBref",
            )?;
            let lb_r = LayoutGuard(lb_r_raw);
            chk(
                cublasLtMatrixLayoutCreate(&mut ld_r_raw, to, n as u64, r as u64, ldc as i64),
                "LayoutDref",
            )?;
            let ld_r = LayoutGuard(ld_r_raw);
            let h = cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc.0,
                la.0,
                lb_r.0,
                ld_r.0,
                ld_r.0,
                pref.0,
                1,
                (&mut result as *mut HeurResult).cast(),
                &mut returned,
            );
            if h == 0 && returned >= 1 {
                if let (Some(key), Ok(mut t)) = (pin_key, pinned_algos().lock()) {
                    t.insert(key, result);
                }
                // ATLAS_LOG_PINNED_ALGO=1: one line per pinned shape, so two processes can be
                // diffed. The heuristic can pick differently when the workspace, stream or
                // allocation alignment differ; this makes that visible instead of inferred.
                if std::env::var("ATLAS_LOG_PINNED_ALGO").as_deref() == Ok("1") {
                    let hex: String = result.algo.0.iter().map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "PINNED_ALGO n={n} k={k} lda={lda} ldc={ldc} in={ti} out={to} no_split_k={no_split_k} ref_m={r} ws={} stream={stream:#x} algo={hex}",
                        ctx.ws_size
                    );
                }
            }
            h
        } else {
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
            )
        };
        if heur != 0 || returned < 1 {
            bail!(
                "cuBLASLt typed: no algorithm for {m}x{n}x{k} {in_dtype:?}->{out_dtype:?} \
                 (heuristic status {heur}, returned {returned})"
            );
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc.0,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la.0,
            act as *const c_void,
            lb.0,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_.0,
            out as *mut c_void,
            ld_.0,
            result.algo.as_ptr().cast(),
            ws as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        chk(status, "MatmulTyped")
    }
}
