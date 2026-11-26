// SPDX-License-Identifier: AGPL-3.0-only

//! Which preparations `ATLAS_QWEN4_PREFILL_MHC_FUSED=injgemm` may run as one
//! kernel (see [`super::prefill_inject_gemm`]) under the cuBLASLt split-K
//! parity gate: row counts whose pinned cuBLASLt GEMM the kernels reproduce
//! bit for bit. Setting `ATLAS_QWEN4_PREFILL_HC_FUSED != 0` explicitly opts
//! out of this gate at the callers (`inject_gemm_admitted` and
//! `plan_cluster_fuse`), admitting arbitrary row counts including >8192 and
//! non-multiples of 8.

use anyhow::Result;
use spark_runtime::cublaslt::{GemmDtype, pinned_algo_config};

pub(super) const HIDDEN: usize = 2560;
pub(super) const STREAMS: usize = 4;
pub(super) const RANK: usize = 320;

/// cuBLASLt configurations (`[ALGO_ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME,
/// CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID]`) of HC_FUSED's per-stream GEMM
/// that bench-prefill/fn_mhc_fused proved bit-identical to the kernels'
/// ascending-k16 chain (micro3/micro4, CUDA 13.0: M 257-512, 545-8192). The split-K
/// picks at other row counts are not, and keep the unfused GEMMs.
const VERIFIED_ALGOS: [[u32; 7]; 5] = [
    [23, 15, 1, 0, 0, 0, 18],
    [23, 17, 1, 0, 0, 0, 15],
    [23, 18, 1, 0, 0, 0, 15],
    [23, 20, 1, 0, 0, 0, 15],
    [23, 24, 1, 0, 0, 0, 9],
];

pub(super) const MAX_VERIFIED_ROWS: usize = 8192;

/// Whether HC_FUSED's pinned per-stream GEMM at `rows` rows (prefill_fused.rs:
/// bf16 [rows, 2560] at lda 10240 x [324, 2560]^T -> f32, split-K allowed)
/// runs a cuBLASLt configuration the kernels reproduce bit for bit.
///
/// Callers with `ATLAS_QWEN4_PREFILL_HC_FUSED != 0` explicitly opt out of this
/// parity gate before calling `gemm_order_verified`.
pub(super) fn gemm_order_verified(rows: usize) -> Result<bool> {
    if rows == 0 || rows > MAX_VERIFIED_ROWS || !rows.is_multiple_of(8) {
        return Ok(false);
    }
    let n = (RANK + STREAMS) as u32;
    let config = pinned_algo_config(
        (HIDDEN * STREAMS) as u32,
        n,
        n,
        HIDDEN as u32,
        GemmDtype::Bf16,
        GemmDtype::F32,
        false,
        u32::try_from(rows)?,
    )?;
    let verified = VERIFIED_ALGOS.contains(&config);
    if !verified {
        // A bounded interleave chunk (or a prompt tail) whose pinned config is
        // not verified runs the unfused GEMMs for this component only.
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "QWEN4_PREFILL_INJGEMM_FALLBACK rows={rows} config={config:?}: pinned cuBLASLt \
                 config is not in VERIFIED_ALGOS; this preparation uses the unfused GEMMs \
                 (logged once)"
            )
        });
    }
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_algorithms_never_split_k() {
        for config in VERIFIED_ALGOS {
            assert_eq!((config[2], config[3]), (1, 0), "{config:?}");
        }
    }

    #[test]
    fn boundary_and_unaligned_rows_reject_before_cublas() {
        for rows in [
            0, 1, 27, 61, 5369, 5370, 8106, 8190, 8193, 8197, 8200, 16384, 32768,
        ] {
            assert!(
                !gemm_order_verified(rows).unwrap(),
                "rows={rows} must reject"
            );
        }
    }
}
