// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `cublaslt` module.
//!
//! The real [`crate::cublaslt`] module (cuBLASLt BF16 act·weightᵀ GEMMs) is
//! gated behind `feature = "cuda"` because it links cuBLASLt, which does not
//! exist on macOS. spark-model names these entry points unconditionally, so
//! the metal build (cuda off) needs the symbols to resolve even though the
//! cuBLASLt route never runs there. The bodies are `unreachable!` — reaching
//! one on metal is a bug (the `ATLAS_CUBLAS_GEMM` gate must stay off).

use anyhow::Result;

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_ref_m(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _ref_m: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t_ref_m is cuda-only (not built for metal)")
}

pub fn bf16_gemm_act_weight_t_tuned(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight_t_tuned is cuda-only (not built for metal)")
}

pub fn bf16_gemm_act_weight(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cublaslt::bf16_gemm_act_weight is cuda-only (not built for metal)")
}

/// No-op twin of the cuda stage timer (`ATLAS_QWEN4_PREFILL_STAGES`).
pub mod stage_timer {
    pub fn enabled() -> bool {
        false
    }
    pub fn decode_enabled() -> bool {
        false
    }
    pub fn gap_enabled() -> bool {
        false
    }
    pub fn record_raw(_stream: u64) -> u64 {
        0
    }
    pub fn elapsed_raw(_a: u64, _b: u64) -> f64 {
        f64::NAN
    }
    pub fn release_raw(_e: u64) {}
    pub fn finish_raw() -> Option<(f64, Vec<(String, f64, usize)>)> {
        None
    }
    pub fn begin(_stream: u64) {}
    pub fn set_group(_group: &'static str) {}
    pub fn mark(_label: &'static str, _stream: u64) {}
    pub fn finish() -> Option<String> {
        None
    }
}
