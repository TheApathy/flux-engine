// SPDX-License-Identifier: AGPL-3.0-only

//! Native Blackwell BitDecoding-FP4 + 6-op SWAR RaZeR-KV route admission,
//! split-K geometry, and CPU numerical verification models.

use spark_runtime::kv_cache::KvCacheDtype;

use crate::layers::ops::{
    BITFP4_DECODE_Q_SUB, BITFP4_KGAMMA_Q_SUB, BITFP4_KV_BLOCK_SIZE, BITFP4_KV_HEAD_DIM,
    BitFp4KvKernels, BitFp4QuantFlags, BitFp4RazerMode,
};

use super::Qwen3AttentionLayer;

/// Deterministic dispatch state for the opt-in BitDecoding-FP4 KV path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BitFp4KvRoute {
    /// `ATLAS_BITFP4_KV` is unset or `0` (default): preserve byte-identical baseline.
    Disabled,
    /// Layer geometry/dtype is outside the SM120 BitDecoding-FP4 envelope
    /// (requires `KvCacheDtype::Nvfp4`, `head_dim == 256`, `block_size == 16`,
    /// no sliding window).
    Ineligible,
    /// Active device is not Blackwell SM120/SM121 (`cc.0 == 12`).
    UnsupportedArch,
    /// PTX module/symbols for `paged_decode_attn_bitfp4_sm120` did not resolve.
    MissingKernel,
    /// Active native Blackwell FP4 Tensor Core KV path.
    Active { razer: BitFp4RazerMode },
}

/// Native `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` is an
/// SM120/SM121 (compute capability `12.x`) instruction. SM103 (`10.3`) uses
/// `tcgen05` and traps on the SM120 path, so runtime dispatch fails closed on
/// any non-`12.x` architecture (and before device properties are populated).
#[inline]
#[must_use]
pub(crate) const fn bitfp4_kv_cc_supported(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((12, _)))
}

/// Pure, GPU-free route admission decision for BitDecoding-FP4 KV.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) const fn resolve_bitfp4_kv_route(
    requested: bool,
    razer_requested: bool,
    kv_dtype: KvCacheDtype,
    head_dim: u32,
    block_size: u32,
    has_sliding_window: bool,
    compute_cap: Option<(u32, u32)>,
    kernels: Option<&BitFp4KvKernels>,
) -> BitFp4KvRoute {
    if !requested {
        return BitFp4KvRoute::Disabled;
    }
    if !matches!(kv_dtype, KvCacheDtype::Nvfp4)
        || head_dim != BITFP4_KV_HEAD_DIM
        || block_size != BITFP4_KV_BLOCK_SIZE
        || has_sliding_window
    {
        return BitFp4KvRoute::Ineligible;
    }
    if !bitfp4_kv_cc_supported(compute_cap) {
        return BitFp4KvRoute::UnsupportedArch;
    }
    let razer = if razer_requested {
        BitFp4RazerMode::Enabled
    } else {
        BitFp4RazerMode::Disabled
    };
    match kernels {
        Some(k) if k.is_complete_for(razer) => BitFp4KvRoute::Active { razer },
        _ => BitFp4KvRoute::MissingKernel,
    }
}

/// Returns `true` iff `block_tables` is non-empty and every row shares the
/// exact same block table (single-sequence speculative `K = gamma` verify).
/// Concurrent independent requests have distinct block tables and return `false`.
#[inline]
#[must_use]
pub(in crate::layers::qwen3_attention) fn block_tables_all_shared(
    block_tables: &[Vec<u32>],
) -> bool {
    !block_tables.is_empty() && block_tables.windows(2).all(|w| w[0] == w[1])
}

/// Returns `true` iff a multi-seq BitDecoding-FP4 decode call may use the
/// single-table `paged_decode_attn_bitfp4_kgamma_sm120` kernel (which indexes
/// `block_tables[0]` for every row). Concurrent independent requests (`!shared_block_table`)
/// MUST route to per-sequence `paged_decode_attn_bitfp4_sm120` (`block_tables[blockIdx.z]`).
#[inline]
#[must_use]
pub(crate) const fn bitfp4_kgamma_eligible(num_rows: usize, shared_block_table: bool) -> bool {
    shared_block_table && num_rows >= 2 && num_rows <= 32
}

/// Compute `num_splits` for BitDecoding-FP4 decode (`Q_SUB = 8`, 2 CTAs/SM target),
/// capped to fit within the existing `splitk_workspace` slot budget.
#[inline]
#[must_use]
pub(crate) fn compute_bitfp4_decode_splits(
    num_q_heads: u32,
    num_kv_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
    sm_count: u32,
    max_splits_cap: u32,
) -> u32 {
    if num_kv_heads == 0 || num_seqs == 0 || max_seq_len_host <= 64 || max_splits_cap <= 1 {
        return 1;
    }
    let gqa_ratio = (num_q_heads / num_kv_heads).max(1);
    let q_groups_per_kv = gqa_ratio.div_ceil(BITFP4_DECODE_Q_SUB);
    let base_ctas = num_kv_heads
        .saturating_mul(q_groups_per_kv)
        .saturating_mul(num_seqs);
    if base_ctas == 0 {
        return 1;
    }
    // Target 2 CTAs/SM (SMEM is 39.0 KiB <= 49.5 KiB on GB10 & SM120),
    // with at least 1 full B_k=64 tile (4 blocks) per split.
    let target_ctas = sm_count.max(1).saturating_mul(2);
    let occupancy_splits = target_ctas.div_ceil(base_ctas);
    let tile_splits = max_seq_len_host.div_ceil(256).max(1);
    occupancy_splits.min(tile_splits).clamp(1, max_splits_cap)
}

/// Compute `num_splits` for BitDecoding-FP4 `K = gamma` verify (`Q_SUB = 16`).
#[inline]
#[must_use]
pub(crate) fn compute_bitfp4_kgamma_splits(
    num_q_heads: u32,
    num_kv_heads: u32,
    num_qtile: u32,
    max_seq_len_host: u32,
    sm_count: u32,
    max_splits_cap: u32,
) -> u32 {
    if num_kv_heads == 0 || num_qtile == 0 || max_seq_len_host <= 128 || max_splits_cap <= 1 {
        return 1;
    }
    let gqa_ratio = (num_q_heads / num_kv_heads).max(1);
    let q_chunks = (num_qtile.saturating_mul(gqa_ratio)).div_ceil(BITFP4_KGAMMA_Q_SUB);
    let base_ctas = num_kv_heads.saturating_mul(q_chunks);
    if base_ctas == 0 {
        return 1;
    }
    let target_ctas = sm_count.max(1).saturating_mul(2);
    let occupancy_splits = target_ctas.div_ceil(base_ctas);
    let tile_splits = max_seq_len_host.div_ceil(256).max(1);
    occupancy_splits.min(tile_splits).clamp(1, max_splits_cap)
}

impl Qwen3AttentionLayer {
    /// Resolve whether this layer should execute the opt-in SM120/SM121
    /// BitDecoding-FP4 KV write/decode pipeline for `(head_dim, block_size)`.
    ///
    /// Returns `Some((&BitFp4KvKernels, BitFp4QuantFlags))` when active;
    /// returns `None` when disabled or unsupported (warning once if explicitly
    /// requested via `ATLAS_BITFP4_KV=1` on an eligible NVFP4 layer whose
    /// architecture or kernel symbols could not engage).
    pub(in crate::layers::qwen3_attention) fn active_bitfp4_kv(
        &self,
        head_dim: u32,
        block_size: u32,
    ) -> Option<(&BitFp4KvKernels, BitFp4QuantFlags)> {
        let requested = crate::layers::bitfp4_kv_enabled();
        if !requested {
            return None;
        }
        let razer_requested = crate::layers::bitfp4_razer_kv_enabled();
        let cc = atlas_core::device::active_compute_capability();
        match resolve_bitfp4_kv_route(
            requested,
            razer_requested,
            self.kv_dtype,
            head_dim,
            block_size,
            self.sliding_window.is_some(),
            cc,
            self.bitfp4_kv_kernels.as_ref(),
        ) {
            BitFp4KvRoute::Active { razer } => {
                let kernels = self.bitfp4_kv_kernels.as_ref()?;
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    tracing::info!(
                        razer = ?razer,
                        cc = ?cc,
                        "ENGAGED ATLAS_BITFP4_KV: native Blackwell SM120/SM121 FP4 Tensor Core \
                         paged KV decode (Dual-Residual Q, Transposed Softmax, Underflow-Aware P, \
                         block-16 WHT+4/6 K + transposed V^T)"
                    );
                });
                Some((kernels, BitFp4QuantFlags::new(razer)))
            }
            BitFp4KvRoute::MissingKernel => {
                static WARNED_MISSING: std::sync::Once = std::sync::Once::new();
                crate::layers::warn_kernel_fallback(
                    &WARNED_MISSING,
                    "ATLAS_BITFP4_KV=1",
                    "paged_decode_attn_bitfp4_sm120",
                    "NVFP4 KV cache stays on the legacy scalar-dequant paged decode path",
                );
                None
            }
            BitFp4KvRoute::UnsupportedArch => {
                static WARNED_ARCH: std::sync::Once = std::sync::Once::new();
                WARNED_ARCH.call_once(|| {
                    tracing::warn!(
                        cc = ?cc,
                        "ATLAS_BITFP4_KV=1 requires Blackwell SM120/SM121 (compute capability 12.x); \
                         falling back to legacy NVFP4 paged decode"
                    );
                });
                None
            }
            BitFp4KvRoute::Disabled | BitFp4KvRoute::Ineligible => None,
        }
    }
}

// ============================================================================
// CPU Reference Implementation & Numerical Verification Suite
// ============================================================================

#[cfg(test)]
pub(crate) mod reference {
    //! Exact CPU reference implementations mirroring every instruction in
    //! `kernels/gb10/common/paged_decode_attn_bitfp4_sm120.cu`.

    pub const HDIM: usize = 256;
    pub const BLOCK_SIZE: usize = 16;
    pub const UNDERFLOW_C: f32 = 8.0;
    pub const UNDERFLOW_TAU: f32 = 21.0 / 1024.0;
    pub const UNDERFLOW_TAU_OVER_C: f32 = 21.0 / 8192.0;

    /// Exact E2M1 decode matching hardware `e2m1` nibbles `0x0..=0xF`.
    #[must_use]
    pub const fn e2m1_decode(nib: u8) -> f32 {
        match nib & 0xF {
            0x0 => 0.0,
            0x1 => 0.5,
            0x2 => 1.0,
            0x3 => 1.5,
            0x4 => 2.0,
            0x5 => 3.0,
            0x6 => 4.0,
            0x7 => 6.0,
            0x8 => -0.0,
            0x9 => -0.5,
            0xA => -1.0,
            0xB => -1.5,
            0xC => -2.0,
            0xD => -3.0,
            0xE => -4.0,
            0xF => -6.0,
            _ => unreachable!(),
        }
    }

    /// Exact RaZeR-KV decode (`0x8` remapped to `+5.0` when `neg_sv == 0`, `-5.0` when `neg_sv == 1`).
    #[must_use]
    pub const fn razer_decode(nib: u8, neg_sv: u8) -> f32 {
        if (nib & 0xF) == 0x8 {
            if (neg_sv & 1) != 0 { -5.0 } else { 5.0 }
        } else {
            e2m1_decode(nib)
        }
    }

    /// Exact round-to-nearest-even E2M1 quantization (hardware `cvt.rn.satfinite.e2m1x2.f32`).
    #[must_use]
    pub fn e2m1_quant(x: f32) -> u8 {
        let sign = if x.is_sign_negative() && x != 0.0 {
            8u8
        } else {
            0u8
        };
        let a = x.abs();
        // Midpoints with tie-to-even (0.0, 1.0, 2.0, 4.0, 6.0 are even mantissa codes).
        let mag = if a <= 0.25 {
            0u8
        } else if a < 0.75 {
            1u8
        } else if a <= 1.25 {
            2u8
        } else if a < 1.75 {
            3u8
        } else if a <= 2.5 {
            4u8
        } else if a < 3.5 {
            5u8
        } else if a <= 5.0 {
            6u8
        } else {
            7u8
        };
        if mag == 0 { 0 } else { sign | mag }
    }

    /// Exact RaZeR-KV scalar quantization (emitting `0x8` only for `±5.0`, never `-0.0`).
    #[must_use]
    pub fn razer_quant(x: f32, neg_sv: u8) -> u8 {
        let sign = if x < 0.0 { 8u8 } else { 0u8 };
        let a = x.abs();
        let same_sign = (sign != 0) == ((neg_sv & 1) != 0);
        if same_sign && (4.5..=5.5).contains(&a) {
            return 0x8;
        }
        let mag = if a < 0.25 {
            0u8
        } else if a <= 0.75 {
            1u8
        } else if a < 1.25 {
            2u8
        } else if a <= 1.75 {
            3u8
        } else if a < 2.5 {
            4u8
        } else if a <= 3.5 {
            5u8
        } else if a < 5.0 {
            6u8
        } else {
            7u8
        };
        if mag == 0 { 0 } else { sign | mag }
    }

    /// Exact unsigned E4M3 (`0x00..=0x7E`, bit 7 always `0`, max `448.0`) decode.
    #[must_use]
    pub fn ue4m3_to_f32(code: u8) -> f32 {
        let c = code & 0x7F;
        if c == 0x7F {
            return f32::NAN;
        }
        let exp = (c >> 3) & 0xF;
        let mant = c & 0x7;
        if exp == 0 {
            (mant as f32) * 2.0f32.powi(-9)
        } else {
            (1.0 + (mant as f32) / 8.0) * 2.0f32.powi((exp as i32) - 7)
        }
    }

    /// Exact round-to-nearest-even conversion to finite UE4M3 (`0x00..=0x7E`).
    #[must_use]
    pub fn f32_to_ue4m3(x: f32) -> u8 {
        let clamped = x.clamp(0.0, 448.0);
        let mut best_code = 0u8;
        let mut best_diff = f32::INFINITY;
        for c in 0u8..=0x7Eu8 {
            let v = ue4m3_to_f32(c);
            let diff = (clamped - v).abs();
            if diff < best_diff || (diff == best_diff && (c & 1) == 0) {
                best_diff = diff;
                best_code = c;
            }
        }
        best_code
    }

    /// Exact mirror of the 2-op `-0.0` (`0x8`) -> `+0.0` (`0x0`) SWAR
    /// canonicalizer in `quant8_e2m1` / `quant2_e2m1`.
    #[must_use]
    pub const fn canonicalize_e2m1_neg_zero_u32(r: u32) -> u32 {
        let neg0 = r & !((r & 0x7777_7777).wrapping_add(0x7777_7777)) & 0x8888_8888;
        r ^ neg0
    }

    /// Exact mirror of `bitfp4::razer_swar_split_u32` (6-op branchless SWAR).
    #[must_use]
    pub const fn razer_swar_split_u32(x: u32, neg_sv: u32) -> (u32, u32) {
        let neg_mask = 0u32.wrapping_sub(neg_sv & 1);
        let lo3 = x & 0x7777_7777;
        let m = x & !(lo3.wrapping_add(0x7777_7777)) & 0x8888_8888;
        let m2 = m >> 2;
        let m4 = m >> 1;
        let x_main = (x ^ (m & !neg_mask)) | m4 | m2;
        let x_comp = (m & neg_mask) | m2;
        (x_main, x_comp)
    }

    /// Normalized 256-point Walsh-Hadamard transform (`H^T H = I`).
    pub fn wht256(vals: &mut [f32; HDIM]) {
        let mut stride = 1usize;
        while stride < HDIM {
            let mut i = 0usize;
            while i < HDIM {
                for j in 0..stride {
                    let a = vals[i + j];
                    let b = vals[i + j + stride];
                    vals[i + j] = a + b;
                    vals[i + j + stride] = a - b;
                }
                i += stride * 2;
            }
            stride <<= 1;
        }
        let norm = 1.0 / (HDIM as f32).sqrt();
        for v in vals.iter_mut() {
            *v *= norm;
        }
    }

    /// Single-FP4 quantization of a 16-element group using standard `amax / 6`.
    #[must_use]
    pub fn quantize_group16_nvfp4(group: &[f32; 16]) -> ([u8; 16], u8) {
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            return ([0u8; 16], 0u8);
        }
        let sc_u8 = f32_to_ue4m3(amax / 6.0);
        let sc = ue4m3_to_f32(sc_u8);
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = e2m1_quant(group[i] * inv);
        }
        (out, sc_u8)
    }

    /// Four-Over-Six (`{amax/6, amax/4}`) + optional RaZeR-KV (`±5.0`, bit 7)
    /// quantization of a 16-element micro-block.
    #[must_use]
    pub fn quantize_group16_bitfp4(
        group: &[f32; 16],
        four_over_six: bool,
        razer: bool,
    ) -> ([u8; 16], u8) {
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            return ([0u8; 16], 0u8);
        }

        let eval_e2m1 = |sc_u8: u8| -> ([u8; 16], f32) {
            let sc = ue4m3_to_f32(sc_u8);
            let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
            let mut nibs = [0u8; 16];
            let mut mse = 0.0f32;
            for i in 0..16 {
                nibs[i] = e2m1_quant(group[i] * inv);
                let d = group[i] - e2m1_decode(nibs[i]) * sc;
                mse += d * d;
            }
            (nibs, mse)
        };

        let s6 = f32_to_ue4m3(amax / 6.0);
        let (mut best_nibs, mut best_err) = eval_e2m1(s6);
        let mut best_sc = s6;

        if four_over_six {
            let s4 = f32_to_ue4m3(amax / 4.0);
            let (n4, e4) = eval_e2m1(s4);
            if e4 < best_err {
                best_err = e4;
                best_nibs = n4;
                best_sc = s4;
            }
        }

        if razer {
            let s5 = f32_to_ue4m3(amax / 5.0);
            for neg_sv in [0u8, 1u8] {
                for &sc_u8 in &[s6, s5] {
                    let sc = ue4m3_to_f32(sc_u8);
                    let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                    let mut nibs = [0u8; 16];
                    let mut mse = 0.0f32;
                    for i in 0..16 {
                        nibs[i] = razer_quant(group[i] * inv, neg_sv);
                        let d = group[i] - razer_decode(nibs[i], neg_sv) * sc;
                        mse += d * d;
                    }
                    if mse < best_err {
                        best_err = mse;
                        best_nibs = nibs;
                        best_sc = sc_u8 | (neg_sv << 7);
                    }
                }
            }
        }

        (best_nibs, best_sc)
    }

    /// Dual-Residual Query Quantization (BitDecoding-FP4 Eq. 3) with dynamic
    /// `256 / q_amax` normalization matching `paged_decode_bitfp4_core`.
    #[must_use]
    pub fn quantize_query_dual_residual(q: &[f32; HDIM]) -> [f32; HDIM] {
        let q_amax = q.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if q_amax <= 1e-30 {
            return [0.0; HDIM];
        }
        let norm_scale = 256.0 / q_amax;
        let inv_norm = q_amax / 256.0;
        let mut out = [0.0f32; HDIM];

        for g in 0..(HDIM / 16) {
            let mut blk = [0.0f32; 16];
            for i in 0..16 {
                blk[i] = q[g * 16 + i] * norm_scale;
            }
            let (n1, s1_u8) = quantize_group16_nvfp4(&blk);
            let s1 = ue4m3_to_f32(s1_u8);
            let mut res = [0.0f32; 16];
            for i in 0..16 {
                let q1_i = e2m1_decode(n1[i]) * s1;
                res[i] = blk[i] - q1_i;
            }
            let (n2, s2_u8) = quantize_group16_nvfp4(&res);
            let s2 = ue4m3_to_f32(s2_u8);
            for i in 0..16 {
                let q_hat = e2m1_decode(n1[i]) * s1 + e2m1_decode(n2[i]) * s2;
                out[g * 16 + i] = q_hat * inv_norm;
            }
        }
        out
    }

    /// Underflow-Aware Probability Scaling (BitDecoding-FP4 Eq. 4) for a 16-key block.
    /// Returns `(quantized_nibbles, ue4m3_scale, kappa, dequantized_c_p)`.
    #[must_use]
    pub fn quantize_prob_block16_underflow_aware(
        p_unscaled: &[f32; 16],
    ) -> ([u8; 16], u8, f32, [f32; 16]) {
        let p_max = p_unscaled.iter().fold(0.0f32, |m, &v| m.max(v));
        let p_max_c = UNDERFLOW_C * p_max;
        let kappa = if p_max_c <= UNDERFLOW_TAU { 2.0 } else { 6.0 };
        let sp_u8 = f32_to_ue4m3(p_max_c / kappa);
        let sp = ue4m3_to_f32(sp_u8);
        let inv = if sp > 0.0 { 1.0 / sp } else { 0.0 };
        let mut nibs = [0u8; 16];
        let mut deq_c_p = [0.0f32; 16];
        for i in 0..16 {
            let p_c = UNDERFLOW_C * p_unscaled[i];
            nibs[i] = e2m1_quant(p_c * inv);
            deq_c_p[i] = e2m1_decode(nibs[i]) * sp;
        }
        (nibs, sp_u8, kappa, deq_c_p)
    }

    /// Self-contained block-16 transposed `V^T` representation matching
    /// `promote_block16_bitfp4_v_transposed` (`[HDIM][8]` bytes + `[HDIM]` scales).
    #[derive(Clone)]
    pub struct Block16Vt {
        pub data: [[u8; 8]; HDIM],
        pub scales: [u8; HDIM],
        pub razer: bool,
    }

    impl Block16Vt {
        #[must_use]
        pub fn encode(tokens_v: &[[f32; HDIM]], four_over_six: bool, razer: bool) -> Self {
            assert!(tokens_v.len() <= BLOCK_SIZE);
            let mut data = [[0u8; 8]; HDIM];
            let mut scales = [0u8; HDIM];
            for d in 0..HDIM {
                let mut ch = [0.0f32; BLOCK_SIZE];
                for (t, tok) in tokens_v.iter().enumerate() {
                    ch[t] = tok[d];
                }
                let (nibs, sc) = quantize_group16_bitfp4(&ch, four_over_six, razer);
                scales[d] = sc;
                for pair in 0..8 {
                    data[d][pair] = (nibs[pair * 2] & 0xF) | ((nibs[pair * 2 + 1] & 0xF) << 4);
                }
            }
            Self {
                data,
                scales,
                razer,
            }
        }

        /// Incremental append into a partial block-16 `V^T` matching
        /// `promote_block16_bitfp4_v_transposed` (`first_off > 0`): decodes existing
        /// `[0..first_off)`, inserts `new_v` at `[first_off..first_off + new_v.len())`,
        /// zeros trailing unused slots, and re-quantizes all 16 slots.
        pub fn append_tokens(
            &mut self,
            first_off: usize,
            new_v: &[[f32; HDIM]],
            four_over_six: bool,
            razer: bool,
        ) {
            assert!(
                first_off + new_v.len() <= BLOCK_SIZE,
                "block-16 V^T append out of bounds: first_off={first_off}, len={}",
                new_v.len()
            );
            assert_eq!(self.razer, razer, "cannot mix RaZeR modes within one block");
            let mut next_data = [[0u8; 8]; HDIM];
            let mut next_scales = [0u8; HDIM];
            for d in 0..HDIM {
                let mut ch = [0.0f32; BLOCK_SIZE];
                if first_off > 0 {
                    let prev = self.decode_channel_via_swar(d);
                    ch[..first_off].copy_from_slice(&prev[..first_off]);
                }
                for (i, tok) in new_v.iter().enumerate() {
                    ch[first_off + i] = tok[d];
                }
                let (nibs, sc) = quantize_group16_bitfp4(&ch, four_over_six, razer);
                next_scales[d] = sc;
                for pair in 0..8 {
                    next_data[d][pair] = (nibs[pair * 2] & 0xF) | ((nibs[pair * 2 + 1] & 0xF) << 4);
                }
            }
            self.data = next_data;
            self.scales = next_scales;
        }

        /// Decode via exact SWAR RaZeR split (`main + comp`) when `self.razer` is true.
        #[must_use]
        pub fn decode_channel_via_swar(&self, d: usize) -> [f32; BLOCK_SIZE] {
            let sc_raw = self.scales[d];
            let neg_sv = u32::from((sc_raw >> 7) & 1);
            let sc = ue4m3_to_f32(sc_raw & 0x7F);
            let w0 = u32::from_le_bytes([
                self.data[d][0],
                self.data[d][1],
                self.data[d][2],
                self.data[d][3],
            ]);
            let w1 = u32::from_le_bytes([
                self.data[d][4],
                self.data[d][5],
                self.data[d][6],
                self.data[d][7],
            ]);
            let mut out = [0.0f32; BLOCK_SIZE];
            for (half_idx, w) in [w0, w1].into_iter().enumerate() {
                let (w_main, w_comp) = if self.razer {
                    razer_swar_split_u32(w, neg_sv)
                } else {
                    (w, 0u32)
                };
                for k in 0..8 {
                    let n_m = ((w_main >> (k * 4)) & 0xF) as u8;
                    let n_c = ((w_comp >> (k * 4)) & 0xF) as u8;
                    out[half_idx * 8 + k] = (e2m1_decode(n_m) + e2m1_decode(n_c)) * sc;
                }
            }
            out
        }
    }

    /// Reference model of the self-contained block-16 paged BitDecoding-FP4 KV cache
    /// (`reshape_and_cache_bitfp4_k_block16` + `promote_block16_bitfp4_v_transposed`).
    pub struct BitFp4PagedKvModel {
        pub k_hat: Vec<[f32; HDIM]>,
        pub vt_blocks: Vec<Block16Vt>,
        pub razer: bool,
    }

    impl BitFp4PagedKvModel {
        #[must_use]
        pub fn new(razer: bool) -> Self {
            Self {
                k_hat: Vec::new(),
                vt_blocks: Vec::new(),
                razer,
            }
        }

        /// Append a contiguous chunk of `(K, V)` tokens into the paged block-16 cache,
        /// handling partial boundary blocks via `Block16Vt::append_tokens` identically
        /// to `promote_block16_bitfp4_v_transposed`.
        pub fn append_chunk(
            &mut self,
            k_chunk: &[[f32; HDIM]],
            v_chunk: &[[f32; HDIM]],
            mu_k: Option<&[f32; HDIM]>,
            apply_wht: bool,
            four_over_six: bool,
        ) {
            assert_eq!(k_chunk.len(), v_chunk.len());
            let start_pos = self.k_hat.len();

            // 1. Token-wise K write (`reshape_and_cache_bitfp4_k_block16`).
            for k in k_chunk {
                let mut k_work = *k;
                if let Some(mu) = mu_k {
                    for d in 0..HDIM {
                        k_work[d] -= mu[d];
                    }
                }
                if apply_wht {
                    wht256(&mut k_work);
                }
                let mut k_rec = [0.0f32; HDIM];
                for g in 0..(HDIM / 16) {
                    let mut grp = [0.0f32; 16];
                    grp.copy_from_slice(&k_work[g * 16..(g + 1) * 16]);
                    let (nibs, sc_raw) = quantize_group16_bitfp4(&grp, four_over_six, self.razer);
                    let sc = ue4m3_to_f32(sc_raw & 0x7F);
                    let neg_sv = u32::from((sc_raw >> 7) & 1);
                    for half in 0..2 {
                        let mut word = 0u32;
                        for j in 0..8 {
                            word |= u32::from(nibs[half * 8 + j] & 0xF) << (j * 4);
                        }
                        if !self.razer {
                            word = canonicalize_e2m1_neg_zero_u32(word);
                        }
                        let (w_main, w_comp) = if self.razer {
                            razer_swar_split_u32(word, neg_sv)
                        } else {
                            (word, 0u32)
                        };
                        for j in 0..8 {
                            let nm = ((w_main >> (j * 4)) & 0xF) as u8;
                            let nc = ((w_comp >> (j * 4)) & 0xF) as u8;
                            k_rec[g * 16 + half * 8 + j] = (e2m1_decode(nm) + e2m1_decode(nc)) * sc;
                        }
                    }
                }
                self.k_hat.push(k_rec);
            }

            // 2. Contiguous block-run transposed V^T promote (`promote_block16_bitfp4_v_transposed`).
            let mut offset = 0usize;
            while offset < v_chunk.len() {
                let abs_pos = start_pos + offset;
                let blk_idx = abs_pos / BLOCK_SIZE;
                let tok_in_blk = abs_pos % BLOCK_SIZE;
                let run_len = (BLOCK_SIZE - tok_in_blk).min(v_chunk.len() - offset);
                let slice = &v_chunk[offset..offset + run_len];
                if tok_in_blk == 0 {
                    assert_eq!(blk_idx, self.vt_blocks.len());
                    self.vt_blocks
                        .push(Block16Vt::encode(slice, four_over_six, self.razer));
                } else {
                    self.vt_blocks[blk_idx].append_tokens(
                        tok_in_blk,
                        slice,
                        four_over_six,
                        self.razer,
                    );
                }
                offset += run_len;
            }
        }
    }

    /// Exact CPU mirror of `prefill_paged_bitfp4_core` (`B_q = 32`, `B_k = 64` = 4 blocks)
    /// supporting arbitrary `q_offset`, `q_len`, `kv_len`, causal masking, optional WHT,
    /// optional BF16 sink tokens (`0`, partial `4`, or full block-0 `16`), and SWAR RaZeR-KV.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn prefill_paged_bitfp4_reference(
        q_chunk: &[[f32; HDIM]],
        kv: &BitFp4PagedKvModel,
        q_offset: usize,
        kv_len: usize,
        inv_sqrt_d: f32,
        causal: bool,
        apply_wht: bool,
        sink_bf16: Option<(&[[f32; HDIM]], &[[f32; HDIM]], usize)>,
    ) -> Vec<[f32; HDIM]> {
        const TILE_K_BLOCKS: usize = 4;
        const NEG_INF: f32 = -1.0e30f32;
        let log2e = std::f32::consts::LOG2_E;
        let mut outputs = vec![[0.0f32; HDIM]; q_chunk.len()];

        let (k_sink, v_sink, num_sink) = match sink_bf16 {
            Some((ks, vs, n)) if n > 0 => (Some(ks), Some(vs), n),
            _ => (None, None, 0usize),
        };
        let sink_skip = num_sink;
        let first_fp4_blk = num_sink / BLOCK_SIZE;

        for (r, q_raw) in q_chunk.iter().enumerate() {
            let q_kv_lim = if causal {
                (q_offset + r + 1).min(kv_len)
            } else {
                kv_len
            };
            if q_kv_lim == 0 {
                continue;
            }

            let mut running_m = NEG_INF;
            let mut running_l = 0.0f32;
            let mut o_acc = [0.0f32; HDIM];

            // Prologue: optional BF16 sink attention on raw unrotated Q.
            if let (Some(ks), Some(vs)) = (k_sink, v_sink) {
                let sink_lim = num_sink.min(q_kv_lim);
                for ts in 0..sink_lim {
                    let dot: f32 = q_raw.iter().zip(ks[ts].iter()).map(|(a, b)| a * b).sum();
                    let s_log2 = dot * inv_sqrt_d * log2e;
                    let m_new = running_m.max(s_log2);
                    let a_old = if running_m > -1.0e29 {
                        (running_m - m_new).exp2()
                    } else {
                        0.0
                    };
                    let p_c = (s_log2 - m_new + 3.0).exp2();
                    running_l = running_l * a_old + p_c;
                    for d in 0..HDIM {
                        o_acc[d] = o_acc[d] * a_old + p_c * vs[ts][d];
                    }
                    running_m = m_new;
                }
            }

            let total_blocks = q_kv_lim.div_ceil(BLOCK_SIZE);
            let fp4_blocks = total_blocks.saturating_sub(first_fp4_blk);
            let num_tiles = fp4_blocks.div_ceil(TILE_K_BLOCKS);

            if num_tiles > 0 {
                let mut q_rot = *q_raw;
                if apply_wht {
                    wht256(&mut q_rot);
                }
                let q_dual = quantize_query_dual_residual(&q_rot);

                for tile in 0..num_tiles {
                    let tile_blk0 = first_fp4_blk + tile * TILE_K_BLOCKS;
                    let mut s_tile = [[NEG_INF; BLOCK_SIZE]; TILE_K_BLOCKS];
                    let mut m_blk = [NEG_INF; TILE_K_BLOCKS];
                    let mut mt = NEG_INF;

                    for b in 0..TILE_K_BLOCKS {
                        let blk_idx = tile_blk0 + b;
                        if blk_idx >= total_blocks {
                            continue;
                        }
                        for t_local in 0..BLOCK_SIZE {
                            let key_pos = blk_idx * BLOCK_SIZE + t_local;
                            if key_pos >= sink_skip && key_pos < q_kv_lim {
                                let dot: f32 = q_dual
                                    .iter()
                                    .zip(kv.k_hat[key_pos].iter())
                                    .map(|(a, k)| a * k)
                                    .sum();
                                let s_log2 = dot * inv_sqrt_d * log2e;
                                s_tile[b][t_local] = s_log2;
                                m_blk[b] = m_blk[b].max(s_log2);
                            }
                        }
                        mt = mt.max(m_blk[b]);
                    }

                    let m_new = running_m.max(mt);
                    let alpha = if running_m > -1.0e29 && m_new > -1.0e29 {
                        (running_m - m_new).exp2()
                    } else {
                        0.0
                    };
                    for val in &mut o_acc {
                        *val *= alpha;
                    }
                    let mut l_tile = 0.0f32;

                    for b in 0..TILE_K_BLOCKS {
                        let blk_idx = tile_blk0 + b;
                        if blk_idx >= total_blocks {
                            continue;
                        }
                        let mut p_unscaled = [0.0f32; BLOCK_SIZE];
                        for t_local in 0..BLOCK_SIZE {
                            let key_pos = blk_idx * BLOCK_SIZE + t_local;
                            if key_pos >= sink_skip && key_pos < q_kv_lim && m_new > -1.0e29 {
                                let p = (s_tile[b][t_local] - m_new).exp2();
                                p_unscaled[t_local] = p;
                                l_tile += UNDERFLOW_C * p;
                            }
                        }
                        let (_, _, _, p_c_hat) = quantize_prob_block16_underflow_aware(&p_unscaled);
                        for d in 0..HDIM {
                            let v_ch = kv.vt_blocks[blk_idx].decode_channel_via_swar(d);
                            for t_local in 0..BLOCK_SIZE {
                                o_acc[d] += p_c_hat[t_local] * v_ch[t_local];
                            }
                        }
                    }

                    running_l = running_l * alpha + l_tile;
                    running_m = m_new;
                }
            }

            if running_l > 0.0 {
                let inv_l = 1.0 / running_l;
                for d in 0..HDIM {
                    outputs[r][d] = o_acc[d] * inv_l;
                }
            }
        }

        outputs
    }

    /// Exact FP32 causal attention reference over `[0..seq_len)`.
    #[must_use]
    pub fn exact_causal_attention_fp32(
        q_all: &[[f32; HDIM]],
        k_all: &[[f32; HDIM]],
        v_all: &[[f32; HDIM]],
        inv_sqrt_d: f32,
    ) -> Vec<[f32; HDIM]> {
        assert_eq!(q_all.len(), k_all.len());
        assert_eq!(q_all.len(), v_all.len());
        let mut out = vec![[0.0f32; HDIM]; q_all.len()];
        for (i, q) in q_all.iter().enumerate() {
            let mut m = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(i + 1);
            for k in &k_all[..=i] {
                let s = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d;
                m = m.max(s);
                scores.push(s);
            }
            let mut l = 0.0f32;
            for (t, &s) in scores.iter().enumerate() {
                let p = (s - m).exp();
                l += p;
                for d in 0..HDIM {
                    out[i][d] += p * v_all[t][d];
                }
            }
            for d in 0..HDIM {
                out[i][d] /= l;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;
    use spark_runtime::gpu::KernelHandle;

    fn lcg_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((*state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        u * 2.0 - 1.0
    }

    fn pseudo_gaussian(state: &mut u64) -> f32 {
        // Sum of 6 uniforms approximates N(0, 1) cleanly and deterministically.
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += lcg_f32(state);
        }
        s * std::f32::consts::FRAC_1_SQRT_2
    }

    #[test]
    fn swar_razer_split_exhaustive_all_32_cases_and_no_cross_nibble_carry() {
        // 1. Exhaustive 32-case check: every 4-bit nibble (0..16) x both neg_sv (0, 1).
        for neg_sv in 0u32..=1u32 {
            for nib in 0u8..16u8 {
                let packed = (nib as u32) * 0x1111_1111u32;
                let (w_main, w_comp) = razer_swar_split_u32(packed, neg_sv);
                let expected = razer_decode(nib, neg_sv as u8);
                for slot in 0..8 {
                    let nm = ((w_main >> (slot * 4)) & 0xF) as u8;
                    let nc = ((w_comp >> (slot * 4)) & 0xF) as u8;
                    let reconstructed = e2m1_decode(nm) + e2m1_decode(nc);
                    assert_eq!(
                        reconstructed, expected,
                        "nib=0x{nib:X} neg_sv={neg_sv} slot={slot}: main=0x{nm:X} comp=0x{nc:X}"
                    );
                    if nib == 0x8 {
                        if neg_sv == 0 {
                            assert_eq!((nm, nc), (0x6, 0x2), "+5.0 must split into +4.0 + +1.0");
                        } else {
                            assert_eq!((nm, nc), (0xE, 0xA), "-5.0 must split into -4.0 + -1.0");
                        }
                    } else {
                        assert_eq!(nm, nib, "non-0x8 nibble must remain intact in main");
                        assert_eq!(nc, 0x0, "non-0x8 nibble must produce +0.0 in comp");
                    }
                }
            }
        }

        // 2. Adversarial mixed-nibble words verifying ZERO cross-nibble carry.
        let patterns = [
            0x8F8F_8F8Fu32,
            0xF8F8_F8F8u32,
            0x8787_8787u32,
            0x7878_7878u32,
            0x089A_BCDEu32,
            0x1234_5678u32,
            0x8081_8E8Fu32,
        ];
        for neg_sv in 0u32..=1u32 {
            for &word in &patterns {
                let (w_main, w_comp) = razer_swar_split_u32(word, neg_sv);
                for slot in 0..8 {
                    let nib = ((word >> (slot * 4)) & 0xF) as u8;
                    let nm = ((w_main >> (slot * 4)) & 0xF) as u8;
                    let nc = ((w_comp >> (slot * 4)) & 0xF) as u8;
                    assert_eq!(
                        e2m1_decode(nm) + e2m1_decode(nc),
                        razer_decode(nib, neg_sv as u8),
                        "mixed word=0x{word:08X} neg_sv={neg_sv} slot={slot}"
                    );
                }
            }
        }

        // 3. Hardware `-0.0` (0x8) -> `+0.0` (0x0) SWAR canonicalizer: exhaustive + mixed.
        for nib in 0u8..16u8 {
            let packed = (nib as u32) * 0x1111_1111u32;
            let canon = canonicalize_e2m1_neg_zero_u32(packed);
            let want = if nib == 0x8 { 0x0u8 } else { nib };
            assert_eq!(canon, (want as u32) * 0x1111_1111u32, "nib=0x{nib:X}");
        }
        for &word in &patterns {
            let canon = canonicalize_e2m1_neg_zero_u32(word);
            for slot in 0..8 {
                let nib = ((word >> (slot * 4)) & 0xF) as u8;
                let got = ((canon >> (slot * 4)) & 0xF) as u8;
                let want = if nib == 0x8 { 0x0u8 } else { nib };
                assert_eq!(got, want, "canon word=0x{word:08X} slot={slot}");
            }
            // After canonicalization, running RaZeR split must NEVER turn a
            // canonicalized vanilla/4o6 word into ±5.0 (comp must be identically 0).
            for neg_sv in 0u32..=1u32 {
                let (w_main, w_comp) = razer_swar_split_u32(canon, neg_sv);
                assert_eq!(w_main, canon);
                assert_eq!(w_comp, 0u32);
            }
        }
    }

    #[test]
    fn ue4m3_bit7_razer_flag_packing_lossless_across_all_127_values() {
        for code in 0u8..=0x7Eu8 {
            let val = ue4m3_to_f32(code);
            assert!(val.is_finite() && val >= 0.0);
            for neg_sv in [0u8, 1u8] {
                let packed = code | (neg_sv << 7);
                assert_eq!((packed >> 7) & 1, neg_sv);
                assert_eq!(packed & 0x7F, code);
                assert_eq!(ue4m3_to_f32(packed & 0x7F), val);
            }
        }
        // Packed 4-byte SFA word clear mask preserves all 4 scales identically.
        let sfa_raw = u32::from_le_bytes([0x80 | 0x38, 0x42, 0x80 | 0x7E, 0x01]);
        let sfa_clean = sfa_raw & 0x7F7F_7F7F;
        assert_eq!(sfa_clean.to_le_bytes(), [0x38, 0x42, 0x7E, 0x01]);
    }

    #[test]
    fn dual_residual_query_cuts_logit_mse_by_over_95_percent() {
        let mut rng = 0xDEAD_BEEF_1234_5678u64;
        let mut q = [0.0f32; HDIM];
        for v in &mut q {
            *v = pseudo_gaussian(&mut rng);
        }
        // Inject realistic RoPE/QK-norm magnitude spike.
        q[17] = 6.5;
        q[143] = -5.8;

        // Single-FP4 Q reconstruction (with same 256/amax normalization).
        let q_amax = q.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let norm_s = 256.0 / q_amax;
        let inv_s = q_amax / 256.0;
        let mut q_single = [0.0f32; HDIM];
        for g in 0..(HDIM / 16) {
            let mut blk = [0.0f32; 16];
            for i in 0..16 {
                blk[i] = q[g * 16 + i] * norm_s;
            }
            let (n1, s1_u8) = quantize_group16_nvfp4(&blk);
            let s1 = ue4m3_to_f32(s1_u8);
            for i in 0..16 {
                q_single[g * 16 + i] = e2m1_decode(n1[i]) * s1 * inv_s;
            }
        }
        let q_dual = quantize_query_dual_residual(&q);

        // Measure query-induced logit error across 128 keys: `(q_hat - q) · k_i`.
        let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();
        let mut mse_single = 0.0f64;
        let mut mse_dual = 0.0f64;
        for _ in 0..128 {
            let mut k = [0.0f32; HDIM];
            for v in &mut k {
                *v = pseudo_gaussian(&mut rng);
            }
            let exact: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d;
            let s1: f32 = q_single
                .iter()
                .zip(k.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * inv_sqrt_d;
            let s2: f32 = q_dual.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d;
            mse_single += f64::from(s1 - exact).powi(2);
            mse_dual += f64::from(s2 - exact).powi(2);
        }
        let reduction = 1.0 - (mse_dual / mse_single);
        assert!(
            reduction > 0.95,
            "Dual-Residual Q must cut query logit MSE by >95%: mse_single={mse_single:.6e}, \
             mse_dual={mse_dual:.6e}, reduction={:.2}%",
            reduction * 100.0
        );
    }

    #[test]
    fn underflow_aware_probability_scaling_preserves_2_neg_12_regime() {
        // Verify exact kappa transition boundary at tau/c = 21/8192.
        let mut below = [0.0f32; 16];
        below[0] = UNDERFLOW_TAU_OVER_C * 0.99;
        let (_, _, kappa_below, _) = quantize_prob_block16_underflow_aware(&below);
        assert_eq!(kappa_below, 2.0);

        let mut above = [0.0f32; 16];
        above[0] = UNDERFLOW_TAU_OVER_C * 1.01;
        let (_, _, kappa_above, _) = quantize_prob_block16_underflow_aware(&above);
        assert_eq!(kappa_above, 6.0);

        // Verify tiny long-context probability blocks down to ~2^-12:
        // Vanilla `amax/6` underflows to EXACT 0.0 (100% loss), while
        // Underflow-Aware Scaling (`c=8, kappa=2`) stays strictly non-zero and accurate.
        for &p_peak in &[
            1.25 * 2.0f32.powi(-12), // right above the 2^-12 floor
            2.0f32.powi(-11),
            2.0f32.powi(-10),
            2.0f32.powi(-9),
            2.0f32.powi(-8),
        ] {
            let mut block = [0.0f32; 16];
            for (i, slot) in block.iter_mut().enumerate() {
                *slot = p_peak * (1.0 - (i as f32) * 0.05);
            }

            // Standard `amax / 6` scale underflows to 0 for all p_peak <= 6 * 2^-10 = 0.005859375.
            let (_, std_sc_u8) = quantize_group16_nvfp4(&block);
            assert_eq!(
                std_sc_u8, 0,
                "standard amax/6 should underflow to 0 at p_peak={p_peak:e}"
            );

            // Underflow-aware scaling (`c=8`, `kappa=2` iff `p_peak <= 21/8192` else `6`) stays non-zero!
            let (_, ua_sc_u8, kappa, deq_c_p) = quantize_prob_block16_underflow_aware(&block);
            let expected_kappa = if p_peak <= UNDERFLOW_TAU_OVER_C {
                2.0
            } else {
                6.0
            };
            assert_eq!(kappa, expected_kappa, "p_peak={p_peak:e}");
            assert!(
                ua_sc_u8 > 0,
                "underflow-aware scale must remain non-zero at p_peak={p_peak:e}"
            );
            let recovered_peak = deq_c_p[0] / UNDERFLOW_C;
            let rel_err = (recovered_peak - p_peak).abs() / p_peak;
            assert!(
                rel_err < 0.65,
                "recovered peak too far at p_peak={p_peak:e}: recovered={recovered_peak:e}, rel_err={rel_err:.3}"
            );
        }
    }

    #[test]
    fn block16_k_wht_4o6_razer_and_transposed_vt_roundtrip_accuracy() {
        let mut rng = 0x1122_3344_5566_7788u64;

        // 1. WHT orthogonality & inner-product preservation.
        let mut x = [0.0f32; HDIM];
        let mut y = [0.0f32; HDIM];
        for i in 0..HDIM {
            x[i] = pseudo_gaussian(&mut rng);
            y[i] = pseudo_gaussian(&mut rng);
        }
        let dot_before: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let mut hx = x;
        let mut hy = y;
        wht256(&mut hx);
        wht256(&mut hy);
        let dot_after: f32 = hx.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
        assert!((dot_before - dot_after).abs() < 1e-3);
        let mut hhx = hx;
        wht256(&mut hhx);
        let max_inv_err = x
            .iter()
            .zip(hhx.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_inv_err < 1e-4, "H(H(x)) != x: max_err={max_inv_err}");

        // 2. Block-16 V^T roundtrip + RaZeR + 4/6 MSE improvement over vanilla NVFP4.
        let mut v_block = [[0.0f32; HDIM]; BLOCK_SIZE];
        for tok in &mut v_block {
            for val in tok.iter_mut() {
                *val = pseudo_gaussian(&mut rng);
            }
        }
        let vt_vanilla = Block16Vt::encode(&v_block, false, false);
        let vt_bitfp4 = Block16Vt::encode(&v_block, true, false);
        let vt_razer = Block16Vt::encode(&v_block, true, true);

        // Verify exact Flux NVFP4 pool byte footprint: 256*8 = 2048 data + 256 scale = 2304 B/head.
        assert_eq!(
            std::mem::size_of_val(&vt_razer.data),
            HDIM * (BLOCK_SIZE / 2)
        );
        assert_eq!(std::mem::size_of_val(&vt_razer.scales), HDIM);

        let mse_of = |vt: &Block16Vt| -> f64 {
            let mut err = 0.0f64;
            for d in 0..HDIM {
                let rec = vt.decode_channel_via_swar(d);
                for t in 0..BLOCK_SIZE {
                    err += f64::from(rec[t] - v_block[t][d]).powi(2);
                }
            }
            err / ((HDIM * BLOCK_SIZE) as f64)
        };
        let m_van = mse_of(&vt_vanilla);
        let m_4o6 = mse_of(&vt_bitfp4);
        let m_raz = mse_of(&vt_razer);
        assert!(
            m_4o6 <= m_van && m_raz < m_4o6,
            "Expected strict MSE hierarchy RaZeR < 4/6 <= Vanilla: van={m_van:.6}, 4o6={m_4o6:.6}, razer={m_raz:.6}"
        );
    }

    #[test]
    fn end_to_end_bitfp4_attention_matches_fp32_reference_and_splitk_invariance() {
        let mut rng = 0xCAFE_BABE_9876_5432u64;
        let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();
        let log2e = std::f32::consts::LOG2_E;

        for &seq_len in &[16usize, 45usize, 128usize] {
            let mut q = [0.0f32; HDIM];
            for v in &mut q {
                *v = pseudo_gaussian(&mut rng);
            }
            // Persistent channel outlier in K (simulating Qwen3.8 RoPE channel spikes).
            let mut mu_k = [0.0f32; HDIM];
            mu_k[3] = 9.0;
            mu_k[71] = -7.5;
            mu_k[200] = 6.0;

            let mut k_seq = vec![[0.0f32; HDIM]; seq_len];
            let mut v_seq = vec![[0.0f32; HDIM]; seq_len];
            for t in 0..seq_len {
                for d in 0..HDIM {
                    k_seq[t][d] = mu_k[d] + pseudo_gaussian(&mut rng);
                    v_seq[t][d] = pseudo_gaussian(&mut rng);
                }
            }
            // Inject realistic needle/retrieval alignment on two context positions
            // so attention has both sharp peaks and long-tail background probabilities.
            for &needle in &[0usize, seq_len / 2] {
                for d in 0..HDIM {
                    k_seq[needle][d] = mu_k[d] + 0.65 * q[d] + 0.35 * pseudo_gaussian(&mut rng);
                }
            }

            // Exact FP32 reference attention.
            let mut scores_ref = vec![0.0f32; seq_len];
            let mut m_ref = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let s = q
                    .iter()
                    .zip(k_seq[t].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * inv_sqrt_d;
                scores_ref[t] = s;
                m_ref = m_ref.max(s);
            }
            let mut l_ref = 0.0f32;
            let mut o_ref = [0.0f32; HDIM];
            for t in 0..seq_len {
                let p = (scores_ref[t] - m_ref).exp();
                l_ref += p;
                for d in 0..HDIM {
                    o_ref[d] += p * v_seq[t][d];
                }
            }
            for v in &mut o_ref {
                *v /= l_ref;
            }

            // Simulate full BitDecoding-FP4 (+ RaZeR-KV) decode pipeline across splits.
            for &razer in &[false, true] {
                let mut q_rot = q;
                wht256(&mut q_rot);
                let q_dual = quantize_query_dual_residual(&q_rot);

                let num_blocks = seq_len.div_ceil(BLOCK_SIZE);
                let mut k_blocks_hat = Vec::with_capacity(seq_len);
                for k in &k_seq {
                    let mut k_sm = [0.0f32; HDIM];
                    for d in 0..HDIM {
                        k_sm[d] = k[d] - mu_k[d];
                    }
                    wht256(&mut k_sm);
                    let mut k_hat = [0.0f32; HDIM];
                    for g in 0..(HDIM / 16) {
                        let mut grp = [0.0f32; 16];
                        grp.copy_from_slice(&k_sm[g * 16..(g + 1) * 16]);
                        let (nibs, sc_raw) = quantize_group16_bitfp4(&grp, true, razer);
                        let sc = ue4m3_to_f32(sc_raw & 0x7F);
                        let neg_sv = (sc_raw >> 7) & 1;
                        for i in 0..16 {
                            k_hat[g * 16 + i] = if razer {
                                razer_decode(nibs[i], neg_sv) * sc
                            } else {
                                e2m1_decode(nibs[i]) * sc
                            };
                        }
                    }
                    k_blocks_hat.push(k_hat);
                }

                let mut vt_blocks = Vec::with_capacity(num_blocks);
                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(seq_len);
                    vt_blocks.push(Block16Vt::encode(&v_seq[start..end], true, razer));
                }

                let run_with_splits = |num_splits: usize| -> [f32; HDIM] {
                    let blks_per_split = num_blocks.div_ceil(num_splits);
                    let mut partials: Vec<([f32; HDIM], f32, f32)> = Vec::with_capacity(num_splits);
                    for s in 0..num_splits {
                        let b0 = s * blks_per_split;
                        let b1 = (b0 + blks_per_split).min(num_blocks);
                        let mut m = -1.0e30f32;
                        let mut l = 0.0f32;
                        let mut o = [0.0f32; HDIM];
                        for b in b0..b1 {
                            let mut s_blk = [-1.0e30f32; BLOCK_SIZE];
                            let mut mb = -1.0e30f32;
                            for t_local in 0..BLOCK_SIZE {
                                let pos = b * BLOCK_SIZE + t_local;
                                if pos < seq_len {
                                    let dot: f32 = q_dual
                                        .iter()
                                        .zip(k_blocks_hat[pos].iter())
                                        .map(|(a, k)| a * k)
                                        .sum();
                                    let s_log2 = dot * inv_sqrt_d * log2e;
                                    s_blk[t_local] = s_log2;
                                    mb = mb.max(s_log2);
                                }
                            }
                            let m_new = m.max(mb);
                            let alpha = if m > -1.0e29 { (m - m_new).exp2() } else { 0.0 };
                            let mut p_unscaled = [0.0f32; BLOCK_SIZE];
                            let mut l_b = 0.0f32;
                            for t_local in 0..BLOCK_SIZE {
                                if b * BLOCK_SIZE + t_local < seq_len {
                                    let p = (s_blk[t_local] - m_new).exp2();
                                    p_unscaled[t_local] = p;
                                    l_b += UNDERFLOW_C * p;
                                }
                            }
                            let (_, _, _, p_c_hat) =
                                quantize_prob_block16_underflow_aware(&p_unscaled);
                            for d in 0..HDIM {
                                o[d] *= alpha;
                                let v_ch = vt_blocks[b].decode_channel_via_swar(d);
                                for t_local in 0..BLOCK_SIZE {
                                    o[d] += p_c_hat[t_local] * v_ch[t_local];
                                }
                            }
                            l = l * alpha + l_b;
                            m = m_new;
                        }
                        partials.push((o, m, l));
                    }

                    // Base-2 split-K reduce (mirroring `paged_decode_attn_bitfp4_reduce`).
                    let mut m_g = -1.0e30f32;
                    let mut l_g = 0.0f32;
                    let mut o_g = [0.0f32; HDIM];
                    for (os, ms, ls) in partials {
                        if ls <= 0.0 {
                            continue;
                        }
                        let m_new = m_g.max(ms);
                        let sc_me = if l_g > 0.0 { (m_g - m_new).exp2() } else { 0.0 };
                        let sc_s = (ms - m_new).exp2();
                        for d in 0..HDIM {
                            o_g[d] = o_g[d] * sc_me + os[d] * sc_s;
                        }
                        l_g = l_g * sc_me + ls * sc_s;
                        m_g = m_new;
                    }
                    for v in &mut o_g {
                        *v /= l_g;
                    }
                    o_g
                };

                let o_s1 = run_with_splits(1);
                let o_s4 = run_with_splits(4); // Includes empty splits at seq_len=16!

                // Both single-split and multi-split outputs must closely track exact FP32 reference.
                for (label, o_cand) in [("s1", &o_s1), ("s4", &o_s4)] {
                    let dot: f64 = o_ref
                        .iter()
                        .zip(o_cand.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let nr: f64 = o_ref
                        .iter()
                        .map(|&a| f64::from(a).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let nc: f64 = o_cand
                        .iter()
                        .map(|&a| f64::from(a).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let cos_sim = dot / (nr * nc);
                    assert!(
                        cos_sim > 0.990,
                        "seq_len={seq_len} razer={razer} ({label}): cosine similarity vs FP32 reference too low ({cos_sim:.5})"
                    );
                }

                // Single-split vs 4-split cosine agreement (different split-local `m` shifts
                // E2M1 probability rounding bins slightly while preserving the attention vector).
                let dot_14: f64 = o_s1
                    .iter()
                    .zip(o_s4.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let n1: f64 = o_s1
                    .iter()
                    .map(|&a| f64::from(a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let n4: f64 = o_s4
                    .iter()
                    .map(|&a| f64::from(a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    dot_14 / (n1 * n4) > 0.990,
                    "seq_len={seq_len} razer={razer}: s1 vs s4 cosine agreement too low ({:.5})",
                    dot_14 / (n1 * n4)
                );
            }

            // Exact algebraic invariance of the base-2 LSE split-K reducer (including
            // empty splits with `l == 0`) when probability weights are unquantized FP32.
            let num_blocks = seq_len.div_ceil(BLOCK_SIZE);
            let blks_per_split = num_blocks.div_ceil(4);
            let mut m_g = -1.0e30f32;
            let mut l_g = 0.0f32;
            let mut o_g = [0.0f32; HDIM];
            for s in 0..4usize {
                let b0 = s * blks_per_split;
                let b1 = (b0 + blks_per_split).min(num_blocks);
                let mut ms = -1.0e30f32;
                let mut ls = 0.0f32;
                let mut os = [0.0f32; HDIM];
                for pos in (b0 * BLOCK_SIZE)..(b1 * BLOCK_SIZE).min(seq_len) {
                    let s_log2 = scores_ref[pos] * log2e;
                    let m_new = ms.max(s_log2);
                    let a_old = if ms > -1.0e29 {
                        (ms - m_new).exp2()
                    } else {
                        0.0
                    };
                    let p_c = (s_log2 - m_new + 3.0).exp2();
                    ls = ls * a_old + p_c;
                    for d in 0..HDIM {
                        os[d] = os[d] * a_old + p_c * v_seq[pos][d];
                    }
                    ms = m_new;
                }
                if ls <= 0.0 {
                    continue;
                }
                let m_new = m_g.max(ms);
                let sc_me = if l_g > 0.0 { (m_g - m_new).exp2() } else { 0.0 };
                let sc_s = (ms - m_new).exp2();
                for d in 0..HDIM {
                    o_g[d] = o_g[d] * sc_me + os[d] * sc_s;
                }
                l_g = l_g * sc_me + ls * sc_s;
                m_g = m_new;
            }
            for v in &mut o_g {
                *v /= l_g;
            }
            let max_reduce_err = o_ref
                .iter()
                .zip(o_g.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_reduce_err < 1e-5,
                "seq_len={seq_len}: exact base-2 split-K LSE reduce drift {max_reduce_err:e}"
            );
        }
    }

    #[test]
    fn route_admission_guards_sm120_sm121_and_defaults_off() {
        let complete = BitFp4KvKernels {
            decode: KernelHandle(1),
            decode_razer: KernelHandle(2),
            kgamma: KernelHandle(3),
            kgamma_razer: KernelHandle(4),
            reduce: KernelHandle(5),
            write_k: KernelHandle(6),
            promote_v: KernelHandle(7),
            prefill_paged: KernelHandle(8),
            prefill_paged_razer: KernelHandle(9),
            prefill_paged_batched: KernelHandle(10),
            prefill_paged_batched_razer: KernelHandle(11),
        };
        let base_only = BitFp4KvKernels {
            decode_razer: KernelHandle(0),
            kgamma_razer: KernelHandle(0),
            prefill_paged_razer: KernelHandle(0),
            prefill_paged_batched_razer: KernelHandle(0),
            ..complete
        };
        let missing_prefill = BitFp4KvKernels {
            prefill_paged: KernelHandle(0),
            ..complete
        };

        // Default OFF -> always Disabled regardless of arch/kernels.
        assert_eq!(
            resolve_bitfp4_kv_route(
                false,
                false,
                KvCacheDtype::Nvfp4,
                256,
                16,
                false,
                Some((12, 1)),
                Some(&complete),
            ),
            BitFp4KvRoute::Disabled
        );

        // Missing chunked prefill kernel fails closed before writing BitFP4 KV.
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                false,
                KvCacheDtype::Nvfp4,
                256,
                16,
                false,
                Some((12, 1)),
                Some(&missing_prefill),
            ),
            BitFp4KvRoute::MissingKernel
        );

        // Non-Blackwell-12x (SM103 / SM90 / unset) fails closed.
        for bad_cc in [None, Some((9, 0)), Some((10, 3))] {
            assert_eq!(
                resolve_bitfp4_kv_route(
                    true,
                    false,
                    KvCacheDtype::Nvfp4,
                    256,
                    16,
                    false,
                    bad_cc,
                    Some(&complete),
                ),
                BitFp4KvRoute::UnsupportedArch,
                "cc={bad_cc:?}"
            );
        }

        // SM120 and SM121 engage cleanly for both standard and RaZeR modes.
        for cc in [Some((12, 0)), Some((12, 1))] {
            assert_eq!(
                resolve_bitfp4_kv_route(
                    true,
                    false,
                    KvCacheDtype::Nvfp4,
                    256,
                    16,
                    false,
                    cc,
                    Some(&complete),
                ),
                BitFp4KvRoute::Active {
                    razer: BitFp4RazerMode::Disabled
                }
            );
            assert_eq!(
                resolve_bitfp4_kv_route(
                    true,
                    true,
                    KvCacheDtype::Nvfp4,
                    256,
                    16,
                    false,
                    cc,
                    Some(&complete),
                ),
                BitFp4KvRoute::Active {
                    razer: BitFp4RazerMode::Enabled
                }
            );
        }

        // Requesting RaZeR when RaZeR symbols are missing fails closed.
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                true,
                KvCacheDtype::Nvfp4,
                256,
                16,
                false,
                Some((12, 1)),
                Some(&base_only),
            ),
            BitFp4KvRoute::MissingKernel
        );

        // Non-NVFP4 / wrong head_dim / wrong block_size / sliding window -> Ineligible.
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                false,
                KvCacheDtype::Bf16,
                256,
                16,
                false,
                Some((12, 1)),
                Some(&complete),
            ),
            BitFp4KvRoute::Ineligible
        );
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                false,
                KvCacheDtype::Nvfp4,
                128,
                16,
                false,
                Some((12, 1)),
                Some(&complete),
            ),
            BitFp4KvRoute::Ineligible
        );
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                false,
                KvCacheDtype::Nvfp4,
                256,
                32,
                false,
                Some((12, 1)),
                Some(&complete),
            ),
            BitFp4KvRoute::Ineligible
        );
        assert_eq!(
            resolve_bitfp4_kv_route(
                true,
                false,
                KvCacheDtype::Nvfp4,
                256,
                16,
                true,
                Some((12, 1)),
                Some(&complete),
            ),
            BitFp4KvRoute::Ineligible
        );
    }

    #[test]
    fn splitk_geometry_targets_2_cta_per_sm_within_cap() {
        // Qwen3.8-27B (24 Q / 4 KV, GQA=6) on 48-SM GB10 at ctx=8192:
        // base_ctas = 4 -> target 96 CTAs -> 24 splits (fits cap 48).
        assert_eq!(compute_bitfp4_decode_splits(24, 4, 1, 8192, 48, 48), 24);
        // Short context (ctx=64) stays single-split.
        assert_eq!(compute_bitfp4_decode_splits(24, 4, 1, 64, 48, 48), 1);
        // High concurrency (C=24 -> 96 base CTAs) needs only 1 split.
        assert_eq!(compute_bitfp4_decode_splits(24, 4, 24, 8192, 48, 48), 1);
        // K=17 verify (102 queries/kv -> 7 chunks x 4 kv = 28 CTAs) -> 4 splits (112 CTAs).
        assert_eq!(compute_bitfp4_kgamma_splits(24, 4, 17, 8192, 48, 64), 4);
    }

    #[test]
    fn kgamma_dispatch_requires_shared_block_table_to_prevent_cross_tenant_reads() {
        // Single-sequence K=17 verify: all 17 rows share the identical block table.
        let shared_k17: Vec<Vec<u32>> = vec![vec![3, 7, 11]; 17];
        assert!(block_tables_all_shared(&shared_k17));
        assert!(bitfp4_kgamma_eligible(
            shared_k17.len(),
            block_tables_all_shared(&shared_k17)
        ));

        // Concurrent batched decode (C=2..32 independent user requests): distinct block tables.
        let distinct_c4: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]];
        assert!(!block_tables_all_shared(&distinct_c4));
        assert!(!bitfp4_kgamma_eligible(
            distinct_c4.len(),
            block_tables_all_shared(&distinct_c4)
        ));

        // Even a single differing trailing row disqualifies single-table kgamma.
        let mostly_shared: Vec<Vec<u32>> = vec![vec![1, 2], vec![1, 2], vec![1, 9]];
        assert!(!block_tables_all_shared(&mostly_shared));
        assert!(!bitfp4_kgamma_eligible(
            mostly_shared.len(),
            block_tables_all_shared(&mostly_shared)
        ));

        // Boundary row counts: n=1, n=0, n=33 stay on per-sequence decode even if shared.
        assert!(!block_tables_all_shared(&[]));
        assert!(!bitfp4_kgamma_eligible(1, true));
        assert!(bitfp4_kgamma_eligible(2, true));
        assert!(bitfp4_kgamma_eligible(32, true));
        assert!(!bitfp4_kgamma_eligible(33, true));
    }

    fn cosine_sim(a: &[f32; HDIM], b: &[f32; HDIM]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let na: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&y| f64::from(y).powi(2)).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn chunked_continuation_prefill_matches_single_shot_and_fp32_reference() {
        let mut rng = 0x51A7_E213_B00B_C0DEu64;
        let total_len = 113usize; // Non-multiple of 16, 32, and 64 (partial tail block & tile).
        let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();

        // Persistent channel outliers in K (simulating Qwen3.8 RoPE channel spikes).
        let mut mu_k = [0.0f32; HDIM];
        mu_k[5] = 7.5;
        mu_k[89] = -6.2;
        mu_k[211] = 5.4;

        let mut q_all = vec![[0.0f32; HDIM]; total_len];
        let mut k_all = vec![[0.0f32; HDIM]; total_len];
        let mut v_all = vec![[0.0f32; HDIM]; total_len];
        for t in 0..total_len {
            for d in 0..HDIM {
                q_all[t][d] = pseudo_gaussian(&mut rng);
                k_all[t][d] = mu_k[d] + pseudo_gaussian(&mut rng);
                v_all[t][d] = pseudo_gaussian(&mut rng);
            }
            // Inject local + long-range retrieval correlation for non-trivial softmax peaks.
            if t >= 12 {
                let target = if t % 3 == 0 { 3 } else { t - 9 };
                for d in 0..HDIM {
                    q_all[t][d] = 0.55 * (k_all[target][d] - mu_k[d]) + 0.65 * q_all[t][d];
                }
            }
        }

        let o_fp32 = exact_causal_attention_fp32(&q_all, &k_all, &v_all, inv_sqrt_d);

        for &razer in &[false, true] {
            // 1. Single-shot BitFP4 KV write + paged prefill over [0..113).
            let mut kv_single = BitFp4PagedKvModel::new(razer);
            kv_single.append_chunk(&k_all, &v_all, Some(&mu_k), true, true);
            let o_single = prefill_paged_bitfp4_reference(
                &q_all, &kv_single, 0, total_len, inv_sqrt_d, true, true, None,
            );

            // 2. Three non-16-aligned continuation chunks: [0..25), [25..73), [73..113).
            //    Crosses partial block-16 boundaries at pos=25 (block 1) and pos=73 (block 4),
            //    exercising incremental `V^T` re-promotion and a 1-token partial tail block (block 7).
            let chunk_bounds = [(0usize, 25usize), (25, 73), (73, total_len)];
            let mut kv_chunked = BitFp4PagedKvModel::new(razer);
            let mut o_chunked = Vec::with_capacity(total_len);
            for &(start, end) in &chunk_bounds {
                kv_chunked.append_chunk(
                    &k_all[start..end],
                    &v_all[start..end],
                    Some(&mu_k),
                    true,
                    true,
                );
                let out = prefill_paged_bitfp4_reference(
                    &q_all[start..end],
                    &kv_chunked,
                    start,
                    end,
                    inv_sqrt_d,
                    true,
                    true,
                    None,
                );
                o_chunked.extend_from_slice(&out);
            }
            assert_eq!(o_chunked.len(), total_len);
            assert_eq!(kv_chunked.vt_blocks.len(), total_len.div_ceil(BLOCK_SIZE));

            let mut sum_fp32_single = 0.0f64;
            let mut sum_fp32_chunked = 0.0f64;
            let mut sum_chunked_vs_single = 0.0f64;
            for t in 0..total_len {
                let c_single = cosine_sim(&o_fp32[t], &o_single[t]);
                let c_chunked = cosine_sim(&o_fp32[t], &o_chunked[t]);
                let c_equiv = cosine_sim(&o_single[t], &o_chunked[t]);
                assert!(
                    c_single > 0.975,
                    "razer={razer} row={t}: single-shot vs FP32 cosine too low ({c_single:.5})"
                );
                assert!(
                    c_chunked > 0.975,
                    "razer={razer} row={t}: chunked vs FP32 cosine too low ({c_chunked:.5})"
                );
                assert!(
                    c_equiv > 0.990,
                    "razer={razer} row={t}: chunked vs single-shot drift too high ({c_equiv:.5})"
                );
                sum_fp32_single += c_single;
                sum_fp32_chunked += c_chunked;
                sum_chunked_vs_single += c_equiv;
            }
            let mean_single = sum_fp32_single / (total_len as f64);
            let mean_chunked = sum_fp32_chunked / (total_len as f64);
            let mean_equiv = sum_chunked_vs_single / (total_len as f64);
            assert!(
                mean_single > 0.990 && mean_chunked > 0.990 && mean_equiv > 0.998,
                "razer={razer}: mean cosines single={mean_single:.5}, chunked={mean_chunked:.5}, equiv={mean_equiv:.5}"
            );
        }
    }

    #[test]
    fn chunked_prefill_causal_mask_blocks_adversarial_future_tokens_in_shared_block16() {
        let mut rng = 0xA11C_E001_55AA_33CCu64;
        let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();
        // Continuation chunk at q_offset = 20, q_len = 23 -> kv_len = 43.
        // Physical block 1 holds tokens [16..32), sharing past (<20), current, and future (>20)
        // tokens in the EXACT same block-16 V^T and B_k=64 tile.
        let q_offset = 20usize;
        let q_len = 23usize;
        let kv_len = q_offset + q_len;

        let mut q_chunk = vec![[0.0f32; HDIM]; q_len];
        let mut k_clean = vec![[0.0f32; HDIM]; kv_len];
        let mut v_clean = vec![[0.0f32; HDIM]; kv_len];
        for t in 0..kv_len {
            for d in 0..HDIM {
                k_clean[t][d] = pseudo_gaussian(&mut rng);
                v_clean[t][d] = pseudo_gaussian(&mut rng);
            }
        }
        for q in &mut q_chunk {
            for d in 0..HDIM {
                q[d] = pseudo_gaussian(&mut rng);
            }
        }

        // Poison keys strictly AFTER query row `probe_r` (abs pos = q_offset + probe_r)
        // inside the SAME physical block-16 (pos 26..31) and next block (pos 32..42).
        let probe_r = 5usize; // abs pos = 25 (midway through block 1 [16..32))
        let probe_abs = q_offset + probe_r;
        let mut k_poisoned = k_clean.clone();
        for k_tok in &mut k_poisoned[(probe_abs + 1)..kv_len] {
            for d in 0..HDIM {
                // Adversarial spike strongly aligned with probe query.
                k_tok[d] = 250.0 * q_chunk[probe_r][d].signum();
            }
        }

        for &razer in &[false, true] {
            let mut kv_a = BitFp4PagedKvModel::new(razer);
            kv_a.append_chunk(&k_clean, &v_clean, None, true, true);
            let mut kv_b = BitFp4PagedKvModel::new(razer);
            kv_b.append_chunk(&k_poisoned, &v_clean, None, true, true);

            let out_a = prefill_paged_bitfp4_reference(
                &q_chunk, &kv_a, q_offset, kv_len, inv_sqrt_d, true, true, None,
            );
            let out_b = prefill_paged_bitfp4_reference(
                &q_chunk, &kv_b, q_offset, kv_len, inv_sqrt_d, true, true, None,
            );

            // Rows 0..=probe_r MUST be bit-identical: future keys in block 1 & block 2
            // are strictly masked out before softmax and cannot leak even 1 ULP.
            for r in 0..=probe_r {
                assert_eq!(
                    out_a[r],
                    out_b[r],
                    "razer={razer} row={r} (abs={}): future key leaked past causal mask!",
                    q_offset + r
                );
            }

            // Row `probe_r + 1` CAN see `probe_abs + 1`, so poisoned K MUST visibly change it
            // (proves the causal boundary is exact `<=` and not over-masked!).
            let diff_next: f32 = out_a[probe_r + 1]
                .iter()
                .zip(out_b[probe_r + 1].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff_next > 1e-2,
                "razer={razer}: causal boundary at pos={} failed to attend to self-token (diff={diff_next:e})",
                probe_abs + 1
            );
        }
    }

    #[test]
    fn chunked_prefill_bf16_block0_and_partial_sinks_eliminate_sink_quant_error() {
        let mut rng = 0x51C0_BF16_0001_9999u64;
        let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();
        let total_len = 85usize; // 5 full blocks + 5-token partial tail block

        let mut q_all = vec![[0.0f32; HDIM]; total_len];
        let mut k_all = vec![[0.0f32; HDIM]; total_len];
        let mut v_all = vec![[0.0f32; HDIM]; total_len];
        for t in 0..total_len {
            for d in 0..HDIM {
                q_all[t][d] = pseudo_gaussian(&mut rng);
                k_all[t][d] = pseudo_gaussian(&mut rng);
                v_all[t][d] = pseudo_gaussian(&mut rng);
            }
        }
        // Inject massive BOS/attention-sink spikes on tokens 0..4.
        for t in 0..4 {
            for d in 0..HDIM {
                k_all[t][d] *= 6.0;
                v_all[t][d] *= 4.0;
            }
        }

        let o_fp32 = exact_causal_attention_fp32(&q_all, &k_all, &v_all, inv_sqrt_d);

        // Test BOTH full block-0 sink (16 tokens) AND partial block-0 sink (4 tokens),
        // covering queries inside block 0 (q_offset=0) and continuation queries (q_offset=48).
        for &num_sink in &[4usize, 16usize] {
            for &razer in &[false, true] {
                let mut kv = BitFp4PagedKvModel::new(razer);
                kv.append_chunk(&k_all[..48], &v_all[..48], None, true, true);
                kv.append_chunk(&k_all[48..], &v_all[48..], None, true, true);

                let sink = Some((&k_all[..num_sink], &v_all[..num_sink], num_sink));
                // Full block-0 sink (16 tokens) skips FP4 block 0 completely; partial sink
                // (4 tokens) exercises `first_fp4_blk = 0` + `key_pos >= 4` masking where
                // block 0's shared V^T scale is still influenced by the 4x sink spikes in [0..4).
                let row_cos_floor = if num_sink == BLOCK_SIZE { 0.985 } else { 0.975 };

                // Chunk 0 (q_offset=0, q_len=48): covers rows INSIDE sink [0..num_sink),
                // non-sink rows inside block 0 [num_sink..16), and FP4 blocks 1..3 [16..48).
                let o_c0 = prefill_paged_bitfp4_reference(
                    &q_all[..48],
                    &kv,
                    0,
                    48,
                    inv_sqrt_d,
                    true,
                    true,
                    sink,
                );
                for r in 0..num_sink {
                    let max_err = o_fp32[r]
                        .iter()
                        .zip(o_c0[r].iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_err < 1e-5,
                        "num_sink={num_sink} razer={razer} row={r}: pure BF16 sink causal drift {max_err:e}"
                    );
                }
                let mut sum_c0 = 0.0f64;
                for r in num_sink..48 {
                    let cos = cosine_sim(&o_fp32[r], &o_c0[r]);
                    assert!(
                        cos > row_cos_floor,
                        "num_sink={num_sink} razer={razer} row={r}: sink+FP4 chunk-0 cosine too low ({cos:.5})"
                    );
                    sum_c0 += cos;
                }
                let mean_c0 = sum_c0 / ((48 - num_sink) as f64);
                assert!(
                    mean_c0 > 0.990,
                    "num_sink={num_sink} razer={razer}: mean chunk-0 cosine too low ({mean_c0:.5})"
                );

                // Chunk 1 continuation (q_offset=48, q_len=37, kv_len=85):
                // merges exact BF16 sink with FP4 blocks (including partial block 0 when num_sink=4
                // and partial tail block 5 with 5 tokens).
                let o_c1 = prefill_paged_bitfp4_reference(
                    &q_all[48..],
                    &kv,
                    48,
                    total_len,
                    inv_sqrt_d,
                    true,
                    true,
                    sink,
                );
                let o_c1_nosink = prefill_paged_bitfp4_reference(
                    &q_all[48..],
                    &kv,
                    48,
                    total_len,
                    inv_sqrt_d,
                    true,
                    true,
                    None,
                );
                let mut sum_c1 = 0.0f64;
                let mut sum_c1_nosink = 0.0f64;
                for (local_r, row_out) in o_c1.iter().enumerate() {
                    let abs_r = 48 + local_r;
                    let cos = cosine_sim(&o_fp32[abs_r], row_out);
                    assert!(
                        cos > row_cos_floor,
                        "num_sink={num_sink} razer={razer} abs_row={abs_r}: sink+FP4 continuation cosine too low ({cos:.5})"
                    );
                    sum_c1 += cos;
                    sum_c1_nosink += cosine_sim(&o_fp32[abs_r], &o_c1_nosink[local_r]);
                }
                let mean_c1 = sum_c1 / (o_c1.len() as f64);
                let mean_c1_nosink = sum_c1_nosink / (o_c1.len() as f64);
                assert!(
                    mean_c1 > 0.990 && mean_c1 > mean_c1_nosink,
                    "num_sink={num_sink} razer={razer}: BF16 sink should improve continuation accuracy \
                     (mean_sink={mean_c1:.5}, mean_nosink={mean_c1_nosink:.5})"
                );
            }
        }
    }

    #[test]
    fn prefill_launcher_validation_enforces_alignment_bounds_and_handles() {
        use crate::layers::ops::{BitFp4Bf16Sink, BitFp4CausalMask, validate_bitfp4_prefill_args};
        use spark_runtime::gpu::DevicePtr;

        assert_eq!(BitFp4CausalMask::Disabled.to_u32(), 0);
        assert_eq!(BitFp4CausalMask::Causal.to_u32(), 1);

        let valid_k = KernelHandle(42);
        let valid_bt = DevicePtr(0x1000);
        let valid_stride = 24 * 256; // 6144 (multiple of 8)
        let no_sink = BitFp4Bf16Sink::default();

        // Valid 512K continuation call succeeds.
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                valid_bt,
                4096,
                524_288,
                520_192,
                24,
                4,
                valid_stride,
                valid_stride,
                no_sink,
                "prefill",
            )
            .is_ok()
        );

        // Null kernel handle fails closed.
        assert!(
            validate_bitfp4_prefill_args(
                KernelHandle(0),
                valid_bt,
                256,
                1024,
                768,
                24,
                4,
                valid_stride,
                valid_stride,
                no_sink,
                "prefill",
            )
            .is_err()
        );

        // Non-16B-aligned stride or undersized stride fails closed.
        for bad_stride in [valid_stride - 256, valid_stride + 4] {
            assert!(
                validate_bitfp4_prefill_args(
                    valid_k,
                    valid_bt,
                    256,
                    1024,
                    768,
                    24,
                    4,
                    bad_stride,
                    valid_stride,
                    no_sink,
                    "prefill",
                )
                .is_err()
            );
            assert!(
                validate_bitfp4_prefill_args(
                    valid_k,
                    valid_bt,
                    256,
                    1024,
                    768,
                    24,
                    4,
                    valid_stride,
                    bad_stride,
                    no_sink,
                    "prefill",
                )
                .is_err()
            );
        }

        // Null block_table when kv_len > 0 fails closed; kv_len == 0 permits null.
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                DevicePtr::NULL,
                256,
                1024,
                768,
                24,
                4,
                valid_stride,
                valid_stride,
                no_sink,
                "prefill",
            )
            .is_err()
        );
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                DevicePtr::NULL,
                256,
                0,
                0,
                24,
                4,
                valid_stride,
                valid_stride,
                no_sink,
                "prefill",
            )
            .is_ok()
        );

        // Non-zero sink token count with null sink pointers fails closed; both non-null passes.
        let bad_sink = BitFp4Bf16Sink {
            k_sink_bf16: DevicePtr(0x2000),
            v_sink_bf16: DevicePtr::NULL,
            num_sink_tokens: 16,
        };
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                valid_bt,
                256,
                1024,
                768,
                24,
                4,
                valid_stride,
                valid_stride,
                bad_sink,
                "prefill",
            )
            .is_err()
        );
        let good_sink = BitFp4Bf16Sink {
            k_sink_bf16: DevicePtr(0x2000),
            v_sink_bf16: DevicePtr(0x3000),
            num_sink_tokens: 16,
        };
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                valid_bt,
                256,
                1024,
                768,
                24,
                4,
                valid_stride,
                valid_stride,
                good_sink,
                "prefill",
            )
            .is_ok()
        );

        // Position overflow (`q_offset + q_len`) fails closed.
        assert!(
            validate_bitfp4_prefill_args(
                valid_k,
                valid_bt,
                16,
                1024,
                u32::MAX - 4,
                24,
                4,
                valid_stride,
                valid_stride,
                no_sink,
                "prefill",
            )
            .is_err()
        );
    }
}
