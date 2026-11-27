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
        };
        let base_only = BitFp4KvKernels {
            decode_razer: KernelHandle(0),
            kgamma_razer: KernelHandle(0),
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
}
