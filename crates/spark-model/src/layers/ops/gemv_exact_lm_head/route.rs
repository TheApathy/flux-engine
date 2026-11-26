// SPDX-License-Identifier: AGPL-3.0-only

//! Dependency-free exact LM-head tier and arithmetic-provenance policy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLmHeadTier {
    M4,
    M8,
    M17,
    M32,
}

impl ExactLmHeadTier {
    pub const fn max_rows(self) -> u32 {
        match self {
            Self::M4 => 4,
            Self::M8 => 8,
            Self::M17 => 17,
            Self::M32 => 32,
        }
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_batch_logits_exact_m4",
            Self::M8 => "w4a16_gemv_batch_logits_exact_m8",
            Self::M17 => "w4a16_gemv_batch_logits_exact_m17",
            Self::M32 => "w4a16_gemv_batch_logits_exact_m32",
        }
    }

    /// Register-tiled twin of `symbol()`: T=2 adjacent output rows per lane
    /// group. Same per-output operand sequence, so it is a numerics-preserving
    /// substitution, not a different arithmetic.
    pub const fn symbol_rt2(self) -> &'static str {
        match self {
            Self::M4 => "w4a16_gemv_batch_logits_exact_rt2_m4",
            Self::M8 => "w4a16_gemv_batch_logits_exact_rt2_m8",
            Self::M17 => "w4a16_gemv_batch_logits_exact_rt2_m17",
            Self::M32 => "w4a16_gemv_batch_logits_exact_rt2_m32",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::M4 => "m4",
            Self::M8 => "m8",
            Self::M17 => "m17",
            Self::M32 => "m32",
        }
    }
}

/// Select the smallest exact tier that can hold `rows`. M=1 stays K1 GEMV.
pub const fn exact_lm_head_tier_for_rows(rows: u32) -> Option<ExactLmHeadTier> {
    match rows {
        2..=4 => Some(ExactLmHeadTier::M4),
        5..=8 => Some(ExactLmHeadTier::M8),
        9..=17 => Some(ExactLmHeadTier::M17),
        18..=32 => Some(ExactLmHeadTier::M32),
        _ => None,
    }
}

/// Qualified exact launch or independent row-major K1 fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLmHeadRoute {
    Exact(ExactLmHeadTier),
    SerialK1(ExactLmHeadTier),
}

impl ExactLmHeadRoute {
    pub const fn tier(self) -> ExactLmHeadTier {
        match self {
            Self::Exact(tier) | Self::SerialK1(tier) => tier,
        }
    }

    pub const fn provenance(self) -> &'static str {
        match self {
            Self::Exact(_) => "exact_dynamic_m_nvfp4",
            Self::SerialK1(_) => "serial_k1_row_major_nvfp4",
        }
    }
}

/// Pure policy decision. Presence refers only to the selected tier.
pub const fn exact_lm_head_route_for_rows(
    rows: u32,
    selected_kernel_present: bool,
) -> Option<ExactLmHeadRoute> {
    let Some(tier) = exact_lm_head_tier_for_rows(rows) else {
        return None;
    };
    Some(if selected_kernel_present {
        ExactLmHeadRoute::Exact(tier)
    } else {
        ExactLmHeadRoute::SerialK1(tier)
    })
}

/// Single-sweep wide-row exact NVFP4 GEMV tier (`26..=128` rows in 1 weight sweep).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactSingleSweepTier {
    M64,
    M80,
    M128,
}

impl ExactSingleSweepTier {
    pub const fn max_rows(self) -> u32 {
        match self {
            Self::M64 => 64,
            Self::M80 => 80,
            Self::M128 => 128,
        }
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::M64 => "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            Self::M80 => "w4a16_gemv_batch_logits_exact_sw_m80_o1w8",
            Self::M128 => "w4a16_gemv_batch_logits_exact_sw_m128_o1w8",
        }
    }

    pub const fn outs_per_block(self) -> u32 {
        8
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::M64 => "sw_m64_o1w8",
            Self::M80 => "sw_m80_o1w8",
            Self::M128 => "sw_m128_o1w8",
        }
    }
}

/// Select the single-sweep exact GEMV tier for `rows` in `26..=128`.
///
/// Routes `26..=64 -> M64` (covering the unstaged `26..=32` `m32` gap),
/// `65..=80 -> M80`, and `81..=128 -> M128`.
pub const fn exact_single_sweep_tier_for_rows(rows: u32) -> Option<ExactSingleSweepTier> {
    match rows {
        26..=64 => Some(ExactSingleSweepTier::M64),
        65..=80 => Some(ExactSingleSweepTier::M80),
        81..=128 => Some(ExactSingleSweepTier::M128),
        _ => None,
    }
}

/// Pure single-sweep routing check: returns the selected tier only when its
/// handle is present, failing closed to `None` otherwise.
pub const fn exact_single_sweep_route_for_rows(
    rows: u32,
    sw_m64_present: bool,
    sw_m80_present: bool,
    sw_m128_present: bool,
) -> Option<ExactSingleSweepTier> {
    let Some(tier) = exact_single_sweep_tier_for_rows(rows) else {
        return None;
    };
    let present = match tier {
        ExactSingleSweepTier::M64 => sw_m64_present,
        ExactSingleSweepTier::M80 => sw_m80_present,
        ExactSingleSweepTier::M128 => sw_m128_present,
    };
    if present { Some(tier) } else { None }
}

#[cfg(test)]
mod single_sweep_tests {
    use super::{
        ExactSingleSweepTier, exact_single_sweep_route_for_rows, exact_single_sweep_tier_for_rows,
    };

    #[test]
    fn single_sweep_tier_table_covers_26_to_128_and_fails_closed() {
        for rows in [0, 1, 2, 16, 25, 129, u32::MAX] {
            assert_eq!(exact_single_sweep_tier_for_rows(rows), None, "rows={rows}");
            assert_eq!(
                exact_single_sweep_route_for_rows(rows, true, true, true),
                None,
                "rows={rows}"
            );
        }
        for (rows, tier, symbol, label) in [
            (
                26,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
                "sw_m64_o1w8",
            ),
            (
                32,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
                "sw_m64_o1w8",
            ),
            (
                33,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
                "sw_m64_o1w8",
            ),
            (
                40,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
                "sw_m64_o1w8",
            ),
            (
                64,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
                "sw_m64_o1w8",
            ),
            (
                65,
                ExactSingleSweepTier::M80,
                "w4a16_gemv_batch_logits_exact_sw_m80_o1w8",
                "sw_m80_o1w8",
            ),
            (
                80,
                ExactSingleSweepTier::M80,
                "w4a16_gemv_batch_logits_exact_sw_m80_o1w8",
                "sw_m80_o1w8",
            ),
            (
                81,
                ExactSingleSweepTier::M128,
                "w4a16_gemv_batch_logits_exact_sw_m128_o1w8",
                "sw_m128_o1w8",
            ),
            (
                128,
                ExactSingleSweepTier::M128,
                "w4a16_gemv_batch_logits_exact_sw_m128_o1w8",
                "sw_m128_o1w8",
            ),
        ] {
            assert_eq!(exact_single_sweep_tier_for_rows(rows), Some(tier));
            assert!(tier.max_rows() >= rows);
            assert_eq!(tier.symbol(), symbol);
            assert_eq!(tier.label(), label);
            assert_eq!(tier.outs_per_block(), 8);
            assert_eq!(
                exact_single_sweep_route_for_rows(rows, true, true, true),
                Some(tier)
            );
            assert_eq!(
                exact_single_sweep_route_for_rows(rows, false, false, false),
                None
            );
        }
    }
}
