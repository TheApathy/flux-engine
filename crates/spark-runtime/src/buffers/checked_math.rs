// SPDX-License-Identifier: AGPL-3.0-only

//! Checked arithmetic local to arena sizing, not a global config validator.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;

pub(super) fn product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |total, &value| {
        total
            .checked_mul(value)
            .with_context(|| format!("buffer arena capacity overflow: {label}"))
    })
}

pub(super) fn sum(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(0usize, |total, &value| {
        total
            .checked_add(value)
            .with_context(|| format!("buffer arena capacity overflow: {label}"))
    })
}

pub(super) fn align_up(value: usize, alignment: usize, label: &str) -> Result<usize> {
    anyhow::ensure!(alignment.is_power_of_two(), "invalid arena alignment");
    Ok(sum(&[value, alignment - 1], label)? & !(alignment - 1))
}

pub(super) struct ArenaWidths {
    pub(super) residual: usize,
    pub(super) ssm_qkvz: usize,
    pub(super) ssm_ba: usize,
    pub(super) linear_value: usize,
    pub(super) mamba_inner: usize,
    pub(super) mamba_xbc: usize,
    pub(super) mamba_in_proj: usize,
}

impl ArenaWidths {
    pub(super) fn checked(config: &ModelConfig) -> Result<Self> {
        let key = product(
            &[config.linear_num_key_heads, config.linear_key_head_dim],
            "linear key width",
        )?;
        let value = product(
            &[config.linear_num_value_heads, config.linear_value_head_dim],
            "linear value width",
        )?;
        let mamba_inner = product(
            &[config.mamba_num_heads, config.mamba_head_dim],
            "Mamba inner width",
        )?;
        let mamba_xbc = sum(
            &[
                mamba_inner,
                product(
                    &[2, config.n_groups, config.ssm_state_size],
                    "Mamba BC width",
                )?,
            ],
            "Mamba xBC width",
        )?;
        Ok(Self {
            residual: product(
                &[config.hidden_size, config.hc_count.max(1)],
                "residual width",
            )?,
            ssm_qkvz: product(&[sum(&[key, value], "SSM KV width")?, 2], "SSM QKVZ width")?,
            ssm_ba: product(&[config.linear_num_value_heads, 2], "SSM BA width")?,
            linear_value: value,
            mamba_inner,
            mamba_xbc,
            mamba_in_proj: sum(
                &[mamba_inner, mamba_xbc, config.mamba_num_heads],
                "Mamba input projection width",
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_accepts_exact_boundaries_and_rejects_overflow() {
        assert_eq!(product(&[usize::MAX, 1], "exact").unwrap(), usize::MAX);
        assert_eq!(sum(&[usize::MAX - 1, 1], "exact").unwrap(), usize::MAX);
        assert!(product(&[usize::MAX, 2], "overflow").is_err());
        assert!(sum(&[usize::MAX, 1], "overflow").is_err());
        assert_eq!(
            align_up(usize::MAX - 7, 8, "exact").unwrap(),
            usize::MAX - 7
        );
        assert!(align_up(usize::MAX, 8, "overflow").is_err());
        assert!(align_up(1, 0, "invalid").is_err());
        assert!(align_up(1, 3, "invalid").is_err());
    }

    #[test]
    fn checked_local_widths_match_existing_valid_model_getters() {
        for (hc_count, mamba_heads, mamba_head_dim, groups, state) in
            [(0, 0, 0, 0, 0), (1, 0, 0, 1, 128), (4, 8, 32, 4, 64)]
        {
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.hc_count = hc_count;
            config.mamba_num_heads = mamba_heads;
            config.mamba_head_dim = mamba_head_dim;
            config.n_groups = groups;
            config.ssm_state_size = state;
            let widths = ArenaWidths::checked(&config).unwrap();
            assert_eq!(widths.residual, config.residual_width());
            assert_eq!(widths.ssm_qkvz, config.ssm_qkvz_size());
            assert_eq!(widths.ssm_ba, config.ssm_ba_size());
            assert_eq!(widths.mamba_inner, config.mamba2_d_inner());
            assert_eq!(widths.mamba_xbc, config.mamba2_d_xbc());
            assert_eq!(widths.mamba_in_proj, config.mamba2_in_proj_size());
        }
    }

    #[test]
    fn derived_widths_reject_malformed_model_products_before_use() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = usize::MAX;
        config.hc_count = 2;
        assert!(ArenaWidths::checked(&config).is_err());
        config.hidden_size = 1;
        config.linear_num_key_heads = usize::MAX;
        config.linear_key_head_dim = 2;
        assert!(ArenaWidths::checked(&config).is_err());
        config.linear_num_key_heads = 0;
        config.mamba_num_heads = usize::MAX;
        config.mamba_head_dim = 2;
        assert!(ArenaWidths::checked(&config).is_err());
    }

    #[test]
    fn derived_widths_reject_sum_gate_and_mamba_projection_overflow() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.linear_num_key_heads = usize::MAX / 2;
        config.linear_key_head_dim = 1;
        config.linear_num_value_heads = 1;
        config.linear_value_head_dim = 1;
        assert!(ArenaWidths::checked(&config).is_err());

        config.linear_num_key_heads = 0;
        config.linear_num_value_heads = usize::MAX;
        config.linear_value_head_dim = 0;
        assert!(ArenaWidths::checked(&config).is_err());

        config.linear_num_value_heads = 0;
        config.n_groups = usize::MAX;
        config.ssm_state_size = 1;
        assert!(ArenaWidths::checked(&config).is_err());

        config.n_groups = 0;
        config.mamba_num_heads = usize::MAX / 2;
        config.mamba_head_dim = 1;
        assert!(ArenaWidths::checked(&config).is_err());
    }
}
