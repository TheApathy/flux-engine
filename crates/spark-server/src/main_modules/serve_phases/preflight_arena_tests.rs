// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn checked_gdn_estimate_preserves_ordinary_geometry() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let key = config.linear_num_key_heads * config.linear_key_head_dim;
    let value = config.linear_num_value_heads * config.linear_value_head_dim;
    for rows in [0, 1, 4_224, 16_512] {
        let expected = rows * (key * 2 + value) * 2
            + rows * config.linear_num_value_heads * 8
            + rows * value * 4;
        assert_eq!(
            checked_gdn_two_phase_bytes(&config, rows).unwrap(),
            expected
        );
    }
}

#[test]
fn checked_gdn_estimate_rejects_row_and_dimension_overflow() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 1;
    config.linear_value_head_dim = 1;
    let exact = usize::MAX / 14;
    assert_eq!(
        checked_gdn_two_phase_bytes(&config, exact).unwrap(),
        exact * 14
    );
    assert!(checked_gdn_two_phase_bytes(&config, exact + 1).is_err());
    config.linear_num_key_heads = usize::MAX;
    config.linear_key_head_dim = 2;
    assert!(checked_gdn_two_phase_bytes(&config, 1).is_err());
}

#[test]
fn inactive_gdn_estimate_stays_zero() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.layer_types = vec![atlas_core::config::LayerType::FullAttention];
    assert_eq!(checked_gdn_two_phase_bytes(&config, usize::MAX).unwrap(), 0);
}
