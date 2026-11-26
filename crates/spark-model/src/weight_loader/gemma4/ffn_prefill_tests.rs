// SPDX-License-Identifier: AGPL-3.0-only

use super::ffn_prefill::{
    FFN_TRANSPOSE_HEADROOM_BYTES, ffn_transpose_bytes, ffn_transpose_fits, parse_ffn_transpose,
};

const GIB: usize = 1024 * 1024 * 1024;

#[test]
fn gemma4_31b_transpose_cost_matches_the_nvfp4_layout() {
    // hidden 5376, intermediate 21504, 60 layers; both dims are 64-aligned.
    let per_weight = 5376 * 21504 / 2 + 5376 * 21504 / 16;
    assert_eq!(ffn_transpose_bytes(5376, 21504, 60), 3 * per_weight * 60);
    let gib = ffn_transpose_bytes(5376, 21504, 60) as f64 / GIB as f64;
    assert!((10.8..11.0).contains(&gib), "{gib}");
}

#[test]
fn transpose_cost_pads_n_to_64_like_transpose_for_gemm() {
    // N = 100 pads to 128 columns; K = 32 gives 16 B data + 2 B scales per row.
    assert_eq!(
        ffn_transpose_bytes(32, 100, 1),
        2 * (16 + 2) * 128 + (50 + 6) * 64
    );
}

#[test]
fn fit_keeps_the_headroom_and_never_underflows() {
    let needed = 12 * GIB;
    assert!(ffn_transpose_fits(
        needed,
        needed + FFN_TRANSPOSE_HEADROOM_BYTES
    ));
    assert!(!ffn_transpose_fits(
        needed,
        needed + FFN_TRANSPOSE_HEADROOM_BYTES - 1
    ));
    assert!(!ffn_transpose_fits(1, 0));
    assert!(ffn_transpose_fits(0, FFN_TRANSPOSE_HEADROOM_BYTES));
}

#[test]
fn request_parser_accepts_only_zero_or_one() {
    assert_eq!(parse_ffn_transpose(None), Ok(false));
    assert_eq!(parse_ffn_transpose(Some("0")), Ok(false));
    assert_eq!(parse_ffn_transpose(Some("1")), Ok(true));
    for bad in ["", "true", "2", "1 "] {
        assert!(parse_ffn_transpose(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn loader_installs_prefill_copies_only_without_bf16_mlp() {
    let loader = include_str!("loader_a.rs");
    let plan = loader
        .find("let ffn_prefill_transpose = plan_ffn_prefill_transpose(config, gpu, bf16_mlp)?;")
        .unwrap();
    let layer_loop = loader.find("for i in 0..config.num_hidden_layers").unwrap();
    assert!(plan < layer_loop, "decide once, before layers allocate");
    assert!(loader.contains("if ffn_prefill_transpose && bf16_mlp_weights.is_none() {"));
}
