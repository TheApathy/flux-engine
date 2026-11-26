// SPDX-License-Identifier: AGPL-3.0-only

use super::{WidePrefillAttnRoute, parse_wide_prefill_attn_tc, wide_prefill_attn_tc_route};

#[test]
fn tc_route_is_opt_in_wide_head_only_and_fail_closed() {
    for hd in [128, 256, 512] {
        for has_tc in [false, true] {
            assert_eq!(
                wide_prefill_attn_tc_route(false, hd, has_tc),
                WidePrefillAttnRoute::Disabled
            );
        }
    }
    for hd in [128, 256, 384] {
        assert_eq!(
            wide_prefill_attn_tc_route(true, hd, true),
            WidePrefillAttnRoute::Ineligible,
            "hd={hd}"
        );
    }
    assert_eq!(
        wide_prefill_attn_tc_route(true, 512, true),
        WidePrefillAttnRoute::Complete
    );
    assert_eq!(
        wide_prefill_attn_tc_route(true, 512, false),
        WidePrefillAttnRoute::Missing
    );
}

#[test]
fn tc_request_parser_accepts_only_zero_or_one() {
    assert_eq!(parse_wide_prefill_attn_tc(None), Ok(false));
    assert_eq!(parse_wide_prefill_attn_tc(Some("0")), Ok(false));
    assert_eq!(parse_wide_prefill_attn_tc(Some("1")), Ok(true));
    for bad in ["", "true", "2", " 1"] {
        assert!(parse_wide_prefill_attn_tc(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn both_contiguous_wide_sites_use_the_shared_dispatch() {
    let cache_skip = include_str!("cache_skip.rs");
    let paged = include_str!("paged_attn.rs");
    for (name, source) in [("cache_skip", cache_skip), ("paged_attn", paged)] {
        assert!(
            source.contains("self.prefill_attention_wide("),
            "{name} must route HDIM>256 through prefill_attention_wide"
        );
        assert!(
            !source.contains("self.prefill_attn_512_k,"),
            "{name} launches the scalar 512 kernel directly"
        );
    }
}

#[test]
fn tc_kernel_is_the_common_br32_entry_at_hdim512_bc16() {
    let tc = include_str!(
        "../../../../../../kernels/gb10/gemma-4-31b/nvfp4/inferspark_prefill_512tc.cu"
    );
    assert!(tc.contains("#define HDIM 512"));
    assert!(tc.contains("#define BC 16"));
    assert!(tc.contains("#define ATLAS_SKIP_PREFILL_64 1"));
    assert!(tc.contains("#define inferspark_prefill inferspark_prefill_512tc"));
    assert!(tc.contains("#include \"../../common/inferspark_prefill.cu\""));

    let parent = include_str!("../../../../../../kernels/gb10/common/inferspark_prefill.cu");
    let br32 = &parent[..parent.find("#ifndef ATLAS_SKIP_PREFILL_64").unwrap()];
    assert!(br32.contains("#ifndef BC\n#define BC 32\n#endif"));
    assert!(br32.contains("float acc_s[BC / 8][4];"));
    assert!(br32.contains("ks < (BC / 16)"));
    assert_eq!(br32.matches("idx < TILE_CHUNKS_KV_TILE;").count(), 3);
    // Literal BC=32 trip counts would overrun the BC=16 instantiation.
    assert!(!br32.contains("nt < 4;"));
    assert!(!br32.contains("ks < 2;"));
    assert!(
        parent
            .trim_end()
            .ends_with("#endif  // ATLAS_SKIP_PREFILL_64")
    );
}
