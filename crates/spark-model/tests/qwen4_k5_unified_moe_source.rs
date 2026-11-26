// SPDX-License-Identifier: AGPL-3.0-only

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).expect("section start");
    let tail = &source[start..];
    let end = tail.find(end).expect("section end");
    &tail[..end]
}

#[test]
fn transposed_k2_k3_chunks_write_moe_output() {
    // The unified chunk helper copies moe_output rows after every K2/K3 call,
    // so each transposed branch must blend into moe_output itself. Without it
    // every K5/K9 verify row reads a stale MoE output and rows 3..4 repeat
    // rows 0..1.
    for (source, silu_down, blend) in [
        (
            include_str!("../src/layers/moe/forward_k2.rs"),
            "ops::moe_expert_silu_down_shared_batch2_t(",
            "ops::moe_weighted_sum_blend_batch2(",
        ),
        (
            include_str!("../src/layers/moe/forward_k3.rs"),
            "ops::moe_expert_silu_down_shared_batch3_t(",
            "ops::moe_weighted_sum_blend_batch3(",
        ),
    ] {
        let branch = section(
            source,
            "} else if self.use_t_layout_for_decode() {",
            "\n        } else",
        );
        let silu_at = branch.find(silu_down).expect("transposed silu/down launch");
        let blend_at = branch.find(blend).expect("transposed branch blends");
        assert!(silu_at < blend_at);
        assert!(branch[blend_at..].contains("output,"));
        assert!(branch[blend_at..].contains("weights_dev,"));
    }
}
