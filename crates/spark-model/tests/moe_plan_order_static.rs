// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contract for ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER (K10 item D): the V8 planner's placement is a
//! permutation of V7's items that equals the `inter` reference order; the selector parses fail-closed; the V8
//! kernel keeps V7's validation and header code; the dispatch uses V8 only through the selector.
//! Standalone (std only): `rustc --test --edition 2024 moe_plan_order_static.rs`.

#[path = "../src/layers/moe/qwen4_plan_order_rule.rs"]
#[allow(dead_code)]
mod rule;

use rule::{EXPERTS, PlanOrder, inter_reference, parse_plan_order, tiles, v7_items, v8_items};

const V8_CU: &str = include_str!(
    "../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_plan_k32_v8.cu"
);
const V7_CU: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32_v7.cu");
const COMPACT: &str = include_str!("../src/layers/moe/qwen4_prefill_compact.rs");
const OPS: &str = include_str!("../src/layers/moe/qwen4_compact_ops.rs");

/// Deterministic skewed row counts (xorshift; Zipf-like: a few experts take thousands of rows).
fn counts(seed: u64, rows: u32) -> [u32; EXPERTS] {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut out = [0u32; EXPERTS];
    for _ in 0..rows * 10 {
        // squared uniform -> skew toward low expert ids, then a seed-dependent rotation
        let u = (next() % 1_000_003) as f64 / 1_000_003.0;
        let e = ((u * u * u) * EXPERTS as f64) as usize % EXPERTS;
        out[(e + seed as usize * 37) % EXPERTS] += 1;
    }
    out
}

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

#[test]
fn v8_matches_the_reference_order_and_is_a_permutation_of_v7() {
    for seed in 0..64u64 {
        for rows in [512u32, 2048, 8192, 16384] {
            let t = tiles(&counts(seed, rows));
            let v7 = v7_items(&t);
            let v8 = v8_items(&t);
            assert_eq!(v8, inter_reference(&t), "seed {seed} rows {rows}");
            assert_eq!(sorted(v8), sorted(v7), "seed {seed} rows {rows}");
        }
    }
}

#[test]
fn degenerate_routings_keep_v7_order() {
    let mut only_single = [0u32; EXPERTS];
    only_single[3] = 5;
    only_single[400] = 128;
    let t = tiles(&only_single);
    assert_eq!(v8_items(&t), v7_items(&t));
    let mut only_big = [0u32; EXPERTS];
    only_big[0] = 129;
    only_big[511] = 1000;
    let t = tiles(&only_big);
    assert_eq!(v8_items(&t), v7_items(&t));
    let mut mixed = [0u32; EXPERTS];
    mixed[0] = 300; // 3 tiles
    mixed[1] = 1;
    mixed[2] = 128;
    let t = tiles(&mixed);
    // B = 3, S = 2: big 0 at 0, small j=0 at ceil(3/2)+0 = 2, big 1 at 1, big 2 at 2+1 = 3, small j=1 at 3+1 = 4
    assert_eq!(v8_items(&t), vec![(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)]);
}

#[test]
fn selector_parses_fail_closed() {
    assert_eq!(parse_plan_order(None, None), Ok(None));
    assert_eq!(parse_plan_order(Some("0"), None), Ok(None));
    assert_eq!(parse_plan_order(Some("v7"), None), Ok(None));
    assert_eq!(
        parse_plan_order(Some("inter"), None),
        Ok(Some(PlanOrder { max_rows: None }))
    );
    assert_eq!(
        parse_plan_order(Some("inter"), Some("8192")),
        Ok(Some(PlanOrder {
            max_rows: Some(8192)
        }))
    );
    assert!(parse_plan_order(Some("inter"), Some("1")).is_err());
    assert!(parse_plan_order(Some("inter"), Some("8k")).is_err());
    assert!(parse_plan_order(None, Some("8192")).is_err());
    assert!(parse_plan_order(Some("1"), None).is_err());
    assert!(parse_plan_order(Some("desc"), None).is_err());
    let gated = PlanOrder {
        max_rows: Some(8192),
    };
    assert!(gated.engages(8192) && !gated.engages(8193) && gated.max_rows_label() == "8192");
    let all = PlanOrder { max_rows: None };
    assert!(all.engages(65536) && all.max_rows_label() == "all");
}

/// The kernel's placement lines are the arithmetic `v8_items` mirrors; its validation/header code is V7's.
#[test]
fn v8_kernel_keeps_v7_validation_and_mirrors_the_host_placement() {
    for line in [
        "workspace->items[k + (k * S) / B] = {tid, t};",
        "workspace->items[((j + 1) * B + S - 1) / S + j] = {tid, 0u};",
        "workspace->items[s_prefix[tid] + t] = {tid, t};",
        "if (B == 0 || S == 0) {",
        "s_class[e] = s_tiles[e] > 1 ? big : single;",
    ] {
        assert!(V8_CU.contains(line), "V8 planner lacks {line:?}");
    }
    for line in [
        "if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) atomicExch(&s_err, oi640::ERR_TOKEN);",
        "if (begin < 0 || end < begin || static_cast<uint32_t>(end) > expanded) atomicExch(&s_err, oi640::ERR_OFFSETS);",
        "else if (status[0] != oi640::PENDING || !oi640::contract_ok(contract)) { s_err = oi640::ERR_ABI; }",
        "hash = (hash ^ rows) * 1099511628211ULL;",
        "workspace->census_hash = hash;",
        "workspace->ready_magic = oi640::MAGIC;",
        "status[0] = oi640::PLANNED;",
    ] {
        assert!(
            V7_CU.contains(line) && V8_CU.contains(line),
            "V7/V8 validation differs at {line:?}"
        );
    }
}

/// Off, nothing changes: the V8 module is loaded only when the selector parses to `inter`, and the routed
/// dispatch picks the V8 planner only through `PlanOrder::engages`.
#[test]
fn dispatch_uses_v8_only_through_the_selector() {
    assert!(COMPACT.contains("parse_plan_order("));
    assert!(COMPACT.contains("PLAN_V8_MODULE") && COMPACT.contains("PLAN_V8_KERNEL"));
    assert!(COMPACT.contains("MOE_PREFILL_PLAN_ORDER_ENGAGED"));
    assert!(OPS.contains(".engages(rows)"));
    assert!(OPS.contains("MOE_PREFILL_PLAN_ORDER_RAN"));
    // the test-only trigger gates V8 per dispatch and logs every state change
    assert!(OPS.contains("if !plan_order_trigger_on(rows) {"));
    assert!(OPS.contains("MOE_PREFILL_PLAN_ORDER_TRIGGER state="));
    assert!(OPS.contains("plan_for_rows(&wide, rows),"));
}
