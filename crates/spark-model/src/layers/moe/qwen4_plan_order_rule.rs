// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER (K10 item D, default off): which planner orders the W4A8 compact
//! MoE's 128-row work items. Dependency-free (std only) so the standalone contract test
//! (`tests/moe_plan_order_static.rs`, `rustc --test`) runs these functions instead of pinning their text.
//!
//! Absent / `0` / `v7`: V7's planner (`qwen4_moe_compact_t_plan_k32_v7`), expert-major. `inter`: the V8 planner
//! (`qwen4_moe_compact_t_plan_k32_v8.cu`): the same items, multi-tile experts contiguous in expert order and the
//! single-tile experts spread evenly between their tiles, so the single-tile experts' DRAM weight stream overlaps
//! the multi-tile experts' L2 re-read tiles. Every GEMM task computes and writes exactly what it does under V7, so
//! outputs are byte-identical (micro 173: four real 8,192-row routings, gate/up + down, with a negative control).
//! `ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER_MAX_ROWS=<n>` limits V8 to dispatches of at most n rows (absent: all).
//! `ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER_TRIGGER=<file>` (test-only) applies V8 only while the file exists.

/// Selector: absent / `0` / `v7` = V7 order (off), `inter` = V8 planner.
pub(crate) const PLAN_ORDER_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER";
/// Optional row gate for `inter`: V8 plans only dispatches with `rows <= n` (n >= 2).
pub(crate) const PLAN_ORDER_MAX_ROWS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER_MAX_ROWS";
/// Test-only: with `inter` engaged, V8 plans a dispatch only while this file exists (checked per dispatch), so one
/// load can interleave the V7 and V8 orders request by request (A/B/A). Unset: V8 plans every admitted dispatch.
pub(crate) const PLAN_ORDER_TRIGGER_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER_TRIGGER";
/// PTX module (file stem) and entry of the V8 planner.
pub(crate) const PLAN_V8_MODULE: &str = "qwen4_moe_compact_t_plan_k32_v8";
pub(crate) const PLAN_V8_KERNEL: &str = "qwen4_moe_compact_t_plan_k32_v8";

/// The planner's tile height (V7_TILE_M = V8_TILE_M = the V16/V28/V17q tile).
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) const TILE_M: u32 = 128;
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) const EXPERTS: usize = 512;

/// `inter` engaged, optionally limited to dispatches of at most `max_rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanOrder {
    pub(crate) max_rows: Option<u32>,
}

impl PlanOrder {
    /// Whether a dispatch of `rows` rows is planned by V8.
    pub(crate) fn engages(self, rows: usize) -> bool {
        self.max_rows.is_none_or(|max| rows <= max as usize)
    }

    /// The receipt's `max_rows=` value.
    pub(crate) fn max_rows_label(self) -> String {
        self.max_rows
            .map_or_else(|| "all".to_string(), |max| max.to_string())
    }
}

/// Parse the two selector values (None = unset). Off returns `Ok(None)`; a row gate without `inter`, an
/// unknown order or a malformed row count is an error, so the selector fails closed.
pub(crate) fn parse_plan_order(
    order: Option<&str>,
    max_rows: Option<&str>,
) -> Result<Option<PlanOrder>, String> {
    let max_rows = match max_rows {
        None => None,
        Some(raw) => match raw.trim().parse::<u32>() {
            Ok(n) if n >= 2 => Some(n),
            _ => {
                return Err(format!(
                    "{PLAN_ORDER_MAX_ROWS_SELECTOR} must be an integer >= 2; got {raw:?}"
                ));
            }
        },
    };
    match order.map(str::trim) {
        None | Some("0") | Some("v7") => match max_rows {
            None => Ok(None),
            Some(_) => Err(format!(
                "{PLAN_ORDER_MAX_ROWS_SELECTOR} needs {PLAN_ORDER_SELECTOR}=inter"
            )),
        },
        Some("inter") => Ok(Some(PlanOrder { max_rows })),
        Some(other) => Err(format!(
            "{PLAN_ORDER_SELECTOR} must be absent, 0, v7 or inter; got {other:?}"
        )),
    }
}

/// ceil(rows / 128) per expert: the planner's `s_tiles`.
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) fn tiles(rows_per_expert: &[u32; EXPERTS]) -> [u32; EXPERTS] {
    let mut out = [0u32; EXPERTS];
    for (o, &r) in out.iter_mut().zip(rows_per_expert) {
        *o = r.div_ceil(TILE_M);
    }
    out
}

/// V7's item list: expert ascending, m_tile ascending.
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) fn v7_items(tiles: &[u32; EXPERTS]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (e, &t) in tiles.iter().enumerate() {
        for m in 0..t {
            out.push((e as u32, m));
        }
    }
    out
}

/// Host mirror of the V8 planner's placement (same integer arithmetic as `qwen4_moe_compact_t_plan_k32_v8`):
/// multi-tile item k -> k + k S / B, single-tile expert j -> ceil((j+1) B / S) + j, V7's order when B or S is 0.
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) fn v8_items(tiles: &[u32; EXPERTS]) -> Vec<(u32, u32)> {
    let count: u32 = tiles.iter().sum();
    let big: u32 = tiles.iter().filter(|&&t| t > 1).sum();
    let single = tiles.iter().filter(|&&t| t == 1).count() as u32;
    if big == 0 || single == 0 {
        return v7_items(tiles);
    }
    let mut out = vec![(u32::MAX, u32::MAX); count as usize];
    let (mut pb, mut ps) = (0u32, 0u32);
    for (e, &t) in tiles.iter().enumerate() {
        if t > 1 {
            for m in 0..t {
                let k = pb + m;
                out[(k + k * single / big) as usize] = (e as u32, m);
            }
            pb += t;
        } else if t == 1 {
            out[((ps + 1) * big).div_ceil(single) as usize + ps as usize] = (e as u32, 0);
            ps += 1;
        }
    }
    out
}

/// Reference interleaved order: emit the multi-tile items in expert order and,
/// after the k-th of them, floor((k+1) S / B) - floor(k S / B) single-tile experts.
#[allow(dead_code)] // host mirror for tests/moe_plan_order_static.rs; the runtime plans on the device
pub(crate) fn inter_reference(tiles: &[u32; EXPERTS]) -> Vec<(u32, u32)> {
    let big: Vec<(u32, u32)> = v7_items(tiles)
        .into_iter()
        .filter(|&(e, _)| tiles[e as usize] > 1)
        .collect();
    let small: Vec<(u32, u32)> = v7_items(tiles)
        .into_iter()
        .filter(|&(e, _)| tiles[e as usize] == 1)
        .collect();
    if big.is_empty() || small.is_empty() {
        return big.into_iter().chain(small).collect();
    }
    let (nb, ns) = (big.len(), small.len());
    let mut out = Vec::with_capacity(nb + ns);
    let mut s = 0usize;
    for (k, item) in big.iter().enumerate() {
        out.push(*item);
        let take = (k + 1) * ns / nb - k * ns / nb;
        out.extend_from_slice(&small[s..s + take]);
        s += take;
    }
    out.extend_from_slice(&small[s..]);
    out
}
