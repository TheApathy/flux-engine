// SPDX-License-Identifier: AGPL-3.0-only

//! K10 item A, `ATLAS_ALLOC_SEQ_ASYNC`: the bytes `alloc_sequence` zeroes in an
//! SSM slot, as two plans over the same pools.
//!
//! - [`sync_plan`] is today's order: `SsmStatePool::zero_slot` (h and conv state
//!   of every layer, stream-ordered, then a host sync) followed by
//!   `SsmStatePool::reset_slot` (h, conv, every MTP intermediate one token at a
//!   time, both checkpoints; one synchronous memset each, 576 on the serve
//!   recipe), then another sync.
//! - [`async_plan`] zeroes every one of those bytes exactly once: per layer h,
//!   conv, the slot's `num_intermediates` contiguous intermediates as one range
//!   per pool (the layout `ATLAS_SSM_RESET_ASYNC` already uses), and both
//!   checkpoints. The caller issues it as stream-ordered memsets, records an
//!   event and makes the prefill stream wait on it, with no host sync.
//!
//! Both plans cover the same byte set (tests below), so the device state every
//! consumer sees is the same: all zero, written before any prefill or decode
//! work on the slot. Offsets mirror the accessors in `ssm_pool.rs`
//! (`h_state(i, slot) = h_state_pools[i] + slot * h_bytes`, `h_intermediate(i,
//! slot, t) = h_intermediate_pools[i] + (slot * ni + t) * h_bytes`, ...).
//!
//! Std-only so `rustc --edition 2024 --test slot_zero_plan.rs` runs its tests.

/// Which per-layer pool a range lies in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Pool {
    HState,
    ConvState,
    HIntermediate,
    ConvIntermediate,
    HCheckpoint,
    ConvCheckpoint,
}

impl Pool {
    pub(crate) const ALL: [Pool; 6] = [
        Pool::HState,
        Pool::ConvState,
        Pool::HIntermediate,
        Pool::ConvIntermediate,
        Pool::HCheckpoint,
        Pool::ConvCheckpoint,
    ];
}

/// One memset: `bytes` at byte `offset` of layer `layer`'s `pool` allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ZeroRange {
    pub(crate) pool: Pool,
    pub(crate) layer: usize,
    pub(crate) offset: usize,
    pub(crate) bytes: usize,
}

/// The pool geometry the plans need (fields of `SsmStatePool`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotGeometry {
    pub(crate) num_layers: usize,
    pub(crate) h_bytes: usize,
    pub(crate) conv_bytes: usize,
    pub(crate) has_mtp: bool,
    pub(crate) compact_lazy_h: bool,
    pub(crate) num_intermediates: usize,
}

impl SlotGeometry {
    fn range(
        &self,
        pool: Pool,
        layer: usize,
        slot: usize,
        token: usize,
        count: usize,
    ) -> ZeroRange {
        let (unit, index) = match pool {
            Pool::HState | Pool::HCheckpoint => (self.h_bytes, slot),
            Pool::ConvState | Pool::ConvCheckpoint => (self.conv_bytes, slot),
            Pool::HIntermediate => (self.h_bytes, slot * self.num_intermediates + token),
            Pool::ConvIntermediate => (self.conv_bytes, slot * self.num_intermediates + token),
        };
        ZeroRange {
            pool,
            layer,
            offset: index * unit,
            bytes: count * unit,
        }
    }
}

/// Today's memsets, in issue order: `zero_slot` then `reset_slot` (sync path).
pub(crate) fn sync_plan(g: &SlotGeometry, slot: usize) -> Vec<ZeroRange> {
    let mut plan = Vec::new();
    for i in 0..g.num_layers {
        plan.push(g.range(Pool::HState, i, slot, 0, 1));
        plan.push(g.range(Pool::ConvState, i, slot, 0, 1));
    }
    for i in 0..g.num_layers {
        plan.push(g.range(Pool::HState, i, slot, 0, 1));
        plan.push(g.range(Pool::ConvState, i, slot, 0, 1));
        if g.has_mtp {
            for t in 0..g.num_intermediates {
                if !g.compact_lazy_h {
                    plan.push(g.range(Pool::HIntermediate, i, slot, t, 1));
                }
                plan.push(g.range(Pool::ConvIntermediate, i, slot, t, 1));
            }
            if !g.compact_lazy_h {
                plan.push(g.range(Pool::HCheckpoint, i, slot, 0, 1));
            }
            plan.push(g.range(Pool::ConvCheckpoint, i, slot, 0, 1));
        }
    }
    plan
}

/// Prefill-essential (`HState`, `ConvState`) vs MTP speculative-decode-only
/// (`HIntermediate`, `ConvIntermediate`, `HCheckpoint`, `ConvCheckpoint`)
/// zero ranges, each byte once, intermediates coalesced per (layer, pool).
pub(crate) fn async_split_plan(g: &SlotGeometry, slot: usize) -> (Vec<ZeroRange>, Vec<ZeroRange>) {
    let mut prefill = Vec::with_capacity(g.num_layers * 2);
    let mut mtp = Vec::with_capacity(if g.has_mtp { g.num_layers * 4 } else { 0 });
    for i in 0..g.num_layers {
        prefill.push(g.range(Pool::HState, i, slot, 0, 1));
        prefill.push(g.range(Pool::ConvState, i, slot, 0, 1));
        if g.has_mtp {
            if g.num_intermediates > 0 {
                if !g.compact_lazy_h {
                    mtp.push(g.range(Pool::HIntermediate, i, slot, 0, g.num_intermediates));
                }
                mtp.push(g.range(Pool::ConvIntermediate, i, slot, 0, g.num_intermediates));
            }
            if !g.compact_lazy_h {
                mtp.push(g.range(Pool::HCheckpoint, i, slot, 0, 1));
            }
            mtp.push(g.range(Pool::ConvCheckpoint, i, slot, 0, 1));
        }
    }
    (prefill, mtp)
}

/// The same bytes, each once, intermediates coalesced per (layer, pool),
/// ordered with prefill-essential (`HState`, `ConvState`) ranges first and
/// MTP speculative-decode-only ranges after.
#[allow(dead_code)]
pub(crate) fn async_plan(g: &SlotGeometry, slot: usize) -> Vec<ZeroRange> {
    let (mut prefill, mtp) = async_split_plan(g, slot);
    prefill.extend(mtp);
    prefill
}

/// Sorted disjoint `[start, end)` byte intervals per (pool, layer).
pub(crate) type Coverage = Vec<((Pool, usize), Vec<(usize, usize)>)>;

/// Total bytes a plan writes (duplicates counted).
pub(crate) fn written_bytes(plan: &[ZeroRange]) -> usize {
    plan.iter().map(|r| r.bytes).sum()
}

/// The byte set a plan covers, as sorted disjoint `[start, end)` intervals per
/// (pool, layer).
pub(crate) fn coverage(plan: &[ZeroRange]) -> Coverage {
    let mut keyed: Vec<((Pool, usize), (usize, usize))> = plan
        .iter()
        .filter(|r| r.bytes > 0)
        .map(|r| ((r.pool, r.layer), (r.offset, r.offset + r.bytes)))
        .collect();
    keyed.sort();
    let mut out: Coverage = Vec::new();
    for (key, (s, e)) in keyed {
        match out.last_mut() {
            Some((k, ivs)) if *k == key => match ivs.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => ivs.push((s, e)),
            },
            _ => out.push((key, vec![(s, e)])),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The serve recipe: 36 GDN layers, 48x128x128 FP32 h, (2*16*128 + 48*128) x 4 x FP32 conv,
    /// MTP head on (ni = 6), slot 0 (`--max-batch-size 1`).
    const SERVE: SlotGeometry = SlotGeometry {
        num_layers: 36,
        h_bytes: 48 * 128 * 128 * 4,
        conv_bytes: (2 * 16 * 128 + 48 * 128) * 4 * 4,
        has_mtp: true,
        compact_lazy_h: false,
        num_intermediates: 6,
    };

    #[test]
    fn serve_plans_cover_the_same_bytes() {
        let s = sync_plan(&SERVE, 0);
        let a = async_plan(&SERVE, 0);
        assert_eq!(coverage(&s), coverage(&a));
        assert_eq!(s.len(), 36 * 2 + 36 * (2 + 2 * 6 + 2)); // 72 + 576 memsets today
        assert_eq!(a.len(), 36 * 6);
        // 0.888 GiB per slot (the "0.89 GB/slot" pre-flight line); today writes h+conv twice.
        let slot_bytes = 36 * 8 * (SERVE.h_bytes + SERVE.conv_bytes);
        assert_eq!(written_bytes(&a), slot_bytes);
        assert_eq!(slot_bytes, 953_155_584);
        assert_eq!(
            written_bytes(&s),
            slot_bytes + 36 * (SERVE.h_bytes + SERVE.conv_bytes)
        );
        // Every async range is disjoint from every other: each byte is written once.
        let cov = coverage(&a);
        let total: usize = cov
            .iter()
            .flat_map(|(_, ivs)| ivs.iter().map(|(s, e)| e - s))
            .sum();
        assert_eq!(total, written_bytes(&a));
    }

    #[test]
    fn plans_agree_on_many_geometries_and_slots() {
        for &(layers, h, c, mtp, ni) in &[
            (1usize, 16usize, 8usize, false, 0usize),
            (3, 64, 24, true, 0),
            (3, 64, 24, true, 1),
            (5, 4096, 512, true, 4),
            (48, 3 << 20, 160 << 10, true, 33),
            (2, 12, 4, false, 7),
        ] {
            for compact in [false, mtp] {
                let g = SlotGeometry {
                    num_layers: layers,
                    h_bytes: h,
                    conv_bytes: c,
                    has_mtp: mtp,
                    compact_lazy_h: compact,
                    num_intermediates: ni,
                };
                for slot in 0..4 {
                    assert_eq!(
                        coverage(&sync_plan(&g, slot)),
                        coverage(&async_plan(&g, slot)),
                        "{g:?} slot {slot}"
                    );
                }
            }
        }
    }

    #[test]
    fn ranges_stay_inside_their_slot() {
        // A slot's ranges never touch a neighbouring slot's bytes.
        for slot in 0..3 {
            for r in async_plan(&SERVE, slot) {
                let (lo, hi) = match r.pool {
                    Pool::HState | Pool::HCheckpoint => {
                        (slot * SERVE.h_bytes, (slot + 1) * SERVE.h_bytes)
                    }
                    Pool::ConvState | Pool::ConvCheckpoint => {
                        (slot * SERVE.conv_bytes, (slot + 1) * SERVE.conv_bytes)
                    }
                    Pool::HIntermediate => {
                        (slot * 6 * SERVE.h_bytes, (slot + 1) * 6 * SERVE.h_bytes)
                    }
                    Pool::ConvIntermediate => (
                        slot * 6 * SERVE.conv_bytes,
                        (slot + 1) * 6 * SERVE.conv_bytes,
                    ),
                };
                assert!(r.offset >= lo && r.offset + r.bytes <= hi, "{r:?}");
            }
        }
    }

    #[test]
    fn coverage_detects_a_missing_range() {
        // Negative control for the equality above.
        let mut a = async_plan(&SERVE, 0);
        a.remove(a.len() / 2);
        assert_ne!(coverage(&sync_plan(&SERVE, 0)), coverage(&a));
        let mut b = async_plan(&SERVE, 0);
        b[3].bytes -= 1;
        assert_ne!(coverage(&sync_plan(&SERVE, 0)), coverage(&b));
    }

    #[test]
    fn no_mtp_means_base_state_only() {
        let g = SlotGeometry {
            has_mtp: false,
            ..SERVE
        };
        let a = async_plan(&g, 1);
        assert_eq!(a.len(), 72);
        assert!(
            a.iter()
                .all(|r| matches!(r.pool, Pool::HState | Pool::ConvState))
        );
        assert_eq!(Pool::ALL.len(), 6);
    }

    #[test]
    fn compact_lazy_h_omits_h_intermediate_and_checkpoint_ranges() {
        let g = SlotGeometry {
            compact_lazy_h: true,
            ..SERVE
        };
        let s = sync_plan(&g, 2);
        let a = async_plan(&g, 2);
        assert_eq!(coverage(&s), coverage(&a));
        assert_eq!(a.len(), 36 * 4);
        assert!(
            a.iter()
                .all(|r| !matches!(r.pool, Pool::HIntermediate | Pool::HCheckpoint))
        );
        assert_eq!(
            written_bytes(&a),
            36 * (SERVE.h_bytes + 8 * SERVE.conv_bytes)
        );
    }
}
