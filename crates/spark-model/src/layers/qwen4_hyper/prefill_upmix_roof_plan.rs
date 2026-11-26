// SPDX-License-Identifier: AGPL-3.0-only

//! Pure selection and admission rules of the up+mix roof kernels
//! (`ATLAS_QWEN4_PREFILL_MHC_ROOF`, default off; see
//! [`super::prefill_upmix_roof`] and
//! `kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_hc_upmix_roof.cu`).
//! Dependency-free so `rustc --test` can check it without the crate.
//!
//! The roof kernels replace `qwen4_hc_upmix_hcfused[_amax]` on the HC_FUSED
//! level-2 path, byte for byte (same mma chains, same epilogue expressions),
//! and only where that kernel would run: `upmix` selected in
//! `ATLAS_QWEN4_PREFILL_MHC_FUSED`. They read hyper' and rstd with 16-byte
//! copies and write `mixed` with 16-byte stores, so those pointers must be
//! 16-byte aligned (the incumbent needs 4); otherwise the incumbent runs.

/// Environment variable of the lever.
pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MHC_ROOF";
/// Opt-out environment variable for fusing MXFP8 `(xq, xs)` quantization and
/// Last-CTA L2-hot shared-expert `gate_scalars` into `qwen4_hc_upmix_roof`
/// (default ON when admitted and exact-parity safe).
pub(crate) const MXFP8_FUSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_UPMIX_MXFP8_FUSED";
/// Maximum row-tiles (`ceil(rows / 128)`) covered by the fixed 4 KiB
/// self-resetting `tile_done` counter buffer (`1024 * 128 = 131,072` rows).
pub(crate) const MXFP8_GATE_MAX_Y_TILES: usize = 1024;
/// Bit-exact router tensor-scaled FP8 activation helper in module
/// `qwen4_hc_upmix_roof` (used only when a consumer still needs `a8_out` +
/// `a_scale_out` after `upmix_roof` already emitted `xq`, `xs`, and `ss_out`).
#[allow(dead_code)]
pub(crate) const FP8_ACT_FROM_BF16_ENTRY: &str = "qwen4_hc_upmix_fp8_act_from_bf16";

/// CTA threads of every roof entry (2 warps along rows x 4 along h).
pub(crate) const THREADS: u32 = 256;
/// h positions per CTA (`kBH`); the grid's x extent is H / 32.
pub(crate) const TILE_H: u32 = 32;

/// L2 cache policy for `mixed` output stores of the up+mix roof kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OutputCachePolicy {
    /// Default kernel behavior (`L2::evict_last` when `rows <= 4096`, plain
    /// stores above that).
    #[default]
    Default,
    /// Mark `mixed` stores `L2::evict_first` so strip-mined wide prefill
    /// keeps `folded`, `up`, `down[strip]`, and `hyper'[strip]` hot in L2.
    StreamEvictFirst,
}

/// Which tile configuration runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tile {
    /// `qwen4_hc_upmix_roof_t64[_amax]`: the incumbent's 64-row tile and K
    /// loop, hyper' requested into L2 at CTA start, 16-byte epilogue
    /// (measured neutral: kept for the record and as a fallback).
    T64,
    /// `qwen4_hc_upmix_roof_t128[_amax]`: 128-row tiles on a 3-stage swizzled
    /// cp.async ring (2 CTAs/SM), with the same epilogue.
    T128,
    /// `qwen4_hc_upmix_roof_t128_w4[_amax]`: `T128` with on-the-fly NVFP4
    /// `up` dequantization into shared memory (skips `s.up` BF16 materialization
    /// and keeps the 1.72 MiB NVFP4 `up` weight in L2 across all M-tiles).
    T128W4,
}

impl Tile {
    #[cfg(test)]
    pub(crate) const ALL: [Tile; 3] = [Tile::T64, Tile::T128, Tile::T128W4];

    /// The tile `1` / `on` selects. Micro 146 (k9y-mhc/roof-run-20261015T193657Z.txt):
    /// t128 1.166 / 2.306 ms per call at 8192 / 16384 rows (87-88 % of the 213 GB/s roof)
    /// against the incumbent's 1.473 / 2.955; t64 alone is neutral (its K loop binds).
    pub(crate) const DEFAULT: Tile = Tile::T128;

    pub(crate) fn is_w4(self) -> bool {
        matches!(self, Tile::T128W4)
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Tile::T64 => "t64",
            Tile::T128 => "t128",
            Tile::T128W4 => "t128_w4",
        }
    }

    /// Kernel entry in module `qwen4_hc_upmix_roof`.
    pub(crate) fn entry(self, amax: bool) -> &'static str {
        match (self, amax) {
            (Tile::T64, false) => "qwen4_hc_upmix_roof_t64",
            (Tile::T64, true) => "qwen4_hc_upmix_roof_t64_amax",
            (Tile::T128, false) => "qwen4_hc_upmix_roof_t128",
            (Tile::T128, true) => "qwen4_hc_upmix_roof_t128_amax",
            (Tile::T128W4, false) => "qwen4_hc_upmix_roof_t128_w4",
            (Tile::T128W4, true) => "qwen4_hc_upmix_roof_t128_w4_amax",
        }
    }

    /// Streaming-output entry (`L2::evict_first` on `mixed` stores) in module
    /// `qwen4_hc_upmix_roof`, when available for this tile.
    pub(crate) fn stream_entry(self, fused_prep: bool, amax: bool) -> Option<&'static str> {
        match (self, fused_prep, amax) {
            (Tile::T64, _, _) => None,
            (Tile::T128, false, false) => Some("qwen4_hc_upmix_roof_t128_stream"),
            (Tile::T128, false, true) => Some("qwen4_hc_upmix_roof_t128_stream_amax"),
            (Tile::T128, true, false) => Some("qwen4_hc_upmix_roof_t128_r4_stream"),
            (Tile::T128, true, true) => Some("qwen4_hc_upmix_roof_t128_r4_stream_amax"),
            (Tile::T128W4, _, false) => Some("qwen4_hc_upmix_roof_t128_w4_stream"),
            (Tile::T128W4, _, true) => Some("qwen4_hc_upmix_roof_t128_w4_stream_amax"),
        }
    }

    /// Fused MXFP8 `(xq, xs)` + optional Last-CTA L2-hot shared-expert gate
    /// scalar entry in module `qwen4_hc_upmix_roof`, when available for this tile.
    pub(crate) fn mxfp8_gate_entry(self, fused_prep: bool, amax: bool) -> Option<&'static str> {
        match (self, fused_prep, amax) {
            (Tile::T64, _, _) => None,
            (Tile::T128, false, false) => Some("qwen4_hc_upmix_roof_t128_mxfp8_gate"),
            (Tile::T128, false, true) => Some("qwen4_hc_upmix_roof_t128_mxfp8_gate_amax"),
            (Tile::T128, true, false) => Some("qwen4_hc_upmix_roof_t128_r4_mxfp8_gate"),
            (Tile::T128, true, true) => Some("qwen4_hc_upmix_roof_t128_r4_mxfp8_gate_amax"),
            (Tile::T128W4, _, false) => Some("qwen4_hc_upmix_roof_t128_w4_mxfp8_gate"),
            (Tile::T128W4, _, true) => Some("qwen4_hc_upmix_roof_t128_w4_mxfp8_gate_amax"),
        }
    }

    /// Streaming-output (`L2::evict_first` on `mixed` stores) + fused MXFP8
    /// `(xq, xs)` + optional Last-CTA gate-scalar entry in module
    /// `qwen4_hc_upmix_roof`, when available for this tile.
    pub(crate) fn stream_mxfp8_gate_entry(
        self,
        fused_prep: bool,
        amax: bool,
    ) -> Option<&'static str> {
        match (self, fused_prep, amax) {
            (Tile::T64, _, _) => None,
            (Tile::T128, false, false) => Some("qwen4_hc_upmix_roof_t128_stream_mxfp8_gate"),
            (Tile::T128, false, true) => Some("qwen4_hc_upmix_roof_t128_stream_mxfp8_gate_amax"),
            (Tile::T128, true, false) => Some("qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate"),
            (Tile::T128, true, true) => Some("qwen4_hc_upmix_roof_t128_r4_stream_mxfp8_gate_amax"),
            (Tile::T128W4, _, false) => Some("qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate"),
            (Tile::T128W4, _, true) => Some("qwen4_hc_upmix_roof_t128_w4_stream_mxfp8_gate_amax"),
        }
    }

    /// Rows per CTA (`BM`).
    pub(crate) fn rows(self) -> u32 {
        match self {
            Tile::T64 => 64,
            Tile::T128 | Tile::T128W4 => 128,
        }
    }

    /// Dynamic shared memory (`kRoofSmemT64` / `kRoofSmemT128`, static_asserts
    /// in the kernel; the kernel traps on less).
    pub(crate) fn smem_bytes(self) -> u32 {
        match self {
            Tile::T64 => 30_752,
            Tile::T128 | Tile::T128W4 => 49_184,
        }
    }

    /// Launch grid for `rows` tokens of a `hidden`-wide row.
    pub(crate) fn grid(self, hidden: u32, rows: u32) -> [u32; 3] {
        [hidden / TILE_H, rows.div_ceil(self.rows()), 1]
    }
}

/// Parse the selector; anything other than these values fails the first prefill.
pub(crate) fn parse_selector(value: Option<&str>) -> Result<Option<Tile>, String> {
    match value.map(str::trim) {
        None | Some("") | Some("0") | Some("off") => Ok(None),
        Some("1") | Some("on") => Ok(Some(Tile::DEFAULT)),
        Some("t64") => Ok(Some(Tile::T64)),
        Some("t128") => Ok(Some(Tile::T128)),
        Some("t128_w4") | Some("w4") => Ok(Some(Tile::T128W4)),
        Some(other) => Err(format!(
            "{SELECTOR} must be absent, 0/off, 1/on, t64, t128 or t128_w4/w4; got {other:?}"
        )),
    }
}

/// Why a selected roof kernel did not run (the incumbent up+mix ran instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// `upmix` is not selected in ATLAS_QWEN4_PREFILL_MHC_FUSED: the
    /// incumbent this lever replaces does not run either.
    UpmixNotSelected,
    /// A pointer the roof kernel reads or writes with 16-byte accesses is not
    /// 16-byte aligned (or one is null).
    Alignment,
}

impl Refusal {
    pub(crate) const COUNT: usize = 2;

    pub(crate) fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn reason(self) -> &'static str {
        match self {
            Refusal::UpmixNotSelected => "upmix_not_selected",
            Refusal::Alignment => "alignment",
        }
    }
}

/// Device addresses of one up+mix launch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Io {
    pub act: u64,
    pub up: u64,
    pub hyper: u64,
    pub rstd: u64,
    pub norm_w: u64,
    pub mixed: u64,
}

/// 16-byte copies: act and up (K-loop cp.async, as the incumbent), hyper'
/// and rstd (epilogue cp.async), mixed (epilogue stores); norm_w is read as
/// bf16 pairs.
pub(crate) fn aligned(io: &Io) -> bool {
    [io.act, io.up, io.hyper, io.rstd, io.mixed]
        .iter()
        .all(|&p| p != 0 && p % 16 == 0)
        && io.norm_w != 0
        && io.norm_w.is_multiple_of(4)
}

/// Admission of a selected roof kernel where the HC_FUSED level-2 path is
/// about to run its up+mix.
pub(crate) fn admit(upmix_selected: bool, io: &Io) -> Result<(), Refusal> {
    if !upmix_selected {
        return Err(Refusal::UpmixNotSelected);
    }
    if !aligned(io) {
        return Err(Refusal::Alignment);
    }
    Ok(())
}

/// Admission of the `t128_w4` roof kernel, which reads `up_packed` (8-byte
/// `ld.global.nc.v2.u32`) and `up_scales` (1-byte `ld.global.nc.u8`) directly
/// instead of `io.up`.
pub(crate) fn admit_w4(
    upmix_selected: bool,
    io: &Io,
    up_packed: u64,
    up_scales: u64,
) -> Result<(), Refusal> {
    if !upmix_selected {
        return Err(Refusal::UpmixNotSelected);
    }
    let io_ok = [io.act, io.hyper, io.rstd, io.mixed]
        .iter()
        .all(|&p| p != 0 && p % 16 == 0)
        && io.norm_w != 0
        && io.norm_w.is_multiple_of(4)
        && up_packed != 0
        && up_packed.is_multiple_of(8)
        && up_scales != 0;
    if !io_ok {
        return Err(Refusal::Alignment);
    }
    Ok(())
}

/// Parse `ATLAS_QWEN4_PREFILL_UPMIX_MXFP8_FUSED` (default ON when unset/empty;
/// `0`/`off`/`false` opts out; anything else fails the first prefill).
pub(crate) fn parse_mxfp8_fused(value: Option<&str>) -> Result<bool, String> {
    let Some(raw) = value else {
        return Ok(true);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        other => Err(format!(
            "{MXFP8_FUSED_SELECTOR} must be absent, 0/off/false, or 1/on/true; got {other:?}"
        )),
    }
}

/// Device addresses and geometry of the fused MXFP8 `(xq, xs)` + optional
/// Last-CTA L2-hot shared-expert gate-scalar epilogue.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mxfp8GateIo {
    pub xq: u64,
    pub xs: u64,
    pub shared_gate: u64,
    pub ss_out: u64,
    pub tile_done: u64,
    pub hidden: usize,
    pub rows: usize,
}

fn ranges_disjoint(a: u64, a_len: usize, b: u64, b_len: usize) -> bool {
    if a == 0 || b == 0 || a_len == 0 || b_len == 0 {
        return true;
    }
    let Some(a_end) = a.checked_add(a_len as u64) else {
        return false;
    };
    let Some(b_end) = b.checked_add(b_len as u64) else {
        return false;
    };
    a_end <= b || b_end <= a
}

/// Pure admission for `qwen4_hc_upmix_roof_t128*_mxfp8_gate[_amax]`: checks
/// `H=2560`, `rows > 0`, `ceil(rows / 128) <= MXFP8_GATE_MAX_Y_TILES`, pointer
/// alignment (`xq` 8-byte for `st.global.v2.u32`, `ss_out`/`tile_done` 4-byte,
/// `shared_gate` 2-byte), and pairwise disjointness of all writable outputs and
/// `shared_gate`.
pub(crate) fn admit_mxfp8_gate(mixed: u64, io: &Mxfp8GateIo) -> bool {
    if io.hidden != 2560 || io.rows == 0 || io.rows.div_ceil(128) > MXFP8_GATE_MAX_Y_TILES {
        return false;
    }
    if mixed == 0 || io.xq == 0 || !io.xq.is_multiple_of(8) || io.xs == 0 {
        return false;
    }
    if io.ss_out != 0
        && (!io.ss_out.is_multiple_of(4)
            || io.tile_done == 0
            || !io.tile_done.is_multiple_of(4)
            || (io.shared_gate != 0 && !io.shared_gate.is_multiple_of(2)))
    {
        return false;
    }
    let Some(mixed_bytes) = io.rows.checked_mul(io.hidden * 2) else {
        return false;
    };
    let Some(xq_bytes) = io.rows.checked_mul(io.hidden) else {
        return false;
    };
    let Some(xs_bytes) = io.rows.checked_mul(io.hidden / 32) else {
        return false;
    };
    let Some(ss_bytes) = io.rows.checked_mul(4) else {
        return false;
    };
    let tile_done_bytes = MXFP8_GATE_MAX_Y_TILES * 4;
    let ranges = [
        (mixed, mixed_bytes),
        (io.xq, xq_bytes),
        (io.xs, xs_bytes),
        (io.ss_out, if io.ss_out != 0 { ss_bytes } else { 0 }),
        (
            io.tile_done,
            if io.ss_out != 0 { tile_done_bytes } else { 0 },
        ),
    ];
    for i in 0..ranges.len() {
        for j in (i + 1)..ranges.len() {
            if !ranges_disjoint(ranges[i].0, ranges[i].1, ranges[j].0, ranges[j].1) {
                return false;
            }
        }
    }
    if io.ss_out != 0 && io.shared_gate != 0 {
        let gate_bytes = io.hidden * 2;
        for &(dst, len) in &ranges {
            if !ranges_disjoint(io.shared_gate, gate_bytes, dst, len) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io() -> Io {
        Io {
            act: 0x7f00_0000_1000,
            up: 0x7f00_0010_0000,
            hyper: 0x7f00_1000_0000,
            rstd: 0x7f00_2000_0000,
            norm_w: 0x7f00_3000_0004,
            mixed: 0x7f00_4000_0000,
        }
    }

    #[test]
    fn selector_values() {
        assert_eq!(parse_selector(None), Ok(None));
        for off in ["", "0", "off", " 0 "] {
            assert_eq!(parse_selector(Some(off)), Ok(None), "{off:?}");
        }
        assert_eq!(parse_selector(Some("1")), Ok(Some(Tile::DEFAULT)));
        assert_eq!(parse_selector(Some("on")), Ok(Some(Tile::DEFAULT)));
        assert_eq!(parse_selector(Some("t64")), Ok(Some(Tile::T64)));
        assert_eq!(parse_selector(Some("t128")), Ok(Some(Tile::T128)));
        for bad in ["2", "true", "T64", "t256", "all", "t64,t128"] {
            assert!(parse_selector(Some(bad)).is_err(), "{bad:?} must fail");
        }
    }

    #[test]
    fn entries_are_distinct_and_named_after_the_tile() {
        let mut seen = std::collections::HashSet::new();
        for t in Tile::ALL {
            for amax in [false, true] {
                let e = t.entry(amax);
                assert!(e.starts_with("qwen4_hc_upmix_roof_"));
                assert!(e.contains(t.name()));
                assert_eq!(e.ends_with("_amax"), amax);
                assert!(seen.insert(e));
            }
        }
    }

    #[test]
    fn smem_matches_the_kernel_layout() {
        // RoofTile<BM, 2, STAGES, SWZ>::kSmem: max(ring, epilogue staging) + 8 warp maxima.
        let smem = |bm: u32, stages: u32, swz: bool| {
            let row = if swz { 64 } else { 80 };
            let ring = stages * (bm * row + 128 * row);
            let epi = bm * (272 + 16 + 80);
            ring.max(epi) + 8 * 4
        };
        assert_eq!(Tile::T64.smem_bytes(), smem(64, 2, false));
        assert_eq!(Tile::T128.smem_bytes(), smem(128, 3, true));
        // t64 keeps the incumbent's 3 CTAs/SM (102,400 B/SM, 1 KiB reserved per CTA);
        // t128 fits 2 (both above the 48 KiB default: the launcher raises the limit).
        assert!(3 * (Tile::T64.smem_bytes() + 1024) <= 102_400);
        assert!(2 * (Tile::T128.smem_bytes() + 1024) <= 102_400);
    }

    #[test]
    fn grid_covers_every_row_and_h_tile() {
        for t in Tile::ALL {
            for rows in [
                1u32, 63, 64, 65, 77, 127, 128, 129, 2048, 8192, 16384, 16385,
            ] {
                let [x, y, z] = t.grid(2560, rows);
                assert_eq!((x, z), (80, 1));
                assert!(
                    y * t.rows() >= rows && (y - 1) * t.rows() < rows,
                    "{t:?} rows {rows}"
                );
            }
        }
    }

    #[test]
    fn admission() {
        assert_eq!(admit(true, &io()), Ok(()));
        assert_eq!(admit(false, &io()), Err(Refusal::UpmixNotSelected));
        fn with(field: usize, f: impl Fn(u64) -> u64) -> Io {
            let mut x = io();
            let p = match field {
                0 => &mut x.act,
                1 => &mut x.up,
                2 => &mut x.hyper,
                3 => &mut x.rstd,
                _ => &mut x.mixed,
            };
            *p = f(*p);
            x
        }
        for field in 0..5 {
            // 4-byte aligned is enough for the incumbent, not here
            assert_eq!(
                admit(true, &with(field, |p| p + 4)),
                Err(Refusal::Alignment),
                "field {field}"
            );
            assert_eq!(
                admit(true, &with(field, |_| 0)),
                Err(Refusal::Alignment),
                "null field {field}"
            );
            assert_eq!(
                admit(true, &with(field, |p| p + 16)),
                Ok(()),
                "field {field} + 16"
            );
        }
        let mut x = io();
        x.norm_w += 2;
        assert_eq!(admit(true, &x), Err(Refusal::Alignment));
        x.norm_w = 0;
        assert_eq!(admit(true, &x), Err(Refusal::Alignment));
    }

    #[test]
    fn stream_entries_match_t128_tiles_and_stay_distinct() {
        assert_eq!(Tile::T64.stream_entry(false, false), None);
        assert_eq!(Tile::T64.stream_entry(true, true), None);
        let mut seen = std::collections::HashSet::new();
        for (tile, fused_prep) in [
            (Tile::T128, false),
            (Tile::T128, true),
            (Tile::T128W4, false),
        ] {
            for amax in [false, true] {
                let entry = tile
                    .stream_entry(fused_prep, amax)
                    .expect("t128 variants must expose streaming entries");
                assert!(entry.starts_with("qwen4_hc_upmix_roof_"));
                assert!(entry.contains("_stream"));
                assert_eq!(entry.ends_with("_amax"), amax);
                assert!(seen.insert(entry));
            }
        }
        assert_eq!(
            Tile::T128W4.stream_entry(true, false),
            Tile::T128W4.stream_entry(false, false)
        );
    }

    #[test]
    fn refusal_indices_are_dense() {
        let all = [Refusal::UpmixNotSelected, Refusal::Alignment];
        assert_eq!(all.len(), Refusal::COUNT);
        for (i, r) in all.iter().enumerate() {
            assert_eq!(r.index(), i);
            assert!(!r.reason().is_empty());
        }
    }

    #[test]
    fn parse_mxfp8_fused_defaults_on_and_validates() {
        assert_eq!(parse_mxfp8_fused(None), Ok(true));
        for on in ["", "1", "on", "true", " TRUE ", "On"] {
            assert_eq!(parse_mxfp8_fused(Some(on)), Ok(true), "{on:?}");
        }
        for off in ["0", "off", "false", " FALSE ", "Off"] {
            assert_eq!(parse_mxfp8_fused(Some(off)), Ok(false), "{off:?}");
        }
        for bad in ["2", "yes", "no", "mxfp8"] {
            assert!(parse_mxfp8_fused(Some(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn mxfp8_gate_entries_match_t128_tiles_and_stay_distinct() {
        assert_eq!(Tile::T64.mxfp8_gate_entry(false, false), None);
        assert_eq!(Tile::T64.mxfp8_gate_entry(true, true), None);
        let mut seen = std::collections::HashSet::new();
        for (tile, fused_prep) in [
            (Tile::T128, false),
            (Tile::T128, true),
            (Tile::T128W4, false),
        ] {
            for amax in [false, true] {
                let entry = tile
                    .mxfp8_gate_entry(fused_prep, amax)
                    .expect("t128 variants must expose mxfp8_gate entries");
                assert!(entry.starts_with("qwen4_hc_upmix_roof_"));
                assert!(entry.contains("_mxfp8_gate"));
                assert_eq!(entry.ends_with("_amax"), amax);
                assert!(seen.insert(entry));
            }
        }
        assert_eq!(
            Tile::T128W4.mxfp8_gate_entry(true, false),
            Tile::T128W4.mxfp8_gate_entry(false, false)
        );
    }

    #[test]
    fn stream_mxfp8_gate_entries_match_t128_tiles_and_stay_distinct() {
        assert_eq!(Tile::T64.stream_mxfp8_gate_entry(false, false), None);
        assert_eq!(Tile::T64.stream_mxfp8_gate_entry(true, true), None);
        let mut seen = std::collections::HashSet::new();
        for (tile, fused_prep) in [
            (Tile::T128, false),
            (Tile::T128, true),
            (Tile::T128W4, false),
        ] {
            for amax in [false, true] {
                let entry = tile
                    .stream_mxfp8_gate_entry(fused_prep, amax)
                    .expect("t128 variants must expose stream_mxfp8_gate entries");
                assert!(entry.starts_with("qwen4_hc_upmix_roof_"));
                assert!(entry.contains("_stream_mxfp8_gate"));
                assert_eq!(entry.ends_with("_amax"), amax);
                assert!(seen.insert(entry));
            }
        }
        assert_eq!(
            Tile::T128W4.stream_mxfp8_gate_entry(true, false),
            Tile::T128W4.stream_mxfp8_gate_entry(false, false)
        );
    }

    #[test]
    fn admit_mxfp8_gate_checks_geometry_alignment_and_disjointness() {
        let rows = 16_384usize;
        let mixed = 0x7f00_0000_0000u64;
        let xq = 0x7f00_1000_0000u64;
        let xs = xq + (rows * 2560) as u64; // contiguous right after xq (expert_up_out ABI)
        let base = Mxfp8GateIo {
            xq,
            xs,
            shared_gate: 0x7f00_2000_0000,
            ss_out: 0x7f00_3000_0000,
            tile_done: 0x7f00_4000_0000,
            hidden: 2560,
            rows,
        };
        assert!(admit_mxfp8_gate(mixed, &base));

        // ss_out == 0 disables the Last-CTA gate path (tile_done and shared_gate may be 0).
        let no_gate = Mxfp8GateIo {
            shared_gate: 0,
            ss_out: 0,
            tile_done: 0,
            ..base
        };
        assert!(admit_mxfp8_gate(mixed, &no_gate));

        // shared_gate == 0 with ss_out != 0 is valid (kernel fills ss_out = 1.0f).
        let ones_gate = Mxfp8GateIo {
            shared_gate: 0,
            ..base
        };
        assert!(admit_mxfp8_gate(mixed, &ones_gate));

        // Reject bad hidden, zero rows, and rows exceeding tile_done capacity.
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                hidden: 2048,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(mixed, &Mxfp8GateIo { rows: 0, ..base }));
        let max_rows = MXFP8_GATE_MAX_Y_TILES * 128;
        let max_xq = 0x7f01_0000_0000u64;
        assert!(admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                rows: max_rows,
                xq: max_xq,
                xs: max_xq + (max_rows * 2560) as u64,
                shared_gate: 0x7f02_0000_0000,
                ss_out: 0x7f03_0000_0000,
                tile_done: 0x7f04_0000_0000,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                rows: max_rows + 1,
                xq: max_xq,
                xs: max_xq + ((max_rows + 1) * 2560) as u64,
                shared_gate: 0x7f02_0000_0000,
                ss_out: 0x7f03_0000_0000,
                tile_done: 0x7f04_0000_0000,
                ..base
            }
        ));

        // Alignment and non-null checks.
        assert!(!admit_mxfp8_gate(0, &base));
        assert!(!admit_mxfp8_gate(mixed, &Mxfp8GateIo { xq: 0, ..base }));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo { xq: xq + 4, ..base }
        ));
        assert!(!admit_mxfp8_gate(mixed, &Mxfp8GateIo { xs: 0, ..base }));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                ss_out: base.ss_out + 2,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                tile_done: 0,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                tile_done: base.tile_done + 2,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                shared_gate: base.shared_gate + 1,
                ..base
            }
        ));

        // Disjointness checks (1-byte overlap rejected).
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo { xs: xs - 1, ..base }
        ));
        assert!(!admit_mxfp8_gate(mixed, &Mxfp8GateIo { xq: mixed, ..base }));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                ss_out: mixed + 16,
                ..base
            }
        ));
        assert!(!admit_mxfp8_gate(
            mixed,
            &Mxfp8GateIo {
                shared_gate: base.ss_out,
                ..base
            }
        ));
    }
}
