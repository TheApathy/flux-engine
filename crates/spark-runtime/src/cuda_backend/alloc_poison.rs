// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_ALLOC_POISON=nan|ff|zero` (default off): fill every device
//! allocation of at least `ATLAS_ALLOC_POISON_MIN_BYTES` (default 1 MiB) at
//! allocation time, before any caller can use it. A kernel that reads memory
//! nobody wrote then sees the same bytes in every process and on every first
//! touch, so a "first request to reach a size diverges" split either
//! disappears (`zero`) or becomes loud (`nan`, `ff`). Grow-only reallocations
//! allocate through the same `alloc`, so they are covered.
//!
//! - `nan`: 32-bit pattern `0x7FC07FC0` — a quiet NaN as fp32 and as each
//!   bf16 half; as an i32 index a large positive value (out of range).
//! - `ff`: every byte `0xFF` — NaN as bf16/fp16/fp32/e4m3, `-1` as i32,
//!   `-6.0` as each e2m1 nibble.
//! - `zero`: the zero-filled control (deterministic, not loud).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_MIN_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoisonMode {
    Nan,
    Ff,
    Zero,
}

impl PoisonMode {
    pub(super) fn pattern(self) -> u32 {
        match self {
            Self::Nan => 0x7FC0_7FC0,
            Self::Ff => 0xFFFF_FFFF,
            Self::Zero => 0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nan => "nan",
            Self::Ff => "ff",
            Self::Zero => "zero",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoisonConfig {
    pub(super) mode: PoisonMode,
    pub(super) min_bytes: usize,
}

/// Parse the two env values. `None` = off. An unrecognised mode or a
/// malformed minimum is an error (logged; the detector stays off, so the
/// missing `ALLOC_POISON_ENGAGED` receipt shows it) rather than a guess.
fn parse(mode: Option<&str>, min_bytes: Option<&str>) -> Result<Option<PoisonConfig>, String> {
    let mode = match mode {
        None | Some("") | Some("0") | Some("off") => return Ok(None),
        Some("nan") => PoisonMode::Nan,
        Some("ff") => PoisonMode::Ff,
        Some("zero") => PoisonMode::Zero,
        Some(other) => return Err(format!("ATLAS_ALLOC_POISON={other:?} is not nan|ff|zero")),
    };
    let min_bytes = match min_bytes {
        None => DEFAULT_MIN_BYTES,
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| format!("ATLAS_ALLOC_POISON_MIN_BYTES={raw:?} is not a byte count"))?,
    };
    Ok(Some(PoisonConfig { mode, min_bytes }))
}

/// The process-wide selection, read once. Logs the `ALLOC_POISON_ENGAGED`
/// receipt on first use when on.
pub(super) fn selected() -> Option<PoisonConfig> {
    static CONFIG: OnceLock<Option<PoisonConfig>> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let mode = std::env::var("ATLAS_ALLOC_POISON").ok();
        let min = std::env::var("ATLAS_ALLOC_POISON_MIN_BYTES").ok();
        match parse(mode.as_deref(), min.as_deref()) {
            Ok(Some(config)) => {
                tracing::warn!(
                    "ALLOC_POISON_ENGAGED selector=ATLAS_ALLOC_POISON mode={} pattern=0x{:08x} \
                     min_bytes={} (diagnostic: every device alloc/alloc_managed at or above \
                     min_bytes is filled before use, grow-only reallocations included)",
                    config.mode.name(),
                    config.mode.pattern(),
                    config.min_bytes
                );
                Some(config)
            }
            Ok(None) => None,
            Err(message) => {
                tracing::error!(
                    "{message}; ATLAS_ALLOC_POISON stays OFF (no ALLOC_POISON_ENGAGED)"
                );
                None
            }
        }
    })
}

/// 32-bit words then trailing bytes for a fill of `bytes` from a
/// `cuMemAlloc`-aligned (256 B) base.
pub(super) fn fill_plan(bytes: usize) -> (usize, usize) {
    (bytes / 4, bytes % 4)
}

static POISONED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Record one filled allocation (debug level: weight loading makes thousands;
/// the lazy scratches the detector targets log their own allocation lines).
pub(super) fn note_poisoned(dptr: u64, bytes: usize) {
    let total = POISONED_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    tracing::debug!("ALLOC_POISON filled ptr=0x{dptr:x} bytes={bytes} total_filled={total}");
}

/// An allocation made while the default stream captures a CUDA graph cannot
/// be filled without the fill joining the graph; it is left unfilled and
/// reported, so a clean poisoned run is not over-claimed.
pub(super) fn note_skipped_in_capture(dptr: u64, bytes: usize) {
    tracing::warn!(
        "ALLOC_POISON skipped ptr=0x{dptr:x} bytes={bytes}: default stream is capturing a graph \
         (this allocation is NOT poisoned)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_a_mode_is_named() {
        assert_eq!(parse(None, None), Ok(None));
        assert_eq!(parse(Some(""), None), Ok(None));
        assert_eq!(parse(Some("0"), None), Ok(None));
        assert_eq!(parse(Some("off"), Some("8")), Ok(None));
    }

    #[test]
    fn modes_and_default_minimum() {
        let nan = parse(Some("nan"), None).unwrap().unwrap();
        assert_eq!(nan.mode, PoisonMode::Nan);
        assert_eq!(nan.min_bytes, 1 << 20);
        assert_eq!(
            parse(Some("ff"), None).unwrap().unwrap().mode,
            PoisonMode::Ff
        );
        assert_eq!(
            parse(Some("zero"), Some("0")).unwrap().unwrap().min_bytes,
            0
        );
    }

    #[test]
    fn refuses_unknown_mode_and_bad_minimum() {
        assert!(parse(Some("NaN"), None).is_err());
        assert!(parse(Some("1"), None).is_err());
        assert!(parse(Some("nan"), Some("1MiB")).is_err());
    }

    #[test]
    fn nan_pattern_is_nan_in_fp32_and_both_bf16_halves() {
        let bits = PoisonMode::Nan.pattern();
        assert!(f32::from_bits(bits).is_nan());
        for half in [bits >> 16, bits & 0xFFFF] {
            assert!(f32::from_bits(half << 16).is_nan());
        }
        assert!(f32::from_bits(PoisonMode::Ff.pattern()).is_nan());
        assert!(f32::from_bits(0xFFFF << 16).is_nan());
    }

    #[test]
    fn fill_plan_covers_every_byte() {
        for bytes in [0usize, 1, 3, 4, 5, 1 << 20, (1 << 20) + 3] {
            let (words, tail) = fill_plan(bytes);
            assert_eq!(words * 4 + tail, bytes);
            assert!(tail < 4);
        }
    }
}
