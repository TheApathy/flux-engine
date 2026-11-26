// SPDX-License-Identifier: AGPL-3.0-only

//! First-touch initialisation of lazily grown device scratch
//! (`ATLAS_PREFILL_SCRATCH_INIT`).
//!
//! The fast prefill keeps grow-only scratch (HC_FUSED, the postpre formed rows
//! and routing copy, the FP8 projection scratch, the QSA sparse scratch) whose
//! first allocation comes back from the driver with whatever an earlier owner
//! left there. A kernel that reads a region it did not write in the same
//! request then sees that content once (on the first request large enough to
//! grow the block) and the previous request's data afterwards. Initialising
//! every newly grown block makes the first touch deterministic, and `poison`
//! turns any such read into NaNs (0xFF bytes: NaN in bf16/f16/f32, -1 as an
//! index), which is how the reader is found.
//!
//! - unset or `zero` (default): grown scratch is zeroed; the arena and the KV
//!   pools are left as is.
//! - `poison`: grown scratch and, at load, every arena buffer and KV pool are
//!   filled with 0xFF, and so is the QSA side cache (zeroed in every other mode).
//! - `arena-zero`: grown scratch, the arena and the KV pools are zeroed.
//! - `off`: no initialisation (the behaviour before this module).
//!
//! The FlashInfer GDN bridges read the same variable (`csrc/fi_gdn`), and the
//! Qwen4 scratch that used to grow and free mid-request is reserved at load
//! (`factory/qwen4_prefill_reserve.rs`); growing past it retires, never frees.

use std::sync::OnceLock;

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend};

pub const SELECTOR: &str = "ATLAS_PREFILL_SCRATCH_INIT";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Zero,
    ArenaZero,
    Poison,
}

impl Mode {
    /// The byte grown scratch is filled with, if any.
    pub fn scratch_byte(self) -> Option<u8> {
        match self {
            Self::Off => None,
            Self::Zero | Self::ArenaZero => Some(0),
            Self::Poison => Some(0xFF),
        }
    }

    /// The byte arena buffers are filled with at load, if any.
    pub fn arena_byte(self) -> Option<u8> {
        match self {
            Self::Off | Self::Zero => None,
            Self::ArenaZero => Some(0),
            Self::Poison => Some(0xFF),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Zero => "zero",
            Self::ArenaZero => "arena-zero",
            Self::Poison => "poison",
        }
    }
}

fn parse(value: Option<&str>) -> Result<Mode, String> {
    match value {
        None | Some("") | Some("zero") => Ok(Mode::Zero),
        Some("off") => Ok(Mode::Off),
        Some("arena-zero") => Ok(Mode::ArenaZero),
        Some("poison") => Ok(Mode::Poison),
        Some(other) => Err(format!(
            "{SELECTOR} must be zero, arena-zero, poison or off; got {other:?}"
        )),
    }
}

pub fn mode() -> Result<Mode> {
    static MODE: OnceLock<Result<Mode, String>> = OnceLock::new();
    MODE.get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Initialise a block a grow-only scratch just allocated (synchronous memset:
/// the block is not in use yet, and the caller's stream may not be ordered
/// with the default stream). Logs `PREFILL_SCRATCH_INIT` per block.
pub fn init_grown(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize, site: &str) -> Result<()> {
    let mode = mode()?;
    if let Some(byte) = mode.scratch_byte() {
        gpu.memset(ptr, byte, bytes)?;
    }
    tracing::info!(
        "PREFILL_SCRATCH_INIT selector={SELECTOR} mode={} site={site} bytes={bytes}",
        mode.name()
    );
    Ok(())
}

/// Initialise one arena buffer or KV pool at load (`poison` / `arena-zero` only).
pub fn init_arena(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<bool> {
    match mode()?.arena_byte() {
        Some(byte) if bytes > 0 => {
            gpu.memset(ptr, byte, bytes)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modes() {
        assert_eq!(parse(None), Ok(Mode::Zero));
        assert_eq!(parse(Some("zero")), Ok(Mode::Zero));
        assert_eq!(parse(Some("off")), Ok(Mode::Off));
        assert_eq!(parse(Some("arena-zero")), Ok(Mode::ArenaZero));
        assert_eq!(parse(Some("poison")), Ok(Mode::Poison));
        assert!(parse(Some("1")).is_err());
    }

    #[test]
    fn bytes_per_mode() {
        assert_eq!(Mode::Zero.scratch_byte(), Some(0));
        assert_eq!(Mode::Zero.arena_byte(), None);
        assert_eq!(Mode::Poison.scratch_byte(), Some(0xFF));
        assert_eq!(Mode::Poison.arena_byte(), Some(0xFF));
        assert_eq!(Mode::ArenaZero.arena_byte(), Some(0));
        assert_eq!(Mode::Off.scratch_byte(), None);
    }
}
