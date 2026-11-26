// SPDX-License-Identifier: AGPL-3.0-only

//! Immutable process-level speculation control.
//!
//! `ATLAS_DISABLE_SPECULATION` is a qualification control, not a hot-path
//! tuning knob.  Parse it exactly once while constructing the model, reject
//! every spelling except absent/`0`/`1`, and carry the resulting bool with the
//! model.  This prevents the scheduler and MTP prompt-tail replay from seeing
//! different values if the process environment is mutated after startup.

use anyhow::{Result, bail};

pub(super) fn parse_forced_disabled(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("ATLAS_DISABLE_SPECULATION must be exactly 0 or 1"),
    }
}

pub(super) fn forced_disabled_from_env() -> Result<bool> {
    match std::env::var("ATLAS_DISABLE_SPECULATION") {
        Ok(value) => parse_forced_disabled(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_forced_disabled(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("ATLAS_DISABLE_SPECULATION is not valid Unicode")
        }
    }
}

/// Execute one complete last-K operation only when the immutable control
/// admits it.  Callers put *all* host-ring, rotary, proposer, and GPU work in
/// `operation`, so a loaded-serial control cannot partially touch state.
pub(super) fn run_lastk_if_enabled<T>(
    forced_disabled: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    if forced_disabled {
        return Ok(None);
    }
    operation().map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ImageSpan, RotaryPositions};

    fn image_positions() -> RotaryPositions {
        RotaryPositions::from_image_spans(
            6,
            32,
            &[ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap()
    }

    #[test]
    fn switch_is_exact_and_fail_closed() {
        assert!(!parse_forced_disabled(None).unwrap());
        assert!(!parse_forced_disabled(Some("0")).unwrap());
        assert!(parse_forced_disabled(Some("1")).unwrap());
        for hostile in ["", "true", "01", " 1", "1 ", "yes"] {
            assert!(parse_forced_disabled(Some(hostile)).is_err(), "{hostile:?}");
        }
    }

    #[test]
    fn loaded_serial_skips_all_text_and_image_lastk_work() {
        for initial_rotary in [RotaryPositions::identity(), image_positions()] {
            let expected_rotary = initial_rotary.clone();
            let mut rotary = initial_rotary;
            let mut ring = vec![0x5a; 16];
            let expected_ring = ring.clone();
            let mut copy_count = 0usize;

            let result = run_lastk_if_enabled(true, || {
                copy_count += 1;
                ring.fill(0xa5);
                rotary = RotaryPositions::identity();
                Ok(())
            })
            .unwrap();

            assert_eq!(result, None);
            assert_eq!(copy_count, 0);
            assert_eq!(ring, expected_ring);
            assert_eq!(rotary, expected_rotary);
        }
    }

    #[test]
    fn engaged_mode_runs_lastk_work() {
        let mut copy_count = 0usize;
        let result = run_lastk_if_enabled(false, || {
            copy_count += 1;
            Ok(17usize)
        })
        .unwrap();
        assert_eq!(result, Some(17));
        assert_eq!(copy_count, 1);
    }
}
