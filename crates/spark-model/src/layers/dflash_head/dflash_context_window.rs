// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, fail-closed resolution of the DFlash resident attention window.

use std::fmt;

use atlas_kernels::{DrafterAdmission, drafter_profiles_for};

/// The qualified resident ring used by native Qwen3.8-Flash-Next drafter
/// checkpoints. Absolute positions may continue to 1M; only this tail is
/// resident in the drafter.
pub(crate) const NATIVE_FLASH_NEXT_WINDOW: usize = 4096;
pub(crate) const MAX_ABSOLUTE_CONTEXT: usize = 1_048_576;

#[allow(clippy::too_many_arguments)]
pub(crate) fn is_native_flash_next(
    architectures: &[String],
    model_type: Option<&str>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_target_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    target_layer_ids: &[usize],
) -> bool {
    let [architecture] = architectures else {
        return false;
    };
    drafter_profiles_for("qwen3.8-flash-next").any(|profile| {
        if !matches!(
            profile.admission,
            DrafterAdmission::Exact | DrafterAdmission::Quorum
        ) {
            return false;
        }
        let id = &profile.drafter;
        id.architecture == Some(architecture.as_str())
            && id
                .model_type
                .is_none_or(|expected| model_type == Some(expected))
            && id.hidden_size == Some(hidden_size)
            && id.intermediate_size == Some(intermediate_size)
            && id.num_hidden_layers == Some(num_hidden_layers)
            && id.num_target_layers == Some(num_target_layers)
            && id.num_attention_heads == Some(num_attention_heads)
            && id.num_key_value_heads == Some(num_key_value_heads)
            && id.head_dim == Some(head_dim)
            && id.vocab_size == Some(vocab_size)
            && id.target_layer_ids == Some(target_layer_ids)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextWindowError {
    ZeroMaxSequenceLength,
    AbsoluteContextTooLong {
        max_seq_len: usize,
    },
    OutsideResidentRange {
        requested: usize,
    },
    NativeFlashNextSequenceTooShort {
        max_seq_len: usize,
        required: usize,
    },
    NativeFlashNextMismatch {
        requested: usize,
        sliding_window: Option<usize>,
    },
}

impl fmt::Display for ContextWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxSequenceLength => {
                f.write_str("DFlash requires max_seq_len greater than zero")
            }
            Self::AbsoluteContextTooLong { max_seq_len } => write!(
                f,
                "DFlash absolute context {max_seq_len} exceeds the qualified limit {MAX_ABSOLUTE_CONTEXT}"
            ),
            Self::OutsideResidentRange { requested } => write!(
                f,
                "DFlash resident context window {requested} is outside the qualified range 1..={NATIVE_FLASH_NEXT_WINDOW}"
            ),
            Self::NativeFlashNextSequenceTooShort {
                max_seq_len,
                required,
            } => write!(
                f,
                "native Flash-Next V3/DFlash2 requires max_seq_len at least {required}; got {max_seq_len}"
            ),
            Self::NativeFlashNextMismatch {
                requested,
                sliding_window: Some(sw),
            } if *sw != NATIVE_FLASH_NEXT_WINDOW => write!(
                f,
                "native Flash-Next V3/DFlash2 requires resident context window {sw} or {NATIVE_FLASH_NEXT_WINDOW}; got {requested}"
            ),
            Self::NativeFlashNextMismatch { requested, .. } => write!(
                f,
                "native Flash-Next V3/DFlash2 requires resident context window {NATIVE_FLASH_NEXT_WINDOW}; got {requested}"
            ),
        }
    }
}

/// Return `Some(sw)` when EVERY drafter attention layer declares
/// `"sliding_attention"` with a valid bounded `sliding_window` (`1..=4096`).
///
/// For all-sliding checkpoints such as `incoai-dflash2-b8` (`5` layers, all
/// `sliding_attention`, `sliding_window = 2048`), keys older than `sw` are
/// mathematically masked out on every layer, so capping the resident ring and
/// KV caches at `sw` is bit-identical and halves per-sequence drafter VRAM.
#[must_use]
pub(crate) fn effective_all_layers_sliding_window(
    layer_types: Option<&[String]>,
    sliding_window: Option<usize>,
    num_layers: usize,
) -> Option<usize> {
    let sw = sliding_window?;
    let types = layer_types?;
    if num_layers == 0 || types.len() != num_layers || !(1..=NATIVE_FLASH_NEXT_WINDOW).contains(&sw)
    {
        return None;
    }
    types.iter().all(|t| t == "sliding_attention").then_some(sw)
}

/// Resolve the resident attention window, honoring an all-layers
/// `sliding_window` (e.g. `2048` for `incoai-dflash2-b8`) when present so
/// checkpoints whose attention layers can never read past `sw` do not
/// over-allocate `4096`-slot KV/hidden rings.
pub(crate) fn resolve_context_window_with_sliding(
    requested: Option<usize>,
    max_seq_len: usize,
    native_flash_next: bool,
    all_layers_sliding_window: Option<usize>,
) -> Result<usize, ContextWindowError> {
    let Some(sw) = all_layers_sliding_window.filter(|w| (1..=NATIVE_FLASH_NEXT_WINDOW).contains(w))
    else {
        return resolve_context_window(requested, max_seq_len, native_flash_next);
    };
    if max_seq_len == 0 {
        return Err(ContextWindowError::ZeroMaxSequenceLength);
    }
    if max_seq_len > MAX_ABSOLUTE_CONTEXT {
        return Err(ContextWindowError::AbsoluteContextTooLong { max_seq_len });
    }
    if native_flash_next {
        let resolved = requested.unwrap_or(max_seq_len);
        if !(1..=NATIVE_FLASH_NEXT_WINDOW).contains(&resolved) {
            return Err(ContextWindowError::OutsideResidentRange {
                requested: resolved,
            });
        }
        // All-sliding checkpoints (e.g. `incoai-dflash2-b8`, `sw = 2048`) accept
        // their native sliding window, the canonical 4096 ceiling (capped to `sw`),
        // or bounded power-of-two sub-windows (`256 | 512 | 1024`) needed to fit
        // wide discrete batched cohorts (`C=48..64`) inside 96 GB VRAM while
        // keeping target verification lossless.
        let allowed_sub_window = matches!(resolved, 256 | 512 | 1024) && resolved <= sw;
        if resolved != sw && resolved != NATIVE_FLASH_NEXT_WINDOW && !allowed_sub_window {
            return Err(ContextWindowError::NativeFlashNextMismatch {
                requested: resolved,
                sliding_window: Some(sw),
            });
        }
        let effective = resolved.min(sw);
        if max_seq_len < effective {
            return Err(ContextWindowError::NativeFlashNextSequenceTooShort {
                max_seq_len,
                required: effective,
            });
        }
        return Ok(effective);
    }
    let base = resolve_context_window(requested, max_seq_len, false)?;
    Ok(base.min(sw))
}

/// Resolve the single value used for allocation and runtime attention.
///
/// `None` preserves the legacy CLI meaning of full attention, but only when
/// the configured maximum sequence length fits inside the qualified resident
/// ring. It never silently truncates a requested full window.
pub(crate) fn resolve_context_window(
    requested: Option<usize>,
    max_seq_len: usize,
    native_flash_next: bool,
) -> Result<usize, ContextWindowError> {
    if max_seq_len == 0 {
        return Err(ContextWindowError::ZeroMaxSequenceLength);
    }
    if max_seq_len > MAX_ABSOLUTE_CONTEXT {
        return Err(ContextWindowError::AbsoluteContextTooLong { max_seq_len });
    }
    if native_flash_next && max_seq_len < NATIVE_FLASH_NEXT_WINDOW {
        return Err(ContextWindowError::NativeFlashNextSequenceTooShort {
            max_seq_len,
            required: NATIVE_FLASH_NEXT_WINDOW,
        });
    }

    let resolved = requested.unwrap_or(max_seq_len);
    if !(1..=NATIVE_FLASH_NEXT_WINDOW).contains(&resolved) {
        return Err(ContextWindowError::OutsideResidentRange {
            requested: resolved,
        });
    }
    if native_flash_next && resolved != NATIVE_FLASH_NEXT_WINDOW {
        return Err(ContextWindowError::NativeFlashNextMismatch {
            requested: resolved,
            sliding_window: None,
        });
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_errors_report_active_sliding_window() {
        let types = vec!["sliding_attention".to_string(); 5];
        assert_eq!(
            effective_all_layers_sliding_window(Some(&types), Some(2048), 5),
            Some(2048)
        );
        let short_err =
            resolve_context_window_with_sliding(Some(2048), 1024, true, Some(2048)).unwrap_err();
        let short_msg = short_err.to_string();
        assert!(
            short_msg.contains("at least 2048"),
            "expected sliding window in error, got: {short_msg}"
        );
        assert!(
            !short_msg.contains("at least 4096"),
            "unexpected hardcoded 4096 in error: {short_msg}"
        );

        assert_eq!(
            resolve_context_window_with_sliding(Some(512), 8192, true, Some(2048)),
            Ok(512)
        );
        let mismatch_err =
            resolve_context_window_with_sliding(Some(768), 8192, true, Some(2048)).unwrap_err();
        let mismatch_msg = mismatch_err.to_string();
        assert!(
            mismatch_msg.contains("2048 or 4096"),
            "expected sliding window in mismatch error, got: {mismatch_msg}"
        );

        let plain_err = resolve_context_window(Some(512), 8192, true).unwrap_err();
        assert!(plain_err.to_string().contains("window 4096; got 512"));
    }
}
