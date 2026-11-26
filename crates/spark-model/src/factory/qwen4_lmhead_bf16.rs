// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_LMHEAD_BF16=1`: opt-in LM-head implementation/precision
//! control for qwen4_exp (default-off diagnostic).
//!
//! Absent or `0` defers to `ModelConfig::skip_lm_head_quantization`, so every
//! model keeps today's head. With the selector set, a qwen4_exp target keeps
//! the checkpoint's BF16 `lm_head.weight` (no runtime NVFP4 copy, no
//! per-tensor scale2) and the target head takes its existing dense branch
//! (`impl_a3.rs` `lm_head` / `lm_head_batched`), which also changes the GEMV
//! arithmetic and reduction order. The control is qualified only for the
//! fixed C1, no-speculation, no-n-gram recipe. The refusals below cover the
//! routes this factory can see; n-gram drafting and batch width are decided
//! outside it and are not covered here.

use anyhow::{Result, bail, ensure};
use spark_runtime::weights::WeightDtype;

pub(crate) const QWEN4_LMHEAD_BF16_ENV: &str = "ATLAS_QWEN4_LMHEAD_BF16";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LmHeadPrecision {
    Nvfp4,
    Bf16,
}

/// Every input the rule reads, captured once so the rule itself is pure.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LmHeadInputs {
    pub(crate) is_qwen4: bool,
    /// `ModelConfig::skip_lm_head_quantization()` (MLA, dense gemma4).
    pub(crate) skip_quant: bool,
    pub(crate) use_native_mtp: bool,
    pub(crate) has_dflash: bool,
    pub(crate) self_speculative: bool,
    /// `ATLAS_QWEN4_NATIVE_BATCH=1`: native C2-C32 decode asserts NVFP4.
    pub(crate) native_batch: bool,
    /// `ATLAS_LM_HEAD_TC=1` / `ATLAS_LM_HEAD_T=1` / `ATLAS_DECODE_TC_PARITY=1`:
    /// transposed-NVFP4 routes that go inert without an NVFP4 head.
    pub(crate) lm_head_tc: bool,
    pub(crate) lm_head_t: bool,
    pub(crate) decode_tc_parity: bool,
    /// `ATLAS_GEMMA4_FP32_LMHEAD`: would also change the logits dtype.
    pub(crate) fp32_logits: bool,
    /// `ATLAS_TARGET_LMHEAD_VOCAB` present with any value other than `0`
    /// (after trim; empty or unparsable values also refuse): verify-only cap.
    pub(crate) verify_vocab_cap: bool,
}

/// Kept for the module's tests; the live read goes through
/// `qwen4_native_selectors`, which applies the same 0/1 rule.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_selector(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{QWEN4_LMHEAD_BF16_ENV} must be exactly 0 or 1"),
    }
}

pub(crate) fn decide_lm_head_precision(
    selected: bool,
    inputs: LmHeadInputs,
) -> Result<LmHeadPrecision> {
    if !selected {
        return Ok(if inputs.skip_quant {
            LmHeadPrecision::Bf16
        } else {
            LmHeadPrecision::Nvfp4
        });
    }
    if !inputs.is_qwen4 {
        bail!("{QWEN4_LMHEAD_BF16_ENV}=1 is a qwen4_exp control; unset it for this model");
    }
    let refusals = [
        (
            inputs.use_native_mtp,
            "native/legacy MTP shares the NVFP4 LM head; disable speculation",
        ),
        (
            inputs.has_dflash,
            "DFlash is not qualified with a BF16 target head (conservative refusal); drop --dflash",
        ),
        (
            inputs.self_speculative,
            "self-speculative verify is not qualified on a BF16 LM head",
        ),
        (
            inputs.native_batch,
            "native Qwen4 batch decode requires an NVFP4 LM head; unset ATLAS_QWEN4_NATIVE_BATCH",
        ),
        (
            inputs.lm_head_tc,
            "ATLAS_LM_HEAD_TC routes need the NVFP4-T head; unset it",
        ),
        (
            inputs.lm_head_t,
            "ATLAS_LM_HEAD_T builds from the NVFP4 head; unset it",
        ),
        (
            inputs.decode_tc_parity,
            "ATLAS_DECODE_TC_PARITY needs the NVFP4-T head; unset it",
        ),
        (
            inputs.fp32_logits,
            "keep BF16 logits so the head weights are the only variable; unset ATLAS_GEMMA4_FP32_LMHEAD",
        ),
        (
            inputs.verify_vocab_cap,
            "verify-side vocab truncation is not part of this control; unset ATLAS_TARGET_LMHEAD_VOCAB",
        ),
    ];
    if let Some((_, reason)) = refusals.iter().find(|(blocked, _)| *blocked) {
        bail!("{QWEN4_LMHEAD_BF16_ENV}=1 refused: {reason}");
    }
    Ok(LmHeadPrecision::Bf16)
}

/// The checkpoint head as the `WeightStore` holds it. `dtype` is the stored
/// safetensors dtype (`weights/loader/load_fns.rs:76,160`,
/// `fast_weights/header.rs:118-130`), recorded before any loader runs.
/// `loaded_ptr` is what `load_lm_head` returned. The BF16 branch is a pointer
/// alias (`qwen35_mixed_precision.rs:111`, `dense()`); the FP8 (:72-75) and
/// ModelOpt NVFP4 (:76-103) branches return a fresh dequantized allocation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StoredHead<'a> {
    pub(crate) dtype: WeightDtype,
    pub(crate) shape: &'a [usize],
    pub(crate) stored_ptr: u64,
    pub(crate) loaded_ptr: u64,
}

/// The control is only meaningful for a head that is BF16 *in the
/// checkpoint*. A shape-compatible FP8 or packed head would otherwise pass
/// as a dequantized BF16 copy.
pub(crate) fn validate_stored_bf16_head(
    key: &str,
    head: StoredHead<'_>,
    vocab: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        head.dtype == WeightDtype::BF16,
        "{QWEN4_LMHEAD_BF16_ENV}=1 refused: {key} is stored as {:?}, not BF16; \
         a dequantized head is not a BF16 control",
        head.dtype
    );
    ensure!(
        head.loaded_ptr == head.stored_ptr,
        "{QWEN4_LMHEAD_BF16_ENV}=1 refused: the loaded head is not the stored {key} tensor"
    );
    ensure!(
        head.shape.len() == 2 && head.shape[0] >= vocab && head.shape[1] == hidden,
        "{QWEN4_LMHEAD_BF16_ENV}=1 refused: {key} shape {:?} does not cover vocab {vocab} x hidden {hidden}",
        head.shape
    );
    ensure!(
        head.loaded_ptr != 0 && head.loaded_ptr.is_multiple_of(16),
        "{QWEN4_LMHEAD_BF16_ENV}=1 refused: {key} is not 16-byte aligned for dense_gemv_bf16"
    );
    Ok(())
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// Reads the selector first and touches no other variable unless it is set,
/// so the default path is the unmodified `skip_lm_head_quantization` answer.
pub(crate) fn lm_head_precision_from_env(
    is_qwen4: bool,
    skip_quant: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    self_speculative: bool,
) -> Result<LmHeadPrecision> {
    // Routed through `qwen4_native_selectors`: `ATLAS_QWEN4_DENSE_NATIVE=1`
    // implies this control unless the env is explicitly `0`.
    let selected =
        crate::qwen4_native_selectors::selected(crate::qwen4_native_selectors::Knob::LmHead)?;
    let mut inputs = LmHeadInputs {
        is_qwen4,
        skip_quant,
        ..LmHeadInputs::default()
    };
    if selected {
        inputs = LmHeadInputs {
            use_native_mtp,
            has_dflash,
            self_speculative,
            native_batch: env_is_one("ATLAS_QWEN4_NATIVE_BATCH"),
            lm_head_tc: env_is_one("ATLAS_LM_HEAD_TC"),
            lm_head_t: env_is_one("ATLAS_LM_HEAD_T"),
            decode_tc_parity: env_is_one("ATLAS_DECODE_TC_PARITY"),
            fp32_logits: matches!(
                std::env::var("ATLAS_GEMMA4_FP32_LMHEAD").ok().as_deref(),
                Some("1") | Some("true")
            ),
            verify_vocab_cap: std::env::var("ATLAS_TARGET_LMHEAD_VOCAB")
                .ok()
                .is_some_and(|raw| raw.trim() != "0"),
            ..inputs
        };
    }
    decide_lm_head_precision(selected, inputs)
}

#[cfg(test)]
mod tests {
    use super::{
        LmHeadInputs, LmHeadPrecision, StoredHead, WeightDtype, decide_lm_head_precision,
        parse_selector, validate_stored_bf16_head,
    };

    const SHAPE: [usize; 2] = [248_320, 2_560];

    fn stored(dtype: WeightDtype) -> StoredHead<'static> {
        StoredHead {
            dtype,
            shape: &SHAPE,
            stored_ptr: 0x1000,
            loaded_ptr: 0x1000,
        }
    }

    #[test]
    fn stored_head_must_be_checkpoint_bf16_alias() {
        let check = |head| validate_stored_bf16_head("lm_head.weight", head, 248_077, 2_560);
        assert!(check(stored(WeightDtype::BF16)).is_ok());
        for dtype in [
            WeightDtype::FP8E4M3,
            WeightDtype::UInt8,
            WeightDtype::FP32,
            WeightDtype::FP8E8M0,
        ] {
            assert!(check(stored(dtype)).is_err(), "{dtype:?}");
        }
        let dequantized = StoredHead {
            loaded_ptr: 0x2000,
            ..stored(WeightDtype::BF16)
        };
        assert!(check(dequantized).is_err());
        let short = StoredHead {
            shape: &[248_000, 2_560],
            ..stored(WeightDtype::BF16)
        };
        assert!(check(short).is_err());
        let misaligned = StoredHead {
            stored_ptr: 0x1008,
            loaded_ptr: 0x1008,
            ..stored(WeightDtype::BF16)
        };
        assert!(check(misaligned).is_err());
    }

    fn qwen4() -> LmHeadInputs {
        LmHeadInputs {
            is_qwen4: true,
            ..LmHeadInputs::default()
        }
    }

    #[test]
    fn default_defers_to_skip_lm_head_quantization() {
        for skip_quant in [false, true] {
            for is_qwen4 in [false, true] {
                let inputs = LmHeadInputs {
                    is_qwen4,
                    skip_quant,
                    use_native_mtp: true,
                    native_batch: true,
                    ..LmHeadInputs::default()
                };
                let want = if skip_quant {
                    LmHeadPrecision::Bf16
                } else {
                    LmHeadPrecision::Nvfp4
                };
                assert_eq!(decide_lm_head_precision(false, inputs).unwrap(), want);
            }
        }
    }

    #[test]
    fn selector_parse_is_exact() {
        assert!(!parse_selector(None).unwrap());
        assert!(!parse_selector(Some("0")).unwrap());
        assert!(parse_selector(Some("1")).unwrap());
        for bad in ["", "true", "01", " 1"] {
            assert!(parse_selector(Some(bad)).is_err());
        }
    }

    #[test]
    fn plain_qwen4_selects_bf16() {
        assert_eq!(
            decide_lm_head_precision(true, qwen4()).unwrap(),
            LmHeadPrecision::Bf16
        );
    }

    #[test]
    fn selector_refuses_other_models_and_nvfp4_only_consumers() {
        assert!(decide_lm_head_precision(true, LmHeadInputs::default()).is_err());
        let blocked = [
            LmHeadInputs {
                use_native_mtp: true,
                ..qwen4()
            },
            LmHeadInputs {
                has_dflash: true,
                ..qwen4()
            },
            LmHeadInputs {
                self_speculative: true,
                ..qwen4()
            },
            LmHeadInputs {
                native_batch: true,
                ..qwen4()
            },
            LmHeadInputs {
                lm_head_tc: true,
                ..qwen4()
            },
            LmHeadInputs {
                lm_head_t: true,
                ..qwen4()
            },
            LmHeadInputs {
                decode_tc_parity: true,
                ..qwen4()
            },
            LmHeadInputs {
                fp32_logits: true,
                ..qwen4()
            },
            LmHeadInputs {
                verify_vocab_cap: true,
                ..qwen4()
            },
        ];
        for inputs in blocked {
            assert!(
                decide_lm_head_precision(true, inputs).is_err(),
                "{inputs:?}"
            );
        }
    }
}
