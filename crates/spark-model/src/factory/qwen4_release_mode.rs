// SPDX-License-Identifier: AGPL-3.0-only

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen4ReleaseMode {
    None,
    Plain,
    Dflash,
    NativeMtp,
}

/// Safety inputs gating target BF16 `lm_head.weight` release under
/// [`Qwen4ReleaseMode::Dflash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen4DflashLmHeadReleaseInput {
    pub(crate) has_nvfp4_lm_head: bool,
    pub(crate) target_bf16_lm_head_enabled: bool,
    pub(crate) lm_head_is_untied: bool,
    pub(crate) has_donor_store: bool,
    pub(crate) dflash_lm_head_fp8_enabled: bool,
    pub(crate) drafter_has_dspine: bool,
    pub(crate) drafter_hidden_matches_target: bool,
    pub(crate) has_exact_lm_head_kernels: bool,
    pub(crate) dflash_debug_dump_enabled: bool,
}

/// Whether the target BF16 `lm_head.weight` (`~1.18 GiB`) can be released
/// losslessly before Step-5 KV pool sizing under [`Qwen4ReleaseMode::Dflash`].
///
/// The target itself only reads `lm_head_nvfp4` / `lm_head_fp8` once
/// `has_nvfp4_lm_head && !target_bf16_lm_head_enabled`. On the DFlash drafter
/// side, `lm_head_shared` (the target BF16 pointer) is never read when either:
/// - a separate `--dflash-donor-model` supplies the drafter's `lm_head`, or
/// - DFlash shares the target's already-resident untransposed NVFP4 `lm_head`
///   (`build.rs` wires `lm_head_shared_nvfp4_exact` whenever
///   `hidden_size == target_hidden_size`), `ATLAS_DFLASH_LM_HEAD_FP8 != 1`,
///   no DSpine code bridge is present, exact multi-row NVFP4 LM-head kernels
///   are compiled, and `ATLAS_DFLASH_DEBUG_DUMP != 1`.
#[must_use]
pub(crate) const fn can_release_qwen4_dflash_bf16_lm_head(
    input: Qwen4DflashLmHeadReleaseInput,
) -> bool {
    if !input.has_nvfp4_lm_head || input.target_bf16_lm_head_enabled || !input.lm_head_is_untied {
        return false;
    }
    if input.has_donor_store {
        return true;
    }
    !input.dflash_lm_head_fp8_enabled
        && !input.drafter_has_dspine
        && input.drafter_hidden_matches_target
        && input.has_exact_lm_head_kernels
        && !input.dflash_debug_dump_enabled
}

#[must_use]
pub(crate) fn select_qwen4_release_mode(
    is_qwen4: bool,
    use_speculative: bool,
    has_dflash: bool,
    has_native_mtp: bool,
    release_plain: bool,
    release_spec: bool,
) -> Qwen4ReleaseMode {
    if !is_qwen4 {
        return Qwen4ReleaseMode::None;
    }
    if use_speculative {
        return if !has_dflash && has_native_mtp && release_spec {
            Qwen4ReleaseMode::NativeMtp
        } else {
            Qwen4ReleaseMode::None
        };
    }
    if !has_dflash && release_plain {
        Qwen4ReleaseMode::Plain
    } else if has_dflash && release_plain {
        Qwen4ReleaseMode::Dflash
    } else {
        Qwen4ReleaseMode::None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen4DflashLmHeadReleaseInput, Qwen4ReleaseMode, can_release_qwen4_dflash_bf16_lm_head,
        select_qwen4_release_mode,
    };

    #[test]
    fn defaults_off_and_rejects_other_models() {
        assert_eq!(
            select_qwen4_release_mode(true, false, false, false, false, false),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, false, true, false, false, false),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(false, false, false, false, true, true),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(false, false, true, false, true, true),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(false, true, false, true, true, true),
            Qwen4ReleaseMode::None
        );
    }

    #[test]
    fn plain_and_dflash_modes_partition_non_mtp_qwen4_launches() {
        assert_eq!(
            select_qwen4_release_mode(true, false, false, false, true, false),
            Qwen4ReleaseMode::Plain
        );
        assert_eq!(
            select_qwen4_release_mode(true, false, true, false, true, false),
            Qwen4ReleaseMode::Dflash
        );
        assert_eq!(
            select_qwen4_release_mode(true, false, true, false, false, true),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, false, false, true, false),
            Qwen4ReleaseMode::None
        );
        // Two-arm `--speculative --dflash` keeps native MTP weights alive.
        assert_eq!(
            select_qwen4_release_mode(true, true, true, true, true, true),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, true, false, true, true),
            Qwen4ReleaseMode::None
        );
    }

    #[test]
    fn native_mtp_release_requires_live_native_proposer() {
        assert_eq!(
            select_qwen4_release_mode(true, true, false, true, false, true),
            Qwen4ReleaseMode::NativeMtp
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, false, false, false, true),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, false, true, false, false),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, true, true, false, true),
            Qwen4ReleaseMode::None
        );
    }

    #[test]
    fn spec_release_cannot_fall_through_to_plain_or_dflash_release() {
        assert_eq!(
            select_qwen4_release_mode(true, true, false, true, true, true),
            Qwen4ReleaseMode::NativeMtp
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, false, false, true, false),
            Qwen4ReleaseMode::None
        );
        assert_eq!(
            select_qwen4_release_mode(true, true, true, false, true, false),
            Qwen4ReleaseMode::None
        );
    }

    #[test]
    fn dflash_bf16_lm_head_release_enforces_every_safety_invariant() {
        let safe_native = Qwen4DflashLmHeadReleaseInput {
            has_nvfp4_lm_head: true,
            target_bf16_lm_head_enabled: false,
            lm_head_is_untied: true,
            has_donor_store: false,
            dflash_lm_head_fp8_enabled: false,
            drafter_has_dspine: false,
            drafter_hidden_matches_target: true,
            has_exact_lm_head_kernels: true,
            dflash_debug_dump_enabled: false,
        };
        assert!(can_release_qwen4_dflash_bf16_lm_head(safe_native));

        // Every single violated safety invariant must keep BF16 lm_head resident.
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                has_nvfp4_lm_head: false,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                target_bf16_lm_head_enabled: true,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                lm_head_is_untied: false,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                dflash_lm_head_fp8_enabled: true,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                drafter_has_dspine: true,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                drafter_hidden_matches_target: false,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                has_exact_lm_head_kernels: false,
                ..safe_native
            }
        ));
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                dflash_debug_dump_enabled: true,
                ..safe_native
            }
        ));

        // Donor bridge supplies its own lm_head; target BF16 head is unused even
        // when drafter hidden/dspine/FP8 differ from the target.
        assert!(can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                has_donor_store: true,
                dflash_lm_head_fp8_enabled: true,
                drafter_has_dspine: true,
                drafter_hidden_matches_target: false,
                has_exact_lm_head_kernels: false,
                ..safe_native
            }
        ));
        // Donor bridge still preserves target BF16 control and tied embeddings.
        assert!(!can_release_qwen4_dflash_bf16_lm_head(
            Qwen4DflashLmHeadReleaseInput {
                has_donor_store: true,
                target_bf16_lm_head_enabled: true,
                ..safe_native
            }
        ));
    }
}
