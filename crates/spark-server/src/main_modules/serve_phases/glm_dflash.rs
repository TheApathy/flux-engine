// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 speculative-decoding admission at serve time.
//!
//! GLM speculates only through its own DFlash2 drafter and policy verifier on
//! the EXL3 checkpoint. The proposal block is pinned by the checkpoint (8);
//! the verify width is bounded separately by `--glm-dflash-max-drafts`.

use anyhow::{Result, bail, ensure};
use spark_model::weight_loader::DsparkVerifyMode;

/// DFlash2 block size the GLM-5.3 drafter checkpoint was trained with.
const GLM53_DFLASH2_BLOCK: usize = 8;

/// Serve flags whose generic DFlash meaning is not implemented by GLM's
/// checkpoint-specific DFlash2 runtime.  Keep these in admission instead of
/// accepting a flag and silently running a different engine.
#[derive(Debug, Clone, Copy)]
pub(in crate::main_modules) struct GlmDflashRuntimeOptions<'a> {
    pub max_seq_len: usize,
    pub dspark_verify_mode: DsparkVerifyMode,
    pub quantization: &'a str,
    pub window_size: usize,
    pub method: &'a str,
    pub ddtree_budget: Option<usize>,
    pub has_donor_model: bool,
    pub mtp_gate: Option<&'a str>,
}

fn validate_runtime_options(options: GlmDflashRuntimeOptions<'_>) -> Result<()> {
    ensure!(
        options.max_seq_len <= spark_model::model::glm53::GLM53_DFLASH2_MAX_SEQUENCE_TOKENS,
        "GLM-5.3 DFlash2 currently retains at most {} target tokens and serves at most {} total \
         tokens including the terminal sampled output, but --max-seq-len is {}; pass \
         --max-seq-len {} or omit --dflash for target-only long-context serving",
        spark_model::model::glm53::GLM53_DFLASH2_CONTEXT_CAPACITY,
        spark_model::model::glm53::GLM53_DFLASH2_MAX_SEQUENCE_TOKENS,
        options.max_seq_len,
        spark_model::model::glm53::GLM53_DFLASH2_MAX_SEQUENCE_TOKENS,
    );
    ensure!(
        options.dspark_verify_mode == DsparkVerifyMode::Static,
        "--dspark-verify-mode {} is not implemented by GLM-5.3 DFlash2; use static",
        options.dspark_verify_mode,
    );
    ensure!(
        options.quantization == "bf16",
        "GLM-5.3 DFlash2 currently loads BF16 drafter tensors only; \
         --dflash-quantization {} would be ignored",
        options.quantization,
    );
    // 4096 is the inherited generic CLI default.  GLM maps that default to
    // its checkpoint-pinned 2048-token window; an explicit 2048 is accepted
    // too.  Every other value would otherwise be silently ignored.
    ensure!(
        matches!(options.window_size, 2048 | 4096),
        "GLM-5.3 DFlash2 has a checkpoint-pinned 2048-token window; \
         --dflash-window-size {} is unsupported",
        options.window_size,
    );
    ensure!(
        options.method == "flat",
        "GLM-5.3 DFlash2 supports only --dflash-method flat; got {}",
        options.method,
    );
    ensure!(
        options.ddtree_budget.is_none(),
        "--ddtree-budget does not apply to GLM-5.3 DFlash2"
    );
    ensure!(
        !options.has_donor_model,
        "--dflash-donor-model does not apply to GLM-5.3 DFlash2"
    );
    match options.mtp_gate {
        None | Some("auto" | "force" | "dflash") => {}
        Some("mtp") => bail!(
            "--mtp-gate mtp requires a wired native MTP arm; GLM-5.3 EXL3 currently installs only DFlash2"
        ),
        Some(other) => {
            bail!("--mtp-gate {other}: unknown value (expected auto, force, or dflash for GLM-5.3)")
        }
    }
    Ok(())
}

/// Resolve GLM speculation for this serve. `Ok(Some(drafts))` is the verify
/// width for a GLM DFlash2 run, `Ok(None)` leaves every other model (and
/// target-only GLM) to its existing path. Anything else GLM cannot serve is
/// refused rather than handed to a generic proposer.
pub(in crate::main_modules) fn resolve_glm_dflash(
    model_type: &str,
    is_glm53_exl3: bool,
    dflash: bool,
    speculative: bool,
    gamma: Option<usize>,
    max_drafts: Option<u8>,
    runtime: GlmDflashRuntimeOptions<'_>,
) -> Result<Option<usize>> {
    if model_type != "glm5_next" {
        ensure!(
            max_drafts.is_none(),
            "--glm-dflash-max-drafts applies only to GLM-5.3"
        );
        return Ok(None);
    }
    if speculative {
        bail!(
            "GLM-5.3 EXL3 native NextN layer 45 is not wired or admitted by this runtime; \
             speculate with --dflash and its DFlash2 drafter"
        );
    }
    if !dflash {
        ensure!(
            max_drafts.is_none(),
            "--glm-dflash-max-drafts requires --dflash"
        );
        return Ok(None);
    }
    ensure!(
        is_glm53_exl3,
        "GLM-5.3 DFlash2 requires the EXL3 checkpoint; serve the GGUF target-only"
    );
    validate_runtime_options(runtime)?;
    ensure!(
        gamma.unwrap_or(GLM53_DFLASH2_BLOCK) == GLM53_DFLASH2_BLOCK,
        "GLM-5.3 DFlash2 has a checkpoint-pinned block of {GLM53_DFLASH2_BLOCK}; \
         pass --dflash-gamma {GLM53_DFLASH2_BLOCK} (bound the verify width with \
         --glm-dflash-max-drafts)"
    );
    let drafts = max_drafts.map_or(GLM53_DFLASH2_BLOCK - 1, usize::from);
    ensure!(
        (1..GLM53_DFLASH2_BLOCK).contains(&drafts),
        "--glm-dflash-max-drafts must be in 1..=7"
    );
    Ok(Some(drafts))
}

/// Resolve the DFlash drafter checkpoint directory without loading it (the
/// GLM-5.3 model installs its own DFlash2 drafter from this directory).
pub(crate) fn resolve_dflash_drafter_dir(
    args: &crate::cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<std::path::PathBuf> {
    use anyhow::Context as _;
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
        .context("Failed to resolve DFlash drafter checkpoint")
}

#[cfg(test)]
mod tests {
    use spark_model::weight_loader::DsparkVerifyMode;

    use super::{GlmDflashRuntimeOptions, resolve_glm_dflash as resolve};

    fn runtime() -> GlmDflashRuntimeOptions<'static> {
        GlmDflashRuntimeOptions {
            max_seq_len: 2048,
            dspark_verify_mode: DsparkVerifyMode::Static,
            quantization: "bf16",
            window_size: 4096,
            method: "flat",
            ddtree_budget: None,
            has_donor_model: false,
            mtp_gate: None,
        }
    }

    #[test]
    fn other_models_are_untouched() {
        assert_eq!(
            resolve("qwen3_next", false, true, false, Some(15), None, runtime()).unwrap(),
            None
        );
        assert_eq!(
            resolve("qwen3_next", false, false, true, None, None, runtime()).unwrap(),
            None
        );
        assert!(resolve("qwen3_next", false, true, false, None, Some(3), runtime()).is_err());
    }

    #[test]
    fn target_only_glm_is_untouched() {
        assert_eq!(
            resolve("glm5_next", true, false, false, None, None, runtime()).unwrap(),
            None
        );
        assert!(resolve("glm5_next", true, false, false, None, Some(3), runtime()).is_err());
    }

    #[test]
    fn glm_dflash_resolves_the_verify_width() {
        assert_eq!(
            resolve("glm5_next", true, true, false, Some(8), Some(3), runtime()).unwrap(),
            Some(3)
        );
        assert_eq!(
            resolve("glm5_next", true, true, false, None, Some(2), runtime()).unwrap(),
            Some(2)
        );
        assert_eq!(
            resolve("glm5_next", true, true, false, Some(8), None, runtime()).unwrap(),
            Some(7)
        );
    }

    #[test]
    fn glm_refuses_what_it_cannot_serve() {
        assert!(resolve("glm5_next", true, false, true, None, None, runtime()).is_err());
        assert!(resolve("glm5_next", false, true, false, Some(8), Some(3), runtime()).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(15), None, runtime()).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(8), Some(0), runtime()).is_err());
        assert!(resolve("glm5_next", true, true, false, Some(8), Some(8), runtime()).is_err());
    }

    #[test]
    fn glm_rejects_generic_dflash_flags_it_does_not_execute() {
        let rejects = [
            GlmDflashRuntimeOptions {
                max_seq_len: 2049,
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                dspark_verify_mode: DsparkVerifyMode::Compact,
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                quantization: "nvfp4",
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                window_size: 0,
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                method: "ddtree",
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                ddtree_budget: Some(3),
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                has_donor_model: true,
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                mtp_gate: Some("mtp"),
                ..runtime()
            },
            GlmDflashRuntimeOptions {
                mtp_gate: Some("always"),
                ..runtime()
            },
        ];
        for options in rejects {
            assert!(resolve("glm5_next", true, true, false, Some(8), Some(3), options,).is_err());
        }

        assert!(
            resolve(
                "glm5_next",
                true,
                true,
                false,
                Some(8),
                Some(3),
                GlmDflashRuntimeOptions {
                    window_size: 2048,
                    ..runtime()
                },
            )
            .is_ok()
        );
    }
}
