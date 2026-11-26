// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-weight-load speculation admission for DeepSeek-V4.1.

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Admission {
    OtherModel,
    Plain { generic_spec_flag_ignored: bool },
    InternalGreedy { generic_spec_flag_redundant: bool },
}

pub(super) fn admit(
    model_type: &str,
    external_dflash: bool,
    generic_speculative: bool,
    dspark_env: Option<&str>,
) -> Result<Admission> {
    if model_type != "deepseek_v41" {
        return Ok(Admission::OtherModel);
    }
    let internal_dspark = match dspark_env {
        None | Some("0") => false,
        Some("1") => true,
        Some(value) => bail!("ATLAS_DSV41_DSPARK={value}: expected exactly 0 or 1"),
    };
    if external_dflash {
        if internal_dspark {
            bail!(
                "DeepSeek-V4.1 external --dflash and internal ATLAS_DSV41_DSPARK=1 are incompatible; choose the internal DSpark path or remove DSpark"
            );
        }
        bail!(
            "DeepSeek-V4.1 external --dflash is not installed by its dedicated model factory; refusing before weight load instead of claiming an inactive drafter"
        );
    }
    if internal_dspark {
        Ok(Admission::InternalGreedy {
            generic_spec_flag_redundant: generic_speculative,
        })
    } else {
        Ok(Admission::Plain {
            generic_spec_flag_ignored: generic_speculative,
        })
    }
}

pub(super) fn log(admission: Admission) {
    match admission {
        Admission::OtherModel => {}
        Admission::Plain {
            generic_spec_flag_ignored: true,
        } => tracing::warn!(
            "DeepSeek-V4.1 speculation admission: PLAIN; --speculative does not load internal DSpark (set ATLAS_DSV41_DSPARK=1)"
        ),
        Admission::Plain {
            generic_spec_flag_ignored: false,
        } => tracing::info!("DeepSeek-V4.1 speculation admission: PLAIN target-only"),
        Admission::InternalGreedy {
            generic_spec_flag_redundant,
        } => {
            tracing::info!(
                "DeepSeek-V4.1 speculation admission: internal DSpark GREEDY admitted; sampled requests HOLD on ordinary decode; external DFlash OFF"
            );
            if generic_spec_flag_redundant {
                tracing::warn!(
                    "DeepSeek-V4.1 --speculative is redundant: internal DSpark is controlled by ATLAS_DSV41_DSPARK=1"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_deepseek_admission_matrix_is_fail_closed() {
        assert_eq!(
            admit("qwen4_exp", true, false, Some("1")).unwrap(),
            Admission::OtherModel
        );
        assert_eq!(
            admit("deepseek_v41", false, false, None).unwrap(),
            Admission::Plain {
                generic_spec_flag_ignored: false
            }
        );
        assert_eq!(
            admit("deepseek_v41", false, true, Some("0")).unwrap(),
            Admission::Plain {
                generic_spec_flag_ignored: true
            }
        );
        assert_eq!(
            admit("deepseek_v41", false, false, Some("1")).unwrap(),
            Admission::InternalGreedy {
                generic_spec_flag_redundant: false
            }
        );
        assert_eq!(
            admit("deepseek_v41", false, true, Some("1")).unwrap(),
            Admission::InternalGreedy {
                generic_spec_flag_redundant: true
            }
        );
        assert!(
            admit("deepseek_v41", true, false, None)
                .unwrap_err()
                .to_string()
                .contains("not installed")
        );
        assert!(
            admit("deepseek_v41", true, false, Some("1"))
                .unwrap_err()
                .to_string()
                .contains("incompatible")
        );
        assert!(
            admit("deepseek_v41", false, false, Some("true"))
                .unwrap_err()
                .to_string()
                .contains("exactly 0 or 1")
        );
    }
}
