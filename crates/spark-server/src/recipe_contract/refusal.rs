// SPDX-License-Identifier: AGPL-3.0-only

//! Typed refusal reasons. Every refusal renders as `key=value` so the startup
//! line `RECIPE_REFUSED a=b c=d` is grep-able and stable.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The recipe file itself is malformed or self-contradictory.
    Invalid(String),
    /// A NUMERICS env var is set that the recipe does not declare.
    UnknownNumericsEnv(String),
    /// A declared NUMERICS/ROUTE var is set to a different value.
    EnvMismatch {
        name: String,
        want: String,
        got: String,
    },
    /// A declared NUMERICS/ROUTE var is not set, so the engine would use its
    /// code default rather than the recipe's value.
    EnvMissing(String),
    /// A ROUTE var is set that the recipe does not declare, under `route_policy = "refuse"`.
    UnknownRouteEnv(String),
    /// An `ATLAS_*`/`FLUX_*` name or value is not UTF-8, so it cannot be checked.
    NonUtf8Env(String),
    /// A recipe-governed value changed after the startup check (MODEL.toml
    /// defaults, prefill budget resolution).
    ResolvedMismatch {
        field: &'static str,
        want: String,
        got: String,
    },
    /// Model swaps are refused while a recipe is active.
    SwapUnderRecipe,
    /// An operational env var is outside its declared range.
    OperationalOutOfRange { name: String, got: String },
    /// The env-derived served format of a module family differs from the plan.
    PrecisionMismatch {
        family: &'static str,
        want: String,
        got: String,
    },
    /// The plan needs a selector this build has no reader for.
    PrecisionUnsupported {
        family: &'static str,
        selector: &'static str,
    },
    /// A serve flag is outside the recipe's declared value or range.
    FlagMismatch {
        flag: &'static str,
        want: String,
        got: String,
    },
    /// A pinned checkpoint file is missing or its sha256 differs.
    CheckpointPin { file: String, detail: String },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid_recipe={why:?}"),
            Self::UnknownNumericsEnv(n) => write!(f, "unknown_numerics_env={n}"),
            Self::EnvMismatch { name, want, got } => {
                write!(f, "env_mismatch={name} want={want:?} got={got:?}")
            }
            Self::EnvMissing(n) => write!(f, "declared_env_missing={n}"),
            Self::UnknownRouteEnv(n) => write!(f, "unknown_route_env={n}"),
            Self::NonUtf8Env(n) => write!(f, "non_utf8_env={n}"),
            Self::ResolvedMismatch { field, want, got } => {
                write!(f, "resolved_mismatch={field} want={want} got={got}")
            }
            Self::SwapUnderRecipe => write!(f, "model_swap_under_recipe=refused"),
            Self::OperationalOutOfRange { name, got } => {
                write!(f, "operational_env_out_of_range={name} got={got:?}")
            }
            Self::PrecisionMismatch { family, want, got } => {
                write!(f, "precision_mismatch={family} want={want} got={got}")
            }
            Self::PrecisionUnsupported { family, selector } => {
                write!(
                    f,
                    "precision_unsupported_by_build={family} selector={selector}"
                )
            }
            Self::FlagMismatch { flag, want, got } => {
                write!(f, "flag_mismatch={flag} want={want} got={got}")
            }
            Self::CheckpointPin { file, detail } => {
                write!(f, "checkpoint_pin={file} detail={detail:?}")
            }
        }
    }
}

/// The single startup line for one or more refusals.
pub fn render(refusals: &[Refusal]) -> String {
    let parts: Vec<String> = refusals.iter().map(ToString::to_string).collect();
    format!("RECIPE_REFUSED {}", parts.join(" ; "))
}
