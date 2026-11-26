// SPDX-License-Identifier: AGPL-3.0-only

//! Which families `ATLAS_QWEN4_PREFILL_NATIVE=1` can engage under the rest of
//! the prefill environment. A family is planned only where its prefill route
//! reaches a native hook; registering a weight whose route never consults the
//! side table would spend bytes on dead copies.

use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;

use super::SELECTOR;
use crate::layers::qwen4_fast_proj;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Family {
    Ssm,
    Attn,
    Hc,
    Router,
    Shared,
}

impl Family {
    pub(crate) const ALL: [Family; 5] =
        [Self::Ssm, Self::Attn, Self::Hc, Self::Router, Self::Shared];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Ssm => "ssm",
            Self::Attn => "attn",
            Self::Hc => "hc",
            Self::Router => "router",
            Self::Shared => "shared",
        }
    }
}

/// Engaged families plus, for each one that is not, the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub ssm: Result<(), &'static str>,
    pub attn: Result<(), &'static str>,
    pub hc: Result<(), &'static str>,
    pub router: Result<(), &'static str>,
    /// Exact-MXFP8 BF16 shared expert (`shared.rs`); no environment input bypasses it.
    pub shared: Result<(), &'static str>,
}

impl Plan {
    pub(crate) fn covers(&self, family: Family) -> bool {
        self.state(family).is_ok()
    }

    fn state(&self, family: Family) -> Result<(), &'static str> {
        match family {
            Family::Ssm => self.ssm,
            Family::Attn => self.attn,
            Family::Hc => self.hc,
            Family::Router => self.router,
            Family::Shared => self.shared,
        }
    }

    /// `ssm:fp8_mxfp8,attn:fp8_mxfp8,hc:off(fused_prep),router:bf16`.
    pub(crate) fn summary(&self) -> String {
        Family::ALL
            .iter()
            .map(|&f| match (self.state(f), f) {
                (Ok(()), Family::Ssm | Family::Attn) => format!("{}:fp8_mxfp8", f.name()),
                (Ok(()), Family::Shared) => "shared:bf16_mxfp8".to_string(),
                (Ok(()), _) => format!("{}:bf16", f.name()),
                (Err(why), _) => format!("{}:off({why})", f.name()),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Every environment input the decision reads, captured so the rule is pure.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Inputs {
    pub is_qwen4: bool,
    pub multi_rank: bool,
    /// `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8` families.
    pub fp8: qwen4_fast_proj::Selection,
    /// `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor`.
    pub tensor_scales: bool,
    /// `ATLAS_QWEN4_PREFILL_FAST_PROJ` families (BF16 dequant route).
    pub fast: qwen4_fast_proj::Selection,
    /// `ATLAS_QWEN4_PREFILL_HC_FUSED` is 1 or 2.
    pub hc_fused: bool,
    /// `ATLAS_QWEN4_PREFILL_HC_FUSED_PREP=1`: the fused prep reads packed NVFP4.
    pub hc_fused_prep: bool,
    /// `ATLAS_QWEN4_PREFILL_MHC_ROOF=t128_w4|w4`: the roof up-mix reads packed NVFP4.
    pub mhc_roof_w4: bool,
    /// `ATLAS_QWEN4_ROUTER_BF16=1` already removed the NVFP4 router.
    pub router_bf16: bool,
}

pub(crate) fn decide(i: Inputs) -> Result<Plan> {
    ensure!(
        i.is_qwen4,
        "{SELECTOR}=1 is qualified only for qwen4_exp (Flash-Next)"
    );
    ensure!(
        !i.multi_rank,
        "{SELECTOR}=1 is not qualified for TP/EP > 1 (sharding frees the keyed NVFP4 source)"
    );
    if (i.fp8.ssm || i.fp8.attn) && !i.tensor_scales {
        bail!(
            "{SELECTOR}=1 needs ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor (the native E4M3 rows carry one tensor scale)"
        );
    }
    let fp8 = |on: bool| {
        if on {
            Ok(())
        } else {
            Err("fast_proj_fp8_not_selected")
        }
    };
    let hc = if !(i.fast.hc || i.hc_fused) {
        Err("hc_fast_or_fused_not_selected")
    } else if i.fp8.hc {
        // FAST_PROJ_FP8 hc routes HC through the FP8 GEMM on NVFP4 rows; the
        // BF16 table is consulted only by the BF16 dequant helpers.
        Err("hc_fast_proj_fp8_reads_nvfp4")
    } else if i.hc_fused_prep {
        Err("hc_fused_prep_reads_nvfp4")
    } else if i.mhc_roof_w4 {
        Err("mhc_roof_w4_reads_nvfp4")
    } else {
        Ok(())
    };
    let plan = Plan {
        ssm: fp8(i.fp8.ssm),
        attn: fp8(i.fp8.attn),
        hc,
        router: if i.router_bf16 {
            Err("router_bf16_already_dense")
        } else {
            Ok(())
        },
        shared: Ok(()),
    };
    ensure!(
        Family::ALL.iter().any(|&f| plan.covers(f)),
        "{SELECTOR}=1 engages nothing under this environment: {}",
        plan.summary()
    );
    Ok(plan)
}

fn env_is(name: &str, values: &[&str]) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| values.contains(&v.trim()))
}

fn inputs(config: &ModelConfig) -> Result<Inputs> {
    Ok(Inputs {
        is_qwen4: config.is_qwen4_exp(),
        multi_rank: config.tp_world_size > 1 || config.ep_world_size > 1,
        fp8: qwen4_fast_proj::fp8_selection()?,
        tensor_scales: qwen4_fast_proj::fp8_tensor_scales(),
        fast: qwen4_fast_proj::selection()?,
        hc_fused: env_is("ATLAS_QWEN4_PREFILL_HC_FUSED", &["1", "2"]),
        hc_fused_prep: env_is("ATLAS_QWEN4_PREFILL_HC_FUSED_PREP", &["1"]),
        mhc_roof_w4: env_is("ATLAS_QWEN4_PREFILL_MHC_ROOF", &["t128_w4", "w4"]),
        router_bf16: crate::layers::moe::qwen4_router_bf16::selected()?,
    })
}

/// Decided once per process from the first target config that asks.
pub(crate) fn process_plan(config: &ModelConfig) -> Result<Plan> {
    static PLAN: OnceLock<Result<Plan, String>> = OnceLock::new();
    PLAN.get_or_init(|| {
        inputs(config)
            .and_then(decide)
            .map_err(|e| format!("{e:#}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen4_fast_proj::Selection;

    fn production() -> Inputs {
        Inputs {
            is_qwen4: true,
            fp8: Selection {
                ssm: true,
                hc: false,
                attn: true,
            },
            tensor_scales: true,
            fast: Selection {
                ssm: true,
                hc: true,
                attn: true,
            },
            hc_fused: true,
            ..Inputs::default()
        }
    }

    #[test]
    fn production_env_engages_every_family() {
        let plan = decide(production()).unwrap();
        assert!(Family::ALL.iter().all(|&f| plan.covers(f)));
        assert_eq!(
            plan.summary(),
            "ssm:fp8_mxfp8,attn:fp8_mxfp8,hc:bf16,router:bf16,shared:bf16_mxfp8"
        );
    }

    #[test]
    fn families_drop_out_with_their_route() {
        let plan = decide(Inputs {
            fp8: Selection {
                ssm: false,
                hc: false,
                attn: true,
            },
            ..production()
        })
        .unwrap();
        assert_eq!(plan.ssm, Err("fast_proj_fp8_not_selected"));
        let fp8_hc = Selection {
            ssm: true,
            hc: true,
            attn: true,
        };
        let plan = decide(Inputs {
            fp8: fp8_hc,
            ..production()
        })
        .unwrap();
        assert_eq!(plan.hc, Err("hc_fast_proj_fp8_reads_nvfp4"));
        let plan = decide(Inputs {
            hc_fused_prep: true,
            ..production()
        })
        .unwrap();
        assert_eq!(plan.hc, Err("hc_fused_prep_reads_nvfp4"));
        let plan = decide(Inputs {
            mhc_roof_w4: true,
            ..production()
        })
        .unwrap();
        assert_eq!(plan.hc, Err("mhc_roof_w4_reads_nvfp4"));
        let plan = decide(Inputs {
            fast: Selection::default(),
            hc_fused: false,
            ..production()
        })
        .unwrap();
        assert_eq!(plan.hc, Err("hc_fast_or_fused_not_selected"));
        let plan = decide(Inputs {
            router_bf16: true,
            ..production()
        })
        .unwrap();
        assert_eq!(plan.router, Err("router_bf16_already_dense"));
    }

    #[test]
    fn unqualified_launches_fail_closed() {
        assert!(
            decide(Inputs {
                is_qwen4: false,
                ..production()
            })
            .is_err()
        );
        assert!(
            decide(Inputs {
                multi_rank: true,
                ..production()
            })
            .is_err()
        );
        assert!(
            decide(Inputs {
                tensor_scales: false,
                ..production()
            })
            .is_err()
        );
        // With family `shared` always covered, a launch whose other routes are
        // all off still engages shared alone instead of failing.
        let nothing = Inputs {
            is_qwen4: true,
            router_bf16: true,
            ..Inputs::default()
        };
        let plan = decide(nothing).unwrap();
        assert!(
            Family::ALL
                .iter()
                .all(|&f| plan.covers(f) == (f == Family::Shared))
        );
    }
}
