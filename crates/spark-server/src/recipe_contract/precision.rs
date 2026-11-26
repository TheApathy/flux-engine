// SPDX-License-Identifier: AGPL-3.0-only

//! The precision plan as fields, and the env selectors that realise it today.
//!
//! The selectors and the umbrella rule match `feat/fnd-umbrella`
//! (`spark_model::qwen4_native_selectors`): `ATLAS_QWEN4_DENSE_NATIVE=1`
//! implies every knob unless that knob is explicitly `0`. A selector with no
//! reader in this build is treated as unset, because the engine ignores it.

use super::catalog::Catalog;
use super::refusal::Refusal;
use super::schema::{DenseFormat, PrecisionPlan, PrefillFormat};
use std::collections::BTreeMap;

pub const DENSE_NATIVE: &str = "ATLAS_QWEN4_DENSE_NATIVE";
pub const DENSE_FP8: &str = "ATLAS_QWEN4_DENSE_FP8";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Knob {
    Router,
    Hc,
    Gdn,
    Attn,
    Shared,
    LmHead,
    Prefill,
}

impl Knob {
    const fn env(self) -> &'static str {
        match self {
            Self::Router => "ATLAS_QWEN4_ROUTER_BF16",
            Self::Hc => "ATLAS_QWEN4_HC_BF16",
            Self::Gdn => "ATLAS_QWEN4_GDN_BF16",
            Self::Attn => "ATLAS_QWEN4_ATTN_BF16",
            Self::Shared => "ATLAS_QWEN4_SHARED_BF16",
            Self::LmHead => "ATLAS_QWEN4_LMHEAD_BF16",
            Self::Prefill => "ATLAS_QWEN4_PREFILL_NATIVE",
        }
    }
}

const KNOBS: [Knob; 7] = [
    Knob::Router,
    Knob::Hc,
    Knob::Gdn,
    Knob::Attn,
    Knob::Shared,
    Knob::LmHead,
    Knob::Prefill,
];

/// Every env name the precision plan owns. These may not appear in
/// `numerics_env`; the plan is their only declaration.
pub fn owned_selectors() -> impl Iterator<Item = &'static str> {
    KNOBS
        .iter()
        .map(|k| k.env())
        .chain([DENSE_NATIVE, DENSE_FP8])
}

pub fn is_owned(name: &str) -> bool {
    owned_selectors().any(|n| n == name)
}

/// One family's resolved format, as the receipt prints it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyFormat {
    pub family: &'static str,
    pub format: String,
}

/// Schema-level consistency that needs no environment.
pub fn validate_plan(plan: &PrecisionPlan) -> Result<(), Refusal> {
    let fp8 = [plan.gdn_proj, plan.attention_proj, plan.shared_expert]
        .iter()
        .filter(|f| **f == DenseFormat::Fp8E4m3)
        .count();
    if fp8 != 0 && fp8 != 3 {
        return Err(Refusal::Invalid(
            "fp8_e4m3 is one selector (DENSE_FP8): gdn_proj, attention_proj and shared_expert must all use it or none".into(),
        ));
    }
    for (family, f) in [
        ("router", plan.router),
        ("hyper_connection", plan.hyper_connection),
        ("lm_head", plan.lm_head),
    ] {
        if f == DenseFormat::Fp8E4m3 {
            return Err(Refusal::Invalid(format!("{family} has no fp8_e4m3 path")));
        }
    }
    Ok(())
}

struct Selectors<'a> {
    env: &'a BTreeMap<String, String>,
    catalog: &'a Catalog,
}

impl Selectors<'_> {
    /// `Some(value)` only when the build reads the selector.
    fn read(&self, name: &str) -> Option<&str> {
        self.catalog
            .has_reader(name)
            .then(|| self.env.get(name).map(String::as_str))
            .flatten()
    }

    /// Strict switch: unset -> None, "0"/"1" -> Some, anything else refuses.
    fn flag(&self, name: &str) -> Result<Option<bool>, String> {
        match self.read(name) {
            None => Ok(None),
            Some("0") => Ok(Some(false)),
            Some("1") => Ok(Some(true)),
            Some(other) => Err(format!("{name}={other:?} (must be 0 or 1)")),
        }
    }

    fn knob(&self, knob: Knob) -> Result<bool, String> {
        match self.flag(knob.env())? {
            Some(on) => Ok(on),
            None => Ok(self.flag(DENSE_NATIVE)?.unwrap_or(false)),
        }
    }

    fn dense(&self, knob: Knob) -> Result<String, String> {
        let bf16 = self.knob(knob)?;
        let fp8 = self.flag(DENSE_FP8)?.unwrap_or(false)
            && matches!(knob, Knob::Gdn | Knob::Attn | Knob::Shared);
        Ok(match (bf16, fp8) {
            (true, true) => "bf16+fp8_e4m3".into(),
            (true, false) => "bf16".into(),
            (false, true) => "fp8_e4m3".into(),
            (false, false) => "nvfp4_requant".into(),
        })
    }
}

fn dense_name(f: DenseFormat) -> &'static str {
    match f {
        DenseFormat::Nvfp4Requant => "nvfp4_requant",
        DenseFormat::Bf16 => "bf16",
        DenseFormat::Fp8E4m3 => "fp8_e4m3",
    }
}

fn required_selector(knob: Knob, want: DenseFormat) -> Option<&'static str> {
    match want {
        DenseFormat::Nvfp4Requant => None,
        DenseFormat::Bf16 => Some(knob.env()),
        DenseFormat::Fp8E4m3 => Some(DENSE_FP8),
    }
}

/// Compare the plan with the env-derived served format of every family.
pub fn check(
    plan: &PrecisionPlan,
    env: &BTreeMap<String, String>,
    catalog: &Catalog,
) -> Result<Vec<FamilyFormat>, Vec<Refusal>> {
    let sel = Selectors { env, catalog };
    let dense = [
        ("router", Knob::Router, plan.router),
        ("hyper_connection", Knob::Hc, plan.hyper_connection),
        ("gdn_proj", Knob::Gdn, plan.gdn_proj),
        ("attention_proj", Knob::Attn, plan.attention_proj),
        ("shared_expert", Knob::Shared, plan.shared_expert),
        ("lm_head", Knob::LmHead, plan.lm_head),
    ];
    let mut out = Vec::new();
    let mut refusals = Vec::new();
    for (family, knob, want) in dense {
        if let Some(selector) = required_selector(knob, want).filter(|s| !catalog.has_reader(s)) {
            refusals.push(Refusal::PrecisionUnsupported { family, selector });
            continue;
        }
        compare(
            family,
            dense_name(want),
            sel.dense(knob),
            &mut out,
            &mut refusals,
        );
    }
    let want_prefill = match plan.prefill_dense {
        PrefillFormat::FastFp8 => "fast_fp8",
        PrefillFormat::Native => "native",
    };
    if plan.prefill_dense == PrefillFormat::Native && !catalog.has_reader(Knob::Prefill.env()) {
        refusals.push(Refusal::PrecisionUnsupported {
            family: "prefill_dense",
            selector: Knob::Prefill.env(),
        });
    } else {
        let got = sel
            .knob(Knob::Prefill)
            .map(|on| (if on { "native" } else { "fast_fp8" }).to_string());
        compare("prefill_dense", want_prefill, got, &mut out, &mut refusals);
    }
    out.push(FamilyFormat {
        family: "routed_experts",
        format: "nvfp4".into(),
    });
    out.push(FamilyFormat {
        family: "gdn_state",
        format: "fp32".into(),
    });
    if refusals.is_empty() {
        Ok(out)
    } else {
        Err(refusals)
    }
}

fn compare(
    family: &'static str,
    want: &str,
    got: Result<String, String>,
    out: &mut Vec<FamilyFormat>,
    refusals: &mut Vec<Refusal>,
) {
    match got {
        Ok(got) if got == want => out.push(FamilyFormat {
            family,
            format: got,
        }),
        Ok(got) | Err(got) => refusals.push(Refusal::PrecisionMismatch {
            family,
            want: want.into(),
            got,
        }),
    }
}
