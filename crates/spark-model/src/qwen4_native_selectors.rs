// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_DENSE_NATIVE=1`: umbrella for the qwen4_exp native dense
//! precision controls (default off).
//!
//! The umbrella implies every per-module control below. A per-module env
//! that is present always wins over the umbrella, so
//! `ATLAS_QWEN4_DENSE_NATIVE=1 ATLAS_QWEN4_HC_BF16=0` is "everything native
//! except HC". With the umbrella absent or `0` each control reads exactly as
//! it did before this module existed, so the default launch is unchanged.
//!
//! Every per-module selector asks [`selected`] instead of reading its own env,
//! so the load gate, every layer, and the plan log line see one answer. Each
//! module's own refusal table still decides whether the launch is admitted.

use std::sync::OnceLock;

use anyhow::{Result, bail};

pub(crate) const DENSE_NATIVE_ENV: &str = "ATLAS_QWEN4_DENSE_NATIVE";

/// One per-module control the umbrella implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Knob {
    Router,
    Hc,
    Gdn,
    Attn,
    Shared,
    LmHead,
    Prefill,
}

impl Knob {
    pub(crate) const ALL: [Knob; 7] = [
        Knob::Router,
        Knob::Hc,
        Knob::Gdn,
        Knob::Attn,
        Knob::Shared,
        Knob::LmHead,
        Knob::Prefill,
    ];

    pub(crate) const fn env(self) -> &'static str {
        match self {
            Knob::Router => "ATLAS_QWEN4_ROUTER_BF16",
            Knob::Hc => "ATLAS_QWEN4_HC_BF16",
            Knob::Gdn => "ATLAS_QWEN4_GDN_BF16",
            Knob::Attn => "ATLAS_QWEN4_ATTN_BF16",
            Knob::Shared => "ATLAS_QWEN4_SHARED_BF16",
            Knob::LmHead => "ATLAS_QWEN4_LMHEAD_BF16",
            Knob::Prefill => "ATLAS_QWEN4_PREFILL_NATIVE",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Knob::Router => "router",
            Knob::Hc => "hc",
            Knob::Gdn => "gdn",
            Knob::Attn => "attn",
            Knob::Shared => "shared",
            Knob::LmHead => "lmhead",
            Knob::Prefill => "prefill",
        }
    }
}

/// Where a resolved value came from, for the plan line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// Neither the module env nor the umbrella is set.
    Default,
    /// The module env is set (`0` or `1`) and decides.
    Explicit,
    /// The module env is absent and the umbrella is `1`.
    Umbrella,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Resolved {
    pub(crate) on: bool,
    pub(crate) source: Source,
}

/// The resolved value of every knob, captured once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativePlan {
    pub(crate) umbrella: bool,
    values: [Resolved; 7],
}

impl NativePlan {
    pub(crate) fn get(&self, knob: Knob) -> Resolved {
        self.values[knob as usize]
    }

    pub(crate) fn on(&self, knob: Knob) -> bool {
        self.get(knob).on
    }

    /// The umbrella or any component env is present.
    pub(crate) fn any_set(&self) -> bool {
        self.umbrella
            || Knob::ALL
                .iter()
                .any(|k| self.get(*k).source != Source::Default)
    }

    /// Env-shaped view of the resolved plan: `Some("1")` for a knob that is on,
    /// `None` otherwise (including names that are not knobs). Lets code that
    /// was written against raw env reads see umbrella-implied values.
    pub(crate) fn reader(&self) -> impl Fn(&str) -> Option<String> + '_ {
        move |name| {
            Knob::ALL
                .into_iter()
                .find(|k| k.env() == name)
                .filter(|k| self.on(*k))
                .map(|_| "1".to_string())
        }
    }

    /// `QWEN4_DENSE_NATIVE plan: router=on(umbrella) hc=off(explicit) ...`
    pub(crate) fn summary_line(&self) -> String {
        let mut line = String::from("QWEN4_DENSE_NATIVE plan:");
        for knob in Knob::ALL {
            let r = self.get(knob);
            let state = if r.on { "on" } else { "off" };
            let source = match r.source {
                Source::Default => "default",
                Source::Explicit => "explicit",
                Source::Umbrella => "umbrella",
            };
            line.push_str(&format!(" {}={state}({source})", knob.label()));
        }
        line
    }
}

fn parse_flag(name: &str, raw: Option<&str>) -> Result<Option<bool>> {
    match raw {
        None => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(_) => bail!("{name} must be exactly 0 or 1"),
    }
}

/// Pure precedence rule: a present module env beats the umbrella.
pub(crate) fn resolve(knob: Knob, umbrella: bool, explicit: Option<&str>) -> Result<Resolved> {
    Ok(match parse_flag(knob.env(), explicit)? {
        Some(on) => Resolved {
            on,
            source: Source::Explicit,
        },
        None if umbrella => Resolved {
            on: true,
            source: Source::Umbrella,
        },
        None => Resolved {
            on: false,
            source: Source::Default,
        },
    })
}

/// Pure plan builder over an env lookup (`Ok(None)` = absent).
pub(crate) fn plan_from(lookup: impl Fn(&str) -> Result<Option<String>>) -> Result<NativePlan> {
    let umbrella =
        parse_flag(DENSE_NATIVE_ENV, lookup(DENSE_NATIVE_ENV)?.as_deref())?.unwrap_or(false);
    let mut values = [Resolved {
        on: false,
        source: Source::Default,
    }; 7];
    for knob in Knob::ALL {
        values[knob as usize] = resolve(knob, umbrella, lookup(knob.env())?.as_deref())?;
    }
    Ok(NativePlan { umbrella, values })
}

fn read_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}

/// Process-wide plan, read once.
pub(crate) fn plan() -> Result<NativePlan> {
    static PLAN: OnceLock<Result<NativePlan, String>> = OnceLock::new();
    PLAN.get_or_init(|| plan_from(read_env).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// What every per-module selector calls instead of reading its env.
pub(crate) fn selected(knob: Knob) -> Result<bool> {
    Ok(plan()?.on(knob))
}

/// Emit the plan line once at load whenever the umbrella env is explicitly set
/// or any component selector is enabled (matching [`Plan::any_set`]). Quiet
/// otherwise, so the default launch log is unchanged.
pub(crate) fn log_plan_once() -> Result<()> {
    let plan = plan()?;
    if !plan.any_set() {
        return Ok(());
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| tracing::info!("{}", plan.summary_line()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn plan_of(pairs: &[(&str, &str)]) -> Result<NativePlan> {
        let env: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        plan_from(|name| Ok(env.get(name).cloned()))
    }

    #[test]
    fn empty_env_is_all_off_default() {
        let plan = plan_of(&[]).unwrap();
        assert!(!plan.umbrella);
        for knob in Knob::ALL {
            assert_eq!(
                plan.get(knob),
                Resolved {
                    on: false,
                    source: Source::Default
                }
            );
        }
    }

    #[test]
    fn umbrella_implies_every_knob() {
        let plan = plan_of(&[(DENSE_NATIVE_ENV, "1")]).unwrap();
        for knob in Knob::ALL {
            assert_eq!(
                plan.get(knob),
                Resolved {
                    on: true,
                    source: Source::Umbrella
                }
            );
        }
    }

    #[test]
    fn explicit_zero_beats_umbrella() {
        let plan = plan_of(&[(DENSE_NATIVE_ENV, "1"), ("ATLAS_QWEN4_HC_BF16", "0")]).unwrap();
        assert_eq!(
            plan.get(Knob::Hc),
            Resolved {
                on: false,
                source: Source::Explicit
            }
        );
        assert!(plan.on(Knob::Router) && plan.on(Knob::Prefill));
    }

    #[test]
    fn explicit_one_without_umbrella_is_unchanged_behaviour() {
        let plan = plan_of(&[("ATLAS_QWEN4_ROUTER_BF16", "1")]).unwrap();
        assert_eq!(
            plan.get(Knob::Router),
            Resolved {
                on: true,
                source: Source::Explicit
            }
        );
        assert!(!plan.on(Knob::LmHead));
    }

    #[test]
    fn plan_line_prints_when_any_component_is_set() {
        assert!(!plan_of(&[]).unwrap().any_set());
        assert!(
            plan_of(&[("ATLAS_QWEN4_SHARED_BF16", "1")])
                .unwrap()
                .any_set()
        );
        assert!(
            plan_of(&[("ATLAS_QWEN4_SHARED_BF16", "0")])
                .unwrap()
                .any_set()
        );
        assert!(plan_of(&[(DENSE_NATIVE_ENV, "1")]).unwrap().any_set());
        // Umbrella explicitly 0 with nothing else set stays quiet.
        assert!(!plan_of(&[(DENSE_NATIVE_ENV, "0")]).unwrap().any_set());
    }

    #[test]
    fn reader_reports_umbrella_implied_knobs() {
        let plan = plan_of(&[(DENSE_NATIVE_ENV, "1"), ("ATLAS_QWEN4_ATTN_BF16", "0")]).unwrap();
        let read = plan.reader();
        assert_eq!(read("ATLAS_QWEN4_GDN_BF16").as_deref(), Some("1"));
        assert_eq!(read("ATLAS_QWEN4_ATTN_BF16"), None);
        assert_eq!(read("ATLAS_QWEN4_DENSE_FP8"), None);
    }

    #[test]
    fn umbrella_zero_is_off() {
        let plan = plan_of(&[(DENSE_NATIVE_ENV, "0")]).unwrap();
        assert!(Knob::ALL.iter().all(|k| !plan.on(*k)));
    }

    #[test]
    fn bad_values_are_refused() {
        assert!(plan_of(&[(DENSE_NATIVE_ENV, "yes")]).is_err());
        let err = plan_of(&[(DENSE_NATIVE_ENV, "1"), ("ATLAS_QWEN4_GDN_BF16", "true")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("ATLAS_QWEN4_GDN_BF16"), "{err}");
        // A malformed module env is refused even without the umbrella.
        assert!(plan_of(&[("ATLAS_QWEN4_SHARED_BF16", "")]).is_err());
    }

    #[test]
    fn env_names_match_the_module_selectors() {
        assert_eq!(
            Knob::Router.env(),
            crate::layers::moe::qwen4_router_bf16::QWEN4_ROUTER_BF16_ENV
        );
        // `factory::qwen4_lmhead_bf16` is private to the factory; its env is
        // pinned against the same literal in that module's own tests.
        assert_eq!(Knob::LmHead.env(), "ATLAS_QWEN4_LMHEAD_BF16");
        assert_eq!(
            Knob::Gdn.env(),
            crate::layers::qwen4_decode_bf16_arms::QWEN4_GDN_BF16_ENV
        );
        assert_eq!(
            Knob::Attn.env(),
            crate::layers::qwen4_decode_bf16_arms::QWEN4_ATTN_BF16_ENV
        );
        assert_eq!(
            Knob::Shared.env(),
            crate::layers::moe::qwen4_shared_bf16::QWEN4_SHARED_BF16_ENV
        );
        assert_eq!(
            Knob::Prefill.env(),
            crate::layers::qwen4_prefill_native::SELECTOR
        );
        assert_eq!(
            Knob::Hc.env(),
            crate::layers::qwen4_hc_bf16::QWEN4_HC_BF16_ENV
        );
    }

    #[test]
    fn summary_line_lists_every_knob_in_order() {
        let plan = plan_of(&[(DENSE_NATIVE_ENV, "1"), ("ATLAS_QWEN4_LMHEAD_BF16", "0")]).unwrap();
        assert_eq!(
            plan.summary_line(),
            "QWEN4_DENSE_NATIVE plan: router=on(umbrella) hc=on(umbrella) gdn=on(umbrella) \
             attn=on(umbrella) shared=on(umbrella) lmhead=off(explicit) prefill=on(umbrella)"
        );
    }
}
