// SPDX-License-Identifier: AGPL-3.0-only

//! "A reader pins its source; a release skips a pinned source and says so."
//!
//! The memlean releases (`ATLAS_QWEN4_RELEASE_MXFP8_SOURCES`, default on, and
//! `ATLAS_QWEN4_DENSE_DEDUP`) free checkpoint store tensors once their
//! NVFP4 copies exist. The native-dense controls keep some of those same
//! store tensors as live operands (`qwen4_prefill_native::register_bf16`
//! keys BF16 and MXFP8 store aliases, `ATLAS_QWEN4_HC_BF16` keeps the HC
//! mixers' store BF16, `ATLAS_QWEN4_ROUTER_BF16` reads `mlp.gate.weight`,
//! `ATLAS_QWEN4_DENSE_FP8` aliases the MXFP8 originals). Freeing a pinned
//! source is a use-after-free on GB10.
//!
//! [`SourcePins`] maps each selector that is on (umbrella-aware through
//! `qwen4_native_selectors`) to the tensor families it reads, before any
//! release runs. Every memlean store-release primitive asks [`pinned_by`]
//! first; a pinned name is skipped and counted, and the counts print as
//! `MXFP8_RELEASE_SKIPPED pinned_by=.. layer=.. family=.. names=.. bytes=..`
//! next to the release receipt of the same layer group.
//!
//! Two kinds of store tensor, pinned separately ([`PinKind`]):
//! - `Mxfp8Source`: the FP8/MXFP8 checkpoint originals
//!   (`release_fp8_store_source`). Only `ATLAS_QWEN4_DENSE_FP8` aliases them
//!   (`MxFp8Proj::alias`). The BF16 arms and prefill-native read them once,
//!   synchronously, into owned buffers before the end-of-layer release
//!   (`quantized_modelopt_mixed`, `register_mxfp8` via a host read,
//!   `dequant_mxfp8_to_bf16`), so they do not pin them. What that frees is
//!   reported as `MXFP8_RELEASE_UNPINNED_BF16_ARMS`.
//! - `Bf16Alias`: store BF16 tensors and the dedup BF16 copies a reader
//!   keeps (router, HC mixers, LM head, attention fallbacks).
//!
//! Owned BF16 dequant buffers the loader frees
//! (`release_replaced_bf16_source`'s owned branch) are not sources; a reader
//! that needs one keeps the handle and the loader skips that free itself.

use std::cell::RefCell;
use std::collections::BTreeMap;

use anyhow::Result;

use crate::qwen4_native_selectors::{Knob, NativePlan};

/// Which store copy a release frees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinKind {
    Mxfp8Source,
    Bf16Alias,
}

/// A family of checkpoint tensors a reader can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PinFamily {
    Attn,
    Gdn,
    Shared,
    Router,
    Hc,
    LmHead,
}

impl PinFamily {
    /// The family of a store tensor name or name prefix, if any.
    pub(crate) fn of(name: &str) -> Option<Self> {
        let attn_proj = ["q_proj", "k_proj", "v_proj", "o_proj"]
            .iter()
            .any(|p| name.contains(&format!(".self_attn.{p}")));
        if attn_proj {
            Some(Self::Attn)
        } else if ["in_proj_qkv", "in_proj_z", "out_proj"]
            .iter()
            .any(|p| name.contains(&format!(".linear_attn.{p}")))
        {
            // in_proj_a/b are consumed into the interleaved BA copy; no reader holds them.
            Some(Self::Gdn)
        } else if name.contains(".mlp.shared_expert.") {
            Some(Self::Shared)
        } else if name.ends_with(".mlp.gate") || name.ends_with(".mlp.gate.weight") {
            Some(Self::Router)
        } else if [
            "input_mix_weight",
            "block_inject_weight",
            "hyper_connection",
        ]
        .iter()
        .any(|p| name.contains(p))
        {
            Some(Self::Hc)
        } else if name.starts_with("lm_head") || name.contains(".lm_head") {
            Some(Self::LmHead)
        } else {
            None
        }
    }
}

const NATIVE_READS: [(Knob, &[PinFamily]); 7] = [
    (Knob::Router, &[PinFamily::Router]),
    (Knob::Hc, &[PinFamily::Hc]),
    (Knob::Gdn, &[PinFamily::Gdn]),
    (Knob::Attn, &[PinFamily::Attn]),
    (Knob::Shared, &[PinFamily::Shared]),
    (Knob::LmHead, &[PinFamily::LmHead]),
    // register_bf16 / register_router / ships_mxfp8: every native family.
    (
        Knob::Prefill,
        &[
            PinFamily::Attn,
            PinFamily::Gdn,
            PinFamily::Shared,
            PinFamily::Router,
            PinFamily::Hc,
        ],
    ),
];

/// Selectors outside the umbrella that alias the MXFP8 originals.
/// `ATLAS_QWEN4_DENSE_MXFP8` is the name dense dedup's earlier refusal used.
const MXFP8_ALIAS_READERS: [&str; 2] = ["ATLAS_QWEN4_DENSE_FP8", "ATLAS_QWEN4_DENSE_MXFP8"];
const MXFP8_ALIAS_FAMILIES: &[PinFamily] = &[PinFamily::Attn, PinFamily::Gdn, PinFamily::Shared];

/// Which selector pins which family, captured before any release runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourcePins {
    /// BF16-alias pins (every reader).
    pins: Vec<(PinFamily, &'static str)>,
    /// MXFP8-source pins (only the selectors that alias the originals).
    mxfp8: Vec<(PinFamily, &'static str)>,
}

impl SourcePins {
    /// No reader engaged: every release runs.
    #[cfg(test)]
    pub(crate) fn none() -> Self {
        Self::default()
    }

    /// The pins of a resolved native plan plus the raw MXFP8-alias selectors.
    pub(crate) fn from_plan(plan: &NativePlan, alias_on: impl Fn(&str) -> bool) -> Self {
        let native = NATIVE_READS
            .iter()
            .filter(|(knob, _)| plan.on(*knob))
            .flat_map(|(knob, families)| families.iter().map(move |f| (*f, knob.env())));
        let alias: Vec<_> = MXFP8_ALIAS_READERS
            .into_iter()
            .filter(|name| alias_on(name))
            .flat_map(|name| MXFP8_ALIAS_FAMILIES.iter().map(move |f| (*f, name)))
            .collect();
        Self {
            pins: native.chain(alias.iter().copied()).collect(),
            mxfp8: alias,
        }
    }

    /// `ATLAS_QWEN4_DENSE_FP8` is not an umbrella knob: the umbrella never
    /// selects it, and it engages the FP8 arm for attn/gdn/shared together.
    /// Its pin reads the same once-latched answer the loader engages on
    /// (`qwen4_dense_fp8::selected`), never the umbrella flag.
    pub(crate) fn from_env() -> Result<Self> {
        let plan = crate::qwen4_native_selectors::plan()?;
        let dense_fp8 = crate::layers::qwen4_dense_fp8::selected()?;
        Ok(Self::from_plan(&plan, |name| match name {
            "ATLAS_QWEN4_DENSE_FP8" => dense_fp8,
            _ => std::env::var(name)
                .ok()
                .is_some_and(|v| !v.is_empty() && v != "0"),
        }))
    }

    /// The first selector that pins `name`'s family for this kind of copy.
    pub(crate) fn pinned_by(&self, name: &str, kind: PinKind) -> Option<&'static str> {
        let family = PinFamily::of(name)?;
        let table = match kind {
            PinKind::Mxfp8Source => &self.mxfp8,
            PinKind::Bf16Alias => &self.pins,
        };
        table
            .iter()
            .find(|(f, _)| *f == family)
            .map(|(_, selector)| *selector)
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }
}

/// Skipped names and bytes per pinning selector, since the last flush.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Skipped {
    pub(crate) names: usize,
    pub(crate) bytes: usize,
}

thread_local! {
    /// Test override of the process pins (loads build layers on one thread).
    static OVERRIDE: RefCell<Option<SourcePins>> = const { RefCell::new(None) };
    static PENDING: RefCell<BTreeMap<&'static str, Skipped>> = const { RefCell::new(BTreeMap::new()) };
    static TOTAL: RefCell<Skipped> = const { RefCell::new(Skipped { names: 0, bytes: 0 }) };
    /// MXFP8 originals released although a BF16 / prefill-native arm reads the family.
    static UNPINNED: RefCell<Skipped> = const { RefCell::new(Skipped { names: 0, bytes: 0 }) };
}

fn current() -> Result<SourcePins> {
    match OVERRIDE.with(|o| o.borrow().clone()) {
        Some(pins) => Ok(pins),
        None => SourcePins::from_env(),
    }
}

fn add(s: &mut Skipped, bytes: usize) {
    s.names += 1;
    s.bytes = s.bytes.saturating_add(bytes);
}

/// `Some(selector)` when `name` is pinned for `kind`; the skip is recorded
/// with `bytes`. An MXFP8 original released while a BF16 arm reads its
/// family is counted for `MXFP8_RELEASE_UNPINNED_BF16_ARMS`.
pub(crate) fn skip_if_pinned(
    name: &str,
    bytes: usize,
    kind: PinKind,
) -> Result<Option<&'static str>> {
    let pins = current()?;
    let Some(selector) = pins.pinned_by(name, kind) else {
        if kind == PinKind::Mxfp8Source && pins.pinned_by(name, PinKind::Bf16Alias).is_some() {
            UNPINNED.with(|u| add(&mut u.borrow_mut(), bytes));
        }
        return Ok(None);
    };
    PENDING.with(|p| add(p.borrow_mut().entry(selector).or_default(), bytes));
    TOTAL.with(|t| add(&mut t.borrow_mut(), bytes));
    Ok(Some(selector))
}

/// One receipt line per pinning selector for this layer group.
pub(crate) fn skipped_lines(layer: &str, family: &str) -> Vec<String> {
    PENDING
        .with(|p| std::mem::take(&mut *p.borrow_mut()))
        .into_iter()
        .map(|(selector, s)| {
            format!(
                "MXFP8_RELEASE_SKIPPED pinned_by={selector} layer={layer} family={family} names={} bytes={}",
                s.names, s.bytes
            )
        })
        .collect()
}

pub(crate) fn flush(layer: &str, family: &str) {
    for line in skipped_lines(layer, family) {
        tracing::info!("{line}");
    }
}

/// Load-wide skipped total, drained once at the end of load.
pub(crate) fn take_total() -> Skipped {
    TOTAL.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

/// Load-wide MXFP8 bytes released under a BF16 / prefill-native arm.
pub(crate) fn take_unpinned() -> Skipped {
    UNPINNED.with(|u| std::mem::take(&mut *u.borrow_mut()))
}

/// The end-of-load lines, after the last release (factory build): skipped
/// and unpinned totals, plus the matching startup-ledger rows.
pub(crate) fn total_lines(skipped: Skipped, unpinned: Skipped) -> Vec<String> {
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mut lines = Vec::new();
    if skipped.names > 0 {
        lines.push(format!(
            "MXFP8_RELEASE_SKIPPED_TOTAL names={} bytes={} gib={:.2}",
            skipped.names,
            skipped.bytes,
            gib(skipped.bytes)
        ));
    }
    if unpinned.names > 0 {
        lines.push(format!(
            "MXFP8_RELEASE_UNPINNED_BF16_ARMS names={} bytes={} gib={:.2}",
            unpinned.names,
            unpinned.bytes,
            gib(unpinned.bytes)
        ));
    }
    lines
}

pub(crate) fn emit_totals() {
    let (skipped, unpinned) = (take_total(), take_unpinned());
    for line in total_lines(skipped, unpinned) {
        tracing::info!("{line}");
    }
    if skipped.names > 0 {
        spark_runtime::startup_ledger::record(
            "mxfp8-release-skipped",
            format!(
                "names={} bytes={} (pinned by native-dense readers)",
                skipped.names, skipped.bytes
            ),
            None,
            None,
        );
    }
    if unpinned.names > 0 {
        spark_runtime::startup_ledger::record(
            "mxfp8-release-unpinned-bf16-arms",
            format!(
                "names={} bytes={} (MXFP8 originals freed under BF16/prefill-native arms)",
                unpinned.names, unpinned.bytes
            ),
            None,
            Some(-i64::try_from(unpinned.bytes).unwrap_or(i64::MAX)),
        );
    }
}

/// Install `pins` for this thread until the guard drops (tests).
#[cfg(test)]
pub(crate) fn scoped(pins: SourcePins) -> Scope {
    OVERRIDE.with(|o| *o.borrow_mut() = Some(pins));
    PENDING.with(|p| p.borrow_mut().clear());
    TOTAL.with(|t| *t.borrow_mut() = Skipped::default());
    UNPINNED.with(|u| *u.borrow_mut() = Skipped::default());
    Scope
}

#[cfg(test)]
pub(crate) struct Scope;

#[cfg(test)]
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|o| *o.borrow_mut() = None);
        PENDING.with(|p| p.borrow_mut().clear());
    }
}
