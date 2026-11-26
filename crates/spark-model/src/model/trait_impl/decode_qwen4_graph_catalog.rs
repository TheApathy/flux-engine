// SPDX-License-Identifier: AGPL-3.0-only

//! Prepared CUDA-graph catalog policy for Qwen4 plain decode.
//!
//! Without the catalog every decode graph is captured at first use and is
//! destroyed with its sequence, and a C2+ frame never replays a graph: the
//! native frame is eager-only and the serialized fallback suppresses graphs
//! for its whole row loop. Under `ATLAS_QWEN4_GRAPH_CATALOG` the server
//! declares every graph it will replay before READY, captures each one once
//! on throwaway sequences, and seals the catalog. After the seal nothing is
//! captured: a frame replays a prepared entry, or runs eagerly with a typed
//! reason (`prepare`), or is refused with that reason (`strict`).
//!
//! This module is the pure policy: selector parsing, the declared key set,
//! load-time refusals, the per-frame decision and the memory verdict. It owns
//! no GPU state, so every rule here is covered by CPU tests.

#![allow(dead_code)]

use std::collections::BTreeSet;

pub(super) const CATALOG_ENV: &str = "ATLAS_QWEN4_GRAPH_CATALOG";
pub(super) const WIDTHS_ENV: &str = "ATLAS_QWEN4_GRAPH_CATALOG_WIDTHS";
pub(super) const QSA_ROWS_ENV: &str = "ATLAS_QWEN4_GRAPH_CATALOG_QSA";
pub(super) const MAX_MIB_ENV: &str = "ATLAS_QWEN4_GRAPH_CATALOG_MAX_MIB";
pub(super) const DEFAULT_MAX_MIB: usize = 512;
const QSA_BOUNDARY: usize = 2_048;
/// Widths below this are all prepared by default; above it only powers of two.
const DEFAULT_EVERY_WIDTH_LIMIT: usize = 8;
const MIB: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CatalogMode {
    /// Lazy per-sequence capture, byte-for-byte today's behaviour.
    Off,
    /// Prepare before READY; an off-catalog frame runs eagerly with a reason.
    Prepare,
    /// Prepare before READY; an off-catalog frame is refused with a reason.
    Strict,
}

pub(super) fn parse_catalog_mode(raw: Option<&str>) -> Result<CatalogMode, &'static str> {
    match raw {
        None | Some("off") => Ok(CatalogMode::Off),
        Some("prepare") => Ok(CatalogMode::Prepare),
        Some("strict") => Ok(CatalogMode::Strict),
        Some(_) => Err("ATLAS_QWEN4_GRAPH_CATALOG must be exactly off, prepare or strict"),
    }
}

pub(super) fn parse_qsa_rows(raw: Option<&str>) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_GRAPH_CATALOG_QSA must be exact value 0 or 1"),
    }
}

pub(super) fn parse_max_mib(raw: Option<&str>) -> Result<usize, &'static str> {
    match raw {
        None => Ok(DEFAULT_MAX_MIB),
        Some(value) => match value.parse::<usize>() {
            Ok(mib) if mib > 0 && mib.checked_mul(MIB).is_some() => Ok(mib),
            _ => Err("ATLAS_QWEN4_GRAPH_CATALOG_MAX_MIB must be a positive MiB count"),
        },
    }
}

/// Attention topology of one decode row. The names match the per-row routes
/// of the batched-QSA lane, so a frame's key and its eager route agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Regime {
    /// No QSA side index; admission keeps the request below 2048 tokens.
    Dense,
    /// QSA-required below 2048: graph-safe indexer, then dense attention.
    IndexedDense,
    /// QSA-required at or past 2048: graph-safe indexer, then sparse attention.
    Sparse,
}

/// `host_seq_len` is the pre-step length, the value the C1 route compares
/// with 2048 (`decode_qwen4_graph_policy.rs`).
pub(super) const fn row_regime(qsa_required: bool, host_seq_len: usize) -> Regime {
    if !qsa_required {
        Regime::Dense
    } else if host_seq_len < QSA_BOUNDARY {
        Regime::IndexedDense
    } else {
        Regime::Sparse
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum GraphKey {
    /// C1 decode of layers 1.. and the head. Recurrent-state pointers are
    /// baked into the graph, so the entry belongs to one pool slot, whose
    /// addresses are fixed for the model lifetime.
    Row { slot: usize, regime: Regime },
    /// Native C2+ frame of layers 1.., the final mixer and the head. Per-row
    /// addresses come from fixed device tables uploaded before replay, so the
    /// entry is slot-agnostic and keyed by the exact active width.
    Native { width: usize, regime: Regime },
}

/// Default native widths: every width up to eight, then powers of two up to
/// `min(max_batch_size, QWEN4_NATIVE_MAX_ROWS)` (e.g. 16, 32, 64, 128, 256).
pub(super) fn default_widths(max_batch_size: usize) -> Vec<usize> {
    let top = max_batch_size.min(crate::layers::QWEN4_NATIVE_MAX_ROWS);
    (2..=top)
        .filter(|&width| width <= DEFAULT_EVERY_WIDTH_LIMIT || width.is_power_of_two())
        .collect()
}

pub(super) fn parse_widths(
    raw: Option<&str>,
    max_batch_size: usize,
) -> Result<Vec<usize>, &'static str> {
    let Some(raw) = raw else {
        return Ok(default_widths(max_batch_size));
    };
    let top = max_batch_size.min(crate::layers::QWEN4_NATIVE_MAX_ROWS);
    let mut widths = Vec::new();
    for field in raw.split(',') {
        let width = field
            .trim()
            .parse::<usize>()
            .map_err(|_| "ATLAS_QWEN4_GRAPH_CATALOG_WIDTHS must be comma-separated widths")?;
        if width < 2 || width > top {
            return Err("ATLAS_QWEN4_GRAPH_CATALOG_WIDTHS admits only C2..min(max batch, C256)");
        }
        if widths.last().is_some_and(|&last| width <= last) {
            return Err("ATLAS_QWEN4_GRAPH_CATALOG_WIDTHS must be strictly increasing");
        }
        widths.push(width);
    }
    Ok(widths)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PlanInput {
    pub max_batch_size: usize,
    pub native_selected: bool,
    pub qsa_enabled: bool,
    pub qsa_graph_safe: bool,
    pub qsa_rows_selected: bool,
    pub widths: Vec<usize>,
}

/// Every entry the warm-up must capture. Native frames are declared only for
/// all-Dense rows: a QSA row keeps the native frame eager until a width-only
/// QSA route exists.
pub(super) fn declared_keys(input: &PlanInput) -> BTreeSet<GraphKey> {
    let mut row_regimes = vec![Regime::Dense];
    if input.qsa_enabled && input.qsa_graph_safe && input.qsa_rows_selected {
        row_regimes.extend([Regime::IndexedDense, Regime::Sparse]);
    }
    let mut keys = BTreeSet::new();
    for slot in 0..input.max_batch_size {
        for &regime in &row_regimes {
            keys.insert(GraphKey::Row { slot, regime });
        }
    }
    if input.native_selected {
        for &width in &input.widths {
            keys.insert(GraphKey::Native {
                width,
                regime: Regime::Dense,
            });
        }
    }
    keys
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LoadRefusal {
    NotQwen4,
    Speculative,
    MultiRank,
    Profile,
    HighSpeedSwap,
    Fp8KvCalibration,
    SsmHFp16,
    SeqDependentSplitK,
    DiagnosticSelector,
}

impl LoadRefusal {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::NotQwen4 => "not_qwen4",
            Self::Speculative => "speculative",
            Self::MultiRank => "multi_rank",
            Self::Profile => "profile",
            Self::HighSpeedSwap => "high_speed_swap",
            Self::Fp8KvCalibration => "fp8_kv_calibration",
            Self::SsmHFp16 => "ssm_h_fp16",
            Self::SeqDependentSplitK => "seq_dependent_split_k",
            Self::DiagnosticSelector => "diagnostic_selector",
        }
    }
}

/// Process facts that make a prepared graph unsound or unreachable. A graph
/// freezes host-side launch decisions; any decision that varies with
/// sequence length or across a process toggle cannot live in the catalog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LoadFacts {
    pub qwen4: bool,
    pub speculative: bool,
    pub multi_rank: bool,
    pub profile: bool,
    pub high_speed_swap: bool,
    pub fp8_kv_calibration: bool,
    pub ssm_h_fp16: bool,
    /// `ATLAS_PAGED_DECODE_SPLITK` or `ATLAS_FLASH_ATTN_KGAMMA_SPLITK`: the
    /// split count follows the host `max_seq_len`.
    pub seq_dependent_split_k: bool,
    /// Per-layer dumps, stage timers or the K1 stage diagnostic.
    pub diagnostic_selector: bool,
}

pub(super) fn load_refusal(facts: &LoadFacts) -> Option<LoadRefusal> {
    let checks = [
        (!facts.qwen4, LoadRefusal::NotQwen4),
        (facts.speculative, LoadRefusal::Speculative),
        (facts.multi_rank, LoadRefusal::MultiRank),
        (facts.profile, LoadRefusal::Profile),
        (facts.high_speed_swap, LoadRefusal::HighSpeedSwap),
        (facts.fp8_kv_calibration, LoadRefusal::Fp8KvCalibration),
        (facts.ssm_h_fp16, LoadRefusal::SsmHFp16),
        (facts.seq_dependent_split_k, LoadRefusal::SeqDependentSplitK),
        (facts.diagnostic_selector, LoadRefusal::DiagnosticSelector),
    ];
    checks
        .into_iter()
        .find_map(|(refused, reason)| refused.then_some(reason))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OffCatalogReason {
    WidthNotPrepared,
    SlotNotPrepared,
    RegimeNotPrepared,
    /// Native frame with a QSA row: eager by the batched-QSA lane's contract.
    NativeQsaEager,
    ImageRotary,
    DiagnosticFrame,
    EntryMissing,
    EntryPoisoned,
}

impl OffCatalogReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::WidthNotPrepared => "width_not_prepared",
            Self::SlotNotPrepared => "slot_not_prepared",
            Self::RegimeNotPrepared => "regime_not_prepared",
            Self::NativeQsaEager => "native_qsa_eager",
            Self::ImageRotary => "image_rotary",
            Self::DiagnosticFrame => "diagnostic_frame",
            Self::EntryMissing => "entry_missing",
            Self::EntryPoisoned => "entry_poisoned",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RowFacts {
    pub slot: usize,
    pub qsa_required: bool,
    pub host_seq_len: usize,
    pub identity_rotary: bool,
}

/// The route a frame already took. Native-versus-serial selection, and its
/// serial reason, belong to the admission owner; the catalog only keys it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Frame<'a> {
    /// One C1 row: plain decode, or one row of the serialized C2+ loop.
    Row(&'a RowFacts),
    /// One native C2+ frame over these active rows.
    Native(&'a [RowFacts]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EntryState {
    Absent,
    Ready,
    /// A replay failed. The entry is destroyed and never captured again.
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Decision {
    /// Catalog off: the lazy path runs unchanged.
    Lazy,
    /// Before the seal only: capture this declared entry, then launch it.
    Capture(GraphKey),
    Replay(GraphKey),
    Eager(OffCatalogReason),
    Refuse(OffCatalogReason),
}

pub(super) fn frame_key(frame: Frame<'_>) -> Result<GraphKey, OffCatalogReason> {
    match frame {
        Frame::Row(row) => {
            if !row.identity_rotary {
                return Err(OffCatalogReason::ImageRotary);
            }
            Ok(GraphKey::Row {
                slot: row.slot,
                regime: row_regime(row.qsa_required, row.host_seq_len),
            })
        }
        Frame::Native(rows) => {
            if rows.iter().any(|row| !row.identity_rotary) {
                return Err(OffCatalogReason::ImageRotary);
            }
            if rows.is_empty() {
                return Err(OffCatalogReason::WidthNotPrepared);
            }
            // A native frame with any QSA row runs the batched-QSA lane's
            // per-row routes (receipt `qsa=true routes=..`), which are eager
            // until that lane provides a width-only device-routed frame.
            if rows.iter().any(|row| row.qsa_required) {
                return Err(OffCatalogReason::NativeQsaEager);
            }
            Ok(GraphKey::Native {
                width: rows.len(),
                regime: Regime::Dense,
            })
        }
    }
}

fn undeclared_reason(key: GraphKey, declared: &BTreeSet<GraphKey>) -> OffCatalogReason {
    match key {
        GraphKey::Row { slot, .. } => {
            if declared
                .iter()
                .any(|entry| matches!(entry, GraphKey::Row { slot: s, .. } if *s == slot))
            {
                OffCatalogReason::RegimeNotPrepared
            } else {
                OffCatalogReason::SlotNotPrepared
            }
        }
        GraphKey::Native { width, .. } => {
            if declared
                .iter()
                .any(|entry| matches!(entry, GraphKey::Native { width: w, .. } if *w == width))
            {
                OffCatalogReason::RegimeNotPrepared
            } else {
                OffCatalogReason::WidthNotPrepared
            }
        }
    }
}

const fn off_catalog(mode: CatalogMode, reason: OffCatalogReason) -> Decision {
    match mode {
        CatalogMode::Strict => Decision::Refuse(reason),
        CatalogMode::Off | CatalogMode::Prepare => Decision::Eager(reason),
    }
}

/// One decision per frame. After the seal the result is never `Capture`.
pub(super) fn decide(
    mode: CatalogMode,
    sealed: bool,
    frame: Frame<'_>,
    diagnostic: bool,
    declared: &BTreeSet<GraphKey>,
    state: impl Fn(&GraphKey) -> EntryState,
) -> Decision {
    if mode == CatalogMode::Off {
        return Decision::Lazy;
    }
    if diagnostic {
        return off_catalog(mode, OffCatalogReason::DiagnosticFrame);
    }
    let key = match frame_key(frame) {
        Ok(key) => key,
        Err(reason) => return off_catalog(mode, reason),
    };
    if !declared.contains(&key) {
        return off_catalog(mode, undeclared_reason(key, declared));
    }
    match state(&key) {
        EntryState::Ready => Decision::Replay(key),
        EntryState::Poisoned => off_catalog(mode, OffCatalogReason::EntryPoisoned),
        EntryState::Absent if !sealed => Decision::Capture(key),
        EntryState::Absent => off_catalog(mode, OffCatalogReason::EntryMissing),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BudgetRefusal {
    /// `/proc/meminfo` gave no `MemAvailable` around the warm-up.
    Unmeasured,
    OverCap,
    UnderFloor,
}

impl BudgetRefusal {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Unmeasured => "unmeasured",
            Self::OverCap => "over_cap",
            Self::UnderFloor => "under_floor",
        }
    }
}

/// Fail closed: the warm-up's measured `MemAvailable` delta must fit the cap,
/// and the host must stay at or above the declared floor afterwards.
pub(super) fn budget_verdict(
    measured_delta_bytes: Option<i64>,
    cap_mib: usize,
    mem_available_after: Option<usize>,
    floor_bytes: usize,
) -> Result<(), BudgetRefusal> {
    let (Some(delta), Some(after)) = (measured_delta_bytes, mem_available_after) else {
        return Err(BudgetRefusal::Unmeasured);
    };
    let cap_bytes = cap_mib.saturating_mul(MIB);
    if delta > 0 && delta as u64 > cap_bytes as u64 {
        return Err(BudgetRefusal::OverCap);
    }
    if after < floor_bytes {
        return Err(BudgetRefusal::UnderFloor);
    }
    Ok(())
}

#[cfg(test)]
#[path = "decode_qwen4_graph_catalog_tests.rs"]
mod tests;
