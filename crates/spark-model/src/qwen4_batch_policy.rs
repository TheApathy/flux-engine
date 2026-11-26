// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 (`qwen4_exp`) batch-versus-serial decode policy.
//!
//! Native C2-C256 batching (`QWEN4_NATIVE_MAX_ROWS`) is the default decode route for Qwen4 whenever the
//! server admits more than one live row. A frame that does not take it always
//! carries a typed [`SerialReason`]; a configuration that can never take it is
//! either resolved to a process-wide serial plan (the operator left
//! `ATLAS_QWEN4_NATIVE_BATCH` unset) or refused at load (the operator set it
//! to `1`). Nothing here touches the GPU. The model applies [`route_frame`]
//! inside `decode_batch`, and the scheduler applies the same function to the
//! same facts only to count and log, so the two cannot disagree.
//!
//! Load-time refusals that already exist for every Qwen4 C2+ server
//! (speculation, prefix caching, SSM snapshots, swap, TP/EP, non-BF16 KV,
//! extended YaRN; `spark-server` `qwen4_concurrency.rs`) are not serial
//! reasons: no C2+ frame can be built under them.

/// Operator selector. Unset is the default-on `Auto` plan.
pub const QWEN4_NATIVE_BATCH_ENV: &str = "ATLAS_QWEN4_NATIVE_BATCH";

/// Widest native frame. Also the server's Qwen4 concurrency ceiling unless
/// the operator opts out with `ATLAS_QWEN4_NATIVE_BATCH=0`.
pub const QWEN4_NATIVE_MAX_ROWS: usize = crate::layers::QWEN4_NATIVE_MAX_ROWS;

/// Dense attention is exact below this many tokens; a row that can reach it
/// must keep its QSA side index and therefore decodes serially.
pub const QWEN4_QSA_DENSE_LIMIT: usize = 2_048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBatchSelector {
    /// Unset: native when the process supports it, otherwise serial with a
    /// logged, counted [`SerialReason`].
    Auto,
    /// `1`: native or refuse at load. Qualification runs use this so an
    /// incompatible recipe cannot silently produce a serial receipt.
    Required,
    /// `0`: the explicit opt-out. Every C2+ frame runs C1 once per row.
    Off,
}

impl NativeBatchSelector {
    pub fn parse(raw: Option<&str>) -> Result<Self, &'static str> {
        match raw {
            None => Ok(Self::Auto),
            Some("1") => Ok(Self::Required),
            Some("0") => Ok(Self::Off),
            Some(_) => Err("ATLAS_QWEN4_NATIVE_BATCH must be exact value 0 or 1, or unset"),
        }
    }

    pub fn from_env() -> Result<Self, &'static str> {
        match std::env::var(QWEN4_NATIVE_BATCH_ENV) {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("ATLAS_QWEN4_NATIVE_BATCH must be valid UTF-8")
            }
        }
    }

    pub const fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Required => "required",
            Self::Off => "off",
        }
    }
}

/// Why a Qwen4 row or frame decodes through the serial C1-per-row path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SerialReason {
    /// Process: `ATLAS_QWEN4_NATIVE_BATCH=0`.
    SelectorOff,
    /// Process: `--profile`; the native frame is deliberately unprofiled.
    Profiling,
    /// Process: the unified `_t_lanes` exact MoE row route is not selected
    /// (`ATLAS_MOE_T_LANES` unset, or a non-unified/hybrid MoE layout).
    MoeExactRowsUnavailable,
    /// Process: `ATLAS_SSM_H_FP16=1`; the native GDN lane keeps FP32 state.
    SsmHFp16,
    /// Process: `ATLAS_LM_HEAD_TC=1`; native logits use the exact LM-head route.
    LmHeadTensorCore,
    /// Process: no NVFP4 LM head, e.g. the `ATLAS_QWEN4_LMHEAD_BF16=1` control.
    LmHeadNotNvfp4,
    /// Request: prompt plus `max_tokens` can reach the QSA boundary and the
    /// frame does not admit QSA rows (`Qwen4BatchPlan::qsa_rows` false, or a
    /// length the per-row QSA route refuses).
    QsaRequired,
    /// Row: the next position reaches the QSA boundary without the request
    /// having been marked QSA-required (defensive guard).
    QsaBoundary,
    /// Frame: this row is eligible, but another row in the frame is not.
    CoScheduledWithSerialRow,
    // Capability: the native frame cannot run this width in this process.
    // Evaluated once per width by the model (`decode_qwen4_capability.rs`)
    // from every state-independent check of the native preflight, so the
    // native dispatch can fail afterwards only on an invariant or GPU error.
    /// `ATLAS_TARGET_LMHEAD_VOCAB` truncates the verify-side vocabulary.
    LmHeadVocabCap,
    /// FP32 logits; the native frame writes BF16 row-major logits.
    Fp32Logits,
    /// No exact (or K1-fallback) LM-head row tier for this width.
    LmHeadRowTier,
    /// Not the canonical BF16 Flash-Next geometry or layer schedule.
    ModelGeometry,
    /// TP/EP or a communicator (refused at load for C2+).
    Topology,
    /// A live speculative proposer (refused at load for C2+).
    TargetOnly,
    /// Prefix-cache ownership (refused at load for C2+).
    PrefixCache,
    /// Non-BF16 or high-speed-swap KV (refused at load for C2+).
    KvLayout,
    /// The final hyperconnection mixer cannot run this width.
    FinalMixer,
    /// Model-level native scratch is too small for this width.
    ArenaCapacity,
    /// The batched embedding kernels or scratch are unavailable.
    EmbedBatch,
    /// The native PLE path (boundary, kernels, scratch or slot pool).
    PleRows,
    /// An attention layer's native capability (kernels, exact tiers,
    /// selectors, hyperconnections or arenas) at this width.
    AttentionLayer,
    /// A recurrent layer's native capability at this width.
    SsmLayer,
    /// A layer kind with no native Qwen4 path.
    UnsupportedLayer,
}

impl SerialReason {
    /// Stable metric label and log token.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SelectorOff => "selector_off",
            Self::Profiling => "profiling",
            Self::MoeExactRowsUnavailable => "moe_exact_rows_unavailable",
            Self::SsmHFp16 => "ssm_h_fp16",
            Self::LmHeadTensorCore => "lm_head_tc",
            Self::LmHeadNotNvfp4 => "lm_head_not_nvfp4",
            Self::QsaRequired => "qsa_required",
            Self::QsaBoundary => "qsa_boundary",
            Self::CoScheduledWithSerialRow => "co_scheduled_with_serial_row",
            Self::LmHeadVocabCap => "lm_head_vocab_cap",
            Self::Fp32Logits => "fp32_logits",
            Self::LmHeadRowTier => "lm_head_row_tier",
            Self::ModelGeometry => "model_geometry",
            Self::Topology => "topology",
            Self::TargetOnly => "target_only",
            Self::PrefixCache => "prefix_cache",
            Self::KvLayout => "kv_layout",
            Self::FinalMixer => "final_mixer",
            Self::ArenaCapacity => "arena_capacity",
            Self::EmbedBatch => "embed_batch",
            Self::PleRows => "ple_rows",
            Self::AttentionLayer => "attention_layer",
            Self::SsmLayer => "ssm_layer",
            Self::UnsupportedLayer => "unsupported_layer",
        }
    }

    /// What the operator changes to get the native route back.
    pub const fn remedy(self) -> &'static str {
        match self {
            Self::SelectorOff => "unset ATLAS_QWEN4_NATIVE_BATCH",
            Self::Profiling => "drop --profile",
            Self::MoeExactRowsUnavailable => {
                "set ATLAS_MOE_T_LANES=1 with the unified MoE layout (ATLAS_UNIFIED_MOE_LAYOUT unset or 1, ATLAS_HYBRID_MOE_LAYOUT unset or 0); the QWEN4_BATCH_PLAN detail names a failing layer check"
            }
            Self::SsmHFp16 => "unset ATLAS_SSM_H_FP16",
            Self::LmHeadTensorCore => "unset ATLAS_LM_HEAD_TC",
            Self::LmHeadNotNvfp4 => "unset ATLAS_QWEN4_LMHEAD_BF16",
            Self::QsaRequired => {
                "send max_tokens so prompt + max_tokens < 2048, or enable native QSA rows"
            }
            Self::QsaBoundary => "keep prompt + max_tokens below 2048",
            Self::CoScheduledWithSerialRow => "none; the frame follows its serial row",
            Self::LmHeadVocabCap => "unset ATLAS_TARGET_LMHEAD_VOCAB (or 0)",
            Self::Fp32Logits => "keep BF16 logits",
            Self::LmHeadRowTier => {
                "use a build with the exact LM-head tier for this width, or a narrower --max-batch-size"
            }
            Self::ModelGeometry => "serve the canonical Qwen3.8-Flash-Next checkpoint",
            Self::Topology => "serve TP1/EP1",
            Self::TargetOnly => "disable speculative decoding",
            Self::PrefixCache => "disable prefix caching",
            Self::KvLayout => "serve resident BF16 KV without high-speed swap",
            Self::FinalMixer => "use a build with the final-mixer tier for this width",
            Self::ArenaCapacity => "lower --max-batch-size or raise the decode arenas",
            Self::EmbedBatch => "use a build with the batched Qwen4 embedding kernels",
            Self::PleRows => {
                "use a build with the native PLE kernels and the canonical layer-2 PLE boundary"
            }
            Self::AttentionLayer => {
                "see the QWEN4_BATCH_PLAN detail; unset the named attention selector or use a build with that tier"
            }
            Self::SsmLayer => {
                "see the QWEN4_BATCH_PLAN detail; unset the named recurrent selector or use a build with that tier"
            }
            Self::UnsupportedLayer => {
                "none; this checkpoint has a layer kind without a native path"
            }
        }
    }
}

/// Why the native frame cannot run a given width: the typed reason plus the
/// failing check's message, which the load-time plan log prints once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeGap {
    pub reason: SerialReason,
    pub detail: String,
}

impl NativeGap {
    pub fn new(reason: SerialReason, detail: impl std::fmt::Display) -> Self {
        Self {
            reason,
            detail: detail.to_string(),
        }
    }

    /// The same gap, attributed to model layer `index`.
    pub fn at_layer(self, index: usize) -> Self {
        Self {
            detail: format!("layer {index}: {}", self.detail),
            ..self
        }
    }
}

/// Native capability of a native plan, per width, from the load-time probe.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeCapabilityReport {
    /// `(rows, gap)` for every width in `2..=min(max_batch, QWEN4_NATIVE_MAX_ROWS)` that the
    /// native frame cannot run. Those widths serialize with `gap.reason`.
    pub gaps: Vec<(usize, NativeGap)>,
}

/// The route of a frame the row policy admitted but the native frame cannot
/// run. `Required` fails closed, because it promised native or nothing (load
/// refuses every width with a gap, so this is a defect); otherwise the whole
/// frame serializes with the gap's reason.
pub fn native_gap_route(
    plan: &Qwen4BatchPlan,
    rows: usize,
    gap: &NativeGap,
) -> Result<FrameRoute, String> {
    if plan.selector == NativeBatchSelector::Required {
        return Err(format!(
            "ATLAS_QWEN4_NATIVE_BATCH=1: native C{rows} frame unavailable ({}): {}",
            gap.reason.label(),
            gap.detail
        ));
    }
    Ok(FrameRoute::Serial {
        frame: gap.reason,
        rows: vec![gap.reason; rows],
    })
}

/// Mirrors `model::impl_a3::target_lmhead_vocab` and `verify_lmhead_vocab`:
/// a parsed cap below `vocab_size` truncates; unset, unparsable, `0` or a cap
/// at or above the vocabulary keeps the full vocabulary.
pub fn target_lmhead_vocab_is_full(raw: Option<&str>, vocab_size: usize) -> bool {
    let cap = raw
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(0) as usize;
    cap == 0 || cap >= vocab_size
}

/// Process-wide facts. All of them are fixed for the life of the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessFacts {
    pub profile: bool,
    pub moe_exact_rows: bool,
    pub ssm_h_fp16: bool,
    pub lm_head_tc: bool,
    pub lm_head_nvfp4: bool,
    /// The verify-side LM head covers the full vocabulary.
    pub lm_head_full_vocab: bool,
    /// The QSA side index exists (`--qwen4-qsa`); QSA rows need it.
    pub qsa_index: bool,
}

impl ProcessFacts {
    /// Facts from the environment and two load facts. The model replaces
    /// `lm_head_nvfp4` with whether it actually holds an NVFP4 head.
    pub fn from_env(is_qwen4: bool, profile: bool, vocab_size: usize, qsa_index: bool) -> Self {
        let var = |name: &str| std::env::var(name).ok();
        Self {
            profile,
            lm_head_full_vocab: target_lmhead_vocab_is_full(
                var("ATLAS_TARGET_LMHEAD_VOCAB").as_deref(),
                vocab_size,
            ),
            qsa_index,
            moe_exact_rows: moe_exact_rows_selected(
                var("ATLAS_MOE_T_LANES").as_deref(),
                var("ATLAS_UNIFIED_MOE_LAYOUT").as_deref(),
                var("ATLAS_HYBRID_MOE_LAYOUT").as_deref(),
                is_qwen4,
            ),
            ssm_h_fp16: var("ATLAS_SSM_H_FP16").as_deref() == Some("1"),
            lm_head_tc: var("ATLAS_LM_HEAD_TC").as_deref() == Some("1"),
            // Same answer as the factory's head selector (umbrella-aware); an
            // invalid value is refused at load, so treat it as NVFP4 here.
            lm_head_nvfp4: !crate::qwen4_native_selectors::selected(
                crate::qwen4_native_selectors::Knob::LmHead,
            )
            .unwrap_or(false),
        }
    }
}

/// Mirrors `moe::forward_phase::t_lanes_selected` and the layout parse in
/// `factory::m2_setup`. The layer preflight remains the authority on kernel
/// presence; this only names the configuration that rules the route out.
pub fn moe_exact_rows_selected(
    t_lanes: Option<&str>,
    unified: Option<&str>,
    hybrid: Option<&str>,
    is_qwen4: bool,
) -> bool {
    let truthy = |value: &str| value == "1" || value.eq_ignore_ascii_case("true");
    let unified = unified.map_or(is_qwen4, truthy);
    let hybrid = hybrid.is_some_and(truthy);
    t_lanes == Some("1") && unified && !hybrid
}

/// First process reason, in a fixed order so logs and metrics are stable.
pub fn process_serial_reason(facts: &ProcessFacts) -> Option<SerialReason> {
    [
        (facts.profile, SerialReason::Profiling),
        (!facts.moe_exact_rows, SerialReason::MoeExactRowsUnavailable),
        (facts.ssm_h_fp16, SerialReason::SsmHFp16),
        (facts.lm_head_tc, SerialReason::LmHeadTensorCore),
        (!facts.lm_head_nvfp4, SerialReason::LmHeadNotNvfp4),
        (!facts.lm_head_full_vocab, SerialReason::LmHeadVocabCap),
    ]
    .into_iter()
    .find_map(|(blocked, reason)| blocked.then_some(reason))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen4BatchPlan {
    pub selector: NativeBatchSelector,
    /// `None`: C2-C256 (`QWEN4_NATIVE_MAX_ROWS`) frames go native subject to the row policy.
    pub process_reason: Option<SerialReason>,
    /// QSA-required rows may join a native frame (`ATLAS_QWEN4_NATIVE_BATCH_QSA`
    /// on). Only ever true on a native plan.
    pub qsa_rows: bool,
}

/// Whether the operator asked for QSA rows in the native frame
/// (`ATLAS_QWEN4_NATIVE_BATCH_QSA=1`, strictly parsed by
/// `layers::qwen4_qsa::native_batch`).
pub fn qsa_rows_requested_from_env() -> anyhow::Result<bool> {
    use crate::layers::qwen4_qsa::native_batch::{NativeQsaMode, native_batch_qsa_mode};
    Ok(native_batch_qsa_mode()? == NativeQsaMode::PerRow)
}

impl Qwen4BatchPlan {
    pub const fn native(&self) -> bool {
        self.process_reason.is_none()
    }
}

/// `qsa_rows_requested` without a native plan is a configuration error, never
/// a silent no-op: the per-row QSA route lives inside the native frame.
pub fn resolve_plan(
    selector: NativeBatchSelector,
    facts: &ProcessFacts,
    qsa_rows_requested: bool,
) -> Result<Qwen4BatchPlan, String> {
    let process_reason = match selector {
        NativeBatchSelector::Off => Some(SerialReason::SelectorOff),
        NativeBatchSelector::Auto => process_serial_reason(facts),
        NativeBatchSelector::Required => match process_serial_reason(facts) {
            None => None,
            Some(reason) => {
                return Err(format!(
                    "ATLAS_QWEN4_NATIVE_BATCH=1 refused: native Qwen4 batching is unavailable ({}); {}, or unset the selector to serve serially",
                    reason.label(),
                    reason.remedy()
                ));
            }
        },
    };
    if let (true, Some(reason)) = (qsa_rows_requested, process_reason) {
        return Err(format!(
            "ATLAS_QWEN4_NATIVE_BATCH_QSA=1 requires native Qwen4 batching, which is unavailable ({}); {}, or unset ATLAS_QWEN4_NATIVE_BATCH_QSA",
            reason.label(),
            reason.remedy()
        ));
    }
    if qsa_rows_requested && !facts.qsa_index {
        return Err(
            "ATLAS_QWEN4_NATIVE_BATCH_QSA=1 requires the QSA side index; serve with --qwen4-qsa, or unset ATLAS_QWEN4_NATIVE_BATCH_QSA"
                .to_string(),
        );
    }
    Ok(Qwen4BatchPlan {
        selector,
        process_reason,
        qsa_rows: qsa_rows_requested,
    })
}

/// The one Qwen4 concurrency ceiling rule, shared by the server and the
/// launcher (`repro-kit/flux-autodetect`): `1..=QWEN4_NATIVE_MAX_ROWS` (C1-C256)
/// while native batching is enabled, and up to the serial ceiling only after
/// the explicit opt-out.
pub fn admit_width(
    selector: NativeBatchSelector,
    max_batch_size: usize,
    serial_ceiling: usize,
) -> Result<(), String> {
    let ceiling = if selector.enabled() {
        QWEN4_NATIVE_MAX_ROWS
    } else {
        serial_ceiling
    };
    if max_batch_size <= ceiling {
        return Ok(());
    }
    Err(if selector.enabled() {
        format!(
            "Qwen4 --max-batch-size {max_batch_size} exceeds the native batching ceiling C{QWEN4_NATIVE_MAX_ROWS}; \
             lower it, or set ATLAS_QWEN4_NATIVE_BATCH=0 to serve up to C{serial_ceiling} serially"
        )
    } else {
        format!("Qwen4 concurrency {max_batch_size} exceeds the admitted C{serial_ceiling} ceiling")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowFacts {
    pub qsa_required: bool,
    pub seq_len: usize,
}

/// A row's own reason, independent of the rest of its frame. With
/// `qsa_rows`, `None` exactly where the per-row QSA route
/// (`layers::qwen4_qsa::native_batch::native_attn_route`) admits the row.
pub fn row_serial_reason(row: RowFacts, qsa_rows: bool) -> Option<SerialReason> {
    let Some(next) = row.seq_len.checked_add(1) else {
        return Some(if row.qsa_required {
            SerialReason::QsaRequired
        } else {
            SerialReason::QsaBoundary
        });
    };
    if row.qsa_required {
        let admitted = qsa_rows && u32::try_from(next).is_ok();
        return (!admitted).then_some(SerialReason::QsaRequired);
    }
    (next >= QWEN4_QSA_DENSE_LIMIT).then_some(SerialReason::QsaBoundary)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameRoute {
    /// One live row: ordinary C1 decode, not a batching decision.
    Single,
    /// One native C2-C256 (`QWEN4_NATIVE_MAX_ROWS`) model traversal.
    Native,
    /// C1 once per row. `frame` is what serialized the frame; `rows[i]` is
    /// row `i`'s own reason, or `CoScheduledWithSerialRow`.
    Serial {
        frame: SerialReason,
        rows: Vec<SerialReason>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteError {
    /// Load admission (`admit_width`) makes this unreachable.
    WidthAboveNativeMax { rows: usize },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WidthAboveNativeMax { rows } => write!(
                f,
                "native Qwen4 batching supports active widths C2-C{QWEN4_NATIVE_MAX_ROWS} only, got C{rows}"
            ),
        }
    }
}

impl std::error::Error for RouteError {}

pub fn route_frame(plan: &Qwen4BatchPlan, rows: &[RowFacts]) -> Result<FrameRoute, RouteError> {
    let n = rows.len();
    if n <= 1 {
        return Ok(FrameRoute::Single);
    }
    if let Some(reason) = plan.process_reason {
        return Ok(FrameRoute::Serial {
            frame: reason,
            rows: vec![reason; n],
        });
    }
    if n > QWEN4_NATIVE_MAX_ROWS {
        return Err(RouteError::WidthAboveNativeMax { rows: n });
    }
    let own: Vec<Option<SerialReason>> = rows
        .iter()
        .map(|&row| row_serial_reason(row, plan.qsa_rows))
        .collect();
    // QsaRequired outranks QsaBoundary so a frame's label names the
    // immutable admission fact when both are present.
    let frame = own
        .iter()
        .flatten()
        .copied()
        .min_by_key(|reason| match reason {
            SerialReason::QsaRequired => 0,
            _ => 1,
        });
    Ok(match frame {
        None => FrameRoute::Native,
        Some(frame) => FrameRoute::Serial {
            frame,
            rows: own
                .into_iter()
                .map(|reason| reason.unwrap_or(SerialReason::CoScheduledWithSerialRow))
                .collect(),
        },
    })
}

/// The intrinsic route of a newly admitted request: what it would do in a
/// frame of other eligible rows. Logged and counted once per request.
pub fn request_serial_reason(plan: &Qwen4BatchPlan, qsa_required: bool) -> Option<SerialReason> {
    plan.process_reason
        .or((qsa_required && !plan.qsa_rows).then_some(SerialReason::QsaRequired))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_facts() -> ProcessFacts {
        ProcessFacts {
            profile: false,
            moe_exact_rows: true,
            ssm_h_fp16: false,
            lm_head_tc: false,
            lm_head_nvfp4: true,
            lm_head_full_vocab: true,
            qsa_index: true,
        }
    }

    fn short(n: usize) -> Vec<RowFacts> {
        vec![
            RowFacts {
                qsa_required: false,
                seq_len: 17,
            };
            n
        ]
    }

    #[test]
    fn selector_is_exact_and_default_auto() {
        assert_eq!(
            NativeBatchSelector::parse(None),
            Ok(NativeBatchSelector::Auto)
        );
        assert_eq!(
            NativeBatchSelector::parse(Some("1")),
            Ok(NativeBatchSelector::Required)
        );
        assert_eq!(
            NativeBatchSelector::parse(Some("0")),
            Ok(NativeBatchSelector::Off)
        );
        for hostile in ["", "true", "01", "2", " 1", "auto"] {
            assert!(
                NativeBatchSelector::parse(Some(hostile)).is_err(),
                "{hostile:?}"
            );
        }
    }

    #[test]
    fn auto_degrades_each_process_fact_to_a_typed_serial_plan() {
        let cases = [
            (
                ProcessFacts {
                    profile: true,
                    ..native_facts()
                },
                SerialReason::Profiling,
            ),
            (
                ProcessFacts {
                    moe_exact_rows: false,
                    ..native_facts()
                },
                SerialReason::MoeExactRowsUnavailable,
            ),
            (
                ProcessFacts {
                    ssm_h_fp16: true,
                    ..native_facts()
                },
                SerialReason::SsmHFp16,
            ),
            (
                ProcessFacts {
                    lm_head_tc: true,
                    ..native_facts()
                },
                SerialReason::LmHeadTensorCore,
            ),
            (
                ProcessFacts {
                    lm_head_nvfp4: false,
                    ..native_facts()
                },
                SerialReason::LmHeadNotNvfp4,
            ),
            (
                ProcessFacts {
                    lm_head_full_vocab: false,
                    ..native_facts()
                },
                SerialReason::LmHeadVocabCap,
            ),
        ];
        for (facts, reason) in cases {
            let plan = resolve_plan(NativeBatchSelector::Auto, &facts, false).unwrap();
            assert_eq!(plan.process_reason, Some(reason));
            let err = resolve_plan(NativeBatchSelector::Required, &facts, false).unwrap_err();
            assert!(err.contains(reason.label()), "{err}");
        }
        assert!(
            resolve_plan(NativeBatchSelector::Auto, &native_facts(), false)
                .unwrap()
                .native()
        );
        assert!(
            resolve_plan(NativeBatchSelector::Required, &native_facts(), false)
                .unwrap()
                .native()
        );
        assert_eq!(
            resolve_plan(NativeBatchSelector::Off, &native_facts(), false)
                .unwrap()
                .process_reason,
            Some(SerialReason::SelectorOff)
        );
    }

    #[test]
    fn moe_route_mirrors_the_t_lanes_and_layout_parses() {
        assert!(moe_exact_rows_selected(Some("1"), None, None, true));
        assert!(!moe_exact_rows_selected(None, None, None, true));
        assert!(!moe_exact_rows_selected(Some("true"), None, None, true));
        assert!(!moe_exact_rows_selected(Some("1"), None, None, false));
        assert!(moe_exact_rows_selected(
            Some("1"),
            Some("TRUE"),
            None,
            false
        ));
        assert!(!moe_exact_rows_selected(Some("1"), Some("0"), None, true));
        assert!(!moe_exact_rows_selected(Some("1"), None, Some("1"), true));
    }

    #[test]
    fn width_ceiling_is_native_max_unless_opted_out() {
        for selector in [NativeBatchSelector::Auto, NativeBatchSelector::Required] {
            assert!(admit_width(selector, QWEN4_NATIVE_MAX_ROWS, 512).is_ok());
            assert!(
                admit_width(selector, QWEN4_NATIVE_MAX_ROWS + 1, 512)
                    .unwrap_err()
                    .contains("ATLAS_QWEN4_NATIVE_BATCH=0")
            );
        }
        assert!(admit_width(NativeBatchSelector::Off, 128, 128).is_ok());
        assert!(
            admit_width(NativeBatchSelector::Off, 129, 128)
                .unwrap_err()
                .contains("C128")
        );
    }

    #[test]
    fn short_frames_are_native_up_to_native_max() {
        let plan = resolve_plan(NativeBatchSelector::Auto, &native_facts(), false).unwrap();
        assert_eq!(route_frame(&plan, &short(1)), Ok(FrameRoute::Single));
        for n in 2..=QWEN4_NATIVE_MAX_ROWS {
            assert_eq!(
                route_frame(&plan, &short(n)),
                Ok(FrameRoute::Native),
                "C{n}"
            );
        }
        let over_max = QWEN4_NATIVE_MAX_ROWS + 1;
        assert_eq!(
            route_frame(&plan, &short(over_max)),
            Err(RouteError::WidthAboveNativeMax { rows: over_max })
        );
    }

    #[test]
    fn one_qsa_row_serializes_the_frame_with_per_row_reasons() {
        let plan = resolve_plan(NativeBatchSelector::Auto, &native_facts(), false).unwrap();
        let mut rows = short(3);
        rows[1].qsa_required = true;
        rows[2].seq_len = QWEN4_QSA_DENSE_LIMIT - 1;
        assert_eq!(
            route_frame(&plan, &rows),
            Ok(FrameRoute::Serial {
                frame: SerialReason::QsaRequired,
                rows: vec![
                    SerialReason::CoScheduledWithSerialRow,
                    SerialReason::QsaRequired,
                    SerialReason::QsaBoundary,
                ],
            })
        );
        assert_eq!(
            row_serial_reason(
                RowFacts {
                    qsa_required: false,
                    seq_len: 2_046
                },
                false
            ),
            None
        );
        assert_eq!(
            row_serial_reason(
                RowFacts {
                    qsa_required: false,
                    seq_len: usize::MAX
                },
                false
            ),
            Some(SerialReason::QsaBoundary)
        );
    }

    #[test]
    fn qsa_rows_join_native_frames_only_on_a_native_plan() {
        let plan = resolve_plan(NativeBatchSelector::Auto, &native_facts(), true).unwrap();
        assert!(plan.qsa_rows);
        let mut rows = short(3);
        rows[1] = RowFacts {
            qsa_required: true,
            seq_len: 9_000,
        };
        assert_eq!(route_frame(&plan, &rows), Ok(FrameRoute::Native));
        assert_eq!(request_serial_reason(&plan, true), None);
        // The per-row route still refuses a non-QSA row at the boundary and a
        // QSA row past u32.
        rows[2].seq_len = QWEN4_QSA_DENSE_LIMIT - 1;
        assert!(matches!(
            route_frame(&plan, &rows),
            Ok(FrameRoute::Serial { .. })
        ));
        let huge = RowFacts {
            qsa_required: true,
            seq_len: u32::MAX as usize,
        };
        assert_eq!(
            row_serial_reason(huge, true),
            Some(SerialReason::QsaRequired)
        );
        // QSA rows without a native plan are refused, never ignored.
        let no_t_lanes = ProcessFacts {
            moe_exact_rows: false,
            ..native_facts()
        };
        for selector in [NativeBatchSelector::Auto, NativeBatchSelector::Off] {
            let err = resolve_plan(selector, &no_t_lanes, true).unwrap_err();
            assert!(err.contains("ATLAS_QWEN4_NATIVE_BATCH_QSA"), "{err}");
        }
        // QSA rows without the side index (`--qwen4-qsa`) are refused for
        // every selector, even on a native plan.
        let no_index = ProcessFacts {
            qsa_index: false,
            ..native_facts()
        };
        for selector in [NativeBatchSelector::Auto, NativeBatchSelector::Required] {
            let err = resolve_plan(selector, &no_index, true).unwrap_err();
            assert!(err.contains("--qwen4-qsa"), "{err}");
            assert!(resolve_plan(selector, &no_index, false).unwrap().native());
        }
    }

    #[test]
    fn target_vocab_cap_mirrors_the_verify_side_parse() {
        let vocab = 151_936;
        for (raw, full) in [
            (None, true),
            (Some("0"), true),
            (Some(" 0 "), true),
            (Some("garbage"), true),
            (Some(""), true),
            (Some("151936"), true),
            (Some("200000"), true),
            (Some("96000"), false),
            (Some(" 96000"), false),
        ] {
            assert_eq!(target_lmhead_vocab_is_full(raw, vocab), full, "{raw:?}");
        }
    }

    #[test]
    fn a_native_gap_serializes_by_default_and_fails_closed_when_required() {
        let gap = NativeGap::new(SerialReason::LmHeadRowTier, "no tier").at_layer(7);
        assert_eq!(gap.detail, "layer 7: no tier");
        let auto = resolve_plan(NativeBatchSelector::Auto, &native_facts(), false).unwrap();
        assert_eq!(
            native_gap_route(&auto, 3, &gap),
            Ok(FrameRoute::Serial {
                frame: SerialReason::LmHeadRowTier,
                rows: vec![SerialReason::LmHeadRowTier; 3],
            })
        );
        let required = resolve_plan(NativeBatchSelector::Required, &native_facts(), false).unwrap();
        let err = native_gap_route(&required, 3, &gap).unwrap_err();
        assert!(
            err.contains("lm_head_row_tier") && err.contains("layer 7"),
            "{err}"
        );
    }

    #[test]
    fn process_reasons_serialize_every_width_without_error() {
        let plan = resolve_plan(NativeBatchSelector::Off, &native_facts(), false).unwrap();
        for n in [2, 32, 33, 128] {
            match route_frame(&plan, &short(n)).unwrap() {
                FrameRoute::Serial { frame, rows } => {
                    assert_eq!(frame, SerialReason::SelectorOff);
                    assert_eq!(rows, vec![SerialReason::SelectorOff; n]);
                }
                other => panic!("C{n}: {other:?}"),
            }
        }
    }

    #[test]
    fn request_reason_prefers_the_process_reason() {
        let native = resolve_plan(NativeBatchSelector::Auto, &native_facts(), false).unwrap();
        assert_eq!(request_serial_reason(&native, false), None);
        assert_eq!(
            request_serial_reason(&native, true),
            Some(SerialReason::QsaRequired)
        );
        let off = resolve_plan(NativeBatchSelector::Off, &native_facts(), false).unwrap();
        assert_eq!(
            request_serial_reason(&off, true),
            Some(SerialReason::SelectorOff)
        );
    }

    #[test]
    fn labels_are_unique() {
        let all = [
            SerialReason::SelectorOff,
            SerialReason::Profiling,
            SerialReason::MoeExactRowsUnavailable,
            SerialReason::SsmHFp16,
            SerialReason::LmHeadTensorCore,
            SerialReason::LmHeadNotNvfp4,
            SerialReason::QsaRequired,
            SerialReason::QsaBoundary,
            SerialReason::CoScheduledWithSerialRow,
            SerialReason::LmHeadVocabCap,
            SerialReason::Fp32Logits,
            SerialReason::LmHeadRowTier,
            SerialReason::ModelGeometry,
            SerialReason::Topology,
            SerialReason::TargetOnly,
            SerialReason::PrefixCache,
            SerialReason::KvLayout,
            SerialReason::FinalMixer,
            SerialReason::ArenaCapacity,
            SerialReason::EmbedBatch,
            SerialReason::PleRows,
            SerialReason::AttentionLayer,
            SerialReason::SsmLayer,
            SerialReason::UnsupportedLayer,
        ];
        let labels: std::collections::HashSet<_> = all.iter().map(|r| r.label()).collect();
        assert_eq!(labels.len(), all.len());
    }
}
