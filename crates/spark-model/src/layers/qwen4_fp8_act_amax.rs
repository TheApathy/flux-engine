// SPDX-License-Identifier: AGPL-3.0-only

//! Producer-side absmax for the FP8 tensor-scale projections
//! (`ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED=1`, default off).
//!
//! A tensor-scale FP8 projection (`qwen4_fast_proj::fp8_gemm_ex`,
//! `fp8_gemm_shared_act`) first reads its whole BF16 activation for the
//! absmax (`qwen4_fp8_act_amax`), then again to write e4m3 rows. When the
//! kernel that writes that activation (the HC_FUSED up-mix,
//! `qwen4_hc_upmix_hcfused_amax`) reduces the same values into a device slot
//! with the same rule (fmaxf of fabsf, atomicMax on the float bits), the
//! projection skips its absmax pass and quantizes from the slot. The max is
//! order-independent, so the e4m3 rows and the scale are byte-identical.
//!
//! Hand-off: a call site that runs a preparation and then its tensor-scale
//! projection holds an [`Arm`] across both. The arm is one-shot: the first
//! preparation inside it takes it ([`take_armed`]), zeroes the slot, reduces
//! into it and records what it covered ([`note_ready`]); the projection takes
//! the record only if it reads exactly that activation (pointer, rows,
//! columns, stream) and otherwise runs its own absmax pass. Every unarmed
//! preparation clears the record and dropping the arm clears it too, so a
//! record never outlives the call site that asked for it. The slot is one
//! device word: producer and consumer run back to back on one stream.

use std::cell::Cell;
use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED";
pub(crate) const MHC_AMAX_ALIAS: &str = "ATLAS_QWEN4_PREFILL_MHC_FUSED_AMAX";
/// `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED` (default off): the same hand-off from the E4M3 QSA
/// prefill core (`qwen4_qsa_prefill_attn_fp8_{qk,qkpv}_amax`) to the attention o_proj.
pub(crate) const OPROJ_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED";
/// `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED_TRIGGER=<path>` (test only, default unset): with the
/// selector on, the o_proj hand-off applies only while that file exists (checked once per attention
/// call site), so one load can serve an off/on/off A/B/A. Exact in either state.
pub(crate) const OPROJ_TRIGGER: &str = "ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED_TRIGGER";

/// Which kernel reduced the slot (receipts only; the hand-off rule is the same).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Producer {
    /// The HC_FUSED up-mix (`qwen4_hc_upmix_hcfused_amax`), Q1.
    HcUpmix,
    /// The E4M3 QSA prefill core's gated-output epilogue, for the attention o_proj.
    QsaCore,
    /// The GDN gated RMS norm (`qwen4_gated_rms_norm_{bf16,f16}in_amax[_v2]`), for the SSM out_proj.
    GdnNorm,
}

impl Producer {
    /// The selector that enables this producer (named in receipts and errors).
    pub(crate) fn selector(self) -> &'static str {
        match self {
            Producer::HcUpmix | Producer::GdnNorm => SELECTOR,
            Producer::QsaCore => OPROJ_SELECTOR,
        }
    }
}

/// What a producer reduced into the slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ready {
    pub act: u64,
    pub rows: usize,
    pub cols: usize,
    pub stream: u64,
    pub producer: Producer,
}

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static READY: Cell<Option<Ready>> = const { Cell::new(None) };
    static OPROJ_TALLY: Cell<OprojTally> = const { Cell::new(OprojTally::ZERO) };
}

/// Per-chunk count of the o_proj hand-off (`ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED`), so a gate can
/// check engagement on every chunk of every request instead of trusting a once-per-process line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OprojTally {
    /// o_proj quantised from the QSA core's slot (its absmax pass skipped).
    pub hits: u32,
    /// selected, but the core that wrote raw_outputs was not the E4M3 QSA core.
    pub refused: u32,
    /// the core reduced, but the next tensor-scale projection was not its o_proj.
    pub missed: u32,
}

impl OprojTally {
    pub(crate) const ZERO: Self = Self {
        hits: 0,
        refused: 0,
        missed: 0,
    };
}

fn tally(f: impl FnOnce(&mut OprojTally)) {
    OPROJ_TALLY.with(|c| {
        let mut t = c.get();
        f(&mut t);
        c.set(t);
    });
}

/// Start of a prefill chunk's layer loop: forget counts left by anything outside the loop.
pub(crate) fn oproj_chunk_begin() {
    OPROJ_TALLY.with(|c| c.set(OprojTally::ZERO));
}

/// The o_proj call site saw the selector on but no QSA-core record (dense-window chunk).
pub(crate) fn note_oproj_refused() {
    tally(|t| t.refused += 1);
}

/// End of a prefill chunk's layer loop: one `OPROJ_AMAX_FUSED_CHUNK` line per chunk while the
/// selector is on (nothing otherwise), then the counts restart.
pub(crate) fn oproj_chunk_receipt(start: usize, rows: usize) -> OprojTally {
    let t = OPROJ_TALLY.with(|c| c.replace(OprojTally::ZERO));
    if matches!(oproj_selected(), Ok(true)) {
        tracing::info!(
            "OPROJ_AMAX_FUSED_CHUNK selector={OPROJ_SELECTOR} start={start} rows={rows} hits={} refused={} missed={}",
            t.hits,
            t.refused,
            t.missed
        );
    }
    t
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Selection {
    pub(crate) hc_upmix: bool,
    pub(crate) gdn_norm: bool,
}

fn parse_selection_named(name: &str, value: Option<&str>) -> Result<Selection, String> {
    let mut sel = Selection::default();
    let Some(value) = value else {
        return Ok(sel);
    };
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return Ok(sel);
    }
    if value == "1" || value == "all" {
        return Ok(Selection {
            hc_upmix: true,
            gdn_norm: true,
        });
    }
    for item in value.split(',') {
        match item.trim() {
            "" | "0" => {}
            "1" | "all" => {
                sel.hc_upmix = true;
                sel.gdn_norm = true;
            }
            "hc_upmix" => sel.hc_upmix = true,
            "gdn_norm" => sel.gdn_norm = true,
            other => {
                return Err(format!(
                    "{name} must be absent, 0, 1, all, hc_upmix, or gdn_norm; got {other:?}"
                ));
            }
        }
    }
    Ok(sel)
}

pub(crate) fn parse_selector(
    primary: Option<&str>,
    alias: Option<&str>,
) -> Result<Selection, String> {
    let p = primary.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    let a = alias.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    });
    if let Some(p) = p {
        parse_selection_named(SELECTOR, Some(p))
    } else if let Some(a) = a {
        parse_selection_named(MHC_AMAX_ALIAS, Some(a))
    } else {
        Ok(Selection::default())
    }
}

#[allow(dead_code)]
pub(crate) fn parse_selection(value: Option<&str>) -> Result<Selection, String> {
    parse_selector(value, None)
}

fn selection() -> Result<Selection> {
    static SEL: OnceLock<Result<Selection, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        parse_selector(
            std::env::var(SELECTOR).ok().as_deref(),
            std::env::var(MHC_AMAX_ALIAS).ok().as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// `ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED`: absent/0 off, 1/all/hc_upmix on for HC_FUSED up-mix.
pub(crate) fn selected() -> Result<bool> {
    Ok(selection()?.hc_upmix)
}

/// `ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED`: absent/0 off, 1/all/gdn_norm on for GDN norm.
pub(crate) fn gdn_norm_selected() -> Result<bool> {
    Ok(selection()?.gdn_norm)
}

#[cfg(test)]
fn parse(value: Option<&str>) -> Result<bool, String> {
    parse_selector(value, None).map(|s| s.hc_upmix)
}

fn parse_named(name: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{name} must be absent, 0 or 1; got {other:?}")),
    }
}

/// `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED`: absent/0 off, 1 on; anything else fails the first
/// prefill.
pub(crate) fn oproj_selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        parse_named(
            OPROJ_SELECTOR,
            std::env::var(OPROJ_SELECTOR).ok().as_deref(),
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// The trigger path, if set (non-empty).
fn oproj_trigger() -> Option<&'static std::path::Path> {
    static T: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    T.get_or_init(|| {
        std::env::var_os(OPROJ_TRIGGER)
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
    })
    .as_deref()
}

/// Whether the o_proj hand-off applies to the attention call starting now: the selector, and the
/// trigger file when [`OPROJ_TRIGGER`] is set. Checked once per call site (the arm carries the
/// decision to the core), so a toggle between calls never splits one hand-off.
pub(crate) fn oproj_active() -> Result<bool> {
    Ok(oproj_selected()? && oproj_trigger().is_none_or(std::path::Path::exists))
}

/// The 4-byte device slot producers reduce into (allocated once).
pub(crate) fn slot(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    static SLOT: OnceLock<Result<DevicePtr, String>> = OnceLock::new();
    SLOT.get_or_init(|| gpu.alloc(256).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| {
            anyhow::anyhow!("{SELECTOR}/{OPROJ_SELECTOR}: act-amax slot alloc failed: {e}")
        })
}

/// Scoped, one-shot request: the next preparation reduces its absmax for the
/// tensor-scale projection the call site runs on its output.
pub(crate) struct Arm(bool);

impl Arm {
    /// Arms only when [`selected`]; `consumer_reads_tensor_fp8` is the call
    /// site's statement that the preparation's output feeds a tensor-scale
    /// FP8 projection before the arm is dropped.
    pub(crate) fn enter(consumer_reads_tensor_fp8: bool) -> Result<Self> {
        let on = consumer_reads_tensor_fp8 && selected()?;
        Ok(Self(ARMED.with(|a| a.replace(on))))
    }

    /// Arms iff `on`: the call site has already checked its own selector (o_proj:
    /// [`oproj_selected`]) and that its consumer reads a tensor-scale FP8 projection.
    pub(crate) fn enter_selected(on: bool) -> Self {
        Self(ARMED.with(|a| a.replace(on)))
    }
}

/// Whether an arm is set and no producer has taken it yet.
pub(crate) fn is_armed() -> bool {
    ARMED.with(Cell::get)
}

impl Drop for Arm {
    fn drop(&mut self) {
        ARMED.with(|a| a.set(self.0));
        clear();
    }
}

/// Producer side: whether this preparation must reduce its absmax; consumes
/// the arm, so later preparations in the same scope run unarmed.
pub(crate) fn take_armed() -> bool {
    ARMED.with(|a| a.replace(false))
}

/// Producer side: the slot now holds the absmax of `ready`.
pub(crate) fn note_ready(ready: Ready) {
    READY.with(|r| r.set(Some(ready)));
}

/// Producer side: an unarmed preparation invalidates any earlier record.
pub(crate) fn clear() {
    READY.with(|r| r.set(None));
}

/// Consumer side (non-consuming): true when the slot holds the absmax of
/// exactly this activation on this stream.
#[allow(dead_code)]
pub(crate) fn has_matching(act: DevicePtr, rows: usize, cols: usize, stream: u64) -> bool {
    READY.with(Cell::get).is_some_and(|ready| {
        ready
            == Ready {
                act: act.0,
                rows,
                cols,
                stream,
                producer: ready.producer,
            }
    })
}

/// Consumer side: true (and the record consumed) when the slot holds the
/// absmax of exactly this activation on this stream.
pub(crate) fn take_matching(act: DevicePtr, rows: usize, cols: usize, stream: u64) -> bool {
    let Some(ready) = READY.with(Cell::take) else {
        return false;
    };
    let hit = ready
        == Ready {
            act: act.0,
            rows,
            cols,
            stream,
            producer: ready.producer,
        };
    if ready.producer == Producer::QsaCore {
        tally(|t| if hit { t.hits += 1 } else { t.missed += 1 });
    }
    if hit {
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        static OPROJ_ENGAGED: std::sync::Once = std::sync::Once::new();
        static GDN_NORM_ENGAGED: std::sync::Once = std::sync::Once::new();
        match ready.producer {
            Producer::HcUpmix => ENGAGED.call_once(|| {
                tracing::info!(
                    "FP8_ACT_AMAX_FUSED_ENGAGED selector={SELECTOR} producer=hc_upmix consumer=fp8_tensor rows={rows} cols={cols}"
                );
            }),
            Producer::QsaCore => OPROJ_ENGAGED.call_once(|| {
                tracing::info!(
                    "OPROJ_AMAX_FUSED_ENGAGED selector={OPROJ_SELECTOR} producer=qsa_e4m3_core consumer=fp8_tensor_o_proj rows={rows} cols={cols} skipped=qwen4_fp8_act_amax"
                );
            }),
            Producer::GdnNorm => GDN_NORM_ENGAGED.call_once(|| {
                tracing::info!(
                    "FP8_ACT_AMAX_FUSED_ENGAGED selector={SELECTOR} producer=gdn_norm consumer=fp8_tensor_ssm_out_proj rows={rows} cols={cols} skipped=qwen4_fp8_act_amax"
                );
            }),
        }
    } else {
        // The preparation reduced for a projection that did not come next:
        // correct (this projection runs its own absmax) but wasted work.
        static MISSED: std::sync::Once = std::sync::Once::new();
        MISSED.call_once(|| {
            tracing::warn!(
                "FP8_ACT_AMAX_FUSED_MISS selector={} ready={ready:?} consumer_rows={rows} consumer_cols={cols}",
                ready.producer.selector()
            );
        });
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACT: u64 = 0x1000;
    const STREAM: u64 = 7;

    fn ready() -> Ready {
        Ready {
            act: ACT,
            rows: 8192,
            cols: 2560,
            stream: STREAM,
            producer: Producer::HcUpmix,
        }
    }

    #[test]
    fn parses_selector() {
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("0")), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert!(parse(Some("on")).is_err());
    }

    #[test]
    fn test_selector_alias_and_precedence() {
        // Default-off: neither set
        assert_eq!(
            parse_selector(None, None),
            Ok(Selection {
                hc_upmix: false,
                gdn_norm: false,
            })
        );
        assert_eq!(
            parse_selector(Some(""), Some("")),
            Ok(Selection {
                hc_upmix: false,
                gdn_norm: false,
            })
        );

        // Primary activation ("1" and "hc_upmix")
        assert_eq!(
            parse_selector(Some("1"), None),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: true,
            })
        );
        assert_eq!(
            parse_selector(Some("hc_upmix"), None),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: false,
            })
        );

        // Alias activation ("1" and "hc_upmix")
        assert_eq!(
            parse_selector(None, Some("1")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: true,
            })
        );
        assert_eq!(
            parse_selector(Some(""), Some("1")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: true,
            })
        );
        assert_eq!(
            parse_selector(None, Some("hc_upmix")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: false,
            })
        );

        // Precedence when both are set (primary wins)
        assert_eq!(
            parse_selector(Some("1"), Some("0")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: true,
            })
        );
        assert_eq!(
            parse_selector(Some("0"), Some("1")),
            Ok(Selection {
                hc_upmix: false,
                gdn_norm: false,
            })
        );
        assert_eq!(
            parse_selector(Some("hc_upmix"), Some("0")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: false,
            })
        );
        assert_eq!(
            parse_selector(Some("1"), Some("invalid")),
            Ok(Selection {
                hc_upmix: true,
                gdn_norm: true,
            })
        );

        // Rejection of invalid values
        assert!(parse_selector(Some("invalid"), None).is_err());
        assert!(parse_selector(Some("on"), None).is_err());
        assert!(parse_selector(None, Some("invalid")).is_err());
        assert!(parse_selector(None, Some("on")).is_err());
        assert!(parse_selector(Some("invalid"), Some("1")).is_err());
    }

    #[test]
    fn a_record_serves_only_the_exact_activation_once() {
        let act = DevicePtr(ACT);
        note_ready(ready());
        assert!(!take_matching(act, 8192, 2048, STREAM), "other shape");
        // A mismatch consumes the record: nothing stale survives.
        assert!(!take_matching(act, 8192, 2560, STREAM));
        note_ready(ready());
        assert!(!take_matching(act, 8192, 2560, STREAM + 1), "other stream");
        note_ready(ready());
        assert!(take_matching(act, 8192, 2560, STREAM));
        assert!(!take_matching(act, 8192, 2560, STREAM), "consumed once");
        note_ready(ready());
        clear();
        assert!(
            !take_matching(act, 8192, 2560, STREAM),
            "cleared by an unarmed producer"
        );
    }

    #[test]
    fn arm_is_scoped_one_shot_and_needs_a_tensor_fp8_consumer() {
        assert!(!take_armed());
        {
            let _off = Arm::enter(false).unwrap();
            assert!(!take_armed());
        }
        // Armed by hand (the selector is process-wide): the first preparation
        // takes the arm, the next one runs unarmed.
        ARMED.with(|a| a.set(true));
        assert!(take_armed());
        assert!(!take_armed());
    }

    #[test]
    fn a_qsa_core_record_serves_the_o_proj_shape_only() {
        let act = DevicePtr(ACT);
        note_ready(Ready {
            act: ACT,
            rows: 8192,
            cols: 6144,
            stream: STREAM,
            producer: Producer::QsaCore,
        });
        assert!(take_matching(act, 8192, 6144, STREAM));
        note_ready(Ready {
            act: ACT,
            rows: 8192,
            cols: 6144,
            stream: STREAM,
            producer: Producer::QsaCore,
        });
        assert!(!take_matching(act, 8192, 2560, STREAM), "other projection");
    }

    #[test]
    fn selected_arm_follows_the_call_site_and_restores() {
        assert_eq!(parse_named(OPROJ_SELECTOR, Some("1")), Ok(true));
        assert!(parse_named(OPROJ_SELECTOR, Some("yes")).is_err());
        assert!(!is_armed());
        {
            let _arm = Arm::enter_selected(true);
            assert!(is_armed());
            assert!(take_armed());
            assert!(!is_armed());
        }
        assert!(!is_armed());
        {
            let _arm = Arm::enter_selected(false);
            assert!(!take_armed());
        }
    }

    #[test]
    fn the_chunk_tally_counts_qsa_core_hand_offs_only() {
        oproj_chunk_begin();
        let act = DevicePtr(ACT);
        let core = |rows| Ready {
            act: ACT,
            rows,
            cols: 6144,
            stream: STREAM,
            producer: Producer::QsaCore,
        };
        note_ready(core(8192));
        assert!(take_matching(act, 8192, 6144, STREAM));
        note_ready(core(8192));
        assert!(
            !take_matching(act, 4096, 6144, STREAM),
            "another projection"
        );
        note_ready(ready()); // Q1's record: not counted
        assert!(take_matching(act, 8192, 2560, STREAM));
        note_oproj_refused();
        let t = oproj_chunk_receipt(0, 8192);
        assert_eq!(
            t,
            OprojTally {
                hits: 1,
                refused: 1,
                missed: 1
            }
        );
        assert_eq!(
            oproj_chunk_receipt(0, 8192),
            OprojTally::ZERO,
            "the receipt restarts the counts"
        );
        note_oproj_refused();
        oproj_chunk_begin();
        assert_eq!(
            oproj_chunk_receipt(8192, 8192),
            OprojTally::ZERO,
            "begin forgets earlier counts"
        );
        assert_eq!(Producer::HcUpmix.selector(), SELECTOR);
        assert_eq!(Producer::QsaCore.selector(), OPROJ_SELECTOR);
    }

    #[test]
    fn dropping_the_arm_clears_an_untaken_record() {
        {
            let _arm = Arm::enter(false).unwrap();
            note_ready(ready());
        }
        assert!(!take_matching(DevicePtr(ACT), 8192, 2560, STREAM));
    }
}
