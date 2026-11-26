// SPDX-License-Identifier: AGPL-3.0-only

//! K8 item A: the prefill work outside the layer loop (the "host bucket").
//!
//! The 12-e1 serve census (`lever-queue/12-e1-stage-census`, serve v3g, `PREFILL_PHASES`)
//! split the 124 ms an 8k request spends outside the layer loop into: `blocks` 42.9 ms
//! (per-block KV zeroing), the MTP last-K replay 31.9 ms (in `finalize`), `embed` 26.0 ms
//! (the token-id H2D waits on chunk 0's `zero_all`), `alloc_sequence` 10.2 ms and
//! `upload_paged` 4.0 ms. Each opt-in `Lever` below is default off and exact by
//! construction, while `zero_all_plan_for` defaults to `skip` on admitted Qwen4
//! single-rank fast prefill (when `default_skip_data == true` and
//! `ATLAS_PREFILL_ZERO_ALL` is unset). Each lever logs one ENGAGED or REFUSED
//! receipt (the call site logs; this file is std-only so
//! `rustc --edition 2024 --test host_bucket.rs` runs its tests):
//!
//! - `ATLAS_PREFILL_KV_ZERO_COALESCE=1`: `ensure_blocks_through_prefill` zeroes the
//!   chunk's fresh KV blocks with one memset per (layer, side, run of consecutive
//!   blocks) instead of two per layer per block (`PagedKvCache::zero_blocks`).
//! - `ATLAS_MTP_LASTK_DEFER=1`: the MTP last-K prompt replay (64 serial single-token
//!   MTP-layer forwards) runs after the first token is emitted, from the scheduler's
//!   `finish_deferred_prefill_tail` hook, instead of at the end of `finalize_last`.
//!   The first token does not depend on it (the replay saves and restores the logits).
//!   K9 item X made it race-free: [`replay_order_plan`] orders the replay against the
//!   default stream (where the first decode step, propose, the K5 verify frames and
//!   `free_sequence` run) with two events and no host wait, and a sequence that never
//!   runs its replay still takes and returns the MTP cache blocks the replay would have
//!   allocated ([`mtp_replay_blocks`]), so the allocator state matches the undeferred order.
//! - `ATLAS_PREFILL_LEAN_FINALIZE=1`: `finalize_last` skips the two diagnostic
//!   readbacks it runs on every prompt longer than 16384 tokens (a stream sync, a
//!   16-value readback, a full-vocab logits D2H and a host sort of 248k values, logged
//!   as `DIAG post-norm` / `DIAG logits`). They read device memory and log only.
//! - `ATLAS_PREFILL_ZERO_ALL=<plan>`: controls what chunk 0's `zero_all`
//!   (19 arena buffers, 4.7 GB at WIDE 16384, ~25 ms of memset per request)
//!   writes. `zero` zeroes all buffers; `fill7f[:list]` writes 0x7F instead of
//!   0x00 (diagnostic detector: outputs identical to zero-fill mean nothing
//!   read pre-request contents); `skip[:list]` leaves them untouched. Without a
//!   list both cover the data buffers, i.e. all but [`ZERO_ALL_INDEX_LIKE`].
//!   When unset, `zero_all_plan_for` defaults to `skip` for admitted Qwen4
//!   single-rank fast prefill (`default_skip_data == true`) and full `zero_all`
//!   otherwise.
//! - `ATLAS_K8A_TRIGGER=<path>` (optional): a selected lever is armed only while
//!   `<path>` exists, so one load can serve an in-process A/B/A (the pattern of
//!   `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP_TRIGGER`).
//! - `ATLAS_MTP_LASTK_DEFER_TRIGGER=<path>` (optional, K9 item X): when set, it alone
//!   gates `ATLAS_MTP_LASTK_DEFER`, so the deferral can be toggled in-process while the
//!   other levers (`ATLAS_PREFILL_KV_ZERO_COALESCE` in serve v6) stay on. Test only.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub(crate) const TRIGGER: &str = "ATLAS_K8A_TRIGGER";

/// The deferral's own trigger (test only): when set, [`armed`] for
/// [`Lever::MtpLastkDefer`] follows this file instead of [`TRIGGER`].
pub(crate) const DEFER_TRIGGER: &str = "ATLAS_MTP_LASTK_DEFER_TRIGGER";

/// Bytes of the head of `buffers.scratch()` saved when the replay is deferred and
/// restored right before it runs: the greedy first-token `argmax_on_device` writes
/// `scratch[0..4]` between the two points (`trait_impl/meta.rs` argmax dispatch).
pub(crate) const SCRATCH_SAVE_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lever {
    KvZeroCoalesce,
    MtpLastkDefer,
    LeanFinalize,
}

impl Lever {
    #[cfg(test)]
    pub(crate) const ALL: [Lever; 3] = [
        Lever::KvZeroCoalesce,
        Lever::MtpLastkDefer,
        Lever::LeanFinalize,
    ];

    pub(crate) fn selector(self) -> &'static str {
        match self {
            Lever::KvZeroCoalesce => "ATLAS_PREFILL_KV_ZERO_COALESCE",
            Lever::MtpLastkDefer => "ATLAS_MTP_LASTK_DEFER",
            Lever::LeanFinalize => "ATLAS_PREFILL_LEAN_FINALIZE",
        }
    }

    fn index(self) -> usize {
        match self {
            Lever::KvZeroCoalesce => 0,
            Lever::MtpLastkDefer => 1,
            Lever::LeanFinalize => 2,
        }
    }
}

/// Absent, empty or `0` = off, `1` = on, anything else is an error (the first use
/// fails instead of silently running the other arm).
pub(crate) fn parse(selector: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{selector} must be absent, 0 or 1; got {other:?}")),
    }
}

/// Whether `lever` is selected; the environment is read once per process.
pub(crate) fn selected(lever: Lever) -> Result<bool, String> {
    static SEL: [OnceLock<Result<bool, String>>; 3] = [const { OnceLock::new() }; 3];
    SEL[lever.index()]
        .get_or_init(|| {
            parse(
                lever.selector(),
                std::env::var(lever.selector()).ok().as_deref(),
            )
        })
        .clone()
}

/// [`selected`] and, when [`TRIGGER`] names a file, that file exists now. For
/// [`Lever::MtpLastkDefer`] a configured [`DEFER_TRIGGER`] replaces [`TRIGGER`].
pub(crate) fn armed(lever: Lever) -> Result<bool, String> {
    if !selected(lever)? {
        return Ok(false);
    }
    static DEFER_TRIGGER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    let own = match lever {
        Lever::MtpLastkDefer => env_path(&DEFER_TRIGGER_PATH, DEFER_TRIGGER),
        Lever::KvZeroCoalesce | Lever::LeanFinalize => None,
    };
    Ok(lever_trigger_allows(trigger_path(), own, |p| p.exists()))
}

/// The path an environment variable names (read once), `None` when unset or empty.
fn env_path(cell: &'static OnceLock<Option<PathBuf>>, var: &str) -> Option<&'static Path> {
    cell.get_or_init(|| {
        std::env::var(var)
            .ok()
            .filter(|t| !t.is_empty())
            .map(PathBuf::from)
    })
    .as_deref()
}

fn trigger_path() -> Option<&'static Path> {
    static TRIGGER_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    env_path(&TRIGGER_PATH, TRIGGER)
}

/// No [`TRIGGER`] configured, or its file exists now.
fn trigger_present() -> bool {
    trigger_allows(trigger_path(), |p| p.exists())
}

/// A lever's own trigger, when configured, decides alone; else [`TRIGGER`] does.
fn lever_trigger_allows(
    shared: Option<&Path>,
    own: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
) -> bool {
    match own {
        Some(path) => exists(path),
        None => trigger_allows(shared, exists),
    }
}

pub(crate) const ZERO_ALL_SELECTOR: &str = "ATLAS_PREFILL_ZERO_ALL";

/// `zero_all` buffers a list-less `fill7f` / `skip` keeps zeroing: gate_logits holds
/// expert indices on some MoE paths (`BufferArena::zero_prefill_essentials` records a
/// CUDA 700 from stale ones), splitk_workspace and scratch hold kernel metadata.
pub(crate) const ZERO_ALL_INDEX_LIKE: [&str; 3] = ["gate_logits", "splitk_workspace", "scratch"];

/// `ATLAS_PREFILL_ZERO_ALL` -> per-buffer fill for `BufferArena::zero_all_with` (index i =
/// `names[i]`, `None` = untouched), or `None` for the plain `zero_all`.
pub(crate) fn parse_zero_all(
    value: Option<&str>,
    names: &[&str],
) -> Result<Option<Vec<Option<u8>>>, String> {
    let value = match value {
        None | Some("") | Some("zero") => return Ok(None),
        Some(v) => v,
    };
    let (mode, list) = match value.split_once(':') {
        Some((mode, list)) => (mode, Some(list)),
        None => (value, None),
    };
    let action = match mode {
        "fill7f" => Some(0x7F),
        "skip" => None,
        other => {
            return Err(format!(
                "{ZERO_ALL_SELECTOR}: mode must be zero, fill7f[:list] or skip[:list]; got {other:?}"
            ));
        }
    };
    let mut plan = vec![Some(0u8); names.len()];
    match list {
        None => {
            for (slot, name) in plan.iter_mut().zip(names) {
                if !ZERO_ALL_INDEX_LIKE.contains(name) {
                    *slot = action;
                }
            }
        }
        Some(list) => {
            let picked: Vec<&str> = list
                .split(',')
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .collect();
            if picked.is_empty() {
                return Err(format!(
                    "{ZERO_ALL_SELECTOR}: empty buffer list in {value:?}"
                ));
            }
            for name in picked {
                let i = names.iter().position(|n| *n == name).ok_or_else(|| {
                    format!(
                        "{ZERO_ALL_SELECTOR}: unknown buffer {name:?} (known: {})",
                        names.join(",")
                    )
                })?;
                plan[i] = action;
            }
        }
    }
    Ok(Some(plan))
}

/// Return the armed `ATLAS_PREFILL_ZERO_ALL` plan (gated by [`TRIGGER`]).
///
/// When `ATLAS_PREFILL_ZERO_ALL` is unset (`None` or `""`) and
/// `default_skip_data` is true (admitted Qwen4 single-rank fast prefill, where
/// every non-index arena buffer is provably written before read), returns the
/// `skip` plan that zeroes [`ZERO_ALL_INDEX_LIKE`] and leaves the data buffers
/// untouched. Explicit `ATLAS_PREFILL_ZERO_ALL=zero` still forces full
/// `zero_all` (`Ok(None)`).
pub(crate) fn zero_all_plan_for(
    names: &[&str],
    default_skip_data: bool,
) -> Result<Option<Vec<Option<u8>>>, String> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW
        .get_or_init(|| std::env::var(ZERO_ALL_SELECTOR).ok())
        .as_deref();
    if matches!(raw, None | Some("")) && default_skip_data && trigger_present() {
        return parse_zero_all(Some("skip"), names);
    }
    type CachedPlan = (Vec<String>, Result<Option<Vec<Option<u8>>>, String>);
    static PLAN: OnceLock<CachedPlan> = OnceLock::new();
    let (cached_names, cached_plan) = PLAN.get_or_init(|| {
        (
            names.iter().map(|s| (*s).to_string()).collect(),
            parse_zero_all(raw, names),
        )
    });
    let plan = if cached_names.len() == names.len()
        && cached_names
            .iter()
            .zip(names)
            .all(|(a, b)| a.as_str() == *b)
    {
        cached_plan.clone()?
    } else {
        parse_zero_all(raw, names)?
    };
    Ok(plan.filter(|_| trigger_present()))
}

fn trigger_allows(trigger: Option<&Path>, exists: impl Fn(&Path) -> bool) -> bool {
    trigger.is_none_or(exists)
}

/// True the first time `key` is seen in this process (one receipt line per key).
pub(crate) fn first_time(key: &'static str) -> bool {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&key) {
        false
    } else {
        seen.push(key);
        true
    }
}

/// What `finalize_last` knows when it would run the MTP last-K replay.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DeferInputs {
    /// `ATLAS_MTP_LASTK_DEFER` armed.
    pub armed: bool,
    /// The replay would run at all: speculation not forced off, capacity > 0, an
    /// active proposer, the capture buffer allocated, rank 0, rows captured.
    pub replay_would_run: bool,
    /// Sequences holding an SSM slot right now (this one included); `None` when the
    /// model has no slot census (no SSM layers).
    pub claimed_sequences: Option<usize>,
    /// Bytes of `mtp_lastk_buf` past the captured rows and the saved logits.
    pub spare_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeferDecision {
    /// Run the replay in `finalize_last` as before.
    Off,
    /// Selected but not exact here: run it in `finalize_last` and log the reason once.
    Refused(&'static str),
    /// Save the scratch head, record the replay, return without running it.
    Defer,
}

pub(crate) fn defer_decision(i: &DeferInputs) -> DeferDecision {
    if !i.armed || !i.replay_would_run {
        return DeferDecision::Off;
    }
    match i.claimed_sequences {
        // Another sequence could prefill or decode between finalize and the hook and
        // change the shared activation buffers the replay reads.
        Some(1) => {}
        Some(_) => return DeferDecision::Refused("other_sequences"),
        None => return DeferDecision::Refused("no_slot_census"),
    }
    if i.spare_bytes < SCRATCH_SAVE_BYTES {
        return DeferDecision::Refused("no_scratch_save_room");
    }
    DeferDecision::Defer
}

/// A replay recorded by `finalize_last` and run by `finish_deferred_prefill_tail`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeferredLastK {
    pub slot_idx: usize,
    /// The full prompt `finalize_last` received (the replay indexes its tail).
    pub tokens: Vec<u32>,
    /// The prefill stream the replay would have run on.
    pub stream: u64,
    /// `speculation_forced_disabled` at finalize time.
    pub speculation_forced_disabled: bool,
}

/// The pending record belongs to `slot_idx` (a record for another slot is stale:
/// that sequence was freed or never reached the hook).
pub(crate) fn pending_matches(pending: Option<&DeferredLastK>, slot_idx: usize) -> bool {
    pending.is_some_and(|d| d.slot_idx == slot_idx)
}

/// The deferred replay's re-run of the first token's raw device argmax, compared with
/// the token the scheduler sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayArgmax {
    Match,
    /// The raw argmax is a stop id the first-token sample masked, and the sampled token
    /// is not: the sampler's host fallback chose it. Not a restore failure.
    MaskedFirstToken,
    /// The restored logits do not give the sampled token: a real defect.
    Mismatch,
}

pub(crate) fn classify_replay_argmax(first: u32, again: u32, suppress_ids: &[u32]) -> ReplayArgmax {
    if again == first {
        ReplayArgmax::Match
    } else if suppress_ids.contains(&again) && !suppress_ids.contains(&first) {
        ReplayArgmax::MaskedFirstToken
    } else {
        ReplayArgmax::Mismatch
    }
}

/// One step of the deferred replay's device schedule (K9 item X).
///
/// The replay runs on the stream `finalize_last` used ([`DeferredLastK::stream`], the
/// scheduler's prefill stream, a non-blocking stream). Everything that consumes or
/// overwrites what it touches runs on the backend's default stream (also non-blocking):
/// the first-token device argmax before it; the first decode step, the MTP propose (it
/// reads the MTP KV rows the replay writes), the K5 verify frames and their graphs,
/// `free_sequence`'s slot zeroing and the next `alloc_sequence` after it. Two
/// non-blocking streams are not ordered by anything but events or a host wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderStep {
    /// `cuEventRecord(event, stream)`: capture the work enqueued on `stream` so far.
    Record { event: u64, stream: u64 },
    /// `cuStreamWaitEvent(stream, event)`: work enqueued on `stream` after this waits
    /// for the captured work. The host does not wait.
    Wait { stream: u64, event: u64 },
    /// Host wait until `stream` is idle (backends without events only).
    Drain { stream: u64 },
    /// The replay body: restore the saved scratch head, then the MTP last-K replay, all
    /// enqueued on `stream`.
    Replay { stream: u64 },
}

/// The device schedule of a deferred replay enqueued on `replay_stream` whose consumers
/// run on `consumer_stream`. `events` = two distinct event handles (in, out), `None` when
/// the backend has none (then two host drains give the same order).
///
/// - same stream: stream order alone;
/// - events: the replay waits for everything already enqueued on the consumer stream
///   (`in`), and every later consumer-stream operation waits for the replay (`out`);
///   work enqueued later on the replay stream (the next request's prefill) follows it by
///   stream order. No host wait anywhere, so nothing is added to the TTFT path or to the
///   scheduler thread beyond the replay's own per-step metadata uploads.
pub(crate) fn replay_order_plan(
    replay_stream: u64,
    consumer_stream: u64,
    events: Option<(u64, u64)>,
) -> Vec<OrderStep> {
    if replay_stream == consumer_stream {
        return vec![OrderStep::Replay {
            stream: replay_stream,
        }];
    }
    match events.filter(|&(ein, eout)| ein != 0 && eout != 0 && ein != eout) {
        Some((ein, eout)) => vec![
            OrderStep::Record {
                event: ein,
                stream: consumer_stream,
            },
            OrderStep::Wait {
                stream: replay_stream,
                event: ein,
            },
            OrderStep::Replay {
                stream: replay_stream,
            },
            OrderStep::Record {
                event: eout,
                stream: replay_stream,
            },
            OrderStep::Wait {
                stream: consumer_stream,
                event: eout,
            },
        ],
        None => vec![
            OrderStep::Drain {
                stream: consumer_stream,
            },
            OrderStep::Replay {
                stream: replay_stream,
            },
            OrderStep::Drain {
                stream: replay_stream,
            },
        ],
    }
}

/// Receipt label of a plan from [`replay_order_plan`].
pub(crate) fn plan_label(plan: &[OrderStep]) -> &'static str {
    if plan
        .iter()
        .any(|s| matches!(s, OrderStep::Wait { .. } | OrderStep::Record { .. }))
    {
        "events"
    } else if plan.iter().any(|s| matches!(s, OrderStep::Drain { .. })) {
        "host_drain"
    } else {
        "same_stream"
    }
}

/// Rows the MTP last-K replay feeds through the MTP layer for `filled` captured rows:
/// `mtp_replay_pairs` pairs row i with token i (legacy, K pairs) or with token i + 1
/// (`ATLAS_MTP_REPLAY_NEXT=1`, K - 1 pairs).
pub(crate) fn mtp_replay_rows(filled: usize, next_token: bool) -> usize {
    if next_token {
        filled.saturating_sub(1)
    } else {
        filled
    }
}

/// MTP KV blocks a replay of `rows` rows allocates from an empty proposer cache: its
/// step `seq_len` (0, 1, .., rows - 1) needs `seq_len / block_size + 1` blocks
/// (`Qwen4MtpHead::step_metadata`), so the last step decides.
pub(crate) fn mtp_replay_blocks(rows: usize, block_size: usize) -> usize {
    if rows == 0 || block_size == 0 {
        0
    } else {
        (rows - 1) / block_size + 1
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn replay_argmax_masked_first_token_is_not_a_mismatch() {
        use super::{ReplayArgmax, classify_replay_argmax};
        const IM_END: u32 = 248046;
        let stops = [IM_END, 248044];
        assert_eq!(
            classify_replay_argmax(54018, 54018, &stops),
            ReplayArgmax::Match
        );
        // P0 receipt: first=54018 after_replay=248046 with the stop ids masked.
        assert_eq!(
            classify_replay_argmax(54018, IM_END, &stops),
            ReplayArgmax::MaskedFirstToken
        );
        // A non-stop raw argmax that differs is a real restore failure.
        assert_eq!(
            classify_replay_argmax(54018, 12, &stops),
            ReplayArgmax::Mismatch
        );
        // Nothing masked: any difference is a mismatch.
        assert_eq!(
            classify_replay_argmax(54018, IM_END, &[]),
            ReplayArgmax::Mismatch
        );
    }

    use super::*;

    #[test]
    fn parse_is_strict() {
        let s = "ATLAS_X";
        assert_eq!(parse(s, None), Ok(false));
        assert_eq!(parse(s, Some("")), Ok(false));
        assert_eq!(parse(s, Some("0")), Ok(false));
        assert_eq!(parse(s, Some("1")), Ok(true));
        assert!(parse(s, Some("true")).is_err());
        assert!(parse(s, Some("2")).is_err());
        assert!(parse(s, Some(" 1")).unwrap_err().contains("ATLAS_X"));
    }

    #[test]
    fn selectors_are_distinct_and_named() {
        let names: Vec<&str> = Lever::ALL.iter().map(|l| l.selector()).collect();
        assert_eq!(
            names,
            [
                "ATLAS_PREFILL_KV_ZERO_COALESCE",
                "ATLAS_MTP_LASTK_DEFER",
                "ATLAS_PREFILL_LEAN_FINALIZE"
            ]
        );
        let mut idx: Vec<usize> = Lever::ALL.iter().map(|l| l.index()).collect();
        idx.dedup();
        assert_eq!(idx, [0, 1, 2]);
    }

    #[test]
    fn trigger_gates_only_when_configured() {
        assert!(trigger_allows(None, |_| false));
        assert!(trigger_allows(Some(Path::new("/x")), |_| true));
        assert!(!trigger_allows(Some(Path::new("/x")), |_| false));
    }

    #[test]
    fn defer_trigger_overrides_the_shared_trigger() {
        let shared = Some(Path::new("/k8a"));
        let own = Some(Path::new("/defer"));
        let only = |present: &'static str| move |p: &Path| p == Path::new(present);
        // No own trigger: the shared one decides (K8 A's behaviour).
        assert!(lever_trigger_allows(None, None, |_| false));
        assert!(lever_trigger_allows(shared, None, only("/k8a")));
        assert!(!lever_trigger_allows(shared, None, only("/defer")));
        // Own trigger configured: it alone decides, whatever the shared one says.
        assert!(lever_trigger_allows(shared, own, only("/defer")));
        assert!(!lever_trigger_allows(shared, own, only("/k8a")));
        assert!(lever_trigger_allows(None, own, only("/defer")));
        assert!(!lever_trigger_allows(None, own, |_| false));
    }

    #[test]
    fn first_time_dedups() {
        assert!(first_time("K8A_TEST_KEY"));
        assert!(!first_time("K8A_TEST_KEY"));
        assert!(first_time("K8A_TEST_KEY_2"));
    }

    fn inputs() -> DeferInputs {
        DeferInputs {
            armed: true,
            replay_would_run: true,
            claimed_sequences: Some(1),
            spare_bytes: 248_077 * 2,
        }
    }

    #[test]
    fn defers_only_when_exact() {
        assert_eq!(defer_decision(&inputs()), DeferDecision::Defer);
        assert_eq!(
            defer_decision(&DeferInputs {
                armed: false,
                ..inputs()
            }),
            DeferDecision::Off
        );
        assert_eq!(
            defer_decision(&DeferInputs {
                replay_would_run: false,
                ..inputs()
            }),
            DeferDecision::Off
        );
        assert_eq!(
            defer_decision(&DeferInputs {
                claimed_sequences: Some(2),
                ..inputs()
            }),
            DeferDecision::Refused("other_sequences")
        );
        assert_eq!(
            defer_decision(&DeferInputs {
                claimed_sequences: Some(0),
                ..inputs()
            }),
            DeferDecision::Refused("other_sequences")
        );
        assert_eq!(
            defer_decision(&DeferInputs {
                claimed_sequences: None,
                ..inputs()
            }),
            DeferDecision::Refused("no_slot_census")
        );
        // FP32 logits fill the vocab * 4 tail of the capture buffer: no room to save scratch.
        assert_eq!(
            defer_decision(&DeferInputs {
                spare_bytes: 0,
                ..inputs()
            }),
            DeferDecision::Refused("no_scratch_save_room")
        );
        assert_eq!(
            defer_decision(&DeferInputs {
                spare_bytes: SCRATCH_SAVE_BYTES,
                ..inputs()
            }),
            DeferDecision::Defer
        );
    }

    const NAMES: [&str; 5] = [
        "hidden_states",
        "gate_logits",
        "ssm_qkvz",
        "splitk_workspace",
        "scratch",
    ];

    #[test]
    fn zero_all_plan_grammar() {
        assert_eq!(parse_zero_all(None, &NAMES), Ok(None));
        assert_eq!(parse_zero_all(Some(""), &NAMES), Ok(None));
        assert_eq!(parse_zero_all(Some("zero"), &NAMES), Ok(None));
        // List-less modes cover the data buffers and keep the index-like ones zeroed.
        assert_eq!(
            parse_zero_all(Some("fill7f"), &NAMES),
            Ok(Some(vec![
                Some(0x7F),
                Some(0),
                Some(0x7F),
                Some(0),
                Some(0)
            ]))
        );
        assert_eq!(
            parse_zero_all(Some("skip"), &NAMES),
            Ok(Some(vec![None, Some(0), None, Some(0), Some(0)]))
        );
        // Explicit lists may name any buffer, index-like ones included.
        assert_eq!(
            parse_zero_all(Some("fill7f:gate_logits, scratch"), &NAMES),
            Ok(Some(vec![
                Some(0),
                Some(0x7F),
                Some(0),
                Some(0),
                Some(0x7F)
            ]))
        );
        assert_eq!(
            parse_zero_all(Some("skip:ssm_qkvz"), &NAMES),
            Ok(Some(vec![Some(0), Some(0), None, Some(0), Some(0)]))
        );
        assert!(
            parse_zero_all(Some("skip:nope"), &NAMES)
                .unwrap_err()
                .contains("unknown buffer")
        );
        assert!(
            parse_zero_all(Some("skip:"), &NAMES)
                .unwrap_err()
                .contains("empty buffer list")
        );
        assert!(
            parse_zero_all(Some("fill00"), &NAMES)
                .unwrap_err()
                .contains("mode must be")
        );
    }

    /// A happens-before model of CUDA stream semantics for two non-blocking streams and
    /// the host: an operation follows the previous one on its stream; a wait marker
    /// follows the operation its event captured; after a host drain of a stream, every
    /// operation enqueued later (on any stream) follows that stream's last operation.
    /// Nothing else orders two non-blocking streams.
    mod order_sim {
        use super::super::OrderStep;
        use std::collections::HashMap;

        pub(super) const D: u64 = 0x10; // consumer (default) stream
        pub(super) const P: u64 = 0x20; // replay (prefill) stream
        pub(super) const EIN: u64 = 0x31;
        pub(super) const EOUT: u64 = 0x32;

        #[derive(Default)]
        pub(super) struct Sim {
            deps: Vec<Vec<usize>>,
            names: Vec<&'static str>,
            last_on: HashMap<u64, usize>,
            captured: HashMap<u64, Option<usize>>,
            host_after: Vec<usize>,
        }

        impl Sim {
            pub(super) fn op(&mut self, stream: u64, name: &'static str) -> usize {
                self.push(stream, name, None)
            }

            fn push(&mut self, stream: u64, name: &'static str, extra: Option<usize>) -> usize {
                let id = self.deps.len();
                let mut deps: Vec<usize> = self.host_after.clone();
                deps.extend(self.last_on.get(&stream).copied());
                deps.extend(extra);
                self.deps.push(deps);
                self.names.push(name);
                self.last_on.insert(stream, id);
                id
            }

            /// Run `plan`; returns the id of its replay operation.
            pub(super) fn run(&mut self, plan: &[OrderStep]) -> usize {
                let mut replay = None;
                for step in plan {
                    match *step {
                        OrderStep::Record { event, stream } => {
                            let at = self.last_on.get(&stream).copied();
                            self.captured.insert(event, at);
                        }
                        OrderStep::Wait { stream, event } => {
                            let at = self.captured.get(&event).copied().flatten();
                            self.push(stream, "wait", at);
                        }
                        OrderStep::Drain { stream } => {
                            self.host_after.extend(self.last_on.get(&stream).copied());
                        }
                        OrderStep::Replay { stream } => {
                            replay = Some(self.push(stream, "replay", None));
                        }
                    }
                }
                replay.expect("plan has a replay step")
            }

            /// The host waits for `stream` (e.g. a D2H read of a result).
            pub(super) fn drain(&mut self, stream: u64) {
                self.host_after.extend(self.last_on.get(&stream).copied());
            }

            /// `a` completes before `b` starts, whatever the GPU's timing.
            pub(super) fn before(&self, a: usize, b: usize) -> bool {
                let mut stack = vec![b];
                let mut seen = vec![false; self.deps.len()];
                while let Some(x) = stack.pop() {
                    for &d in &self.deps[x] {
                        if d == a {
                            return true;
                        }
                        if !seen[d] {
                            seen[d] = true;
                            stack.push(d);
                        }
                    }
                }
                false
            }

            pub(super) fn name(&self, id: usize) -> &'static str {
                self.names[id]
            }
        }
    }

    use order_sim::{D, EIN, EOUT, P, Sim};

    /// The serve schedule around the hook: the first-token argmax kernel on the default
    /// stream (device branch, no host read yet), the hook's plan, then the consumers.
    /// Returns (sim, argmax, replay, consumers).
    fn hook_schedule(plan: &[OrderStep]) -> (Sim, usize, usize, Vec<usize>) {
        let mut s = Sim::default();
        s.op(P, "finalize_last lm_head + scratch save");
        s.op(D, "prefill handoff wait");
        let argmax = s.op(D, "first-token argmax (writes scratch[0..4])");
        let replay = s.run(plan);
        let consumers = vec![
            s.op(D, "first decode step"),
            s.op(D, "MTP propose (reads the replayed MTP KV rows)"),
            s.op(D, "K5 verify frame graph"),
            s.op(D, "free_sequence zero_slot"),
            s.op(P, "next request zero_all"),
        ];
        (s, argmax, replay, consumers)
    }

    fn assert_ordered(s: &Sim, before: usize, replay: usize, after: &[usize]) {
        assert!(
            s.before(before, replay),
            "replay may overlap '{}'",
            s.name(before)
        );
        for &c in after {
            assert!(
                s.before(replay, c),
                "'{}' may overlap the replay",
                s.name(c)
            );
        }
    }

    #[test]
    fn order_plan_shapes() {
        let ev = replay_order_plan(P, D, Some((EIN, EOUT)));
        assert_eq!(
            ev,
            vec![
                OrderStep::Record {
                    event: EIN,
                    stream: D
                },
                OrderStep::Wait {
                    stream: P,
                    event: EIN
                },
                OrderStep::Replay { stream: P },
                OrderStep::Record {
                    event: EOUT,
                    stream: P
                },
                OrderStep::Wait {
                    stream: D,
                    event: EOUT
                },
            ]
        );
        assert_eq!(plan_label(&ev), "events");
        let same = replay_order_plan(D, D, Some((EIN, EOUT)));
        assert_eq!(same, vec![OrderStep::Replay { stream: D }]);
        assert_eq!(plan_label(&same), "same_stream");
        for bad in [None, Some((0, EOUT)), Some((EIN, 0)), Some((EIN, EIN))] {
            let drain = replay_order_plan(P, D, bad);
            assert_eq!(
                drain,
                vec![
                    OrderStep::Drain { stream: D },
                    OrderStep::Replay { stream: P },
                    OrderStep::Drain { stream: P },
                ]
            );
            assert_eq!(plan_label(&drain), "host_drain");
        }
    }

    #[test]
    fn order_plan_is_race_free() {
        for plan in [
            replay_order_plan(P, D, Some((EIN, EOUT))),
            replay_order_plan(P, D, None),
        ] {
            let (s, argmax, replay, consumers) = hook_schedule(&plan);
            assert_ordered(&s, argmax, replay, &consumers);
        }
        // Same stream: stream order alone.
        let mut s = Sim::default();
        let argmax = s.op(D, "argmax");
        let replay = s.run(&replay_order_plan(D, D, None));
        let decode = s.op(D, "decode");
        assert_ordered(&s, argmax, replay, &[decode]);
    }

    /// Negative controls: the schedules K8 item A shipped. Each must be caught.
    #[test]
    fn k8a_schedules_race() {
        // The late-replay fallback at the first propose: replay on the prefill stream, then
        // a host drain of that stream only. The bootstrap decode just enqueued on the default
        // stream (it wrote the shared activation buffers and the logits) is not before it.
        let k8a = [
            OrderStep::Replay { stream: P },
            OrderStep::Drain { stream: P },
        ];
        let mut s = Sim::default();
        let bootstrap = s.op(D, "bootstrap decode");
        let replay = s.run(&k8a);
        let propose = s.op(D, "propose");
        assert!(!s.before(bootstrap, replay), "negative control not caught");
        assert!(s.before(replay, propose));
        // The hook path was ordered only by the host read inside the first-token sample;
        // without that read (a device-only producer) the same schedule races too.
        let (s, argmax, replay, _) = hook_schedule(&k8a);
        assert!(!s.before(argmax, replay), "negative control not caught");
        let mut s = Sim::default();
        let argmax = s.op(D, "argmax");
        s.drain(D); // copy_d2h of the sampled token
        let replay = s.run(&k8a);
        assert!(s.before(argmax, replay));
        // Dropping K8 A's trailing host drain without an event: consumers race the replay.
        let (s, _, replay, consumers) = hook_schedule(&[OrderStep::Replay { stream: P }]);
        let raced: Vec<&str> = consumers
            .iter()
            .filter(|&&c| !s.before(replay, c))
            .map(|&c| s.name(c))
            .collect();
        assert_eq!(
            raced,
            [
                "first decode step",
                "MTP propose (reads the replayed MTP KV rows)",
                "K5 verify frame graph",
                "free_sequence zero_slot"
            ]
        );
    }

    #[test]
    fn replay_blocks_match_step_metadata_allocation() {
        // Qwen4MtpHead::step_metadata: before step seq_len, grow the table to
        // seq_len / bs + 1 blocks.
        fn simulate(rows: usize, bs: usize) -> usize {
            let mut table = 0usize;
            for seq_len in 0..rows {
                while table < seq_len / bs + 1 {
                    table += 1;
                }
            }
            table
        }
        for bs in [1usize, 2, 16, 64] {
            for rows in 0..200 {
                assert_eq!(
                    mtp_replay_blocks(rows, bs),
                    simulate(rows, bs),
                    "rows={rows} bs={bs}"
                );
            }
        }
        assert_eq!(mtp_replay_blocks(64, 16), 4);
        assert_eq!(mtp_replay_blocks(63, 16), 4);
        assert_eq!(mtp_replay_blocks(65, 16), 5);
        assert_eq!(mtp_replay_blocks(5, 0), 0);
        assert_eq!(mtp_replay_rows(64, false), 64);
        assert_eq!(mtp_replay_rows(64, true), 63);
        assert_eq!(mtp_replay_rows(0, true), 0);
    }

    #[test]
    fn pending_match_is_per_slot() {
        let d = DeferredLastK {
            slot_idx: 0,
            tokens: vec![1, 2, 3],
            stream: 7,
            speculation_forced_disabled: false,
        };
        assert!(pending_matches(Some(&d), 0));
        assert!(!pending_matches(Some(&d), 1));
        assert!(!pending_matches(None, 0));
    }
}
