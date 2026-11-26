// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in CUDA-event stage timer for one prefill chunk
//! (`ATLAS_QWEN4_PREFILL_STAGES=1`, default off).
//!
//! `nsys` cannot attach to a Flash-Next server on GB10 (the load ends below
//! the session memory floor under CUPTI), so this attributes a chunk's GPU
//! timeline without a profiler: [`mark`] records a timing event on the
//! prefill stream and charges the interval since the previous mark to its
//! label; [`finish`] waits for the last event once and returns the per-label
//! sums. No mid-chunk host synchronization; each mark costs one
//! `cuEventRecord`. State is per thread (one prefill thread drives a chunk).

use std::cell::RefCell;
use std::sync::OnceLock;

use super::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_STAGES").ok().as_deref() == Some("1") || decode_enabled()
    })
}

/// `ATLAS_QWEN4_DECODE_STAGES=1` (fn-decode): the same event marks on the
/// single-token decode path. Decode runs eagerly while it is set (graph
/// replay would hide the stage boundaries), so absolute numbers include
/// launch gaps; compare stages against each other and against a graphed run.
pub fn decode_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("ATLAS_QWEN4_DECODE_STAGES").ok().as_deref() == Some("1"))
}

#[derive(Default)]
struct State {
    pool: Vec<u64>,
    marks: Vec<(u64, &'static str, &'static str)>,
    group: &'static str,
    active: bool,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn record(st: &mut State, label: &'static str, stream: u64) {
    let event = match st.pool.pop() {
        Some(e) => e,
        None => {
            let mut e = 0u64;
            // Flags 0: timing enabled (the backend's own events disable it).
            // SAFETY: `&mut e` is a valid out-pointer for `cuEventCreate` on the active CUDA context.
            if unsafe { cuEventCreate(&mut e, 0) } != 0 {
                return;
            }
            e
        }
    };
    // SAFETY: `event` was created by `cuEventCreate`; `stream` is a caller-provided CUDA stream handle.
    if unsafe { cuEventRecord(event, stream) } != 0 {
        st.pool.push(event);
        return;
    }
    let group = st.group;
    st.marks.push((event, group, label));
}

/// Start a chunk: drops any unfinished marks and records the origin.
pub fn begin(stream: u64) {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let used: Vec<u64> = st.marks.drain(..).map(|m| m.0).collect();
        st.pool.extend(used);
        st.group = "";
        st.active = true;
        record(&mut st, "start", stream);
    });
}

/// Prefix for the labels of the following marks (e.g. the layer kind).
pub fn set_group(group: &'static str) {
    if !enabled() {
        return;
    }
    STATE.with(|s| s.borrow_mut().group = group);
}

/// Charge the time since the previous mark to `label` (in the current group).
pub fn mark(label: &'static str, stream: u64) {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if st.active {
            record(&mut st, label, stream);
        }
    });
}

fn elapsed(a: u64, b: u64) -> f64 {
    let mut ms = 0f32;
    // SAFETY: `a` and `b` are CUDA timing events created by `cuEventCreate`; `&mut ms` is writable.
    if unsafe { cuEventElapsedTime(&mut ms, a, b) } != 0 {
        return 0.0;
    }
    f64::from(ms)
}

/// `ATLAS_QWEN4_DECODE_GAP=1`: per-step GPU busy vs idle-between-steps
/// attribution for graphed decode (events outside the graph only).
pub fn gap_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("ATLAS_QWEN4_DECODE_GAP").ok().as_deref() == Some("1"))
}

/// Record a timing event on `stream`; 0 on failure.
pub fn record_raw(stream: u64) -> u64 {
    let mut e = 0u64;
    // SAFETY: `&mut e` is a valid out-pointer; `e` is destroyed immediately if `cuEventRecord` fails.
    unsafe {
        if cuEventCreate(&mut e, 0) != 0 {
            return 0;
        }
        if cuEventRecord(e, stream) != 0 {
            super::cuEventDestroy_v2(e);
            return 0;
        }
    }
    e
}

/// Milliseconds between two recorded events (waits for `b`); NaN on failure.
pub fn elapsed_raw(a: u64, b: u64) -> f64 {
    // SAFETY: non-zero `a` and `b` are event handles returned by `record_raw`.
    if a == 0 || b == 0 || unsafe { cuEventSynchronize(b) } != 0 {
        return f64::NAN;
    }
    let mut ms = 0f32;
    // SAFETY: both events have completed (after `cuEventSynchronize(b)`) and `&mut ms` is writable.
    if unsafe { cuEventElapsedTime(&mut ms, a, b) } != 0 {
        return f64::NAN;
    }
    f64::from(ms)
}

pub fn release_raw(e: u64) {
    if e != 0 {
        // SAFETY: `e` is a live event handle returned by `record_raw` and destroyed at most once.
        unsafe {
            super::cuEventDestroy_v2(e);
        }
    }
}

/// `(total_ms, [(group.label, ms, count)])`.
pub type RawStageTimes = (f64, Vec<(String, f64, usize)>);

/// End the region and return `(total_ms, [(group.label, ms, count)])`
/// unsorted, in first-seen order. Waits for the last mark.
pub fn finish_raw() -> Option<RawStageTimes> {
    if !enabled() {
        return None;
    }
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if !std::mem::replace(&mut st.active, false) || st.marks.len() < 2 {
            return None;
        }
        let first = st.marks[0].0;
        let last = st.marks[st.marks.len() - 1].0;
        // SAFETY: `last` is a recorded CUDA event created by `record`.
        if unsafe { cuEventSynchronize(last) } != 0 {
            return None;
        }
        let mut sums: Vec<(String, f64, usize)> = Vec::new();
        for w in st.marks.windows(2) {
            let ms = elapsed(w[0].0, w[1].0);
            let key = if w[1].1.is_empty() {
                w[1].2.to_string()
            } else {
                format!("{}.{}", w[1].1, w[1].2)
            };
            match sums.iter_mut().find(|x| x.0 == key) {
                Some(x) => {
                    x.1 += ms;
                    x.2 += 1;
                }
                None => sums.push((key, ms, 1)),
            }
        }
        Some((elapsed(first, last), sums))
    })
}

/// End the chunk: waits for the last mark and returns
/// `total_ms=<t> <group.label>=<ms>/<count> ...`, largest first.
pub fn finish() -> Option<String> {
    if !enabled() {
        return None;
    }
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if !std::mem::replace(&mut st.active, false) || st.marks.len() < 2 {
            return None;
        }
        let first = st.marks[0].0;
        let last = st.marks[st.marks.len() - 1].0;
        // SAFETY: `last` is a recorded CUDA event created by `record`.
        if unsafe { cuEventSynchronize(last) } != 0 {
            return None;
        }
        let mut sums: Vec<(String, f64, usize)> = Vec::new();
        for w in st.marks.windows(2) {
            let ms = elapsed(w[0].0, w[1].0);
            let key = if w[1].1.is_empty() {
                w[1].2.to_string()
            } else {
                format!("{}.{}", w[1].1, w[1].2)
            };
            match sums.iter_mut().find(|x| x.0 == key) {
                Some(x) => {
                    x.1 += ms;
                    x.2 += 1;
                }
                None => sums.push((key, ms, 1)),
            }
        }
        let total = elapsed(first, last);
        sums.sort_by(|a, b| b.1.total_cmp(&a.1));
        let body: Vec<String> = sums
            .iter()
            .map(|(k, ms, n)| format!("{k}={ms:.1}/{n}"))
            .collect();
        Some(format!("total_ms={total:.1} {}", body.join(" ")))
    })
}
