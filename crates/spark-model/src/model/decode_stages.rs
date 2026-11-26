// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token decode stage attribution (`ATLAS_QWEN4_DECODE_STAGES=1`, fn-decode).
//! Accumulates `stage_timer` regions over decode tokens and logs one
//! `QWEN4_DECODE_STAGES` line (ms per token per stage, largest first) every
//! `ATLAS_QWEN4_DECODE_STAGES_EVERY` tokens (default 32). Decode is eager while
//! enabled, so stage times include launch gaps.

use std::sync::Mutex;

use spark_runtime::cublaslt::stage_timer;

struct Acc {
    tokens: usize,
    total_ms: f64,
    stages: Vec<(String, f64, usize)>,
}

static ACC: Mutex<Acc> = Mutex::new(Acc {
    tokens: 0,
    total_ms: 0.0,
    stages: Vec::new(),
});

pub(crate) fn enabled() -> bool {
    stage_timer::decode_enabled()
}

fn every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_DECODE_STAGES_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n > 0)
            .unwrap_or(32)
    })
}

fn format_stage_count(count: usize, tokens: usize) -> String {
    if count.is_multiple_of(tokens) {
        (count / tokens).to_string()
    } else {
        format!("{:.2}", count as f64 / tokens as f64)
    }
}

/// Close the token's region and fold it into the running sums.
pub(crate) fn token_done(seq_len: usize) {
    if !enabled() {
        return;
    }
    let Some((total, stages)) = stage_timer::finish_raw() else {
        return;
    };
    let mut acc = ACC.lock().unwrap_or_else(|e| e.into_inner());
    acc.tokens += 1;
    acc.total_ms += total;
    for (key, ms, n) in stages {
        match acc.stages.iter_mut().find(|x| x.0 == key) {
            Some(x) => {
                x.1 += ms;
                x.2 += n;
            }
            None => acc.stages.push((key, ms, n)),
        }
    }
    if acc.tokens >= every() {
        let t = acc.tokens as f64;
        let mut stages = std::mem::take(&mut acc.stages);
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        let body: Vec<String> = stages
            .iter()
            .map(|(k, ms, n)| format!("{k}={:.3}/{}", ms / t, format_stage_count(*n, acc.tokens)))
            .collect();
        tracing::info!(
            "QWEN4_DECODE_STAGES tokens={} seq_len={seq_len} ms_per_token={:.3} {}",
            acc.tokens,
            acc.total_ms / t,
            body.join(" ")
        );
        acc.tokens = 0;
        acc.total_ms = 0.0;
    }
}

static VACC: Mutex<Vec<(usize, Acc)>> = Mutex::new(Vec::new());

/// Close a K-row verify step's region (same marks as decode, `v_*` labels
/// inside the batched layers) and log `QWEN4_VERIFY_STAGES k=..` every
/// `ATLAS_QWEN4_DECODE_STAGES_EVERY` steps of that K.
pub(crate) fn verify_done(k: usize, seq_len: usize) {
    if !enabled() {
        return;
    }
    let Some((total, stages)) = stage_timer::finish_raw() else {
        return;
    };
    let mut all = VACC.lock().unwrap_or_else(|e| e.into_inner());
    let idx = match all.iter().position(|(kk, _)| *kk == k) {
        Some(i) => i,
        None => {
            all.push((
                k,
                Acc {
                    tokens: 0,
                    total_ms: 0.0,
                    stages: Vec::new(),
                },
            ));
            all.len() - 1
        }
    };
    let acc = &mut all[idx].1;
    acc.tokens += 1;
    acc.total_ms += total;
    for (key, ms, n) in stages {
        match acc.stages.iter_mut().find(|x| x.0 == key) {
            Some(x) => {
                x.1 += ms;
                x.2 += n;
            }
            None => acc.stages.push((key, ms, n)),
        }
    }
    if acc.tokens >= every() {
        let t = acc.tokens as f64;
        let mut stages = std::mem::take(&mut acc.stages);
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        let body: Vec<String> = stages
            .iter()
            .map(|(key, ms, n)| {
                format!("{key}={:.3}/{}", ms / t, format_stage_count(*n, acc.tokens))
            })
            .collect();
        tracing::info!(
            "QWEN4_VERIFY_STAGES k={k} steps={} seq_len={seq_len} ms_per_step={:.3} {}",
            acc.tokens,
            acc.total_ms / t,
            body.join(" ")
        );
        acc.tokens = 0;
        acc.total_ms = 0.0;
    }
}

/// Per-step (start, end) events of graphed Qwen4 decode, kept one step back so
/// reading them never waits on work in flight.
struct Gap {
    prev: Option<(u64, u64)>,
    steps: usize,
    busy: f64,
    gap: f64,
    wall: f64,
}

static GAP: Mutex<Gap> = Mutex::new(Gap {
    prev: None,
    steps: 0,
    busy: 0.0,
    gap: 0.0,
    wall: 0.0,
});

/// Call with the events recorded at this step's start (before layer 0) and end
/// (after the graph launch). Folds in the PREVIOUS step: busy = its start->end,
/// gap = its end -> this step's start.
pub(crate) fn gap_step(start: u64, end: u64, seq_len: usize) {
    let mut g = GAP.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((ps, pe)) = g.prev.take() {
        let busy = stage_timer::elapsed_raw(ps, pe);
        let gap = stage_timer::elapsed_raw(pe, start);
        if busy.is_finite() && busy >= 0.0 && gap.is_finite() && (0.0..1000.0).contains(&gap) {
            g.steps += 1;
            g.busy += busy;
            g.gap += gap;
            g.wall += busy + gap;
        }
        stage_timer::release_raw(ps);
        stage_timer::release_raw(pe);
    }
    g.prev = Some((start, end));
    if g.steps >= every() {
        let n = g.steps as f64;
        tracing::info!(
            "QWEN4_DECODE_GAP steps={} seq_len={seq_len} busy_ms={:.3} gap_ms={:.3} step_ms={:.3}",
            g.steps,
            g.busy / n,
            g.gap / n,
            g.wall / n
        );
        g.steps = 0;
        g.busy = 0.0;
        g.gap = 0.0;
        g.wall = 0.0;
    }
}
