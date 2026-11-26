// SPDX-License-Identifier: AGPL-3.0-only

//! In-process `MemAvailable` floor watcher.
//!
//! GB10 shares host memory with the GPU: when `MemAvailable` falls below
//! about 12 GiB the driver fails allocations (`NV_ERR_NO_MEMORY`) or the host
//! reboots. Until 9.15.7 only external harnesses watched the floor. This
//! thread samples `/proc/meminfo` every `FLUX_MEM_WATCH_INTERVAL_MS` during
//! load and serve and, below `FLUX_MEM_FLOOR_MIB`, logs `MEM_FLOOR_BREACH`
//! and takes the phase's action:
//! - `abort` (load default): flush logs and exit 75; a breaching load does
//!   not recover.
//! - `refuse` (serve default): new inference POST requests under `/v1/` get 503 until
//!   `MemAvailable` is back above floor + `FLUX_MEM_FLOOR_MARGIN_MIB`;
//!   in-flight requests run to completion.
//! - `log`: the line only.
//!
//! The watcher never touches the scheduler: refusal is one atomic flag read
//! by an HTTP middleware.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};

const MIB: usize = 1024 * 1024;
pub(crate) const ABORT_EXIT_CODE: i32 = 75;
const DEFAULT_FLOOR_MIB: usize = spark_runtime::startup_ledger::DEFAULT_SERVE_WATCH_FLOOR_MIB;
const DEFAULT_DISCRETE_FLOOR_MIB: usize =
    spark_runtime::startup_ledger::DEFAULT_DISCRETE_HOST_FLOOR_MIB;
const DEFAULT_MARGIN_MIB: usize = 1_024;
const DEFAULT_INTERVAL_MS: u64 = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Load,
    Serve,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Load => "load",
            Phase::Serve => "serve",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloorAction {
    Refuse,
    Abort,
    Log,
}

impl FloorAction {
    fn parse(key: &str, raw: &str) -> Result<Self> {
        match raw.trim() {
            "refuse" => Ok(Self::Refuse),
            "abort" => Ok(Self::Abort),
            "log" => Ok(Self::Log),
            other => bail!("{key}={other:?}: expected refuse, abort or log"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::Abort => "abort",
            Self::Log => "log",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WatchConfig {
    pub(crate) floor_bytes: usize,
    pub(crate) recover_margin_bytes: usize,
    pub(crate) interval: Duration,
    pub(crate) load_action: FloorAction,
    pub(crate) serve_action: FloorAction,
}

/// `FLUX_<suffix>` first, then the `ATLAS_<suffix>` alias `main` mirrors.
fn flux_value(get: &impl Fn(&str) -> Option<String>, suffix: &str) -> Option<(String, String)> {
    ["FLUX_", "ATLAS_"].iter().find_map(|prefix| {
        let key = format!("{prefix}{suffix}");
        get(&key).map(|value| (key, value))
    })
}

fn parse_mib(get: &impl Fn(&str) -> Option<String>, suffix: &str, default: usize) -> Result<usize> {
    let Some((key, raw)) = flux_value(get, suffix) else {
        return Ok(default * MIB);
    };
    raw.trim()
        .parse::<usize>()
        .ok()
        .and_then(|mib| mib.checked_mul(MIB))
        .with_context(|| format!("{key}={raw:?}: expected a MiB count"))
}

fn parse_interval(get: &impl Fn(&str) -> Option<String>) -> Result<Duration> {
    let Some((key, raw)) = flux_value(get, "MEM_WATCH_INTERVAL_MS") else {
        return Ok(Duration::from_millis(DEFAULT_INTERVAL_MS));
    };
    match raw.trim().parse::<u64>() {
        Ok(ms) if ms > 0 => Ok(Duration::from_millis(ms)),
        _ => bail!("{key}={raw:?}: expected milliseconds > 0"),
    }
}

fn parse_action(
    get: &impl Fn(&str) -> Option<String>,
    suffix: &str,
    default: FloorAction,
) -> Result<FloorAction> {
    flux_value(get, suffix).map_or(Ok(default), |(key, raw)| FloorAction::parse(&key, &raw))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WatchToggle {
    Default,
    On,
    Off,
}

fn parse_watch_toggle(get: &impl Fn(&str) -> Option<String>) -> Result<WatchToggle> {
    match flux_value(get, "MEM_WATCH").map(|(k, v)| (k, v.trim().to_string())) {
        None => Ok(WatchToggle::Default),
        Some((_, v)) if v == "on" || v == "1" => Ok(WatchToggle::On),
        Some((_, v)) if v == "off" || v == "0" => Ok(WatchToggle::Off),
        Some((k, v)) => bail!("{k}={v:?}: expected on or off"),
    }
}

fn parse_discrete_ledger_floor_bytes(raw: &str) -> Result<usize> {
    let selector = spark_runtime::startup_ledger::FLOOR_SELECTOR;
    let mib = raw
        .trim()
        .parse::<usize>()
        .with_context(|| format!("{selector}={raw:?} is not a MiB count"))?;
    mib.checked_mul(MIB)
        .with_context(|| format!("{selector}={mib} overflows"))
}

impl WatchConfig {
    /// `None` when `FLUX_MEM_WATCH=off`. Malformed values are launch errors:
    /// a wrong floor here is a host OOM on UMA devices.
    pub(crate) fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        Self::from_lookup_for_device(get, false)
    }

    /// Parse watcher configuration for UMA (`is_discrete == false`) or discrete
    /// GPUs (`is_discrete == true`).
    ///
    /// On discrete GPUs (dedicated VRAM isolated from Linux host RAM), `mem_watch`
    /// defaults to disabled unless explicitly forced via `FLUX_MEM_WATCH=on|1`
    /// (or `ATLAS_MEM_WATCH=on|1`).
    pub(crate) fn from_lookup_for_device(
        get: impl Fn(&str) -> Option<String>,
        is_discrete: bool,
    ) -> Result<Option<Self>> {
        match parse_watch_toggle(&get)? {
            WatchToggle::Off => return Ok(None),
            WatchToggle::Default if is_discrete => return Ok(None),
            WatchToggle::Default | WatchToggle::On => {}
        }
        let default_floor_mib = if is_discrete {
            DEFAULT_DISCRETE_FLOOR_MIB
        } else {
            DEFAULT_FLOOR_MIB
        };
        let watch_floor = parse_mib(&get, "MEM_FLOOR_MIB", default_floor_mib)?;
        // Honor an explicit `ATLAS_MEMAVAILABLE_FLOOR_MIB` so the watcher and
        // startup ledger share one canonical floor (`max(ledger, watch)`).
        let floor_bytes = match get(spark_runtime::startup_ledger::FLOOR_SELECTOR) {
            Some(raw) if is_discrete => {
                let ledger_floor = parse_discrete_ledger_floor_bytes(&raw)?;
                watch_floor.max(ledger_floor)
            }
            Some(raw) => {
                let override_selected = get(spark_runtime::startup_ledger::FLOOR_OVERRIDE_SELECTOR)
                    .as_deref()
                    == Some("1");
                let ledger_floor = spark_runtime::startup_ledger::parse_floor_bytes_with_override(
                    Some(&raw),
                    override_selected,
                )?;
                watch_floor.max(ledger_floor)
            }
            None => watch_floor,
        };
        Ok(Some(Self {
            floor_bytes,
            recover_margin_bytes: parse_mib(&get, "MEM_FLOOR_MARGIN_MIB", DEFAULT_MARGIN_MIB)?,
            interval: parse_interval(&get)?,
            load_action: parse_action(&get, "MEM_FLOOR_ACTION_LOAD", FloorAction::Abort)?,
            serve_action: parse_action(&get, "MEM_FLOOR_ACTION_SERVE", FloorAction::Refuse)?,
        }))
    }

    pub(crate) fn from_env() -> Result<Option<Self>> {
        Self::from_lookup_for_device(
            |key| std::env::var(key).ok(),
            spark_runtime::startup_ledger::is_discrete_device(),
        )
    }

    fn action(&self, phase: Phase) -> FloorAction {
        match phase {
            Phase::Load => self.load_action,
            Phase::Serve => self.serve_action,
        }
    }
}

/// Validate `FLUX_MEM_WATCH*` syntax before CUDA initialization determines
/// whether the active GPU is UMA or discrete.
///
/// Topology-dependent policy (such as GB10's 6656 MiB UMA qualification bar or
/// disabling the watcher by default on discrete GPUs) is enforced in
/// [`start_for_gpu`] right after GPU initialization.
pub(crate) fn validate_env() -> Result<()> {
    validate_env_lookup(|key| std::env::var(key).ok())
}

fn validate_env_lookup(get: impl Fn(&str) -> Option<String>) -> Result<()> {
    if parse_watch_toggle(&get)? == WatchToggle::Off {
        return Ok(());
    }
    let _ = parse_mib(&get, "MEM_FLOOR_MIB", DEFAULT_FLOOR_MIB)?;
    if let Some(raw) = get(spark_runtime::startup_ledger::FLOOR_SELECTOR) {
        let _ = parse_discrete_ledger_floor_bytes(&raw)?;
    }
    let _ = parse_mib(&get, "MEM_FLOOR_MARGIN_MIB", DEFAULT_MARGIN_MIB)?;
    let _ = parse_interval(&get)?;
    let _ = parse_action(&get, "MEM_FLOOR_ACTION_LOAD", FloorAction::Abort)?;
    let _ = parse_action(&get, "MEM_FLOOR_ACTION_SERVE", FloorAction::Refuse)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WatchState {
    Healthy,
    Breached,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    Breach {
        mem_available: usize,
        phase: Phase,
        action: FloorAction,
    },
    Recovered {
        mem_available: usize,
        phase: Phase,
    },
    Abort,
}

/// One sample through the state machine. Pure: the caller applies effects.
pub(crate) fn step(
    cfg: &WatchConfig,
    state: WatchState,
    phase: Phase,
    sample: Option<usize>,
) -> (WatchState, Vec<Effect>) {
    let Some(mem_available) = sample else {
        return (state, Vec::new());
    };
    let recover_at = cfg.floor_bytes.saturating_add(cfg.recover_margin_bytes);
    match state {
        WatchState::Healthy if mem_available < cfg.floor_bytes => {
            let action = cfg.action(phase);
            let breach = Effect::Breach {
                mem_available,
                phase,
                action,
            };
            let effects = if action == FloorAction::Abort {
                vec![breach, Effect::Abort]
            } else {
                vec![breach]
            };
            (WatchState::Breached, effects)
        }
        WatchState::Breached if mem_available >= recover_at => (
            WatchState::Healthy,
            vec![Effect::Recovered {
                mem_available,
                phase,
            }],
        ),
        unchanged => (unchanged, Vec::new()),
    }
}

/// True when new inference requests must be refused.
pub(crate) fn refusing(cfg: &WatchConfig, state: WatchState, phase: Phase) -> bool {
    state == WatchState::Breached && cfg.action(phase) == FloorAction::Refuse
}

pub(crate) fn breach_line(mem_available: usize, cfg: &WatchConfig, phase: Phase) -> String {
    format!(
        "MEM_FLOOR_BREACH memavail_mib={} floor_mib={} phase={} action={}",
        mem_available / MIB,
        cfg.floor_bytes / MIB,
        phase.name(),
        cfg.action(phase).name(),
    )
}

/// State shared between the watcher thread, the HTTP middleware and the
/// ready line. Atomics only: readers never block the watcher or the
/// scheduler.
#[derive(Debug)]
pub(crate) struct WatchShared {
    phase: AtomicU8,
    refusing: AtomicBool,
    baseline: usize,
    min_load_available: AtomicUsize,
    fit_need: AtomicUsize,
}

const PHASE_LOAD: u8 = 0;
const PHASE_SERVE: u8 = 1;
const FIT_NEED_UNSET: usize = usize::MAX;

impl WatchShared {
    pub(crate) fn new(baseline: usize) -> Self {
        Self {
            phase: AtomicU8::new(PHASE_LOAD),
            refusing: AtomicBool::new(false),
            baseline,
            min_load_available: AtomicUsize::new(baseline),
            fit_need: AtomicUsize::new(FIT_NEED_UNSET),
        }
    }

    pub(crate) fn phase(&self) -> Phase {
        if self.phase.load(Ordering::Acquire) == PHASE_SERVE {
            Phase::Serve
        } else {
            Phase::Load
        }
    }

    pub(crate) fn set_phase(&self, phase: Phase) {
        let raw = if phase == Phase::Serve {
            PHASE_SERVE
        } else {
            PHASE_LOAD
        };
        self.phase.store(raw, Ordering::Release);
    }

    pub(crate) fn observe(&self, phase: Phase, mem_available: usize) {
        if phase == Phase::Load {
            self.min_load_available
                .fetch_min(mem_available, Ordering::AcqRel);
        }
    }

    /// `MemAvailable` at watcher start minus the lowest load-phase sample.
    pub(crate) fn peak_load_bytes(&self) -> usize {
        self.baseline
            .saturating_sub(self.min_load_available.load(Ordering::Acquire))
    }

    pub(crate) fn is_refusing(&self) -> bool {
        self.refusing.load(Ordering::Acquire)
    }

    pub(crate) fn set_refusing(&self, value: bool) {
        self.refusing.store(value, Ordering::Release);
    }

    pub(crate) fn set_fit_need(&self, bytes: usize) {
        self.fit_need.store(bytes, Ordering::Release);
    }

    pub(crate) fn fit_need(&self) -> Option<usize> {
        let raw = self.fit_need.load(Ordering::Acquire);
        (raw != FIT_NEED_UNSET).then_some(raw)
    }
}

fn apply(cfg: &WatchConfig, effects: &[Effect], exit: &dyn Fn(i32)) {
    for effect in effects {
        match effect {
            Effect::Breach {
                mem_available,
                phase,
                ..
            } => tracing::error!(
                target: "atlas_mem_watch",
                "{}",
                breach_line(*mem_available, cfg, *phase)
            ),
            Effect::Recovered {
                mem_available,
                phase,
            } => tracing::warn!(
                target: "atlas_mem_watch",
                "MEM_FLOOR_RECOVERED memavail_mib={} floor_mib={} margin_mib={} phase={}",
                mem_available / MIB,
                cfg.floor_bytes / MIB,
                cfg.recover_margin_bytes / MIB,
                phase.name(),
            ),
            Effect::Abort => exit(ABORT_EXIT_CODE),
        }
    }
}

/// One watcher iteration: record, step, apply, publish the refusal flag.
pub(crate) fn tick(
    cfg: &WatchConfig,
    shared: &WatchShared,
    state: WatchState,
    sample: Option<usize>,
    exit: &dyn Fn(i32),
) -> WatchState {
    let phase = shared.phase();
    if let Some(bytes) = sample {
        shared.observe(phase, bytes);
    }
    let (next, effects) = step(cfg, state, phase, sample);
    apply(cfg, &effects, exit);
    shared.set_refusing(refusing(cfg, next, phase));
    next
}

fn flush_and_exit(code: i32) {
    use std::io::Write as _;
    crate::tui::init::flush_tee();
    let _ = std::io::stderr().flush();
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}

static SHARED: OnceLock<Arc<WatchShared>> = OnceLock::new();

/// The process watcher's shared state, once started.
pub(crate) fn shared() -> Option<&'static Arc<WatchShared>> {
    SHARED.get()
}

/// True while the process memory-floor watcher is refusing new generation work.
pub(crate) fn is_refusing() -> bool {
    shared().is_some_and(|s| s.is_refusing())
}

/// Start the watcher thread once per process for the initialized GPU backend.
///
/// Disabled by `FLUX_MEM_WATCH=off`, and disabled by default on discrete GPUs
/// (`!gpu.is_unified_memory()`) unless explicitly forced via `FLUX_MEM_WATCH=on`.
pub(crate) fn start_for_gpu(gpu: &dyn spark_runtime::gpu::GpuBackend) -> Result<()> {
    let is_discrete =
        !gpu.is_unified_memory() || spark_runtime::startup_ledger::is_discrete_device();
    let lookup = |key: &str| std::env::var(key).ok();
    let toggle = parse_watch_toggle(&lookup)?;
    let explicit_on = toggle == WatchToggle::On;
    let Some(cfg) = WatchConfig::from_lookup_for_device(lookup, is_discrete)? else {
        if is_discrete && toggle == WatchToggle::Default {
            tracing::info!(
                target: "atlas_mem_watch",
                "MEM_WATCH disabled by default on discrete GPU (set FLUX_MEM_WATCH=on to enable)"
            );
        } else {
            tracing::warn!(target: "atlas_mem_watch", "MEM_WATCH disabled (FLUX_MEM_WATCH=off)");
        }
        return Ok(());
    };
    let Some(baseline) = spark_runtime::startup_ledger::mem_available_bytes() else {
        tracing::warn!(target: "atlas_mem_watch", "MEM_WATCH disabled: /proc/meminfo unreadable");
        return Ok(());
    };
    let shared = Arc::new(WatchShared::new(baseline));
    if SHARED.set(shared.clone()).is_err() {
        return Ok(());
    }
    tracing::info!(
        target: "atlas_mem_watch",
        "MEM_WATCH armed floor_mib={} margin_mib={} interval_ms={} load={} serve={} memavail_mib={}",
        cfg.floor_bytes / MIB,
        cfg.recover_margin_bytes / MIB,
        cfg.interval.as_millis(),
        cfg.load_action.name(),
        cfg.serve_action.name(),
        baseline / MIB,
    );
    std::thread::Builder::new()
        .name("mem-watch".to_string())
        .spawn(move || {
            let mut state = WatchState::Healthy;
            loop {
                // Defensive check in case the watcher was started before CUDA
                // initialization revealed a discrete GPU.
                if !explicit_on && spark_runtime::startup_ledger::is_discrete_device() {
                    shared.set_refusing(false);
                    tracing::info!(
                        target: "atlas_mem_watch",
                        "MEM_WATCH disarming after discrete GPU detected (set FLUX_MEM_WATCH=on to force)"
                    );
                    break;
                }
                let sample = spark_runtime::startup_ledger::mem_available_bytes();
                state = tick(&cfg, &shared, state, sample, &flush_and_exit);
                std::thread::sleep(cfg.interval);
            }
        })
        .context("spawn mem-watch thread")?;
    Ok(())
}

/// Exact HTTP `POST` endpoints that start new GPU work. Read-only metadata
/// (`/v1/models`), stored-item reads (`GET /v1/responses/{id}`,
/// `GET /v1/chat/completions/{id}`), token counting (`POST /v1/messages/count_tokens`),
/// health, and metrics stay reachable during a breach.
const GENERATION_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/messages",
    "/v1/responses",
    "/v1/decisions",
    "/v1/disagg/prefill",
    "/v1/disagg/decode",
    "/v1/disagg/generate",
];

pub(crate) fn is_generation_request(method: &axum::http::Method, path: &str) -> bool {
    *method == axum::http::Method::POST && GENERATION_PATHS.contains(&path)
}

/// The 503 for a new inference request while the floor is breached.
pub(crate) fn floor_rejection(
    method: &axum::http::Method,
    path: &str,
    refusing: bool,
) -> Option<(axum::http::StatusCode, serde_json::Value)> {
    if !refusing || !is_generation_request(method, path) {
        return None;
    }
    Some((
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        serde_json::json!({
            "error": {
                "message": "Host MemAvailable is below the server's memory floor; \
                            new requests are refused until it recovers. Retry later.",
                "type": "server_error",
                "param": null,
                "code": "memory_floor",
            }
        }),
    ))
}

/// Build the HTTP 503 response (with `Retry-After: 5`) when the memory floor
/// is breached for a generation request.
pub(crate) fn floor_rejection_response(
    method: &axum::http::Method,
    path: &str,
    refusing: bool,
) -> Option<axum::response::Response> {
    use axum::http::{HeaderValue, header};
    use axum::response::IntoResponse;

    let (status, body) = floor_rejection(method, path, refusing)?;
    let mut resp = (status, axum::Json(body)).into_response();
    resp.headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
    Some(resp)
}

pub(crate) async fn mem_floor_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    match floor_rejection_response(req.method(), req.uri().path(), is_refusing()) {
        None => next.run(req).await,
        Some(resp) => resp,
    }
}

pub(crate) fn ready_line(
    mem_available: usize,
    peak_load: usize,
    fit_need: Option<usize>,
    floor: usize,
) -> String {
    let headroom = (mem_available as i64 - floor as i64) / MIB as i64;
    let need = fit_need.map_or_else(|| "none".to_string(), |b| (b / MIB).to_string());
    format!(
        "READY memavail_mib={} peak_load_mib={} fit_need_mib={need} headroom_mib={headroom}",
        mem_available / MIB,
        peak_load / MIB,
    )
}

/// Switch to the serve phase and print the ready line.
pub(crate) fn mark_ready() {
    let Some(shared) = shared() else {
        return;
    };
    shared.set_phase(Phase::Serve);
    let floor = WatchConfig::from_env()
        .ok()
        .flatten()
        .map_or(DEFAULT_FLOOR_MIB * MIB, |c| c.floor_bytes);
    let mem_available = spark_runtime::startup_ledger::mem_available_bytes().unwrap_or(0);
    tracing::info!(
        target: "atlas_startup_budget",
        "{}",
        ready_line(mem_available, shared.peak_load_bytes(), shared.fit_need(), floor)
    );
}

#[cfg(test)]
#[path = "mem_watch_tests.rs"]
mod tests;

#[cfg(test)]
mod floor_unification_tests {
    use std::collections::HashMap;

    use super::{MIB, WatchConfig};

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    #[test]
    fn explicit_memavailable_floor_raises_watch_floor_to_unified_max() {
        let cfg = WatchConfig::from_lookup(lookup(&[
            ("ATLAS_MEMAVAILABLE_FLOOR_MIB", "16384"),
            ("FLUX_MEM_FLOOR_MIB", "8192"),
        ]))
        .expect("parse")
        .expect("enabled");
        assert_eq!(cfg.floor_bytes, 16_384 * MIB);

        let cfg = WatchConfig::from_lookup(lookup(&[
            ("ATLAS_MEMAVAILABLE_FLOOR_MIB", "8192"),
            ("FLUX_MEM_FLOOR_MIB", "6656"),
        ]))
        .expect("parse")
        .expect("enabled");
        assert_eq!(cfg.floor_bytes, 8_192 * MIB);
    }

    #[test]
    fn unauthorized_reduced_memavailable_floor_fails_closed() {
        assert!(
            WatchConfig::from_lookup(lookup(&[("ATLAS_MEMAVAILABLE_FLOOR_MIB", "4096")])).is_err()
        );
    }

    #[test]
    fn discrete_gpu_disables_mem_watch_by_default_and_honors_explicit_on() {
        // Default on discrete GPU: disabled without error even on low host RAM.
        assert_eq!(
            WatchConfig::from_lookup_for_device(lookup(&[]), true).expect("parse"),
            None
        );
        // Explicit off stays disabled.
        assert_eq!(
            WatchConfig::from_lookup_for_device(lookup(&[("FLUX_MEM_WATCH", "off")]), true)
                .expect("parse"),
            None
        );
        // Malformed toggle still fails closed on discrete.
        assert!(
            WatchConfig::from_lookup_for_device(lookup(&[("FLUX_MEM_WATCH", "maybe")]), true)
                .is_err()
        );
        // Explicit FLUX_MEM_WATCH=on arms with discrete default floor (1024 MiB).
        let cfg = WatchConfig::from_lookup_for_device(lookup(&[("FLUX_MEM_WATCH", "on")]), true)
            .expect("parse")
            .expect("explicitly enabled");
        assert_eq!(cfg.floor_bytes, super::DEFAULT_DISCRETE_FLOOR_MIB * MIB);

        // Explicit FLUX_MEM_WATCH=1 honors lower discrete floor without UMA 6656 MiB refusal.
        let cfg = WatchConfig::from_lookup_for_device(
            lookup(&[
                ("FLUX_MEM_WATCH", "1"),
                ("ATLAS_MEMAVAILABLE_FLOOR_MIB", "512"),
                ("FLUX_MEM_FLOOR_MIB", "256"),
            ]),
            true,
        )
        .expect("parse")
        .expect("explicitly enabled");
        assert_eq!(cfg.floor_bytes, 512 * MIB);

        // Pre-CUDA validation rejects typos while accepting valid discrete floors.
        assert!(
            super::validate_env_lookup(lookup(&[("ATLAS_MEMAVAILABLE_FLOOR_MIB", "512")])).is_ok()
        );
        assert!(super::validate_env_lookup(lookup(&[("FLUX_MEM_WATCH", "bogus")])).is_err());
        assert!(super::validate_env_lookup(lookup(&[("FLUX_MEM_FLOOR_MIB", "12G")])).is_err());
    }
}
