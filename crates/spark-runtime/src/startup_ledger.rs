// SPDX-License-Identifier: AGPL-3.0-only

//! Startup memory ledger and the declared `MemAvailable` floor.
//!
//! On GB10 the GPU shares host memory, so a server that over-allocates kills
//! the host rather than itself. Every byte the factory allocates after the
//! weight store is recorded here by pool name, with the bytes the caller
//! asked for (when it computed them) and the host-wide `MemAvailable` delta
//! measured around the allocation. The server prints the ledger once at the
//! end of load and refuses to serve when `MemAvailable` is under the floor.
//!
//! `MemAvailable` is read from `/proc/meminfo` directly. The backend's
//! `free_memory()` is `max(cuMemGetInfo, MemAvailable)` and every budget
//! guard reads it; this module never uses that optimistic metric.

use anyhow::{Context, Result};
use parking_lot::Mutex;

/// `ATLAS_MEMAVAILABLE_FLOOR_MIB`: the server must never push the host's
/// `MemAvailable` under this many MiB. Values below the sealed-container
/// qualification bar require the explicit, nonqualifying override.
pub const FLOOR_SELECTOR: &str = "ATLAS_MEMAVAILABLE_FLOOR_MIB";
/// `ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1`: opt out of floor qualification.
/// Required both for a reduced floor and for serving below the declared floor.
pub const FLOOR_OVERRIDE_SELECTOR: &str = "ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE";
/// Sealed-container qualification bar (`memavailable.json` floor), MiB.
pub const DEFAULT_FLOOR_MIB: usize = 6656;
pub const MIB: usize = 1024 * 1024;
pub const QUALIFICATION_FLOOR_BYTES: usize = DEFAULT_FLOOR_MIB * MIB;
/// Default post-load VRAM headroom floor on discrete GPUs (`SM120` / `SM103`), MiB.
///
/// Discrete GPUs have dedicated VRAM isolated from Linux host RAM and cannot
/// trigger host `earlyoom`, so they do not require GB10's 6.5–12 GiB UMA reserve.
pub const DEFAULT_DISCRETE_VRAM_FLOOR_MIB: usize = 512;
/// Default host RAM sanity floor on discrete-GPU machines, MiB.
pub const DEFAULT_DISCRETE_HOST_FLOOR_MIB: usize = 1024;

/// True when the active GPU is a confirmed discrete device (dedicated VRAM).
///
/// Returns `false` on integrated/UMA devices (e.g. GB10) and before device
/// properties have been initialized, preserving legacy UMA behavior by default.
#[must_use]
pub fn is_discrete_device() -> bool {
    atlas_core::device::active_device_is_integrated() == Some(false)
}

/// Parse the floor selector's value. `None` (unset) is the default; a
/// malformed value is a launch error, not a silent default, because a wrong
/// bar here is a host OOM on GB10.
pub fn parse_floor_bytes(value: Option<&str>) -> Result<usize> {
    parse_floor_bytes_with_override(value, false)
}

/// Parse an explicitly overridden floor without treating it as qualifying.
pub fn parse_floor_bytes_with_override(
    value: Option<&str>,
    override_selected: bool,
) -> Result<usize> {
    let mib = match value {
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{FLOOR_SELECTOR}={raw:?} is not a MiB count"))?,
        None => DEFAULT_FLOOR_MIB,
    };
    let bytes = mib
        .checked_mul(MIB)
        .with_context(|| format!("{FLOOR_SELECTOR}={mib} overflows"))?;
    anyhow::ensure!(
        bytes >= QUALIFICATION_FLOOR_BYTES || override_selected,
        "{FLOOR_SELECTOR}={mib} MiB is below the {DEFAULT_FLOOR_MIB} MiB qualification floor; \
         set {FLOOR_OVERRIDE_SELECTOR}=1 only for an explicitly NONQUALIFYING run"
    );
    Ok(bytes)
}

/// The declared floor in bytes, read from the environment.
pub fn memavailable_floor_bytes() -> Result<usize> {
    let override_selected = floor_override_selected();
    match std::env::var(FLOOR_SELECTOR) {
        Ok(value) => parse_floor_bytes_with_override(Some(&value), override_selected),
        Err(std::env::VarError::NotPresent) => {
            parse_floor_bytes_with_override(None, override_selected)
        }
        Err(error) => anyhow::bail!("{FLOOR_SELECTOR}: {error}"),
    }
}

/// True when the operator explicitly opted out of the floor refusal.
pub fn floor_override_selected() -> bool {
    std::env::var(FLOOR_OVERRIDE_SELECTOR).ok().as_deref() == Some("1")
}

/// `MemAvailable` in bytes from `/proc/meminfo` text.
pub fn parse_mem_available_bytes(meminfo: &str) -> Option<usize> {
    let mut matching_lines = meminfo
        .lines()
        .filter_map(|line| line.strip_prefix("MemAvailable:"));
    let mut fields = matching_lines.next()?.split_whitespace();
    let kb = fields.next()?.parse::<usize>().ok()?;
    if fields.next() != Some("kB") || fields.next().is_some() || matching_lines.next().is_some() {
        return None;
    }
    kb.checked_mul(1024)
}

/// Real host `MemAvailable` from `/proc/meminfo`, independent of GPU topology.
#[must_use]
pub fn host_mem_available_bytes() -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| parse_mem_available_bytes(&contents))
}

/// Real host `MemAvailable`, or `None` where `/proc/meminfo` is unavailable.
#[must_use]
pub fn mem_available_bytes() -> Option<usize> {
    host_mem_available_bytes()
}

/// Select the available-memory metric appropriate for the active device topology:
/// - Unified memory (`is_integrated != Some(false)`): host `MemAvailable`.
/// - Discrete GPU (`is_integrated == Some(false)`): exact free CUDA VRAM,
///   falling back to host `MemAvailable` only when no CUDA reading is available.
#[must_use]
pub(crate) fn select_available_memory_bytes(
    is_integrated: Option<bool>,
    host_mem_available: Option<usize>,
    cuda_free_vram: Option<usize>,
) -> Option<usize> {
    if is_integrated == Some(false) {
        cuda_free_vram.or(host_mem_available)
    } else {
        host_mem_available
    }
}

/// Available memory relevant to GPU allocation deltas and device headroom.
///
/// On UMA (`active_device_is_integrated() != Some(false)`), reads host
/// `/proc/meminfo` `MemAvailable`. On discrete GPUs
/// (`active_device_is_integrated() == Some(false)`), reads exact free VRAM via
/// `cuMemGetInfo_v2` (falling back to host `MemAvailable` if CUDA is
/// unavailable or no context is bound on the calling thread).
#[must_use]
pub fn available_device_memory_bytes() -> Option<usize> {
    let is_integrated = atlas_core::device::active_device_is_integrated();
    #[cfg(feature = "cuda")]
    let cuda_free = if is_integrated == Some(false) {
        crate::cuda_backend::cuda_free_memory_bytes()
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_free = None;

    let host_mem = if is_integrated == Some(false) && cuda_free.is_some() {
        None
    } else {
        host_mem_available_bytes()
    };
    select_available_memory_bytes(is_integrated, host_mem, cuda_free)
}

/// Outcome of checking post-load device headroom against the active floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceHeadroomDecision {
    /// Available memory meets or exceeds the active floor.
    Pass { headroom_bytes: usize },
    /// Available memory is below the active floor.
    Refused {
        available_bytes: usize,
        floor_bytes: usize,
        shortfall_bytes: usize,
    },
    /// Floor enforcement was explicitly overridden by the operator.
    Overridden { shortfall_bytes: usize },
    /// Required memory telemetry was unavailable.
    Unmeasurable,
}

/// Inputs for evaluating post-load headroom across UMA and discrete GPUs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostLoadHeadroomInput {
    /// Device topology from `atlas_core::device::active_device_is_integrated()`.
    pub is_integrated: Option<bool>,
    /// Host `/proc/meminfo` `MemAvailable`, in bytes.
    pub host_mem_available_bytes: Option<usize>,
    /// Discrete GPU free VRAM from `cuMemGetInfo_v2`, in bytes.
    pub cuda_free_vram_bytes: Option<usize>,
    /// Declared UMA `MemAvailable` floor, in bytes.
    pub uma_floor_bytes: usize,
    /// Discrete GPU VRAM headroom floor, in bytes.
    pub discrete_vram_floor_bytes: usize,
    /// Host RAM sanity floor on discrete-GPU machines, in bytes.
    pub discrete_host_floor_bytes: usize,
    /// Whether `ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1` is active.
    pub override_selected: bool,
}

/// Evaluate post-load headroom for either UMA or discrete GPUs.
///
/// - On UMA (`is_integrated != Some(false)`), enforces `host_mem_available_bytes`
///   against `uma_floor_bytes` (identical to the GB10 qualification contract).
/// - On discrete GPUs (`is_integrated == Some(false)`), enforces
///   `cuda_free_vram_bytes` against `discrete_vram_floor_bytes` first, then
///   applies a reasonable `discrete_host_floor_bytes` sanity check against host
///   RAM so discrete runs neither fail on GB10's 6.5 GiB UMA host bar nor pass
///   blindly when VRAM is exhausted on a high-RAM host.
#[must_use]
pub fn evaluate_post_load_headroom(input: PostLoadHeadroomInput) -> DeviceHeadroomDecision {
    if input.is_integrated != Some(false) {
        return match input.host_mem_available_bytes {
            None => DeviceHeadroomDecision::Unmeasurable,
            Some(avail) if input.override_selected => DeviceHeadroomDecision::Overridden {
                shortfall_bytes: input.uma_floor_bytes.saturating_sub(avail),
            },
            Some(avail) if avail >= input.uma_floor_bytes => DeviceHeadroomDecision::Pass {
                headroom_bytes: avail - input.uma_floor_bytes,
            },
            Some(avail) => DeviceHeadroomDecision::Refused {
                available_bytes: avail,
                floor_bytes: input.uma_floor_bytes,
                shortfall_bytes: input.uma_floor_bytes - avail,
            },
        };
    }

    let Some(vram) = input.cuda_free_vram_bytes else {
        return DeviceHeadroomDecision::Unmeasurable;
    };
    if vram < input.discrete_vram_floor_bytes {
        let shortfall_bytes = input.discrete_vram_floor_bytes - vram;
        return if input.override_selected {
            DeviceHeadroomDecision::Overridden { shortfall_bytes }
        } else {
            DeviceHeadroomDecision::Refused {
                available_bytes: vram,
                floor_bytes: input.discrete_vram_floor_bytes,
                shortfall_bytes,
            }
        };
    }
    if let Some(host) = input.host_mem_available_bytes
        && host < input.discrete_host_floor_bytes
    {
        let shortfall_bytes = input.discrete_host_floor_bytes - host;
        return if input.override_selected {
            DeviceHeadroomDecision::Overridden { shortfall_bytes }
        } else {
            DeviceHeadroomDecision::Refused {
                available_bytes: host,
                floor_bytes: input.discrete_host_floor_bytes,
                shortfall_bytes,
            }
        };
    }
    if input.override_selected {
        DeviceHeadroomDecision::Overridden { shortfall_bytes: 0 }
    } else {
        DeviceHeadroomDecision::Pass {
            headroom_bytes: vram - input.discrete_vram_floor_bytes,
        }
    }
}

/// Default in-process serve memory-floor watcher threshold, MiB.
pub const DEFAULT_SERVE_WATCH_FLOOR_MIB: usize = 12_288;

fn flux_or_atlas<'a>(
    get: &impl Fn(&str) -> Option<String>,
    flux_key: &'a str,
    atlas_key: &'a str,
) -> Option<(&'a str, String)> {
    get(flux_key)
        .map(|v| (flux_key, v))
        .or_else(|| get(atlas_key).map(|v| (atlas_key, v)))
}

fn serve_watch_floor_bytes_from_lookup_with_default(
    get: &impl Fn(&str) -> Option<String>,
    default_mib: usize,
) -> Result<usize> {
    if let Some((key, raw)) = flux_or_atlas(get, "FLUX_MEM_WATCH", "ATLAS_MEM_WATCH") {
        match raw.trim() {
            "on" | "1" => {}
            "off" | "0" => return Ok(0),
            other => anyhow::bail!("{key}={other:?}: expected on or off"),
        }
    }
    if let Some((key, raw)) = flux_or_atlas(
        get,
        "FLUX_MEM_FLOOR_ACTION_SERVE",
        "ATLAS_MEM_FLOOR_ACTION_SERVE",
    ) {
        match raw.trim() {
            "refuse" | "abort" => {}
            "log" => return Ok(0),
            other => anyhow::bail!("{key}={other:?}: expected refuse, abort or log"),
        }
    }
    let Some((key, raw)) = flux_or_atlas(get, "FLUX_MEM_FLOOR_MIB", "ATLAS_MEM_FLOOR_MIB") else {
        return Ok(default_mib * MIB);
    };
    raw.trim()
        .parse::<usize>()
        .ok()
        .and_then(|mib| mib.checked_mul(MIB))
        .with_context(|| format!("{key}={raw:?}: expected a MiB count"))
}

/// Active `MemAvailable` serve refusal/abort floor from the in-process memory
/// watcher (`FLUX_MEM_FLOOR_MIB` / `ATLAS_MEM_FLOOR_MIB`, default `12_288 MiB`
/// unless `FLUX_MEM_WATCH=off|0` or `FLUX_MEM_FLOOR_ACTION_SERVE=log`).
pub fn serve_watch_floor_bytes_from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<usize> {
    serve_watch_floor_bytes_from_lookup_with_default(&get, DEFAULT_SERVE_WATCH_FLOOR_MIB)
}

/// Canonical effective `MemAvailable` floor in bytes:
/// `max(ATLAS_MEMAVAILABLE_FLOOR_MIB, FLUX_MEM_FLOOR_MIB / ATLAS_MEM_FLOOR_MIB)`.
///
/// Matches the optional FP8 weight-cache floor policy so runtime buffer pools
/// and the serve watcher enforce one unified UMA floor.
pub fn effective_mem_floor_bytes_from_lookup(
    get: impl Fn(&str) -> Option<String>,
) -> Result<usize> {
    let override_selected = get(FLOOR_OVERRIDE_SELECTOR).as_deref() == Some("1");
    let ledger_floor =
        parse_floor_bytes_with_override(get(FLOOR_SELECTOR).as_deref(), override_selected)?;
    let watch_floor = serve_watch_floor_bytes_from_lookup(&get)?;
    Ok(ledger_floor.max(watch_floor))
}

/// Topology-aware effective memory floor in bytes.
///
/// - On UMA (`is_integrated != Some(false)`): delegates verbatim to
///   [`effective_mem_floor_bytes_from_lookup`] (GB10 6.5 GiB ledger / 12 GiB
///   serve-watch UMA floor).
/// - On discrete GPUs (`is_integrated == Some(false)`): defaults to
///   [`DEFAULT_DISCRETE_HOST_FLOOR_MIB`] (1 GiB) instead of GB10's 6.5/12 GiB
///   UMA floors, while honoring explicit operator overrides and still rejecting
///   malformed or overflowing values.
pub fn effective_mem_floor_bytes_for_device(
    get: impl Fn(&str) -> Option<String>,
    is_integrated: Option<bool>,
) -> Result<usize> {
    if is_integrated != Some(false) {
        return effective_mem_floor_bytes_from_lookup(get);
    }
    let ledger_floor = match get(FLOOR_SELECTOR) {
        Some(raw) => {
            let mib = raw
                .trim()
                .parse::<usize>()
                .with_context(|| format!("{FLOOR_SELECTOR}={raw:?} is not a MiB count"))?;
            mib.checked_mul(MIB)
                .with_context(|| format!("{FLOOR_SELECTOR}={mib} overflows"))?
        }
        None => DEFAULT_DISCRETE_HOST_FLOOR_MIB * MIB,
    };
    let watch_floor =
        serve_watch_floor_bytes_from_lookup_with_default(&get, DEFAULT_DISCRETE_HOST_FLOOR_MIB)?;
    Ok(ledger_floor.max(watch_floor))
}

/// Canonical effective memory floor in bytes, read from the environment and
/// adapted for UMA vs discrete GPU topology.
pub fn effective_mem_floor_bytes() -> Result<usize> {
    effective_mem_floor_bytes_for_device(
        |key| std::env::var(key).ok(),
        atlas_core::device::active_device_is_integrated(),
    )
}

/// One allocation phase of startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    /// Stable pool name (`kv-cache`, `buffer-arena`, ...). The server merges
    /// preflight-requested bytes into factory-measured entries by this name.
    pub pool: String,
    /// Free-form sizing context (`8450 blocks x 12 layers`).
    pub detail: String,
    /// Bytes the caller asked the allocator for, when it computed them.
    pub requested_bytes: Option<usize>,
    /// `MemAvailable` before minus after the phase: positive consumed,
    /// negative reclaimed. Host-wide, so other processes perturb it.
    pub actual_delta_bytes: Option<i64>,
}

static ENTRIES: Mutex<Vec<LedgerEntry>> = Mutex::new(Vec::new());

/// Allocation-receipt source for startup phases: reads host `MemAvailable` on
/// UMA devices (GB10) and exact CUDA free VRAM on discrete GPUs.
fn ledger_source() -> crate::alloc_receipt::Source {
    crate::alloc_receipt::Source::new(&crate::alloc_receipt::GLOBAL, available_device_memory_bytes)
}

/// Drop every entry. Called at the start of each model load so a swap does
/// not report the previous model's pools. Also starts this thread's
/// allocation-receipt session (`STARTUP_ALLOC_RECEIPT_TOTAL`).
pub fn reset() {
    ENTRIES.lock().clear();
    crate::alloc_receipt::begin_session(ledger_source());
}

/// Append an entry without measuring anything.
pub fn record(
    pool: &str,
    detail: impl Into<String>,
    requested_bytes: Option<usize>,
    actual_delta_bytes: Option<i64>,
) {
    ENTRIES.lock().push(LedgerEntry {
        pool: pool.to_string(),
        detail: detail.into(),
        requested_bytes,
        actual_delta_bytes,
    });
}

/// `MemAvailable` before minus after, as a signed byte count.
pub fn signed_delta(before: Option<usize>, after: Option<usize>) -> Option<i64> {
    let (before, after) = (before?, after?);
    let before = i64::try_from(before).ok()?;
    let after = i64::try_from(after).ok()?;
    before.checked_sub(after)
}

/// Run `phase` and record its device-memory delta under `pool` (host
/// `MemAvailable` on UMA, free VRAM on discrete GPUs). The entry is recorded
/// even when `phase` fails, so a refused start still prints what it had
/// consumed up to the failure. The phase also logs its allocation receipt
/// (`STARTUP_ALLOC_RECEIPT`), whose delta is the one recorded here.
pub fn measure<T>(
    pool: &str,
    detail: impl Into<String>,
    requested_bytes: Option<usize>,
    phase: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let (result, receipt) = crate::alloc_receipt::run_phase(ledger_source(), pool, phase);
    record(
        pool,
        detail,
        requested_bytes,
        receipt.total.mem_available_delta_bytes,
    );
    receipt.log();
    result
}

/// Run `phase` for its allocation receipt only: it logs a
/// `STARTUP_ALLOC_RECEIPT` line like a ledger phase but adds no ledger row.
/// For load steps the ledger has no pool for (drafter, LM-head source).
pub fn scope<T>(name: &str, phase: impl FnOnce() -> T) -> T {
    let (value, receipt) = crate::alloc_receipt::run_phase(ledger_source(), name, phase);
    receipt.log();
    value
}

/// [`scope`] for a run of statements: the guard logs the receipt when dropped.
pub fn scope_guard(name: &str) -> crate::alloc_receipt::PhaseGuard {
    crate::alloc_receipt::begin_logged_phase(ledger_source(), name)
}

/// Snapshot of every entry recorded since the last [`reset`].
pub fn entries() -> Vec<LedgerEntry> {
    ENTRIES.lock().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_defaults_to_the_qualification_bar() {
        assert_eq!(parse_floor_bytes(None).unwrap(), 6656 * MIB);
    }

    #[test]
    fn floor_parses_a_trimmed_mib_count() {
        assert_eq!(parse_floor_bytes(Some(" 8192 ")).unwrap(), 8192 * MIB);
        assert_eq!(
            parse_floor_bytes(Some("6656")).unwrap(),
            QUALIFICATION_FLOOR_BYTES
        );
    }

    #[test]
    fn reduced_floors_require_the_explicit_nonqualifying_override() {
        for raw in ["0", "1", "6655"] {
            let error = parse_floor_bytes(Some(raw)).unwrap_err().to_string();
            assert!(error.contains(FLOOR_OVERRIDE_SELECTOR), "{error}");
            assert!(error.contains("NONQUALIFYING"), "{error}");
            let mib = raw.parse::<usize>().unwrap();
            assert_eq!(
                parse_floor_bytes_with_override(Some(raw), true).unwrap(),
                mib * MIB
            );
        }
    }

    #[test]
    fn floor_rejects_malformed_and_overflowing_values() {
        let error = parse_floor_bytes(Some("7g")).unwrap_err().to_string();
        assert!(error.contains(FLOOR_SELECTOR), "{error}");
        assert!(parse_floor_bytes(Some("")).is_err());
        assert!(parse_floor_bytes(Some(&usize::MAX.to_string())).is_err());
    }

    #[test]
    fn mem_available_is_read_from_meminfo_text() {
        let text = "MemTotal:       125000000 kB\nMemFree:        19000000 kB\n\
                    MemAvailable:   13412000 kB\nBuffers:          1234 kB\n";
        assert_eq!(parse_mem_available_bytes(text), Some(13_412_000 * 1024));
        assert_eq!(parse_mem_available_bytes("MemFree: 1 kB\n"), None);
        assert_eq!(parse_mem_available_bytes("MemAvailable: lots kB\n"), None);
    }

    #[test]
    fn malformed_memavailable_telemetry_is_not_a_measurement() {
        for text in [
            "MemAvailable: 100\n",
            "MemAvailable: 100 MB\n",
            "MemAvailable: 100 kB unexpected\n",
            "MemAvailable: 100 kB\nMemAvailable: 200 kB\n",
            "MemAvailable: -1 kB\n",
        ] {
            assert_eq!(parse_mem_available_bytes(text), None, "{text}");
        }
        assert_eq!(parse_mem_available_bytes("MemAvailable: 0 kB\n"), Some(0));
        assert_eq!(
            parse_mem_available_bytes(&format!("MemAvailable: {} kB\n", usize::MAX)),
            None
        );
    }

    #[test]
    fn signed_delta_keeps_direction() {
        assert_eq!(signed_delta(Some(100), Some(40)), Some(60));
        assert_eq!(signed_delta(Some(40), Some(100)), Some(-60));
        assert_eq!(signed_delta(None, Some(1)), None);
        assert_eq!(signed_delta(Some(1), None), None);
    }

    #[test]
    fn effective_mem_floor_unifies_ledger_and_serve_watch_floors() {
        let lookup = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<&str, &str> = pairs.iter().copied().collect();
            effective_mem_floor_bytes_from_lookup(move |k| map.get(k).map(|v| (*v).to_string()))
        };
        // Default: max(6656 MiB ledger, 12288 MiB serve watcher) = 12288 MiB.
        assert_eq!(lookup(&[]).unwrap(), 12_288 * MIB);

        // FLUX_MEM_FLOOR_MIB wins over ATLAS_MEM_FLOOR_MIB and stays >= ledger floor.
        assert_eq!(
            lookup(&[
                ("FLUX_MEM_FLOOR_MIB", "8192"),
                ("ATLAS_MEM_FLOOR_MIB", "4096"),
            ])
            .unwrap(),
            8_192 * MIB
        );

        // Explicitly raised ATLAS_MEMAVAILABLE_FLOOR_MIB dominates a lower watch floor.
        assert_eq!(
            lookup(&[
                ("ATLAS_MEMAVAILABLE_FLOOR_MIB", "16384"),
                ("FLUX_MEM_FLOOR_MIB", "8192"),
            ])
            .unwrap(),
            16_384 * MIB
        );

        // Disabled watcher or log-only serve action leaves the ledger floor in force.
        assert_eq!(lookup(&[("FLUX_MEM_WATCH", "off")]).unwrap(), 6_656 * MIB);
        assert_eq!(
            lookup(&[("FLUX_MEM_FLOOR_ACTION_SERVE", "log")]).unwrap(),
            6_656 * MIB
        );

        // Unauthorized reduced ledger floor still fails closed.
        assert!(lookup(&[("ATLAS_MEMAVAILABLE_FLOOR_MIB", "4096")]).is_err());
        assert!(lookup(&[("FLUX_MEM_FLOOR_MIB", "12G")]).is_err());
    }

    #[test]
    fn measure_records_even_when_the_phase_fails() {
        // The ledger is process-global: use a pool name no other test uses.
        let pool = "test-pool-measure-failure";
        let result: Result<()> = measure(pool, "unit", Some(7), || anyhow::bail!("boom"));
        assert!(result.is_err());
        let entry = entries()
            .into_iter()
            .find(|entry| entry.pool == pool)
            .expect("entry recorded");
        assert_eq!(entry.requested_bytes, Some(7));
        assert_eq!(entry.detail, "unit");
    }

    #[test]
    fn select_available_memory_uses_vram_on_discrete_and_host_on_uma() {
        const GIB: usize = 1024 * MIB;
        // Discrete GPU prefers exact CUDA free VRAM over host RAM.
        assert_eq!(
            select_available_memory_bytes(Some(false), Some(240 * GIB), Some(28 * GIB)),
            Some(28 * GIB)
        );
        // Discrete GPU falls back to host RAM only if CUDA reading is unavailable.
        assert_eq!(
            select_available_memory_bytes(Some(false), Some(240 * GIB), None),
            Some(240 * GIB)
        );
        // Integrated UMA and unset topology always use host MemAvailable.
        assert_eq!(
            select_available_memory_bytes(Some(true), Some(95 * GIB), Some(40 * GIB)),
            Some(95 * GIB)
        );
        assert_eq!(
            select_available_memory_bytes(None, Some(95 * GIB), Some(40 * GIB)),
            Some(95 * GIB)
        );
    }

    #[test]
    fn discrete_effective_floor_uses_discrete_defaults_without_uma_bar() {
        let lookup = |pairs: &[(&str, &str)], is_integrated: Option<bool>| {
            let map: std::collections::HashMap<&str, &str> = pairs.iter().copied().collect();
            effective_mem_floor_bytes_for_device(
                move |k| map.get(k).map(|v| (*v).to_string()),
                is_integrated,
            )
        };

        // Discrete GPU defaults to 1024 MiB instead of GB10's 12288 MiB UMA floor.
        assert_eq!(
            lookup(&[], Some(false)).unwrap(),
            DEFAULT_DISCRETE_HOST_FLOOR_MIB * MIB
        );
        // Explicit lower floor is allowed on discrete without UMA override.
        assert_eq!(
            lookup(
                &[
                    ("ATLAS_MEMAVAILABLE_FLOOR_MIB", "512"),
                    ("FLUX_MEM_FLOOR_MIB", "512"),
                ],
                Some(false)
            )
            .unwrap(),
            512 * MIB
        );
        // Explicit higher watch floor still dominates on discrete.
        assert_eq!(
            lookup(&[("FLUX_MEM_FLOOR_MIB", "4096")], Some(false)).unwrap(),
            4096 * MIB
        );
        // Disabling watch leaves the discrete ledger floor in force.
        assert_eq!(
            lookup(&[("FLUX_MEM_WATCH", "off")], Some(false)).unwrap(),
            DEFAULT_DISCRETE_HOST_FLOOR_MIB * MIB
        );
        // Malformed values still fail closed on discrete.
        assert!(lookup(&[("ATLAS_MEMAVAILABLE_FLOOR_MIB", "bad")], Some(false)).is_err());
        assert!(
            lookup(
                &[("ATLAS_MEMAVAILABLE_FLOOR_MIB", &usize::MAX.to_string())],
                Some(false)
            )
            .is_err()
        );

        // UMA (Some(true) and None) preserves identical 12288 MiB default and 6656 MiB bar.
        assert_eq!(lookup(&[], Some(true)).unwrap(), 12_288 * MIB);
        assert_eq!(lookup(&[], None).unwrap(), 12_288 * MIB);
        assert!(lookup(&[("ATLAS_MEMAVAILABLE_FLOOR_MIB", "4096")], Some(true)).is_err());
    }

    #[test]
    fn post_load_headroom_checks_vram_on_discrete_and_host_ram_on_uma() {
        const GIB: usize = 1024 * MIB;
        let base = PostLoadHeadroomInput {
            is_integrated: Some(false),
            host_mem_available_bytes: Some(4 * GIB),
            cuda_free_vram_bytes: Some(4 * GIB),
            uma_floor_bytes: QUALIFICATION_FLOOR_BYTES,
            discrete_vram_floor_bytes: DEFAULT_DISCRETE_VRAM_FLOOR_MIB * MIB,
            discrete_host_floor_bytes: DEFAULT_DISCRETE_HOST_FLOOR_MIB * MIB,
            override_selected: false,
        };

        // Discrete GPU with 4 GiB free VRAM and 4 GiB host RAM passes (does not
        // falsely refuse under GB10's 6.5 GiB UMA host floor).
        assert_eq!(
            evaluate_post_load_headroom(base),
            DeviceHeadroomDecision::Pass {
                headroom_bytes: 4 * GIB - DEFAULT_DISCRETE_VRAM_FLOOR_MIB * MIB,
            }
        );

        // Discrete GPU with exhausted VRAM (200 MiB) in a 240 GiB RAM host is refused.
        assert_eq!(
            evaluate_post_load_headroom(PostLoadHeadroomInput {
                host_mem_available_bytes: Some(240 * GIB),
                cuda_free_vram_bytes: Some(200 * MIB),
                ..base
            }),
            DeviceHeadroomDecision::Refused {
                available_bytes: 200 * MIB,
                floor_bytes: DEFAULT_DISCRETE_VRAM_FLOOR_MIB * MIB,
                shortfall_bytes: (DEFAULT_DISCRETE_VRAM_FLOOR_MIB - 200) * MIB,
            }
        );

        // Discrete GPU still enforces a reasonable host RAM sanity floor (1 GiB).
        assert_eq!(
            evaluate_post_load_headroom(PostLoadHeadroomInput {
                host_mem_available_bytes: Some(256 * MIB),
                cuda_free_vram_bytes: Some(10 * GIB),
                ..base
            }),
            DeviceHeadroomDecision::Refused {
                available_bytes: 256 * MIB,
                floor_bytes: DEFAULT_DISCRETE_HOST_FLOOR_MIB * MIB,
                shortfall_bytes: (DEFAULT_DISCRETE_HOST_FLOOR_MIB - 256) * MIB,
            }
        );

        // Discrete GPU with missing CUDA telemetry reports Unmeasurable.
        assert_eq!(
            evaluate_post_load_headroom(PostLoadHeadroomInput {
                cuda_free_vram_bytes: None,
                ..base
            }),
            DeviceHeadroomDecision::Unmeasurable
        );

        // Discrete GPU override converts VRAM shortfall to Overridden.
        assert_eq!(
            evaluate_post_load_headroom(PostLoadHeadroomInput {
                cuda_free_vram_bytes: Some(200 * MIB),
                override_selected: true,
                ..base
            }),
            DeviceHeadroomDecision::Overridden {
                shortfall_bytes: (DEFAULT_DISCRETE_VRAM_FLOOR_MIB - 200) * MIB,
            }
        );

        // UMA (Some(true) and None) enforces the 6.5 GiB host floor identically.
        for uma_flag in [Some(true), None] {
            assert_eq!(
                evaluate_post_load_headroom(PostLoadHeadroomInput {
                    is_integrated: uma_flag,
                    host_mem_available_bytes: Some(4 * GIB),
                    cuda_free_vram_bytes: Some(20 * GIB),
                    ..base
                }),
                DeviceHeadroomDecision::Refused {
                    available_bytes: 4 * GIB,
                    floor_bytes: QUALIFICATION_FLOOR_BYTES,
                    shortfall_bytes: QUALIFICATION_FLOOR_BYTES - 4 * GIB,
                }
            );
            assert_eq!(
                evaluate_post_load_headroom(PostLoadHeadroomInput {
                    is_integrated: uma_flag,
                    host_mem_available_bytes: Some(10 * GIB),
                    cuda_free_vram_bytes: Some(100 * MIB),
                    ..base
                }),
                DeviceHeadroomDecision::Pass {
                    headroom_bytes: 10 * GIB - QUALIFICATION_FLOOR_BYTES,
                }
            );
        }
    }
}
