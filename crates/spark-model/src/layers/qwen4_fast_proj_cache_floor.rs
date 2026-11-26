// SPDX-License-Identifier: AGPL-3.0-only

//! Fill policy for the optional FP8 tensor-scale weight cache
//! (`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1`).
//!
//! The cache converts each NVFP4 projection to e4m3 on its first prefill use
//! and keeps it resident, so by design it drives free memory *down* until a
//! stop line. The old stop line was `..._RESERVE_GB` (default 4, recipes 7)
//! measured on `free_memory()` = `max(cuMemGetInfo, MemAvailable)`: with the
//! recipes' 7 GiB that was 0.35 GiB above the 6,656 MiB qualification floor,
//! read on the optimistic metric. The stop line is now the larger of the
//! legacy reserve and the declared floor plus a margin, measured on the
//! host's real `MemAvailable`.

use anyhow::{Context, Result};
use spark_runtime::startup_ledger::{self, MIB};

/// `ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB`: how far above the
/// `MemAvailable` floor the cache stops filling. The margin covers the
/// request-time grow-only scratches (~0.3-0.5 GiB at W16K) and prefill
/// activation transients that land after the cache has filled.
pub const FLOOR_MARGIN_SELECTOR: &str = "ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB";
pub const DEFAULT_FLOOR_MARGIN_MIB: usize = 2048;

/// Where the cache stops admitting new weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheFillPolicy {
    /// `..._WEIGHT_CACHE_RESERVE_GB` in bytes (the pre-floor stop line).
    pub legacy_reserve_bytes: usize,
    /// `ATLAS_MEMAVAILABLE_FLOOR_MIB` in bytes.
    pub floor_bytes: usize,
    /// `ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB` in bytes.
    pub margin_bytes: usize,
}

impl CacheFillPolicy {
    /// `MemAvailable` must stay at or above this after a weight is cached.
    pub fn stop_line_bytes(&self) -> usize {
        self.legacy_reserve_bytes
            .max(self.floor_bytes.saturating_add(self.margin_bytes))
    }

    /// True when caching `bytes` more keeps `mem_available` on or above the
    /// stop line.
    pub fn admits(&self, mem_available: usize, bytes: usize) -> bool {
        mem_available.saturating_sub(bytes) >= self.stop_line_bytes()
    }

    /// One-line description for the startup ledger.
    pub fn summary(&self) -> String {
        format!(
            "fills at first prefill until MemAvailable - weight < {} MiB (max of reserve {} MiB, floor {} MiB + margin {} MiB)",
            self.stop_line_bytes() / MIB,
            self.legacy_reserve_bytes / MIB,
            self.floor_bytes / MIB,
            self.margin_bytes / MIB,
        )
    }
}

/// Parse the margin selector's value (`None` = default).
pub fn parse_margin_bytes(value: Option<&str>) -> Result<usize> {
    let mib = match value {
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{FLOOR_MARGIN_SELECTOR}={raw:?} is not a MiB count"))?,
        None => DEFAULT_FLOOR_MARGIN_MIB,
    };
    mib.checked_mul(MIB)
        .with_context(|| format!("{FLOOR_MARGIN_SELECTOR}={mib} overflows"))
}

const DEFAULT_SERVE_WATCH_FLOOR_MIB: usize = 12_288;

fn flux_or_atlas<'a>(
    get: &impl Fn(&str) -> Option<String>,
    flux_key: &'a str,
    atlas_key: &'a str,
) -> Option<(&'a str, String)> {
    get(flux_key)
        .map(|v| (flux_key, v))
        .or_else(|| get(atlas_key).map(|v| (atlas_key, v)))
}

/// Active `MemAvailable` serve refusal/abort floor from the in-process memory
/// watcher (`FLUX_MEM_FLOOR_MIB` / `ATLAS_MEM_FLOOR_MIB`, default `12_288 MiB`
/// unless `FLUX_MEM_WATCH=off|0` or `FLUX_MEM_FLOOR_ACTION_SERVE=log`).
/// Keeping `FP8_WEIGHT_CACHE` above `max(memavailable_floor, serve_watch_floor) + margin`
/// prevents optional weight caching from pushing `MemAvailable` below `12 GiB`
/// and tripping HTTP 503 `memory_floor` refusals on high-KV serves.
pub fn serve_watch_floor_bytes_from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<usize> {
    if let Some((key, raw)) = flux_or_atlas(&get, "FLUX_MEM_WATCH", "ATLAS_MEM_WATCH") {
        match raw.trim() {
            "on" | "1" => {}
            "off" | "0" => return Ok(0),
            other => anyhow::bail!("{key}={other:?}: expected on or off"),
        }
    }
    if let Some((key, raw)) = flux_or_atlas(
        &get,
        "FLUX_MEM_FLOOR_ACTION_SERVE",
        "ATLAS_MEM_FLOOR_ACTION_SERVE",
    ) {
        match raw.trim() {
            "refuse" | "abort" => {}
            "log" => return Ok(0),
            other => anyhow::bail!("{key}={other:?}: expected refuse, abort or log"),
        }
    }
    let Some((key, raw)) = flux_or_atlas(&get, "FLUX_MEM_FLOOR_MIB", "ATLAS_MEM_FLOOR_MIB") else {
        return Ok(DEFAULT_SERVE_WATCH_FLOOR_MIB * MIB);
    };
    raw.trim()
        .parse::<usize>()
        .ok()
        .and_then(|mib| mib.checked_mul(MIB))
        .with_context(|| format!("{key}={raw:?}: expected a MiB count"))
}

/// The policy from the environment, given the legacy reserve.
pub fn policy_from_env(legacy_reserve_bytes: usize) -> Result<CacheFillPolicy> {
    let ledger_floor_bytes = startup_ledger::memavailable_floor_bytes()?;
    let watch_floor_bytes = serve_watch_floor_bytes_from_lookup(|k| std::env::var(k).ok())?;
    let floor_bytes = ledger_floor_bytes.max(watch_floor_bytes);
    let margin_bytes = match std::env::var(FLOOR_MARGIN_SELECTOR) {
        Ok(value) => parse_margin_bytes(Some(&value))?,
        Err(std::env::VarError::NotPresent) => parse_margin_bytes(None)?,
        Err(error) => anyhow::bail!("{FLOOR_MARGIN_SELECTOR}: {error}"),
    };
    Ok(CacheFillPolicy {
        legacy_reserve_bytes,
        floor_bytes,
        margin_bytes,
    })
}

/// The one-line fill report logged at the cache's first reuse. `uncached`
/// weights run the per-call NVFP4 -> e4m3 conversion on every prefill pass.
pub fn fill_summary(
    cached: usize,
    uncached: usize,
    bytes: usize,
    stop_line_bytes: usize,
) -> String {
    format!(
        "FAST_PROJ_FP8_WEIGHT_CACHE_FILLED weights={cached} uncached={uncached} bytes={bytes} resident_mib={} stop_line_mib={} complete={}",
        bytes / MIB,
        stop_line_bytes / MIB,
        uncached == 0,
    )
}

/// Missing Linux telemetry disables this optional allocation. Only platforms
/// without Linux meminfo retain the historical backend-memory fallback.
pub fn cache_available_bytes(
    host_available: Option<usize>,
    is_linux: bool,
    backend_fallback: impl FnOnce() -> Result<usize>,
) -> Result<Option<usize>> {
    match host_available {
        Some(bytes) => Ok(Some(bytes)),
        None if is_linux => Ok(None),
        None => backend_fallback().map(Some),
    }
}

/// Maximum default VRAM reserve for discrete GPUs (`2 GiB`).
///
/// On discrete GPUs, cached FP8 projections live in dedicated VRAM (never host
/// RAM), so gating them on Linux `/proc/meminfo` vs the 14 GiB UMA floor both
/// causes instant CUDA OOM on 16–32 GB cards (where host RAM >> 14 GiB) and
/// falsely disables caching on 96 GB cards with tight host RAM.
pub const DISCRETE_VRAM_MAX_STOP_LINE_BYTES: usize = 2048 * MIB;

/// Compute the discrete-GPU VRAM stop line in bytes: `min(2 GiB, total_vram / 8)`.
#[must_use]
pub const fn discrete_vram_stop_line_bytes(total_vram_bytes: usize) -> usize {
    let eighth = total_vram_bytes / 8;
    if eighth < DISCRETE_VRAM_MAX_STOP_LINE_BYTES {
        eighth
    } else {
        DISCRETE_VRAM_MAX_STOP_LINE_BYTES
    }
}

/// True when caching `bytes` more into dedicated VRAM keeps `free_vram` at or
/// above [`discrete_vram_stop_line_bytes`].
#[must_use]
pub const fn discrete_vram_admits(free_vram: usize, total_vram: usize, bytes: usize) -> bool {
    match free_vram.checked_sub(bytes) {
        Some(remaining) => remaining >= discrete_vram_stop_line_bytes(total_vram),
        None => false,
    }
}

/// One-line ledger description for discrete-GPU VRAM cache gating.
#[must_use]
pub fn discrete_vram_summary(total_vram_bytes: usize) -> String {
    format!(
        "fills at first prefill until free VRAM - weight < {} MiB (discrete VRAM stop line = min(2048 MiB, total VRAM {} MiB / 8))",
        discrete_vram_stop_line_bytes(total_vram_bytes) / MIB,
        total_vram_bytes / MIB,
    )
}

/// Memory topology governing optional FP8 weight-cache admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMemoryTopology {
    /// Unified host/device memory (e.g. GB10 UMA): governed by host
    /// `MemAvailable` against [`CacheFillPolicy`].
    Unified,
    /// Discrete GPU with dedicated VRAM: governed by device `free_memory()`
    /// against `min(2 GiB, total_vram / 8)`.
    Discrete { total_vram_bytes: usize },
}

/// Admission verdict for one candidate FP8 weight-cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheAdmitDecision {
    Admit,
    UnmeasurableHost,
    BelowStopLine {
        available_bytes: usize,
        stop_line_bytes: usize,
        is_discrete_vram: bool,
    },
}

/// Evaluate whether an optional FP8 weight of `bytes` may be cached under
/// `topology`, consulting device free VRAM on discrete GPUs and host
/// `MemAvailable` on unified-memory GPUs.
pub fn evaluate_cache_admission(
    topology: CacheMemoryTopology,
    policy: &CacheFillPolicy,
    bytes: usize,
    host_available: Option<usize>,
    is_linux: bool,
    gpu_free: impl FnOnce() -> Result<usize>,
) -> Result<CacheAdmitDecision> {
    match topology {
        CacheMemoryTopology::Discrete { total_vram_bytes } => {
            let free_vram = gpu_free()?;
            let stop_line_bytes = discrete_vram_stop_line_bytes(total_vram_bytes);
            if discrete_vram_admits(free_vram, total_vram_bytes, bytes) {
                Ok(CacheAdmitDecision::Admit)
            } else {
                Ok(CacheAdmitDecision::BelowStopLine {
                    available_bytes: free_vram,
                    stop_line_bytes,
                    is_discrete_vram: true,
                })
            }
        }
        CacheMemoryTopology::Unified => {
            let Some(mem_available) = cache_available_bytes(host_available, is_linux, gpu_free)?
            else {
                return Ok(CacheAdmitDecision::UnmeasurableHost);
            };
            if policy.admits(mem_available, bytes) {
                Ok(CacheAdmitDecision::Admit)
            } else {
                Ok(CacheAdmitDecision::BelowStopLine {
                    available_bytes: mem_available,
                    stop_line_bytes: policy.stop_line_bytes(),
                    is_discrete_vram: false,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1024 * MIB;

    fn recipe_policy() -> CacheFillPolicy {
        // flux-v9-max-prefill.env: RESERVE_GB=7; default floor and margin.
        CacheFillPolicy {
            legacy_reserve_bytes: 7 * GIB,
            floor_bytes: 6656 * MIB,
            margin_bytes: 2048 * MIB,
        }
    }

    #[test]
    fn stop_line_is_floor_plus_margin_when_that_exceeds_the_legacy_reserve() {
        assert_eq!(recipe_policy().stop_line_bytes(), 8704 * MIB);
        let big_reserve = CacheFillPolicy {
            legacy_reserve_bytes: 10 * GIB,
            ..recipe_policy()
        };
        assert_eq!(big_reserve.stop_line_bytes(), 10 * GIB);
    }

    #[test]
    fn fill_stops_at_the_stop_line_not_at_the_legacy_reserve() {
        let policy = recipe_policy();
        let weight = 100 * MIB;
        // The old rule (free < bytes + 7 GiB) would have admitted this one.
        assert!(!policy.admits(7 * GIB + 200 * MIB, weight));
        assert!(!policy.admits(8704 * MIB + weight - 1, weight));
        assert!(policy.admits(8704 * MIB + weight, weight));
        assert!(policy.admits(20 * GIB, weight));
    }

    #[test]
    fn admits_never_underflows() {
        assert!(!recipe_policy().admits(0, 1));
        assert!(!recipe_policy().admits(10, 20));
    }

    #[test]
    fn margin_parses_with_a_default_and_rejects_garbage() {
        assert_eq!(parse_margin_bytes(None).unwrap(), 2048 * MIB);
        assert_eq!(parse_margin_bytes(Some("512")).unwrap(), 512 * MIB);
        assert!(parse_margin_bytes(Some("2G")).is_err());
    }

    #[test]
    fn summary_names_every_term() {
        let summary = recipe_policy().summary();
        assert!(summary.contains("8704 MiB"), "{summary}");
        assert!(summary.contains("reserve 7168 MiB"), "{summary}");
        assert!(summary.contains("floor 6656 MiB"), "{summary}");
    }

    #[test]
    fn fill_summary_reports_a_complete_and_a_partial_fill() {
        // Flash-Next: 36 x (qkvz 40 MiB + out 15 MiB) + 12 x (q 30 + k 1.25 + v 1.25 + o 15 MiB).
        let full = fill_summary(120, 0, 2_673_899_520, 8704 * MIB);
        assert!(
            full.starts_with("FAST_PROJ_FP8_WEIGHT_CACHE_FILLED weights=120 uncached=0 "),
            "{full}"
        );
        assert!(
            full.contains("resident_mib=2550 stop_line_mib=8704 complete=true"),
            "{full}"
        );
        let partial = fill_summary(96, 24, 2_178_441_216, 8704 * MIB);
        assert!(partial.contains("uncached=24"), "{partial}");
        assert!(partial.ends_with("complete=false"), "{partial}");
    }

    #[test]
    fn unavailable_linux_telemetry_skips_cache_without_backend_fallback() {
        assert_eq!(
            cache_available_bytes(None, true, || panic!("must not query backend memory")).unwrap(),
            None
        );
        assert_eq!(
            cache_available_bytes(Some(123), true, || panic!(
                "host measurement is authoritative"
            ))
            .unwrap(),
            Some(123)
        );
    }

    #[test]
    fn non_linux_telemetry_preserves_backend_fallback_and_errors() {
        assert_eq!(
            cache_available_bytes(None, false, || Ok(456)).unwrap(),
            Some(456)
        );
        assert!(
            cache_available_bytes(None, false, || anyhow::bail!("backend unavailable")).is_err()
        );
    }

    #[test]
    fn serve_watch_floor_protects_default_12gib_refusal_bar_and_respects_overrides() {
        let lookup = |pairs: &[(&str, &str)]| {
            let map: std::collections::HashMap<&str, &str> = pairs.iter().copied().collect();
            serve_watch_floor_bytes_from_lookup(move |k| map.get(k).map(|v| (*v).to_string()))
        };
        // Default watch (refuse @ 12,288 MiB) raises stop line to 14,336 MiB with 2,048 MiB margin.
        let default_watch = lookup(&[]).unwrap();
        assert_eq!(default_watch, 12_288 * MIB);
        let effective = CacheFillPolicy {
            legacy_reserve_bytes: 6 * GIB,
            floor_bytes: (6656 * MIB).max(default_watch),
            margin_bytes: 2048 * MIB,
        };
        assert_eq!(effective.stop_line_bytes(), 14_336 * MIB);

        // Log-only serve action or disabled watch contributes 0 so ledger floor governs alone.
        assert_eq!(
            lookup(&[("FLUX_MEM_FLOOR_ACTION_SERVE", "log")]).unwrap(),
            0
        );
        assert_eq!(lookup(&[("FLUX_MEM_WATCH", "off")]).unwrap(), 0);
        assert_eq!(lookup(&[("ATLAS_MEM_WATCH", "0")]).unwrap(), 0);

        // Explicit FLUX_ wins over ATLAS_ alias.
        assert_eq!(
            lookup(&[
                ("FLUX_MEM_FLOOR_MIB", "10240"),
                ("ATLAS_MEM_FLOOR_MIB", "8192")
            ])
            .unwrap(),
            10_240 * MIB
        );
        assert!(lookup(&[("FLUX_MEM_FLOOR_MIB", "12G")]).is_err());
        assert!(lookup(&[("FLUX_MEM_FLOOR_ACTION_SERVE", "bogus")]).is_err());
    }

    #[test]
    fn discrete_vram_cache_floor_governs_by_device_vram_not_host_meminfo() {
        // Stop line = min(2 GiB, total_vram / 8).
        assert_eq!(discrete_vram_stop_line_bytes(8 * GIB), GIB);
        assert_eq!(discrete_vram_stop_line_bytes(16 * GIB), 2 * GIB);
        assert_eq!(discrete_vram_stop_line_bytes(32 * GIB), 2 * GIB);
        assert_eq!(discrete_vram_stop_line_bytes(96 * GIB), 2 * GIB);
        assert_eq!(discrete_vram_stop_line_bytes(0), 0);
        // Never underflows or admits when weight exceeds free VRAM.
        assert!(!discrete_vram_admits(100, 0, 200));

        let uma_policy = CacheFillPolicy {
            legacy_reserve_bytes: 7 * GIB,
            floor_bytes: 12_288 * MIB,
            margin_bytes: 2_048 * MIB,
        };
        let weight = 100 * MIB;

        // 16 GB discrete GPU with 128 GiB host RAM but only 1.5 GiB free VRAM MUST refuse
        // (prevents instant CUDA OOM).
        assert_eq!(
            evaluate_cache_admission(
                CacheMemoryTopology::Discrete {
                    total_vram_bytes: 16 * GIB,
                },
                &uma_policy,
                weight,
                Some(128 * GIB),
                true,
                || Ok(1536 * MIB),
            )
            .unwrap(),
            CacheAdmitDecision::BelowStopLine {
                available_bytes: 1536 * MIB,
                stop_line_bytes: 2 * GIB,
                is_discrete_vram: true,
            }
        );

        // 96 GB discrete GPU with tight host RAM (4 GiB < 14 GiB UMA floor) but 48 GiB free VRAM
        // MUST admit (host MemAvailable does not gate VRAM allocations).
        assert_eq!(
            evaluate_cache_admission(
                CacheMemoryTopology::Discrete {
                    total_vram_bytes: 96 * GIB,
                },
                &uma_policy,
                weight,
                Some(4 * GIB),
                true,
                || Ok(48 * GIB),
            )
            .unwrap(),
            CacheAdmitDecision::Admit
        );

        // Unified memory (GB10) preserves exact host MemAvailable policy without calling gpu_free.
        assert_eq!(
            evaluate_cache_admission(
                CacheMemoryTopology::Unified,
                &uma_policy,
                weight,
                Some(10 * GIB),
                true,
                || panic!("UMA path must use host MemAvailable"),
            )
            .unwrap(),
            CacheAdmitDecision::BelowStopLine {
                available_bytes: 10 * GIB,
                stop_line_bytes: 14_336 * MIB,
                is_discrete_vram: false,
            }
        );

        let summary = discrete_vram_summary(96 * GIB);
        assert!(summary.contains("2048 MiB"), "{summary}");
        assert!(summary.contains("98304 MiB"), "{summary}");
    }
}
