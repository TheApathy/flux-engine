// SPDX-License-Identifier: AGPL-3.0-only

//! The one startup budget table and the fail-closed `MemAvailable` floor.
//!
//! Printed once at the end of load, after the factory has allocated every
//! pool: weights by family (from the store), every reserve term the
//! pre-load preflight sized, every phase the factory measured
//! (`spark_runtime::startup_ledger`), the FP8 weight cache's fill policy and
//! the CUDA-graph note, then the host's real `MemAvailable` against the
//! declared floor. Below the floor the server refuses to serve unless
//! `ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE=1` is set.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::{Context, Result};
use spark_runtime::startup_ledger::{
    self, DEFAULT_DISCRETE_HOST_FLOOR_MIB, DEFAULT_DISCRETE_VRAM_FLOOR_MIB, DeviceHeadroomDecision,
    FLOOR_OVERRIDE_SELECTOR, FLOOR_SELECTOR, LedgerEntry, MIB, PostLoadHeadroomInput,
    QUALIFICATION_FLOOR_BYTES,
};

/// Discrete GPUs at or below 24 GiB total VRAM use a tighter 256 MiB post-load
/// VRAM headroom floor so 16–24 GiB cards are not falsely refused at 90% utilization.
const SMALL_DISCRETE_VRAM_THRESHOLD_BYTES: usize = 24 * 1024 * MIB;
const SMALL_DISCRETE_VRAM_FLOOR_MIB: usize = 256;

pub(crate) fn discrete_vram_floor_bytes(total_vram_bytes: Option<usize>) -> usize {
    let mib = if total_vram_bytes
        .is_some_and(|total| total > 0 && total <= SMALL_DISCRETE_VRAM_THRESHOLD_BYTES)
    {
        SMALL_DISCRETE_VRAM_FLOOR_MIB
    } else {
        DEFAULT_DISCRETE_VRAM_FLOOR_MIB
    };
    mib * MIB
}

fn parse_discrete_host_floor_bytes(value: Option<&str>) -> Result<usize> {
    let mib = match value {
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .with_context(|| format!("{FLOOR_SELECTOR}={raw:?} is not a MiB count"))?,
        None => DEFAULT_DISCRETE_HOST_FLOOR_MIB,
    };
    mib.checked_mul(MIB)
        .with_context(|| format!("{FLOOR_SELECTOR}={mib} overflows"))
}

fn discrete_host_floor_bytes_from_env() -> Result<usize> {
    match std::env::var(FLOOR_SELECTOR) {
        Ok(value) => parse_discrete_host_floor_bytes(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_discrete_host_floor_bytes(None),
        Err(error) => anyhow::bail!("{FLOOR_SELECTOR}: {error}"),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LedgerRow {
    pub(crate) pool: String,
    pub(crate) detail: String,
    pub(crate) requested_bytes: Option<usize>,
    pub(crate) actual_delta_bytes: Option<i64>,
}

impl From<LedgerEntry> for LedgerRow {
    fn from(entry: LedgerEntry) -> Self {
        Self {
            pool: entry.pool,
            detail: entry.detail,
            requested_bytes: entry.requested_bytes,
            actual_delta_bytes: entry.actual_delta_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloorDecision {
    Pass { headroom_bytes: usize },
    Refused { shortfall_bytes: usize },
    Overridden { shortfall_bytes: usize },
    InvalidFloor,
    Unmeasurable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartupBudgetLedger {
    pub(crate) rows: Vec<LedgerRow>,
    pub(crate) mem_available_bytes: Option<usize>,
    pub(crate) floor_bytes: usize,
    pub(crate) override_selected: bool,
}

impl StartupBudgetLedger {
    pub(crate) fn decision(&self) -> FloorDecision {
        if self.floor_bytes < QUALIFICATION_FLOOR_BYTES && !self.override_selected {
            return FloorDecision::InvalidFloor;
        }
        match self.mem_available_bytes {
            None => FloorDecision::Unmeasurable,
            Some(available) if self.override_selected => FloorDecision::Overridden {
                shortfall_bytes: self.floor_bytes.saturating_sub(available),
            },
            Some(available) if available >= self.floor_bytes => FloorDecision::Pass {
                headroom_bytes: available - self.floor_bytes,
            },
            Some(available) => FloorDecision::Refused {
                shortfall_bytes: self.floor_bytes - available,
            },
        }
    }

    /// The table. Fixed-width columns in MiB; `-` where a value is unknown.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "STARTUP_BUDGET_LEDGER (MiB; actual = host MemAvailable delta around the phase, negative = reclaimed)"
        );
        let _ = writeln!(
            out,
            "  {:<32} {:>10} {:>10}  detail",
            "pool", "requested", "actual"
        );
        for row in &self.rows {
            let _ = writeln!(
                out,
                "  {:<32} {:>10} {:>10}  {}",
                row.pool,
                row.requested_bytes
                    .map_or_else(|| "-".to_string(), |b| (b / MIB).to_string()),
                row.actual_delta_bytes
                    .map_or_else(|| "-".to_string(), |b| (b / MIB as i64).to_string()),
                row.detail
            );
        }
        let _ = writeln!(
            out,
            "  {:<32} {:>10}",
            "MemAvailable now",
            self.mem_available_bytes
                .map_or_else(|| "-".to_string(), |b| (b / MIB).to_string())
        );
        let _ = writeln!(
            out,
            "  {:<32} {:>10}  {}",
            "floor",
            self.floor_bytes / MIB,
            FLOOR_SELECTOR
        );
        let verdict = match self.decision() {
            FloorDecision::Pass { headroom_bytes } => {
                format!("{:>10}  PASS", headroom_bytes / MIB)
            }
            FloorDecision::Refused { shortfall_bytes } => {
                format!(
                    "{:>10}  REFUSED (short by {} MiB)",
                    "-",
                    shortfall_bytes / MIB
                )
            }
            FloorDecision::Overridden { shortfall_bytes } => format!(
                "{:>10}  OVERRIDDEN by {FLOOR_OVERRIDE_SELECTOR}=1, NONQUALIFYING (short by {} MiB)",
                "-",
                shortfall_bytes / MIB
            ),
            FloorDecision::InvalidFloor => format!(
                "{:>10}  REFUSED (floor below qualification minimum without {FLOOR_OVERRIDE_SELECTOR}=1)",
                "-"
            ),
            FloorDecision::Unmeasurable if self.override_selected => format!(
                "{:>10}  UNMEASURABLE (/proc/meminfo unreadable), OVERRIDDEN by {FLOOR_OVERRIDE_SELECTOR}=1, NONQUALIFYING",
                "-"
            ),
            FloorDecision::Unmeasurable => {
                format!("{:>10}  UNMEASURABLE (/proc/meminfo unreadable)", "-")
            }
        };
        let _ = write!(out, "  {:<32} {verdict}", "headroom above floor");
        out
    }

    /// Fail closed: refuse below the floor, and refuse when the floor cannot
    /// be measured on Linux, unless the override selector is set.
    pub(crate) fn enforce(&self) -> Result<()> {
        match self.decision() {
            FloorDecision::Pass { .. } => Ok(()),
            FloorDecision::Overridden { shortfall_bytes } => {
                tracing::warn!(
                    "Startup budget: {}=1 selected, NONQUALIFYING run (declared floor {} MiB, shortfall {} MiB)",
                    FLOOR_OVERRIDE_SELECTOR,
                    self.floor_bytes / MIB,
                    shortfall_bytes / MIB,
                );
                Ok(())
            }
            FloorDecision::InvalidFloor => anyhow::bail!(
                "Startup budget refused: {}={} MiB is below the {} MiB qualification minimum; \
                 set {}=1 only for an explicitly NONQUALIFYING run.",
                FLOOR_SELECTOR,
                self.floor_bytes / MIB,
                QUALIFICATION_FLOOR_BYTES / MIB,
                FLOOR_OVERRIDE_SELECTOR
            ),
            FloorDecision::Refused { shortfall_bytes } => anyhow::bail!(
                "Startup budget refused: after load the host has {} MiB MemAvailable, {} MiB under \
                 the {} MiB floor ({}). Lower --max-seq-len / --max-prefill-tokens / \
                 --max-batch-size, free host memory, or set {}=1 only with a measured receipt.",
                self.mem_available_bytes.unwrap_or(0) / MIB,
                shortfall_bytes / MIB,
                self.floor_bytes / MIB,
                FLOOR_SELECTOR,
                FLOOR_OVERRIDE_SELECTOR
            ),
            FloorDecision::Unmeasurable if cfg!(target_os = "linux") && !self.override_selected => {
                anyhow::bail!(
                    "Startup budget refused: /proc/meminfo has no readable MemAvailable, so the {} MiB \
                     floor ({}) cannot be checked. Set {}=1 to serve without it.",
                    self.floor_bytes / MIB,
                    FLOOR_SELECTOR,
                    FLOOR_OVERRIDE_SELECTOR
                )
            }
            FloorDecision::Unmeasurable => {
                tracing::warn!(
                    "Startup budget: MemAvailable unreadable; floor not enforced, NONQUALIFYING run"
                );
                Ok(())
            }
        }
    }
}

/// Which family a weight tensor belongs to, from its HuggingFace name.
pub(crate) fn weight_family(name: &str) -> &'static str {
    if name.starts_with("mtp.") {
        "weights/mtp"
    } else if name.contains(".mlp.experts.") {
        "weights/routed-experts"
    } else if name.contains(".mlp.shared_expert") {
        "weights/shared-expert"
    } else if name.contains(".linear_attn.") {
        "weights/ssm"
    } else if name.contains(".self_attn.") {
        "weights/attention"
    } else if name.contains(".ple.") || name.contains(".ple_") {
        "weights/ple"
    } else if name.contains("embed_tokens") {
        "weights/embedding"
    } else if name.contains("lm_head") {
        "weights/lm-head"
    } else if name.contains("visual") || name.contains("vision") {
        "weights/vision"
    } else if name.contains(".mlp.") {
        "weights/dense-ffn"
    } else {
        "weights/other"
    }
}

/// One row per weight family, sorted by name, from `(tensor name, bytes)`.
pub(crate) fn weight_family_rows<'a>(
    tensors: impl Iterator<Item = (&'a str, usize)>,
) -> Vec<LedgerRow> {
    let mut families: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for (name, bytes) in tensors {
        let slot = families.entry(weight_family(name)).or_default();
        slot.0 += bytes;
        slot.1 += 1;
    }
    families
        .into_iter()
        .map(|(family, (bytes, count))| LedgerRow {
            pool: family.to_string(),
            detail: format!("{count} tensors (logical store bytes)"),
            requested_bytes: Some(bytes),
            actual_delta_bytes: None,
        })
        .collect()
}

/// Attach a preflight-requested size to the factory row of the same pool
/// (filling only an unknown `requested`), or add a requested-only row.
pub(crate) fn merge_requested(
    rows: &mut Vec<LedgerRow>,
    pool: &str,
    requested_bytes: usize,
    detail: &str,
) {
    if let Some(row) = rows
        .iter_mut()
        .find(|row| row.pool == pool && row.requested_bytes.is_none())
    {
        row.requested_bytes = Some(requested_bytes);
        if row.detail.is_empty() {
            row.detail = detail.to_string();
        }
        return;
    }
    rows.push(LedgerRow {
        pool: pool.to_string(),
        detail: detail.to_string(),
        requested_bytes: Some(requested_bytes),
        actual_delta_bytes: None,
    });
}

/// Assemble the ledger from the live store, the preflight breakdown and the
/// factory's measured phases, then read the host floor state.
pub(crate) fn assemble(
    weights: Vec<LedgerRow>,
    preflight_requested: &[(&str, usize, &str)],
    measured: Vec<LedgerEntry>,
    extra: Vec<LedgerRow>,
    mem_available_bytes: Option<usize>,
    floor_bytes: usize,
    override_selected: bool,
) -> StartupBudgetLedger {
    let mut rows = weights;
    rows.extend(measured.into_iter().map(LedgerRow::from));
    for (pool, bytes, detail) in preflight_requested {
        merge_requested(&mut rows, pool, *bytes, detail);
    }
    rows.extend(extra);
    StartupBudgetLedger {
        rows,
        mem_available_bytes,
        floor_bytes,
        override_selected,
    }
}

/// End-of-load startup budget ledger for discrete GPUs (`SM120` / `SM103`),
/// enforcing dedicated VRAM headroom via [`startup_ledger::evaluate_post_load_headroom`]
/// instead of GB10's 6656 MiB UMA host `MemAvailable` qualification bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiscreteBudgetLedger {
    pub(crate) rows: Vec<LedgerRow>,
    pub(crate) vram_free_bytes: Option<usize>,
    pub(crate) host_mem_available_bytes: Option<usize>,
    pub(crate) vram_floor_bytes: usize,
    pub(crate) host_floor_bytes: usize,
    pub(crate) override_selected: bool,
}

impl DiscreteBudgetLedger {
    pub(crate) fn headroom_decision(&self) -> DeviceHeadroomDecision {
        startup_ledger::evaluate_post_load_headroom(PostLoadHeadroomInput {
            is_integrated: Some(false),
            host_mem_available_bytes: self.host_mem_available_bytes,
            cuda_free_vram_bytes: self.vram_free_bytes,
            uma_floor_bytes: QUALIFICATION_FLOOR_BYTES,
            discrete_vram_floor_bytes: self.vram_floor_bytes,
            discrete_host_floor_bytes: self.host_floor_bytes,
            override_selected: self.override_selected,
        })
    }

    /// Map discrete VRAM headroom evaluation onto [`FloorDecision`].
    /// Never returns [`FloorDecision::InvalidFloor`], because discrete GPUs
    /// intentionally enforce a 256–512 MiB VRAM floor rather than GB10's 6656 MiB UMA bar.
    pub(crate) fn decision(&self) -> FloorDecision {
        match self.headroom_decision() {
            DeviceHeadroomDecision::Pass { headroom_bytes } => {
                FloorDecision::Pass { headroom_bytes }
            }
            DeviceHeadroomDecision::Refused {
                shortfall_bytes, ..
            } => FloorDecision::Refused { shortfall_bytes },
            DeviceHeadroomDecision::Overridden { shortfall_bytes } => {
                FloorDecision::Overridden { shortfall_bytes }
            }
            DeviceHeadroomDecision::Unmeasurable => FloorDecision::Unmeasurable,
        }
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "STARTUP_BUDGET_LEDGER (MiB; actual = discrete GPU VRAM delta around the phase, negative = reclaimed)"
        );
        let _ = writeln!(
            out,
            "  {:<32} {:>10} {:>10}  detail",
            "pool", "requested", "actual"
        );
        for row in &self.rows {
            let _ = writeln!(
                out,
                "  {:<32} {:>10} {:>10}  {}",
                row.pool,
                row.requested_bytes
                    .map_or_else(|| "-".to_string(), |b| (b / MIB).to_string()),
                row.actual_delta_bytes
                    .map_or_else(|| "-".to_string(), |b| (b / MIB as i64).to_string()),
                row.detail
            );
        }
        let _ = writeln!(
            out,
            "  {:<32} {:>10}",
            "VRAM free now",
            self.vram_free_bytes
                .map_or_else(|| "-".to_string(), |b| (b / MIB).to_string())
        );
        let _ = writeln!(
            out,
            "  {:<32} {:>10}  host RAM sanity floor {} MiB",
            "MemAvailable now",
            self.host_mem_available_bytes
                .map_or_else(|| "-".to_string(), |b| (b / MIB).to_string()),
            self.host_floor_bytes / MIB,
        );
        let _ = writeln!(
            out,
            "  {:<32} {:>10}  discrete VRAM floor",
            "floor",
            self.vram_floor_bytes / MIB,
        );
        let verdict = match self.headroom_decision() {
            DeviceHeadroomDecision::Pass { headroom_bytes } => {
                format!("{:>10}  PASS", headroom_bytes / MIB)
            }
            DeviceHeadroomDecision::Refused {
                shortfall_bytes, ..
            } => format!(
                "{:>10}  REFUSED (short by {} MiB)",
                "-",
                shortfall_bytes / MIB
            ),
            DeviceHeadroomDecision::Overridden { shortfall_bytes } => format!(
                "{:>10}  OVERRIDDEN by {FLOOR_OVERRIDE_SELECTOR}=1, NONQUALIFYING (short by {} MiB)",
                "-",
                shortfall_bytes / MIB
            ),
            DeviceHeadroomDecision::Unmeasurable if self.override_selected => format!(
                "{:>10}  UNMEASURABLE (VRAM telemetry unreadable), OVERRIDDEN by {FLOOR_OVERRIDE_SELECTOR}=1, NONQUALIFYING",
                "-"
            ),
            DeviceHeadroomDecision::Unmeasurable => {
                format!("{:>10}  UNMEASURABLE (VRAM telemetry unreadable)", "-")
            }
        };
        let _ = write!(out, "  {:<32} {verdict}", "headroom above floor");
        out
    }

    pub(crate) fn enforce(&self) -> Result<()> {
        match self.headroom_decision() {
            DeviceHeadroomDecision::Pass { .. } => Ok(()),
            DeviceHeadroomDecision::Overridden { shortfall_bytes } => {
                tracing::warn!(
                    "Startup budget (discrete GPU): {}=1 selected, NONQUALIFYING run \
                     (VRAM floor {} MiB, host floor {} MiB, shortfall {} MiB)",
                    FLOOR_OVERRIDE_SELECTOR,
                    self.vram_floor_bytes / MIB,
                    self.host_floor_bytes / MIB,
                    shortfall_bytes / MIB,
                );
                Ok(())
            }
            DeviceHeadroomDecision::Refused {
                available_bytes,
                floor_bytes,
                shortfall_bytes,
            } => {
                let vram_short = self
                    .vram_free_bytes
                    .is_some_and(|vram| vram < self.vram_floor_bytes);
                if vram_short {
                    anyhow::bail!(
                        "Startup budget refused: after load the discrete GPU has {} MiB free VRAM, \
                         {} MiB under the {} MiB discrete VRAM floor. Lower --max-seq-len / \
                         --max-prefill-tokens / --max-batch-size, or set {}=1 only with a measured receipt.",
                        available_bytes / MIB,
                        shortfall_bytes / MIB,
                        floor_bytes / MIB,
                        FLOOR_OVERRIDE_SELECTOR,
                    )
                } else {
                    anyhow::bail!(
                        "Startup budget refused: after load the host has {} MiB MemAvailable, \
                         {} MiB under the {} MiB discrete host RAM sanity floor ({}). Free host \
                         memory or set {}=1 only with a measured receipt.",
                        available_bytes / MIB,
                        shortfall_bytes / MIB,
                        floor_bytes / MIB,
                        FLOOR_SELECTOR,
                        FLOOR_OVERRIDE_SELECTOR,
                    )
                }
            }
            DeviceHeadroomDecision::Unmeasurable if !self.override_selected => anyhow::bail!(
                "Startup budget refused: discrete GPU free VRAM is unreadable, so the {} MiB \
                 VRAM floor cannot be checked. Set {}=1 to serve without it.",
                self.vram_floor_bytes / MIB,
                FLOOR_OVERRIDE_SELECTOR,
            ),
            DeviceHeadroomDecision::Unmeasurable => {
                tracing::warn!(
                    "Startup budget (discrete GPU): VRAM telemetry unreadable; floor not enforced, NONQUALIFYING run"
                );
                Ok(())
            }
        }
    }
}

/// Print the table once and enforce the floor.
pub(crate) fn audit_startup_budget(
    config: &atlas_core::config::ModelConfig,
    store: &spark_runtime::weights::WeightStore,
    device_store_bytes: usize,
    breakdown: &super::preflight::ReserveBreakdown,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<()> {
    let mut weights = weight_family_rows(
        store
            .names()
            .filter_map(|name| store.get(name).ok().map(|w| (name, w.byte_size()))),
    );
    if weights.is_empty() && device_store_bytes > 0 {
        weights.push(LedgerRow {
            pool: "weights/device-store".to_string(),
            detail: "owned device store (GLM-5.3 EXL3/GGUF)".to_string(),
            requested_bytes: Some(device_store_bytes),
            actual_delta_bytes: None,
        });
    }
    let preflight_requested: Vec<(&str, usize, &str)> = vec![
        (
            "model-state-pools",
            breakdown.ssm_pool_bytes + breakdown.ssm_snapshot_bytes + breakdown.ple_history_bytes,
            "SSM pool + snapshot + PLE history (preflight)",
        ),
        (
            "gdn-two-phase-reserve",
            breakdown.gdn_two_phase_bytes,
            "reserve, allocated lazily by chunked GDN prefill",
        ),
        (
            "cuda-headroom-reserve",
            breakdown.cuda_headroom_bytes,
            "reserve, not an allocation (speculative CUDA headroom)",
        ),
        (
            "qsa-side-cache",
            breakdown.qsa_side_cache_bytes,
            "reserve, allocated lazily",
        ),
        (
            "qsa-fp8-shadows",
            breakdown.qsa_fp8_shadow_bytes,
            "E4M3 K/V shadows over the target KV pool",
        ),
        (
            "batched-k5-verify-lanes",
            breakdown.batched_k5_lane_bytes,
            "reserve, batched-K5 GDN verify lanes",
        ),
    ];
    let mut extra = Vec::new();
    if config.is_qwen4_exp() {
        extra.push(LedgerRow {
            pool: "fp8-weight-cache".to_string(),
            detail: spark_model::layers::qwen4_fast_proj::fp8_weight_cache_ledger_summary()
                .unwrap_or_else(|| "off".to_string()),
            requested_bytes: None,
            actual_delta_bytes: None,
        });
        let graphs = if std::env::var("ATLAS_QWEN4_VERIFY_GRAPH").ok().as_deref() == Some("1") {
            "driver-internal exec memory, captured at first verify (ATLAS_QWEN4_VERIFY_GRAPH=1)"
        } else {
            "off"
        };
        extra.push(LedgerRow {
            pool: "cuda-graphs".to_string(),
            detail: graphs.to_string(),
            requested_bytes: None,
            actual_delta_bytes: None,
        });
    }
    let is_discrete = !gpu.is_unified_memory() || startup_ledger::is_discrete_device();
    if is_discrete {
        let mut rows = weights;
        rows.extend(startup_ledger::entries().into_iter().map(LedgerRow::from));
        for (pool, bytes, detail) in &preflight_requested {
            merge_requested(&mut rows, pool, *bytes, detail);
        }
        rows.extend(extra);
        let total_vram = gpu
            .total_memory()
            .ok()
            .filter(|&b| b > 0)
            .or_else(|| atlas_core::device::active_device_props().map(|p| p.total_mem_bytes));
        let vram_free = gpu
            .free_memory()
            .ok()
            .or_else(startup_ledger::available_device_memory_bytes);
        let ledger = DiscreteBudgetLedger {
            rows,
            vram_free_bytes: vram_free,
            host_mem_available_bytes: startup_ledger::mem_available_bytes(),
            vram_floor_bytes: discrete_vram_floor_bytes(total_vram),
            host_floor_bytes: discrete_host_floor_bytes_from_env()?,
            override_selected: startup_ledger::floor_override_selected(),
        };
        let decision = format!("{:?}", ledger.decision());
        tracing::info!(
            target: "atlas_startup_budget",
            vram_free_mib = ?ledger.vram_free_bytes.map(|b| b / MIB),
            mem_available_mib = ?ledger.host_mem_available_bytes.map(|b| b / MIB),
            floor_mib = ledger.vram_floor_bytes / MIB,
            decision = %decision,
            "{}",
            ledger.render()
        );
        if let Some(receipt) = spark_runtime::alloc_receipt::session_receipt() {
            tracing::info!(target: "atlas_startup_budget", "{}", receipt.line());
        }
        return ledger.enforce();
    }
    let ledger = assemble(
        weights,
        &preflight_requested,
        startup_ledger::entries(),
        extra,
        startup_ledger::mem_available_bytes(),
        startup_ledger::memavailable_floor_bytes()?,
        startup_ledger::floor_override_selected(),
    );
    let (mem_available_mib, decision) = (
        ledger.mem_available_bytes.map(|b| b / MIB),
        format!("{:?}", ledger.decision()),
    );
    tracing::info!(
        target: "atlas_startup_budget",
        mem_available_mib = ?mem_available_mib,
        floor_mib = ledger.floor_bytes / MIB,
        decision = %decision,
        "{}",
        ledger.render()
    );
    // Allocation receipt for the whole load: the `STARTUP_ALLOC_RECEIPT`
    // phase lines above, summed, plus what ran outside every phase.
    if let Some(receipt) = spark_runtime::alloc_receipt::session_receipt() {
        tracing::info!(target: "atlas_startup_budget", "{}", receipt.line());
    }
    ledger.enforce()
}

#[cfg(test)]
#[path = "startup_ledger_tests.rs"]
mod tests;

#[cfg(test)]
mod discrete_tests {
    use super::{
        DiscreteBudgetLedger, FloorDecision, MIB, discrete_vram_floor_bytes,
        parse_discrete_host_floor_bytes,
    };

    const GIB: usize = 1024 * MIB;

    #[test]
    fn discrete_vram_floor_scales_for_small_and_large_gpus() {
        // <= 24 GiB cards (RTX 5060 Ti / 5070 Ti / 5080 16 GB, 24 GB): 256 MiB floor.
        assert_eq!(discrete_vram_floor_bytes(Some(16 * GIB)), 256 * MIB);
        assert_eq!(discrete_vram_floor_bytes(Some(24 * GIB)), 256 * MIB);
        // > 24 GiB cards (RTX 5090 32 GB, 72 GB, RTX PRO 6000 96 GB, B300 288 GB): 512 MiB floor.
        assert_eq!(discrete_vram_floor_bytes(Some(32 * GIB)), 512 * MIB);
        assert_eq!(discrete_vram_floor_bytes(Some(72 * GIB)), 512 * MIB);
        assert_eq!(discrete_vram_floor_bytes(Some(96 * GIB)), 512 * MIB);
        assert_eq!(discrete_vram_floor_bytes(Some(288 * GIB)), 512 * MIB);
        assert_eq!(discrete_vram_floor_bytes(None), 512 * MIB);
    }

    #[test]
    fn discrete_gpu_passes_with_low_host_ram_and_never_returns_invalid_floor() {
        // 16 GB GPU @ 90% util (1.6 GiB free VRAM) in a 16 GB RAM host with 3 GiB MemAvailable
        // (< GB10's 6656 MiB UMA floor): must PASS, never InvalidFloor or UMA refusal!
        for (total_gib, free_mib, host_mib) in [
            (16, 1_600, 3_072),
            (32, 1_024, 4_096),
            (72, 2_048, 4_096),
            (96, 2_048, 4_096),
        ] {
            let floor = discrete_vram_floor_bytes(Some(total_gib * GIB));
            let ledger = DiscreteBudgetLedger {
                rows: Vec::new(),
                vram_free_bytes: Some(free_mib * MIB),
                host_mem_available_bytes: Some(host_mib * MIB),
                vram_floor_bytes: floor,
                host_floor_bytes: parse_discrete_host_floor_bytes(None).unwrap(),
                override_selected: false,
            };
            assert_eq!(
                ledger.decision(),
                FloorDecision::Pass {
                    headroom_bytes: free_mib * MIB - floor,
                },
                "total_gib={total_gib} free_mib={free_mib} host_mib={host_mib}"
            );
            assert!(ledger.enforce().is_ok());
            let rendered = ledger.render();
            assert!(rendered.contains("VRAM free now"), "{rendered}");
            assert!(rendered.contains("PASS"), "{rendered}");
            assert!(!rendered.contains("qualification minimum"), "{rendered}");
        }
    }

    #[test]
    fn discrete_gpu_refuses_exhausted_vram_or_host_ram_unless_overridden() {
        // 32 GB GPU with only 200 MiB free VRAM (< 512 MiB floor) in a 240 GiB RAM host.
        let vram_exhausted = DiscreteBudgetLedger {
            rows: Vec::new(),
            vram_free_bytes: Some(200 * MIB),
            host_mem_available_bytes: Some(240 * GIB),
            vram_floor_bytes: discrete_vram_floor_bytes(Some(32 * GIB)),
            host_floor_bytes: 1024 * MIB,
            override_selected: false,
        };
        assert_eq!(
            vram_exhausted.decision(),
            FloorDecision::Refused {
                shortfall_bytes: 312 * MIB,
            }
        );
        let err = vram_exhausted.enforce().unwrap_err().to_string();
        assert!(err.contains("free VRAM"), "{err}");

        let mut overridden = vram_exhausted.clone();
        overridden.override_selected = true;
        assert_eq!(
            overridden.decision(),
            FloorDecision::Overridden {
                shortfall_bytes: 312 * MIB,
            }
        );
        assert!(overridden.enforce().is_ok());

        // Host RAM below 1 GiB sanity floor still refuses cleanly.
        let host_exhausted = DiscreteBudgetLedger {
            rows: Vec::new(),
            vram_free_bytes: Some(12 * GIB),
            host_mem_available_bytes: Some(256 * MIB),
            vram_floor_bytes: 512 * MIB,
            host_floor_bytes: 1024 * MIB,
            override_selected: false,
        };
        assert_eq!(
            host_exhausted.decision(),
            FloorDecision::Refused {
                shortfall_bytes: 768 * MIB,
            }
        );
        let err = host_exhausted.enforce().unwrap_err().to_string();
        assert!(err.contains("host RAM sanity floor"), "{err}");

        // Missing VRAM telemetry fails closed unless overridden.
        let unmeasurable = DiscreteBudgetLedger {
            vram_free_bytes: None,
            ..vram_exhausted
        };
        assert_eq!(unmeasurable.decision(), FloorDecision::Unmeasurable);
        assert!(unmeasurable.enforce().is_err());

        // Explicit discrete host floor parser rejects malformed input without UMA 6656 MiB bar.
        assert_eq!(
            parse_discrete_host_floor_bytes(Some("512")).unwrap(),
            512 * MIB
        );
        assert!(parse_discrete_host_floor_bytes(Some("bad")).is_err());
    }
}
