// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-load fit check (`FLUX_FIT_CHECK`, default `warn`; `refuse` is opt-in
//! until the estimate is calibrated against two real launches).
//!
//! Before the weight store allocates anything, estimate the serve peak from
//! the checkpoint and the preflight reserve terms and refuse when
//! `need + FLUX_MEM_FLOOR_MIB > MemAvailable`. On GB10 an over-allocation
//! reboots the host or dies in `NV_ERR_NO_MEMORY` halfway through load, so a
//! refusal with exact numbers is the cheap outcome.
//!
//! Every term prints with its source so a wrong term is visible:
//! - `modelled`: computed from config/CLI (KV, arena, pools, QSA, the
//!   construction overhead model, the Qwen4 transient, the LM head).
//! - `measured`: a constant or ratio calibrated from GB10 memory measurements
//!   (store rounding, construction correction, serve growth to the
//!   first-prefill peak, residual, DFlash drafter, Qwen4 prefill scratch).
//!
//! Calibrated for Flash-Next (`qwen4_exp`) and the hybrid dense 27B family.
//! Other families get the 27B constants and `warn` instead of `refuse`
//! (`fit_estimate_tests.rs` pins both reference peaks).
//!
//! Pre-load counterpart to `ATLAS_FIT_CHECK`, running before the weight store
//! allocates.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Result, bail};
use spark_runtime::startup_ledger::MIB;

use crate::cli;
use atlas_core::config::ModelConfig;

pub(crate) const SELECTOR: &str = "FLUX_FIT_CHECK";
/// The `ATLAS_` alias `main` mirrors for `FLUX_FIT_CHECK`.
const SELECTOR_ALIAS: &str = "ATLAS_FIT_CHECK";
const PPM: i128 = 1_000_000;

/// Measured Flash-Next store overhead: 74.25 GiB over 72.17 logical (rounding + pinned).
const QWEN4_STORE_OVERHEAD_PPM: i128 = 28_820;
/// Measured 27B store overhead: 21.61 GiB over 20.41 logical (2194 allocations).
const DENSE_STORE_OVERHEAD_PPM: i128 = 58_795;
/// Measured 27B construction ratio: 45.52 GiB measured over 38.50 modelled.
const CONSTRUCTION_CORRECTION_PPM: i128 = 1_182_338;
/// Measured Flash-Next `target-layers` floor: 3.99 GiB.
const QWEN4_TARGET_LAYERS_BYTES: i64 = 4_086 * MIB as i64;
/// Measured Flash-Next persistent `prefill-scratch`: 0.30 GiB (`never_freed=1`).
const QWEN4_PREFILL_SCRATCH_BYTES: i64 = 307 * MIB as i64;
/// Measured ready-to-first-prefill growth: Flash-Next +4.17 GiB / 27B +5.87 GiB.
/// The Flash-Next figure excludes its FP8 weight cache (2.49 GiB), which is its own term.
const QWEN4_SERVE_GROWTH_BYTES: i64 = 1_720 * MIB as i64;
/// Measured opt-in FP8 e4m3 projection-weight cache footprint (`2550 MiB`).
const QWEN4_FP8_WEIGHT_CACHE_BYTES: i64 = 2_550 * MIB as i64;
const FP8_WEIGHT_CACHE_SELECTORS: [&str; 2] = [
    "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE",
    "ATLAS_QWEN4_WEIGHT_CACHE",
];
/// Upper bound on a safetensors JSON header; larger is a corrupt file.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const DENSE_SERVE_GROWTH_BYTES: i64 = 6_011 * MIB as i64;
/// Measured residual rows: host RSS, context, drafter rounding.
const QWEN4_RESIDUAL_BYTES: i64 = 215 * MIB as i64;
const DENSE_RESIDUAL_BYTES: i64 = 737 * MIB as i64;
/// Measured DFlash drafter footprint: 5.45 GiB.
const DFLASH_DRAFTER_BYTES: i64 = 5_581 * MIB as i64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    /// Flash-Next (`qwen4_exp`).
    Qwen4,
    /// Hybrid GDN/attention dense (Qwen3.8-27B).
    HybridDense,
    /// Not calibrated: 27B constants, warn only.
    Other,
}

impl Family {
    pub(crate) fn of(config: &ModelConfig) -> Self {
        if config.is_qwen4_exp() {
            Self::Qwen4
        } else if config.num_ssm_layers() > 0 && config.num_experts == 0 {
            Self::HybridDense
        } else {
            Self::Other
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TermSource {
    Measured,
    Modelled,
}

impl TermSource {
    fn name(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Modelled => "modelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FitTerm {
    pub(crate) name: &'static str,
    /// Signed: a release (Qwen4 BF16 LM head) is a credit.
    pub(crate) bytes: i64,
    pub(crate) source: TermSource,
}

/// Inputs known before any allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FitFacts {
    pub(crate) family: Family,
    /// Tensor bytes the loader keeps resident: every safetensors tensor less
    /// the serving skip set (the offloaded PLE n-gram table).
    pub(crate) checkpoint_bytes: usize,
    /// `mtp.*` bytes released before the transpose when speculation is off
    /// (`QWEN4_RELEASE_UNUSED_MTP_EARLY`); 0 otherwise.
    pub(crate) mtp_release_bytes: usize,
    /// The opt-in FP8 weight cache fills at the first prefill.
    pub(crate) fp8_weight_cache: bool,
    /// `construction_overhead_bytes(config).total()`.
    pub(crate) construction_modelled_bytes: usize,
    /// `qwen4_construction_transient_bytes` (routed slab, MTP layer, NVFP4 head).
    pub(crate) qwen4_transient_bytes: usize,
    pub(crate) lm_head_elems: usize,
    /// SSM pool + snapshot + PLE history from the preflight.
    pub(crate) pools_bytes: usize,
    pub(crate) arena_bytes: usize,
    /// KV pool for `max_seq_len x max_batch_size`, capped like the factory.
    pub(crate) kv_bytes: usize,
    /// QSA side cache, E4M3 shadows, and batched-K5 verify lanes from preflight.
    pub(crate) qsa_bytes: usize,
    pub(crate) dflash: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FitEstimate {
    pub(crate) terms: Vec<FitTerm>,
    pub(crate) mem_available_bytes: usize,
    pub(crate) floor_bytes: usize,
    pub(crate) is_discrete: bool,
}

fn scale_ppm(bytes: usize, ppm: i128) -> i64 {
    i64::try_from(bytes as i128 * ppm / PPM).unwrap_or(i64::MAX)
}

fn signed(bytes: usize) -> i64 {
    i64::try_from(bytes).unwrap_or(i64::MAX)
}

fn term(name: &'static str, bytes: i64, source: TermSource) -> FitTerm {
    FitTerm {
        name,
        bytes,
        source,
    }
}

fn store_term(f: &FitFacts) -> FitTerm {
    let ppm = match f.family {
        Family::Qwen4 => QWEN4_STORE_OVERHEAD_PPM,
        Family::HybridDense | Family::Other => DENSE_STORE_OVERHEAD_PPM,
    };
    term(
        "weights-store",
        scale_ppm(f.checkpoint_bytes, PPM + ppm),
        TermSource::Measured,
    )
}

fn construction_term(f: &FitFacts) -> FitTerm {
    // The correction ratio is the 27B's; FN uses its measured row directly.
    let bytes = match f.family {
        Family::Qwen4 => QWEN4_TARGET_LAYERS_BYTES,
        Family::HybridDense | Family::Other => {
            scale_ppm(f.construction_modelled_bytes, CONSTRUCTION_CORRECTION_PPM)
        }
    };
    term("construction", bytes, TermSource::Measured)
}

/// Qwen4 frees the checkpoint BF16 head (its NVFP4 copy is in the transient);
/// dense keeps a BF16 dequant plus the re-quantized NVFP4 head.
fn lm_head_term(f: &FitFacts) -> FitTerm {
    let bf16 = signed(f.lm_head_elems.saturating_mul(2));
    let bytes = match f.family {
        Family::Qwen4 => -bf16,
        Family::HybridDense | Family::Other => {
            bf16.saturating_add(signed(f.lm_head_elems.saturating_mul(9) / 16))
        }
    };
    term("lm-head", bytes, TermSource::Modelled)
}

fn family_constants(f: &FitFacts) -> Vec<FitTerm> {
    let (growth, residual) = match f.family {
        Family::Qwen4 => (QWEN4_SERVE_GROWTH_BYTES, QWEN4_RESIDUAL_BYTES),
        Family::HybridDense | Family::Other => (DENSE_SERVE_GROWTH_BYTES, DENSE_RESIDUAL_BYTES),
    };
    let scratch = (f.family == Family::Qwen4).then(|| {
        term(
            "prefill-scratch",
            QWEN4_PREFILL_SCRATCH_BYTES,
            TermSource::Measured,
        )
    });
    let fp8_cache = (f.family == Family::Qwen4 && f.fp8_weight_cache).then(|| {
        term(
            "fp8-weight-cache",
            QWEN4_FP8_WEIGHT_CACHE_BYTES,
            TermSource::Measured,
        )
    });
    let mtp_release = (f.mtp_release_bytes > 0).then(|| {
        term(
            "mtp-release",
            -signed(f.mtp_release_bytes),
            TermSource::Modelled,
        )
    });
    let drafter = f
        .dflash
        .then(|| term("dflash-drafter", DFLASH_DRAFTER_BYTES, TermSource::Measured));
    [
        scratch,
        fp8_cache,
        mtp_release,
        drafter,
        Some(term("residual", residual, TermSource::Measured)),
        Some(term("serve-growth", growth, TermSource::Measured)),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Discrete GPU VRAM headroom floor for pre-load fit checks:
/// `min(512 MiB, 5% of total VRAM)`.
pub(crate) fn discrete_fit_floor_bytes(total_vram_bytes: usize) -> usize {
    let cap = spark_runtime::startup_ledger::DEFAULT_DISCRETE_VRAM_FLOOR_MIB * MIB;
    cap.min(total_vram_bytes / 20)
}

/// The estimate for `facts` against discrete GPU VRAM (`SM120` / `SM103`).
///
/// Omits GB10 UMA host-only constants (`DENSE_SERVE_GROWTH_BYTES = 6011 MiB`,
/// host RSS residuals, UMA store-rounding inflation, and the 12 GiB UMA floor)
/// so 16–96 GB discrete GPUs are never falsely refused on host-RAM heuristics.
pub(crate) fn estimate_discrete(
    f: &FitFacts,
    vram_free_bytes: usize,
    vram_total_bytes: usize,
) -> FitEstimate {
    let store = term(
        "weights-store",
        signed(f.checkpoint_bytes),
        TermSource::Modelled,
    );
    let construction = match f.family {
        Family::Qwen4 => term(
            "construction",
            QWEN4_TARGET_LAYERS_BYTES,
            TermSource::Measured,
        ),
        Family::HybridDense | Family::Other => term(
            "construction",
            signed(f.construction_modelled_bytes),
            TermSource::Modelled,
        ),
    };
    let modelled = [
        ("qwen4-transient", f.qwen4_transient_bytes),
        ("model-state-pools", f.pools_bytes),
        ("buffer-arena", f.arena_bytes),
        ("kv-cache", f.kv_bytes),
        ("qsa-caches", f.qsa_bytes),
    ]
    .into_iter()
    .filter(|(_, bytes)| *bytes > 0)
    .map(|(name, bytes)| term(name, signed(bytes), TermSource::Modelled));
    let extras = [
        (f.family == Family::Qwen4).then(|| {
            term(
                "prefill-scratch",
                QWEN4_PREFILL_SCRATCH_BYTES,
                TermSource::Measured,
            )
        }),
        (f.family == Family::Qwen4 && f.fp8_weight_cache).then(|| {
            term(
                "fp8-weight-cache",
                QWEN4_FP8_WEIGHT_CACHE_BYTES,
                TermSource::Measured,
            )
        }),
        (f.mtp_release_bytes > 0).then(|| {
            term(
                "mtp-release",
                -signed(f.mtp_release_bytes),
                TermSource::Modelled,
            )
        }),
        f.dflash
            .then(|| term("dflash-drafter", DFLASH_DRAFTER_BYTES, TermSource::Measured)),
    ]
    .into_iter()
    .flatten();
    let terms = [store, construction, lm_head_term(f)]
        .into_iter()
        .chain(modelled)
        .chain(extras)
        .collect();
    FitEstimate {
        terms,
        mem_available_bytes: vram_free_bytes,
        floor_bytes: discrete_fit_floor_bytes(vram_total_bytes),
        is_discrete: true,
    }
}

/// The estimate for `facts` against `MemAvailable` now and the floor.
pub(crate) fn estimate(
    f: &FitFacts,
    mem_available_bytes: usize,
    floor_bytes: usize,
) -> FitEstimate {
    let modelled = [
        ("qwen4-transient", f.qwen4_transient_bytes),
        ("model-state-pools", f.pools_bytes),
        ("buffer-arena", f.arena_bytes),
        ("kv-cache", f.kv_bytes),
        ("qsa-caches", f.qsa_bytes),
    ]
    .into_iter()
    .filter(|(_, bytes)| *bytes > 0)
    .map(|(name, bytes)| term(name, signed(bytes), TermSource::Modelled));
    let terms = [store_term(f), construction_term(f), lm_head_term(f)]
        .into_iter()
        .chain(modelled)
        .chain(family_constants(f))
        .collect();
    FitEstimate {
        terms,
        mem_available_bytes,
        floor_bytes,
        is_discrete: false,
    }
}

impl FitEstimate {
    pub(crate) fn need_bytes(&self) -> usize {
        let sum: i64 = self.terms.iter().map(|t| t.bytes).sum();
        usize::try_from(sum.max(0)).unwrap_or(0)
    }

    pub(crate) fn fits(&self) -> bool {
        self.need_bytes().saturating_add(self.floor_bytes) <= self.mem_available_bytes
    }

    pub(crate) fn term(&self, name: &str) -> Option<&FitTerm> {
        self.terms.iter().find(|t| t.name == name)
    }

    fn headroom_mib(&self) -> i64 {
        (signed(self.mem_available_bytes) - signed(self.need_bytes()) - signed(self.floor_bytes))
            / MIB as i64
    }

    fn terms_json(&self) -> String {
        let body = self
            .terms
            .iter()
            .map(|t| format!("\"{}\":{}", t.name, t.bytes / MIB as i64))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{body}}}")
    }

    /// One line per term with its source, then the decision.
    pub(crate) fn render(&self) -> String {
        let mut out = String::from("FIT_ESTIMATE terms (MiB):");
        for t in &self.terms {
            let _ = write!(
                out,
                "\n  {:<18} {:>8}  {}",
                t.name,
                t.bytes / MIB as i64,
                t.source.name()
            );
        }
        let _ = write!(
            out,
            "\n  need_mib={} avail_mib={} floor_mib={} headroom_mib={}",
            self.need_bytes() / MIB,
            self.mem_available_bytes / MIB,
            self.floor_bytes / MIB,
            self.headroom_mib(),
        );
        out
    }

    pub(crate) fn refusal_line(&self) -> String {
        self.verdict_line("FIT_REFUSED", "or set {SELECTOR}=off to start anyway")
    }

    /// The `warn`-mode line: same numbers, the server starts anyway.
    pub(crate) fn warn_line(&self) -> String {
        self.verdict_line(
            "FIT_WARN",
            "starting anyway; FLUX_FIT_CHECK=refuse makes this fatal",
        )
    }

    fn verdict_line(&self, tag: &str, tail: &str) -> String {
        let tail = tail.replace("{SELECTOR}", SELECTOR);
        let (limit_label, free_hint) = if self.is_discrete {
            ("GPU VRAM", "free GPU memory")
        } else {
            ("MemAvailable", "free host memory")
        };
        format!(
            "{tag} need_mib={} avail_mib={} floor_mib={} terms={} \
             (need + floor exceeds {limit_label}; {free_hint}, lower \
             --max-seq-len / --max-batch-size, {tail})",
            self.need_bytes() / MIB,
            self.mem_available_bytes / MIB,
            self.floor_bytes / MIB,
            self.terms_json(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FitMode {
    Off,
    Warn,
    Refuse,
}

pub(crate) fn parse_mode(value: Option<&str>) -> Result<FitMode> {
    match value.map(str::trim) {
        Some("refuse") | Some("on") | Some("1") => Ok(FitMode::Refuse),
        None | Some("") | Some("warn") => Ok(FitMode::Warn),
        Some("off") | Some("0") => Ok(FitMode::Off),
        Some(other) => bail!("{SELECTOR}={other:?}: expected refuse, warn or off"),
    }
}

/// Uncalibrated families never hard-refuse on 27B constants.
pub(crate) fn effective_mode(mode: FitMode, family: Family) -> FitMode {
    match (mode, family) {
        (FitMode::Refuse, Family::Other) => FitMode::Warn,
        (mode, _) => mode,
    }
}

pub(crate) fn decide(mode: FitMode, est: &FitEstimate) -> Result<()> {
    match mode {
        FitMode::Off => Ok(()),
        _ if est.fits() => {
            tracing::info!(target: "atlas_startup_budget", "{}", est.render());
            Ok(())
        }
        FitMode::Warn => {
            tracing::warn!(target: "atlas_startup_budget", "{}\n{}", est.render(), est.warn_line());
            Ok(())
        }
        FitMode::Refuse => {
            tracing::error!(target: "atlas_startup_budget", "{}", est.render());
            bail!("{}", est.refusal_line())
        }
    }
}

/// `(name, bytes)` of every tensor in a safetensors JSON header.
pub(crate) fn header_tensor_bytes(header: &str) -> Result<Vec<(String, u64)>> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header)?;
    map.into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, entry)| {
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .and_then(|a| Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?)));
            match offsets {
                Some((start, end)) if end >= start => Ok((name, end - start)),
                _ => bail!("safetensors header: bad data_offsets for {name}"),
            }
        })
        .collect()
}

fn read_header(path: &Path) -> Result<String> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    anyhow::ensure!(
        len <= MAX_HEADER_BYTES,
        "{}: header {len} B too large",
        path.display()
    );
    let mut buf = vec![0u8; usize::try_from(len)?];
    file.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

/// Resident and `mtp.*` bytes of the checkpoint after the serving skip set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResidentBytes {
    pub(crate) store: usize,
    pub(crate) mtp: usize,
}

pub(crate) fn resident_bytes(
    tensors: impl IntoIterator<Item = (String, u64)>,
    skip: &dyn Fn(&str) -> bool,
) -> ResidentBytes {
    tensors.into_iter().filter(|(name, _)| !skip(name)).fold(
        ResidentBytes::default(),
        |acc, (name, bytes)| {
            let bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
            ResidentBytes {
                store: acc.store.saturating_add(bytes),
                mtp: if name.starts_with("mtp.") {
                    acc.mtp.saturating_add(bytes)
                } else {
                    acc.mtp
                },
            }
        },
    )
}

/// The serving skip the loader will apply: the offloaded PLE n-gram table
/// when `ple-offload/manifest.json` exists, plus non-local PP layers / EP experts.
fn preload_skip(config: &ModelConfig, model_dir: &Path) -> Box<dyn Fn(&str) -> bool> {
    let ple_prefix = (config.is_qwen4_exp()
        && model_dir.join("ple-offload/manifest.json").is_file())
    .then(|| {
        format!(
            "{}.ple.ple_embedding.ngram_embedding",
            config.layer_prefix(1)
        )
    });
    let pp_skip = (config.pp_world_size > 1).then(|| {
        let (pp_start, pp_end) = config.pp_layer_range();
        (pp_start, pp_end, config.pp_rank)
    });
    let ep_skip =
        (config.ep_world_size > 1 && config.num_experts > 0).then(|| config.local_expert_range());
    Box::new(move |name| {
        if let Some((pp_start, pp_end, pp_rank)) = pp_skip {
            if name.starts_with("mtp.") && pp_rank > 0 {
                return true;
            }
            let tail = name
                .find(".layers.")
                .map(|pos| &name[pos + ".layers.".len()..])
                .or_else(|| name.strip_prefix("layers."));
            if let Some(tail) = tail
                && let Some(end) = tail.find('.')
                && let Ok(layer_idx) = tail[..end].parse::<usize>()
                && (layer_idx < pp_start || layer_idx >= pp_end)
            {
                return true;
            }
        }
        if let Some((ep_start, ep_end)) = ep_skip
            && super::weights::is_ep_nonlocal_expert_tensor(name, ep_start, ep_end)
        {
            return true;
        }
        ple_prefix.as_deref().is_some_and(|prefix| {
            spark_model::layers::qwen4_ple::skip_offloaded_ple_tensor(prefix, name)
        })
    })
}

fn checkpoint_resident(config: &ModelConfig, model_dir: &Path) -> Result<ResidentBytes> {
    let skip = preload_skip(config, model_dir);
    let mut tensors = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|x| x == "safetensors") {
            tensors.extend(header_tensor_bytes(&read_header(&path)?)?);
        }
    }
    Ok(resident_bytes(tensors, skip.as_ref()))
}

fn fp8_weight_cache_selected() -> bool {
    FP8_WEIGHT_CACHE_SELECTORS
        .iter()
        .any(|k| std::env::var(k).ok().as_deref() == Some("1"))
}

fn mode_from_env() -> Result<FitMode> {
    let value = std::env::var(SELECTOR)
        .or_else(|_| std::env::var(SELECTOR_ALIAS))
        .ok();
    parse_mode(value.as_deref())
}

/// Pre-load entry point: `Ok(Some(need))` when estimated, `Err` on refusal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_preload(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    breakdown: &super::preflight::ReserveBreakdown,
    buffer_arena_bytes: usize,
    inference_reserve: usize,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<Option<usize>> {
    let family = Family::of(config);
    let mode = effective_mode(mode_from_env()?, family);
    if mode == FitMode::Off {
        tracing::warn!(target: "atlas_startup_budget", "{SELECTOR}=off: pre-load fit check skipped");
        return Ok(None);
    }
    let resident = match checkpoint_resident(config, model_dir) {
        Ok(r) => r,
        Err(error) => {
            tracing::warn!("{SELECTOR}: checkpoint headers unreadable ({error:#}); skipped");
            return Ok(None);
        }
    };
    let checkpoint = resident.store;
    let is_discrete =
        !gpu.is_unified_memory() || spark_runtime::startup_ledger::is_discrete_device();
    let (mem_available, discrete_total_vram) = if is_discrete {
        let Some(vram_free) = gpu
            .free_memory()
            .ok()
            .filter(|&b| b > 0)
            .or_else(spark_runtime::startup_ledger::available_device_memory_bytes)
        else {
            tracing::warn!("{SELECTOR}: discrete GPU VRAM telemetry unreadable; skipped");
            return Ok(None);
        };
        let vram_total = gpu
            .total_memory()
            .ok()
            .filter(|&b| b > 0)
            .unwrap_or(vram_free);
        (vram_free, Some(vram_total))
    } else {
        let Some(host_mem) = spark_runtime::startup_ledger::mem_available_bytes() else {
            tracing::warn!("{SELECTOR}: MemAvailable unreadable; skipped");
            return Ok(None);
        };
        (host_mem, None)
    };
    if checkpoint == 0 {
        tracing::warn!(
            "{SELECTOR}: no safetensors in {}; skipped",
            model_dir.display()
        );
        return Ok(None);
    }
    let block = super::preflight::kv_block_bytes_all_layers_for_args(config, args)?;
    let addressable = match args.kv_pool_tokens {
        Some(tokens) => spark_model::factory::kv_pool::requested_pool_blocks_with_prefix_cache(
            tokens,
            args.max_seq_len,
            args.max_batch_size,
            args.block_size,
            args.enable_prefix_caching,
        )
        .checked_mul(block)
        .ok_or_else(|| anyhow::anyhow!("KV pool byte overflow"))?,
        None => super::preflight::kv_cache_addressable_bytes(
            args.max_batch_size,
            args.max_seq_len,
            args.block_size,
            block,
        )?,
    };
    let dflash_releases_mtp = args.dflash
        && !args.speculative
        && std::env::var("ATLAS_QWEN4_RELEASE_UNUSED").ok().as_deref() == Some("1");
    let facts = FitFacts {
        family,
        checkpoint_bytes: checkpoint,
        mtp_release_bytes: if family == Family::Qwen4
            && (super::speculative_draft_count(args).is_none() || dflash_releases_mtp)
        {
            resident.mtp
        } else {
            0
        },
        fp8_weight_cache: fp8_weight_cache_selected(),
        construction_modelled_bytes: super::weights::construction_overhead_bytes(config).total(),
        qwen4_transient_bytes: super::preflight::qwen4_construction_transient_bytes(
            config,
            super::speculative_draft_count(args).is_some(),
        )?,
        lm_head_elems: config.vocab_size.saturating_mul(config.hidden_size),
        pools_bytes: breakdown
            .ssm_pool_bytes
            .saturating_add(breakdown.ssm_snapshot_bytes)
            .saturating_add(breakdown.ple_history_bytes),
        arena_bytes: buffer_arena_bytes,
        kv_bytes: super::preflight::kv_cache_projected_bytes(
            addressable,
            mem_available.saturating_sub(checkpoint),
            inference_reserve,
            args.gpu_memory_utilization,
        ),
        qsa_bytes: breakdown
            .qsa_side_cache_bytes
            .saturating_add(breakdown.qsa_fp8_shadow_bytes)
            .saturating_add(breakdown.batched_k5_lane_bytes),
        dflash: args.dflash,
    };
    let est = if let Some(total_vram) = discrete_total_vram {
        estimate_discrete(&facts, mem_available, total_vram)
    } else {
        let floor = crate::main_modules::mem_watch::WatchConfig::from_env()?
            .map_or(12_288 * MIB, |c| c.floor_bytes);
        estimate(&facts, mem_available, floor)
    };
    decide(mode, &est)?;
    Ok(Some(est.need_bytes()))
}

#[cfg(test)]
#[path = "fit_estimate_tests.rs"]
mod tests;

#[cfg(test)]
mod discrete_tests {
    use super::{
        Family, FitFacts, FitMode, MIB, decide, discrete_fit_floor_bytes, estimate,
        estimate_discrete,
    };

    const GIB: usize = 1024 * MIB;

    #[test]
    fn discrete_floor_caps_at_512_mib_or_five_percent_vram() {
        assert_eq!(discrete_fit_floor_bytes(8 * GIB), (8 * GIB) / 20);
        assert_eq!(discrete_fit_floor_bytes(16 * GIB), 512 * MIB);
        assert_eq!(discrete_fit_floor_bytes(32 * GIB), 512 * MIB);
        assert_eq!(discrete_fit_floor_bytes(72 * GIB), 512 * MIB);
        assert_eq!(discrete_fit_floor_bytes(96 * GIB), 512 * MIB);
    }

    #[test]
    fn discrete_16gb_and_32gb_dense_models_fit_without_gb10_uma_serve_growth() {
        // 16 GB discrete GPU (RTX 5060 Ti / 5070 Ti / 5080) with a ~9 GiB dense model.
        let dense_16gb = FitFacts {
            family: Family::HybridDense,
            checkpoint_bytes: 9 * GIB,
            mtp_release_bytes: 0,
            fp8_weight_cache: false,
            construction_modelled_bytes: GIB,
            qwen4_transient_bytes: 0,
            lm_head_elems: 151_936 * 2_048,
            pools_bytes: 512 * MIB,
            arena_bytes: GIB,
            kv_bytes: 2 * GIB,
            qsa_bytes: 0,
            dflash: false,
        };
        let est_16 = estimate_discrete(&dense_16gb, 15_500 * MIB, 16 * GIB);
        assert!(est_16.term("serve-growth").is_none());
        assert!(est_16.term("residual").is_none());
        assert_eq!(est_16.floor_bytes, 512 * MIB);
        assert!(est_16.fits(), "{}", est_16.render());
        assert!(decide(FitMode::Refuse, &est_16).is_ok());

        // Under GB10 UMA constants (12 GiB floor + 6011 MiB serve-growth), 16 GB would falsely refuse.
        let uma_refused = estimate(&dense_16gb, 15_500 * MIB, 12_288 * MIB);
        assert!(!uma_refused.fits());

        // 32 GB discrete GPU (RTX 5090) with a 27B NVFP4 dense model (~18 GiB weights).
        let dense_32gb = FitFacts {
            family: Family::HybridDense,
            checkpoint_bytes: 18 * GIB,
            mtp_release_bytes: 0,
            fp8_weight_cache: false,
            construction_modelled_bytes: 3 * GIB,
            qwen4_transient_bytes: 0,
            lm_head_elems: 248_320 * 5_120,
            pools_bytes: 2 * GIB,
            arena_bytes: 2 * GIB,
            kv_bytes: 2 * GIB,
            qsa_bytes: 0,
            dflash: false,
        };
        let est_32 = estimate_discrete(&dense_32gb, 31_500 * MIB, 32 * GIB);
        assert!(est_32.fits(), "{}", est_32.render());
        assert!(decide(FitMode::Refuse, &est_32).is_ok());
    }

    #[test]
    fn discrete_72gb_and_96gb_gpus_fit_large_models_and_still_refuse_true_oversubscription() {
        let model_72gb = FitFacts {
            family: Family::HybridDense,
            checkpoint_bytes: 52 * GIB,
            mtp_release_bytes: 0,
            fp8_weight_cache: false,
            construction_modelled_bytes: 4 * GIB,
            qwen4_transient_bytes: 0,
            lm_head_elems: 248_320 * 5_120,
            pools_bytes: 3 * GIB,
            arena_bytes: 3 * GIB,
            kv_bytes: 4 * GIB,
            qsa_bytes: 0,
            dflash: false,
        };
        let est_72 = estimate_discrete(&model_72gb, 70 * GIB, 72 * GIB);
        assert!(est_72.fits(), "{}", est_72.render());

        // 96 GB RTX PRO 6000 running Flash-Next (~72.17 GiB weights).
        let fn_96gb = FitFacts {
            family: Family::Qwen4,
            checkpoint_bytes: 72 * GIB,
            mtp_release_bytes: 0,
            fp8_weight_cache: false,
            construction_modelled_bytes: 4 * GIB,
            qwen4_transient_bytes: 3 * GIB,
            lm_head_elems: 248_320 * 2_560,
            pools_bytes: 2 * GIB,
            arena_bytes: 3 * GIB,
            kv_bytes: 3 * GIB,
            qsa_bytes: 2 * GIB,
            dflash: false,
        };
        let est_96 = estimate_discrete(&fn_96gb, 94 * GIB, 96 * GIB);
        assert!(est_96.fits(), "{}", est_96.render());

        // Same 72 GiB model on a 16 GiB GPU genuinely does not fit and refuses in Refuse mode.
        let oversubscribed = estimate_discrete(&fn_96gb, 15 * GIB, 16 * GIB);
        assert!(!oversubscribed.fits());
        let err = decide(FitMode::Refuse, &oversubscribed)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds GPU VRAM"), "{err}");
        assert!(err.contains("free GPU memory"), "{err}");
    }

    #[test]
    fn nvfp4_1m_fit_estimate_uses_actual_kv_dtype_and_scales_across_1x_tp2_tp4() {
        use crate::cli::{Cli, Command};
        use crate::main_modules::serve_phases::preflight::{
            kv_block_bytes_all_layers_for_args, kv_cache_addressable_bytes,
            kv_cache_projected_bytes,
        };
        use atlas_core::config::ModelConfig;
        use clap::Parser;

        let Command::Serve(mut args) = Cli::try_parse_from(["spark", "serve", "nvidia/model"])
            .unwrap()
            .command;
        args.block_size = 16;
        args.max_batch_size = 1;
        args.max_seq_len = 1_048_576;
        args.kv_cache_dtype = "nvfp4".to_string();
        args.kv_high_precision_layers = "0".to_string();

        let mut cfg_1x = ModelConfig::qwen3_next_80b_nvfp4();
        cfg_1x.model_type = "qwen4_exp".to_string();
        cfg_1x.qwen4_qsa = true;

        let block_nvfp4_hp0 = kv_block_bytes_all_layers_for_args(&cfg_1x, &args).unwrap();
        let addr_nvfp4_hp0 = kv_cache_addressable_bytes(1, 1_048_576, 16, block_nvfp4_hp0).unwrap();

        let mut args_bf16 = args.clone();
        args_bf16.kv_cache_dtype = "bf16".to_string();
        let block_bf16 = kv_block_bytes_all_layers_for_args(&cfg_1x, &args_bf16).unwrap();
        let addr_bf16 = kv_cache_addressable_bytes(1, 1_048_576, 16, block_bf16).unwrap();

        // Pure NVFP4 KV is exact 9/32 (~3.56x smaller) vs BF16: 6.75 GiB vs 24.0 GiB at 1M!
        assert_eq!(addr_nvfp4_hp0 * 32, addr_bf16 * 9);
        assert_eq!(addr_nvfp4_hp0 / GIB, 6);
        assert_eq!(addr_bf16 / GIB, 24);

        // Mixed NVFP4 (hp=auto -> 4 BF16 + 8 NVFP4 = 12.5 GiB at 1M) fits 1x 96 GB cleanly!
        let mut args_auto = args.clone();
        args_auto.kv_high_precision_layers = "auto".to_string();
        let block_auto = kv_block_bytes_all_layers_for_args(&cfg_1x, &args_auto).unwrap();
        let addr_auto = kv_cache_addressable_bytes(1, 1_048_576, 16, block_auto).unwrap();
        assert_eq!(addr_auto / MIB, 12_800);
        let projected_1x = kv_cache_projected_bytes(addr_auto, 22 * GIB, 2 * GIB, 0.92);
        assert_eq!(projected_1x, addr_auto);

        let fn_1x_1m = FitFacts {
            family: Family::Qwen4,
            checkpoint_bytes: 72 * GIB,
            mtp_release_bytes: 1_556 * MIB,
            fp8_weight_cache: false,
            construction_modelled_bytes: 4 * GIB,
            qwen4_transient_bytes: 1_700 * MIB,
            lm_head_elems: 248_320 * 2_560,
            pools_bytes: 600 * MIB,
            arena_bytes: 1_800 * MIB,
            kv_bytes: projected_1x,
            qsa_bytes: 1_650 * MIB,
            dflash: false,
        };
        let est_1x = estimate_discrete(&fn_1x_1m, 95 * GIB, 96 * GIB);
        assert!(est_1x.fits(), "{}", est_1x.render());

        // TP=2 / TP=4 shards Flash-Next KV heads (2 -> 1/rank), halving KV bytes per GPU
        // so 4M NVFP4 shared pool (25 GiB/GPU) fits effortlessly on 2x/4x 96 GB.
        let mut cfg_tp = cfg_1x.clone();
        cfg_tp.tp_world_size = 2;
        cfg_tp.num_key_value_heads = 1;
        let block_tp_auto = kv_block_bytes_all_layers_for_args(&cfg_tp, &args_auto).unwrap();
        assert_eq!(block_tp_auto * 2, block_auto);
        let pool_4m_tp2 = spark_model::factory::kv_pool::requested_pool_blocks_with_prefix_cache(
            4_194_304, 1_048_576, 16, 16, false,
        ) * block_tp_auto;
        assert_eq!(pool_4m_tp2 / GIB, 25);
        let fn_tp2_4m = FitFacts {
            checkpoint_bytes: 38 * GIB,
            kv_bytes: pool_4m_tp2,
            ..fn_1x_1m
        };
        let est_tp2 = estimate_discrete(&fn_tp2_4m, 95 * GIB, 96 * GIB);
        assert!(est_tp2.fits(), "{}", est_tp2.render());
    }
}
