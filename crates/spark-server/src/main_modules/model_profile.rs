// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model launch profiles: the facts about a model that a recipe's `defaults:` flags
//! cannot express — how much memory it needs, whether it must run alone on the box, and the
//! process environment its loader reads.
//!
//! Read from the `atlas:` block of a built-in recipe (`recipe::builtin`), keyed by the
//! checkpoint's `model_type`, so the hot-swap path can look a profile up from nothing but the
//! `ServeArgs` it is handed.
//!
//! ## Why run-alone admission runs on live free memory
//! GB10 memory is unified: an over-allocation takes the HOST down, not just this process. A
//! "run alone" model (DeepSeek-V4.1: ~88 GB resident + 16 GB headroom) only fits when no
//! other large GPU workload occupies the box, so empty-host loads check free memory before
//! allocating weights, and live recipe swaps cross a clean process `exec` boundary via
//! `swap_env::exec`.

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::recipe::yaml::{self, Yaml};

#[path = "model_profile_contract.rs"]
mod contract;
pub(crate) use contract::{log_receipt, unicode_environment_value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProfileReceipt {
    pub recipe_id: String,
    pub profile_kind: String,
    pub environment_sha256: String,
    pub build_track: Option<String>,
    /// DFlash lookup selector overlaid on the sealed base, when engaged.
    pub lookup: Option<&'static str>,
}

/// What the launcher needs to know about one model beyond its flags.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelProfile {
    pub recipe_id: String,
    pub model_type: String,
    /// Host memory the model takes once loaded, in GB (unified memory on GB10).
    pub resident_gb: f64,
    /// Extra when its speculative drafter is enabled.
    pub dspark_extra_gb: f64,
    /// Memory that must remain free AFTER the model is resident, in GB.
    pub headroom_gb: f64,
    /// Refuse to load unless the box can give it resident + headroom after the previous model
    /// is released — i.e. it cannot share the machine.
    pub run_alone: bool,
    /// Process environment its loader reads (set before the load).
    pub env: BTreeMap<String, String>,
    /// Optional fail-closed identity for profiles whose measured environment must be exact.
    pub(crate) contract: Option<contract::ProfileContract>,
}

impl ModelProfile {
    /// Memory to require free before the load, in bytes. DSpark counts only when its env
    /// switch is on in the environment the load will see.
    pub fn required_bytes(&self, dspark_on: bool) -> u64 {
        let gb = self.resident_gb
            + self.headroom_gb
            + if dspark_on { self.dspark_extra_gb } else { 0.0 };
        (gb * 1e9) as u64
    }
}

/// Parse the `atlas:` block of one recipe YAML. `None` when the recipe has none.
pub(crate) fn parse_profile(recipe_id: &str, text: &str) -> Result<Option<ModelProfile>> {
    let doc = yaml::parse(text).with_context(|| format!("reading recipe {recipe_id}"))?;
    let Some(block) = doc
        .as_map()
        .and_then(|m| m.get("atlas"))
        .and_then(Yaml::as_map)
    else {
        return Ok(None);
    };
    let s = |key: &str| block.get(key).and_then(Yaml::as_str).map(str::to_string);
    let num = |key: &str| -> Result<f64> {
        match s(key) {
            Some(v) => v
                .parse()
                .with_context(|| format!("{recipe_id}: atlas.{key} {v:?} is not a number")),
            None => Ok(0.0),
        }
    };
    let mut env = BTreeMap::new();
    if let Some(map) = block.get("env").and_then(Yaml::as_map) {
        for (k, v) in map {
            let v = v
                .as_str()
                .with_context(|| format!("{recipe_id}: atlas.env.{k} is not a scalar"))?;
            env.insert(k.clone(), v.to_string());
        }
    }
    let contract = contract::ProfileContract::parse(recipe_id, block, &env)?;
    Ok(Some(ModelProfile {
        recipe_id: recipe_id.to_string(),
        model_type: s("model_type")
            .with_context(|| format!("{recipe_id}: atlas.model_type is required"))?,
        resident_gb: num("resident_gb")?,
        dspark_extra_gb: num("dspark_extra_gb")?,
        headroom_gb: num("headroom_gb")?,
        run_alone: s("run_alone").as_deref() == Some("true"),
        env,
        contract,
    }))
}

/// The built-in profile for the model being launched.
///
/// Matched by the checkpoint's directory name first (a built-in recipe's `model:`): several
/// served checkpoints share a `model_type` (for example, multiple Qwen3.5/Qwen3.8 dense 27B
/// variants all report `qwen3_5`) but need different kernel targets and environments. Falls
/// back to the `model_type` match only when exactly one built-in declares that type.
pub(crate) fn profile_for(model: Option<&str>, model_type: &str) -> Result<Option<ModelProfile>> {
    let dir_name = model
        .map(|m| m.trim_end_matches('/'))
        .and_then(|m| std::path::Path::new(m).file_name())
        .and_then(|n| n.to_str());
    let mut with_model = Vec::new();
    for (id, text) in crate::recipe::builtin::BUILTIN_YAML {
        let recipe_model = crate::recipe::Recipe::parse(*id, text)
            .with_context(|| format!("built-in recipe {id} does not parse"))?
            .model;
        if let Some(profile) = parse_profile(id, text)? {
            with_model.push((recipe_model, profile));
        }
    }
    if let Some(name) = dir_name
        && let Some((_, p)) = with_model.iter().find(|(m, _)| m == name)
    {
        return Ok(Some(p.clone()));
    }
    let mut by_type = with_model
        .into_iter()
        .filter(|(_, p)| p.model_type == model_type);
    match (by_type.next(), by_type.next()) {
        (Some((_, p)), None) => Ok(Some(p)),
        _ => Ok(None),
    }
}

/// The launch profile belonging to this exact built-in recipe.
///
/// Matching the whole parsed recipe, not only its id, preserves the rule that
/// a fetched recipe with the same id replaces a built-in.  It also disambiguates
/// multiple named profiles for one checkpoint: selecting the safe MTP card must
/// not apply the environment of the first target-only card with the same
/// `model:` value.
pub(crate) fn profile_for_recipe(recipe: &crate::recipe::Recipe) -> Result<Option<ModelProfile>> {
    if !recipe.built_in {
        return Ok(None);
    }
    for (id, text) in crate::recipe::builtin::BUILTIN_YAML {
        let mut built_in = crate::recipe::Recipe::parse(*id, text)
            .with_context(|| format!("built-in recipe {id} does not parse"))?;
        built_in.built_in = true;
        if &built_in == recipe {
            return parse_profile(id, text);
        }
    }
    Ok(None)
}

/// The built-in profile for a `model_type`, if exactly one built-in declares it.
pub(crate) fn profile_for_model_type(model_type: &str) -> Result<Option<ModelProfile>> {
    profile_for(None, model_type)
}

fn host_mem_available_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    parse_mem_available(&text)
}

fn host_mem_total_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;
    parse_meminfo_field(&text, "MemTotal:")
}

fn select_device_memory_bytes(
    is_discrete: bool,
    gpu_bytes: Option<u64>,
    host_fallback: impl FnOnce() -> Result<u64>,
) -> Result<u64> {
    if is_discrete && let Some(bytes) = gpu_bytes {
        return Ok(bytes);
    }
    host_fallback()
}

/// Available memory for launch-profile admission, in bytes.
///
/// On GB10 UMA this reads host `MemAvailable` from `/proc/meminfo`. On discrete
/// GPUs it reads GPU VRAM free bytes (`gpu.free_memory()` or exact CUDA VRAM
/// via `available_device_memory_bytes()`), falling back to `/proc/meminfo` only
/// when no GPU telemetry is available.
pub(crate) fn mem_available_bytes() -> Result<u64> {
    mem_available_bytes_for_backend(None)
}

pub(crate) fn mem_available_bytes_for_backend(
    gpu: Option<&dyn spark_runtime::gpu::GpuBackend>,
) -> Result<u64> {
    let is_discrete = gpu.map_or_else(
        spark_runtime::startup_ledger::is_discrete_device,
        |backend| !backend.is_unified_memory(),
    );
    let gpu_free = if is_discrete {
        gpu.and_then(|backend| backend.free_memory().ok())
            .or_else(spark_runtime::startup_ledger::available_device_memory_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
    } else {
        None
    };
    select_device_memory_bytes(is_discrete, gpu_free, host_mem_available_bytes)
}

/// Total device memory for prospective run-alone admission, in bytes.
///
/// On GB10 UMA this reads host `MemTotal` from `/proc/meminfo`. On discrete GPUs
/// it reads total GPU VRAM (`gpu.total_memory()` or cached device props),
/// falling back to `/proc/meminfo` only when no GPU telemetry is available.
pub(crate) fn mem_total_bytes() -> Result<u64> {
    mem_total_bytes_for_backend(None)
}

pub(crate) fn mem_total_bytes_for_backend(
    gpu: Option<&dyn spark_runtime::gpu::GpuBackend>,
) -> Result<u64> {
    let is_discrete = gpu.map_or_else(
        spark_runtime::startup_ledger::is_discrete_device,
        |backend| !backend.is_unified_memory(),
    );
    let gpu_total = if is_discrete {
        gpu.and_then(|backend| backend.total_memory().ok())
            .or_else(|| {
                atlas_core::device::active_device_props().map(|props| props.total_mem_bytes)
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .filter(|&bytes| bytes > 0)
    } else {
        None
    };
    select_device_memory_bytes(is_discrete, gpu_total, host_mem_total_bytes)
}

pub(crate) fn parse_mem_available(meminfo: &str) -> Result<u64> {
    parse_meminfo_field(meminfo, "MemAvailable:")
}

fn parse_meminfo_field(meminfo: &str, field: &str) -> Result<u64> {
    let line = meminfo
        .lines()
        .find(|l| l.starts_with(field))
        .with_context(|| format!("/proc/meminfo has no {field} line"))?;
    let kb: u64 = line
        .split_whitespace()
        .nth(1)
        .with_context(|| format!("{field} line has no value"))?
        .parse()
        .with_context(|| format!("{field} value is not a number"))?;
    Ok(kb * 1024)
}

/// The run-alone admission rule, pure so it is testable: refuse unless `available` covers the
/// profile's requirement. Non-run-alone profiles are admitted (their memory is the load's own
/// business, as before).
#[cfg(test)]
pub(crate) fn admit(profile: &ModelProfile, available: u64, dspark_on: bool) -> Result<()> {
    admit_for_device(profile, available, dspark_on, false)
}

pub(crate) fn admit_for_device(
    profile: &ModelProfile,
    available: u64,
    dspark_on: bool,
    is_discrete: bool,
) -> Result<()> {
    if !profile.run_alone {
        return Ok(());
    }
    let need = profile.required_bytes(dspark_on);
    let dspark_suffix = if dspark_on {
        format!(" + {:.0} GB DSpark", profile.dspark_extra_gb)
    } else {
        String::new()
    };
    if is_discrete {
        anyhow::ensure!(
            available >= need,
            "{} ({}) must run alone: it needs {:.1} GB free ({:.0} GB resident{} + {:.0} GB headroom) \
             but only {:.1} GB GPU VRAM is available after the previous model was released. \
             Stop other GPU work on this device and retry.",
            profile.recipe_id,
            profile.model_type,
            need as f64 / 1e9,
            profile.resident_gb,
            dspark_suffix,
            profile.headroom_gb,
            available as f64 / 1e9,
        );
    } else {
        anyhow::ensure!(
            available >= need,
            "{} ({}) must run alone: it needs {:.1} GB free ({:.0} GB resident{} + {:.0} GB headroom) \
             but only {:.1} GB is available after the previous model was released. GB10 memory is \
             unified — loading anyway would risk taking the host down. Stop other GPU work on this \
             box and retry.",
            profile.recipe_id,
            profile.model_type,
            need as f64 / 1e9,
            profile.resident_gb,
            dspark_suffix,
            profile.headroom_gb,
            available as f64 / 1e9,
        );
    }
    Ok(())
}

/// Verify that every environment variable required by `profile` is already
/// present in the process environment.
///
/// Mutating the process environment (`std::env::set_var`) once Tokio workers or
/// the TUI thread exist is unsound in Rust 2024; recipe launches install profile
/// environments cleanly across `exec` via [`super::swap_env::exec`]. An
/// in-process empty-host load without restart argv is therefore allowed only
/// when the operator already exported the profile's required keys (or the
/// profile declares none).
pub(crate) fn ensure_env_present(profile: &ModelProfile) -> Result<()> {
    ensure_env_present_with(profile, |k| std::env::var_os(k).is_some())
}

fn ensure_env_present_with(profile: &ModelProfile, is_set: impl Fn(&str) -> bool) -> Result<()> {
    let missing: Vec<String> = profile
        .env
        .iter()
        .filter(|(k, _)| !is_set(k))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{} requires launch environment variables that cannot be mutated after worker threads \
         start (launch via recipe re-exec or export before startup): {}",
        profile.recipe_id,
        missing.join(", "),
    );
    Ok(())
}

#[cfg(test)]
#[path = "model_profile_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "model_profile_sealed_tests.rs"]
mod sealed_tests;

#[cfg(test)]
mod discrete_tests {
    use std::collections::BTreeMap;

    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

    use super::{
        ModelProfile, admit_for_device, mem_available_bytes_for_backend,
        mem_total_bytes_for_backend, select_device_memory_bytes,
    };

    const GIB: usize = 1024 * 1024 * 1024;

    struct StubDiscreteGpu {
        total_bytes: usize,
        free_bytes: usize,
        unified: bool,
    }

    impl GpuBackend for StubDiscreteGpu {
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            Ok(DevicePtr(0x1000))
        }
        fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
            Ok(DevicePtr(0x1000))
        }
        fn free(&self, _ptr: DevicePtr) -> Result<()> {
            Ok(())
        }
        fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
            Ok(())
        }
        fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            Ok(())
        }
        fn launch(
            &self,
            _func: KernelHandle,
            _grid: [u32; 3],
            _block: [u32; 3],
            _shared_mem: u32,
            _stream: u64,
            _params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            Ok(())
        }
        fn synchronize(&self, _stream: u64) -> Result<()> {
            Ok(())
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
            Ok(KernelHandle(1))
        }
        fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
            Ok(())
        }
        fn memset_async(
            &self,
            _ptr: DevicePtr,
            _value: u8,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            Ok(())
        }
        fn total_memory(&self) -> Result<usize> {
            Ok(self.total_bytes)
        }
        fn free_memory(&self) -> Result<usize> {
            Ok(self.free_bytes)
        }
        fn is_unified_memory(&self) -> bool {
            self.unified
        }
    }

    #[test]
    fn discrete_gpu_profile_queries_use_vram_not_host_meminfo() {
        let discrete_96gb = StubDiscreteGpu {
            total_bytes: 96 * GIB,
            free_bytes: 92 * GIB,
            unified: false,
        };
        assert_eq!(
            mem_available_bytes_for_backend(Some(&discrete_96gb)).unwrap(),
            (92 * GIB) as u64
        );
        assert_eq!(
            mem_total_bytes_for_backend(Some(&discrete_96gb)).unwrap(),
            (96 * GIB) as u64
        );

        // UMA ignores discrete GPU VRAM values and uses host /proc/meminfo.
        assert_eq!(
            select_device_memory_bytes(false, Some(32 * GIB as u64), || Ok(110 * GIB as u64))
                .unwrap(),
            110 * GIB as u64
        );
        // Discrete uses GPU VRAM when present, falling back only when absent.
        assert_eq!(
            select_device_memory_bytes(true, Some(28 * GIB as u64), || Ok(240 * GIB as u64))
                .unwrap(),
            28 * GIB as u64
        );
        assert_eq!(
            select_device_memory_bytes(true, None, || Ok(240 * GIB as u64)).unwrap(),
            240 * GIB as u64
        );
    }

    #[test]
    fn discrete_run_alone_admission_reports_vram_shortfall() {
        let profile = ModelProfile {
            recipe_id: "test/discrete-96gb".to_string(),
            model_type: "qwen4_exp".to_string(),
            resident_gb: 80.0,
            dspark_extra_gb: 0.0,
            headroom_gb: 4.0,
            run_alone: true,
            env: BTreeMap::new(),
            contract: None,
        };
        assert!(admit_for_device(&profile, 84_000_000_000, false, true).is_ok());
        let err = admit_for_device(&profile, 32_000_000_000, false, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("GPU VRAM"), "{err}");
        assert!(!err.contains("GB10 memory is unified"), "{err}");
    }
}
