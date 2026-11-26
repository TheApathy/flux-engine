// SPDX-License-Identifier: AGPL-3.0-only
//
// Hardware target aliases and `extends` inheritance resolution for build.rs.
// Also included under `#[cfg(test)]` in `src/lib.rs` so `cargo test` exercises
// the production resolution logic directly.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Canonicalize a hardware target name or convenience SM alias.
///
/// Maps `sm120` -> `rtx6000`, `sm121` -> `gb10`, and `sm103` -> `gb300`
/// (case-insensitive after trimming). Any other name is returned trimmed.
pub(crate) fn canonical_hw_name(raw: &str) -> Cow<'_, str> {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "sm120" => Cow::Borrowed("rtx6000"),
        "sm121" => Cow::Borrowed("gb10"),
        "sm103" => Cow::Borrowed("gb300"),
        _ => Cow::Borrowed(trimmed),
    }
}

/// Resolve the target hardware name from `ATLAS_TARGET_HW` / `FLUX_TARGET_HW`,
/// defaulting to `"gb10"` when neither is set or non-empty.
pub(crate) fn resolve_hw_env(atlas_hw: Option<&str>, flux_hw: Option<&str>) -> String {
    let raw = atlas_hw
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .or_else(|| flux_hw.map(str::trim).filter(|v| !v.is_empty()))
        .unwrap_or("gb10");
    canonical_hw_name(raw).into_owned()
}

/// Resolved hardware directory chain from child (index 0) to base ancestor(s).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HwChain {
    /// Canonical name of the requested (child) hardware target.
    pub hw: String,
    /// Compiler architecture from the child `HARDWARE.toml` (e.g. `sm_120a`, `sm_121f`).
    pub arch: String,
    /// Compute vendor (`nvidia`, `apple`, ...), child first then base fallback.
    pub vendor: Option<String>,
    /// FP32 residual toggle, child first then base fallback.
    pub use_fp32_residual: Option<bool>,
    /// Hardware kernel directories ordered child-first (`[child, base, ...]`).
    pub hw_dirs: Vec<PathBuf>,
    /// Canonical hardware names corresponding to `hw_dirs` (child-first).
    pub hw_names: Vec<String>,
    /// Every `HARDWARE.toml` inspected while building the chain.
    pub hw_toml_paths: Vec<PathBuf>,
}

impl HwChain {
    /// Whether `name` (canonicalized) appears anywhere in this hardware chain.
    pub(crate) fn includes_hw(&self, name: &str) -> bool {
        let needle = canonical_hw_name(name);
        self.hw_names.iter().any(|h| h == needle.as_ref())
    }

    /// True when the hardware target inherits from at least one base directory.
    pub(crate) fn is_inherited(&self) -> bool {
        self.hw_dirs.len() > 1
    }
}

/// Resolve a hardware target and any `[hardware].extends` base chain under `kernels_root`.
///
/// The child `HARDWARE.toml` always supplies `arch` and takes priority for optional
/// hardware properties (`vendor`, `use_fp32_residual`). Cycles and missing directories
/// return an error string.
pub(crate) fn resolve_hw_chain(kernels_root: &Path, hw_name: &str) -> Result<HwChain, String> {
    let root_hw = canonical_hw_name(hw_name).into_owned();
    if root_hw.is_empty() {
        return Err("hardware target name must not be empty".to_owned());
    }

    let mut current = root_hw.clone();
    let mut visited = BTreeSet::new();
    let mut hw_dirs = Vec::new();
    let mut hw_names = Vec::new();
    let mut hw_toml_paths = Vec::new();
    let mut arch: Option<String> = None;
    let mut vendor: Option<String> = None;
    let mut use_fp32_residual: Option<bool> = None;

    loop {
        if !visited.insert(current.clone()) {
            return Err(format!(
                "circular hardware `extends` chain involving {current:?} (visited: {:?})",
                hw_names
            ));
        }
        let hw_dir = kernels_root.join(&current);
        if !hw_dir.is_dir() {
            return Err(format!(
                "Hardware kernel directory not found: {}",
                hw_dir.display()
            ));
        }
        let hw_toml_path = hw_dir.join("HARDWARE.toml");
        let content = std::fs::read_to_string(&hw_toml_path)
            .map_err(|e| format!("Cannot read {}: {e}", hw_toml_path.display()))?;
        let parsed: toml::Value = toml::from_str(&content)
            .map_err(|e| format!("Bad TOML in {}: {e}", hw_toml_path.display()))?;
        let hardware = parsed
            .get("hardware")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| format!("Missing [hardware] table in {}", hw_toml_path.display()))?;

        if arch.is_none() {
            let a = hardware
                .get("arch")
                .and_then(toml::Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    format!(
                        "hardware.arch must be a non-empty string in {}",
                        hw_toml_path.display()
                    )
                })?;
            arch = Some(a.to_owned());
        }
        if vendor.is_none()
            && let Some(v) = hardware
                .get("vendor")
                .and_then(toml::Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
        {
            vendor = Some(v.to_owned());
        }
        if use_fp32_residual.is_none()
            && let Some(fp32) = hardware
                .get("use_fp32_residual")
                .and_then(toml::Value::as_bool)
        {
            use_fp32_residual = Some(fp32);
        }

        hw_dirs.push(hw_dir);
        hw_names.push(current);
        hw_toml_paths.push(hw_toml_path.clone());

        match hardware.get("extends") {
            None => break,
            Some(val) => {
                let raw_parent = val.as_str().ok_or_else(|| {
                    format!(
                        "hardware.extends must be a string in {}",
                        hw_toml_path.display()
                    )
                })?;
                let next = canonical_hw_name(raw_parent).into_owned();
                if next.is_empty() {
                    return Err(format!(
                        "hardware.extends must not be empty in {}",
                        hw_toml_path.display()
                    ));
                }
                current = next;
            }
        }
    }

    Ok(HwChain {
        hw: root_hw,
        arch: arch.expect("first hardware entry sets arch"),
        vendor,
        use_fp32_residual,
        hw_dirs,
        hw_names,
        hw_toml_paths,
    })
}

/// Find the highest-priority model directory containing `MODEL.toml`.
pub(crate) fn resolve_model_meta_dir(hw_dirs: &[PathBuf], model: &str) -> Option<PathBuf> {
    hw_dirs
        .iter()
        .map(|d| d.join(model))
        .find(|model_dir| model_dir.join("MODEL.toml").is_file())
}

/// Find the highest-priority directory containing `KERNEL.toml` for `(model, quant)`.
///
/// Checks model-specific directories (`$hw/$model/$quant/KERNEL.toml`, child first)
/// before quant-common directories (`$hw/$quant/KERNEL.toml`, child first).
pub(crate) fn resolve_kernel_toml_dir(
    hw_dirs: &[PathBuf],
    model: &str,
    quant: &str,
) -> Option<PathBuf> {
    if let Some(dir) = hw_dirs
        .iter()
        .map(|d| d.join(model).join(quant))
        .find(|kdir| kdir.join("KERNEL.toml").is_file())
    {
        return Some(dir);
    }
    hw_dirs
        .iter()
        .map(|d| d.join(quant))
        .find(|kdir| kdir.join("KERNEL.toml").is_file())
}

/// Collect ordered `-I` include directories for an inherited hardware build.
///
/// Emits child directories strictly before base directories so child `.cuh`
/// overrides win cleanly over inherited headers.
pub(crate) fn collect_include_dirs(hw_dirs: &[PathBuf], model: &str, quant: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for hw_dir in hw_dirs {
        for candidate in [
            hw_dir.join(model).join(quant),
            hw_dir.join(quant),
            hw_dir.join("common"),
            hw_dir.clone(),
        ] {
            if candidate.is_dir() && !dirs.contains(&candidate) {
                dirs.push(candidate);
            }
        }
    }
    dirs
}

/// Find all kernel source files with extension `ext` (e.g. `"cu"`, `"metal"`)
/// directly inside `kernel_dir`. Returns empty vec if absent.
pub(crate) fn find_kernel_files(kernel_dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(kernel_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

/// Collect kernel source files matching `ext` across the inheritance chain with
/// shadowing across three priority layers (lowest to highest):
///   1. shared (`$hw/common/`) — base to child
///   2. common (`$hw/$quant/`) — base to child
///   3. model  (`$hw/$model/$quant/`) — base to child
///
/// Within each layer, child files override base files of the same stem; higher
/// layers always override lower layers.
pub(crate) fn collect_kernel_files_layered(
    hw_dirs: &[PathBuf],
    model: &str,
    quant: &str,
    ext: &str,
) -> Vec<PathBuf> {
    let mut files: HashMap<String, PathBuf> = HashMap::new();

    let mut insert_dir = |dir: &Path| {
        if !dir.is_dir() {
            return;
        }
        for f in find_kernel_files(dir, ext) {
            if let Some(stem) = f.file_stem().and_then(|s| s.to_str()) {
                files.insert(stem.to_owned(), f);
            }
        }
    };

    // Layer 1: shared (quant-independent) kernels, base -> child
    for hw_dir in hw_dirs.iter().rev() {
        insert_dir(&hw_dir.join("common"));
    }
    // Layer 2: quant-specific common kernels, base -> child
    for hw_dir in hw_dirs.iter().rev() {
        insert_dir(&hw_dir.join(quant));
    }
    // Layer 3: model-specific overrides, base -> child
    for hw_dir in hw_dirs.iter().rev() {
        insert_dir(&hw_dir.join(model).join(quant));
    }

    let mut result: Vec<PathBuf> = files.into_values().collect();
    result.sort();
    result
}

/// Collect `.cu` files across the inheritance chain (convenience wrapper).
#[allow(dead_code)]
pub(crate) fn collect_cu_files_layered(
    hw_dirs: &[PathBuf],
    model: &str,
    quant: &str,
) -> Vec<PathBuf> {
    collect_kernel_files_layered(hw_dirs, model, quant, "cu")
}
