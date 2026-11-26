// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-oracle tensor dump.
//!
//! `ATLAS_GLM53_ORACLE_DUMP=<dir>` captures the exact device-side inputs and
//! outputs of ONE layer of each instrumented site, so a replacement kernel can
//! be unit-tested offline without loading the 2.05 bpw checkpoint or holding
//! the GPU. Each site fires exactly once per process (the first layer that
//! reaches it), so each instrumented site incurs one D2H burst on its first
//! execution and nothing afterwards.
//!
//! Layout: `<dir>/<site>/<tensor>.<ext>` raw headerless bytes, plus
//! `<dir>/<site>/manifest.json` recording every tensor's dtype, byte length
//! and logical shape, and the scalar kernel arguments. The extension names the
//! true element width (`.bf16`, `.f16`, `.f32`, `.u32`, `.i64`) exactly as the
//! `ATLAS_GLM53_LOGITS_DUMP` hook does -- never assume 2 bytes.
//!
//! This follows `dump_last_row_logits` in `target_staged_exl3.rs`: off unless
//! the env var is set, `Result`-returning, and placed after an explicit fence.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;

/// One tensor to capture: logical name, device pointer, byte length, element
/// dtype tag (used as the file extension) and the logical shape.
pub(crate) struct OracleTensor<'a> {
    pub name: &'a str,
    pub ptr: DevicePtr,
    pub bytes: usize,
    pub dtype: &'a str,
    pub shape: &'a [usize],
}

fn oracle_dir() -> Option<std::ffi::OsString> {
    static DIR: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("ATLAS_GLM53_ORACLE_DUMP").filter(|v| !v.is_empty()))
        .clone()
}

fn already_done(site: &str) -> bool {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    !guard.insert(site.to_string())
}

/// Capture `tensors` for `site` the first time this site is reached.
///
/// `scalars` are the non-pointer kernel arguments (row counts, head counts,
/// epsilons) that a standalone harness needs in order to call the kernel; they
/// are written into the manifest verbatim.
pub(crate) fn dump_site(
    gpu: &dyn GpuBackend,
    stream: u64,
    site: &str,
    tensors: &[OracleTensor<'_>],
    scalars: &[(&str, String)],
) -> Result<()> {
    let Some(dir) = oracle_dir() else {
        return Ok(());
    };
    if already_done(site) {
        return Ok(());
    }
    // The capture reads buffers a kernel has just written on a non-default
    // stream; `copy_d2h` does not order against it.
    gpu.synchronize(stream)?;
    let base = std::path::Path::new(&dir).join(site);
    std::fs::create_dir_all(&base)
        .with_context(|| format!("GLM oracle dump dir {}", base.display()))?;

    let mut tensor_entries = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let mut host = vec![0u8; tensor.bytes];
        gpu.copy_d2h(tensor.ptr, &mut host)?;
        let file = format!("{}.{}", tensor.name, tensor.dtype);
        let path = base.join(&file);
        std::fs::write(&path, &host)
            .with_context(|| format!("GLM oracle dump {}", path.display()))?;
        tensor_entries.push(serde_json::json!({
            "name": tensor.name,
            "file": file,
            "dtype": tensor.dtype,
            "bytes": tensor.bytes,
            "shape": tensor.shape,
        }));
    }
    let mut scalar_entries = serde_json::Map::with_capacity(scalars.len());
    for (key, value) in scalars {
        let parsed = serde_json::from_str::<serde_json::Value>(value)
            .unwrap_or_else(|_| serde_json::Value::String(value.clone()));
        scalar_entries.insert((*key).to_string(), parsed);
    }
    let manifest_value = serde_json::json!({
        "site": site,
        "tensors": tensor_entries,
        "scalars": scalar_entries,
    });
    let mut manifest = serde_json::to_string_pretty(&manifest_value)
        .context("GLM oracle manifest serialization failed")?;
    manifest.push('\n');
    let path = base.join("manifest.json");
    std::fs::write(&path, manifest)
        .with_context(|| format!("GLM oracle manifest {}", path.display()))?;
    Ok(())
}

/// Terse constructor so a capture site reads as a table of tensors.
pub(crate) fn t<'a>(
    name: &'a str,
    ptr: DevicePtr,
    bytes: usize,
    dtype: &'a str,
    shape: &'a [usize],
) -> OracleTensor<'a> {
    OracleTensor {
        name,
        ptr,
        bytes,
        dtype,
        shape,
    }
}
