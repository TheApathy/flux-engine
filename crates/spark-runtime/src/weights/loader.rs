// SPDX-License-Identifier: AGPL-3.0-only

//! `impl WeightLoader for SafetensorsLoader` + sharded/single loader helpers.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

use super::{SafetensorsLoader, WeightLoader, WeightStore};
use crate::gpu::GpuBackend;

impl WeightLoader for SafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        // Allocation receipt for stores loaded this way (the DFlash drafter,
        // donors); nested in `weights/store` it counts as one of its passes.
        let _receipt = crate::startup_ledger::scope_guard("weights/safetensors-loader");
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Collect all safetensor files (indexed, single, or unindexed shards).
        // Supports both HuggingFace standard (model.safetensors*) and Mistral
        // consolidated format (consolidated.safetensors*).
        let index_path = model_dir.join("model.safetensors.index.json");
        let consolidated_index = model_dir.join("consolidated.safetensors.index.json");
        let shard_files: Vec<std::path::PathBuf>;
        let actual_index_path = if index_path.exists() {
            Some(index_path)
        } else if consolidated_index.exists() {
            Some(consolidated_index)
        } else {
            None
        };
        // Resolve once before admission, then load that same shard/tensor plan.
        let indexed_shards = actual_index_path
            .as_deref()
            .map(read_index_shards)
            .transpose()?;
        if let Some(shards) = &indexed_shards {
            let mut files: Vec<_> = shards.keys().map(|name| model_dir.join(name)).collect();
            files.sort();
            shard_files = files;
        } else {
            let single = model_dir.join("model.safetensors");
            if single.exists() {
                shard_files = vec![single];
            } else {
                // Try both model.safetensors-* and consolidated-* shard patterns
                let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                            (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                                && n.ends_with(".safetensors")
                        })
                    })
                    .collect();
                shards.sort();
                if shards.is_empty() {
                    bail!(
                        "No safetensor files found in {}. Expected model.safetensors*, \
                         consolidated.safetensors*, or consolidated-*-of-*.safetensors",
                        model_dir.display()
                    );
                }
                shard_files = shards;
            }
        }

        let extra = model_dir.join("extra_weights.safetensors");
        let extra_exists = extra.exists();

        // Pre-flight OOM estimate: scan safetensor headers (no data) to compute
        // total bytes this rank will load, then apply a model-building overhead
        // multiplier and abort early if the model won't fit.
        //
        // Model building creates additional GPU allocations on top of the raw
        // weight store: transposed weight copies for prefill GEMM, predequanted
        // FP8 copies, NVFP4 quantized copies (for FP8 checkpoints), and transient
        // BF16 intermediates during FP8→NVFP4 conversion.
        //
        // Empirical overhead multipliers (peak memory / on-disk weight bytes):
        //   NVFP4 (Sehyo): ~2.0x  (store aliased + transposed/predequant copies)
        //   FP8 native:    ~1.5x  (store stays FP8, only attention prefill gets NVFP4 copies)
        //
        // The ratio covers the store and its transient staging only. Buffers
        // the model *builder* retains afterwards are architecture-dependent
        // and arrive as absolute bytes in `construction_overhead_bytes`.
        {
            let mut preflight_files = shard_files.clone();
            if extra_exists {
                preflight_files.push(extra.clone());
            }
            let estimated = estimate_load_bytes(&preflight_files, &skip_fn)?;
            let has_fp8 = estimate_has_fp8(&preflight_files, &skip_fn)?;
            let overhead_multiplier: f64 =
                self.peak_memory_multiplier
                    .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let construction = self.construction_overhead_bytes;
            let (peak_estimated, required) = combined_admission_bytes(
                estimated,
                overhead_multiplier,
                construction,
                oom_reserve_bytes,
            )?;
            let free = gpu.free_memory()?;
            let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
            let est_gb = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let construction_gb = construction as f64 / (1024.0 * 1024.0 * 1024.0);
            let peak_gb = peak_estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let reserve_gb = oom_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Pre-flight estimate: {:.2} GB on-disk weights, {:.1}x overhead \
                 + {:.2} GB model-construction = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                est_gb,
                overhead_multiplier,
                construction_gb,
                peak_gb,
                free_gb,
                reserve_gb,
                has_fp8,
            );
            if required > free {
                bail!(
                    "OOM pre-flight: model peak memory ({:.2} GB = {:.2} GB weights × {:.1}x \
                     load overhead + {:.2} GB retained model-construction buffers) \
                     + {:.1} GB reserve = {:.2} GB, \
                     but only {:.2} GB GPU memory is available. \
                     This model is too large. Use a smaller quantization (NVFP4 instead of FP8) \
                     or add more GPUs for expert parallelism.",
                    peak_gb,
                    est_gb,
                    overhead_multiplier,
                    construction_gb,
                    reserve_gb,
                    required as f64 / (1024.0 * 1024.0 * 1024.0),
                    free_gb,
                );
            }
        }

        let mut weight_map = if let Some(shards) = &indexed_shards {
            load_sharded(model_dir, shards, gpu, oom_reserve_bytes, &skip_fn)?
        } else if shard_files.len() == 1 {
            load_single(&shard_files[0], gpu, oom_reserve_bytes, &skip_fn)?
        } else {
            tracing::info!("Loading {} unindexed safetensor shards", shard_files.len());
            let initial_free = gpu.free_memory()?;
            let mut combined = HashMap::new();
            for (i, shard) in shard_files.iter().enumerate() {
                let map = load_single(shard, gpu, oom_reserve_bytes, &skip_fn)?;
                let free_now = gpu.free_memory().unwrap_or(0);
                let used = initial_free.saturating_sub(free_now);
                tracing::info!(
                    "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                    i + 1,
                    shard_files.len(),
                    used as f64 / (1024.0 * 1024.0 * 1024.0),
                    free_now as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("weight loading (shard {}/{})", i + 1, shard_files.len()),
                )?;
                combined.extend(map);
            }
            combined
        };

        // Load extra weight files (e.g. MTP weights grafted from another quantization).
        // `skip_fn` still replicates `mtp.*` under EP while honoring `extra_skip` and
        // `tensor_allowlist` consistently with pre-flight estimation above.
        if extra_exists {
            let extra_weights = load_single(&extra, gpu, oom_reserve_bytes, &skip_fn)?;
            tracing::info!(
                "Loaded {} extra weight tensors from extra_weights.safetensors",
                extra_weights.len()
            );
            weight_map.extend(extra_weights);
        }

        Ok(WeightStore::from_map(weight_map))
    }
}

/// Index file format: { "weight_map": { "tensor_name": "shard_filename" } }
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn combined_admission_bytes(
    weight_bytes: usize,
    multiplier: f64,
    construction_bytes: usize,
    reserve_bytes: usize,
) -> Result<(usize, usize)> {
    if !multiplier.is_finite() || multiplier < 1.0 {
        bail!("OOM pre-flight: invalid peak-memory multiplier {multiplier}");
    }
    // Scale the exact binary significand and round upward without first
    // rounding large integer byte counts through f64.
    let bits = multiplier.to_bits();
    let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1075;
    let numerator = (weight_bytes as u128)
        .checked_mul(significand as u128)
        .context("OOM pre-flight: weight peak estimate overflows u128")?;
    let scaled = if numerator == 0 {
        0
    } else if exponent < 0 {
        numerator.div_ceil(1u128 << (-exponent as u32))
    } else {
        let factor = 1u128
            .checked_shl(exponent as u32)
            .context("OOM pre-flight: weight peak multiplier overflows u128")?;
        numerator
            .checked_mul(factor)
            .context("OOM pre-flight: weight peak estimate overflows u128")?
    };
    let scaled =
        usize::try_from(scaled).context("OOM pre-flight: weight peak estimate overflows usize")?;
    let peak = scaled
        .checked_add(construction_bytes)
        .context("OOM pre-flight: retained construction estimate overflows usize")?;
    let required = peak
        .checked_add(reserve_bytes)
        .context("OOM pre-flight: peak plus safety reserve overflows usize")?;
    Ok((peak, required))
}

/// Read only the safetensor header from a file (no mmap, no GPU memory impact).
/// The header is typically a few KB of JSON — safe to read on GB10 unified memory
/// without consuming GPU pages.
pub(crate) fn read_safetensor_header(
    path: &Path,
) -> Result<Vec<(String, Vec<usize>, safetensors::Dtype)>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Pre-flight: failed to open {}", path.display()))?;

    // Safetensors format: 8-byte LE header size, then JSON header, then data.
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Sanity check: header shouldn't exceed 64 MB.
    if header_size > 64 * 1024 * 1024 {
        bail!(
            "Safetensor header too large ({} bytes) in {}",
            header_size,
            path.display()
        );
    }

    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    // Parse the JSON header manually to extract tensor metadata.
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let obj = header.as_object().context("Invalid safetensor header")?;

    let mut tensors = Vec::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let dtype = match dtype_str {
            "F64" => safetensors::Dtype::F64,
            "I64" => safetensors::Dtype::I64,
            "U64" => safetensors::Dtype::U64,
            "F32" => safetensors::Dtype::F32,
            "I32" => safetensors::Dtype::I32,
            "U32" => safetensors::Dtype::U32,
            "F16" => safetensors::Dtype::F16,
            "BF16" => safetensors::Dtype::BF16,
            "I16" => safetensors::Dtype::I16,
            "U16" => safetensors::Dtype::U16,
            "I8" => safetensors::Dtype::I8,
            "U8" => safetensors::Dtype::U8,
            "F8_E4M3" => safetensors::Dtype::F8_E4M3,
            "F8_E5M2" => safetensors::Dtype::F8_E5M2,
            "F8_E8M0" => safetensors::Dtype::F8_E8M0,
            _ => safetensors::Dtype::BF16,
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        tensors.push((name.clone(), shape, dtype));
    }
    Ok(tensors)
}

/// Scan safetensor file headers (metadata only, no data loaded) to estimate
/// total GPU bytes this rank will load. Reads only the JSON header from each
/// file — does NOT mmap, so it's safe on GB10 unified memory.
pub(crate) fn estimate_load_bytes(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
) -> Result<usize> {
    let mut total = 0usize;
    for path in files {
        for (name, shape, dtype) in read_safetensor_header(path)? {
            if skip_fn(&name) {
                continue;
            }
            let bytes = estimated_tensor_bytes(&shape, dtype)?;
            total = total
                .checked_add(bytes)
                .context("OOM pre-flight: total weight bytes overflow usize")?;
        }
    }
    Ok(total)
}

/// Check if the model is predominantly FP8 (>50% of weight bytes are FP8).
/// Sehyo NVFP4 models have a few FP8 scale tensors but the bulk is uint8 (NVFP4 packed).
/// True FP8 checkpoints (e.g. Qwen/Qwen3.5-122B-A10B-FP8) have most bytes as FP8.
pub(crate) fn estimate_has_fp8(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
) -> Result<bool> {
    let mut fp8_bytes = 0usize;
    let mut total_bytes = 0usize;
    for path in files {
        for (name, shape, dtype) in read_safetensor_header(path)? {
            if skip_fn(&name) {
                continue;
            }
            let bytes = estimated_tensor_bytes(&shape, dtype)?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .context("OOM pre-flight: total weight bytes overflow usize")?;
            if matches!(
                dtype,
                safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2
            ) {
                fp8_bytes = fp8_bytes
                    .checked_add(bytes)
                    .context("OOM pre-flight: FP8 weight bytes overflow usize")?;
            }
        }
    }
    let fp8_frac = if total_bytes > 0 {
        fp8_bytes as f64 / total_bytes as f64
    } else {
        0.0
    };
    tracing::debug!(
        "FP8 fraction: {:.1}% ({} / {} bytes)",
        fp8_frac * 100.0,
        fp8_bytes,
        total_bytes
    );
    Ok(fp8_frac > 0.5)
}

fn estimated_tensor_bytes(shape: &[usize], dtype: safetensors::Dtype) -> Result<usize> {
    let numel = shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .context("OOM pre-flight: tensor element count overflows usize")
    })?;
    let elem_bytes = match dtype {
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F16
        | safetensors::Dtype::BF16
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::I8
        | safetensors::Dtype::U8
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::F8_E5M2
        | safetensors::Dtype::F8_E8M0 => 1,
        _ => 2,
    };
    numel
        .checked_mul(elem_bytes)
        .context("OOM pre-flight: tensor byte count overflows usize")
}

pub(crate) fn check_oom_guard(
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
    phase: &str,
) -> Result<()> {
    let free = gpu.free_memory()?;
    if free < reserve_bytes {
        let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
        let reserve_gb = reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        bail!(
            "OOM guard: aborting during {phase}. \
             Free GPU memory ({free_gb:.2} GB) is below the {reserve_gb:.1} GB safety reserve. \
             This model is too large for available GPU memory. \
             Reduce --max-seq-len, lower --oom-guard-mb (if safe), or use a smaller model."
        );
    }
    Ok(())
}
mod load_fns;
use load_fns::{load_sharded, load_single, read_index_shards};

#[cfg(test)]
mod tests;
