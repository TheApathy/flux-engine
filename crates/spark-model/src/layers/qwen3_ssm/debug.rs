// SPDX-License-Identifier: AGPL-3.0-only
//
// Env-gated debug dumpers. Wired through ATLAS_GDN_DUMP* environment variables
// and only activated for offline diagnostic runs, so the items here are
// legitimately "never used" in a default build.
#![allow(dead_code)]

//! Debug print helpers + env-gated GDN intermediate dumper for
//! per-layer numerical comparison against an external reference. See
//! `docs/DEBUGGING_METHODOLOGY.md`.
//!
//! Set `ATLAS_GDN_DUMP=<dir>` to enable, optionally
//! `ATLAS_GDN_DUMP_LAYERS=0,15,29` (comma-list of SSM-layer indices, default
//! `0`), and `ATLAS_GDN_DUMP_N_SSM=<count>` (default `30`) to fold the
//! monotonic call counter modulo the model's SSM layer count across chunks.
//!
//! Filenames: `gdnsub_step0_L{effective_layer}_{stage}.bin` where stage ∈
//! {conv, l2, gdn, gnorm}. Headerless little-endian BF16, last-token slice of
//! the prefill. Calls overwrite the file on each chunk so the final file holds
//! the last chunk's capture.

use std::sync::atomic::AtomicUsize;

use super::*;

/// Max SSM layers we can track. A3B has 30 SSM layers; 27B has 48.
/// Set well above either.
pub(super) const MAX_SSM_LAYERS: usize = 64;

/// Monotonic call counter incremented per SSM layer invocation; folded modulo
/// `ATLAS_GDN_DUMP_N_SSM` inside `maybe_dump_gdn_buf` so chunked prefills map
/// each layer back to its `effective_layer` index and overwrite with the final
/// chunk.
pub(super) static SSM_LAYER_CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn dump_layers_from_env() -> Vec<usize> {
    std::env::var("ATLAS_GDN_DUMP_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Snapshot `n_elements` BF16 values starting at `ptr + byte_offset` to
/// `<ATLAS_GDN_DUMP>/gdnsub_step0_L{effective_layer}_{stage}.bin`, where
/// `effective_layer = layer_idx % n_ssm.max(1)`. Skips when `ATLAS_GDN_DUMP`
/// is unset or `effective_layer` is not in `ATLAS_GDN_DUMP_LAYERS`. Overwrites
/// the dump file on each call so chunked prefills retain the last chunk.
pub(super) fn maybe_dump_gdn_buf(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    stage: &str,
    stream: u64,
) -> Result<()> {
    // Dumps the LAST CHUNK's last token: under chunked prefill, this
    // function fires once per scheduler chunk per SSM layer. We do NOT
    // latch — we OVERWRITE the dump file on every call so the final
    // file on disk corresponds to the LAST chunk's last-token capture
    // (= position L-1 of the full prefill, not position chunk_len-1 of
    // the first chunk). The `layer_idx` modulo num-SSM-layers handles
    // the monotonic SSM_LAYER_CALL_COUNTER wrapping across multiple
    // scheduler chunks so each chunk re-dumps the same layers.
    let dir = match std::env::var("ATLAS_GDN_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return Ok(()),
    };
    // Resolve layer_idx % num_linear_attention_layers via env hint
    // (default 30 for A3B; 48 for dense 27B). Set via
    // ATLAS_GDN_DUMP_N_SSM=<count> if needed.
    let n_ssm: usize = std::env::var("ATLAS_GDN_DUMP_N_SSM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let effective_layer = layer_idx % n_ssm.max(1);
    if effective_layer >= MAX_SSM_LAYERS {
        return Ok(());
    }
    let layers = dump_layers_from_env();
    if !layers.contains(&effective_layer) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let bytes = n_elements * 2;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(ptr.offset(byte_offset), &mut buf)?;
    let path =
        std::path::Path::new(&dir).join(format!("gdnsub_step0_L{effective_layer}_{stage}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &buf)?;
    tracing::info!(
        "ATLAS_GDN_DUMP: wrote {} ({} bf16 elements, raw_idx={layer_idx}, effective_layer={effective_layer})",
        path.display(),
        n_elements
    );
    Ok(())
}

impl Qwen3SsmLayer {
    /// Debug: read first N BF16 values from device and log them.
    pub(super) fn debug_bf16(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 2];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let lo = buf[i * 2];
                let hi = buf[i * 2 + 1];
                f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }

    /// Debug: read first N FP32 values from device and log them.
    pub(super) fn debug_f32(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 4];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }
}
