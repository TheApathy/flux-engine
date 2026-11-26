// SPDX-License-Identifier: AGPL-3.0-only

//! Operand capture for the tensor-scale FP8 projections (diagnostic, default
//! off): the inputs a hand FP8 GEMM must reproduce cuBLASLt on, byte for byte
//! (gate G1 of the conv -> QKVZ epilogue lever).
//!
//! `ATLAS_QWEN4_FP8_PROJ_DUMP_DIR=<dir>` writes, for every tensor-scale call of
//! output width `ATLAS_QWEN4_FP8_PROJ_DUMP_N` (default 16384, the Flash-Next
//! QKVZ projection) whose index among those calls is below
//! `ATLAS_QWEN4_FP8_PROJ_DUMP_MAX_CALLS` (default 36, the first chunk) and whose
//! slot (`index % 36`, one per GDN layer) is in `ATLAS_QWEN4_FP8_PROJ_DUMP_SLOTS`
//! (default "0,17,35": layers 0, 22 and 46): the e4m3 activation rows (`.a8`,
//! [m, k]), the e4m3 weight rows (`.w8`, [n, k]), the cuBLASLt BF16 output
//! (`.out`, [m, n]) and `.meta.json` with the shape, both device scales and
//! alpha. Every write follows a stream sync, outside any timed run.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const SLOTS: u64 = 36;

struct DumpCfg {
    dir: std::path::PathBuf,
    n: usize,
    slots: Vec<u64>,
    max_calls: u64,
}

fn cfg() -> Option<&'static DumpCfg> {
    static CFG: OnceLock<Option<DumpCfg>> = OnceLock::new();
    CFG.get_or_init(|| {
        let dir = std::env::var("ATLAS_QWEN4_FP8_PROJ_DUMP_DIR").ok()?;
        let var = |name: &str| std::env::var(name).ok();
        Some(DumpCfg {
            dir: dir.into(),
            n: var("ATLAS_QWEN4_FP8_PROJ_DUMP_N")
                .and_then(|v| v.parse().ok())
                .unwrap_or(16384),
            slots: parse_slots(
                var("ATLAS_QWEN4_FP8_PROJ_DUMP_SLOTS")
                    .as_deref()
                    .unwrap_or("0,17,35"),
            ),
            max_calls: var("ATLAS_QWEN4_FP8_PROJ_DUMP_MAX_CALLS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(SLOTS),
        })
    })
    .as_ref()
}

fn parse_slots(list: &str) -> Vec<u64> {
    list.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn stem(call: u64, m: usize, n: usize) -> String {
    format!("fp8proj_call{call:04}_slot{:02}_m{m}_n{n}", call % SLOTS)
}

/// One tensor-scale GEMM `out[m, n] (ldc) = alpha * a_scale * w_scale *
/// (a8[m, k] . w8[n, k])`, after it was enqueued on `stream`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_dump(
    gpu: &dyn GpuBackend,
    a8: DevicePtr,
    a_scale: DevicePtr,
    w8: DevicePtr,
    w_scale: DevicePtr,
    alpha: f32,
    out: DevicePtr,
    (m, n, k, ldc): (usize, usize, usize, usize),
    stream: u64,
) -> Result<()> {
    let Some(cfg) = cfg() else {
        return Ok(());
    };
    if n != cfg.n {
        return Ok(());
    }
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    if call >= cfg.max_calls || !cfg.slots.contains(&(call % SLOTS)) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let mut a = vec![0u8; m * k];
    let mut w = vec![0u8; n * k];
    let mut scales = [0u8; 8];
    gpu.copy_d2h(a8, &mut a)?;
    gpu.copy_d2h(w8, &mut w)?;
    gpu.copy_d2h(a_scale, &mut scales[..4])?;
    gpu.copy_d2h(w_scale, &mut scales[4..])?;
    let mut o = vec![0u8; m * n * 2];
    if ldc == n {
        gpu.copy_d2h(out, &mut o)?;
    } else {
        for (row, dst) in o.chunks_exact_mut(n * 2).enumerate() {
            gpu.copy_d2h(out.offset(row * ldc * 2), dst)?;
        }
    }
    let a_scale = f32::from_le_bytes(scales[..4].try_into()?);
    let w_scale = f32::from_le_bytes(scales[4..].try_into()?);
    std::fs::create_dir_all(&cfg.dir)?;
    let path = cfg.dir.join(stem(call, m, n));
    std::fs::write(path.with_extension("a8"), &a)?;
    std::fs::write(path.with_extension("w8"), &w)?;
    std::fs::write(path.with_extension("out"), &o)?;
    std::fs::write(
        path.with_extension("meta.json"),
        format!(
            "{{\"m\":{m},\"n\":{n},\"k\":{k},\"ldc\":{ldc},\"a_scale_bits\":{},\"w_scale_bits\":{},\"alpha_bits\":{}}}\n",
            a_scale.to_bits(),
            w_scale.to_bits(),
            alpha.to_bits()
        ),
    )?;
    tracing::info!(
        "QWEN4_FP8_PROJ_DUMP call={call} m={m} n={n} k={k} -> {}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_and_names() {
        assert_eq!(parse_slots("0,17, 35,x"), vec![0, 17, 35]);
        assert_eq!(
            stem(53, 8192, 16384),
            "fp8proj_call0053_slot17_m8192_n16384"
        );
    }
}
