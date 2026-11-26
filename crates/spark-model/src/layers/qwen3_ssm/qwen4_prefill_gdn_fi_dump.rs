// SPDX-License-Identifier: AGPL-3.0-only

//! Offline-parity capture for the Flash-Next GDN block (diagnostic, default off).
//!
//! `ATLAS_QWEN4_GDN_FI_DUMP_DIR=<dir>` writes, for every exact-route SSM layer
//! call whose in-chunk slot (`call % 36`, Flash-Next's 36 GDN layers) is in
//! `ATLAS_QWEN4_GDN_FI_DUMP_LAYERS` (comma list, default "0") while the call
//! counter is below `ATLAS_QWEN4_GDN_FI_DUMP_MAX_CALLS` (default 4096), the
//! block's inputs BEFORE the conv runs, on either route: the deinterleaved
//! projection rows (`.x`, bf16 [rows, QKVZ]), the gate rows (`.gb`, fp32
//! [rows, 96]) and the incoming H (`.h`) and conv (`.cs`) states for offline
//! parity replay across the native and FlashInfer GDN routes.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::qwen4_prefill_exact_plan::SsmShape;
use super::*;

const SLOTS: u64 = 36;

struct DumpCfg {
    dir: std::path::PathBuf,
    layers: Vec<u64>,
    max_calls: u64,
}

fn cfg() -> Option<&'static DumpCfg> {
    static CFG: OnceLock<Option<DumpCfg>> = OnceLock::new();
    CFG.get_or_init(|| {
        let dir = std::env::var("ATLAS_QWEN4_GDN_FI_DUMP_DIR").ok()?;
        let layers = std::env::var("ATLAS_QWEN4_GDN_FI_DUMP_LAYERS")
            .unwrap_or_else(|_| "0".into())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let max_calls = std::env::var("ATLAS_QWEN4_GDN_FI_DUMP_MAX_CALLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        Some(DumpCfg {
            dir: dir.into(),
            layers,
            max_calls,
        })
    })
    .as_ref()
}

pub(super) fn maybe_dump_inputs(
    state: &SsmLayerState,
    deinterleaved: DevicePtr,
    gates: DevicePtr,
    rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let Some(cfg) = cfg() else {
        return Ok(());
    };
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    if call >= cfg.max_calls || !cfg.layers.contains(&(call % SLOTS)) {
        return Ok(());
    }
    let s = SsmShape::from_heads(
        ctx.config.linear_num_key_heads,
        ctx.config.linear_num_value_heads,
    );
    ctx.gpu.synchronize(stream)?;
    let mut x = vec![0u8; rows * s.qkvz * 2];
    let mut gb = vec![0u8; rows * s.gates * 4];
    let mut h = vec![0u8; s.h_state_bytes];
    let mut cs = vec![0u8; s.conv_state_bytes];
    ctx.gpu.copy_d2h(deinterleaved, &mut x)?;
    ctx.gpu.copy_d2h(gates, &mut gb)?;
    ctx.gpu.copy_d2h(state.h_state, &mut h)?;
    ctx.gpu.copy_d2h(state.conv_state, &mut cs)?;
    std::fs::create_dir_all(&cfg.dir)?;
    let stem = cfg
        .dir
        .join(format!("call{call:06}_slot{:02}_T{rows}", call % SLOTS));
    for (ext, bytes) in [("x", &x), ("gb", &gb), ("h", &h), ("cs", &cs)] {
        std::fs::write(stem.with_extension(ext), bytes)?;
    }
    tracing::info!(
        "QWEN4_GDN_FI_DUMP call={call} rows={rows} -> {}",
        stem.display()
    );
    Ok(())
}
