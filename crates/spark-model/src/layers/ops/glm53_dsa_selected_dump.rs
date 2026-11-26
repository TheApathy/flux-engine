// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM53_DSA_SELECTED_DUMP=<dir>` (diagnostic, default off): before the
//! first `ATLAS_GLM53_DSA_SELECTED_DUMP_MAX` (default 2) prompt-chunk launches of
//! DSA selected attention, synchronize the stream and write the launch's raw
//! inputs (absorbed query, latent cache, selected indices, positions, validity,
//! sequence lengths) plus a JSON header to `<dir>/dsa-<n>-*` for offline
//! microbenchmark replay through the exact and tensor-core kernels. Never on in
//! a timed run: it adds a host fence and ~300 MB of I/O.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm53_dsa_selected_attention::{
    Glm53DsaSelectedAttentionBuffers, Glm53DsaSelectedAttentionPlan,
};

pub(super) fn maybe_dump(
    gpu: &dyn GpuBackend,
    plan: Glm53DsaSelectedAttentionPlan,
    buffers: Glm53DsaSelectedAttentionBuffers,
    stream: u64,
) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = std::env::var_os("ATLAS_GLM53_DSA_SELECTED_DUMP") else {
        return Ok(());
    };
    let max: u64 = match std::env::var("ATLAS_GLM53_DSA_SELECTED_DUMP_MAX") {
        Ok(v) => v
            .parse()
            .context("ATLAS_GLM53_DSA_SELECTED_DUMP_MAX must be an integer")?,
        Err(_) => 2,
    };
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    if index >= max {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).context("GLM DSA selected dump dir")?;
    let parts: [(&str, DevicePtr, usize); 6] = [
        ("q", buffers.absorbed_query_bf16.ptr, plan.query_bytes),
        (
            "latent",
            buffers.latent_cache_bf16.ptr,
            plan.latent_cache_bytes,
        ),
        (
            "idx",
            buffers.selected_indices_i32.ptr,
            plan.selected_index_bytes,
        ),
        (
            "seq",
            buffers.sequence_lengths_u32.ptr,
            plan.sequence_length_bytes,
        ),
        (
            "pos",
            buffers.query_positions_u32.ptr,
            plan.query_position_bytes,
        ),
        (
            "valid",
            buffers.query_validity_u8.ptr,
            plan.query_validity_bytes,
        ),
    ];
    for (name, ptr, bytes) in parts {
        let mut host = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut host)?;
        let path = dir.join(format!("dsa-{index}-{name}.bin"));
        std::fs::write(&path, &host).with_context(|| format!("write {}", path.display()))?;
    }
    let header = format!(
        "{{\"batch\":{},\"queries\":{},\"kv_capacity\":{}}}\n",
        plan.batch, plan.queries, plan.kv_capacity
    );
    std::fs::write(dir.join(format!("dsa-{index}.json")), header)
        .context("GLM DSA selected dump header")?;
    Ok(())
}
