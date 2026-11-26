// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic dump of the QSA-sparse prefill core (offline parity and union
//! statistics). `ATLAS_QWEN4_PREFILL_QSA_DUMP=<dir>` together with
//! `ATLAS_QWEN4_PREFILL_QSA_DUMP_LAYERS=<attn layer idx,...>` writes, per
//! listed layer and chunk, `<dir>/L<layer>_S<start>_R<rows>/`:
//! `meta.json`, `hidden`, `iq` (post norm/RoPE), `masks`, `counts`, `q`,
//! `out` (raw core output), and the LOGICAL-order `ckeys`, `k`, `v` gathered
//! page by page through the block table. `ATLAS_QWEN4_PREFILL_QSA_STATS=1`
//! logs union sizes for every layer and chunk without writing files.
//! Both synchronise the prefill stream; never time a run that sets them.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::prefill_sparse::{SELECTOR, SparseChunk, SparsePlan, TILE_ROWS};
use super::{COMPRESS_RATIO, INDEX_DIM, INDEX_HEADS, QUERY_HEADS, Qwen4QsaIndexer};
use crate::layer::AttnMetadataDev;

struct DumpConfig {
    dir: Option<PathBuf>,
    layers: Vec<usize>,
    stats: bool,
}

fn config() -> &'static DumpConfig {
    static C: OnceLock<DumpConfig> = OnceLock::new();
    C.get_or_init(|| {
        let dir = std::env::var("ATLAS_QWEN4_PREFILL_QSA_DUMP")
            .ok()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from);
        let layers = std::env::var("ATLAS_QWEN4_PREFILL_QSA_DUMP_LAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default();
        let stats = std::env::var("ATLAS_QWEN4_PREFILL_QSA_STATS").is_ok_and(|v| v == "1");
        DumpConfig { dir, layers, stats }
    })
}

pub(crate) fn dump_enabled() -> bool {
    config().dir.is_some()
}

pub(crate) fn stats_or_dump_enabled() -> bool {
    let cfg = config();
    cfg.stats || cfg.dir.is_some()
}

fn read(gpu: &dyn GpuBackend, src: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(src, &mut buf)?;
    Ok(buf)
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    std::fs::write(dir.join(name), bytes).with_context(|| format!("{SELECTOR} dump {name}"))
}

/// Gather `count` logical pages of `page_bytes` each through the block table.
fn gather_pages(
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    table: &[i32],
    page_bytes: usize,
    used_bytes: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(table.len() * page_bytes);
    for &phys in table {
        out.extend(read(
            gpu,
            pool.offset(phys as usize * page_bytes),
            page_bytes,
        )?);
    }
    out.truncate(used_bytes);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn maybe_dump(
    qsa: &Qwen4QsaIndexer,
    c: &SparseChunk,
    p: &SparsePlan,
    base: DevicePtr,
    kv_cache: &PagedKvCache,
    meta: AttnMetadataDev,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    let cfg = config();
    let dump = cfg.dir.is_some() && cfg.layers.contains(&c.attn_layer_idx);
    if !cfg.stats && !dump {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let counts = read(gpu, base.offset(p.count_off), p.tiles * 4)?;
    let counts: Vec<u32> = counts
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mean = counts.iter().map(|&x| x as f64).sum::<f64>() / counts.len().max(1) as f64;
    let max = counts.iter().copied().max().unwrap_or(0);
    tracing::info!(
        "QWEN4_PREFILL_QSA_STATS layer={} start={} rows={} tile_rows={TILE_ROWS} union_mean={mean:.1} union_max={max} union_over_513={:.3}",
        c.attn_layer_idx,
        c.start,
        c.rows,
        mean / 513.0
    );
    let Some(root) = cfg.dir.as_ref().filter(|_| dump) else {
        return Ok(());
    };
    let dir = root.join(format!("L{}_S{}_R{}", c.attn_layer_idx, c.start, c.rows));
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("{SELECTOR} dump dir {}", dir.display()))?;
    let total = c.start + c.rows;
    let page_size = kv_cache.block_size();
    let pages = total.div_ceil(page_size);
    let table = read(gpu, meta.block_table, pages * 4)?;
    let table: Vec<i32> = table
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let nq = if c.num_query_heads == 0 {
        QUERY_HEADS
    } else {
        c.num_query_heads
    };
    let row = nq * 256 * 2;
    let iq_row = INDEX_HEADS as usize * INDEX_DIM as usize * 2;
    write(
        &dir,
        "hidden.bin",
        &read(gpu, c.hidden, c.rows * c.hidden_size as usize * 2)?,
    )?;
    write(
        &dir,
        "iq.bin",
        &read(gpu, base.offset(p.iq_off), c.rows * iq_row)?,
    )?;
    write(
        &dir,
        "masks.bin",
        &read(gpu, base.offset(p.mask_off), c.rows * p.mask_words * 4)?,
    )?;
    write(
        &dir,
        "counts.bin",
        &counts
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    // Q rows are strided (fused QKV); keep only the nq*256 query columns.
    let q_span = read(gpu, c.q, ((c.rows - 1) * c.q_stride as usize) * 2 + row)?;
    let q_rows: Vec<u8> = (0..c.rows)
        .flat_map(|r| {
            let o = r * c.q_stride as usize * 2;
            q_span[o..o + row].to_vec()
        })
        .collect();
    write(&dir, "q.bin", &q_rows)?;
    write(&dir, "out.bin", &read(gpu, c.out, c.rows * row)?)?;
    if !c.gate.is_null() {
        // Strided gate rows exactly as the fused epilogue reads them.
        let elems = (c.rows - 1) * c.gate_stride as usize + nq * 256;
        write(&dir, "gate.bin", &read(gpu, c.gate, elems * 2)?)?;
    }
    write(
        &dir,
        "block_table.bin",
        &table
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let layer = c.attn_layer_idx;
    let kv_page = kv_cache.k_block_stride_bytes_for_layer(layer);
    let kv_used = total * kv_page / page_size;
    write(
        &dir,
        "k.bin",
        &gather_pages(gpu, kv_cache.k_pool_ptr(layer), &table, kv_page, kv_used)?,
    )?;
    write(
        &dir,
        "v.bin",
        &gather_pages(gpu, kv_cache.v_pool_ptr(layer), &table, kv_page, kv_used)?,
    )?;
    let cache = qsa.cache(gpu, kv_cache)?;
    let ck_page = page_size / COMPRESS_RATIO * INDEX_DIM as usize * 2;
    let ck_used = total / COMPRESS_RATIO * INDEX_DIM as usize * 2;
    write(
        &dir,
        "ckeys.bin",
        &gather_pages(gpu, cache.compressed_keys, &table, ck_page, ck_used)?,
    )?;
    let meta_json = format!(
        "{{\"layer\":{layer},\"start\":{},\"rows\":{},\"blocks\":{},\"mask_words\":{},\"tile_rows\":{TILE_ROWS},\"tiles\":{},\"sparse_row0\":{},\"page_size\":{page_size},\"hidden\":{},\"softmax_scale\":{},\"eps\":{},\"rope_theta\":{},\"rotary_dim\":{},\"gate_stride\":{}}}\n",
        c.start,
        c.rows,
        p.blocks,
        p.mask_words,
        p.tiles,
        p.sparse_row0,
        c.hidden_size,
        c.softmax_scale,
        c.eps,
        c.rope_theta,
        c.rotary_dim,
        c.gate_stride
    );
    write(&dir, "meta.json", meta_json.as_bytes())?;
    tracing::info!("QWEN4_PREFILL_QSA_DUMP wrote {}", dir.display());
    Ok(())
}
