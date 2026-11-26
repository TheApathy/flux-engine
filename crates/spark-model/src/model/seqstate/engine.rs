// SPDX-License-Identifier: AGPL-3.0-only

//! Device side of a `SeqStateBlob`: a sequence's state as an ordered list of
//! sections, each a list of device byte ranges (`parts`). Export copies the
//! parts device-to-host in order; import copies the same number of bytes back
//! into the destination sequence's parts. The model adapter builds both lists
//! with one function, so source and destination geometry agree by
//! construction and the blob's layout digest checks it.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::format::{BlobReader, BlobWriter, SectionKind, SectionSpec, SeqStateHeader};

/// Largest single host staging copy.
const STAGING_BYTES: usize = 8 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRegion {
    pub kind: SectionKind,
    pub index: u16,
    /// Device ranges in logical order. Adjacent ranges are merged.
    pub parts: Vec<(DevicePtr, usize)>,
}

impl StateRegion {
    pub(crate) fn new(kind: SectionKind, index: u16) -> Self {
        Self {
            kind,
            index,
            parts: Vec::new(),
        }
    }

    /// Appends a range, merging it into the previous one when contiguous
    /// (consecutive physical KV blocks become one copy).
    pub(crate) fn push(&mut self, ptr: DevicePtr, bytes: usize) {
        if bytes == 0 {
            return;
        }
        if let Some((last_ptr, last_bytes)) = self.parts.last_mut()
            && last_ptr.0.checked_add(*last_bytes as u64) == Some(ptr.0)
        {
            *last_bytes += bytes;
            return;
        }
        self.parts.push((ptr, bytes));
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.parts.iter().map(|(_, n)| *n as u64).sum()
    }

    pub(crate) fn spec(&self) -> SectionSpec {
        SectionSpec {
            kind: self.kind,
            index: self.index,
            bytes: self.bytes(),
        }
    }
}

/// K and V of every cache layer over `table`, in logical block order:
/// `[K0, V0, K1, V1, ...]`, each block at its full per-side stride (NVFP4
/// scales live inside the stride), so the bytes are exactly as stored.
pub(crate) fn kv_block_regions(
    kv: &PagedKvCache,
    table: &[u32],
    k_kind: SectionKind,
    v_kind: SectionKind,
) -> Result<Vec<StateRegion>> {
    let num_blocks = kv.num_blocks();
    for &b in table {
        ensure!(
            (b as usize) < num_blocks,
            "seqstate: block {b} outside the {num_blocks}-block pool"
        );
    }
    let mut out = Vec::with_capacity(2 * kv.num_layers());
    for layer in 0..kv.num_layers() {
        let index = u16::try_from(layer)?;
        let k_stride = kv.k_block_stride_bytes_for_layer(layer);
        let v_stride = kv.v_block_stride_bytes_for_layer(layer);
        let mut k = StateRegion::new(k_kind, index);
        let mut v = StateRegion::new(v_kind, index);
        for &b in table {
            k.push(kv.k_cache_ptr(layer, b), k_stride);
            v.push(kv.v_cache_ptr(layer, b), v_stride);
        }
        out.push(k);
        out.push(v);
    }
    Ok(out)
}

pub(crate) fn layout_of(regions: &[StateRegion]) -> Vec<SectionSpec> {
    regions.iter().map(StateRegion::spec).collect()
}

/// Every range must lie in device memory the caller owns; overlapping
/// destinations would make import order-dependent.
pub(crate) fn check_disjoint(regions: &[StateRegion]) -> Result<()> {
    let mut ranges: Vec<(u64, u64)> = regions
        .iter()
        .flat_map(|r| r.parts.iter().map(|(p, n)| (p.0, p.0 + *n as u64)))
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        ensure!(
            pair[0].1 <= pair[1].0,
            "seqstate: state ranges overlap ({:#x}..{:#x} and {:#x}..{:#x})",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1
        );
    }
    for (p, _) in regions.iter().flat_map(|r| r.parts.iter()) {
        ensure!(p.0 != 0, "seqstate: null device range");
    }
    Ok(())
}

/// Writes the blob. The caller must have synchronized every stream that may
/// still write these ranges.
pub(crate) fn capture(
    gpu: &dyn GpuBackend,
    header: &SeqStateHeader,
    regions: &[StateRegion],
    w: &mut dyn std::io::Write,
) -> Result<()> {
    check_disjoint(regions)?;
    let mut writer = BlobWriter::begin(w, header, layout_of(regions))?;
    let mut staging = Vec::new();
    for region in regions {
        writer.begin_section(region.kind, region.index)?;
        for &(ptr, bytes) in &region.parts {
            let mut done = 0usize;
            while done < bytes {
                let n = (bytes - done).min(STAGING_BYTES);
                staging.resize(n, 0);
                gpu.copy_d2h(ptr.offset(done), &mut staging[..n])?;
                writer.write(&staging[..n])?;
                done += n;
            }
        }
        writer.end_section()?;
    }
    writer.finish()
}

/// Reads the rest of an opened blob into `regions`. The geometry is checked
/// before any device write; a digest failure mid-way leaves the destination
/// partially written, so the caller must discard the destination sequence on
/// error.
pub(crate) fn apply(
    gpu: &dyn GpuBackend,
    mut reader: BlobReader<'_>,
    regions: &[StateRegion],
) -> Result<()> {
    check_disjoint(regions)?;
    reader.expect_layout(layout_of(regions))?;
    let mut staging = Vec::new();
    for region in regions {
        reader.begin_section(region.kind, region.index)?;
        for &(ptr, bytes) in &region.parts {
            let mut done = 0usize;
            while done < bytes {
                let n = (bytes - done).min(STAGING_BYTES);
                staging.resize(n, 0);
                reader.read(&mut staging[..n])?;
                gpu.copy_h2d(&staging[..n], ptr.offset(done))?;
                done += n;
            }
        }
        reader.end_section()?;
    }
    reader.finish()
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
