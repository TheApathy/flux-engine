// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side scheduling for the Flash-Next PLE prefill stage (default off).
//!
//! - `ATLAS_QWEN4_PLE_PREFILL_ASYNC=1`: the chunk's packed rows go through a
//!   pinned double buffer with `cuMemcpyHtoDAsync` on the prefill stream, and
//!   the stage ends with an event instead of the pageable group copy (which
//!   synchronizes the stream first) and the closing `synchronize`.
//! - `ATLAS_QWEN4_PLE_PREFILL_PIPELINE=1` (implies ASYNC): a chunk whose row
//!   read is still in flight when layer 1 needs it is consumed in blocks of
//!   [`BLOCK_TOKENS`] tokens as the reader publishes them, so the PLE kernels
//!   of the head of the chunk run under the I/O of its tail. Per block, the
//!   projections run with the algorithm cuBLASLt picks for the whole chunk
//!   (`bf16_gemm_act_weight_t_ref_m`); per-row identity with the whole-chunk
//!   GEMM holds only for algorithms whose reduction does not depend on M, so
//!   the endpoint identity check decides.
//! - `ATLAS_QWEN4_PLE_PREFILL_PIPELINE=2`: the fallback that is identical by
//!   construction: only staging and dequant run per block; the projections,
//!   prepare and conv run once over the whole chunk as today.
//!
//! - `ATLAS_QWEN4_PLE_PREFILL_TRIGGER=<path>` (optional): both apply only while
//!   that file exists (checked per chunk), so one load can serve an A/B/A.
//!   Either read kind is consumable by either path, so a flip is always safe.
//!
//! The causal conv of a block reads its left context from a BF16 history the
//! chunk-end commit kernel writes from the previous block's prepared rows,
//! the same values the in-chunk path rounds. Only timing changes.

use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, PinnedHostBuffer};

pub(crate) const ASYNC_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_ASYNC";
pub(crate) const PIPELINE_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_PIPELINE";
pub(crate) const TRIGGER: &str = "ATLAS_QWEN4_PLE_PREFILL_TRIGGER";

/// Tokens per pipelined block.
pub(crate) const BLOCK_TOKENS: usize = 1024;

fn parse_switch(name: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{name} must be absent, 0 or 1; got {other:?}")),
    }
}

fn switch(cell: &'static OnceLock<Result<bool, String>>, name: &'static str) -> Result<bool> {
    cell.get_or_init(|| parse_switch(name, std::env::var(name).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn parse_level(value: Option<&str>) -> Result<u8, String> {
    match value {
        None | Some("") | Some("0") => Ok(0),
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some(other) => Err(format!(
            "{PIPELINE_SELECTOR} must be absent, 0, 1 or 2; got {other:?}"
        )),
    }
}

fn armed_by(trigger: Option<&std::path::Path>) -> bool {
    trigger.is_none_or(std::path::Path::exists)
}

/// [`TRIGGER`] is absent, or its file exists now.
fn trigger_armed() -> bool {
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    armed_by(
        PATH.get_or_init(|| {
            std::env::var(TRIGGER)
                .ok()
                .filter(|path| !path.is_empty())
                .map(std::path::PathBuf::from)
        })
        .as_deref(),
    )
}

/// 0 off, 1 per-block projections, 2 per-block staging and dequant only.
pub(crate) fn pipeline_level() -> Result<u8> {
    static LEVEL: OnceLock<Result<u8, String>> = OnceLock::new();
    let level = LEVEL
        .get_or_init(|| parse_level(std::env::var(PIPELINE_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)?;
    Ok(if trigger_armed() { level } else { 0 })
}

pub(crate) fn pipeline_selected() -> Result<bool> {
    Ok(pipeline_level()? > 0)
}

pub(crate) fn async_selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    Ok((switch(&SEL, ASYNC_SELECTOR)? && trigger_armed()) || pipeline_selected()?)
}

/// Token ranges of `rows` tokens in blocks of `block` (the last may be shorter).
pub(crate) fn block_ranges(rows: usize, block: usize) -> Vec<Range<usize>> {
    if block == 0 {
        return Vec::new();
    }
    (0..rows.div_ceil(block))
        .map(|b| b * block..((b + 1) * block).min(rows))
        .collect()
}

/// Execution schedule for a pipelined PLE block read once entry-time block
/// readiness has been sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockReadSchedule {
    /// Every block of the chunk is already ready at layer-1 entry: stage,
    /// dequantize, project, and inject the whole chunk in one coalesced pass.
    WholeChunkCoalesced,
    /// Storage reads are still in flight: optionally coalesce a multi-block
    /// ready prefix (`ready_prefix >= 2`) and stream remaining blocks as they
    /// arrive so GPU work overlaps tail NVMe I/O.
    Streaming {
        ready_prefix: usize,
        per_block_projection: bool,
    },
}

impl BlockReadSchedule {
    /// Log label describing the projection strategy taken for this chunk.
    #[must_use]
    pub(crate) const fn projection_label(self) -> &'static str {
        match self {
            Self::WholeChunkCoalesced => "whole_chunk_coalesced",
            Self::Streaming {
                per_block_projection: true,
                ..
            } => "per_block_ref_m",
            Self::Streaming {
                per_block_projection: false,
                ..
            } => "whole_chunk",
        }
    }
}

/// Choose whether a pipelined block read should coalesce the whole chunk or
/// stream blocks while storage I/O finishes.
#[must_use]
pub(crate) const fn schedule_block_read(
    ready_at_entry: usize,
    total_blocks: usize,
    level: u8,
) -> BlockReadSchedule {
    if total_blocks > 0 && ready_at_entry >= total_blocks {
        BlockReadSchedule::WholeChunkCoalesced
    } else {
        BlockReadSchedule::Streaming {
            ready_prefix: if ready_at_entry >= 2 {
                ready_at_entry
            } else {
                0
            },
            per_block_projection: level == 1,
        }
    }
}

/// One pinned staging buffer: `max_rows` record rows, then `max_rows` scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StagingLayout {
    pub record_bytes: usize,
    pub max_rows: usize,
}

impl StagingLayout {
    pub(crate) fn bytes(&self) -> usize {
        self.max_rows * (self.record_bytes + std::mem::size_of::<f32>())
    }

    /// Byte ranges of the records and of the scales of rows `rows`.
    pub(crate) fn ranges(&self, rows: Range<usize>) -> (Range<usize>, Range<usize>) {
        let scales = self.max_rows * self.record_bytes;
        let scale = std::mem::size_of::<f32>();
        (
            rows.start * self.record_bytes..rows.end * self.record_bytes,
            scales + rows.start * scale..scales + rows.end * scale,
        )
    }
}

/// Pinned double buffer, the event that frees each half, and the device
/// scratch that holds a block's conv history.
pub(crate) struct Staging {
    pub layout: StagingLayout,
    buffers: [PinnedHostBuffer; 2],
    events: [u64; 2],
    recorded: [bool; 2],
    next: usize,
    pub history_dev: DevicePtr,
    history_bytes: usize,
}

impl Staging {
    /// Take the next half, waiting (host side) until the copies last issued
    /// from it have completed.
    pub(crate) fn acquire(&mut self, gpu: &dyn GpuBackend) -> Result<usize> {
        let index = self.next;
        self.next ^= 1;
        if self.recorded[index] {
            while !gpu.poll_event(self.events[index])? {
                std::thread::yield_now();
            }
            self.recorded[index] = false;
        }
        Ok(index)
    }

    /// Wait for any in-flight copies on both halves so the staging buffers
    /// can be safely replaced and freed.
    fn drain(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        for index in 0..self.events.len() {
            if self.recorded[index] {
                while !gpu.poll_event(self.events[index])? {
                    std::thread::yield_now();
                }
                self.recorded[index] = false;
            }
        }
        Ok(())
    }

    /// Release GPU events and device scratch owned by this staging instance.
    /// Must only be called after [`Self::drain`] has succeeded.
    fn free_gpu_resources(&self, gpu: &dyn GpuBackend) {
        for ev in self.events {
            let _ = gpu.destroy_event(ev);
        }
        let _ = gpu.free(self.history_dev);
    }

    pub(crate) fn buffer(&mut self, index: usize) -> &mut [u8] {
        self.buffers[index].as_mut_slice()
    }

    /// Mark the copies issued from `index` so far on `stream`.
    pub(crate) fn release(
        &mut self,
        gpu: &dyn GpuBackend,
        index: usize,
        stream: u64,
    ) -> Result<()> {
        gpu.record_event(self.events[index], stream)?;
        self.recorded[index] = true;
        Ok(())
    }
}

/// The process-wide staging (one Flash-Next PLE layer per process), grown
/// on demand after draining any in-flight copies and releasing prior GPU
/// resources.
pub(crate) fn staging(
    gpu: &dyn GpuBackend,
    layout: StagingLayout,
    history_bytes: usize,
) -> Result<std::sync::MutexGuard<'static, Option<Staging>>> {
    static STAGING: Mutex<Option<Staging>> = Mutex::new(None);
    let mut guard = STAGING
        .lock()
        .map_err(|_| anyhow::anyhow!("Qwen4 PLE staging poisoned"))?;
    let fits = guard
        .as_ref()
        .is_some_and(|s| s.layout.bytes() >= layout.bytes() && s.history_bytes >= history_bytes);
    if !fits {
        if let Some(old) = guard.as_mut() {
            old.drain(gpu)?;
        }
        let bytes = layout.bytes();
        let buffers = [gpu.alloc_host_pinned(bytes)?, gpu.alloc_host_pinned(bytes)?];
        let ev0 = gpu.create_event()?;
        let ev1 = match gpu.create_event() {
            Ok(ev) => ev,
            Err(err) => {
                let _ = gpu.destroy_event(ev0);
                return Err(err);
            }
        };
        let events = [ev0, ev1];
        let history_dev = match gpu
            .alloc(history_bytes)
            .context("Qwen4 PLE block conv history scratch")
        {
            Ok(ptr) => ptr,
            Err(err) => {
                for ev in events {
                    let _ = gpu.destroy_event(ev);
                }
                return Err(err);
            }
        };
        tracing::info!(
            "QWEN4_PLE_PREFILL_STAGING pinned_bytes={} history_bytes={history_bytes}",
            2 * bytes
        );
        if let Some(old) = guard.replace(Staging {
            layout,
            buffers,
            events,
            recorded: [false; 2],
            next: 0,
            history_dev,
            history_bytes,
        }) {
            old.free_gpu_resources(gpu);
        }
    }
    Ok(guard)
}

/// Log a pipeline decision once per distinct line.
pub(crate) fn log_once(line: String) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&line) {
        tracing::info!("{line}");
        seen.push(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse_strictly() {
        assert_eq!(parse_switch("X", None), Ok(false));
        assert_eq!(parse_switch("X", Some("0")), Ok(false));
        assert_eq!(parse_switch("X", Some("")), Ok(false));
        assert_eq!(parse_switch("X", Some("1")), Ok(true));
        assert!(parse_switch("X", Some("2")).is_err());
        assert!(parse_switch("X", Some("yes")).is_err());
    }

    #[test]
    fn pipeline_levels_parse_strictly() {
        assert_eq!(parse_level(None), Ok(0));
        assert_eq!(parse_level(Some("0")), Ok(0));
        assert_eq!(parse_level(Some("1")), Ok(1));
        assert_eq!(parse_level(Some("2")), Ok(2));
        assert!(parse_level(Some("3")).is_err());
        assert!(parse_level(Some("on")).is_err());
    }

    #[test]
    fn trigger_arms_only_while_its_file_exists() {
        assert!(armed_by(None));
        assert!(armed_by(Some(std::path::Path::new("/"))));
        assert!(!armed_by(Some(std::path::Path::new(
            "/nonexistent/atlas-ple-trigger-for-tests"
        ))));
    }

    #[test]
    fn blocks_cover_the_chunk_once_in_order() {
        assert_eq!(block_ranges(2048, 1024), vec![0..1024, 1024..2048]);
        assert_eq!(
            block_ranges(2500, 1024),
            vec![0..1024, 1024..2048, 2048..2500]
        );
        assert_eq!(block_ranges(7, 1024), vec![0..7]);
        assert!(block_ranges(0, 1024).is_empty());
        assert!(block_ranges(5, 0).is_empty());
    }

    #[test]
    fn staging_layout_keeps_records_then_scales() {
        let layout = StagingLayout {
            record_bytes: 90,
            max_rows: 100,
        };
        assert_eq!(layout.bytes(), 100 * 94);
        let (records, scales) = layout.ranges(10..20);
        assert_eq!(records, 900..1800);
        assert_eq!(scales, 9000 + 40..9000 + 80);
        let (all_records, all_scales) = layout.ranges(0..100);
        assert_eq!(all_records.end, all_scales.start);
        assert_eq!(all_scales.end, layout.bytes());
    }

    #[test]
    fn block_read_schedule_coalesces_ready_chunks_and_keeps_streaming_for_inflight_reads() {
        for blocks in [2usize, 4, 8, 15, 16] {
            assert_eq!(
                schedule_block_read(blocks, blocks, 1),
                BlockReadSchedule::WholeChunkCoalesced
            );
            assert_eq!(
                schedule_block_read(blocks, blocks, 2),
                BlockReadSchedule::WholeChunkCoalesced
            );
            assert_eq!(
                BlockReadSchedule::WholeChunkCoalesced.projection_label(),
                "whole_chunk_coalesced"
            );
        }
        assert_eq!(
            schedule_block_read(0, 15, 1),
            BlockReadSchedule::Streaming {
                ready_prefix: 0,
                per_block_projection: true,
            }
        );
        assert_eq!(
            schedule_block_read(1, 15, 1),
            BlockReadSchedule::Streaming {
                ready_prefix: 0,
                per_block_projection: true,
            }
        );
        assert_eq!(
            schedule_block_read(10, 15, 1),
            BlockReadSchedule::Streaming {
                ready_prefix: 10,
                per_block_projection: true,
            }
        );
        assert_eq!(
            schedule_block_read(10, 15, 2).projection_label(),
            "whole_chunk"
        );
        assert_eq!(
            schedule_block_read(10, 15, 1).projection_label(),
            "per_block_ref_m"
        );
    }
}
