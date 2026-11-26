// SPDX-License-Identifier: AGPL-3.0-only

//! Learned prefill chunk cost for the SLO-budgeted interleave.
//!
//! A prediction is keyed by (row bucket, prompt-offset band). The row bucket
//! is the smallest power of two `>=` the chunk's rows (at least
//! [`MIN_BUCKET`]), so every prepared width is its own bucket and a prompt
//! tail shares the bucket of the width that contains it. The offset band
//! captures context-length cost (QSA sparse attention and the KV reads grow
//! with the offset).
//!
//! Each cell starts from [`static_chunk_ms`] and becomes an EWMA of measured
//! chunk wall time once it has [`MIN_SAMPLES`] samples. A cell without
//! samples predicts the table value scaled by a global measured/table ratio,
//! so a box running hot or cold shifts every prediction, not only the cells
//! it has seen. The first sample of each bucket is dropped as warm-up
//! (grow-only scratch and first-use library setup land there).

use std::collections::{BTreeMap, HashMap};

#[path = "prefill_slo_cost_cache.rs"]
pub mod cache;

/// Smallest row bucket (short prompts and tails).
pub const MIN_BUCKET: usize = 128;
/// Samples a cell needs before its EWMA replaces the table.
pub const MIN_SAMPLES: u32 = 2;
/// EWMA weight of a new sample.
pub const EWMA_ALPHA: f64 = 0.3;
/// Upper offset bound of each band; the last band is open.
pub const BAND_UPPER: [usize; 3] = [8_192, 32_768, 131_072];
/// Prometheus label of each band.
pub const BAND_LABELS: [&str; 4] = ["0-8k", "8k-32k", "32k-128k", "128k+"];
/// Static offset factor per band (whole-prompt rates fall from 5,276 tok/s
/// at 8K to 4,900 at 128K on the w4096 profile; the factor is the extra cost
/// of a chunk deep in the prompt, an estimate).
const BAND_FACTOR: [f64; 4] = [1.0, 1.03, 1.08, 1.15];
/// Clamp of the global measured/table ratio.
const RATIO_RANGE: (f64, f64) = (0.25, 4.0);

/// Row bucket of a chunk.
pub fn bucket(rows: usize) -> usize {
    rows.max(MIN_BUCKET).next_power_of_two()
}

/// Offset band index of a chunk start.
pub fn band(offset: usize) -> usize {
    BAND_UPPER
        .iter()
        .position(|&upper| offset < upper)
        .unwrap_or(BAND_UPPER.len())
}

/// Static per-chunk estimate in milliseconds for Qwen3.8-Flash-Next on GB10
/// with the fast-prefill family, used as an initial prior until runtime
/// measurements replace it.
///
/// Each chunk streams the active experts it touches (`touched_GB(rows) / 141 GB/s`)
/// plus ~85 us/token non-MoE overhead. Touched expert fractions (0.824 / 0.889
/// at 2048 / 4096 rows, extrapolated for smaller chunks) serve only as bootstrap
/// estimates before observed chunk timings take over.
pub fn static_chunk_ms(bucket_rows: usize, band_index: usize) -> f64 {
    let base = match bucket_rows {
        0..=128 => 155.0,
        129..=256 => 240.0,
        257..=512 => 350.0,
        513..=1024 => 445.0,
        1025..=2048 => 570.0,
        2049..=4096 => 780.0,
        4097..=8192 => 1490.0,
        8193..=16384 => 2980.0,
        16385..=32768 => 5960.0,
        other => other as f64 * 0.19,
    };
    base * BAND_FACTOR[band_index.min(BAND_FACTOR.len() - 1)]
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
    samples: u32,
    ewma_ms: f64,
}

/// One accepted measurement, for `/metrics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub bucket: usize,
    pub band: usize,
    pub ewma_ms: f64,
    /// measured − predicted (the prediction the decision used).
    pub error_ms: f64,
}

/// Per-(bucket, band) chunk-time model.
#[derive(Clone, Debug, Default)]
pub struct ChunkCostModel {
    cells: HashMap<(usize, usize), Cell>,
    warmed: Vec<usize>,
    ratio: Option<f64>,
    ratio_count: u32,
}

impl ChunkCostModel {
    /// Restore learned chunk costs from a persisted cache snapshot.
    ///
    /// Valid finite positive cells load with `samples = MIN_SAMPLES` so tick-0
    /// immediately predicts the persisted latency, while `warmed` stays empty so
    /// each bucket's first post-restart kernel-warmup sample is still dropped.
    pub fn from_persisted(entry: &cache::PersistedSloCostEntry) -> Self {
        let mut cells = HashMap::new();
        for (raw_key, persisted) in &entry.cells {
            let Some((bucket_rows, band_index)) = cache::decode_cell_key(raw_key) else {
                continue;
            };
            if !persisted.ewma_ms.is_finite() || persisted.ewma_ms <= 0.0 || persisted.samples == 0
            {
                continue;
            }
            cells.insert(
                (bucket_rows, band_index),
                Cell {
                    samples: MIN_SAMPLES,
                    ewma_ms: persisted.ewma_ms,
                },
            );
        }
        let (ratio, ratio_count) =
            if entry.global_ratio_count > 0 && entry.global_ratio_sum.is_finite() {
                let mean = entry.global_ratio_sum / f64::from(entry.global_ratio_count);
                if mean.is_finite() && mean > 0.0 {
                    (
                        Some(mean.clamp(RATIO_RANGE.0, RATIO_RANGE.1)),
                        entry.global_ratio_count,
                    )
                } else {
                    (None, 0)
                }
            } else {
                (None, 0)
            };
        Self {
            cells,
            warmed: Vec::new(),
            ratio,
            ratio_count,
        }
    }

    /// Snapshot calibrated cells and ratio state for on-disk persistence.
    pub fn to_persisted(&self, decode_step_ms: Option<f64>) -> cache::PersistedSloCostEntry {
        let mut cells = BTreeMap::new();
        for (&(bucket_rows, band_index), cell) in &self.cells {
            if cell.samples >= MIN_SAMPLES && cell.ewma_ms.is_finite() && cell.ewma_ms > 0.0 {
                cells.insert(
                    cache::encode_cell_key(bucket_rows, band_index),
                    cache::PersistedCell {
                        ewma_ms: cell.ewma_ms,
                        samples: cell.samples,
                    },
                );
            }
        }
        let (global_ratio_sum, global_ratio_count) = match self.ratio {
            Some(r) if r.is_finite() && r > 0.0 => {
                let count = self.ratio_count.max(1);
                (r * f64::from(count), count)
            }
            _ => (0.0, 0),
        };
        let decode_step_ms = decode_step_ms.filter(|ms| ms.is_finite() && *ms > 0.0);
        cache::PersistedSloCostEntry {
            cells,
            decode_step_ms,
            global_ratio_sum,
            global_ratio_count,
            updated_unix_secs: cache::unix_timestamp_secs(),
        }
    }

    /// Predicted wall time of a `rows`-row chunk starting at `offset`.
    pub fn predict_ms(&self, rows: usize, offset: usize) -> f64 {
        let key = (bucket(rows), band(offset));
        match self.cells.get(&key) {
            Some(cell) if cell.samples >= MIN_SAMPLES => cell.ewma_ms,
            _ => static_chunk_ms(key.0, key.1) * self.ratio.unwrap_or(1.0),
        }
    }

    /// Whether the prediction for this cell is measured (not table-based).
    pub fn is_measured(&self, rows: usize, offset: usize) -> bool {
        self.cells
            .get(&(bucket(rows), band(offset)))
            .is_some_and(|c| c.samples >= MIN_SAMPLES)
    }

    /// Record one measured chunk. `predicted_ms` is what the decision used.
    /// Returns `None` for the bucket's warm-up sample and for invalid input.
    pub fn observe(
        &mut self,
        rows: usize,
        offset: usize,
        measured_ms: f64,
        predicted_ms: f64,
    ) -> Option<Observation> {
        if rows == 0 || !measured_ms.is_finite() || measured_ms < 0.0 {
            return None;
        }
        let key = (bucket(rows), band(offset));
        if !self.warmed.contains(&key.0) {
            self.warmed.push(key.0);
            return None;
        }
        let cell = self.cells.entry(key).or_default();
        cell.ewma_ms = if cell.samples == 0 {
            measured_ms
        } else {
            cell.ewma_ms + EWMA_ALPHA * (measured_ms - cell.ewma_ms)
        };
        cell.samples = cell.samples.saturating_add(1);
        let ewma_ms = cell.ewma_ms;
        let table = static_chunk_ms(key.0, key.1);
        let sample_ratio = (measured_ms / table).clamp(RATIO_RANGE.0, RATIO_RANGE.1);
        self.ratio = Some(match self.ratio {
            Some(r) => r + EWMA_ALPHA * (sample_ratio - r),
            None => sample_ratio,
        });
        self.ratio_count = self.ratio_count.saturating_add(1);
        Some(Observation {
            bucket: key.0,
            band: key.1,
            ewma_ms,
            error_ms: measured_ms - predicted_ms,
        })
    }
}

/// EWMA of decode-only tick wall time (the decode step the gap target must
/// leave room for).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DecodeStepClock {
    ewma_ms: Option<f64>,
}

impl DecodeStepClock {
    /// Assumed decode step before the first measurement (K5 MTP decode of
    /// Flash-Next is ~57-68 ms per step, native C4 frames less).
    pub const DEFAULT_MS: f64 = 50.0;

    /// Seed the decode-step EWMA from a persisted cache snapshot, ignoring
    /// missing, non-finite, or non-positive values.
    pub fn from_persisted(ewma_ms: Option<f64>) -> Self {
        Self {
            ewma_ms: ewma_ms.filter(|ms| ms.is_finite() && *ms > 0.0),
        }
    }

    pub fn observe(&mut self, ms: f64) {
        if ms.is_finite() && ms >= 0.0 {
            self.ewma_ms = Some(match self.ewma_ms {
                Some(p) => p + EWMA_ALPHA * (ms - p),
                None => ms,
            });
        }
    }

    pub fn measured_ms(&self) -> Option<f64> {
        self.ewma_ms
    }

    pub fn step_ms(&self) -> f64 {
        self.ewma_ms.unwrap_or(Self::DEFAULT_MS)
    }
}

#[cfg(test)]
#[path = "prefill_slo_cost_tests.rs"]
mod tests;
