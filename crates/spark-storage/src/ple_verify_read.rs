// SPDX-License-Identifier: AGPL-3.0-only

//! Read scheduling for speculative-verify PLE misses (`ATLAS_PLE_VERIFY_PARALLEL_PREAD`).
//!
//! Without io_uring (Docker's default seccomp), a verify frame's 16-80 rows
//! fall below `MIN_PREAD_LANE_SELECTIONS` and the planned reader would issue its
//! 512-byte sector reads one blocking `pread64` at a time on one thread. With
//! parallel verify enabled (the default when `ATLAS_PLE_VERIFY_PARALLEL_PREAD`
//! is unset and misses reach `MIN_AUTO_VERIFY_PARALLEL_MISSES`, or unconditionally
//! below `MIN_PREAD_LANE_SELECTIONS` when set to `1`), those same planned sector
//! reads are spread over up to `MAX_PREAD_DECODE_LANES` scoped lanes, each into
//! its own page buffer, and then consumed in plan order exactly like the serial
//! loop. Only the read schedule changes: same spans, same bytes, same caller
//! order, so verify outputs are unchanged by construction. With io_uring present
//! the selector is inert (the planned reader already submits every read in one
//! batch).
//!
//! Selector values: unset = auto-parallel for `>= 8` misses without io_uring
//! (no per-call receipts); `0` = serial with receipts, for the control arm of
//! an A/B; `1` = parallel for `1..128` misses with receipts. Any other value
//! refuses to open the reader.

use std::os::fd::RawFd;
use std::time::{Duration, Instant};

use anyhow::{Result, bail, ensure};

use super::{
    AlignedPage, MAX_PREAD_DECODE_LANES, MIN_PREAD_LANE_SELECTIONS, PleNvfp4Row, PleOffloadReader,
    pread_exact_aligned,
};

/// The selector name.
pub const VERIFY_PARALLEL_PREAD: &str = "ATLAS_PLE_VERIFY_PARALLEL_PREAD";

/// Minimum uncached verify misses before auto-enabling parallel `pread64`
/// when `io_uring` is unavailable and `ATLAS_PLE_VERIFY_PARALLEL_PREAD` is unset.
pub(super) const MIN_AUTO_VERIFY_PARALLEL_MISSES: usize = 8;

/// Parsed selector state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VerifyReadConfig {
    /// The selector was explicitly set (to `0` or `1`): emit per-call receipts.
    pub(super) receipts: bool,
    /// Parallel verify `pread64` is permitted (`true` for unset auto-mode and
    /// `"1"`; `false` only when explicitly disabled with `"0"`).
    pub(super) parallel: bool,
}

impl Default for VerifyReadConfig {
    fn default() -> Self {
        Self {
            receipts: false,
            parallel: true,
        }
    }
}

impl VerifyReadConfig {
    pub(super) fn from_env() -> Result<Self> {
        parse_verify_read(std::env::var(VERIFY_PARALLEL_PREAD).ok().as_deref())
    }
}

/// `None` = unset (auto-parallel for `>= 8` misses without io_uring, no per-call
/// receipts); `"0"` = serial with receipts; `"1"` = parallel for `1..128` misses
/// with receipts; anything else is refused.
pub(super) fn parse_verify_read(value: Option<&str>) -> Result<VerifyReadConfig> {
    match value {
        None => Ok(VerifyReadConfig::default()),
        Some("0") => Ok(VerifyReadConfig {
            receipts: true,
            parallel: false,
        }),
        Some("1") => Ok(VerifyReadConfig {
            receipts: true,
            parallel: true,
        }),
        Some(other) => bail!(
            "PLE_VERIFY_READ refused: {VERIFY_PARALLEL_PREAD} must be unset, 0 or 1; got {other:?}"
        ),
    }
}

/// True when a verify read of `misses` uncached rows takes the parallel
/// decode route: selector on, no io_uring, more than one pread lane, and
/// below the planned-lanes threshold (at or above it the planned reader
/// already fans out over its lanes).
pub(super) const fn small_parallel_route(
    parallel: bool,
    has_ring: bool,
    pread_lanes: usize,
    misses: usize,
) -> bool {
    parallel && !has_ring && pread_lanes > 1 && misses < MIN_PREAD_LANE_SELECTIONS
}

/// One planned read for [`pread_spans_parallel`].
#[derive(Clone, Copy, Debug)]
pub(super) struct SpanRead {
    pub(super) fd: RawFd,
    pub(super) buffer_offset: usize,
    pub(super) len: usize,
    pub(super) file_offset: u64,
}

fn run_span_chunk(
    lane_pages: &mut [AlignedPage],
    lane_jobs: &[Option<SpanRead>],
    deadline: Instant,
    io_timeout: Duration,
) -> Result<()> {
    for (page, job) in lane_pages.iter_mut().zip(lane_jobs) {
        let Some(job) = job else { continue };
        ensure!(
            Instant::now() < deadline,
            "PLE planned read exceeded its {io_timeout:?} operation deadline"
        );
        pread_exact_aligned(
            job.fd,
            page.as_mut_slice(job.buffer_offset, job.len),
            job.file_offset,
        )?;
    }
    Ok(())
}

/// Read `jobs[i]` (when `Some`) into `pages[i]` at its buffer offset, with the
/// indices split into contiguous chunks over at most `lanes` concurrent lanes
/// (the first chunk runs on the calling thread, the rest on scoped threads).
/// All threads join before return; each owns a disjoint slice of `pages`.
pub(super) fn pread_spans_parallel(
    pages: &mut [AlignedPage],
    jobs: &[Option<SpanRead>],
    lanes: usize,
    deadline: Instant,
    io_timeout: Duration,
) -> Result<()> {
    ensure!(
        pages.len() == jobs.len(),
        "PLE parallel verify read: {} page buffers for {} reads",
        pages.len(),
        jobs.len()
    );
    if jobs.is_empty() {
        return Ok(());
    }
    let per_lane = jobs.len().div_ceil(lanes.max(1));
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let mut page_chunks = pages.chunks_mut(per_lane);
        let mut job_chunks = jobs.chunks(per_lane);
        let first = page_chunks.next().zip(job_chunks.next());
        let handles: Vec<_> = page_chunks
            .zip(job_chunks)
            .map(|(lane_pages, lane_jobs)| {
                scope.spawn(move || run_span_chunk(lane_pages, lane_jobs, deadline, io_timeout))
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len() + usize::from(first.is_some()));
        if let Some((first_pages, first_jobs)) = first {
            out.push(run_span_chunk(
                first_pages,
                first_jobs,
                deadline,
                io_timeout,
            ));
        }
        for handle in handles {
            out.push(handle.join().unwrap_or_else(|_| {
                Err(anyhow::anyhow!("PLE parallel verify read lane panicked"))
            }));
        }
        out
    });
    results.into_iter().collect()
}

/// Which read served a windowed call's misses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyRoute {
    CacheOnly,
    Serial,
    Parallel,
    PlannedLanes,
    Streamed,
    Bounded,
}

impl VerifyRoute {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::CacheOnly => "cache_only",
            Self::Serial => "serial",
            Self::Parallel => "parallel",
            Self::PlannedLanes => "planned_lanes",
            Self::Streamed => "streamed",
            Self::Bounded => "bounded",
        }
    }
}

/// Record-cache accounting for one windowed read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WindowedTally {
    /// Selections served from the record cache.
    pub(super) hits: usize,
    /// Unique uncached `(shard, row)` pairs read from storage.
    pub(super) misses: usize,
    pub(super) route: VerifyRoute,
}

impl PleOffloadReader {
    /// [`Self::read_rows_windowed`] for the speculative-verify callers: same
    /// rows, bytes and order, with the `ATLAS_PLE_VERIFY_PARALLEL_PREAD`
    /// schedule for small misses and a per-call `PLE_VERIFY_MISSES` receipt
    /// while the selector is set.
    pub fn read_rows_verify(&mut self, selections: &[(usize, usize)]) -> Result<Vec<PleNvfp4Row>> {
        let (rows, tally) = self.read_rows_windowed_inner(selections, true)?;
        if self.verify_read.receipts {
            tracing::info!(
                "PLE_VERIFY_MISSES n={} hits={} rows={} route={}",
                tally.misses,
                tally.hits,
                selections.len(),
                tally.route.label()
            );
        }
        Ok(rows)
    }

    /// Threads for one parallel verify read.
    pub(super) fn verify_pread_lanes(&self) -> usize {
        self.pread_pages.len().min(MAX_PREAD_DECODE_LANES)
    }

    pub(super) fn verify_small_parallel(&self, misses: usize) -> bool {
        if !self.verify_read.receipts && misses < MIN_AUTO_VERIFY_PARALLEL_MISSES {
            return false;
        }
        small_parallel_route(
            self.verify_read.parallel,
            self.ring.is_some(),
            self.pread_pages.len(),
            misses,
        )
    }

    /// `(mode, threads, io)` for the startup receipt.
    fn verify_read_mode(&self) -> (&'static str, usize, &'static str) {
        if self.ring.is_some() {
            return ("inert", 0, "io_uring");
        }
        if !self.verify_read.parallel {
            return ("serial", 1, "pread");
        }
        if self.pread_pages.len() <= 1 {
            return ("inert", 1, "pread");
        }
        ("parallel", self.verify_pread_lanes(), "pread")
    }

    pub(super) fn log_verify_read_receipt(&self) {
        if self.verify_read.receipts {
            let (mode, threads, io) = self.verify_read_mode();
            tracing::info!(
                "PLE_VERIFY_READ mode={mode} threads={threads} threshold={MIN_PREAD_LANE_SELECTIONS} io={io} selector={VERIFY_PARALLEL_PREAD}"
            );
            return;
        }
        if self.ring.is_none() && self.pread_pages.len() > 1 {
            tracing::info!(
                "PLE_VERIFY_READ mode=auto_parallel threads={} min_misses={MIN_AUTO_VERIFY_PARALLEL_MISSES} threshold={MIN_PREAD_LANE_SELECTIONS} io=pread selector={VERIFY_PARALLEL_PREAD}",
                self.verify_pread_lanes()
            );
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::super::{
        EXPECTED_FORMAT, EXPECTED_PAGE_BYTES, EXPECTED_RECORD_BYTES, EXPECTED_RECORDS_PER_PAGE,
        PleIoMode,
    };
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    const SHARDS: usize = 2;
    const PAGES_PER_SHARD: usize = 48;
    const SCALE2: [f32; SHARDS] = [0.5, 1.25];

    fn byte_at(shard: usize, offset: usize) -> u8 {
        (shard.wrapping_mul(131) ^ offset.wrapping_mul(2_654_435_761) >> 7) as u8
    }

    /// A two-shard O_DIRECT sidecar in a fresh temp directory.
    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "atlas-ple-verify-{tag}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("create fixture dir");
            let rows = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
            let bytes = PAGES_PER_SHARD * EXPECTED_PAGE_BYTES;
            let mut entries = Vec::new();
            for (shard, &scale2) in SCALE2.iter().enumerate().take(SHARDS) {
                let data: Vec<u8> = (0..bytes).map(|offset| byte_at(shard, offset)).collect();
                let file = format!("shard{shard}.bin");
                std::fs::write(dir.join(&file), data).expect("write shard");
                entries.push(serde_json::json!({
                    "shard": shard, "file": file, "rows": rows, "width": 160,
                    "scale2": scale2, "bytes": bytes,
                }));
            }
            let manifest = serde_json::json!({
                "format": EXPECTED_FORMAT, "page_bytes": EXPECTED_PAGE_BYTES,
                "record_bytes": EXPECTED_RECORD_BYTES,
                "records_per_page": EXPECTED_RECORDS_PER_PAGE,
                "packed_bytes": 80, "scale_bytes": 10, "group_size": 16,
                "entries": entries,
            });
            std::fs::write(dir.join("manifest.json"), manifest.to_string())
                .expect("write manifest");
            Self { dir }
        }

        fn manifest(&self) -> PathBuf {
            self.dir.join("manifest.json")
        }

        /// A no-io_uring reader on the production planned route (sector reads,
        /// no private page cache) with the given verify schedule.
        fn reader(&self, parallel: bool) -> PleOffloadReader {
            let mut reader = PleOffloadReader::open_inner(
                Path::new(&self.manifest()),
                256,
                0,
                PleIoMode::Direct,
                Duration::from_secs(30),
                true,
            )
            .expect("open fixture reader");
            assert!(reader.ring.is_none(), "forced pread must skip io_uring");
            assert!(
                reader.pread_pages.len() > 1,
                "default pread lanes must be > 1"
            );
            reader.read_plan.fast_plan = true;
            reader.read_plan.sector_ready = true;
            reader.verify_read = VerifyReadConfig {
                receipts: true,
                parallel,
            };
            reader
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Ground truth straight from the generator: record slot of the row's page.
    fn expected(shard: usize, row: usize) -> PleNvfp4Row {
        let page = row / EXPECTED_RECORDS_PER_PAGE;
        let slot = row % EXPECTED_RECORDS_PER_PAGE;
        let base = page * EXPECTED_PAGE_BYTES + slot * EXPECTED_RECORD_BYTES;
        let mut record = [0u8; EXPECTED_RECORD_BYTES];
        for (index, byte) in record.iter_mut().enumerate() {
            *byte = byte_at(shard, base + index);
        }
        PleNvfp4Row {
            record,
            scale2: SCALE2[shard],
        }
    }

    /// Deterministic picks with forced duplicates (every 5th repeats an earlier pick).
    fn picks(seed: u64, count: usize) -> Vec<(usize, usize)> {
        let rows = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(count);
        for index in 0..count {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if index % 5 == 4 {
                out.push(out[(state as usize) % index]);
            } else {
                out.push(((state as usize) % SHARDS, ((state >> 20) as usize) % rows));
            }
        }
        out
    }

    fn assert_rows(got: &[PleNvfp4Row], want: &[(usize, usize)]) {
        assert_eq!(got.len(), want.len());
        for (index, (row, &(shard, id))) in got.iter().zip(want).enumerate() {
            let truth = expected(shard, id);
            assert_eq!(row.record, truth.record, "row {index} ({shard},{id}) bytes");
            assert_eq!(
                row.scale2.to_bits(),
                truth.scale2.to_bits(),
                "row {index} scale2"
            );
        }
    }

    #[test]
    fn selector_parses_and_refuses_unknown_values() {
        assert_eq!(
            parse_verify_read(None).unwrap(),
            VerifyReadConfig::default()
        );
        assert_eq!(
            parse_verify_read(Some("0")).unwrap(),
            VerifyReadConfig {
                receipts: true,
                parallel: false
            }
        );
        assert_eq!(
            parse_verify_read(Some("1")).unwrap(),
            VerifyReadConfig {
                receipts: true,
                parallel: true
            }
        );
        for bad in ["", "2", "true", " 1", "1 ", "yes", "-1"] {
            let error = parse_verify_read(Some(bad)).expect_err(bad).to_string();
            assert!(
                error.contains("PLE_VERIFY_READ refused"),
                "{bad:?}: {error}"
            );
            assert!(error.contains(VERIFY_PARALLEL_PREAD), "{bad:?}: {error}");
        }
    }

    #[test]
    fn route_threshold_boundary_is_strictly_below_128() {
        assert_eq!(MIN_PREAD_LANE_SELECTIONS, 128);
        assert!(small_parallel_route(true, false, 32, 1));
        assert!(small_parallel_route(true, false, 32, 127));
        assert!(!small_parallel_route(true, false, 32, 128));
        assert!(!small_parallel_route(true, false, 32, 129));
        assert!(!small_parallel_route(false, false, 32, 32), "selector off");
        assert!(
            !small_parallel_route(true, true, 32, 32),
            "io_uring present"
        );
        assert!(
            !small_parallel_route(true, false, 1, 32),
            "single pread lane"
        );
    }

    #[test]
    fn serial_and_parallel_return_identical_rows() {
        let fixture = Fixture::new("equal");
        for (seed, count) in [
            (1u64, 16usize),
            (2, 32),
            (3, 80),
            (4, 127),
            (5, 128),
            (6, 129),
            (7, 300),
        ] {
            let selections = picks(seed, count);
            // Fresh readers per set: an empty record cache forces every route to read.
            let serial = fixture
                .reader(false)
                .read_rows_verify(&selections)
                .expect("serial");
            let parallel = fixture
                .reader(true)
                .read_rows_verify(&selections)
                .expect("parallel");
            assert_rows(&serial, &selections);
            assert_rows(&parallel, &selections);
        }
    }

    #[test]
    fn routes_follow_the_threshold_on_the_planned_reader() {
        let fixture = Fixture::new("route");
        // Unique picks only, so `misses` equals the selection count.
        let unique = |count: usize| -> Vec<(usize, usize)> {
            (0..count).map(|row| (row % SHARDS, row)).collect()
        };
        for (count, parallel, want) in [
            (127, true, VerifyRoute::Parallel),
            (128, true, VerifyRoute::PlannedLanes),
            (129, true, VerifyRoute::PlannedLanes),
            (127, false, VerifyRoute::Serial),
            (128, false, VerifyRoute::PlannedLanes),
        ] {
            let selections = unique(count);
            let mut reader = fixture.reader(parallel);
            let (rows, tally) = reader
                .read_rows_windowed_inner(&selections, true)
                .expect("read");
            assert_eq!(tally.route, want, "count={count} parallel={parallel}");
            assert_eq!(tally.misses, count);
            assert_eq!(tally.hits, 0);
            assert_rows(&rows, &selections);
            // The same rows again are all record-cache hits.
            let (again, tally) = reader
                .read_rows_windowed_inner(&selections, true)
                .expect("reread");
            assert_eq!(tally.route, VerifyRoute::CacheOnly);
            assert_eq!((tally.hits, tally.misses), (count, 0));
            assert_rows(&again, &selections);
        }
        // The non-verify entry point never takes the parallel verify route.
        let selections = unique(64);
        let (_, tally) = fixture
            .reader(true)
            .read_rows_windowed_inner(&selections, false)
            .expect("windowed read");
        assert_eq!(tally.route, VerifyRoute::Serial);
    }

    #[test]
    fn duplicates_count_once_as_misses() {
        let fixture = Fixture::new("dups");
        let selections = vec![(0, 5), (1, 7), (0, 5), (0, 5), (1, 7), (1, 900)];
        let (rows, tally) = fixture
            .reader(true)
            .read_rows_windowed_inner(&selections, true)
            .expect("read");
        assert_eq!((tally.misses, tally.hits), (3, 0));
        assert_rows(&rows, &selections);
    }

    #[test]
    fn out_of_range_ids_fail_on_both_routes() {
        let fixture = Fixture::new("range");
        let rows = PAGES_PER_SHARD * EXPECTED_RECORDS_PER_PAGE;
        for bad in [
            vec![(0, 3), (0, rows)],
            vec![(SHARDS, 0)],
            vec![(1, rows + 91)],
        ] {
            for parallel in [false, true] {
                let error = fixture
                    .reader(parallel)
                    .read_rows_verify(&bad)
                    .expect_err("out-of-range selection must fail")
                    .to_string();
                assert!(
                    error.contains("out of range"),
                    "{bad:?} parallel={parallel}: {error}"
                );
            }
        }
    }

    #[test]
    fn startup_receipt_modes() {
        let fixture = Fixture::new("receipt");
        let parallel = fixture.reader(true);
        assert_eq!(
            parallel.verify_read_mode(),
            (
                "parallel",
                parallel.pread_pages.len().min(MAX_PREAD_DECODE_LANES),
                "pread"
            )
        );
        assert_eq!(parallel.verify_pread_lanes(), 16);
        assert_eq!(
            fixture.reader(false).verify_read_mode(),
            ("serial", 1, "pread")
        );
        let mut single = fixture.reader(true);
        single.pread_pages.truncate(1);
        assert_eq!(single.verify_read_mode(), ("inert", 1, "pread"));
        assert!(!single.verify_small_parallel(32));
    }

    /// The speculative-verify PLE callers use the verify entry point.
    #[test]
    fn verify_callers_consult_the_selector_route() {
        let source = include_str!("../../spark-model/src/layers/qwen4_ple/runtime/verify_batch.rs");
        let verify_fn = source
            .split("pub fn forward_tokens_verify(")
            .nth(1)
            .expect("forward_tokens_verify present");
        let verify_fn = &verify_fn[..verify_fn.find("\n    pub fn ").unwrap_or(verify_fn.len())];
        assert!(
            verify_fn.contains(".read_rows_verify(&request)"),
            "single-seq verify"
        );
        assert!(
            !verify_fn.contains("read_rows_windowed"),
            "single-seq verify bypass"
        );
        let cross = source
            .split("pub fn forward_cross_seq_verify(")
            .nth(1)
            .expect("forward_cross_seq_verify present");
        let cross = &cross[..cross.find("\n    pub fn ").unwrap_or(cross.len())];
        assert!(
            cross.contains(".read_rows_verify(&request)"),
            "cross-seq verify"
        );
        assert!(
            !cross.contains("read_rows_windowed"),
            "cross-seq verify bypass"
        );
        let reader = include_str!("ple_offload.rs");
        let compact_reader: String = reader.split_whitespace().collect();
        assert!(
            compact_reader.contains(
                ".read_rows_streamed_planned(&miss_selections,None,None,true,mode,lanes)?"
            ) || compact_reader.contains(
                ".read_rows_streamed_planned(&miss_selections,None,None,true,mode,lanes,)?"
            )
        );
        assert!(reader.contains(
            "let lanes = if verify && self.verify_small_parallel(miss_selections.len()) {"
        ));
        // The prefill caller of the planned reader stays serial.
        assert!(reader.contains("                mode,\n                0,\n            );"));
    }

    /// The parallel lanes live only inside the `ring.is_none()` branch of the
    /// planned reader; the io_uring submission loop is unchanged.
    #[test]
    fn io_uring_path_is_untouched() {
        let reader = include_str!("ple_offload.rs");
        let planned = reader
            .split("fn read_rows_streamed_planned(")
            .nth(1)
            .expect("planned reader present");
        let no_ring = planned
            .find("if self.ring.is_none() {")
            .expect("pread branch");
        let lanes = planned
            .find("verify_read::pread_spans_parallel(")
            .expect("parallel call");
        let ring_loop = planned
            .find("while done < reads {")
            .expect("io_uring submission loop");
        assert!(
            no_ring < lanes && lanes < ring_loop,
            "parallel reads outside the pread branch"
        );
        assert_eq!(planned.matches("pread_spans_parallel(").count(), 1);
        let ring_body = &planned[ring_loop..];
        let ring_body = &ring_body[..ring_body.find("\n    fn ").unwrap_or(ring_body.len())];
        assert!(
            !ring_body.contains("pread_lanes"),
            "io_uring loop consults the lanes"
        );
        assert!(
            !ring_body.contains("verify_read"),
            "io_uring loop consults the selector"
        );
        // And the route itself refuses to engage when a ring exists.
        assert!(!small_parallel_route(true, true, 256, 32));
    }

    #[test]
    fn parallel_span_reader_checks_extent_and_empty() {
        let soon = || Instant::now() + Duration::from_secs(1);
        let mut none: Vec<AlignedPage> = Vec::new();
        pread_spans_parallel(&mut none, &[], 16, soon(), Duration::from_secs(1))
            .expect("empty is a no-op");
        let mut one = vec![AlignedPage::new()];
        assert!(
            pread_spans_parallel(&mut one, &[None, None], 16, soon(), Duration::from_secs(1))
                .is_err()
        );
    }

    #[test]
    fn auto_parallel_engages_when_unset_without_io_uring_and_honors_explicit_zero() {
        let auto = parse_verify_read(None).expect("unset selector");
        assert_eq!(
            auto,
            VerifyReadConfig {
                receipts: false,
                parallel: true,
            }
        );

        let fixture = Fixture::new("auto-parallel");
        let unique = |count: usize| -> Vec<(usize, usize)> {
            (0..count).map(|row| (row % SHARDS, row)).collect()
        };

        // Unset env (`VerifyReadConfig::default()`): serial below 8 misses,
        // parallel for 8..128 misses, planned lanes at 128+.
        let mut reader = fixture.reader(true);
        reader.verify_read = VerifyReadConfig::default();
        assert!(!reader.verify_small_parallel(1));
        assert!(!reader.verify_small_parallel(MIN_AUTO_VERIFY_PARALLEL_MISSES - 1));
        assert!(reader.verify_small_parallel(MIN_AUTO_VERIFY_PARALLEL_MISSES));
        assert!(reader.verify_small_parallel(16));
        assert!(reader.verify_small_parallel(127));
        assert!(!reader.verify_small_parallel(128));

        let small = unique(MIN_AUTO_VERIFY_PARALLEL_MISSES - 1);
        let (rows, tally) = reader
            .read_rows_windowed_inner(&small, true)
            .expect("small auto serial read");
        assert_eq!(tally.route, VerifyRoute::Serial);
        assert_rows(&rows, &small);

        let mut reader = fixture.reader(true);
        reader.verify_read = VerifyReadConfig::default();
        let medium = unique(16);
        let (rows, tally) = reader
            .read_rows_windowed_inner(&medium, true)
            .expect("medium auto parallel read");
        assert_eq!(tally.route, VerifyRoute::Parallel);
        assert_eq!((tally.hits, tally.misses), (0, 16));
        assert_rows(&rows, &medium);

        // Explicit "0" stays strictly serial for < 128 misses.
        let mut serial_reader = fixture.reader(false);
        assert!(!serial_reader.verify_small_parallel(16));
        let (rows, tally) = serial_reader
            .read_rows_windowed_inner(&medium, true)
            .expect("explicit zero serial read");
        assert_eq!(tally.route, VerifyRoute::Serial);
        assert_rows(&rows, &medium);
    }

    #[test]
    fn parked_cache_serves_verify_hits_even_when_record_cache_is_disabled() {
        use super::super::{ParkedPleCache, RecordCache};
        use std::sync::Arc;

        let fixture = Fixture::new("parked-verify");
        let selections = picks(42, 64);

        // Simulate ATLAS_PLE_CACHE_MB=0 (RecordCache disabled) with a hermetic
        // fixed-size ParkedPleCache (4096 slots, immune to process env overrides).
        let mut reader = fixture.reader(true);
        reader.record_cache = RecordCache::new(0, 0);
        let shard_rows = reader.shard_rows();
        reader.parked_cache = Arc::new(ParkedPleCache::with_slots(
            4_096,
            shard_rows.clone(),
            SCALE2.to_vec(),
        ));
        assert!(reader.parked_cache().is_enabled());

        // Cold read misses and populates ParkedPleCache.
        let (cold_rows, cold_tally) = reader
            .read_rows_windowed_inner(&selections, true)
            .expect("cold verify read");
        assert_eq!(cold_tally.hits, 0);
        assert!(cold_tally.misses > 0);
        assert_rows(&cold_rows, &selections);

        // Reread hits 100% in ParkedPleCache with 0 storage I/O even though RecordCache=0.
        let (warm_rows, warm_tally) = reader
            .read_rows_windowed_inner(&selections, true)
            .expect("warm parked verify read");
        assert_eq!(warm_tally.route, VerifyRoute::CacheOnly);
        assert_eq!((warm_tally.hits, warm_tally.misses), (selections.len(), 0));
        assert_rows(&warm_rows, &selections);

        // Lock-free direct handle serves exact identical rows & staging bytes without reader lock.
        let parked = reader.parked_cache();
        let direct = parked
            .try_read_all(&selections)
            .expect("direct read")
            .expect("all rows cached");
        assert_rows(&direct, &selections);

        let mut rec_stage = vec![0u8; selections.len() * EXPECTED_RECORD_BYTES];
        let mut scale_stage = vec![0u8; selections.len() * 4];
        assert!(
            parked
                .try_fill_staging(&selections, &mut rec_stage, &mut scale_stage)
                .expect("direct staging fill")
        );
        for (i, row) in warm_rows.iter().enumerate() {
            assert_eq!(
                &rec_stage[i * EXPECTED_RECORD_BYTES..(i + 1) * EXPECTED_RECORD_BYTES],
                &row.record
            );
            assert_eq!(&scale_stage[i * 4..(i + 1) * 4], &row.scale2.to_le_bytes());
        }

        // Disabling ParkedPleCache alongside RecordCache forces storage reads again.
        reader.parked_cache = Arc::new(ParkedPleCache::with_slots(0, shard_rows, SCALE2.to_vec()));
        assert!(!reader.parked_cache().is_enabled());
        let (uncached_rows, uncached_tally) = reader
            .read_rows_windowed_inner(&selections, true)
            .expect("uncached verify read");
        assert_eq!(uncached_tally.hits, 0);
        assert!(uncached_tally.misses > 0);
        assert_rows(&uncached_rows, &selections);
    }
}
