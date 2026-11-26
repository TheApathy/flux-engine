// SPDX-License-Identifier: AGPL-3.0-only

//! Read planning for sparse PLE prefill reads (K9 item W, default off).
//!
//! Dependency-free (std only) so it is unit-tested with `rustc --test` and
//! included by path into the I/O micro harness.
//!
//! A prefill chunk selects 16 rows per token. The streamed reader turns the
//! selections into unique reads in first-use order and remembers, per read,
//! which selections it serves (the waiters). Two levers live here:
//!
//! - [`plan_fast`] builds exactly the plan of the reader's original inline
//!   loop ([`plan_reference`]): same unique keys in the same first-use order,
//!   same waiters in the same order. It replaces the `HashMap` plus one `Vec`
//!   per key with an open-addressing table and flat waiter arrays, so the first
//!   read is submitted sooner (`ATLAS_PLE_FAST_PLAN=1`).
//! - [`ReadGranularity::Sector`] reads the 512-byte-aligned span that holds a
//!   90-byte record (one or two sectors) instead of its 4 KiB half page
//!   (`ATLAS_PLE_SECTOR_READ=1`). The record bytes are the same file bytes.

/// Bytes per sidecar page (`qwen38-flash-next-ple-nvfp4-direct-v1`).
pub const PAGE_BYTES: usize = 8192;
/// Bytes per packed NVFP4 record.
pub const RECORD_BYTES: usize = 90;
/// Records per page (91 × 90 = 8190: no record crosses a page).
pub const RECORDS_PER_PAGE: usize = 91;
/// O_DIRECT alignment unit of a sector read (the device's logical block size).
pub const SECTOR_BYTES: usize = 512;

/// How much of the sidecar one read fetches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadGranularity {
    /// The whole 8 KiB page (the reader's default).
    Page,
    /// The 4 KiB half that holds the record, or the whole page for the one
    /// slot in 91 that straddles the halves (`ATLAS_PLE_HALF_PAGE=1`).
    HalfPage,
    /// The one or two 512-byte sectors that hold the record.
    Sector,
}

impl ReadGranularity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Page => "page",
            Self::HalfPage => "half_page",
            Self::Sector => "sector",
        }
    }
}

/// Where one read lands: file byte range `[file_offset, file_offset + len)`
/// into the 8 KiB buffer at `buffer_offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadSpan {
    pub file_offset: u64,
    pub len: usize,
    pub buffer_offset: usize,
}

/// Read key of `row` within its shard. Page and half-page keys are exactly
/// the reader's original keys (`page`, `page * 4 + part`).
pub fn read_key(granularity: ReadGranularity, row: usize) -> usize {
    let page = row / RECORDS_PER_PAGE;
    let slot = row % RECORDS_PER_PAGE;
    match granularity {
        ReadGranularity::Page => page,
        ReadGranularity::HalfPage => {
            // part 0/1 = first/second 4 KiB half, part 3 = whole page (straddles).
            let first = slot * RECORD_BYTES;
            let last = first + RECORD_BYTES - 1;
            let half = PAGE_BYTES / 2;
            let part = if last < half {
                0
            } else if first >= half {
                1
            } else {
                3
            };
            page * 4 + part
        }
        ReadGranularity::Sector => {
            let offset = page * PAGE_BYTES + slot * RECORD_BYTES;
            let first = offset / SECTOR_BYTES;
            let last = (offset + RECORD_BYTES - 1) / SECTOR_BYTES;
            first * 2 + (last - first)
        }
    }
}

/// The file span a key reads and where it lands in the buffer.
pub fn read_span(granularity: ReadGranularity, key: usize) -> ReadSpan {
    match granularity {
        ReadGranularity::Page => ReadSpan {
            file_offset: (key * PAGE_BYTES) as u64,
            len: PAGE_BYTES,
            buffer_offset: 0,
        },
        ReadGranularity::HalfPage => {
            let page = key / 4;
            let half = PAGE_BYTES / 2;
            let (skip, len) = match key % 4 {
                0 => (0, half),
                1 => (half, half),
                _ => (0, PAGE_BYTES),
            };
            ReadSpan {
                file_offset: (page * PAGE_BYTES + skip) as u64,
                len,
                buffer_offset: skip,
            }
        }
        ReadGranularity::Sector => {
            let first = key / 2;
            let sectors = key % 2 + 1;
            ReadSpan {
                file_offset: (first * SECTOR_BYTES) as u64,
                len: sectors * SECTOR_BYTES,
                buffer_offset: 0,
            }
        }
    }
}

/// Offset of `row`'s record inside the 8 KiB buffer after its key's read.
pub fn record_offset(granularity: ReadGranularity, row: usize) -> usize {
    let slot = row % RECORDS_PER_PAGE;
    match granularity {
        ReadGranularity::Page | ReadGranularity::HalfPage => slot * RECORD_BYTES,
        ReadGranularity::Sector => {
            let offset = (row / RECORDS_PER_PAGE) * PAGE_BYTES + slot * RECORD_BYTES;
            offset % SECTOR_BYTES
        }
    }
}

/// One selection served by a read: where its record sits in the buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanWaiter {
    pub request_index: u32,
    pub record_offset: u16,
    pub scale2: f32,
}

/// Unique reads in first-use order and the selections each one serves.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadPlan {
    pub granularity: ReadGranularity,
    /// `(shard, key)` per unique read, in first-use order.
    pub keys: Vec<(usize, usize)>,
    /// Waiters of read `i` are `waiters[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<u32>,
    /// In request order within each read.
    pub waiters: Vec<PlanWaiter>,
}

impl ReadPlan {
    pub fn reads(&self) -> usize {
        self.keys.len()
    }

    pub fn waiters_of(&self, read: usize) -> &[PlanWaiter] {
        &self.waiters[self.offsets[read] as usize..self.offsets[read + 1] as usize]
    }

    pub fn span(&self, read: usize) -> ReadSpan {
        read_span(self.granularity, self.keys[read].1)
    }

    /// Bytes the plan reads from storage.
    pub fn bytes(&self) -> u64 {
        (0..self.reads())
            .map(|read| self.span(read).len as u64)
            .sum()
    }
}

fn validate(
    shard_rows: &[usize],
    shard_scale2: &[f32],
    shard: usize,
    row: usize,
) -> Result<f32, String> {
    if shard_rows.len() != shard_scale2.len() {
        return Err("PLE shard metadata length mismatch".to_string());
    }
    let rows = *shard_rows
        .get(shard)
        .ok_or_else(|| format!("PLE shard {shard} out of range"))?;
    if row >= rows {
        return Err(format!("PLE row {row} out of range for shard {shard}"));
    }
    Ok(shard_scale2[shard])
}

/// The reader's original planning loop (one `HashMap` lookup and one waiter
/// `Vec` per unique key). Kept as the definition of the plan.
pub fn plan_reference(
    selections: &[(usize, usize)],
    shard_rows: &[usize],
    shard_scale2: &[f32],
    granularity: ReadGranularity,
) -> Result<ReadPlan, String> {
    use std::collections::HashMap;
    if selections.len() > u32::MAX as usize {
        return Err("PLE selection count exceeds u32".to_string());
    }
    let mut keys = Vec::<(usize, usize)>::new();
    let mut lookup = HashMap::<(usize, usize), usize>::new();
    let mut per_key = Vec::<Vec<PlanWaiter>>::new();
    for (request_index, &(shard, row)) in selections.iter().enumerate() {
        let scale2 = validate(shard_rows, shard_scale2, shard, row)?;
        let key = (shard, read_key(granularity, row));
        let index = *lookup.entry(key).or_insert_with(|| {
            keys.push(key);
            per_key.push(Vec::new());
            keys.len() - 1
        });
        per_key[index].push(PlanWaiter {
            request_index: request_index as u32,
            record_offset: record_offset(granularity, row) as u16,
            scale2,
        });
    }
    let mut offsets = Vec::with_capacity(keys.len() + 1);
    let mut waiters = Vec::with_capacity(selections.len());
    offsets.push(0u32);
    for list in per_key {
        waiters.extend(list);
        offsets.push(waiters.len() as u32);
    }
    Ok(ReadPlan {
        granularity,
        keys,
        offsets,
        waiters,
    })
}

const SHARD_SHIFT: u32 = 40;

struct PlanFastScratch {
    table: Vec<(u64, u32)>,
    touched: Vec<u32>,
}

/// [`plan_reference`]'s plan, built with an open-addressing table and flat
/// waiter arrays (two passes over the selections, no per-key allocation).
/// Reuses the open-addressing table via `thread_local!` and resets only
/// touched slots so per-chunk calls avoid multi-megabyte mmap/page-fault/munmap.
pub fn plan_fast(
    selections: &[(usize, usize)],
    shard_rows: &[usize],
    shard_scale2: &[f32],
    granularity: ReadGranularity,
) -> Result<ReadPlan, String> {
    let n = selections.len();
    if n > u32::MAX as usize {
        return Err("PLE selection count exceeds u32".to_string());
    }
    let capacity = (2 * n).next_power_of_two().max(64);
    let bits = capacity.trailing_zeros();
    let mask = capacity - 1;
    let mut keys = Vec::<(usize, usize)>::with_capacity(n / 2 + 1);
    let mut counts = Vec::<u32>::with_capacity(n / 2 + 1);
    let mut selection_read = Vec::<u32>::with_capacity(n);
    let mut selection_meta = Vec::<(u16, f32)>::with_capacity(n);

    thread_local! {
        static SCRATCH: std::cell::RefCell<PlanFastScratch> = const {
            std::cell::RefCell::new(PlanFastScratch {
                table: Vec::new(),
                touched: Vec::new(),
            })
        };
    }

    SCRATCH.with(|cell| -> Result<(), String> {
        let s = &mut *cell.borrow_mut();
        for &slot in &s.touched {
            s.table[slot as usize].0 = u64::MAX;
        }
        s.touched.clear();
        if s.table.len() < capacity {
            s.table.resize(capacity, (u64::MAX, 0));
        }
        for &(shard, row) in selections {
            let scale2 = validate(shard_rows, shard_scale2, shard, row)?;
            let key = read_key(granularity, row);
            if shard >= 1 << (64 - SHARD_SHIFT - 1) || key >= 1 << SHARD_SHIFT {
                return Err(format!(
                    "PLE read key out of range: shard {shard} key {key}"
                ));
            }
            let packed = ((shard as u64) << SHARD_SHIFT) | (key as u64);
            let mut slot = (packed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize;
            let index = loop {
                let (held, idx) = s.table[slot];
                if held == packed {
                    break idx;
                }
                if held == u64::MAX {
                    let index = keys.len() as u32;
                    s.table[slot] = (packed, index);
                    s.touched.push(slot as u32);
                    keys.push((shard, key));
                    counts.push(0);
                    break index;
                }
                slot = (slot + 1) & mask;
            };
            counts[index as usize] += 1;
            selection_read.push(index);
            selection_meta.push((record_offset(granularity, row) as u16, scale2));
        }
        for &slot in &s.touched {
            s.table[slot as usize].0 = u64::MAX;
        }
        s.touched.clear();
        Ok(())
    })?;

    let mut offsets = Vec::with_capacity(keys.len() + 1);
    let mut total = 0u32;
    offsets.push(0u32);
    for &count in &counts {
        total += count;
        offsets.push(total);
    }
    let mut cursor: Vec<u32> = offsets[..keys.len()].to_vec();
    let mut waiters = vec![
        PlanWaiter {
            request_index: 0,
            record_offset: 0,
            scale2: 0.0,
        };
        n
    ];
    for (request_index, (&read, &(record_offset, scale2))) in
        selection_read.iter().zip(&selection_meta).enumerate()
    {
        let at = &mut cursor[read as usize];
        waiters[*at as usize] = PlanWaiter {
            request_index: request_index as u32,
            record_offset,
            scale2,
        };
        *at += 1;
    }
    Ok(ReadPlan {
        granularity,
        keys,
        offsets,
        waiters,
    })
}

/// Selector value of `ATLAS_PLE_FAST_PLAN` / `ATLAS_PLE_SECTOR_READ` /
/// `ATLAS_PLE_READ_STATS`: absent, empty or `0` = off, `1` = on.
pub fn parse_switch(name: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{name} must be absent, 0 or 1; got {other:?}")),
    }
}

/// What a streamed prefill read does, chosen per read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadMode {
    /// The reader's original loop.
    Legacy,
    /// [`plan_fast`] at the reader's own granularity (same reads).
    Planned,
    /// [`plan_fast`] with sector reads.
    Sector,
}

impl ReadMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Planned => "planned",
            Self::Sector => "sector",
        }
    }
}

/// The mode a read uses. Without a trigger file (`ATLAS_PLE_READ_TRIGGER`
/// unset) the load-time selectors decide. With one, the selectors apply only
/// while the file exists, and its content may narrow them: `plan` = planned
/// reads at the reader's granularity, `legacy` = off, anything else (empty
/// or `on`) = the load-time mode. `sector_ready` is false when
/// `ATLAS_PLE_SECTOR_READ` is off or its load-time probe refused.
pub fn resolve_mode(
    fast_plan: bool,
    sector_ready: bool,
    trigger: Option<Option<&str>>,
) -> ReadMode {
    let configured = if sector_ready {
        ReadMode::Sector
    } else if fast_plan {
        ReadMode::Planned
    } else {
        ReadMode::Legacy
    };
    match trigger {
        None => configured,
        Some(None) => ReadMode::Legacy,
        Some(Some(content)) => match content.split_whitespace().next().unwrap_or("") {
            "legacy" => ReadMode::Legacy,
            "plan" if configured != ReadMode::Legacy => ReadMode::Planned,
            _ => configured,
        },
    }
}

/// `ATLAS_PLE_READ_RINGS`: absent, empty or `1` = one ring (off), `2` or `4` =
/// that many io_uring rings, each drained by its own thread (planned and
/// sector reads only). Exact values, as `ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH`.
pub fn parse_rings(name: &str, value: Option<&str>) -> Result<usize, String> {
    match value {
        None | Some("") | Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some("4") => Ok(4),
        Some(other) => Err(format!("{name} must be absent, 1, 2 or 4; got {other:?}")),
    }
}

/// Rings one read uses: the configured count, or one when the trigger file
/// (see [`resolve_mode`]) asks for a single-ring attribution phase with the
/// word `single` (for example `plan single`). Legacy reads always use one.
pub fn resolve_rings(configured: usize, mode: ReadMode, trigger: Option<Option<&str>>) -> usize {
    if mode == ReadMode::Legacy {
        return 1;
    }
    match trigger {
        Some(Some(content)) if content.split_whitespace().any(|word| word == "single") => 1,
        _ => configured.max(1),
    }
}

/// The ring that drains read `read` when `rings` rings share a plan: reads are
/// dealt round-robin, so every ring works through the chunk in first-use
/// (token) order and early blocks complete first.
pub fn ring_of(read: usize, rings: usize) -> usize {
    read % rings.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHARD_ROWS: usize = 2_500_012;

    fn meta(shards: usize) -> (Vec<usize>, Vec<f32>) {
        (
            vec![SHARD_ROWS; shards],
            (0..shards).map(|s| 1.0e-5 + s as f32 * 1.0e-7).collect(),
        )
    }

    /// Deterministic xorshift selections with heavy reuse (a hot set) and
    /// rows near page, half-page and sector edges.
    fn selections(n: usize, shards: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let hot: Vec<(usize, usize)> = (0..64)
            .map(|_| ((next() as usize) % shards, (next() as usize) % SHARD_ROWS))
            .collect();
        (0..n)
            .map(|i| match next() % 4 {
                0 => hot[(next() as usize) % hot.len()],
                1 => {
                    let page = (next() as usize) % (SHARD_ROWS / RECORDS_PER_PAGE);
                    let slot = [0, 44, 45, 46, 90, 5, 6, 11, 39][i % 9];
                    ((next() as usize) % shards, page * RECORDS_PER_PAGE + slot)
                }
                _ => ((next() as usize) % shards, (next() as usize) % SHARD_ROWS),
            })
            .collect()
    }

    #[test]
    fn page_and_half_page_keys_match_the_reader_original() {
        for row in 0..(3 * RECORDS_PER_PAGE) {
            let page = row / RECORDS_PER_PAGE;
            let slot = row % RECORDS_PER_PAGE;
            assert_eq!(read_key(ReadGranularity::Page, row), page);
            // Original reader: part from first/last record byte vs the 4 KiB half.
            let first = slot * 90;
            let last = first + 89;
            let part = if last < 4096 {
                0
            } else if first >= 4096 {
                1
            } else {
                3
            };
            assert_eq!(read_key(ReadGranularity::HalfPage, row), page * 4 + part);
            let span = read_span(ReadGranularity::HalfPage, page * 4 + part);
            let (skip, len) = match part {
                0 => (0, 4096),
                1 => (4096, 4096),
                _ => (0, 8192),
            };
            assert_eq!(
                span,
                ReadSpan {
                    file_offset: (page * 8192 + skip) as u64,
                    len,
                    buffer_offset: skip
                }
            );
            assert_eq!(record_offset(ReadGranularity::HalfPage, row), slot * 90);
        }
        // Exactly one slot in 91 straddles the 4 KiB halves.
        let straddlers = (0..RECORDS_PER_PAGE)
            .filter(|&row| read_key(ReadGranularity::HalfPage, row) % 4 == 3)
            .count();
        assert_eq!(straddlers, 1);
    }

    #[test]
    fn every_read_span_covers_its_record_inside_one_page_and_the_buffer() {
        for granularity in [
            ReadGranularity::Page,
            ReadGranularity::HalfPage,
            ReadGranularity::Sector,
        ] {
            for page in [0usize, 1, 7, SHARD_ROWS / RECORDS_PER_PAGE] {
                for slot in 0..RECORDS_PER_PAGE {
                    let row = page * RECORDS_PER_PAGE + slot;
                    let record = (page * PAGE_BYTES + slot * RECORD_BYTES) as u64;
                    let span = read_span(granularity, read_key(granularity, row));
                    let at = record_offset(granularity, row);
                    // The record's file bytes land at buffer[at..at + 90].
                    assert!(span.file_offset <= record, "{granularity:?} {row}");
                    assert!(
                        record + RECORD_BYTES as u64 <= span.file_offset + span.len as u64,
                        "{granularity:?} {row}"
                    );
                    assert_eq!(
                        (record - span.file_offset) as usize + span.buffer_offset,
                        at,
                        "{granularity:?} {row}"
                    );
                    assert!(span.buffer_offset + span.len <= PAGE_BYTES);
                    assert!(at + RECORD_BYTES <= PAGE_BYTES);
                    // O_DIRECT: offset, length and buffer offset sector aligned.
                    assert_eq!(span.file_offset % SECTOR_BYTES as u64, 0);
                    assert_eq!(span.len % SECTOR_BYTES, 0);
                    assert_eq!(span.buffer_offset % SECTOR_BYTES, 0);
                    // A read never leaves the record's page.
                    let page_start = (page * PAGE_BYTES) as u64;
                    assert!(span.file_offset >= page_start);
                    assert!(span.file_offset + span.len as u64 <= page_start + PAGE_BYTES as u64);
                }
            }
        }
    }

    #[test]
    fn sector_reads_are_one_or_two_sectors_and_crossers_are_counted() {
        let mut two = 0;
        for row in 0..(512 * RECORDS_PER_PAGE) {
            let span = read_span(
                ReadGranularity::Sector,
                read_key(ReadGranularity::Sector, row),
            );
            assert!(span.len == 512 || span.len == 1024);
            two += usize::from(span.len == 1024);
        }
        // 89 of every 512 byte offsets put a 90-byte record across a sector edge.
        let rate = two as f64 / (512 * RECORDS_PER_PAGE) as f64;
        assert!((0.15..0.20).contains(&rate), "{rate}");
    }

    #[test]
    fn fast_plan_equals_the_reference_plan() {
        let (rows, scales) = meta(128);
        for granularity in [
            ReadGranularity::Page,
            ReadGranularity::HalfPage,
            ReadGranularity::Sector,
        ] {
            for (n, seed) in [
                (1usize, 1u64),
                (16, 2),
                (1000, 3),
                (131_072, 4),
                (262_144, 5),
            ] {
                let sel = selections(n, 128, seed);
                let reference = plan_reference(&sel, &rows, &scales, granularity).unwrap();
                let fast = plan_fast(&sel, &rows, &scales, granularity).unwrap();
                assert_eq!(fast, reference, "{granularity:?} n={n}");
                assert_eq!(fast.waiters.len(), n);
                assert_eq!(*fast.offsets.last().unwrap() as usize, n);
            }
        }
    }

    #[test]
    fn plan_serves_every_selection_once_with_its_own_record() {
        let (rows, scales) = meta(4);
        let sel = selections(20_000, 4, 9);
        for granularity in [ReadGranularity::HalfPage, ReadGranularity::Sector] {
            let plan = plan_fast(&sel, &rows, &scales, granularity).unwrap();
            let mut seen = vec![false; sel.len()];
            for read in 0..plan.reads() {
                let (shard, key) = plan.keys[read];
                let span = plan.span(read);
                for waiter in plan.waiters_of(read) {
                    let (s, row) = sel[waiter.request_index as usize];
                    assert_eq!(s, shard);
                    assert_eq!(read_key(granularity, row), key);
                    assert_eq!(
                        waiter.record_offset as usize,
                        record_offset(granularity, row)
                    );
                    assert_eq!(waiter.scale2, scales[s]);
                    let record = (row / 91 * 8192 + row % 91 * 90) as u64;
                    assert_eq!(
                        span.file_offset
                            + (waiter.record_offset as usize - span.buffer_offset) as u64,
                        record
                    );
                    assert!(!seen[waiter.request_index as usize]);
                    seen[waiter.request_index as usize] = true;
                }
                // Waiters of one read are in request order.
                assert!(
                    plan.waiters_of(read)
                        .windows(2)
                        .all(|w| w[0].request_index < w[1].request_index)
                );
            }
            assert!(seen.iter().all(|&s| s));
            // Reads are in first-use order.
            let mut first_use = Vec::new();
            for &(shard, row) in &sel {
                let key = (shard, read_key(granularity, row));
                if !first_use.contains(&key) && first_use.len() < 200 {
                    first_use.push(key);
                }
            }
            assert_eq!(&plan.keys[..first_use.len()], &first_use[..]);
        }
    }

    #[test]
    fn sector_plan_reads_far_fewer_bytes_than_half_pages() {
        let (rows, scales) = meta(128);
        let sel = selections(131_072, 128, 11);
        let half = plan_fast(&sel, &rows, &scales, ReadGranularity::HalfPage).unwrap();
        let sector = plan_fast(&sel, &rows, &scales, ReadGranularity::Sector).unwrap();
        assert!(
            sector.bytes() * 5 < half.bytes(),
            "{} vs {}",
            sector.bytes(),
            half.bytes()
        );
    }

    #[test]
    fn invalid_selections_fail_with_the_reader_messages() {
        let (rows, scales) = meta(2);
        for plan in [plan_fast, plan_reference] {
            let error =
                plan(&[(0, 1), (2, 0)], &rows, &scales, ReadGranularity::Sector).unwrap_err();
            assert_eq!(error, "PLE shard 2 out of range");
            let error = plan(
                &[(1, SHARD_ROWS)],
                &rows,
                &scales,
                ReadGranularity::HalfPage,
            )
            .unwrap_err();
            assert_eq!(
                error,
                format!("PLE row {SHARD_ROWS} out of range for shard 1")
            );
        }
        assert!(plan_fast(&[(0, 0)], &rows, &scales[..1], ReadGranularity::Page).is_err());
    }

    #[test]
    fn switches_and_trigger_modes_parse_strictly() {
        assert_eq!(parse_switch("X", None), Ok(false));
        assert_eq!(parse_switch("X", Some("0")), Ok(false));
        assert_eq!(parse_switch("X", Some("")), Ok(false));
        assert_eq!(parse_switch("X", Some("1")), Ok(true));
        assert!(parse_switch("X", Some("2")).is_err());
        assert!(parse_switch("X", Some("yes")).is_err());

        use ReadMode::*;
        assert_eq!(resolve_mode(false, false, None), Legacy);
        assert_eq!(resolve_mode(true, false, None), Planned);
        assert_eq!(resolve_mode(true, true, None), Sector);
        assert_eq!(resolve_mode(false, true, None), Sector);
        // Trigger file absent: off whatever the selectors say.
        assert_eq!(resolve_mode(true, true, Some(None)), Legacy);
        // Present: the load-time mode, narrowed by its content.
        assert_eq!(resolve_mode(true, true, Some(Some(""))), Sector);
        assert_eq!(resolve_mode(true, true, Some(Some("on\n"))), Sector);
        assert_eq!(resolve_mode(true, true, Some(Some("plan\n"))), Planned);
        assert_eq!(resolve_mode(true, true, Some(Some("legacy"))), Legacy);
        assert_eq!(resolve_mode(false, false, Some(Some("plan"))), Legacy);
        assert_eq!(resolve_mode(false, false, Some(Some("sector"))), Legacy);
        assert_eq!(resolve_mode(true, false, Some(Some("sector"))), Planned);
        assert_eq!(
            resolve_mode(true, true, Some(Some("plan single\n"))),
            Planned
        );
        assert_eq!(
            resolve_mode(true, true, Some(Some("sector single"))),
            Sector
        );
        assert_eq!(resolve_mode(true, true, Some(Some("single"))), Sector);
    }

    #[test]
    fn rings_parse_strictly_and_resolve_per_read() {
        assert_eq!(parse_rings("R", None), Ok(1));
        assert_eq!(parse_rings("R", Some("")), Ok(1));
        assert_eq!(parse_rings("R", Some("1")), Ok(1));
        assert_eq!(parse_rings("R", Some("2")), Ok(2));
        assert_eq!(parse_rings("R", Some("4")), Ok(4));
        for bad in ["0", "3", "8", " 2", "two"] {
            assert!(parse_rings("R", Some(bad)).is_err(), "{bad}");
        }
        use ReadMode::*;
        assert_eq!(resolve_rings(4, Legacy, None), 1);
        assert_eq!(resolve_rings(4, Sector, None), 4);
        assert_eq!(resolve_rings(4, Sector, Some(Some(""))), 4);
        assert_eq!(resolve_rings(4, Planned, Some(Some("plan single"))), 1);
        assert_eq!(resolve_rings(2, Sector, Some(Some("sector single\n"))), 1);
        assert_eq!(resolve_rings(2, Planned, Some(Some("plan"))), 2);
        assert_eq!(resolve_rings(0, Planned, None), 1);
        // Round-robin: every ring gets an equal share, in read order.
        let shares: Vec<usize> = (0..4)
            .map(|r| (0..10).filter(|&i| ring_of(i, 4) == r).count())
            .collect();
        assert_eq!(shares, vec![3, 3, 2, 2]);
        assert!((0..10).all(|i| ring_of(i, 1) == 0));
    }
}
