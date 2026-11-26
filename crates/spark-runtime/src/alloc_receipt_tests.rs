// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::Cell;

use super::*;
use crate::gpu::GpuBackend;
use crate::gpu::mock::MockGpuBackend;

const MIB: u64 = 1024 * 1024;

thread_local! {
    // Fake host `MemAvailable`: each test thread starts at 64 GiB and a test
    // subtracts what the "host" lost next to each mock allocation.
    static FAKE_MEM: Cell<usize> = const { Cell::new(64 << 30) };
}

fn fake_mem() -> Option<usize> {
    Some(FAKE_MEM.with(Cell::get))
}

fn lose(bytes: u64) {
    FAKE_MEM.with(|mem| mem.set(mem.get() - bytes as usize));
}

fn gain(bytes: u64) {
    FAKE_MEM.with(|mem| mem.set(mem.get() + bytes as usize));
}

fn mib(bytes: u64) -> i64 {
    (bytes * MIB) as i64
}

#[test]
fn mock_backend_counts_requested_freed_small_and_rounded_bytes() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let gpu = MockGpuBackend::with_alloc_counters(&COUNTERS);
    let source = Source::new(&COUNTERS, fake_mem);
    let start = source.mark();
    let big = gpu.alloc((3 * MIB) as usize).unwrap();
    let small = gpu.alloc(1024).unwrap();
    gpu.free(small).unwrap();
    let receipt = Receipt::between(start, source.mark());
    assert_eq!(receipt.requested_bytes, 3 * MIB + 1024);
    assert_eq!(
        (receipt.allocs, receipt.frees, receipt.small_allocs),
        (2, 1, 1)
    );
    assert_eq!(receipt.freed_bytes, 1024);
    assert_eq!(receipt.net_bytes(), mib(3));
    // 3 MiB rounds to 4 MiB; the 1 KiB allocation and its free cancel at 2 MiB.
    assert_eq!(receipt.rounded_net_bytes, mib(4));
    assert!(!big.is_null());
    // A plain mock counts nothing.
    let plain = MockGpuBackend::new();
    let before = COUNTERS.snapshot();
    plain.alloc(4096).unwrap();
    assert_eq!(COUNTERS.snapshot(), before);
}

#[test]
fn phase_receipt_reports_the_gap_between_net_bytes_and_memavailable() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let gpu = MockGpuBackend::with_alloc_counters(&COUNTERS);
    let source = Source::new(&COUNTERS, fake_mem);
    let ((), receipt) = run_phase(source, "unit-phase", || {
        gpu.alloc((10 * MIB) as usize).unwrap();
        // The host lost 2 MiB more than was requested (rounding, driver).
        lose(12 * MIB);
    });
    assert_eq!(receipt.phase, "unit-phase");
    assert_eq!(receipt.total.requested_bytes, 10 * MIB);
    assert_eq!(receipt.total.mem_available_delta_bytes, Some(mib(12)));
    assert_eq!(receipt.total.unattributed_bytes(), Some(mib(2)));
    assert!(receipt.passes.is_empty());
    assert_eq!(receipt.lines().len(), 1);
}

#[test]
fn passes_sum_over_instances_and_the_rest_is_outside_passes() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let gpu = MockGpuBackend::with_alloc_counters(&COUNTERS);
    let source = Source::new(&COUNTERS, fake_mem);
    let ((), receipt) = run_phase(source, "target-layers", || {
        for _layer in 0..3 {
            {
                let _pass = pass("ssm");
                let scratch = gpu.alloc((2 * MIB) as usize).unwrap();
                gpu.alloc(MIB as usize).unwrap();
                lose(3 * MIB);
                gpu.free(scratch).unwrap();
                gain(2 * MIB);
            }
            let _pass = pass("ffn");
            gpu.alloc((2 * MIB) as usize).unwrap();
            lose(2 * MIB);
        }
        gpu.alloc((4 * MIB) as usize).unwrap();
        lose(5 * MIB);
    });
    let names: Vec<_> = receipt
        .passes
        .iter()
        .map(|p| (p.name.as_str(), p.instances))
        .collect();
    assert_eq!(names, [("ssm", 3), ("ffn", 3)]);
    let ssm = &receipt.passes[0].receipt;
    assert_eq!((ssm.requested_bytes, ssm.freed_bytes), (9 * MIB, 6 * MIB));
    assert_eq!(ssm.mem_available_delta_bytes, Some(mib(3)));
    assert_eq!(ssm.unattributed_bytes(), Some(0));
    let outside = receipt.outside_passes();
    assert_eq!(outside.requested_bytes, 4 * MIB);
    assert_eq!(outside.mem_available_delta_bytes, Some(mib(5)));
    assert_eq!(outside.unattributed_bytes(), Some(mib(1)));
    assert_eq!(receipt.total.requested_bytes, 19 * MIB);
    assert_eq!(receipt.total.net_bytes(), mib(13));
    assert_eq!(receipt.total.mem_available_delta_bytes, Some(mib(14)));
    let lines = receipt.lines();
    assert_eq!(lines.len(), 4, "{lines:#?}");
    assert!(
        lines[1].contains("phase=target-layers pass=ssm instances=3 "),
        "{}",
        lines[1]
    );
    assert!(lines[3].contains("pass=outside-passes "), "{}", lines[3]);
    assert!(lines[3].ends_with("unattributed_mib=1.0"), "{}", lines[3]);
}

#[test]
fn an_inner_pass_is_folded_into_the_outer_one() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let gpu = MockGpuBackend::with_alloc_counters(&COUNTERS);
    let source = Source::new(&COUNTERS, fake_mem);
    let ((), receipt) = run_phase(source, "nested-pass", || {
        let _outer = pass("outer");
        gpu.alloc(MIB as usize).unwrap();
        let _inner = pass("inner");
        gpu.alloc(MIB as usize).unwrap();
    });
    assert_eq!(receipt.passes.len(), 1);
    assert_eq!(receipt.passes[0].name, "outer");
    assert_eq!(receipt.passes[0].receipt.requested_bytes, 2 * MIB);
    assert_eq!(receipt.outside_passes().requested_bytes, 0);
}

#[test]
fn a_nested_phase_is_one_pass_of_its_parent_and_the_session_counts_it_once() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let gpu = MockGpuBackend::with_alloc_counters(&COUNTERS);
    let source = Source::new(&COUNTERS, fake_mem);
    begin_session(source);
    gpu.alloc(MIB as usize).unwrap();
    lose(MIB);
    let (inner, outer) = run_phase(source, "outer-phase", || {
        gpu.alloc(MIB as usize).unwrap();
        lose(MIB);
        let ((), inner) = run_phase(source, "inner-phase", || {
            gpu.alloc((2 * MIB) as usize).unwrap();
            lose(2 * MIB);
        });
        inner
    });
    assert_eq!(inner.total.requested_bytes, 2 * MIB);
    assert_eq!(outer.total.requested_bytes, 3 * MIB);
    assert_eq!(outer.passes.len(), 1);
    assert_eq!(outer.passes[0].name, "inner-phase");
    assert_eq!(outer.outside_passes().requested_bytes, MIB);
    let session = session_receipt().expect("session started");
    assert_eq!(session.total.requested_bytes, 4 * MIB);
    assert_eq!(session.in_phases.requested_bytes, 3 * MIB);
    assert_eq!(session.in_phases.mem_available_delta_bytes, Some(mib(3)));
    let outside = session.outside_phases();
    assert_eq!(outside.requested_bytes, MIB);
    assert_eq!(outside.mem_available_delta_bytes, Some(mib(1)));
    let line = session.line();
    assert!(
        line.starts_with("STARTUP_ALLOC_RECEIPT_TOTAL since=ledger-reset "),
        "{line}"
    );
    assert!(
        line.contains("in_phases_mem_available_delta_mib=3.0"),
        "{line}"
    );
    assert!(
        line.ends_with("outside_phases_mem_available_delta_mib=1.0"),
        "{line}"
    );
}

#[test]
fn a_pass_outside_every_phase_measures_nothing() {
    let guard = pass("orphan");
    assert!(guard.start.is_none());
    drop(guard);
}

#[test]
fn a_panicking_phase_leaves_no_frame_behind() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let source = Source::new(&COUNTERS, fake_mem);
    let caught = std::panic::catch_unwind(|| {
        run_phase(source, "panics", || -> u8 { panic!("boom") });
    });
    assert!(caught.is_err());
    FRAMES.with(|frames| assert!(frames.borrow().is_empty()));
    let guard = pass("after-panic");
    assert!(guard.start.is_none());
}

#[test]
fn missing_memavailable_prints_a_dash_not_a_number() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let source = Source::new(&COUNTERS, || None);
    let ((), receipt) = run_phase(source, "no-meminfo", || {});
    assert_eq!(receipt.total.mem_available_delta_bytes, None);
    assert_eq!(receipt.total.unattributed_bytes(), None);
    let line = &receipt.lines()[0];
    assert!(
        line.ends_with("mem_available_delta_mib=- unattributed_mib=-"),
        "{line}"
    );
}

#[test]
fn pinned_host_bytes_count_toward_net() {
    static COUNTERS: AllocCounters = AllocCounters::new();
    let source = Source::new(&COUNTERS, fake_mem);
    let start = source.mark();
    COUNTERS.note_pinned_alloc((4 * MIB) as usize);
    COUNTERS.note_pinned_free(MIB as usize);
    let receipt = Receipt::between(start, source.mark());
    assert_eq!(receipt.pinned_net_bytes, mib(3));
    assert_eq!(receipt.net_bytes(), mib(3));
    assert_eq!(receipt.requested_bytes, 0);
}
