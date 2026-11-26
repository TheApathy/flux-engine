// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Result, bail};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::*;

struct TraceGpu {
    inner: MockGpuBackend,
    trace: Mutex<Vec<String>>,
}

impl TraceGpu {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            trace: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, value: impl Into<String>) {
        self.trace.lock().unwrap().push(value.into());
    }
}

impl GpuBackend for TraceGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        Ok(())
    }
    fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.push(format!("record:{event}:{stream}"));
        Ok(())
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.push(format!("wait:{stream}:{event}"));
        Ok(())
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.push(format!("sync:{stream}"));
        Ok(())
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        self.push(format!("destroy:{event}"));
        Ok(())
    }
}

fn overlap() -> Glm53MoeOverlap {
    Glm53MoeOverlap {
        side_stream: 11,
        fork: 21,
        join: 22,
        executions: AtomicU64::new(0),
        busy: AtomicBool::new(false),
    }
}

fn run<'a>(overlap: &'a Glm53MoeOverlap, gpu: &'a TraceGpu) -> Joined<'a> {
    overlap
        .run(
            gpu,
            7,
            2047,
            |stream| {
                gpu.push(format!("shared:{stream}"));
                Ok(())
            },
            || {
                gpu.push("routed:7");
                Ok(())
            },
        )
        .unwrap()
}

#[test]
fn successful_completion_has_no_sync_and_records_one_receipt() {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    let joined = run(&overlap, &gpu);
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 0);
    joined.complete(2047);
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 1);
    assert_eq!(
        *gpu.trace.lock().unwrap(),
        [
            "record:21:7",
            "wait:11:21",
            "shared:11",
            "record:22:11",
            "routed:7",
            "wait:7:22",
        ]
    );
}

fn inject_post_join_error(joined: Joined<'_>, site: &str) -> Result<()> {
    let _joined = joined;
    bail!("injected {site} failure")
}

fn assert_post_join_error_quiesces(site: &str) {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    assert!(inject_post_join_error(run(&overlap, &gpu), site).is_err());
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 0);
    assert!(
        gpu.trace
            .lock()
            .unwrap()
            .ends_with(&["sync:7".into(), "sync:11".into()])
    );
}

#[test]
fn oracle_error_after_join_quiesces_both_streams_and_has_no_receipt() {
    assert_post_join_error_quiesces("oracle");
}

#[test]
fn combine_error_after_join_quiesces_both_streams_and_has_no_receipt() {
    assert_post_join_error_quiesces("combine");
}

#[test]
fn concurrent_event_reuse_is_rejected_before_launch_then_released() {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    let joined = run(&overlap, &gpu);
    let before = gpu.trace.lock().unwrap().clone();
    assert!(overlap.run(&gpu, 9, 2047, |_| Ok(()), || Ok(())).is_err());
    assert_eq!(*gpu.trace.lock().unwrap(), before);
    joined.complete(2047);
    run(&overlap, &gpu).complete(2047);
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 2);
}

#[test]
fn admission_error_releases_busy_without_launch_or_sync() {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    assert!(overlap.run(&gpu, 7, 1023, |_| Ok(()), || Ok(())).is_err());
    assert!(gpu.trace.lock().unwrap().is_empty());
    run(&overlap, &gpu).complete(2047);
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 1);
}

#[test]
fn routed_enqueue_failure_quiesces_both_streams_and_has_no_receipt() {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    assert!(
        overlap
            .run(&gpu, 7, 2048, |_| Ok(()), || bail!("injected"))
            .is_err()
    );
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 0);
    assert!(
        gpu.trace
            .lock()
            .unwrap()
            .ends_with(&["sync:7".into(), "sync:11".into()])
    );
    run(&overlap, &gpu).complete(2047);
    assert_eq!(overlap.executions.load(Ordering::Relaxed), 1);
}

#[test]
fn free_drains_side_stream_and_destroys_events() {
    let gpu = TraceGpu::new();
    let overlap = overlap();
    overlap.free(&gpu).unwrap();
    assert_eq!(
        *gpu.trace.lock().unwrap(),
        ["sync:11", "destroy:21", "destroy:22"]
    );
}
