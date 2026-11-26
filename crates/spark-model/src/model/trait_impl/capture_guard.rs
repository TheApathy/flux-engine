// SPDX-License-Identifier: AGPL-3.0-only

//! RAII guard over a `begin_capture`/`end_capture` pair, shared by every
//! `trait_impl` sibling that captures a CUDA graph.
//!
//! `?` on any fallible call between `begin_capture` and `end_capture` (a
//! layer's `decode`/`verify`, a final norm, an LM head) used to return
//! straight out of the enclosing function, skipping `end_capture` entirely.
//! The CUDA stream stays reported as capturing (`stream_is_capturing` /
//! `cuStreamIsCapturing`) forever after that: every later blocking wait on
//! it — including an unrelated decode step's debug sync, and
//! `free_sequence`'s teardown-time `cuCtxSynchronize` — fails with
//! `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (or, under
//! `ATLAS_BOUNDED_HOST_COPY`, the clean `ensure_bounded_wait_outside_capture`
//! error), with no way to recover for the rest of the process. This module
//! centralizes RAII capture teardown for every capture region (including the
//! Qwen4 PLE suffix-graph path in `decode_qwen4_ple_graph.rs`).
//!
//! This guard closes the capture on every exit path. `Drop` is the error
//! path: if `end()` was never called (an earlier `?` propagated through the
//! scope that owns this guard), it best-effort ends the now-partial capture
//! and destroys the resulting graph — a graph captured from an aborted,
//! incomplete layer sequence must never be cached or replayed. `end()` is
//! the success path: it disarms the guard and returns the real handle,
//! identical to a direct `gpu.end_capture(stream)?` call.
use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, GraphHandle};

pub(crate) struct CaptureGuard<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    /// Static name of this capture region (e.g. `"decode_a_default"`,
    /// `"verify_b"`). Published via `spark_runtime::gpu_capture_region`
    /// while the guard is open so a bounded-host-copy refusal fired from
    /// deep inside the backend can name which region it happened in, and
    /// used in this guard's own diagnostics and nested-begin refusal.
    region: &'static str,
    /// `true` from `begin_capture` until `end()` consumes the guard. `Drop`
    /// only acts while this is still `true` — i.e. only on the error path.
    open: bool,
}

impl<'a> CaptureGuard<'a> {
    /// Begins a capture on `stream`, tagged `region` for diagnostics.
    /// Refuses to nest: if a `CaptureGuard` is already open on this thread
    /// (an earlier `begin` whose `end`/`Drop` has not yet run), this backend
    /// never supports two concurrently-open captures — `cuStreamBeginCapture`
    /// on an already-capturing stream is undefined by the CUDA driver, and a
    /// nested `CaptureGuard` would make the inner `Drop` close the outer
    /// capture by mistake. The only way this can happen in practice is a new
    /// bug (a capture region calling into another capturing code path), so
    /// it is treated as a hard error rather than something to work around.
    pub(crate) fn begin(
        gpu: &'a dyn GpuBackend,
        stream: u64,
        region: &'static str,
    ) -> Result<Self> {
        if let Some(existing) = spark_runtime::gpu_capture_region::active_capture_region() {
            anyhow::bail!(
                "CaptureGuard::begin({region}) refused: region '{existing}' is already \
                 open on this thread. Nested CUDA graph captures are not supported — \
                 this means a capture region called into another capturing code path."
            );
        }
        gpu.begin_capture(stream)?;
        spark_runtime::gpu_capture_region::set_active_capture_region(Some(region));
        Ok(Self {
            gpu,
            stream,
            region,
            open: true,
        })
    }

    /// Ends the capture and returns its graph handle — the normal,
    /// everything-succeeded exit. Disarms the guard first so `Drop` does
    /// not also try to end a capture this already closed, and clears the
    /// published region name.
    pub(crate) fn end(mut self) -> Result<GraphHandle> {
        self.open = false;
        spark_runtime::gpu_capture_region::set_active_capture_region(None);
        self.gpu.end_capture(self.stream)
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if !self.open {
            return;
        }
        // Reached on two paths: (a) something between `begin_capture` and
        // `end()` returned `Err` and propagated out through this guard's
        // scope, or (b) that same span panicked and is unwinding — `Drop`
        // runs during unwind too (as long as the process does not abort),
        // so this closes the capture either way. Best-effort close the
        // capture so the stream is usable again for the NEXT call instead
        // of staying wedged for the rest of the process; this can itself
        // fail (e.g. if the earlier error was already a stream/context
        // fault), which is logged, not propagated — `Drop` cannot return a
        // `Result` and the original error/panic already in flight is what
        // the caller needs to see.
        spark_runtime::gpu_capture_region::set_active_capture_region(None);
        match self.gpu.end_capture(self.stream) {
            Ok(graph) if graph.0 != 0 => {
                // This graph captured only the layers that ran before the
                // failure — a partial, inconsistent sequence. Destroy it
                // rather than leak the GPU graph object; it must never be
                // inserted into a graph cache or launched.
                if let Err(e) = self.gpu.destroy_graph(graph) {
                    tracing::error!(
                        "CaptureGuard({}): failed to destroy the partial graph from an \
                         aborted capture on stream {}: {e:#}",
                        self.region,
                        self.stream
                    );
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::error!(
                    "CaptureGuard({}): failed to close a capture left open by an earlier \
                     error on stream {}: {e:#}. The stream may remain wedged for \
                     subsequent calls.",
                    self.region,
                    self.stream
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaptureGuard;
    use anyhow::Result;
    use parking_lot::Mutex;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal `GpuBackend` whose only job is to answer the three capture
    /// calls `CaptureGuard` makes (`begin_capture`, `end_capture`,
    /// `destroy_graph`) in a scriptable way and record what happened. Every
    /// other trait method is unreachable from these tests.
    ///
    /// `GpuBackend: Send + Sync` (it is held behind `&dyn GpuBackend` and
    /// shared across threads in production), so the mock's interior
    /// mutability must be thread-safe: `Mutex`/`AtomicUsize`, not `Cell`.
    struct CaptureOnlyGpu {
        end_capture_result: Mutex<Option<Result<GraphHandle>>>,
        begin_calls: AtomicUsize,
        end_calls: AtomicUsize,
        destroy_calls: Mutex<Vec<GraphHandle>>,
    }

    impl CaptureOnlyGpu {
        fn new(end_capture_result: Result<GraphHandle>) -> Self {
            Self {
                end_capture_result: Mutex::new(Some(end_capture_result)),
                begin_calls: AtomicUsize::new(0),
                end_calls: AtomicUsize::new(0),
                destroy_calls: Mutex::new(Vec::new()),
            }
        }

        fn begin_calls(&self) -> usize {
            self.begin_calls.load(Ordering::SeqCst)
        }

        fn end_calls(&self) -> usize {
            self.end_calls.load(Ordering::SeqCst)
        }

        fn destroy_calls(&self) -> Vec<GraphHandle> {
            std::mem::take(&mut *self.destroy_calls.lock())
        }
    }

    impl GpuBackend for CaptureOnlyGpu {
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn free(&self, _ptr: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn launch(
            &self,
            _func: KernelHandle,
            _grid: [u32; 3],
            _block: [u32; 3],
            _shared_mem: u32,
            _stream: u64,
            _params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn synchronize(&self, _stream: u64) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn default_stream(&self) -> u64 {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn memset_async(
            &self,
            _ptr: DevicePtr,
            _value: u8,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn total_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by CaptureGuard tests")
        }
        fn free_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by CaptureGuard tests")
        }

        fn begin_capture(&self, _stream: u64) -> Result<()> {
            self.begin_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn end_capture(&self, _stream: u64) -> Result<GraphHandle> {
            self.end_calls.fetch_add(1, Ordering::SeqCst);
            // `Result<GraphHandle>` is not `Clone`; consume the scripted
            // answer on first use (every test here calls `end_capture` at
            // most once, whether directly via `CaptureGuard::end` or
            // indirectly via `Drop`).
            self.end_capture_result
                .lock()
                .take()
                .expect("end_capture called more than once in this test")
        }

        fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
            self.destroy_calls.lock().push(graph);
            Ok(())
        }
    }

    #[test]
    fn end_disarms_the_guard_so_drop_does_not_close_it_again() {
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0xBEEF)));
        let guard = CaptureGuard::begin(&gpu, 7, "test_region").expect("begin_capture");
        assert_eq!(gpu.begin_calls(), 1);

        let graph = guard.end().expect("end_capture");
        assert_eq!(graph.0, 0xBEEF);
        assert_eq!(gpu.end_calls(), 1, "end_capture must run exactly once");
        assert!(
            gpu.destroy_calls().is_empty(),
            "a successfully captured graph must not be destroyed by the guard"
        );
        // Guard already consumed by `end()` — nothing left to drop that
        // could call `end_capture` a second time.
    }

    #[test]
    fn drop_closes_an_abandoned_capture_when_end_was_never_called() {
        // Models the bug class this guard closes: a layer's `decode()?`/
        // `verify()?` (or a final norm, an LM head) returns `Err` between
        // `begin_capture` and `end()`. Before this fix nothing closed the
        // capture; now `Drop` does.
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0)));
        {
            let _guard = CaptureGuard::begin(&gpu, 7, "test_region").expect("begin_capture");
            // (a fallible layer call would `?` out of scope right here)
        }
        assert_eq!(
            gpu.end_calls(),
            1,
            "Drop must close a capture that was never explicitly ended"
        );
    }

    #[test]
    fn drop_destroys_a_nonzero_graph_from_an_aborted_capture() {
        // If the capture had already recorded real work by the time the
        // error fired, `end_capture` inside `Drop` can still hand back a
        // non-null handle. That graph is a partial, inconsistent sequence
        // of layers and must be destroyed, never cached or launched.
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0xDEAD)));
        {
            let _guard = CaptureGuard::begin(&gpu, 7, "test_region").expect("begin_capture");
        }
        assert_eq!(gpu.end_calls(), 1);
        let destroyed: Vec<u64> = gpu.destroy_calls().into_iter().map(|g| g.0).collect();
        assert_eq!(
            destroyed,
            vec![0xDEAD],
            "Drop must destroy the partial graph instead of leaking or keeping it"
        );
    }

    #[test]
    fn drop_does_not_panic_when_end_capture_itself_fails() {
        // The original error in flight is what the caller needs to see;
        // Drop cannot return a Result, so a failure to close the capture
        // here must be logged, not panic the unwind.
        let gpu = CaptureOnlyGpu::new(Err(anyhow::anyhow!("cuStreamEndCapture failed")));
        {
            let _guard = CaptureGuard::begin(&gpu, 7, "test_region").expect("begin_capture");
        }
        assert_eq!(gpu.end_calls(), 1);
        assert!(gpu.destroy_calls().is_empty());
    }

    #[test]
    fn nested_begin_is_refused_while_a_guard_is_open() {
        // `cuStreamBeginCapture` on an already-capturing stream is undefined;
        // a second `CaptureGuard::begin` while the first is still open must
        // fail cleanly instead of starting a capture CUDA will reject (or,
        // worse, one whose `Drop` would close the OUTER capture by mistake).
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0)));
        let outer = CaptureGuard::begin(&gpu, 7, "outer_region").expect("begin_capture");
        assert_eq!(gpu.begin_calls(), 1);

        let err = CaptureGuard::begin(&gpu, 7, "inner_region")
            .err()
            .expect("nested begin must be refused");
        let msg = err.to_string();
        assert!(msg.contains("outer_region"), "message: {msg}");
        assert!(msg.contains("inner_region"), "message: {msg}");
        // The refused nested begin never called the backend at all.
        assert_eq!(gpu.begin_calls(), 1);

        // The outer guard still closes normally afterward.
        let graph = outer.end().expect("end_capture");
        assert_eq!(graph.0, 0);
        assert_eq!(gpu.end_calls(), 1);
    }

    #[test]
    fn begin_is_allowed_again_after_the_first_guard_closes() {
        // Not nesting — sequential captures on the same thread must both
        // succeed once the first guard's `end()`/`Drop` has cleared the
        // published region.
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0)));
        let first = CaptureGuard::begin(&gpu, 7, "region_a").expect("first begin_capture");
        first.end().expect("end_capture");

        let gpu2 = CaptureOnlyGpu::new(Ok(GraphHandle(0)));
        CaptureGuard::begin(&gpu2, 7, "region_b").expect("second begin_capture should succeed");
    }

    #[test]
    fn drop_closes_the_capture_when_the_body_panics() {
        // The general case this guard exists for is not just an early `?`
        // return — it is ANY early exit from the scope that owns the guard,
        // including an unwinding panic from a layer's `decode()`/`verify()`.
        // `Drop` runs during unwind (the process does not abort here), so
        // `catch_unwind` around a panicking body must still observe exactly
        // one `end_capture` call once the panic has been caught.
        let gpu = CaptureOnlyGpu::new(Ok(GraphHandle(0xFEED)));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = CaptureGuard::begin(&gpu, 7, "panicking_region").expect("begin_capture");
            panic!("simulated layer failure mid-capture");
        }));
        assert!(
            result.is_err(),
            "the panic must still propagate to catch_unwind"
        );
        assert_eq!(
            gpu.end_calls(),
            1,
            "Drop must close the capture even when the body panicked"
        );
        let destroyed: Vec<u64> = gpu.destroy_calls().into_iter().map(|g| g.0).collect();
        assert_eq!(
            destroyed,
            vec![0xFEED],
            "the partial graph from the panicking capture must be destroyed"
        );
        // The region must not be left published after the panic unwound.
        assert_eq!(
            spark_runtime::gpu_capture_region::active_capture_region(),
            None,
            "a panicking capture must not leave its region published forever"
        );
    }
}
