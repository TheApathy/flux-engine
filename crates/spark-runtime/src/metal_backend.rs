// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Implements [`GpuBackend`] on top of the Metal framework via the
//! `objc2-metal` bindings. Apple Silicon is unified-memory (UMA), so
//! every `MTLBuffer` is allocated with `StorageModeShared` and host
//! `memcpy` against `buffer.contents()` is the canonical H2D/D2H path
//! — no PCIe staging, no pinned-host bounce.
//!
//! # Pointer model
//!
//! `DevicePtr` carries a real GPU virtual address obtained from
//! `MTLBuffer::gpuAddress()` (Metal 3+, native to all Apple Silicon).
//! That makes pointer arithmetic (`DevicePtr::offset`) a plain integer
//! add — no buffer/offset pair to thread through. To recover the
//! owning `MTLBuffer` for `free` / blit-copy / `setBuffer:`, we keep
//! a side table `BTreeMap<base_gpu_address, MTLBuffer>` and look up
//! the largest key ≤ ptr. The buffer's gpuAddress range is
//! `[base, base + length)`, so a binary search is enough.
//!
//! # Streams
//!
//! A stream handle directly indexes a slab of `Arc<Mutex<MetalStream>>`
//! (`queue`, `in_flight`). Handle 0 is the default stream and is materialized
//! eagerly in [`MetalGpuBackend::new`]. Each per-stream mutex serializes
//! encoder creation through `endEncoding()` and command-buffer commits so
//! multiple threads never concurrently encode onto the same
//! `MTLCommandBuffer`. `synchronize(stream)` commits the in-flight
//! `MTLCommandBuffer` and `waitUntilCompleted()`s; the next encoder opens on
//! a fresh buffer.
//!
//! # Kernel handles
//!
//! `KernelHandle` indexes a slab of `MTLComputePipelineState`. The
//! library cache (one `MTLLibrary` per `metallib_modules()` entry) is
//! built once at construction and never mutated; pipeline lookups go
//! through the slab + a `(module, fn_name)` HashMap so repeated
//! `kernel()` calls are O(1) cached.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLEvent, MTLLibrary, MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use parking_lot::Mutex;

use crate::gpu::{
    DevicePtr, GpuBackend, KernelArg, KernelHandle, PinnedHostBuffer, PinnedHostStorage,
};

// ── Internal type aliases (Retained<ProtocolObject<dyn _>> is verbose) ────

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

struct MetalPinnedHostStorage {
    _buffer: ObjBuffer,
    ptr: NonNull<u8>,
    bytes: usize,
}

// SAFETY: `_buffer` is a retained `StorageModeShared` `MTLBuffer`, which Metal
// guarantees is safe to retain/release from any thread. `ptr` points into that
// buffer's zero-initialized UMA backing store and remains valid for `bytes`
// bytes until `_buffer` drops, while `PinnedHostBuffer` enforces Rust's `&` /
// `&mut` aliasing rules on slice access.
unsafe impl Send for MetalPinnedHostStorage {}
// SAFETY: Shared (`&`) access to `MetalPinnedHostStorage` only exposes
// immutable slice views over `ptr..ptr + bytes` and the thread-safe `MTLBuffer`
// handle; exclusive mutation requires `&mut PinnedHostBuffer`.
unsafe impl Sync for MetalPinnedHostStorage {}

impl PinnedHostStorage for MetalPinnedHostStorage {
    fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

// ── Stream + slab types ──────────────────────────────────────────────────

struct MetalStream {
    queue: ObjQueue,
    /// In-flight command buffer accumulating encoded work. Committed +
    /// waited on by `synchronize()`; replaced by a fresh buffer on
    /// next encoder open.
    in_flight: Option<ObjCmdBuf>,
}

impl MetalStream {
    /// Return (or create) the in-flight command buffer while the caller holds
    /// this stream's mutex.
    fn cmd_buf(&mut self, stream_handle: u64) -> Result<ObjCmdBuf> {
        if let Some(ref cb) = self.in_flight {
            return Ok(cb.clone());
        }
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null on stream {stream_handle}"))?;
        self.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// Take and commit the in-flight command buffer (no wait), returning it so
    /// the caller may `waitUntilCompleted()` after dropping the stream mutex.
    fn commit_in_flight(&mut self) -> Option<ObjCmdBuf> {
        let cb = self.in_flight.take()?;
        cb.commit();
        Some(cb)
    }
}

/// Tracks one outstanding shared event so `record_event` can write the
/// next signal value and `stream_wait_event` can wait on the same
/// counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// Monotonic value sequence — record_event signals `next`, then
    /// increments. stream_wait_event waits on `next - 1` (the most
    /// recently recorded value). Atomic via the surrounding Mutex.
    next: u64,
}

/// Key for the pipeline cache. Stored as owned strings because the
/// `&str` arguments to `kernel()` come from arbitrary call sites.
type PipelineKey = (String, String);

// ── MetalGpuBackend struct + state ───────────────────────────────────────

pub struct MetalGpuBackend {
    device: ObjDevice,
    /// Side table mapping a buffer's base gpuAddress to the owning
    /// `MTLBuffer`. BTreeMap so we can find the buffer containing an
    /// arbitrary `DevicePtr` via `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// Stream slab directly indexed by stream handle (`0, 1, 2, ...`). Handle 0
    /// is the default stream, materialized eagerly into slot 0 in `new()`.
    /// Each stream has its own `Mutex<MetalStream>` so command-buffer encoding
    /// and commits on a given stream never race across threads.
    streams: Arc<Mutex<Vec<Arc<Mutex<MetalStream>>>>>,
    /// Loaded metallibs keyed by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// Pipeline-state cache + slab. The HashMap maps `(module, fn)` to
    /// the slab index; the slab owns the `MTLComputePipelineState`.
    /// Both are mutexed so `kernel()` can be called from any thread.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// Shared-event slab for cross-stream synchronization (`None` = destroyed).
    events: Arc<Mutex<Vec<Option<EventSlot>>>>,
}

// SAFETY: Apple's Metal framework guarantees `MTLDevice`, `MTLLibrary`,
// `MTLComputePipelineState`, `MTLBuffer`, `MTLCommandQueue`, and
// `MTLSharedEvent` are safe to transfer across threads. Every mutable
// container and non-thread-safe `MTLCommandBuffer` is protected by a
// `parking_lot::Mutex` (per-stream `Arc<Mutex<MetalStream>>` for command
// buffers).
unsafe impl Send for MetalGpuBackend {}
// SAFETY: Concurrent callers share immutable `MTLDevice`/`MTLLibrary` handles
// (thread-safe per Metal docs) while all mutable state (`allocations`,
// `pipeline_cache`, `pipeline_slab`, `events`) and per-stream
// `MTLCommandBuffer` encoding/commit transitions are serialized by mutexes.
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// Initialize the Metal backend with the embedded metallib modules.
    ///
    /// `kernel_modules` is the `metallib_modules()` slice produced by
    /// `atlas-kernels`' build script — `(module_name, metallib_bytes)`.
    /// Each entry is loaded into its own `MTLLibrary` via
    /// `newLibraryWithData_error:`. The default stream (handle 0) is
    /// materialized eagerly so the first launch doesn't pay queue-
    /// creation latency.
    pub fn new(ordinal: usize, kernel_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU")
        })?;

        // Build the library cache up-front. `newLibraryWithData_error`
        // takes a `DispatchData`, which is libdispatch's reference-
        // counted byte container. We wrap the &'static slice via
        // dispatch2::DispatchData (zero-copy) — the metallibs are
        // embedded by include_bytes! and outlive the backend.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device.newLibraryWithData_error(&data).map_err(|e| {
                anyhow!(
                    "newLibraryWithData failed for module '{name}': {}",
                    e.localizedDescription()
                )
            })?;
            libraries.insert((*name).to_string(), lib);
        }

        // Materialize the default stream eagerly (slot 0 = handle 0).
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![Arc::new(Mutex::new(MetalStream {
            queue: default_queue,
            in_flight: None,
        }))];

        tracing::info!(
            "MetalGpuBackend initialized on device '{}' with {} metallib modules",
            device.name().to_string(),
            libraries.len()
        );

        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Return the underlying `MTLDevice` (escape hatch for advanced
    /// use cases — graph capture, custom resource creation, etc.).
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Look up the `MTLBuffer` owning `ptr` and the byte offset of
    /// `ptr` within it (half-open interval `[base, base + length)`).
    /// Returns `None` if no allocation contains `ptr`.
    fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = usize::try_from(ptr.0 - *base).ok()?;
        if offset >= buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// Checked bounds validation for `[offset, offset + bytes)` within `buf_len`.
    fn check_range(op: &str, offset: usize, bytes: usize, buf_len: usize) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("{op}: range overflows usize ({offset} + {bytes})"))?;
        if end > buf_len {
            bail!("{op}: range overflows buffer ({offset} + {bytes} > {buf_len})");
        }
        Ok(())
    }

    /// Resolve a stream handle directly to its per-stream lock.
    /// Handle `0` is the default stream (slot 0); handles returned by
    /// `create_stream()` index slots `1, 2, ...` directly without aliasing.
    fn stream_lock(&self, handle: u64) -> Result<Arc<Mutex<MetalStream>>> {
        let slab = self.streams.lock();
        let idx = usize::try_from(handle)
            .map_err(|_| anyhow!("Metal: invalid stream handle {handle}"))?;
        slab.get(idx)
            .cloned()
            .ok_or_else(|| anyhow!("Metal: invalid stream handle {handle}"))
    }

    /// Copy bytes out of a shared `MTLBuffer` without synchronizing any stream
    /// first. Callers (`copy_d2h`, `copy_d2h_on_stream`) synchronize the
    /// relevant stream beforehand.
    fn copy_d2h_unsynced(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        Self::check_range("copy_d2h", offset, dst.len(), buf.length())?;
        let contents: NonNull<c_void> = buf.contents();
        // SAFETY: `check_range` verified `offset + dst.len() <= buf.length()`
        // without overflow, `buf` is a live `StorageModeShared` buffer held by
        // `allocs` across the copy, and `dst` is a disjoint Rust slice.
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

// ── GpuBackend impl ──────────────────────────────────────────────────────

impl GpuBackend for MetalGpuBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        // StorageModeShared is the UMA-friendly mode: `contents()`
        // returns a CPU-mappable pointer that aliases GPU memory.
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // Apple Silicon UMA: managed and shared are the same thing.
        // No paged virtual memory swap mechanism (cuMemAllocManaged on
        // GB10) — Metal lets the OS handle pressure via its memory
        // pool. Defer to plain alloc.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Removing the entry drops the last `Retained` reference; the
        // ObjC runtime releases the underlying MTLBuffer.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        Self::check_range("copy_h2d", offset, src.len(), buf.length())?;
        let contents: NonNull<c_void> = buf.contents();
        // SAFETY: `check_range` verified `offset + src.len() <= buf.length()`
        // without overflow, `buf` is a live `StorageModeShared` buffer retained
        // by `allocs` across the copy, and `src` is a disjoint host slice.
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        // Synchronize the default stream before reading UMA memory so prior
        // default-stream GPU writes are visible (matching CUDA's blocking D2H).
        self.synchronize(0)?;
        self.copy_d2h_unsynced(src, dst)
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        // UMA: synchronize the requested stream so prior kernels have written
        // their bytes back through the cache, then memcpy.
        self.synchronize(stream)?;
        self.copy_d2h_unsynced(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (src_buf, src_off, dst_buf, dst_off) = {
            let allocs = self.allocations.lock();
            let (src_buf, src_off) = Self::find_buffer(&allocs, src)
                .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
            let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
                .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
            (src_buf, src_off, dst_buf, dst_off)
        };
        Self::check_range("copy_d2d(src)", src_off, bytes, src_buf.length())?;
        Self::check_range("copy_d2d(dst)", dst_off, bytes, dst_buf.length())?;

        let stream_lock = self.stream_lock(0)?;
        let cb = {
            let mut s = stream_lock.lock();
            let cmd_buf = s.cmd_buf(0)?;
            let enc = cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
            // SAFETY: Both `src_buf` and `dst_buf` are retained live `MTLBuffer`s,
            // `check_range` verified `[src_off, src_off + bytes)` and
            // `[dst_off, dst_off + bytes)` lie within their respective lengths,
            // and `s` serializes encoding on stream 0 until `endEncoding()`.
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src_buf, src_off, &dst_buf, dst_off, bytes,
                );
            }
            enc.endEncoding();
            s.commit_in_flight()
        };
        // Synchronize outside the stream lock so the d2d behaves like CUDA's
        // synchronous variant without blocking unrelated encoders.
        if let Some(cb) = cb {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (src_buf, src_off, dst_buf, dst_off) = {
            let allocs = self.allocations.lock();
            let (src_buf, src_off) = Self::find_buffer(&allocs, src)
                .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
            let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
                .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
            (src_buf, src_off, dst_buf, dst_off)
        };
        Self::check_range("copy_d2d_async(src)", src_off, bytes, src_buf.length())?;
        Self::check_range("copy_d2d_async(dst)", dst_off, bytes, dst_buf.length())?;

        let stream_lock = self.stream_lock(stream)?;
        let mut s = stream_lock.lock();
        let cmd_buf = s.cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        // SAFETY: Both `src_buf` and `dst_buf` are retained live `MTLBuffer`s,
        // `check_range` verified both ranges fit inside their buffers, and `s`
        // serializes command-buffer encoding on `stream` through `endEncoding()`.
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        Ok(())
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
        // Metal can't safely interpret untyped `*mut c_void` slots as
        // either buffers or scalars (CUDA gets away with this because
        // the driver cross-references the kernel signature). Callers
        // must use `launch_typed`; the cuda-style untyped path is
        // intentionally unsupported here.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // Resolve the pipeline state.
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            slab.get(func.0 as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        // Snapshot the alloc registry once so we can both `useResource:` every
        // live buffer and resolve `KernelArg::Buffer` args consistently without
        // holding the alloc lock during encoding.
        let allocs_snapshot: BTreeMap<u64, ObjBuffer> = self.allocations.lock().clone();

        // Validate and resolve all arguments BEFORE opening the compute encoder
        // so an invalid argument never drops an un-ended `MTLComputeCommandEncoder`.
        enum ResolvedArg<'a> {
            Buffer(ObjBuffer, usize),
            Bytes(&'a [u8]),
        }
        let mut resolved_args = Vec::with_capacity(args.len());
        for (idx, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Buffer(p) => {
                    let (buf, offset) = Self::find_buffer(&allocs_snapshot, *p)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {p} not allocated"))?;
                    resolved_args.push(ResolvedArg::Buffer(buf, offset));
                }
                KernelArg::Bytes(b) => {
                    if b.len() > 4096 {
                        bail!(
                            "launch_typed: arg #{idx} inline bytes length {} exceeds Metal's 4096-byte setBytes limit",
                            b.len()
                        );
                    }
                    resolved_args.push(ResolvedArg::Bytes(b));
                }
            }
        }

        let stream_lock = self.stream_lock(stream)?;
        let mut s = stream_lock.lock();
        let cmd_buf = s.cmd_buf(stream)?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        enc.setComputePipelineState(&pipeline);

        // Mark every live allocation as in-use so Metal's automatic
        // hazard tracking keeps them resident. Cheap on Apple Silicon
        // because `useResource:` is a hint, not a copy.
        for buf in allocs_snapshot.values() {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(
                resource,
                objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
            );
        }

        // Bind each validated arg to its index.
        for (idx, arg) in resolved_args.iter().enumerate() {
            match arg {
                ResolvedArg::Buffer(buf, offset) => {
                    // SAFETY: `buf` is a live `MTLBuffer` retained in `resolved_args`,
                    // `find_buffer` verified `*offset < buf.length()`, and `enc` is
                    // exclusively owned under the per-stream mutex `s`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&**buf), *offset, idx);
                    }
                }
                ResolvedArg::Bytes(b) => {
                    let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                        .expect("Rust slice pointer is always non-null");
                    // SAFETY: `b` is a valid initialized byte slice of length
                    // `<= 4096` (checked above against Metal's inline `setBytes`
                    // limit), and Metal copies the bytes synchronously during
                    // `setBytes_length_atIndex`.
                    unsafe {
                        enc.setBytes_length_atIndex(ptr, b.len(), idx);
                    }
                }
            }
        }

        let threadgroups = MTLSize {
            width: grid[0] as usize,
            height: grid[1] as usize,
            depth: grid[2] as usize,
        };
        let threads_per_tg = MTLSize {
            width: block[0] as usize,
            height: block[1] as usize,
            depth: block[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        enc.endEncoding();
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        let stream_lock = self.stream_lock(stream)?;
        let cb = stream_lock.lock().commit_in_flight();
        if let Some(cb) = cb {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    // Matches the `#[track_caller]` on the trait declaration. This backend does
    // not feed `kernel_audit` — the audit exists for the CUDA serve path's
    // `--check-kernels`, and Metal resolves against a different module set.
    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let lib = self
            .libraries
            .get(module)
            .ok_or_else(|| anyhow!("Metal: unknown module '{module}'"))?;
        let ns_name = NSString::from_str(func_name);
        let function = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            anyhow!("Metal: function '{func_name}' not found in module '{module}'")
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        let handle = KernelHandle(slab.len() as u64);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // Synchronize default stream first so synchronous CPU write_bytes
        // never races with uncommitted default-stream GPU work.
        self.synchronize(0)?;
        // On UMA we can write through `contents()` directly once stream 0 is
        // idle — cheaper than committing a separate blit fillBuffer.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        Self::check_range("memset", offset, bytes, buf.length())?;
        let contents = buf.contents();
        // SAFETY: `check_range` verified `offset + bytes <= buf.length()` without
        // overflow, `buf` is a live `StorageModeShared` buffer retained by `allocs`,
        // and stream 0 has been synchronized.
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (buf, offset) = {
            let allocs = self.allocations.lock();
            Self::find_buffer(&allocs, ptr)
                .ok_or_else(|| anyhow!("memset_async: ptr {ptr} not allocated"))?
        };
        Self::check_range("memset_async", offset, bytes, buf.length())?;

        // Encode a Metal blit fill on `stream` so the memset executes in exact
        // stream order relative to prior and subsequent kernels on `stream`.
        let stream_lock = self.stream_lock(stream)?;
        let mut s = stream_lock.lock();
        let cmd_buf = s.cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        enc.fillBuffer_range_value(&buf, NSRange::new(offset, bytes), value);
        enc.endEncoding();
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        // On Apple Silicon UMA, "device memory" = system RAM. Probe
        // hw.memsize via sysctl for the authoritative number; fall
        // back to MTLDevice.recommendedMaxWorkingSetSize otherwise.
        Ok(sysctl_memsize().unwrap_or_else(|| self.device.recommendedMaxWorkingSetSize() as usize))
    }

    fn free_memory(&self) -> Result<usize> {
        // No direct API for "free GPU memory" on UMA. Approximate via
        // `recommendedMaxWorkingSetSize - currentAllocatedSize`,
        // which matches the headroom Metal will let us allocate
        // before performance degrades.
        let max = self.device.recommendedMaxWorkingSetSize() as usize;
        let used = self.device.currentAllocatedSize();
        Ok(max.saturating_sub(used))
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        let handle = slab.len() as u64;
        slab.push(Arc::new(Mutex::new(MetalStream {
            queue,
            in_flight: None,
        })));
        // Direct slab index: slot 0 is default stream (handle 0), created
        // streams receive distinct handles 1, 2, ... without aliasing.
        Ok(handle)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Metal devices/queues are thread-safe; no binding required.
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(Some(EventSlot { event, next: 1 }));
        // Handle = slab index + 1 (0 reserved for "no event").
        Ok(slab.len() as u64)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let (event_obj, value) = {
            let mut slab = self.events.lock();
            let idx = usize::try_from(event)
                .ok()
                .and_then(|n| n.checked_sub(1))
                .ok_or_else(|| anyhow!("record_event: invalid event handle {event}"))?;
            let slot = slab.get_mut(idx).and_then(Option::as_mut).ok_or_else(|| {
                anyhow!("record_event: event handle {event} invalid or destroyed")
            })?;
            let v = slot.next;
            slot.next += 1;
            (slot.event.clone(), v)
        };
        let stream_lock = self.stream_lock(stream)?;
        let mut s = stream_lock.lock();
        let cmd_buf = s.cmd_buf(stream)?;
        // Encode the signal on the active command buffer and commit immediately
        // so another stream waiting on this event never deadlocks on an
        // uncommitted signal.
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeSignalEvent_value(proto, value);
        let _ = s.commit_in_flight();
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = usize::try_from(event)
                .ok()
                .and_then(|n| n.checked_sub(1))
                .ok_or_else(|| anyhow!("stream_wait_event: invalid event handle {event}"))?;
            let slot = slab.get(idx).and_then(Option::as_ref).ok_or_else(|| {
                anyhow!("stream_wait_event: event handle {event} invalid or destroyed")
            })?;
            // Wait on the most-recently-recorded value (next - 1).
            // If nothing has been recorded yet, slot.next is 1 and
            // value is 0 — Metal treats wait-for-0 as a no-op.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        let stream_lock = self.stream_lock(stream)?;
        let mut s = stream_lock.lock();
        let cmd_buf = s.cmd_buf(stream)?;
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeWaitForEvent_value(proto, value);
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let Some(idx) = usize::try_from(event).ok().and_then(|n| n.checked_sub(1)) else {
            return Ok(());
        };
        if let Some(slot) = slab.get_mut(idx) {
            // Drop the `MTLSharedEvent` immediately and mark the slot destroyed
            // without reusing slab indices.
            *slot = None;
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        if bytes == 0 {
            bail!("pinned host allocation must be non-empty");
        }
        // UMA: a Shared MTLBuffer's contents() pointer IS host-pinned memory
        // from the GPU's perspective, and Metal guarantees newly allocated
        // buffers are zero-initialized (satisfying `PinnedHostStorage`'s
        // initialized-bytes invariant). The returned RAII owner retains the
        // MTLBuffer and releases it automatically on drop.
        let buf = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let ptr = NonNull::new(buf.contents().as_ptr() as *mut u8)
            .ok_or_else(|| anyhow!("alloc_host_pinned: contents returned null"))?;
        Ok(PinnedHostBuffer::from_storage(Box::new(
            MetalPinnedHostStorage {
                _buffer: buf,
                ptr,
                bytes,
            },
        )))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Probe `hw.memsize` via libc::sysctl on macOS. Returns the total
/// system RAM in bytes — on Apple Silicon UMA this is also the upper
/// bound on Metal-addressable memory.
fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `name` is a valid NUL-terminated C string, `&mut value` points to
    // `size == size_of::<u64>()` writable bytes, and `newp == null, newlen == 0`
    // requests a read-only `sysctlbyname` query.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}

// ── Smoke test ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
