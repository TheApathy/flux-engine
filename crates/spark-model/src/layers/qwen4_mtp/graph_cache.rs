// SPDX-License-Identifier: AGPL-3.0-only

//! Pinned H2D staging ring and CUDA graph cache for batched Qwen4 MTP chains.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy};

use super::policy::MTP_GRAPH_CACHE_CAP;

/// Capacity of the MTP head's page-locked H2D staging ring (`8 MiB`), matching
/// the target model's fast-H2D ring so `propose_batch` / `propose_chain` upload
/// attention and Gumbel metadata via `cuMemcpyHtoDAsync` without draining the GPU.
pub(super) const MTP_PINNED_H2D_RING_BYTES: usize = 8 * 1024 * 1024;

pub(super) struct MtpPinnedH2dRing {
    pub(super) buffer: spark_runtime::gpu::PinnedHostBuffer,
    pub(super) cursor: usize,
    /// Per-half CUDA events fencing the latest async H2D DMA reading from
    /// half 0 (`[0..mid)`) and half 1 (`[mid..cap)`).
    pub(super) half_events: [u64; 2],
    /// Whether `half_events[h]` tracks at least one async H2D DMA in half `h`
    /// that has not yet been confirmed complete via `poll_event`.
    pub(super) half_in_flight: [bool; 2],
    /// Stream on which the latest in-flight DMA in half `h` was recorded.
    pub(super) half_stream: [Option<u64>; 2],
    /// Set when the cursor wraps past half `h` while it still has in-flight
    /// DMAs from the prior pass; the next pass must confirm `half_events[h]`
    /// completed before overwriting any byte in half `h`.
    pub(super) half_needs_fence: [bool; 2],
    /// Permanently disables the ring after an unfenceable stream/event failure
    /// or after [`Self::destroy`] releases its events.
    pub(super) poisoned: bool,
}

impl MtpPinnedH2dRing {
    pub(super) fn new(gpu: &dyn GpuBackend, bytes: usize) -> Option<Self> {
        let buffer = gpu.alloc_host_pinned(bytes).ok()?;
        let event0 = gpu.create_event().ok()?;
        let event1 = match gpu.create_event() {
            Ok(ev) => ev,
            Err(_) => {
                if event0 != 0 {
                    let _ = gpu.destroy_event(event0);
                }
                return None;
            }
        };
        Some(Self {
            buffer,
            cursor: 0,
            half_events: [event0, event1],
            half_in_flight: [false, false],
            half_stream: [None, None],
            half_needs_fence: [false, false],
            poisoned: false,
        })
    }

    pub(super) fn destroy(&mut self, gpu: &dyn GpuBackend) {
        self.poisoned = true;
        for ev in &mut self.half_events {
            let event = *ev;
            if event != 0 {
                if let Err(error) = gpu.destroy_event(event) {
                    tracing::error!(
                        "MtpPinnedH2dRing::destroy: destroy_event({event}) failed: {error:#}"
                    );
                }
                *ev = 0;
            }
        }
        self.half_in_flight = [false, false];
        self.half_stream = [None, None];
        self.half_needs_fence = [false, false];
    }

    pub(super) fn plan_range(&self, bytes: usize) -> Option<(usize, usize, bool)> {
        let cap = self.buffer.len();
        if bytes > cap {
            return None;
        }
        let aligned = self.cursor.checked_add(7)? & !7;
        let (start, wrapped) = if aligned.checked_add(bytes).is_some_and(|end| end <= cap) {
            (aligned, false)
        } else {
            (0, true)
        };
        let end = start.checked_add(bytes)?;
        if end > cap {
            return None;
        }
        Some((start, end, wrapped))
    }

    pub(super) fn copy_h2d_group(
        &mut self,
        gpu: &dyn GpuBackend,
        copies: &[HostToDeviceCopy<'_>],
        stream: u64,
    ) -> Result<bool> {
        if copies.is_empty() {
            return Ok(true);
        }
        if self.poisoned || gpu.stream_is_capturing(stream) {
            return Ok(false);
        }
        let mut total_aligned = 0usize;
        for copy in copies {
            let Some(aligned) = copy.src().len().checked_add(7).map(|n| n & !7) else {
                return Ok(false);
            };
            match total_aligned.checked_add(aligned) {
                Some(sum) => total_aligned = sum,
                None => return Ok(false),
            }
        }
        if total_aligned == 0 {
            return Ok(true);
        }
        let Some((start, end, wrapped)) = self.plan_range(total_aligned) else {
            return Ok(false);
        };
        let cap = self.buffer.len();
        let mid = (cap / 2) & !7;
        let touches = [start < mid, end > mid];

        // Confirm every half we are about to reuse after wrap-around (or on a
        // different stream) has finished all prior async DMAs before writing.
        for (h, &touched) in touches.iter().enumerate() {
            if !touched {
                continue;
            }
            let must_confirm_complete = self.half_in_flight[h]
                && (wrapped || self.half_needs_fence[h] || self.half_stream[h] != Some(stream));
            if must_confirm_complete {
                if !gpu.poll_event(self.half_events[h])? {
                    return Ok(false);
                }
                self.half_in_flight[h] = false;
                self.half_needs_fence[h] = false;
                self.half_stream[h] = None;
            }
        }

        let mut cursor = start;
        let mut enqueued = false;
        let mut copy_err: Option<anyhow::Error> = None;
        for copy in copies {
            let src = copy.src();
            if src.is_empty() {
                continue;
            }
            let len = src.len();
            // SAFETY: `plan_range` validated `[start .. end) <= self.buffer.len()`,
            // so `[cursor .. cursor + len)` is within `self.buffer` and disjoint
            // from `src`. Any prior async DMA reading from the overlapped
            // half(ves) was confirmed complete via `poll_event` above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    self.buffer.as_mut_ptr().add(cursor),
                    len,
                );
            }
            let pinned_slice = match self.buffer.pinned_slice_range(cursor, len) {
                Ok(slice) => slice,
                Err(err) => {
                    copy_err = Some(err);
                    break;
                }
            };
            // SAFETY: `self.buffer` is backend-owned page-locked memory held
            // until teardown synchronizes streams and calls
            // `release_rejection_resources`. Each touched half records its
            // CUDA event on `stream` below and is polled to completion before
            // wrap-around or cross-stream reuse can overwrite those bytes.
            if let Err(err) = unsafe { gpu.copy_h2d_pinned_async(pinned_slice, copy.dst(), stream) }
            {
                copy_err = Some(err);
                break;
            }
            enqueued = true;
            cursor = (cursor + len + 7) & !7;
        }

        if enqueued {
            if wrapped {
                for (h, &touched) in touches.iter().enumerate() {
                    if !touched && self.half_in_flight[h] {
                        self.half_needs_fence[h] = true;
                    }
                }
            }
            self.cursor = end;
            for (h, &touched) in touches.iter().enumerate() {
                if !touched {
                    continue;
                }
                if let Err(record_err) = gpu.record_event(self.half_events[h], stream) {
                    if gpu.synchronize(stream).is_err() {
                        self.poisoned = true;
                    } else {
                        self.half_in_flight[h] = false;
                        self.half_needs_fence[h] = false;
                        self.half_stream[h] = None;
                    }
                    return Err(copy_err.unwrap_or(record_err));
                }
                self.half_in_flight[h] = true;
                self.half_stream[h] = Some(stream);
                self.half_needs_fence[h] = false;
            }
        }

        if let Some(err) = copy_err {
            return Err(err);
        }
        Ok(true)
    }
}

/// Shape- and address-exact key for cached multi-step batched MTP CUDA graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MtpGraphKey {
    pub(super) batch_size: usize,
    pub(super) step_count: usize,
    pub(super) want_prob: bool,
    pub(super) bt_blocks_cap: u32,
    pub(super) vocab: usize,
    pub(super) write_step_token_out: bool,
    pub(super) tc_dense_scope: bool,
    pub(super) coalesced_fc: bool,
    pub(super) coalesced_lm_head: bool,
    pub(super) scratch_ptr: u64,
    pub(super) hidden_ptr: u64,
    pub(super) residual_ptr: u64,
    pub(super) logits_ptr: u64,
    pub(super) k_pool_ptr: u64,
    pub(super) v_pool_ptr: u64,
    pub(super) stream: u64,
}

#[derive(Default)]
pub(super) struct MtpGraphCache {
    pub(super) warmed: HashSet<MtpGraphKey>,
    pub(super) graphs: HashMap<MtpGraphKey, GraphHandle>,
    pub(super) order: VecDeque<MtpGraphKey>,
    pub(super) poisoned: HashSet<MtpGraphKey>,
}

impl MtpGraphCache {
    pub(super) fn get(&self, key: &MtpGraphKey) -> Option<GraphHandle> {
        self.graphs.get(key).copied()
    }

    pub(super) fn is_poisoned(&self, key: &MtpGraphKey) -> bool {
        self.poisoned.contains(key)
    }

    pub(super) fn mark_warmed(&mut self, key: MtpGraphKey) -> bool {
        self.warmed.insert(key)
    }

    pub(super) fn mark_poisoned(&mut self, key: MtpGraphKey) {
        self.poisoned.insert(key);
    }

    pub(super) fn insert(&mut self, gpu: &dyn GpuBackend, key: MtpGraphKey, graph: GraphHandle) {
        if let Some(prev) = self.graphs.insert(key, graph) {
            if let Err(error) = gpu.destroy_graph(prev) {
                tracing::error!(
                    "Qwen4MtpHead: destroy_graph({prev:?}) on replace failed: {error:#}"
                );
            }
            return;
        }
        self.order.push_back(key);
        while self.graphs.len() > MTP_GRAPH_CACHE_CAP {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.graphs.remove(&oldest)
                && let Err(error) = gpu.destroy_graph(evicted)
            {
                tracing::error!(
                    "Qwen4MtpHead: destroy_graph({evicted:?}) on eviction failed: {error:#}"
                );
            }
            self.warmed.remove(&oldest);
        }
    }

    pub(super) fn clear(&mut self, gpu: &dyn GpuBackend) {
        for (_, graph) in self.graphs.drain() {
            if let Err(error) = gpu.destroy_graph(graph) {
                tracing::error!(
                    "Qwen4MtpHead::release_rejection_resources: destroy_graph({graph:?}) failed: {error:#}"
                );
            }
        }
        self.order.clear();
        self.warmed.clear();
        self.poisoned.clear();
    }
}

/// Static parameters for one capture-safe batched MTP multi-step replay.
pub(super) struct MtpBatchedChainLaunch<'a> {
    pub(super) batch_size: usize,
    pub(super) step_count: usize,
    pub(super) want_prob: bool,
    pub(super) bt_blocks_cap: usize,
    pub(super) stride: usize,
    pub(super) last_tok_off: usize,
    pub(super) slot_off: usize,
    pub(super) seq_len_off: usize,
    pub(super) hidden_ptrs_off: usize,
    pub(super) bt_off: usize,
    pub(super) vocab: usize,
    pub(super) meta_base: DevicePtr,
    pub(super) argmax_base: DevicePtr,
    pub(super) prob_base: DevicePtr,
    pub(super) token_out_base: Option<DevicePtr>,
    pub(super) base_seq_lens: &'a [usize],
}
