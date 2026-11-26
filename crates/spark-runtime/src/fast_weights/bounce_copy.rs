// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded pinned H2D loading. A failed/uncertain transfer poisons this copier;
//! its source, event and exact destination allocation remain retained rather
//! than being overwritten, retried or freed under DMA.

use crate::gpu::{DevicePtr, GpuBackend, PinnedHostBuffer};
use anyhow::{Result, bail, ensure};
use std::time::{Duration, Instant};

const EVENT_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(2);
const SLOTS: usize = 3;
const CHUNK_BYTES: usize = 64 * 1024 * 1024;

fn shard_chunk_bytes(max_tensor_bytes: usize, scalar_slab_bytes: usize) -> usize {
    max_tensor_bytes
        .max(scalar_slab_bytes)
        .clamp(1, CHUNK_BYTES)
}

fn shard_slot_count(max_tensor_bytes: usize, scalar_slab_bytes: usize) -> usize {
    let largest_payload = max_tensor_bytes.max(scalar_slab_bytes);
    // copy_once restarts at slot zero, so a nonempty one-chunk copy uses no other slot.
    if largest_payload > 0 && largest_payload <= CHUNK_BYTES {
        1
    } else {
        SLOTS
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Flight {
    #[default]
    Idle,
    // A fresh, unrecorded event can report ready. Never poll this state.
    Untracked {
        event: u64,
        destination: DevicePtr,
    },
    Recorded {
        event: u64,
        destination: DevicePtr,
    },
}

struct Slot {
    buffer: Option<PinnedHostBuffer>,
    flight: Flight,
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.flight != Flight::Idle
            && let Some(buffer) = self.buffer.take()
        {
            // Driver progress is uncertain. Retain until process/context
            // teardown; dropping the owner would call cuMemFreeHost.
            std::mem::forget(buffer);
        }
    }
}

pub(super) struct PinnedBounceCopier<'a> {
    gpu: &'a dyn GpuBackend,
    slots: Vec<Slot>,
    stream: u64,
    chunk_bytes: usize,
    event_timeout: Duration,
    poisoned: bool,
}

impl<'a> PinnedBounceCopier<'a> {
    pub(super) fn new(
        gpu: &'a dyn GpuBackend,
        max_tensor_bytes: usize,
        scalar_slab_bytes: usize,
    ) -> Result<Self> {
        // Tiny expert shards need the slab's full image, not three 64-MiB slots.
        let chunk_bytes = shard_chunk_bytes(max_tensor_bytes, scalar_slab_bytes);
        let slot_count = shard_slot_count(max_tensor_bytes, scalar_slab_bytes);
        let copier = Self::with_limits(gpu, slot_count, chunk_bytes, EVENT_TIMEOUT)?;
        tracing::info!(
            target: "atlas_weight_allocation_audit",
            max_tensor_bytes,
            scalar_slab_bytes,
            slot_count,
            slot_bytes = chunk_bytes,
            requested_pool_bytes = slot_count * chunk_bytes,
            "FAST_WEIGHT_PINNED_BOUNCE_POOL"
        );
        Ok(copier)
    }

    fn with_limits(
        gpu: &'a dyn GpuBackend,
        slot_count: usize,
        chunk_bytes: usize,
        event_timeout: Duration,
    ) -> Result<Self> {
        ensure!(slot_count > 0 && chunk_bytes > 0, "empty pinned copy pool");
        let stream = gpu.default_stream();
        ensure!(
            !gpu.stream_is_capturing(stream),
            "fast weight loading refuses CUDA graph capture"
        );
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(Slot {
                buffer: Some(gpu.alloc_host_pinned(chunk_bytes)?),
                flight: Flight::Idle,
            });
        }
        Ok(Self {
            gpu,
            slots,
            stream,
            chunk_bytes,
            event_timeout,
            poisoned: false,
        })
    }

    pub(super) fn copy(&mut self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.copy_into_owned_range(src, dst, 0)
    }

    /// Copy into a view while retaining its exact owning allocation, not the
    /// interior destination. The caller validates the complete view's group
    /// membership and bounds before copying.
    pub(super) fn copy_into_owned_range(
        &mut self,
        src: &[u8],
        owning_base: DevicePtr,
        view_offset: usize,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "pinned H2D copier is poisoned; no retry is safe"
        );
        // Set before any backend call so a caught panic also forbids reuse.
        self.poisoned = true;
        let result = self.copy_once(src, owning_base, view_offset);
        if result.is_ok() {
            self.poisoned = false;
        }
        result
    }

    fn copy_once(&mut self, src: &[u8], owning_base: DevicePtr, view_offset: usize) -> Result<()> {
        // Reject address arithmetic failures before submitting any prefix.
        let end = view_offset
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("pinned H2D destination range overflow"))?;
        owning_base
            .0
            .checked_add(u64::try_from(end)?)
            .ok_or_else(|| anyhow::anyhow!("pinned H2D destination overflow"))?;
        let mut offset = 0usize;
        let mut slot_idx = 0usize;
        while offset < src.len() {
            let len = (src.len() - offset).min(self.chunk_bytes);
            let destination_offset = view_offset
                .checked_add(offset)
                .ok_or_else(|| anyhow::anyhow!("pinned H2D destination range overflow"))?;
            let chunk_dst = DevicePtr(
                owning_base
                    .0
                    .checked_add(u64::try_from(destination_offset)?)
                    .ok_or_else(|| anyhow::anyhow!("pinned H2D destination overflow"))?,
            );
            let slot = &mut self.slots[slot_idx];
            wait_slot_bounded(self.gpu, slot, Instant::now() + self.event_timeout)?;
            let buffer = slot.buffer.as_mut().expect("live slot owns its buffer");
            buffer.as_mut_slice()[..len].copy_from_slice(&src[offset..offset + len]);
            let pinned = buffer.pinned_slice(len)?;

            let event = self.gpu.create_event()?;
            if let Err(error) =
                self.gpu
                    .retain_h2d_destination(owning_base, destination_offset, len)
            {
                // Nothing was submitted with this event, so destruction is safe.
                let _ = self.gpu.destroy_event(event);
                return Err(error);
            }
            slot.flight = Flight::Untracked {
                event,
                destination: owning_base,
            };
            // SAFETY: (1) Destination retention (`retain_h2d_destination`) and `Flight::Untracked`
            // state precede submission, so `owning_base` cannot be freed while DMA is in flight;
            // (2) Host lifetime: `pinned` borrows `slot.buffer`, which is never overwritten until
            // `wait_slot_bounded` observes positive event completion, and on error, timeout, or
            // unwind `Slot::drop` calls `std::mem::forget(buffer)` whenever `flight != Flight::Idle`
            // so the pinned host allocation stays valid for any uncertain in-flight transfer.
            unsafe {
                self.gpu
                    .copy_h2d_pinned_async(pinned, chunk_dst, self.stream)?;
            }
            self.gpu.record_event(event, self.stream)?;
            slot.flight = Flight::Recorded {
                event,
                destination: owning_base,
            };
            offset += len;
            slot_idx = (slot_idx + 1) % self.slots.len();
        }
        Ok(())
    }

    pub(super) fn drain(&mut self) -> Result<()> {
        let was_poisoned = self.poisoned;
        self.poisoned = true;
        let deadline = Instant::now() + self.event_timeout;
        let mut first_error = None;
        // Attempt every slot, with one shared deadline. One failed event must
        // not discard ownership of later slots or prevent their cleanup.
        for slot in &mut self.slots {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                wait_slot_bounded(self.gpu, slot, deadline)
            }))
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "backend panicked while draining pinned H2D"
                ))
            });
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        self.poisoned = was_poisoned;
        Ok(())
    }
}

impl Drop for PinnedBounceCopier<'_> {
    fn drop(&mut self) {
        // A backend panic during error unwind must not trigger a second panic.
        // Slot::drop independently retains any still-uncertain host owner.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.drain()));
    }
}

fn wait_slot_bounded(gpu: &dyn GpuBackend, slot: &mut Slot, deadline: Instant) -> Result<()> {
    let (event, destination) = match slot.flight {
        Flight::Idle => return Ok(()),
        Flight::Untracked { event, destination } => bail!(
            "H2D completion tracking uncertain for event {event:#x}, destination {destination}; resources retained"
        ),
        Flight::Recorded { event, destination } => (event, destination),
    };
    let mut polls = 0u32;
    loop {
        if gpu.poll_event(event)? {
            // Positive completion, not a timeout/error or an unrecorded event,
            // is the only permission to release the source and device lease.
            slot.flight = Flight::Idle;
            gpu.release_h2d_destination(destination)?;
            gpu.destroy_event(event)?;
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("H2D event {event:#x} timed out; source/event/destination retained");
        }
        polls = polls.saturating_add(1);
        if polls <= 64 {
            std::hint::spin_loop();
        } else if polls <= 256 {
            std::thread::sleep(Duration::from_micros(25));
        } else {
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

#[cfg(test)]
#[path = "bounce_copy_tests.rs"]
mod tests;
