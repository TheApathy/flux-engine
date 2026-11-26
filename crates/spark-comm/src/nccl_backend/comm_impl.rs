// SPDX-License-Identifier: AGPL-3.0-only

//! `impl CommBackend for NcclBackend` — collective operations.
//!
//! Every `unsafe` block here carries a per-site `// SAFETY:` justification.
//! All NCCL communicator calls borrow `&OwnedNcclComm` while holding
//! `self.comm.lock()`, ensuring serialized access and preventing UAF across
//! `ncclCommAbort`/`Drop`. ABI targets NCCL 2.28+ (`2.24+` baseline).

use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::{
    COLLECTIVE_TIMEOUT_SECS, NcclBackend, OwnedNcclComm, P2pAllReduceOverride, memcpy_d2d_async,
    take_tracked_handle,
};
use crate::nccl::{self, NcclDataType, NcclRedOp};
use crate::{CollectiveDtype, CollectiveToken, CommBackend, RoutePacketLayout};

/// Maximum per-rank payload size (256 KiB) for the low-latency 3-rank (1-RTT)
/// and 4-rank (2-RTT bidirectional half-ring) route-gather P2P relay.
/// Larger payloads fall back to `ncclAllGather`.
pub(crate) const SMALL_ROUTE_RELAY_MAX_BYTES: usize = 256 * 1024;

/// Returns `true` when a synchronous `broadcast` wall-clock duration indicates
/// a collective stall rather than a worker rank waiting idly for the next
/// command from `root`.
///
/// Worker ranks (`rank != root`) legitimately block inside `ep_broadcast_u32`
/// between client HTTP requests for arbitrarily long idle intervals; flagging
/// them unhealthy after `COLLECTIVE_TIMEOUT_SECS` would falsely poison an idle
/// ConnectX-7 cluster. Real transport faults on any rank are still caught by
/// `check_async_error`.
#[must_use]
pub(crate) const fn should_flag_broadcast_timeout(
    rank: usize,
    root: usize,
    elapsed_secs: u64,
) -> bool {
    rank == root && elapsed_secs >= COLLECTIVE_TIMEOUT_SECS
}

/// Map a typed collective byte count and [`CollectiveDtype`] to `(NcclDataType, element_count)`.
///
/// Rejects byte counts that are not an exact multiple of `dtype.element_bytes()`.
pub(crate) fn nccl_dtype_and_count(
    bytes: usize,
    dtype: CollectiveDtype,
) -> Result<(NcclDataType, usize)> {
    let elem_bytes = dtype.element_bytes();
    anyhow::ensure!(
        bytes.is_multiple_of(elem_bytes),
        "reduce_scatter_typed: byte count {bytes} is not a multiple of {:?} element size ({elem_bytes})",
        dtype
    );
    let nccl_dtype = match dtype {
        CollectiveDtype::Bf16 => NcclDataType::Bfloat16,
        CollectiveDtype::Fp32 => NcclDataType::Float32,
        CollectiveDtype::Uint8 => NcclDataType::Uint8,
    };
    Ok((nccl_dtype, bytes / elem_bytes))
}

trait RouteGatherTransport {
    fn memcpy_d2d_async(&mut self, dst: u64, src: u64, bytes: usize, stream: u64) -> Result<()>;
    fn group_start(&mut self) -> Result<()>;
    fn group_end(&mut self) -> Result<()>;
    fn send(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()>;
    fn recv(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()>;
    fn all_gather(&mut self, send_ptr: u64, recv_ptr: u64, bytes: usize, stream: u64)
    -> Result<()>;
}

struct NcclRouteGatherTransport<'a> {
    comm: &'a OwnedNcclComm,
}

impl RouteGatherTransport for NcclRouteGatherTransport<'_> {
    fn memcpy_d2d_async(&mut self, dst: u64, src: u64, bytes: usize, stream: u64) -> Result<()> {
        // SAFETY: Caller supplies valid device route buffers covering at least `bytes`
        // bytes and a live CUDA stream on this rank's device.
        unsafe { memcpy_d2d_async(dst, src, bytes, stream) }
    }

    fn group_start(&mut self) -> Result<()> {
        // SAFETY: Begins a thread-local NCCL group closed by `run_group`'s `group_end`.
        nccl::check_nccl(unsafe { nccl::ncclGroupStart() }, "ncclGroupStart")
    }

    fn group_end(&mut self) -> Result<()> {
        // SAFETY: Closes the matching `ncclGroupStart` opened by `run_group`.
        nccl::check_nccl(unsafe { nccl::ncclGroupEnd() }, "ncclGroupEnd")
    }

    fn send(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()> {
        // SAFETY: `self.comm` borrows a live `OwnedNcclComm` under `self.comm.lock()`,
        // `ptr` spans at least `bytes` readable device bytes, and `stream` is live.
        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                bytes,
                NcclDataType::Uint8,
                peer as i32,
                self.comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend (gather_selected_routes)")
    }

    fn recv(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()> {
        // SAFETY: `self.comm` borrows a live `OwnedNcclComm` under `self.comm.lock()`,
        // `ptr` spans at least `bytes` writable device bytes, and `stream` is live.
        let result = unsafe {
            nccl::ncclRecv(
                ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                peer as i32,
                self.comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv (gather_selected_routes)")
    }

    fn all_gather(
        &mut self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // SAFETY: `self.comm` borrows a live `OwnedNcclComm` under `self.comm.lock()`,
        // `send_ptr`/`recv_ptr` span `bytes` / `world_size * bytes` device bytes, and `stream` is live.
        let result = unsafe {
            nccl::ncclAllGather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                self.comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclAllGather (gather_selected_routes)")
    }
}

fn run_group<T, F>(transport: &mut T, body: F) -> Result<()>
where
    T: RouteGatherTransport,
    F: FnOnce(&mut T) -> Result<()>,
{
    transport.group_start()?;
    let body_res = body(transport);
    let end_res = transport.group_end();
    body_res.and(end_res)
}

fn slot_ptr(base: u64, rank_idx: usize, rank_bytes: usize) -> Result<u64> {
    let byte_off = rank_idx
        .checked_mul(rank_bytes)
        .ok_or_else(|| anyhow::anyhow!("route gather slot byte offset overflow"))?;
    let byte_off_u64 = u64::try_from(byte_off)
        .map_err(|_| anyhow::anyhow!("route gather slot byte offset exceeds u64"))?;
    base.checked_add(byte_off_u64)
        .ok_or_else(|| anyhow::anyhow!("route gather slot pointer overflow"))
}

struct RouteGatherParams {
    rank: usize,
    world_size: usize,
    send_routes_ptr: u64,
    recv_routes_ptr: u64,
    local_tokens: usize,
    layout: RoutePacketLayout,
    stream: u64,
}

fn execute_route_gather<T: RouteGatherTransport>(
    transport: &mut T,
    params: &RouteGatherParams,
) -> Result<()> {
    let RouteGatherParams {
        rank,
        world_size,
        send_routes_ptr,
        recv_routes_ptr,
        local_tokens,
        layout,
        stream,
    } = *params;
    if local_tokens == 0 || world_size <= 1 {
        return Ok(());
    }
    anyhow::ensure!(
        rank < world_size,
        "gather_selected_routes_with_layout: rank {rank} >= world_size {world_size}"
    );
    let rank_bytes = local_tokens
        .checked_mul(layout.packet_bytes())
        .ok_or_else(|| anyhow::anyhow!("SelectedRoute gather byte count overflow"))?;

    if world_size > 4 || (world_size >= 3 && rank_bytes > SMALL_ROUTE_RELAY_MAX_BYTES) {
        return transport.all_gather(send_routes_ptr, recv_routes_ptr, rank_bytes, stream);
    }

    // Place local rank's slice directly without ever issuing self-send/self-recv.
    // When the caller packed in-place (`send_routes_ptr == slot(rank)`), skip D2D copy.
    let self_slot = slot_ptr(recv_routes_ptr, rank, rank_bytes)?;
    if send_routes_ptr != self_slot {
        transport.memcpy_d2d_async(self_slot, send_routes_ptr, rank_bytes, stream)?;
    }

    match world_size {
        2 => {
            let peer = 1 - rank;
            let peer_slot = slot_ptr(recv_routes_ptr, peer, rank_bytes)?;
            run_group(transport, |t| {
                t.send(send_routes_ptr, rank_bytes, peer, stream)?;
                t.recv(peer_slot, rank_bytes, peer, stream)
            })
        }
        3 => {
            // 1-RTT simultaneous bidirectional all-to-all: Send(cw)+Recv(ccw)+Send(ccw)+Recv(cw).
            let cw = (rank + 1) % 3;
            let ccw = (rank + 2) % 3;
            let cw_slot = slot_ptr(recv_routes_ptr, cw, rank_bytes)?;
            let ccw_slot = slot_ptr(recv_routes_ptr, ccw, rank_bytes)?;
            run_group(transport, |t| {
                t.send(send_routes_ptr, rank_bytes, cw, stream)?;
                t.recv(ccw_slot, rank_bytes, ccw, stream)?;
                t.send(send_routes_ptr, rank_bytes, ccw, stream)?;
                t.recv(cw_slot, rank_bytes, cw, stream)
            })
        }
        4 => {
            // 2-RTT bidirectional half-ring utilizing both ConnectX-7 ports full-duplex:
            //   Round 1: exchange own slice with cw and ccw neighbors.
            //   Round 2: forward lower half of ccw's slice along cw and upper half of cw's
            //            slice along ccw into disjoint halves of opp's slot.
            let cw = (rank + 1) % 4;
            let opp = (rank + 2) % 4;
            let ccw = (rank + 3) % 4;
            let cw_slot = slot_ptr(recv_routes_ptr, cw, rank_bytes)?;
            let ccw_slot = slot_ptr(recv_routes_ptr, ccw, rank_bytes)?;
            let opp_slot = slot_ptr(recv_routes_ptr, opp, rank_bytes)?;

            run_group(transport, |t| {
                t.send(send_routes_ptr, rank_bytes, cw, stream)?;
                t.recv(ccw_slot, rank_bytes, ccw, stream)?;
                t.send(send_routes_ptr, rank_bytes, ccw, stream)?;
                t.recv(cw_slot, rank_bytes, cw, stream)
            })?;

            let half_lo = rank_bytes / 2;
            let half_hi = rank_bytes - half_lo;
            let half_lo_u64 = u64::try_from(half_lo)
                .map_err(|_| anyhow::anyhow!("route gather half offset exceeds u64"))?;
            let cw_hi_src = cw_slot
                .checked_add(half_lo_u64)
                .ok_or_else(|| anyhow::anyhow!("route gather cw upper-half pointer overflow"))?;
            let opp_hi_dst = opp_slot
                .checked_add(half_lo_u64)
                .ok_or_else(|| anyhow::anyhow!("route gather opp upper-half pointer overflow"))?;

            run_group(transport, |t| {
                if half_lo > 0 {
                    t.send(ccw_slot, half_lo, cw, stream)?;
                    t.recv(opp_slot, half_lo, ccw, stream)?;
                }
                if half_hi > 0 {
                    t.send(cw_hi_src, half_hi, ccw, stream)?;
                    t.recv(opp_hi_dst, half_hi, cw, stream)?;
                }
                Ok(())
            })
        }
        _ => unreachable!("world_size outside 2..=4 handled above"),
    }
}

trait SplitAllReduceOps {
    fn record_event(&mut self, event: u64, stream: u64) -> Result<()>;
    fn stream_wait_event(&mut self, stream: u64, event: u64) -> Result<()>;
    fn launch_all_reduce(&mut self, ptr: u64, bytes: usize, comm_stream: u64) -> Result<()>;
}

struct NcclSplitAllReduceOps<'a> {
    backend: &'a NcclBackend,
    comm: Option<&'a OwnedNcclComm>,
}

impl SplitAllReduceOps for NcclSplitAllReduceOps<'_> {
    fn record_event(&mut self, event: u64, stream: u64) -> Result<()> {
        // SAFETY: `event` and `stream` belong to `NcclBackend` / caller's live CUDA stream.
        unsafe { nccl::record_event(event, stream) }
    }

    fn stream_wait_event(&mut self, stream: u64, event: u64) -> Result<()> {
        // SAFETY: `stream` and `event` are live CUDA handles owned by this rank.
        unsafe { nccl::stream_wait_event(stream, event) }
    }

    fn launch_all_reduce(&mut self, ptr: u64, bytes: usize, comm_stream: u64) -> Result<()> {
        let comm = self.comm.ok_or_else(|| {
            anyhow::anyhow!("NCCL communicator guard missing in launch_all_reduce")
        })?;
        if self.backend.use_roce_send_recv_allreduce(bytes) {
            return self
                .backend
                .all_reduce_2rank_locked(comm, ptr, bytes, comm_stream);
        }
        let count = bytes / 2;
        // SAFETY: `comm` is borrowed under `self.backend.comm.lock()`, `ptr` spans at least
        // `bytes` (`count` BF16 elements) on this rank's device, and `comm_stream` is live.
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm.as_raw(),
                comm_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce (begin)")?;
        self.backend.check_async_error(comm);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct SplitAllReduceConfig {
    compute_stream: u64,
    comm_stream: u64,
    compute_done_event: u64,
    comm_done_event: u64,
    world_size: usize,
}

fn execute_all_reduce_begin<O: SplitAllReduceOps>(
    ops: &mut O,
    ptr: u64,
    bytes: usize,
    cfg: SplitAllReduceConfig,
) -> Result<CollectiveToken> {
    anyhow::ensure!(
        bytes.is_multiple_of(2),
        "all_reduce_begin: BF16 byte count {bytes} must be even"
    );
    if bytes == 0 || cfg.world_size <= 1 {
        return Ok(CollectiveToken::noop());
    }
    // 1. Record "compute done" on compute stream.
    ops.record_event(cfg.compute_done_event, cfg.compute_stream)?;
    // 2. Comm stream waits for compute to finish.
    ops.stream_wait_event(cfg.comm_stream, cfg.compute_done_event)?;
    // 3. Launch all-reduce on dedicated comm stream.
    ops.launch_all_reduce(ptr, bytes, cfg.comm_stream)?;
    // 4. Record "comm done" on comm stream — compute_stream is NOT stalled here!
    ops.record_event(cfg.comm_done_event, cfg.comm_stream)?;
    Ok(CollectiveToken {
        comm_done_event: cfg.comm_done_event,
    })
}

fn execute_all_reduce_end<O: SplitAllReduceOps>(
    ops: &mut O,
    token: CollectiveToken,
    compute_stream: u64,
) -> Result<()> {
    if token.comm_done_event == 0 {
        return Ok(());
    }
    // 5. Compute stream waits for all-reduce completion before consuming reduced buffer.
    ops.stream_wait_event(compute_stream, token.comm_done_event)
}

impl CommBackend for NcclBackend {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        anyhow::ensure!(
            bytes.is_multiple_of(2),
            "all_reduce: BF16 byte count {bytes} must be even"
        );
        if bytes == 0 || self.world_size <= 1 {
            return Ok(());
        }
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "all_reduce")?;
        if self.use_roce_send_recv_allreduce(bytes) {
            return self.all_reduce_2rank_locked(comm, ptr, bytes, self.legacy_stream);
        }
        let count = bytes / 2;
        // SAFETY: `comm` is borrowed under `comm_guard`, `ptr` spans at least `bytes`
        // (`count` BF16 elements) on this rank's device, and `self.legacy_stream` is live.
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm.as_raw(),
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn all_reduce_begin(
        &self,
        ptr: u64,
        bytes: usize,
        compute_stream: u64,
    ) -> Result<CollectiveToken> {
        anyhow::ensure!(
            bytes.is_multiple_of(2),
            "all_reduce_begin: BF16 byte count {bytes} must be even"
        );
        if bytes == 0 || self.world_size <= 1 {
            return Ok(CollectiveToken::noop());
        }
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "all_reduce_begin")?;
        let mut ops = NcclSplitAllReduceOps {
            backend: self,
            comm: Some(comm),
        };
        execute_all_reduce_begin(
            &mut ops,
            ptr,
            bytes,
            SplitAllReduceConfig {
                compute_stream,
                comm_stream: self.comm_stream,
                compute_done_event: self.compute_done_event,
                comm_done_event: self.comm_done_event,
                world_size: self.world_size,
            },
        )
    }

    fn all_reduce_end(&self, token: CollectiveToken, compute_stream: u64) -> Result<()> {
        let mut ops = NcclSplitAllReduceOps {
            backend: self,
            comm: None,
        };
        execute_all_reduce_end(&mut ops, token, compute_stream)
    }

    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let tok = self.all_reduce_begin(ptr, bytes, compute_stream)?;
        self.all_reduce_end(tok, compute_stream)
    }

    fn register_buffer(&self, ptr: u64, bytes: usize) -> Result<u64> {
        let mut handle: *mut c_void = ptr::null_mut();
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "register_buffer")?;
        // SAFETY: `comm` is borrowed under `comm_guard`, `ptr` spans at least `bytes`
        // valid device bytes on this rank's device, and `&mut handle` is a valid out-pointer.
        let result = unsafe {
            nccl::ncclCommRegister(comm.as_raw(), ptr as *mut c_void, bytes, &mut handle)
        };
        nccl::check_nccl(result, "ncclCommRegister")?;
        let handle_u64 = handle as u64;
        if handle_u64 != 0 {
            self.registered_handles.lock().insert(handle_u64);
        }
        Ok(handle_u64)
    }

    fn deregister_buffer(&self, handle: u64) -> Result<()> {
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "deregister_buffer")?;
        take_tracked_handle(&self.registered_handles, handle, "deregister_buffer")?;
        // SAFETY: `comm` is borrowed under `comm_guard` and `handle` was validated
        // and removed from `self.registered_handles` (guaranteed live on this comm and
        // not previously deregistered).
        let result = unsafe { nccl::ncclCommDeregister(comm.as_raw(), handle as *mut c_void) };
        nccl::check_nccl(result, "ncclCommDeregister")?;
        Ok(())
    }

    fn symmetric_alloc(&self, bytes: usize) -> Result<u64> {
        // SAFETY: Returned pointer is recorded in `self.symmetric_allocs` so
        // `symmetric_free` (or `Drop`) releases it at most once via `nccl_mem_free`.
        let ptr = unsafe { nccl::nccl_mem_alloc(bytes) }?;
        let ptr_u64 = ptr as u64;
        anyhow::ensure!(ptr_u64 != 0, "ncclMemAlloc returned a null device pointer");
        self.symmetric_allocs.lock().insert(ptr_u64);
        Ok(ptr_u64)
    }

    fn symmetric_free(&self, ptr: u64) -> Result<()> {
        take_tracked_handle(&self.symmetric_allocs, ptr, "symmetric_free")?;
        // SAFETY: `ptr` was validated and removed from `self.symmetric_allocs`,
        // guaranteeing it came from `nccl_mem_alloc` and has not yet been freed.
        unsafe { nccl::nccl_mem_free(ptr as *mut c_void) }
    }

    fn set_add_kernel(&self, handle: u64) {
        self.add_kernel.store(handle, Ordering::Relaxed);
        if self.world_size == 2
            && (!self.locality.is_single_host()
                || matches!(self.p2p_override, P2pAllReduceOverride::ForceP2p))
        {
            tracing::info!(
                "NCCL backend: bf16_add_inplace kernel set (2-rank RoCE/P2P send/recv enabled)"
            );
        } else {
            tracing::info!(
                "NCCL backend: bf16_add_inplace kernel registered \
                 (native ncclAllReduce active; locality={:?}, world_size={})",
                self.locality,
                self.world_size
            );
        }
    }

    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "all_gather")?;
        // SAFETY: `comm` is borrowed under `comm_guard`, `send_ptr`/`recv_ptr` span
        // `bytes` / `world_size * bytes` device bytes, and `self.legacy_stream` is live.
        let result = unsafe {
            nccl::ncclAllGather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                comm.as_raw(),
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllGather")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn reduce_scatter_typed(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        dtype: CollectiveDtype,
        stream: u64,
    ) -> Result<()> {
        let (nccl_dtype, count) = nccl_dtype_and_count(bytes, dtype)?;
        if count == 0 || self.world_size <= 1 {
            return Ok(());
        }
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "reduce_scatter_typed")?;
        // SAFETY: `comm` is borrowed under `comm_guard`, `send_ptr`/`recv_ptr` span
        // `world_size * bytes` / `bytes` (`count` elements of `nccl_dtype`), and `stream` is live.
        let result = unsafe {
            nccl::ncclReduceScatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                count,
                nccl_dtype,
                NcclRedOp::Sum,
                comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclReduceScatter")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        self.reduce_scatter_typed(
            send_ptr,
            recv_ptr,
            bytes,
            CollectiveDtype::Bf16,
            self.legacy_stream,
        )
    }

    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        let start = Instant::now();
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "broadcast")?;

        // Broadcast raw bytes as Uint8
        // SAFETY: `comm` is borrowed under `comm_guard`, `ptr` spans at least `bytes`
        // device bytes on this rank's device, and `self.legacy_stream` is live.
        let result = unsafe {
            nccl::ncclBroadcast(
                ptr as *const _,
                ptr as *mut _,
                bytes,
                NcclDataType::Uint8,
                root as i32,
                comm.as_raw(),
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclBroadcast")?;

        // Synchronize to measure wall-clock time for timeout detection.
        // SAFETY: `self.legacy_stream` is a live CUDA stream owned by this rank.
        unsafe { nccl::sync_stream(self.legacy_stream)? };

        let elapsed = start.elapsed();
        if should_flag_broadcast_timeout(self.rank, root, elapsed.as_secs()) {
            tracing::error!(
                "NCCL broadcast from root {root} took {:.1}s (threshold: {}s) \
                 — marking communicator unhealthy",
                elapsed.as_secs_f64(),
                COLLECTIVE_TIMEOUT_SECS,
            );
            self.unhealthy.store(true, Ordering::Release);
        }

        // Also check for async errors.
        self.check_async_error(comm);

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size <= 1 {
            return Ok(());
        }
        anyhow::ensure!(
            self.barrier_buf != 0,
            "NCCL barrier scratch buffer is not allocated"
        );
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "barrier")?;
        let buf = self.barrier_buf as *mut c_void;
        // SAFETY: `comm` is borrowed under `comm_guard`, `self.barrier_buf` is an owned
        // 4-byte zeroed device allocation on this rank's device serialized by `comm_guard`,
        // and `self.legacy_stream` is a live CUDA stream.
        let result = unsafe {
            nccl::ncclAllReduce(
                buf as *const c_void,
                buf,
                1,
                NcclDataType::Int32,
                NcclRedOp::Sum,
                comm.as_raw(),
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "barrier (ncclAllReduce 1xInt32)")?;
        // Block the host until all ranks complete the 1-word AllReduce.
        // SAFETY: `self.legacy_stream` is a live CUDA stream owned by this rank.
        unsafe { nccl::sync_stream(self.legacy_stream)? };
        self.check_async_error(comm);
        Ok(())
    }

    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()> {
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "send_to")?;
        // SAFETY: `comm` is borrowed under `comm_guard`, `ptr` spans at least `bytes`
        // readable device bytes, and `stream` is live.
        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                bytes,
                NcclDataType::Uint8,
                dest_rank as i32,
                comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend (send_to)")
    }

    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()> {
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "recv_from")?;
        // SAFETY: `comm` is borrowed under `comm_guard`, `ptr` spans at least `bytes`
        // writable device bytes, and `stream` is live.
        let result = unsafe {
            nccl::ncclRecv(
                ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                src_rank as i32,
                comm.as_raw(),
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv (recv_from)")
    }

    fn gather_selected_routes_with_layout(
        &self,
        send_routes_ptr: u64,
        recv_routes_ptr: u64,
        local_tokens: usize,
        layout: RoutePacketLayout,
        stream: u64,
    ) -> Result<()> {
        if local_tokens == 0 || self.world_size <= 1 {
            return Ok(());
        }
        let comm_guard = self.comm.lock();
        let comm = Self::require_comm(&comm_guard, "gather_selected_routes_with_layout")?;
        let mut transport = NcclRouteGatherTransport { comm };
        execute_route_gather(
            &mut transport,
            &RouteGatherParams {
                rank: self.rank,
                world_size: self.world_size,
                send_routes_ptr,
                recv_routes_ptr,
                local_tokens,
                layout,
                stream,
            },
        )?;
        self.check_async_error(comm);
        Ok(())
    }

    fn group_start(&self) -> Result<()> {
        // SAFETY: Begins a thread-local NCCL group matched by the caller's `group_end`.
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")
    }

    fn group_end(&self) -> Result<()> {
        // SAFETY: Closes the thread-local NCCL group opened by `group_start`.
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")
    }

    fn is_healthy(&self) -> bool {
        if self.unhealthy.load(Ordering::Acquire) {
            return false;
        }
        // Actively probe the communicator for async errors while holding the lock.
        let comm_guard = self.comm.lock();
        let Some(comm) = comm_guard.as_ref() else {
            return false;
        };
        self.check_async_error(comm)
    }

    fn attempt_reconnect(&self) -> Result<()> {
        self.reconnect_inner()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::super::RECV_BUFFER_SIZE;
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum RecordedOp {
        Memcpy {
            dst: u64,
            src: u64,
            bytes: usize,
            stream: u64,
        },
        GroupStart,
        GroupEnd,
        Send {
            ptr: u64,
            bytes: usize,
            peer: usize,
            stream: u64,
        },
        Recv {
            ptr: u64,
            bytes: usize,
            peer: usize,
            stream: u64,
        },
        AllGather {
            send_ptr: u64,
            recv_ptr: u64,
            bytes: usize,
            stream: u64,
        },
    }

    struct MockRouteTransport {
        ops: Vec<RecordedOp>,
        group_depth: isize,
        fail_on_p2p_idx: Option<usize>,
        p2p_count: usize,
    }

    impl MockRouteTransport {
        fn new(fail_on_p2p_idx: Option<usize>) -> Self {
            Self {
                ops: Vec::new(),
                group_depth: 0,
                fail_on_p2p_idx,
                p2p_count: 0,
            }
        }
    }

    impl RouteGatherTransport for MockRouteTransport {
        fn memcpy_d2d_async(
            &mut self,
            dst: u64,
            src: u64,
            bytes: usize,
            stream: u64,
        ) -> Result<()> {
            self.ops.push(RecordedOp::Memcpy {
                dst,
                src,
                bytes,
                stream,
            });
            Ok(())
        }

        fn group_start(&mut self) -> Result<()> {
            self.group_depth += 1;
            self.ops.push(RecordedOp::GroupStart);
            Ok(())
        }

        fn group_end(&mut self) -> Result<()> {
            self.group_depth -= 1;
            self.ops.push(RecordedOp::GroupEnd);
            Ok(())
        }

        fn send(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()> {
            let idx = self.p2p_count;
            self.p2p_count += 1;
            if self.fail_on_p2p_idx == Some(idx) {
                anyhow::bail!("injected send failure at op {idx}");
            }
            self.ops.push(RecordedOp::Send {
                ptr,
                bytes,
                peer,
                stream,
            });
            Ok(())
        }

        fn recv(&mut self, ptr: u64, bytes: usize, peer: usize, stream: u64) -> Result<()> {
            let idx = self.p2p_count;
            self.p2p_count += 1;
            if self.fail_on_p2p_idx == Some(idx) {
                anyhow::bail!("injected recv failure at op {idx}");
            }
            self.ops.push(RecordedOp::Recv {
                ptr,
                bytes,
                peer,
                stream,
            });
            Ok(())
        }

        fn all_gather(
            &mut self,
            send_ptr: u64,
            recv_ptr: u64,
            bytes: usize,
            stream: u64,
        ) -> Result<()> {
            self.ops.push(RecordedOp::AllGather {
                send_ptr,
                recv_ptr,
                bytes,
                stream,
            });
            Ok(())
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_collective_timeout_constant() {
        // Sanity: timeout should be a reasonable value (not 0, not too large).
        assert!(COLLECTIVE_TIMEOUT_SECS >= 10);
        assert!(COLLECTIVE_TIMEOUT_SECS <= 300);
    }

    #[test]
    fn test_recv_buffer_size_sufficient() {
        // Verify buffer covers all use cases including large models.
        let single_decode = 6144 * 2; // 12 KB (122B hidden)
        let k3_verify = 6144 * 3 * 2; // 36 KB
        let prefill_chunk = 6144 * 4096 * 2; // 48 MB (122B, 4096 chunk)
        assert!(RECV_BUFFER_SIZE >= single_decode);
        assert!(RECV_BUFFER_SIZE >= k3_verify);
        assert!(RECV_BUFFER_SIZE >= prefill_chunk);
    }

    #[test]
    fn test_nccl_dtype_and_count_mapping_and_alignment() {
        let (bf16_ty, bf16_cnt) =
            nccl_dtype_and_count(1024, CollectiveDtype::Bf16).expect("1024B valid Bf16");
        assert!(matches!(bf16_ty, NcclDataType::Bfloat16));
        assert_eq!(bf16_cnt, 512);

        let (fp32_ty, fp32_cnt) =
            nccl_dtype_and_count(1024, CollectiveDtype::Fp32).expect("1024B valid Fp32");
        assert!(matches!(fp32_ty, NcclDataType::Float32));
        assert_eq!(fp32_cnt, 256);

        let (u8_ty, u8_cnt) =
            nccl_dtype_and_count(1023, CollectiveDtype::Uint8).expect("1023B valid Uint8");
        assert!(matches!(u8_ty, NcclDataType::Uint8));
        assert_eq!(u8_cnt, 1023);

        assert!(nccl_dtype_and_count(3, CollectiveDtype::Bf16).is_err());
        assert!(nccl_dtype_and_count(6, CollectiveDtype::Fp32).is_err());
    }

    #[test]
    fn test_route_gather_zero_self_sends_and_inplace_skip_2_3_4_ranks() {
        for world_size in [2usize, 3, 4] {
            for layout in [
                RoutePacketLayout::TopK6,
                RoutePacketLayout::TopK8,
                RoutePacketLayout::TopK10,
            ] {
                let local_tokens = 4usize;
                let rank_bytes = local_tokens * layout.packet_bytes();
                let recv_base = 0x10_0000_u64;

                for rank in 0..world_size {
                    // Out-of-place: issues 1 D2D self-copy and 0 self-sends.
                    let send_oop = 0x01_0000_u64;
                    let mut oop = MockRouteTransport::new(None);
                    execute_route_gather(
                        &mut oop,
                        &RouteGatherParams {
                            rank,
                            world_size,
                            send_routes_ptr: send_oop,
                            recv_routes_ptr: recv_base,
                            local_tokens,
                            layout,
                            stream: 99,
                        },
                    )
                    .expect("out-of-place route gather must succeed");
                    assert_eq!(oop.group_depth, 0);

                    let memcpy_count = oop
                        .ops
                        .iter()
                        .filter(|op| matches!(op, RecordedOp::Memcpy { .. }))
                        .count();
                    assert_eq!(memcpy_count, 1, "out-of-place must issue 1 D2D self-copy");

                    let group_starts = oop
                        .ops
                        .iter()
                        .filter(|op| matches!(op, RecordedOp::GroupStart))
                        .count();
                    let expected_groups = if world_size == 4 { 2 } else { 1 };
                    assert_eq!(group_starts, expected_groups);

                    for op in &oop.ops {
                        if let RecordedOp::Send { peer, .. } | RecordedOp::Recv { peer, .. } = op {
                            assert_ne!(
                                *peer, rank,
                                "self-send/self-recv forbidden (world={world_size}, rank={rank})"
                            );
                        }
                    }

                    // In-place: send_routes_ptr == recv_base + rank * rank_bytes skips D2D copy.
                    let send_inplace = recv_base + (rank * rank_bytes) as u64;
                    let mut inp = MockRouteTransport::new(None);
                    execute_route_gather(
                        &mut inp,
                        &RouteGatherParams {
                            rank,
                            world_size,
                            send_routes_ptr: send_inplace,
                            recv_routes_ptr: recv_base,
                            local_tokens,
                            layout,
                            stream: 99,
                        },
                    )
                    .expect("in-place route gather must succeed");
                    let inp_memcpy = inp
                        .ops
                        .iter()
                        .filter(|op| matches!(op, RecordedOp::Memcpy { .. }))
                        .count();
                    assert_eq!(inp_memcpy, 0, "in-place gather must skip D2D self-copy");
                }
            }
        }

        // Large payload (> 256 KiB per rank) on 3/4 ranks falls back to AllGather.
        let mut large = MockRouteTransport::new(None);
        execute_route_gather(
            &mut large,
            &RouteGatherParams {
                rank: 1,
                world_size: 4,
                send_routes_ptr: 0x1000,
                recv_routes_ptr: 0x20_0000,
                local_tokens: 4096, // 4096 * 80B = 320 KiB > 256 KiB
                layout: RoutePacketLayout::TopK10,
                stream: 7,
            },
        )
        .expect("large 4-rank gather must fall back to all_gather");
        assert_eq!(
            large.ops,
            vec![RecordedOp::AllGather {
                send_ptr: 0x1000,
                recv_ptr: 0x20_0000,
                bytes: 4096 * 80,
                stream: 7,
            }]
        );
    }

    #[test]
    fn test_route_gather_multi_rank_byte_exact_2_3_4_ranks() {
        for world_size in [2usize, 3, 4] {
            for layout in [
                RoutePacketLayout::TopK6,
                RoutePacketLayout::TopK8,
                RoutePacketLayout::TopK10,
            ] {
                let local_tokens = 3usize;
                let rank_bytes = local_tokens * layout.packet_bytes();
                let send_base = 0usize;
                let recv_base = rank_bytes;
                let total_mem = recv_base + world_size * rank_bytes;

                // Seed each rank with unique deterministic payload bytes.
                let mut mem: Vec<Vec<u8>> = (0..world_size)
                    .map(|r| {
                        let mut buf = vec![0u8; total_mem];
                        for (i, slot) in buf[send_base..send_base + rank_bytes]
                            .iter_mut()
                            .enumerate()
                        {
                            *slot = ((r * 37 + i * 13 + 7) & 0xFF) as u8;
                        }
                        buf
                    })
                    .collect();

                // Collect each rank's recorded plan.
                let mut plans: Vec<Vec<RecordedOp>> = Vec::with_capacity(world_size);
                for r in 0..world_size {
                    let mut tr = MockRouteTransport::new(None);
                    execute_route_gather(
                        &mut tr,
                        &RouteGatherParams {
                            rank: r,
                            world_size,
                            send_routes_ptr: send_base as u64,
                            recv_routes_ptr: recv_base as u64,
                            local_tokens,
                            layout,
                            stream: 1,
                        },
                    )
                    .expect("route gather simulation plan must succeed");
                    assert_eq!(tr.group_depth, 0);
                    plans.push(tr.ops);
                }

                // Split each rank's plan into initial memcpys + per-round groups.
                let mut rounds_per_rank: Vec<Vec<Vec<RecordedOp>>> = vec![Vec::new(); world_size];
                for ((rank_mem, rank_rounds), plan) in mem
                    .iter_mut()
                    .zip(rounds_per_rank.iter_mut())
                    .zip(plans.iter())
                {
                    let mut current_group: Option<Vec<RecordedOp>> = None;
                    for op in plan {
                        match op {
                            RecordedOp::Memcpy {
                                dst, src, bytes, ..
                            } => {
                                let d = *dst as usize;
                                let s = *src as usize;
                                let copy_buf = rank_mem[s..s + *bytes].to_vec();
                                rank_mem[d..d + *bytes].copy_from_slice(&copy_buf);
                            }
                            RecordedOp::GroupStart => {
                                assert!(current_group.is_none());
                                current_group = Some(Vec::new());
                            }
                            RecordedOp::GroupEnd => {
                                let grp = current_group.take().expect("matching GroupStart");
                                rank_rounds.push(grp);
                            }
                            RecordedOp::Send { .. } | RecordedOp::Recv { .. } => {
                                current_group
                                    .as_mut()
                                    .expect("send/recv inside group")
                                    .push(op.clone());
                            }
                            RecordedOp::AllGather { .. } => {
                                panic!("unexpected AllGather in small-payload relay test");
                            }
                        }
                    }
                }

                let num_rounds = rounds_per_rank[0].len();
                assert_eq!(num_rounds, if world_size == 4 { 2 } else { 1 });

                for round in 0..num_rounds {
                    // Snapshot all Sends in this round before applying Recvs.
                    let mut in_flight: Vec<(usize, usize, Vec<u8>)> = Vec::new();
                    for (r, (rank_rounds, rank_mem)) in
                        rounds_per_rank.iter().zip(mem.iter()).enumerate()
                    {
                        for op in &rank_rounds[round] {
                            if let RecordedOp::Send {
                                ptr, bytes, peer, ..
                            } = op
                            {
                                assert_ne!(*peer, r, "self-send forbidden");
                                let p = *ptr as usize;
                                in_flight.push((r, *peer, rank_mem[p..p + *bytes].to_vec()));
                            }
                        }
                    }

                    // Match every Recv against a unique in-flight Send and verify disjoint writes.
                    for (r, (rank_rounds, rank_mem)) in
                        rounds_per_rank.iter().zip(mem.iter_mut()).enumerate()
                    {
                        let mut written_ranges: Vec<(usize, usize)> = Vec::new();
                        for op in &rank_rounds[round] {
                            if let RecordedOp::Recv {
                                ptr, bytes, peer, ..
                            } = op
                            {
                                assert_ne!(*peer, r, "self-recv forbidden");
                                let p = *ptr as usize;
                                let end = p + *bytes;
                                for &(prev_s, prev_e) in &written_ranges {
                                    assert!(
                                        end <= prev_s || p >= prev_e,
                                        "overlapping recv writes in round {round} on rank {r}"
                                    );
                                }
                                written_ranges.push((p, end));

                                let pos = in_flight
                                    .iter()
                                    .position(|(src, dst, data)| {
                                        *src == *peer && *dst == r && data.len() == *bytes
                                    })
                                    .expect("every recv must match an in-flight send");
                                let (_, _, payload) = in_flight.remove(pos);
                                rank_mem[p..end].copy_from_slice(&payload);
                            }
                        }
                    }
                    assert!(
                        in_flight.is_empty(),
                        "every send must be consumed by a matching recv in round {round}"
                    );
                }

                // Verify every rank holds the exact concatenated slices of all ranks.
                let mut expected_all = Vec::with_capacity(world_size * rank_bytes);
                for src_mem in &mem {
                    expected_all.extend_from_slice(&src_mem[send_base..send_base + rank_bytes]);
                }
                for (r, rank_mem) in mem.iter().enumerate() {
                    assert_eq!(
                        &rank_mem[recv_base..recv_base + world_size * rank_bytes],
                        expected_all.as_slice(),
                        "byte mismatch for world_size={world_size}, layout={layout:?}, rank={r}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_route_gather_balanced_groups_on_mid_collective_error_2_3_4_ranks() {
        let cases = [(2usize, 2usize), (3, 4), (4, 8)];
        for (world_size, total_p2p_ops) in cases {
            for fail_idx in 0..total_p2p_ops {
                let mut tr = MockRouteTransport::new(Some(fail_idx));
                let res = execute_route_gather(
                    &mut tr,
                    &RouteGatherParams {
                        rank: 0,
                        world_size,
                        send_routes_ptr: 0x1000,
                        recv_routes_ptr: 0x20_0000,
                        local_tokens: 2,
                        layout: RoutePacketLayout::TopK8,
                        stream: 3,
                    },
                );
                assert!(
                    res.is_err(),
                    "failure at op {fail_idx} (world={world_size}) must propagate"
                );
                assert_eq!(
                    tr.group_depth, 0,
                    "group_end must always balance group_start (world={world_size}, fail_idx={fail_idx})"
                );
            }
        }
    }

    #[test]
    fn test_split_all_reduce_begin_end_non_blocking_contract() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        enum SplitEvent {
            Record(u64, u64),
            Wait(u64, u64),
            Launch(u64, usize, u64),
        }

        struct RecordingSplitOps {
            events: Vec<SplitEvent>,
        }

        impl SplitAllReduceOps for RecordingSplitOps {
            fn record_event(&mut self, event: u64, stream: u64) -> Result<()> {
                self.events.push(SplitEvent::Record(event, stream));
                Ok(())
            }
            fn stream_wait_event(&mut self, stream: u64, event: u64) -> Result<()> {
                self.events.push(SplitEvent::Wait(stream, event));
                Ok(())
            }
            fn launch_all_reduce(
                &mut self,
                ptr: u64,
                bytes: usize,
                comm_stream: u64,
            ) -> Result<()> {
                self.events
                    .push(SplitEvent::Launch(ptr, bytes, comm_stream));
                Ok(())
            }
        }

        let cfg = SplitAllReduceConfig {
            compute_stream: 10,
            comm_stream: 20,
            compute_done_event: 100,
            comm_done_event: 200,
            world_size: 2,
        };

        let mut ops = RecordingSplitOps { events: Vec::new() };
        let token = execute_all_reduce_begin(&mut ops, 0x5000, 256, cfg)
            .expect("all_reduce_begin must succeed");
        assert_eq!(token.raw_event(), cfg.comm_done_event);
        // Critical overlap invariant: compute_stream must NOT wait on comm_done during begin!
        assert_eq!(
            ops.events,
            vec![
                SplitEvent::Record(cfg.compute_done_event, cfg.compute_stream),
                SplitEvent::Wait(cfg.comm_stream, cfg.compute_done_event),
                SplitEvent::Launch(0x5000, 256, cfg.comm_stream),
                SplitEvent::Record(cfg.comm_done_event, cfg.comm_stream),
            ]
        );

        // End phase orders compute_stream after comm_done.
        execute_all_reduce_end(&mut ops, token, cfg.compute_stream)
            .expect("all_reduce_end must succeed");
        assert_eq!(
            ops.events.last(),
            Some(&SplitEvent::Wait(cfg.compute_stream, cfg.comm_done_event))
        );

        // Zero bytes or single-rank returns noop token without touching events.
        let mut noop_ops = RecordingSplitOps { events: Vec::new() };
        let noop_tok =
            execute_all_reduce_begin(&mut noop_ops, 0x5000, 0, cfg).expect("0-byte begin is noop");
        assert!(noop_tok.is_noop());
        execute_all_reduce_end(&mut noop_ops, noop_tok, cfg.compute_stream)
            .expect("noop end succeeds");
        assert!(noop_ops.events.is_empty());

        // Odd BF16 byte count fails cleanly.
        assert!(execute_all_reduce_begin(&mut noop_ops, 0x5000, 3, cfg).is_err());
    }

    #[test]
    fn test_broadcast_timeout_flags_root_only_not_idle_workers() {
        // Root sender stalling >= COLLECTIVE_TIMEOUT_SECS indicates a hung collective.
        assert!(!should_flag_broadcast_timeout(
            0,
            0,
            COLLECTIVE_TIMEOUT_SECS - 1
        ));
        assert!(should_flag_broadcast_timeout(0, 0, COLLECTIVE_TIMEOUT_SECS));
        assert!(should_flag_broadcast_timeout(
            0,
            0,
            COLLECTIVE_TIMEOUT_SECS + 120
        ));

        // Non-root worker ranks waiting idly for root's next request must NOT be
        // flagged unhealthy even across multi-hour idle gaps.
        for worker_rank in 1..4 {
            assert!(!should_flag_broadcast_timeout(
                worker_rank,
                0,
                COLLECTIVE_TIMEOUT_SECS
            ));
            assert!(!should_flag_broadcast_timeout(worker_rank, 0, 3600));
        }
    }
}
