// SPDX-License-Identifier: AGPL-3.0-only

#![deny(clippy::all)]

//! Communication backend abstraction (SBIO IORouter for collective ops).
//!
//! All distributed communication flows through [`CommBackend`]. Business
//! logic never calls NCCL or MPI directly.
//!
//! - [`SingleGpuBackend`] — all ops are no-ops (single GPU).
//! - [`NcclBackend`] — real multi-GPU via NCCL (expert parallelism).

use anyhow::Result;

// NCCL FFI + the multi-GPU `NcclBackend` are gated on `cuda` because
// they `#[link(name = "nccl")]` + `#[link(name = "cuda")]`. On metal
// builds (single Apple Silicon device) only `SingleGpuBackend` below
// is needed.
#[cfg(feature = "cuda")]
pub mod nccl;
#[cfg(feature = "cuda")]
pub mod nccl_backend;
#[cfg(feature = "cuda")]
pub use nccl_backend::NcclBackend;
#[cfg(feature = "cuda")]
pub mod tcp_backend;
#[cfg(feature = "cuda")]
pub use tcp_backend::TcpCommBackend;

/// Number of top-k routed expert slots carried in one [`SelectedRoute`] packet.
pub const SELECTED_ROUTE_SLOTS: usize = 6;

/// Exact wire size of one [`SelectedRoute`] packet (`6 × (u32 + f32) = 48` bytes).
pub const SELECTED_ROUTE_BYTES: usize = 48;

/// Compact 48-byte selected-route packet (`6 × (u32 expert_id, f32 weight)`)
/// for multi-Spark MoE route exchange without full activation broadcast.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectedRoute {
    /// Selected expert IDs (`u32::MAX` marks an unused slot when `top_k < 6`).
    pub expert_ids: [u32; SELECTED_ROUTE_SLOTS],
    /// IEEE-754 bit patterns of the corresponding `f32` routing weights.
    pub weight_bits: [u32; SELECTED_ROUTE_SLOTS],
}

const _: () = assert!(core::mem::size_of::<SelectedRoute>() == SELECTED_ROUTE_BYTES);

/// Wire layout for compact GPU-resident selected-route packets exchanged across EP ranks.
///
/// Each slot holds one `(u32 expert_id, f32 weight)` pair (`8` bytes). Unused slots
/// when `top_k < slots` are padded with `expert_id = u32::MAX` and `weight = 0.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePacketLayout {
    /// 6 routed-expert slots (`48` bytes, matching [`SelectedRoute`]).
    TopK6,
    /// 8 routed-expert slots (`64` bytes, e.g. 48B/80B MoE models with `top_k = 8`).
    TopK8,
    /// 10 routed-expert slots (`80` bytes, e.g. MoE models with `top_k = 10`).
    TopK10,
}

impl RoutePacketLayout {
    /// Select the smallest compact packet layout that fits `top_k` routed experts.
    ///
    /// Returns `None` when `top_k == 0` or `top_k > 10`.
    #[must_use]
    pub const fn for_top_k(top_k: usize) -> Option<Self> {
        match top_k {
            1..=6 => Some(Self::TopK6),
            7..=8 => Some(Self::TopK8),
            9..=10 => Some(Self::TopK10),
            _ => None,
        }
    }

    /// Number of `(expert_id, weight)` slots carried by one packet (`6`, `8`, or `10`).
    #[must_use]
    pub const fn slots(self) -> usize {
        match self {
            Self::TopK6 => 6,
            Self::TopK8 => 8,
            Self::TopK10 => 10,
        }
    }

    /// Exact wire size of one packet in bytes (`48`, `64`, or `80`).
    #[must_use]
    pub const fn packet_bytes(self) -> usize {
        self.slots() * 8
    }
}

const _: () = assert!(RoutePacketLayout::TopK6.slots() == SELECTED_ROUTE_SLOTS);
const _: () = assert!(RoutePacketLayout::TopK6.packet_bytes() == SELECTED_ROUTE_BYTES);

/// Opaque completion token returned by [`CommBackend::all_reduce_begin`].
///
/// Intentionally non-`Copy` and non-`Clone` so each in-flight collective is
/// synchronized exactly once via [`CommBackend::all_reduce_end`].
#[must_use = "must pass token to `all_reduce_end` before consuming reduced buffer"]
#[derive(Debug, PartialEq, Eq)]
pub struct CollectiveToken {
    pub(crate) comm_done_event: u64,
}

impl CollectiveToken {
    /// Construct a no-op token (`comm_done_event == 0`) for synchronous or single-GPU backends.
    pub const fn noop() -> Self {
        Self { comm_done_event: 0 }
    }

    /// Construct a token wrapping a raw backend completion event handle (for mock/custom backends).
    pub const fn from_raw_event(comm_done_event: u64) -> Self {
        Self { comm_done_event }
    }

    /// Return the underlying completion event handle (`0` when no wait is required).
    #[must_use]
    pub const fn raw_event(&self) -> u64 {
        self.comm_done_event
    }

    /// Returns `true` when this token requires no stream wait.
    #[must_use]
    pub const fn is_noop(&self) -> bool {
        self.comm_done_event == 0
    }
}

/// Element data type for typed collective reductions such as [`CommBackend::reduce_scatter_typed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveDtype {
    /// BFloat16 (`2` bytes per element).
    Bf16,
    /// IEEE-754 single-precision float (`4` bytes per element).
    Fp32,
    /// Raw unsigned byte (`1` byte per element).
    Uint8,
}

impl CollectiveDtype {
    /// Size of one element in bytes (`2` for `Bf16`, `4` for `Fp32`, `1` for `Uint8`).
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp32 => 4,
            Self::Uint8 => 1,
        }
    }
}

impl SelectedRoute {
    /// Construct an empty route packet (`u32::MAX` expert IDs, `0.0` weights).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            expert_ids: [u32::MAX; SELECTED_ROUTE_SLOTS],
            weight_bits: [0; SELECTED_ROUTE_SLOTS],
        }
    }

    /// Construct a 48-byte route packet from `(expert_ids, weights)`.
    #[must_use]
    pub fn from_experts_and_weights(
        expert_ids: [u32; SELECTED_ROUTE_SLOTS],
        weights: [f32; SELECTED_ROUTE_SLOTS],
    ) -> Self {
        Self {
            expert_ids,
            weight_bits: weights.map(f32::to_bits),
        }
    }

    /// Decode the stored routing weights as `[f32; 6]`.
    #[must_use]
    pub fn weights(&self) -> [f32; SELECTED_ROUTE_SLOTS] {
        self.weight_bits.map(f32::from_bits)
    }

    /// Serialize this packet into its canonical little-endian 48-byte wire representation.
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; SELECTED_ROUTE_BYTES] {
        let mut out = [0u8; SELECTED_ROUTE_BYTES];
        for (i, &id) in self.expert_ids.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
        }
        for (i, &wb) in self.weight_bits.iter().enumerate() {
            let base = 24 + i * 4;
            out[base..base + 4].copy_from_slice(&wb.to_le_bytes());
        }
        out
    }

    /// Reconstruct a [`SelectedRoute`] from its canonical little-endian 48-byte wire representation.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8; SELECTED_ROUTE_BYTES]) -> Self {
        let mut expert_ids = [0u32; SELECTED_ROUTE_SLOTS];
        let mut weight_bits = [0u32; SELECTED_ROUTE_SLOTS];
        for (i, dst) in expert_ids.iter_mut().enumerate() {
            let off = i * 4;
            *dst = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        for (i, dst) in weight_bits.iter_mut().enumerate() {
            let off = 24 + i * 4;
            *dst = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        Self {
            expert_ids,
            weight_bits,
        }
    }
}

/// Communication backend trait for distributed operations.
///
/// All collective operations take raw device pointer + byte count.
/// The pointer type is u64 (matching CUDA CUdeviceptr) to avoid
/// coupling this crate to spark-runtime's DevicePtr.
pub trait CommBackend: Send + Sync {
    /// All-reduce: sum across all ranks, result on all ranks.
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;

    /// All-gather: each rank contributes a chunk, all ranks get full buffer.
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Reduce-scatter: reduce + scatter (inverse of all-gather).
    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Typed reduce-scatter: sum-reduce and scatter `bytes` per rank as `dtype` elements on `stream`.
    ///
    /// Validates that `bytes` is an exact multiple of `dtype.element_bytes()`.
    /// Default implementation checks alignment and delegates to [`Self::reduce_scatter`].
    fn reduce_scatter_typed(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        dtype: CollectiveDtype,
        stream: u64,
    ) -> Result<()> {
        let elem_bytes = dtype.element_bytes();
        anyhow::ensure!(
            bytes.is_multiple_of(elem_bytes),
            "reduce_scatter_typed: byte count {bytes} is not a multiple of {:?} element size ({elem_bytes})",
            dtype
        );
        let _ = stream;
        self.reduce_scatter(send_ptr, recv_ptr, bytes)
    }

    /// Broadcast from root rank to all ranks.
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;

    /// Barrier: block until all ranks reach this point.
    fn barrier(&self) -> Result<()>;

    /// Async all-reduce using GPU-side event synchronization.
    ///
    /// Replaces `gpu.synchronize(stream) + all_reduce(ptr, bytes)`.
    /// Uses a dedicated comm stream + CUDA events so the CPU never blocks.
    /// `compute_stream` is where MoE kernels ran and where residual_add will run.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let _ = compute_stream;
        self.all_reduce(ptr, bytes)
    }

    /// Begin a split-phase asynchronous all-reduce on `ptr` (`bytes` of BF16 elements).
    ///
    /// Returns a [`CollectiveToken`] without waiting `compute_stream` on collective
    /// completion so callers can overlap independent GPU work (e.g. a MoE shared expert)
    /// before calling [`Self::all_reduce_end`].
    ///
    /// Default implementation runs [`Self::all_reduce_async`] and returns a no-op token.
    fn all_reduce_begin(
        &self,
        ptr: u64,
        bytes: usize,
        compute_stream: u64,
    ) -> Result<CollectiveToken> {
        self.all_reduce_async(ptr, bytes, compute_stream)?;
        Ok(CollectiveToken { comm_done_event: 0 })
    }

    /// Complete a split-phase asynchronous all-reduce started by [`Self::all_reduce_begin`].
    ///
    /// Consumes `token` and orders subsequent work on `compute_stream` after the collective
    /// completion event. Default implementation is a no-op when `token.comm_done_event == 0`.
    fn all_reduce_end(&self, token: CollectiveToken, compute_stream: u64) -> Result<()> {
        let _ = compute_stream;
        anyhow::ensure!(
            token.comm_done_event == 0,
            "non-zero CollectiveToken event ({:#x}) passed to default CommBackend::all_reduce_end",
            token.comm_done_event
        );
        Ok(())
    }

    /// Pre-register a GPU buffer with the communication backend.
    ///
    /// For NCCL over IB/RoCE, this caches the IB memory registration
    /// (`ibv_reg_mr`), avoiding per-call overhead in all_reduce.
    /// Returns an opaque handle for deregistration.
    fn register_buffer(&self, _ptr: u64, _bytes: usize) -> Result<u64> {
        Ok(0)
    }

    /// Deregister a previously registered buffer.
    ///
    /// # Errors
    /// Returns an error if `handle` is null, unknown, or already deregistered,
    /// or if the underlying backend deregistration fails.
    fn deregister_buffer(&self, _handle: u64) -> Result<()> {
        Ok(())
    }

    /// Allocate a GPU buffer in NCCL's symmetric-memory window
    /// (NCCL ≥ 2.28 / `ncclMemAlloc`). Returns the device pointer as `u64`.
    ///
    /// Symmetric-memory allocations are the substrate for:
    ///   1. Copy-engine offload of NVLink collectives (frees SMs for compute).
    ///   2. Device-side communication API (kernels invoke collectives in-kernel),
    ///      which TokenWeave-style fused AR+RMSNorm+Residual builds on.
    ///
    /// On Atlas's 2-rank Spark over RoCE, the copy-engine offload itself does
    /// not apply (RoCE is not NVLink), but the symmetric windows are still
    /// required for future device-API fusions and to reduce per-call setup.
    /// Returns an error if the linked NCCL is < 2.28; backends that don't
    /// support symmetric memory return `Err` and callers must fall back.
    fn symmetric_alloc(&self, _bytes: usize) -> Result<u64> {
        anyhow::bail!("symmetric_alloc not supported by this CommBackend");
    }

    /// Free a buffer previously returned by `symmetric_alloc`.
    ///
    /// # Errors
    /// Returns an error if `ptr` is null, was not allocated via `symmetric_alloc`,
    /// was already freed, or if symmetric memory is unsupported by this backend.
    fn symmetric_free(&self, _ptr: u64) -> Result<()> {
        anyhow::bail!("symmetric_free not supported by this CommBackend");
    }

    /// Provide a kernel handle for the BF16 in-place addition kernel.
    ///
    /// Used by the 2-rank send/recv all-reduce path. The kernel is loaded
    /// by the model layer (which has access to AtlasRegistry) and passed
    /// to the comm backend at init time.
    fn set_add_kernel(&self, _handle: u64) {
        // Default: no-op (single GPU or backends that don't need it)
    }

    /// Send tokens to a specific rank (for EP token dispatch).
    ///
    /// Sends `bytes` from `ptr` on this rank to `dest_rank`.
    /// Must be paired with a matching `recv_from` on the destination rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()>;

    /// Receive tokens from a specific rank (for EP token combine).
    ///
    /// Receives `bytes` into `ptr` on this rank from `src_rank`.
    /// Must be paired with a matching `send_to` on the source rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()>;

    /// Gather compact selected-route packets (`local_tokens * layout.packet_bytes()` bytes per
    /// rank) across all ranks on `stream`.
    ///
    /// Never issues self-sends (`dest_rank == self.rank()`). Backends with GPU device-to-device
    /// copy support (such as `NcclBackend`) override this to place the local slice directly when
    /// out-of-place and run 1-RTT/2-RTT bidirectional ring relays for 3- and 4-rank worlds.
    fn gather_selected_routes_with_layout(
        &self,
        send_routes_ptr: u64,
        recv_routes_ptr: u64,
        local_tokens: usize,
        layout: RoutePacketLayout,
        stream: u64,
    ) -> Result<()> {
        if local_tokens == 0 || self.world_size() <= 1 {
            return Ok(());
        }
        let bytes = local_tokens
            .checked_mul(layout.packet_bytes())
            .ok_or_else(|| anyhow::anyhow!("SelectedRoute gather byte count overflow"))?;
        if self.world_size() == 2 {
            let rank = self.rank();
            let peer = 1 - rank;
            let peer_off = peer
                .checked_mul(bytes)
                .ok_or_else(|| anyhow::anyhow!("SelectedRoute peer byte offset overflow"))?;
            let peer_off_u64 = u64::try_from(peer_off)
                .map_err(|_| anyhow::anyhow!("SelectedRoute peer byte offset exceeds u64"))?;
            let recv_peer_ptr = recv_routes_ptr
                .checked_add(peer_off_u64)
                .ok_or_else(|| anyhow::anyhow!("SelectedRoute recv pointer overflow"))?;
            self.group_start()?;
            let send_recv_res = (|| -> Result<()> {
                self.send_to(send_routes_ptr, bytes, peer, stream)?;
                self.recv_from(recv_peer_ptr, bytes, peer, stream)
            })();
            let end_res = self.group_end();
            send_recv_res.and(end_res)
        } else {
            let _ = stream;
            self.all_gather(send_routes_ptr, recv_routes_ptr, bytes)
        }
    }

    /// Gather compact 48-byte [`SelectedRoute`] packets (`local_tokens * 48` bytes per rank)
    /// across all ranks on `stream`.
    fn gather_selected_routes(
        &self,
        send_routes_ptr: u64,
        recv_routes_ptr: u64,
        local_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        self.gather_selected_routes_with_layout(
            send_routes_ptr,
            recv_routes_ptr,
            local_tokens,
            RoutePacketLayout::TopK6,
            stream,
        )
    }

    /// Begin a group of point-to-point operations (send_to/recv_from).
    ///
    /// All send_to/recv_from calls between group_start and group_end are
    /// batched into a single NCCL launch for efficiency.
    fn group_start(&self) -> Result<()> {
        Ok(())
    }

    /// End a group of point-to-point operations.
    fn group_end(&self) -> Result<()> {
        Ok(())
    }

    /// Check if the communicator is healthy (no async errors, no timeouts).
    ///
    /// Returns `true` if the communicator is operational. Implementations
    /// may actively probe the underlying transport (e.g., `ncclCommGetAsyncError`).
    fn is_healthy(&self) -> bool {
        true
    }

    /// Attempt to recover a degraded communicator.
    ///
    /// For NCCL, this aborts the dead communicator and re-initializes
    /// via TCP bootstrap. Both ranks must call this concurrently.
    /// Returns `Ok(())` on successful recovery, `Err` if recovery failed.
    fn attempt_reconnect(&self) -> Result<()> {
        Ok(())
    }

    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}

/// Single-GPU backend: all collective ops are no-ops.
///
/// Used in Phase 1 where the entire model fits on one GPU.
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn send_to(&self, _ptr: u64, _bytes: usize, _dest_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn recv_from(&self, _ptr: u64, _bytes: usize, _src_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_noop() {
        let comm = SingleGpuBackend;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        comm.all_reduce(0x1000, 1024).unwrap();
        comm.all_gather(0x1000, 0x2000, 512).unwrap();
        comm.reduce_scatter(0x1000, 0x2000, 512).unwrap();
        comm.broadcast(0x1000, 256, 0).unwrap();
        comm.barrier().unwrap();
        comm.send_to(0x1000, 256, 0, 0).unwrap();
        comm.recv_from(0x2000, 256, 0, 0).unwrap();
        comm.gather_selected_routes(0x1000, 0x2000, 4, 0).unwrap();
        comm.group_start().unwrap();
        comm.group_end().unwrap();
        assert!(comm.is_healthy());
        comm.attempt_reconnect().unwrap();
    }

    #[test]
    fn test_selected_route_48b_layout_and_roundtrip() {
        assert_eq!(core::mem::size_of::<SelectedRoute>(), 48);
        let empty = SelectedRoute::empty();
        assert_eq!(empty.expert_ids, [u32::MAX; 6]);
        assert_eq!(empty.weights(), [0.0; 6]);

        let ids = [3, 19, 104, 257, 411, 510];
        let weights = [0.42_f32, 0.21, 0.14, 0.11, 0.07, 0.05];
        let pkt = SelectedRoute::from_experts_and_weights(ids, weights);
        let raw = pkt.to_le_bytes();
        assert_eq!(raw.len(), SELECTED_ROUTE_BYTES);
        let decoded = SelectedRoute::from_le_bytes(&raw);
        assert_eq!(decoded, pkt);
        assert_eq!(decoded.expert_ids, ids);
        assert_eq!(decoded.weights(), weights);
    }

    #[test]
    fn test_single_gpu_symmetric_alloc_unsupported() {
        // SingleGpuBackend doesn't override symmetric_alloc/free — it must
        // fall back to the trait default which returns an error. This is
        // the contract callers depend on for fallback paths.
        let comm = SingleGpuBackend;
        assert!(comm.symmetric_alloc(1024).is_err());
        assert!(comm.symmetric_free(0x1000).is_err());
    }

    #[test]
    fn test_gather_selected_routes_closes_group_on_send_error() {
        use std::sync::atomic::{AtomicIsize, Ordering};

        struct FailingTwoRankComm {
            group_depth: AtomicIsize,
        }
        impl CommBackend for FailingTwoRankComm {
            fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
                Ok(())
            }
            fn barrier(&self) -> Result<()> {
                Ok(())
            }
            fn send_to(&self, _: u64, _: usize, dest_rank: usize, _: u64) -> Result<()> {
                if dest_rank == 1 {
                    anyhow::bail!("simulated peer send failure");
                }
                Ok(())
            }
            fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn group_start(&self) -> Result<()> {
                self.group_depth.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn group_end(&self) -> Result<()> {
                self.group_depth.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
            fn rank(&self) -> usize {
                0
            }
            fn world_size(&self) -> usize {
                2
            }
        }

        let comm = FailingTwoRankComm {
            group_depth: AtomicIsize::new(0),
        };
        let err = comm
            .gather_selected_routes(0x1000, 0x2000, 4, 0)
            .unwrap_err();
        assert!(err.to_string().contains("simulated peer send failure"));
        assert_eq!(
            comm.group_depth.load(Ordering::SeqCst),
            0,
            "group_end must always balance group_start on error"
        );
    }

    #[test]
    fn test_route_packet_layout_for_top_k_and_sizes() {
        assert_eq!(RoutePacketLayout::for_top_k(0), None);
        assert_eq!(RoutePacketLayout::for_top_k(11), None);
        assert_eq!(RoutePacketLayout::for_top_k(usize::MAX), None);

        for top_k in 1..=6 {
            let layout =
                RoutePacketLayout::for_top_k(top_k).expect("top_k in 1..=6 must map to TopK6");
            assert_eq!(layout, RoutePacketLayout::TopK6);
            assert_eq!(layout.slots(), 6);
            assert_eq!(layout.packet_bytes(), 48);
        }
        for top_k in 7..=8 {
            let layout =
                RoutePacketLayout::for_top_k(top_k).expect("top_k in 7..=8 must map to TopK8");
            assert_eq!(layout, RoutePacketLayout::TopK8);
            assert_eq!(layout.slots(), 8);
            assert_eq!(layout.packet_bytes(), 64);
        }
        for top_k in 9..=10 {
            let layout =
                RoutePacketLayout::for_top_k(top_k).expect("top_k in 9..=10 must map to TopK10");
            assert_eq!(layout, RoutePacketLayout::TopK10);
            assert_eq!(layout.slots(), 10);
            assert_eq!(layout.packet_bytes(), 80);
        }
    }

    #[test]
    fn test_collective_dtype_sizes_and_misalignment_rejection() {
        assert_eq!(CollectiveDtype::Bf16.element_bytes(), 2);
        assert_eq!(CollectiveDtype::Fp32.element_bytes(), 4);
        assert_eq!(CollectiveDtype::Uint8.element_bytes(), 1);

        let comm = SingleGpuBackend;
        comm.reduce_scatter_typed(0x1000, 0x2000, 0, CollectiveDtype::Bf16, 0)
            .expect("0 bytes is always aligned");
        comm.reduce_scatter_typed(0x1000, 0x2000, 64, CollectiveDtype::Bf16, 0)
            .expect("64 bytes aligned for Bf16");
        comm.reduce_scatter_typed(0x1000, 0x2000, 64, CollectiveDtype::Fp32, 0)
            .expect("64 bytes aligned for Fp32");
        comm.reduce_scatter_typed(0x1000, 0x2000, 63, CollectiveDtype::Uint8, 0)
            .expect("63 bytes aligned for Uint8");

        assert!(
            comm.reduce_scatter_typed(0x1000, 0x2000, 3, CollectiveDtype::Bf16, 0)
                .is_err(),
            "odd byte count must fail for Bf16"
        );
        assert!(
            comm.reduce_scatter_typed(0x1000, 0x2000, 6, CollectiveDtype::Fp32, 0)
                .is_err(),
            "non-4B-multiple byte count must fail for Fp32"
        );
    }

    #[test]
    fn test_all_reduce_begin_end_default_contract() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingComm {
            async_calls: AtomicUsize,
        }
        impl CommBackend for CountingComm {
            fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn all_reduce_async(&self, _: u64, _: usize, _: u64) -> Result<()> {
                self.async_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
                Ok(())
            }
            fn barrier(&self) -> Result<()> {
                Ok(())
            }
            fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
                Ok(())
            }
            fn rank(&self) -> usize {
                0
            }
            fn world_size(&self) -> usize {
                2
            }
        }

        let comm = CountingComm {
            async_calls: AtomicUsize::new(0),
        };
        let tok = comm
            .all_reduce_begin(0x1000, 128, 7)
            .expect("default all_reduce_begin must succeed");
        assert!(tok.is_noop());
        assert_eq!(tok.raw_event(), 0);
        assert_eq!(comm.async_calls.load(Ordering::SeqCst), 1);
        comm.all_reduce_end(tok, 7)
            .expect("default all_reduce_end must accept noop token");

        let bogus = CollectiveToken::from_raw_event(0xDEAD_BEEF);
        assert!(!bogus.is_noop());
        assert_eq!(bogus.raw_event(), 0xDEAD_BEEF);
        assert!(
            comm.all_reduce_end(bogus, 7).is_err(),
            "default all_reduce_end must reject unhandled non-zero event token"
        );
    }

    #[test]
    fn test_gather_selected_routes_with_layout_zero_self_sends_two_rank() {
        use parking_lot::Mutex;

        struct RecordingTwoRankComm {
            rank: usize,
            sends: Mutex<Vec<(usize, usize)>>,
            recvs: Mutex<Vec<(usize, usize, u64)>>,
        }
        impl CommBackend for RecordingTwoRankComm {
            fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
                Ok(())
            }
            fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
                Ok(())
            }
            fn barrier(&self) -> Result<()> {
                Ok(())
            }
            fn send_to(&self, _: u64, bytes: usize, dest_rank: usize, _: u64) -> Result<()> {
                self.sends.lock().push((dest_rank, bytes));
                Ok(())
            }
            fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, _: u64) -> Result<()> {
                self.recvs.lock().push((src_rank, bytes, ptr));
                Ok(())
            }
            fn rank(&self) -> usize {
                self.rank
            }
            fn world_size(&self) -> usize {
                2
            }
        }

        for layout in [
            RoutePacketLayout::TopK6,
            RoutePacketLayout::TopK8,
            RoutePacketLayout::TopK10,
        ] {
            let comm = RecordingTwoRankComm {
                rank: 0,
                sends: Mutex::new(Vec::new()),
                recvs: Mutex::new(Vec::new()),
            };
            let local_tokens = 3;
            let expected_bytes = local_tokens * layout.packet_bytes();
            comm.gather_selected_routes_with_layout(0x1000, 0x2000, local_tokens, layout, 0)
                .expect("two-rank gather must succeed");
            let sends = comm.sends.lock().clone();
            let recvs = comm.recvs.lock().clone();
            assert_eq!(sends, vec![(1, expected_bytes)]);
            assert_eq!(
                recvs,
                vec![(1, expected_bytes, 0x2000 + expected_bytes as u64)]
            );
        }
    }
}
