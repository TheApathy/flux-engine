// SPDX-License-Identifier: AGPL-3.0-only

//! NCCL-based communication backend for expert parallelism.
//!
//! Uses TCP bootstrap: rank 0 generates a unique ID and sends it to
//! all other ranks via a TCP listener. Then all ranks call
//! `ncclCommInitRankConfig` (non-blocking with bounded async-error poll,
//! falling back to `ncclCommInitRank` only on `NcclConfig` ABI mismatch)
//! with the shared ID.
//!
//! Optimizations for 2-rank EP with small messages (4 KB):
//! - Pre-registers buffers with NCCL (`ncclCommRegister`) to cache IB memory registration
//! - Uses paired `ncclSend`/`ncclRecv` + local BF16 add instead of `ncclAllReduce`
//!
//! Health monitoring and recovery:
//! - Checks `ncclCommGetAsyncError` after each collective
//! - Detects broadcast timeouts (`>=30s` on root rank) via stream sync + wall-clock check
//! - Aborts dead communicators via `ncclCommAbort` and reconnects
//!
//! ## Safety contract
//!
//! All `unsafe` blocks in this module carry per-site `// SAFETY:` comments and
//! wrap FFI calls into NCCL (`nccl*`, baseline NCCL 2.24+, `ncclConfig_v22800` /
//! symmetric memory on NCCL 2.28+) or the CUDA Driver API (`cu*`):
//!
//! - **NCCL handles**: `NcclComm` is wrapped in the non-`Copy` [`OwnedNcclComm`]
//!   type stored inside `Mutex<Option<OwnedNcclComm>>`. Every NCCL operation
//!   holds the mutex guard across the full FFI call so the raw handle never
//!   escapes the lock and cannot race with `ncclCommAbort` or `Drop`.
//! - **CUDA buffers** passed to NCCL come from valid device allocations on the
//!   device that owns the comm, with sizes matching the allocation.
//! - **Streams** referenced via `u64` are owned by this backend or the caller
//!   and remain live for the duration of the FFI call.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::nccl::{self, NcclComm, NcclDataType, NcclResult, NcclUniqueId};

// CUDA driver API for recv/barrier buffer allocation, async D2D copy, and kernel launch.
// SAFETY: External symbol signatures match the 64-bit CUDA Driver API C ABI in `cuda.h`.
unsafe extern "C" {
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemcpyDtoDAsync_v2(
        dst_device: u64,
        src_device: u64,
        byte_count: usize,
        h_stream: u64,
    ) -> i32;
    fn cuLaunchKernel(
        f: u64,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: u64,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

/// Enqueue an asynchronous device-to-device copy on `stream`.
///
/// No-op when `bytes == 0` or `dst == src` (in-place buffer).
///
/// # Safety
/// `dst` and `src` must each point to at least `bytes` valid device bytes in
/// the active CUDA context, and `stream` must be `0` or a valid live `CUstream`.
pub(super) unsafe fn memcpy_d2d_async(dst: u64, src: u64, bytes: usize, stream: u64) -> Result<()> {
    if bytes == 0 || dst == src {
        return Ok(());
    }
    // SAFETY: Caller guarantees `dst`, `src`, and `stream` are valid live CUDA
    // device pointers/stream covering at least `bytes` bytes.
    let status = unsafe { cuMemcpyDtoDAsync_v2(dst, src, bytes, stream) };
    if status != 0 {
        anyhow::bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
    }
    Ok(())
}

/// Maximum recv buffer size (covers hidden_size * max_batch_tokens * 2 bytes).
/// Recv buffer for multi-rank collectives (e.g. 2-rank send/recv all-reduce).
/// Must cover the largest all-reduce payload: prefill chunk output.
/// For 122B (hidden=6144) at max_prefill_tokens=4096:
///   4096 * 6144 * 2 = 48 MB.
/// Allocate 64 MB to cover all models with headroom.
pub(crate) const RECV_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// Timeout threshold for a single synchronous collective operation.
/// If a broadcast + stream sync takes longer than this, mark the communicator unhealthy.
pub(super) const COLLECTIVE_TIMEOUT_SECS: u64 = 30;

/// Bounded wait timeout for non-blocking `ncclCommInitRankConfig` completion.
pub(super) const COMM_INIT_TIMEOUT_SECS: u64 = 60;

/// Poll interval while waiting on `NcclResult::InProgress` during communicator init.
const COMM_INIT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1);

/// Default ConnectX-7 RoCE NCCL tuning variables applied only when unset by the user.
pub(crate) const CX7_ROCE_DEFAULT_ENVS: [(&str, &str); 3] = [
    ("NCCL_IB_QPS_PER_CONNECTION", "4"),
    ("NCCL_IB_SPLIT_DATA_ON_QPS", "0"),
    ("NCCL_PROTO", "LL,Simple"),
];

/// Environment keys that explicitly declare single-host vs multi-host communicator locality.
const SINGLE_HOST_ENV_KEYS: &[&str] = &[
    "FLUX_SINGLE_HOST",
    "FLUX_COMM_SINGLE_HOST",
    "ATLAS_SINGLE_HOST",
    "ATLAS_COMM_SINGLE_HOST",
];

/// Environment keys that explicitly enable or suppress ConnectX-7 RoCE defaults/mode.
const CX7_ROCE_ENV_KEYS: &[&str] = &["FLUX_CX7_ROCE", "ATLAS_CX7_ROCE"];

/// Environment keys that force the 2-rank P2P `send/recv + bf16_add_inplace` path
/// even on single-host intra-node runs.
const FORCE_P2P_ALLREDUCE_ENV_KEYS: &[&str] =
    &["FLUX_FORCE_P2P_ALLREDUCE", "ATLAS_FORCE_P2P_ALLREDUCE"];

/// Host locality of the NCCL communicator clique.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum HostLocality {
    /// All ranks run on a single host (PCIe P2P / NVLink / SHM).
    SingleHost,
    /// Ranks span multiple hosts over a network fabric (e.g. ConnectX-7 RoCE).
    MultiHost,
}

impl HostLocality {
    /// Returns `true` when all ranks reside on the same physical host.
    #[must_use]
    pub(crate) const fn is_single_host(self) -> bool {
        matches!(self, Self::SingleHost)
    }
}

/// Source of communicator locality resolution.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum LocalityOverride {
    /// No explicit locality override set in the environment; auto-detect from
    /// `master_addr` (and accepted TCP peer addresses on rank 0).
    Auto,
    /// Explicitly forced via `FLUX_SINGLE_HOST` / `FLUX_COMM_SINGLE_HOST` / `FLUX_CX7_ROCE`.
    Explicit(HostLocality),
}

/// Override mode for the 2-rank P2P `send/recv + bf16_add_inplace` all-reduce path.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum P2pAllReduceOverride {
    /// Select automatically from [`HostLocality`]: RoCE send/recv on multi-host,
    /// native `ncclAllReduce` on single-host.
    Auto,
    /// Force the 2-rank send/recv + `bf16_add_inplace` path (`FLUX_FORCE_P2P_ALLREDUCE=1`).
    ForceP2p,
}

/// Inputs to the 2-rank RoCE `send/recv + bf16_add_inplace` vs native `ncclAllReduce`
/// dispatch decision.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct AllReduceDispatchCtx {
    pub(crate) world_size: usize,
    pub(crate) locality: HostLocality,
    pub(crate) p2p_override: P2pAllReduceOverride,
    pub(crate) bytes: usize,
    pub(crate) recv_buffer: u64,
    pub(crate) add_kernel: u64,
}

fn parse_env_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn lookup_env_bool<F>(lookup: &F, keys: &[&str]) -> Option<bool>
where
    F: Fn(&str) -> Option<String>,
{
    for key in keys {
        if let Some(val) = lookup(key)
            && let Some(parsed) = parse_env_bool(&val)
        {
            return Some(parsed);
        }
    }
    None
}

fn is_loopback_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    }
}

/// Returns `true` when `master_addr` explicitly addresses the local loopback interface
/// (`127.0.0.0/8`, `localhost`, `::1`, or `[::1]`).
///
/// Wildcard bind addresses (`0.0.0.0`, `::`) and routable LAN IPs return `false` so
/// multi-node rank 0 deployments are never misclassified as single-host.
#[must_use]
pub(crate) fn is_single_host_addr(master_addr: &str) -> bool {
    let trimmed = master_addr.trim();
    if trimmed.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = trimmed.parse::<std::net::IpAddr>() {
        return is_loopback_ip(ip);
    }
    if let Some(inner) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        && let Ok(ip) = inner.parse::<std::net::Ipv6Addr>()
    {
        return is_loopback_ip(std::net::IpAddr::V6(ip));
    }
    false
}

/// Resolve any explicit single-host / CX-7 RoCE override from environment variables.
#[must_use]
pub(crate) fn resolve_locality_override<F>(lookup: F) -> LocalityOverride
where
    F: Fn(&str) -> Option<String>,
{
    if lookup_env_bool(&lookup, CX7_ROCE_ENV_KEYS) == Some(true) {
        return LocalityOverride::Explicit(HostLocality::MultiHost);
    }
    if let Some(single_host) = lookup_env_bool(&lookup, SINGLE_HOST_ENV_KEYS) {
        return LocalityOverride::Explicit(if single_host {
            HostLocality::SingleHost
        } else {
            HostLocality::MultiHost
        });
    }
    LocalityOverride::Auto
}

/// Resolve initial communicator [`HostLocality`] from environment overrides and `master_addr`.
#[must_use]
pub(crate) fn resolve_host_locality<F>(master_addr: &str, lookup: F) -> HostLocality
where
    F: Fn(&str) -> Option<String>,
{
    match resolve_locality_override(lookup) {
        LocalityOverride::Explicit(locality) => locality,
        LocalityOverride::Auto => {
            if is_single_host_addr(master_addr) {
                HostLocality::SingleHost
            } else {
                HostLocality::MultiHost
            }
        }
    }
}

/// Refine rank 0's auto-detected locality using an accepted worker TCP peer address.
///
/// When rank 0 uses the CLI default `--master-addr 127.0.0.1` on a multi-node cluster
/// while workers connect over LAN/RoCE, observing a non-loopback peer IP downgrades
/// rank 0 to [`HostLocality::MultiHost`] so all ranks agree on transport and collective paths.
#[must_use]
pub(crate) fn refine_rank0_locality(
    current: HostLocality,
    override_mode: LocalityOverride,
    peer_ip: std::net::IpAddr,
) -> HostLocality {
    match override_mode {
        LocalityOverride::Explicit(locality) => locality,
        LocalityOverride::Auto => {
            if current.is_single_host() && !is_loopback_ip(peer_ip) {
                HostLocality::MultiHost
            } else {
                current
            }
        }
    }
}

/// Resolve [`P2pAllReduceOverride`] from `FLUX_FORCE_P2P_ALLREDUCE` / `ATLAS_FORCE_P2P_ALLREDUCE`.
#[must_use]
pub(crate) fn resolve_p2p_allreduce_override<F>(lookup: F) -> P2pAllReduceOverride
where
    F: Fn(&str) -> Option<String>,
{
    if lookup_env_bool(&lookup, FORCE_P2P_ALLREDUCE_ENV_KEYS) == Some(true) {
        P2pAllReduceOverride::ForceP2p
    } else {
        P2pAllReduceOverride::Auto
    }
}

/// Returns `true` when ConnectX-7 RoCE default NCCL tuning variables
/// ([`CX7_ROCE_DEFAULT_ENVS`]) should be applied for any unset keys.
///
/// - Explicit `FLUX_CX7_ROCE` / `ATLAS_CX7_ROCE` overrides win (`1` enables, `0` disables).
/// - Otherwise defaults apply only on [`HostLocality::MultiHost`]; single-host PCIe P2P /
///   NVLink runs skip them so NCCL retains `LL128`.
#[must_use]
pub(crate) fn should_apply_cx7_roce_defaults<F>(locality: HostLocality, lookup: F) -> bool
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(explicit) = lookup_env_bool(&lookup, CX7_ROCE_ENV_KEYS) {
        return explicit;
    }
    matches!(locality, HostLocality::MultiHost)
}

/// Returns `true` when the communicator requires a pre-allocated & registered 64 MiB `recv_buffer`.
///
/// Multi-host clusters retain allocation for all multi-rank worlds (`world_size >= 2`).
/// Single-host intra-node runs skip the 64 MiB staging buffer unless `world_size == 2`
/// and `FLUX_FORCE_P2P_ALLREDUCE=1` (`P2pAllReduceOverride::ForceP2p`).
#[must_use]
pub(crate) const fn should_allocate_recv_buffer(
    world_size: usize,
    locality: HostLocality,
    p2p_override: P2pAllReduceOverride,
) -> bool {
    if world_size < 2 {
        return false;
    }
    match locality {
        HostLocality::MultiHost => true,
        HostLocality::SingleHost => {
            world_size == 2 && matches!(p2p_override, P2pAllReduceOverride::ForceP2p)
        }
    }
}

/// Returns `true` when BF16 `all_reduce` should use the 2-rank RoCE
/// `ncclSend`/`ncclRecv` + `bf16_add_inplace` path instead of native `ncclAllReduce`.
///
/// Single-host `world_size == 2` routes directly through native `ncclAllReduce` (fusing
/// PCIe P2P / NVLink reduction with zero extra kernel launch) unless forced via
/// `FLUX_FORCE_P2P_ALLREDUCE=1`. Multi-host `world_size == 2` keeps the RoCE fast path.
#[must_use]
pub(crate) const fn should_use_roce_send_recv_allreduce(ctx: AllReduceDispatchCtx) -> bool {
    if ctx.world_size != 2 || ctx.bytes == 0 || ctx.bytes > RECV_BUFFER_SIZE {
        return false;
    }
    if ctx.recv_buffer == 0 || ctx.add_kernel == 0 {
        return false;
    }
    match (ctx.locality, ctx.p2p_override) {
        (HostLocality::MultiHost, _) => true,
        (HostLocality::SingleHost, P2pAllReduceOverride::ForceP2p) => true,
        (HostLocality::SingleHost, P2pAllReduceOverride::Auto) => false,
    }
}

/// Pure decision helper: returns the `(key, value)` pairs from [`CX7_ROCE_DEFAULT_ENVS`]
/// that are currently unset according to `lookup`.
#[must_use]
pub(crate) fn resolve_cx7_roce_defaults<F>(lookup: F) -> Vec<(&'static str, &'static str)>
where
    F: Fn(&str) -> Option<String>,
{
    CX7_ROCE_DEFAULT_ENVS
        .into_iter()
        .filter(|(key, _)| lookup(key).is_none())
        .collect()
}

/// Populate ConnectX-7 RoCE NCCL defaults for any unset environment variables when
/// running on a multi-host fabric (or when explicitly enabled via `FLUX_CX7_ROCE=1`).
///
/// Skipped on single-host intra-node multi-GPU runs so NCCL retains `LL128` and
/// PCIe P2P / NVLink fast paths automatically.
///
/// # Safety
/// Must only be called during single-threaded process startup before spawning any
/// background threads (including Tokio workers, rayon pools, or CUDA/NCCL driver
/// threads) because mutating the process environment (`std::env::set_var`) is
/// undefined behavior if any other thread concurrently reads environment variables.
pub unsafe fn apply_cx7_roce_env_defaults(master_addr: &str) {
    let lookup = |k: &str| std::env::var(k).ok();
    let locality = resolve_host_locality(master_addr, lookup);
    if !should_apply_cx7_roce_defaults(locality, lookup) {
        tracing::info!(
            "NCCL single-host/non-RoCE mode ({locality:?}): skipping CX-7 RoCE defaults \
             (preserving LL128 / PCIe P2P / NVLink)"
        );
        return;
    }
    for (key, value) in resolve_cx7_roce_defaults(lookup) {
        // SAFETY: Caller upholds the `# Safety` contract that this process is still
        // single-threaded with no concurrent `getenv`/env readers.
        unsafe { std::env::set_var(key, value) };
        tracing::info!("NCCL CX-7 RoCE default applied: {key}={value}");
    }
}

/// Log ConnectX-7 RoCE NCCL default tuning status without mutating process environment
/// variables from multi-threaded library code.
fn log_cx7_roce_defaults_status(locality: HostLocality) {
    let lookup = |k: &str| std::env::var(k).ok();
    if !should_apply_cx7_roce_defaults(locality, lookup) {
        tracing::info!(
            "NCCL single-host/non-RoCE mode ({locality:?}): skipping CX-7 RoCE defaults \
             (preserving LL128 / PCIe P2P / NVLink)"
        );
        return;
    }
    let unset = resolve_cx7_roce_defaults(lookup);
    if !unset.is_empty() {
        let recommended: Vec<String> = unset.into_iter().map(|(k, v)| format!("{k}={v}")).collect();
        tracing::info!(
            "NCCL multi-host CX-7 RoCE tuning env vars unset ({}); set them in the launcher \
             or call `spark_comm::nccl_backend::apply_cx7_roce_env_defaults` during \
             single-threaded startup before spawning threads",
            recommended.join(", ")
        );
    }
}

/// Bounded poll loop for non-blocking `ncclCommInitRankConfig` completion.
///
/// Aborts the communicator if initialization reports an error or exceeds `timeout`.
fn poll_comm_init_until_ready<FPoll, FAbort>(
    initial: NcclResult,
    timeout: std::time::Duration,
    poll_interval: std::time::Duration,
    context: &str,
    mut poll_async_error: FPoll,
    mut abort_comm: FAbort,
) -> Result<()>
where
    FPoll: FnMut() -> std::result::Result<NcclResult, NcclResult>,
    FAbort: FnMut(),
{
    match initial {
        NcclResult::Success => return Ok(()),
        NcclResult::InProgress => {}
        other => {
            abort_comm();
            return nccl::check_nccl(other, context);
        }
    }

    let deadline = std::time::Instant::now() + timeout;
    loop {
        match poll_async_error() {
            Ok(NcclResult::Success) => return Ok(()),
            Ok(NcclResult::InProgress) => {
                if std::time::Instant::now() >= deadline {
                    abort_comm();
                    anyhow::bail!(
                        "NCCL timeout in {context}: communicator remained InProgress after {}s",
                        timeout.as_secs()
                    );
                }
                if !poll_interval.is_zero() {
                    std::thread::sleep(poll_interval);
                }
            }
            Ok(err_res) => {
                abort_comm();
                return nccl::check_nccl(err_res, context);
            }
            Err(query_err) => {
                abort_comm();
                return nccl::check_nccl(query_err, "ncclCommGetAsyncError (init poll)");
            }
        }
    }
}

/// Non-`Copy` owning wrapper around a live non-null `NcclComm` handle.
///
/// Stored inside `Mutex<Option<OwnedNcclComm>>` on [`NcclBackend`] so no caller
/// can copy the raw pointer out and drop the `MutexGuard` before invoking NCCL
/// FFI. All collectives borrow `&OwnedNcclComm` while holding the mutex guard,
/// preventing concurrent NCCL calls and `ncclCommAbort` use-after-free.
pub(super) struct OwnedNcclComm(NcclComm);

// SAFETY: `OwnedNcclComm` exclusively owns a live non-null `NcclComm` handle
// and is non-`Copy` and `!Sync`. Storing it in `Mutex<Option<OwnedNcclComm>>`
// serializes all NCCL operations on the communicator across threads and ensures
// `ncclCommAbort` / `ncclCommDestroy` can never race with active collectives.
unsafe impl Send for OwnedNcclComm {}

impl OwnedNcclComm {
    /// Wrap a non-null `NcclComm` returned by `ncclCommInitRankConfig` / `ncclCommInitRank`.
    fn new(raw: NcclComm) -> Result<Self> {
        anyhow::ensure!(!raw.is_null(), "NCCL returned a null communicator handle");
        Ok(Self(raw))
    }

    /// Borrow the underlying raw handle while the enclosing `MutexGuard` is held.
    pub(super) fn as_raw(&self) -> NcclComm {
        self.0
    }

    /// Destroy this communicator via `ncclCommDestroy`, consuming ownership.
    fn destroy(self) -> NcclResult {
        // SAFETY: `self` exclusively owns a live non-null `NcclComm`; consuming
        // `self` guarantees it is destroyed at most once.
        unsafe { nccl::ncclCommDestroy(self.0) }
    }

    /// Abort this communicator via `ncclCommAbort`, consuming ownership.
    fn abort(self) -> NcclResult {
        // SAFETY: `self` exclusively owns a live non-null `NcclComm`; consuming
        // `self` guarantees it is aborted at most once.
        unsafe { nccl::ncclCommAbort(self.0) }
    }
}

/// Validate and remove a tracked handle/pointer from `registry` before passing
/// it to FFI teardown (`ncclCommDeregister` / `ncclMemFree`).
pub(super) fn take_tracked_handle(
    registry: &Mutex<HashSet<u64>>,
    handle: u64,
    label: &str,
) -> Result<()> {
    anyhow::ensure!(handle != 0, "{label}: null handle");
    let removed = registry.lock().remove(&handle);
    anyhow::ensure!(
        removed,
        "{label}: handle {handle:#x} is not a live tracked allocation/registration"
    );
    Ok(())
}

/// Size in bytes of the 1-element `Int32` device scratch word used by `barrier()`.
const BARRIER_BUF_BYTES: usize = std::mem::size_of::<i32>();

/// NCCL communication backend for multi-GPU / multi-node EP.
pub struct NcclBackend {
    /// Protected for serialized NCCL access and abort-and-reconnect.
    /// Every NCCL FFI call holds this mutex guard for its entire duration.
    comm: Mutex<Option<OwnedNcclComm>>,
    rank: usize,
    world_size: usize,
    /// Dedicated stream for NCCL collectives (separate from compute).
    comm_stream: u64,
    /// Event: signals "MoE compute done" on the compute stream.
    compute_done_event: u64,
    /// Event: signals "all_reduce done" on the comm stream.
    comm_done_event: u64,
    /// Legacy compute stream for synchronous barrier/broadcast.
    legacy_stream: u64,
    /// Persistent 4-byte zeroed device scratch word for blocking 1-element `barrier()`.
    barrier_buf: u64,
    /// Persistent receive buffer for 2-rank send/recv all-reduce.
    recv_buffer: u64,
    /// Internal `ncclCommRegister` handle for `recv_buffer` (deregistered on abort/Drop).
    recv_buffer_handle: Mutex<Option<u64>>,
    /// Live handles returned by `register_buffer` (validated before `deregister_buffer` and cleaned up on Drop).
    registered_handles: Mutex<HashSet<u64>>,
    /// Live device pointers returned by `symmetric_alloc` (validated before `symmetric_free` and cleaned up on Drop).
    symmetric_allocs: Mutex<HashSet<u64>>,
    /// Kernel handle for bf16_add_inplace (set via set_add_kernel).
    add_kernel: AtomicU64,
    /// Whether the communicator is in a degraded/unhealthy state.
    unhealthy: AtomicBool,
    /// Number of successful reconnections (for diagnostics).
    reconnect_count: AtomicU64,
    /// Bootstrap parameters stored for reconnection.
    master_addr: String,
    master_port: u16,
    /// Resolved host locality (`SingleHost` vs `MultiHost`).
    locality: HostLocality,
    /// Override mode for 2-rank P2P `send/recv + bf16_add_inplace` all-reduce.
    p2p_override: P2pAllReduceOverride,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NcclBackend>();
};

impl NcclBackend {
    /// Borrow the live [`OwnedNcclComm`] from an already-acquired `self.comm` mutex guard,
    /// failing fast if the communicator was aborted and has not been reconnected.
    pub(super) fn require_comm<'a>(
        guard: &'a Option<OwnedNcclComm>,
        context: &str,
    ) -> Result<&'a OwnedNcclComm> {
        guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "NCCL communicator is not initialized (aborted or awaiting reconnect) in {context}"
            )
        })
    }

    /// Initialize a communicator using `ncclCommInitRankConfig` with `NcclConfig::non_blocking()`
    /// and a bounded async-error poll loop.
    ///
    /// Falls back cleanly to blocking `ncclCommInitRank` only if the linked `libnccl` rejects
    /// the `NcclConfig` struct ABI (`InvalidArgument`).
    fn init_comm_nonblocking(
        world_size: usize,
        unique_id: NcclUniqueId,
        rank: usize,
        context: &str,
    ) -> Result<OwnedNcclComm> {
        let mut comm: NcclComm = ptr::null_mut();
        let config = nccl::NcclConfig::non_blocking();
        tracing::debug!(
            "{context}: ncclCommInitRankConfig (size={}, version={})",
            config.size,
            config.version
        );
        // SAFETY: `&mut comm` and `&config` are valid aligned pointers matching
        // `ncclConfig_v22800` ABI; `unique_id` was generated by `ncclGetUniqueId`.
        let result = unsafe {
            nccl::ncclCommInitRankConfig(
                &mut comm,
                world_size as i32,
                unique_id,
                rank as i32,
                &config,
            )
        };
        if result == NcclResult::InvalidArgument {
            tracing::warn!(
                "{context}: ncclCommInitRankConfig returned InvalidArgument \
                 (NcclConfig ABI mismatch with linked libnccl; size={}, version={}); \
                 falling back to ncclCommInitRank",
                config.size,
                config.version
            );
            if !comm.is_null() {
                // SAFETY: `comm` was partially initialized by `ncclCommInitRankConfig`
                // and is immediately nulled so it is never reused.
                unsafe { nccl::ncclCommAbort(comm) };
                comm = ptr::null_mut();
            }
            // SAFETY: `&mut comm` is a valid out-pointer and `unique_id` is a valid NCCL unique ID.
            let fallback = unsafe {
                nccl::ncclCommInitRank(&mut comm, world_size as i32, unique_id, rank as i32)
            };
            nccl::check_nccl(fallback, context)?;
            return OwnedNcclComm::new(comm);
        }
        if result == NcclResult::InProgress && comm.is_null() {
            anyhow::bail!("{context}: ncclCommInitRankConfig returned InProgress with null comm");
        }
        poll_comm_init_until_ready(
            result,
            std::time::Duration::from_secs(COMM_INIT_TIMEOUT_SECS),
            COMM_INIT_POLL_INTERVAL,
            context,
            || {
                let mut async_err = NcclResult::Success;
                // SAFETY: `comm` is a live handle being polled on this thread before publication,
                // and `&mut async_err` is a valid aligned `NcclResult` out-pointer.
                let query = unsafe { nccl::ncclCommGetAsyncError(comm, &mut async_err) };
                if query != NcclResult::Success {
                    Err(query)
                } else {
                    Ok(async_err)
                }
            },
            || {
                if !comm.is_null() {
                    // SAFETY: `comm` is exclusively owned locally and aborted once on error/timeout.
                    unsafe { nccl::ncclCommAbort(comm) };
                }
            },
        )?;
        OwnedNcclComm::new(comm)
    }

    /// Initialize NCCL with TCP bootstrap.
    ///
    /// Rank 0 listens on `master_addr:master_port`, generates a unique ID,
    /// and sends it to all connecting ranks. All ranks then call
    /// `ncclCommInitRankConfig` (non-blocking with bounded poll) which
    /// internally synchronizes.
    pub fn new(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
        stream: u64,
    ) -> Result<Self> {
        let env_lookup = |k: &str| std::env::var(k).ok();
        let override_mode = resolve_locality_override(env_lookup);
        let initial_locality = resolve_host_locality(master_addr, env_lookup);
        let p2p_override = resolve_p2p_allreduce_override(env_lookup);

        let (unique_id, locality) = if rank == 0 {
            Self::bootstrap_rank0(
                master_addr,
                master_port,
                world_size,
                initial_locality,
                override_mode,
            )?
        } else {
            log_cx7_roce_defaults_status(initial_locality);
            Self::log_nccl_env_vars();
            let id = Self::receive_id(master_addr, master_port)?;
            (id, initial_locality)
        };

        tracing::info!(
            "NCCL communicator topology: rank={rank}/{world_size}, master={master_addr}:{master_port}, \
             locality={locality:?}, p2p_override={p2p_override:?}"
        );

        let comm =
            Self::init_comm_nonblocking(world_size, unique_id, rank, "ncclCommInitRankConfig")?;

        let comm_stream = match nccl::create_stream() {
            Ok(s) => s,
            Err(err) => {
                comm.destroy();
                return Err(err);
            }
        };
        let compute_done_event = match nccl::create_event() {
            Ok(ev) => ev,
            Err(err) => {
                // SAFETY: `comm_stream` was just created above and is not shared.
                unsafe { nccl::destroy_stream(comm_stream) };
                comm.destroy();
                return Err(err);
            }
        };
        let comm_done_event = match nccl::create_event() {
            Ok(ev) => ev,
            Err(err) => {
                // SAFETY: `compute_done_event` and `comm_stream` were just created
                // above and have no remaining uses.
                unsafe {
                    nccl::destroy_event(compute_done_event);
                    nccl::destroy_stream(comm_stream);
                }
                comm.destroy();
                return Err(err);
            }
        };

        // Allocate and zero-initialize the 4-byte device scratch word used by `barrier()`.
        let mut barrier_buf: u64 = 0;
        // SAFETY: `&mut barrier_buf` is a valid aligned out-pointer in the active CUDA context.
        let barrier_alloc_status = unsafe { cuMemAlloc_v2(&mut barrier_buf, BARRIER_BUF_BYTES) };
        if barrier_alloc_status != 0 {
            // SAFETY: Handles were created above and are exclusively owned here.
            unsafe {
                nccl::destroy_event(compute_done_event);
                nccl::destroy_event(comm_done_event);
                nccl::destroy_stream(comm_stream);
            }
            comm.destroy();
            anyhow::bail!("cuMemAlloc_v2 for barrier_buf failed: status {barrier_alloc_status}");
        }
        // SAFETY: `barrier_buf` was just allocated with `BARRIER_BUF_BYTES` (4) bytes.
        if let Err(err) = unsafe { nccl::memcpy_h2d(barrier_buf, &0i32.to_ne_bytes()) } {
            // SAFETY: `barrier_buf` and CUDA handles are exclusively owned locally.
            unsafe {
                cuMemFree_v2(barrier_buf);
                nccl::destroy_event(compute_done_event);
                nccl::destroy_event(comm_done_event);
                nccl::destroy_stream(comm_stream);
            }
            comm.destroy();
            return Err(err.context("failed to zero-initialize NCCL barrier scratch buffer"));
        }

        // Allocate and register persistent recv buffer only when the RoCE / forced P2P
        // staging path is active; single-host runs use native `ncclAllReduce` directly.
        let mut recv_buffer: u64 = 0;
        let mut recv_buffer_handle: Option<u64> = None;
        if should_allocate_recv_buffer(world_size, locality, p2p_override) {
            // SAFETY: `&mut recv_buffer` is a valid aligned out-pointer in the active CUDA context.
            let status = unsafe { cuMemAlloc_v2(&mut recv_buffer, RECV_BUFFER_SIZE) };
            if status != 0 {
                // SAFETY: Handles and `barrier_buf` were created above and are exclusively owned here.
                unsafe {
                    cuMemFree_v2(barrier_buf);
                    nccl::destroy_event(compute_done_event);
                    nccl::destroy_event(comm_done_event);
                    nccl::destroy_stream(comm_stream);
                }
                comm.destroy();
                anyhow::bail!("cuMemAlloc_v2 for recv_buffer failed: status {status}");
            }
            // Register recv buffer with NCCL for IB memory caching and track handle for Drop.
            let mut handle: *mut c_void = ptr::null_mut();
            // SAFETY: `comm` is live, `recv_buffer` is a valid `RECV_BUFFER_SIZE`-byte
            // device allocation, and `&mut handle` is a valid aligned out-pointer.
            let result = unsafe {
                nccl::ncclCommRegister(
                    comm.as_raw(),
                    recv_buffer as *mut c_void,
                    RECV_BUFFER_SIZE,
                    &mut handle,
                )
            };
            if result != NcclResult::Success {
                tracing::warn!("ncclCommRegister for recv_buffer failed (non-fatal): {result:?}");
            } else {
                if !handle.is_null() {
                    recv_buffer_handle = Some(handle as u64);
                }
                tracing::info!(
                    "Registered recv_buffer ({} KB) with NCCL (world_size={world_size})",
                    RECV_BUFFER_SIZE / 1024
                );
            }
        } else if world_size >= 2 {
            tracing::info!(
                "Skipping 64 MiB recv_buffer allocation on single-host world_size={world_size} \
                 (native ncclAllReduce active)"
            );
        }

        Ok(Self {
            comm: Mutex::new(Some(comm)),
            rank,
            world_size,
            comm_stream,
            compute_done_event,
            comm_done_event,
            legacy_stream: stream,
            barrier_buf,
            recv_buffer,
            recv_buffer_handle: Mutex::new(recv_buffer_handle),
            registered_handles: Mutex::new(HashSet::new()),
            symmetric_allocs: Mutex::new(HashSet::new()),
            add_kernel: AtomicU64::new(0),
            unhealthy: AtomicBool::new(false),
            reconnect_count: AtomicU64::new(0),
            master_addr: master_addr.to_owned(),
            master_port,
            locality,
            p2p_override,
        })
    }

    /// Returns `true` when BF16 `all_reduce` should take the 2-rank RoCE `send/recv`
    /// + `bf16_add_inplace` path instead of native `ncclAllReduce`.
    pub(super) fn use_roce_send_recv_allreduce(&self, bytes: usize) -> bool {
        should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            world_size: self.world_size,
            locality: self.locality,
            p2p_override: self.p2p_override,
            bytes,
            recv_buffer: self.recv_buffer,
            add_kernel: self.add_kernel.load(Ordering::Relaxed),
        })
    }

    /// Log NCCL-related environment variables at init time for diagnostics.
    fn log_nccl_env_vars() {
        let vars = [
            "FLUX_SINGLE_HOST",
            "FLUX_COMM_SINGLE_HOST",
            "FLUX_CX7_ROCE",
            "FLUX_FORCE_P2P_ALLREDUCE",
            "NCCL_TIMEOUT",
            "NCCL_WATCHDOG_TIMEOUT",
            "NCCL_IB_DISABLE",
            "NCCL_IB_HCA",
            "NCCL_IB_GID_INDEX",
            "NCCL_IB_TIMEOUT",
            "NCCL_IB_RETRY_CNT",
            "NCCL_IB_QPS_PER_CONNECTION",
            "NCCL_IB_SPLIT_DATA_ON_QPS",
            "NCCL_CROSS_NIC",
            "NCCL_NET_GDR_LEVEL",
            "NCCL_SOCKET_IFNAME",
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_DEBUG",
        ];
        for var in &vars {
            match std::env::var(var) {
                Ok(val) => tracing::info!("NCCL env: {var}={val}"),
                Err(_) => tracing::debug!("NCCL env: {var} not set"),
            }
        }
    }

    /// Check the NCCL communicator for asynchronous errors.
    ///
    /// Returns `true` if the communicator is healthy (no async errors).
    /// If an async error is detected, sets the unhealthy flag and returns `false`.
    pub(super) fn check_async_error(&self, comm: &OwnedNcclComm) -> bool {
        let mut async_err = NcclResult::Success;
        // SAFETY: `comm` is borrowed while the caller holds `self.comm.lock()`,
        // guaranteeing a live non-null `NcclComm` with serialized access, and
        // `&mut async_err` is a valid aligned `NcclResult` out-pointer.
        let result = unsafe { nccl::ncclCommGetAsyncError(comm.as_raw(), &mut async_err) };
        if result != NcclResult::Success {
            tracing::error!(
                "ncclCommGetAsyncError call itself failed: {result:?} \
                 — marking unhealthy"
            );
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        if async_err != NcclResult::Success {
            tracing::error!("NCCL async error detected: {async_err:?} — marking unhealthy");
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        true
    }

    /// Abort the current communicator and re-initialize via TCP bootstrap.
    ///
    /// Both ranks must call this concurrently (the reconnect protocol
    /// mirrors the initial bootstrap). After reconnect, the recv_buffer
    /// is re-registered with the new communicator.
    fn reconnect_inner(&self) -> Result<()> {
        let mut comm_guard = self.comm.lock();

        // Double-check: another thread may have already reconnected.
        if comm_guard.is_some() && !self.unhealthy.load(Ordering::Acquire) {
            tracing::info!("NCCL communicator already recovered by another thread");
            return Ok(());
        }

        let attempt = self.reconnect_count.load(Ordering::Relaxed) + 1;
        tracing::warn!(
            "NCCL reconnect: aborting old communicator \
             (rank={}, reconnect #{})",
            self.rank,
            attempt,
        );

        // Immediately mark unhealthy, clear stale registration handles, and take
        // ownership of the dead communicator *before* any fallible step (`?`) so
        // an error during re-bootstrap never leaves a dangling `NcclComm` or stale
        // registration handles for subsequent retries or `Drop`.
        self.unhealthy.store(true, Ordering::Release);
        {
            let mut handles = self.registered_handles.lock();
            if !handles.is_empty() {
                tracing::warn!(
                    "Clearing {} stale registered buffer handles before reconnect",
                    handles.len()
                );
                handles.clear();
            }
        }
        *self.recv_buffer_handle.lock() = None;

        if let Some(old_comm) = comm_guard.take() {
            let result = old_comm.abort();
            if result != NcclResult::Success {
                tracing::warn!("ncclCommAbort returned {result:?} (proceeding anyway)");
            }
        }

        // Re-bootstrap: rank 0 distributes a new unique ID.
        // Use master_port + 1 to avoid bind conflicts with a lingering listener.
        let reconnect_port = self.master_port.wrapping_add(1);
        let unique_id = if self.rank == 0 {
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, &self.master_addr, reconnect_port, self.world_size)?;
            id
        } else {
            Self::receive_id(&self.master_addr, reconnect_port)?
        };

        let new_comm = Self::init_comm_nonblocking(
            self.world_size,
            unique_id,
            self.rank,
            "ncclCommInitRankConfig (reconnect)",
        )?;

        // Re-register recv buffer with the new communicator and track handle for Drop.
        if should_allocate_recv_buffer(self.world_size, self.locality, self.p2p_override)
            && self.recv_buffer != 0
        {
            let mut handle: *mut c_void = ptr::null_mut();
            // SAFETY: `new_comm` is a live owned communicator, `self.recv_buffer` is a
            // valid `RECV_BUFFER_SIZE`-byte device allocation, and `&mut handle` is a
            // valid aligned out-pointer.
            let result = unsafe {
                nccl::ncclCommRegister(
                    new_comm.as_raw(),
                    self.recv_buffer as *mut c_void,
                    RECV_BUFFER_SIZE,
                    &mut handle,
                )
            };
            if result != NcclResult::Success {
                tracing::warn!(
                    "ncclCommRegister for recv_buffer after reconnect \
                     failed: {result:?}"
                );
            } else {
                if !handle.is_null() {
                    *self.recv_buffer_handle.lock() = Some(handle as u64);
                }
                tracing::info!("Re-registered recv_buffer after reconnect");
            }
        }

        *comm_guard = Some(new_comm);
        self.unhealthy.store(false, Ordering::Release);
        self.reconnect_count.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            "NCCL reconnect successful (rank={}, total reconnects={})",
            self.rank,
            self.reconnect_count.load(Ordering::Relaxed),
        );

        Ok(())
    }

    /// 2-rank all-reduce using send/recv + local BF16 add while holding `self.comm.lock()`.
    ///
    /// For world_size == 2:
    ///   1. Group send+recv (one RDMA write each direction)
    ///   2. Local BF16 add: `ptr[i] += recv_buffer[i]`
    pub(super) fn all_reduce_2rank_locked(
        &self,
        comm: &OwnedNcclComm,
        ptr: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            bytes.is_multiple_of(2),
            "all_reduce_2rank: BF16 byte count {bytes} must be even"
        );
        anyhow::ensure!(
            bytes <= RECV_BUFFER_SIZE,
            "all_reduce_2rank: payload {bytes} exceeds RECV_BUFFER_SIZE ({RECV_BUFFER_SIZE})"
        );
        anyhow::ensure!(
            self.recv_buffer != 0,
            "all_reduce_2rank: recv_buffer is not allocated (locality={:?})",
            self.locality
        );
        let count = bytes / 2; // BF16 element count
        let partner = (1 - self.rank) as i32;

        // Paired send/recv in a group (single NCCL launch) with guaranteed ncclGroupEnd.
        // SAFETY: `ncclGroupStart` begins a thread-local NCCL group closed below by `ncclGroupEnd`.
        nccl::check_nccl(unsafe { nccl::ncclGroupStart() }, "ncclGroupStart")?;
        let send_recv_res = (|| -> Result<()> {
            nccl::check_nccl(
                // SAFETY: `comm` is borrowed under `self.comm.lock()`, `ptr` is a caller-provided
                // device buffer of at least `bytes` (`count` BF16 elements), and `stream` is live.
                unsafe {
                    nccl::ncclSend(
                        ptr as *const c_void,
                        count,
                        NcclDataType::Bfloat16,
                        partner,
                        comm.as_raw(),
                        stream,
                    )
                },
                "ncclSend",
            )?;
            nccl::check_nccl(
                // SAFETY: `comm` is borrowed under `self.comm.lock()`, `self.recv_buffer` is an
                // owned `RECV_BUFFER_SIZE`-byte device buffer (`bytes <= RECV_BUFFER_SIZE`)
                // serialized by `self.comm.lock()`, and `stream` is live.
                unsafe {
                    nccl::ncclRecv(
                        self.recv_buffer as *mut c_void,
                        count,
                        NcclDataType::Bfloat16,
                        partner,
                        comm.as_raw(),
                        stream,
                    )
                },
                "ncclRecv",
            )
        })();
        // SAFETY: Balances the `ncclGroupStart()` called unconditionally above.
        let end_res = nccl::check_nccl(unsafe { nccl::ncclGroupEnd() }, "ncclGroupEnd");
        send_recv_res.and(end_res)?;

        // Check for async errors after the group operation.
        self.check_async_error(comm);

        // Local BF16 addition: ptr[i] += recv_buffer[i]
        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel != 0 {
            let threads: u32 = 256;
            let blocks: u32 = (count as u32).div_ceil(threads);
            let mut p_dst = ptr;
            let mut p_src = self.recv_buffer;
            let mut p_n = count as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_dst as *mut u64 as *mut c_void,
                &mut p_src as *mut u64 as *mut c_void,
                &mut p_n as *mut i32 as *mut c_void,
            ];
            // SAFETY: `kernel` was registered via `set_add_kernel` as the `bf16_add_inplace`
            // `CUfunction`, `params` points to live stack locals matching `(u64, u64, i32)` for
            // the duration of the launch call, and `self.recv_buffer` is serialized by `self.comm`.
            let status = unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if status != 0 {
                anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
            }
        } else {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }

        Ok(())
    }

    fn generate_unique_id() -> Result<NcclUniqueId> {
        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        // SAFETY: `&mut id` is a valid aligned writable `NcclUniqueId` out-pointer.
        let result = unsafe { nccl::ncclGetUniqueId(&mut id) };
        nccl::check_nccl(result, "ncclGetUniqueId")?;
        Ok(id)
    }

    fn rank0_bind_addr(master_addr: &str, port: u16) -> String {
        let trimmed = master_addr.trim();
        let inner = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(trimmed);
        if inner.parse::<std::net::Ipv6Addr>().is_ok() {
            format!("[::]:{port}")
        } else {
            format!("0.0.0.0:{port}")
        }
    }

    fn format_bootstrap_target(addr: &str, port: u16) -> String {
        let trimmed = addr.trim();
        if trimmed.parse::<std::net::Ipv6Addr>().is_ok() {
            format!("[{trimmed}]:{port}")
        } else {
            format!("{trimmed}:{port}")
        }
    }

    /// Bootstrap rank 0: bind TCP listener, accept all worker connections before
    /// finalizing host locality and calling `ncclGetUniqueId()`, and distribute
    /// the generated unique ID to all worker ranks.
    fn bootstrap_rank0(
        master_addr: &str,
        master_port: u16,
        world_size: usize,
        initial_locality: HostLocality,
        override_mode: LocalityOverride,
    ) -> Result<(NcclUniqueId, HostLocality)> {
        if world_size <= 1
            || !initial_locality.is_single_host()
            || !matches!(override_mode, LocalityOverride::Auto)
        {
            log_cx7_roce_defaults_status(initial_locality);
            Self::log_nccl_env_vars();
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, master_addr, master_port, world_size)?;
            return Ok((id, initial_locality));
        }

        // Auto-detected loopback `master_addr` (e.g. CLI default `127.0.0.1`).
        // Accept all `world_size - 1` workers first so `locality` is finalized
        // across every peer before logging tuning status and calling `ncclGetUniqueId()`.
        let bind_addr = Self::rank0_bind_addr(master_addr, master_port);
        let listener = TcpListener::bind(&bind_addr)
            .with_context(|| format!("Rank 0: failed to bind {bind_addr}"))?;
        tracing::info!(
            "Rank 0: waiting for {} worker(s) on {}",
            world_size - 1,
            bind_addr
        );

        let mut accepted_streams: Vec<(TcpStream, std::net::SocketAddr)> =
            Vec::with_capacity(world_size - 1);
        let mut locality = initial_locality;
        for i in 0..(world_size - 1) {
            let (stream, peer_addr) = listener.accept().context("Rank 0: accept failed")?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
            let updated = refine_rank0_locality(locality, override_mode, peer_addr.ip());
            if updated != locality {
                locality = updated;
                tracing::info!(
                    "Rank 0: remote peer {peer_addr} connected on worker {} despite loopback \
                     master_addr={master_addr}; switching to multi-host mode",
                    i + 1
                );
            }
            accepted_streams.push((stream, peer_addr));
        }

        log_cx7_roce_defaults_status(locality);
        Self::log_nccl_env_vars();
        let id = Self::generate_unique_id()?;
        for (i, (mut stream, peer_addr)) in accepted_streams.into_iter().enumerate() {
            stream
                .write_all(&id.internal)
                .context("Rank 0: failed to send unique ID")?;
            tracing::info!("Rank 0: sent unique ID to worker {} ({peer_addr})", i + 1);
        }
        Ok((id, locality))
    }

    /// Rank 0: listen, accept (world_size - 1) connections, send the unique ID.
    fn distribute_id(id: &NcclUniqueId, addr: &str, port: u16, world_size: usize) -> Result<()> {
        let bind_addr = Self::rank0_bind_addr(addr, port);
        let listener = TcpListener::bind(&bind_addr)
            .with_context(|| format!("Rank 0: failed to bind {bind_addr}"))?;
        tracing::info!(
            "Rank 0: waiting for {} worker(s) on {}",
            world_size - 1,
            bind_addr
        );

        for i in 0..(world_size - 1) {
            let (mut stream, peer_addr) = listener.accept().context("Rank 0: accept failed")?;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
            stream
                .write_all(&id.internal)
                .context("Rank 0: failed to send unique ID")?;
            tracing::info!("Rank 0: sent unique ID to worker {} ({})", i + 1, peer_addr);
        }
        Ok(())
    }

    /// Non-zero rank: connect to rank 0, receive the unique ID.
    fn receive_id(addr: &str, port: u16) -> Result<NcclUniqueId> {
        let target = Self::format_bootstrap_target(addr, port);
        tracing::info!("Rank N: connecting to master at {target}");

        // Retry with backoff — rank 0 may not be listening yet.
        // Large models (>100B) take 3-5 minutes to load weights before opening
        // the NCCL master port. The previous 30s ceiling consistently timed
        // out on 122B-ep2 / nemotron-super-120B-ep2 sweep rounds. 10 minutes
        // gives any model time to load shards on a Spark; if rank 0 actually
        // crashed the worker still surfaces the failure — just later.
        const MAX_ATTEMPTS: u32 = 600; // 10 minutes at 1s each
        let mut stream = None;
        for attempt in 0..MAX_ATTEMPTS {
            match TcpStream::connect(&target) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => {
                    if attempt + 1 < MAX_ATTEMPTS {
                        if attempt < 30 || attempt.is_multiple_of(30) {
                            tracing::info!(
                                "Connect attempt {}/{MAX_ATTEMPTS}: {e}, retrying in 1s (rank 0 may be loading weights)",
                                attempt + 1
                            );
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    } else {
                        return Err(e).with_context(|| {
                            format!(
                                "Failed to connect to master at {target} \
                                 after {MAX_ATTEMPTS} attempts (~{} minutes)",
                                MAX_ATTEMPTS / 60
                            )
                        });
                    }
                }
            }
        }

        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        let mut connected = stream
            .ok_or_else(|| anyhow::anyhow!("master TCP stream unavailable after connect loop"))?;
        let _ = connected.set_nodelay(true);
        let _ = connected.set_read_timeout(Some(std::time::Duration::from_secs(120)));
        connected
            .read_exact(&mut id.internal)
            .context("Failed to receive unique ID from rank 0")?;
        tracing::info!("Received NCCL unique ID from master");
        Ok(id)
    }
}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        let comm = self.comm.get_mut().take();
        // Deregister all NCCL-registered buffers before destroying the communicator.
        let mut handles = self.registered_handles.lock();
        let mut recv_handle = self.recv_buffer_handle.lock();
        if let Some(ref live_comm) = comm {
            if let Some(h) = recv_handle.take() {
                // SAFETY: `live_comm` is still live and `h` was returned by
                // `ncclCommRegister` on this communicator and not yet deregistered.
                unsafe { nccl::ncclCommDeregister(live_comm.as_raw(), h as *mut c_void) };
            }
            for handle in handles.drain() {
                // SAFETY: `live_comm` is still live and each `handle` in `registered_handles`
                // was returned by `ncclCommRegister` on this communicator and not yet deregistered.
                unsafe { nccl::ncclCommDeregister(live_comm.as_raw(), handle as *mut c_void) };
            }
        } else {
            *recv_handle = None;
            handles.clear();
        }
        drop(recv_handle);
        drop(handles);

        // Free any remaining tracked symmetric-memory allocations.
        let mut sym = self.symmetric_allocs.lock();
        if !sym.is_empty() {
            tracing::warn!(
                "NcclBackend::drop freeing {} unfreed symmetric_alloc buffer(s)",
                sym.len()
            );
            for ptr in sym.drain() {
                // SAFETY: Each pointer in `symmetric_allocs` was returned by `nccl_mem_alloc`
                // and has not yet been freed via `symmetric_free`.
                let _ = unsafe { nccl::nccl_mem_free(ptr as *mut c_void) };
            }
        }
        drop(sym);

        // Free owned barrier scratch word and recv buffer.
        if self.barrier_buf != 0 {
            // SAFETY: `self.barrier_buf` was allocated via `cuMemAlloc_v2` in `NcclBackend::new`
            // and is freed once here during `Drop`.
            unsafe { cuMemFree_v2(self.barrier_buf) };
        }
        if self.recv_buffer != 0 {
            // SAFETY: `self.recv_buffer` was allocated via `cuMemAlloc_v2` in `NcclBackend::new`
            // and is freed once here during `Drop`.
            unsafe { cuMemFree_v2(self.recv_buffer) };
        }
        // SAFETY: `NcclBackend` owns `compute_done_event`, `comm_done_event`,
        // and `comm_stream` and destroys each exactly once during Drop.
        unsafe {
            nccl::destroy_event(self.compute_done_event);
            nccl::destroy_event(self.comm_done_event);
            nccl::destroy_stream(self.comm_stream);
        }
        if let Some(live_comm) = comm {
            let _ = live_comm.destroy();
        }
    }
}

mod comm_impl;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_resolve_cx7_roce_defaults_sets_only_unset_vars() {
        let all_missing = resolve_cx7_roce_defaults(|_| None);
        assert_eq!(
            all_missing,
            vec![
                ("NCCL_IB_QPS_PER_CONNECTION", "4"),
                ("NCCL_IB_SPLIT_DATA_ON_QPS", "0"),
                ("NCCL_PROTO", "LL,Simple"),
            ]
        );

        let partial = resolve_cx7_roce_defaults(|key| match key {
            "NCCL_PROTO" => Some("Simple".to_string()),
            "NCCL_IB_QPS_PER_CONNECTION" => Some("8".to_string()),
            _ => None,
        });
        assert_eq!(partial, vec![("NCCL_IB_SPLIT_DATA_ON_QPS", "0")]);

        let all_set = resolve_cx7_roce_defaults(|_| Some("custom".to_string()));
        assert!(all_set.is_empty());
    }

    #[test]
    fn test_is_single_host_addr_loopback_vs_routable_and_wildcard() {
        for addr in [
            "127.0.0.1",
            " 127.0.0.1 ",
            "127.0.1.1",
            "localhost",
            "LOCALHOST",
            "::1",
            "[::1]",
            "::ffff:127.0.0.1",
            "[::ffff:127.0.0.1]",
        ] {
            assert!(
                is_single_host_addr(addr),
                "expected loopback addr {addr:?} to be single-host"
            );
        }

        for addr in [
            "",
            "   ",
            "0.0.0.0",
            "::",
            "[::]",
            "10.0.0.1",
            "192.168.1.10",
            "172.16.0.5",
            "spark-node-0",
            "[127.0.0.1]",
            "[localhost]",
        ] {
            assert!(
                !is_single_host_addr(addr),
                "expected non-loopback/wildcard addr {addr:?} NOT to be single-host"
            );
        }
    }

    #[test]
    fn test_resolve_host_locality_and_rank0_peer_refinement() {
        let empty = |_: &str| None;
        assert_eq!(
            resolve_host_locality("127.0.0.1", empty),
            HostLocality::SingleHost
        );
        assert_eq!(
            resolve_host_locality("localhost", empty),
            HostLocality::SingleHost
        );
        assert_eq!(
            resolve_host_locality("0.0.0.0", empty),
            HostLocality::MultiHost
        );
        assert_eq!(
            resolve_host_locality("10.0.0.1", empty),
            HostLocality::MultiHost
        );

        // Explicit FLUX_SINGLE_HOST / FLUX_COMM_SINGLE_HOST overrides non-loopback addr.
        assert_eq!(
            resolve_host_locality("0.0.0.0", |k| {
                (k == "FLUX_SINGLE_HOST").then(|| "1".to_string())
            }),
            HostLocality::SingleHost
        );
        assert_eq!(
            resolve_host_locality("192.168.1.20", |k| {
                (k == "FLUX_COMM_SINGLE_HOST").then(|| "true".to_string())
            }),
            HostLocality::SingleHost
        );
        // Explicit FLUX_SINGLE_HOST=0 or FLUX_CX7_ROCE=1 forces MultiHost even on loopback.
        assert_eq!(
            resolve_host_locality("127.0.0.1", |k| {
                (k == "FLUX_SINGLE_HOST").then(|| "0".to_string())
            }),
            HostLocality::MultiHost
        );
        assert_eq!(
            resolve_host_locality("127.0.0.1", |k| {
                (k == "FLUX_CX7_ROCE").then(|| "1".to_string())
            }),
            HostLocality::MultiHost
        );

        // Rank 0 peer refinement: auto-detected loopback downgrades when remote peer connects.
        let loopback_v4: std::net::IpAddr = "127.0.0.1".parse().expect("valid v4");
        let loopback_v6: std::net::IpAddr = "::1".parse().expect("valid v6");
        let remote_lan: std::net::IpAddr = "10.0.0.2".parse().expect("valid lan");
        assert_eq!(
            refine_rank0_locality(
                HostLocality::SingleHost,
                LocalityOverride::Auto,
                loopback_v4
            ),
            HostLocality::SingleHost
        );
        assert_eq!(
            refine_rank0_locality(
                HostLocality::SingleHost,
                LocalityOverride::Auto,
                loopback_v6
            ),
            HostLocality::SingleHost
        );
        assert_eq!(
            refine_rank0_locality(HostLocality::SingleHost, LocalityOverride::Auto, remote_lan),
            HostLocality::MultiHost
        );
        // Explicit override is preserved even if peer IP is non-loopback.
        assert_eq!(
            refine_rank0_locality(
                HostLocality::SingleHost,
                LocalityOverride::Explicit(HostLocality::SingleHost),
                remote_lan,
            ),
            HostLocality::SingleHost
        );
    }

    #[test]
    fn test_should_apply_cx7_roce_defaults_respects_locality_and_overrides() {
        let empty = |_: &str| None;
        // Single-host skips CX-7 RoCE defaults so NCCL retains LL128 + P2P/NVLink.
        assert!(!should_apply_cx7_roce_defaults(
            HostLocality::SingleHost,
            empty
        ));
        // Multi-host applies CX-7 RoCE defaults for unset keys.
        assert!(should_apply_cx7_roce_defaults(
            HostLocality::MultiHost,
            empty
        ));
        // Explicit FLUX_CX7_ROCE=1 enables defaults even on single-host.
        assert!(should_apply_cx7_roce_defaults(
            HostLocality::SingleHost,
            |k| (k == "FLUX_CX7_ROCE").then(|| "1".to_string())
        ));
        // Explicit FLUX_CX7_ROCE=0 suppresses defaults even on multi-host.
        assert!(!should_apply_cx7_roce_defaults(
            HostLocality::MultiHost,
            |k| (k == "FLUX_CX7_ROCE").then(|| "0".to_string())
        ));
    }

    #[test]
    fn test_should_allocate_recv_buffer_multi_rank() {
        // Multi-host keeps 64 MiB recv_buffer for all multi-rank worlds (world_size >= 2).
        assert!(!should_allocate_recv_buffer(
            0,
            HostLocality::MultiHost,
            P2pAllReduceOverride::Auto
        ));
        assert!(!should_allocate_recv_buffer(
            1,
            HostLocality::MultiHost,
            P2pAllReduceOverride::Auto
        ));
        for world_size in [2usize, 3, 4, 8] {
            assert!(
                should_allocate_recv_buffer(
                    world_size,
                    HostLocality::MultiHost,
                    P2pAllReduceOverride::Auto
                ),
                "multi-host world_size={world_size} must allocate recv_buffer"
            );
        }

        // Single-host skips 64 MiB recv_buffer unless world_size == 2 + ForceP2p.
        for world_size in [0usize, 1, 2, 3, 4, 8] {
            assert!(
                !should_allocate_recv_buffer(
                    world_size,
                    HostLocality::SingleHost,
                    P2pAllReduceOverride::Auto
                ),
                "single-host auto world_size={world_size} must skip recv_buffer"
            );
        }
        assert!(should_allocate_recv_buffer(
            2,
            HostLocality::SingleHost,
            P2pAllReduceOverride::ForceP2p
        ));
        assert!(!should_allocate_recv_buffer(
            4,
            HostLocality::SingleHost,
            P2pAllReduceOverride::ForceP2p
        ));
    }

    #[test]
    fn test_should_use_roce_send_recv_allreduce_dispatch() {
        let base = AllReduceDispatchCtx {
            world_size: 2,
            locality: HostLocality::MultiHost,
            p2p_override: P2pAllReduceOverride::Auto,
            bytes: 12_288,
            recv_buffer: 0x1000,
            add_kernel: 0x2000,
        };
        // Multi-host 2-rank uses RoCE send/recv + bf16_add_inplace fast path.
        assert!(should_use_roce_send_recv_allreduce(base));
        assert!(should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            bytes: RECV_BUFFER_SIZE,
            ..base
        }));
        // Oversized, zero-byte, missing buffer/kernel, or non-2-rank falls back to ncclAllReduce.
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            bytes: RECV_BUFFER_SIZE + 2,
            ..base
        }));
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            bytes: 0,
            ..base
        }));
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            recv_buffer: 0,
            ..base
        }));
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            add_kernel: 0,
            ..base
        }));
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            world_size: 4,
            ..base
        }));

        // Single-host 2-rank routes directly to native ncclAllReduce by default.
        assert!(!should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            locality: HostLocality::SingleHost,
            ..base
        }));
        // Single-host 2-rank with FLUX_FORCE_P2P_ALLREDUCE=1 re-enables send/recv path.
        assert!(should_use_roce_send_recv_allreduce(AllReduceDispatchCtx {
            locality: HostLocality::SingleHost,
            p2p_override: P2pAllReduceOverride::ForceP2p,
            ..base
        }));
    }

    #[test]
    fn test_resolve_p2p_allreduce_override_env_keys() {
        assert_eq!(
            resolve_p2p_allreduce_override(|_| None),
            P2pAllReduceOverride::Auto
        );
        assert_eq!(
            resolve_p2p_allreduce_override(
                |k| (k == "FLUX_FORCE_P2P_ALLREDUCE").then(|| "0".to_string())
            ),
            P2pAllReduceOverride::Auto
        );
        assert_eq!(
            resolve_p2p_allreduce_override(
                |k| (k == "FLUX_FORCE_P2P_ALLREDUCE").then(|| "1".to_string())
            ),
            P2pAllReduceOverride::ForceP2p
        );
        assert_eq!(
            resolve_p2p_allreduce_override(
                |k| (k == "ATLAS_FORCE_P2P_ALLREDUCE").then(|| "true".to_string())
            ),
            P2pAllReduceOverride::ForceP2p
        );
        assert_eq!(
            resolve_p2p_allreduce_override(
                |k| (k == "FLUX_FORCE_P2P_ALLREDUCE").then(|| "invalid".to_string())
            ),
            P2pAllReduceOverride::Auto
        );
    }

    #[test]
    fn test_bootstrap_addr_formatting_handles_ipv4_and_ipv6() {
        assert_eq!(
            NcclBackend::rank0_bind_addr("127.0.0.1", 29500),
            "0.0.0.0:29500"
        );
        assert_eq!(
            NcclBackend::rank0_bind_addr("localhost", 29500),
            "0.0.0.0:29500"
        );
        assert_eq!(
            NcclBackend::rank0_bind_addr("10.0.0.1", 29500),
            "0.0.0.0:29500"
        );
        assert_eq!(NcclBackend::rank0_bind_addr("::1", 29500), "[::]:29500");
        assert_eq!(NcclBackend::rank0_bind_addr("[::1]", 29500), "[::]:29500");

        assert_eq!(
            NcclBackend::format_bootstrap_target("127.0.0.1", 29500),
            "127.0.0.1:29500"
        );
        assert_eq!(
            NcclBackend::format_bootstrap_target("::1", 29500),
            "[::1]:29500"
        );
        assert_eq!(
            NcclBackend::format_bootstrap_target("[::1]", 29500),
            "[::1]:29500"
        );
    }

    #[test]
    fn test_poll_comm_init_success_and_in_progress() {
        let mut aborted = false;
        poll_comm_init_until_ready(
            NcclResult::Success,
            Duration::from_millis(50),
            Duration::ZERO,
            "test_immediate",
            || Ok(NcclResult::Success),
            || aborted = true,
        )
        .expect("immediate Success must return Ok");
        assert!(!aborted);

        let mut polls = 0usize;
        let mut aborted2 = false;
        poll_comm_init_until_ready(
            NcclResult::InProgress,
            Duration::from_millis(50),
            Duration::ZERO,
            "test_in_progress",
            || {
                polls += 1;
                if polls < 4 {
                    Ok(NcclResult::InProgress)
                } else {
                    Ok(NcclResult::Success)
                }
            },
            || aborted2 = true,
        )
        .expect("InProgress resolving to Success must return Ok");
        assert_eq!(polls, 4);
        assert!(!aborted2);
    }

    #[test]
    fn test_poll_comm_init_aborts_on_error_and_timeout() {
        let mut aborted_err = false;
        let err = poll_comm_init_until_ready(
            NcclResult::InProgress,
            Duration::from_millis(50),
            Duration::ZERO,
            "test_async_err",
            || Ok(NcclResult::RemoteError),
            || aborted_err = true,
        );
        assert!(err.is_err());
        assert!(aborted_err, "async error during init must abort comm");

        let mut aborted_timeout = false;
        let timeout_res = poll_comm_init_until_ready(
            NcclResult::InProgress,
            Duration::ZERO,
            Duration::ZERO,
            "test_timeout",
            || Ok(NcclResult::InProgress),
            || aborted_timeout = true,
        );
        assert!(timeout_res.is_err());
        assert!(aborted_timeout, "init timeout must abort comm");
    }

    #[test]
    fn test_tracked_handle_validation_rejects_null_unknown_and_double_free() {
        let reg = Mutex::new(HashSet::new());
        assert!(take_tracked_handle(&reg, 0, "symmetric_free").is_err());
        assert!(take_tracked_handle(&reg, 0x1000, "symmetric_free").is_err());

        reg.lock().insert(0x2000);
        take_tracked_handle(&reg, 0x2000, "symmetric_free")
            .expect("live handle must be removed once");
        assert!(
            take_tracked_handle(&reg, 0x2000, "symmetric_free").is_err(),
            "second removal (double-free) must fail"
        );
    }

    #[test]
    fn test_owned_nccl_comm_rejects_null_and_none() {
        assert!(OwnedNcclComm::new(ptr::null_mut()).is_err());
        assert!(NcclBackend::require_comm(&None, "unit_test").is_err());
    }
}
