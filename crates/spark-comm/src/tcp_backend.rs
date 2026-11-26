// SPDX-License-Identifier: AGPL-3.0-only

//! Full-mesh TCP communication backend for NATed / Docker-bridge multi-Spark clusters.
//!
//! Unlike [`crate::NcclBackend`] (which advertises each container's local interface IP
//! and ephemeral TCP ports inside `NcclUniqueId`), [`TcpCommBackend`] establishes a
//! persistent full-mesh of `TCP_NODELAY` sockets across explicit externally mapped
//! `host:port` endpoints (`ATLAS_PEER_ADDRS`) while binding a single local port per
//! rank (`ATLAS_TCP_BIND_PORT`, defaulting to `--master-port`).
//!
//! This enables `2x`/`3x`/`4x` Spark `PP=2/3/4`, `TP=2`, `EP=2`, and `TP+PP` topologies
//! across cloud container hosts (e.g. Docker bridge / NAT-hairpin setups without
//! `--network=host`, `/dev/infiniband`, or `/dev/net/tun`), while preserving
//! [`crate::NcclBackend`] as the default fast path for direct ConnectX-7 RoCEv2 fabrics.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;

use crate::{CollectiveDtype, CommBackend, RoutePacketLayout, nccl};

const HANDSHAKE_MAGIC: [u8; 4] = *b"ATCP";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(3);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(100);
const IO_TIMEOUT: Duration = Duration::from_secs(300);

/// Device memory copy/sync abstraction so [`TcpCommBackend`] can execute against
/// real CUDA device pointers in production and host buffers in CPU-only tests.
trait DeviceMem: Send + Sync {
    fn sync_stream(&self, stream: u64) -> Result<()>;
    fn d2h(&self, dst: &mut [u8], src_dev: u64) -> Result<()>;
    fn h2d(&self, dst_dev: u64, src: &[u8]) -> Result<()>;
}

struct CudaDeviceMem;

impl DeviceMem for CudaDeviceMem {
    fn sync_stream(&self, stream: u64) -> Result<()> {
        // SAFETY: Caller passes `0` or a valid live `CUstream` handle.
        unsafe { nccl::sync_stream(stream) }
    }

    fn d2h(&self, dst: &mut [u8], src_dev: u64) -> Result<()> {
        // SAFETY: Caller passes a valid CUDA device pointer sized >= dst.len().
        unsafe { nccl::memcpy_d2h(dst, src_dev) }
    }

    fn h2d(&self, dst_dev: u64, src: &[u8]) -> Result<()> {
        // SAFETY: Caller passes a valid CUDA device pointer sized >= src.len().
        unsafe { nccl::memcpy_h2d(dst_dev, src) }
    }
}

struct PeerLink {
    rx: Mutex<TcpStream>,
    tx: Mutex<TcpStream>,
}

impl PeerLink {
    fn from_stream(stream: TcpStream) -> Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;
        let tx = stream.try_clone().context("clone TCP stream for tx half")?;
        Ok(Self {
            rx: Mutex::new(stream),
            tx: Mutex::new(tx),
        })
    }

    fn send_frame(&self, data: &[u8]) -> Result<()> {
        let mut tx = self.tx.lock();
        tx.write_all(&(data.len() as u64).to_le_bytes())?;
        if !data.is_empty() {
            tx.write_all(data)?;
        }
        Ok(())
    }

    fn recv_frame(&self, expected_len: usize) -> Result<Vec<u8>> {
        let mut rx = self.rx.lock();
        let mut hdr = [0u8; 8];
        rx.read_exact(&mut hdr)?;
        let len = u64::from_le_bytes(hdr) as usize;
        ensure!(
            len == expected_len,
            "TCP comm frame length mismatch: expected {expected_len} bytes, got {len}"
        );
        let mut buf = vec![0u8; len];
        if len > 0 {
            rx.read_exact(&mut buf)?;
        }
        Ok(buf)
    }
}

enum PendingOp {
    Send {
        ptr: u64,
        bytes: usize,
        dest: usize,
        stream: u64,
    },
    Recv {
        ptr: u64,
        bytes: usize,
        src: usize,
        stream: u64,
    },
}

struct GroupState {
    depth: usize,
    ops: Vec<PendingOp>,
}

/// Full-mesh TCP implementation of [`CommBackend`] using explicit per-rank endpoints.
pub struct TcpCommBackend {
    rank: usize,
    world_size: usize,
    default_stream: u64,
    /// Indexed by peer rank; `peers[self.rank]` is `None`.
    peers: Vec<Option<PeerLink>>,
    group: Mutex<GroupState>,
    mem: Arc<dyn DeviceMem>,
    healthy: AtomicBool,
}

impl TcpCommBackend {
    /// Resolve the effective TCP bind address.
    ///
    /// If `FLUX_TCP_BIND_ADDR` (or compatibility alias `ATLAS_TCP_BIND_ADDR`) is
    /// set to a non-empty value, it overrides `default_bind_addr` so operators
    /// can pin collective listeners to a specific private interface rather than
    /// binding `0.0.0.0`.
    pub fn resolve_bind_addr(default_bind_addr: &str) -> String {
        for var in ["FLUX_TCP_BIND_ADDR", "ATLAS_TCP_BIND_ADDR"] {
            if let Ok(val) = std::env::var(var) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_owned();
                }
            }
        }
        default_bind_addr.to_owned()
    }

    /// Establish a full-mesh TCP communicator across `world_size` ranks.
    ///
    /// - `bind_addr`: local address to bind for inbound connections from lower ranks
    ///   (e.g. `"127.0.0.1:29500"` or a private cluster interface IP; overridable
    ///   via `FLUX_TCP_BIND_ADDR`).
    /// - `peer_addrs`: slice of length `world_size` with the externally reachable
    ///   `host:port` for every rank `0..world_size`.
    ///
    /// # Security
    /// `TcpCommBackend` transfers unencrypted tensor payloads using a lightweight
    /// rank handshake. Bind to a private cluster interface via `FLUX_TCP_BIND_ADDR`
    /// and restrict port access via firewall/VPC rules; never expose this port on
    /// an untrusted public network.
    pub fn new(
        rank: usize,
        world_size: usize,
        bind_addr: &str,
        peer_addrs: &[String],
        default_stream: u64,
    ) -> Result<Self> {
        let effective_bind = Self::resolve_bind_addr(bind_addr);
        Self::new_with_mem(
            rank,
            world_size,
            &effective_bind,
            peer_addrs,
            default_stream,
            Arc::new(CudaDeviceMem),
        )
    }

    fn new_with_mem(
        rank: usize,
        world_size: usize,
        bind_addr: &str,
        peer_addrs: &[String],
        default_stream: u64,
        mem: Arc<dyn DeviceMem>,
    ) -> Result<Self> {
        Self::new_with_mem_and_timeout(
            rank,
            world_size,
            bind_addr,
            peer_addrs,
            default_stream,
            mem,
            CONNECT_TIMEOUT,
        )
    }

    fn new_with_mem_and_timeout(
        rank: usize,
        world_size: usize,
        bind_addr: &str,
        peer_addrs: &[String],
        default_stream: u64,
        mem: Arc<dyn DeviceMem>,
        connect_timeout: Duration,
    ) -> Result<Self> {
        ensure!(world_size >= 2, "TcpCommBackend requires world_size >= 2");
        ensure!(
            rank < world_size,
            "TcpCommBackend invalid rank {rank} for world_size {world_size}"
        );
        ensure!(
            peer_addrs.len() == world_size,
            "TcpCommBackend requires {world_size} peer addresses, got {}",
            peer_addrs.len()
        );

        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("TcpCommBackend rank {rank}: failed to bind {bind_addr}"))?;
        let local_addr = listener.local_addr()?;
        if local_addr.ip().is_unspecified() {
            tracing::warn!(
                "TcpCommBackend rank {rank}/{world_size}: binding unauthenticated plaintext \
                 collective listener on {local_addr} (all interfaces). Restrict access via \
                 private VPC/firewall rules or set FLUX_TCP_BIND_ADDR=<private-ip>:<port>."
            );
        }
        tracing::info!(
            "TcpCommBackend rank {rank}/{world_size}: listening on {local_addr}, \
             connecting to peers {:?}",
            &peer_addrs[(rank + 1)..]
        );

        let peers: Vec<Mutex<Option<PeerLink>>> =
            (0..world_size).map(|_| Mutex::new(None)).collect();

        if rank > 0 {
            listener
                .set_nonblocking(true)
                .context("TcpCommBackend: failed to set listener non-blocking for timed accept")?;
        }

        std::thread::scope(|scope| -> Result<()> {
            let accept_handle = scope.spawn(|| -> Result<()> {
                let deadline = Instant::now() + connect_timeout;
                let mut accepted = 0usize;
                while accepted < rank {
                    let (mut stream, remote) = match listener.accept() {
                        Ok(pair) => pair,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                bail!(
                                    "TcpCommBackend rank {rank}: timed out waiting for inbound \
                                     peer connections ({accepted}/{rank} accepted) after {}ms",
                                    connect_timeout.as_millis()
                                );
                            }
                            std::thread::sleep(CONNECT_RETRY_INTERVAL);
                            continue;
                        }
                        Err(e) => {
                            return Err(e).with_context(|| {
                                format!(
                                    "TcpCommBackend rank {rank}: accept failed ({accepted}/{rank})"
                                )
                            });
                        }
                    };
                    // Accepted socket may inherit non-blocking mode from the listener; switch
                    // back to blocking before timed handshake and framed I/O.
                    stream.set_nonblocking(false)?;
                    stream.set_nodelay(true)?;
                    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
                    let mut hello = [0u8; 8];
                    stream.read_exact(&mut hello).with_context(|| {
                        format!("TcpCommBackend rank {rank}: handshake read from {remote}")
                    })?;
                    ensure!(
                        hello[0..4] == HANDSHAKE_MAGIC,
                        "TcpCommBackend rank {rank}: bad handshake magic from {remote}"
                    );
                    let peer_rank =
                        u32::from_le_bytes([hello[4], hello[5], hello[6], hello[7]]) as usize;
                    ensure!(
                        peer_rank < rank,
                        "TcpCommBackend rank {rank}: unexpected inbound peer rank {peer_rank} from {remote}"
                    );
                    let link = PeerLink::from_stream(stream)?;
                    let mut slot = peers[peer_rank].lock();
                    ensure!(
                        slot.is_none(),
                        "TcpCommBackend rank {rank}: duplicate connection from peer {peer_rank}"
                    );
                    *slot = Some(link);
                    accepted += 1;
                    tracing::info!(
                        "TcpCommBackend rank {rank}: accepted peer {peer_rank} from {remote}"
                    );
                }
                Ok(())
            });

            let mut connect_handles = Vec::with_capacity(world_size - 1 - rank);
            for (peer_rank, addr_str) in peer_addrs.iter().enumerate().skip(rank + 1) {
                let peers_ref = &peers;
                let addr_owned = addr_str.clone();
                connect_handles.push(scope.spawn(move || -> Result<()> {
                    let stream = connect_with_retry(rank, peer_rank, &addr_owned, connect_timeout)?;
                    let link = PeerLink::from_stream(stream)?;
                    *peers_ref[peer_rank].lock() = Some(link);
                    tracing::info!(
                        "TcpCommBackend rank {rank}: connected to peer {peer_rank} at {addr_owned}"
                    );
                    Ok(())
                }));
            }

            accept_handle
                .join()
                .map_err(|_| anyhow::anyhow!("TcpCommBackend accept thread panicked"))??;
            for h in connect_handles {
                h.join()
                    .map_err(|_| anyhow::anyhow!("TcpCommBackend connect thread panicked"))??;
            }
            Ok(())
        })?;

        let resolved: Vec<Option<PeerLink>> = peers.into_iter().map(Mutex::into_inner).collect();
        for (p, slot) in resolved.iter().enumerate() {
            if p != rank {
                ensure!(
                    slot.is_some(),
                    "TcpCommBackend rank {rank}: missing peer link for rank {p}"
                );
            }
        }

        let backend = Self {
            rank,
            world_size,
            default_stream,
            peers: resolved,
            group: Mutex::new(GroupState {
                depth: 0,
                ops: Vec::new(),
            }),
            mem,
            healthy: AtomicBool::new(true),
        };
        backend
            .barrier()
            .context("TcpCommBackend initial mesh barrier failed")?;
        tracing::info!("TcpCommBackend rank {rank}/{world_size}: full-mesh TCP communicator ready");
        Ok(backend)
    }

    fn peer(&self, peer_rank: usize) -> Result<&PeerLink> {
        ensure!(
            peer_rank < self.world_size && peer_rank != self.rank,
            "TcpCommBackend rank {}: invalid peer rank {peer_rank} (world_size={})",
            self.rank,
            self.world_size
        );
        self.peers[peer_rank]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TcpCommBackend missing link to rank {peer_rank}"))
    }

    fn mark_unhealthy_on_err<T>(&self, res: Result<T>) -> Result<T> {
        if res.is_err() {
            self.healthy.store(false, Ordering::Release);
        }
        res
    }

    fn all_reduce_bf16_on_stream(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        ensure!(
            bytes.is_multiple_of(2),
            "TcpCommBackend all_reduce: byte count {bytes} must be a multiple of 2 (BF16)"
        );
        self.mem.sync_stream(stream)?;
        let mut local = vec![0u8; bytes];
        self.mem.d2h(&mut local, ptr)?;

        let reduced = if self.world_size == 2 {
            let peer = self.peer(1 - self.rank)?;
            let remote = std::thread::scope(|s| -> Result<Vec<u8>> {
                let tx = s.spawn(|| peer.send_frame(&local));
                let rx = peer.recv_frame(bytes);
                tx.join()
                    .map_err(|_| anyhow::anyhow!("TCP all_reduce tx panicked"))??;
                rx
            })?;
            sum_bf16_buffers(&[&local, &remote])
        } else if self.rank == 0 {
            let remote_bufs: Vec<Vec<u8>> = std::thread::scope(|s| -> Result<Vec<Vec<u8>>> {
                let mut handles = Vec::with_capacity(self.world_size - 1);
                for p in 1..self.world_size {
                    let peer = self.peer(p)?;
                    handles.push(s.spawn(move || peer.recv_frame(bytes)));
                }
                let mut out = Vec::with_capacity(self.world_size - 1);
                for h in handles {
                    out.push(
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP all_reduce root rx panicked"))??,
                    );
                }
                Ok(out)
            })?;
            let mut slices: Vec<&[u8]> = Vec::with_capacity(self.world_size);
            slices.push(&local);
            for b in &remote_bufs {
                slices.push(b);
            }
            let sum = sum_bf16_buffers(&slices);
            std::thread::scope(|s| -> Result<()> {
                let mut handles = Vec::with_capacity(self.world_size - 1);
                for p in 1..self.world_size {
                    let peer = self.peer(p)?;
                    let sum_ref = &sum;
                    handles.push(s.spawn(move || peer.send_frame(sum_ref)));
                }
                for h in handles {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("TCP all_reduce root tx panicked"))??;
                }
                Ok(())
            })?;
            sum
        } else {
            let root = self.peer(0)?;
            root.send_frame(&local)?;
            root.recv_frame(bytes)?
        };

        self.mem.h2d(ptr, &reduced)?;
        Ok(())
    }

    fn execute_ops_batch(&self, ops: Vec<PendingOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        // Sync any unique streams referenced by sends or receives before D2H/H2D
        // so legacy-stream copies never race with prior kernels on non-blocking streams.
        let mut synced_streams = Vec::new();
        for op in &ops {
            let stream = match *op {
                PendingOp::Send { stream, .. } | PendingOp::Recv { stream, .. } => stream,
            };
            if !synced_streams.contains(&stream) {
                self.mem.sync_stream(stream)?;
                synced_streams.push(stream);
            }
        }
        // Stage all outgoing buffers from GPU on the calling thread (which owns the CUDA context),
        // grouping sends by destination peer and receives by source peer in submission order so
        // FIFO TCP framing order on each peer socket is strictly preserved.
        let mut sends_by_peer: BTreeMap<usize, Vec<Vec<u8>>> = BTreeMap::new();
        let mut recvs_by_peer: BTreeMap<usize, Vec<(u64, usize)>> = BTreeMap::new();
        for op in ops {
            match op {
                PendingOp::Send {
                    ptr, bytes, dest, ..
                } => {
                    let mut buf = vec![0u8; bytes];
                    self.mem.d2h(&mut buf, ptr)?;
                    sends_by_peer.entry(dest).or_default().push(buf);
                }
                PendingOp::Recv {
                    ptr, bytes, src, ..
                } => {
                    recvs_by_peer.entry(src).or_default().push((ptr, bytes));
                }
            }
        }
        // Run at most one TX thread per destination peer and one RX thread per source peer so
        // different peers proceed concurrently while same-peer frames stay strictly FIFO.
        let received: Vec<(u64, Vec<u8>)> =
            std::thread::scope(|s| -> Result<Vec<(u64, Vec<u8>)>> {
                let mut tx_handles = Vec::with_capacity(sends_by_peer.len());
                for (dest, bufs) in sends_by_peer {
                    let peer = self.peer(dest)?;
                    tx_handles.push(s.spawn(move || -> Result<()> {
                        for buf in bufs {
                            peer.send_frame(&buf)?;
                        }
                        Ok(())
                    }));
                }
                let mut rx_handles = Vec::with_capacity(recvs_by_peer.len());
                for (src, reqs) in recvs_by_peer {
                    let peer = self.peer(src)?;
                    rx_handles.push(s.spawn(move || -> Result<Vec<(u64, Vec<u8>)>> {
                        let mut peer_out = Vec::with_capacity(reqs.len());
                        for (ptr, bytes) in reqs {
                            let data = peer.recv_frame(bytes)?;
                            peer_out.push((ptr, data));
                        }
                        Ok(peer_out)
                    }));
                }
                for h in tx_handles {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("TCP grouped send thread panicked"))??;
                }
                let mut out = Vec::new();
                for h in rx_handles {
                    out.extend(
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP grouped recv thread panicked"))??,
                    );
                }
                Ok(out)
            })?;
        // Upload received buffers back to GPU on the calling thread.
        for (ptr, data) in received {
            self.mem.h2d(ptr, &data)?;
        }
        Ok(())
    }
}

impl CommBackend for TcpCommBackend {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        let res = self.all_reduce_bf16_on_stream(ptr, bytes, self.default_stream);
        self.mark_unhealthy_on_err(res)
    }

    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let res = self.all_reduce_bf16_on_stream(ptr, bytes, compute_stream);
        self.mark_unhealthy_on_err(res)
    }

    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let res = (|| -> Result<()> {
            if bytes == 0 {
                return Ok(());
            }
            self.mem.sync_stream(self.default_stream)?;
            let mut local = vec![0u8; bytes];
            self.mem.d2h(&mut local, send_ptr)?;
            let mut full = vec![0u8; bytes * self.world_size];
            full[self.rank * bytes..(self.rank + 1) * bytes].copy_from_slice(&local);

            let received: Vec<(usize, Vec<u8>)> =
                std::thread::scope(|s| -> Result<Vec<(usize, Vec<u8>)>> {
                    let mut tx_handles = Vec::with_capacity(self.world_size - 1);
                    let mut rx_handles = Vec::with_capacity(self.world_size - 1);
                    for p in 0..self.world_size {
                        if p == self.rank {
                            continue;
                        }
                        let peer = self.peer(p)?;
                        let local_ref = &local;
                        tx_handles.push(s.spawn(move || peer.send_frame(local_ref)));
                        rx_handles.push(s.spawn(move || -> Result<(usize, Vec<u8>)> {
                            Ok((p, peer.recv_frame(bytes)?))
                        }));
                    }
                    for h in tx_handles {
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP all_gather tx panicked"))??;
                    }
                    let mut out = Vec::with_capacity(rx_handles.len());
                    for h in rx_handles {
                        out.push(
                            h.join()
                                .map_err(|_| anyhow::anyhow!("TCP all_gather rx panicked"))??,
                        );
                    }
                    Ok(out)
                })?;

            for (p, slice) in received {
                full[p * bytes..(p + 1) * bytes].copy_from_slice(&slice);
            }
            self.mem.h2d(recv_ptr, &full)
        })();
        self.mark_unhealthy_on_err(res)
    }

    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        self.reduce_scatter_typed(
            send_ptr,
            recv_ptr,
            bytes,
            CollectiveDtype::Bf16,
            self.default_stream,
        )
    }

    fn reduce_scatter_typed(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        dtype: CollectiveDtype,
        stream: u64,
    ) -> Result<()> {
        let res = (|| -> Result<()> {
            let elem_bytes = dtype.element_bytes();
            ensure!(
                bytes.is_multiple_of(elem_bytes),
                "reduce_scatter_typed: byte count {bytes} is not a multiple of {:?} element size ({elem_bytes})",
                dtype
            );
            if bytes == 0 {
                return Ok(());
            }
            self.mem.sync_stream(stream)?;
            let mut full_send = vec![0u8; bytes * self.world_size];
            self.mem.d2h(&mut full_send, send_ptr)?;

            let remote_slices: Vec<Vec<u8>> = std::thread::scope(|s| -> Result<Vec<Vec<u8>>> {
                let mut tx_handles = Vec::with_capacity(self.world_size - 1);
                let mut rx_handles = Vec::with_capacity(self.world_size - 1);
                for p in 0..self.world_size {
                    if p == self.rank {
                        continue;
                    }
                    let peer = self.peer(p)?;
                    let slice = &full_send[p * bytes..(p + 1) * bytes];
                    tx_handles.push(s.spawn(move || peer.send_frame(slice)));
                    rx_handles.push(s.spawn(move || peer.recv_frame(bytes)));
                }
                for h in tx_handles {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("TCP reduce_scatter tx panicked"))??;
                }
                let mut out = Vec::with_capacity(rx_handles.len());
                for h in rx_handles {
                    out.push(
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP reduce_scatter rx panicked"))??,
                    );
                }
                Ok(out)
            })?;

            let mut slices: Vec<&[u8]> = Vec::with_capacity(self.world_size);
            slices.push(&full_send[self.rank * bytes..(self.rank + 1) * bytes]);
            for b in &remote_slices {
                slices.push(b);
            }
            let reduced = match dtype {
                CollectiveDtype::Bf16 => sum_bf16_buffers(&slices),
                CollectiveDtype::Fp32 => sum_fp32_buffers(&slices),
                CollectiveDtype::Uint8 => sum_u8_buffers(&slices),
            };
            self.mem.h2d(recv_ptr, &reduced)
        })();
        self.mark_unhealthy_on_err(res)
    }

    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        let res = (|| -> Result<()> {
            ensure!(
                root < self.world_size,
                "TcpCommBackend broadcast invalid root {root}"
            );
            if bytes == 0 {
                return Ok(());
            }
            if self.rank == root {
                self.mem.sync_stream(self.default_stream)?;
                let mut data = vec![0u8; bytes];
                self.mem.d2h(&mut data, ptr)?;
                std::thread::scope(|s| -> Result<()> {
                    let mut handles = Vec::with_capacity(self.world_size - 1);
                    for p in 0..self.world_size {
                        if p == root {
                            continue;
                        }
                        let peer = self.peer(p)?;
                        let data_ref = &data;
                        handles.push(s.spawn(move || peer.send_frame(data_ref)));
                    }
                    for h in handles {
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP broadcast tx panicked"))??;
                    }
                    Ok(())
                })?;
            } else {
                let data = self.peer(root)?.recv_frame(bytes)?;
                self.mem.sync_stream(self.default_stream)?;
                self.mem.h2d(ptr, &data)?;
            }
            Ok(())
        })();
        self.mark_unhealthy_on_err(res)
    }

    fn barrier(&self) -> Result<()> {
        let res = (|| -> Result<()> {
            if self.rank == 0 {
                std::thread::scope(|s| -> Result<()> {
                    let mut rx_handles = Vec::with_capacity(self.world_size - 1);
                    for p in 1..self.world_size {
                        let peer = self.peer(p)?;
                        rx_handles.push(s.spawn(move || peer.recv_frame(1)));
                    }
                    for h in rx_handles {
                        let _ = h
                            .join()
                            .map_err(|_| anyhow::anyhow!("TCP barrier rx panicked"))??;
                    }
                    let mut tx_handles = Vec::with_capacity(self.world_size - 1);
                    for p in 1..self.world_size {
                        let peer = self.peer(p)?;
                        tx_handles.push(s.spawn(move || peer.send_frame(&[1u8])));
                    }
                    for h in tx_handles {
                        h.join()
                            .map_err(|_| anyhow::anyhow!("TCP barrier tx panicked"))??;
                    }
                    Ok(())
                })
            } else {
                let root = self.peer(0)?;
                root.send_frame(&[1u8])?;
                let _ = root.recv_frame(1)?;
                Ok(())
            }
        })();
        self.mark_unhealthy_on_err(res)
    }

    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()> {
        let mut g = self.group.lock();
        if g.depth > 0 {
            g.ops.push(PendingOp::Send {
                ptr,
                bytes,
                dest: dest_rank,
                stream,
            });
            return Ok(());
        }
        drop(g);
        let res = (|| -> Result<()> {
            self.mem.sync_stream(stream)?;
            let mut buf = vec![0u8; bytes];
            self.mem.d2h(&mut buf, ptr)?;
            self.peer(dest_rank)?.send_frame(&buf)
        })();
        self.mark_unhealthy_on_err(res)
    }

    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()> {
        let mut g = self.group.lock();
        if g.depth > 0 {
            g.ops.push(PendingOp::Recv {
                ptr,
                bytes,
                src: src_rank,
                stream,
            });
            return Ok(());
        }
        drop(g);
        let res = (|| -> Result<()> {
            let buf = self.peer(src_rank)?.recv_frame(bytes)?;
            self.mem.sync_stream(stream)?;
            self.mem.h2d(ptr, &buf)
        })();
        self.mark_unhealthy_on_err(res)
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
        let bytes = local_tokens
            .checked_mul(layout.packet_bytes())
            .ok_or_else(|| anyhow::anyhow!("SelectedRoute gather byte count overflow"))?;
        self.mem.sync_stream(stream)?;
        let res = self.all_gather(send_routes_ptr, recv_routes_ptr, bytes);
        self.mark_unhealthy_on_err(res)
    }

    fn group_start(&self) -> Result<()> {
        let mut g = self.group.lock();
        g.depth += 1;
        Ok(())
    }

    fn group_end(&self) -> Result<()> {
        let ops = {
            let mut g = self.group.lock();
            ensure!(g.depth > 0, "TcpCommBackend group_end without group_start");
            g.depth -= 1;
            if g.depth > 0 {
                return Ok(());
            }
            std::mem::take(&mut g.ops)
        };
        let res = self.execute_ops_batch(ops);
        self.mark_unhealthy_on_err(res)
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn attempt_reconnect(&self) -> Result<()> {
        bail!(
            "TcpCommBackend rank {}/{} does not support in-place socket reconnect \
             (healthy={}); restart cluster ranks to re-establish the TCP mesh",
            self.rank,
            self.world_size,
            self.is_healthy()
        )
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

fn connect_with_retry(
    rank: usize,
    peer_rank: usize,
    addr: &str,
    timeout: Duration,
) -> Result<TcpStream> {
    let deadline = Instant::now() + timeout;
    loop {
        let sock_addr: Option<SocketAddr> =
            addr.to_socket_addrs().ok().and_then(|mut it| it.next());
        let attempt = match sock_addr {
            Some(sa) => TcpStream::connect_timeout(&sa, CONNECT_ATTEMPT_TIMEOUT),
            None => TcpStream::connect(addr),
        };
        match attempt {
            Ok(mut stream) => {
                stream.set_nodelay(true)?;
                let mut hello = [0u8; 8];
                hello[0..4].copy_from_slice(&HANDSHAKE_MAGIC);
                hello[4..8].copy_from_slice(&(rank as u32).to_le_bytes());
                stream.write_all(&hello)?;
                return Ok(stream);
            }
            Err(e) => {
                if Instant::now() >= deadline {
                    bail!(
                        "TcpCommBackend rank {rank}: timed out connecting to peer {peer_rank} \
                         at {addr} after {}ms: {e}",
                        timeout.as_millis()
                    );
                }
                std::thread::sleep(CONNECT_RETRY_INTERVAL);
            }
        }
    }
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn f32_to_bf16_rn(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x0000_7FFF_u32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn sum_bf16_buffers(slices: &[&[u8]]) -> Vec<u8> {
    let bytes = slices[0].len();
    let count = bytes / 2;
    let mut out = vec![0u8; bytes];
    for i in 0..count {
        let off = i * 2;
        let mut acc = 0.0f32;
        for s in slices {
            let bits = u16::from_le_bytes([s[off], s[off + 1]]);
            acc += bf16_to_f32(bits);
        }
        let rounded = f32_to_bf16_rn(acc).to_le_bytes();
        out[off] = rounded[0];
        out[off + 1] = rounded[1];
    }
    out
}

fn sum_fp32_buffers(slices: &[&[u8]]) -> Vec<u8> {
    let bytes = slices[0].len();
    let count = bytes / 4;
    let mut out = vec![0u8; bytes];
    for i in 0..count {
        let off = i * 4;
        let mut acc = 0.0f32;
        for s in slices {
            acc += f32::from_le_bytes([s[off], s[off + 1], s[off + 2], s[off + 3]]);
        }
        out[off..off + 4].copy_from_slice(&acc.to_le_bytes());
    }
    out
}

fn sum_u8_buffers(slices: &[&[u8]]) -> Vec<u8> {
    let bytes = slices[0].len();
    let mut out = vec![0u8; bytes];
    for i in 0..bytes {
        let mut acc = 0u8;
        for s in slices {
            acc = acc.wrapping_add(s[i]);
        }
        out[i] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host-memory mock implementing [`DeviceMem`] via raw host addresses for
    /// CPU-only multi-rank TCP collective verification.
    struct HostMem;

    impl DeviceMem for HostMem {
        fn sync_stream(&self, _stream: u64) -> Result<()> {
            Ok(())
        }
        fn d2h(&self, dst: &mut [u8], src_dev: u64) -> Result<()> {
            if !dst.is_empty() {
                // SAFETY: Tests pass valid pointers into live local Vec<u8> buffers.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_dev as *const u8,
                        dst.as_mut_ptr(),
                        dst.len(),
                    );
                }
            }
            Ok(())
        }
        fn h2d(&self, dst_dev: u64, src: &[u8]) -> Result<()> {
            if !src.is_empty() {
                // SAFETY: Tests pass valid pointers into live local Vec<u8> buffers.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst_dev as *mut u8, src.len());
                }
            }
            Ok(())
        }
    }

    fn reserve_ports(n: usize) -> Vec<u16> {
        let listeners: Vec<TcpListener> = (0..n)
            .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
            .collect();
        listeners
            .iter()
            .map(|l| l.local_addr().unwrap().port())
            .collect()
    }

    #[test]
    fn four_rank_tcp_mesh_all_collectives_and_pp_ring() {
        let world = 4usize;
        let ports = reserve_ports(world);
        let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{p}")).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, port) in ports.iter().enumerate() {
                let bind = format!("127.0.0.1:{port}");
                let peer_addrs = addrs.clone();
                handles.push(s.spawn(move || -> Result<()> {
                    let comm = TcpCommBackend::new_with_mem(
                        rank,
                        world,
                        &bind,
                        &peer_addrs,
                        0,
                        Arc::new(HostMem),
                    )?;
                    assert_eq!(comm.rank(), rank);
                    assert_eq!(comm.world_size(), world);
                    assert!(comm.is_healthy());

                    // 1. Broadcast u32 command from rank 0
                    let mut cmd_buf = if rank == 0 {
                        0xDEAD_BEEF_u32.to_le_bytes().to_vec()
                    } else {
                        vec![0u8; 4]
                    };
                    comm.broadcast(cmd_buf.as_mut_ptr() as u64, 4, 0)?;
                    assert_eq!(u32::from_le_bytes(cmd_buf.try_into().unwrap()), 0xDEAD_BEEF);

                    // 2. BF16 all_reduce across all 4 ranks: (1 + 2 + 3 + 4) = 10.0
                    let val = f32_to_bf16_rn((rank + 1) as f32).to_le_bytes();
                    let mut ar_buf = vec![val[0], val[1], val[0], val[1]];
                    comm.all_reduce(ar_buf.as_mut_ptr() as u64, 4)?;
                    let got0 = bf16_to_f32(u16::from_le_bytes([ar_buf[0], ar_buf[1]]));
                    let got1 = bf16_to_f32(u16::from_le_bytes([ar_buf[2], ar_buf[3]]));
                    assert_eq!(got0, 10.0);
                    assert_eq!(got1, 10.0);

                    // 3. PP=4 ring pass: 0 -> 1 -> 2 -> 3 -> 0 (each rank adds +10)
                    let mut pp_buf = if rank == 0 {
                        100u32.to_le_bytes().to_vec()
                    } else {
                        vec![0u8; 4]
                    };
                    if rank > 0 {
                        comm.recv_from(pp_buf.as_mut_ptr() as u64, 4, rank - 1, 0)?;
                    }
                    let v = u32::from_le_bytes(pp_buf.as_slice().try_into().unwrap()) + 10;
                    pp_buf.copy_from_slice(&v.to_le_bytes());
                    if rank == 0 {
                        comm.send_to(pp_buf.as_ptr() as u64, 4, 1, 0)?;
                        comm.recv_from(pp_buf.as_mut_ptr() as u64, 4, world - 1, 0)?;
                        assert_eq!(u32::from_le_bytes(pp_buf.try_into().unwrap()), 140);
                    } else {
                        let dst = if rank + 1 == world { 0 } else { rank + 1 };
                        comm.send_to(pp_buf.as_ptr() as u64, 4, dst, 0)?;
                    }

                    // 4. All-gather across 4 ranks
                    let send_ag = (rank as u32 + 50).to_le_bytes().to_vec();
                    let mut recv_ag = vec![0u8; 4 * world];
                    comm.all_gather(send_ag.as_ptr() as u64, recv_ag.as_mut_ptr() as u64, 4)?;
                    for r in 0..world {
                        let got =
                            u32::from_le_bytes(recv_ag[r * 4..(r + 1) * 4].try_into().unwrap());
                        assert_eq!(got, (r as u32) + 50);
                    }

                    // 5. Grouped simultaneous send_to + recv_from (deadlock-freedom proof)
                    let peer = (rank + 2) % world;
                    let send_g = vec![(rank as u8) + 1; 65_536];
                    let mut recv_g = vec![0u8; 65_536];
                    comm.group_start()?;
                    comm.send_to(send_g.as_ptr() as u64, send_g.len(), peer, 0)?;
                    comm.recv_from(recv_g.as_mut_ptr() as u64, recv_g.len(), peer, 0)?;
                    comm.group_end()?;
                    assert!(recv_g.iter().all(|&b| b == (peer as u8) + 1));

                    // 6. Multiple grouped sends/recvs to the SAME peer preserve FIFO order
                    let send_a = [(rank as u8).wrapping_add(10); 17];
                    let send_b = vec![(rank as u8).wrapping_add(20); 311];
                    let send_c = vec![(rank as u8).wrapping_add(30); 4099];
                    let mut recv_a = [0u8; 17];
                    let mut recv_b = vec![0u8; 311];
                    let mut recv_c = vec![0u8; 4099];
                    comm.group_start()?;
                    comm.send_to(send_a.as_ptr() as u64, send_a.len(), peer, 0)?;
                    comm.send_to(send_b.as_ptr() as u64, send_b.len(), peer, 0)?;
                    comm.send_to(send_c.as_ptr() as u64, send_c.len(), peer, 0)?;
                    comm.recv_from(recv_a.as_mut_ptr() as u64, recv_a.len(), peer, 0)?;
                    comm.recv_from(recv_b.as_mut_ptr() as u64, recv_b.len(), peer, 0)?;
                    comm.recv_from(recv_c.as_mut_ptr() as u64, recv_c.len(), peer, 0)?;
                    comm.group_end()?;
                    assert!(recv_a.iter().all(|&b| b == (peer as u8).wrapping_add(10)));
                    assert!(recv_b.iter().all(|&b| b == (peer as u8).wrapping_add(20)));
                    assert!(recv_c.iter().all(|&b| b == (peer as u8).wrapping_add(30)));

                    assert!(
                        comm.attempt_reconnect().is_err(),
                        "TcpCommBackend::attempt_reconnect must return Err rather than silent Ok"
                    );

                    comm.barrier()?;
                    Ok(())
                }));
            }
            for h in handles {
                h.join().unwrap().unwrap();
            }
        });
    }

    #[test]
    fn tcp_accept_enforces_connect_timeout_when_lower_rank_missing() {
        let ports = reserve_ports(2);
        let addrs: Vec<String> = ports.iter().map(|p| format!("127.0.0.1:{p}")).collect();
        let bind = format!("127.0.0.1:{}", ports[1]);
        let start = Instant::now();
        let res = TcpCommBackend::new_with_mem_and_timeout(
            1,
            2,
            &bind,
            &addrs,
            0,
            Arc::new(HostMem),
            Duration::from_millis(250),
        );
        let err = match res {
            Ok(_) => panic!("rank 1 without rank 0 must time out waiting for accept"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("timed out waiting for inbound"),
            "unexpected error: {err:#}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "accept timeout took too long: {:?}",
            start.elapsed()
        );
    }
}
