// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-0 micro-benchmarks for the storage layer. Three paths:
//   1. cuFile (compat or GDS, depending on nvidia-fs availability)
//   2. io_uring + pinned-host bounce + cuMemcpyHtoDAsync
//   3. POSIX pread + cuMemcpyHtoDAsync (floor for comparison)
//
// All benchmarks read from a pre-allocated test file with O_DIRECT, into a
// device or pinned-host buffer of `BUF_BYTES` size, with sequential 4 MiB and
// random 64 KiB workloads. Results in MiB/s.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Instant;

use crate::cuda_min::{CudaCtx, DeviceBuffer, PinnedBuffer, copy_h_to_d_pinned};

pub const SEQ_IO_BYTES: usize = 4 * 1024 * 1024;
pub const RAND_IO_BYTES: usize = 64 * 1024;
pub const SEQ_ITERS: usize = 64;
pub const RAND_ITERS: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct BenchResult {
    pub mib_per_sec: f64,
    pub iters: usize,
    pub bytes_per_io: usize,
}

fn open_o_direct(path: &Path) -> Result<std::fs::File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("open O_DIRECT {}", path.display()))
}

fn drop_pagecache(fd: RawFd) {
    // SAFETY: `fd` is a valid open file descriptor borrowed from the caller's `File`.
    unsafe {
        libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// Validate benchmark buffer and file extents and return the number of sequential
/// `io_bytes`-sized slots that fit inside `file_bytes` (`>= 1`).
fn validate_bench_bounds(
    file_bytes: u64,
    dev_bytes: usize,
    pinned_bytes: Option<usize>,
    io_bytes: usize,
) -> Result<usize> {
    anyhow::ensure!(io_bytes > 0, "benchmark io_bytes must be > 0");
    anyhow::ensure!(
        io_bytes <= u32::MAX as usize,
        "benchmark io_bytes ({io_bytes}) exceeds u32::MAX"
    );
    anyhow::ensure!(
        io_bytes <= dev_bytes,
        "benchmark io_bytes ({io_bytes}) exceeds device buffer size ({dev_bytes})"
    );
    if let Some(pinned_len) = pinned_bytes {
        anyhow::ensure!(
            io_bytes <= pinned_len,
            "benchmark io_bytes ({io_bytes}) exceeds pinned host buffer size ({pinned_len})"
        );
    }
    anyhow::ensure!(
        file_bytes >= io_bytes as u64,
        "benchmark file size ({file_bytes}) is smaller than io_bytes ({io_bytes})"
    );
    let seq_slots = (file_bytes / io_bytes as u64).max(1) as usize;
    Ok(seq_slots)
}

fn seq_offset(seq_slots: usize, io_bytes: usize, i: usize) -> u64 {
    ((i % seq_slots) as u64) * (io_bytes as u64)
}

fn rand_offset(file_bytes: u64, io_bytes: usize, i: usize) -> u64 {
    let max = file_bytes.saturating_sub(io_bytes as u64) / RAND_IO_BYTES as u64;
    let pseudo = (i as u64).wrapping_mul(2_654_435_761) % max.max(1);
    pseudo * RAND_IO_BYTES as u64
}

pub fn bench_cufile(
    cufile: &cufile_sys::CuFile,
    fd: RawFd,
    file_bytes: u64,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    use cufile_sys::*;
    let seq_slots = validate_bench_bounds(file_bytes, dev.bytes, None, io_bytes)?;
    let mut descr = CUfileDescr_t {
        type_: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
        handle: CUfileDescrHandle::from_fd(fd),
        fs_ops: std::ptr::null(),
    };
    let mut handle: CUfileHandle_t = std::ptr::null_mut();
    // SAFETY: `cufile` is a loaded `CuFile` table, `&mut handle` is a valid out-pointer,
    // and `&mut descr` is an initialized `CUfileDescr_t` wrapping open `fd`.
    let err = unsafe { (cufile.handle_register)(&mut handle, &mut descr) };
    if err.err != CU_FILE_SUCCESS {
        bail!(
            "cuFileHandleRegister failed: {} ({})",
            err.err,
            err_to_str(err.err)
        );
    }
    // SAFETY: `dev.ptr` points to a valid CUDA device allocation of `dev.bytes` bytes.
    let _ = unsafe { (cufile.buf_register)(dev.ptr as *const c_void, dev.bytes, 0) };
    let t = Instant::now();
    for i in 0..iters {
        let off = if sequential {
            seq_offset(seq_slots, io_bytes, i) as i64
        } else {
            rand_offset(file_bytes, io_bytes, i) as i64
        };
        // SAFETY: `handle` is registered above, `dev.ptr` spans at least `io_bytes`
        // device bytes (validated above), and `off + io_bytes <= file_bytes`.
        let n = unsafe { (cufile.read)(handle, dev.ptr as *mut c_void, io_bytes, off as _, 0) };
        if n != io_bytes as isize {
            // SAFETY: Deregistering the same `dev.ptr` and `handle` registered above on error exit.
            unsafe {
                let _ = (cufile.buf_deregister)(dev.ptr as *const c_void);
                (cufile.handle_deregister)(handle);
            }
            bail!("cuFileRead returned {n}, expected {io_bytes}");
        }
    }
    let dt = t.elapsed().as_secs_f64();
    // SAFETY: Deregistering the same `dev.ptr` and `handle` registered above at benchmark end.
    unsafe {
        let _ = (cufile.buf_deregister)(dev.ptr as *const c_void);
        (cufile.handle_deregister)(handle);
    }
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bench_io_uring(
    ctx: &CudaCtx,
    fd: RawFd,
    file_bytes: u64,
    pinned: &mut PinnedBuffer,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    use io_uring::{IoUring, opcode, types};
    let seq_slots = validate_bench_bounds(file_bytes, dev.bytes, Some(pinned.len()), io_bytes)?;
    anyhow::ensure!(
        io_bytes <= i32::MAX as usize,
        "bench_io_uring io_bytes ({io_bytes}) exceeds i32::MAX"
    );
    let mut ring = IoUring::builder()
        .setup_sqpoll(2_000)
        .build(64)
        .context("io_uring setup_sqpoll")?;
    let t = Instant::now();
    let chunk = io_bytes;
    let host_ptr = pinned.as_mut_ptr().cast::<u8>();
    for i in 0..iters {
        let off = if sequential {
            seq_offset(seq_slots, chunk, i)
        } else {
            rand_offset(file_bytes, chunk, i)
        };
        let read_e = opcode::Read::new(types::Fd(fd), host_ptr, chunk as u32)
            .offset(off)
            .build()
            .user_data(0);
        // SAFETY: `fd` is open, `host_ptr` points to `pinned` with `pinned.len() >= chunk`
        // (validated above), and we wait for completion synchronously before reusing `pinned`.
        unsafe {
            ring.submission()
                .push(&read_e)
                .map_err(|_| anyhow::anyhow!("sq full"))?
        };
        ring.submit_and_wait(1)
            .context("io_uring submit_and_wait")?;
        let mut cq = ring.completion();
        let entry: io_uring::cqueue::Entry = cq.next().expect("cqe");
        if entry.result() != chunk as i32 {
            bail!(
                "io_uring read returned {}, expected {}",
                entry.result(),
                chunk
            );
        }
        drop(cq);
        copy_h_to_d_pinned(dev.ptr, pinned.pinned_slice(chunk)?, ctx.stream)?;
    }
    let dt = t.elapsed().as_secs_f64();
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bench_posix(
    ctx: &CudaCtx,
    fd: RawFd,
    file_bytes: u64,
    pinned: &mut PinnedBuffer,
    dev: &DeviceBuffer,
    io_bytes: usize,
    iters: usize,
    sequential: bool,
) -> Result<BenchResult> {
    let seq_slots = validate_bench_bounds(file_bytes, dev.bytes, Some(pinned.len()), io_bytes)?;
    let host_ptr = pinned.as_mut_ptr().cast::<u8>();
    let t = Instant::now();
    for i in 0..iters {
        let off = if sequential {
            seq_offset(seq_slots, io_bytes, i) as i64
        } else {
            rand_offset(file_bytes, io_bytes, i) as i64
        };
        // SAFETY: `fd` is open, `host_ptr` points to `pinned` with `pinned.len() >= io_bytes`
        // (validated above), and `off + io_bytes <= file_bytes`.
        let n = unsafe { libc::pread(fd, host_ptr as *mut c_void, io_bytes, off) };
        if n != io_bytes as isize {
            bail!(
                "pread returned {n}, expected {io_bytes} (errno {})",
                std::io::Error::last_os_error()
            );
        }
        copy_h_to_d_pinned(dev.ptr, pinned.pinned_slice(io_bytes)?, ctx.stream)?;
    }
    let dt = t.elapsed().as_secs_f64();
    let bytes = (io_bytes * iters) as f64;
    Ok(BenchResult {
        mib_per_sec: bytes / dt / (1024.0 * 1024.0),
        iters,
        bytes_per_io: io_bytes,
    })
}

pub fn open_test_file(path: &Path) -> Result<(std::fs::File, u64)> {
    let f = open_o_direct(path)?;
    let len = f.metadata()?.len();
    drop_pagecache(f.as_raw_fd());
    Ok((f, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_bench_bounds_rejects_invalid_extents_and_wraps_seq_slots() {
        assert!(validate_bench_bounds(4096, 4096, Some(4096), 0).is_err());
        assert!(validate_bench_bounds(4096, 2048, Some(4096), 4096).is_err());
        assert!(validate_bench_bounds(4096, 4096, Some(2048), 4096).is_err());
        assert!(validate_bench_bounds(2048, 4096, Some(4096), 4096).is_err());

        let slots = validate_bench_bounds(12 * 1024, 4096, Some(4096), 4096).unwrap();
        assert_eq!(slots, 3);
        assert_eq!(seq_offset(slots, 4096, 0), 0);
        assert_eq!(seq_offset(slots, 4096, 2), 8192);
        assert_eq!(seq_offset(slots, 4096, 3), 0);
        assert_eq!(rand_offset(4096, 4096, 7), 0);
    }
}
