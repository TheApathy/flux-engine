// SPDX-License-Identifier: AGPL-3.0-only

//! Page-aligned O_DIRECT I/O primitives for [`super::FastSafetensorsLoader`].

use anyhow::{Result, bail};
use std::alloc::{Layout, alloc, dealloc};
use std::fs::File;
use std::path::Path;

/// O_DIRECT requires offsets, sizes, and buffers to be aligned to the
/// filesystem block size. 4 KiB is the upper bound on every Linux arch we
/// target and satisfies every fs we've seen.
pub(super) const O_DIRECT_ALIGN: usize = 4096;

/// Heap buffer aligned to [`O_DIRECT_ALIGN`]. `Send` so it can cross a channel
/// from the reader thread to the copier thread.
pub(super) struct AlignedBuffer {
    ptr: *mut u8,
    cap: usize,
    initialized: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` owns the allocation pointed to by `ptr`; the
// pointer is created by `std::alloc::alloc` with the recorded `layout` and
// is freed in `Drop` with the matching layout. The struct holds no shared
// references and exposes only initialized immutable `&[u8]` slices
// (`0..self.initialized`) outside `read_tensor_aligned`, so sharing
// `Arc<AlignedBuffer>` across threads for coalesced read runs is safe.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    fn new(cap_bytes: usize) -> Self {
        let cap = cap_bytes
            .max(1)
            .checked_next_multiple_of(O_DIRECT_ALIGN)
            .expect("AlignedBuffer size overflow");
        let layout = Layout::from_size_align(cap, O_DIRECT_ALIGN).expect("valid layout");
        // SAFETY: `cap >= O_DIRECT_ALIGN > 0`, so `layout.size()` is strictly non-zero.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            cap,
            initialized: 0,
            layout,
        }
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to a valid allocation of `self.cap` bytes and
        // `self.initialized <= self.cap` bytes have been initialized by `pread`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.initialized) }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Mark the first `len` bytes as initialized after populating them via `pread`.
    ///
    /// # Safety
    /// Caller must have initialized `self.ptr[..len]` with `len <= self.cap`.
    unsafe fn set_initialized(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.initialized = len.min(self.cap);
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated by `std::alloc::alloc(self.layout)` in
        // `AlignedBuffer::new` and is deallocated once on drop.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[cfg(target_os = "linux")]
pub(super) fn open_direct(path: &Path) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let flags = libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC;
    // SAFETY: `cstr` is a valid NUL-terminated path and `O_RDONLY | O_DIRECT | O_CLOEXEC`
    // does not include `O_CREAT`, so `libc::open` takes no mode argument.
    let fd = unsafe { libc::open(cstr.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd >= 0` is a freshly opened, uniquely owned file descriptor from `libc::open`.
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_os = "linux"))]
pub(super) fn open_direct(_path: &Path) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "O_DIRECT only supported on Linux",
    ))
}

/// Read a tensor's bytes into an aligned buffer. Returns the buffer and the
/// offset within it where the tensor's `len` bytes start.
///
/// With `using_direct=true`, the read window is widened to the nearest 4 KiB
/// boundaries so the offset/size/buffer alignment constraints are met. If the
/// aligned window extends past end-of-file, we let the kernel return a short
/// read for the trailing fragment (Linux ≥ 2.6 accepts this for O_DIRECT on
/// mainstream filesystems — we only require that the tensor's exact bytes
/// have been populated, which the post-loop check enforces).
pub(super) fn read_tensor_aligned(
    fd: std::os::unix::io::RawFd,
    abs_offset: u64,
    len: usize,
    using_direct: bool,
) -> Result<(AlignedBuffer, usize)> {
    let (window_start, window_len, slice_off) = if using_direct {
        let ws = abs_offset - (abs_offset % O_DIRECT_ALIGN as u64);
        let unaligned_end = abs_offset
            .checked_add(u64::try_from(len)?)
            .ok_or_else(|| anyhow::anyhow!("read_tensor_aligned: offset + len overflow"))?;
        let aligned_end = unaligned_end
            .checked_next_multiple_of(O_DIRECT_ALIGN as u64)
            .ok_or_else(|| anyhow::anyhow!("read_tensor_aligned: aligned_end overflow"))?;
        let wl = usize::try_from(aligned_end - ws)?;
        (ws, wl, usize::try_from(abs_offset - ws)?)
    } else {
        (abs_offset, len, 0usize)
    };

    let mut buf = AlignedBuffer::new(window_len);
    let mut filled = 0usize;
    while filled < window_len {
        let off = window_start
            .checked_add(filled as u64)
            .and_then(|v| libc::off_t::try_from(v).ok())
            .ok_or_else(|| anyhow::anyhow!("read_tensor_aligned: pread offset overflow"))?;
        // SAFETY: `fd` is a valid open descriptor borrowed for this call; `buf.as_mut_ptr().add(filled)`
        // points to `window_len - filled` writable bytes within `buf.cap` (`window_len <= buf.cap`).
        let ret = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr().add(filled) as *mut libc::c_void,
                window_len - filled,
                off,
            )
        };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            // On O_DIRECT, a non-aligned tail at EOF may return EINVAL on
            // some filesystems. Accept the short read as long as we've
            // already covered the tensor's exact range.
            if filled >= slice_off + len {
                break;
            }
            bail!(
                "pread failed at offset {}: {err}",
                window_start + filled as u64
            );
        }
        if ret == 0 {
            break; // EOF
        }
        filled += ret as usize;
    }
    if filled < slice_off + len {
        bail!(
            "short read: got {} bytes, need at least {} (tensor spans offset {}..{})",
            filled,
            slice_off + len,
            abs_offset,
            abs_offset + len as u64
        );
    }
    // SAFETY: `pread` initialized `buf[..filled]` above and `filled <= window_len <= buf.cap`.
    unsafe { buf.set_initialized(filled) };
    Ok((buf, slice_off))
}
