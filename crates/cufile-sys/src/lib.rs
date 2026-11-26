// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal raw FFI for NVIDIA's libcufile (GPUDirect Storage / GDS).
//!
//! Loaded via dlopen so a binary can probe whether GDS is available
//! without failing to launch when `libcufile.so` is absent. Only the
//! symbols Atlas actually uses for the high-speed-swap path are bound
//! here; extend as needed.
//!
//! GDS is not currently supported on GB10 hardware (see
//! `docs/adr/0008-nvme-high-speed-swap.md`) — this crate is dormant on
//! that target and will become live when a GDS-capable platform lands.

#![deny(clippy::all)]
#![allow(non_camel_case_types, non_snake_case)]

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use std::os::raw::c_long;

pub const CUFILEOP_BASE_ERR: i32 = 5000;
pub const CU_FILE_SUCCESS: i32 = 0;
pub const CU_FILE_DRIVER_NOT_INITIALIZED: i32 = CUFILEOP_BASE_ERR + 1;
pub const CU_FILE_PLATFORM_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 7;
pub const CU_FILE_IO_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 8;
pub const CU_FILE_DEVICE_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 9;

pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

pub type CUfileHandle_t = *mut c_void;
pub type CUresult = c_int;
pub type CUfileOpError = c_int;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileError_t {
    pub err: CUfileOpError,
    pub cu_err: CUresult,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union CUfileDescrHandle {
    pub fd: c_int,
    pub handle: *mut c_void,
}

impl std::fmt::Debug for CUfileDescrHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not read union variants in `Debug`: callers may construct
        // `CUfileDescrHandle { fd }` directly leaving upper bytes uninitialized.
        f.debug_struct("CUfileDescrHandle").finish_non_exhaustive()
    }
}

impl CUfileDescrHandle {
    pub const fn from_fd(fd: c_int) -> Self {
        // Zero the full union first so upper bytes on 64-bit targets are
        // deterministic before populating the Linux `fd` field.
        let mut u = Self {
            handle: core::ptr::null_mut(),
        };
        u.fd = fd;
        u
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileDescr_t {
    pub type_: c_int,
    pub handle: CUfileDescrHandle,
    pub fs_ops: *const c_void,
}

// Verify C ABI layout matches `/usr/local/cuda/include/cufile.h` on 64-bit targets:
//   struct CUfileDescr_t {
//       enum CUfileFileHandleType type;      // offset 0 (4 bytes + 4 bytes pad)
//       union { int fd; void *handle; };     // offset 8 (8 bytes, align 8)
//       const CUfileFSOps_t *fs_ops;         // offset 16 (8 bytes, align 8)
//   };                                       // total 24 bytes
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(core::mem::size_of::<CUfileDescrHandle>() == 8);
    assert!(core::mem::align_of::<CUfileDescrHandle>() == 8);
    assert!(core::mem::size_of::<CUfileDescr_t>() == 24);
    assert!(core::mem::align_of::<CUfileDescr_t>() == 8);
    assert!(core::mem::offset_of!(CUfileDescr_t, type_) == 0);
    assert!(core::mem::offset_of!(CUfileDescr_t, handle) == 8);
    assert!(core::mem::offset_of!(CUfileDescr_t, fs_ops) == 16);
};

pub type FnDriverOpen = unsafe extern "C" fn() -> CUfileError_t;
pub type FnDriverClose = unsafe extern "C" fn() -> CUfileError_t;
pub type FnHandleRegister =
    unsafe extern "C" fn(*mut CUfileHandle_t, *mut CUfileDescr_t) -> CUfileError_t;
pub type FnHandleDeregister = unsafe extern "C" fn(CUfileHandle_t);
pub type FnBufRegister = unsafe extern "C" fn(*const c_void, libc::size_t, c_int) -> CUfileError_t;
pub type FnBufDeregister = unsafe extern "C" fn(*const c_void) -> CUfileError_t;
pub type FnRead = unsafe extern "C" fn(
    CUfileHandle_t,
    *mut c_void,
    libc::size_t,
    c_long,
    c_long,
) -> libc::ssize_t;
pub type FnWrite = unsafe extern "C" fn(
    CUfileHandle_t,
    *const c_void,
    libc::size_t,
    c_long,
    c_long,
) -> libc::ssize_t;
pub type FnGetVersion = unsafe extern "C" fn(*mut c_int) -> CUfileError_t;

pub struct CuFile {
    _lib: Library,
    pub driver_open: FnDriverOpen,
    pub driver_close: FnDriverClose,
    pub handle_register: FnHandleRegister,
    pub handle_deregister: FnHandleDeregister,
    pub buf_register: FnBufRegister,
    pub buf_deregister: FnBufDeregister,
    pub read: FnRead,
    pub write: FnWrite,
    pub get_version: FnGetVersion,
}

const SEARCH_PATHS: &[&str] = &[
    "libcufile.so.0",
    "libcufile.so",
    "/usr/local/cuda/targets/sbsa-linux/lib/libcufile.so.0",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0",
    "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcufile.so.0",
];

impl CuFile {
    pub fn load() -> Result<Self, String> {
        let mut last_err = String::new();
        for path in SEARCH_PATHS {
            // SAFETY: Loading NVIDIA's `libcufile.so` shared library from
            // standard loader search paths; library initialization routines
            // are standard C shared-object constructors.
            match unsafe { Library::new(path) } {
                Ok(lib) => return Self::resolve(lib).map_err(|e| format!("{path}: {e}")),
                Err(e) => last_err = format!("{path}: {e}"),
            }
        }
        Err(format!("libcufile not found ({last_err})"))
    }

    fn resolve(lib: Library) -> Result<Self, String> {
        /// # Safety
        /// `name` must be a NUL-terminated symbol name exported by `lib` with
        /// the exact C ABI signature represented by `T`.
        unsafe fn sym<'a, T: Copy + 'a>(lib: &'a Library, name: &[u8]) -> Result<T, String> {
            // SAFETY: Caller guarantees `name` matches the C symbol signature `T`,
            // and `lib` outlives the copied function pointer via `CuFile._lib`.
            unsafe {
                let s: Symbol<'a, T> = lib
                    .get(name)
                    .map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy(name)))?;
                Ok(*s)
            }
        }
        // SAFETY: Every symbol name and function pointer signature below matches
        // NVIDIA's `cufile.h` C ABI, and `lib` is retained inside `Self._lib`
        // for the lifetime of the resolved function pointers.
        unsafe {
            let driver_open = sym::<FnDriverOpen>(&lib, b"cuFileDriverOpen\0")?;
            let driver_close = sym::<FnDriverClose>(&lib, b"cuFileDriverClose\0")?;
            let handle_register = sym::<FnHandleRegister>(&lib, b"cuFileHandleRegister\0")?;
            let handle_deregister = sym::<FnHandleDeregister>(&lib, b"cuFileHandleDeregister\0")?;
            let buf_register = sym::<FnBufRegister>(&lib, b"cuFileBufRegister\0")?;
            let buf_deregister = sym::<FnBufDeregister>(&lib, b"cuFileBufDeregister\0")?;
            let read = sym::<FnRead>(&lib, b"cuFileRead\0")?;
            let write = sym::<FnWrite>(&lib, b"cuFileWrite\0")?;
            let get_version = sym::<FnGetVersion>(&lib, b"cuFileGetVersion\0")?;
            Ok(Self {
                _lib: lib,
                driver_open,
                driver_close,
                handle_register,
                handle_deregister,
                buf_register,
                buf_deregister,
                read,
                write,
                get_version,
            })
        }
    }
}

pub fn err_to_str(err: CUfileOpError) -> &'static str {
    match err {
        0 => "success",
        x if x == CU_FILE_DRIVER_NOT_INITIALIZED => "CU_FILE_DRIVER_NOT_INITIALIZED",
        x if x == CU_FILE_PLATFORM_NOT_SUPPORTED => "CU_FILE_PLATFORM_NOT_SUPPORTED",
        x if x == CU_FILE_IO_NOT_SUPPORTED => "CU_FILE_IO_NOT_SUPPORTED",
        x if x == CU_FILE_DEVICE_NOT_SUPPORTED => "CU_FILE_DEVICE_NOT_SUPPORTED",
        _ => "CU_FILE_OTHER_ERROR",
    }
}

pub fn nvidia_fs_loaded() -> bool {
    let modules = match std::fs::read_to_string("/proc/modules") {
        Ok(s) => s,
        Err(_) => return false,
    };
    modules
        .lines()
        .any(|l| l.starts_with("nvidia_fs ") || l.starts_with("nvidia-fs "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cufile_descr_layout_matches_c_abi() {
        assert_eq!(
            core::mem::align_of::<CUfileDescrHandle>(),
            core::mem::align_of::<*mut c_void>()
        );
        assert_eq!(
            core::mem::size_of::<CUfileDescrHandle>(),
            core::mem::size_of::<*mut c_void>()
        );

        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(core::mem::size_of::<CUfileDescr_t>(), 24);
            assert_eq!(core::mem::align_of::<CUfileDescr_t>(), 8);
            assert_eq!(core::mem::offset_of!(CUfileDescr_t, type_), 0);
            assert_eq!(core::mem::offset_of!(CUfileDescr_t, handle), 8);
            assert_eq!(core::mem::offset_of!(CUfileDescr_t, fs_ops), 16);

            let descr = CUfileDescr_t {
                type_: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
                handle: CUfileDescrHandle::from_fd(42),
                fs_ops: core::ptr::null(),
            };
            // SAFETY: `descr` is a valid initialized `CUfileDescr_t`; reading the
            // initialized `type_` (offset 0) and `handle.fd` (offset 8) fields
            // confirms the C ABI layout places `fd` at byte offset 8, not 4.
            unsafe {
                let base = core::ptr::from_ref(&descr).cast::<u8>();
                let raw_type = core::ptr::read_unaligned(base.cast::<c_int>());
                let raw_fd = core::ptr::read_unaligned(base.add(8).cast::<c_int>());
                assert_eq!(raw_type, CU_FILE_HANDLE_TYPE_OPAQUE_FD);
                assert_eq!(raw_fd, 42);
                assert_eq!(descr.handle.fd, 42);
            }
        }
    }
}
