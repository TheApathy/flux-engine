// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal DLPack tensor types retained for source compatibility with the
// upstream `xgrammar-rs` FFI surface.
//
// The pure-Rust `GrammarMatcher::fill_next_token_bitmask` operates on
// `&mut [i32]` directly rather than crossing an FFI boundary via
// `DLTensor*`. These `#[repr(C)]` structs preserve the DLPack ABI layout
// for downstream callers that still reference the re-exported type names.

use std::ffi::c_void;

/// DLPack device type enum (`DLDeviceType`).
///
/// Discriminants match the DLPack ABI so FFI consumers stay compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum DLDeviceType {
    /// CPU device.
    kDLCPU = 1,
    /// CUDA GPU device.
    kDLCUDA = 2,
}

/// DLPack data type code enum (`DLDataTypeCode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum DLDataTypeCode {
    /// Signed integer.
    kDLInt = 0,
    /// Unsigned integer.
    kDLUInt = 1,
    /// IEEE floating point.
    kDLFloat = 2,
}

/// DLPack device descriptor (`DLDevice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct DLDevice {
    /// The device type.
    pub device_type: DLDeviceType,
    /// The device index.
    pub device_id: i32,
}

/// DLPack data type descriptor (`DLDataType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct DLDataType {
    /// The type code (see [`DLDataTypeCode`]).
    pub code: u8,
    /// Bits per element.
    pub bits: u8,
    /// Number of lanes (vector width).
    pub lanes: u16,
}

/// DLPack tensor view (`DLTensor`) — does not own memory.
///
/// Retains the standard DLPack C ABI layout for API compatibility; the
/// pure-Rust matcher façade takes `&mut [i32]` slices directly.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the underlying buffer.
    pub data: *mut c_void,
    /// The device the buffer lives on.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// The element data type.
    pub dtype: DLDataType,
    /// Pointer to the shape array (`ndim` entries).
    pub shape: *mut i64,
    /// Pointer to the strides array, or null for row-major.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first element.
    pub byte_offset: u64,
}
