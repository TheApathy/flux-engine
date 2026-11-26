// SPDX-License-Identifier: AGPL-3.0-only

//! Type-safe kernel argument builder for CUDA + Metal kernel launches.
//!
//! Replaces manual `Vec<*mut c_void>` construction with a builder
//! pattern that prevents parameter type/order mismatches AND records
//! per-arg type information so the metal backend can dispatch buffer
//! args via `setBuffer:offset:atIndex:` and scalar args via
//! `setBytes:length:atIndex:` (cuda's untyped `cuLaunchKernel` cannot
//! distinguish the two; metal cannot conflate them).
//!
//! # Usage
//!
//! ```ignore
//! KernelLaunch::new(gpu, kernel)
//!     .grid([num_tokens, 1, 1])
//!     .block([256, 1, 1])
//!     .arg_ptr(input)
//!     .arg_u32(hidden_size)
//!     .arg_f32(eps)
//!     .launch(stream)?;
//! ```
//!
//! Internally, every arg is recorded with its kind (Buffer / Scalar)
//! and its native byte width. `launch()` materializes a typed
//! `KernelArg` slice and calls `GpuBackend::launch_typed`. The
//! cuda backend's default `launch_typed` impl flattens that back
//! into the legacy `void**` shape; the metal backend overrides
//! `launch_typed` to thread the type info through to the encoder.

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

/// Per-arg metadata: which slot of `storage` it lives at, and how
/// many native bytes it occupies. Buffer args set `is_buffer = true`
/// and ignore `byte_len`. Scalar args set `is_buffer = false` and
/// record `byte_len = sizeof::<T>()`.
const INLINE_ARGS: usize = 32;

#[derive(Clone, Copy)]
struct ArgKind {
    is_buffer: bool,
    /// Byte count for scalar args (4 for u32/i32/f32, 8 for u64).
    /// Unused when `is_buffer` is true.
    byte_len: u8,
}

/// Builder for type-safe kernel launches across CUDA + Metal.
///
/// Accumulates grid dimensions, block dimensions, and typed kernel
/// arguments on the stack without heap allocations for kernels with
/// `<= 32` arguments.
pub struct KernelLaunch<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    pdl: bool,
    len: usize,
    inline_storage: [u64; INLINE_ARGS],
    inline_kinds: [ArgKind; INLINE_ARGS],
    overflow: Option<Vec<(u64, ArgKind)>>,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, kernel: KernelHandle) -> Self {
        Self {
            gpu,
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem: 0,
            pdl: false,
            len: 0,
            inline_storage: [0u64; INLINE_ARGS],
            inline_kinds: [ArgKind {
                is_buffer: false,
                byte_len: 0,
            }; INLINE_ARGS],
            overflow: None,
        }
    }

    pub fn grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }

    pub fn block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Enable CUDA Programmatic Dependent Launch (`PDL`) when `enabled` is true.
    pub fn pdl(mut self, enabled: bool) -> Self {
        self.pdl = enabled;
        self
    }

    #[inline]
    fn push_arg(&mut self, raw: u64, kind: ArgKind) {
        if self.len < INLINE_ARGS {
            self.inline_storage[self.len] = raw;
            self.inline_kinds[self.len] = kind;
        } else {
            self.overflow.get_or_insert_with(Vec::new).push((raw, kind));
        }
        self.len += 1;
    }

    /// Add a DevicePtr (u64) argument.
    pub fn arg_ptr(mut self, p: DevicePtr) -> Self {
        self.push_arg(
            p.0,
            ArgKind {
                is_buffer: true,
                byte_len: 0,
            },
        );
        self
    }

    /// Add a u32 argument.
    pub fn arg_u32(mut self, v: u32) -> Self {
        self.push_arg(
            v as u64,
            ArgKind {
                is_buffer: false,
                byte_len: 4,
            },
        );
        self
    }

    /// Add a u64 argument.
    pub fn arg_u64(mut self, v: u64) -> Self {
        self.push_arg(
            v,
            ArgKind {
                is_buffer: false,
                byte_len: 8,
            },
        );
        self
    }

    /// Add an i32 argument.
    pub fn arg_i32(mut self, v: i32) -> Self {
        self.push_arg(
            v as u32 as u64,
            ArgKind {
                is_buffer: false,
                byte_len: 4,
            },
        );
        self
    }

    /// Add an f32 argument.
    pub fn arg_f32(mut self, v: f32) -> Self {
        self.push_arg(
            f32::to_bits(v) as u64,
            ArgKind {
                is_buffer: false,
                byte_len: 4,
            },
        );
        self
    }

    /// Execute the kernel launch via `GpuBackend::launch_typed_pdl`.
    pub fn launch(self, stream: u64) -> Result<()> {
        if self.len <= INLINE_ARGS {
            let mut args = [KernelArg::Buffer(DevicePtr(0)); INLINE_ARGS];
            for (idx, dst) in args.iter_mut().enumerate().take(self.len) {
                let slot = &self.inline_storage[idx];
                let kind = self.inline_kinds[idx];
                *dst = if kind.is_buffer {
                    KernelArg::Buffer(DevicePtr(*slot))
                } else {
                    // SAFETY: `slot` is an in-bounds `u64` inside `self.inline_storage`
                    // (`kind.byte_len <= 8`) that outlives `args` in this stack frame.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            slot as *const u64 as *const u8,
                            kind.byte_len as usize,
                        )
                    };
                    KernelArg::Bytes(bytes)
                };
            }
            if self.pdl {
                return self.gpu.launch_typed_pdl(
                    self.kernel,
                    self.grid,
                    self.block,
                    self.shared_mem,
                    stream,
                    &args[..self.len],
                    true,
                );
            }
            return self.gpu.launch_typed(
                self.kernel,
                self.grid,
                self.block,
                self.shared_mem,
                stream,
                &args[..self.len],
            );
        }

        let overflow = self.overflow.as_deref().unwrap_or(&[]);
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(self.len);
        for (slot, kind) in self.inline_storage.iter().zip(self.inline_kinds.iter()) {
            if kind.is_buffer {
                args.push(KernelArg::Buffer(DevicePtr(*slot)));
            } else {
                // SAFETY: `slot` is a valid `u64` in `self.inline_storage` (`byte_len <= 8`)
                // whose borrow outlives `args`.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        slot as *const u64 as *const u8,
                        kind.byte_len as usize,
                    )
                };
                args.push(KernelArg::Bytes(bytes));
            }
        }
        for (slot, kind) in overflow {
            if kind.is_buffer {
                args.push(KernelArg::Buffer(DevicePtr(*slot)));
            } else {
                // SAFETY: `slot` is a valid `u64` in `overflow` (`byte_len <= 8`) without
                // reallocation before `launch_typed[_pdl]` returns.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        slot as *const u64 as *const u8,
                        kind.byte_len as usize,
                    )
                };
                args.push(KernelArg::Bytes(bytes));
            }
        }
        if self.pdl {
            self.gpu.launch_typed_pdl(
                self.kernel,
                self.grid,
                self.block,
                self.shared_mem,
                stream,
                &args,
                true,
            )
        } else {
            self.gpu.launch_typed(
                self.kernel,
                self.grid,
                self.block,
                self.shared_mem,
                stream,
                &args,
            )
        }
    }
}

/// Convenience: divide and round up.
pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn test_kernel_launch_builder() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "test_kernel").unwrap();

        let result = KernelLaunch::new(&gpu, kernel)
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr(0x1000))
            .arg_u32(42)
            .arg_f32(1.5)
            .launch(0);

        assert!(result.is_ok());
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 256), 1);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
    }
}
