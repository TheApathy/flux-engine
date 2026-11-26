// SPDX-License-Identifier: AGPL-3.0-only

//! Device-context wrapper + per-hardware constants.
//!
//! `sm121` is a pure-constants module (NUM_SMS, SMEM_PER_SM, …) that
//! kernel-launch heuristics in spark-model consume regardless of the
//! active backend — keeping it always-compiled lets metal-feature
//! builds reuse the same dispatch tables and code paths.
//!
//! `AtlasDevice` is the cudarc-wrapped device handle, gated behind
//! the `cuda` feature.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// Runtime properties queried from the active GPU device.
///
/// Populated once when the backend initializes (e.g. `AtlasCudaBackend::new`)
/// so hot-path kernel launch heuristics and memory-topology checks can adapt
/// dynamically across integrated (GB10 UMA) and discrete GPUs (`SM120` / `SM103`)
/// without hardcoded hardware assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDeviceProps {
    /// CUDA compute capability major version (e.g. `12` for SM120/SM121, `10` for SM103).
    pub compute_major: u32,
    /// CUDA compute capability minor version (e.g. `0` for SM120, `1` for SM121, `3` for SM103).
    pub compute_minor: u32,
    /// Number of streaming multiprocessors reported by the device.
    pub sm_count: u32,
    /// Maximum shared memory per streaming multiprocessor, in bytes.
    pub smem_per_sm: usize,
    /// `true` for unified/integrated host-shared memory (e.g. GB10 UMA);
    /// `false` for discrete GPUs with dedicated VRAM.
    pub is_integrated: bool,
    /// Total device memory reported by the driver, in bytes.
    pub total_mem_bytes: usize,
}

static ACTIVE_DEVICE_PROPS: RwLock<Option<GpuDeviceProps>> = RwLock::new(None);
/// Lock-free mirror of [`GpuDeviceProps::sm_count`] (`0` = unset).
static ACTIVE_SM_COUNT: AtomicU32 = AtomicU32::new(0);
/// Lock-free mirror of [`GpuDeviceProps::is_integrated`]:
/// `0` = unset, `1` = discrete (`false`), `2` = integrated (`true`).
static ACTIVE_IS_INTEGRATED: AtomicU8 = AtomicU8::new(0);
/// Lock-free mirror of `(compute_major, compute_minor)` packed as
/// `((major as u64) << 32) | (minor as u64)`, published before
/// `ACTIVE_IS_INTEGRATED` becomes non-zero with `Release` ordering.
static ACTIVE_COMPUTE_CAP: AtomicU64 = AtomicU64::new(0);

const INTEGRATED_UNSET: u8 = 0;
const INTEGRATED_FALSE: u8 = 1;
const INTEGRATED_TRUE: u8 = 2;

/// Record runtime properties for the active GPU device.
///
/// Updates both the full property snapshot and the lock-free hot-path atomics
/// read by [`active_sm_count`], [`active_device_is_integrated`], and
/// [`active_compute_capability`].
pub fn set_active_device_props(props: GpuDeviceProps) {
    let mut guard = ACTIVE_DEVICE_PROPS
        .write()
        .expect("active device props lock poisoned");
    *guard = Some(props);
    let packed_cc = (u64::from(props.compute_major) << 32) | u64::from(props.compute_minor);
    ACTIVE_COMPUTE_CAP.store(packed_cc, Ordering::Relaxed);
    ACTIVE_SM_COUNT.store(props.sm_count, Ordering::Relaxed);
    let integrated_tag = if props.is_integrated {
        INTEGRATED_TRUE
    } else {
        INTEGRATED_FALSE
    };
    ACTIVE_IS_INTEGRATED.store(integrated_tag, Ordering::Release);
}

/// Return the full [`GpuDeviceProps`] snapshot recorded for the active device,
/// or `None` if no backend has populated it yet.
#[must_use]
pub fn active_device_props() -> Option<GpuDeviceProps> {
    *ACTIVE_DEVICE_PROPS
        .read()
        .expect("active device props lock poisoned")
}

/// Return the active device's streaming-multiprocessor count if populated and
/// non-zero, falling back to [`sm121::NUM_SMS`] (`48`).
#[must_use]
pub fn active_sm_count() -> u32 {
    let count = ACTIVE_SM_COUNT.load(Ordering::Relaxed);
    if count > 0 { count } else { sm121::NUM_SMS }
}

/// Return whether the active device uses integrated/unified memory (`Some(true)`)
/// or discrete VRAM (`Some(false)`), or `None` before device properties are set.
#[must_use]
pub fn active_device_is_integrated() -> Option<bool> {
    match ACTIVE_IS_INTEGRATED.load(Ordering::Acquire) {
        INTEGRATED_FALSE => Some(false),
        INTEGRATED_TRUE => Some(true),
        _ => None,
    }
}

/// Return the active device's `(compute_major, compute_minor)` once populated,
/// or `None` before device properties are set.
#[must_use]
pub fn active_compute_capability() -> Option<(u32, u32)> {
    if ACTIVE_IS_INTEGRATED.load(Ordering::Acquire) == INTEGRATED_UNSET {
        return None;
    }
    let packed = ACTIVE_COMPUTE_CAP.load(Ordering::Relaxed);
    Some(((packed >> 32) as u32, packed as u32))
}

/// SM121 hardware constants for DGX Spark GB10.
pub mod sm121 {
    /// Number of streaming multiprocessors
    pub const NUM_SMS: u32 = 48;

    /// Shared memory per SM (bytes)
    pub const SMEM_PER_SM: usize = 99 * 1024; // 99 KB

    /// Max registers per thread
    pub const MAX_REGS_PER_THREAD: u32 = 255;

    /// Max threads per block
    pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

    /// Warp size
    pub const WARP_SIZE: u32 = 32;

    /// Memory bandwidth (GB/s) — LPDDR5X unified
    pub const MEMORY_BW_GBS: f64 = 273.0;

    /// Compute capability
    pub const COMPUTE_MAJOR: u32 = 12;
    pub const COMPUTE_MINOR: u32 = 1;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    use crate::error::{AtlasError, Result};

    /// Wrapper around a cudarc CudaContext with SM121-specific configuration.
    #[derive(Clone)]
    pub struct AtlasDevice {
        pub ctx: Arc<CudaContext>,
        pub ordinal: usize,
    }

    impl AtlasDevice {
        /// Initialize an Atlas device on the given GPU ordinal.
        pub fn new(ordinal: usize) -> Result<Self> {
            let ctx = CudaContext::new(ordinal).map_err(AtlasError::CudaDriver)?;
            Ok(Self { ctx, ordinal })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::AtlasDevice;

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn clear_active_device_props_for_test() {
        let mut guard = ACTIVE_DEVICE_PROPS
            .write()
            .expect("active device props lock poisoned");
        *guard = None;
        ACTIVE_IS_INTEGRATED.store(INTEGRATED_UNSET, Ordering::Release);
        ACTIVE_SM_COUNT.store(0, Ordering::Relaxed);
        ACTIVE_COMPUTE_CAP.store(0, Ordering::Relaxed);
    }

    struct ResetGuard;

    impl Drop for ResetGuard {
        fn drop(&mut self) {
            clear_active_device_props_for_test();
        }
    }

    #[test]
    fn unset_device_props_fall_back_to_sm121_sm_count() {
        let _lock = TEST_LOCK.lock().expect("test lock poisoned");
        clear_active_device_props_for_test();
        let _reset = ResetGuard;

        assert_eq!(active_device_props(), None);
        assert_eq!(active_sm_count(), sm121::NUM_SMS);
        assert_eq!(active_device_is_integrated(), None);
        assert_eq!(active_compute_capability(), None);
    }

    #[test]
    fn set_active_device_props_updates_discrete_and_integrated_views() {
        let _lock = TEST_LOCK.lock().expect("test lock poisoned");
        clear_active_device_props_for_test();
        let _reset = ResetGuard;

        let discrete_sm120 = GpuDeviceProps {
            compute_major: 12,
            compute_minor: 0,
            sm_count: 188,
            smem_per_sm: 100 * 1024,
            is_integrated: false,
            total_mem_bytes: 96 * 1024 * 1024 * 1024,
        };
        set_active_device_props(discrete_sm120);
        assert_eq!(active_device_props(), Some(discrete_sm120));
        assert_eq!(active_sm_count(), 188);
        assert_eq!(active_device_is_integrated(), Some(false));
        assert_eq!(active_compute_capability(), Some((12, 0)));

        let integrated_sm121 = GpuDeviceProps {
            compute_major: 12,
            compute_minor: 1,
            sm_count: 48,
            smem_per_sm: sm121::SMEM_PER_SM,
            is_integrated: true,
            total_mem_bytes: 128 * 1024 * 1024 * 1024,
        };
        set_active_device_props(integrated_sm121);
        assert_eq!(active_device_props(), Some(integrated_sm121));
        assert_eq!(active_sm_count(), 48);
        assert_eq!(active_device_is_integrated(), Some(true));
        assert_eq!(active_compute_capability(), Some((12, 1)));
    }

    #[test]
    fn zero_sm_count_falls_back_to_sm121_while_preserving_other_fields() {
        let _lock = TEST_LOCK.lock().expect("test lock poisoned");
        clear_active_device_props_for_test();
        let _reset = ResetGuard;

        let props = GpuDeviceProps {
            compute_major: 10,
            compute_minor: 3,
            sm_count: 0,
            smem_per_sm: 228 * 1024,
            is_integrated: false,
            total_mem_bytes: 288 * 1024 * 1024 * 1024,
        };
        set_active_device_props(props);
        assert_eq!(active_device_props(), Some(props));
        assert_eq!(active_sm_count(), sm121::NUM_SMS);
        assert_eq!(active_device_is_integrated(), Some(false));
        assert_eq!(active_compute_capability(), Some((10, 3)));
    }
}
