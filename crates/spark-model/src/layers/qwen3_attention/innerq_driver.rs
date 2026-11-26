// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side driver for TurboQuant+ InnerQ per-channel K equalization.
//!
//! Triggers via `TURBO_INNERQ=N` env var (N = calibration token count). The
//! kernel-side state lives in `kernels/gb10/common/tq_plus_innerq.cu` as a
//! set of `__device__` globals inside `namespace tq_plus`. PTX strips the
//! companion host functions in that translation unit, so this driver
//! reproduces their work directly via the CUDA Driver API:
//!
//!   `cuModuleGetGlobal_v2` → device pointer for each symbol
//!   synchronous pageable-safe CUDA copies → push/pull state
//!
//! Two-phase operation:
//!   1. `start()`     — zero counters, set `d_innerq_calibrating = 1`.
//!   2. `maybe_finalize()` — read `d_innerq_count`; once it crosses
//!      `target_tokens`, read `d_innerq_sq_accum`, compute per-channel
//!      scale + scale_inv, upload, set `d_innerq_active = 1`.
//!
//! Math identity: `<(H q)/s, s·(H k)> = <H q, H k> = <q, k>` — the kernel-side
//! apply pass multiplies both Q (by `scale_inv`) and K (by `scale`) post-WHT,
//! leaving attention dot products unchanged while smoothing K variance across
//! channels.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use atlas_core::registry::AtlasRegistry;

// Itanium-mangled names for `tq_plus::*` device globals. The kernel TU is
// `kernels/gb10/common/tq_plus_innerq.cu`, which compiles to PTX module
// `tq_plus_innerq` (no [modules] override in common/KERNEL.toml).
const MODULE: &str = "tq_plus_innerq";
const SYM_SCALE: &str = "_ZN7tq_plus14d_innerq_scaleE";
const SYM_SCALE_INV: &str = "_ZN7tq_plus18d_innerq_scale_invE";
const SYM_SQ_ACCUM: &str = "_ZN7tq_plus17d_innerq_sq_accumE";
const SYM_COUNT: &str = "_ZN7tq_plus14d_innerq_countE";
const SYM_ACTIVE: &str = "_ZN7tq_plus15d_innerq_activeE";
const SYM_CALIBRATING: &str = "_ZN7tq_plus20d_innerq_calibratingE";

// Matches INNERQ_MAX_CHANNELS in tq_plus_innerq.cuh. Head dim = 128 today.
const MAX_CHANNELS: usize = 128;

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and every byte is initialized while the
    // returned slice cannot outlive the shared source borrow.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: every `f32` bit pattern is valid and the exclusive byte view is
    // bounded by the exclusive source borrow.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

fn i32_bytes(value: &i32) -> &[u8] {
    // SAFETY: `i32` has no padding and the byte view shares its source borrow.
    unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(value).cast::<u8>(),
            std::mem::size_of::<i32>(),
        )
    }
}

fn i32_bytes_mut(value: &mut i32) -> &mut [u8] {
    // SAFETY: every `i32` bit pattern is valid and the byte view is exclusive.
    unsafe {
        std::slice::from_raw_parts_mut(
            std::ptr::from_mut(value).cast::<u8>(),
            std::mem::size_of::<i32>(),
        )
    }
}

pub struct InnerQDriver {
    pub target_tokens: i32,
    pub strength: f32,
    pub calibrating: AtomicBool,
    pub finalized: AtomicBool,
}

impl InnerQDriver {
    /// Reads `TURBO_INNERQ` and `TURBO_INNERQ_STRENGTH` env vars. Returns
    /// `None` if `TURBO_INNERQ` is unset, unparsable, or `<= 0`.
    pub fn from_env() -> Option<Self> {
        let n = std::env::var("TURBO_INNERQ")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&n| n > 0)?;
        let strength: f32 = std::env::var("TURBO_INNERQ_STRENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s: &f32| s > 0.0 && s <= 1.0)
            .unwrap_or(0.5);
        Some(Self {
            target_tokens: n,
            strength,
            calibrating: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
        })
    }

    /// Enter calibration phase: zero `d_innerq_sq_accum` / `d_innerq_count`
    /// / `d_innerq_active`, set `d_innerq_calibrating = 1`. Idempotent.
    pub fn start(&self) -> Result<()> {
        let reg = AtlasRegistry::get();
        let stream = reg.raw_stream();

        let zeros_f32 = [0.0f32; MAX_CHANNELS];
        let zero_i32: i32 = 0;
        let one_i32: i32 = 1;

        let (sq_ptr, sq_bytes) = reg
            .device_symbol(MODULE, SYM_SQ_ACCUM)
            .with_context(|| format!("resolve {MODULE}::{SYM_SQ_ACCUM}"))?;
        let (count_ptr, count_bytes) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let (active_ptr, active_bytes) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let (calib_ptr, calib_bytes) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;

        let zeros_bytes = f32_bytes(&zeros_f32);
        let zero_i32_bytes = i32_bytes(&zero_i32);
        let one_i32_bytes = i32_bytes(&one_i32);
        anyhow::ensure!(
            sq_bytes >= zeros_bytes.len(),
            "{MODULE}::{SYM_SQ_ACCUM} too small: {sq_bytes} < {}",
            zeros_bytes.len()
        );
        anyhow::ensure!(
            count_bytes >= zero_i32_bytes.len(),
            "{MODULE}::{SYM_COUNT} too small: {count_bytes} < {}",
            zero_i32_bytes.len()
        );
        anyhow::ensure!(
            active_bytes >= zero_i32_bytes.len(),
            "{MODULE}::{SYM_ACTIVE} too small: {active_bytes} < {}",
            zero_i32_bytes.len()
        );
        anyhow::ensure!(
            calib_bytes >= one_i32_bytes.len(),
            "{MODULE}::{SYM_CALIBRATING} too small: {calib_bytes} < {}",
            one_i32_bytes.len()
        );
        // SAFETY: `sq_ptr`, `count_ptr`, `active_ptr`, and `calib_ptr` were
        // resolved from live module globals in `tq_plus_innerq` with verified
        // byte sizes >= each host slice, and `stream` comes from `reg.raw_stream()`.
        unsafe {
            reg.copy_h2d_group(
                &[
                    (sq_ptr, zeros_bytes),
                    (count_ptr, zero_i32_bytes),
                    (active_ptr, zero_i32_bytes),
                    (calib_ptr, one_i32_bytes),
                ],
                stream,
            )?;
        }

        self.calibrating.store(true, Ordering::Release);
        self.finalized.store(false, Ordering::Release);
        tracing::info!(
            "InnerQ calibration started: target={} tokens, strength={:.2}",
            self.target_tokens,
            self.strength,
        );
        Ok(())
    }

    /// Poll `d_innerq_count`. When it crosses `target_tokens`, pull
    /// `d_innerq_sq_accum`, compute per-channel scale/scale_inv, upload,
    /// and flip `d_innerq_active = 1`. Returns `Ok(true)` on the call
    /// that activates, `Ok(false)` on every other call (including
    /// auto-disable when channels are already balanced).
    pub fn maybe_finalize(&self, group_size: i32) -> Result<bool> {
        if self.finalized.load(Ordering::Acquire) {
            return Ok(false);
        }
        let gs = group_size as usize;
        if gs == 0 || gs > MAX_CHANNELS {
            bail!("group_size {group_size} out of range (1..={MAX_CHANNELS})");
        }

        let reg = AtlasRegistry::get();
        let stream = reg.raw_stream();

        let (count_ptr, count_bytes) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let mut count: i32 = 0;
        let count_slice = i32_bytes_mut(&mut count);
        anyhow::ensure!(
            count_bytes >= count_slice.len(),
            "{MODULE}::{SYM_COUNT} too small: {count_bytes} < {}",
            count_slice.len()
        );
        // SAFETY: `count_ptr` is the resolved device `int` symbol `d_innerq_count`
        // with verified byte size >= `count_slice.len()`, and `stream` is `reg.raw_stream()`.
        unsafe { reg.copy_d2h(count_slice, count_ptr, stream)? };

        if count < self.target_tokens {
            return Ok(false);
        }

        let (sq_ptr, sq_bytes) = reg.device_symbol(MODULE, SYM_SQ_ACCUM)?;
        let mut sq_accum = [0.0f32; MAX_CHANNELS];
        let sq_slice = f32_bytes_mut(&mut sq_accum[..gs]);
        anyhow::ensure!(
            sq_bytes >= sq_slice.len(),
            "{MODULE}::{SYM_SQ_ACCUM} too small: {sq_bytes} < {}",
            sq_slice.len()
        );
        // SAFETY: `sq_ptr` is the resolved `[MAX_CHANNELS] f32` device symbol
        // `d_innerq_sq_accum` with verified byte size >= `sq_slice.len()`, `gs <= MAX_CHANNELS`,
        // and `stream` is live.
        unsafe { reg.copy_d2h(sq_slice, sq_ptr, stream)? };

        // Identity-preserving equalization (mirrors turbo_innerq_finalize in
        // tq_plus_innerq.cu): scale[i] = (mean_rms / rms[i])^strength, clamped
        // to [0.5, 2.0]; auto-disable if max/min ratio < 1.2 either way.
        let count_f = count as f32;
        let mut rms = [0.0f32; MAX_CHANNELS];
        let mut mean_rms = 0.0f32;
        for i in 0..gs {
            rms[i] = (sq_accum[i] / count_f).sqrt();
            mean_rms += rms[i];
        }
        mean_rms /= gs as f32;

        let mut scale = [1.0f32; MAX_CHANNELS];
        let mut scale_inv = [1.0f32; MAX_CHANNELS];
        let mut max_ratio = 0.0f32;
        let mut min_ratio = 1e30f32;
        for i in 0..gs {
            let ratio = if rms[i] > 1e-10 {
                mean_rms / rms[i]
            } else {
                1.0
            };
            let s = ratio.powf(self.strength).clamp(0.5, 2.0);
            scale[i] = s;
            scale_inv[i] = 1.0 / s;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }

        let (calib_ptr, calib_bytes) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;
        let zero_i32: i32 = 0;
        let zero_i32_bytes = i32_bytes(&zero_i32);
        anyhow::ensure!(
            calib_bytes >= zero_i32_bytes.len(),
            "{MODULE}::{SYM_CALIBRATING} too small: {calib_bytes} < {}",
            zero_i32_bytes.len()
        );
        let calibration_off = (calib_ptr, zero_i32_bytes);

        if max_ratio < 1.2 && min_ratio > (1.0 / 1.2) {
            // SAFETY: `calib_ptr` is the resolved device `int` symbol
            // `d_innerq_calibrating` with verified byte size >= `zero_i32_bytes.len()`,
            // and `stream` is `reg.raw_stream()`.
            unsafe { reg.copy_h2d_group(&[calibration_off], stream)? };
            self.calibrating.store(false, Ordering::Release);
            self.finalized.store(true, Ordering::Release);
            tracing::info!(
                "InnerQ auto-disabled (channels already balanced: max_ratio={max_ratio:.3}, \
                 min_ratio={min_ratio:.3})"
            );
            return Ok(false);
        }

        let (scale_ptr, scale_bytes) = reg.device_symbol(MODULE, SYM_SCALE)?;
        let (scale_inv_ptr, scale_inv_bytes) = reg.device_symbol(MODULE, SYM_SCALE_INV)?;
        let (active_ptr, active_bytes) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let one_i32: i32 = 1;
        let scale_slice = f32_bytes(&scale[..gs]);
        let scale_inv_slice = f32_bytes(&scale_inv[..gs]);
        let one_i32_bytes = i32_bytes(&one_i32);
        anyhow::ensure!(
            scale_bytes >= scale_slice.len(),
            "{MODULE}::{SYM_SCALE} too small: {scale_bytes} < {}",
            scale_slice.len()
        );
        anyhow::ensure!(
            scale_inv_bytes >= scale_inv_slice.len(),
            "{MODULE}::{SYM_SCALE_INV} too small: {scale_inv_bytes} < {}",
            scale_inv_slice.len()
        );
        anyhow::ensure!(
            active_bytes >= one_i32_bytes.len(),
            "{MODULE}::{SYM_ACTIVE} too small: {active_bytes} < {}",
            one_i32_bytes.len()
        );
        // Ordered synchronous group: scales retire before active flips, so a
        // kernel on any stream cannot observe active=1 with stale scales.
        // SAFETY: All destination symbols are resolved from `tq_plus_innerq`
        // with verified byte sizes >= each host slice (`gs <= MAX_CHANNELS`),
        // and `stream` is live.
        unsafe {
            reg.copy_h2d_group(
                &[
                    calibration_off,
                    (scale_ptr, scale_slice),
                    (scale_inv_ptr, scale_inv_slice),
                    (active_ptr, one_i32_bytes),
                ],
                stream,
            )?;
        }

        self.calibrating.store(false, Ordering::Release);
        self.finalized.store(true, Ordering::Release);
        tracing::info!(
            "InnerQ scales activated (group_size={group_size}, max_ratio={max_ratio:.3}, \
             strength={:.2})",
            self.strength,
        );
        Ok(true)
    }
}
