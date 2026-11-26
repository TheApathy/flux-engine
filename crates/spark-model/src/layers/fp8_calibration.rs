// SPDX-License-Identifier: AGPL-3.0-only

//! Online FP8 KV cache scale calibration.
//!
//! Tracks running max |K| and max |V| values during the first N tokens of
//! inference to compute per-tensor scales: `scale = max / 448.0` (mapping the
//! observed dynamic range to FP8 E4M3 [-448, 448]).
//!
//! After the warmup period, initial scales are computed and subsequently
//! recalibrated every 128 tokens via an adaptive EMA to track distribution
//! shifts. During warmup, FP8 KV cache writes use scale=2.0 (uncalibrated,
//! ±896 effective FP8 E4M3 range) so models with large norm weights do not clip
//! before calibration completes.
//!
//! Thread safety: uses `parking_lot::Mutex` to serialize both calibration state
//! and access to the shared GPU `absmax_buf` scratch. The lock is uncontended
//! under single-threaded inference so lock overhead is negligible.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// FP8 E4M3 max representable magnitude.
const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to prevent division by zero or denormalized values.
const MIN_SCALE: f32 = 1e-12;

/// Mutable calibration state protected by Mutex for Send + Sync.
struct CalibrationInner {
    /// Running max of |K| values observed so far.
    k_running_max: f32,
    /// Running max of |V| values observed so far.
    v_running_max: f32,
    /// Total tokens processed (used for warmup and periodic 128-token recalibration).
    tokens_seen: usize,
    /// Whether initial warmup calibration is complete (periodic EMA recalibration active).
    frozen: bool,
    /// Active k_scale (`2.0` during warmup, calibrated thereafter).
    k_scale: f32,
    /// Active v_scale (`2.0` during warmup, calibrated thereafter).
    v_scale: f32,
}

/// Online FP8 KV cache scale calibration tracker for one attention layer.
///
/// Wraps calibration state in a Mutex so it can live inside a `Send + Sync`
/// struct (required by `TransformerLayer` trait).
pub struct Fp8KvCalibration {
    inner: Mutex<CalibrationInner>,
    /// Number of warmup tokens before initial scale calibration.
    warmup_tokens: usize,
    /// GPU buffer for absmax reduction output: `[1]` f32 for K, `[1]` f32 for V.
    /// Layout: `[k_absmax: f32, v_absmax: f32]` = 8 bytes.
    /// Access is serialized by `self.inner` mutex.
    absmax_buf: DevicePtr,
    /// Kernel handle for bf16_absmax reduction.
    absmax_kernel: KernelHandle,
}

// SAFETY: `DevicePtr` and `KernelHandle` are immutable plain GPU handles (`u64`)
// after construction. All mutable calibration state and every GPU read/write of
// `self.absmax_buf` (memset, kernel launches, synchronize, and D2H copy) are
// performed while holding `self.inner.lock()`, preventing concurrent races even
// if `&Fp8KvCalibration` is shared across threads.
unsafe impl Send for Fp8KvCalibration {}
unsafe impl Sync for Fp8KvCalibration {}

impl Fp8KvCalibration {
    /// Create a new calibration tracker.
    ///
    /// `warmup_tokens`: number of tokens to observe before freezing scales.
    /// `gpu`: GPU backend for allocating the absmax reduction buffer.
    pub fn new(warmup_tokens: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let absmax_kernel = gpu.kernel("reshape_and_cache", "bf16_absmax")?;
        // Allocate 8 bytes: [k_absmax: f32, v_absmax: f32]
        let absmax_buf = gpu.alloc(8)?;
        // Initialize to zero
        let zeros = [0u8; 8];
        gpu.copy_h2d(&zeros, absmax_buf)?;

        Ok(Self {
            inner: Mutex::new(CalibrationInner {
                k_running_max: 0.0,
                v_running_max: 0.0,
                tokens_seen: 0,
                frozen: false,
                // Start with scale=2.0 (effective range ±896) during warmup.
                // Models with large norm weights (Gemma-4 26B, Mistral) can produce
                // K/V values up to ~600, which exceeds FP8 E4M3 range at scale=1.0 (±448).
                // Scale=2.0 covers ±896 which is safe for all known models.
                k_scale: 2.0,
                v_scale: 2.0,
            }),
            warmup_tokens,
            absmax_buf,
            absmax_kernel,
        })
    }

    /// Whether calibration is still in warmup phase (scales not yet frozen).
    pub fn is_calibrating(&self) -> bool {
        let inner = self.inner.lock();
        !inner.frozen
    }

    /// Get current scales. Returns (k_scale, v_scale).
    ///
    /// During warmup: returns `(2.0, 2.0)` (uncalibrated, `±896` effective FP8 E4M3 range).
    /// After warmup: returns calibrated scales.
    pub fn scales(&self) -> (f32, f32) {
        let inner = self.inner.lock();
        (inner.k_scale, inner.v_scale)
    }

    /// Observe K/V projection outputs and update running max.
    ///
    /// Launches absmax reduction kernels on the K and V buffers, then reads
    /// the results back to CPU after a sync. Call this AFTER K/V projections
    /// and BEFORE writing to the KV cache.
    ///
    /// `k_data`: device BF16 buffer of K projection output
    /// `v_data`: device BF16 buffer of V projection output
    /// `num_tokens`: number of tokens in the batch
    /// `num_kv_heads`: number of KV heads
    /// `head_dim`: dimension per head
    pub fn observe(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
    ) -> Result<()> {
        if num_tokens == 0 {
            return Ok(());
        }

        // Hold `inner` across the entire GPU absmax reduction + sync + D2H copy
        // so concurrent `&self` callers cannot race on `self.absmax_buf`.
        // Observe continuously during warmup and recalibrate every 128 tokens
        // afterward to adapt to distribution shifts in multi-turn conversations
        // before FP8 E4M3 quantization buckets go stale.
        let mut inner = self.inner.lock();
        let prev_tokens = inner.tokens_seen;
        let next_tokens = prev_tokens.saturating_add(num_tokens as usize);
        let crossed_recalib = (next_tokens / 128) > (prev_tokens / 128);
        if inner.frozen && !crossed_recalib {
            inner.tokens_seen = next_tokens;
            return Ok(());
        }

        let n_elems = num_tokens * num_kv_heads * head_dim;

        // Reset absmax buffer to 0.0 before reduction (async to avoid sync/async conflict)
        gpu.memset_async(self.absmax_buf, 0, 8, stream)?;

        // Launch absmax for K
        let k_out = self.absmax_buf;
        super::ops::bf16_absmax(gpu, self.absmax_kernel, k_data, k_out, n_elems, stream)?;

        // Launch absmax for V (write to offset 4 = second f32)
        let v_out = self.absmax_buf.offset(4);
        super::ops::bf16_absmax(gpu, self.absmax_kernel, v_data, v_out, n_elems, stream)?;

        // Sync and read back
        gpu.synchronize(stream)?;
        let mut result_buf = [0u8; 8];
        gpu.copy_d2h(self.absmax_buf, &mut result_buf)?;
        let k_max =
            f32::from_le_bytes([result_buf[0], result_buf[1], result_buf[2], result_buf[3]]);
        let v_max =
            f32::from_le_bytes([result_buf[4], result_buf[5], result_buf[6], result_buf[7]]);

        // Update running max and check if warmup is complete or periodic recalibration is due
        inner.k_running_max = inner.k_running_max.max(k_max);
        inner.v_running_max = inner.v_running_max.max(v_max);
        inner.tokens_seen = next_tokens;

        if !inner.frozen && inner.tokens_seen >= self.warmup_tokens {
            // Initial calibration: compute scales from warmup observations.
            inner.k_scale = (inner.k_running_max / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.v_scale = (inner.v_running_max / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.frozen = true;
        } else if inner.frozen && crossed_recalib {
            // Periodic recalibration: blend new observations with existing scales.
            // Adaptive EMA: use α=0.3 when distribution shifts >20%, else α=0.1.
            // This responds faster to multi-turn topic switches without oscillating
            // on stable distributions.
            let new_k = (k_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let new_v = (v_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let k_shift = (new_k - inner.k_scale).abs() / inner.k_scale.max(MIN_SCALE);
            let v_shift = (new_v - inner.v_scale).abs() / inner.v_scale.max(MIN_SCALE);
            let alpha = if k_shift > 0.2 || v_shift > 0.2 {
                0.3
            } else {
                0.1
            };
            inner.k_scale = (1.0 - alpha) * inner.k_scale + alpha * new_k;
            inner.v_scale = (1.0 - alpha) * inner.v_scale + alpha * new_v;
            // Reset running max for next observation window
            inner.k_running_max = k_max;
            inner.v_running_max = v_max;

            tracing::info!(
                "FP8 KV calibrated after {} tokens: k_scale={:.6} (max={:.2}), v_scale={:.6} (max={:.2})",
                inner.tokens_seen,
                inner.k_scale,
                inner.k_running_max,
                inner.v_scale,
                inner.v_running_max,
            );
        }

        Ok(())
    }
}
