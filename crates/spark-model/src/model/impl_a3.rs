// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Verify-side LM-head vocab truncation cap (`ATLAS_TARGET_LMHEAD_VOCAB`).
///
/// The TARGET model's verify `lm_head` GEMV computes logits over the FULL
/// 248320-row vocab for every spec-decode verify step (k≈2 rows/step). The
/// argmax that picks the verified token only needs the top scoring token,
/// and BPE places frequent tokens at low IDs — so reading only the first N
/// weight rows makes the GEMV proportionally cheaper (it's memory-bound on
/// the NVFP4 weight read at 273 GB/s) with negligible quality risk for N
/// large enough to cover normal text.
///
/// 0 (default) = full vocab (no truncation). Any value ≥ vocab_size is also
/// treated as full. Only the batched-verify / multi-row LM-head paths
/// (`lm_head_batched`, `lm_head_qwen4_rows`, `lm_head_qwen4_batched_verify_rows`)
/// honor this — single-token decode and prefill keep the full vocab.
pub(super) fn target_lmhead_vocab() -> u32 {
    static CACHE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_TARGET_LMHEAD_VOCAB")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0)
    })
}

pub(crate) const QWEN4_TARGET_LMHEAD_MODE_ENV: &str = "ATLAS_QWEN4_TARGET_LMHEAD_MODE";
pub(crate) const FLUX_QWEN4_TARGET_LMHEAD_MODE_ENV: &str = "FLUX_QWEN4_TARGET_LMHEAD_MODE";
pub(crate) const QWEN4_TARGET_LMHEAD_BF16_ENV: &str = "ATLAS_QWEN4_TARGET_LMHEAD_BF16";
pub(crate) const FLUX_QWEN4_TARGET_LMHEAD_BF16_ENV: &str = "FLUX_QWEN4_TARGET_LMHEAD_BF16";

/// Target-side `lm_head` precision mode for `qwen4_exp`.
///
/// - `Fp8` (default): per-row scaled FP8 E4M3 (`635 MB`, `0.5x` BF16 bandwidth,
///   3 mantissa + 4 exponent bits) via `dense_gemv_fp8w` (`M == 1`) and
///   `fp8_gemm_coalesced_m16/m32/m64/m128` (`M >= 2`). Eliminates the 4-bit
///   NVFP4 `lm_head` reasoning loops without BF16's bandwidth regression.
/// - `Bf16`: exact checkpoint BF16 `lm_head.weight` (`1.27 GB`).
/// - `Nvfp4`: 4-bit NVFP4 `lm_head` (fastest bandwidth, lowest precision).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4TargetLmHeadMode {
    Fp8,
    Bf16,
    Nvfp4,
}

pub(crate) fn parse_qwen4_target_lmhead_mode(
    atlas_mode: Option<&str>,
    flux_mode: Option<&str>,
    atlas_bf16: Option<&str>,
    flux_bf16: Option<&str>,
) -> Qwen4TargetLmHeadMode {
    for raw in [atlas_mode, flux_mode].into_iter().flatten() {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fp8" | "e4m3" | "fp8_e4m3" => return Qwen4TargetLmHeadMode::Fp8,
            "bf16" | "fp16" | "exact" => return Qwen4TargetLmHeadMode::Bf16,
            "nvfp4" | "fp4" | "w4a16" => return Qwen4TargetLmHeadMode::Nvfp4,
            _ => {}
        }
    }
    for raw in [atlas_bf16, flux_bf16].into_iter().flatten() {
        match raw.trim() {
            "1" | "true" | "TRUE" | "on" | "ON" => return Qwen4TargetLmHeadMode::Bf16,
            "0" | "false" | "FALSE" | "off" | "OFF" => return Qwen4TargetLmHeadMode::Nvfp4,
            _ => {}
        }
    }
    Qwen4TargetLmHeadMode::Fp8
}

/// Active target-side `lm_head` precision mode for `qwen4_exp` (default `Fp8`).
///
/// Honors `ATLAS_QWEN4_TARGET_LMHEAD_MODE` / `FLUX_QWEN4_TARGET_LMHEAD_MODE`
/// (`fp8` | `bf16` | `nvfp4`) and legacy `ATLAS_QWEN4_TARGET_LMHEAD_BF16` /
/// `FLUX_QWEN4_TARGET_LMHEAD_BF16` (`1` -> `Bf16`, `0` -> `Nvfp4`).
pub(crate) fn qwen4_target_lmhead_mode() -> Qwen4TargetLmHeadMode {
    let read_live = || {
        let atlas_mode = std::env::var(QWEN4_TARGET_LMHEAD_MODE_ENV).ok();
        let flux_mode = std::env::var(FLUX_QWEN4_TARGET_LMHEAD_MODE_ENV).ok();
        let atlas_bf16 = std::env::var(QWEN4_TARGET_LMHEAD_BF16_ENV).ok();
        let flux_bf16 = std::env::var(FLUX_QWEN4_TARGET_LMHEAD_BF16_ENV).ok();
        parse_qwen4_target_lmhead_mode(
            atlas_mode.as_deref(),
            flux_mode.as_deref(),
            atlas_bf16.as_deref(),
            flux_bf16.as_deref(),
        )
    };
    #[cfg(not(test))]
    {
        static MODE: std::sync::OnceLock<Qwen4TargetLmHeadMode> = std::sync::OnceLock::new();
        *MODE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// True only when target logits are pinned specifically to checkpoint BF16
/// `lm_head.weight` (used by `factory/build.rs` to retain BF16 on `Plain`/
/// `NativeMtp` while releasing it under `Fp8` and `Nvfp4`).
pub(crate) fn qwen4_target_lmhead_bf16_enabled() -> bool {
    qwen4_target_lmhead_mode() == Qwen4TargetLmHeadMode::Bf16
}

fn log_qwen4_target_fp8_lm_head_engagement(
    route_bit: u8,
    route: &'static str,
    rows: u32,
    vocab: u32,
    hidden: u32,
) {
    static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let mask = 1u8 << route_bit;
    if SEEN.fetch_or(mask, std::sync::atomic::Ordering::Relaxed) & mask == 0 {
        tracing::info!(
            target: "atlas::lm_head",
            selector = QWEN4_TARGET_LMHEAD_MODE_ENV,
            route,
            rows,
            vocab,
            hidden,
            weight_dtype = "fp8_e4m3_rowscale",
            output_dtype = "bf16",
            output_layout = "row_major",
            "QWEN4_TARGET_LMHEAD_FP8_ENGAGED"
        );
    }
}

fn log_qwen4_target_bf16_lm_head_engagement(
    route_bit: u8,
    route: &'static str,
    rows: u32,
    vocab: u32,
    hidden: u32,
) {
    static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let mask = 1u8 << route_bit;
    if SEEN.fetch_or(mask, std::sync::atomic::Ordering::Relaxed) & mask == 0 {
        tracing::info!(
            target: "atlas::lm_head",
            selector = QWEN4_TARGET_LMHEAD_BF16_ENV,
            route,
            rows,
            vocab,
            hidden,
            output_dtype = "bf16",
            output_layout = "row_major",
            "QWEN4_TARGET_LMHEAD_BF16_ENGAGED"
        );
    }
}

/// `ATLAS_LM_HEAD_TC=1` routes the target verify lm_head (M=2..=32) through
/// the transposed-NVFP4 m32_n64 tensor-core kernel (`w4a16_gemm_t_m32_n64`)
/// instead of the byte-exact scalar-FMA family. This is the same kernel the
/// DFlash drafter already uses for its propose head (`draft_lm_head_nvfp4_t`),
/// so the weight layout and MMA rounding are proven coherent; the target
/// committed token may differ from the scalar oracle by MMA-vs-FMA rounding
/// (a re-reference in the same class as `ATLAS_FFN_TC=1`). Default OFF: the
/// exact scalar path remains the commit authority until this is qualified.
#[inline]
#[must_use]
pub(crate) const fn lm_head_tc_enabled_with_policy(
    raw_enabled: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    !det_batch.is_enabled() && raw_enabled
}

pub(super) fn lm_head_tc_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let raw = *CACHE.get_or_init(|| std::env::var("ATLAS_LM_HEAD_TC").ok().as_deref() == Some("1"));
    lm_head_tc_enabled_with_policy(
        raw,
        crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
    )
}

fn log_lm_head_tc_engagement(rows: u32, vocab: u32, hidden: u32) {
    static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            target: "atlas::lm_head",
            route = "tc_m32_n64",
            rows,
            vocab,
            hidden,
            output_dtype = "bf16",
            output_layout = "row_major",
            "LM_HEAD_TC_ENGAGEMENT"
        );
    }
}

fn log_exact_lm_head_engagement(route: ops::ExactLmHeadRoute, rows: u32, vocab: u32, hidden: u32) {
    static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    let tier_bit = match route.tier() {
        ops::ExactLmHeadTier::M4 => 0,
        ops::ExactLmHeadTier::M8 => 1,
        ops::ExactLmHeadTier::M17 => 2,
        ops::ExactLmHeadTier::M32 => 3,
    };
    let route_bit = match route {
        ops::ExactLmHeadRoute::Exact(_) => tier_bit,
        ops::ExactLmHeadRoute::SerialK1(_) => tier_bit + 4,
    };
    let mask = 1u8 << route_bit;
    if SEEN.fetch_or(mask, std::sync::atomic::Ordering::Relaxed) & mask == 0 {
        tracing::info!(
            target: "atlas::lm_head",
            route = route.provenance(),
            tier = route.tier().label(),
            rows,
            vocab,
            hidden,
            output_dtype = "bf16",
            output_layout = "row_major",
            "LM_HEAD_EXACT_ENGAGEMENT"
        );
    }
}

fn cached_kernel_result(res: &std::result::Result<KernelHandle, String>) -> Result<KernelHandle> {
    match res {
        Ok(handle) => Ok(*handle),
        Err(msg) => bail!("{msg}"),
    }
}

/// Per-backend kernel-handle cache with a lock-free `OnceLock` fast-path for
/// the primary production backend and a backend-pointer-keyed fallback map so
/// distinct mock backends in unit tests never poison each other.
struct BackendKernelCache {
    primary: std::sync::OnceLock<(usize, std::result::Result<KernelHandle, String>)>,
    fallback: std::sync::OnceLock<Mutex<HashMap<usize, std::result::Result<KernelHandle, String>>>>,
}

impl BackendKernelCache {
    const fn new() -> Self {
        Self {
            primary: std::sync::OnceLock::new(),
            fallback: std::sync::OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend, module: &str, name: &str) -> Result<KernelHandle> {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some((cached_key, cached_res)) = self.primary.get() {
            if *cached_key == key {
                return cached_kernel_result(cached_res);
            }
        } else {
            let mut probed = Some(gpu.kernel(module, name).map_err(|err| format!("{err:#}")));
            let (stored_key, stored_res) = self.primary.get_or_init(|| {
                (
                    key,
                    probed
                        .take()
                        .expect("kernel probe result initialized once before get_or_init"),
                )
            });
            if *stored_key == key {
                return cached_kernel_result(stored_res);
            }
            let val = probed
                .unwrap_or_else(|| gpu.kernel(module, name).map_err(|err| format!("{err:#}")));
            let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
            let mut guard = map.lock();
            let entry = guard.entry(key).or_insert(val);
            return cached_kernel_result(entry);
        }
        let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
        let cached = map.lock().get(&key).cloned();
        if let Some(res) = cached {
            return cached_kernel_result(&res);
        }
        let probed = gpu.kernel(module, name).map_err(|err| format!("{err:#}"));
        let mut guard = map.lock();
        let entry = guard.entry(key).or_insert(probed);
        cached_kernel_result(entry)
    }

    fn get_or_try_load(&self, gpu: &dyn GpuBackend, module: &str, name: &str) -> KernelHandle {
        match self.get_or_load(gpu, module, name) {
            Ok(handle) => handle,
            Err(_) => KernelHandle(0),
        }
    }
}

static QWEN4_BATCHED_EMBED_EXPAND_KERNEL: BackendKernelCache = BackendKernelCache::new();
static BATCHED_EMBED_KERNEL: BackendKernelCache = BackendKernelCache::new();
static QWEN4_HC_EXPAND_EMBEDDING_KERNEL: BackendKernelCache = BackendKernelCache::new();
static W4A16_GEMM_COALESCED_V2_N64_M16_KERNEL: BackendKernelCache = BackendKernelCache::new();
static W4A16_GEMM_COALESCED_V2_N64_M16_FALLBACK_KERNEL: BackendKernelCache =
    BackendKernelCache::new();
static W4A16_GEMM_COALESCED_M32_KERNEL: BackendKernelCache = BackendKernelCache::new();
static W4A16_GEMM_COALESCED_M64_KERNEL: BackendKernelCache = BackendKernelCache::new();
static W4A16_GEMM_COALESCED_M128_KERNEL: BackendKernelCache = BackendKernelCache::new();
static LM_HEAD_NVFP4_ARGMAX_PARTIAL_KERNEL: BackendKernelCache = BackendKernelCache::new();
static LM_HEAD_ARGMAX_REDUCE_KERNEL: BackendKernelCache = BackendKernelCache::new();

/// SM count threshold above which NVFP4 `lm_head` dispatches v2 coalesced
/// 2-stage K-prefetch kernels (`v2_n64_m80`/`v2_n64_m128` for `M > 64` verify,
/// and `v2_n64_m16` for `M <= 16` when loaded in the bundle) while keeping
/// GB10 (`<= 64` SMs) untouched.
const HIGH_SM_LM_HEAD_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
const fn should_use_v2_coalesced_verify_lm_head(
    has_tc_transposed: bool,
    num_tokens: u32,
    sm_count: u32,
) -> bool {
    !has_tc_transposed && num_tokens > 64 && sm_count > HIGH_SM_LM_HEAD_THRESHOLD
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LmHeadTcEnvOverride {
    Disabled,
    Enabled,
    Auto,
}

fn parse_lm_head_tc_env_override(raw: Option<&str>) -> LmHeadTcEnvOverride {
    match raw.map(str::trim) {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
            LmHeadTcEnvOverride::Disabled
        }
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => {
            LmHeadTcEnvOverride::Enabled
        }
        _ => LmHeadTcEnvOverride::Auto,
    }
}

fn raw_lm_head_tc_env_override() -> LmHeadTcEnvOverride {
    static CACHE: std::sync::OnceLock<LmHeadTcEnvOverride> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        parse_lm_head_tc_env_override(std::env::var("ATLAS_LM_HEAD_TC").ok().as_deref())
    })
}

/// Whether `lm_head_batched` auto-dispatches SM120/SM103 Tensor Core paths
/// (coalesced small-M for `2..=16`, native cuBLASLt NVFP4 for `24..=128`) even
/// when `ATLAS_LM_HEAD_TC` is unset. Preserves GB10 (`<= 64` SMs) and Qwen4
/// (`is_qwen4_exp`) unchanged, and honors explicit `ATLAS_LM_HEAD_TC=0`.
/// Policy-aware auto-TC batched LM-head eligibility: `ATLAS_DETERMINISTIC_BATCH=1`
/// keeps `M=1..=128` on the 0-ULP exact GEMV family instead of flipping `M>=2`
/// onto SM120 coalesced / cuBLASLt NVFP4 Tensor Cores.
#[inline]
#[must_use]
pub(crate) const fn should_auto_tc_batched_lm_head_for_policy(
    is_qwen4_exp: bool,
    env_disabled: bool,
    sm_count: u32,
    num_tokens: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    !is_qwen4_exp && num_tokens >= 2 && sm_count > HIGH_SM_LM_HEAD_THRESHOLD && !env_disabled
}

#[cfg(test)]
#[inline]
#[must_use]
const fn should_auto_tc_batched_lm_head(
    is_qwen4_exp: bool,
    env_override: LmHeadTcEnvOverride,
    sm_count: u32,
    num_tokens: u32,
) -> bool {
    should_auto_tc_batched_lm_head_for_policy(
        is_qwen4_exp,
        matches!(env_override, LmHeadTcEnvOverride::Disabled),
        sm_count,
        num_tokens,
        crate::layers::deterministic_batch::DeterministicBatchPolicy::Disabled,
    )
}

#[inline]
#[must_use]
const fn should_prefer_m16_coalesced_lm_head(num_tokens: u32, sm_count: u32) -> bool {
    num_tokens >= 2 && num_tokens <= 16 && sm_count > HIGH_SM_LM_HEAD_THRESHOLD
}
static W4A16_GEMM_PIPE_M128N128_KERNEL: BackendKernelCache = BackendKernelCache::new();
static W4A16_GEMM_PIPE_KERNEL: BackendKernelCache = BackendKernelCache::new();
static DENSE_GEMV_BF16_BATCHN_KERNEL: BackendKernelCache = BackendKernelCache::new();
static BF16_GEMM_COALESCED_M16_KERNEL: BackendKernelCache = BackendKernelCache::new();
static BF16_GEMM_COALESCED_M32_KERNEL: BackendKernelCache = BackendKernelCache::new();
static BF16_GEMM_COALESCED_M64_KERNEL: BackendKernelCache = BackendKernelCache::new();
static BF16_GEMM_COALESCED_M128_KERNEL: BackendKernelCache = BackendKernelCache::new();
static DENSE_GEMV_FP8W_KERNEL: BackendKernelCache = BackendKernelCache::new();
static FP8_GEMM_COALESCED_M16_KERNEL: BackendKernelCache = BackendKernelCache::new();
static FP8_GEMM_COALESCED_M32_KERNEL: BackendKernelCache = BackendKernelCache::new();
static FP8_GEMM_COALESCED_M64_KERNEL: BackendKernelCache = BackendKernelCache::new();
static FP8_GEMM_COALESCED_M128_KERNEL: BackendKernelCache = BackendKernelCache::new();

impl TransformerModel {
    /// True when stream-ordered pinned H2D staging (`copy_h2d_pinned_async`) is
    /// enabled for hot model H2D uploads, bypassing `bounded_copy::h2d_group`'s
    /// per-slice stream drains and event spin-wait loops under
    /// `ATLAS_BOUNDED_HOST_COPY=1`.
    #[inline]
    pub(in crate::model) fn fast_host_h2d_enabled() -> bool {
        static VERIFY_FAST_HOST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let verify_fast = *VERIFY_FAST_HOST.get_or_init(|| {
            std::env::var("ATLAS_QWEN4_VERIFY_FAST_HOST")
                .ok()
                .as_deref()
                == Some("1")
        });
        verify_fast
            || crate::model::host_bucket::armed(crate::model::host_bucket::Lever::LeanFinalize)
                .unwrap_or(false)
    }

    /// Record `stream` as having an active async DMA in the current fast-H2D
    /// pinned ring epoch. When `wrapped` is true (ring cursor wrapped back to
    /// `h2d_ring_base`), returns every stream from the previous epoch so the
    /// caller can drain them before overwriting pinned host memory.
    fn note_h2d_ring_epoch(&self, stream: u64, first_use: bool, wrapped: bool) -> Vec<u64> {
        static EPOCH_STREAMS: std::sync::OnceLock<Mutex<HashMap<usize, Vec<u64>>>> =
            std::sync::OnceLock::new();
        let map = EPOCH_STREAMS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock();
        let streams = guard.entry(self.instance_key()).or_default();
        if first_use {
            streams.clear();
        }
        let drain = if wrapped {
            std::mem::take(streams)
        } else {
            Vec::new()
        };
        if !streams.contains(&stream) {
            streams.push(stream);
        }
        drain
    }

    /// Carve an 8-byte-aligned slice of `bytes` from the dedicated fast-H2D
    /// ring tail `[h2d_ring_base..buffer.len())` of `self.pinned_staging`.
    /// Returns `None` if `bytes` exceeds the ring capacity.
    #[inline]
    pub(in crate::model) fn reserve_pinned_h2d_range(&self, bytes: usize) -> Option<usize> {
        let mut stg = self.pinned_staging.lock();
        let first_use = stg.h2d_ring_cursor == stg.h2d_ring_base;
        let prev_aligned = (stg.h2d_ring_cursor + 7) & !7;
        let start = stg.reserve_h2d_range(bytes)?;
        let wrapped = !first_use && start < prev_aligned;
        let drain = self.note_h2d_ring_epoch(self.gpu.default_stream(), first_use, wrapped);
        for s in drain {
            let _ = self.gpu.synchronize(s);
        }
        Some(start)
    }

    /// Stage `copies` into disjoint 8-byte-aligned sub-slices of the fast-H2D
    /// pinned ring and enqueue stream-ordered `copy_h2d_pinned_async` transfers
    /// without per-call stream synchronizations when `fast_host_h2d_enabled()`
    /// is active. When the ring cursor wraps back to `h2d_ring_base`, all
    /// streams that enqueued DMA during the previous ring epoch are
    /// synchronized before any pinned bytes are overwritten. Falls back to
    /// `self.gpu.copy_h2d_group_on_stream(copies, stream)` when disabled or if
    /// a payload exceeds the ring capacity.
    pub(in crate::model) fn copy_h2d_group_fast(
        &self,
        copies: &[spark_runtime::gpu::HostToDeviceCopy<'_>],
        stream: u64,
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        if !Self::fast_host_h2d_enabled() {
            return self.gpu.copy_h2d_group_on_stream(copies, stream);
        }
        let mut total_aligned = 0usize;
        for copy in copies {
            let aligned = copy.src().len().checked_add(7).map(|n| n & !7);
            match aligned.and_then(|a| total_aligned.checked_add(a)) {
                Some(sum) => total_aligned = sum,
                None => return self.gpu.copy_h2d_group_on_stream(copies, stream),
            }
        }
        if total_aligned == 0 {
            return Ok(());
        }
        let mut stg = self.pinned_staging.lock();
        let first_use = stg.h2d_ring_cursor == stg.h2d_ring_base;
        let prev_aligned = (stg.h2d_ring_cursor + 7) & !7;
        let Some(mut cursor) = stg.reserve_h2d_range(total_aligned) else {
            drop(stg);
            return self.gpu.copy_h2d_group_on_stream(copies, stream);
        };
        let wrapped = !first_use && cursor < prev_aligned;
        let drain_streams = self.note_h2d_ring_epoch(stream, first_use, wrapped);
        for drain_stream in drain_streams {
            self.gpu.synchronize(drain_stream)?;
        }
        for copy in copies {
            let src = copy.src();
            if src.is_empty() {
                continue;
            }
            let len = src.len();
            // SAFETY: `reserve_h2d_range` validated `[cursor..cursor+total_aligned)`
            // within `stg.buffer`, `stg` is exclusively locked, and if the ring
            // wrapped, every stream from the previous ring epoch was drained
            // via `gpu.synchronize` above so no in-flight DMA reads this slice.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    stg.buffer.as_mut_ptr().add(cursor),
                    len,
                );
            }
            let pinned_slice = stg.buffer.pinned_slice_range(cursor, len)?;
            // SAFETY: `stg.buffer` is page-locked with model lifetime, this
            // ring slice is disjoint from all earlier uploads in the current
            // epoch, and any prior epoch was synchronized on wrap before reuse.
            unsafe {
                self.gpu
                    .copy_h2d_pinned_async(pinned_slice, copy.dst(), stream)?;
            }
            cursor = (cursor + len + 7) & !7;
        }
        Ok(())
    }

    /// Collapse Qwen4's four persistent streams to the core hidden width.
    /// Conventional architectures retain their terminal RMSNorm path at the
    /// call site.
    pub(super) fn qwen4_final_hidden(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        self.qwen4_final_hidden_with_bf16(hidden, residual, stream, true)
    }

    /// Same as [`Self::qwen4_final_hidden`], but never takes the
    /// `ATLAS_QWEN4_HC_BF16` arm. The prefill-last-row and speculative
    /// verify-row callers below reuse this single-token entry point on a
    /// row that is not genuine single-token decode, so they call this
    /// directly instead.
    pub(super) fn qwen4_final_hidden_nvfp4(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        self.qwen4_final_hidden_with_bf16(hidden, residual, stream, false)
    }

    fn qwen4_final_hidden_with_bf16(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        stream: u64,
        allow_bf16: bool,
    ) -> Result<Option<DevicePtr>> {
        let Some(mixer) = &self.qwen4_final_mixer else {
            return Ok(None);
        };
        let (mixed, inject) = if allow_bf16 {
            mixer.prepare_decode(
                hidden,
                residual,
                &self.buffers,
                self.gpu.as_ref(),
                self.config.rms_norm_eps as f32,
                stream,
            )?
        } else {
            mixer.prepare_decode_nvfp4(
                hidden,
                residual,
                &self.buffers,
                self.gpu.as_ref(),
                self.config.rms_norm_eps as f32,
                stream,
            )?
        };
        debug_assert!(inject.is_none());
        Ok(Some(mixed))
    }

    pub(super) fn embed(&self, token: u32, output: DevicePtr, stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let row_bytes = h * 2; // BF16 embedding row
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        if self.config.is_qwen4_exp() {
            let scratch = self.buffers.norm_output();
            self.gpu.copy_d2d_async(src, scratch, row_bytes, stream)?;
            self.expand_qwen4_embedding(scratch, output, stream)?;
            return Ok(());
        }
        if self.bf16_to_f32_kernel.0 != 0 {
            // FP32 residual: embed BF16 to scratch, convert to FP32 output.
            // The scratch buffer is norm_output which is BF16 regardless of
            // residual dtype — use the BF16 scaler explicitly.
            let scratch = self.buffers.norm_output();
            self.gpu.copy_d2d_async(src, scratch, row_bytes, stream)?;
            self.scale_embeddings_bf16(scratch, 1, stream)?;
            crate::layers::ops::bf16_to_f32(
                self.gpu.as_ref(),
                self.bf16_to_f32_kernel,
                scratch,
                output,
                h as u32,
                stream,
            )
        } else {
            self.gpu.copy_d2d_async(src, output, row_bytes, stream)?;
            // Scale embeddings (Gemma-4: sqrt(hidden_size))
            self.scale_embeddings(output, 1, stream)
        }
    }

    /// `ATLAS_QWEN4_PREFILL_EMBED_BATCH=1` (or
    /// `ATLAS_QWEN4_PREFILL_EMBED_EXPAND_FUSED=1`): embed a whole prefill chunk
    /// instead of a D2D copy + expand launch per token. Both paths are pure
    /// copies of the same BF16 rows.
    pub(super) fn qwen4_embed_batch_selected() -> bool {
        static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SEL.get_or_init(|| {
            std::env::var("ATLAS_QWEN4_PREFILL_EMBED_BATCH")
                .ok()
                .as_deref()
                == Some("1")
                || std::env::var("ATLAS_QWEN4_PREFILL_EMBED_EXPAND_FUSED")
                    .ok()
                    .as_deref()
                    == Some("1")
        })
    }

    /// `ATLAS_QWEN4_PREFILL_EMBED_EXPAND_FUSED`: default on when
    /// `embed_qwen4_batch` runs (`"0"` opts back into the 2-kernel
    /// `batched_embed` + `qwen4_hc_expand_embedding` path).
    pub(super) fn qwen4_embed_expand_fused_selected() -> bool {
        static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SEL.get_or_init(|| {
            !matches!(
                std::env::var("ATLAS_QWEN4_PREFILL_EMBED_EXPAND_FUSED")
                    .ok()
                    .as_deref(),
                Some("0")
            )
        })
    }

    /// Resolve the exact batched-embedding route without uploading token IDs or
    /// launching work. The native model frame calls this before KV growth so a
    /// missing lazily loaded symbol cannot fail after request-owned mutation.
    pub(super) fn preflight_embed_qwen4_batch(&self, num_tokens: usize) -> Result<()> {
        anyhow::ensure!(
            num_tokens > 0,
            "Qwen4 batched embed needs at least one token"
        );
        let fused = Self::qwen4_embed_expand_fused_selected();
        anyhow::ensure!(
            self.buffers.sizes().scratch >= num_tokens * 4
                && (fused
                    || self.buffers.sizes().norm_output
                        >= num_tokens * self.config.hidden_size * 2),
            "Qwen4 batched embed scratch is undersized for {num_tokens} tokens"
        );
        if fused {
            QWEN4_BATCHED_EMBED_EXPAND_KERNEL.get_or_load(
                self.gpu.as_ref(),
                "qwen4_hyper",
                "qwen4_batched_embed_expand",
            )?;
        } else {
            BATCHED_EMBED_KERNEL.get_or_load(
                self.gpu.as_ref(),
                "embed_from_argmax",
                "batched_embed",
            )?;
            QWEN4_HC_EXPAND_EMBEDDING_KERNEL.get_or_load(
                self.gpu.as_ref(),
                "qwen4_hyper",
                "qwen4_hc_expand_embedding",
            )?;
        }
        Ok(())
    }

    /// Batched twin of calling [`Self::embed`] for each of `tokens` into
    /// consecutive `[hc_count, hidden_size]` rows of `output`.
    pub(super) fn embed_qwen4_batch(
        &self,
        tokens: &[u32],
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let n = tokens.len();
        let h = self.config.hidden_size;
        let fused = Self::qwen4_embed_expand_fused_selected();
        self.preflight_embed_qwen4_batch(n)?;
        // SAFETY: `tokens` is a live initialized `&[u32]` (`size_of_val(tokens)`
        // bytes, POD); `u8` has alignment 1 and `copy_h2d_group_fast` finishes
        // reading `token_ids_bytes` before returning.
        let token_ids_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(tokens.as_ptr().cast::<u8>(), std::mem::size_of_val(tokens))
        };
        let token_ids_dev = self.buffers.scratch();
        self.copy_h2d_group_fast(
            &[spark_runtime::gpu::HostToDeviceCopy::new(
                token_ids_bytes,
                token_ids_dev,
            )],
            stream,
        )?;
        if fused {
            let kernel = QWEN4_BATCHED_EMBED_EXPAND_KERNEL.get_or_load(
                self.gpu.as_ref(),
                "qwen4_hyper",
                "qwen4_batched_embed_expand",
            )?;
            return KernelLaunch::new(self.gpu.as_ref(), kernel)
                .grid([n as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(token_ids_dev)
                .arg_ptr(self.embed_tokens.weight)
                .arg_ptr(output)
                .arg_u32(h as u32)
                .arg_u32(self.config.hc_count as u32)
                .launch(stream);
        }
        let staged = self.buffers.norm_output();
        let gather = BATCHED_EMBED_KERNEL.get_or_load(
            self.gpu.as_ref(),
            "embed_from_argmax",
            "batched_embed",
        )?;
        crate::layers::ops::batched_embed(
            self.gpu.as_ref(),
            gather,
            token_ids_dev,
            self.embed_tokens.weight,
            staged,
            n as u32,
            h as u32,
            stream,
        )?;
        let kernel = QWEN4_HC_EXPAND_EMBEDDING_KERNEL.get_or_load(
            self.gpu.as_ref(),
            "qwen4_hyper",
            "qwen4_hc_expand_embedding",
        )?;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([
                n as u32,
                div_ceil(self.config.residual_width() as u32, 256),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(staged)
            .arg_ptr(output)
            .arg_u32(h as u32)
            .arg_u32(self.config.hc_count as u32)
            .launch(stream)
    }

    /// Expand one BF16 `[hidden_size]` embedding into Qwen4's
    /// `[hc_count, hidden_size]` residual-stream layout.
    pub(super) fn expand_qwen4_embedding(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = QWEN4_HC_EXPAND_EMBEDDING_KERNEL.get_or_load(
            self.gpu.as_ref(),
            "qwen4_hyper",
            "qwen4_hc_expand_embedding",
        )?;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([1, div_ceil(self.config.residual_width() as u32, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(output)
            .arg_u32(self.config.hidden_size as u32)
            .arg_u32(self.config.hc_count as u32)
            .launch(stream)
    }

    /// Scale in-place embeddings by config.embed_scale. Picks the kernel
    /// matching `data`'s actual dtype:
    ///   - when `use_fp32_residual()` is true, `hidden` is FP32 and we
    ///     dispatch `embed_scale::f32_scale_inplace`
    ///   - otherwise (`hidden` is BF16) we dispatch the usual
    ///     `embed_scale::bf16_scale_inplace`
    ///
    /// For the rare case of scaling a BF16 buffer while FP32 residual is
    /// ALSO active (e.g. the decode embed() scratch which is deliberately
    /// BF16 before a bf16_to_f32 cast), use `scale_embeddings_bf16`.
    pub(super) fn scale_embeddings(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.config.use_fp32_residual() {
            self.scale_embeddings_fp32(data, num_tokens, stream)
        } else {
            self.scale_embeddings_bf16(data, num_tokens, stream)
        }
    }

    pub(super) fn scale_embeddings_bf16(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), self.embed_scale_kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    pub(super) fn scale_embeddings_fp32(
        &self,
        data: DevicePtr,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        // Symmetric with scale_embeddings_bf16: models without embedding
        // scaling (non-Gemma, e.g. qwen3.6-27b) have no embed_scale kernel
        // registered (handle == 0). Without this guard the FP8 fp32-residual
        // path hard-fails ("Module 'embed_scale' not loaded").
        if self.embed_scale_kernel.0 == 0 {
            return Ok(());
        }
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = self.gpu.kernel("embed_scale", "f32_scale_inplace")?;
        let n = (num_tokens * self.config.hidden_size) as u32;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(n)
            .arg_f32(self.config.embed_scale)
            .launch(stream)
    }

    /// LM head for K tokens: hidden[K, H] → logits[K, V].
    /// Effective verify-side vocab for the batched-decode/verify `lm_head`
    /// argmax: the full vocab, or the `ATLAS_TARGET_LMHEAD_VOCAB` cap when it
    /// is set and smaller. The scheduler's verify argmax MUST read this (not
    /// `config.vocab_size`) so the per-row logits stride and the argmax range
    /// match what `lm_head_batched` actually wrote. Returns the full vocab in
    /// every non-truncated case (cap==0 or cap≥vocab).
    pub(super) fn verify_lmhead_vocab(&self) -> u32 {
        let v = self.config.vocab_size as u32;
        let cap = target_lmhead_vocab();
        if cap == 0 || cap >= v { v } else { cap }
    }

    /// Attempt zero-VRAM, CUDA-graph-safe coalesced NVFP4 LM-head dispatch for
    /// scoped batched-verify frames (`rows >= 4` by default). Returns `Ok(false)` when
    /// disabled (`ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4=0`) or when shape,
    /// alignment, or kernel prerequisites are absent so [`Self::lm_head_batched`]
    /// falls back cleanly to exact scalar GEMV.
    #[allow(clippy::too_many_arguments)]
    fn try_lm_head_coalesced_nvfp4(
        &self,
        hidden: DevicePtr,
        nvfp4: &QuantizedWeight,
        logits: DevicePtr,
        num_tokens: u32,
        v: u32,
        h: u32,
        stream: u64,
    ) -> Result<bool> {
        if !crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled() {
            return Ok(false);
        }
        let dispatched = crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
            self.gpu.as_ref(),
            hidden,
            nvfp4,
            logits,
            num_tokens as usize,
            v as usize,
            h as usize,
            stream,
        )?;
        if dispatched {
            static LM_HEAD_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
            LM_HEAD_COALESCED_LOGGED.call_once(|| {
                let kernel = crate::layers::qwen4_fast_proj::select_coalesced_nvfp4_tile(
                    num_tokens as usize,
                    v as usize,
                )
                .map_or(
                    "w4a16_gemm_coalesced",
                    crate::layers::qwen4_fast_proj::CoalescedNvfp4Tile::symbol,
                );
                tracing::info!(
                    selector = crate::layers::qwen4_fast_proj::COALESCED_NVFP4_DENSE_ENV,
                    component = "lm_head_batched",
                    kernel,
                    rows = num_tokens,
                    vocab = v,
                    hidden = h,
                    "QWEN4_BATCHED_VERIFY_COALESCED_NVFP4_ENGAGED"
                );
            });
        }
        Ok(dispatched)
    }

    /// Whether target logits should run against the per-row scaled FP8 E4M3
    /// `lm_head` (default for `qwen4_exp`; drafters retain `lm_head_nvfp4`).
    pub(super) fn use_target_fp8_lm_head(&self) -> bool {
        self.config.is_qwen4_exp()
            && self.lm_head_fp8.is_some()
            && !self.use_fp32_logits
            && qwen4_target_lmhead_mode() == Qwen4TargetLmHeadMode::Fp8
    }

    /// Whether target logits should run directly against the checkpoint's
    /// BF16 `lm_head.weight` (while drafters retain `lm_head_nvfp4`).
    pub(super) fn use_target_bf16_lm_head(&self) -> bool {
        self.config.is_qwen4_exp()
            && !self.lm_head_weight.weight.is_null()
            && !self.use_fp32_logits
            && qwen4_target_lmhead_mode() == Qwen4TargetLmHeadMode::Bf16
    }

    /// True whenever target logits bypass the 4-bit NVFP4 `lm_head` in favor
    /// of the higher-precision FP8 E4M3 or BF16 target head.
    pub(super) fn use_target_high_precision_lm_head(&self) -> bool {
        self.use_target_fp8_lm_head() || self.use_target_bf16_lm_head()
    }

    /// Per-row scaled FP8 E4M3 target LM-head projection (`hidden[num_tokens, h]
    /// -> logits[num_tokens, v]`).
    ///
    /// - `M == 1`: `gemv_fp8w::dense_gemv_fp8w` (single-sweep coalesced FP8 GEMV).
    /// - `M >= 2`: single-sweep coalesced-K FP8 Tensor Core GEMM
    ///   (`fp8_gemm_coalesced_m16/m32/m64/m128`), reading `0.5x` BF16 weight
    ///   bytes directly into `logits` with native odd-vocab (`N = 248,077`)
    ///   support. Falls back cleanly to serial `dense_gemv_fp8w` or BF16.
    fn lm_head_fp8_rows(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        v: u32,
        h: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let fp8 = self
            .lm_head_fp8
            .as_ref()
            .expect("caller verified lm_head_fp8 is present");
        let logits = self.buffers.logits();
        let mut dispatched = false;
        if num_tokens == 1
            && h > 0
            && h.is_multiple_of(16)
            && hidden.0.is_multiple_of(16)
            && fp8.weight.0.is_multiple_of(16)
        {
            let gemv_k = DENSE_GEMV_FP8W_KERNEL.get_or_try_load(
                self.gpu.as_ref(),
                "gemv_fp8w",
                "dense_gemv_fp8w",
            );
            if gemv_k.0 != 0 {
                log_qwen4_target_fp8_lm_head_engagement(0, "dense_gemv_fp8w", num_tokens, v, h);
                ops::dense_gemv_fp8w(self.gpu.as_ref(), gemv_k, hidden, fp8, logits, v, h, stream)?;
                dispatched = true;
            }
        } else if select_target_fp8_lm_head_uses_coalesced_with_policy(num_tokens, det_batch)
            && h > 0
            && h.is_multiple_of(64)
            && hidden.0.is_multiple_of(16)
            && fp8.weight.0.is_multiple_of(16)
            && fp8.row_scale.0.is_multiple_of(4)
            && logits.0.is_multiple_of(2)
        {
            let (tile_m, route_bit, symbol, cache) = if num_tokens <= 16 {
                (
                    16u32,
                    1u8,
                    "fp8_gemm_coalesced_m16",
                    &FP8_GEMM_COALESCED_M16_KERNEL,
                )
            } else if num_tokens <= 32 {
                (
                    32u32,
                    2u8,
                    "fp8_gemm_coalesced_m32",
                    &FP8_GEMM_COALESCED_M32_KERNEL,
                )
            } else if num_tokens <= 64 {
                (
                    64u32,
                    3u8,
                    "fp8_gemm_coalesced_m64",
                    &FP8_GEMM_COALESCED_M64_KERNEL,
                )
            } else {
                (
                    128u32,
                    4u8,
                    "fp8_gemm_coalesced_m128",
                    &FP8_GEMM_COALESCED_M128_KERNEL,
                )
            };
            let k = cache.get_or_try_load(self.gpu.as_ref(), "w4a16", symbol);
            if k.0 != 0 {
                log_qwen4_target_fp8_lm_head_engagement(route_bit, symbol, num_tokens, v, h);
                launch_fp8_gemm_coalesced(
                    self.gpu.as_ref(),
                    k,
                    tile_m,
                    hidden,
                    fp8,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                dispatched = true;
            }
        }
        if !dispatched {
            let gemv_k = if h > 0
                && h.is_multiple_of(16)
                && hidden.0.is_multiple_of(16)
                && fp8.weight.0.is_multiple_of(16)
            {
                DENSE_GEMV_FP8W_KERNEL.get_or_try_load(
                    self.gpu.as_ref(),
                    "gemv_fp8w",
                    "dense_gemv_fp8w",
                )
            } else {
                KernelHandle(0)
            };
            if gemv_k.0 != 0 {
                log_qwen4_target_fp8_lm_head_engagement(
                    5,
                    "dense_gemv_fp8w_serial",
                    num_tokens,
                    v,
                    h,
                );
                for row in 0..num_tokens {
                    ops::dense_gemv_fp8w(
                        self.gpu.as_ref(),
                        gemv_k,
                        hidden.offset(row as usize * h as usize * 2),
                        fp8,
                        logits.offset(row as usize * v as usize * 2),
                        v,
                        h,
                        stream,
                    )?;
                }
            } else if !self.lm_head_weight.weight.is_null() {
                return self.lm_head_bf16_rows(hidden, num_tokens, v, h, stream);
            } else {
                bail!("FP8 target LM head kernels unavailable and BF16 lm_head was released");
            }
        }
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    /// BF16 target LM-head projection (`hidden[num_tokens, h] -> logits[num_tokens, v]`).
    ///
    /// - `M == 1`: exact `dense_gemv_bf16` (0 ULP, single weight read).
    /// - `M >= 2`: single-sweep coalesced-K BF16 Tensor Core GEMM
    ///   (`bf16_gemm_coalesced_m16/m32/m64/m128`), which streams `lm_head.weight`
    ///   once at full bandwidth directly into `logits` with zero scratch buffer
    ///   and native odd-vocab (`N = 248,077`) support. Falls back cleanly to
    ///   cuBLASLt (when aligned), `dense_gemv_bf16_batchn`, or `dense_gemm_bf16`.
    fn lm_head_bf16_rows(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        v: u32,
        h: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let logits = self.buffers.logits();
        if select_target_bf16_lm_head_serial_gemv_with_policy(num_tokens, det_batch) {
            log_qwen4_target_bf16_lm_head_engagement(0, "dense_gemv_bf16", num_tokens, v, h);
            for row in 0..num_tokens {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden.offset(row as usize * h as usize * 2),
                    &self.lm_head_weight,
                    logits.offset(row as usize * v as usize * 2),
                    v,
                    h,
                    stream,
                )?;
            }
        } else {
            let mut dispatched = false;
            if h > 0
                && h.is_multiple_of(64)
                && hidden.0.is_multiple_of(16)
                && self.lm_head_weight.weight.0.is_multiple_of(16)
                && logits.0.is_multiple_of(2)
            {
                let (tile_m, route_bit, symbol, cache) = if num_tokens <= 16 {
                    (
                        16u32,
                        4u8,
                        "bf16_gemm_coalesced_m16",
                        &BF16_GEMM_COALESCED_M16_KERNEL,
                    )
                } else if num_tokens <= 32 {
                    (
                        32u32,
                        5u8,
                        "bf16_gemm_coalesced_m32",
                        &BF16_GEMM_COALESCED_M32_KERNEL,
                    )
                } else if num_tokens <= 64 {
                    (
                        64u32,
                        6u8,
                        "bf16_gemm_coalesced_m64",
                        &BF16_GEMM_COALESCED_M64_KERNEL,
                    )
                } else {
                    (
                        128u32,
                        7u8,
                        "bf16_gemm_coalesced_m128",
                        &BF16_GEMM_COALESCED_M128_KERNEL,
                    )
                };
                let k = cache.get_or_try_load(self.gpu.as_ref(), "w4a16", symbol);
                if k.0 != 0 {
                    log_qwen4_target_bf16_lm_head_engagement(route_bit, symbol, num_tokens, v, h);
                    launch_bf16_gemm_coalesced(
                        self.gpu.as_ref(),
                        k,
                        tile_m,
                        hidden,
                        &self.lm_head_weight,
                        logits,
                        num_tokens,
                        v,
                        h,
                        stream,
                    )?;
                    dispatched = true;
                }
            }
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if !dispatched
                && num_tokens > 25
                && self.gpu.sm_count().is_some()
                && v.is_multiple_of(8)
                && h.is_multiple_of(8)
                && hidden.0.is_multiple_of(16)
                && self.lm_head_weight.weight.0.is_multiple_of(16)
                && logits.0.is_multiple_of(16)
            {
                log_qwen4_target_bf16_lm_head_engagement(1, "cublaslt_bf16_gemm", num_tokens, v, h);
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    hidden.0,
                    self.lm_head_weight.weight.0,
                    logits.0,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                dispatched = true;
            }
            if !dispatched {
                let batchn_k = DENSE_GEMV_BF16_BATCHN_KERNEL.get_or_try_load(
                    self.gpu.as_ref(),
                    "gemv",
                    "dense_gemv_bf16_batchn",
                );
                if batchn_k.0 != 0 {
                    log_qwen4_target_bf16_lm_head_engagement(
                        2,
                        "dense_gemv_bf16_batchn",
                        num_tokens,
                        v,
                        h,
                    );
                    ops::dense_gemv_batchn(
                        self.gpu.as_ref(),
                        batchn_k,
                        hidden,
                        &self.lm_head_weight,
                        logits,
                        num_tokens,
                        v,
                        h,
                        stream,
                    )?;
                } else {
                    log_qwen4_target_bf16_lm_head_engagement(
                        3,
                        "dense_gemm_bf16",
                        num_tokens,
                        v,
                        h,
                    );
                    ops::dense_gemm(
                        self.gpu.as_ref(),
                        self.dense_gemm_kernel,
                        hidden,
                        &self.lm_head_weight,
                        logits,
                        num_tokens,
                        v,
                        h,
                        stream,
                    )?;
                }
            }
        }
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    /// Resolve the `INT8 Block-32 BF16-Scale` LM-head weight, preferring the
    /// startup-quantized owned weight on `self` (so it remains available even
    /// after `QWEN4_RELEASE_*_UNUSED` frees BF16 `self.lm_head_weight.weight`)
    /// and falling back to the capture-safe lazy cache when BF16 is still live.
    fn resolve_lm_head_int8_block32_weight(
        &self,
        full_vocab: u32,
        h: u32,
        quantize_k: KernelHandle,
        stream: u64,
    ) -> Result<Option<ops::Int8Block32DeviceWeight>> {
        if let Some(owned) = self.lm_head_int8_block32 {
            if owned.rows >= full_vocab && owned.cols == h {
                return Ok(Some(owned));
            }
            return Ok(None);
        }
        if self.lm_head_weight.weight.is_null() || !self.lm_head_weight.weight.0.is_multiple_of(2) {
            return Ok(None);
        }
        let capturing = spark_runtime::gpu_capture_region::active_capture_region().is_some()
            || self.gpu.stream_is_capturing(stream);
        let alloc_policy = if capturing {
            ops::Int8WeightAllocPolicy::CaptureNoAlloc
        } else {
            ops::Int8WeightAllocPolicy::AllowAlloc
        };
        ops::ensure_cached_lm_head_int8_block32_weight(
            self.gpu.as_ref(),
            quantize_k,
            self.lm_head_weight.weight,
            full_vocab,
            h,
            stream,
            alloc_policy,
        )
    }

    /// Opt-in (`ATLAS_LM_HEAD_INT8_BLOCK32=1`, default OFF) two-stage fused
    /// `INT8 Block-32 BF16-Scale (8.50 bpw)` LM-head + online warp/block argmax.
    fn try_lm_head_int8_block32_fused_argmax(
        &self,
        hidden: DevicePtr,
        out_tokens: DevicePtr,
        rows: u32,
        vocab: u32,
        stream: u64,
    ) -> Result<bool> {
        let policy = ops::LmHeadInt8Block32Policy::from_env()?;
        if policy == ops::LmHeadInt8Block32Policy::Disabled {
            return Ok(false);
        }
        let h = self.config.hidden_size as u32;
        let full_vocab = self.config.vocab_size as u32;
        if self.use_fp32_logits
            || self.logit_softcap_kernel.0 != 0
            || self.config.final_logit_softcapping != 0.0
            || !self.config.final_logit_softcapping.is_finite()
            || rows == 0
            || rows > ops::LM_HEAD_INT8_ARGMAX_MAX_M
            || vocab == 0
            || vocab > full_vocab
            || h < ops::LM_HEAD_INT8_BLOCK_SIZE
            || !h.is_multiple_of(ops::LM_HEAD_INT8_BLOCK_SIZE)
            || hidden.is_null()
            || !hidden.0.is_multiple_of(16)
            || out_tokens.is_null()
            || !out_tokens.0.is_multiple_of(4)
        {
            return Ok(false);
        }
        let kernels = ops::cached_lm_head_int8_block32_kernels(self.gpu.as_ref());
        if !kernels.argmax_ready() {
            return Ok(false);
        }
        let Ok(plan) = ops::plan_lm_head_int8_block32_argmax(rows, vocab, h) else {
            return Ok(false);
        };
        let scratch = self.buffers.logits();
        let scratch_bytes = self.buffers.sizes().logits;
        if scratch.is_null()
            || !scratch.0.is_multiple_of(4)
            || scratch_bytes < plan.total_scratch_bytes
        {
            return Ok(false);
        }
        let Some(dev_weight) =
            self.resolve_lm_head_int8_block32_weight(full_vocab, h, kernels.quantize, stream)?
        else {
            return Ok(false);
        };
        ops::lm_head_int8_block32_fused_argmax(
            self.gpu.as_ref(),
            kernels,
            &plan,
            hidden,
            &dev_weight,
            scratch,
            scratch_bytes,
            out_tokens,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                target: "atlas::lm_head",
                mode = "fused_argmax",
                eager_owned = self.lm_head_int8_block32.is_some(),
                bf16_reclaimed = self.lm_head_weight.weight.is_null(),
                rows = plan.rows,
                vocab = plan.vocab,
                hidden = plan.hidden,
                num_v_tiles = plan.num_v_tiles,
                scratch_bytes = plan.total_scratch_bytes,
                "LM_HEAD_INT8_BLOCK32_ENGAGED"
            );
        });
        Ok(true)
    }

    /// Opt-in (`ATLAS_LM_HEAD_INT8_BLOCK32=1`, default OFF) full BF16 logit
    /// GEMM (`M = 1..=64`, single-CTA `grid.y = 1`) using `INT8 Block-32
    /// BF16-Scale (8.50 bpw)` weights.
    fn try_lm_head_int8_block32_logits(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        vocab: u32,
        h: u32,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let policy = ops::LmHeadInt8Block32Policy::from_env()?;
        if policy == ops::LmHeadInt8Block32Policy::Disabled {
            return Ok(None);
        }
        let full_vocab = self.config.vocab_size as u32;
        let logits = self.buffers.logits();
        if self.use_fp32_logits
            || num_tokens == 0
            || num_tokens > ops::LM_HEAD_INT8_GEMM_MAX_M
            || vocab == 0
            || vocab > full_vocab
            || h < ops::LM_HEAD_INT8_BLOCK_SIZE
            || !h.is_multiple_of(ops::LM_HEAD_INT8_BLOCK_SIZE)
            || hidden.is_null()
            || !hidden.0.is_multiple_of(16)
            || logits.is_null()
            || !logits.0.is_multiple_of(2)
        {
            return Ok(None);
        }
        let kernels = ops::cached_lm_head_int8_block32_kernels(self.gpu.as_ref());
        if !kernels.gemm_ready() && !kernels.gemm_m64_ready() {
            return Ok(None);
        }
        let Ok(plan) = ops::plan_lm_head_int8_block32_gemm(num_tokens, vocab, h) else {
            return Ok(None);
        };
        let logits_bytes = self.buffers.sizes().logits;
        if logits_bytes < plan.c_bytes {
            return Ok(None);
        }
        let Some(dev_weight) =
            self.resolve_lm_head_int8_block32_weight(full_vocab, h, kernels.quantize, stream)?
        else {
            return Ok(None);
        };
        let block_threads = ops::lm_head_int8_block32_gemm_dispatch(
            self.gpu.as_ref(),
            kernels,
            &plan,
            hidden,
            &dev_weight,
            logits,
            logits_bytes,
            stream,
        )?;
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            self.apply_logit_softcap(logits, num_tokens * vocab, cap, stream)?;
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                target: "atlas::lm_head",
                mode = "full_logits",
                eager_owned = self.lm_head_int8_block32.is_some(),
                bf16_reclaimed = self.lm_head_weight.weight.is_null(),
                rows = plan.rows,
                vocab = plan.vocab,
                hidden = plan.hidden,
                grid_y = plan.grid_y,
                m_blocks = plan.m_blocks,
                block_threads,
                "LM_HEAD_INT8_BLOCK32_ENGAGED"
            );
        });
        Ok(Some(logits))
    }

    /// Opt-in fused two-stage NVFP4 LM-head + online warp/block argmax
    /// (`ATLAS_LM_HEAD_FUSED_ARGMAX=1`, default OFF).
    ///
    /// Computes `[rows]` greedy argmax token IDs directly into `out_tokens`
    /// using `self.buffers.logits()` purely as compact partial-reduction
    /// scratch (`<= 124 KiB` for `rows <= 16`, `V = 248,077`), eliminating the
    /// full `[rows, V]` BF16 logit write+read slab. Returns `Ok(false)` (failing
    /// closed to standard `lm_head_batched` + `argmax_bf16*`) whenever disabled
    /// or any admission guard is unmet.
    pub(super) fn try_lm_head_fused_argmax(
        &self,
        hidden: DevicePtr,
        out_tokens: DevicePtr,
        rows: u32,
        vocab: u32,
        stream: u64,
    ) -> Result<bool> {
        if self.try_lm_head_int8_block32_fused_argmax(hidden, out_tokens, rows, vocab, stream)? {
            return Ok(true);
        }
        let policy = ops::FusedLmHeadArgmaxPolicy::from_env()?;
        if policy == ops::FusedLmHeadArgmaxPolicy::Disabled {
            return Ok(false);
        }
        if self.use_fp32_logits {
            return Ok(false);
        }
        let weight_format = if self.use_target_fp8_lm_head() {
            ops::LmHeadWeightFormat::Fp8
        } else if self.use_target_bf16_lm_head() {
            ops::LmHeadWeightFormat::Bf16
        } else if self.lm_head_nvfp4.is_some() {
            ops::LmHeadWeightFormat::Nvfp4
        } else {
            ops::LmHeadWeightFormat::Bf16
        };
        let Some(ref nvfp4) = self.lm_head_nvfp4 else {
            return Ok(false);
        };
        let partial_k = LM_HEAD_NVFP4_ARGMAX_PARTIAL_KERNEL.get_or_try_load(
            self.gpu.as_ref(),
            "lm_head_nvfp4_argmax_sm120",
            "lm_head_nvfp4_argmax_partial_sm120",
        );
        let reduce_k = LM_HEAD_ARGMAX_REDUCE_KERNEL.get_or_try_load(
            self.gpu.as_ref(),
            "lm_head_nvfp4_argmax_sm120",
            "lm_head_argmax_reduce_sm120",
        );
        let kernels = if partial_k.0 != 0 && reduce_k.0 != 0 {
            ops::LmHeadKernelAvailability::Ready
        } else {
            ops::LmHeadKernelAvailability::Unavailable
        };
        let softcap = if self.logit_softcap_kernel.0 != 0
            || self.config.final_logit_softcapping != 0.0
            || !self.config.final_logit_softcapping.is_finite()
        {
            ops::LmHeadSoftcapMode::Enabled
        } else {
            ops::LmHeadSoftcapMode::Disabled
        };
        let req = ops::LmHeadArgmaxAdmissionRequest {
            sampling: ops::LmHeadSamplingMode::GreedyArgmax,
            penalties: ops::LmHeadPenaltyMode::Inactive,
            grammar: ops::LmHeadGrammarMode::Inactive,
            logprobs: ops::LmHeadLogprobsMode::Omitted,
            softcap,
            weight_format,
            kernels,
            rows,
            vocab,
            hidden: self.config.hidden_size as u32,
        };
        let Ok(plan) = ops::admit_fused_lm_head_argmax(policy, &req) else {
            return Ok(false);
        };
        let scratch = self.buffers.logits();
        let scratch_bytes = self.buffers.sizes().logits;
        if scratch == DevicePtr::NULL || scratch_bytes < plan.total_scratch_bytes {
            return Ok(false);
        }
        ops::lm_head_nvfp4_fused_argmax(
            self.gpu.as_ref(),
            partial_k,
            reduce_k,
            &plan,
            hidden,
            nvfp4,
            scratch,
            scratch_bytes,
            out_tokens,
            stream,
        )?;
        static FUSED_LOGGED: std::sync::Once = std::sync::Once::new();
        FUSED_LOGGED.call_once(|| {
            tracing::info!(
                target: "atlas::lm_head",
                rows = plan.rows,
                vocab = plan.vocab,
                hidden = plan.hidden,
                num_v_tiles = plan.num_v_tiles,
                scratch_bytes = plan.total_scratch_bytes,
                "ENGAGED ATLAS_LM_HEAD_FUSED_ARGMAX: two-stage NVFP4 LM-head + online argmax"
            );
        });
        Ok(true)
    }

    pub(super) fn lm_head_batched(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let h = self.config.hidden_size as u32;
        // Verify-side vocab truncation: shrink the logical GEMM/GEMV output
        // dimension (and the matching argmax range in the scheduler) to the
        // first `v` rows. BPE places frequent tokens at low IDs, so reading
        // fewer vocab rows is a clean bandwidth reduction with negligible
        // quality risk for `v` large enough to cover normal text.
        // `verify_lmhead_vocab()` returns the full vocab when truncation is
        // disabled (`ATLAS_TARGET_LMHEAD_VOCAB` unset / 0 / ≥ vocab).
        let v = self.verify_lmhead_vocab();
        if let Some(logits) =
            self.try_lm_head_int8_block32_logits(hidden, num_tokens, v, h, stream)?
        {
            return Ok(logits);
        }
        if self.use_target_fp8_lm_head() {
            return self.lm_head_fp8_rows(hidden, num_tokens, v, h, stream);
        }
        if self.use_target_bf16_lm_head() {
            return self.lm_head_bf16_rows(hidden, num_tokens, v, h, stream);
        }
        let logits = self.buffers.logits();
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            anyhow::ensure!(
                (1..=128).contains(&num_tokens),
                "NVFP4 speculative LM-head rows must be in 1..=128, got {num_tokens}"
            );
            let logits_cap_rows = self
                .buffers
                .sizes()
                .logits
                .checked_div((v as usize).saturating_mul(2))
                .unwrap_or(0);
            anyhow::ensure!(
                (num_tokens as usize) <= logits_cap_rows,
                "NVFP4 speculative LM-head rows {num_tokens} exceed logits buffer capacity {logits_cap_rows}"
            );
            let sm_count = atlas_core::device::active_sm_count();
            let sm120_auto_tc = should_auto_tc_batched_lm_head_for_policy(
                self.config.is_qwen4_exp(),
                matches!(raw_lm_head_tc_env_override(), LmHeadTcEnvOverride::Disabled),
                sm_count,
                num_tokens,
                det_batch,
            );
            let graph_capture = spark_runtime::gpu_capture_region::active_capture_region()
                .is_some()
                || self.gpu.stream_is_capturing(stream);
            if num_tokens == 1 && !det_batch.is_enabled() && self.decode_tc_parity_lm_head() {
                self.lm_head_tc_rows(hidden, logits, 1, v, stream)?;
            } else if num_tokens == 1 {
                // A one-row tail (for example, a chunked diagnostic verify)
                // stays on the ordinary qualified K1 LM-head path.
                ops::w4a16_decode_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    self.w4a16_gemv_sw_kernel,
                    self.gemv_sw,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else if sm120_auto_tc
                || (!det_batch.is_enabled()
                    && lm_head_tc_enabled()
                    && self.lm_head_nvfp4_t.is_some()
                    && self.w4a16_gemm_t_m32_n64_kernel.0 != 0)
            {
                let mut dispatched = false;
                let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
                if sm_count > HIGH_SM_LM_HEAD_THRESHOLD
                    && (num_tokens as usize) <= sm120_max_m
                    && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                        self.gpu.as_ref(),
                        hidden,
                        nvfp4,
                        logits,
                        num_tokens,
                        v,
                        h,
                        sm120_max_m,
                        stream,
                    )?
                {
                    static M16_SK4_LOGGED: std::sync::Once = std::sync::Once::new();
                    M16_SK4_LOGGED.call_once(|| {
                        tracing::info!(
                            route = "sm120_coalesced_m16_small_m",
                            rows = num_tokens,
                            vocab = v,
                            hidden = h,
                            "LM_HEAD_TC_COALESCED_M16_ENGAGED"
                        );
                    });
                    dispatched = true;
                } else if should_prefer_m16_coalesced_lm_head(num_tokens, sm_count)
                    && h.is_multiple_of(64)
                    && hidden.0.is_multiple_of(16)
                    && nvfp4.weight.0.is_multiple_of(16)
                    && nvfp4.weight_scale.0.is_multiple_of(4)
                    && logits.0.is_multiple_of(2)
                    && nvfp4.weight_scale_2.is_finite()
                {
                    let mut m16_k = W4A16_GEMM_COALESCED_V2_N64_M16_KERNEL.get_or_try_load(
                        self.gpu.as_ref(),
                        "w4a16",
                        "w4a16_gemm_coalesced_v2_n64_m16",
                    );
                    if m16_k.0 == 0 {
                        m16_k = W4A16_GEMM_COALESCED_V2_N64_M16_FALLBACK_KERNEL.get_or_try_load(
                            self.gpu.as_ref(),
                            "w4a16_gemm",
                            "w4a16_gemm_coalesced_v2_n64_m16",
                        );
                    }
                    if m16_k.0 != 0 {
                        KernelLaunch::new(self.gpu.as_ref(), m16_k)
                            .grid([div_ceil(v, 64), div_ceil(num_tokens, 16), 1])
                            .block([256, 1, 1])
                            .arg_ptr(hidden)
                            .arg_ptr(nvfp4.weight)
                            .arg_ptr(nvfp4.weight_scale)
                            .arg_f32(nvfp4.weight_scale_2)
                            .arg_ptr(logits)
                            .arg_u32(num_tokens)
                            .arg_u32(v)
                            .arg_u32(h)
                            .launch(stream)?;
                        static M16_LOGGED: std::sync::Once = std::sync::Once::new();
                        M16_LOGGED.call_once(|| {
                            tracing::info!(
                                route = "w4a16_gemm_coalesced_v2_n64_m16",
                                rows = num_tokens,
                                vocab = v,
                                hidden = h,
                                "LM_HEAD_TC_COALESCED_M16_ENGAGED"
                            );
                        });
                        dispatched = true;
                    }
                }
                if !dispatched
                    && sm_count > HIGH_SM_LM_HEAD_THRESHOLD
                    && num_tokens >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M
                    && ops::try_cublaslt_nvfp4_multiseq_single(
                        self.gpu.as_ref(),
                        hidden,
                        nvfp4,
                        logits,
                        num_tokens,
                        v,
                        h,
                        v,
                        graph_capture,
                        "lm_head_batched",
                        stream,
                    )?
                {
                    static CUBLASLT_LOGGED: std::sync::Once = std::sync::Once::new();
                    CUBLASLT_LOGGED.call_once(|| {
                        tracing::info!(
                            route = "cublaslt_nvfp4",
                            rows = num_tokens,
                            vocab = v,
                            hidden = h,
                            "LM_HEAD_TC_CUBLASLT_ENGAGED"
                        );
                    });
                    dispatched = true;
                }
                if !dispatched
                    && self.lm_head_nvfp4_t.is_some()
                    && self.w4a16_gemm_t_m32_n64_kernel.0 != 0
                {
                    log_lm_head_tc_engagement(num_tokens, v, h);
                    self.lm_head_tc_rows(hidden, logits, num_tokens, v, stream)?;
                    dispatched = true;
                }
                if !dispatched
                    && self.try_lm_head_coalesced_nvfp4(
                        hidden, nvfp4, logits, num_tokens, v, h, stream,
                    )?
                {
                    dispatched = true;
                }
                if !dispatched {
                    let route = self
                        .w4a16_exact_lm_head_kernels
                        .route_for_rows(num_tokens)
                        .expect("NVFP4 M=2..=128 has an exact tier");
                    log_exact_lm_head_engagement(route, num_tokens, v, h);
                    match route {
                        ops::ExactLmHeadRoute::Exact(_) => {
                            ops::w4a16_gemv_batch_logits_exact(
                                self.gpu.as_ref(),
                                self.w4a16_exact_lm_head_kernels,
                                hidden,
                                nvfp4,
                                logits,
                                num_tokens,
                                v,
                                h,
                                stream,
                            )?;
                        }
                        ops::ExactLmHeadRoute::SerialK1(_) => {
                            for row in 0..num_tokens {
                                ops::w4a16_decode_gemv(
                                    self.gpu.as_ref(),
                                    self.w4a16_gemv_kernel,
                                    self.w4a16_gemv_sw_kernel,
                                    self.gemv_sw,
                                    hidden.offset(row as usize * h as usize * 2),
                                    nvfp4,
                                    logits.offset(row as usize * v as usize * 2),
                                    v,
                                    h,
                                    stream,
                                )?;
                            }
                        }
                    }
                }
            } else if crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
                && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                    crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                    num_tokens as usize,
                    false,
                )
                && self
                    .try_lm_head_coalesced_nvfp4(hidden, nvfp4, logits, num_tokens, v, h, stream)?
            {
                // Dispatched via zero-VRAM, CUDA-graph-safe coalesced NVFP4
                // GEMM for scoped batched-verify widths (`rows >= 4` default;
                // `C=1` stays on the strict 0-ULP scalar GEMV path because
                // TC-dense scope is entered only for `C >= 2`).
            } else {
                let route = self
                    .w4a16_exact_lm_head_kernels
                    .route_for_rows(num_tokens)
                    .expect("NVFP4 M=2..=128 has an exact tier");
                log_exact_lm_head_engagement(route, num_tokens, v, h);
                match route {
                    ops::ExactLmHeadRoute::Exact(_) => {
                        ops::w4a16_gemv_batch_logits_exact(
                            self.gpu.as_ref(),
                            self.w4a16_exact_lm_head_kernels,
                            hidden,
                            nvfp4,
                            logits,
                            num_tokens,
                            v,
                            h,
                            stream,
                        )?;
                    }
                    ops::ExactLmHeadRoute::SerialK1(_) => {
                        // Missing exact symbols fail closed to M independent
                        // ordinary K1 GEMVs. Row-major output uses the logical
                        // (possibly truncated) vocab as its stride.
                        for row in 0..num_tokens {
                            ops::w4a16_decode_gemv(
                                self.gpu.as_ref(),
                                self.w4a16_gemv_kernel,
                                self.w4a16_gemv_sw_kernel,
                                self.gemv_sw,
                                hidden.offset(row as usize * h as usize * 2),
                                nvfp4,
                                logits.offset(row as usize * v as usize * 2),
                                v,
                                h,
                                stream,
                            )?;
                        }
                    }
                }
            }
        } else if select_dense_bf16_lm_head_serial_gemv_with_policy(num_tokens, det_batch) {
            // Preserve dense M=1..=2 as BF16 GEMVs, and under
            // ATLAS_DETERMINISTIC_BATCH=1 keep all M=1..=128 on identical
            // per-row dense_gemv_bf16 (no M>=3 dense_gemm flip).
            for row in 0..num_tokens {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden.offset(row as usize * h as usize * 2),
                    &self.lm_head_weight,
                    logits.offset(row as usize * v as usize * 2),
                    v,
                    h,
                    stream,
                )?;
            }
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        }
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    /// Multi-sequence Qwen4 LM-head dispatch admitting `1..=QWEN4_NATIVE_MAX_ROWS`
    /// (`1..=256`), delegating `1..=32` directly to [`Self::lm_head_batched`] and
    /// routing `33..=256` through the tiled exact / K1-fallback NVFP4 path
    /// (chunking `> 128` rows into `<= 128`-row exact slices when TC is off).
    pub(super) fn lm_head_qwen4_rows(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        if num_tokens <= 32 {
            return self.lm_head_batched(hidden, num_tokens, stream);
        }
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        anyhow::ensure!(
            (33..=crate::layers::QWEN4_NATIVE_MAX_ROWS as u32).contains(&num_tokens),
            "Qwen4 multi-sequence LM-head rows must be in 1..={}, got {num_tokens}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        let h = self.config.hidden_size as u32;
        let v = self.verify_lmhead_vocab();
        if let Some(logits) =
            self.try_lm_head_int8_block32_logits(hidden, num_tokens, v, h, stream)?
        {
            return Ok(logits);
        }
        if self.use_target_fp8_lm_head() {
            return self.lm_head_fp8_rows(hidden, num_tokens, v, h, stream);
        }
        if self.use_target_bf16_lm_head() {
            return self.lm_head_bf16_rows(hidden, num_tokens, v, h, stream);
        }
        let logits = self.buffers.logits();
        if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            if !det_batch.is_enabled()
                && lm_head_tc_enabled()
                && self.lm_head_nvfp4_t.is_some()
                && self.w4a16_gemm_t_m32_n64_kernel.0 != 0
            {
                log_lm_head_tc_engagement(num_tokens, v, h);
                self.lm_head_tc_rows(hidden, logits, num_tokens, v, stream)?;
            } else {
                const EXACT_LM_HEAD_MAX_CHUNK: u32 = 128;
                let mut offset = 0u32;
                while offset < num_tokens {
                    let remaining = num_tokens - offset;
                    // Exact multi-row tiers cover `2..=128`; split `remaining == 129`
                    // as `64 + 65` so no chunk ever lands on `1`.
                    let chunk = if remaining == EXACT_LM_HEAD_MAX_CHUNK + 1 {
                        64
                    } else {
                        remaining.min(EXACT_LM_HEAD_MAX_CHUNK)
                    };
                    let chunk_hidden = hidden.offset(offset as usize * h as usize * 2);
                    let chunk_logits = logits.offset(offset as usize * v as usize * 2);
                    let route = self
                        .w4a16_exact_lm_head_kernels
                        .route_for_rows(chunk)
                        .expect("NVFP4 M=2..=128 has an exact tier");
                    log_exact_lm_head_engagement(route, chunk, v, h);
                    match route {
                        ops::ExactLmHeadRoute::Exact(_) => {
                            ops::w4a16_gemv_batch_logits_exact(
                                self.gpu.as_ref(),
                                self.w4a16_exact_lm_head_kernels,
                                chunk_hidden,
                                nvfp4,
                                chunk_logits,
                                chunk,
                                v,
                                h,
                                stream,
                            )?;
                        }
                        ops::ExactLmHeadRoute::SerialK1(_) => {
                            for row in 0..chunk {
                                ops::w4a16_decode_gemv(
                                    self.gpu.as_ref(),
                                    self.w4a16_gemv_kernel,
                                    self.w4a16_gemv_sw_kernel,
                                    self.gemv_sw,
                                    chunk_hidden.offset(row as usize * h as usize * 2),
                                    nvfp4,
                                    chunk_logits.offset(row as usize * v as usize * 2),
                                    v,
                                    h,
                                    stream,
                                )?;
                            }
                        }
                    }
                    offset += chunk;
                }
            }
        } else if select_dense_bf16_lm_head_serial_gemv_with_policy(num_tokens, det_batch) {
            for row in 0..num_tokens {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden.offset(row as usize * h as usize * 2),
                    &self.lm_head_weight,
                    logits.offset(row as usize * v as usize * 2),
                    v,
                    h,
                    stream,
                )?;
            }
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        }
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    /// `ATLAS_DECODE_TC_PARITY=1` with `ATLAS_LM_HEAD_TC=1`: single-token
    /// decode shares the verify's tensor-core LM head.
    fn decode_tc_parity_lm_head(&self) -> bool {
        crate::layers::decode_tc_parity_enabled()
            && lm_head_tc_enabled()
            && self.lm_head_nvfp4.is_some()
            && self.lm_head_nvfp4_t.is_some()
            && self.w4a16_gemm_t_m32_n64_kernel.0 != 0
    }

    /// `rows` hidden rows through the transposed-NVFP4 m32_n64 tensor-core
    /// LM head. ldb is the 64-padded vocab stride (== vocab for 248320);
    /// output stays row-major `[rows, v]`.
    fn lm_head_tc_rows(
        &self,
        hidden: DevicePtr,
        logits: DevicePtr,
        rows: u32,
        v: u32,
        stream: u64,
    ) -> Result<()> {
        let ldb = (self.config.vocab_size.div_ceil(64) * 64) as u32;
        ops::w4a16_gemm_n64_m32_ldb(
            self.gpu.as_ref(),
            self.w4a16_gemm_t_m32_n64_kernel,
            hidden,
            self.lm_head_nvfp4_t
                .as_ref()
                .expect("caller checked the transposed LM head"),
            logits,
            rows,
            v,
            self.config.hidden_size as u32,
            ldb,
            stream,
        )
    }

    pub(super) fn lm_head(&self, hidden: DevicePtr, stream: u64) -> Result<DevicePtr> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // Pick the output buffer: FP32 scratch when use_fp32_logits is on,
        // shared BF16 buffer otherwise. The sampler must use the matching
        // dtype — see `decode_logits_fp32()`.
        let (logits, fp32) = if self.use_fp32_logits {
            (self.logits_fp32_buf, true)
        } else {
            (self.buffers.logits(), false)
        };
        if !fp32
            && let Some(int8_logits) =
                self.try_lm_head_int8_block32_logits(hidden, 1, v, h, stream)?
        {
            return Ok(int8_logits);
        }
        if !fp32 && self.use_target_fp8_lm_head() {
            return self.lm_head_fp8_rows(hidden, 1, v, h, stream);
        }
        if !fp32 && self.use_target_bf16_lm_head() {
            return self.lm_head_bf16_rows(hidden, 1, v, h, stream);
        }
        if !fp32 && !det_batch.is_enabled() && self.decode_tc_parity_lm_head() {
            // ATLAS_DECODE_TC_PARITY=1: one row through the verify's
            // ATLAS_LM_HEAD_TC=1 kernel so plain decode and speculative
            // verify produce bit-identical logits.
            self.lm_head_tc_rows(hidden, logits, 1, v, stream)?;
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // Pick FP32-output variant when the FP32 logits buffer is the
            // destination. Same packed-NVFP4 weights, same activation, but the
            // accumulator is NOT downcast to BF16 — closes the 0.125-logit
            // BF16-rounding tiebreak flip that triggers Gemma-4-31B's
            // creative-collapse stop-word loop.
            if fp32 {
                ops::w4a16_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_logits_kernel,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_decode_gemv(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_kernel,
                    self.w4a16_gemv_sw_kernel,
                    self.gemv_sw,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            }
        } else if fp32 {
            // FP32-output dense GEMV: same precision-preservation reason as
            // the NVFP4 variant above. Used when Gemma keeps the LM head
            // as BF16 (skip_lm_head_quantization=true).
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_fp32out_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        }
        // Apply logit softcapping: logits = cap * tanh(logits / cap)
        if self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            self.apply_logit_softcap_dtype(logits, v, cap, fp32, stream)?;
        }
        Ok(logits)
    }

    /// Apply logit softcapping in-place: `logits[i] = cap * tanh(logits[i] / cap)`.
    /// BF16 path. No-op when `cap <= 0.0` or the kernel is absent. Use
    /// `apply_logit_softcap_dtype` to dispatch by buffer dtype.
    pub(super) fn apply_logit_softcap(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        if self.logit_softcap_kernel.0 == 0 || cap <= 0.0 {
            return Ok(());
        }
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), self.logit_softcap_kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// Dtype-aware softcap dispatcher. Picks the BF16 or FP32 kernel based on
    /// whether the buffer holds FP32 logits. No-op when softcap is disabled
    /// (`cap <= 0.0`). Used by the single-token decode `lm_head` to keep the
    /// FP32 path symmetrical when `use_fp32_logits` is on.
    pub(super) fn apply_logit_softcap_dtype(
        &self,
        logits: DevicePtr,
        num_elements: u32,
        cap: f32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let kernel = if is_fp32 {
            self.logit_softcap_fp32_kernel
        } else {
            self.logit_softcap_kernel
        };
        if kernel.0 == 0 || cap <= 0.0 {
            return Ok(());
        }
        let inv_cap = 1.0f32 / cap;
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([num_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_u32(num_elements)
            .arg_f32(inv_cap)
            .arg_f32(cap)
            .launch(stream)
    }

    /// True when single-token decode `lm_head` writes FP32 logits to
    /// `logits_fp32_buf`. Callers that consume those logits (sampler) MUST
    /// read with the matching dtype. Prefill / batched-decode lm_head still
    /// produce BF16, so this only applies to the `lm_head` (single-token)
    /// return value.
    pub fn decode_logits_fp32(&self) -> bool {
        self.use_fp32_logits
    }

    /// Buffer pointer the single-token decode `lm_head` last wrote to. FP32
    /// scratch when `use_fp32_logits`, otherwise the shared BF16 logits
    /// buffer. Callers that previously hard-coded `self.buffers.logits()`
    /// after `self.lm_head(...)` must use this so the sampler reads the
    /// correct buffer dtype (the BF16 buffer is stale/empty in the FP32
    /// path because lm_head writes elsewhere). Pair with
    /// `logits_ptr_is_fp32` / `decode_logits_fp32` for dtype-aware reads.
    pub fn decode_logits_ptr(&self) -> DevicePtr {
        if self.use_fp32_logits {
            self.logits_fp32_buf
        } else {
            self.buffers.logits()
        }
    }

    /// Batched-verify Qwen4 LM-head dispatch (`1..=QWEN4_NATIVE_MAX_ROWS`).
    ///
    /// Preserves the exact 0-ULP scalar GEMV path ([`Self::lm_head_qwen4_rows`])
    /// whenever `ATLAS_QWEN4_BATCHED_VERIFY_TC_DENSE` is off or
    /// `num_tokens < min_rows` (`LmHead` default `4`, so `rows <= 3` stays 100%
    /// exact; `C=1` stays exact because TC-dense scope is entered only for
    /// `C >= 2`). For `rows >= 4` (`C >= 2` scoped frames), dispatches
    /// Tensor Core NVFP4 GEMM:
    /// - `w4a16_gemm_t_m32_n64` when `ATLAS_LM_HEAD_TC=1` + transposed weights
    ///   are present;
    /// - otherwise the coalesced-K single-sweep row-major NVFP4 Tensor-Core
    ///   tier (`w4a16_gemm_coalesced_m32` for `4..=32`, `m64` for `33..=64`,
    ///   `m128` for `65..=128`) directly on `lm_head_nvfp4` with zero extra
    ///   weight memory and full odd-vocab (`N = 248077`) support;
    /// - falling back to `w4a16_gemm_pipe_m128n128` / `w4a16_gemm_pipe` /
    ///   `w4a16_gemm` / exact GEMV if a newer symbol is absent.
    pub(super) fn lm_head_qwen4_batched_verify_rows(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.verify_lmhead_vocab();
        if let Some(logits) =
            self.try_lm_head_int8_block32_logits(hidden, num_tokens, v, h, stream)?
        {
            return Ok(logits);
        }
        if self.use_target_fp8_lm_head() {
            anyhow::ensure!(
                (1..=crate::layers::QWEN4_NATIVE_MAX_ROWS as u32).contains(&num_tokens),
                "FP8 Qwen4 batched-verify LM-head rows must be in 1..={}, got {num_tokens}",
                crate::layers::QWEN4_NATIVE_MAX_ROWS
            );
            let h = self.config.hidden_size as u32;
            let v = self.verify_lmhead_vocab();
            return self.lm_head_fp8_rows(hidden, num_tokens, v, h, stream);
        }
        if self.use_target_bf16_lm_head() {
            anyhow::ensure!(
                (1..=crate::layers::QWEN4_NATIVE_MAX_ROWS as u32).contains(&num_tokens),
                "BF16 Qwen4 batched-verify LM-head rows must be in 1..={}, got {num_tokens}",
                crate::layers::QWEN4_NATIVE_MAX_ROWS
            );
            let h = self.config.hidden_size as u32;
            let v = self.verify_lmhead_vocab();
            return self.lm_head_bf16_rows(hidden, num_tokens, v, h, stream);
        }
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let tc_dense_active = !det_batch.is_enabled()
            && crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                num_tokens as usize,
                false,
            );
        let Some(ref nvfp4) = self.lm_head_nvfp4 else {
            return self.lm_head_qwen4_rows(hidden, num_tokens, stream);
        };
        if !tc_dense_active {
            return self.lm_head_qwen4_rows(hidden, num_tokens, stream);
        }
        anyhow::ensure!(
            (1..=crate::layers::QWEN4_NATIVE_MAX_ROWS as u32).contains(&num_tokens),
            "NVFP4 Qwen4 batched-verify LM-head rows must be in 1..={}, got {num_tokens}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS
        );
        let h = self.config.hidden_size as u32;
        let v = self.verify_lmhead_vocab();
        let logits = self.buffers.logits();
        let ab_aligned16 = hidden.0.is_multiple_of(16)
            && nvfp4.weight.0.is_multiple_of(16)
            && nvfp4.weight_scale.0.is_multiple_of(2)
            && logits.0.is_multiple_of(2);
        let aligned16 = hidden.0.is_multiple_of(16)
            && nvfp4.weight.0.is_multiple_of(16)
            && logits.0.is_multiple_of(16);
        let has_tc_transposed = lm_head_tc_enabled()
            && self.lm_head_nvfp4_t.is_some()
            && self.w4a16_gemm_t_m32_n64_kernel.0 != 0;
        if !det_batch.is_enabled()
            && should_use_v2_coalesced_verify_lm_head(
                has_tc_transposed,
                num_tokens,
                atlas_core::device::active_sm_count(),
            )
            && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                self.gpu.as_ref(),
                hidden,
                nvfp4,
                logits,
                num_tokens as usize,
                v as usize,
                h as usize,
                stream,
            )?
        {
            let route_label = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                num_tokens as usize,
                v as usize,
                h as usize,
            );
            static LOGGED_V2: std::sync::Once = std::sync::Once::new();
            LOGGED_V2.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "lm_head",
                    route = route_label,
                    rows = num_tokens,
                    vocab = v,
                    hidden = h,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
            if self.logit_softcap_kernel.0 != 0 {
                let cap = self.config.final_logit_softcapping;
                let total = num_tokens * v;
                self.apply_logit_softcap(logits, total, cap, stream)?;
            }
            return Ok(logits);
        }
        let coalesced_m32_k =
            if !has_tc_transposed && num_tokens <= 32 && ab_aligned16 && h.is_multiple_of(32) {
                W4A16_GEMM_COALESCED_M32_KERNEL.get_or_try_load(
                    self.gpu.as_ref(),
                    "w4a16",
                    "w4a16_gemm_coalesced_m32",
                )
            } else {
                KernelHandle(0)
            };
        let coalesced_m64_k = if !has_tc_transposed
            && matches!(num_tokens, 33..=64)
            && ab_aligned16
            && h.is_multiple_of(32)
        {
            W4A16_GEMM_COALESCED_M64_KERNEL.get_or_try_load(
                self.gpu.as_ref(),
                "w4a16",
                "w4a16_gemm_coalesced_m64",
            )
        } else {
            KernelHandle(0)
        };
        let coalesced_m128_k =
            if !has_tc_transposed && num_tokens > 64 && ab_aligned16 && h.is_multiple_of(32) {
                W4A16_GEMM_COALESCED_M128_KERNEL.get_or_try_load(
                    self.gpu.as_ref(),
                    "w4a16",
                    "w4a16_gemm_coalesced_m128",
                )
            } else {
                KernelHandle(0)
            };
        let has_selected_coalesced =
            coalesced_m32_k.0 != 0 || coalesced_m64_k.0 != 0 || coalesced_m128_k.0 != 0;
        let pipe_m128_k = if !has_tc_transposed
            && !has_selected_coalesced
            && num_tokens > 64
            && aligned16
            && v.is_multiple_of(128)
            && h.is_multiple_of(32)
        {
            W4A16_GEMM_PIPE_M128N128_KERNEL.get_or_try_load(
                self.gpu.as_ref(),
                "w4a16",
                "w4a16_gemm_pipe_m128n128",
            )
        } else {
            KernelHandle(0)
        };
        let pipe_k = if !has_tc_transposed
            && !has_selected_coalesced
            && pipe_m128_k.0 == 0
            && aligned16
            && h.is_multiple_of(32)
        {
            W4A16_GEMM_PIPE_KERNEL.get_or_try_load(self.gpu.as_ref(), "w4a16", "w4a16_gemm_pipe")
        } else {
            KernelHandle(0)
        };
        let route = select_batched_verify_lm_head_route_with_policy(
            det_batch,
            tc_dense_active,
            num_tokens,
            v,
            h,
            ab_aligned16,
            aligned16,
            has_tc_transposed,
            coalesced_m32_k.0 != 0,
            coalesced_m64_k.0 != 0,
            coalesced_m128_k.0 != 0,
            pipe_m128_k.0 != 0,
            pipe_k.0 != 0,
            self.w4a16_gemm_kernel.0 != 0,
        );
        let route_label = match route {
            BatchedVerifyLmHeadRoute::Exact => {
                return self.lm_head_qwen4_rows(hidden, num_tokens, stream);
            }
            BatchedVerifyLmHeadRoute::TcTransposedM32N64 => {
                log_lm_head_tc_engagement(num_tokens, v, h);
                self.lm_head_tc_rows(hidden, logits, num_tokens, v, stream)?;
                "w4a16_gemm_t_m32_n64"
            }
            BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM32 => {
                launch_w4a16_gemm_coalesced(
                    self.gpu.as_ref(),
                    coalesced_m32_k,
                    32,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm_coalesced_m32"
            }
            BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM64 => {
                launch_w4a16_gemm_coalesced(
                    self.gpu.as_ref(),
                    coalesced_m64_k,
                    64,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm_coalesced_m64"
            }
            BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM128 => {
                launch_w4a16_gemm_coalesced(
                    self.gpu.as_ref(),
                    coalesced_m128_k,
                    128,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm_coalesced_m128"
            }
            BatchedVerifyLmHeadRoute::W4a16GemmPipeM128N128 => {
                ops::w4a16_gemm_pipe_m128n128(
                    self.gpu.as_ref(),
                    pipe_m128_k,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm_pipe_m128n128"
            }
            BatchedVerifyLmHeadRoute::W4a16GemmPipe => {
                ops::w4a16_gemm_pipe(
                    self.gpu.as_ref(),
                    pipe_k,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm_pipe"
            }
            BatchedVerifyLmHeadRoute::W4a16Gemm => {
                ops::w4a16_gemm(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_kernel,
                    hidden,
                    nvfp4,
                    logits,
                    num_tokens,
                    v,
                    h,
                    stream,
                )?;
                "w4a16_gemm"
            }
        };
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                component = "lm_head",
                route = route_label,
                rows = num_tokens,
                vocab = v,
                hidden = h,
                "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
            );
        });
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_fp8_gemm_coalesced(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tile_m: u32,
    input: DevicePtr,
    weight: &crate::weight_map::Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, tile_m), 1])
        .block([512, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_bf16_gemm_coalesced(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tile_m: u32,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, tile_m), 1])
        .block([512, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_w4a16_gemm_coalesced(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tile_m: u32,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, tile_m), 1])
        .block([512, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Whether `lm_head_fp8_rows` engages coalesced FP8 Tensor Core GEMMs (`M>=2`)
/// rather than per-row exact `dense_gemv_fp8w`. Under `ATLAS_DETERMINISTIC_BATCH=1`,
/// returns `false` across all `M=1..=128` so every row runs identical `dense_gemv_fp8w`.
#[inline]
#[must_use]
pub(crate) const fn select_target_fp8_lm_head_uses_coalesced_with_policy(
    num_tokens: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    !det_batch.is_enabled() && num_tokens >= 2
}

/// Whether `lm_head_bf16_rows` runs per-row `dense_gemv_bf16` instead of
/// `bf16_gemm_coalesced_m*` / `cublaslt_bf16_gemm`. Under `ATLAS_DETERMINISTIC_BATCH=1`,
/// returns `true` across all `M=1..=128` so every row matches `M=1` bit-for-bit.
#[inline]
#[must_use]
pub(crate) const fn select_target_bf16_lm_head_serial_gemv_with_policy(
    num_tokens: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    num_tokens == 1 || det_batch.is_enabled()
}

/// Whether the non-Qwen4 dense BF16 LM-head fallback (`lm_head_batched` /
/// `lm_head_qwen4_rows`) runs per-row `dense_gemv_bf16` instead of `dense_gemm`.
/// Under `ATLAS_DETERMINISTIC_BATCH=1`, returns `true` for all `M=1..=128`,
/// preventing the `M>=3` and `M>=33` `dense_gemm` flips.
#[inline]
#[must_use]
pub(crate) const fn select_dense_bf16_lm_head_serial_gemv_with_policy(
    num_tokens: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    num_tokens <= 2 || det_batch.is_enabled()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchedVerifyLmHeadRoute {
    Exact,
    TcTransposedM32N64,
    W4a16GemmCoalescedM32,
    W4a16GemmCoalescedM64,
    W4a16GemmCoalescedM128,
    W4a16GemmPipeM128N128,
    W4a16GemmPipe,
    W4a16Gemm,
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub(crate) const fn select_batched_verify_lm_head_route_with_policy(
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
    tc_dense_active: bool,
    num_tokens: u32,
    vocab: u32,
    hidden: u32,
    ab_aligned16: bool,
    aligned16: bool,
    has_tc_transposed: bool,
    has_coalesced_m32: bool,
    has_coalesced_m64: bool,
    has_coalesced_m128: bool,
    has_pipe_m128n128: bool,
    has_pipe: bool,
    has_gemm: bool,
) -> BatchedVerifyLmHeadRoute {
    if det_batch.is_enabled() || !tc_dense_active {
        return BatchedVerifyLmHeadRoute::Exact;
    }
    if has_tc_transposed {
        return BatchedVerifyLmHeadRoute::TcTransposedM32N64;
    }
    if ab_aligned16 && hidden.is_multiple_of(32) {
        if num_tokens <= 32 && has_coalesced_m32 {
            return BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM32;
        }
        if matches!(num_tokens, 33..=64) && has_coalesced_m64 {
            return BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM64;
        }
        if num_tokens > 64 && has_coalesced_m128 {
            return BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM128;
        }
    }
    if num_tokens > 64
        && has_pipe_m128n128
        && aligned16
        && vocab.is_multiple_of(128)
        && hidden.is_multiple_of(32)
    {
        return BatchedVerifyLmHeadRoute::W4a16GemmPipeM128N128;
    }
    if has_pipe && aligned16 && hidden.is_multiple_of(32) {
        return BatchedVerifyLmHeadRoute::W4a16GemmPipe;
    }
    if has_gemm && hidden.is_multiple_of(16) {
        return BatchedVerifyLmHeadRoute::W4a16Gemm;
    }
    BatchedVerifyLmHeadRoute::Exact
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
const fn select_batched_verify_lm_head_route(
    tc_dense_active: bool,
    num_tokens: u32,
    vocab: u32,
    hidden: u32,
    ab_aligned16: bool,
    aligned16: bool,
    has_tc_transposed: bool,
    has_coalesced_m32: bool,
    has_coalesced_m64: bool,
    has_coalesced_m128: bool,
    has_pipe_m128n128: bool,
    has_pipe: bool,
    has_gemm: bool,
) -> BatchedVerifyLmHeadRoute {
    select_batched_verify_lm_head_route_with_policy(
        crate::layers::deterministic_batch::DeterministicBatchPolicy::Disabled,
        tc_dense_active,
        num_tokens,
        vocab,
        hidden,
        ab_aligned16,
        aligned16,
        has_tc_transposed,
        has_coalesced_m32,
        has_coalesced_m64,
        has_coalesced_m128,
        has_pipe_m128n128,
        has_pipe,
        has_gemm,
    )
}

#[cfg(test)]
mod batched_verify_lm_head_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{Result, bail};
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

    use super::{
        BackendKernelCache, BatchedVerifyLmHeadRoute, select_batched_verify_lm_head_route,
    };
    use crate::layers::qwen4_hyper::{
        BatchedVerifyTcComponent, batched_verify_tc_component_admits,
    };

    struct CountingKernelGpu {
        handle: Option<KernelHandle>,
        lookups: AtomicUsize,
    }

    impl CountingKernelGpu {
        fn new(handle: Option<KernelHandle>) -> Self {
            Self {
                handle,
                lookups: AtomicUsize::new(0),
            }
        }

        fn lookups(&self) -> usize {
            self.lookups.load(Ordering::SeqCst)
        }
    }

    impl GpuBackend for CountingKernelGpu {
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn free(&self, _ptr: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn launch(
            &self,
            _func: KernelHandle,
            _grid: [u32; 3],
            _block: [u32; 3],
            _shared_mem: u32,
            _stream: u64,
            _params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn synchronize(&self, _stream: u64) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            match self.handle {
                Some(h) => Ok(h),
                None => bail!("missing {module}::{func_name}"),
            }
        }
        fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn memset_async(
            &self,
            _ptr: DevicePtr,
            _value: u8,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn total_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by kernel-cache tests")
        }
        fn free_memory(&self) -> Result<usize> {
            unimplemented!("not exercised by kernel-cache tests")
        }
    }

    #[test]
    fn batched_verify_lm_head_preserves_exact_below_4_and_selects_tc_above() {
        // Rows <= 3 stay strictly on Exact scalar GEMV (C=1 stays unscoped on strict 0-ULP).
        for rows in [1u32, 2, 3] {
            let active = batched_verify_tc_component_admits(
                BatchedVerifyTcComponent::LmHead,
                true,
                None,
                rows as usize,
                false,
            );
            assert_eq!(
                select_batched_verify_lm_head_route(
                    active, rows, 248_077, 2_560, true, true, false, true, true, true, true, true,
                    true,
                ),
                BatchedVerifyLmHeadRoute::Exact,
                "rows={rows}"
            );
        }
        // C=2 K=2..7 and C=4 K=5 (4..=32) select W4a16GemmCoalescedM32
        // for both odd logical vocab (248_077) and 128-aligned vocab (248_320), even
        // without 16B logits alignment (ab_aligned16=true, aligned16=false).
        for vocab in [248_077u32, 248_320] {
            for rows in [4u32, 5, 6, 7, 8, 10, 16, 20, 32] {
                let active = batched_verify_tc_component_admits(
                    BatchedVerifyTcComponent::LmHead,
                    true,
                    None,
                    rows as usize,
                    false,
                );
                assert_eq!(
                    select_batched_verify_lm_head_route(
                        active, rows, vocab, 2_560, true, false, false, true, true, true, true,
                        true, true,
                    ),
                    BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM32,
                    "vocab={vocab} rows={rows}"
                );
            }
            // C=8 K=5 (40) selects W4a16GemmCoalescedM64 (single 64-row coalesced sweep).
            for rows in [33u32, 40, 64] {
                let active = batched_verify_tc_component_admits(
                    BatchedVerifyTcComponent::LmHead,
                    true,
                    None,
                    rows as usize,
                    false,
                );
                assert_eq!(
                    select_batched_verify_lm_head_route(
                        active, rows, vocab, 2_560, true, false, false, true, true, true, true,
                        true, true,
                    ),
                    BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM64,
                    "vocab={vocab} rows={rows}"
                );
            }
            // C=16 K=5 (80) through 256 (C=32 K=8 / C=64 K=4) select
            // W4a16GemmCoalescedM128 (`grid.y = div_ceil(rows, 128)`).
            for rows in [65u32, 80, 128, 129, 192, 256] {
                let active = batched_verify_tc_component_admits(
                    BatchedVerifyTcComponent::LmHead,
                    true,
                    None,
                    rows as usize,
                    false,
                );
                assert_eq!(
                    select_batched_verify_lm_head_route(
                        active, rows, vocab, 2_560, true, false, false, true, true, true, true,
                        true, true,
                    ),
                    BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM128,
                    "vocab={vocab} rows={rows}"
                );
            }
        }
        // Explicit ATLAS_LM_HEAD_TC=1 with transposed head takes TcTransposedM32N64.
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_077, 2_560, true, true, true, true, true, true, true, true, true,
            ),
            BatchedVerifyLmHeadRoute::TcTransposedM32N64
        );
        // Missing coalesced kernels fall back cleanly through PipeM128N128 -> Pipe -> Gemm -> Exact.
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_320, 2_560, true, true, false, false, false, false, true, true, true,
            ),
            BatchedVerifyLmHeadRoute::W4a16GemmPipeM128N128
        );
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_077, 2_560, true, true, false, false, false, false, true, true, true,
            ),
            BatchedVerifyLmHeadRoute::W4a16GemmPipe
        );
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_320, 2_560, true, true, false, false, false, false, false, true,
                true,
            ),
            BatchedVerifyLmHeadRoute::W4a16GemmPipe
        );
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_320, 2_560, true, true, false, false, false, false, false, false,
                true,
            ),
            BatchedVerifyLmHeadRoute::W4a16Gemm
        );
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_320, 2_560, false, false, false, true, true, true, true, true, true,
            ),
            BatchedVerifyLmHeadRoute::W4a16Gemm
        );
        assert_eq!(
            select_batched_verify_lm_head_route(
                true, 80, 248_320, 2_560, true, true, false, false, false, false, false, false,
                false,
            ),
            BatchedVerifyLmHeadRoute::Exact
        );
    }

    #[test]
    fn backend_kernel_cache_isolates_distinct_backends_and_avoids_repeat_lookups() {
        let gpu_a = Box::new(CountingKernelGpu::new(Some(KernelHandle(0xA1))));
        let gpu_b = Box::new(CountingKernelGpu::new(Some(KernelHandle(0xB2))));
        let gpu_missing = Box::new(CountingKernelGpu::new(None));

        let cache = BackendKernelCache::new();
        assert_eq!(
            cache
                .get_or_load(gpu_a.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .unwrap()
                .0,
            0xA1
        );
        assert_eq!(gpu_a.lookups(), 1);
        assert_eq!(
            cache
                .get_or_load(gpu_a.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .unwrap()
                .0,
            0xA1
        );
        assert_eq!(
            gpu_a.lookups(),
            1,
            "primary backend repeat must not re-query driver"
        );

        assert_eq!(
            cache
                .get_or_load(gpu_b.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .unwrap()
                .0,
            0xB2
        );
        assert_eq!(gpu_b.lookups(), 1);
        assert_eq!(
            cache
                .get_or_try_load(gpu_b.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .0,
            0xB2
        );
        assert_eq!(
            gpu_b.lookups(),
            1,
            "fallback backend repeat must not re-query driver"
        );

        assert!(
            cache
                .get_or_load(gpu_missing.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .is_err()
        );
        assert_eq!(gpu_missing.lookups(), 1);
        assert_eq!(
            cache
                .get_or_try_load(gpu_missing.as_ref(), "w4a16", "w4a16_gemm_pipe")
                .0,
            0
        );
        assert_eq!(
            gpu_missing.lookups(),
            1,
            "cached error must not re-query driver"
        );
    }

    #[test]
    fn target_lm_head_mode_defaults_fp8_and_honors_mode_and_bf16_overrides() {
        use super::{Qwen4TargetLmHeadMode, parse_qwen4_target_lmhead_mode};
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, None, None, None),
            Qwen4TargetLmHeadMode::Fp8
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(Some("bf16"), None, None, None),
            Qwen4TargetLmHeadMode::Bf16
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(Some(" NVFP4 "), None, Some("1"), None),
            Qwen4TargetLmHeadMode::Nvfp4
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, Some("e4m3"), Some("0"), None),
            Qwen4TargetLmHeadMode::Fp8
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(Some("fp8"), Some("bf16"), None, None),
            Qwen4TargetLmHeadMode::Fp8
        );
        // Legacy ATLAS/FLUX_QWEN4_TARGET_LMHEAD_BF16 compatibility
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, None, Some("1"), None),
            Qwen4TargetLmHeadMode::Bf16
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, None, Some("0"), None),
            Qwen4TargetLmHeadMode::Nvfp4
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, None, None, Some("false")),
            Qwen4TargetLmHeadMode::Nvfp4
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(None, None, Some("1"), Some("0")),
            Qwen4TargetLmHeadMode::Bf16
        );
        assert_eq!(
            parse_qwen4_target_lmhead_mode(Some("bogus"), None, Some("bogus"), None),
            Qwen4TargetLmHeadMode::Fp8
        );
    }

    #[test]
    fn sm120_lm_head_v2_coalesced_and_m16_gates_preserve_gb10() {
        use super::{should_prefer_m16_coalesced_lm_head, should_use_v2_coalesced_verify_lm_head};

        // GB10 (<= 64 SMs): never diverts to v2 coalesced verify or m16 override.
        for gb10_sms in [48u32, 64u32] {
            for rows in [2u32, 8, 16, 64, 65, 80, 128, 256] {
                assert!(!should_use_v2_coalesced_verify_lm_head(
                    false, rows, gb10_sms
                ));
                assert!(!should_prefer_m16_coalesced_lm_head(rows, gb10_sms));
            }
        }
        // SM120 / SM103 (> 64 SMs): verify lm_head upgrades rows > 64 when not transposed TC;
        // batched lm_head prefers coalesced v2_n64_m16 for 2..=16 when loaded.
        for sm120_sms in [65u32, 148u32, 188u32] {
            assert!(!should_use_v2_coalesced_verify_lm_head(
                false, 64, sm120_sms
            ));
            assert!(should_use_v2_coalesced_verify_lm_head(false, 65, sm120_sms));
            assert!(should_use_v2_coalesced_verify_lm_head(
                false, 128, sm120_sms
            ));
            assert!(!should_use_v2_coalesced_verify_lm_head(
                true, 128, sm120_sms
            ));

            assert!(!should_prefer_m16_coalesced_lm_head(1, sm120_sms));
            assert!(should_prefer_m16_coalesced_lm_head(2, sm120_sms));
            assert!(should_prefer_m16_coalesced_lm_head(8, sm120_sms));
            assert!(should_prefer_m16_coalesced_lm_head(16, sm120_sms));
            assert!(!should_prefer_m16_coalesced_lm_head(17, sm120_sms));
            assert!(!should_prefer_m16_coalesced_lm_head(32, sm120_sms));
        }
    }

    #[test]
    fn sm120_batched_lm_head_auto_tc_gates_preserve_gb10_and_qwen4() {
        use super::{LmHeadTcEnvOverride, should_auto_tc_batched_lm_head};

        // GB10 (<= 64 SMs): never auto-engages TC regardless of row count.
        for sm in [48u32, 64u32] {
            for rows in [1u32, 2, 8, 16, 24, 32, 64, 128] {
                assert!(!should_auto_tc_batched_lm_head(
                    false,
                    LmHeadTcEnvOverride::Auto,
                    sm,
                    rows
                ));
            }
        }
        // Qwen4 stays on its dedicated Qwen4 LM-head dispatch.
        assert!(!should_auto_tc_batched_lm_head(
            true,
            LmHeadTcEnvOverride::Auto,
            188,
            64
        ));
        // Non-Qwen4 on SM120 (> 64 SMs): auto-engages for rows >= 2 even when unset,
        // honors explicit disable (`ATLAS_LM_HEAD_TC=0`), and leaves M=1 alone.
        assert!(!should_auto_tc_batched_lm_head(
            false,
            LmHeadTcEnvOverride::Auto,
            188,
            1
        ));
        for rows in [2u32, 8, 16, 24, 32, 64, 128] {
            assert!(should_auto_tc_batched_lm_head(
                false,
                LmHeadTcEnvOverride::Auto,
                188,
                rows
            ));
            assert!(should_auto_tc_batched_lm_head(
                false,
                LmHeadTcEnvOverride::Enabled,
                188,
                rows
            ));
            assert!(!should_auto_tc_batched_lm_head(
                false,
                LmHeadTcEnvOverride::Disabled,
                188,
                rows
            ));
        }
    }
}
