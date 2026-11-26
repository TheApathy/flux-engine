// SPDX-License-Identifier: AGPL-3.0-only

//! Launch sequence of the QSA-sparse prefill core (see `prefill_sparse`).

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::prefill_fp8::{self, Mode};
use super::prefill_fused_topk as fused_topk;
use super::prefill_sparse::{
    ATTN_CAP, ATTN_SMEM, ATTN_THREADS, ATTN_THREADS_G6, HEAD_DIM, KV_HEADS, PLAN_SORT_SMEM,
    SELECTOR, SparseChunk, SparsePlan, TILE_ROWS, fused_kernels, iq_norm_rope_fused_kernel,
    kernels, score_v2_kernels, scratch, topk_v2_variant, topk_variant, v2_selected,
};
use super::{COMPRESS_RATIO, INDEX_DIM, INDEX_HEADS, QUERY_HEADS, Qwen4QsaIndexer};
use crate::layer::AttnMetadataDev;
use crate::layers::{ops, qwen4_attn_prep};

/// SM-count threshold above which QSA prefill enables SM120 L2 / 4-warp tuning
/// (`<= 64` keeps 48-SM GB10 defaults untouched).
const QSA_PREFILL_WIDE_SM_THRESHOLD: u32 = 64;
/// Selector for overlapping QSA `iq_proj` + `iq_norm_rope` on a secondary stream
/// concurrently with main-stream `qkv_proj` + `kv_prep` (joining right before
/// `score_v2` / `score_gmax`).
/// - unset / `""`: auto-ON on `active_sm_count() > 64` (SM120/SM103); OFF on GB10 (`<= 64`).
/// - `"0"` / `"off"` / `"false"` / `"no"`: disabled.
/// - `"1"` / `"on"` / `"true"` / `"yes"`: enabled.
pub const QSA_IQ_OVERLAP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_IQ_OVERLAP";
const SCORE_RUN_BATCH_CAP_DEFAULT: usize = 2048;
/// Wide-SM (`> 64` SMs) score/top-k run batch cap (`4096` rows @ 16K = `64 MiB`,
/// fitting cleanly inside SM120's `96 MiB` L2 cache).
const SCORE_RUN_BATCH_CAP_WIDE_SM: usize = 4096;

#[inline]
#[must_use]
fn resolve_qsa_iq_overlap_enabled(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v)
            if v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off") =>
        {
            false
        }
        Some(v)
            if v == "1"
                || v.eq_ignore_ascii_case("true")
                || v.eq_ignore_ascii_case("yes")
                || v.eq_ignore_ascii_case("on") =>
        {
            true
        }
        _ => sm_count > QSA_PREFILL_WIDE_SM_THRESHOLD,
    }
}

#[must_use]
pub(crate) fn qsa_iq_overlap_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(QSA_IQ_OVERLAP_SELECTOR).ok());
    resolve_qsa_iq_overlap_enabled(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[inline]
#[must_use]
const fn score_run_batch_cap_for_sm(sm_count: u32) -> usize {
    if sm_count > QSA_PREFILL_WIDE_SM_THRESHOLD {
        SCORE_RUN_BATCH_CAP_WIDE_SM
    } else {
        SCORE_RUN_BATCH_CAP_DEFAULT
    }
}

#[derive(Clone, Copy, Debug)]
struct QsaIqOverlapStream {
    side_stream: u64,
    fork_event: u64,
    done_event: u64,
}

fn get_or_init_qsa_iq_overlap_runtime(
    gpu: &dyn GpuBackend,
    main_stream: u64,
) -> Result<Option<QsaIqOverlapStream>> {
    static RUNTIME: Mutex<Option<HashMap<usize, Option<QsaIqOverlapStream>>>> = Mutex::new(None);
    let key = (gpu as *const dyn GpuBackend as *const ()) as usize;
    {
        let guard = RUNTIME
            .lock()
            .expect("QSA IQ overlap runtime lock poisoned");
        if let Some(map) = guard.as_ref()
            && let Some(&cached) = map.get(&key)
        {
            return Ok(cached.filter(|rt| rt.side_stream != main_stream));
        }
    }
    let side_stream = gpu.create_stream()?;
    if side_stream == 0 {
        let mut guard = RUNTIME
            .lock()
            .expect("QSA IQ overlap runtime lock poisoned");
        guard.get_or_insert_with(HashMap::new).insert(key, None);
        return Ok(None);
    }
    // Give `side_stream` its own private cuBLASLt workspace so concurrent `iq_proj`
    // (`gemm_act_weight_t_typed`) never races with main-stream `qkv_proj` / `ik_proj`.
    if spark_runtime::cublaslt::register_stream_workspace(side_stream).is_err() {
        let mut guard = RUNTIME
            .lock()
            .expect("QSA IQ overlap runtime lock poisoned");
        guard.get_or_insert_with(HashMap::new).insert(key, None);
        return Ok(None);
    }
    let fork_event = gpu.create_event()?;
    let done_event = gpu.create_event()?;
    let created = (fork_event != 0 && done_event != 0).then_some(QsaIqOverlapStream {
        side_stream,
        fork_event,
        done_event,
    });
    let mut guard = RUNTIME
        .lock()
        .expect("QSA IQ overlap runtime lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = *map.entry(key).or_insert(created);
    Ok(entry.filter(|rt| rt.side_stream != main_stream))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingQsaIqOverlap {
    layer_idx: usize,
    hidden: DevicePtr,
    iq_base: DevicePtr,
    rows: usize,
    start: usize,
    main_stream: u64,
    side_stream: u64,
    done_event: u64,
}

thread_local! {
    static PENDING_QSA_IQ_OVERLAP: Cell<Option<PendingQsaIqOverlap>> = const { Cell::new(None) };
}

/// RAII guard for an in-flight side-stream QSA `iq_proj` + `iq_norm_rope` launch.
///
/// Guarantees that `done_event` is joined back onto `main_stream` before `hidden`
/// (`norm_output`) is overwritten by `o_proj` or on any early error / unwind path.
struct QsaIqOverlapGuard<'a> {
    gpu: &'a dyn GpuBackend,
    main_stream: u64,
    done_event: u64,
}

impl Drop for QsaIqOverlapGuard<'_> {
    fn drop(&mut self) {
        let unjoined = PENDING_QSA_IQ_OVERLAP.with(|cell| {
            let cur = cell.get()?;
            if cur.done_event == self.done_event && cur.main_stream == self.main_stream {
                cell.set(None);
                Some(cur)
            } else {
                None
            }
        });
        if let Some(pending) = unjoined
            && let Err(wait_err) = self
                .gpu
                .stream_wait_event(pending.main_stream, pending.done_event)
        {
            tracing::error!(
                main_stream = pending.main_stream,
                side_stream = pending.side_stream,
                done_event = pending.done_event,
                error = %format!("{wait_err:#}"),
                "QsaIqOverlapGuard failed to join side stream via event; falling back to synchronize"
            );
            if let Err(sync_err) = self.gpu.synchronize(pending.side_stream) {
                tracing::error!(
                    side_stream = pending.side_stream,
                    error = %format!("{sync_err:#}"),
                    "QsaIqOverlapGuard fallback side-stream synchronize also failed"
                );
            }
        }
    }
}

#[must_use]
fn ranges_disjoint(a: (u64, usize), b: (u64, usize)) -> bool {
    if a.1 == 0 || b.1 == 0 {
        return true;
    }
    let Some(a_end) = a.0.checked_add(a.1 as u64) else {
        return false;
    };
    let Some(b_end) = b.0.checked_add(b.1 as u64) else {
        return false;
    };
    a_end <= b.0 || b_end <= a.0
}

#[must_use]
fn iq_scratch_disjoint_from_qkv_and_hidden(
    iq_base: DevicePtr,
    iq_bytes: usize,
    hidden: DevicePtr,
    hidden_bytes: usize,
    qkv_output: DevicePtr,
    qkv_output_bytes: usize,
) -> bool {
    if iq_base.is_null() || iq_bytes == 0 || hidden.is_null() {
        return false;
    }
    let iq = (iq_base.0, iq_bytes);
    ranges_disjoint(iq, (hidden.0, hidden_bytes))
        && (qkv_output.is_null() || ranges_disjoint(iq, (qkv_output.0, qkv_output_bytes)))
}

struct SparseRunLayout {
    plan: SparsePlan,
    fused: Option<(
        &'static super::prefill_sparse::FusedKernels,
        fused_topk::FusedLaunch,
        usize,
    )>,
    score_v2: Option<(
        &'static super::prefill_sparse::ScoreV2Kernels,
        super::prefill_sparse::ScoreV2Mode,
    )>,
    v2: bool,
    sm_count: u32,
    run_batch: usize,
    run_gmax_off: usize,
    run_bytes: usize,
}

fn compute_sparse_run_layout(
    gpu: &dyn GpuBackend,
    rows: usize,
    start: usize,
) -> Result<SparseRunLayout> {
    let v2 = v2_selected()?;
    let fused = fused_kernels(gpu)?;
    let score_v2 = score_v2_kernels(gpu)?;
    let p = SparsePlan::for_chunk(rows, start, v2, fused.is_some())?;
    let fused = match fused {
        Some(fk) => fused_launch(fk, &p, score_v2.is_some())?,
        None => None,
    };
    let sm_count = atlas_core::device::active_sm_count();
    let batch_cap = score_run_batch_cap_for_sm(sm_count);
    let (run_batch, run_gmax_off, run_bytes) = if let Some((_, _, min_blocks)) = fused
        && rows > p.sparse_row0
        && fused_topk::batch_engages(start + p.sparse_row0, min_blocks)
    {
        let align = |x: usize| x.next_multiple_of(256);
        let cap: usize = 160 << 20;
        let avail = cap.saturating_sub(p.score_off + 512);
        let row_bytes = p.ld * 4 + p.ldg * 4;
        let max_batch = ((130 << 20).min(avail) / row_bytes).clamp(64, batch_cap) / 64 * 64;
        let sparse_rows = rows - p.sparse_row0;
        let launches = sparse_rows.div_ceil(max_batch).max(1);
        let batch = sparse_rows
            .div_ceil(launches)
            .next_multiple_of(64)
            .max(p.score_batch);
        let score_rows = sparse_rows.min(batch);
        let gmax_off = align(p.score_off + score_rows * p.ld * 4);
        let bytes = align(gmax_off + score_rows * p.ldg * 4);
        if bytes <= cap {
            (batch, gmax_off, bytes.max(p.bytes))
        } else {
            (p.score_batch, p.gmax_off, p.bytes)
        }
    } else if score_v2.is_some() && rows > p.sparse_row0 {
        let align = |x: usize| x.next_multiple_of(256);
        let cap: usize = if rows <= 4096 { 128 << 20 } else { 160 << 20 };
        let avail = cap.saturating_sub(p.score_off + 256);
        let max_batch = ((64 << 20).min(avail) / (p.ld * 4)).clamp(64, batch_cap) / 64 * 64;
        let sparse_rows = rows - p.sparse_row0;
        let launches = sparse_rows.div_ceil(max_batch).max(1);
        let batch = sparse_rows
            .div_ceil(launches)
            .next_multiple_of(64)
            .max(p.score_batch);
        let score_rows = sparse_rows.min(batch);
        let bytes = align(p.score_off + score_rows * p.ld * 4);
        if bytes <= cap {
            (batch, bytes, bytes.max(p.bytes))
        } else {
            (p.score_batch, p.gmax_off, p.bytes)
        }
    } else {
        (p.score_batch, p.gmax_off, p.bytes)
    };
    Ok(SparseRunLayout {
        plan: p,
        fused,
        score_v2,
        v2,
        sm_count,
        run_batch,
        run_gmax_off,
        run_bytes,
    })
}

struct PlanFusedB128KernelCache {
    primary: OnceLock<(usize, KernelHandle)>,
    fallback: OnceLock<Mutex<HashMap<usize, KernelHandle>>>,
}

impl PlanFusedB128KernelCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn resolve(&self, gpu: &dyn GpuBackend) -> KernelHandle {
        let key = (gpu as *const dyn GpuBackend as *const ()) as usize;
        if let Some(&(cached_key, handle)) = self.primary.get()
            && cached_key == key
        {
            return handle;
        }
        let lookup = || {
            crate::layers::try_kernel(
                gpu,
                "qwen4_qsa_prefill_attn_fp8",
                "qwen4_qsa_prefill_plan_fused_sort_desc_b128",
            )
        };
        if self.primary.get().is_none() {
            let handle = lookup();
            let _ = self.primary.set((key, handle));
            return handle;
        }
        let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map
            .lock()
            .expect("QSA plan_fused_b128 kernel cache lock poisoned");
        *guard.entry(key).or_insert_with(lookup)
    }
}

fn plan_fused_b128_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: PlanFusedB128KernelCache = PlanFusedB128KernelCache::new();
    CACHE.resolve(gpu)
}

#[inline]
#[must_use]
const fn plan_fused_use_b128_for_sm(sm_count: u32, has_b128_kernel: bool) -> bool {
    sm_count > QSA_PREFILL_WIDE_SM_THRESHOLD && has_b128_kernel
}

impl Qwen4QsaIndexer {
    /// Whether `ATLAS_QWEN4_PREFILL_QSA_IQ_OVERLAP` is active for the current GPU.
    #[must_use]
    pub(crate) fn prefill_iq_overlap_enabled() -> bool {
        qsa_iq_overlap_enabled()
    }

    /// Kick off QSA `iq_proj` + `iq_norm_rope` on a secondary stream ahead of
    /// main-stream `qkv_proj` + `kv_prep`, returning an RAII guard that ensures
    /// `done_event` is joined back onto `main_stream` on every exit path.
    ///
    /// Returns `Ok(None)` (falling back to the exact serial path) when disabled,
    /// under `dump_enabled()`, under `Crop::Tail`, when `iq_rows == 0`, or when
    /// scratch disjointness / side-stream initialization declines.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_launch_prefill_iq_overlap<'a>(
        &self,
        hidden: DevicePtr,
        hidden_bytes: usize,
        qkv_output: DevicePtr,
        qkv_output_bytes: usize,
        rows: usize,
        start: usize,
        attn_layer_idx: usize,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        meta: AttnMetadataDev,
        gpu: &'a dyn GpuBackend,
        main_stream: u64,
    ) -> Result<Option<impl Drop + use<'a>>> {
        if !qsa_iq_overlap_enabled()
            || super::prefill_sparse_dump::dump_enabled()
            || crate::layers::qwen4_prefill_moe::last_layer_crop::active_tail_attn_core().is_some()
        {
            return Ok(None);
        }
        if hidden.is_null()
            || meta.num_seqs != 1
            || meta.block_table.is_null()
            || meta.positions.is_null()
            || self.index_qk_proj.weight.is_null()
            || self.q_norm.weight.is_null()
        {
            return Ok(None);
        }
        // Never nest two overlapped launches on the same thread/event pair.
        if PENDING_QSA_IQ_OVERLAP.with(|cell| cell.get().is_some()) {
            return Ok(None);
        }
        let layout = compute_sparse_run_layout(gpu, rows, start)?;
        let p = layout.plan;
        let iq_row = INDEX_HEADS as usize * INDEX_DIM as usize;
        let iq_row0 = p.sparse_row0 & !127;
        let iq_rows = rows.saturating_sub(iq_row0);
        if iq_rows == 0 {
            return Ok(None);
        }
        let base = scratch(gpu, layout.run_bytes)?;
        let iq = base.offset(p.iq_off);
        let iq_bytes = rows.saturating_mul(iq_row).saturating_mul(2);
        if !iq_scratch_disjoint_from_qkv_and_hidden(
            iq,
            iq_bytes,
            hidden,
            hidden_bytes,
            qkv_output,
            qkv_output_bytes,
        ) {
            return Ok(None);
        }
        let Some(ov) = get_or_init_qsa_iq_overlap_runtime(gpu, main_stream)? else {
            return Ok(None);
        };
        // Pre-resolve kernels and initialize any lazy RoPE frequency table on
        // `main_stream` BEFORE `fork_event` so `ov.side_stream` sees complete data.
        let unit_yarn = (self.yarn_attention_factor - 1.0).abs() <= f32::EPSILON;
        let iq_fused = if (self.yarn_inv_freq.is_null() || unit_yarn)
            && rotary_dim == 64
            && INDEX_DIM == 128
        {
            iq_norm_rope_fused_kernel(gpu)?
        } else {
            None
        };
        let use_prep_rope = iq_fused.is_none()
            && qwen4_attn_prep::selected()?
            && (self.yarn_inv_freq.is_null() || unit_yarn);
        let freq = if iq_fused.is_some() || use_prep_rope {
            Some(if !self.yarn_inv_freq.is_null() {
                self.yarn_inv_freq
            } else {
                qwen4_attn_prep::freq_table(gpu, rotary_dim, rope_theta, main_stream)?
            })
        } else {
            None
        };

        gpu.record_event(ov.fork_event, main_stream)?;
        gpu.stream_wait_event(ov.side_stream, ov.fork_event)?;

        let iq_hidden = hidden.offset(iq_row0 * hidden_size as usize * 2);
        let iq_slice = iq.offset(iq_row0 * iq_row * 2);
        let iq_pos = meta.positions.offset(iq_row0 * 4);
        let launch_res: Result<()> = (|| {
            spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                iq_hidden.0,
                hidden_size,
                self.index_qk_proj.weight.0,
                iq_slice.0,
                iq_row as u32,
                iq_rows as u32,
                iq_row as u32,
                hidden_size,
                spark_runtime::cublaslt::GemmDtype::Bf16,
                spark_runtime::cublaslt::GemmDtype::Bf16,
                false,
                iq_rows as u32,
                ov.side_stream,
            )?;
            if let Some(iq_fused) = iq_fused {
                let freq = freq.expect("freq table pre-resolved for fused iq_norm_rope");
                KernelLaunch::new(gpu, iq_fused)
                    .grid([iq_rows as u32, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(iq_slice)
                    .arg_ptr(self.q_norm.weight)
                    .arg_ptr(iq_pos)
                    .arg_ptr(freq)
                    .arg_u32(iq_rows as u32)
                    .arg_f32(eps)
                    .launch(ov.side_stream)?;
            } else {
                ops::rms_norm(
                    gpu,
                    self.rms_norm,
                    iq_slice,
                    &self.q_norm,
                    iq_slice,
                    (iq_rows * INDEX_HEADS as usize) as u32,
                    INDEX_DIM,
                    eps,
                    ov.side_stream,
                )?;
                if let Some(freq) = freq {
                    qwen4_attn_prep::rope_with_freq(
                        gpu,
                        iq_slice,
                        iq_pos,
                        freq,
                        iq_rows,
                        INDEX_HEADS,
                        INDEX_DIM,
                        rotary_dim,
                        ov.side_stream,
                    )?;
                } else {
                    self.apply_rope_n(
                        iq_slice,
                        iq_pos,
                        iq_rows,
                        INDEX_HEADS,
                        rotary_dim,
                        rope_theta,
                        gpu,
                        ov.side_stream,
                    )?;
                }
            }
            Ok(())
        })();
        let record_res = gpu.record_event(ov.done_event, ov.side_stream);
        if let Err(err) = launch_res {
            if let Err(join_err) =
                record_res.and_then(|()| gpu.stream_wait_event(main_stream, ov.done_event))
            {
                tracing::error!(
                    main_stream,
                    side_stream = ov.side_stream,
                    done_event = ov.done_event,
                    error = %format!("{join_err:#}"),
                    "QSA IQ overlap error-path event join failed; synchronizing side stream"
                );
                if let Err(sync_err) = gpu.synchronize(ov.side_stream) {
                    tracing::error!(
                        side_stream = ov.side_stream,
                        error = %format!("{sync_err:#}"),
                        "QSA IQ overlap error-path side-stream synchronize also failed"
                    );
                }
            }
            return Err(err);
        }
        if let Err(rec_err) = record_res {
            tracing::error!(
                side_stream = ov.side_stream,
                done_event = ov.done_event,
                error = %format!("{rec_err:#}"),
                "QSA IQ overlap failed to record done_event; synchronizing side stream before returning"
            );
            if let Err(sync_err) = gpu.synchronize(ov.side_stream) {
                tracing::error!(
                    side_stream = ov.side_stream,
                    error = %format!("{sync_err:#}"),
                    "QSA IQ overlap side-stream synchronize also failed after record_event error"
                );
            }
            return Err(rec_err);
        }
        PENDING_QSA_IQ_OVERLAP.with(|cell| {
            cell.set(Some(PendingQsaIqOverlap {
                layer_idx: attn_layer_idx,
                hidden,
                iq_base: iq,
                rows,
                start,
                main_stream,
                side_stream: ov.side_stream,
                done_event: ov.done_event,
            }));
        });
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::warn!(
                "QSA_PREFILL_IQ_OVERLAP_ENGAGED selector={QSA_IQ_OVERLAP_SELECTOR} layer={attn_layer_idx} rows={rows} iq_rows={iq_rows} start={start} sm_count={}",
                layout.sm_count
            );
        });
        Ok(Some(QsaIqOverlapGuard {
            gpu,
            main_stream,
            done_event: ov.done_event,
        }))
    }

    /// Run the QSA-sparse attention core for one prefill chunk. The chunk's
    /// K/V and compressed index keys must already be in the paged caches.
    pub fn prefill_sparse_attention(
        &self,
        c: &SparseChunk,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            meta.num_seqs == 1 && !meta.block_table.is_null() && !meta.positions.is_null(),
            "{SELECTOR} requires C1 paged metadata"
        );
        let layer = c.attn_layer_idx;
        let kv_dtype = kv_cache.dtype_for_layer(layer);
        ensure!(
            matches!(kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4),
            "{SELECTOR} requires BF16 or NVFP4 KV (got {kv_dtype})"
        );
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(layer);
        ensure!(
            k_stride == kv_cache.v_block_stride_bytes_for_layer(layer),
            "{SELECTOR}: K/V page strides differ"
        );
        let page_size = kv_cache.block_size();
        let per_kv_head_page_bytes = match kv_dtype {
            KvCacheDtype::Bf16 => page_size * HEAD_DIM * 2,
            KvCacheDtype::Nvfp4 => {
                let elems = page_size * HEAD_DIM;
                elems / 2 + elems / 16
            }
            _ => unreachable!("dtype checked above"),
        };
        let local_kv_heads = if per_kv_head_page_bytes > 0 {
            k_stride / per_kv_head_page_bytes
        } else {
            0
        };
        ensure!(
            page_size >= COMPRESS_RATIO
                && page_size.is_multiple_of(COMPRESS_RATIO)
                && (local_kv_heads == 1 || local_kv_heads == KV_HEADS)
                && k_stride == local_kv_heads * per_kv_head_page_bytes,
            "{SELECTOR}: unexpected KV page geometry (dtype={kv_dtype}, k_stride={k_stride}, page_size={page_size})"
        );
        let layout = compute_sparse_run_layout(gpu, c.rows, c.start)?;
        let SparseRunLayout {
            plan: p,
            fused,
            score_v2,
            v2,
            sm_count,
            run_batch,
            run_gmax_off,
            run_bytes,
        } = layout;
        let k = kernels(gpu)?;
        let cache = self.cache(gpu, kv_cache)?;
        let base = scratch(gpu, run_bytes)?;
        let (iq, masks) = (base.offset(p.iq_off), base.offset(p.mask_off));
        let (entries, counts, scores) = (
            base.offset(p.entry_off),
            base.offset(p.count_off),
            base.offset(p.score_off),
        );
        let iq_row = INDEX_HEADS as usize * INDEX_DIM as usize;

        let mut pending_iq_overlap = PENDING_QSA_IQ_OVERLAP.with(|cell| {
            let cur = cell.get()?;
            if cur.layer_idx == layer
                && cur.hidden == c.hidden
                && cur.iq_base == iq
                && cur.rows == c.rows
                && cur.start == c.start
                && cur.main_stream == stream
            {
                Some(cur)
            } else {
                None
            }
        });
        if pending_iq_overlap.is_none() {
            // Defensive: drain any stale mismatched overlap before touching `iq`.
            if let Some(stale) = PENDING_QSA_IQ_OVERLAP.with(|cell| cell.take()) {
                gpu.stream_wait_event(stream, stale.done_event)?;
            }
            let iq_row0 = if super::prefill_sparse_dump::dump_enabled() {
                0
            } else {
                p.sparse_row0 & !127
            };
            let iq_rows = c.rows.saturating_sub(iq_row0);
            if iq_rows > 0 {
                let iq_hidden = c.hidden.offset(iq_row0 * c.hidden_size as usize * 2);
                let iq_slice = iq.offset(iq_row0 * iq_row * 2);
                let iq_pos = meta.positions.offset(iq_row0 * 4);
                // Under Layer-47 Crop::Tail attention crop, pin the cuBLASLt BF16 GEMM
                // heuristic to the uncropped chunk's `iq_rows` so the indexer queries
                // stay 0-ULP bit-identical to the full chunk.
                let ref_iq_rows =
                    crate::layers::qwen4_prefill_moe::last_layer_crop::active_tail_attn_core()
                        .map(|tail_ctx| {
                            let full_sparse_row0 = (super::prefill_sparse::DENSE_LAST_POS + 1)
                                .saturating_sub(tail_ctx.chunk_start)
                                .min(tail_ctx.chunk_rows);
                            let full_iq_row0 = full_sparse_row0 & !127;
                            tail_ctx.chunk_rows.saturating_sub(full_iq_row0)
                        })
                        .filter(|&ref_m| ref_m >= iq_rows);
                if let Some(ref_m) = ref_iq_rows {
                    spark_runtime::cublaslt::bf16_gemm_act_weight_t_ref_m(
                        iq_hidden.0,
                        self.index_qk_proj.weight.0,
                        iq_slice.0,
                        iq_rows as u32,
                        iq_row as u32,
                        c.hidden_size,
                        ref_m as u32,
                        stream,
                    )?;
                } else {
                    spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                        iq_hidden.0,
                        self.index_qk_proj.weight.0,
                        iq_slice.0,
                        iq_rows as u32,
                        iq_row as u32,
                        c.hidden_size,
                        stream,
                    )?;
                }
                let unit_yarn = (self.yarn_attention_factor - 1.0).abs() <= f32::EPSILON;
                if let Some(iq_fused) = iq_norm_rope_fused_kernel(gpu)?
                    && (self.yarn_inv_freq.is_null() || unit_yarn)
                    && c.rotary_dim == 64
                    && INDEX_DIM == 128
                {
                    let freq = if !self.yarn_inv_freq.is_null() {
                        self.yarn_inv_freq
                    } else {
                        qwen4_attn_prep::freq_table(gpu, c.rotary_dim, c.rope_theta, stream)?
                    };
                    KernelLaunch::new(gpu, iq_fused)
                        .grid([iq_rows as u32, 1, 1])
                        .block([128, 1, 1])
                        .arg_ptr(iq_slice)
                        .arg_ptr(self.q_norm.weight)
                        .arg_ptr(iq_pos)
                        .arg_ptr(freq)
                        .arg_u32(iq_rows as u32)
                        .arg_f32(c.eps)
                        .launch(stream)?;
                } else {
                    ops::rms_norm(
                        gpu,
                        self.rms_norm,
                        iq_slice,
                        &self.q_norm,
                        iq_slice,
                        (iq_rows * INDEX_HEADS as usize) as u32,
                        INDEX_DIM,
                        c.eps,
                        stream,
                    )?;
                    if qwen4_attn_prep::selected()? && (self.yarn_inv_freq.is_null() || unit_yarn) {
                        let freq = if !self.yarn_inv_freq.is_null() {
                            self.yarn_inv_freq
                        } else {
                            qwen4_attn_prep::freq_table(gpu, c.rotary_dim, c.rope_theta, stream)?
                        };
                        qwen4_attn_prep::rope_with_freq(
                            gpu,
                            iq_slice,
                            iq_pos,
                            freq,
                            iq_rows,
                            INDEX_HEADS,
                            INDEX_DIM,
                            c.rotary_dim,
                            stream,
                        )?;
                    } else {
                        self.apply_rope_n(
                            iq_slice,
                            iq_pos,
                            iq_rows,
                            INDEX_HEADS,
                            c.rotary_dim,
                            c.rope_theta,
                            gpu,
                            stream,
                        )?;
                    }
                }
            }
        }

        // Rows whose scores fit the staged kernel's shared memory use it; longer
        // rows (past 90112 tokens) select from global memory, same algorithm.
        let topk = |rows: usize, first: usize, sparse_row0: usize, mask: DevicePtr, n: usize| {
            let (kernel, smem) = match score_v2 {
                Some((sv2, _)) => topk_v2_variant(sv2, n),
                None => topk_variant(k, n),
            };
            KernelLaunch::new(gpu, kernel)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .shared_mem(smem)
                .arg_ptr(scores)
                .arg_ptr(mask)
                .arg_u32(rows as u32)
                .arg_u32(first as u32)
                .arg_u32(sparse_row0 as u32)
                .arg_u32(p.ld as u32)
                .arg_u32(p.mask_words as u32)
                .launch(stream)
        };
        if p.sparse_row0 > 0 {
            topk(p.sparse_row0, c.start, p.sparse_row0, masks, 0)?;
        }
        // Join overlapped `iq_proj` + `iq_norm_rope` right before the first `iq` consumer (`score`).
        if let Some(pending) = pending_iq_overlap.take() {
            PENDING_QSA_IQ_OVERLAP.with(|cell| {
                if cell.get() == Some(pending) {
                    cell.set(None);
                }
            });
            gpu.stream_wait_event(stream, pending.done_event)?;
        }
        for b0 in (p.sparse_row0..c.rows).step_by(run_batch) {
            let nb = run_batch.min(c.rows - b0);
            let first = c.start + b0;
            let n_max = (first + nb) / COMPRESS_RATIO;
            let blk_step = score_v2.map_or(64, |(_, mode)| mode.blocks_per_cta(n_max));
            if let Some((fk, g, min_blocks)) = fused
                && fused_topk::batch_engages(first, min_blocks)
            {
                let gmax = base.offset(run_gmax_off);
                KernelLaunch::new(gpu, fk.score_gmax)
                    .grid([nb.div_ceil(64) as u32, n_max.div_ceil(blk_step) as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(iq.offset(b0 * iq_row * 2))
                    .arg_ptr(cache.compressed_keys)
                    .arg_ptr(meta.block_table)
                    .arg_ptr(scores)
                    .arg_ptr(gmax)
                    .arg_u32(nb as u32)
                    .arg_u32(first as u32)
                    .arg_u32(p.ld as u32)
                    .arg_u32(p.ldg as u32)
                    .arg_u32(page_size as u32)
                    .launch(stream)?;
                KernelLaunch::new(gpu, fk.topk_fused)
                    .grid([nb as u32, 1, 1])
                    .block([256, 1, 1])
                    .shared_mem(g.dyn_smem as u32)
                    .arg_ptr(scores)
                    .arg_ptr(gmax)
                    .arg_ptr(masks.offset(b0 * p.mask_words * 4))
                    .arg_u32(nb as u32)
                    .arg_u32(first as u32)
                    .arg_u32(0)
                    .arg_u32(p.ld as u32)
                    .arg_u32(p.ldg as u32)
                    .arg_u32(p.mask_words as u32)
                    .arg_u32(g.a_words as u32)
                    .arg_u32(g.cap_groups as u32)
                    .launch(stream)?;
                static ENGAGED: std::sync::Once = std::sync::Once::new();
                ENGAGED.call_once(|| {
                    tracing::warn!(
                        "QWEN4_PREFILL_QSA_FUSED_TOPK_ENGAGED selector={} phase=1 select=group_max_superset_c3 group={} min_blocks={min_blocks} start={} rows={} first={first} batch={nb} ld={} a_words={} cap_groups={} smem={} exact=same_masks",
                        fused_topk::SELECTOR, fused_topk::GROUP, c.start, c.rows, p.ld, g.a_words,
                        g.cap_groups, g.dyn_smem);
                    tracing::info!(
                        "QSA_FUSED_RADIX_ENGAGED selector={}",
                        fused_topk::active_selector()
                    );
                });
                continue;
            }
            let score_kernel = score_v2.map_or(k.score, |(sv2, _)| sv2.score);
            KernelLaunch::new(gpu, score_kernel)
                .grid([nb.div_ceil(64) as u32, n_max.div_ceil(blk_step) as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(iq.offset(b0 * iq_row * 2))
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.block_table)
                .arg_ptr(scores)
                .arg_u32(nb as u32)
                .arg_u32(first as u32)
                .arg_u32(p.ld as u32)
                .arg_u32(page_size as u32)
                .launch(stream)?;
            topk(nb, first, 0, masks.offset(b0 * p.mask_words * 4), n_max)?;
        }
        let local_query_heads = if c.num_query_heads > 0 {
            c.num_query_heads
        } else {
            local_kv_heads * (QUERY_HEADS / KV_HEADS)
        };
        let gqa_group = local_query_heads / local_kv_heads;
        ensure!(
            local_query_heads.is_multiple_of(local_kv_heads) && (gqa_group == 12 || gqa_group == 6),
            "{SELECTOR}: unsupported local GQA ratio {local_query_heads}/{local_kv_heads} (expected 12 or 6)"
        );

        // Consume `e4m3_out_armed` once for E4M3 chunks so we can decide before
        // the plan launch whether the SM120 TILE_ROWS=4 / kBc=64 register-Q core
        // supersedes both the T=10 plan and the e4m3_2cta attention launch.
        let e4m3_out_armed = if c.e4m3 && v2 {
            prefill_fp8::take_e4m3_out_armed()
        } else {
            false
        };
        let t4_tiles = c.rows.div_ceil(super::QSA_PREFILL_T4_TILE_ROWS);
        let t4_counts_off =
            (p.entry_off + t4_tiles * super::QSA_PREFILL_T4_CAP * 4).next_multiple_of(256);
        let t4_fits_scratch = t4_counts_off + t4_tiles * 4 <= run_bytes;
        let kbc64_t4_plan = if c.e4m3
            && v2
            && gqa_group == 12
            && local_kv_heads == KV_HEADS
            && kv_cache.block_size().is_power_of_two()
            && t4_fits_scratch
            && !super::prefill_sparse_dump::stats_or_dump_enabled()
            && super::qsa_prefill_kbc64_enabled()
        {
            super::qsa_prefill_kbc64_t4_kernels(gpu).and_then(|kbc64| {
                let configured_mode = prefill_fp8::mode().ok().flatten()?;
                let tail_core =
                    crate::layers::qwen4_prefill_moe::last_layer_crop::active_tail_attn_core();
                let mode = prefill_fp8::effective_chunk_mode(
                    configured_mode,
                    tail_core.map_or(c.start, |t| t.chunk_start),
                );
                let pipe = prefill_fp8::pipe().ok()?;
                let wdec_active = prefill_fp8::warp_decouple().ok()?.is_some();
                let sub_cta_active = prefill_fp8::sub_cta().ok()?.is_some();
                let e4m3_out_selected = prefill_fp8::e4m3_out().ok()?;
                let diagnostics_active = if e4m3_out_armed && e4m3_out_selected {
                    prefill_fp8::e4m3_out_diagnostics_active().ok()?
                } else {
                    false
                };
                if prefill_fp8::e4m3_out_eligible(prefill_fp8::E4m3OutEligibility {
                    selector_enabled: e4m3_out_selected,
                    armed_for_fp8_oproj: e4m3_out_armed,
                    mode,
                    pipe,
                    wdec_active,
                    sub_cta_active,
                    rows: tail_core.map_or(c.rows, |t| t.chunk_rows),
                    hidden_size: c.hidden_size as usize,
                    diagnostics_active,
                }) {
                    Some((kbc64, base.offset(t4_counts_off)))
                } else {
                    None
                }
            })
        } else {
            None
        };

        if let Some((kbc64, t4_counts)) = kbc64_t4_plan {
            super::launch_qsa_prefill_plan_t4_b128(
                gpu,
                kbc64.plan_t4_b128,
                masks,
                entries,
                t4_counts,
                c.rows,
                c.start,
                p.mask_words,
                super::QSA_PREFILL_T4_CAP,
                stream,
            )?;
        } else if c.e4m3 && v2 && prefill_fp8::plan_fused()? {
            let b128_k = if sm_count > QSA_PREFILL_WIDE_SM_THRESHOLD {
                plan_fused_b128_kernel(gpu)
            } else {
                KernelHandle(0)
            };
            let (plan_k, block_threads) = if plan_fused_use_b128_for_sm(sm_count, b128_k.0 != 0) {
                (b128_k, 128u32)
            } else {
                (prefill_fp8::plan_fused_kernel(gpu)?, 32u32)
            };
            KernelLaunch::new(gpu, plan_k)
                .grid([p.tiles as u32, 1, 1])
                .block([block_threads, 1, 1])
                .shared_mem(PLAN_SORT_SMEM as u32)
                .arg_ptr(masks)
                .arg_ptr(entries)
                .arg_ptr(counts)
                .arg_u32(c.rows as u32)
                .arg_u32(c.start as u32)
                .arg_u32(p.mask_words as u32)
                .arg_u32(TILE_ROWS as u32)
                .arg_u32(ATTN_CAP as u32)
                .arg_u32(p.tiles as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, k.plan)
                .grid([p.tiles.div_ceil(4) as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(masks)
                .arg_ptr(entries)
                .arg_ptr(counts)
                .arg_u32(c.rows as u32)
                .arg_u32(c.start as u32)
                .arg_u32(p.mask_words as u32)
                .arg_u32(TILE_ROWS as u32)
                .arg_u32(ATTN_CAP as u32)
                .arg_u32(p.tiles as u32)
                .launch(stream)?;
            if v2 {
                // Group each tile's entries by need mask so warps skip whole tiles. The E4M3
                // core takes the same groups with blocks descending (newest first, sink last).
                let sort = if c.e4m3 {
                    prefill_fp8::kernels(gpu)?.plan_sort_desc
                } else {
                    k.plan_sort
                };
                KernelLaunch::new(gpu, sort)
                    .grid([p.tiles as u32, 1, 1])
                    .block([32, 1, 1])
                    .shared_mem(PLAN_SORT_SMEM as u32)
                    .arg_ptr(entries)
                    .arg_ptr(counts)
                    .arg_u32(c.start as u32)
                    .arg_u32(TILE_ROWS as u32)
                    .arg_u32(ATTN_CAP as u32)
                    .launch(stream)?;
            }
        }
        let use_g6 = gqa_group == 6;
        let attn_threads = if use_g6 {
            ATTN_THREADS_G6
        } else {
            ATTN_THREADS
        };
        let row = (local_query_heads * HEAD_DIM) as u32;
        if c.e4m3 {
            ensure!(
                v2,
                "{}: requires {}=1",
                prefill_fp8::SELECTOR,
                super::prefill_sparse::V2_SELECTOR
            );
            let (core_counts, kbc64_kernel) = match kbc64_t4_plan {
                Some((kbc64, t4_counts)) => (t4_counts, Some(kbc64.attn_kbc64_t4)),
                None => (counts, None),
            };
            self.launch_e4m3_core(
                c,
                kv_cache,
                meta,
                entries,
                core_counts,
                p.tiles,
                e4m3_out_armed,
                kbc64_kernel,
                gpu,
                stream,
            )?;
            return super::prefill_sparse_dump::maybe_dump(
                self, c, &p, base, kv_cache, meta, gpu, stream,
            );
        }
        if kv_dtype == KvCacheDtype::Nvfp4 {
            let nvfp4_kernel = match (use_g6, v2) {
                (true, true) => k.attn_skip_nvfp4_g6,
                (true, false) => k.attn_nvfp4_g6,
                (false, true) => k.attn_skip_nvfp4,
                (false, false) => k.attn_nvfp4,
            };
            ensure!(
                nvfp4_kernel.0 != 0,
                "{SELECTOR}: NVFP4 QSA prefill attention kernel unavailable (gqa_group={gqa_group})"
            );
            KernelLaunch::new(gpu, nvfp4_kernel)
                .grid([p.tiles as u32, local_kv_heads as u32, 1])
                .block([attn_threads as u32, 1, 1])
                .shared_mem(ATTN_SMEM as u32)
                .arg_ptr(c.q)
                .arg_ptr(kv_cache.k_pool_ptr(layer))
                .arg_ptr(kv_cache.v_pool_ptr(layer))
                .arg_ptr(meta.block_table)
                .arg_ptr(entries)
                .arg_ptr(counts)
                .arg_ptr(c.out)
                .arg_ptr(c.gate)
                .arg_u32(c.rows as u32)
                .arg_u32(c.start as u32)
                .arg_u32(c.q_stride)
                .arg_u32(row)
                .arg_u32(c.gate_stride)
                .arg_u32(page_size as u32)
                .arg_u64(k_stride as u64)
                .arg_u64(kv_cache.nvfp4_data_bytes() as u64)
                .arg_u32((local_kv_heads * HEAD_DIM) as u32)
                .arg_f32(c.softmax_scale * std::f32::consts::LOG2_E)
                .launch(stream)?;
        } else {
            let bf16_kernel = match (use_g6, v2) {
                (true, true) => k.attn_skip_g6,
                (true, false) => k.attn_g6,
                (false, true) => k.attn_skip,
                (false, false) => k.attn,
            };
            ensure!(
                bf16_kernel.0 != 0,
                "{SELECTOR}: BF16 QSA prefill attention kernel unavailable (gqa_group={gqa_group})"
            );
            KernelLaunch::new(gpu, bf16_kernel)
                .grid([p.tiles as u32, local_kv_heads as u32, 1])
                .block([attn_threads as u32, 1, 1])
                .shared_mem(ATTN_SMEM as u32)
                .arg_ptr(c.q)
                .arg_ptr(kv_cache.k_pool_ptr(layer))
                .arg_ptr(kv_cache.v_pool_ptr(layer))
                .arg_ptr(meta.block_table)
                .arg_ptr(entries)
                .arg_ptr(counts)
                .arg_ptr(c.out)
                .arg_ptr(c.gate)
                .arg_u32(c.rows as u32)
                .arg_u32(c.start as u32)
                .arg_u32(c.q_stride)
                .arg_u32(row)
                .arg_u32(c.gate_stride)
                .arg_u32(page_size as u32)
                .arg_u64((k_stride / 2) as u64)
                .arg_u32((local_kv_heads * HEAD_DIM) as u32)
                .arg_f32(c.softmax_scale * std::f32::consts::LOG2_E)
                .launch(stream)?;
        }

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| tracing::warn!(
            "QWEN4_PREFILL_QSA_ENGAGED selector={SELECTOR} start={} rows={} tile_rows={TILE_ROWS} kv_dtype={kv_dtype} core=qsa_sparse_non_bit_exact v2={v2}",
            c.start, c.rows));
        // The one-time engagement message above proves that the BF16 sparse
        // core ran, but it cannot prove which later request reached its final
        // partial chunk.  Emit a stable endpoint receipt for partial chunks so
        // long-context qualification can prove both the retrieval and native
        // boundary requests without depending on process-global log order.
        let max_chunk_rows = crate::layers::qwen4_prefill_moe::max_chunk_rows();
        if c.rows < max_chunk_rows {
            let core_name = if kv_dtype == KvCacheDtype::Nvfp4 {
                "qsa_sparse_nvfp4_non_bit_exact"
            } else {
                "qsa_sparse_bf16_non_bit_exact"
            };
            tracing::debug!(
                "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector={SELECTOR} start={} rows={} end={} chunk_rows={max_chunk_rows} tile_rows={TILE_ROWS} core={core_name}",
                c.start,
                c.rows,
                c.start + c.rows
            );
        }
        super::prefill_sparse_dump::maybe_dump(self, c, &p, base, kv_cache, meta, gpu, stream)
    }

    /// The E4M3 core (`ATLAS_QWEN4_PREFILL_QSA_FP8`) on the chunk's plan: Q8 rows
    /// from the prep scratch, K8 (and with `qkpv` V8) from the layer's shadow
    /// pages; `qk` reads P*V's V from the BF16 cache. Same grid, plan and gate
    /// epilogue as `qwen4_qsa_prefill_attn_t10_skip`.
    #[allow(clippy::too_many_arguments)]
    fn launch_e4m3_core(
        &self,
        c: &SparseChunk,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        entries: DevicePtr,
        counts: DevicePtr,
        tiles: usize,
        e4m3_out_armed: bool,
        kbc64_kernel: Option<KernelHandle>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let configured_mode = prefill_fp8::mode()?.ok_or_else(|| {
            anyhow::anyhow!("{}: E4M3 chunk without a mode", prefill_fp8::SELECTOR)
        })?;
        let _ = prefill_fp8::v8_shared()?;
        // Under Layer-47 Crop::Tail attention crop, full-chunk KV prep has
        // already populated `k8` and (for `chunk_start == 0` under shared V8)
        // `v8` across `0..chunk_start + chunk_rows`, so check shared-V8 mode
        // validity and `E4M3_OUT_MIN_ROWS` eligibility against the uncropped
        // chunk's `(chunk_start, chunk_rows)` rather than the cropped tail slice.
        let tail_core = crate::layers::qwen4_prefill_moe::last_layer_crop::active_tail_attn_core();
        let mode_start = tail_core.map_or(c.start, |t| t.chunk_start);
        let mode = prefill_fp8::effective_chunk_mode(configured_mode, mode_start);
        if configured_mode == Mode::QkPv && mode == Mode::Qk {
            static SHARED_V8_FALLBACK: std::sync::Once = std::sync::Once::new();
            SHARED_V8_FALLBACK.call_once(|| {
                tracing::info!(
                    "QWEN4_PREFILL_QSA_FP8 shared_v8 continuation fallback start={} rows={} layer={} mode=qk",
                    c.start,
                    c.rows,
                    c.attn_layer_idx
                );
            });
        }
        let f = prefill_fp8::kernels(gpu)?;
        let layer = c.attn_layer_idx;
        let sc = prefill_fp8::scales(layer)?;
        let q8 = prefill_fp8::q8_scratch(gpu, c.rows)?;
        let (k8, v8) = prefill_fp8::shadow(gpu, layer, kv_cache)?;
        let page8 = u64::try_from(prefill_fp8::shadow_block_bytes(kv_cache)?)?;
        let tok8 = (KV_HEADS * HEAD_DIM) as u32;
        let (kernel, v_pool, v_page, v_tok) = match mode {
            Mode::Qk => (
                f.attn_qk,
                kv_cache.v_pool_ptr(layer),
                kv_cache.v_block_stride_bytes_for_layer(layer) as u64,
                2 * tok8,
            ),
            Mode::QkPv => (f.attn_qkpv, v8, page8, tok8),
        };
        let wdec_cfg = prefill_fp8::warp_decouple()?;
        if wdec_cfg.is_some() {
            ensure!(
                configured_mode == Mode::QkPv,
                "{}: requires {}=qkpv",
                prefill_fp8::WARP_DECOUPLE_SELECTOR,
                prefill_fp8::SELECTOR
            );
        }
        let sub_cta_cfg = prefill_fp8::sub_cta()?;
        if sub_cta_cfg.is_some() {
            ensure!(
                configured_mode == Mode::QkPv && wdec_cfg.is_none(),
                "{}: requires {}=qkpv and {}=0",
                prefill_fp8::SUB_CTA_SELECTOR,
                prefill_fp8::SELECTOR,
                prefill_fp8::WARP_DECOUPLE_SELECTOR
            );
        }
        let pipe = prefill_fp8::pipe()?;
        if pipe {
            ensure!(
                wdec_cfg.is_none() && sub_cta_cfg.is_none(),
                "{}: requires {}=0 and {}=0",
                prefill_fp8::PIPE_SELECTOR,
                prefill_fp8::WARP_DECOUPLE_SELECTOR,
                prefill_fp8::SUB_CTA_SELECTOR
            );
        }
        let scale_log2 =
            |h: usize| c.softmax_scale * std::f32::consts::LOG2_E / (sc.q[h] * sc.k[h]);
        let pf = prefill_fp8::plan_fused().unwrap_or(false);
        let v8_shared = prefill_fp8::v8_shared_enabled();
        let e4m3_out_selected = prefill_fp8::e4m3_out()?;
        let diagnostics_active = if e4m3_out_armed && e4m3_out_selected {
            prefill_fp8::e4m3_out_diagnostics_active()?
        } else {
            false
        };
        if prefill_fp8::e4m3_out_eligible(prefill_fp8::E4m3OutEligibility {
            selector_enabled: e4m3_out_selected,
            armed_for_fp8_oproj: e4m3_out_armed,
            mode,
            pipe,
            wdec_active: wdec_cfg.is_some(),
            sub_cta_active: sub_cta_cfg.is_some(),
            rows: tail_core.map_or(c.rows, |t| t.chunk_rows),
            hidden_size: c.hidden_size as usize,
            diagnostics_active,
        }) {
            // Consume any concurrent o_proj amax arm so prefill_fast_attn does not
            // log OPROJ_AMAX_FUSED_REFUSED when direct E4M3 output supersedes it.
            let _ = crate::layers::qwen4_fp8_act_amax::take_armed();
            crate::layers::qwen4_fp8_act_amax::clear();
            let e4m3_scale = prefill_fp8::e4m3_out_scale_for_v_prescales(sc.v)?;
            let out_cols = prefill_fp8::E4M3_OUT_COLS;
            let (_, ws, a8, _) = crate::layers::qwen4_fast_proj::fp8_scratch(
                gpu,
                c.hidden_size as usize,
                c.rows,
                out_cols,
            )?;
            let a_scale_ptr = ws.offset(12);
            let use_2cta = prefill_fp8::e4m3_2cta()? && kv_cache.block_size().is_power_of_two();
            if let Some(kbc64_k) = kbc64_kernel {
                super::launch_qsa_prefill_attn_kbc64_t4(
                    gpu,
                    kbc64_k,
                    &super::QsaPrefillKbc64T4Args {
                        q8,
                        kc8: k8,
                        vc8: v8,
                        block_table: meta.block_table,
                        entries,
                        counts,
                        out_e4m3: a8,
                        gate: c.gate,
                        rows: c.rows as u32,
                        first_pos: c.start as u32,
                        q_stride: out_cols as u32,
                        o_stride: out_cols as u32,
                        gate_stride: c.gate_stride,
                        page_size: kv_cache.block_size() as u32,
                        k_page_stride: page8,
                        k_tok_stride: tok8,
                        v_page_stride: page8,
                        v_tok_stride: tok8,
                        scale_log2_h0: scale_log2(0),
                        scale_log2_h1: scale_log2(1),
                        out_scale_h0: e4m3_scale.head_out_scale[0],
                        out_scale_h1: e4m3_scale.head_out_scale[1],
                        a_scale_out: a_scale_ptr,
                        a_scale_val: e4m3_scale.a_scale,
                        local_kv_heads: KV_HEADS as u32,
                        cap: super::QSA_PREFILL_T4_CAP as u32,
                    },
                    stream,
                )?;
            } else {
                let (e4m3_kernel, grid_y, dyn_smem) = if use_2cta {
                    (
                        prefill_fp8::pipe_e4m3_2cta_kernel(gpu)?,
                        (KV_HEADS as u32) * prefill_fp8::E4M3_2CTA_SUB_FACTOR,
                        prefill_fp8::E4M3_2CTA_SMEM as u32,
                    )
                } else {
                    (
                        prefill_fp8::pipe_e4m3_kernel(gpu)?,
                        KV_HEADS as u32,
                        mode.smem() as u32,
                    )
                };
                KernelLaunch::new(gpu, e4m3_kernel)
                    .grid([tiles as u32, grid_y, 1])
                    .block([ATTN_THREADS as u32, 1, 1])
                    .shared_mem(dyn_smem)
                    .arg_ptr(q8)
                    .arg_ptr(k8)
                    .arg_ptr(v8)
                    .arg_ptr(meta.block_table)
                    .arg_ptr(entries)
                    .arg_ptr(counts)
                    .arg_ptr(a8)
                    .arg_ptr(c.gate)
                    .arg_u32(c.rows as u32)
                    .arg_u32(c.start as u32)
                    .arg_u32(out_cols as u32)
                    .arg_u32(out_cols as u32)
                    .arg_u32(c.gate_stride)
                    .arg_u32(kv_cache.block_size() as u32)
                    .arg_u64(page8)
                    .arg_u32(tok8)
                    .arg_u64(page8)
                    .arg_u32(tok8)
                    .arg_f32(scale_log2(0))
                    .arg_f32(scale_log2(1))
                    .arg_f32(e4m3_scale.head_out_scale[0])
                    .arg_f32(e4m3_scale.head_out_scale[1])
                    .arg_ptr(a_scale_ptr)
                    .arg_f32(e4m3_scale.a_scale)
                    .launch(stream)?;
            }
            // Ordering fence between `a_scale` store (`ws + 12`) and cuBLASLt:
            // `fp8_gemm_ex` passes `fence_scale = !prequant = false`, so mirror
            // `fp8_weight_tensor_gemm`'s 4-byte non-kernel memset on unused `ws`.
            gpu.memset_async(ws, 0, 4, stream)?;
            crate::layers::qwen4_fast_proj::note_prequant_fp8_act(c.out, c.rows, out_cols, stream);
            let kbc64_t4 = kbc64_kernel.is_some();
            let e4m3_2cta_active = use_2cta && !kbc64_t4;
            let eff_tile_rows = if kbc64_t4 {
                super::QSA_PREFILL_T4_TILE_ROWS
            } else {
                TILE_ROWS
            };
            static E4M3_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            E4M3_OUT_LOGGED.call_once(|| {
                tracing::info!(
                    "QSA_PREFILL_E4M3_OUT_ENGAGED selector={} start={} rows={} layer={} bound={} inv_scale={} a_scale={} v_prescale={:?} pipe={pipe} e4m3_2cta={e4m3_2cta_active} kbc64_t4={kbc64_t4} plan_fused={pf} v8_shared={v8_shared}",
                    prefill_fp8::E4M3_OUT_SELECTOR,
                    c.start,
                    c.rows,
                    layer,
                    e4m3_scale.bound,
                    e4m3_scale.inv_scale,
                    e4m3_scale.a_scale,
                    sc.v
                );
            });
            log_e4m3_core_engaged(
                c,
                mode,
                None,
                None,
                pipe,
                pf,
                v8_shared,
                eff_tile_rows,
                layer,
                sc,
            );
            return Ok(());
        }
        // ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED: when the call site armed the hand-off (its o_proj
        // is a tensor-scale FP8 projection), the `_amax` twin also reduces the absmax of the gated
        // output it stores into the zeroed slot, and the o_proj skips its own absmax pass.
        use crate::layers::qwen4_fp8_act_amax as act_amax;
        let amax_slot = if act_amax::oproj_selected()? && act_amax::take_armed() {
            let slot = act_amax::slot(gpu)?;
            gpu.memset_async(slot, 0, 4, stream)?;
            Some(slot)
        } else {
            None
        };
        let wdec = if mode == Mode::QkPv { wdec_cfg } else { None };
        let (kernel, dyn_smem) = match (wdec, amax_slot, mode) {
            (None, None, _) => (kernel, mode.smem() as u32),
            (None, Some(_), Mode::Qk) => {
                (prefill_fp8::amax_kernels(gpu)?.attn_qk, mode.smem() as u32)
            }
            (None, Some(_), Mode::QkPv) => (
                prefill_fp8::amax_kernels(gpu)?.attn_qkpv,
                mode.smem() as u32,
            ),
            (Some(wd @ prefill_fp8::WarpDecouple::Wdec4k), None, _) => {
                (prefill_fp8::wdec_kernels(gpu)?.qkpv_4k, wd.smem() as u32)
            }
            (Some(wd @ prefill_fp8::WarpDecouple::Wdec4k), Some(_), _) => (
                prefill_fp8::wdec_kernels(gpu)?.qkpv_amax_4k,
                wd.smem() as u32,
            ),
            (Some(wd @ prefill_fp8::WarpDecouple::Wdec8k), None, _) => {
                (prefill_fp8::wdec_kernels(gpu)?.qkpv_8k, wd.smem() as u32)
            }
            (Some(wd @ prefill_fp8::WarpDecouple::Wdec8k), Some(_), _) => (
                prefill_fp8::wdec_kernels(gpu)?.qkpv_amax_8k,
                wd.smem() as u32,
            ),
        };
        let sub_cta = if mode == Mode::QkPv {
            sub_cta_cfg
        } else {
            None
        };
        let (kernel, dyn_smem, sub_factor, block_threads) = match (sub_cta, amax_slot) {
            (None, _) => (kernel, dyn_smem, 1u32, ATTN_THREADS as u32),
            (Some(sub @ prefill_fp8::SubCta::Half), None) => (
                prefill_fp8::sub_kernels(gpu)?.qkpv_half,
                sub.smem() as u32,
                sub.sub_factor(),
                sub.threads(),
            ),
            (Some(sub @ prefill_fp8::SubCta::Half), Some(_)) => (
                prefill_fp8::sub_kernels(gpu)?.qkpv_amax_half,
                sub.smem() as u32,
                sub.sub_factor(),
                sub.threads(),
            ),
            (Some(sub @ prefill_fp8::SubCta::Quarter), None) => (
                prefill_fp8::sub_kernels(gpu)?.qkpv_qtr,
                sub.smem() as u32,
                sub.sub_factor(),
                sub.threads(),
            ),
            (Some(sub @ prefill_fp8::SubCta::Quarter), Some(_)) => (
                prefill_fp8::sub_kernels(gpu)?.qkpv_amax_qtr,
                sub.smem() as u32,
                sub.sub_factor(),
                sub.threads(),
            ),
        };
        let kernel = match (pipe, amax_slot, mode) {
            (false, _, _) => kernel,
            (true, None, Mode::Qk) => prefill_fp8::pipe_kernels(gpu)?.qk,
            (true, Some(_), Mode::Qk) => prefill_fp8::pipe_kernels(gpu)?.qk_amax,
            (true, None, Mode::QkPv) => prefill_fp8::pipe_kernels(gpu)?.qkpv,
            (true, Some(_), Mode::QkPv) => prefill_fp8::pipe_kernels(gpu)?.qkpv_amax,
        };
        // Per KV head h: 1/s_v[h] with E4M3 V.
        let out_scale = |h: usize| {
            if mode == Mode::QkPv {
                1.0 / sc.v[h]
            } else {
                1.0
            }
        };
        let launch = KernelLaunch::new(gpu, kernel)
            .grid([tiles as u32, (KV_HEADS as u32) * sub_factor, 1])
            .block([block_threads, 1, 1])
            .shared_mem(dyn_smem)
            .arg_ptr(q8)
            .arg_ptr(k8)
            .arg_ptr(v_pool)
            .arg_ptr(meta.block_table)
            .arg_ptr(entries)
            .arg_ptr(counts)
            .arg_ptr(c.out)
            .arg_ptr(c.gate)
            .arg_u32(c.rows as u32)
            .arg_u32(c.start as u32)
            .arg_u32((QUERY_HEADS * HEAD_DIM) as u32)
            .arg_u32((QUERY_HEADS * HEAD_DIM) as u32)
            .arg_u32(c.gate_stride)
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u64(page8)
            .arg_u32(tok8)
            .arg_u64(v_page)
            .arg_u32(v_tok)
            .arg_f32(scale_log2(0))
            .arg_f32(scale_log2(1))
            .arg_f32(out_scale(0))
            .arg_f32(out_scale(1));
        match amax_slot {
            Some(slot) => {
                launch.arg_ptr(slot).launch(stream)?;
                act_amax::note_ready(act_amax::Ready {
                    act: c.out.0,
                    rows: c.rows,
                    cols: QUERY_HEADS * HEAD_DIM,
                    stream,
                    producer: act_amax::Producer::QsaCore,
                });
            }
            None => launch.launch(stream)?,
        }
        log_e4m3_core_engaged(
            c, mode, wdec, sub_cta, pipe, pf, v8_shared, TILE_ROWS, layer, sc,
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn log_e4m3_core_engaged(
    c: &SparseChunk,
    mode: Mode,
    wdec: Option<prefill_fp8::WarpDecouple>,
    sub_cta: Option<prefill_fp8::SubCta>,
    pipe: bool,
    pf: bool,
    v8_shared: bool,
    tile_rows: usize,
    layer: usize,
    sc: prefill_fp8::LayerScales,
) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::warn!(
            "QWEN4_PREFILL_QSA_ENGAGED selector={SELECTOR} start={} rows={} tile_rows={tile_rows} core=qsa_sparse_e4m3_{} v2=true order=sink_last wdec={} sub_cta={} pipe={pipe} plan_fused={pf} v8_shared={v8_shared} layer={layer} prescale q={:?} k={:?} v={:?}",
            c.start,
            c.rows,
            mode.name(),
            wdec.map_or("off", |w| w.name()),
            sub_cta.map_or("off", |s| s.name()),
            sc.q,
            sc.k,
            sc.v
        );
        tracing::info!(
            "QSA_FP8_CORE_ENGAGED mode={} selector={}",
            mode.name(),
            prefill_fp8::SELECTOR
        );
    });
}

/// The compact fused select's launch geometry for this chunk with the engagement
/// threshold, or `None` (logged once) when the chunk's maxima and candidates do
/// not fit shared memory: the chunk then keeps the unfused select.
fn fused_launch(
    fk: &'static super::prefill_sparse::FusedKernels,
    p: &SparsePlan,
    score_v2_active: bool,
) -> Result<
    Option<(
        &'static super::prefill_sparse::FusedKernels,
        fused_topk::FusedLaunch,
        usize,
    )>,
> {
    let min_blocks = fused_topk::min_blocks_for(score_v2_active).map_err(|e| anyhow::anyhow!(e))?;
    match fused_topk::launch_geometry_compact(p.ld, p.mask_words) {
        Some(g) => Ok(Some((fk, g, min_blocks))),
        None => {
            static REFUSED: std::sync::Once = std::sync::Once::new();
            REFUSED.call_once(|| {
                tracing::warn!(
                    "QWEN4_PREFILL_QSA_FUSED_TOPK_REFUSED selector={} reason=smem start={} rows={} ld={} mask_words={}",
                    fused_topk::SELECTOR,
                    p.start,
                    p.rows,
                    p.ld,
                    p.mask_words
                )
            });
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

    use super::{
        PENDING_QSA_IQ_OVERLAP, PendingQsaIqOverlap, QsaIqOverlapGuard,
        SCORE_RUN_BATCH_CAP_DEFAULT, SCORE_RUN_BATCH_CAP_WIDE_SM,
        iq_scratch_disjoint_from_qkv_and_hidden, plan_fused_use_b128_for_sm,
        resolve_qsa_iq_overlap_enabled, score_run_batch_cap_for_sm,
    };

    #[test]
    fn qsa_prefill_wide_sm_run_batch_and_plan_b128_preserve_gb10_defaults() {
        for sm in [0u32, 48, 64] {
            assert_eq!(score_run_batch_cap_for_sm(sm), SCORE_RUN_BATCH_CAP_DEFAULT);
            assert!(!plan_fused_use_b128_for_sm(sm, true));
            assert!(!plan_fused_use_b128_for_sm(sm, false));
        }
        for sm in [65u32, 110, 170, 188] {
            assert_eq!(score_run_batch_cap_for_sm(sm), SCORE_RUN_BATCH_CAP_WIDE_SM);
            assert!(plan_fused_use_b128_for_sm(sm, true));
            assert!(!plan_fused_use_b128_for_sm(sm, false));
        }
    }

    #[test]
    fn qsa_iq_overlap_selector_auto_on_wide_sm_and_off_on_gb10() {
        for sm in [0u32, 48, 64] {
            assert!(!resolve_qsa_iq_overlap_enabled(None, sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some(""), sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some("0"), sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some("off"), sm));
            assert!(resolve_qsa_iq_overlap_enabled(Some("1"), sm));
            assert!(resolve_qsa_iq_overlap_enabled(Some("true"), sm));
        }
        for sm in [65u32, 110, 170, 188] {
            assert!(resolve_qsa_iq_overlap_enabled(None, sm));
            assert!(resolve_qsa_iq_overlap_enabled(Some(""), sm));
            assert!(resolve_qsa_iq_overlap_enabled(Some("1"), sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some("0"), sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some("false"), sm));
            assert!(!resolve_qsa_iq_overlap_enabled(Some("off"), sm));
        }
    }

    #[test]
    fn qsa_iq_scratch_disjointness_checks_hidden_and_qkv() {
        let iq = DevicePtr(0x1000_0000);
        let hidden = DevicePtr(0x2000_0000);
        let qkv = DevicePtr(0x3000_0000);
        assert!(iq_scratch_disjoint_from_qkv_and_hidden(
            iq, 4096, hidden, 8192, qkv, 16384
        ));
        assert!(!iq_scratch_disjoint_from_qkv_and_hidden(
            DevicePtr::NULL,
            4096,
            hidden,
            8192,
            qkv,
            16384
        ));
        assert!(!iq_scratch_disjoint_from_qkv_and_hidden(
            iq,
            4096,
            DevicePtr::NULL,
            8192,
            qkv,
            16384
        ));
        // Overlapping hidden fails.
        assert!(!iq_scratch_disjoint_from_qkv_and_hidden(
            iq,
            4096,
            DevicePtr(0x1000_0800),
            4096,
            qkv,
            16384
        ));
        // Overlapping qkv fails.
        assert!(!iq_scratch_disjoint_from_qkv_and_hidden(
            iq,
            4096,
            hidden,
            8192,
            DevicePtr(0x1000_0FF0),
            16384
        ));
        // Adjacent buffers are disjoint.
        assert!(iq_scratch_disjoint_from_qkv_and_hidden(
            iq,
            4096,
            DevicePtr(0x1000_1000),
            4096,
            DevicePtr(0x1000_2000),
            4096
        ));
    }

    struct WaitRecordingGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        waits: Mutex<Vec<(u64, u64)>>,
        syncs: Mutex<Vec<u64>>,
        fail_wait: bool,
    }

    impl WaitRecordingGpu {
        fn new() -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                waits: Mutex::new(Vec::new()),
                syncs: Mutex::new(Vec::new()),
                fail_wait: false,
            }
        }

        fn failing_wait() -> Self {
            Self {
                fail_wait: true,
                ..Self::new()
            }
        }

        fn take_waits(&self) -> Vec<(u64, u64)> {
            std::mem::take(&mut *self.waits.lock().expect("waits lock poisoned"))
        }

        fn take_syncs(&self) -> Vec<u64> {
            std::mem::take(&mut *self.syncs.lock().expect("syncs lock poisoned"))
        }
    }

    impl GpuBackend for WaitRecordingGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.syncs.lock().expect("syncs lock poisoned").push(stream);
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, func_name)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
        fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
            self.waits
                .lock()
                .expect("waits lock poisoned")
                .push((stream, event));
            if self.fail_wait {
                anyhow::bail!("injected stream_wait_event failure");
            }
            Ok(())
        }
    }

    #[test]
    fn qsa_iq_overlap_guard_joins_unconsumed_and_skips_already_consumed() {
        let gpu = WaitRecordingGpu::new();
        let pending = PendingQsaIqOverlap {
            layer_idx: 3,
            hidden: DevicePtr(0x1000),
            iq_base: DevicePtr(0x2000),
            rows: 16_384,
            start: 0,
            main_stream: 77,
            side_stream: 88,
            done_event: 99,
        };

        // Case 1: unconsumed guard joins on drop and clears TLS.
        PENDING_QSA_IQ_OVERLAP.with(|cell| cell.set(Some(pending)));
        {
            let _guard = QsaIqOverlapGuard {
                gpu: &gpu,
                main_stream: 77,
                done_event: 99,
            };
        }
        assert_eq!(gpu.take_waits(), vec![(77, 99)]);
        assert!(gpu.take_syncs().is_empty());
        assert!(PENDING_QSA_IQ_OVERLAP.with(|cell| cell.get().is_none()));

        // Case 2: already-consumed guard does NOT double-wait on drop.
        PENDING_QSA_IQ_OVERLAP.with(|cell| cell.set(Some(pending)));
        {
            let _guard = QsaIqOverlapGuard {
                gpu: &gpu,
                main_stream: 77,
                done_event: 99,
            };
            PENDING_QSA_IQ_OVERLAP.with(|cell| cell.set(None));
        }
        assert!(gpu.take_waits().is_empty());
        assert!(gpu.take_syncs().is_empty());

        // Case 3: failed stream_wait_event falls back to synchronizing side_stream.
        let failing_gpu = WaitRecordingGpu::failing_wait();
        PENDING_QSA_IQ_OVERLAP.with(|cell| cell.set(Some(pending)));
        {
            let _guard = QsaIqOverlapGuard {
                gpu: &failing_gpu,
                main_stream: 77,
                done_event: 99,
            };
        }
        assert_eq!(failing_gpu.take_waits(), vec![(77, 99)]);
        assert_eq!(failing_gpu.take_syncs(), vec![88]);
        assert!(PENDING_QSA_IQ_OVERLAP.with(|cell| cell.get().is_none()));
    }

    #[test]
    #[ignore = "needs an idle GB10 GPU and the flash-next kernel target"]
    fn qsa_prefill_plan_fused_sort_desc_b128_zero_ulp_parity_on_gb10() {
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::GpuBackend;
        use spark_runtime::kernel_args::KernelLaunch;

        use super::super::prefill_sparse::{ATTN_CAP, PLAN_SORT_SMEM, TILE_ROWS};

        let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
        let gpu = AtlasCudaBackend::new(0, &ptx.modules).expect("init GB10 CUDA backend");
        let stream = gpu.default_stream();
        let k32 = gpu
            .kernel(
                "qwen4_qsa_prefill_attn_fp8",
                "qwen4_qsa_prefill_plan_fused_sort_desc",
            )
            .expect("plan_fused_sort_desc");
        let k128 = gpu
            .kernel(
                "qwen4_qsa_prefill_attn_fp8",
                "qwen4_qsa_prefill_plan_fused_sort_desc_b128",
            )
            .expect("plan_fused_sort_desc_b128");

        // Cover: small (< half_cap), large (> half_cap = 2560), partial last tile, and 16K/64K masks.
        for &(rows, first_pos, mask_words, density_mod) in &[
            (25usize, 2048usize, 128usize, 7u32),
            (40usize, 12_288usize, 256usize, 3u32), // dense enough to exceed half_cap = 2560
            (17usize, 32_765usize, 512usize, 11u32), // ragged tail + partial last tile
        ] {
            let tiles = rows.div_ceil(TILE_ROWS);
            let mut host_masks = vec![0u32; rows * mask_words];
            let mut seed = (rows * 1337 + first_pos) as u64;
            for r in 0..rows {
                let valid_blocks = (first_pos + r + 1) / 4;
                for w in 0..mask_words {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let mut word = 0u32;
                    for b in 0..32u32 {
                        let blk = w * 32 + b as usize;
                        if blk < valid_blocks
                            && ((seed >> (b & 31)) as u32).is_multiple_of(density_mod)
                        {
                            word |= 1u32 << b;
                        }
                    }
                    host_masks[r * mask_words + w] = word;
                }
            }
            let mask_bytes: Vec<u8> = host_masks.iter().flat_map(|w| w.to_le_bytes()).collect();
            let masks_dev = gpu.alloc(mask_bytes.len()).unwrap();
            gpu.copy_h2d(&mask_bytes, masks_dev).unwrap();

            let entries_bytes = tiles * ATTN_CAP * 4;
            let counts_bytes = tiles * 4;
            let e32 = gpu.alloc(entries_bytes).unwrap();
            let c32 = gpu.alloc(counts_bytes).unwrap();
            let e128 = gpu.alloc(entries_bytes).unwrap();
            let c128 = gpu.alloc(counts_bytes).unwrap();
            gpu.memset(e32, 0xA5, entries_bytes).unwrap();
            gpu.memset(e128, 0xA5, entries_bytes).unwrap();
            gpu.memset(c32, 0, counts_bytes).unwrap();
            gpu.memset(c128, 0, counts_bytes).unwrap();

            for (kernel, block_x, entries_dev, counts_dev) in
                [(k32, 32u32, e32, c32), (k128, 128u32, e128, c128)]
            {
                KernelLaunch::new(&gpu, kernel)
                    .grid([tiles as u32, 1, 1])
                    .block([block_x, 1, 1])
                    .shared_mem(PLAN_SORT_SMEM as u32)
                    .arg_ptr(masks_dev)
                    .arg_ptr(entries_dev)
                    .arg_ptr(counts_dev)
                    .arg_u32(rows as u32)
                    .arg_u32(first_pos as u32)
                    .arg_u32(mask_words as u32)
                    .arg_u32(TILE_ROWS as u32)
                    .arg_u32(ATTN_CAP as u32)
                    .arg_u32(tiles as u32)
                    .launch(stream)
                    .unwrap();
            }
            gpu.synchronize(stream).unwrap();

            let mut out_c32 = vec![0u8; counts_bytes];
            let mut out_c128 = vec![0u8; counts_bytes];
            let mut out_e32 = vec![0u8; entries_bytes];
            let mut out_e128 = vec![0u8; entries_bytes];
            gpu.copy_d2h(c32, &mut out_c32).unwrap();
            gpu.copy_d2h(c128, &mut out_c128).unwrap();
            gpu.copy_d2h(e32, &mut out_e32).unwrap();
            gpu.copy_d2h(e128, &mut out_e128).unwrap();

            assert_eq!(
                out_c128, out_c32,
                "rows={rows} first_pos={first_pos} counts mismatch"
            );
            for t in 0..tiles {
                let cnt =
                    u32::from_le_bytes(out_c32[t * 4..(t + 1) * 4].try_into().unwrap()) as usize;
                let base = t * ATTN_CAP * 4;
                assert_eq!(
                    &out_e128[base..base + cnt * 4],
                    &out_e32[base..base + cnt * 4],
                    "rows={rows} first_pos={first_pos} tile={t} cnt={cnt} entries mismatch"
                );
            }
        }
    }
}
