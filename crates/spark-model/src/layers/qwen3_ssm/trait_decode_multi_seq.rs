// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq.

use std::sync::OnceLock;

use super::*;

/// Number of page-locked host staging slots rotated per SSM layer for
/// `refresh_multi_seq_ptr_table`. Paired with per-slot CUDA completion events
/// so host-side writes never race with in-flight async H2D DMAs across
/// pipelined decode steps.
const MULTI_SEQ_PTR_PINNED_SLOTS: usize = 4;
const MULTI_SEQ_PTR_MAX_U64: usize = 2 * crate::layers::QWEN4_NATIVE_MAX_ROWS;
const MULTI_SEQ_PTR_SLOT_BYTES: usize = MULTI_SEQ_PTR_MAX_U64 * std::mem::size_of::<u64>();

pub(super) struct MultiSeqPtrPinnedRing {
    buf: spark_runtime::gpu::PinnedHostBuffer,
    slot_events: [u64; MULTI_SEQ_PTR_PINNED_SLOTS],
    slot_busy: [bool; MULTI_SEQ_PTR_PINNED_SLOTS],
    slot_uncertain_stream: [Option<u64>; MULTI_SEQ_PTR_PINNED_SLOTS],
    next_slot: usize,
    last_slot: usize,
    poisoned: bool,
}

impl MultiSeqPtrPinnedRing {
    fn try_allocate(gpu: &dyn GpuBackend) -> Option<Self> {
        let buf = gpu
            .alloc_host_pinned(MULTI_SEQ_PTR_PINNED_SLOTS * MULTI_SEQ_PTR_SLOT_BYTES)
            .ok()?;
        let mut slot_events = [0u64; MULTI_SEQ_PTR_PINNED_SLOTS];
        for (i, slot_event) in slot_events.iter_mut().enumerate() {
            match gpu.create_event() {
                Ok(ev) => *slot_event = ev,
                Err(_) => {
                    for &created in &slot_events[..i] {
                        if created != 0 {
                            let _ = gpu.destroy_event(created);
                        }
                    }
                    return None;
                }
            }
        }
        Some(Self {
            buf,
            slot_events,
            slot_busy: [false; MULTI_SEQ_PTR_PINNED_SLOTS],
            slot_uncertain_stream: [None; MULTI_SEQ_PTR_PINNED_SLOTS],
            next_slot: 0,
            last_slot: 0,
            poisoned: false,
        })
    }

    /// Wait until any prior async H2D DMA from `slot_idx` has completed so
    /// the host can overwrite that slot without racing the GPU copy engine.
    fn wait_slot(&mut self, slot_idx: usize, gpu: &dyn GpuBackend) -> Result<()> {
        anyhow::ensure!(
            !self.poisoned,
            "multi_seq_ptr_pinned_ring is poisoned after teardown or event failure"
        );
        if let Some(prev_stream) = self.slot_uncertain_stream[slot_idx] {
            if let Err(err) = gpu.synchronize(prev_stream) {
                self.poisoned = true;
                return Err(err);
            }
            self.slot_uncertain_stream[slot_idx] = None;
            self.slot_busy[slot_idx] = false;
            return Ok(());
        }
        if !self.slot_busy[slot_idx] {
            return Ok(());
        }
        let event = self.slot_events[slot_idx];
        if event == 0 {
            self.slot_busy[slot_idx] = false;
            return Ok(());
        }
        let mut spins = 0u32;
        loop {
            match gpu.poll_event(event) {
                Ok(true) => {
                    self.slot_busy[slot_idx] = false;
                    return Ok(());
                }
                Ok(false) => {
                    spins = spins.saturating_add(1);
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
                Err(err) => {
                    self.poisoned = true;
                    return Err(err);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiSeqNvfp4ProjRoute {
    TcM32N64SplitK,
    TcM32N64,
    TcM128,
    TcN128,
    ExactRows,
    LegacyGemm,
    SerialGemv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmEnvOverride {
    Disabled,
    Enabled,
    Auto,
}

fn parse_sm_env_override(raw: Option<&str>) -> SmEnvOverride {
    match raw.map(str::trim) {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
            SmEnvOverride::Disabled
        }
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => SmEnvOverride::Enabled,
        _ => SmEnvOverride::Auto,
    }
}

fn raw_ssm_proj_tc_override() -> SmEnvOverride {
    static RAW: OnceLock<SmEnvOverride> = OnceLock::new();
    *RAW.get_or_init(|| parse_sm_env_override(std::env::var("ATLAS_SSM_PROJ_TC").ok().as_deref()))
}

fn raw_cublas_gemm_override() -> SmEnvOverride {
    static RAW: OnceLock<SmEnvOverride> = OnceLock::new();
    *RAW.get_or_init(|| parse_sm_env_override(std::env::var("ATLAS_CUBLAS_GEMM").ok().as_deref()))
}

const fn resolve_ssm_proj_tc_for_sm_with_policy(
    env_override: SmEnvOverride,
    sm_count: u32,
    rows: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    resolve_ssm_proj_tc_for_sm(env_override, sm_count, rows)
}

const fn resolve_ssm_proj_tc_for_sm(env_override: SmEnvOverride, sm_count: u32, rows: u32) -> bool {
    match env_override {
        SmEnvOverride::Disabled => false,
        SmEnvOverride::Enabled => true,
        SmEnvOverride::Auto => sm_count > 64 && rows >= 16,
    }
}

const fn resolve_cublas_ba_for_sm_with_policy(
    env_override: SmEnvOverride,
    sm_count: u32,
    rows: usize,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    resolve_cublas_ba_for_sm(env_override, sm_count, rows)
}

const fn resolve_cublas_ba_for_sm(env_override: SmEnvOverride, sm_count: u32, rows: usize) -> bool {
    match env_override {
        SmEnvOverride::Disabled => false,
        SmEnvOverride::Enabled => rows >= 4,
        SmEnvOverride::Auto => sm_count > 64 && rows >= 4,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "pure projection dispatch table over hardware and kernel capabilities"
)]
const fn select_multi_seq_nvfp4_proj_route(
    rows: u32,
    tc_enabled: bool,
    has_transposed: bool,
    splitk_splits: u32,
    has_m32_n64_splitk: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_t_n128: bool,
    has_legacy_gemm: bool,
) -> MultiSeqNvfp4ProjRoute {
    if tc_enabled && has_transposed {
        if rows <= crate::layers::TC_VERIFY_MAX_ROWS as u32 {
            if splitk_splits > 0 && has_m32_n64_splitk {
                return MultiSeqNvfp4ProjRoute::TcM32N64SplitK;
            }
            if has_m32_n64 {
                return MultiSeqNvfp4ProjRoute::TcM32N64;
            }
        }
        if has_m128 {
            return MultiSeqNvfp4ProjRoute::TcM128;
        }
        if has_t_n128 {
            return MultiSeqNvfp4ProjRoute::TcN128;
        }
    }
    if rows >= 2 && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS as u32 {
        return MultiSeqNvfp4ProjRoute::ExactRows;
    }
    if has_legacy_gemm {
        MultiSeqNvfp4ProjRoute::LegacyGemm
    } else {
        MultiSeqNvfp4ProjRoute::SerialGemv
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "policy-aware multi-seq NVFP4 projection route over hardware and kernel capabilities"
)]
const fn select_multi_seq_nvfp4_proj_route_for_sm_with_policy(
    rows: u32,
    n: u32,
    sm_count: u32,
    tc_enabled: bool,
    has_transposed: bool,
    splitk_splits: u32,
    has_m32_n64_splitk: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_t_n128: bool,
    has_legacy_gemm: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> MultiSeqNvfp4ProjRoute {
    if det_batch.is_enabled() {
        return if rows >= 2
            && rows <= crate::layers::deterministic_batch::DETERMINISTIC_BATCH_MAX_ROWS as u32
        {
            MultiSeqNvfp4ProjRoute::ExactRows
        } else {
            MultiSeqNvfp4ProjRoute::SerialGemv
        };
    }
    select_multi_seq_nvfp4_proj_route_for_sm(
        rows,
        n,
        sm_count,
        tc_enabled,
        has_transposed,
        splitk_splits,
        has_m32_n64_splitk,
        has_m32_n64,
        has_m128,
        has_t_n128,
        has_legacy_gemm,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "SM-aware multi-seq NVFP4 projection route over hardware and kernel capabilities"
)]
const fn select_multi_seq_nvfp4_proj_route_for_sm(
    rows: u32,
    n: u32,
    sm_count: u32,
    tc_enabled: bool,
    has_transposed: bool,
    splitk_splits: u32,
    has_m32_n64_splitk: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_t_n128: bool,
    has_legacy_gemm: bool,
) -> MultiSeqNvfp4ProjRoute {
    // On high-SM GPUs (> 64 SMs), `w4a16_gemm_t_m128` at small/medium N
    // (<= 8192) launches only <= 64 CTAs across 148-188 SMs. Prefer the
    // m32_n64 tile (`ceil(N/64) * ceil(M/32)` CTAs) for high SM occupancy.
    if sm_count > 64
        && tc_enabled
        && has_transposed
        && rows > crate::layers::TC_VERIFY_MAX_ROWS as u32
        && n <= 8192
        && has_m32_n64
    {
        return MultiSeqNvfp4ProjRoute::TcM32N64;
    }
    select_multi_seq_nvfp4_proj_route(
        rows,
        tc_enabled,
        has_transposed,
        splitk_splits,
        has_m32_n64_splitk,
        has_m32_n64,
        has_m128,
        has_t_n128,
        has_legacy_gemm,
    )
}

/// Partition `num_seqs` dense FFN rows into batched `forward_kgamma` slices
/// `(start_row, row_count)`. Every slice has `row_count >= 2`; when `wide_ready`
/// is false, slices stay `<= TC_VERIFY_MAX_ROWS` (32) by splitting a 33-row tail
/// into `17 + 16`, preserving exactness and avoiding legacy `w4a16_gemm`.
fn plan_multi_seq_ffn_chunks(num_seqs: usize, wide_ready: bool) -> Vec<(usize, usize)> {
    if num_seqs < 2 {
        return Vec::new();
    }
    let max_chunk = crate::layers::TC_VERIFY_MAX_ROWS;
    if num_seqs <= max_chunk || wide_ready {
        return vec![(0, num_seqs)];
    }
    let mut chunks = Vec::with_capacity(num_seqs.div_ceil(max_chunk) + 1);
    let mut offset = 0;
    while offset < num_seqs {
        let rem = num_seqs - offset;
        let len = if rem <= max_chunk {
            rem
        } else if rem == max_chunk + 1 {
            max_chunk / 2 + 1
        } else {
            max_chunk
        };
        chunks.push((offset, len));
        offset += len;
    }
    chunks
}

impl Qwen3SsmLayer {
    #[expect(
        clippy::too_many_arguments,
        reason = "BF16 dense cuBLASLt projection helper matching GEMM operand arity"
    )]
    fn launch_cublas_bf16_proj(
        act: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        graph_capture: bool,
        stream: u64,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if graph_capture {
                // `bf16_gemm_act_weight_t` performs no stream syncs or runtime
                // tuning allocations and is safe inside CUDA graph capture.
                return spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    act.0,
                    weight.weight.0,
                    out.0,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            ops::cublas_bf16_proj_dense(act, weight, out, m, n, k, stream)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (act, weight, out, m, n, k, graph_capture, stream);
            anyhow::bail!("cuBLASLt BF16 projection requires the cuda feature")
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "batched multi-seq NVFP4 projection helper matching GEMM operand arity"
    )]
    fn project_multi_seq_nvfp4(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        weight_t: Option<&QuantizedWeight>,
        splitk_splits: u32,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        graph_capture: bool,
        label: &'static str,
        stream: u64,
    ) -> Result<()> {
        let det_batch =
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled();
        let sm_count = atlas_core::device::active_sm_count();
        let tc_override = raw_ssm_proj_tc_override();
        let tc_enabled =
            resolve_ssm_proj_tc_for_sm_with_policy(tc_override, sm_count, rows, det_batch);

        // On wide discrete GPUs (`sm_count > 64`) at `M = 4..=sm120_max_m`
        // (up to `C=64` when `ATLAS_SM120_COALESCED_WIDE_M!=0`, else `C=16`),
        // route `in_proj_qkvz` (`N=16384`) and `out_proj` (`N=5120`) directly
        // through single-launch unclipped BF16xBF16 MMA coalesced warp-split-K
        // kernels (zero activation quantization, zero global Split-K partials,
        // full 188-SM 600W saturation, graph-capture-safe). Always skipped on
        // GB10 (`<= 64` SMs) or when `ATLAS_SSM_PROJ_TC=0`.
        let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
        if !det_batch.is_enabled()
            && rows >= 4
            && (rows as usize) <= sm120_max_m
            && tc_override != SmEnvOverride::Disabled
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                input,
                weight,
                output,
                rows,
                n,
                k,
                sm120_max_m,
                stream,
            )?
        {
            return Ok(());
        }

        // On SM120/SM103 (`sm_count > 64`) at M >= 24 (immediately above the
        // `M <= 16` coalesced BF16 MMA tier, or `M > 64` when wide coalesced
        // is enabled), route through native block-scaled cuBLASLt NVFP4 unless
        // the caller explicitly disabled TC via `ATLAS_SSM_PROJ_TC=0`.
        if !det_batch.is_enabled()
            && rows >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M
            && tc_override != SmEnvOverride::Disabled
            && ops::try_cublaslt_nvfp4_multiseq_single(
                gpu,
                input,
                weight,
                output,
                rows,
                n,
                k,
                n,
                graph_capture,
                label,
                stream,
            )?
        {
            return Ok(());
        }

        let ws = if splitk_splits > 0 && !det_batch.is_enabled() {
            *self
                .ssm_splitk_workspace
                .lock()
                .map_err(|_| anyhow::anyhow!("ssm_splitk_workspace lock poisoned"))?
        } else {
            None
        };
        let route = select_multi_seq_nvfp4_proj_route_for_sm_with_policy(
            rows,
            n,
            sm_count,
            tc_enabled,
            weight_t.is_some(),
            splitk_splits,
            self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && ws.is_some(),
            self.w4a16_gemm_t_m32_n64_k.0 != 0,
            self.w4a16_gemm_t_m128_k.0 != 0,
            self.w4a16_gemm_t_k.0 != 0,
            self.w4a16_gemm_k.0 != 0,
            det_batch,
        );
        match route {
            MultiSeqNvfp4ProjRoute::TcM32N64SplitK => {
                let wt = weight_t.expect("TcM32N64SplitK requires transposed weight");
                let ws_ptr = ws.expect("TcM32N64SplitK requires split-K workspace");
                ops::w4a16_gemm_n64_m32_splitk(
                    gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    input,
                    wt,
                    output,
                    ws_ptr,
                    rows,
                    n,
                    k,
                    n,
                    splitk_splits,
                    stream,
                )
            }
            MultiSeqNvfp4ProjRoute::TcM32N64 => {
                let wt = weight_t.expect("TcM32N64 requires transposed weight");
                ops::w4a16_gemm_n64_m32(
                    gpu,
                    self.w4a16_gemm_t_m32_n64_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            MultiSeqNvfp4ProjRoute::TcM128 => {
                let wt = weight_t.expect("TcM128 requires transposed weight");
                ops::w4a16_gemm_n128_m128(
                    gpu,
                    self.w4a16_gemm_t_m128_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            MultiSeqNvfp4ProjRoute::TcN128 => {
                let wt = weight_t.expect("TcN128 requires transposed weight");
                ops::w4a16_gemm_n128(
                    gpu,
                    self.w4a16_gemm_t_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            MultiSeqNvfp4ProjRoute::ExactRows => {
                let _ = self.project_nvfp4_rows_exact_or_k1(
                    gpu, input, weight, output, rows, n, k, stream,
                )?;
                Ok(())
            }
            MultiSeqNvfp4ProjRoute::LegacyGemm => ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            ),
            MultiSeqNvfp4ProjRoute::SerialGemv => {
                const BF16_BYTES: usize = 2;
                for row in 0..rows as usize {
                    ops::w4a16_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        input.offset(row * k as usize * BF16_BYTES),
                        weight,
                        output.offset(row * n as usize * BF16_BYTES),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    /// Multi-sequence decode (one token per sequence, independent SSM state).
    ///
    /// Batches state-free ops (`rms_norm_residual`, QKVZ, BA + GDN gates,
    /// `gated_rms_norm_f32_multi_seq`, `out_proj`, `residual_add_rms_norm`,
    /// and Dense FFN `forward_kgamma`) across all `num_seqs` sequences. When
    /// `ATLAS_SSM_MULTI_SEQ_KERNEL` is active and the FP32 multi-seq kernels
    /// are loaded, state-bearing `conv1d_update_l2norm` and `gdn_decode` also
    /// run in single batched launches via a device-resident per-sequence
    /// state pointer table (`ssm_multi_seq_ptr_scratch`); otherwise steps 2-7
    /// fall back to a per-sequence loop over disjoint slices of
    /// `ssm_conv_out_f32` (`m * ssm_qkvz_size * 4`), mirroring single-seq
    /// `ssm_forward`. MoE/None FFN falls back to per-sequence `ffn.forward`
    /// interleaved with `residual_add`.
    ///
    /// If the FP32 conv1d / GDN / gated-RMS kernels aren't loaded on the
    /// active backend (e.g. Metal), fall back to per-sequence
    /// [`decode_inner`] — the BF16 path stores GDN output in `attn_output`,
    /// which the safe layout above can't accommodate.
    pub(super) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let use_batched = self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0;
        if !use_batched {
            return self.decode_multi_seq_per_seq_fallback(
                hidden,
                residual,
                num_seqs,
                states,
                kv_cache,
                seq_lens,
                block_tables,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let n = num_seqs as u32;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = (key_dim * 2 + value_dim) as u32;
        let qk_channels = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim as u32;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();

        // ── 1. Batched RMS norm + residual across all N sequences ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2-7. Per-seq SSM forward (state-bearing) ──
        //
        // Buffer layout per seq i:
        //   ssm_deinterleaved[i * qkvz_size * bf16 ..]            — [Q|K|V|Z]
        //   ssm_gates[i * nv * 2 * fp32 ..]                       — [gate(nv) | beta(nv)]
        //   ssm_conv_out_f32[i * qkvz_size * fp32 ..]             — conv1d FP32 out
        //     + (key_dim*2 + value_dim)*fp32                      — GDN FP32 out (in-slice)
        //   ssm_qkvz[i * qkvz_size * bf16 ..]                     — normed_out (gated_rms)
        //   moe_output[i * h * bf16 ..]                           — SSM output projection
        //
        // All strides match the per-token natural stride of the respective
        // arena buffer (see `BufferSizes::from_config`), so per-seq writes
        // stay in-bounds for any `num_seqs ≤ max_batch_tokens`.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let gates_buf = ctx.buffers.ssm_gates();
        let conv_out_f32 = ctx.buffers.ssm_conv_out_f32();
        let normed_out_buf = ctx.buffers.ssm_qkvz();
        let ssm_out_buf = ctx.buffers.moe_output();
        let gdn_local_offset = (key_dim * 2 + value_dim) * fp32;

        // ── ATLAS_SSM_MULTI_SEQ_KERNEL multi-seq batched dispatch ──
        //
        // Gated by `ssm_multi_seq_kernel_enabled()` AND the FP32 multi-seq
        // kernel handles AND `num_seqs <= ssm_multi_seq_ptr_max`. When ON:
        //   - QKVZ projection runs batched via `project_multi_seq_nvfp4`
        //     (sequential NVFP4) or per-seq fallback for interleaved/BF16.
        //   - BA + GDN gates run batched via `dense_gemm`/cuBLASLt +
        //     `compute_gdn_gates_multi_seq` when available.
        //   - `conv1d_update_l2norm` and `gdn_decode` each collapse to a SINGLE
        //     launch via the FP32/FP16 multi-seq variants, fed a device-resident
        //     per-seq state pointer table uploaded once per layer.
        //   - `gated_rms_norm_f32_multi_seq` + batched `out_proj` run in single
        //     launches when the multi-seq norm kernel is present.
        let multi_seq_kernel_path = num_seqs >= 2
            && num_seqs <= self.ssm_multi_seq_ptr_max
            && self.conv1d_l2norm_f32_multi_seq_k.0 != 0
            && self.gdn_decode_f32_multi_seq_k.0 != 0
            && crate::layers::ssm_multi_seq_kernel_enabled();
        if multi_seq_kernel_path {
            // Collect per-seq h_state + conv_state device pointers into the
            // layer's 4-slot rotating pinned host ring and upload to
            // `ssm_multi_seq_ptr_scratch`:
            //   [h_state_ptrs[N] u64, conv_state_ptrs[N] u64]
            //
            // Under CUDA graph capture (`ctx.graph_capture`), the graph driver
            // (`decode_a2_piecewise` / `decode_a2`) already refreshes
            // `ssm_multi_seq_ptr_scratch` on `stream` BEFORE both capture and
            // every replay. Skipping the upload here ensures the captured graph
            // contains ZERO tiny H2D nodes that would overwrite the pre-replay
            // fresh pointer table once batch composition drifts.
            if !ctx.graph_capture {
                self.refresh_multi_seq_ptr_table(states, num_seqs, ctx.gpu, stream)?;
            }
            let h_state_ptrs_dev = self.ssm_multi_seq_ptr_scratch;
            let conv_state_ptrs_dev = self
                .ssm_multi_seq_ptr_scratch
                .offset(num_seqs * std::mem::size_of::<u64>());

            // ── Step 2: QKVZ projection ──
            // Batched path: sequential_qkvz + nvfp4 → cuBLASLt NVFP4 (SM120
            //   M >= 32), TC transposed GEMM, or single-sweep exact rows.
            // Fallback: per-seq loop (interleaved / dense / non-nvfp4).
            if self.sequential_qkvz
                && let Some(ref nvfp4) = self.qkvz_nvfp4
            {
                self.project_multi_seq_nvfp4(
                    ctx.gpu,
                    normed,
                    nvfp4,
                    self.qkvz_nvfp4_t.as_ref(),
                    super::super::ssm_qkvz_splitk(),
                    deinterleaved,
                    num_seqs as u32,
                    qkvz_size as u32,
                    h as u32,
                    ctx.graph_capture,
                    "ssm_qkvz_multi_seq",
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let normed_i = normed.offset(i * h * bf16);
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    if self.sequential_qkvz {
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_i,
                            self.bf16_qkvz()?,
                            deint_i,
                            qkvz_size as u32,
                            h as u32,
                            stream,
                        )?;
                    } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                        ops::w4a16_gemv_qkvz(
                            ctx.gpu,
                            self.w4a16_gemv_qkvz_k,
                            normed_i,
                            nvfp4,
                            deint_i,
                            qkvz_size as u32,
                            h as u32,
                            nk as u32,
                            kd as u32,
                            vpg as u32,
                            vd as u32,
                            stream,
                        )?;
                    } else {
                        // Write interleaved output into `ssm_qkvz` (disjoint from
                        // `deint_i` and unused until Step 6) before deinterleaving.
                        let qkvz_i = normed_out_buf.offset(i * qkvz_size * bf16);
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_i,
                            self.bf16_qkvz()?,
                            qkvz_i,
                            qkvz_size as u32,
                            h as u32,
                            stream,
                        )?;
                        ops::deinterleave_qkvz(
                            ctx.gpu,
                            self.deinterleave_k,
                            qkvz_i,
                            deint_i,
                            1,
                            nk as u32,
                            kd as u32,
                            vpg as u32,
                            vd as u32,
                            stream,
                        )?;
                    }
                }
            }

            // ── Step 3: BA projection + GDN gates ──
            // Batched path: split the per-seq `dense_gemv_ba_gates` fusion
            //   into one batched dense_gemm (BA at M=num_seqs writing BF16
            //   [num_seqs, ba_size] into the head of `ssm_conv_out_f32` —
            //   safe because conv1d_update_l2norm writes that buffer LATER
            //   in step 4) + one `compute_gdn_gates_multi_seq` consuming
            //   the BA output.
            // Fallback: per-seq `dense_gemv_ba_gates` loop.
            let det_batch =
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled();
            let use_cublas_ba = cfg!(feature = "cuda")
                && resolve_cublas_ba_for_sm_with_policy(
                    raw_cublas_gemm_override(),
                    atlas_core::device::active_sm_count(),
                    num_seqs,
                    det_batch,
                );
            let ba_batched_split = !det_batch.is_enabled()
                && (self.dense_gemm_k.0 != 0 || use_cublas_ba)
                && self.compute_gdn_gates_multi_seq_k.0 != 0;
            if ba_batched_split {
                // Scratch: `conv_out_f32` reinterpreted as BF16 head. We
                // need num_seqs * ba_size BF16 elements; the buffer is
                // sized num_seqs * qkvz_size FP32 = num_seqs * 2 *
                // qkvz_size BF16 ≫ num_seqs * ba_size BF16. Safe.
                let ba_scratch = conv_out_f32;
                if use_cublas_ba {
                    Self::launch_cublas_bf16_proj(
                        normed,
                        &self.ssm.in_proj_ba,
                        ba_scratch,
                        num_seqs as u32,
                        ba_size as u32,
                        h as u32,
                        ctx.graph_capture,
                        stream,
                    )?;
                } else {
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm_k,
                        normed,
                        &self.ssm.in_proj_ba,
                        ba_scratch,
                        num_seqs as u32,
                        ba_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                ops::compute_gdn_gates_multi_seq(
                    ctx.gpu,
                    self.compute_gdn_gates_multi_seq_k,
                    ba_scratch,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    num_seqs as u32,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,  // BF16 elements between seqs in ba_scratch
                    (nv * 2) as u32, // FP32 elements between seqs in gates_buf
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let normed_i = normed.offset(i * h * bf16);
                    let gate_i = gates_buf.offset(i * nv * 2 * fp32);
                    let beta_i = gate_i.offset(nv * fp32);
                    ops::dense_gemv_ba_gates(
                        ctx.gpu,
                        self.ba_gates_k,
                        normed_i,
                        &self.ssm.in_proj_ba,
                        self.ssm.a_log.weight,
                        self.ssm.dt_bias.weight,
                        gate_i,
                        beta_i,
                        ba_size as u32,
                        h as u32,
                        vpg as u32,
                        stream,
                    )?;
                }
            }

            // Step 4: ONE conv1d_update_l2norm_f32_multi_seq launch.
            // Input layout: deinterleaved [num_seqs, qkvz_size] BF16
            //   → input_stride = qkvz_size BF16 elements between seqs
            // Output layout: conv_out_f32 [num_seqs, qkvz_size] FP32
            //   → output_stride = qkvz_size FP32 elements between seqs
            ops::conv1d_update_l2norm_f32_multi_seq(
                ctx.gpu,
                self.conv1d_l2norm_f32_multi_seq_k,
                conv_state_ptrs_dev,
                deinterleaved,
                &self.ssm.conv1d,
                conv_out_f32,
                conv_dim,
                d_conv,
                num_seqs as u32,
                qk_channels,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                stream,
            )?;

            // Step 5: ONE gdn_decode_f32_multi_seq launch.
            //   Q/K split: query at conv_out_f32 + offset 0
            //              key   at conv_out_f32 + key_dim FP32 elements
            //   V        : conv_out_f32 + 2*key_dim FP32 elements
            //   gate/beta: gates_buf, gate_beta_stride = 2*nv FP32 elements
            //   output   : conv_out_f32 + gdn_local_offset (Z region tail
            //              per ssm_forward.rs single-seq layout —
            //              re-uses the qkv region's tail in the FP32 slot)
            //   v_out_stride = qkvz_size FP32 elements between seqs
            // q/k/v strides are FP32 elements (kernel reads FP32 inputs
            // matching conv1d_update_l2norm_f32_multi_seq output layout).
            let qk_stride_fp32 = qkvz_size as u32;
            let v_in_stride_fp32 = qkvz_size as u32;
            let gdn_out = conv_out_f32.offset(gdn_local_offset);
            let h_fp16 = super::ssm_h_fp16_enabled();
            for (i, state) in states.iter().enumerate().take(num_seqs) {
                let ssm_state = state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                if h_fp16 {
                    super::ssm_h_fp16::require_h_f16(ssm_state)?;
                } else {
                    anyhow::ensure!(
                        !ssm_state.h_is_f16,
                        "decode_multi_seq_inner received FP16 h-state for seq {i} while ATLAS_SSM_H_FP16 is disabled"
                    );
                }
            }
            let gdn_kernel = if h_fp16 {
                anyhow::ensure!(
                    self.gdn_decode_f16_multi_seq_k.0 != 0,
                    "ATLAS_SSM_H_FP16: gated_delta_rule_decode_f16_multi_seq kernel is unavailable for multi-seq decode"
                );
                self.gdn_decode_f16_multi_seq_k
            } else {
                self.gdn_decode_f32_multi_seq_k
            };
            ops::gdn_decode_f32_multi_seq(
                ctx.gpu,
                gdn_kernel,
                h_state_ptrs_dev,
                conv_out_f32,
                conv_out_f32.offset(key_dim * fp32),
                conv_out_f32.offset(key_dim * 2 * fp32),
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out,
                num_seqs as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                (nv * 2) as u32,
                qk_stride_fp32,
                v_in_stride_fp32,
                qkvz_size as u32,
                stream,
            )?;

            // ── Step 6: Gated RMS norm ──
            // Batched path: ONE `gated_rms_norm_f32_multi_seq` launch
            //   reading from `conv_out_f32 + gdn_local_offset` (stride
            //   qkvz_size FP32) + `deinterleaved + Z offset` (stride
            //   qkvz_size BF16) and WRITING into a value_dim-contig
            //   scratch at the head of `normed_out_buf` (= ssm_qkvz).
            //   That buffer is sized num_seqs * qkvz_size BF16 ≫ the
            //   num_seqs * value_dim BF16 we need, and it isn't read
            //   until step 7 (out_proj) which we batch directly out of
            //   the value_dim-contig layout we just wrote.
            //   NOTE: `deinterleaved` IS the gate source, so it cannot
            //   double as the compact output (writes would alias gate
            //   reads of later seqs — produces garbage).
            // Fallback: per-seq loop.
            let normed_out_compact = normed_out_buf; // value_dim-contig scratch in ssm_qkvz
            let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
            let rms_batched = self.gated_rms_norm_f32_multi_seq_k.0 != 0;
            if rms_batched {
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    conv_out_f32.offset(gdn_local_offset),
                    z_base,
                    &self.ssm.norm,
                    normed_out_compact,
                    nv as u32, // num_v_heads (32)
                    num_seqs as u32,
                    vd as u32, // head_dim (128) — norm is per-head
                    eps,
                    qkvz_size as u32, // FP32 elements between seqs (input)
                    qkvz_size as u32, // BF16 elements between seqs (gate)
                    value_dim as u32, // BF16 elements between seqs (output)
                    stream,
                )?;
            } else {
                for i in 0..num_seqs {
                    let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
                    let conv_out_i = conv_out_f32.offset(i * qkvz_size * fp32);
                    let gdn_out_i = conv_out_i.offset(gdn_local_offset);
                    let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
                    let z_ptr = deint_i.offset((key_dim * 2 + value_dim) * bf16);
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_out_i,
                        z_ptr,
                        &self.ssm.norm,
                        normed_out_i,
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                }
            }

            // ── Step 7: Output projection ──
            // Batched path: ONE w4a16_gemm at M=num_seqs reading the
            //   value_dim-contig output from step 6 (`normed_out_compact`,
            //   stride value_dim BF16 = K) and writing into `ssm_out_buf`
            //   [num_seqs, h] BF16 contig (stride h BF16 = N). Falls back
            //   to dense_gemm M=num_seqs when no nvfp4 out_proj, else
            //   per-seq.
            if rms_batched {
                if let Some(ref dense_out) = self.out_proj_dense {
                    if use_cublas_ba {
                        Self::launch_cublas_bf16_proj(
                            normed_out_compact,
                            dense_out,
                            ssm_out_buf,
                            num_seqs as u32,
                            h as u32,
                            value_dim as u32,
                            ctx.graph_capture,
                            stream,
                        )?;
                    } else if !det_batch.is_enabled() && self.dense_gemm_k.0 != 0 {
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_k,
                            normed_out_compact,
                            dense_out,
                            ssm_out_buf,
                            num_seqs as u32,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    } else {
                        for i in 0..num_seqs {
                            let normed_out_i = normed_out_compact.offset(i * value_dim * bf16);
                            let ssm_out_i = ssm_out_buf.offset(i * h * bf16);
                            ops::dense_gemv(
                                ctx.gpu,
                                self.dense_gemv_k,
                                normed_out_i,
                                dense_out,
                                ssm_out_i,
                                h as u32,
                                value_dim as u32,
                                stream,
                            )?;
                        }
                    }
                } else {
                    self.project_multi_seq_nvfp4(
                        ctx.gpu,
                        normed_out_compact,
                        &self.ssm.out_proj,
                        self.out_proj_nvfp4_t.as_ref(),
                        super::super::ssm_out_splitk(),
                        ssm_out_buf,
                        num_seqs as u32,
                        h as u32,
                        value_dim as u32,
                        ctx.graph_capture,
                        "ssm_out_proj_multi_seq",
                        stream,
                    )?;
                }
            } else {
                // RMS norm fallback wrote per-seq into normed_out_buf with
                // qkvz_size stride; reuse the original per-seq out_proj.
                for i in 0..num_seqs {
                    let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
                    let ssm_out_i = ssm_out_buf.offset(i * h * bf16);
                    if let Some(ref dense_out) = self.out_proj_dense {
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_out_i,
                            dense_out,
                            ssm_out_i,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv(
                            ctx.gpu,
                            self.w4a16_gemv_k,
                            normed_out_i,
                            &self.ssm.out_proj,
                            ssm_out_i,
                            h as u32,
                            value_dim as u32,
                            stream,
                        )?;
                    }
                }
            }

            // Skip the legacy per-seq loop below and continue with the
            // existing batched step 8 (residual_add_rms_norm) + step 9
            // (per-seq MoE) at the bottom of this function.
            // We did the equivalent of steps 2-7 above. Use a labeled
            // loop break by overwriting the entire per-seq block with a
            // sentinel.
        }

        for i in 0..num_seqs {
            if multi_seq_kernel_path {
                break; // already done above
            }
            let normed_i = normed.offset(i * h * bf16);
            let deint_i = deinterleaved.offset(i * qkvz_size * bf16);
            let gate_i = gates_buf.offset(i * nv * 2 * fp32);
            let beta_i = gate_i.offset(nv * fp32);
            let conv_out_i = conv_out_f32.offset(i * qkvz_size * fp32);
            let gdn_out_i = conv_out_i.offset(gdn_local_offset);
            let normed_out_i = normed_out_buf.offset(i * qkvz_size * bf16);
            let ssm_out_i = ssm_out_buf.offset(i * h * bf16);

            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;

            // ── 2. QKVZ projection (+ deinterleave if needed) ──
            if self.sequential_qkvz {
                if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed_i,
                        nvfp4,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed_i,
                        self.bf16_qkvz()?,
                        deint_i,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_qkvz(
                    ctx.gpu,
                    self.w4a16_gemv_qkvz_k,
                    normed_i,
                    nvfp4,
                    deint_i,
                    qkvz_size as u32,
                    h as u32,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            } else {
                // Dense fallback: interleaved GEMV into `ssm_qkvz` (disjoint
                // from `deint_i` and unused until Step 6), then deinterleave.
                let qkvz_i = normed_out_buf.offset(i * qkvz_size * bf16);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    self.bf16_qkvz()?,
                    qkvz_i,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    qkvz_i,
                    deint_i,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }

            // ── 3. Fused BA projection + GDN gates ──
            ops::dense_gemv_ba_gates(
                ctx.gpu,
                self.ba_gates_k,
                normed_i,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gate_i,
                beta_i,
                ba_size as u32,
                h as u32,
                vpg as u32,
                stream,
            )?;

            // ── 4. Conv1d update + SiLU + L2 norm (FP32 output) ──
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                ssm_state.conv_state,
                deint_i,
                &self.ssm.conv1d,
                conv_out_i,
                conv_dim,
                d_conv,
                1,
                qk_channels,
                kd as u32,
                1e-6,
                stream,
            )?;

            // ── 5. GDN decode (FP32 state + output) ──
            anyhow::ensure!(
                !ssm_state.h_is_f16 && !super::ssm_h_fp16_enabled(),
                "ATLAS_SSM_H_FP16 requires the multi-seq GDN kernel path; \
                 per-seq FP32 fallback in decode_multi_seq_inner cannot run on FP16 h-state"
            );
            let q_conv = conv_out_i;
            let k_conv = conv_out_i.offset(key_dim * fp32);
            let v_conv = conv_out_i.offset(key_dim * 2 * fp32);
            ops::gdn_decode(
                ctx.gpu,
                self.gdn_f32_k,
                ssm_state.h_state,
                q_conv,
                k_conv,
                v_conv,
                gate_i,
                beta_i,
                gdn_out_i,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                stream,
            )?;

            // ── 6. Gated RMS norm (FP32 GDN input → BF16 normed output) ──
            let z_ptr = deint_i.offset((key_dim * 2 + value_dim) * bf16);
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_f32_k,
                gdn_out_i,
                z_ptr,
                &self.ssm.norm,
                normed_out_i,
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;

            // ── 7. Output projection [value_dim → hidden_size] ──
            if let Some(ref dense_out) = self.out_proj_dense {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_out_i,
                    dense_out,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed_out_i,
                    &self.ssm.out_proj,
                    ssm_out_i,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(ssm_out_buf.0, num_seqs * h * bf16, stream)?;
        }

        // ── 8. Batched residual + post-attn RMS norm across all N seqs ──
        // Reads SSM output from `ssm_out_buf[0..n*h*bf16]` (contiguous,
        // written in step 7) and produces post-norm input for the MoE.
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out_buf,
            &self.post_attn_norm,
            normed2,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 9. Batched Dense FFN + residual_add (MoE/None fall back per-seq) ──
        //
        // Dense FFN services `num_seqs >= 2` in batched slices via
        // `forward_kgamma` (contiguous `normed2` → `moe_output[0..count*h]`)
        // followed by one batched `residual_add` per slice. When `num_seqs > 32`
        // and the wide M=128 TC route is not active, chunks stay `<= 32` rows
        // (splitting a 33-row tail into `17 + 16`) so default exactness holds
        // and legacy `w4a16_gemm` is never hit. MoE / BF16-dense / None return
        // `Ok(false)` on the first chunk and execute the per-seq loop below.
        let mut ffn_batched = false;
        if num_seqs >= 2 {
            let wide_ready = !crate::layers::deterministic_batch::deterministic_batch_enabled()
                && self.ffn.wide_kgamma_ready(num_seqs as u32);
            let chunks = plan_multi_seq_ffn_chunks(num_seqs, wide_ready);
            ffn_batched = !chunks.is_empty();
            for (start, count) in chunks {
                let normed2_chunk = normed2.offset(start * h * bf16);
                if !self
                    .ffn
                    .forward_kgamma(normed2_chunk, count as u32, ctx, stream)?
                {
                    ffn_batched = false;
                    break;
                }
                let moe_out = ctx.buffers.moe_output();
                let hidden_chunk = hidden.offset(start * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_chunk,
                    moe_out,
                    (count * h) as u32,
                    stream,
                )?;
            }
        }
        if !ffn_batched {
            for i in 0..num_seqs {
                let normed2_i = normed2.offset(i * h * bf16);
                let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
                let hidden_i = hidden.offset(i * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_i,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }

        Ok(())
    }

    /// True when this SSM layer will take the multi-seq kernel path for
    /// `num_seqs` sequences (the path that indirects per-seq h_state /
    /// conv_state through `ssm_multi_seq_ptr_scratch` rather than baking raw
    /// `SsmLayerState` pointers into kernel args). This is the precondition
    /// for a segment containing this layer to be CUDA-graph-safe under the
    /// piecewise dispatcher.
    pub(crate) fn multi_seq_kernel_path_active(&self, num_seqs: usize) -> bool {
        let gdn_multi_ok = if super::ssm_h_fp16_enabled() {
            self.gdn_decode_f16_multi_seq_k.0 != 0
        } else {
            self.gdn_decode_f32_multi_seq_k.0 != 0
        };
        num_seqs >= 2
            && num_seqs <= self.ssm_multi_seq_ptr_max
            && self.conv1d_l2norm_f32_multi_seq_k.0 != 0
            && gdn_multi_ok
            && self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0
            && crate::layers::ssm_multi_seq_kernel_enabled()
    }

    /// Release CUDA completion events owned by this layer's multi-seq pointer
    /// ring during model teardown (`TransformerModel::drop` has already drained
    /// streams and destroyed CUDA graphs).
    pub(super) fn release_multi_seq_ptr_ring(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut guard = self
            .multi_seq_ptr_pinned_ring
            .lock()
            .map_err(|_| anyhow::anyhow!("multi_seq_ptr_pinned_ring lock poisoned"))?;
        let Some(ring) = guard.as_mut() else {
            return Ok(());
        };
        ring.poisoned = true;
        for slot_event in &mut ring.slot_events {
            let event = std::mem::replace(slot_event, 0);
            if event != 0 {
                gpu.destroy_event(event)
                    .map_err(|err| anyhow::anyhow!("multi-seq ptr ring event destroy: {err:#}"))?;
            }
        }
        Ok(())
    }

    /// Gather-before-replay: refresh this layer's rotating 4-slot page-locked
    /// host staging buffer (fenced per slot with CUDA completion events) with
    /// the current active sequences' `h_state` / `conv_state` device addresses
    /// and upload it to the fixed device scratch (`ssm_multi_seq_ptr_scratch`)
    /// on `stream`.
    ///
    /// Captured segment graphs bake ONLY the fixed scratch address; the
    /// per-step pointer values are uploaded here, OUTSIDE the captured region,
    /// before each capture and replay. Rotating across 4 event-fenced pinned
    /// host slots guarantees pipelined decode steps cannot overwrite an
    /// in-flight async H2D DMA source buffer.
    ///
    /// Layout mirrors `decode_multi_seq_inner` exactly:
    ///   `[h_state_ptrs[N] u64, conv_state_ptrs[N] u64]`
    /// `h_state_ptrs` at offset 0, `conv_state_ptrs` at offset `N * 8`.
    pub(crate) fn refresh_multi_seq_ptr_table<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        num_seqs: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            num_seqs * 2 <= MULTI_SEQ_PTR_MAX_U64,
            "multi_seq_ptr_host capacity exceeded: {num_seqs}*2 > {MULTI_SEQ_PTR_MAX_U64}",
        );
        // Invalidate before touching any pinned bytes. If pointer extraction or
        // the H2D enqueue fails, a later native frame must not mistake a
        // partially rewritten host table for device-resident state.
        self.multi_seq_ptr_uploaded_rows
            .store(0, std::sync::atomic::Ordering::Release);

        let upload_bytes = num_seqs * 2 * std::mem::size_of::<u64>();
        let mut ring_guard = self
            .multi_seq_ptr_pinned_ring
            .lock()
            .map_err(|_| anyhow::anyhow!("multi_seq_ptr_pinned_ring lock poisoned"))?;
        if ring_guard.is_none() {
            *ring_guard = MultiSeqPtrPinnedRing::try_allocate(gpu);
        }

        if let Some(ring) = ring_guard.as_mut() {
            let slot_idx = ring.next_slot;
            ring.wait_slot(slot_idx, gpu)?;
            let byte_off = slot_idx * MULTI_SEQ_PTR_SLOT_BYTES;
            {
                let slot_bytes = &mut ring.buf.as_mut_slice()[byte_off..byte_off + upload_bytes];
                for (i, state) in states.iter_mut().enumerate().take(num_seqs) {
                    let ssm_state = state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                    let h_off = i * std::mem::size_of::<u64>();
                    let c_off = (num_seqs + i) * std::mem::size_of::<u64>();
                    slot_bytes[h_off..h_off + 8]
                        .copy_from_slice(&ssm_state.h_state.0.to_ne_bytes());
                    slot_bytes[c_off..c_off + 8]
                        .copy_from_slice(&ssm_state.conv_state.0.to_ne_bytes());
                }
            }
            let ptr_bytes = ring.buf.pinned_slice_range(byte_off, upload_bytes)?;
            // Mark the slot uncertain before enqueuing the async DMA so that if
            // `record_event` fails (or the backend has no events), the next
            // wrap onto `slot_idx` synchronizes `stream` before mutating it.
            ring.slot_uncertain_stream[slot_idx] = Some(stream);
            // SAFETY: `ring.buf` is owned by this layer for the model lifetime
            // (and `TransformerModel::drop` synchronizes streams before drop).
            // `ring.wait_slot(slot_idx, gpu)` proved any previous async H2D from
            // this slot has completed, and `slot_uncertain_stream` / `slot_events`
            // fence future reuses of `slot_idx` until this transfer completes.
            unsafe {
                gpu.copy_h2d_pinned_async(ptr_bytes, self.ssm_multi_seq_ptr_scratch, stream)?;
            }
            let event = ring.slot_events[slot_idx];
            if event != 0 {
                if let Err(err) = gpu.record_event(event, stream) {
                    ring.poisoned = true;
                    return Err(err);
                }
                ring.slot_uncertain_stream[slot_idx] = None;
                ring.slot_busy[slot_idx] = true;
            }
            ring.last_slot = slot_idx;
            ring.next_slot = (slot_idx + 1) % MULTI_SEQ_PTR_PINNED_SLOTS;
            self.multi_seq_ptr_uploaded_rows
                .store(num_seqs, std::sync::atomic::Ordering::Release);
            return Ok(());
        }
        drop(ring_guard);

        // Fallback when backend cannot allocate an additional pinned ring:
        // use the pre-allocated per-layer staging buffer with a synchronous
        // H2D upload after draining `stream` so the single-slot buffer can
        // never race an in-flight GPU DMA after the mutex guard drops.
        let mut ptr_buf = if num_seqs * 2 <= 64 {
            self.multi_seq_ptr_host.lock()
        } else {
            self.multi_seq_ptr_wide_host.lock()
        };
        {
            let slot_bytes = &mut ptr_buf.as_mut_slice()[..upload_bytes];
            for (i, state) in states.iter_mut().enumerate().take(num_seqs) {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let h_off = i * std::mem::size_of::<u64>();
                let c_off = (num_seqs + i) * std::mem::size_of::<u64>();
                slot_bytes[h_off..h_off + 8].copy_from_slice(&ssm_state.h_state.0.to_ne_bytes());
                slot_bytes[c_off..c_off + 8].copy_from_slice(&ssm_state.conv_state.0.to_ne_bytes());
            }
        }
        gpu.synchronize(stream)?;
        gpu.copy_h2d(
            &ptr_buf.as_mut_slice()[..upload_bytes],
            self.ssm_multi_seq_ptr_scratch,
        )?;
        self.multi_seq_ptr_uploaded_rows
            .store(num_seqs, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Native eager `C2..=QWEN4_NATIVE_MAX_ROWS` variant of
    /// [`Self::refresh_multi_seq_ptr_table`].
    ///
    /// Recurrent state allocations are stable for a request's lifetime, so a
    /// scheduler wave with the same active rows in the same order has the same
    /// table on every token. Reuse is exact: both row count and every h/conv
    /// pointer must match the last successfully submitted pinned table. A
    /// mismatch delegates to the ordinary uploader. Device contents are never
    /// inferred from slot ids or sequence lengths.
    ///
    /// Returns `true` only when the H2D submission was safely elided.
    pub(crate) fn refresh_native_multi_seq_ptr_table_cached<'a, 'b: 'a>(
        &self,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        num_seqs: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        anyhow::ensure!(
            num_seqs * 2 <= MULTI_SEQ_PTR_MAX_U64,
            "multi_seq_ptr_host capacity exceeded: {num_seqs}*2 > {MULTI_SEQ_PTR_MAX_U64}",
        );
        if self
            .multi_seq_ptr_uploaded_rows
            .load(std::sync::atomic::Ordering::Acquire)
            == num_seqs
        {
            let mut ring_guard = self
                .multi_seq_ptr_pinned_ring
                .lock()
                .map_err(|_| anyhow::anyhow!("multi_seq_ptr_pinned_ring lock poisoned"))?;
            if let Some(ring) = ring_guard.as_mut() {
                let byte_off = ring.last_slot * MULTI_SEQ_PTR_SLOT_BYTES;
                let upload_bytes = num_seqs * 2 * std::mem::size_of::<u64>();
                let end = byte_off
                    .checked_add(upload_bytes)
                    .ok_or_else(|| anyhow::anyhow!("multi-seq pointer ring range overflow"))?;
                anyhow::ensure!(
                    end <= ring.buf.len(),
                    "multi-seq pointer ring slice exceeds allocation: {byte_off} + {upload_bytes} > {}",
                    ring.buf.len()
                );
                let slot_bytes = &ring.buf.as_mut_slice()[byte_off..end];
                let mut identical = true;
                for (row, state) in states.iter().enumerate().take(num_seqs) {
                    let ssm_state = state
                        .as_any()
                        .downcast_ref::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {row}"))?;
                    let h_off = row * 8;
                    let c_off = (num_seqs + row) * 8;
                    let h_prev = u64::from_ne_bytes(
                        slot_bytes[h_off..h_off + 8]
                            .try_into()
                            .expect("8-byte u64 slice"),
                    );
                    let c_prev = u64::from_ne_bytes(
                        slot_bytes[c_off..c_off + 8]
                            .try_into()
                            .expect("8-byte u64 slice"),
                    );
                    if h_prev != ssm_state.h_state.0 || c_prev != ssm_state.conv_state.0 {
                        identical = false;
                        break;
                    }
                }
                if identical {
                    return Ok(true);
                }
            } else {
                drop(ring_guard);
                let mut ptr_buf = if num_seqs * 2 <= 64 {
                    self.multi_seq_ptr_host.lock()
                } else {
                    self.multi_seq_ptr_wide_host.lock()
                };
                let upload_bytes = num_seqs * 2 * std::mem::size_of::<u64>();
                anyhow::ensure!(
                    upload_bytes <= ptr_buf.len(),
                    "multi-seq pointer staging slice exceeds allocation: {upload_bytes} > {}",
                    ptr_buf.len()
                );
                let raw = &ptr_buf.as_mut_slice()[..upload_bytes];
                let mut slots = [0u64; MULTI_SEQ_PTR_MAX_U64];
                for (dst, chunk) in slots[..num_seqs * 2].iter_mut().zip(raw.chunks_exact(8)) {
                    *dst = u64::from_ne_bytes(chunk.try_into().expect("8-byte u64 slice"));
                }
                let mut identical = true;
                for (row, state) in states.iter().enumerate().take(num_seqs) {
                    let ssm_state = state
                        .as_any()
                        .downcast_ref::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {row}"))?;
                    if slots[row] != ssm_state.h_state.0
                        || slots[num_seqs + row] != ssm_state.conv_state.0
                    {
                        identical = false;
                        break;
                    }
                }
                if identical {
                    return Ok(true);
                }
            }
        }

        self.refresh_multi_seq_ptr_table(states, num_seqs, gpu, stream)?;
        Ok(false)
    }

    /// Per-sequence single-decode fallback. Used when FP32 conv1d / GDN
    /// kernels aren't loaded (e.g. Metal backend) — the BF16 path uses
    /// `attn_output` for GDN output, which the batched layout above can't
    /// safely accommodate for N>1 without overflow.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_per_seq_fallback<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        for i in 0..num_seqs {
            let hidden_i = hidden.offset(i * h * residual_elem);
            let residual_i = residual.offset(i * h * residual_elem);
            self.decode(
                hidden_i,
                residual_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut block_tables[i].clone(),
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MultiSeqNvfp4ProjRoute, SmEnvOverride, parse_sm_env_override, plan_multi_seq_ffn_chunks,
        resolve_cublas_ba_for_sm, resolve_cublas_ba_for_sm_with_policy, resolve_ssm_proj_tc_for_sm,
        resolve_ssm_proj_tc_for_sm_with_policy, select_multi_seq_nvfp4_proj_route,
        select_multi_seq_nvfp4_proj_route_for_sm,
        select_multi_seq_nvfp4_proj_route_for_sm_with_policy,
    };
    use crate::layers::deterministic_batch::DeterministicBatchPolicy;

    #[test]
    fn deterministic_batch_pins_ssm_multi_seq_routes_across_1_to_128() {
        for sm in [48u32, 188u32] {
            for m in [1u32, 2, 4, 8, 16, 32, 64, 128] {
                for ov in [
                    SmEnvOverride::Auto,
                    SmEnvOverride::Enabled,
                    SmEnvOverride::Disabled,
                ] {
                    assert!(!resolve_ssm_proj_tc_for_sm_with_policy(
                        ov,
                        sm,
                        m,
                        DeterministicBatchPolicy::Enabled,
                    ));
                    assert!(!resolve_cublas_ba_for_sm_with_policy(
                        ov,
                        sm,
                        m as usize,
                        DeterministicBatchPolicy::Enabled,
                    ));
                }
                let expected = if m == 1 {
                    MultiSeqNvfp4ProjRoute::SerialGemv
                } else {
                    MultiSeqNvfp4ProjRoute::ExactRows
                };
                assert_eq!(
                    select_multi_seq_nvfp4_proj_route_for_sm_with_policy(
                        m,
                        5_120,
                        sm,
                        true,
                        true,
                        4,
                        true,
                        true,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    expected,
                    "M={m} sm={sm}"
                );
            }
        }
    }

    #[test]
    fn sm120_auto_defaults_leave_gb10_untouched() {
        // GB10 (48 SMs <= 64): Auto stays off for both TC proj and cuBLASLt BA.
        assert!(!resolve_ssm_proj_tc_for_sm(SmEnvOverride::Auto, 48, 32));
        assert!(!resolve_ssm_proj_tc_for_sm(SmEnvOverride::Auto, 64, 64));
        assert!(!resolve_cublas_ba_for_sm(SmEnvOverride::Auto, 48, 32));
        assert!(!resolve_cublas_ba_for_sm(SmEnvOverride::Auto, 64, 64));

        // SM120 (188 SMs > 64): Auto enables TC proj at M >= 16 and cuBLASLt BA at M >= 4.
        assert!(!resolve_ssm_proj_tc_for_sm(SmEnvOverride::Auto, 188, 8));
        assert!(resolve_ssm_proj_tc_for_sm(SmEnvOverride::Auto, 188, 16));
        assert!(resolve_ssm_proj_tc_for_sm(SmEnvOverride::Auto, 188, 64));
        assert!(!resolve_cublas_ba_for_sm(SmEnvOverride::Auto, 188, 2));
        assert!(resolve_cublas_ba_for_sm(SmEnvOverride::Auto, 188, 4));
        assert!(resolve_cublas_ba_for_sm(SmEnvOverride::Auto, 188, 32));

        // Explicit overrides always win.
        assert!(!resolve_ssm_proj_tc_for_sm(
            SmEnvOverride::Disabled,
            188,
            64
        ));
        assert!(resolve_ssm_proj_tc_for_sm(SmEnvOverride::Enabled, 48, 8));
        assert!(!resolve_cublas_ba_for_sm(SmEnvOverride::Disabled, 188, 64));
        assert!(resolve_cublas_ba_for_sm(SmEnvOverride::Enabled, 48, 4));
        assert_eq!(parse_sm_env_override(Some("0")), SmEnvOverride::Disabled);
        assert_eq!(parse_sm_env_override(Some("1")), SmEnvOverride::Enabled);
        assert_eq!(parse_sm_env_override(None), SmEnvOverride::Auto);

        // On SM120, C=64 with N=5120 prefers m32_n64 (160 CTAs) over m128 (40 CTAs),
        // while GB10 keeps m128 unchanged.
        assert_eq!(
            select_multi_seq_nvfp4_proj_route_for_sm(
                64, 5120, 188, true, true, 4, true, true, true, true, true
            ),
            MultiSeqNvfp4ProjRoute::TcM32N64
        );
        assert_eq!(
            select_multi_seq_nvfp4_proj_route_for_sm(
                64, 5120, 48, true, true, 4, true, true, true, true, true
            ),
            MultiSeqNvfp4ProjRoute::TcM128
        );
    }

    #[test]
    fn multi_seq_nvfp4_proj_route_covers_c2_to_c128_exact_and_tc() {
        // Default exact mode (tc_enabled=false): C=2..=128 always routes to single-sweep exact rows.
        for rows in [2u32, 4, 8, 16, 32, 48, 64, 128] {
            assert_eq!(
                select_multi_seq_nvfp4_proj_route(
                    rows, false, true, 4, true, true, true, true, true
                ),
                MultiSeqNvfp4ProjRoute::ExactRows,
                "rows={rows}"
            );
        }

        // TC mode with transposed weights: C<=32 uses m32_n64[_splitk], C=33..=128 uses m128.
        for rows in [2u32, 4, 16, 32] {
            assert_eq!(
                select_multi_seq_nvfp4_proj_route(
                    rows, true, true, 4, true, true, true, true, true
                ),
                MultiSeqNvfp4ProjRoute::TcM32N64SplitK,
                "rows={rows}"
            );
            assert_eq!(
                select_multi_seq_nvfp4_proj_route(
                    rows, true, true, 0, true, true, true, true, true
                ),
                MultiSeqNvfp4ProjRoute::TcM32N64,
                "rows={rows}"
            );
        }
        for rows in [33u32, 48, 64, 128] {
            assert_eq!(
                select_multi_seq_nvfp4_proj_route(
                    rows, true, true, 4, true, true, true, true, true
                ),
                MultiSeqNvfp4ProjRoute::TcM128,
                "rows={rows}"
            );
            assert_eq!(
                select_multi_seq_nvfp4_proj_route(
                    rows, true, true, 4, true, true, false, true, true
                ),
                MultiSeqNvfp4ProjRoute::TcN128,
                "rows={rows}"
            );
        }

        // TC mode without transposed weights falls back cleanly to ExactRows rather than legacy GEMM.
        assert_eq!(
            select_multi_seq_nvfp4_proj_route(64, true, false, 4, true, true, true, true, true),
            MultiSeqNvfp4ProjRoute::ExactRows
        );
    }

    #[test]
    fn multi_seq_ffn_chunk_planner_covers_all_rows_without_singleton_tails() {
        assert!(plan_multi_seq_ffn_chunks(0, false).is_empty());
        assert!(plan_multi_seq_ffn_chunks(1, false).is_empty());

        for num_seqs in 2usize..=128 {
            let chunks = plan_multi_seq_ffn_chunks(num_seqs, false);
            assert!(!chunks.is_empty(), "num_seqs={num_seqs}");
            let mut cursor = 0;
            for (start, len) in chunks {
                assert_eq!(start, cursor, "num_seqs={num_seqs}");
                assert!(
                    (2..=crate::layers::TC_VERIFY_MAX_ROWS).contains(&len),
                    "invalid chunk len={len} for num_seqs={num_seqs}"
                );
                cursor += len;
            }
            assert_eq!(cursor, num_seqs);
        }

        // Specific boundary checks.
        assert_eq!(plan_multi_seq_ffn_chunks(32, false), vec![(0, 32)]);
        assert_eq!(
            plan_multi_seq_ffn_chunks(33, false),
            vec![(0, 17), (17, 16)]
        );
        assert_eq!(
            plan_multi_seq_ffn_chunks(48, false),
            vec![(0, 32), (32, 16)]
        );
        assert_eq!(
            plan_multi_seq_ffn_chunks(64, false),
            vec![(0, 32), (32, 32)]
        );
        assert_eq!(
            plan_multi_seq_ffn_chunks(65, false),
            vec![(0, 32), (32, 17), (49, 16)]
        );
        // Wide TC mode dispatches C=48/64 in a single shot.
        assert_eq!(plan_multi_seq_ffn_chunks(48, true), vec![(0, 48)]);
        assert_eq!(plan_multi_seq_ffn_chunks(64, true), vec![(0, 64)]);
    }
}
