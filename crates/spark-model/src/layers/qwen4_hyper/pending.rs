// SPDX-License-Identifier: AGPL-3.0-only

//! The one saved-scale hyperconnection injection a Flash-Next prefill may
//! owe at a time (`ATLAS_QWEN4_PREFILL_MHC_FUSED`).
//!
//! A deferring hyperconnection records it here instead of launching
//! `qwen4_hc_inject_saved`. The next preparation of the same rows fuses it
//! into its normalization; every other reader of the hyper rows (any other
//! hyperconnection method, PLE, diagnostics, the end of the layer loop)
//! applies it first with the unfused kernel, so the hyper state a reader sees
//! is always the injected one. State is per thread: one prefill runs on one
//! thread, and the layer loop discards a stale entry left by a failed chunk.

use std::cell::Cell;
use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// `hyper[t, s, h] += core[t, h] * scale[t, s]`, with the scales saved in the
/// tail of each `residual` row by `qwen4_hc_save_inject`. With `combine`, the
/// MoE output in `core` has not been formed yet (lever `combine`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingInject {
    pub hyper: DevicePtr,
    pub core: DevicePtr,
    pub residual: DevicePtr,
    pub rows: usize,
    pub combine: Option<MoeCombine>,
}

/// The two MoE steps that form its output rows, deferred:
/// `moe_unpermute_reduce_indexed` (routed slots, `expert_out[perm[t, k]]`
/// weighted by `topk_weights[t, k]`) then `moe_batched_blend` (shared expert
/// gated by `sigmoid(gate_input[t] . shared_gate)`; null gate = 1).
///
/// With `slot_scales` set, the slot rows are MXFP8 (fn-moe's down epilogue):
/// `expert_out` holds e4m3 bytes `[slots, 2560]` and `slot_scales` e8m0 bytes
/// `[slots, 80]` (one per 32 columns, value = e4m3 * 2^(scale - 127)). With
/// `slot_corrections` set, `[slots, 80]` packed u32 records follow: two
/// lane-plus-E4M3 residual corrections per MX32 group under that same scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MoeCombine {
    pub slot_scales: DevicePtr,
    pub slot_corrections: DevicePtr,
    pub expert_out: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub topk_weights: DevicePtr,
    pub shared_out: DevicePtr,
    pub gate_input: DevicePtr,
    pub shared_gate: DevicePtr,
    pub gate_scalars: DevicePtr,
    pub topk: u32,
}

/// The fused path only defers canonical Flash-Next rows.
const HIDDEN: u32 = 2560;
const STREAMS: u32 = 4;

thread_local! {
    static PENDING: Cell<Option<PendingInject>> = const { Cell::new(None) };
    static COMBINE_SINK: Cell<Sink> = const { Cell::new(Sink::Off) };
}

#[derive(Clone, Copy)]
enum Sink {
    Off,
    Armed,
    Recorded(MoeCombine),
}

/// The MLP hyperconnection asks the MoE FFN it is about to run to leave its
/// output combine to the next hyperconnection preparation.
pub(crate) fn arm_combine_sink() {
    COMBINE_SINK.with(|c| c.set(Sink::Armed));
}

/// Called by the MoE FFN where it would combine its output: true means the
/// combine was recorded and must not run now.
pub(crate) fn record_combine(combine: MoeCombine) -> bool {
    COMBINE_SINK.with(|c| match c.get() {
        Sink::Armed => {
            c.set(Sink::Recorded(combine));
            true
        }
        _ => false,
    })
}

/// Disarm the sink and return the combine the FFN left, if it left one.
pub(crate) fn take_combine_sink() -> Option<MoeCombine> {
    match COMBINE_SINK.with(|c| c.replace(Sink::Off)) {
        Sink::Recorded(combine) => Some(combine),
        Sink::Off | Sink::Armed => None,
    }
}

fn moe_kernel(gpu: &dyn GpuBackend, name: &'static str) -> Result<KernelHandle> {
    gpu.kernel("moe", name)
        .map_err(|e| anyhow::anyhow!("pending MoE combine: {name} unavailable: {e}"))
}

/// True when [`materialize_combine`] can execute the fused vectorized 128-bit
/// `moe_unpermute_reduce_blend_mxfp8_bf16` kernel reading precomputed
/// `c.gate_scalars` directly.
///
/// Avoiding a second read of `c.gate_input` (`norm_output`) is mandatory for
/// correctness whenever `rows >= 1024` and `ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED=1`
/// (which runs `qwen4_mxfp8_quant_rows_v16q4_gate` with `discard.global.L2` on
/// `norm_output` after writing `c.gate_scalars`), and also fuses unpermute +
/// shared-expert blend into one launch.
#[inline]
#[must_use]
fn can_use_fused_v17_materialize(c: &MoeCombine, rows: usize) -> bool {
    rows > 0
        && !c.slot_scales.is_null()
        && c.slot_corrections.is_null()
        && !c.gate_scalars.is_null()
        && c.topk == 10
}

/// Form the deferred MoE output rows in `core` with the unfused kernels (or the
/// bit-exact fused v17 MXFP8 unpermute+blend kernel when `gate_scalars` are
/// already computed).
pub(crate) fn materialize_combine(
    c: &MoeCombine,
    core: DevicePtr,
    rows: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    static UNPERMUTE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static UNPERMUTE_MX: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static UNPERMUTE_MX_R2: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static UNPERMUTE_BLEND_MX: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static BLEND: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    if can_use_fused_v17_materialize(c, rows) {
        let k = UNPERMUTE_BLEND_MX
            .get_or_init(|| {
                gpu.kernel(
                    "qwen4_moe_compact_t_k32_v17",
                    "moe_unpermute_reduce_blend_mxfp8_bf16",
                )
                .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| {
                anyhow::anyhow!(
                    "pending MoE combine: qwen4_moe_compact_t_k32_v17::moe_unpermute_reduce_blend_mxfp8_bf16 unavailable: {e}"
                )
            })?;
        return KernelLaunch::new(gpu, k)
            .grid([rows as u32, 10, 1])
            .block([32, 1, 1])
            .arg_ptr(c.expert_out)
            .arg_ptr(c.slot_scales)
            .arg_ptr(c.shared_out)
            .arg_ptr(c.gate_scalars)
            .arg_ptr(core)
            .arg_ptr(c.token_to_perm)
            .arg_ptr(c.topk_weights)
            .arg_u32(HIDDEN)
            .arg_u32(rows as u32)
            .arg_u32(c.topk)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(DevicePtr::NULL)
            .launch(stream);
    }
    ensure!(
        c.gate_scalars.is_null(),
        "pending MoE combine: cannot fall back to moe_batched_blend(gate_input) when precomputed \
         gate_scalars are set (slot_scales={}, slot_corrections={}, topk={}, rows={rows}); \
         gate_input may be L2-discarded",
        !c.slot_scales.is_null(),
        !c.slot_corrections.is_null(),
        c.topk,
    );
    let get = |cell: &'static OnceLock<Result<KernelHandle, String>>, name| {
        cell.get_or_init(|| moe_kernel(gpu, name).map_err(|e| e.to_string()))
            .clone()
            .map_err(anyhow::Error::msg)
    };
    if c.slot_scales.is_null() {
        KernelLaunch::new(gpu, get(&UNPERMUTE, "moe_unpermute_reduce_indexed")?)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c.expert_out)
            .arg_ptr(core)
            .arg_ptr(c.token_to_perm)
            .arg_ptr(c.topk_weights)
            .arg_u32(HIDDEN)
            .arg_u32(rows as u32)
            .arg_u32(c.topk)
            .launch(stream)?;
    } else {
        let corrected = !c.slot_corrections.is_null();
        let (cell, name) = if corrected {
            (&UNPERMUTE_MX_R2, "qwen4_hc_moe_unpermute_mx_r2")
        } else {
            (&UNPERMUTE_MX, "qwen4_hc_moe_unpermute_mx")
        };
        let k = cell
            .get_or_init(|| {
                gpu.kernel("qwen4_hc_inject", name)
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(anyhow::Error::msg)?;
        if corrected {
            crate::model::qwen4_prefill_engagement::preflight_slot_r2(
                crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::Materialized,
                rows,
            )?;
        }
        let launch = KernelLaunch::new(gpu, k)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c.expert_out)
            .arg_ptr(c.slot_scales);
        let launch = if corrected {
            launch.arg_ptr(c.slot_corrections)
        } else {
            launch
        };
        launch
            .arg_ptr(core)
            .arg_ptr(c.token_to_perm)
            .arg_ptr(c.topk_weights)
            .arg_u32(rows as u32)
            .arg_u32(HIDDEN)
            .arg_u32(c.topk)
            .launch(stream)?;
        if corrected {
            crate::model::qwen4_prefill_engagement::commit_slot_r2(
                crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::Materialized,
            );
        }
    }
    KernelLaunch::new(gpu, get(&BLEND, "moe_batched_blend")?)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(core)
        .arg_ptr(c.shared_out)
        .arg_ptr(c.gate_input)
        .arg_ptr(c.shared_gate)
        .arg_u32(HIDDEN)
        .arg_u32(rows as u32)
        .launch(stream)
}

fn inject_saved_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hyper", "qwen4_hc_inject_saved")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("pending HC injection: kernel unavailable: {e}"))
}

/// Launch the unfused injection for `p` (forming a deferred MoE output first).
pub(crate) fn apply(p: &PendingInject, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    if let Some(c) = &p.combine {
        materialize_combine(c, p.core, p.rows, gpu, stream)?;
    }
    KernelLaunch::new(gpu, inject_saved_kernel(gpu)?)
        .grid([p.rows as u32, div_ceil(HIDDEN * STREAMS, 256), 1])
        .block([256, 1, 1])
        .arg_ptr(p.hyper)
        .arg_ptr(p.core)
        .arg_ptr(p.residual)
        .arg_u32(HIDDEN)
        .arg_u32(STREAMS)
        .launch(stream)
}

/// Record `p`. At most one injection is ever owed: a second one means a
/// reader skipped [`flush`], so fail closed instead of reordering them.
pub(crate) fn defer(p: PendingInject) -> Result<()> {
    if let Some(prior) = PENDING.with(|c| c.get()) {
        bail!("pending HC injection {prior:?} was never applied before deferring {p:?}");
    }
    PENDING.with(|c| c.set(Some(p)));
    Ok(())
}

pub(crate) fn take() -> Option<PendingInject> {
    PENDING.with(Cell::take)
}

/// Apply the owed injection, if any.
pub(crate) fn flush(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    match take() {
        Some(p) => apply(&p, gpu, stream),
        None => Ok(()),
    }
}

/// Drop an entry left behind by a chunk that failed before its loop ended.
/// Its buffers belong to that failed prefill; applying it would corrupt the
/// next one. Returns whether anything was dropped.
pub(crate) fn discard_stale() -> bool {
    take().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(rows: usize) -> PendingInject {
        PendingInject {
            hyper: DevicePtr(0x1000),
            core: DevicePtr(0x9000),
            residual: DevicePtr(0x5000),
            rows,
            combine: None,
        }
    }

    fn combine() -> MoeCombine {
        MoeCombine {
            slot_scales: DevicePtr::NULL,
            slot_corrections: DevicePtr::NULL,
            expert_out: DevicePtr(0x10_0000),
            token_to_perm: DevicePtr(0x20_0000),
            topk_weights: DevicePtr(0x30_0000),
            shared_out: DevicePtr(0x40_0000),
            gate_input: DevicePtr(0x50_0000),
            shared_gate: DevicePtr(0x60_0000),
            gate_scalars: DevicePtr::NULL,
            topk: 10,
        }
    }

    #[test]
    fn combine_sink_records_only_when_armed() {
        assert!(!record_combine(combine()));
        assert_eq!(take_combine_sink(), None);
        arm_combine_sink();
        assert_eq!(take_combine_sink(), None);
        arm_combine_sink();
        assert!(record_combine(combine()));
        assert!(!record_combine(combine()));
        assert_eq!(take_combine_sink(), Some(combine()));
        assert_eq!(take_combine_sink(), None);
    }

    #[test]
    fn take_empties_and_discard_reports() {
        assert!(!discard_stale());
        PENDING.with(|c| c.set(Some(entry(16))));
        assert_eq!(take(), Some(entry(16)));
        assert_eq!(take(), None);
        PENDING.with(|c| c.set(Some(entry(32))));
        assert!(discard_stale());
        assert!(!discard_stale());
    }

    #[test]
    fn fused_v17_materialize_requires_mxfp8_uncorrected_topk10_with_gate_scalars() {
        let mut c = combine();
        assert!(!can_use_fused_v17_materialize(&c, 1360));
        c.slot_scales = DevicePtr(0x70_0000);
        assert!(!can_use_fused_v17_materialize(&c, 1360));
        c.gate_scalars = DevicePtr(0x80_0000);
        assert!(can_use_fused_v17_materialize(&c, 1360));
        assert!(!can_use_fused_v17_materialize(&c, 0));
        c.slot_corrections = DevicePtr(0x90_0000);
        assert!(!can_use_fused_v17_materialize(&c, 1360));
        c.slot_corrections = DevicePtr::NULL;
        c.topk = 8;
        assert!(!can_use_fused_v17_materialize(&c, 1360));
    }
}
