// SPDX-License-Identifier: AGPL-3.0-only
//
// `ATLAS_QWEN4_DENSE_DEDUP=1` (default off): release the dense-projection
// copies Flash-Next construction leaves behind once the NVFP4 operands every
// runtime route reads exist.
//
// Flash-Next ships its dense projections as ModelOpt MXFP8 (E4M3 + E8M0
// per-32 scales). The loader dequantizes each to an owned BF16 buffer, then
// quantizes that to the NVFP4 copy the prefill, decode, verify and FP8
// weight-cache routes consume. Per layer, this selector releases:
//
// * attention (`Fp8Dequanted` / `Bf16Raw` attention arm only, when
//   `ATLAS_QWEN4_RELEASE_BF16_QKVZ=1`): the FP8 store originals of q/k/v/o,
//   the BF16 dequant of q/k/v (kept in `AttentionWeights` as a fallback no
//   qwen4 route takes once NVFP4 q/k/v weights exist) and the BF16 dequant
//   of o (`o_dense`). On Flash-Next MXFP8 checkpoints (`Standard` /
//   `CompressedTensors` attention arm), `mxfp8_release` handles releasing
//   the MXFP8 attention store originals and no BF16 attention fallbacks are
//   retained unless `ATLAS_QWEN4_ATTN_BF16=1`;
// * linear attention: the MXFP8 store originals of in_proj_qkv/z/a/b and
//   out_proj (their BF16 dequants are already freed by
//   `release_replaced_bf16_source`).
//
// It also releases the sources of the other runtime NVFP4 copies, whose
// originals the store otherwise keeps for the whole process:
//
// * hyper-connection mixers: the BF16 `input_mix_weight_down/up` and
//   `block_inject_weight` of every layer's attn/mlp hyper-connection and of
//   the model-level mixer (`Qwen4HyperConnection` holds only NVFP4 copies
//   when `ATLAS_QWEN4_HC_BF16` is off; when `ATLAS_QWEN4_HC_BF16=1` or
//   `ATLAS_QWEN4_PREFILL_NATIVE=1`, `source_pins` pins the BF16 sources);
// * MoE routers: the BF16 `mlp.gate.weight`, once the NVFP4 router exists
//   (the BF16 router arms run only without an NVFP4 router);
// * shared experts: the MXFP8 store originals of gate/up/down (their BF16
//   dequants are already freed by `quantized_modelopt_mixed`).
//
// None of these bytes is read after construction, so releasing them cannot
// change what any kernel reads. Every release is tombstoned in the store
// (later lookups fail closed) and every dropped BF16 fallback is nulled; the
// fallback branches `ensure!` a non-null weight, so a route that would have
// read a released copy errors instead of reading freed memory. Frees run in
// the same quiescent per-layer construction window, after a stream sync, as
// the established releases in this loader (BUG #29).

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

pub(crate) const DENSE_DEDUP_SELECTOR: &str = "ATLAS_QWEN4_DENSE_DEDUP";

// Readers of the sources dense dedup releases (the native-dense BF16, FP8
// and prefill controls) no longer refuse dedup: each release primitive asks
// `source_pins` and skips what a reader pins (MXFP8_RELEASE_SKIPPED).
/// Strict `1` parse of the dedup selector.
pub(super) fn parse_selected(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

pub(super) fn selected() -> bool {
    parse_selected(std::env::var(DENSE_DEDUP_SELECTOR).ok().as_deref())
}

/// Dedup applies to target decoder layers only, never the `mtp.*` layer.
pub(super) fn is_target_layer_prefix(layer_prefix: &str) -> bool {
    !layer_prefix.starts_with("mtp.")
}

/// Scale tensors a block-scaled FP8 projection carries next to `.weight`.
const FP8_SCALE_SUFFIXES: [&str; 2] = ["weight_scale", "weight_scale_inv"];

/// Release the FP8 (MXFP8 / block-scaled) store original of `{prefix}` and
/// its scale tensors. Returns the bytes released. A non-FP8 `.weight` (a BF16
/// alias, or packed NVFP4 the runtime may read in place) is left alone and
/// returns 0: only a source the loader is known to have converted is freed.
pub(super) fn release_fp8_store_source(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let weight_name = format!("{prefix}.weight");
    if !store.contains(&weight_name) {
        // Already consumed: a BF16 alias released as the replaced source.
        return Ok(0);
    }
    let (ptr, dtype) = {
        let weight = store
            .get(&weight_name)
            .with_context(|| format!("dense dedup: {weight_name} not in store"))?;
        (weight.ptr, weight.dtype)
    };
    if dtype != WeightDtype::FP8E4M3 {
        return Ok(0);
    }
    let mut bytes = store.get(&weight_name)?.byte_size();
    for suffix in FP8_SCALE_SUFFIXES {
        let name = format!("{prefix}.{suffix}");
        if store.contains(&name) {
            bytes += store.get(&name)?.byte_size();
        }
    }
    if crate::source_pins::skip_if_pinned(prefix, bytes, crate::source_pins::PinKind::Mxfp8Source)?
        .is_some()
    {
        return Ok(0);
    }
    let mut released = store.release_consumed_alias(&weight_name, ptr, dtype, gpu)?;
    for suffix in FP8_SCALE_SUFFIXES {
        let name = format!("{prefix}.{suffix}");
        if !store.contains(&name) {
            continue;
        }
        let (ptr, dtype) = {
            let scale = store.get(&name)?;
            (scale.ptr, scale.dtype)
        };
        released += store.release_consumed_alias(&name, ptr, dtype, gpu)?;
    }
    Ok(released)
}

/// Release the sources of a hyper-connection mixer whose NVFP4 copies are
/// built. `sources` pairs each projection prefix (`{prefix}.weight` in the
/// store) with the `DenseWeight` the quantizer read: a BF16 store alias is
/// tombstoned; an owned dequant (an FP8 checkpoint) is freed together with its
/// FP8 store original. Returns the bytes released.
pub(super) fn release_bf16_sources(
    store: &WeightStore,
    sources: &[(String, crate::weight_map::DenseWeight)],
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let mut released = 0usize;
    for (prefix, dense) in sources {
        let name = format!("{prefix}.weight");
        if !store.contains(&name) {
            continue;
        }
        let logical = store.get(&name)?.num_elements() * 2;
        if crate::source_pins::skip_if_pinned(
            prefix,
            logical,
            crate::source_pins::PinKind::Bf16Alias,
        )?
        .is_some()
        {
            continue;
        }
        released += super::linear_attn_arms::release_replaced_bf16_source(
            store, &name, *dense, logical, gpu,
        )?;
        released += release_fp8_store_source(store, prefix, gpu)?;
    }
    Ok(released)
}

/// Release a MoE router's BF16 store alias once its NVFP4 copy exists, and
/// null the layer's BF16 router so its arms fail closed. Without an NVFP4
/// router the BF16 one is the only router and is kept.
pub(super) fn release_router_bf16(
    store: &WeightStore,
    layer_prefix: &str,
    gate: &mut crate::weight_map::DenseWeight,
    has_nvfp4_router: bool,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    if !has_nvfp4_router {
        return Ok(0);
    }
    let name = format!("{layer_prefix}.mlp.gate.weight");
    if !store.contains(&name) {
        return Ok(0);
    }
    let logical = store.get(&name)?.num_elements() * 2;
    if crate::source_pins::skip_if_pinned(&name, logical, crate::source_pins::PinKind::Bf16Alias)?
        .is_some()
    {
        return Ok(0);
    }
    let released = store.release_consumed_alias(&name, gate.weight, WeightDtype::BF16, gpu)?;
    if released != 0 {
        gate.weight = spark_runtime::gpu::DevicePtr::NULL;
    }
    Ok(released)
}

/// Release a shared expert's MXFP8 gate/up/down store originals.
pub(super) fn release_shared_expert_sources(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<usize> {
    let mut released = 0usize;
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        released += release_fp8_store_source(
            store,
            &format!("{layer_prefix}.mlp.shared_expert.{proj}"),
            gpu,
        )?;
    }
    Ok(released)
}

/// Per-layer byte accounting for the load log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DedupBytes {
    pub store_originals: usize,
    pub bf16_copies: usize,
}

impl DedupBytes {
    pub(super) fn total(&self) -> usize {
        self.store_originals + self.bf16_copies
    }
}

pub(super) fn log_layer(layer: &str, family: &str, bytes: DedupBytes) {
    crate::source_pins::flush(layer, family);
    tracing::info!(
        "QWEN4_DENSE_DEDUP_RELEASED selector={DENSE_DEDUP_SELECTOR} layer={layer} family={family} store_originals_mib={:.2} bf16_copies_mib={:.2} total_mib={:.2}",
        bytes.store_originals as f64 / (1024.0 * 1024.0),
        bytes.bf16_copies as f64 / (1024.0 * 1024.0),
        bytes.total() as f64 / (1024.0 * 1024.0),
    );
}

#[cfg(test)]
#[path = "dense_dedup_tests.rs"]
mod tests;
