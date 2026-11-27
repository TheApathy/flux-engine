// SPDX-License-Identifier: AGPL-3.0-only

//! W4A4 E2M1 activation scratch, checkpoint scale loading, and FlashInfer merged FFN prefill route.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result, ensure};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[cfg(feature = "cuda")]
use super::layer::DenseFfnWeights;
use super::layer::{DenseFfnLayer, FfnActivation};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::policy::{
    CublasLtNvfp4Policy, is_blackwell_dc_compute_capability, is_sm120_compute_capability,
    prefill_ffn_cublaslt_nvfp4_policy, supports_warp_nvfp4_mma,
};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// Scratch buffers for the inline BF16 → NVFP4 activation prequant
/// (W4A4 `nvfp4_nvfp4_gemm` fast path). The exact FlashInfer route
/// preallocates its maximum admitted shape during model construction; other
/// routes allocate on first prefill and resize in-place if M or K grows.
///
/// Sizes (per row M, per col K):
///   - `a_packed`: M × K/2 bytes (E2M1 nibbles)
///   - `a_scale`:  ceil(M/128)×128 × K/16 bytes (FP8 E4M3 scales; the
///     default layout uses the logical prefix while the optional
///     CUTLASS route uses the full padded 128x4 layout)
///   - `a_max`:    4 bytes (FP32 per-tensor absmax scratch)
///
/// One arena per `DenseFfnLayer` (= per transformer layer). Reused across
/// gate / up / down GEMMs within a layer: gate/up share K=H, down has
/// K=Inter, so the arena is sized for `max(H, Inter)` × `max_M`.
pub(super) struct E2m1Scratch {
    pub(super) a_packed: DevicePtr,
    pub(super) a_scale: DevicePtr,
    pub(super) a_max: DevicePtr,
    /// Current row capacity (M) the buffers can hold.
    pub(super) cap_m: usize,
    /// Current column capacity (K) the buffers can hold (full K, not K/2).
    pub(super) cap_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct E2m1ScratchLayout {
    pub(super) cap_m: usize,
    pub(super) cap_k: usize,
    pub(super) packed_bytes: usize,
    pub(super) scale_bytes: usize,
    pub(super) max_bytes: usize,
}

pub(super) fn e2m1_scratch_layout(m: usize, k: usize) -> Result<E2m1ScratchLayout> {
    ensure!(m > 0, "W4A4 activation scratch requires M > 0");
    ensure!(
        k >= 16 && k.is_multiple_of(16),
        "W4A4 activation scratch requires K >= 16 and K % 16 == 0; got {k}"
    );
    let cap_m = m.max(128);
    let cap_k = k;
    let elements = cap_m
        .checked_mul(cap_k)
        .context("W4A4 activation scratch size overflow")?;
    let packed_bytes = elements / 2;
    let scale_rows = cap_m
        .div_ceil(128)
        .checked_mul(128)
        .context("W4A4 activation scratch row-padding overflow")?;
    let scale_bytes = scale_rows
        .checked_mul(cap_k)
        .context("W4A4 padded activation-scale scratch size overflow")?
        / 16;
    Ok(E2m1ScratchLayout {
        cap_m,
        cap_k,
        packed_bytes,
        scale_bytes,
        max_bytes: std::mem::size_of::<f32>(),
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) struct E2m1CheckpointScales {
    /// SGLang's merged gate/up linear takes the maximum of the checkpoint's
    /// logical-partition scales before quantizing their shared input.
    pub(super) gate_up: f32,
    pub(super) down: f32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) struct FlashinferFfnProjection {
    pub(super) weight: DevicePtr,
    pub(super) weight_scales_128x4: DevicePtr,
    pub(super) weight_scales_hash: u64,
    pub(super) alpha_f32: DevicePtr,
    pub(super) input_scale_bits: u32,
    pub(super) alpha_bits: u32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) struct FlashinferMergedGateUp {
    pub(super) layer: usize,
    pub(super) weight: DevicePtr,
    pub(super) weight_scales_128x4: DevicePtr,
    pub(super) weight_hash: u64,
    pub(super) weight_scales_hash: u64,
    pub(super) alpha_f32: DevicePtr,
    pub(super) input_scale_bits: u32,
    pub(super) alpha_bits: u32,
    pub(super) source_gate_weight: DevicePtr,
    pub(super) source_up_weight: DevicePtr,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) struct FlashinferFfnPrefill {
    pub(super) layer: usize,
    pub(super) library: ops::flashinfer_sm121::FlashInferSm121,
    pub(super) merged_gate_up: FlashinferMergedGateUp,
    pub(super) down: FlashinferFfnProjection,
    pub(super) quantize_atlas_128x4_k: KernelHandle,
    pub(super) quantize_merged_silu_atlas_128x4_k: KernelHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlashinferFfnPrefillRoute {
    Disabled,
    Ineligible,
    Missing,
    Complete,
}

/// One W4A4/FlashInfer activation arena and stream binding shared by every
/// dense FFN layer of a model (`ATLAS_MEMLEAN_SHARED_SCRATCH`). Layers run
/// one after another on the one stream the binding admits, and each launch
/// group fully rewrites the rows it reads, so one arena serves all of them;
/// the per-layer copies were 64 x 80 MB on Qwen3.8-27B.
#[derive(Clone, Default)]
pub struct SharedE2m1Scratch {
    pub(super) scratch: Arc<Mutex<Option<E2m1Scratch>>>,
    pub(super) stream: Arc<Mutex<Option<u64>>>,
}

/// Packed-weight views of gate and up inside the merged gate-then-up
/// FlashInfer operand. `build_flashinfer_merged_gate_up` uploads the gate
/// bytes followed by the up bytes, so the two views read exactly the bytes
/// the separate checkpoint allocations held.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn merged_gate_up_views(
    merged: DevicePtr,
    gate_packed_bytes: usize,
) -> Result<(DevicePtr, DevicePtr)> {
    ensure!(!merged.is_null(), "merged gate/up operand is null");
    ensure!(
        gate_packed_bytes > 0 && gate_packed_bytes.is_multiple_of(256),
        "gate packed extent {gate_packed_bytes} must be a positive multiple of 256 bytes"
    );
    Ok((merged, merged.offset(gate_packed_bytes)))
}

/// `m_qualified` is supplied by the caller rather than matched inline so the
/// extended ladder (`ATLAS_FLASHINFER_FFN_EXTRA_M=1`) shares one source of truth
/// with `qwen38_ffn_launch_candidate`; the frozen pair stays the default.
pub(super) const fn flashinfer_ffn_prefill_route(
    requested: bool,
    exact_qwen38_dense: bool,
    m_qualified: bool,
    prepared: bool,
) -> FlashinferFfnPrefillRoute {
    if !requested {
        FlashinferFfnPrefillRoute::Disabled
    } else if !exact_qwen38_dense || !m_qualified {
        FlashinferFfnPrefillRoute::Ineligible
    } else if !prepared {
        FlashinferFfnPrefillRoute::Missing
    } else {
        FlashinferFfnPrefillRoute::Complete
    }
}

pub(super) const fn e2m1_kmajor_projection_shape(m: u32, n: u32, k: u32) -> bool {
    m >= 128 && n.is_multiple_of(128) && k.is_multiple_of(64)
}

pub(super) const E2M1_KMAJOR_M256_MIN_M: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum E2m1KmajorKernel {
    M128,
    M256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum E2m1Scope {
    Full,
    DownOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum E2m1PrefillRoute {
    Disabled,
    Ineligible,
    Complete(E2m1Scope),
    Missing(E2m1Scope),
}

pub(super) const fn e2m1_prefill_route(
    full_requested: bool,
    down_requested: bool,
    shape_eligible: bool,
    full_ready: bool,
    down_ready: bool,
) -> E2m1PrefillRoute {
    let scope = if full_requested {
        E2m1Scope::Full
    } else if down_requested {
        E2m1Scope::DownOnly
    } else {
        return E2m1PrefillRoute::Disabled;
    };
    if !shape_eligible {
        return E2m1PrefillRoute::Ineligible;
    }
    match scope {
        E2m1Scope::Full if full_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::DownOnly if down_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::Full | E2m1Scope::DownOnly => E2m1PrefillRoute::Missing(scope),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum E2m1SelectedKernel {
    RowMajorM64,
    KmajorM128,
    KmajorM256,
}

pub(super) const fn e2m1_selected_kernel(
    kmajor: bool,
    m256_requested: bool,
    m: u32,
) -> E2m1SelectedKernel {
    if !kmajor {
        E2m1SelectedKernel::RowMajorM64
    } else if matches!(
        e2m1_kmajor_kernel(m, m256_requested),
        E2m1KmajorKernel::M256
    ) {
        E2m1SelectedKernel::KmajorM256
    } else {
        E2m1SelectedKernel::KmajorM128
    }
}

pub(super) const fn e2m1_kmajor_kernel(m: u32, m256_requested: bool) -> E2m1KmajorKernel {
    if m256_requested && m >= E2M1_KMAJOR_M256_MIN_M {
        E2m1KmajorKernel::M256
    } else {
        E2m1KmajorKernel::M128
    }
}

pub(super) fn validated_e2m1_checkpoint_scales(
    gate: f32,
    up: f32,
    down: f32,
) -> Result<E2m1CheckpointScales> {
    for (name, value) in [("gate", gate), ("up", up), ("down", down)] {
        ensure!(
            value.is_finite() && value > 0.0,
            "ATLAS_E2M1_STATIC_SCALE requires a finite positive {name} input_scale; got {value}"
        );
    }
    Ok(E2m1CheckpointScales {
        gate_up: gate.max(up),
        down,
    })
}

pub(super) fn read_e2m1_input_scale(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    projection: &str,
) -> Result<f32> {
    ensure!(
        ptr != DevicePtr::NULL,
        "ATLAS_E2M1_STATIC_SCALE requires {projection}.input_scale in the checkpoint"
    );
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(ptr, &mut bytes)
        .with_context(|| format!("read {projection}.input_scale for ATLAS_E2M1_STATIC_SCALE"))?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn flashinfer_scale_fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn ensure_flashinfer_ffn_disjoint(
    left: DevicePtr,
    left_len: usize,
    right: DevicePtr,
    right_len: usize,
) -> Result<()> {
    let left_end = left
        .0
        .checked_add(u64::try_from(left_len).context("left FFN pointer extent exceeds u64")?)
        .context("left FFN pointer range overflow")?;
    let right_end = right
        .0
        .checked_add(u64::try_from(right_len).context("right FFN pointer extent exceeds u64")?)
        .context("right FFN pointer range overflow")?;
    ensure!(
        left_end <= right.0 || right_end <= left.0,
        "FlashInfer FFN buffers overlap"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn flashinfer_ffn_extent(
    rows: usize,
    cols: usize,
    element_bytes: usize,
) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .context("FlashInfer FFN buffer extent overflow")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn flashinfer_ffn_activation_scale_extent(rows: usize, cols: usize) -> Result<usize> {
    use crate::weight_map::cutlass_scale_layout::{CUTLASS_SCALE_ROW_TILE, NVFP4_GROUP_SIZE};

    ensure!(
        rows > 0,
        "FlashInfer FFN activation-scale extent requires rows > 0"
    );
    ensure!(
        cols >= NVFP4_GROUP_SIZE && cols.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN activation-scale extent requires cols divisible by {NVFP4_GROUP_SIZE}"
    );
    let padded_rows = rows
        .checked_add(CUTLASS_SCALE_ROW_TILE - 1)
        .context("FlashInfer FFN activation-scale row padding overflow")?
        / CUTLASS_SCALE_ROW_TILE;
    let padded_rows = padded_rows
        .checked_mul(CUTLASS_SCALE_ROW_TILE)
        .context("FlashInfer FFN padded activation-scale rows overflow")?;
    flashinfer_ffn_extent(padded_rows, cols / NVFP4_GROUP_SIZE, 1)
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn read_flashinfer_ffn_checkpoint_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<(
    Vec<u8>,
    Vec<u8>,
    crate::weight_map::modelopt_scale_admission::AdmittedModeloptScale,
)> {
    use crate::weight_map::cutlass_scale_layout::NVFP4_GROUP_SIZE;
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && !weight.input_scale.is_null(),
        "FlashInfer FFN {projection:?} requires original non-NULL ModelOpt operands"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN {projection:?} has invalid N/K geometry"
    );
    let packed_len = n
        .checked_mul(k / 2)
        .context("FlashInfer FFN packed-weight length overflow")?;
    let scale_len = n
        .checked_mul(k / NVFP4_GROUP_SIZE)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut packed = vec![0_u8; packed_len];
    let mut logical_scales = vec![0_u8; scale_len];
    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.weight, &mut packed)
        .with_context(|| format!("read ModelOpt FFN {projection:?} packed weight"))?;
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .with_context(|| format!("read ModelOpt FFN {projection:?} block scales"))?;
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .with_context(|| format!("read ModelOpt FFN {projection:?} input scale"))?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    Ok((packed, logical_scales, input_scale))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_merged_gate_up(
    gpu: &dyn GpuBackend,
    layer: usize,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Result<FlashinferMergedGateUp> {
    use crate::weight_map::flashinfer_ffn_admission::{
        QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnCheckpointProjection, Qwen38FfnProjection,
        Qwen38FfnSource, admit_qwen38_merged_gate_up,
    };

    let (gate_packed, gate_scales, gate_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Gate,
        gate,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let (up_packed, up_scales, up_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Up,
        up,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let admitted = admit_qwen38_merged_gate_up(
        layer,
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Gate,
            },
            packed_weight: &gate_packed,
            logical_weight_scales: &gate_scales,
            input_scale: gate_input_scale,
            weight_scale_2_le_bytes: gate.weight_scale_2.to_le_bytes(),
        },
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Up,
            },
            packed_weight: &up_packed,
            logical_weight_scales: &up_scales,
            input_scale: up_input_scale,
            weight_scale_2_le_bytes: up.weight_scale_2.to_le_bytes(),
        },
    )?;

    let merged_weight = gpu.alloc(admitted.packed_weight().len())?;
    gpu.copy_h2d(admitted.packed_weight(), merged_weight)
        .context("upload immutable merged FFN gate/up packed weight")?;
    let merged_scales = gpu.alloc(admitted.physical_weight_scales().len())?;
    gpu.copy_h2d(admitted.physical_weight_scales(), merged_scales)
        .context("upload immutable merged FFN gate/up physical scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&admitted.shared_alpha_bits().to_le_bytes(), alpha_f32)
        .context("upload immutable merged FFN gate/up alpha")?;

    Ok(FlashinferMergedGateUp {
        layer,
        weight: merged_weight,
        weight_scales_128x4: merged_scales,
        weight_hash: flashinfer_scale_fingerprint(admitted.packed_weight()),
        weight_scales_hash: flashinfer_scale_fingerprint(admitted.physical_weight_scales()),
        alpha_f32,
        input_scale_bits: admitted.input_scales()[0].value_bits(),
        alpha_bits: admitted.shared_alpha_bits(),
        source_gate_weight: gate.weight,
        source_up_weight: up.weight,
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_ffn_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<FlashinferFfnProjection> {
    use crate::weight_map::cutlass_scale_layout::{
        NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
    };
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null(),
        "FlashInfer FFN packed weight is null"
    );
    ensure!(
        !weight.weight_scale.is_null(),
        "FlashInfer FFN block-scale pointer is null"
    );
    ensure!(
        !weight.input_scale.is_null(),
        "FlashInfer FFN input-scale pointer is null"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN projection has invalid N/K geometry"
    );

    let groups = k / NVFP4_GROUP_SIZE;
    let scale_len = n
        .checked_mul(groups)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut logical_scales = vec![0_u8; scale_len];
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .context("read ModelOpt FFN block scales for CUTLASS interleave")?;
    let physical_scales =
        interleave_nvfp4_scales_128x4(&logical_scales, &[n, groups], NVFP4_GROUP_SIZE)?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(&physical_scales, &[n, groups], NVFP4_GROUP_SIZE,)?
            == logical_scales,
        "FlashInfer FFN physical block scales failed exact round trip"
    );

    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .context("read ModelOpt FFN input scale")?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "FlashInfer FFN weight_scale_2 must be finite and positive"
    );
    let alpha = input_scale.value() * weight.weight_scale_2;
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "FlashInfer FFN alpha must be finite and positive"
    );

    let weight_scales_128x4 = gpu.alloc(physical_scales.len())?;
    gpu.copy_h2d(&physical_scales, weight_scales_128x4)
        .context("upload FlashInfer FFN physical block scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&alpha.to_le_bytes(), alpha_f32)
        .context("upload FlashInfer FFN alpha")?;

    Ok(FlashinferFfnProjection {
        weight: weight.weight,
        weight_scales_128x4,
        weight_scales_hash: flashinfer_scale_fingerprint(&physical_scales),
        alpha_f32,
        input_scale_bits: input_scale.value_bits(),
        alpha_bits: alpha.to_bits(),
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn log_flashinfer_ffn_success(
    layer: usize,
    operation: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnOperation,
    m: usize,
    tactic: usize,
) {
    use crate::weight_map::flashinfer_ffn_admission::Qwen38FfnOperation;
    use std::collections::HashSet;

    static RECEIPTS: std::sync::OnceLock<Mutex<HashSet<(u8, usize, usize)>>> =
        std::sync::OnceLock::new();
    let (family_id, family) = match operation {
        Qwen38FfnOperation::MergedGateUp => (0u8, "ffn_merged_gate_up"),
        Qwen38FfnOperation::Down => (1u8, "ffn_down"),
        Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up => {
            unreachable!("production FFN route does not launch separate gate/up")
        }
    };
    let receipts = RECEIPTS.get_or_init(|| Mutex::new(HashSet::new()));
    let newly_logged = receipts
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((family_id, m, tactic));
    if !newly_logged {
        return;
    }
    tracing::info!(
        layer,
        m,
        tactic,
        family,
        "routed Qwen3.8 dense FFN prefill through FlashInfer SM121"
    );
    eprintln!(
        "ATLAS_PREFILL_FFN_FLASHINFER ENGAGED family={family} layer={layer} M={m} tactic={tactic}"
    );
}

pub(super) fn load_e2m1_checkpoint_scales(
    weights: &DenseFfnWeights,
    gpu: &dyn GpuBackend,
) -> Result<Option<E2m1CheckpointScales>> {
    let flashinfer_requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
    let requested = flashinfer_requested
        || std::env::var("ATLAS_E2M1_STATIC_SCALE").ok().as_deref() == Some("1");
    if !requested {
        return Ok(None);
    }
    let full_w4a4 = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
    let down_w4a4 = std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
    ensure!(
        full_w4a4 || down_w4a4 || flashinfer_requested,
        "ATLAS_E2M1_STATIC_SCALE=1 requires an active W4A4 or FlashInfer FFN route"
    );

    let scales = validated_e2m1_checkpoint_scales(
        read_e2m1_input_scale(gpu, weights.gate_proj.input_scale, "gate_proj")?,
        read_e2m1_input_scale(gpu, weights.up_proj.input_scale, "up_proj")?,
        read_e2m1_input_scale(gpu, weights.down_proj.input_scale, "down_proj")?,
    )?;
    tracing::info!(
        gate_up_input_scale = scales.gate_up,
        down_input_scale = scales.down,
        "enabled checkpoint-static W4A4 activation scales"
    );
    Ok(Some(scales))
}

/// Opportunistically load ModelOpt checkpoint `input_scale` scalars for the
/// cuBLASLt NVFP4 prefill route on SM103/SM120 (or explicit `ATLAS_CUBLASLT_NVFP4=1`),
/// eliminating per-layer `nvfp4_global_absmax` + GPU stream syncs when present.
/// Falls back cleanly to `None` (dynamic absmax) if any pointer is null or invalid.
#[cfg(feature = "cuda")]
pub(super) fn try_load_cublaslt_checkpoint_scales(
    weights: &DenseFfnWeights,
    gpu: &dyn GpuBackend,
    existing: Option<E2m1CheckpointScales>,
) -> Option<E2m1CheckpointScales> {
    if let Some(scales) = existing {
        return Some(scales);
    }
    let policy = prefill_ffn_cublaslt_nvfp4_policy();
    let cc = gpu.compute_capability();
    let may_engage = match policy {
        CublasLtNvfp4Policy::Disabled => false,
        CublasLtNvfp4Policy::Auto => {
            is_blackwell_dc_compute_capability(cc) || is_sm120_compute_capability(cc)
        }
        CublasLtNvfp4Policy::Enabled => cc.is_some(),
    };
    if !may_engage
        || weights.gate_proj.input_scale.is_null()
        || weights.up_proj.input_scale.is_null()
        || weights.down_proj.input_scale.is_null()
    {
        return None;
    }
    let gate = read_e2m1_input_scale(gpu, weights.gate_proj.input_scale, "gate_proj").ok()?;
    let up = read_e2m1_input_scale(gpu, weights.up_proj.input_scale, "up_proj").ok()?;
    let down = read_e2m1_input_scale(gpu, weights.down_proj.input_scale, "down_proj").ok()?;
    validated_e2m1_checkpoint_scales(gate, up, down).ok()
}

impl DenseFfnLayer {
    /// Whether the W4A4 (NVFP4×NVFP4) native tensor-core FFN prefill path
    /// is wired up. Returns true when all three required kernel symbols
    /// are present in the loaded module. Used by `forward_prefill` to
    /// choose between the W4A4 fast path and the existing fp8/v2/m128
    /// fallbacks.
    pub fn has_e2m1_ffn(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    /// Whether the K-major W4A4 implementation has every runtime component
    /// other than the per-layer transformed weights (installed after `new`).
    pub(super) fn has_e2m1_kmajor_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m128_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    pub(super) fn has_e2m1_kmajor_m256_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m256_k.0 != 0 && self.has_e2m1_kmajor_runtime()
    }

    /// Ensure the W4A4 activation scratch arena has capacity for `m` rows
    /// and `k` columns. Reallocates in-place if either dimension grew.
    ///
    /// Returns `(a_packed, a_scale, a_max)`.
    pub(super) fn ensure_e2m1_scratch(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let layout = e2m1_scratch_layout(m, k)?;
        let mut slot = self.e2m1_scratch.lock().unwrap();
        let needs_realloc = match slot.as_ref() {
            Some(s) => s.cap_m < layout.cap_m || s.cap_k < layout.cap_k,
            None => true,
        };
        if needs_realloc {
            // Prior launches (either this layer's or another layer's on a
            // shared arena) may still read the previous scratch buffers on the
            // active stream; drain before freeing.
            if slot.is_some() {
                gpu.synchronize_device()?;
            }
            // Free previous (if any) so we don't leak when M or K grows.
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.a_packed);
                let _ = gpu.free(prev.a_scale);
                let _ = gpu.free(prev.a_max);
            }
            let a_packed = gpu.alloc(layout.packed_bytes)?;
            let a_scale = match gpu.alloc(layout.scale_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    return Err(err);
                }
            };
            let a_max = match gpu.alloc(layout.max_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    let _ = gpu.free(a_scale);
                    return Err(err);
                }
            };
            *slot = Some(E2m1Scratch {
                a_packed,
                a_scale,
                a_max,
                cap_m: layout.cap_m,
                cap_k: layout.cap_k,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.a_packed, s.a_scale, s.a_max))
    }

    /// Use `shared`'s activation arena and stream binding instead of this
    /// layer's own (`ATLAS_MEMLEAN_SHARED_SCRATCH`). Must run before the
    /// layer allocates any scratch, i.e. before `prepare_flashinfer_prefill`.
    pub fn adopt_shared_e2m1_scratch(&mut self, shared: &SharedE2m1Scratch) -> Result<()> {
        ensure!(
            self.e2m1_scratch
                .lock()
                .map_err(|_| anyhow::anyhow!("W4A4 activation scratch lock poisoned"))?
                .is_none(),
            "a shared W4A4 activation scratch must be adopted before the layer allocates its own"
        );
        self.e2m1_scratch = Arc::clone(&shared.scratch);
        self.e2m1_use_stream = Arc::clone(&shared.stream);
        self.e2m1_scratch_shared = true;
        Ok(())
    }

    pub(super) fn lock_e2m1_use(&self, stream: u64) -> Result<MutexGuard<'_, Option<u64>>> {
        let mut admitted_stream = self
            .e2m1_use_stream
            .lock()
            .map_err(|_| anyhow::anyhow!("W4A4 activation scratch stream-binding lock poisoned"))?;
        match *admitted_stream {
            Some(existing) => ensure!(
                existing == stream,
                "W4A4 activation scratch is bound to CUDA stream {existing:#x}; rejected concurrent/reordered use on {stream:#x}"
            ),
            None => *admitted_stream = Some(stream),
        }
        Ok(admitted_stream)
    }

    /// Quantize one BF16 activation matrix for the W4A4 path.
    ///
    /// The returned packed values and scales remain owned by this layer's
    /// scratch arena. Callers may issue multiple same-stream GEMMs from them
    /// before preparing a different input. Gate and up share the same BF16
    /// input, so preparing once avoids a duplicate absmax scan, host readback,
    /// stream synchronization, and quantization launch without changing any
    /// quantized byte or GEMM arithmetic.
    pub(super) fn prepare_e2m1_input(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;

        let a_scale2 = if let Some(scale2) = checkpoint_scale2 {
            // ModelOpt calibrated this scalar offline and stored it as
            // `.input_scale`. It was read once during layer construction, so
            // the unchanged on-device quantizer can launch immediately.
            scale2
        } else {
            // Dynamic fallback: scan BF16 [M,K], drain the stream, and derive
            // a per-request scale. This remains the default W4A4 contract.
            ctx.gpu.memset_async(a_max, 0, 4, stream)?;
            ops::nvfp4_global_absmax(ctx.gpu, self.nvfp4_absmax_k, input, a_max, m * k, stream)?;
            ctx.gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(a_max, &mut bytes)?;
            let global_max = f32::from_le_bytes(bytes);
            if global_max > 0.0 {
                global_max / (6.0 * 448.0)
            } else {
                1.0
            }
        };

        // Phase 3: per-row E2M1 quantization of the activation. Writes
        // `a_packed` [M, K/2] + `a_scale` [M, K/16] in place.
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            input,
            a_packed,
            a_scale,
            a_scale2,
            m,
            k,
            stream,
        )?;

        Ok((a_packed, a_scale, a_scale2))
    }

    /// Materialize the SwiGLU down input directly in NVFP4 scratch.
    ///
    /// The CUDA kernel rounds each SiLU(gate)*up value through BF16 before
    /// deriving group scales and nibbles, matching the standalone activation
    /// kernel's numerical boundary without writing the temporary BF16 matrix.
    pub(super) fn prepare_e2m1_silu_input(
        &self,
        ctx: &ForwardContext,
        gate: DevicePtr,
        up: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: f32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, _) = self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;
        let fused_silu_policy = ops::effective_fused_rmsnorm_nvfp4_policy()?;
        let use_fused_silu = fused_silu_policy.is_fail_closed()
            || (fused_silu_policy.is_enabled() && self.fused_rmsnorm_nvfp4_k.has_silu_mul());
        if use_fused_silu {
            self.launch_fused_silu_nvfp4(
                ctx.gpu,
                gate,
                up,
                a_packed,
                a_scale,
                checkpoint_scale2,
                m,
                k,
                ops::SwiGluInputLayout::Split,
                ops::Nvfp4ScaleLayout::Linear,
                stream,
            )?;
        } else {
            ops::quantize_silu_mul_bf16_to_nvfp4(
                ctx.gpu,
                self.nvfp4_silu_quantize_k,
                gate,
                up,
                a_packed,
                a_scale,
                checkpoint_scale2,
                m,
                k,
                stream,
            )?;
        }
        Ok((a_packed, a_scale, checkpoint_scale2))
    }

    /// Dispatch one native W4A4 GEMM from an already prepared activation.
    ///
    /// `weight` is either the standard `[N, K/2]` row-major NVFP4 weight or,
    /// when `kmajor_m128` is true, the existing `[K/2,N]` / `[K/16,N]`
    /// transformed pair. Both kernels retain the same native block-scaled
    /// OMMA and BF16 epilogue order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_e2m1_prepared(
        &self,
        ctx: &ForwardContext,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        a_scale2: f32,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            supports_warp_nvfp4_mma(ctx.gpu.compute_capability()),
            "warp-level SM120 NVFP4 MMA (cutlass_nvfp4_gemm.cu) is unsupported on {:?}; use cuBLASLt NVFP4 or w4a16",
            ctx.gpu.compute_capability()
        );
        let scale2_ab = a_scale2 * weight.weight_scale_2;
        if kmajor_m128 {
            if e2m1_kmajor_kernel(m, crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled())
                == E2m1KmajorKernel::M256
            {
                ops::nvfp4_nvfp4_gemm_kmajor_m256(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m256_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            } else {
                ops::nvfp4_nvfp4_gemm_kmajor_m128(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m128_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            }
        } else {
            ops::nvfp4_nvfp4_gemm(
                ctx.gpu,
                self.nvfp4_gemm_k,
                a_packed,
                a_scale,
                weight.weight,
                weight.weight_scale,
                scale2_ab,
                output,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }

    /// Run the W4A4 fast path for one FFN projection: prequant BF16 input
    /// to NVFP4, then dispatch `nvfp4_nvfp4_gemm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_e2m1_proj(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        let (a_packed, a_scale, a_scale2) =
            self.prepare_e2m1_input(ctx, input, m, k, checkpoint_scale2, stream)?;
        self.forward_e2m1_prepared(
            ctx,
            a_packed,
            a_scale,
            a_scale2,
            weight,
            output,
            m,
            n,
            k,
            kmajor_m128,
            stream,
        )
    }

    /// Install transposed (`nvfp4_t` layout) FFN projection weights for
    /// the `forward_kgamma` M_TILE=16 fast path. Called by the loader
    /// after `DenseFfnLayer::new` when `ATLAS_FFN_M16_TRANSPOSED=1`.
    /// Takes ownership of `gate_proj_t` / `up_proj_t` / `down_proj_t`
    /// (additional allocations alongside the standard `weights.*_proj`;
    /// the originals are kept for decode-side GEMV paths that target the
    /// HuggingFace `[N, K/2]` layout).
    pub fn set_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.gate_proj_t = Some(gate_proj_t);
        self.up_proj_t = Some(up_proj_t);
        self.down_proj_t = Some(down_proj_t);
    }

    /// Install transposed FFN weights for the M>=128 `w4a16_gemm_t_m128`
    /// prefill route alone (Gemma-4 dense, whose target ships no
    /// `w4a16_gemm_t_m16`). Decode and verify routing is unchanged: they gate
    /// on `has_transposed_ffn()`, which still requires the m16 kernel.
    pub fn set_prefill_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.set_transposed_weights(gate_proj_t, up_proj_t, down_proj_t);
        self.prefill_t_m128_only = true;
    }

    /// Build the exact, default-off FlashInfer FFN prefill operands for one
    /// Qwen3.8 layer. Gate and up are copied once into an immutable merged
    /// gate-then-up operand; down retains the original checkpoint allocation.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn prepare_flashinfer_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        layer: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, Qwen38FfnProjection,
            qwen38_ffn_launch_candidate, qwen38_merged_gate_up_retained_bytes,
        };
        use ops::flashinfer_sm121::{FlashInferSm121, FlashInferSm121Shape};

        ensure!(
            crate::layers::prefill_ffn_flashinfer_enabled()?,
            "prepare_flashinfer_prefill called while its route is disabled"
        );
        ensure!(
            self.flashinfer_ffn_prefill.is_none(),
            "FlashInfer FFN operands are already prepared"
        );
        ensure!(
            hidden == QWEN38_HIDDEN && intermediate == QWEN38_INTERMEDIATE,
            "FlashInfer FFN route requires Qwen3.8 H={QWEN38_HIDDEN} I={QWEN38_INTERMEDIATE}; got H={hidden} I={intermediate}"
        );
        ensure!(
            self.activation == FfnActivation::SiLU,
            "FlashInfer FFN route requires a SiLU-gated dense FFN"
        );
        ensure!(
            self.e2m1_checkpoint_scales.is_some(),
            "FlashInfer FFN route requires admitted checkpoint-static activation scales"
        );

        let library_path = std::env::var_os("ATLAS_FLASHINFER_SM121_LIB").ok_or_else(|| {
            anyhow::anyhow!("ATLAS_PREFILL_FFN_FLASHINFER=1 requires ATLAS_FLASHINFER_SM121_LIB")
        })?;
        let library = FlashInferSm121::open_qualified(std::path::Path::new(&library_path))?;
        let quantize_atlas_128x4_k = gpu.kernel(
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        )?;
        let quantize_merged_silu_atlas_128x4_k = gpu.kernel(
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?;
        ensure!(
            quantize_atlas_128x4_k.0 != 0 && quantize_merged_silu_atlas_128x4_k.0 != 0,
            "FlashInfer FFN preparation resolved a NULL required kernel"
        );

        // Freeze the exact zero-workspace tactics before retaining any route.
        // Querying at construction catches a mismatched native library rather
        // than discovering it after some layer projections have run.
        let stream = gpu.default_stream();
        for (operation, m) in [
            (Qwen38FfnOperation::MergedGateUp, 2_079),
            (Qwen38FfnOperation::Down, 2_079),
            (Qwen38FfnOperation::MergedGateUp, 8_192),
            (Qwen38FfnOperation::Down, 8_192),
        ] {
            let plan = qwen38_ffn_launch_candidate(operation, m)?;
            ensure!(
                plan.performance_qualified && plan.workspace_bytes == 0,
                "FlashInfer FFN launch is not fully qualified: {operation:?} M={m}"
            );
            let shape = FlashInferSm121Shape::new(
                u8::try_from(plan.tactic).context("FlashInfer tactic exceeds u8")?,
                plan.m,
                plan.n,
                plan.k,
                1,
            )?;
            let prepared = library.prepare_borrowed_zero_workspace(gpu, shape, stream)?;
            ensure!(
                prepared.shape() == shape,
                "FlashInfer FFN preparation changed its frozen shape"
            );
        }

        // This exact allocation is already charged by the server's model-load
        // preflight. Materialize it now so the first measured M=2079 request
        // cannot pay 64 lazy allocations (or discover an OOM after serving has
        // started). The runtime route reuses this fixed-address maximum shape.
        let scratch_layout = e2m1_scratch_layout(8_192, intermediate)?;
        let (scratch_packed, scratch_scales, scratch_max) =
            self.ensure_e2m1_scratch(gpu, scratch_layout.cap_m, scratch_layout.cap_k)?;
        for pointer in [scratch_packed, scratch_scales, scratch_max] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN construction scratch is NULL or misaligned"
            );
        }
        {
            let retained = self.e2m1_scratch.lock().unwrap();
            let retained = retained
                .as_ref()
                .context("FlashInfer FFN construction scratch was not retained")?;
            ensure!(
                retained.cap_m == scratch_layout.cap_m && retained.cap_k == scratch_layout.cap_k,
                "FlashInfer FFN construction scratch retained the wrong capacity"
            );
        }

        let merged_gate_up = build_flashinfer_merged_gate_up(
            gpu,
            layer,
            &self.weights.gate_proj,
            &self.weights.up_proj,
        )?;
        let down = build_flashinfer_ffn_projection(
            gpu,
            layer,
            Qwen38FfnProjection::Down,
            &self.weights.down_proj,
            hidden,
            intermediate,
        )?;

        let (library_device, library_inode) = library.file_identity();
        tracing::info!(
            layer,
            library = %library.path().display(),
            library_sha256 = %library.sha256_hex(),
            library_device,
            library_inode,
            merged_gate_up_weight = format_args!("{:#x}", merged_gate_up.weight.0),
            down_weight = format_args!("{:#x}", down.weight.0),
            merged_gate_up_weight_hash = format_args!("{:016x}", merged_gate_up.weight_hash),
            merged_gate_up_scale_hash = format_args!("{:016x}", merged_gate_up.weight_scales_hash),
            down_scale_hash = format_args!("{:016x}", down.weight_scales_hash),
            merged_gate_up_alpha_bits = format_args!("{:08x}", merged_gate_up.alpha_bits),
            down_alpha_bits = format_args!("{:08x}", down.alpha_bits),
            merged_gate_up_retained_bytes = qwen38_merged_gate_up_retained_bytes(),
            scratch_cap_m = scratch_layout.cap_m,
            scratch_cap_k = scratch_layout.cap_k,
            scratch_packed_bytes = scratch_layout.packed_bytes,
            scratch_scale_bytes = scratch_layout.scale_bytes,
            "admitted FlashInfer SM121 FFN prefill operands"
        );
        self.flashinfer_ffn_prefill = Some(FlashinferFfnPrefill {
            layer,
            library,
            merged_gate_up,
            down,
            quantize_atlas_128x4_k,
            quantize_merged_silu_atlas_128x4_k,
        });
        Ok(())
    }

    /// `ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS=1`: point `gate_proj` / `up_proj` at
    /// their bytes inside the prepared merged FlashInfer operand, so the
    /// separate checkpoint allocations can be released. Every route that
    /// reads the original-layout packed weights (the pipe and baseline
    /// prefill GEMMs, the W4A4 routes, and the transposes built after this)
    /// then reads identical bytes from the merged buffer; block scales,
    /// scalars and down are untouched. Returns the replaced gate and up
    /// packed-weight pointers for the caller to release. The caller must have
    /// synchronized the construction stream.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn alias_gate_up_to_flashinfer_merged(&mut self) -> Result<(DevicePtr, DevicePtr)> {
        use crate::weight_map::flashinfer_ffn_admission::{QWEN38_HIDDEN, QWEN38_INTERMEDIATE};

        let route = self
            .flashinfer_ffn_prefill
            .as_mut()
            .context("gate/up alias requires prepared FlashInfer FFN operands")?;
        let replaced = (self.weights.gate_proj.weight, self.weights.up_proj.weight);
        ensure!(
            route.merged_gate_up.source_gate_weight == replaced.0
                && route.merged_gate_up.source_up_weight == replaced.1,
            "merged gate/up operand was not built from this layer's gate/up weights"
        );
        ensure!(
            route.merged_gate_up.weight != replaced.0 && route.merged_gate_up.weight != replaced.1,
            "gate/up are already aliased to the merged operand"
        );
        let (gate, up) = merged_gate_up_views(
            route.merged_gate_up.weight,
            QWEN38_INTERMEDIATE * QWEN38_HIDDEN / 2,
        )?;
        self.weights.gate_proj.weight = gate;
        self.weights.up_proj.weight = up;
        route.merged_gate_up.source_gate_weight = gate;
        route.merged_gate_up.source_up_weight = up;
        Ok(replaced)
    }

    /// Try the fully-admitted FlashInfer FFN route for an exact qualified
    /// chunk size. Other rows (notably a final short prefill tail) retain the
    /// existing path unchanged.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn try_forward_flashinfer_prefill(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, select_qwen38_ffn_launch,
        };
        use ops::flashinfer_sm121::{FlashInferSm121Buffers, FlashInferSm121Shape};

        let requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && h as usize == QWEN38_HIDDEN
            && inter as usize == QWEN38_INTERMEDIATE
            && ctx.config.tp_world_size.max(1) == 1;
        let m_qualified = matches!(m, 2_079 | 8_192)
            || crate::weight_map::flashinfer_ffn_admission::qwen38_ffn_launch_candidate(
                Qwen38FfnOperation::MergedGateUp,
                m as usize,
            )
            .is_ok();
        match flashinfer_ffn_prefill_route(
            requested,
            exact_qwen38_dense,
            m_qualified,
            self.flashinfer_ffn_prefill.is_some(),
        ) {
            FlashinferFfnPrefillRoute::Disabled | FlashinferFfnPrefillRoute::Ineligible => {
                return Ok(false);
            }
            FlashinferFfnPrefillRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_FFN_FLASHINFER=1 requires prepared merged FFN operands for exact M={m}"
            ),
            FlashinferFfnPrefillRoute::Complete => {}
        }
        let route = self.flashinfer_ffn_prefill.as_ref().unwrap();
        ensure!(
            route.merged_gate_up.layer == route.layer
                && route.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && route.merged_gate_up.source_up_weight == self.weights.up_proj.weight
                && route.down.weight == self.weights.down_proj.weight,
            "FlashInfer FFN checkpoint operand identity changed after preparation"
        );

        let mut merged_plan =
            select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, m as usize)?;
        let mut down_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::Down, m as usize)?;
        if let Some((gu, dn)) = crate::layers::flashinfer_ffn_tactic_override() {
            static SEEN: std::sync::Once = std::sync::Once::new();
            SEEN.call_once(|| tracing::warn!("ATLAS_FLASHINFER_FFN_TACTIC override: merged_gate_up={gu} down={dn} (diagnostic, not exact-qualified)"));
            merged_plan.tactic = gu;
            down_plan.tactic = dn;
        }
        ensure!(
            merged_plan.workspace_bytes == 0 && down_plan.workspace_bytes == 0,
            "FlashInfer FFN runtime selected a nonzero-workspace plan"
        );

        let rows = m as usize;
        let input_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let projection_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 2)?;
        let merged_bytes = flashinfer_ffn_extent(rows, 2 * QWEN38_INTERMEDIATE, 2)?;
        let output_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let activation_fp4_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 1)? / 2;
        let activation_scale_bytes =
            flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)?;
        ensure!(
            ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes,
            "merged FlashInfer FFN output arena is undersized"
        );
        ensure!(
            ctx.buffers.sizes().expert_gate_out >= projection_bytes
                && ctx.buffers.sizes().expert_up_out >= projection_bytes
                && ctx.buffers.sizes().moe_output >= output_bytes,
            "FlashInfer FFN destination arena is undersized"
        );

        let merged_out = ctx.buffers.ffn_gate_up_bf16();
        let output = ctx.buffers.moe_output();
        for pointer in [input, merged_out, gate_out, up_out, output] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN input/output pointer is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(input, input_bytes, merged_out, merged_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(up_out, projection_bytes, output, output_bytes)?;

        let merged_shape = FlashInferSm121Shape::new(
            u8::try_from(merged_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            merged_plan.m,
            merged_plan.n,
            merged_plan.k,
            1,
        )?;
        let down_shape = FlashInferSm121Shape::new(
            u8::try_from(down_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            down_plan.m,
            down_plan.n,
            down_plan.k,
            1,
        )?;
        let fused_rmsnorm_policy = ops::effective_fused_rmsnorm_nvfp4_policy()?;
        let mut merged_launch =
            route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)?;
        let mut down_launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)?;

        // Allocate once at the larger down-projection K so no asynchronous
        // activation operand can be freed while it is still in flight. All
        // route shapes, symbols, destinations and native plans are already
        // preflighted before the first quantization/device output effect.
        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (activation_fp4, activation_scales, _) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
        for pointer in [activation_fp4, activation_scales] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN activation scratch is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(
            activation_fp4,
            activation_fp4_bytes,
            activation_scales,
            activation_scale_bytes,
        )?;
        for (pointer, bytes) in [
            (input, input_bytes),
            (merged_out, merged_bytes),
            (gate_out, projection_bytes),
            (up_out, projection_bytes),
            (output, output_bytes),
        ] {
            ensure_flashinfer_ffn_disjoint(activation_fp4, activation_fp4_bytes, pointer, bytes)?;
            ensure_flashinfer_ffn_disjoint(
                activation_scales,
                activation_scale_bytes,
                pointer,
                bytes,
            )?;
        }

        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_atlas_128x4_k,
            input,
            activation_fp4,
            activation_scales,
            f32::from_bits(route.merged_gate_up.input_scale_bits),
            m,
            h,
            stream,
        )?;
        merged_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: merged_out,
                activation_fp4,
                weight_fp4: route.merged_gate_up.weight,
                activation_scales,
                weight_scales: route.merged_gate_up.weight_scales_128x4,
                global_scale_f32: route.merged_gate_up.alpha_f32,
            },
            stream,
        )?;
        if fused_rmsnorm_policy.is_fail_closed()
            || (fused_rmsnorm_policy.is_enabled() && self.fused_rmsnorm_nvfp4_k.has_silu_mul())
        {
            self.launch_fused_silu_nvfp4(
                ctx.gpu,
                merged_out,
                merged_out.offset((inter as usize) * 2),
                activation_fp4,
                activation_scales,
                f32::from_bits(route.down.input_scale_bits),
                m,
                inter,
                ops::SwiGluInputLayout::MergedGateUp,
                ops::Nvfp4ScaleLayout::Swizzled128x4,
                stream,
            )?;
        } else {
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                ctx.gpu,
                route.quantize_merged_silu_atlas_128x4_k,
                merged_out,
                activation_fp4,
                activation_scales,
                f32::from_bits(route.down.input_scale_bits),
                m,
                inter,
                stream,
            )?;
        }

        down_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: output,
                activation_fp4,
                weight_fp4: route.down.weight,
                activation_scales,
                weight_scales: route.down.weight_scales_128x4,
                global_scale_f32: route.down.alpha_f32,
            },
            stream,
        )?;

        log_flashinfer_ffn_success(
            route.layer,
            Qwen38FfnOperation::MergedGateUp,
            merged_plan.m,
            merged_plan.tactic,
        );
        log_flashinfer_ffn_success(
            route.layer,
            Qwen38FfnOperation::Down,
            down_plan.m,
            down_plan.tactic,
        );
        Ok(true)
    }
}
