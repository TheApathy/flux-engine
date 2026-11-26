// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in canonical unified transpose with bounded, model-lifetime owners.
//! This changes requested allocation geometry, not payload or driver policy.

use super::helpers_a::{
    ROUTED_TRANSPOSE_SLAB_ALIGNMENT as ALIGN, canonical_qwen4_routed_slab_selected,
    transpose_component_bytes,
};
use super::*;
use anyhow::ensure;
use atlas_core::config::ModelConfig;
use std::ffi::OsStr;

mod owner_cap;
pub(crate) use owner_cap::OwnerCap;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS";
pub(crate) const OWNER_CAP: usize = 64 * 1024 * 1024;
const LEGACY_GUARD_BYTES: usize = 512;

fn requested(value: Option<&OsStr>) -> Result<bool> {
    match value.and_then(OsStr::to_str) {
        None if value.is_none() => Ok(false),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("{SELECTOR} must be exactly 0 or 1"),
    }
}

fn validate_mode(enabled: bool, config: &ModelConfig, keep_originals: bool) -> Result<bool> {
    ensure!(
        !enabled || canonical_qwen4_routed_slab_selected(config, keep_originals),
        "{SELECTOR}=1 requires canonical Qwen4 target geometry and unified layout"
    );
    Ok(enabled)
}

pub(crate) fn selected(config: &ModelConfig, keep_originals: bool) -> Result<Option<OwnerCap>> {
    let selection = owner_cap::selection()?;
    Ok(validate_mode(selection.enabled, config, keep_originals)?.then_some(selection.cap))
}

fn layer_mode(enabled: bool, config: &ModelConfig, keep_originals: bool) -> Result<bool> {
    // The native MTP loader reuses this helper with qwen4_mtp_config(target).
    // Its individual destination path is deliberately outside the target flag.
    if enabled
        && !keep_originals
        && config.weight_prefix == "mtp"
        && config.num_hidden_layers == 1
        && config.full_attention_interval == 1
        && config.layer_types.as_slice() == [atlas_core::config::LayerType::FullAttention]
        && config.vision.is_none()
        && config.dflash_capture_layers.is_empty()
    {
        let mut target_geometry = config.clone();
        target_geometry.num_hidden_layers = 48;
        if canonical_qwen4_routed_slab_selected(&target_geometry, false) {
            return Ok(false);
        }
    }
    validate_mode(enabled, config, keep_originals)
}

pub(super) fn selected_for_layer(
    config: &ModelConfig,
    keep_originals: bool,
) -> Result<Option<OwnerCap>> {
    let selection = owner_cap::selection()?;
    Ok(layer_mode(selection.enabled, config, keep_originals)?.then_some(selection.cap))
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| anyhow::anyhow!("bounded transpose size overflow"))
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow::anyhow!("bounded transpose product overflow"))
}

fn align(bytes: usize) -> Result<usize> {
    Ok(add(bytes, ALIGN - 1)? & !(ALIGN - 1))
}

#[derive(Clone, Copy, Debug)]
struct ProjectionLayout {
    experts: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    scale_offset: usize,
    pair_stride: usize,
    experts_per_owner: usize,
    owner_count: usize,
    requested_bytes: usize,
    payload_bytes: usize,
    cap: usize,
}

#[derive(Clone, Copy, Debug)]
struct OwnerLayout {
    first_expert: usize,
    experts: usize,
    bytes: usize,
}

impl ProjectionLayout {
    fn build(experts: usize, rows: usize, columns: usize, cap: usize) -> Result<Self> {
        ensure!(
            experts > 0 && rows > 0 && columns > 0 && columns.is_multiple_of(16),
            "bounded transpose requires nonempty NVFP4 geometry"
        );
        ensure!(
            cap <= OWNER_CAP && cap > ALIGN,
            "invalid bounded transpose owner cap"
        );
        let (packed_bytes, scale_bytes) = transpose_component_bytes(rows, columns)?;
        let scale_offset = align(packed_bytes)?;
        let pair_stride = align(add(scale_offset, scale_bytes)?)?;
        let experts_per_owner = (cap - ALIGN) / pair_stride;
        ensure!(
            experts_per_owner > 0,
            "one transpose pair exceeds bounded owner cap"
        );
        let owner_count = add(experts, experts_per_owner - 1)? / experts_per_owner;
        let requested_bytes = add(mul(experts, pair_stride)?, mul(owner_count, ALIGN)?)?;
        let payload_bytes = mul(experts, add(packed_bytes, scale_bytes)?)?;
        Ok(Self {
            experts,
            packed_bytes,
            scale_bytes,
            scale_offset,
            pair_stride,
            experts_per_owner,
            owner_count,
            requested_bytes,
            payload_bytes,
            cap,
        })
    }

    fn owner(self, index: usize) -> Result<OwnerLayout> {
        ensure!(index < self.owner_count, "bounded owner index out of range");
        let first_expert = mul(index, self.experts_per_owner)?;
        let experts = (self.experts - first_expert).min(self.experts_per_owner);
        let bytes = add(ALIGN, mul(experts, self.pair_stride)?)?;
        ensure!(bytes <= self.cap, "bounded owner exceeds cap");
        Ok(OwnerLayout {
            first_expert,
            experts,
            bytes,
        })
    }

    fn offsets(self, owner: OwnerLayout, expert: usize) -> Result<(usize, usize)> {
        ensure!(
            expert >= owner.first_expert && expert < add(owner.first_expert, owner.experts)?,
            "expert does not belong to bounded owner"
        );
        let weight = add(ALIGN, mul(expert - owner.first_expert, self.pair_stride)?)?;
        let scale = add(weight, self.scale_offset)?;
        ensure!(
            weight >= ALIGN
                && weight % ALIGN == 0
                && scale % ALIGN == 0
                && add(weight, self.packed_bytes)? <= scale
                && add(scale, self.scale_bytes)? <= owner.bytes,
            "invalid bounded transpose component range"
        );
        Ok((weight, scale))
    }
}

/// No Drop/free/claim: backend context teardown owns successful destinations.
pub(super) struct RetainedOwner {
    base: DevicePtr,
    bytes: usize,
}

fn allocate_owner(
    owners: &mut Vec<RetainedOwner>,
    bytes: usize,
    allocate: impl FnOnce(usize) -> Result<DevicePtr>,
) -> Result<DevicePtr> {
    ensure!(
        bytes > ALIGN && bytes <= OWNER_CAP,
        "invalid bounded allocation request"
    );
    let base = allocate(bytes)?;
    // Keep every successful allocation before any validation or possible DMA.
    owners.push(RetainedOwner { base, bytes });
    ensure!(
        !base.is_null() && base.0 % ALIGN as u64 == 0,
        "bounded transpose owner is null or unaligned"
    );
    let end = base
        .0
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("bounded owner address overflow"))?;
    for other in &owners[..owners.len() - 1] {
        let other_end = other
            .base
            .0
            .checked_add(u64::try_from(other.bytes)?)
            .ok_or_else(|| anyhow::anyhow!("retained bounded owner address overflow"))?;
        ensure!(
            end <= other.base.0 || other_end <= base.0,
            "bounded transpose owners overlap"
        );
    }
    Ok(base)
}

fn traced_allocation(allocate: impl FnOnce() -> Result<DevicePtr>) -> Result<DevicePtr> {
    tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "allocation_begin");
    let result = allocate();
    match &result {
        Ok(_) => {
            tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "allocation_success")
        }
        Err(_) => {
            tracing::debug!(target: "atlas_bounded_transpose_allocation", event = "allocation_error")
        }
    }
    result
}

pub(super) fn transpose_projection(
    experts: &[ExpertWeight],
    projection: impl Fn(&ExpertWeight) -> &QuantizedWeight,
    owners: &mut Vec<RetainedOwner>,
    gpu: &dyn GpuBackend,
    rows: usize,
    columns: usize,
    cap: OwnerCap,
    projection_label: &'static str,
) -> Result<Vec<QuantizedWeight>> {
    transpose_projection_with_cap(
        experts,
        projection,
        owners,
        gpu,
        rows,
        columns,
        cap.bytes(),
        projection_label,
    )
}

fn transpose_projection_with_cap(
    experts: &[ExpertWeight],
    projection: impl Fn(&ExpertWeight) -> &QuantizedWeight,
    owners: &mut Vec<RetainedOwner>,
    gpu: &dyn GpuBackend,
    rows: usize,
    columns: usize,
    cap: usize,
    projection_label: &'static str,
) -> Result<Vec<QuantizedWeight>> {
    let layout = ProjectionLayout::build(experts.len(), rows, columns, cap)?;
    ensure!(
        experts.iter().all(|expert| {
            let source = projection(expert);
            !source.weight.is_null() && !source.weight_scale.is_null()
        }),
        "bounded transpose projection membership is incomplete"
    );
    let mut transposed = Vec::with_capacity(experts.len());
    for index in 0..layout.owner_count {
        let owner = layout.owner(index)?;
        let base = allocate_owner(owners, owner.bytes, |bytes| {
            let span = tracing::debug_span!(target: "atlas_bounded_transpose_allocation",
                "bounded_transpose_owner", projection = projection_label,
                projection_owner_ordinal = index + 1, owner_count = layout.owner_count,
                first_expert = owner.first_expert, experts = owner.experts,
                requested_bytes = bytes, owner_cap_bytes = cap,
                backend_operation = "GpuBackend::alloc");
            let _entered = span.enter();
            traced_allocation(|| gpu.alloc(bytes))
        })?;
        for expert in owner.first_expert..add(owner.first_expert, owner.experts)? {
            ensure!(
                expert == transposed.len(),
                "bounded transpose expert order changed"
            );
            let (weight, scale) = layout.offsets(owner, expert)?;
            // allocate_owner checked the entire address range before offset().
            transposed.push(projection(&experts[expert]).transpose_for_gemm_into(
                gpu,
                rows,
                columns,
                base.offset(weight),
                base.offset(scale),
            )?);
        }
    }
    ensure!(
        transposed.len() == experts.len(),
        "bounded transpose projection is incomplete"
    );
    Ok(transposed)
}

fn canonical_layouts(config: &ModelConfig, cap: OwnerCap) -> Result<[ProjectionLayout; 3]> {
    validate_mode(true, config, false)?;
    let gate = ProjectionLayout::build(
        config.num_experts,
        config.moe_intermediate_size,
        config.hidden_size,
        cap.bytes(),
    )?;
    let down = ProjectionLayout::build(
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
        cap.bytes(),
    )?;
    Ok([gate, gate, down])
}

pub(crate) fn admission_guard_delta(config: &ModelConfig, cap: OwnerCap) -> Result<usize> {
    let guard_bytes = canonical_layouts(config, cap)?
        .iter()
        .try_fold(0usize, |sum, layout| {
            add(sum, layout.requested_bytes - layout.payload_bytes)
        })?;
    guard_bytes
        .checked_sub(LEGACY_GUARD_BYTES)
        .ok_or_else(|| anyhow::anyhow!("bounded transpose guard accounting underflow"))
}

pub(super) fn record_ready(config: &ModelConfig, owner_count: usize, cap: OwnerCap) -> Result<()> {
    let layouts = canonical_layouts(config, cap)?;
    let expected_count = layouts
        .iter()
        .try_fold(0usize, |sum, layout| add(sum, layout.owner_count))?;
    let requested_bytes = layouts
        .iter()
        .try_fold(0usize, |sum, layout| add(sum, layout.requested_bytes))?;
    let payload_bytes = layouts
        .iter()
        .try_fold(0usize, |sum, layout| add(sum, layout.payload_bytes))?;
    ensure!(
        owner_count == expected_count,
        "bounded transpose owner membership is incomplete"
    );
    tracing::info!(target: "atlas_transposed_expert_slab",
        schema = "atlas.moe-bounded-transpose-owners.v1", selector = SELECTOR,
        owner_count, owner_cap_bytes = cap.bytes(), payload_bytes, requested_bytes,
        guard_and_alignment_bytes = requested_bytes - payload_bytes,
        experts = config.num_experts, projections = 3, managed = false,
        base_unexposed = true, down_allocation_after_phase_b = true,
        "BOUNDED_TRANSPOSE_OWNERS_READY: requested geometry only, not physical allocator or runtime qualification");
    Ok(())
}

#[cfg(test)]
mod tests {
    // CPU Rust definitions only. Execution needs a separate compiler window.
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn canonical_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.num_hidden_layers = 48;
        config.num_experts = 512;
        config.hidden_size = 2560;
        config.moe_intermediate_size = 640;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        config
    }

    #[test]
    fn bounded_selector_is_strict_default_off_and_target_unified_only() {
        assert!(!requested(None).unwrap());
        assert!(!requested(Some(OsStr::new("0"))).unwrap());
        assert!(requested(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "false", "01", " 1", "1 ", "2"] {
            assert!(requested(Some(OsStr::new(value))).is_err());
        }
        let config = canonical_config();
        assert!(validate_mode(true, &config, false).unwrap());
        assert!(validate_mode(true, &config, true).is_err());
        for field in 0..7 {
            let mut other = canonical_config();
            match field {
                0 => other.model_type = "minimax_m2".into(),
                1 => other.num_hidden_layers -= 1,
                2 => other.num_experts -= 1,
                3 => other.hidden_size -= 1,
                4 => other.moe_intermediate_size -= 1,
                5 => other.ep_world_size = 2,
                _ => other.ep_rank = 1,
            }
            assert!(validate_mode(true, &other, false).is_err());
            assert!(!validate_mode(false, &other, true).unwrap());
        }
    }

    #[cfg(unix)]
    #[test]
    fn bounded_selector_rejects_non_unicode() {
        use std::os::unix::ffi::OsStrExt;
        assert!(requested(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }

    #[test]
    fn bounded_target_flag_preserves_exact_native_mtp_submodel_scope() {
        let target = canonical_config();
        let mtp = crate::weight_loader::qwen4_mtp_config(&target);
        assert!(!layer_mode(true, &mtp, false).unwrap());
        assert!(validate_mode(true, &mtp, false).is_err());
        assert!(layer_mode(true, &mtp, true).is_err());
        assert!(layer_mode(true, &target, false).unwrap());
        for change in 0..4 {
            let mut other = mtp.clone();
            match change {
                0 => other.weight_prefix = "other".into(),
                1 => other.num_experts -= 1,
                2 => other.full_attention_interval = 2,
                _ => other.num_hidden_layers = 2,
            }
            assert!(layer_mode(true, &other, false).is_err());
        }
    }

    #[test]
    fn bounded_canonical_geometry_has_exact_counts_and_membership() -> Result<()> {
        let layouts = canonical_layouts(&canonical_config(), OwnerCap::DEFAULT)?;
        let mut requested = 0;
        for layout in layouts {
            assert_eq!(layout.packed_bytes, 819_200);
            assert_eq!(layout.scale_bytes, 102_400);
            assert_eq!(layout.pair_stride, 921_600);
            assert_eq!(layout.experts_per_owner, 72);
            assert_eq!(layout.owner_count, 8);
            assert_eq!(layout.requested_bytes, 471_861_248);
            assert!(add(ALIGN, mul(73, layout.pair_stride)?)? > OWNER_CAP);
            requested += layout.requested_bytes;
            let mut seen = Vec::new();
            for index in 0..layout.owner_count {
                let owner = layout.owner(index)?;
                assert_eq!(owner.bytes, if index < 7 { 66_355_456 } else { 7_373_056 });
                let mut end = ALIGN;
                for expert in owner.first_expert..owner.first_expert + owner.experts {
                    let (weight, scale) = layout.offsets(owner, expert)?;
                    assert!(weight >= end && weight >= ALIGN);
                    assert_eq!(weight % ALIGN, 0);
                    assert_eq!(scale % ALIGN, 0);
                    end = scale + layout.scale_bytes;
                    seen.push(expert);
                }
                assert_eq!(end, owner.bytes);
            }
            assert_eq!(seen, (0..512).collect::<Vec<_>>());
            for expert in [0, 71, 72, 503, 504, 511] {
                let owner = layout.owner(expert / 72)?;
                assert_eq!(
                    layout.offsets(owner, expert)?.0,
                    ALIGN + (expert % 72) * 921_600
                );
            }
            assert!(layout.owner(8).is_err());
            assert!(layout.offsets(layout.owner(0)?, 72).is_err());
        }
        assert_eq!(requested, 1_415_583_744);
        assert_eq!(requested * 48, 67_948_019_712);
        assert_eq!(
            admission_guard_delta(&canonical_config(), OwnerCap::DEFAULT)?,
            5_632
        );
        Ok(())
    }

    #[test]
    fn bounded_layout_refuses_overflow_empty_and_undersized_caps() {
        for (experts, rows, columns, cap) in [
            (0, 640, 2560, OWNER_CAP),
            (512, 0, 2560, OWNER_CAP),
            (512, 640, 0, OWNER_CAP),
            (512, 640, 15, OWNER_CAP),
            (usize::MAX, 640, 2560, OWNER_CAP),
            (512, usize::MAX, 2560, OWNER_CAP),
            (512, 640, usize::MAX - 15, OWNER_CAP),
            (512, 640, 2560, 256),
            (512, 640, 2560, 921_855),
            (512, 640, 2560, OWNER_CAP + 1),
        ] {
            assert!(ProjectionLayout::build(experts, rows, columns, cap).is_err());
        }
    }

    #[test]
    fn bounded_failed_allocation_retains_prior_owners_without_free_or_retry() -> Result<()> {
        let mut owners = Vec::new();
        let first = allocate_owner(&mut owners, 512, |_| Ok(DevicePtr(0x1000)))?;
        assert!(
            allocate_owner(&mut owners, 512, |_| {
                anyhow::bail!("injected allocation failure")
            })
            .is_err()
        );
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].base, first);
        assert_eq!(owners[0].bytes, 512);
        for base in [
            DevicePtr::NULL,
            DevicePtr(0x3001),
            DevicePtr(u64::MAX - 255),
        ] {
            let before = owners.len();
            assert!(allocate_owner(&mut owners, 512, |_| Ok(base)).is_err());
            assert_eq!(owners.len(), before + 1);
            assert_eq!(owners.last().unwrap().base, base);
        }
        let mut overlaps = Vec::new();
        allocate_owner(&mut overlaps, 512, |_| Ok(DevicePtr(0x1000)))?;
        assert!(allocate_owner(&mut overlaps, 512, |_| Ok(DevicePtr(0x1100))).is_err());
        assert_eq!(overlaps.len(), 2);
        Ok(())
    }

    #[test]
    fn bounded_small_payloads_tables_and_scalar_bits_match_legacy_transpose() -> Result<()> {
        let gpu = MockGpuBackend::new();
        let mut experts = Vec::new();
        let mut reference = Vec::new();
        for index in 0..512 {
            let packed: Vec<u8> = (0..24).map(|byte| (byte + index * 29) as u8).collect();
            let scales = [
                index as u8,
                (index as u8).wrapping_add(7),
                (index as u8).wrapping_add(13),
            ];
            let weight = gpu.alloc(packed.len())?;
            let weight_scale = gpu.alloc(scales.len())?;
            gpu.copy_h2d(&packed, weight)?;
            gpu.copy_h2d(&scales, weight_scale)?;
            let source = QuantizedWeight {
                weight,
                weight_scale,
                weight_scale_2: f32::from_bits(0x7fc0_0100 + index as u32),
                input_scale: DevicePtr(0xfeed_0000 + index as u64 * 256),
            };
            reference.push(source.transpose_for_gemm(&gpu, 3, 16)?);
            experts.push(ExpertWeight {
                gate_proj: source,
                up_proj: source,
                down_proj: source,
            });
        }
        let mut owners = Vec::new();
        // 72 tiny padded pairs per owner exercises all canonical expert edges.
        let outputs = transpose_projection_with_cap(
            &experts,
            |e| &e.gate_proj,
            &mut owners,
            &gpu,
            3,
            16,
            55_552,
            "gate",
        )?;
        assert_eq!(owners.len(), 8);
        for (index, actual) in outputs.iter().enumerate() {
            let expected = reference[index];
            for (source, destination, bytes) in [
                (expected.weight, actual.weight, 512),
                (expected.weight_scale, actual.weight_scale, 64),
            ] {
                let mut a = vec![0; bytes];
                let mut b = vec![0; bytes];
                gpu.copy_d2h(source, &mut a)?;
                gpu.copy_d2h(destination, &mut b)?;
                assert_eq!(a, b);
            }
            assert_eq!(
                actual.weight_scale_2.to_bits(),
                expected.weight_scale_2.to_bits()
            );
            assert_eq!(actual.input_scale, expected.input_scale);
        }
        for owner in &owners {
            let data = gpu
                .read_alloc(owner.base)
                .expect("retained destination owner");
            assert!(data[..ALIGN].iter().all(|byte| *byte == 0));
            assert!(
                outputs
                    .iter()
                    .all(|weight| weight.weight != owner.base && weight.weight_scale != owner.base)
            );
        }
        let table = build_ptr_table_from_qw(&outputs, &gpu)?;
        for (pointer, expected) in [
            (
                table.packed_ptrs,
                outputs
                    .iter()
                    .flat_map(|w| w.weight.0.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                table.scale_ptrs,
                outputs
                    .iter()
                    .flat_map(|w| w.weight_scale.0.to_le_bytes())
                    .collect(),
            ),
            (
                table.scale2_vals,
                outputs
                    .iter()
                    .flat_map(|w| w.weight_scale_2.to_bits().to_le_bytes())
                    .collect(),
            ),
        ] {
            let mut actual = vec![0; expected.len()];
            gpu.copy_d2h(pointer, &mut actual)?;
            assert_eq!(actual, expected);
        }
        let count = gpu.alloc_count();
        experts[2].gate_proj.weight_scale = DevicePtr::NULL;
        assert!(
            transpose_projection_with_cap(
                &experts,
                |e| &e.gate_proj,
                &mut Vec::new(),
                &gpu,
                3,
                16,
                55_552,
                "gate"
            )
            .is_err()
        );
        assert_eq!(gpu.alloc_count(), count);
        Ok(())
    }
}
