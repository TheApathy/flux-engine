// SPDX-License-Identifier: AGPL-3.0-only

//! Strict, opt-in canonical target NVFP4 groups. Plan the complete selected
//! inventory before allocating; each backing stays inside one shard/layer/
//! projection and is bounded to 64 MiB, not an eager whole-model slab.

use super::header::{TensorMeta, parse_header};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::{RoutedBackingMember, RoutedGroupRegistry, RoutedProjection, WeightDtype};
use anyhow::{Context, Result, ensure};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

const PREFIX: &str = "model.language_model.layers.";
const ALIGN: usize = 256;
const GROUP_LIMIT: usize = 64 * 1024 * 1024;
const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const COMPONENTS: usize = LAYERS * EXPERTS * 3 * 2;
const PAYLOAD_BYTES: usize = 67_947_724_800;

pub const ROUTED_GROUPS_ENV: &str = "ATLAS_FAST_LOAD_ROUTED_GROUPS";
/// Opt-in sub-flag of [`ROUTED_GROUPS_ENV`]: allocate, register and claim every
/// planned base before the first shard read instead of at each group's first
/// view. OFF when absent or exactly `0`; exactly `1` opts in; requires
/// `ATLAS_FAST_LOAD_ROUTED_GROUPS=1`.
pub const ROUTED_GROUPS_EAGER_ENV: &str = "ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER";

pub fn routed_grouping_requested(value: Option<&std::ffi::OsStr>) -> Result<bool> {
    match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => Ok(false),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("{ROUTED_GROUPS_ENV} must be exactly 0 or 1"),
    }
}

/// Strict parse of [`ROUTED_GROUPS_EAGER_ENV`]; `grouped` is the resolved
/// [`ROUTED_GROUPS_ENV`] decision. Eager without grouping fails closed.
pub fn routed_grouping_eager_requested(
    value: Option<&std::ffi::OsStr>,
    grouped: bool,
) -> Result<bool> {
    let eager = match value.and_then(std::ffi::OsStr::to_str) {
        None if value.is_none() => false,
        Some("0") => false,
        Some("1") => true,
        _ => anyhow::bail!("{ROUTED_GROUPS_EAGER_ENV} must be exactly 0 or 1"),
    };
    ensure!(
        !eager || grouped,
        "{ROUTED_GROUPS_EAGER_ENV}=1 requires {ROUTED_GROUPS_ENV}=1"
    );
    Ok(eager)
}

#[derive(Clone)]
struct View {
    name: String,
    expert: usize,
    scale: bool,
    offset: usize,
    bytes: usize,
    shape: Vec<usize>,
    dtype: WeightDtype,
    abs_offset: u64,
}

struct PlanGroup {
    shard: PathBuf,
    layer_prefix: String,
    projection: RoutedProjection,
    bytes: usize,
    views: Vec<View>,
    base: Option<DevicePtr>,
}

pub(super) struct RoutedBackingLoad {
    groups: Vec<PlanGroup>,
    by_name: HashMap<String, (usize, usize)>,
    registry: Option<RoutedGroupRegistry>,
    copied: HashSet<String>,
}

fn align(bytes: usize) -> Result<usize> {
    bytes
        .checked_add(ALIGN - 1)
        .map(|n| n & !(ALIGN - 1))
        .ok_or_else(|| anyhow::anyhow!("routed group layout overflow"))
}

fn parse_member(meta: &TensorMeta) -> Result<Option<(usize, usize, RoutedProjection, bool)>> {
    let Some(tail) = meta.name.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let Some((layer, tail)) = tail.split_once(".mlp.experts.") else {
        return Ok(None);
    };
    let parts: Vec<_> = tail.split('.').collect();
    if !matches!(parts.last(), Some(&"weight") | Some(&"weight_scale")) {
        return Ok(None);
    }
    ensure!(
        parts.len() == 3,
        "invalid grouped routed name '{}'",
        meta.name
    );
    let layer: usize = layer.parse().context("invalid grouped routed layer")?;
    let expert: usize = parts[0].parse().context("invalid grouped routed expert")?;
    ensure!(
        layer < LAYERS && expert < EXPERTS,
        "grouped routed index out of range"
    );
    let projection = match parts[1] {
        "gate_proj" => RoutedProjection::Gate,
        "up_proj" => RoutedProjection::Up,
        "down_proj" => RoutedProjection::Down,
        _ => anyhow::bail!("unsupported grouped routed projection"),
    };
    let scale = parts[2] == "weight_scale";
    let (rows, cols) = if projection == RoutedProjection::Down {
        (2560, 640)
    } else {
        (640, 2560)
    };
    let shape = vec![rows, cols / if scale { 16 } else { 2 }];
    let dtype = if scale {
        WeightDtype::FP8E4M3
    } else {
        WeightDtype::UInt8
    };
    ensure!(
        meta.shape == shape && meta.dtype == dtype && meta.len == shape[0] * shape[1],
        "unsupported grouped routed dtype/shape/length for '{}'",
        meta.name
    );
    // Disallow alternate spellings that could map distinct names onto one slot.
    ensure!(
        meta.name
            == format!(
                "{PREFIX}{layer}.mlp.experts.{expert}.{}.{}",
                projection.name(),
                if scale { "weight_scale" } else { "weight" }
            ),
        "noncanonical grouped routed name"
    );
    Ok(Some((layer, expert, projection, scale)))
}

impl RoutedBackingLoad {
    pub(super) fn overhead_bytes(&self) -> usize {
        self.groups.iter().map(|group| group.bytes).sum::<usize>() - PAYLOAD_BYTES
    }

    pub(super) fn plan(
        shards: &[PathBuf],
        index: Option<&HashMap<String, String>>,
        skip: &dyn Fn(&str) -> bool,
        extra: Option<&Path>,
    ) -> Result<Self> {
        let mut inventories = Vec::with_capacity(shards.len());
        for shard in shards {
            let mut file = File::open(shard)?;
            let file_bytes = file.metadata()?.len();
            let shard_name = shard
                .file_name()
                .and_then(|s| s.to_str())
                .context("non-Unicode grouped shard filename")?;
            let mut tensors = parse_header(&mut file)?;
            tensors.retain(|meta| {
                !skip(&meta.name)
                    && index.is_none_or(|map| map.get(&meta.name).is_some_and(|s| s == shard_name))
            });
            for meta in &tensors {
                ensure!(
                    meta.abs_offset
                        .checked_add(u64::try_from(meta.len)?)
                        .is_some_and(|end| end <= file_bytes),
                    "grouped inventory range exceeds shard"
                );
            }
            inventories.push((shard.clone(), tensors));
        }
        // Supplemental MTP remains individual. A supplemental target alias
        // cannot overwrite a view after the complete target plan is validated.
        if let Some(extra) = extra {
            let mut file = File::open(extra)?;
            for meta in parse_header(&mut file)? {
                ensure!(
                    parse_member(&meta)?.is_none(),
                    "extra weights must not replace grouped target originals"
                );
            }
        }
        Self::from_inventories(inventories)
    }

    fn from_inventories(inventories: Vec<(PathBuf, Vec<TensorMeta>)>) -> Result<Self> {
        let mut groups: Vec<PlanGroup> = Vec::new();
        let mut current: HashMap<(PathBuf, usize, RoutedProjection), usize> = HashMap::new();
        let mut by_name = HashMap::new();
        let mut slots = HashSet::new();
        let mut payload = 0usize;
        for (shard, tensors) in inventories {
            for meta in tensors {
                let Some((layer, expert, projection, scale)) = parse_member(&meta)? else {
                    continue;
                };
                ensure!(
                    slots.insert((layer, expert, projection, scale)),
                    "duplicate grouped routed slot"
                );
                let key = (shard.clone(), layer, projection);
                let group_index = match current.get(&key).copied() {
                    Some(i)
                        if align(groups[i].bytes)?
                            .checked_add(meta.len)
                            .is_some_and(|end| end <= GROUP_LIMIT) =>
                    {
                        i
                    }
                    _ => {
                        let i = groups.len();
                        groups.push(PlanGroup {
                            shard: shard.clone(),
                            layer_prefix: format!("{PREFIX}{layer}"),
                            projection,
                            bytes: ALIGN,
                            views: Vec::new(),
                            base: None,
                        });
                        current.insert(key, i);
                        i
                    }
                };
                let group = &mut groups[group_index];
                let offset = align(group.bytes)?;
                group.bytes = offset
                    .checked_add(meta.len)
                    .ok_or_else(|| anyhow::anyhow!("grouped backing size overflow"))?;
                ensure!(
                    group.bytes <= GROUP_LIMIT,
                    "grouped tensor exceeds backing limit"
                );
                payload = payload
                    .checked_add(meta.len)
                    .ok_or_else(|| anyhow::anyhow!("grouped payload overflow"))?;
                let view_index = group.views.len();
                ensure!(
                    by_name
                        .insert(meta.name.clone(), (group_index, view_index))
                        .is_none(),
                    "duplicate grouped routed name"
                );
                group.views.push(View {
                    name: meta.name,
                    expert,
                    scale,
                    offset,
                    bytes: meta.len,
                    shape: meta.shape,
                    dtype: meta.dtype,
                    abs_offset: meta.abs_offset,
                });
            }
        }
        ensure!(
            slots.len() == COMPONENTS && payload == PAYLOAD_BYTES,
            "grouping requires complete canonical target routed NVFP4 inventory"
        );
        // Native-MTP tensors keep their individual allocations: the MTP layer
        // frees them through the ordinary path and never attaches a group.
        ensure!(
            !by_name.keys().any(|name| name.starts_with("mtp.")),
            "native-MTP tensors must never be planned into a routed group"
        );
        let requested_bytes = groups
            .iter()
            .try_fold(0usize, |sum, group| sum.checked_add(group.bytes))
            .ok_or_else(|| anyhow::anyhow!("grouped requested byte count overflow"))?;
        tracing::info!(target: "atlas_weight_allocation_audit",
            mode = "bounded-per-shard-layer-projection-v1", group_count = groups.len(),
            component_views = by_name.len(), payload_bytes = payload, requested_bytes,
            guard_and_alignment_bytes = requested_bytes - payload,
            maximum_group_bytes = GROUP_LIMIT, base_guard_bytes = ALIGN,
            "FAST_WEIGHT_ROUTED_GROUP_PLAN: requested layout only, not allocator granularity or runtime qualification");
        Ok(Self {
            groups,
            by_name,
            registry: Some(RoutedGroupRegistry::default()),
            copied: HashSet::new(),
        })
    }

    pub(super) fn destination(
        &mut self,
        shard: &Path,
        meta: &TensorMeta,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<(DevicePtr, usize, Option<usize>)>> {
        let Some(&(group_index, view_index)) = self.by_name.get(&meta.name) else {
            ensure!(
                parse_member(meta)?.is_none(),
                "unplanned grouped routed member"
            );
            return Ok(None);
        };
        let group = &self.groups[group_index];
        let view = &group.views[view_index];
        ensure!(
            group.shard == shard
                && view.dtype == meta.dtype
                && view.shape == meta.shape
                && view.bytes == meta.len
                && view.abs_offset == meta.abs_offset,
            "grouped routed metadata changed after preflight"
        );
        let (offset, needs_base) = (view.offset, group.base.is_none());
        ensure!(
            self.copied.insert(meta.name.clone()),
            "grouped routed view copied twice"
        );
        // Eager mode materialized every base before the first shard read, so
        // this first-view path then finds its base already present.
        let allocated = if needs_base {
            Some(self.materialize(group_index, gpu)?)
        } else {
            None
        };
        let base = self.groups[group_index]
            .base
            .expect("registered group owns its base");
        Ok(Some((base, offset, allocated)))
    }

    /// Allocate, register and claim one planned base; returns its bytes.
    /// Register before claim, and undo on either failure, so a refused
    /// registration or claim never leaves a claimed or registered orphan:
    /// nothing has copied into or published a view of the base yet.
    fn materialize(&mut self, group_index: usize, gpu: &dyn GpuBackend) -> Result<usize> {
        let group = &self.groups[group_index];
        ensure!(group.base.is_none(), "grouped backing already allocated");
        let (prefix, projection, bytes) =
            (group.layer_prefix.clone(), group.projection, group.bytes);
        let members: Vec<RoutedBackingMember> = group
            .views
            .iter()
            .map(|view| RoutedBackingMember {
                name: view.name.clone(),
                expert: view.expert,
                scale: view.scale,
                offset: view.offset,
                bytes: view.bytes,
                shape: view.shape.clone(),
                dtype: view.dtype,
            })
            .collect();
        // No managed fallback for grouped owners: the pointer must remain a
        // checked device-owned range, not a writable host/UVM alias.
        let base = gpu.alloc(bytes)?;
        let registry = self.registry.as_mut().expect("live grouped registry");
        if let Err(error) = registry.register(prefix, projection, base, bytes, members) {
            return Err(release_unclaimed(gpu, base, error));
        }
        if let Err(error) = gpu.claim_grouped_backing(base) {
            registry.unregister_unclaimed(base)?;
            return Err(release_unclaimed(gpu, base, error));
        }
        self.groups[group_index].base = Some(base);
        Ok(bytes)
    }

    /// `ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER=1`: materialize every planned base
    /// before the first shard read. Returns each base's bytes in plan order.
    pub(super) fn allocate_all(&mut self, gpu: &dyn GpuBackend) -> Result<Vec<usize>> {
        (0..self.groups.len())
            .map(|group_index| self.materialize(group_index, gpu))
            .collect()
    }

    pub(super) fn finish(mut self) -> Result<RoutedGroupRegistry> {
        ensure!(
            self.copied.len() == self.by_name.len()
                && self.groups.iter().all(|group| group.base.is_some()),
            "grouped routed copies are incomplete"
        );
        Ok(self.registry.take().expect("live grouped registry"))
    }
}

/// Free a base that was never claimed and never copied into (ordinary free),
/// keeping the original error and reporting a failed free beside it.
fn release_unclaimed(gpu: &dyn GpuBackend, base: DevicePtr, error: anyhow::Error) -> anyhow::Error {
    match gpu.free(base) {
        Ok(()) => error,
        Err(free_error) => error.context(format!(
            "releasing the unclaimed grouped base {base} also failed: {free_error:#}"
        )),
    }
}

impl Drop for RoutedBackingLoad {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.take() {
            if self.groups.iter().any(|group| group.base.is_some()) {
                std::mem::forget(std::mem::take(&mut self.groups));
            }
            if !registry.is_empty() {
                // Failed loads/panics retain the explicit owner records as
                // well as the backend's allocations until process teardown.
                std::mem::forget(registry);
            }
        }
    }
}

#[cfg(test)]
#[path = "routed_groups_tests.rs"]
mod tests;
