// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit owners for opt-in routed-original backing allocations. Views never
//! own storage and no destructor calls the driver (including on failed loads).

use super::{WeightDtype, WeightTensor};
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RoutedProjection {
    Gate,
    Up,
    Down,
}

impl RoutedProjection {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

pub(crate) struct RoutedBackingMember {
    pub name: String,
    pub expert: usize,
    pub scale: bool,
    pub offset: usize,
    pub bytes: usize,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Live,
    Released,
    // Driver errors and panics do not establish whether a free completed.
    Quarantined,
}

struct Group {
    layer_prefix: String,
    projection: RoutedProjection,
    base: DevicePtr,
    bytes: usize,
    members: Vec<RoutedBackingMember>,
    state: Mutex<State>,
}

#[derive(Default)]
pub(crate) struct RoutedGroupRegistry {
    groups: Vec<Arc<Group>>,
    names: HashSet<String>,
    pub(crate) released: Arc<Mutex<HashSet<String>>>,
}

impl RoutedGroupRegistry {
    pub(crate) fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// Register before DMA or publishing a view. The base guard must remain
    /// unexposed and every member must be aligned, disjoint and in bounds.
    pub(crate) fn register(
        &mut self,
        layer_prefix: String,
        projection: RoutedProjection,
        base: DevicePtr,
        bytes: usize,
        mut members: Vec<RoutedBackingMember>,
    ) -> Result<()> {
        ensure!(
            !base.is_null() && base.0 % 256 == 0,
            "invalid grouped owning base"
        );
        let end = base
            .0
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("grouped backing address overflow"))?;
        ensure!(bytes > 256 && !members.is_empty(), "empty grouped backing");
        for group in &self.groups {
            let other_end = group
                .base
                .0
                .checked_add(u64::try_from(group.bytes)?)
                .ok_or_else(|| anyhow::anyhow!("registered backing address overflow"))?;
            ensure!(
                end <= group.base.0 || other_end <= base.0,
                "overlapping grouped owners"
            );
        }
        members.sort_by_key(|member| member.offset);
        let mut previous_end = 256usize;
        let mut names = HashSet::new();
        for member in &members {
            ensure!(
                member.name
                    == format!(
                        "{layer_prefix}.mlp.experts.{}.{}.{}",
                        member.expert,
                        projection.name(),
                        if member.scale {
                            "weight_scale"
                        } else {
                            "weight"
                        }
                    ),
                "grouped member crosses owner layer/projection scope"
            );
            ensure!(
                member.offset >= previous_end && member.offset % 256 == 0,
                "overlapping or unaligned grouped member"
            );
            previous_end = member
                .offset
                .checked_add(member.bytes)
                .ok_or_else(|| anyhow::anyhow!("grouped member range overflow"))?;
            ensure!(
                member.bytes > 0 && previous_end <= bytes,
                "grouped member out of bounds"
            );
            ensure!(
                !self.names.contains(&member.name) && names.insert(member.name.clone()),
                "duplicate grouped member '{}'",
                member.name
            );
            let elements = member
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("grouped member shape overflow"))?;
            ensure!(
                elements.checked_mul(member.dtype.byte_size()) == Some(member.bytes),
                "grouped member shape/byte mismatch"
            );
        }
        self.names.extend(names);
        self.groups.push(Arc::new(Group {
            layer_prefix,
            projection,
            base,
            bytes,
            members,
            state: Mutex::new(State::Live),
        }));
        Ok(())
    }

    /// Undo the most recent [`Self::register`] of `base` when the backend
    /// refuses its claim. Only valid before any DMA or view publication.
    pub(crate) fn unregister_unclaimed(&mut self, base: DevicePtr) -> Result<()> {
        let last = self
            .groups
            .last()
            .ok_or_else(|| anyhow::anyhow!("no grouped owner to unregister"))?;
        ensure!(
            last.base == base && *last.state.lock() == State::Live,
            "only the latest live grouped owner can be unregistered"
        );
        let group = self.groups.pop().expect("checked latest grouped owner");
        for member in &group.members {
            self.names.remove(&member.name);
        }
        Ok(())
    }

    pub(crate) fn validate_store(&self, weights: &HashMap<String, WeightTensor>) -> Result<()> {
        for group in &self.groups {
            ensure!(
                *group.state.lock() == State::Live,
                "group owner is not live"
            );
            for member in &group.members {
                let tensor = weights
                    .get(&member.name)
                    .ok_or_else(|| anyhow::anyhow!("missing grouped view '{}'", member.name))?;
                ensure!(
                    tensor.ptr == group.base.offset(member.offset)
                        && tensor.shape == member.shape
                        && tensor.dtype == member.dtype,
                    "grouped view changed for '{}'",
                    member.name
                );
            }
        }
        Ok(())
    }

    pub(crate) fn layer(&self, prefix: &str) -> Option<RoutedOriginalGroups> {
        let groups: Vec<_> = self
            .groups
            .iter()
            .filter(|group| group.layer_prefix == prefix)
            .cloned()
            .collect();
        (!groups.is_empty()).then(|| RoutedOriginalGroups {
            groups,
            released: self.released.clone(),
        })
    }
}

impl Drop for RoutedGroupRegistry {
    fn drop(&mut self) {
        if self
            .groups
            .iter()
            .any(|group| *group.state.lock() != State::Released)
        {
            // Preserve explicit live/quarantined owner records through error
            // unwind or model teardown. Context/process teardown owns storage.
            std::mem::forget(std::mem::take(&mut self.groups));
        }
    }
}

/// Shared handles into a WeightStore. A model layer must validate every routed
/// pair before attaching these owners, then release only at a quiescent phase.
pub struct RoutedOriginalGroups {
    groups: Vec<Arc<Group>>,
    released: Arc<Mutex<HashSet<String>>>,
}

impl RoutedOriginalGroups {
    pub fn validate_projection(
        &self,
        projection: RoutedProjection,
        pairs: &[(DevicePtr, DevicePtr)],
    ) -> Result<()> {
        let mut seen = HashSet::new();
        for group in self
            .groups
            .iter()
            .filter(|group| group.projection == projection)
        {
            ensure!(
                *group.state.lock() == State::Live,
                "grouped originals are not live"
            );
            for member in &group.members {
                let pair = pairs
                    .get(member.expert)
                    .ok_or_else(|| anyhow::anyhow!("grouped expert index out of range"))?;
                let ptr = if member.scale { pair.1 } else { pair.0 };
                ensure!(
                    ptr == group.base.offset(member.offset)
                        && seen.insert((member.expert, member.scale)),
                    "grouped model alias mismatch or duplicate"
                );
            }
        }
        ensure!(
            seen.len()
                == pairs
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::anyhow!("grouped pair count overflow"))?,
            "grouped model projection membership is incomplete"
        );
        Ok(())
    }

    /// Caller must establish GPU quiescence before this call. Failed frees are
    /// tombstoned and quarantined, never retried or released through a view.
    pub fn release_projection(
        &self,
        projection: RoutedProjection,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.release_projection_with(projection, |base| gpu.free_grouped_backing(base))
    }

    fn release_projection_with(
        &self,
        projection: RoutedProjection,
        mut free: impl FnMut(DevicePtr) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            self.groups
                .iter()
                .any(|group| group.projection == projection),
            "no grouped owners for projection"
        );
        for group in self
            .groups
            .iter()
            .filter(|group| group.projection == projection)
        {
            let mut state = group.state.lock();
            ensure!(
                *state == State::Live,
                "grouped backing already released or quarantined"
            );
            *state = State::Quarantined;
            self.released
                .lock()
                .extend(group.members.iter().map(|member| member.name.clone()));
            // Quarantine is installed first, including across a backend panic.
            free(group.base)?;
            *state = State::Released;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "routed_backing_tests.rs"]
mod tests;
