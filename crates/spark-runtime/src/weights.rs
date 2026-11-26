// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loading from safetensors files (SBIO IORouter for filesystem I/O).

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

mod safetensor_receipt;
// Source-pinned by repro-kit test_routed_backing_groups_source: lint here,
// not in the file.
#[allow(clippy::manual_is_multiple_of)]
mod routed_backing;

pub(crate) use routed_backing::{RoutedBackingMember, RoutedGroupRegistry};
pub use routed_backing::{RoutedOriginalGroups, RoutedProjection};
pub use safetensor_receipt::{Bf16FinitenessReceipt, attest_exact_bf16_safetensors};

/// Advise the OS to evict a file's pages from the page cache.
///
/// On GB10 (unified memory), mmap'd safetensors share the GPU memory pool.
/// After copying tensors to GPU, the mmap pages linger in the page cache,
/// consuming memory that should be available for KV cache and inference buffers.
/// This function tells the kernel those pages are no longer needed.
#[cfg(target_os = "linux")]
pub(crate) fn evict_page_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    // POSIX_FADV_DONTNEED = 4 on Linux (POSIX standard).
    // macOS lacks posix_fadvise — see the non-linux branch below.
    const POSIX_FADV_DONTNEED: libc::c_int = 4;
    // SAFETY: `file.as_raw_fd()` is a valid open fd borrowed for this syscall;
    // `offset = 0`, `len = 0` advises the entire file.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evict_page_cache(_file: &std::fs::File) {
    // No-op: macOS/BSD do not implement `posix_fadvise`; page-cache reclaim is
    // left to the OS virtual memory subsystem and memory compressor.
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    BF16,
    FP32,
    FP8E4M3,
    /// NVFP4 block-scale exponent (1 byte, E8M0). DeepSeek-V4.1's MTP head
    /// ships its expert scales in this format.
    FP8E8M0,
    UInt8,
    Int64,
}

impl WeightDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::FP8E4M3 => 1,
            Self::FP8E8M0 => 1,
            Self::UInt8 => 1,
            Self::Int64 => 8,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            safetensors::Dtype::I64 => Ok(Self::Int64),
            safetensors::Dtype::F8_E4M3 => Ok(Self::FP8E4M3),
            safetensors::Dtype::F8_E8M0 => Ok(Self::FP8E8M0),
            // Raw 1-byte container (DeepSeek-V4.1's I8-packed DSpark experts); see fast_weights.
            safetensors::Dtype::I8 => Ok(Self::UInt8),
            other => bail!("Unsupported safetensors dtype: {other:?}"),
        }
    }
}

/// A weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
    released: Arc<Mutex<HashSet<String>>>,
    routed_groups: RoutedGroupRegistry,
}

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
            released: Arc::default(),
            routed_groups: RoutedGroupRegistry::default(),
        }
    }

    /// Wrap a pre-built map. Used by alternate loaders (e.g.
    /// `fast_weights::FastSafetensorsLoader`) and by GLM-5.3's DFlash2
    /// admission, which validates against an empty store from another crate.
    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self {
            weights,
            released: Arc::default(),
            routed_groups: RoutedGroupRegistry::default(),
        }
    }

    pub(crate) fn with_routed_groups(
        weights: HashMap<String, WeightTensor>,
        routed_groups: RoutedGroupRegistry,
    ) -> Result<Self> {
        routed_groups.validate_store(&weights)?;
        Ok(Self {
            weights,
            released: routed_groups.released.clone(),
            routed_groups,
        })
    }

    pub fn routed_original_groups(&self, layer_prefix: &str) -> Option<RoutedOriginalGroups> {
        self.routed_groups.layer(layer_prefix)
    }

    /// Get a weight tensor by name. Fails fast if not found.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        ensure!(
            !self.released.lock().contains(name),
            "Weight '{name}' was consumed during model construction"
        );
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name) && !self.released.lock().contains(name)
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len() - self.released.lock().len()
    }

    /// True if no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights
            .keys()
            .filter(|name| !self.released.lock().contains(name.as_str()))
            .map(|name| name.as_str())
    }

    /// Total bytes across all weight tensors on the GPU.
    pub fn total_bytes(&self) -> usize {
        let released = self.released.lock();
        self.weights
            .iter()
            .filter(|(name, _)| !released.contains(name.as_str()))
            .map(|(_, weight)| weight.byte_size())
            .sum()
    }

    fn reserve_consumed_alias(
        &self,
        name: &str,
        expected_ptr: DevicePtr,
        expected_dtype: WeightDtype,
    ) -> Result<Option<(DevicePtr, usize)>> {
        ensure!(
            !self.routed_groups.contains(name),
            "Weight '{name}' is a grouped view and cannot be released individually"
        );
        let mut released = self.released.lock();
        ensure!(
            !released.contains(name),
            "Weight '{name}' was already consumed"
        );
        let weight = self
            .weights
            .get(name)
            .with_context(|| format!("Weight '{name}' not found in store"))?;
        if weight.ptr != expected_ptr || weight.dtype != expected_dtype {
            return Ok(None);
        }
        released.insert(name.to_string());
        Ok(Some((weight.ptr, weight.byte_size())))
    }

    fn rollback_consumed(&self, name: &str) {
        self.released.lock().remove(name);
    }

    /// Release a source tensor after model construction has durably replaced
    /// it. A tombstone makes every later lookup fail closed instead of exposing
    /// the stale device pointer retained in the immutable name map.
    pub fn release_consumed_alias(
        &self,
        name: &str,
        expected_ptr: DevicePtr,
        expected_dtype: WeightDtype,
        gpu: &dyn GpuBackend,
    ) -> Result<usize> {
        let Some((ptr, bytes)) = self.reserve_consumed_alias(name, expected_ptr, expected_dtype)?
        else {
            return Ok(0);
        };
        if let Err(error) = gpu.free(ptr) {
            self.rollback_consumed(name);
            return Err(error).with_context(|| format!("release consumed weight '{name}'"));
        }
        Ok(bytes)
    }

    /// Merge a separately loaded, disjoint sidecar store without copying GPU
    /// allocations. Duplicate names are rejected so a supplemental checkpoint
    /// cannot silently replace target weights.
    pub fn merge_disjoint(&mut self, other: Self) -> Result<()> {
        let Self {
            weights: other_weights,
            released: other_released,
            routed_groups: other_groups,
        } = other;
        ensure!(
            other_released.lock().is_empty(),
            "cannot merge a WeightStore containing consumed tensors"
        );
        ensure!(
            other_groups.is_empty(),
            "cannot merge a grouped sidecar store"
        );
        if let Some(name) = other_weights
            .keys()
            .find(|name| self.weights.contains_key(*name))
        {
            bail!("Weight sidecar duplicates target tensor '{name}'");
        }
        self.weights.extend(other_weights);
        Ok(())
    }

    /// Remove every live tensor named under `prefix` and return it so the
    /// caller can free the allocation. This is the only way a sidecar may
    /// replace target weights: the caller takes the whole sub-bank first, then
    /// `merge_disjoint` still rejects any name the sidecar would shadow.
    /// Consumed or grouped tensors fail closed because this map entry does not
    /// own their allocation individually.
    pub fn take_prefix(&mut self, prefix: &str) -> Result<Vec<(String, WeightTensor)>> {
        ensure!(
            !prefix.is_empty(),
            "take_prefix requires a non-empty prefix"
        );
        let mut names: Vec<String> = self
            .weights
            .keys()
            .filter(|name| name.starts_with(prefix))
            .cloned()
            .collect();
        names.sort_unstable();
        if let Some(name) = names
            .iter()
            .find(|name| self.released.lock().contains(name.as_str()))
        {
            bail!("cannot take consumed weight '{name}' under prefix '{prefix}'");
        }
        if let Some(name) = names.iter().find(|name| self.routed_groups.contains(name)) {
            bail!("cannot take grouped weight '{name}' under prefix '{prefix}'");
        }
        Ok(names
            .into_iter()
            .map(|name| {
                let tensor = self
                    .weights
                    .remove(&name)
                    .expect("name collected from this map");
                (name, tensor)
            })
            .collect())
    }

    /// Check if any tensor has FP8 dtype.
    pub fn has_fp8_weights(&self) -> bool {
        let released = self.released.lock();
        self.weights.iter().any(|(name, weight)| {
            !released.contains(name.as_str()) && matches!(weight.dtype, WeightDtype::FP8E4M3)
        })
    }
}

/// SBIO IORouter trait for weight loading.
pub trait WeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore>;
}

#[cfg(test)]
mod consumed_weight_tests {
    use super::*;

    fn store() -> WeightStore {
        WeightStore::from_map(HashMap::from([
            (
                "bf16".to_string(),
                WeightTensor {
                    ptr: DevicePtr(0x1000),
                    shape: vec![2, 4],
                    dtype: WeightDtype::BF16,
                },
            ),
            (
                "fp8".to_string(),
                WeightTensor {
                    ptr: DevicePtr(0x2000),
                    shape: vec![8],
                    dtype: WeightDtype::FP8E4M3,
                },
            ),
        ]))
    }

    #[test]
    fn reserved_consumed_weight_is_tombstoned_from_every_public_view() {
        let store = store();
        let (ptr, bytes) = store
            .reserve_consumed_alias("bf16", DevicePtr(0x1000), WeightDtype::BF16)
            .unwrap()
            .unwrap();
        assert_eq!(ptr, DevicePtr(0x1000));
        assert_eq!(bytes, 16);
        assert!(store.get("bf16").is_err());
        assert!(!store.contains("bf16"));
        assert_eq!(store.names().collect::<Vec<_>>(), ["fp8"]);
        assert_eq!(store.len(), 1);
        assert_eq!(store.total_bytes(), 8);
        assert!(store.has_fp8_weights());
        assert!(
            store
                .reserve_consumed_alias("bf16", DevicePtr(0x1000), WeightDtype::BF16)
                .is_err()
        );
        assert!(
            store
                .reserve_consumed_alias("missing", DevicePtr(0x1000), WeightDtype::BF16)
                .is_err()
        );
    }

    #[test]
    fn failed_device_free_can_roll_back_tombstone() {
        let store = store();
        store
            .reserve_consumed_alias("bf16", DevicePtr(0x1000), WeightDtype::BF16)
            .unwrap();
        store.rollback_consumed("bf16");
        assert_eq!(store.get("bf16").unwrap().ptr, DevicePtr(0x1000));
        assert!(store.contains("bf16"));
        assert_eq!(store.len(), 2);
        assert_eq!(store.total_bytes(), 24);
    }

    #[test]
    fn mismatched_alias_is_retained_without_a_tombstone() {
        let store = store();
        assert_eq!(
            store
                .reserve_consumed_alias("bf16", DevicePtr(0x1001), WeightDtype::BF16)
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .reserve_consumed_alias("bf16", DevicePtr(0x1000), WeightDtype::FP8E4M3)
                .unwrap(),
            None
        );
        assert_eq!(store.get("bf16").unwrap().ptr, DevicePtr(0x1000));
        assert_eq!(store.len(), 2);
    }

    fn tensor(ptr: u64) -> WeightTensor {
        WeightTensor {
            ptr: DevicePtr(ptr),
            shape: vec![4],
            dtype: WeightDtype::FP32,
        }
    }

    fn mtp_store() -> WeightStore {
        WeightStore::from_map(HashMap::from([
            ("mtp.fc_hidden.weight".to_string(), tensor(0x100)),
            (
                "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                tensor(0x200),
            ),
            (
                "mtp.layers.0.mlp.experts.1.down_proj.weight_scale_2".to_string(),
                tensor(0x300),
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                tensor(0x400),
            ),
        ]))
    }

    #[test]
    fn take_prefix_removes_only_the_named_sub_bank() {
        let mut store = mtp_store();
        let taken = store.take_prefix("mtp.layers.0.mlp.experts.").unwrap();
        let names: Vec<_> = taken.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "mtp.layers.0.mlp.experts.0.gate_proj.weight",
                "mtp.layers.0.mlp.experts.1.down_proj.weight_scale_2",
            ]
        );
        assert_eq!(taken[0].1.ptr, DevicePtr(0x200));
        assert_eq!(store.len(), 2);
        assert!(store.contains("mtp.fc_hidden.weight"));
        assert!(store.contains("model.language_model.layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(
            store
                .take_prefix("mtp.layers.0.mlp.experts.")
                .unwrap()
                .is_empty()
        );
        assert!(store.take_prefix("").is_err());
    }

    #[test]
    fn take_prefix_then_merge_replaces_a_colliding_bank() {
        let mut store = mtp_store();
        let sidecar = || {
            WeightStore::from_map(HashMap::from([
                (
                    "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_string(),
                    tensor(0x900),
                ),
                (
                    "mtp.layers.0.mlp.experts.0.gate_proj.input_scale".to_string(),
                    tensor(0x910),
                ),
            ]))
        };
        let error = store.merge_disjoint(sidecar()).unwrap_err().to_string();
        assert!(error.contains("duplicates target tensor"), "{error}");

        store.take_prefix("mtp.layers.0.mlp.experts.").unwrap();
        store.merge_disjoint(sidecar()).unwrap();
        assert_eq!(
            store
                .get("mtp.layers.0.mlp.experts.0.gate_proj.weight")
                .unwrap()
                .ptr,
            DevicePtr(0x900)
        );
        assert!(!store.contains("mtp.layers.0.mlp.experts.1.down_proj.weight_scale_2"));
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn take_prefix_refuses_a_consumed_tensor_without_removing_anything() {
        let mut store = mtp_store();
        store
            .reserve_consumed_alias(
                "mtp.layers.0.mlp.experts.0.gate_proj.weight",
                DevicePtr(0x200),
                WeightDtype::FP32,
            )
            .unwrap()
            .unwrap();
        assert!(store.take_prefix("mtp.layers.0.mlp.experts.").is_err());
        assert!(store.contains("mtp.layers.0.mlp.experts.1.down_proj.weight_scale_2"));
    }
}

/// Loads weights from safetensors files using mmap.
pub struct SafetensorsLoader {
    /// EP rank (0-based). Only used when ep_world_size > 1.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert tensors are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak memory multiplier in the pre-flight OOM check.
    /// Set from QuantFormat::peak_memory_multiplier() in the caller.
    /// When None, the pre-flight uses its own heuristic (1.3x NVFP4 / 1.5x FP8).
    pub peak_memory_multiplier: Option<f64>,
    /// Absolute bytes the *model builder* will retain on top of the weight
    /// store, added to the pre-flight peak. Architecture-dependent (see
    /// `spark-server`'s `construction_overhead_bytes`), so it cannot be
    /// expressed as a ratio of on-disk bytes: a ratio large enough for a
    /// hybrid linear-attention model false-OOMs a plain one. Zero means
    /// "unknown / nothing extra", which is the pre-existing behaviour.
    pub construction_overhead_bytes: usize,
    /// Optional exact tensor-name allowlist. When present, every tensor not
    /// named here is skipped before OOM estimation and allocation. This is
    /// used for small compatibility sidecars sourced from a full checkpoint
    /// (for example a DFlash donor embedding + LM head).
    pub tensor_allowlist: Option<HashSet<String>>,
    /// Extra "don't load this tensor" predicate, ORed with the EP rule and the allowlist.
    ///
    /// Unlike EP sharding this is unconditional — it applies at `ep_world_size == 1` and to
    /// `mtp.*` — because its user is DeepSeek-V4.1's engram table (~95 GB per layer, must never
    /// be loaded into the WeightStore; `deepseek_v41_engram::EngramGather` reads it directly
    /// from the checkpoint file instead). Skipped tensors are excluded from the pre-flight OOM
    /// estimate too, since the same predicate feeds `estimate_load_bytes`. Ported from
    /// dsv41/integration.
    pub extra_skip: Option<TensorSkipFn>,
}

/// A "don't load this tensor" predicate, shareable across the loader's pre-flight estimate and
/// its per-shard passes.
pub type TensorSkipFn = std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>;

impl Default for SafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            construction_overhead_bytes: 0,
            tensor_allowlist: None,
            extra_skip: None,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            construction_overhead_bytes: 0,
            tensor_allowlist: None,
            extra_skip: None,
        }
    }

    /// Check if a tensor should be skipped under EP, the allowlist, or [`Self::extra_skip`].
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped by the EP rule (small, fully replicated) — only
    /// `extra_skip` can drop them.
    fn should_skip_tensor(&self, name: &str) -> bool {
        if let Some(ref extra) = self.extra_skip
            && extra(name)
        {
            return true;
        }
        if self
            .tensor_allowlist
            .as_ref()
            .is_some_and(|allow| !allow.contains(name))
        {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        // MTP head experts are small — always replicate, never shard.
        if name.starts_with("mtp.") {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}

/// Parse expert index from tensor name (e.g. "model.layers.3.mlp.experts.42.gate_proj.weight" → 42).
pub(crate) fn parse_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

pub mod deepseek_v41_pack;
pub mod gguf;
mod loader;
pub mod mlx_int8;
pub(crate) use loader::{check_oom_guard, estimate_has_fp8, estimate_load_bytes};
