// SPDX-License-Identifier: AGPL-3.0-only

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.

use crate::gpu::GpuBackend;
use crate::weights::{
    WeightDtype, WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8,
    estimate_load_bytes, evict_page_cache, parse_expert_index,
};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

mod bounce_copy;
mod direct_io;
mod header;
mod routed_groups;

use routed_groups::RoutedBackingLoad;
pub use routed_groups::{
    ROUTED_GROUPS_EAGER_ENV, ROUTED_GROUPS_ENV, routed_grouping_eager_requested,
    routed_grouping_requested,
};

use bounce_copy::PinnedBounceCopier;

use header::{parse_header, resolve_shards};

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    pub peak_memory_multiplier: Option<f64>,
    /// Absolute bytes the *model builder* will retain on top of the weight
    /// store, added to the pre-flight peak. See the field of the same name
    /// on [`crate::weights::SafetensorsLoader`].
    pub construction_overhead_bytes: usize,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
    /// Optional tensor-name prefixes. When non-empty, indexed checkpoints
    /// load only matching tensors and skip unrelated/missing shard files.
    pub name_prefixes: Vec<String>,
    /// Extra "don't load this tensor" predicate, ORed with the EP rule; same contract as
    /// [`crate::weights::SafetensorsLoader::extra_skip`]. DeepSeek-V4.1 uses it to keep its two
    /// ~95 GB engram tables (row-gathered from NVMe, never resident) out of the store.
    pub extra_skip: Option<crate::weights::TensorSkipFn>,
    /// Strict canonical target-only NVFP4 groups. OFF by default. Serving
    /// validates architecture/geometry before enabling this diagnostic path.
    pub grouped_routed_originals: bool,
    /// `ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER=1`: allocate every planned group
    /// base before the first shard read. OFF by default; requires
    /// [`Self::grouped_routed_originals`].
    pub grouped_routed_originals_eager: bool,
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

const DIRECT_IO_TENSOR_CAP_ENV: &str = "ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP";

fn effective_direct_io_tensor_cap(
    configured_cap: usize,
    override_value: Option<&std::ffi::OsStr>,
) -> Result<usize> {
    let Some(value) = override_value else {
        return Ok(configured_cap);
    };
    let invalid = || {
        anyhow::anyhow!(
            "{DIRECT_IO_TENSOR_CAP_ENV} must be a nonempty ASCII decimal positive usize"
        )
    };
    let value = value.to_str().ok_or_else(invalid)?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid());
    }
    let cap = value.parse::<usize>().map_err(|_| invalid())?;
    if cap == 0 {
        return Err(invalid());
    }
    Ok(cap)
}

/// Diagnostic model for the per-allocation cost suspected on GB10. This is
/// deliberately not used to size or align allocations: only a runtime receipt
/// can establish the allocator's real granularity.
const HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES: usize = 64 * 1024;

/// CUDA allocations are at least 256-byte aligned, and several weight kernels
/// rely on that alignment even for tiny metadata tensors. Preserve the same
/// alignment when scalar expert metadata is sub-allocated from one slab.
const EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES: usize = 256;

/// Layout for immutable four-byte target-model expert metadata. These tensors
/// are read as scalar calibration values during model construction and are not
/// passed to `GpuBackend::free` individually. MTP tensors are deliberately
/// excluded because the plain MTP release path frees aliases by name. Keeping
/// only target-model metadata in a backend-owned slab therefore preserves the
/// old lifetime while avoiding hundreds of thousands of CUDA allocator calls.
#[derive(Debug)]
struct ExpertScalarSlabLayout {
    offsets: Vec<Option<usize>>,
    allocation_bytes: usize,
    tensor_count: usize,
    payload_bytes: usize,
}

fn is_expert_scalar_slab_candidate(meta: &header::TensorMeta) -> bool {
    meta.name.starts_with("model.language_model.layers.")
        && !meta.name.starts_with("mtp.")
        && meta.dtype == WeightDtype::FP32
        && meta.shape.is_empty()
        && meta.len == std::mem::size_of::<f32>()
        && meta.name.contains(".mlp.experts.")
        && (meta.name.ends_with(".weight_scale_2") || meta.name.ends_with(".input_scale"))
}

fn checked_align_up(bytes: usize, alignment: usize) -> Result<usize> {
    debug_assert!(alignment.is_power_of_two());
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab layout overflow"))
}

fn build_expert_scalar_slab_layout(
    tensors: &[header::TensorMeta],
) -> Result<ExpertScalarSlabLayout> {
    let mut offsets = Vec::with_capacity(tensors.len());
    // Leave the allocation base unexposed. All WeightTensor pointers are
    // interior views, while the backend exclusively owns/frees the base.
    let mut cursor = EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES;
    let mut tensor_count = 0usize;
    let mut payload_bytes = 0usize;

    for meta in tensors {
        if !is_expert_scalar_slab_candidate(meta) {
            offsets.push(None);
            continue;
        }
        let offset = checked_align_up(cursor, EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES)?;
        cursor = offset
            .checked_add(meta.len)
            .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab payload overflow"))?;
        offsets.push(Some(offset));
        tensor_count = tensor_count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab tensor-count overflow"))?;
        payload_bytes = payload_bytes
            .checked_add(meta.len)
            .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab byte-count overflow"))?;
    }

    let allocation_bytes = if tensor_count == 0 {
        0
    } else {
        checked_align_up(cursor, EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES)?
    };
    Ok(ExpertScalarSlabLayout {
        offsets,
        allocation_bytes,
        tensor_count,
        payload_bytes,
    })
}

/// Read contiguous runs of candidate scalars once, then scatter their exact
/// four-byte payloads into the aligned host image that will be copied to the
/// device slab. QAD5500 stores each shard's thousands of scalar entries
/// contiguously, so this also replaces thousands of four-byte `pread` calls
/// with one bounded read per run.
fn read_expert_scalar_slab(
    raw_fd: std::os::unix::io::RawFd,
    using_direct: bool,
    tensors: &[header::TensorMeta],
    layout: &ExpertScalarSlabLayout,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        tensors.len() == layout.offsets.len(),
        "fast-loader scalar slab layout does not match tensor list"
    );
    let mut host = vec![0u8; layout.allocation_bytes];
    let mut idx = 0usize;
    while idx < tensors.len() {
        if layout.offsets[idx].is_none() {
            idx += 1;
            continue;
        }

        let run_start_idx = idx;
        let run_file_start = tensors[idx].abs_offset;
        let mut run_file_end = run_file_start
            .checked_add(tensors[idx].len as u64)
            .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab file range overflow"))?;
        idx += 1;
        while idx < tensors.len()
            && layout.offsets[idx].is_some()
            && tensors[idx].abs_offset == run_file_end
        {
            run_file_end = run_file_end
                .checked_add(tensors[idx].len as u64)
                .ok_or_else(|| anyhow::anyhow!("fast-loader scalar slab file range overflow"))?;
            idx += 1;
        }

        let run_len = usize::try_from(run_file_end - run_file_start)
            .context("fast-loader scalar slab run is too large to address")?;
        let (buf, slice_start) =
            direct_io::read_tensor_aligned(raw_fd, run_file_start, run_len, using_direct)?;
        let run = &buf.as_slice()[slice_start..slice_start + run_len];
        for (tensor_idx, meta) in (run_start_idx..idx).zip(&tensors[run_start_idx..idx]) {
            let src_offset = usize::try_from(meta.abs_offset - run_file_start)
                .context("fast-loader scalar slab source offset is too large")?;
            let dst_offset = layout.offsets[tensor_idx]
                .expect("scalar slab run must contain only candidate tensors");
            host[dst_offset..dst_offset + meta.len]
                .copy_from_slice(&run[src_offset..src_offset + meta.len]);
        }
    }
    Ok(host)
}

#[derive(Debug, Default)]
struct FastLoadAllocationAudit {
    allocation_count: usize,
    requested_bytes: usize,
    hypothetical_64k_rounded_bytes: usize,
    managed_allocation_count: usize,
    managed_requested_bytes: usize,
    expert_scalar_slab_count: usize,
    expert_scalar_slab_tensor_count: usize,
    expert_scalar_slab_payload_bytes: usize,
    expert_scalar_slab_allocation_bytes: usize,
}

impl FastLoadAllocationAudit {
    fn record(&mut self, requested_bytes: usize, managed: bool) {
        self.allocation_count = self.allocation_count.saturating_add(1);
        self.requested_bytes = self.requested_bytes.saturating_add(requested_bytes);
        self.hypothetical_64k_rounded_bytes = self
            .hypothetical_64k_rounded_bytes
            .saturating_add(hypothetical_64k_round_up(requested_bytes));
        if managed {
            self.managed_allocation_count = self.managed_allocation_count.saturating_add(1);
            self.managed_requested_bytes =
                self.managed_requested_bytes.saturating_add(requested_bytes);
        }
    }

    fn record_expert_scalar_slab(
        &mut self,
        tensor_count: usize,
        payload_bytes: usize,
        allocation_bytes: usize,
        managed: bool,
    ) {
        self.record(allocation_bytes, managed);
        self.expert_scalar_slab_count = self.expert_scalar_slab_count.saturating_add(1);
        self.expert_scalar_slab_tensor_count = self
            .expert_scalar_slab_tensor_count
            .saturating_add(tensor_count);
        self.expert_scalar_slab_payload_bytes = self
            .expert_scalar_slab_payload_bytes
            .saturating_add(payload_bytes);
        self.expert_scalar_slab_allocation_bytes = self
            .expert_scalar_slab_allocation_bytes
            .saturating_add(allocation_bytes);
    }

    fn emit(
        &self,
        retained_tensor_count: usize,
        logical_store_bytes: usize,
        backend_free_before_bytes: usize,
        backend_free_after_bytes: Option<usize>,
    ) {
        let backend_available_delta_bytes =
            backend_free_after_bytes.map(|after| backend_free_before_bytes.saturating_sub(after));
        let backend_delta_over_requested_bytes =
            backend_available_delta_bytes.map(|delta| delta.saturating_sub(self.requested_bytes));
        let requested_over_backend_delta_bytes =
            backend_available_delta_bytes.map(|delta| self.requested_bytes.saturating_sub(delta));
        let hypothetical_64k_overhead_bytes = self
            .hypothetical_64k_rounded_bytes
            .saturating_sub(self.requested_bytes);

        tracing::info!(
            target: "atlas_weight_allocation_audit",
            allocation_count = self.allocation_count,
            retained_tensor_count = retained_tensor_count,
            requested_bytes = self.requested_bytes,
            logical_store_bytes = logical_store_bytes,
            managed_allocation_count = self.managed_allocation_count,
            managed_requested_bytes = self.managed_requested_bytes,
            expert_scalar_slab_count = self.expert_scalar_slab_count,
            expert_scalar_slab_tensor_count = self.expert_scalar_slab_tensor_count,
            expert_scalar_slab_payload_bytes = self.expert_scalar_slab_payload_bytes,
            expert_scalar_slab_allocation_bytes = self.expert_scalar_slab_allocation_bytes,
            backend_free_before_bytes = backend_free_before_bytes,
            backend_free_after_bytes = ?backend_free_after_bytes,
            backend_available_delta_bytes = ?backend_available_delta_bytes,
            backend_delta_over_requested_bytes = ?backend_delta_over_requested_bytes,
            requested_over_backend_delta_bytes = ?requested_over_backend_delta_bytes,
            hypothetical_64k_rounded_bytes = self.hypothetical_64k_rounded_bytes,
            hypothetical_64k_overhead_bytes = hypothetical_64k_overhead_bytes,
            "FAST_WEIGHT_ALLOCATION_AUDIT: hypothetical_64k_* models per-allocation \
             rounding only; it is not a measured allocator quantum. backend_free_* and \
             backend_available_delta_* use backend-specific availability accounting and \
             must not be interpreted as device-resident bytes without a matched receipt"
        );
    }
}

fn hypothetical_64k_round_up(bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    bytes
        .checked_add(HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES - 1)
        .map(|rounded| {
            rounded / HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES * HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES
        })
        .unwrap_or(usize::MAX)
}

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            construction_overhead_bytes: 0,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            name_prefixes: Vec::new(),
            extra_skip: None,
            grouped_routed_originals: false,
            grouped_routed_originals_eager: false,
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            construction_overhead_bytes: 0,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            name_prefixes: Vec::new(),
            extra_skip: None,
            grouped_routed_originals: false,
            grouped_routed_originals_eager: false,
        }
    }

    pub fn with_name_prefixes(mut self, prefixes: impl IntoIterator<Item = String>) -> Self {
        self.name_prefixes = prefixes.into_iter().collect();
        self
    }

    fn should_skip_tensor(&self, name: &str) -> bool {
        if let Some(ref extra) = self.extra_skip
            && extra(name)
        {
            return true;
        }
        if !self.name_prefixes.is_empty()
            && !self
                .name_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
        {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        if name.starts_with("mtp.") {
            return false;
        }
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
            false
        }
    }
}

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let direct_io_override = std::env::var_os(DIRECT_IO_TENSOR_CAP_ENV);
        let direct_io_tensor_cap = effective_direct_io_tensor_cap(
            self.direct_io_tensor_cap,
            direct_io_override.as_deref(),
        )?;
        if direct_io_override.is_some() {
            tracing::info!(
                "Fast loader: diagnostic {DIRECT_IO_TENSOR_CAP_ENV}={direct_io_tensor_cap} \
                 (configured cap {}, try_direct_io {}); selection only, \
                 O_DIRECT open may still fall back to buffered reads",
                self.direct_io_tensor_cap,
                self.try_direct_io,
            );
        }
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (mut shard_files, mut tensor_to_shard): (
            Vec<PathBuf>,
            Option<HashMap<String, String>>,
        ) = resolve_shards(model_dir)?;
        if !self.name_prefixes.is_empty() {
            let Some(ref mut map) = tensor_to_shard else {
                bail!("Prefix-filtered fast loading requires a safetensors index");
            };
            map.retain(|name, _| !self.should_skip_tensor(name));
            let wanted_shards: std::collections::HashSet<&str> =
                map.values().map(String::as_str).collect();
            shard_files.retain(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| wanted_shards.contains(name))
            });
            if map.is_empty() || shard_files.is_empty() {
                bail!(
                    "No tensors matched fast-loader prefixes {:?}",
                    self.name_prefixes
                );
            }
        }

        let extra = model_dir.join("extra_weights.safetensors");
        let mut routed_load = if self.grouped_routed_originals {
            anyhow::ensure!(
                gpu.supports_grouped_backing(),
                "backend does not support explicit grouped backing ownership"
            );
            anyhow::ensure!(
                self.ep_rank == 0 && self.ep_world_size == 1 && self.name_prefixes.is_empty(),
                "routed grouping requires an unfiltered EP1 target loader"
            );
            Some(RoutedBackingLoad::plan(
                &shard_files,
                tensor_to_shard.as_ref(),
                &skip_fn,
                extra.exists().then_some(extra.as_path()),
            )?)
        } else {
            None
        };

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        {
            let estimated = estimate_load_bytes(&shard_files, &skip_fn)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &skip_fn)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            // The multiplier covers the store + its transient staging;
            // buffers the model builder retains afterwards are
            // architecture-dependent and come in as absolute bytes.
            let construction = self.construction_overhead_bytes;
            let grouped_overhead = routed_load
                .as_ref()
                .map_or(0, RoutedBackingLoad::overhead_bytes);
            let peak = ((estimated as f64 * mult) as usize)
                .checked_add(construction)
                .and_then(|n| n.checked_add(grouped_overhead))
                .context("fast-load peak estimate overflow")?;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead \
                 + {:.2} GB model-construction + {:.2} GB grouped-routed overhead \
                 = {:.2} GB peak, {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(construction),
                gib(grouped_overhead),
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            crate::progress::preflight(gib(estimated), gib(free));
            if peak.saturating_add(oom_reserve_bytes) > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB ({:.2} GB weights × {:.1}x \
                     + {:.2} GB retained model-construction buffers \
                     + {:.2} GB grouped-routed overhead) \
                     + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(estimated),
                    mult,
                    gib(construction),
                    gib(grouped_overhead),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Load each shard. Reserve from the index up front: Flash-Next has
        // roughly 300k expert tensors, and growing the map at the ~200k
        // boundary while 50+ GiB of UVM is resident can turn a sub-second
        // rehash into minutes of unified-memory pressure.
        let expected_tensors = tensor_to_shard.as_ref().map_or(0, HashMap::len);
        let mut weights: HashMap<String, WeightTensor> = HashMap::with_capacity(expected_tensors);
        if expected_tensors > 0 {
            tracing::info!("Fast loader: reserved weight index for {expected_tensors} tensors");
        }
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;
        let mut allocation_audit = FastLoadAllocationAudit::default();
        if self.grouped_routed_originals_eager {
            // Every planned base exists before the first shard read; the
            // pre-flight above already counts their bytes, so admission and
            // peak are unchanged. Per-shard first views then find their base.
            let load = routed_load
                .as_mut()
                .context("eager routed groups require ATLAS_FAST_LOAD_ROUTED_GROUPS=1")?;
            let sizes = load.allocate_all(gpu)?;
            for &bytes in &sizes {
                allocation_audit.record(bytes, false);
            }
            tracing::info!(target: "atlas_weight_allocation_audit",
                bases = sizes.len(), bytes = sizes.iter().sum::<usize>(),
                "FAST_WEIGHT_ROUTED_GROUPS_EAGER: every planned base allocated before the first shard read");
        }

        let mut shards_to_tensors: Option<HashMap<String, Vec<String>>> =
            tensor_to_shard.as_ref().map(|map| {
                let mut grouped: HashMap<String, Vec<String>> =
                    HashMap::with_capacity(shard_files.len());
                for (tensor_name, shard_file) in map {
                    grouped
                        .entry(shard_file.clone())
                        .or_default()
                        .push(tensor_name.clone());
                }
                grouped
            });

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = shards_to_tensors
                .as_mut()
                .map(|grouped| grouped.remove(shard_name).unwrap_or_default());

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );
            crate::progress::shard_start(i + 1, total_shards, shard_name);

            load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                self.try_direct_io,
                direct_io_tensor_cap,
                &mut weights,
                &mut offload_logged,
                &mut allocation_audit,
                routed_load.as_mut(),
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
            let used = initial_free.saturating_sub(free_now);
            tracing::info!(
                "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            crate::progress::shard_done(
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if !offload_logged {
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let no_skip = |_: &str| false;
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            load_shard_fast(
                &extra,
                None,
                gpu,
                &no_skip,
                self.try_direct_io,
                direct_io_tensor_cap,
                &mut weights,
                &mut extra_offload,
                &mut allocation_audit,
                None,
            )?;
        }

        let logical_store_bytes = weights
            .values()
            .map(WeightTensor::byte_size)
            .fold(0usize, usize::saturating_add);
        allocation_audit.emit(
            weights.len(),
            logical_store_bytes,
            initial_free,
            gpu.free_memory().ok(),
        );
        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        if let Some(routed_load) = routed_load {
            WeightStore::with_routed_groups(weights, routed_load.finish()?)
        } else {
            Ok(WeightStore::from_map(weights))
        }
    }
}

/// `ATLAS_LOAD_TRACE=1`: one line per step and per tensor in
/// `load_shard_fast` (filter, skip_fn, reader, alloc, H2D) with elapsed
/// times, so a stalled load names the tensor and the step it stopped in.
fn load_trace_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_LOAD_TRACE").ok().as_deref() == Some("1"))
}

macro_rules! load_trace {
    ($($arg:tt)*) => {
        if load_trace_enabled() {
            tracing::info!(target: "atlas_load_trace", $($arg)*);
        }
    };
}

/// Load a single shard with optional O_DIRECT coalesced reads + pipelined copy.
///
/// Pipeline:
///   reader thread: coalesces contiguous tensor runs up to 16 MiB into one
///                  `AlignedBuffer` via `pread` -> `sync_channel(32)` ->
///   main thread:   recv -> `PinnedBounceCopier` (or `copy_h2d_managed` on UVM
///                  fallback) -> insert into `out`.
#[allow(clippy::too_many_arguments)]
fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    try_direct_io: bool,
    direct_io_tensor_cap: usize,
    out: &mut HashMap<String, WeightTensor>,
    offload_logged: &mut bool,
    allocation_audit: &mut FastLoadAllocationAudit,
    mut routed_load: Option<&mut RoutedBackingLoad>,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let t_shard = std::time::Instant::now();
    load_trace!("shard {}: parse_header start", shard_path.display());
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();
    load_trace!(
        "shard: header parsed, {} tensors ({:?})",
        tensors.len(),
        t_shard.elapsed()
    );

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: std::collections::HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    load_trace!(
        "shard: tensor_filter applied, {} tensors ({:?})",
        tensors.len(),
        t_shard.elapsed()
    );
    tensors.retain(|t| !skip_fn(&t.name));
    load_trace!(
        "shard: skip_fn applied, {} tensors ({:?})",
        tensors.len(),
        t_shard.elapsed()
    );

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;
    if try_direct_io && !wants_direct {
        tracing::info!(
            "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
            tensors.len(),
            direct_io_tensor_cap
        );
    }

    // File for data reads. Try O_DIRECT; if it fails, fall through to buffered.
    let (direct_file, using_direct) = match wants_direct
        .then(|| direct_io::open_direct(shard_path))
        .transpose()
    {
        Ok(Some(f)) => (Some(f), true),
        Ok(None) => (None, false),
        Err(e) => {
            tracing::warn!(
                "O_DIRECT open failed for {} ({e}); falling back to buffered reads",
                shard_path.display()
            );
            (None, false)
        }
    };
    let buffered_file = File::open(shard_path)?;
    let data_fd = direct_file.as_ref().unwrap_or(&buffered_file);
    let scalar_raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };

    // QAD-style target routed-expert banks contain one FP32 scalar per
    // projection for both `weight_scale_2` and `input_scale`. On large MoE
    // checkpoints that is ~150k four-byte CUDA allocations. Target scalars
    // have model lifetime, so stage them into one aligned slab per shard while
    // leaving MTP and every independently-freeable tensor on the established
    // allocation path.
    let expert_scalar_layout = build_expert_scalar_slab_layout(&tensors)?;
    let expert_scalar_slab = if expert_scalar_layout.tensor_count == 0 {
        None
    } else {
        let (base, managed) = match gpu.alloc(expert_scalar_layout.allocation_bytes) {
            Ok(ptr) => (ptr, false),
            Err(error) => {
                if !*offload_logged {
                    tracing::warn!(
                        "GPU alloc failed for expert scalar slab ({} bytes; {error:#}) — \
                         switching to managed (UVM) memory",
                        expert_scalar_layout.allocation_bytes
                    );
                    *offload_logged = true;
                }
                (
                    gpu.alloc_managed(expert_scalar_layout.allocation_bytes)?,
                    true,
                )
            }
        };
        let host =
            read_expert_scalar_slab(scalar_raw_fd, using_direct, &tensors, &expert_scalar_layout)?;
        allocation_audit.record_expert_scalar_slab(
            expert_scalar_layout.tensor_count,
            expert_scalar_layout.payload_bytes,
            expert_scalar_layout.allocation_bytes,
            managed,
        );
        tracing::info!(
            target: "atlas_weight_allocation_audit",
            tensor_count = expert_scalar_layout.tensor_count,
            payload_bytes = expert_scalar_layout.payload_bytes,
            allocation_bytes = expert_scalar_layout.allocation_bytes,
            alignment_bytes = EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES,
            managed = managed,
            "FAST_WEIGHT_EXPERT_SCALAR_SLAB"
        );
        Some((base, managed, host))
    };

    // Pipelined reader: sends (tensor_index, aligned_buffer, slice_start) to main.
    // Contiguous tensor runs up to 16 MiB share one coalesced NVMe pread buffer.
    type ReadMsg = (usize, std::sync::Arc<direct_io::AlignedBuffer>, usize);
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(32);
    let tensors_for_reader: Vec<(usize, u64, usize)> = tensors
        .iter()
        .enumerate()
        .filter(|(idx, _)| expert_scalar_layout.offsets[*idx].is_none())
        .map(|(idx, tensor)| (idx, tensor.abs_offset, tensor.len))
        .collect();

    let _ = file_size; // retained for future use (tail-fragment buffered read)
    // Error unwind can detach the reader while closing the original files.
    // Move an owned duplicate of the selected file, never just its raw fd.
    let reader_file = data_fd
        .try_clone()
        .context("failed to duplicate fast-loader reader descriptor")?;
    let reader_handle = std::thread::spawn(move || {
        use std::os::unix::io::AsRawFd;
        const MAX_COALESCED_READ_BYTES: usize = 16 * 1024 * 1024;
        let raw_fd = reader_file.as_raw_fd();
        let mut pos = 0usize;
        while pos < tensors_for_reader.len() {
            let (first_idx, run_start, first_len) = tensors_for_reader[pos];
            let mut run_end = run_start.saturating_add(first_len as u64);
            let mut end_pos = pos + 1;
            while end_pos < tensors_for_reader.len() {
                let (_, next_offset, next_len) = tensors_for_reader[end_pos];
                let Some(candidate_end) = run_end.checked_add(next_len as u64) else {
                    break;
                };
                if next_offset != run_end
                    || (candidate_end - run_start) > MAX_COALESCED_READ_BYTES as u64
                {
                    break;
                }
                run_end = candidate_end;
                end_pos += 1;
            }
            let run_len = (run_end - run_start) as usize;
            let t_read = std::time::Instant::now();
            match direct_io::read_tensor_aligned(raw_fd, run_start, run_len, using_direct) {
                Ok((buf, base_slice_start)) => {
                    let shared_buf = std::sync::Arc::new(buf);
                    load_trace!(
                        "reader: #{first_idx}..#{} ({} tensors, {run_len} bytes) read ({:?}) ok=true",
                        tensors_for_reader[end_pos - 1].0,
                        end_pos - pos,
                        t_read.elapsed()
                    );
                    let mut send_failed = false;
                    for &(idx, abs_offset, _len) in &tensors_for_reader[pos..end_pos] {
                        let rel_offset = (abs_offset - run_start) as usize;
                        if tx
                            .send(Ok((
                                idx,
                                std::sync::Arc::clone(&shared_buf),
                                base_slice_start + rel_offset,
                            )))
                            .is_err()
                        {
                            send_failed = true;
                            break;
                        }
                    }
                    if send_failed {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err));
                    break;
                }
            }
            pos = end_pos;
        }
    });

    // Copier: drains the channel, does gpu alloc + copy_h2d, inserts into the map.
    //
    // The H2D leg uses a pinned bounce buffer, not the plain pageable
    // `gpu.copy_h2d`: that call is a blocking, timeout-free driver
    // synchronize that has been observed to stall indefinitely on an
    // ordinary few-MB tensor with tens of GB of GPU memory free. The bounce
    // copier stages through page-locked memory and waits on a polled,
    // bounded timeout instead — see `bounce_copy` for why.
    let max_tensor_bytes = tensors.iter().map(|tensor| tensor.len).max().unwrap_or(0);
    let mut bounce =
        PinnedBounceCopier::new(gpu, max_tensor_bytes, expert_scalar_layout.allocation_bytes)?;
    if let Some((base, managed, host)) = expert_scalar_slab.as_ref() {
        if *managed {
            gpu.copy_h2d_managed(host, *base)?;
        } else {
            bounce.copy(host, *base)?;
        }
        for (idx, offset) in expert_scalar_layout.offsets.iter().enumerate() {
            let Some(offset) = offset else { continue };
            let meta = &tensors[idx];
            out.insert(
                meta.name.clone(),
                WeightTensor {
                    ptr: base.offset(*offset),
                    shape: meta.shape.clone(),
                    dtype: meta.dtype,
                },
            );
        }
    }
    for result in rx {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let src = &buf.as_slice()[slice_start..slice_start + meta.len];

        let t_tensor = std::time::Instant::now();
        load_trace!(
            "copier: #{idx} {} {} bytes alloc start",
            meta.name,
            meta.len
        );
        let grouped_destination = routed_load
            .as_deref_mut()
            .map(|load| load.destination(shard_path, meta, gpu))
            .transpose()?
            .flatten();
        let (ptr, _managed) = if let Some((base, offset, allocated_bytes)) = grouped_destination {
            if let Some(bytes) = allocated_bytes {
                allocation_audit.record(bytes, false);
            }
            bounce.copy_into_owned_range(src, base, offset)?;
            (base.offset(offset), false)
        } else {
            let allocation = match gpu.alloc(meta.len) {
                Ok(p) => {
                    load_trace!(
                        "copier: #{idx} alloc done ({:?}), h2d start",
                        t_tensor.elapsed()
                    );
                    bounce.copy(src, p)?;
                    load_trace!("copier: #{idx} h2d done ({:?})", t_tensor.elapsed());
                    (p, false)
                }
                Err(_) => {
                    if !*offload_logged {
                        tracing::warn!(
                            "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory",
                            meta.name,
                            meta.len
                        );
                        *offload_logged = true;
                    }
                    let p = gpu.alloc_managed(meta.len)?;
                    gpu.copy_h2d_managed(src, p)?;
                    (p, true)
                }
            };
            allocation_audit.record(meta.len, allocation.1);
            allocation
        };

        out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }

    // Drop also bounded-drains on every earlier error/unwind. Uncertain DMA
    // retains its pinned source and backend-owned device allocation.
    bounce.drain()?;
    load_trace!(
        "shard: copier drained ({:?}), joining reader",
        t_shard.elapsed()
    );
    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_io_cap_unset_preserves_configured_values() -> Result<()> {
        for configured in [0, 1, DEFAULT_DIRECT_IO_TENSOR_CAP, usize::MAX] {
            assert_eq!(
                effective_direct_io_tensor_cap(configured, None)?,
                configured
            );
        }
        Ok(())
    }

    #[test]
    fn direct_io_cap_accepts_only_positive_ascii_decimal_usize() -> Result<()> {
        for (value, expected) in [("1", 1), ("20000", 20_000), ("00020000", 20_000)] {
            assert_eq!(
                effective_direct_io_tensor_cap(5000, Some(std::ffi::OsStr::new(value)))?,
                expected,
            );
        }
        assert_eq!(
            effective_direct_io_tensor_cap(
                5000,
                Some(std::ffi::OsStr::new(&usize::MAX.to_string())),
            )?,
            usize::MAX,
        );
        Ok(())
    }

    #[test]
    fn direct_io_cap_rejects_zero_whitespace_signs_unicode_and_overflow() {
        for value in [
            "",
            "0",
            "0000",
            " 20000",
            "20000 ",
            "20\n000",
            "\t20000",
            "+20000",
            "-1",
            "20_000",
            "2e4",
            "0x5000",
            "\u{ff12}0000",
            "\u{0662}0000",
            "20000\0",
        ] {
            assert!(
                effective_direct_io_tensor_cap(5000, Some(std::ffi::OsStr::new(value))).is_err(),
                "accepted invalid cap {value:?}",
            );
        }
        let overflow = format!("{}0", usize::MAX);
        assert!(
            effective_direct_io_tensor_cap(5000, Some(std::ffi::OsStr::new(&overflow))).is_err()
        );
    }

    #[test]
    fn direct_io_cap_rejects_non_unicode_environment_bytes() {
        use std::os::unix::ffi::OsStrExt;

        let value = std::ffi::OsStr::from_bytes(&[b'2', 0xff, b'0']);
        assert!(effective_direct_io_tensor_cap(5000, Some(value)).is_err());
    }

    #[test]
    fn reader_owned_clone_survives_parent_drop_and_path_removal() -> Result<()> {
        use std::io::Write;
        use std::os::fd::AsRawFd;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "spark-fast-reader-fd-{}-{unique}.bin",
            std::process::id()
        ));
        let mut parent_file = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        parent_file.write_all(&[1, 2, 3, 4])?;
        let reader_file = parent_file.try_clone()?;
        // The worker cannot reopen this path, and the parent closes its fd
        // before releasing the worker to read from its independent owner.
        std::fs::remove_file(&path)?;
        let (ready, wait) = sync_channel::<()>(0);
        let reader = std::thread::spawn(move || -> Result<Vec<u8>> {
            let raw_fd = reader_file.as_raw_fd();
            wait.recv_timeout(std::time::Duration::from_secs(2))?;
            let (buffer, start) = direct_io::read_tensor_aligned(raw_fd, 0, 4, false)?;
            Ok(buffer.as_slice()[start..start + 4].to_vec())
        });
        drop(parent_file);
        ready.send(())?;
        let bytes = reader
            .join()
            .map_err(|_| anyhow::anyhow!("reader ownership regression panicked"))??;
        assert_eq!(bytes, [1, 2, 3, 4]);
        Ok(())
    }

    fn meta(
        name: &str,
        dtype: WeightDtype,
        shape: Vec<usize>,
        abs_offset: u64,
        len: usize,
    ) -> header::TensorMeta {
        header::TensorMeta {
            name: name.to_string(),
            dtype,
            shape,
            abs_offset,
            len,
        }
    }

    #[test]
    fn allocation_audit_models_rounding_per_allocation() {
        let quantum = HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES;
        let mut audit = FastLoadAllocationAudit::default();

        audit.record(1, false);
        audit.record(quantum - 1, true);
        audit.record(quantum, false);
        audit.record(quantum + 1, true);

        assert_eq!(audit.allocation_count, 4);
        assert_eq!(audit.requested_bytes, 3 * quantum + 1);
        assert_eq!(audit.hypothetical_64k_rounded_bytes, 5 * quantum);
        assert_eq!(audit.managed_allocation_count, 2);
        assert_eq!(audit.managed_requested_bytes, 2 * quantum);
    }

    #[test]
    fn expert_scalar_slab_is_narrow_and_preserves_device_alignment() {
        let tensors = vec![
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                WeightDtype::FP32,
                vec![],
                100,
                4,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                WeightDtype::UInt8,
                vec![2, 8],
                104,
                16,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale",
                WeightDtype::FP32,
                vec![],
                120,
                4,
            ),
            meta(
                "model.language_model.layers.0.self_attn.q_proj.input_scale",
                WeightDtype::FP32,
                vec![],
                124,
                4,
            ),
        ];

        let layout = build_expert_scalar_slab_layout(&tensors).unwrap();
        assert_eq!(
            layout.offsets,
            vec![
                Some(EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES),
                None,
                Some(2 * EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES),
                None,
            ]
        );
        assert_eq!(layout.tensor_count, 2);
        assert_eq!(layout.payload_bytes, 8);
        assert_eq!(
            layout.allocation_bytes,
            3 * EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES
        );
        assert!(
            layout
                .offsets
                .iter()
                .flatten()
                .all(|offset| offset % EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES == 0)
        );
    }

    #[test]
    fn expert_scalar_slab_rejects_near_miss_metadata() {
        for tensor in [
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                WeightDtype::BF16,
                vec![],
                0,
                4,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                WeightDtype::FP32,
                vec![1],
                0,
                4,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                WeightDtype::FP32,
                vec![],
                0,
                8,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                WeightDtype::FP32,
                vec![],
                0,
                4,
            ),
        ] {
            assert!(!is_expert_scalar_slab_candidate(&tensor));
        }
    }

    #[test]
    fn expert_scalar_slab_keeps_mtp_release_aliases_independent() {
        for suffix in ["weight_scale_2", "input_scale"] {
            let mtp = meta(
                &format!("mtp.layers.0.mlp.experts.0.gate_proj.{suffix}"),
                WeightDtype::FP32,
                vec![],
                0,
                4,
            );
            assert!(
                !is_expert_scalar_slab_candidate(&mtp),
                "plain MTP release must receive an independently allocated pointer"
            );

            let target = meta(
                &format!("model.language_model.layers.0.mlp.experts.0.gate_proj.{suffix}"),
                WeightDtype::FP32,
                vec![],
                0,
                4,
            );
            assert!(is_expert_scalar_slab_candidate(&target));
        }
    }

    #[test]
    fn expert_scalar_slab_reads_contiguous_and_split_runs_exactly() -> Result<()> {
        use std::io::Write;
        use std::os::fd::AsRawFd;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "spark-fast-weight-scalar-slab-{}-{unique}.bin",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)?;
        let mut file_bytes = vec![0xA5; 32];
        file_bytes[8..12].copy_from_slice(&[1, 2, 3, 4]);
        file_bytes[12..16].copy_from_slice(&[5, 6, 7, 8]);
        file_bytes[24..28].copy_from_slice(&[9, 10, 11, 12]);
        file.write_all(&file_bytes)?;

        let tensors = vec![
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                WeightDtype::FP32,
                vec![],
                8,
                4,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale",
                WeightDtype::FP32,
                vec![],
                12,
                4,
            ),
            meta(
                "model.language_model.layers.0.self_attn.q_proj.input_scale",
                WeightDtype::FP32,
                vec![],
                16,
                4,
            ),
            meta(
                "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale_2",
                WeightDtype::FP32,
                vec![],
                24,
                4,
            ),
        ];
        let layout = build_expert_scalar_slab_layout(&tensors)?;
        let host = read_expert_scalar_slab(file.as_raw_fd(), false, &tensors, &layout)?;

        let scalar_bytes: Vec<&[u8]> = layout
            .offsets
            .iter()
            .flatten()
            .map(|offset| &host[*offset..*offset + std::mem::size_of::<f32>()])
            .collect();
        assert_eq!(
            scalar_bytes,
            vec![&[1, 2, 3, 4][..], &[5, 6, 7, 8][..], &[9, 10, 11, 12][..]]
        );
        assert_eq!(layout.tensor_count, 3);
        assert_eq!(layout.payload_bytes, 12);
        assert_eq!(
            layout.allocation_bytes,
            4 * EXPERT_SCALAR_SLAB_ALIGNMENT_BYTES
        );
        drop(file);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn allocation_audit_counts_one_backend_allocation_per_scalar_slab() {
        let mut audit = FastLoadAllocationAudit::default();
        audit.record_expert_scalar_slab(4_608, 18_432, 1_179_904, false);

        assert_eq!(audit.allocation_count, 1);
        assert_eq!(audit.requested_bytes, 1_179_904);
        assert_eq!(audit.expert_scalar_slab_count, 1);
        assert_eq!(audit.expert_scalar_slab_tensor_count, 4_608);
        assert_eq!(audit.expert_scalar_slab_payload_bytes, 18_432);
        assert_eq!(audit.expert_scalar_slab_allocation_bytes, 1_179_904);
    }

    #[test]
    fn hypothetical_rounding_is_zero_safe_and_saturating() {
        assert_eq!(hypothetical_64k_round_up(0), 0);
        assert_eq!(
            hypothetical_64k_round_up(HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES),
            HYPOTHETICAL_ALLOCATION_QUANTUM_BYTES
        );
        assert_eq!(hypothetical_64k_round_up(usize::MAX), usize::MAX);
    }
}
