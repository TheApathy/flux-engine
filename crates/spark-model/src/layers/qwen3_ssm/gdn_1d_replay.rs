// SPDX-License-Identifier: AGPL-3.0-only

//! Layer-level integration for native SM120/SM121 1D GDN correction-record
//! verify & device-driven cross-layer replay commit (`ATLAS_GDN_1D_REPLAY=1`).
//!
//! When `ATLAS_GDN_1D_REPLAY` is unset or `"0"` (default), this module is a
//! zero-overhead no-op and the baseline verify/commit path runs unchanged.
//!
//! When `ATLAS_GDN_1D_REPLAY=1` is active:
//! - During `ExactSequence` verify, each SSM layer records compact 1D FP32
//!   `(delta, k, decay)` vectors into its lazy per-layer [`Gdn1dReplayArena`]
//!   without mutating `h_state` or writing 2D `h_state_intermediates`
//!   (`63.75x`–`95.4x` less DRAM write traffic).
//! - Each recorded verify window binds its slot [`GdnCommitSlotDesc`] to a
//!   live `generation` token and enforces `sharing_count <= 1`.
//! - A single device-driven cross-layer commit launch
//!   ([`Gdn1dReplayCrossLayerCommit::launch`]) reads device-resident
//!   `d_num_accepted` (`0` host sync) across all SSM layers and replays
//!   `H = decay * H + k (x) delta` bit-identically in-place.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::ops::{
    self, GDN_1D_HEAD_DIM, Gdn1dRecordGeometry, Gdn1dReplayPolicy, GdnCommitReplay1dLaunchArgs,
    GdnCommitSlotDesc, GdnCorrectionReplayKernels, GdnVerifyNumericConfig,
    GdnVerifyRecord1dLaunchArgs, GdnZeroTailRowsLaunchArgs,
};

/// Whether an arena allocation call occurs inside an active CUDA graph capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn1dAllocContext {
    /// Normal execution outside CUDA graph capture (allocation allowed).
    OutsideGraphCapture,
    /// Active CUDA graph capture (lazy allocation fails closed).
    DuringGraphCapture,
}

/// Outcome of attempting the opt-in 1D GDN correction-record verify route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn1dReplayVerifyOutcome {
    /// `ATLAS_GDN_1D_REPLAY` is OFF — caller continues on baseline path.
    Disabled,
    /// `gdn_verify_record_1d_sm120` ran and captured compact 1D correction records.
    Executed,
}

/// Per-layer device arena holding compact 1D FP32 `(delta, k, decay)` records
/// and the last recorded single-sequence slot guard state.
#[derive(Debug)]
pub struct Gdn1dReplayArena {
    base: DevicePtr,
    geometry: Gdn1dRecordGeometry,
    slot_state: Gdn1dReplaySlotState,
}

/// Fail-closed generation and sharing guard for one recorded GDN slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gdn1dReplaySlotState {
    pub h_state_ptr: DevicePtr,
    pub delta_ptr: DevicePtr,
    pub k_ptr: DevicePtr,
    pub decay_ptr: DevicePtr,
    pub generation: u64,
    pub expected_generation: u64,
    pub sharing_count: u32,
    pub live_tokens: u32,
    pub max_tokens: u32,
    pub recorded: bool,
}

impl Gdn1dReplaySlotState {
    /// Create an unrecorded slot state initialized at generation `1` with
    /// exclusive ownership (`sharing_count = 1`).
    #[must_use]
    pub const fn new_exclusive(
        delta_ptr: DevicePtr,
        k_ptr: DevicePtr,
        decay_ptr: DevicePtr,
    ) -> Self {
        Self {
            h_state_ptr: DevicePtr::NULL,
            delta_ptr,
            k_ptr,
            decay_ptr,
            generation: 1,
            expected_generation: 0,
            sharing_count: 1,
            live_tokens: 0,
            max_tokens: 0,
            recorded: false,
        }
    }

    /// Bind a newly recorded verify window to the current live generation.
    ///
    /// # Errors
    /// Fails closed if `sharing_count > 1`, `h_state_ptr` is null/misaligned,
    /// or `live_tokens > max_tokens` / `max_tokens == 0`.
    pub fn record_verify(
        &mut self,
        h_state_ptr: DevicePtr,
        live_tokens: u32,
        max_tokens: u32,
    ) -> Result<()> {
        ensure!(
            self.sharing_count <= 1,
            "cannot record 1D GDN verify on shared slot (sharing_count={})",
            self.sharing_count
        );
        ensure!(
            !h_state_ptr.is_null() && h_state_ptr.0.is_multiple_of(16),
            "1D GDN verify h_state_ptr ({:#x}) must be non-null and 16-byte aligned",
            h_state_ptr.0
        );
        ensure!(
            max_tokens > 0 && live_tokens <= max_tokens,
            "1D GDN verify invalid window live_tokens={live_tokens}, max_tokens={max_tokens}"
        );
        self.h_state_ptr = h_state_ptr;
        self.expected_generation = self.generation;
        self.live_tokens = live_tokens;
        self.max_tokens = max_tokens;
        self.recorded = true;
        Ok(())
    }

    /// Advance the live slot generation (e.g., on rollback, reset, or eviction),
    /// invalidating any stale uncommitted 1D record fail-closed.
    pub fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    /// Update the active sharing count (`> 1` forces fail-closed rejection on commit).
    pub fn set_sharing_count(&mut self, sharing_count: u32) {
        self.sharing_count = sharing_count;
    }

    /// Build the 64-byte ABI descriptor for `gdn_commit_replay_1d_sm120`.
    ///
    /// # Errors
    /// Returns an error if no verify window has been recorded yet.
    pub fn to_commit_desc(&self) -> Result<GdnCommitSlotDesc> {
        ensure!(
            self.recorded,
            "GDN 1D slot has no recorded correction window to commit"
        );
        Ok(GdnCommitSlotDesc {
            h_state_ptr: self.h_state_ptr.0,
            delta_ptr: self.delta_ptr.0,
            k_ptr: self.k_ptr.0,
            decay_ptr: self.decay_ptr.0,
            generation: self.generation,
            expected_generation: self.expected_generation,
            sharing_count: self.sharing_count,
            max_tokens: self.max_tokens,
            flags: 0,
            pad: 0,
        })
    }
}

impl Gdn1dReplayArena {
    /// Allocate a contiguous 16-byte-aligned device arena for `(delta, k, decay)`.
    ///
    /// # Errors
    /// Returns an error if allocation fails or if invoked during CUDA graph capture.
    pub fn allocate(
        gpu: &dyn GpuBackend,
        geometry: Gdn1dRecordGeometry,
        alloc_context: Gdn1dAllocContext,
        stream: u64,
    ) -> Result<Self> {
        Self::allocate_carrying(gpu, geometry, alloc_context, stream, None)
    }

    /// Allocate a replacement arena while carrying forward `generation + 1` and
    /// `sharing_count` from `prior_slot` so arena resizing never resets generation
    /// back to `1` or clears an active `sharing_count > 1` guard.
    ///
    /// # Errors
    /// Returns an error if allocation fails or if invoked during CUDA graph capture.
    pub fn allocate_carrying(
        gpu: &dyn GpuBackend,
        geometry: Gdn1dRecordGeometry,
        alloc_context: Gdn1dAllocContext,
        stream: u64,
        prior_slot: Option<&Gdn1dReplaySlotState>,
    ) -> Result<Self> {
        let capturing = matches!(alloc_context, Gdn1dAllocContext::DuringGraphCapture)
            || spark_runtime::gpu_capture_region::active_capture_region().is_some()
            || gpu.stream_is_capturing(stream)
            || gpu.stream_is_capturing(gpu.default_stream());
        ensure!(
            !capturing,
            "ATLAS_GDN_1D_REPLAY=1 cannot lazily allocate 1D correction arena during CUDA graph capture; call preallocate_gdn_1d_replay_arena first"
        );
        let base = gpu.alloc(geometry.total_arena_bytes).with_context(|| {
            format!(
                "allocate GDN 1D correction-record arena ({} bytes)",
                geometry.total_arena_bytes
            )
        })?;
        let delta_ptr = base;
        let k_ptr = base.offset(geometry.k_offset_bytes);
        let decay_ptr = base.offset(geometry.decay_offset_bytes);
        let mut slot_state = Gdn1dReplaySlotState::new_exclusive(delta_ptr, k_ptr, decay_ptr);
        if let Some(prev) = prior_slot {
            slot_state.generation = prev.generation.wrapping_add(1).max(1);
            slot_state.sharing_count = prev.sharing_count;
        }
        Ok(Self {
            base,
            geometry,
            slot_state,
        })
    }

    /// Base device allocation pointer.
    #[must_use]
    pub const fn base(&self) -> DevicePtr {
        self.base
    }

    /// Active record geometry.
    #[must_use]
    pub const fn geometry(&self) -> Gdn1dRecordGeometry {
        self.geometry
    }

    /// Device pointer to 1D FP32 `delta` records.
    #[must_use]
    pub const fn delta_ptr(&self) -> DevicePtr {
        self.slot_state.delta_ptr
    }

    /// Device pointer to 1D FP32 `k` records.
    #[must_use]
    pub const fn k_ptr(&self) -> DevicePtr {
        self.slot_state.k_ptr
    }

    /// Device pointer to 1D FP32 `decay` records.
    #[must_use]
    pub const fn decay_ptr(&self) -> DevicePtr {
        self.slot_state.decay_ptr
    }

    /// Current slot guard state.
    #[must_use]
    pub const fn slot_state(&self) -> &Gdn1dReplaySlotState {
        &self.slot_state
    }

    /// Mutable slot guard state.
    pub fn slot_state_mut(&mut self) -> &mut Gdn1dReplaySlotState {
        &mut self.slot_state
    }
}

/// Cross-layer single-launch commit coordinator reading device `d_num_accepted`
/// (`0` host sync) across all `[num_ssm_layers, batch_size]` GDN slots.
#[derive(Debug)]
pub struct Gdn1dReplayCrossLayerCommit {
    num_ssm_layers: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    descs: Vec<GdnCommitSlotDesc>,
}

impl Gdn1dReplayCrossLayerCommit {
    /// Build and fail-closed-validate a cross-layer commit descriptor table.
    ///
    /// # Errors
    /// Fails closed if geometry is invalid, any descriptor has a generation
    /// mismatch or `sharing_count > 1`, or two batch slots in the same layer
    /// alias the same `h_state_ptr`.
    pub fn try_new(
        num_ssm_layers: u32,
        batch_size: u32,
        num_k_heads: u32,
        num_v_heads: u32,
        descs: Vec<GdnCommitSlotDesc>,
    ) -> Result<Self> {
        ensure!(
            num_k_heads > 0 && num_v_heads > 0 && num_v_heads.is_multiple_of(num_k_heads),
            "invalid GDN head geometry for cross-layer commit: nk={num_k_heads}, nv={num_v_heads}"
        );
        ops::validate_commit_slot_descs(&descs, num_ssm_layers, batch_size)?;
        Ok(Self {
            num_ssm_layers,
            batch_size,
            num_k_heads,
            num_v_heads,
            descs,
        })
    }

    /// Exact byte size required in device descriptor scratch for H2D upload.
    #[must_use]
    pub fn descriptor_bytes(&self) -> usize {
        self.descs.len() * GdnCommitSlotDesc::ABI_BYTES
    }

    /// Serialize all descriptors into a contiguous little-endian byte buffer.
    #[must_use]
    pub fn serialize_descs(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.descriptor_bytes());
        for desc in &self.descs {
            bytes.extend_from_slice(&desc.to_le_bytes());
        }
        bytes
    }

    /// Upload descriptors asynchronously and dispatch ONE single cross-layer
    /// `gdn_commit_replay_1d_sm120` kernel reading device `d_num_accepted`
    /// (`0` host sync!).
    ///
    /// # Errors
    /// Returns an error if the kernel handle is missing, H2D copy fails, or
    /// any launch invariant fails.
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        kernels: &GdnCorrectionReplayKernels,
        descs_dev: DevicePtr,
        d_num_accepted: DevicePtr,
        status_out: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        let launch_args = GdnCommitReplay1dLaunchArgs {
            descs_dev,
            host_descs: Some(&self.descs),
            d_num_accepted,
            status_out,
            num_ssm_layers: self.num_ssm_layers,
            batch_size: self.batch_size,
            num_k_heads: self.num_k_heads,
            num_v_heads: self.num_v_heads,
        };
        // Preflight ALL launch arguments BEFORE enqueuing H2D upload so hostile
        // inputs perform zero device writes and zero kernel launches.
        ops::validate_commit_replay_1d_launch_args(kernels.commit_replay_1d, &launch_args)?;
        let serialized = self.serialize_descs();
        gpu.copy_h2d_async(&serialized, descs_dev, stream)
            .context("upload GdnCommitSlotDesc table for cross-layer 1D commit")?;
        ops::launch_gdn_commit_replay_1d_sm120(gpu, kernels.commit_replay_1d, &launch_args, stream)
    }
}

/// Parameters passed from `trait_decode_batched.rs` into the opt-in 1D verify seam.
#[derive(Debug, Clone, Copy)]
pub(super) struct ExactSequenceGdn1dVerifyArgs {
    pub h_state: DevicePtr,
    pub h_is_f16: bool,
    pub conv_f32: DevicePtr,
    pub gdn_f32: DevicePtr,
    pub gates_buf: DevicePtr,
    /// Verified available byte capacity starting at `gdn_f32` inside its
    /// backing allocation so tail-row zeroing `[live_tokens, max_tokens)` can
    /// never write past the end of `ssm_conv_out_f32`.
    pub output_capacity_bytes: usize,
    pub live_tokens: u32,
    pub max_tokens: u32,
    pub nk: u32,
    pub nv: u32,
    pub kd: u32,
    pub vd: u32,
    pub key_dim: usize,
    pub qkvz_size: u32,
    pub graph_capture: bool,
    pub has_tree_parent_ids: bool,
}

/// Compute the exact required byte span written from `output` (`gdn_f32`) for
/// `max_tokens` rows at FP32 stride `out_stride` and `nv * 128` active columns.
///
/// # Errors
/// Returns an error if `max_tokens == 0`, `out_stride < nv * 128`, or byte
/// calculation overflows `usize`.
pub fn required_verify_output_bytes(
    max_tokens: u32,
    out_stride: u32,
    num_v_heads: u32,
) -> Result<usize> {
    ensure!(
        max_tokens > 0,
        "required_verify_output_bytes needs max_tokens > 0"
    );
    let value_dim = (num_v_heads as usize)
        .checked_mul(GDN_1D_HEAD_DIM as usize)
        .context("verify output value_dim overflow")?;
    ensure!(
        (out_stride as usize) >= value_dim,
        "verify output stride ({out_stride}) < value_dim ({value_dim})"
    );
    let last_row_elem_offset = ((max_tokens - 1) as usize)
        .checked_mul(out_stride as usize)
        .context("verify output last-row offset overflow")?;
    let total_elems = last_row_elem_offset
        .checked_add(value_dim)
        .context("verify output total elements overflow")?;
    total_elems
        .checked_mul(4)
        .context("verify output byte span overflow")
}

/// Compute the maximum `max_tokens` (`>= live_tokens` and `<= desired_max_tokens`)
/// whose FP32 output + tail-zero span fits strictly inside `output_capacity_bytes`.
///
/// # Errors
/// Fails closed if `output_capacity_bytes` cannot even fit `live_tokens` rows.
pub fn bounded_verify_max_tokens(
    live_tokens: u32,
    desired_max_tokens: u32,
    out_stride: u32,
    num_v_heads: u32,
    output_capacity_bytes: usize,
) -> Result<u32> {
    ensure!(
        live_tokens > 0,
        "bounded_verify_max_tokens requires live_tokens > 0"
    );
    let live_req = required_verify_output_bytes(live_tokens, out_stride, num_v_heads)?;
    ensure!(
        live_req <= output_capacity_bytes,
        "ATLAS_GDN_1D_REPLAY=1 output buffer capacity ({output_capacity_bytes} B) cannot hold live_tokens={live_tokens} (needs {live_req} B)"
    );
    let value_dim_bytes = (num_v_heads as usize)
        .checked_mul(GDN_1D_HEAD_DIM as usize)
        .and_then(|v| v.checked_mul(4))
        .context("verify output value_dim bytes overflow")?;
    let row_stride_bytes = (out_stride as usize)
        .checked_mul(4)
        .context("verify output row stride bytes overflow")?;
    let extra_rows = (output_capacity_bytes - value_dim_bytes) / row_stride_bytes;
    let max_fit = u32::try_from(extra_rows.saturating_add(1)).unwrap_or(u32::MAX);
    let target = desired_max_tokens.max(live_tokens);
    Ok(target.min(max_fit).max(live_tokens))
}

impl Qwen3SsmLayer {
    /// Pre-allocate this layer's 1D correction-record arena outside CUDA graph
    /// capture so `ATLAS_GDN_1D_REPLAY=1` can run safely under graph capture.
    ///
    /// # Errors
    /// Returns an error if the arena lock is poisoned, geometry is invalid, or
    /// GPU allocation fails.
    pub fn preallocate_gdn_1d_replay_arena(
        &self,
        gpu: &dyn GpuBackend,
        batch_size: u32,
        max_tokens: u32,
        num_k_heads: u32,
        num_v_heads: u32,
    ) -> Result<()> {
        let geom = Gdn1dRecordGeometry::try_new(batch_size, max_tokens, num_k_heads, num_v_heads)?;
        let mut guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        let needs_alloc = match guard.as_ref() {
            Some(existing) => existing.geometry() != geom,
            None => true,
        };
        if needs_alloc {
            let default_stream = gpu.default_stream();
            let new_arena = Gdn1dReplayArena::allocate_carrying(
                gpu,
                geom,
                Gdn1dAllocContext::OutsideGraphCapture,
                default_stream,
                guard.as_ref().map(Gdn1dReplayArena::slot_state),
            )?;
            if guard.is_some()
                && let Err(err) = gpu.synchronize(default_stream)
            {
                let _ = gpu.free(new_arena.base());
                return Err(err)
                    .context("synchronize before replacing GDN 1D correction-record arena");
            }
            if let Some(old) = guard.replace(new_arena) {
                let _ = gpu.free(old.base());
            }
        }
        Ok(())
    }

    /// Snapshot this layer's current [`GdnCommitSlotDesc`] for cross-layer commit.
    ///
    /// # Errors
    /// Returns an error if no 1D correction record is currently staged.
    pub fn current_gdn_1d_commit_desc(&self) -> Result<GdnCommitSlotDesc> {
        self.try_current_gdn_1d_commit_desc()?
            .context("GDN 1D replay arena has no recorded correction window on this layer")
    }

    /// Snapshot this layer's current [`GdnCommitSlotDesc`] if a verify window is
    /// staged, returning `Ok(None)` when no 1D record has been recorded yet.
    ///
    /// # Errors
    /// Returns an error if the arena lock is poisoned.
    pub fn try_current_gdn_1d_commit_desc(&self) -> Result<Option<GdnCommitSlotDesc>> {
        let guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        let Some(arena) = guard.as_ref() else {
            return Ok(None);
        };
        if !arena.slot_state().recorded {
            return Ok(None);
        }
        Ok(Some(arena.slot_state().to_commit_desc()?))
    }

    /// Advance this layer's slot generation on rollback/reset/eviction, invalidating
    /// any uncommitted 1D record fail-closed.
    ///
    /// # Errors
    /// Returns an error if the arena lock is poisoned.
    pub fn invalidate_gdn_1d_replay_generation(&self) -> Result<()> {
        let mut guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        if let Some(arena) = guard.as_mut() {
            arena.slot_state_mut().advance_generation();
        }
        Ok(())
    }

    /// Release this layer's 1D correction-record device arena cleanly.
    ///
    /// # Errors
    /// Returns an error if the arena lock is poisoned or freeing fails.
    pub fn release_gdn_1d_replay_arena(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        if let Some(arena) = guard.take() {
            gpu.free(arena.base())
                .context("free GDN 1D correction-record arena")?;
        }
        Ok(())
    }

    /// Access this layer's optional SM120/SM121 1D GDN replay kernel handles.
    #[must_use]
    pub const fn gdn_1d_replay_kernels(&self) -> &GdnCorrectionReplayKernels {
        &self.gdn_1d_replay_k
    }

    /// Explicit tail-row zeroing helper for this layer's 1D record buffers and
    /// optional FP32 verify output buffer.
    ///
    /// # Errors
    /// Fails closed if `gdn_zero_tail_rows_sm120` is unavailable or arguments
    /// fail validation.
    pub fn zero_gdn_1d_tail_rows(
        &self,
        gpu: &dyn GpuBackend,
        output: Option<DevicePtr>,
        out_stride: u32,
        live_tokens: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.gdn_1d_replay_k.has_zero_tail_rows(),
            "ATLAS_GDN_1D_REPLAY=1 requires gdn_zero_tail_rows_sm120 kernel"
        );
        let guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        let arena = guard
            .as_ref()
            .context("GDN 1D replay arena has not been initialized on this layer")?;
        ops::launch_gdn_zero_tail_rows_sm120(
            gpu,
            self.gdn_1d_replay_k.zero_tail_rows,
            &GdnZeroTailRowsLaunchArgs {
                output,
                delta_out: Some(arena.delta_ptr()),
                k_out: Some(arena.k_ptr()),
                decay_out: Some(arena.decay_ptr()),
                geometry: arena.geometry(),
                live_tokens,
                out_stride,
            },
            stream,
        )
    }

    /// Preflight all fail-closed `ATLAS_GDN_1D_REPLAY=1` preconditions BEFORE
    /// running `conv1d_update_l2norm_f32_sequence` so incompatible verify steps
    /// fail with zero GPU state mutations.
    pub(super) fn preflight_gdn_1d_replay_exact_sequence(
        &self,
        gpu: &dyn GpuBackend,
        args: &ExactSequenceGdn1dVerifyArgs,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.gdn_1d_replay_k.is_complete(),
            "ATLAS_GDN_1D_REPLAY=1 requested but gdn_correction_replay_sm120 kernels are not loaded"
        );
        ensure!(
            args.kd == GDN_1D_HEAD_DIM && args.vd == GDN_1D_HEAD_DIM,
            "ATLAS_GDN_1D_REPLAY=1 requires kd==128 and vd==128; got kd={}, vd={}",
            args.kd,
            args.vd
        );
        ensure!(
            !args.h_is_f16,
            "ATLAS_GDN_1D_REPLAY=1 requires canonical FP32 h_state (h_is_f16 must be false)"
        );
        ensure!(
            !args.has_tree_parent_ids,
            "ATLAS_GDN_1D_REPLAY=1 requires flat-chain verify (tree parent_ids unsupported)"
        );
        if args.live_tokens == 0 {
            bail!("ATLAS_GDN_1D_REPLAY=1 requires live_tokens > 0");
        }
        ensure!(
            args.max_tokens >= args.live_tokens,
            "ATLAS_GDN_1D_REPLAY=1 requires max_tokens ({}) >= live_tokens ({})",
            args.max_tokens,
            args.live_tokens
        );
        ensure!(
            !args.h_state.is_null() && args.h_state.0.is_multiple_of(16),
            "ATLAS_GDN_1D_REPLAY=1 h_state ({:#x}) must be non-null and 16-byte aligned",
            args.h_state.0
        );
        for (ptr, label) in [
            (args.conv_f32, "conv_f32"),
            (args.gdn_f32, "gdn_f32"),
            (args.gates_buf, "gates_buf"),
        ] {
            ensure!(
                !ptr.is_null() && ptr.0.is_multiple_of(4),
                "ATLAS_GDN_1D_REPLAY=1 {label} ({:#x}) must be non-null and 4-byte aligned",
                ptr.0
            );
        }
        let required_out_bytes =
            required_verify_output_bytes(args.max_tokens, args.qkvz_size, args.nv)?;
        ensure!(
            required_out_bytes <= args.output_capacity_bytes,
            "ATLAS_GDN_1D_REPLAY=1 output span ({required_out_bytes} B for max_tokens={}) exceeds backing buffer capacity ({} B)",
            args.max_tokens,
            args.output_capacity_bytes
        );

        let geom = Gdn1dRecordGeometry::try_new(1, args.max_tokens, args.nk, args.nv)?;
        let capturing = args.graph_capture
            || spark_runtime::gpu_capture_region::active_capture_region().is_some()
            || gpu.stream_is_capturing(stream)
            || gpu.stream_is_capturing(gpu.default_stream());
        let guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        if let Some(existing) = guard.as_ref() {
            ensure!(
                existing.slot_state().sharing_count <= 1,
                "ATLAS_GDN_1D_REPLAY=1 refuses verify on shared slot (sharing_count={})",
                existing.slot_state().sharing_count
            );
        }
        let needs_alloc = match guard.as_ref() {
            Some(existing) => existing.geometry() != geom,
            None => true,
        };
        if needs_alloc {
            ensure!(
                !capturing,
                "ATLAS_GDN_1D_REPLAY=1 cannot lazily allocate/reallocate 1D correction arena during CUDA graph capture; call preallocate_gdn_1d_replay_arena first"
            );
        }
        Ok(())
    }

    /// Execute the opt-in 1D correction-record verify kernel inside
    /// `ExactSequence` (`trait_decode_batched.rs`).
    ///
    /// - Returns `Ok(Disabled)` immediately when `ATLAS_GDN_1D_REPLAY` is OFF.
    /// - Fails closed (`Err`) when `ATLAS_GDN_1D_REPLAY=1` is requested on an
    ///   incompatible state, geometry, tree topology, or missing PTX bundle.
    pub(super) fn try_gdn_1d_replay_verify_exact_sequence(
        &self,
        gpu: &dyn GpuBackend,
        args: &ExactSequenceGdn1dVerifyArgs,
        stream: u64,
    ) -> Result<Gdn1dReplayVerifyOutcome> {
        if ops::gdn_1d_replay_policy()? == Gdn1dReplayPolicy::Disabled {
            return Ok(Gdn1dReplayVerifyOutcome::Disabled);
        }
        self.preflight_gdn_1d_replay_exact_sequence(gpu, args, stream)?;

        let geom = Gdn1dRecordGeometry::try_new(1, args.max_tokens, args.nk, args.nv)?;

        let mut guard = self
            .gdn_1d_replay_arena
            .lock()
            .map_err(|_| anyhow::anyhow!("gdn_1d_replay_arena lock poisoned"))?;
        let needs_alloc = match guard.as_ref() {
            Some(existing) => existing.geometry() != geom,
            None => true,
        };
        if needs_alloc {
            let alloc_ctx = if args.graph_capture {
                Gdn1dAllocContext::DuringGraphCapture
            } else {
                Gdn1dAllocContext::OutsideGraphCapture
            };
            let new_arena = Gdn1dReplayArena::allocate_carrying(
                gpu,
                geom,
                alloc_ctx,
                stream,
                guard.as_ref().map(Gdn1dReplayArena::slot_state),
            )?;
            if guard.is_some()
                && let Err(err) = gpu.synchronize(stream)
            {
                let _ = gpu.free(new_arena.base());
                return Err(err)
                    .context("synchronize stream before replacing GDN 1D correction-record arena");
            }
            if let Some(old) = guard.replace(new_arena) {
                let _ = gpu.free(old.base());
            }
        }
        let arena = guard
            .as_mut()
            .context("GDN 1D replay arena unavailable after allocation")?;
        // Re-check sharing_count under the held lock BEFORE launching on GPU.
        ensure!(
            arena.slot_state().sharing_count <= 1,
            "ATLAS_GDN_1D_REPLAY=1 refuses verify on shared slot (sharing_count={})",
            arena.slot_state().sharing_count
        );

        let fp32 = 4usize;
        ops::launch_gdn_verify_record_1d_sm120(
            gpu,
            self.gdn_1d_replay_k.verify_record_1d,
            &GdnVerifyRecord1dLaunchArgs {
                h_state: args.h_state,
                query: args.conv_f32,
                key: args.conv_f32.offset(args.key_dim * fp32),
                value: args.conv_f32.offset(args.key_dim * 2 * fp32),
                gate: args.gates_buf,
                beta: args.gates_buf.offset((args.nv as usize) * fp32),
                output: args.gdn_f32,
                delta_out: arena.delta_ptr(),
                k_out: arena.k_ptr(),
                decay_out: arena.decay_ptr(),
                geometry: geom,
                live_tokens: args.live_tokens,
                qk_stride: args.qkvz_size,
                v_stride: args.qkvz_size,
                gb_stride: args.nv * 2,
                out_stride: args.qkvz_size,
                numeric_config: GdnVerifyNumericConfig::exact_fp32(),
            },
            stream,
        )?;
        arena
            .slot_state_mut()
            .record_verify(args.h_state, args.live_tokens, args.max_tokens)?;
        Ok(Gdn1dReplayVerifyOutcome::Executed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::{
        CpuGdnCommitSlot, CpuGdnVerify1dInputs, GDN_1D_STATUS_ACCEPTED_OOB,
        GDN_1D_STATUS_GEN_MISMATCH, GDN_1D_STATUS_SHARED_SLOT, GdnBetaInputMode,
        GdnBetaRoundingMode, GdnOutputRoundingMode, cpu_gdn_2d_snapshot_reference,
        cpu_gdn_commit_replay_1d, cpu_gdn_verify_record_1d, cpu_gdn_zero_tail_rows,
        gdn_1d_round_f32_to_bf16_rn,
    };
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn deterministic_f32(seed: usize, scale: f32) -> f32 {
        let mut x = (seed as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        let unit = ((x & 0x00ff_ffff) as f32) / (0x0100_0000 as f32);
        (unit * 2.0 - 1.0) * scale
    }

    fn build_synthetic_batch(
        batch_size: u32,
        max_tokens: u32,
        nk: u32,
        nv: u32,
        raw_beta_logits: bool,
    ) -> (
        Gdn1dRecordGeometry,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let geom = Gdn1dRecordGeometry::try_new(batch_size, max_tokens, nk, nv).unwrap();
        let b = batch_size as usize;
        let rows = b * (max_tokens as usize);
        let d = GDN_1D_HEAD_DIM as usize;
        let h_state: Vec<f32> = (0..b * (nv as usize) * d * d)
            .map(|i| deterministic_f32(i + 1, 0.25))
            .collect();
        let query: Vec<f32> = (0..rows * (nk as usize) * d)
            .map(|i| deterministic_f32(i + 10_000, 0.35))
            .collect();
        let key: Vec<f32> = (0..rows * (nk as usize) * d)
            .map(|i| deterministic_f32(i + 20_000, 0.35))
            .collect();
        let value: Vec<f32> = (0..rows * (nv as usize) * d)
            .map(|i| deterministic_f32(i + 30_000, 0.5))
            .collect();
        let gate: Vec<f32> = (0..rows * (nv as usize))
            .map(|i| 0.5 + 0.45 * deterministic_f32(i + 40_000, 1.0))
            .collect();
        let beta: Vec<f32> = (0..rows * (nv as usize))
            .map(|i| {
                let raw = deterministic_f32(i + 50_000, 2.5);
                if raw_beta_logits {
                    raw
                } else {
                    1.0 / (1.0 + (-raw).exp())
                }
            })
            .collect();
        (geom, h_state, query, key, value, gate, beta)
    }

    #[test]
    fn zero_ulp_bit_exact_parity_across_windows_batches_and_all_accepted_counts() {
        // Test W in {1, 8, 16} and B in {1, 4} (including GQA 1:3 ratio matching Qwen3.8).
        for (batch_size, max_tokens, live_tokens, nk, nv) in [
            (1u32, 1u32, 1u32, 1u32, 3u32),
            (1, 8, 8, 1, 3),
            (1, 16, 12, 1, 3),
            (4, 8, 6, 1, 3),
        ] {
            let (geom, h_init, q, k, v, gate, beta) =
                build_synthetic_batch(batch_size, max_tokens, nk, nv, false);
            let inputs = CpuGdnVerify1dInputs {
                h_state: &h_init,
                query: &q,
                key: &k,
                value: &v,
                gate: &gate,
                beta: &beta,
                geometry: geom,
                live_tokens,
                numeric_config: GdnVerifyNumericConfig::exact_fp32(),
            };

            let verify_1d = cpu_gdn_verify_record_1d(&inputs).unwrap();
            let ref_2d = cpu_gdn_2d_snapshot_reference(&inputs).unwrap();

            // 1. Output vectors are 0-ULP bit-identical on every active and inactive row.
            assert_eq!(verify_1d.output.len(), ref_2d.output.len());
            for (idx, (&got, &want)) in verify_1d
                .output
                .iter()
                .zip(ref_2d.output.iter())
                .enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "output mismatch at idx={idx} (B={batch_size}, W={max_tokens}, live={live_tokens})"
                );
            }

            // 2. Inactive tail rows [live_tokens, max_tokens) are strictly +0.0f across all buffers.
            let d = GDN_1D_HEAD_DIM as usize;
            for seq in 0..batch_size as usize {
                for t in live_tokens as usize..max_tokens as usize {
                    let row = seq * (max_tokens as usize) + t;
                    for &x in
                        &verify_1d.output[row * (nv as usize) * d..(row + 1) * (nv as usize) * d]
                    {
                        assert_eq!(x.to_bits(), 0);
                    }
                    for &x in
                        &verify_1d.delta[row * (nv as usize) * d..(row + 1) * (nv as usize) * d]
                    {
                        assert_eq!(x.to_bits(), 0);
                    }
                    for &x in &verify_1d.k[row * (nk as usize) * d..(row + 1) * (nk as usize) * d] {
                        assert_eq!(x.to_bits(), 0);
                    }
                    for &x in &verify_1d.decay[row * (nv as usize)..(row + 1) * (nv as usize)] {
                        assert_eq!(x.to_bits(), 0);
                    }
                }
            }

            // 3. For EVERY accepted count in 0..=live_tokens, 1D commit replay produces
            //    0-ULP bit-for-bit identical H^(accepted) to the 2D snapshot reference!
            let seq_h_len = (nv as usize) * d * d;
            let seq_delta_len = (max_tokens as usize) * (nv as usize) * d;
            let seq_k_len = (max_tokens as usize) * (nk as usize) * d;
            let seq_decay_len = (max_tokens as usize) * (nv as usize);

            // Test every accepted count (including mixed per-seq accepted counts in batch).
            for base_acc in 0..=live_tokens {
                let mut h_working = h_init.clone();
                let d_num_accepted: Vec<u32> = (0..batch_size)
                    .map(|seq| (base_acc + seq) % (live_tokens + 1))
                    .collect();

                let mut h_slices: Vec<&mut [f32]> = h_working.chunks_mut(seq_h_len).collect();
                let mut slots: Vec<CpuGdnCommitSlot<'_>> = h_slices
                    .iter_mut()
                    .enumerate()
                    .map(|(seq, h_slice)| CpuGdnCommitSlot {
                        h_state: h_slice,
                        delta: &verify_1d.delta[seq * seq_delta_len..(seq + 1) * seq_delta_len],
                        k: &verify_1d.k[seq * seq_k_len..(seq + 1) * seq_k_len],
                        decay: &verify_1d.decay[seq * seq_decay_len..(seq + 1) * seq_decay_len],
                        generation: 7,
                        expected_generation: 7,
                        sharing_count: 1,
                        max_tokens,
                        flags: 0,
                    })
                    .collect();

                let status =
                    cpu_gdn_commit_replay_1d(&mut slots, &d_num_accepted, 1, batch_size, nk, nv)
                        .unwrap();
                status.ensure_ok().unwrap();

                for (seq, &acc) in d_num_accepted.iter().enumerate() {
                    let got_h = &h_working[seq * seq_h_len..(seq + 1) * seq_h_len];
                    let want_h = &ref_2d.h_snapshots[seq][acc as usize];
                    assert_eq!(got_h.len(), want_h.len());
                    for (idx, (&g, &w)) in got_h.iter().zip(want_h.iter()).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "0-ULP replay mismatch at seq={seq}, acc={acc}, elem={idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn explicit_bf16_ptx_rounding_boundary_matches_prefill_vs_verify_and_differs_from_unrounded() {
        let (geom, h_init, q, k, v, gate, raw_beta_logits) =
            build_synthetic_batch(1, 4, 1, 2, true);
        let cfg_bf16 =
            GdnVerifyNumericConfig::prefill_bf16_parity(GdnBetaInputMode::RawLogitSigmoid);
        assert_eq!(
            cfg_bf16.to_kernel_flags(),
            ops::GDN_1D_FLAG_BETA_FROM_RAW_LOGIT
                | ops::GDN_1D_FLAG_ROUND_BETA_BF16
                | ops::GDN_1D_FLAG_ROUND_OUT_BF16
        );

        let verify_bf16 = cpu_gdn_verify_record_1d(&CpuGdnVerify1dInputs {
            h_state: &h_init,
            query: &q,
            key: &k,
            value: &v,
            gate: &gate,
            beta: &raw_beta_logits,
            geometry: geom,
            live_tokens: 4,
            numeric_config: cfg_bf16,
        })
        .unwrap();

        // Every emitted verify output element MUST lie on the exact BF16 grid
        // (lower 16 mantissa bits == 0) before downstream FP32 RMSNorm.
        for &o in &verify_bf16.output {
            assert_eq!(
                o.to_bits() & 0x0000_ffff,
                0,
                "BF16-rounded output {o} must have zero low 16 bits"
            );
            assert_eq!(o.to_bits(), gdn_1d_round_f32_to_bf16_rn(o).to_bits());
        }

        // Prefill equivalent passing pre-rounded BF16 sigmoid(raw_b) as precomputed beta
        // must produce 0-ULP identical outputs AND 0-ULP identical final H state!
        let prefill_beta: Vec<f32> = raw_beta_logits
            .iter()
            .map(|&b| gdn_1d_round_f32_to_bf16_rn(1.0_f32 / (1.0_f32 + (-b).exp())))
            .collect();
        let prefill_ref = cpu_gdn_2d_snapshot_reference(&CpuGdnVerify1dInputs {
            h_state: &h_init,
            query: &q,
            key: &k,
            value: &v,
            gate: &gate,
            beta: &prefill_beta,
            geometry: geom,
            live_tokens: 4,
            numeric_config: GdnVerifyNumericConfig {
                beta_input: GdnBetaInputMode::PrecomputedBeta,
                beta_rounding: GdnBetaRoundingMode::ExactFp32,
                output_rounding: GdnOutputRoundingMode::RoundToBf16Ptx,
            },
        })
        .unwrap();

        for (&got, &want) in verify_bf16.output.iter().zip(prefill_ref.output.iter()) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        let mut h_replayed = h_init.clone();
        let mut slots = [CpuGdnCommitSlot {
            h_state: &mut h_replayed,
            delta: &verify_bf16.delta,
            k: &verify_bf16.k,
            decay: &verify_bf16.decay,
            generation: 3,
            expected_generation: 3,
            sharing_count: 1,
            max_tokens: 4,
            flags: 0,
        }];
        cpu_gdn_commit_replay_1d(&mut slots, &[4], 1, 1, 1, 2)
            .unwrap()
            .ensure_ok()
            .unwrap();
        for (&g, &w) in h_replayed.iter().zip(prefill_ref.h_snapshots[0][4].iter()) {
            assert_eq!(g.to_bits(), w.to_bits());
        }

        // Confirm unrounded FP32 differs on at least one output (proving the boundary is live).
        let unrounded = cpu_gdn_verify_record_1d(&CpuGdnVerify1dInputs {
            h_state: &h_init,
            query: &q,
            key: &k,
            value: &v,
            gate: &gate,
            beta: &raw_beta_logits,
            geometry: geom,
            live_tokens: 4,
            numeric_config: GdnVerifyNumericConfig {
                beta_input: GdnBetaInputMode::RawLogitSigmoid,
                beta_rounding: GdnBetaRoundingMode::ExactFp32,
                output_rounding: GdnOutputRoundingMode::ExactFp32,
            },
        })
        .unwrap();
        assert_ne!(verify_bf16.output, unrounded.output);
    }

    #[test]
    fn fail_closed_commit_guards_protect_h_state_on_gen_mismatch_sharing_and_oob() {
        let (geom, h_init, q, k, v, gate, beta) = build_synthetic_batch(1, 4, 1, 2, false);
        let verify_1d = cpu_gdn_verify_record_1d(&CpuGdnVerify1dInputs {
            h_state: &h_init,
            query: &q,
            key: &k,
            value: &v,
            gate: &gate,
            beta: &beta,
            geometry: geom,
            live_tokens: 4,
            numeric_config: GdnVerifyNumericConfig::exact_fp32(),
        })
        .unwrap();

        // 1. Generation mismatch (stale record after advance_generation) -> untouched h_state + GEN_MISMATCH
        let mut h_gen = h_init.clone();
        let mut slots = [CpuGdnCommitSlot {
            h_state: &mut h_gen,
            delta: &verify_1d.delta,
            k: &verify_1d.k,
            decay: &verify_1d.decay,
            generation: 10,
            expected_generation: 9,
            sharing_count: 1,
            max_tokens: 4,
            flags: 0,
        }];
        let st_gen = cpu_gdn_commit_replay_1d(&mut slots, &[3], 1, 1, 1, 2).unwrap();
        assert!(st_gen.has_flag(GDN_1D_STATUS_GEN_MISMATCH));
        assert!(st_gen.ensure_ok().is_err());
        assert_eq!(h_gen, h_init);

        // 2. Shared slot (sharing_count == 2 > 1) -> untouched h_state + SHARED_SLOT
        let mut h_shared = h_init.clone();
        let mut slots = [CpuGdnCommitSlot {
            h_state: &mut h_shared,
            delta: &verify_1d.delta,
            k: &verify_1d.k,
            decay: &verify_1d.decay,
            generation: 10,
            expected_generation: 10,
            sharing_count: 2,
            max_tokens: 4,
            flags: 0,
        }];
        let st_shared = cpu_gdn_commit_replay_1d(&mut slots, &[3], 1, 1, 1, 2).unwrap();
        assert!(st_shared.has_flag(GDN_1D_STATUS_SHARED_SLOT));
        assert!(st_shared.ensure_ok().is_err());
        assert_eq!(h_shared, h_init);

        // 3. accepted > max_tokens -> untouched h_state + ACCEPTED_OOB
        let mut h_oob = h_init.clone();
        let mut slots = [CpuGdnCommitSlot {
            h_state: &mut h_oob,
            delta: &verify_1d.delta,
            k: &verify_1d.k,
            decay: &verify_1d.decay,
            generation: 10,
            expected_generation: 10,
            sharing_count: 1,
            max_tokens: 4,
            flags: 0,
        }];
        let st_oob = cpu_gdn_commit_replay_1d(&mut slots, &[5], 1, 1, 1, 2).unwrap();
        assert!(st_oob.has_flag(GDN_1D_STATUS_ACCEPTED_OOB));
        assert!(st_oob.ensure_ok().is_err());
        assert_eq!(h_oob, h_init);
    }

    #[test]
    fn standalone_tail_row_zeroing_preserves_live_rows_and_clears_poisoned_tail() {
        let geom = Gdn1dRecordGeometry::try_new(2, 6, 1, 2).unwrap();
        let d = GDN_1D_HEAD_DIM as usize;
        let rows = (geom.batch_size * geom.max_tokens) as usize;
        let mut output = vec![3.25_f32; rows * 2 * d];
        let mut delta = vec![-1.5_f32; rows * 2 * d];
        let mut k = vec![2.5_f32; rows * d];
        let mut decay = vec![0.9_f32; rows * 2];

        cpu_gdn_zero_tail_rows(
            Some(&mut output),
            Some(&mut delta),
            Some(&mut k),
            Some(&mut decay),
            geom,
            4,
        )
        .unwrap();

        for seq in 0..2 {
            for t in 0..6 {
                let row = seq * 6 + t;
                let expect_zero = t >= 4;
                let o_val = output[row * 2 * d];
                let d_val = delta[row * 2 * d];
                let k_val = k[row * d];
                let g_val = decay[row * 2];
                if expect_zero {
                    assert_eq!(o_val.to_bits(), 0);
                    assert_eq!(d_val.to_bits(), 0);
                    assert_eq!(k_val.to_bits(), 0);
                    assert_eq!(g_val.to_bits(), 0);
                } else {
                    assert_eq!(o_val, 3.25);
                    assert_eq!(d_val, -1.5);
                    assert_eq!(k_val, 2.5);
                    assert_eq!(g_val, 0.9);
                }
            }
        }
    }

    #[test]
    fn slot_state_lifecycle_and_cross_layer_single_launch_commit_on_mock_gpu() {
        let gpu = MockGpuBackend::new();
        let geom = Gdn1dRecordGeometry::try_new(1, 16, 16, 48).unwrap();
        let mut arena_l0 =
            Gdn1dReplayArena::allocate(&gpu, geom, Gdn1dAllocContext::OutsideGraphCapture, 1)
                .unwrap();
        let mut arena_l1 =
            Gdn1dReplayArena::allocate(&gpu, geom, Gdn1dAllocContext::OutsideGraphCapture, 1)
                .unwrap();

        let h_bytes = geom.h_state_bytes().unwrap();
        let h_l0 = gpu.alloc(h_bytes).unwrap();
        let h_l1 = gpu.alloc(h_bytes).unwrap();

        // Recording on exclusive slot succeeds and binds generation.
        arena_l0
            .slot_state_mut()
            .record_verify(h_l0, 8, 16)
            .unwrap();
        arena_l1
            .slot_state_mut()
            .record_verify(h_l1, 8, 16)
            .unwrap();

        let desc_l0 = arena_l0.slot_state().to_commit_desc().unwrap();
        let desc_l1 = arena_l1.slot_state().to_commit_desc().unwrap();

        // Single cross-layer GPU commit launch across 2 SSM layers (0 host sync!).
        let commit =
            Gdn1dReplayCrossLayerCommit::try_new(2, 1, 16, 48, vec![desc_l0, desc_l1]).unwrap();
        assert_eq!(commit.descriptor_bytes(), 128);
        let kernels = GdnCorrectionReplayKernels {
            verify_record_1d: KernelHandle(10),
            commit_replay_1d: KernelHandle(11),
            zero_tail_rows: KernelHandle(12),
        };
        let descs_dev = gpu.alloc(commit.descriptor_bytes()).unwrap();
        let d_num_accepted = gpu.alloc(4).unwrap();
        let status_out = gpu.alloc(4).unwrap();
        commit
            .launch(
                &gpu,
                &kernels,
                descs_dev,
                d_num_accepted,
                Some(status_out),
                1,
            )
            .unwrap();
        // Exactly ONE kernel launch across both layers!
        assert_eq!(gpu.launch_count(), 1);

        // Hostile commit launch with null d_num_accepted or misaligned descs_dev must
        // fail closed before H2D upload and enqueue zero launches.
        assert!(
            commit
                .launch(&gpu, &kernels, descs_dev, DevicePtr::NULL, None, 1)
                .is_err()
        );
        assert!(
            commit
                .launch(&gpu, &kernels, DevicePtr(0x1004), d_num_accepted, None, 1)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 1);

        // Cross-layer aliasing of h_state_ptr must fail closed.
        let mut dup_layer_desc = desc_l1;
        dup_layer_desc.h_state_ptr = desc_l0.h_state_ptr;
        assert!(
            Gdn1dReplayCrossLayerCommit::try_new(2, 1, 16, 48, vec![desc_l0, dup_layer_desc])
                .is_err()
        );

        // Advancing generation invalidates uncommitted record fail-closed.
        arena_l0.slot_state_mut().advance_generation();
        let stale_desc = arena_l0.slot_state().to_commit_desc().unwrap();
        assert!(Gdn1dReplayCrossLayerCommit::try_new(1, 1, 16, 48, vec![stale_desc]).is_err());

        // Sharing count > 1 fails closed on both record and commit.
        arena_l1.slot_state_mut().set_sharing_count(2);
        assert!(
            arena_l1
                .slot_state_mut()
                .record_verify(DevicePtr(0x2000_0000), 8, 16)
                .is_err()
        );
        let shared_desc = arena_l1.slot_state().to_commit_desc().unwrap();
        assert!(Gdn1dReplayCrossLayerCommit::try_new(1, 1, 16, 48, vec![shared_desc]).is_err());

        // Lazy allocation during CUDA graph capture fails closed.
        assert!(
            Gdn1dReplayArena::allocate(&gpu, geom, Gdn1dAllocContext::DuringGraphCapture, 1)
                .is_err()
        );

        // Arena resize carries forward monotonic generation (gen -> gen + 1) and sharing_count.
        let resized = Gdn1dReplayArena::allocate_carrying(
            &gpu,
            Gdn1dRecordGeometry::try_new(1, 32, 16, 48).unwrap(),
            Gdn1dAllocContext::OutsideGraphCapture,
            1,
            Some(arena_l1.slot_state()),
        )
        .unwrap();
        assert_eq!(
            resized.slot_state().generation,
            arena_l1.slot_state().generation + 1
        );
        assert_eq!(
            resized.slot_state().sharing_count,
            arena_l1.slot_state().sharing_count
        );
        assert!(!resized.slot_state().recorded);
    }

    #[test]
    fn bounded_verify_max_tokens_prevents_oob_tail_zero_writes_and_fails_closed() {
        // Qwen3.8-Flash-Next geometry: nv=48, value_dim=6144, conv_dim=10240, qkvz_size=16384.
        let out_stride = 16_384u32;
        let nv = 48u32;
        let conv_dim_bytes = 10_240usize * 4;
        // Suppose ssm_conv_out_f32 is sized for exactly 8 rows (8 * 16384 * 4 = 524,288 B).
        let buf_8_rows = 8usize * (out_stride as usize) * 4;
        let cap_from_gdn = buf_8_rows - conv_dim_bytes;

        // Desired max_tokens=16 is bounded down to 8 so tail zeroing NEVER writes past buf_8_rows!
        let bounded = bounded_verify_max_tokens(5, 16, out_stride, nv, cap_from_gdn).unwrap();
        assert_eq!(bounded, 8);
        assert!(required_verify_output_bytes(bounded, out_stride, nv).unwrap() <= cap_from_gdn);
        assert!(required_verify_output_bytes(bounded + 1, out_stride, nv).unwrap() > cap_from_gdn);

        // If buffer capacity cannot even hold live_tokens=9, fail closed!
        assert!(bounded_verify_max_tokens(9, 16, out_stride, nv, cap_from_gdn).is_err());
    }
}
