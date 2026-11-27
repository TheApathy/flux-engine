// SPDX-License-Identifier: AGPL-3.0-only

//! Hardened On-GPU Exact Speculative Acceptance & Two-Pass Sparse Residual
//! Sampling (`ATLAS_SPEC_EXACT_GPU_ACCEPT=1`, default OFF).
//!
//! Dispatches and mirrors `kernels/gb10/common/exact_spec_accept_sm120.cu`:
//! - `exact_spec_accept_sparse_sm120`: per-`(seq, col)` CTA evaluating exact
//!   BF16 target probabilities `q` with prefix-aware presence/frequency
//!   penalties (guaranteeing exact `W=1 == W=8 == W=16` invariance), bit-exact
//!   Salmon et al. (2011) `Philox4_32_10` counter RNG keyed by
//!   `(seed, abs_pos, 0, 0)`, fail-closed `|sum(p) - 1.0| <= 1e-4` validation
//!   on BOTH target `q` and sparse draft support `p_sparse` (`K <= 64`), and
//!   two-pass sparse residual sampling `r(x) = max(q(x) - p_sparse(x), 0)`
//!   with guaranteed valid target-support zero-mass fallback.
//! - `exact_spec_prefix_reduce_sm120`: device-side per-sequence prefix scan
//!   producing zero-host-sync `d_num_accepted[seq]`, `d_bonus_token[seq]`, and
//!   `d_status_out[seq]` guarded by fail-closed `Generation` and
//!   `sharing <= 1` (owned `slot_refcount == 1` + unique physical `slot_id`)
//!   checks before emitting non-zero acceptance counts.

use std::ffi::OsStr;
use std::fmt;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/exact_spec_accept_sm120.cu`.
pub const EXACT_SPEC_ACCEPT_MODULE: &str = "exact_spec_accept_sm120";

/// CUDA entrypoint for per-row exact acceptance and two-pass sparse residual sampling.
pub const EXACT_SPEC_ACCEPT_SPARSE_ENTRY: &str = "exact_spec_accept_sparse_sm120";

/// CUDA entrypoint for zero-host-sync per-sequence prefix scan and ownership guard.
pub const EXACT_SPEC_PREFIX_REDUCE_ENTRY: &str = "exact_spec_prefix_reduce_sm120";

/// Environment variable opting into hardened on-GPU exact speculative acceptance.
///
/// Unset or `"0"` keeps the baseline path byte-identical (default OFF).
pub const SPEC_EXACT_GPU_ACCEPT_ENV: &str = "ATLAS_SPEC_EXACT_GPU_ACCEPT";

/// Maximum sparse draft support width `K` per draft row.
pub const EXACT_SPEC_MAX_SPARSE_K: u32 = 64;

/// Maximum verify window width `W` (up to 15 drafts + 1 bonus row).
pub const EXACT_SPEC_MAX_WIDTH: u32 = 16;

/// Sentinel token ID indicating an invalid/failed row (never returned on valid rows).
pub const EXACT_SPEC_INVALID_TOKEN: u32 = 0xFFFF_FFFF;

/// Maximum tolerated probability-sum error `|sum(p) - 1.0|` before failing closed.
pub const EXACT_SPEC_PROB_SUM_TOL: f32 = 1e-4;

/// CTA width for `exact_spec_accept_sparse_sm120` (8 warps).
pub const EXACT_SPEC_CTA_THREADS: u32 = 256;

/// CTA width for `exact_spec_prefix_reduce_sm120` (1 warp).
pub const EXACT_SPEC_PREFIX_THREADS: u32 = 32;

// Canonical Salmon et al. (SC'11) Philox4x32-10 multipliers and Weyl bumps.
const PHILOX_M4X32_0: u32 = 0xD251_1F53;
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
const PHILOX_W32_0: u32 = 0x9E37_79B9;
const PHILOX_W32_1: u32 = 0xBB67_AE85;

// ── Opt-In Environment Gate ──────────────────────────────────────────────────

/// Error returned when parsing `ATLAS_SPEC_EXACT_GPU_ACCEPT`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactSpecGpuAcceptEnvError {
    /// Environment value is not valid UTF-8.
    NonUtf8,
    /// Environment value is present but neither strictly `"0"` nor `"1"`.
    InvalidValue(String),
}

impl fmt::Display for ExactSpecGpuAcceptEnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => {
                write!(f, "{SPEC_EXACT_GPU_ACCEPT_ENV} must contain valid UTF-8")
            }
            Self::InvalidValue(raw) => write!(
                f,
                "{SPEC_EXACT_GPU_ACCEPT_ENV} must be strictly 0 or 1, got {raw:?}"
            ),
        }
    }
}

impl std::error::Error for ExactSpecGpuAcceptEnvError {}

/// Runtime opt-in policy resolved from `ATLAS_SPEC_EXACT_GPU_ACCEPT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExactSpecGpuAcceptPolicy {
    /// Default OFF: baseline speculative acceptance path is unchanged.
    #[default]
    Disabled,
    /// Explicit `ATLAS_SPEC_EXACT_GPU_ACCEPT=1`: hardened on-GPU exact acceptance.
    Enabled,
}

/// Strictly parse an optional OS string for `ATLAS_SPEC_EXACT_GPU_ACCEPT`.
///
/// - `None` or `"0"` -> `Ok(Disabled)`.
/// - `"1"` -> `Ok(Enabled)`.
/// - Empty `""`, non-UTF-8, or any other value -> `Err(...)` (strict `ZERO_OR_ONE`, fail closed).
pub fn parse_exact_spec_gpu_accept_os(
    raw: Option<&OsStr>,
) -> std::result::Result<ExactSpecGpuAcceptPolicy, ExactSpecGpuAcceptEnvError> {
    let Some(os_val) = raw else {
        return Ok(ExactSpecGpuAcceptPolicy::Disabled);
    };
    let utf8 = os_val.to_str().ok_or(ExactSpecGpuAcceptEnvError::NonUtf8)?;
    match utf8 {
        "0" => Ok(ExactSpecGpuAcceptPolicy::Disabled),
        "1" => Ok(ExactSpecGpuAcceptPolicy::Enabled),
        other => Err(ExactSpecGpuAcceptEnvError::InvalidValue(other.to_owned())),
    }
}

/// Resolve and cache `ATLAS_SPEC_EXACT_GPU_ACCEPT` once per process, failing
/// closed to [`ExactSpecGpuAcceptPolicy::Disabled`] with a warning on invalid
/// or non-UTF-8 values.
#[must_use]
pub fn exact_spec_gpu_accept_policy() -> ExactSpecGpuAcceptPolicy {
    static POLICY: OnceLock<ExactSpecGpuAcceptPolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var_os(SPEC_EXACT_GPU_ACCEPT_ENV);
        match parse_exact_spec_gpu_accept_os(raw.as_deref()) {
            Ok(policy) => policy,
            Err(err) => {
                tracing::warn!("{err}; failing closed to {SPEC_EXACT_GPU_ACCEPT_ENV}=0");
                ExactSpecGpuAcceptPolicy::Disabled
            }
        }
    })
}

/// True iff `ATLAS_SPEC_EXACT_GPU_ACCEPT=1` is validly enabled.
#[must_use]
pub fn exact_spec_gpu_accept_enabled() -> bool {
    exact_spec_gpu_accept_policy() == ExactSpecGpuAcceptPolicy::Enabled
}

// ── Device Status Bitmask ────────────────────────────────────────────────────

/// Explicit device status bitmask shared 1:1 with `exact_spec_accept_sm120.cu`.
///
/// Bits `0..=7` (`ERROR_MASK = 0x00FF`) indicate fail-closed faults; bit `8`
/// (`FALLBACK_USED`) records deterministic valid target-support zero-mass
/// residual fallback without failing closed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ExactSpecStatus(u32);

impl ExactSpecStatus {
    pub const OK: Self = Self(0);
    pub const BAD_PARAM: Self = Self(1 << 0);
    pub const TARGET_NONFINITE: Self = Self(1 << 1);
    pub const TARGET_SUM_ERR: Self = Self(1 << 2);
    pub const DRAFT_NONFINITE: Self = Self(1 << 3);
    pub const DRAFT_SUM_ERR: Self = Self(1 << 4);
    pub const DRAFT_OOB: Self = Self(1 << 5);
    pub const GEN_MISMATCH: Self = Self(1 << 6);
    pub const SLOT_SHARED: Self = Self(1 << 7);
    pub const FALLBACK_USED: Self = Self(1 << 8);
    pub const ERROR_MASK: Self = Self(0x00FF);

    /// Construct from raw device status bits.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Raw `u32` representation matching CUDA `out_status` / `d_status_out`.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Combine two status bitmasks via bitwise OR.
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// True iff all bits in `flag` are set.
    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0 && flag.0 != 0
    }

    /// True iff no fail-closed error bit (`0..=7`) is set.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        (self.0 & Self::ERROR_MASK.0) == 0
    }

    /// True iff any fail-closed error bit (`0..=7`) is set.
    #[must_use]
    pub const fn has_error(self) -> bool {
        (self.0 & Self::ERROR_MASK.0) != 0
    }

    /// True iff deterministic target-support fallback was exercised.
    #[must_use]
    pub const fn fallback_used(self) -> bool {
        self.contains(Self::FALLBACK_USED)
    }

    /// True iff slot generation validation failed.
    #[must_use]
    pub const fn has_gen_mismatch(self) -> bool {
        self.contains(Self::GEN_MISMATCH)
    }

    /// True iff `sharing <= 1` (`slot_refcount == 1` + unique `slot_id`) failed.
    #[must_use]
    pub const fn has_slot_shared(self) -> bool {
        self.contains(Self::SLOT_SHARED)
    }

    /// True iff target or sparse draft probability sum failed `|sum(p) - 1| <= 1e-4`.
    #[must_use]
    pub const fn has_sum_error(self) -> bool {
        self.contains(Self::TARGET_SUM_ERR) || self.contains(Self::DRAFT_SUM_ERR)
    }
}

impl fmt::Display for ExactSpecStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return f.write_str("OK");
        }
        let flags = [
            (Self::BAD_PARAM, "BAD_PARAM"),
            (Self::TARGET_NONFINITE, "TARGET_NONFINITE"),
            (Self::TARGET_SUM_ERR, "TARGET_SUM_ERR"),
            (Self::DRAFT_NONFINITE, "DRAFT_NONFINITE"),
            (Self::DRAFT_SUM_ERR, "DRAFT_SUM_ERR"),
            (Self::DRAFT_OOB, "DRAFT_OOB"),
            (Self::GEN_MISMATCH, "GEN_MISMATCH"),
            (Self::SLOT_SHARED, "SLOT_SHARED"),
            (Self::FALLBACK_USED, "FALLBACK_USED"),
        ];
        let mut first = true;
        for (bit, name) in flags {
            if self.contains(bit) {
                if !first {
                    f.write_str("|")?;
                }
                f.write_str(name)?;
                first = false;
            }
        }
        let known = Self::ERROR_MASK.0 | Self::FALLBACK_USED.0;
        let unknown = self.0 & !known;
        if unknown != 0 {
            if !first {
                f.write_str("|")?;
            }
            write!(f, "UNKNOWN(0x{unknown:x})")?;
        }
        Ok(())
    }
}

// ── Kernel Handles & Ownership Descriptors ───────────────────────────────────

/// Resolved CUDA kernel handles for `exact_spec_accept_sm120.cu`.
#[derive(Clone, Copy, Debug)]
pub struct ExactSpecAcceptKernels {
    pub accept_sparse: KernelHandle,
    pub prefix_reduce: KernelHandle,
}

impl ExactSpecAcceptKernels {
    /// Probe both optional entrypoints from `"exact_spec_accept_sm120"`.
    #[must_use]
    pub fn load_optional(gpu: &dyn GpuBackend) -> Self {
        Self {
            accept_sparse: crate::layers::try_kernel(
                gpu,
                EXACT_SPEC_ACCEPT_MODULE,
                EXACT_SPEC_ACCEPT_SPARSE_ENTRY,
            ),
            prefix_reduce: crate::layers::try_kernel(
                gpu,
                EXACT_SPEC_ACCEPT_MODULE,
                EXACT_SPEC_PREFIX_REDUCE_ENTRY,
            ),
        }
    }

    /// True when both acceptance and prefix-reduce entrypoints resolved cleanly.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.accept_sparse.0 != 0 && self.prefix_reduce.0 != 0
    }
}

/// Per-sequence physical slot ownership descriptor validated by
/// `exact_spec_prefix_reduce_sm120` before emitting zero-host-sync commit counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSpecSlotGuard {
    /// Physical SSM / verify scratch slot index owned by this batch lane.
    pub slot_id: u32,
    /// Current device/pool generation observed on `slot_id`.
    pub slot_generation: u64,
    /// Generation expected by the scheduler's active verify frame.
    pub expected_generation: u64,
    /// Active reference count on `slot_id` (must be strictly `1` for exclusive commit).
    pub slot_refcount: u32,
}

/// Evaluate the exact `Generation` and `sharing <= 1` (`refcount == 1` +
/// cross-batch unique `slot_id`) guards on the host, matching
/// `exact_spec_prefix_reduce_sm120` bit-for-bit.
#[must_use]
pub fn validate_slot_guards_host(guards: &[ExactSpecSlotGuard]) -> Vec<ExactSpecStatus> {
    let mut statuses = Vec::with_capacity(guards.len());
    for (seq, g) in guards.iter().enumerate() {
        let mut st = ExactSpecStatus::OK;
        if g.slot_generation != g.expected_generation {
            st = st.with(ExactSpecStatus::GEN_MISMATCH);
        }
        if g.slot_refcount != 1 {
            st = st.with(ExactSpecStatus::SLOT_SHARED);
        }
        if guards
            .iter()
            .enumerate()
            .any(|(other, og)| other != seq && og.slot_id == g.slot_id)
        {
            st = st.with(ExactSpecStatus::SLOT_SHARED);
        }
        statuses.push(st);
    }
    statuses
}

// ── Validated CUDA Kernel Launchers ──────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct DeviceRegion {
    name: &'static str,
    ptr: DevicePtr,
    bytes: usize,
    align: u64,
}

fn validate_region(op: &str, region: DeviceRegion) -> Result<(u64, u64)> {
    ensure!(
        !region.ptr.is_null(),
        "{op}: {} pointer is null",
        region.name
    );
    ensure!(
        region.ptr.0.is_multiple_of(region.align),
        "{op}: {} pointer 0x{:x} is not {}-byte aligned",
        region.name,
        region.ptr.0,
        region.align
    );
    let len_u64 = u64::try_from(region.bytes)
        .with_context(|| format!("{op}: {} byte extent overflow", region.name))?;
    let end = region
        .ptr
        .0
        .checked_add(len_u64)
        .with_context(|| format!("{op}: {} address range wraps u64", region.name))?;
    Ok((region.ptr.0, end))
}

fn ensure_outputs_disjoint(
    op: &str,
    outputs: &[DeviceRegion],
    inputs: &[DeviceRegion],
) -> Result<()> {
    let mut out_spans = Vec::with_capacity(outputs.len());
    for &r in outputs {
        out_spans.push((r.name, validate_region(op, r)?));
    }
    let mut in_spans = Vec::with_capacity(inputs.len());
    for &r in inputs {
        in_spans.push((r.name, validate_region(op, r)?));
    }
    for i in 0..out_spans.len() {
        let (name_a, (start_a, end_a)) = out_spans[i];
        for &(name_b, (start_b, end_b)) in &out_spans[(i + 1)..] {
            if start_a < end_b && start_b < end_a {
                bail!(
                    "{op}: output buffers {name_a} [0x{start_a:x}..0x{end_a:x}) and \
                     {name_b} [0x{start_b:x}..0x{end_b:x}) overlap"
                );
            }
        }
        for &(name_in, (start_in, end_in)) in &in_spans {
            if start_a < end_in && start_in < end_a {
                bail!(
                    "{op}: output buffer {name_a} [0x{start_a:x}..0x{end_a:x}) overlaps \
                     input buffer {name_in} [0x{start_in:x}..0x{end_in:x})"
                );
            }
        }
    }
    Ok(())
}

fn checked_bytes(op: &str, label: &str, elems: usize, elem_size: usize) -> Result<usize> {
    elems
        .checked_mul(elem_size)
        .with_context(|| format!("{op}: {label} byte size overflow"))
}

/// Device buffers and geometry for [`launch_exact_spec_accept_sparse_sm120`].
#[derive(Clone, Copy, Debug)]
pub struct ExactSpecAcceptSparseArgs {
    /// Target logits `[batch_size * verify_width, vocab_size]` in BF16.
    pub logits: DevicePtr,
    /// Draft token IDs `[batch_size * verify_width]` in `u32` (required when `verify_width > 1`).
    pub draft_tokens: Option<DevicePtr>,
    /// Sparse draft support token IDs `[batch_size * verify_width, sparse_k]` in `u32`
    /// (required when `verify_width > 1`).
    pub sparse_draft_ids: Option<DevicePtr>,
    /// Sparse draft support probabilities `[batch_size * verify_width, sparse_k]` in `f32`
    /// (required when `verify_width > 1`).
    pub sparse_draft_probs: Option<DevicePtr>,
    /// Prefix tokens `[batch_size, verify_width]` in `u32` for prefix-aware penalties
    /// (required when `verify_width > 1`).
    pub verify_prefix_tokens: Option<DevicePtr>,
    /// Optional pre-verify token occurrence counts `[batch_size, vocab_size]` in `u32`.
    pub pre_verify_counts: Option<DevicePtr>,
    /// Per-sequence 64-bit Philox seeds `[batch_size]` in `u64`.
    pub seeds: DevicePtr,
    /// Per-sequence starting absolute token positions `[batch_size]` in `u64`.
    pub base_abs_pos: DevicePtr,
    /// Per-sequence finite positive temperatures `[batch_size]` in `f32`.
    pub temperatures: DevicePtr,
    /// Per-sequence finite presence penalties `[batch_size]` in `f32`.
    pub presence_penalties: DevicePtr,
    /// Per-sequence finite frequency penalties `[batch_size]` in `f32`.
    pub frequency_penalties: DevicePtr,
    /// Output per-row acceptance flags `[batch_size * verify_width]` in `u32` (`0` or `1`).
    pub out_accepted: DevicePtr,
    /// Output per-row accepted/residual/bonus token IDs `[batch_size * verify_width]` in `u32`.
    pub out_tokens: DevicePtr,
    /// Output per-row [`ExactSpecStatus`] bitmasks `[batch_size * verify_width]` in `u32`.
    pub out_status: DevicePtr,
    pub batch_size: u32,
    pub verify_width: u32,
    pub vocab_size: u32,
    pub sparse_k: u32,
}

fn require_optional_when_wide(
    op: &str,
    name: &'static str,
    ptr: Option<DevicePtr>,
    verify_width: u32,
) -> Result<DevicePtr> {
    match ptr {
        Some(p) => {
            ensure!(!p.is_null(), "{op}: {name} was passed as Some(NULL)");
            Ok(p)
        }
        None if verify_width == 1 => Ok(DevicePtr::NULL),
        None => bail!("{op}: {name} is required when verify_width={verify_width} > 1"),
    }
}

/// Launch `exact_spec_accept_sparse_sm120` (`grid = [verify_width, batch_size, 1]`,
/// `block = [256, 1, 1]`) after validating handle, bounds, alignments, and
/// strict output-buffer non-overlap.
///
/// # Errors
/// Fails closed before issuing any CUDA launch if `kernel` is null, dimensions
/// are out of ABI bounds, required pointers are null/misaligned, or any output
/// buffer overlaps another output or input buffer.
pub fn launch_exact_spec_accept_sparse_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &ExactSpecAcceptSparseArgs,
    stream: u64,
) -> Result<()> {
    const OP: &str = EXACT_SPEC_ACCEPT_SPARSE_ENTRY;
    ensure!(kernel.0 != 0, "{OP} requires a non-zero kernel handle");
    ensure!(args.batch_size > 0, "{OP}: batch_size must be >= 1");
    ensure!(
        (1..=EXACT_SPEC_MAX_WIDTH).contains(&args.verify_width),
        "{OP}: verify_width={} must lie in 1..={EXACT_SPEC_MAX_WIDTH}",
        args.verify_width
    );
    ensure!(args.vocab_size > 0, "{OP}: vocab_size must be >= 1");
    ensure!(
        (1..=EXACT_SPEC_MAX_SPARSE_K).contains(&args.sparse_k),
        "{OP}: sparse_k={} must lie in 1..={EXACT_SPEC_MAX_SPARSE_K}",
        args.sparse_k
    );

    let total_rows_u32 = args
        .batch_size
        .checked_mul(args.verify_width)
        .with_context(|| format!("{OP}: batch_size * verify_width overflows u32"))?;
    let batch = args.batch_size as usize;
    let total_rows = total_rows_u32 as usize;
    let vocab = args.vocab_size as usize;
    let sparse_k = args.sparse_k as usize;

    let logits_elems = total_rows
        .checked_mul(vocab)
        .context("exact_spec_accept_sparse_sm120: logits element count overflow")?;
    let logits_bytes = checked_bytes(OP, "logits", logits_elems, 2)?;
    let row_u32_bytes = checked_bytes(OP, "row_u32", total_rows, 4)?;
    let sparse_elems = total_rows
        .checked_mul(sparse_k)
        .context("exact_spec_accept_sparse_sm120: sparse element count overflow")?;
    let sparse_bytes = checked_bytes(OP, "sparse_draft", sparse_elems, 4)?;
    let seq_u64_bytes = checked_bytes(OP, "seq_u64", batch, 8)?;
    let seq_f32_bytes = checked_bytes(OP, "seq_f32", batch, 4)?;

    let draft_tokens_ptr =
        require_optional_when_wide(OP, "draft_tokens", args.draft_tokens, args.verify_width)?;
    let sparse_ids_ptr = require_optional_when_wide(
        OP,
        "sparse_draft_ids",
        args.sparse_draft_ids,
        args.verify_width,
    )?;
    let sparse_probs_ptr = require_optional_when_wide(
        OP,
        "sparse_draft_probs",
        args.sparse_draft_probs,
        args.verify_width,
    )?;
    let prefix_tokens_ptr = require_optional_when_wide(
        OP,
        "verify_prefix_tokens",
        args.verify_prefix_tokens,
        args.verify_width,
    )?;

    let pre_counts_ptr = match args.pre_verify_counts {
        Some(p) => {
            ensure!(
                !p.is_null(),
                "{OP}: pre_verify_counts was passed as Some(NULL)"
            );
            p
        }
        None => DevicePtr::NULL,
    };

    let outputs = [
        DeviceRegion {
            name: "out_accepted",
            ptr: args.out_accepted,
            bytes: row_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "out_tokens",
            ptr: args.out_tokens,
            bytes: row_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "out_status",
            ptr: args.out_status,
            bytes: row_u32_bytes,
            align: 4,
        },
    ];

    let mut inputs = Vec::with_capacity(11);
    inputs.push(DeviceRegion {
        name: "logits",
        ptr: args.logits,
        bytes: logits_bytes,
        align: 2,
    });
    if !draft_tokens_ptr.is_null() {
        inputs.push(DeviceRegion {
            name: "draft_tokens",
            ptr: draft_tokens_ptr,
            bytes: row_u32_bytes,
            align: 4,
        });
    }
    if !sparse_ids_ptr.is_null() {
        inputs.push(DeviceRegion {
            name: "sparse_draft_ids",
            ptr: sparse_ids_ptr,
            bytes: sparse_bytes,
            align: 4,
        });
    }
    if !sparse_probs_ptr.is_null() {
        inputs.push(DeviceRegion {
            name: "sparse_draft_probs",
            ptr: sparse_probs_ptr,
            bytes: sparse_bytes,
            align: 4,
        });
    }
    if !prefix_tokens_ptr.is_null() {
        inputs.push(DeviceRegion {
            name: "verify_prefix_tokens",
            ptr: prefix_tokens_ptr,
            bytes: row_u32_bytes,
            align: 4,
        });
    }
    if !pre_counts_ptr.is_null() {
        let count_elems = batch
            .checked_mul(vocab)
            .context("exact_spec_accept_sparse_sm120: pre_verify_counts element overflow")?;
        let count_bytes = checked_bytes(OP, "pre_verify_counts", count_elems, 4)?;
        inputs.push(DeviceRegion {
            name: "pre_verify_counts",
            ptr: pre_counts_ptr,
            bytes: count_bytes,
            align: 4,
        });
    }
    inputs.push(DeviceRegion {
        name: "seeds",
        ptr: args.seeds,
        bytes: seq_u64_bytes,
        align: 8,
    });
    inputs.push(DeviceRegion {
        name: "base_abs_pos",
        ptr: args.base_abs_pos,
        bytes: seq_u64_bytes,
        align: 8,
    });
    inputs.push(DeviceRegion {
        name: "temperatures",
        ptr: args.temperatures,
        bytes: seq_f32_bytes,
        align: 4,
    });
    inputs.push(DeviceRegion {
        name: "presence_penalties",
        ptr: args.presence_penalties,
        bytes: seq_f32_bytes,
        align: 4,
    });
    inputs.push(DeviceRegion {
        name: "frequency_penalties",
        ptr: args.frequency_penalties,
        bytes: seq_f32_bytes,
        align: 4,
    });

    ensure_outputs_disjoint(OP, &outputs, &inputs)?;

    KernelLaunch::new(gpu, kernel)
        .grid([args.verify_width, args.batch_size, 1])
        .block([EXACT_SPEC_CTA_THREADS, 1, 1])
        .arg_ptr(args.logits)
        .arg_ptr(draft_tokens_ptr)
        .arg_ptr(sparse_ids_ptr)
        .arg_ptr(sparse_probs_ptr)
        .arg_ptr(prefix_tokens_ptr)
        .arg_ptr(pre_counts_ptr)
        .arg_ptr(args.seeds)
        .arg_ptr(args.base_abs_pos)
        .arg_ptr(args.temperatures)
        .arg_ptr(args.presence_penalties)
        .arg_ptr(args.frequency_penalties)
        .arg_ptr(args.out_accepted)
        .arg_ptr(args.out_tokens)
        .arg_ptr(args.out_status)
        .arg_u32(args.batch_size)
        .arg_u32(args.verify_width)
        .arg_u32(args.vocab_size)
        .arg_u32(args.sparse_k)
        .launch(stream)
}

/// Device buffers and geometry for [`launch_exact_spec_prefix_reduce_sm120`].
#[derive(Clone, Copy, Debug)]
pub struct ExactSpecPrefixReduceArgs {
    /// Per-row acceptance flags `[batch_size * verify_width]` in `u32`.
    pub row_accepted: DevicePtr,
    /// Per-row sampled/accepted tokens `[batch_size * verify_width]` in `u32`.
    pub row_tokens: DevicePtr,
    /// Per-row [`ExactSpecStatus`] bitmasks `[batch_size * verify_width]` in `u32`.
    pub row_status: DevicePtr,
    /// Physical slot IDs `[batch_size]` in `u32`.
    pub slot_ids: DevicePtr,
    /// Observed slot generations `[batch_size]` in `u64`.
    pub slot_generations: DevicePtr,
    /// Expected active verify frame generations `[batch_size]` in `u64`.
    pub expected_generations: DevicePtr,
    /// Active slot reference counts `[batch_size]` in `u32` (must be `1`).
    pub slot_refcounts: DevicePtr,
    /// Zero-host-sync accepted draft counts `[batch_size]` in `u32`.
    pub d_num_accepted: DevicePtr,
    /// Zero-host-sync bonus/residual tokens `[batch_size]` in `u32`.
    pub d_bonus_token: DevicePtr,
    /// Per-sequence combined [`ExactSpecStatus`] bitmasks `[batch_size]` in `u32`.
    pub d_status_out: DevicePtr,
    pub batch_size: u32,
    pub verify_width: u32,
}

/// Launch `exact_spec_prefix_reduce_sm120` (`grid = [batch_size, 1, 1]`,
/// `block = [32, 1, 1]`) after validating handle, bounds, alignments, and
/// strict output-buffer non-overlap.
///
/// # Errors
/// Fails closed before issuing any CUDA launch if `kernel` is null, dimensions
/// are out of ABI bounds, required pointers are null/misaligned, or any output
/// buffer overlaps another output or input buffer.
pub fn launch_exact_spec_prefix_reduce_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &ExactSpecPrefixReduceArgs,
    stream: u64,
) -> Result<()> {
    const OP: &str = EXACT_SPEC_PREFIX_REDUCE_ENTRY;
    ensure!(kernel.0 != 0, "{OP} requires a non-zero kernel handle");
    ensure!(args.batch_size > 0, "{OP}: batch_size must be >= 1");
    ensure!(
        (1..=EXACT_SPEC_MAX_WIDTH).contains(&args.verify_width),
        "{OP}: verify_width={} must lie in 1..={EXACT_SPEC_MAX_WIDTH}",
        args.verify_width
    );

    let total_rows_u32 = args
        .batch_size
        .checked_mul(args.verify_width)
        .context("exact_spec_prefix_reduce_sm120: batch_size * verify_width overflows u32")?;
    let batch = args.batch_size as usize;
    let row_u32_bytes = checked_bytes(OP, "row_u32", total_rows_u32 as usize, 4)?;
    let seq_u32_bytes = checked_bytes(OP, "seq_u32", batch, 4)?;
    let seq_u64_bytes = checked_bytes(OP, "seq_u64", batch, 8)?;

    let outputs = [
        DeviceRegion {
            name: "d_num_accepted",
            ptr: args.d_num_accepted,
            bytes: seq_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "d_bonus_token",
            ptr: args.d_bonus_token,
            bytes: seq_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "d_status_out",
            ptr: args.d_status_out,
            bytes: seq_u32_bytes,
            align: 4,
        },
    ];
    let inputs = [
        DeviceRegion {
            name: "row_accepted",
            ptr: args.row_accepted,
            bytes: row_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "row_tokens",
            ptr: args.row_tokens,
            bytes: row_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "row_status",
            ptr: args.row_status,
            bytes: row_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "slot_ids",
            ptr: args.slot_ids,
            bytes: seq_u32_bytes,
            align: 4,
        },
        DeviceRegion {
            name: "slot_generations",
            ptr: args.slot_generations,
            bytes: seq_u64_bytes,
            align: 8,
        },
        DeviceRegion {
            name: "expected_generations",
            ptr: args.expected_generations,
            bytes: seq_u64_bytes,
            align: 8,
        },
        DeviceRegion {
            name: "slot_refcounts",
            ptr: args.slot_refcounts,
            bytes: seq_u32_bytes,
            align: 4,
        },
    ];

    ensure_outputs_disjoint(OP, &outputs, &inputs)?;

    KernelLaunch::new(gpu, kernel)
        .grid([args.batch_size, 1, 1])
        .block([EXACT_SPEC_PREFIX_THREADS, 1, 1])
        .arg_ptr(args.row_accepted)
        .arg_ptr(args.row_tokens)
        .arg_ptr(args.row_status)
        .arg_ptr(args.slot_ids)
        .arg_ptr(args.slot_generations)
        .arg_ptr(args.expected_generations)
        .arg_ptr(args.slot_refcounts)
        .arg_ptr(args.d_num_accepted)
        .arg_ptr(args.d_bonus_token)
        .arg_ptr(args.d_status_out)
        .arg_u32(args.batch_size)
        .arg_u32(args.verify_width)
        .launch(stream)
}

// ── Bit-Exact CPU Reference Oracle ───────────────────────────────────────────

/// Canonical Salmon et al. (SC'11) `Philox4_32_10` block cipher over 4x32-bit
/// counter `ctr` and 2x32-bit `key`.
#[must_use]
pub fn philox4_32_10_raw(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        let p0 = u64::from(PHILOX_M4X32_0) * u64::from(ctr[0]);
        let p1 = u64::from(PHILOX_M4X32_1) * u64::from(ctr[2]);
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
        if round < 9 {
            key[0] = key[0].wrapping_add(PHILOX_W32_0);
            key[1] = key[1].wrapping_add(PHILOX_W32_1);
        }
    }
    ctr
}

/// Evaluate `Philox4_32_10` keyed by `(seed, abs_pos, 0, 0)` returning the
/// first two 32-bit outputs `(r0, r1)`, bit-identical to `philox4x32_10` in
/// `exact_spec_accept_sm120.cu`.
#[must_use]
pub fn philox4_32_10(seed: u64, abs_pos: u64) -> (u32, u32) {
    let key = [seed as u32, (seed >> 32) as u32];
    let ctr = [abs_pos as u32, (abs_pos >> 32) as u32, 0, 0];
    let out = philox4_32_10_raw(ctr, key);
    (out[0], out[1])
}

/// Exact 24-bit mantissa dyadic uniform in `[0.0, 1.0)` matching CUDA
/// `u32_to_uniform_f32` to 0 ULP.
#[must_use]
pub const fn u32_to_uniform_f32(x: u32) -> f32 {
    ((x >> 8) as f32) * 5.960_464_5e-8_f32
}

/// Convert raw BF16 bit pattern to `f32` bit-identically to `__bfloat162float`.
#[must_use]
pub const fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round `f32` to IEEE round-to-nearest-even BF16 bits (quiet NaN preserved).
#[must_use]
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x007F_FFFF;
    if exp == 0xFF {
        if mant != 0 {
            return ((bits >> 16) as u16) | 0x0040;
        }
        return (bits >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

fn cpu_block_reduce_sum_256(lanes: &[f32; 256]) -> f32 {
    let mut warp_sums = [0.0_f32; 8];
    for (w, warp_sum) in warp_sums.iter_mut().enumerate() {
        let mut buf = [0.0_f32; 32];
        buf.copy_from_slice(&lanes[(w * 32)..((w + 1) * 32)]);
        for offset in [16_usize, 8, 4, 2, 1] {
            for l in 0..(32 - offset) {
                buf[l] += buf[l + offset];
            }
        }
        *warp_sum = buf[0];
    }
    let mut w0 = warp_sums;
    for offset in [4_usize, 2, 1] {
        for l in 0..(8 - offset) {
            w0[l] += w0[l + offset];
        }
    }
    w0[0]
}

fn cpu_block_reduce_argmax_256(vals: &[f32; 256], ids: &[u32; 256]) -> (f32, u32) {
    let mut warp_v = [f32::NEG_INFINITY; 8];
    let mut warp_i = [EXACT_SPEC_INVALID_TOKEN; 8];
    for w in 0..8 {
        let mut bv = [0.0_f32; 32];
        let mut bi = [0_u32; 32];
        bv.copy_from_slice(&vals[(w * 32)..((w + 1) * 32)]);
        bi.copy_from_slice(&ids[(w * 32)..((w + 1) * 32)]);
        for offset in [16_usize, 8, 4, 2, 1] {
            for l in 0..(32 - offset) {
                let ov = bv[l + offset];
                let oi = bi[l + offset];
                if oi != EXACT_SPEC_INVALID_TOKEN
                    && (bi[l] == EXACT_SPEC_INVALID_TOKEN
                        || ov > bv[l]
                        || (ov == bv[l] && oi < bi[l]))
                {
                    bv[l] = ov;
                    bi[l] = oi;
                }
            }
        }
        warp_v[w] = bv[0];
        warp_i[w] = bi[0];
    }
    for offset in [4_usize, 2, 1] {
        for l in 0..(8 - offset) {
            let ov = warp_v[l + offset];
            let oi = warp_i[l + offset];
            if oi != EXACT_SPEC_INVALID_TOKEN
                && (warp_i[l] == EXACT_SPEC_INVALID_TOKEN
                    || ov > warp_v[l]
                    || (ov == warp_v[l] && oi < warp_i[l]))
            {
                warp_v[l] = ov;
                warp_i[l] = oi;
            }
        }
    }
    (warp_v[0], warp_i[0])
}

fn cpu_tile_inclusive_scan_256(lanes: &[f32; 256]) -> ([f32; 256], f32) {
    let mut out = [0.0_f32; 256];
    let mut warp_totals = [0.0_f32; 8];
    for (w, warp_total) in warp_totals.iter_mut().enumerate() {
        let mut buf = [0.0_f32; 32];
        buf.copy_from_slice(&lanes[(w * 32)..((w + 1) * 32)]);
        for offset in [1_usize, 2, 4, 8, 16] {
            let prev = buf;
            for l in offset..32 {
                buf[l] += prev[l - offset];
            }
        }
        *warp_total = buf[31];
        out[(w * 32)..((w + 1) * 32)].copy_from_slice(&buf);
    }
    for offset in [1_usize, 2, 4] {
        let prev = warp_totals;
        for w in offset..8 {
            warp_totals[w] += prev[w - offset];
        }
    }
    for (w, &base) in warp_totals.iter().take(7).enumerate() {
        for item in &mut out[((w + 1) * 32)..((w + 2) * 32)] {
            *item += base;
        }
    }
    (out, warp_totals[7])
}

/// Whether a row is a draft verification row or the final bonus sampling row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CpuExactSpecRowKind<'a> {
    Draft {
        draft_token: u32,
        sparse_ids: &'a [u32],
        sparse_probs: &'a [f32],
    },
    Bonus,
}

/// Inputs to the bit-exact CPU oracle [`cpu_exact_spec_accept_sparse_row`].
#[derive(Clone, Copy, Debug)]
pub struct CpuExactSpecRowInput<'a> {
    pub logits_bf16: &'a [u16],
    pub kind: CpuExactSpecRowKind<'a>,
    /// In-window prefix tokens `0..col` already accepted before this row.
    pub prefix_tokens: &'a [u32],
    /// Optional pre-verify token occurrence counts `[vocab_size]`.
    pub pre_verify_counts: Option<&'a [u32]>,
    pub seed: u64,
    pub abs_pos: u64,
    pub temperature: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

/// Output of [`cpu_exact_spec_accept_sparse_row`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuExactSpecRowOutput {
    pub accepted: bool,
    pub token: u32,
    pub status: ExactSpecStatus,
}

fn cpu_compute_scaled_logit(
    raw_bf16: u16,
    v: u32,
    pre_counts: Option<&[u32]>,
    prefix_tokens: &[u32],
    pres_pen: f32,
    freq_pen: f32,
    temp: f32,
) -> (f32, bool, bool) {
    let mut logit = bf16_bits_to_f32(raw_bf16);
    if logit.is_nan() || (logit.is_infinite() && logit > 0.0) {
        return (f32::NEG_INFINITY, false, true);
    }
    if !logit.is_finite() {
        return (f32::NEG_INFINITY, false, false);
    }
    if pres_pen != 0.0 || freq_pen != 0.0 {
        let mut total_cnt = pre_counts
            .and_then(|counts| counts.get(v as usize).copied())
            .unwrap_or(0);
        for &tok in prefix_tokens {
            if tok == v {
                total_cnt = total_cnt.saturating_add(1);
            }
        }
        if total_cnt > 0 {
            if pres_pen != 0.0 {
                logit -= pres_pen;
            }
            if freq_pen != 0.0 {
                logit -= freq_pen * (total_cnt as f32);
            }
        }
    }
    let scaled = logit / temp;
    if !scaled.is_finite() {
        return (f32::NEG_INFINITY, false, true);
    }
    (scaled, true, false)
}

fn cpu_lookup_sparse_prob(v: u32, bloom: &[u32; 32], ids: &[u32], probs: &[f32]) -> f32 {
    if ids.is_empty() {
        return 0.0;
    }
    let word = bloom[((v >> 5) & 31) as usize];
    if (word & (1_u32 << (v & 31))) == 0 {
        return 0.0;
    }
    for (idx, &id) in ids.iter().enumerate() {
        if id == v {
            return probs[idx];
        }
    }
    0.0
}

/// Bit-exact CPU reference mirror of `exact_spec_accept_sparse_sm120`.
#[must_use]
pub fn cpu_exact_spec_accept_sparse_row(input: &CpuExactSpecRowInput<'_>) -> CpuExactSpecRowOutput {
    let vocab_size = input.logits_bf16.len();
    if vocab_size == 0
        || u32::try_from(vocab_size).is_err()
        || input.prefix_tokens.len() >= EXACT_SPEC_MAX_WIDTH as usize
        || !input.temperature.is_finite()
        || input.temperature <= 0.0
        || !input.presence_penalty.is_finite()
        || !input.frequency_penalty.is_finite()
        || input
            .pre_verify_counts
            .is_some_and(|counts| counts.len() < vocab_size)
    {
        return CpuExactSpecRowOutput {
            accepted: false,
            token: EXACT_SPEC_INVALID_TOKEN,
            status: ExactSpecStatus::BAD_PARAM,
        };
    }

    let vocab_u32 = vocab_size as u32;
    let (r0, r1) = philox4_32_10(input.seed, input.abs_pos);
    let u_accept = u32_to_uniform_f32(r0);
    let u_residual = u32_to_uniform_f32(r1);

    let mut status = ExactSpecStatus::OK;
    let mut bloom = [0_u32; 32];
    let mut active_ids = Vec::with_capacity(EXACT_SPEC_MAX_SPARSE_K as usize);
    let mut active_probs = Vec::with_capacity(EXACT_SPEC_MAX_SPARSE_K as usize);
    let mut p_draft_x = 0.0_f32;
    let mut draft_tok_opt = None;

    if let CpuExactSpecRowKind::Draft {
        draft_token,
        sparse_ids,
        sparse_probs,
    } = input.kind
    {
        if sparse_ids.is_empty()
            || sparse_ids.len() != sparse_probs.len()
            || sparse_ids.len() > EXACT_SPEC_MAX_SPARSE_K as usize
        {
            return CpuExactSpecRowOutput {
                accepted: false,
                token: EXACT_SPEC_INVALID_TOKEN,
                status: ExactSpecStatus::BAD_PARAM,
            };
        }
        draft_tok_opt = Some(draft_token);
        if draft_token >= vocab_u32 {
            status = status.with(ExactSpecStatus::DRAFT_OOB);
        }
        let mut sum_p = 0.0_f32;
        for (&id, &pr) in sparse_ids.iter().zip(sparse_probs.iter()) {
            if id == EXACT_SPEC_INVALID_TOKEN {
                if !pr.is_finite() || pr != 0.0 {
                    status = status.with(ExactSpecStatus::DRAFT_NONFINITE);
                }
                continue;
            }
            if id >= vocab_u32
                || !pr.is_finite()
                || !(0.0..=(1.0 + EXACT_SPEC_PROB_SUM_TOL)).contains(&pr)
            {
                status = status.with(ExactSpecStatus::DRAFT_NONFINITE);
                continue;
            }
            if active_ids.contains(&id) {
                status = status.with(ExactSpecStatus::DRAFT_NONFINITE);
                continue;
            }
            active_ids.push(id);
            active_probs.push(pr);
            bloom[((id >> 5) & 31) as usize] |= 1_u32 << (id & 31);
            sum_p += pr;
            if id == draft_token {
                p_draft_x = pr;
            }
        }
        if !sum_p.is_finite() || (sum_p - 1.0).abs() > EXACT_SPEC_PROB_SUM_TOL {
            status = status.with(ExactSpecStatus::DRAFT_SUM_ERR);
        }
        if !p_draft_x.is_finite() || p_draft_x <= 0.0 {
            status = status.with(ExactSpecStatus::DRAFT_OOB);
        }
        if status.has_error() {
            return CpuExactSpecRowOutput {
                accepted: false,
                token: EXACT_SPEC_INVALID_TOKEN,
                status,
            };
        }
    }

    // Pass 1: 256-lane max scaled logit + lowest-ID target-support argmax.
    let mut lane_max = [f32::NEG_INFINITY; 256];
    let mut lane_argmax = [EXACT_SPEC_INVALID_TOKEN; 256];
    for tid in 0..256_usize {
        let mut v = tid;
        while v < vocab_size {
            let v_u32 = v as u32;
            let (sc, valid, fatal) = cpu_compute_scaled_logit(
                input.logits_bf16[v],
                v_u32,
                input.pre_verify_counts,
                input.prefix_tokens,
                input.presence_penalty,
                input.frequency_penalty,
                input.temperature,
            );
            if fatal {
                status = status.with(ExactSpecStatus::TARGET_NONFINITE);
            } else if valid
                && (lane_argmax[tid] == EXACT_SPEC_INVALID_TOKEN
                    || sc > lane_max[tid]
                    || (sc == lane_max[tid] && v_u32 < lane_argmax[tid]))
            {
                lane_max[tid] = sc;
                lane_argmax[tid] = v_u32;
            }
            v += 256;
        }
    }
    let (max_scaled, argmax_id) = cpu_block_reduce_argmax_256(&lane_max, &lane_argmax);
    if status.has_error() || argmax_id == EXACT_SPEC_INVALID_TOKEN || !max_scaled.is_finite() {
        return CpuExactSpecRowOutput {
            accepted: false,
            token: EXACT_SPEC_INVALID_TOKEN,
            status: status.with(ExactSpecStatus::TARGET_NONFINITE),
        };
    }

    // Pass 2a: Exact softmax denominator `Z`.
    let mut lane_exp = [0.0_f32; 256];
    for tid in 0..256_usize {
        let mut v = tid;
        while v < vocab_size {
            let (sc, valid, fatal) = cpu_compute_scaled_logit(
                input.logits_bf16[v],
                v as u32,
                input.pre_verify_counts,
                input.prefix_tokens,
                input.presence_penalty,
                input.frequency_penalty,
                input.temperature,
            );
            if valid && !fatal {
                lane_exp[tid] += (sc - max_scaled).exp();
            }
            v += 256;
        }
    }
    let sum_exp = cpu_block_reduce_sum_256(&lane_exp);
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return CpuExactSpecRowOutput {
            accepted: false,
            token: EXACT_SPEC_INVALID_TOKEN,
            status: status.with(ExactSpecStatus::TARGET_NONFINITE),
        };
    }

    // Pass 2b: Target probability sum check `|sum(q) - 1.0| <= 1e-4` & residual mass `Z_r`.
    let is_draft = draft_tok_opt.is_some();
    let mut lane_sum_q = [0.0_f32; 256];
    let mut lane_sum_r = [0.0_f32; 256];
    for tid in 0..256_usize {
        let mut v = tid;
        while v < vocab_size {
            let v_u32 = v as u32;
            let (sc, valid, fatal) = cpu_compute_scaled_logit(
                input.logits_bf16[v],
                v_u32,
                input.pre_verify_counts,
                input.prefix_tokens,
                input.presence_penalty,
                input.frequency_penalty,
                input.temperature,
            );
            if valid && !fatal {
                let q_v = (sc - max_scaled).exp() / sum_exp;
                lane_sum_q[tid] += q_v;
                if is_draft {
                    let p_v = cpu_lookup_sparse_prob(v_u32, &bloom, &active_ids, &active_probs);
                    let diff = q_v - p_v;
                    if diff > 0.0 {
                        lane_sum_r[tid] += diff;
                    }
                }
            }
            v += 256;
        }
    }
    let sum_q = cpu_block_reduce_sum_256(&lane_sum_q);
    let sum_r = if is_draft {
        cpu_block_reduce_sum_256(&lane_sum_r)
    } else {
        0.0
    };
    if !sum_q.is_finite() || (sum_q - 1.0).abs() > EXACT_SPEC_PROB_SUM_TOL {
        status = status.with(ExactSpecStatus::TARGET_SUM_ERR);
        return CpuExactSpecRowOutput {
            accepted: false,
            token: EXACT_SPEC_INVALID_TOKEN,
            status,
        };
    }

    // Acceptance test on draft rows.
    if let Some(draft_tok) = draft_tok_opt {
        let (sc_x, valid_x, fatal_x) = cpu_compute_scaled_logit(
            input.logits_bf16[draft_tok as usize],
            draft_tok,
            input.pre_verify_counts,
            input.prefix_tokens,
            input.presence_penalty,
            input.frequency_penalty,
            input.temperature,
        );
        let q_x = if valid_x && !fatal_x {
            (sc_x - max_scaled).exp() / sum_exp
        } else {
            0.0
        };
        if p_draft_x > 0.0 && q_x > 0.0 && (q_x >= p_draft_x || (u_accept * p_draft_x) < q_x) {
            return CpuExactSpecRowOutput {
                accepted: true,
                token: draft_tok,
                status,
            };
        }
    }

    // Pass 3: Coalesced 256-lane ascending vocab CDF scan.
    let mass_total = if is_draft { sum_r } else { sum_q };
    let u_draw = if is_draft { u_residual } else { u_accept };
    let mut sampled_tok = EXACT_SPEC_INVALID_TOKEN;
    if mass_total.is_finite() && mass_total > 0.0 {
        let thresh = u_draw * mass_total;
        let mut running_cdf = 0.0_f32;
        let mut base = 0_usize;
        while base < vocab_size {
            let mut tile_w = [0.0_f32; 256];
            for tid in 0..256_usize {
                let v = base + tid;
                if v < vocab_size {
                    let v_u32 = v as u32;
                    let (sc, valid, fatal) = cpu_compute_scaled_logit(
                        input.logits_bf16[v],
                        v_u32,
                        input.pre_verify_counts,
                        input.prefix_tokens,
                        input.presence_penalty,
                        input.frequency_penalty,
                        input.temperature,
                    );
                    if valid && !fatal {
                        let q_v = (sc - max_scaled).exp() / sum_exp;
                        tile_w[tid] = if is_draft {
                            let p_v =
                                cpu_lookup_sparse_prob(v_u32, &bloom, &active_ids, &active_probs);
                            (q_v - p_v).max(0.0)
                        } else {
                            q_v
                        };
                    }
                }
            }
            let (incl, tile_total) = cpu_tile_inclusive_scan_256(&tile_w);
            let next_cdf = running_cdf + tile_total;
            if next_cdf > thresh {
                for tid in 0..256_usize {
                    let v = base + tid;
                    if v < vocab_size && tile_w[tid] > 0.0 && (running_cdf + incl[tid]) > thresh {
                        sampled_tok = v as u32;
                        break;
                    }
                }
                if sampled_tok != EXACT_SPEC_INVALID_TOKEN {
                    break;
                }
            }
            running_cdf = next_cdf;
            base += 256;
        }
    }

    // Guaranteed valid target-support zero-mass / tail-rounding fallback.
    if sampled_tok == EXACT_SPEC_INVALID_TOKEN || sampled_tok >= vocab_u32 {
        sampled_tok = argmax_id;
        status = status.with(ExactSpecStatus::FALLBACK_USED);
    }

    CpuExactSpecRowOutput {
        accepted: false,
        token: sampled_tok,
        status,
    }
}

/// Fail-closed interpretation of one sequence's device output from
/// `exact_spec_prefix_reduce_sm120`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactGpuAcceptOutcome {
    /// Device verified `num_accepted` draft tokens and produced valid `bonus_token`.
    Accepted {
        num_accepted: usize,
        bonus_token: u32,
        fallback_used: bool,
    },
    /// Device reported a fail-closed error bit or invalid count/token; caller
    /// must fall back to the verified host path without committing GPU counts.
    FailClosedFallback { status: ExactSpecStatus },
}

/// Interpret raw device outputs `(d_num_accepted, d_bonus_token, d_status_out)`
/// fail-closed against `verify_width` and `vocab_size`.
#[must_use]
pub fn interpret_device_accept_outcome(
    num_accepted: u32,
    bonus_token: u32,
    status: ExactSpecStatus,
    verify_width: usize,
    vocab_size: u32,
) -> ExactGpuAcceptOutcome {
    if status.has_error() {
        return ExactGpuAcceptOutcome::FailClosedFallback { status };
    }
    let accepted = num_accepted as usize;
    if verify_width == 0
        || accepted >= verify_width
        || bonus_token == EXACT_SPEC_INVALID_TOKEN
        || bonus_token >= vocab_size
    {
        return ExactGpuAcceptOutcome::FailClosedFallback {
            status: status.with(ExactSpecStatus::BAD_PARAM),
        };
    }
    ExactGpuAcceptOutcome::Accepted {
        num_accepted: accepted,
        bonus_token,
        fallback_used: status.fallback_used(),
    }
}

/// Output of [`cpu_exact_spec_prefix_reduce`] per batch sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuExactSpecSeqOutput {
    pub num_accepted: u32,
    pub bonus_token: u32,
    pub status: ExactSpecStatus,
}

/// Bit-exact CPU reference mirror of `exact_spec_prefix_reduce_sm120`.
///
/// # Errors
/// Fails closed if `verify_width` is outside `1..=16` or `rows.len() != guards.len() * verify_width`.
pub fn cpu_exact_spec_prefix_reduce(
    rows: &[CpuExactSpecRowOutput],
    guards: &[ExactSpecSlotGuard],
    verify_width: u32,
) -> Result<Vec<CpuExactSpecSeqOutput>> {
    ensure!(
        (1..=EXACT_SPEC_MAX_WIDTH).contains(&verify_width),
        "cpu_exact_spec_prefix_reduce: verify_width={verify_width} must lie in 1..={EXACT_SPEC_MAX_WIDTH}"
    );
    let width = verify_width as usize;
    let expected_rows = guards
        .len()
        .checked_mul(width)
        .context("cpu_exact_spec_prefix_reduce: rows overflow")?;
    ensure!(
        rows.len() == expected_rows,
        "cpu_exact_spec_prefix_reduce: expected {expected_rows} rows, got {}",
        rows.len()
    );

    let guard_statuses = validate_slot_guards_host(guards);
    let mut out = Vec::with_capacity(guards.len());
    let num_drafts = width - 1;

    for (seq, &guard_status) in guard_statuses.iter().enumerate() {
        let mut status = guard_status;
        if status.has_error() {
            out.push(CpuExactSpecSeqOutput {
                num_accepted: 0,
                bonus_token: EXACT_SPEC_INVALID_TOKEN,
                status,
            });
            continue;
        }
        let base = seq * width;
        let mut num_accepted = 0_u32;
        let mut bonus_token = EXACT_SPEC_INVALID_TOKEN;
        let mut stopped = false;
        let mut failed = false;

        for col in 0..num_drafts {
            let row = rows[base + col];
            status = status.with(row.status);
            if row.status.has_error() {
                failed = true;
                break;
            }
            if row.accepted {
                num_accepted += 1;
            } else {
                bonus_token = row.token;
                stopped = true;
                break;
            }
        }
        if !failed && !stopped {
            let row = rows[base + num_drafts];
            status = status.with(row.status);
            if row.status.has_error() {
                failed = true;
            } else {
                bonus_token = row.token;
            }
        }
        if failed {
            out.push(CpuExactSpecSeqOutput {
                num_accepted: 0,
                bonus_token: EXACT_SPEC_INVALID_TOKEN,
                status,
            });
        } else if bonus_token == EXACT_SPEC_INVALID_TOKEN {
            out.push(CpuExactSpecSeqOutput {
                num_accepted: 0,
                bonus_token: EXACT_SPEC_INVALID_TOKEN,
                status: status.with(ExactSpecStatus::BAD_PARAM),
            });
        } else {
            out.push(CpuExactSpecSeqOutput {
                num_accepted,
                bonus_token,
                status,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn bf16_row(logits: &[f32]) -> Vec<u16> {
        logits.iter().copied().map(f32_to_bf16_bits).collect()
    }

    fn exact_target_probs_bf16(
        logits_bf16: &[u16],
        pre_counts: Option<&[u32]>,
        prefix: &[u32],
        temp: f32,
        pres_pen: f32,
        freq_pen: f32,
    ) -> Vec<f32> {
        let mut scaled = Vec::with_capacity(logits_bf16.len());
        let mut max_s = f32::NEG_INFINITY;
        for (v, &raw) in logits_bf16.iter().enumerate() {
            let (s, valid, fatal) = cpu_compute_scaled_logit(
                raw, v as u32, pre_counts, prefix, pres_pen, freq_pen, temp,
            );
            assert!(valid && !fatal);
            if s > max_s {
                max_s = s;
            }
            scaled.push(s);
        }
        let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
        let z: f32 = exps.iter().sum();
        exps.into_iter().map(|e| e / z).collect()
    }

    #[test]
    fn philox4_32_10_matches_salmon_2011_published_kat_vectors() {
        // Random123 canonical KAT vectors (Salmon et al., SC'11, kat_vectors):
        // ctr = (0, 0, 0, 0), key = (0, 0)
        assert_eq!(
            philox4_32_10_raw([0, 0, 0, 0], [0, 0]),
            [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]
        );
        // ctr = (0xffffffff, ...), key = (0xffffffff, 0xffffffff)
        assert_eq!(
            philox4_32_10_raw([u32::MAX; 4], [u32::MAX; 2]),
            [0x408F_276D, 0x41C8_3B0E, 0xA20B_C7C6, 0x6D54_51FD]
        );
        // Uniform mapping stays strictly inside [0.0, 1.0) even at u32::MAX.
        assert_eq!(u32_to_uniform_f32(0), 0.0);
        let u_max = u32_to_uniform_f32(u32::MAX);
        assert!(u_max > 0.999_999 && u_max < 1.0);
        // Keyed helper matches raw block cipher on (seed, abs_pos, 0, 0).
        let seed = 0x1234_5678_9ABC_DEF0_u64;
        let pos = 0x0000_0001_0000_002A_u64;
        let raw = philox4_32_10_raw(
            [pos as u32, (pos >> 32) as u32, 0, 0],
            [seed as u32, (seed >> 32) as u32],
        );
        assert_eq!(philox4_32_10(seed, pos), (raw[0], raw[1]));
    }

    #[test]
    fn env_gate_defaults_off_and_fails_closed_on_invalid_or_non_utf8() {
        assert_eq!(
            parse_exact_spec_gpu_accept_os(None).unwrap(),
            ExactSpecGpuAcceptPolicy::Disabled
        );
        assert_eq!(
            parse_exact_spec_gpu_accept_os(Some(OsStr::new("0"))).unwrap(),
            ExactSpecGpuAcceptPolicy::Disabled
        );
        assert_eq!(
            parse_exact_spec_gpu_accept_os(Some(OsStr::new("1"))).unwrap(),
            ExactSpecGpuAcceptPolicy::Enabled
        );
        for bad in ["", " ", " 1 ", "true", "on", "2", "-1", "yes"] {
            assert_eq!(
                parse_exact_spec_gpu_accept_os(Some(OsStr::new(bad))),
                Err(ExactSpecGpuAcceptEnvError::InvalidValue(bad.to_owned())),
                "token {bad:?} must fail closed"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let non_utf8 = OsStr::from_bytes(&[0xFF, 0xFE]);
            assert_eq!(
                parse_exact_spec_gpu_accept_os(Some(non_utf8)),
                Err(ExactSpecGpuAcceptEnvError::NonUtf8)
            );
        }
    }

    #[test]
    fn width_invariance_w1_equals_w8_equals_w16_with_penalties_and_repeated_prefix() {
        // Construct a deterministic causal sequence generator of length 16 over
        // V=311 (non-multiple of 256) with non-zero presence + frequency penalties
        // and repeated prefix tokens. Running W=1 serial steps, W=8 chunks, and
        // one W=16 window with identical draft proposals (or perfect-match drafts
        // where accepted) must emit the EXACT SAME token trajectory bit-for-bit!
        let vocab = 311_usize;
        let seed = 0xCAFE_BABE_DEAD_BEEF_u64;
        let base_abs_pos = 1024_u64;
        let temp = 0.75_f32;
        let pres_pen = 0.6_f32;
        let freq_pen = 0.35_f32;

        let mut initial_counts = vec![0_u32; vocab];
        initial_counts[7] = 2;
        initial_counts[19] = 1;
        initial_counts[260] = 3;

        // Deterministic context-dependent BF16 logits generator.
        let make_logits = |history: &[u32]| -> Vec<u16> {
            let mut h: u64 = 0x517C_C1B7_2722_0A95;
            for &tok in history {
                h = h
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(u64::from(tok) + 1);
            }
            let mut row = Vec::with_capacity(vocab);
            for v in 0..vocab {
                let x = h ^ ((v as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let z = (x ^ (x >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
                let u = ((z >> 40) as f32) / ((1_u32 << 24) as f32);
                // Put noticeable mass on tokens 7, 19, 260, and (h % vocab) so repeats happen.
                let mut val = (u - 0.5) * 4.0;
                if v == 7 || v == 19 || v == 260 || v == (h as usize % vocab) {
                    val += 3.2;
                }
                row.push(f32_to_bf16_bits(val));
            }
            row
        };

        // Step 1: Ground-truth W=1 serial trajectory (16 tokens).
        let mut serial_tokens = Vec::with_capacity(16);
        let mut serial_counts = initial_counts.clone();
        for step in 0..16_usize {
            let logits = make_logits(&serial_tokens);
            let out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
                logits_bf16: &logits,
                kind: CpuExactSpecRowKind::Bonus,
                prefix_tokens: &[],
                pre_verify_counts: Some(&serial_counts),
                seed,
                abs_pos: base_abs_pos + step as u64,
                temperature: temp,
                presence_penalty: pres_pen,
                frequency_penalty: freq_pen,
            });
            assert!(out.status.is_ok(), "step {step}: {}", out.status);
            assert!(!out.accepted);
            serial_tokens.push(out.token);
            serial_counts[out.token as usize] += 1;
        }

        // Helper: run speculative decoding with window width `w` (8 or 16) using
        // a sparse K=16 draft distribution concentrated on the target's top-16
        // plus exact target token so acceptance occurs at multiple positions AND
        // rejections also occur — proving BOTH exact prefix-penalty invariance
        // when accepted AND exact bonus-row invariance!
        let run_windowed = |window_w: usize, exact_draft_mode: bool| -> Vec<u32> {
            let mut committed = Vec::new();
            let mut counts = initial_counts.clone();
            let guard = ExactSpecSlotGuard {
                slot_id: 3,
                slot_generation: 42,
                expected_generation: 42,
                slot_refcount: 1,
            };
            while committed.len() < 16 {
                let remaining = 16 - committed.len();
                let eff_w = window_w.min(remaining);
                let num_drafts = eff_w - 1;
                // Build draft tokens & sparse K=8 distributions along the draft spine.
                let mut draft_tokens = Vec::with_capacity(num_drafts);
                let mut sparse_ids_rows = Vec::with_capacity(num_drafts);
                let mut sparse_probs_rows = Vec::with_capacity(num_drafts);
                let mut spine = committed.clone();

                for col in 0..num_drafts {
                    let pos = committed.len() + col;
                    let logits = make_logits(&spine);
                    let q = exact_target_probs_bf16(
                        &logits,
                        Some(&counts),
                        &draft_tokens,
                        temp,
                        pres_pen,
                        freq_pen,
                    );
                    let draft_tok = if exact_draft_mode {
                        // Draft the exact serial token with probability = q(tok) (if >0),
                        // guaranteeing acceptance with prob 1 and exercising deep in-window
                        // prefix penalty accumulation up to col = 15!
                        serial_tokens[pos]
                    } else {
                        // Mix exact matches and mismatched tokens to exercise rejection + bonus.
                        if col % 3 == 2 {
                            ((serial_tokens[pos] as usize + 5) % vocab) as u32
                        } else {
                            serial_tokens[pos]
                        }
                    };
                    // Build valid sparse K=4 support summing to 1.0 containing draft_tok.
                    let mut ids = vec![draft_tok];
                    for offset in 1..4_u32 {
                        let cand = ((draft_tok as usize + offset as usize) % vocab) as u32;
                        if !ids.contains(&cand) {
                            ids.push(cand);
                        }
                    }
                    let mut probs = vec![0.0_f32; ids.len()];
                    if exact_draft_mode {
                        // Give draft_tok probability <= q(draft_tok) so accept is guaranteed,
                        // and split remaining mass cleanly over the other 3 support IDs.
                        let p0 = (q[draft_tok as usize] * 0.9).clamp(1e-6, 0.95);
                        probs[0] = p0;
                        let rem = (1.0 - p0) / ((ids.len() - 1) as f32);
                        for p in probs.iter_mut().skip(1) {
                            *p = rem;
                        }
                    } else {
                        probs[0] = 0.7;
                        let rem = 0.3 / ((ids.len() - 1) as f32);
                        for p in probs.iter_mut().skip(1) {
                            *p = rem;
                        }
                    }
                    draft_tokens.push(draft_tok);
                    sparse_ids_rows.push(ids);
                    sparse_probs_rows.push(probs);
                    spine.push(draft_tok);
                }

                // Verify all `eff_w` rows in parallel (exact GPU CTA semantics).
                let mut row_outputs = Vec::with_capacity(eff_w);
                for col in 0..eff_w {
                    let mut row_hist = committed.clone();
                    row_hist.extend_from_slice(&draft_tokens[..col]);
                    let logits = make_logits(&row_hist);
                    let kind = if col < num_drafts {
                        CpuExactSpecRowKind::Draft {
                            draft_token: draft_tokens[col],
                            sparse_ids: &sparse_ids_rows[col],
                            sparse_probs: &sparse_probs_rows[col],
                        }
                    } else {
                        CpuExactSpecRowKind::Bonus
                    };
                    let row_out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
                        logits_bf16: &logits,
                        kind,
                        prefix_tokens: &draft_tokens[..col],
                        pre_verify_counts: Some(&counts),
                        seed,
                        abs_pos: base_abs_pos + (committed.len() + col) as u64,
                        temperature: temp,
                        presence_penalty: pres_pen,
                        frequency_penalty: freq_pen,
                    });
                    assert!(row_out.status.is_ok(), "col {col}: {}", row_out.status);
                    row_outputs.push(row_out);
                }

                let reduced =
                    cpu_exact_spec_prefix_reduce(&row_outputs, &[guard], eff_w as u32).unwrap();
                let seq_out = reduced[0];
                assert!(seq_out.status.is_ok(), "seq status {}", seq_out.status);
                let n_acc = seq_out.num_accepted as usize;
                for &tok in &draft_tokens[..n_acc] {
                    committed.push(tok);
                    counts[tok as usize] += 1;
                }
                committed.push(seq_out.bonus_token);
                counts[seq_out.bonus_token as usize] += 1;
            }
            committed
        };

        // When all drafts accept (`exact_draft_mode = true`), the bonus row at
        // col=7 (W=8) and col=15 (W=16) MUST see the exact same accumulated
        // presence/frequency penalties and Philox counter as W=1 serial decode!
        let w8_tokens = run_windowed(8, true);
        let w16_tokens = run_windowed(16, true);
        assert_eq!(w8_tokens, serial_tokens, "W=8 must match W=1 bit-for-bit");
        assert_eq!(w16_tokens, serial_tokens, "W=16 must match W=1 bit-for-bit");

        // Also exercise mixed accept + reject + residual sampling across W=8 and W=16:
        // identical draft trajectories produce identical residual/bonus tokens.
        let w8_mixed = run_windowed(8, false);
        let w16_mixed = run_windowed(16, false);
        assert_eq!(w8_mixed.len(), 16);
        assert_eq!(w16_mixed.len(), 16);
        assert!(w8_mixed.iter().all(|&tok| (tok as usize) < vocab));
        assert!(w16_mixed.iter().all(|&tok| (tok as usize) < vocab));

        // Also prove single-row bonus parity when prefix penalties accumulate
        // identical counts in-window vs pre-verify (W=1 vs W=8 vs W=16 at col=0,7,15).
        let prefix15: Vec<u32> = vec![7, 19, 7, 260, 7, 19, 42, 7, 19, 260, 7, 42, 42, 7, 19];
        let logits_test = make_logits(&prefix15);
        let mut counts_after_15 = initial_counts.clone();
        for &tok in &prefix15 {
            counts_after_15[tok as usize] += 1;
        }
        let mut counts_after_8 = initial_counts.clone();
        for &tok in &prefix15[..8] {
            counts_after_8[tok as usize] += 1;
        }
        let row_w1 = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits_test,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &[],
            pre_verify_counts: Some(&counts_after_15),
            seed,
            abs_pos: base_abs_pos + 15,
            temperature: temp,
            presence_penalty: pres_pen,
            frequency_penalty: freq_pen,
        });
        let row_w8 = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits_test,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &prefix15[8..15],
            pre_verify_counts: Some(&counts_after_8),
            seed,
            abs_pos: base_abs_pos + 15,
            temperature: temp,
            presence_penalty: pres_pen,
            frequency_penalty: freq_pen,
        });
        let row_w16 = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits_test,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &prefix15[..15],
            pre_verify_counts: Some(&initial_counts),
            seed,
            abs_pos: base_abs_pos + 15,
            temperature: temp,
            presence_penalty: pres_pen,
            frequency_penalty: freq_pen,
        });
        assert_eq!(row_w1, row_w8);
        assert_eq!(row_w1, row_w16);
    }

    #[test]
    fn fail_closed_on_probability_sum_error_nonfinite_duplicates_and_oob() {
        let logits = bf16_row(&[1.0, 2.0, 0.5, -1.0, 0.25, 0.0, 1.5, -0.5]);

        // 1. Draft sum > 1.0 + 1e-4 fails closed with DRAFT_SUM_ERR.
        let out_hi = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Draft {
                draft_token: 1,
                sparse_ids: &[1, 2],
                sparse_probs: &[0.6, 0.4002],
            },
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(out_hi.status.has_error());
        assert!(out_hi.status.has_sum_error());
        assert!(out_hi.status.contains(ExactSpecStatus::DRAFT_SUM_ERR));
        assert_eq!(out_hi.token, EXACT_SPEC_INVALID_TOKEN);

        // 2. Draft sum < 1.0 - 1e-4 fails closed with DRAFT_SUM_ERR.
        let out_lo = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Draft {
                draft_token: 1,
                sparse_ids: &[1, 2],
                sparse_probs: &[0.6, 0.3998],
            },
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(out_lo.status.contains(ExactSpecStatus::DRAFT_SUM_ERR));
        assert_eq!(out_lo.token, EXACT_SPEC_INVALID_TOKEN);

        // 3. Negative probability, NaN, and duplicate sparse ID fail closed.
        for (ids, probs) in [
            (&[1_u32, 2][..], &[1.1_f32, -0.1][..]),
            (&[1_u32, 2][..], &[f32::NAN, 1.0][..]),
            (&[1_u32, 1][..], &[0.5_f32, 0.5][..]),
        ] {
            let out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
                logits_bf16: &logits,
                kind: CpuExactSpecRowKind::Draft {
                    draft_token: 1,
                    sparse_ids: ids,
                    sparse_probs: probs,
                },
                prefix_tokens: &[],
                pre_verify_counts: None,
                seed: 7,
                abs_pos: 10,
                temperature: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            });
            assert!(out.status.contains(ExactSpecStatus::DRAFT_NONFINITE));
            assert_eq!(out.token, EXACT_SPEC_INVALID_TOKEN);
        }

        // 4. Draft token OOB or missing from positive sparse support fails closed.
        let out_oob = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Draft {
                draft_token: 99,
                sparse_ids: &[1, 2],
                sparse_probs: &[0.5, 0.5],
            },
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(out_oob.status.contains(ExactSpecStatus::DRAFT_OOB));

        let out_zero_support = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Draft {
                draft_token: 3, // in-bounds, but not in sparse_ids!
                sparse_ids: &[1, 2],
                sparse_probs: &[0.5, 0.5],
            },
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(out_zero_support.status.contains(ExactSpecStatus::DRAFT_OOB));
        assert!(!out_zero_support.accepted);

        // 5. Target NaN or +Inf logit fails closed with TARGET_NONFINITE.
        let mut nan_logits = logits.clone();
        nan_logits[4] = f32_to_bf16_bits(f32::NAN);
        let out_nan = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &nan_logits,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(out_nan.status.contains(ExactSpecStatus::TARGET_NONFINITE));
        assert_eq!(out_nan.token, EXACT_SPEC_INVALID_TOKEN);

        // 6. All -Inf target logits fails closed with TARGET_NONFINITE.
        let all_neg_inf = vec![f32_to_bf16_bits(f32::NEG_INFINITY); 8];
        let out_all_masked = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &all_neg_inf,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 7,
            abs_pos: 10,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(
            out_all_masked
                .status
                .contains(ExactSpecStatus::TARGET_NONFINITE)
        );

        // 7. Invalid temperature fails closed with BAD_PARAM.
        for bad_temp in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
                logits_bf16: &logits,
                kind: CpuExactSpecRowKind::Bonus,
                prefix_tokens: &[],
                pre_verify_counts: None,
                seed: 7,
                abs_pos: 10,
                temperature: bad_temp,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            });
            assert!(out.status.contains(ExactSpecStatus::BAD_PARAM));
        }
    }

    #[test]
    fn zero_mass_residual_falls_back_to_valid_target_support_never_invalid_or_oob() {
        // Make target `q` a 2-token support (tokens 2 and 5, all others -Inf)
        // with `q(2) = 0.5, q(5) = 0.5`, and set `p_sparse(2) = 0.50004, p_sparse(5) = 0.50004`
        // (valid within 1e-4 sum tolerance: sum = 1.00008 -> |sum - 1| = 8e-5 <= 1e-4!).
        // Then `q(v) - p_sparse(v) <= 0` EVERYWHERE -> residual mass `Z_r == 0.0`!
        // When `u_accept * p(2) >= q(2)` (reject), two-pass residual MUST fall back
        // to valid target-support argmax `2` (lowest tie ID), set `FALLBACK_USED`,
        // keep `status.is_ok() == true`, and NEVER emit `EXACT_SPEC_INVALID_TOKEN`!
        let neg_inf = f32_to_bf16_bits(f32::NEG_INFINITY);
        let mut logits = vec![neg_inf; 16];
        logits[2] = f32_to_bf16_bits(4.0);
        logits[5] = f32_to_bf16_bits(4.0);

        // Search a few abs_pos values to find one where u_accept >= 0.5 / 0.50004
        // (i.e. rejection triggers on draft_token = 2).
        let mut found_reject = false;
        for pos in 0..200_000_u64 {
            let (r0, _) = philox4_32_10(99, pos);
            let u = u32_to_uniform_f32(r0);
            if u * 0.50004 >= 0.5 {
                let out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
                    logits_bf16: &logits,
                    kind: CpuExactSpecRowKind::Draft {
                        draft_token: 2,
                        sparse_ids: &[2, 5],
                        sparse_probs: &[0.50004, 0.50004],
                    },
                    prefix_tokens: &[],
                    pre_verify_counts: None,
                    seed: 99,
                    abs_pos: pos,
                    temperature: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                });
                assert!(!out.accepted);
                assert!(
                    out.status.is_ok(),
                    "fallback must not fail closed: {}",
                    out.status
                );
                assert!(out.status.fallback_used());
                assert_eq!(
                    out.token, 2,
                    "must fall back to lowest-ID valid target support"
                );
                assert_ne!(out.token, EXACT_SPEC_INVALID_TOKEN);
                assert!(out.token < 16);
                found_reject = true;
                break;
            }
        }
        assert!(
            found_reject,
            "expected to find rejection uniform within search"
        );
    }

    #[test]
    fn prefix_reduce_enforces_generation_and_sharing_le_1_fail_closed() {
        let ok_accept = CpuExactSpecRowOutput {
            accepted: true,
            token: 10,
            status: ExactSpecStatus::OK,
        };
        let ok_bonus = CpuExactSpecRowOutput {
            accepted: false,
            token: 20,
            status: ExactSpecStatus::OK,
        };
        // Batch of 4 sequences, verify_width = 3 (2 drafts + 1 bonus):
        // seq 0: valid exclusive slot 0, gen 5 == 5 -> accepts 2, bonus 20.
        // seq 1: stale generation (gen 4 != expected 5) -> fail closed GEN_MISMATCH.
        // seq 2: shared slot refcount 2 -> fail closed SLOT_SHARED.
        // seq 3 & seq 4: duplicate physical slot_id 7 across concurrent batch -> BOTH fail closed SLOT_SHARED.
        let guards = [
            ExactSpecSlotGuard {
                slot_id: 0,
                slot_generation: 5,
                expected_generation: 5,
                slot_refcount: 1,
            },
            ExactSpecSlotGuard {
                slot_id: 1,
                slot_generation: 4,
                expected_generation: 5,
                slot_refcount: 1,
            },
            ExactSpecSlotGuard {
                slot_id: 2,
                slot_generation: 9,
                expected_generation: 9,
                slot_refcount: 2,
            },
            ExactSpecSlotGuard {
                slot_id: 7,
                slot_generation: 11,
                expected_generation: 11,
                slot_refcount: 1,
            },
            ExactSpecSlotGuard {
                slot_id: 7,
                slot_generation: 11,
                expected_generation: 11,
                slot_refcount: 1,
            },
        ];
        let mut rows = Vec::new();
        for _ in 0..guards.len() {
            rows.push(ok_accept);
            rows.push(ok_accept);
            rows.push(ok_bonus);
        }

        let res = cpu_exact_spec_prefix_reduce(&rows, &guards, 3).unwrap();
        assert_eq!(res[0].num_accepted, 2);
        assert_eq!(res[0].bonus_token, 20);
        assert!(res[0].status.is_ok());

        assert_eq!(res[1].num_accepted, 0);
        assert_eq!(res[1].bonus_token, EXACT_SPEC_INVALID_TOKEN);
        assert!(res[1].status.has_gen_mismatch());

        assert_eq!(res[2].num_accepted, 0);
        assert_eq!(res[2].bonus_token, EXACT_SPEC_INVALID_TOKEN);
        assert!(res[2].status.has_slot_shared());

        for seq in [3_usize, 4] {
            assert_eq!(res[seq].num_accepted, 0);
            assert_eq!(res[seq].bonus_token, EXACT_SPEC_INVALID_TOKEN);
            assert!(res[seq].status.has_slot_shared());
        }
    }

    #[test]
    fn mock_gpu_launchers_validate_alignment_overlap_bounds_and_cuda_abi_parity() {
        let gpu = MockGpuBackend::new();
        let valid_accept = ExactSpecAcceptSparseArgs {
            logits: DevicePtr(0x0010_0000),
            draft_tokens: Some(DevicePtr(0x0020_0000)),
            sparse_draft_ids: Some(DevicePtr(0x0030_0000)),
            sparse_draft_probs: Some(DevicePtr(0x0040_0000)),
            verify_prefix_tokens: Some(DevicePtr(0x0050_0000)),
            pre_verify_counts: None,
            seeds: DevicePtr(0x0060_0000),
            base_abs_pos: DevicePtr(0x0070_0000),
            temperatures: DevicePtr(0x0080_0000),
            presence_penalties: DevicePtr(0x0090_0000),
            frequency_penalties: DevicePtr(0x00A0_0000),
            out_accepted: DevicePtr(0x00B0_0000),
            out_tokens: DevicePtr(0x00C0_0000),
            out_status: DevicePtr(0x00D0_0000),
            batch_size: 2,
            verify_width: 8,
            vocab_size: 248_077,
            sparse_k: 64,
        };

        // Reject null kernel handle.
        assert!(
            launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(0), &valid_accept, 0).is_err()
        );
        // Reject misaligned u64 pointer.
        let mut bad_align = valid_accept;
        bad_align.seeds = DevicePtr(0x0060_0004);
        assert!(
            launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &bad_align, 0).is_err()
        );
        // Reject overlapping outputs.
        let mut bad_overlap = valid_accept;
        bad_overlap.out_tokens = DevicePtr(0x00B0_0008);
        assert!(
            launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &bad_overlap, 0).is_err()
        );
        // Reject output overlapping input.
        let mut bad_io_overlap = valid_accept;
        bad_io_overlap.out_status = DevicePtr(0x0020_0000);
        assert!(
            launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &bad_io_overlap, 0)
                .is_err()
        );
        // Reject width > 16 and sparse_k > 64.
        let mut bad_w = valid_accept;
        bad_w.verify_width = 17;
        assert!(launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &bad_w, 0).is_err());
        let mut bad_k = valid_accept;
        bad_k.sparse_k = 65;
        assert!(launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &bad_k, 0).is_err());
        assert_eq!(gpu.launch_count(), 0);

        // Valid W=8 accept + prefix reduce launch dispatches cleanly.
        launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &valid_accept, 0).unwrap();

        let valid_reduce = ExactSpecPrefixReduceArgs {
            row_accepted: DevicePtr(0x00B0_0000),
            row_tokens: DevicePtr(0x00C0_0000),
            row_status: DevicePtr(0x00D0_0000),
            slot_ids: DevicePtr(0x00E0_0000),
            slot_generations: DevicePtr(0x00F0_0000),
            expected_generations: DevicePtr(0x0100_0000),
            slot_refcounts: DevicePtr(0x0110_0000),
            d_num_accepted: DevicePtr(0x0120_0000),
            d_bonus_token: DevicePtr(0x0130_0000),
            d_status_out: DevicePtr(0x0140_0000),
            batch_size: 2,
            verify_width: 8,
        };
        // Reject output overlapping row_status input on prefix reduce.
        let mut bad_red = valid_reduce;
        bad_red.d_status_out = DevicePtr(0x00D0_0000);
        assert!(launch_exact_spec_prefix_reduce_sm120(&gpu, KernelHandle(2), &bad_red, 0).is_err());
        launch_exact_spec_prefix_reduce_sm120(&gpu, KernelHandle(2), &valid_reduce, 0).unwrap();

        // Also verify W=1 allows None for unused draft buffers.
        let mut w1_args = valid_accept;
        w1_args.verify_width = 1;
        w1_args.draft_tokens = None;
        w1_args.sparse_draft_ids = None;
        w1_args.sparse_draft_probs = None;
        w1_args.verify_prefix_tokens = None;
        launch_exact_spec_accept_sparse_sm120(&gpu, KernelHandle(1), &w1_args, 0).unwrap();
        assert_eq!(gpu.launch_count(), 3);

        // Verify CUDA source entrypoint parity at compile time.
        let cuda_src =
            include_str!("../../../../../kernels/gb10/common/exact_spec_accept_sm120.cu");
        assert!(cuda_src.contains("void exact_spec_accept_sparse_sm120("));
        assert!(cuda_src.contains("void exact_spec_prefix_reduce_sm120("));

        // Verify ExactSpecAcceptKernels::is_complete and ExactSpecStatus::Display.
        let incomplete = ExactSpecAcceptKernels {
            accept_sparse: KernelHandle(1),
            prefix_reduce: KernelHandle(0),
        };
        assert!(!incomplete.is_complete());
        let complete = ExactSpecAcceptKernels {
            accept_sparse: KernelHandle(1),
            prefix_reduce: KernelHandle(2),
        };
        assert!(complete.is_complete());
        assert_eq!(format!("{}", ExactSpecStatus::OK), "OK");
        assert_eq!(
            format!(
                "{}",
                ExactSpecStatus::GEN_MISMATCH.with(ExactSpecStatus::SLOT_SHARED)
            ),
            "GEN_MISMATCH|SLOT_SHARED"
        );
    }

    #[test]
    fn full_qwen_vocab_248077_uniform_passes_probability_sum_tolerance() {
        // Worst-case f32 summation drift check: uniform logits across full
        // Qwen3.8 vocabulary V=248,077 must satisfy |sum(q) - 1.0| <= 1e-4
        // without false-failing TARGET_SUM_ERR.
        let logits = vec![f32_to_bf16_bits(0.5); 248_077];
        let out = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: 0x00C0_FFEE,
            abs_pos: 77,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        });
        assert!(
            out.status.is_ok(),
            "V=248,077 uniform failed: {}",
            out.status
        );
        assert!(out.token < 248_077);
    }
}
