// SPDX-License-Identifier: AGPL-3.0-only

//! Policy gates, confidence thresholds, rotary/Gumbel metadata layout, and ragged compaction planning for Qwen4 MTP.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layers::qwen4_mtp_gate::{self, GatePolicy};
use crate::layers::qwen4_mtp_rows::{self, AfterVerifyPlan, RowPolicy};

/// Opt-in draft block truncation (`ATLAS_QWEN4_MTP_BOUNDED_BLOCKS=1`).
/// Defaults to 0 / off so full state.block_table is preserved across long contexts.
pub fn mtp_bounded_blocks_selected() -> bool {
    static SELECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SELECTED.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_BOUNDED_BLOCKS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(false)
    })
}

/// Qwen4 MTP sparse-attention (QSA) mode for `mtp.layers.0`.
///
/// Controlled by `ATLAS_QWEN4_MTP_QSA` (`1` enables; unset/`0` -> `Disabled`)
/// and `ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE` (default `1` when
/// `ATLAS_QWEN4_MTP_QSA=1`; `0` -> `PerStep`). Invalid values fail closed to
/// `Disabled`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpQsaMode {
    /// Default: `qwen4_qsa_required: false` (dense/legacy MTP path).
    Disabled,
    /// Full QSA indexer update + selection on every committed and draft step.
    PerStep,
    /// Full QSA selection on step 0 (snapshotted into `reused_draft_indices`),
    /// then draft steps `1..num_drafts` skip index/score/select and attend over
    /// `OUTPUT_WIDTH + step` indices with appended speculative draft positions.
    DraftSelectionReuse,
}

pub const MTP_QSA_SELECTOR: &str = "ATLAS_QWEN4_MTP_QSA";
pub const MTP_DRAFT_SELECTION_REUSE_SELECTOR: &str = "ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE";

/// Strict parser for `ATLAS_QWEN4_MTP_QSA` + `ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE`.
pub fn parse_mtp_qsa_mode(
    mtp_qsa: Option<&str>,
    draft_selection_reuse: Option<&str>,
) -> Result<MtpQsaMode> {
    let qsa_enabled = match mtp_qsa.map(str::trim) {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => {
            anyhow::bail!("invalid {MTP_QSA_SELECTOR}={other:?}: expected \"0\" or \"1\"");
        }
    };
    if !qsa_enabled {
        return Ok(MtpQsaMode::Disabled);
    }
    match draft_selection_reuse.map(str::trim) {
        None | Some("1") => Ok(MtpQsaMode::DraftSelectionReuse),
        Some("0") => Ok(MtpQsaMode::PerStep),
        Some(other) => {
            anyhow::bail!(
                "invalid {MTP_DRAFT_SELECTION_REUSE_SELECTOR}={other:?}: expected \"0\" or \"1\""
            );
        }
    }
}

/// Fail-closed resolver: returns `MtpQsaMode::Disabled` if either selector is invalid.
pub fn resolve_mtp_qsa_mode(
    mtp_qsa: Option<&str>,
    draft_selection_reuse: Option<&str>,
) -> MtpQsaMode {
    parse_mtp_qsa_mode(mtp_qsa, draft_selection_reuse).unwrap_or(MtpQsaMode::Disabled)
}

/// Process-wide cached `MtpQsaMode` read from environment variables.
pub fn mtp_qsa_mode() -> MtpQsaMode {
    static MODE: std::sync::OnceLock<MtpQsaMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let qsa_raw = std::env::var(MTP_QSA_SELECTOR).ok();
        let reuse_raw = std::env::var(MTP_DRAFT_SELECTION_REUSE_SELECTOR).ok();
        match parse_mtp_qsa_mode(qsa_raw.as_deref(), reuse_raw.as_deref()) {
            Ok(mode) => {
                if mode != MtpQsaMode::Disabled {
                    tracing::info!(
                        "QWEN4_MTP_QSA_ENGAGED mode={mode:?} \
                         {MTP_QSA_SELECTOR}={qsa_raw:?} \
                         {MTP_DRAFT_SELECTION_REUSE_SELECTOR}={reuse_raw:?}"
                    );
                }
                mode
            }
            Err(err) => {
                tracing::warn!(
                    "QWEN4_MTP_QSA_REFUSED reason=invalid_selector ({err}); failing closed to Disabled"
                );
                MtpQsaMode::Disabled
            }
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DraftStepReuseAction {
    SnapshotStep0,
    ReuseStep {
        base_seq_len: usize,
        step_offset: usize,
        max_drafts: usize,
    },
}

/// Minimum sequence length at which Qwen4 switches from dense fallback to QSA.
pub(super) const MTP_QSA_SPARSE_THRESHOLD: usize = 2048;

/// Decide `(qwen4_qsa_required, reuse_action)` for a single MTP draft step.
pub(super) fn draft_step_qsa_plan(
    mode: MtpQsaMode,
    has_qsa_side_cache: bool,
    first_seq_len: usize,
    step: usize,
    num_drafts: usize,
) -> (bool, Option<DraftStepReuseAction>) {
    if !has_qsa_side_cache || mode == MtpQsaMode::Disabled {
        return (false, None);
    }
    if mode == MtpQsaMode::PerStep {
        return (true, None);
    }
    let max_drafts = num_drafts.saturating_sub(1);
    let reuse_eligible = first_seq_len >= MTP_QSA_SPARSE_THRESHOLD
        && (1..=crate::layers::qwen4_qsa::QSA_REUSE_MAX_DRAFTS).contains(&max_drafts);
    if step == 0 {
        (
            true,
            reuse_eligible.then_some(DraftStepReuseAction::SnapshotStep0),
        )
    } else if reuse_eligible && step <= max_drafts {
        (
            true,
            Some(DraftStepReuseAction::ReuseStep {
                base_seq_len: first_seq_len,
                step_offset: step,
                max_drafts,
            }),
        )
    } else if first_seq_len.saturating_add(num_drafts) <= MTP_QSA_SPARSE_THRESHOLD {
        // Pure dense prefix (< 2048): skip QSA side-cache writes on speculative drafts.
        (false, None)
    } else {
        // 2048-straddling boundary chain or wider-than-reuse chain: PerStep fallback.
        (true, None)
    }
}

/// Checked `OUTPUT_WIDTH + step` index width for an MTP draft step under
/// `DraftSelectionReuse`.
pub fn checked_mtp_reused_index_width(step: usize, max_drafts: usize) -> Result<usize> {
    crate::layers::qwen4_qsa::checked_reused_index_width(step, max_drafts)
}

/// Parse `ATLAS_QWEN4_MTP_LM_HEAD_TC`: opt-in (`"1"`) only.
///
/// `w4a16_gemm_pipe` issues strided 16-byte global loads across `N`, saturating
/// memory bandwidth at ~53 GB/s on GB10 (~2.0-3.5 ms per draft step at `b=16`).
/// Defaulting off routes `propose_batch` LM head through coalesced single-sweep
/// exact GEMV (`w4a16_gemv_batch_logits_exact_with`, ~225 GB/s, 0-ULP exact).
pub(super) fn parse_mtp_lm_head_tc_pipe_env(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

pub(super) fn mtp_lm_head_tc_pipe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_mtp_lm_head_tc_pipe_env(std::env::var("ATLAS_QWEN4_MTP_LM_HEAD_TC").ok().as_deref())
    })
}

#[inline]
pub(super) fn should_use_mtp_lm_head_pipe(
    enabled: bool,
    batch_rows: usize,
    hidden: usize,
    pipe_available: bool,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> bool {
    enabled
        && batch_rows >= 8
        && pipe_available
        && hidden.is_multiple_of(32)
        && input.0.is_multiple_of(16)
        && weight.0.is_multiple_of(16)
        && output.0.is_multiple_of(16)
}

#[inline]
pub(super) const fn should_use_mtp_lm_head_coalesced(
    tc_active: bool,
    batch_rows: usize,
    hidden: usize,
) -> bool {
    tc_active
        && batch_rows >= 6
        && batch_rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && hidden > 0
        && hidden.is_multiple_of(32)
}

/// Maximum `batch_rows * hc_count` admitted to coalesced NVFP4 MTP FC
/// (`batch_rows <= 64` at `hc_count == 4`), executed via
/// `grid.y = div_ceil(hc_rows, tile_m)` on `w4a16_gemm_coalesced*`.
pub(super) const MTP_FC_COALESCED_MAX_HC_ROWS: usize = 256;

#[inline]
#[must_use]
pub(super) const fn should_use_batched_argmax_prob(rows: usize, batched_available: bool) -> bool {
    rows > 1 && batched_available
}

#[inline]
pub(super) const fn should_use_mtp_fc_coalesced(
    tc_dense_scope: bool,
    batch_rows: usize,
    hc_count: usize,
    hidden: usize,
    qkv_output_bytes: usize,
    broadcast_add_available: bool,
) -> bool {
    if !tc_dense_scope
        || batch_rows < 6
        || hc_count != 4
        || hidden == 0
        || !hidden.is_multiple_of(32)
        || !broadcast_add_available
    {
        return false;
    }
    let Some(hc_rows) = batch_rows.checked_mul(hc_count) else {
        return false;
    };
    if hc_rows > MTP_FC_COALESCED_MAX_HC_ROWS {
        return false;
    }
    let Some(embed_elems) = batch_rows.checked_mul(hidden) else {
        return false;
    };
    let Some(embed_bytes) = embed_elems.checked_mul(2) else {
        return false;
    };
    embed_bytes <= qkv_output_bytes
}

/// Environment selector for batched Native MTP proposer CUDA-graph capture
/// (`"0"` / `"false"` / `"off"` disables; `"1"` / `"true"` / `"on"` forces on;
/// unset / `"auto"` auto-enables on `active_sm_count() > 64` while keeping GB10
/// `<= 64` SMs default OFF).
pub const MTP_GRAPH_SELECTOR: &str = "ATLAS_QWEN4_MTP_GRAPH";

/// Minimum SM count for auto-enabling batched MTP CUDA-graph capture.
pub(super) const MTP_GRAPH_AUTO_SM_THRESHOLD: u32 = 64;

/// Maximum cached CUDA graphs per [`Qwen4MtpHead`] before FIFO eviction.
pub(super) const MTP_GRAPH_CACHE_CAP: usize = 32;

/// Minimum bucketed block-table stride (`max_blocks_per_seq`) for graphed MTP
/// launches so growing sequences hit one cached graph across block boundaries.
pub(super) const MTP_GRAPH_MIN_BT_BUCKET: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MtpGraphPolicy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
pub(super) fn parse_mtp_graph_policy(raw: Option<&str>) -> MtpGraphPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") => {
            MtpGraphPolicy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on") => {
            MtpGraphPolicy::Enabled
        }
        _ => MtpGraphPolicy::Auto,
    }
}

#[must_use]
pub(super) const fn resolve_mtp_graph_enabled(policy: MtpGraphPolicy, sm_count: u32) -> bool {
    match policy {
        MtpGraphPolicy::Disabled => false,
        MtpGraphPolicy::Enabled => true,
        MtpGraphPolicy::Auto => sm_count > MTP_GRAPH_AUTO_SM_THRESHOLD,
    }
}

#[must_use]
pub(super) fn mtp_graph_enabled() -> bool {
    static POLICY: std::sync::OnceLock<MtpGraphPolicy> = std::sync::OnceLock::new();
    let policy = *POLICY.get_or_init(|| {
        let raw = std::env::var(MTP_GRAPH_SELECTOR).ok();
        parse_mtp_graph_policy(raw.as_deref())
    });
    resolve_mtp_graph_enabled(policy, atlas_core::device::active_sm_count())
}

/// Bucket `max_mb` up to a power of two (`>= 16`) when the padded per-step
/// metadata slab still fits inside `scratch_bytes`; otherwise return `max_mb`.
#[must_use]
pub(super) fn mtp_graph_bt_blocks_cap(
    max_mb: usize,
    batch_size: usize,
    step_count: usize,
    bt_off: usize,
    scratch_bytes: usize,
) -> usize {
    let bucket = max_mb.max(MTP_GRAPH_MIN_BT_BUCKET).next_power_of_two();
    let fits = bucket
        .checked_mul(4)
        .and_then(|row_bytes| batch_size.checked_mul(row_bytes))
        .and_then(|bt_bytes| bt_off.checked_add(bt_bytes))
        .and_then(|raw_stride| raw_stride.checked_next_multiple_of(256))
        .and_then(|stride| step_count.checked_mul(stride))
        .and_then(|meta_bytes| CHAIN_META_OFFSET.checked_add(meta_bytes))
        .is_some_and(|end| end <= scratch_bytes);
    if fits { bucket } else { max_mb }
}

/// Scratch layout of the sync-free draft chain. The single-token MoE router
/// writes the first bytes of scratch and the per-step attention metadata
/// starts at `CHAIN_META_OFFSET` (the same base the one-step path uses). That
/// base is the target's `STEP_META_SCRATCH_OFFSET`: every draft step and chain
/// uploads its metadata before launching, and the target re-uploads its own
/// before each decode/verify frame, so neither may read the other's.
pub(super) const CHAIN_ARGMAX_OFFSET: usize = 16384;
pub(super) const CHAIN_TOKEN_OFFSET: usize = CHAIN_ARGMAX_OFFSET + 1024;
/// Draft probabilities live in the upper half of the 1 KiB argmax span.
pub(super) const CHAIN_PROB_OFFSET: usize = CHAIN_ARGMAX_OFFSET + 512;
pub(super) const CHAIN_PROB_SPAN: usize = 512;
pub(super) const CHAIN_META_OFFSET: usize = crate::layer::STEP_META_SCRATCH_OFFSET;
pub(super) const POSITION_H_OFFSET: usize = 20;
pub(super) const POSITION_W_OFFSET: usize = 24;
pub(super) const MAX_ROTARY_METADATA_ROWS: usize = 1_048_576;

/// Where one MTP step takes its inputs from and leaves its draft.
#[derive(Clone, Copy)]
pub(super) struct ChainStep {
    /// Pre-uploaded attention metadata for this step.
    pub(super) meta: DevicePtr,
    /// Number of block-table entries uploaded in `meta`.
    pub(super) max_blocks_per_seq: u32,
    /// Device u32 holding the previous step's argmax (None: host token).
    pub(super) device_token: Option<DevicePtr>,
    /// Device u32 receiving this step's argmax; no host readback.
    pub(super) argmax_out: DevicePtr,
    /// Device f32 receiving the argmax's softmax probability (draft confidence).
    pub(super) prob_out: Option<DevicePtr>,
    /// Optional CRN Gumbel perturbation `(spec, abs_pos)` for this step.
    pub(super) gumbel: Option<(spark_runtime::sampler::gumbel::GumbelDraftSpec, u64)>,
}

/// Number of drafts kept under the active gate policy (`0..=probs.len()`).
pub(super) fn confident_prefix(probs: &[f32], threshold: f32) -> usize {
    qwen4_mtp_gate::confident_prefix_env(probs, threshold)
}

/// Number of drafts kept for a batched (`b >= 2`) cohort-uniform chain
/// (`1..=probs.len()` for non-empty `probs`, never ejecting a sequence to `0`).
#[cfg(test)]
pub(super) fn confident_prefix_batch(probs: &[f32], threshold: f32) -> usize {
    qwen4_mtp_gate::confident_prefix_batch_env(probs, threshold)
}

/// Confidence-gating plan for one MTP draft chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ChainConfidenceGate {
    /// Whether to compute and copy back per-step raw top-1 softmax probabilities.
    pub(super) want_prob: bool,
    /// Optional mid-chain early-stop threshold (`Some` triggers a per-step D2H sync).
    pub(super) early: Option<f32>,
    /// Whether to truncate the finished draft prefix by raw top-1 probability.
    pub(super) apply_truncate: bool,
}

/// Resize and zero-fill a reusable host metadata staging buffer to `len` bytes
/// without releasing its existing capacity across draft steps.
pub(super) fn prepare_packed_meta_scratch(buf: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buf.len() < len {
        buf.resize(len, 0);
    }
    let slice = &mut buf[..len];
    slice.fill(0);
    slice
}

/// Whether confidence gating runs for a single sequence (`b == 1`) or a
/// cohort-uniform batched proposal (`b >= 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChainConfidenceScope {
    /// Single-sequence chain (`b == 1`): honors `ATLAS_QWEN4_MTP_CONF`
    /// early-stop syncs and post-chain prefix truncation under greedy drafting.
    SingleSequence,
    /// Cross-sequence batched cohort (`b >= 2`): never applies per-sequence
    /// `MTP_CONF` early-stop syncs or prefix truncation so every sequence in
    /// the batch preserves one uniform verify width `K`.
    BatchedCohort,
}

/// Resolve the confidence-gating plan for a draft chain under `scope`.
///
/// - Under [`ChainConfidenceScope::BatchedCohort`] (`b >= 2`), per-sequence
///   early-stop and truncation are always disabled so `C=1` can retain full
///   `MTP_CONF` early-stop speed on `C>1` servers without shattering `C>=2`
///   cohorts into ragged widths or disabling `argmax_bf16_rows`.
/// - Under non-greedy Gumbel-CRN (`gumbel_active == true`) on `SingleSequence`,
///   per-step greedy floors (`MTP_CONF` / `MTP_CONF_STEPS`) are replaced by a
///   `0.0` per-step floor plus running-product `MTP_CONF_CUM` over true
///   post-temperature `p_T(chosen_tok)` (default `0.38`; `ATLAS_QWEN4_MTP_CONF_CUM=0`
///   disables gating).
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_chain_confidence_gate_with_cum(
    scope: ChainConfidenceScope,
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
    gumbel_cum: Option<f32>,
    early_explicitly_disabled: bool,
) -> ChainConfidenceGate {
    if matches!(scope, ChainConfidenceScope::BatchedCohort) {
        return ChainConfidenceGate {
            want_prob: policy.log && argmax_prob_available,
            early: None,
            apply_truncate: false,
        };
    }
    if gumbel_active {
        let cum_active = gumbel_cum.is_some();
        let early_enabled = cum_active && (policy.early_stop || !early_explicitly_disabled);
        return ChainConfidenceGate {
            want_prob: policy.log || cum_active,
            early: early_enabled.then_some(0.0),
            apply_truncate: cum_active,
        };
    }
    let want_prob = policy_wants_prob && argmax_prob_available;
    let early = policy_early.filter(|_| want_prob);
    ChainConfidenceGate {
        want_prob,
        early,
        apply_truncate: want_prob,
    }
}

pub(super) fn resolve_chain_confidence_gate_for_scope(
    scope: ChainConfidenceScope,
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
) -> ChainConfidenceGate {
    resolve_chain_confidence_gate_with_cum(
        scope,
        policy,
        policy_wants_prob,
        policy_early,
        argmax_prob_available,
        gumbel_active,
        qwen4_mtp_gate::gumbel_cum_threshold(),
        qwen4_mtp_gate::early_stop_explicitly_disabled(),
    )
}

/// Resolve the confidence-gating plan for a single-sequence (`b == 1`) draft chain.
pub(super) fn resolve_chain_confidence_gate(
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
) -> ChainConfidenceGate {
    resolve_chain_confidence_gate_for_scope(
        ChainConfidenceScope::SingleSequence,
        policy,
        policy_wants_prob,
        policy_early,
        argmax_prob_available,
        gumbel_active,
    )
}

/// Packed device metadata offsets (relative to `meta_base`) for a homogeneous
/// batched Gumbel-CRN launch: `[seeds: u64; b]` followed by step-major
/// `[abs_positions: u64; num_drafts * b]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchGumbelMetaLayout {
    pub(super) seeds_off: usize,
    pub(super) pos_base_off: usize,
    pub(super) total_bytes: usize,
}

impl BatchGumbelMetaLayout {
    pub(super) fn compute(base_off: usize, b: usize, num_drafts: usize) -> Option<Self> {
        let seeds_off = base_off.checked_add(7)? & !7;
        let seeds_bytes = b.checked_mul(8)?;
        let pos_base_off = seeds_off.checked_add(seeds_bytes)?;
        let pos_bytes = num_drafts.checked_mul(b)?.checked_mul(8)?;
        let end = pos_base_off.checked_add(pos_bytes)?;
        let raw_span = end.checked_sub(base_off)?;
        let total_bytes = raw_span.checked_add(255)? & !255;
        Some(Self {
            seeds_off,
            pos_base_off,
            total_bytes,
        })
    }

    #[inline]
    pub(super) fn step_pos_off(self, step: usize, b: usize) -> usize {
        self.pos_base_off + step * b * 8
    }
}

pub(super) fn prompt_tail_start(base: usize, count: usize) -> Result<usize> {
    let end = base
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP prompt position overflow"))?;
    ensure!(
        count <= end,
        "Qwen4 MTP prompt tail exceeds physical history"
    );
    end.checked_sub(count)
        .and_then(|start| start.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP predict-into position overflow"))
}

pub(super) fn rotary_axes(
    positions: &crate::traits::RotaryPositions,
    physical: usize,
    mrope_interleaved: bool,
    mrope_kernel_available: bool,
    yarn_factor: f32,
) -> Result<[u32; 3]> {
    if !positions.is_identity() {
        ensure!(
            mrope_interleaved,
            "Qwen4 native MTP image positions require MRoPE"
        );
        ensure!(
            mrope_kernel_available,
            "Qwen4 native MTP MRoPE kernel is unavailable"
        );
        ensure!(
            yarn_factor == 0.0,
            "Qwen4 native MTP image YaRN positions require separate numerical qualification"
        );
    }
    positions.position(physical)
}

pub(super) fn validate_rotary_span(
    positions: &crate::traits::RotaryPositions,
    start: usize,
    count: usize,
    mrope_interleaved: bool,
    mrope_kernel_available: bool,
    yarn_factor: f32,
) -> Result<()> {
    ensure!(
        count <= MAX_ROTARY_METADATA_ROWS,
        "Qwen4 MTP rotary span exceeds metadata bound"
    );
    if count == 0 {
        return Ok(());
    }
    let last = start
        .checked_add(count - 1)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP rotary span overflow"))?;
    rotary_axes(
        positions,
        start,
        mrope_interleaved,
        mrope_kernel_available,
        yarn_factor,
    )?;
    rotary_axes(
        positions,
        last,
        mrope_interleaved,
        mrope_kernel_available,
        yarn_factor,
    )?;
    Ok(())
}

pub(super) fn write_rotary_metadata(
    positions: &crate::traits::RotaryPositions,
    physical: usize,
    packed: &mut [u8],
) -> Result<()> {
    ensure!(
        packed.len() >= POSITION_W_OFFSET + 4,
        "Qwen4 MTP rotary metadata header is truncated"
    );
    let [t, h, w] = positions.position(physical)?;
    packed[0..4].copy_from_slice(&t.to_le_bytes());
    if !positions.is_identity() {
        packed[POSITION_H_OFFSET..POSITION_H_OFFSET + 4].copy_from_slice(&h.to_le_bytes());
        packed[POSITION_W_OFFSET..POSITION_W_OFFSET + 4].copy_from_slice(&w.to_le_bytes());
    }
    Ok(())
}

pub(super) fn rotary_metadata_ptrs(
    meta_base: DevicePtr,
    positions: &crate::traits::RotaryPositions,
) -> (DevicePtr, DevicePtr) {
    if positions.is_identity() {
        (meta_base, meta_base)
    } else {
        (
            meta_base.offset(POSITION_H_OFFSET),
            meta_base.offset(POSITION_W_OFFSET),
        )
    }
}

pub const MTP_STEP0_DUAL_GEMV_SELECTOR: &str = "ATLAS_QWEN4_MTP_STEP0_DUAL_GEMV";

pub fn mtp_step0_dual_gemv_selected() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let on = std::env::var(MTP_STEP0_DUAL_GEMV_SELECTOR).ok().as_deref() != Some("0");
        if on {
            static STEP0_LOGGED: std::sync::Once = std::sync::Once::new();
            STEP0_LOGGED.call_once(|| {
                tracing::info!(
                    "MTP_STEP0_OPTIM_ENGAGED dual_gemv=fc_hidden+fc_embedding hidden=2560"
                );
            });
        }
        on
    })
}

pub const MTP_DEVICE_PROPOSE_SELECTOR: &str = "ATLAS_QWEN4_MTP_DEVICE_PROPOSE";
pub const MTP_CHAIN_ALIAS: &str = "ATLAS_QWEN4_MTP_CHAIN";

pub fn mtp_device_propose_selected() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let dev = std::env::var(MTP_DEVICE_PROPOSE_SELECTOR).ok();
        let chain = std::env::var(MTP_CHAIN_ALIAS).ok();
        let on = dev.as_deref() != Some("0") && chain.as_deref() != Some("0");
        if on {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!("mtp_propose: MTP_DEVICE_PROPOSE_CHAIN_ENGAGED chained_steps=1..3 host_roundtrips=0");
            });
        }
        on
    })
}

/// Enabled by default; disabled when either `ATLAS_QWEN4_MTP_DEVICE_PROPOSE=0` or `ATLAS_QWEN4_MTP_CHAIN=0` is set.
pub(super) fn mtp_chain_enabled() -> bool {
    mtp_device_propose_selected()
}

/// True when batched multi-row MTP execution (`forward_rows`) is enabled for
/// `ATLAS_QWEN4_MTP_RECONDITION` + `draft[0]` and `prefill_last_k`. Enabled by
/// default whenever `ATLAS_QWEN4_MTP_RECONDITION` or `ATLAS_QWEN4_VERIFY_FAST_HOST=1`
/// or `ATLAS_QWEN4_MTP_BATCH_REPLAY=1` is active unless explicitly disabled via
/// `ATLAS_QWEN4_MTP_BATCH_REPLAY=0`.
pub(super) fn mtp_batch_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("ATLAS_QWEN4_MTP_BATCH_REPLAY").ok();
        if env.as_deref() == Some("0") {
            return false;
        }
        let on = env.as_deref() == Some("1")
            || crate::layers::qwen4_verify_fast_host_selected()
            || qwen4_mtp_rows::recondition_engaged();
        if on {
            tracing::info!("QWEN4_MTP_BATCH_ROWS_ENGAGED selector=ATLAS_QWEN4_MTP_BATCH_REPLAY");
        }
        on
    })
}

/// True when KV-only MTP replay (`decode_kv_only` / `decode_qwen4_batched_mtp`)
/// is enabled for non-emitting replay rows. Default on unless
/// `ATLAS_QWEN4_MTP_KV_ONLY_REPLAY=0`.
pub(super) fn mtp_kv_only_replay_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_KV_ONLY_REPLAY")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Selector enabling per-sequence running-product `p_cum` early stop and
/// active-lane compaction inside cross-sequence batched MTP (`propose_batch`).
/// Default ON when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` unless explicitly set to `0`.
pub const BATCHED_RAGGED_MTP_ENV: &str = "ATLAS_QWEN4_BATCHED_RAGGED_MTP";

/// Cumulative probability threshold (`p_cum = prod p_i`) below which a sequence
/// stops drafting and is compacted out of subsequent batched MTP steps.
/// Default `0.30` (aligned with `C=1`'s `0.38` Gumbel cumulative gate while
/// preserving deeper drafts on high-confidence prose/code lanes); `0` disables early drop.
pub const MTP_RAGGED_PCUM_THRESH_ENV: &str = "ATLAS_QWEN4_MTP_RAGGED_PCUM_THRESH";
pub const DEFAULT_MTP_RAGGED_PCUM_THRESH: f32 = 0.30;

#[inline]
#[must_use]
pub(super) fn parse_batched_ragged_mtp(raw: Option<&str>, adaptive_k: bool) -> bool {
    if !adaptive_k {
        return false;
    }
    !matches!(
        raw.map(str::trim),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")
    )
}

/// True when ragged mixed-`K` batched verify and active-subset compacted MTP
/// drafting are enabled (default ON when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`;
/// set `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0` to restore uniform cohort widths).
pub fn batched_ragged_mtp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let adaptive = crate::layers::qwen4_verify_lanes::adaptive_k_selected().unwrap_or(false);
        let raw = std::env::var(BATCHED_RAGGED_MTP_ENV).ok();
        let on = parse_batched_ragged_mtp(raw.as_deref(), adaptive);
        if on {
            tracing::info!(
                selector = BATCHED_RAGGED_MTP_ENV,
                pcum_thresh = mtp_ragged_pcum_thresh(),
                "QWEN4_BATCHED_RAGGED_MTP_ENGAGED"
            );
        }
        on
    })
}

#[inline]
#[must_use]
pub(super) fn parse_mtp_ragged_pcum_thresh(raw: Option<&str>) -> f32 {
    let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_MTP_RAGGED_PCUM_THRESH;
    };
    match trimmed.parse::<f32>() {
        Ok(v) if v.is_finite() && (0.0..1.0).contains(&v) => v,
        _ => DEFAULT_MTP_RAGGED_PCUM_THRESH,
    }
}

/// Cumulative confidence floor (`ATLAS_QWEN4_MTP_RAGGED_PCUM_THRESH`, default `0.30`).
pub fn mtp_ragged_pcum_thresh() -> f32 {
    static THRESH: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *THRESH.get_or_init(|| {
        parse_mtp_ragged_pcum_thresh(std::env::var(MTP_RAGGED_PCUM_THRESH_ENV).ok().as_deref())
    })
}

thread_local! {
    static RAGGED_MTP_TARGET_DRAFTS: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Optional per-sequence draft-depth ceiling for `propose_batch` under ragged
/// compaction (`target_drafts[s] == 0` excludes lane `s` from step 0).
pub struct RaggedMtpTargetDraftsGuard {
    prev: Option<Vec<usize>>,
}

impl RaggedMtpTargetDraftsGuard {
    #[must_use]
    pub fn enter(target_drafts: &[usize]) -> Self {
        let prev =
            RAGGED_MTP_TARGET_DRAFTS.with(|cell| cell.borrow_mut().replace(target_drafts.to_vec()));
        Self { prev }
    }
}

impl Drop for RaggedMtpTargetDraftsGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        RAGGED_MTP_TARGET_DRAFTS.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

pub(super) fn current_ragged_mtp_target_drafts() -> Option<Vec<usize>> {
    RAGGED_MTP_TARGET_DRAFTS.with(|cell| cell.borrow().clone())
}

/// Whether `propose_chain` (C=1) performs mid-chain D2H probability syncs so
/// `MTP_CONF` stops drafting as soon as token probability drops below threshold,
/// honoring explicit `ATLAS_QWEN4_MTP_CONF_EARLY_SYNC=0|1`.
#[inline]
#[must_use]
pub(super) fn should_sync_mid_chain(raw: Option<&str>, _num_drafts: usize) -> bool {
    match raw.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => true,
    }
}

/// Keep draft token when cumulative confidence meets `pcum_thresh` (or early stop is disabled).
#[inline]
#[must_use]
pub(super) fn ragged_mtp_keep_step(p_cum: f32, pcum_thresh: f32) -> bool {
    pcum_thresh <= 0.0 || p_cum >= pcum_thresh
}

/// Whether `propose_batch_ragged_compacted` can chain all `0..max_steps`
/// strictly on GPU (`embed_from_argmax_k` reading device-resident `argmax`
/// directly, ONE pre-loop H2D metadata upload, and ONE post-loop batched D2H
/// readback) with 0-ULP bit-identical results.
///
/// Requires:
/// - `pcum_thresh <= 0.0` (no lane can drop out early on host between steps),
/// - `max_steps > 1` (single-step chains already perform only one final D2H),
/// - every active lane targets the exact same `max_steps`,
/// - `[max_steps, active_lanes.len()]` `(token, prob)` spans and pre-packed
///   per-step metadata fit within the scratch arena.
#[inline]
#[must_use]
pub(super) fn can_gpu_chain_ragged_steps(
    pcum_thresh: f32,
    active_lanes: &[usize],
    target_drafts: &[usize],
    max_steps: usize,
    argmax_span_bytes: usize,
    prob_span_bytes: usize,
    meta_end_bytes: usize,
    scratch_bytes: usize,
) -> bool {
    if pcum_thresh > 0.0 || max_steps <= 1 || active_lanes.is_empty() {
        return false;
    }
    if !active_lanes
        .iter()
        .all(|&orig_s| target_drafts.get(orig_s).copied() == Some(max_steps))
    {
        return false;
    }
    let Some(total_step_bytes) = max_steps
        .checked_mul(active_lanes.len())
        .and_then(|n| n.checked_mul(4))
    else {
        return false;
    };
    total_step_bytes <= argmax_span_bytes
        && total_step_bytes <= prob_span_bytes
        && meta_end_bytes <= scratch_bytes
}

/// Whether batched ragged MTP may emit `0` drafts (`K_s = 1`) on a hopeless
/// step-0 lane (`ATLAS_QWEN4_MTP_ALLOW_ZERO=1`). Default `false` keeps the
/// already-computed step-0 draft (`K_s = 2`) while still dropping the lane
/// from steps `1..`, avoiding ejecting the lane into a separate Phase-A
/// bootstrap pass in `mtp_step.rs` and preserving `min_k >= 2` KEEP_LATE
/// + CUDA graph capture/replay.
pub(super) fn ragged_mtp_allow_zero_drafts() -> bool {
    static ALLOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOW.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_ALLOW_ZERO")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("1")
    })
}

/// Plan strictly non-overlapping left-to-right row copies `(src_row, dst_row, count)`
/// that compact `prev_active` rows into `next_active` rows in-place.
///
/// Every emitted tuple satisfies `dst_row + count <= src_row` so
/// `cudaMemcpyAsync` (device-to-device) never operates on overlapping spans.
pub(super) fn plan_hidden_compaction_copies(
    prev_active: &[usize],
    next_active: &[usize],
) -> Vec<(usize, usize, usize)> {
    let mut old_indices = Vec::with_capacity(next_active.len());
    let mut prev_pos = 0usize;
    for &lane in next_active {
        while prev_pos < prev_active.len() && prev_active[prev_pos] != lane {
            prev_pos += 1;
        }
        if prev_pos < prev_active.len() {
            old_indices.push(prev_pos);
            prev_pos += 1;
        }
    }
    let mut copies = Vec::new();
    let mut new_i = 0usize;
    while new_i < old_indices.len() {
        let old_i = old_indices[new_i];
        if old_i == new_i {
            new_i += 1;
            continue;
        }
        let shift = old_i - new_i;
        let mut run = 1usize;
        while new_i + run < old_indices.len()
            && old_indices[new_i + run] == old_i + run
            && run < shift
        {
            run += 1;
        }
        copies.push((old_i, new_i, run));
        new_i += run;
    }
    copies
}

/// Advance one step of active-subset ragged MTP compaction on host after reading
/// back `[b_act]` `(token, prob)` pairs. Updates `steps_run`, `computed_drafts`,
/// `emitted_drafts`, `p_cum`, and `current_tokens`, returning the surviving
/// `next_active_lanes` for `step + 1`.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance_ragged_mtp_step(
    step: usize,
    max_steps: usize,
    pcum_thresh: f32,
    allow_zero: bool,
    active_lanes: &[usize],
    step_tokens: &[u32],
    step_probs: &[f32],
    target_drafts: &[usize],
    p_cum: &mut [f32],
    steps_run: &mut [usize],
    computed_drafts: &mut [Vec<u32>],
    emitted_drafts: &mut [Vec<u32>],
    current_tokens: &mut [u32],
    mut computed_probs: Option<&mut [Vec<f32>]>,
) -> Vec<usize> {
    let mut next_active = Vec::with_capacity(active_lanes.len());
    for (local_i, &orig_s) in active_lanes.iter().enumerate() {
        let tok = step_tokens[local_i];
        let prob = step_probs[local_i];
        steps_run[orig_s] += 1;
        computed_drafts[orig_s].push(tok);
        if let Some(probs_log) = computed_probs.as_deref_mut() {
            probs_log[orig_s].push(prob);
        }
        p_cum[orig_s] *= prob;
        let meets_thresh = ragged_mtp_keep_step(p_cum[orig_s], pcum_thresh);
        if meets_thresh {
            emitted_drafts[orig_s].push(tok);
            current_tokens[orig_s] = tok;
            if emitted_drafts[orig_s].len() < target_drafts[orig_s] && step + 1 < max_steps {
                next_active.push(orig_s);
            }
        } else if step == 0 && !allow_zero {
            // Keep the already-computed step-0 draft (`K_s = 2`) so the sequence
            // stays inside the batched verify cohort (no extra bootstrap forward)
            // while still stopping immediately after step 0 (zero work on steps 1..).
            emitted_drafts[orig_s].push(tok);
        }
    }
    next_active
}

/// Graph-bucket row granularity for ragged batched-verify frames of `b`
/// sequences so `BatchedVerifyGraphKey = (leader_slots, total_rows)` collapses
/// from ~45 distinct row counts per cohort down to ~3-5 cached CUDA graphs.
#[inline]
#[must_use]
pub(super) const fn ragged_row_bucket_step(b: usize) -> usize {
    match b {
        0..=1 => 1,
        2 => 2,
        3..=5 => 4,
        6..=16 => 8,
        _ => 16,
    }
}

/// Stop ragged MTP drafting early at `C >= 4` once `step >= 2` (at least 3
/// steps run) and at most 25% of lanes survive, avoiding dragging an entire
/// wide cohort through extra MTP passes and D2H syncs for 1-4 stragglers.
#[inline]
#[must_use]
pub(super) const fn ragged_mtp_should_stop_stragglers(
    cohort_b: usize,
    step: usize,
    next_active_len: usize,
) -> bool {
    cohort_b >= 4
        && step >= 2
        && next_active_len > 0
        && next_active_len.saturating_mul(4) <= cohort_b
}

/// Quantize ragged `emitted_drafts` total verify rows (`sum(drafts_s + 1)`)
/// down to a multiple of [`ragged_row_bucket_step`] by trimming the lowest
/// tail-`p_cum` draft(s) while preserving at least `min_drafts` per lane
/// (`1` when `!allow_zero`, so every lane retains `K_s >= 2`).
pub(super) fn quantize_ragged_drafts_to_graph_bucket(
    emitted_drafts: &mut [Vec<u32>],
    emitted_pcum: &mut [Vec<f32>],
    allow_zero: bool,
) {
    let b = emitted_drafts.len();
    let step = ragged_row_bucket_step(b);
    if step <= 1 || b != emitted_pcum.len() {
        return;
    }
    let min_drafts = usize::from(!allow_zero);
    let total_rows: usize = emitted_drafts.iter().map(|d| d.len() + 1).sum();
    let mut rem = total_rows % step;
    while rem > 0 {
        let mut victim: Option<(usize, f32, usize)> = None;
        for s in 0..b {
            let len = emitted_drafts[s].len();
            if len <= min_drafts {
                continue;
            }
            let tail_p = emitted_pcum[s].last().copied().unwrap_or(1.0);
            let better = match victim {
                None => true,
                Some((best_s, best_p, best_len)) => {
                    tail_p < best_p
                        || (tail_p == best_p && (len > best_len || (len == best_len && s > best_s)))
                }
            };
            if better {
                victim = Some((s, tail_p, len));
            }
        }
        let Some((s, _, _)) = victim else {
            break;
        };
        emitted_drafts[s].pop();
        emitted_pcum[s].pop();
        rem -= 1;
    }
}

/// True when cross-sequence batched MTP draft proposals (`propose_batch`) are
/// enabled for C2..C8 cohorts. Default on unless `ATLAS_QWEN4_MTP_CROSS_SEQ=0`.
pub(super) fn mtp_cross_seq_propose_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_MTP_CROSS_SEQ").ok().as_deref() != Some("0");
        if on {
            tracing::info!("QWEN4_MTP_CROSS_SEQ_ENGAGED selector=ATLAS_QWEN4_MTP_CROSS_SEQ");
        }
        on
    })
}

/// The K7 head-row policy (`qwen4_mtp_rows`), its receipt logged once.
pub(super) fn rows_policy_logged() -> RowPolicy {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let policy = qwen4_mtp_rows::policy();
    LOGGED.call_once(|| {
        if let Some(err) = qwen4_mtp_rows::policy_error() {
            tracing::warn!("QWEN4_MTP_ROWS_REFUSED reason=bad_selector ({err}); legacy head rows");
            return;
        }
        match policy {
            RowPolicy::Legacy => {}
            RowPolicy::KeepLast => tracing::info!(
                "QWEN4_MTP_KEEP_LAST_ROW_ENGAGED selector={}: the last committed row stays in the \
                 head cache",
                qwen4_mtp_rows::KEEP_LAST_ROW_SELECTOR
            ),
            RowPolicy::Recondition { max_rows } => {
                if qwen4_mtp_rows::recondition_engaged() {
                    tracing::info!(
                        "QWEN4_MTP_RECONDITION_REQUESTED selector={} max_rows={max_rows} \
                         staged_rows={}",
                        qwen4_mtp_rows::RECONDITION_SELECTOR,
                        qwen4_mtp_rows::staged_rows_capacity()
                    );
                } else {
                    tracing::warn!(
                        "QWEN4_MTP_RECONDITION_REFUSED reason=no_staging_buffer staged_rows={} \
                         need={}; KEEP_LAST_ROW semantics",
                        qwen4_mtp_rows::staged_rows_capacity(),
                        qwen4_mtp_rows::MAX_STAGED_ROWS
                    );
                }
            }
        }
    });
    policy
}

/// Running counts for the K7 head-row levers (frames planned, frames whose
/// last committed row is now present, rewritten rows, frames still leaving a
/// hole), logged at 64, 128, 256, ... frames.
pub(super) fn record_row_stats(computed: usize, accepted: usize, plan: AfterVerifyPlan) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static FILLED: AtomicU64 = AtomicU64::new(0);
    static REPLAYED: AtomicU64 = AtomicU64::new(0);
    static HOLES: AtomicU64 = AtomicU64::new(0);
    if computed == 0 {
        return;
    }
    let accepted = accepted.min(computed);
    let covered = plan.keep_rows + plan.replay_rows;
    let frames = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if covered > accepted {
        FILLED.fetch_add(1, Ordering::Relaxed);
    }
    if covered < accepted + 1 {
        HOLES.fetch_add(1, Ordering::Relaxed);
    }
    REPLAYED.fetch_add(plan.replay_rows as u64, Ordering::Relaxed);
    if frames >= 64 && frames.is_power_of_two() {
        tracing::info!(
            "QWEN4_MTP_ROWS_STATS frames={frames} filled={} replayed_rows={} holes_left={}",
            FILLED.load(Ordering::Relaxed),
            REPLAYED.load(Ordering::Relaxed),
            HOLES.load(Ordering::Relaxed),
        );
    }
}
