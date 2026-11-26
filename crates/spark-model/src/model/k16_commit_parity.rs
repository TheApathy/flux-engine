// SPDX-License-Identifier: AGPL-3.0-only

//! One-shot exact committed-state receipt for native Qwen4 verification.
//!
//! K16 is reached from the legacy serial oracle. The sealed K5 probe selects
//! on the normal trajectory and runs its one-row oracle only after acceptance.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{DflashCaptureMode, LayerType};

use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::SequenceState;

pub const ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY";
pub const SEQ_LEN_ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY_SEQ_LEN";
pub const TOKENS_ENV: &str = "ATLAS_DFLASH_K16_COMMIT_PARITY_TOKENS";
pub const K5_ENV: &str = "ATLAS_DFLASH_K5_COMMIT_PARITY";
pub const K5_SEQ_LEN_ENV: &str = "ATLAS_DFLASH_K5_COMMIT_PARITY_SEQ_LEN";
pub const K5_TOKENS_ENV: &str = "ATLAS_DFLASH_K5_COMMIT_PARITY_TOKENS";

const SCHEMA: u32 = 1;
const K5: usize = 5;
const K16: usize = 16;
const K5_PRE_VERIFY_LEN: usize = 8307;
const K5_TOKENS: [u32; K5] = [248069, 2962, 11, 1179, 11];
const QWEN4_LAYERS: usize = 48;
const QWEN4_GDN_LAYERS: usize = 36;
const QWEN4_HIDDEN: usize = 2560;
const QWEN4_RESIDUAL: usize = 10240;
const QWEN38_FLASH_NEXT_PHYSICAL_VOCAB: usize = 248320;
const QWEN38_FLASH_NEXT_LOGICAL_VOCAB: usize = 248077;
const QWEN4_TAPS: [usize; 8] = [1, 7, 13, 20, 26, 33, 39, 46];

static COMPLETED: AtomicBool = AtomicBool::new(false);

pub(super) fn is_qwen38_flash_next_target_vocab(vocab_size: usize) -> bool {
    matches!(
        vocab_size,
        QWEN38_FLASH_NEXT_PHYSICAL_VOCAB | QWEN38_FLASH_NEXT_LOGICAL_VOCAB
    )
}

#[derive(Debug)]
pub struct CanonicalStateSnapshot {
    schema: u32,
    slot_idx: usize,
    gdn: Vec<CanonicalGdnState>,
    ple: CanonicalPleState,
}

/// Exact verifier-owned inputs consumed by the later production commit.  The
/// serial oracle runs between verify and commit, so it must not be allowed to
/// replace these K5 sources with its own one-row scratch contents.
#[derive(Debug)]
pub struct CommitSourceSnapshot {
    schema: u32,
    slot_idx: usize,
    k: usize,
    last_inter_slot: usize,
    gdn: Vec<CommitSourceGdnState>,
    ple_stride: usize,
    ple_intermediate: Vec<u8>,
}

#[derive(Debug)]
struct CommitSourceGdnState {
    layer_idx: usize,
    h_intermediate: Vec<u8>,
    conv_intermediate: Vec<u8>,
    lazy_retain: Option<Vec<u8>>,
}

#[derive(Debug)]
struct CanonicalGdnState {
    layer_idx: usize,
    h_bytes: usize,
    conv_bytes: usize,
    h: Vec<u8>,
    conv: Vec<u8>,
}

#[derive(Debug)]
struct CanonicalPleState {
    stride: usize,
    conv: Vec<u8>,
}

#[derive(Debug)]
struct CommittedStateSnapshot {
    schema: u32,
    slot_idx: usize,
    gdn: Vec<CommittedGdnState>,
    ple: CommittedPleState,
}

#[derive(Debug)]
struct CommittedGdnState {
    layer_idx: usize,
    h_bytes: usize,
    conv_bytes: usize,
    h_live: Vec<u8>,
    h_checkpoint: Vec<u8>,
    conv_live: Vec<u8>,
    conv_checkpoint: Vec<u8>,
}

#[derive(Debug)]
struct CommittedPleState {
    stride: usize,
    conv_live: Vec<u8>,
    conv_checkpoint: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComparisonReceipt {
    compared_buffers: usize,
    compared_bytes: usize,
    state_fnv1a64: u64,
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}

fn canonical_usize(name: &str, raw: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} must be a canonical decimal usize"))?;
    ensure!(
        value.to_string() == raw,
        "{name} must be a canonical decimal usize"
    );
    Ok(value)
}

fn canonical_u32_csv(name: &str, raw: &str) -> Result<Vec<u32>> {
    ensure!(
        !raw.is_empty()
            && !raw
                .bytes()
                .any(|byte| !(byte.is_ascii_digit() || byte == b',')),
        "{name} must be a nonempty comma-separated canonical u32 vector"
    );
    raw.split(',')
        .map(|item| {
            let value = item.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("{name} must be a nonempty comma-separated canonical u32 vector")
            })?;
            ensure!(
                value.to_string() == item,
                "{name} must be a nonempty comma-separated canonical u32 vector"
            );
            Ok(value)
        })
        .collect()
}

fn selector_values_match(
    env: &str,
    seq_len_env: &str,
    tokens_env: &str,
    width: usize,
    enabled: Option<&str>,
    seq_len: Option<&str>,
    selected_tokens: Option<&str>,
    pre_verify_len: usize,
    tokens: &[u32],
    completed: bool,
) -> Result<bool> {
    let seq_len = seq_len.filter(|value| !value.is_empty());
    let selected_tokens = selected_tokens.filter(|value| !value.is_empty());
    match enabled {
        None | Some("0") => {
            ensure!(
                seq_len.is_none() && selected_tokens.is_none(),
                "commit-parity selector variables require {env}=1"
            );
            Ok(false)
        }
        Some("1") => {
            let selected_seq_len = canonical_usize(
                seq_len_env,
                seq_len.ok_or_else(|| anyhow::anyhow!("{env}=1 requires {seq_len_env}"))?,
            )?;
            let selected_tokens = canonical_u32_csv(
                tokens_env,
                selected_tokens.ok_or_else(|| anyhow::anyhow!("{env}=1 requires {tokens_env}"))?,
            )?;
            ensure!(
                selected_tokens.len() == width,
                "{tokens_env} must contain exactly {width} inputs"
            );
            Ok(!completed && selected_seq_len == pre_verify_len && selected_tokens == tokens)
        }
        Some(_) => bail!("{env} must be exactly 0 or 1"),
    }
}

fn validate_k5_selector_seal(seq_len: Option<&str>, tokens: Option<&str>) -> Result<()> {
    ensure!(
        seq_len == Some("8307") && tokens == Some("248069,2962,11,1179,11"),
        "K5 commit parity is sealed to pre_verify_len=8307 and tokens=248069,2962,11,1179,11"
    );
    Ok(())
}

fn require_exact_off(name: &str, value: Option<&str>) -> Result<()> {
    match value {
        None | Some("0") => Ok(()),
        Some(_) => bail!("K5 commit parity requires {name} unset or exactly 0"),
    }
}

fn require_exact_on(name: &str, value: Option<&str>) -> Result<()> {
    ensure!(value == Some("1"), "K5 commit parity requires {name}=1");
    Ok(())
}

pub(super) fn requested_at(
    model: &TransformerModel,
    pre_verify_len: usize,
    tokens: &[u32],
) -> Result<bool> {
    let k16_requested = selector_values_match(
        ENV,
        SEQ_LEN_ENV,
        TOKENS_ENV,
        K16,
        optional_env(ENV)?.as_deref(),
        optional_env(SEQ_LEN_ENV)?.as_deref(),
        optional_env(TOKENS_ENV)?.as_deref(),
        pre_verify_len,
        tokens,
        COMPLETED.load(Ordering::Acquire),
    )?;
    let k5_enabled = optional_env(K5_ENV)?;
    let k5_seq_len = optional_env(K5_SEQ_LEN_ENV)?;
    let k5_tokens = optional_env(K5_TOKENS_ENV)?;
    ensure!(
        !(optional_env(ENV)?.as_deref() == Some("1") && k5_enabled.as_deref() == Some("1")),
        "K16 and K5 commit-parity selectors are mutually exclusive"
    );
    if k5_enabled.as_deref() == Some("1") {
        validate_k5_selector_seal(k5_seq_len.as_deref(), k5_tokens.as_deref())?;
        require_exact_off(
            "ATLAS_DFLASH_SERIAL_COMMIT",
            optional_env("ATLAS_DFLASH_SERIAL_COMMIT")?.as_deref(),
        )?;
        require_exact_off(
            "ATLAS_DFLASH_SKIP_REPROPOSE",
            optional_env("ATLAS_DFLASH_SKIP_REPROPOSE")?.as_deref(),
        )?;
        for name in [
            "ATLAS_QWEN4_K5_HYBRID",
            "ATLAS_QWEN4_K5_BATCH_SSM",
            "ATLAS_QWEN4_K5_BATCH_GDN",
            "ATLAS_QWEN4_K5_BATCH_HYPER",
            "ATLAS_QWEN4_K5_BATCH_ATTN_QKV",
            "ATLAS_SSM_GDN_LAZY",
        ] {
            require_exact_on(name, optional_env(name)?.as_deref())?;
        }
    }
    let k5_requested = selector_values_match(
        K5_ENV,
        K5_SEQ_LEN_ENV,
        K5_TOKENS_ENV,
        K5,
        k5_enabled.as_deref(),
        k5_seq_len.as_deref(),
        k5_tokens.as_deref(),
        pre_verify_len,
        tokens,
        COMPLETED.load(Ordering::Acquire),
    )?;
    let requested = k16_requested || k5_requested;
    if !requested {
        return Ok(false);
    }
    let width = if k5_requested { K5 } else { K16 };
    validate_exact_model(model, width)?;
    if width == K16 {
        ensure!(
            optional_env("ATLAS_QWEN4_K16_BATCHED_VERIFY")?.as_deref() == Some("1"),
            "K16 commit parity requires ATLAS_QWEN4_K16_BATCHED_VERIFY=1"
        );
        let serial_commit = optional_env("ATLAS_DFLASH_SERIAL_COMMIT")?;
        let skip_repropose = optional_env("ATLAS_DFLASH_SKIP_REPROPOSE")?;
        ensure!(
            serial_commit.as_deref() == Some("1") && skip_repropose.as_deref() == Some("1"),
            "K16 commit parity requires the serial-commit/skip-repropose oracle"
        );
    }
    Ok(true)
}

fn validate_exact_model(model: &TransformerModel, width: usize) -> Result<()> {
    ensure!(
        matches!(width, K5 | K16),
        "unsupported commit-parity width {width}"
    );
    let config = &model.config;
    ensure!(config.is_qwen4_exp(), "commit parity requires Qwen4-Exp");
    ensure!(
        config.hidden_size == QWEN4_HIDDEN
            && config.residual_width() == QWEN4_RESIDUAL
            && is_qwen38_flash_next_target_vocab(config.vocab_size)
            && config.num_hidden_layers == QWEN4_LAYERS,
        "commit parity requires exact Qwen4 H2560/R10240/V248320-or-248077/T48 geometry"
    );
    let proposer = model
        .active_proposer()
        .context("commit parity requires an active proposer")?;
    if width == K16 {
        ensure!(
            model.dflash_capture_width == QWEN4_HIDDEN
                && model.dflash_capture_offset == 0
                && model.dflash_capture_mode == DflashCaptureMode::Qwen4HyperProjected
                && model.dflash_capture_layers == QWEN4_TAPS,
            "K16 commit parity requires exact native Qwen4 hyper-projected capture"
        );
        ensure!(
            proposer.kind() == crate::speculative::DraftProposerKind::DFlash
                && proposer.is_dflash()
                && proposer.physical_verify_k() == Some(K16)
                && model.ddtree_parent_ids_capacity == K16
                && model.ssm_pool.num_intermediates == K16,
            "K16 commit parity requires exact native DFlash physical/capacity/intermediate width"
        );
    } else {
        ensure!(
            proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
                && !proposer.is_dflash()
                && proposer.physical_verify_k().is_none()
                && model.dflash_capture_layers.is_empty()
                && model.ddtree_parent_ids_capacity == 0
                && model.ssm_pool.num_intermediates == K5 + 1,
            "K5 commit parity requires the exact native-MTP proposer and six-slot flat verify pool"
        );
    }
    ensure!(
        model.qwen4_ple.is_some(),
        "commit parity requires Qwen4 PLE state"
    );
    let gdn_layers = (0..config.num_hidden_layers)
        .filter(|&layer| config.layer_type(layer) == LayerType::LinearAttention)
        .count();
    ensure!(
        gdn_layers == QWEN4_GDN_LAYERS,
        "commit parity requires exactly 36 GDN layers, got {gdn_layers}"
    );
    Ok(())
}

fn state_sizes(model: &TransformerModel) -> Result<(usize, usize)> {
    let config = &model.config;
    let h_elements = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .and_then(|value| value.checked_mul(config.linear_key_head_dim))
        .context("commit parity h-state extent overflow")?;
    let conv_dim = config
        .linear_num_key_heads
        .checked_mul(config.linear_key_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            config
                .linear_num_value_heads
                .checked_mul(config.linear_value_head_dim)
                .and_then(|tail| value.checked_add(tail))
        })
        .context("commit parity conv-state extent overflow")?;
    let h_bytes = h_elements
        .checked_mul(4)
        .context("commit parity h-state byte extent overflow")?;
    let conv_bytes = conv_dim
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(4))
        .context("commit parity conv-state byte extent overflow")?;
    ensure!(
        h_bytes > 0 && conv_bytes > 0,
        "commit parity empty GDN extent"
    );
    Ok((h_bytes, conv_bytes))
}

fn validate_sequence(model: &TransformerModel, seq: &SequenceState) -> Result<(usize, usize)> {
    let proposer = model
        .active_proposer()
        .context("commit parity requires an active proposer")?;
    let width = match proposer.kind() {
        crate::speculative::DraftProposerKind::DFlash => proposer
            .physical_verify_k()
            .context("DFlash commit parity requires an active physical verify width")?,
        crate::speculative::DraftProposerKind::Qwen4NativeMtp => K5,
        crate::speculative::DraftProposerKind::Other => {
            bail!("commit parity does not support the active proposer")
        }
    };
    validate_exact_model(model, width)?;
    ensure!(
        seq.layer_states.len() == QWEN4_LAYERS,
        "commit parity sequence has {} layer states, expected 48",
        seq.layer_states.len()
    );
    ensure!(
        seq.slot_idx < model.ssm_pool.max_slots,
        "commit parity slot {} exceeds SSM pool capacity {}",
        seq.slot_idx,
        model.ssm_pool.max_slots
    );
    state_sizes(model)
}

fn gdn_state<'a>(
    model: &TransformerModel,
    seq: &'a SequenceState,
    layer_idx: usize,
) -> Result<&'a SsmLayerState> {
    ensure!(
        model.config.layer_type(layer_idx) == LayerType::LinearAttention,
        "commit parity layer {layer_idx} is not GDN"
    );
    let state = seq.layer_states[layer_idx]
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .ok_or_else(|| {
            anyhow::anyhow!("commit parity missing SsmLayerState at layer {layer_idx}")
        })?;
    ensure!(
        !state.h_is_f16,
        "commit parity requires FP32 h-state at layer {layer_idx}"
    );
    ensure!(
        !state.h_state.is_null() && !state.conv_state.is_null(),
        "commit parity null live state at layer {layer_idx}"
    );
    Ok(state)
}

pub(super) fn capture_live(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<CanonicalStateSnapshot> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    let stream = model.gpu.default_stream();
    model
        .gpu
        .synchronize(stream)
        .context("commit parity drain live-state producer stream")?;
    let mut gdn = Vec::with_capacity(QWEN4_GDN_LAYERS);
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let state = gdn_state(model, seq, layer_idx)?;
        let mut h = vec![0u8; h_bytes];
        let mut conv = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(state.h_state, &mut h)?;
        model.gpu.copy_d2h(state.conv_state, &mut conv)?;
        gdn.push(CanonicalGdnState {
            layer_idx,
            h_bytes,
            conv_bytes,
            h,
            conv,
        });
    }
    ensure!(
        gdn.len() == QWEN4_GDN_LAYERS,
        "commit parity captured {} GDN layers, expected 36",
        gdn.len()
    );
    let ple = model
        .qwen4_ple
        .as_ref()
        .context("commit parity missing PLE")?;
    let (stride, conv, _) = ple.parity_snapshot(seq.slot_idx, model.gpu.as_ref(), stream)?;
    let snapshot = CanonicalStateSnapshot {
        schema: SCHEMA,
        slot_idx: seq.slot_idx,
        gdn,
        ple: CanonicalPleState { stride, conv },
    };
    validate_canonical(&snapshot)?;
    Ok(snapshot)
}

pub(super) fn restore_live(
    model: &TransformerModel,
    seq: &SequenceState,
    snapshot: &CanonicalStateSnapshot,
) -> Result<()> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    validate_canonical(snapshot)?;
    ensure!(
        snapshot.slot_idx == seq.slot_idx,
        "commit parity restore slot mismatch"
    );
    let mut cursor = 0usize;
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let expected = snapshot
            .gdn
            .get(cursor)
            .context("commit parity restore missing ordered GDN layer")?;
        ensure!(
            expected.layer_idx == layer_idx
                && expected.h_bytes == h_bytes
                && expected.conv_bytes == conv_bytes,
            "commit parity restore GDN order/extent mismatch at layer {layer_idx}"
        );
        let state = gdn_state(model, seq, layer_idx)?;
        model.gpu.copy_h2d(&expected.h, state.h_state)?;
        model.gpu.copy_h2d(&expected.conv, state.conv_state)?;
        cursor += 1;
    }
    ensure!(
        cursor == snapshot.gdn.len(),
        "commit parity restore trailing GDN state"
    );
    model
        .qwen4_ple
        .as_ref()
        .context("commit parity restore missing PLE")?
        .parity_restore_live(
            seq.slot_idx,
            snapshot.ple.stride,
            &snapshot.ple.conv,
            model.gpu.as_ref(),
        )?;
    Ok(())
}

fn lazy_retain_bytes(model: &TransformerModel) -> Result<usize> {
    crate::layers::qwen3_ssm::gdn_lazy_retain_lane_stride_bytes(
        model.config.ssm_qkvz_size(),
        model.config.linear_num_value_heads,
        crate::layers::qwen3_ssm::gdn_keep_late_enabled(),
    )
    .context("commit parity lazy-retain extent overflow")
}

pub(super) fn capture_commit_sources(
    model: &TransformerModel,
    seq: &SequenceState,
    k: usize,
    last_inter_slot: usize,
) -> Result<CommitSourceSnapshot> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    ensure!(
        k == K5 && last_inter_slot == 0,
        "commit-source preservation is sealed to K5 root-only slot 0"
    );
    let stream = model.gpu.default_stream();
    model
        .gpu
        .synchronize(stream)
        .context("commit parity drain wide verifier before source capture")?;
    let retain_bytes = lazy_retain_bytes(model)?;
    let mut gdn = Vec::with_capacity(QWEN4_GDN_LAYERS);
    let mut ssm_layer_idx = 0usize;
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let mut h_intermediate = vec![0u8; h_bytes];
        let mut conv_intermediate = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(
            model
                .ssm_pool
                .h_intermediate(ssm_layer_idx, seq.slot_idx, last_inter_slot),
            &mut h_intermediate,
        )?;
        model.gpu.copy_d2h(
            model
                .ssm_pool
                .conv_intermediate(ssm_layer_idx, seq.slot_idx, last_inter_slot),
            &mut conv_intermediate,
        )?;
        // The pure lazy predicate does not include the per-frame flat-tree
        // injection used by graph replay. Snapshot a retain buffer only when
        // this layer actually owns one; inter[0] is always preserved above.
        let lazy_retain = if let Some(retain) = model.layers[layer_idx].gdn_seq_lazy_retain() {
            let mut bytes = vec![0u8; retain_bytes];
            model.gpu.copy_d2h(retain, &mut bytes)?;
            Some(bytes)
        } else {
            None
        };
        gdn.push(CommitSourceGdnState {
            layer_idx,
            h_intermediate,
            conv_intermediate,
            lazy_retain,
        });
        ssm_layer_idx += 1;
    }
    ensure!(
        gdn.len() == QWEN4_GDN_LAYERS,
        "K5 commit parity source GDN census mismatch"
    );
    let ple = model
        .qwen4_ple
        .as_ref()
        .context("K5 commit parity source capture missing PLE")?;
    let (ple_stride, ple_intermediate) = ple.parity_intermediate_snapshot(
        seq.slot_idx,
        last_inter_slot,
        model.gpu.as_ref(),
        stream,
    )?;
    Ok(CommitSourceSnapshot {
        schema: SCHEMA,
        slot_idx: seq.slot_idx,
        k,
        last_inter_slot,
        gdn,
        ple_stride,
        ple_intermediate,
    })
}

pub(super) fn restore_commit_sources(
    model: &TransformerModel,
    seq: &SequenceState,
    snapshot: &CommitSourceSnapshot,
) -> Result<()> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    ensure!(
        snapshot.schema == SCHEMA
            && snapshot.slot_idx == seq.slot_idx
            && snapshot.k == K5
            && snapshot.last_inter_slot == 0
            && snapshot.gdn.len() == QWEN4_GDN_LAYERS,
        "K5 commit parity source snapshot identity mismatch"
    );
    let retain_bytes = lazy_retain_bytes(model)?;
    let mut ssm_layer_idx = 0usize;
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let source = snapshot
            .gdn
            .get(ssm_layer_idx)
            .context("K5 commit parity source restore missing GDN layer")?;
        ensure!(
            source.layer_idx == layer_idx
                && source.h_intermediate.len() == h_bytes
                && source.conv_intermediate.len() == conv_bytes,
            "K5 commit parity source restore order/extent mismatch at layer {layer_idx}"
        );
        model.gpu.copy_h2d(
            &source.h_intermediate,
            model
                .ssm_pool
                .h_intermediate(ssm_layer_idx, seq.slot_idx, snapshot.last_inter_slot),
        )?;
        model.gpu.copy_h2d(
            &source.conv_intermediate,
            model
                .ssm_pool
                .conv_intermediate(ssm_layer_idx, seq.slot_idx, snapshot.last_inter_slot),
        )?;
        if let Some(bytes) = source.lazy_retain.as_deref() {
            ensure!(
                bytes.len() == retain_bytes,
                "K5 commit parity lazy-retain extent mismatch at layer {layer_idx}"
            );
            let retain = model.layers[layer_idx]
                .gdn_seq_lazy_retain()
                .with_context(|| {
                    format!("K5 commit parity missing lazy retain at layer {layer_idx}")
                })?;
            model.gpu.copy_h2d(bytes, retain)?;
        }
        ssm_layer_idx += 1;
    }
    model
        .qwen4_ple
        .as_ref()
        .context("K5 commit parity source restore missing PLE")?
        .parity_restore_intermediate(
            seq.slot_idx,
            snapshot.last_inter_slot,
            snapshot.ple_stride,
            &snapshot.ple_intermediate,
            model.gpu.as_ref(),
        )?;
    Ok(())
}

pub(super) fn drain_default(model: &TransformerModel) -> Result<()> {
    model
        .gpu
        .synchronize(model.gpu.default_stream())
        .context("commit parity drain restored default-stream state")
}

fn capture_committed(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<CommittedStateSnapshot> {
    let (h_bytes, conv_bytes) = validate_sequence(model, seq)?;
    let default_stream = model.gpu.default_stream();
    ensure!(
        model.secondary_event != 0,
        "commit parity missing secondary event"
    );
    ensure!(
        model.secondary_stream != default_stream,
        "commit parity secondary/default stream identity collision"
    );
    model
        .gpu
        .stream_wait_event(default_stream, model.secondary_event)
        .context("commit parity post-commit event wait enqueue")?;
    model
        .gpu
        .synchronize(default_stream)
        .context("commit parity post-commit event wait completion")?;

    let mut gdn = Vec::with_capacity(QWEN4_GDN_LAYERS);
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let state = gdn_state(model, seq, layer_idx)?;
        let h_checkpoint = state
            .h_state_checkpoint
            .filter(|ptr| !ptr.is_null())
            .with_context(|| format!("commit parity missing h checkpoint at layer {layer_idx}"))?;
        let conv_checkpoint = state
            .conv_state_checkpoint
            .filter(|ptr| !ptr.is_null())
            .with_context(|| {
                format!("commit parity missing conv checkpoint at layer {layer_idx}")
            })?;
        let mut h_live = vec![0u8; h_bytes];
        let mut h_checkpoint_bytes = vec![0u8; h_bytes];
        let mut conv_live = vec![0u8; conv_bytes];
        let mut conv_checkpoint_bytes = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(state.h_state, &mut h_live)?;
        model.gpu.copy_d2h(h_checkpoint, &mut h_checkpoint_bytes)?;
        model.gpu.copy_d2h(state.conv_state, &mut conv_live)?;
        model
            .gpu
            .copy_d2h(conv_checkpoint, &mut conv_checkpoint_bytes)?;
        gdn.push(CommittedGdnState {
            layer_idx,
            h_bytes,
            conv_bytes,
            h_live,
            h_checkpoint: h_checkpoint_bytes,
            conv_live,
            conv_checkpoint: conv_checkpoint_bytes,
        });
    }
    let ple = model
        .qwen4_ple
        .as_ref()
        .context("commit parity missing PLE")?;
    let (stride, conv_live, conv_checkpoint) =
        ple.parity_snapshot(seq.slot_idx, model.gpu.as_ref(), default_stream)?;
    Ok(CommittedStateSnapshot {
        schema: SCHEMA,
        slot_idx: seq.slot_idx,
        gdn,
        ple: CommittedPleState {
            stride,
            conv_live,
            conv_checkpoint,
        },
    })
}

fn validate_canonical(snapshot: &CanonicalStateSnapshot) -> Result<()> {
    ensure!(
        snapshot.schema == SCHEMA,
        "commit parity canonical schema mismatch"
    );
    ensure!(
        snapshot.gdn.len() == QWEN4_GDN_LAYERS,
        "commit parity canonical GDN census mismatch"
    );
    let mut previous = None;
    for state in &snapshot.gdn {
        ensure!(
            previous.is_none_or(|layer| state.layer_idx > layer),
            "commit parity canonical GDN layer order mismatch"
        );
        ensure!(
            state.h_bytes > 0
                && state.conv_bytes > 0
                && state.h.len() == state.h_bytes
                && state.conv.len() == state.conv_bytes,
            "commit parity canonical GDN extent mismatch at layer {}",
            state.layer_idx
        );
        previous = Some(state.layer_idx);
    }
    ensure!(
        snapshot.ple.stride > 0 && snapshot.ple.conv.len() == snapshot.ple.stride,
        "commit parity canonical PLE stride mismatch"
    );
    Ok(())
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn first_mismatch(expected: &[u8], actual: &[u8]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())))
}

fn compare_buffer(label: &str, expected: &[u8], actual: &[u8]) -> Result<()> {
    if let Some(byte) = first_mismatch(expected, actual) {
        let expected_hash = fnv1a64_update(0xcbf29ce484222325, expected);
        let actual_hash = fnv1a64_update(0xcbf29ce484222325, actual);
        bail!(
            "commit parity byte mismatch at {label} byte={byte} expected_len={} actual_len={} expected_fnv1a64={expected_hash:016x} actual_fnv1a64={actual_hash:016x}",
            expected.len(),
            actual.len()
        );
    }
    Ok(())
}

fn compare_snapshots(
    expected: &CanonicalStateSnapshot,
    actual: &CommittedStateSnapshot,
) -> Result<ComparisonReceipt> {
    validate_canonical(expected)?;
    ensure!(
        actual.schema == SCHEMA,
        "commit parity committed schema mismatch"
    );
    ensure!(
        expected.slot_idx == actual.slot_idx,
        "commit parity committed slot mismatch"
    );
    ensure!(
        actual.gdn.len() == QWEN4_GDN_LAYERS,
        "commit parity committed GDN census mismatch"
    );
    let mut compared_buffers = 0usize;
    let mut compared_bytes = 0usize;
    let mut hash = 0xcbf29ce484222325;
    for (expected_state, actual_state) in expected.gdn.iter().zip(&actual.gdn) {
        ensure!(
            expected_state.layer_idx == actual_state.layer_idx,
            "commit parity committed GDN order mismatch: expected layer {}, got {}",
            expected_state.layer_idx,
            actual_state.layer_idx
        );
        ensure!(
            expected_state.h_bytes == actual_state.h_bytes
                && expected_state.conv_bytes == actual_state.conv_bytes,
            "commit parity committed GDN extent metadata mismatch at layer {}",
            expected_state.layer_idx
        );
        for (name, expected_bytes, actual_bytes) in [
            (
                "h_live",
                expected_state.h.as_slice(),
                actual_state.h_live.as_slice(),
            ),
            (
                "h_checkpoint",
                expected_state.h.as_slice(),
                actual_state.h_checkpoint.as_slice(),
            ),
            (
                "conv_live",
                expected_state.conv.as_slice(),
                actual_state.conv_live.as_slice(),
            ),
            (
                "conv_checkpoint",
                expected_state.conv.as_slice(),
                actual_state.conv_checkpoint.as_slice(),
            ),
        ] {
            compare_buffer(
                &format!("gdn.layer_{:02}.{name}", expected_state.layer_idx),
                expected_bytes,
                actual_bytes,
            )?;
            compared_buffers += 1;
            compared_bytes = compared_bytes
                .checked_add(expected_bytes.len())
                .context("commit parity compared-byte overflow")?;
        }
        hash = fnv1a64_update(hash, &(expected_state.layer_idx as u64).to_le_bytes());
        hash = fnv1a64_update(hash, &expected_state.h);
        hash = fnv1a64_update(hash, &expected_state.conv);
    }
    ensure!(
        expected.ple.stride == actual.ple.stride,
        "commit parity committed PLE stride mismatch"
    );
    for (name, actual_bytes) in [
        ("ple.conv_live", actual.ple.conv_live.as_slice()),
        ("ple.conv_checkpoint", actual.ple.conv_checkpoint.as_slice()),
    ] {
        compare_buffer(name, &expected.ple.conv, actual_bytes)?;
        compared_buffers += 1;
        compared_bytes = compared_bytes
            .checked_add(expected.ple.conv.len())
            .context("commit parity compared-byte overflow")?;
    }
    hash = fnv1a64_update(hash, &expected.ple.conv);
    ensure!(
        compared_buffers == QWEN4_GDN_LAYERS * 4 + 2,
        "commit parity compared-buffer census mismatch"
    );
    Ok(ComparisonReceipt {
        compared_buffers,
        compared_bytes,
        state_fnv1a64: hash,
    })
}

fn validate_frame(total_accepted: usize, k: usize, last_inter_slot: usize) -> Result<()> {
    if k == K5 {
        ensure!(
            total_accepted == 1 && last_inter_slot == 0,
            "K5 commit parity is sealed to root-only total_accepted=1/last_inter_slot=0"
        );
        return Ok(());
    }
    ensure!(k == K16, "commit parity expected k=5 or k=16, got {k}");
    ensure!(
        (1..=k).contains(&total_accepted),
        "K16 commit parity total_accepted must be in 1..=16, got {total_accepted}"
    );
    ensure!(
        last_inter_slot == total_accepted - 1,
        "K16 commit parity requires flat row order: last_inter_slot={last_inter_slot}, total_accepted={total_accepted}"
    );
    Ok(())
}

fn canonical_tokens(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn compare_committed(
    model: &TransformerModel,
    seq: &SequenceState,
    expected: &CanonicalStateSnapshot,
    pre_verify_len: usize,
    tokens: &[u32],
    total_accepted: usize,
    k: usize,
    last_inter_slot: usize,
) -> Result<()> {
    ensure!(
        tokens.len() == k,
        "commit parity frame token census mismatch"
    );
    validate_frame(total_accepted, k, last_inter_slot)?;
    if k == K5 {
        ensure!(
            pre_verify_len == K5_PRE_VERIFY_LEN && tokens == K5_TOKENS,
            "K5 commit parity reached an unsealed frame"
        );
    }
    let actual = capture_committed(model, seq)?;
    let receipt = match compare_snapshots(expected, &actual) {
        Ok(receipt) => receipt,
        Err(error) if k == K5 => {
            tracing::error!(
                "DFLASH_K5_COMMIT_PARITY_FIRST_DIVERGENCE schema={} route=native_qwen4_mtp_k5_root_only scope=recurrent_gdn_conv_ple qsa_kv_compared=false pre_verify_len={} tokens={} total_committed_inputs={} k={} last_inter_slot={} production_sources_restored=true match=false error={:#}",
                SCHEMA,
                pre_verify_len,
                canonical_tokens(tokens),
                total_accepted,
                k,
                last_inter_slot,
                error,
            );
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    COMPLETED.store(true, Ordering::Release);
    if k == K5 {
        tracing::info!(
            "DFLASH_K5_COMMIT_PARITY schema={} route=native_qwen4_mtp_k5_root_only scope=recurrent_gdn_conv_ple qsa_kv_compared=false oracle=post_accept_k1 pre_verify_len={} tokens={} total_committed_inputs={} k={} last_inter_slot={} slot={} gdn_layers={} compared_buffers={} compared_bytes={} state_fnv1a64={:016x} production_sources_restored=true match=true",
            SCHEMA,
            pre_verify_len,
            canonical_tokens(tokens),
            total_accepted,
            k,
            last_inter_slot,
            seq.slot_idx,
            QWEN4_GDN_LAYERS,
            receipt.compared_buffers,
            receipt.compared_bytes,
            receipt.state_fnv1a64,
        );
    } else {
        tracing::info!(
            "DFLASH_K16_COMMIT_PARITY schema={} route=native_qwen4_dflash_physical_k16_batched pre_verify_len={} tokens={} total_accepted={} k={} last_inter_slot={} slot={} gdn_layers={} compared_buffers={} compared_bytes={} state_fnv1a64={:016x} match=true",
            SCHEMA,
            pre_verify_len,
            canonical_tokens(tokens),
            total_accepted,
            k,
            last_inter_slot,
            seq.slot_idx,
            QWEN4_GDN_LAYERS,
            receipt.compared_buffers,
            receipt.compared_bytes,
            receipt.state_fnv1a64,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_next_target_vocab_accepts_only_physical_or_canonical_logical() {
        for admitted in [248320, 248077] {
            assert!(is_qwen38_flash_next_target_vocab(admitted));
        }
        for rejected in [0, 248076, 248078, 248319, 248321, usize::MAX] {
            assert!(
                !is_qwen38_flash_next_target_vocab(rejected),
                "hostile target vocabulary unexpectedly admitted: {rejected}"
            );
        }
    }

    fn canonical() -> CanonicalStateSnapshot {
        CanonicalStateSnapshot {
            schema: SCHEMA,
            slot_idx: 3,
            gdn: (0..QWEN4_GDN_LAYERS)
                .map(|layer_idx| CanonicalGdnState {
                    layer_idx,
                    h_bytes: 2,
                    conv_bytes: 3,
                    h: vec![layer_idx as u8, 1],
                    conv: vec![layer_idx as u8, 2, 3],
                })
                .collect(),
            ple: CanonicalPleState {
                stride: 4,
                conv: vec![4, 5, 6, 7],
            },
        }
    }

    fn committed(expected: &CanonicalStateSnapshot) -> CommittedStateSnapshot {
        CommittedStateSnapshot {
            schema: SCHEMA,
            slot_idx: expected.slot_idx,
            gdn: expected
                .gdn
                .iter()
                .map(|state| CommittedGdnState {
                    layer_idx: state.layer_idx,
                    h_bytes: state.h_bytes,
                    conv_bytes: state.conv_bytes,
                    h_live: state.h.clone(),
                    h_checkpoint: state.h.clone(),
                    conv_live: state.conv.clone(),
                    conv_checkpoint: state.conv.clone(),
                })
                .collect(),
            ple: CommittedPleState {
                stride: expected.ple.stride,
                conv_live: expected.ple.conv.clone(),
                conv_checkpoint: expected.ple.conv.clone(),
            },
        }
    }

    #[test]
    fn selector_is_exact_default_off_and_one_shot() {
        let tokens: Vec<u32> = (1..=16).collect();
        let select = |enabled, seq_len, selected, pre, completed| {
            selector_values_match(
                ENV,
                SEQ_LEN_ENV,
                TOKENS_ENV,
                K16,
                enabled,
                seq_len,
                selected,
                pre,
                &tokens,
                completed,
            )
        };
        assert!(!select(None, None, None, 7, false).unwrap());
        assert!(!select(Some("0"), None, None, 7, false).unwrap());
        assert!(select(Some("0"), Some("7"), None, 7, false).is_err());
        assert!(select(Some("2"), Some("7"), Some("1"), 7, false).is_err());
        assert!(select(Some("1"), None, Some("1"), 7, false).is_err());
        assert!(select(Some("1"), Some("07"), Some("1"), 7, false).is_err());
        assert!(select(Some("1"), Some("7"), Some("01"), 7, false).is_err());
        let csv = canonical_tokens(&tokens);
        assert!(select(Some("1"), Some("7"), Some(&csv), 7, false).unwrap());
        assert!(!select(Some("1"), Some("7"), Some(&csv), 8, false).unwrap());
        assert!(!select(Some("1"), Some("7"), Some(&csv), 7, true).unwrap());
    }

    #[test]
    fn k5_selector_is_exact_and_one_shot() {
        for value in [None, Some("0")] {
            require_exact_off("CONTROL", value).unwrap();
        }
        for value in [Some(""), Some("1"), Some("false"), Some("2")] {
            assert!(require_exact_off("CONTROL", value).is_err());
        }
        require_exact_on("ROUTE", Some("1")).unwrap();
        for value in [None, Some(""), Some("0"), Some("2")] {
            assert!(require_exact_on("ROUTE", value).is_err());
        }
        validate_k5_selector_seal(Some("8307"), Some("248069,2962,11,1179,11")).unwrap();
        for (seq_len, tokens) in [
            (None, Some("248069,2962,11,1179,11")),
            (Some("8308"), Some("248069,2962,11,1179,11")),
            (Some("8307"), None),
            (Some("8307"), Some("248069,2962,11,1179,12")),
        ] {
            assert!(validate_k5_selector_seal(seq_len, tokens).is_err());
        }
        let csv = canonical_tokens(&K5_TOKENS);
        assert!(
            selector_values_match(
                K5_ENV,
                K5_SEQ_LEN_ENV,
                K5_TOKENS_ENV,
                K5,
                Some("1"),
                Some("8307"),
                Some(&csv),
                K5_PRE_VERIFY_LEN,
                &K5_TOKENS,
                false,
            )
            .unwrap()
        );
        for hostile_width in [0, 1, 4, 6, 16] {
            let hostile = vec![1u32; hostile_width];
            assert!(
                selector_values_match(
                    K5_ENV,
                    K5_SEQ_LEN_ENV,
                    K5_TOKENS_ENV,
                    K5,
                    Some("1"),
                    Some("8307"),
                    Some(&canonical_tokens(&hostile)),
                    K5_PRE_VERIFY_LEN,
                    &hostile,
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn exact_match_counts_all_live_and_checkpoint_buffers() {
        let expected = canonical();
        let actual = committed(&expected);
        let receipt = compare_snapshots(&expected, &actual).unwrap();
        assert_eq!(receipt.compared_buffers, 146);
        assert_eq!(
            receipt.compared_bytes,
            QWEN4_GDN_LAYERS * (2 * 2 + 3 * 2) + 4 * 2
        );
        assert_ne!(receipt.state_fnv1a64, 0);
    }

    #[test]
    fn rejects_missing_reordered_or_shape_drifted_state() {
        let expected = canonical();
        let mut missing = committed(&expected);
        missing.gdn.pop();
        assert!(compare_snapshots(&expected, &missing).is_err());

        let mut reordered = committed(&expected);
        reordered.gdn.swap(0, 1);
        assert!(compare_snapshots(&expected, &reordered).is_err());

        let mut extent = committed(&expected);
        extent.gdn[0].h_bytes += 1;
        assert!(compare_snapshots(&expected, &extent).is_err());

        let mut ple = committed(&expected);
        ple.ple.stride += 1;
        assert!(compare_snapshots(&expected, &ple).is_err());
    }

    #[test]
    fn rejects_any_live_or_checkpoint_byte_drift() {
        let expected = canonical();
        for mutate in 0..6 {
            let mut actual = committed(&expected);
            match mutate {
                0 => actual.gdn[0].h_live[0] ^= 1,
                1 => actual.gdn[0].h_checkpoint[0] ^= 1,
                2 => actual.gdn[0].conv_live[0] ^= 1,
                3 => actual.gdn[0].conv_checkpoint[0] ^= 1,
                4 => actual.ple.conv_live[0] ^= 1,
                _ => actual.ple.conv_checkpoint[0] ^= 1,
            }
            assert!(compare_snapshots(&expected, &actual).is_err());
        }
    }

    #[test]
    fn frame_contract_rejects_wrong_width_bounds_or_order() {
        validate_frame(1, 5, 0).unwrap();
        assert!(validate_frame(2, 5, 1).is_err());
        assert!(validate_frame(1, 5, 1).is_err());
        validate_frame(1, 16, 0).unwrap();
        validate_frame(16, 16, 15).unwrap();
        assert!(validate_frame(0, 16, 0).is_err());
        assert!(validate_frame(17, 16, 16).is_err());
        assert!(validate_frame(8, 15, 7).is_err());
        assert!(validate_frame(8, 16, 8).is_err());
    }

    #[test]
    fn static_contract_records_commit_event_after_every_state_write() {
        let source = include_str!("trait_impl/async_chkpt.rs");
        let commit = source
            .split_once("fn commit_verify_state_async_dispatch(")
            .unwrap()
            .1;
        let ple_write = commit
            .rfind("ple.rollback_and_checkpoint")
            .expect("commit must write PLE live/checkpoint state");
        let event = commit
            .rfind("record_event(self.secondary_event, stream)")
            .expect("commit must record its completion event");
        assert!(
            ple_write < event,
            "commit event must follow all recurrent writes"
        );
    }
}
