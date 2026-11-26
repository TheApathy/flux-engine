// SPDX-License-Identifier: AGPL-3.0-only

//! Qualification-only exact target-state receipt for Qwen3.8 Flash Next.
//!
//! The digest is deliberately independent of proposer state, physical KV block
//! IDs, and the sequence slot.  It covers the canonical target-fed tokens,
//! every live GDN recurrent buffer, live PLE history, and every logically
//! committed BF16 K/V row.  A loaded-serial process and an engaged native-MTP
//! process can therefore compare the same semantic target state across fresh
//! processes even though their allocator identities differ.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context as _, Result, bail, ensure};
use atlas_core::config::LayerType;
use ring::digest::{Context, SHA256};
use spark_runtime::kv_cache::KvCacheDtype;

use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::speculative::DraftProposerKind;
use crate::traits::{Model, SequenceState};

pub const ENV: &str = "ATLAS_QWEN4_TARGET_STATE_RECEIPT";
pub const LIMIT_ENV: &str = "ATLAS_QWEN4_TARGET_STATE_RECEIPT_LIMIT";

const SCHEMA: u32 = 1;
const QWEN4_LAYERS: usize = 48;
const QWEN4_GDN_LAYERS: usize = 36;
const QWEN4_ATTN_LAYERS: usize = QWEN4_LAYERS - QWEN4_GDN_LAYERS;
const QWEN4_HIDDEN: usize = 2560;
const QWEN4_RESIDUAL: usize = 10240;

static CAPTURES: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4TargetStateReceipt {
    pub schema: u32,
    pub mode: &'static str,
    pub scheduler_path: &'static str,
    pub proposer_loaded: bool,
    pub seq_len: usize,
    pub token_count: usize,
    pub gdn_layers: usize,
    pub attention_layers: usize,
    pub logical_kv_blocks: usize,
    pub kv_token_positions: usize,
    pub compared_buffers: usize,
    pub compared_bytes: usize,
    pub fed_tokens_sha256: String,
    pub target_state_sha256: String,
}

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}

fn binary_switch(name: &str) -> Result<bool> {
    match optional_env(name)?.as_deref() {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{name} must be exactly 0 or 1"),
    }
}

fn capture_limit() -> Result<usize> {
    let Some(raw) = optional_env(LIMIT_ENV)? else {
        return Ok(usize::MAX);
    };
    let limit = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{LIMIT_ENV} must be a canonical positive usize"))?;
    ensure!(
        limit > 0 && limit.to_string() == raw,
        "{LIMIT_ENV} must be a canonical positive usize"
    );
    Ok(limit)
}

fn hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn update_field(context: &mut Context, label: &[u8], bytes: &[u8]) -> Result<()> {
    let label_len = u64::try_from(label.len()).context("state-receipt label length overflow")?;
    let byte_len = u64::try_from(bytes.len()).context("state-receipt buffer length overflow")?;
    context.update(&label_len.to_le_bytes());
    context.update(label);
    context.update(&byte_len.to_le_bytes());
    context.update(bytes);
    Ok(())
}

fn update_usize(context: &mut Context, label: &[u8], value: usize) -> Result<()> {
    update_field(
        context,
        label,
        &u64::try_from(value)
            .context("state-receipt usize overflow")?
            .to_le_bytes(),
    )
}

fn token_digest(tokens: &[u32]) -> Result<String> {
    let mut context = Context::new(&SHA256);
    update_field(&mut context, b"domain", b"atlas-qwen4-fed-tokens-v1")?;
    update_usize(&mut context, b"count", tokens.len())?;
    for token in tokens {
        context.update(&token.to_le_bytes());
    }
    Ok(hex(context.finish().as_ref()))
}

fn classify_capture_mode(
    proposer_kind: Option<DraftProposerKind>,
    force_serial: bool,
) -> Result<(&'static str, &'static str, bool)> {
    Ok(match (proposer_kind, force_serial) {
        // The immutable kill switch changes dispatch, not proposer ownership.
        // With no proposer, either setting is the same unloaded target path.
        (None, _) => ("unloaded-target-only", "ordinary-target-only", false),
        (Some(DraftProposerKind::Qwen4NativeMtp), true) => {
            ("loaded-serial", "ordinary-target-only", true)
        }
        (Some(DraftProposerKind::Qwen4NativeMtp), false) => {
            ("engaged-native-mtp", "native-mtp", true)
        }
        (Some(kind), _) => bail!("target-state receipt rejects proposer kind {kind:?}"),
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn validate_target(model: &TransformerModel, seq: &SequenceState) -> Result<()> {
    let config = &model.config;
    ensure!(
        config.is_qwen4_exp(),
        "target-state receipt requires Qwen4-Exp"
    );
    ensure!(
        config.hidden_size == QWEN4_HIDDEN
            && config.residual_width() == QWEN4_RESIDUAL
            && super::k16_commit_parity::is_qwen38_flash_next_target_vocab(config.vocab_size)
            && config.num_hidden_layers == QWEN4_LAYERS,
        "target-state receipt requires exact Qwen3.8 Flash Next geometry"
    );
    ensure!(
        seq.seq_len > 0 && seq.tokens.len() == seq.seq_len,
        "target-state receipt requires exact target-fed token/length ownership"
    );
    ensure!(
        seq.layer_states.len() == QWEN4_LAYERS,
        "target-state receipt layer-state census drifted"
    );
    ensure!(
        seq.slot_idx < model.ssm_pool.max_slots,
        "target-state receipt SSM slot is out of range"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn state_sizes(model: &TransformerModel) -> Result<(usize, usize)> {
    let config = &model.config;
    let h_bytes = config
        .linear_num_value_heads
        .checked_mul(config.linear_value_head_dim)
        .and_then(|value| value.checked_mul(config.linear_key_head_dim))
        .and_then(|value| value.checked_mul(4))
        .context("target-state receipt h-state extent overflow")?;
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
        .context("target-state receipt conv-state extent overflow")?;
    let conv_bytes = conv_dim
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(4))
        .context("target-state receipt conv-state byte extent overflow")?;
    ensure!(
        h_bytes > 0 && conv_bytes > 0,
        "target-state receipt empty GDN extent"
    );
    Ok((h_bytes, conv_bytes))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn add_buffer(
    context: &mut Context,
    label: &[u8],
    bytes: &[u8],
    compared_buffers: &mut usize,
    compared_bytes: &mut usize,
) -> Result<()> {
    update_field(context, label, bytes)?;
    *compared_buffers = compared_buffers
        .checked_add(1)
        .context("target-state receipt buffer-count overflow")?;
    *compared_bytes = compared_bytes
        .checked_add(bytes.len())
        .context("target-state receipt byte-count overflow")?;
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn capture_enabled(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<Qwen4TargetStateReceipt> {
    validate_target(model, seq)?;
    let force_serial = model.speculation_forced_disabled;
    let proposer_kind = model.active_proposer().map(|proposer| proposer.kind());
    let (mode, scheduler_path, proposer_loaded) =
        classify_capture_mode(proposer_kind, force_serial)?;

    // The secondary-stream event is the ownership boundary for speculative
    // recurrent-state commits. Enqueue its wait only for the engaged arm: an
    // unloaded or loaded-serial process has not recorded that event and owns
    // all target writes on the default stream.
    if mode == "engaged-native-mtp" {
        model
            .sync_secondary()
            .context("target-state receipt order speculative commit")?;
    }
    let stream = model.gpu.default_stream();
    model
        .gpu
        .synchronize(stream)
        .context("target-state receipt drain target producer stream")?;

    let mut context = Context::new(&SHA256);
    update_field(&mut context, b"domain", b"atlas-qwen4-target-state-v1")?;
    update_usize(&mut context, b"seq_len", seq.seq_len)?;
    update_usize(&mut context, b"hidden", model.config.hidden_size)?;
    update_usize(&mut context, b"residual", model.config.residual_width())?;
    update_usize(&mut context, b"layers", model.config.num_hidden_layers)?;
    for token in &seq.tokens {
        context.update(&token.to_le_bytes());
    }

    let mut compared_buffers = 0usize;
    let mut compared_bytes = 0usize;
    let (h_bytes, conv_bytes) = state_sizes(model)?;
    let mut gdn_layers = 0usize;
    for layer_idx in 0..QWEN4_LAYERS {
        if model.config.layer_type(layer_idx) != LayerType::LinearAttention {
            continue;
        }
        let state = seq.layer_states[layer_idx]
            .as_any()
            .downcast_ref::<SsmLayerState>()
            .with_context(|| {
                format!("target-state receipt missing GDN state at layer {layer_idx}")
            })?;
        ensure!(
            !state.h_is_f16,
            "target-state receipt requires FP32 GDN h-state"
        );
        ensure!(
            !state.h_state.is_null() && !state.conv_state.is_null(),
            "target-state receipt found null GDN live state"
        );
        let mut h = vec![0u8; h_bytes];
        let mut conv = vec![0u8; conv_bytes];
        model.gpu.copy_d2h(state.h_state, &mut h)?;
        model.gpu.copy_d2h(state.conv_state, &mut conv)?;
        let h_label = format!("gdn.layer.{layer_idx}.h.live");
        let conv_label = format!("gdn.layer.{layer_idx}.conv.live");
        add_buffer(
            &mut context,
            h_label.as_bytes(),
            &h,
            &mut compared_buffers,
            &mut compared_bytes,
        )?;
        add_buffer(
            &mut context,
            conv_label.as_bytes(),
            &conv,
            &mut compared_buffers,
            &mut compared_bytes,
        )?;
        gdn_layers += 1;
    }
    ensure!(
        gdn_layers == QWEN4_GDN_LAYERS,
        "target-state receipt GDN census drifted"
    );

    let ple = model
        .qwen4_ple
        .as_ref()
        .context("target-state receipt missing PLE")?;
    let (ple_stride, ple_live, _) =
        ple.parity_snapshot(seq.slot_idx, model.gpu.as_ref(), stream)?;
    ensure!(
        ple_stride > 0 && ple_live.len() == ple_stride,
        "target-state receipt PLE live-state extent drifted"
    );
    add_buffer(
        &mut context,
        b"ple.conv.live",
        &ple_live,
        &mut compared_buffers,
        &mut compared_bytes,
    )?;

    let kv_cache = model.kv_cache.lock();
    ensure!(
        kv_cache.config().cache_blocks_per_seq.is_none() && seq.disk_block_ids.is_empty(),
        "target-state receipt does not admit high-speed-swap state"
    );
    let block_size = kv_cache.block_size();
    ensure!(block_size > 0, "target-state receipt KV block size is zero");
    let required_blocks = seq.seq_len.div_ceil(block_size);
    ensure!(
        seq.block_table.len() >= required_blocks,
        "target-state receipt block table is shorter than the committed prefix"
    );
    ensure!(
        kv_cache.num_layers() == QWEN4_ATTN_LAYERS,
        "target-state receipt attention-layer census drifted"
    );
    for layer_idx in 0..kv_cache.num_layers() {
        ensure!(
            kv_cache.dtype_for_layer(layer_idx) == KvCacheDtype::Bf16,
            "target-state receipt requires BF16 KV at attention layer {layer_idx}"
        );
        let k_stride = kv_cache.k_block_stride_bytes_for_layer(layer_idx);
        let v_stride = kv_cache.v_block_stride_bytes_for_layer(layer_idx);
        ensure!(
            k_stride.is_multiple_of(block_size) && v_stride.is_multiple_of(block_size),
            "target-state receipt KV block stride is not position-addressable"
        );
        let k_per_position = k_stride / block_size;
        let v_per_position = v_stride / block_size;
        for logical_block in 0..required_blocks {
            let used_positions = (seq.seq_len - logical_block * block_size).min(block_size);
            let physical_block = seq.block_table[logical_block];
            let (k, v) = kv_cache.read_block(layer_idx, physical_block, model.gpu.as_ref())?;
            let k_used = used_positions
                .checked_mul(k_per_position)
                .context("target-state receipt K prefix extent overflow")?;
            let v_used = used_positions
                .checked_mul(v_per_position)
                .context("target-state receipt V prefix extent overflow")?;
            ensure!(
                k_used <= k.len() && v_used <= v.len(),
                "target-state receipt KV read extent drifted"
            );
            let k_label = format!("kv.layer.{layer_idx}.block.{logical_block}.k");
            let v_label = format!("kv.layer.{layer_idx}.block.{logical_block}.v");
            add_buffer(
                &mut context,
                k_label.as_bytes(),
                &k[..k_used],
                &mut compared_buffers,
                &mut compared_bytes,
            )?;
            add_buffer(
                &mut context,
                v_label.as_bytes(),
                &v[..v_used],
                &mut compared_buffers,
                &mut compared_bytes,
            )?;
        }
    }

    let kv_token_positions = seq
        .seq_len
        .checked_mul(kv_cache.num_layers())
        .context("target-state receipt KV position census overflow")?;
    Ok(Qwen4TargetStateReceipt {
        schema: SCHEMA,
        mode,
        scheduler_path,
        proposer_loaded,
        seq_len: seq.seq_len,
        token_count: seq.tokens.len(),
        gdn_layers,
        attention_layers: kv_cache.num_layers(),
        logical_kv_blocks: required_blocks,
        kv_token_positions,
        compared_buffers,
        compared_bytes,
        fed_tokens_sha256: token_digest(&seq.tokens)?,
        target_state_sha256: hex(context.finish().as_ref()),
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) fn capture(
    model: &TransformerModel,
    seq: &SequenceState,
) -> Result<Option<Qwen4TargetStateReceipt>> {
    if !binary_switch(ENV)? {
        ensure!(
            optional_env(LIMIT_ENV)?.is_none(),
            "{LIMIT_ENV} requires {ENV}=1"
        );
        return Ok(None);
    }
    let limit = capture_limit()?;
    if CAPTURES.load(Ordering::Acquire) >= limit {
        return Ok(None);
    }
    let receipt = capture_enabled(model, seq)?;
    loop {
        let current = CAPTURES.load(Ordering::Acquire);
        if current >= limit {
            return Ok(None);
        }
        if CAPTURES
            .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Ok(Some(receipt));
        }
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
pub(super) fn capture(
    _model: &TransformerModel,
    _seq: &SequenceState,
) -> Result<Option<Qwen4TargetStateReceipt>> {
    if binary_switch(ENV)? {
        bail!("{ENV}=1 requires the Linux CUDA Qwen4 backend");
    }
    ensure!(
        optional_env(LIMIT_ENV)?.is_none(),
        "{LIMIT_ENV} requires {ENV}=1"
    );
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_mode_preserves_plain_loaded_serial_and_engaged_identity() {
        let cases = [
            (
                None,
                false,
                ("unloaded-target-only", "ordinary-target-only", false),
            ),
            (
                None,
                true,
                ("unloaded-target-only", "ordinary-target-only", false),
            ),
            (
                Some(DraftProposerKind::Qwen4NativeMtp),
                true,
                ("loaded-serial", "ordinary-target-only", true),
            ),
            (
                Some(DraftProposerKind::Qwen4NativeMtp),
                false,
                ("engaged-native-mtp", "native-mtp", true),
            ),
        ];
        for (kind, force_serial, expected) in cases {
            assert_eq!(classify_capture_mode(kind, force_serial).unwrap(), expected);
        }
    }

    #[test]
    fn capture_mode_rejects_foreign_proposers_with_either_control() {
        for kind in [DraftProposerKind::DFlash, DraftProposerKind::Other] {
            for force_serial in [false, true] {
                let error = classify_capture_mode(Some(kind), force_serial).unwrap_err();
                assert!(error.to_string().contains("rejects proposer kind"));
            }
        }
    }

    #[test]
    fn only_engaged_native_capture_uses_the_secondary_commit_wait() {
        for kind in [None, Some(DraftProposerKind::Qwen4NativeMtp)] {
            for force_serial in [false, true] {
                let (mode, _, _) = classify_capture_mode(kind, force_serial).unwrap();
                assert_eq!(
                    mode == "engaged-native-mtp",
                    kind == Some(DraftProposerKind::Qwen4NativeMtp) && !force_serial,
                );
            }
        }
    }

    #[test]
    fn token_digest_is_framed_and_order_sensitive() {
        let left = token_digest(&[1, 2, 3]).unwrap();
        let right = token_digest(&[1, 3, 2]).unwrap();
        assert_eq!(left.len(), 64);
        assert_ne!(left, right);
        assert_ne!(left, token_digest(&[1, 2, 3, 0]).unwrap());
    }

    #[test]
    fn framed_fields_do_not_alias_concatenation() {
        let digest = |parts: &[(&[u8], &[u8])]| {
            let mut context = Context::new(&SHA256);
            for (label, bytes) in parts {
                update_field(&mut context, label, bytes).unwrap();
            }
            hex(context.finish().as_ref())
        };
        assert_ne!(digest(&[(b"a", b"bc")]), digest(&[(b"ab", b"c")]),);
    }
}
