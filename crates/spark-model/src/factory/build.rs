// SPDX-License-Identifier: AGPL-3.0-only

//! `build_model` — entry point that wires up the configured loader,
//! buffers, KV cache, and (optional) DFlash drafter into a `TransformerModel`.

use std::sync::Arc;

use anyhow::{Context, Result};
use atlas_core::config::{DflashCaptureMode, ModelConfig};
use spark_runtime::buffers::{BufferArena, BufferSizes};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::startup_ledger;
use spark_runtime::weights::WeightDtype;
use spark_runtime::weights::WeightStore;

use super::DflashBuildArgs;
use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use super::qwen4_prefill_reserve::{
    maybe_reserve_qwen4_prefill_scratch, maybe_reserve_qwen4_target_shadows,
};
use super::qwen4_stream_t::maybe_setup_qwen4_stream_t;
use crate::layers::MtpQuantization;
use crate::model::TransformerModel;
use crate::traits::Model;
use crate::weight_loader::load_dflash_weights;
use crate::weight_map::quantize_to_nvfp4;

#[path = "qwen4_release_mode.rs"]
mod qwen4_release_mode;
use qwen4_release_mode::{
    Qwen4DflashLmHeadReleaseInput, Qwen4ReleaseMode, can_release_qwen4_dflash_bf16_lm_head,
    select_qwen4_release_mode,
};
#[path = "qwen4_lmhead_bf16.rs"]
mod qwen4_lmhead_bf16;
use qwen4_lmhead_bf16::{
    LmHeadPrecision, QWEN4_LMHEAD_BF16_ENV, StoredHead, lm_head_precision_from_env,
    validate_stored_bf16_head,
};
mod kv_carveout;

/// Whether E4M3 QSA prefill shadows can be materialized against `kv_config`.
///
/// E4M3 QSA shadows (`prefill_fp8::validate_target_shadow_geometry`) require
/// every attention layer to store BF16 KV; NVFP4 / FP8 / mixed-precision KV
/// layouts never materialize shadows and must never deduct shadow bytes from
/// KV pool sizing.
fn qwen4_qsa_fp8_shadows_apply_to_kv_config(kv_config: &KvCacheConfig) -> bool {
    if kv_config.layer_dtypes.is_empty() {
        kv_config.dtype == KvCacheDtype::Bf16
    } else {
        kv_config
            .layer_dtypes
            .iter()
            .all(|&dt| dt == KvCacheDtype::Bf16)
    }
}

/// Per-block system-memory bytes that grow with `num_kv_blocks` outside the
/// per-layer `PagedKvCache` pools (Qwen4 QSA side caches + optional E4M3 shadows).
fn qwen4_qsa_auxiliary_block_bytes(
    config: &ModelConfig,
    kv_config: &KvCacheConfig,
) -> Result<usize> {
    if !config.is_qwen4_exp() || !config.qwen4_qsa || kv_config.num_layers == 0 {
        return Ok(0);
    }
    let one = crate::layers::qwen4_qsa::checked_side_cache_bytes(1, kv_config.block_size)?;
    let two = crate::layers::qwen4_qsa::checked_side_cache_bytes(2, kv_config.block_size)?;
    let side_per_block = two
        .saturating_sub(one)
        .checked_mul(kv_config.num_layers)
        .context("Qwen4 QSA side-cache block budget overflow")?;
    let shadow_per_block = if qwen4_qsa_fp8_shadows_apply_to_kv_config(kv_config)
        && let Some(fp8_mode) = crate::layers::qwen4_qsa::prefill_fp8::mode()?
    {
        crate::layers::qwen4_qsa::prefill_fp8::checked_shadow_reserve_bytes_for_mode(
            fp8_mode,
            kv_config.block_size,
            1,
            kv_config.num_layers,
        )?
    } else {
        0
    };
    side_per_block
        .checked_add(shadow_per_block)
        .context("Qwen4 QSA auxiliary block budget overflow")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen4DflashPairing {
    Native,
    DenseDonorBridge,
}

fn validate_dedicated_model_dflash_boundary(model_type: &str, has_dflash_args: bool) -> Result<()> {
    anyhow::ensure!(
        model_type != "deepseek_v41" || !has_dflash_args,
        "DeepSeek-V4.1 uses internal DSpark and its dedicated factory cannot install external DFlash"
    );
    Ok(())
}

fn release_unused_qwen4_bf16_lm_head(
    store: &WeightStore,
    mut lm_head: crate::weight_map::DenseWeight,
    has_nvfp4_lm_head: bool,
    gpu: &dyn GpuBackend,
) -> Result<(crate::weight_map::DenseWeight, usize)> {
    if !has_nvfp4_lm_head {
        return Ok((lm_head, 0));
    }
    for key in [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ] {
        if store.contains(key) {
            // ATLAS_QWEN4_LMHEAD_BF16 (or the umbrella) keeps the BF16 head.
            let bytes = store.get(key)?.num_elements() * 2;
            let kind = crate::source_pins::PinKind::Bf16Alias;
            if crate::source_pins::skip_if_pinned(key, bytes, kind)?.is_some() {
                crate::source_pins::flush(key, "lm-head");
                return Ok((lm_head, 0));
            }
            let released =
                store.release_consumed_alias(key, lm_head.weight, WeightDtype::BF16, gpu)?;
            if released != 0 {
                lm_head.weight = spark_runtime::gpu::DevicePtr::NULL;
            }
            return Ok((lm_head, released));
        }
    }
    Ok((lm_head, 0))
}

/// Whether a dense (non-Qwen4, expert-free) model can safely release its
/// resident BF16 `lm_head.weight` once `lm_head_nvfp4` has been quantized.
///
/// Every target-model and native-MTP LM-head path reads `lm_head_nvfp4` when
/// present; the only runtime consumer of the untied BF16 `lm_head` is a
/// donor-less DFlash drafter sharing the target head.
const fn should_release_unused_dense_bf16_lm_head(
    is_qwen4_exp: bool,
    num_experts: usize,
    has_nvfp4_lm_head: bool,
    has_dflash_without_donor: bool,
    lm_head_ptr: u64,
    embed_ptr: u64,
    memlean_enabled: bool,
) -> bool {
    !is_qwen4_exp
        && num_experts == 0
        && has_nvfp4_lm_head
        && !has_dflash_without_donor
        && lm_head_ptr != 0
        && lm_head_ptr != embed_ptr
        && memlean_enabled
}

fn release_unused_qwen4_mtp(store: &WeightStore, gpu: &dyn GpuBackend) -> Result<(usize, usize)> {
    let names: Vec<String> = store
        .names()
        .filter(|name| name.starts_with("mtp."))
        .map(String::from)
        .collect();
    if names.is_empty() {
        return Ok((0, 0));
    }
    gpu.synchronize(gpu.default_stream())?;
    let mut bytes = 0usize;
    for name in &names {
        let weight = store.get(name)?;
        let released = store.release_consumed_alias(name, weight.ptr, weight.dtype, gpu)?;
        bytes = bytes
            .checked_add(released)
            .context("unused Qwen4 MTP released-byte count overflow")?;
    }
    Ok((names.len(), bytes))
}

fn maybe_release_unused_qwen4_mtp_early(
    store: &WeightStore,
    is_qwen4: bool,
    use_native_mtp: bool,
    has_dflash: bool,
    release_plain: bool,
    gpu: &dyn GpuBackend,
) -> Result<(usize, usize)> {
    if !matches!(
        select_qwen4_release_mode(
            is_qwen4,
            use_native_mtp,
            has_dflash,
            false,
            release_plain,
            false,
        ),
        Qwen4ReleaseMode::Plain | Qwen4ReleaseMode::Dflash
    ) {
        return Ok((0, 0));
    }
    startup_ledger::measure(
        "release-unused-mtp-early",
        "Plain/DFlash unused mtp.* aliases before transpose (negative = reclaimed)",
        None,
        || release_unused_qwen4_mtp(store, gpu),
    )
}

/// `ATLAS_MEMLEAN_UNUSED_MTP=1`: a dense (non-Qwen4, expert-free) checkpoint
/// builds its MTP head only with `--speculative`; otherwise `impl_a1_init`
/// skips it and every `mtp.*` tensor is unread (on Qwen3.8-27B: 810 MiB of
/// BF16). Release them right after the target layers, before the remaining
/// construction, and tell the caller to skip the dense MTP probe.
fn maybe_release_unused_dense_mtp(
    store: &WeightStore,
    config: &ModelConfig,
    use_native_mtp: bool,
    gpu: &dyn GpuBackend,
) -> Result<bool> {
    if config.is_qwen4_exp()
        || config.num_experts != 0
        || use_native_mtp
        || !crate::memlean::selected(crate::memlean::UNUSED_MTP)?
    {
        return Ok(false);
    }
    let (tensors, bytes) = startup_ledger::measure(
        "release-unused-mtp-dense",
        "MEMLEAN: dense mtp.* without --speculative (negative = reclaimed)",
        None,
        || release_unused_qwen4_mtp(store, gpu),
    )?;
    tracing::info!(
        "MEMLEAN_UNUSED_MTP_RELEASED selector={} tensors={tensors} mib={}",
        crate::memlean::UNUSED_MTP,
        bytes >> 20
    );
    Ok(true)
}

/// `ATLAS_MEMLEAN_DRAFTER_BF16=1`: release the drafter-store BF16 tensors
/// `from_weights` quantized to NVFP4 (`fc` and every layer's seven
/// projections). The head holds no pointer to them: its NVFP4 layers carry
/// none and its BF16 `fc` is NULL. On the Qwen3.8-27B z-lab DFlash2 drafter
/// that is 3.2 GiB. The device is drained first so no queued kernel still reads
/// the buffers, and every released name is tombstoned.
fn maybe_release_drafter_bf16_sources(
    store: &WeightStore,
    head: &mut crate::layers::BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if head.quantized_bf16_sources.is_empty()
        || !crate::memlean::selected(crate::memlean::DRAFTER_BF16)?
    {
        return Ok(());
    }
    let sources: std::collections::HashSet<u64> = head
        .quantized_bf16_sources
        .iter()
        .map(|ptr| ptr.0)
        .collect();
    // Release a source only when exactly one live store name owns it: a
    // second name for the same allocation would keep a dangling pointer.
    let mut owners: std::collections::BTreeMap<u64, Vec<String>> = Default::default();
    for name in store.names() {
        if let Ok(weight) = store.get(name)
            && weight.dtype == WeightDtype::BF16
            && sources.contains(&weight.ptr.0)
        {
            owners
                .entry(weight.ptr.0)
                .or_default()
                .push(name.to_string());
        }
    }
    let names: Vec<String> = owners
        .into_values()
        .filter_map(|mut names| (names.len() == 1).then(|| names.remove(0)))
        .collect();
    gpu.synchronize_device()?;
    let mut bytes = 0usize;
    for name in &names {
        let ptr = store.get(name)?.ptr;
        bytes += store.release_consumed_alias(name, ptr, WeightDtype::BF16, gpu)?;
    }
    tracing::info!(
        "MEMLEAN_DRAFTER_BF16_RELEASED selector={} tensors={} of {} quantized sources mib={}",
        crate::memlean::DRAFTER_BF16,
        names.len(),
        sources.len(),
        bytes >> 20
    );
    head.quantized_bf16_sources.clear();
    Ok(())
}

/// `[[dflash.profile]]` rows that pair a drafter natively with a Qwen4
/// (Flash-Next) target: their capture collapses the four-stream residual.
fn native_qwen4_profiles() -> impl Iterator<Item = &'static atlas_kernels::DrafterProfile> {
    atlas_kernels::drafter_profiles().iter().filter(|profile| {
        matches!(
            profile.capture_mode,
            atlas_kernels::DrafterCaptureMode::Qwen4HyperProjected
                | atlas_kernels::DrafterCaptureMode::Qwen4LayerAttnHc
                | atlas_kernels::DrafterCaptureMode::Qwen4HcMean
        )
    })
}

#[cfg(test)]
fn is_qwen38_flash_next_target_vocab(vocab_size: usize) -> bool {
    native_qwen4_profiles().any(|profile| {
        profile
            .target
            .vocab_sizes
            .is_some_and(|sizes| sizes.contains(&vocab_size))
    })
}

fn target_matches_profile(target: &ModelConfig, profile: &atlas_kernels::DrafterProfile) -> bool {
    let id = &profile.target;
    // The target store is loaded and preflighted at physical V248320, then
    // server admission narrows ModelConfig to the tokenizer's canonical
    // logical V248077 before factory construction. A row lists both as exact
    // identities for its target; no other cap is native.
    id.hidden_size
        .is_none_or(|hidden| target.hidden_size == hidden)
        && id
            .residual_width
            .is_none_or(|width| target.residual_width() == width)
        && id
            .num_layers
            .is_none_or(|layers| target.num_hidden_layers == layers)
        && id
            .vocab_sizes
            .is_none_or(|sizes| sizes.contains(&target.vocab_size))
}

/// The native Qwen4 profile row this target/drafter pair matches exactly.
fn native_qwen38_flash_next_profile(
    target: &ModelConfig,
    drafter: &crate::weight_loader::DflashConfig,
) -> Option<&'static atlas_kernels::DrafterProfile> {
    if !target.is_qwen4_exp() {
        return None;
    }
    native_qwen4_profiles().find(|profile| {
        target_matches_profile(target, profile)
            && crate::weight_loader::drafter_matches_profile(drafter, profile)
    })
}

#[cfg(test)]
fn is_native_qwen38_flash_next_drafter(
    target: &ModelConfig,
    drafter: &crate::weight_loader::DflashConfig,
) -> bool {
    native_qwen38_flash_next_profile(target, drafter).is_some()
}

/// The native Qwen4 row whose architecture the drafter declares, for checks
/// that must name the row even when the drafter's geometry has drifted.
fn native_qwen4_profile_for_architecture(
    drafter: &crate::weight_loader::DflashConfig,
) -> Result<&'static atlas_kernels::DrafterProfile> {
    if let Some(exact) = native_qwen4_profiles()
        .find(|profile| crate::weight_loader::drafter_matches_profile(drafter, profile))
    {
        return Ok(exact);
    }
    native_qwen4_profiles()
        .find(|profile| {
            matches!(
                (drafter.architectures.as_slice(), profile.drafter.architecture),
                ([architecture], Some(expected)) if architecture == expected
            )
        })
        .with_context(|| {
            format!(
                "no native Qwen3.8-Flash-Next drafter profile declares architectures={:?}",
                drafter.architectures
            )
        })
}

fn resolve_native_dflash2_proposal_vocab(
    drafter: &crate::weight_loader::DflashConfig,
    requested_vocab: u32,
) -> Result<usize> {
    anyhow::ensure!(
        drafter.is_dflash2(),
        "native Flash-Next DFlash2 proposal-vocabulary resolution requires DFlash2 semantics"
    );
    let profile = native_qwen4_profile_for_architecture(drafter)?;
    let proposal_vocab = profile.proposal_vocab.with_context(|| {
        format!(
            "native drafter profile {} declares no proposal_vocab",
            profile.label()
        )
    })?;
    let selector_vocab = drafter
        .dflash_config
        .as_ref()
        .and_then(|sub| sub.selector_vocab_size)
        .context("native Flash-Next DFlash2 requires selector_vocab_size")?;
    anyhow::ensure!(
        selector_vocab == proposal_vocab,
        "native Flash-Next DFlash2 selector_vocab_size={selector_vocab}; expected {proposal_vocab}"
    );
    anyhow::ensure!(
        requested_vocab == 0 || requested_vocab as usize == selector_vocab,
        "native Flash-Next DFlash2 proposal vocabulary must be uncapped or exactly {selector_vocab}; got {requested_vocab}"
    );
    Ok(selector_vocab)
}

fn classify_qwen4_dflash_pairing(
    native_geometry: bool,
    has_donor: bool,
) -> Result<Qwen4DflashPairing> {
    match (native_geometry, has_donor) {
        (true, false) => Ok(Qwen4DflashPairing::Native),
        (false, true) => Ok(Qwen4DflashPairing::DenseDonorBridge),
        (true, true) => anyhow::bail!(
            "native Qwen3.8-Flash-Next DFlash must share the target embedding/lm_head; remove --dflash-donor-model"
        ),
        (false, false) => anyhow::bail!(
            "Qwen3.8-Flash-Next requires either the exact native H2560/V248320/T48 DFlash checkpoint or --dflash-donor-model for an explicit dense-DFlash bridge"
        ),
    }
}

fn validate_native_qwen4_dflash_width(
    drafter: &crate::weight_loader::DflashConfig,
    requested_gamma: Option<usize>,
) -> Result<()> {
    let profile = native_qwen4_profile_for_architecture(drafter)?;
    let (block, (min_k, max_k)) = profile
        .drafter
        .block_size
        .zip(profile.verify_k)
        .with_context(|| {
            format!(
                "native drafter profile {} must pin block_size and verify_k",
                profile.label()
            )
        })?;
    let trained_block = drafter.resolved_block_size();
    anyhow::ensure!(
        trained_block == block,
        "native Qwen3.8-Flash-Next {} requires the exact B{block}/gamma{} recipe, but checkpoint block_size={trained_block}",
        profile.label(),
        max_k - 1
    );
    let gamma = requested_gamma.unwrap_or(max_k - 1);
    anyhow::ensure!(
        (min_k..=max_k).contains(&gamma.saturating_add(1)),
        "native Qwen3.8-Flash-Next {} uses bidirectional B{block} attention and verifies K{min_k}..=K{max_k}, but gamma={gamma} was requested",
        profile.label()
    );
    Ok(())
}

/// The runtime capture transform a profile row selects. Rows may name
/// declared-but-unimplemented modes; pairing refuses them here.
fn runtime_capture_mode(profile: &atlas_kernels::DrafterProfile) -> Result<DflashCaptureMode> {
    use atlas_kernels::DrafterCaptureMode as Profile;
    match profile.capture_mode {
        Profile::ResidualSlice => Ok(DflashCaptureMode::ResidualSlice),
        Profile::Qwen4HyperProjected => Ok(DflashCaptureMode::Qwen4HyperProjected),
        Profile::Qwen4LayerAttnHc => Ok(DflashCaptureMode::Qwen4LayerAttnHc),
        Profile::Qwen4HcMean => Ok(DflashCaptureMode::Qwen4HcMean),
        mode @ Profile::ResidualSliceF32Downcast => anyhow::bail!(
            "drafter profile {} selects capture mode {mode:?}, which has no engine capture path yet",
            profile.label()
        ),
    }
}

/// Every implemented capture mode copies BF16 rows out of the residual
/// stream. An FP32-residual target (dense Gemma-4) would hand the drafter
/// FP32 bytes read as BF16 — garbage that only surfaces as acceptance — so
/// the pairing is refused until a downcast capture mode exists.
fn validate_dflash_capture_residual(config: &ModelConfig) -> Result<()> {
    anyhow::ensure!(
        !config.use_fp32_residual(),
        "DFlash capture mode {:?} reads BF16 residual rows, but this {} target runs an FP32 \
         residual; no FP32->BF16 downcast capture mode is implemented",
        config.dflash_capture_mode,
        config.model_type,
    );
    Ok(())
}

fn remap_capture_depths(
    ids: &[usize],
    source_layers: usize,
    target_layers: usize,
) -> Result<Vec<usize>> {
    anyhow::ensure!(
        source_layers > 1 && target_layers > 1,
        "invalid DFlash capture-depth bridge {source_layers}->{target_layers}"
    );
    let source_last = source_layers - 1;
    let target_last = target_layers - 1;
    ids.iter()
        .map(|&id| {
            anyhow::ensure!(
                id < source_layers,
                "DFlash capture layer {id} is outside donor depth {source_layers}"
            );
            Ok((id * target_last + source_last / 2) / source_last)
        })
        .collect()
}

fn donor_tensor<'a>(
    store: &'a WeightStore,
    candidates: &[&str],
    expected_shape: &[usize],
    role: &str,
) -> Result<&'a spark_runtime::weights::WeightTensor> {
    let matches: Vec<_> = candidates
        .iter()
        .filter_map(|name| store.get(name).ok().map(|tensor| (*name, tensor)))
        .collect();
    anyhow::ensure!(
        matches.len() == 1,
        "DFlash donor must contain exactly one {role}; found {:?}",
        matches.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );
    let (name, tensor) = matches[0];
    anyhow::ensure!(
        tensor.dtype == WeightDtype::BF16,
        "DFlash donor tensor {name} must be BF16, got {:?}",
        tensor.dtype
    );
    anyhow::ensure!(
        tensor.shape == expected_shape,
        "DFlash donor tensor {name} has shape {:?}; expected {:?}",
        tensor.shape,
        expected_shape
    );
    Ok(tensor)
}

pub fn build_model(
    mut config: ModelConfig,
    store: &WeightStore,
    gpu: std::sync::Arc<dyn GpuBackend>,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    mtp_quant: MtpQuantization,
    use_native_mtp: bool,
    prefix_cache: Box<dyn PrefixCache>,
    mtp_vocab_size: u32,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    inference_reserve: usize,
    gpu_memory_utilization: f64,
    ssm_cache_slots: usize,
    layer_dtypes: Vec<KvCacheDtype>,
    ssm_checkpoint_interval: usize,
    // Per-sequence HBM cache cap: `Some(N)` enables `--high-speed-swap`
    // HBM-shrink behavior; `None` preserves unbounded per-sequence HBM cache.
    hss_cache_blocks_per_seq: Option<u32>,
    // `--kv-pool-tokens`: shared target KV pool size. `None` keeps the
    // per-slot pool (every slot reachable at max_seq_len) byte-identical.
    kv_pool_tokens: Option<usize>,
    // DFlash speculative-decoding pairing. `None` = no DFlash; existing
    // MTP / no-spec paths unchanged.
    dflash_args: Option<DflashBuildArgs<'_>>,
) -> Result<Box<dyn Model>> {
    validate_dedicated_model_dflash_boundary(&config.model_type, dflash_args.is_some())?;
    // DeepSeek-V4.1 (forward-ported from dsv41/integration): standalone model over the CED +
    // SWA-replay forward (`model::dsv41`). Dispatched FIRST: none of the decoder-only
    // construction below describes it.
    #[cfg(feature = "cuda")]
    if config.model_type == "deepseek_v41" {
        if max_batch_size > 1 {
            tracing::warn!(
                "DeepSeek-V4.1 serves ONE live sequence (model-wide compressed-KV caches); \
                 --max-batch-size {max_batch_size} is ignored"
            );
        }
        let model_dir = crate::weight_loader::deepseek_v41::resolve_model_dir()?;
        let max_chunk = std::env::var("ATLAS_DSV41_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let model = crate::model::dsv41::Dsv41Model::new(
            &config,
            store,
            gpu,
            &model_dir,
            max_seq_len,
            max_chunk,
        )?;
        return Ok(Box::new(model));
    }
    // ── Step 1: Select weight loader (only model-specific dispatch) ──
    let loader = loader_for_config(&config)?;

    // Pre-construction: when DFlash is active, populate the target's
    // capture-layer indices from the drafter's `dflash_config.target_layer_ids`
    // so `TransformerModel::new` allocates the `target_layer_ids.len() × capture_width`
    // capture buffer.
    //
    // Capture-layer alignment: use `target_layer_ids` verbatim (offset 0).
    //
    //   training  — SpecForge reads `outputs.hidden_states[idx + 1]`, where
    //               `+1` skips HuggingFace's element 0 (embedding output), so
    //               `hidden_states[N + 1]` is the output of layer N.
    //   serving   — `try_dflash_capture(i, ..)` runs immediately after
    //               `layer.decode()` for layer `i`, which already holds the
    //               output of layer `i` with no embedding tuple offset.
    //
    // Both sides therefore reference output-of-layer-N at offset 0.
    // `ATLAS_DFLASH_CAPTURE_LAYER_OFFSET` is strictly a diagnostic override;
    // any nonzero value misaligns the drafter against its training distribution.
    if let Some(ref args) = dflash_args
        && let Some(ref sub) = args.drafter_config.dflash_config
    {
        let offset: i64 = std::env::var("ATLAS_DFLASH_CAPTURE_LAYER_OFFSET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if offset != 0 {
            tracing::warn!(
                "ATLAS_DFLASH_CAPTURE_LAYER_OFFSET={offset}: capture layers are now \
                 MISALIGNED against the drafter's training data, which captured \
                 output-of-layer-N for the raw target_layer_ids. This degrades draft \
                 acceptance silently — it will not error. Only 0 is correct; use a \
                 nonzero value for A/B testing that claim and nothing else."
            );
        }
        let raw_with_offset: Vec<_> = sub
            .target_layer_ids
            .iter()
            .map(|&id| (id as i64 + offset).max(0) as usize)
            .collect();
        if config.is_qwen4_exp() {
            let native_profile = native_qwen38_flash_next_profile(&config, &args.drafter_config);
            let pairing = classify_qwen4_dflash_pairing(
                native_profile.is_some(),
                args.donor_store.is_some(),
            )?;
            config.dflash_capture_width = args.drafter_config.hidden_size;
            config.dflash_capture_offset = 0;
            match pairing {
                Qwen4DflashPairing::Native => {
                    let profile = native_profile.context("native pairing without a profile row")?;
                    validate_native_qwen4_dflash_width(&args.drafter_config, args.gamma)?;
                    anyhow::ensure!(
                        profile.qualified
                            || std::env::var("ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH")
                                .ok()
                                .as_deref()
                                == Some("1"),
                        "native Qwen4 DFlash support is unqualified and disabled by default; set ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH=1 only for correctness research"
                    );
                    anyhow::ensure!(
                        offset == 0,
                        "native Qwen3.8-Flash-Next DFlash requires exact unshifted capture layers"
                    );
                    config.dflash_capture_layers = raw_with_offset;
                    config.dflash_capture_mode = runtime_capture_mode(profile)?;
                    tracing::warn!(
                        "EXPERIMENTAL unqualified native Qwen3.8-Flash-Next DFlash (profile {}): exact capture layers {:?}; capture_mode={:?} to H={}. Do not use as a performance candidate until acceptance qualification passes.",
                        profile.label(),
                        config.dflash_capture_layers,
                        config.dflash_capture_mode,
                        config.dflash_capture_width,
                    );
                }
                Qwen4DflashPairing::DenseDonorBridge => {
                    let source_layers = args.drafter_config.num_target_layers;
                    anyhow::ensure!(
                        source_layers > 1,
                        "Qwen4 DFlash bridge requires drafter config num_target_layers"
                    );
                    config.dflash_capture_layers = remap_capture_depths(
                        &raw_with_offset,
                        source_layers,
                        config.num_hidden_layers,
                    )?;
                    config.dflash_capture_mode = DflashCaptureMode::ResidualSlice;
                    anyhow::ensure!(
                        config.dflash_capture_width <= config.residual_width(),
                        "dense DFlash bridge width {} exceeds Flash-Next residual width {}",
                        config.dflash_capture_width,
                        config.residual_width(),
                    );
                    tracing::warn!(
                        "EXPERIMENTAL Qwen4->dense-DFlash bridge: capture depths {:?}->{:?}; exposing BF16 residual slice [{}..{}) of {}. Target verification remains authoritative; qualification is required before production use.",
                        raw_with_offset,
                        config.dflash_capture_layers,
                        config.dflash_capture_offset,
                        config.dflash_capture_offset + config.dflash_capture_width,
                        config.residual_width(),
                    );
                }
            }
        } else {
            anyhow::ensure!(
                args.donor_store.is_none(),
                "--dflash-donor-model is only valid for an explicit Qwen3.8-Flash-Next bridge"
            );
            config.dflash_capture_layers = raw_with_offset;
            // Load-time admission already refused a DFlash2-family drafter
            // that matches no exact row; an admitted one takes its row's
            // capture mode. Generic DFlash/DSpark keeps the BF16 slice.
            if let Ok(profile) = crate::weight_loader::exact_dflash2_profile(&args.drafter_config) {
                config.dflash_capture_mode = runtime_capture_mode(profile)?;
                tracing::info!(
                    "DFlash drafter profile {} (qualified={})",
                    profile.label(),
                    profile.qualified,
                );
            }
        }
        tracing::info!(
            "DFlash: target layer capture indices = {:?} (offset={offset} from raw {:?})",
            config.dflash_capture_layers,
            sub.target_layer_ids,
        );
        validate_dflash_capture_residual(&config)?;
    }

    // ── Step 2: Load weights (model-agnostic from here) ──
    let attn_layer_dtypes: Vec<KvCacheDtype> = if layer_dtypes.is_empty() {
        vec![kv_dtype; config.num_attention_layers()]
    } else {
        layer_dtypes.clone()
    };

    // Populate per-layer KV dims for heterogeneous-attention models (Gemma-4).
    // Homogeneous models return an empty Vec which the KV cache treats as
    // "use global num_kv_heads/head_dim for all layers" (backward compatible).
    config.kv_layer_dims = loader.kv_layer_dims(&config);

    // ATLAS_QWEN4_DENSE_NATIVE=1 (default off): one INFO line naming every
    // native-precision knob and where its value came from.
    crate::qwen4_native_selectors::log_plan_once()?;
    // ATLAS_QWEN4_ROUTER_BF16=1 (default off): refuse every launch that would
    // reach an NVFP4-only router consumer before any layer drops the copy.
    let router_precision = crate::layers::moe::qwen4_router_bf16::admit_launch(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
        max_batch_size,
        config.tp_world_size > 1 || config.ep_world_size > 1,
    )?;
    if router_precision == crate::layers::moe::qwen4_router_bf16::RouterPrecision::Bf16 {
        let bytes = crate::layers::moe::qwen4_router_bf16::extra_resident_bytes(
            config.num_experts,
            config.hidden_size,
            config.num_hidden_layers,
        );
        tracing::info!(
            "{}",
            crate::layers::moe::qwen4_router_bf16::engaged_line(bytes)
        );
    }
    // ATLAS_QWEN4_GDN_BF16=1 / ATLAS_QWEN4_ATTN_BF16=1 (default off): same
    // refusal shape as the router control above — refuse every launch
    // that would reach an NVFP4-only decode consumer before any layer is
    // built.
    crate::layers::qwen4_decode_bf16_arms::admit_launch(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
        max_batch_size,
        config.tp_world_size > 1 || config.ep_world_size > 1,
    )?;
    // ATLAS_QWEN4_SHARED_BF16=1 (default off): same refusal table as the
    // router control — refuse every launch that would reach an NVFP4-only
    // shared-expert consumer before any layer drops the copy.
    crate::layers::moe::qwen4_shared_bf16::admit_launch(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
        max_batch_size,
        config.tp_world_size > 1 || config.ep_world_size > 1,
    )?;
    // ATLAS_QWEN4_DENSE_FP8=1 (default off): same refusals for the native
    // MXFP8 decode arms of the dense modules (and BF16/FP8 exclusion).
    crate::layers::qwen4_dense_fp8::admit_launch(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
        max_batch_size,
        config.tp_world_size > 1 || config.ep_world_size > 1,
    )?;
    // ATLAS_QWEN4_HC_BF16=1 (default off): refuse every launch that would
    // reach the HC mixer's batched-verify/native-batch tail, which this
    // control has no BF16 arm for, before any layer keeps a BF16 copy.
    let keep_hc_bf16 = crate::layers::qwen4_hc_bf16::admit_launch(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
        max_batch_size,
        config.tp_world_size > 1 || config.ep_world_size > 1,
    )?;
    if keep_hc_bf16 {
        // `aliased_bytes` is an alias into WeightStore tensors `dense_auto`
        // already holds and that are never freed on the default path
        // either — not new allocation. `extra_resident_bytes` is the true
        // cost of this control (always 0; see EXTRA_RESIDENT_BYTES).
        let aliased_bytes = crate::layers::qwen4_hc_bf16::aliased_bf16_bytes(
            config.hc_lowrank,
            config.residual_width(),
            config.hc_count,
            config.num_hidden_layers,
        );
        tracing::info!(
            extra_resident_bytes = crate::layers::qwen4_hc_bf16::EXTRA_RESIDENT_BYTES,
            aliased_bytes,
            aliased_mib = aliased_bytes as f64 / (1024.0 * 1024.0),
            "QWEN4_HC_BF16 engaged scope=decode-only"
        );
    }
    crate::weight_loader::transform_cache::configure_construction_mode(use_native_mtp)?;
    let mut layers = startup_ledger::measure(
        "target-layers",
        format!(
            "{} layers; SSM/attention transposes, PLE rings (estimate: construction_overhead_bytes)",
            config.num_hidden_layers
        ),
        None,
        || loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes),
    )?;
    // Plain mode never consumes mtp.*. Release only those aliases before the
    // transpose peak; the BF16 LM head must survive until quantization below.
    let (early_mtp_tensors, early_mtp_bytes) = maybe_release_unused_qwen4_mtp_early(
        store,
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        std::env::var("ATLAS_QWEN4_RELEASE_UNUSED").ok().as_deref() == Some("1"),
        gpu.as_ref(),
    )?;
    if early_mtp_tensors != 0 {
        tracing::info!(
            tensors = early_mtp_tensors,
            requested_bytes = early_mtp_bytes,
            "QWEN4_RELEASE_UNUSED_MTP_EARLY before target MoE transpose"
        );
    }
    let dense_mtp_released =
        maybe_release_unused_dense_mtp(store, &config, use_native_mtp, gpu.as_ref())?;
    // Qwen4 (Flash-Next): consolidate every layer's 1,536 per-expert routed
    // allocations into two phased transposed slabs per layer NOW, before the
    // native-MTP layer, the LM-head quantization and the arena add to the
    // footprint. The unified pass replaces each layer's routed payload;
    // the two slabs add 512 guard bytes. The earlier single-slab pass
    // reclaimed 10.9 GB of allocator overhead on the
    // sealed r8 receipts (MOE_TRANSPOSE_PASS_MEMORY free_delta_bytes=
    // 10914893824). Running it last, in the MiniMax position below, placed
    // the startup peak and the NVRM NV_ERR_NO_MEMORY fallback storm (247/601
    // kernel lines at 15.6-17.0 GB free) inside the MTP construction window.
    //
    // Invariant checked for this position: nothing between here and the old
    // call site reads the untransposed routed-expert layout. The pass frees
    // the originals and NULLs the per-layer pointers (`helpers_a.rs` phases
    // B/D) without tombstoning the `WeightStore` names, so a later
    // `store.get("...mlp.experts...")` would return a dangling pointer; the
    // code in between only looks up embedding / norm / mixer / PLE / lm_head
    // / `mtp.*` / vision names, the native-MTP layer loads its own experts
    // under the `mtp` prefix and shares only `embed` and the NVFP4 LM head,
    // the release paths free by name (`mtp.*`, BF16 lm_head), and the
    // transform cache never sees the pass (slab transposes bypass
    // `transpose_for_gemm_cached`). MiniMax keeps the post-LM-head position
    // because its non-unified pass needs the memory the BF16 LM head frees.
    if config.is_qwen4_exp() {
        startup_ledger::measure(
            "moe-transpose-pass",
            "unified; two phased transposed slabs per layer, originals freed (negative = reclaimed)",
            None,
            || maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers),
        )?;
    }
    if config.dflash_capture_mode == DflashCaptureMode::Qwen4LayerAttnHc {
        for &tap in &config.dflash_capture_layers {
            let next = tap
                .checked_add(1)
                .filter(|&idx| idx < layers.len())
                .with_context(|| {
                    format!(
                        "Qwen4LayerAttnHc tap layer {tap} has no next layer (num_layers={})",
                        layers.len()
                    )
                })?;
            anyhow::ensure!(
                layers[next].qwen4_attn_hyper().is_some(),
                "Qwen4LayerAttnHc tap layer {tap} requires layers[{next}].qwen4_attn_hyper(), \
                 but layer {next} exposes no attention hyperconnection mixer"
            );
        }
    }
    // Receipt-only scopes (`STARTUP_ALLOC_RECEIPT`, no ledger pool): the loads
    // between the target layers and the LM-head quantization.
    let aux_receipt = startup_ledger::scope_guard("aux-weights");
    let embed = loader.load_embedding(store, &config, gpu.as_ref())?;
    let final_norm = loader.load_final_norm(store, &config, gpu.as_ref())?;
    let qwen4_final_mixer = loader.load_qwen4_final_mixer(store, &config, gpu.as_ref())?;
    if config.is_qwen4_exp() {
        // ATLAS_QWEN4_PREFILL_NATIVE=1: load receipt; fails on a planned family with no weights.
        crate::layers::qwen4_prefill_native::log_load_receipt(&config)?;
        // ATLAS_QWEN4_PREFILL_NATIVE=1 family shared: load receipt; fails on a missing layer.
        crate::layers::qwen4_prefill_native::shared::log_load_receipt(&config)?;
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    let qwen4_ple =
        crate::layers::Qwen4PleLayer::load(store, &config, gpu.as_ref(), max_batch_size)?;
    drop(aux_receipt);
    let lm_head = startup_ledger::scope("lm-head-source", || {
        loader.load_lm_head(store, &config, gpu.as_ref())
    })?;
    let mtp_vision_receipt = startup_ledger::scope_guard("mtp-and-vision-weights");
    let mtp_weights = loader.load_mtp_weights_multi(store, &config, gpu.as_ref())?;
    // Probe dense MTP path for non-MoE models (Qwen3.5/3.6 27B family).
    // Loader returns None for MoE models so this is a no-op there. The full
    // DenseMtpHead layer is not yet wired — for now we just log presence so
    // the user sees the loader works.
    let mtp_dense_weights = if dense_mtp_released {
        None
    } else {
        loader.load_mtp_dense_weights(store, &config, gpu.as_ref())?
    };
    let vision_encoder = loader.load_vision_encoder(store, &config, gpu.as_ref())?;
    drop(mtp_vision_receipt);

    // If the checkpoint's `quantization_config.ignore_modules` lists MTP
    // (e.g. Sehyo/Qwen3.5-35B-A3B-NVFP4 ignores `mtp.*`), the MTP weights
    // were stored as BF16 on disk. Runtime-quantizing them to NVFP4
    // anyway — which is what `mtp_quant` would otherwise do — produces
    // garbage drafts (vllm PR #38832). Force BF16 in that case.
    let effective_mtp_quant = if !mtp_weights.is_empty() {
        let quant_fmt = crate::quant_format::detect_quant_format(&config, store);
        if quant_fmt.is_ignored("mtp.fc.weight")
            || quant_fmt.is_ignored("mtp.layers.0.self_attn.q_proj.weight")
        {
            if mtp_quant != MtpQuantization::Bf16 {
                tracing::info!(
                    "MTP head listed in checkpoint ignore_modules — overriding \
                     --mtp-quantization {:?} → Bf16 to preserve precision",
                    mtp_quant,
                );
            }
            MtpQuantization::Bf16
        } else {
            mtp_quant
        }
    } else {
        mtp_quant
    };

    // ── Step 3: Quantize LM head to NVFP4 for fast decode ──
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu
        .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
        .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
    let stream = gpu.default_stream();
    let skip_quant = config.skip_lm_head_quantization();
    let lm_head_precision = lm_head_precision_from_env(
        config.is_qwen4_exp(),
        skip_quant,
        use_native_mtp,
        dflash_args.is_some(),
        self_speculative,
    )?;
    let lm_head_nvfp4 = if lm_head_precision == LmHeadPrecision::Bf16 && skip_quant {
        tracing::info!("LM head kept as BF16 (skip NVFP4 quantization per model config)");
        None
    } else if lm_head_precision == LmHeadPrecision::Bf16 {
        // Every dense consumer reads the first `vocab_size` rows of the
        // physical [rows, hidden] tensor with 16-byte vector loads. Same
        // key order as `qwen35_mixed_precision::load_lm_head`.
        let key = [
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ]
        .into_iter()
        .find(|key| store.contains(key))
        .context("BF16 LM-head control needs a checkpoint lm_head.weight (untied)")?;
        let stored = store.get(key)?;
        validate_stored_bf16_head(
            key,
            StoredHead {
                dtype: stored.dtype,
                shape: &stored.shape,
                stored_ptr: stored.ptr.0,
                loaded_ptr: lm_head.weight.0,
            },
            config.vocab_size,
            config.hidden_size,
        )?;
        tracing::info!(
            "LM head kept as BF16 ({QWEN4_LMHEAD_BF16_ENV}=1 control, stored {:?}, vocab={}, physical rows={})",
            stored.dtype,
            config.vocab_size,
            stored.shape[0]
        );
        // Grep-able receipt in the same form as the other native controls:
        // the BF16 head kept resident (gross; no NVFP4 head is built).
        tracing::info!(
            "QWEN4_LMHEAD_BF16 engaged scope=decode+prefill extra_resident_bytes={}",
            config.vocab_size as u64 * config.hidden_size as u64 * 2
        );
        None
    } else {
        // packed e2m1 (1/2 B) + e4m3 block scales (1/16 B) per element.
        let requested = config.vocab_size * config.hidden_size * 9 / 16;
        let q = startup_ledger::measure(
            "lm-head-nvfp4",
            format!(
                "vocab {} x hidden {} x 9/16",
                config.vocab_size, config.hidden_size
            ),
            Some(requested),
            || {
                quantize_to_nvfp4(
                    &lm_head,
                    config.vocab_size,
                    config.hidden_size,
                    gpu.as_ref(),
                    absmax_k,
                    quantize_k,
                    stream,
                )
            },
        )?;
        tracing::info!("LM head quantized to NVFP4 (vocab={})", config.vocab_size);
        Some(q)
    };
    let lm_head_fp8 = if config.is_qwen4_exp()
        && config.pp_rank == 0
        && lm_head_nvfp4.is_some()
        && crate::model::impl_a3::qwen4_target_lmhead_mode()
            == crate::model::impl_a3::Qwen4TargetLmHeadMode::Fp8
        && !lm_head.weight.is_null()
    {
        match gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8") {
            Ok(quantize_fp8_k) => {
                // 1 B/weight FP8 E4M3 + 4 B per-row FP32 scale.
                let requested = config.vocab_size * config.hidden_size + config.vocab_size * 4;
                let q = startup_ledger::measure(
                    "lm-head-fp8",
                    format!(
                        "vocab {} x hidden {} x 1 B + row scales",
                        config.vocab_size, config.hidden_size
                    ),
                    Some(requested),
                    || {
                        crate::weight_map::quantize_to_fp8(
                            &lm_head,
                            config.vocab_size,
                            config.hidden_size,
                            gpu.as_ref(),
                            quantize_fp8_k,
                            stream,
                        )
                    },
                )?;
                tracing::info!(
                    "LM head quantized to FP8 E4M3 per-row scaled (vocab={}, {:.1} MiB)",
                    config.vocab_size,
                    requested as f64 / (1024.0 * 1024.0),
                );
                Some(q)
            }
            Err(err) => {
                tracing::warn!(
                    "gemv_fp8w::quantize_bf16_to_fp8 unavailable ({err:#}); \
                     skipping FP8 target LM head"
                );
                None
            }
        }
    } else {
        None
    };

    let is_head_rank = config.tp_rank == 0 && config.ep_rank == 0 && config.pp_rank == 0;
    let qwen4_mtp_layout = if use_native_mtp && config.is_qwen4_exp() && is_head_rank {
        crate::weight_loader::qwen4_mtp::classify_qwen4_mtp_store(store, &config)?
    } else {
        None
    };
    let qwen4_mtp_proposer: Option<Arc<dyn crate::speculative::DraftProposer>> = if let Some(
        mtp_layout,
    ) =
        qwen4_mtp_layout
    {
        tracing::info!(
            layout = ?mtp_layout,
            "Qwen4 native MTP schema admitted for direct proposer construction"
        );
        let mtp_config = crate::weight_loader::qwen4_mtp_config(&config);
        let head = startup_ledger::measure(
            "mtp-native-layer",
            format!(
                "1 layer: routed experts (+transposed copy), MTP KV for max_seq_len {max_seq_len} x batch {max_batch_size}"
            ),
            None,
            || {
                let mtp_layer = crate::weight_loader::load_qwen4_mtp_layer(
                    store,
                    &config,
                    gpu.as_ref(),
                    KvCacheDtype::Bf16,
                )?;
                let mtp_final_mixer = loader
                    .load_qwen4_final_mixer(store, &mtp_config, gpu.as_ref())?
                    .context("Qwen4 MTP checkpoint is missing its final hyperconnection mixer")?;
                let shared_lm_head =
                    lm_head_nvfp4.context("Qwen4 native MTP requires the target NVFP4 LM head")?;
                crate::layers::Qwen4MtpHead::new(
                    store,
                    &mtp_config,
                    mtp_layer,
                    mtp_final_mixer,
                    embed,
                    shared_lm_head,
                    gpu.as_ref(),
                    mtp_vocab_size,
                    max_seq_len,
                    max_batch_size,
                    super::kv_pool::shared_pool_tokens(
                        kv_pool_tokens,
                        max_seq_len,
                        max_batch_size,
                        kv_block_size,
                    ),
                )
            },
        )?;
        Some(Arc::new(head))
    } else {
        None
    };
    if config.is_qwen4_exp() {
        // Qwen4's native MTP layer, final mixer, and packed BF16 expert bank
        // are constructed after the target layers. Publish only once every
        // selected transform has succeeded; an error leaves no cache index.
        crate::weight_loader::transform_cache::finish();
    }

    let release_mode = select_qwen4_release_mode(
        config.is_qwen4_exp(),
        use_native_mtp,
        dflash_args.is_some(),
        qwen4_mtp_proposer.is_some(),
        std::env::var("ATLAS_QWEN4_RELEASE_UNUSED").ok().as_deref() == Some("1"),
        std::env::var("ATLAS_QWEN4_RELEASE_SPEC_UNUSED")
            .ok()
            .as_deref()
            == Some("1"),
    );
    let release_mem_before = startup_ledger::available_device_memory_bytes();
    let release_receipt = startup_ledger::scope_guard("release-unused");
    let lm_head = match release_mode {
        // Flash-Next loads the whole checkpoint. Without speculation or a DFlash
        // drafter, the MTP head is never read, and the BF16 lm_head
        // (~1.3 GB) is only read when no NVFP4 head exists. Release both from the
        // store; later lookups of these names fail closed.
        Qwen4ReleaseMode::Plain => {
            gpu.synchronize(stream)?;
            let (mtp_tensors, mtp_bytes) = release_unused_qwen4_mtp(store, gpu.as_ref())?;
            let (head, head_bytes) = release_unused_qwen4_bf16_lm_head(
                store,
                lm_head,
                !crate::model::impl_a3::qwen4_target_lmhead_bf16_enabled()
                    && lm_head_nvfp4.is_some(),
                gpu.as_ref(),
            )?;
            tracing::info!(
                "QWEN4_RELEASE_UNUSED_ENGAGED mtp_tensors={} mtp_mib={} lm_head_bf16_mib={}",
                mtp_tensors,
                mtp_bytes >> 20,
                head_bytes >> 20
            );
            head
        }
        // DFlash never reads `mtp.*` (already released early before transpose;
        // repeated here idempotently). The target BF16 `lm_head.weight`
        // (~1.18 GiB) is also released before Step-5 KV sizing whenever every
        // DFlash LM-head safety invariant holds (NVFP4 head built, target BF16
        // head disabled, untied, and either a donor store supplies the drafter
        // head or the drafter shares `lm_head_shared_nvfp4_exact` without FP8
        // conversion, DSpine, or debug dump).
        Qwen4ReleaseMode::Dflash => {
            gpu.synchronize(stream)?;
            let (mtp_tensors, mtp_bytes) = release_unused_qwen4_mtp(store, gpu.as_ref())?;
            let target_hidden = if config.dflash_capture_width > 0 {
                config.dflash_capture_width
            } else {
                config.residual_width()
            };
            let has_donor_store = dflash_args
                .as_ref()
                .is_some_and(|args| args.donor_store.is_some());
            let drafter_has_dspine = dflash_args
                .as_ref()
                .is_some_and(|args| args.drafter_config.is_dspine());
            let drafter_hidden_matches_target = dflash_args.as_ref().is_some_and(|args| {
                args.drafter_config.hidden_size == target_hidden && target_hidden.is_multiple_of(16)
            });
            let has_exact_lm_head_kernels = [
                crate::layers::ops::ExactLmHeadTier::M4,
                crate::layers::ops::ExactLmHeadTier::M8,
                crate::layers::ops::ExactLmHeadTier::M17,
                crate::layers::ops::ExactLmHeadTier::M32,
            ]
            .into_iter()
            .all(|tier| gpu.kernel("w4a16_gemv", tier.symbol()).is_ok());
            let can_release_bf16_head =
                can_release_qwen4_dflash_bf16_lm_head(Qwen4DflashLmHeadReleaseInput {
                    has_nvfp4_lm_head: lm_head_nvfp4.is_some(),
                    target_bf16_lm_head_enabled:
                        crate::model::impl_a3::qwen4_target_lmhead_bf16_enabled(),
                    lm_head_is_untied: !lm_head.weight.is_null()
                        && lm_head.weight.0 != embed.weight.0,
                    has_donor_store,
                    dflash_lm_head_fp8_enabled: std::env::var("ATLAS_DFLASH_LM_HEAD_FP8")
                        .ok()
                        .as_deref()
                        == Some("1"),
                    drafter_has_dspine,
                    drafter_hidden_matches_target,
                    has_exact_lm_head_kernels,
                    dflash_debug_dump_enabled: std::env::var("ATLAS_DFLASH_DEBUG_DUMP")
                        .ok()
                        .as_deref()
                        == Some("1"),
                });
            let (head, head_bytes) = release_unused_qwen4_bf16_lm_head(
                store,
                lm_head,
                can_release_bf16_head,
                gpu.as_ref(),
            )?;
            tracing::info!(
                "QWEN4_RELEASE_DFLASH_UNUSED_ENGAGED mtp_tensors={} mtp_mib={} lm_head_bf16_mib={}",
                mtp_tensors,
                mtp_bytes >> 20,
                head_bytes >> 20
            );
            head
        }
        // Native Qwen4 MTP gets its routed experts from the numbered NVFP4
        // sidecar. When those tensors exist, the target checkpoint's packed BF16
        // expert banks (~4.8 GB) are unread. The proposer also shares the target
        // NVFP4 LM head; keep the BF16 LM head when target BF16 lm_head is on.
        Qwen4ReleaseMode::NativeMtp => {
            gpu.synchronize(stream)?;
            let mut packed_bytes = 0usize;
            for name in [
                "mtp.layers.0.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj",
            ] {
                if store.contains(name) {
                    let (ptr, dtype) = {
                        let weight = store.get(name)?;
                        (weight.ptr, weight.dtype)
                    };
                    packed_bytes += store.release_consumed_alias(name, ptr, dtype, gpu.as_ref())?;
                }
            }
            let (head, head_bytes) = release_unused_qwen4_bf16_lm_head(
                store,
                lm_head,
                !crate::model::impl_a3::qwen4_target_lmhead_bf16_enabled()
                    && lm_head_nvfp4.is_some(),
                gpu.as_ref(),
            )?;
            tracing::info!(
                "QWEN4_RELEASE_SPEC_UNUSED_ENGAGED mtp_packed_mib={} lm_head_bf16_mib={}",
                packed_bytes >> 20,
                head_bytes >> 20
            );
            head
        }
        Qwen4ReleaseMode::None => {
            let memlean_lm_head = crate::memlean::selected(crate::memlean::FP8_SOURCES)?
                || crate::memlean::selected(crate::memlean::SSM_BF16_QKVZ)?
                || crate::memlean::selected(crate::memlean::UNUSED_MTP)?;
            let has_dflash_without_donor = dflash_args
                .as_ref()
                .is_some_and(|args| args.donor_store.is_none());
            if should_release_unused_dense_bf16_lm_head(
                config.is_qwen4_exp(),
                config.num_experts,
                lm_head_nvfp4.is_some(),
                has_dflash_without_donor,
                lm_head.weight.0,
                embed.weight.0,
                memlean_lm_head,
            ) {
                gpu.synchronize(stream)?;
                let (head, head_bytes) =
                    release_unused_qwen4_bf16_lm_head(store, lm_head, true, gpu.as_ref())?;
                if head_bytes != 0 {
                    tracing::info!(
                        "MEMLEAN_DENSE_BF16_LM_HEAD_RELEASED lm_head_bf16_mib={}",
                        head_bytes >> 20
                    );
                }
                head
            } else {
                lm_head
            }
        }
    };
    drop(release_receipt);
    startup_ledger::record(
        "release-unused",
        format!("{release_mode:?}: BF16 lm_head / mtp.* aliases freed (negative = reclaimed)"),
        None,
        startup_ledger::signed_delta(
            release_mem_before,
            startup_ledger::available_device_memory_bytes(),
        ),
    );
    // After the last memlean release (layers, final mixers, LM head).
    crate::source_pins::emit_totals();

    // ── Step 3b: Post-load MoE prefill transpose (MiniMax EP=2 TTFT fix) ──
    //
    // MiniMax M2.7-NVFP4 EP=2 has ~46 GB free at layer-0 load time but
    // ~65 GB free here (the BF16 lm_head just freed ~22 GB during NVFP4
    // quantization). The transpose costs ~59 GB — fits in the post-load
    // window but not the pre-load one. Other loaders (qwen35, qwen3,
    // gemma4) still call `transpose_for_prefill` inline during layer
    // construction; this default-no-op hook doesn't perturb them. Qwen4
    // already ran its unified pass right after `load_layers` (above).
    if !config.is_qwen4_exp() {
        maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;
    }
    maybe_setup_qwen4_stream_t(&config, gpu.as_ref(), &mut layers)?;
    // ── Step 4: Create buffer arena ──
    let arena_requested = BufferSizes::from_config(
        &config,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        max_batch_size,
    )?
    .total_bytes()?;
    let buffers = startup_ledger::measure(
        "buffer-arena",
        format!(
            "max_batch_tokens {max_batch_tokens}, max_seq_len {max_seq_len}, batch {max_batch_size}"
        ),
        Some(arena_requested),
        || {
            BufferArena::new(
                &config,
                max_batch_tokens,
                max_seq_len,
                kv_block_size,
                max_batch_size,
                gpu.as_ref(),
            )
        },
    )?;
    // Qwen4 prefill scratch that must never grow (and free) mid-request.
    startup_ledger::measure(
        "prefill-scratch",
        format!("GDN FI bridge + QSA sparse + QSA q8 for max_batch_tokens {max_batch_tokens}"),
        None,
        || {
            maybe_reserve_qwen4_prefill_scratch(
                &config,
                gpu.as_ref(),
                max_batch_tokens,
                max_seq_len,
            )
        },
    )?;

    // ── Step 5: Size KV cache from actual free memory ──
    // MLA absorbed: cache compressed latent [kv_lora + rope] instead of expanded [nkv * hd]
    // This gives 12.8x smaller KV cache AND better precision (no expand→cache→read roundtrip)
    let (kv_num_heads, kv_head_dim) = if config.kv_lora_rank > 0 {
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        tracing::info!(
            "MLA absorbed KV cache: 1 head × {} dims ({}+{}) per token (vs {} heads × {})",
            mla_cache_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.num_key_value_heads,
            config.head_dim,
        );
        (1, mla_cache_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let kv_config = KvCacheConfig {
        block_size: kv_block_size,
        num_kv_heads: kv_num_heads,
        head_dim: kv_head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: layer_dtypes.clone(),
        layer_dims: config.kv_layer_dims.clone(),
        cache_blocks_per_seq: hss_cache_blocks_per_seq,
    };

    // KV-dtype gating for `--high-speed-swap`.
    //
    // All quantization variants are now supported via host-side dequant before
    // disk-write (the orchestrator's tiled-attention kernel reads BF16):
    //   - BF16    : direct stream; predictor anchor (K_lr) computed natively.
    //   - FP8     : E4M3 → BF16 (per-tensor calibration scale). Predictor
    //               degrades to LRU (BF16-only kernel can't read FP8 layout).
    //   - NVFP4   : E2M1 nibble + per-group FP8 scale → BF16. Predictor LRU.
    //   - Turbo4  : Lloyd-Max 16-level + per-group FP8 scale + WHT(K/V) on
    //               disk. Decode flow's WHT(Q)/iWHT(out) bookends handle the
    //               Walsh-Hadamard round-trip transparently. Predictor LRU.
    //   - Turbo3  : 3-bit packed (8 vals per 3 bytes), 8-level codebook,
    //               per-group FP8 scales, WHT bookended. Predictor LRU.
    //   - Turbo8  : FP8 E4M3 + per-group FP8 scales + WHT bookended.
    //               Predictor LRU.
    if hss_cache_blocks_per_seq.is_some() {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        if kv_config.layer_dtypes.is_empty() {
            *counts
                .entry(exact_kv_dtype_name(kv_config.dtype))
                .or_default() += kv_config.num_layers;
        } else {
            for dt in &kv_config.layer_dtypes {
                *counts.entry(exact_kv_dtype_name(*dt)).or_default() += 1;
            }
        }
        let total: usize = counts.values().sum();
        let summary: Vec<String> = counts
            .iter()
            .map(|(name, n)| format!("{n} {name}"))
            .collect();
        tracing::info!(
            "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
             (host dequant for FP8/NVFP4/Turbo2/Turbo3/Turbo4/Turbo8; \
             predictor scoring uses LRU for non-BF16 layers)",
            total,
            summary.join(" + ")
        );
    }
    let actual_free = gpu.free_memory()?;
    let allocatable = actual_free.saturating_sub(inference_reserve);
    let kv_budget = (allocatable as f64 * gpu_memory_utilization) as usize;
    // When HBM-shrink is active, size the production cache to
    // `max_batch_size × cache_blocks_per_seq` rather than the unbounded
    // budget-driven sum. This is the *whole point* of the HBM-shrink
    // feature — the production cache becomes write staging only; older
    // blocks live on disk under the orchestrator's control.
    let (num_kv_blocks, kv_placement) = match hss_cache_blocks_per_seq {
        Some(cap) => {
            // Pool = max_batch × cap + 1 dummy + 1 spare per seq.
            // Reasons:
            //   * +1 dummy: the dummy_kv_block (allocated once at model init,
            //     used for OOB-safe paged-kernel reads) permanently consumes
            //     one slot.
            //   * +1 spare per seq: the slide-then-alloc round-trip in
            //     `ensure_blocks_through_decode` needs the just-freed block
            //     back from the LIFO free list. With exactly cap blocks, the
            //     last grow-to-cap step has zero free blocks, and the next
            //     step's alloc fires before the slide can free one (the loop
            //     orders slide-before-alloc, but alloc-without-slide hits the
            //     final block first).
            //
            // The +1-per-seq covers the gap between bt_len=cap-1 (last
            // pre-slide alloc) and bt_len=cap (slide-then-alloc steady state).
            let n = max_batch_size * (cap as usize + 1) + 1;
            tracing::info!(
                "--high-speed-swap: HBM cache sized to {n} blocks ({} batch × ({cap}+1 spare) + 1 dummy); \
                 older blocks stream from disk via the orchestrator",
                max_batch_size
            );
            (n, spark_runtime::kv_cache::KvPlacement::default())
        }
        None => {
            let raw_budget_blocks = PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;
            let bytes_per_block = kv_config.block_bytes_kv_all_layers();
            let (kv_pool_uncapped, explicit_max_tokens) = parse_kv_pool_env_overrides(
                std::env::var(KV_POOL_UNCAPPED_ENV).ok().as_deref(),
                std::env::var(KV_MAX_TOKENS_ENV).ok().as_deref(),
            )?;

            // Cap the pool at what the configured sequences + retained prefix
            // cache can address, bounded strictly by `budget_blocks` AND a hard
            // 16 GiB GB10 UMA `MemAvailable` safety floor (+ Step-6 state reserve)
            // so widened prefix pools never trigger host `earlyoom`.
            //
            // Same shape as the hss cap: +1 spare block per sequence for the
            // slide-then-alloc round trip, +1 dummy block for OOB-safe reads.
            let blocks_per_seq = max_seq_len.div_ceil(kv_block_size);
            let _active_reachable = max_batch_size
                .saturating_mul(blocks_per_seq.saturating_add(1))
                .saturating_add(1);
            let is_unified_memory = gpu.is_unified_memory();
            let total_vram = gpu.total_memory().unwrap_or(actual_free);
            let declared_floor = if is_unified_memory {
                startup_ledger::memavailable_floor_bytes().unwrap_or(0)
            } else {
                0
            };
            let (guard_mem_available, safety_floor_bytes) = kv_pool_memory_guard(
                is_unified_memory,
                actual_free,
                total_vram,
                is_unified_memory
                    .then(startup_ledger::mem_available_bytes)
                    .flatten(),
                declared_floor,
            );
            let aux_block_bytes = qwen4_qsa_auxiliary_block_bytes(&config, &kv_config)?;
            let step6_reserve_bytes =
                estimate_step6_state_reserve_bytes(&config, max_batch_size, ssm_cache_slots);
            let sizing_plan = plan_kv_pool_blocks(KvPoolSizingInput {
                budget_blocks: raw_budget_blocks,
                bytes_per_block_all_layers: bytes_per_block.saturating_add(aux_block_bytes),
                kv_block_size,
                max_seq_len,
                max_batch_size,
                prefix_cache_active: prefix_cache.is_active(),
                ssm_cache_slots,
                kv_pool_uncapped,
                explicit_max_tokens,
                host_mem_available_bytes: guard_mem_available,
                uma_safety_floor_bytes: safety_floor_bytes,
                downstream_state_reserve_bytes: step6_reserve_bytes,
            });
            let budget_blocks = sizing_plan.effective_budget_blocks;
            let reachable = sizing_plan.reachable_blocks;
            let n = if kv_pool_tokens.is_some() {
                let pool = super::kv_pool::plan_kv_pool_with_prefix_cache(
                    budget_blocks,
                    max_seq_len,
                    max_batch_size,
                    kv_block_size,
                    kv_pool_tokens,
                    prefix_cache.is_active(),
                )?;
                tracing::info!(
                    "KV_POOL mode=shared kv_pool_tokens={} blocks={} ({} tok/block, incl. {} spare + 1 dummy) \
                     slots={} max_seq_len={} per_slot_blocks_would_be={} budget_blocks={} prefix_cache={}",
                    kv_pool_tokens.unwrap_or(0),
                    pool.blocks,
                    kv_block_size,
                    max_batch_size.max(1),
                    max_batch_size,
                    max_seq_len,
                    super::kv_pool::reachable_blocks(max_seq_len, max_batch_size, kv_block_size),
                    budget_blocks,
                    prefix_cache.is_active(),
                );
                pool.blocks
            } else {
                let n = budget_blocks.min(reachable);
                if n < raw_budget_blocks {
                    tracing::info!(
                        "KV cache capped to addressable size: {} blocks ({} effective seqs × \
                         ({} blocks/seq + 1 spare) + 1 dummy, uma_capped={}) instead of the \
                         budget-derived {} blocks — saves {:.1} GB",
                        n,
                        sizing_plan.effective_seq_capacity,
                        blocks_per_seq,
                        sizing_plan.uma_floor_clamped,
                        raw_budget_blocks,
                        (raw_budget_blocks - n) as f64
                            * (kv_budget as f64 / raw_budget_blocks.max(1) as f64)
                            / (1024.0 * 1024.0 * 1024.0),
                    );
                }
                n
            };
            // The GB10 display carveout adds blocks on top of `n` at zero
            // `MemAvailable` cost (`kv_carveout::extend` is a no-op when absent).
            let kv_shape = kv_carveout::KvShape {
                config: &kv_config,
                v_aliases_k: false,
                auxiliary_block_bytes: 0,
            }
            .with_per_block_aux(aux_block_bytes)?;
            let (n, kv_placement) = kv_carveout::extend(gpu.as_ref(), kv_shape, n);
            let max_kv_tokens = n * kv_block_size;
            let kv_gib = (n.saturating_mul(bytes_per_block)) as f64 / (1024.0 * 1024.0 * 1024.0);
            let (kv_dtype_desc, turboquant_active) = format_kv_pool_dtype_summary(&kv_config);
            tracing::info!(
                "KV cache (post-construction): {:.1} GB free, {:.1} GB allocatable, \
                 {} blocks × {} tok/block = {} max tokens",
                actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
                allocatable as f64 / (1024.0 * 1024.0 * 1024.0),
                n,
                kv_block_size,
                max_kv_tokens,
            );
            tracing::info!(
                "KV_POOL_RECEIPT total_kv_gib={kv_gib:.2} max_kv_tokens={max_kv_tokens} \
                 kv_blocks={n} kv_dtype=\"{kv_dtype_desc}\" turboquant_compression={} \
                 prefix_caching={} shared_snapshot_slots={ssm_cache_slots} uma_reserve_gib={:.1}",
                if turboquant_active {
                    "TurboQuant+"
                } else {
                    "off"
                },
                prefix_cache.is_active(),
                safety_floor_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            (n, kv_placement)
        }
    };
    let _max_kv_tokens = num_kv_blocks * kv_block_size;
    // When --high-speed-swap is on with HBM-shrink, the
    // production KV cache only has to fit the per-seq HBM window, not the full
    // sequence (older blocks live on disk). Compare against `cache_blocks_per_seq`
    // in that mode; the legacy "blocks per max_seq_len" check is invalid for
    // HBM-shrunk pools by design.
    let blocks_per_seq = match hss_cache_blocks_per_seq {
        Some(cap) => cap as usize,
        None => max_seq_len.div_ceil(kv_block_size),
    };
    let max_concurrent = num_kv_blocks / blocks_per_seq.max(1);
    // A shared pool (`--kv-pool-tokens`) was already checked to hold one full
    // sequence; admission keeps the in-flight reservations inside it.
    if max_concurrent < max_batch_size && kv_pool_tokens.is_none() {
        // Suggest a max_seq_len that lets the requested batch size fit.
        let suggested_max_seq_len = (num_kv_blocks / max_batch_size.max(1)) * kv_block_size;
        anyhow::bail!(
            "KV cache can hold at most {} concurrent sequence(s) at --max-seq-len={}, \
             but --max-batch-size={} was requested. \
             KV pool has {} block(s) of {} tokens each; each sequence needs {} block(s). \
             Try --max-seq-len {} (keeps max_batch_size={}) or reduce --max-batch-size.",
            max_concurrent,
            max_seq_len,
            max_batch_size,
            num_kv_blocks,
            kv_block_size,
            blocks_per_seq,
            suggested_max_seq_len.max(kv_block_size),
            max_batch_size,
        );
    }
    let kv_requested = num_kv_blocks.saturating_mul(kv_config.block_bytes_kv_all_layers());
    let kv_cache = if kv_placement.is_empty() {
        startup_ledger::measure(
            "kv-cache",
            format!(
                "{num_kv_blocks} blocks x {} tok x {} layers",
                kv_block_size, kv_config.num_layers
            ),
            Some(kv_requested),
            || PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref()),
        )?
    } else {
        let carved = kv_placement.carved_bytes(&PagedKvCache::buffer_sizes(
            &kv_config,
            num_kv_blocks,
            false,
        ));
        let sys_requested = kv_requested.saturating_sub(carved);
        startup_ledger::measure(
            "kv-cache",
            format!(
                "{num_kv_blocks} blocks x {} tok x {} layers ({} in display carveout)",
                kv_block_size,
                kv_config.num_layers,
                kv_placement.len()
            ),
            Some(sys_requested),
            || {
                PagedKvCache::new_placed(
                    kv_config,
                    num_kv_blocks,
                    gpu.as_ref(),
                    false,
                    kv_placement,
                )
            },
        )?
    };
    startup_ledger::measure(
        "qsa-fp8-shadows",
        "E4M3 K/V shadows over the target KV pool",
        None,
        || maybe_reserve_qwen4_target_shadows(&config, gpu.as_ref(), &kv_cache),
    )?;

    // ── Step 6: Assemble model ──
    // Capture pointers for any post-construction sharing (DFlash drafter
    // shares embed_tokens + lm_head with the target). DenseWeight is Copy
    // so this clones the device pointer cheaply.
    let (target_embed_for_dflash, target_lm_head_for_dflash) = if let Some(args) =
        dflash_args.as_ref()
        && let Some(donor) = args.donor_store
    {
        let hidden = args.drafter_config.hidden_size;
        let vocab = args.drafter_config.vocab_size;
        let embed = donor_tensor(
            donor,
            &[
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            ],
            &[vocab, hidden],
            "embedding",
        )?;
        let lm_head = donor_tensor(
            donor,
            &["lm_head.weight", "model.lm_head.weight"],
            &[vocab, hidden],
            "lm_head",
        )?;
        (embed.ptr, lm_head.ptr)
    } else {
        (embed.weight, lm_head.weight)
    };
    // DFlash trains against the target model's exposed intermediate states.
    // Qwen4 exposes the full four-stream hyperconnection row (4H); ordinary
    // targets have residual_width()==hidden_size, preserving their ABI.
    let target_hidden_for_dflash = if config.dflash_capture_width > 0 {
        config.dflash_capture_width
    } else {
        config.residual_width()
    };
    // Honor --mtp-vocab for the DFlash drafter lm_head, mirroring the MTP
    // head: drafts only need argmax over the high-frequency vocab prefix,
    // and the full 248k-row lm_head GEMM at M=γ+1 dominates the propose
    // tail (~32ms of 67ms propose at ctx≈390).
    // mtp_vocab_size=0 means uncapped.
    let native_dflash2_selector_vocab = dflash_args
        .as_ref()
        .filter(|args| {
            args.drafter_config.is_dflash2()
                && native_qwen38_flash_next_profile(&config, &args.drafter_config)
                    .is_some_and(|profile| profile.proposal_vocab.is_some())
        })
        .map(|args| resolve_native_dflash2_proposal_vocab(&args.drafter_config, mtp_vocab_size))
        .transpose()?;
    let target_vocab_for_dflash = if let Some(selector_vocab) = native_dflash2_selector_vocab {
        selector_vocab
    } else if mtp_vocab_size > 0 {
        (mtp_vocab_size as usize).min(config.vocab_size)
    } else {
        config.vocab_size
    };

    if config.is_qwen4_exp() {
        // Qwen4 QSA prefill writes KV + compressed selector keys starting at
        // `ssm_snapshot_tokens` without a `kv_write_start` skip. Enforce
        // `PC_WRITE_FLOOR` on paired Radix lookups so suffix replay allocates
        // private blocks instead of mutating live shared cached blocks.
        prefix_cache.set_paired_truncate_to_snapshot_floor(true);
    }

    let model_state_mem_before = startup_ledger::available_device_memory_bytes();
    let model_state_receipt = startup_ledger::scope_guard("model-state-pools");
    let mut model = TransformerModel::new(
        config,
        embed,
        final_norm,
        qwen4_final_mixer,
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        qwen4_ple,
        lm_head,
        lm_head_nvfp4,
        layers,
        buffers,
        kv_cache,
        mtp_weights,
        mtp_dense_weights,
        qwen4_mtp_proposer,
        gpu,
        max_seq_len,
        max_batch_size,
        effective_mtp_quant,
        use_native_mtp,
        prefix_cache,
        mtp_vocab_size,
        comm,
        self_speculative,
        num_drafts,
        vision_encoder,
        ssm_cache_slots,
        ssm_checkpoint_interval,
    )?;
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    model.initialize_qwen4_ple_prefill(max_batch_tokens)?;
    if let Some(fp8) = lm_head_fp8 {
        model.set_lm_head_fp8(fp8);
    }
    drop(model_state_receipt);
    startup_ledger::record(
        "model-state-pools",
        "SSM pool + rollback snapshot + PLE history + decode scratch (TransformerModel::new)",
        None,
        startup_ledger::signed_delta(
            model_state_mem_before,
            startup_ledger::available_device_memory_bytes(),
        ),
    );

    // ── Step 7: DFlash drafter (optional, post-construction) ──
    //
    // Loaded last because it depends on the target's `embed_tokens` and
    // `lm_head` pointers (the drafter checkpoint omits these — they're
    // shared at runtime, mirroring vLLM PR #40898's `skip_substrs` flow).
    if let Some(args) = dflash_args {
        let drafter_receipt = startup_ledger::scope_guard("dflash-drafter");
        let drafter_mem_before = startup_ledger::available_device_memory_bytes();
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1, // tp_size for the drafter side: replicated, so always 1
        )?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "DFlash was explicitly requested, but the drafter checkpoint contains no \
                 DFlash weight schema (`fc.weight` or `model.fc.weight`)"
            )
        })?;
        let mut head = crate::layers::BlockDiffusionDraftHead::from_weights(
            weights,
            target_embed_for_dflash,
            target_lm_head_for_dflash,
            target_hidden_for_dflash,
            target_vocab_for_dflash,
            args.gamma,
            model.ddtree_parent_ids_capacity,
            args.dspark_verify_mode,
            args.window_size,
            model.gpu_backend(),
            max_seq_len,
            args.quantization,
            args.draft_sample_method,
            args.probabilistic_qualification_enabled,
        )?;
        maybe_release_drafter_bf16_sources(args.drafter_store, &mut head, model.gpu_backend())?;
        // Share the target's NVFP4-T lm_head (ATLAS_LM_HEAD_T) with the
        // drafter's propose lm_head fast path (gated at the call site by
        // ATLAS_DFLASH_LM_HEAD_NVFP4=1). Same device allocation — the
        // drafter reads the --mtp-vocab column prefix via ldb.
        if args.donor_store.is_none()
            && let Some((t, ldb)) = model.dflash_lm_head_t()
        {
            head.lm_head_shared_t = Some(t);
            head.lm_head_shared_t_ldb = ldb;
        }
        // Share the target's already-resident untransposed NVFP4 lm_head and
        // exact multi-row kernel bundle (zero extra GPU allocation; pure
        // KernelLaunch so 100% CUDA-graph capture-safe).
        if args.donor_store.is_none()
            && head.hidden_size == head.target_hidden_size
            && let Some((nvfp4, exact_kernels)) = model.dflash_lm_head_nvfp4_exact()
        {
            head.lm_head_shared_nvfp4_exact = Some(nvfp4);
            head.lm_head_exact_kernels = Some(exact_kernels);
        }
        model.set_dflash_proposer(std::sync::Arc::new(head))?;
        startup_ledger::record(
            "dflash-drafter",
            "drafter construction; its store loads before the target (negative = reclaimed)",
            None,
            startup_ledger::signed_delta(
                drafter_mem_before,
                startup_ledger::available_device_memory_bytes(),
            ),
        );
        drop(drafter_receipt);
        tracing::info!("DFlash drafter installed as the active proposer");
    }

    Ok(Box::new(model))
}

#[cfg(test)]
mod early_mtp_release_tests {
    use super::{
        maybe_release_unused_qwen4_mtp_early, release_unused_qwen4_bf16_lm_head,
        release_unused_qwen4_mtp,
    };
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Default)]
    struct ReleaseGpu {
        inner: MockGpuBackend,
        fail_sync: AtomicBool,
        fail_free_at: AtomicUsize,
        free_calls: AtomicUsize,
        operations: Mutex<Vec<&'static str>>,
    }

    impl GpuBackend for ReleaseGpu {
        fn alloc(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> anyhow::Result<()> {
            self.operations.lock().unwrap().push("free");
            let call = self.free_calls.fetch_add(1, Ordering::SeqCst) + 1;
            anyhow::ensure!(
                call != self.fail_free_at.load(Ordering::SeqCst),
                "injected MTP free failure"
            );
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> anyhow::Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> anyhow::Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> anyhow::Result<()> {
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> anyhow::Result<()> {
            self.operations.lock().unwrap().push("synchronize");
            anyhow::ensure!(
                !self.fail_sync.load(Ordering::SeqCst),
                "injected MTP synchronize failure"
            );
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }
        fn kernel(&self, module: &str, name: &str) -> anyhow::Result<KernelHandle> {
            self.inner.kernel(module, name)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> anyhow::Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(
            &self,
            ptr: DevicePtr,
            value: u8,
            bytes: usize,
            stream: u64,
        ) -> anyhow::Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> anyhow::Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> anyhow::Result<usize> {
            self.inner.free_memory()
        }
    }

    fn fixture(gpu: &ReleaseGpu) -> WeightStore {
        let mut weights = HashMap::new();
        for (name, shape, dtype) in [
            ("mtp.fc_embedding.weight", vec![2, 2], WeightDtype::BF16),
            ("mtp.layers.0.scalar", vec![], WeightDtype::FP32),
            (
                "model.language_model.layers.0.target",
                vec![2],
                WeightDtype::BF16,
            ),
            ("lm_head.weight", vec![2, 2], WeightDtype::BF16),
        ] {
            let bytes = shape.iter().product::<usize>() * dtype.byte_size();
            weights.insert(
                name.to_string(),
                WeightTensor {
                    ptr: gpu.alloc(bytes).unwrap(),
                    shape,
                    dtype,
                },
            );
        }
        WeightStore::from_map(weights)
    }

    #[test]
    fn early_mtp_release_gate_is_exhaustively_plain_only() {
        for is_qwen4 in [false, true] {
            for use_native_mtp in [false, true] {
                for has_dflash in [false, true] {
                    for release_plain in [false, true] {
                        let gpu = ReleaseGpu::default();
                        let store = fixture(&gpu);
                        let result = maybe_release_unused_qwen4_mtp_early(
                            &store,
                            is_qwen4,
                            use_native_mtp,
                            has_dflash,
                            release_plain,
                            &gpu,
                        )
                        .unwrap();
                        if is_qwen4 && !use_native_mtp && release_plain {
                            assert_eq!(result, (2, 12));
                            assert_eq!(gpu.inner.alloc_count(), 2);
                        } else {
                            assert_eq!(result, (0, 0));
                            assert_eq!(gpu.inner.alloc_count(), 4);
                            assert!(gpu.operations.lock().unwrap().is_empty());
                            assert!(store.contains("mtp.fc_embedding.weight"));
                        }
                        assert!(store.contains("model.language_model.layers.0.target"));
                        assert!(store.contains("lm_head.weight"));
                    }
                }
            }
        }
    }

    /// ATLAS_QWEN4_LMHEAD_BF16 pins the BF16 head even beside an NVFP4 head.
    #[test]
    fn pinned_bf16_lm_head_is_kept_and_unpinned_is_released() {
        let gpu = MockGpuBackend::new();
        let lookup_on =
            |name: &str| Ok((name == "ATLAS_QWEN4_LMHEAD_BF16").then(|| "1".to_string()));
        let plan = crate::qwen4_native_selectors::plan_from(lookup_on).unwrap();
        for (pins, kept) in [
            (
                crate::source_pins::SourcePins::from_plan(&plan, |_| false),
                true,
            ),
            (crate::source_pins::SourcePins::none(), false),
        ] {
            let ptr = gpu.alloc(64).unwrap();
            let store = WeightStore::from_map(HashMap::from([(
                "lm_head.weight".to_string(),
                WeightTensor {
                    ptr,
                    shape: vec![32],
                    dtype: WeightDtype::BF16,
                },
            )]));
            let _scope = crate::source_pins::scoped(pins);
            let head = crate::weight_map::DenseWeight { weight: ptr };
            let (head, bytes) =
                release_unused_qwen4_bf16_lm_head(&store, head, true, &gpu).unwrap();
            assert_eq!(store.get("lm_head.weight").is_ok(), kept);
            assert_eq!(head.weight.is_null(), !kept);
            assert_eq!(bytes == 0, kept);
        }
    }

    #[test]
    fn early_mtp_release_syncs_tombstones_and_late_repeat_is_noop() {
        let gpu = ReleaseGpu::default();
        let store = fixture(&gpu);
        assert_eq!(release_unused_qwen4_mtp(&store, &gpu).unwrap(), (2, 12));
        assert_eq!(
            *gpu.operations.lock().unwrap(),
            ["synchronize", "free", "free"]
        );
        assert!(store.get("mtp.fc_embedding.weight").is_err());
        assert!(store.get("mtp.layers.0.scalar").is_err());
        assert!(!store.contains("mtp.fc_embedding.weight"));
        assert!(store.names().all(|name| !name.starts_with("mtp.")));
        assert_eq!(store.len(), 2);
        assert_eq!(store.total_bytes(), 12);
        assert_eq!(release_unused_qwen4_mtp(&store, &gpu).unwrap(), (0, 0));
        assert_eq!(gpu.operations.lock().unwrap().len(), 3);
        assert_eq!(gpu.inner.alloc_count(), 2);
    }

    #[test]
    fn dense_bf16_lm_head_release_policy_guards_dflash_and_tied_embeddings() {
        use super::should_release_unused_dense_bf16_lm_head;

        // Dense untied model with NVFP4 head + MemLean active releases BF16 lm_head.
        assert!(should_release_unused_dense_bf16_lm_head(
            false, 0, true, false, 0x2000, 0x1000, true
        ));
        // Donor-less DFlash shares BF16 lm_head: must NOT release.
        assert!(!should_release_unused_dense_bf16_lm_head(
            false, 0, true, true, 0x2000, 0x1000, true
        ));
        // Tied embeddings (`lm_head_ptr == embed_ptr`): must NOT release.
        assert!(!should_release_unused_dense_bf16_lm_head(
            false, 0, true, false, 0x1000, 0x1000, true
        ));
        // No NVFP4 head (e.g. skip_lm_head_quantization or BF16 control): must NOT release.
        assert!(!should_release_unused_dense_bf16_lm_head(
            false, 0, false, false, 0x2000, 0x1000, true
        ));
        // MemLean inactive (GB10 UMA default): must NOT release.
        assert!(!should_release_unused_dense_bf16_lm_head(
            false, 0, true, false, 0x2000, 0x1000, false
        ));
        // Qwen4 or MoE models keep their own release modes.
        assert!(!should_release_unused_dense_bf16_lm_head(
            true, 0, true, false, 0x2000, 0x1000, true
        ));
        assert!(!should_release_unused_dense_bf16_lm_head(
            false, 128, true, false, 0x2000, 0x1000, true
        ));
    }

    #[test]
    fn early_mtp_release_sync_failure_does_not_free_or_tombstone() {
        let gpu = ReleaseGpu::default();
        let store = fixture(&gpu);
        gpu.fail_sync.store(true, Ordering::SeqCst);
        assert!(release_unused_qwen4_mtp(&store, &gpu).is_err());
        assert_eq!(*gpu.operations.lock().unwrap(), ["synchronize"]);
        assert_eq!(gpu.inner.alloc_count(), 4);
        assert_eq!(store.len(), 4);
        assert!(store.get("mtp.fc_embedding.weight").is_ok());
        assert!(store.get("mtp.layers.0.scalar").is_ok());
    }

    #[test]
    fn early_mtp_release_partial_free_failure_rolls_back_only_failed_alias() {
        let gpu = ReleaseGpu::default();
        let store = fixture(&gpu);
        gpu.fail_free_at.store(2, Ordering::SeqCst);
        assert!(release_unused_qwen4_mtp(&store, &gpu).is_err());
        assert_eq!(gpu.inner.alloc_count(), 3);
        assert_eq!(store.len(), 3);
        let remaining: Vec<String> = store
            .names()
            .filter(|name| name.starts_with("mtp."))
            .map(String::from)
            .collect();
        assert_eq!(remaining.len(), 1);
        let remaining_bytes = store.get(&remaining[0]).unwrap().byte_size();
        assert_eq!(
            release_unused_qwen4_mtp(&store, &gpu).unwrap(),
            (1, remaining_bytes)
        );
        assert_eq!(gpu.inner.alloc_count(), 2);
        assert!(store.names().all(|name| !name.starts_with("mtp.")));
        assert!(store.contains("model.language_model.layers.0.target"));
        assert!(store.contains("lm_head.weight"));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen4DflashPairing, classify_qwen4_dflash_pairing, is_native_qwen38_flash_next_drafter,
        is_qwen38_flash_next_target_vocab, remap_capture_depths,
        resolve_native_dflash2_proposal_vocab, validate_dedicated_model_dflash_boundary,
        validate_native_qwen4_dflash_width,
    };
    use atlas_core::config::ModelConfig;

    #[test]
    fn dedicated_deepseek_factory_never_discards_external_dflash() {
        assert!(validate_dedicated_model_dflash_boundary("deepseek_v41", false).is_ok());
        assert!(validate_dedicated_model_dflash_boundary("deepseek_v41", true).is_err());
        assert!(validate_dedicated_model_dflash_boundary("qwen4_exp", false).is_ok());
        assert!(validate_dedicated_model_dflash_boundary("qwen4_exp", true).is_ok());
    }

    fn flash_next_target() -> ModelConfig {
        let mut target = ModelConfig::qwen3_next_80b_nvfp4();
        target.model_type = "qwen4_exp".to_string();
        target.hidden_size = 2560;
        target.hc_count = 4;
        target.vocab_size = 248_320;
        target.num_hidden_layers = 48;
        target
    }

    fn native_v3_json(block_size: usize) -> String {
        format!(
            r#"{{
                "architectures":["DFlashDraftModel"],
                "model_type":"qwen3",
                "hidden_size":2560,
                "num_hidden_layers":6,
                "num_target_layers":48,
                "intermediate_size":8704,
                "num_attention_heads":20,
                "num_key_value_heads":4,
                "head_dim":128,
                "vocab_size":248320,
                "block_size":{block_size},
                "tie_word_embeddings":false,
                "is_causal":false,
                "dflash_config":{{
                    "block_size":{block_size},
                    "mask_token_id":248077,
                    "target_layer_ids":[1,7,13,20,26,33,39,46]
                }}
            }}"#
        )
    }

    fn native_dflash2_json() -> &'static str {
        r#"{
            "architectures":["DFlash2DraftModel"],
            "model_type":"qwen3",
            "hidden_size":2560,
            "num_hidden_layers":6,
            "num_target_layers":48,
            "intermediate_size":8704,
            "num_attention_heads":20,
            "num_key_value_heads":4,
            "head_dim":128,
            "vocab_size":248320,
            "block_size":16,
            "tie_word_embeddings":false,
            "is_causal":false,
            "layer_types":[
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","full_attention"
            ],
            "sliding_window":4096,
            "dflash_config":{
                "block_size":16,
                "mask_token_id":248077,
                "target_layer_ids":[1,7,13,20,26,33,39,46],
                "conv_kernel_size":2,
                "conv_group_size":16,
                "selector_rank":256,
                "selector_top_k":16,
                "selector_vocab_size":248077
            }
        }"#
    }

    #[test]
    fn remaps_dense_qwen38_capture_depths_to_flash_next() {
        let mapped =
            remap_capture_depths(&[1, 10, 18, 27, 35, 44, 52, 61], 64, 48).expect("valid bridge");
        assert_eq!(mapped, [1, 7, 13, 20, 26, 33, 39, 46]);
    }

    #[test]
    fn rejects_capture_outside_donor_depth() {
        assert!(remap_capture_depths(&[64], 64, 48).is_err());
    }

    #[test]
    fn native_qwen4_dflash_requires_no_donor() {
        assert_eq!(
            classify_qwen4_dflash_pairing(true, false).unwrap(),
            Qwen4DflashPairing::Native
        );
        assert!(classify_qwen4_dflash_pairing(true, true).is_err());
    }

    #[test]
    fn dense_qwen4_dflash_requires_donor() {
        assert_eq!(
            classify_qwen4_dflash_pairing(false, true).unwrap(),
            Qwen4DflashPairing::DenseDonorBridge
        );
        assert!(classify_qwen4_dflash_pairing(false, false).is_err());
    }

    #[test]
    fn native_qwen38_geometry_is_exact() {
        let target = flash_next_target();
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(16))
                .expect("valid native drafter fixture");

        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_ok());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(15)).is_ok());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(14)).is_err());
        drafter.dflash_config.as_mut().unwrap().target_layer_ids[1] = 8;
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
    }

    #[test]
    fn native_qwen38_accepts_only_physical_or_canonical_logical_target_vocab() {
        assert!(is_qwen38_flash_next_target_vocab(248_320));
        assert!(is_qwen38_flash_next_target_vocab(248_077));
        assert!(!is_qwen38_flash_next_target_vocab(248_076));
        assert!(!is_qwen38_flash_next_target_vocab(248_319));

        let drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(native_dflash2_json())
                .expect("valid native DFlash2 fixture");
        let mut target = flash_next_target();
        target.vocab_size = 248_077;
        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        target.vocab_size = 248_076;
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
    }

    #[test]
    fn native_qwen38_rejects_b32_even_with_matching_gamma() {
        let target = flash_next_target();
        assert!(
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(32)).is_err()
        );
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(&native_v3_json(16))
                .expect("valid B16 fixture");
        drafter.block_size = 32;
        drafter.dflash_config.as_mut().unwrap().block_size = Some(32);

        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_err());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(31)).is_err());
        assert!(validate_native_qwen4_dflash_width(&drafter, Some(15)).is_err());
    }

    #[test]
    fn native_qwen38_dflash2_geometry_and_logical_vocab_are_exact() {
        let target = flash_next_target();
        let mut drafter =
            crate::weight_loader::dflash_loader::parse_dflash_config(native_dflash2_json())
                .expect("valid native DFlash2 fixture");

        assert!(is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(validate_native_qwen4_dflash_width(&drafter, None).is_ok());
        assert_eq!(
            resolve_native_dflash2_proposal_vocab(&drafter, 0).unwrap(),
            248_077
        );
        assert_eq!(
            resolve_native_dflash2_proposal_vocab(&drafter, 248_077).unwrap(),
            248_077
        );
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 100_000).is_err());
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 248_320).is_err());

        drafter.dflash_config.as_mut().unwrap().selector_vocab_size = Some(248_320);
        assert!(!is_native_qwen38_flash_next_drafter(&target, &drafter));
        assert!(resolve_native_dflash2_proposal_vocab(&drafter, 0).is_err());
    }
}

const UMA_PREFIX_KV_SAFETY_FLOOR_BYTES: usize = 16 * 1024 * 1024 * 1024;
const DISCRETE_PREFIX_KV_MAX_RESERVE_BYTES: usize = 1024 * 1024 * 1024;
const KV_POOL_UNCAPPED_ENV: &str = "FLUX_KV_POOL_UNCAPPED";
const KV_MAX_TOKENS_ENV: &str = "FLUX_KV_MAX_TOKENS";

/// Select the available-memory ceiling and safety reserve for prefix/KV pool sizing.
///
/// - On UMA (`is_unified_memory == true`, e.g. GB10): preserves the 16 GiB
///   `UMA_PREFIX_KV_SAFETY_FLOOR_BYTES` (or higher declared floor) checked against
///   host `/proc/meminfo` `MemAvailable`.
/// - On discrete GPUs (`is_unified_memory == false`): sizes the prefix/KV pool
///   directly from GPU VRAM (`actual_free_vram`) minus a discrete VRAM reserve of
///   `min(1 GiB, total_vram / 16)`.
fn kv_pool_memory_guard(
    is_unified_memory: bool,
    actual_free_vram: usize,
    total_vram: usize,
    host_mem_available: Option<usize>,
    declared_uma_floor: usize,
) -> (Option<usize>, usize) {
    if is_unified_memory {
        (
            host_mem_available,
            UMA_PREFIX_KV_SAFETY_FLOOR_BYTES.max(declared_uma_floor),
        )
    } else {
        let discrete_reserve = DISCRETE_PREFIX_KV_MAX_RESERVE_BYTES.min(total_vram / 16);
        (Some(actual_free_vram), discrete_reserve)
    }
}

#[derive(Clone, Copy, Debug)]
struct KvPoolSizingInput {
    budget_blocks: usize,
    bytes_per_block_all_layers: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    prefix_cache_active: bool,
    ssm_cache_slots: usize,
    kv_pool_uncapped: bool,
    explicit_max_tokens: Option<usize>,
    host_mem_available_bytes: Option<usize>,
    uma_safety_floor_bytes: usize,
    downstream_state_reserve_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KvPoolSizingPlan {
    reachable_blocks: usize,
    effective_budget_blocks: usize,
    selected_blocks: usize,
    effective_seq_capacity: usize,
    uma_floor_clamped: bool,
}

fn parse_kv_pool_env_overrides(
    uncapped_raw: Option<&str>,
    max_tokens_raw: Option<&str>,
) -> Result<(bool, Option<usize>)> {
    let uncapped = match uncapped_raw.map(str::trim) {
        None | Some("" | "0" | "false" | "FALSE" | "no" | "NO") => false,
        Some("1" | "true" | "TRUE" | "yes" | "YES") => true,
        Some(other) => {
            anyhow::bail!("{KV_POOL_UNCAPPED_ENV}={other:?} must be 0, 1, true, or false")
        }
    };
    let explicit_max_tokens = match max_tokens_raw.map(str::trim) {
        None | Some("") => None,
        Some(raw) => {
            let parsed = raw.parse::<usize>().with_context(|| {
                format!("{KV_MAX_TOKENS_ENV}={raw:?} must be a positive integer")
            })?;
            anyhow::ensure!(
                parsed > 0,
                "{KV_MAX_TOKENS_ENV} must be strictly positive when set"
            );
            Some(parsed)
        }
    };
    Ok((uncapped, explicit_max_tokens))
}

fn estimate_step6_state_reserve_bytes(
    config: &ModelConfig,
    max_batch_size: usize,
    ssm_cache_slots: usize,
) -> usize {
    let num_ssm = config.num_ssm_layers();
    let per_layer_ssm = if num_ssm > 0 {
        let h_bytes = config
            .linear_num_value_heads
            .saturating_mul(config.linear_value_head_dim)
            .saturating_mul(config.linear_key_head_dim)
            .saturating_mul(4);
        let conv_dim = config
            .linear_num_key_heads
            .saturating_mul(config.linear_key_head_dim)
            .saturating_mul(2)
            .saturating_add(
                config
                    .linear_num_value_heads
                    .saturating_mul(config.linear_value_head_dim),
            );
        let conv_bytes = conv_dim
            .saturating_mul(config.linear_conv_kernel_dim)
            .saturating_mul(4);
        h_bytes.saturating_add(conv_bytes)
    } else {
        0
    };
    let per_slot_ssm = num_ssm.saturating_mul(per_layer_ssm);
    let per_slot_ple = if config.is_qwen4_exp() {
        crate::layers::qwen4_ple::checked_ple_conv_slot_bytes(config.residual_width())
            .unwrap_or(184_320)
            .saturating_mul(config.ple_layer_ids.len().max(1))
    } else {
        0
    };
    let decode_ring = (atlas_kernels::ROLLBACK_RESTEER_CAP as usize).saturating_add(1);
    let total_slots = max_batch_size
        .saturating_mul(decode_ring.saturating_add(4))
        .saturating_add(ssm_cache_slots);
    let lazy_qwen4_prefill_reserve = if config.is_qwen4_exp()
        && std::env::var("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE")
            .ok()
            .as_deref()
            == Some("1")
    {
        // Account for the ~2.55 GiB lazy FP8 fast-proj weight cache + first-prefill
        // CUDA graph / workspace growth so widened prefix KV pools stay >= 16 GiB
        // MemAvailable even after the first wide prefill populates the cache.
        3072usize << 20
    } else {
        0
    };
    total_slots
        .saturating_mul(per_slot_ssm.saturating_add(per_slot_ple))
        .saturating_add(lazy_qwen4_prefill_reserve)
}

fn plan_kv_pool_blocks(input: KvPoolSizingInput) -> KvPoolSizingPlan {
    let block_size = input.kv_block_size.max(1);
    let blocks_per_seq = input.max_seq_len.div_ceil(block_size);
    let active_reachable = input
        .max_batch_size
        .saturating_mul(blocks_per_seq.saturating_add(1))
        .saturating_add(1);
    let index_max_blocks = (i32::MAX as usize) / block_size;
    let prefix_unlocked = input.prefix_cache_active
        || input.kv_pool_uncapped
        || input.ssm_cache_slots > 0
        || input.explicit_max_tokens.is_some();

    let (reachable_blocks, effective_seq_capacity) = if !prefix_unlocked {
        (active_reachable, input.max_batch_size)
    } else if let Some(target_tokens) = input.explicit_max_tokens {
        let target_blocks = target_tokens
            .div_ceil(block_size)
            .saturating_add(1)
            .max(active_reachable)
            .min(index_max_blocks);
        let approx_seqs = target_blocks / blocks_per_seq.max(1);
        (target_blocks, approx_seqs.max(input.max_batch_size))
    } else if input.kv_pool_uncapped {
        let approx_seqs = input.budget_blocks / blocks_per_seq.max(1);
        (index_max_blocks, approx_seqs.max(input.max_batch_size))
    } else {
        let effective_seqs = input
            .max_batch_size
            .saturating_add(input.ssm_cache_slots.max(8));
        let widened = effective_seqs
            .saturating_mul(blocks_per_seq.saturating_add(1))
            .saturating_add(1)
            .min(index_max_blocks);
        (widened, effective_seqs)
    };

    let (effective_budget_blocks, uma_floor_clamped) =
        if prefix_unlocked && let Some(mem_avail) = input.host_mem_available_bytes {
            let total_reserve = input
                .uma_safety_floor_bytes
                .saturating_add(input.downstream_state_reserve_bytes);
            let safe_headroom = mem_avail.saturating_sub(total_reserve);
            let uma_safe_blocks = if input.bytes_per_block_all_layers > 0 {
                safe_headroom / input.bytes_per_block_all_layers
            } else {
                input.budget_blocks
            };
            let clamped = uma_safe_blocks < input.budget_blocks;
            (input.budget_blocks.min(uma_safe_blocks), clamped)
        } else {
            (input.budget_blocks, false)
        };

    let selected_blocks = effective_budget_blocks.min(reachable_blocks);
    KvPoolSizingPlan {
        reachable_blocks,
        effective_budget_blocks,
        selected_blocks,
        effective_seq_capacity,
        uma_floor_clamped,
    }
}

fn is_turboquant_dtype(dt: KvCacheDtype) -> bool {
    matches!(
        dt,
        KvCacheDtype::Turbo2
            | KvCacheDtype::Turbo3
            | KvCacheDtype::Turbo4
            | KvCacheDtype::Turbo8
            | KvCacheDtype::Bf16KTurbo4V
            | KvCacheDtype::Bf16KTurbo3V
            | KvCacheDtype::Bf16KTurbo2V
            | KvCacheDtype::Fp8KTurbo4V
            | KvCacheDtype::Fp8KTurbo3V
            | KvCacheDtype::Fp8KTurbo2V
            | KvCacheDtype::Turbo3KTurbo8V
            | KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
    )
}

fn exact_kv_dtype_name(dt: KvCacheDtype) -> &'static str {
    match dt {
        KvCacheDtype::Bf16 => "BF16",
        KvCacheDtype::Fp8 => "FP8",
        KvCacheDtype::Nvfp4 => "NVFP4",
        KvCacheDtype::Turbo2 => "Turbo2",
        KvCacheDtype::Turbo3 => "Turbo3",
        KvCacheDtype::Turbo4 => "Turbo4",
        KvCacheDtype::Turbo8 => "Turbo8",
        KvCacheDtype::Bf16KTurbo4V => "BF16-K/Turbo4-V",
        KvCacheDtype::Bf16KTurbo3V => "BF16-K/Turbo3-V",
        KvCacheDtype::Bf16KTurbo2V => "BF16-K/Turbo2-V",
        KvCacheDtype::Fp8KTurbo4V => "FP8-K/Turbo4-V",
        KvCacheDtype::Fp8KTurbo3V => "FP8-K/Turbo3-V",
        KvCacheDtype::Fp8KTurbo2V => "FP8-K/Turbo2-V",
        KvCacheDtype::Turbo3KTurbo8V => "Turbo3-K/Turbo8-V",
        KvCacheDtype::Turbo4KTurbo3V => "Turbo4-K/Turbo3-V",
        KvCacheDtype::Turbo4KTurbo8V => "Turbo4-K/Turbo8-V",
    }
}

fn format_kv_pool_dtype_summary(kv_config: &KvCacheConfig) -> (String, bool) {
    if kv_config.layer_dtypes.is_empty() {
        let turbo = is_turboquant_dtype(kv_config.dtype);
        let base = exact_kv_dtype_name(kv_config.dtype);
        let text = if turbo {
            format!("{base} (TurboQuant+ compressed)")
        } else {
            base.to_string()
        };
        return (text, turbo);
    }
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    let mut turbo = false;
    for &dt in &kv_config.layer_dtypes {
        if is_turboquant_dtype(dt) {
            turbo = true;
        }
        *counts.entry(exact_kv_dtype_name(dt)).or_default() += 1;
    }
    let joined: Vec<String> = counts
        .into_iter()
        .map(|(name, n)| format!("{n}×{name}"))
        .collect();
    let base = joined.join("+");
    let text = if turbo {
        format!("{base} (TurboQuant+ compressed)")
    } else {
        base
    };
    (text, turbo)
}

#[cfg(test)]
mod kv_pool_sizing_tests {
    use super::*;

    #[test]
    fn disabled_prefix_cache_preserves_exact_addressable_cap() {
        let plan = plan_kv_pool_blocks(KvPoolSizingInput {
            budget_blocks: 200_000,
            bytes_per_block_all_layers: 221_184,
            kv_block_size: 16,
            max_seq_len: 1_000_000,
            max_batch_size: 1,
            prefix_cache_active: false,
            ssm_cache_slots: 0,
            kv_pool_uncapped: false,
            explicit_max_tokens: None,
            host_mem_available_bytes: Some(96 * 1024 * 1024 * 1024),
            uma_safety_floor_bytes: UMA_PREFIX_KV_SAFETY_FLOOR_BYTES,
            downstream_state_reserve_bytes: 0,
        });
        assert_eq!(plan.selected_blocks, 62_502);
        assert!(!plan.uma_floor_clamped);
    }

    #[test]
    fn prefix_caching_widens_reachable_pool_and_respects_16gib_uma_floor() {
        // C=1, max_seq_len=32,768 (2,048 blocks/seq), ssm_cache_slots=32 ->
        // effective_seqs = 1 + 32 = 33 -> reachable = 33 * 2049 + 1 = 67,618 blocks (~1.08M tokens!).
        let plan = plan_kv_pool_blocks(KvPoolSizingInput {
            budget_blocks: 250_000,
            bytes_per_block_all_layers: 221_184, // ~216 KiB/block (NVFP4 12 attn layers)
            kv_block_size: 16,
            max_seq_len: 32_768,
            max_batch_size: 1,
            prefix_cache_active: true,
            ssm_cache_slots: 32,
            kv_pool_uncapped: false,
            explicit_max_tokens: None,
            host_mem_available_bytes: Some(64 * 1024 * 1024 * 1024),
            uma_safety_floor_bytes: UMA_PREFIX_KV_SAFETY_FLOOR_BYTES,
            downstream_state_reserve_bytes: 4 * 1024 * 1024 * 1024,
        });
        assert_eq!(plan.effective_seq_capacity, 33);
        assert_eq!(plan.reachable_blocks, 33 * 2049 + 1);
        assert_eq!(plan.selected_blocks, 33 * 2049 + 1);
        assert!(plan.selected_blocks * 16 > 1_000_000);

        // Tight UMA host (MemAvailable = 18 GiB, floor = 16 GiB, downstream = 1 GiB -> 1 GiB safe KV):
        // must clamp strictly to UMA floor so `earlyoom` can NEVER fire!
        let tight = plan_kv_pool_blocks(KvPoolSizingInput {
            budget_blocks: 250_000,
            bytes_per_block_all_layers: 1024 * 1024, // 1 MiB/block
            kv_block_size: 16,
            max_seq_len: 32_768,
            max_batch_size: 1,
            prefix_cache_active: true,
            ssm_cache_slots: 96,
            kv_pool_uncapped: true,
            explicit_max_tokens: None,
            host_mem_available_bytes: Some(18 * 1024 * 1024 * 1024),
            uma_safety_floor_bytes: UMA_PREFIX_KV_SAFETY_FLOOR_BYTES,
            downstream_state_reserve_bytes: 1024 * 1024 * 1024,
        });
        assert!(tight.uma_floor_clamped);
        assert_eq!(tight.selected_blocks, 1024);
    }

    #[test]
    fn explicit_max_tokens_and_turboquant_receipt_formatting_work() {
        let (uncapped, max_toks) = parse_kv_pool_env_overrides(Some("0"), Some("2097152")).unwrap();
        assert!(!uncapped);
        assert_eq!(max_toks, Some(2_097_152));
        assert!(parse_kv_pool_env_overrides(Some("bad"), None).is_err());
        assert!(parse_kv_pool_env_overrides(None, Some("0")).is_err());

        let cfg = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 4,
            head_dim: 256,
            num_layers: 12,
            dtype: KvCacheDtype::Turbo4,
            layer_dtypes: vec![KvCacheDtype::Turbo4; 10]
                .into_iter()
                .chain([KvCacheDtype::Bf16; 2])
                .collect(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        };
        let (desc, turbo) = format_kv_pool_dtype_summary(&cfg);
        assert!(turbo);
        assert!(desc.contains("TurboQuant+ compressed"), "{desc}");
        assert!(desc.contains("10×Turbo4"), "{desc}");
    }

    #[test]
    fn discrete_gpu_prefix_kv_sizes_from_vram_without_16gib_uma_subtraction() {
        const MIB: usize = 1024 * 1024;
        const GIB: usize = 1024 * MIB;

        // 8 GB discrete GPU: reserve = min(1 GiB, 8 GiB / 16) = 512 MiB.
        assert_eq!(
            kv_pool_memory_guard(false, 4 * GIB, 8 * GIB, Some(4 * GIB), 6_656 * MIB),
            (Some(4 * GIB), 512 * MIB)
        );
        // 16 GB discrete GPU (RTX 5060 Ti / 5070 Ti / 5080): reserve = 1 GiB, uses VRAM not host RAM.
        let (avail_16, reserve_16) =
            kv_pool_memory_guard(false, 6 * GIB, 16 * GIB, Some(3 * GIB), 6_656 * MIB);
        assert_eq!((avail_16, reserve_16), (Some(6 * GIB), GIB));
        let plan_16 = plan_kv_pool_blocks(KvPoolSizingInput {
            budget_blocks: 20_000,
            bytes_per_block_all_layers: MIB,
            kv_block_size: 16,
            max_seq_len: 16_384,
            max_batch_size: 1,
            prefix_cache_active: true,
            ssm_cache_slots: 16,
            kv_pool_uncapped: true,
            explicit_max_tokens: None,
            host_mem_available_bytes: avail_16,
            uma_safety_floor_bytes: reserve_16,
            downstream_state_reserve_bytes: 512 * MIB,
        });
        // Safe VRAM headroom = 6 GiB - 1 GiB reserve - 512 MiB downstream = 4608 MiB -> 4608 blocks!
        assert_eq!(plan_16.selected_blocks, 4_608);

        // 32 GB / 72 GB / 96 GB discrete GPUs: reserve capped at 1 GiB, bounded by VRAM even in 256 GiB host.
        for (total_gib, free_gib) in [(32, 12), (72, 24), (96, 20)] {
            let (avail, reserve) = kv_pool_memory_guard(
                false,
                free_gib * GIB,
                total_gib * GIB,
                Some(240 * GIB),
                6_656 * MIB,
            );
            assert_eq!(avail, Some(free_gib * GIB));
            assert_eq!(reserve, GIB);
        }

        // UMA GB10 preserves the exact 16 GiB UMA safety floor and host MemAvailable source.
        assert_eq!(
            kv_pool_memory_guard(true, 40 * GIB, 128 * GIB, Some(64 * GIB), 6_656 * MIB),
            (Some(64 * GIB), UMA_PREFIX_KV_SAFETY_FLOOR_BYTES)
        );
        assert_eq!(
            kv_pool_memory_guard(true, 40 * GIB, 128 * GIB, Some(64 * GIB), 20 * GIB),
            (Some(64 * GIB), 20 * GIB)
        );
    }

    #[test]
    fn qsa_fp8_shadow_aux_budget_applies_only_to_all_bf16_kv() {
        let base = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 2,
            head_dim: 256,
            num_layers: 12,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        };
        assert!(qwen4_qsa_fp8_shadows_apply_to_kv_config(&base));

        let all_bf16_explicit = KvCacheConfig {
            layer_dtypes: vec![KvCacheDtype::Bf16; 12],
            ..base.clone()
        };
        assert!(qwen4_qsa_fp8_shadows_apply_to_kv_config(&all_bf16_explicit));

        let pure_nvfp4 = KvCacheConfig {
            dtype: KvCacheDtype::Nvfp4,
            ..base.clone()
        };
        assert!(!qwen4_qsa_fp8_shadows_apply_to_kv_config(&pure_nvfp4));

        let mixed_nvfp4 = KvCacheConfig {
            dtype: KvCacheDtype::Nvfp4,
            layer_dtypes: vec![KvCacheDtype::Bf16; 4]
                .into_iter()
                .chain(vec![KvCacheDtype::Nvfp4; 8])
                .collect(),
            ..base
        };
        assert!(!qwen4_qsa_fp8_shadows_apply_to_kv_config(&mixed_nvfp4));
    }
}

#[cfg(test)]
#[path = "native_profile_tests.rs"]
mod native_profile_tests;
