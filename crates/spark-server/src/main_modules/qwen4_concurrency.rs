// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed concurrency admission for Qwen4 multi-request serving.

use spark_model::qwen4_batch_policy::{
    self, NativeBatchSelector, NativeCapabilityReport, NativeGap, ProcessFacts,
    QWEN4_NATIVE_MAX_ROWS, Qwen4BatchPlan, SerialReason,
};

/// Serial (C1-per-row) ceiling. With native batching enabled (the default)
/// the effective ceiling is `QWEN4_NATIVE_MAX_ROWS`; see `admit_width`.
const MAX_ADMITTED_CONCURRENCY: usize = 128;

/// Opt-in serial native-MTP K5 at C2-C8 (`ATLAS_QWEN4_K5_CMULTI=1`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SerialK5Request {
    /// The exact gate value parsed to `1`.
    pub gate: bool,
    /// `--speculative` without DFlash, self-speculative or ngram speculation.
    pub native_mtp_only: bool,
    pub num_drafts: usize,
    /// `ATLAS_MTP_K2_BATCH_CSEQ`, `ATLAS_MTP_K3_BATCH_CSEQ` or
    /// `ATLAS_DFLASH_BATCHED_VERIFY` would select a cross-request verifier.
    pub cross_request_verify: bool,
}

impl SerialK5Request {
    pub(crate) const OFF: Self = Self {
        gate: false,
        native_mtp_only: false,
        num_drafts: 0,
        cross_request_verify: false,
    };
}

pub(crate) struct Qwen4Concurrency<'a> {
    pub qsa: bool,
    pub max_batch_size: usize,
    pub kv_cache_dtype: &'a str,
    pub native_context: usize,
    pub max_seq_len: usize,
    pub speculative: bool,
    /// `--speculative` with the native MTP head only (no DFlash, ngram or
    /// self-speculation).
    pub native_mtp_only: bool,
    /// `--dflash` without self-speculative or ngram speculation.
    pub dflash_only: bool,
    /// `ATLAS_QWEN4_BATCHED_K5=1`: cross-sequence batched native-MTP verify
    /// with per-slot verify lanes.
    pub batched_k5: bool,
    /// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`: the batched cohort narrows its
    /// verify width with concurrency (`K8..K2` across `C2-C128` for native MTP;
    /// up to `C16` UMA / `C64` wide-SM for batched DFlash2).
    pub batched_adaptive_k: bool,
    pub num_drafts: usize,
    pub swap_space_gb: usize,
    pub high_speed_swap: bool,
    pub prefix_caching: bool,
    pub ssm_cache_slots: usize,
    pub world_size: usize,
    pub tp_size: usize,
    pub ep_size: usize,
    pub serial_k5: SerialK5Request,
    /// `--kv-pool-tokens`: one shared KV pool instead of one max_seq_len
    /// reservation per slot.
    pub kv_pool_tokens: Option<usize>,
}

/// Maximum Qwen4 YaRN extended context admitted under C>1 concurrency (`1M` tokens).
const MAX_QWEN4_EXTENDED_CONTEXT: usize = 1_048_576;

pub(crate) fn validate_qwen4_concurrency(c: Qwen4Concurrency<'_>) -> Result<(), String> {
    if c.qsa && !matches!(c.kv_cache_dtype, "bf16" | "nvfp4") {
        return Err("the Qwen4 QSA path requires --kv-cache-dtype bf16 or nvfp4".into());
    }
    if c.max_batch_size > MAX_ADMITTED_CONCURRENCY {
        return Err(format!(
            "Qwen4 concurrency {} exceeds the admitted C{} ceiling",
            c.max_batch_size, MAX_ADMITTED_CONCURRENCY
        ));
    }
    // At C1, Marconi snapshots save and restore both the 36 GDN recurrent
    // layers and Qwen4's PLE convolution history (`ple_conv_snapshots` or
    // `slot_history` under `ATLAS_QWEN4_SEQSTATE=1`).
    if c.max_batch_size <= 1 {
        return Ok(());
    }

    if !matches!(c.kv_cache_dtype, "bf16" | "nvfp4") {
        return Err("Qwen4 concurrency requires --kv-cache-dtype bf16 or nvfp4".into());
    }
    if c.batched_adaptive_k && !c.batched_k5 {
        return Err("ATLAS_QWEN4_BATCHED_ADAPTIVE_K requires ATLAS_QWEN4_BATCHED_K5=1".into());
    }
    if c.speculative && !c.batched_k5 {
        if !c.serial_k5.gate {
            return Err(
                "Qwen4 concurrency is target-only unless native K5 is selected with ATLAS_QWEN4_K5_CMULTI=1 or ATLAS_QWEN4_BATCHED_K5=1; disable DFlash, self-speculative and ngram speculation"
                    .into(),
            );
        }
        validate_serial_k5(c.max_batch_size, c.serial_k5)?;
    }
    if c.batched_k5 {
        validate_batched_k5(&c)?;
    }
    if c.high_speed_swap {
        return Err(
            "Qwen4 concurrency requires --high-speed-swap disabled because QSA selector rings do not slide with HSS windows"
                .into(),
        );
    }
    if !matches!(c.tp_size, 1 | 2 | 4) {
        return Err(format!(
            "Qwen4 concurrency supports TP=1, TP=2, or TP=4 (got TP={}); use PP=3 for 3x Spark",
            c.tp_size
        ));
    }
    if !matches!(c.ep_size, 1 | 2 | 4) {
        return Err(format!(
            "Qwen4 concurrency supports EP=1, EP=2, or EP=4 (got EP={}); use PP=3 for 3x Spark",
            c.ep_size
        ));
    }
    if c.world_size == 0 || c.world_size > 4 {
        return Err(format!(
            "Qwen4 concurrency supports 1..=4 distributed ranks (got world_size={})",
            c.world_size
        ));
    }
    let max_admitted_seq_len = c.native_context.max(MAX_QWEN4_EXTENDED_CONTEXT);
    if c.max_seq_len > max_admitted_seq_len {
        return Err(format!(
            "Qwen4 concurrency supports max_seq_len up to {max_admitted_seq_len} (1M YaRN ceiling), got {}",
            c.max_seq_len
        ));
    }
    // Past native context under BF16 KV, every slot reserves its full max_seq_len
    // of KV at startup (2 x 524K BF16 = 24 GiB on unified memory) unless bounded
    // by a shared pool (`--kv-pool-tokens`) or 4-bit KV (`--kv-cache-dtype nvfp4`).
    if c.kv_cache_dtype == "bf16" && c.max_seq_len > c.native_context && c.kv_pool_tokens.is_none()
    {
        return Err(
            "Qwen4 BF16 concurrency past the checkpoint's native context requires --kv-pool-tokens or --kv-cache-dtype nvfp4"
                .into(),
        );
    }
    if c.swap_space_gb > 0 && c.world_size > 1 {
        return Err(format!(
            "Qwen4 multi-Spark concurrency (world_size={}) requires --swap-space-gb 0 because disk swap is single-rank only",
            c.world_size
        ));
    }
    let _ = (c.prefix_caching, c.ssm_cache_slots);
    Ok(())
}

/// Widest adaptive-K server: up to two balanced 128-row K2 sub-frames (C128 = 256 verify rows/cycle).
const ADAPTIVE_MAX_CONCURRENCY: usize = MAX_ADMITTED_CONCURRENCY;
/// Widest batched DFlash2 server on GB10 UMA (`<= 64` SMs): one 128-row K8 frame (`16 × 8 = 128`).
const DFLASH_BATCHED_MAX_CONCURRENCY_UMA: usize = 16;
/// Widest batched DFlash2 server on discrete / `> 64`-SM GPUs (e.g. SM120):
/// one 256-row frame (`32 × 8 = 256`, `48 × 5 = 240`, `64 × 4 = 256` under
/// adaptive-K; `DFLASH_BATCH_MAX_SEQS = 64`, `shared_q_row_cap = 512`).
const DFLASH_BATCHED_MAX_CONCURRENCY_WIDE_SM: usize = 64;
const DFLASH_WIDE_SM_THRESHOLD: u32 = 64;

/// Maximum batched DFlash2 concurrency for a given GPU topology (`sm_count`,
/// `is_integrated`) and optional `ATLAS_QWEN4_ADAPTIVE_MAX_ROWS` override.
///
/// GB10 UMA (`sm_count <= 64`, not discrete) stays strictly at `C=16`;
/// discrete or `> 64`-SM GPUs admit up to `C=64`.
#[must_use]
fn dflash_batched_max_concurrency_for(
    sm_count: u32,
    is_integrated: Option<bool>,
    env_max_rows: Option<&str>,
) -> usize {
    if let Some(raw) = env_max_rows
        && let Ok(rows) = raw.trim().parse::<usize>()
        && rows >= 256
    {
        return DFLASH_BATCHED_MAX_CONCURRENCY_WIDE_SM;
    }
    if sm_count > DFLASH_WIDE_SM_THRESHOLD || is_integrated == Some(false) {
        DFLASH_BATCHED_MAX_CONCURRENCY_WIDE_SM
    } else {
        DFLASH_BATCHED_MAX_CONCURRENCY_UMA
    }
}

/// Context-free CUDA driver probe for `(sm_count, is_integrated)` used when
/// `validate_qwen4_concurrency` runs at Phase 2 (config) before
/// `AtlasCudaBackend::new` populates `atlas_core::device::active_device_props`.
/// Never creates a CUDA context or allocates VRAM.
#[cfg(all(feature = "cuda", not(test)))]
fn probe_cuda_device_topology() -> Option<(u32, bool)> {
    static PROBED: std::sync::OnceLock<Option<(u32, bool)>> = std::sync::OnceLock::new();
    *PROBED.get_or_init(|| {
        const CU_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
        const CU_ATTR_INTEGRATED: i32 = 18;
        // SAFETY: `libcuda.so.1` exposes the standard C ABI CUDA driver entry points
        // (`cuInit`, `cuDeviceGet`, `cuDeviceGetAttribute`). All queries are read-only
        // and context-free.
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1").ok()?;
            let cu_init: libloading::Symbol<'_, unsafe extern "C" fn(u32) -> i32> =
                lib.get(b"cuInit\0").ok()?;
            let cu_device_get: libloading::Symbol<'_, unsafe extern "C" fn(*mut i32, i32) -> i32> =
                lib.get(b"cuDeviceGet\0").ok()?;
            let cu_get_attr: libloading::Symbol<
                '_,
                unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
            > = lib.get(b"cuDeviceGetAttribute\0").ok()?;
            if cu_init(0) != 0 {
                return None;
            }
            let mut dev: i32 = 0;
            if cu_device_get(&mut dev, 0) != 0 {
                return None;
            }
            let mut sm_count: i32 = 0;
            let mut integrated: i32 = 1;
            if cu_get_attr(&mut sm_count, CU_ATTR_MULTIPROCESSOR_COUNT, dev) != 0
                || cu_get_attr(&mut integrated, CU_ATTR_INTEGRATED, dev) != 0
            {
                return None;
            }
            Some((u32::try_from(sm_count).ok()?, integrated != 0))
        }
    })
}

fn active_dflash_batched_max_concurrency() -> usize {
    let (sm_count, is_integrated) = match atlas_core::device::active_device_props() {
        Some(props) => (props.sm_count, Some(props.is_integrated)),
        None => {
            #[cfg(all(feature = "cuda", not(test)))]
            {
                match probe_cuda_device_topology() {
                    Some((sm, integrated)) => (sm, Some(integrated)),
                    None => (
                        atlas_core::device::active_sm_count(),
                        atlas_core::device::active_device_is_integrated(),
                    ),
                }
            }
            #[cfg(any(not(feature = "cuda"), test))]
            {
                (
                    atlas_core::device::active_sm_count(),
                    atlas_core::device::active_device_is_integrated(),
                )
            }
        }
    };
    let env_max_rows =
        std::env::var(spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_ROWS_ENV).ok();
    dflash_batched_max_concurrency_for(sm_count, is_integrated, env_max_rows.as_deref())
}

fn validate_batched_k5(c: &Qwen4Concurrency<'_>) -> Result<(), String> {
    validate_batched_k5_with_dflash_cap(c, active_dflash_batched_max_concurrency())
}

fn validate_batched_k5_with_dflash_cap(
    c: &Qwen4Concurrency<'_>,
    dflash_max_concurrency: usize,
) -> Result<(), String> {
    if !c.speculative {
        return Err("ATLAS_QWEN4_BATCHED_K5 is available only for the native MTP head".into());
    }
    let batched_dflash = !c.native_mtp_only && c.batched_adaptive_k;
    if !c.native_mtp_only && !batched_dflash {
        return Err("ATLAS_QWEN4_BATCHED_K5 is available only for the native MTP head".into());
    }
    if batched_dflash && !c.dflash_only {
        return Err(
            "ATLAS_QWEN4_BATCHED_K5 with ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1 admits only native MTP or batched DFlash2; disable self-speculative and ngram speculation"
                .into(),
        );
    }
    // Fixed K5 fits C2-C25 in one 128-row frame and splits C26-C32 into two
    // balanced sub-frames. Adaptive K keeps one frame of <= 128 rows through
    // C64 on GB10 (and <= 256 rows on >64-SM discrete GPUs, unlocking C48/C64
    // at K4) and splits C65-C128 into two balanced sub-frames.
    // Batched DFlash2 admits C2-C16 on GB10 UMA (`16 x 8 = 128` rows) and
    // C2-C64 on discrete / `> 64`-SM GPUs (`32 x 8 = 256`, `48 x 5 = 240`,
    // `64 x 4 = 256` rows under adaptive-K), validating `--dflash-gamma`
    // separately during drafter load.
    if batched_dflash {
        if !(2..=dflash_max_concurrency).contains(&c.max_batch_size) {
            return Err(format!(
                "batched DFlash2 admits C2-C{dflash_max_concurrency} only; serve C{}+ target-only",
                dflash_max_concurrency + 1
            ));
        }
    } else if c.batched_adaptive_k {
        if !(2..=ADAPTIVE_MAX_CONCURRENCY).contains(&c.max_batch_size) {
            return Err(format!(
                "adaptive-K native batching admits C2-C{ADAPTIVE_MAX_CONCURRENCY} only; serve C{}+ target-only",
                ADAPTIVE_MAX_CONCURRENCY + 1
            ));
        }
    } else if !(2..=32).contains(&c.max_batch_size) {
        return Err("owned native K5 batching admits C2-C32 only (C33-C128 needs ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1)".into());
    }
    if !batched_dflash {
        if c.batched_adaptive_k {
            let max_drafts = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K - 1;
            if !(4..=max_drafts).contains(&c.num_drafts) {
                return Err(format!(
                    "adaptive-K native batching requires --num-drafts 4..={max_drafts}, got {}",
                    c.num_drafts
                ));
            }
        } else if c.num_drafts != 4 {
            return Err("owned native K5 batching requires --num-drafts 4".into());
        }
    }
    if !c.qsa {
        return Err("owned native K5 batching requires --qwen4-qsa".into());
    }
    if c.world_size > 4 {
        return Err("owned native K5 batching supports 1..=4 Spark ranks (world_size <= 4)".into());
    }
    if c.serial_k5.gate || c.serial_k5.cross_request_verify {
        return Err(
            "owned native K5 batching refuses serial K5 and other cross-request verify selectors"
                .into(),
        );
    }
    Ok(())
}

/// Serial K5 runs each live request's exact C1 verify -> commit transaction
/// back to back. It admits only the native MTP proposer at K5 and C2-C8, and
/// never a verifier that batches rows across requests.
fn validate_serial_k5(max_batch_size: usize, k5: SerialK5Request) -> Result<(), String> {
    use spark_model::model::qwen4_k5_cmulti::{
        QWEN4_K5_CMULTI_MAX_CONCURRENCY, QWEN4_K5_CMULTI_NUM_DRAFTS,
    };
    if !k5.native_mtp_only {
        return Err(
            "ATLAS_QWEN4_K5_CMULTI admits only native MTP; DFlash, self-speculative and ngram speculation stay target-only at C>1"
                .into(),
        );
    }
    if max_batch_size > QWEN4_K5_CMULTI_MAX_CONCURRENCY {
        return Err(format!(
            "ATLAS_QWEN4_K5_CMULTI admits C2-C{QWEN4_K5_CMULTI_MAX_CONCURRENCY} only, got C{max_batch_size}"
        ));
    }
    if k5.num_drafts != QWEN4_K5_CMULTI_NUM_DRAFTS {
        return Err(format!(
            "ATLAS_QWEN4_K5_CMULTI requires --num-drafts {QWEN4_K5_CMULTI_NUM_DRAFTS} (K5), got {}",
            k5.num_drafts
        ));
    }
    if k5.cross_request_verify {
        return Err(
            "ATLAS_QWEN4_K5_CMULTI requires ATLAS_MTP_K2_BATCH_CSEQ, ATLAS_MTP_K3_BATCH_CSEQ and ATLAS_DFLASH_BATCHED_VERIFY off; their cross-request verifiers have no Qwen4 transaction"
                .into(),
        );
    }
    Ok(())
}

/// Pre-load half of the default-on native batch route: the concurrency
/// ceiling and the `ATLAS_QWEN4_NATIVE_BATCH=1` / `_QSA=1` refusals, decided
/// from the same environment facts the model re-reads every frame. `None` at
/// C1, where no frame can batch.
pub(crate) fn admit_qwen4_native_batch(
    selector: NativeBatchSelector,
    facts: &ProcessFacts,
    qsa_rows: bool,
    max_batch_size: usize,
) -> Result<Option<Qwen4BatchPlan>, String> {
    qwen4_batch_policy::admit_width(selector, max_batch_size, MAX_ADMITTED_CONCURRENCY)?;
    if max_batch_size <= 1 {
        return Ok(None);
    }
    qwen4_batch_policy::resolve_plan(selector, facts, qsa_rows).map(Some)
}

/// Post-load: log the model's authoritative plan once. The model knows facts
/// the environment does not (whether it holds an NVFP4 LM head), so a
/// `Required` plan can still be refused here.
pub(crate) fn log_qwen4_batch_plan(
    model: &dyn spark_model::traits::Model,
    max_batch_size: usize,
) -> anyhow::Result<()> {
    if max_batch_size <= 1 {
        return Ok(());
    }
    let Some(plan) = model.qwen4_batch_plan()? else {
        return Ok(());
    };
    let selector = plan.selector.label();
    match plan.process_reason {
        None => tracing::info!(
            "QWEN4_BATCH_PLAN route=native selector={selector} widths=C2-C{QWEN4_NATIVE_MAX_ROWS} max_batch_size={max_batch_size} qsa_rows={}",
            plan.qsa_rows
        ),
        Some(SerialReason::SelectorOff) => tracing::info!(
            "QWEN4_BATCH_PLAN route=serial selector={selector} reason=selector_off max_batch_size={max_batch_size}"
        ),
        Some(reason) => tracing::warn!(
            "QWEN4_BATCH_PLAN route=serial selector={selector} reason={} max_batch_size={max_batch_size}: every C2+ frame runs C1 once per row; to batch natively, {}",
            reason.label(),
            reason.remedy()
        ),
    }
    if plan.native() {
        let report = model.qwen4_native_capability_report(max_batch_size)?;
        for line in native_capability_verdict(plan.selector, &report).map_err(anyhow::Error::msg)? {
            tracing::warn!("{line}");
        }
    }
    Ok(())
}

/// Load verdict on a native plan's per-width capability. `Required` refuses
/// any width it cannot run; otherwise each gap becomes one warning, and those
/// widths serialize with the gap's reason at run time.
pub(crate) fn native_capability_verdict(
    selector: NativeBatchSelector,
    report: &NativeCapabilityReport,
) -> Result<Vec<String>, String> {
    let groups = group_gaps(&report.gaps);
    if selector == NativeBatchSelector::Required
        && let Some((first, last, gap)) = groups.first()
    {
        return Err(format!(
            "ATLAS_QWEN4_NATIVE_BATCH=1 refused: native Qwen4 C{first}-C{last} is unavailable ({}): {}; {}, or unset the selector to serve those widths serially",
            gap.reason.label(),
            gap.detail,
            gap.reason.remedy()
        ));
    }
    Ok(groups
        .into_iter()
        .map(|(first, last, gap)| {
            format!(
                "QWEN4_BATCH_PLAN widths=C{first}-C{last} route=serial reason={}: {}; to batch natively, {}",
                gap.reason.label(),
                gap.detail,
                gap.reason.remedy()
            )
        })
        .collect())
}

/// Runs of consecutive widths with the same gap, as `(first, last, gap)`.
fn group_gaps(gaps: &[(usize, NativeGap)]) -> Vec<(usize, usize, &NativeGap)> {
    let mut groups: Vec<(usize, usize, &NativeGap)> = Vec::new();
    for (rows, gap) in gaps {
        match groups.last_mut() {
            Some((_, last, prior)) if *last + 1 == *rows && *prior == gap => *last = *rows,
            _ => groups.push((*rows, *rows, gap)),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_facts() -> ProcessFacts {
        ProcessFacts {
            profile: false,
            moe_exact_rows: true,
            ssm_h_fp16: false,
            lm_head_tc: false,
            lm_head_nvfp4: true,
            lm_head_full_vocab: true,
            qsa_index: true,
        }
    }

    #[test]
    fn qsa_rows_without_the_qsa_index_refuse_at_load() {
        let facts = ProcessFacts {
            qsa_index: false,
            ..native_facts()
        };
        for selector in [NativeBatchSelector::Auto, NativeBatchSelector::Required] {
            assert!(
                admit_qwen4_native_batch(selector, &facts, true, 4)
                    .unwrap_err()
                    .contains("--qwen4-qsa")
            );
            assert!(admit_qwen4_native_batch(selector, &facts, false, 4).is_ok());
        }
        // C1 never builds a native frame.
        assert_eq!(
            admit_qwen4_native_batch(NativeBatchSelector::Auto, &facts, true, 1),
            Ok(None)
        );
    }

    #[test]
    fn capability_gaps_warn_by_default_and_refuse_when_required() {
        let tier = NativeGap::new(SerialReason::LmHeadRowTier, "no tier");
        let moe = NativeGap::new(SerialReason::MoeExactRowsUnavailable, "layer 3: bounded");
        let report = NativeCapabilityReport {
            gaps: vec![
                (17, tier.clone()),
                (18, tier.clone()),
                (19, tier),
                (32, moe),
            ],
        };
        let lines = native_capability_verdict(NativeBatchSelector::Auto, &report).unwrap();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].contains("widths=C17-C19") && lines[0].contains("lm_head_row_tier"));
        assert!(
            lines[1].contains("widths=C32-C32") && lines[1].contains("moe_exact_rows_unavailable")
        );
        let err = native_capability_verdict(NativeBatchSelector::Required, &report).unwrap_err();
        assert!(
            err.contains("C17-C19") && err.contains("lm_head_row_tier"),
            "{err}"
        );
        let clean = NativeCapabilityReport::default();
        assert_eq!(
            native_capability_verdict(NativeBatchSelector::Required, &clean),
            Ok(Vec::new())
        );
    }

    #[test]
    fn target_vocab_cap_serializes_by_default_and_refuses_when_required() {
        let facts = ProcessFacts {
            lm_head_full_vocab: false,
            ..native_facts()
        };
        let plan = admit_qwen4_native_batch(NativeBatchSelector::Auto, &facts, false, 8)
            .unwrap()
            .unwrap();
        assert_eq!(plan.process_reason, Some(SerialReason::LmHeadVocabCap));
        assert!(
            admit_qwen4_native_batch(NativeBatchSelector::Required, &facts, false, 8)
                .unwrap_err()
                .contains("lm_head_vocab_cap")
        );
    }

    #[test]
    fn native_batch_is_the_default_c2_c32_plan() {
        let plan = admit_qwen4_native_batch(NativeBatchSelector::Auto, &native_facts(), false, 8)
            .unwrap()
            .unwrap();
        assert!(plan.native());
        assert_eq!(
            admit_qwen4_native_batch(NativeBatchSelector::Auto, &native_facts(), false, 1),
            Ok(None)
        );
    }

    #[test]
    fn native_ceiling_matches_qwen4_native_max_rows_and_selector_off_forces_serial() {
        for selector in [NativeBatchSelector::Auto, NativeBatchSelector::Required] {
            assert!(admit_qwen4_native_batch(selector, &native_facts(), false, 32).is_ok());
            assert!(
                admit_qwen4_native_batch(selector, &native_facts(), false, QWEN4_NATIVE_MAX_ROWS)
                    .is_ok()
            );
            assert!(
                admit_qwen4_native_batch(
                    selector,
                    &native_facts(),
                    false,
                    QWEN4_NATIVE_MAX_ROWS + 1
                )
                .unwrap_err()
                .contains("ATLAS_QWEN4_NATIVE_BATCH=0")
            );
        }
        let off = admit_qwen4_native_batch(NativeBatchSelector::Off, &native_facts(), false, 128)
            .unwrap()
            .unwrap();
        assert_eq!(off.process_reason, Some(SerialReason::SelectorOff));
    }

    #[test]
    fn missing_t_lanes_serializes_by_default_and_refuses_when_required() {
        let facts = ProcessFacts {
            moe_exact_rows: false,
            ..native_facts()
        };
        let plan = admit_qwen4_native_batch(NativeBatchSelector::Auto, &facts, false, 8)
            .unwrap()
            .unwrap();
        assert_eq!(
            plan.process_reason,
            Some(SerialReason::MoeExactRowsUnavailable)
        );
        assert!(
            admit_qwen4_native_batch(NativeBatchSelector::Required, &facts, false, 8)
                .unwrap_err()
                .contains("moe_exact_rows_unavailable")
        );
        // C1 qualification controls keep working with the selector set.
        assert_eq!(
            admit_qwen4_native_batch(NativeBatchSelector::Required, &facts, false, 1),
            Ok(None)
        );
    }

    #[test]
    fn qsa_rows_without_a_native_plan_refuse_at_load() {
        let facts = ProcessFacts {
            moe_exact_rows: false,
            ..native_facts()
        };
        assert!(
            admit_qwen4_native_batch(NativeBatchSelector::Auto, &facts, true, 4)
                .unwrap_err()
                .contains("ATLAS_QWEN4_NATIVE_BATCH_QSA")
        );
        let plan = admit_qwen4_native_batch(NativeBatchSelector::Auto, &native_facts(), true, 4)
            .unwrap()
            .unwrap();
        assert!(plan.qsa_rows);
    }

    #[test]
    fn bf16_head_control_serializes_by_default() {
        let facts = ProcessFacts {
            lm_head_nvfp4: false,
            ..native_facts()
        };
        let plan = admit_qwen4_native_batch(NativeBatchSelector::Auto, &facts, false, 4)
            .unwrap()
            .unwrap();
        assert_eq!(plan.process_reason, Some(SerialReason::LmHeadNotNvfp4));
    }

    fn native_bf16_c4() -> Qwen4Concurrency<'static> {
        Qwen4Concurrency {
            qsa: true,
            max_batch_size: 4,
            kv_cache_dtype: "bf16",
            native_context: 262_144,
            // Policy admission is not a claim that C128 x 262K BF16 KV fits
            // on one device. The capacity preflight separately gates the
            // aggregate requested depth.
            max_seq_len: 2_048,
            speculative: false,
            native_mtp_only: false,
            dflash_only: false,
            batched_k5: false,
            batched_adaptive_k: false,
            num_drafts: 4,
            swap_space_gb: 0,
            high_speed_swap: false,
            prefix_caching: false,
            ssm_cache_slots: 0,
            world_size: 1,
            tp_size: 1,
            ep_size: 1,
            serial_k5: SerialK5Request::OFF,
            kv_pool_tokens: None,
        }
    }

    #[test]
    fn shared_kv_pool_admits_extended_context_at_c2_and_keeps_every_other_gate() {
        let mut c = native_bf16_c4();
        c.max_batch_size = 2;
        c.max_seq_len = 524_288;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("native context")
        );
        let mut c = native_bf16_c4();
        c.max_batch_size = 2;
        c.max_seq_len = 524_288;
        c.kv_pool_tokens = Some(2 * 524_288);
        validate_qwen4_concurrency(c).unwrap();
        // The shared pool lifts only the BF16 per-slot context gate; invalid
        // dtypes, HSS, and multi-Spark disk swap stay fail-closed.
        let mut c = native_bf16_c4();
        c.max_seq_len = 524_288;
        c.kv_pool_tokens = Some(2 * 524_288);
        c.kv_cache_dtype = "fp8";
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("bf16 or nvfp4")
        );
        let mut c = native_bf16_c4();
        c.max_seq_len = 524_288;
        c.kv_pool_tokens = Some(2 * 524_288);
        c.world_size = 2;
        c.tp_size = 2;
        c.swap_space_gb = 3;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("--swap-space-gb 0")
        );
    }

    fn serial_k5_c(max_batch_size: usize) -> Qwen4Concurrency<'static> {
        let mut c = native_bf16_c4();
        c.max_batch_size = max_batch_size;
        c.speculative = true;
        c.serial_k5 = SerialK5Request {
            gate: true,
            native_mtp_only: true,
            num_drafts: 4,
            cross_request_verify: false,
        };
        c
    }

    #[test]
    fn serial_k5_gate_admits_native_mtp_k5_at_c2_through_c8() {
        for rows in 2..=8 {
            validate_qwen4_concurrency(serial_k5_c(rows)).unwrap();
        }
        validate_qwen4_concurrency(serial_k5_c(1)).unwrap();
    }

    #[test]
    fn unset_selectors_keep_the_target_only_refusal() {
        let mut c = serial_k5_c(4);
        c.serial_k5 = SerialK5Request::OFF;
        assert_eq!(
            validate_qwen4_concurrency(c).unwrap_err(),
            "Qwen4 concurrency is target-only unless native K5 is selected with ATLAS_QWEN4_K5_CMULTI=1 or ATLAS_QWEN4_BATCHED_K5=1; disable DFlash, self-speculative and ngram speculation"
        );
        // Gate fields other than `gate` never admit on their own.
        let mut c = serial_k5_c(4);
        c.serial_k5.gate = false;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("target-only")
        );
    }

    #[test]
    fn serial_k5_gate_refuses_other_proposers_widths_and_batched_verifiers() {
        for rows in [9, 16, 128] {
            assert!(
                validate_qwen4_concurrency(serial_k5_c(rows))
                    .unwrap_err()
                    .contains("C2-C8 only"),
                "C{rows}"
            );
        }
        let mut c = serial_k5_c(4);
        c.serial_k5.native_mtp_only = false;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("only native MTP")
        );
        for drafts in [0, 1, 3, 5, 15] {
            let mut c = serial_k5_c(4);
            c.serial_k5.num_drafts = drafts;
            assert!(
                validate_qwen4_concurrency(c)
                    .unwrap_err()
                    .contains("--num-drafts 4"),
                "drafts={drafts}"
            );
        }
        let mut c = serial_k5_c(4);
        c.serial_k5.cross_request_verify = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("cross-request verifiers")
        );
    }

    #[test]
    fn serial_k5_gate_admits_nvfp4_yarn_prefix_swap_and_tp2_while_rejecting_invalid_bounds() {
        let mut single = serial_k5_c(4);
        single.kv_cache_dtype = "nvfp4";
        single.swap_space_gb = 4;
        single.prefix_caching = true;
        single.ssm_cache_slots = 32;
        single.max_seq_len = 1_048_576;
        validate_qwen4_concurrency(single).unwrap();

        let mut tp2 = serial_k5_c(4);
        tp2.kv_cache_dtype = "nvfp4";
        tp2.prefix_caching = true;
        tp2.ssm_cache_slots = 32;
        tp2.world_size = 2;
        tp2.tp_size = 2;
        tp2.max_seq_len = 1_048_576;
        validate_qwen4_concurrency(tp2).unwrap();

        let mut tp2_swap = serial_k5_c(4);
        tp2_swap.world_size = 2;
        tp2_swap.tp_size = 2;
        tp2_swap.swap_space_gb = 4;
        assert!(
            validate_qwen4_concurrency(tp2_swap)
                .unwrap_err()
                .contains("--swap-space-gb 0")
        );

        let mut c = serial_k5_c(4);
        c.kv_cache_dtype = "fp8";
        assert!(validate_qwen4_concurrency(c).unwrap_err().contains("nvfp4"));
        let mut c = serial_k5_c(4);
        c.high_speed_swap = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("high-speed-swap")
        );
        let mut c = serial_k5_c(4);
        c.tp_size = 3;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("TP=1, TP=2, or TP=4")
        );
        let mut c = serial_k5_c(4);
        c.max_seq_len = 1_048_577;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("1048576")
        );
    }

    #[test]
    fn admits_target_only_native_context() {
        validate_qwen4_concurrency(native_bf16_c4()).unwrap();
    }

    #[test]
    fn policy_admits_short_context_c128_and_rejects_c129() {
        let mut c = native_bf16_c4();
        c.max_batch_size = 128;
        validate_qwen4_concurrency(c).unwrap();

        let mut c = native_bf16_c4();
        c.max_batch_size = 129;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("admitted C128 ceiling")
        );
    }

    #[test]
    fn c1_prefix_snapshots_admitted_with_ple_conv_history_snapshots() {
        for qsa in [false, true] {
            for dtype in ["bf16", "nvfp4"] {
                for slots in [16, 32, 96] {
                    let mut c = native_bf16_c4();
                    c.max_batch_size = 1;
                    c.qsa = qsa;
                    c.kv_cache_dtype = dtype;
                    c.prefix_caching = true;
                    c.ssm_cache_slots = slots;
                    validate_qwen4_concurrency(c).unwrap();
                }
            }
        }
    }

    #[test]
    fn c1_without_paired_prefix_snapshots_keeps_full_prefill_admitted() {
        for (prefix_caching, ssm_cache_slots) in [(false, 0), (true, 0), (false, 16)] {
            let mut c = native_bf16_c4();
            c.max_batch_size = 1;
            c.prefix_caching = prefix_caching;
            c.ssm_cache_slots = ssm_cache_slots;
            validate_qwen4_concurrency(c).unwrap();
        }
    }

    #[test]
    fn concurrent_prefix_and_swap_lifecycles_admitted_across_1m_and_multispark() {
        for (prefix_caching, ssm_cache_slots) in [(true, 0), (false, 16), (true, 16)] {
            let mut c = native_bf16_c4();
            c.kv_cache_dtype = "nvfp4";
            c.prefix_caching = prefix_caching;
            c.ssm_cache_slots = ssm_cache_slots;
            c.swap_space_gb = 8;
            c.max_seq_len = 1_048_576;
            validate_qwen4_concurrency(c).unwrap();
        }
        // 2x Spark TP=2 and 3x Spark PP=3 (world_size=3, tp=1, ep=1)
        let mut tp2 = native_bf16_c4();
        tp2.world_size = 2;
        tp2.tp_size = 2;
        validate_qwen4_concurrency(tp2).unwrap();

        let mut pp3 = native_bf16_c4();
        pp3.world_size = 3;
        pp3.tp_size = 1;
        pp3.ep_size = 1;
        validate_qwen4_concurrency(pp3).unwrap();
    }

    #[test]
    fn keeps_c1_nvfp4_compatible() {
        let mut c = native_bf16_c4();
        c.max_batch_size = 1;
        c.kv_cache_dtype = "nvfp4";
        validate_qwen4_concurrency(c).unwrap();
    }

    #[test]
    fn concurrent_invalid_modes_fail_closed() {
        let mut c = native_bf16_c4();
        c.speculative = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("target-only")
        );

        let mut c = native_bf16_c4();
        c.high_speed_swap = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("high-speed-swap")
        );

        let mut c = native_bf16_c4();
        c.max_seq_len = 1_048_577;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("1048576")
        );
    }

    fn batched_native_mtp_c4() -> Qwen4Concurrency<'static> {
        let mut c = native_bf16_c4();
        c.speculative = true;
        c.native_mtp_only = true;
        c.batched_k5 = true;
        c
    }

    #[test]
    fn batched_k5_admits_concurrent_native_mtp() {
        validate_qwen4_concurrency(batched_native_mtp_c4()).unwrap();
        for width in [8, 16, 25, 32] {
            let mut c = batched_native_mtp_c4();
            c.max_batch_size = width;
            c.max_seq_len = 32_768;
            validate_qwen4_concurrency(c).unwrap_or_else(|e| panic!("C{width}: {e}"));
        }
        let mut c = batched_native_mtp_c4();
        c.max_batch_size = 8;
        c.kv_cache_dtype = "nvfp4";
        c.max_seq_len = 1_048_576;
        c.world_size = 2;
        c.tp_size = 2;
        validate_qwen4_concurrency(c).unwrap();
    }

    #[test]
    fn batched_k5_refuses_other_proposers() {
        let mut c = batched_native_mtp_c4();
        c.native_mtp_only = false;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("only for the native MTP head")
        );
    }

    #[test]
    fn batched_dflash2_admits_c2_through_c16_under_adaptive_k_and_fails_closed() {
        let dflash_c = |max_batch_size: usize| {
            let mut c = batched_native_mtp_c4();
            c.native_mtp_only = false;
            c.dflash_only = true;
            c.batched_adaptive_k = true;
            c.max_batch_size = max_batch_size;
            // `--dflash` leaves `--num-drafts` at its default of 1 while
            // `--dflash-gamma` is validated during drafter load.
            c.num_drafts = 1;
            c
        };
        for width in [2, 4, 8, 16] {
            validate_qwen4_concurrency(dflash_c(width))
                .unwrap_or_else(|e| panic!("batched DFlash2 C{width}: {e}"));
        }
        let mut tp2 = dflash_c(16);
        tp2.world_size = 2;
        tp2.tp_size = 2;
        validate_qwen4_concurrency(tp2).unwrap();

        assert!(
            validate_qwen4_concurrency(dflash_c(17))
                .unwrap_err()
                .contains("C2-C16")
        );
        let mut no_qsa = dflash_c(16);
        no_qsa.qsa = false;
        assert!(
            validate_qwen4_concurrency(no_qsa)
                .unwrap_err()
                .contains("--qwen4-qsa")
        );
        let mut tp4 = dflash_c(16);
        tp4.world_size = 4;
        tp4.tp_size = 4;
        tp4.ep_size = 4;
        validate_qwen4_concurrency(tp4).unwrap();

        let mut world5 = dflash_c(8);
        world5.world_size = 5;
        assert!(validate_qwen4_concurrency(world5).is_err());
        let mut serial_gate = dflash_c(8);
        serial_gate.serial_k5.gate = true;
        assert!(validate_qwen4_concurrency(serial_gate).is_err());
        let mut cross_verify = dflash_c(8);
        cross_verify.serial_k5.cross_request_verify = true;
        assert!(validate_qwen4_concurrency(cross_verify).is_err());
        let mut non_dflash_aux = dflash_c(8);
        non_dflash_aux.dflash_only = false;
        assert!(
            validate_qwen4_concurrency(non_dflash_aux)
                .unwrap_err()
                .contains("disable self-speculative and ngram speculation")
        );

        // SM-aware / discrete topology check: GB10 UMA stays C=16, discrete / >64-SM lifts to C=64.
        assert_eq!(dflash_batched_max_concurrency_for(48, Some(true), None), 16);
        assert_eq!(dflash_batched_max_concurrency_for(64, Some(true), None), 16);
        assert_eq!(dflash_batched_max_concurrency_for(48, None, None), 16);
        assert_eq!(
            dflash_batched_max_concurrency_for(188, Some(false), None),
            64
        );
        assert_eq!(dflash_batched_max_concurrency_for(65, None, None), 64);
        assert_eq!(
            dflash_batched_max_concurrency_for(48, Some(false), None),
            64
        );
        assert_eq!(
            dflash_batched_max_concurrency_for(48, Some(true), Some("256")),
            64
        );

        let wide_cap = dflash_batched_max_concurrency_for(188, Some(false), None);
        for width in [16usize, 24, 32, 48, 64] {
            validate_batched_k5_with_dflash_cap(&dflash_c(width), wide_cap)
                .unwrap_or_else(|e| panic!("SM120 batched DFlash2 C{width}: {e}"));
        }
        assert!(
            validate_batched_k5_with_dflash_cap(&dflash_c(65), wide_cap)
                .unwrap_err()
                .contains("C2-C64")
        );
    }

    #[test]
    fn concurrent_native_mtp_without_the_selector_stays_refused() {
        let mut c = batched_native_mtp_c4();
        c.batched_k5 = false;
        let error = validate_qwen4_concurrency(c).unwrap_err();
        assert!(error.contains("target-only") && error.contains("ATLAS_QWEN4_BATCHED_K5"));
    }

    #[test]
    fn batched_k5_does_not_waive_invalid_lifecycle_guards() {
        let mut c = batched_native_mtp_c4();
        c.high_speed_swap = true;
        assert!(validate_qwen4_concurrency(c).is_err());
        let mut c = batched_native_mtp_c4();
        c.kv_cache_dtype = "fp8";
        assert!(validate_qwen4_concurrency(c).is_err());
        let mut c = batched_native_mtp_c4();
        c.world_size = 5;
        assert!(validate_qwen4_concurrency(c).is_err());
        let mut c = batched_native_mtp_c4();
        c.max_seq_len = 1_048_577;
        assert!(validate_qwen4_concurrency(c).is_err());
        for drafts in [0, 1, 3, 5, usize::MAX] {
            let mut c = batched_native_mtp_c4();
            c.num_drafts = drafts;
            assert!(validate_qwen4_concurrency(c).is_err());
        }
        for width in [33, 64, 128] {
            let mut c = batched_native_mtp_c4();
            c.max_batch_size = width;
            assert!(validate_qwen4_concurrency(c).is_err());
        }
    }

    #[test]
    fn adaptive_k_admits_c2_through_c128_and_nothing_wider() {
        for width in [2, 8, 16, 25, 32, 42, 64, 65, 96, 128] {
            let mut c = batched_native_mtp_c4();
            c.batched_adaptive_k = true;
            c.max_batch_size = width;
            validate_qwen4_concurrency(c).unwrap_or_else(|e| panic!("C{width}: {e}"));
        }
        for width in [129, 256] {
            let mut c = batched_native_mtp_c4();
            c.batched_adaptive_k = true;
            c.max_batch_size = width;
            assert!(
                validate_qwen4_concurrency(c)
                    .unwrap_err()
                    .contains("admitted C128 ceiling"),
                "C{width}"
            );
        }
        let mut c = batched_native_mtp_c4();
        c.max_batch_size = 64;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("ATLAS_QWEN4_BATCHED_ADAPTIVE_K")
        );
    }

    #[test]
    fn adaptive_k_requires_the_batched_selector_and_keeps_every_other_guard() {
        let mut c = batched_native_mtp_c4();
        c.batched_k5 = false;
        c.batched_adaptive_k = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("requires ATLAS_QWEN4_BATCHED_K5=1")
        );
        for drafts in [0, 1, 3, 8] {
            let mut c = batched_native_mtp_c4();
            c.batched_adaptive_k = true;
            c.max_batch_size = 64;
            c.num_drafts = drafts;
            assert!(
                validate_qwen4_concurrency(c).is_err(),
                "--num-drafts {drafts}"
            );
        }
        for drafts in [4, 5, 6, 7] {
            let mut c = batched_native_mtp_c4();
            c.batched_adaptive_k = true;
            c.max_batch_size = 16;
            c.num_drafts = drafts;
            assert!(
                validate_qwen4_concurrency(c).is_ok(),
                "--num-drafts {drafts} should be admitted under adaptive-K"
            );
        }
        let mut c = batched_native_mtp_c4();
        c.batched_adaptive_k = true;
        c.max_batch_size = 64;
        c.qsa = false;
        assert!(validate_qwen4_concurrency(c).is_err());
    }

    #[test]
    fn c1_speculation_is_unaffected_by_the_selector() {
        for batched_k5 in [false, true] {
            let mut c = native_bf16_c4();
            c.max_batch_size = 1;
            c.speculative = true;
            c.batched_k5 = batched_k5;
            validate_qwen4_concurrency(c).unwrap();
        }
    }

    #[test]
    fn qsa_disabled_still_enforces_qwen4_concurrent_lifecycle() {
        let mut c = native_bf16_c4();
        c.qsa = false;
        validate_qwen4_concurrency(c).unwrap();

        let mut c = native_bf16_c4();
        c.qsa = false;
        c.speculative = true;
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("target-only")
        );

        let mut c = native_bf16_c4();
        c.qsa = false;
        c.kv_cache_dtype = "fp8";
        assert!(
            validate_qwen4_concurrency(c)
                .unwrap_err()
                .contains("bf16 or nvfp4")
        );
    }

    #[test]
    fn single_spark_swap_is_admitted_and_multispark_swap_is_rejected() {
        let mut single = native_bf16_c4();
        single.swap_space_gb = 16;
        validate_qwen4_concurrency(single).expect("single-rank Qwen4 swap should be admitted");

        let mut tp2 = native_bf16_c4();
        tp2.world_size = 2;
        tp2.tp_size = 2;
        tp2.swap_space_gb = 16;
        assert!(
            validate_qwen4_concurrency(tp2)
                .unwrap_err()
                .contains("--swap-space-gb 0")
        );

        let mut pp3 = native_bf16_c4();
        pp3.world_size = 3;
        pp3.swap_space_gb = 8;
        assert!(
            validate_qwen4_concurrency(pp3)
                .unwrap_err()
                .contains("--swap-space-gb 0")
        );
    }
}
