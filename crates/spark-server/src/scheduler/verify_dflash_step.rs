// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash-based verify step (drafted token verification).

use super::spec_timing::SpecCycle;
use super::*;

#[path = "verify_dflash_capacity.rs"]
pub(super) mod dflash_capacity;
#[path = "verify_dflash_k16_fixture.rs"]
mod k16_acceptance_fixture;

/// Hard cap on debug branch-audit stash entries so abandoned sessions cannot
/// leak memory unboundedly when `ATLAS_DFLASH_BRANCH_AUDIT=1` is active.
const BRANCH_AUDIT_STASH_CAP: usize = 256;

static BRANCH_AUDIT_STASH: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<u64, (u32, u32)>>,
> = std::sync::OnceLock::new();

/// ATLAS_DFLASH_BRANCH_AUDIT=1 cross-step stash: session_hash →
/// `(expected_argmax_after_fork, fork_token)` recorded on a step whose
/// flat-safe commit ended with the fork token as bonus. The NEXT verify for
/// the same session compares its root row (`verified[0]`, the TRUE greedy
/// after the fork) against the fork ROW's argmax from the previous step —
/// a direct, per-step-deterministic measurement of branch-row conditioning
/// correctness (the 2026-07-08 deep-branch root cause). Debug-only.
fn branch_audit_stash() -> &'static std::sync::Mutex<std::collections::HashMap<u64, (u32, u32)>> {
    BRANCH_AUDIT_STASH.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Remove any stashed branch-audit entry when a sequence finishes or errors.
/// Zero-cost no-op unless `ATLAS_DFLASH_BRANCH_AUDIT=1` initialized the stash.
pub(super) fn prune_branch_audit_stash(session_hash: u64) {
    if let Some(stash) = BRANCH_AUDIT_STASH.get()
        && let Ok(mut guard) = stash.lock()
    {
        guard.remove(&session_hash);
    }
}

/// Running audit tally: (matches, mismatches). Logged on every comparison.
static BRANCH_AUDIT_TALLY: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));

fn ep_kgamma_result(
    num_accepted: usize,
    policy_walk_active: bool,
    sequence_finished: bool,
    bonus_available: bool,
) -> u32 {
    if policy_walk_active && (sequence_finished || !bonus_available) {
        EP_VERIFY_KGAMMA_ABORT
    } else {
        u32::try_from(num_accepted).unwrap_or(EP_VERIFY_KGAMMA_ABORT)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WideVerifyHiddenSave {
    DflashToken(u32),
    NativeMtpRow(usize),
}

pub(super) fn wide_verify_metric_k(
    kind: Option<spark_model::speculative::DraftProposerKind>,
    origin: DraftOrigin,
    rows: usize,
) -> std::borrow::Cow<'static, str> {
    if kind == Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp)
        && origin == DraftOrigin::Proposer
    {
        match rows {
            2 => std::borrow::Cow::Borrowed("2"),
            3 => std::borrow::Cow::Borrowed("3"),
            4 => std::borrow::Cow::Borrowed("4"),
            5 => std::borrow::Cow::Borrowed("5"),
            6 => std::borrow::Cow::Borrowed("6"),
            7 => std::borrow::Cow::Borrowed("7"),
            8 => std::borrow::Cow::Borrowed("8"),
            9 => std::borrow::Cow::Borrowed("9"),
            other => std::borrow::Cow::Owned(other.to_string()),
        }
    } else {
        std::borrow::Cow::Borrowed("dflash")
    }
}

#[inline]
fn dflash_step_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_STEP_TIMING").ok().as_deref() == Some("1"))
}

#[inline]
fn ddtree_tree_tokens_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DDTREE_TREE_TOKENS_VERIFY")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[inline]
fn dflash_tree_commit_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_TREE_COMMIT").ok().as_deref() == Some("1"))
}

#[inline]
fn dflash_early_exit_profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_EARLY_EXIT_PROFILE")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[inline]
fn dflash_recycle_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_RECYCLE").ok().as_deref() == Some("1"))
}

#[inline]
fn dflash_propose_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_PROPOSE_LOG").ok().as_deref() == Some("1"))
}

#[inline]
fn dflash_think_mask_verify_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_THINK_MASK_VERIFY")
            .ok()
            .as_deref()
            == Some("0")
    })
}

/// Select the hidden-state handoff consumed by the active proposer after a
/// wide verify. Row `n` is the target hidden after the prefix plus `n`
/// accepted drafts; its argmax/sample is the bonus token passed separately to
/// native MTP. A DDTree fork can end at a non-contiguous compact row, but that
/// case belongs to DFlash and is retained here for a fail-safe caller contract.
fn wide_verify_hidden_save(
    proposer_is_dflash: bool,
    bonus: u32,
    num_accepted: usize,
    tree_bonus_row: Option<usize>,
) -> WideVerifyHiddenSave {
    if proposer_is_dflash {
        WideVerifyHiddenSave::DflashToken(bonus)
    } else {
        WideVerifyHiddenSave::NativeMtpRow(tree_bonus_row.unwrap_or(num_accepted))
    }
}

/// ECHO may reuse a target tail only when both pieces of proposer state that
/// condition the next frame were refreshed successfully. This gate does not
/// change the ordinary DFlash recovery policy; it only suppresses the
/// optional ECHO shortcut when that policy continued after an error.
const fn echo_state_ready(hidden_saved: bool, proposer_trimmed: bool) -> bool {
    hidden_saved && proposer_trimmed
}

fn dispatch_echo_stash<F>(
    enabled: bool,
    state_ready: bool,
    route_eligible: bool,
    verified: &[u32],
    num_accepted: usize,
    mut stash: F,
) -> Result<()>
where
    F: FnMut(&[u32], usize) -> Result<()>,
{
    if !enabled {
        return Ok(());
    }
    if state_ready && route_eligible {
        stash(verified, num_accepted)
    } else {
        // Empty is below EchoConfig's enforced min_tail >= 1 and therefore
        // invalidates a pre-existing one-step stash.
        stash(&[], 0)
    }
}

const fn suppress_reproposal(grammar_skip: bool, echo_dispatch_failed: bool) -> bool {
    grammar_skip || echo_dispatch_failed
}

#[derive(Debug)]
struct SerialOracleReplay {
    verified: Vec<u32>,
    logits_rows: Vec<Vec<u8>>,
    logits_format: VerifyLogitsFormat,
    k16_commit_parity: bool,
}

fn serial_oracle_requested(model: &dyn Model, a: &ActiveSeq) -> bool {
    std::env::var("ATLAS_DFLASH_SERIAL_COMMIT").ok().as_deref() == Some("1")
        && std::env::var("ATLAS_DFLASH_SKIP_REPROPOSE").ok().as_deref() == Some("1")
        && a.pending_tree_payload.is_none()
        && !model.is_ep()
}

const K1_STAGE_SHADOW_ENV: &str = "ATLAS_DFLASH_K1_STAGE_SHADOW";
const K1_STAGE_DIAG_ENV: &str = "ATLAS_DFLASH_K1_STAGE_DIAG";
const K1_STAGE_SHADOW_PRE_VERIFY_LEN: usize = 8308;
const K1_STAGE_SHADOW_TOKENS: [u32; 5] = [271, 1596, 1144, 4087, 1156];
const TRAJECTORY_DIAG_ENV: &str = "ATLAS_DFLASH_K1_TRAJECTORY_DIAG";
const TRAJECTORY_SEQ_LEN_ENV: &str = "ATLAS_DFLASH_K1_TRAJECTORY_SEQ_LEN";
const TRAJECTORY_TOKENS_ENV: &str = "ATLAS_DFLASH_K1_TRAJECTORY_TOKENS";

fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} is not valid Unicode")
        }
    }
}

fn canonical_usize(name: &str, raw: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} must be a canonical decimal usize"))?;
    if value.to_string() != raw {
        anyhow::bail!("{name} must be a canonical decimal usize");
    }
    Ok(value)
}

fn canonical_u32_csv(name: &str, raw: &str) -> Result<Vec<u32>> {
    if raw.is_empty()
        || raw
            .bytes()
            .any(|byte| !(byte.is_ascii_digit() || byte == b','))
    {
        anyhow::bail!("{name} must be a nonempty comma-separated canonical u32 vector");
    }
    raw.split(',')
        .map(|item| {
            let value = item.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("{name} must be a nonempty comma-separated canonical u32 vector")
            })?;
            if value.to_string() != item {
                anyhow::bail!("{name} must be a nonempty comma-separated canonical u32 vector");
            }
            Ok(value)
        })
        .collect()
}

fn trajectory_selector_matches(pre_verify_len: usize, tokens: &[u32]) -> Result<bool> {
    let enabled = optional_env(TRAJECTORY_DIAG_ENV)?;
    // The serving wrapper exports the whole snapshotted selector contract on
    // every run. Treat only empty selector values as absent while retaining a
    // fail-closed error for any nonempty selector when diagnostics are off.
    let seq_len = optional_env(TRAJECTORY_SEQ_LEN_ENV)?.filter(|value| !value.is_empty());
    let selected_tokens = optional_env(TRAJECTORY_TOKENS_ENV)?.filter(|value| !value.is_empty());
    trajectory_selector_values_match(
        enabled.as_deref(),
        seq_len.as_deref(),
        selected_tokens.as_deref(),
        pre_verify_len,
        tokens,
    )
}

fn trajectory_selector_values_match(
    enabled: Option<&str>,
    seq_len: Option<&str>,
    selected_tokens: Option<&str>,
    pre_verify_len: usize,
    tokens: &[u32],
) -> Result<bool> {
    match enabled {
        None | Some("0") => {
            if seq_len.is_some() || selected_tokens.is_some() {
                anyhow::bail!("trajectory selector variables require {TRAJECTORY_DIAG_ENV}=1");
            }
            Ok(false)
        }
        Some("1") => {
            let selected_seq_len = canonical_usize(
                TRAJECTORY_SEQ_LEN_ENV,
                seq_len.ok_or_else(|| {
                    anyhow::anyhow!("{TRAJECTORY_DIAG_ENV}=1 requires {TRAJECTORY_SEQ_LEN_ENV}")
                })?,
            )?;
            let selected_tokens = canonical_u32_csv(
                TRAJECTORY_TOKENS_ENV,
                selected_tokens.ok_or_else(|| {
                    anyhow::anyhow!("{TRAJECTORY_DIAG_ENV}=1 requires {TRAJECTORY_TOKENS_ENV}")
                })?,
            )?;
            if selected_tokens.len() != 5 {
                anyhow::bail!("{TRAJECTORY_TOKENS_ENV} must contain exact gamma4 K=5 inputs");
            }
            Ok(pre_verify_len == selected_seq_len && tokens == selected_tokens)
        }
        Some(_) => anyhow::bail!("{TRAJECTORY_DIAG_ENV} must be exactly 0 or 1"),
    }
}

fn canonical_u32_csv_text(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn require_exactly_off_or_unset(name: &str, raw: Option<&str>) -> Result<()> {
    match raw {
        None | Some("0") => Ok(()),
        Some("1") => anyhow::bail!("{K1_STAGE_SHADOW_ENV}=1 requires {name}=0 or unset"),
        Some(_) => anyhow::bail!("{name} must be exactly 0 or 1"),
    }
}

#[allow(clippy::too_many_arguments)]
fn k1_stage_shadow_contract_values(
    shadow_env: Option<&str>,
    diag_env: Option<&str>,
    serial_commit_env: Option<&str>,
    skip_repropose_env: Option<&str>,
    selected: bool,
    pre_verify_len: usize,
    tokens: &[u32],
    c1: bool,
    flat: bool,
    ep: bool,
    k: usize,
) -> Result<bool> {
    match shadow_env {
        None | Some("0") => return Ok(false),
        Some("1") => {}
        Some(_) => anyhow::bail!("{K1_STAGE_SHADOW_ENV} must be exactly 0 or 1"),
    }
    anyhow::ensure!(
        diag_env == Some("1"),
        "{K1_STAGE_SHADOW_ENV}=1 requires {K1_STAGE_DIAG_ENV}=1"
    );
    require_exactly_off_or_unset("ATLAS_DFLASH_SERIAL_COMMIT", serial_commit_env)?;
    require_exactly_off_or_unset("ATLAS_DFLASH_SKIP_REPROPOSE", skip_repropose_env)?;
    if !selected {
        return Ok(false);
    }
    anyhow::ensure!(
        pre_verify_len == K1_STAGE_SHADOW_PRE_VERIFY_LEN && tokens == K1_STAGE_SHADOW_TOKENS,
        "{K1_STAGE_SHADOW_ENV}=1 is sealed to pre_verify_len={} tokens={:?}",
        K1_STAGE_SHADOW_PRE_VERIFY_LEN,
        K1_STAGE_SHADOW_TOKENS,
    );
    anyhow::ensure!(c1, "{K1_STAGE_SHADOW_ENV}=1 requires validated C=1");
    anyhow::ensure!(flat, "{K1_STAGE_SHADOW_ENV}=1 requires a flat verify frame");
    anyhow::ensure!(!ep, "{K1_STAGE_SHADOW_ENV}=1 does not support EP");
    anyhow::ensure!(
        k == K1_STAGE_SHADOW_TOKENS.len(),
        "{K1_STAGE_SHADOW_ENV}=1 requires exact K={} (got {k})",
        K1_STAGE_SHADOW_TOKENS.len(),
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn k1_stage_shadow_contract(
    selected: bool,
    pre_verify_len: usize,
    tokens: &[u32],
    c1: bool,
    flat: bool,
    ep: bool,
    k: usize,
) -> Result<bool> {
    let shadow = optional_env(K1_STAGE_SHADOW_ENV)?;
    let diag = optional_env(K1_STAGE_DIAG_ENV)?;
    let serial_commit = optional_env("ATLAS_DFLASH_SERIAL_COMMIT")?;
    let skip_repropose = optional_env("ATLAS_DFLASH_SKIP_REPROPOSE")?;
    k1_stage_shadow_contract_values(
        shadow.as_deref(),
        diag.as_deref(),
        serial_commit.as_deref(),
        skip_repropose.as_deref(),
        selected,
        pre_verify_len,
        tokens,
        c1,
        flat,
        ep,
        k,
    )
}

fn k5_commit_parity_route_contract(selected: bool, c1: bool, flat: bool, ep: bool) -> Result<bool> {
    if !selected {
        return Ok(false);
    }
    anyhow::ensure!(c1, "DFLASH_K5_COMMIT_PARITY requires validated C=1");
    anyhow::ensure!(flat, "DFLASH_K5_COMMIT_PARITY requires a flat verify frame");
    anyhow::ensure!(!ep, "DFLASH_K5_COMMIT_PARITY does not support EP");
    Ok(true)
}

/// Abort a verify frame without short-circuiting either cache owner. QSA is
/// restored first on the default stream, then recurrent/PLE live state is
/// restored from the canonical checkpoint on that same stream.
pub(super) fn rollback_failed_verify_frame(
    model: &dyn Model,
    seq: &mut SequenceState,
    k_verify: usize,
) -> Result<()> {
    model.abort_verify_state(seq, k_verify)
}

/// Run the target's ordinary one-token path over the full forced verify input,
/// preserving both argmaxes and logits rows for the later sampler-policy walk.
/// The recurrent checkpoint and host sequence counters are restored before
/// returning, so the normal batched verifier still starts from the same state.
fn collect_serial_oracle(
    model: &dyn Model,
    seq: &mut SequenceState,
    tokens: &[u32],
) -> Result<SerialOracleReplay> {
    let pre_verify_len = seq.seq_len;
    let pre_verify_tokens = seq.tokens.len();
    let capture_stages = model.k1_stage_diag_requested(pre_verify_len, tokens)?;
    let k16_commit_parity = model.k16_commit_parity_requested(pre_verify_len, tokens)?;
    if trajectory_selector_matches(pre_verify_len, tokens)? {
        if pre_verify_tokens != pre_verify_len {
            anyhow::bail!(
                "DFLASH_K1_TRAJECTORY sequence extent mismatch: tokens={} seq_len={}",
                pre_verify_tokens,
                pre_verify_len
            );
        }
        tracing::info!(
            "DFLASH_K1_TRAJECTORY schema=1 pre_verify_len={} prefix_tokens={} input_tokens={}",
            pre_verify_len,
            canonical_u32_csv_text(&seq.tokens),
            canonical_u32_csv_text(tokens),
        );
    }
    model.pre_verify_copy_async(seq)?;
    if capture_stages {
        model.k1_stage_diag_begin(pre_verify_len, tokens)?;
    }

    let logits_format = if model.decode_logits_fp32() {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let elem_bytes = if logits_format == VerifyLogitsFormat::Fp32 {
        4
    } else {
        2
    };
    let vocab = model.vocab_size();
    let replay = (|| {
        let mut verified = Vec::with_capacity(tokens.len());
        let mut logits_rows = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let logits = model.decode(token, seq, 0)?;
            verified.push(model.argmax_on_device(logits, 0)?);
            let mut row = vec![0u8; vocab * elem_bytes];
            model.copy_logits_to_host(logits, &mut row)?;
            logits_rows.push(row);
            if capture_stages {
                model.k1_stage_diag_finish_serial_row()?;
            }
        }
        if capture_stages {
            model.k1_stage_diag_finish_serial()?;
        }
        Ok(SerialOracleReplay {
            verified,
            logits_rows,
            logits_format,
            k16_commit_parity,
        })
    })();

    // Always put the caller back in the pre-verify frame, including on a
    // mid-replay D2H/decode failure. The KV rows written above are harmless:
    // the immediately-following batched verifier overwrites the same slots.
    seq.seq_len = pre_verify_len;
    seq.tokens.truncate(pre_verify_tokens);
    let restore = model.pre_verify_copy_async(seq);
    if replay.is_err() || restore.is_err() {
        model.k1_stage_diag_abort();
    }
    match (replay, restore) {
        (Ok(replay), Ok(())) => Ok(replay),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

/// Run the eager batched half of the exact stage comparator, then restore the
/// host and recurrent state before the ordinary production verifier executes.
/// The production call overwrites the same speculative KV rows and is the only
/// result allowed to reach acceptance or commit.
fn run_k5_stage_shadow(
    model: &dyn Model,
    seq: &mut SequenceState,
    tokens: &[u32],
) -> Result<Vec<u32>> {
    let pre_verify_len = seq.seq_len;
    let pre_verify_tokens = seq.tokens.len();
    anyhow::ensure!(
        pre_verify_tokens == pre_verify_len,
        "K5 stage shadow sequence extent mismatch: tokens={pre_verify_tokens} seq_len={pre_verify_len}",
    );
    let expected_post_len = pre_verify_len
        .checked_add(tokens.len())
        .ok_or_else(|| anyhow::anyhow!("K5 stage shadow sequence length overflow"))?;

    let shadow = (|| {
        model.stage_next_verify_rejection_rows(&[]);
        let verified = model.decode_verify_dflash(tokens, seq, 0)?;
        anyhow::ensure!(
            verified.len() == tokens.len(),
            "K5 stage shadow returned {} rows for {} inputs",
            verified.len(),
            tokens.len(),
        );
        anyhow::ensure!(
            seq.seq_len == expected_post_len && seq.tokens.len() == expected_post_len,
            "K5 stage shadow post-verify extent mismatch: seq_len={} tokens={} expected={expected_post_len}",
            seq.seq_len,
            seq.tokens.len(),
        );
        Ok(verified)
    })();

    // Restore even when the diagnostic verifier fails. QSA requires an
    // explicit rejected-prefix commit: unlike ordinary KV, its modulo-four
    // raw ring and compressed group stores are not hidden by seq_len rollback.
    // A failed restoration terminates the request; no partially-shadowed
    // frame may reach acceptance.
    seq.seq_len = pre_verify_len;
    seq.tokens.truncate(pre_verify_tokens);
    let qsa_restore = model.commit_qwen4_qsa_for_sequence(seq, 0, tokens.len());
    let restore = model.pre_verify_copy_async(seq);
    let restored_owner = if qsa_restore.is_ok() && restore.is_ok() {
        model.finish_verify_diagnostic(seq, tokens.len())
    } else {
        Ok(())
    };
    restored_owner?;
    match (shadow, qsa_restore, restore) {
        (Ok(verified), Ok(()), Ok(())) => {
            anyhow::ensure!(
                seq.seq_len == pre_verify_len && seq.tokens.len() == pre_verify_tokens,
                "K5 stage shadow did not restore the host sequence extent",
            );
            Ok(verified)
        }
        (Err(shadow_error), Ok(()), Ok(())) => Err(shadow_error),
        (Ok(_), Err(qsa_error), _) => Err(qsa_error),
        (Ok(_), Ok(()), Err(restore_error)) => Err(restore_error),
        (Err(shadow_error), Err(qsa_error), Ok(())) => Err(anyhow::anyhow!(
            "K5 stage shadow failed: {shadow_error:#}; QSA restoration also failed: {qsa_error:#}"
        )),
        (Err(shadow_error), Ok(()), Err(restore_error)) => Err(anyhow::anyhow!(
            "K5 stage shadow failed: {shadow_error:#}; recurrent restoration also failed: {restore_error:#}"
        )),
        (Err(shadow_error), Err(qsa_error), Err(restore_error)) => Err(anyhow::anyhow!(
            "K5 stage shadow failed: {shadow_error:#}; QSA restoration failed: {qsa_error:#}; recurrent restoration failed: {restore_error:#}"
        )),
    }
}

fn selected_verified<'a>(batched: &'a [u32], serial: Option<&'a [u32]>) -> &'a [u32] {
    serial.unwrap_or(batched)
}

pub(super) fn exact_accept_prefix(drafts: &[u32], verified: &[u32]) -> usize {
    drafts
        .iter()
        .zip(verified.iter())
        .take_while(|(draft, target)| draft == target)
        .count()
}

/// Exact target-state boundary for a verify frame which finished while its
/// accepted tokens (or its bonus) were being emitted.
///
/// A normal target-only decode keeps the final emitted token in
/// `ActiveSeq::last_token`; that token is not fed to the target until the next
/// scheduler step.  A wide verify has already fed `[last_token, drafts..]`, so
/// a terminal frame must retain exactly as many verify *inputs* as rows it
/// classified, including hidden controls. For an accepted-draft terminal this is
/// `root + emitted_drafts_without_the_final_one`; for a bonus terminal it is
/// `root + every accepted draft`. Callers supply a classified-row interval
/// through the legacy `output_before` / `output_after` plan arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TerminalVerifyCommitPlan {
    pub(super) committed_inputs: usize,
    pub(super) target_seq_len: usize,
    pub(super) last_inter_slot: Option<usize>,
}

pub(super) fn terminal_verify_commit_plan(
    pre_verify_len: usize,
    current_seq_len: usize,
    current_token_count: usize,
    k_verify: usize,
    verifier_accepted: usize,
    output_before: usize,
    output_after: usize,
    accepted_tree_path: Option<&[usize]>,
) -> Result<TerminalVerifyCommitPlan> {
    let committed_inputs = output_after
        .checked_sub(output_before)
        .ok_or_else(|| anyhow::anyhow!("terminal verify output length regressed"))?;
    let max_committed_inputs = verifier_accepted
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("terminal verify accepted-prefix overflow"))?;
    anyhow::ensure!(
        max_committed_inputs <= k_verify,
        "terminal verify accepted {} inputs but K is {k_verify}",
        max_committed_inputs,
    );
    anyhow::ensure!(
        committed_inputs <= max_committed_inputs,
        "terminal verify emitted {committed_inputs} tokens but only {max_committed_inputs} inputs are commit-eligible",
    );
    let expected_current_len = pre_verify_len
        .checked_add(max_committed_inputs)
        .ok_or_else(|| anyhow::anyhow!("terminal verify current length overflow"))?;
    anyhow::ensure!(
        current_seq_len == expected_current_len && current_token_count == expected_current_len,
        "terminal verify rolled-back extent drifted: seq_len={current_seq_len} tokens={current_token_count} expected={expected_current_len}",
    );
    let target_seq_len = pre_verify_len
        .checked_add(committed_inputs)
        .ok_or_else(|| anyhow::anyhow!("terminal verify target length overflow"))?;

    let last_inter_slot = match committed_inputs {
        0 => None,
        1 => Some(0), // root (`last_token`) only
        n => match accepted_tree_path {
            // `n - 1` accepted drafts precede the final, deliberately-unfed
            // emitted token.  Tree compact indices are one-based because row
            // zero is the root.
            Some(path) => Some(*path.get(n - 2).ok_or_else(|| {
                anyhow::anyhow!(
                    "terminal tree path has {} nodes but commit needs {}",
                    path.len(),
                    n - 1,
                )
            })?),
            None => Some(n - 1),
        },
    };

    Ok(TerminalVerifyCommitPlan {
        committed_inputs,
        target_seq_len,
        last_inter_slot,
    })
}

/// Reconcile a terminal wide-verify frame to the same one-token-behind target
/// state as ordinary decoding.  This runs before the terminal return and
/// before tree metadata is cleared, so GDN/PLE commit can still select the
/// exact intermediate produced by this verify.
#[allow(clippy::too_many_arguments)]
pub(super) fn reconcile_terminal_verify_state(
    model: &dyn Model,
    a: &mut ActiveSeq,
    pre_verify_len: usize,
    k_verify: usize,
    verifier_accepted: usize,
    consumed_rows: usize,
    verify_tokens: &[u32],
    tree_mode: bool,
    accepted_tree_path: Option<&[usize]>,
) -> Result<TerminalVerifyCommitPlan> {
    let plan = terminal_verify_commit_plan(
        pre_verify_len,
        a.seq.seq_len,
        a.seq.tokens.len(),
        k_verify,
        verifier_accepted,
        0,
        consumed_rows,
        accepted_tree_path,
    )?;

    if let Some(path) = accepted_tree_path {
        anyhow::ensure!(
            verify_tokens.len() == k_verify && !verify_tokens.is_empty(),
            "terminal tree reconciliation requires the exact verify input frame",
        );
        // Rebuild logical token ownership from the semantic tree path.  The
        // verifier appended compact-slot order, which is not token order when
        // the accepted path crosses a sibling fork.
        a.seq.tokens.truncate(pre_verify_len);
        if plan.committed_inputs > 0 {
            a.seq.tokens.push(verify_tokens[0]);
            for &compact_idx in path.iter().take(plan.committed_inputs - 1) {
                let token = verify_tokens.get(compact_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "terminal tree compact index {compact_idx} is outside K={k_verify}"
                    )
                })?;
                a.seq.tokens.push(*token);
            }
        }
    } else {
        a.seq.tokens.truncate(plan.target_seq_len);
    }
    a.seq.seq_len = plan.target_seq_len;
    anyhow::ensure!(
        a.seq.tokens.len() == a.seq.seq_len,
        "terminal verify token/length ownership drifted after reconciliation",
    );

    // KV compaction is fallible and must run while the QSA journal is still
    // armed. The commit calls below consume QSA as their final action, after
    // which no recoverable operation may fail.
    if let Some(path) = accepted_tree_path {
        let retained = &path[..plan.committed_inputs.saturating_sub(1)];
        if !retained.is_empty()
            && !retained
                .iter()
                .enumerate()
                .all(|(i, &compact)| compact == i + 1)
        {
            model.compact_verify_kv(&a.seq, retained, pre_verify_len)?;
        }
    }

    match plan.last_inter_slot {
        None => {
            // A pre-existing cancellation/error may end the frame before it
            // emits anything. Consume the QSA transaction first so rejected
            // modulo-four raw-ring aliases and compressed groups cannot
            // survive a terminal K5 frame. A zero-accept commit still does not
            // restore live recurrent state, so restore checkpoint -> live
            // explicitly afterward. The caller's subsequent
            // `clear_ddtree_parent_ids` disarms the verify route.
            rollback_failed_verify_frame(model, &mut a.seq, k_verify)?;
        }
        Some(last_inter_slot) if tree_mode => {
            model.commit_verify_state_async_with_slot(
                &mut a.seq,
                plan.committed_inputs,
                k_verify,
                last_inter_slot,
            )?;
        }
        Some(_) => {
            model.commit_verify_state_async(&mut a.seq, plan.committed_inputs, k_verify)?;
        }
    }

    tracing::info!(
        pre_verify_len,
        k_verify,
        verifier_accepted,
        committed_inputs = plan.committed_inputs,
        target_seq_len = plan.target_seq_len,
        last_inter_slot = ?plan.last_inter_slot,
        tree_mode,
        "DFLASH_TERMINAL_COMMIT_RECEIPT"
    );
    Ok(plan)
}

/// DFlash γ-token verify with accept-prefix and full hybrid state commit.
///
/// Routes `[last_token, drafts...]` through the DFlash verify path (including
/// tree verification when `pending_tree_payload` is active, SSM/GDN snapshot
/// commit/rollback on hybrid architectures, hidden-state capture for the next
/// proposer call, and grammar/thinking guards), accepts `0..first_mismatch`
/// plus the verified bonus token at the mismatch position, and drops trailing
/// rejected drafts.
pub fn step_verify_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) {
    // ATLAS_MTP_ACCEPT_LOG=1 committed-acceptance receipt inputs, at entry.
    let thinking_entry = a.inside_thinking;
    let emitted_before = a.output_tokens.len();
    let record_settled = |a: &ActiveSeq,
                          verified_rows: usize,
                          accepted: usize,
                          consumed_rows: usize,
                          terminal: bool| {
        super::spec_accept_log::record_settled_frame(
            &super::spec_accept_log::CommittedFrameReceipt {
                verified_rows,
                drafted: drafts.len(),
                thinking_entry,
                consumed_rows,
                accepted,
                emitted: a.output_tokens.len().saturating_sub(emitted_before),
                terminal,
            },
        );
    };
    let sampled = a.pending_dflash_sampled_proposal.take();
    if model.dflash_draft_sample_method()
        == spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic
    {
        let Some(sampled) = sampled else {
            tracing::error!(
                "probabilistic DFlash draft frame reached verify without its sampled-q proposal"
            );
            proposal_lifecycle::clear_seq(a);
            a.finished = true;
            return;
        };
        step_verify_sampled_dflash(model, a, drafts, num_drafts, think, sampled);
        return;
    }
    if sampled.is_some() {
        tracing::error!("greedy DFlash verifier rejected an unexpected sampled-q proposal");
        proposal_lifecycle::clear_seq(a);
        a.finished = true;
        return;
    }
    // Independent of the optional SPEC_CYCLE_V2 instrumentation: terminal
    // verify-state ownership is derived from the actual response-token delta.
    let (mut spec_cycle, spec_phase) = SpecCycle::begin(a, drafts);
    // Provenance for the retrieval counters only — taken here so an early
    // return cannot misattribute a later frame's acceptances. Nothing
    // downstream reads it, so acceptance is decided exactly as before.
    let draft_origin = std::mem::take(&mut a.draft_origin);
    // ATLAS_DFLASH_STEP_TIMING=1: per-phase wall-clock breakdown of the
    // verify step, logged once per step. Companion to ATLAS_DFLASH_PROPOSE_LOG
    // (which only covers the re-propose at the tail of this function).
    let step_timing = dflash_step_timing_enabled();
    let t_step = Instant::now();

    let sync_result = dflash_capacity::preflight_frame_then(
        model.proposer_is_dflash(),
        || model.dflash_verify_capacity_k(),
        num_drafts,
        drafts.len(),
        a.pending_tree_payload.as_ref(),
        || model.sync_secondary(),
    );
    let sync_result = match sync_result {
        Ok(result) => result,
        Err(error) => {
            tracing::error!("DFlash pre-verify capacity guard rejected the step: {error:#}");
            proposal_lifecycle::clear_seq(a);
            return;
        }
    };
    if let Err(e) = sync_result {
        tracing::error!("sync_secondary: {e:#}");
        fail_speculative_step(a, "verify sync_secondary", &e);
        return;
    }
    let spec_phase = spec_cycle.secondary_wait_enqueue(spec_phase);
    let t_sync_secondary_us = t_step.elapsed().as_micros();

    // EP's variable-width command synchronizes a flat token chain only. Tree
    // topology and sparse branch-compaction metadata are rank-0-local, so a
    // direct caller that bypassed `step_mtp` must still fail closed to the
    // linear drafter spine before constructing/uploading a tree verify.
    if model.is_ep() && a.pending_tree_payload.is_some() {
        tracing::warn!(
            "EP K=gamma verify dropping unsynchronized DDTree payload; using flat spine"
        );
        a.pending_tree_payload = None;
    }

    // Qualification-only target-chain fixture. It is strictly selector-bound,
    // reconstructs the greedy chain with ordinary K1 decode, then restores the
    // exact pre-verify state before substituting a same-width flat draft frame.
    let neutral_greedy = !think.enabled
        && !a.inside_thinking
        && !a.think_ended
        && !a.think_just_ended
        && !a.require_tool_call
        && !a.suppress_tool_call
        && !a.inside_tool_body
        && a.output_tokens.len() >= a.min_tokens
        && !content_policy_required(a, think);
    let fixture_last_token = a.last_token;
    let fixture_draft_len = drafts.len();
    let fixture_flat_frame = a.pending_tree_payload.is_none();
    let fixture_c1_diagnostic = spec_cycle.qualification_c1_active();
    let prepared_fixture = match k16_acceptance_fixture::prepare(
        model,
        &mut a.seq,
        fixture_last_token,
        fixture_draft_len,
        num_drafts,
        fixture_flat_frame,
        fixture_c1_diagnostic,
        neutral_greedy,
    ) {
        Ok(fixture) => fixture,
        Err(error) => {
            tracing::error!("K16 acceptance fixture rejected the step: {error:#}");
            proposal_lifecycle::clear_seq(a);
            a.finished = true;
            return;
        }
    };
    let (mut acceptance_fixture, forced_drafts) = match prepared_fixture {
        Some(prepared) => (Some(prepared.frame), Some(prepared.drafts)),
        None => (None, None),
    };
    if forced_drafts.is_some() {
        super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
    }
    let drafts = forced_drafts.as_deref().unwrap_or(drafts);

    // tokens = [last_verified, draft_0, draft_1, ..., draft_{γ-1}]
    //
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1 (the final piece of the M4B v2 tree-
    // aware verify chain): when a tree payload is present, replace the linear
    // top-1 `drafts` with `payload.tree_token_ids` — the actual tokens the
    // tree topology says live at each compact slot. With this off (default),
    // verify is fed [last_token, ...linear_drafts], which puts semantically
    // wrong tokens at chain-depth slots whenever the tree builder produced
    // sibling/branch nodes (proven to corrupt counting output: "1 to-do"
    // instead of "1 to 60").
    //
    // Source-frame contract: `tokens[0]` is the bonus (kernel slot 0 = root
    // of the tree, always carries `a.last_token`); `tokens[i]` for i >= 1
    // is the token at kernel/compact slot i. `payload.tree_token_ids[i-1]`
    // is the compact-slot-i token by construction (TreePayload skips the
    // bonus). The DFS reorder layer in verify_d.rs permutes these into DFS
    // slots internally via `tokens[dfs_perm[t]]`, and un-permutes verified
    // outputs back to original-compact frame before returning, so the
    // scheduler always sees the kernel-compact frame.
    //
    // Backward compatibility: when chain_only=1 / chain_seed=true with no
    // branching, `tree_token_ids == drafts` and the two paths produce
    // identical bytes. When the tree is non-flat (M4B v2 with branches),
    // this fix puts the RIGHT tokens at the RIGHT slots so the verifier's
    // per-position argmax aligns with the tree topology.
    let tree_tokens_verify = ddtree_tree_tokens_verify_enabled();
    let tree_verify_capable = model.dflash_tree_verify_capable();
    let use_tree_tokens = tree_tokens_verify
        && tree_verify_capable
        && a.pending_tree_payload
            .as_ref()
            .is_some_and(|p| !p.is_empty());

    // Fail closed before widening the target verify. A partial tree path
    // (for example FP8 indirection in only six layers of a mixed BF16/FP8
    // cache) cannot safely commit branches, so K=32 would spend roughly a
    // second verifier width for rows the scheduler must discard. Dropping the
    // payload here keeps the proven flat DFlash path and prevents the later
    // GDN parent upload / tree sampler from seeing mismatched linear tokens.
    if tree_tokens_verify && !tree_verify_capable && a.pending_tree_payload.is_some() {
        static INCAPABLE_TREE_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = INCAPABLE_TREE_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 3 {
            tracing::warn!(
                "DDTree payload dropped before verify: model lacks exact tree support for every \
                 attention/SSM layer; preserving the flat DFlash verifier"
            );
        }
        a.pending_tree_payload = None;
    }

    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    // Tracks the per-compact-slot tokens (excluding the bonus at slot 0)
    // that were ACTUALLY fed to the verifier; emit/compare paths below
    // need this to match the topology-induced acceptance pattern.
    let mut verify_input_tokens: Vec<u32> = Vec::with_capacity(drafts.len());
    if use_tree_tokens {
        // The tree builder caps at `budget` candidates while `drafts.len()
        // == γ_eff` may be larger. Kernel/parent_ids padding (verify_d.rs)
        // appends linear-chain links for indices >= tree_token_ids.len(),
        // so the only sensible tokens for those tail slots are `drafts[i]`
        // (the linear top-1 chain). For indices < tree_token_ids.len(),
        // substitute the tree topology's token. Verifier sees the right
        // token at every kernel slot.
        let payload = a.pending_tree_payload.as_ref().expect("checked above");
        let tree_len = payload.tree_token_ids.len();
        // Verify width = the FULL tree. When the DDTree budget exceeds γ the
        // tree has MORE nodes than `drafts` (wide branching: top-k siblings
        // expanded per depth), so the verifier must process all `tree_len`
        // nodes — not just `drafts.len()`. Truncating to γ would feed nodes
        // whose `parent_indices` reference siblings beyond the verified set,
        // corrupting the tree topology (observed: counting non-lossless +
        // accept collapse). When the tree is NARROWER than γ (budget < γ),
        // chain-pad the tail with the linear top-1 `drafts` so verify still
        // covers γ depths (verify_d.rs pads parent_ids with linear-chain
        // links for indices >= tree_len, matching this padding).
        let verify_len = tree_len.max(drafts.len());
        for i in 0..verify_len {
            let tok = if i < tree_len {
                payload.tree_token_ids[i]
            } else {
                drafts[i]
            };
            tokens.push(tok);
            verify_input_tokens.push(tok);
        }
        static TREE_TOK_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = TREE_TOK_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 3 {
            tracing::info!(
                "ATLAS_DDTREE_TREE_TOKENS_VERIFY #{n}: γ={} tree_len={} tree_tokens[..8]={:?} drafts[..8]={:?} mix[..8]={:?}",
                drafts.len(),
                tree_len,
                &payload.tree_token_ids[..tree_len.min(8)],
                &drafts[..drafts.len().min(8)],
                &verify_input_tokens[..verify_input_tokens.len().min(8)],
            );
        }
    } else {
        tokens.extend_from_slice(drafts);
        verify_input_tokens.extend_from_slice(drafts);
    }

    let stage_pre_verify_len = a.seq.seq_len;
    // Unlike the legacy K16 diagnostic, the sealed K5 commit comparator must
    // select on the untouched steady-state trajectory. It needs no serial
    // logits for acceptance: the K1 replay happens only after production
    // acceptance has fixed total_committed_inputs.
    let k5_selector_selected =
        match model.k16_commit_parity_requested(stage_pre_verify_len, &tokens) {
            Ok(requested) => requested && tokens.len() == 5,
            Err(e) => {
                tracing::error!("DFLASH_K5_COMMIT_PARITY selector invalid: {e:#}");
                a.finished = true;
                return;
            }
        };
    let k5_commit_parity = match k5_commit_parity_route_contract(
        k5_selector_selected,
        spec_cycle.qualification_c1_active(),
        a.pending_tree_payload.is_none(),
        model.is_ep(),
    ) {
        Ok(enabled) => enabled,
        Err(e) => {
            tracing::error!("DFLASH_K5_COMMIT_PARITY route invalid: {e:#}");
            a.finished = true;
            return;
        }
    };
    let stage_capture_requested = match model.k1_stage_diag_requested(stage_pre_verify_len, &tokens)
    {
        Ok(requested) => requested,
        Err(e) => {
            tracing::error!("DFLASH_K1_STAGE_DIAG selector invalid: {e:#}");
            model.k1_stage_diag_abort();
            a.finished = true;
            return;
        }
    };
    let stage_shadow = match k1_stage_shadow_contract(
        stage_capture_requested,
        stage_pre_verify_len,
        &tokens,
        spec_cycle.qualification_c1_active(),
        a.pending_tree_payload.is_none(),
        model.is_ep(),
        tokens.len(),
    ) {
        Ok(enabled) => enabled,
        Err(e) => {
            tracing::error!("DFLASH K1 stage shadow contract rejected: {e:#}");
            model.k1_stage_diag_abort();
            a.finished = true;
            return;
        }
    };
    let serial_oracle_requested = serial_oracle_requested(model, a);
    if tokens.len() > 1 && a.seq.qwen4_qsa_required && (serial_oracle_requested || stage_shadow) {
        // Ordinary K1 decode has no multi-row QSA journal. Running it before a
        // production verify of any width (2..=9 rows) would leave future
        // modulo-four raw rows available to the first endpoint pool and turn
        // a diagnostic into trajectory corruption. Hold these legacy
        // diagnostics until they gain their own QSA transaction instead of
        // emitting false comparison evidence.
        tracing::error!(
            "DFLASH K1 serial/shadow diagnostic is held for QSA-backed K5: transactional K1 replay is not implemented"
        );
        model.k1_stage_diag_abort();
        a.finished = true;
        return;
    }

    // Diagnostic fail-closed oracle. Collect the ordinary one-token target
    // rows BEFORE the batched verifier so the latter still owns the resident
    // logits/capture buffers on the normal path. Legacy serial-commit mode
    // keeps the saved serial rows for acceptance. Shadow mode retains only
    // its K1 argmax vector; neither K1 nor the eager shadow K5 may drive the
    // production acceptance/commit path.
    let (serial_oracle, stage_shadow_k1_verified) = if serial_oracle_requested {
        match collect_serial_oracle(model, &mut a.seq, &tokens) {
            Ok(replay) => (Some(replay), None),
            Err(e) => {
                tracing::error!("DFLASH serial oracle replay failed: {e:#}");
                a.finished = true;
                return;
            }
        }
    } else if stage_shadow {
        match collect_serial_oracle(model, &mut a.seq, &tokens) {
            Ok(replay) => (None, Some(replay.verified)),
            Err(e) => {
                tracing::error!("DFLASH K1 stage shadow serial replay failed: {e:#}");
                model.k1_stage_diag_abort();
                a.finished = true;
                return;
            }
        }
    } else if stage_capture_requested {
        tracing::error!(
            "DFLASH_K1_STAGE_DIAG requires C=1 serial-commit/skip-repropose oracle mode"
        );
        model.k1_stage_diag_abort();
        a.finished = true;
        return;
    } else {
        (None, None)
    };

    // M8A: if a tree payload is present from the previous propose, upload its
    // parent_indices to the model's per-step scratch so the GDN dispatch can
    // fire gdn_tree_k. Cleared after verify completes (Ok or Err).
    if let Some(payload) = a.pending_tree_payload.as_ref()
        && let Err(e) = model.set_ddtree_parent_ids(payload)
    {
        tracing::warn!("set_ddtree_parent_ids failed (falling back to flat): {e:#}");
    }

    // EP rank 0 owns proposal and acceptance policy, but all ranks must enter
    // the same arbitrary-K target forward for MoE collectives. Send one framed
    // command and a bulk token payload before invoking the verifier locally.
    if model.is_ep() {
        let broadcast = model
            .ep_broadcast_cmd(EP_CMD_VERIFY_KGAMMA)
            .and_then(|()| model.ep_broadcast_cmd(tokens.len() as u32))
            .and_then(|()| model.ep_broadcast_tokens(&tokens).map(|_| ()));
        if let Err(e) = broadcast {
            tracing::error!("EP broadcast K=gamma verify frame: {e:#}");
            fail_speculative_step(a, "verify EP frame broadcast", &e);
            return;
        }
    }

    let spec_phase = spec_cycle.setup(spec_phase);
    // Arm only the already-selected one-shot fixture, after its serial replay
    // has restored this exact frame and immediately before production verify.
    // The guard aborts both receipts on every early-return path.
    let mut fixture_evidence = if acceptance_fixture.is_some() {
        match k16_acceptance_fixture::EvidenceGuard::begin(model, a.seq.seq_len, &tokens) {
            Ok(guard) => Some(guard),
            Err(error) => {
                tracing::error!("K16 acceptance fixture evidence arm failed: {error:#}");
                model.k16_fixture_evidence_abort();
                model.clear_ddtree_parent_ids();
                a.finished = true;
                return;
            }
        }
    } else {
        None
    };
    let stage_shadow_verified = if stage_shadow {
        let shadow_verified = run_k5_stage_shadow(model, &mut a.seq, &tokens);
        let shadow_verified = match shadow_verified {
            Ok(verified) => verified,
            Err(e) => {
                tracing::error!("DFLASH K1 stage shadow K5/restore failed: {e:#}");
                model.k1_stage_diag_abort();
                model.clear_ddtree_parent_ids();
                a.finished = true;
                return;
            }
        };
        let k1_verified = stage_shadow_k1_verified
            .as_deref()
            .expect("stage shadow serial replay is present");
        tracing::info!(
            schema = 1,
            pre_verify_len = stage_pre_verify_len,
            tokens = ?tokens,
            k1_verified = ?k1_verified,
            shadow_verified = ?shadow_verified,
            restored_seq_len = a.seq.seq_len,
            restored_token_count = a.seq.tokens.len(),
            production_pending = true,
            "DFLASH_K1_STAGE_SHADOW_RESTORED"
        );
        Some(shadow_verified)
    } else {
        None
    };

    let production_pre_verify_len = a.seq.seq_len;
    let production_pre_verify_tokens = a.seq.tokens.len();
    let rejection_policy = super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection;
    if serial_oracle.is_none() && a.pending_tree_payload.is_none() {
        if rejection_policy {
            model.stage_next_verify_gumbel_rows(&[]);
            super::spec_policy_accept::stage_single_seq_rejection_if_eligible(
                model,
                a,
                &verify_input_tokens,
                tokens.len(),
                think,
            );
        } else {
            model.stage_next_verify_rejection_rows(&[]);
            super::spec_policy_accept::stage_single_seq_gumbel_if_eligible(
                model,
                a,
                tokens.len(),
                think,
            );
        }
    } else {
        model.stage_next_verify_rejection_rows(&[]);
    }
    let t_verify = Instant::now();
    let mut verified = match model.decode_verify_dflash(&tokens, &mut a.seq, 0) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("decode_verify_dflash: {e:#}");
            let _ = model.take_in_graph_rejection_verify_words();
            super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            a.seq.seq_len = production_pre_verify_len;
            a.seq.tokens.truncate(production_pre_verify_tokens);
            if let Err(rollback_error) =
                rollback_failed_verify_frame(model, &mut a.seq, tokens.len())
            {
                tracing::error!("decode_verify_dflash abort rollback failed: {rollback_error:#}");
            }
            model.k1_stage_diag_abort();
            if model.is_ep() {
                let _ = model.ep_broadcast_cmd(EP_VERIFY_KGAMMA_ABORT);
            }
            model.clear_ddtree_parent_ids();
            fail_speculative_step(a, "decode_verify_dflash", &e);
            return;
        }
    };
    let spec_phase = spec_cycle.verify_complete(spec_phase);
    let t_verify_us = t_verify.elapsed().as_micros();
    if verified.len() != tokens.len() {
        tracing::error!(
            expected = tokens.len(),
            actual = verified.len(),
            "K=gamma verifier returned the wrong row count"
        );
        let _ = model.take_in_graph_rejection_verify_words();
        super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
        a.seq.seq_len = production_pre_verify_len;
        a.seq.tokens.truncate(production_pre_verify_tokens);
        if let Err(rollback_error) = rollback_failed_verify_frame(model, &mut a.seq, tokens.len()) {
            tracing::error!("wrong-row-count abort rollback failed: {rollback_error:#}");
        }
        if model.is_ep() {
            let _ = model.ep_broadcast_cmd(EP_VERIFY_KGAMMA_ABORT);
        }
        model.k1_stage_diag_abort();
        model.clear_ddtree_parent_ids();
        fail_speculative_step(
            a,
            "verify target row count",
            format!(
                "expected {} target rows, received {}",
                tokens.len(),
                verified.len()
            ),
        );
        return;
    }

    if let Some(shadow_verified) = stage_shadow_verified.as_deref() {
        let k1_verified = stage_shadow_k1_verified
            .as_deref()
            .expect("stage shadow serial replay is present");
        tracing::info!(
            schema = 1,
            pre_verify_len = stage_pre_verify_len,
            tokens = ?tokens,
            k1_verified = ?k1_verified,
            shadow_verified = ?shadow_verified,
            production_verified = ?verified,
            shadow_matches_production = shadow_verified == verified.as_slice(),
            acceptance_source = "production",
            "DFLASH_K1_STAGE_PRODUCTION"
        );
    }

    let presampled_rejection =
        if rejection_policy && serial_oracle.is_none() && a.pending_tree_payload.is_none() {
            let res = super::spec_policy_accept::presample_single_seq_rejection(
                model,
                a,
                &verify_input_tokens,
                0,
                verified.len(),
                think,
            );
            super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            match res {
                Ok(words) => words,
                Err(e) => {
                    tracing::error!("single-seq rejection verify: {e:#}");
                    a.seq.seq_len = production_pre_verify_len;
                    a.seq.tokens.truncate(production_pre_verify_tokens);
                    if let Err(rollback_error) =
                        rollback_failed_verify_frame(model, &mut a.seq, tokens.len())
                    {
                        tracing::error!(
                            "single-seq rejection verify abort rollback failed: {rollback_error:#}"
                        );
                    }
                    if model.is_ep() {
                        let _ = model.ep_broadcast_cmd(EP_VERIFY_KGAMMA_ABORT);
                    }
                    model.k1_stage_diag_abort();
                    model.clear_ddtree_parent_ids();
                    fail_speculative_step(a, "single-seq rejection verify", &e);
                    return;
                }
            }
        } else {
            let _ = model.take_in_graph_rejection_verify_words();
            super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            None
        };

    // ── Verify-time think mask (throughput fix) ──
    // The verify oracle above returns the target's RAW unmasked argmax, but the
    // plain sample path suppresses `<think>`/`</think>` once thinking has ended.
    // A raw `<think>` argmax at a verify position scores the drafter's correct
    // content guess against `<think>` and REJECTS it, collapsing the γ chain to
    // a 0-accept (measured: `<think>` is the single most common first-token miss
    // on coding). Re-derive those positions as the masked content argmax so the
    // verified stream matches the masked plain path — turning forced rejections
    // into accepts. Runs BEFORE the accept/bonus logic so every consumer sees
    // the corrected stream. Gated on think_ended (never inside an active span).
    // Preserve raw batch rows for the diagnostic, select serial when active,
    // then mask the selected vector from its matching logits rows.
    let batched_verified_for_oracle = serial_oracle.as_ref().map(|_| verified.clone());
    verified = selected_verified(
        &verified,
        serial_oracle
            .as_ref()
            .map(|oracle| oracle.verified.as_slice()),
    )
    .to_vec();
    let think_masked = if let Some(ref oracle) = serial_oracle {
        dflash_think_mask_verified_rows(a, &mut verified, oracle.logits_format, |i, buf| {
            match oracle.logits_rows.get(i) {
                Some(row) => {
                    buf.clear();
                    buf.extend_from_slice(row);
                    true
                }
                None => false,
            }
        })
    } else if presampled_rejection.is_some() {
        0
    } else {
        dflash_think_mask_verified(model, a, &mut verified)
    };
    if think_masked > 0 {
        tracing::debug!("think mask: verified patched {think_masked} position(s) this step");
    }
    if let Some(fixture) = acceptance_fixture.as_mut()
        && let Err(error) = fixture.observe(drafts, &verified, serial_oracle.is_some())
    {
        tracing::error!("K16 acceptance fixture target-chain mismatch: {error:#}");
        a.seq.seq_len = production_pre_verify_len;
        a.seq.tokens.truncate(production_pre_verify_tokens);
        if let Err(rollback_error) = rollback_failed_verify_frame(model, &mut a.seq, tokens.len()) {
            tracing::error!("acceptance-fixture abort rollback failed: {rollback_error:#}");
        }
        model.k1_stage_diag_abort();
        model.clear_ddtree_parent_ids();
        a.finished = true;
        return;
    }
    // Note: clear_ddtree_parent_ids is deferred until AFTER
    // commit_verify_state_async so the commit knows tree mode was active
    // and routes through the partial-accept (intermediate→h_state) path.
    a.last_token_time = Instant::now();
    if super::defer_emit::dflash_defer_emit_enabled() {
        a.defer_emit.begin();
    }

    // ATLAS_DFLASH_FUSED=1: record the propose-ordering event NOW, before
    // commit kernels are enqueued. The drafter reads dflash_hidden_save
    // (populated by verify) and ctx_hidden_acc (per-sequence). Commit writes
    // h_state and KV cache — disjoint. This lets the propose stream run in
    // parallel with the ~10ms SSM commit + KV reshape.
    if let Err(e) = model.dflash_arm_propose_overlap() {
        tracing::warn!("dflash_arm_propose_overlap: {e:#}");
    }

    // ── ATLAS_DFLASH_BRANCH_AUDIT=1: settle the previous step's stash ──
    // `verified[0]` is the target's TRUE argmax after `tokens[0]`
    // (= the committed bonus of the previous step). If the previous step
    // stashed a fork-row prediction for that exact context, compare it now.
    // Only judged when the committed token actually IS the stashed fork
    // token (grammar/think interventions could have replaced it).
    if let Some((expected, fork_tok)) = BRANCH_AUDIT_STASH
        .get()
        .and_then(|stash| stash.lock().ok()?.remove(&a.session_hash))
        && tokens[0] == fork_tok
        && let Some(&actual) = verified.first()
    {
        let is_match = actual == expected;
        let (m, mm) = {
            let mut t = BRANCH_AUDIT_TALLY.lock().unwrap_or_else(|e| e.into_inner());
            if is_match {
                t.0 += 1;
            } else {
                t.1 += 1;
            }
            *t
        };
        tracing::info!(
            "BRANCH_AUDIT: fork_tok={fork_tok} expected_after_fork={expected} \
             true_after_fork={actual} match={} tally={m}/{}",
            is_match as u8,
            m + mm,
        );
    }

    // `decode_verify` already advanced `seq.seq_len` by `tokens.len()` and
    // pushed all γ+1 tokens into `seq.tokens`. The accept-prefix logic below
    // determines how many to keep — the rest must be rolled back so the
    // KV cache, SSM state, and emitted token sequence stay consistent.

    // Accept-prefix: drafts[i] is "accepted" iff drafts[i] == verified[i].
    // verified[i] is the target's argmax at position i (i.e. its
    // prediction for what should follow `tokens[i]`). drafts[i] was the
    // proposer's guess for the same slot. First mismatch terminates the
    // accepted prefix; verified[first_mismatch] becomes the bonus token.
    //
    // DDTree mode: when `a.pending_tree_payload` is `Some`, the drafts/verified
    // pair corresponds to compact tree-row indices rather than a flat chain.
    // The tree walker in `spark_model::layers::ddtree::greedy_sample_ddtree`
    // walks the tree topology and returns the flat-prefix-safe accepted compact
    // indices plus the bonus token, keeping KV cache slots contiguous across
    // rollback.
    // correctly truncates when greedy walk diverges from compact-index-order.
    // For flat-chain payloads it degenerates to the linear loop below.
    //
    // Returns `(num_accepted, tree_last_inter_slot)`:
    //   - `num_accepted`: count of drafts accepted (excludes the prefix
    //     bonus, which is always "accepted" downstream).
    //   - `tree_last_inter_slot`: `Some(slot)` when the tree-aware GDN
    //     kernel was used → the kernel intermediate slot of the last
    //     accepted state (== max compact index, or 0 when no drafts
    //     accepted). `None` for flat-chain → commit derives slot from
    //     `num_accepted - 1` (chain-contiguous).
    // Stage-2 grammar enforcement (ATLAS_DFLASH_GRAMMAR_MODE=verify): the
    // `verified` vec came from an UNMASKED GPU argmax, so for a
    // grammar-constrained sequence each position's target token must be
    // recomputed as masked argmax over the verify logits (still resident in
    // the model's `[K, vocab]` BF16 logits buffer at this point — nothing
    // touches the model between the verify above and here). `Some((n, b))`
    // overrides the accept count and bonus token below; `None` falls
    // through to the unmasked paths (grammar inactive / tree mode / fp32
    // logits / D2H failure).
    // ATLAS_THINK_SPEC=1 thinking-span accept filter (sibling of the
    // grammar path below, same walk-and-truncate shape): re-derives the
    // PLAIN-path token per verify position — F1 reflection suppression,
    // F2 confidence early stop, efficiency wave, tool-call mask, forced
    // `</think>` injection, in-thinking EOS suppression — and truncates
    // acceptance at the first divergence or phase boundary. The walk
    // COMMITS (streams) its accepted tokens + bonus itself, because the
    // in-thinking side effects (suppressed EOS, `</think>` transition,
    // fence parity, THINK_LOOP watchdog) don't fit the plain emit loop
    // below. Mutually exclusive with the grammar path: grammar masking is
    // suspended inside `<think>` (dflash_masked_accept bails on
    // `inside_thinking`), and tree payloads are cleared for thinking
    // sequences in step_mtp before verify.
    let thinking_accept: Option<SpecAcceptOutcome> =
        if think.enabled && a.inside_thinking && a.pending_tree_payload.is_none() {
            if let Some(ref oracle) = serial_oracle {
                if oracle.logits_format == VerifyLogitsFormat::Fp32 {
                    tracing::error!(
                        "DFLASH serial oracle cannot run BF16 think policy over FP32 logits"
                    );
                    a.finished = true;
                    Some(SpecAcceptOutcome {
                        num_accepted: 0,
                        bonus: None,
                        consumed_rows: 0,
                    })
                } else {
                    Some(dflash_thinking_accept(
                        a,
                        &verify_input_tokens,
                        &verified,
                        think,
                        |i, buf| match oracle.logits_rows.get(i) {
                            Some(row) => {
                                buf.clear();
                                buf.extend_from_slice(row);
                                true
                            }
                            None => false,
                        },
                        // Defer the SSM boundary snapshot until after verify
                        // commit + `sync_secondary()` via `outcome_ended_thinking`.
                        |_| {},
                    ))
                }
            } else if presampled_rejection.is_some() {
                super::spec_policy_accept::run_dflash_thinking_rejection_accept_at_row(
                    model,
                    a,
                    0,
                    &verify_input_tokens,
                    &verified,
                    think,
                    presampled_rejection.as_deref(),
                )
            } else {
                run_dflash_thinking_accept(model, a, &verify_input_tokens, &verified, think)
            }
        } else {
            None
        };

    // Outside thinking, raw target argmax is only an oracle for a neutral
    // greedy sampler.  Qwen's default presence/LZ/DRY penalties are
    // non-neutral, and `require_tool_call` / `inside_tool_body` / `min_tokens`
    // enforce EOS-suppression and tool-close boundaries (matching
    // `verify_qwen4_batched_step.rs`), so re-sample each flat verify row
    // through the same policy pipeline as serial decode and truncate at the
    // first true mismatch.
    let content_accept: Option<SpecAcceptOutcome> = if thinking_accept.is_none()
        && !a.inside_thinking
        && a.pending_tree_payload.is_none()
        && flat_content_policy_required(a, think)
    {
        if let Some(ref oracle) = serial_oracle {
            Some(dflash_content_accept(
                a,
                &verify_input_tokens,
                &verified,
                think,
                oracle.logits_format,
                model.vocab_size(),
                |i, buf| match oracle.logits_rows.get(i) {
                    Some(row) => {
                        buf.clear();
                        buf.extend_from_slice(row);
                        true
                    }
                    None => false,
                },
            ))
        } else if presampled_rejection.is_some() {
            super::spec_policy_accept::run_dflash_content_accept_for_frame_with_rejection(
                model,
                a,
                &tokens,
                production_pre_verify_len,
                &verify_input_tokens,
                &verified,
                think,
                presampled_rejection.as_deref(),
            )
        } else {
            run_dflash_content_accept_for_frame(
                model,
                a,
                &tokens,
                production_pre_verify_len,
                &verify_input_tokens,
                &verified,
                think,
            )
        }
    } else {
        None
    };

    let grammar_accept: Option<(usize, u32)> = if thinking_accept.is_none()
        && content_accept.is_none()
        && dflash_grammar_mode() == DflashGrammarMode::Verify
        && a.pending_tree_payload.is_none()
    {
        dflash_masked_accept(model, a, &verify_input_tokens, &verified)
    } else {
        None
    };

    // FIX 1 — Tree-path commit (ATLAS_DFLASH_TREE_COMMIT=1, default off).
    // When active for a tree payload, the greedy walk commits the WHOLE
    // accepted path (incl. a sibling-fork tail), not just the contiguous flat
    // prefix. `tree_accepted_path` carries the (possibly non-contiguous)
    // compact-index path + the bonus row so the emit + KV-compaction below
    // can lay tokens/KV correctly. `None` on every flat-chain / non-tree path
    // → legacy contiguous behavior unchanged.
    let tree_commit_enabled = dflash_tree_commit_enabled();
    // 2026-07-08 deep-branch-tail root-cause guard: the FULL walker consumes
    // branch rows' argmaxes (child-acceptance + bonus at the path tip), so it
    // is only sound when the verify that just ran gave EVERY tree node
    // ancestor-exact attention (per-row KV indirection —
    // ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1). Under
    // prefix-read metadata (incl. ATLAS_DDTREE_DFS_REORDER=1) only the spine
    // rows are exact: a branch row reads its spine SIBLING and misses its own
    // key, so committing through it diverges from the greedy oracle
    // (VALIDATION-36 TEST 1 md5 corruption). Degrade to the flat-safe walker,
    // which never reads a branch row.
    let ancestor_attn_exact = model.dflash_tree_ancestor_attn_exact();
    let tree_commit_active = tree_commit_enabled && ancestor_attn_exact;
    if tree_commit_enabled && !ancestor_attn_exact && a.pending_tree_payload.is_some() {
        static DOWNGRADE_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = DOWNGRADE_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 3 {
            tracing::warn!(
                "ATLAS_DFLASH_TREE_COMMIT=1 but the verify was NOT ancestor-exact \
                 (prefix/DFS metadata on a non-flat tree) — deep-branch commit would \
                 corrupt; degrading to the flat-safe walker. Set \
                 ATLAS_DDTREE_TREE_AWARE_VERIFY=1 + ATLAS_TREE_AWARE_ATTN=1 (and unset \
                 ATLAS_DDTREE_DFS_REORDER) for lossless deep-branch commits."
            );
        }
    }
    let mut tree_accepted_path: Option<(Vec<usize>, usize)> = None;
    let committed_accept = thinking_accept.as_ref().or(content_accept.as_ref());
    let deferred_think_end_snapshot = thinking_accept
        .as_ref()
        .is_some_and(|t| super::think_spec_accept::outcome_ended_thinking(a, t));
    let (num_accepted, tree_last_inter_slot) = if let Some(t) = committed_accept {
        (t.num_accepted, None)
    } else if let Some((n, _)) = grammar_accept {
        (n, None)
    } else if let Some(payload) = a.pending_tree_payload.as_ref() {
        use spark_model::layers::dflash_head::ddtree::{
            DDTreeRequestRuntime, greedy_sample_ddtree, greedy_sample_ddtree_full,
            last_accepted_inter_slot,
        };
        let req = DDTreeRequestRuntime {
            req_id: String::new(),
            tree_token_ids: payload.tree_token_ids.clone(),
            parent_indices: payload.parent_indices.clone(),
        };
        // greedy_sample_ddtree expects argmax for [root_row, ...compact_rows].
        // Our `verified` is exactly [verify_at_last_token, verify_at_drafts[0],
        // ..., verify_at_drafts[n-1]] — so verified[0..=req.num_nodes()] is
        // what the sampler wants.
        let expected_rows = 1 + req.num_nodes();
        let argmax = if verified.len() >= expected_rows {
            verified[..expected_rows].to_vec()
        } else {
            // Defensive: pad with last-known to avoid length mismatch.
            let mut a = verified.to_vec();
            a.resize(expected_rows, *verified.last().unwrap_or(&0));
            a
        };
        // ── ATLAS_DDTREE_CEILING_LOG=1: branch-headroom measurement ──
        // Counterfactual: on ANY run (incl. the lossless flat-safe path),
        // compute what the FULL tree walk would accept vs the flat (top-1
        // chain) prefix. tree_depth > flat_depth means a sibling branch
        // matched the target argmax where the top-1 chain diverged — i.e.
        // real branch headroom. This is pure observation (nothing committed),
        // so downstream state stays clean and the numbers are uncontaminated
        // by the fork-commit bug. The gap is the ceiling on branching gain.
        if std::env::var("ATLAS_DDTREE_CEILING_LOG").ok().as_deref() == Some("1") {
            let full = greedy_sample_ddtree_full(&req, &argmax)
                .map(|s| s.accepted_compact_indices.len())
                .unwrap_or(0);
            // Flat (top-1 chain) depth: contiguous prefix of the full walk.
            let flat = greedy_sample_ddtree(&req, &argmax)
                .map(|s| s.accepted_compact_indices.len())
                .unwrap_or(0);
            // RETRIEVAL ceiling: the target's argmax at the FIRST divergence
            // (row `flat`, i.e. where the top-1 chain stopped). Is that token
            // present in recent generated context? If so, a retrieval / n-gram
            // sibling (Graft, arXiv 2605.20104) could supply it where the
            // diffusion drafter's marginals can't. Measures the repetitive-
            // span headroom that diffusion branching alone (branch_gain) misses.
            let div_tok = argmax.get(flat).copied();
            let retr_hit = match div_tok {
                Some(t) => {
                    let toks = &a.seq.tokens;
                    let lo = toks.len().saturating_sub(1024);
                    toks[lo..].contains(&t)
                }
                None => false,
            };
            tracing::info!(
                "DDTREE_CEILING: flat_depth={flat} tree_depth={full} branch_gain={} \
                 retr_hit={} tree_nodes={}",
                full as i64 - flat as i64,
                retr_hit as u8,
                req.tree_token_ids.len(),
            );
        }
        // ── ATLAS_DFLASH_BRANCH_AUDIT=1: cross-step conditioning audit ──
        // Pure observation (commits nothing). When the flat-safe walk stops
        // at a divergence whose fork token the FULL walk would take, this
        // step commits [spine prefix] + bonus == the fork token, so the NEXT
        // verify's root row (verified[0]) is the target's TRUE argmax after
        // the fork under exact flat conditioning. The fork ROW of THIS step
        // claims that same conditioning ([ctx, spine prefix, fork]) — its
        // argmax must therefore equal next step's verified[0]. Mismatches
        // measure exactly the branch-row mis-conditioning that corrupts the
        // deep tree-commit; ~100% match certifies ancestor-exact attention.
        // Stash keyed by session hash (debug-only; batch=1 serve config).
        let audit_enabled = std::env::var("ATLAS_DFLASH_BRANCH_AUDIT").ok().as_deref() == Some("1");
        if audit_enabled
            && !tree_commit_active
            && let Ok(full) = greedy_sample_ddtree_full(&req, &argmax)
            && let Ok(flat) = greedy_sample_ddtree(&req, &argmax)
        {
            let flat_n = flat.accepted_compact_indices.len();
            let fullp = &full.accepted_compact_indices;
            if fullp.len() > flat_n {
                let fork_c = fullp[flat_n];
                let fork_tok = req.tree_token_ids.get(fork_c.saturating_sub(1)).copied();
                let expected = argmax.get(fork_c).copied();
                if let (Some(ft), Some(exp)) = (fork_tok, expected) {
                    let mut stash = branch_audit_stash()
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if stash.len() >= BRANCH_AUDIT_STASH_CAP && !stash.contains_key(&a.session_hash)
                    {
                        stash.clear();
                    }
                    stash.insert(a.session_hash, (exp, ft));
                }
            }
        }
        // FIX 1: full-path sampler when tree-commit is enabled AND the verify
        // was ancestor-exact (see the guard above), else the flat-safe
        // sampler (spine rows only — always correctly conditioned).
        let sample_res = if tree_commit_active {
            greedy_sample_ddtree_full(&req, &argmax)
        } else {
            greedy_sample_ddtree(&req, &argmax)
        };
        match sample_res {
            Ok(sample) => {
                let n = sample.accepted_compact_indices.len();
                // Derive the kernel-frame slot from the actual compact
                // indices, NOT from `n - 1`. In chain-only mode these are
                // equal; for a fork-crossing path they diverge
                // (e.g. [1, 2, 7] → slot 7, not 2).
                let slot = last_accepted_inter_slot(&sample.accepted_compact_indices);
                // Record the sparse path for the emit + KV-compaction below.
                // The bonus row is the compact tip the walk ended at
                // (== last accepted compact index, or 0 if nothing accepted).
                if tree_commit_active {
                    let bonus_row = sample.accepted_compact_indices.last().copied().unwrap_or(0);
                    tree_accepted_path = Some((sample.accepted_compact_indices.clone(), bonus_row));
                }
                (n, Some(slot))
            }
            Err(e) => {
                tracing::warn!("greedy_sample_ddtree failed ({e}); falling back to linear");
                // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1: compare against the
                // tokens the verifier actually saw at each slot
                // (verify_input_tokens, which mixes tree_token_ids for
                // the first tree_len slots + drafts for the chain-padded
                // tail). When use_tree_tokens is false, verify_input_tokens
                // == drafts so this matches legacy chain-arithmetic.
                let mut n = 0usize;
                for i in 0..verify_input_tokens.len() {
                    if i + 1 >= verified.len() {
                        break;
                    }
                    if verify_input_tokens[i] == verified[i] {
                        n += 1;
                    } else {
                        break;
                    }
                }
                // Fallback path is chain-arithmetic but the GDN kernel
                // that just ran was still tree-aware — slot 0 if nothing
                // accepted, else `n` (compact-index of last linear-prefix
                // node == n under chain-equivalent payload).
                let slot = if n == 0 { 0 } else { n };
                (n, Some(slot))
            }
        }
    } else {
        // Relaxed acceptance (ATLAS_DFLASH_RELAX_ACCEPT=1, default off,
        // works at temp0 AND temp>0): accept a near-miss DRAFT token (and
        // COMMIT IT, not the argmax) when it is a high-probability token
        // under the TARGET — within the target's top-k OR with
        // p(draft)/p(argmax) >= ratio at the would-be-mismatch position.
        // Quality is preserved because we only ever commit a token the
        // target itself ranks highly; the PPL guardrail bounds the drift.
        // `None` ⇒ inactive ⇒ fall through to typical-accept, then to the
        // legacy exact-match prefix.
        //
        // Tried BEFORE typical-accept: relaxed is the more general gate
        // (subsumes typical's α·p_max test as a special case) and is the
        // one that also fires at temp0.
        let n = if serial_oracle.is_some() {
            // The serial diagnostic is an exact, fail-closed oracle. Never
            // widen its acceptance with batch-resident relaxed/typical logits.
            exact_accept_prefix(drafts, &verified)
        } else {
            match dflash_relax_accept(model, a, drafts, &verified) {
                Some(n) => n,
                None => match dflash_typical_accept(model, a, drafts, &verified) {
                    Some(n) => n,
                    None => exact_accept_prefix(drafts, &verified),
                },
            }
        };
        (n, None)
    };

    // Per-position acceptance diagnostics (ATLAS_MTP_ACCEPT_LOG=1, inert
    // otherwise). Draft slot i compares against verified[i] — the target's
    // argmax row for that slot, the same equality `exact_accept_prefix` uses.
    // Logged after acceptance is derived so `num_accepted` is the final,
    // policy/grammar/tree-adjusted count. seq_len is the pre-verify length
    // (the verifier has over-extended it by tokens.len() at this point).
    record_accept(
        if model.proposer_is_dflash() {
            "dflash"
        } else {
            "mtp"
        },
        a.seq.seq_len.saturating_sub(tokens.len()),
        num_drafts,
        drafts,
        &verified,
        num_accepted,
    );
    // Release the EP worker immediately after rank 0 derives acceptance. A
    // policy/D2H failure cannot commit a valid prefix, so send the abort
    // sentinel; otherwise the worker mirrors seq trim + recurrent checkpoint
    // from the accepted-draft count before returning to its command loop.
    if model.is_ep() {
        let policy_walk_active = committed_accept.is_some();
        let bonus_available = committed_accept.is_none_or(|outcome| outcome.bonus.is_some());
        let result = ep_kgamma_result(
            num_accepted,
            policy_walk_active,
            a.finished,
            bonus_available,
        );
        let policy_failed = result == EP_VERIFY_KGAMMA_ABORT;
        if let Err(e) = model.ep_broadcast_cmd(result) {
            tracing::error!("EP broadcast K=gamma verify result: {e:#}");
            a.seq.seq_len = production_pre_verify_len;
            a.seq.tokens.truncate(production_pre_verify_tokens);
            if let Err(rollback_error) =
                rollback_failed_verify_frame(model, &mut a.seq, tokens.len())
            {
                tracing::error!("EP-result abort rollback failed: {rollback_error:#}");
            }
            model.clear_ddtree_parent_ids();
            flush_deferred_emit(a);
            fail_speculative_step(a, "verify EP result broadcast", &e);
            return;
        }
        if policy_failed {
            a.seq.seq_len = production_pre_verify_len;
            a.seq.tokens.truncate(production_pre_verify_tokens);
            if let Err(rollback_error) =
                rollback_failed_verify_frame(model, &mut a.seq, tokens.len())
            {
                tracing::error!("EP-policy abort rollback failed: {rollback_error:#}");
            }
            model.clear_ddtree_parent_ids();
            flush_deferred_emit(a);
            return;
        }
    }

    // Roll back the over-extended `seq_len` and `seq.tokens`. The verify
    // advanced both by `tokens.len() = γ+1` (all γ drafts + the prefix
    // bonus slot). We keep the original prefix + `num_accepted` drafts +
    // 1 bonus position. So the post-rollback target is
    // `pre_verify_len + num_accepted + 1` — note we do NOT push the bonus
    // again via emit_token's path (emit_token only updates the user-facing
    // output buffer, not seq.tokens), so the bonus stays in seq.tokens
    // exactly where decode_verify put it.
    let pre_verify_len = a.seq.seq_len.saturating_sub(tokens.len());
    let target_seq_len = pre_verify_len + num_accepted + 1;
    let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
    if to_drop > 0 {
        a.seq.seq_len = target_seq_len;
        let pop_n = to_drop.min(a.seq.tokens.len());
        for _ in 0..pop_n {
            a.seq.tokens.pop();
        }
    }

    // ── ATLAS_DFLASH_EARLY_EXIT_PROFILE=1: per-step accept counter ──
    // Logs the accept count for THIS K=γ verify step so the early-exit
    // sweep can read mean accept/γ straight from the server log. Cheap
    // (one log line per verify); gated so it is a no-op in production.
    if dflash_early_exit_profile_enabled() {
        tracing::info!(
            "DFLASH_EE_VERIFY: accepted={num_accepted}/{} drafts[..min(6)]={:?}",
            drafts.len(),
            &drafts[..drafts.len().min(6)],
        );
    }

    tracing::debug!(
        "DFLASH K=γ verify: γ={} accepted={}/{} ({:.0}%) seq_len={} verify_input={:?} verified={:?}",
        drafts.len(),
        num_accepted,
        drafts.len(),
        100.0 * (num_accepted as f64) / (drafts.len() as f64),
        a.seq.seq_len,
        verify_input_tokens,
        verified,
    );

    // One-sample evidence of how the drafter is doing on this text. Read
    // by the retrieval tiers on the NEXT propose, never by this frame.
    a.last_verify_accepted = u16::try_from(num_accepted).unwrap_or(u16::MAX);

    match draft_origin {
        DraftOrigin::RestStore => crate::rest_store::record_accepted(num_accepted),
        DraftOrigin::SelfContext => crate::rest_store::self_context::record_accepted(num_accepted),
        DraftOrigin::Proposer => {}
    }

    // Commit geometry is needed by both ordinary completion and the terminal
    // exits inside the emission walk below.
    let k_verify = tokens.len();
    let total_accepted = num_accepted + 1; // root is always accepted
    let last_inter_slot = match tree_last_inter_slot {
        Some(slot) => slot,
        None => total_accepted.saturating_sub(1),
    };
    let tree_mode = tree_last_inter_slot.is_some();
    let accepted_tree_path = tree_accepted_path.as_ref().map(|(path, _)| path.as_slice());

    // Emit accepted drafts.
    //
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1: when the verify input was built
    // from `tree_token_ids` (not `drafts`), the accepted tokens at compact
    // slots [1..=num_accepted] are `tree_token_ids[0..num_accepted]`, NOT
    // `drafts[0..num_accepted]`. greedy_sample_ddtree returns
    // accepted_compact_indices in flat-prefix-safe form ([1, 2, ..., n]),
    // so compact slot i+1 corresponds to tree_token_ids[i]. With this off
    // (default), the old `drafts[i]` emission stays — backward compatible
    // when tree_token_ids == drafts (flat-chain payloads).
    // Emit the tokens that were actually accepted. With
    // ATLAS_DDTREE_TREE_TOKENS_VERIFY=1, accepted_compact_indices = [1..=n]
    // (flat-prefix-safe contract) and the token at compact slot i+1 is
    // verify_input_tokens[i] (which = tree_token_ids[i] for i < tree_len
    // else drafts[i] for the chain-padded tail). With the env off,
    // verify_input_tokens == drafts so this is identical to the legacy
    // emission. Defensive bound: take from drafts if i exceeds vector.
    // FIX 1: when the tree-commit path is active, the accepted compact
    // indices may be NON-contiguous (a sibling fork tail). Look up each
    // accepted token by its compact index (compact slot `c` carries
    // `verify_input_tokens[c-1]`) rather than slicing the contiguous prefix.
    // Falls back to the legacy contiguous slice on every flat/non-tree path.
    if let Some(t) = committed_accept {
        // The policy walk already committed/streamed its accepted prefix
        // AND bonus with serial sampler semantics. A missing bonus means
        // the sequence finished mid-emission (stream drop / cancel / D2H
        // failure), so bail out like the legacy emit loop.
        if a.finished || t.bonus.is_none() {
            if let Err(e) = reconcile_terminal_verify_state(
                model,
                a,
                pre_verify_len,
                k_verify,
                num_accepted,
                t.consumed_rows,
                &tokens,
                tree_mode,
                accepted_tree_path,
            ) {
                tracing::error!("terminal verify-state commit (policy walk): {e:#}");
                a.seq.seq_len = pre_verify_len;
                a.seq.tokens.truncate(pre_verify_len);
                if let Err(rollback_error) =
                    rollback_failed_verify_frame(model, &mut a.seq, k_verify)
                {
                    tracing::error!(
                        "terminal policy-walk abort rollback failed: {rollback_error:#}"
                    );
                }
                fail_speculative_step(a, "terminal policy verify commit", &e);
            } else {
                record_settled(a, k_verify, num_accepted, t.consumed_rows, true);
            }
            model.clear_ddtree_parent_ids();
            flush_deferred_emit(a);
            if let Some(record) =
                spec_cycle.terminal(spec_phase, num_accepted, a.output_tokens.len())
            {
                tracing::info!("{record}");
            }
            return;
        }
    } else {
        let mut consumed_rows = 0usize;
        let emit_tokens: Vec<u32> = if let Some((ref path, _)) = tree_accepted_path {
            path.iter()
                .filter_map(|&c| verify_input_tokens.get(c.saturating_sub(1)).copied())
                .collect()
        } else {
            let emit_take = num_accepted.min(verify_input_tokens.len());
            verify_input_tokens[..emit_take].to_vec()
        };
        for &tok in &emit_tokens {
            if emit_token_for_verify(a, tok, None) {
                consumed_rows += 1;
            }
            if a.finished {
                if let Err(e) = reconcile_terminal_verify_state(
                    model,
                    a,
                    pre_verify_len,
                    k_verify,
                    num_accepted,
                    consumed_rows,
                    &tokens,
                    tree_mode,
                    accepted_tree_path,
                ) {
                    tracing::error!("terminal verify-state commit (accepted draft): {e:#}");
                    a.seq.seq_len = pre_verify_len;
                    a.seq.tokens.truncate(pre_verify_len);
                    if let Err(rollback_error) =
                        rollback_failed_verify_frame(model, &mut a.seq, k_verify)
                    {
                        tracing::error!(
                            "terminal accepted-draft abort rollback failed: {rollback_error:#}"
                        );
                    }
                    fail_speculative_step(a, "terminal accepted-draft verify commit", &e);
                } else {
                    record_settled(a, k_verify, num_accepted, consumed_rows, true);
                }
                model.clear_ddtree_parent_ids();
                flush_deferred_emit(a);
                if let Some(record) =
                    spec_cycle.terminal(spec_phase, num_accepted, a.output_tokens.len())
                {
                    tracing::info!("{record}");
                }
                return;
            }
        }

        // Bonus token = verified[num_accepted] (the one that "corrected" the draft
        // at the first mismatch, or the next-prediction past the full-accept case).
        // Grammar verify mode substitutes the MASKED argmax at that position —
        // safe because the bonus has no KV/seq.tokens entry yet (it is fed as
        // verify input position 0 next step), exactly like the MTP masked path.
        //
        // FIX 1: for a tree-fork accept the bonus is the target's greedy at the
        // path TIP (the last accepted compact row), NOT verified[num_accepted]
        // (which is a contiguous-index assumption that is false once the path
        // forked). `tree_accepted_path.1` carries that compact row.
        let bonus_idx = match tree_accepted_path.as_ref() {
            Some((_, bonus_row)) => *bonus_row,
            None => num_accepted,
        };
        let bonus_tok = match grammar_accept {
            Some((_, b)) => Some(b),
            // V4 bonus-source fix: re-derive a <think>/</think>/EOS bonus into a
            // content token when thinking has already ended, so the spurious tag
            // never enters the verified/KV stream. No-op on content bonuses.
            None => verified
                .get(bonus_idx)
                .copied()
                .map(|b| reargmax_bonus_skip_think(model, a, bonus_idx, b)),
        };
        if let Some(bonus) = bonus_tok {
            if emit_token_for_verify(a, bonus, None) {
                consumed_rows += 1;
            }
            if a.finished {
                if let Err(e) = reconcile_terminal_verify_state(
                    model,
                    a,
                    pre_verify_len,
                    k_verify,
                    num_accepted,
                    consumed_rows,
                    &tokens,
                    tree_mode,
                    accepted_tree_path,
                ) {
                    tracing::error!("terminal verify-state commit (bonus): {e:#}");
                    a.seq.seq_len = pre_verify_len;
                    a.seq.tokens.truncate(pre_verify_len);
                    if let Err(rollback_error) =
                        rollback_failed_verify_frame(model, &mut a.seq, k_verify)
                    {
                        tracing::error!("terminal bonus abort rollback failed: {rollback_error:#}");
                    }
                    fail_speculative_step(a, "terminal bonus verify commit", &e);
                } else {
                    record_settled(a, k_verify, num_accepted, consumed_rows, true);
                }
                model.clear_ddtree_parent_ids();
                flush_deferred_emit(a);
                if let Some(record) =
                    spec_cycle.terminal(spec_phase, num_accepted, a.output_tokens.len())
                {
                    tracing::info!("{record}");
                }
                return;
            }
            a.last_token = bonus;
        }
    }

    // This shared verifier also handles native MTP. Numeric K describes the
    // actual target input rows, not a claim about the backend's kernel route.
    let metric_k = wide_verify_metric_k(
        model.proposer_arm_kind(model.proposer_arm()),
        draft_origin,
        k_verify,
    );
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            &metric_k,
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();
    crate::metrics::compat_metrics::record_spec_decode(drafts.len() as u64, num_accepted as u64);
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

    // SSM commit / rollback. Hybrid models (Qwen3.6-A3B has 30 GDN layers)
    // advance recurrent SSM state per-position during verify; without this
    // commit, the canonical h_state stays at position+γ even if only a few
    // drafts were accepted, producing gibberish on subsequent decodes.
    //
    // Semantics (default trait impl):
    //  - num_accepted == k_verify (full accept): canonical = h_state
    //  - 0 < num_accepted < k_verify (partial): canonical = intermediate[num_accepted-1]
    //  - num_accepted == 0: canonical untouched (rollback to checkpoint)
    //
    // k_verify = total verified positions = tokens.len() (bonus slot 0 + all
    // tree/draft nodes). For a WIDE DDTree (budget > γ) this is tree_len+1,
    // NOT drafts.len()+1 — the SSM rollback range must span every verified
    // node or the canonical state lands at the wrong intermediate slot.
    // Kernel slot of the LAST accepted state in `h_state_intermediates`:
    //   - Chain mode (no tree payload): `total_accepted - 1`. Slot N is the
    //     post-N-acceptances state; matches legacy arithmetic.
    //   - Tree mode: derived above from `accepted_compact_indices.last()`.
    //     For chain-shaped trees this == `total_accepted - 1`; for branch-
    //     diverged trees it's the max compact index (kernel slot directly).
    let spec_phase = spec_cycle.accept(spec_phase);
    let t_commit = Instant::now();
    // Diagnostic oracle: acceptance, bonus selection, and emission above used
    // the ordinary one-token argmax/logit rows collected before the batched
    // verifier. Discard the batched target state, restore the pre-verify
    // checkpoint, and rebuild exactly that serial-selected committed prefix.
    // This is deliberately slow and is only supported with SKIP_REPROPOSE:
    // sequential decodes overwrite DFlash's captured hidden rows, so they are
    // a correctness oracle rather than a production proposal handoff.
    let commit_res = if k5_commit_parity {
        (|| {
            // Preserve every verifier-owned recurrent/PLE source consumed by
            // the root-only production commit. The following ordinary K1
            // decode is an oracle only; it must not replace these sources.
            let commit_sources =
                model.k16_commit_parity_capture_sources(&a.seq, k_verify, last_inter_slot)?;

            // Remove every speculative QSA row before the ordinary K1 oracle
            // rebuilds the accepted prefix. The replay then leaves QSA in the
            // exact committed-prefix state while recurrent/PLE verifier
            // sources remain available for the production commit below.
            model.commit_qwen4_qsa_for_sequence(&a.seq, 0, k_verify)?;
            a.seq.seq_len = pre_verify_len;
            a.seq.tokens.truncate(pre_verify_len);
            model.pre_verify_copy_async(&mut a.seq)?;
            for &token in tokens.iter().take(total_accepted) {
                model.decode(token, &mut a.seq, 0)?;
            }
            let expected = model.k16_commit_parity_capture_live(&a.seq)?;

            // Re-seed live recurrent/PLE state from the untouched pre-verify
            // checkpoint, then restore the actual K5 verifier sources. The
            // production commit below remains the only writer of canonical
            // live/checkpoint destinations.
            model.pre_verify_copy_async(&mut a.seq)?;
            model.k16_commit_parity_restore_sources(&a.seq, &commit_sources)?;
            model.k16_commit_parity_drain_default()?;
            model.commit_verify_state_async(&mut a.seq, total_accepted, k_verify)?;
            model.k16_commit_parity_compare_committed(
                &a.seq,
                &expected,
                pre_verify_len,
                &tokens,
                total_accepted,
                k_verify,
                last_inter_slot,
            )
        })()
    } else if let Some(ref oracle) = serial_oracle {
        let batched = batched_verified_for_oracle
            .as_deref()
            .expect("serial oracle always preserves batched results");
        for (row, &serial_argmax) in oracle.verified.iter().enumerate() {
            let batch_argmax = batched.get(row).copied().unwrap_or(u32::MAX);
            tracing::info!(
                "DFLASH_SERIAL_ORACLE pre_verify_len={} row={} input={} batch_argmax={} serial_argmax={} match={}",
                pre_verify_len,
                row,
                tokens.get(row).copied().unwrap_or(u32::MAX),
                batch_argmax,
                serial_argmax,
                batch_argmax == serial_argmax,
            );
        }
        if oracle.k16_commit_parity {
            (|| {
                // Full accept is the one real-commit branch that consumes the
                // verifier's live H/conv state instead of an intermediate.
                // Preserve that exact live image before the serial replay.
                let wide_live = if total_accepted == k_verify {
                    Some(model.k16_commit_parity_capture_live(&a.seq)?)
                } else {
                    None
                };

                // Build exactly the serial K1 state selected by acceptance,
                // capture it once, then restore the real commit's required
                // pre-state (partial) or verifier live state (full).
                a.seq.seq_len = pre_verify_len;
                a.seq.tokens.truncate(pre_verify_len);
                model.pre_verify_copy_async(&mut a.seq)?;
                for &token in tokens.iter().take(total_accepted) {
                    model.decode(token, &mut a.seq, 0)?;
                }
                let expected = model.k16_commit_parity_capture_live(&a.seq)?;
                model.pre_verify_copy_async(&mut a.seq)?;
                if let Some(ref wide) = wide_live {
                    model.k16_commit_parity_restore_live(&a.seq, wide)?;
                }
                model.k16_commit_parity_drain_default()?;

                // Execute the production flat-chain K16 commit itself.  The
                // comparison hook waits for its owned event before reading
                // any live/checkpoint bytes.
                model.commit_verify_state_async(&mut a.seq, total_accepted, k_verify)?;
                model.k16_commit_parity_compare_committed(
                    &a.seq,
                    &expected,
                    pre_verify_len,
                    &tokens,
                    total_accepted,
                    k_verify,
                    last_inter_slot,
                )
            })()
        } else {
            let mut result = model.commit_qwen4_qsa_for_sequence(&a.seq, 0, k_verify);
            if result.is_ok() {
                a.seq.seq_len = pre_verify_len;
                a.seq.tokens.truncate(pre_verify_len);
                result = model.pre_verify_copy_async(&mut a.seq);
            }
            if result.is_ok() {
                for &token in tokens.iter().take(total_accepted) {
                    match model.decode(token, &mut a.seq, 0) {
                        Ok(_) => {}
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
            }
            if result.is_ok() {
                result = model.finish_verify_rebuild(&mut a.seq, total_accepted, k_verify);
            }
            if result.is_ok() {
                tracing::info!(
                    "DFLASH_SERIAL_COMMIT rebuilt={} seq_len={} pre_verify_len={}",
                    total_accepted,
                    a.seq.seq_len,
                    pre_verify_len,
                );
            }
            result
        }
    } else if tree_last_inter_slot.is_some() {
        model.commit_verify_state_async_with_slot(
            &mut a.seq,
            total_accepted,
            k_verify,
            last_inter_slot,
        )
    } else {
        model.commit_verify_state_async(&mut a.seq, total_accepted, k_verify)
    };
    let t_commit_us = t_commit.elapsed().as_micros();
    let spec_phase = spec_cycle.commit_enqueue(spec_phase);
    if let Err(e) = commit_res {
        tracing::error!("commit_verify_state_async (dflash): {e:#}");
        a.seq.seq_len = pre_verify_len;
        a.seq.tokens.truncate(pre_verify_len);
        if let Err(rollback_error) = rollback_failed_verify_frame(model, &mut a.seq, k_verify) {
            tracing::error!("commit-error abort rollback failed: {rollback_error:#}");
        }
        model.clear_ddtree_parent_ids();
        flush_deferred_emit(a);
        fail_speculative_step(a, "verify commit", &e);
        return;
    }
    if let Some(fixture) = acceptance_fixture.as_ref() {
        let Some(evidence) = fixture_evidence.take() else {
            tracing::error!("K16 acceptance fixture lacks its armed evidence guard");
            model.clear_ddtree_parent_ids();
            flush_deferred_emit(a);
            a.finished = true;
            return;
        };
        if let Err(error) = fixture.finish(evidence, num_accepted, pre_verify_len, &tokens) {
            tracing::error!("K16 acceptance fixture commit receipt failed: {error:#}");
            model.clear_ddtree_parent_ids();
            flush_deferred_emit(a);
            a.finished = true;
            return;
        }
    }
    // M8A: now safe to clear — commit has finished reading the tree-mode flag.
    model.clear_ddtree_parent_ids();
    if k5_commit_parity {
        // The K1 oracle intentionally overwrites proposer/capture scratch that
        // is outside this recurrent/PLE receipt. Stop after the production
        // commit instead of letting that diagnostic scratch seed reproposal.
        tracing::info!(
            "DFLASH_K5_COMMIT_PARITY_TERMINAL schema=1 production_commit_preserved=true reproposal_skipped=true"
        );
        flush_deferred_emit(a);
        a.finished = true;
        return;
    }

    // FIX 2 — KV compaction for a sparse (tree-fork) accept. When the accepted
    // path crossed a sibling fork, its attention KV sits at the scattered
    // compact slots `pre_verify_len + compact_idx`; gather it down to the
    // contiguous run `pre_verify_len + 1 .. pre_verify_len + num_accepted` so
    // the next decode/propose reads correct contiguous KV (mirrors the flat
    // chain layout). No-op when the path is already contiguous (every
    // flat-chain accept) or tree-commit is off. LOSSLESS: relocates committed
    // K/V bytes only. Must run AFTER the SSM commit (which reads its own
    // intermediate pool, untouched by KV) and BEFORE the re-propose decode.
    // Guard on NON-contiguity explicitly (mirrors set_dflash_accepted_compact
    // below): a contiguous accepted path `[1,2,..,n]` needs no compaction —
    // its KV already sits at the contiguous run, exactly like the flat path,
    // which never calls compact_verify_kv at all. Only a fork-crossing
    // (sparse) path requires the gather. Calling the gather on a contiguous
    // path must be a true identity; making the skip explicit here removes any
    // dependence on the kernel's internal contiguity detection being exact.
    if let Some((ref path, _)) = tree_accepted_path
        && !path.is_empty()
        && !path.iter().enumerate().all(|(i, &c)| c == i + 1)
        && let Err(e) = model.compact_verify_kv(&a.seq, path, pre_verify_len)
    {
        tracing::error!("compact_verify_kv (dflash tree-fork): {e:#}");
        flush_deferred_emit(a);
        fail_speculative_step(a, "verify tree KV compaction", &e);
        return;
    }

    if deferred_think_end_snapshot && !a.finished {
        // `</think>` was emitted during the thinking accept walk before the
        // recurrent-state commit. Order the default stream after the
        // secondary-stream accepted-intermediate restore so the rollback
        // snapshot captures the committed prefix, not the rejected tail.
        if let Err(e) = model.sync_secondary() {
            tracing::error!("sync_secondary before SSM boundary snapshot (dflash): {e:#}");
            flush_deferred_emit(a);
            fail_speculative_step(a, "verify think-end snapshot sync", &e);
            return;
        }
        rollback::snapshot_boundary_if_ssm(a, model);
    }

    // Save the bonus token's hidden state for the NEXT propose() call.
    //
    // This verifier is selected by draft width (`drafts.len() >= 4`), not by
    // proposer type.  A native MTP proposer configured with four or more
    // drafts therefore lands here too.  DFlash consumes its layer-capture
    // buffer, whereas native MTP consumes `mtp_hidden_save`; updating only the
    // former left native MTP conditioned on a stale hidden row after its first
    // wide verify.  The resulting proposals rapidly collapsed to reserved
    // chat-control tokens with ~0 acceptance.  Preserve the DFlash path and
    // save the verified bonus row explicitly for native MTP.
    let t_save = Instant::now();
    // `a.last_token` already holds the emitted bonus (masked argmax under
    // grammar verify mode, else `verified[bonus_idx]`; unchanged when no
    // bonus row existed).
    let bonus = a.last_token;
    let proposer_is_dflash = model.proposer_is_dflash();
    let tree_bonus_row = tree_accepted_path.as_ref().map(|(_, row)| *row);
    let save_hidden_result =
        match wide_verify_hidden_save(proposer_is_dflash, bonus, num_accepted, tree_bonus_row) {
            WideVerifyHiddenSave::DflashToken(token) => {
                model.save_hidden_for_dflash(token, &mut a.seq, 0)
            }
            WideVerifyHiddenSave::NativeMtpRow(row) => {
                model.save_hidden_for_mtp_sequence(row, &a.seq, 0)
            }
        };
    let dflash_hidden_saved = save_hidden_result.is_ok();
    if let Err(e) = save_hidden_result {
        tracing::error!("save hidden after wide verify: {e:#}");
        // Native MTP consumes this buffer immediately. Its established K=2/3/4
        // paths abort the step on a failed save so the next scheduler tick can
        // bootstrap from a clean one-token decode. Continuing here would feed
        // the proposer a known-stale hidden row—the exact corruption this path
        // exists to prevent. Preserve DFlash's prior log-and-continue behavior.
        if !proposer_is_dflash {
            flush_deferred_emit(a);
            return;
        }
    }
    let t_save_us = t_save.elapsed().as_micros();

    let spec_phase = spec_cycle.post_commit_enqueue(spec_phase);
    let t_trim = Instant::now();
    let dflash_proposer_trimmed = match model.trim_proposer_state(&mut a.seq, num_accepted, 0) {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("trim_proposer_state: {e:#}");
            false
        }
    };
    if dflash_proposer_trimmed {
        // Settled: state commit, hidden save and proposer trim succeeded.
        let consumed = a.output_tokens.len().saturating_sub(emitted_before);
        record_settled(a, k_verify, num_accepted, consumed, false);
    }
    // FIX 1: stamp the sparse accepted path onto the proposer (AFTER
    // trim/after_verify, which clears it) so the next propose's ctx-hidden
    // append reads the scattered fork-capture rows, not the contiguous
    // prefix. Only when the path actually forked (non-contiguous): a
    // contiguous `[1..N]` path needs no override and stays on the fast path.
    if let Some((ref path, _)) = tree_accepted_path {
        let is_contiguous = path.iter().enumerate().all(|(i, &c)| c == i + 1);
        if !is_contiguous {
            model.set_dflash_accepted_compact(&mut a.seq, path);
        }
    }
    let t_trim_us = t_trim.elapsed().as_micros();

    // ATLAS_DFLASH_RECYCLE=1: stash the discarded draft tail BEFORE the
    // re-propose below, keyed by the corrected token the target just committed
    // (= a.last_token, which is also the `token` fed to the next propose). The
    // next propose re-offers `drafts[num_accepted+1..]` when its last_token
    // matches this key. No-op for the default (non-recycle) path. Skipped under
    // tree mode (drafts/verified are compact-tree indices, not a flat chain, so
    // the tail-after-mismatch arithmetic does not hold). Lossless: only changes
    // what is PROPOSED next step, never what is committed.
    if dflash_recycle_enabled()
        && a.pending_tree_payload.is_none()
        && let Err(e) = model.dflash_stash_recycle(&mut a.seq, drafts, num_accepted, a.last_token)
    {
        tracing::warn!("dflash_stash_recycle: {e:#}");
    }

    // ── ATLAS_DFLASH_ECHO=1: stash the target's own rejected-tail argmaxes ──
    // `verified[num_accepted+1 ..]` (the tokens AFTER the bonus) are the
    // target's OWN next-token choices conditioned on a near-miss prefix —
    // usually still right after the one-token bonus substitution. Stash them
    // (keyed by the bonus = a.last_token) so the NEXT propose offers them as
    // a target-authored draft chain and skips the drafter forward (the
    // 25-50ms propose slice). Gated to the plain FLAT chain path only:
    //   - no tree payload this step and no tree/portfolio draft method active
    //     (`verified` rows are compact-tree indices there, and the accept
    //     walk is topology-aware — chain arithmetic does not hold);
    //   - not the think-spec accept walk (it commits its own tokens with
    //     forced-injection semantics; its bonus may not be verified[n]);
    //   - not the grammar-masked accept (its bonus is a MASKED argmax, and
    //     grammar-active sequences drop pending drafts in step_mtp anyway).
    // The min-accept floor / min-tail gates live in dflash_stash_echo.
    // LOSSLESS: only changes what is PROPOSED next step, never what is
    // committed — same oracle contract as recycle above.
    let echo_state_ready = echo_state_ready(dflash_hidden_saved, dflash_proposer_trimmed);
    let echo_route_eligible = a.pending_tree_payload.is_none()
        && !dflash_tree_method_active()
        && !dflash_portfolio_active()
        && thinking_accept.is_none()
        && content_accept.is_none()
        && grammar_accept.is_none();
    let echo_result = dispatch_echo_stash(
        spark_model::layers::dflash_head::echo::EchoConfig::enabled(),
        echo_state_ready,
        echo_route_eligible,
        &verified,
        num_accepted,
        |tokens, accepted| model.dflash_stash_echo(&mut a.seq, tokens, accepted, a.last_token),
    );
    let echo_dispatch_failed = echo_result.is_err();
    if let Err(e) = echo_result {
        tracing::warn!("dflash_stash_echo failed; suppressing immediate reproposal: {e:#}");
    }

    // Re-propose for next step — unless the stage-1 grammar gate fires
    // (grammar now constrains output, e.g. this verify emitted the token
    // that opened a tool-call body): leave `pending_drafts` empty so the
    // next step runs the grammar-enforced bootstrap decode.
    let skip_propose =
        suppress_reproposal(dflash_grammar_skip_propose(model, a), echo_dispatch_failed);
    let _mtp_grammar_mask = if skip_propose {
        None
    } else {
        mtp_grammar_mask_for(a)
    };
    let spec_phase = spec_cycle.proposer_state(spec_phase);
    let t_propose = std::time::Instant::now();
    // Diagnostic isolation: leave drafts empty after each verify so the next
    // scheduler tick runs one ordinary bootstrap decode against the just-
    // committed target state.  This distinguishes a bad partial-state restore
    // from a K=gamma verify-logit mismatch without unloading the proposer.
    let skip_repropose_diag =
        std::env::var("ATLAS_DFLASH_SKIP_REPROPOSE").ok().as_deref() == Some("1");
    let proposal_token = a.last_token;
    let num_drafts = super::verify_qwen4_batched_step::dflash_gumbel_num_drafts(
        model,
        a.temperature.is_finite() && a.temperature > 0.0,
        1,
        num_drafts,
    );
    // ── Retrieval pre-emption for the NEXT frame ──
    //
    // The same cascade as the Phase A bootstrap (`mtp_step`), and the
    // reason it is repeated here: in steady-state decoding a sequence
    // almost never returns to the bootstrap, so a tier wired only there
    // would measure as inert no matter how well it does offline.
    //
    // It rides the existing proposal seam rather than installing
    // separately. `should_propose = false` skips the drafter's forward
    // pass entirely — the win, on top of the accepted tokens — and the
    // `after_propose` hook substitutes the retrieved chain into the same
    // result the model would have returned, so `install_collected` still
    // publishes exactly one frame and drains the producer tree exactly
    // once. Phase accounting is untouched.
    let retrieved = if skip_propose || skip_repropose_diag {
        None
    } else {
        super::think_budget_nudge::full_forced_draft_chain(a, num_drafts)
            .map(|chain| (DraftOrigin::Proposer, chain))
            .or_else(|| retrieval_chain(a, proposal_token, num_drafts))
    };
    let should_propose = !skip_propose && !skip_repropose_diag && retrieved.is_none();
    let used_retrieval = retrieved.is_some();
    // Retire the just-committed verify capture rows into the sequence's
    // DFlash ring whenever a forced-close nudge, retrieval chain, or
    // non-diagnostic suppressed proposal skips `propose()`, so subsequent
    // verify or sibling-sequence frames never overwrite unretired captures and
    // leave `ctx_len` behind `seq.seq_len`. If reconciliation fails, withhold
    // any external draft chain rather than verifying over a broken cursor.
    let skip_propose_reconciled =
        if used_retrieval || (skip_propose && !skip_repropose_diag && dflash_proposer_trimmed) {
            if proposer_is_dflash && !dflash_proposer_trimmed {
                false
            } else {
                super::mtp_step::reconcile_dflash_before_external_draft(model, a)
            }
        } else {
            true
        };
    if should_propose {
        super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
    } else {
        super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
    }
    let (_, spec_phase) = proposal_lifecycle::propose_and_install_with(
        model,
        a,
        proposal_token,
        num_drafts,
        _mtp_grammar_mask.as_deref(),
        should_propose,
        |a, propose_result| {
            if skip_propose_reconciled && let Some((origin, chain)) = retrieved {
                *propose_result = Ok(chain);
                a.draft_origin = origin;
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                if let Some(prop_state) = a.seq.proposer_state.as_mut()
                    && let Some(mtp_state) = prop_state
                        .as_any_mut()
                        .downcast_mut::<spark_model::layers::qwen4_mtp::Qwen4MtpState>(
                    )
                {
                    mtp_state.notify_external_draft();
                }
            }
            let spec_phase = spec_cycle.propose_complete(spec_phase);
            let propose_us = t_propose.elapsed().as_micros();
            if dflash_propose_log_enabled() {
                tracing::info!(
                    "DFlash propose: {}μs num_accepted={} seq_len={}",
                    propose_us,
                    num_accepted,
                    a.seq.seq_len
                );
            }
            match propose_result {
                Ok(drafts) if !drafts.is_empty() => {
                    // ── ATLAS_DFLASH_CFG_JF=1: CFG jump-forward splice (default off) ──
                    //
                    // Splice structurally-forced tokens (closing brackets/quotes) into
                    // the freshly-proposed draft chain at positions where the neural
                    // drafter disagreed with the single legal next token. LOSSLESS: the
                    // verify above is a greedy oracle, so a wrong splice is rejected
                    // exactly like a wrong drafter token. Skipped when a
                    // NON-terminated grammar is active (xgrammar already forces those
                    // positions; we must not fight it) and when the tree payload is set
                    // (drafts are compact tree indices, not a flat chain). No-op unless
                    // the classification table was built at startup.
                    // Truncate any GPU rejection `q` fence at the first mutated draft index.
                    let before_splice = drafts.clone();
                    cfg_jf_splice_drafts(a, drafts);
                    super::think_budget_nudge::splice_forced_drafts(a, drafts);
                    super::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
                        a,
                        &before_splice,
                        drafts,
                    );
                }
                Ok(_) => {}
                Err(e) => tracing::error!("run_mtp_propose_multi (dflash): {e:#}"),
            }

            if step_timing {
                let total_us = t_step.elapsed().as_micros();
                let other_us = total_us.saturating_sub(
                    t_sync_secondary_us
                        + t_verify_us
                        + t_commit_us
                        + t_save_us
                        + t_trim_us
                        + propose_us,
                );
                tracing::info!(
                    "DFLASH step timing: total={total_us}μs sync_secondary={t_sync_secondary_us}μs \
                     verify={t_verify_us}μs commit={t_commit_us}μs save_hidden={t_save_us}μs \
                     trim={t_trim_us}μs propose={propose_us}μs other={other_us}μs accepted={num_accepted}",
                );
            }

            // ATLAS_FULL_PROFILE: emit per-kernel KPROF every 150 verify steps — the
            // DFlash scheduler bypasses the lifecycle.rs dump so wire it here.
            if spark_model::full_profile::is_enabled() {
                static DFKP_CTR: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                if DFKP_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 150 == 149 {
                    spark_model::full_profile::dump();
                }
            }

            spec_phase
        },
    );
    if used_retrieval {
        // A retrieved chain is flat by construction. `install` publishes
        // whatever the producer's tree slot held, and no propose ran to
        // refill it, so drop any topology that predates this frame rather
        // than letting it describe token positions it was not built for.
        a.pending_tree_payload = None;
    } else if should_propose && dflash_pcum_enabled_for_seq(a) {
        super::verify_qwen4_batched_step::apply_dflash_pcum_truncation(
            a,
            model.proposer_is_dflash(),
        );
    }
    // Pre-warm ParkedPleCache on the background PLE worker for this sequence's
    // upcoming verify window `[a.last_token, ...a.pending_drafts]` paired with
    // `a.seq.tokens` before the next verify step's Layer 0 begins.
    super::verify_qwen4_batched_step::warm_ple_draft_ahead_for_seq(model, a);
    // ATLAS_DFLASH_DEFER_EMIT=1: dispatch coalesced SSE TokenBatch events NOW,
    // strictly after GPU commit + async propose have already been enqueued.
    flush_deferred_emit(a);
    let spec_phase = spec_cycle.finalize(spec_phase);
    if let Some(record) = spec_cycle.complete(spec_phase, num_accepted, a.output_tokens.len()) {
        tracing::info!("{record}");
    }
}

/// Whether DFlash2 `p_cum` confidence truncation should run for `a` inside a
/// cohort of `cohort_size` sequences.
///
/// Disabled by default on dense (non-MoE) targets so full `K` acceptance and
/// fixed-width verify CUDA graphs are preserved; opt-in `ATLAS_DFLASH_DENSE_PCUM=1`
/// enables dense 27B truncation at `cohort_size >= 4` or inside `<think>` blocks.
/// MoE targets and explicit `ATLAS_QWEN4_DFLASH_PCUM_THRESH` overrides preserve
/// their established behavior.
pub(super) fn dflash_pcum_enabled_for_seq_in_cohort(a: &ActiveSeq, cohort_size: usize) -> bool {
    let target_is_moe = a
        .seq
        .proposer_state
        .as_ref()
        .and_then(|s| {
            s.as_any()
                .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
        })
        .and_then(spark_model::layers::dflash_head::DflashProposerState::target_is_moe);
    spark_model::layers::dflash_head::dflash_pcum_enabled_for_target_context_env(
        target_is_moe,
        cohort_size,
        a.inside_thinking,
    )
}

/// Whether DFlash2 `p_cum` confidence truncation should run for single-sequence `a`.
pub(super) fn dflash_pcum_enabled_for_seq(a: &ActiveSeq) -> bool {
    dflash_pcum_enabled_for_seq_in_cohort(a, 1)
}

/// Whether any DFlash TREE draft method is active (BRANCH / CATERPILLAR /
/// DDTree). CFG jump-forward operates on a FLAT draft chain only; when a tree
/// method is on, the proposer emits compact tree-index tokens whose positions
/// don't correspond to a left-to-right token stream, so splicing would be
/// meaningless. Read once. CFG_JF is a distinct opt-in flag, so the operator is
/// not expected to combine them — this is a defensive guard.
pub(super) fn dflash_tree_method_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BRANCH").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_CATERPILLAR").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_METHOD").ok().as_deref() == Some("ddtree")
            || std::env::var("ATLAS_DFLASH_FREE_SLOTS")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .is_some_and(|n| n >= 1)
    })
}

/// Whether the 2-root PORTFOLIO forest verify is enabled
/// (ATLAS_DFLASH_PORTFOLIO=1). Not covered by `dflash_tree_method_active`
/// (portfolio emits a flat chain on steps where the retrieval sibling
/// doesn't fire), but echo-drafting is excluded whenever the mode is on:
/// its verify rows may belong to a forest topology. Read once.
pub(super) fn dflash_portfolio_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_PORTFOLIO").ok().as_deref() == Some("1"))
}

/// Apply the CFG jump-forward splice to a freshly-proposed flat draft chain
/// in place. No-op unless `ATLAS_DFLASH_CFG_JF=1`, the startup classification
/// table is present, no tree method is active, and the sequence has no active
/// (non-terminated) grammar. Emits a one-shot stats log on the first splice.
///
/// LOSSLESS: only mutates the PROPOSED tokens; the verify path commits solely
/// the target's greedy token, so a wrong splice is rejected for free and output
/// is byte-identical to the flag-off path when no splice ever helps.
pub(super) fn cfg_jf_splice_drafts(a: &ActiveSeq, drafts: &mut [u32]) {
    use super::cfg_jump_forward as jf;

    if !jf::cfg_jf_enabled() || drafts.is_empty() || dflash_tree_method_active() {
        return;
    }
    // Do not fight an active grammar (xgrammar already forces those slots).
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return;
    }
    let (Some(table), Some(forced)) = (jf::delim_table(), jf::forced_ids()) else {
        return; // table not built (flag was off at startup) → inert
    };

    let stats = jf::splice_forced(drafts, &a.seq.tokens, &table, &forced);

    if stats.splices > 0 {
        static JF_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = JF_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 16 {
            tracing::info!(
                "DFlash CFG_JF #{n}: spliced {} forced token(s) into draft chain \
                 (first_pos={}, seq_len={})",
                stats.splices,
                stats.first_pos,
                a.seq.seq_len,
            );
        }
    }
}

/// Grammar-masked re-derivation of the DFlash accept prefix + bonus
/// (stage 2, `ATLAS_DFLASH_GRAMMAR_MODE=verify`).
///
/// Row `i` of the model's verify logits buffer (`[K, vocab]` BF16, row
/// stride `vocab_size`) is the target's prediction for the slot
/// `drafts[i]` occupies; row `drafts.len()` is the bonus slot. Walk the
/// positions in order: fill the matcher's bitmask for the CURRENT grammar
/// state, take the MASKED argmax as the target token, and stop acceptance
/// at the first draft that diverges from it or that the matcher rejects.
/// The matcher is advanced per accepted draft (so each position's mask
/// reflects its prefix) and rolled back before returning — `emit_token`
/// re-advances it for real on emission, mirroring
/// `truncate_drafts_at_grammar_boundary`'s transient-advance pattern.
///
/// Mask-application convention mirrors the MTP masked-draft path in
/// `spark-model/src/layers/mtp_head/forward.rs`: bit `tok` set in the i32
/// bitmask ⇒ token allowed; the pick is the shared first-wins greedy rule
/// (`spark_runtime::sampler::argmax_bf16_first_wins`).
///
/// Returns `(num_accepted, bonus_token)`, or `None` when masking does not
/// apply (thinking span, no/terminated grammar, fp32 logits) or the
/// logits D2H fails — the caller falls back to the unmasked accept path.
fn dflash_masked_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<(usize, u32)> {
    if a.inside_thinking || verified.is_empty() {
        return None;
    }
    if a.grammar_state.as_ref().is_none_or(|gs| gs.is_terminated()) {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // masked argmax below assumes BF16 rows
    }
    let vocab = model.vocab_size();
    let mut row_buf = vec![0u8; vocab * 2];

    let gs = a.grammar_state.as_mut().expect("checked above");
    let mut accepted = 0usize;
    let mut bonus: Option<u32> = None;
    for i in 0..verified.len() {
        let target_tok = if gs.fill_bitmask() {
            if let Err(e) =
                model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf)
            {
                tracing::warn!(
                    "DFlash grammar verify: logits D2H failed ({e:#}); unmasked fallback"
                );
                if accepted > 0 {
                    gs.rollback(accepted);
                }
                return None;
            }
            match masked_argmax_bf16(&row_buf, gs.bitmask_data(), vocab) {
                Some(t) => t,
                None => {
                    // Degenerate empty allowed set (dead grammar state):
                    // keep the unmasked argmax — emit_token tolerates the
                    // failed accept exactly as the pre-fix path did.
                    tracing::warn!(
                        "DFlash grammar verify: mask allowed zero tokens at verify pos {i}"
                    );
                    verified[i]
                }
            }
        } else {
            // No constraint at this position (e.g. grammar just terminated
            // mid-walk after a stop token was accepted).
            verified[i]
        };
        // Accept drafts[i] only while a bonus row remains past it (mirrors
        // the unmasked loop's `i + 1 >= verified.len()` guard); otherwise
        // this row's target token becomes the bonus.
        if i < drafts.len()
            && i + 1 < verified.len()
            && drafts[i] == target_tok
            && gs.accept_token(drafts[i])
        {
            accepted += 1;
            continue;
        }
        bonus = Some(target_tok);
        break;
    }
    if accepted > 0 {
        gs.rollback(accepted);
    }
    Some((accepted, bonus?))
}

/// Typical acceptance for the flat-chain DFlash accept prefix
/// (`ATLAS_DFLASH_TYPICAL_ACCEPT=<epsilon>`, opt-in).
///
/// At temperature > 0 the exact-match rule (`drafts[i] == verified[i]`,
/// where `verified` is the target's UNMASKED GPU argmax) is needlessly
/// strict: a draft the target itself would plausibly sample gets rejected
/// just for not being the argmax, collapsing acceptance on creative/story
/// prompts. Instead, accept `drafts[i]` when
///
///   p_target(drafts[i]) >= max(epsilon, alpha * p_max)
///
/// where `p_target` is the temperature-scaled softmax of the target's
/// verify logits at position `i` (row `i` of the `[K, vocab]` BF16 logits
/// buffer, still resident on device — same lazy per-row D2H pattern as
/// `dflash_masked_accept`), `p_max` its max, `alpha` from
/// `ATLAS_DFLASH_TYPICAL_ALPHA` (default 0.3). Exact argmax matches are
/// accepted without the test (and without the D2H) — the typical rule
/// only ever WIDENS acceptance, so greedy-equivalent behavior is the
/// floor. The first position failing the test ends the prefix; the bonus
/// stays `verified[num_accepted]` (the existing path — note this verify
/// path emits the target's argmax as the bonus even at temp > 0).
///
/// Returns `Some(num_accepted)` when the rule is active for this request,
/// `None` to fall back to exact-match: env absent, temperature == 0
/// (greedy completely unaffected), an active (non-terminated) grammar —
/// loosened acceptance must not bypass constraint enforcement — or fp32
/// logits (row layout below assumes BF16). A failed row D2H ends the
/// prefix at that position, which equals the exact-match outcome there.
fn dflash_typical_accept(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<usize> {
    let epsilon = dflash_typical_epsilon()?;
    if a.temperature <= 0.0 {
        return None;
    }
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // softmax below assumes BF16 rows
    }
    let vocab = model.vocab_size();
    let alpha = dflash_typical_alpha();
    // Lazily allocated on the first non-argmax position; pure argmax
    // prefixes never touch the device.
    let mut row_buf: Vec<u8> = Vec::new();
    let mut accepted = 0usize;
    let mut typical_hits = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            accepted += 1;
            continue;
        }
        if drafts[i] as usize >= vocab {
            break;
        }
        if row_buf.is_empty() {
            row_buf = vec![0u8; vocab * 2];
        }
        if let Err(e) = model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf) {
            tracing::warn!("DFlash typical accept: logits D2H failed ({e:#}); stopping prefix");
            break;
        }
        let (p_draft, p_max) = typical_row_probs_bf16(&row_buf, vocab, a.temperature, drafts[i]);
        if p_draft >= epsilon.max(alpha * p_max) {
            accepted += 1;
            typical_hits += 1;
        } else {
            break;
        }
    }
    if typical_hits > 0 {
        tracing::debug!(
            "DFlash typical accept: +{typical_hits} non-argmax draft(s) accepted \
             (total {accepted}/{}, ε={epsilon}, α={alpha}, T={})",
            drafts.len(),
            a.temperature,
        );
    }
    Some(accepted)
}

/// Temperature-scaled softmax probabilities over one host-copied BF16
/// logits row: `(p(tok), p_max)`. Two passes, f32 accumulation; numerically
/// stable via max-logit subtraction (so `p_max == 1/Z` exactly).
fn typical_row_probs_bf16(bytes: &[u8], vocab: usize, temperature: f32, tok: u32) -> (f32, f32) {
    let inv_t = 1.0 / temperature.max(1e-6);
    let mut max_logit = f32::NEG_INFINITY;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        if l > max_logit {
            max_logit = l;
        }
    }
    let mut z = 0f32;
    let mut e_tok = 0f32;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        let e = ((l - max_logit) * inv_t).exp();
        z += e;
        if i == tok as usize {
            e_tok = e;
        }
    }
    if z <= 0.0 || !z.is_finite() {
        return (0.0, 0.0);
    }
    (e_tok / z, 1.0 / z)
}

/// `ATLAS_DFLASH_TYPICAL_ACCEPT` epsilon, read once at first use. `None`
/// when the env is absent (feature off); a set-but-unparseable value
/// falls back to 0.05 with a warning.
fn dflash_typical_epsilon() -> Option<f32> {
    static EPS: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *EPS.get_or_init(|| {
        let raw = std::env::var("ATLAS_DFLASH_TYPICAL_ACCEPT").ok()?;
        match raw.trim().parse::<f32>() {
            Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => Some(v),
            _ => {
                tracing::warn!(
                    "ATLAS_DFLASH_TYPICAL_ACCEPT={raw:?} unparseable (want float in [0,1]); \
                     defaulting to 0.05"
                );
                Some(0.05)
            }
        }
    })
}

/// `ATLAS_DFLASH_TYPICAL_ALPHA` (fraction of `p_max` a draft must reach),
/// read once at first use. Default 0.3.
fn dflash_typical_alpha() -> f32 {
    static ALPHA: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *ALPHA.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_TYPICAL_ALPHA")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.3)
    })
}

/// Parsed `ATLAS_DFLASH_RELAX_*` configuration (read once).
#[derive(Clone, Copy)]
struct RelaxConfig {
    /// Accept a draft if it is within the target's top-`k` logits. 0 = off.
    topk: usize,
    /// Accept a draft if `p_target(draft)/p_target(argmax) >= ratio`.
    /// `<= 0.0` = off. The ratio is logit-based so it is temperature-free:
    /// `p(d)/p(amax) = exp(l_d - l_amax)` (the partition function cancels),
    /// which is exactly the relaxation we want at temp0 (no softmax temp).
    ratio: f32,
}

/// `ATLAS_DFLASH_RELAX_ACCEPT` gate config. Returns `None` (feature off)
/// unless `ATLAS_DFLASH_RELAX_ACCEPT=1` AND at least one of
/// `ATLAS_DFLASH_RELAX_TOPK` (int >= 1) / `ATLAS_DFLASH_RELAX_RATIO`
/// (float in (0,1]) is set. Read once at first use.
fn dflash_relax_config() -> Option<RelaxConfig> {
    static CFG: std::sync::OnceLock<Option<RelaxConfig>> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_RELAX_ACCEPT").ok().as_deref() == Some("1");
        if !on {
            return None;
        }
        let topk = std::env::var("ATLAS_DFLASH_RELAX_TOPK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(0);
        let ratio = std::env::var("ATLAS_DFLASH_RELAX_RATIO")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
            .unwrap_or(0.0);
        if topk == 0 && ratio <= 0.0 {
            tracing::warn!(
                "ATLAS_DFLASH_RELAX_ACCEPT=1 but neither RELAX_TOPK (>=1) nor \
                 RELAX_RATIO (0<r<=1) is set — relaxed accept inactive (falling \
                 back to exact / typical accept)"
            );
            return None;
        }
        tracing::info!(
            "ATLAS_DFLASH_RELAX_ACCEPT active: topk={topk} ratio={ratio} \
             (commits high-prob target near-miss drafts; PPL-bounded, default-off)"
        );
        Some(RelaxConfig { topk, ratio })
    })
}

/// PPL-bounded relaxed acceptance for the flat-chain DFlash accept prefix
/// (`ATLAS_DFLASH_RELAX_ACCEPT=1`, default off; works at temp0 and temp>0).
///
/// At each would-be-mismatch position the strict greedy rule rejects the
/// draft purely because it is not the target's argmax — even when the draft
/// is the target's 2nd/3rd choice with near-argmax probability. On novel
/// code the drafter's near-misses are usually exactly such high-probability
/// alternates, so strict greedy collapses acceptance to ~3/16.
///
/// Relaxed accept commits the DRAFT token (NOT the argmax) at a mismatch
/// when the draft is a high-probability token under the TARGET:
///
///   - draft is within the target's top-`topk` logits at this row, OR
///   - `p_target(draft) / p_target(argmax) >= ratio`, i.e.
///     `l_draft - l_argmax >= ln(ratio)` (logit-space, temperature-free).
///
/// Quality is preserved because every committed token is one the target
/// itself ranks at/near the top — the PPL guardrail bounds the drift. The
/// committed draft's KV + SSM intermediate already sit at compact slot `i`
/// (the drafter token was embedded there in `decode_verify`), so the
/// downstream chain commit (`h_state_intermediates[num_accepted-1]`) stays
/// consistent — identical plumbing to `dflash_typical_accept`.
///
/// The bonus stays `verified[num_accepted]` (the target argmax at the first
/// genuinely-rejected position). Exact argmax matches accept without any
/// D2H (the rule only ever WIDENS acceptance vs strict greedy).
///
/// Returns `Some(num_accepted)` when active, `None` to defer: env off, an
/// active grammar (relaxed accept must not bypass constraint enforcement),
/// or fp32 logits (row layout assumes BF16). A failed row D2H ends the
/// prefix at that position (== strict outcome there).
fn dflash_relax_accept(
    model: &dyn Model,
    a: &ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
) -> Option<usize> {
    let cfg = dflash_relax_config()?;
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
    {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        return None; // row scan below assumes BF16
    }
    let vocab = model.vocab_size();
    let ln_ratio = if cfg.ratio > 0.0 {
        cfg.ratio.ln()
    } else {
        f32::NEG_INFINITY
    };
    // Lazily allocated on the first non-argmax position; pure argmax
    // prefixes never touch the device.
    let mut row_buf: Vec<u8> = Vec::new();
    let mut accepted = 0usize;
    let mut relax_hits = 0usize;
    for i in 0..drafts.len() {
        if i + 1 >= verified.len() {
            break;
        }
        if drafts[i] == verified[i] {
            accepted += 1;
            continue;
        }
        if drafts[i] as usize >= vocab {
            break;
        }
        if row_buf.is_empty() {
            row_buf = vec![0u8; vocab * 2];
        }
        if let Err(e) = model.copy_logits_to_host(logits_base.offset(i * vocab * 2), &mut row_buf) {
            tracing::warn!("DFlash relax accept: logits D2H failed ({e:#}); stopping prefix");
            break;
        }
        if relax_row_accepts(&row_buf, vocab, drafts[i], cfg.topk, ln_ratio) {
            accepted += 1;
            relax_hits += 1;
        } else {
            break;
        }
    }
    if relax_hits > 0 {
        tracing::debug!(
            "DFlash relax accept: +{relax_hits} non-argmax draft(s) committed \
             (total {accepted}/{}, topk={}, ratio={}, T={})",
            drafts.len(),
            cfg.topk,
            cfg.ratio,
            a.temperature,
        );
    }
    Some(accepted)
}

/// Relaxed-accept test for one host-copied BF16 logits row. Accepts `tok`
/// when it is within the top-`topk` logits OR its logit gap to the argmax
/// satisfies `l_tok - l_max >= ln_ratio` (== `p(tok)/p(max) >= ratio`,
/// temperature-free). Logits are compared in f32.
fn relax_row_accepts(bytes: &[u8], vocab: usize, tok: u32, topk: usize, ln_ratio: f32) -> bool {
    let ti = tok as usize;
    if ti >= vocab {
        return false;
    }
    let l_tok = bf16_to_f32(bytes[2 * ti], bytes[2 * ti + 1]);
    // Single pass: find max logit and count how many logits strictly exceed
    // l_tok (the draft's rank-1 position is `n_greater`). top-k holds when
    // n_greater < topk (i.e. at most topk-1 tokens beat the draft).
    let mut max_logit = f32::NEG_INFINITY;
    let mut n_greater = 0usize;
    for i in 0..vocab {
        let l = bf16_to_f32(bytes[2 * i], bytes[2 * i + 1]);
        if l > max_logit {
            max_logit = l;
        }
        if l > l_tok {
            n_greater += 1;
            // Early exit only safe if top-k is the only criterion; with the
            // ratio test we still need the true max, so don't break here.
        }
    }
    if topk >= 1 && n_greater < topk {
        return true;
    }
    if ln_ratio.is_finite() && (l_tok - max_logit) >= ln_ratio {
        return true;
    }
    false
}

/// Verify-time think mask — the throughput fix. Patches `verified` in place:
/// for every position whose raw target argmax is `<think>`/`</think>` (a tag
/// the plain sample path suppresses once thinking has ended), re-derive the
/// masked content argmax from that position's logits row and substitute it.
/// Returns the number of positions patched.
///
/// Why this recovers throughput: the DFlash accept loop scores each drafter
/// token against `verified[i]`. When the target's raw argmax is a spurious
/// `<think>`, the drafter's (correct) content prediction never matches, so the
/// entire γ chain is rejected at that position (a 0-2 token accept). Masking
/// `verified[i]` to the content token the plain path WOULD have committed makes
/// the drafter's guess match again → the chain accepts. This is a correctness
/// alignment: DFlash must reproduce the masked plain-path stream, not the
/// unmasked argmax.
///
/// EOS is deliberately NOT masked (a target EOS is a legitimate stop). No-op
/// when thinking never ended / is active, the flag is disabled, logits are
/// fp32, or the row's raw argmax doesn't match `verified[i]` (layout guard).
fn dflash_think_mask_verified_rows(
    a: &ActiveSeq,
    verified: &mut [u32],
    logits_format: VerifyLogitsFormat,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> usize {
    if !a.think_ended || a.inside_thinking || crate::dsv41::serving() {
        return 0;
    }
    if dflash_think_mask_verify_disabled() {
        return 0;
    }
    let mut skip_buf = [0u32; 2];
    let mut skip_len = 0usize;
    if let Some(t) = a.think_start_token {
        skip_buf[skip_len] = t;
        skip_len += 1;
    }
    if let Some(t) = a.think_end_token {
        skip_buf[skip_len] = t;
        skip_len += 1;
    }
    let skip = &skip_buf[..skip_len];
    if skip.is_empty() || !verified.iter().any(|t| skip.contains(t)) {
        return 0;
    }
    if logits_format == VerifyLogitsFormat::Fp32 {
        return 0; // row scan assumes BF16
    }
    let mut row = Vec::new();
    let mut patched = 0usize;
    for i in 0..verified.len() {
        if !skip.contains(&verified[i]) {
            continue;
        }
        if !fetch_row(i, &mut row) || row.is_empty() || !row.len().is_multiple_of(2) {
            tracing::warn!("think mask: verified logits unavailable at pos {i}; leaving raw");
            continue;
        }
        let vocab = row.len() / 2;
        // Layout guard: the row's unskipped argmax must equal the raw verified
        // token, else `i` doesn't map to a contiguous logits row (tree
        // compaction) and re-deriving would corrupt an unrelated position.
        match argmax_bf16_skip_tokens(&row, &[], vocab) {
            Some(raw) if raw == verified[i] => {}
            _ => continue,
        }
        if let Some(clean) = argmax_bf16_skip_tokens(&row, skip, vocab) {
            tracing::info!(
                "think mask: verified pos={i} raw={} -> clean={clean}",
                verified[i]
            );
            verified[i] = clean;
            patched += 1;
        }
    }
    patched
}

fn dflash_think_mask_verified(model: &dyn Model, a: &ActiveSeq, verified: &mut [u32]) -> usize {
    dflash_think_mask_verified_at_row(model, a, 0, verified)
}

pub(super) fn dflash_think_mask_verified_at_row(
    model: &dyn Model,
    a: &ActiveSeq,
    row_base: usize,
    verified: &mut [u32],
) -> usize {
    let base = model.logits_buffer_ptr();
    let logits_format = if model.logits_ptr_is_fp32(base) {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let vocab = model.vocab_size();
    dflash_think_mask_verified_rows(a, verified, logits_format, |i, row| {
        row.resize(vocab * 2, 0);
        match model.copy_logits_to_host(base.offset((row_base + i) * vocab * 2), row) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("think mask: verified D2H failed at pos {i} ({e:#}); leaving raw");
                false
            }
        }
    })
}

/// V4 bonus-source fix. The DDTree/flat bonus is the target's raw GPU argmax
/// at the path-tip logits row. When thinking has ALREADY ended, that argmax is
/// occasionally `<think>` or `</think>` — committing it re-opens a spurious
/// thinking block mid-content (the coding-throughput collapse). The downstream
/// `emit_step` force-exit only papers over it after the fact and still pollutes
/// the KV with the tag. This re-derives the bonus from the SAME logits row,
/// skipping `<think>` and `</think>` (never EOS, which is a legitimate stop),
/// so a valid content token is committed instead — `<think>` never enters the
/// verified stream.
///
/// Returns the ORIGINAL bonus (no-op) when: thinking is not ended, thinking is
/// active, `ATLAS_DFLASH_THINK_MASK_VERIFY=0`, `dsv41::serving()` is active,
/// the bonus is already a content token, logits are fp32 (row scan assumes
/// BF16), the D2H fails, or the row's UNSKIPPED argmax doesn't match the passed
/// bonus (a row-index sanity guard — under tree compaction `bonus_idx` may not
/// map to a contiguous logits row, and re-deriving there would commit an
/// unrelated token). Only the `None`-grammar bonus path calls this; the
/// grammar-masked bonus is an intentional constrained argmax and is left
/// untouched.
pub(super) fn reargmax_bonus_skip_think(
    model: &dyn Model,
    a: &ActiveSeq,
    bonus_idx: usize,
    bonus: u32,
) -> u32 {
    if !a.think_ended
        || a.inside_thinking
        || dflash_think_mask_verify_disabled()
        || crate::dsv41::serving()
    {
        return bonus;
    }
    // Only the think tags are masked — a legitimate EOS bonus MUST stop the
    // sequence, so it is never substituted. (The upstream verify-time think
    // mask usually already cleaned this; this is defense-in-depth for the
    // bonus slot, e.g. tree-compaction rows the layout guard skipped there.)
    let mut skip_buf = [0u32; 2];
    let mut skip_len = 0usize;
    if let Some(t) = a.think_start_token {
        skip_buf[skip_len] = t;
        skip_len += 1;
    }
    if let Some(t) = a.think_end_token {
        skip_buf[skip_len] = t;
        skip_len += 1;
    }
    let skip = &skip_buf[..skip_len];
    if !skip.contains(&bonus) {
        return bonus; // already a content token — nothing to fix
    }
    let base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(base) {
        return bonus; // row scan assumes BF16
    }
    let vocab = model.vocab_size();
    let mut row = vec![0u8; vocab * 2];
    if let Err(e) = model.copy_logits_to_host(base.offset(bonus_idx * vocab * 2), &mut row) {
        tracing::warn!("DDTree bonus source: logits D2H failed ({e:#}); keeping raw bonus");
        return bonus;
    }
    // Row-index sanity: the row's unskipped argmax MUST equal the raw bonus,
    // else the logits layout doesn't match `bonus_idx` (tree compaction) and
    // re-deriving would commit an unrelated token. Bail to the raw bonus.
    match argmax_bf16_skip_tokens(&row, &[], vocab) {
        Some(raw) if raw == bonus => {}
        _ => return bonus,
    }
    match argmax_bf16_skip_tokens(&row, skip, vocab) {
        Some(clean) => {
            tracing::info!(
                "DDTree bonus source: think tag masked (raw={bonus} -> clean={clean}, row={bonus_idx})"
            );
            clean
        }
        None => bonus, // whole row was skip tokens (degenerate) — keep raw
    }
}

/// Whether a flat non-thinking verify frame must route through the per-row
/// content policy walker instead of raw target-argmax acceptance.
///
/// Mirrors `verify_qwen4_batched_step.rs` so C=1 DFlash2 and batched native K5
/// enforce identical EOS-suppression, tool-body, and `min_tokens` boundaries.
fn flat_content_policy_required(a: &ActiveSeq, think: &ThinkSpecCtx<'_>) -> bool {
    content_policy_required(a, think)
        || a.require_tool_call
        || a.inside_tool_body
        || a.output_tokens.len() < a.min_tokens
}

/// Argmax over a host-copied BF16 logits row, skipping a set of token indices.
/// Returns `None` only if `vocab == 0` or all tokens are in the skip set.
/// Used by the DDTree bonus source fix to exclude `<think>` and `</think>`
/// tokens when `think_ended` is true so the re-derived bonus never re-opens a
/// thinking block (EOS is never skipped).
fn argmax_bf16_skip_tokens(bytes: &[u8], skip_toks: &[u32], vocab: usize) -> Option<u32> {
    spark_runtime::sampler::argmax_bf16_first_wins(&bytes[..2 * vocab], |tok| {
        !skip_toks.contains(&tok)
    })
}

/// Masked argmax over one host-copied BF16 logits row. `None` when the
/// bitmask allows zero tokens. Bit `tok` set in the i32 bitmask ⇒ allowed;
/// the pick is the shared first-wins rule
/// ([`spark_runtime::sampler::argmax_bf16_first_wins`]).
fn masked_argmax_bf16(bytes: &[u8], bitmask: &[i32], vocab: usize) -> Option<u32> {
    spark_runtime::sampler::argmax_bf16_first_wins(&bytes[..2 * vocab], |tok| {
        let (word, bit) = ((tok / 32) as usize, tok % 32);
        word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0
    })
}

#[cfg(test)]
#[path = "generic_consumed_rows_tests.rs"]
mod generic_consumed_rows_tests;

#[cfg(test)]
mod wide_verify_hidden_save_tests {
    use super::{
        DraftOrigin, K1_STAGE_SHADOW_PRE_VERIFY_LEN, K1_STAGE_SHADOW_TOKENS, WideVerifyHiddenSave,
        canonical_u32_csv, canonical_u32_csv_text, canonical_usize, dispatch_echo_stash,
        echo_state_ready, ep_kgamma_result, exact_accept_prefix, k1_stage_shadow_contract_values,
        k5_commit_parity_route_contract, selected_verified, suppress_reproposal,
        terminal_verify_commit_plan, trajectory_selector_values_match, wide_verify_hidden_save,
        wide_verify_metric_k,
    };
    use spark_model::traits::EP_VERIFY_KGAMMA_ABORT;

    #[test]
    fn ep_rank0_aborts_failed_policy_walks_but_not_valid_results() {
        assert_eq!(ep_kgamma_result(3, true, false, true), 3);
        assert_eq!(ep_kgamma_result(3, false, false, false), 3);
        assert_eq!(
            ep_kgamma_result(usize::MAX, false, false, true),
            EP_VERIFY_KGAMMA_ABORT
        );
        assert_eq!(
            ep_kgamma_result(1, true, true, false),
            EP_VERIFY_KGAMMA_ABORT
        );
        assert_eq!(
            ep_kgamma_result(1, true, false, false),
            EP_VERIFY_KGAMMA_ABORT
        );
    }

    #[test]
    fn terminal_accepted_draft_commits_actual_output_delta() {
        // v6 reproduction: the K5 verifier accepted three drafts, but max
        // output stopped on the third. The final emitted draft stays unfed,
        // exactly like target-only decoding.
        let plan = terminal_verify_commit_plan(220, 224, 224, 5, 3, 93, 96, None).unwrap();
        assert_eq!(plan.committed_inputs, 3);
        assert_eq!(plan.target_seq_len, 223);
        assert_eq!(plan.last_inter_slot, Some(2));
    }

    #[test]
    fn terminal_bonus_commits_root_and_every_accepted_draft() {
        let plan = terminal_verify_commit_plan(220, 224, 224, 5, 3, 92, 96, None).unwrap();
        assert_eq!(plan.committed_inputs, 4);
        assert_eq!(plan.target_seq_len, 224);
        assert_eq!(plan.last_inter_slot, Some(3));
    }

    #[test]
    fn terminal_tree_slot_comes_from_retained_path_prefix() {
        let path = [1, 4, 7];
        let root_only =
            terminal_verify_commit_plan(100, 104, 104, 8, 3, 10, 11, Some(&path)).unwrap();
        assert_eq!(root_only.last_inter_slot, Some(0));

        let mid_path =
            terminal_verify_commit_plan(100, 104, 104, 8, 3, 10, 13, Some(&path)).unwrap();
        assert_eq!(mid_path.committed_inputs, 3);
        assert_eq!(mid_path.last_inter_slot, Some(4));

        let bonus = terminal_verify_commit_plan(100, 104, 104, 8, 3, 10, 14, Some(&path)).unwrap();
        assert_eq!(bonus.last_inter_slot, Some(7));
    }

    #[test]
    fn terminal_zero_emission_selects_checkpoint_restore() {
        let plan = terminal_verify_commit_plan(220, 224, 224, 5, 3, 93, 93, None).unwrap();
        assert_eq!(plan.committed_inputs, 0);
        assert_eq!(plan.target_seq_len, 220);
        assert_eq!(plan.last_inter_slot, None);
    }

    #[test]
    fn terminal_zero_emission_rolls_back_qsa_before_recurrent_restore() {
        let source = include_str!("verify_dflash_step.rs");
        let body = source
            .split("match plan.last_inter_slot {")
            .nth(1)
            .expect("terminal reconciliation match");
        let none = body
            .split("Some(last_inter_slot) if tree_mode")
            .next()
            .unwrap();
        assert!(none.contains("rollback_failed_verify_frame(model, &mut a.seq, k_verify)"));
        let helper = source
            .split("pub(super) fn rollback_failed_verify_frame")
            .nth(1)
            .expect("shared verify rollback helper")
            .split("\n}")
            .next()
            .unwrap();
        assert!(helper.contains("model.abort_verify_state(seq, k_verify)"));
    }

    #[test]
    fn terminal_plan_rejects_impossible_or_truncated_state() {
        assert!(terminal_verify_commit_plan(220, 224, 224, 5, 3, 96, 95, None).is_err());
        assert!(terminal_verify_commit_plan(220, 225, 225, 5, 3, 90, 95, None).is_err());
        assert!(terminal_verify_commit_plan(220, 222, 222, 5, 3, 90, 93, None).is_err());
        assert!(terminal_verify_commit_plan(220, 224, 224, 3, 3, 93, 96, None).is_err());
        assert!(terminal_verify_commit_plan(100, 104, 104, 8, 3, 10, 13, Some(&[1])).is_err());
    }

    #[test]
    fn all_generic_terminal_returns_reconcile_before_tree_clear() {
        let source = include_str!("verify_dflash_step.rs");
        for label in [
            "terminal verify-state commit (policy walk)",
            "terminal verify-state commit (accepted draft)",
            "terminal verify-state commit (bonus)",
        ] {
            let error_log = source.find(label).expect("terminal branch exists");
            let branch = &source[..error_log];
            let reconcile = branch
                .rfind("reconcile_terminal_verify_state(")
                .expect("terminal branch reconciles target state");
            let clear = source[error_log..]
                .find("model.clear_ddtree_parent_ids();")
                .map(|offset| error_log + offset)
                .expect("terminal branch clears tree state");
            assert!(reconcile < error_log && error_log < clear);
        }
    }

    #[test]
    fn native_mtp_metric_uses_observed_verify_input_rows() {
        use spark_model::speculative::DraftProposerKind::Qwen4NativeMtp;

        for rows in [4, 5, 6, 9, 17] {
            assert_eq!(
                wide_verify_metric_k(Some(Qwen4NativeMtp), DraftOrigin::Proposer, rows).as_ref(),
                rows.to_string(),
            );
        }
    }

    #[test]
    fn wide_verify_metric_never_attests_unknown_or_retrieved_drafts_as_native_k5() {
        use spark_model::speculative::DraftProposerKind::{DFlash, Other, Qwen4NativeMtp};

        for kind in [None, Some(DFlash), Some(Other)] {
            assert_eq!(
                wide_verify_metric_k(kind, DraftOrigin::Proposer, 5).as_ref(),
                "dflash"
            );
        }
        for origin in [DraftOrigin::RestStore, DraftOrigin::SelfContext] {
            assert_eq!(
                wide_verify_metric_k(Some(Qwen4NativeMtp), origin, 5).as_ref(),
                "dflash"
            );
        }
    }

    #[test]
    fn c1_verify_warms_ple_draft_ahead_after_repropose() {
        let source = include_str!("verify_dflash_step.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let propose = source
            .find("proposal_lifecycle::propose_and_install_with(")
            .expect("propose call site");
        let warm = source[propose..]
            .find("warm_ple_draft_ahead_for_seq(model, a);")
            .map(|off| propose + off)
            .expect("PLE draft-ahead pre-warm after C=1 propose");
        let finalize = source[propose..]
            .find("let spec_phase = spec_cycle.finalize(spec_phase);")
            .map(|off| propose + off)
            .expect("spec_cycle finalize");
        assert!(propose < warm && warm < finalize);
    }

    #[test]
    fn native_mtp_metric_is_wired_to_actual_installed_arm_and_frame() {
        let source = include_str!("verify_dflash_step.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let call = source.find("let metric_k = wide_verify_metric_k(").unwrap();
        let metric = &source[call..source[call..].find(".inc();").unwrap() + call];
        assert!(metric.contains("model.proposer_arm_kind(model.proposer_arm())"));
        assert!(metric.contains("draft_origin,"));
        assert!(metric.contains("k_verify,"));
        assert!(metric.contains("&metric_k,"));
    }

    #[test]
    fn native_mtp_uses_bonus_input_row_for_every_nd4_accept_count() {
        for accepted in 0..=4 {
            assert_eq!(
                wide_verify_hidden_save(false, 99, accepted, None),
                WideVerifyHiddenSave::NativeMtpRow(accepted),
                "accepted={accepted}"
            );
        }
    }

    #[test]
    fn proposer_kind_keeps_dflash_token_and_tree_row_contracts_distinct() {
        assert_eq!(
            wide_verify_hidden_save(true, 99, 2, Some(7)),
            WideVerifyHiddenSave::DflashToken(99)
        );
        assert_eq!(
            wide_verify_hidden_save(false, 99, 2, Some(7)),
            WideVerifyHiddenSave::NativeMtpRow(7)
        );
    }

    #[test]
    fn echo_requires_both_hidden_save_and_proposer_trim_success() {
        assert!(echo_state_ready(true, true));
        assert!(!echo_state_ready(false, true));
        assert!(!echo_state_ready(true, false));
        assert!(!echo_state_ready(false, false));
    }

    #[test]
    fn echo_dispatch_clears_preseeded_stash_on_each_state_failure() {
        for (hidden_saved, proposer_trimmed) in [(false, true), (true, false), (false, false)] {
            let mut stored = vec![91, 92, 93];
            let mut calls = 0;
            dispatch_echo_stash(
                true,
                echo_state_ready(hidden_saved, proposer_trimmed),
                true,
                &[1, 2, 3],
                1,
                |tokens, accepted| {
                    calls += 1;
                    assert_eq!(accepted, 0);
                    stored.clear();
                    stored.extend_from_slice(tokens);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(calls, 1);
            assert!(stored.is_empty());
        }
    }

    #[test]
    fn echo_dispatch_is_once_only_for_success_policy_clear_and_disabled() {
        let verified = [10, 11, 12, 13];
        let mut calls = Vec::new();
        dispatch_echo_stash(true, true, true, &verified, 2, |tokens, accepted| {
            calls.push((tokens.to_vec(), accepted));
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, [(verified.to_vec(), 2)]);

        calls.clear();
        dispatch_echo_stash(true, true, false, &verified, 2, |tokens, accepted| {
            calls.push((tokens.to_vec(), accepted));
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, [(Vec::new(), 0)]);

        calls.clear();
        dispatch_echo_stash(false, false, false, &verified, 2, |tokens, accepted| {
            calls.push((tokens.to_vec(), accepted));
            Ok(())
        })
        .unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn echo_callback_error_suppresses_reproposal() {
        let result = dispatch_echo_stash(true, false, false, &[10, 11, 12], 1, |_, _| {
            anyhow::bail!("hostile stale-stash clear failure")
        });
        assert!(result.is_err());
        assert!(suppress_reproposal(false, result.is_err()));
        assert!(suppress_reproposal(true, false));
        assert!(!suppress_reproposal(false, false));
    }

    #[test]
    fn echo_failure_gate_precedes_reproposal_and_preserves_native_return() {
        let source = include_str!("verify_dflash_step.rs");
        let hidden = source.find("let dflash_hidden_saved =").unwrap();
        let trim = source[hidden..]
            .find("let dflash_proposer_trimmed =")
            .map(|offset| hidden + offset)
            .unwrap();
        let dispatch = source[trim..]
            .find("let echo_result = dispatch_echo_stash(")
            .map(|offset| trim + offset)
            .unwrap();
        let echo_error = source[dispatch..]
            .find("let echo_dispatch_failed =")
            .map(|offset| dispatch + offset)
            .unwrap();
        let repropose = source[echo_error..]
            .find("let skip_propose =")
            .map(|offset| echo_error + offset)
            .unwrap();
        let retrieval = source[repropose..]
            .find("let retrieved = if skip_propose || skip_repropose_diag")
            .map(|offset| repropose + offset)
            .unwrap();
        let should_propose = source[retrieval..]
            .find("let should_propose =")
            .map(|offset| retrieval + offset)
            .unwrap();
        let lifecycle = source[should_propose..]
            .find("proposal_lifecycle::propose_and_install_with(")
            .map(|offset| should_propose + offset)
            .unwrap();
        assert!(
            hidden < trim
                && trim < dispatch
                && dispatch < echo_error
                && echo_error < repropose
                && repropose < retrieval
                && retrieval < should_propose
                && should_propose < lifecycle
        );
        assert!(source[repropose..].starts_with("let skip_propose ="));
        let scheduling = &source[repropose..lifecycle];
        assert!(scheduling.contains("suppress_reproposal("));
        let compact: String = scheduling.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains(
            "suppress_reproposal(dflash_grammar_skip_propose(model,a),echo_dispatch_failed)"
        ));
        assert!(scheduling.contains("if skip_propose || skip_repropose_diag"));
        assert!(scheduling.contains(
            "let should_propose = !skip_propose && !skip_repropose_diag && retrieved.is_none();"
        ));
        let lifecycle_end = source[lifecycle..]
            .find("|a, propose_result|")
            .map(|offset| lifecycle + offset)
            .unwrap();
        let lifecycle_call = &source[lifecycle..lifecycle_end];
        assert!(lifecycle_call.contains("should_propose,"));
        assert!(!lifecycle_call.contains("true,"));
        let native_failure = &source[hidden..trim];
        assert!(native_failure.contains("if !proposer_is_dflash"));
        assert!(native_failure.contains("return;"));
    }

    #[test]
    fn serial_selection_controls_exact_acceptance_bonus_and_emission() {
        let drafts = [10, 20, 30, 40];
        let batched = [10, 20, 30, 40, 50];
        let serial = [10, 99, 30, 40, 50];

        let selected = selected_verified(&batched, Some(&serial));
        let accepted = exact_accept_prefix(&drafts, selected);
        let bonus = selected[accepted];
        let mut emitted = drafts[..accepted].to_vec();
        emitted.push(bonus);

        assert_eq!(accepted, 1);
        assert_eq!(bonus, 99);
        assert_eq!(emitted, [10, 99]);
    }

    #[test]
    fn normal_path_keeps_batched_acceptance_when_serial_is_absent() {
        let drafts = [10, 20, 30, 40];
        let batched = [10, 20, 30, 40, 50];

        let selected = selected_verified(&batched, None);
        let accepted = exact_accept_prefix(&drafts, selected);

        assert_eq!(accepted, drafts.len());
        assert_eq!(selected[accepted], 50);
    }

    #[test]
    fn k5_commit_parity_requires_c1_flat_single_rank() {
        assert!(k5_commit_parity_route_contract(true, true, true, false).unwrap());
        assert!(!k5_commit_parity_route_contract(false, false, false, true).unwrap());
        for (c1, flat, ep) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert!(k5_commit_parity_route_contract(true, c1, flat, ep).is_err());
        }
    }

    #[test]
    fn k1_stage_shadow_contract_is_explicit_exact_and_production_only() {
        let selected = true;
        assert!(
            k1_stage_shadow_contract_values(
                Some("1"),
                Some("1"),
                None,
                None,
                selected,
                K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                &K1_STAGE_SHADOW_TOKENS,
                true,
                true,
                false,
                5,
            )
            .unwrap()
        );
        assert!(
            !k1_stage_shadow_contract_values(
                Some("1"),
                Some("1"),
                None,
                None,
                false,
                K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                &K1_STAGE_SHADOW_TOKENS,
                true,
                true,
                false,
                5,
            )
            .unwrap()
        );
        assert!(
            !k1_stage_shadow_contract_values(
                None,
                None,
                None,
                None,
                selected,
                K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                &K1_STAGE_SHADOW_TOKENS,
                true,
                true,
                false,
                5,
            )
            .unwrap()
        );

        for invalid in [Some(""), Some("yes"), Some("2")] {
            assert!(
                k1_stage_shadow_contract_values(
                    invalid,
                    Some("1"),
                    None,
                    None,
                    selected,
                    K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                    &K1_STAGE_SHADOW_TOKENS,
                    true,
                    true,
                    false,
                    5,
                )
                .is_err()
            );
        }
        for invalid in [
            (Some("0"), None, None, true, true, false, 5),
            (Some("1"), Some("1"), None, true, true, false, 5),
            (Some("1"), None, Some("1"), true, true, false, 5),
            (Some("1"), None, None, false, true, false, 5),
            (Some("1"), None, None, true, false, false, 5),
            (Some("1"), None, None, true, true, true, 5),
            (Some("1"), None, None, true, true, false, 4),
        ] {
            let (diag, serial, skip, c1, flat, ep, k) = invalid;
            assert!(
                k1_stage_shadow_contract_values(
                    Some("1"),
                    diag,
                    serial,
                    skip,
                    selected,
                    K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                    &K1_STAGE_SHADOW_TOKENS,
                    c1,
                    flat,
                    ep,
                    k,
                )
                .is_err()
            );
        }
        assert!(
            k1_stage_shadow_contract_values(
                Some("1"),
                Some("1"),
                None,
                None,
                selected,
                K1_STAGE_SHADOW_PRE_VERIFY_LEN + 1,
                &K1_STAGE_SHADOW_TOKENS,
                true,
                true,
                false,
                5,
            )
            .is_err()
        );
        assert!(
            k1_stage_shadow_contract_values(
                Some("1"),
                Some("1"),
                None,
                None,
                selected,
                K1_STAGE_SHADOW_PRE_VERIFY_LEN,
                &[271, 1596, 1144, 4087, 1157],
                true,
                true,
                false,
                5,
            )
            .is_err()
        );
    }

    #[test]
    fn stage_shadow_restores_before_the_production_verifier_controls_acceptance() {
        let source = include_str!("verify_dflash_step.rs");
        let shadow = source
            .find("let shadow_verified = run_k5_stage_shadow(")
            .expect("selected frame runs a diagnostic K5 shadow");
        let restored = source[shadow..]
            .find("DFLASH_K1_STAGE_SHADOW_RESTORED")
            .map(|offset| shadow + offset)
            .expect("shadow restoration has a receipt");
        let production = source[restored..]
            .find("let mut verified = match model.decode_verify_dflash(")
            .map(|offset| restored + offset)
            .expect("production K5 runs after restoration");
        let production_receipt = source[production..]
            .find("DFLASH_K1_STAGE_PRODUCTION")
            .map(|offset| production + offset)
            .expect("production ownership has a receipt");
        let selection = source[production_receipt..]
            .find("verified = selected_verified(")
            .map(|offset| production_receipt + offset)
            .expect("acceptance selection follows the production receipt");
        assert!(shadow < restored && restored < production);
        assert!(production < production_receipt && production_receipt < selection);
    }

    #[test]
    fn qsa_k5_holds_untransactional_preverify_k1_and_rolls_back_before_postverify_k1() {
        let source = include_str!("verify_dflash_step.rs");
        let production = source.split("#[cfg(test)]").next().unwrap();
        assert!(production.contains(
            "if tokens.len() > 1 && a.seq.qwen4_qsa_required && (serial_oracle_requested || stage_shadow) {"
        ));
        assert!(production.contains("transactional K1 replay is not implemented"));
        let parity = source
            .split("let commit_res = if k5_commit_parity")
            .nth(1)
            .expect("K5 commit-parity branch");
        let rollback = parity
            .find("commit_qwen4_qsa_for_sequence(&a.seq, 0, k_verify)")
            .expect("QSA rollback before K1 replay");
        let replay = parity
            .find("for &token in tokens.iter().take(total_accepted)")
            .expect("ordinary K1 replay");
        assert!(rollback < replay);
    }

    #[test]
    fn trajectory_selector_values_are_canonical_and_exact() {
        let tokens = [15, 11, 271, 13, 11];
        assert_eq!(canonical_usize("SEQ", "31").unwrap(), 31);
        assert_eq!(
            canonical_u32_csv("TOKENS", "15,11,271,13,11").unwrap(),
            [15, 11, 271, 13, 11]
        );
        assert_eq!(
            canonical_u32_csv_text(&[15, 11, 271, 13, 11]),
            "15,11,271,13,11"
        );

        for invalid in ["", "031", "+31", " 31", "31 "] {
            assert!(canonical_usize("SEQ", invalid).is_err(), "{invalid:?}");
        }
        for invalid in ["", "01", "1,", ",1", "1,,2", "1, 2", "+1", "4294967296"] {
            assert!(canonical_u32_csv("TOKENS", invalid).is_err(), "{invalid:?}");
        }
        assert!(!trajectory_selector_values_match(None, None, None, 31, &tokens).unwrap());
        assert!(!trajectory_selector_values_match(Some("0"), None, None, 31, &tokens).unwrap());
        assert!(
            trajectory_selector_values_match(
                Some("1"),
                Some("31"),
                Some("15,11,271,13,11"),
                31,
                &tokens,
            )
            .unwrap()
        );
        assert!(
            !trajectory_selector_values_match(
                Some("1"),
                Some("32"),
                Some("15,11,271,13,11"),
                31,
                &tokens,
            )
            .unwrap()
        );
        assert!(
            trajectory_selector_values_match(
                Some("0"),
                Some("31"),
                Some("15,11,271,13,11"),
                31,
                &tokens,
            )
            .is_err()
        );
        assert!(
            trajectory_selector_values_match(
                Some("1"),
                Some("31"),
                Some("15,11,271,13"),
                31,
                &tokens,
            )
            .is_err()
        );
        assert!(trajectory_selector_values_match(Some("yes"), None, None, 31, &tokens).is_err());
    }

    #[test]
    fn skipped_propose_reconciles_dflash_captures_before_external_draft_install() {
        let source = include_str!("verify_dflash_step.rs");
        let production = source.split("#[cfg(test)]").next().unwrap();
        let should_propose = production
            .find("let should_propose = !skip_propose && !skip_repropose_diag && retrieved.is_none();")
            .expect("should_propose gate");
        let reconcile = production[should_propose..]
            .find("let skip_propose_reconciled =")
            .map(|offset| should_propose + offset)
            .expect("reconcile gate for skipped proposals");
        let lifecycle = production[reconcile..]
            .find("proposal_lifecycle::propose_and_install_with(")
            .map(|offset| reconcile + offset)
            .expect("proposal lifecycle handoff");
        let install_guard = production[lifecycle..]
            .find("if skip_propose_reconciled")
            .map(|offset| lifecycle + offset)
            .expect("external draft install gated on reconciliation");
        assert!(should_propose < reconcile && reconcile < lifecycle && lifecycle < install_guard);

        let reconcile_block = &production[reconcile..lifecycle];
        assert!(reconcile_block.contains("used_retrieval"));
        assert!(
            reconcile_block
                .contains("skip_propose && !skip_repropose_diag && dflash_proposer_trimmed")
        );
        assert!(reconcile_block.contains("proposer_is_dflash && !dflash_proposer_trimmed"));
        assert!(
            reconcile_block
                .contains("super::mtp_step::reconcile_dflash_before_external_draft(model, a)")
        );
        assert!(production[install_guard..].contains("let Some((origin, chain)) = retrieved"));
    }
}

#[cfg(test)]
mod host_argmax_tests {
    use super::{argmax_bf16_skip_tokens, masked_argmax_bf16};

    fn row(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn skip_argmax_orders_an_all_negative_row_numerically() {
        let r = row(&[-8.0, -1.0, -2.0, -8.0]);
        assert_eq!(argmax_bf16_skip_tokens(&r, &[], 4), Some(1));
        assert_eq!(argmax_bf16_skip_tokens(&r, &[1], 4), Some(2));
    }

    #[test]
    fn masked_argmax_orders_an_all_negative_row_numerically() {
        let r = row(&[-8.0, -1.0, -2.0, -8.0]);
        assert_eq!(masked_argmax_bf16(&r, &[0b1111], 4), Some(1));
        assert_eq!(masked_argmax_bf16(&r, &[0b1101], 4), Some(2));
        assert_eq!(masked_argmax_bf16(&r, &[0], 4), None);
    }

    #[test]
    fn host_argmaxes_keep_the_first_of_a_signed_zero_tie() {
        let r = row(&[-8.0, -0.0, 0.0, -8.0]);
        assert_eq!(argmax_bf16_skip_tokens(&r, &[], 4), Some(1));
        assert_eq!(masked_argmax_bf16(&r, &[0b1111], 4), Some(1));
    }
}

#[cfg(test)]
mod content_policy_gate_tests {
    use super::{ResponseSink, ThinkSpecCtx, flat_content_policy_required};

    fn neutral_greedy_content_seq() -> super::ActiveSeq {
        let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
        a.inside_thinking = false;
        a.think_ended = false;
        a.think_just_ended = false;
        a.require_tool_call = false;
        a.inside_tool_body = false;
        a.suppress_tool_call = false;
        a.min_tokens = 0;
        a.output_tokens = vec![1, 2];
        a
    }

    fn neutral_ctx() -> ThinkSpecCtx<'static> {
        ThinkSpecCtx {
            enabled: true,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        }
    }

    #[test]
    fn c1_dflash_content_policy_gate_matches_batched_k5_for_tool_and_min_tokens() {
        let ctx = neutral_ctx();
        let baseline = neutral_greedy_content_seq();
        assert!(!flat_content_policy_required(&baseline, &ctx));

        let mut req_tool = neutral_greedy_content_seq();
        req_tool.require_tool_call = true;
        assert!(flat_content_policy_required(&req_tool, &ctx));

        let mut in_tool = neutral_greedy_content_seq();
        in_tool.inside_tool_body = true;
        assert!(flat_content_policy_required(&in_tool, &ctx));

        let mut under_min = neutral_greedy_content_seq();
        under_min.min_tokens = 3;
        assert!(flat_content_policy_required(&under_min, &ctx));
        under_min.output_tokens.push(3);
        assert!(!flat_content_policy_required(&under_min, &ctx));
    }
}
