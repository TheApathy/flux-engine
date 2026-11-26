// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed planning for exact M16 QSA and causal attention batching.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::ModelConfig;

use crate::traits::Qwen4PrefillLineage;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_CORE16";
pub(crate) const DEVICE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_DEVICE16";
pub(crate) const HC_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_HC16";
pub(crate) const CORE32_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_CORE32";
pub(crate) const MARCONI_FAST_CONT_SELECTOR: &str = "ATLAS_QWEN4_MARCONI_FAST_CONT";
pub(crate) const TILE_ROWS: usize = 16;
pub(crate) const TILE32_ROWS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub(crate) full_tiles: usize,
    pub(crate) tail_rows: usize,
    pub(crate) block_table_words: usize,
    pub(crate) seq_lens_offset: usize,
    pub(crate) scratch_bytes: usize,
}

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{SELECTOR} must be absent, 0, or 1; got {other:?}"),
    }
}

fn parse_device(value: Option<&str>, core_selected: bool) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{DEVICE_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || core_selected,
        "{DEVICE_SELECTOR}=1 requires {SELECTOR}=1"
    );
    Ok(selected)
}

fn parse_hc(value: Option<&str>, core_selected: bool, device_selected: bool) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{HC_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || (core_selected && device_selected),
        "{HC_SELECTOR}=1 requires {SELECTOR}=1 and {DEVICE_SELECTOR}=1"
    );
    Ok(selected)
}

fn parse_core32(
    value: Option<&str>,
    core_selected: bool,
    device_selected: bool,
    hc_selected: bool,
) -> Result<bool> {
    let selected = match value {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => bail!("{CORE32_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    };
    ensure!(
        !selected || (core_selected && device_selected && hc_selected),
        "{CORE32_SELECTOR}=1 requires {SELECTOR}=1, {DEVICE_SELECTOR}=1, and {HC_SELECTOR}=1"
    );
    Ok(selected)
}

/// Parse `ATLAS_QWEN4_MARCONI_FAST_CONT` (default ON; `0|off|false` disables).
fn parse_marconi_fast_cont(value: Option<&str>) -> bool {
    let off = |v: &str| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false"
        )
    };
    !value.is_some_and(off)
}

pub(crate) fn marconi_fast_cont_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_marconi_fast_cont(std::env::var(MARCONI_FAST_CONT_SELECTOR).ok().as_deref())
    })
}

/// Turns the whole fast-prefill family off on this thread until the guard
/// drops (see [`super::family_suppressed`]). A prefill that `admit_surface`
/// rejects (including a non-selected long prompt, a mismatched continuation,
/// vision, or high-speed-swap) takes the ordinary prefill path instead of
/// failing the request; prompts the family admits are untouched.
#[must_use]
pub(crate) struct SuppressGuard(bool);

impl Drop for SuppressGuard {
    fn drop(&mut self) {
        super::set_family_suppressed(self.0);
    }
}

/// Suppress the experimental family until the returned guard drops.
/// Alternate model entrypoints use this before any stateful work when they
/// cannot preserve the selected C1 request contract.
pub(crate) fn suppress_family() -> SuppressGuard {
    SuppressGuard(super::set_family_suppressed(true))
}

/// Keeps any request-scoped suppression alive and records whether the exact
/// fast surface was admitted. Default-off is deliberately not admission.
#[must_use]
pub(crate) struct SurfaceAdmission {
    _suppression: Option<SuppressGuard>,
    admitted: bool,
}

impl SurfaceAdmission {
    pub(crate) fn admitted(&self) -> bool {
        self.admitted
    }
}

/// Owned proof that one exact fast chunk may advance lineage after every
/// model effect and finalization step has returned successfully. Beginning an
/// attempt first leaves the sequence terminally `Ordinary`; dropping this
/// ticket on any error or early return therefore fails closed.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct FastLineageTicket {
    chunk_start: usize,
    next_offset: usize,
    prompt_tokens: Box<[u32]>,
}

/// Begin one exact 8K/2K fast-lineage transition.
///
/// Callers invoke this only when the multi-chunk selector is explicitly on.
/// A live lineage presented to any other selected route is poisoned. For the
/// selected prompt, every rejected surface, wrong offset, rollback, identity
/// change, or prior `Ordinary` state remains permanently ordinary.
pub(crate) fn begin_fast_lineage(
    state: &mut Qwen4PrefillLineage,
    prompt_tokens: &[u32],
    chunk_start: usize,
    chunk_rows: usize,
    surface_admitted: bool,
) -> Option<FastLineageTicket> {
    begin_fast_lineage_for_geometry(
        super::geometry_or_legacy(),
        state,
        prompt_tokens,
        chunk_start,
        chunk_rows,
        surface_admitted,
    )
}

/// Whether this chunk needs the persistent lineage: the prompt is one the
/// `W` lineage owns, or a bounded prepared chunk splits a shorter prompt. A
/// fresh whole-prompt chunk (start 0, all rows) never does, so a single-chunk
/// prompt keeps its historical `Unset` state.
fn chunk_needs_lineage(
    geometry: super::ChunkGeometry,
    state: &Qwen4PrefillLineage,
    prompt_rows: usize,
    chunk_start: usize,
    chunk_rows: usize,
) -> bool {
    geometry.lineage_prompt(prompt_rows)
        || (geometry.chunked_prompt(prompt_rows)
            && !(chunk_start == 0
                && chunk_rows == prompt_rows
                && matches!(state, Qwen4PrefillLineage::Unset)))
}

pub(super) fn begin_fast_lineage_for_geometry(
    geometry: super::ChunkGeometry,
    state: &mut Qwen4PrefillLineage,
    prompt_tokens: &[u32],
    chunk_start: usize,
    chunk_rows: usize,
    surface_admitted: bool,
) -> Option<FastLineageTicket> {
    if !chunk_needs_lineage(
        geometry,
        state,
        prompt_tokens.len(),
        chunk_start,
        chunk_rows,
    ) {
        if matches!(state, Qwen4PrefillLineage::Fast { .. }) {
            *state = Qwen4PrefillLineage::Ordinary;
        }
        return None;
    }

    // Install the rollback state before any caller-visible effects. The only
    // path out of Ordinary is commit_fast_lineage after complete success.
    let previous = std::mem::replace(state, Qwen4PrefillLineage::Ordinary);
    let next_offset = chunk_start.checked_add(chunk_rows)?;
    if !surface_admitted
        || !geometry.chunk_admissible(chunk_start, chunk_rows, prompt_tokens.len())
        || next_offset > prompt_tokens.len()
    {
        return None;
    }

    let prompt_tokens = match previous {
        Qwen4PrefillLineage::Unset if chunk_start == 0 => prompt_tokens.into(),
        Qwen4PrefillLineage::Fast {
            next_offset,
            prompt_tokens: identity,
        } if next_offset == chunk_start && identity.as_ref() == prompt_tokens => identity,
        Qwen4PrefillLineage::Unset
        | Qwen4PrefillLineage::Fast { .. }
        | Qwen4PrefillLineage::Ordinary => return None,
    };

    Some(FastLineageTicket {
        chunk_start,
        next_offset,
        prompt_tokens,
    })
}

/// Permanently select the ordinary route before an alternate entrypoint does
/// any work. Non-8K requests leave a fresh `Unset` state alone, but switching
/// an already-live fast sequence to a different prompt also poisons it.
pub(crate) fn mark_ordinary_for_selected_route(
    state: &mut Qwen4PrefillLineage,
    prompt_rows: usize,
) -> bool {
    let marked = super::geometry_or_legacy().lineage_prompt(prompt_rows)
        || matches!(state, Qwen4PrefillLineage::Fast { .. });
    if marked {
        *state = Qwen4PrefillLineage::Ordinary;
    }
    marked
}

/// [`mark_ordinary_for_selected_route`] for an alternate entrypoint that
/// itself splits the prompt into `chunk_rows`-row chunks: a prompt of at most
/// `W` rows that it splits could otherwise start a bounded-chunk lineage
/// through the guarded C1 entry it calls per chunk.
pub(crate) fn mark_ordinary_for_split_route(
    state: &mut Qwen4PrefillLineage,
    prompt_rows: usize,
    chunk_rows: usize,
) -> bool {
    let split = chunk_rows < prompt_rows && super::geometry_or_legacy().chunked_prompt(prompt_rows);
    let marked = mark_ordinary_for_selected_route(state, prompt_rows);
    if split && !marked {
        *state = Qwen4PrefillLineage::Ordinary;
    }
    marked || split
}

/// Commit a ticket only when the caller proves the same prompt and completed
/// extent. The state must still be the rollback `Ordinary` installed by begin.
pub(crate) fn commit_fast_lineage(
    state: &mut Qwen4PrefillLineage,
    ticket: FastLineageTicket,
    prompt_tokens: &[u32],
    chunk_start: usize,
    chunk_end: usize,
) -> Result<()> {
    commit_fast_lineage_for_geometry(
        super::geometry_or_legacy(),
        state,
        ticket,
        prompt_tokens,
        chunk_start,
        chunk_end,
    )
}

pub(super) fn commit_fast_lineage_for_geometry(
    geometry: super::ChunkGeometry,
    state: &mut Qwen4PrefillLineage,
    ticket: FastLineageTicket,
    prompt_tokens: &[u32],
    chunk_start: usize,
    chunk_end: usize,
) -> Result<()> {
    if !matches!(state, Qwen4PrefillLineage::Ordinary) {
        *state = Qwen4PrefillLineage::Ordinary;
        bail!("fast-prefill lineage changed while a chunk was in flight");
    }
    if ticket.chunk_start != chunk_start
        || ticket.next_offset != chunk_end
        || !chunk_end
            .checked_sub(chunk_start)
            .is_some_and(|rows| geometry.chunk_admissible(chunk_start, rows, prompt_tokens.len()))
        || chunk_end > prompt_tokens.len()
    {
        bail!("fast-prefill lineage ticket extent mismatch");
    }
    if !geometry.chunked_prompt(prompt_tokens.len())
        || ticket.prompt_tokens.as_ref() != prompt_tokens
    {
        bail!("fast-prefill lineage ticket prompt identity mismatch");
    }
    *state = Qwen4PrefillLineage::Fast {
        next_offset: ticket.next_offset,
        prompt_tokens: ticket.prompt_tokens,
    };
    Ok(())
}

/// Advance fast lineage across a fully cached Marconi early-return chunk.
pub(crate) fn advance_fast_lineage_for_marconi_skip(
    state: &mut Qwen4PrefillLineage,
    ticket: FastLineageTicket,
    tokens: &[u32],
    start: usize,
    end: usize,
) -> Result<()> {
    advance_marconi_skip_for_geometry(
        super::geometry_or_legacy(),
        state,
        ticket,
        tokens,
        start,
        end,
    )
}

fn advance_marconi_skip_for_geometry(
    g: super::ChunkGeometry,
    state: &mut Qwen4PrefillLineage,
    ticket: FastLineageTicket,
    tokens: &[u32],
    start: usize,
    end: usize,
) -> Result<()> {
    ensure!(
        end < tokens.len(),
        "Marconi skip advance requires a non-final chunk"
    );
    commit_fast_lineage_for_geometry(g, state, ticket, tokens, start, end)
}

fn suppressed() -> bool {
    super::family_suppressed()
}

/// Admit this prefill surface to the family, or suppress the family for the
/// rest of the caller's scope (returned guard) when it cannot serve it.
pub(crate) fn admit_or_suppress(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    is_last_chunk: bool,
    request_chain_check: impl FnOnce() -> bool,
    continuation_state_check: impl FnOnce() -> bool,
    high_speed_swap: bool,
) -> Result<SurfaceAdmission> {
    if !selected()? {
        return Ok(SurfaceAdmission {
            _suppression: None,
            admitted: false,
        });
    }
    match admit_surface(
        has_vision,
        sequence_start,
        chunk_start,
        chunk_rows,
        total_rows,
        is_last_chunk,
        request_chain_check(),
        continuation_state_check(),
        high_speed_swap,
    ) {
        Ok(()) => {
            // Per-chunk receipt for wide chunks (rows above the legacy 2048 cap).
            if chunk_rows > super::MULTI_CHUNK_ROWS {
                tracing::info!(
                    "QWEN4_PREFILL_WIDE_CHUNK start={chunk_start} rows={chunk_rows} total={total_rows} last={is_last_chunk}"
                );
            }
            // Per-chunk receipt for a bounded prepared chunk (interleave):
            // a prepared width below W that is not the prompt's remainder.
            let geometry = super::geometry_or_legacy();
            if geometry.tail
                && chunk_rows < geometry.rows
                && !is_last_chunk
                && geometry.is_prepared(chunk_rows)
            {
                tracing::info!(
                    "QWEN4_PREFILL_BOUNDED_CHUNK start={chunk_start} rows={chunk_rows} total={total_rows} wide={}",
                    geometry.rows
                );
            }
            Ok(SurfaceAdmission {
                _suppression: None,
                admitted: true,
            })
        }
        Err(reason) => {
            if super::wide_rows().ok().flatten().is_some() {
                tracing::warn!(
                    "QWEN4_PREFILL_WIDE_SUPPRESSED start={chunk_start} rows={chunk_rows} total={total_rows} ({reason:#})"
                );
            }
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "{SELECTOR}: fast-prefill family suppressed for this prefill ({reason:#}); using the ordinary prefill path (logged once)"
                )
            });
            Ok(SurfaceAdmission {
                _suppression: Some(SuppressGuard(super::set_family_suppressed(true))),
                admitted: false,
            })
        }
    }
}

pub(crate) fn selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    match std::env::var(SELECTOR) {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(error) => Err(error).with_context(|| format!("invalid {SELECTOR}")),
    }
}

pub(crate) fn device_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    match std::env::var(DEVICE_SELECTOR) {
        Ok(value) => parse_device(Some(&value), core),
        Err(std::env::VarError::NotPresent) => parse_device(None, core),
        Err(error) => Err(error).with_context(|| format!("invalid {DEVICE_SELECTOR}")),
    }
}

pub(crate) fn hc_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    let device = device_selected()?;
    match std::env::var(HC_SELECTOR) {
        Ok(value) => parse_hc(Some(&value), core, device),
        Err(std::env::VarError::NotPresent) => parse_hc(None, core, device),
        Err(error) => Err(error).with_context(|| format!("invalid {HC_SELECTOR}")),
    }
}

pub(crate) fn core32_selected() -> Result<bool> {
    if suppressed() {
        return Ok(false);
    }
    let core = selected()?;
    let device = device_selected()?;
    let hc = hc_selected()?;
    match std::env::var(CORE32_SELECTOR) {
        Ok(value) => parse_core32(Some(&value), core, device, hc),
        Err(std::env::VarError::NotPresent) => parse_core32(None, core, device, hc),
        Err(error) => Err(error).with_context(|| format!("invalid {CORE32_SELECTOR}")),
    }
}

pub(crate) fn admit_surface(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    is_last_chunk: bool,
    request_chain_matches: bool,
    continuation_state_ready: bool,
    high_speed_swap: bool,
) -> Result<()> {
    admit_surface_for_geometry(
        has_vision,
        sequence_start,
        chunk_start,
        chunk_rows,
        total_rows,
        is_last_chunk,
        super::multi_chunk_selected()?,
        request_chain_matches,
        continuation_state_ready,
        high_speed_swap,
        super::geometry()?,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn admit_surface_for_selection(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    is_last_chunk: bool,
    multi_chunk_selected: bool,
    request_chain_matches: bool,
    continuation_state_ready: bool,
    high_speed_swap: bool,
) -> Result<()> {
    admit_surface_for_geometry(
        has_vision,
        sequence_start,
        chunk_start,
        chunk_rows,
        total_rows,
        is_last_chunk,
        multi_chunk_selected,
        request_chain_matches,
        continuation_state_ready,
        high_speed_swap,
        super::LEGACY_GEOMETRY,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn admit_surface_for_geometry(
    has_vision: bool,
    sequence_start: usize,
    chunk_start: usize,
    chunk_rows: usize,
    total_rows: usize,
    is_last_chunk: bool,
    multi_chunk_selected: bool,
    request_chain_matches: bool,
    continuation_state_ready: bool,
    high_speed_swap: bool,
    geometry: super::ChunkGeometry,
) -> Result<()> {
    ensure!(
        !has_vision,
        "{SELECTOR} does not yet support vision prompts"
    );
    ensure!(
        request_chain_matches,
        "{SELECTOR} requires the exact cold prompt/sequence chain"
    );
    ensure!(
        continuation_state_ready,
        "{SELECTOR} requires live absolute paged-continuation metadata"
    );
    let single_chunk = sequence_start == 0
        && chunk_start == 0
        && chunk_rows == total_rows
        && chunk_rows <= geometry.rows
        && is_last_chunk;
    let chunk_end = chunk_start.checked_add(chunk_rows);
    let multi_chunk = multi_chunk_selected
        && geometry.chunked_prompt(total_rows)
        && geometry.chunk_admissible(chunk_start, chunk_rows, total_rows)
        && (chunk_rows.is_multiple_of(16) || (geometry.tail && is_last_chunk && chunk_rows >= 2))
        && sequence_start == chunk_start
        && chunk_end.is_some_and(|end| end <= total_rows)
        && is_last_chunk == (chunk_end == Some(total_rows));
    ensure!(
        single_chunk || multi_chunk,
        "{SELECTOR} requires one cold chunk or the selected cold aligned 8K/2K surface"
    );
    ensure!(
        !high_speed_swap,
        "{SELECTOR} is incompatible with high-speed-swap"
    );
    Ok(())
}

pub(crate) fn request_chain_matches(
    prompt_tokens: &[u32],
    sequence_tokens: &[u32],
    sequence_start: usize,
    sequence_prompt_len: usize,
    chunk_start: usize,
) -> bool {
    sequence_start == chunk_start
        && sequence_tokens.len() == chunk_start
        && ((chunk_start == 0 && sequence_prompt_len == 0)
            || (chunk_start > 0 && sequence_prompt_len == prompt_tokens.len()))
        && prompt_tokens.get(..chunk_start) == Some(sequence_tokens)
}

pub(crate) fn continuation_state_ready(
    start: usize,
    total: usize,
    bs: usize,
    bt_len: usize,
    meta: Option<(usize, usize, bool)>,
) -> bool {
    continuation_state_ready_for_marconi(start, total, bs, bt_len, 0, meta)
}

/// Continuation metadata check permitting Marconi-matched prefix blocks in `meta`.
pub(crate) fn continuation_state_ready_for_marconi(
    start: usize,
    total: usize,
    bs: usize,
    bt_len: usize,
    skip_to: usize,
    meta: Option<(usize, usize, bool)>,
) -> bool {
    if bs == 0 || total == 0 {
        return false;
    }
    if start == 0 {
        return bt_len == 0 && meta.is_none();
    }
    if !start.is_multiple_of(bs) || !skip_to.is_multiple_of(bs) {
        return false;
    }
    let (prior, needed) = (start / bs, total.div_ceil(bs));
    let Some((cap, uploaded, owns)) = meta else {
        return false;
    };
    let ok = if skip_to > 0 {
        (prior..=needed).contains(&bt_len) && uploaded == bt_len
    } else {
        bt_len == prior && uploaded == prior
    };
    owns && ok && cap >= needed
}

pub(crate) fn admit_request(
    config: &ModelConfig,
    rows: usize,
    start: usize,
    moe_selected: bool,
    exact_hyper: bool,
    exact_qkv16: bool,
    exact_o16: bool,
) -> Result<()> {
    core32_selected()?;
    if !selected()? {
        return Ok(());
    }
    ensure!(
        moe_selected && exact_hyper && exact_qkv16 && exact_o16,
        "{SELECTOR} requires exact MoE, HC, QKV16, and O16 prefill"
    );
    let tp = config.tp_world_size.max(1);
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2560
            && config.num_attention_heads * tp == 24
            && config.num_key_value_heads * config.kv_tp_size() == 2
            && config.head_dim == 256,
        "{SELECTOR} requires canonical Flash-Next attention geometry"
    );
    let min_rows = if super::geometry_or_legacy().tail {
        2
    } else {
        TILE_ROWS
    };
    ensure!(
        rows >= min_rows
            && rows <= super::max_chunk_rows()
            && super::exact_chunk_window_admitted(rows, start)?,
        "{SELECTOR} requires an admitted 16..=max-chunk-row prefill window"
    );
    for name in [
        "ATLAS_QWEN4_QSA_PREFILL_GEMM",
        "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
        "ATLAS_QWEN4_PREFILL_SSM_GEMM",
        "ATLAS_PAGED_DECODE_SPLITK",
    ] {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => {}
            Ok(value) if value == "0" => {}
            Ok(value) => bail!("{SELECTOR} rejects {name}={value:?}"),
            Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
        }
    }
    Ok(())
}

pub(crate) fn plan(rows: usize, start: usize, max_blocks: usize, arena: usize) -> Result<Plan> {
    plan_for_tile(rows, start, max_blocks, arena, TILE_ROWS)
}

pub(crate) fn plan_for_tile(
    rows: usize,
    start: usize,
    max_blocks: usize,
    arena: usize,
    tile_rows: usize,
) -> Result<Plan> {
    ensure!(
        super::exact_chunk_window_admitted(rows, start)?,
        "attention16 requires an admitted prefill window"
    );
    ensure!(
        tile_rows == TILE_ROWS || tile_rows == TILE32_ROWS,
        "attention tile must be 16 or 32 rows"
    );
    let min_rows = if super::geometry_or_legacy().tail {
        2
    } else {
        tile_rows
    };
    ensure!(
        rows >= min_rows && rows <= super::max_chunk_rows(),
        "attention16 row range is invalid"
    );
    ensure!(
        max_blocks > 0,
        "attention16 requires a complete block table"
    );
    let block_table_words = tile_rows
        .checked_mul(max_blocks)
        .ok_or_else(|| anyhow::anyhow!("attention16 block-table overflow"))?;
    let block_table_bytes = block_table_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("attention16 block-table byte overflow"))?;
    let seq_lens_offset = block_table_bytes.next_multiple_of(8);
    let scratch_bytes = seq_lens_offset
        .checked_add(tile_rows * std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("attention16 metadata overflow"))?;
    ensure!(
        scratch_bytes <= arena,
        "attention16 metadata exceeds scratch arena"
    );
    Ok(Plan {
        full_tiles: rows / tile_rows,
        tail_rows: rows % tile_rows,
        block_table_words,
        seq_lens_offset,
        scratch_bytes,
    })
}

#[cfg(test)]
#[path = "attn16_tests.rs"]
mod tests;

#[cfg(test)]
mod marconi_fast_cont_tests {
    use super::*;

    #[test]
    fn parse_marconi_fast_cont_defaults_on_and_honors_disable_values() {
        for v in [None, Some(""), Some("1"), Some("on"), Some("true")] {
            assert!(parse_marconi_fast_cont(v), "{v:?}");
        }
        for v in [Some("0"), Some("off"), Some("false"), Some(" OFF ")] {
            assert!(!parse_marconi_fast_cont(v), "{v:?}");
        }
    }

    #[test]
    fn marconi_skip_lineage_admits_suffix_and_rejects_mismatches() {
        let g = super::super::ChunkGeometry {
            rows: 16_384,
            total: 1_048_576,
            tail: true,
        };
        let p: Vec<u32> = (0..17_603).collect();
        let skipped = || {
            let mut s = Qwen4PrefillLineage::Unset;
            let t = begin_fast_lineage_for_geometry(g, &mut s, &p, 0, 16_384, true).unwrap();
            advance_marconi_skip_for_geometry(g, &mut s, t, &p, 0, 16_384).unwrap();
            s
        };
        let (mut bad_off, mut bad_id, mut ok) = (skipped(), skipped(), skipped());
        assert!(begin_fast_lineage_for_geometry(g, &mut bad_off, &p, 8192, 1219, true).is_none());
        let mut p2 = p.clone();
        p2[17_000] ^= 1;
        assert!(begin_fast_lineage_for_geometry(g, &mut bad_id, &p2, 16_384, 1219, true).is_none());
        let mut bad_end = skipped();
        let te = begin_fast_lineage_for_geometry(g, &mut bad_end, &p, 16_384, 1219, true).unwrap();
        let err = advance_marconi_skip_for_geometry(g, &mut bad_end, te, &p, 16_384, 17_603);
        assert!(err.is_err());
        let t1 = begin_fast_lineage_for_geometry(g, &mut ok, &p, 16_384, 1219, true).unwrap();
        commit_fast_lineage_for_geometry(g, &mut ok, t1, &p, 16_384, 17_603).unwrap();
        let ready = |bt, up| {
            let meta = Some((2267, up, true));
            continuation_state_ready_for_marconi(16_384, 36_259, 16, bt, 32_768, meta)
        };
        assert!(ready(2048, 2048) && !ready(2048, 1024) && !ready(1023, 1023));
    }
}
