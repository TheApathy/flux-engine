// SPDX-License-Identifier: AGPL-3.0-only

//! Experimental serial-core/grouped-MoE prefill. No numerical qualification implied.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

use super::{FfnComponent, Qwen4HyperConnection};
use crate::layer::ForwardContext;
use plan::{Limits, Plan};

pub(crate) mod attn16;
mod geometry_prepared;
mod hyper;
pub(crate) mod last_layer_crop;
mod plan;
pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_BATCH";
pub(crate) const HYPER_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_EXACT";
pub(crate) const HYPER_CHECK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_HC_CHECK";
const PLE_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_BATCH";
const PLE_COMPOSITION_SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_COMBINE";
pub(crate) const EXACT_QKV16_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_QKV16";
pub(crate) const EXACT_O16_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_O16";
pub(crate) const MULTI_CHUNK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MULTI_CHUNK";
pub(crate) const MULTI_CHUNK_ROWS: usize = 2048;
pub(crate) const MULTI_TOTAL_ROWS: usize = 8192;
/// Wide-chunk geometry (default off): `ATLAS_QWEN4_PREFILL_WIDE=<W>` raises
/// the per-chunk row cap of the whole fast family from 2048 to W and lets the
/// multi-chunk lineage cover any prompt up to [`WIDE_TOTAL_ROWS`] as
/// ceil(L/W) W-aligned chunks (the last one may be shorter). Serve it with
/// `--max-prefill-tokens W`.
pub(crate) const WIDE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_WIDE";
pub(crate) const WIDE_TOTAL_ROWS: usize = 1_048_576;
const EXACT_K16_SELECTOR: &str = "ATLAS_QWEN4_K16_EXACT";

/// Chunk geometry of the fast family: the per-chunk row cap and the largest
/// prompt the multi-chunk lineage covers. `tail` admits a shorter final chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ChunkGeometry {
    pub rows: usize,
    pub total: usize,
    pub tail: bool,
}

pub(crate) const LEGACY_GEOMETRY: ChunkGeometry = ChunkGeometry {
    rows: MULTI_CHUNK_ROWS,
    total: MULTI_TOTAL_ROWS,
    tail: false,
};

impl ChunkGeometry {
    /// Prompts the multi-chunk lineage owns (longer than one chunk).
    pub(crate) fn lineage_prompt(&self, prompt_rows: usize) -> bool {
        if self.tail {
            prompt_rows > self.rows && prompt_rows <= self.total
        } else {
            prompt_rows == self.total
        }
    }

    /// Row count of a `width`-row prepared chunk starting at `start`: `width`,
    /// or the prompt remainder when the prompt ends inside it. `None` unless
    /// `width` is a prepared width ([`Self::is_prepared`]) and `start` is a
    /// multiple of it. The legacy geometry prepares only its own width.
    pub(crate) fn chunk_rows_at(
        &self,
        start: usize,
        prompt_rows: usize,
        width: usize,
    ) -> Option<usize> {
        if !self.is_prepared(width) || !start.is_multiple_of(width) || start >= prompt_rows {
            return None;
        }
        Some(width.min(prompt_rows - start))
    }
}

pub(crate) fn parse_wide(value: Option<&str>) -> Result<Option<usize>, &'static str> {
    match value {
        None | Some("0") => Ok(None),
        Some("2048") => Ok(Some(2048)),
        Some("4096") => Ok(Some(4096)),
        Some("8192") => Ok(Some(8192)),
        Some("16384") => Ok(Some(16384)),
        Some("32768") => Ok(Some(32768)),
        _ => Err("must be absent, 0, 2048, 4096, 8192, 16384, or 32768"),
    }
}

/// The selected wide chunk width. Geometry, not a family member: suppression
/// does not change it, so it reads the environment directly (once).
pub(crate) fn wide_rows() -> Result<Option<usize>> {
    static WIDE: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    WIDE.get_or_init(|| {
        let raw = std::env::var(WIDE_SELECTOR).ok();
        parse_wide(raw.as_deref()).map_err(|error| format!("{WIDE_SELECTOR}: {error}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

pub(crate) fn geometry() -> Result<ChunkGeometry> {
    if let Some(rows) = EXPERT_MAJOR_SLICE.with(std::cell::Cell::get) {
        return Ok(ChunkGeometry {
            rows,
            total: WIDE_TOTAL_ROWS,
            tail: true,
        });
    }
    if let Some(rows) = CHUNK_ROWS_OVERRIDE.with(std::cell::Cell::get) {
        return Ok(ChunkGeometry {
            rows,
            total: WIDE_TOTAL_ROWS,
            tail: true,
        });
    }
    Ok(match wide_rows()? {
        Some(rows) => ChunkGeometry {
            rows,
            total: WIDE_TOTAL_ROWS,
            tail: true,
        },
        None => LEGACY_GEOMETRY,
    })
}

/// Pure-function view of [`geometry`]; a malformed selector fails at
/// [`admit_request`], which every fast route passes first.
pub(crate) fn geometry_or_legacy() -> ChunkGeometry {
    geometry().unwrap_or(LEGACY_GEOMETRY)
}

/// Prepared chunk widths a bounded interleave chunk may use (ascending).
/// Empty unless the wide geometry and the multi-chunk lineage are both
/// selected; the expert-major super-chunk mode keeps W-only chunks (its
/// mixer slices narrow the geometry per slice, which bounded chunks would
/// interleave with). A malformed selector yields empty here and fails at
/// [`admit_request`].
pub(crate) fn interleave_widths() -> Vec<usize> {
    let wide = matches!(wide_rows(), Ok(Some(_)));
    if !wide || expert_major_selected() || !multi_chunk_selected().unwrap_or(false) {
        return Vec::new();
    }
    geometry_or_legacy().prepared_widths()
}

/// Largest row count one prefill chunk may hand the fast family in the active scope.
pub(crate) fn max_chunk_rows() -> usize {
    geometry_or_legacy().rows
}

/// Configured server-wide chunk ceiling (`ATLAS_QWEN4_PREFILL_WIDE` or `MULTI_CHUNK_ROWS`),
/// unaffected by transient per-chunk `CHUNK_ROWS_OVERRIDE` / `EXPERT_MAJOR_SLICE` guards.
pub(crate) fn configured_max_chunk_rows() -> usize {
    wide_rows()
        .ok()
        .flatten()
        .unwrap_or(MULTI_CHUNK_ROWS)
        .max(MULTI_CHUNK_ROWS)
}

thread_local! {
    static FAMILY_SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Mixer row width while an expert-major super-chunk runs its mixers slice by slice.
    static EXPERT_MAJOR_SLICE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Scoped chunk-width step-down while interleaved prefill runs 2048-row chunks on a wide server.
    static CHUNK_ROWS_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Scoped single-forward mixed prefill+decode MoE row window (`P` prefill + `D` decode).
    static MIXED_MOE_ROWS: std::cell::Cell<Option<MixedMoeRows>> = const { std::cell::Cell::new(None) };
}

/// Active single-forward mixed prefill+decode MoE row breakdown (`P` prefill
/// rows followed by `D` decode verify rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MixedMoeRows {
    pub prefill_rows: usize,
    pub decode_rows: usize,
}

impl MixedMoeRows {
    pub(crate) fn total_rows(self) -> Option<usize> {
        self.prefill_rows.checked_add(self.decode_rows)
    }
}

/// Scoped mixed prefill+decode MoE guard (sibling to [`ExpertMajorSliceGuard`]):
/// permits `rows = prefill_rows + decode_rows` up to `buffers.max_batch_tokens()`
/// while keeping prefill chunk-window admission checked against `prefill_rows`.
/// Restores prior thread-local state on drop, including early returns and panics.
pub(crate) struct MixedMoeRowsGuard(Option<MixedMoeRows>);

impl MixedMoeRowsGuard {
    pub(crate) fn enter(prefill_rows: usize, decode_rows: usize) -> Self {
        let next = (prefill_rows > 0 && decode_rows > 0).then_some(MixedMoeRows {
            prefill_rows,
            decode_rows,
        });
        Self(MIXED_MOE_ROWS.with(|c| c.replace(next)))
    }
}

impl Drop for MixedMoeRowsGuard {
    fn drop(&mut self) {
        MIXED_MOE_ROWS.with(|c| c.set(self.0));
    }
}

/// Current mixed prefill+decode MoE row window, if [`MixedMoeRowsGuard`] is active.
pub(crate) fn active_mixed_moe_rows() -> Option<MixedMoeRows> {
    MIXED_MOE_ROWS.with(std::cell::Cell::get)
}

/// `ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR=1` (default off; needs WIDE): each layer runs its
/// mixer (HC prepare, attention/GDN core, inject) per `ATLAS_QWEN4_PREFILL_MIXER_ROWS` slice of
/// the super-chunk, then ONE MoE sublayer over all of its rows, so the expert weights stream
/// once per super-chunk instead of once per slice. Row-local MoE => same math as WIDE=slice.
pub(crate) fn expert_major_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR")
            .ok()
            .as_deref()
            == Some("1")
    })
}

pub(crate) fn mixer_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_MIXER_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&r| r >= 2048 && r.is_multiple_of(2048))
            .unwrap_or(16384)
    })
}

/// True while a mixer slice runs: the MoE sublayer and its engagement receipt are deferred
/// to the super-chunk's single MoE pass.
pub(crate) fn expert_major_mixer_slice() -> bool {
    EXPERT_MAJOR_SLICE.with(std::cell::Cell::get).is_some()
}

/// Scoped mixer-slice mode: chunk geometry narrows to the slice width and MoE is deferred.
pub(crate) struct ExpertMajorSliceGuard(Option<usize>);

impl ExpertMajorSliceGuard {
    pub(crate) fn enter(slice_rows: usize) -> Self {
        Self(EXPERT_MAJOR_SLICE.with(|c| c.replace(Some(slice_rows))))
    }
}

impl Drop for ExpertMajorSliceGuard {
    fn drop(&mut self) {
        EXPERT_MAJOR_SLICE.with(|c| c.set(self.0));
    }
}

/// Scoped chunk-width override: narrows a wide (`>= 4096`) fast-prefill
/// configuration down to `MULTI_CHUNK_ROWS` (`2048`) while running a
/// roofline-stepped interleaved prefill chunk alongside active decode.
pub(crate) struct ChunkRowsOverrideGuard(Option<Option<usize>>);

impl ChunkRowsOverrideGuard {
    pub(crate) fn maybe_enter_interleaved_2k(
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
    ) -> Self {
        let narrow = wide_rows()
            .ok()
            .flatten()
            .is_some_and(|w| w > MULTI_CHUNK_ROWS)
            && chunk_start.is_multiple_of(MULTI_CHUNK_ROWS)
            && ((!is_last_chunk && chunk_len == MULTI_CHUNK_ROWS)
                || (is_last_chunk && chunk_start > 0 && chunk_len <= MULTI_CHUNK_ROWS));
        if narrow {
            Self(Some(
                CHUNK_ROWS_OVERRIDE.with(|c| c.replace(Some(MULTI_CHUNK_ROWS))),
            ))
        } else {
            Self(None)
        }
    }
}

impl Drop for ChunkRowsOverrideGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0 {
            CHUNK_ROWS_OVERRIDE.with(|c| c.set(prev));
        }
    }
}

/// True while [`attn16::admit_or_suppress`] has turned the whole fast-prefill
/// family (batched MoE prefill, HC/SSM exact, compact MoE, QKV16/O16, CORE16
/// and its dependents) off for the prefill running on this thread. Every family
/// selector then reads as absent, so the request takes the ordinary prefill
/// path instead of failing admission.
pub(crate) fn family_suppressed() -> bool {
    FAMILY_SUPPRESSED.with(std::cell::Cell::get)
}

fn set_family_suppressed(on: bool) -> bool {
    FAMILY_SUPPRESSED.with(|s| s.replace(on))
}

pub(crate) fn hyper_check_selected() -> Result<bool> {
    plan::parse_selector(env_value(HYPER_CHECK_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{HYPER_CHECK_SELECTOR}: {error}"))
}

pub(crate) fn hyper_selected() -> Result<bool> {
    plan::parse_selector(env_value(HYPER_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{HYPER_SELECTOR}: {error}"))
}

fn env_value(name: &str) -> Result<Option<String>> {
    // The PLE whole-prompt batch handles continuation chunks itself; only its
    // composition with batched MoE prefill (PLE_COMPOSITION_SELECTOR) is family-scoped.
    if family_suppressed() && name != PLE_SELECTOR {
        return Ok(None);
    }
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

pub(crate) fn selected() -> Result<bool> {
    plan::parse_selector(env_value(SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

pub(crate) fn exact_qkv16_selected() -> Result<bool> {
    plan::parse_selector(env_value(EXACT_QKV16_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{EXACT_QKV16_SELECTOR}: {error}"))
}

pub(crate) fn exact_o16_selected() -> Result<bool> {
    plan::parse_selector(env_value(EXACT_O16_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{EXACT_O16_SELECTOR}: {error}"))
}

pub(crate) fn multi_chunk_selected() -> Result<bool> {
    plan::parse_selector(env_value(MULTI_CHUNK_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{MULTI_CHUNK_SELECTOR}: {error}"))
}

fn multi_window(rows: usize, start: usize, end: usize, g: ChunkGeometry) -> bool {
    g.window_rows_admitted(rows, start) && end <= g.total
}

pub(crate) fn chunk_window_for_geometry(
    rows: usize,
    start: usize,
    multi_chunk: bool,
    g: ChunkGeometry,
) -> bool {
    let Some(end) = start.checked_add(rows) else {
        return false;
    };
    rows > 0 && (end <= g.rows || (multi_chunk && multi_window(rows, start, end, g)))
}

pub(crate) fn exact_chunk_window_for_geometry(
    rows: usize,
    start: usize,
    multi_chunk: bool,
    g: ChunkGeometry,
) -> bool {
    let Some(end) = start.checked_add(rows) else {
        return false;
    };
    rows > 0
        && ((start == 0 && end <= g.rows) || (multi_chunk && multi_window(rows, start, end, g)))
}

pub(crate) fn chunk_window_for_selection(rows: usize, start: usize, multi_chunk: bool) -> bool {
    let g = geometry_or_legacy();
    if let Some(mixed) = active_mixed_moe_rows() {
        return start == 0
            && mixed.total_rows() == Some(rows)
            && chunk_window_for_geometry(mixed.prefill_rows, 0, multi_chunk, g);
    }
    chunk_window_for_geometry(rows, start, multi_chunk, g)
}

pub(crate) fn exact_chunk_window_for_selection(
    rows: usize,
    start: usize,
    multi_chunk: bool,
) -> bool {
    exact_chunk_window_for_geometry(rows, start, multi_chunk, geometry_or_legacy())
}

pub(crate) fn chunk_window_admitted(rows: usize, start: usize) -> Result<bool> {
    Ok(chunk_window_for_selection(
        rows,
        start,
        multi_chunk_selected()?,
    ))
}

pub(crate) fn exact_chunk_window_admitted(rows: usize, start: usize) -> Result<bool> {
    Ok(exact_chunk_window_for_selection(
        rows,
        start,
        multi_chunk_selected()?,
    ))
}

/// Called before embedding, buffer writes, or recurrent-state changes.
pub(crate) fn admit_request(config: &ModelConfig, rows: usize, start: usize) -> Result<()> {
    let wide = wide_rows()?;
    ensure!(
        wide.is_none() || multi_chunk_selected()? || family_suppressed(),
        "{WIDE_SELECTOR} requires {MULTI_CHUNK_SELECTOR}=1"
    );
    // `qwen4_prefill_compact` is suppressed while `family_suppressed()` is true
    // (`request_active_from` returns false), so its admission check is skipped here.
    if !family_suppressed() {
        crate::layers::moe::qwen4_prefill_compact::admit_request(config, rows, start)?;
    }
    crate::layers::qwen3_ssm::qwen4_prefill_exact::admit_request(config, rows, start)?;
    let hyper = hyper_selected()?;
    ensure!(
        !hyper_check_selected()? || hyper,
        "{HYPER_CHECK_SELECTOR} requires {HYPER_SELECTOR}=1"
    );
    let selected = selected()?;
    let exact_qkv16 = exact_qkv16_selected()?;
    let exact_o16 = exact_o16_selected()?;
    let multi_chunk = multi_chunk_selected()?;
    let ple_composition = plan::parse_selector(env_value(PLE_COMPOSITION_SELECTOR)?.as_deref())
        .map_err(|error| anyhow::anyhow!("{PLE_COMPOSITION_SELECTOR}: {error}"))?;
    ensure!(
        !multi_chunk || attn16::selected()?,
        "{MULTI_CHUNK_SELECTOR}=1 requires {}=1",
        attn16::SELECTOR
    );
    attn16::admit_request(config, rows, start, selected, hyper, exact_qkv16, exact_o16)?;
    if !selected {
        ensure!(!hyper, "{HYPER_SELECTOR} requires {SELECTOR}=1");
        ensure!(!exact_qkv16, "{EXACT_QKV16_SELECTOR} requires {SELECTOR}=1");
        ensure!(!exact_o16, "{EXACT_O16_SELECTOR} requires {SELECTOR}=1");
        ensure!(
            !multi_chunk,
            "{MULTI_CHUNK_SELECTOR}=1 requires {SELECTOR}=1"
        );
        ensure!(
            !ple_composition,
            "{PLE_COMPOSITION_SELECTOR} requires {SELECTOR}=1"
        );
        return Ok(());
    }
    let pp = config.pp_world_size.max(1);
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2560
            && config.residual_width() == 10240
            && config.num_hidden_layers * pp == 48
            && config.num_attention_layers() * pp == 12
            && config.num_ssm_layers() * pp == 36
            && config.num_experts == 512
            && config.num_experts_per_tok == 10
            && config.moe_intermediate_size == 640
            && config.shared_expert_intermediate_size == 640
            && !config.use_fp32_residual()
            && (config.ep_world_size <= 1 || config.ep_world_size == config.tp_world_size)
            && config.tp_world_size <= 4,
        "{SELECTOR} requires the canonical BF16-residual Flash-Next geometry"
    );
    plan::validate_exact_qkv16_composition(
        env_value(EXACT_QKV16_SELECTOR)?.as_deref(),
        env_value(EXACT_K16_SELECTOR)?.as_deref(),
        selected,
        hyper,
        start,
        rows,
        multi_chunk,
    )
    .map_err(|error| anyhow::anyhow!("{EXACT_QKV16_SELECTOR}: {error}"))?;
    plan::validate_exact_o16_composition(env_value(EXACT_O16_SELECTOR)?.as_deref(), exact_qkv16)
        .map_err(|error| anyhow::anyhow!("{EXACT_O16_SELECTOR}: {error}"))?;
    ensure!(
        chunk_window_for_selection(rows, start, multi_chunk),
        "{SELECTOR} requires the initial window or an admitted 2K continuation chunk"
    );
    for name in [
        "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
        "ATLAS_QWEN4_SSM_PREFILL_BATCH",
        "ATLAS_QWEN4_SSM_PREFILL_ROWWISE",
        "ATLAS_QWEN4_SSM_PREFILL_FP32",
        "ATLAS_QWEN4_SSM_PREFILL_CONVSEQ",
        "ATLAS_QWEN4_HYPER_PREFILL_GEMM",
        "ATLAS_QWEN4_QSA_PREFILL_GEMM",
        "ATLAS_NVFP4_MOE_WORKLIST",
        "ATLAS_SSM_H_FP16",
    ] {
        plan::validate_conflict(env_value(name)?.as_deref(), false)
            .map_err(|error| anyhow::anyhow!("{SELECTOR}: {name}: {error}"))?;
    }
    plan::validate_conflict(env_value("ATLAS_QWEN4_SSM_PREFILL_TILE")?.as_deref(), true)
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: ATLAS_QWEN4_SSM_PREFILL_TILE: {error}"))?;
    plan::validate_ple_composition(
        env_value(PLE_SELECTOR)?.as_deref(),
        env_value(PLE_COMPOSITION_SELECTOR)?.as_deref(),
    )
    .map_err(|error| {
        anyhow::anyhow!("{SELECTOR}: {PLE_SELECTOR}+{PLE_COMPOSITION_SELECTOR}: {error}")
    })
}

fn layout(ctx: &ForwardContext, rows: usize, start: usize) -> Result<Plan> {
    let sizes = ctx.buffers.sizes();
    Plan::new(
        rows,
        start,
        multi_chunk_selected()?,
        ctx.config.hidden_size,
        ctx.config.residual_width(),
        Limits {
            capacity: ctx.buffers.max_batch_tokens(),
            qkv: sizes.qkv_output,
            norm: sizes.norm_output,
            hidden: sizes.hidden_states,
            residual: sizes.residual,
            output: sizes.moe_output,
        },
    )
    .map_err(anyhow::Error::msg)
}

/// Read-only admission for the unified MoE pass over `prefill_rows + decode_rows`
/// rows inside a single-forward mixed prefill+decode round.
pub(crate) fn validate_mixed_moe(
    ctx: &ForwardContext,
    prefill_rows: usize,
    decode_rows: usize,
) -> Result<()> {
    let total_rows = prefill_rows
        .checked_add(decode_rows)
        .context("mixed prefill+decode MoE row overflow")?;
    let _guard = MixedMoeRowsGuard::enter(prefill_rows, decode_rows);
    layout(ctx, total_rows, 0)?;
    crate::layers::moe::qwen4_prefill_compact::validate_context(ctx, total_rows)?;
    Ok(())
}

/// Layer admission must precede the first serial core call.
pub(crate) fn validate(ctx: &ForwardContext, rows: usize, start: usize) -> Result<()> {
    admit_request(ctx.config, rows, start)?;
    crate::layers::moe::qwen4_prefill_compact::validate_context(ctx, rows)?;
    ensure!(
        (ctx.config.ep_world_size <= 1 || ctx.config.ep_world_size == ctx.config.tp_world_size)
            && ctx.config.tp_world_size <= 4
            && !ctx.graph_capture
            && ctx.ddtree_parent_ids_dev.is_none()
            && ctx.tree_aware_attn.is_none()
            && ctx.ssm_multi_seq_ptr_table_override.is_none()
            && ctx.self_spec_sparse_draft.is_none()
            && ctx.ffn_defer.is_none(),
        "{SELECTOR} requires eager single-sequence prefill without verify overrides"
    );
    let meta = ctx
        .attn_metadata
        .context("MoE-only prefill requires paged metadata")?;
    ensure!(
        meta.num_seqs == 1
            && meta.positions != DevicePtr::NULL
            && meta.slot != DevicePtr::NULL
            && meta.seq_len != DevicePtr::NULL
            && meta.block_table != DevicePtr::NULL,
        "{SELECTOR} requires live single-sequence paged metadata"
    );
    ensure!(
        ctx.buffers.sizes().ssm_ba >= 2056,
        "MoE-only HC scratch is undersized"
    );
    layout(ctx, rows, start)?;
    Ok(())
}

pub(crate) fn validate_ffn(ffn: &FfnComponent) -> Result<()> {
    match ffn {
        FfnComponent::Moe(moe) => moe.validate_qwen4_prefill_moe(),
        _ => anyhow::bail!("{SELECTOR} requires grouped original-layout NVFP4 experts"),
    }
}

/// All serial core rows have finished before entering this function.
/// Keep both HC operations identical to decode; change only the FFN dispatch.
pub(crate) fn finish(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    if expert_major_mixer_slice() {
        return Ok(());
    }
    let p = layout(ctx, rows, 0)?;
    validate_ffn(ffn)?;
    ensure!(
        mlp.inject.is_some() && matches!(ffn, FfnComponent::Moe(_)),
        "MoE-only prefill requires an injecting MLP hyperconnection and routed MoE"
    );
    if hyper_selected()? {
        return hyper::finish(mlp, ffn, hidden, residual, rows, ctx, stream);
    }
    let staging = ctx.buffers.qkv_output().offset(p.staging_offset);
    for row in 0..rows {
        let (mixed, _) = mlp.prepare_decode_nvfp4(
            hidden.offset(row * p.row_bytes),
            residual.offset(row * p.row_bytes),
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        ctx.gpu.copy_d2d_async(
            mixed,
            staging.offset(row * p.core_bytes),
            p.core_bytes,
            stream,
        )?;
    }
    // prepare_decode overwrites norm_output row zero each time. Pack only
    // after every row has been preserved outside its QKV projection scratch.
    let input = ctx.buffers.norm_output();
    ctx.gpu
        .copy_d2d_async(staging, input, p.input_bytes, stream)?;
    ffn.forward_prefill(input, rows, ctx, stream)?;
    for row in 0..rows {
        mlp.inject_decode(
            hidden.offset(row * p.row_bytes),
            ctx.buffers.moe_output().offset(row * p.core_bytes),
            mlp.saved_inject(residual.offset(row * p.row_bytes)),
            ctx.gpu,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_ceiling_ignores_scoped_interleaved_override() {
        assert_eq!(parse_wide(Some("2048")), Ok(Some(2048)));
        assert_eq!(parse_wide(Some("4096")), Ok(Some(4096)));
        let base_ceiling = configured_max_chunk_rows();
        let prev = CHUNK_ROWS_OVERRIDE.with(|c| c.replace(Some(1024)));
        assert_eq!(max_chunk_rows(), 1024);
        assert_eq!(
            configured_max_chunk_rows(),
            base_ceiling,
            "persistent scratch ceiling must not shrink under scoped chunk override"
        );
        CHUNK_ROWS_OVERRIDE.with(|c| c.set(prev));
    }

    #[test]
    fn mixed_moe_rows_guard_restores_state_on_drop_early_return_and_panic() {
        assert_eq!(active_mixed_moe_rows(), None);
        let max_p = max_chunk_rows();

        {
            let _outer = MixedMoeRowsGuard::enter(max_p, 40);
            assert_eq!(
                active_mixed_moe_rows(),
                Some(MixedMoeRows {
                    prefill_rows: max_p,
                    decode_rows: 40,
                })
            );
            assert!(chunk_window_for_selection(max_p + 40, 0, false));
            // Unmatched row count or non-zero start stays fail-closed.
            assert!(!chunk_window_for_selection(max_p + 41, 0, false));
            assert!(!chunk_window_for_selection(max_p + 40, max_p, true));

            let early_return = || -> Result<(), &'static str> {
                let _inner = MixedMoeRowsGuard::enter(512, 128);
                assert_eq!(
                    active_mixed_moe_rows(),
                    Some(MixedMoeRows {
                        prefill_rows: 512,
                        decode_rows: 128,
                    })
                );
                Err("early exit")
            };
            assert!(early_return().is_err());
            assert_eq!(
                active_mixed_moe_rows(),
                Some(MixedMoeRows {
                    prefill_rows: max_p,
                    decode_rows: 40,
                })
            );
        }
        assert_eq!(active_mixed_moe_rows(), None);

        let panicked = std::panic::catch_unwind(|| {
            let _guard = MixedMoeRowsGuard::enter(max_p, 128);
            assert_eq!(
                active_mixed_moe_rows(),
                Some(MixedMoeRows {
                    prefill_rows: max_p,
                    decode_rows: 128,
                })
            );
            panic!("simulated panic inside mixed MoE");
        });
        assert!(panicked.is_err());
        assert_eq!(
            active_mixed_moe_rows(),
            None,
            "MixedMoeRowsGuard must restore state on unwind"
        );
    }
}
