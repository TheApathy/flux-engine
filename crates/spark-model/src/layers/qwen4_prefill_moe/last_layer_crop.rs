// SPDX-License-Identifier: AGPL-3.0-only

//! Last-layer row crop for Flash-Next prefill
//! (`ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1`, default off).
//!
//! A prefill chunk's final-layer output is read at one row only: the last
//! chunk's row `proc_count - 1` (`finalize_last`: final HC mix, lm_head).
//! Earlier chunks hand the next one nothing but the KV/QSA caches and the
//! recurrent state. Flash-Next's final layer is a full-attention layer, so:
//!
//! - [`Crop::Dead`] (no row of this chunk is read): skips `q_proj` and Q-side
//!   attention prep on the full-row flash path (populating only `0..rows` KV
//!   cache, QSA indexer compressed keys, and E4M3 K/V shadows), and skips the
//!   attention core, output gate, `o_proj`, core injection, and the whole MoE
//!   sublayer.
//! - [`Crop::Tail`] (the chunk's trailing rows are read): KV cache, QSA indexer
//!   compressed keys, and E4M3 K/V shadows always populate every row `0..rows`.
//!   When static-scale QSA E4M3 output (`ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT`) is
//!   active, `o_proj` uses a static tensor scale (`1 / min(s_v)`) with zero
//!   cross-row dynamic amax dependence, so Q-side `q_proj`, Q-prep, QSA indexer
//!   Q + score/top-k + attention core, `o_proj`, and attention HC injection also
//!   crop to the trailing `attn_rows` rows (aligned so `skip_rows` is a multiple
//!   of `lcm(16, 10) = 80`, preserving identical 10-row QSA tile boundaries and
//!   16-row cuBLASLt alignment; opt out with
//!   `ATLAS_QWEN4_PREFILL_LAST_LAYER_ATTN_CROP=0`). When `o_proj` uses a dynamic
//!   cross-row amax, attention runs on every row unchanged. The MoE sublayer
//!   prepares only the last `hc_rows` rows and runs the FFN and its injection on
//!   the last `moe_rows`. HC_FUSED is the only preparation at >= 64 rows, so the
//!   tail prepares at least [`HC_ROWS`] rows with the full chunk's per-stream
//!   GEMM order ([`crate::layers::qwen4_hyper::prefill_fused::ReferenceRowsGuard`]);
//!   router, top-k, the compact W4A8 experts, the shared expert, the combine
//!   and the injection are row-local kernels.
//!
//! The rows read are the last chunk's row `proc_count - 1` (lm_head) and, with
//! an installed MTP proposer, the prompt's last `K` final-layer rows, which
//! the last-K capture (`impl_b3::mtp_lastk_capture_chunk`) copies from every
//! chunk's trailing `min(K, rows)` rows into its ring: a chunk followed by at
//! least `K` prompt rows is overwritten there, so it is dead; otherwise its
//! trailing `K - rows_after` rows are kept ([`rows_read`]).
//!
//! Byte-identical by construction for every value read downstream. Refused
//! (the full layer runs; logged once per reason) whenever anything reads
//! other rows of the final layer's output: the quality-gate dump, the hidden
//! dump, DFlash capture, ATLAS_PROFILE, the ctx-hidden dump, expert-major
//! super-chunks, or a last-K capacity above [`MAX_TAIL_ROWS`].
//! ATLAS_NEMO_DUMP reads only the last row and stays allowed (it is the
//! identity probe).
//!
//! `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP_TRIGGER=<path>` (optional): crop only
//! while that file exists, so one load can serve an A/B/A.

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::qwen4_hyper::pending::{self, PendingInject};
use crate::layers::{FfnComponent, Qwen4HyperConnection};

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP";
pub(crate) const TRIGGER: &str = "ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP_TRIGGER";
/// Opt-out toggle for Layer-47 `Crop::Tail` full attention crop (default on
/// when static-scale `ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT` is active; `0`/`off`/`false`
/// reverts to full-chunk attention on `Crop::Tail`).
pub(crate) const ATTN_CROP_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_LAST_LAYER_ATTN_CROP";

/// Minimum tail rows the MoE-side HC preparation covers: HC_FUSED's minimum
/// (`prepare_prefill_exact_staged` takes another route below 64 rows).
pub(crate) const HC_ROWS: usize = 64;
/// Minimum tail rows the FFN and its injection cover (the compact MoE needs >= 2).
pub(crate) const MOE_ROWS: usize = 16;
/// Largest tail (a last-K capacity above it refuses).
pub(crate) const MAX_TAIL_ROWS: usize = 1024;
/// QSA prefill tile height (`qwen4_qsa::prefill_sparse::TILE_ROWS`).
pub(crate) const QSA_TILE_ROWS: usize = 10;
/// Alignment of skipped leading rows on `Crop::Tail` attention crop:
/// `lcm(16, QSA_TILE_ROWS) = 80`, preserving both exact 10-row QSA online-softmax tile
/// boundaries (`plan_fused_sort_desc` / `qkpv_e4m3_pipe`) and 16-row cuBLASLt
/// FP8 alignment for `q_proj` and `o_proj`.
pub(crate) const TAIL_ATTN_SKIP_ALIGN: usize = QSA_TILE_ROWS * 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Crop {
    /// No row of this chunk is read: everything after the KV writes is dead.
    Dead { rows_after: usize },
    /// The MoE sublayer runs on the chunk's last `moe_rows` rows only, which
    /// cover every row read (`lastk` is the MTP last-K capacity, 0 without a
    /// proposer).
    Tail {
        hc_rows: usize,
        moe_rows: usize,
        lastk: usize,
        rows_after: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    Off,
    Refused(&'static str),
    Engage(Crop),
}

/// What the layer loop knows about the chunk and its readers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Request {
    /// This layer is the model's last one and it is full attention.
    pub final_attention_layer: bool,
    pub decode_path: bool,
    pub rows: usize,
    pub last_chunk: bool,
    /// Prompt rows after this chunk.
    pub rows_after: usize,
    /// MTP last-K capture capacity when a proposer is installed, else 0.
    pub lastk: usize,
    /// The first reader of other rows of the final layer's output, if any.
    pub reader: Option<&'static str>,
    /// The tail preparation reproduces the full one (HC_FUSED level 2).
    pub tail_prepare_exact: bool,
}

/// `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`: absent/0 off, 1 on; anything else
/// fails the first prefill.
pub(crate) fn selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// [`selected`] and, with [`TRIGGER`] set, its file exists now.
pub(crate) fn armed() -> Result<bool> {
    static TRIGGER_PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    if !selected()? {
        return Ok(false);
    }
    let trigger = TRIGGER_PATH.get_or_init(|| {
        std::env::var(TRIGGER)
            .ok()
            .filter(|t| !t.is_empty())
            .map(std::path::PathBuf::from)
    });
    Ok(trigger.as_ref().is_none_or(|t| t.exists()))
}

fn parse(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{SELECTOR} must be absent, 0 or 1; got {other:?}")),
    }
}

fn parse_attn_crop(value: Option<&str>) -> Result<bool, String> {
    let Some(raw) = value else {
        return Ok(true);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{ATTN_CROP_SELECTOR} must be absent, 1, on, true, 0, off, or false; got {raw:?}"
        )),
    }
}

/// Whether `ATLAS_QWEN4_PREFILL_LAST_LAYER_ATTN_CROP` is enabled (read once per
/// process; default on, opt out with `0`, `off`, or `false`).
pub(crate) fn attn_crop_selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| parse_attn_crop(std::env::var(ATTN_CROP_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Query-side tail slice for Layer-47 `Crop::Tail` attention crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TailAttnPlan {
    /// Leading chunk rows skipped by Q-side `q_proj`, Q-prep, QSA indexer Q / core,
    /// `o_proj`, and attention HC injection (`skip_rows % 80 == 0`).
    pub skip_rows: usize,
    /// Trailing chunk rows executed through Q-side attention + `o_proj`
    /// (`skip_rows + attn_rows == chunk_rows`, `attn_rows >= hc_rows`, multiple of 16).
    pub attn_rows: usize,
    /// Absolute first query position (`chunk_start + skip_rows`).
    pub first_pos: usize,
    /// Trailing rows required by `finish_tail`'s MoE HC preparation.
    pub hc_rows: usize,
}

/// Attention-sublayer execution plan for Layer 47 under [`Crop`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttnCropPlan {
    /// Run full-chunk attention unchanged.
    Full,
    /// `Crop::Dead`: populate full `0..rows` KV cache, QSA indexer K cache, and
    /// E4M3 K/V shadows while skipping `q_proj`, Q-side prep, core, `o_proj`, and injection.
    DeadKvOnly,
    /// `Crop::Tail`: populate full `0..rows` KV / QSA-K / E4M3-KV state, then run
    /// Q-side attention + `o_proj` + HC injection on only the trailing `attn_rows` rows.
    Tail(TailAttnPlan),
}

/// Pure inputs governing Layer-47 attention cropping inside `prefill_fast_attn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttnCropRequest {
    /// Opt-out toggle (`ATLAS_QWEN4_PREFILL_LAST_LAYER_ATTN_CROP`, default on).
    pub attn_crop_enabled: bool,
    /// Active last-layer crop on this layer, if any.
    pub crop: Option<Crop>,
    /// Total rows in this prefill chunk (`num_tokens`).
    pub chunk_rows: usize,
    /// Absolute start position of this prefill chunk (`seq_len_start`).
    pub chunk_start: usize,
    /// Flash prefill attention core is enabled.
    pub flash: bool,
    /// Full-row attention prep is enabled (`ATLAS_QWEN4_PREFILL_ATTN_FULLROW=1`).
    pub fullrow: bool,
    /// Full-chunk static-scale `ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT` is eligible and
    /// active (`o_proj` has zero cross-row dynamic absmax dependence).
    pub static_qsa_e4m3_out_active: bool,
    /// Canonical BF16 unscaled FP8 fused prep (`rotary_dim == 64`, `nq == 24`,
    /// `nkv == 2`, shared-activation tensor-scale FP8 QKV) is active.
    pub fp8_prep_fused_active: bool,
    /// `finish_tail` preconditions hold (`!expert_major_mixer_slice && hyper_selected && hc_fused_level == 2`).
    pub finish_tail_ready: bool,
    /// QSA sparse prefill core is active for this chunk.
    pub qsa_sparse_active: bool,
}

/// Decide the Layer-47 attention crop mode and exact tile-aligned tail slice.
#[must_use]
pub(crate) const fn plan_attn_crop(req: AttnCropRequest) -> AttnCropPlan {
    if !req.flash || !req.fullrow {
        return AttnCropPlan::Full;
    }
    match req.crop {
        None => AttnCropPlan::Full,
        Some(Crop::Dead { .. }) => AttnCropPlan::DeadKvOnly,
        Some(Crop::Tail { hc_rows, .. }) => {
            if !req.attn_crop_enabled
                || !req.static_qsa_e4m3_out_active
                || !req.fp8_prep_fused_active
                || !req.finish_tail_ready
                || !req.qsa_sparse_active
            {
                return AttnCropPlan::Full;
            }
            if hc_rows < HC_ROWS
                || !req.chunk_rows.is_multiple_of(16)
                || req.chunk_rows < 2 * hc_rows
            {
                return AttnCropPlan::Full;
            }
            let raw_skip = req.chunk_rows - hc_rows;
            let skip_rows = (raw_skip / TAIL_ATTN_SKIP_ALIGN) * TAIL_ATTN_SKIP_ALIGN;
            if skip_rows == 0 {
                return AttnCropPlan::Full;
            }
            let attn_rows = req.chunk_rows - skip_rows;
            let Some(first_pos) = req.chunk_start.checked_add(skip_rows) else {
                return AttnCropPlan::Full;
            };
            AttnCropPlan::Tail(TailAttnPlan {
                skip_rows,
                attn_rows,
                first_pos,
                hc_rows,
            })
        }
    }
}

/// Trailing final-layer rows of this chunk that anything reads: lm_head's
/// last row on the last chunk, and the rows that end up in the last-K ring.
pub(crate) fn rows_read(r: &Request) -> usize {
    let lastk = r.lastk.saturating_sub(r.rows_after).min(r.rows);
    usize::from(r.last_chunk).max(lastk)
}

/// `(hc_rows, moe_rows)` for a tail that must keep `need` rows. The MoE rows
/// are copied from the end of the HC staging to its front, so they either
/// coincide with it (`hc_rows == moe_rows`) or do not overlap it.
pub(crate) fn tail_rows(need: usize) -> Option<(usize, usize)> {
    match need {
        0 => None,
        1..=16 => Some((HC_ROWS, MOE_ROWS)),
        17..=32 => Some((HC_ROWS, 32)),
        _ if need <= MAX_TAIL_ROWS => {
            let rows = need.next_multiple_of(16).max(HC_ROWS);
            Some((rows, rows))
        }
        _ => None,
    }
}

pub(crate) fn decide(selected: bool, r: Request) -> Decision {
    if !selected || !r.final_attention_layer || r.decode_path {
        return Decision::Off;
    }
    if let Some(reader) = r.reader {
        return Decision::Refused(reader);
    }
    if r.lastk > MAX_TAIL_ROWS {
        return Decision::Refused("mtp_lastk_capacity");
    }
    let need = rows_read(&r);
    let staged = tail_rows(need).map_or(HC_ROWS, |(hc_rows, _)| hc_rows);
    // The fast attention path runs 16-row multiples; the tail rows must not
    // overlap the front of norm_output where the MoE reads its input.
    if !r.rows.is_multiple_of(16) || r.rows < 2 * staged {
        return Decision::Off;
    }
    let Some((hc_rows, moe_rows)) = tail_rows(need) else {
        return Decision::Engage(Crop::Dead {
            rows_after: r.rows_after,
        });
    };
    if !r.tail_prepare_exact {
        return Decision::Refused("hc_fused_level");
    }
    Decision::Engage(Crop::Tail {
        hc_rows,
        moe_rows,
        lastk: r.lastk,
        rows_after: r.rows_after,
    })
}

/// Log a refusal once per reason per process.
pub(crate) fn log_refusal(reason: &'static str, rows: usize, start: usize) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(
            "QWEN4_PREFILL_LAST_LAYER_CROP_REFUSED selector={SELECTOR} reason={reason} rows={rows} start={start} (full final layer)"
        );
    }
}

/// Full-chunk provenance scoped around `Qwen4QsaIndexer::prefill_sparse_attention`
/// while running a Layer-47 `Crop::Tail` cropped attention slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TailAttnCoreContext {
    /// Original uncropped chunk's start position (`seq_len_start`), governing
    /// `V8Layout::Shared` validity: when `chunk_start == 0`, this layer's
    /// full-chunk KV prep just populated the shared `V8` shadow plane for all
    /// `0..chunk_rows` positions on the same stream.
    pub chunk_start: usize,
    /// Original uncropped chunk's row count (`num_tokens`), governing
    /// `E4M3_OUT_MIN_ROWS` eligibility and `iq` projection `ref_m` parity.
    pub chunk_rows: usize,
}

thread_local! {
    static ACTIVE: Cell<Option<Crop>> = const { Cell::new(None) };
    static ENGAGED: Cell<bool> = const { Cell::new(false) };
    static TAIL_ATTN_CORE: Cell<Option<TailAttnCoreContext>> = const { Cell::new(None) };
    static TAIL_ATTN_CROPPED_ROWS: Cell<Option<usize>> = const { Cell::new(None) };
    static DEAD_Q_SKIPPED: Cell<bool> = const { Cell::new(false) };
}

/// The crop the final layer of the running chunk may apply.
pub(crate) fn active() -> Option<Crop> {
    ACTIVE.with(Cell::get)
}

/// Active Layer-47 `Crop::Tail` full-chunk provenance for `prefill_sparse_run`.
pub(crate) fn active_tail_attn_core() -> Option<TailAttnCoreContext> {
    TAIL_ATTN_CORE.with(Cell::get)
}

/// Scoped [`active_tail_attn_core`] guard around a cropped QSA sparse attention call.
pub(crate) struct TailAttnCoreGuard(Option<TailAttnCoreContext>);

impl TailAttnCoreGuard {
    #[must_use]
    pub(crate) fn enter(ctx: TailAttnCoreContext) -> Self {
        Self(TAIL_ATTN_CORE.with(|c| c.replace(Some(ctx))))
    }
}

impl Drop for TailAttnCoreGuard {
    fn drop(&mut self) {
        TAIL_ATTN_CORE.with(|c| c.set(self.0));
    }
}

/// Record that `prefill_fast_attn` cropped Layer-47 attention to `attn_rows` tail rows.
pub(crate) fn note_tail_attn_cropped(attn_rows: usize) {
    TAIL_ATTN_CROPPED_ROWS.with(|c| c.set(Some(attn_rows)));
}

/// Record that `prefill_fast_attn` skipped `q_proj` and Q-side prep on `Crop::Dead`.
pub(crate) fn note_dead_q_skipped() {
    DEAD_Q_SKIPPED.with(|c| c.set(true));
}

/// Scoped [`active`] crop around the final layer's prefill call.
pub(crate) struct CropGuard(Option<Crop>);

impl CropGuard {
    pub(crate) fn enter(crop: Crop) -> Self {
        TAIL_ATTN_CROPPED_ROWS.with(|c| c.set(None));
        DEAD_Q_SKIPPED.with(|c| c.set(false));
        Self(ACTIVE.with(|c| c.replace(Some(crop))))
    }
}

impl Drop for CropGuard {
    fn drop(&mut self) {
        ACTIVE.with(|c| c.set(self.0));
        TAIL_ATTN_CROPPED_ROWS.with(|c| c.set(None));
        DEAD_Q_SKIPPED.with(|c| c.set(false));
    }
}

/// Receipt: the final layer of this chunk was cropped.
pub(crate) fn note_engaged(crop: Crop, rows: usize, start: usize, attention_skipped: bool) {
    ENGAGED.with(|c| c.set(true));
    let tail_attn_rows = TAIL_ATTN_CROPPED_ROWS.with(|c| c.replace(None));
    let dead_q_skipped = DEAD_Q_SKIPPED.with(|c| c.replace(false));
    match crop {
        Crop::Dead { rows_after } => tracing::info!(
            "QWEN4_PREFILL_LAST_LAYER_CROP_ENGAGED selector={SELECTOR} mode=dead rows={rows} start={start} rows_after={rows_after} q_proj_skipped={dead_q_skipped} attention_core_skipped={attention_skipped} moe_skipped=true"
        ),
        Crop::Tail {
            hc_rows,
            moe_rows,
            lastk,
            rows_after,
        } => {
            let attn_cropped = tail_attn_rows.is_some();
            let attn_rows = tail_attn_rows.unwrap_or(rows);
            tracing::info!(
                "QWEN4_PREFILL_LAST_LAYER_CROP_ENGAGED selector={SELECTOR} mode=tail rows={rows} start={start} hc_rows={hc_rows} moe_rows={moe_rows} rows_kept={moe_rows} lastk={lastk} rows_after={rows_after} hc_ref_rows={rows} attn_cropped={attn_cropped} attn_rows={attn_rows}"
            );
        }
    }
}

/// Whether the chunk's final layer was cropped since the last call (consumed
/// by the per-chunk readers that must not see cropped rows).
pub(crate) fn take_engaged() -> bool {
    TAIL_ATTN_CROPPED_ROWS.with(|c| c.set(None));
    DEAD_Q_SKIPPED.with(|c| c.set(false));
    ENGAGED.with(|c| c.replace(false))
}

/// Narrow a deferred attention-core [`PendingInject`] covering either the full
/// chunk (`rows`) or a trailing attention-cropped suffix (`hc_rows <= p.rows < rows`)
/// down to the final `hc_rows` rows expected by `super::hyper::finish_tail`.
#[must_use]
fn slice_pending_inject_for_tail(
    p: PendingInject,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    hc_rows: usize,
    row_bytes: usize,
    core_bytes: usize,
) -> Option<PendingInject> {
    if p.combine.is_some() || p.rows < hc_rows || p.rows > rows {
        return None;
    }
    let attn_skip = rows - p.rows;
    if p.hyper != hidden.offset(attn_skip * row_bytes)
        || p.residual != residual.offset(attn_skip * row_bytes)
    {
        return None;
    }
    let skip = rows - hc_rows;
    let delta = p.rows - hc_rows;
    Some(PendingInject {
        hyper: hidden.offset(skip * row_bytes),
        core: p.core.offset(delta * core_bytes),
        residual: residual.offset(skip * row_bytes),
        rows: hc_rows,
        combine: None,
    })
}

/// [`Crop::Tail`] MoE sublayer of the final layer over `rows` chunk rows:
/// `hc_rows` prepared, the last `moe_rows` of them through the FFN. Returns
/// false, having run the full sublayer, when the HC route cannot be cropped
/// exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_tail(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    crop: Crop,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    let Crop::Tail {
        hc_rows, moe_rows, ..
    } = crop
    else {
        super::finish(mlp, ffn, hidden, residual, rows, ctx, stream)?;
        return Ok(false);
    };
    if super::expert_major_mixer_slice()
        || !super::hyper_selected()?
        || crate::layers::qwen4_hyper::prefill_fused::level() != 2
        || hc_rows < HC_ROWS
        || rows < 2 * hc_rows
    {
        super::finish(mlp, ffn, hidden, residual, rows, ctx, stream)?;
        return Ok(false);
    }
    let row_bytes = ctx.config.residual_width() * 2;
    let core_bytes = ctx.config.hidden_size * 2;
    ensure!(
        rows * row_bytes <= ctx.buffers.sizes().hidden_states
            && rows * row_bytes <= ctx.buffers.sizes().residual,
        "{SELECTOR}: chunk rows exceed the hidden/residual arena"
    );
    let skip = rows - hc_rows;
    let hidden_tail = hidden.offset(skip * row_bytes);
    let residual_tail = residual.offset(skip * row_bytes);
    // The attention core's deferred injection (MHC_FUSED=inject) covers either
    // every row (full attention) or a trailing `attn_rows >= hc_rows` slice
    // (cropped tail attention). Owe only the tail's `hc_rows` rows to the tail
    // preparation, which fuses it as the full-row preparation would; anything
    // else is applied now, exactly as the full-row preparation applies an
    // injection it cannot fuse.
    if let Some(p) = pending::take() {
        if let Some(sliced) =
            slice_pending_inject_for_tail(p, hidden, residual, rows, hc_rows, row_bytes, core_bytes)
        {
            pending::defer(sliced)?;
        } else {
            pending::apply(&p, ctx.gpu, stream)?;
        }
    }
    super::hyper::finish_tail(
        mlp,
        ffn,
        hidden_tail,
        residual_tail,
        hc_rows,
        moe_rows,
        rows,
        ctx,
        stream,
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAIL: Crop = Crop::Tail {
        hc_rows: HC_ROWS,
        moe_rows: MOE_ROWS,
        lastk: 0,
        rows_after: 0,
    };

    fn tail_request(rows: usize) -> Request {
        Request {
            final_attention_layer: true,
            decode_path: false,
            rows,
            last_chunk: true,
            rows_after: 0,
            lastk: 0,
            reader: None,
            tail_prepare_exact: true,
        }
    }

    fn earlier(rows: usize, rows_after: usize) -> Request {
        Request {
            last_chunk: false,
            rows_after,
            ..tail_request(rows)
        }
    }

    #[test]
    fn parses_selector() {
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("")), Ok(false));
        assert_eq!(parse(Some("0")), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert!(parse(Some("yes")).is_err());
        assert!(parse(Some("2")).is_err());
    }

    #[test]
    fn last_chunk_crops_the_tail_and_earlier_chunks_are_dead() {
        assert_eq!(decide(true, tail_request(8192)), Decision::Engage(TAIL));
        assert_eq!(
            decide(true, earlier(16384, 8192)),
            Decision::Engage(Crop::Dead { rows_after: 8192 })
        );
    }

    #[test]
    fn off_unless_selected_on_the_final_attention_layer_of_a_batch_chunk() {
        assert_eq!(decide(false, tail_request(8192)), Decision::Off);
        let not_final = Request {
            final_attention_layer: false,
            ..tail_request(8192)
        };
        assert_eq!(decide(true, not_final), Decision::Off);
        let decode = Request {
            decode_path: true,
            ..tail_request(8192)
        };
        assert_eq!(decide(true, decode), Decision::Off);
    }

    #[test]
    fn short_or_unaligned_chunks_run_the_full_layer() {
        assert_eq!(decide(true, tail_request(2 * HC_ROWS - 16)), Decision::Off);
        assert_eq!(decide(true, tail_request(8191)), Decision::Off);
        assert_eq!(
            decide(true, tail_request(2 * HC_ROWS)),
            Decision::Engage(TAIL)
        );
    }

    #[test]
    fn any_row_reader_refuses_both_modes() {
        let tail = Request {
            reader: Some("qgate_dump"),
            ..tail_request(8192)
        };
        assert_eq!(decide(true, tail), Decision::Refused("qgate_dump"));
        let dead = Request {
            last_chunk: false,
            rows_after: 8192,
            ..tail
        };
        assert_eq!(decide(true, dead), Decision::Refused("qgate_dump"));
    }

    #[test]
    fn tail_needs_the_level2_preparation_but_dead_does_not() {
        let tail = Request {
            tail_prepare_exact: false,
            ..tail_request(8192)
        };
        assert_eq!(decide(true, tail), Decision::Refused("hc_fused_level"));
        let dead = Request {
            last_chunk: false,
            rows_after: 8192,
            ..tail
        };
        assert_eq!(
            decide(true, dead),
            Decision::Engage(Crop::Dead { rows_after: 8192 })
        );
    }

    #[test]
    fn rows_read_covers_lm_head_and_the_last_k_ring() {
        let with_k = |r: Request| Request { lastk: 64, ..r };
        // lm_head only.
        assert_eq!(rows_read(&tail_request(8192)), 1);
        assert_eq!(rows_read(&earlier(8192, 1)), 0);
        // The ring takes the prompt's last 64 rows.
        assert_eq!(rows_read(&with_k(tail_request(8192))), 64);
        assert_eq!(rows_read(&with_k(tail_request(20))), 20);
        assert_eq!(rows_read(&with_k(earlier(16384, 20))), 44);
        assert_eq!(rows_read(&with_k(earlier(16384, 64))), 0);
        assert_eq!(rows_read(&with_k(earlier(16384, 16384))), 0);
    }

    #[test]
    fn tail_rows_never_overlap_their_staging() {
        assert_eq!(tail_rows(0), None);
        assert_eq!(tail_rows(1), Some((HC_ROWS, MOE_ROWS)));
        assert_eq!(tail_rows(20), Some((HC_ROWS, 32)));
        assert_eq!(tail_rows(44), Some((HC_ROWS, HC_ROWS)));
        assert_eq!(tail_rows(64), Some((64, 64)));
        assert_eq!(tail_rows(100), Some((112, 112)));
        assert_eq!(tail_rows(MAX_TAIL_ROWS + 1), None);
        for need in 1..=MAX_TAIL_ROWS {
            let (hc, moe) = tail_rows(need).expect("tail");
            let skip = hc - moe;
            assert!(
                moe >= need && moe.is_multiple_of(16) && hc >= HC_ROWS,
                "need {need}"
            );
            assert!(
                skip == 0 || skip >= moe,
                "need {need}: {moe} MoE rows overlap {hc} staged"
            );
        }
    }

    #[test]
    fn an_installed_proposer_keeps_the_last_k_rows_instead_of_refusing() {
        use crate::model::impl_b3::mtp_lastk_capture_runs;
        // Qwen4's default ATLAS_MTP_LASTK_PREFILL=64 without a proposer reads
        // nothing; with one, the tail keeps the prompt's last 64 rows.
        let lastk = |proposer| {
            if mtp_lastk_capture_runs(64, proposer) {
                64
            } else {
                0
            }
        };
        let no_proposer = Request {
            lastk: lastk(false),
            ..tail_request(8192)
        };
        assert_eq!(decide(true, no_proposer), Decision::Engage(TAIL));
        let proposer = Request {
            lastk: lastk(true),
            ..tail_request(8192)
        };
        assert_eq!(
            decide(true, proposer),
            Decision::Engage(Crop::Tail {
                hc_rows: 64,
                moe_rows: 64,
                lastk: 64,
                rows_after: 0
            })
        );
        // A 16384-row chunk followed by a 20-row final chunk keeps 44 rows (as 64).
        let penultimate = Request {
            lastk: 64,
            ..earlier(16384, 20)
        };
        assert_eq!(
            decide(true, penultimate),
            Decision::Engage(Crop::Tail {
                hc_rows: 64,
                moe_rows: 64,
                lastk: 64,
                rows_after: 20
            })
        );
        // Followed by at least 64 rows, the ring is refilled later: dead.
        let dead = Request {
            lastk: 64,
            ..earlier(16384, 16404)
        };
        assert_eq!(
            decide(true, dead),
            Decision::Engage(Crop::Dead { rows_after: 16404 })
        );
        // The 20-row final chunk itself is too short to crop.
        assert_eq!(
            decide(
                true,
                Request {
                    lastk: 64,
                    ..tail_request(20)
                }
            ),
            Decision::Off
        );
    }

    #[test]
    fn a_larger_tail_needs_twice_its_staging_and_a_bounded_capacity() {
        let k256 = |rows| Request {
            lastk: 256,
            ..tail_request(rows)
        };
        assert_eq!(
            decide(true, k256(512)),
            Decision::Engage(Crop::Tail {
                hc_rows: 256,
                moe_rows: 256,
                lastk: 256,
                rows_after: 0
            })
        );
        assert_eq!(decide(true, k256(496)), Decision::Off);
        let huge = Request {
            lastk: MAX_TAIL_ROWS + 1,
            ..tail_request(16384)
        };
        assert_eq!(decide(true, huge), Decision::Refused("mtp_lastk_capacity"));
    }

    #[test]
    fn guard_scopes_the_active_crop() {
        let dead = Crop::Dead { rows_after: 1 };
        assert_eq!(active(), None);
        {
            let _outer = CropGuard::enter(TAIL);
            assert_eq!(active(), Some(TAIL));
            {
                let _inner = CropGuard::enter(dead);
                assert_eq!(active(), Some(dead));
            }
            assert_eq!(active(), Some(TAIL));
        }
        assert_eq!(active(), None);
    }

    #[test]
    fn engagement_is_consumed_once() {
        assert!(!take_engaged());
        ENGAGED.with(|c| c.set(true));
        assert!(take_engaged());
        assert!(!take_engaged());
    }

    fn attn_crop_req(crop: Option<Crop>, chunk_rows: usize, chunk_start: usize) -> AttnCropRequest {
        AttnCropRequest {
            attn_crop_enabled: true,
            crop,
            chunk_rows,
            chunk_start,
            flash: true,
            fullrow: true,
            static_qsa_e4m3_out_active: true,
            fp8_prep_fused_active: true,
            finish_tail_ready: true,
            qsa_sparse_active: true,
        }
    }

    #[test]
    fn parses_attn_crop_selector_defaults_on_and_rejects_invalid() {
        assert_eq!(parse_attn_crop(None), Ok(true));
        assert_eq!(parse_attn_crop(Some("")), Ok(true));
        assert_eq!(parse_attn_crop(Some("1")), Ok(true));
        assert_eq!(parse_attn_crop(Some("on")), Ok(true));
        assert_eq!(parse_attn_crop(Some("true")), Ok(true));
        assert_eq!(parse_attn_crop(Some(" TRUE ")), Ok(true));
        assert_eq!(parse_attn_crop(Some("0")), Ok(false));
        assert_eq!(parse_attn_crop(Some("off")), Ok(false));
        assert_eq!(parse_attn_crop(Some("false")), Ok(false));
        assert_eq!(parse_attn_crop(Some(" OFF ")), Ok(false));
        assert!(parse_attn_crop(Some("2")).is_err());
        assert!(parse_attn_crop(Some("yes")).is_err());
    }

    #[test]
    fn dead_crop_skips_q_even_with_dynamic_amax_or_tail_opt_out() {
        let dead = Some(Crop::Dead { rows_after: 16_384 });
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                attn_crop_enabled: false,
                static_qsa_e4m3_out_active: false,
                fp8_prep_fused_active: false,
                finish_tail_ready: false,
                qsa_sparse_active: false,
                ..attn_crop_req(dead, 16_384, 0)
            }),
            AttnCropPlan::DeadKvOnly
        );
        // Non-flash or non-fullrow cannot skip Q (paged core still reads Q).
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                flash: false,
                ..attn_crop_req(dead, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                fullrow: false,
                ..attn_crop_req(dead, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
    }

    #[test]
    fn tail_attn_crop_gates_on_static_qsa_e4m3_out_and_opt_out_toggle() {
        let tail64 = Some(Crop::Tail {
            hc_rows: 64,
            moe_rows: 64,
            lastk: 64,
            rows_after: 0,
        });
        // Cold 16K with static QSA E4M3 out: exact 64-row tail (16320 % 80 == 0).
        assert_eq!(
            plan_attn_crop(attn_crop_req(tail64, 16_384, 0)),
            AttnCropPlan::Tail(TailAttnPlan {
                skip_rows: 16_320,
                attn_rows: 64,
                first_pos: 16_320,
                hc_rows: 64,
            })
        );
        // Dynamic cross-row amax (`ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT=0` or Mode::Qk continuation):
        // must refuse and run full attention.
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                static_qsa_e4m3_out_active: false,
                ..attn_crop_req(tail64, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        // Opt-out toggle (`ATLAS_QWEN4_PREFILL_LAST_LAYER_ATTN_CROP=0`): must refuse.
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                attn_crop_enabled: false,
                ..attn_crop_req(tail64, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        // Missing fused FP8 prep, finish_tail readiness, QSA sparse, or active crop: must refuse.
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                fp8_prep_fused_active: false,
                ..attn_crop_req(tail64, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                finish_tail_ready: false,
                ..attn_crop_req(tail64, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        assert_eq!(
            plan_attn_crop(AttnCropRequest {
                qsa_sparse_active: false,
                ..attn_crop_req(tail64, 16_384, 0)
            }),
            AttnCropPlan::Full
        );
        assert_eq!(
            plan_attn_crop(attn_crop_req(None, 16_384, 0)),
            AttnCropPlan::Full
        );
    }

    #[test]
    fn tail_attn_crop_slice_arithmetic_preserves_qsa_tile_and_cublaslt_alignment() {
        let cases = [
            // (chunk_start, chunk_rows, hc_rows, expected_skip, expected_attn)
            (0usize, 16_384usize, 64usize, 16_320usize, 64usize),
            (0, 8_192, 64, 8_080, 112),
            (0, 4_096, 64, 4_000, 96),
            (0, 32_768, 64, 32_640, 128),
            (16_384, 16_384, 64, 16_320, 64),
            (0, 16_384, 112, 16_240, 144),
            (0, 16_384, 256, 16_080, 304),
            (0, 16_384, MAX_TAIL_ROWS, 15_360, MAX_TAIL_ROWS),
        ];
        for (start, rows, hc, want_skip, want_attn) in cases {
            let crop = Some(Crop::Tail {
                hc_rows: hc,
                moe_rows: hc,
                lastk: hc,
                rows_after: 0,
            });
            let plan = match plan_attn_crop(attn_crop_req(crop, rows, start)) {
                AttnCropPlan::Tail(p) => p,
                other => panic!("start={start} rows={rows} hc={hc}: expected Tail, got {other:?}"),
            };
            assert_eq!(
                plan,
                TailAttnPlan {
                    skip_rows: want_skip,
                    attn_rows: want_attn,
                    first_pos: start + want_skip,
                    hc_rows: hc,
                },
                "start={start} rows={rows} hc={hc}"
            );
            assert_eq!(plan.skip_rows + plan.attn_rows, rows);
            assert!(plan.skip_rows.is_multiple_of(QSA_TILE_ROWS));
            assert!(plan.skip_rows.is_multiple_of(16));
            assert!(plan.attn_rows.is_multiple_of(16));
            assert!(plan.attn_rows >= hc);
        }
        // Too-short chunks (raw_skip < TAIL_ATTN_SKIP_ALIGN or rows < 2 * hc_rows) stay Full.
        assert_eq!(
            plan_attn_crop(attn_crop_req(Some(TAIL), 128, 0)),
            AttnCropPlan::Full
        );
        assert_eq!(
            plan_attn_crop(attn_crop_req(Some(TAIL), 144, 0)),
            AttnCropPlan::Tail(TailAttnPlan {
                skip_rows: 80,
                attn_rows: 64,
                first_pos: 80,
                hc_rows: 64,
            })
        );
        // Unaligned chunk rows stay Full.
        assert_eq!(
            plan_attn_crop(attn_crop_req(Some(TAIL), 16_380, 0)),
            AttnCropPlan::Full
        );
    }

    #[test]
    fn tail_attn_core_guard_scopes_and_restores_provenance() {
        let outer_ctx = TailAttnCoreContext {
            chunk_start: 0,
            chunk_rows: 16_384,
        };
        let inner_ctx = TailAttnCoreContext {
            chunk_start: 16_384,
            chunk_rows: 8_192,
        };
        assert_eq!(active_tail_attn_core(), None);
        {
            let _outer = TailAttnCoreGuard::enter(outer_ctx);
            assert_eq!(active_tail_attn_core(), Some(outer_ctx));
            {
                let _inner = TailAttnCoreGuard::enter(inner_ctx);
                assert_eq!(active_tail_attn_core(), Some(inner_ctx));
            }
            assert_eq!(active_tail_attn_core(), Some(outer_ctx));
        }
        assert_eq!(active_tail_attn_core(), None);
    }

    #[test]
    fn slice_pending_inject_supports_full_and_cropped_tail_offsets() {
        let hidden = DevicePtr(0x1000_0000);
        let residual = DevicePtr(0x2000_0000);
        let norm_out = DevicePtr(0x3000_0000);
        let row_bytes = 4 * 2560 * 2;
        let core_bytes = 2560 * 2;

        // 1. Full 16384-row pending inject -> narrowed to trailing 64 rows.
        let full = PendingInject {
            hyper: hidden,
            core: norm_out,
            residual,
            rows: 16_384,
            combine: None,
        };
        assert_eq!(
            slice_pending_inject_for_tail(
                full, hidden, residual, 16_384, 64, row_bytes, core_bytes
            ),
            Some(PendingInject {
                hyper: hidden.offset(16_320 * row_bytes),
                core: norm_out.offset(16_320 * core_bytes),
                residual: residual.offset(16_320 * row_bytes),
                rows: 64,
                combine: None,
            })
        );

        // 2. Cropped 16K exact 64-row tail pending inject (delta = 0).
        let exact_tail = PendingInject {
            hyper: hidden.offset(16_320 * row_bytes),
            core: norm_out.offset(16_320 * core_bytes),
            residual: residual.offset(16_320 * row_bytes),
            rows: 64,
            combine: None,
        };
        assert_eq!(
            slice_pending_inject_for_tail(
                exact_tail, hidden, residual, 16_384, 64, row_bytes, core_bytes
            ),
            Some(exact_tail)
        );

        // 3. Cropped 8K 112-row tail pending inject (skip=8080, delta=48 -> 8128).
        let wide_tail = PendingInject {
            hyper: hidden.offset(8_080 * row_bytes),
            core: norm_out.offset(8_080 * core_bytes),
            residual: residual.offset(8_080 * row_bytes),
            rows: 112,
            combine: None,
        };
        assert_eq!(
            slice_pending_inject_for_tail(
                wide_tail, hidden, residual, 8_192, 64, row_bytes, core_bytes
            ),
            Some(PendingInject {
                hyper: hidden.offset(8_128 * row_bytes),
                core: norm_out.offset(8_128 * core_bytes),
                residual: residual.offset(8_128 * row_bytes),
                rows: 64,
                combine: None,
            })
        );

        // 4. Mismatched base pointers or too-short pending rows refuse cleanly.
        let wrong_hyper = PendingInject {
            hyper: hidden.offset(row_bytes),
            ..exact_tail
        };
        assert_eq!(
            slice_pending_inject_for_tail(
                wrong_hyper,
                hidden,
                residual,
                16_384,
                64,
                row_bytes,
                core_bytes
            ),
            None
        );
        let too_short = PendingInject {
            rows: 32,
            ..exact_tail
        };
        assert_eq!(
            slice_pending_inject_for_tail(
                too_short, hidden, residual, 16_384, 64, row_bytes, core_bytes
            ),
            None
        );
    }
}
