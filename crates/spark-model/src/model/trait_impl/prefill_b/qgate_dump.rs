// SPDX-License-Identifier: AGPL-3.0-only

//! Quality-gate diagnostic (default off, zero cost when unset).
//!
//! `ATLAS_QGATE_DUMP=<dir>` writes, for sampled rows of EVERY prefill chunk,
//! the lm_head input the engine computes for that row (Qwen4 final
//! hyperconnection mix, or the final RMSNorm), as BF16. Scoring those rows
//! offline against the BF16 lm_head gives teacher-forced NLL / top-1 / KL of
//! the PREFILL path over the whole prompt, not just its last token.
//!
//! Sampled rows: absolute position `p % ATLAS_QGATE_STRIDE == 0` (default 16)
//! or `p` within the last `ATLAS_QGATE_TAIL` (default 256) prompt positions.
//! `ATLAS_QGATE_REQS=1,3` or `13-21` (optional) limits the dump to those
//! request ordinals (1-based, counted per process, inclusive ranges), so
//! other requests of the same process run undisturbed.
//! `ATLAS_QGATE_TRIGGER=<path>` (optional) dumps only while that file exists,
//! so a timing session can switch the dump on for a later gate collect.
//! One file per chunk:
//! `<dir>/qgate_req{N}_start{S}_rows{R}_total{T}_n{K}_h{H}.bin`
//! = K little-endian u32 absolute positions, then K x H BF16 rows.
//!
//! The residual stream is only read. The per-row final mix stages into the
//! same scratch `finalize_last` uses for the last row (`norm_output`, and the
//! last row's `residual` slot), so the chunk's state is never touched.

use anyhow::Result;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::super::types::TransformerModel;

struct QgateConfig {
    dir: String,
    stride: usize,
    tail: usize,
    reqs: Option<Vec<usize>>,
    trigger: Option<std::path::PathBuf>,
}

fn qgate_config() -> Option<&'static QgateConfig> {
    static CFG: OnceLock<Option<QgateConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let dir = std::env::var("ATLAS_QGATE_DUMP").ok().filter(|d| !d.is_empty())?;
        let num = |key: &str, default: usize| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        let stride = num("ATLAS_QGATE_STRIDE", 16).max(1);
        let tail = num("ATLAS_QGATE_TAIL", 256);
        // Ordinals and inclusive ranges: "13-21" or "13,14,15".
        let reqs = std::env::var("ATLAS_QGATE_REQS")
            .ok()
            .map(|v| {
                v.split(',')
                    .flat_map(|x| {
                        let x = x.trim();
                        let (a, b) = x.split_once('-').unwrap_or((x, x));
                        match (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                            (Ok(a), Ok(b)) if a <= b => a..=b,
                            _ => std::ops::RangeInclusive::new(1, 0),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());
        let trigger = std::env::var("ATLAS_QGATE_TRIGGER")
            .ok()
            .filter(|t| !t.is_empty())
            .map(std::path::PathBuf::from);
        tracing::warn!(
            "ATLAS_QGATE_DUMP={dir} stride={stride} tail={tail} reqs={reqs:?} trigger={trigger:?} (diagnostic, slow)"
        );
        Some(QgateConfig { dir, stride, tail, reqs, trigger })
    })
    .as_ref()
}

/// Whether a chunk starting now may be dumped (the dump is configured and its
/// trigger file, if any, exists). `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP` refuses
/// while this holds; a chunk that was cropped anyway is never dumped.
pub(super) fn qgate_dump_armed() -> bool {
    qgate_config().is_some_and(|cfg| cfg.trigger.as_ref().is_none_or(|t| t.exists()))
}

impl TransformerModel {
    pub(super) fn prefill_b_qgate_dump(
        &self,
        tokens: &[u32],
        effective_seq_len_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cfg) = qgate_config() else {
            return Ok(());
        };
        // A new request starts at position 0; number requests so several
        // prompts can share one dump directory.
        static REQ: AtomicUsize = AtomicUsize::new(0);
        if effective_seq_len_start == 0 {
            REQ.fetch_add(1, Ordering::Relaxed);
        }
        let req = REQ.load(Ordering::Relaxed);
        if cfg.reqs.as_ref().is_some_and(|reqs| !reqs.contains(&req))
            || cfg.trigger.as_ref().is_some_and(|t| !t.exists())
        {
            return Ok(());
        }
        // The trigger appeared after the layer loop chose to crop its final
        // layer: this chunk's rows other than the last are not the model's.
        if crate::layers::qwen4_prefill_moe::last_layer_crop::take_engaged() {
            tracing::warn!(
                "ATLAS_QGATE_DUMP: skipped req{req} start{effective_seq_len_start}: the final layer was cropped ({})",
                crate::layers::qwen4_prefill_moe::last_layer_crop::SELECTOR
            );
            return Ok(());
        }

        let total = tokens.len();
        let rows: Vec<usize> = (0..proc_count)
            .filter(|&r| {
                let p = effective_seq_len_start + r;
                p.is_multiple_of(cfg.stride) || p + cfg.tail >= total
            })
            .collect();
        if rows.is_empty() || proc_count == 0 {
            return Ok(());
        }

        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let rw = self.config.residual_width();
        let hidden = self.buffers.hidden_states();
        let stage = self.buffers.residual().offset((proc_count - 1) * rw * fp32);
        let normed = self.buffers.norm_output();

        let mut out = Vec::with_capacity(rows.len() * (4 + h * 2));
        for &r in &rows {
            out.extend_from_slice(&((effective_seq_len_start + r) as u32).to_le_bytes());
        }
        let data_off = out.len();
        out.resize(data_off + rows.len() * h * 2, 0);
        for (k, &r) in rows.iter().enumerate() {
            let row = hidden.offset(r * rw * fp32);
            let final_input = self
                .qwen4_final_hidden_nvfp4(row, stage, stream)?
                .unwrap_or(row);
            self.final_norm_rows(final_input, normed, 1, stream)?;
            let src = normed;
            self.gpu.synchronize(stream)?;
            let at = data_off + k * h * 2;
            self.gpu.copy_d2h(src, &mut out[at..at + h * 2])?;
        }
        std::fs::create_dir_all(&cfg.dir)?;
        let name = format!(
            "qgate_req{req}_start{effective_seq_len_start}_rows{proc_count}_total{total}_n{}_h{h}.bin",
            rows.len()
        );
        std::fs::write(std::path::Path::new(&cfg.dir).join(name), &out)?;
        Ok(())
    }
}
