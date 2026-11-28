// SPDX-License-Identifier: AGPL-3.0-only

//! Per-row QSA ownership for the native Qwen4 C2-C256 decode frame.
//!
//! `ATLAS_QWEN4_NATIVE_BATCH_QSA` (default on; `0` disables) requires native Qwen4
//! batching (on by default; `ATLAS_QWEN4_NATIVE_BATCH=0` disables) and admits
//! requests that can reach sparse QSA attention into the native frame. This
//! first increment is eager and correctness-first: every QSA row runs the
//! unchanged C1 QSA functions on a one-row view of the frame metadata,
//! serialized on the frame's stream, so the single-row side-cache scratch
//! (projected QK, pooled key, logits, selected tokens, sparse partials) is
//! reused exactly as C1 reuses it.
//!
//! The side index itself is already per sequence: raw-ring and
//! compressed-key rows are keyed by physical KV block, and the native frame
//! refuses any block shared between rows or with the dummy block.

use anyhow::{Result, ensure};
use spark_runtime::kv_cache::PagedKvCache;

use super::position_contract::valid_physical_query;
use super::{COMPRESS_RATIO, QsaCacheLayout, Qwen4QsaIndexer};
use crate::layer::AttnMetadataDev;

pub(crate) const NATIVE_BATCH_QSA_ENV: &str = "ATLAS_QWEN4_NATIVE_BATCH_QSA";

/// First pre-step sequence length whose query takes the sparse QSA core.
/// Mirrors the C1 route in `qwen3_attention/decode/attention_forward.rs`
/// (`qwen4_qsa_required && seq_len >= 2048`).
pub(crate) const SPARSE_START: usize = 2_048;

const POSITION_BYTES: usize = 4;
const SLOT_BYTES: usize = 8;
const SEQ_LEN_BYTES: usize = 4;
const BLOCK_ENTRY_BYTES: usize = 4;

/// Whether the native frame may carry QSA rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeQsaMode {
    /// Requests that can reach QSA stay on the serialized C1 path.
    Off,
    /// Each QSA row runs the C1 QSA functions inside the native frame.
    PerRow,
}

/// Attention core one native row takes in this decode step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeAttnRoute {
    /// No QSA side index: the request can never reach position 2048.
    Dense,
    /// QSA side index maintained; dense attention below position 2048.
    IndexedDense,
    /// QSA side index maintained; top-512 sparse attention.
    Sparse,
}

impl NativeAttnRoute {
    pub fn maintains_index(self) -> bool {
        !matches!(self, Self::Dense)
    }

    pub fn is_sparse(self) -> bool {
        matches!(self, Self::Sparse)
    }

    /// One-character receipt label: n(one), i(ndexed dense), s(parse).
    pub fn label(self) -> char {
        match self {
            Self::Dense => 'n',
            Self::IndexedDense => 'i',
            Self::Sparse => 's',
        }
    }
}

pub(crate) fn parse_native_batch_qsa_selector(
    raw: Option<&str>,
) -> std::result::Result<NativeQsaMode, &'static str> {
    match raw {
        None | Some("1") => Ok(NativeQsaMode::PerRow),
        Some("0") => Ok(NativeQsaMode::Off),
        Some(_) => Err("ATLAS_QWEN4_NATIVE_BATCH_QSA must be exact value 0 or 1"),
    }
}

/// QSA rows without the native frame are a configuration error, never a
/// silent no-op.
pub(crate) fn resolve_native_batch_qsa_mode(
    native_selected: bool,
    mode: NativeQsaMode,
) -> std::result::Result<NativeQsaMode, &'static str> {
    match (native_selected, mode) {
        (_, NativeQsaMode::Off) => Ok(NativeQsaMode::Off),
        (true, NativeQsaMode::PerRow) => Ok(NativeQsaMode::PerRow),
        (false, NativeQsaMode::PerRow) => {
            Err("ATLAS_QWEN4_NATIVE_BATCH_QSA=1 requires native Qwen4 batching")
        }
    }
}

/// Resolve `ATLAS_QWEN4_NATIVE_BATCH_QSA` under `det_batch`:
/// `DeterministicBatchPolicy::Enabled` forces [`NativeQsaMode::Off`] so QSA
/// rows always take the canonical serialized C1 path.
#[inline]
#[must_use]
pub(crate) const fn native_batch_qsa_mode_with_policy(
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
    env_mode: NativeQsaMode,
) -> NativeQsaMode {
    if det_batch.is_enabled() {
        NativeQsaMode::Off
    } else {
        env_mode
    }
}

pub(crate) fn native_batch_qsa_mode() -> Result<NativeQsaMode> {
    let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
    let raw = match std::env::var(NATIVE_BATCH_QSA_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{NATIVE_BATCH_QSA_ENV} must be valid UTF-8")
        }
    };
    let parsed = parse_native_batch_qsa_selector(raw.as_deref()).map_err(anyhow::Error::msg)?;
    Ok(native_batch_qsa_mode_with_policy(det_batch, parsed))
}

/// Route of one row from its immutable admission flag and pre-step length.
pub(crate) fn native_attn_route(
    mode: NativeQsaMode,
    qsa_required: bool,
    seq_len: usize,
) -> std::result::Result<NativeAttnRoute, &'static str> {
    let next = seq_len
        .checked_add(1)
        .ok_or("native Qwen4 QSA row sequence length overflow")?;
    match (qsa_required, mode) {
        (false, _) if next < SPARSE_START => Ok(NativeAttnRoute::Dense),
        // Admission keeps such a request below 2048 for its whole life; C1
        // would silently run dense attention here, the native frame refuses.
        (false, _) => Err("native Qwen4 row without QSA cannot reach the sparse-QSA boundary"),
        (true, NativeQsaMode::Off) => Err("native Qwen4 decode cannot maintain the QSA side index"),
        (true, NativeQsaMode::PerRow) => {
            u32::try_from(next).map_err(|_| "native Qwen4 QSA sequence length exceeds u32")?;
            Ok(if seq_len < SPARSE_START {
                NativeAttnRoute::IndexedDense
            } else {
                NativeAttnRoute::Sparse
            })
        }
    }
}

pub(crate) fn native_attn_routes(
    mode: NativeQsaMode,
    qsa_required: &[bool],
    seq_lens: &[usize],
) -> std::result::Result<Vec<NativeAttnRoute>, &'static str> {
    if qsa_required.len() != seq_lens.len() {
        return Err("native Qwen4 QSA routing needs one flag per active row");
    }
    qsa_required
        .iter()
        .zip(seq_lens)
        .map(|(&required, &seq_len)| native_attn_route(mode, required, seq_len))
        .collect()
}

/// Lengths handed to the dense-lane admission policies. QSA rows were
/// admitted by [`native_attn_route`], so the dense policies' 2048 horizon
/// applies only to rows that keep no side index.
pub(crate) fn dense_policy_seq_lens(
    routes: &[NativeAttnRoute],
    seq_lens: &[usize],
) -> std::result::Result<Vec<usize>, &'static str> {
    if routes.len() != seq_lens.len() {
        return Err("native Qwen4 QSA routes and sequence lengths differ");
    }
    Ok(routes
        .iter()
        .zip(seq_lens)
        .map(|(route, &seq_len)| if route.maintains_index() { 0 } else { seq_len })
        .collect())
}

/// Comma-separated receipt labels, row order.
pub(crate) fn route_labels(routes: &[NativeAttnRoute]) -> String {
    routes
        .iter()
        .map(|route| route.label().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowMetaOffsets {
    pub position: usize,
    pub slot: usize,
    pub seq_len: usize,
    pub block_table: usize,
}

/// Byte offsets of one row inside the batch metadata arrays written by
/// `upload_batch_metadata_at`: positions `[N]` u32, slots `[N]` i64,
/// seq_lens `[N]` i32 and block tables `[N * max_blocks]` i32.
pub(crate) fn row_metadata_offsets(
    row: usize,
    max_blocks_per_seq: usize,
) -> std::result::Result<RowMetaOffsets, &'static str> {
    let overflow = "native Qwen4 QSA row metadata offset overflow";
    Ok(RowMetaOffsets {
        position: row.checked_mul(POSITION_BYTES).ok_or(overflow)?,
        slot: row.checked_mul(SLOT_BYTES).ok_or(overflow)?,
        seq_len: row.checked_mul(SEQ_LEN_BYTES).ok_or(overflow)?,
        block_table: row
            .checked_mul(max_blocks_per_seq)
            .and_then(|entries| entries.checked_mul(BLOCK_ENTRY_BYTES))
            .ok_or(overflow)?,
    })
}

/// One-row view of the frame metadata, the shape every C1 QSA entry
/// requires (`num_seqs == 1`). Only pointers move; nothing is uploaded.
pub(crate) fn row_metadata_view(
    meta: AttnMetadataDev,
    row: usize,
    qsa_required: bool,
) -> Result<AttnMetadataDev> {
    ensure!(
        row < meta.num_seqs as usize,
        "native Qwen4 QSA row {row} is outside the {}-row frame metadata",
        meta.num_seqs
    );
    let offsets =
        row_metadata_offsets(row, meta.max_blocks_per_seq as usize).map_err(anyhow::Error::msg)?;
    Ok(AttnMetadataDev {
        qwen4_qsa_required: qsa_required,
        positions: meta.positions.offset(offsets.position),
        positions_h: meta.positions_h.offset(offsets.position),
        positions_w: meta.positions_w.offset(offsets.position),
        slot: meta.slot.offset(offsets.slot),
        seq_len: meta.seq_len.offset(offsets.seq_len),
        block_table: meta.block_table.offset(offsets.block_table),
        num_seqs: 1,
        ..meta
    })
}

impl Qwen4QsaIndexer {
    /// Read-only admission of a native frame's QSA rows: no allocation, no
    /// launch. A layer whose side cache is not initialized yet is checked
    /// against the geometry its lazy allocator will use.
    pub(crate) fn preflight_native_rows(
        &self,
        kv_cache: &PagedKvCache,
        routes: &[NativeAttnRoute],
        seq_lens: &[usize],
    ) -> Result<()> {
        ensure!(
            routes.len() == seq_lens.len(),
            "native Qwen4 QSA preflight needs one length per routed row"
        );
        let max_groups = match self.cache.get() {
            Some(cache) => {
                ensure!(
                    cache.num_main_blocks == kv_cache.num_blocks()
                        && cache.block_size == kv_cache.block_size(),
                    "QSA main-cache geometry changed after initialization"
                );
                cache.max_compressed_groups
            }
            None => {
                QsaCacheLayout::checked(kv_cache.num_blocks(), kv_cache.block_size())?
                    .max_compressed_groups
            }
        };
        ensure!(
            !self.has_precomputed_verify_rows(),
            "native Qwen4 QSA frame found pending precomputed verify rows"
        );
        if super::graph_safe_enabled() {
            ensure!(
                self.score_device.0 != 0 && self.select_expand_device.0 != 0,
                "graph-safe QSA needs the device-metadata score/select kernels"
            );
        }
        for (row, (route, &seq_len)) in routes.iter().zip(seq_lens).enumerate() {
            if !route.maintains_index() {
                continue;
            }
            let next = seq_len
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("native Qwen4 QSA row {row} length overflow"))?;
            ensure!(
                valid_physical_query(seq_len, next),
                "native Qwen4 QSA row {row} has no valid physical query"
            );
            ensure!(
                next / COMPRESS_RATIO <= max_groups,
                "native Qwen4 QSA row {row} needs {} groups but the side cache holds {max_groups}",
                next / COMPRESS_RATIO
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::DevicePtr;

    use super::*;

    #[test]
    fn selector_is_exact_default_on_and_needs_the_native_frame() {
        assert_eq!(
            parse_native_batch_qsa_selector(None),
            Ok(NativeQsaMode::PerRow)
        );
        assert_eq!(
            parse_native_batch_qsa_selector(Some("0")),
            Ok(NativeQsaMode::Off)
        );
        assert_eq!(
            parse_native_batch_qsa_selector(Some("1")),
            Ok(NativeQsaMode::PerRow)
        );
        for hostile in ["", "true", "01", "2", " 1", "1 "] {
            assert!(
                parse_native_batch_qsa_selector(Some(hostile)).is_err(),
                "{hostile:?}"
            );
        }
        assert_eq!(
            resolve_native_batch_qsa_mode(false, NativeQsaMode::Off),
            Ok(NativeQsaMode::Off)
        );
        assert_eq!(
            resolve_native_batch_qsa_mode(true, NativeQsaMode::PerRow),
            Ok(NativeQsaMode::PerRow)
        );
        assert!(resolve_native_batch_qsa_mode(false, NativeQsaMode::PerRow).is_err());
    }

    #[test]
    fn routes_follow_the_c1_switch_across_the_2048_boundary() {
        use NativeAttnRoute::{IndexedDense, Sparse};
        for seq_len in 2_040..=2_053 {
            let per_row = native_attn_route(NativeQsaMode::PerRow, true, seq_len).unwrap();
            assert_eq!(
                per_row,
                if seq_len < 2_048 {
                    IndexedDense
                } else {
                    Sparse
                },
                "{seq_len}"
            );
            assert!(native_attn_route(NativeQsaMode::Off, true, seq_len).is_err());
            let dense = native_attn_route(NativeQsaMode::PerRow, false, seq_len);
            assert_eq!(dense.is_ok(), seq_len + 1 < 2_048, "{seq_len}");
        }
        assert_eq!(
            native_attn_route(NativeQsaMode::Off, false, 0),
            Ok(NativeAttnRoute::Dense)
        );
        assert_eq!(
            native_attn_route(NativeQsaMode::PerRow, true, 0),
            Ok(NativeAttnRoute::IndexedDense)
        );
        assert!(native_attn_route(NativeQsaMode::PerRow, true, usize::MAX).is_err());
        assert!(native_attn_route(NativeQsaMode::PerRow, true, u32::MAX as usize).is_err());
    }

    #[test]
    fn mixed_frames_keep_row_order_and_reject_length_mismatch() {
        let routes = native_attn_routes(
            NativeQsaMode::PerRow,
            &[false, true, true, true],
            &[17, 17, 2_047, 32_768],
        )
        .unwrap();
        assert_eq!(
            routes,
            [
                NativeAttnRoute::Dense,
                NativeAttnRoute::IndexedDense,
                NativeAttnRoute::IndexedDense,
                NativeAttnRoute::Sparse
            ]
        );
        assert_eq!(route_labels(&routes), "n,i,i,s");
        assert_eq!(
            dense_policy_seq_lens(&routes, &[17, 17, 2_047, 32_768]).unwrap(),
            [17, 0, 0, 0]
        );
        assert!(native_attn_routes(NativeQsaMode::PerRow, &[true], &[1, 2]).is_err());
        assert!(dense_policy_seq_lens(&routes, &[1]).is_err());
        assert!(
            native_attn_routes(NativeQsaMode::Off, &[false, true], &[1, 1]).is_err(),
            "selector off keeps every QSA row out of the native frame"
        );
    }

    #[test]
    fn row_offsets_match_the_batch_metadata_layout() {
        assert_eq!(
            row_metadata_offsets(0, 16_384),
            Ok(RowMetaOffsets {
                position: 0,
                slot: 0,
                seq_len: 0,
                block_table: 0
            })
        );
        assert_eq!(
            row_metadata_offsets(31, 1),
            Ok(RowMetaOffsets {
                position: 124,
                slot: 248,
                seq_len: 124,
                block_table: 124
            })
        );
        assert_eq!(
            row_metadata_offsets(3, 16_384),
            Ok(RowMetaOffsets {
                position: 12,
                slot: 24,
                seq_len: 12,
                block_table: 196_608
            })
        );
        assert!(row_metadata_offsets(usize::MAX, 2).is_err());
        assert!(row_metadata_offsets(usize::MAX / 2, 4).is_err());
    }

    fn frame(rows: u32, max_blocks: u32) -> AttnMetadataDev {
        AttnMetadataDev {
            qwen4_qsa_required: true,
            positions: DevicePtr(0x1000),
            positions_h: DevicePtr(0x1000),
            positions_w: DevicePtr(0x1000),
            slot: DevicePtr(0x2000),
            seq_len: DevicePtr(0x3000),
            block_table: DevicePtr(0x10_0000),
            max_blocks_per_seq: max_blocks,
            num_seqs: rows,
        }
    }

    #[test]
    fn row_view_is_a_one_row_pointer_window_with_the_row_flag() {
        let view = row_metadata_view(frame(4, 8), 2, false).unwrap();
        assert_eq!(view.num_seqs, 1);
        assert!(!view.qwen4_qsa_required);
        assert_eq!(view.positions, DevicePtr(0x1000 + 8));
        assert_eq!(view.positions_h, view.positions);
        assert_eq!(view.positions_w, view.positions);
        assert_eq!(view.slot, DevicePtr(0x2000 + 16));
        assert_eq!(view.seq_len, DevicePtr(0x3000 + 8));
        assert_eq!(view.block_table, DevicePtr(0x10_0000 + 2 * 8 * 4));
        assert_eq!(view.max_blocks_per_seq, 8);
        assert!(row_metadata_view(frame(4, 8), 4, true).is_err());
        assert!(
            row_metadata_view(frame(4, 8), 0, true)
                .unwrap()
                .qwen4_qsa_required
        );
    }

    #[test]
    fn route_labels_are_stable_receipt_tokens() {
        assert_eq!(NativeAttnRoute::Dense.label(), 'n');
        assert_eq!(NativeAttnRoute::IndexedDense.label(), 'i');
        assert_eq!(NativeAttnRoute::Sparse.label(), 's');
        assert!(!NativeAttnRoute::Dense.maintains_index());
        assert!(NativeAttnRoute::IndexedDense.maintains_index());
        assert!(!NativeAttnRoute::IndexedDense.is_sparse());
        assert!(NativeAttnRoute::Sparse.is_sparse());
    }

    #[test]
    fn deterministic_batch_forces_native_batch_qsa_off_and_fails_closed() {
        use crate::layers::deterministic_batch::{
            DETERMINISTIC_BATCH_ENV, DeterministicBatchPolicy,
        };
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        assert_eq!(
            native_batch_qsa_mode_with_policy(
                DeterministicBatchPolicy::Enabled,
                NativeQsaMode::PerRow
            ),
            NativeQsaMode::Off
        );
        assert_eq!(
            native_batch_qsa_mode_with_policy(
                DeterministicBatchPolicy::Disabled,
                NativeQsaMode::PerRow
            ),
            NativeQsaMode::PerRow
        );

        let _lock = test_env_lock();
        {
            let _det = ScopedEnvVar::set(DETERMINISTIC_BATCH_ENV, "1");
            let _qsa = ScopedEnvVar::set(NATIVE_BATCH_QSA_ENV, "1");
            assert_eq!(native_batch_qsa_mode().unwrap(), NativeQsaMode::Off);
        }
        {
            let _det = ScopedEnvVar::set(DETERMINISTIC_BATCH_ENV, "invalid");
            assert!(native_batch_qsa_mode().is_err());
        }
    }
}
