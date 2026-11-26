// SPDX-License-Identifier: AGPL-3.0-only

//! Dependency-free admission policy for Qwen4 native multi-sequence attention.
//!
//! The lane is intentionally narrow. It owns only C2-C256
//! (`2..=QWEN4_NATIVE_MAX_ROWS`) target decode below the sparse-QSA boundary;
//! widening either dimension requires a separate parity receipt rather than
//! silently inheriting the generic path.

pub const QWEN4_QSA_SPARSE_START: usize = 2_048;
pub const QWEN4_HC_COUNT: usize = 4;
pub const QWEN4_NATIVE_MAX_ROWS: usize = 256;

const BF16_BYTES: usize = 2;
const POSITION_BYTES: usize = 4;
const SLOT_BYTES: usize = 8;
const SEQ_LEN_BYTES: usize = 4;
const BLOCK_ENTRY_BYTES: usize = 4;

/// Byte offsets for one sequence row across the native-attention inputs.
///
/// Metadata arrays have independent bases, so these offsets are relative to
/// their corresponding `AttnMetadataDev` pointer rather than to one packed
/// allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowOffsets {
    pub hyper: usize,
    pub core: usize,
    pub position: usize,
    pub slot: usize,
    pub seq_len: usize,
    pub block_table: usize,
}

/// Total byte extents needed to address `rows` contiguous sequence rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowExtents {
    pub hyper: usize,
    pub core: usize,
    pub positions: usize,
    pub slots: usize,
    pub seq_lens: usize,
    pub block_table: usize,
}

/// Checked row geometry for the first native Qwen4 lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLayout {
    pub hyper_row_bytes: usize,
    pub core_row_bytes: usize,
    pub block_table_row_bytes: usize,
}

impl RowLayout {
    pub fn new(
        hidden_size: usize,
        hc_count: usize,
        max_blocks_per_seq: usize,
    ) -> Result<Self, &'static str> {
        if hidden_size == 0 {
            return Err("native Qwen4 attention hidden size must be non-zero");
        }
        if hc_count != QWEN4_HC_COUNT {
            return Err("native Qwen4 attention requires four hyperconnection streams");
        }
        if max_blocks_per_seq == 0 {
            return Err("native Qwen4 attention block-table stride must be non-zero");
        }
        let core_row_bytes = hidden_size
            .checked_mul(BF16_BYTES)
            .ok_or("native Qwen4 attention core-row extent overflow")?;
        let hyper_row_bytes = core_row_bytes
            .checked_mul(hc_count)
            .ok_or("native Qwen4 attention hyper-row extent overflow")?;
        let block_table_row_bytes = max_blocks_per_seq
            .checked_mul(BLOCK_ENTRY_BYTES)
            .ok_or("native Qwen4 attention block-table row extent overflow")?;
        Ok(Self {
            hyper_row_bytes,
            core_row_bytes,
            block_table_row_bytes,
        })
    }

    pub fn offsets(self, row: usize) -> Result<RowOffsets, &'static str> {
        Ok(RowOffsets {
            hyper: checked_row_offset(row, self.hyper_row_bytes)?,
            core: checked_row_offset(row, self.core_row_bytes)?,
            position: checked_row_offset(row, POSITION_BYTES)?,
            slot: checked_row_offset(row, SLOT_BYTES)?,
            seq_len: checked_row_offset(row, SEQ_LEN_BYTES)?,
            block_table: checked_row_offset(row, self.block_table_row_bytes)?,
        })
    }

    pub fn extents(self, rows: usize) -> Result<RowExtents, &'static str> {
        if rows == 0 {
            return Err("native Qwen4 attention row count must be non-zero");
        }
        let last = self.offsets(rows - 1)?;
        Ok(RowExtents {
            hyper: checked_row_end(last.hyper, self.hyper_row_bytes)?,
            core: checked_row_end(last.core, self.core_row_bytes)?,
            positions: checked_row_end(last.position, POSITION_BYTES)?,
            slots: checked_row_end(last.slot, SLOT_BYTES)?,
            seq_lens: checked_row_end(last.seq_len, SEQ_LEN_BYTES)?,
            block_table: checked_row_end(last.block_table, self.block_table_row_bytes)?,
        })
    }
}

fn checked_row_offset(row: usize, stride: usize) -> Result<usize, &'static str> {
    row.checked_mul(stride)
        .ok_or("native Qwen4 attention row offset overflow")
}

fn checked_row_end(offset: usize, width: usize) -> Result<usize, &'static str> {
    offset
        .checked_add(width)
        .ok_or("native Qwen4 attention row extent overflow")
}

#[derive(Clone, Copy, Debug)]
pub struct Admission<'a> {
    pub rows: usize,
    pub metadata_rows: usize,
    pub seq_lens: &'a [usize],
    pub block_table_rows: usize,
    pub qsa_required: bool,
    pub is_qwen4: bool,
    /// True when the layer's KV storage dtype is native-supported (BF16 or NVFP4).
    pub bf16_kv: bool,
    pub bf16_residual: bool,
    pub tp_world_size: usize,
    pub ep_world_size: usize,
    pub comm_present: bool,
    pub graph_capture: bool,
    pub tree_aware: bool,
    pub ddtree_parent_ids: bool,
    pub self_spec_sparse: bool,
    pub ffn_defer: bool,
    pub mrope_interleaved: bool,
    pub qkv_mega: bool,
}

/// Validate the complete pre-mutation contract for the first native attention
/// lane.  `seq_lens` are the lengths before the token written by this step, so
/// `seq_len + 1 == 2048` already reaches the sparse boundary and is rejected.
pub fn validate(admission: Admission<'_>) -> Result<(), &'static str> {
    if !admission.is_qwen4 {
        return Err("native attention is restricted to Qwen4");
    }
    if !(2..=QWEN4_NATIVE_MAX_ROWS).contains(&admission.rows) {
        return Err("native Qwen4 attention admits only C2-C256");
    }
    if admission.metadata_rows != admission.rows
        || admission.seq_lens.len() != admission.rows
        || admission.block_table_rows != admission.rows
    {
        return Err("native Qwen4 attention metadata must have one row per sequence");
    }
    if admission.qsa_required {
        return Err("native Qwen4 attention has no per-row QSA side cache");
    }
    for &seq_len in admission.seq_lens {
        let Some(next_len) = seq_len.checked_add(1) else {
            return Err("native Qwen4 attention sequence length overflow");
        };
        if next_len >= QWEN4_QSA_SPARSE_START {
            return Err("native Qwen4 attention cannot reach the sparse-QSA boundary");
        }
    }
    if !admission.bf16_kv || !admission.bf16_residual {
        return Err("native Qwen4 attention requires BF16/NVFP4 KV and BF16 residual storage");
    }
    if admission.tp_world_size == 0
        || admission.tp_world_size > 4
        || !(admission.ep_world_size <= 1 || admission.ep_world_size == admission.tp_world_size)
        || ((admission.tp_world_size > 1 || admission.ep_world_size > 1) && !admission.comm_present)
    {
        return Err("native Qwen4 attention supports TP<=4 with EP=1 or EP=TP (and comm when >1)");
    }
    if admission.graph_capture {
        return Err("native Qwen4 attention is eager-only in the first increment");
    }
    if admission.tree_aware || admission.ddtree_parent_ids {
        return Err("native Qwen4 attention does not admit tree verification");
    }
    if admission.self_spec_sparse || admission.ffn_defer {
        return Err("native Qwen4 attention is target-only");
    }
    if admission.mrope_interleaved && admission.qkv_mega {
        return Err("native Qwen4 interleaved MRoPE is incompatible with ATLAS_ATTN_QKV_MEGA");
    }
    Ok(())
}
