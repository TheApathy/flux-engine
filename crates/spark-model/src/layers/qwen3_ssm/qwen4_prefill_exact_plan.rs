// SPDX-License-Identifier: AGPL-3.0-only

//! Checked layout for canonical BF16 Flash-Next input and FP32 recurrence.

use std::ffi::OsStr;

#[cfg(test)]
mod tests;

pub(super) fn parse_selector(value: Option<&OsStr>) -> Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("selector must be absent or exactly 0 or 1"),
    }
}

pub(super) fn validate_request(
    rows: usize,
    start: usize,
    multi_chunk: bool,
) -> Result<(), &'static str> {
    if !crate::layers::qwen4_prefill_moe::chunk_window_for_selection(rows, start, multi_chunk) {
        return Err("exact SSM prefill requires an admitted nonempty prompt chunk");
    }
    Ok(())
}

pub(super) const fn grid32_partition(rows: usize) -> (usize, usize) {
    let full_rows = rows / 32 * 32;
    (full_rows, rows - full_rows)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SsmShape {
    pub nk: usize,
    pub nv: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub qkvz: usize,
    pub gates: usize,
    pub h_state_bytes: usize,
    pub conv_state_bytes: usize,
}

impl SsmShape {
    pub const fn from_heads(nk: usize, nv: usize) -> Self {
        let key_dim = nk * 128;
        let value_dim = nv * 128;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz = conv_dim + value_dim;
        let gates = nv * 2;
        let h_state_bytes = nv * 128 * 128 * 4;
        let conv_state_bytes = conv_dim * 4 * 4;
        Self {
            nk,
            nv,
            key_dim,
            value_dim,
            conv_dim,
            qkvz,
            gates,
            h_state_bytes,
            conv_state_bytes,
        }
    }
}

pub(super) struct Plan {
    pub rows: usize,
    pub shape: SsmShape,
    /// norm input, QKVZ, FP32 conv/GDN, gates, normalized GDN, output.
    pub bytes: [usize; 6],
    pub residual_bytes: usize,
}

impl Plan {
    pub const H: usize = 2560;
    pub const KEY_DIM: usize = 2048;
    pub const VALUE_DIM: usize = 6144;
    pub const CONV_DIM: usize = Self::KEY_DIM * 2 + Self::VALUE_DIM;
    pub const QKVZ: usize = Self::CONV_DIM + Self::VALUE_DIM;
    pub const GATES: usize = 96;
    #[cfg(test)]
    pub const H_STATE_BYTES: usize = 48 * 128 * 128 * 4;
    #[cfg(test)]
    pub const CONV_STATE_BYTES: usize = Self::CONV_DIM * 4 * 4;

    #[cfg(test)]
    pub fn new(rows: usize, capacity: usize, limits: [usize; 6]) -> Result<Self, &'static str> {
        Self::new_for_heads(rows, capacity, limits, 16, 48)
    }

    pub fn new_for_heads(
        rows: usize,
        capacity: usize,
        limits: [usize; 6],
        nk: usize,
        nv: usize,
    ) -> Result<Self, &'static str> {
        if !matches!((nk, nv), (16, 48) | (8, 24) | (4, 12)) {
            return Err(
                "exact SSM staging requires (16, 48), TP=2 (8, 24), or TP=4 (4, 12) GDN heads",
            );
        }
        let shape = SsmShape::from_heads(nk, nv);
        validate_request(rows, 0, false)?;
        if rows < 2 || rows > capacity {
            return Err("exact SSM staging requires 2..2048 rows within arena capacity");
        }
        let fi_active = rows > 64
            && super::qwen4_gdn_fi::selected()
            && !super::qwen4_prefill_exact::check_selected().unwrap_or(false);
        let conv_row_bytes = if fi_active {
            if matches!(
                super::qwen4_gdn_norm_fp8::mode().ok().flatten(),
                Some(super::qwen4_gdn_norm_fp8::Mode::FiStatic)
            ) {
                shape.conv_dim * 2
            } else {
                shape.qkvz * 2
            }
        } else {
            shape.qkvz * 4
        };
        let strides = [
            Self::H * 2,
            shape.qkvz * 2,
            conv_row_bytes,
            shape.gates * 4,
            shape.value_dim * 2,
            Self::H * 2,
        ];
        let mut bytes = [0; 6];
        for index in 0..bytes.len() {
            bytes[index] = rows
                .checked_mul(strides[index])
                .ok_or("SSM extent overflow")?;
            if bytes[index] > limits[index] {
                return Err("exact SSM staging exceeds its arena");
            }
        }
        let residual_bytes = rows
            .checked_mul(10_240 * 2)
            .ok_or("SSM residual overflow")?;
        Ok(Self {
            rows,
            shape,
            bytes,
            residual_bytes,
        })
    }

    pub fn tiles(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.tiles_from(0)
    }

    pub fn tiles_from(&self, first: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        (first..self.rows)
            .step_by(32)
            .map(|start| (start, (self.rows - start).min(32)))
    }
}

/// Validate typed, simultaneously live device spans without touching the GPU.
pub(super) fn validate_regions(regions: &[(u64, usize, usize)]) -> Result<(), &'static str> {
    for (index, &(ptr, bytes, alignment)) in regions.iter().enumerate() {
        if ptr == 0 || bytes == 0 || !matches!(alignment, 2 | 4) || ptr % alignment as u64 != 0 {
            return Err("invalid exact SSM typed device span");
        }
        let end = ptr
            .checked_add(bytes.try_into().map_err(|_| "SSM extent overflow")?)
            .ok_or("SSM device address overflow")?;
        for &(other, count, _) in &regions[..index] {
            let other_end = other
                .checked_add(count.try_into().map_err(|_| "SSM extent overflow")?)
                .ok_or("SSM device address overflow")?;
            if end > other && other_end > ptr {
                return Err("exact SSM live device spans overlap");
            }
        }
    }
    Ok(())
}
