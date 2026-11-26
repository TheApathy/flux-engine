// SPDX-License-Identifier: AGPL-3.0-only

//! Eight-threads-per-group K-row MoE down on the unified (transposed) layout
//! (verify-cost, `ATLAS_MOE_T_LANES_ROWS_DOWN_T8=1`).
//!
//! `moe_expert_{silu_down,gate_up}_shared_t_lanes_rows_t8_r{3,5,8,10}_k{40,160}` reproduces the
//! `_t_lanes_rows` down per row byte for byte (same lane partials, act
//! expression and 32-lane tree), with eight lane-threads per two-output group
//! so the tree runs in registers and width-8 shuffles instead of a
//! shared-memory reduction with two block barriers per row, and each weight
//! word is decoded once for all member rows.

/// Threads per block of the t8 down.
pub const T_LANES_T8_THREADS: u32 = 256;
/// Outputs per block: 32 groups of two adjacent outputs.
pub const T_LANES_T8_OUT_BLOCK: u32 = 64;
/// Floats per 16-element k16 block of the staged act rows (2 floats of padding
/// make the eight lane-threads' float2 reads bank-conflict free).
pub const T_LANES_T8_ACT_STRIDE: u32 = 18;
/// uint32 words per 16-element k16 block of the staged BF16-pair input rows
/// for the t8 gate/up (1 uint32 of padding makes the eight lane-threads' reads
/// bank-conflict free).
pub const T_LANES_T8_A32_STRIDE: u32 = 9;
/// The t8 down is compiled for K = 16 * 40 (Flash-Next moe_intermediate_size).
pub const T_LANES_T8_K: u32 = 640;
/// The t8 gate/up is compiled for K = 16 * 160 (Flash-Next hidden_size).
pub const T_LANES_GATE_UP_T8_K: u32 = 2560;
/// Row cap of the r3 t8 twin (`_t8_r3_k40` / `_t8_r3_k160`).
pub const T_LANES_T8_R3_MAX_ROWS: u32 = 3;
/// Row cap of the r5 t8 twin (`_t8_r5_k40` / `_t8_r5_k160`).
pub const T_LANES_T8_MAX_ROWS: u32 = 5;
/// Row cap of the r8 t8 twin (`_t8_r8_k40` / `_t8_r8_k160`, MTP K=5..7).
pub const T_LANES_T8_R8_MAX_ROWS: u32 = 8;
/// Row cap of the r10 t8 twin (`_t8_r10_k40` / `_t8_r10_k160`, 9..=10 rows).
pub const T_LANES_T8_R10_MAX_ROWS: u32 = 10;
/// Narrowest verify width served by the standalone r5 twin.
pub const T_LANES_T8_MIN_ROWS: u32 = 4;

/// Shapes the r5 t8 down was compiled for and wins at; anything else keeps the
/// rows twins unless the r3/r8 tier bundle is active.
pub const fn t_lanes_t8_admits(rows: u32, k: u32) -> bool {
    rows >= T_LANES_T8_MIN_ROWS && rows <= T_LANES_T8_MAX_ROWS && k == T_LANES_T8_K
}

/// Select the `[r3, r5, r8, r10]` tier (`0` for 2..=3, `1` for 4..=5, `2` for 6..=8, `3` for 9..=10)
/// for the t8 down when `k == T_LANES_T8_K`.
pub const fn t_lanes_t8_tier_admits(rows: u32, k: u32) -> Option<usize> {
    if k != T_LANES_T8_K {
        return None;
    }
    match rows {
        2..=3 => Some(0),
        4..=5 => Some(1),
        6..=8 => Some(2),
        9..=10 => Some(3),
        _ => None,
    }
}

/// Dynamic shared memory of the t8 down: `rows` padded act rows of width `k`.
pub const fn t_lanes_t8_smem_bytes(rows: u32, k: u32) -> u32 {
    rows * (k / 16) * T_LANES_T8_ACT_STRIDE * 4
}

/// Shapes the r5 t8 gate/up was compiled for (`_t8_r5_k160`, K=2560, 4..5 rows).
pub const fn t_lanes_gate_up_t8_admits(rows: u32, k: u32) -> bool {
    rows >= T_LANES_T8_MIN_ROWS && rows <= T_LANES_T8_MAX_ROWS && k == T_LANES_GATE_UP_T8_K
}

/// Select the `[r3, r5, r8, r10]` tier (`0` for 2..=3, `1` for 4..=5, `2` for 6..=8, `3` for 9..=10)
/// for the t8 gate/up when `k == T_LANES_GATE_UP_T8_K`.
pub const fn t_lanes_gate_up_t8_tier_admits(rows: u32, k: u32) -> Option<usize> {
    if k != T_LANES_GATE_UP_T8_K {
        return None;
    }
    match rows {
        2..=3 => Some(0),
        4..=5 => Some(1),
        6..=8 => Some(2),
        9..=10 => Some(3),
        _ => None,
    }
}

/// Dynamic shared memory of the t8 gate/up: `rows` padded BF16-pair rows of width `k`.
pub const fn t_lanes_gate_up_t8_smem_bytes(rows: u32, k: u32) -> u32 {
    rows * (k / 16) * T_LANES_T8_A32_STRIDE * 4
}

// Tiers r3..=r8 (and r10 down) fit within the 48 KiB default dynamic shared
// memory limit; r10 gate/up (57,600 B) fits within the 99 KiB extended dynamic
// shared memory limit (opted in automatically by `atlas-core`).
const _: () = assert!(t_lanes_t8_smem_bytes(T_LANES_T8_R8_MAX_ROWS, T_LANES_T8_K) <= 48 * 1024);
const _: () = assert!(t_lanes_t8_smem_bytes(T_LANES_T8_R10_MAX_ROWS, T_LANES_T8_K) <= 48 * 1024);
const _: () = assert!(
    t_lanes_gate_up_t8_smem_bytes(T_LANES_T8_R8_MAX_ROWS, T_LANES_GATE_UP_T8_K) <= 48 * 1024
);
const _: () = assert!(
    t_lanes_gate_up_t8_smem_bytes(T_LANES_T8_R10_MAX_ROWS, T_LANES_GATE_UP_T8_K) <= 99 * 1024
);

#[cfg(test)]
mod tests {
    use super::{
        T_LANES_T8_MAX_ROWS, T_LANES_T8_MIN_ROWS, t_lanes_gate_up_t8_admits,
        t_lanes_gate_up_t8_smem_bytes, t_lanes_gate_up_t8_tier_admits, t_lanes_t8_admits,
        t_lanes_t8_smem_bytes, t_lanes_t8_tier_admits,
    };

    #[test]
    fn t8_admits_only_the_widths_it_wins() {
        assert!(t_lanes_t8_admits(4, 640));
        assert!(t_lanes_t8_admits(5, 640));
        assert!(!t_lanes_t8_admits(T_LANES_T8_MIN_ROWS - 1, 640));
        assert!(!t_lanes_t8_admits(1, 640));
        assert!(!t_lanes_t8_admits(T_LANES_T8_MAX_ROWS + 1, 640));
        assert!(!t_lanes_t8_admits(5, 512));
        assert!(t_lanes_gate_up_t8_admits(4, 2560));
        assert!(t_lanes_gate_up_t8_admits(5, 2560));
        assert!(!t_lanes_gate_up_t8_admits(3, 2560));
        assert!(!t_lanes_gate_up_t8_admits(5, 2048));
        assert_eq!(t_lanes_t8_tier_admits(2, 640), Some(0));
        assert_eq!(t_lanes_t8_tier_admits(3, 640), Some(0));
        assert_eq!(t_lanes_t8_tier_admits(4, 640), Some(1));
        assert_eq!(t_lanes_t8_tier_admits(5, 640), Some(1));
        assert_eq!(t_lanes_t8_tier_admits(6, 640), Some(2));
        assert_eq!(t_lanes_t8_tier_admits(8, 640), Some(2));
        assert_eq!(t_lanes_t8_tier_admits(10, 640), Some(3));
        assert_eq!(t_lanes_t8_tier_admits(11, 640), None);
        assert_eq!(t_lanes_gate_up_t8_tier_admits(3, 2560), Some(0));
        assert_eq!(t_lanes_gate_up_t8_tier_admits(5, 2560), Some(1));
        assert_eq!(t_lanes_gate_up_t8_tier_admits(8, 2560), Some(2));
        assert_eq!(t_lanes_gate_up_t8_tier_admits(10, 2560), Some(3));
    }

    #[test]
    fn t8_smem_is_padded_act_rows() {
        // K5: 5 rows x 40 k16 blocks x 18 floats.
        assert_eq!(t_lanes_t8_smem_bytes(5, 640), 5 * 40 * 18 * 4);
        // K8: 8 rows x 40 k16 blocks x 18 floats = 23,040 B.
        assert_eq!(t_lanes_t8_smem_bytes(8, 640), 8 * 40 * 18 * 4);
        // K5 gate/up: 5 rows x 160 k16 blocks x 9 u32 words.
        assert_eq!(t_lanes_gate_up_t8_smem_bytes(5, 2560), 5 * 160 * 9 * 4);
        // K8 gate/up: 8 rows x 160 k16 blocks x 9 u32 words = 46,080 B.
        assert_eq!(t_lanes_gate_up_t8_smem_bytes(8, 2560), 8 * 160 * 9 * 4);
    }
}
