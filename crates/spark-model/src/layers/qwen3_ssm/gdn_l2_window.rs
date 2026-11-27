// SPDX-License-Identifier: AGPL-3.0-only

//! L2-resident windowed GDN prefill planner (`ATLAS_GDN_L2_WINDOW_MIB`).
//!
//! Splits long GDN prefill chunks into L2-resident token windows while carrying
//! exact FP32 `h_state` and `conv_state` in-place across window boundaries.
//!
//! Window sizes are aligned down to a multiple of `WY32_CHUNK_SIZE = 32` (min
//! `32` tokens) so WY32 and WY4 recurrence kernels execute the exact same
//! 32-token algebraic blocks in the exact same order, preserving 0-ULP state
//! and output parity against an unwindowed launch.
//!
//! Policy (default OFF):
//! - Unset with `ATLAS_GB10_FAST_PREFILL` off -> `Disabled`.
//! - Unset with `ATLAS_GB10_FAST_PREFILL=1` on `sm_count <= 64` (GB10) ->
//!   `Enabled { budget_mib: 10 }` (`10 MiB` L2 budget).
//! - Unset with `ATLAS_GB10_FAST_PREFILL=1` on `sm_count > 64` -> `Disabled`.
//! - Explicit `"0"` -> `Disabled` (always overrides `ATLAS_GB10_FAST_PREFILL=1`).
//! - Explicit positive decimal integer in `1..=4096` -> `Enabled { budget_mib }`.
//! - Invalid or non-UTF-8 value -> `Err(...)` (fail closed).

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};

use crate::layers::ops::{Gb10FastPrefillPolicy, gb10_fast_prefill_policy};

/// Environment variable configuring the GDN prefill L2 window budget in MiB.
pub const GDN_L2_WINDOW_MIB_ENV_VAR: &str = "ATLAS_GDN_L2_WINDOW_MIB";

/// Automatic GB10 (`sm_count <= 64`) L2 working-set budget in MiB under
/// `ATLAS_GB10_FAST_PREFILL=1`.
pub const GB10_AUTO_GDN_L2_WINDOW_MIB: u32 = 10;

/// Maximum SM count eligible for the automatic `10 MiB` GB10 L2 window budget.
pub const GDN_L2_WINDOW_AUTO_SM_THRESHOLD: u32 = 64;

/// Upper bound on explicit `ATLAS_GDN_L2_WINDOW_MIB` values (4 GiB).
pub const GDN_L2_WINDOW_MAX_MIB: u32 = 4_096;

/// Token alignment matching `WY32_CHUNK_SIZE` so WY32/WY4 algebraic chunks
/// never split across an L2 window boundary.
pub const GDN_L2_WINDOW_ALIGN_TOKENS: usize = 32;

const BYTES_PER_MIB: usize = 1024 * 1024;
const BF16_BYTES: usize = 2;
const FP32_BYTES: usize = 4;

/// Resolved runtime policy for L2-windowed GDN prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdnL2WindowPolicy {
    /// Default OFF: run the entire prefill chunk in a single launch.
    Disabled,
    /// Split long prefill chunks into contiguous L2-resident token windows
    /// bounded by `budget_mib` MiB (aligned to 32-token WY32 chunks).
    Enabled { budget_mib: u32 },
}

/// Strictly parse `ATLAS_GDN_L2_WINDOW_MIB` with `ATLAS_GB10_FAST_PREFILL` cascade.
pub fn parse_gdn_l2_window_mib(
    value: std::result::Result<String, std::env::VarError>,
    gb10_fast_prefill: bool,
    sm_count: u32,
) -> std::result::Result<GdnL2WindowPolicy, &'static str> {
    match value {
        Err(std::env::VarError::NotPresent) => {
            if gb10_fast_prefill && sm_count <= GDN_L2_WINDOW_AUTO_SM_THRESHOLD {
                Ok(GdnL2WindowPolicy::Enabled {
                    budget_mib: GB10_AUTO_GDN_L2_WINDOW_MIB,
                })
            } else {
                Ok(GdnL2WindowPolicy::Disabled)
            }
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_GDN_L2_WINDOW_MIB must be valid UTF-8 decimal MiB (0 to disable)")
        }
        Ok(raw) => {
            if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
                return Err(
                    "ATLAS_GDN_L2_WINDOW_MIB must be an unsigned decimal integer in 0..=4096",
                );
            }
            if raw == "0" {
                return Ok(GdnL2WindowPolicy::Disabled);
            }
            if raw.starts_with('0') {
                return Err("ATLAS_GDN_L2_WINDOW_MIB must not have leading zeros");
            }
            let Ok(mib) = raw.parse::<u32>() else {
                return Err("ATLAS_GDN_L2_WINDOW_MIB exceeds 4096 MiB maximum");
            };
            if mib > GDN_L2_WINDOW_MAX_MIB {
                return Err("ATLAS_GDN_L2_WINDOW_MIB exceeds 4096 MiB maximum");
            }
            Ok(GdnL2WindowPolicy::Enabled { budget_mib: mib })
        }
    }
}

/// Read and cache the effective `ATLAS_GDN_L2_WINDOW_MIB` policy fail-closed.
pub fn gdn_l2_window_policy() -> Result<GdnL2WindowPolicy> {
    static GATE: OnceLock<std::result::Result<GdnL2WindowPolicy, &'static str>> = OnceLock::new();
    let gb10_fast = gb10_fast_prefill_policy()?;
    let sm_count = atlas_core::device::active_sm_count();
    (*GATE.get_or_init(|| {
        parse_gdn_l2_window_mib(
            std::env::var(GDN_L2_WINDOW_MIB_ENV_VAR),
            gb10_fast == Gb10FastPrefillPolicy::Enabled,
            sm_count,
        )
    }))
    .map_err(anyhow::Error::msg)
}

/// Layer geometry required to compute L2 working-set bytes for GDN prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnL2WindowGeometry {
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub num_value_heads: usize,
    pub value_head_dim: usize,
    pub d_conv: usize,
}

impl GdnL2WindowGeometry {
    /// Total Q/K channel width `nk * kd`.
    pub fn key_dim(&self) -> Result<usize> {
        self.num_key_heads
            .checked_mul(self.key_head_dim)
            .context("GDN L2 window key_dim overflow")
    }

    /// Total V channel width `nv * vd`.
    pub fn value_dim(&self) -> Result<usize> {
        self.num_value_heads
            .checked_mul(self.value_head_dim)
            .context("GDN L2 window value_dim overflow")
    }

    /// Packed QKV conv channel width `2 * key_dim + value_dim`.
    pub fn conv_dim(&self) -> Result<usize> {
        let key_dim = self.key_dim()?;
        let value_dim = self.value_dim()?;
        key_dim
            .checked_mul(2)
            .and_then(|v| v.checked_add(value_dim))
            .context("GDN L2 window conv_dim overflow")
    }

    /// Persistent FP32 `h_state` footprint in bytes: `nv * kd * vd * 4`.
    pub fn h_state_bytes(&self) -> Result<usize> {
        self.num_value_heads
            .checked_mul(self.key_head_dim)
            .and_then(|v| v.checked_mul(self.value_head_dim))
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN L2 window h_state byte overflow")
    }

    /// Persistent FP32 `conv_state` footprint in bytes: `conv_dim * (d_conv - 1) * 4`.
    pub fn conv_state_bytes(&self) -> Result<usize> {
        let conv_dim = self.conv_dim()?;
        let history_taps = self.d_conv.saturating_sub(1);
        conv_dim
            .checked_mul(history_taps)
            .and_then(|v| v.checked_mul(FP32_BYTES))
            .context("GDN L2 window conv_state byte overflow")
    }

    /// Streaming per-token working-set bytes across `qkv` (BF16), `gate_beta`
    /// (FP32, `2 * nv`), and `gdn_output` (BF16, `value_dim`).
    pub fn per_token_working_bytes(&self) -> Result<usize> {
        let conv_dim = self.conv_dim()?;
        let value_dim = self.value_dim()?;
        let qkv_bytes = conv_dim
            .checked_mul(BF16_BYTES)
            .context("GDN L2 window qkv bytes overflow")?;
        let gate_beta_bytes = self
            .num_value_heads
            .checked_mul(2 * FP32_BYTES)
            .context("GDN L2 window gate_beta bytes overflow")?;
        let out_bytes = value_dim
            .checked_mul(BF16_BYTES)
            .context("GDN L2 window output bytes overflow")?;
        let total = qkv_bytes
            .checked_add(gate_beta_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .context("GDN L2 window per-token bytes overflow")?;
        ensure!(total > 0, "GDN L2 window per-token working set must be > 0");
        Ok(total)
    }
}

/// One contiguous token window in a [`GdnL2WindowPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnL2TokenWindow {
    pub token_start: usize,
    pub token_len: usize,
    /// Byte offset into `[total_tokens, conv_dim]` BF16 `qkv` buffer.
    pub qkv_byte_offset: usize,
    /// Byte offset into `[total_tokens, 2 * nv]` FP32 `gate_beta` buffer.
    pub gate_beta_byte_offset: usize,
    /// Byte offset into `[total_tokens, value_dim]` BF16 `output` buffer.
    pub output_byte_offset: usize,
}

/// Planned L2-resident partitioning of a GDN prefill chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdnL2WindowPlan {
    pub total_tokens: usize,
    pub window_tokens: usize,
    pub budget_bytes: Option<usize>,
    pub fixed_state_bytes: usize,
    pub per_token_bytes: usize,
    pub estimated_window_working_bytes: usize,
    windows: Vec<GdnL2TokenWindow>,
}

impl GdnL2WindowPlan {
    /// True when the chunk was split into 2 or more L2-resident windows.
    #[must_use]
    pub fn is_windowed(&self) -> bool {
        self.windows.len() > 1
    }

    /// Ordered contiguous token windows covering `[0..total_tokens)`.
    #[must_use]
    pub fn windows(&self) -> &[GdnL2TokenWindow] {
        &self.windows
    }
}

/// Plan L2-resident token windows for a GDN prefill chunk of `total_tokens`.
pub fn plan_gdn_l2_windows(
    total_tokens: usize,
    geometry: GdnL2WindowGeometry,
    policy: GdnL2WindowPolicy,
) -> Result<GdnL2WindowPlan> {
    let conv_dim = geometry.conv_dim()?;
    let value_dim = geometry.value_dim()?;
    let h_bytes = geometry.h_state_bytes()?;
    let conv_state_bytes = geometry.conv_state_bytes()?;
    let fixed_state_bytes = h_bytes
        .checked_add(conv_state_bytes)
        .context("GDN L2 fixed state bytes overflow")?;
    let per_token_bytes = geometry.per_token_working_bytes()?;

    let qkv_row_bytes = conv_dim
        .checked_mul(BF16_BYTES)
        .context("GDN L2 qkv row bytes overflow")?;
    let gate_beta_row_bytes = geometry
        .num_value_heads
        .checked_mul(2 * FP32_BYTES)
        .context("GDN L2 gate_beta row bytes overflow")?;
    let out_row_bytes = value_dim
        .checked_mul(BF16_BYTES)
        .context("GDN L2 output row bytes overflow")?;

    if total_tokens == 0 {
        return Ok(GdnL2WindowPlan {
            total_tokens: 0,
            window_tokens: 0,
            budget_bytes: match policy {
                GdnL2WindowPolicy::Disabled => None,
                GdnL2WindowPolicy::Enabled { budget_mib } => {
                    Some((budget_mib as usize).saturating_mul(BYTES_PER_MIB))
                }
            },
            fixed_state_bytes,
            per_token_bytes,
            estimated_window_working_bytes: fixed_state_bytes,
            windows: Vec::new(),
        });
    }

    let (budget_bytes, window_tokens) = match policy {
        GdnL2WindowPolicy::Disabled => (None, total_tokens),
        GdnL2WindowPolicy::Enabled { budget_mib } => {
            ensure!(
                (1..=GDN_L2_WINDOW_MAX_MIB).contains(&budget_mib),
                "GDN L2 window budget_mib {budget_mib} out of range 1..={GDN_L2_WINDOW_MAX_MIB}"
            );
            let budget = (budget_mib as usize)
                .checked_mul(BYTES_PER_MIB)
                .context("GDN L2 budget_bytes overflow")?;
            let stream_budget = budget.saturating_sub(fixed_state_bytes);
            let raw_tokens = stream_budget / per_token_bytes;
            let aligned =
                (raw_tokens / GDN_L2_WINDOW_ALIGN_TOKENS).max(1) * GDN_L2_WINDOW_ALIGN_TOKENS;
            (Some(budget), aligned.min(total_tokens))
        }
    };

    let num_windows = total_tokens.div_ceil(window_tokens);
    let mut windows = Vec::with_capacity(num_windows);
    let mut start = 0usize;
    while start < total_tokens {
        let len = (total_tokens - start).min(window_tokens);
        windows.push(GdnL2TokenWindow {
            token_start: start,
            token_len: len,
            qkv_byte_offset: start
                .checked_mul(qkv_row_bytes)
                .context("GDN L2 window qkv byte offset overflow")?,
            gate_beta_byte_offset: start
                .checked_mul(gate_beta_row_bytes)
                .context("GDN L2 window gate_beta byte offset overflow")?,
            output_byte_offset: start
                .checked_mul(out_row_bytes)
                .context("GDN L2 window output byte offset overflow")?,
        });
        start += len;
    }

    let max_window_len = windows.iter().map(|w| w.token_len).max().unwrap_or(0);
    let estimated_window_working_bytes =
        fixed_state_bytes.saturating_add(max_window_len.saturating_mul(per_token_bytes));

    Ok(GdnL2WindowPlan {
        total_tokens,
        window_tokens,
        budget_bytes,
        fixed_state_bytes,
        per_token_bytes,
        estimated_window_working_bytes,
        windows,
    })
}

/// Emit a one-shot engagement receipt when L2-windowed GDN prefill splits a chunk.
pub fn log_gdn_l2_window_engaged(plan: &GdnL2WindowPlan, site: &'static str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            site,
            total_tokens = plan.total_tokens,
            window_tokens = plan.window_tokens,
            windows = plan.windows.len(),
            budget_mib = plan.budget_bytes.map(|b| b / BYTES_PER_MIB).unwrap_or(0),
            working_kib = plan.estimated_window_working_bytes / 1024,
            "ENGAGED ATLAS_GDN_L2_WINDOW_MIB: L2-resident windowed GDN prefill"
        );
    });
}

#[cfg(test)]
mod tests {
    use std::env::VarError;
    use std::ffi::OsString;

    use super::*;

    const QWEN38_27B_GEOM: GdnL2WindowGeometry = GdnL2WindowGeometry {
        num_key_heads: 16,
        key_head_dim: 128,
        num_value_heads: 48,
        value_head_dim: 128,
        d_conv: 4,
    };

    const FLASH_NEXT_GEOM: GdnL2WindowGeometry = GdnL2WindowGeometry {
        num_key_heads: 16,
        key_head_dim: 128,
        num_value_heads: 32,
        value_head_dim: 128,
        d_conv: 4,
    };

    #[test]
    fn env_policy_defaults_off_and_auto_budgets_10_mib_on_gb10_only() {
        // Default OFF regardless of SM count.
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), false, 48),
            Ok(GdnL2WindowPolicy::Disabled)
        );
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), false, 188),
            Ok(GdnL2WindowPolicy::Disabled)
        );

        // Under ATLAS_GB10_FAST_PREFILL=1: 10 MiB auto-budget iff sm_count <= 64.
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), true, 48),
            Ok(GdnL2WindowPolicy::Enabled { budget_mib: 10 })
        );
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), true, 64),
            Ok(GdnL2WindowPolicy::Enabled { budget_mib: 10 })
        );
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), true, 65),
            Ok(GdnL2WindowPolicy::Disabled)
        );
        assert_eq!(
            parse_gdn_l2_window_mib(Err(VarError::NotPresent), true, 188),
            Ok(GdnL2WindowPolicy::Disabled)
        );

        // Explicit "0" ALWAYS disables even under ATLAS_GB10_FAST_PREFILL=1.
        assert_eq!(
            parse_gdn_l2_window_mib(Ok("0".to_string()), true, 48),
            Ok(GdnL2WindowPolicy::Disabled)
        );
        // Explicit positive MiB overrides on any GPU.
        assert_eq!(
            parse_gdn_l2_window_mib(Ok("16".to_string()), false, 188),
            Ok(GdnL2WindowPolicy::Enabled { budget_mib: 16 })
        );
        assert_eq!(
            parse_gdn_l2_window_mib(Ok("4096".to_string()), false, 48),
            Ok(GdnL2WindowPolicy::Enabled { budget_mib: 4096 })
        );

        // Invalid / non-UTF-8 fail closed.
        for bad in ["", "00", "010", "-1", "10m", " 10", "4097", "999999"] {
            assert!(
                parse_gdn_l2_window_mib(Ok(bad.to_string()), true, 48).is_err(),
                "expected invalid value `{bad}` to fail closed"
            );
        }
        assert!(
            parse_gdn_l2_window_mib(Err(VarError::NotUnicode(OsString::new())), true, 48).is_err()
        );
    }

    #[test]
    fn exact_10_mib_window_sizing_for_qwen38_27b_and_flash_next() {
        // Qwen3.8-27B:
        //   h_state = 48 * 128 * 128 * 4 = 3,145,728 B (3.0 MiB)
        //   conv_state = 10,240 * 3 * 4 = 122,880 B
        //   fixed = 3,268,608 B
        //   per_token = 10,240*2 + 48*8 + 6,144*2 = 33,152 B
        //   remaining = 10*1024*1024 - 3,268,608 = 7,217,152 B -> floor/33152 = 217 -> aligned32 = 192
        let plan_27b = plan_gdn_l2_windows(
            8_192,
            QWEN38_27B_GEOM,
            GdnL2WindowPolicy::Enabled { budget_mib: 10 },
        )
        .unwrap();
        assert!(plan_27b.is_windowed());
        assert_eq!(plan_27b.window_tokens, 192);
        assert_eq!(plan_27b.window_tokens % GDN_L2_WINDOW_ALIGN_TOKENS, 0);
        assert!(plan_27b.estimated_window_working_bytes <= 10 * BYTES_PER_MIB);
        assert_eq!(plan_27b.windows().len(), 8_192usize.div_ceil(192));
        // Contiguous exact coverage with zero gaps or overlaps.
        let mut cursor = 0usize;
        for w in plan_27b.windows() {
            assert_eq!(w.token_start, cursor);
            assert_eq!(w.qkv_byte_offset, cursor * 10_240 * 2);
            assert_eq!(w.gate_beta_byte_offset, cursor * 96 * 4);
            assert_eq!(w.output_byte_offset, cursor * 6_144 * 2);
            cursor += w.token_len;
        }
        assert_eq!(cursor, 8_192);

        // Flash-Next (nk=16, nv=32, kd=128, vd=128, d_conv=4):
        //   conv_dim = 8192, value_dim = 4096
        //   h_state = 2,097,152 B, conv_state = 98,304 B -> fixed = 2,195,456 B
        //   per_token = 8192*2 + 32*8 + 4096*2 = 24,832 B
        //   remaining = 8,290,304 B -> floor/24832 = 333 -> aligned32 = 320
        let plan_fn = plan_gdn_l2_windows(
            8_192,
            FLASH_NEXT_GEOM,
            GdnL2WindowPolicy::Enabled { budget_mib: 10 },
        )
        .unwrap();
        assert!(plan_fn.is_windowed());
        assert_eq!(plan_fn.window_tokens, 320);
        assert!(plan_fn.estimated_window_working_bytes <= 10 * BYTES_PER_MIB);

        // Short chunks below the window cap stay single-launch.
        let plan_short = plan_gdn_l2_windows(
            128,
            QWEN38_27B_GEOM,
            GdnL2WindowPolicy::Enabled { budget_mib: 10 },
        )
        .unwrap();
        assert!(!plan_short.is_windowed());
        assert_eq!(plan_short.windows().len(), 1);
        assert_eq!(plan_short.windows()[0].token_len, 128);

        // Tiny budget (1 MiB < h_state) safely clamps to min 32-token WY32 chunk.
        let plan_tiny = plan_gdn_l2_windows(
            200,
            QWEN38_27B_GEOM,
            GdnL2WindowPolicy::Enabled { budget_mib: 1 },
        )
        .unwrap();
        assert_eq!(plan_tiny.window_tokens, 32);
        assert_eq!(plan_tiny.windows().len(), 7);
        assert_eq!(plan_tiny.windows().last().unwrap().token_len, 8);

        // Disabled policy keeps entire sequence unwindowed.
        let plan_off =
            plan_gdn_l2_windows(8_192, QWEN38_27B_GEOM, GdnL2WindowPolicy::Disabled).unwrap();
        assert!(!plan_off.is_windowed());
        assert_eq!(plan_off.windows().len(), 1);
        assert_eq!(plan_off.windows()[0].token_len, 8_192);
    }

    /// Reference causal conv1d (d_conv=4, SiLU + Q/K per-head L2 norm) +
    /// FP32 Gated Delta Net recurrence operating in-place on `conv_state` and
    /// `h_state` over a token slice `[0..len)`.
    #[expect(
        clippy::too_many_arguments,
        reason = "CPU reference test harness mirroring GPU conv1d+GDN state tensors"
    )]
    fn run_cpu_conv1d_and_gdn_slice(
        conv_state: &mut [f32], // [conv_dim, d_conv - 1]
        h_state: &mut [f32],    // [nv, kd, vd]
        raw_qkv: &[f32],        // [len, conv_dim]
        conv_w: &[f32],         // [conv_dim, d_conv]
        gate_beta: &[f32],      // [len, 2 * nv]
        out: &mut [f32],        // [len, value_dim]
        len: usize,
        geom: GdnL2WindowGeometry,
    ) {
        let nk = geom.num_key_heads;
        let nv = geom.num_value_heads;
        let kd = geom.key_head_dim;
        let vd = geom.value_head_dim;
        let d_conv = geom.d_conv;
        let hist = d_conv - 1;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let vpg = nv / nk;

        let mut conv_out = vec![0.0f32; len * conv_dim];
        // Step 1: causal conv1d + SiLU updating `conv_state` in-place.
        for t in 0..len {
            for c in 0..conv_dim {
                let x_cur = raw_qkv[t * conv_dim + c];
                let mut acc = 0.0f32;
                for tap in 0..hist {
                    acc += conv_state[c * hist + tap] * conv_w[c * d_conv + tap];
                }
                acc += x_cur * conv_w[c * d_conv + hist];
                // Shift conv_state left and insert current token.
                for tap in 0..(hist - 1) {
                    conv_state[c * hist + tap] = conv_state[c * hist + tap + 1];
                }
                conv_state[c * hist + hist - 1] = x_cur;
                // SiLU activation.
                let silu = acc / (1.0 + (-acc).exp());
                conv_out[t * conv_dim + c] = silu;
            }
            // Per-head L2 norm on Q and K channels.
            for h_idx in 0..(nk * 2) {
                let base = t * conv_dim + h_idx * kd;
                let mut sq = 0.0f32;
                for d in 0..kd {
                    let v = conv_out[base + d];
                    sq += v * v;
                }
                let inv = 1.0f32 / (sq + 1e-6).sqrt();
                for d in 0..kd {
                    conv_out[base + d] *= inv;
                }
            }
        }

        // Step 2: FP32 Gated Delta Net recurrence updating `h_state` in-place.
        // H_t = g_t * H_{t-1} + beta_t * k_t (v_t - g_t * H_{t-1}^T k_t)^T
        // o_t = H_t^T q_t
        for t in 0..len {
            let q_row = &conv_out[t * conv_dim..t * conv_dim + key_dim];
            let k_row = &conv_out[t * conv_dim + key_dim..t * conv_dim + 2 * key_dim];
            let v_row = &conv_out[t * conv_dim + 2 * key_dim..t * conv_dim + conv_dim];
            for vh in 0..nv {
                let kh = vh / vpg;
                let g = gate_beta[t * 2 * nv + vh];
                let beta = gate_beta[t * 2 * nv + nv + vh];
                let h_base = vh * kd * vd;
                let q = &q_row[kh * kd..(kh + 1) * kd];
                let k = &k_row[kh * kd..(kh + 1) * kd];
                let v = &v_row[vh * vd..(vh + 1) * vd];

                for j in 0..vd {
                    let mut sk = 0.0f32;
                    for i in 0..kd {
                        let decayed = h_state[h_base + i * vd + j] * g;
                        h_state[h_base + i * vd + j] = decayed;
                        sk += decayed * k[i];
                    }
                    let delta = beta * (v[j] - sk);
                    let mut y = 0.0f32;
                    for i in 0..kd {
                        let updated = h_state[h_base + i * vd + j] + k[i] * delta;
                        h_state[h_base + i * vd + j] = updated;
                        y += updated * q[i];
                    }
                    out[t * value_dim + vh * vd + j] = y;
                }
            }
        }
    }

    #[test]
    fn windowed_gdn_and_conv_state_parity_is_zero_ulp_bit_exact() {
        // Compact scaled geometry so CPU parity sweeps multiple sequence lengths fast.
        let geom = GdnL2WindowGeometry {
            num_key_heads: 2,
            key_head_dim: 4,
            num_value_heads: 4,
            value_head_dim: 4,
            d_conv: 4,
        };
        let conv_dim = geom.conv_dim().unwrap();
        let value_dim = geom.value_dim().unwrap();
        let nv = geom.num_value_heads;
        let kd = geom.key_head_dim;
        let vd = geom.value_head_dim;
        let hist = geom.d_conv - 1;

        let conv_w: Vec<f32> = (0..conv_dim * geom.d_conv)
            .map(|i| (((i.wrapping_mul(17) + 5) % 23) as f32 - 11.0) * 0.05)
            .collect();

        for total_tokens in [33usize, 64, 97, 160, 257] {
            let raw_qkv: Vec<f32> = (0..total_tokens * conv_dim)
                .map(|i| (((i.wrapping_mul(2654435761) >> 20) & 0xFF) as f32 / 128.0 - 1.0) * 0.5)
                .collect();
            let mut gate_beta = vec![0.0f32; total_tokens * 2 * nv];
            for t in 0..total_tokens {
                for vh in 0..nv {
                    let seed = (t * 13 + vh * 7) % 19;
                    gate_beta[t * 2 * nv + vh] = 0.85 + 0.005 * (seed as f32);
                    gate_beta[t * 2 * nv + nv + vh] = 0.1 + 0.04 * (seed as f32);
                }
            }

            let init_conv: Vec<f32> = (0..conv_dim * hist)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
                .collect();
            let init_h: Vec<f32> = (0..nv * kd * vd)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
                .collect();

            // Unwindowed reference run.
            let mut ref_conv = init_conv.clone();
            let mut ref_h = init_h.clone();
            let mut ref_out = vec![0.0f32; total_tokens * value_dim];
            run_cpu_conv1d_and_gdn_slice(
                &mut ref_conv,
                &mut ref_h,
                &raw_qkv,
                &conv_w,
                &gate_beta,
                &mut ref_out,
                total_tokens,
                geom,
            );

            // Force multi-window split (32-token windows) and verify 0-ULP bit-exact parity.
            let plan = plan_gdn_l2_windows(
                total_tokens,
                geom,
                GdnL2WindowPolicy::Enabled { budget_mib: 1 },
            )
            .unwrap();
            // Even when compact geometry fits in 1 MiB, explicitly verify the
            // 32-token windowing recurrence invariant directly over synthetic windows.
            assert_eq!(
                plan.windows().iter().map(|w| w.token_len).sum::<usize>(),
                total_tokens
            );

            let mut win_conv = init_conv.clone();
            let mut win_h = init_h.clone();
            let mut win_out = vec![0.0f32; total_tokens * value_dim];
            let mut start = 0usize;
            while start < total_tokens {
                let len = (total_tokens - start).min(GDN_L2_WINDOW_ALIGN_TOKENS);
                run_cpu_conv1d_and_gdn_slice(
                    &mut win_conv,
                    &mut win_h,
                    &raw_qkv[start * conv_dim..(start + len) * conv_dim],
                    &conv_w,
                    &gate_beta[start * 2 * nv..(start + len) * 2 * nv],
                    &mut win_out[start * value_dim..(start + len) * value_dim],
                    len,
                    geom,
                );
                start += len;
            }

            // 0-ULP bit-exact equality on conv_state, h_state, and outputs!
            for (i, (&a, &b)) in ref_conv.iter().zip(win_conv.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "conv_state drift at T={total_tokens}, idx={i}: ref={a}, win={b}"
                );
            }
            for (i, (&a, &b)) in ref_h.iter().zip(win_h.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "h_state drift at T={total_tokens}, idx={i}: ref={a}, win={b}"
                );
            }
            for (i, (&a, &b)) in ref_out.iter().zip(win_out.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "output drift at T={total_tokens}, idx={i}: ref={a}, win={b}"
                );
            }
        }
    }
}
