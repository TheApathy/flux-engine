// SPDX-License-Identifier: AGPL-3.0-only
//! Strict candidate-eager contract for graph-safe Qwen4 K5 attention.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH";
pub(crate) const FULL_ATTENTION_LAYERS: [usize; 12] =
    [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
const VERIFY_MAX_ROWS: usize = crate::layers::QWEN4_HYBRID_MAX_ROWS;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Qwen4DeviceAttentionRoute {
    DensePaged { num_splits: u32 },
    SparseQsa,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4DeviceAttentionRow {
    pub qsa_pool_endpoint: bool,
    pub qsa_visible_groups: u32,
    pub qsa_score_grid_x: u32,
    pub route: Qwen4DeviceAttentionRoute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4K5DeviceAttentionPlan {
    num_rows: usize,
    rows: [Qwen4DeviceAttentionRow; VERIFY_MAX_ROWS],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Qwen4K5DeviceAttentionTopology {
    num_rows: u8,
    pool_endpoint_mask: u16,
    score_grids: [u32; VERIFY_MAX_ROWS],
    routes: [Qwen4DeviceAttentionRoute; VERIFY_MAX_ROWS],
}

fn parse_exact_bool(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name} must be absent, 0, or 1; got {other:?}"),
    }
}

fn exact_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_exact_bool(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_bool(name, None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be UTF-8 0 or 1"),
    }
}

fn exact_zero_env(name: &str) -> Result<bool> {
    Ok(std::env::var_os(name).is_some() && !exact_env(name)?)
}

fn exact_layer_map(config: &ModelConfig) -> bool {
    config.num_hidden_layers == 48
        && (0..48).all(|l| {
            let full = FULL_ATTENTION_LAYERS.contains(&l);
            config.layer_type(l)
                == if full {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
        })
}

fn classify(position: usize) -> Result<Qwen4DeviceAttentionRow> {
    let sequence_length = position.checked_add(1).context("position overflow")?;
    let visible_groups = sequence_length / 4;
    let score_grid = if visible_groups > 512 {
        u32::try_from(visible_groups.div_ceil(8))?
            .checked_next_power_of_two()
            .context("QSA score grid overflow")?
    } else {
        0
    };
    let route = if position >= 2_048 {
        Qwen4DeviceAttentionRoute::SparseQsa
    } else {
        Qwen4DeviceAttentionRoute::DensePaged { num_splits: 2 }
    };
    Ok(Qwen4DeviceAttentionRow {
        qsa_pool_endpoint: sequence_length.is_multiple_of(4),
        qsa_visible_groups: u32::try_from(visible_groups)?,
        qsa_score_grid_x: score_grid,
        route,
    })
}

impl Qwen4K5DeviceAttentionPlan {
    fn from_validated_rows(num_tokens: usize, seq_len: usize) -> Result<Self> {
        ensure!(
            (2..=VERIFY_MAX_ROWS).contains(&num_tokens),
            "{SELECTOR} bad row count"
        );
        let mut rows = [classify(seq_len)?; VERIFY_MAX_ROWS];
        for (row, item) in rows.iter_mut().take(num_tokens).enumerate() {
            *item = classify(seq_len.checked_add(row).context("row position overflow")?)?;
        }
        Ok(Self {
            num_rows: num_tokens,
            rows,
        })
    }

    pub(crate) fn from_host(
        config: &ModelConfig,
        num_tokens: usize,
        seq_len: usize,
        metadata: AttnMetadataDev,
    ) -> Result<Option<Self>> {
        if !exact_env(SELECTOR)? {
            return Ok(None);
        }
        // Specialized for K=5 by default, or 2..=VERIFY_MAX_ROWS under ATLAS_QWEN4_HYBRID_ALL_K=1.
        let allow_all_k =
            crate::layers::qwen4_hybrid_all_k() && (2..=VERIFY_MAX_ROWS).contains(&num_tokens);
        if num_tokens != 5 && !allow_all_k || !config.is_qwen4_exp() || !metadata.qwen4_qsa_required
        {
            return Ok(None);
        }
        ensure!(
            exact_env("ATLAS_QWEN4_K5_HYBRID")? && exact_env("ATLAS_QWEN4_K5_BATCH_ATTN_QKV")?,
            "{SELECTOR} requires the qualified exact K5 attention route"
        );
        ensure!(
            exact_zero_env("ATLAS_QWEN4_K5_BATCH_HYPER")?
                && !exact_env("ATLAS_PAGED_DECODE_SPLITK")?,
            "{SELECTOR} requires row-exact hyper and legacy two-way paged splits"
        );
        let c = config;
        ensure!(
            c.is_qwen4_exp()
                && c.hidden_size == 2_560
                && c.num_attention_heads == 24
                && c.num_key_value_heads == 2
                && c.head_dim == 256
                && exact_layer_map(c),
            "{SELECTOR} requires exact Qwen3.8-Flash-Next target geometry"
        );
        let m = metadata;
        ensure!(
            m.qwen4_qsa_required && m.num_seqs as usize == num_tokens && m.max_blocks_per_seq > 0,
            "{SELECTOR} requires complete QSA metadata for all {num_tokens} rows"
        );
        let ptrs = [m.positions.0, m.slot.0, m.seq_len.0, m.block_table.0];
        ensure!(
            !ptrs.contains(&0) && ptrs.iter().enumerate().all(|(i, p)| !ptrs[..i].contains(p)),
            "{SELECTOR} requires stable non-aliasing metadata pointers"
        );
        Self::from_validated_rows(num_tokens, seq_len).map(Some)
    }

    pub(crate) fn row(self, row: usize) -> Result<Qwen4DeviceAttentionRow> {
        ensure!(row < self.num_rows, "{SELECTOR} row out of range");
        Ok(self.rows[row])
    }

    fn max_visible_groups(self) -> u32 {
        self.rows[..self.num_rows]
            .iter()
            .map(|r| r.qsa_visible_groups)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn topology(self) -> Qwen4K5DeviceAttentionTopology {
        let mut pool_endpoint_mask = 0u16;
        let mut score_grids = [0u32; VERIFY_MAX_ROWS];
        let mut routes = [Qwen4DeviceAttentionRoute::SparseQsa; VERIFY_MAX_ROWS];
        for (index, row) in self.rows[..self.num_rows].iter().enumerate() {
            pool_endpoint_mask |= u16::from(row.qsa_pool_endpoint) << index;
            score_grids[index] = row.qsa_score_grid_x;
            routes[index] = row.route;
        }
        Qwen4K5DeviceAttentionTopology {
            num_rows: self.num_rows as u8,
            pool_endpoint_mask,
            score_grids,
            routes,
        }
    }
}

impl Qwen3AttentionLayer {
    /// Resolve every candidate dependency before hyperconnection or state kernels
    /// execute; a requested route may fail, never fall through to host-scalar attention.
    pub(super) fn prepare_qwen4_k5_device_attention(
        &self,
        plan: Qwen4K5DeviceAttentionPlan,
        kv_cache: &PagedKvCache,
        metadata: AttnMetadataDev,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let num_tokens = plan.num_rows;
        ensure!(
            self.attn_layer_idx < FULL_ATTENTION_LAYERS.len(),
            "{SELECTOR} bad attn ordinal"
        );
        let a = &self.attn;
        ensure!(
            self.gated
                && self.mla.is_none()
                && self.num_q_heads_override.is_none()
                && self.num_kv_heads_override.is_none()
                && self.head_dim_override.is_none()
                && a.q_norm_full.is_none()
                && a.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "{SELECTOR} requires ordinary gated Qwen4 attention geometry"
        );
        ensure!(
            [&self.q_weight, &self.k_weight, &self.v_weight]
                .iter()
                .all(|w| w.as_ref().and_then(|w| w.as_nvfp4()).is_some()),
            "{SELECTOR} requires ordinary NVFP4 Q/K/V weights"
        );
        let k = &self.w4a16_exact_qkv_kernels;
        ensure!(
            k.qg_for_rows(num_tokens).0 != 0 && k.dual_kv_for_rows(num_tokens).0 != 0,
            "{SELECTOR} exact K{num_tokens} QKV kernels are unavailable"
        );
        let qkv_ok = num_tokens
            .checked_mul((24 + 2) * 256 * 4)
            .is_some_and(|b| b <= ctx.buffers.sizes().qkv_output);
        ensure!(qkv_ok, "{SELECTOR} QKV arena too small");
        ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none(),
            "{SELECTOR} rejects HSS cache"
        );
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .context("requires QSA on every attention layer")?;
        qsa.prepare_device_attention(kv_cache, metadata, plan.max_visible_groups(), ctx.gpu)
            .map(|_| ())
    }
}

#[cfg(test)]
#[path = "qwen4_device_attention_tests.rs"]
mod tests;
