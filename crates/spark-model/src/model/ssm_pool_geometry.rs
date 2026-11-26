// SPDX-License-Identifier: AGPL-3.0-only

//! Checked single source of truth for persistent SSM active-pool geometry.

use anyhow::{Result, anyhow, bail};
use atlas_core::config::ModelConfig;

#[path = "ssm_state_geometry.rs"]
mod state_geometry;
use state_geometry::checked_ssm_state_bytes;
pub const DDTREE_KERNEL_KMAX: usize = 32;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsmSpeculativeGeometry {
    pub dflash_verify_width: usize,
    pub ddtree_capacity: usize,
    pub num_intermediates: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct SsmPoolGeometryInput {
    pub max_slots: usize,
    pub num_ssm_layers: usize,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub has_mtp: bool,
    pub compact_lazy_h: bool,
    pub num_intermediates: usize,
    pub lazy_commit: bool,
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub num_value_heads: usize,
    pub value_head_dim: usize,
}
impl SsmPoolGeometryInput {
    pub fn from_config(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        lazy_commit: bool,
    ) -> Result<Self> {
        let num_ssm_layers = config.num_ssm_layers();
        let (h_bytes, conv_bytes) = if num_ssm_layers == 0 {
            (0, 0)
        } else {
            checked_ssm_state_bytes(config)?
        };
        Ok(Self {
            max_slots,
            num_ssm_layers,
            h_bytes,
            conv_bytes,
            has_mtp,
            compact_lazy_h: false,
            num_intermediates,
            lazy_commit,
            num_key_heads: config.linear_num_key_heads,
            key_head_dim: config.linear_key_head_dim,
            num_value_heads: config.linear_num_value_heads,
            value_head_dim: config.linear_value_head_dim,
        })
    }

    pub fn with_qwen4_native_mtp(mut self, config: &ModelConfig, native_qwen4_mtp: bool) -> Self {
        self.compact_lazy_h = native_qwen4_mtp
            && qwen4_compact_ssm_eligible(config, self.has_mtp, self.num_intermediates);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsmPoolGeometry {
    pub total_slots: usize,
    pub state_copies: usize,
    pub compact_lazy_h: bool,
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub h_state_allocation_bytes: usize,
    pub conv_state_allocation_bytes: usize,
    pub h_intermediate_allocation_bytes: usize,
    pub conv_intermediate_allocation_bytes: usize,
    pub h_verify_scratch_bytes: usize,
    /// Slots that own speculative state (intermediates + checkpoint): every
    /// dummy-inclusive slot, or only the claimable ones under
    /// `ATLAS_SSM_DUMMY_STATE_ONLY=1` / `compact_lazy_h`.
    pub spec_slots: usize,
    pub h_checkpoint_allocation_bytes: usize,
    pub conv_checkpoint_allocation_bytes: usize,
    pub kv_retain_bytes: usize,
    pub gate_retain_bytes: usize,
    pub kv_retain_allocation_bytes: usize,
    pub gate_retain_allocation_bytes: usize,
    pub bytes_per_layer_slot: usize,
    pub total_bytes: usize,
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("SSM pool geometry overflow: {label} ({left} + {right})"))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("SSM pool geometry overflow: {label} ({left} * {right})"))
}

pub fn checked_ssm_speculative_geometry(
    has_mtp: bool,
    dflash_enabled: bool,
    num_drafts: usize,
    requested_ddtree_capacity: Option<usize>,
) -> Result<SsmSpeculativeGeometry> {
    let uncap_tree = std::env::var("ATLAS_DDTREE_UNCAP").ok().as_deref() == Some("1");
    checked_ssm_speculative_geometry_with_uncap(
        has_mtp,
        dflash_enabled,
        num_drafts,
        requested_ddtree_capacity,
        uncap_tree,
    )
}

/// Pure variant of [`checked_ssm_speculative_geometry`] taking `uncap_tree`
/// (`ATLAS_DDTREE_UNCAP=1`) explicitly so callers and tests do not mutate the
/// process environment.
pub fn checked_ssm_speculative_geometry_with_uncap(
    has_mtp: bool,
    dflash_enabled: bool,
    num_drafts: usize,
    requested_ddtree_capacity: Option<usize>,
    uncap_tree: bool,
) -> Result<SsmSpeculativeGeometry> {
    let dflash_verify_width = if dflash_enabled {
        let width = checked_add(num_drafts, 1, "DFlash verify width")?;
        if width > DDTREE_KERNEL_KMAX {
            bail!(
                "SSM pool geometry: DFlash verify width {width} exceeds tree-WY kernel maximum {DDTREE_KERNEL_KMAX}"
            );
        }
        width
    } else {
        0
    };
    let ddtree_capacity = if dflash_enabled {
        requested_ddtree_capacity
            .unwrap_or(dflash_verify_width)
            .clamp(dflash_verify_width, DDTREE_KERNEL_KMAX)
    } else {
        0
    };
    let num_intermediates = if has_mtp {
        let flat = checked_add(num_drafts, 2, "drafts + 2 intermediates")?;
        // The tree term is only paid for when the tree can actually place a
        // node, which requires `ATLAS_DDTREE_UNCAP=1`.
        //
        // `draft_budget::DflashDraftBudget::new` caps `tree_nodes` at `flat`
        // unless that flag is set, so `remaining = tree_nodes - flat` is always
        // zero and the branch builder exits before placing its first node.
        // Sizing the intermediate pool for `ddtree_capacity + 1` regardless
        // therefore allocates unused slots (e.g., `max(9, 17) = 17` for
        // `num_drafts = 7, ddtree_capacity = 16`, or `max(17, 32) = 32` for
        // `num_drafts = 15, ddtree_capacity = 31`). Both the pool and the draft
        // budget must share this gate so they agree on row capacity.
        if uncap_tree {
            let tree = checked_add(ddtree_capacity, 1, "DDTree capacity + 1 intermediates")?;
            flat.max(tree)
        } else {
            flat
        }
    } else {
        0
    };
    Ok(SsmSpeculativeGeometry {
        dflash_verify_width,
        ddtree_capacity,
        num_intermediates,
    })
}

/// `ATLAS_SSM_DUMMY_STATE_ONLY=1` (default off): the reserved padding slot
/// keeps only its base h/conv state. Padding rows never address intermediates
/// or checkpoints (`decode_a2`, `decode_a2_piecewise` and `decode_b` build
/// their states with `None` / empty), so the speculative pools are sized for
/// the claimable slots alone.
pub fn ssm_dummy_state_only() -> bool {
    std::env::var("ATLAS_SSM_DUMMY_STATE_ONLY").ok().as_deref() == Some("1")
}

/// Whether Qwen4 native-MTP speculative verify can omit per-layer `H`
/// intermediate and checkpoint pools (`ATLAS_QWEN4_REPLAY_SSM != 0`).
///
/// Requires the qualified Qwen4 `lazyfinal` + `nosnap` replay GDN verify route
/// across every width the proposer can emit (`K in 2..=num_intermediates - 1`,
/// capped at `QWEN4_HYBRID_MAX_ROWS = 9` -> `num_intermediates <= 10` under
/// `ATLAS_QWEN4_HYBRID_ALL_K=1`), so even shortened frames (`K=2..=4`) keep
/// `ssm.h_state` step-initial during verify and reconstruct accepted `H` in
/// place from retained `(conv_rows, gates)` on commit.
pub fn qwen4_compact_ssm_eligible(
    config: &ModelConfig,
    has_mtp: bool,
    num_intermediates: usize,
) -> bool {
    config.is_qwen4_exp()
        && has_mtp
        && (3..=10).contains(&num_intermediates)
        && std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY")
            .ok()
            .as_deref()
            != Some("1")
        && std::env::var("ATLAS_QWEN4_REPLAY_SSM").ok().as_deref() != Some("0")
        && std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() != Some("1")
        && std::env::var("ATLAS_SSM_H_FP16").ok().as_deref() != Some("1")
        && std::env::var("ATLAS_DDTREE_UNCAP").ok().as_deref() != Some("1")
        && std::env::var("ATLAS_SPEC_SKIP_SSM_RESTORE_CONTROL")
            .ok()
            .as_deref()
            != Some("1")
        && std::env::var("ATLAS_QWEN4_K5_HYBRID").ok().as_deref() == Some("1")
        && std::env::var("ATLAS_QWEN4_K5_BATCH_SSM").ok().as_deref() == Some("1")
        && std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1")
        && std::env::var("ATLAS_QWEN4_HYBRID_ALL_K").ok().as_deref() == Some("1")
}

thread_local! {
    static QWEN4_SNAPSHOT_SCOPE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Scoped RAII binding of `config.is_qwen4_exp()` across `SsmSnapshotPool`
/// construction on the calling thread, keeping `SsmSnapshotPool::new`'s pinned
/// constructor ABI intact without cross-thread global state or env heuristics.
pub(crate) struct Qwen4SnapshotScope {
    prev: Option<bool>,
}

impl Qwen4SnapshotScope {
    pub(crate) fn enter(is_qwen4_exp: bool) -> Self {
        let prev = QWEN4_SNAPSHOT_SCOPE.with(|cell| cell.replace(Some(is_qwen4_exp)));
        Self { prev }
    }
}

impl Drop for Qwen4SnapshotScope {
    fn drop(&mut self) {
        QWEN4_SNAPSHOT_SCOPE.with(|cell| cell.set(self.prev));
    }
}

/// Pure fail-closed decision helper for [`ssm_decode_ring_enabled_for_model`].
pub fn ssm_decode_ring_enabled_for_env(
    explicit: Option<&str>,
    is_qwen4_exp: bool,
    decode_max_seqs: usize,
) -> Result<bool> {
    match explicit.map(str::trim) {
        None => Ok(!(is_qwen4_exp && decode_max_seqs > 1)),
        Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        Some(other) => {
            bail!(
                "ATLAS_SSM_DECODE_RING / FLUX_SSM_DECODE_RING must be 0/false/off or 1/true/on, got {other:?}"
            )
        }
    }
}

/// Whether Phase-C SSM decode-rollback ring allocation is enabled for a
/// `decode_max_seqs`-slot pool (`ATLAS_SSM_DECODE_RING` / `FLUX_SSM_DECODE_RING`).
///
/// Explicit boolean (`0`/`false`/`off` or `1`/`true`/`on`) wins; invalid values
/// fail closed. Otherwise, Qwen4
/// multi-sequence serving (`is_qwen4_exp && decode_max_seqs > 1`, including
/// default-on native batching with zero env vars set) omits the unreachable
/// `3 × 114 MiB/slot` Phase-C boundary ring (`21.3 GiB` at `C=64`, `42.6 GiB`
/// at `C=128`), while non-Qwen4 hybrid SSM models retain their ring.
pub fn ssm_decode_ring_enabled_for_model(
    is_qwen4_exp: bool,
    decode_max_seqs: usize,
) -> Result<bool> {
    let read_var = |name: &str| -> Result<Option<String>> {
        match std::env::var(name) {
            Ok(v) => Ok(Some(v)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => bail!("{name} must be valid UTF-8"),
        }
    };
    let explicit = match read_var("FLUX_SSM_DECODE_RING")? {
        Some(v) => Some(v),
        None => read_var("ATLAS_SSM_DECODE_RING")?,
    };
    ssm_decode_ring_enabled_for_env(explicit.as_deref(), is_qwen4_exp, decode_max_seqs)
}

pub(crate) fn ssm_decode_ring_enabled_in_scope(decode_max_seqs: usize) -> Result<bool> {
    let is_qwen4_exp = QWEN4_SNAPSHOT_SCOPE
        .with(|cell| cell.get())
        .unwrap_or(false);
    ssm_decode_ring_enabled_for_model(is_qwen4_exp, decode_max_seqs)
}

pub fn checked_ssm_pool_geometry(input: SsmPoolGeometryInput) -> Result<SsmPoolGeometry> {
    if input.num_ssm_layers == 0 {
        return Ok(SsmPoolGeometry::default());
    }
    if input.h_bytes == 0 {
        bail!("SSM pool geometry: enabled h state bytes must be positive");
    }
    if input.conv_bytes == 0 {
        bail!("SSM pool geometry: enabled conv state bytes must be positive");
    }
    if input.has_mtp && input.num_intermediates == 0 {
        bail!("SSM pool geometry: enabled intermediate count must be positive");
    }
    if !input.has_mtp && input.num_intermediates != 0 {
        bail!("SSM pool geometry: intermediates require speculative state");
    }
    if input.compact_lazy_h && !input.has_mtp {
        bail!("SSM pool geometry: compact lazy H requires speculative state");
    }

    let total_slots = checked_add(input.max_slots, 1, "dummy-inclusive slots")?;
    let spec_slots =
        if input.has_mtp && input.max_slots > 0 && (input.compact_lazy_h || ssm_dummy_state_only())
        {
            input.max_slots
        } else {
            total_slots
        };
    let state_bytes = checked_add(input.h_bytes, input.conv_bytes, "state bytes per copy")?;
    let state_copies = if input.has_mtp {
        checked_add(
            input.num_intermediates,
            2,
            "base + intermediates + checkpoint state copies",
        )?
    } else {
        1
    };

    let (kv_retain_bytes, gate_retain_bytes) = if input.has_mtp && input.lazy_commit {
        let key_width = checked_mul(
            input.num_key_heads,
            input.key_head_dim,
            "retention key width",
        )?;
        let doubled_keys = checked_mul(key_width, 2, "retention doubled key width")?;
        let value_width = checked_mul(
            input.num_value_heads,
            input.value_head_dim,
            "retention value width",
        )?;
        let conv_width = checked_add(doubled_keys, value_width, "retention conv width")?;
        if conv_width == 0 || input.num_value_heads == 0 {
            bail!("SSM pool geometry: enabled lazy retention dimensions must be positive");
        }
        let kv_elements =
            checked_mul(input.num_intermediates, conv_width, "KV retention elements")?;
        let kv_bytes = checked_mul(kv_elements, 2, "KV retention bytes")?;
        let gate_width = checked_mul(input.num_value_heads, 2, "gate width")?;
        let gate_elements = checked_mul(
            input.num_intermediates,
            gate_width,
            "gate retention elements",
        )?;
        let gate_bytes = checked_mul(gate_elements, 4, "gate retention bytes")?;
        (kv_bytes, gate_bytes)
    } else {
        (0, 0)
    };
    let retention_bytes_per_slot = checked_add(
        kv_retain_bytes,
        gate_retain_bytes,
        "retention bytes per slot",
    )?;
    // The base state (and lazy retention) covers every dummy-inclusive slot;
    // the intermediates and the checkpoint cover `spec_slots`.
    let base_bytes_per_slot = checked_add(
        state_bytes,
        retention_bytes_per_slot,
        "base state + retention bytes per slot",
    )?;
    let (bytes_per_layer_slot, spec_bytes_per_slot, h_verify_scratch_bytes) = if input
        .compact_lazy_h
    {
        let conv_spec = checked_mul(
            input.conv_bytes,
            state_copies - 1,
            "compact conv intermediate + checkpoint bytes per slot",
        )?;
        let per_layer_slot = checked_add(
            base_bytes_per_slot,
            conv_spec,
            "compact state + retention bytes per layer slot",
        )?;
        let verify_scratch = checked_mul(spec_slots, input.h_bytes, "compact h verify scratch")?;
        (per_layer_slot, conv_spec, verify_scratch)
    } else {
        let state_bytes_per_slot = checked_mul(state_bytes, state_copies, "state bytes per slot")?;
        let per_layer_slot = checked_add(
            state_bytes_per_slot,
            retention_bytes_per_slot,
            "state + retention bytes per layer slot",
        )?;
        let spec = checked_mul(
            state_bytes,
            state_copies - 1,
            "intermediate + checkpoint bytes per slot",
        )?;
        (per_layer_slot, spec, 0)
    };
    let bytes_per_layer = checked_add(
        checked_mul(
            total_slots,
            base_bytes_per_slot,
            "dummy-inclusive base bytes per layer",
        )?,
        checked_mul(
            spec_slots,
            spec_bytes_per_slot,
            "speculative bytes per layer",
        )?,
        "bytes per layer",
    )?;
    let layer_total_bytes = checked_mul(input.num_ssm_layers, bytes_per_layer, "all SSM layers")?;
    let total_bytes = checked_add(
        layer_total_bytes,
        h_verify_scratch_bytes,
        "all SSM layers + compact h verify scratch",
    )?;

    let h_state_allocation_bytes = checked_mul(total_slots, input.h_bytes, "h state allocation")?;
    let conv_state_allocation_bytes =
        checked_mul(total_slots, input.conv_bytes, "conv state allocation")?;
    let (h_intermediate_allocation_bytes, conv_intermediate_allocation_bytes) = if input.has_mtp {
        let intermediate_slots = checked_mul(
            spec_slots,
            input.num_intermediates,
            "intermediate allocation slots",
        )?;
        let h_inter = if input.compact_lazy_h {
            0
        } else {
            checked_mul(
                intermediate_slots,
                input.h_bytes,
                "h intermediate allocation",
            )?
        };
        let conv_inter = checked_mul(
            intermediate_slots,
            input.conv_bytes,
            "conv intermediate allocation",
        )?;
        (h_inter, conv_inter)
    } else {
        (0, 0)
    };
    let h_checkpoint_allocation_bytes = if input.has_mtp && !input.compact_lazy_h {
        checked_mul(spec_slots, input.h_bytes, "h checkpoint allocation")?
    } else {
        0
    };
    let conv_checkpoint_allocation_bytes = if input.has_mtp {
        checked_mul(spec_slots, input.conv_bytes, "conv checkpoint allocation")?
    } else {
        0
    };
    let kv_retain_allocation_bytes =
        checked_mul(total_slots, kv_retain_bytes, "KV retention allocation")?;
    let gate_retain_allocation_bytes =
        checked_mul(total_slots, gate_retain_bytes, "gate retention allocation")?;

    Ok(SsmPoolGeometry {
        total_slots,
        state_copies: if input.compact_lazy_h {
            1
        } else {
            state_copies
        },
        compact_lazy_h: input.compact_lazy_h,
        h_bytes: input.h_bytes,
        conv_bytes: input.conv_bytes,
        h_state_allocation_bytes,
        conv_state_allocation_bytes,
        h_intermediate_allocation_bytes,
        conv_intermediate_allocation_bytes,
        h_verify_scratch_bytes,
        spec_slots,
        h_checkpoint_allocation_bytes,
        conv_checkpoint_allocation_bytes,
        kv_retain_bytes,
        gate_retain_bytes,
        kv_retain_allocation_bytes,
        gate_retain_allocation_bytes,
        bytes_per_layer_slot,
        total_bytes,
    })
}
