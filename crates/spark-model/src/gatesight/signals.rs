// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time static GDN slow-head classification and L2-resident 6-GDN rolling signal ring.

use half::f16;

use super::{GATESIGHT_SIGNALS_PER_HEAD, GB10_L2_CACHE_BYTES, GateSightError, Result};

/// Upper clamp matching `ssm_preprocess.cu` (`fminf(a_log, 20.0f)`).
const A_LOG_CLAMP_MAX: f32 = 20.0;

/// Numerically stable `softplus(x) = ln(1 + exp(x))`.
#[inline]
#[must_use]
fn stable_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

/// Compute GateSight's zero-input static head half-life in tokens from checkpoint weights.
///
/// Formula (GateSight §2.1, arXiv:2610.14754):
/// `tau_static = ln(2) / (exp(A_log) * softplus(dt_bias))`.
///
/// Empirically matches full-dataset median GDN half-life ordering at Spearman 0.942 on
/// Qwen3.8-27B with zero forward passes.
#[must_use]
pub fn static_half_life_tokens(a_log: f32, dt_bias: f32) -> f32 {
    if !a_log.is_finite() || !dt_bias.is_finite() {
        return 0.0;
    }
    let a_val = a_log.min(A_LOG_CLAMP_MAX).exp();
    let dt = stable_softplus(dt_bias);
    let decay_rate = a_val * dt;
    if decay_rate <= f32::MIN_POSITIVE {
        return f32::INFINITY;
    }
    std::f32::consts::LN_2 / decay_rate
}

/// Static slow-head classification for a single GDN layer derived from `(ssm.a_log, ssm.dt_bias)`.
#[derive(Clone, Debug, PartialEq)]
pub struct GdnLayerSlowHeads {
    /// Model layer index of this GDN layer.
    pub layer_idx: usize,
    /// Per-v-head static half-life in tokens (`len == num_v_heads`).
    pub tau_static: Vec<f32>,
    /// V-head indices ordered slowest-first (descending `tau_static`) that exceed
    /// `half_life_threshold` (or the slowest fallback head when none exceed the threshold).
    pub slow_heads: Vec<usize>,
    /// Threshold (tokens) used to select `slow_heads`.
    pub half_life_threshold: f32,
}

impl GdnLayerSlowHeads {
    /// Classify slow heads for one GDN layer directly from its `a_log` and `dt_bias` vectors.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] if `a_log` and `dt_bias` differ in length
    /// or are empty, or [`GateSightError::InvalidConfig`] if `half_life_threshold <= 0`.
    pub fn classify(
        layer_idx: usize,
        a_log: &[f32],
        dt_bias: &[f32],
        half_life_threshold: f32,
    ) -> Result<Self> {
        if a_log.is_empty() {
            return Err(GateSightError::InvalidConfig(
                "a_log must not be empty".to_string(),
            ));
        }
        if a_log.len() != dt_bias.len() {
            return Err(GateSightError::DimensionMismatch {
                context: "a_log vs dt_bias",
                expected: a_log.len(),
                actual: dt_bias.len(),
            });
        }
        if !half_life_threshold.is_finite() || half_life_threshold <= 0.0 {
            return Err(GateSightError::InvalidConfig(format!(
                "half_life_threshold={half_life_threshold} must be > 0"
            )));
        }

        let tau_static: Vec<f32> = a_log
            .iter()
            .zip(dt_bias.iter())
            .map(|(&a, &b)| static_half_life_tokens(a, b))
            .collect();

        let mut ranked: Vec<(usize, f32)> = tau_static.iter().copied().enumerate().collect();
        ranked.sort_by(|(ia, ta), (ib, tb)| tb.total_cmp(ta).then_with(|| ia.cmp(ib)));

        let mut slow_heads: Vec<usize> = ranked
            .iter()
            .filter_map(|&(idx, tau)| (tau >= half_life_threshold).then_some(idx))
            .collect();
        if slow_heads.is_empty()
            && let Some(&(slowest_idx, _)) = ranked.first()
        {
            slow_heads.push(slowest_idx);
        }

        Ok(Self {
            layer_idx,
            tau_static,
            slow_heads,
            half_life_threshold,
        })
    }

    /// Number of value heads in this GDN layer.
    #[must_use]
    pub fn num_v_heads(&self) -> usize {
        self.tau_static.len()
    }

    /// Return all head indices ranked slowest-first (descending `tau_static`).
    #[must_use]
    pub fn ranked_heads_descending(&self) -> Vec<usize> {
        let mut ranked: Vec<(usize, f32)> = self.tau_static.iter().copied().enumerate().collect();
        ranked.sort_by(|(ia, ta), (ib, tb)| tb.total_cmp(ta).then_with(|| ia.cmp(ib)));
        ranked.into_iter().map(|(idx, _)| idx).collect()
    }
}

/// Load-time static slow-head table spanning all GDN layers of a hybrid model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSlowHeadTable {
    layers: Vec<GdnLayerSlowHeads>,
}

impl ModelSlowHeadTable {
    /// Construct a model-wide table from per-GDN-layer classifications.
    #[must_use]
    pub fn new(layers: Vec<GdnLayerSlowHeads>) -> Self {
        Self { layers }
    }

    /// Number of GDN layers registered in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// True when no GDN layers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Borrow all registered GDN layer classifications.
    #[must_use]
    pub fn layers(&self) -> &[GdnLayerSlowHeads] {
        &self.layers
    }

    /// Total number of classified slow heads across all GDN layers.
    #[must_use]
    pub fn total_slow_heads(&self) -> usize {
        self.layers.iter().map(|l| l.slow_heads.len()).sum()
    }

    /// Return up to `window` GDN layers immediately preceding `attn_layer_idx`, ordered
    /// oldest-to-newest (`layer_idx < attn_layer_idx`).
    #[must_use]
    pub fn preceding_gdn_layers(
        &self,
        attn_layer_idx: usize,
        window: usize,
    ) -> Vec<&GdnLayerSlowHeads> {
        let mut before: Vec<&GdnLayerSlowHeads> = self
            .layers
            .iter()
            .filter(|l| l.layer_idx < attn_layer_idx)
            .collect();
        before.sort_by_key(|l| l.layer_idx);
        if before.len() > window {
            before.drain(0..before.len() - window);
        }
        before
    }
}

/// One captured GDN `(beta, g, ||v||_2)` triple matching the CUDA kernel FP16 layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GdnSignalSample {
    /// Write gate `beta = sigmoid(b_raw) in (0, 1)`.
    pub beta: f32,
    /// Log-decay `g = -exp(A_log) * softplus(a_raw + dt_bias) <= 0`.
    pub log_decay_g: f32,
    /// Post-conv1d+SiLU value L2 norm `||v_t||_2 >= 0`.
    pub v_norm: f32,
}

impl GdnSignalSample {
    /// Construct a signal sample from scalar values.
    #[must_use]
    pub const fn new(beta: f32, log_decay_g: f32, v_norm: f32) -> Self {
        Self {
            beta,
            log_decay_g,
            v_norm,
        }
    }

    /// Encode this sample into the 3-element FP16 layout emitted by `ssm_preprocess.cu` / `qwen4_gdn_fi_glue.cu`.
    #[must_use]
    pub fn to_f16_triple(self) -> [f16; GATESIGHT_SIGNALS_PER_HEAD] {
        [
            f16::from_f32(self.beta),
            f16::from_f32(self.log_decay_g),
            f16::from_f32(self.v_norm),
        ]
    }

    /// Decode a 3-element FP16 slice into a [`GdnSignalSample`].
    #[must_use]
    pub fn from_f16_triple(raw: [f16; GATESIGHT_SIGNALS_PER_HEAD]) -> Self {
        Self {
            beta: raw[0].to_f32(),
            log_decay_g: raw[1].to_f32(),
            v_norm: raw[2].to_f32(),
        }
    }
}

/// Layout and slot-rotation descriptor for the L2-resident 6-GDN rolling signal ring.
///
/// Tensor layout on device: `[window_layers, chunk_tokens, num_v_heads, 3]` in FP16 (`2 B`).
/// For Qwen3.8-27B & Qwen3.8-Flash-Next (`6 x 8192 x 48 x 3 x 2 B = 14,155,776 B = 13.5 MiB`),
/// the entire ring fits within 42.2% of GB10's 32 MiB L2 cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnSignalRingDescriptor {
    window_layers: usize,
    chunk_tokens: usize,
    num_v_heads: usize,
    /// Circular write cursor (`total_writes % window_layers`).
    total_writes: usize,
    /// Model layer index currently occupying each physical ring slot.
    slot_layer_ids: Vec<Option<usize>>,
}

impl GdnSignalRingDescriptor {
    /// Create a validated rolling-ring descriptor.
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidConfig`] if any dimension is zero or byte size overflows.
    pub fn new(window_layers: usize, chunk_tokens: usize, num_v_heads: usize) -> Result<Self> {
        if window_layers == 0 || chunk_tokens == 0 || num_v_heads == 0 {
            return Err(GateSightError::InvalidConfig(format!(
                "ring dimensions must be non-zero: layers={window_layers}, tokens={chunk_tokens}, heads={num_v_heads}"
            )));
        }
        let _ = Self::checked_total_bytes(window_layers, chunk_tokens, num_v_heads)?;
        Ok(Self {
            window_layers,
            chunk_tokens,
            num_v_heads,
            total_writes: 0,
            slot_layer_ids: vec![None; window_layers],
        })
    }

    fn checked_total_bytes(
        window_layers: usize,
        chunk_tokens: usize,
        num_v_heads: usize,
    ) -> Result<usize> {
        window_layers
            .checked_mul(chunk_tokens)
            .and_then(|v| v.checked_mul(num_v_heads))
            .and_then(|v| v.checked_mul(GATESIGHT_SIGNALS_PER_HEAD))
            .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| GateSightError::InvalidConfig("signal ring byte size overflow".into()))
    }

    /// Number of rolling GDN layer slots in the ring.
    #[must_use]
    pub const fn window_layers(&self) -> usize {
        self.window_layers
    }

    /// Maximum token rows per slot (prefill chunk width).
    #[must_use]
    pub const fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }

    /// Number of GDN value heads per layer.
    #[must_use]
    pub const fn num_v_heads(&self) -> usize {
        self.num_v_heads
    }

    /// Number of FP16 elements in a single GDN layer slot (`chunk_tokens * num_v_heads * 3`).
    #[must_use]
    pub fn slot_elements(&self) -> usize {
        self.chunk_tokens * self.num_v_heads * GATESIGHT_SIGNALS_PER_HEAD
    }

    /// Size in bytes of a single GDN layer slot.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_elements() * std::mem::size_of::<u16>()
    }

    /// Total FP16 byte footprint across all `window_layers` slots.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.slot_bytes() * self.window_layers
    }

    /// True when the entire rolling ring fits inside `l2_budget_bytes` (e.g. GB10's 32 MiB L2).
    #[must_use]
    pub fn fits_in_l2(&self, l2_budget_bytes: usize) -> bool {
        self.total_bytes() <= l2_budget_bytes
    }

    /// Fraction of GB10's 32 MiB L2 occupied by this rolling ring.
    #[must_use]
    pub fn gb10_l2_fraction(&self) -> f64 {
        (self.total_bytes() as f64) / (GB10_L2_CACHE_BYTES as f64)
    }

    /// Byte offset from ring base pointer for slot `slot_idx`.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] if `slot_idx >= window_layers`.
    pub fn slot_byte_offset(&self, slot_idx: usize) -> Result<usize> {
        if slot_idx >= self.window_layers {
            return Err(GateSightError::DimensionMismatch {
                context: "signal ring slot_idx",
                expected: self.window_layers,
                actual: slot_idx,
            });
        }
        Ok(slot_idx * self.slot_bytes())
    }

    /// Reserve the next circular slot for GDN layer `gdn_layer_idx` and return `(slot_idx, byte_offset)`.
    pub fn record_gdn_layer_write(&mut self, gdn_layer_idx: usize) -> (usize, usize) {
        let slot = self.total_writes % self.window_layers;
        self.slot_layer_ids[slot] = Some(gdn_layer_idx);
        self.total_writes += 1;
        (slot, slot * self.slot_bytes())
    }

    /// Reset ring slot tracking between independent prefill chunks / sequences.
    pub fn reset(&mut self) {
        self.total_writes = 0;
        self.slot_layer_ids.fill(None);
    }

    /// Return populated `(slot_idx, gdn_layer_idx)` pairs ordered oldest-to-newest for the
    /// upcoming FullAttention layer.
    #[must_use]
    pub fn ordered_active_slots(&self) -> Vec<(usize, usize)> {
        let count = self.total_writes.min(self.window_layers);
        let oldest = if self.total_writes >= self.window_layers {
            self.total_writes % self.window_layers
        } else {
            0
        };
        let mut out = Vec::with_capacity(count);
        for offset in 0..count {
            let slot = (oldest + offset) % self.window_layers;
            if let Some(layer_idx) = self.slot_layer_ids[slot] {
                out.push((slot, layer_idx));
            }
        }
        out
    }
}

/// Host-resident FP16 mirror of the rolling GDN signal ring for CPU scoring and verification.
#[derive(Clone, Debug)]
pub struct GdnSignalRingHost {
    descriptor: GdnSignalRingDescriptor,
    data: Vec<f16>,
}

impl GdnSignalRingHost {
    /// Allocate a zero-initialized host rolling ring matching `descriptor`.
    #[must_use]
    pub fn new(descriptor: GdnSignalRingDescriptor) -> Self {
        let total_elems = descriptor.slot_elements() * descriptor.window_layers();
        Self {
            descriptor,
            data: vec![f16::ZERO; total_elems],
        }
    }

    /// Borrow the underlying ring descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &GdnSignalRingDescriptor {
        &self.descriptor
    }

    /// Write a single GDN layer's `(tokens x num_v_heads)` samples into the next ring slot.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] if `samples.len()` exceeds
    /// `chunk_tokens * num_v_heads` or is not divisible by `num_v_heads`.
    pub fn push_layer_samples(
        &mut self,
        gdn_layer_idx: usize,
        num_tokens: usize,
        samples: &[GdnSignalSample],
    ) -> Result<usize> {
        let expected = num_tokens * self.descriptor.num_v_heads();
        if samples.len() != expected {
            return Err(GateSightError::DimensionMismatch {
                context: "push_layer_samples",
                expected,
                actual: samples.len(),
            });
        }
        if num_tokens > self.descriptor.chunk_tokens() {
            return Err(GateSightError::DimensionMismatch {
                context: "push_layer_samples chunk_tokens",
                expected: self.descriptor.chunk_tokens(),
                actual: num_tokens,
            });
        }
        let (slot, _) = self.descriptor.record_gdn_layer_write(gdn_layer_idx);
        let slot_base = slot * self.descriptor.slot_elements();
        for (idx, sample) in samples.iter().enumerate() {
            let triple = sample.to_f16_triple();
            let base = slot_base + idx * GATESIGHT_SIGNALS_PER_HEAD;
            self.data[base] = triple[0];
            self.data[base + 1] = triple[1];
            self.data[base + 2] = triple[2];
        }
        Ok(slot)
    }

    /// Read one decoded [`GdnSignalSample`] at `(slot, token, v_head)`.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] on out-of-bounds indices.
    pub fn sample_at(&self, slot: usize, token: usize, v_head: usize) -> Result<GdnSignalSample> {
        if slot >= self.descriptor.window_layers()
            || token >= self.descriptor.chunk_tokens()
            || v_head >= self.descriptor.num_v_heads()
        {
            return Err(GateSightError::DimensionMismatch {
                context: "sample_at bounds",
                expected: self.descriptor.num_v_heads(),
                actual: v_head,
            });
        }
        let base = slot * self.descriptor.slot_elements()
            + (token * self.descriptor.num_v_heads() + v_head) * GATESIGHT_SIGNALS_PER_HEAD;
        Ok(GdnSignalSample::from_f16_triple([
            self.data[base],
            self.data[base + 1],
            self.data[base + 2],
        ]))
    }
}
