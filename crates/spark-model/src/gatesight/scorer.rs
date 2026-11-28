// SPDX-License-Identifier: AGPL-3.0-only

//! Path-A zero-training `block_size = 16` scorer and Path-B `867 -> 64 -> 1` MLP sidecar loader/scorer.

use std::fs;
use std::path::Path;

use ring::digest::{Context as Sha256Context, SHA256};

use super::signals::{GdnSignalRingHost, ModelSlowHeadTable};
use super::{
    GATESIGHT_AGE_FEATURES, GATESIGHT_BLOCK_SIZE, GATESIGHT_SIGNALS_PER_HEAD, GateSightError,
    Result,
};

/// Binary magic header for authenticated GateSight Path-B MLP sidecar files.
pub const GATESIGHT_SIDECAR_MAGIC: [u8; 8] = *b"ATLGTS01";

const SIDECAR_FORMAT_VERSION: u32 = 1;
const SIDECAR_HEADER_BYTES: usize = 8 + 4 * 8 + 32; // magic + 8 u32 fields + sha256

/// Decay-weighting rule used by the Path-A zero-training heuristic scorer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DecayWeighting {
    /// Pure GateSight slow-head write strength `beta * ||v||_2` (arXiv:2610.14754 Table 6).
    #[default]
    PureWriteNorm,
    /// Forgetting-boosted score `beta * ||v||_2 * (1 + lambda * (-g))` exploiting the positive
    /// `+0.106` correlation between GDN forgetting `-g` and FullAttention importance.
    ForgettingBoost { lambda: f32 },
}

/// Path-A 100% zero-training scorer operating on slow GDN heads at `block_size = 16` granularity.
///
/// Each 16-token KV block receives the **maximum** token score across its constituent tokens so
/// a single high-salience needle token preserves its entire self-contained `V^T` block.
#[derive(Clone, Debug)]
pub struct ZeroTrainingBlockScorer {
    block_size: usize,
    decay_weighting: DecayWeighting,
}

impl Default for ZeroTrainingBlockScorer {
    fn default() -> Self {
        const { assert!(GATESIGHT_BLOCK_SIZE > 0) };
        Self {
            block_size: GATESIGHT_BLOCK_SIZE,
            decay_weighting: DecayWeighting::PureWriteNorm,
        }
    }
}

impl ZeroTrainingBlockScorer {
    /// Create a zero-training block scorer for `block_size` (must equal `16` for BitDecoding-FP4).
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidConfig`] if `block_size == 0` or `lambda` is invalid.
    pub fn new(block_size: usize, decay_weighting: DecayWeighting) -> Result<Self> {
        if block_size == 0 {
            return Err(GateSightError::InvalidConfig(
                "block_size must be > 0".to_string(),
            ));
        }
        if let DecayWeighting::ForgettingBoost { lambda } = decay_weighting
            && (!lambda.is_finite() || lambda < 0.0)
        {
            return Err(GateSightError::InvalidConfig(format!(
                "forgetting boost lambda={lambda} must be finite and >= 0"
            )));
        }
        Ok(Self {
            block_size,
            decay_weighting,
        })
    }

    /// Block granularity used for pooling.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Compute per-token zero-training write-strength scores from the active rolling ring slots
    /// and the model's load-time static slow-head table.
    ///
    /// # Errors
    /// Returns an error if the ring contains no active slots or `num_tokens` exceeds the ring chunk.
    pub fn score_tokens(
        &self,
        ring: &GdnSignalRingHost,
        slow_heads: &ModelSlowHeadTable,
        num_tokens: usize,
    ) -> Result<Vec<f32>> {
        if num_tokens > ring.descriptor().chunk_tokens() {
            return Err(GateSightError::DimensionMismatch {
                context: "ZeroTrainingBlockScorer::score_tokens",
                expected: ring.descriptor().chunk_tokens(),
                actual: num_tokens,
            });
        }
        let active_slots = ring.descriptor().ordered_active_slots();
        if active_slots.is_empty() {
            return Err(GateSightError::InvalidConfig(
                "signal ring has no populated GDN slots".to_string(),
            ));
        }

        // Resolve `(slot, slow_head_slice)` pairs; fall back to all heads if a layer is absent.
        let all_heads: Vec<usize> = (0..ring.descriptor().num_v_heads()).collect();
        let mut pairs: Vec<(usize, &[usize])> = Vec::with_capacity(active_slots.len());
        let mut total_head_terms = 0usize;
        for (slot, layer_idx) in active_slots {
            let heads = slow_heads
                .layers()
                .iter()
                .find(|l| l.layer_idx == layer_idx)
                .map_or(all_heads.as_slice(), |l| l.slow_heads.as_slice());
            total_head_terms += heads.len();
            pairs.push((slot, heads));
        }
        if total_head_terms == 0 {
            return Ok(vec![0.0; num_tokens]);
        }
        let inv_count = 1.0f32 / (total_head_terms as f32);

        let mut scores = Vec::with_capacity(num_tokens);
        for tok in 0..num_tokens {
            let mut sum = 0.0f32;
            for &(slot, heads) in &pairs {
                for &vh in heads {
                    let s = ring.sample_at(slot, tok, vh)?;
                    let base = s.beta.max(0.0) * s.v_norm.max(0.0);
                    let weighted = match self.decay_weighting {
                        DecayWeighting::PureWriteNorm => base,
                        DecayWeighting::ForgettingBoost { lambda } => {
                            let forget = (-s.log_decay_g).max(0.0);
                            base * (1.0 + lambda * forget)
                        }
                    };
                    sum += weighted;
                }
            }
            scores.push(sum * inv_count);
        }
        Ok(scores)
    }

    /// Pool per-token scores into `ceil(token_scores.len() / block_size)` block scores using max pooling.
    #[must_use]
    pub fn pool_blocks(&self, token_scores: &[f32]) -> Vec<f32> {
        token_scores
            .chunks(self.block_size)
            .map(|chunk| chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            .collect()
    }

    /// Score all tokens in the ring and immediately max-pool into `block_size`-aligned block scores.
    ///
    /// # Errors
    /// Propagates validation errors from [`Self::score_tokens`].
    pub fn score_blocks(
        &self,
        ring: &GdnSignalRingHost,
        slow_heads: &ModelSlowHeadTable,
        num_tokens: usize,
    ) -> Result<Vec<f32>> {
        let token_scores = self.score_tokens(ring, slow_heads, num_tokens)?;
        Ok(self.pool_blocks(&token_scores))
    }
}

/// Hidden activation function for Path-B GateSight MLPs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MlpActivation {
    /// Standard ReLU (`max(0, x)`), default in GateSight `867 -> 64 -> 1`.
    #[default]
    Relu,
    /// SiLU (`x * sigmoid(x)`).
    Silu,
}

impl MlpActivation {
    const fn wire_tag(self) -> u32 {
        match self {
            Self::Relu => 0,
            Self::Silu => 1,
        }
    }

    fn from_wire_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::Relu),
            1 => Ok(Self::Silu),
            other => Err(GateSightError::InvalidSidecar(format!(
                "unknown MLP activation tag {other}"
            ))),
        }
    }

    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Silu => x / (1.0 + (-x).exp()),
        }
    }
}

/// Weights for one `(attn_layer, kv_head)` `input_dim -> hidden_dim -> 1` sigmoid MLP.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadMlpWeights {
    /// First-layer weights in row-major `[hidden_dim, input_dim]` order.
    pub w1: Vec<f32>,
    /// First-layer biases `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Output projection weights `[hidden_dim]`.
    pub w2: Vec<f32>,
    /// Output scalar bias.
    pub b2: f32,
}

impl HeadMlpWeights {
    /// Validate tensor shapes and finiteness against `(input_dim, hidden_dim)`.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] on any shape mismatch or
    /// [`GateSightError::InvalidSidecar`] if any weight or bias is NaN/Inf.
    pub fn validate(&self, input_dim: usize, hidden_dim: usize) -> Result<()> {
        let expected_w1 = hidden_dim.checked_mul(input_dim).ok_or_else(|| {
            GateSightError::InvalidSidecar("HeadMlpWeights w1 size overflow".to_string())
        })?;
        if self.w1.len() != expected_w1 {
            return Err(GateSightError::DimensionMismatch {
                context: "HeadMlpWeights::w1",
                expected: expected_w1,
                actual: self.w1.len(),
            });
        }
        if self.b1.len() != hidden_dim {
            return Err(GateSightError::DimensionMismatch {
                context: "HeadMlpWeights::b1",
                expected: hidden_dim,
                actual: self.b1.len(),
            });
        }
        if self.w2.len() != hidden_dim {
            return Err(GateSightError::DimensionMismatch {
                context: "HeadMlpWeights::w2",
                expected: hidden_dim,
                actual: self.w2.len(),
            });
        }
        if !self.b2.is_finite()
            || !self.w1.iter().all(|v| v.is_finite())
            || !self.b1.iter().all(|v| v.is_finite())
            || !self.w2.iter().all(|v| v.is_finite())
        {
            return Err(GateSightError::InvalidSidecar(
                "MLP weights and biases must all be finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Evaluate `sigmoid(w2 . act(w1 . x + b1) + b2)` for one feature vector `x`.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] if `x`, `w1`, `b1`, or `w2` shapes disagree.
    pub fn score_feature(&self, x: &[f32], activation: MlpActivation) -> Result<f32> {
        let hidden_dim = self.b1.len();
        if hidden_dim == 0 {
            return Err(GateSightError::InvalidConfig(
                "MLP hidden_dim must be > 0".to_string(),
            ));
        }
        if self.w2.len() != hidden_dim {
            return Err(GateSightError::DimensionMismatch {
                context: "HeadMlpWeights::w2",
                expected: hidden_dim,
                actual: self.w2.len(),
            });
        }
        let expected_w1 = hidden_dim.checked_mul(x.len()).ok_or_else(|| {
            GateSightError::InvalidConfig("HeadMlpWeights::w1 size overflow".to_string())
        })?;
        if self.w1.len() != expected_w1 {
            return Err(GateSightError::DimensionMismatch {
                context: "HeadMlpWeights::w1",
                expected: expected_w1,
                actual: self.w1.len(),
            });
        }
        let input_dim = x.len();
        let mut logit = self.b2;
        for h in 0..hidden_dim {
            let row = &self.w1[h * input_dim..(h + 1) * input_dim];
            let mut acc = self.b1[h];
            for (&w, &xi) in row.iter().zip(x.iter()) {
                acc += w * xi;
            }
            logit += self.w2[h] * activation.apply(acc);
        }
        Ok(1.0f32 / (1.0f32 + (-logit).exp()))
    }
}

/// Authenticated Path-B GateSight sidecar containing all per-`(attn_layer, kv_head)` MLPs
/// and optional cross-layer attention output-gate budget weights `w_l`.
#[derive(Clone, Debug, PartialEq)]
pub struct GateSightMlpSidecar {
    pub num_attn_layers: usize,
    pub num_kv_heads: usize,
    pub window_gdn_layers: usize,
    pub num_v_heads: usize,
    pub hidden_dim: usize,
    pub activation: MlpActivation,
    /// Per-attention-layer output-gate importance weights `w_l` (GateSight §4.2), length `num_attn_layers`.
    pub layer_gate_weights: Vec<f32>,
    /// Flattened `[num_attn_layers * num_kv_heads]` per-head MLPs.
    pub heads: Vec<HeadMlpWeights>,
}

impl GateSightMlpSidecar {
    /// Expected input feature dimensionality (`window_gdn_layers * num_v_heads * 3 + 3`, 867 on Qwen3.8).
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.window_gdn_layers * self.num_v_heads * GATESIGHT_SIGNALS_PER_HEAD
            + GATESIGHT_AGE_FEATURES
    }

    /// Validate all dimensions and per-head weights.
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidSidecar`] or [`GateSightError::DimensionMismatch`].
    pub fn validate(&self) -> Result<()> {
        if self.num_attn_layers == 0
            || self.num_kv_heads == 0
            || self.window_gdn_layers == 0
            || self.num_v_heads == 0
            || self.hidden_dim == 0
        {
            return Err(GateSightError::InvalidSidecar(
                "all sidecar dimensions must be > 0".to_string(),
            ));
        }
        if self.layer_gate_weights.len() != self.num_attn_layers {
            return Err(GateSightError::DimensionMismatch {
                context: "layer_gate_weights",
                expected: self.num_attn_layers,
                actual: self.layer_gate_weights.len(),
            });
        }
        if !self
            .layer_gate_weights
            .iter()
            .all(|w| w.is_finite() && *w >= 0.0)
        {
            return Err(GateSightError::InvalidSidecar(
                "layer_gate_weights must all be finite and >= 0".to_string(),
            ));
        }
        let expected_heads = self
            .num_attn_layers
            .checked_mul(self.num_kv_heads)
            .ok_or_else(|| GateSightError::InvalidSidecar("head count overflow".to_string()))?;
        if self.heads.len() != expected_heads {
            return Err(GateSightError::DimensionMismatch {
                context: "sidecar heads",
                expected: expected_heads,
                actual: self.heads.len(),
            });
        }
        let input_dim = self
            .window_gdn_layers
            .checked_mul(self.num_v_heads)
            .and_then(|v| v.checked_mul(GATESIGHT_SIGNALS_PER_HEAD))
            .and_then(|v| v.checked_add(GATESIGHT_AGE_FEATURES))
            .ok_or_else(|| GateSightError::InvalidSidecar("input_dim overflow".to_string()))?;
        for head in &self.heads {
            head.validate(input_dim, self.hidden_dim)?;
        }
        Ok(())
    }

    /// Borrow the MLP weights for `(attn_layer, kv_head)`.
    ///
    /// # Errors
    /// Returns [`GateSightError::DimensionMismatch`] if indices are out of bounds.
    pub fn head(&self, attn_layer: usize, kv_head: usize) -> Result<&HeadMlpWeights> {
        if attn_layer >= self.num_attn_layers {
            return Err(GateSightError::DimensionMismatch {
                context: "attn_layer",
                expected: self.num_attn_layers,
                actual: attn_layer,
            });
        }
        if kv_head >= self.num_kv_heads {
            return Err(GateSightError::DimensionMismatch {
                context: "kv_head",
                expected: self.num_kv_heads,
                actual: kv_head,
            });
        }
        Ok(&self.heads[attn_layer * self.num_kv_heads + kv_head])
    }

    /// Serialize this sidecar into the SHA-256-authenticated `ATLGTS01` binary format.
    ///
    /// The SHA-256 digest covers both the 40-byte header prefix (`magic`, `version`, dimensions,
    /// `activation`, `payload_len`) and the full float payload so neither header fields nor weights
    /// can be altered undetected.
    ///
    /// # Errors
    /// Propagates validation errors if dimensions are inconsistent or exceed `u32::MAX`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let to_u32 = |val: usize, field: &'static str| -> Result<u32> {
            u32::try_from(val).map_err(|_| {
                GateSightError::InvalidSidecar(format!("{field}={val} exceeds u32::MAX"))
            })
        };
        let input_dim = self.input_dim();
        let floats_per_head = self
            .hidden_dim
            .checked_mul(input_dim)
            .and_then(|v| {
                self.hidden_dim
                    .checked_mul(2)
                    .and_then(|b| b.checked_add(1))
                    .and_then(|b| v.checked_add(b))
            })
            .ok_or_else(|| {
                GateSightError::InvalidSidecar("head parameter count overflow".into())
            })?;
        let payload_floats = self
            .heads
            .len()
            .checked_mul(floats_per_head)
            .and_then(|v| self.num_attn_layers.checked_add(v))
            .ok_or_else(|| GateSightError::InvalidSidecar("payload float count overflow".into()))?;
        let payload_bytes = payload_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| GateSightError::InvalidSidecar("payload byte overflow".into()))?;

        let mut payload = Vec::with_capacity(payload_bytes);
        for &w in &self.layer_gate_weights {
            payload.extend_from_slice(&w.to_le_bytes());
        }
        for head in &self.heads {
            for &v in &head.w1 {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &head.b1 {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &head.w2 {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            payload.extend_from_slice(&head.b2.to_le_bytes());
        }

        let mut header_prefix = Vec::with_capacity(40);
        header_prefix.extend_from_slice(&GATESIGHT_SIDECAR_MAGIC);
        header_prefix.extend_from_slice(&SIDECAR_FORMAT_VERSION.to_le_bytes());
        header_prefix
            .extend_from_slice(&to_u32(self.num_attn_layers, "num_attn_layers")?.to_le_bytes());
        header_prefix.extend_from_slice(&to_u32(self.num_kv_heads, "num_kv_heads")?.to_le_bytes());
        header_prefix
            .extend_from_slice(&to_u32(self.window_gdn_layers, "window_gdn_layers")?.to_le_bytes());
        header_prefix.extend_from_slice(&to_u32(self.num_v_heads, "num_v_heads")?.to_le_bytes());
        header_prefix.extend_from_slice(&to_u32(self.hidden_dim, "hidden_dim")?.to_le_bytes());
        header_prefix.extend_from_slice(&self.activation.wire_tag().to_le_bytes());
        header_prefix.extend_from_slice(&to_u32(payload.len(), "payload_len")?.to_le_bytes());

        let mut sha = Sha256Context::new(&SHA256);
        sha.update(&header_prefix);
        sha.update(&payload);
        let digest = sha.finish();

        let mut out = Vec::with_capacity(SIDECAR_HEADER_BYTES + payload.len());
        out.extend_from_slice(&header_prefix);
        out.extend_from_slice(digest.as_ref());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Parse and cryptographically verify an `ATLGTS01` sidecar byte slice.
    ///
    /// # Errors
    /// Returns [`GateSightError::InvalidSidecar`] on magic, version, length, or SHA-256 mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SIDECAR_HEADER_BYTES {
            return Err(GateSightError::InvalidSidecar(format!(
                "truncated sidecar: {} bytes < header {}",
                bytes.len(),
                SIDECAR_HEADER_BYTES
            )));
        }
        if bytes[0..8] != GATESIGHT_SIDECAR_MAGIC {
            return Err(GateSightError::InvalidSidecar(
                "invalid magic header (expected ATLGTS01)".to_string(),
            ));
        }

        let read_u32 = |offset: usize| -> u32 {
            let mut raw = [0u8; 4];
            raw.copy_from_slice(&bytes[offset..offset + 4]);
            u32::from_le_bytes(raw)
        };

        let version = read_u32(8);
        if version != SIDECAR_FORMAT_VERSION {
            return Err(GateSightError::InvalidSidecar(format!(
                "unsupported sidecar version {version}"
            )));
        }
        let num_attn_layers = read_u32(12) as usize;
        let num_kv_heads = read_u32(16) as usize;
        let window_gdn_layers = read_u32(20) as usize;
        let num_v_heads = read_u32(24) as usize;
        let hidden_dim = read_u32(28) as usize;
        let activation = MlpActivation::from_wire_tag(read_u32(32))?;
        let payload_len = read_u32(36) as usize;
        let expected_digest = &bytes[40..72];
        let payload = &bytes[SIDECAR_HEADER_BYTES..];

        if payload.len() != payload_len {
            return Err(GateSightError::InvalidSidecar(format!(
                "payload length mismatch: header declares {payload_len} B, actual {} B",
                payload.len()
            )));
        }

        let mut sha = Sha256Context::new(&SHA256);
        sha.update(&bytes[0..40]);
        sha.update(payload);
        let actual_digest = sha.finish();
        if actual_digest.as_ref() != expected_digest {
            return Err(GateSightError::InvalidSidecar(
                "SHA-256 header+payload checksum mismatch".to_string(),
            ));
        }

        let input_dim = window_gdn_layers
            .checked_mul(num_v_heads)
            .and_then(|v| v.checked_mul(GATESIGHT_SIGNALS_PER_HEAD))
            .and_then(|v| v.checked_add(GATESIGHT_AGE_FEATURES))
            .ok_or_else(|| GateSightError::InvalidSidecar("input_dim overflow".into()))?;
        let total_heads = num_attn_layers
            .checked_mul(num_kv_heads)
            .ok_or_else(|| GateSightError::InvalidSidecar("head count overflow".into()))?;
        let bias_and_out = hidden_dim
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                GateSightError::InvalidSidecar("head parameter count overflow".into())
            })?;
        let floats_per_head = hidden_dim
            .checked_mul(input_dim)
            .and_then(|v| v.checked_add(bias_and_out))
            .ok_or_else(|| {
                GateSightError::InvalidSidecar("head parameter count overflow".into())
            })?;
        let head_floats = total_heads
            .checked_mul(floats_per_head)
            .ok_or_else(|| GateSightError::InvalidSidecar("payload float count overflow".into()))?;
        let expected_bytes = num_attn_layers
            .checked_add(head_floats)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| GateSightError::InvalidSidecar("payload byte overflow".into()))?;

        if payload.len() != expected_bytes {
            return Err(GateSightError::InvalidSidecar(format!(
                "payload byte count {} does not match header dimensions ({expected_bytes} B)",
                payload.len()
            )));
        }

        let mut cursor = 0usize;
        let mut take_f32_vec = |count: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let mut raw = [0u8; 4];
                raw.copy_from_slice(&payload[cursor..cursor + 4]);
                cursor += 4;
                v.push(f32::from_le_bytes(raw));
            }
            v
        };

        let layer_gate_weights = take_f32_vec(num_attn_layers);
        let mut heads = Vec::with_capacity(total_heads);
        for _ in 0..total_heads {
            let w1 = take_f32_vec(hidden_dim * input_dim);
            let b1 = take_f32_vec(hidden_dim);
            let w2 = take_f32_vec(hidden_dim);
            let b2 = take_f32_vec(1)[0];
            heads.push(HeadMlpWeights { w1, b1, w2, b2 });
        }

        let sidecar = Self {
            num_attn_layers,
            num_kv_heads,
            window_gdn_layers,
            num_v_heads,
            hidden_dim,
            activation,
            layer_gate_weights,
            heads,
        };
        sidecar.validate()?;
        Ok(sidecar)
    }

    /// Load and verify an `ATLGTS01` sidecar file from disk.
    ///
    /// # Errors
    /// Returns [`GateSightError::Io`] on read errors or [`GateSightError::InvalidSidecar`] on corruption.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .map_err(|err| GateSightError::Io(format!("read {}: {err}", path.display())))?;
        Self::from_bytes(&bytes)
    }
}

/// Assemble the exact GateSight Eq. (3) feature vector (`867` floats for `6 x 48 x 3 + 3`)
/// for token `local_token_idx` in `ring` whose global prompt index is `global_pos` inside a
/// prompt of known total length `total_prompt_tokens`.
///
/// # Errors
/// Returns [`GateSightError::DimensionMismatch`] if the ring does not have `window_gdn_layers`
/// populated slots or indices are out of bounds.
pub fn build_token_features(
    ring: &GdnSignalRingHost,
    local_token_idx: usize,
    global_pos: usize,
    total_prompt_tokens: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let slots = ring.descriptor().ordered_active_slots();
    if slots.len() != ring.descriptor().window_layers() {
        return Err(GateSightError::DimensionMismatch {
            context: "build_token_features active GDN slots",
            expected: ring.descriptor().window_layers(),
            actual: slots.len(),
        });
    }
    let num_v_heads = ring.descriptor().num_v_heads();
    let total_dim = slots.len() * num_v_heads * GATESIGHT_SIGNALS_PER_HEAD + GATESIGHT_AGE_FEATURES;
    out.clear();
    out.reserve(total_dim);

    for (slot, _) in slots {
        for vh in 0..num_v_heads {
            let s = ring.sample_at(slot, local_token_idx, vh)?;
            out.push(s.beta);
            out.push(s.log_decay_g);
            out.push(s.v_norm);
        }
    }

    // Exact relative age features from GateSight Eq. (3):
    //   (i / n, (n - i) / n, ln(1 + n - i) / ln(1 + n))
    // Because `total_prompt_tokens` is known at prefill entry, these are 100% exact mid-prefill.
    let n = total_prompt_tokens.max(1) as f32;
    let pos = global_pos.min(total_prompt_tokens) as f32;
    let dist = (n - pos).max(0.0);
    let log_denom = (1.0f32 + n).ln().max(f32::EPSILON);
    out.push(pos / n);
    out.push(dist / n);
    out.push((1.0f32 + dist).ln() / log_denom);
    Ok(())
}

/// Path-B scorer evaluating a loaded [`GateSightMlpSidecar`] over a populated [`GdnSignalRingHost`].
#[derive(Clone, Debug)]
pub struct GateSightMlpScorer {
    sidecar: GateSightMlpSidecar,
}

impl GateSightMlpScorer {
    /// Wrap a validated [`GateSightMlpSidecar`].
    ///
    /// # Errors
    /// Propagates [`GateSightMlpSidecar::validate`] errors.
    pub fn new(sidecar: GateSightMlpSidecar) -> Result<Self> {
        sidecar.validate()?;
        Ok(Self { sidecar })
    }

    /// Borrow the underlying sidecar weights.
    #[must_use]
    pub fn sidecar(&self) -> &GateSightMlpSidecar {
        &self.sidecar
    }

    /// Score all `chunk_len` tokens in `ring` (starting at `chunk_global_offset` inside a prompt
    /// of total length `total_prompt_tokens`) for attention layer `attn_layer`, aggregating across
    /// `num_kv_heads` via mean importance (matching layer-level KV retention).
    ///
    /// # Errors
    /// Returns an error on ring/sidecar dimension mismatches.
    pub fn score_layer_tokens(
        &self,
        attn_layer: usize,
        ring: &GdnSignalRingHost,
        chunk_global_offset: usize,
        chunk_len: usize,
        total_prompt_tokens: usize,
    ) -> Result<Vec<f32>> {
        if attn_layer >= self.sidecar.num_attn_layers {
            return Err(GateSightError::DimensionMismatch {
                context: "GateSightMlpScorer::attn_layer",
                expected: self.sidecar.num_attn_layers,
                actual: attn_layer,
            });
        }
        if ring.descriptor().window_layers() != self.sidecar.window_gdn_layers {
            return Err(GateSightError::DimensionMismatch {
                context: "ring window_layers vs sidecar",
                expected: self.sidecar.window_gdn_layers,
                actual: ring.descriptor().window_layers(),
            });
        }
        if ring.descriptor().num_v_heads() != self.sidecar.num_v_heads {
            return Err(GateSightError::DimensionMismatch {
                context: "ring num_v_heads vs sidecar",
                expected: self.sidecar.num_v_heads,
                actual: ring.descriptor().num_v_heads(),
            });
        }

        let mut feat_buf = Vec::with_capacity(self.sidecar.input_dim());
        let mut scores = Vec::with_capacity(chunk_len);
        let inv_heads = 1.0f32 / (self.sidecar.num_kv_heads as f32);

        for local_i in 0..chunk_len {
            let global_pos = chunk_global_offset + local_i;
            build_token_features(
                ring,
                local_i,
                global_pos,
                total_prompt_tokens,
                &mut feat_buf,
            )?;
            let mut sum = 0.0f32;
            for kv_h in 0..self.sidecar.num_kv_heads {
                let mlp = self.sidecar.head(attn_layer, kv_h)?;
                sum += mlp.score_feature(&feat_buf, self.sidecar.activation)?;
            }
            scores.push(sum * inv_heads);
        }
        Ok(scores)
    }
}
