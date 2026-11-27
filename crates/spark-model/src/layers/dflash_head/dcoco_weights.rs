// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed authenticated ``ATLDC001`` sidecar loader for Flux-DCOCO-v2 weights.
//!
//! Loads offline-trained BF16 DCOCO-v2 corrector sidecars produced by
//! ``tools/train_dcoco_corrector.py`` and uploads all 10 tensors transactionally
//! to GPU memory for [`super::dcoco::DcocoCorrector`].
//!
//! Security and fail-closed admission invariants:
//!   - Gated on three environment variables (`ATLAS_DCOCO_SIDECAR`,
//!     `ATLAS_DCOCO_SIDECAR_SHA256`, `ATLAS_DCOCO_SIDECAR_SIZE`). When all three
//!     are absent or empty, loading is disabled (`Ok(None)`, zero allocations,
//!     bit-identical baseline).
//!   - Partial configuration or non-UTF-8 environment values fail closed.
//!   - Linux `O_NOFOLLOW` open, canonical absolute path enforcement, `nlink == 1`,
//!     pre/open/post file-identity verification, and whole-file SHA-256 + byte-size
//!     authentication prior to parsing.
//!   - Embedded SHA-256 verification over header prefix + payload, strict ABI
//!     dimension validation (`rank == 128`, `mlp_hidden == 256`, `1 <= top_k <= 16`,
//!     finite positive `ln_eps`), and full BF16 finiteness scan rejecting any
//!     `NaN` or `±Inf` weight element prior to GPU allocation.
//!   - Transactional GPU upload via [`DeviceAllocations`] rolling back partial
//!     allocations automatically on any allocation or H2D copy error.

use std::ffi::OsStr;
use std::fmt;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use ring::digest;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::dcoco::{
    DCOCO_MLP_HIDDEN, DCOCO_MLP_IN, DCOCO_RANK, DcocoCpuWeights, DcocoSpec, DcocoWeights,
};
use crate::device_allocations::DeviceAllocations;
use crate::weight_map::DenseWeight;

/// 8-byte binary magic prefix identifying Flux-DCOCO-v2 sidecars.
pub const DCOCO_SIDECAR_MAGIC: [u8; 8] = *b"ATLDC001";

/// Supported `ATLDC001` binary schema version.
pub const DCOCO_SIDECAR_VERSION: u32 = 1;

/// Byte length of the hashed header prefix (`magic..=payload_bytes`).
pub const DCOCO_SIDECAR_HEADER_PREFIX_BYTES: usize = 40;

/// Total byte length of the `ATLDC001` binary header (`prefix + embedded_sha256`).
pub const DCOCO_SIDECAR_HEADER_BYTES: usize = 72;

/// Environment variable specifying the canonical path to the DCOCO sidecar.
pub const DCOCO_SIDECAR_ENV: &str = "ATLAS_DCOCO_SIDECAR";

/// Environment variable specifying the expected 64-hex-char whole-file SHA-256 digest.
pub const DCOCO_SIDECAR_SHA256_ENV: &str = "ATLAS_DCOCO_SIDECAR_SHA256";

/// Environment variable specifying the exact sidecar file size in bytes.
pub const DCOCO_SIDECAR_SIZE_ENV: &str = "ATLAS_DCOCO_SIDECAR_SIZE";

/// Hard upper bound on sidecar size to prevent untrusted memory exhaustion (256 MiB).
const MAX_DCOCO_SIDECAR_BYTES: u64 = 256 << 20;

/// BF16 exponent mask (`0x7F80`); any value with all 8 exponent bits set is `NaN` or `±Inf`.
const BF16_EXP_MASK: u16 = 0x7F80;

// ── Fail-closed environment admission ───────────────────────────────────────

/// Immutable operator request for an authenticated `ATLDC001` DCOCO sidecar load.
#[derive(Clone, PartialEq, Eq)]
pub struct DcocoSidecarRequest {
    path: PathBuf,
    sha256: [u8; 32],
    size: u64,
}

impl fmt::Debug for DcocoSidecarRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DcocoSidecarRequest")
            .field("path", &self.path)
            .field("sha256_prefix", &hex_prefix(&self.sha256))
            .field("size", &self.size)
            .finish()
    }
}

impl DcocoSidecarRequest {
    /// Read the three-field DCOCO sidecar request from the environment.
    ///
    /// Returns `Ok(None)` when all three variables are absent or empty (default
    /// OFF, zero allocations). Fails closed on any partial configuration,
    /// non-UTF-8 value, relative path, malformed hex digest, or invalid size.
    ///
    /// # Errors
    /// Returns an error if only a subset of the three variables is set or if
    /// any field fails validation.
    pub fn from_env() -> Result<Option<Self>> {
        let values = [
            DCOCO_SIDECAR_ENV,
            DCOCO_SIDECAR_SHA256_ENV,
            DCOCO_SIDECAR_SIZE_ENV,
        ]
        .map(std::env::var_os);
        Self::from_os_values(
            values[0].as_deref(),
            values[1].as_deref(),
            values[2].as_deref(),
        )
    }

    /// Parse from optional OS strings, failing closed on non-UTF-8 values.
    ///
    /// # Errors
    /// Returns an error on invalid UTF-8, partial configuration, or malformed fields.
    pub fn from_os_values(
        path: Option<&OsStr>,
        sha256: Option<&OsStr>,
        size: Option<&OsStr>,
    ) -> Result<Option<Self>> {
        let path = utf8_field(DCOCO_SIDECAR_ENV, path)?;
        let sha256 = utf8_field(DCOCO_SIDECAR_SHA256_ENV, sha256)?;
        let size = utf8_field(DCOCO_SIDECAR_SIZE_ENV, size)?;
        Self::from_values(path, sha256, size)
    }

    /// Parse from optional UTF-8 string slices.
    ///
    /// # Errors
    /// Fails closed on partial env configs, relative paths, non-lowercase/non-64-hex
    /// digests, or non-canonical/oversized byte counts.
    pub fn from_values(
        path: Option<&str>,
        sha256: Option<&str>,
        size: Option<&str>,
    ) -> Result<Option<Self>> {
        let fields = [path, sha256, size].map(|value| value.filter(|raw| !raw.is_empty()));
        if fields.iter().all(Option::is_none) {
            return Ok(None);
        }
        let [Some(path_raw), Some(sha_raw), Some(size_raw)] = fields else {
            bail!(
                "DCOCO sidecar request requires {DCOCO_SIDECAR_ENV}, \
                 {DCOCO_SIDECAR_SHA256_ENV}, and {DCOCO_SIDECAR_SIZE_ENV} together"
            );
        };

        let path = PathBuf::from(path_raw);
        ensure!(
            path.is_absolute(),
            "{DCOCO_SIDECAR_ENV} must be an absolute path"
        );
        let sha256 = parse_sha256(sha_raw)?;
        let size = parse_size(size_raw)?;
        Ok(Some(Self { path, sha256, size }))
    }

    /// Canonical path to the sidecar file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Expected whole-file SHA-256 digest.
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// Expected exact sidecar file size in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
}

fn utf8_field<'a>(name: &str, value: Option<&'a OsStr>) -> Result<Option<&'a str>> {
    value
        .map(|raw| {
            raw.to_str()
                .ok_or_else(|| anyhow::anyhow!("{name} must contain valid UTF-8"))
        })
        .transpose()
}

fn parse_sha256(raw: &str) -> Result<[u8; 32]> {
    ensure!(
        raw.len() == 64
            && raw
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{DCOCO_SIDECAR_SHA256_ENV} must be exactly 64 lowercase hexadecimal characters"
    );
    let mut out = [0_u8; 32];
    for (index, pair) in raw.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).context("validated hex bytes must be ASCII")?;
        out[index] = u8::from_str_radix(text, 16).context("validated hex pair must parse")?;
    }
    Ok(out)
}

fn parse_size(raw: &str) -> Result<u64> {
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{DCOCO_SIDECAR_SIZE_ENV} must be a canonical positive decimal integer");
    }
    let size = raw
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("{DCOCO_SIDECAR_SIZE_ENV} integer overflow"))?;
    ensure!(
        size > 0 && size.to_string() == raw,
        "{DCOCO_SIDECAR_SIZE_ENV} must be a canonical positive decimal integer"
    );
    ensure!(
        size <= MAX_DCOCO_SIDECAR_BYTES,
        "{DCOCO_SIDECAR_SIZE_ENV}={size} exceeds the {MAX_DCOCO_SIDECAR_BYTES}-byte safety limit"
    );
    Ok(size)
}

fn hex_prefix(digest_bytes: &[u8; 32]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}..{:02x}{:02x}",
        digest_bytes[0],
        digest_bytes[1],
        digest_bytes[2],
        digest_bytes[3],
        digest_bytes[30],
        digest_bytes[31],
    )
}

// ── Binary schema, artifact, and audit receipt ──────────────────────────────

/// Validated `ATLDC001` binary header metadata.
#[derive(Clone, Copy, PartialEq)]
pub struct DcocoSidecarHeader {
    pub version: u32,
    pub spec: DcocoSpec,
    pub ln_eps: f32,
    pub payload_bytes: u64,
    pub header_payload_sha256: [u8; 32],
}

impl fmt::Debug for DcocoSidecarHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DcocoSidecarHeader")
            .field("version", &self.version)
            .field("spec", &self.spec)
            .field("ln_eps", &self.ln_eps)
            .field("payload_bytes", &self.payload_bytes)
            .field(
                "header_payload_sha256_prefix",
                &hex_prefix(&self.header_payload_sha256),
            )
            .finish()
    }
}

/// Cryptographic audit receipt recorded when an `ATLDC001` file passes authentication.
#[derive(Clone, PartialEq, Eq)]
pub struct DcocoSidecarReceipt {
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub header_payload_sha256: [u8; 32],
    pub size: u64,
    pub device: u64,
    pub inode: u64,
    pub spec: DcocoSpec,
}

impl fmt::Debug for DcocoSidecarReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DcocoSidecarReceipt")
            .field("path", &self.path)
            .field("sha256_prefix", &hex_prefix(&self.sha256))
            .field(
                "header_payload_sha256_prefix",
                &hex_prefix(&self.header_payload_sha256),
            )
            .field("size", &self.size)
            .field("device", &self.device)
            .field("inode", &self.inode)
            .field("spec", &self.spec)
            .finish()
    }
}

/// Byte offsets of all 10 contiguous BF16 tensors inside the `ATLDC001` payload.
#[derive(Clone, Debug)]
struct TensorByteLayout {
    w_cand: Range<usize>,
    w_ctx: Range<usize>,
    w_fuse: Range<usize>,
    fuse_ln_w: Range<usize>,
    fuse_ln_b: Range<usize>,
    w_pair: Range<usize>,
    mlp_w1: Range<usize>,
    mlp_b1: Range<usize>,
    mlp_w2: Range<usize>,
    mlp_b2: Range<usize>,
    total_bytes: usize,
}

fn checked_param_count_u64(hidden_size: usize) -> Result<u64> {
    let d = u64::try_from(hidden_size).context("DCOCO hidden_size overflow")?;
    d.checked_mul(512)
        .and_then(|v| v.checked_add(132_353))
        .context("DCOCO parameter count overflow")
}

fn compute_tensor_layout(spec: &DcocoSpec) -> Result<TensorByteLayout> {
    let d = spec.hidden_size;
    let mut cursor = 0_usize;
    let mut next_range = |elems: usize| -> Result<Range<usize>> {
        let bytes = elems
            .checked_mul(2)
            .context("DCOCO BF16 tensor byte size overflow")?;
        let end = cursor
            .checked_add(bytes)
            .context("DCOCO payload byte offset overflow")?;
        let range = cursor..end;
        cursor = end;
        Ok(range)
    };

    let w_cand = next_range(
        DCOCO_RANK
            .checked_mul(d)
            .context("DCOCO w_cand element count overflow")?,
    )?;
    let w_ctx = next_range(
        DCOCO_RANK
            .checked_mul(d)
            .context("DCOCO w_ctx element count overflow")?,
    )?;
    let w_fuse = next_range(DCOCO_RANK * (2 * DCOCO_RANK))?;
    let fuse_ln_w = next_range(DCOCO_RANK)?;
    let fuse_ln_b = next_range(DCOCO_RANK)?;
    let w_pair = next_range(
        DCOCO_RANK
            .checked_mul(2)
            .and_then(|v| v.checked_mul(d))
            .context("DCOCO w_pair element count overflow")?,
    )?;
    let mlp_w1 = next_range(DCOCO_MLP_HIDDEN * DCOCO_MLP_IN)?;
    let mlp_b1 = next_range(DCOCO_MLP_HIDDEN)?;
    let mlp_w2 = next_range(DCOCO_MLP_HIDDEN)?;
    let mlp_b2 = next_range(1)?;

    Ok(TensorByteLayout {
        w_cand,
        w_ctx,
        w_fuse,
        fuse_ln_w,
        fuse_ln_b,
        w_pair,
        mlp_w1,
        mlp_b1,
        mlp_w2,
        mlp_b2,
        total_bytes: cursor,
    })
}

/// Validated, host-resident `ATLDC001` DCOCO-v2 sidecar artifact.
///
/// Redacts raw weight payload bytes in [`fmt::Debug`] so logs never dump model
/// tensors.
#[derive(Clone)]
pub struct DcocoSidecarArtifact {
    header: DcocoSidecarHeader,
    payload: Vec<u8>,
    receipt: Option<DcocoSidecarReceipt>,
}

impl fmt::Debug for DcocoSidecarArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DcocoSidecarArtifact")
            .field("header", &self.header)
            .field("payload_bytes", &self.payload.len())
            .field("receipt", &self.receipt)
            .finish()
    }
}

impl DcocoSidecarArtifact {
    /// Parse and authenticate an `ATLDC001` byte slice (header, dimensions,
    /// embedded SHA-256, and BF16 finiteness across every weight element).
    ///
    /// # Errors
    /// Fails closed on truncation, wrong magic/version, ABI dimension mismatch,
    /// non-finite/non-positive `ln_eps`, payload length mismatch, embedded
    /// SHA-256 mismatch, or any `NaN`/`±Inf` BF16 weight element.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= DCOCO_SIDECAR_HEADER_BYTES,
            "ATLDC001 sidecar truncated: {} < {DCOCO_SIDECAR_HEADER_BYTES} bytes",
            bytes.len()
        );
        ensure!(
            bytes[0..8] == DCOCO_SIDECAR_MAGIC,
            "ATLDC001 magic mismatch"
        );
        let version = read_u32_le(bytes, 8)?;
        ensure!(
            version == DCOCO_SIDECAR_VERSION,
            "unsupported ATLDC001 version {version}; expected {DCOCO_SIDECAR_VERSION}"
        );
        let hidden_size = usize::try_from(read_u32_le(bytes, 12)?)
            .context("ATLDC001 hidden_size does not fit host usize")?;
        let top_k = usize::try_from(read_u32_le(bytes, 16)?)
            .context("ATLDC001 top_k does not fit host usize")?;
        let rank = usize::try_from(read_u32_le(bytes, 20)?)
            .context("ATLDC001 rank does not fit host usize")?;
        let mlp_hidden = usize::try_from(read_u32_le(bytes, 24)?)
            .context("ATLDC001 mlp_hidden does not fit host usize")?;
        let ln_eps_bits = read_u32_le(bytes, 28)?;
        let payload_bytes = read_u64_le(bytes, 32)?;

        ensure!(
            rank == DCOCO_RANK,
            "ATLDC001 rank={rank} does not match kernel ABI DCOCO_RANK={DCOCO_RANK}"
        );
        ensure!(
            mlp_hidden == DCOCO_MLP_HIDDEN,
            "ATLDC001 mlp_hidden={mlp_hidden} does not match kernel ABI DCOCO_MLP_HIDDEN={DCOCO_MLP_HIDDEN}"
        );
        let spec = DcocoSpec::new(hidden_size, top_k)?;
        let ln_eps = f32::from_bits(ln_eps_bits);
        ensure!(
            ln_eps.is_finite() && ln_eps > 0.0,
            "ATLDC001 ln_eps={ln_eps} must be finite and positive"
        );

        let expected_params = checked_param_count_u64(spec.hidden_size)?;
        let expected_payload_bytes = expected_params
            .checked_mul(2)
            .context("ATLDC001 expected payload byte size overflow")?;
        ensure!(
            payload_bytes == expected_payload_bytes,
            "ATLDC001 payload_bytes={payload_bytes} does not match expected {expected_payload_bytes} \
             for hidden_size={hidden_size}"
        );
        let expected_total = (DCOCO_SIDECAR_HEADER_BYTES as u64)
            .checked_add(payload_bytes)
            .context("ATLDC001 total file size overflow")?;
        ensure!(
            expected_total <= MAX_DCOCO_SIDECAR_BYTES,
            "ATLDC001 total size {expected_total} exceeds {MAX_DCOCO_SIDECAR_BYTES}-byte safety limit"
        );
        ensure!(
            bytes.len() as u64 == expected_total,
            "ATLDC001 length {} does not match header total {expected_total}",
            bytes.len()
        );

        let mut embedded_digest = [0_u8; 32];
        embedded_digest
            .copy_from_slice(&bytes[DCOCO_SIDECAR_HEADER_PREFIX_BYTES..DCOCO_SIDECAR_HEADER_BYTES]);
        let payload_slice = &bytes[DCOCO_SIDECAR_HEADER_BYTES..];
        let computed = sha256_two_parts(&bytes[..DCOCO_SIDECAR_HEADER_PREFIX_BYTES], payload_slice);
        ensure!(
            computed == embedded_digest,
            "ATLDC001 embedded SHA-256 verification failed (header or payload corrupted)"
        );

        validate_all_bf16_finite(payload_slice)?;

        let header = DcocoSidecarHeader {
            version,
            spec,
            ln_eps,
            payload_bytes,
            header_payload_sha256: embedded_digest,
        };
        Ok(Self {
            header,
            payload: payload_slice.to_vec(),
            receipt: None,
        })
    }

    /// Encode host [`DcocoCpuWeights`] into a deterministic, RNE-rounded `ATLDC001` artifact.
    ///
    /// # Errors
    /// Returns an error if any tensor length disagrees with `spec`, `ln_eps` is
    /// non-finite/non-positive, or any weight value is non-finite or overflows BF16.
    pub fn from_cpu_weights(spec: DcocoSpec, weights: &DcocoCpuWeights) -> Result<Self> {
        ensure!(
            spec.rank == DCOCO_RANK && spec.mlp_hidden == DCOCO_MLP_HIDDEN,
            "DCOCO spec must match kernel ABI (rank={DCOCO_RANK}, mlp_hidden={DCOCO_MLP_HIDDEN})"
        );
        ensure!(
            weights.hidden_size == spec.hidden_size,
            "DcocoCpuWeights hidden_size={} != spec hidden_size={}",
            weights.hidden_size,
            spec.hidden_size
        );
        ensure!(
            weights.ln_eps.is_finite() && weights.ln_eps > 0.0,
            "DcocoCpuWeights ln_eps={} must be finite and positive",
            weights.ln_eps
        );

        let d = spec.hidden_size;
        let expected_slices: [(&str, &[f32], usize); 10] = [
            ("w_cand", &weights.w_cand, DCOCO_RANK * d),
            ("w_ctx", &weights.w_ctx, DCOCO_RANK * d),
            ("w_fuse", &weights.w_fuse, DCOCO_RANK * 2 * DCOCO_RANK),
            ("fuse_ln_w", &weights.fuse_ln_w, DCOCO_RANK),
            ("fuse_ln_b", &weights.fuse_ln_b, DCOCO_RANK),
            ("w_pair", &weights.w_pair, DCOCO_RANK * 2 * d),
            ("mlp_w1", &weights.mlp_w1, DCOCO_MLP_HIDDEN * DCOCO_MLP_IN),
            ("mlp_b1", &weights.mlp_b1, DCOCO_MLP_HIDDEN),
            ("mlp_w2", &weights.mlp_w2, DCOCO_MLP_HIDDEN),
            ("mlp_b2", std::slice::from_ref(&weights.mlp_b2), 1),
        ];

        let layout = compute_tensor_layout(&spec)?;
        let mut payload = Vec::with_capacity(layout.total_bytes);
        for (name, slice, expected_len) in expected_slices {
            ensure!(
                slice.len() == expected_len,
                "DCOCO tensor `{name}` has len {} != expected {expected_len}",
                slice.len()
            );
            for (idx, &val) in slice.iter().enumerate() {
                let bf16 = f32_to_bf16_u16_checked(val)
                    .with_context(|| format!("non-finite DCOCO weight `{name}[{idx}]`"))?;
                payload.extend_from_slice(&bf16.to_le_bytes());
            }
        }

        let payload_bytes = u64::try_from(payload.len()).context("payload length overflow")?;
        let prefix = encode_header_prefix(&spec, weights.ln_eps, payload_bytes)?;
        let embedded_digest = sha256_two_parts(&prefix, &payload);
        let header = DcocoSidecarHeader {
            version: DCOCO_SIDECAR_VERSION,
            spec,
            ln_eps: weights.ln_eps,
            payload_bytes,
            header_payload_sha256: embedded_digest,
        };
        Ok(Self {
            header,
            payload,
            receipt: None,
        })
    }

    /// Serialize this artifact into deterministic `ATLDC001` wire bytes.
    ///
    /// # Errors
    /// Returns an error if header fields cannot be encoded into `u32`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let prefix = encode_header_prefix(
            &self.header.spec,
            self.header.ln_eps,
            self.header.payload_bytes,
        )?;
        let mut out = Vec::with_capacity(DCOCO_SIDECAR_HEADER_BYTES + self.payload.len());
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&self.header.header_payload_sha256);
        out.extend_from_slice(&self.payload);
        Ok(out)
    }

    /// Authenticate and load an `ATLDC001` sidecar from disk via `request`.
    ///
    /// Enforces Linux `O_NOFOLLOW`, canonical absolute path, `nlink == 1`,
    /// pre/open/post file-identity verification, exact size + external SHA-256
    /// authentication, embedded SHA-256 verification, BF16 finiteness scan, and
    /// optional `expected_hidden_size` match.
    ///
    /// # Errors
    /// Fails closed on any filesystem race, symlink/hardlink violation, digest
    /// mismatch, corrupted payload, or dimension disagreement.
    pub fn load_authenticated(
        request: &DcocoSidecarRequest,
        expected_hidden_size: Option<usize>,
    ) -> Result<Self> {
        let (raw, identity) = read_authenticated_file(request)?;
        ensure!(
            sha256_bytes(&raw) == request.sha256(),
            "DCOCO sidecar whole-file SHA-256 differs from admitted {DCOCO_SIDECAR_SHA256_ENV}"
        );
        let mut artifact = Self::from_bytes(&raw)?;
        if let Some(expected_d) = expected_hidden_size {
            ensure!(
                artifact.header.spec.hidden_size == expected_d,
                "DCOCO sidecar hidden_size={} does not match target model hidden_size={expected_d}",
                artifact.header.spec.hidden_size
            );
        }
        artifact.receipt = Some(DcocoSidecarReceipt {
            path: request.path().to_path_buf(),
            sha256: request.sha256(),
            header_payload_sha256: artifact.header.header_payload_sha256,
            size: request.size(),
            device: identity.device,
            inode: identity.inode,
            spec: artifact.header.spec,
        });
        Ok(artifact)
    }

    /// Validated `ATLDC001` header metadata.
    #[must_use]
    pub const fn header(&self) -> &DcocoSidecarHeader {
        &self.header
    }

    /// Validated DCOCO geometry specification.
    #[must_use]
    pub const fn spec(&self) -> DcocoSpec {
        self.header.spec
    }

    /// Audit receipt populated when loaded via [`Self::load_authenticated`].
    #[must_use]
    pub fn receipt(&self) -> Option<&DcocoSidecarReceipt> {
        self.receipt.as_ref()
    }

    /// Decode all 10 BF16 tensors to host FP32 [`DcocoCpuWeights`] for CPU reference evaluation.
    ///
    /// # Errors
    /// Returns an error if the internal tensor layout overflows (unreachable for
    /// validated artifacts).
    pub fn to_cpu_weights(&self) -> Result<DcocoCpuWeights> {
        let layout = compute_tensor_layout(&self.header.spec)?;
        let decode_slice = |range: &Range<usize>| -> Vec<f32> {
            self.payload[range.clone()]
                .chunks_exact(2)
                .map(|pair| bf16_u16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                .collect()
        };
        let mlp_b2_vec = decode_slice(&layout.mlp_b2);
        let mlp_b2 = *mlp_b2_vec
            .first()
            .context("validated ATLDC001 payload missing mlp_b2 element")?;
        Ok(DcocoCpuWeights {
            hidden_size: self.header.spec.hidden_size,
            w_cand: decode_slice(&layout.w_cand),
            w_ctx: decode_slice(&layout.w_ctx),
            w_fuse: decode_slice(&layout.w_fuse),
            fuse_ln_w: decode_slice(&layout.fuse_ln_w),
            fuse_ln_b: decode_slice(&layout.fuse_ln_b),
            w_pair: decode_slice(&layout.w_pair),
            mlp_w1: decode_slice(&layout.mlp_w1),
            mlp_b1: decode_slice(&layout.mlp_b1),
            mlp_w2: decode_slice(&layout.mlp_w2),
            mlp_b2,
            ln_eps: self.header.ln_eps,
        })
    }

    /// Transactionally upload all 10 BF16 weight tensors to `gpu`.
    ///
    /// Uses [`DeviceAllocations`] so any allocation or H2D copy failure rolls
    /// back every previously allocated device buffer cleanly.
    ///
    /// # Errors
    /// Returns an error if device allocation or host-to-device transfer fails.
    pub fn upload_to_gpu(&self, gpu: &dyn GpuBackend) -> Result<DcocoSidecarWeights> {
        let layout = compute_tensor_layout(&self.header.spec)?;
        ensure!(
            self.payload.len() == layout.total_bytes,
            "ATLDC001 payload byte length {} != layout {}",
            self.payload.len(),
            layout.total_bytes
        );

        let mut tx = DeviceAllocations::new(gpu);
        let mut upload_tensor = |range: &Range<usize>, label: &str| -> Result<DevicePtr> {
            let bytes = &self.payload[range.clone()];
            let ptr = tx
                .alloc(bytes.len())
                .with_context(|| format!("allocate DCOCO sidecar tensor `{label}`"))?;
            gpu.copy_h2d(bytes, ptr)
                .with_context(|| format!("upload DCOCO sidecar tensor `{label}`"))?;
            Ok(ptr)
        };

        let w_cand = upload_tensor(&layout.w_cand, "w_cand")?;
        let w_ctx = upload_tensor(&layout.w_ctx, "w_ctx")?;
        let w_fuse = upload_tensor(&layout.w_fuse, "w_fuse")?;
        let fuse_ln_w = upload_tensor(&layout.fuse_ln_w, "fuse_ln_w")?;
        let fuse_ln_b = upload_tensor(&layout.fuse_ln_b, "fuse_ln_b")?;
        let w_pair = upload_tensor(&layout.w_pair, "w_pair")?;
        let mlp_w1 = upload_tensor(&layout.mlp_w1, "mlp_w1")?;
        let mlp_b1 = upload_tensor(&layout.mlp_b1, "mlp_b1")?;
        let mlp_w2 = upload_tensor(&layout.mlp_w2, "mlp_w2")?;
        let mlp_b2 = upload_tensor(&layout.mlp_b2, "mlp_b2")?;

        tx.commit();

        Ok(DcocoSidecarWeights {
            spec: self.header.spec,
            ln_eps: self.header.ln_eps,
            w_cand,
            w_ctx,
            w_fuse,
            fuse_ln_w,
            fuse_ln_b,
            w_pair,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            receipt: self.receipt.clone(),
        })
    }
}

/// Owned GPU-resident weight buffers uploaded from an authenticated `ATLDC001` sidecar.
#[derive(Debug)]
pub struct DcocoSidecarWeights {
    spec: DcocoSpec,
    ln_eps: f32,
    w_cand: DevicePtr,
    w_ctx: DevicePtr,
    w_fuse: DevicePtr,
    fuse_ln_w: DevicePtr,
    fuse_ln_b: DevicePtr,
    w_pair: DevicePtr,
    mlp_w1: DevicePtr,
    mlp_b1: DevicePtr,
    mlp_w2: DevicePtr,
    mlp_b2: DevicePtr,
    receipt: Option<DcocoSidecarReceipt>,
}

impl DcocoSidecarWeights {
    /// Borrow the non-owning [`DcocoWeights`] descriptor for kernel launches.
    #[must_use]
    pub const fn as_weights(&self) -> DcocoWeights {
        DcocoWeights {
            spec: self.spec,
            w_cand: DenseWeight {
                weight: self.w_cand,
            },
            w_ctx: DenseWeight { weight: self.w_ctx },
            w_fuse: DenseWeight {
                weight: self.w_fuse,
            },
            fuse_ln_w: DenseWeight {
                weight: self.fuse_ln_w,
            },
            fuse_ln_b: DenseWeight {
                weight: self.fuse_ln_b,
            },
            w_pair: DenseWeight {
                weight: self.w_pair,
            },
            mlp_w1: DenseWeight {
                weight: self.mlp_w1,
            },
            mlp_b1: DenseWeight {
                weight: self.mlp_b1,
            },
            mlp_w2: DenseWeight {
                weight: self.mlp_w2,
            },
            mlp_b2: DenseWeight {
                weight: self.mlp_b2,
            },
            ln_eps: self.ln_eps,
        }
    }

    /// Audit receipt from authenticated disk load, when present.
    #[must_use]
    pub fn receipt(&self) -> Option<&DcocoSidecarReceipt> {
        self.receipt.as_ref()
    }

    /// Release all 10 owned GPU weight buffers idempotently.
    pub fn release(&mut self, gpu: &dyn GpuBackend) {
        for ptr in [
            &mut self.w_cand,
            &mut self.w_ctx,
            &mut self.w_fuse,
            &mut self.fuse_ln_w,
            &mut self.fuse_ln_b,
            &mut self.w_pair,
            &mut self.mlp_w1,
            &mut self.mlp_b1,
            &mut self.mlp_w2,
            &mut self.mlp_b2,
        ] {
            if !ptr.is_null() {
                let _ = gpu.free(*ptr);
                *ptr = DevicePtr::NULL;
            }
        }
    }
}

// ── Binary & cryptographic helpers ──────────────────────────────────────────

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice: [u8; 4] = bytes
        .get(offset..offset + 4)
        .context("ATLDC001 header u32 out of bounds")?
        .try_into()
        .context("ATLDC001 header u32 slice length")?;
    Ok(u32::from_le_bytes(slice))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice: [u8; 8] = bytes
        .get(offset..offset + 8)
        .context("ATLDC001 header u64 out of bounds")?
        .try_into()
        .context("ATLDC001 header u64 slice length")?;
    Ok(u64::from_le_bytes(slice))
}

fn encode_header_prefix(
    spec: &DcocoSpec,
    ln_eps: f32,
    payload_bytes: u64,
) -> Result<[u8; DCOCO_SIDECAR_HEADER_PREFIX_BYTES]> {
    let mut prefix = [0_u8; DCOCO_SIDECAR_HEADER_PREFIX_BYTES];
    prefix[0..8].copy_from_slice(&DCOCO_SIDECAR_MAGIC);
    prefix[8..12].copy_from_slice(&DCOCO_SIDECAR_VERSION.to_le_bytes());
    prefix[12..16].copy_from_slice(
        &u32::try_from(spec.hidden_size)
            .context("hidden_size exceeds u32")?
            .to_le_bytes(),
    );
    prefix[16..20].copy_from_slice(
        &u32::try_from(spec.top_k)
            .context("top_k exceeds u32")?
            .to_le_bytes(),
    );
    prefix[20..24].copy_from_slice(
        &u32::try_from(spec.rank)
            .context("rank exceeds u32")?
            .to_le_bytes(),
    );
    prefix[24..28].copy_from_slice(
        &u32::try_from(spec.mlp_hidden)
            .context("mlp_hidden exceeds u32")?
            .to_le_bytes(),
    );
    prefix[28..32].copy_from_slice(&ln_eps.to_bits().to_le_bytes());
    prefix[32..40].copy_from_slice(&payload_bytes.to_le_bytes());
    Ok(prefix)
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let digest_val = digest::digest(&digest::SHA256, bytes);
    let mut out = [0_u8; 32];
    out.copy_from_slice(digest_val.as_ref());
    out
}

fn sha256_two_parts(first: &[u8], second: &[u8]) -> [u8; 32] {
    let mut ctx = digest::Context::new(&digest::SHA256);
    ctx.update(first);
    ctx.update(second);
    let digest_val = ctx.finish();
    let mut out = [0_u8; 32];
    out.copy_from_slice(digest_val.as_ref());
    out
}

fn f32_to_bf16_u16_checked(x: f32) -> Result<u16> {
    ensure!(x.is_finite(), "weight value {x} is non-finite");
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32.wrapping_add(lsb);
    let bf16 = (bits.wrapping_add(rounding_bias) >> 16) as u16;
    ensure!(
        (bf16 & BF16_EXP_MASK) != BF16_EXP_MASK,
        "weight value {x} overflows BF16 to Inf/NaN"
    );
    Ok(bf16)
}

#[inline]
fn bf16_u16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn validate_all_bf16_finite(payload: &[u8]) -> Result<()> {
    ensure!(
        payload.len().is_multiple_of(2),
        "ATLDC001 BF16 payload length {} is not even",
        payload.len()
    );
    for (idx, pair) in payload.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        ensure!(
            (bits & BF16_EXP_MASK) != BF16_EXP_MASK,
            "ATLDC001 payload contains non-finite BF16 weight (0x{bits:04x}) at element index {idx}"
        );
    }
    Ok(())
}

// ── File authentication (Linux O_NOFOLLOW + anti-TOCTOU) ────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
    )
))]
const LINUX_O_NOFOLLOW: i32 = 0o00100000;

#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "loongarch64",
    )
))]
const LINUX_O_NOFOLLOW: i32 = 0o00400000;

#[cfg(target_os = "linux")]
fn read_authenticated_file(request: &DcocoSidecarRequest) -> Result<(Vec<u8>, FileIdentity)> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    ensure!(
        request.size() <= MAX_DCOCO_SIDECAR_BYTES,
        "DCOCO sidecar exceeds the {MAX_DCOCO_SIDECAR_BYTES}-byte safety limit"
    );
    let canonical_path = std::fs::canonicalize(request.path())
        .with_context(|| format!("canonicalize DCOCO sidecar {}", request.path().display()))?;
    ensure!(
        canonical_path == request.path(),
        "DCOCO sidecar path must already be canonical"
    );
    let pre_metadata = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("stat DCOCO sidecar {}", request.path().display()))?;
    ensure!(
        pre_metadata.file_type().is_file(),
        "DCOCO sidecar must be a regular file"
    );
    ensure!(
        pre_metadata.nlink() == 1,
        "DCOCO sidecar must not have hard links"
    );
    let pre_identity = file_identity(&pre_metadata);

    let mut file: File = OpenOptions::new()
        .read(true)
        .custom_flags(LINUX_O_NOFOLLOW)
        .open(request.path())
        .with_context(|| format!("open DCOCO sidecar {}", request.path().display()))?;
    let opened_metadata = file
        .metadata()
        .context("stat opened DCOCO sidecar descriptor")?;
    ensure!(
        opened_metadata.file_type().is_file(),
        "opened DCOCO sidecar is not a regular file"
    );
    let opened_identity = file_identity(&opened_metadata);
    ensure!(
        pre_identity == opened_identity,
        "DCOCO sidecar changed between path stat and O_NOFOLLOW open"
    );
    ensure!(
        opened_identity.size == request.size(),
        "DCOCO sidecar size {} differs from admitted {}",
        opened_identity.size,
        request.size()
    );

    let expected =
        usize::try_from(request.size()).context("DCOCO sidecar size does not fit host usize")?;
    let read_limit = expected
        .checked_add(1)
        .context("DCOCO sidecar read limit overflow")?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(read_limit)
        .context("reserve DCOCO sidecar read buffer")?;
    (&mut file)
        .take(read_limit as u64)
        .read_to_end(&mut raw)
        .context("read authenticated DCOCO sidecar bytes")?;
    ensure!(
        raw.len() == expected,
        "DCOCO sidecar changed size while reading ({} != {expected})",
        raw.len()
    );

    let post_descriptor = file
        .metadata()
        .context("restat held DCOCO sidecar descriptor")?;
    let post_path = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("restat DCOCO sidecar {}", request.path().display()))?;
    ensure!(
        post_path.file_type().is_file(),
        "DCOCO sidecar path ceased to be a regular file"
    );
    ensure!(
        post_path.nlink() == 1,
        "DCOCO sidecar acquired a hard link while reading"
    );
    ensure!(
        file_identity(&post_descriptor) == opened_identity
            && file_identity(&post_path) == opened_identity,
        "DCOCO sidecar identity changed during authenticated read"
    );

    Ok((raw, opened_identity))
}

#[cfg(target_os = "linux")]
fn file_identity(metadata: &std::fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;

    FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        links: metadata.nlink(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

#[cfg(not(target_os = "linux"))]
fn read_authenticated_file(_request: &DcocoSidecarRequest) -> Result<(Vec<u8>, FileIdentity)> {
    bail!("fail-closed DCOCO sidecar loading currently requires Linux O_NOFOLLOW")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dflash_head::dcoco::{
        Bf16Rounding, DCOCO_DEFAULT_LN_EPS, DcocoCorrector, DcocoStepInputs,
        dcoco_pair_mlp_factored_cpu, dcoco_proj_fuse_cpu,
    };
    use crate::layers::ops::DcocoWalkMode;
    use spark_runtime::gpu::{KernelHandle, PinnedHostBuffer, mock::MockGpuBackend};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "atlas_dcoco_weights_{label}_{}_{seq}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).expect("create temp dir");
            let canonical = std::fs::canonicalize(&dir).expect("canonicalize temp dir");
            Self { path: canonical }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        use std::fmt::Write;
        let digest = sha256_bytes(bytes);
        let mut s = String::with_capacity(64);
        for b in digest {
            let _ = write!(&mut s, "{b:02x}");
        }
        s
    }

    fn make_synthetic_cpu_weights(hidden_size: usize) -> DcocoCpuWeights {
        let mk = |len: usize, scale: f32, seed: u32| -> Vec<f32> {
            let mut out = Vec::with_capacity(len);
            let mut st = seed.wrapping_add(0x9E37_79B9);
            for _ in 0..len {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = ((st >> 8) as f32) / ((1_u32 << 24) as f32) * 2.0 - 1.0;
                out.push(u * scale);
            }
            out
        };
        DcocoCpuWeights {
            hidden_size,
            w_cand: mk(DCOCO_RANK * hidden_size, 0.12, 1),
            w_ctx: mk(DCOCO_RANK * hidden_size, 0.12, 2),
            w_fuse: mk(DCOCO_RANK * 2 * DCOCO_RANK, 0.08, 3),
            fuse_ln_w: vec![1.0; DCOCO_RANK],
            fuse_ln_b: vec![0.0; DCOCO_RANK],
            w_pair: mk(DCOCO_RANK * 2 * hidden_size, 0.10, 4),
            mlp_w1: mk(DCOCO_MLP_HIDDEN * DCOCO_MLP_IN, 0.06, 5),
            mlp_b1: mk(DCOCO_MLP_HIDDEN, 0.02, 6),
            mlp_w2: mk(DCOCO_MLP_HIDDEN, 0.05, 7),
            mlp_b2: 0.015_625,
            ln_eps: DCOCO_DEFAULT_LN_EPS,
        }
    }

    fn write_artifact_file(dir: &Path, name: &str, bytes: &[u8]) -> (PathBuf, String, String) {
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write sidecar fixture");
        let canonical = std::fs::canonicalize(&path).expect("canonicalize sidecar");
        let sha = hex_sha256(bytes);
        let size = bytes.len().to_string();
        (canonical, sha, size)
    }

    #[test]
    fn env_request_defaults_off_and_fails_closed_on_partial_or_malformed() {
        // Absent or all-empty -> Ok(None) (default OFF).
        assert!(
            DcocoSidecarRequest::from_values(None, None, None)
                .expect("all absent")
                .is_none()
        );
        assert!(
            DcocoSidecarRequest::from_values(Some(""), Some(""), Some(""))
                .expect("all empty")
                .is_none()
        );

        let good_sha = "a".repeat(64);
        // Any partial subset fails closed.
        assert!(DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), None, None).is_err());
        assert!(
            DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), Some(&good_sha), None).is_err()
        );
        assert!(DcocoSidecarRequest::from_values(None, Some(&good_sha), Some("1024")).is_err());

        // Relative path fails closed.
        assert!(
            DcocoSidecarRequest::from_values(Some("relative/a.atldc"), Some(&good_sha), Some("72"))
                .is_err()
        );
        // Uppercase or wrong-length SHA-256 fails closed.
        let upper_sha = "A".repeat(64);
        assert!(
            DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), Some(&upper_sha), Some("72"))
                .is_err()
        );
        assert!(
            DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), Some("abcd"), Some("72"))
                .is_err()
        );
        // Non-canonical / zero / oversized size fails closed.
        assert!(
            DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), Some(&good_sha), Some("0"))
                .is_err()
        );
        assert!(
            DcocoSidecarRequest::from_values(Some("/tmp/a.atldc"), Some(&good_sha), Some("072"))
                .is_err()
        );
        assert!(
            DcocoSidecarRequest::from_values(
                Some("/tmp/a.atldc"),
                Some(&good_sha),
                Some(&(MAX_DCOCO_SIDECAR_BYTES + 1).to_string())
            )
            .is_err()
        );

        // Non-UTF-8 OS value fails closed.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let bad = OsStr::from_bytes(&[0xFF, 0xFE]);
            assert!(
                DcocoSidecarRequest::from_os_values(
                    Some(bad),
                    Some(OsStr::new(&good_sha)),
                    Some(OsStr::new("72"))
                )
                .is_err()
            );
        }

        // Valid request parses cleanly and redacts full digest in Debug.
        let req = DcocoSidecarRequest::from_values(
            Some("/tmp/model.atldc"),
            Some(&good_sha),
            Some("1024"),
        )
        .expect("valid request")
        .expect("present");
        assert_eq!(req.size(), 1024);
        let dbg = format!("{req:?}");
        assert!(!dbg.contains(&good_sha));
    }

    #[test]
    fn binary_parser_rejects_every_tamper_and_non_finite_bf16_weight() {
        let spec = DcocoSpec::new(32, 4).expect("valid spec");
        let cpu = make_synthetic_cpu_weights(32);
        let artifact = DcocoSidecarArtifact::from_cpu_weights(spec, &cpu).expect("encode");
        let valid_bytes = artifact.to_bytes().expect("serialize");

        // Valid round-trip succeeds and redacts raw bytes in Debug.
        let parsed = DcocoSidecarArtifact::from_bytes(&valid_bytes).expect("parse valid");
        assert_eq!(parsed.spec(), spec);
        assert_eq!(parsed.to_bytes().expect("reserialize"), valid_bytes);
        let dbg = format!("{parsed:?}");
        assert!(dbg.contains("payload_bytes"));

        // Truncated header / truncated payload.
        assert!(DcocoSidecarArtifact::from_bytes(&valid_bytes[..71]).is_err());
        assert!(DcocoSidecarArtifact::from_bytes(&valid_bytes[..valid_bytes.len() - 2]).is_err());

        // Helper that modifies header prefix AND recomputes embedded digest so we test
        // semantic validation independently of the SHA check.
        let resign = |mut buf: Vec<u8>| -> Vec<u8> {
            let digest = sha256_two_parts(
                &buf[..DCOCO_SIDECAR_HEADER_PREFIX_BYTES],
                &buf[DCOCO_SIDECAR_HEADER_BYTES..],
            );
            buf[DCOCO_SIDECAR_HEADER_PREFIX_BYTES..DCOCO_SIDECAR_HEADER_BYTES]
                .copy_from_slice(&digest);
            buf
        };

        // 1. Embedded SHA catches unresigned header tamper & payload tamper.
        let mut hdr_tamper = valid_bytes.clone();
        hdr_tamper[12] ^= 0x01;
        assert!(DcocoSidecarArtifact::from_bytes(&hdr_tamper).is_err());

        let mut body_tamper = valid_bytes.clone();
        let last = body_tamper.len() - 1;
        body_tamper[last] ^= 0x01;
        assert!(DcocoSidecarArtifact::from_bytes(&body_tamper).is_err());

        // 2. Bad magic, bad version, bad rank, bad mlp_hidden, bad top_k, bad ln_eps, bad payload_bytes.
        let mut bad_magic = valid_bytes.clone();
        bad_magic[0] = b'X';
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_magic)).is_err());

        let mut bad_ver = valid_bytes.clone();
        bad_ver[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_ver)).is_err());

        let mut bad_topk = valid_bytes.clone();
        bad_topk[16..20].copy_from_slice(&17_u32.to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_topk)).is_err());

        let mut bad_rank = valid_bytes.clone();
        bad_rank[20..24].copy_from_slice(&64_u32.to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_rank)).is_err());

        let mut bad_mlp = valid_bytes.clone();
        bad_mlp[24..28].copy_from_slice(&128_u32.to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_mlp)).is_err());

        let mut bad_eps_zero = valid_bytes.clone();
        bad_eps_zero[28..32].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_eps_zero)).is_err());

        let mut bad_eps_nan = valid_bytes.clone();
        bad_eps_nan[28..32].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_eps_nan)).is_err());

        let mut bad_len_hdr = valid_bytes.clone();
        bad_len_hdr[32..40].copy_from_slice(&16_u64.to_le_bytes());
        assert!(DcocoSidecarArtifact::from_bytes(&resign(bad_len_hdr)).is_err());

        // 3. Re-signed NaN and +/-Inf BF16 weight elements MUST fail closed.
        for bad_bf16 in [0x7FC0_u16, 0x7F80_u16, 0xFF80_u16] {
            let mut poisoned = valid_bytes.clone();
            poisoned[DCOCO_SIDECAR_HEADER_BYTES..DCOCO_SIDECAR_HEADER_BYTES + 2]
                .copy_from_slice(&bad_bf16.to_le_bytes());
            assert!(
                DcocoSidecarArtifact::from_bytes(&resign(poisoned)).is_err(),
                "expected non-finite BF16 0x{bad_bf16:04x} to be rejected"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn authenticated_disk_loader_enforces_symlink_hardlink_sha_size_and_hidden_dim() {
        let tmp = TempDirGuard::new("disk_auth");
        let spec = DcocoSpec::new(32, 4).expect("spec");
        let bytes = DcocoSidecarArtifact::from_cpu_weights(spec, &make_synthetic_cpu_weights(32))
            .expect("encode")
            .to_bytes()
            .expect("serialize");
        let (path, sha, size) = write_artifact_file(&tmp.path, "valid.atldc", &bytes);

        // Valid load + hidden_size check succeeds and populates receipt.
        let req = DcocoSidecarRequest::from_values(
            Some(path.to_str().expect("utf8 path")),
            Some(&sha),
            Some(&size),
        )
        .expect("parse req")
        .expect("req");
        let loaded = DcocoSidecarArtifact::load_authenticated(&req, Some(32)).expect("load valid");
        let receipt = loaded.receipt().expect("receipt populated");
        assert_eq!(receipt.path, path);
        assert_eq!(receipt.size, bytes.len() as u64);
        assert_eq!(receipt.spec, spec);

        // Wrong expected_hidden_size fails closed.
        assert!(DcocoSidecarArtifact::load_authenticated(&req, Some(64)).is_err());

        // Wrong external SHA-256 fails closed.
        let mut wrong_sha = sha.clone();
        wrong_sha.replace_range(0..1, if sha.starts_with('0') { "1" } else { "0" });
        let bad_sha_req = DcocoSidecarRequest::from_values(
            Some(path.to_str().expect("utf8")),
            Some(&wrong_sha),
            Some(&size),
        )
        .expect("parse")
        .expect("req");
        assert!(DcocoSidecarArtifact::load_authenticated(&bad_sha_req, Some(32)).is_err());

        // Wrong size fails closed.
        let bad_size_req = DcocoSidecarRequest::from_values(
            Some(path.to_str().expect("utf8")),
            Some(&sha),
            Some(&(bytes.len() + 2).to_string()),
        )
        .expect("parse")
        .expect("req");
        assert!(DcocoSidecarArtifact::load_authenticated(&bad_size_req, Some(32)).is_err());

        // Symlink path fails closed.
        let symlink_path = tmp.path.join("link.atldc");
        std::os::unix::fs::symlink(&path, &symlink_path).expect("create symlink");
        let symlink_req = DcocoSidecarRequest::from_values(
            Some(symlink_path.to_str().expect("utf8")),
            Some(&sha),
            Some(&size),
        )
        .expect("parse")
        .expect("req");
        assert!(DcocoSidecarArtifact::load_authenticated(&symlink_req, Some(32)).is_err());

        // Hard-linked file (nlink == 2) fails closed.
        let hardlink_path = tmp.path.join("hardlink.atldc");
        std::fs::hard_link(&path, &hardlink_path).expect("create hardlink");
        assert!(DcocoSidecarArtifact::load_authenticated(&req, Some(32)).is_err());
    }

    struct FailingUploadGpu {
        inner: MockGpuBackend,
        alloc_calls: AtomicUsize,
        fail_at_alloc: usize,
    }

    impl GpuBackend for FailingUploadGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            let n = self.alloc_calls.fetch_add(1, Ordering::SeqCst);
            ensure!(
                n != self.fail_at_alloc,
                "injected GPU alloc failure at #{n}"
            );
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.alloc(bytes)
        }
        fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
            self.inner.alloc_host_pinned(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            kernel: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(kernel, grid, block, shared, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }
        fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, name)
        }
        fn memset(&self, dst: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(dst, value, bytes)
        }
        fn memset_async(&self, dst: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(dst, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    #[test]
    fn public_sidecar_types_are_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DcocoSidecarRequest>();
        assert_send_sync::<DcocoSidecarHeader>();
        assert_send_sync::<DcocoSidecarReceipt>();
        assert_send_sync::<DcocoSidecarArtifact>();
        assert_send_sync::<DcocoSidecarWeights>();
    }

    #[test]
    fn transactional_gpu_upload_rolls_back_on_partial_failure_and_runs_end_to_end() {
        let spec = DcocoSpec::new(32, 4).expect("spec");
        let artifact =
            DcocoSidecarArtifact::from_cpu_weights(spec, &make_synthetic_cpu_weights(32))
                .expect("encode");

        // Failure on the 6th tensor allocation rolls back the first 5 buffers cleanly.
        let failing_gpu = FailingUploadGpu {
            inner: MockGpuBackend::new(),
            alloc_calls: AtomicUsize::new(0),
            fail_at_alloc: 5,
        };
        assert!(artifact.upload_to_gpu(&failing_gpu).is_err());
        assert_eq!(failing_gpu.inner.alloc_count(), 0);
        assert_eq!(failing_gpu.inner.invalid_free_count(), 0);

        // Full upload -> release leaves 0 allocations.
        let gpu = MockGpuBackend::new();
        let mut uploaded = artifact.upload_to_gpu(&gpu).expect("upload");
        assert_eq!(gpu.alloc_count(), 10);
        uploaded.release(&gpu);
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);

        // Scratch failure inside DcocoCorrector::from_sidecar_request rolls back all 10 sidecar bufs.
        #[cfg(target_os = "linux")]
        {
            let tmp = TempDirGuard::new("scratch_rollback");
            let raw = artifact.to_bytes().expect("serialize");
            let (path, sha, size) = write_artifact_file(&tmp.path, "rb.atldc", &raw);
            let req = DcocoSidecarRequest::from_values(
                Some(path.to_str().expect("utf8")),
                Some(&sha),
                Some(&size),
            )
            .expect("parse")
            .expect("req");
            assert!(DcocoCorrector::from_sidecar_request(&gpu, &req, 32, 0, 4).is_err());
            assert_eq!(gpu.alloc_count(), 0);
            assert_eq!(gpu.invalid_free_count(), 0);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn python_trainer_atldc001_round_trip_matches_rust_cpu_reference_and_launches_gpu() {
        let script =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/train_dcoco_corrector.py");
        assert!(
            script.exists(),
            "missing DCOCO trainer script at {}",
            script.display()
        );

        let tmp = TempDirGuard::new("py_roundtrip");
        let sidecar_path = tmp.path.join("trained_fixture.atldc");
        let receipt_path = tmp.path.join("trained_fixture.json");

        let output = std::process::Command::new("python3")
            .env("CUDA_VISIBLE_DEVICES", "")
            .arg(&script)
            .arg("--emit-test-fixture")
            .arg(&sidecar_path)
            .arg("--receipt")
            .arg(&receipt_path)
            .arg("--hidden-size")
            .arg("32")
            .arg("--top-k")
            .arg("4")
            .arg("--steps")
            .arg("20")
            .arg("--backend")
            .arg("numpy")
            .arg("--seed")
            .arg("17")
            .output()
            .expect("spawn tools/train_dcoco_corrector.py");
        assert!(
            output.status.success(),
            "train_dcoco_corrector.py failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );

        let receipt_text = std::fs::read_to_string(&receipt_path).expect("read python receipt");
        let receipt_json: serde_json::Value =
            serde_json::from_str(&receipt_text).expect("parse python receipt JSON");
        let env_obj = receipt_json["env"].as_object().expect("receipt.env object");
        let req = DcocoSidecarRequest::from_values(
            env_obj[DCOCO_SIDECAR_ENV].as_str(),
            env_obj[DCOCO_SIDECAR_SHA256_ENV].as_str(),
            env_obj[DCOCO_SIDECAR_SIZE_ENV].as_str(),
        )
        .expect("parse env request")
        .expect("request present");

        let artifact = DcocoSidecarArtifact::load_authenticated(&req, Some(32))
            .expect("load Python-emitted ATLDC001 sidecar");
        assert_eq!(artifact.spec(), DcocoSpec::new(32, 4).expect("spec"));
        let cpu_weights = artifact.to_cpu_weights().expect("decode CPU weights");

        // Verify exact numerical parity against Python BF16-rounded reference scores.
        let ref_obj = &receipt_json["reference_check"];
        let gamma = usize::try_from(ref_obj["gamma"].as_u64().expect("gamma")).expect("usize");
        let top_k = usize::try_from(ref_obj["top_k"].as_u64().expect("top_k")).expect("usize");
        let anchor_id =
            u32::try_from(ref_obj["anchor_id"].as_u64().expect("anchor_id")).expect("u32");
        let anchor_logit = ref_obj["anchor_logit"].as_f64().expect("anchor_logit") as f32;
        let embed_tokens: Vec<f32> = ref_obj["embed_tokens"]
            .as_array()
            .expect("embed_tokens")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect();
        let hidden_states: Vec<f32> = ref_obj["hidden_states"]
            .as_array()
            .expect("hidden_states")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect();
        let candidates: Vec<u32> = ref_obj["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(|v| u32::try_from(v.as_u64().expect("u64")).expect("u32"))
            .collect();
        let unary: Vec<f32> = ref_obj["unary"]
            .as_array()
            .expect("unary")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect();

        let (z_cand, s_pair) = dcoco_proj_fuse_cpu(
            &embed_tokens,
            &hidden_states,
            anchor_id,
            &candidates,
            &cpu_weights,
            gamma,
            top_k,
            Bf16Rounding::Enabled,
        );
        let rust_scores = dcoco_pair_mlp_factored_cpu(
            &s_pair,
            &z_cand,
            &unary,
            anchor_logit,
            &cpu_weights,
            gamma,
            top_k,
            Bf16Rounding::Enabled,
        );
        let expected_scores = ref_obj["expected_scores"]
            .as_array()
            .expect("expected_scores");
        assert_eq!(rust_scores.len(), expected_scores.len());
        let mut max_abs_diff = 0.0_f32;
        for (idx, (got, want_val)) in rust_scores.iter().zip(expected_scores.iter()).enumerate() {
            match want_val.as_f64() {
                None => assert!(
                    got.is_infinite() && got.is_sign_negative(),
                    "index {idx}: expected -inf for unused anchor row, got {got}"
                ),
                Some(want) => {
                    let diff = (*got - want as f32).abs();
                    max_abs_diff = max_abs_diff.max(diff);
                    assert!(
                        diff < 1.0e-4,
                        "index {idx}: Rust/Python DCOCO score mismatch got={got} want={want} diff={diff}"
                    );
                }
            }
        }
        assert!(max_abs_diff < 1.0e-4);

        // End-to-end GPU sidecar constructor -> 3-kernel launch -> clean teardown (0 leaked bufs).
        let gpu = MockGpuBackend::new();
        let mut corrector = DcocoCorrector::from_sidecar_request(&gpu, &req, 32, 2, 4)
            .expect("construct DcocoCorrector from authenticated sidecar");
        assert!(corrector.sidecar_receipt().is_some());
        assert_eq!(gpu.alloc_count(), 13); // 10 sidecar tensors + 3 scratch buffers
        let p = DevicePtr(0x2_0000);
        corrector
            .run_on_stream(
                &gpu,
                &DcocoStepInputs {
                    embed_tokens: p,
                    hidden_states: p,
                    anchor_ids: p,
                    anchor_logits: None,
                    candidates: p,
                    unary: p,
                    out_tokens: p,
                    out_max_prob: p,
                    out_q_ids: Some(p),
                    out_q_probs: Some(p),
                    out_q_row_bases: Some(p),
                    seeds: Some(p),
                    step0_abs_pos: Some(p),
                    temperatures: Some(p),
                    draft_temp_scales: Some(p),
                    top_ks: Some(p),
                    top_ps: Some(p),
                    min_ps: Some(p),
                    walk_mode: DcocoWalkMode::MaximalCoupling,
                    out_q_stride: 20,
                    batch_size: 1,
                    gamma: 2,
                    vocab_size: 64,
                },
                0,
            )
            .expect("launch 3-kernel DCOCO pipeline with sidecar weights");
        assert_eq!(gpu.launch_count(), 3);
        corrector.release(&gpu);
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}
