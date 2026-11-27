// SPDX-License-Identifier: AGPL-3.0-only

//! Authenticated AutoEMatic (`+0.4 bpw`) AutoFP6 sidecar loader and host mirror.
//!
//! Loads offline-generated sparse 2-bit mantissa offset sidecars (produced by
//! `tools/build_autoematic_sidecar.py`) using the same fail-closed security
//! posture as `weight_map::w3_sidecar`:
//!   - All four environment variables (`ATLAS_AUTOEMATIC_LAYERS`,
//!     `ATLAS_AUTOEMATIC_SIDECAR`, `ATLAS_AUTOEMATIC_SIDECAR_SHA256`,
//!     `ATLAS_AUTOEMATIC_SIDECAR_SIZE`) must be specified together; when all
//!     four are absent or empty, loading is disabled (default OFF, bit-exact
//!     NVFP4 baseline).
//!   - Linux `O_NOFOLLOW` open, canonical path enforcement, `nlink == 1`,
//!     pre/open/post [`FileIdentity`] verification, and SHA-256 authentication
//!     before parsing the duplicate-free safetensors header.
//!   - Strict prefix-sum and `popcount(mask) <= 2` validation across every
//!     16-element NVFP4 group prior to any GPU allocation.
//!   - Transactional GPU upload with automatic rollback on partial failure.
//!
//! Sidecar tensors per requested layer prefix `<lp>` and projection
//! `<proj> in {gate_proj, up_proj, down_proj}`:
//!   - `{lp}.mlp.{proj}.ae_refine_codes`  U8  `[N, num_refined_sg]`
//!   - `{lp}.mlp.{proj}.ae_refine_mask`   U8  `[K/16]` (low 4 bits = SG bitmask)
//!   - `{lp}.mlp.{proj}.ae_group_prefix`  U32 `[K/16]` (exclusive popcount prefix)
//!   - `{lp}.mlp.{proj}.ae_meta`          F32 `[2]`    (`[rho_actual, 0.25]`)

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fmt;
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::e4m3_to_f32;
use crate::weight_map::w3_sidecar::sha256;

/// Environment variable naming layers enabled for AutoEMatic AutoFP6 refinement.
pub const AUTOEMATIC_LAYERS_ENV: &str = "ATLAS_AUTOEMATIC_LAYERS";

/// Environment variable specifying the canonical path to the AutoEMatic sidecar.
pub const AUTOEMATIC_SIDECAR_ENV: &str = "ATLAS_AUTOEMATIC_SIDECAR";

/// Environment variable specifying the expected 64-hex-char SHA-256 digest.
pub const AUTOEMATIC_SIDECAR_SHA256_ENV: &str = "ATLAS_AUTOEMATIC_SIDECAR_SHA256";

/// Environment variable specifying the exact sidecar file size in bytes.
pub const AUTOEMATIC_SIDECAR_SIZE_ENV: &str = "ATLAS_AUTOEMATIC_SIDECAR_SIZE";

/// NVFP4 block-scale group size (elements per UE4M3 scale byte).
pub const AUTOEMATIC_GROUP_SIZE: usize = 16;

/// Number of weights per AutoEMatic refinement subgroup.
pub const AUTOEMATIC_SUBGROUP_SIZE: usize = 4;

/// Number of 4-element subgroups inside one 16-element NVFP4 group.
pub const AUTOEMATIC_SUBGROUPS_PER_GROUP: usize = AUTOEMATIC_GROUP_SIZE / AUTOEMATIC_SUBGROUP_SIZE;

/// Hardware-enforced cap on refined subgroups per 16-element group (max 50%).
pub const AUTOEMATIC_MAX_REFINED_PER_GROUP: u32 = 2;

/// Fixed epilogue refinement scale factor (`delta = 0.25 * (4*delta)`).
pub const AUTOEMATIC_REFINEMENT_SCALE: f32 = 0.25;

/// Standard E2M1 dequantization LUT matching `autoematic_fp6_gemm.cu`.
pub const AUTOEMATIC_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const MAX_AUTOEMATIC_LAYERS: usize = 160;
const MAX_HEADER_BYTES: usize = 128 << 20;
#[cfg(target_os = "linux")]
const MAX_SIDECAR_BYTES: u64 = 16 << 30;

// ── Host-side exact format mirror (matches autoematic_fp6_gemm.cu) ──────────

/// Branchless 2-bit two's-complement offset -> 4-bit E2M1 code for `4 * delta`.
///
/// Exact mapping (arXiv:2611.10165 §3.2):
///   - `0 (0b00)` -> `d =  0` -> `4*delta =  0.0` -> `0x0`
///   - `1 (0b01)` -> `d = +1` -> `4*delta = +0.5` -> `0x1`
///   - `2 (0b10)` -> `d = -2` -> `4*delta = -1.0` -> `0xA`
///   - `3 (0b11)` -> `d = -1` -> `4*delta = -0.5` -> `0x9`
#[must_use]
pub const fn decode_code2_to_e2m1(code2: u8) -> u8 {
    let c = (code2 & 0x03) as u32;
    let sign = (c & 2) << 2;
    let mag = (c & 1) | ((c & 2) & (!(c << 1) & 2));
    (sign | mag) as u8
}

/// Exact normalized signed offset `delta = d * 2^-3` for a 2-bit code in `0..4`.
#[must_use]
pub const fn decode_code2_to_delta(code2: u8) -> f32 {
    match code2 & 0x03 {
        0 => 0.0,
        1 => 0.125,
        2 => -0.25,
        _ => -0.125,
    }
}

/// Expand one sparse subgroup refinement byte (four 2-bit codes) into two
/// packed E2M1 bytes (little-endian `u16` matching `ae_expand_sg_to_e2m1x4`).
#[must_use]
pub const fn expand_subgroup_to_e2m1x4(raw: u8) -> u16 {
    let n0 = decode_code2_to_e2m1(raw & 0x03) as u16;
    let n1 = decode_code2_to_e2m1((raw >> 2) & 0x03) as u16;
    let n2 = decode_code2_to_e2m1((raw >> 4) & 0x03) as u16;
    let n3 = decode_code2_to_e2m1((raw >> 6) & 0x03) as u16;
    let b0 = (n1 << 4) | n0;
    let b1 = (n3 << 4) | n2;
    (b1 << 8) | b0
}

/// Host reference reconstruction of one 16-element NVFP4 + AutoFP6 group.
///
/// Mirrors `autoematic_fp6_gemv_batch` bit-for-bit:
/// `w[i] = (E2M1_LUT[base_nibble] + 0.25 * E2M1_LUT[ref_nibble]) * eff_scale`.
///
/// # Errors
/// Returns an error if `mask4` has high bits set, `popcount(mask4) > 2`, or
/// `refined_sg_bytes.len()` does not equal `mask4.count_ones()`.
pub fn reconstruct_autofp6_group(
    base_packed8: &[u8; 8],
    scale_ue4m3: u8,
    scale2: f32,
    mask4: u8,
    refined_sg_bytes: &[u8],
) -> Result<[f32; AUTOEMATIC_GROUP_SIZE]> {
    ensure!(
        mask4 & !0x0F == 0,
        "AutoEMatic subgroup mask {mask4:#x} has upper bits set"
    );
    let pop = mask4.count_ones();
    ensure!(
        pop <= AUTOEMATIC_MAX_REFINED_PER_GROUP,
        "AutoEMatic group popcount {pop} exceeds max {AUTOEMATIC_MAX_REFINED_PER_GROUP}"
    );
    ensure!(
        refined_sg_bytes.len() == pop as usize,
        "AutoEMatic refined subgroup slice len {} != popcount {pop}",
        refined_sg_bytes.len()
    );

    let mut ref_packed = [0_u8; 8];
    let mut sg_cursor = 0_usize;
    for sg in 0..AUTOEMATIC_SUBGROUPS_PER_GROUP {
        if (mask4 & (1_u8 << sg)) != 0 {
            let expanded = expand_subgroup_to_e2m1x4(refined_sg_bytes[sg_cursor]);
            ref_packed[sg * 2] = (expanded & 0xFF) as u8;
            ref_packed[sg * 2 + 1] = (expanded >> 8) as u8;
            sg_cursor += 1;
        }
    }

    let eff_scale = e4m3_to_f32(scale_ue4m3 & 0x7F) * scale2;
    let mut out = [0.0_f32; AUTOEMATIC_GROUP_SIZE];
    for b in 0..8 {
        let bv = base_packed8[b];
        let rv = ref_packed[b];
        let base_lo = AUTOEMATIC_E2M1_LUT[(bv & 0x0F) as usize];
        let base_hi = AUTOEMATIC_E2M1_LUT[(bv >> 4) as usize];
        let ref_lo = AUTOEMATIC_E2M1_LUT[(rv & 0x0F) as usize];
        let ref_hi = AUTOEMATIC_E2M1_LUT[(rv >> 4) as usize];
        out[b * 2] = (base_lo + AUTOEMATIC_REFINEMENT_SCALE * ref_lo) * eff_scale;
        out[b * 2 + 1] = (base_hi + AUTOEMATIC_REFINEMENT_SCALE * ref_hi) * eff_scale;
    }
    Ok(out)
}

// ── GPU-side sidecar descriptors ────────────────────────────────────────────

/// Device-resident AutoEMatic sparse refinement tensors for one projection.
#[derive(Clone, Copy, Debug)]
pub struct AutoEmaticProjectionSidecar {
    /// Packed 2-bit offsets: `[N, num_refined_sg]` U8.
    pub refine_codes: DevicePtr,
    /// Per-group 4-bit subgroup masks: `[K/16]` U8.
    pub refine_mask: DevicePtr,
    /// Exclusive prefix sums of `popcount(refine_mask)`: `[K/16]` U32.
    pub group_prefix: DevicePtr,
    /// Total number of refined 4-element subgroups across K.
    pub num_refined_sg: u32,
    /// Actual fraction of refined subgroups (`num_refined_sg / (K/4)`).
    pub rho_actual: f32,
    /// Epilogue scale applied to the refinement MMA accumulator (`0.25`).
    pub refinement_scale: f32,
    /// Output channels `N`.
    pub n: usize,
    /// Input channels `K`.
    pub k: usize,
}

impl AutoEmaticProjectionSidecar {
    /// Null placeholder (used when refinement is disabled for a projection).
    #[must_use]
    pub const fn null() -> Self {
        Self {
            refine_codes: DevicePtr::NULL,
            refine_mask: DevicePtr::NULL,
            group_prefix: DevicePtr::NULL,
            num_refined_sg: 0,
            rho_actual: 0.0,
            refinement_scale: AUTOEMATIC_REFINEMENT_SCALE,
            n: 0,
            k: 0,
        }
    }

    /// Whether no refinement buffers are attached.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.refine_codes == DevicePtr::NULL
    }

    /// Added bits-per-weight overhead of this sparse sidecar (`2.0 * rho_actual`).
    #[must_use]
    pub fn added_bpw(&self) -> f32 {
        2.0 * self.rho_actual
    }
}

/// One transformer layer's uploaded AutoEMatic FFN refinement sidecars.
#[derive(Clone, Copy, Debug)]
pub struct AutoEmaticFfnLayerWeights {
    pub gate: AutoEmaticProjectionSidecar,
    pub up: AutoEmaticProjectionSidecar,
    pub down: AutoEmaticProjectionSidecar,
}

// ── Fail-closed environment admission ───────────────────────────────────────

/// Immutable operator request for an authenticated AutoEMatic sidecar load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoEmaticSidecarRequest {
    layers: BTreeSet<usize>,
    path: PathBuf,
    sha256: [u8; 32],
    size: u64,
}

impl AutoEmaticSidecarRequest {
    /// Read the four-field AutoEMatic request from the environment.
    ///
    /// Returns `Ok(None)` when all four variables are absent or empty (default
    /// OFF). Fails closed on any partial specification or malformed field.
    ///
    /// # Errors
    /// Returns an error if only a subset of the four env vars are set, or if
    /// layers, path, digest, or size fail validation.
    pub fn from_env(total_layers: usize) -> Result<Option<Self>> {
        let values = [
            AUTOEMATIC_LAYERS_ENV,
            AUTOEMATIC_SIDECAR_ENV,
            AUTOEMATIC_SIDECAR_SHA256_ENV,
            AUTOEMATIC_SIDECAR_SIZE_ENV,
        ]
        .map(std::env::var_os);
        Self::from_os_values(
            total_layers,
            values[0].as_deref(),
            values[1].as_deref(),
            values[2].as_deref(),
            values[3].as_deref(),
        )
    }

    /// Parse from optional OS strings (used by [`Self::from_env`] and tests).
    ///
    /// # Errors
    /// Returns an error on invalid UTF-8 or invalid request fields.
    pub fn from_os_values(
        total_layers: usize,
        layers: Option<&OsStr>,
        path: Option<&OsStr>,
        sha256: Option<&OsStr>,
        size: Option<&OsStr>,
    ) -> Result<Option<Self>> {
        let layers = utf8_field(AUTOEMATIC_LAYERS_ENV, layers)?;
        let path = utf8_field(AUTOEMATIC_SIDECAR_ENV, path)?;
        let sha256 = utf8_field(AUTOEMATIC_SIDECAR_SHA256_ENV, sha256)?;
        let size = utf8_field(AUTOEMATIC_SIDECAR_SIZE_ENV, size)?;
        Self::from_values(total_layers, layers, path, sha256, size)
    }

    /// Parse from optional UTF-8 string slices.
    ///
    /// # Errors
    /// Fails closed on partial env configs or invalid values.
    pub fn from_values(
        total_layers: usize,
        layers: Option<&str>,
        path: Option<&str>,
        sha256: Option<&str>,
        size: Option<&str>,
    ) -> Result<Option<Self>> {
        let fields = [layers, path, sha256, size].map(|value| value.filter(|raw| !raw.is_empty()));
        if fields.iter().all(Option::is_none) {
            return Ok(None);
        }
        ensure!(
            (1..=MAX_AUTOEMATIC_LAYERS).contains(&total_layers),
            "AutoEMatic request requires total_layers in 1..={MAX_AUTOEMATIC_LAYERS}"
        );
        let [
            Some(layers_raw),
            Some(path_raw),
            Some(sha_raw),
            Some(size_raw),
        ] = fields
        else {
            bail!(
                "AutoEMatic request requires {AUTOEMATIC_LAYERS_ENV}, {AUTOEMATIC_SIDECAR_ENV}, \
                 {AUTOEMATIC_SIDECAR_SHA256_ENV}, and {AUTOEMATIC_SIDECAR_SIZE_ENV} together"
            );
        };

        let layers = parse_layers(layers_raw, total_layers)?;
        let path = PathBuf::from(path_raw);
        ensure!(
            path.is_absolute(),
            "{AUTOEMATIC_SIDECAR_ENV} must be an absolute path"
        );
        let sha256 = parse_sha256(sha_raw)?;
        let size = parse_size(size_raw)?;
        Ok(Some(Self {
            layers,
            path,
            sha256,
            size,
        }))
    }

    #[must_use]
    pub fn layers(&self) -> &BTreeSet<usize> {
        &self.layers
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

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

fn parse_layers(raw: &str, total_layers: usize) -> Result<BTreeSet<usize>> {
    ensure!(
        !raw.is_empty(),
        "{AUTOEMATIC_LAYERS_ENV} must not be empty when AutoEMatic is enabled"
    );
    let mut parsed = BTreeSet::new();
    for item in raw.split(',') {
        ensure!(
            !item.is_empty(),
            "{AUTOEMATIC_LAYERS_ENV} contains an empty item"
        );
        let mut bounds = item.split('-');
        let first_str = bounds
            .next()
            .context("AutoEMatic layer range missing lower bound")?;
        let first = canonical_usize(AUTOEMATIC_LAYERS_ENV, first_str)?;
        let last = match bounds.next() {
            Some(end) => canonical_usize(AUTOEMATIC_LAYERS_ENV, end)?,
            None => first,
        };
        ensure!(
            bounds.next().is_none(),
            "{AUTOEMATIC_LAYERS_ENV} item has multiple hyphens"
        );
        ensure!(first <= last, "{AUTOEMATIC_LAYERS_ENV} range is descending");
        ensure!(
            last < total_layers,
            "{AUTOEMATIC_LAYERS_ENV} layer {last} is out of range for total_layers={total_layers}"
        );
        for layer in first..=last {
            ensure!(
                parsed.insert(layer),
                "{AUTOEMATIC_LAYERS_ENV} repeats layer {layer}"
            );
        }
    }
    ensure!(
        !parsed.is_empty(),
        "{AUTOEMATIC_LAYERS_ENV} selects no layers"
    );
    Ok(parsed)
}

fn canonical_usize(name: &str, raw: &str) -> Result<usize> {
    ensure!(
        !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()),
        "{name} must use canonical decimal integers"
    );
    let value = raw
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} integer overflow"))?;
    ensure!(value.to_string() == raw, "{name} integer is not canonical");
    Ok(value)
}

fn parse_sha256(raw: &str) -> Result<[u8; 32]> {
    ensure!(
        raw.len() == 64
            && raw
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{AUTOEMATIC_SIDECAR_SHA256_ENV} must be exactly 64 lowercase hexadecimal characters"
    );
    let mut digest = [0_u8; 32];
    for (index, pair) in raw.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).context("validated hex bytes must be ASCII")?;
        digest[index] = u8::from_str_radix(text, 16).context("validated hex pair must parse")?;
    }
    Ok(digest)
}

fn parse_size(raw: &str) -> Result<u64> {
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{AUTOEMATIC_SIDECAR_SIZE_ENV} must be a canonical positive decimal integer");
    }
    let size = raw
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("{AUTOEMATIC_SIDECAR_SIZE_ENV} integer overflow"))?;
    ensure!(
        size > 0 && size.to_string() == raw,
        "{AUTOEMATIC_SIDECAR_SIZE_ENV} must be a canonical positive decimal integer"
    );
    Ok(size)
}

// ── Receipt & transactional session ─────────────────────────────────────────

/// Cryptographic audit receipt returned when every requested AutoEMatic layer
/// has been validated, uploaded, and installed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoEmaticSidecarReceipt {
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub size: u64,
    pub device: u64,
    pub inode: u64,
    pub requested_layers: Vec<usize>,
    pub validated_layers: Vec<usize>,
    pub uploaded_layers: Vec<usize>,
    pub installed_layers: Vec<usize>,
    pub requested_count: usize,
    pub validated_count: usize,
    pub uploaded_count: usize,
    pub installed_count: usize,
}

/// Transactional authenticated loader for an AutoEMatic AutoFP6 sidecar.
pub struct AutoEmaticSidecarSession {
    request: AutoEmaticSidecarRequest,
    _held_file: File,
    artifact: Vec<u8>,
    identity: FileIdentity,
    plans: BTreeMap<usize, LayerPlan>,
    uploaded: BTreeSet<usize>,
    installed: BTreeSet<usize>,
    awaiting_install: Option<usize>,
    poisoned: Option<String>,
}

impl AutoEmaticSidecarSession {
    /// Authenticate the held sidecar file (size, identity, SHA-256) and
    /// validate every requested layer's tensor census and prefix-sum invariants.
    ///
    /// # Errors
    /// Fails closed on any file-identity race, digest mismatch, manifest
    /// tampering, or invalid mask/prefix-sum invariant.
    pub fn prepare(
        request: AutoEmaticSidecarRequest,
        layer_prefixes: &[String],
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let (held_file, artifact, identity) = read_held_artifact(&request)?;
        ensure!(
            sha256(&artifact) == request.sha256(),
            "AutoEMatic sidecar SHA256 differs from the admitted digest"
        );
        let plans = validate_manifest(&artifact, &request, layer_prefixes, hidden, intermediate)?;
        ensure!(
            plans.keys().copied().collect::<BTreeSet<_>>() == *request.layers(),
            "AutoEMatic sidecar validation did not cover the full requested layer set"
        );
        Ok(Self {
            request,
            _held_file: held_file,
            artifact,
            identity,
            plans,
            uploaded: BTreeSet::new(),
            installed: BTreeSet::new(),
            awaiting_install: None,
            poisoned: None,
        })
    }

    /// Upload one prevalidated layer's AutoEMatic refinement tensors to GPU.
    ///
    /// Unrequested layers return `Ok(None)`. Every requested layer must be
    /// uploaded once and then passed to [`Self::mark_installed`] before the
    /// next layer upload begins.
    ///
    /// # Errors
    /// Fails closed and poisons the session on out-of-order upload or GPU
    /// allocation/copy failure (with automatic rollback of partial buffers).
    pub fn upload_layer(
        &mut self,
        layer: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<AutoEmaticFfnLayerWeights>> {
        self.upload_layer_with(layer, |artifact, plan| {
            upload_layer_plan_with(&GpuUploadTransaction { gpu }, artifact, plan)
        })
    }

    /// Confirm that the last uploaded layer was installed into the unpublished model.
    ///
    /// # Errors
    /// Poisons the session and returns an error if called out of order.
    pub fn mark_installed(&mut self, layer: usize) -> Result<()> {
        self.require_healthy()?;
        if !self.request.layers().contains(&layer) {
            return self.poison(format!(
                "cannot install unrequested AutoEMatic layer {layer}"
            ));
        }
        if self.awaiting_install != Some(layer) {
            return self.poison(format!(
                "AutoEMatic layer {layer} install is out of order; awaiting {:?}",
                self.awaiting_install
            ));
        }
        if !self.uploaded.contains(&layer) || !self.installed.insert(layer) {
            return self.poison(format!(
                "AutoEMatic layer {layer} install census is inconsistent"
            ));
        }
        self.awaiting_install = None;
        Ok(())
    }

    /// Finalize the session and emit an immutable audit receipt.
    ///
    /// # Errors
    /// Returns an error if the session is poisoned or any requested layer was
    /// not uploaded and installed.
    pub fn finish(self) -> Result<AutoEmaticSidecarReceipt> {
        if let Some(reason) = self.poisoned {
            bail!("AutoEMatic sidecar session is poisoned: {reason}");
        }
        ensure!(
            self.awaiting_install.is_none(),
            "AutoEMatic sidecar finish called before the last uploaded layer was installed"
        );
        let requested = self.request.layers().clone();
        let validated = self.plans.keys().copied().collect::<BTreeSet<_>>();
        ensure!(
            validated == requested,
            "AutoEMatic validated layer census is incomplete"
        );
        ensure!(
            self.uploaded == requested,
            "AutoEMatic uploaded layer census is incomplete"
        );
        ensure!(
            self.installed == requested,
            "AutoEMatic installed layer census is incomplete"
        );
        Ok(AutoEmaticSidecarReceipt {
            path: self.request.path().to_path_buf(),
            sha256: self.request.sha256(),
            size: self.request.size(),
            device: self.identity.device,
            inode: self.identity.inode,
            requested_layers: requested.iter().copied().collect(),
            validated_layers: validated.iter().copied().collect(),
            uploaded_layers: self.uploaded.iter().copied().collect(),
            installed_layers: self.installed.iter().copied().collect(),
            requested_count: requested.len(),
            validated_count: validated.len(),
            uploaded_count: self.uploaded.len(),
            installed_count: self.installed.len(),
        })
    }

    fn upload_layer_with<T>(
        &mut self,
        layer: usize,
        upload: impl FnOnce(&[u8], &LayerPlan) -> Result<T>,
    ) -> Result<Option<T>> {
        self.require_healthy()?;
        if !self.request.layers().contains(&layer) {
            return Ok(None);
        }
        if let Some(pending) = self.awaiting_install {
            return self.poison(format!(
                "AutoEMatic layer {pending} must be installed before uploading layer {layer}"
            ));
        }
        if self.uploaded.contains(&layer) {
            return self.poison(format!(
                "AutoEMatic layer {layer} was uploaded more than once"
            ));
        }
        let Some(plan) = self.plans.get(&layer).cloned() else {
            return self.poison(format!(
                "missing validated plan for AutoEMatic layer {layer}"
            ));
        };
        let uploaded = match upload(&self.artifact, &plan) {
            Ok(uploaded) => uploaded,
            Err(error) => {
                self.poisoned = Some(format!("AutoEMatic layer {layer} upload failed: {error:#}"));
                return Err(error).context(format!("upload AutoEMatic layer {layer}"));
            }
        };
        ensure!(self.uploaded.insert(layer));
        self.awaiting_install = Some(layer);
        Ok(Some(uploaded))
    }

    fn require_healthy(&self) -> Result<()> {
        if let Some(reason) = &self.poisoned {
            bail!("AutoEMatic sidecar session is poisoned: {reason}");
        }
        Ok(())
    }

    fn poison<T>(&mut self, reason: String) -> Result<T> {
        self.poisoned = Some(reason.clone());
        bail!("AutoEMatic sidecar session poisoned: {reason}")
    }
}

// ── File authentication (Linux O_NOFOLLOW) ──────────────────────────────────

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
fn read_held_artifact(request: &AutoEmaticSidecarRequest) -> Result<(File, Vec<u8>, FileIdentity)> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    ensure!(
        request.size() <= MAX_SIDECAR_BYTES,
        "AutoEMatic sidecar exceeds the {MAX_SIDECAR_BYTES}-byte safety limit"
    );
    let canonical_path = std::fs::canonicalize(request.path()).with_context(|| {
        format!(
            "canonicalize AutoEMatic sidecar {}",
            request.path().display()
        )
    })?;
    ensure!(
        canonical_path == request.path(),
        "AutoEMatic sidecar path must already be canonical"
    );
    let pre_metadata = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("stat AutoEMatic sidecar {}", request.path().display()))?;
    ensure!(
        pre_metadata.file_type().is_file(),
        "AutoEMatic sidecar must be a regular file"
    );
    ensure!(
        pre_metadata.nlink() == 1,
        "AutoEMatic sidecar must not have hard links"
    );
    let pre_identity = file_identity(&pre_metadata);
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(LINUX_O_NOFOLLOW)
        .open(request.path())
        .with_context(|| format!("open held AutoEMatic sidecar {}", request.path().display()))?;
    let opened_metadata = file
        .metadata()
        .context("stat held AutoEMatic sidecar descriptor")?;
    ensure!(
        opened_metadata.file_type().is_file(),
        "held AutoEMatic sidecar is not regular"
    );
    let opened_identity = file_identity(&opened_metadata);
    ensure!(
        pre_identity == opened_identity,
        "AutoEMatic sidecar changed between path stat and held open"
    );
    ensure!(
        opened_identity.size == request.size(),
        "AutoEMatic sidecar size differs from the admitted size"
    );
    let expected =
        usize::try_from(request.size()).context("AutoEMatic sidecar size does not fit host")?;
    let read_limit = expected
        .checked_add(1)
        .context("AutoEMatic sidecar read limit overflow")?;
    let mut artifact = Vec::new();
    artifact
        .try_reserve_exact(read_limit)
        .context("reserve authenticated AutoEMatic sidecar bytes")?;
    (&mut file)
        .take(read_limit as u64)
        .read_to_end(&mut artifact)
        .context("read held AutoEMatic sidecar bytes")?;
    ensure!(
        artifact.len() == expected,
        "AutoEMatic sidecar changed size while reading"
    );

    let post_descriptor = file
        .metadata()
        .context("restat held AutoEMatic sidecar descriptor")?;
    let post_path = std::fs::symlink_metadata(request.path())
        .with_context(|| format!("restat AutoEMatic sidecar {}", request.path().display()))?;
    ensure!(
        post_path.file_type().is_file(),
        "AutoEMatic sidecar path ceased to be regular"
    );
    ensure!(
        post_path.nlink() == 1,
        "AutoEMatic sidecar acquired a hard link while reading"
    );
    ensure!(
        file_identity(&post_descriptor) == opened_identity
            && file_identity(&post_path) == opened_identity,
        "AutoEMatic sidecar identity or metadata changed during authenticated read"
    );
    Ok((file, artifact, opened_identity))
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
fn read_held_artifact(
    _request: &AutoEmaticSidecarRequest,
) -> Result<(File, Vec<u8>, FileIdentity)> {
    bail!("fail-closed AutoEMatic sidecar sessions currently require Linux O_NOFOLLOW")
}

// ── Duplicate-free safetensors manifest validation ──────────────────────────

#[derive(Clone, Debug)]
struct ProjectionPlan {
    refine_codes: Range<usize>,
    refine_mask: Range<usize>,
    group_prefix: Range<usize>,
    meta_bytes: Range<usize>,
    num_refined_sg: u32,
    rho_actual: f32,
    refinement_scale: f32,
    n: usize,
    k: usize,
}

#[derive(Clone, Debug)]
struct LayerPlan {
    gate: ProjectionPlan,
    up: ProjectionPlan,
    down: ProjectionPlan,
}

fn validate_manifest(
    artifact: &[u8],
    request: &AutoEmaticSidecarRequest,
    layer_prefixes: &[String],
    hidden: usize,
    intermediate: usize,
) -> Result<BTreeMap<usize, LayerPlan>> {
    ensure!(
        hidden > 0 && hidden.is_multiple_of(AUTOEMATIC_GROUP_SIZE),
        "AutoEMatic hidden={hidden} must be a positive multiple of {AUTOEMATIC_GROUP_SIZE}"
    );
    ensure!(
        intermediate > 0 && intermediate.is_multiple_of(AUTOEMATIC_GROUP_SIZE),
        "AutoEMatic intermediate={intermediate} must be a positive multiple of {AUTOEMATIC_GROUP_SIZE}"
    );
    let header_len_bytes = artifact
        .get(..8)
        .context("AutoEMatic sidecar is shorter than the safetensors length prefix")?;
    let header_len_arr: [u8; 8] = header_len_bytes
        .try_into()
        .context("safetensors prefix must be 8 bytes")?;
    let header_len = usize::try_from(u64::from_le_bytes(header_len_arr))
        .context("AutoEMatic sidecar header length does not fit this host")?;
    ensure!(header_len > 0, "AutoEMatic sidecar header is empty");
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "AutoEMatic sidecar header exceeds {MAX_HEADER_BYTES} bytes"
    );
    let data_start = 8_usize
        .checked_add(header_len)
        .context("AutoEMatic sidecar header offset overflow")?;
    let header_bytes = artifact
        .get(8..data_start)
        .context("AutoEMatic sidecar header extends beyond the artifact")?;
    let header: UniqueJson =
        serde_json::from_slice(header_bytes).context("parse duplicate-free AutoEMatic header")?;
    let root = header.object("AutoEMatic sidecar header")?;

    let mut layers = BTreeMap::new();
    let mut all_ranges = Vec::new();
    for &layer in request.layers() {
        let layer_prefix = layer_prefixes
            .get(layer)
            .with_context(|| format!("AutoEMatic layer {layer} has no model prefix"))?;
        ensure!(
            !layer_prefix.is_empty(),
            "AutoEMatic layer {layer} prefix is empty"
        );
        let mlp_prefix = format!("{layer_prefix}.mlp.");
        let expected = expected_names(layer_prefix);
        let actual = root
            .keys()
            .filter(|name| name.starts_with(&mlp_prefix))
            .cloned()
            .collect::<BTreeSet<_>>();
        ensure!(
            actual == expected,
            "AutoEMatic layer {layer} tensor census differs: expected {expected:?}, got {actual:?}"
        );

        let gate = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.gate_proj"),
            intermediate,
            hidden,
        )?;
        let up = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.up_proj"),
            intermediate,
            hidden,
        )?;
        let down = projection_plan(
            root,
            artifact,
            data_start,
            &format!("{layer_prefix}.mlp.down_proj"),
            hidden,
            intermediate,
        )?;
        for proj in [&gate, &up, &down] {
            all_ranges.push(proj.refine_codes.clone());
            all_ranges.push(proj.refine_mask.clone());
            all_ranges.push(proj.group_prefix.clone());
            all_ranges.push(proj.meta_bytes.clone());
        }
        layers.insert(layer, LayerPlan { gate, up, down });
    }
    ensure!(
        layers.keys().copied().collect::<BTreeSet<_>>() == *request.layers(),
        "AutoEMatic validated layer census differs from request"
    );
    all_ranges.sort_by_key(|range| (range.start, range.end));
    for pair in all_ranges.windows(2) {
        ensure!(
            pair[0].end <= pair[1].start,
            "AutoEMatic requested tensor payloads overlap at {:?} and {:?}",
            pair[0],
            pair[1]
        );
    }
    Ok(layers)
}

fn expected_names(layer_prefix: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        for suffix in [
            "ae_refine_codes",
            "ae_refine_mask",
            "ae_group_prefix",
            "ae_meta",
        ] {
            names.insert(format!("{layer_prefix}.mlp.{proj}.{suffix}"));
        }
    }
    names
}

fn projection_plan(
    root: &BTreeMap<String, UniqueJson>,
    artifact: &[u8],
    data_start: usize,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<ProjectionPlan> {
    let groups = k / AUTOEMATIC_GROUP_SIZE;
    let refine_mask = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.ae_refine_mask"),
        "U8",
        &[groups],
    )?;
    let group_prefix = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.ae_group_prefix"),
        "U32",
        &[groups],
    )?;
    let meta_bytes = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.ae_meta"),
        "F32",
        &[2],
    )?;

    // Validate mask low-nibble + popcount <= 2 and exact exclusive prefix sum.
    let mask_slice = &artifact[refine_mask.clone()];
    let prefix_slice = &artifact[group_prefix.clone()];
    let mut running_sg = 0_u32;
    for (g, (&mask_byte, prefix_chunk)) in mask_slice
        .iter()
        .zip(prefix_slice.chunks_exact(4))
        .enumerate()
    {
        ensure!(
            mask_byte & !0x0F == 0,
            "{prefix}.ae_refine_mask[{g}]={mask_byte:#x} has upper bits set"
        );
        let pop = mask_byte.count_ones();
        ensure!(
            pop <= AUTOEMATIC_MAX_REFINED_PER_GROUP,
            "{prefix}.ae_refine_mask[{g}] popcount {pop} exceeds {AUTOEMATIC_MAX_REFINED_PER_GROUP}"
        );
        let p_arr: [u8; 4] = prefix_chunk
            .try_into()
            .context("prefix U32 chunk must be 4 bytes")?;
        let stored_prefix = u32::from_le_bytes(p_arr);
        ensure!(
            stored_prefix == running_sg,
            "{prefix}.ae_group_prefix[{g}]={stored_prefix} != expected exclusive prefix {running_sg}"
        );
        running_sg = running_sg
            .checked_add(pop)
            .context("AutoEMatic total refined subgroup count overflow")?;
    }
    ensure!(
        running_sg > 0,
        "{prefix}: AutoEMatic sidecar has zero refined subgroups"
    );

    let refine_codes = tensor_range(
        root,
        artifact.len(),
        data_start,
        &format!("{prefix}.ae_refine_codes"),
        "U8",
        &[n, running_sg as usize],
    )?;

    let meta_slice = &artifact[meta_bytes.clone()];
    let rho_arr: [u8; 4] = meta_slice[0..4]
        .try_into()
        .context("meta rho_actual must be 4 bytes")?;
    let scale_arr: [u8; 4] = meta_slice[4..8]
        .try_into()
        .context("meta refinement_scale must be 4 bytes")?;
    let rho_actual = f32::from_le_bytes(rho_arr);
    let refinement_scale = f32::from_le_bytes(scale_arr);
    let expected_rho = (running_sg as f32) / ((groups * AUTOEMATIC_SUBGROUPS_PER_GROUP) as f32);
    ensure!(
        rho_actual.is_finite()
            && rho_actual > 0.0
            && rho_actual <= 0.5
            && (rho_actual - expected_rho).abs() <= 1e-5,
        "{prefix}.ae_meta[0] rho_actual={rho_actual} disagrees with mask ratio {expected_rho}"
    );
    ensure!(
        refinement_scale.is_finite()
            && (refinement_scale - AUTOEMATIC_REFINEMENT_SCALE).abs() <= 1e-6,
        "{prefix}.ae_meta[1] refinement_scale={refinement_scale} != {AUTOEMATIC_REFINEMENT_SCALE}"
    );

    Ok(ProjectionPlan {
        refine_codes,
        refine_mask,
        group_prefix,
        meta_bytes,
        num_refined_sg: running_sg,
        rho_actual,
        refinement_scale,
        n,
        k,
    })
}

fn tensor_range(
    root: &BTreeMap<String, UniqueJson>,
    artifact_len: usize,
    data_start: usize,
    name: &str,
    dtype: &str,
    shape: &[usize],
) -> Result<Range<usize>> {
    let metadata = root
        .get(name)
        .with_context(|| format!("AutoEMatic sidecar is missing {name}"))?
        .object(name)?;
    ensure!(
        metadata.len() == 3
            && metadata.contains_key("dtype")
            && metadata.contains_key("shape")
            && metadata.contains_key("data_offsets"),
        "{name}: tensor metadata must contain exactly dtype, shape, and data_offsets"
    );
    ensure!(
        metadata["dtype"].string(name)? == dtype,
        "{name}: expected dtype {dtype}"
    );
    let actual_shape = metadata["shape"].usize_array(&format!("{name}.shape"))?;
    ensure!(
        actual_shape == shape,
        "{name}: wrong shape {actual_shape:?}, expected {shape:?}"
    );
    let offsets = metadata["data_offsets"].usize_array(&format!("{name}.data_offsets"))?;
    ensure!(
        offsets.len() == 2,
        "{name}: data_offsets must have length 2"
    );
    ensure!(offsets[0] <= offsets[1], "{name}: descending data offsets");
    let start = data_start
        .checked_add(offsets[0])
        .context("AutoEMatic tensor start offset overflow")?;
    let end = data_start
        .checked_add(offsets[1])
        .context("AutoEMatic tensor end offset overflow")?;
    ensure!(
        end <= artifact_len,
        "{name}: tensor payload is out of bounds"
    );
    let elements = shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .context("AutoEMatic tensor element-count overflow")
    })?;
    let bytes_per_element = match dtype {
        "U8" => 1,
        "U32" | "F32" => 4,
        _ => bail!("{name}: unsupported AutoEMatic dtype {dtype}"),
    };
    let expected_len = elements
        .checked_mul(bytes_per_element)
        .context("AutoEMatic tensor byte-count overflow")?;
    ensure!(
        end - start == expected_len,
        "{name}: tensor span {} != expected {expected_len}",
        end - start
    );
    Ok(start..end)
}

#[derive(Debug)]
enum UniqueJson {
    Null,
    Bool,
    Number(serde_json::Number),
    String(String),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

impl UniqueJson {
    fn object(&self, label: &str) -> Result<&BTreeMap<String, Self>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => bail!("{label} must be a JSON object"),
        }
    }

    fn string(&self, label: &str) -> Result<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => bail!("{label} must be a JSON string"),
        }
    }

    fn usize_array(&self, label: &str) -> Result<Vec<usize>> {
        let Self::Array(values) = self else {
            bail!("{label} must be a JSON array");
        };
        values
            .iter()
            .map(|value| match value {
                Self::Number(number) => number
                    .as_u64()
                    .context("expected an unsigned integer")
                    .and_then(|raw| usize::try_from(raw).context("integer does not fit this host")),
                _ => bail!("{label} entries must be unsigned integers"),
            })
            .collect()
    }
}

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("duplicate-free JSON")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJson::Bool)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJson::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJson::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(UniqueJson::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJson::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJson::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueJson::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some((key, value)) = map.next_entry()? {
            if values.insert(key, value).is_some() {
                return Err(serde::de::Error::custom("duplicate JSON object key"));
            }
        }
        Ok(UniqueJson::Object(values))
    }
}

// ── Transactional GPU upload & rollback ─────────────────────────────────────

trait UploadTransaction {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;
    fn copy_h2d(&self, bytes: &[u8], destination: DevicePtr) -> Result<()>;
    fn free(&self, pointer: DevicePtr) -> Result<()>;
}

struct GpuUploadTransaction<'a> {
    gpu: &'a dyn GpuBackend,
}

impl UploadTransaction for GpuUploadTransaction<'_> {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.gpu.alloc(bytes)
    }

    fn copy_h2d(&self, bytes: &[u8], destination: DevicePtr) -> Result<()> {
        self.gpu.copy_h2d(bytes, destination)
    }

    fn free(&self, pointer: DevicePtr) -> Result<()> {
        self.gpu.free(pointer)
    }
}

fn upload_layer_plan_with(
    transaction: &dyn UploadTransaction,
    artifact: &[u8],
    plan: &LayerPlan,
) -> Result<AutoEmaticFfnLayerWeights> {
    let mut allocations = Vec::with_capacity(9);
    let result = (|| {
        let gate = upload_projection(transaction, artifact, &plan.gate, &mut allocations)?;
        let up = upload_projection(transaction, artifact, &plan.up, &mut allocations)?;
        let down = upload_projection(transaction, artifact, &plan.down, &mut allocations)?;
        Ok(AutoEmaticFfnLayerWeights { gate, up, down })
    })();
    if result.is_err() {
        for pointer in allocations.into_iter().rev() {
            if let Err(error) = transaction.free(pointer) {
                tracing::error!(?pointer, %error, "failed to roll back partial AutoEMatic upload");
            }
        }
    }
    result
}

fn upload_projection(
    transaction: &dyn UploadTransaction,
    artifact: &[u8],
    plan: &ProjectionPlan,
    allocations: &mut Vec<DevicePtr>,
) -> Result<AutoEmaticProjectionSidecar> {
    let refine_codes = upload_bytes(
        transaction,
        &artifact[plan.refine_codes.clone()],
        allocations,
    )?;
    let refine_mask = upload_bytes(
        transaction,
        &artifact[plan.refine_mask.clone()],
        allocations,
    )?;
    let group_prefix = upload_bytes(
        transaction,
        &artifact[plan.group_prefix.clone()],
        allocations,
    )?;
    Ok(AutoEmaticProjectionSidecar {
        refine_codes,
        refine_mask,
        group_prefix,
        num_refined_sg: plan.num_refined_sg,
        rho_actual: plan.rho_actual,
        refinement_scale: plan.refinement_scale,
        n: plan.n,
        k: plan.k,
    })
}

fn upload_bytes(
    transaction: &dyn UploadTransaction,
    bytes: &[u8],
    allocations: &mut Vec<DevicePtr>,
) -> Result<DevicePtr> {
    let pointer = transaction.alloc(bytes.len())?;
    allocations.push(pointer);
    transaction.copy_h2d(bytes, pointer)?;
    Ok(pointer)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::weight_map::dual_plane::{float_to_e2m1, float_to_ue4m3};

    fn hex_sha256(bytes: &[u8]) -> String {
        let digest = sha256(bytes);
        let mut s = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(&mut s, "{b:02x}");
        }
        s
    }

    fn build_synthetic_safetensors(
        layer_prefix: &str,
        hidden: usize,
        intermediate: usize,
        corrupt_prefix: bool,
        corrupt_popcount: bool,
    ) -> Vec<u8> {
        let specs = [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ];
        let mut header = BTreeMap::<String, serde_json::Value>::new();
        let mut payload = Vec::<u8>::new();

        for (proj, n, k) in specs {
            let groups = k / AUTOEMATIC_GROUP_SIZE;
            let mut mask = vec![0_u8; groups];
            for (g, m) in mask.iter_mut().enumerate() {
                *m = if corrupt_popcount && g == 0 {
                    0b0111 // invalid popcount = 3
                } else if g % 2 == 0 {
                    0b0101 // popcount = 2 (sg 0 and sg 2)
                } else {
                    0b0000
                };
            }
            let mut prefix = Vec::<u32>::with_capacity(groups);
            let mut acc = 0_u32;
            for &m in &mask {
                prefix.push(acc);
                acc += m.count_ones();
            }
            if corrupt_prefix && !prefix.is_empty() {
                prefix[0] = 99;
            }
            let num_refined_sg = acc as usize;
            let codes = vec![0b11_10_01_00_u8; n * num_refined_sg];
            let rho_actual =
                (num_refined_sg as f32) / ((groups * AUTOEMATIC_SUBGROUPS_PER_GROUP) as f32);
            let meta = [rho_actual, AUTOEMATIC_REFINEMENT_SCALE];

            let mut push_tensor = |suffix: &str, dtype: &str, shape: Vec<usize>, raw: &[u8]| {
                let start = payload.len();
                payload.extend_from_slice(raw);
                let end = payload.len();
                let name = format!("{layer_prefix}.mlp.{proj}.{suffix}");
                header.insert(
                    name,
                    serde_json::json!({
                        "dtype": dtype,
                        "shape": shape,
                        "data_offsets": [start, end],
                    }),
                );
            };

            let prefix_bytes: Vec<u8> = prefix.iter().flat_map(|v| v.to_le_bytes()).collect();
            let meta_bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
            push_tensor("ae_refine_codes", "U8", vec![n, num_refined_sg], &codes);
            push_tensor("ae_refine_mask", "U8", vec![groups], &mask);
            push_tensor("ae_group_prefix", "U32", vec![groups], &prefix_bytes);
            push_tensor("ae_meta", "F32", vec![2], &meta_bytes);
        }

        let mut header_bytes = serde_json::to_vec(&header).unwrap();
        let pad = (8 - (header_bytes.len() % 8)) % 8;
        header_bytes.extend(std::iter::repeat_n(b' ', pad));

        let mut artifact = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        artifact.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        artifact.extend_from_slice(&header_bytes);
        artifact.extend_from_slice(&payload);
        artifact
    }

    struct MockTx {
        next_ptr: RefCell<u64>,
        live: RefCell<BTreeSet<u64>>,
        fail_on_alloc: Option<usize>,
        alloc_count: RefCell<usize>,
    }

    impl MockTx {
        fn new(fail_on_alloc: Option<usize>) -> Self {
            Self {
                next_ptr: RefCell::new(0x1000),
                live: RefCell::new(BTreeSet::new()),
                fail_on_alloc,
                alloc_count: RefCell::new(0),
            }
        }

        fn live_count(&self) -> usize {
            self.live.borrow().len()
        }
    }

    impl UploadTransaction for MockTx {
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            let mut count = self.alloc_count.borrow_mut();
            *count += 1;
            if self.fail_on_alloc == Some(*count) {
                bail!("injected OOM on allocation {}", *count);
            }
            let mut next = self.next_ptr.borrow_mut();
            let ptr = *next;
            *next += 0x1000;
            self.live.borrow_mut().insert(ptr);
            Ok(DevicePtr(ptr))
        }

        fn copy_h2d(&self, _bytes: &[u8], destination: DevicePtr) -> Result<()> {
            ensure!(self.live.borrow().contains(&destination.0));
            Ok(())
        }

        fn free(&self, pointer: DevicePtr) -> Result<()> {
            ensure!(self.live.borrow_mut().remove(&pointer.0));
            Ok(())
        }
    }

    #[test]
    fn code2_to_e2m1_matches_exact_autofp6_offsets() {
        let expected: [(u8, u8, f32); 4] = [
            (0, 0x0, 0.0),
            (1, 0x1, 0.125),
            (2, 0xA, -0.25),
            (3, 0x9, -0.125),
        ];
        for (code2, e2m1_code, delta) in expected {
            assert_eq!(decode_code2_to_e2m1(code2), e2m1_code);
            let four_delta = AUTOEMATIC_E2M1_LUT[e2m1_code as usize];
            assert!(
                (AUTOEMATIC_REFINEMENT_SCALE * four_delta - delta).abs() < 1e-9,
                "code {code2}: 0.25 * {four_delta} != {delta}"
            );
            assert_eq!(decode_code2_to_delta(code2), delta);
        }

        // Pack [0, 1, 2, 3] into one byte and check expanded E2M1 nibbles.
        let packed_sg = 0b11_10_01_00_u8;
        let expanded = expand_subgroup_to_e2m1x4(packed_sg);
        let b0 = (expanded & 0xFF) as u8;
        let b1 = (expanded >> 8) as u8;
        assert_eq!(b0 & 0x0F, 0x0);
        assert_eq!(b0 >> 4, 0x1);
        assert_eq!(b1 & 0x0F, 0xA);
        assert_eq!(b1 >> 4, 0x9);
    }

    #[test]
    fn reconstruct_autofp6_group_reduces_quantization_mse() {
        // Construct 16 weights sitting midway between NVFP4 grid points where
        // 2-bit offsets {-0.25, -0.125, 0, +0.125} cut error dramatically.
        let weights: [f32; 16] = [
            0.625, 1.125, -0.875, 1.375, // sg 0 (high residual -> refined)
            0.500, 1.000, -1.000, 1.500, // sg 1 (exact E2M1 -> unrefined)
            -2.125, 1.875, -0.125, 0.375, // sg 2 (high residual -> refined)
            2.000, -2.000, 3.000, -3.000, // sg 3 (exact E2M1 -> unrefined)
        ];
        let scale2 = 1.0_f32;
        let gmax = weights.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let scale_u8 = float_to_ue4m3(gmax / 6.0);
        let eff_scale = e4m3_to_f32(scale_u8) * scale2;

        let mut base_packed = [0_u8; 8];
        let mut base_vals = [0.0_f32; 16];
        let mut best_codes = [0_u8; 16];
        for i in 0..16 {
            let norm = weights[i] / eff_scale;
            let q = float_to_e2m1(norm);
            base_vals[i] = AUTOEMATIC_E2M1_LUT[q as usize];
            if i % 2 == 0 {
                base_packed[i / 2] |= q & 0x0F;
            } else {
                base_packed[i / 2] |= q << 4;
            }
            let resid = norm - base_vals[i];
            let mut best_c = 0_u8;
            let mut best_err = f32::INFINITY;
            for c in 0_u8..4 {
                let err = (resid - decode_code2_to_delta(c)).abs();
                if err < best_err {
                    best_err = err;
                    best_c = c;
                }
            }
            best_codes[i] = best_c;
        }

        let base_only = reconstruct_autofp6_group(&base_packed, scale_u8, scale2, 0, &[]).unwrap();
        let sg0 =
            best_codes[0] | (best_codes[1] << 2) | (best_codes[2] << 4) | (best_codes[3] << 6);
        let sg2 =
            best_codes[8] | (best_codes[9] << 2) | (best_codes[10] << 4) | (best_codes[11] << 6);
        let refined =
            reconstruct_autofp6_group(&base_packed, scale_u8, scale2, 0b0101, &[sg0, sg2]).unwrap();

        let mse = |approx: &[f32; 16]| -> f32 {
            weights
                .iter()
                .zip(approx.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / 16.0
        };
        let base_mse = mse(&base_only);
        let ref_mse = mse(&refined);
        assert!(
            ref_mse < base_mse * 0.5,
            "AutoFP6 refinement must slash residual MSE: base={base_mse}, refined={ref_mse}"
        );
    }

    #[test]
    fn sidecar_request_defaults_off_and_fails_closed() {
        assert_eq!(
            AutoEmaticSidecarRequest::from_values(64, None, None, None, None).unwrap(),
            None
        );
        assert_eq!(
            AutoEmaticSidecarRequest::from_values(64, Some(""), Some(""), Some(""), Some(""))
                .unwrap(),
            None
        );
        // Partial specification must fail closed.
        assert!(
            AutoEmaticSidecarRequest::from_values(64, Some("0,2-3"), None, None, None).is_err()
        );
        let digest = "a".repeat(64);
        // Relative path rejected.
        assert!(
            AutoEmaticSidecarRequest::from_values(
                64,
                Some("0"),
                Some("relative/sidecar.safetensors"),
                Some(&digest),
                Some("128"),
            )
            .is_err()
        );
        // Uppercase hex rejected.
        assert!(
            AutoEmaticSidecarRequest::from_values(
                64,
                Some("0"),
                Some("/tmp/sidecar.safetensors"),
                Some(&"A".repeat(64)),
                Some("128"),
            )
            .is_err()
        );
        // Valid 4-tuple succeeds.
        let req = AutoEmaticSidecarRequest::from_values(
            64,
            Some("1,4-5"),
            Some("/tmp/sidecar.safetensors"),
            Some(&digest),
            Some("4096"),
        )
        .unwrap()
        .expect("valid request must parse");
        assert_eq!(
            req.layers().iter().copied().collect::<Vec<_>>(),
            vec![1, 4, 5]
        );
        assert_eq!(req.size(), 4096);
    }

    #[test]
    fn sidecar_session_validates_rolls_back_and_rejects_tampering() {
        let unique = format!(
            "atlas-autoematic-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        let canonical_dir = std::fs::canonicalize(&dir).unwrap();

        let good_bytes = build_synthetic_safetensors("model.layers.0", 32, 64, false, false);
        let good_path = canonical_dir.join("good.safetensors");
        std::fs::write(&good_path, &good_bytes).unwrap();
        let good_sha = hex_sha256(&good_bytes);

        let prefixes = vec!["model.layers.0".to_owned(), "model.layers.1".to_owned()];

        // 1. Happy path end-to-end session + receipt.
        let req = AutoEmaticSidecarRequest::from_values(
            2,
            Some("0"),
            good_path.to_str(),
            Some(&good_sha),
            Some(&good_bytes.len().to_string()),
        )
        .unwrap()
        .unwrap();
        let mut session =
            AutoEmaticSidecarSession::prepare(req.clone(), &prefixes, 32, 64).unwrap();
        let tx = MockTx::new(None);
        let uploaded = session
            .upload_layer_with(0, |artifact, plan| {
                upload_layer_plan_with(&tx, artifact, plan)
            })
            .unwrap()
            .expect("requested layer 0 must upload");
        assert_eq!(tx.live_count(), 9);
        assert!(!uploaded.gate.is_null());
        assert!((uploaded.gate.added_bpw() - 0.5).abs() < 1e-5);
        // Unrequested layer returns None.
        assert!(
            session
                .upload_layer_with(1, |artifact, plan| upload_layer_plan_with(
                    &tx, artifact, plan
                ))
                .unwrap()
                .is_none()
        );
        session.mark_installed(0).unwrap();
        let receipt = session.finish().unwrap();
        assert_eq!(receipt.installed_layers, vec![0]);
        assert_eq!(receipt.installed_count, 1);

        // 2. Partial upload OOM rolls back every prior allocation to 0 live buffers.
        let mut rollback_session =
            AutoEmaticSidecarSession::prepare(req, &prefixes, 32, 64).unwrap();
        let failing_tx = MockTx::new(Some(5));
        let err = rollback_session.upload_layer_with(0, |artifact, plan| {
            upload_layer_plan_with(&failing_tx, artifact, plan)
        });
        assert!(err.is_err(), "injected 5th allocation failure must error");
        assert_eq!(
            failing_tx.live_count(),
            0,
            "transactional rollback must free all prior allocations"
        );
        assert!(
            rollback_session.finish().is_err(),
            "poisoned session must fail finish"
        );

        // 3. Tampered SHA256 rejected.
        let bad_sha_req = AutoEmaticSidecarRequest::from_values(
            2,
            Some("0"),
            good_path.to_str(),
            Some(&"0".repeat(64)),
            Some(&good_bytes.len().to_string()),
        )
        .unwrap()
        .unwrap();
        assert!(AutoEmaticSidecarSession::prepare(bad_sha_req, &prefixes, 32, 64).is_err());

        // 4. Corrupted exclusive prefix sum rejected even with matching SHA256.
        let bad_prefix = build_synthetic_safetensors("model.layers.0", 32, 64, true, false);
        let bad_prefix_path = canonical_dir.join("bad_prefix.safetensors");
        std::fs::write(&bad_prefix_path, &bad_prefix).unwrap();
        let bad_prefix_req = AutoEmaticSidecarRequest::from_values(
            2,
            Some("0"),
            bad_prefix_path.to_str(),
            Some(&hex_sha256(&bad_prefix)),
            Some(&bad_prefix.len().to_string()),
        )
        .unwrap()
        .unwrap();
        assert!(AutoEmaticSidecarSession::prepare(bad_prefix_req, &prefixes, 32, 64).is_err());

        // 5. Mask popcount > 2 rejected even with matching SHA256.
        let bad_pop = build_synthetic_safetensors("model.layers.0", 32, 64, false, true);
        let bad_pop_path = canonical_dir.join("bad_pop.safetensors");
        std::fs::write(&bad_pop_path, &bad_pop).unwrap();
        let bad_pop_req = AutoEmaticSidecarRequest::from_values(
            2,
            Some("0"),
            bad_pop_path.to_str(),
            Some(&hex_sha256(&bad_pop)),
            Some(&bad_pop.len().to_string()),
        )
        .unwrap()
        .unwrap();
        assert!(AutoEmaticSidecarSession::prepare(bad_pop_req, &prefixes, 32, 64).is_err());

        let _ = std::fs::remove_dir_all(&canonical_dir);
    }
}
