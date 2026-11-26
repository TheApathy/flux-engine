// SPDX-License-Identifier: AGPL-3.0-only

//! W3 FFN sidecar loader — mixed-precision 3-bit FFN byte-reduction lane.
//!
//! Active W3 loading runs exclusively through [`W3SidecarRequest`] and
//! [`W3SidecarSession`], which authenticate an offline-repacked 3-bit FFN
//! safetensors sidecar file and upload both GEMV (`[N, 3K/8]`) and host-
//! transposed GEMM (`[3K/8, N_pad64]`) copies for the requested layers.
//!
//! Gating (all four environment variables must be set together when enabled):
//!   * `ATLAS_FFN_W3_LAYERS="3,7,12"` (or `a-b` ranges) names the layers
//!   * `ATLAS_FFN_W3_SIDECAR=/canonical/path/to/w3_ffn_sidecar.safetensors`
//!   * `ATLAS_FFN_W3_SIDECAR_SHA256=<64-hex-char SHA-256 digest>`
//!   * `ATLAS_FFN_W3_SIDECAR_SIZE=<exact byte length>`
//!
//! Sidecar tensor names, per layer prefix `lp` (e.g.
//! `model.language_model.layers.3`) and proj in `{gate,up,down}_proj`:
//!   * `{lp}.mlp.{proj}.w3_weight`         U8  `[N, 3K/8]`
//!   * `{lp}.mlp.{proj}.w3_weight_scale`   U8  `[N, K/16]` (FP8-E4M3 bytes)
//!   * `{lp}.mlp.{proj}.w3_weight_scale_2` F32 `[1]`
//!
//! Failure policy: **fail-closed**. Partial env configuration, non-canonical
//! or symlinked paths, size/digest mismatches, missing/malformed tensors, or
//! upload errors abort model loading rather than silently falling back to W4.

use super::QuantizedWeight;

#[path = "w3_sidecar/admission.rs"]
mod admission;
pub use admission::W3SidecarRequest;

#[path = "w3_sidecar/digest.rs"]
mod digest;

#[path = "w3_sidecar/manifest.rs"]
mod manifest;

#[path = "w3_sidecar/session.rs"]
mod session;
pub use session::{W3SidecarReceipt, W3SidecarSession};

#[cfg(test)]
#[path = "w3_sidecar/admission_tests.rs"]
mod admission_tests;

#[cfg(test)]
#[path = "w3_sidecar/session_tests.rs"]
mod session_tests;

#[cfg(test)]
#[path = "w3_sidecar/session_upload_tests.rs"]
mod session_upload_tests;

/// One layer's uploaded W3 FFN weights: GEMV (`[N, 3K/8]`) + transposed
/// GEMM (`[3K/8, N_pad64]`) copies of gate / up / down.
pub struct W3FfnLayerWeights {
    pub gate: QuantizedWeight,
    pub up: QuantizedWeight,
    pub down: QuantizedWeight,
    pub gate_t: QuantizedWeight,
    pub up_t: QuantizedWeight,
    pub down_t: QuantizedWeight,
}
