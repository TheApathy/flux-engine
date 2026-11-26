// SPDX-License-Identifier: AGPL-3.0-only

//! Which bf16 -> MXFP8 x quantiser entry a W4A8 bundle loads. Dependency-free (std only) so the
//! standalone contract test (`tests/slot_bf16_fused_static.rs`, `rustc --test`) runs the real
//! function instead of pinning its text.

use std::borrow::Cow;

/// The quad-lane twin of `qwen4_mxfp8_quant_rows_v16`. It is defined in the V16 module; the V28
/// module `#include`s the V16 source as its arithmetic SSOT, so it carries the same entry.
pub(crate) const XQ_QUANT_KERNEL_V16Q4: &str = "qwen4_mxfp8_quant_rows_v16q4";
/// Fused quad-lane MXFP8 row quantiser + shared-expert gate scalar precomputation (`gatepre`).
pub(crate) const XQ_QUANT_KERNEL_V16Q4_GATE: &str = "qwen4_mxfp8_quant_rows_v16q4_gate";
/// Every W4A8 set ships its own row quantiser under this prefix (`..._v10`, `..._v12`, `..._v14`,
/// `..._v16`, `..._v28`, and whatever a later set adds).
const XQ_QUANT_KERNEL_PREFIX: &str = "qwen4_mxfp8_quant_rows_";
/// The W4A8 sets whose module carries the twin: V16 itself and V28 (V16 + packed B scales, whose
/// `qwen4_mxfp8_quant_rows_v28` is V16's quantiser under the V28 tag).
pub(crate) const XQ_QUANT_TWIN_SETS: [&str; 2] = ["v16", "v28"];

/// The x quantiser entry for the loaded W4A8 set `w4a8_tag` ("v10", "v12", "v14", "v16", "v28", ...).
///
/// Selector off: the set's own entry, `qwen4_mxfp8_quant_rows_<tag>`, exactly what the loader did
/// before the selector existed; the tag list is open, so a set added later brings its own entry
/// without a change here. Selector on: the v16q4 twin, only on a set whose module carries it
/// ([`XQ_QUANT_TWIN_SETS`]); every other set is refused so the selector fails closed instead of
/// loading an entry the module does not have.
pub(crate) fn xq_quant_kernel(
    fused: bool,
    w4a8_tag: &str,
) -> Result<Cow<'static, str>, &'static str> {
    match (fused, w4a8_tag) {
        (false, tag) => Ok(Cow::Owned(format!("{XQ_QUANT_KERNEL_PREFIX}{tag}"))),
        (true, tag) if XQ_QUANT_TWIN_SETS.contains(&tag) => {
            Ok(Cow::Borrowed(XQ_QUANT_KERNEL_V16Q4))
        }
        (true, _) => Err("requires ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=2 or 7 (the V16 or V28 set)"),
    }
}
