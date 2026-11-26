// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in releases of model-construction copies that no runtime route reads.
//! Each selector removes a buffer only after the copy every kernel actually
//! reads exists, so the bytes the kernels see are unchanged. Four selectors
//! (`SSM_BF16_QKVZ`, `FP8_SOURCES`, `SHARED_SCRATCH`, `UNUSED_MTP`) auto-enable
//! when unset on discrete GPUs (`is_integrated == Some(false)`); all selectors
//! default off on unified-memory / unknown topologies unless explicitly set.
//!
//! | selector | what it releases | where |
//! |---|---|---|
//! | `ATLAS_MEMLEAN_SSM_BF16_QKVZ` | the BF16 QKVZ concat of each dense-hybrid SSM layer, once its NVFP4 copy exists | `weight_loader/qwen35_dense.rs` |
//! | `ATLAS_MEMLEAN_FP8_SOURCES` | FP8 E4M3 checkpoint tensors the dense loader converted (SSM `in_proj_qkv/z`, `out_proj`; attention `q/k/v/o_proj`) | `weight_loader/qwen35_dense.rs` |
//! | `ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS` | the separate gate/up packed weights, replaced by views into the merged FlashInfer operand that holds the same bytes | `weight_loader/qwen35_dense/ffn.rs` |
//! | `ATLAS_MEMLEAN_SHARED_SCRATCH` | per-layer scratch: one W4A4/FlashInfer FFN activation arena, one cuBLASLt SSM e4m3 activation arena and one split-K FP32 workspace serve every layer | `weight_loader/qwen35_dense.rs`, `qwen35_dense/ffn.rs` |
//! | `ATLAS_MEMLEAN_UNUSED_MTP` | `mtp.*` store tensors of a dense checkpoint when no MTP head is built | `factory/build.rs` |
//! | `ATLAS_MEMLEAN_DRAFTER_BF16` | the DFlash drafter's BF16 `fc` and projections once `--dflash-quantization nvfp4` has quantized them | `factory/build.rs` |
//!
//! Every selector is strict: explicit `"0"` is off, `"1"` is on, unset uses the
//! topology default above (ON on discrete GPUs for the four auto selectors, OFF
//! otherwise), and any other value is a load error.

use anyhow::{Result, bail};

pub const SSM_BF16_QKVZ: &str = "ATLAS_MEMLEAN_SSM_BF16_QKVZ";
pub const FP8_SOURCES: &str = "ATLAS_MEMLEAN_FP8_SOURCES";
pub const FFN_GATE_UP_ALIAS: &str = "ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS";
pub const SHARED_SCRATCH: &str = "ATLAS_MEMLEAN_SHARED_SCRATCH";
pub const UNUSED_MTP: &str = "ATLAS_MEMLEAN_UNUSED_MTP";
pub const DRAFTER_BF16: &str = "ATLAS_MEMLEAN_DRAFTER_BF16";

/// Selectors proven safe to auto-enable on discrete GPUs
/// (`active_device_is_integrated() == Some(false)`), where dead resident
/// construction copies directly reduce available KV cache / batch concurrency.
/// Unified-memory hosts (`Some(true)`, e.g. GB10) and unknown topologies
/// (`None`) remain default-OFF. Explicit `"0"` opts out on any topology.
const fn is_discrete_auto_selector(name: &str) -> bool {
    matches!(
        name.as_bytes(),
        b"ATLAS_MEMLEAN_SSM_BF16_QKVZ"
            | b"ATLAS_MEMLEAN_FP8_SOURCES"
            | b"ATLAS_MEMLEAN_SHARED_SCRATCH"
            | b"ATLAS_MEMLEAN_UNUSED_MTP"
    )
}

/// Strict parse of one selector value (topology-neutral: unset is `false`).
pub fn parse(name: &str, value: Option<&str>) -> Result<bool> {
    parse_for_topology(name, value, None)
}

/// Strict parse of one selector value with device memory topology context.
///
/// Explicit `"0"` and `"1"` always win; any other value is a load error.
/// When unset (`None`), the four proven-safe selectors (`SSM_BF16_QKVZ`,
/// `FP8_SOURCES`, `SHARED_SCRATCH`, `UNUSED_MTP`) auto-enable on discrete GPUs
/// (`is_integrated == Some(false)`) while staying off on UMA (`Some(true)`) and
/// unknown (`None`) topologies.
pub fn parse_for_topology(
    name: &str,
    value: Option<&str>,
    is_integrated: Option<bool>,
) -> Result<bool> {
    match value {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        None => Ok(is_integrated == Some(false) && is_discrete_auto_selector(name)),
        Some(other) => bail!("{name} must be exactly 0 or 1, got {other:?}"),
    }
}

/// Read one selector from the process environment.
pub fn selected(name: &str) -> Result<bool> {
    let is_integrated = atlas_core::device::active_device_is_integrated();
    match std::env::var(name) {
        Ok(value) => parse_for_topology(name, Some(value.as_str()), is_integrated),
        Err(std::env::VarError::NotPresent) => parse_for_topology(name, None, is_integrated),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

/// The dense-hybrid (Qwen3.8-27B family) loader's selectors, read once per
/// load so every layer sees the same decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseLoadPlan {
    pub ssm_bf16_qkvz: bool,
    pub fp8_sources: bool,
    pub ffn_gate_up_alias: bool,
    pub shared_scratch: bool,
}

impl DenseLoadPlan {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            ssm_bf16_qkvz: selected(SSM_BF16_QKVZ)?,
            fp8_sources: selected(FP8_SOURCES)?,
            ffn_gate_up_alias: selected(FFN_GATE_UP_ALIAS)?,
            shared_scratch: selected(SHARED_SCRATCH)?,
        })
    }
}

/// Per-layer byte accounting for the `MEMLEAN_RELEASED` load-log line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReleasedBytes {
    pub bf16_qkvz: usize,
    pub fp8_sources: usize,
    pub ffn_gate_up: usize,
}

impl ReleasedBytes {
    pub fn total(&self) -> usize {
        self.bf16_qkvz + self.fp8_sources + self.ffn_gate_up
    }

    pub fn log(&self, layer: usize) {
        if self.total() == 0 {
            return;
        }
        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "MEMLEAN_RELEASED layer={layer} bf16_qkvz_mib={:.2} fp8_sources_mib={:.2} ffn_gate_up_mib={:.2} total_mib={:.2}",
            mib(self.bf16_qkvz),
            mib(self.fp8_sources),
            mib(self.ffn_gate_up),
            mib(self.total()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DRAFTER_BF16, FFN_GATE_UP_ALIAS, FP8_SOURCES, ReleasedBytes, SHARED_SCRATCH, SSM_BF16_QKVZ,
        UNUSED_MTP, parse, parse_for_topology,
    };

    #[test]
    fn selectors_are_strict_and_default_off() {
        assert!(!parse(FFN_GATE_UP_ALIAS, None).unwrap());
        assert!(!parse(FFN_GATE_UP_ALIAS, Some("0")).unwrap());
        assert!(parse(FFN_GATE_UP_ALIAS, Some("1")).unwrap());
        for bad in ["", "true", "on", " 1", "2"] {
            let error = parse(FFN_GATE_UP_ALIAS, Some(bad)).unwrap_err().to_string();
            assert!(
                error.contains("ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS must be exactly 0 or 1"),
                "{bad:?}: {error}"
            );
        }
    }

    #[test]
    fn discrete_topology_auto_enables_four_safe_selectors_with_zero_opt_out() {
        let auto_on = [SSM_BF16_QKVZ, FP8_SOURCES, SHARED_SCRATCH, UNUSED_MTP];
        let stay_off = [FFN_GATE_UP_ALIAS, DRAFTER_BF16];

        for name in auto_on {
            // Discrete GPU (is_integrated == Some(false)): unset auto-enables, "0" opts out.
            assert!(
                parse_for_topology(name, None, Some(false)).unwrap(),
                "{name}"
            );
            assert!(
                !parse_for_topology(name, Some("0"), Some(false)).unwrap(),
                "{name}"
            );
            assert!(
                parse_for_topology(name, Some("1"), Some(false)).unwrap(),
                "{name}"
            );
            // UMA GB10 (Some(true)) and unknown (None) remain default-OFF.
            assert!(
                !parse_for_topology(name, None, Some(true)).unwrap(),
                "{name}"
            );
            assert!(!parse_for_topology(name, None, None).unwrap(), "{name}");
            assert!(
                parse_for_topology(name, Some("1"), Some(true)).unwrap(),
                "{name}"
            );
        }

        for name in stay_off {
            assert!(
                !parse_for_topology(name, None, Some(false)).unwrap(),
                "{name}"
            );
            assert!(
                !parse_for_topology(name, None, Some(true)).unwrap(),
                "{name}"
            );
            assert!(
                parse_for_topology(name, Some("1"), Some(false)).unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn released_bytes_total_sums_every_family() {
        let bytes = ReleasedBytes {
            bf16_qkvz: 1,
            fp8_sources: 20,
            ffn_gate_up: 300,
        };
        assert_eq!(bytes.total(), 321);
        assert_eq!(ReleasedBytes::default().total(), 0);
    }
}
