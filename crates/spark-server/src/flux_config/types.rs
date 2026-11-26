// SPDX-License-Identifier: AGPL-3.0-only

//! Domain value types and error definitions for [`super::FluxConfig`].

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use thiserror::Error;

/// Wrapper around an API secret that redacts its contents in [`fmt::Debug`].
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wrap a raw secret string.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the underlying secret value.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"<redacted>\"")
    }
}

/// KV cache storage precision (`FLUX_KV_CACHE_DTYPE`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvCacheDtype {
    /// Select checkpoint/profile default (`MODEL.toml`).
    #[default]
    Auto,
    /// Full BF16 KV cache.
    Bf16,
    /// FP8 E4M3 quantized KV cache.
    Fp8,
    /// TurboQuant+ compressed KV cache.
    TurboQuant,
    /// NVFP4 block-scaled KV cache.
    Nvfp4,
}

impl FromStr for KvCacheDtype {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp8" | "fp8_e4m3" => Ok(Self::Fp8),
            "turboquant" | "tq" | "tq+" => Ok(Self::TurboQuant),
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            _ => Err("one of auto|bf16|fp8|turboquant|nvfp4"),
        }
    }
}

/// Qwen4 C2–C32 native batched decode selection (`FLUX_QWEN4_NATIVE_BATCH`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NativeBatchMode {
    /// Batch natively when eligible, serializing unsupported frames with metrics.
    #[default]
    Auto,
    /// Require native batching or fail closed at startup (`1`).
    ForceNative,
    /// Always run C1 serially per active row (`0`).
    Serial,
}

impl FromStr for NativeBatchMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "1" | "true" | "on" | "native" => Ok(Self::ForceNative),
            "0" | "false" | "off" | "serial" => Ok(Self::Serial),
            _ => Err("one of auto|1|0"),
        }
    }
}

/// In-process `MemAvailable` floor watchdog mode (`FLUX_MEM_WATCH`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemWatchMode {
    /// Platform default (enabled on unified-memory GB10 hosts).
    #[default]
    Auto,
    /// Force watchdog enabled (`1` / `on` / `true`).
    On,
    /// Disable watchdog (`0` / `off` / `false`).
    Off,
}

impl FromStr for MemWatchMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "1" | "on" | "true" => Ok(Self::On),
            "0" | "off" | "false" => Ok(Self::Off),
            _ => Err("one of auto|on|off|1|0"),
        }
    }
}

/// MTP routed-expert source selection (`FLUX_MTP_SOURCE`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MtpSource {
    /// Checkpoint bank wins; sidecar fills in when absent.
    #[default]
    Auto,
    /// Always use checkpoint MTP experts; refuse sidecar override.
    Checkpoint,
    /// Replace checkpoint MTP experts with companion sidecar weights.
    Sidecar,
}

impl FromStr for MtpSource {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "checkpoint" => Ok(Self::Checkpoint),
            "sidecar" => Ok(Self::Sidecar),
            _ => Err("one of auto|checkpoint|sidecar"),
        }
    }
}

/// Opt-in DFlash lookup drafter mode (`FLUX_DFLASH_LOOKUP`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DflashLookupMode {
    /// Lookup drafter disabled.
    #[default]
    Off,
    /// Longest-suffix automaton matcher (`sam`).
    Sam,
    /// Fixed-window retrieval matcher (`retrieval`).
    Retrieval,
}

impl FromStr for DflashLookupMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "none" => Ok(Self::Off),
            "sam" => Ok(Self::Sam),
            "retrieval" => Ok(Self::Retrieval),
            _ => Err("one of off|sam|retrieval"),
        }
    }
}

/// Load-phase diagnostic capture policy (`FLUX_LOAD_DEBUG`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadDebugMode {
    /// Capture load diagnostics on failure or stall.
    #[default]
    Auto,
    /// Always persist load diagnostics.
    Always,
    /// Disable load-phase diagnostic capture.
    Off,
}

impl FromStr for LoadDebugMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "always" | "1" | "on" => Ok(Self::Always),
            "off" | "0" => Ok(Self::Off),
            _ => Err("one of auto|always|off"),
        }
    }
}

/// Runtime settings persistence target (`FLUX_SETTINGS_FILE`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SettingsFileMode {
    /// Persistence disabled (`off`).
    #[default]
    Off,
    /// Persist to the default runtime settings path (`on`).
    DefaultPath,
    /// Persist to an explicit operator-supplied path.
    Custom(PathBuf),
}

impl SettingsFileMode {
    pub(super) fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "" | "off" | "0" | "false" => Self::Off,
            "on" | "1" | "true" => Self::DefaultPath,
            _ => Self::Custom(PathBuf::from(trimmed)),
        }
    }
}

/// Error returned when parsing or validating [`super::FluxConfig`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FluxConfigError {
    /// A variable could not be parsed as its expected type or enum variant.
    #[error("invalid value for {key}={raw:?}: expected {expected}")]
    InvalidValue {
        /// Environment variable name.
        key: &'static str,
        /// Offending raw value.
        raw: String,
        /// Human-readable expected format.
        expected: &'static str,
    },
    /// A parsed numeric value violated its domain bounds.
    #[error("out-of-range value for {key}={raw:?}: {constraint}")]
    OutOfRange {
        /// Environment variable name.
        key: &'static str,
        /// Offending raw value.
        raw: String,
        /// Human-readable invariant description.
        constraint: &'static str,
    },
    /// Two or more operator fields conflict with each other.
    #[error("conflicting configuration: {message}")]
    Conflict {
        /// Description of the conflicting settings.
        message: String,
    },
}
