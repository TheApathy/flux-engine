// SPDX-License-Identifier: AGPL-3.0-only

//! Process-wide policy for `ATLAS_DETERMINISTIC_BATCH` (`ZERO_OR_ONE`, default off).
//!
//! When `ATLAS_DETERMINISTIC_BATCH=1`, every `M`-thresholded decode and verify
//! dispatch selector pins a single `M`-invariant, zero-atomic, fixed-order route
//! across `M = 1..=128` so concurrent batches (`B=1, 2, 4, 8, 16, ...`) produce
//! bit-exact identical hidden states and logits.

use std::ffi::OsStr;
use std::fmt;
#[cfg(not(test))]
use std::sync::OnceLock;

/// Environment variable gating `M`-invariant deterministic batch decode/verify.
///
/// Strict `ZERO_OR_ONE`:
/// - unset or `"0"` -> [`DeterministicBatchPolicy::Disabled`] (default; 100% baseline)
/// - `"1"` -> [`DeterministicBatchPolicy::Enabled`]
/// - any other value (including `""`, whitespace, `"true"`, or non-UTF-8) fails closed.
pub const DETERMINISTIC_BATCH_ENV: &str = "ATLAS_DETERMINISTIC_BATCH";

/// Maximum batch row count covered by the `M`-invariant deterministic batch contract.
pub const DETERMINISTIC_BATCH_MAX_ROWS: usize = 128;

/// Policy selecting baseline dynamic-`M` dispatch vs strict `M`-invariant dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeterministicBatchPolicy {
    /// Default baseline behavior: dynamic `M`-thresholded dispatch unchanged.
    #[default]
    Disabled,
    /// Opt-in strict `M`-invariant, zero-atomic, fixed-order dispatch (`M = 1..=128`).
    Enabled,
}

impl DeterministicBatchPolicy {
    /// Returns `true` when deterministic `M`-invariant batching is enabled.
    #[inline]
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }

    /// Canonical `ZERO_OR_ONE` string representation (`"0"` or `"1"`).
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "0",
            Self::Enabled => "1",
        }
    }
}

impl fmt::Display for DeterministicBatchPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when [`DETERMINISTIC_BATCH_ENV`] violates strict `ZERO_OR_ONE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeterministicBatchEnvError {
    /// The environment value contains non-UTF-8 bytes.
    NonUtf8,
    /// The environment value is valid UTF-8 but not exact `"0"` or `"1"`.
    InvalidValue(String),
}

impl fmt::Display for DeterministicBatchEnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => {
                write!(
                    f,
                    "{DETERMINISTIC_BATCH_ENV} contains non-UTF-8 bytes (expected \"0\" or \"1\")"
                )
            }
            Self::InvalidValue(raw) => {
                write!(
                    f,
                    "{DETERMINISTIC_BATCH_ENV}={raw:?} is invalid (expected exact \"0\" or \"1\")"
                )
            }
        }
    }
}

impl std::error::Error for DeterministicBatchEnvError {}

const _: () = {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<DeterministicBatchPolicy>();
    assert_send_sync_static::<DeterministicBatchEnvError>();
};

/// Parse a UTF-8 environment value for [`DETERMINISTIC_BATCH_ENV`] using strict
/// `ZERO_OR_ONE` rules.
///
/// # Errors
/// Returns [`DeterministicBatchEnvError::InvalidValue`] for any string other than
/// exact `"0"` or `"1"` (including `""`, whitespace-padded values, or `"true"`).
pub fn parse_deterministic_batch_str(
    raw: Option<&str>,
) -> std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError> {
    match raw {
        None | Some("0") => Ok(DeterministicBatchPolicy::Disabled),
        Some("1") => Ok(DeterministicBatchPolicy::Enabled),
        Some(other) => Err(DeterministicBatchEnvError::InvalidValue(other.to_owned())),
    }
}

/// Parse a raw OS environment value for [`DETERMINISTIC_BATCH_ENV`] using strict
/// `ZERO_OR_ONE` rules.
///
/// # Errors
/// Returns [`DeterministicBatchEnvError::NonUtf8`] if `raw` is not valid UTF-8, or
/// [`DeterministicBatchEnvError::InvalidValue`] for any string other than exact `"0"`
/// or `"1"`.
pub fn parse_deterministic_batch_os(
    raw: Option<&OsStr>,
) -> std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError> {
    let Some(raw) = raw else {
        return Ok(DeterministicBatchPolicy::Disabled);
    };
    let Some(text) = raw.to_str() else {
        return Err(DeterministicBatchEnvError::NonUtf8);
    };
    parse_deterministic_batch_str(Some(text))
}

/// Return the cached process-wide [`DeterministicBatchPolicy`] read from
/// [`DETERMINISTIC_BATCH_ENV`].
///
/// Fallible decode/verify/LM-head entry points propagate this error via `?` so
/// malformed or non-UTF-8 values fail closed instead of silently running
/// non-deterministic batch dispatch.
///
/// # Errors
/// Returns [`DeterministicBatchEnvError`] if [`DETERMINISTIC_BATCH_ENV`] contains
/// non-UTF-8 bytes or a value other than exact `"0"` or `"1"`.
pub fn deterministic_batch_policy()
-> std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError> {
    let read_live = || {
        let raw = std::env::var_os(DETERMINISTIC_BATCH_ENV);
        parse_deterministic_batch_os(raw.as_deref())
    };
    #[cfg(not(test))]
    {
        static POLICY: OnceLock<
            std::result::Result<DeterministicBatchPolicy, DeterministicBatchEnvError>,
        > = OnceLock::new();
        POLICY
            .get_or_init(|| {
                let parsed = read_live();
                match &parsed {
                    Ok(DeterministicBatchPolicy::Enabled) => {
                        tracing::info!(
                            selector = DETERMINISTIC_BATCH_ENV,
                            max_rows = DETERMINISTIC_BATCH_MAX_ROWS,
                            "ATLAS_DETERMINISTIC_BATCH=1 engaged: forcing M-invariant, \
                             zero-atomic, fixed-order decode/verify dispatch"
                        );
                    }
                    Ok(DeterministicBatchPolicy::Disabled) => {}
                    Err(err) => {
                        tracing::error!(
                            selector = DETERMINISTIC_BATCH_ENV,
                            error = %err,
                            "invalid ATLAS_DETERMINISTIC_BATCH value; failing closed"
                        );
                    }
                }
                parsed
            })
            .clone()
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Validate [`DETERMINISTIC_BATCH_ENV`] and return the active policy as an
/// [`anyhow::Result`], failing closed on invalid or non-UTF-8 values.
///
/// # Errors
/// Returns an error if [`DETERMINISTIC_BATCH_ENV`] is neither unset, `"0"`, nor `"1"`.
#[inline]
pub fn deterministic_batch_policy_checked() -> anyhow::Result<DeterministicBatchPolicy> {
    deterministic_batch_policy().map_err(anyhow::Error::from)
}

/// Return the active [`DeterministicBatchPolicy`] for dispatch selectors,
/// failing closed to [`DeterministicBatchPolicy::Disabled`] on invalid env values.
#[inline]
#[must_use]
pub fn deterministic_batch_policy_or_disabled() -> DeterministicBatchPolicy {
    deterministic_batch_policy().unwrap_or(DeterministicBatchPolicy::Disabled)
}

/// Return `true` when [`DETERMINISTIC_BATCH_ENV`] is set to exact `"1"`.
///
/// Fails closed to `false` (emitting a one-time warning) on malformed or non-UTF-8
/// values so invalid configuration can never silently engage an unintended route.
#[inline]
#[must_use]
pub fn deterministic_batch_enabled() -> bool {
    deterministic_batch_policy_or_disabled().is_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_zero_or_one_accepts_only_none_zero_and_one() {
        assert_eq!(
            parse_deterministic_batch_str(None),
            Ok(DeterministicBatchPolicy::Disabled)
        );
        assert_eq!(
            parse_deterministic_batch_str(Some("0")),
            Ok(DeterministicBatchPolicy::Disabled)
        );
        assert_eq!(
            parse_deterministic_batch_str(Some("1")),
            Ok(DeterministicBatchPolicy::Enabled)
        );
        assert_eq!(
            parse_deterministic_batch_os(None),
            Ok(DeterministicBatchPolicy::Disabled)
        );
        assert_eq!(
            parse_deterministic_batch_os(Some(OsStr::new("0"))),
            Ok(DeterministicBatchPolicy::Disabled)
        );
        assert_eq!(
            parse_deterministic_batch_os(Some(OsStr::new("1"))),
            Ok(DeterministicBatchPolicy::Enabled)
        );
    }

    #[test]
    fn parse_zero_or_one_rejects_empty_whitespace_and_invalid_tokens() {
        for bad in [
            "", " ", " 0", "0 ", " 1", "1 ", "\t1", "1\n", "true", "false", "yes", "no", "on",
            "off", "2", "-1", "01",
        ] {
            let err = parse_deterministic_batch_str(Some(bad)).expect_err("must reject bad token");
            assert_eq!(
                err,
                DeterministicBatchEnvError::InvalidValue(bad.to_owned()),
                "bad={bad:?}"
            );
            assert!(
                err.to_string().contains(DETERMINISTIC_BATCH_ENV),
                "display must name env var for bad={bad:?}"
            );
            assert_eq!(
                parse_deterministic_batch_os(Some(OsStr::new(bad))),
                Err(DeterministicBatchEnvError::InvalidValue(bad.to_owned())),
                "os bad={bad:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn parse_zero_or_one_rejects_non_utf8_os_str() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let bad = OsString::from_vec(vec![0xFF, 0xFE]);
        let err = parse_deterministic_batch_os(Some(bad.as_os_str()))
            .expect_err("must reject non-UTF-8 bytes");
        assert_eq!(err, DeterministicBatchEnvError::NonUtf8);
        assert!(err.to_string().contains("non-UTF-8"));
    }

    #[test]
    fn policy_defaults_and_formatting_match_contract() {
        assert_eq!(
            DeterministicBatchPolicy::default(),
            DeterministicBatchPolicy::Disabled
        );
        assert!(!DeterministicBatchPolicy::Disabled.is_enabled());
        assert!(DeterministicBatchPolicy::Enabled.is_enabled());
        assert_eq!(DeterministicBatchPolicy::Disabled.to_string(), "0");
        assert_eq!(DeterministicBatchPolicy::Enabled.to_string(), "1");
    }

    #[test]
    fn lm_head_and_tc_dense_components_are_m_invariant_across_1_to_128() {
        use crate::layers::qwen4_hyper::{
            BatchedVerifyTcComponent, batched_verify_tc_component_admits_with_policy,
        };
        use crate::model::impl_a3::{
            BatchedVerifyLmHeadRoute, lm_head_tc_enabled_with_policy,
            select_batched_verify_lm_head_route_with_policy,
            select_dense_bf16_lm_head_serial_gemv_with_policy,
            select_target_bf16_lm_head_serial_gemv_with_policy,
            select_target_fp8_lm_head_uses_coalesced_with_policy,
            should_auto_tc_batched_lm_head_for_policy,
        };

        const BATCH_SIZES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        for m in BATCH_SIZES {
            let route = select_batched_verify_lm_head_route_with_policy(
                DeterministicBatchPolicy::Enabled,
                true,
                m as u32,
                248_320,
                2_560,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
            );
            assert_eq!(
                route,
                BatchedVerifyLmHeadRoute::Exact,
                "LM head must pin Exact at M={m}"
            );

            for component in [
                BatchedVerifyTcComponent::LmHead,
                BatchedVerifyTcComponent::PleProj,
                BatchedVerifyTcComponent::FinalMixer,
                BatchedVerifyTcComponent::Fp8Dense,
                BatchedVerifyTcComponent::HcPrepare,
            ] {
                assert!(
                    !batched_verify_tc_component_admits_with_policy(
                        component,
                        true,
                        Some(1),
                        m,
                        false,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    "TC dense component {component:?} must stay disabled at M={m}"
                );
            }
        }

        // Baseline (Disabled) retains M-thresholded behavior unchanged.
        assert_eq!(
            select_batched_verify_lm_head_route_with_policy(
                DeterministicBatchPolicy::Disabled,
                true,
                8,
                248_320,
                2_560,
                true,
                true,
                false,
                true,
                true,
                true,
                true,
                true,
                true,
            ),
            BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM32
        );
        assert_eq!(
            select_batched_verify_lm_head_route_with_policy(
                DeterministicBatchPolicy::Disabled,
                true,
                64,
                248_320,
                2_560,
                true,
                true,
                false,
                true,
                true,
                true,
                true,
                true,
                true,
            ),
            BatchedVerifyLmHeadRoute::W4a16GemmCoalescedM64
        );
        assert!(batched_verify_tc_component_admits_with_policy(
            BatchedVerifyTcComponent::LmHead,
            true,
            None,
            8,
            false,
            DeterministicBatchPolicy::Disabled,
        ));
        assert!(!batched_verify_tc_component_admits_with_policy(
            BatchedVerifyTcComponent::LmHead,
            true,
            None,
            1,
            false,
            DeterministicBatchPolicy::Disabled,
        ));

        for m in [1u32, 2, 4, 8, 16, 32, 64, 128] {
            assert!(
                !select_target_fp8_lm_head_uses_coalesced_with_policy(
                    m,
                    DeterministicBatchPolicy::Enabled
                ),
                "FP8 target LM head must stay on serial dense_gemv_fp8w at M={m}"
            );
            assert!(
                select_target_bf16_lm_head_serial_gemv_with_policy(
                    m,
                    DeterministicBatchPolicy::Enabled
                ),
                "BF16 target LM head must stay on serial dense_gemv_bf16 at M={m}"
            );
            assert!(
                select_dense_bf16_lm_head_serial_gemv_with_policy(
                    m,
                    DeterministicBatchPolicy::Enabled
                ),
                "dense BF16 LM-head fallback must stay on serial dense_gemv_bf16 at M={m}"
            );
            assert!(
                !lm_head_tc_enabled_with_policy(true, DeterministicBatchPolicy::Enabled),
                "ATLAS_LM_HEAD_TC must stay disabled at M={m}"
            );
            assert!(
                !should_auto_tc_batched_lm_head_for_policy(
                    false,
                    false,
                    188,
                    m,
                    DeterministicBatchPolicy::Enabled
                ),
                "SM120 auto-TC batched LM head must stay disabled at M={m}"
            );
        }
        // Baseline (Disabled) retains M=1 GEMV vs M>=2 coalesced/auto-TC/dense_gemm dispatch.
        assert!(!select_target_fp8_lm_head_uses_coalesced_with_policy(
            1,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(select_target_fp8_lm_head_uses_coalesced_with_policy(
            8,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(select_target_bf16_lm_head_serial_gemv_with_policy(
            1,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(!select_target_bf16_lm_head_serial_gemv_with_policy(
            8,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(select_dense_bf16_lm_head_serial_gemv_with_policy(
            2,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(!select_dense_bf16_lm_head_serial_gemv_with_policy(
            3,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(!select_dense_bf16_lm_head_serial_gemv_with_policy(
            33,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(lm_head_tc_enabled_with_policy(
            true,
            DeterministicBatchPolicy::Disabled
        ));
        assert!(should_auto_tc_batched_lm_head_for_policy(
            false,
            false,
            188,
            8,
            DeterministicBatchPolicy::Disabled
        ));
    }

    #[test]
    fn deterministic_batch_policy_fails_closed_on_invalid_live_env_values() {
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        let _lock = test_env_lock();
        {
            let _unset = ScopedEnvVar::remove(DETERMINISTIC_BATCH_ENV);
            assert_eq!(
                deterministic_batch_policy(),
                Ok(DeterministicBatchPolicy::Disabled)
            );
            assert_eq!(
                deterministic_batch_policy_checked().unwrap(),
                DeterministicBatchPolicy::Disabled
            );
        }
        {
            let _zero = ScopedEnvVar::set(DETERMINISTIC_BATCH_ENV, "0");
            assert_eq!(
                deterministic_batch_policy(),
                Ok(DeterministicBatchPolicy::Disabled)
            );
            assert_eq!(
                deterministic_batch_policy_checked().unwrap(),
                DeterministicBatchPolicy::Disabled
            );
        }
        {
            let _one = ScopedEnvVar::set(DETERMINISTIC_BATCH_ENV, "1");
            assert_eq!(
                deterministic_batch_policy(),
                Ok(DeterministicBatchPolicy::Enabled)
            );
            assert_eq!(
                deterministic_batch_policy_checked().unwrap(),
                DeterministicBatchPolicy::Enabled
            );
        }
        for bad in ["", " ", " 1 ", "true", "false", "yes", "on", "2", "-1"] {
            let _bad = ScopedEnvVar::set(DETERMINISTIC_BATCH_ENV, bad);
            let err = deterministic_batch_policy().expect_err("invalid env must fail closed");
            assert_eq!(
                err,
                DeterministicBatchEnvError::InvalidValue(bad.to_owned())
            );
            let checked = deterministic_batch_policy_checked()
                .expect_err("checked policy must propagate error via ?");
            assert!(checked.to_string().contains(DETERMINISTIC_BATCH_ENV));
        }
    }

    #[test]
    fn fast_proj_nvfp4_and_splitk_selectors_are_m_invariant_across_1_to_128() {
        use crate::layers::qwen4_fast_proj::nvfp4::Nvfp4FastProjMode;
        use crate::layers::qwen4_fast_proj::{
            DirectNvfp4GemmRoute, SharedExpertDownTile, SharedExpertGateUpTile,
            batched_verify_dense_proj_tc_admits_with_policy,
            coalesced_nvfp4_batched_verify_shape_eligible_with_policy,
            coalesced_nvfp4_prefers_fp8_cached_with_policy,
            direct_nvfp4_batched_verify_shape_eligible_with_policy,
            select_coalesced_nvfp4_tile_for_shape_v2_with_sms_and_policy,
            select_coalesced_nvfp4_tile_for_shape_with_policy,
            select_direct_nvfp4_gemm_route_with_policy, select_narrow_grid_split_k_z_with_policy,
            select_shared_expert_nvfp4_t_tiles_with_policy,
        };
        use crate::layers::{
            resolve_narrow_proj_splitk_with_policy, resolve_proj_splitk_with_policy,
        };

        const BATCH_SIZES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        for (n, k) in [
            (512usize, 2_560usize),
            (2_560, 6_144),
            (16_384, 2_560),
            (248_320, 2_560),
        ] {
            let expected_v1 = select_coalesced_nvfp4_tile_for_shape_with_policy(
                1,
                n,
                k,
                DeterministicBatchPolicy::Enabled,
            );
            assert!(expected_v1.is_some());
            for m in BATCH_SIZES {
                assert_eq!(
                    select_coalesced_nvfp4_tile_for_shape_with_policy(
                        m,
                        n,
                        k,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    expected_v1,
                    "v1 coalesced tile drifted at M={m} shape=({n},{k})"
                );
                for sm in [48u32, 188u32] {
                    assert_eq!(
                        select_coalesced_nvfp4_tile_for_shape_v2_with_sms_and_policy(
                            m,
                            n,
                            k,
                            sm,
                            DeterministicBatchPolicy::Enabled,
                        ),
                        expected_v1,
                        "v2 coalesced tile drifted at M={m} sm={sm} shape=({n},{k})"
                    );
                }
                assert_eq!(
                    select_narrow_grid_split_k_z_with_policy(
                        m,
                        n,
                        k,
                        188,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    None,
                    "narrow grid split-K must be disabled at M={m}"
                );
                assert_eq!(
                    select_direct_nvfp4_gemm_route_with_policy(
                        m,
                        n,
                        k,
                        true,
                        true,
                        true,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    Some(DirectNvfp4GemmRoute::Pipe),
                    "direct NVFP4 route must stay M-invariant at M={m}"
                );
                assert!(!batched_verify_dense_proj_tc_admits_with_policy(
                    m,
                    n,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!coalesced_nvfp4_prefers_fp8_cached_with_policy(
                    m,
                    n,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!coalesced_nvfp4_batched_verify_shape_eligible_with_policy(
                    true,
                    true,
                    m,
                    n,
                    k,
                    n,
                    None,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!direct_nvfp4_batched_verify_shape_eligible_with_policy(
                    true,
                    true,
                    m,
                    n,
                    k,
                    n,
                    None,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!Nvfp4FastProjMode::All.admits_call_with_policy(
                    m,
                    n,
                    k,
                    n,
                    None,
                    true,
                    32,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify_with_policy(
                    true,
                    m,
                    n,
                    k,
                    32,
                    DeterministicBatchPolicy::Enabled,
                ));
            }
        }

        for m in [2usize, 4, 8, 16, 32, 64, 128] {
            assert_eq!(
                select_shared_expert_nvfp4_t_tiles_with_policy(
                    m,
                    DeterministicBatchPolicy::Enabled
                ),
                Some((SharedExpertGateUpTile::Sk4M16, SharedExpertDownTile::M16)),
                "shared expert tile must stay fixed at M={m}"
            );
        }
        assert_ne!(
            select_shared_expert_nvfp4_t_tiles_with_policy(8, DeterministicBatchPolicy::Disabled),
            select_shared_expert_nvfp4_t_tiles_with_policy(64, DeterministicBatchPolicy::Disabled),
        );

        for sm in [48u32, 188u32] {
            assert_eq!(
                resolve_proj_splitk_with_policy(Some("4"), sm, DeterministicBatchPolicy::Enabled),
                0
            );
            assert_eq!(
                resolve_narrow_proj_splitk_with_policy(None, sm, DeterministicBatchPolicy::Enabled,),
                0
            );
        }
        assert_eq!(
            resolve_narrow_proj_splitk_with_policy(None, 188, DeterministicBatchPolicy::Disabled),
            8
        );
    }
}
