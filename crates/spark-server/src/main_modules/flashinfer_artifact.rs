// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed admission for the qualified FlashInfer SM121 library.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use super::model_profile::ModelProfile;

const FLASHINFER_GATES: [&str; 3] = [
    "ATLAS_PREFILL_FFN_FLASHINFER",
    "ATLAS_PREFILL_PROJ_FLASHINFER",
    "ATLAS_PREFILL_SSM_FLASHINFER",
];
const FLASHINFER_LIBRARY: &str = "ATLAS_FLASHINFER_SM121_LIB";

/// Gates disabled single-threaded in `main()` on non-`aarch64` hosts before Tokio starts,
/// recorded so `preflight_live` can log the bypass once tracing is initialized.
static ARCH_DISABLED_GATES: OnceLock<Vec<&'static str>> = OnceLock::new();
static ARCH_BYPASS_LOGGED: AtomicBool = AtomicBool::new(false);

/// Whether the `(target_arch, compute_capability)` platform matches `(aarch64, SM121)`,
/// the sole architecture for which the pinned `libatlas_fi_fp4_sm121.so` SHA-256 artifact
/// is compiled and qualified.
///
/// Returns:
/// - `Some(false)` on any non-`aarch64` host (e.g. `x86_64`) or any non-SM121 GPU (`SM120`, `SM103`).
/// - `Some(true)` on `(aarch64, SM121)` (GB10).
/// - `None` on `aarch64` before CUDA initialization has reported compute capability.
pub(crate) fn platform_supports_sm121_flashinfer(
    target_arch: &str,
    compute_capability: Option<(u32, u32)>,
) -> Option<bool> {
    if target_arch != "aarch64" {
        return Some(false);
    }
    compute_capability.map(|cc| cc == (12, 1))
}

/// Disable enabled SM121 FlashInfer prefill gates in the process environment on
/// non-`aarch64` hosts before any worker threads exist.
///
/// Malformed gate values (anything other than `"1"`) are left untouched so
/// [`preflight_live`] still fails closed on invalid syntax.
///
/// # Safety
/// Caller MUST invoke this only while the process is single-threaded (before
/// constructing the Tokio runtime or spawning background threads).
pub(crate) unsafe fn disable_on_unsupported_arch() {
    if platform_supports_sm121_flashinfer(std::env::consts::ARCH, None) != Some(false) {
        return;
    }
    let mut disabled = Vec::new();
    for gate in FLASHINFER_GATES {
        if std::env::var_os(gate).as_deref() == Some(OsStr::new("1")) {
            // SAFETY: upheld by this function's `# Safety` precondition (single-threaded `main`).
            unsafe { std::env::set_var(gate, "0") };
            disabled.push(gate);
        }
    }
    if !disabled.is_empty() {
        let _ = ARCH_DISABLED_GATES.set(disabled);
    }
}

/// Strip enabled SM121 FlashInfer gates from a child process environment map before
/// `exec` when the current host/GPU is known not to be `(aarch64, SM121)`.
pub(crate) fn sanitize_reexec_env(env: &mut std::collections::BTreeMap<String, String>) {
    let arch = std::env::consts::ARCH;
    let cc = atlas_core::device::active_compute_capability();
    if platform_supports_sm121_flashinfer(arch, cc) != Some(false) {
        return;
    }
    for gate in FLASHINFER_GATES {
        if env.get(gate).map(String::as_str) == Some("1") {
            env.insert(gate.to_string(), "0".to_string());
        }
    }
}

fn log_arch_disabled_bypass_once(model_type: &str) {
    if model_type != "qwen3_5" {
        return;
    }
    if let Some(gates) = ARCH_DISABLED_GATES.get()
        && !ARCH_BYPASS_LOGGED.swap(true, Ordering::Relaxed)
    {
        tracing::warn!(
            target: "atlas_flashinfer",
            arch = std::env::consts::ARCH,
            compute_capability = ?atlas_core::device::active_compute_capability(),
            gates = ?gates,
            "FlashInfer SM121 artifact is qualified only for (aarch64, SM121); \
             disabled FlashInfer prefill gates at startup to use native prefill kernels"
        );
    }
}

fn unsupported_live_platform_error(
    gates: &[&str],
    target_arch: &str,
    compute_capability: Option<(u32, u32)>,
) -> String {
    let cc_desc = match compute_capability {
        Some((major, minor)) => format!("SM{major}{minor}"),
        None => "unknown SM".to_string(),
    };
    format!(
        "FlashInfer SM121 artifact is qualified only for (aarch64, SM121), got ({target_arch}, \
         {cc_desc}); cannot mutate process environment after worker threads start — unset or \
         set {gates:?}=0 to use native prefill kernels"
    )
}

/// Check the artifact a selected profile would need before a swap releases its live model.
///
/// An operator export has the same precedence as [`super::swap_env::reexec_env`]:
/// it wins over the selected built-in profile. Values installed by an earlier
/// profile are excluded because a re-exec removes them before applying the new one.
/// On non-`(aarch64, SM121)` platforms (e.g. `x86_64` or discrete `SM120`/`SM103`),
/// logs and bypasses the single pinned `aarch64/sm_121a` `.so` SHA-256 gate.
pub(crate) fn preflight_prospective(
    model_type: &str,
    profile: Option<&ModelProfile>,
) -> Result<(), String> {
    let arch = std::env::consts::ARCH;
    let cc = atlas_core::device::active_compute_capability();
    preflight_for_platform(
        arch,
        cc,
        true,
        model_type,
        profile,
        super::swap_env::operator_environment_value,
        spark_model::layers::ops::flashinfer_sm121_artifact::preflight_qualified_source,
        |gates| {
            tracing::info!(
                target: "atlas_flashinfer",
                arch,
                compute_capability = ?cc,
                gates = ?gates,
                "FlashInfer SM121 prospective preflight bypassed on non-(aarch64, SM121) platform"
            );
            Ok(())
        },
    )
}

/// Recheck the realized process environment on direct boot, in-process load, and re-exec.
///
/// On non-`aarch64` hosts, [`disable_on_unsupported_arch`] already disabled any `"1"`
/// gates single-threaded in `main()` and this logs that bypass once tracing is up.
/// On `aarch64` before GPU init, validates gate syntax and defers `.so` verification
/// to [`preflight_live_for_device`].
pub(crate) fn preflight_live(model_type: &str) -> Result<(), String> {
    log_arch_disabled_bypass_once(model_type);
    let arch = std::env::consts::ARCH;
    let cc = atlas_core::device::active_compute_capability();
    preflight_for_platform(
        arch,
        cc,
        true,
        model_type,
        None,
        |name| std::env::var_os(name),
        spark_model::layers::ops::flashinfer_sm121_artifact::preflight_qualified_source,
        |gates| Err(unsupported_live_platform_error(gates, arch, cc)),
    )
}

/// Finalize live FlashInfer SM121 preflight right after GPU initialization once
/// `compute_capability` is known.
///
/// Fails closed on `(aarch64, SM121)` (and conservatively on `aarch64` if compute
/// capability is unreadable). On non-`aarch64` hosts, enabled gates were already
/// disabled single-threaded in `main()`; if an enabled gate remains set on a
/// non-SM121 `aarch64` GPU (e.g. `SM120`/`SM103`), fails closed with an actionable
/// message rather than mutating process env from a multi-threaded runtime.
pub(crate) fn preflight_live_for_device(
    model_type: &str,
    compute_capability: Option<(u32, u32)>,
) -> Result<(), String> {
    log_arch_disabled_bypass_once(model_type);
    let arch = std::env::consts::ARCH;
    preflight_for_platform(
        arch,
        compute_capability,
        false,
        model_type,
        None,
        |name| std::env::var_os(name),
        spark_model::layers::ops::flashinfer_sm121_artifact::preflight_qualified_source,
        |gates| {
            Err(unsupported_live_platform_error(
                gates,
                arch,
                compute_capability,
            ))
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn preflight_for_platform(
    target_arch: &str,
    compute_capability: Option<(u32, u32)>,
    defer_unknown_cc: bool,
    model_type: &str,
    profile: Option<&ModelProfile>,
    mut operator_value: impl FnMut(&str) -> Option<OsString>,
    verify: impl FnMut(&Path) -> Result<(), String>,
    mut on_bypass: impl FnMut(&[&'static str]) -> Result<(), String>,
) -> Result<(), String> {
    if model_type != "qwen3_5" {
        return Ok(());
    }
    let mut enabled_gates = Vec::new();
    for name in FLASHINFER_GATES {
        if parse_gate(name, selected_value(name, profile, &mut operator_value))? {
            enabled_gates.push(name);
        }
    }
    if enabled_gates.is_empty() {
        return Ok(());
    }
    match platform_supports_sm121_flashinfer(target_arch, compute_capability) {
        Some(false) => on_bypass(&enabled_gates),
        None if defer_unknown_cc => Ok(()),
        Some(true) | None => preflight_model_with(model_type, profile, operator_value, verify),
    }
}

fn selected_value(
    name: &str,
    profile: Option<&ModelProfile>,
    operator_value: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<OsString> {
    operator_value(name).or_else(|| {
        profile
            .and_then(|candidate| candidate.env.get(name))
            .map(OsString::from)
    })
}

fn parse_gate(name: &str, value: Option<OsString>) -> Result<bool, String> {
    let Some(value) = value else {
        return Ok(false);
    };
    let value = value
        .to_str()
        .ok_or_else(|| format!("{name} is not valid Unicode; expected it to be unset, 0, or 1"))?;
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{name} must be unset, 0, or 1; got {value:?}")),
    }
}

fn preflight_model_with(
    model_type: &str,
    profile: Option<&ModelProfile>,
    operator_value: impl FnMut(&str) -> Option<OsString>,
    verify: impl FnMut(&Path) -> Result<(), String>,
) -> Result<(), String> {
    if model_type != "qwen3_5" {
        return Ok(());
    }
    preflight_with(profile, operator_value, verify)
}

fn preflight_with(
    profile: Option<&ModelProfile>,
    mut operator_value: impl FnMut(&str) -> Option<OsString>,
    mut verify: impl FnMut(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let mut enabled = false;
    for name in FLASHINFER_GATES {
        enabled |= parse_gate(name, selected_value(name, profile, &mut operator_value))?;
    }
    if !enabled {
        return Ok(());
    }

    let path =
        selected_value(FLASHINFER_LIBRARY, profile, &mut operator_value).ok_or_else(|| {
            format!("a FlashInfer prefill gate is 1 but {FLASHINFER_LIBRARY} is unset")
        })?;
    verify(Path::new(&path))
}

#[cfg(test)]
#[path = "flashinfer_artifact_tests.rs"]
mod tests;

#[cfg(test)]
mod platform_tests {
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    use super::{
        FLASHINFER_GATES, FLASHINFER_LIBRARY, ModelProfile, platform_supports_sm121_flashinfer,
        preflight_for_platform,
    };

    fn qwen38_profile() -> ModelProfile {
        ModelProfile {
            recipe_id: "qwen3.8/qwen3.8-27b-optimized-local".into(),
            model_type: "qwen3_5".into(),
            resident_gb: 66.0,
            dspark_extra_gb: 0.0,
            headroom_gb: 8.0,
            run_alone: false,
            env: BTreeMap::from([
                (FLASHINFER_GATES[0].to_string(), "1".to_string()),
                (
                    FLASHINFER_LIBRARY.to_string(),
                    "/missing/libatlas_fi_fp4_sm121.so".to_string(),
                ),
            ]),
            contract: None,
        }
    }

    #[test]
    fn platform_support_matches_only_aarch64_sm121() {
        assert_eq!(
            platform_supports_sm121_flashinfer("aarch64", Some((12, 1))),
            Some(true)
        );
        assert_eq!(
            platform_supports_sm121_flashinfer("aarch64", Some((12, 0))),
            Some(false)
        );
        assert_eq!(
            platform_supports_sm121_flashinfer("aarch64", Some((10, 3))),
            Some(false)
        );
        assert_eq!(platform_supports_sm121_flashinfer("aarch64", None), None);
        assert_eq!(
            platform_supports_sm121_flashinfer("x86_64", None),
            Some(false)
        );
        assert_eq!(
            platform_supports_sm121_flashinfer("x86_64", Some((12, 0))),
            Some(false)
        );
        assert_eq!(
            platform_supports_sm121_flashinfer("x86_64", Some((12, 1))),
            Some(false)
        );
    }

    #[test]
    fn non_sm121_platforms_bypass_sha256_gate_and_disable_enabled_gates() {
        let profile = qwen38_profile();
        for (arch, cc) in [
            ("x86_64", None),
            ("x86_64", Some((12, 0))),
            ("x86_64", Some((10, 3))),
            ("aarch64", Some((12, 0))),
            ("aarch64", Some((10, 3))),
        ] {
            let mut bypassed = Vec::new();
            let mut verified = false;
            preflight_for_platform(
                arch,
                cc,
                false,
                "qwen3_5",
                Some(&profile),
                |_| None,
                |_| {
                    verified = true;
                    Err("must not run SM121 SHA-256 check on non-SM121".into())
                },
                |gates| {
                    bypassed.extend(gates.iter().copied());
                    Ok(())
                },
            )
            .unwrap();
            assert!(!verified, "{arch}/{cc:?}");
            assert_eq!(bypassed, vec![FLASHINFER_GATES[0]], "{arch}/{cc:?}");
        }
    }

    #[test]
    fn aarch64_sm121_remains_fail_closed_and_defers_only_before_gpu_init() {
        let profile = qwen38_profile();

        // (aarch64, SM121) must still fail closed on SHA-256 mismatch.
        let err = preflight_for_platform(
            "aarch64",
            Some((12, 1)),
            false,
            "qwen3_5",
            Some(&profile),
            |_| None,
            |_| Err("SHA-256 mismatch".into()),
            |_| panic!("SM121 must not bypass"),
        )
        .unwrap_err();
        assert_eq!(err, "SHA-256 mismatch");

        // Before GPU init on aarch64 (cc=None, defer=true), defers .so check but still rejects bad syntax.
        let mut verified = false;
        preflight_for_platform(
            "aarch64",
            None,
            true,
            "qwen3_5",
            Some(&profile),
            |_| None,
            |_| {
                verified = true;
                Ok(())
            },
            |_| panic!("deferred aarch64 check must not bypass"),
        )
        .unwrap();
        assert!(!verified);

        let err = preflight_for_platform(
            "aarch64",
            None,
            true,
            "qwen3_5",
            None,
            |name| (name == FLASHINFER_GATES[0]).then(|| OsString::from("bad")),
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(err.contains("unset, 0, or 1"), "{err}");

        // Post-GPU-init on aarch64 with unreadable CC (defer=false) stays fail-closed.
        let err = preflight_for_platform(
            "aarch64",
            None,
            false,
            "qwen3_5",
            Some(&profile),
            |_| None,
            |_| Err("SHA-256 mismatch".into()),
            |_| panic!("post-init aarch64 without CC must fail closed"),
        )
        .unwrap_err();
        assert_eq!(err, "SHA-256 mismatch");
    }
}
