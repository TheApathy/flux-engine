// SPDX-License-Identifier: AGPL-3.0-only

//! One process-wide snapshot keeps admission, allocation and receipts aligned.

use super::{OWNER_CAP, SELECTOR, requested};
use anyhow::{Result, ensure};
use std::ffi::OsStr;
use std::sync::OnceLock;

const CAP_SELECTOR: &str = "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OwnerCap(usize);

impl OwnerCap {
    pub(crate) const DEFAULT: Self = Self(OWNER_CAP);

    pub(super) fn bytes(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    pub(super) enabled: bool,
    pub(super) cap: OwnerCap,
}

fn parse(enabled: Option<&OsStr>, cap: Option<&OsStr>) -> Result<Selection> {
    let enabled = requested(enabled)?;
    let cap = match cap {
        None => OwnerCap::DEFAULT,
        Some(value) => {
            ensure!(enabled, "{CAP_SELECTOR} requires {SELECTOR}=1");
            match value.to_str() {
                Some("16") => OwnerCap(16 * 1024 * 1024),
                Some("64") => OwnerCap::DEFAULT,
                _ => anyhow::bail!("{CAP_SELECTOR} must be exactly 16 or 64"),
            }
        }
    };
    Ok(Selection { enabled, cap })
}

pub(super) fn selection() -> Result<Selection> {
    static SELECTION: OnceLock<std::result::Result<Selection, String>> = OnceLock::new();
    match SELECTION.get_or_init(|| {
        parse(
            std::env::var_os(SELECTOR).as_deref(),
            std::env::var_os(CAP_SELECTOR).as_deref(),
        )
        .map_err(|error| error.to_string())
    }) {
        Ok(selection) => Ok(*selection),
        Err(error) => anyhow::bail!("{error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_cap_preserves_default_for_enabled_and_disabled_modes() -> Result<()> {
        for enabled in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
            let selected = parse(enabled, None)?;
            assert_eq!(selected.cap.bytes(), 64 * 1024 * 1024);
            assert_eq!(selected.enabled, enabled == Some(OsStr::new("1")));
        }
        Ok(())
    }

    #[test]
    fn explicit_caps_require_bounded_owners_and_accept_only_exact_values() -> Result<()> {
        for (value, bytes) in [("16", 16 * 1024 * 1024), ("64", OWNER_CAP)] {
            let selected = parse(Some(OsStr::new("1")), Some(OsStr::new(value)))?;
            assert!(selected.enabled);
            assert_eq!(selected.cap.bytes(), bytes);
            for disabled in [None, Some(OsStr::new("0"))] {
                assert!(parse(disabled, Some(OsStr::new(value))).is_err());
            }
        }
        for value in [
            "",
            "0",
            "8",
            "32",
            "65",
            "016",
            "+16",
            "-16",
            " 16",
            "16 ",
            "16\n",
            "16.0",
            "16777216",
            "18446744073709551616",
            "\u{ff11}\u{ff16}",
        ] {
            assert!(parse(Some(OsStr::new("1")), Some(OsStr::new(value))).is_err());
        }
        for enabled in ["", "true", "01", " 1", "2"] {
            assert!(parse(Some(OsStr::new(enabled)), None).is_err());
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_cap_is_rejected_without_environment_mutation() {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse(Some(OsStr::new("1")), Some(OsStr::from_bytes(b"\xff"))).is_err());
    }

    #[test]
    fn sixteen_mib_layout_guard_and_membership_are_exact() -> Result<()> {
        use atlas_core::config::ModelConfig;
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.num_hidden_layers = 48;
        config.num_experts = 512;
        config.hidden_size = 2560;
        config.moe_intermediate_size = 640;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        let cap = parse(Some(OsStr::new("1")), Some(OsStr::new("16")))?.cap;
        let mut requested = 0;
        for layout in super::super::canonical_layouts(&config, cap)? {
            assert_eq!((layout.experts_per_owner, layout.owner_count), (18, 29));
            assert_eq!(
                (layout.payload_bytes, layout.requested_bytes),
                (471_859_200, 471_866_624)
            );
            let mut seen = Vec::new();
            for index in 0..layout.owner_count {
                let owner = layout.owner(index)?;
                assert_eq!(owner.bytes, if index < 28 { 16_589_056 } else { 7_373_056 });
                for expert in owner.first_expert..owner.first_expert + owner.experts {
                    let (weight, scale) = layout.offsets(owner, expert)?;
                    assert!(weight >= 256 && weight % 256 == 0 && scale % 256 == 0);
                    seen.push(expert);
                }
            }
            assert_eq!(seen, (0..512).collect::<Vec<_>>());
            requested += layout.requested_bytes;
        }
        assert_eq!(requested, 1_415_599_872);
        assert_eq!(requested * 48, 67_948_793_856);
        assert_eq!(super::super::admission_guard_delta(&config, cap)?, 21_760);
        Ok(())
    }

    #[test]
    fn traced_failure_is_not_wrapped_retried_or_installed_as_an_owner() -> Result<()> {
        use super::super::{allocate_owner, traced_allocation};
        use spark_runtime::gpu::DevicePtr;
        #[derive(Debug)]
        struct OriginalError;
        impl std::fmt::Display for OriginalError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("original allocation error")
            }
        }
        impl std::error::Error for OriginalError {}
        let mut owners = Vec::new();
        allocate_owner(&mut owners, 512, |_| Ok(DevicePtr(0x1000)))?;
        let mut calls = 0;
        let error = allocate_owner(&mut owners, 512, |_| {
            traced_allocation(|| {
                calls += 1;
                Err(anyhow::Error::new(OriginalError))
            })
        })
        .unwrap_err();
        assert!(error.downcast_ref::<OriginalError>().is_some());
        assert_eq!(calls, 1);
        assert_eq!(
            (owners.len(), owners[0].base, owners[0].bytes),
            (1, DevicePtr(0x1000), 512)
        );
        Ok(())
    }
}
