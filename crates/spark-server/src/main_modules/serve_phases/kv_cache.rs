// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-budget + KV-cache dtype resolution.

use anyhow::{Context, Result};
use std::ffi::OsStr;

use atlas_core::config::ModelConfig;

use crate::cli;

const QWEN4_WIDE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_WIDE";
const QWEN4_MULTI_CHUNK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MULTI_CHUNK";
const MAX_BATCH_TOKENS_SELECTOR: &str = "ATLAS_MAX_BATCH_TOKENS";

fn selector_text<'a>(name: &str, value: Option<&'a OsStr>) -> Result<Option<&'a str>> {
    value
        .map(|value| {
            value
                .to_str()
                .with_context(|| format!("{name} must be valid Unicode"))
        })
        .transpose()
}

fn validate_qwen4_prefill_geometry_values(
    is_qwen4: bool,
    wide: Option<&str>,
    multi_chunk: Option<&str>,
    prefill_budget: usize,
) -> Result<()> {
    if !is_qwen4 {
        return Ok(());
    }
    let wide = match wide {
        None | Some("0") => return Ok(()),
        Some("2048") => 2048,
        Some("4096") => 4096,
        Some("8192") => 8192,
        Some("16384") => 16384,
        Some("32768") => 32768,
        Some(other) => anyhow::bail!(
            "{QWEN4_WIDE_SELECTOR} must be absent, 0, 2048, 4096, 8192, 16384, or 32768; got {other:?}"
        ),
    };
    anyhow::ensure!(
        multi_chunk == Some("1"),
        "{QWEN4_WIDE_SELECTOR}={wide} requires {QWEN4_MULTI_CHUNK_SELECTOR}=1"
    );
    anyhow::ensure!(
        wide == prefill_budget,
        "{QWEN4_WIDE_SELECTOR}={wide} requires the resolved scheduler prefill chunk to be {wide}, \
         but --max-prefill-tokens resolved to {prefill_budget}. This mismatch suppresses the \
         selected Qwen4 fast-prefill route. Set {QWEN4_WIDE_SELECTOR}={prefill_budget} or \
         --max-prefill-tokens {wide}."
    );
    Ok(())
}

/// The selected wide geometry is also the scheduler chunk contract.  Reject a
/// mismatch before model construction instead of silently routing a very wide
/// chunk through the ordinary fallback while level-3 PLE read-ahead is armed.
pub(crate) fn validate_qwen4_prefill_geometry(
    config: &ModelConfig,
    prefill_budget: usize,
) -> Result<()> {
    let wide = std::env::var_os(QWEN4_WIDE_SELECTOR);
    let multi_chunk = std::env::var_os(QWEN4_MULTI_CHUNK_SELECTOR);
    validate_qwen4_prefill_geometry_os_values(
        config.is_qwen4_exp(),
        wide.as_deref(),
        multi_chunk.as_deref(),
        prefill_budget,
    )
}

fn validate_qwen4_prefill_geometry_os_values(
    is_qwen4: bool,
    wide: Option<&OsStr>,
    multi_chunk: Option<&OsStr>,
    prefill_budget: usize,
) -> Result<()> {
    if !is_qwen4 {
        return Ok(());
    }
    let wide = selector_text(QWEN4_WIDE_SELECTOR, wide)?;
    // Disabled WIDE does not consume the multi-chunk selector.
    if matches!(wide, None | Some("0")) {
        return Ok(());
    }
    let multi_chunk = selector_text(QWEN4_MULTI_CHUNK_SELECTOR, multi_chunk)?;
    validate_qwen4_prefill_geometry_values(true, wide, multi_chunk, prefill_budget)
}

pub(crate) struct PrefillBudget {
    pub(crate) prefill_budget: usize,
    pub(crate) max_batch_tokens: usize,
}

pub(crate) fn resolve_prefill_budget(
    args: &cli::ServeArgs,
    ssm_prefill_chunk: usize,
) -> Result<PrefillBudget> {
    let override_value = std::env::var_os(MAX_BATCH_TOKENS_SELECTOR);
    let override_value = selector_text(MAX_BATCH_TOKENS_SELECTOR, override_value.as_deref())?;
    resolve_prefill_budget_with_override(args, ssm_prefill_chunk, override_value)
}

fn parse_max_batch_tokens_override(value: Option<&str>, minimum: usize) -> Result<usize> {
    let Some(value) = value else {
        return Ok(minimum);
    };
    anyhow::ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "{MAX_BATCH_TOKENS_SELECTOR} must contain only ASCII decimal digits"
    );
    let tokens = value
        .parse::<usize>()
        .with_context(|| format!("{MAX_BATCH_TOKENS_SELECTOR} exceeds usize capacity"))?;
    anyhow::ensure!(
        tokens >= minimum,
        "{MAX_BATCH_TOKENS_SELECTOR}={tokens} must be >= resolved minimum {minimum}"
    );
    Ok(tokens)
}

fn resolve_prefill_budget_with_override(
    args: &cli::ServeArgs,
    ssm_prefill_chunk: usize,
    override_value: Option<&str>,
) -> Result<PrefillBudget> {
    let wide_env = std::env::var_os(QWEN4_WIDE_SELECTOR);
    let wide_override = selector_text(QWEN4_WIDE_SELECTOR, wide_env.as_deref())?;
    resolve_prefill_budget_with_explicit_overrides(
        args,
        ssm_prefill_chunk,
        override_value,
        wide_override,
    )
}

fn resolve_prefill_budget_with_explicit_overrides(
    args: &cli::ServeArgs,
    ssm_prefill_chunk: usize,
    override_value: Option<&str>,
    wide_override: Option<&str>,
) -> Result<PrefillBudget> {
    let spec_tokens = super::speculative_draft_count(args).map_or(1, |drafts| drafts + 2);
    let explicit_wide = match wide_override {
        Some("2048") => Some(2048),
        Some("4096") => Some(4096),
        Some("8192") => Some(8192),
        Some("16384") => Some(16384),
        Some("32768") => Some(32768),
        _ => None,
    };
    let user_set_prefill = args.max_prefill_tokens != 8192;
    let prefill_budget_pre_hss = if let Some(wide) = explicit_wide {
        if !user_set_prefill || args.max_prefill_tokens == wide {
            wide
        } else if args.max_prefill_tokens > 0 {
            args.max_prefill_tokens
        } else {
            args.max_seq_len
        }
    } else if user_set_prefill && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15: when prefix caching + SSM snapshots are both enabled, the
    // SSM snapshot taken at finalize_last (token_count = full prompt length)
    // is unreachable from future requests because the radix-tree match for
    // a different prompt length will be < tokens.len(). Intermediate
    // checkpoints are saved by `prefill_b_save_checkpoint` only at chunk
    // ends whose end_block is a multiple of `ssm_checkpoint_interval`, so a
    // single-chunk prefill produces zero reachable snapshots. Auto-clamp
    // the prefill budget to a checkpoint-aligned size so chunked prefill
    // actually fires and downstream agentic conversations get cache hits.
    // When ATLAS_QWEN4_PREFILL_WIDE is explicitly set (and not overridden by
    // a different max_prefill_tokens), honor the fast wide prefill geometry
    // instead of auto-clamping to ssm_checkpoint_interval * block_size.
    let prefill_budget_pre_hss = if !user_set_prefill
        && explicit_wide.is_none()
        && args.enable_prefix_caching
        && args.ssm_checkpoint_interval > 0
        && args.ssm_cache_slots > 0
    {
        let target = args.ssm_checkpoint_interval * args.block_size;
        if prefill_budget_pre_hss > target && target > 0 {
            tracing::info!(
                "--enable-prefix-caching with --ssm-checkpoint-interval={} \
                 and --block-size={}: auto-clamping max_prefill_tokens \
                 from {} to {} so chunked prefill fires at SSM-checkpoint \
                 boundaries (issue #15). Override with --max-prefill-tokens \
                 to keep the larger value.",
                args.ssm_checkpoint_interval,
                args.block_size,
                prefill_budget_pre_hss,
                target,
            );
            target
        } else {
            prefill_budget_pre_hss
        }
    } else {
        prefill_budget_pre_hss
    };
    let prefill_budget = if args.high_speed_swap {
        let hss_cap_tokens = args.high_speed_swap_cache_blocks_per_seq as usize * args.block_size;
        let hss_chunk_max = hss_cap_tokens.saturating_sub(args.max_batch_size);
        let clamped = prefill_budget_pre_hss.min(hss_chunk_max);
        if clamped < prefill_budget_pre_hss {
            tracing::info!(
                "--high-speed-swap: clamping max_prefill_tokens from {} to {} \
                 (cap {} × bs {} − max_batch_size {}) to keep chunked prefill \
                 within the rolling HBM window",
                prefill_budget_pre_hss,
                clamped,
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                args.max_batch_size,
            );
        }
        // Prefill slides advance per-layer offload cursors during both
        // `block_mgmt::ensure_blocks_through_{prefill,decode}` and post-prefix-cache-hit
        // continuation (`prefill_b/prefix_lookup.rs`), so long prompts with HSS stay valid.
        if args.max_seq_len > hss_cap_tokens {
            tracing::info!(
                "--high-speed-swap engaged: cap={} blocks × bs={} = {} tokens HBM-resident, \
                 --max-seq-len={} tokens total. Prefill slides will advance per-layer offload \
                 cursors as the window moves so older blocks stay reachable on disk.",
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                hss_cap_tokens,
                args.max_seq_len,
            );
        }
        clamped
    } else {
        prefill_budget_pre_hss
    };
    // Default: max_batch_tokens = prefill_budget + max_batch_size (decode slots).
    // ATLAS_MAX_BATCH_TOKENS env var override allows engaging the batched
    // kernel-dispatch path which requires `arena_cap >= N_streams × chunk_len`.
    // Set to (e.g.) 16384 with max_batch_size=4 to fit 4 stacked 4K chunks.
    // Memory cost: arena buffers scale ~linearly with max_batch_tokens —
    // 8× value → ~8× arena footprint (~5GB for Qwen3-Next-80B). Use sparingly.
    // Explicit overrides must be ASCII decimal, fit usize, cover the resolved
    // minimum and satisfy the capacity bound below; invalid values cannot
    // silently select default geometry.
    let qwen4_batched_verify_rows = if (args.speculative || args.dflash)
        && args.max_batch_size > 1
        && spark_model::layers::qwen4_verify_lanes::batched_k5_selected().unwrap_or(false)
    {
        args.max_batch_size
            .saturating_mul(spec_tokens.saturating_sub(1).max(2))
            .min(spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS)
    } else {
        0
    };
    let default_max_batch_tokens = resolve_default_max_batch_tokens(
        prefill_budget,
        args.max_batch_size,
        spec_tokens,
        qwen4_batched_verify_rows,
    )?;
    let max_batch_tokens =
        parse_max_batch_tokens_override(override_value, default_max_batch_tokens)?;
    // A numeric usize alone is not valid arena geometry. Keep physical/speculative
    // floors for tiny contexts while bounding overrides by total sequence capacity.
    let max_batch_tokens_capacity = args
        .max_seq_len
        .checked_mul(args.max_batch_size)
        .context("--max-seq-len * --max-batch-size exceeds usize capacity")?
        .max(default_max_batch_tokens);
    anyhow::ensure!(
        max_batch_tokens <= max_batch_tokens_capacity,
        "{MAX_BATCH_TOKENS_SELECTOR}={max_batch_tokens} exceeds resolved capacity \
         {max_batch_tokens_capacity} from sequence capacity and the default minimum"
    );
    if override_value.is_some() {
        tracing::info!(
            "ATLAS_MAX_BATCH_TOKENS override: {} (default would be {})",
            max_batch_tokens,
            default_max_batch_tokens
        );
    }
    tracing::info!(
        "Prefill config: ssm_prefill_chunk={}, args.max_prefill_tokens={}, prefill_budget={}, max_batch_tokens={}",
        ssm_prefill_chunk,
        args.max_prefill_tokens,
        prefill_budget,
        max_batch_tokens,
    );
    if args.max_prefill_tokens == 0 && args.max_seq_len > 32768 {
        tracing::warn!(
            "--max-prefill-tokens=0 with --max-seq-len={} disables chunked prefill. \
             Long agentic sessions may eventually fail with 'CUDA kernel launch failed (status 1)' \
             when an unchunked prefill exceeds device launch grid limits. \
             Consider --max-prefill-tokens=8192 (default) for sessions that grow past 32K tokens.",
            args.max_seq_len,
        );
    }
    Ok(PrefillBudget {
        prefill_budget,
        max_batch_tokens,
    })
}

fn resolve_default_max_batch_tokens(
    prefill_budget: usize,
    max_batch_size: usize,
    spec_tokens: usize,
    qwen4_batched_verify_rows: usize,
) -> Result<usize> {
    // Provision decode headroom for single-forward mixed prefill+verify rounds
    // (`P + D` up to `prefill_budget + QWEN4_CROSS_SEQ_MAX_ROWS`) when Qwen4
    // batched verify is enabled, without shrinking below `max_batch_size`.
    let decode_headroom = if qwen4_batched_verify_rows > 0 {
        let capped_verify =
            qwen4_batched_verify_rows.min(spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS);
        max_batch_size.max(capped_verify)
    } else {
        max_batch_size
    };
    Ok(prefill_budget
        .checked_add(decode_headroom)
        .context("max batch token count overflow")?
        .max(spec_tokens)
        .max(qwen4_batched_verify_rows)
        // Decode kernels pad non-rung concurrency to the next shared ladder
        // width. The arena must cover that physical width even when a tiny
        // prefill budget plus the logical batch would otherwise be smaller.
        .max(spark_runtime::buffers::padded_decode_rows(max_batch_size)))
}

#[cfg(test)]
mod qwen4_prefill_geometry_tests {
    use super::{
        validate_qwen4_prefill_geometry_os_values, validate_qwen4_prefill_geometry_values,
    };
    use std::ffi::OsStr;

    #[test]
    fn every_supported_wide_width_requires_its_exact_scheduler_budget() {
        for (value, width) in [
            ("2048", 2048),
            ("4096", 4096),
            ("8192", 8192),
            ("16384", 16384),
            ("32768", 32768),
        ] {
            validate_qwen4_prefill_geometry_values(true, Some(value), Some("1"), width).unwrap();
            assert!(
                validate_qwen4_prefill_geometry_values(true, Some(value), Some("1"), width + 1)
                    .is_err()
            );
        }
    }

    #[test]
    fn malformed_wide_values_do_not_select_the_disabled_path() {
        for value in [
            "",
            "1",
            "1024",
            "65536",
            "+4096",
            "04096",
            " 4096",
            "4096 ",
            "true",
            "\u{ff14}096",
        ] {
            let error = validate_qwen4_prefill_geometry_values(true, Some(value), Some("1"), 4096)
                .expect_err("unsupported WIDE must fail");
            assert!(error.to_string().contains("ATLAS_QWEN4_PREFILL_WIDE"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn scoped_non_unicode_selectors_fail_but_disabled_and_unrelated_modes_stay_inert() {
        use std::os::unix::ffi::OsStrExt;
        let invalid = OsStr::from_bytes(b"\xff");
        assert!(
            validate_qwen4_prefill_geometry_os_values(
                true,
                Some(invalid),
                Some(OsStr::new("1")),
                4096,
            )
            .is_err()
        );
        assert!(
            validate_qwen4_prefill_geometry_os_values(
                true,
                Some(OsStr::new("4096")),
                Some(invalid),
                4096,
            )
            .is_err()
        );
        for disabled in [None, Some(OsStr::new("0"))] {
            validate_qwen4_prefill_geometry_os_values(true, disabled, Some(invalid), 4096).unwrap();
        }
        validate_qwen4_prefill_geometry_os_values(false, Some(invalid), Some(invalid), 7).unwrap();
    }

    #[test]
    fn wide_geometry_must_equal_the_resolved_scheduler_chunk() {
        validate_qwen4_prefill_geometry_values(true, Some("8192"), Some("1"), 8192)
            .expect("matched geometry must pass");
        let error = validate_qwen4_prefill_geometry_values(true, Some("16384"), Some("1"), 8192)
            .expect_err("16K geometry with an 8K scheduler chunk must fail");
        let message = error.to_string();
        assert!(message.contains("ATLAS_QWEN4_PREFILL_WIDE=16384"));
        assert!(message.contains("resolved to 8192"));
    }

    #[test]
    fn wide_geometry_requires_exact_multi_chunk_selection() {
        for multi_chunk in [None, Some("0"), Some("true")] {
            let error =
                validate_qwen4_prefill_geometry_values(true, Some("8192"), multi_chunk, 8192)
                    .expect_err("wide geometry without exact multi-chunk selection must fail");
            assert!(
                error
                    .to_string()
                    .contains("ATLAS_QWEN4_PREFILL_MULTI_CHUNK=1")
            );
        }
    }

    #[test]
    fn default_off_and_unrelated_models_are_unchanged() {
        validate_qwen4_prefill_geometry_values(true, None, None, 8192).unwrap();
        validate_qwen4_prefill_geometry_values(true, Some("0"), None, 8192).unwrap();
        validate_qwen4_prefill_geometry_values(false, Some("not-a-width"), None, 7).unwrap();
    }
}

#[cfg(test)]
mod prefill_budget_tests {
    use super::{
        MAX_BATCH_TOKENS_SELECTOR, parse_max_batch_tokens_override,
        resolve_prefill_budget_with_explicit_overrides, resolve_prefill_budget_with_override,
        selector_text,
    };
    use clap::Parser as _;

    fn args() -> crate::cli::ServeArgs {
        crate::cli::ServeArgs::parse_from([
            "spark",
            "org/model",
            "--max-prefill-tokens",
            "4096",
            "--max-seq-len",
            "4096",
            "--max-batch-size",
            "128",
        ])
    }

    #[test]
    fn preflight_and_runtime_share_additive_c128_arena_geometry() {
        let resolved = resolve_prefill_budget_with_override(&args(), 0, None).unwrap();
        assert_eq!(resolved.prefill_budget, 4096);
        assert_eq!(resolved.max_batch_tokens, 4224);

        let widened = resolve_prefill_budget_with_override(&args(), 0, Some("8192")).unwrap();
        assert_eq!(widened.max_batch_tokens, 8192);

        for invalid in ["0", "1", "4223", "malformed"] {
            assert!(
                resolve_prefill_budget_with_override(&args(), 0, Some(invalid)).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn prefill_budget_honors_explicit_wide_when_max_prefill_tokens_omitted() {
        let mut args = crate::cli::ServeArgs::parse_from(["spark", "org/model"]);
        args.enable_prefix_caching = true;
        args.ssm_checkpoint_interval = 64;
        args.ssm_cache_slots = 64;
        args.block_size = 16;
        // max_prefill_tokens is default (8192); prefix caching auto-clamp target is 1024
        // explicit wide 4096 must be honored:
        let resolved_4k =
            resolve_prefill_budget_with_explicit_overrides(&args, 0, None, Some("4096")).unwrap();
        assert_eq!(resolved_4k.prefill_budget, 4096);

        // explicit wide 16384 must be honored:
        let resolved_16k =
            resolve_prefill_budget_with_explicit_overrides(&args, 0, None, Some("16384")).unwrap();
        assert_eq!(resolved_16k.prefill_budget, 16384);

        // explicit wide 8192 must be honored:
        let resolved_8k =
            resolve_prefill_budget_with_explicit_overrides(&args, 0, None, Some("8192")).unwrap();
        assert_eq!(resolved_8k.prefill_budget, 8192);

        // when wide is None, auto-clamps to target 1024:
        let resolved_default =
            resolve_prefill_budget_with_explicit_overrides(&args, 0, None, None).unwrap();
        assert_eq!(resolved_default.prefill_budget, 1024);
    }

    #[test]
    fn batch_override_preserves_unset_equal_and_larger_decimal_values() {
        assert_eq!(parse_max_batch_tokens_override(None, 4224).unwrap(), 4224);
        for (value, expected) in [("4224", 4224), ("8192", 8192), ("00008192", 8192)] {
            assert_eq!(
                parse_max_batch_tokens_override(Some(value), 4224).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn batch_override_rejects_non_decimal_spellings() {
        for value in [
            "",
            " ",
            "+8192",
            "-8192",
            " 8192",
            "8192 ",
            "8192\n",
            "8_192",
            "8e3",
            "0x2000",
            "\u{ff18}192",
        ] {
            let error = parse_max_batch_tokens_override(Some(value), 4224).unwrap_err();
            assert!(
                error.to_string().contains("ASCII decimal digits"),
                "{value:?}"
            );
        }
    }

    #[test]
    fn batch_override_rejects_values_below_the_resolved_minimum() {
        for value in ["0", "1", "4096", "4223"] {
            let error = parse_max_batch_tokens_override(Some(value), 4224).unwrap_err();
            let message = error.to_string();
            assert!(message.contains("ATLAS_MAX_BATCH_TOKENS="));
            assert!(message.contains("resolved minimum 4224"));
        }
    }

    #[test]
    fn batch_override_checks_usize_overflow_without_an_arbitrary_upper_cap() {
        let maximum = usize::MAX.to_string();
        assert_eq!(
            parse_max_batch_tokens_override(Some(&maximum), 4224).unwrap(),
            usize::MAX
        );
        let overflow = format!("{maximum}0");
        let error = parse_max_batch_tokens_override(Some(&overflow), 4224).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ATLAS_MAX_BATCH_TOKENS exceeds usize capacity")
        );
    }

    #[test]
    fn batch_override_accepts_exact_sequence_capacity_but_not_larger_values() {
        let args = args();
        let capacity = args.max_seq_len * args.max_batch_size;
        let resolved =
            resolve_prefill_budget_with_override(&args, 0, Some(&capacity.to_string())).unwrap();
        assert_eq!(resolved.max_batch_tokens, capacity);
        for oversized in [(capacity + 1).to_string(), usize::MAX.to_string()] {
            let error = resolve_prefill_budget_with_override(&args, 0, Some(&oversized))
                .err()
                .expect("oversized numeric overrides must be rejected");
            assert!(error.to_string().contains("exceeds resolved capacity"));
        }
    }

    #[test]
    fn batch_capacity_multiplication_overflow_fails_without_an_override() {
        let mut args = args();
        args.max_seq_len = usize::MAX;
        args.max_batch_size = 2;
        for override_value in [None, Some("8192")] {
            let error = resolve_prefill_budget_with_override(&args, 0, override_value)
                .err()
                .expect("overflowing sequence capacity must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("--max-seq-len * --max-batch-size exceeds usize capacity")
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn batch_override_non_unicode_is_not_treated_as_absent() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let error =
            selector_text(MAX_BATCH_TOKENS_SELECTOR, Some(OsStr::from_bytes(b"\xff"))).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ATLAS_MAX_BATCH_TOKENS must be valid Unicode")
        );
        assert_eq!(
            selector_text(MAX_BATCH_TOKENS_SELECTOR, None).unwrap(),
            None
        );
    }

    #[test]
    fn high_speed_swap_clamp_is_included_in_the_shared_geometry() {
        let mut args = args();
        args.high_speed_swap = true;
        args.high_speed_swap_cache_blocks_per_seq = 128;
        args.block_size = 16;
        let resolved = resolve_prefill_budget_with_override(&args, 0, None).unwrap();
        assert_eq!(resolved.prefill_budget, 1920);
        assert_eq!(resolved.max_batch_tokens, 2048);
        assert!(resolve_prefill_budget_with_override(&args, 0, Some("2047")).is_err());
        assert_eq!(
            resolve_prefill_budget_with_override(&args, 0, Some("2048"))
                .unwrap()
                .max_batch_tokens,
            2048
        );
    }

    #[test]
    fn prefix_checkpoint_clamp_remains_part_of_the_override_minimum() {
        let mut args = args();
        args.max_prefill_tokens = 8192;
        args.enable_prefix_caching = true;
        args.ssm_checkpoint_interval = 16;
        args.ssm_cache_slots = 1;
        args.block_size = 16;
        let resolved = resolve_prefill_budget_with_override(&args, 8192, None).unwrap();
        assert_eq!(resolved.prefill_budget, 256);
        assert_eq!(resolved.max_batch_tokens, 384);
        assert!(resolve_prefill_budget_with_override(&args, 8192, Some("383")).is_err());
        assert_eq!(
            resolve_prefill_budget_with_override(&args, 8192, Some("384"))
                .unwrap()
                .max_batch_tokens,
            384
        );
    }

    #[test]
    fn speculative_rows_remain_part_of_the_override_minimum() {
        let mut args = args();
        args.max_prefill_tokens = 1;
        args.max_seq_len = 1;
        args.max_batch_size = 1;
        args.speculative = true;
        args.num_drafts = 4;
        let resolved = resolve_prefill_budget_with_override(&args, 0, None).unwrap();
        assert_eq!(resolved.prefill_budget, 1);
        assert_eq!(resolved.max_batch_tokens, 6);
        assert!(resolve_prefill_budget_with_override(&args, 0, Some("5")).is_err());
        assert_eq!(
            resolve_prefill_budget_with_override(&args, 0, Some("6"))
                .unwrap()
                .max_batch_tokens,
            6
        );
        assert!(resolve_prefill_budget_with_override(&args, 0, Some("7")).is_err());
    }

    #[test]
    fn non_rung_decode_widths_reserve_their_physical_padding() {
        for (concurrency, padded) in [(33, 48), (65, 96), (97, 128)] {
            let mut args = args();
            args.max_prefill_tokens = 0;
            args.max_seq_len = 1;
            args.max_batch_size = concurrency;
            let resolved = resolve_prefill_budget_with_override(&args, 0, None).unwrap();
            assert_eq!(resolved.prefill_budget, 1);
            assert_eq!(resolved.max_batch_tokens, padded, "C{concurrency}");
            assert_eq!(
                resolve_prefill_budget_with_override(&args, 0, Some(&padded.to_string()))
                    .unwrap()
                    .max_batch_tokens,
                padded,
                "C{concurrency}"
            );
            assert!(
                resolve_prefill_budget_with_override(&args, 0, Some(&(padded + 1).to_string()))
                    .is_err(),
                "C{concurrency}"
            );
            assert!(
                resolve_prefill_budget_with_override(&args, 0, Some(&(padded - 1).to_string()))
                    .is_err(),
                "C{concurrency}"
            );
        }
    }

    #[test]
    fn qwen4_batched_verify_provisions_mixed_prefill_plus_decode_headroom() {
        // Batched verify disabled: unchanged prefill_budget + max_batch_size.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 8, 6, 0).unwrap(),
            4104
        );
        // C=8, K=5 (40 verify rows): 4096 + 40 = 4136.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 8, 6, 40).unwrap(),
            4136
        );
        // C=16, DFlash2 gamma=7 (spec_tokens=9, 16 * 8 = 128 verify rows): 4096 + 128 = 4224.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 16, 9, 128).unwrap(),
            4224
        );
        let cap = spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS;
        // C=32, DFlash2 gamma=7 (at QWEN4_CROSS_SEQ_MAX_ROWS = 256): 4096 + 256 = 4352.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 32, 9, cap).unwrap(),
            4096 + cap
        );
        // Verify rows above QWEN4_CROSS_SEQ_MAX_ROWS stay capped in headroom.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 32, 6, cap + 1).unwrap(),
            4096 + cap
        );
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, 32, 6, cap * 2).unwrap(),
            4096 + cap
        );
        // Large non-Qwen4 batch sizes above cap never shrink below max_batch_size.
        assert_eq!(
            super::resolve_default_max_batch_tokens(4096, cap + 64, 6, cap).unwrap(),
            4096 + cap + 64
        );
    }
}

pub(crate) struct KvCacheConfig {
    pub(crate) effective_kv_dtype_str: String,
    pub(crate) kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    pub(crate) layer_dtypes: Vec<spark_runtime::kv_cache::KvCacheDtype>,
    pub(crate) hss_cache_blocks_per_seq: Option<u32>,
}

pub(crate) fn resolve_kv_cache_config(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    behavior_default_kv_dtype: &str,
) -> Result<KvCacheConfig> {
    // Resolution rules:
    //   1. No MODEL.toml override        → use args.kv_cache_dtype as-is.
    //   2. User matches MODEL.toml       → silent (correct config).
    //   3. User at CLI default ("fp8")   → apply MODEL.toml override + info log.
    //   4. User explicitly mismatches    → respect user, warn loudly.
    // Rule 4 catches the gemma/mistral collapse (NVFP4 KV → `<unused>` /
    // `后汉书` token loop) and the FP8 KV mismatch on bf16-required attention
    // paths. We respect the user's choice so experimentation isn't blocked,
    // but the warning makes the cause traceable when decode goes degenerate.
    let effective_kv_dtype_str: String = if behavior_default_kv_dtype.is_empty()
        || args.kv_cache_dtype == behavior_default_kv_dtype
    {
        args.kv_cache_dtype.clone()
    } else if args.kv_cache_dtype == "fp8" {
        tracing::info!(
            "KV cache dtype: {} (from MODEL.toml default_kv_dtype, override with --kv-cache-dtype)",
            behavior_default_kv_dtype,
        );
        behavior_default_kv_dtype.to_string()
    } else {
        tracing::warn!(
            "KV cache dtype: {} (user override). MODEL.toml recommends '{}' for this \
             model — mismatched KV dtype is a known cause of decode-path corruption \
             (e.g. gemma `<unused>` collapse, mistral character-token loops on NVFP4 KV). \
             Pass --kv-cache-dtype {} to use the recommended value.",
            args.kv_cache_dtype,
            behavior_default_kv_dtype,
            behavior_default_kv_dtype,
        );
        args.kv_cache_dtype.clone()
    };
    let kv_dtype: spark_runtime::kv_cache::KvCacheDtype = effective_kv_dtype_str.parse()?;
    if kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Fp8 {
        if config.fp8_kv_calibration_tokens > 0 {
            tracing::info!(
                "FP8 KV cache with online calibration: tracking max |K|/|V| during \
                 first {} tokens to compute per-tensor scales.{}",
                config.fp8_kv_calibration_tokens,
                if args.fp8_kv_calibration_tokens == 0 {
                    " (auto-enabled from MODEL.toml)"
                } else {
                    ""
                },
            );
        } else {
            tracing::warn!(
                "FP8 KV cache selected. This requires calibrated k_scale/v_scale in the model \
                 checkpoint. Without scales (default=1.0), BF16 values are silently clipped to \
                 E4M3 range [-448, 448], destroying dynamic range. Use --fp8-kv-calibration-tokens 256 \
                 for online calibration, or --kv-cache-dtype nvfp4/bf16 if your model lacks k/v scales."
            );
        }
    }
    let num_attn_layers = config.num_attention_layers();
    let kv_hp_layers: usize = match args.kv_high_precision_layers.to_lowercase().as_str() {
        "max" | "all" => num_attn_layers,
        "auto" => 2,
        s => s.parse().unwrap_or_else(|_| {
            tracing::warn!("Invalid --kv-high-precision-layers '{}', using 0", s);
            0
        }),
    };
    let kv_hp_layers = if kv_hp_layers == 0
        && matches!(
            kv_dtype,
            spark_runtime::kv_cache::KvCacheDtype::Turbo3
                | spark_runtime::kv_cache::KvCacheDtype::Turbo4
                | spark_runtime::kv_cache::KvCacheDtype::Turbo8
                | spark_runtime::kv_cache::KvCacheDtype::Turbo4KTurbo3V
                | spark_runtime::kv_cache::KvCacheDtype::Turbo4KTurbo8V
                | spark_runtime::kv_cache::KvCacheDtype::Turbo3KTurbo8V
                | spark_runtime::kv_cache::KvCacheDtype::Bf16KTurbo4V
                | spark_runtime::kv_cache::KvCacheDtype::Bf16KTurbo3V
                | spark_runtime::kv_cache::KvCacheDtype::Fp8KTurbo4V
                | spark_runtime::kv_cache::KvCacheDtype::Fp8KTurbo3V
                | spark_runtime::kv_cache::KvCacheDtype::Bf16KTurbo2V
                | spark_runtime::kv_cache::KvCacheDtype::Fp8KTurbo2V
        ) {
        let auto_hp = ((num_attn_layers as f32 / 3.0).ceil() as usize).max(2);
        tracing::info!(
            "Auto-enabling --kv-high-precision-layers {} for {} ({}/{} attn layers BF16; \
             scaled with attn-layer count to keep accumulated turbo quant error tractable)",
            auto_hp,
            effective_kv_dtype_str,
            (auto_hp * 2).min(num_attn_layers),
            num_attn_layers,
        );
        auto_hp
    } else {
        kv_hp_layers
    };
    // A MEASURED sensitivity set (--kv-high-precision-layer-set) overrides the positional
    // first-N/last-N heuristic: spend the same BF16 budget on the attention layers a
    // per-layer sweep found most sensitive to KV quantization.
    let hp_layer_set: Vec<usize> = args
        .kv_high_precision_layer_set
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    if kv_hp_layers == 0
        && hp_layer_set.is_empty()
        && kv_dtype != spark_runtime::kv_cache::KvCacheDtype::Bf16
    {
        if kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Nvfp4
            && spark_model::layers::bitfp4_kv_enabled()
        {
            tracing::info!(
                "Uniform NVFP4 KV cache (--kv-high-precision-layers 0) with ATLAS_BITFP4_KV=1: \
                 all {} attention layers use native Blackwell BitDecoding-FP4{}.",
                num_attn_layers,
                if spark_model::layers::bitfp4_razer_kv_enabled() {
                    " + SWAR RaZeR-KV"
                } else {
                    ""
                },
            );
        } else {
            tracing::warn!(
                "⚠ --kv-high-precision-layers is 0: all KV cache layers use {} precision. \
                 Vanilla NVFP4 KV may hallucinate or lose coherence at long context. \
                 Consider ATLAS_BITFP4_KV=1 (SM120/SM121 native BitDecoding-FP4) or \
                 --kv-high-precision-layers 2-5 / --kv-high-precision-layer-set for better quality.",
                effective_kv_dtype_str,
            );
        }
    }
    let layer_dtypes = if !hp_layer_set.is_empty() {
        tracing::info!(
            "Using MEASURED --kv-high-precision-layer-set {:?} (overrides positional \
             --kv-high-precision-layers {})",
            hp_layer_set,
            kv_hp_layers,
        );
        crate::main_modules::build_layer_kv_dtypes_from_set(
            kv_dtype,
            num_attn_layers,
            &hp_layer_set,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
        )
    } else {
        crate::main_modules::build_layer_kv_dtypes(
            kv_dtype,
            num_attn_layers,
            kv_hp_layers,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
        )
    };
    let hss_cache_blocks_per_seq = if args.high_speed_swap {
        Some(args.high_speed_swap_cache_blocks_per_seq)
    } else {
        None
    };
    Ok(KvCacheConfig {
        effective_kv_dtype_str,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    })
}
