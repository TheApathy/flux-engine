// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime helpers run after model build: EOS / sampling defaults from
//! generation_config.json, dump-writer open, response-store / behavior
//! audit logging, model-name resolution, tool-call parser dispatch.

#[cfg(test)]
#[path = "glm_tool_contract_registry.rs"]
mod glm_tool_contract_registry;

use std::path::Path;

use anyhow::Result;

use atlas_core::config::ModelConfig;

use crate::cli;

/// Opt-in gate (`FLUX_SIMHASH_WATCHDOG=1` / `ATLAS_SIMHASH_WATCHDOG=1`, default `false`)
/// for the legacy SSE streaming Jaccard/SimHash loop guard so direct `spark serve`
/// launches match container launches and blocking `/v1/chat/completions` parity.
pub(crate) fn simhash_watchdog_env_enabled() -> bool {
    ["FLUX_SIMHASH_WATCHDOG", "ATLAS_SIMHASH_WATCHDOG"]
        .into_iter()
        .find_map(|k| std::env::var(k).ok())
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true"))
}

pub(crate) fn load_eos_tokens(model_dir: &Path, config: &ModelConfig) -> Vec<u32> {
    let gen_config_path = model_dir.join("generation_config.json");
    if let Ok(gen_json) = std::fs::read_to_string(&gen_config_path) {
        if let Ok(gen_cfg) = serde_json::from_str::<serde_json::Value>(&gen_json) {
            return match gen_cfg.get("eos_token_id") {
                Some(serde_json::Value::Array(arr)) => {
                    let ids: Vec<u32> = arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect();
                    if !ids.is_empty() {
                        tracing::info!("EOS tokens (from generation_config.json): {:?}", ids);
                        ids
                    } else {
                        config.stop_token_ids()
                    }
                }
                Some(serde_json::Value::Number(n)) => {
                    if let Some(id) = n.as_u64().and_then(|v| u32::try_from(v).ok()) {
                        tracing::info!("EOS token (from generation_config.json): {}", id);
                        vec![id]
                    } else {
                        tracing::warn!(
                            "Ignoring invalid eos_token_id ({n}) in generation_config.json; \
                             falling back to config.json"
                        );
                        config.stop_token_ids()
                    }
                }
                _ => config.stop_token_ids(),
            };
        }
        return config.stop_token_ids();
    }
    tracing::info!(
        "EOS tokens (from config.json): {:?}",
        config.stop_token_ids()
    );
    config.stop_token_ids()
}

pub(crate) struct SamplingDefaults {
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) top_n_sigma: f32,
    pub(crate) min_p: f32,
}

pub(crate) fn load_sampling_defaults(
    model_dir: &Path,
    args: &cli::ServeArgs,
    preset: &atlas_kernels::SamplingCategory,
) -> SamplingDefaults {
    let gen_config_path = model_dir.join("generation_config.json");
    let gen_cfg = std::fs::read_to_string(&gen_config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    let defaults = resolve_sampling_defaults(gen_cfg.as_ref(), args, preset);
    tracing::info!(
        "Default sampling: temperature={}, top_k={}, top_p={}, top_n_sigma={}, min_p={}",
        defaults.temperature,
        defaults.top_k,
        defaults.top_p,
        defaults.top_n_sigma,
        defaults.min_p,
    );
    defaults
}

/// Resolve each field independently. Checkpoint generation_config.json is
/// most specific, then MODEL.toml, then the operator's CLI default. The
/// optional filters use `Option` in the model preset because absent and
/// explicit zero have opposite meanings.
pub(crate) fn resolve_sampling_defaults(
    gen_cfg: Option<&serde_json::Value>,
    args: &cli::ServeArgs,
    preset: &atlas_kernels::SamplingCategory,
) -> SamplingDefaults {
    let temperature = gen_cfg
        .and_then(|v| v.get("temperature")?.as_f64())
        .map(|t| t as f32)
        .unwrap_or(preset.temperature);
    let top_k = gen_cfg
        .and_then(|v| v.get("top_k")?.as_u64())
        .map(|k| k as u32)
        .unwrap_or(preset.top_k);
    let top_p = gen_cfg
        .and_then(|v| v.get("top_p")?.as_f64())
        .map(|p| p as f32)
        .unwrap_or(preset.top_p);
    let top_n_sigma = gen_cfg
        .and_then(|v| v.get("top_n_sigma")?.as_f64())
        .map(|s| s as f32)
        .or(preset.top_n_sigma)
        .unwrap_or(args.default_top_n_sigma);
    let min_p = gen_cfg
        .and_then(|v| v.get("min_p")?.as_f64())
        .map(|p| p as f32)
        .or(preset.min_p)
        .unwrap_or(args.default_min_p);
    SamplingDefaults {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
    }
}

pub(crate) fn open_dump_writer(args: &cli::ServeArgs) -> Option<crate::request_dumper::DumpHandle> {
    use crate::request_dumper;
    match args.dump.as_deref() {
        Some(arg) => {
            let path = request_dumper::resolve_path(arg);
            match request_dumper::DumpHandle::open(path) {
                Ok(h) => {
                    tracing::info!(
                        path = %h.path().display(),
                        "Request dump enabled (JSONL append)"
                    );
                    Some(h)
                }
                Err(e) => {
                    tracing::error!("Failed to open --dump target: {e}. Dumping is disabled.");
                    None
                }
            }
        }
        None => None,
    }
}

pub(crate) fn log_response_store_audit(
    response_store: &crate::response_store::ResponseStore,
    rate_limiter: &crate::rate_limiter::RateLimiter,
) {
    if rate_limiter.config().is_enabled() {
        let cfg = rate_limiter.config();
        tracing::info!(
            "Rate limiter active: {} req/min, {} tok/min (bursts {}/{})",
            cfg.rpm,
            cfg.tpm,
            cfg.burst_rpm,
            cfg.burst_tpm
        );
    }
    tracing::info!(
        "Response store: max_entries={}, ttl={:?}, persist={}",
        response_store.max_entries(),
        response_store.ttl(),
        match response_store.persist_dir() {
            Some(p) => format!("filesystem ({})", p.display()),
            None => "memory-only".to_string(),
        }
    );
    if response_store.is_persistent() && response_store.len() > 0 {
        tracing::info!(
            "Response store: replayed {} entries from disk",
            response_store.len()
        );
    }
}

/// Publish model-specific scheduler behavior before the scheduler thread can
/// observe requests. This must run on every initial load and swap.
pub(crate) fn publish_scheduler_behavior(behavior: &atlas_kernels::ModelBehavior) {
    crate::scheduler::set_honor_eos_inside_thinking(behavior.honor_eos_inside_thinking);
}

pub(crate) fn log_behavior_audit(args: &cli::ServeArgs, ptx_set: &atlas_kernels::TargetPtxSet) {
    if !ptx_set.behavior.thinking_in_tools {
        tracing::info!("Model behavior: thinking disabled when tools active (MODEL.toml)");
    }
    let effective_thinking_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    tracing::info!(
        "Model behavior: max_thinking_budget={}{}, thinking_default={}",
        effective_thinking_budget,
        if args.max_thinking_budget.is_some() {
            " (CLI override)"
        } else {
            ""
        },
        ptx_set.behavior.thinking_default,
    );
    let loop_watchdog_enabled =
        ptx_set.behavior.enable_loop_watchdog && !args.disable_loop_watchdog;
    crate::scheduler::set_enable_loop_watchdog(loop_watchdog_enabled);
    let catastrophic_loop_watchdog_enabled = !args.disable_loop_watchdog;
    crate::scheduler::set_enable_catastrophic_loop_watchdog(catastrophic_loop_watchdog_enabled);
    if args.disable_loop_watchdog {
        tracing::info!(
            "Model behavior: content-loop watchdogs DISABLED (CLI override, including catastrophic guard)"
        );
    } else if loop_watchdog_enabled {
        tracing::info!(
            "Model behavior: content-loop watchdog ENABLED (period-{}…{} repetition detector)",
            crate::scheduler::CONTENT_LOOP_PERIOD_MIN,
            crate::scheduler::CONTENT_LOOP_PERIOD_MAX,
        );
    } else {
        tracing::info!(
            "Model behavior: configured content-loop watchdog disabled; catastrophic guard ENABLED"
        );
    }
    let thinking_loop_watchdog_enabled =
        ptx_set.behavior.enable_loop_watchdog && !args.disable_thinking_loop_watchdog;
    crate::scheduler::set_enable_thinking_loop_watchdog(thinking_loop_watchdog_enabled);
    tracing::info!(
        "Model behavior: thinking-loop watchdog {}{}",
        if thinking_loop_watchdog_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
        if args.disable_thinking_loop_watchdog {
            " (CLI override; thinking budget remains active)"
        } else if !ptx_set.behavior.enable_loop_watchdog {
            " (MODEL.toml default off; thinking budget remains active)"
        } else {
            ""
        },
    );
    // Phase-A: per-model watchdog tunables from MODEL.toml [behavior].
    let b = &ptx_set.behavior;
    let env_max_inter_tool_prose = match std::env::var("ATLAS_MAX_INTER_TOOL_PROSE") {
        Ok(value) => match value.parse::<u32>() {
            Ok(value) => Some(value),
            Err(_) => {
                tracing::warn!(
                    value = %value,
                    "ATLAS_MAX_INTER_TOOL_PROSE is not a u32; ignoring override"
                );
                None
            }
        },
        Err(_) => None,
    };
    let default_max_inter_tool_prose = if crate::api::chat_phases::agent_guards_enabled() {
        b.max_inter_tool_prose
    } else {
        u32::MAX
    };
    let max_inter_tool_prose = crate::scheduler::resolve_max_inter_tool_prose(
        default_max_inter_tool_prose,
        env_max_inter_tool_prose,
        args.max_inter_tool_prose,
    );
    let confidence_early_stop = b.confidence_early_stop && !args.disable_confidence_early_stop;
    crate::scheduler::set_watchdog_params(crate::scheduler::WatchdogParams {
        think_loop_min_repeats: b.think_loop_min_repeats as usize,
        think_loop_scan_window: b.think_loop_scan_window as usize,
        confidence_early_stop,
        confidence_run_length: b.confidence_run_length,
        fuzzy_repeat_tolerance_div: b.fuzzy_repeat_tolerance_div as usize,
        max_inter_tool_prose,
        rollback_resteer: b.rollback_resteer,
    });
    if b.honor_eos_inside_thinking {
        tracing::info!("Model behavior: mid-thinking EOS implicit close ENABLED");
    }
    tracing::info!(
        max_inter_tool_prose,
        "Model behavior: effective inter-tool prose cap (u32::MAX = disabled)"
    );
    if !confidence_early_stop {
        tracing::info!("Model behavior: confidence early-stop DISABLED");
    }
    if args.disable_simhash_watchdog || !simhash_watchdog_env_enabled() {
        tracing::info!("Model behavior: API SimHash semantic-loop watchdog DISABLED");
    } else {
        tracing::info!(
            "Model behavior: API SimHash semantic-loop watchdog ENABLED (FLUX_SIMHASH_WATCHDOG=1)"
        );
    }
    // Watchdog rollback + re-steer (arXiv:2603.27905).
    if b.rollback_resteer {
        tracing::info!(
            "Model behavior: watchdog rollback+re-steer ENABLED (cap {} per sequence)",
            atlas_kernels::ROLLBACK_RESTEER_CAP,
        );
    } else {
        tracing::info!("Model behavior: watchdog rollback+re-steer DISABLED (legacy hard-stop)");
    }
    // ROM (arXiv:2603.22016) scaffold. A trained repetition-onset detection
    // head can be dropped in via MODEL.toml [behavior].rom_head; the runtime
    // would load the artifact and call `set_rom_head`. No trained head ships
    // with Flux, so when `rom_head` is empty (the default) the confidence
    // heuristic stays as the fallback — unchanged. Loading the artifact is
    // intentionally a future step: only the optional hook (the `RomHead` trait
    // seam) is wired now.
    if !b.rom_head.is_empty() {
        tracing::warn!(
            rom_head = b.rom_head,
            "Model behavior: [behavior].rom_head is set but ROM artifact \
             loading is not yet implemented — confidence heuristic \
             remains the active detector"
        );
    }
    // TSCG tool-schema compilation (MODEL.toml [behavior].tscg).
    crate::tscg::set_tscg_enabled(b.tscg);
    if b.tscg {
        tracing::info!("Model behavior: TSCG tool-schema compilation ENABLED (compact signatures)");
    }
    if args.disable_thinking {
        tracing::info!("--disable-thinking set: thinking is forced OFF for every request");
    }
    if let Some(threshold) = args.auto_compact {
        tracing::info!(
            "Auto-compact enabled: threshold={:.0}% of max_seq_len ({})",
            threshold * 100.0,
            args.max_seq_len
        );
    }
}

#[cfg(test)]
mod scheduler_behavior_publish_tests {
    #[test]
    fn model_behavior_is_published_before_scheduler_spawn_and_read_live() {
        let loader = include_str!("../serve_load.rs");
        let publish = loader
            .find("serve_phases::publish_scheduler_behavior(&ptx_set.behavior)")
            .expect("loader must publish model behavior");
        let spawn = loader
            .find("std::thread::spawn")
            .expect("loader must spawn scheduler");
        assert!(publish < spawn, "behavior publication must precede spawn");

        let scheduler = include_str!("../../scheduler/mod.rs");
        assert!(scheduler.contains("honor_eos_inside_thinking: None"));
    }
}

pub(crate) fn resolve_model_name(
    args: &cli::ServeArgs,
    config_json: &str,
    model_dir: &Path,
) -> String {
    args.model_name
        .clone()
        .or_else(|| args.model.clone())
        .or_else(|| {
            serde_json::from_str::<serde_json::Value>(config_json)
                .ok()
                .and_then(|v| v.get("_name_or_path")?.as_str().map(String::from))
        })
        .unwrap_or_else(|| {
            model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "atlas".to_string())
        })
}

pub(crate) fn resolve_tool_call_parser(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    config: &ModelConfig,
) -> Result<Option<std::sync::Arc<dyn crate::tool_parser::ToolCallParser>>> {
    use crate::tool_parser;
    let tool_call_format: Option<tool_parser::ToolCallFormat> =
        if let Some(ref parser) = args.tool_call_parser {
            let format: tool_parser::ToolCallFormat =
                parser.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            tracing::info!("Tool call parser: {} (user-specified)", format.name());
            Some(format)
        } else if !ptx_set.behavior.tool_call_parser.is_empty() {
            let format: tool_parser::ToolCallFormat = ptx_set
                .behavior
                .tool_call_parser
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            tracing::info!(
                "Tool call parser: {} (MODEL.toml [behavior].tool_call_parser)",
                format.name()
            );
            Some(format)
        } else {
            let defaults: toml::Table = toml::from_str(include_str!("../../../tool_defaults.toml"))
                .expect("invalid tool_defaults.toml");
            let auto_format = defaults
                .get("model_type")
                .and_then(|t| t.as_table())
                .and_then(|t| t.get(config.model_type.as_str()))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<tool_parser::ToolCallFormat>().ok());
            if let Some(format) = auto_format {
                tracing::info!(
                    "Tool call parser: {} (auto-detected from model_type '{}')",
                    format.name(),
                    config.model_type
                );
                Some(format)
            } else {
                tracing::info!(
                    "Tool call parser: disabled (no mapping for model_type '{}')",
                    config.model_type
                );
                None
            }
        };

    if let Some(format) = tool_call_format {
        if format.has_grammar() {
            tracing::info!(
                "Tool call parser: '{}' has registered XGrammar grammar — constrained decoding ENABLED for tool requests",
                format.name()
            );
        } else {
            tracing::warn!(
                "Tool call parser: '{}' has NO XGrammar grammar registered — constrained decoding DISABLED. \
                 Tool calls rely entirely on model-trained behavior; degraded quality possible.",
                format.name()
            );
        }
    }
    Ok(tool_call_format.map(|f| std::sync::Arc::from(f.into_parser())))
}

#[cfg(test)]
mod sampling_defaults_tests {
    use clap::Parser;

    use super::resolve_sampling_defaults;
    use crate::cli;

    fn preset(min_p: Option<f32>, top_n_sigma: Option<f32>) -> atlas_kernels::SamplingCategory {
        atlas_kernels::SamplingCategory {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 40,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p,
            top_n_sigma,
        }
    }

    fn args() -> cli::ServeArgs {
        // Use a nonzero operator default so `None` and `Some(0.0)` have
        // observably different outcomes even though this fork's CLI default
        // currently happens to be zero.
        cli::ServeArgs::parse_from([
            "spark",
            "org/model",
            "--default-min-p",
            "0.08",
            "--default-top-n-sigma",
            "1.25",
        ])
    }

    #[test]
    fn absent_model_min_p_preserves_cli_default() {
        let args = args();
        assert!(args.default_min_p > 0.0, "negative-control prerequisite");
        let defaults = resolve_sampling_defaults(None, &args, &preset(None, None));
        assert_eq!(defaults.min_p, args.default_min_p);
    }

    #[test]
    fn explicit_model_min_p_zero_beats_cli_default() {
        let args = args();
        assert!(args.default_min_p > 0.0, "negative-control prerequisite");
        let defaults = resolve_sampling_defaults(None, &args, &preset(Some(0.0), None));
        assert_eq!(defaults.min_p, 0.0);
    }

    #[test]
    fn checkpoint_min_p_outranks_model_preset() {
        let config = serde_json::json!({"min_p": 0.31});
        let defaults = resolve_sampling_defaults(Some(&config), &args(), &preset(Some(0.0), None));
        assert_eq!(defaults.min_p, 0.31);
    }

    #[test]
    fn absent_checkpoint_fields_fall_back_independently() {
        let config = serde_json::json!({"top_k": 64});
        let defaults = resolve_sampling_defaults(Some(&config), &args(), &preset(Some(0.0), None));
        assert_eq!(defaults.top_k, 64);
        assert_eq!(defaults.temperature, 0.7);
        assert_eq!(defaults.top_p, 0.8);
        assert_eq!(defaults.min_p, 0.0);
    }

    #[test]
    fn explicit_model_top_n_sigma_zero_beats_cli_default() {
        let args = args();
        assert!(
            args.default_top_n_sigma > 0.0,
            "negative-control prerequisite"
        );
        let defaults = resolve_sampling_defaults(None, &args, &preset(None, Some(0.0)));
        assert_eq!(defaults.top_n_sigma, 0.0);
    }

    #[test]
    fn checkpoint_top_n_sigma_outranks_model_preset() {
        let config = serde_json::json!({"top_n_sigma": 0.31});
        let defaults = resolve_sampling_defaults(Some(&config), &args(), &preset(None, Some(0.0)));
        assert_eq!(defaults.top_n_sigma, 0.31);
    }
}
