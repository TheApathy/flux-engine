// SPDX-License-Identifier: AGPL-3.0-only

//! Live GB10 UMA, KV/prefix cache, and MoE/speculative runtime telemetry
//! endpoints (`GET /v1/memory`, `GET /v1/experts/usage`).

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;

use crate::main_modules::model_host::ModelHost;
use crate::metrics;
use crate::scheduler::snapshot::MtpModeSnap;

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct HostUmaMemory {
    pub mem_total_bytes: Option<u64>,
    pub mem_available_bytes: Option<u64>,
    pub mem_free_bytes: Option<u64>,
    pub cached_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub swap_free_bytes: Option<u64>,
}

/// Parse Linux `/proc/meminfo` text into byte-denominated UMA fields.
#[must_use]
pub fn parse_proc_meminfo(text: &str) -> HostUmaMemory {
    let mut map: BTreeMap<&str, u64> = BTreeMap::new();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let Some(val_str) = parts.next() else {
            continue;
        };
        let Ok(val) = val_str.parse::<u64>() else {
            continue;
        };
        let bytes = match parts.next() {
            Some("kB" | "KB") | None => val.saturating_mul(1024),
            Some("mB" | "MB") => val.saturating_mul(1024 * 1024),
            Some("B") => val,
            _ => val.saturating_mul(1024),
        };
        map.insert(key.trim(), bytes);
    }
    HostUmaMemory {
        mem_total_bytes: map.get("MemTotal").copied(),
        mem_available_bytes: map.get("MemAvailable").copied(),
        mem_free_bytes: map.get("MemFree").copied(),
        cached_bytes: map.get("Cached").copied(),
        swap_total_bytes: map.get("SwapTotal").copied(),
        swap_free_bytes: map.get("SwapFree").copied(),
    }
}

fn read_host_uma_memory() -> HostUmaMemory {
    match std::fs::read_to_string("/proc/meminfo") {
        Ok(raw) => parse_proc_meminfo(&raw),
        Err(_) => HostUmaMemory::default(),
    }
}

fn mtp_mode_label(mode: MtpModeSnap) -> &'static str {
    match mode {
        MtpModeSnap::Mtp => "speculative",
        MtpModeSnap::Serial => "serial",
        MtpModeSnap::Probing => "probing",
        MtpModeSnap::Off => "off",
    }
}

/// `GET /v1/memory` — structured GB10 `/proc/meminfo` UMA headroom, GPU
/// allocated/free bytes, KV cache blocks & token capacity, SSM snapshot slots,
/// and prefix-cache telemetry.
pub async fn memory_telemetry(State(host): State<Arc<ModelHost>>) -> Response {
    let uma = read_host_uma_memory();
    let gpu_free_bytes = crate::tui::data::gpu_free_bytes().map(|b| b as u64);
    let gpu_baseline_bytes = spark_runtime::gpu::baseline_free_bytes().map(|b| b as u64);
    let gpu_allocated_bytes = match (gpu_baseline_bytes, gpu_free_bytes) {
        (Some(base), Some(free)) => Some(base.saturating_sub(free)),
        _ => None,
    };

    let args = host.args();
    let block_size = args.as_ref().map_or(16usize, |a| a.block_size.max(1));
    let kv_dtype = args
        .as_ref()
        .map_or_else(|| "fp8".to_string(), |a| a.kv_cache_dtype.clone());

    let sched = crate::scheduler::snapshot::read();
    let kv_cache = match sched {
        Some(s) => {
            let used = s.kv_blocks_total.saturating_sub(s.kv_blocks_free);
            let max_tokens = (s.kv_blocks_total as u64).saturating_mul(block_size as u64);
            let used_tokens = (used as u64).saturating_mul(block_size as u64);
            let free_tokens = (s.kv_blocks_free as u64).saturating_mul(block_size as u64);
            let utilization = if s.kv_blocks_total > 0 {
                f64::from(used) / f64::from(s.kv_blocks_total)
            } else {
                0.0
            };
            serde_json::json!({
                "dtype": kv_dtype,
                "block_size": block_size,
                "blocks_total": s.kv_blocks_total,
                "blocks_used": used,
                "blocks_free": s.kv_blocks_free,
                "max_tokens": max_tokens,
                "used_tokens": used_tokens,
                "free_tokens": free_tokens,
                "utilization": utilization,
                "ssm_slots_used": s.ssm_slots_used,
                "ssm_slots_total": s.ssm_slots_total,
            })
        }
        None => serde_json::json!({
            "dtype": kv_dtype,
            "block_size": block_size,
            "blocks_total": 0,
            "blocks_used": 0,
            "blocks_free": 0,
            "max_tokens": 0,
            "used_tokens": 0,
            "free_tokens": 0,
            "utilization": 0.0,
            "ssm_slots_used": 0,
            "ssm_slots_total": args.as_ref().map_or(0, |a| a.ssm_cache_slots),
        }),
    };

    let hits = spark_runtime::prefix_cache::cache_hit_count();
    let misses = spark_runtime::prefix_cache::cache_miss_count();
    let hit_tokens = spark_runtime::prefix_cache::cache_hit_tokens_total();
    let lookups = hits.saturating_add(misses);
    let hit_rate = if lookups > 0 {
        (hits as f64) / (lookups as f64)
    } else {
        0.0
    };

    let current = host.current();
    let body = serde_json::json!({
        "object": "memory.telemetry",
        "model": current.as_ref().map(|s| s.model_name.as_str()),
        "max_seq_len": current.as_ref().map(|s| s.max_seq_len),
        "uma": uma,
        "gpu": {
            "allocated_bytes": gpu_allocated_bytes,
            "free_bytes": gpu_free_bytes,
            "baseline_free_bytes": gpu_baseline_bytes,
        },
        "kv_cache": kv_cache,
        "prefix_cache": {
            "hits": hits,
            "misses": misses,
            "lookups": lookups,
            "hit_tokens": hit_tokens,
            "hit_rate": hit_rate,
        },
    });
    Json(body).into_response()
}

/// `GET /v1/experts/usage` — MoE routing, speculative verification acceptance,
/// sampler entropy, and active scheduler telemetry.
pub async fn experts_usage(State(host): State<Arc<ModelHost>>) -> Response {
    let mut spec_by_k: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut moe_frames: BTreeMap<String, u64> = BTreeMap::new();
    for mf in prometheus::gather() {
        match mf.get_name() {
            "atlas_spec_decode_verify_total" => {
                for m in mf.get_metric() {
                    let mut k = String::new();
                    let mut outcome = String::new();
                    for l in m.get_label() {
                        match l.get_name() {
                            "k" => k = l.get_value().to_string(),
                            "outcome" => outcome = l.get_value().to_string(),
                            _ => {}
                        }
                    }
                    let count = m.get_counter().get_value() as u64;
                    let entry = spec_by_k.entry(k).or_default();
                    entry.1 = entry.1.saturating_add(count);
                    if outcome.contains("accept") {
                        entry.0 = entry.0.saturating_add(count);
                    }
                }
            }
            "atlas_qwen4_decode_frames_total" => {
                for m in mf.get_metric() {
                    let mut route = String::new();
                    let mut reason = String::new();
                    for l in m.get_label() {
                        match l.get_name() {
                            "route" => route = l.get_value().to_string(),
                            "reason" => reason = l.get_value().to_string(),
                            _ => {}
                        }
                    }
                    let key = format!("{route}:{reason}");
                    let count = m.get_counter().get_value() as u64;
                    *moe_frames.entry(key).or_default() += count;
                }
            }
            _ => {}
        }
    }

    let speculative_buckets: Vec<serde_json::Value> = spec_by_k
        .iter()
        .map(|(k, &(accepted, total))| {
            let rate = if total > 0 {
                (accepted as f64) / (total as f64)
            } else {
                0.0
            };
            serde_json::json!({
                "k": k,
                "accepted": accepted,
                "total": total,
                "acceptance_rate": rate,
            })
        })
        .collect();

    let sched = crate::scheduler::snapshot::read();
    let args = host.args();
    let current = host.current();
    let ep_size = args.as_ref().map_or(1, |a| a.ep_size);
    let tp_size = args.as_ref().map_or(1, |a| a.tp_size);

    let entropy_last = spark_runtime::sampler::last_entropy();
    let low_entropy_tokens = spark_runtime::sampler::low_entropy_token_count();
    let sampled_tokens = spark_runtime::sampler::total_sampled_token_count();

    let body = serde_json::json!({
        "object": "experts.telemetry",
        "model": current.as_ref().map(|s| s.model_name.as_str()),
        "topology": {
            "ep_size": ep_size,
            "tp_size": tp_size,
            "max_batch_size": current.as_ref().map(|s| s.max_batch_size),
        },
        "moe_routing": {
            "per_expert_histogram_enabled": false,
            "decode_frame_routes": moe_frames,
        },
        "speculative": {
            "mode": sched.map_or("off", |s| mtp_mode_label(s.mtp_mode)),
            "delivered_tok_per_s": sched.map_or(0.0, |s| f64::from(s.delivered_tps)),
            "verify_by_k": speculative_buckets,
        },
        "sampler": {
            "last_entropy_nats": entropy_last,
            "low_entropy_tokens_total": low_entropy_tokens,
            "sampled_tokens_total": sampled_tokens,
        },
        "scheduler": {
            "active_seqs": sched.map_or(0, |s| s.active_seqs),
            "prefilling_seqs": sched.map_or(0, |s| s.prefilling_seqs),
            "pending_seqs": sched.map_or(0, |s| s.pending_len),
            "steps_total": sched.map_or(0, |s| s.steps_total),
            "requests_total": metrics::REQUESTS_TOTAL.get(),
            "requests_active": metrics::REQUESTS_ACTIVE.get(),
            "prompt_tokens_total": metrics::PROMPT_TOKENS_TOTAL.get(),
            "decoded_tokens_total": metrics::DECODED_TOKENS_TOTAL.get(),
            "tool_calls_total": metrics::TOOL_CALLS_TOTAL.get(),
        },
    });
    Json(body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_meminfo_kb_lines_into_bytes() {
        let sample = "\
MemTotal:       131941376 kB
MemFree:         48201200 kB
MemAvailable:   108542976 kB
Buffers:           512000 kB
Cached:          61234560 kB
SwapTotal:              0 kB
SwapFree:               0 kB
";
        let parsed = parse_proc_meminfo(sample);
        assert_eq!(parsed.mem_total_bytes, Some(131_941_376 * 1024));
        assert_eq!(parsed.mem_available_bytes, Some(108_542_976 * 1024));
        assert_eq!(parsed.mem_free_bytes, Some(48_201_200 * 1024));
        assert_eq!(parsed.cached_bytes, Some(61_234_560 * 1024));
        assert_eq!(parsed.swap_total_bytes, Some(0));
    }
}
