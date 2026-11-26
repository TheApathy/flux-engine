// SPDX-License-Identifier: AGPL-3.0-only

//! The one `RECIPE_RECEIPT {json}` startup line.

use super::contract::Effective;
use super::env_check::Entry;
use super::schema::SamplingValues;
use serde_json::{Value, json};

pub const RECEIPT_TAG: &str = "RECIPE_RECEIPT";
pub const LEGACY_TAG: &str = "RECIPE_LEGACY_MODE";

fn entries(list: &[Entry]) -> Value {
    Value::Array(
        list.iter()
            .map(|e| json!({ "name": e.name, "value": e.value, "source": e.source.as_str() }))
            .collect(),
    )
}

fn sampling(v: &SamplingValues) -> Value {
    json!({
        "temperature": v.temperature,
        "top_p": v.top_p,
        "top_k": v.top_k,
        "presence_penalty": v.presence_penalty,
    })
}

/// Every effective value with its source. Key order is fixed
/// (`serde_json` is built with `preserve_order`), so the line is stable.
pub fn to_json(e: &Effective) -> Value {
    let r = &e.loaded.recipe;
    let precision: serde_json::Map<String, Value> = e
        .precision
        .iter()
        .map(|f| {
            (
                f.family.to_string(),
                json!({ "format": f.format, "source": "recipe" }),
            )
        })
        .collect();
    let flags: serde_json::Map<String, Value> = e
        .flags
        .iter()
        .map(|(k, v)| (k.to_string(), json!({ "value": v, "source": "recipe" })))
        .collect();
    let ranges = &r.sampling.ranges;
    json!({
        "schema": r.schema,
        "recipe_id": r.id,
        "recipe_sha256": e.loaded.sha256,
        "checkpoint": {
            "model_type": r.checkpoint.model_type,
            "model_id": r.checkpoint.model_id,
            "revision": r.checkpoint.revision,
            "sha256_verified": r.checkpoint.sha256,
        },
        "precision": precision,
        "flags": flags,
        "sampling": {
            "defaults": sampling(&r.sampling.defaults),
            "thinking": r.sampling.thinking.as_ref().map(sampling),
            "ranges": {
                "temperature": ranges.temperature,
                "top_p": ranges.top_p,
                "top_k": ranges.top_k,
                "presence_penalty": ranges.presence_penalty,
            },
            "enforced": false,
        },
        "numerics_env": entries(&e.env.numerics),
        "route_env": entries(&e.env.route),
        "route_undeclared": entries(&e.env.route_undeclared),
        "operational_env": entries(&e.env.operational),
        "ignored_env_no_reader": e.env.ignored,
        "alias_shadowed": e.env.alias_shadowed,
    })
}

pub fn line(e: &Effective) -> String {
    format!("{RECEIPT_TAG} {}", to_json(e))
}
