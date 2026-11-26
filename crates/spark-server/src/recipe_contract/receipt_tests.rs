// SPDX-License-Identifier: AGPL-3.0-only

//! Receipt snapshot: the exact JSON for the fixture recipe and environment.

use super::contract;
use super::receipt;
use super::test_fixtures::*;
use serde_json::json;

fn effective() -> contract::Effective {
    contract::evaluate(&loaded(), &good_env(), &serve(), &ckpt(), &catalog()).expect("accepted")
}

#[test]
fn receipt_snapshot() {
    let sha = contract::sha256_hex(RECIPE.as_bytes());
    let src = |format: &str| json!({ "format": format, "source": "recipe" });
    let flag = |value: &str| json!({ "value": value, "source": "recipe" });
    let mut flags = serde_json::Map::new();
    for (k, v) in [
        ("--max-seq-len", "131072"),
        ("--max-prefill-tokens", "16384"),
        ("--max-batch-size", "1"),
        ("--kv-cache-dtype", "bf16 (checked after resolution)"),
        ("--qwen4-qsa", "true"),
        ("speculation", "off"),
    ] {
        flags.insert(k.into(), flag(v));
    }
    let defaults = default_cli();
    for f in super::facts::CLI_NUMERICS {
        flags.insert(f.to_string(), flag(&defaults[f]));
    }
    let expected = json!({
        "schema": 1,
        "recipe_id": "test-recipe",
        "recipe_sha256": sha,
        "checkpoint": {
            "model_type": "qwen4_exp",
            "model_id": "local/Test",
            "revision": "r1",
            "sha256_verified": { "config.json": SHA },
        },
        "precision": {
            "router": src("nvfp4_requant"),
            "hyper_connection": src("nvfp4_requant"),
            "gdn_proj": src("nvfp4_requant"),
            "attention_proj": src("nvfp4_requant"),
            "shared_expert": src("nvfp4_requant"),
            "lm_head": src("nvfp4_requant"),
            "prefill_dense": src("fast_fp8"),
            "routed_experts": src("nvfp4"),
            "gdn_state": src("fp32"),
        },
        "flags": flags,
        "sampling": {
            "defaults": { "temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0 },
            "thinking": null,
            "ranges": {
                "temperature": [0.0, 2.0],
                "top_p": [0.0, 1.0],
                "top_k": [0, 1000],
                "presence_penalty": [-2.0, 2.0],
            },
            "enforced": false,
        },
        "numerics_env": [
            { "name": "ATLAS_QWEN4_PREFILL_ATTN_FI", "value": null, "source": "default" },
            { "name": "ATLAS_QWEN4_PREFILL_GDN_FI", "value": "2", "source": "recipe" },
            { "name": "ATLAS_SSM_H_FP16", "value": null, "source": "default" },
        ],
        "route_env": [ { "name": "ATLAS_MOE_T_LANES", "value": "1", "source": "recipe" } ],
        "route_undeclared": [],
        "operational_env": [],
        "ignored_env_no_reader": [],
        "alias_shadowed": [],
    });
    assert_eq!(receipt::to_json(&effective()), expected);
}

#[test]
fn receipt_is_one_tagged_line_with_stable_key_order() {
    let line = receipt::line(&effective());
    assert!(
        line.starts_with("RECIPE_RECEIPT {\"schema\":1,\"recipe_id\":\"test-recipe\","),
        "{line}"
    );
    assert!(!line.contains('\n'));
    assert_eq!(line, receipt::line(&effective()), "deterministic");
    let keys: Vec<String> = receipt::to_json(&effective())
        .as_object()
        .expect("object")
        .keys()
        .cloned()
        .collect();
    assert_eq!(
        keys,
        [
            "schema",
            "recipe_id",
            "recipe_sha256",
            "checkpoint",
            "precision",
            "flags",
            "sampling",
            "numerics_env",
            "route_env",
            "route_undeclared",
            "operational_env",
            "ignored_env_no_reader",
            "alias_shadowed",
        ]
    );
}
