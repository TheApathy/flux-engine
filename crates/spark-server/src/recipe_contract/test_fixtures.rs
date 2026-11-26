// SPDX-License-Identifier: AGPL-3.0-only

//! Shared fixtures for the recipe-contract tests.

use super::catalog::Catalog;
use super::contract::{self, Loaded};
use super::facts::{CheckpointFacts, ServeFacts};
use std::collections::BTreeMap;

pub const SHA: &str = "2cb54e1f854b60eb62268cef738252957a2af4c457e55966fa442e1406b66418";

pub const CATALOG: &str = "\
# test catalog
ATLAS_QWEN4_PREFILL_GDN_FI\tNUMERICS
ATLAS_QWEN4_PREFILL_ATTN_FI\tNUMERICS
ATLAS_QWEN4_LMHEAD_BF16\tNUMERICS
ATLAS_SSM_H_FP16\tNUMERICS
ATLAS_MOE_T_LANES\tROUTE
ATLAS_QWEN4_PREFILL_WIDE\tROUTE
ATLAS_SCHED_BURST_GATHER_MS\tRESOURCE
ATLAS_FULL_PROFILE\tOBSERVABILITY
";

/// The base catalog plus the native-dense selectors of `feat/fnd-umbrella`.
pub const NATIVE_EXTRA: &str = "\
ATLAS_QWEN4_DENSE_NATIVE\tNUMERICS
ATLAS_QWEN4_ROUTER_BF16\tNUMERICS
ATLAS_QWEN4_HC_BF16\tNUMERICS
ATLAS_QWEN4_GDN_BF16\tNUMERICS
ATLAS_QWEN4_ATTN_BF16\tNUMERICS
ATLAS_QWEN4_SHARED_BF16\tNUMERICS
ATLAS_QWEN4_PREFILL_NATIVE\tNUMERICS
";

pub const RECIPE: &str = r#"
schema = 1
id = "test-recipe"
kv_cache_dtype = "bf16"
route_policy = "flag"

[checkpoint]
model_type = "qwen4_exp"
model_id = "local/Test"
revision = "r1"
sha256 = { "config.json" = "2cb54e1f854b60eb62268cef738252957a2af4c457e55966fa442e1406b66418" }

[precision]
router = "nvfp4_requant"
hyper_connection = "nvfp4_requant"
gdn_proj = "nvfp4_requant"
attention_proj = "nvfp4_requant"
shared_expert = "nvfp4_requant"
lm_head = "nvfp4_requant"
prefill_dense = "fast_fp8"
routed_experts = "nvfp4"
gdn_state = "fp32"

[numerics_env]
ATLAS_QWEN4_PREFILL_GDN_FI = "2"

[route_env]
ATLAS_MOE_T_LANES = "1"

[sampling.defaults]
temperature = 0.0
top_p = 1.0
top_k = 0
presence_penalty = 0.0

[sampling.ranges]
temperature = [0.0, 2.0]
top_p = [0.0, 1.0]
top_k = [0, 1000]
presence_penalty = [-2.0, 2.0]

[limits]
max_seq_len = [2048, 262144]
max_prefill_tokens = [1, 16384]
max_batch_size = [1, 1]
qwen4_qsa = true

[speculation]
mode = "off"

[operational_env]
ATLAS_SCHED_BURST_GATHER_MS = { min = 0, max = 100 }
"#;

pub fn catalog() -> Catalog {
    Catalog::parse(CATALOG).expect("fixture catalog")
}

pub fn native_catalog() -> Catalog {
    Catalog::parse(&format!("{CATALOG}{NATIVE_EXTRA}")).expect("native catalog")
}

pub fn loaded_from(text: &str, catalog: &Catalog) -> Loaded {
    contract::parse(text, catalog).expect("fixture recipe parses")
}

pub fn loaded() -> Loaded {
    loaded_from(RECIPE, &catalog())
}

pub fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

pub fn good_env() -> BTreeMap<String, String> {
    env(&[
        ("ATLAS_QWEN4_PREFILL_GDN_FI", "2"),
        ("ATLAS_MOE_T_LANES", "1"),
    ])
}

pub fn serve() -> ServeFacts {
    ServeFacts {
        max_seq_len: 131_072,
        max_prefill_tokens: 16_384,
        max_batch_size: 1,
        qwen4_qsa: true,
        mtp_speculative: false,
        other_speculative: false,
        cli: default_cli(),
        cli_defaults: default_cli(),
    }
}

pub fn ckpt() -> CheckpointFacts {
    CheckpointFacts {
        model_type: Some("qwen4_exp".into()),
        digests: [("config.json".to_string(), Ok(SHA.to_string()))]
            .into_iter()
            .collect(),
    }
}

pub fn default_cli() -> BTreeMap<&'static str, String> {
    use clap::Parser;
    super::cli_values(&crate::cli::ServeArgs::parse_from(["spark", "defaults"]))
}
