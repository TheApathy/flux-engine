// SPDX-License-Identifier: AGPL-3.0-only

//! Sub-modules of `main.rs`, factored out to keep the binary entry-point file ≤500 LoC.

pub(crate) mod app_state;
pub(crate) mod byte_count;
mod context_extension;
mod deepseek_spec_admission;
pub(crate) mod flashinfer_artifact;
pub(crate) mod glm53_gguf_resolver;
mod http_drain;
pub(crate) mod kv_dtypes;
pub(crate) mod mem_watch;
pub(crate) mod middleware;
pub(crate) mod model_host;
pub(crate) mod model_profile;
pub(crate) mod model_swap;
mod mtp_source;
mod qwen4_concurrency;
mod scheduler_retire;
mod scheduler_supervisor;
pub(crate) mod serve;
pub(crate) mod serve_load;
pub(crate) mod serve_phases;
mod serve_router;
mod serve_shutdown;
pub(crate) mod swap_env;

pub(crate) mod content_specials;
#[cfg(test)]
mod content_specials_tests;
#[cfg(test)]
mod tests;

pub(crate) use app_state::AppState;
pub(crate) use kv_dtypes::{build_layer_kv_dtypes, build_layer_kv_dtypes_from_set};
pub(crate) use serve::serve;
