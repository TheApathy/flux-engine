// SPDX-License-Identifier: AGPL-3.0-only

//! Admission gate for GLM-5.3 greedy DFlash2 speculation.
//!
//! Speculation is admitted on evidence that it is bit-identical to the target
//! walk, not on a reference comparison of its own: a speculative step that
//! reproduces the admitted target path exactly inherits that path's
//! correctness. The evidence is measured (`bench/glm53-spec/make_evidence.py`
//! over the recorded windows), written to `bench/glm53-spec/evidence.json`,
//! and compiled in here; a unit test requires the two to agree field for field,
//! so the record cannot drift from the receipts or be edited in one place only.
//! `admit()` never reads files at runtime.

use anyhow::{Result, ensure};

/// Minimum coverage an evidence record must show.
const MIN_PROMPTS: usize = 5;
const MIN_IDENTITY_TRIALS: u32 = 25;
const MIN_PREFIX_POSITIONS: u32 = 100;
const MIN_FULL_POSITIONS: u32 = 50;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Glm53SpeculativeEvidence {
    pub schema: &'static str,
    /// The engine the verification runs measured. Only `flux-engine` evidence
    /// admits this binary; historical records from prior binaries or pre-port
    /// sources are refused.
    pub engine: &'static str,
    pub scope: &'static str,
    pub identity_binary_sha256: &'static str,
    pub state_binary_sha256: &'static str,
    pub identity_prompts: &'static [&'static str],
    pub identity_trials_equal: u32,
    pub identity_trials_differ: u32,
    pub perturbed_prompt_control_differs: bool,
    pub state_prefix_positions_equal: u32,
    pub state_full_positions_equal: u32,
    pub state_gate_passed: bool,
    pub state_layers: &'static str,
    pub control_legacy_restage_fails_state: bool,
    pub control_legacy_restage_fails_identity: bool,
    pub control_extra_row_fails_state: bool,
    pub control_extra_row_fails_identity: bool,
    pub kernel_harness_bit_identical: bool,
    pub chunked_prefill_covered: bool,
    /// FNV-1a 64 over the speculation-path sources the evidence was measured
    /// with ([`SPEC_SOURCES`]); a unit test fails when they change without
    /// the evidence being regenerated.
    pub spec_source_fnv1a64: &'static str,
}

/// Declares the certified speculation-path manifest once: `SPEC_SOURCES` (repo-
/// relative paths, also parsed by bench/glm53-spec/ts_identity.py) and the bytes
/// compiled into this binary.
macro_rules! spec_sources {
    ($($path:literal),* $(,)?) => {
        pub(crate) const SPEC_SOURCES: &[&str] = &[$($path),*];
        const SPEC_SOURCE_BYTES: &[&[u8]] = &[$(include_bytes!(concat!("../../../../../", $path))),*];
    };
}

// The GLM-5.3 speculation execution path: every non-test source this port
// changed, the row-exact/rowbatch kernels and the shipping DFlash2 recipe.
// Excluded on purpose: this file (it holds the evidence, so hashing it would be
// circular) and generic wiring that changes for unrelated reasons (cli.rs,
// scheduler/mod.rs, serve_phases/mod.rs, recipe/builtin.rs).
spec_sources!(
    "crates/spark-model/src/layers/ops/glm53_dsa_norm_projection.rs",
    "crates/spark-model/src/layers/ops/glm53_exl3.rs",
    "crates/spark-model/src/layers/ops/glm53_hyper.rs",
    "crates/spark-model/src/layers/ops/glm53_kda_decode.rs",
    "crates/spark-model/src/layers/ops/glm53_kda_prefill.rs",
    "crates/spark-model/src/model/glm53/dflash2_runtime.rs",
    "crates/spark-model/src/model/glm53/dispatch.rs",
    "crates/spark-model/src/model/glm53/dsa_attention.rs",
    "crates/spark-model/src/model/glm53/dsa_policy_execution.rs",
    "crates/spark-model/src/model/glm53/dsa_verify_execution.rs",
    "crates/spark-model/src/model/glm53/ffn_graph.rs",
    "crates/spark-model/src/model/glm53/ffn_graph_runtime.rs",
    "crates/spark-model/src/model/glm53/kda_attention.rs",
    "crates/spark-model/src/model/glm53/ordered_capture.rs",
    "crates/spark-model/src/model/glm53/prefix_commit.rs",
    "crates/spark-model/src/model/glm53/target_model.rs",
    "crates/spark-model/src/model/glm53/target_model_exl3.rs",
    "crates/spark-model/src/model/glm53/target_staged_exl3.rs",
    "crates/spark-model/src/model/glm53/verify_policy_transaction.rs",
    "crates/spark-server/src/main_modules/serve_phases/glm_dflash.rs",
    "crates/spark-server/src/recipe/builtin/glm-5.3-flash-exl3-dflash2-local.yaml",
    "crates/spark-server/src/scheduler/decode_logits_content.rs",
    "crates/spark-server/src/scheduler/glm53_policy_driver.rs",
    "crates/spark-server/src/scheduler/rollback.rs",
    "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k2_cb2.cu",
    "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k3_cb2.cu",
    "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k4_cb2.cu",
    "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_k5_cb2.cu",
    "kernels/gb10/glm5.3-flash/exl3/glm53_exl3_rowexact.cuh",
    "kernels/gb10/glm5.3-flash/iq3/glm53_dsa_norm_projection.cu",
    "kernels/gb10/glm5.3-flash/iq3/glm53_hyper.cu",
    "kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu",
);

/// FNV-1a 64 over `SPEC_SOURCES` exactly as compiled into this binary (path, NUL,
/// contents, NUL per source; the order make_evidence.py uses). Evidence only
/// admits when it names this value, so evidence measured on other sources, or
/// sources edited after measuring, cannot open the gate.
pub(crate) fn compiled_spec_source_fnv1a64() -> &'static str {
    static HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HASH.get_or_init(|| {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (path, body) in SPEC_SOURCES.iter().zip(SPEC_SOURCE_BYTES) {
            for part in [path.as_bytes(), b"\0", *body, b"\0"] {
                for &byte in part {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        format!("{hash:016x}")
    })
}

/// PENDING: no flux measurement yet, so this record fails `validate()` and the
/// Speculative scope stays closed (the operator escape is still required).
/// Regenerate it with `bench/glm53-spec/make_evidence.py --flux` from the
/// output-identity (`ts_identity.py`) and per-layer-state (`p3_state.py`)
/// verification runs on the flux binary being admitted.
pub(crate) const GLM53_SPECULATIVE_EVIDENCE: Glm53SpeculativeEvidence = Glm53SpeculativeEvidence {
    schema: "atlas.glm53.speculative_admission.v1",
    engine: "flux-engine",
    scope: "greedy",
    identity_binary_sha256: "",
    state_binary_sha256: "",
    identity_prompts: &[],
    identity_trials_equal: 0,
    identity_trials_differ: 0,
    perturbed_prompt_control_differs: false,
    state_prefix_positions_equal: 0,
    state_full_positions_equal: 0,
    state_gate_passed: false,
    state_layers: "",
    control_legacy_restage_fails_state: false,
    control_legacy_restage_fails_identity: false,
    control_extra_row_fails_state: false,
    control_extra_row_fails_identity: false,
    kernel_harness_bit_identical: false,
    chunked_prefill_covered: false,
    spec_source_fnv1a64: "91450abd80196d85",
};

/// The speculation gate controls (`bench/glm53-spec`) that deliberately break
/// the accept path. Admission refuses while either is set.
pub(crate) const SPECULATIVE_CONTROL_ENVS: [&str; 2] = [
    "ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE",
    "ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW",
];

pub(crate) fn speculative_control_active() -> bool {
    SPECULATIVE_CONTROL_ENVS
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|value| value == "1"))
}

impl Glm53SpeculativeEvidence {
    /// Admit greedy speculation only when every identity and state check held,
    /// with enough coverage, and every negative control failed as it must.
    /// Chunked prefill is recorded, not required: the EXL3 target refuses
    /// chunked prompts itself, so no path can exercise it yet.
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "atlas.glm53.speculative_admission.v1" && self.scope == "greedy",
            "GLM speculative evidence has an unknown schema or scope"
        );
        ensure!(
            self.engine == "flux-engine",
            "GLM speculative evidence was measured on {:?}, not flux-engine; regenerate it on \
             the flux binary (bench/glm53-spec/make_evidence.py --flux)",
            self.engine
        );
        ensure!(
            self.identity_prompts.len() >= MIN_PROMPTS
                && self.identity_trials_equal >= MIN_IDENTITY_TRIALS
                && self.identity_trials_differ == 0,
            "GLM speculative evidence: output identity is missing or failed"
        );
        ensure!(
            self.state_gate_passed
                && self.state_prefix_positions_equal >= MIN_PREFIX_POSITIONS
                && self.state_full_positions_equal >= MIN_FULL_POSITIONS,
            "GLM speculative evidence: per-layer state identity is missing or failed"
        );
        ensure!(
            self.perturbed_prompt_control_differs
                && self.control_legacy_restage_fails_state
                && self.control_legacy_restage_fails_identity
                && self.control_extra_row_fails_state
                && self.control_extra_row_fails_identity,
            "GLM speculative evidence: a negative control did not fail, so its gate cannot admit speculation"
        );
        ensure!(
            self.kernel_harness_bit_identical,
            "GLM speculative evidence: the row-batched kernel is not bit-identical"
        );
        ensure!(
            self.identity_binary_sha256.len() == 64 && self.state_binary_sha256.len() == 64,
            "GLM speculative evidence must name the binaries it was measured on"
        );
        let is_sha = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        ensure!(
            is_sha(self.identity_binary_sha256) && is_sha(self.state_binary_sha256),
            "GLM speculative evidence binary identities must be lowercase hex sha256"
        );
        let mut prompts: Vec<&str> = self.identity_prompts.to_vec();
        prompts.sort_unstable();
        prompts.dedup();
        ensure!(
            prompts.len() == self.identity_prompts.len() && prompts.iter().all(|p| !p.is_empty()),
            "GLM speculative evidence prompt names must be unique and non-empty"
        );
        ensure!(
            self.identity_binary_sha256 == self.state_binary_sha256,
            "GLM speculative evidence: the output-identity and per-layer-state verification runs used different binaries"
        );
        ensure!(
            self.spec_source_fnv1a64 == compiled_spec_source_fnv1a64(),
            "GLM speculative evidence certifies speculation sources {} but this binary was built \
             from {}; re-run output-identity and per-layer-state verification runs on this build \
             and regenerate the evidence",
            self.spec_source_fnv1a64,
            compiled_spec_source_fnv1a64()
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "speculative_admission_tests.rs"]
mod tests;
