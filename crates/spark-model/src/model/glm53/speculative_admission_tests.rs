// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const RECORD: &str = include_str!("../../../../../bench/glm53-spec/evidence.json");

#[test]
fn compiled_evidence_matches_the_recorded_json_field_for_field() {
    let json: serde_json::Value = serde_json::from_str(RECORD).unwrap();
    let object = json.as_object().unwrap();
    let e = &GLM53_SPECULATIVE_EVIDENCE;
    let expected = serde_json::json!({
        "schema": e.schema,
        "engine": e.engine,
        "scope": e.scope,
        "identity_binary_sha256": e.identity_binary_sha256,
        "state_binary_sha256": e.state_binary_sha256,
        "identity_prompts": e.identity_prompts,
        "identity_trials_equal": e.identity_trials_equal,
        "identity_trials_differ": e.identity_trials_differ,
        "perturbed_prompt_control_differs": e.perturbed_prompt_control_differs,
        "state_prefix_positions_equal": e.state_prefix_positions_equal,
        "state_full_positions_equal": e.state_full_positions_equal,
        "state_gate_passed": e.state_gate_passed,
        "state_layers": e.state_layers,
        "control_legacy_restage_fails_state": e.control_legacy_restage_fails_state,
        "control_legacy_restage_fails_identity": e.control_legacy_restage_fails_identity,
        "control_extra_row_fails_state": e.control_extra_row_fails_state,
        "control_extra_row_fails_identity": e.control_extra_row_fails_identity,
        "kernel_harness_bit_identical": e.kernel_harness_bit_identical,
        "chunked_prefill_covered": e.chunked_prefill_covered,
        "spec_source_fnv1a64": e.spec_source_fnv1a64,
    });
    assert_eq!(
        object.len(),
        expected.as_object().unwrap().len(),
        "field set drift"
    );
    assert_eq!(&json, &expected);
}

/// A complete flux record with the coverage the gate requires (test fixture,
/// shaped like what make_evidence.py --flux writes after a passing P2 + P3),
/// naming this build's speculation sources.
fn complete_flux() -> Glm53SpeculativeEvidence {
    Glm53SpeculativeEvidence {
        spec_source_fnv1a64: compiled_spec_source_fnv1a64(),
        ..COMPLETE_FLUX_SHAPE
    }
}

const COMPLETE_FLUX_SHAPE: Glm53SpeculativeEvidence = Glm53SpeculativeEvidence {
    schema: "atlas.glm53.speculative_admission.v1",
    engine: "flux-engine",
    scope: "greedy",
    identity_binary_sha256: "1111111111111111111111111111111111111111111111111111111111111111",
    state_binary_sha256: "1111111111111111111111111111111111111111111111111111111111111111",
    identity_prompts: &["code2", "long", "prose", "short", "think"],
    identity_trials_equal: 30,
    identity_trials_differ: 0,
    perturbed_prompt_control_differs: true,
    state_prefix_positions_equal: 400,
    state_full_positions_equal: 300,
    state_gate_passed: true,
    state_layers: "kda 0/17/33 conv+recurrent; dsa 0..10 latent/pools/tail + pool-ahead validity",
    control_legacy_restage_fails_state: true,
    control_legacy_restage_fails_identity: true,
    control_extra_row_fails_state: true,
    control_extra_row_fails_identity: true,
    kernel_harness_bit_identical: true,
    chunked_prefill_covered: false,
    spec_source_fnv1a64: "0000000000000000",
};

/// No flux measurement exists yet: the compiled record must keep speculation closed.
#[test]
fn pending_flux_evidence_refuses() {
    let err = format!("{:#}", GLM53_SPECULATIVE_EVIDENCE.validate().unwrap_err());
    assert!(err.contains("output identity"), "{err}");
}

/// The apathy-atlas w4/w5 record passed every gate there, but it does not admit flux.
#[test]
fn apathy_evidence_does_not_admit_flux() {
    let apathy = Glm53SpeculativeEvidence {
        engine: "apathy-atlas",
        ..complete_flux()
    };
    let err = format!("{:#}", apathy.validate().unwrap_err());
    assert!(err.contains("not flux-engine"), "{err}");
    let historical: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../bench/glm53-spec/evidence.apathy-w4w5.json"
    ))
    .unwrap();
    assert!(
        historical.get("engine").is_none(),
        "the historical record predates the engine field"
    );
}

#[test]
fn complete_flux_evidence_admits() {
    complete_flux().validate().unwrap();
}

/// The runtime admission gate must verify the source hash itself, not rely solely on unit tests.
#[test]
fn evidence_for_other_sources_refuses_at_runtime() {
    let stale = Glm53SpeculativeEvidence {
        spec_source_fnv1a64: "0000000000000000",
        ..complete_flux()
    };
    let err = format!("{:#}", stale.validate().unwrap_err());
    assert!(
        err.contains("re-run output-identity and per-layer-state verification runs on this build"),
        "{err}"
    );
}

#[test]
fn compiled_hash_matches_the_test_recomputation_and_the_pending_record() {
    assert_eq!(
        GLM53_SPECULATIVE_EVIDENCE.spec_source_fnv1a64,
        compiled_spec_source_fnv1a64()
    );
}

/// install_dflash2 must gate only through admit(): a second unconditional
/// validate() would refuse even with the bring-up escape, so P2/P3 could never run.
#[test]
fn install_dflash2_gates_only_through_admit() {
    let src = include_str!("target_model_exl3.rs");
    let body = &src[src.find("pub fn install_dflash2").unwrap()..];
    let body = &body[..body.find("\n    }\n").unwrap()];
    assert!(body.contains("Glm53Model::admit(Glm53AdmissionScope::Speculative)?"));
    assert!(body.contains("GLM53_SPEC_SOURCES fnv1a64="));
    assert!(
        !body.contains(".validate()\n            .context"),
        "unconditional evidence validate is back"
    );
}

#[test]
fn malformed_identities_refuse() {
    let upper = "A".repeat(64);
    let upper: &'static str = Box::leak(upper.into_boxed_str());
    let cases = [
        Glm53SpeculativeEvidence {
            identity_binary_sha256: upper,
            state_binary_sha256: upper,
            ..complete_flux()
        },
        Glm53SpeculativeEvidence {
            identity_prompts: &["short", "short", "long", "prose", "code2"],
            ..complete_flux()
        },
        Glm53SpeculativeEvidence {
            identity_prompts: &["", "short", "long", "prose", "code2"],
            ..complete_flux()
        },
    ];
    for (i, e) in cases.iter().enumerate() {
        assert!(e.validate().is_err(), "malformed case {i} admitted");
    }
}

#[test]
fn manifest_excludes_this_file_and_covers_the_driver() {
    assert!(
        !SPEC_SOURCES
            .iter()
            .any(|p| p.ends_with("speculative_admission.rs"))
    );
    for must in [
        "glm53_policy_driver.rs",
        "prefix_commit.rs",
        "glm53_exl3_rowexact.cuh",
        "dflash2-local.yaml",
    ] {
        assert!(SPEC_SOURCES.iter().any(|p| p.ends_with(must)), "{must}");
    }
}

#[test]
fn evidence_from_different_binaries_refuses() {
    let mixed = Glm53SpeculativeEvidence {
        state_binary_sha256: "2222222222222222222222222222222222222222222222222222222222222222",
        ..complete_flux()
    };
    assert!(mixed.validate().is_err());
}

#[test]
fn every_failure_mode_refuses() {
    let base = complete_flux();
    let cases: [fn(&mut Glm53SpeculativeEvidence); 10] = [
        |e| e.identity_trials_differ = 1,
        |e| e.identity_trials_equal = 24,
        |e| e.identity_prompts = &["short"],
        |e| e.state_gate_passed = false,
        |e| e.state_prefix_positions_equal = 99,
        |e| e.state_full_positions_equal = 49,
        |e| e.control_legacy_restage_fails_state = false,
        |e| e.control_extra_row_fails_identity = false,
        |e| e.perturbed_prompt_control_differs = false,
        |e| e.kernel_harness_bit_identical = false,
    ];
    for (index, mutate) in cases.iter().enumerate() {
        let mut e = base.clone();
        mutate(&mut e);
        assert!(e.validate().is_err(), "mutation {index} was admitted");
    }
}

fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for &byte in *part {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[test]
fn certified_sources_are_unchanged_since_the_evidence_was_measured() {
    assert_eq!(SPEC_SOURCES.len(), SPEC_SOURCE_BYTES.len());
    let mut parts: Vec<&[u8]> = Vec::new();
    for (path, body) in SPEC_SOURCES.iter().zip(SPEC_SOURCE_BYTES) {
        parts.extend([path.as_bytes(), &b"\0"[..], *body, &b"\0"[..]]);
    }
    let hash = format!("{:016x}", fnv1a64(&parts));
    assert_eq!(
        hash, GLM53_SPECULATIVE_EVIDENCE.spec_source_fnv1a64,
        "speculation-path sources changed: re-run the gate windows and regenerate \
         bench/glm53-spec/evidence.json (make_evidence.py) before admitting"
    );
}
