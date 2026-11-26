// SPDX-License-Identifier: AGPL-3.0-only

use std::path::Path;

use super::provenance::sha256_bytes;
use super::scheduler_authority::{RELEASE_SCHEDULER_MANIFEST_PATH, bind_manifest, parse_manifest};
use super::scheduler_build::{GATE_SOURCE_PATHS, RUNTIME_ENVIRONMENT, TOOL_NAMES};
use super::{BinaryIdentity, BundleIdentity};

const ENTRY: &str = include_str!("../qwen38_flashinfer_direct_merged_silu_microgate.rs");
const AUTHORITY: &str = include_str!("authority.rs");
const SCHEDULER: &str = include_str!("scheduler_authority.rs");
const BUILD: &str = include_str!("scheduler_build.rs");
const TRUST: &str = include_str!("scheduler_trust.rs");
const LAUNCH: &str = include_str!("launch.rs");
const PROVENANCE: &str = include_str!("provenance.rs");
const ENTRY_SHA256: &str = "8839607bde6c528863670cb77b352348f433af5b0b8f602c0ef1a1f52ca7337c";
const AUTHORITY_SHA256: &str = "c598e64aa3b7ecb83caa52e0456bd2297a9ad8fb8f94583feb206286fce806c7";
const SCHEDULER_SHA256: &str = "96ba0ba6ce0f4ff797012e8f65c2384d75cc6a9420109d04752ba0a058be3792";
const BUILD_SHA256: &str = "d5e2c4df3f474b37fae39fe4d5da972c6e89ad0a850c09486e72c8e7d71adc30";
const TRUST_SHA256: &str = "f9aad11ef09bf5966d089fc51fa05d51162cd8fdaee9dec400c635219d3154ec";
const LAUNCH_SHA256: &str = "a3fef6f59534349cad28d620210561a6f8794f9daff2f4ad3157a09432291411";
const PROVENANCE_SHA256: &str = "45221b4cab0b9a87533e2c5920486bbd8c4eadf58c379c824dc2a9e3eee4f556";

fn hex_records(records: &[&str]) -> String {
    format!(
        "hex:{}",
        records
            .join("\0")
            .bytes()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

pub(super) fn manifest(
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> String {
    let build_argv = hex_records(&[
        "/sealed/cargo",
        "build",
        "--release",
        "--locked",
        "-p",
        "spark-model",
        "--example",
        "qwen38_flashinfer_direct_merged_silu_microgate",
    ]);
    let build_environment = hex_records(&[
        "CARGO_HOME=/sealed/cargo-home",
        "CUDA_HOME=/sealed/cuda",
        "CUDA_VISIBLE_DEVICES=",
        "LC_ALL=C",
        "PATH=/sealed/bin:/usr/bin:/bin",
        "RUSTUP_HOME=/sealed/rustup-home",
        "SOURCE_DATE_EPOCH=0",
    ]);
    let mut text = format!(
        "schema=qwen38-direct-merged-silu-scheduler-v1\n\
         binary_sha256={}\n\
         bundle_sha256={}\n\
         bundle_target={}\n\
         bundle_module_count={}\n\
         session_id=session-0123456789abcdef0123456789abcdef\n\
         ticket_sha256={}\n\
         ticket_fd=3\n\
         ticket_dev=11\n\
         ticket_ino=12\n\
         receipt_fd=4\n\
         receipt_dev=21\n\
         receipt_ino=22\n\
         receipt_path=/run/atlas-qualification/session.receipt\n\
         build_receipt_path=/run/atlas-qualification/build.receipt\n\
         build_receipt_sha256={}\n\
         build_argv={}\n\
         build_environment={}\n\
         runtime_environment={}\n",
        binary.sha256,
        bundle.sha256,
        bundle.target,
        bundle.module_count,
        "ab".repeat(32),
        "bc".repeat(32),
        build_argv,
        build_environment,
        RUNTIME_ENVIRONMENT,
    );
    for (path, sha256) in sources {
        text.push_str(&format!("source:{path}={sha256}\n"));
    }
    for (name, sha256) in &bundle.direct_modules {
        text.push_str(&format!("ptx:{name}={sha256}\n"));
    }
    for path in GATE_SOURCE_PATHS {
        text.push_str(&format!("gate_source:{path}={}\n", "cd".repeat(32)));
    }
    for name in TOOL_NAMES {
        text.push_str(&format!("tool:{name}:path=/sealed/{name}\n"));
        text.push_str(&format!("tool:{name}:sha256={}\n", "de".repeat(32)));
    }
    text
}

pub(super) fn identities() -> (
    BinaryIdentity,
    BundleIdentity,
    Vec<(&'static str, &'static str)>,
) {
    (
        BinaryIdentity {
            path: "/sealed/bin".to_owned(),
            sha256: "11".repeat(32),
            profile: "release",
        },
        BundleIdentity {
            target: "sm_121/qwen3.8-27b/nvfp4".to_owned(),
            module_count: 3,
            sha256: "22".repeat(32),
            direct_modules: vec![
                ("flashinfer_projection_split".to_owned(), "33".repeat(32)),
                ("quantize_nvfp4".to_owned(), "44".repeat(32)),
                ("nvfp4_cutlass".to_owned(), "55".repeat(32)),
            ],
        },
        vec![("source-a", "66"), ("source-b", "77")],
    )
}

#[test]
fn scheduler_manifest_binding_is_exact_and_strict() {
    let (binary, bundle, sources) = identities();
    let text = manifest(&binary, &bundle, &sources);
    let fields = parse_manifest(&text).unwrap();
    assert!(bind_manifest(&fields, &binary, &bundle, &sources).is_ok());
    for mutant in [
        text.replace(&binary.sha256, &"99".repeat(32)),
        text.replace("source:source-b=77\n", ""),
        text.replace(
            &format!("gate_source:{}={}", GATE_SOURCE_PATHS[3], "cd".repeat(32)),
            "",
        ),
        text.replace("tool:nvcc:path=/sealed/nvcc\n", ""),
        text.replace(RUNTIME_ENVIRONMENT, "LC_ALL=C"),
        format!("{text}unexpected=value\n"),
        format!("{text}ticket_fd=9\n"),
    ] {
        assert!(
            parse_manifest(&mutant)
                .and_then(|fields| bind_manifest(&fields, &binary, &bundle, &sources))
                .is_err()
        );
    }
    assert!(parse_manifest(&text.replace("ticket_fd=3", "ticket fd=3")).is_err());
    assert!(parse_manifest(&text.replace("\n", "\r\n")).is_err());
}

#[test]
fn authority_has_no_embedded_self_hash_or_caller_nonce_path() {
    for (source, expected) in [
        (ENTRY, ENTRY_SHA256),
        (AUTHORITY, AUTHORITY_SHA256),
        (SCHEDULER, SCHEDULER_SHA256),
        (BUILD, BUILD_SHA256),
        (TRUST, TRUST_SHA256),
    ] {
        assert_eq!(sha256_bytes(source.as_bytes()).unwrap(), expected);
    }
    assert!(RELEASE_SCHEDULER_MANIFEST_PATH.starts_with("UNRELEASED-"));
    assert!(!Path::new(RELEASE_SCHEDULER_MANIFEST_PATH).is_absolute());
    for forbidden in [
        "RELEASE_BINARY_SHA256",
        "RELEASE_BUILD_RECEIPT_SHA256",
        "ATLAS_DIRECT_MERGED_SILU_NONCE",
        "create_new(true)",
        "/var/tmp/atlas-direct-merged-silu-",
    ] {
        assert!(
            ![ENTRY, AUTHORITY, SCHEDULER, BUILD, TRUST, PROVENANCE]
                .concat()
                .contains(forbidden)
        );
    }
    for required in [
        "root-owned immutable regular file",
        "File::from_raw_fd(ticket_fd)",
        "File::from_raw_fd(receipt_fd)",
        "scheduler ticket has trailing bytes",
        "self.verify_receipt(0)?",
        "self.verify_receipt(u64::try_from(bytes.len())?)?",
        "scheduler_session_id",
        "build_receipt_path",
        "gate_source_sha256",
        "qualification runtime environment is not sanitized",
    ] {
        assert!(
            [ENTRY, AUTHORITY, SCHEDULER, BUILD, TRUST, PROVENANCE]
                .concat()
                .contains(required)
        );
    }
}

#[test]
fn launch_and_provenance_are_complete_source_sealed() {
    assert_eq!(sha256_bytes(LAUNCH.as_bytes()).unwrap(), LAUNCH_SHA256);
    assert_eq!(
        sha256_bytes(PROVENANCE.as_bytes()).unwrap(),
        PROVENANCE_SHA256
    );
    let candidate = "ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(";
    let incumbent = "ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(";
    assert_eq!(LAUNCH.matches(candidate).count(), 1);
    let launch_mutant = LAUNCH.replacen(candidate, incumbent, 1);
    assert_ne!(
        sha256_bytes(launch_mutant.as_bytes()).unwrap(),
        LAUNCH_SHA256
    );
    assert_eq!(PROVENANCE.matches(".env_clear()").count(), 1);
    let provenance_mutant = PROVENANCE.replacen(".env_clear()", "", 1);
    assert_ne!(
        sha256_bytes(provenance_mutant.as_bytes()).unwrap(),
        PROVENANCE_SHA256
    );
}
