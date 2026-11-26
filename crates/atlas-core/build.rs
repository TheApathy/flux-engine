// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;

fn canonical_hw_name(raw: &str) -> String {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "sm120" => "rtx6000".to_owned(),
        "sm121" => "gb10".to_owned(),
        "sm103" => "gb300".to_owned(),
        _ => trimmed.to_owned(),
    }
}

fn parse_hardware_toml_fields(content: &str) -> (Option<String>, Option<bool>) {
    let mut in_hardware = false;
    let mut extends = None;
    let mut use_fp32_residual = None;

    for raw_line in content.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_hardware = line == "[hardware]";
            continue;
        }
        if !in_hardware {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim();
        match key {
            "extends" => {
                if let Some(stripped) = val.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                    && !stripped.trim().is_empty()
                {
                    extends = Some(canonical_hw_name(stripped));
                }
            }
            "use_fp32_residual" => match val {
                "true" => use_fp32_residual = Some(true),
                "false" => use_fp32_residual = Some(false),
                _ => {}
            },
            _ => {}
        }
    }

    (extends, use_fp32_residual)
}

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");
    println!("cargo:rerun-if-env-changed=FLUX_TARGET_HW");

    let hw = env::var("ATLAS_TARGET_HW")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            env::var("FLUX_TARGET_HW")
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        })
        .map(|v| canonical_hw_name(&v))
        .unwrap_or_else(|| "gb10".to_owned());

    let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let Some(workspace_root) = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
    else {
        return;
    };
    let kernels_root = workspace_root.join("kernels");

    let mut current = hw;
    let mut visited = BTreeSet::new();
    let mut fp32_residual: Option<bool> = None;

    while visited.insert(current.clone()) {
        let toml_path = kernels_root.join(&current).join("HARDWARE.toml");
        if !toml_path.is_file() {
            break;
        }
        println!("cargo:rerun-if-changed={}", toml_path.display());
        let Ok(content) = std::fs::read_to_string(&toml_path) else {
            break;
        };
        let (extends, value) = parse_hardware_toml_fields(&content);
        if fp32_residual.is_none() {
            fp32_residual = value;
        }
        match extends {
            Some(base) => current = base,
            None => break,
        }
    }

    if let Some(enabled) = fp32_residual {
        println!(
            "cargo:rustc-env=ATLAS_HW_FP32_RESIDUAL={}",
            if enabled { "true" } else { "false" }
        );
    }
}
