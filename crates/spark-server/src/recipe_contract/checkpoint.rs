// SPDX-License-Identifier: AGPL-3.0-only

//! Read checkpoint identity from disk: `config.json` model_type and the
//! sha256 of every pinned file.

use super::facts::CheckpointFacts;
use super::schema::Checkpoint;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The local model directory, or `None` when the model is only a hub id
/// (a recipe then refuses: identity cannot be checked before download).
pub fn model_dir(model_from_path: Option<&Path>, model: Option<&str>) -> Option<PathBuf> {
    model_from_path
        .map(Path::to_path_buf)
        .or_else(|| model.map(PathBuf::from).filter(|p| p.is_dir()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    Ok(ctx
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn model_type(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value.get("model_type")?.as_str().map(str::to_string)
}

pub fn read(dir: Option<&Path>, pins: &Checkpoint) -> CheckpointFacts {
    let Some(dir) = dir else {
        let digests = pins
            .sha256
            .keys()
            .map(|f| (f.clone(), Err("model is not a local directory".to_string())))
            .collect();
        return CheckpointFacts {
            model_type: None,
            digests,
        };
    };
    let digests = pins
        .sha256
        .keys()
        .map(|file| {
            let rel = Path::new(file);
            let safe = rel
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)));
            let digest = if safe {
                sha256_file(&dir.join(rel))
            } else {
                Err("path escapes the model directory".into())
            };
            (file.clone(), digest)
        })
        .collect();
    CheckpointFacts {
        model_type: model_type(dir),
        digests,
    }
}
