// SPDX-License-Identifier: AGPL-3.0-only

//! Optional persistence (`FLUX_SETTINGS_FILE`, off by default): the runtime
//! overlay is written on every change and reapplied at the next load of the
//! SAME model. A file recorded for another model is left alone.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

const SCHEMA: &str = "flux.runtime_settings.v1";
/// The path `FLUX_SETTINGS_FILE=on` selects.
pub const DEFAULT_SETTINGS_FILE: &str = "/results/flux-runtime-settings.json";

#[derive(Debug, Clone)]
pub struct SettingsFile {
    path: PathBuf,
}

impl SettingsFile {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// From `FLUX_SETTINGS_FILE` / `ATLAS_SETTINGS_FILE`: unset, empty or
    /// `off` is off; `on` is [`DEFAULT_SETTINGS_FILE`]; anything else is a
    /// path.
    pub fn from_env() -> Result<Option<Self>> {
        let raw = crate::scheduler::think_nudge::env_setting("SETTINGS_FILE")?;
        Ok(match raw.as_deref().map(str::trim) {
            None | Some("") | Some("off") | Some("0") | Some("false") => None,
            Some("on") | Some("1") | Some("true") => {
                Some(Self::new(PathBuf::from(DEFAULT_SETTINGS_FILE)))
            }
            Some(path) => Some(Self::new(PathBuf::from(path))),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The overlay recorded for `model`. `Ok(None)`: no file, or a file for
    /// another model (logged).
    pub fn read(&self, model: &str) -> Result<Option<BTreeMap<String, Value>>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("read {}", self.path.display())),
        };
        let doc: Value = serde_json::from_str(&text)
            .with_context(|| format!("{} is not JSON", self.path.display()))?;
        if doc.get("schema").and_then(Value::as_str) != Some(SCHEMA) {
            bail!("{}: schema is not {SCHEMA}", self.path.display());
        }
        let recorded = doc.get("model").and_then(Value::as_str).unwrap_or("");
        if recorded != model {
            tracing::warn!(
                file = %self.path.display(),
                recorded,
                loaded = model,
                "Runtime settings file is for another model; not applied"
            );
            return Ok(None);
        }
        let Some(settings) = doc.get("settings").and_then(Value::as_object) else {
            bail!("{}: 'settings' must be an object", self.path.display());
        };
        Ok(Some(
            settings
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ))
    }

    /// Atomically replace the file: a fresh temp file in the same directory
    /// (random name, `O_EXCL | O_NOFOLLOW`, so a planted file or symlink is
    /// never written through), owned like the directory and mode 0600, then
    /// fsync and rename.
    pub fn write(&self, model: &str, overlay: &BTreeMap<String, Value>) -> Result<()> {
        let doc = serde_json::json!({
            "schema": SCHEMA,
            "model": model,
            "settings": overlay,
        });
        let dir = self
            .path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = self
            .path
            .file_name()
            .context("settings file path has no file name")?
            .to_string_lossy();
        let tmp = dir.join(format!(".{name}.{}.tmp", random_suffix()?));
        let mut file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?
        };
        {
            use std::os::unix::fs::MetadataExt;
            let owner = std::fs::metadata(dir).ok().map(|m| (m.uid(), m.gid()));
            crate::debug_log::adopt_owner(&file, &tmp, owner);
        }
        let body = serde_json::to_vec_pretty(&doc)?;
        let written = file
            .write_all(&body)
            .and_then(|()| file.write_all(b"\n"))
            .and_then(|()| file.sync_all());
        drop(file);
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("write {}", tmp.display()));
        }
        if let Err(e) = std::fs::rename(&tmp, &self.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("rename onto {}", self.path.display()));
        }
        Ok(())
    }
}

/// 16 random hex characters for a temp file name.
fn random_suffix() -> Result<String> {
    use ring::rand::SecureRandom;
    let mut bytes = [0u8; 8];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("no system randomness for the settings temp file"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
