// SPDX-License-Identifier: AGPL-3.0-only

//! The local admin token: the secret a keyless settings change must present.
//!
//! Without a configured API key, `PATCH /v1/flux/settings` needs header
//! [`HEADER`] equal to this token. It is generated at every server start
//! (32 random bytes, hex), written to `admin.<port>.token` in a directory
//! that only this user owns and can write (the file is mode 0600), and never
//! logged or returned by an endpoint.
//! `flux set` inside the container reads the file and sends it, so
//! `docker exec <ctr> flux set ...` keeps working with no setup, while a
//! proxy or tunnel that forwards a remote client (with or without forwarding
//! headers) still has no secret to send.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};

/// The request header carrying the token.
pub const HEADER: &str = "x-flux-admin-token";
/// Where the token goes when `FLUX_ADMIN_TOKEN_FILE` is unset: the first
/// of these directories that is ours and private (see [`dir_is_private`]),
/// then `$XDG_RUNTIME_DIR/flux`.
pub const DEFAULT_DIRS: &[&str] = &["/run/flux"];

/// `admin.<port>.token`: one file per server port, so two servers in one
/// container (or on one host) never overwrite each other's token.
pub fn file_name(port: u16) -> String {
    format!("admin.{port}.token")
}

static TOKEN: OnceLock<Option<String>> = OnceLock::new();

/// The token of this process, `None` if it could not be written (keyless
/// settings changes are then refused).
pub fn current() -> Option<&'static str> {
    TOKEN.get().and_then(|t| t.as_deref())
}

/// Generate the token and write its file for the server on `port`; once
/// per process, at startup. A failure is logged (never fatal): keyless
/// changes are then refused.
pub fn init(port: u16) {
    TOKEN.get_or_init(|| match generate().and_then(|t| write_token(&t, port).map(|p| (t, p))) {
        Ok((token, path)) => {
            tracing::info!(
                file = %path.display(),
                "Settings admin token written (keyless `flux set` inside this container reads it)"
            );
            Some(token)
        }
        Err(e) => {
            tracing::warn!(
                "No settings admin token ({e:#}): without an API key, settings cannot be \
                 changed at runtime; set FLUX_ADMIN_TOKEN_FILE to a writable path or start with \
                 --require-auth"
            );
            None
        }
    });
}

/// Candidate token paths for `port`, in order: `FLUX_ADMIN_TOKEN_FILE` /
/// `ATLAS_ADMIN_TOKEN_FILE` (used as given), then [`DEFAULT_DIRS`], then
/// `$XDG_RUNTIME_DIR/flux`.
pub fn candidate_paths(port: u16) -> Vec<PathBuf> {
    if let Ok(Some(path)) = crate::scheduler::think_nudge::env_setting("ADMIN_TOKEN_FILE")
        && !path.trim().is_empty()
    {
        return vec![PathBuf::from(path.trim())];
    }
    let name = file_name(port);
    let mut paths: Vec<PathBuf> = DEFAULT_DIRS
        .iter()
        .map(|d| Path::new(d).join(&name))
        .collect();
    if let Some(dir) = std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        paths.push(Path::new(&dir).join("flux").join(&name));
    }
    paths
}

fn generate() -> Result<String> {
    use ring::rand::SecureRandom;
    let mut bytes = [0u8; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("no system randomness"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn write_token(token: &str, port: u16) -> Result<PathBuf> {
    let mut last = anyhow::anyhow!("no candidate path");
    for path in candidate_paths(port) {
        match write_at(&path, token) {
            Ok(()) => return Ok(path),
            Err(e) => {
                tracing::debug!(path = %path.display(), "admin token path skipped: {e:#}");
                last = e;
            }
        }
    }
    Err(last)
}

/// The token directory must be a real directory (not a symlink), owned by
/// this user, and not group- or world-writable: otherwise another local
/// user could swap the file under us or read it.
pub fn dir_is_private(dir: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let meta =
        std::fs::symlink_metadata(dir).with_context(|| format!("lstat {}", dir.display()))?;
    if !meta.file_type().is_dir() {
        anyhow::bail!("{} is not a directory (or is a symlink)", dir.display());
    }
    // SAFETY: geteuid has no preconditions and cannot fail.
    let euid = unsafe { libc::geteuid() };
    if meta.uid() != euid {
        anyhow::bail!(
            "{} is owned by uid {}, not {euid}",
            dir.display(),
            meta.uid()
        );
    }
    if meta.mode() & 0o022 != 0 {
        anyhow::bail!("{} is group- or world-writable", dir.display());
    }
    Ok(())
}

/// Replace `path` with a fresh 0600 file holding `token`. A missing
/// directory is created 0700; an existing one must pass [`dir_is_private`].
/// The old file is unlinked first and the new one is opened
/// `O_EXCL | O_NOFOLLOW`, so nothing planted there is written through.
pub fn write_at(path: &Path, token: &str) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .context("admin token path has no directory")?;
    if std::fs::symlink_metadata(dir).is_err() {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .with_context(|| format!("create {}", dir.display()))?;
    }
    dir_is_private(dir)?;
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("remove {}", path.display())),
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(token.as_bytes())
        .and_then(|()| file.sync_all())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Constant-time token comparison.
pub fn matches(expected: &str, presented: &[u8]) -> bool {
    let expected = expected.as_bytes();
    if expected.len() != presented.len() {
        return false;
    }
    expected
        .iter()
        .zip(presented)
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}
