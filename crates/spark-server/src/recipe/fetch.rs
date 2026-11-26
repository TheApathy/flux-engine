// SPDX-License-Identifier: AGPL-3.0-only

//! Fetching recipes from GitHub, with the cache as a first-class answer.
//!
//! **Offline is a normal state, not an error screen.** Air-gapped servers and
//! offline hosts are normal operating environments; a Library that is empty
//! without a network is a broken Library. So every read returns whatever is on
//! disk, annotated with its age, and the network only ever *improves* that
//! answer.
//!
//! # Threading
//!
//! **The one rule, for the whole dashboard: the render thread never polls a
//! future. Its only interaction with work happening elsewhere is `try_recv` on
//! a channel.**
//!
//! Background workers that need async I/O (such as `tui/chat.rs`) spawn tokio
//! tasks and answer over a `std::sync::mpsc`. Within that rule, this module
//! uses blocking `ureq` on plain `std::thread`s to keep the dependency tree
//! lean while keeping network waits off the render loop:
//! [`refresh_in_background`] spawns a thread and returns a `Receiver`, plus a
//! cancel flag so quitting mid-refresh does not leave workers running.
//!
//! One request to list, then one per file, fetched concurrently over a shared
//! agent:
//!
//! 1. `GET /repos/{repo}/git/trees/main?recursive=1` — one API call, and it
//!    yields the tree sha.
//! 2. `GET raw.githubusercontent.com/{repo}/{sha}/{path}` per recipe. Pinning
//!    the sha guarantees every file comes from one commit rather than from
//!    whatever `main` pointed at between requests. `raw.` is also not subject
//!    to the API's 60 req/hr unauthenticated limit, so a refresh spends
//!    exactly one rate-limited call.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::Recipe;
use super::fetch_github::{self, try_refresh};

/// The GitHub repo the recipe index is fetched from.
///
/// Override with `FLUX_RECIPES_REPO` (or legacy `ATLAS_RECIPES_REPO`).
const DEFAULT_REPO: &str = "TheApathy/flux-engine";

/// `FLUX_RECIPES_REPO` / `ATLAS_RECIPES_REPO` (owner/name), else [`DEFAULT_REPO`].
///
/// A set-but-empty override falls back rather than building
/// `https://api.github.com/repos//git/trees/...`, which would 404 in a way
/// that reads as "no recipes" instead of "you set the variable wrong".
pub(super) fn repo() -> &'static str {
    static REPO: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    REPO.get_or_init(|| {
        resolve_repo(
            std::env::var("FLUX_RECIPES_REPO")
                .ok()
                .or_else(|| std::env::var("ATLAS_RECIPES_REPO").ok()),
        )
    })
    .as_str()
}

/// [`repo`] over an explicit input, so the rule can be tested.
///
/// Pure over the ENVIRONMENT for the same reason `atlas_home::resolve_from` is:
/// `set_var` is unsafe and process-global, and a test that mutated it could
/// race another test's read.
pub(super) fn resolve_repo(env: Option<String>) -> String {
    match env {
        Some(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => DEFAULT_REPO.to_string(),
    }
}
pub(super) const CACHE: &str = "atlas-recipes";
pub(super) const INDEX: &str = "index.json";
/// GitHub rejects a request with no User-Agent.
pub(super) const AGENT: &str = concat!("flux-engine/", env!("CARGO_PKG_VERSION"));
pub(super) const TIMEOUT: Duration = Duration::from_secs(20);

/// What the Library renders: the recipes, and how fresh they are.
#[derive(Clone, Debug, Default)]
pub struct Index {
    pub recipes: Vec<Recipe>,
    pub tree_sha: String,
    /// Unix seconds when this was fetched from the network. 0 = never.
    pub fetched_at: u64,
    /// Set when the network failed and this came off disk instead.
    pub offline: Option<String>,
    /// Set when the network was REACHED but the result was not good enough to
    /// replace the cache with — some recipe files did not come back, or the
    /// write itself failed.
    ///
    /// Distinct from [`Self::offline`], which means the repository was never
    /// reached at all. Both mean "what is on disk is not what you just asked
    /// for", but only this one can happen on a working network, and it used to
    /// be reported as a clean success: the fetch loop logged unreachable files
    /// at `warn!` and cached whatever did arrive, so a partial fetch silently
    /// replaced a complete cache with a smaller one.
    pub incomplete: Option<String>,
}

impl Index {
    /// How old the data is, for the panel title (or `"never fetched"`).
    pub fn age_text(&self) -> String {
        if self.fetched_at == 0 {
            return "never fetched".into();
        }
        let now = unix_now();
        let secs = now.saturating_sub(self.fetched_at);
        match secs {
            0..=3599 => format!("{} m old", secs / 60),
            3600..=86399 => format!("{} h old", secs / 3600),
            _ => format!("{} d old", secs / 86400),
        }
    }

    /// The one line the Library puts in its title.
    pub fn status_text(&self) -> String {
        let age = self.age_text();
        match &self.offline {
            Some(_) => format!("⚠ {age} — offline"),
            None => age,
        }
    }

    /// Why the last fetch failed, phrased for someone who has to fix it.
    ///
    /// The title only has room for "offline", which tells the reader nothing
    /// they cannot already see. The cause matters and differs completely: a
    /// box with no default route needs a proxy, a 403 needs a rate-limit wait,
    /// and a TLS failure needs neither.
    pub fn offline_detail(&self) -> Option<String> {
        let raw = self.offline.as_ref()?;
        let lowered = raw.to_lowercase();
        let hint = if lowered.contains("dns")
            || lowered.contains("resolve")
            || lowered.contains("unreachable")
            || lowered.contains("no route")
        {
            "This machine has no route to github.com. Set HTTPS_PROXY to a host \
             that does — recipes are then fetched through it — or copy the \
             cached index (~/.atlas/atlas-recipes/index.json) from a machine \
             that can reach it."
        } else if lowered.contains("403") || lowered.contains("rate") {
            "GitHub is rate-limiting this IP. The listing costs one API call per \
             refresh; the cached recipes below are still usable."
        } else if lowered.contains("timed out") || lowered.contains("timeout") {
            "The request timed out. A slow or filtered link will do this; the \
             cached recipes below are still usable."
        } else {
            "The cached recipes below are still usable."
        };
        Some(format!("{raw}. {hint}"))
    }
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The recipe cache under `root` (`<root>/atlas-recipes`).
///
/// `pub(crate)` rather than `pub(super)` because CLI help messages that name
/// the cache location share this single source of truth.
pub(crate) fn cache_dir(root: &Path) -> PathBuf {
    root.join(CACHE)
}

/// Read whatever is cached. Never touches the network, so the Library can draw
/// before a fetch has finished — or without one ever succeeding.
pub fn cached(root: &Path) -> Index {
    let path = cache_dir(root).join(INDEX);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        // An index that is not there yet is the ordinary state of a box that
        // has never synced, and an empty Library is the right answer for it.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Index::default(),
        // Any other I/O error (such as EACCES when the cache directory is owned
        // by a different UID) must be surfaced rather than collapsed into an
        // empty store, matching `parse_cache`'s handling of corrupt files.
        Err(e) => {
            return Index {
                offline: Some(format!("{} could not be read: {e}", path.display())),
                ..Index::default()
            };
        }
    };
    match parse_cache(&text) {
        Ok(index) => index,
        // A corrupt cache is not worth a crash or a modal; the next refresh
        // overwrites it.
        Err(e) => Index {
            offline: Some(format!("cached index unreadable: {e}")),
            ..Index::default()
        },
    }
}

fn parse_cache(text: &str) -> Result<Index> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let files = doc
        .get("files")
        .and_then(|f| f.as_object())
        .context("no `files` object")?;
    let mut recipes = Vec::new();
    for (id, content) in files {
        let Some(body) = content.as_str() else {
            bail!("{id} is not text");
        };
        // A single unreadable recipe must not blank the whole Library.
        match Recipe::parse(id.clone(), body) {
            Ok(r) => recipes.push(r),
            Err(e) => tracing::warn!("skipping cached recipe {id}: {e:#}"),
        }
    }
    recipes.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Index {
        recipes,
        tree_sha: doc
            .get("tree_sha")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string(),
        fetched_at: doc.get("fetched_at").and_then(|s| s.as_u64()).unwrap_or(0),
        offline: None,
        incomplete: None,
    })
}

/// Fetch from GitHub, falling back to the cache on any failure.
///
/// Blocking, and safe to call directly only from a thread that is allowed to
/// block for [`TIMEOUT`]. From the UI use [`refresh_in_background`].
pub fn refresh(root: &Path, cancel: &AtomicBool) -> Index {
    refresh_with(root, || try_refresh(root, cancel))
}

/// The fallback rule, with the network injected so it can be tested without
/// one. Any fetch failure serves the cache, annotated with why it is stale.
fn refresh_with(root: &Path, fetch: impl FnOnce() -> Result<Index>) -> Index {
    match fetch() {
        Ok(index) => index,
        Err(e) => {
            let mut fallback = cached(root);
            fallback.offline = Some(one_line(&format!("{e:#}")));
            fallback
        }
    }
}

/// Run [`refresh`] on a dedicated thread, delivering the result over a channel.
///
/// A plain `std::thread` rather than `tokio::spawn_blocking`: this side of the
/// program has no runtime, and borrowing one only to run blocking I/O on it
/// would be exactly the mixing this module avoids. A failed spawn yields a
/// receiver that resolves to the cache, so the caller has no error path.
///
/// The returned flag cancels the refresh. It is checked between files rather
/// than inside one, which is enough: the files are a few KB each, so the wait
/// it removes is the remaining *queue*, not the request in flight. Quitting the
/// dashboard mid-refresh no longer leaves workers running to the 20 s timeout.
pub fn refresh_in_background(root: &Path) -> (std::sync::mpsc::Receiver<Index>, Arc<AtomicBool>) {
    let owned = root.to_path_buf();
    let cancel = Arc::new(AtomicBool::new(false));
    let rx = crate::tui::worker::spawn(
        "atlas-recipes",
        {
            let cancel = Arc::clone(&cancel);
            move || refresh(&owned, &cancel)
        },
        // Still answer, with what is on disk.
        |e| {
            let mut index = cached(root);
            index.offline = Some(format!("fetcher thread unavailable: {e}"));
            index
        },
    );
    (rx, cancel)
}

/// Collapse a multi-line error chain into something a title bar can hold.
fn one_line(s: &str) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() <= 90 {
        return flat;
    }
    flat.chars().take(90).collect::<String>() + "…"
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod tests;

/// Ask GitHub when one recipe was last changed, off the render thread.
///
/// The lazy half of the recipe date: `metadata.updated` is authoritative when a
/// recipe carries one, and this fills the gap for the rest. It is deliberately
/// **one recipe at a time, on demand** — see [`fetch_github::commit_date`] for
/// why dating the whole index is not an option.
///
/// Same threading contract as [`refresh_in_background`], and for the same
/// reason: a plain `std::thread` and a `std::sync::mpsc::Receiver` the UI polls
/// on its tick. Nothing here touches the async runtime.
///
/// The recipe id is echoed back in the message because by the time an answer
/// arrives the selection may have moved, and a date applied to whatever happens
/// to be selected then would be silently wrong.
pub fn updated_in_background(id: &str) -> std::sync::mpsc::Receiver<(String, Option<String>)> {
    let owned = id.to_string();
    let fallback_id = id.to_string();
    crate::tui::worker::spawn(
        "atlas-recipe-date",
        move || {
            let date = fetch_github::commit_date(&owned)
                .map_err(|e| {
                    // Offline is normal here, exactly as for the index: a date
                    // we could not look up is an absent row, never an error
                    // screen.
                    tracing::debug!("could not date recipe {owned}: {e:#}");
                })
                .ok();
            (owned, date)
        },
        |_| (fallback_id, None),
    )
}
