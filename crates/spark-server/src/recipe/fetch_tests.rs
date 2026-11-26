// SPDX-License-Identifier: AGPL-3.0-only

//! Cache and offline behaviour. Nothing here touches the network: the point of
//! the cache is precisely that it works when the network does not.

use super::super::fetch_github::{recipe_id, select_recipe_paths, write_cache};
use super::*;
use std::collections::BTreeMap;

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("atlas-recipes-{tag}-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn a_recipe() -> String {
    "recipe_version: \"2\"\nmodel: Qwen/Qwen3.6-27B\nruntime: atlas\ncontainer: c\n\
     metadata:\n  description: test\n  maintainer: atlas\ndefaults:\n  port: 8888\n"
        .to_string()
}

fn seed(dir: &Dir, fetched_at: u64) {
    let files = BTreeMap::from([("qwen3.6/test".to_string(), a_recipe())]);
    write_cache(&dir.0, "deadbeef", fetched_at, &files).expect("cache written");
}

#[test]
fn an_empty_store_reports_never_fetched_rather_than_failing() {
    // The first ever launch must render a Library, not an error.
    let dir = Dir::new("empty");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert_eq!(index.fetched_at, 0);
    assert_eq!(index.age_text(), "never fetched");
}

#[test]
fn a_cached_index_round_trips() {
    let dir = Dir::new("roundtrip");
    seed(&dir, unix_now());
    let index = cached(&dir.0);
    assert_eq!(index.recipes.len(), 1);
    assert_eq!(index.recipes[0].id, "qwen3.6/test");
    assert_eq!(index.recipes[0].model, "Qwen/Qwen3.6-27B");
    assert_eq!(index.tree_sha, "deadbeef");
    assert!(
        index.offline.is_none(),
        "reading a cache is not being offline"
    );
}

#[test]
fn a_corrupt_cache_degrades_instead_of_crashing() {
    let dir = Dir::new("corrupt");
    std::fs::create_dir_all(cache_dir(&dir.0)).expect("dir");
    std::fs::write(cache_dir(&dir.0).join(INDEX), "{ not json").expect("write");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert!(index.offline.is_some(), "says why it is empty");
}

#[test]
fn one_unreadable_recipe_does_not_blank_the_others() {
    // Upstream is a separate repo; a single malformed file there must cost one
    // row, not the whole Library.
    let dir = Dir::new("partial");
    let files = BTreeMap::from([
        ("good/one".to_string(), a_recipe()),
        (
            "bad/two".to_string(),
            "this: is\n\tnot: valid\n".to_string(),
        ),
    ]);
    write_cache(&dir.0, "sha", unix_now(), &files).expect("write");
    let index = cached(&dir.0);
    assert_eq!(index.recipes.len(), 1);
    assert_eq!(index.recipes[0].id, "good/one");
}

#[test]
fn age_is_reported_in_the_largest_useful_unit() {
    let mut index = Index {
        fetched_at: unix_now(),
        ..Index::default()
    };
    assert_eq!(index.age_text(), "0 m old");
    index.fetched_at = unix_now() - 7200;
    assert_eq!(index.age_text(), "2 h old");
    index.fetched_at = unix_now() - 3 * 86400;
    assert_eq!(index.age_text(), "3 d old");
}

#[test]
fn offline_is_visible_in_the_status_line() {
    let index = Index {
        fetched_at: unix_now() - 3 * 86400,
        offline: Some("dns failure".into()),
        ..Index::default()
    };
    let text = index.status_text();
    assert!(text.contains("3 d old"), "{text}");
    assert!(text.contains("offline"), "{text}");
}

#[test]
fn a_failed_fetch_serves_the_cache_and_says_it_is_stale() {
    // The network is injected rather than attempted so this test is
    // deterministic in air-gapped and CI environments.
    let dir = Dir::new("fallback");
    seed(&dir, unix_now() - 86400);
    let index = refresh_with(&dir.0, || anyhow::bail!("dns failure"));
    assert_eq!(index.recipes.len(), 1, "the cache still answered");
    assert!(index.offline.is_some(), "and it is marked stale");
    let text = index.status_text();
    assert!(text.contains("offline"), "{text}");
    assert!(text.contains("1 d old"), "with its age: {text}");
}

#[test]
fn a_successful_fetch_is_not_marked_offline() {
    let dir = Dir::new("live");
    seed(&dir, 0);
    let fresh = Index {
        recipes: Vec::new(),
        tree_sha: "abc".into(),
        fetched_at: unix_now(),
        offline: None,
        incomplete: None,
    };
    let index = refresh_with(&dir.0, || Ok(fresh));
    assert!(index.offline.is_none());
    assert_eq!(index.tree_sha, "abc", "the fetch wins over the cache");
}

#[test]
fn a_recipe_id_is_its_path_without_prefix_or_extension() {
    assert_eq!(recipe_id("recipes/qwen3.6/foo.yaml"), "qwen3.6/foo");
    assert_eq!(
        recipe_id("crates/spark-server/src/recipe/builtin/qwen3.8-27b-dflash2-safe-local.yaml"),
        "qwen3.8-27b-dflash2-safe-local"
    );
}

#[test]
fn select_recipe_paths_falls_back_to_monorepo_builtin_dir() {
    let entries = vec![
        serde_json::json!({"type": "blob", "path": "recipes/flux-fast-prefill.env"}),
        serde_json::json!({"type": "blob", "path": "crates/spark-server/src/recipe/builtin/b.yaml"}),
        serde_json::json!({"type": "blob", "path": "crates/spark-server/src/recipe/builtin/a.yaml"}),
    ];
    assert_eq!(
        select_recipe_paths(&entries),
        vec![
            "crates/spark-server/src/recipe/builtin/a.yaml",
            "crates/spark-server/src/recipe/builtin/b.yaml",
        ]
    );
}

#[test]
fn a_long_error_is_trimmed_to_fit_a_title_bar() {
    let long = "e".repeat(500);
    let out = one_line(&long);
    assert_eq!(out.chars().count(), 91);
    assert!(
        !one_line("a\nb").contains('\n'),
        "newlines would break layout"
    );
}

#[test]
fn the_cache_is_written_atomically() {
    // A half-written index read at the next start-up is indistinguishable from
    // a corrupt one, so the write goes through a temp file and a rename.
    let dir = Dir::new("atomic");
    seed(&dir, unix_now());
    assert!(cache_dir(&dir.0).join(INDEX).exists());
    assert!(
        !cache_dir(&dir.0).join("index.json.tmp").exists(),
        "the temp file is renamed, not left behind"
    );
}

/// Live fetch test against GitHub. `#[ignore]` because it requires network
/// access and CI runs offline. Run manually after changing the fetch logic:
/// `cargo test -p spark-server --bins live_fetch -- --ignored --nocapture`
#[test]
#[ignore = "needs the network"]
fn live_fetch_against_github() {
    let dir = Dir::new("live-github");
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    assert!(index.offline.is_none(), "fetch failed: {:?}", index.offline);
    assert!(!index.recipes.is_empty(), "expected at least one recipe");
    assert_eq!(index.tree_sha.len(), 40, "a full tree sha");
    for r in index.recipes.iter().filter(|r| r.is_atlas()) {
        r.serve_args(&BTreeMap::new())
            .unwrap_or_else(|e| panic!("live recipe {} is not servable: {e:#}", r.id));
    }
    // And the cache it just wrote must read back identically.
    let reread = cached(&dir.0);
    assert_eq!(reread.recipes.len(), index.recipes.len());
    assert_eq!(reread.tree_sha, index.tree_sha);
    eprintln!("{} recipes @ {}", index.recipes.len(), index.tree_sha);
}

#[test]
fn a_no_route_failure_tells_the_user_what_to_do_about_it() {
    // On an air-gapped host with no default route, "offline" alone is not
    // actionable; hinting HTTPS_PROXY tells the operator how to fix it.
    let index = Index {
        offline: Some("GET https://api.github.com/…: dns error: Network is unreachable".into()),
        fetched_at: unix_now() - 86400,
        ..Index::default()
    };
    let detail = index.offline_detail().expect("a reason");
    assert!(detail.contains("no route"), "{detail}");
    assert!(detail.contains("HTTPS_PROXY"), "names the fix: {detail}");
    // The short form still fits a title bar.
    assert!(index.status_text().len() < 40, "{}", index.status_text());
}

#[test]
fn a_rate_limit_is_distinguished_from_a_dead_link() {
    let index = Index {
        offline: Some("GET https://api.github.com/…: status 403 rate limit exceeded".into()),
        ..Index::default()
    };
    let detail = index.offline_detail().expect("a reason");
    assert!(detail.contains("rate-limiting"), "{detail}");
    assert!(
        !detail.contains("HTTPS_PROXY"),
        "a proxy does not fix a rate limit: {detail}"
    );
}

#[test]
fn a_healthy_index_has_no_reason_to_show() {
    let index = Index {
        fetched_at: unix_now(),
        ..Index::default()
    };
    assert!(index.offline_detail().is_none());
}

#[test]
fn a_cancelled_refresh_serves_the_cache_rather_than_a_partial_index() {
    // Cancellation must not look like a successful fetch that happened to
    // return fewer recipes — that would overwrite the cache with a subset.
    let dir = Dir::new("cancelled");
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let index = refresh(&dir.0, &cancel);
    assert!(
        index.offline.is_some(),
        "a cancelled refresh is not a live index"
    );
}

/// Network test: the concurrent fetch must return the same corpus the
/// sequential one did, in the same order.
#[test]
#[ignore = "needs the network"]
fn a_concurrent_refresh_is_ordered_and_complete() {
    let dir = Dir::new("concurrent-order");
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    assert!(index.offline.is_none(), "fetch failed: {:?}", index.offline);
    assert_eq!(index.recipes.len(), 25);
    // Fetch order is now nondeterministic; row order must not be.
    let mut sorted = index.recipes.clone();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        index.recipes.iter().map(|r| &r.id).collect::<Vec<_>>(),
        sorted.iter().map(|r| &r.id).collect::<Vec<_>>(),
        "recipes must be sorted regardless of which worker finished first"
    );
}

/// Network test: how long a cold refresh actually takes. Not an assertion on a
/// wall-clock number — that would be a flaky test — just a measurement, so the
/// claim that the concurrent fetch is faster can be checked rather than
/// asserted from a comment.
#[test]
#[ignore = "needs the network"]
fn measure_refresh_wall_time() {
    let dir = Dir::new("measure-refresh");
    let t = std::time::Instant::now();
    let index = refresh(&dir.0, &std::sync::atomic::AtomicBool::new(false));
    let elapsed = t.elapsed();
    eprintln!(
        "refresh: {:?} for {} recipes (offline={:?})",
        elapsed,
        index.recipes.len(),
        index.offline
    );
    assert!(index.offline.is_none());
}

/// An index that EXISTS and cannot be read is not an empty index.
///
/// For example, if the cache directory is owned by a different user and
/// unreadable (`EACCES`), `cached` must surface the I/O error rather than
/// collapsing it into an empty store.
#[test]
fn an_unreadable_index_is_not_reported_as_an_absent_one() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = Dir::new("unreadable");
        seed(&dir, unix_now());
        let path = cache_dir(&dir.0).join(INDEX);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        // Staged, then PROVEN staged. root — and any ACL — ignores the mode
        // bits, and a test that cannot create the fault would pass for a
        // reason that has nothing to do with the code. Skip there rather than
        // report a green nobody can trust.
        if std::fs::read_to_string(&path).is_ok() {
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            return;
        }

        let index = cached(&dir.0);
        let why = index.offline.as_deref().unwrap_or_default();
        assert!(
            !why.is_empty(),
            "an unreadable index must say so, not answer `no recipes`"
        );
        assert!(
            why.contains(&path.display().to_string()),
            "the operator needs the path that could not be read: {why}"
        );

        // Restore so `Dir`'s cleanup can remove it.
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
}

/// And the ordinary case must stay quiet.
///
/// The control for the test above: a check that fires on a healthy box is
/// worse than no check, and "never synced" is the state of every fresh
/// install. `NotFound` is the one read error that IS an empty index.
#[test]
fn an_index_that_was_never_written_is_still_an_ordinary_empty_store() {
    let dir = Dir::new("never-written");
    let index = cached(&dir.0);
    assert!(index.recipes.is_empty());
    assert!(
        index.offline.is_none(),
        "a box that has never synced has no fault to report, got: {:?}",
        index.offline
    );
}

/// The cache directory lives under `<root>/atlas-recipes`.
#[test]
fn the_cache_dir_uses_the_canonical_subdirectory() {
    let dir = Dir::new("cache-current");
    assert_eq!(cache_dir(&dir.0), dir.0.join(CACHE));
    std::fs::create_dir_all(dir.0.join(CACHE)).expect("current");
    assert_eq!(cache_dir(&dir.0), dir.0.join(CACHE));
}

/// The recipe index repo is one constant with an env override.
mod repo_override {
    use crate::recipe::fetch::resolve_repo;

    /// An override is honoured and trimmed — a trailing newline from
    /// `FLUX_RECIPES_REPO=$(cat file)` would otherwise build a URL with a
    /// newline in the path.
    #[test]
    fn an_override_is_used_and_trimmed() {
        assert_eq!(
            resolve_repo(Some("  me/my-recipes \n".into())),
            "me/my-recipes"
        );
    }

    /// A set-but-empty override falls back rather than building
    /// `https://api.github.com/repos//git/trees/...`, which 404s in a way that
    /// reads as "no recipes exist" instead of "the variable is wrong".
    #[test]
    fn a_set_but_empty_override_falls_back_instead_of_building_an_empty_path() {
        assert_eq!(resolve_repo(Some("   ".into())), resolve_repo(None));
        assert!(!resolve_repo(Some(String::new())).is_empty());
    }

    /// The default points to the Flux Engine repository.
    #[test]
    fn the_default_points_to_flux_engine() {
        assert_eq!(resolve_repo(None), "TheApathy/flux-engine");
    }
}
