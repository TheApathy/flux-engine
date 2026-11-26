// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::model::seqstate::format::{BlobWriter, SeqStateHeader};
use std::sync::atomic::{AtomicU64, Ordering};

/// MemAvailable that tests set; it does not grow as the store frees RAM.
struct FakeMem(Arc<AtomicU64>);

impl MemProbe for FakeMem {
    fn mem_available(&self) -> Result<u64> {
        Ok(self.0.load(Ordering::SeqCst))
    }
}

fn store(mem: u64, cfg: StoreConfig) -> (StateStore, Arc<AtomicU64>) {
    let cell = Arc::new(AtomicU64::new(mem));
    (
        StateStore::new(cfg, Box::new(FakeMem(cell.clone()))).unwrap(),
        cell,
    )
}

fn cfg(dir: Option<PathBuf>) -> StoreConfig {
    StoreConfig {
        ram_margin_bytes: 1000,
        ram_release_slack_bytes: 100,
        disk_dir: dir,
        disk_cap_bytes: 10_000,
        disk_free_min_bytes: 0,
        write_bytes_per_hour: 0,
    }
}

const SCOPE: Scope = Scope {
    tenant: 1,
    fingerprint: [9u8; DIGEST_BYTES],
};

fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "flux-seqstate-store-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A real blob header, so the disk tier can be re-indexed after a restart.
fn blob(tokens: &[u32], fingerprint: [u8; DIGEST_BYTES], pad: usize) -> Vec<u8> {
    let header = SeqStateHeader {
        fingerprint,
        seq_len: tokens.len() as u64,
        block_size: 16,
        num_blocks: tokens.len().div_ceil(16) as u32,
        flags: 0,
        prompt_len: tokens.len() as u64,
        mtp_num_blocks: 0,
        mtp_hidden_selected_row: 0,
        mtp_seq_len: 0,
        session_hash: 0,
        mtp_last_num_drafted: 0,
        mtp_hidden_position: 0,
        layout_digest: [0u8; DIGEST_BYTES],
        tokens: tokens.to_vec(),
        mtp_last_drafts: vec![],
    };
    let mut out = Vec::new();
    let w = BlobWriter::begin(&mut out, &header, vec![]).unwrap();
    w.finish().unwrap();
    out.resize(out.len() + pad, 0xEE);
    out
}

#[test]
fn lookup_returns_the_longest_exact_prefix_within_scope() {
    let (s, _) = store(1 << 40, cfg(None));
    s.put(SCOPE, vec![1, 2], blob(&[1, 2], SCOPE.fingerprint, 0))
        .unwrap();
    s.put(
        SCOPE,
        vec![1, 2, 3, 4],
        blob(&[1, 2, 3, 4], SCOPE.fingerprint, 0),
    )
    .unwrap();
    s.put(SCOPE, vec![1, 9, 3], blob(&[1, 9, 3], SCOPE.fingerprint, 0))
        .unwrap();
    assert_eq!(s.lookup(&SCOPE, &[1, 2, 3, 4, 5]).unwrap().tokens, 4);
    assert_eq!(s.lookup(&SCOPE, &[1, 2, 3]).unwrap().tokens, 2);
    assert!(s.lookup(&SCOPE, &[2, 2]).is_none());
    // A different tenant or model configuration never sees the entry.
    let other_tenant = Scope { tenant: 2, ..SCOPE };
    let other_model = Scope {
        fingerprint: [8u8; DIGEST_BYTES],
        ..SCOPE
    };
    assert!(s.lookup(&other_tenant, &[1, 2, 3, 4]).is_none());
    assert!(s.lookup(&other_model, &[1, 2, 3, 4]).is_none());
}

#[test]
fn put_below_the_margin_without_disk_is_refused_not_kept() {
    let (s, _) = store(500, cfg(None));
    let events = s
        .put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 0))
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [Event::Refused { tokens: 1, .. }]
    ));
    assert!(s.lookup(&SCOPE, &[1]).is_none());
    assert_eq!(s.stats().1, 0);
}

#[test]
fn margin_pressure_demotes_lru_to_disk_and_hits_still_resolve() {
    let dir = tmpdir("demote");
    let (s, mem) = store(1 << 40, cfg(Some(dir.clone())));
    s.put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 10))
        .unwrap();
    s.put(SCOPE, vec![2], blob(&[2], SCOPE.fingerprint, 10))
        .unwrap();
    // Touch [1] so [2] is least recently used.
    assert!(s.lookup(&SCOPE, &[1]).is_some());
    mem.store(900, Ordering::SeqCst);
    let events = s.enforce_margin().unwrap();
    assert!(!events.is_empty());
    assert!(matches!(events[0], Event::Demoted { tokens: 1, .. }));
    // Both demoted (freeing never reached margin + slack under the fake),
    // LRU first; both still resolve, now from disk.
    for t in [1u32, 2] {
        match s.lookup(&SCOPE, &[t]).unwrap().payload {
            Payload::Disk(path) => assert!(path.exists()),
            Payload::Ram(_) => panic!("entry {t} still in RAM under pressure"),
        }
    }
    let (ram, ram_bytes, disk, _) = s.stats();
    assert_eq!((ram, ram_bytes, disk), (0, 0, 2));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn without_disk_pressure_drops_and_says_why() {
    let (s, mem) = store(1 << 40, cfg(None));
    s.put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 0))
        .unwrap();
    mem.store(10, Ordering::SeqCst);
    let events = s.enforce_margin().unwrap();
    assert!(matches!(
        events.as_slice(),
        [Event::Dropped {
            reason: "no disk tier",
            ..
        }]
    ));
    assert!(s.lookup(&SCOPE, &[1]).is_none());
}

#[test]
fn spent_write_budget_drops_instead_of_writing() {
    let dir = tmpdir("budget");
    let mut c = cfg(Some(dir.clone()));
    c.write_bytes_per_hour = 1; // smaller than any blob
    let (s, mem) = store(1 << 40, c);
    s.put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 0))
        .unwrap();
    mem.store(10, Ordering::SeqCst);
    let events = s.enforce_margin().unwrap();
    assert!(matches!(
        events.as_slice(),
        [Event::Dropped {
            reason: "write budget spent",
            ..
        }]
    ));
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn disk_cap_evicts_the_oldest_disk_entry() {
    let dir = tmpdir("cap");
    let one = blob(&[1], SCOPE.fingerprint, 0).len() as u64;
    let mut c = cfg(Some(dir.clone()));
    c.disk_cap_bytes = one * 2 + one / 2; // room for two
    let (s, _) = store(10, c); // below the margin: every put goes to disk
    for t in 1..=3u32 {
        s.put(SCOPE, vec![t], blob(&[t], SCOPE.fingerprint, 0))
            .unwrap();
    }
    assert!(s.lookup(&SCOPE, &[1]).is_none(), "oldest evicted");
    assert!(s.lookup(&SCOPE, &[2]).is_some());
    assert!(s.lookup(&SCOPE, &[3]).is_some());
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn restart_reindexes_matching_entries_and_deletes_the_rest() {
    let dir = tmpdir("reindex");
    {
        let (s, _) = store(10, cfg(Some(dir.clone())));
        s.put(SCOPE, vec![5, 6], blob(&[5, 6], SCOPE.fingerprint, 3))
            .unwrap();
        let stale = Scope {
            fingerprint: [7u8; DIGEST_BYTES],
            ..SCOPE
        };
        s.put(stale, vec![5], blob(&[5], stale.fingerprint, 0))
            .unwrap();
    }
    fs::write(dir.join("junk.fss.tmp"), b"half").unwrap();
    let (s, _) = store(1 << 40, cfg(Some(dir.clone())));
    assert_eq!(s.load_disk(SCOPE.fingerprint).unwrap(), 1);
    assert_eq!(s.lookup(&SCOPE, &[5, 6, 7]).unwrap().tokens, 2);
    assert_eq!(
        fs::read_dir(&dir).unwrap().count(),
        1,
        "stale and temp files deleted"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn replacing_an_entry_frees_its_old_bytes() {
    let (s, _) = store(1 << 40, cfg(None));
    s.put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 50))
        .unwrap();
    s.put(SCOPE, vec![1], blob(&[1], SCOPE.fingerprint, 0))
        .unwrap();
    let (ram, bytes, _, _) = s.stats();
    assert_eq!(ram, 1);
    assert_eq!(bytes, blob(&[1], SCOPE.fingerprint, 0).len() as u64);
    s.remove(&SCOPE, &[1]);
    assert_eq!(s.stats(), (0, 0, 0, 0));
}

#[test]
fn meminfo_parsing() {
    let text = "MemTotal:       124000000 kB\nMemFree: 1 kB\nMemAvailable:   2048 kB\n";
    assert_eq!(parse_mem_available(text).unwrap(), 2048 * 1024);
    assert!(parse_mem_available("MemTotal: 1 kB\n").is_err());
    assert!(parse_mem_available("MemAvailable: x kB\n").is_err());
}

#[test]
fn concurrent_demote_and_remove_preserves_accounting_and_leaves_no_orphans() {
    let dir = tmpdir("concurrent-demote");
    let mut c = cfg(Some(dir.clone()));
    c.disk_cap_bytes = 100_000_000;
    let (s, mem) = store(1 << 40, c);
    let s = Arc::new(s);
    for t in 1..=16u32 {
        s.put(SCOPE, vec![t], blob(&[t], SCOPE.fingerprint, 2048))
            .unwrap();
    }
    mem.store(100, Ordering::SeqCst);

    let barrier = Arc::new(std::sync::Barrier::new(3));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let s = s.clone();
        let b = barrier.clone();
        handles.push(std::thread::spawn(move || {
            b.wait();
            s.enforce_margin().unwrap();
        }));
    }
    {
        let s = s.clone();
        let b = barrier.clone();
        handles.push(std::thread::spawn(move || {
            b.wait();
            for t in 1..=16u32 {
                s.remove(&SCOPE, &[t]);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    for t in 1..=16u32 {
        s.remove(&SCOPE, &[t]);
    }
    assert_eq!(s.stats(), (0, 0, 0, 0));
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    fs::remove_dir_all(dir).unwrap();
}
