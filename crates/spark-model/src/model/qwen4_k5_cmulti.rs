// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off serial native-MTP K5 for Qwen4 C2-C8 (`ATLAS_QWEN4_K5_CMULTI=1`).
//!
//! Every live request keeps the exact C1 K5 transaction: verify, commit and
//! hidden save run back to back for one request before the scheduler touches
//! the next. Nothing here batches verify rows across requests. The checks make
//! the cross-request ownership facts that serial execution relies on explicit,
//! so a violation fails the frame instead of committing another request's
//! state.

use std::ffi::OsStr;

pub const QWEN4_K5_CMULTI_ENV: &str = "ATLAS_QWEN4_K5_CMULTI";
/// Widest concurrency admitted by the serial K5 gate.
pub const QWEN4_K5_CMULTI_MAX_CONCURRENCY: usize = 8;
/// K5: four native MTP drafts plus the bonus row.
pub const QWEN4_K5_CMULTI_NUM_DRAFTS: usize = 4;

pub fn parse_k5_cmulti(raw: Option<&OsStr>) -> Result<bool, &'static str> {
    match raw.map(OsStr::to_str) {
        None | Some(Some("0")) => Ok(false),
        Some(Some("1")) => Ok(true),
        Some(Some(_)) => Err("ATLAS_QWEN4_K5_CMULTI must be exact value 0 or 1"),
        Some(None) => Err("ATLAS_QWEN4_K5_CMULTI must be valid UTF-8 and exactly 0 or 1"),
    }
}

pub fn k5_cmulti_selected() -> Result<bool, &'static str> {
    parse_k5_cmulti(std::env::var_os(QWEN4_K5_CMULTI_ENV).as_deref())
}

/// Model-side enforcement is on unless the gate is cleanly off. Startup
/// refuses a malformed value; enforcing on one costs nothing and can only
/// refuse a frame.
pub(crate) fn k5_cmulti_enforced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let enforced = !matches!(k5_cmulti_selected(), Ok(false));
        if enforced {
            tracing::info!("QWEN4_K5_CMULTI_ENGAGED serial=true journal_owner=true tree=refused");
        }
        enforced
    })
}

/// The request that armed the per-layer QSA verify journals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QsaJournalOwner {
    pub slot: usize,
    pub session: u64,
}

/// Host-side owner of the one-per-layer QSA undo journal.
///
/// The journal has no slot dimension (`QsaCache::verify_undo_*`). Serial K5
/// keeps it correct only while every frame that arms it is committed or
/// aborted before another request's frame starts. Arming is idempotent for
/// the same owner; a different owner is refused, and so is a frame of any
/// width whose commit would consume a journal it does not own.
#[derive(Debug, Default)]
pub(crate) struct QsaJournalOwnership {
    armed: Option<QsaJournalOwner>,
}

impl QsaJournalOwnership {
    /// Refuse to start a frame while another request owns the journal.
    pub(crate) fn admit_frame(&self, owner: QsaJournalOwner) -> Result<(), &'static str> {
        match self.armed {
            Some(current) if current != owner => Err(
                "Qwen4 serial K5: QSA verify journal is still armed by another request; \
                 its frame must commit or abort before the next slot verifies",
            ),
            _ => Ok(()),
        }
    }

    pub(crate) fn arm(&mut self, owner: QsaJournalOwner) -> Result<(), &'static str> {
        self.admit_frame(owner)?;
        self.armed = Some(owner);
        Ok(())
    }

    /// Check, before a commit is launched, that `owner` may consume the
    /// journal. An unarmed journal (a frame below the QSA boundary) passes.
    pub(crate) fn check_commit(&self, owner: QsaJournalOwner) -> Result<(), &'static str> {
        match self.armed {
            Some(current) if current != owner => {
                Err("Qwen4 serial K5: QSA journal commit by a request that does not own it")
            }
            _ => Ok(()),
        }
    }

    /// Every launched commit or restore consumes the journal.
    pub(crate) fn release(&mut self) {
        self.armed = None;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.armed.is_none()
    }
}

/// Model-wide DFlash tree state that a flat native-MTP chain never sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeSingletons {
    pub tree_tokens: usize,
    pub host_parents: bool,
    pub device_parents: bool,
    pub flat_tree_route: bool,
}

pub(crate) fn check_tree_singletons_unset(state: TreeSingletons) -> Result<(), &'static str> {
    let tree_active = state.tree_tokens != 0
        || state.host_parents
        || state.device_parents
        || state.flat_tree_route;
    if tree_active {
        return Err("Qwen4 serial K5 requires the model-wide DFlash tree state to stay unset");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn gate_is_exact_and_default_off() {
        assert_eq!(parse_k5_cmulti(None), Ok(false));
        assert_eq!(parse_k5_cmulti(Some(OsStr::new("0"))), Ok(false));
        assert_eq!(parse_k5_cmulti(Some(OsStr::new("1"))), Ok(true));
        for hostile in ["", "true", "01", "2", " 1", "1 "] {
            assert!(
                parse_k5_cmulti(Some(OsStr::new(hostile))).is_err(),
                "{hostile:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_gate_value_is_refused() {
        use std::os::unix::ffi::OsStringExt;
        let raw = OsString::from_vec(vec![0xff]);
        assert!(parse_k5_cmulti(Some(raw.as_os_str())).is_err());
    }

    const A: QsaJournalOwner = QsaJournalOwner {
        slot: 0,
        session: 0xa,
    };
    const B: QsaJournalOwner = QsaJournalOwner {
        slot: 1,
        session: 0xb,
    };

    #[test]
    fn journal_refuses_cross_owner_prepare_until_released() {
        let mut journal = QsaJournalOwnership::default();
        assert!(journal.is_empty());
        journal.arm(A).unwrap();
        journal.arm(A).unwrap();
        assert!(journal.arm(B).unwrap_err().contains("another request"));
        assert!(journal.admit_frame(B).is_err());
        assert!(
            journal
                .check_commit(B)
                .unwrap_err()
                .contains("does not own")
        );
        journal.check_commit(A).unwrap();
        journal.release();
        assert!(journal.is_empty());
        journal.arm(B).unwrap();
        assert!(journal.admit_frame(A).is_err());
        journal.release();
        journal.admit_frame(A).unwrap();
    }

    #[test]
    fn same_slot_new_session_is_a_different_owner() {
        let mut journal = QsaJournalOwnership::default();
        journal.arm(A).unwrap();
        let reused = QsaJournalOwner {
            slot: A.slot,
            session: 0xc,
        };
        assert!(journal.arm(reused).is_err());
        assert!(journal.check_commit(reused).is_err());
    }

    #[test]
    fn unarmed_journal_admits_any_frame_and_commit() {
        let journal = QsaJournalOwnership::default();
        journal.admit_frame(B).unwrap();
        journal.check_commit(B).unwrap();
    }

    #[test]
    fn every_tree_singleton_is_refused() {
        let clean = TreeSingletons {
            tree_tokens: 0,
            host_parents: false,
            device_parents: false,
            flat_tree_route: false,
        };
        check_tree_singletons_unset(clean).unwrap();
        for dirty in [
            TreeSingletons {
                tree_tokens: 3,
                ..clean
            },
            TreeSingletons {
                host_parents: true,
                ..clean
            },
            TreeSingletons {
                device_parents: true,
                ..clean
            },
            TreeSingletons {
                flat_tree_route: true,
                ..clean
            },
        ] {
            assert!(check_tree_singletons_unset(dirty).is_err(), "{dirty:?}");
        }
    }
}
