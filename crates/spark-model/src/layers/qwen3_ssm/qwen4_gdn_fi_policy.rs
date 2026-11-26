// SPDX-License-Identifier: AGPL-3.0-only

// Shape admission for the opt-in FlashInfer GDN prefill route.
//
// Repeated cold-prefill qualification found an asynchronous state race at
// 64 rows and below.  The native FP32 recurrence is already the exact,
// same-stream fallback, so keep the unqualified short shapes on that route.

/// Largest row count for which the asynchronous FI-GDN state path is not
/// qualified.  The boundary is inclusive: 64 uses the exact recurrence and
/// 65 remains eligible for FlashInfer.
pub(super) const SHORT_FALLBACK_MAX_ROWS: usize = 64;

/// Whether one prefill shape may use the selected FlashInfer GDN route.
///
/// Fail closed: a disabled selector or an unqualified short shape always
/// resolves to the exact FP32 recurrence.
pub(super) const fn use_fi(requested: bool, rows: usize) -> bool {
    requested && rows > SHORT_FALLBACK_MAX_ROWS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_selector_never_uses_fi() {
        for rows in [0, 1, 23, 64, 65, 128, 8_192] {
            assert!(!use_fi(false, rows), "rows={rows}");
        }
    }

    #[test]
    fn selected_route_has_an_inclusive_short_fallback() {
        assert!(!use_fi(true, 1));
        assert!(!use_fi(true, 23));
        assert!(!use_fi(true, 64));
        assert!(use_fi(true, 65));
        assert!(use_fi(true, 128));
        assert!(use_fi(true, 8_192));
    }
}
