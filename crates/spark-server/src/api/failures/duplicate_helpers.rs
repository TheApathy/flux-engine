// SPDX-License-Identifier: AGPL-3.0-only

//! F12 tool-call cap + loop watchdog helpers, hoisted from `duplicate.rs`
//! to keep that file under the 500 LoC cap.
//!
//! These two helpers are independent of the F49/F50 duplicate-write pipeline
//! that owns `duplicate.rs`; they live as siblings rather than peers so the
//! file split is invisible to callers (re-exported through `failures/mod.rs`).

/// Reserve one slot in the per-response tool-call budget. Returns `false`
/// without incrementing once the cap has been reached, so a capped call is
/// never partially published. Catches pathological responses emitting dozens
/// of tool calls under heavy looping. Default cap in `chat_stream` is
/// `usize::MAX` (`12` when `FLUX_AGENT_GUARDS=1`; env override
/// `FLUX_MAX_TOOL_CALLS_PER_RESPONSE`, fallback
/// `ATLAS_MAX_TOOL_CALLS_PER_RESPONSE`).
pub fn bump_f12_tool_call_count(count: &mut usize, max: usize) -> bool {
    if *count >= max {
        tracing::warn!(
            emitted = *count,
            attempted = (*count).saturating_add(1),
            max,
            "tool-call cap reached; ending response"
        );
        return false;
    }
    *count += 1;
    true
}

/// Repetition-loop watchdog over post-detector content (`loop_scan_buf`, ~8 KB
/// rolling window). Returns `true` when the most recent non-trivial line (>15
/// trimmed chars, outside code-fence openers) repeats >= 4 times under
/// whitespace-collapsed lowercase equality (or >= 4 substring hits for >=30-char
/// phrases). Must operate on post-detector Content only, never tool-call args.
pub fn check_loop_watchdog(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
) -> bool {
    if already_triggered || text.is_empty() {
        return false;
    }
    loop_scan_buf.push_str(text);
    if loop_scan_buf.len() > 10_240 {
        let drop = loop_scan_buf.len() - 8_192;
        let cut = loop_scan_buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= drop)
            .unwrap_or(drop);
        loop_scan_buf.drain(..cut);
    }
    let last_line = loop_scan_buf
        .lines()
        .rev()
        .find(|l| l.trim().len() > 15 && !l.trim_start().starts_with("```"))
        .map(|s| s.to_string());
    let Some(line) = last_line else {
        return false;
    };
    fn norm(s: &str) -> String {
        let lowered = s.trim().to_ascii_lowercase();
        let mut out = String::with_capacity(lowered.len());
        let mut prev_space = false;
        for ch in lowered.chars() {
            if ch.is_ascii_whitespace() {
                if !prev_space && !out.is_empty() {
                    out.push(' ');
                }
                prev_space = true;
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        if out.ends_with(' ') {
            out.pop();
        }
        out
    }

    /// Alloc-free normalised iterator: yields lowercased non-whitespace
    /// chars with whitespace runs collapsed to a single space, matching
    /// `norm()`'s output shape but without allocating a String.
    ///
    /// Used by the inner-loop `norm_eq` to count duplicate lines in the
    /// loop watchdog without paying a `norm()` String alloc per
    /// candidate line per emitted token.
    fn norm_chars(s: &str) -> impl Iterator<Item = char> + '_ {
        let trimmed = s.trim();
        let mut prev_space = false;
        let mut started = false;
        trimmed.chars().filter_map(move |raw| {
            let ch = raw.to_ascii_lowercase();
            if ch.is_ascii_whitespace() {
                if !prev_space && started {
                    prev_space = true;
                    Some(' ')
                } else {
                    None
                }
            } else {
                prev_space = false;
                started = true;
                Some(ch)
            }
        })
    }

    fn norm_eq(a: &str, b: &str) -> bool {
        norm_chars(a).eq(norm_chars(b))
    }

    let needle = norm(&line);
    if needle.is_empty() {
        return false;
    }
    // `needle` is already normalised; pass `&line` here so `norm_eq`
    // doesn't trim+lowercase it twice — both arms produce the same
    // normalised sequence (line is the source we computed `needle` from).
    let exact_occurrences = loop_scan_buf.lines().filter(|l| norm_eq(l, &line)).count();
    if exact_occurrences >= 4 {
        tracing::warn!(
            occurrences = exact_occurrences,
            line_len = needle.len(),
            "loop watchdog fired — repeated line (fuzzy-match) in post-detector content"
        );
        return true;
    }
    // Substring fallback: catches a phrase that recurs whole even
    // when one occurrence is glued onto another line (mid-stream
    // narration ramping). Only count for ≥30-char phrases so we
    // don't false-positive on short common fragments.
    //
    // Earlier this allocated a per-call lowered copy of loop_scan_buf
    // (up to 10 KB) just to do `.find(&needle)`. Replaced with an
    // alloc-free case-insensitive byte search: walks `loop_scan_buf`
    // once, lowercasing each byte inline while checking equality with
    // `needle`. Short-circuits at 4 occurrences.
    if needle.len() >= 30 {
        let needle_bytes = needle.as_bytes();
        let buf_bytes = loop_scan_buf.as_bytes();
        let nlen = needle_bytes.len();
        let mut count = 0usize;
        let mut i = 0usize;
        while i + nlen <= buf_bytes.len() {
            let matches = buf_bytes[i..i + nlen]
                .iter()
                .zip(needle_bytes.iter())
                .all(|(b1, b2)| b1.eq_ignore_ascii_case(b2));
            if matches {
                count += 1;
                if count >= 4 {
                    break;
                }
                i += nlen;
            } else {
                i += 1;
            }
        }
        if count >= 4 {
            tracing::warn!(
                occurrences = count,
                line_len = needle.len(),
                "loop watchdog fired — repeated phrase (substring) in post-detector content"
            );
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── bump_f12_tool_call_count ──────────────────────────────────

    #[test]
    fn f12_under_cap_reserves_slot() {
        let mut count = 0usize;
        assert!(bump_f12_tool_call_count(&mut count, 12));
        assert_eq!(count, 1);
    }

    #[test]
    fn f12_twelfth_call_is_allowed() {
        let mut count = 11usize;
        assert!(bump_f12_tool_call_count(&mut count, 12));
        assert_eq!(count, 12);
    }

    #[test]
    fn f12_thirteenth_call_is_rejected_without_counting() {
        let mut count = 12usize;
        assert!(!bump_f12_tool_call_count(&mut count, 12));
        assert_eq!(count, 12);
    }

    #[test]
    fn f12_zero_budget_rejects_first_call() {
        let mut count = 0usize;
        assert!(!bump_f12_tool_call_count(&mut count, 0));
        assert_eq!(count, 0);
    }

    // ── check_loop_watchdog ───────────────────────────────────────

    #[test]
    fn watchdog_already_triggered_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("anything", &mut buf, true));
    }

    #[test]
    fn watchdog_empty_text_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("", &mut buf, false));
    }

    #[test]
    fn watchdog_single_line_returns_false() {
        let mut buf = String::new();
        // Just one line of content — no repeats.
        assert!(!check_loop_watchdog(
            "this is a single long enough line to qualify\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_four_identical_lines_fires() {
        let mut buf = String::new();
        let line = "Running cargo test on the project\n";
        // First three accumulations should not fire.
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        // Fourth occurrence trips the watchdog.
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_fuzzy_normalization_collapses_whitespace() {
        let mut buf = String::new();
        // Same phrase, different whitespace each time — must still fuzzy-match.
        assert!(!check_loop_watchdog(
            "Running cargo test now\n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            "  Running cargo test now  \n",
            &mut buf,
            false
        ));
        assert!(!check_loop_watchdog(
            "Running cargo  test  now\n",
            &mut buf,
            false
        ));
        assert!(check_loop_watchdog(
            "Running\tcargo test now\n",
            &mut buf,
            false
        ));
    }

    #[test]
    fn watchdog_short_lines_skipped() {
        // Lines whose trimmed length ≤ 15 chars don't qualify as the
        // needle, so identical short lines don't trigger.
        let mut buf = String::new();
        let short = "ok\n"; // 2 chars
        for _ in 0..10 {
            assert!(!check_loop_watchdog(short, &mut buf, false));
        }
    }

    #[test]
    fn watchdog_buffer_caps_at_10kb() {
        let mut buf = String::new();
        let big = "x".repeat(5000);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        // After two 5KB pushes the buffer is 10KB; a third triggers the
        // 10_240-byte cap and drains down to 8KB.
        check_loop_watchdog(&big, &mut buf, false);
        assert!(
            buf.len() <= 10_240,
            "buffer should self-trim, got {}",
            buf.len()
        );
    }
}
