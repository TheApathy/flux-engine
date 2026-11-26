// SPDX-License-Identifier: AGPL-3.0-only
//
// Integer range -> regex generation — port of
// `JSONSchemaConverter::{MakePatternForDigitRange,GenerateNumberPatterns,
// GenerateSubRangeRegex,GenerateRangeRegex}` from
// `cpp/json_schema_converter.cc`.
//
// Produces a regex (anchored with `^...$`) matching all integers in a
// closed `[start, end]` range, where `None` means unbounded.

/// A digit-range pattern for one fixed-length slot: either a literal
/// digit or a `[a-b]` class, optionally followed by `\d{n}`.
fn make_pattern_for_digit_range(start: char, end: char, remaining: i32) -> String {
    let mut out = String::new();
    if start == end {
        out.push(start);
    } else {
        out.push('[');
        out.push(start);
        out.push('-');
        out.push(end);
        out.push(']');
    }
    if remaining > 0 {
        out.push_str(&format!("\\d{{{remaining}}}"));
    }
    out
}

/// Generate the alternation of fixed-length digit patterns covering
/// the positive range `[lower, upper]` (`1 <= lower <= upper`).
/// Port of `GenerateNumberPatterns`.
fn generate_number_patterns(lower: u64, upper: u64) -> Vec<String> {
    debug_assert!(lower >= 1 && lower <= upper);
    let mut patterns = Vec::new();
    let lower_len = lower.to_string().len() as u32;
    let upper_len = upper.to_string().len() as u32;

    for len in lower_len..=upper_len {
        let digit_min = 10u128.pow(len - 1) as u64;
        // 10^20 - 1 exceeds u64::MAX, so cap at u64::MAX (upper <= u64::MAX).
        let digit_max = (10u128.pow(len) - 1).min(u64::MAX as u128) as u64;
        let start = if len == lower_len { lower } else { digit_min };
        let end = if len == upper_len { upper } else { digit_max };
        push_fixed_len_range(&mut patterns, start, end);
    }
    patterns
}

/// Append exact fixed-length digit patterns covering `[start, end]`,
/// where `start` and `end` have the same decimal digit count and
/// `start <= end`.
fn push_fixed_len_range(patterns: &mut Vec<String>, start: u64, end: u64) {
    let start_str = start.to_string();
    let end_str = end.to_string();
    let sb: Vec<char> = start_str.chars().collect();
    let eb: Vec<char> = end_str.chars().collect();
    let len = sb.len();
    debug_assert_eq!(eb.len(), len);

    let mut common_len = 0usize;
    while common_len < len && sb[common_len] == eb[common_len] {
        common_len += 1;
    }
    if common_len == len {
        patterns.push(start_str);
        return;
    }

    let common: String = sb[..common_len].iter().collect();
    if common_len == len - 1 {
        patterns.push(format!(
            "{common}{}",
            make_pattern_for_digit_range(sb[common_len], eb[common_len], 0)
        ));
        return;
    }

    // Operate on the suffixes starting at the first differing digit (`s[0] < e[0]`).
    let s = &sb[common_len..];
    let e = &eb[common_len..];
    let rem = s.len();
    let s_tail_all_zeros = s[1..].iter().all(|&c| c == '0');
    let e_tail_all_nines = e[1..].iter().all(|&c| c == '9');

    // Lower boundary tail when `s[1..]` is not all zeros.
    if !s_tail_all_zeros {
        patterns.push(start_str);
        for (i, &sc) in s.iter().enumerate().skip(1) {
            if sc < '9' {
                let pref: String = s[..i].iter().collect();
                patterns.push(format!(
                    "{common}{pref}{}",
                    make_pattern_for_digit_range(next_char(sc), '9', (rem - i - 1) as i32)
                ));
            }
        }
    }

    // Full-block digits strictly between (or including full-tail endpoints) `s[0]` and `e[0]`.
    let mid_lo = if s_tail_all_zeros {
        s[0]
    } else {
        next_char(s[0])
    };
    let mid_hi = if e_tail_all_nines {
        e[0]
    } else {
        prev_char(e[0])
    };
    if mid_lo <= mid_hi {
        patterns.push(format!(
            "{common}{}",
            make_pattern_for_digit_range(mid_lo, mid_hi, (rem - 1) as i32)
        ));
    }

    // Upper boundary tail when `e[1..]` is not all nines.
    if !e_tail_all_nines {
        for (i, &ec) in e.iter().enumerate().skip(1) {
            if ec > '0' {
                let pref: String = e[..i].iter().collect();
                patterns.push(format!(
                    "{common}{pref}{}",
                    make_pattern_for_digit_range('0', prev_char(ec), (rem - i - 1) as i32)
                ));
            }
        }
        patterns.push(end_str);
    }
}

/// Append same-length magnitude patterns strictly greater than `digits`,
/// each prefixed with `sign` (`""` for positive, `"-"` for negative).
fn push_mag_above(parts: &mut Vec<String>, sign: &str, digits: &[char], len: i32) {
    for (i, &dc) in digits.iter().enumerate() {
        if dc < '9' {
            let pref: String = digits[..i].iter().collect();
            parts.push(format!(
                "{sign}{pref}{}",
                make_pattern_for_digit_range(next_char(dc), '9', len - i as i32 - 1)
            ));
        }
    }
}

fn next_char(c: char) -> char {
    char::from(c as u8 + 1)
}
fn prev_char(c: char) -> char {
    char::from(c as u8 - 1)
}

/// Generate the parenthesized alternation for the positive range
/// `[lower, upper]` (`1 <= lower <= upper`). Port of `GenerateSubRangeRegex`.
pub(super) fn generate_sub_range_regex(lower: u64, upper: u64) -> String {
    let patterns = generate_number_patterns(lower, upper);
    format!("({})", patterns.join("|"))
}

/// Generate an unanchored regex fragment matching all positive integers
/// `>= lower` (`lower >= 1`). Used by `float_regex`.
pub(super) fn generate_pos_ge_regex(lower: u64) -> String {
    debug_assert!(lower >= 1);
    if lower == 1 {
        return "[1-9]\\d*".to_string();
    }
    let start_str = lower.to_string();
    let sb: Vec<char> = start_str.chars().collect();
    let len = sb.len() as i32;
    let mut parts = Vec::new();
    if len == 1 {
        parts.push(make_pattern_for_digit_range(sb[0], '9', 0));
        parts.push("[1-9]\\d+".to_string());
    } else {
        parts.push(start_str);
        push_mag_above(&mut parts, "", &sb, len);
        parts.push(format!("[1-9]\\d{{{len},}}"));
    }
    format!("({})", parts.join("|"))
}

/// Generate a regex matching integers in `[start, end]` (inclusive),
/// each bound `None` meaning unbounded. Port of `GenerateRangeRegex`.
pub fn generate_range_regex(start: Option<i64>, end: Option<i64>) -> String {
    let mut parts: Vec<String> = Vec::new();

    match (start, end) {
        (None, None) => return "^-?\\d+$".to_string(),
        (Some(s), None) => {
            if s <= 0 {
                if s < 0 {
                    parts.push(format!(
                        "-{}",
                        generate_sub_range_regex(1, s.unsigned_abs())
                    ));
                }
                parts.push("0".to_string());
                parts.push("[1-9]\\d*".to_string());
            } else {
                let start_str = s.to_string();
                let sb: Vec<char> = start_str.chars().collect();
                let len = sb.len() as i32;
                if len == 1 {
                    parts.push(make_pattern_for_digit_range(sb[0], '9', 0));
                    parts.push("[1-9]\\d+".to_string());
                } else {
                    parts.push(start_str);
                    push_mag_above(&mut parts, "", &sb, len);
                    parts.push(format!("[1-9]\\d{{{len},}}"));
                }
            }
        }
        (None, Some(e)) => {
            if e >= 0 {
                parts.push("-[1-9]\\d*".to_string());
                parts.push("0".to_string());
                if e > 0 {
                    parts.push(generate_sub_range_regex(1, e as u64));
                }
            } else {
                let end_str = e.unsigned_abs().to_string();
                let eb: Vec<char> = end_str.chars().collect();
                let len = eb.len() as i32;
                if len == 1 {
                    parts.push(format!("-{}", make_pattern_for_digit_range(eb[0], '9', 0)));
                    parts.push("-[1-9]\\d+".to_string());
                } else {
                    parts.push(format!("-{end_str}"));
                    push_mag_above(&mut parts, "-", &eb, len);
                    parts.push(format!("-[1-9]\\d{{{len},}}"));
                }
            }
        }
        (Some(s), Some(e)) => {
            if s > e {
                return "^()$".to_string();
            }
            if s < 0 {
                let neg_end = (-1i64).min(e);
                parts.push(format!(
                    "-{}",
                    generate_sub_range_regex(neg_end.unsigned_abs(), s.unsigned_abs())
                ));
            }
            if s <= 0 && e >= 0 {
                parts.push("0".to_string());
            }
            if e > 0 {
                let pos_start = 1i64.max(s);
                parts.push(generate_sub_range_regex(pos_start as u64, e as u64));
            }
        }
    }

    format!("^({})$", parts.join("|"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::earley::EarleyParser;
    use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
    use crate::regex::regex_to_grammar;

    fn regex_accepts(regex: &str, input: &str) -> bool {
        let g = regex_to_grammar(regex).expect("generated range regex must parse");
        let opt = Arc::new(GrammarOptimizer::apply(GrammarNormalizer::apply(g)));
        let mut p = EarleyParser::from_grammar(opt);
        for b in input.bytes() {
            if !p.advance(b) {
                return false;
            }
        }
        p.is_completed()
    }

    #[test]
    fn matches_cpp_examples() {
        assert_eq!(generate_range_regex(Some(12), Some(16)), r"^((1[2-6]))$");
        assert_eq!(generate_range_regex(Some(1), Some(10)), r"^(([1-9]|10))$");
        assert_eq!(
            generate_range_regex(Some(-5), Some(10)),
            r"^(-([1-5])|0|([1-9]|10))$"
        );
        assert_eq!(generate_range_regex(Some(-15), Some(-10)), r"^(-(1[0-5]))$");
        assert_eq!(generate_range_regex(None, None), r"^-?\d+$");
        assert_eq!(generate_range_regex(Some(5), None), r"^([5-9]|[1-9]\d+)$");
        assert_eq!(generate_range_regex(None, Some(0)), r"^(-[1-9]\d*|0)$");
        assert_eq!(generate_range_regex(Some(5), Some(5)), r"^((5))$");
        assert_eq!(
            generate_range_regex(Some(-10), Some(0)),
            r"^(-([1-9]|10)|0)$"
        );
        assert_eq!(generate_range_regex(Some(0), Some(10)), r"^(0|([1-9]|10))$");
    }

    #[test]
    fn matches_multi_length_examples() {
        assert_eq!(
            generate_range_regex(Some(1), Some(9999)),
            r"^(([1-9]|[1-9]\d{1}|[1-9]\d{2}|[1-9]\d{3}))$"
        );
        assert_eq!(
            generate_range_regex(Some(s_int()), None),
            r"^([5-9]|[1-9]\d+)$"
        );
    }

    #[test]
    fn negative_lower_bound_and_i64_min_are_valid() {
        let r5 = generate_range_regex(Some(-5), None);
        for ok in ["-5", "-1", "0", "1", "42"] {
            assert!(regex_accepts(&r5, ok), "{r5} should accept {ok}");
        }
        for bad in ["-6", "-10", "-"] {
            assert!(!regex_accepts(&r5, bad), "{r5} should reject {bad}");
        }

        let r10 = generate_range_regex(Some(-10), None);
        for ok in ["-10", "-5", "0", "99"] {
            assert!(regex_accepts(&r10, ok), "{r10} should accept {ok}");
        }
        for bad in ["-11", "-100", "-"] {
            assert!(!regex_accepts(&r10, bad), "{r10} should reject {bad}");
        }

        // i64::MIN must never overflow `abs()` / unary negation.
        let rmin = generate_range_regex(Some(i64::MIN), None);
        assert!(regex_accepts(&rmin, &i64::MIN.to_string()));
        assert!(regex_accepts(&rmin, "0"));
        let rmax_neg = generate_range_regex(None, Some(i64::MIN));
        assert!(regex_accepts(&rmax_neg, &i64::MIN.to_string()));
        assert!(!regex_accepts(&rmax_neg, &(i64::MIN + 1).to_string()));
        let rfull = generate_range_regex(Some(i64::MIN), Some(i64::MAX));
        assert!(regex_accepts(&rfull, &i64::MIN.to_string()));
        assert!(regex_accepts(&rfull, &i64::MAX.to_string()));
    }

    #[test]
    fn single_digit_and_negative_upper_bounds_are_exact() {
        let r_ge5 = generate_range_regex(Some(5), None);
        for ok in ["5", "9", "10", "100"] {
            assert!(regex_accepts(&r_ge5, ok), "{r_ge5} should accept {ok}");
        }
        for bad in ["-5", "0", "1", "4"] {
            assert!(!regex_accepts(&r_ge5, bad), "{r_ge5} should reject {bad}");
        }

        let r_le_neg5 = generate_range_regex(None, Some(-5));
        for ok in ["-5", "-9", "-10", "-100"] {
            assert!(
                regex_accepts(&r_le_neg5, ok),
                "{r_le_neg5} should accept {ok}"
            );
        }
        for bad in ["-4", "-1", "0", "5"] {
            assert!(
                !regex_accepts(&r_le_neg5, bad),
                "{r_le_neg5} should reject {bad}"
            );
        }

        let r_le_neg25 = generate_range_regex(None, Some(-25));
        for ok in ["-25", "-26", "-30", "-99", "-100", "-1000"] {
            assert!(
                regex_accepts(&r_le_neg25, ok),
                "{r_le_neg25} should accept {ok}"
            );
        }
        for bad in ["-24", "-20", "-10", "-1", "0", "25"] {
            assert!(
                !regex_accepts(&r_le_neg25, bad),
                "{r_le_neg25} should reject {bad}"
            );
        }
    }

    #[test]
    fn shared_prefix_and_19_digit_ranges_are_exact() {
        let r1 = generate_range_regex(Some(123), Some(145));
        for n in 115..=155 {
            let s = n.to_string();
            let want = (123..=145).contains(&n);
            assert_eq!(regex_accepts(&r1, &s), want, "{r1} on {s}");
        }

        let r2 = generate_range_regex(Some(1234), Some(1567));
        for ok in [
            "1234", "1235", "1299", "1300", "1499", "1500", "1566", "1567",
        ] {
            assert!(regex_accepts(&r2, ok), "{r2} should accept {ok}");
        }
        for bad in ["1000", "1200", "1233", "1568", "1600", "1999"] {
            assert!(!regex_accepts(&r2, bad), "{r2} should reject {bad}");
        }

        // 19-digit bounds must not overflow `10^19`.
        let r19 = generate_range_regex(Some(1_000_000_000_000_000_000), Some(i64::MAX));
        assert!(regex_accepts(&r19, "1000000000000000000"));
        assert!(regex_accepts(&r19, &i64::MAX.to_string()));
        assert!(!regex_accepts(&r19, "999999999999999999"));
    }

    fn s_int() -> i64 {
        5
    }
}
