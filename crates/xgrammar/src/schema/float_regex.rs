// SPDX-License-Identifier: AGPL-3.0-only
//
// Float range -> regex generation — port of
// `JSONSchemaConverter::{FormatFloat,GenerateFloatRangeRegex}` from
// `cpp/json_schema_converter.cc`.

use super::range_regex::{generate_pos_ge_regex, generate_sub_range_regex};

/// Format a double to at most `precision` fractional digits, trimming
/// trailing zeros. Port of `FormatFloat`.
fn format_float(value: f64, precision: i32) -> String {
    if value.is_finite()
        && value >= i64::MIN as f64
        && value < i64::MAX as f64
        && value == value.trunc()
    {
        return (value as i64).to_string();
    }
    let mut result = format!("{value:.*}", precision as usize);
    if let Some(dot) = result.find('.') {
        let trimmed_len = result.trim_end_matches('0').len();
        if trimmed_len > dot {
            result.truncate(trimmed_len);
        } else if trimmed_len == dot {
            result.truncate(dot);
        }
    }
    if result == "-0" {
        return "0".to_string();
    }
    result
}

/// Non-negative decimal magnitude `int_part + 0.<frac>` with trailing
/// zeros stripped from `frac`.
#[derive(Debug, Clone)]
struct MagBound {
    int_part: u64,
    frac: Vec<char>,
}

/// Signed decimal bound parsed from `format_float(value, precision)` so
/// sign, integer part, and fractional digits stay consistent after
/// rounding (including `-0.x` and carry-over like `0.9999999 -> 1`).
#[derive(Debug, Clone)]
struct DecimalBound {
    is_neg: bool,
    mag: MagBound,
}

impl DecimalBound {
    fn from_f64(value: f64, precision: i32) -> Self {
        let formatted = format_float(value, precision);
        let is_neg = formatted.starts_with('-');
        let unsigned = formatted.strip_prefix('-').unwrap_or(&formatted);
        let (int_str, frac) = match unsigned.find('.') {
            Some(dot) => (&unsigned[..dot], unsigned[dot + 1..].chars().collect()),
            None => (unsigned, Vec::new()),
        };
        let int_part = int_str.parse::<u64>().unwrap_or(u64::MAX);
        DecimalBound {
            is_neg: is_neg && (int_part > 0 || !frac.is_empty()),
            mag: MagBound { int_part, frac },
        }
    }
}

fn digit_range_pattern(lo: char, hi: char) -> String {
    if lo == hi {
        lo.to_string()
    } else {
        format!("[{lo}-{hi}]")
    }
}

fn next_char(c: char) -> char {
    char::from(c as u8 + 1)
}

fn prev_char(c: char) -> char {
    char::from(c as u8 - 1)
}

fn trailing_digits(max_digits: i32) -> String {
    if max_digits <= 0 {
        String::new()
    } else {
        format!("\\d{{0,{max_digits}}}")
    }
}

fn trailing_zeros(max_zeros: i32) -> String {
    if max_zeros <= 0 {
        String::new()
    } else {
        format!("0{{0,{max_zeros}}}")
    }
}

/// Emit patterns matching `{base}<f>` for all 1..=`max_len` digit strings
/// `<f>` containing at least one non-zero digit.
fn emit_nonzero_suffix(parts: &mut Vec<String>, base: &str, max_len: i32) {
    for zeros in 0..max_len {
        let lead = "0".repeat(zeros as usize);
        let tail = trailing_digits(max_len - zeros - 1);
        parts.push(format!("{base}{lead}[1-9]{tail}"));
    }
}

/// Emit patterns matching `{base}<f>` (1..=`max_len` digits) with
/// `0.<f> >= 0.<lo>`. `lo` must be non-empty and trimmed of trailing `'0'`.
fn emit_frac_suffix_ge(parts: &mut Vec<String>, base: &str, lo: &[char], max_len: i32) {
    let lo_str: String = lo.iter().collect();
    let exact_tail = trailing_digits(max_len - lo.len() as i32);
    parts.push(format!("{base}{lo_str}{exact_tail}"));

    for (i, &lc) in lo.iter().enumerate() {
        if lc < '9' {
            let pref: String = lo[..i].iter().collect();
            let dr = digit_range_pattern(next_char(lc), '9');
            let tail = trailing_digits(max_len - i as i32 - 1);
            parts.push(format!("{base}{pref}{dr}{tail}"));
        }
    }
}

/// Emit patterns matching `{base}<f>` (1..=`max_len` digits) with
/// `0.<f> <= 0.<hi>` (and `0.<f> > 0` when `!allow_zero`).
/// `hi` must be non-empty and trimmed of trailing `'0'`.
fn emit_frac_suffix_le(
    parts: &mut Vec<String>,
    base: &str,
    hi: &[char],
    max_len: i32,
    allow_zero: bool,
) {
    let hi_str: String = hi.iter().collect();
    let exact_tail = trailing_zeros(max_len - hi.len() as i32);
    parts.push(format!("{base}{hi_str}{exact_tail}"));

    // Every proper prefix of `hi` is strictly smaller than `0.<hi>`
    // because `hi` ends in a non-zero digit.
    for i in 1..hi.len() {
        let pref = &hi[..i];
        if allow_zero || pref.iter().any(|&c| c != '0') {
            let pref_str: String = pref.iter().collect();
            parts.push(format!("{base}{pref_str}"));
        }
    }

    for (i, &hc) in hi.iter().enumerate() {
        if hc > '0' {
            let pref = &hi[..i];
            let pref_str: String = pref.iter().collect();
            let rem = max_len - i as i32 - 1;
            if allow_zero || pref.iter().any(|&c| c != '0') {
                let dr = digit_range_pattern('0', prev_char(hc));
                let tail = trailing_digits(rem);
                parts.push(format!("{base}{pref_str}{dr}{tail}"));
            } else {
                if hc > '1' {
                    let dr = digit_range_pattern('1', prev_char(hc));
                    let tail = trailing_digits(rem);
                    parts.push(format!("{base}{pref_str}{dr}{tail}"));
                }
                if rem > 0 {
                    let zero_base = format!("{base}{pref_str}0");
                    emit_nonzero_suffix(parts, &zero_base, rem);
                }
            }
        }
    }
}

/// Emit patterns matching `{base}<f>` (1..=`max_len` digits) with
/// `0.<lo> <= 0.<f> <= 0.<hi>`. Both `lo` and `hi` must be non-empty,
/// trimmed of trailing `'0'`, and satisfy `0.<lo> <= 0.<hi>`.
fn emit_frac_suffix_between(
    parts: &mut Vec<String>,
    base: &str,
    lo: &[char],
    hi: &[char],
    max_len: i32,
) {
    let m = lo.len().max(hi.len());
    let mut d = 0usize;
    while d < m {
        let lc = lo.get(d).copied().unwrap_or('0');
        let hc = hi.get(d).copied().unwrap_or('0');
        if lc != hc {
            break;
        }
        d += 1;
    }

    if d == m {
        let s: String = lo.iter().collect();
        let tail = trailing_zeros(max_len - lo.len() as i32);
        parts.push(format!("{base}{s}{tail}"));
        return;
    }

    let mut common = String::with_capacity(d);
    for i in 0..d {
        common.push(lo.get(i).copied().unwrap_or('0'));
    }

    // If `lo` ends at or before `d`, every prefix `common[..j]` for
    // `lo.len() <= j <= d` equals `0.<lo>` exactly.
    if lo.len() <= d {
        for j in lo.len()..=d {
            parts.push(format!("{base}{}", &common[..j]));
        }
    }

    let lc = lo.get(d).copied().unwrap_or('0');
    let hc = hi.get(d).copied().unwrap_or('0');
    debug_assert!(lc < hc);
    let rem = max_len - d as i32 - 1;

    // Lower-boundary digit `lc`.
    let lo_ends_at_d = lo.len() <= d + 1;
    if !lo_ends_at_d {
        let sub_base = format!("{base}{common}{lc}");
        emit_frac_suffix_ge(parts, &sub_base, &lo[d + 1..], rem);
    }

    // Middle digits strictly below `hc`.
    let mid_lo = if lo_ends_at_d { lc } else { next_char(lc) };
    let mid_hi = prev_char(hc);
    if mid_lo <= mid_hi {
        let dr = digit_range_pattern(mid_lo, mid_hi);
        let tail = trailing_digits(rem);
        parts.push(format!("{base}{common}{dr}{tail}"));
    }

    // Upper-boundary digit `hc`.
    if hi.len() <= d + 1 {
        let tail = trailing_zeros(rem);
        parts.push(format!("{base}{common}{hc}{tail}"));
    } else {
        parts.push(format!("{base}{common}{hc}"));
        let sub_base = format!("{base}{common}{hc}");
        emit_frac_suffix_le(parts, &sub_base, &hi[d + 1..], rem, true);
    }
}

/// Emit patterns for one sign half (`sign` is `""` for non-negative or
/// `"-"` for strictly negative) covering magnitudes in `[lo, hi]`.
/// When `lo` is `None`, matches strictly positive magnitudes `> 0`.
fn emit_magnitude_range(
    parts: &mut Vec<String>,
    sign: &str,
    lo: Option<&MagBound>,
    hi: Option<&MagBound>,
    p: i32,
) {
    match (lo, hi) {
        (None, None) => {
            emit_nonzero_suffix(parts, &format!("{sign}0\\."), p);
            parts.push(format!("{sign}[1-9]\\d*(\\.\\d{{1,{p}}})?"));
        }
        (Some(lo), None) => {
            if lo.frac.is_empty() {
                if lo.int_part == 0 {
                    parts.push(format!("{sign}0(\\.\\d{{1,{p}}})?"));
                    parts.push(format!("{sign}[1-9]\\d*(\\.\\d{{1,{p}}})?"));
                } else {
                    let ir = generate_pos_ge_regex(lo.int_part);
                    parts.push(format!("{sign}{ir}(\\.\\d{{1,{p}}})?"));
                }
            } else {
                let lo_base = format!("{sign}{}\\.", lo.int_part);
                emit_frac_suffix_ge(parts, &lo_base, &lo.frac, p);
                if lo.int_part < u64::MAX {
                    let ir = generate_pos_ge_regex(lo.int_part + 1);
                    parts.push(format!("{sign}{ir}(\\.\\d{{1,{p}}})?"));
                }
            }
        }
        (None, Some(hi)) => {
            if hi.int_part == 0 {
                if !hi.frac.is_empty() {
                    emit_frac_suffix_le(parts, &format!("{sign}0\\."), &hi.frac, p, false);
                }
            } else {
                emit_nonzero_suffix(parts, &format!("{sign}0\\."), p);
                if hi.int_part >= 2 {
                    let ir = generate_sub_range_regex(1, hi.int_part - 1);
                    parts.push(format!("{sign}{ir}(\\.\\d{{1,{p}}})?"));
                }
                let hi_prefix = format!("{sign}{}", hi.int_part);
                if hi.frac.is_empty() {
                    parts.push(format!("{hi_prefix}(\\.0{{1,{p}}})?"));
                } else {
                    parts.push(hi_prefix.clone());
                    emit_frac_suffix_le(parts, &format!("{hi_prefix}\\."), &hi.frac, p, true);
                }
            }
        }
        (Some(lo), Some(hi)) => {
            if lo.int_part == hi.int_part {
                let prefix = format!("{sign}{}", lo.int_part);
                match (lo.frac.is_empty(), hi.frac.is_empty()) {
                    (true, true) => {
                        parts.push(format!("{prefix}(\\.0{{1,{p}}})?"));
                    }
                    (true, false) => {
                        parts.push(prefix.clone());
                        emit_frac_suffix_le(parts, &format!("{prefix}\\."), &hi.frac, p, true);
                    }
                    (false, false) => {
                        emit_frac_suffix_between(
                            parts,
                            &format!("{prefix}\\."),
                            &lo.frac,
                            &hi.frac,
                            p,
                        );
                    }
                    (false, true) => {
                        // Unreachable when `lo <= hi`.
                    }
                }
            } else if lo.int_part < hi.int_part {
                if !lo.frac.is_empty() {
                    let lo_base = format!("{sign}{}\\.", lo.int_part);
                    emit_frac_suffix_ge(parts, &lo_base, &lo.frac, p);
                }

                let mid_lo = if lo.frac.is_empty() {
                    lo.int_part
                } else {
                    lo.int_part + 1
                };
                let mid_hi = hi.int_part - 1;
                if mid_lo <= mid_hi {
                    if mid_lo == 0 {
                        parts.push(format!("{sign}0(\\.\\d{{1,{p}}})?"));
                        if mid_hi >= 1 {
                            let ir = generate_sub_range_regex(1, mid_hi);
                            parts.push(format!("{sign}{ir}(\\.\\d{{1,{p}}})?"));
                        }
                    } else {
                        let ir = generate_sub_range_regex(mid_lo, mid_hi);
                        parts.push(format!("{sign}{ir}(\\.\\d{{1,{p}}})?"));
                    }
                }

                let hi_prefix = format!("{sign}{}", hi.int_part);
                if hi.frac.is_empty() {
                    parts.push(format!("{hi_prefix}(\\.0{{1,{p}}})?"));
                } else {
                    parts.push(hi_prefix.clone());
                    emit_frac_suffix_le(parts, &format!("{hi_prefix}\\."), &hi.frac, p, true);
                }
            }
        }
    }
}

/// Generate a regex matching floats in `[start, end]` inclusive, each
/// bound `None` meaning unbounded, with at most `precision`
/// fractional digits. Port of `GenerateFloatRangeRegex`.
pub fn generate_float_range_regex(start: Option<f64>, end: Option<f64>, precision: i32) -> String {
    if start.is_some_and(f64::is_nan) || end.is_some_and(f64::is_nan) {
        return "^()$".to_string();
    }
    if start == Some(f64::INFINITY) || end == Some(f64::NEG_INFINITY) {
        return "^()$".to_string();
    }
    let start = start.filter(|v| *v != f64::NEG_INFINITY);
    let end = end.filter(|v| *v != f64::INFINITY);

    if let (Some(s), Some(e)) = (start, end)
        && s > e
    {
        return "^()$".to_string();
    }
    let p = precision.max(1);
    if start.is_none() && end.is_none() {
        return format!("^-?\\d+(\\.\\d{{1,{p}}})?$");
    }

    let sb = start.map(|s| DecimalBound::from_f64(s, p));
    let eb = end.map(|e| DecimalBound::from_f64(e, p));
    let zero = MagBound {
        int_part: 0,
        frac: Vec::new(),
    };

    let mut parts: Vec<String> = Vec::new();

    // Negative half `(-inf, 0)`: active if lower bound is unbounded or negative.
    let has_neg = sb.as_ref().is_none_or(|b| b.is_neg);
    if has_neg {
        let neg_lo = eb.as_ref().filter(|b| b.is_neg).map(|b| &b.mag);
        let neg_hi = sb.as_ref().filter(|b| b.is_neg).map(|b| &b.mag);
        emit_magnitude_range(&mut parts, "-", neg_lo, neg_hi, p);
    }

    // Non-negative half `[0, +inf)`: active if upper bound is unbounded or >= 0.
    let has_nonneg = eb.as_ref().is_none_or(|b| !b.is_neg);
    if has_nonneg {
        let pos_lo = match &sb {
            Some(b) if !b.is_neg => &b.mag,
            _ => &zero,
        };
        let pos_hi = eb.as_ref().filter(|b| !b.is_neg).map(|b| &b.mag);
        emit_magnitude_range(&mut parts, "", Some(pos_lo), pos_hi, p);
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
        let g = regex_to_grammar(regex).expect("generated float regex must parse");
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
    fn unbounded_float() {
        assert_eq!(
            generate_float_range_regex(None, None, 6),
            r"^-?\d+(\.\d{1,6})?$"
        );
    }

    #[test]
    fn inverted_range_is_empty() {
        assert_eq!(generate_float_range_regex(Some(5.0), Some(1.0), 6), "^()$");
    }

    #[test]
    fn whole_number_format() {
        assert_eq!(format_float(3.0, 6), "3");
        assert_eq!(format_float(3.5, 6), "3.5");
        assert_eq!(format_float(3.120000, 6), "3.12");
    }

    #[test]
    fn bounded_produces_anchored_regex() {
        let r = generate_float_range_regex(Some(1.0), Some(10.0), 6);
        assert!(r.starts_with("^(") && r.ends_with(")$"));
    }

    #[test]
    fn float_boundary_dot_is_escaped_not_wildcard() {
        // Regression for upstream c4cf39f (#642): the decimal point of a
        // float boundary must be a literal `\.`, never an unescaped `.`
        // wildcard that would accept `0,5` etc.
        let r = generate_float_range_regex(Some(0.5), None, 6);
        assert!(r.contains(r"0\.5"), "expected escaped boundary in {r}");
        // No bare `.` may appear except as part of an escaped `\.`.
        let bytes: Vec<char> = r.chars().collect();
        for (i, &c) in bytes.iter().enumerate() {
            if c == '.' {
                assert!(i > 0 && bytes[i - 1] == '\\', "unescaped dot in {r}");
            }
        }
    }

    #[test]
    fn float_boundary_dot_escaped_both_bounds() {
        let r = generate_float_range_regex(Some(0.25), Some(0.75), 6);
        assert!(r.contains(r"0\.25"), "expected escaped start in {r}");
        assert!(r.contains(r"0\.75"), "expected escaped end in {r}");
    }

    #[test]
    fn same_integer_part_accepts_interior_floats() {
        let r = generate_float_range_regex(Some(0.25), Some(0.75), 6);
        for ok in [
            "0.25", "0.250", "0.251", "0.3", "0.5", "0.7", "0.749999", "0.75", "0.750",
        ] {
            assert!(regex_accepts(&r, ok), "{r} should accept {ok}");
        }
        for bad in ["0", "0.2", "0.249999", "0.750001", "0.8", "1", "-0.5"] {
            assert!(!regex_accepts(&r, bad), "{r} should reject {bad}");
        }

        let r_neg = generate_float_range_regex(Some(-0.75), Some(-0.25), 6);
        for ok in ["-0.75", "-0.7", "-0.5", "-0.3", "-0.25"] {
            assert!(regex_accepts(&r_neg, ok), "{r_neg} should accept {ok}");
        }
        for bad in ["-0.750001", "-0.249999", "0", "0.5"] {
            assert!(!regex_accepts(&r_neg, bad), "{r_neg} should reject {bad}");
        }
    }

    #[test]
    fn exact_integer_endpoints_do_not_overflow() {
        let r_pos = generate_float_range_regex(Some(1.0), Some(5.0), 6);
        for ok in ["1", "1.0", "1.5", "4.999999", "5", "5.0"] {
            assert!(regex_accepts(&r_pos, ok), "{r_pos} should accept {ok}");
        }
        for bad in ["0.999999", "5.000001", "5.5", "5.999999", "-1"] {
            assert!(!regex_accepts(&r_pos, bad), "{r_pos} should reject {bad}");
        }

        let r_neg = generate_float_range_regex(Some(-5.0), Some(-1.0), 6);
        for ok in ["-5", "-5.0", "-4.999999", "-2.5", "-1.0", "-1"] {
            assert!(regex_accepts(&r_neg, ok), "{r_neg} should accept {ok}");
        }
        for bad in ["-5.999999", "-5.000001", "-0.999999", "0", "1"] {
            assert!(!regex_accepts(&r_neg, bad), "{r_neg} should reject {bad}");
        }
    }

    #[test]
    fn negative_fractional_bounds_and_zero_crossing() {
        let r_ge = generate_float_range_regex(Some(-1.5), None, 6);
        for ok in [
            "-1.5",
            "-1.499999",
            "-1",
            "-1.0",
            "-0.5",
            "-0.000001",
            "0",
            "0.5",
            "10",
        ] {
            assert!(regex_accepts(&r_ge, ok), "{r_ge} should accept {ok}");
        }
        for bad in ["-1.500001", "-1.6", "-1.999999", "-2"] {
            assert!(!regex_accepts(&r_ge, bad), "{r_ge} should reject {bad}");
        }

        let r_le = generate_float_range_regex(None, Some(-1.5), 6);
        for ok in ["-1.5", "-1.500001", "-1.6", "-2", "-2.5", "-100"] {
            assert!(regex_accepts(&r_le, ok), "{r_le} should accept {ok}");
        }
        for bad in ["-1.499999", "-1", "-0.5", "0", "1.5"] {
            assert!(!regex_accepts(&r_le, bad), "{r_le} should reject {bad}");
        }

        let r_zero = generate_float_range_regex(Some(-0.5), Some(0.5), 6);
        for ok in ["-0.5", "-0.25", "-0.000001", "0", "0.0", "0.25", "0.5"] {
            assert!(regex_accepts(&r_zero, ok), "{r_zero} should accept {ok}");
        }
        for bad in ["-0.500001", "-1", "0.500001", "1"] {
            assert!(!regex_accepts(&r_zero, bad), "{r_zero} should reject {bad}");
        }
    }
}
