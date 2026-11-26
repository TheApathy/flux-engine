// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking-nudge tests: detector (including false-positive cases), the
//! injection state machine, the plain decode path with a mock model, the
//! think-spec accept walk, the response counter and default-off identity.

use super::*;
use crate::scheduler::test_fixtures;
use crate::scheduler::{ResponseSink, process_decode_logits};
use std::collections::HashMap;

// ── Synthetic tokenizer ─────────────────────────────────────────────────────

/// `"\n"` and `"\n\n"` have fixed ids; every other piece gets an id from 100.
const NL: u32 = 1;
const PARA: u32 = 2;

/// A word-level stand-in for the BPE tokenizer: words, single punctuation
/// marks, single digits (like Qwen) and newline runs are tokens; a space
/// attaches to the next piece (like BPE's `Ġ`).
struct WordTok {
    ids: HashMap<String, u32>,
}

impl WordTok {
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
        }
    }

    fn id(&mut self, piece: &str) -> u32 {
        let next = 100 + self.ids.len() as u32;
        *self.ids.entry(piece.to_string()).or_insert(next)
    }

    fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut word = String::new();
        let mut space = false;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_alphabetic() || c == '_' {
                if word.is_empty() && space {
                    word.push(' ');
                    space = false;
                }
                word.push(c);
                i += 1;
                continue;
            }
            if !word.is_empty() {
                out.push(self.id(&word));
                word.clear();
            }
            if c == '\n' {
                space = false;
                if chars.get(i + 1) == Some(&'\n') {
                    out.push(PARA);
                    while chars.get(i) == Some(&'\n') {
                        i += 1;
                    }
                } else {
                    out.push(NL);
                    i += 1;
                }
                continue;
            }
            if c == ' ' {
                space = true;
            } else {
                let piece = if space {
                    format!(" {c}")
                } else {
                    c.to_string()
                };
                space = false;
                out.push(self.id(&piece));
            }
            i += 1;
        }
        if !word.is_empty() {
            out.push(self.id(&word));
        }
        out
    }
}

fn class_of(id: u32) -> NewlineClass {
    match id {
        NL => NewlineClass::Newline,
        PARA => NewlineClass::EndsParagraph,
        _ => NewlineClass::Other,
    }
}

/// Deterministic, essentially non-repeating "reasoning" prose: words drawn
/// from a 4000-word vocabulary, a paragraph break every 60 tokens.
fn novel_prose(tok: &mut WordTok, n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let w = (state >> 33) % 4000;
        out.push(tok.id(&format!(" w{w}")));
        if out.len() % 60 == 59 {
            out.push(PARA);
        }
    }
    out.truncate(n);
    out
}

/// Feed `ids` to a fresh-or-given watch; returns the observed count at each
/// point where a nudge was due (and marks it fired).
fn run_watch(w: &mut ThinkWatch, ids: &[u32]) -> Vec<(u32, u8)> {
    let mut fires = Vec::new();
    for &id in ids {
        w.observe(id, class_of(id));
        let d = w.due();
        if d > 0 {
            fires.push((w.observed(), d));
            w.fired();
        }
    }
    fires
}

const CIRCLING: &str = "Wait, let me recompute the total. The first part gives 12 and the \
second part gives 30, so the sum is 42. But the statement says the answer should be odd, \
so I must have made an error somewhere. Let me recompute the total again from the start.\n\n";

// ── Policy and settings ─────────────────────────────────────────────────────

#[test]
fn default_policy_matches_the_reference_served_values() {
    let p = ThinkNudgePolicy::default();
    assert_eq!(
        (p.min_tokens, p.window, p.rate, p.ngram, p.max_nudges),
        (3072, 1024, 0.25, 8, 2)
    );
    assert_eq!((p.min_gap, p.boundary_wait), (4096, 256));
    p.validate().unwrap();
}

#[test]
fn policy_validation_rejects_out_of_range_values() {
    let bad = [
        ThinkNudgePolicy {
            window: 8,
            ..Default::default()
        },
        ThinkNudgePolicy {
            rate: 0.0,
            ..Default::default()
        },
        ThinkNudgePolicy {
            rate: 1.5,
            ..Default::default()
        },
        ThinkNudgePolicy {
            rate: f64::NAN,
            ..Default::default()
        },
        ThinkNudgePolicy {
            ngram: 1,
            ..Default::default()
        },
        ThinkNudgePolicy {
            ngram: 33,
            ..Default::default()
        },
        ThinkNudgePolicy {
            max_nudges: 3,
            ..Default::default()
        },
    ];
    for p in bad {
        assert!(p.validate().is_err(), "{p:?}");
    }
}

fn settings(pairs: &[(&str, &str)]) -> Result<Option<ThinkNudgePolicy>> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    parse_settings(|k| Ok(map.get(k).cloned()))
}

#[test]
fn settings_default_off_and_parse_strictly() {
    assert_eq!(settings(&[]).unwrap(), None, "off unless asked for");
    assert_eq!(settings(&[("THINK_NUDGE", "off")]).unwrap(), None);
    // Tuning keys are inert while the nudge is off.
    assert_eq!(
        settings(&[("THINK_NUDGE", "0"), ("THINK_NUDGE_RATE", "x")]).unwrap(),
        None
    );
    assert_eq!(
        settings(&[("THINK_NUDGE", "on")]).unwrap(),
        Some(ThinkNudgePolicy::default())
    );
    let tuned = settings(&[
        ("THINK_NUDGE", "1"),
        ("THINK_NUDGE_RATE", "0.3"),
        ("THINK_NUDGE_MIN_TOKENS", "4096"),
        ("THINK_NUDGE_MAX", "1"),
    ])
    .unwrap()
    .unwrap();
    assert_eq!(
        (tuned.rate, tuned.min_tokens, tuned.max_nudges),
        (0.3, 4096, 1)
    );
    for bad in [
        &[("THINK_NUDGE", "yes")][..],
        &[("THINK_NUDGE", "on"), ("THINK_NUDGE_RATE", "1.2")],
        &[("THINK_NUDGE", "on"), ("THINK_NUDGE_RATE", "abc")],
        &[("THINK_NUDGE", "on"), ("THINK_NUDGE_MIN_TOKENS", "-1")],
        &[("THINK_NUDGE", "on"), ("THINK_NUDGE_MAX", "3")],
    ] {
        assert!(settings(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn only_the_two_qualified_models_are_supported() {
    assert!(model_supported("qwen3.8-flash-next"));
    assert!(model_supported("qwen3.8-27b"));
    for other in [
        "qwen3.6-27b",
        "glm5.3-flash",
        "gemma-4-31b",
        "deepseek-v4.1",
        "",
    ] {
        assert!(!model_supported(other), "{other}");
    }
}

#[test]
fn newline_classes_follow_the_text_ending() {
    assert_eq!(NewlineClass::of(""), NewlineClass::Empty);
    assert_eq!(NewlineClass::of("\n"), NewlineClass::Newline);
    assert_eq!(NewlineClass::of(".\n"), NewlineClass::EndsNewline);
    assert_eq!(NewlineClass::of("\n\n"), NewlineClass::EndsParagraph);
    assert_eq!(NewlineClass::of(".\n\n"), NewlineClass::EndsParagraph);
    assert_eq!(NewlineClass::of("\n\nThe"), NewlineClass::Other);
}

// ── Detector ────────────────────────────────────────────────────────────────

#[test]
fn novel_reasoning_never_triggers() {
    let mut tok = WordTok::new();
    let ids = novel_prose(&mut tok, 12_000, 7);
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    assert!(run_watch(&mut w, &ids).is_empty());
    assert!(w.window_rate() < 0.05, "rate {}", w.window_rate());
}

#[test]
fn circling_after_long_thinking_triggers_at_a_paragraph_break() {
    let mut tok = WordTok::new();
    let mut ids = novel_prose(&mut tok, 3500, 1);
    ids.push(PARA);
    let start = ids.len() as u32;
    let lap = tok.encode(CIRCLING);
    for _ in 0..40 {
        ids.extend_from_slice(&lap);
    }
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    let fires = run_watch(&mut w, &ids);
    assert!(!fires.is_empty(), "a loop of {} laps must trigger", 40);
    let (at, which) = fires[0];
    assert_eq!(which, 1);
    // The first lap is new text; 256 self-copies (25% of 1024) follow within
    // the next ~256 tokens, then the wait for a paragraph break (<= one lap).
    assert!(
        at > start && at < start + 256 + 2 * lap.len() as u32 + 8,
        "at {at}"
    );
    assert_eq!(
        ids[at as usize - 1],
        PARA,
        "lands right after a paragraph break"
    );
}

#[test]
fn no_trigger_before_min_tokens_even_when_fully_repetitive() {
    let mut tok = WordTok::new();
    let lap = tok.encode(CIRCLING);
    let ids: Vec<u32> = lap.iter().copied().cycle().take(3000).collect();
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    assert!(run_watch(&mut w, &ids).is_empty(), "3000 < min_tokens 3072");
    assert!(w.window_rate() > 0.9);
    // The first paragraph break past 3072 then fires it.
    let more: Vec<u32> = lap
        .iter()
        .copied()
        .cycle()
        .skip(3000 % lap.len())
        .take(400)
        .collect();
    let fires = run_watch(&mut w, &more);
    assert_eq!(fires.len(), 1);
    assert!(fires[0].0 >= 3072 && fires[0].0 <= 3072 + lap.len() as u32);
}

#[test]
fn a_due_nudge_without_paragraph_break_waits_boundary_wait_tokens() {
    // One long looping line: no newline ever.
    let mut tok = WordTok::new();
    let mut ids = novel_prose(&mut tok, 3100, 3);
    ids.retain(|&t| t != PARA);
    let lap: Vec<u32> = tok
        .encode(CIRCLING)
        .into_iter()
        .filter(|&t| t != PARA)
        .collect();
    for _ in 0..30 {
        ids.extend_from_slice(&lap);
    }
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    let mut armed_at = None;
    for (i, &id) in ids.iter().enumerate() {
        w.observe(id, class_of(id));
        if armed_at.is_none() && w.armed_at.is_some() {
            armed_at = Some(i as u32 + 1);
        }
        if w.due() > 0 {
            assert!(!w.at_paragraph());
            assert_eq!(w.observed() - armed_at.unwrap(), 256);
            return;
        }
    }
    panic!("never due");
}

#[test]
fn second_nudge_needs_the_gap_and_respects_the_per_turn_cap() {
    let mut tok = WordTok::new();
    let lap = tok.encode(CIRCLING);
    let mut ids = novel_prose(&mut tok, 3200, 9);
    for _ in 0..400 {
        ids.extend_from_slice(&lap);
    }
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    let fires = run_watch(&mut w, &ids);
    assert_eq!(fires.len(), 2, "at most two per turn: {fires:?}");
    let (first, second) = (fires[0].0, fires[1].0);
    assert_eq!((fires[0].1, fires[1].1), (1, 2));
    assert!(
        second >= (2 * first).max(first + 4096),
        "{first} -> {second}"
    );

    for (cap, expected) in [(1u8, 1usize), (0, 0)] {
        let policy = ThinkNudgePolicy {
            max_nudges: cap,
            ..Default::default()
        };
        let mut w = ThinkWatch::new(policy);
        assert_eq!(run_watch(&mut w, &ids).len(), expected, "cap {cap}");
    }
}

// False positives: legitimately repetitive reasoning content. Each case runs
// after 3200 tokens of ordinary reasoning, so only the content decides.

fn after_reasoning(content: &str, laps: usize) -> (Vec<u32>, WordTok) {
    let mut tok = WordTok::new();
    let mut ids = novel_prose(&mut tok, 3200, 11);
    ids.push(PARA);
    for _ in 0..laps {
        ids.extend(tok.encode(content));
    }
    (ids, tok)
}

/// A table of distinct cases (what a model writes when it enumerates).
fn case_table(rows: usize) -> String {
    let mut s = String::from("| case | input | expected | why |\n|---|---|---|---|\n");
    for i in 0..rows {
        let (a, b) = (i * 7 % 97, i * 13 % 89);
        s.push_str(&format!(
            "| {i} | n = {a}, k = {b} | {} | {} |\n",
            a * b % 101,
            ["edge", "small", "carry", "overflow", "empty"][i % 5]
        ));
    }
    s
}

/// A numbered plan whose steps differ: four rotating sentence frames over
/// varied verbs, objects and conditions (real Qwen3.8 tokens: max rate 0.20).
fn step_list(steps: usize) -> String {
    const V: [&str; 16] = [
        "read", "parse", "sort", "merge", "count", "check", "print", "store", "split", "scan",
        "hash", "compress", "validate", "index", "group", "reverse",
    ];
    const O: [&str; 12] = [
        "the grid",
        "each query",
        "the edges",
        "both arrays",
        "the answer",
        "the prefix sums",
        "the visited set",
        "the heap",
        "every row",
        "the input string",
        "the segment tree",
        "the left half",
    ];
    const G: [&str; 6] = [
        "the totals line up",
        "no index overflows",
        "duplicates are dropped",
        "the order is stable",
        "we stay under the limit",
        "the last case is covered",
    ];
    const C: [&str; 6] = [
        "the list is empty",
        "n is odd",
        "two values tie",
        "the query is out of range",
        "the graph has a cycle",
        "k exceeds n",
    ];
    const T: [&str; 6] = [
        "a deque",
        "binary search",
        "a running maximum",
        "two pointers",
        "a counter",
        "bit masks",
    ];
    let mut s = String::new();
    let mut x: u64 = 7;
    for i in 0..steps {
        x = (x * 1103515245 + 12345) % (1 << 31);
        let (v, o) = (V[(x % 16) as usize], O[((x >> 4) % 12) as usize]);
        let (g, c, t) = (
            G[((x >> 8) % 6) as usize],
            C[((x >> 12) % 6) as usize],
            T[((x >> 16) % 6) as usize],
        );
        let n = i + 1;
        s.push_str(&match i % 4 {
            0 => format!("{n}. {v} {o} so that {g}.\n"),
            1 => format!("{n}. If {c}, {v} {o} first.\n"),
            2 => format!("{n}. Then {v} {o} with {t}.\n"),
            _ => format!("{n}. {o} should be ready before step {}.\n", i + 3),
        });
    }
    s
}

/// One sentence frame repeated for every line: only the slot words change.
fn templated_list(steps: usize) -> String {
    let verbs = [
        "read", "parse", "sort", "merge", "count", "check", "print", "store",
    ];
    let objects = [
        "the grid",
        "each query",
        "the edges",
        "both arrays",
        "the answer",
    ];
    let mut s = String::new();
    for i in 0..steps {
        s.push_str(&format!(
            "{}. Next I {} {} and keep the result for step {}.\n",
            i + 1,
            verbs[i % verbs.len()],
            objects[(i / 3) % objects.len()],
            i + 2
        ));
    }
    s
}

#[test]
fn false_positive_table_of_distinct_cases_does_not_trigger() {
    let (ids, _) = after_reasoning(&case_table(160), 1);
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    assert!(
        run_watch(&mut w, &ids).is_empty(),
        "max rate {}",
        w.window_rate()
    );
}

#[test]
fn false_positive_numbered_plan_does_not_trigger() {
    let (ids, _) = after_reasoning(&step_list(200), 1);
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    assert!(
        run_watch(&mut w, &ids).is_empty(),
        "max rate {}",
        w.window_rate()
    );
}

#[test]
fn false_positive_reasoning_that_varies_its_numbers_does_not_trigger() {
    // Re-deriving a recurrence for successive n: the frame repeats, the
    // numbers do not (tokens are single digits, as in Qwen).
    let mut text = String::new();
    for n in 2..140u64 {
        text.push_str(&format!(
            "For n = {n}, f({n}) = f({}) + f({}) = {} which is {}.\n",
            n - 1,
            n - 2,
            n * n + 3 * n + 1,
            if n % 2 == 0 { "even" } else { "odd" }
        ));
    }
    let (ids, _) = after_reasoning(&text, 1);
    let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
    let fires = run_watch(&mut w, &ids);
    assert!(fires.is_empty(), "fired at {fires:?}");
}

fn boilerplate_tests(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "def test_case_{i}(self):\n    result = solve({}, {})\n    self.assertEqual(result, {})\n\n",
            i * 3 % 17,
            i * 5 % 23,
            i * 11 % 31
        ));
    }
    s
}

/// Known limitation, pinned so a change is deliberate: boilerplate code
/// written UNFENCED in the thinking, and a list that repeats one long
/// sentence frame per line, repeat enough 8-grams to trigger (real Qwen3.8
/// tokens: 0.64 and 0.35). Fenced code is never read
/// (`fenced_code_is_not_read_by_the_detector`). The GPU validation plan
/// measures how often this happens on real long outputs.
#[test]
fn unfenced_boilerplate_and_templated_lists_are_known_triggers() {
    for content in [boilerplate_tests(120), templated_list(120)] {
        let (ids, _) = after_reasoning(&content, 1);
        let mut w = ThinkWatch::new(ThinkNudgePolicy::default());
        assert!(!run_watch(&mut w, &ids).is_empty());
    }
}

// ── Injection state machine ─────────────────────────────────────────────────

/// Byte-free test encoding of the nudge texts: short, distinct id runs.
fn test_encode(text: &str) -> Result<Vec<u32>> {
    let (prefixed, body) = match text.strip_prefix("\n\n") {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let mut ids = if prefixed { vec![49] } else { vec![] };
    if body == NUDGE_TEXT_1 {
        ids.extend([50, 51, 52]);
    } else if body == NUDGE_TEXT_2 {
        ids.extend([53, 54]);
    } else {
        bail!("unexpected nudge text {body:?}");
    }
    Ok(ids)
}

fn newline_table() -> Vec<u8> {
    let mut t = vec![NewlineClass::Other as u8; 64];
    t[NL as usize] = NewlineClass::Newline as u8;
    t[PARA as usize] = NewlineClass::EndsParagraph as u8;
    t
}

/// A policy small enough to drive by hand: 2-grams, 16-token window, due
/// the moment it arms.
fn tiny_policy() -> ThinkNudgePolicy {
    ThinkNudgePolicy {
        min_tokens: 16,
        window: 16,
        rate: 0.5,
        min_gap: 0,
        boundary_wait: 0,
        ngram: 2,
        max_nudges: 2,
    }
}

fn tiny_cfg(policy: ThinkNudgePolicy) -> Arc<ThinkNudgeConfig> {
    Arc::new(ThinkNudgeConfig::build(policy, test_encode, newline_table(), &[7, 8]).unwrap())
}

fn thinking_view() -> SeqView {
    SeqView {
        inside_thinking: true,
        max_output_tokens: 100_000,
        ..Default::default()
    }
}

/// Commit `tok` (forced if the state says so) the way the scheduler does.
fn commit(s: &mut ThinkNudgeState, v: &mut SeqView, sampled: u32) -> (u32, NudgeEvent) {
    let tok = s.forced_token(v).unwrap_or(sampled);
    v.thinking_tokens += 1;
    v.output_len += 1;
    (tok, s.on_thinking_token(v, tok))
}

#[test]
fn config_rejects_structural_tokens_in_a_nudge() {
    let err = ThinkNudgeConfig::build(tiny_policy(), test_encode, newline_table(), &[51]);
    assert!(err.is_err());
}

#[test]
fn a_started_nudge_is_forced_token_by_token_then_counted() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    let mut out = Vec::new();
    let mut events = Vec::new();
    for _ in 0..24 {
        let (tok, ev) = commit(&mut s, &mut v, 40);
        out.push(tok);
        if ev != NudgeEvent::None {
            events.push(ev);
        }
    }
    let at = out.iter().position(|&t| t != 40).expect("nudge injected");
    // Not at a paragraph break: the "\n\n"-prefixed variant.
    assert_eq!(&out[at..at + 4], &[49, 50, 51, 52]);
    assert!(out[at + 4..].iter().all(|&t| t == 40));
    assert!(matches!(events[0], NudgeEvent::Started { which: 0, .. }));
    assert_eq!(events[1], NudgeEvent::Completed { which: 0 });
    assert_eq!(s.completed(), 1);
    // Injected tokens are not observed by the detector.
    assert_eq!(s.watch.observed() as usize, 24 - 4);
}

#[test]
fn a_nudge_at_a_paragraph_break_uses_the_plain_variant() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    let mut out = Vec::new();
    for i in 0..24 {
        let sampled = if i % 2 == 0 { 40 } else { PARA };
        out.push(commit(&mut s, &mut v, sampled).0);
    }
    let at = out.iter().position(|t| (49..=54).contains(t)).unwrap();
    assert_eq!(out[at - 1], PARA);
    assert_eq!(&out[at..at + 3], &[50, 51, 52]);
}

#[test]
fn second_nudge_uses_the_firm_text_and_the_cap_holds() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    let out: Vec<u32> = (0..200).map(|_| commit(&mut s, &mut v, 40).0).collect();
    let injected: Vec<u32> = out.iter().copied().filter(|&t| t != 40).collect();
    assert_eq!(injected, vec![49, 50, 51, 52, 49, 53, 54]);
    assert_eq!(s.completed(), 2);
}

#[test]
fn a_nudge_never_starts_inside_a_code_fence() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    // Arm on prose first, then open a fence right as it becomes due.
    while !matches!(commit(&mut s, &mut v, 40).1, NudgeEvent::Started { .. }) {}
    v.in_code_fence = true;
    for _ in 0..30 {
        assert_eq!(s.forced_token(&v), None, "deferred while fenced");
        let (tok, ev) = commit(&mut s, &mut v, 41);
        assert_eq!((tok, ev), (41, NudgeEvent::None));
    }
    v.in_code_fence = false;
    assert_eq!(s.forced_token(&v), Some(49));
}

#[test]
fn matching_prefix_token_inside_code_fence_or_closed_think_never_starts_injection() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    // Arm a prefixed nudge (`tokens = [49, 50, 51, 52]`).
    while !matches!(commit(&mut s, &mut v, 40).1, NudgeEvent::Started { .. }) {}
    v.in_code_fence = true;
    // Model naturally emits `49` (e.g. blank line) inside the code fence:
    // must NOT advance `inj.next` to 1 or force `50` into the code block.
    assert_eq!(s.forced_token(&v), None);
    assert_eq!(s.on_thinking_token(&v, 49), NudgeEvent::None);
    assert_eq!(
        s.forced_token(&v),
        None,
        "must not start injecting nudge tokens inside a code fence"
    );
    v.in_code_fence = false;
    assert_eq!(
        s.forced_token(&v),
        Some(49),
        "nudge stays parked at token 0 until the fence closes"
    );

    // If `force_end_thinking` arms before injection starts, seeing `49` must
    // abandon the nudge rather than advancing it.
    v.force_end_thinking = true;
    assert_eq!(s.forced_token(&v), None);
    assert_eq!(
        s.on_thinking_token(&v, 49),
        NudgeEvent::Abandoned {
            which: 0,
            injected: 0,
        }
    );
}

#[test]
fn fenced_code_is_not_read_by_the_detector() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    v.in_code_fence = true;
    for _ in 0..200 {
        assert_eq!(commit(&mut s, &mut v, 40), (40, NudgeEvent::None));
    }
    assert_eq!(s.watch.observed(), 0);
    assert_eq!(s.completed(), 0);
}

#[test]
fn a_forced_think_close_abandons_a_nudge_in_progress() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    while !matches!(commit(&mut s, &mut v, 40).1, NudgeEvent::Started { .. }) {}
    assert_eq!(commit(&mut s, &mut v, 40).0, 49);
    // The thinking budget arms `</think>`: the nudge yields at once.
    v.force_end_thinking = true;
    assert_eq!(s.forced_token(&v), None);
    let ev = s.on_thinking_token(&v, 99);
    assert_eq!(
        ev,
        NudgeEvent::Abandoned {
            which: 0,
            injected: 1
        }
    );
    assert_eq!(s.completed(), 0);
    assert_eq!(s.forced_token(&thinking_view()), None);
}

#[test]
fn a_nudge_that_does_not_fit_is_not_started() {
    for (budget, max_out) in [(Some(16 + 3 + 255), 100_000), (None, 16 + 4 + 255)] {
        let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
        let mut v = thinking_view();
        v.thinking_budget = budget;
        v.max_output_tokens = max_out;
        for _ in 0..64 {
            assert_eq!(commit(&mut s, &mut v, 40).0, 40, "{budget:?} {max_out}");
        }
    }
}

#[test]
fn a_ready_nudge_waits_out_a_speculative_frame_then_gives_up() {
    let mut s = ThinkNudgeState::new(tiny_cfg(tiny_policy()));
    let mut v = thinking_view();
    while !matches!(commit(&mut s, &mut v, 40).1, NudgeEvent::Started { .. }) {}
    // A verifier that ignores `forced_token` keeps committing its own
    // tokens: the nudge waits (bounded), then is given up without a trace.
    for _ in 0..MAX_START_WAIT {
        v.thinking_tokens += 1;
        assert_eq!(s.on_thinking_token(&v, 40), NudgeEvent::None);
    }
    assert!(s.forced_token(&v).is_some());
    v.thinking_tokens += 1;
    assert_eq!(
        s.on_thinking_token(&v, 40),
        NudgeEvent::Abandoned {
            which: 0,
            injected: 0
        }
    );
    assert_eq!(s.forced_token(&v), None);
}

// ── Scheduler paths (mock model) ────────────────────────────────────────────

const TE: u32 = 60;
const LOOP_TOK: u32 = 40;

/// A greedy sequence inside `<think>` whose model always prefers `LOOP_TOK`.
fn looping_seq(
    nudge: Option<Box<ThinkNudgeState>>,
) -> (
    ActiveSeq,
    tokio::sync::oneshot::Receiver<anyhow::Result<crate::api::InferenceResponse>>,
) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = test_fixtures::active_seq(ResponseSink::Blocking(Some(tx)), 0);
    a.output_tokens.clear();
    a.max_output_tokens = 4096;
    a.remaining = 4096;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.content_started = false;
    a.content_tokens = 0;
    a.think_end_token = Some(TE);
    a.think_nudge = nudge;
    (a, rx)
}

fn looping_model() -> test_fixtures::model::TestModel {
    let m = test_fixtures::model::recording();
    let mut row = vec![0.0f32; 64];
    row[LOOP_TOK as usize] = 8.0;
    row[41] = 2.0;
    *m.host_logits.lock().unwrap() = row;
    m
}

fn plain_steps(model: &dyn spark_model::traits::Model, a: ActiveSeq, n: usize) -> ActiveSeq {
    let mut active = vec![a];
    for _ in 0..n {
        process_decode_logits(
            model,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            Some(TE),
            None,
            None,
            None,
            None,
            &[],
            false,
            false,
        );
    }
    active.pop().expect("sequence stays active")
}

#[test]
fn plain_decode_injects_the_nudge_inside_open_thinking() {
    let model = looping_model();
    let state = Box::new(ThinkNudgeState::new(tiny_cfg(tiny_policy())));
    let (a, _rx) = looping_seq(Some(state));
    let a = plain_steps(&model, a, 40);
    let injected: Vec<u32> = a
        .output_tokens
        .iter()
        .copied()
        .filter(|&t| t != LOOP_TOK)
        .collect();
    assert_eq!(injected, vec![49, 50, 51, 52, 49, 53, 54]);
    assert_eq!(a.output_tokens.len(), 40, "one committed token per step");
    assert_eq!(a.thinking_tokens, 40, "injected tokens are thinking tokens");
    assert!(
        a.inside_thinking && !a.think_ended && !a.finished,
        "never closes or stops"
    );
    assert_eq!(
        a.last_token,
        *a.output_tokens.last().unwrap(),
        "fed back as input"
    );
    assert_eq!(completed(&a), 2);
}

#[test]
fn plain_decode_reports_model_logprobs_for_injected_tokens() {
    let model = looping_model();
    let state = Box::new(ThinkNudgeState::new(tiny_cfg(tiny_policy())));
    let (mut a, _rx) = looping_seq(Some(state));
    a.top_logprobs = Some(2);
    let a = plain_steps(&model, a, 24);
    assert_eq!(a.logprobs_data.len(), a.output_tokens.len());
    for (lp, &tok) in a.logprobs_data.iter().zip(&a.output_tokens) {
        assert_eq!(lp.token_id, tok);
        if tok != LOOP_TOK {
            // The model's own (low) probability for the injected token.
            assert!(lp.logprob < -5.0, "{tok}: {}", lp.logprob);
            assert_eq!(lp.top[0].0, LOOP_TOK);
        }
    }
}

#[test]
fn default_off_plain_decode_is_unchanged() {
    // The nudge off is `think_nudge: None` (what `new_state` returns with no
    // config installed): the model's own tokens, nothing else.
    assert!(settings(&[]).unwrap().is_none());
    let model = looping_model();
    let (a, _rx) = looping_seq(None);
    // 47 steps: below THINK_LOOP_MIN_TOKENS, so the period watchdog (which
    // force-closes a single-token loop) stays out of this comparison.
    let off = plain_steps(&model, a, 47);
    assert_eq!(off.output_tokens, vec![LOOP_TOK; 47]);
    assert_eq!(off.thinking_tokens, 47);
    assert_eq!(completed(&off), 0);
    assert!(!injection_pending(&off));

    // A configured nudge whose detector never fires is equally invisible.
    let quiet = ThinkNudgePolicy {
        max_nudges: 0,
        ..tiny_policy()
    };
    let (a, _rx2) = looping_seq(Some(Box::new(ThinkNudgeState::new(tiny_cfg(quiet)))));
    let on = plain_steps(&model, a, 47);
    assert_eq!(on.output_tokens, off.output_tokens);
}

#[test]
fn injection_pending_holds_speculation_only_while_injecting() {
    let model = looping_model();
    let state = Box::new(ThinkNudgeState::new(tiny_cfg(tiny_policy())));
    let (mut a, _rx) = looping_seq(Some(state));
    let mut pending = Vec::new();
    for _ in 0..24 {
        pending.push(injection_pending(&a));
        a = plain_steps(&model, a, 1);
    }
    let first = pending.iter().position(|&p| p).unwrap();
    assert_eq!(&pending[first..first + 4], &[true; 4]);
    assert!(pending[first + 4..].iter().all(|&p| !p));
    assert!(pending[..first].iter().all(|&p| !p));
}

#[test]
fn think_spec_walk_forces_the_nudge_and_rejects_the_rest_of_the_frame() {
    use crate::scheduler::think_spec_accept::{ThinkSpecCtx, dflash_thinking_accept};
    let state = Box::new(ThinkNudgeState::new(tiny_cfg(tiny_policy())));
    let (a, _rx) = looping_seq(Some(state));
    let model = looping_model();
    // Plain decode up to the point where the nudge is ready.
    let mut a = a;
    while !injection_pending(&a) {
        a = plain_steps(&model, a, 1);
    }
    let before = a.output_tokens.len();
    let ctx = ThinkSpecCtx {
        enabled: true,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    };
    // Drafts and verifier both say LOOP_TOK; the fast path must still commit
    // the nudge token at slot 0, as the frame's bonus.
    let drafts = [LOOP_TOK, LOOP_TOK, LOOP_TOK];
    let verified = [LOOP_TOK; 4];
    let out = dflash_thinking_accept(&mut a, &drafts, &verified, &ctx, |_, _| false, |_| {});
    assert_eq!(out.num_accepted, 0);
    assert_eq!(out.bonus, Some(49));
    assert_eq!(&a.output_tokens[before..], &[49]);
    assert_eq!(a.last_token, 49);
    assert!(
        injection_pending(&a),
        "the rest is injected on plain decode"
    );
}

#[test]
fn finish_reports_the_nudge_count_on_both_sinks() {
    let model = looping_model();
    let state = Box::new(ThinkNudgeState::new(tiny_cfg(tiny_policy())));
    let (a, mut rx) = looping_seq(Some(state));
    let mut a = plain_steps(&model, a, 40);
    crate::scheduler::finish_sequence_with_cache(&model, &mut a, false);
    let response = rx.try_recv().unwrap().unwrap();
    assert_eq!(response.think_nudges, 2);

    let (tx, mut srx) = tokio::sync::mpsc::channel(4);
    let (mut b, _rx) = looping_seq(None);
    b.sink = ResponseSink::Streaming(tx);
    crate::scheduler::finish_sequence_with_cache(&model, &mut b, false);
    match srx.try_recv().unwrap() {
        crate::api::StreamEvent::Done { think_nudges, .. } => assert_eq!(think_nudges, 0),
        _ => panic!("expected Done"),
    }
}

#[test]
fn usage_omits_think_nudges_when_none_fired() {
    let details = |n| crate::openai::CompletionTokensDetails {
        reasoning_tokens: 5,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
        think_nudges: n,
    };
    let off = serde_json::to_string(&details(0)).unwrap();
    assert_eq!(
        off,
        r#"{"reasoning_tokens":5,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}"#
    );
    let on = serde_json::to_value(details(2)).unwrap();
    assert_eq!(on["think_nudges"], 2);
}
