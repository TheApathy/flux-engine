// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking nudge (`FLUX_THINK_NUDGE=on`, default OFF).
//!
//! When a turn's thinking goes in circles, the scheduler writes one short
//! first-person sentence into the open think block, suggesting the model
//! commit to an approach. It NEVER closes the think block, never stops or
//! truncates generation, and never touches content after `</think>`. The
//! model decides whether to wrap up.
//!
//! # Trigger
//!
//! At least `rate` (0.25) of the last `window` (1024) thinking tokens end an
//! `ngram` (8) token sequence that already occurred earlier in this turn's
//! thinking, and thinking is at least `min_tokens` (3072) long. Only sampled
//! thinking OUTSIDE ``` code fences is read (fenced code repeats legitimately)
//! and the counts are in those tokens. A due nudge waits up to
//! `boundary_wait` (256) tokens for a paragraph break, and never starts
//! inside a code fence. At most `max_nudges` (2) per turn; the
//! second (firmer) one needs the circling to go on past
//! `max(2 x, x + min_gap)` thinking tokens, `x` being where the first fired.
//!
//! # Injection mechanism
//!
//! An injected token is a FORCED SAMPLE at one decode position, the same seam
//! as the forced `</think>` of the thinking budget: [`process_seq_logits`]
//! returns the next nudge token instead of sampling, and the caller commits it
//! through the ordinary per-token path (pushed, streamed, fed back as
//! `last_token`, counted as a thinking token). The KV/SSM state therefore
//! always describes exactly the committed tokens. No position is rewritten
//! after the fact and no rollback path is added.
//!
//! Speculation is held on plain decode while an injection is pending
//! ([`injection_pending`] joins the per-sequence MTP/n-gram/self-spec gates,
//! which use the existing spec→decode transition). A nudge that becomes due
//! inside a speculative frame starts at the frame's next derived position:
//! the think-spec accept walk asks [`forced_token`] like the plain path does,
//! so the nudge token is that position's target and the rest of the frame is
//! rejected by the existing verify machinery.
//!
//! [`process_seq_logits`]: super::decode_logits_seq::process_seq_logits

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};

use super::ActiveSeq;

/// Nudge 1 (first person, Qwen3.8 thinking voice, no `</think>`).
pub const NUDGE_TEXT_1: &str = "I keep going back over the same ground. I already have enough to \
choose an approach, so I will commit to the most promising one and carry it through.\n\n";
/// Nudge 2, firmer: fired only if the circling goes on well past nudge 1.
pub const NUDGE_TEXT_2: &str = "I am still going in circles. I will take the simplest approach \
that meets the stated requirements, finish it, and then check the result.\n\n";

/// Tokens a nudge must leave free under both the thinking budget and the
/// response cap. A nudge that would land this close to either is skipped.
const MIN_ROOM_AFTER_NUDGE: usize = 256;
/// A ready injection that a speculative frame did not start within this many
/// committed tokens is dropped (defensive: the plain-decode hold makes the
/// next step serial, so this only bounds a logic error).
const MAX_START_WAIT: u32 = 64;

/// The two model families the nudge is qualified for.
pub fn model_supported(model: &str) -> bool {
    matches!(model, "qwen3.8-flash-next" | "qwen3.8-27b")
}

/// Detector and pacing knobs for thinking-nudge detection.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkNudgePolicy {
    pub min_tokens: u32,
    pub window: u32,
    pub rate: f64,
    pub min_gap: u32,
    pub boundary_wait: u32,
    pub ngram: usize,
    pub max_nudges: u8,
}

impl Default for ThinkNudgePolicy {
    fn default() -> Self {
        Self {
            min_tokens: 3072,
            window: 1024,
            rate: 0.25,
            min_gap: 4096,
            boundary_wait: 256,
            ngram: 8,
            max_nudges: 2,
        }
    }
}

impl ThinkNudgePolicy {
    pub fn validate(&self) -> Result<()> {
        if self.window < 16 {
            bail!("think nudge window {} must be >= 16", self.window);
        }
        if !(self.rate > 0.0 && self.rate <= 1.0) {
            bail!("think nudge rate {} must be in (0, 1]", self.rate);
        }
        if !(2..=32).contains(&self.ngram) {
            bail!("think nudge ngram {} must be in 2..=32", self.ngram);
        }
        if self.max_nudges > 2 {
            bail!(
                "think nudge max per turn {} must be 0, 1 or 2",
                self.max_nudges
            );
        }
        Ok(())
    }
}

/// Parse the server settings. `get(suffix)` returns the raw value of
/// `FLUX_<suffix>` / `ATLAS_<suffix>` (see [`env_setting`]).
///
/// `Ok(None)`: the nudge is off (the default). Every value is strict: a
/// malformed setting is a startup error rather than a silent fallback.
pub fn parse_settings(
    get: impl Fn(&str) -> Result<Option<String>>,
) -> Result<Option<ThinkNudgePolicy>> {
    let on = match get("THINK_NUDGE")?.as_deref().map(str::trim) {
        None | Some("") | Some("off") | Some("0") | Some("false") => false,
        Some("on") | Some("1") | Some("true") => true,
        Some(other) => bail!("FLUX_THINK_NUDGE={other:?}: expected off|on"),
    };
    if !on {
        return Ok(None);
    }
    let mut policy = ThinkNudgePolicy::default();
    if let Some(raw) = get("THINK_NUDGE_RATE")? {
        policy.rate = raw
            .trim()
            .parse::<f64>()
            .map_err(|e| anyhow::anyhow!("FLUX_THINK_NUDGE_RATE={raw:?}: {e}"))?;
    }
    if let Some(raw) = get("THINK_NUDGE_MIN_TOKENS")? {
        policy.min_tokens = raw
            .trim()
            .parse::<u32>()
            .map_err(|e| anyhow::anyhow!("FLUX_THINK_NUDGE_MIN_TOKENS={raw:?}: {e}"))?;
    }
    if let Some(raw) = get("THINK_NUDGE_MAX")? {
        policy.max_nudges = raw
            .trim()
            .parse::<u8>()
            .map_err(|e| anyhow::anyhow!("FLUX_THINK_NUDGE_MAX={raw:?}: {e}"))?;
    }
    policy.validate()?;
    Ok(Some(policy))
}

/// Read `FLUX_<suffix>`, falling back to `ATLAS_<suffix>` (the env bridge
/// mirrors one to the other). Both set to different values is an error.
pub fn env_setting(suffix: &str) -> Result<Option<String>> {
    let flux = std::env::var(format!("FLUX_{suffix}")).ok();
    let atlas = std::env::var(format!("ATLAS_{suffix}")).ok();
    match (flux, atlas) {
        (Some(f), Some(a)) if f != a => {
            bail!("FLUX_{suffix}={f:?} and ATLAS_{suffix}={a:?} disagree")
        }
        (Some(v), _) | (None, Some(v)) => Ok(Some(v)),
        (None, None) => Ok(None),
    }
}

/// How a token's decoded text ends, for the paragraph-break wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NewlineClass {
    /// Decodes to no text: leaves the paragraph state unchanged.
    Empty = 0,
    /// Ends in a character other than `\n`.
    Other = 1,
    /// Exactly `"\n"`: a paragraph break if the text before it ended in `\n`.
    Newline = 2,
    /// Ends in one `\n` preceded by a non-newline character.
    EndsNewline = 3,
    /// Ends in `"\n\n"`.
    EndsParagraph = 4,
}

impl NewlineClass {
    pub fn of(text: &str) -> Self {
        if text.is_empty() {
            Self::Empty
        } else if text == "\n" {
            Self::Newline
        } else if text.ends_with("\n\n") {
            Self::EndsParagraph
        } else if text.ends_with('\n') {
            Self::EndsNewline
        } else {
            Self::Other
        }
    }
}

/// Everything a sequence needs, resolved once per model load. The token
/// tables are shared, so a policy change ([`Self::with_policy`]) is cheap.
#[derive(Debug, Clone)]
pub struct ThinkNudgeConfig {
    pub policy: ThinkNudgePolicy,
    /// `texts[k][p]`: nudge `k + 1`; `p = 1` is the variant prefixed with
    /// `"\n\n"`, used when the nudge did not land on a paragraph break.
    texts: Arc<[[Vec<u32>; 2]; 2]>,
    /// Per-vocab-id [`NewlineClass`] (`u8`); ids past the end are `Other`.
    newline: Arc<Vec<u8>>,
}

impl ThinkNudgeConfig {
    /// `encode` tokenizes plain text (no special tokens); `newline` is the
    /// per-id class table. `forbidden` are structural ids (EOS, think and
    /// tool-call delimiters) that a nudge must never contain.
    pub fn build(
        policy: ThinkNudgePolicy,
        encode: impl Fn(&str) -> Result<Vec<u32>>,
        newline: Vec<u8>,
        forbidden: &[u32],
    ) -> Result<Self> {
        policy.validate()?;
        let mut texts: [[Vec<u32>; 2]; 2] = Default::default();
        for (k, text) in [NUDGE_TEXT_1, NUDGE_TEXT_2].into_iter().enumerate() {
            for (p, prefix) in ["", "\n\n"].into_iter().enumerate() {
                let ids = encode(&format!("{prefix}{text}"))?;
                if ids.is_empty() {
                    bail!("think nudge {} tokenizes to nothing", k + 1);
                }
                if let Some(bad) = ids.iter().find(|id| forbidden.contains(id)) {
                    bail!("think nudge {} contains structural token {bad}", k + 1);
                }
                texts[k][p] = ids;
            }
        }
        Ok(Self {
            policy,
            texts: Arc::new(texts),
            newline: Arc::new(newline),
        })
    }

    /// The same token tables under another (validated) policy.
    pub fn with_policy(&self, policy: ThinkNudgePolicy) -> Result<Self> {
        policy.validate()?;
        Ok(Self {
            policy,
            texts: Arc::clone(&self.texts),
            newline: Arc::clone(&self.newline),
        })
    }

    pub fn tokens(&self, which: usize, prefixed: bool) -> &[u32] {
        &self.texts[which][usize::from(prefixed)]
    }

    fn class(&self, id: u32) -> NewlineClass {
        match self.newline.get(id as usize).copied() {
            Some(0) => NewlineClass::Empty,
            Some(2) => NewlineClass::Newline,
            Some(3) => NewlineClass::EndsNewline,
            Some(4) => NewlineClass::EndsParagraph,
            _ => NewlineClass::Other,
        }
    }
}

/// Follows one turn's SAMPLED thinking tokens (never the injected ones) and
/// says when a nudge is due.
#[derive(Debug, Clone)]
pub struct ThinkWatch {
    policy: ThinkNudgePolicy,
    n: u32,
    /// The previous `ngram - 1` tokens, a ring starting at `last_at`.
    last: Vec<u32>,
    last_at: usize,
    filled: usize,
    seen: HashSet<u64>,
    /// Ring of the last `window` self-copy flags.
    hits: Vec<bool>,
    window_hits: u32,
    nudges: u8,
    first_at: Option<u32>,
    armed_at: Option<u32>,
    para: bool,
    tail_newline: bool,
}

impl ThinkWatch {
    pub fn new(policy: ThinkNudgePolicy) -> Self {
        let last = vec![0; policy.ngram - 1];
        let hits = vec![false; policy.window as usize];
        Self {
            policy,
            n: 0,
            last,
            last_at: 0,
            filled: 0,
            seen: HashSet::new(),
            hits,
            window_hits: 0,
            nudges: 0,
            first_at: None,
            armed_at: None,
            para: false,
            tail_newline: false,
        }
    }

    /// One sampled thinking token and how its text ends.
    pub fn observe(&mut self, id: u32, class: NewlineClass) {
        // FNV-1a over the ids of the n-gram ending here (oldest first).
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let len = self.last.len();
        for k in 0..len {
            for b in self.last[(self.last_at + k) % len].to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        for b in id.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        let full = self.filled == len;
        self.last[self.last_at] = id;
        self.last_at = (self.last_at + 1) % len;
        self.filled = (self.filled + 1).min(len);
        let hit = full && !self.seen.insert(h);
        let slot = (self.n % self.policy.window) as usize;
        self.window_hits = self.window_hits + u32::from(hit) - u32::from(self.hits[slot]);
        self.hits[slot] = hit;
        self.n += 1;
        match class {
            NewlineClass::Empty => {}
            NewlineClass::Other => (self.para, self.tail_newline) = (false, false),
            NewlineClass::Newline => (self.para, self.tail_newline) = (self.tail_newline, true),
            NewlineClass::EndsNewline => (self.para, self.tail_newline) = (false, true),
            NewlineClass::EndsParagraph => (self.para, self.tail_newline) = (true, true),
        }
        if self.armed_at.is_none() && self.nudges < self.policy.max_nudges && self.condition() {
            self.armed_at = Some(self.n);
        }
    }

    fn condition(&self) -> bool {
        let p = &self.policy;
        let repeating =
            self.n >= p.min_tokens && self.n >= p.window && self.window_rate() >= p.rate;
        match self.first_at {
            None => repeating,
            Some(first) => {
                let after = (2 * first).max(first.saturating_add(p.min_gap));
                self.n >= after && repeating
            }
        }
    }

    /// 0: nothing due; 1 / 2: nudge 1 / 2 is due now.
    pub fn due(&self) -> u8 {
        match self.armed_at {
            Some(at) if self.para || self.n - at >= self.policy.boundary_wait => self.nudges + 1,
            _ => 0,
        }
    }

    /// The text so far ends in a paragraph break.
    pub fn at_paragraph(&self) -> bool {
        self.para
    }

    /// The nudge [`Self::due`] named was used (injected or given up).
    pub fn fired(&mut self) {
        if self.nudges == 0 {
            self.first_at = Some(self.n);
        }
        self.nudges += 1;
        self.armed_at = None;
        // Every nudge text ends with a paragraph break.
        (self.para, self.tail_newline) = (true, true);
    }

    pub fn observed(&self) -> u32 {
        self.n
    }

    pub fn window_rate(&self) -> f64 {
        f64::from(self.window_hits) / f64::from(self.policy.window)
    }
}

/// The sequence state the nudge decisions read (a copy, so the decision
/// logic is testable without an `ActiveSeq`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SeqView {
    pub inside_thinking: bool,
    pub think_ended: bool,
    pub force_end_thinking: bool,
    pub in_code_fence: bool,
    pub thinking_tokens: u32,
    pub thinking_budget: Option<u32>,
    pub output_len: usize,
    pub max_output_tokens: usize,
}

impl SeqView {
    fn open(&self) -> bool {
        self.inside_thinking && !self.think_ended && !self.force_end_thinking
    }

    fn fits(&self, len: usize) -> bool {
        let budget_ok = self.thinking_budget.is_none_or(|b| {
            self.thinking_tokens as usize + len + MIN_ROOM_AFTER_NUDGE <= b as usize
        });
        budget_ok && self.output_len + len + MIN_ROOM_AFTER_NUDGE <= self.max_output_tokens
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Injection {
    which: usize,
    prefixed: bool,
    next: usize,
    waited: u32,
}

/// What one committed thinking token did to the nudge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NudgeEvent {
    None,
    /// Nudge `which + 1` will be forced from the next position.
    Started {
        which: usize,
        observed: u32,
        rate: f64,
    },
    /// Every token of nudge `which + 1` is committed.
    Completed {
        which: usize,
    },
    /// Nudge `which + 1` was given up after `injected` of its tokens.
    Abandoned {
        which: usize,
        injected: usize,
    },
}

/// Per-sequence nudge state. `None` on `ActiveSeq` when the nudge is off.
#[derive(Debug, Clone)]
pub struct ThinkNudgeState {
    cfg: Arc<ThinkNudgeConfig>,
    watch: ThinkWatch,
    inject: Option<Injection>,
    completed: u32,
    abandoned: u32,
}

impl ThinkNudgeState {
    pub fn new(cfg: Arc<ThinkNudgeConfig>) -> Self {
        let watch = ThinkWatch::new(cfg.policy.clone());
        Self {
            cfg,
            watch,
            inject: None,
            completed: 0,
            abandoned: 0,
        }
    }

    /// The token the next thinking position must commit, if a nudge is being
    /// injected there.
    pub fn forced_token(&self, v: &SeqView) -> Option<u32> {
        let inj = self.inject?;
        if !v.open() || (inj.next == 0 && v.in_code_fence) {
            return None;
        }
        self.cfg
            .tokens(inj.which, inj.prefixed)
            .get(inj.next)
            .copied()
    }

    /// Account one committed thinking token (sampled or forced). `v` is the
    /// sequence state after the commit.
    pub fn on_thinking_token(&mut self, v: &SeqView, tok: u32) -> NudgeEvent {
        if let Some(mut inj) = self.inject {
            let tokens = self.cfg.tokens(inj.which, inj.prefixed);
            let injection_active = v.open() && !(inj.next == 0 && v.in_code_fence);
            if injection_active && tokens.get(inj.next) == Some(&tok) {
                inj.next += 1;
                if inj.next == tokens.len() {
                    self.inject = None;
                    self.watch.fired();
                    self.completed += 1;
                    return NudgeEvent::Completed { which: inj.which };
                }
                self.inject = Some(inj);
                return NudgeEvent::None;
            }
            let give_up = inj.next > 0 || !v.open() || {
                if !v.in_code_fence {
                    inj.waited += 1;
                }
                inj.waited > MAX_START_WAIT
            };
            if give_up {
                return self.abandon(inj);
            }
            self.inject = Some(inj);
        }
        // Fenced code is not read: repetitive code (tests, tables of cases)
        // repeats 8-grams legitimately, and a nudge never lands inside it.
        if v.in_code_fence {
            return NudgeEvent::None;
        }
        self.watch.observe(tok, self.cfg.class(tok));
        if self.inject.is_none() && v.open() {
            let due = self.watch.due();
            if due > 0 {
                let which = usize::from(due - 1);
                let prefixed = !self.watch.at_paragraph();
                if v.fits(self.cfg.tokens(which, prefixed).len()) {
                    self.inject = Some(Injection {
                        which,
                        prefixed,
                        next: 0,
                        waited: 0,
                    });
                    return NudgeEvent::Started {
                        which,
                        observed: self.watch.observed(),
                        rate: self.watch.window_rate(),
                    };
                }
            }
        }
        NudgeEvent::None
    }

    fn abandon(&mut self, inj: Injection) -> NudgeEvent {
        self.inject = None;
        self.watch.fired();
        self.abandoned += 1;
        NudgeEvent::Abandoned {
            which: inj.which,
            injected: inj.next,
        }
    }

    /// Nudges injected in full this turn (the per-response counter).
    pub fn completed(&self) -> u32 {
        self.completed
    }
}

// ── Per-request configuration ────────────────────────────────────────────────

/// A new sequence's nudge state from the config its request carries
/// (`InferenceRequest::think_nudge`, taken from the request's settings
/// snapshot at HTTP admission); `None` when the nudge is off. Each sequence
/// keeps the config it was admitted with.
pub(super) fn new_state(cfg: Option<Arc<ThinkNudgeConfig>>) -> Option<Box<ThinkNudgeState>> {
    cfg.map(|cfg| Box::new(ThinkNudgeState::new(cfg)))
}

// ── Scheduler hooks ──────────────────────────────────────────────────────────

fn view(a: &ActiveSeq) -> SeqView {
    SeqView {
        inside_thinking: a.inside_thinking,
        think_ended: a.think_ended,
        force_end_thinking: a.force_end_thinking,
        in_code_fence: a.in_code_fence,
        thinking_tokens: a.thinking_tokens,
        thinking_budget: a.thinking_budget,
        output_len: a.output_tokens.len(),
        max_output_tokens: a.max_output_tokens,
    }
}

/// The nudge token this sequence's next position must commit (`None` when
/// the nudge is off or nothing is being injected).
pub(super) fn forced_token(a: &ActiveSeq) -> Option<u32> {
    a.think_nudge.as_ref()?.forced_token(&view(a))
}

/// Speculation must hold on plain decode for this sequence.
///
/// Repetition nudges (`think_nudge`) still hold on plain decode, whereas
/// thinking-budget close nudges (`think_budget_nudge`) are spliced directly
/// into flat speculative drafts and accepted inside `think_spec_accept` so a
/// single sequence hitting `thinking_budget` never ejects a concurrent batched
/// cohort onto serial `M=1` decode.
pub(super) fn injection_pending(a: &ActiveSeq) -> bool {
    forced_token(a).is_some()
}

/// Nudges injected in full for this sequence.
pub(super) fn completed(a: &ActiveSeq) -> u32 {
    a.think_nudge.as_ref().map_or(0, |s| s.completed())
}

/// Call once per committed thinking-content token (after `thinking_tokens`
/// and the fence parity are updated). A no-op when the nudge is off.
pub(super) fn on_thinking_token(a: &mut ActiveSeq, tok: u32) {
    if a.think_nudge.is_none() {
        return;
    }
    let v = view(a);
    let Some(state) = a.think_nudge.as_mut() else {
        return;
    };
    match state.on_thinking_token(&v, tok) {
        NudgeEvent::None => {}
        NudgeEvent::Started {
            which,
            observed,
            rate,
        } => tracing::info!(
            nudge = which + 1,
            thinking_tokens = v.thinking_tokens,
            observed,
            self_copy_rate = format_args!("{:.2}", rate),
            "THINK_NUDGE: thinking is going in circles; injecting nudge"
        ),
        NudgeEvent::Completed { which } => {
            crate::openai::record_shared_intervention(
                a.intervention_ledger.as_ref(),
                "think_nudge",
                "inject_reasoning_text",
            );
            tracing::info!(
                nudge = which + 1,
                thinking_tokens = v.thinking_tokens,
                "THINK_NUDGE: nudge injected"
            );
        }
        NudgeEvent::Abandoned { which, injected } => tracing::warn!(
            nudge = which + 1,
            injected,
            thinking_tokens = v.thinking_tokens,
            force_end_thinking = v.force_end_thinking,
            "THINK_NUDGE: nudge abandoned before completion"
        ),
    }
}

#[cfg(test)]
#[path = "think_nudge_tests.rs"]
mod think_nudge_tests;
