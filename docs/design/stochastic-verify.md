# Stochastic speculative verification (`FLUX_SPEC_VERIFY=stochastic`)

Goal: MTP speculation that pays at the checkpoint's own sampling (`T=1.0`,
`top_p 0.95`, `top_k 20`) without changing the output distribution.

## Status

| Piece | State |
|---|---|
| Primary rule: sample-then-match for greedy MTP drafts | done (`stochastic_verify_walk.rs`) |
| General p/q rule for sampled drafts (extension) | done, host-only (`stochastic_verify.rs`) |
| Adaptive depth controller (`adaptive_depth.rs`) | done, wired on the Qwen4 batched-K5 route, depth 0 = plain decode |
| Device-side sample-then-match (no logits D2H) | **GPU TODO** |

Default (`FLUX_SPEC_VERIFY` unset, `argmax`, or any other value) is untouched:
the dispatch check returns before anything else runs.

## 0. Primary rule: sample-then-match

The Qwen4 MTP drafter is greedy: each chain step writes only its argmax
(`argmax_out`, `qwen4_mtp.rs` `CHAIN_ARGMAX_OFFSET`) and feeds it to the next
step through `embed_from_argmax`. A greedy draft is a point mass, `q = δ_x`.
For a point mass, exact speculative sampling reduces to standard
sample-then-match verification:

> at each verify position, sample the target token with the plain sampler and
> the RNG stream keyed by the absolute output position; accept while the
> sample equals the draft; the first mismatch commits the sample and ends the
> frame; if all drafts match, the next sample is the bonus.

Lossless: the committed token at every position is literally the plain
sampler's draw for that position, so the stream is distributed as plain
decode. No draft probabilities are needed, and there is no GPU plumbing for q.

**Identity gate.** Flux keys its RNG by position already: plain decode and
every verify walk call `ActiveSeq::sampling_params`, whose seed is
`request_seed + output_tokens.len()`, and sample with
`sample_with_params_history` (`StdRng::seed_from_u64`). The stochastic walk
calls the same function on the same masked row with the same history, so for
a **seeded** request spec and nospec are byte-identical at T=1.0, whatever
the drafts. Test: `seeded_walk_is_token_identical_to_plain_sampling` (200
seeds, all-correct and wrong-at-position-1 drafts). Unseeded requests use
`rand::random` per draw in both paths, so they are distribution-equal only;
deriving a per-request seed at admission would extend the gate to them.

**Tie rule.** Verify and plain decode only need to agree with each other, and
they do by construction: both run the one sampler. For the record, the house
rule is FIRST-wins (`first_max_index`, commit 9b75cf825), replacing earlier
last-wins tie-breaking.

**Relation to today's code.** The existing T>0 content walk
(`dflash_content_accept` → `process_seq_logits`) already samples each verify
row with that seeded sampler and accepts on a match, so Flux already has
sample-then-match on the host. The stochastic walk is the same rule without
the per-position intervention machinery, behind a strict eligibility gate.
Its value is as the reference for the device kernel (GPU TODO), not as a
host speedup.

## 1. Extension: the general p/q rule (sampled drafts)

For a frame with drafts `x_0..x_{γ-1}`, target rows `p_0..p_γ`, and draft rows
`q_0..q_{γ-1}`:

1. For `i = 0..γ-1`: draw `u ~ U[0,1)`. Accept `x_i` iff `u·q_i(x_i) < p_i(x_i)`
   (equivalently `u < min(1, p/q)`; strict, so `p(x)=0` is never accepted).
2. On the first rejection at `i`: emit `y ~ norm(max(0, p_i − q_i))` and stop.
   If that residual has no mass (only reachable by rounding, since
   `Σ max(0,p−q) = TV(p,q) = 1 − P(accept) > 0` whenever rejection is
   possible), sample `y ~ p_i` instead (`RejectBranch::TargetFallback`).
3. If all `γ` are accepted, emit the bonus `y ~ p_γ`.

The frame commits `accepted + 1` tokens. Acceptance length per frame is the
count of accepted drafts; `adaptive_depth` folds it as Bernoulli trials
(every accepted draft, plus the one rejected draft if the frame stopped early).

### What p and q are

- **p** is the target's *post-policy* distribution, built by
  `prepare_sampling_distribution_with_history` — the same function the plain
  sampler uses (penalties, logit bias, top-n-sigma, temperature, top-k, min-p,
  top-p, in that order) — from the verify row after the plain path's
  content-position hard masks (`<think>`/`</think>` after thinking ended),
  with history = `output_tokens` *including the drafts already accepted in
  this frame*. The walk commits each token before preparing the next row, so
  penalties see exactly the prefix plain decode would.
- **q** is whatever distribution the draft token was actually drawn from.
  - A deterministic draft is a point mass; the walk then uses section 0
    instead of this rule (same distribution, plus token identity).
  - With sampled drafts, q must be the MTP head's distribution **after the
    same T/top_k/top_p processing** (the request's `SamplingParams`) and the
    draft must be sampled from it. Any q works for correctness as long as the
    draft really is drawn from the q passed to the verifier; using the same
    processing makes q close to p, which is what raises acceptance.

### Why top-k / top-p truncation is consistent

Truncation is part of p's definition, not something applied to the rule. p is
the exact distribution plain decode samples from (truncated and renormalized),
so tokens outside its nucleus have `p=0`: they are never accepted (strict
`<`), and `max(0, p−q)` is 0 there so the residual cannot emit them
(`sample_weights` only crosses the CDF on positive steps; tested). q may be
truncated differently (its own nucleus over the MTP logits, or its
`--mtp-vocab` cap); a q-support that misses part of p's support is fine — that
mass comes back through the residual.

### Lossless-ness (proof sketch)

Fix one position. Draft `x ~ q`. Then

    P(emit y) = q(y)·min(1, p(y)/q(y)) + P(reject)·r(y)
              = min(q(y), p(y)) + (1 − Σ_z min(p(z), q(z))) · max(0, p(y)−q(y)) / Σ_z max(0, p(z)−q(z))

and `Σ_z max(0, p−q) = 1 − Σ_z min(p, q)`, so the second term is
`max(0, p(y) − q(y))` and the total is `min(p,q) + max(0, p−q) = p(y)`.
For a point mass `q = δ_x` this reads `P(y) = p(y)` with acceptance `p(x)`.
By induction over positions (each row is conditioned on the committed prefix,
and p_i is computed from exactly that prefix), the emitted sequence is
distributed as sampling the target alone. The bonus row is a plain sample.

Tests: chi-square over an 8-token vocab, 1e5 samples, dense q that disagrees
with p and puts mass on p's truncated ids, and the point-mass case; empirical
acceptance equals `Σ min(p,q)` (dense) and `p(argmax)` (point mass).

## 2. RNG and seeds

Section 0 consumes exactly the plain sampler's draw at each position (seed
`request_seed + output index`, caller-advanced per output token), which is what makes spec and nospec token-identical when seeded.
The p/q extension needs two draws per position and a vocab-order CDF, so it
is distribution-equal but not token-identical; it draws them from the same
position seed (`sampling_uniforms_from_seed(seed, 2)`). Either way the stream
is per request and per output index, so batch packing cannot change it.

## 3. T = 0 reduces exactly to today's rule

At `temperature <= 0` the sampler returns a one-hot on the post-penalty
argmax, chosen with the house **first-wins** tie rule (`first_max_index`,
commit 9b75cf825, superseding earlier last-wins `max_by` behavior). With one-hot `p = δ_a`:

- accept iff `u·q(x) < [x == a]`, i.e. iff `x == a` (any `u < 1`, any q with `q(x)>0`);
- reject → residual `max(0, δ_a − q)` is positive only at `a` → emit `a`;
- bonus → `a`.

So the frame is exactly "accept while draft == first-wins argmax, then emit
the argmax", with no dependence on the uniforms. A test runs 500 random frames
over coarse (tie-heavy) logits and compares against the first-wins argmax
reference. In serving, T=0 never even reaches the stochastic walk: the
eligibility gate requires `temperature > 0`, so greedy keeps its
device-argmax fast path byte for byte.

## 4. Interaction with thinking and logprobs

- **Thinking.** Positions with `inside_thinking` stay on
  `run_dflash_thinking_accept` (think_spec_accept.rs). Thinking applies F1
  reflection suppression, F2 confidence early-stop, the decoding-efficiency
  wave, nudges and forced `</think>` injection — stateful interventions the
  stochastic walk does not reproduce. `think_just_ended` also closes the gate
  (tool-call pin). After thinking ends, the walk reproduces the plain path's
  only content hard mask (`<think>`/`</think>` → −inf).
- **Logprobs.** Any `top_logprobs` request closes the gate: logprobs must keep
  routing through `process_seq_logits`, the exact slow path (comparing
  logprobs vs non-logprobs output is a primary diagnostic invariant for
  verifying lossless fast paths, so the logprobs arm must remain the exact
  reference). Logits-FNV receipts
  (`ATLAS_LOGITS_FNV`) close it for the same reason.
- Also closed: adaptive sampling, grammar, tool-call suppression/requirement,
  tool body (penalties differ), `min_tokens` not reached, active nudge
  injection, `deepseek_v41` serving. A closed gate falls through to the
  existing walk unchanged.

## 5. Dispatch and wiring

- One dispatch point: the top of `run_dflash_content_accept_at_row`
  (spec_policy_accept.rs). Every content verify path (MTP serial, DFlash,
  self-spec, Qwen4 batched) funnels through it. At T>0
  `content_policy_required` is already true (`fast_path_seq_eligible` needs
  T=0), so these frames reach that point today.
- Adaptive depth: `effective_num_drafts` (verify_qwen4_batched_step.rs) on
  the batched-K5 route takes `min(width rule, controller depth)`. The verify
  width floors at 1 (drafts already proposed must be verified); a depth-0
  decision suppresses the next proposal in `propose_and_install_with`, so the
  following step has no drafts and runs the plain bootstrap decode.
  Acceptance is recorded at the K5 dispatch for content and thinking frames. Opt out with `FLUX_SPEC_ADAPTIVE_DEPTH=0`. The controller picks
  `argmax_k E(k)/(1 + c·k)` with `E(k) = Σ_{j≤k} a^j`, `a` an EWMA of
  per-draft acceptance, `c = 0.15` (to be recalibrated from receipts), and
  probes with one draft every 16 frames while at depth 0. One process-wide
  controller is correct at C1 (the native-precision recipe); per-request
  state is a follow-up for C>1.

## 6. What the drafter exposes today

Qwen4 MTP (`spark-model/src/layers/qwen4_mtp.rs`): per step the NVFP4 LM
head writes BF16 logits over `min(vocab, --mtp-vocab)`; `propose_chain`
keeps only the argmax (`CHAIN_ARGMAX_OFFSET`) and, when the gate wants it,
the argmax's softmax (`CHAIN_PROB_OFFSET`). The next step embeds the device
argmax and overwrites the logits. So drafts are greedy, and section 0 applies.

## 7. GPU TODO (no CUDA in this change)

1. **Device sample-then-match.** Today every T>0 frame copies all γ+1 full
   logits rows to the host and samples them there (about 2 ms per committed
   position; worth +8-18%, see section 8). A device kernel should sample each verify row with the
   position-keyed seed and compare against the drafts, returning
   `(accepted, next_token)`. For the identity gate to survive, it must
   reproduce `prepare_sampling_distribution_with_history` +
   `sample_with_uniform` bit for bit: penalties, bias, top-n-sigma,
   temperature, a deterministic descending sort (the host uses
   `sort_unstable_by`, whose tie order must be pinned first), top-k, min-p,
   top-p, the f32 cumulative walk, and `StdRng`'s first f32 draw (or switch
   both paths to a counter RNG such as Philox at once). The host walk is its
   oracle.
2. **Depth 0 routing.** Let a depth-0 frame run plain decode (the controller
   already emits 0).
3. Dropped: top-k draft-probability plumbing. It is only needed if the
   drafter ever samples; section 1 documents the rule for that case.

## 8. Expected payoff

Measured on GB10: spec already pays at T=1.0, +27% (37.7 vs 29.6 tok/s) with adaptive K and pooled
acceptance about 0.63-0.68. An earlier version of this doc said T>0 was
"effectively K1"; that was wrong. The host walk costs about 2 ms per
committed position, so the device kernel is worth about +8-18%. Sampled
drafts (section 1) would raise acceptance but lose the identity gate.
