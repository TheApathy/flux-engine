# Device sample-then-match verify (design, no CUDA yet)

Companion to `stochastic-verify.md` section 0. Goal: run the T>0 verify accept
rule on the device so a frame returns `(accepted, token)` instead of copying
`γ+1` full logits rows (248,320 BF16 each on Flash-Next) to the host and
running the full sampler there per position.

## 1. Where the time goes today (code, not yet profiled)

- T=0: `fast_path_seq_eligible` (`think_spec_accept.rs`) lets both walks
  commit the device argmax with no D2H.
- T>0, thinking: one synchronous row D2H per position
  (`run_dflash_thinking_accept` in `think_spec_accept.rs`), then
  `process_seq_logits` (`think_spec_accept.rs`): widen 248k values, masks,
  penalties, then `sort_unstable_by` over every finite logit in
  `crates/spark-runtime/src/sampler/sample_impl.rs` before top-k truncates to 20.
- T>0, content: one bulk D2H of all rows, then the same per-row sampler.
- The next frame's drafts depend on the accepted count, so the GPU idles
  for the whole host loop.

Measured on GB10 at `T=1.0`:
speculation already pays at T=1.0 today, 37.7 vs 29.6 tok/s (+27%) on the
canaries, with adaptive K2-K5 and pooled acceptance about 0.63-0.68. The
host sampler costs about 1.7 ms per position (release bench, 248k vocab),
so the host loop is roughly 8% of a spec frame, up to ~18% if the in-server
cost is twice the bench. Removing it is worth +8-18%, not a step change.

## 2. Kernel contract

One CTA per (sequence, verify row); rows of a sequence are independent given
the draft prefix, and the accept scan is a tiny second pass.

Inputs per sequence:
- `logits: bf16[γ+1][V]` (the resident verify buffer, row base as today);
- `drafts: u32[γ]`;
- `params`: temperature, top_k (1..=64, required), top_p, min_p,
  repetition/presence/frequency penalty + window, logit_bias list;
- `history`: the committed tokens the penalties read (device copy of the
  window, appended with `drafts[0..i]` for row i);
- `seed: u64` (the request's effective seed; admission already materializes
  one for every T>0 request, `materialize_effective_sampling_seed_with`) and
  `pos0: u64 = output_tokens.len()` at frame start (output index, not
  `seq_len`);
- per row `row_policy` (section 6): `forced: i32` (−1 none), mask list,
  additive-bias list.

Per row i:
1. widen BF16 to f32; apply `row_policy` masks/biases; apply penalties
   against `history ++ drafts[0..i]` in the host's order
   (`prepare_sampling_distribution_impl`: repetition, presence/frequency,
   LZ, DRY, logit_bias);
2. if `forced >= 0`, the sample is `forced`;
3. else select the top_k by the total order of section 3, scale by 1/T,
   softmax in that order with the shared exp (section 4), min-p, top-p with
   the host's sequential f32 cumsum, then
   `threshold = u * total; first cumulative >= threshold`;
4. `u` = the `target`-domain coin for absolute position `pos0 + i`
   (section 5); never keyed by `i` alone.

Accept scan: `accepted = first i < γ with sample[i] != drafts[i]` (or γ);
output `token = sample[accepted]` (the correction token on a mismatch, the
bonus when all matched) and `top1_prob[0..=accepted]` (for the thinking
replay, section 6). Rows past the first mismatch are wasted work, not wrong
work: their prefix assumption was false and their samples are discarded.

Fail closed to the host walk when: top_k = 0 or > 64, top_n_sigma > 0
(needs a sequential f32 mean/sigma over V), LZ or DRY active (v1; v2 ports
them), adaptive sampling, grammar, logprobs, logits receipts,
`deepseek_v41`, or FP32 verify logits.

## 3. Host tie-order fix (prerequisite) — implemented, opt-in

`sort_unstable_by` on the logit alone is deterministic for a given input, but
the order of EQUAL logits is whatever the unstable sort leaves. No partial
select or device kernel can reproduce it, and since the seeded draw walks the
CDF in that order, it decides which of several tied tokens is sampled.

Pinned order (`spark-runtime/src/sampler/candidate_order.rs`): logit
descending, then token id ascending (`total_desc`). With a total order,
select-k-then-sort equals sort-all-then-truncate exactly, so the top-k stage
is `select_nth_unstable_by` + a sort of k entries. **This is the tie order the
device kernel must implement.**

Default stays the legacy sort (byte-identical; it cannot be reproduced by a
partial select, see `candidate_order.rs`). `FLUX_SAMPLER_TIE_ORDER=total` (receipt: `SAMPLER_TIE_ORDER_RECEIPT` at scheduler start)
switches the one sampler function process-wide, so plain decode and every
verify walk still agree with each other. Versus legacy, the opt-in path
produces a bit-identical weight sequence (so the same CDF position for every
draw); the sampled id can differ only inside a group of exactly equal logits
(tested over 1e4 tie-heavy rows; identical tokens on rows without ties).
Making it the default changes seeded outputs once on such ties.

## 4. Floating-point parity

- `exp`: Rust `f32::exp` (libm) and CUDA `expf`/`__expf` differ by ulps, which
  moves CDF boundaries. Use one shared exp: a fixed polynomial implemented in
  Rust and CUDA with explicit operation order and no FMA contraction
  (`__fmul_rn`/`__fadd_rn`). The host switches to it only under the same
  opt-in as the total tie order (default keeps `f32::exp`, byte-identical);
  9.15.7 may flip both together with a release note and an identity-gate
  re-baseline. Not implemented yet.
- Sums over ≤ 64 kept entries: one thread, sequential, same order as host.
- Penalty pass is elementwise; `max` is order-free. Full-vocab sums only
  appear in top-n-sigma and entropy metrics; the first is fail-closed, the
  second is host-only telemetry.

## 5. RNG

### Invariant (must hold for every option)

Every coin is keyed by **(request seed, committed absolute output position,
draw-domain tag)** and by nothing else: never by frame start, window column,
draft depth, batch slot or acceptance history. Then spec and no-spec consume
the same coin for the same output position, whatever K is and wherever the
frame boundaries fall. A key like `hash(seed, frame_seq_len, column)` breaks
this: when an earlier rejection moves the frame boundary, the same absolute
position becomes a different (frame, column) pair and gets a different coin.
(SGLang's `hash(seed, seq_len, column)` is a stateless mapping, not a proof
of window-boundary equivalence.)

Today's host already satisfies it: `ActiveSeq::sampling_params`
(sample_step.rs) sets the draw seed to `request_seed + output_tokens.len()`,
the committed output index, NOT `seq.seq_len` (which includes the prompt and
rows of the verify frame not yet settled). The plain sampler and both verify
walks read it at commit time, so the position the device kernel must use is
`output_tokens.len()` at frame start plus the row index. Test:
`same_seed_commits_identical_tokens_at_draft_depth_1_and_5` (100 seeds,
K=1 vs K=5 with mostly-wrong drafts, identical committed tokens).

### Draw domains

| Domain | Flux draw | Used by |
|---|---|---|
| `target` | the plain sampler's draw at position n | plain decode, sample-then-match verify (every row incl. correction/bonus) |
| `accept` | accept coin `u < p/q` | p/q extension only (sampled drafts) |
| `residual` | residual-distribution draw after a rejection | p/q extension only |
| `draft` | drafter's own sample | only if the MTP head ever samples (today: argmax, no coin) |

Sample-then-match uses only the `target` domain, so for greedy drafts it is
the one coin plain decode uses: that is what makes the identity gate hold.

### Options

- **Recommended for the kernel: SGLang-style hashed coins** (following SGLang
  `speculative/eagle_utils.py#L453-L518, L776-852`):
  `u = murmur_hash32(seed, position, domain)` → FP64 → FP32, clamped to
  `1 − 2^−24`, with `position` = committed absolute output index and
  `domain` from the table. Stateless, a few integer ops, trivially identical
  on host and device. Cost: every seeded output changes once. Ship it behind
  the same opt-in as the tie order (`FLUX_SAMPLER_TIE_ORDER=total` selects
  the total order and these coins together, or a sibling
  `FLUX_SAMPLER_COINS=hashed`), flip the default in 9.15.7 with a release
  note and an identity-gate re-baseline.
- **No-output-change alternative: port ChaCha12.** Today's draw is
  `StdRng::seed_from_u64(seed + position)` first `gen::<f32>()` (rand 0.8.6:
  PCG32 seed expansion to a ChaCha12 key, `(word >> 8)·2^−24`). Porting it
  keeps every seeded output as is (apart from the tie order and exp). Cost:
  ~1 extra day of kernel work and validation, a hard pin of rand 0.8.6 for
  the server sampler (StdRng's algorithm is not stable across rand
  versions), and the existing `seed + position` key, which collides across
  requests with adjacent seeds (harmless within a request).

## 6. Thinking channel

Thinking is where most tokens are, so v1 must cover it. What
`dflash_thinking_accept` (`think_spec_accept.rs:168-300`) does per position
beyond the sampler, and how each maps to the kernel:

| Intervention | Depends on | Device mapping |
|---|---|---|
| Reflection penalty −10 (`decode_logits_seq.rs` F1) | inside_thinking | `row_policy` additive bias list |
| Min-thinking floor mask on `</think>` | thinking_tokens | `row_policy` mask, precomputed per row |
| Forced `</think>` (budget / F2 armed; deferred in code fence, `forced_injection_pending`) | thinking_tokens, budget, in_code_fence, force_end_thinking | `row_policy.forced` |
| Nudge injection (`think_nudge::forced_token`) | nudge state, fence | `row_policy.forced` |
| F2 confidence early-stop (top-1 ≥ 0.95 for 30 tokens past 400) | the target row's top-1 prob | **data-dependent**: host replay |
| Decoding-efficiency wave, adaptive sampling | env, off by default | fail closed |

Key observation: every state input except F2 evolves only from the committed
tokens, and within a frame the committed prefix up to the first mismatch is
exactly `drafts[0..i]`. So the host can precompute `row_policy[i]` for all
rows by simulating the per-token bookkeeping (`commit_thinking_token`:
thinking_tokens, fence parity, budget arming, nudge cursor) over the drafts,
without logits. Rows after a mismatch get wrong policies but are discarded.

F2: the kernel returns `top1_prob` per row; the host replays
`consecutive_confident` over the accepted rows. If F2 arms at row j < accepted,
plain decode would have forced `</think>` at row j+1 (unless deferred by a
fence); the host truncates `accepted` to j+1, commits `</think>` there itself
(deterministic, no row needed) and discards the rest. Already-emitted
`</think>`/EOS stay frame boundaries as today.

Smallest change to admit thinking:
1. extract a pure `thinking_row_policy(a_snapshot, drafts) -> Vec<RowPolicy>`
   from `process_seq_logits`'s thinking branch plus a dry-run of
   `commit_thinking_token`'s counters, with a test that it equals what the
   host walk applies row by row (feed logits, compare masks);
2. a host `replay_f2(top1_prob, accepted)` and truncation;
3. route `run_dflash_thinking_accept_at_row` to the kernel when eligible,
   then call `commit_thinking_token` for each committed token unchanged (it
   has no vocab work, so it stays on the host).

## 7. Validation

- Host walk is the bit-exact reference. Harness: 1e4 seeded frames from
  recorded verify logits (dump hook already exists for prefill qualification)
  plus synthetic tie-heavy rows; params grid T ∈ {0.6, 1.0}, top_k ∈ {20, 64},
  top_p ∈ {0.8, 0.95, 1.0}, penalties on/off; drafts sampled from the plain
  path (mostly correct) and perturbed (wrong at a random position); thinking
  rows with budget/fence/nudge/F2 states. Gate: `(accepted, token)` and every
  `sample[i]` for i ≤ accepted identical, 0 mismatches.
- End-to-end: seeded spec vs nospec byte-identical on the thinking panels at
  T=1.0 (the identity gate), then accuracy panels and a decode A/B.

## 8. Payoff model

Per-position acceptance `a` (geometric), K drafts:
tokens/frame `E(K) = (1 − a^{K+1}) / (1 − a)`.

| a | E(1) | E(3) | E(5) |
|---|---|---|---|
| 0.5 | 1.50 | 1.88 | 1.97 |
| 0.6 | 1.60 | 2.18 | 2.38 |
| 0.7 | 1.70 | 2.53 | 2.94 |
| 0.8 | 1.80 | 2.95 | 3.69 |

Speedup over plain = `E(K) · (S + h) / (F_K + c·h')` with S the plain step,
F_K the verify frame, h the host sampling time per plain token and `c·h'` the
small host replay left after the kernel. With the Flash-Next K5H numbers in
memory (S ≈ 24 ms, F_5 ≈ 79.5 ms) and h → 0: speedup ≈ 0.30·E(5), i.e.
1.11x at a = 0.8 and 0.72x at a = 0.6. So K5 only pays at T=1 if acceptance
stays ≥ ~0.75; adaptive depth (shorter frames when a drops) and a cheaper
F_K matter as much as the kernel. Measure `a` at T=1.0 on the thinking arms
before committing to K5 (one host-only receipt run).

## 9. Effort

| Piece | Days |
|---|---|
| Measure h and a at T=1.0 (receipts, no kernel) | 0.5 |
| Host tie-order fix + select_nth top-k (done, opt-in) | 0 |
| Shared exp on host and device, behind the same opt-in, re-baselined at 9.15.7 | 1 |
| Hashed coins (recommended) or ChaCha12 port (+1 day) | 0.5 / 1.5 |
| Kernel v1 (penalties without LZ/DRY, top-k ≤ 64, ChaCha12 port, accept scan) | 4-5 |
| Thinking row policy + F2 replay + routing | 2-3 |
| Bit-exact harness (1e4 frames) + identity gate + panels | 2-3 GPU |
| LZ/DRY on device (v2, only if the preset uses them) | 2-3 |

Risks: ulp-level parity (exp, sums) breaking the identity gate (fallback:
qualify by distribution); the F2 replay boundary; penalty history on device
for long windows; acceptance at T=1 too low for K5 to pay regardless.
