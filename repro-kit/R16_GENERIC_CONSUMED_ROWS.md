# R16 Generic, Tree And Batched Classified-Row Ownership

Base: `70370490d8ff8701c4a128169f04759bd24b26e0`, tree
`953950c1bd318f83fb589ed20fb2594e5b5bc959`.

This isolated source donor extends the base's classified-row convention from
fixed K3/K4 to the generic/tree and batched terminal reconciliation paths. It
does not include the separate startup, QSA-width admission or fixed-K2 donors.
Those remain independently composed and reviewed changes.

## Failure And Repair

`reconcile_terminal_verify_state` previously derived retained verifier inputs
from the change in `output_tokens.len()`. A classified `</think>`, silent think
transition, suppressed EOS or implicit thinking close can advance the serial
input prefix without appearing in that output vector. If another row then
terminates the frame, visible output undercounts the state to retain. Tree
frames also risk selecting an earlier intermediate than their semantic path
requires.

The reconciliation helper now receives explicit `consumed_rows`. It passes
`(0, consumed_rows)` through the unchanged pure terminal-plan interface, as
`ShortVerifyFrame` already does. The plan retains its width, accepted-prefix,
current host extent, checked arithmetic and semantic tree-index validation.
Token rebuilding, KV compaction, QSA/recurrent commit and zero-input abort are
unchanged. A classified final output remains deliberately one token ahead of
target state, matching ordinary decode.

Generic ordinary/tree emission and batched finalization each count only true
returns from the canonical `emit_token_for_verify` helper. Hidden structural
rows count, classified terminal rows count, and cancellation before the
classifier does not. All three generic terminal calls and both batched calls
use the appropriate explicit counter. Visible output remains response data,
not ownership evidence.

`SpecAcceptOutcome` adds `consumed_rows`, separate from accepted drafts and the
bonus. All sixteen outcome constructors carry their prior classified count,
or explicit zero for failures before a walk starts. Both policy walks count
after successful classification and before acceptance or terminal branching.
On denied admission they return immediately with `bonus: None` and the prior
accepted/classified counts. A sampled but unclassified row cannot become a
phantom accepted draft or bonus. Existing successful sampler choices, phase
boundaries and accepted-draft arithmetic remain unchanged.

The thinking commit helper now reports classification. Its direct hidden-EOS
branch checks cooperative cancellation before counter, watchdog or implicit
close effects, then reports true even without visible output. Ordinary and
closing rows preserve the canonical emitter's Boolean. Thinking-close
confidence/fence resets follow successful emitter admission, including its
second cancellation check, so a racing cancellation cannot perform those
resets for an unclassified row.

## External Callers And Scope

`sampled_dflash.rs` is byte unchanged. Its `emit_sampled_token` Boolean means
deferred thinking-boundary snapshot, NOT classified-row consumption. It still
ignores the thinking helper's new return, as does the existing bootstrap
caller. This donor does not repair or qualify either external lifecycle.
Successful snapshot ordering and stop/thinking/grammar selectors are retained.

This patch does not redesign EP result messages or prove cross-rank terminal
state parity. It does not repair batched failure rollback, model-side journals,
backend commit failures, sampled DFlash transactions or native concurrency.
Sampling can still update difficulty/F2 state before a racing cancellation is
observed by emission. This fixes target ownership accounting, not all sampler
side effects under cancellation. Source correctness is not model accuracy,
long-context, performance, binary or container qualification.

## Executed Source Checks

All checks run offline in fresh processes with 90 MiB address-space and
five-second CPU limits. No Rust/compiler/native/GPU/model/server/container,
resource claim, lock, cache or host-maintenance action occurs.

- NEW classified-row source/mutation suite: 11 PASS.
- Unchanged short-frame source suite: 8 PASS.
- Unchanged shortened K3/K4 source suite: 7 PASS.
- Unchanged terminal-error source functions: 8 PASS.
- Unchanged QSA oracle-journal source suite: 14 PASS.
- Unchanged Plain-QSA graph source suite: 8 PASS.
- Total: 56 source checks. Whitespace check: PASS.

Mutations substitute visible counters, use the wrong outcome field, remove or
zero increments, bypass classifier admission, drop or invent constructor
counts, admit phantom cancelled bonuses and move thinking-close effects back
ahead of admission. They are rejected. The first mutation run exposed a test
fixture targeting a helper argument instead of a constructor; it was corrected
to mutate actual constructor fields. That initial failed run is not a PASS.

Ten NEW GPU-free Rust regression definitions cover hidden row plus terminal,
suppressed EOS, cancellation after hidden progress, zero-input cancellation,
semantic tree slots, batched finalization, silent thinking close, cancelled
direct EOS, cancellation between sampling and emission, and partial-progress
download failure. They are defined and registered but UNCOMPILED and NOT_RUN.

Example bounded source command:

```sh
ulimit -v 92160
ulimit -t 5
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  repro-kit/tests/test_r16_generic_consumed_rows_source.py -v
```

Independent pinned review and compilation remain required before integration
can be treated as native-qualified. Forced-prefix raw QSA/recurrent/PLE state
parity, cancellation/failure injection, accuracy and exact-container gates
remain open.
