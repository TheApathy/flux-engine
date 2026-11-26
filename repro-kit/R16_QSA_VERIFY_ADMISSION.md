# R16 QSA Verify Width And Tree Admission

Base: `82e429db05ae80ca170a84e2e871427e45dfadb6`.

This isolated source repair owns only generic verify admission in `verify_d.rs`
and its focused tests/documentation. It contains no compiler, native, GPU,
model, server, container, resource or performance result.

## Failures

The canonical `qwen4_qsa::verify_journal_rows(k)` predicate journals K5
unconditionally and physical widths2..=9 unless the diagnostic
`ATLAS_QWEN4_QSA_JOURNAL_ALL_K=0` disables those other widths. Physical widths
above9, including DFlash gamma15/K16, have no QSA undo journal. The generic
verifier previously allowed them to write QSA state despite prefix acceptance
being unable to roll back a rejected suffix.

The existing tree-frame refusal covered only K5 and ran after SSM conversion,
pre-verify snapshots, embeddings, KV allocation and metadata uploads. Other
QSA-journal widths could enter a tree frame even though QSA commit is a flat
prefix transaction, not an arbitrary branch commit.

## Admission Repair

The new pure scope predicate is exactly:

```rust
target_is_qwen4 && qsa_required && rows > 1
```

It leaves empty and single-row verification, non-Qwen4 targets and requests
that do not require QSA unchanged. Inside that scope the generic entry checks
the ACTUAL canonical `verify_journal_rows(k)` result and refuses a frame when
that result is false. This intentionally turns the unsafe all-K diagnostic
opt-out into an early error for affected QSA speculative frames. The lower
predicate, journal format, capacity, commit logic and other diagnostic callers
are unchanged. No unsupported width becomes journaled or qualified.

The same early decision refuses any active tree metadata channel:

- Nonzero `ddtree_num_tree_tokens`.
- Nonempty `ddtree_parent_ids_host`.
- Present `ddtree_parent_ids_dev`.

Checking all three prevents an inconsistent partial tree capture from passing
only because its count is zero. The real setter populates all three and the
clearer resets all three. Normal graph-safe flat-chain injection uses a LOCAL
device-parent variable later in the verifier, so it is not mistaken for an
active stored tree frame by this check. Stored forced-tree diagnostic frames
remain refused for QSA, matching the conservative existing K5 restriction.

The decision runs after the empty-frame return and before rotary-tail checks,
route receipts, legacy-K2 delegation, SSM conversion, stage capture, snapshots,
profiling, embeddings, KV allocation and H2D metadata uploads. It performs only
metadata reads and propagates admission errors. The existing K5 requirement
for the qualified batched route is retained. Non-QSA tree, row-serial,
experimental K16 and other model route selection are not changed.

The late K5-only tree guard is removed because the earlier all-width decision
now owns this policy. The canonical journal selector may initialize its
existing once-only diagnostic log during the early check. No model/GPU/cache
effect occurs before refusal.

## Executed Source Checks

The NEW `test_r16_qsa_verify_admission_source.py` has11 passing source/mutation
methods. It binds the pure policy, exact target/QSA/physical-width inputs,
all three tree fields, error propagation, pre-effect ordering, unchanged
qualified K5 restriction, canonical journal/commit predicate and real tree
setter/clearer semantics. Hostile mutations remove scope terms, substitute
K5/K16 for physical width, force journal availability, drop tree channels,
change OR to AND, swallow errors or move admission after effects. All reject.

The unchanged oracle-journal14, shortened K3/K4 delegation7 and Plain-QSA graph8
source methods also pass, totaling40 across separate processes. Six NEW pure
Rust test definitions cover the Boolean admission
matrix, non-Qwen4/non-QSA/K0/K1 scope, K5, shortened/all journal widths, absent
journaling and unsupported widths through `usize::MAX`. They are uncompiled
and unexecuted. Python source checks are not substitutes for running them.

Run each source module in a fresh bounded process:

```sh
ulimit -v 92160
ulimit -t 5
env PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/timeout --signal=KILL 50s \
  /usr/bin/prlimit --as=94371840 --cpu=5 -- \
  /usr/bin/python3 -B repro-kit/tests/test_r16_qsa_verify_admission_source.py -v
```

## Remaining Work

This fixes admission availability and flat-prefix scope in the generic entry,
not the complete speculative lifecycle. Independent review, compilation and
forced-prefix committed-state/device-failure parity remain required before
runtime qualification. Scheduler error/terminal cleanup is a separate repair.

The existing direct/diagnostic fixed-K2 verifier can bypass generic QSA undo
preparation. It is not fixed or certified by this donor. A separate isolated
fixed-K2 delegation and recursion-guard repair is assigned next. Frozen profile
/4 readers and the full native-K5 context boundary HOLD remain untouched.
