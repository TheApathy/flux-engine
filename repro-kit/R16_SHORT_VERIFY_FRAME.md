# Shortened Verification State Boundaries

Base: `82e429db05ae80ca170a84e2e871427e45dfadb6`.

This source-only donor addresses the fixed-width K3/K4 scheduler paths after
the Qwen4 sized entries began delegating to the QSA-journaled generic verifier.
It must be composed with that delegation and journal repair, not qualified
independently as a device accuracy fix.

## Error Boundary

A shared frame captures host sequence and token extents immediately before
verifier entry. Verifier, EP-result and commit failures restore those extents,
attempt the existing QSA/recurrent abort, aggregate any rollback error into
the terminal notification, and defer resource retirement to the scheduler.
Pre-entry sync/EP-command/token errors do not abort an unstarted transaction.
Existing sink consumption, proposal clearing and failed-frame cache exclusion
remain mandatory. No cleanup retry or early free is introduced.

Hidden-save, proposer-trim and next-proposal errors after a successful commit
also notify and terminate, without trying to abort the consumed journal. K4
uses the existing active-proposer hidden-save selector instead of calling both
MTP and DFlash saves unconditionally. Partial-accept proposer trimming moves
after successful ordinary commit so terminal frames do not mutate that owner.

An abort after partially enqueued commit is best-effort: a backend commit can
already have overwritten a recurrent checkpoint. This donor does not prove
exact rollback of every device state on a backend failure. Such failures stay
terminal and non-cacheable, not retryable successful requests.

## Terminal Boundary

Previously the full-accept branches returned on EOS, cancellation or output
limits without consuming the verification journal. Partial branches committed
before emission, making an earlier terminal boundary unrecoverable.

Every branch now emits through a row-counting frame before its ordinary commit.
The terminal path uses the existing pure terminal commit plan and commits only
the classified input prefix, or aborts a zero-row frame. Terminal and ordinary
commits are mutually exclusive. The zero-row abort is attempted once even if
it fails, then notifies without retrying the journal.

Client-visible output length is not the target input count. Silent think tags,
suppressed EOS and structural tokens still pass through serial target state.
The internal emission routine now reports whether a row reached classification:
only pre-classification cancellation returns false. Existing public emission
wrappers discard this return value, preserving their behavior. Shortened
verification counts the true results, including hidden rows.

## Verification And Limits

New source/mutation tests bind capture, all branch error and terminal routes,
host restore ordering, classified-row counting and no zero-row double abort.
Nine new Rust GPU-free recording test definitions exercise errors, rollback
failure, all accept/length/EOS positions, hidden control rows, cancellation and
single terminal error delivery and postcommit proposer faults. These definitions
are NOT_RUN until a fresh
coordinated native test window compiles and executes them.

Legacy fixed-width EP workers receive nominal acceptance before root emission
and have no terminal-row/abort protocol. Root cleanup is not proof of EP-rank
state parity. Current qualification must retain the TP1/EP1 restriction.
K2 and generic/tree/batched skipped-row accounting
are separate audit scopes. No native load, accuracy, throughput, model-context,
concurrency or container readiness claim follows from these source checks.
