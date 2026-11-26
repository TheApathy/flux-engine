"""Frozen composition contract, not native execution.

Guards the fix for the "rollback skips recurrent-state restore" bug:
a watchdog rollback used
to truncate `output_tokens`/KV length (the attention rewind) without undoing
the SSM recurrent state (`h_state`/`conv_state`), so the next decoded token
saw a recurrent state conditioned on tokens the KV/position no longer
remembered.

`rollback_to_boundary` in rollback.rs now restores the SSM state from the
Phase-C decode ring (`Model::restore_decode_ssm_snapshot`) strictly BEFORE
truncating any buffer (`apply_rollback`), and declines the rollback
(`RollbackFallback::NoSsmSnapshot`) rather than truncate on a model-side
restore failure or a missing ring snapshot — see
scheduler/rollback_tests.rs for the behavioral tests. This file guards only
the source-level ordering invariant: restore-or-refuse must stay strictly
ahead of the mutation, so a future edit cannot reorder it back into a
silent partial rollback.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ROLLBACK = ROOT / 'crates/spark-server/src/scheduler/rollback.rs'
CONTINUATION = ROOT / 'crates/spark-server/src/scheduler/continuation.rs'


class RollbackRecurrentStateSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = ROLLBACK.read_text()
        cls.func = cls.src.split(
            'pub fn rollback_to_boundary(', 1
        )[1].split('\n/// Record an SSM-state snapshot', 1)[0]
        continuation = CONTINUATION.read_text()
        cls.cont_owned = continuation.split(
            'impl ContinuationRecord {', 1
        )[1].split(
            'pub fn restore_for_rollback(', 1
        )[1].split('\n    pub fn ', 1)[0]
        cls.cont = continuation.split(
            "impl<'a> ContinuationRecordMut<'a> {", 1
        )[1].split(
            'pub fn restore_for_rollback(', 1
        )[1].split('\n    pub fn ', 1)[0]

    def test_ssm_restore_or_refuse_precedes_the_buffer_mutation(self):
        func = self.func
        # The model-side restore attempt (via ContinuationRecordMut::restore_for_rollback),
        # which runs restore_decode_ssm_snapshot + ring eviction, must execute strictly
        # before the actual buffer/KV/position truncation (`apply_rollback`).
        restore_call_idx = func.index('.restore_for_rollback(')
        apply_rollback_idx = func.index('apply_rollback(a, keep_len, dropped)')
        self.assertLess(restore_call_idx, apply_rollback_idx)
        for label, body in (('ContinuationRecordMut', self.cont), ('ContinuationRecord', self.cont_owned)):
            with self.subTest(impl=label):
                restore_idx = body.index('model.restore_decode_ssm_snapshot(')
                truncate_after_idx = body.index('ssm_rollback_ring.truncate_after(keep_len)')
                self.assertLess(restore_idx, truncate_after_idx)

    def test_a_failed_restore_returns_before_any_mutation_runs(self):
        func = self.func
        # The `if let Err(fallback) = ... { return ... }` block must come
        # strictly before `apply_rollback`, and `restore_for_rollback` returns
        # `Err(RollbackFallback::NoSsmSnapshot)` after a failed
        # `restore_decode_ssm_snapshot` call strictly before `truncate_after`.
        err_return_idx = func.index('return RollbackOutcome::Fallback(fallback);')
        apply_rollback_idx = func.index('apply_rollback(a, keep_len, dropped)')
        self.assertLess(err_return_idx, apply_rollback_idx)
        for label, body in (('ContinuationRecordMut', self.cont), ('ContinuationRecord', self.cont_owned)):
            with self.subTest(impl=label):
                restore_idx = body.index('model.restore_decode_ssm_snapshot(')
                cont_err_idx = body.index('return Err(RollbackFallback::NoSsmSnapshot);', restore_idx)
                truncate_after_idx = body.index('ssm_rollback_ring.truncate_after(keep_len)')
                self.assertLess(restore_idx, cont_err_idx)
                self.assertLess(cont_err_idx, truncate_after_idx)

    def test_hybrid_boundary_search_requires_a_live_ring_snapshot(self):
        # The hybrid path must search with the snapshot-aware finder, not
        # the plain one, else it could select a boundary with no
        # restorable state at all.
        self.assertIn('find_last_boundary_with_snapshot(', self.func)
        hybrid_branch = self.func.split('let hybrid =', 1)[1].split('} else {', 1)[0]
        self.assertIn('find_last_boundary_with_snapshot(', hybrid_branch)
        self.assertNotIn('find_last_boundary(&a.output_tokens, &mask, min_keep)', hybrid_branch.split('None => {', 1)[0])


if __name__ == '__main__':
    unittest.main()
