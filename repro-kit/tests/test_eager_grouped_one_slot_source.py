"""Frozen composition contracts and independent references, not native execution."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'crates/spark-runtime/src'


def strict_selector(value):
    if value is None or value == '0':
        return False
    if type(value) is str and value == '1':
        return True
    raise ValueError('strict selector')


def eager_scope(group_value, eager_value, *, tp=1, ep=1, tp_rank=0, ep_rank=0,
                speculative=False, dflash=False, self_speculative=False, ngram=False):
    grouped, eager = strict_selector(group_value), strict_selector(eager_value)
    if eager and not grouped:
        raise ValueError('eager requires groups')
    if grouped and (tp != 1 or ep != 1 or tp_rank != 0 or ep_rank != 0
                    or self_speculative or ngram):
        raise ValueError('Plain, native-MTP, or DFlash TP1/EP1 only')
    return grouped, eager


def materialize_reference(count, fail_at=None, failure=None):
    """Tiny ownership reference only. No Rust, driver, real allocation or DMA."""
    registered, claimed, events = [], [], []
    for group in range(count):
        events.append(('alloc', group))
        if group == fail_at and failure == 'alloc':
            return registered, claimed, events
        events.append(('register', group))
        if group == fail_at and failure == 'register':
            events.append(('ordinary-free-attempt', group))
            return registered, claimed, events
        registered.append(group)
        events.append(('claim', group))
        if group == fail_at and failure == 'claim-panic':
            # The registered owner is retained, with no rollback/free attempt.
            return registered, claimed, events
        if group == fail_at and failure in ('claim', 'ordinary-free-error'):
            registered.pop()
            events.extend([('unregister', group), ('ordinary-free-attempt', group)])
            return registered, claimed, events
        claimed.append(group)
        events.append(('publish-base', group))
    return registered, claimed, events


class EagerCompositionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = (RUNTIME / 'fast_weights/mod.rs').read_text()
        cls.plan = (RUNTIME / 'fast_weights/routed_groups.rs').read_text()
        cls.owner = (RUNTIME / 'weights/routed_backing.rs').read_text()
        cls.table = (RUNTIME / 'cuda_backend/allocation_lifetime.rs').read_text()
        cls.bounce = (RUNTIME / 'fast_weights/bounce_copy.rs').read_text()
        cls.serve = (ROOT / 'crates/spark-server/src/main_modules/serve_phases/weights.rs').read_text()

    def test_production_donor_bytes_and_accepted_one_slot_copier_are_exact(self):
        pins = {
            # Re-pinned after Stage 3B truth-audit fixes and Stage 4 workspace cargo fmt.
            'crates/spark-runtime/src/fast_weights/mod.rs': '182f9e15feecafb49bc113716d7e0fc8a5360a7e8b3721050b71963971dfcb46',
            'crates/spark-runtime/src/fast_weights/routed_groups.rs': '6c196b7b1c0f8984c32305c648787e123e9ff360cb88436b6c0a7f858eb6c4da',
            'crates/spark-runtime/src/weights/routed_backing.rs': 'b708af5074a59f4db9b7e2d2f14c4658cab5a64a2499fcb32151129afc38859e',
            'crates/spark-server/src/main_modules/serve_phases/weights.rs': '4824cf485d34c51278197cd93e730f67fe6b924b124e2818ffd767d8970f83d7',
            'crates/spark-runtime/src/fast_weights/bounce_copy.rs': 'ed78ce0bfcfb34506e1bf3994c8d63d3029f6ee2d2d96c5e8b09cba28abe3791',
            'crates/spark-runtime/src/fast_weights/bounce_copy_tests.rs': '82ae42e6e9485f6b9e7f4fab2209756ac90e22763e20c6fb7d91581ff3a9fc16',
        }
        for path, expected in pins.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_eager_has_no_selector_or_speculation_override_bypass(self):
        self.assertEqual(eager_scope(None, None), (False, False))
        self.assertEqual(eager_scope('1', '0'), (True, False))
        self.assertEqual(eager_scope('1', '1'), (True, True))
        for invalid in ('', 'true', '01', ' 1', '1\n', '\uff11', b'1', b'\xff', True):
            for group, eager in ((invalid, '1'), ('1', invalid)):
                with self.subTest(group=group, eager=eager), self.assertRaises(ValueError):
                    eager_scope(group, eager)
        for group in (None, '0'):
            with self.assertRaises(ValueError):
                eager_scope(group, '1')
        self.assertEqual(eager_scope('1', '1', speculative=True), (True, True))
        self.assertEqual(eager_scope('1', '1', dflash=True), (True, True))
        for key, value in (('tp', 2), ('ep', 2), ('tp_rank', 1), ('ep_rank', 1),
                           ('self_speculative', True), ('ngram', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                eager_scope('1', '1', **{key: value})
        gate = self.serve.split('let grouped_routed_originals = {', 1)[1].split('let store =', 1)[0]
        self.assertNotIn('ATLAS_DISABLE_SPECULATION', gate)
        self.assertLess(gate.index('if enabled'), gate.index('routed_grouping_eager_requested('))

    def test_eager_materialization_precedes_payload_loop_but_not_geometry_preflight(self):
        load = self.loader.split('fn load(', 1)[1]
        order = ['RoutedBackingLoad::plan(', 'if peak.saturating_add(oom_reserve_bytes) > free {',
                 'if self.grouped_routed_originals_eager {', 'load.allocate_all(gpu)?',
                 'for (i, shard_path) in shard_files.iter().enumerate() {']
        positions = [load.index(fragment) for fragment in order]
        self.assertEqual(positions, sorted(positions))
        block = load[positions[2]:positions[4]]
        block_ns = ''.join(block.split())
        self.assertIn('routed_load.as_mut()', block_ns)
        self.assertIn('eager routed groups require ATLAS_FAST_LOAD_ROUTED_GROUPS=1', block)
        self.assertNotIn('alloc_managed', block)

    def test_exact_cuda_claim_errors_precede_its_only_state_transition(self):
        claim = self.table.split('pub(super) fn claim_grouped(', 1)[1].split('/// Keep', 1)[0]
        claim_ns = ''.join(claim.split())
        self.assertLess(claim_ns.index('live.get_mut(&base)'), claim_ns.index('allocation.grouped=true'))
        self.assertLess(claim_ns.index('allocation.pending_h2d==0'), claim_ns.index('allocation.grouped=true'))
        self.assertEqual(claim.split('allocation.grouped = true;', 1)[1].strip(), 'Ok(())\n}')

    def test_register_claim_rollback_and_panic_owner_retention_are_not_destructor_frees(self):
        materialize = self.plan.split('fn materialize(', 1)[1].split('pub(super) fn allocate_all(', 1)[0]
        self.assertEqual(materialize.count('gpu.alloc(bytes)?'), 1)
        self.assertEqual(materialize.count('release_unclaimed(gpu, base, error)'), 2)
        self.assertLess(materialize.index('registry.unregister_unclaimed(base)?'),
                        materialize.index('self.groups[group_index].base = Some(base)'))
        drop = self.plan.split('impl Drop for RoutedBackingLoad', 1)[1]
        self.assertIn('std::mem::forget(registry)', drop)
        self.assertNotIn('gpu.free', drop)
        registry_drop = self.owner.split('impl Drop for RoutedGroupRegistry', 1)[1].split('/// Shared', 1)[0]
        self.assertIn('State::Released', registry_drop)
        self.assertIn('std::mem::forget', registry_drop)
        self.assertNotIn('gpu.free', registry_drop)

    def test_reference_failed_prefix_never_frees_previously_claimed_bases_or_retries(self):
        for failure in ('alloc', 'register', 'claim', 'ordinary-free-error', 'claim-panic'):
            for index in range(3):
                registered, claimed, events = materialize_reference(3, index, failure)
                with self.subTest(failure=failure, index=index):
                    self.assertEqual(claimed, list(range(index)))
                    expected_registered = list(range(index + (failure == 'claim-panic')))
                    self.assertEqual(registered, expected_registered)
                    self.assertNotIn(('alloc', index + 1), events)
                    frees = [group for event, group in events if event == 'ordinary-free-attempt']
                    self.assertEqual(frees, [index] if failure in ('register', 'claim', 'ordinary-free-error') else [])
                    self.assertFalse(set(frees) & set(claimed))
                    self.assertNotIn(('publish-base', index), events)

    def test_one_slot_dma_still_retains_base_and_reuses_only_after_wait(self):
        once = self.bounce.split('fn copy_once(', 1)[1].split('pub(super) fn drain(', 1)[0]
        self.assertIn('let mut slot_idx = 0usize;', once)
        self.assertLess(once.index('wait_slot_bounded('), once.index('copy_from_slice('))
        self.assertIn('.retain_h2d_destination(owning_base, destination_offset, len)',
                      ' '.join(once.split()))
        self.assertEqual(once.count('destination: owning_base'), 2)
        self.assertNotIn('destination: chunk_dst', once)
        self.assertIn('if largest_payload > 0 && largest_payload <= CHUNK_BYTES', self.bounce)


if __name__ == '__main__':
    unittest.main()
