# SPDX-License-Identifier: AGPL-3.0-only
"""Offline registration, recipe, broker and independent evidence regressions."""
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
import r15_exact_image_release as Q


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    loader.exec_module(result)
    return result


S = load(KIT / 'r15-container-suite.py', '_native_context_test_suite')
C = Q._native_context_required_module()
W = load(KIT / 'tests/test_native262k_w4096_owner16.py', '_native_context_w4096_fixture')
P = load(KIT / 'tests/test_r15_container_suite.py', '_native_context_plan_fixture')
F = load(KIT / 'tests/profile_fixture_v2.py', '_native_context_profile_fixture')
N = load(KIT / 'r15-gates/native-262k', '_native_context_w16384_driver')
GATE = 'native-262k-w4096-owner16'
CONTAINER = 'c' * 64


def contract():
    return {'schema': S.NATIVE_CONTEXT_SCHEMA, 'gates': copy.deepcopy(S.NATIVE_CONTEXT_RECIPES)}


def launch():
    return {'schema': S.PLAN_SCHEMA_V4, 'native_context_contract': contract(),
            'in_container_binary_path': '/opt/flux/bin/flux-server-real',
            'in_container_binary_sha256': '1' * 64,
            'in_container_recipe_path': '/opt/flux/repro-kit/recipes/release.env',
            'recipe_sha256': '2' * 64}


def image():
    return {key: value for key, value in launch().items()
            if key not in ('schema', 'native_context_contract')}


def proof(gate):
    return {'schema': S.NATIVE_CONTEXT_PROOF, 'status': 'PASS', 'mode': 'none',
            'gate': gate, 'container_id': CONTAINER, 'run_id': 'offline-fixture',
            'host_pid': 31415, 'start_time_ticks': 814,
            'executable_sha256': '1' * 64,
            'recipe_sha256': S.NATIVE_CONTEXT_RECIPES[gate]['recipe_sha256'],
            'chunk_tokens': S.NATIVE_CONTEXT_RECIPES[gate]['chunk_tokens'],
            'native_context_contract_sha256': Q.sha256_bytes(Q.canonical_bytes(contract())),
            **{key: True for key in ('recipe_environment_verified', 'mount_namespace_match',
                                    'cgroup_match', 'paths_read_only')}}


def job(gate, root, runtime, suite_sha='1' * 64):
    authority = C.required_authority(gate)
    path = str(root / authority['program'][0])
    options = [*C.IDENTITY_OPTIONS, ('--runtime-dir', str(runtime)),
               ('--suite-program', str(root / C.SUITE_RELPATH)),
               ('--suite-sha256', suite_sha),
               *[(flag, str(root / rel)) for flag, rel, sha in authority['helpers']]]
    return {'program_path': path, 'program_sha256': authority['program'][1],
            'argv': [path, *[part for pair in options for part in pair]],
            'timeout_seconds': 86400,
            **{key: prefix + '/' + gate + suffix for key, prefix, suffix in (
                ('receipt_relpath', 'gates', '.json'), ('stdout_relpath', 'stdout', '.log'),
                ('stderr_relpath', 'stderr', '.log'))}}


class Offline(unittest.TestCase):
    def setUp(self):
        binding = mock.patch.dict(sys.modules, {Q.__name__: Q})
        binding.start()
        self.addCleanup(binding.stop)
        for name in ('run', 'Popen'):
            patch = mock.patch.object(subprocess, name, side_effect=AssertionError('process boundary forbidden'))
            patch.start()
            self.addCleanup(patch.stop)


class RecipeTests(Offline):
    def test_sealed_native_validator_uses_this_fixture_wrapper_module(self):
        self.assertIs(Q._native_context_required_module().release, Q)

    def test_both_gates_are_required_with_distinct_recipes(self):
        self.assertEqual(S.REQUIRED_GATES, Q.REQUIRED_GATES)
        self.assertEqual(len(S.REQUIRED_GATES), 12)
        for module in (S, Q):
            self.assertIn('native-262k', module.REQUIRED_GATES)
            self.assertIn(GATE, module.REQUIRED_GATES)
            self.assertEqual(module.validate_native_context_contract(contract()), contract())
        self.assertNotEqual(S.NATIVE_CONTEXT_RECIPES['native-262k']['recipe_sha256'],
                            S.NATIVE_CONTEXT_RECIPES[GATE]['recipe_sha256'])

    def test_missing_swapped_unknown_and_type_drift_fail(self):
        mutations = [lambda c: c['gates'].pop('native-262k'),
                     lambda c: c['gates'].update(extra={}),
                     lambda c: c['gates'][GATE].update(chunk_tokens=True),
                     lambda c: c['gates'][GATE].update(chunk_tokens=16384),
                     lambda c: c['gates'][GATE].update(recipe_sha256='1' * 64),
                     lambda c: c.update(schema='flux-native-context-recipe-contract/2')]
        for mutate in mutations:
            value = contract(); mutate(value)
            for module in (S, Q):
                with self.subTest(module=module.__name__, value=value), self.assertRaises(ValueError):
                    module.validate_native_context_contract(value)

    def test_each_recipe_plan_is_independent_of_default_image_recipe(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            selected = S.native_context_recipe_plan(launch(), image(), gate)
            self.assertEqual(selected['recipe_sha256'], S.NATIVE_CONTEXT_RECIPES[gate]['recipe_sha256'])
            self.assertEqual(selected['chunk_tokens'], S.NATIVE_CONTEXT_RECIPES[gate]['chunk_tokens'])
            self.assertEqual(selected['in_container_binary_sha256'], image()['in_container_binary_sha256'])
            self.assertEqual(selected['in_container_recipe_path'], '/opt/flux/' + S.NATIVE_CONTEXT_RECIPES[gate]['recipe_relpath'])

    def test_old_plan_and_different_image_cannot_select_recipe(self):
        old = launch(); old['schema'] = S.PLAN_SCHEMA_V3
        changed = image(); changed['in_container_binary_sha256'] = '9' * 64
        for args in ((old, image(), GATE), (launch(), changed, GATE), (launch(), image(), 'foreign')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                S.native_context_recipe_plan(*args)

    def test_native_request_is_bound_to_gate_recipe_mode_and_pid(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            selected = S.native_context_recipe_plan(launch(), image(), gate)
            request = S.native_context_request(selected, gate, 42)
            self.assertEqual(S.validate_native_context_request(request, selected), gate)
            for key, value in (('gate', 'foreign'), ('mode', 'native-mtp-k5'), ('host_pid', True), ('host_pid', 0)):
                changed = dict(request); changed[key] = value
                with self.subTest(gate=gate, key=key), self.assertRaises(ValueError):
                    S.validate_native_context_request(changed, selected)
            for key, value in (('recipe_sha256', '9' * 64), ('chunk_tokens', 8192),
                               ('in_container_recipe_path', '/tmp/recipe.env')):
                changed = dict(selected); changed[key] = value
                with self.subTest(gate=gate, key=key), self.assertRaises(ValueError):
                    S.validate_native_context_request(request, changed)

    def test_live_geometry_refuses_wrong_wide_chunk_capacity_kv_and_arena(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            selected = S.native_context_recipe_plan(launch(), image(), gate)
            chunk = str(selected['chunk_tokens']).encode()
            env = {b'ATLAS_QWEN4_PREFILL_WIDE': chunk, b'FLUX_QWEN4_PREFILL_WIDE': chunk}
            argv = [b'engine', b'--qwen4-qsa', b'--max-prefill-tokens', chunk,
                    b'--max-seq-len', b'262144', b'--kv-cache-dtype', b'bf16']
            S.validate_native_context_live_geometry(selected, argv, env)
            bad_envs = [{**env, b'ATLAS_QWEN4_PREFILL_WIDE': b'8192'},
                        {**env, b'FLUX_MAX_BATCH_TOKENS': b'1'}]
            bad_argvs = [argv[:3] + [b'8192'] + argv[4:],
                         argv[:5] + [b'135168'] + argv[6:], argv[:-1] + [b'nvfp4'], argv + [b'--qwen4-qsa']]
            for a, e in [(argv, e) for e in bad_envs] + [(a, env) for a in bad_argvs]:
                with self.subTest(gate=gate, argv=a, env=e), self.assertRaises(ValueError):
                    S.validate_native_context_live_geometry(selected, a, e)

    def test_plan4_load_requires_all_twelve_jobs_and_native_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = P.make_plan(temp)
            plan['schema'] = S.PLAN_SCHEMA_V4
            plan['native_context_contract'] = contract()
            plan['performance_profile_contract'] = F.contract(Q, plan['performance_profile_contract'], plan['in_container_binary_path'])
            self.assertEqual(set(P.load_plan(plan)['gate_jobs']), set(S.REQUIRED_GATES))
            for gate in ('native-262k', GATE):
                changed = copy.deepcopy(plan); del changed['gate_jobs'][gate]
                with self.subTest(missing=gate), self.assertRaises(ValueError):
                    P.load_plan(changed)
            for schema in (S.PLAN_SCHEMA, S.PLAN_SCHEMA_V3):
                changed = copy.deepcopy(plan); changed['schema'] = schema
                with self.subTest(schema=schema), self.assertRaises(ValueError):
                    P.load_plan(changed)

    def test_native_recipe_loader_reads_sealed_parent_and_exact_image_recipe(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            row = S.NATIVE_CONTEXT_RECIPES[gate]
            payload = (KIT.parent / row['recipe_relpath']).read_bytes()
            document = {'Id': CONTAINER, 'Image': 'sha256:' + 'a' * 64,
                        'State': {'Running': True, 'Pid': 4141},
                        'HostConfig': {'ReadonlyRootfs': True}}
            args = types.SimpleNamespace(container_id=CONTAINER, image_id=document['Image'],
                lifecycle=types.SimpleNamespace(docker=lambda command: Q.canonical_bytes(document)))
            with (mock.patch.object(S.os, 'getppid', return_value=2222),
                  mock.patch.object(Path, 'read_bytes', return_value=b'python\0--plan-fd\0' + b'5\0'),
                  mock.patch.object(S.os, 'open', side_effect=[11, 12]) as opened,
                  mock.patch.object(S.os, 'close'),
                  mock.patch.object(S, 'load_plan', return_value=launch()) as sealed,
                  mock.patch.object(S, '_open_process_file', return_value=13) as recipe,
                  mock.patch.object(S, '_hash_regular_fd', return_value=(row['recipe_sha256'], payload))):
                selected, environment = S.load_native_context_recipe(args, image(), gate)
            sealed.assert_called_once_with(11)
            self.assertEqual(opened.call_args_list[0].args[0], '/proc/2222/fd/5')
            recipe.assert_called_once_with(12, '/opt/flux/' + row['recipe_relpath'])
            self.assertEqual(environment['ATLAS_QWEN4_PREFILL_WIDE'], str(row['chunk_tokens']))
            self.assertEqual(selected['native_gate'], gate)

    def test_native_recipe_loader_refuses_parent_change_before_image_access(self):
        docker = mock.Mock(side_effect=AssertionError('image access must not occur'))
        args = types.SimpleNamespace(lifecycle=types.SimpleNamespace(docker=docker))
        with (mock.patch.object(S.os, 'getppid', side_effect=[2222, 3333]),
              mock.patch.object(Path, 'read_bytes', return_value=b'python\0--plan-fd\0' + b'5\0'),
              mock.patch.object(S.os, 'open', return_value=11), mock.patch.object(S.os, 'close'),
              mock.patch.object(S, 'load_plan', return_value=launch())):
            with self.assertRaisesRegex(ValueError, 'parent changed'):
                S.load_native_context_recipe(args, image(), GATE)
        docker.assert_not_called()

    def test_independent_validator_hash_is_checked_before_execution(self):
        with (mock.patch.object(Q, '_open_stable', return_value=(7, b'raise AssertionError("executed")')),
              mock.patch.object(Q.os, 'close')):
            with self.assertRaisesRegex(ValueError, 'source seal differs'):
                Q._native_context_required_module()

    def test_versioned_example_has_both_exact_native_jobs(self):
        plan = json.loads((KIT / 'examples/r15-container-suite-plan.example.json').read_text())
        self.assertEqual(plan['schema'], S.PLAN_SCHEMA_V4)
        self.assertEqual(set(plan['gate_jobs']), set(S.REQUIRED_GATES))
        Q.validate_native_context_contract(plan['native_context_contract'])
        Q._profile_contract(plan['performance_profile_contract'], plan)
        root = Path(plan['source_root_path'])
        for gate in S.NATIVE_CONTEXT_RECIPES:
            value = plan['gate_jobs'][gate]
            C.validate_required_gate_job(gate, value, source_root=root, runtime_dir=plan['runtime_dir'])
            source = KIT.parent / Path(value['program_path']).relative_to(root)
            self.assertEqual(Q.sha256_file(source), value['program_sha256'])


class LifecycleTests(Offline):
    def matrix(self, gate, drift=None):
        calls = {'inspect': [], 'select': []}
        default_recipe = {'DEFAULT_RELEASE_CONTROL': 'fixture'}

        def inspect(*args):
            calls['inspect'].append(None)
            snapshot = (image(), dict(default_recipe))
            if len(calls['inspect']) == 2:
                if drift == 'image-plan':
                    snapshot[0]['in_container_binary_sha256'] = '9' * 64
                elif drift == 'image-recipe':
                    snapshot[1]['DEFAULT_RELEASE_CONTROL'] = 'changed'
            return snapshot

        def select(args, image_plan, name):
            self.assertEqual(name, gate)
            self.assertEqual(image_plan, image())
            self.assertEqual(len(image_plan), 4)
            selected = S.native_context_recipe_plan(launch(), image_plan, name)
            self.assertEqual(len(selected), 7)
            row = S.NATIVE_CONTEXT_RECIPES[name]
            payload = (KIT.parent / row['recipe_relpath']).read_bytes()
            environment = {k.decode(): v.decode() for k, v in S._recipe_environment(payload).items()}
            self.assertNotEqual(environment, default_recipe)
            calls['select'].append(copy.deepcopy(selected))
            if len(calls['select']) == 2:
                if drift == 'selected-plan':
                    selected['recipe_sha256'] = '9' * 64
                elif drift == 'selected-recipe':
                    environment['ATLAS_QWEN4_PREFILL_WIDE'] = '8192'
                elif drift == 'sealed-parent':
                    raise ValueError('native context parent changed')
            return selected, environment

        driver = N if gate == 'native-262k' else W.G
        receipt, state = W.MatrixTests.run_matrix(self, gate_module=driver,
            native_recipe_loader=select, image_inspector=inspect)
        return receipt, state, calls

    def test_both_real_orchestrations_reinspect_distinct_default_and_selected_inputs(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate):
                receipt, state, calls = self.matrix(gate)
                self.assertEqual(receipt['gate'], gate)
                self.assertEqual(len(state['calls']), 9)
                self.assertEqual(len(calls['inspect']), 2)
                self.assertEqual(len(calls['select']), 2)
                self.assertEqual(calls['select'][0], calls['select'][1])

    def test_both_orchestrations_reject_default_image_plan_drift(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'container release inputs changed'):
                self.matrix(gate, 'image-plan')

    def test_both_orchestrations_reject_default_image_recipe_drift(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'container release inputs changed'):
                self.matrix(gate, 'image-recipe')

    def test_both_orchestrations_reject_selected_native_plan_drift(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'selected native recipe inputs changed'):
                self.matrix(gate, 'selected-plan')

    def test_both_orchestrations_reject_selected_native_recipe_drift(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'selected native recipe inputs changed'):
                self.matrix(gate, 'selected-recipe')

    def test_both_orchestrations_reload_the_sealed_parent(self):
        for gate in S.NATIVE_CONTEXT_RECIPES:
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, 'parent changed'):
                self.matrix(gate, 'sealed-parent')


class ClosureTests(Offline):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'source'; self.root.mkdir()
        self.runtime = Path(temp.name) / 'runtime'; self.runtime.mkdir()
        files = {}
        for gate in C.NATIVE_CONTEXT_REQUIRED_GATES:
            a = C.required_authority(gate)
            for rel, digest, mode in [(*a['program'], 0o755), (*a['recipe'], 0o644),
                    *[(r, d, 0o644) for f, r, d in a['helpers']],
                    ('repro-kit/r15-container-suite.py', Q.sha256_file(KIT / 'r15-container-suite.py'), 0o755),
                    (Q.NATIVE_CONTEXT_REQUIRED_MODULE, Q.NATIVE_CONTEXT_REQUIRED_MODULE_SHA256, 0o644)]:
                dst = self.root / rel; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(KIT.parent / rel, dst); dst.chmod(mode)
                files[rel] = {'path': 'flux-engine/' + rel, 'role': 'engine-source',
                              'mode': mode, 'bytes': dst.stat().st_size, 'sha256': digest}
        self.entries = {row['path']: row for row in files.values()}
        self.inventory = Path(temp.name) / 'inventory.json'
        self.inventory.write_bytes(Q.canonical_bytes({'files': list(files.values())}))
        self.provenance = {'source_bundle_manifest_path': str(self.inventory),
                           'source_bundle_manifest_sha256': Q.sha256_file(self.inventory)}
        self.suite = {'program_path': str(self.root / 'repro-kit/r15-container-suite.py'),
                      'program_sha256': Q.sha256_file(KIT / 'r15-container-suite.py')}
        self.plan = {**launch(), 'runtime_dir': str(self.runtime),
                     'gate_jobs': {g: job(g, self.root, self.runtime, self.suite['program_sha256'])
                                   for g in C.NATIVE_CONTEXT_REQUIRED_GATES}}

    def close(self):
        Q._validate_native_context_source_closure(self.provenance, self.suite, self.plan,
                                                 self.root, self.entries)

    def test_actual_program_helpers_recipes_and_validator_close(self):
        self.close()

    def test_outer_program_closure_reaches_the_required_native_validator(self):
        plan = copy.deepcopy(self.plan)
        provenance = dict(self.provenance)
        for key, value in (('source_revision', 'a' * 40), ('source_tree', 'b' * 40),
                           ('checkpoint_id', 'example/model'), ('checkpoint_revision', 'c' * 40),
                           ('checkpoint_manifest_sha256', 'd' * 64), ('recipe_sha256', 'e' * 64)):
            provenance[key] = plan[key] = value
        provenance['binary_sha256'] = plan['in_container_binary_sha256']
        plan['source_bundle_manifest_sha256'] = provenance['source_bundle_manifest_sha256']
        plan['source_root_path'] = str(self.root)
        # 295a6aaa1 binds the plan's release version to the provenance's.
        plan['release_version'] = provenance['release_version'] = Q.RELEASE_VERSION
        for gate in Q.REQUIRED_GATES:
            plan['gate_jobs'].setdefault(gate, dict(self.suite))
        with mock.patch.object(Q, '_validate_profile_source_closure'):
            Q._validate_program_closure(provenance, self.suite, plan, self.root)
            argv = plan['gate_jobs'][GATE]['argv']
            argv[argv.index('--engagement-program') + 1] = '/tmp/wrong-helper'
            with self.assertRaises(ValueError):
                Q._validate_program_closure(provenance, self.suite, plan, self.root)

    def test_wrong_job_program_hash_helpers_and_extra_options_fail(self):
        original = copy.deepcopy(self.plan)
        mutations = [('program_sha256', '1' * 64), ('program_path', '/tmp/driver')]
        for key, value in mutations:
            self.plan = copy.deepcopy(original); self.plan['gate_jobs'][GATE][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.close()
        for flag, value in (('--engagement-program', '/tmp/helper'), ('--suite-sha256', '9' * 64)):
            self.plan = copy.deepcopy(original); argv = self.plan['gate_jobs'][GATE]['argv']
            argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError): self.close()
        self.plan = copy.deepcopy(original); self.plan['gate_jobs'][GATE]['argv'] += ['--recipe', '/tmp/env']
        with self.assertRaises(ValueError): self.close()

    def test_inventory_mode_hash_role_missing_and_actual_byte_drift_fail(self):
        original = copy.deepcopy(self.entries)
        key = 'flux-engine/repro-kit/r16_native_context_required.py'
        for field, value in (('mode', 0o755), ('sha256', '1' * 64), ('role', 'fixture')):
            self.entries = copy.deepcopy(original); self.entries[key][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.close()
        self.entries = copy.deepcopy(original); self.entries.pop(key)
        with self.assertRaises(ValueError): self.close()
        self.entries = original
        path = self.root / C.required_authority(GATE)['recipe'][0]
        path.write_bytes(path.read_bytes() + b'# drift\n')
        with self.assertRaises(ValueError): self.close()


class EvidenceTests(Offline):
    @classmethod
    def setUpClass(cls):
        case = W.MatrixTests('test_real_orchestration_preserves_all_nine_cases_and_64_chunk_canary')
        case.setUp()
        try: cls.receipt, state = case.run_matrix()
        finally: case.doCleanups()
        docs = [json.loads(bytes(raw)) for raw in state['storage'].values() if raw.startswith(b'{"container_id"')]
        cls.validator, = [doc for doc in docs if doc.get('raw')]

    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.runtime = Path(temp.name)
        self.receipt = copy.deepcopy(self.receipt); self.validator = copy.deepcopy(self.validator)
        raw = self.validator['raw']
        raw['live_controls'] = {'tokenizer_sha256': W.G.TOKENIZER_SHA256,
                               'controls': C._plain(C.NATIVE262K_W4096_OWNER16_CONTROLS),
                               'argv_sha256': '3' * 64, 'environment_sha256': '4' * 64}
        raw['process_bindings'] = [proof(GATE)]
        self.manifest = {'_runtime_dir': str(self.runtime), 'launch': {'run_id': 'offline-fixture'},
                         'image': {'id': self.receipt['image_id'], 'digest': self.receipt['image_digest'],
                                   'recipe_sha256': '2' * 64, 'in_container_binary_sha256': '1' * 64}}
        patch = mock.patch.object(Q, '_profile_plan', return_value=launch())
        patch.start(); self.addCleanup(patch.stop)

    def evidence(self):
        path = self.runtime / self.receipt['evidence']['validator_artifact']
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = Q.canonical_bytes(self.validator); path.write_bytes(payload)
        self.receipt['evidence']['validator_receipt_sha256'] = Q.sha256_bytes(payload)
        return self.receipt['evidence']

    def validate(self):
        return Q._validate_gate_evidence(GATE, self.evidence(), self.manifest, CONTAINER)

    def test_real_mocked_nine_request_raw_receipt_passes_independent_wrapper(self):
        self.assertEqual(self.validate()['prefill_chunk_tokens'], 4096)

    def test_wrapper_gate_receipt_requires_exact_gate_and_release_identity(self):
        self.evidence()
        path = self.runtime / 'required-gate.json'
        for key, value in ((None, None), ('gate', 'native-262k'), ('container_id', 'd' * 64),
                           ('release_qualified', False)):
            receipt = copy.deepcopy(self.receipt)
            if key is not None: receipt[key] = value
            payload = Q.canonical_bytes(receipt); path.write_bytes(payload)
            args = (GATE, 'required-gate.json', Q.sha256_bytes(payload), self.manifest, CONTAINER)
            if key is None:
                self.assertEqual(Q._validate_gate_artifact(*args)['status'], 'PASS')
            else:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    Q._validate_gate_artifact(*args)

    def test_native_process_proof_recipe_gate_geometry_and_safety_drift_fail(self):
        original = copy.deepcopy(self.validator)
        for key, value in (('schema', 'flux-r15-live-process-proof/1'), ('gate', 'native-262k'),
                           ('recipe_sha256', '1' * 64), ('chunk_tokens', 16384),
                           ('paths_read_only', False), ('host_pid', True),
                           ('native_context_contract_sha256', '2' * 64)):
            self.validator = copy.deepcopy(original); self.validator['raw']['process_bindings'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate()

    def test_resealed_census_live_controls_retrieval_and_64chunk_drift_fail(self):
        original = copy.deepcopy(self.validator)
        for path, value in ((('post_rejection_metrics', 'metric_deltas', 'atlas_prefill_progress_chunks_total'), 16),
                            (('live_controls', 'controls', 'QWEN4_PREFILL_WIDE'), '16384'),
                            (('owner16_engagement', 'before_generation', 'destination_owners'), 48),
                            (('prefill_chunk_tokens',), 16384)):
            self.validator = copy.deepcopy(original); row = self.validator['raw']
            for key in path[:-1]: row = row[key]
            row[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError): self.validate()
        self.validator = copy.deepcopy(original)
        self.validator['raw']['retrievals'][0]['answers'][0] = 'wrong'
        with self.assertRaises(ValueError): self.validate()


if __name__ == '__main__':
    unittest.main()
