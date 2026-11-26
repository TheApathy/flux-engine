#!/usr/bin/env python3
"""Pure/source and mocked-launch controls. No server, subprocess or network runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
KIT = HERE.parent


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


GATES = KIT / 'r15-gates'
S = load(GATES / 'startup-readiness', 'review_staged_startup')
T = load(GATES / 'tool-api-streaming-blocking', 'review_staged_tool')
C = load(GATES / 'cancellation-slot-reuse', 'review_staged_cancellation')
H = load(KIT / 'r15-gates/mmlu-pro', 'review_original_lifecycle')
PLAN = {'in_container_binary_path': '/opt/flux/bin/flux-server-real'}
ARGS = types.SimpleNamespace(container_id='a' * 64, container_port=8888)


def recipe(chunk='4096'):
    return {'FLUX_MODE': {'4096': 'interactive-w4096', '16384': 'max-prefill'}[chunk],
            'ATLAS_QWEN4_PREFILL_WIDE': chunk, 'FLUX_QWEN4_PREFILL_WIDE': chunk}


def environment(values):
    return b''.join(key.encode() + b'=' + value.encode() + b'\0' for key, value in sorted(values.items()))


def startup_log(mode):
    native = mode == 'native-mtp-k5'
    rows = [b'Atlas Spark starting...']
    if native:
        rows.extend((b'Qwen4 native MTP schema admitted for direct proposer construction',
                     b'Qwen4 native MTP head constructed hidden=2560 residual=10240 experts=512 vocab=248077',
                     b'QWEN4_SPECULATION_ADMISSION_ENGAGED mode=native-only'))
    rows.append(b'Scheduler started (batched mode, max_batch=1, mtp=' + (b'true' if native else b'false')
                + b', ngram=false, num_drafts=4, fifo=true)')
    rows.append(b'Listening on 0.0.0.0:8888')
    return b'\n'.join(rows) + b'\n'


class RepairControls(unittest.TestCase):
    def setUp(self):
        self.no_subprocess = mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('no subprocess'))
        self.no_subprocess.start()
        self.addCleanup(self.no_subprocess.stop)
        self.no_network = mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('no network'))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def captured_helper_command(self, mode, values):
        class Captured(RuntimeError):
            pass
        captured = []
        def capture(command, **kwargs):
            captured.append(command)
            raise Captured('stopped before launch')
        with mock.patch.object(H.subprocess, 'Popen', side_effect=capture):
            with self.assertRaises(Captured):
                with H.server(ARGS, mode, PLAN, values):
                    self.fail('no process may be yielded')
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_startup_extends_actual_sealed_helper_launch_in_both_profiles_and_modes(self):
        self.assertEqual(hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest(), S.LIFECYCLE_SHA256)
        for chunk in ('4096', '16384'):
            for mode in S.MODES:
                values = recipe(chunk)
                old = self.captured_helper_command(mode, values)
                new = S.startup_launch(old, ARGS, PLAN, values, mode)
                self.assertEqual(new[:len(old)], old)
                for flag, value in (*S.STARTUP_ENGINE_PAIRS, ('--max-prefill-tokens', chunk)):
                    self.assertEqual(new.count(flag), 1)
                    self.assertEqual(new[new.index(flag) + 1], value)
                self.assertEqual(new.count('--qwen4-qsa'), 1)
                self.assertEqual(new[new.index('--kv-cache-dtype') + 1], 'bf16')
                self.assertEqual(new.count('--speculative'), int(mode == 'native-mtp-k5'))

    def test_startup_adapter_refuses_changed_prefix_env_or_argv(self):
        values = recipe()
        old = self.captured_helper_command('native-mtp-k5', values)
        for mutated in ([*old, '--mtp-vocab', '100000'],
                        [*old, '--max-batch-size', '8'],
                        [item.replace('ATLAS_DISABLE_SPECULATION=0', 'ATLAS_DISABLE_SPECULATION=1') for item in old],
                        [item.replace('/opt/flux/bin/flux-server-real', '/opt/flux/bin/foreign') for item in old]):
            with self.assertRaises(ValueError):
                S.startup_launch(mutated, ARGS, PLAN, values, 'native-mtp-k5')

    def test_startup_recipe_alias_mode_and_target_vocab_checks(self):
        for values in ({}, dict(recipe(), FLUX_QWEN4_PREFILL_WIDE='16384'),
                       dict(recipe(), FLUX_MODE='max-prefill'),
                       dict(recipe(), ATLAS_TARGET_LMHEAD_VOCAB='100000')):
            with self.assertRaises(ValueError):
                S.recipe_chunk(values)

    def test_startup_engine_log_forwards_only_its_validated_explicit_command(self):
        values = recipe()
        old = self.captured_helper_command('none', values)
        proxy = S.EngineLog(99, ARGS, PLAN, values)
        proxy.mode = 'none'
        with mock.patch.object(S.subprocess, 'Popen', return_value='mock-child') as launched:
            self.assertEqual(proxy.Popen(old, stdin=subprocess.DEVNULL), 'mock-child')
            self.assertEqual(launched.call_args.args[0], S.startup_launch(old, ARGS, PLAN, values, 'none'))
            self.assertEqual(launched.call_args.kwargs['stdout'], 99)
            self.assertEqual(launched.call_args.kwargs['stderr'], 99)

    def test_startup_native_requires_actual_full_head_and_plain_forbids_one(self):
        self.assertTrue(S.anchor_census(startup_log('none'), 'none', 8888)['plain_scheduler_ready'])
        good = startup_log('native-mtp-k5')
        self.assertTrue(S.anchor_census(good, 'native-mtp-k5', 8888)['k5_scheduler_ready'])
        for bad in (good.replace(b'vocab=248077', b'vocab=100000'),
                    good.replace(b'hidden=2560', b'hidden=4096'),
                    good.replace(b'Qwen4 native MTP schema admitted for direct proposer construction\n', b''),
                    good + b'DFlash drafter installed as the active proposer\n',
                    good + b'Qwen4 native MTP head constructed hidden=2560 vocab=248077\n'):
            with self.assertRaises(ValueError):
                S.anchor_census(bad, 'native-mtp-k5', 8888)
        with self.assertRaises(ValueError):
            S.anchor_census(startup_log('none') + b'Qwen4 native MTP head constructed hidden=2560 vocab=248077\n',
                            'none', 8888)

    def test_tool_exact_live_contract_is_full_vocab_bf16_qsa_c1(self):
        argv = [value.encode() for value in T.server_argv(PLAN['in_container_binary_path'], 'tool-test', 8888)]
        env = environment(dict(recipe(), ATLAS_DISABLE_SPECULATION='1'))
        controls = T.check_live_controls(argv, env, 'tool-test')
        self.assertEqual(controls['contract']['--mtp-vocab'], '0')
        self.assertEqual(controls['contract']['--mtp-quantization'], 'bf16')
        self.assertEqual(controls['contract']['--kv-cache-dtype'], 'bf16')
        self.assertEqual(controls['contract']['--max-batch-size'], '1')
        self.assertEqual(controls['speculation'], 'disabled')

    def test_tool_rejects_missing_qsa_truncated_vocab_and_unbound_extra_flags(self):
        good = [value.encode() for value in T.server_argv(PLAN['in_container_binary_path'], 'tool-test', 8888)]
        env = environment(dict(recipe(), ATLAS_DISABLE_SPECULATION='1'))
        for bad in ([v for v in good if v != b'--qwen4-qsa'],
                    [*good, b'--mtp-vocab', b'100000'],
                    [*good, b'--tp-size=2'],
                    [*good, b'--speculative'],
                    [*good, b'--unknown-engine-override', b'1']):
            with self.assertRaises(ValueError):
                T.check_live_controls(bad, env, 'tool-test')

    def test_tool_high_prefill_or_unbound_default_recipe_fails_before_launch(self):
        for values in ({}, recipe('16384'), dict(recipe(), ATLAS_TARGET_LMHEAD_VOCAB='100000')):
            with self.assertRaises(ValueError):
                T.require_tool_recipe(values)
            args = types.SimpleNamespace(lifecycle=types.SimpleNamespace())
            with self.assertRaises(ValueError):
                with T.server(args, PLAN, values, 'tool-test'):
                    self.fail('invalid recipe cannot yield a process')

    def test_live_tool_environment_must_match_interactive_aliases_and_plain(self):
        argv = [value.encode() for value in T.server_argv(PLAN['in_container_binary_path'], 'tool-test', 8888)]
        for values in (dict(recipe('16384'), ATLAS_DISABLE_SPECULATION='1'),
                       dict(recipe(), ATLAS_DISABLE_SPECULATION='0'),
                       dict(recipe(), ATLAS_DISABLE_SPECULATION='1', FLUX_QWEN4_PREFILL_WIDE='16384')):
            with self.assertRaises(ValueError):
                T.check_live_controls(argv, environment(values), 'tool-test')

    def test_cancellation_both_mode_gate_unchanged_and_plain_subset_cannot_pass(self):
        self.assertEqual(hashlib.sha256((GATES / 'cancellation-slot-reuse').read_bytes()).hexdigest(),
                         '6c7747ee15de31af021a5bb3eb801009bb46fbeccf58c5e675e835b32021b379')
        self.assertEqual(C.MODES, ('none', 'native-mtp-k5'))
        self.assertEqual(C.WIDTH, 8)
        with self.assertRaisesRegex(C.GateFailure, 'both plain and native-K5'):
            C.census([{'mode': 'none'}])

    def test_engine_source_speculative_concurrency_and_capability_contracts(self):
        server = (KIT.parent / 'crates/spark-server/src/main_modules/qwen4_concurrency.rs').read_text()
        model = (KIT.parent / 'crates/spark-model/src/model/trait_impl/decode_qwen4_capability.rs').read_text()
        # Concurrent speculative Qwen4 decode admits serial/batched native K5 and
        # adaptive-K batched DFlash2; other speculative modes (self/ngram) remain refused.
        self.assertIn('if c.speculative && !c.batched_k5 {', server)
        self.assertIn('Qwen4 concurrency is target-only unless native K5 is selected', server)
        self.assertIn('let batched_dflash = !c.native_mtp_only && c.batched_adaptive_k;', server)
        self.assertIn('if !c.native_mtp_only && !batched_dflash {', server)
        # Native decode capability admits target-only runs and native-MTP/DFlash bootstrap frames.
        self.assertIn(
            'let target_only =\n            !self.self_speculative && (!self.any_proposer() || self.speculation_forced_disabled);',
            model,
        )
        self.assertIn('let native_mtp_bootstrap = self.qwen4_native_mtp_bootstrap_admitted();', model)
        self.assertIn(
            'require(\n            target_only || native_mtp_bootstrap,\n            R::TargetOnly,\n            "native Qwen4 decode is target-only; disable speculative decoding",\n        )?;',
            model,
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
