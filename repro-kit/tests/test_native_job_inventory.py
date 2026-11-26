# SPDX-License-Identifier: AGPL-3.0-only
"""Offline inventory counterexamples; no tool, depfile or output is generated."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import unittest

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parent
spec = importlib.util.spec_from_file_location('native_jobs_test_subject', KIT / 'native_job_inventory.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def blob(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture():
    output = 'release/build/fixture/out/kernel.ptx'
    job = {'schema': M.JOB_SCHEMA, 'output': output,
        'source': {'root': 'source', 'path': 'kernels/public/kernel.cu'},
        'depfile': output + '.d', 'tool_role': 'nvcc',
        'command_record': output + '.native-command'}
    source, result, depfile = ('/checkout/kernels/public/kernel.cu', '/target/' + output,
                              '/target/' + output + '.d')
    fields = [M.COMMAND_SCHEMA, '/tools/nvcc', source, result, depfile, '0',
              '--ptx', '-arch=sm_121a', source, '-o', result, '-MD', '-MF', depfile]
    memory = {output + '.native-job.json': blob(job),
              job['command_record']: ('\0'.join(fields) + '\0').encode()}
    parameters = {'target_paths': [output, job['depfile'], *memory], 'atlas_outputs': [output],
        'read_artifact': memory.__getitem__, 'target_root': '/target',
        'source_roots': {'source': '/checkout'}, 'tools': {'nvcc': '/tools/nvcc', 'cxx': '/tools/cxx'}}
    return job, fields, memory, parameters


class InventoryTests(unittest.TestCase):
    def mutate_job(self, mutate):
        job, fields, memory, parameters = fixture()
        mutate(job)
        memory[next(key for key in memory if key.endswith('.json'))] = blob(job)
        return parameters

    def mutate_record(self, mutate):
        job, fields, memory, parameters = fixture()
        mutate(fields)
        memory[job['command_record']] = ('\0'.join(fields) + '\0').encode()
        return parameters

    def test_folds_actual_shape_without_reading_or_manufacturing_dependency_payload(self):
        job, fields, memory, parameters = fixture()
        reads = []
        def read(path):
            reads.append(path)
            return memory[path]
        parameters['read_artifact'] = read
        self.assertEqual(M.fold_native_jobs(**parameters), {
            job['output']: {key: job[key] for key in ('source', 'depfile', 'tool_role', 'command_record')}})
        self.assertEqual(set(reads), set(memory))
        self.assertNotIn(job['depfile'], reads)

    def test_missing_and_extra_coverage_are_refused(self):
        for expected in ([], ['absent.o']):
            job, fields, memory, parameters = fixture()
            parameters['atlas_outputs'] = expected
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                M.fold_native_jobs(**parameters)

    def test_duplicate_inventory_and_expected_outputs_are_refused(self):
        for key in ('target_paths', 'atlas_outputs'):
            job, fields, memory, parameters = fixture()
            parameters[key] = [*parameters[key], parameters[key][0]]
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.fold_native_jobs(**parameters)

    def test_omitted_or_unbound_sidecar_artifacts_are_refused(self):
        for suffix in ('.d', '.native-command', '.ptx'):
            job, fields, memory, parameters = fixture()
            missing = job['output'] if suffix == '.ptx' else job['output'] + suffix
            parameters['target_paths'].remove(missing)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                M.fold_native_jobs(**parameters)

    def test_sidecar_unknown_missing_and_duplicate_fields_are_refused(self):
        for raw in (b'{"schema":1,"schema":2}', b'[]', b'NaN'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                M.job_document(raw)
        for mutate in (lambda j: j.update(extra=True), lambda j: j.pop('source'),
                       lambda j: j.update(schema='flux.native-job/2')):
            with self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_job(mutate))

    def test_traversal_absolute_and_ambiguous_job_paths_are_refused(self):
        for path in ('../x', '/x', 'a//b', 'a/./b', 'a/../b', 'a\\b', 'a\n', ''):
            for key in ('output', 'depfile', 'command_record'):
                with self.subTest(path=path, key=key), self.assertRaises(ValueError):
                    M.fold_native_jobs(**self.mutate_job(lambda j: j.update({key: path})))

    def test_output_specific_suffixes_cannot_be_reused(self):
        for key, value in (('depfile', 'release/shared.d'),
                           ('command_record', 'release/shared.native-command')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_job(lambda j: j.update({key: value})))

    def test_wrong_source_or_tool_role_is_refused(self):
        for mutate in (lambda j: j.update(tool_role='ar'), lambda j: j.update(tool_role='cxx'),
                       lambda j: j['source'].update(root='headers'),
                       lambda j: j['source'].update(path='../escape.cu'),
                       lambda j: j['source'].update(extra='x')):
            with self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_job(mutate))

    def test_wrong_actual_command_identity_is_refused(self):
        for index in range(1, 6):
            with self.subTest(index=index), self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_record(lambda fields: fields.__setitem__(index, '/wrong')))

    def test_missing_duplicate_or_mmd_dependency_options_are_refused(self):
        mutations = [lambda f: f.remove('-MD'), lambda f: f.append('-MD'),
            lambda f: f.__setitem__(f.index('-MD'), '-MMD'), lambda f: f.append('-MF'),
            lambda f: f.append('-o'), lambda f: f.__setitem__(f.index('-MF') + 1, '/target/wrong.d'),
            lambda f: f.__setitem__(f.index('-o') + 1, '/target/wrong.ptx'),
            lambda f: f.append('/checkout/kernels/public/kernel.cu')]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_record(mutate))

    def test_dependency_aliases_and_forwarded_overrides_are_refused_by_real_folder(self):
        aliases = ['-M', '-MM', '-MMD', '-MG', '-MP', '-MTforeign', '-MQforeign',
            '-MF/target/foreign.d', '--generate-dependencies',
            '--generate-nonsystem-dependencies', '--generate-dependencies-with-compile',
            '--generate-nonsystem-dependencies-with-compile',
            '--generate-dependency-targets', '--dependency-output=/target/foreign.d',
            '--dependency-target-name=foreign', '--dependency-drive-prefix=/foreign']
        for option in aliases:
            for forwarded in (option, '-Xcompiler=' + option, '-Wp,' + option,
                              '--compiler-options=' + option):
                with self.subTest(option=forwarded), self.assertRaises(ValueError):
                    M.fold_native_jobs(**self.mutate_record(lambda f: f.append(forwarded)))

    def test_response_files_and_alternate_outputs_are_refused_by_real_folder(self):
        aliases = ['@/target/options', '-optf', '-optf=/target/options',
            '--options-file', '--options-file=/target/options',
            '-o/target/foreign.ptx', '--output-file=/target/foreign.ptx']
        for option in aliases:
            for forwarded in (option, '-Xcompiler=' + option, '-Wp,' + option):
                with self.subTest(option=forwarded), self.assertRaises(ValueError):
                    M.fold_native_jobs(**self.mutate_record(lambda f: f.append(forwarded)))

    def test_forwarded_canonical_spellings_are_not_top_level_options(self):
        for forwarder in M.FORWARDERS:
            for option in ('-MD', '-MF', '-o'):
                def mutate(fields):
                    fields.insert(fields.index(option), forwarder)
                with self.subTest(forwarder=forwarder, option=option), self.assertRaises(ValueError):
                    M.fold_native_jobs(**self.mutate_record(mutate))

    def test_quoted_forwarded_overrides_are_refused(self):
        for value in ("-Xcompiler='-MF,/target/foreign.d'", '-Wp,"-MD",/target/foreign.d',
                      "--compiler-options='@/target/options'"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_record(lambda f: f.append(value)))

    def test_whitespace_hidden_options_are_refused_by_real_folder(self):
        for whitespace in (' ', '\t', '\n', '\r', '\v', '\f', '\u0085', '\u00a0', '\u001c'):
            for option in ('-MF', '-MD', '--dependency-output=/foreign',
                           '--output-file=/foreign', '@/target/options'):
                value = '-Xcompiler=-fPIC' + whitespace + option + whitespace + '/foreign'
                with self.subTest(whitespace=repr(whitespace), option=option), self.assertRaises(ValueError):
                    M.fold_native_jobs(**self.mutate_record(lambda f: f.append(value)))

    def test_argument_vector_and_field_bounds_are_enforced(self):
        for extra in ([''], ['x' * 8193], ['x'] * 4097):
            with self.subTest(extra_count=len(extra)), self.assertRaises(ValueError):
                M.fold_native_jobs(**self.mutate_record(lambda f: f.extend(extra)))

    def test_multibyte_argument_is_bounded_by_utf8_bytes(self):
        value = '\u00e9' * 4097
        self.assertLess(len(value), 8192)
        self.assertGreater(len(value.encode('utf-8')), 8192)
        with self.assertRaises(ValueError):
            M.fold_native_jobs(**self.mutate_record(lambda f: f.append(value)))

    def test_canonical_commands_retain_remap_architecture_and_math_flags(self):
        job, fields, memory, parameters = fixture()
        fields.extend(['-O3', '--use_fast_math', '-Xptxas', '-O3', '-Xcompiler',
            '-ffile-prefix-map=/private/source=/flux-build',
            '-Xcompiler=-fmacro-prefix-map=/private/source=/flux-build',
            '--compiler-options=-fdebug-prefix-map=/private/source=/flux-build',
            '-std=c++17', '--expt-relaxed-constexpr'])
        memory[job['command_record']] = ('\0'.join(fields) + '\0').encode()
        self.assertIn(job['output'], M.fold_native_jobs(**parameters))

    def test_nul_utf8_and_record_bounds_fail_closed(self):
        for raw in (b'', b'not-nul', b'\xff\0', b'x' * (M.MAX_COMMAND_BYTES + 1)):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                M.command_fields(raw)
        with self.assertRaises(ValueError):
            M.job_document(b' ' * (M.MAX_JOB_BYTES + 1))

    def test_flashinfer_reference_and_host_object_record_are_supported(self):
        job, fields, memory, parameters = fixture()
        output = job['output'].replace('.ptx', '.o')
        job.update(output=output, depfile=output + '.d', command_record=output + '.native-command',
                   tool_role='cxx', source={'root': 'flashinfer', 'path': 'src/bridge.cpp'})
        fields = [M.COMMAND_SCHEMA, '/tools/cxx', '/fi/src/bridge.cpp', '/target/' + output,
            '/target/' + output + '.d', '0', '-c', '/fi/src/bridge.cpp', '-o', '/target/' + output,
            '-MD', '-MF', '/target/' + output + '.d']
        memory = {output + '.native-job.json': blob(job),
                  job['command_record']: ('\0'.join(fields) + '\0').encode()}
        parameters.update(target_paths=[output, job['depfile'], *memory], atlas_outputs=[output],
            read_artifact=memory.__getitem__, source_roots={'source': '/checkout', 'flashinfer': '/fi'})
        self.assertEqual(M.fold_native_jobs(**parameters)[output]['tool_role'], 'cxx')

    def test_two_output_suffixes_have_independent_job_identity(self):
        job, fields, memory, parameters = fixture()
        second = copy.deepcopy(job)
        second['output'] = second['output'].replace('.ptx', '.o')
        second.update(depfile=second['output'] + '.d', command_record=second['output'] + '.native-command')
        second_fields = [part.replace(job['output'], second['output']) for part in fields]
        memory[second['output'] + '.native-job.json'] = blob(second)
        memory[second['command_record']] = ('\0'.join(second_fields) + '\0').encode()
        parameters['target_paths'] = [job['output'], job['depfile'], second['output'], second['depfile'], *memory]
        parameters['atlas_outputs'].append(second['output'])
        self.assertEqual(len(M.fold_native_jobs(**parameters)), 2)


class SourceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.helper = (ROOT / 'build-support/native_command_trace.rs').read_text()
        self.original = json.loads((KIT / 'tests/fixtures/native_command_default_flags.json').read_text())

    def source(self, name):
        return (ROOT / name).read_text()

    def test_optin_precedes_any_command_or_filesystem_mutation(self):
        prepare = self.helper.split('pub fn prepare(', 1)[1].split('fn json_string(', 1)[0]
        self.assertLess(prepare.index('if !enabled()'), prepare.index('canonical_existing(source'))
        self.assertIn('Some(value) if value.as_os_str() == OsStr::new(SCHEMA)', self.helper)
        self.assertIn('Some(_) => panic!', self.helper)

    def test_actual_dependency_flags_and_per_output_identity_are_emitted(self):
        self.assertIn('command.arg("-MD").arg("-MF").arg(&trace.depfile);', self.helper)
        for suffix in ('.d', '.native-command', '.native-job.json', '.native-trace.lock'):
            self.assertIn('append(output, "' + suffix + '")', self.helper)
        self.assertNotIn('with_extension("d")', self.helper)
        self.assertIn('const MAX_DEPFILE_BYTES: u64 = 8 << 20;', self.helper)
        self.assertIn('const MAX_RECORD_BYTES: usize = 1 << 20;', self.helper)

    def test_success_record_follows_real_compile_and_depfile_checks(self):
        finish = self.helper.split('pub fn finish(', 1)[1].split('#[cfg(test)]', 1)[0]
        self.assertLess(finish.index('if !success'), finish.index('fs::symlink_metadata(&trace.depfile)'))
        self.assertLess(finish.index('metadata.is_file()'), finish.index('sealed_write(&trace.record'))
        self.assertIn('text(command.get_program())', finish)
        self.assertIn('command.get_args().map(text)', finish)
        self.assertNotIn('sealed_write(&trace.depfile', self.helper)
        self.assertIn('remove_previous(path)', self.helper)

    def test_all_three_build_scripts_import_shared_policy(self):
        for name in ('crates/atlas-kernels/build_target.rs', 'crates/spark-storage/build.rs',
                     'crates/spark-model/build.rs'):
            text = self.source(name)
            with self.subTest(name=name):
                self.assertIn('#[path = "../../build-support/native_command_trace.rs"]', text)
                self.assertIn('native_command_trace::configure();', text)
                self.assertIn('native_command_trace::prepare(', text)
                self.assertIn('native_command_trace::finish(', text)

    def test_native_builds_keep_original_compiler_flags_when_unset(self):
        def normalize_rustfmt(snippet):
            collapsed = re.sub(r'\s+', ' ', snippet.strip())
            collapsed = re.sub(r',\s*([\]\)])', r'\1', collapsed)
            collapsed = re.sub(r'([\[\(])\s+', r'\1', collapsed)
            collapsed = re.sub(r'\s+([\]\)])', r'\1', collapsed)
            return collapsed

        name = 'crates/spark-model/build.rs'
        pattern = r'(compiler_with_release_path_remap\(&(?:nvcc|cxx), (?:true|false)\).*?)\n\s*"(?:nvcc|g\+\+) [^"]+"'
        original = [normalize_rustfmt(cmd) for cmd in self.original['model_commands']]
        updated = [normalize_rustfmt(cmd) for cmd in re.findall(pattern, self.source(name), re.S)]
        self.assertEqual(len(original), 5)
        self.assertEqual(original, updated)
        name = 'crates/atlas-kernels/build_target.rs'
        old = normalize_rustfmt(self.original['atlas_arguments'])
        new = normalize_rustfmt(self.source(name).split('let ieee =', 1)[1].split('let mut command =', 1)[0])
        self.assertEqual(old, new)
        name = 'crates/spark-storage/build.rs'
        old = normalize_rustfmt(self.original['storage_arguments'])
        new = normalize_rustfmt(self.source(name).split('.args([', 1)[1].split('])', 1)[0])
        self.assertEqual(old, new)

    def test_five_native_model_jobs_use_their_exact_source_and_output(self):
        text = self.source('crates/spark-model/build.rs')
        compact = re.sub(r'\s+', ' ', text)
        for source, output, role in [('bridge_cu', 'bridge_o', 'nvcc'),
                ('bridge_f16_cu', 'bridge_f16_o', 'nvcc'),
                ('bridge_f16_nf8_cu', 'bridge_f16_nf8_o', 'nvcc'),
                ('shim_cpp', 'shim_o', 'cxx'), ('fi_cu', 'fi_o', 'nvcc')]:
            self.assertIn(f'run_native( &{source}, &{output}, "{role}",', compact)
        self.assertEqual(len(re.findall(r'^\s+run_native\(\s*$', text, re.M)), 5)
        self.assertEqual(text.count('Command::new("ar")'), 2)
        wrapper = text.split('fn run_native(', 1)[1].split('fn compiler_with_', 1)[0]
        self.assertLess(wrapper.index('prepare('), wrapper.index('run(cmd, what)'))
        self.assertLess(wrapper.index('run(cmd, what)'), wrapper.index('finish('))

    def test_pinned_tool_is_optin_and_storage_finder_remains_default(self):
        text = self.source('crates/spark-storage/build.rs').split('fn find_nvcc()', 1)[1]
        self.assertLess(text.index('if native_command_trace::enabled()'), text.index('let candidates ='))
        self.assertIn('return native_command_trace::pinned_tool("ATLAS_NVCC");', text)
        self.assertIn('let candidates = ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"];', text)
        self.assertIn('letcheckout=canonical_existing(manifest.parent().and_then(Path::parent)', ''.join(self.helper.split()))
        self.assertNotIn('ATLAS_RELEASE_PATH_REMAP_FROM', self.helper)


if __name__ == '__main__':
    unittest.main()
