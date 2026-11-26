# SPDX-License-Identifier: AGPL-3.0-only
"""Fold real per-output build sidecars into native.jobs for the release producer.

The caller supplies paths and a reader from its sealed target inventory. This
module does not scan build directories, generate dependencies, capture headers,
or qualify a compiler, source closure, image, or runtime.
"""
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import PurePosixPath
import re

JOB_SCHEMA = 'flux.native-job/1'
COMMAND_SCHEMA = 'flux.native-command/1'
MAX_JOB_BYTES = 128 << 10
MAX_COMMAND_BYTES = 1 << 20
JOB_KEYS = {'schema', 'output', 'source', 'depfile', 'tool_role', 'command_record'}
FORWARDERS = frozenset({'-Xcompiler', '--compiler-options', '-Xlinker', '--linker-options',
    '-Xptxas', '--ptxas-options', '-Xcudafe', '--cudafe-options', '-Xnvlink', '--nvlink-options',
    '-Xpreprocessor', '--preprocessor-options', '-Xassembler', '-Xarchive', '--archive-options'})


def relative(value):
    if (type(value) is not str or not value or len(value.encode('utf-8')) > 4096
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or '\\' in value or value.startswith('/')
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('native job requires normalized relative paths')
    return value


def absolute(value):
    if type(value) is not str or not value.startswith('/'):
        raise ValueError('native job requires absolute roots and tools')
    relative(value[1:])
    return value


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate native job key')
        result[key] = value
    return result


def job_document(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_JOB_BYTES:
        raise ValueError('native job sidecar must be bounded bytes')
    value = json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite native job value')))
    if type(value) is not dict or set(value) != JOB_KEYS or value['schema'] != JOB_SCHEMA:
        raise ValueError('native job sidecar keys or schema differ')
    if type(value['source']) is not dict or set(value['source']) != {'root', 'path'}:
        raise ValueError('native job source reference keys differ')
    if value['source']['root'] not in ('source', 'flashinfer'):
        raise ValueError('native job source root differs')
    relative(value['source']['path'])
    for key in ('output', 'depfile', 'command_record'):
        relative(value[key])
    if PurePosixPath(value['output']).suffix not in ('.ptx', '.o'):
        raise ValueError('native job output is not PTX or object')
    if (value['depfile'] != value['output'] + '.d'
            or value['command_record'] != value['output'] + '.native-command'):
        raise ValueError('native job sidecar identities differ from output')
    if value['tool_role'] not in ('nvcc', 'cxx'):
        raise ValueError('native job tool role differs')
    if value['tool_role'] == 'cxx' and PurePosixPath(value['output']).suffix != '.o':
        raise ValueError('host compiler cannot produce PTX')
    return value


def command_fields(raw):
    if (type(raw) is not bytes or not 0 < len(raw) <= MAX_COMMAND_BYTES
            or not raw.endswith(b'\0')):
        raise ValueError('native command record must be bounded NUL-terminated bytes')
    fields = raw[:-1].decode('utf-8', 'strict').split('\0')
    if len(fields) < 7 or fields[0] != COMMAND_SCHEMA or fields[5] != '0':
        raise ValueError('native command schema or success status differs')
    for value in fields[1:5]:
        absolute(value)
    return fields


def conflicting_option(argument):
    """Reserve dependency/output options, including forwarded and response forms."""
    for value in re.split(r'[=,\s]+', argument):
        value = value.strip("\"'")
        if (value.startswith(('-M', '--generate-dependenc', '--generate-nonsystem-dependenc',
                              '--dependency-', '@', '-optf', '--options-file',
                              '-o', '--output-file'))):
            return True
    return False


def validate_command_options(argv, source_path, output_path, depfile_path):
    if (type(argv) is not list or not 0 < len(argv) <= 4096
            or any(type(value) is not str or not 0 < len(value.encode('utf-8')) <= 8192 or '\0' in value
                   for value in argv)):
        raise ValueError('native argument inventory bound differs')
    if (argv.count(source_path) != 1 or argv.count('-o') != 1
            or argv.count('-MD') != 1 or argv.count('-MF') != 1):
        raise ValueError('native command dependency or source options differ')
    canonical = {argv.index('-MD'), argv.index('-MF'), argv.index('-o')}
    forwarded = {index + 1 for index, value in enumerate(argv[:-1]) if value in FORWARDERS}
    canonical -= forwarded
    for flag, path in (('-o', output_path), ('-MF', depfile_path)):
        index = argv.index(flag)
        if index + 1 >= len(argv) or argv[index + 1] != path:
            raise ValueError('native command option binding differs')
    if any(conflicting_option(value) for index, value in enumerate(argv) if index not in canonical):
        raise ValueError('native command has a conflicting or hidden dependency/output override')


def fold_native_jobs(*, target_paths: Sequence[str], atlas_outputs: Sequence[str],
                     read_artifact: Callable[[str], bytes], target_root: str,
                     source_roots: Mapping[str, str], tools: Mapping[str, str]):
    """Return the native.jobs map, requiring exact caller-declared Atlas coverage.

    Artifact hashes, source references, real depfile rules, system/header bundle
    closure and tool identity are independently verified by the release consumer.
    No header or coverage count is invented by this folding step.
    """
    paths = [relative(path) for path in target_paths]
    expected = [relative(path) for path in atlas_outputs]
    if len(paths) != len(set(paths)) or len(expected) != len(set(expected)):
        raise ValueError('duplicate target artifact or Atlas output')
    paths = set(paths)
    if not set(expected) <= paths:
        raise ValueError('Atlas output missing from sealed target inventory')
    absolute(target_root)
    if set(source_roots) - {'source', 'flashinfer'} or 'source' not in source_roots:
        raise ValueError('native job source roots differ')
    for path in source_roots.values():
        absolute(path)
    if set(tools) != {'nvcc', 'cxx'}:
        raise ValueError('native job tools must be independently pinned')
    for path in tools.values():
        absolute(path)
    jobs, seen_depfiles, seen_records = {}, set(), set()
    for sidecar in sorted(path for path in paths if path.endswith('.native-job.json')):
        value = job_document(read_artifact(sidecar))
        output, depfile, record = (value[key] for key in ('output', 'depfile', 'command_record'))
        if sidecar != output + '.native-job.json':
            raise ValueError('native job sidecar is not bound to output')
        if output in jobs or depfile in seen_depfiles or record in seen_records:
            raise ValueError('native job output or proof identity was reused')
        if not {output, depfile, record} <= paths:
            raise ValueError('native job artifact is missing from sealed inventory')
        source = value['source']
        if source['root'] not in source_roots:
            raise ValueError('native job source root is not supplied')
        source_path = source_roots[source['root']] + '/' + source['path']
        output_path, depfile_path = target_root + '/' + output, target_root + '/' + depfile
        fields = command_fields(read_artifact(record))
        if fields[1:5] != [tools[value['tool_role']], source_path, output_path, depfile_path]:
            raise ValueError('native command identity differs from job or pinned tool')
        argv = fields[6:]
        validate_command_options(argv, source_path, output_path, depfile_path)
        jobs[output] = {key: value[key] for key in ('source', 'depfile', 'tool_role', 'command_record')}
        seen_depfiles.add(depfile)
        seen_records.add(record)
    if set(jobs) != set(expected):
        raise ValueError('native job coverage differs from declared Atlas outputs')
    return jobs
