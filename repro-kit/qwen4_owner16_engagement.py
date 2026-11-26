# SPDX-License-Identifier: AGPL-3.0-only
"""Pure, strict owner16 startup census over supplied raw bytes. No I/O."""


import hashlib
import re


PARENT_TEMPLATE_SHA256 = "5836e96551aed35484c119f44656ea64f48a1709895377731eac0f51216cd2b7"
PARENT_ADAPTER_SHA256 = "7d79e1773bf9a96ba8bfba065c714b638608394b44fa57d525cdea218f700497"


MAX_LOG_BYTES = 32 * 1024 * 1024


READY = 'BOUNDED_TRANSPOSE_OWNERS_READY'


SELECTOR = 'ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS'


CAP_SELECTOR = 'ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB'


CAP_VALUE = '16'


EXPECTED_READY = {
    'schema': 'atlas.moe-bounded-transpose-owners.v1', 'selector': SELECTOR,
    'owner_count': '87', 'owner_cap_bytes': '16777216', 'payload_bytes': '1415577600',
    'requested_bytes': '1415599872', 'guard_and_alignment_bytes': '22272',
    'experts': '512', 'projections': '3', 'managed': 'false',
    'base_unexposed': 'true', 'down_allocation_after_phase_b': 'true',
}


EVENT = re.compile(
    r'\bINFO[ \t]+'
    r'(?:bounded_transpose_layer\{layer_index=(?P<index>0|[1-9][0-9]*) '
    r'layer_ordinal=(?P<ordinal>[1-9][0-9]*)\}:[ \t]*)?'
    r'(?P<target>[a-zA-Z_][a-zA-Z0-9_:]*):[ \t]+(?P<body>.*)$')


ANSI = re.compile(rb'\x1b\[[0-9;]*m')


FIELDS = re.compile(r'\b([a-z_][a-z0-9_]*)=("[^"\r\n]*"|[^\s]+)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_record(line):
    found = EVENT.search(line)
    if found is None:
        return None
    if found['index'] is not None:
        index, ordinal = int(found['index']), int(found['ordinal'])
        require(0 <= index < 48 and ordinal == index + 1,
                'exact bounded transpose layer span differs')
    return found['target'], found['body']


def logger_matches(line, target, marker):
    found = event_record(line)
    return (found is not None and found[0] == target
            and re.match(re.escape(marker) + r'(?=[:\s])', found[1]) is not None)


def require_layer_span(line, layer_index):
    found = EVENT.search(line)
    require(found is not None and found['index'] is not None
            and int(found['index']) == layer_index and int(found['ordinal']) == layer_index + 1,
            '48 exact ordered bounded transpose layer spans required')


def body_fields(line):
    result = {}
    for match in FIELDS.finditer(line):
        key, value = match.groups()
        require(key not in result, 'duplicate receipt field')
        result[key] = value[1:-1] if value.startswith('"') else value
    return result


def line_fields(line):
    # Tracing span fields belong to the prefix, not READY event geometry.
    found = event_record(line)
    require(found is not None, "exact INFO event record required")
    return body_fields(found[1])


def engagement_census(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_LOG_BYTES and raw.endswith(b'\n'),
            'complete bounded raw server log required')
    text = ANSI.sub(b'', raw).decode('utf-8', 'strict')
    require('\x1b' not in text and '\x00' not in text, 'unsupported raw-log escape or NUL')
    lines = text.splitlines()
    require('MOE_TRANSPOSE_ROUTED_SLAB_READY' not in text,
            'legacy two-slab engagement is incompatible with bounded target owners')

    def one(marker, expected, target):
        found = [(index, line) for index, line in enumerate(lines) if marker in line]
        require(len(found) == text.count(marker) == 1, 'missing or repeated ' + marker)
        index, line = found[0]
        require(logger_matches(line, target, marker), 'receipt logger differs for ' + marker)
        fields = line_fields(line)
        require(all(fields.get(key) == value for key, value in expected.items()),
                'exact fields differ for ' + marker)
        return index, fields

    plan, _ = one('FAST_WEIGHT_ROUTED_GROUP_PLAN', {
        'mode': 'bounded-per-shard-layer-projection-v1', 'group_count': '2304',
        'component_views': '147456', 'payload_bytes': '67947724800',
        'requested_bytes': '67948314624', 'guard_and_alignment_bytes': '589824',
        'maximum_group_bytes': '67108864', 'base_guard_bytes': '256'},
        'atlas_weight_allocation_audit')
    eager, _ = one('FAST_WEIGHT_ROUTED_GROUPS_EAGER', {'bases': '2304', 'bytes': '67948314624'},
                   'atlas_weight_allocation_audit')
    bounded_copy, _ = one('ENGAGED ATLAS_BOUNDED_HOST_COPY', {'timeout_seconds': '60'},
                          'spark_runtime::cuda_backend::bounded_copy')
    starts = [index for index, line in enumerate(lines) if
              re.search(r'MoE transpose pass \(unified layout\): ATLAS_UNIFIED_MOE_LAYOUT=1, '
                        r'phased transpose with frees, free pre-pass [0-9]+(?:\.[0-9]+)? GB \u2192 RUNNING', line)]
    ends = [index for index, line in enumerate(lines) if
            re.search(r'MoE transpose pass \(unified layout\): done, [0-9]+(?:\.[0-9]+)? GB free', line)]
    require(len(starts) == len(ends) == 1 and text.count('MoE transpose pass (') == 2,
            'one complete unified transpose pass required')
    require(all(logger_matches(lines[index], 'spark_model::factory::m2_setup', 'MoE transpose pass')
                for index in (starts[0], ends[0])), 'unified transpose pass logger differs')
    found = [(index, line) for index, line in enumerate(lines) if READY in line]
    require(len(found) == text.count(READY) == 48, 'exactly 48 bounded target receipts required')
    for layer_index, (index, line) in enumerate(found):
        require_layer_span(line, layer_index)
        require(logger_matches(line, 'atlas_transposed_expert_slab', READY)
                and line_fields(line) == EXPECTED_READY,
                'exact bounded-owner receipt fields or logger differ')
        require(starts[0] < index < ends[0], 'bounded receipt is outside the unified pass')
    memory, values = one('MOE_TRANSPOSE_PASS_MEMORY', {
        'schema': 'atlas.moe-transpose-pass-memory.v1', 'layout': 'unified'},
        'atlas_transposed_expert_slab')
    for key in ('free_pre_bytes', 'free_post_bytes', 'free_delta_bytes'):
        require(re.fullmatch(r'-?[0-9]+', values.get(key, '')) is not None,
                'transpose-pass memory values differ')
    require(int(values['free_pre_bytes']) >= 0 and int(values['free_post_bytes']) >= 0
            and int(values['free_post_bytes']) - int(values['free_pre_bytes'])
            == int(values['free_delta_bytes']), 'transpose-pass memory delta differs')
    require(plan < eager < bounded_copy < starts[0] < ends[0] < memory,
            'startup engagement phases are out of order')
    return {'schema': 'r16.bounded-transpose.engagement.v1', 'status': 'PASS',
            'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'target_receipts': 48, 'destination_owners': 4176,
            'destination_payload_bytes': 67947724800, 'destination_requested_bytes': 67948793856,
            'receipt_fields': dict(EXPECTED_READY), 'legacy_two_slab_receipts': 0,
            'phases': ['group-plan', 'eager-source-owners', 'bounded-host-copy',
                       'unified-pass-start', '48-after-Phase-D-target-receipts',
                       'unified-pass-done', 'pass-memory-receipt'],
            'phase_proof_scope': 'source-reviewed after-Phase-D receipts, not per-call driver proof',
            'runtime_qualified': False, 'release_qualified': False}
