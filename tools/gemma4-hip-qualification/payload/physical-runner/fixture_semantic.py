"""Typed acceptance for the eight Gemma PLE modes; actual caller seams only."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from pins import require

GIB = 1024**3
MIB = 1024**2
ROW_BYTES = 17920
RESIDENT_BYTES = 4_558_946_304
DIAGNOSTIC_LIMIT = 64*MIB
MATRIX_MODES = ('resident', 'resident-row-unknown', 'unknown', 'resident-unknown')
FINAL_MODES = ('session-known', 'session-release-fault', 'snapshot-known', 'snapshot-release-fault')
MODES = MATRIX_MODES + FINAL_MODES
MATRIX_MARKER = ('GEMMA_COMPLETE_SOURCE_MATRIX_PASS cases=7 full35=true '
                 'all_boundary_KV_fullV_words=exact source_guard=actual native_modules=actual_unloaded '
                 'model_reservations=0 context_baseline=retained_until_process_exit serving_performance=false')
UNKNOWN_MARKER = ('GEMMA_COMPLETE_UNKNOWN_PASS actual_completed_producer=1 published_roots=0 '
                  'original_envelope=retained original_source_close=refused original_context=process_lifetime')
KNOWN_V1 = 'zg.gemma4-e2b-original-hip-unknown.v1'
MATRIX_V1 = 'zg.gemma4-e2b-original-hip-complete-qualification.v1'
FINAL_V1 = 'zg.gemma4-original-hip-final-feedback-owner.v1'
# Real source-derived cold inventory, per storage policy (see layout.rs / storage.rs).
COMPLETE_SOURCE = 9_256_566_784
VALUE_SLOT_BYTES = 2_502_656
SOURCE_GAMMA = 1_146_880
MATH_SCRATCH = 114_688
STATUS_PACKET = 2_560
GREEDY = 12_336
TABLES = 2_048
ALIGNMENT_GUARD = 6*256
ROOT_BYTES = 2_359_296
ROOT_INIT = 3*ROOT_BYTES
SIX_ROOTS = 6*ROOT_BYTES
ROOT_POOL_METADATA = 256_000
STORAGE = {
    True: {'projection_scratch_bytes': 2_121_728, 'projection_max_rows': 262_144,
           'resident_matrix_bytes': RESIDENT_BYTES, 'largest_binding_bytes': 805_306_368,
           'resident_matrices': 277, 'resident_upload_windows': 4_516,
           'resident_control_host_bytes': 18_153_472,
           'selected_row_cache_capacity': 58, 'selected_row_cache_row_bytes': ROW_BYTES},
    False: {'projection_scratch_bytes': 40_960, 'projection_max_rows': 2_048,
            'resident_matrix_bytes': 0, 'largest_binding_bytes': 1_048_576,
            'resident_matrices': 0, 'resident_upload_windows': 0,
            'resident_control_host_bytes': 0,
            'selected_row_cache_capacity': 0, 'selected_row_cache_row_bytes': 0},
}
# The whole-report contract of the shipped fixtures: every top-level member, once.
REPORT_KEYS = {
    KNOWN_V1: (
        'cached_row_fault', 'complete_source_bytes', 'completed_producers',
        'context_retirement', 'diagnostic_host_reserved', 'faulted_call_commits', 'host_fenced',
        'host_reserved', 'inventory', 'physical_fenced', 'physical_reserved', 'published_roots',
        'schema', 'selected_row_avoided_h2d', 'selected_row_h2d', 'selected_row_reuses',
        'selected_row_uploads', 'source_close_error', 'source_close_refused', 'status',
        'total_value_upload_h2d', 'trained_native_serving_or_performance_qualified',
        'warmup_calls', 'warmup_commits'),
    MATRIX_V1: (
        'context_retirement', 'diagnostic_host_reserved', 'events', 'exact',
        'expected_persistent_context_device', 'expected_persistent_context_host', 'failure',
        'host_fenced', 'host_reserved', 'inventory', 'managed_source_closed', 'physical_fenced',
        'physical_reserved', 'schema', 'source_close_error', 'source_page_policy', 'status',
        'storage', 'trained_native_serving_or_performance_qualified'),
    FINAL_V1: (
        'actual_completed_fullV', 'actual_completed_layers', 'actual_completed_proposed_state_planes',
        'actual_selected', 'commits', 'diagnostic_host_reserved', 'failure',
        'final_feedback_on_completed_hit', 'final_hit', 'host_fenced', 'host_reserved', 'inventory',
        'mode', 'numerical_replay_performed', 'observed_completed_full35_calls',
        'old_order_physical_negative_performed', 'original_final_owner', 'physical_fenced',
        'physical_reserved', 'prompt_ids', 'prompt_text', 'root_transitions', 'schema',
        'source_close_error', 'source_close_refused', 'source_frontier', 'status',
        'total_completed_full35_calls', 'trained_native_natural_serving_or_performance_qualified',
        'warmup', 'warmup_calls', 'warmup_commits', 'warmup_completed_full35_calls'),
}
# The whole-event contract of the shipped matrix fixture: every member of every kind.
EVENT_KEYS = {
    'actual-completed-proposal': ('before', 'call', 'case', 'comparison', 'input_ids', 'kind', 'phase'),
    'actual-committed-call': ('call', 'case', 'eos', 'kind', 'root', 'selected_id'),
    'source-ID-decoded-output': ('case', 'ids', 'kind', 'sequence_decode_only', 'text'),
    'pre-cancel': ('kind', 'root'),
    'completed-Ready-drop': ('comparison', 'exact', 'kind', 'root'),
    'dynamic': ('comparison', 'error', 'kind', 'root'),
    'resident-normal-parity': (
        'actual_max_bits', 'actual_selected_id', 'base', 'call', 'case', 'frontier', 'input_ids',
        'kind', 'max_bits', 'observer', 'phase', 'ple_row_avoided_h2d', 'ple_row_h2d',
        'ple_row_reuses', 'ple_row_uploads', 'producer_kv_d2h', 'resident_weight_bytes',
        'schedule', 'selected_id', 'source_bound', 'state_domains_exact', 'value_upload_total'),
    'resident-ple-row-reuse': (
        'actual_max_bits', 'actual_rows', 'avoided_h2d', 'base', 'call', 'case', 'comparison',
        'frontier', 'full_kv_exact', 'input_ids', 'kind', 'kv_diagnostic_d2h', 'max_bits',
        'observer', 'phase', 'ple_row_h2d', 'producer_d2h', 'repeat', 'resident_weight_bytes',
        'reuses', 'schedule', 'selected_id', 'source_bound', 'state_domains_exact', 'uploads',
        'value_upload_total'),
}
DYNAMIC_KINDS = ('actual-completed-late-cancel', 'actual-completed-refusal')
SOURCE_FILES = ('gemma4_hip_matrix/run.rs', 'gemma4_hip_final_owner.rs')


def number(value, key, expected=None, low=0, high=2**63-1):
    require(type(value) is dict and key in value, 'present integer '+key)
    actual = value[key]
    require(type(actual) is int and low <= actual <= high
            and (expected is None or actual == expected), 'exact integer '+key)
    return actual


def boolean(value, key, expected=None):
    require(type(value) is dict and key in value, 'present boolean '+key)
    actual = value[key]
    require(type(actual) is bool and (expected is None or actual is expected),
            'exact boolean '+key)
    return actual


def diagnostic_bytes(value, key):
    actual = number(value, key, low=2, high=DIAGNOSTIC_LIMIT)
    require(actual % 4 == 0, 'four-byte-aligned diagnostic '+key)
    return actual


def ordinary_bytes(value, key):
    return number(value, key, expected=0)


def row_traffic(value, label, rows=None, uploads=None, hits=None, diagnostic=False,
                parity=False, exact_row=True):
    """PLE row bytes and whole-service value H2D are separate truthful fields."""
    require(type(value) is dict, label+' traffic object')
    keys = (('ple_row_uploads', 'ple_row_reuses', 'ple_row_h2d', 'ple_row_avoided_h2d')
            if parity else ('uploads', 'reuses', 'ple_row_h2d', 'avoided_h2d'))
    for key in keys:
        require(key in value, label+' traffic field '+key)
    up = number(value, keys[0], uploads, high=64)
    hit = number(value, keys[1], hits, high=64)
    if rows is not None:
        require(up + hit == rows, label+' uploads+hits=input rows')
    row = number(value, keys[2], low=0, high=COMPLETE_SOURCE)
    if exact_row:
        require(row == up*ROW_BYTES, label+' PLE row bytes are exactly its uploaded row pages')
    require(number(value, keys[3], hit*ROW_BYTES) == hit*ROW_BYTES,
            label+' avoided H2D is exactly the reused row pages')
    if 'value_upload_total' in value:
        require(number(value, 'value_upload_total', low=0, high=COMPLETE_SOURCE) == row,
                label+' whole-service value H2D equals its PLE row traffic')
    if parity:
        ordinary_bytes(value, 'producer_kv_d2h')
    elif diagnostic:
        diagnostic_bytes(value, 'producer_d2h')
        diagnostic_bytes(value, 'kv_diagnostic_d2h')
    else:
        ordinary_bytes(value, 'producer_d2h')
        ordinary_bytes(value, 'kv_diagnostic_d2h')
    return up, hit


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_duplicates(pairs):
    seen = set()
    for key, _ in pairs:
        require(key not in seen, 'duplicate report member '+str(key))
        seen.add(key)
    return dict(pairs)


def _no_constant(text):
    raise ValueError('non-finite report constant '+str(text))


def strict_report(raw):
    """One UTF-8 report object: no duplicate member, no non-finite constant, no trailer."""
    require(type(raw) is bytes and 0 < len(raw) <= 4*MIB, 'bounded fixture report bytes')
    text = raw.decode('utf-8')
    value = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    require(type(value) is dict and type(value.get('schema')) is str,
            'single schema-tagged fixture report')
    return value


def _object_keys(text, start):
    require(text[start] == '{', 'json object literal start')
    keys, depth, index, last = [], 0, start, start
    quoted = escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
        elif char in '{([':
            depth += 1
        elif char in '})]':
            depth -= 1
            if depth == 0:
                return tuple(keys)
        elif char == ',' and depth == 1:
            last = index
        elif char == ':' and depth == 1:
            match = re.search(r'"([^"\\]+)"\s*$', text[last+1:index])
            if match:
                keys.append(match.group(1))
        index += 1
    raise AssertionError('unbalanced json object literal')


def _literal_keys(text, marker):
    found = []
    for match in re.finditer(re.escape(marker), text):
        start = text.rfind('json!({', 0, match.start())
        require(start >= 0, 'enclosing json literal for '+marker)
        found.append(_object_keys(text, start+6))
    return found


def shipped_seams(source_dir):
    """Derive the report/event member sets from the shipped fixture sources."""
    root = Path(source_dir)
    text = ''
    for name in SOURCE_FILES:
        path = root/name
        require(path.is_file() and not path.is_symlink(), 'shipped fixture source '+name)
        text += path.read_text()
    reports = {}
    for schema in REPORT_KEYS:
        found = _literal_keys(text, '"schema":"%s"' % schema)
        require(len(found) == 1, 'one shipped report literal '+schema)
        reports[schema] = sorted(found[0])
    events = {}
    for kind in EVENT_KEYS:
        if kind == 'dynamic':
            found = _literal_keys(text, '"kind":kind')
            require(len(found) == 1, 'one shipped dynamic event literal')
            events[kind] = found[0]
            continue
        found = _literal_keys(text, '"kind":"%s"' % kind)
        require(len(found) == 1, 'one shipped event literal '+kind)
        events[kind] = found[0]
    return {'reports': reports, 'events': events}


def check_seams(source_dir):
    """The declared acceptance contract is exactly what the shipped sources emit."""
    seams = shipped_seams(source_dir)
    for schema, keys in REPORT_KEYS.items():
        require(seams['reports'][schema] == sorted(keys), 'shipped report contract '+schema)
    for kind, keys in EVENT_KEYS.items():
        require(tuple(sorted(seams['events'][kind])) == tuple(sorted(keys)),
                'shipped event contract '+kind)
    return {'reports': len(REPORT_KEYS), 'events': len(EVENT_KEYS),
            'sources': list(SOURCE_FILES)}


def report_contract(schema):
    require(schema in REPORT_KEYS, 'eight-mode declared report schema')
    return REPORT_KEYS[schema]


def event_contract(event):
    kind = event.get('kind')
    require(kind in EVENT_KEYS or kind in DYNAMIC_KINDS, 'declared matrix event kind '+str(kind))
    return EVENT_KEYS['dynamic' if kind in DYNAMIC_KINDS else kind]


def check_report(result, schema):
    require(type(result) is dict and result.get('schema') == schema,
            'exact fixture report schema '+schema)
    require(tuple(sorted(result)) == tuple(sorted(report_contract(schema))),
            'exact fixture report member set '+schema)
    return result


def check_events(events):
    require(type(events) is list and events, 'ordered actual matrix events')
    for event in events:
        require(type(event) is dict, 'matrix event object')
        require(tuple(sorted(event)) == tuple(sorted(event_contract(event))),
                'exact matrix event member set '+str(event.get('kind')))
    return events


def expected_cells(invocation_path):
    """The held original capture/cell JSONs are the seven-cell authority."""
    path = Path(invocation_path)
    receipt, receipt_pin = json.loads(path.read_text()), {'bytes': path.stat().st_size,
                                                         'sha256': sha256(path)}
    require(receipt.get('status') == 'EXECUTED_ORIGINAL_GEMMA4_MATRIX_EXPECTED_ONLY'
            and receipt.get('exit_code') == 0
            and receipt.get('source_before') == receipt.get('source_after')
            and receipt.get('source_FDs_closed') is True
            and receipt.get('cleanup_errors') == []
            and receipt.get('source_FD_cleanup_errors') == []
            and receipt.get('foreground_cleanup', {}).get('leader_reaped') is True
            and receipt.get('foreground_cleanup', {}).get('group_absent') is True,
            'whole original expected-capture closure')
    parent = path.parent
    capture_path = parent/'capture.json'
    capture_raw = capture_path.read_bytes()
    capture_pin = receipt.get('capture')
    require(type(capture_pin) is dict and set(capture_pin) == {'bytes', 'sha256'}
            and len(capture_raw) == capture_pin['bytes']
            and hashlib.sha256(capture_raw).hexdigest() == capture_pin['sha256'],
            'capture.json bound by the original invocation')
    capture = json.loads(capture_raw)
    request = capture.get('request')
    cells_meta = capture.get('cells')
    require(capture.get('schema') == 'zg.gemma4-e2b.ordinary-matrix.v1'
            and capture.get('plain_vs_observed_equal') is True
            and capture.get('trained_rust_qualified') is False
            and capture.get('native_serving_chat_or_performance_qualified') is False
            and type(cells_meta) is list and len(cells_meta) == 7
            and type(request) is dict and type(request.get('cases')) is list
            and len(request['cases']) == 7, 'entire independent seven-cell capture')
    expected = []
    names = []
    for index, case in enumerate(request['cases']):
        require(type(case) is dict and set(case) == {'name', 'text', 'rows'}
                and type(case['name']) is str and case['name'] not in names
                and type(case['rows']) is int and 1 <= case['rows'] <= 64
                and type(case['text']) is str, 'source request case')
        names.append(case['name'])
        entry = cells_meta[index]
        meta = entry.get('metadata') if type(entry) is dict else None
        require(type(meta) is dict and set(meta) == {'name', 'bytes', 'sha256'}
                and meta['name'] == case['name']+'.json'
                and type(meta['name']) is str and type(meta['bytes']) is int
                and type(meta['sha256']) is str and len(meta['sha256']) == 64,
                'capture metadata {name,bytes,sha256} joined at case order: '+case['name'])
        cell_path = parent/meta['name']
        cell_raw = cell_path.read_bytes()
        require(len(cell_raw) == meta['bytes']
                and hashlib.sha256(cell_raw).hexdigest() == meta['sha256'],
                'cell metadata pin: '+case['name'])
        cell = json.loads(cell_raw)
        require(cell.get('schema') == 'zg.gemma4-e2b.ordinary-cell.v1'
                and cell.get('request', {}).get('name') == case['name']
                and cell.get('request', {}).get('text') == case['text']
                and cell.get('request', {}).get('rows') == case['rows']
                and type(cell.get('prompt_ids')) is list
                and len(cell['prompt_ids']) == case['rows']
                and type(cell.get('calls')) is list and 1 <= len(cell['calls']) <= 8,
                'authentic cell: '+case['name'])
        calls = cell['calls']
        frontier = 0
        clean = []
        for n, call in enumerate(calls):
            require(type(call) is dict and call.get('call') == n
                    and type(call.get('input_ids')) is list and call['input_ids']
                    and call.get('base') == frontier
                    and type(call.get('frontier')) is int
                    and call['frontier'] == frontier + len(call['input_ids'])
                    and call['frontier'] <= 64
                    and type(call.get('selected_id')) is int
                    and 0 <= call['selected_id'] < 262144
                    and type(call.get('eos')) is bool
                    and (not call['eos'] or n + 1 == len(calls)),
                    'natural call chain: '+case['name'])
            frontier = call['frontier']
            clean.append({'call': n, 'base': call['base'], 'frontier': call['frontier'],
                          'selected_id': call['selected_id'], 'eos': call['eos'],
                          'input_ids': call['input_ids']})
        require((calls[-1]['eos'] or len(calls) == 8), 'natural stop: '+case['name'])
        expected.append({'name': case['name'], 'rows': case['rows'],
                         'prompt_ids': cell['prompt_ids'], 'calls': clean})
    return {'invocation': receipt_pin, 'capture': capture_pin, 'cells': expected}


def inventory(value, resident=True):
    text = value['inventory']
    require(type(text) is str and len(text) <= 16384, 'bounded actual inventory')
    parsed = {key: int(amount) for key, amount in re.findall(r'(\w+): (\d+)', text)}
    required = (
        'unique_source_matrices', 'complete_source_bytes', 'source_page_bytes',
        'selected_row_cache_capacity', 'selected_row_cache_row_bytes', 'value_slot_bytes',
        'value_slots', 'source_gamma_bytes', 'math_scratch_bytes', 'projection_scratch_bytes',
        'status_and_packet_bytes', 'original_greedy_workspace_bytes', 'single_root_payload_bytes',
        'six_root_device_bytes', 'owned_device_bytes', 'device_with_six_roots_bytes',
        'host_upper_bytes', 'native_host_bytes', 'native_device_bytes', 'largest_binding_bytes',
        'resident_matrices', 'resident_matrix_bytes', 'resident_upload_windows',
        'resident_control_host_bytes', 'root_pool_metadata_bytes',
        'root_initialization_host_bytes', 'projection_max_rows',
        'persistent_context_device_bytes', 'persistent_context_host_bytes',
        'complete_physical_device_bytes')
    require(set(required) <= set(parsed), 'complete actual cold inventory set')
    for key in required:
        number(parsed, key, high=32*GIB)
    number(parsed, 'unique_source_matrices', 278)
    number(parsed, 'complete_source_bytes', COMPLETE_SOURCE)
    number(parsed, 'source_page_bytes', MIB)
    number(parsed, 'value_slot_bytes', VALUE_SLOT_BYTES)
    number(parsed, 'value_slots', 24)
    number(parsed, 'source_gamma_bytes', SOURCE_GAMMA)
    number(parsed, 'math_scratch_bytes', MATH_SCRATCH)
    number(parsed, 'status_and_packet_bytes', STATUS_PACKET)
    number(parsed, 'original_greedy_workspace_bytes', GREEDY)
    number(parsed, 'single_root_payload_bytes', ROOT_BYTES)
    number(parsed, 'root_pool_metadata_bytes', ROOT_POOL_METADATA)
    number(parsed, 'persistent_context_device_bytes', 4*GIB)
    number(parsed, 'persistent_context_host_bytes', 2*GIB)
    policy = STORAGE[bool(resident)]
    for key, amount in policy.items():
        number(parsed, key, amount)
    number(parsed, 'six_root_device_bytes', SIX_ROOTS)
    number(parsed, 'root_initialization_host_bytes', ROOT_INIT)
    owned = (MIB + parsed['resident_matrix_bytes'] + VALUE_SLOT_BYTES + SOURCE_GAMMA
             + MATH_SCRATCH + parsed['projection_scratch_bytes'] + STATUS_PACKET + TABLES)
    number(parsed, 'owned_device_bytes', owned)
    number(parsed, 'device_with_six_roots_bytes',
           owned + SIX_ROOTS + GREEDY + ALIGNMENT_GUARD + parsed['native_device_bytes'])
    number(parsed, 'complete_physical_device_bytes',
           parsed['device_with_six_roots_bytes'] + 4*GIB)
    require(8*GIB <= parsed['native_device_bytes'] <= 9*GIB
            and 8*GIB <= parsed['native_host_bytes'] <= 10*GIB + 2*MIB,
            'opaque native 8GiB plus the finite module code/host allowances')
    require(parsed['host_upper_bytes'] > parsed['native_host_bytes']
            and 2*GIB < parsed['host_upper_bytes'] <= 16*GIB,
            'model/lowerer host upper covers one native owner inside the source domain')
    for key in ('largest_binding_bytes',):
        number(parsed, key, low=1)
    return parsed


def matrix_expected(value):
    cells = value['cells']
    require(type(cells) is list and len(cells) == 7
            and len({c['name'] for c in cells}) == 7, 'exact seven distinct cells')
    return cells


def matrix(mode, result, log, expected=None):
    if mode in ('resident', 'resident-row-unknown', 'unknown', 'resident-unknown'):
        check_report(result, MATRIX_V1 if mode == 'resident' else KNOWN_V1)
    require(result.get('failure') is None
            and result.get('trained_native_serving_or_performance_qualified') is False,
            'actual finite Gemma scope')
    require(type(result.get('context_retirement')) is str
            and 'process lifetime' in result['context_retirement'],
            'original process-lifetime context retirement')
    number(result, 'diagnostic_host_reserved', 32*MIB)
    if mode == 'resident-row-unknown':
        require(result['status'] == 'ACTUAL_UNKNOWN_RETAINED'
                and result['cached_row_fault'] is True, 'actual cached PLE row unknown scope')
        number(result, 'completed_producers', 1)
        number(result, 'warmup_calls', 1)
        number(result, 'warmup_commits', 1)
        number(result, 'faulted_call_commits', 0)
        number(result, 'selected_row_uploads', 0)
        number(result, 'selected_row_reuses', 1)
        number(result, 'selected_row_avoided_h2d', ROW_BYTES)
        number(result, 'selected_row_h2d', 0)
        number(result, 'total_value_upload_h2d', 0)
        number(result, 'complete_source_bytes', COMPLETE_SOURCE)
        number(result, 'published_roots', 0)
        require(result['source_close_refused'] is True and result['physical_fenced'] is True
                and boolean(result, 'host_fenced', False) is False
                and result['physical_reserved'] == 20*GIB
                and 2*GIB + 32*MIB <= result['host_reserved'] <= 16*GIB,
                'cached-row fault retains/fences original owners at report phase')
        require(log.splitlines().count(UNKNOWN_MARKER) == 1
                and 'GEMMA_COMPLETE_SOURCE_MATRIX_PASS' not in log
                and 'GEMMA_FINAL_OWNER_PASS' not in log,
                'exactly the cached-row unknown completion marker')
        return {'status': result['status'], 'mode': mode, 'cached_row_fault': True,
                'warmup_separate_commit': True, 'faulted_roots': 0, 'row_bytes': ROW_BYTES,
                'value_upload_h2d_distinct': True}
    if mode in ('unknown', 'resident-unknown'):
        require(result['status'] == 'ACTUAL_UNKNOWN_RETAINED'
                and result['cached_row_fault'] is False, 'ordinary fresh producer unknown scope')
        require(expected is not None, 'fresh unknown needs the seven expected cells')
        expected = matrix_expected(expected)
        first = expected[0]
        rows = len(first['calls'][0]['input_ids'])
        number(result, 'completed_producers', 1)
        number(result, 'warmup_calls', 0)
        number(result, 'warmup_commits', 0)
        number(result, 'faulted_call_commits', 0)
        number(result, 'selected_row_reuses', 0)
        number(result, 'published_roots', 0)
        number(result, 'complete_source_bytes', COMPLETE_SOURCE)
        total = number(result, 'total_value_upload_h2d', low=1, high=COMPLETE_SOURCE)
        uploads = number(result, 'selected_row_uploads', low=0, high=64)
        number(result, 'selected_row_h2d', uploads*ROW_BYTES)
        if mode == 'unknown':
            require(uploads == 0 and result['selected_row_h2d'] == 0,
                    'streamed source has no resident PLE row page to upload')
        else:
            require(uploads == rows and result['selected_row_h2d'] == total,
                    'resident cold source uploads its exact first prompt row page')
        require(result['source_close_refused'] is True and result['physical_fenced'] is True
                and boolean(result, 'host_fenced', False) is False
                and result['physical_reserved'] == (20 if mode == 'resident-unknown' else 16)*GIB
                and 2*GIB + 32*MIB <= result['host_reserved'] <= 16*GIB,
                'ordinary fresh armed producer fence at report phase')
        require(log.splitlines().count(UNKNOWN_MARKER) == 1
                and 'GEMMA_COMPLETE_SOURCE_MATRIX_PASS' not in log
                and 'GEMMA_FINAL_OWNER_PASS' not in log,
                'exactly the fresh producer unknown completion marker')
        return {'status': result['status'], 'mode': mode, 'cached_row_fault': False,
                'faulted_roots': 0, 'value_upload_h2d': total,
                'ple_row_h2d': result['selected_row_h2d'],
                'value_upload_h2d_distinct': mode == 'unknown'}
    if mode == 'resident':
        require(expected is not None, 'resident needs the seven expected cells')
        expected = matrix_expected(expected)
    require(mode == 'resident'
            and result['status'] == 'PASS_EXACT_DIAGNOSTIC_MATRIX'
            and result['exact'] is True and result['managed_source_closed'] is True
            and result['source_close_error'] is None
            and result['storage'] == 'ResidentProjections'
            and 'PLE selected row page only' in result['source_page_policy'],
            'seven-cell resident exact scope')
    require(result['physical_reserved'] == 20*GIB
            and result['host_reserved'] == 2*GIB + 32*MIB
            and result['physical_fenced'] is False and result['host_fenced'] is False,
            'report phase retains 32MiB harness over persistent baselines')
    number(result, 'expected_persistent_context_device', 4*GIB)
    number(result, 'expected_persistent_context_host', 2*GIB)
    inv = inventory(result, True)
    events = check_events(result['events'])
    seen = set()
    index = 0
    parity = [e for e in events if e.get('kind') == 'resident-normal-parity']
    total_calls = sum(len(c['calls']) for c in expected)
    require(len(parity) == total_calls, 'exact one ordinary parity per expected call')
    for cell in expected:
        for call in cell['calls']:
            event = parity[index]
            index += 1
            require(event.get('case') == cell['name'] and event.get('call') == call['call']
                    and event.get('frontier') == call['frontier']
                    and event.get('selected_id') == call['selected_id']
                    and event.get('actual_selected_id') == call['selected_id']
                    and event.get('phase') == 'proposal'
                    and event.get('schedule') == call['call']
                    and event.get('source_bound') is True
                    and event.get('state_domains_exact') is True
                    and event.get('observer') is False
                    and event.get('producer_kv_d2h') == 0
                    and event.get('resident_weight_bytes') == RESIDENT_BYTES,
                    'exact (case,call)/frontier/selected ordinary parity')
            up, _ = row_traffic(event, 'ordinary parity', rows=len(call['input_ids']),
                                parity=True)
            require(up == len(call['input_ids']), 'ordinary parity uploads its whole row page')
            require((cell['name'], call['call']) not in seen, 'no duplicate ordinary call')
            seen.add((cell['name'], call['call']))
    require(len(seen) == total_calls, 'complete ordinary call population')
    proposals = [e for e in events if e.get('kind') == 'actual-completed-proposal']
    require(proposals, 'actual completed proposals present')
    population = {(c['name'], call['call']): (tuple(call['input_ids']), call['base'],
                                              call['frontier'], call['selected_id'])
                  for c in expected for call in c['calls']}
    joined, replayed, sequences = {}, set(), []
    for event in proposals:
        comparison = event.get('comparison')
        require(type(comparison) is dict and comparison.get('exact') is True
                and comparison.get('mismatched_words') == 0
                and comparison.get('all_block_kv_and_output_comparisons') is True
                and type(comparison.get('boundary_events')) is int
                and comparison['boundary_events'] > 0,
                'independent completed proposal comparison exact')
        require(comparison.get('case') == event.get('case')
                and comparison.get('phase') == 'proposal'
                and comparison.get('call') == event.get('call')
                and comparison.get('input_ids') == event.get('input_ids')
                and comparison.get('source_bound') is True
                and comparison.get('unique_order') is True
                and comparison.get('full_vocab_words') == 262144
                and comparison.get('final_kv_planes') == 30,
                'proposal comparison joins its own case/call/input order')
        sequences.append(number(comparison, 'sequence', low=0, high=10**6))
        key = (event['case'], event['call'])
        require(key in population, 'proposal case/call exists in the source cells')
        ids, base, frontier, selected = population[key]
        require(tuple(comparison['input_ids']) == ids and comparison['base'] == base
                and comparison['frontier'] == frontier
                and comparison['expected_selected_id'] == selected
                and comparison['actual_selected_id'] == selected,
                'proposal joins the exact source cell call/order/selected token')
        bits = int(comparison['actual_max_bits'], 16)
        require(bits == number(comparison, 'max_bits', low=0, high=2**32-1),
                'actual fullV max bits equal the private packet')
        signature = (frontier, selected, bits)
        if key in joined:
            require(signature == joined[key],
                    'replayed proposal repeats one exact completed packet')
            replayed.add(key)
        else:
            joined[key] = signature
    require(len(replayed) <= len(joined), 'bounded replayed proposal population')
    require(joined.keys() == set(population), 'every expected call has a completed proposal')
    require(sequences == list(range(len(sequences))), 'unique contiguous proposal sequence')
    reuse = [e for e in events if e.get('kind') == 'resident-ple-row-reuse']
    require([e.get('repeat') for e in reuse] == [0, 1, 2]
            and [e.get('observer') for e in reuse] == [False, True, False],
            'warm, separately instrumented hit, observer-free hit order')
    shortest = min(expected, key=lambda c: c['rows'])
    rows = shortest['rows']
    for repeat, event in enumerate(reuse):
        require(event.get('case') == shortest['name'] and event.get('call') == 0
                and event.get('actual_rows') == rows
                and event.get('schedule') == repeat
                and event.get('phase') == 'proposal'
                and event.get('base') == shortest['calls'][0]['base']
                and event.get('frontier') == shortest['calls'][0]['frontier']
                and event.get('selected_id') == shortest['calls'][0]['selected_id']
                and tuple(event.get('input_ids') or ()) == tuple(shortest['calls'][0]['input_ids'])
                and event.get('source_bound') is True
                and event.get('state_domains_exact') is True
                and event.get('resident_weight_bytes') == RESIDENT_BYTES,
                'repeat joins the exact shortest cell first call')
    row_traffic(reuse[0], 'warmup', rows=rows)
    row_traffic(reuse[1], 'diagnostic hit', rows=rows, uploads=0, hits=rows, diagnostic=True)
    row_traffic(reuse[2], 'ordinary hit', rows=rows, uploads=0, hits=rows)
    c = reuse[1].get('comparison')
    require(type(c) is dict and c.get('exact') is True and c.get('mismatched_words') == 0
            and c.get('all_block_kv_and_output_comparisons') is True
            and c.get('boundary_events') == 70*rows
            and c.get('call') == 0 and c.get('frontier') == shortest['calls'][0]['frontier']
            and c.get('case') == shortest['name']
            and c.get('actual_selected_id') == shortest['calls'][0]['selected_id']
            and c.get('expected_selected_id') == shortest['calls'][0]['selected_id'],
            'actual hit full35/fullV/all15KV exact before commit')
    require(reuse[1].get('full_kv_exact') is True and reuse[2].get('full_kv_exact') is True
            and reuse[2].get('comparison') is None,
            'committed hit KV exact; observer-free repeat has no diagnostic comparison')
    require(log.splitlines().count(MATRIX_MARKER) == 1
            and 'GEMMA_COMPLETE_UNKNOWN_PASS' not in log
            and 'GEMMA_FINAL_OWNER_PASS' not in log,
            'exactly the seven-cell matrix completion marker')
    return {'status': result['status'], 'mode': mode, 'cells': 7, 'warmup_calls': 1,
            'diagnostic_hit_full35_fullV_fullKV': True, 'ordinary_hit_optional_d2h': 0,
            'selected_row_bytes': ROW_BYTES, 'resident_projection_reupload': 0,
            'cold_inventory': inv, 'actual_calls': total_calls,
            'proposal_sequences': len(sequences), 'replayed_proposals': len(replayed)}


def native_records(log):
    native = re.findall(
        r'^GEMMA_NATIVE_CODE module=(\d+) bytes=(\d+) load=(\d+) retained_host=(\d+) sha256=([0-9a-f]{64})$',
        log, re.M)
    require([int(m[0]) for m in native] == list(range(1, len(native)+1)),
            'contiguous actual native module records exactly once')
    require(native, 'actual native module records')
    elf = sum(int(m[1]) for m in native)
    load = sum(int(m[2]) for m in native)
    host = sum(int(m[3]) for m in native)
    inventory = re.findall(r'^GEMMA_NATIVE_INVENTORY kernels=(\d+) host=(\d+) device=(\d+) ',
                           log, re.M)
    require(len(inventory) == 1 and int(inventory[0][0]) == len(native),
            'one actual source-bound native inventory over every module record')
    require(0 < elf <= GIB and 0 < load <= GIB and 0 < host <= 2*GIB,
            'fixed actual native bounds')
    return {'count': len(native), 'elf': elf, 'load': load, 'host': host,
            'inventory_host': int(inventory[0][1]), 'inventory_device': int(inventory[0][2])}


def final_owner(mode, result, log):
    fault = mode.endswith('release-fault')
    snapshot = mode.startswith('snapshot')
    check_report(result, FINAL_V1)
    require(result['status'] == 'PASS_ACTUAL_FINAL_OWNER'
            and result['mode'] == mode and result['failure'] is None
            and result['numerical_replay_performed'] is False
            and result['old_order_physical_negative_performed'] is False
            and result['trained_native_natural_serving_or_performance_qualified'] is False,
            'actual Gemma ownership scope only')
    for key, expected in [('source_frontier', 0), ('warmup_completed_full35_calls', 1),
                          ('warmup_calls', 1), ('warmup_commits', 1),
                          ('observed_completed_full35_calls', 1),
                          ('total_completed_full35_calls', 2),
                          ('actual_completed_layers', 35),
                          ('actual_completed_proposed_state_planes', 30),
                          ('actual_completed_fullV', 1), ('commits', 2),
                          ('diagnostic_host_reserved', 32*MIB)]:
        number(result, key, expected)
    require(result['final_feedback_on_completed_hit'] is True,
            'final committed packet produced by an actual completed PLE hit')
    row_traffic(result['warmup'], 'final warmup', rows=1, exact_row=False)
    row_traffic(result['final_hit'], 'final hit', rows=1, uploads=0, hits=1, diagnostic=True,
                exact_row=False)
    require(result['source_close_refused'] is fault and result['physical_fenced'] is fault
            and type(result['host_fenced']) is bool, 'original managed source/physical fence')
    number(result, 'physical_reserved', 20*GIB if fault else 4*GIB)
    if fault:
        number(result, 'host_reserved', low=2*GIB+1, high=16*GIB)
    else:
        number(result, 'host_reserved', 2*GIB+32*MIB)
        require(result['host_fenced'] is False, 'known persistent host baseline not fenced')
    require((isinstance(result['source_close_error'], str)
             and 0 < len(result['source_close_error']) <= 4096)
            if fault else result['source_close_error'] is None,
            'original close refusal/known persistent baseline')
    transitions = result['root_transitions']
    require(set(transitions) == {'warmup', 'reset', 'final'}
            and all(type(v) is dict for v in transitions.values()),
            'the three published original root bindings')
    warm, reset, final = (transitions[k] for k in ('warmup', 'reset', 'final'))
    require(warm['session'] == reset['session'] == final['session']
            and warm['root'] + 1 == reset['root'] == final['root'] - 1
            and reset['session_epoch'] == warm['session_epoch'] + 1
            and final['session_epoch'] == reset['session_epoch']
            and warm['frontier'] == final['frontier'] == 1
            and reset['frontier'] == 0
            and warm['source_bound'] is True and reset['source_bound'] is True
            and final['source_bound'] is True,
            'one original session/authority: warm root+1, reset root/epoch, hit root+1')
    reader = result['original_final_owner']
    require(type(reader) is dict, 'one original final-reader receipt')
    require(reader['wrapper'] == ('Snapshot' if snapshot else 'Session')
            and reader['fault'] is fault and reader['candidate_retired'] is True
            and reader['native_admission_retained_after'] is fault
            and reader['original_service_retained_after'] is fault
            and isinstance(reader['source'], str) and reader['source']
            and re.fullmatch(r'BufferLeaseId\([0-9]+\)', reader['target_lease']),
            'original sole wrapper/source/weak native lifetime')
    native = native_records(log)
    for key, expected in [('candidate_owners_before_drop', 1),
                          ('model_owners_before_drop', 1 if snapshot else 2),
                          ('candidate_workspace_target_aliases', 0),
                          ('target_final_reader_hits', 1),
                          ('native_retirement_attempts', 1),
                          ('reader_hits_before_native_retirement', 1),
                          ('actual_native_modules_before', native['count']),
                          ('actual_native_modules_after', native['count'] if fault else 0)]:
        number(reader, key, expected)
    number(reader, 'service_generation', low=1)
    number(reader, 'target_lease_bytes', low=1, high=GIB)
    number(reader, 'aliases_inside_only_wrapper', 1)
    number(reader, 'actual_quarantined_handles_after', 0 if not fault else None,
           low=0, high=4096)
    require(not fault or reader['actual_quarantined_handles_after'] >= 1,
            'unknown-release fault quarantines the unretired original handles')
    number(reader, 'actual_native_PT_LOAD_bytes_before', native['load'])
    number(reader, 'actual_native_host_bytes_before', native['host'])
    number(reader, 'actual_native_device_bytes_before', 8*GIB+native['load'])
    require(native['inventory_device'] == 8*GIB + native['load']
            and native['inventory_host'] >= 8*GIB + native['host'],
            'actual source-bound native inventory joins every module record')
    inv = inventory(result, True)
    require(inv['native_device_bytes'] >= 8*GIB+native['load']
            and inv['native_host_bytes'] >= native['inventory_host'],
            'planned source-bound native caps cover actual device/host totals')
    require(result['prompt_text'] == 'One' and len(result['prompt_ids']) == 1
            and type(result['prompt_ids'][0]) is int
            and 0 <= result['prompt_ids'][0] < 262144,
            'genuine source tokenizer INITIAL1 text')
    selected = result['actual_selected']
    require(type(selected) is list and len(selected) == 2
            and all(type(x) is int for x in selected)
            and 0 <= selected[0] < 262144 and 0 <= selected[1] < 2**32,
            'actual one natural committed fullV greedy packet')
    require(math.isfinite(struct.unpack('<f', struct.pack('<I', selected[1]))[0]),
            'finite actual greedy score')
    marker = ('GEMMA_FINAL_OWNER_PASS mode='+mode+' real_producer=full35 target_final_reader=1 '
              'native_retirement_after_reader=true source_close_refused='+str(fault).lower() +
              ' process_retirement=true serving_performance=false')
    require(log.splitlines().count(marker) == 1
            and 'GEMMA_FINAL_OWNER_REFUSED:' not in log
            and 'GEMMA_COMPLETE_UNKNOWN_PASS' not in log
            and 'GEMMA_COMPLETE_SOURCE_MATRIX_PASS' not in log,
            'exactly the final-owner completion marker')
    return {'status': result['status'], 'mode': mode, 'fault': fault,
            'actual_native_modules': native['count'],
            'actual_native_ELF_bytes': native['elf'],
            'actual_native_PT_LOAD_bytes': native['load'],
            'actual_retained_native_host_bytes': native['host'],
            'cold_inventory': inv, 'warmup_calls': 1, 'observed_completed_full35_calls': 1,
            'total_completed_full35_calls': 2, 'commits': 2,
            'root_warm': warm['root'], 'root_reset': reset['root'], 'root_final': final['root'],
            'final_feedback_on_completed_hit': True,
            'numeric_replay_or_old_order_physical_negative_claimed': False}


def validate_fixture(mode, result, log, expected=None):
    require(mode in MODES, 'eight exact Gemma modes only')
    require(type(log) is str and 0 < len(log) <= 4*MIB, 'bounded actual fixture log')
    return (matrix(mode, result, log, expected) if mode in MATRIX_MODES
            else final_owner(mode, result, log))
