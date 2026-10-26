"""Failed capture late-reflection facts only; never reconstruct or qualify packets."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import struct

from contract import read_json, request, require, sha
from discovery import HeldFiles, MIB, archive, compare_archives, compose_layouts, layout_key, owned_invocation

HERE = Path(__file__).resolve().parent
ABSENT = 2**32 - 1


def option(command, name):
    require(command.count(name) == 1, 'one explicit failed-controller argument: ' + name)
    at = command.index(name)
    require(at + 1 < len(command), 'complete failed-controller argument')
    return command[at + 1]


def environment(command, name):
    values = [value[len(name) + 1:] for value in command if value.startswith(name + '=')]
    require(len(values) == 1, 'one actual failed-container environment field: ' + name)
    return values[0]


def matching_discovery(tokenized, discoveries):
    matches = [item for item in discoveries if item.result['tokenized_request'] == tokenized]
    require(len(matches) == 1, 'exactly one held discovery matches the failed parent request; no numerical target substitution')
    return matches[0]


def prepared(held, root, value):
    inheritance = held.document(root / 'source-inheritance.json', limit=MIB)
    original = read_json(HERE / 'capture-pins.json')
    require(inheritance['ancestry_stage'] == 'capture'
            and inheritance['ancestor_manifest_sha256'] == sha(HERE / 'capture-pins.json')
            and inheritance['original_files'] == original
            and inheritance['original_observers_unchanged'] is True
            and inheritance['remaining_original_validator_bytes_unchanged'] is True
            and inheritance['native_recipes_qualified'] is False
            and inheritance['replaced_original_files'] == ['package/matrix_contract.py', 'package/validate_native.py'],
            'exact declared full-ABI source inheritance, no unchanged-successor claim')
    pins = held.document(root / 'prepared-package.json', limit=MIB)
    expected = {str(Path(name).relative_to('package')): pin for name, pin in original.items()
                if name.startswith('package/')}
    require(set(pins) == set(expected) | {'reference_request.py', 'tokenize_prompt.py', 'request.json'},
            'complete failed capture source package')
    for relative, digest in pins.items():
        held.hold(root / 'package' / relative, expected_sha=digest)
    for relative, pin in expected.items():
        if relative not in ('matrix_contract.py', 'validate_native.py'):
            held.hold(root / 'package' / relative, expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    native = held.content(root / 'package/validate_native.py').decode()
    require(native.count('prompt_rows == PROMPT_ROWS') == 1
            and hashlib.sha256(native.replace('prompt_rows == PROMPT_ROWS',
                'prompt_rows in (47,57,62,67,72)').encode()).hexdigest() == expected['validate_native.py']['sha256'],
            'only declared finite native scope guard changed')
    for staged, source in (('matrix_contract.py', 'matrix_contract.py'), ('reference_request.py', 'contract.py'),
                           ('tokenize_prompt.py', 'tokenize_prompt.py')):
        held.hold(root / 'package' / staged, expected_sha=sha(HERE / source))
    require(held.document(root / 'package/request.json', limit=MIB) == value,
            'same actual failed capture tokenizer-derived request')
    primary = held.document(root / 'package/primary-runtime-pins.json', limit=MIB)
    require(len(primary) == 20, 'complete unchanged primary runtime/math source cohort')
    for pin in primary:
        held.hold(Path(pin['path']), expected_sha=pin['sha256'], expected_bytes=pin['bytes'])
    functions = [node for node in ast.parse(native).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('validate_scopes', 'validate_pointer_observations')]
    require(len(functions) == 2, 'original strict scope and pointer validators')
    namespace = {'require': require, 'PROMPT_ROWS': value['prompt_tokens']}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<authenticated original native guards>', 'exec'), namespace)
    return namespace


def symbol(raw, name):
    require(len(raw) >= 64 and raw[:6] == b'\x7fELF\x02\x01', 'actual whole CUDA ELF resource')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    require(header[2] == 190 and header[11] == 64 and 0 < header[12] <= 8192
            and header[6] + header[12] * 64 <= len(raw), 'complete actual CUDA symbol tables')
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, header[6] + index * 64) for index in range(header[12])]
    for section in sections:
        if section[1] not in (2, 11):
            continue
        require(section[6] < len(sections) and section[9] == 24 and section[5] % 24 == 0
                and section[4] + section[5] <= len(raw), 'complete actual function-symbol table')
        strings = sections[section[6]]
        require(strings[1] == 3 and strings[4] + strings[5] <= len(raw), 'complete actual symbol-name table')
        for offset in range(section[4], section[4] + section[5], 24):
            item = struct.unpack_from('<IBBHQQ', raw, offset)
            if item[1] & 15 != 2 or item[3] == 0:
                continue
            require(item[0] < strings[5], 'actual symbol name inside bounded table')
            text = raw[strings[4] + item[0]:strings[4] + strings[5]].split(b'\0', 1)[0]
            if text == name.encode('ascii'):
                return True
    return False


def late_facts(driver, catalog, arena, guards, prompt_tokens):
    require(driver['schema'] == 'zg.vllm.blas-driver.v1' and driver['mode'] == 2
            and driver['failures'] == 64 and driver['replay_qualified'] is False
            and len(driver['contexts']) == 1, 'only original INCOMPLETE bit; no other collector failure')
    events = driver['events'] + driver['startup_events']
    require(all(event['complete'] is True and event['decoded'] is True and event['status'] == 0
                and type(event['order']) is int and event['order'] < event['exit_order'] for event in events),
            'every actual event complete, decoded and successful')
    guards['validate_scopes'](driver['scopes'], driver['events'], 8, prompt_tokens)
    pointers = guards['validate_pointer_observations'](driver, 'capture')
    functions = {row['index']: row for row in driver['functions']}
    modules = {row['index']: row for row in driver['modules']}
    require(len(functions) == len(driver['functions']) and len(modules) == len(driver['modules'])
            and len(functions) <= 1024 and len(driver['events']) <= 131072,
            'unique complete finite resource/function/event population')
    layouts = {layout_key(row): row for row in catalog['layouts']}
    require(len(layouts) == len(catalog['layouts']) and 0 < len(layouts) <= 128,
            'exact actual supplied full-image layout union')
    candidates = duplicates = 0
    present, missing, rows = set(), [], {}
    calls = set()
    for event in driver['events']:
        if not event['launch']:
            continue
        function = functions[event['function']]
        module = modules[function['module']]
        row = {key: function[key] for key in ('callback_name', 'name', 'params')}
        row['module_sha256'] = module['sha256']
        key = layout_key(row)
        require(function['reflected'] is True and function['prelaunch_bound'] is True
                and function['prelaunch_binding_kind'] in (1, 2)
                and 0 <= function['prelaunch_binding_order'] < event['order']
                and function['handle'] > 0 and function['native_module'] > 0 and module['ready'] is True
                and function['context'] == module['context'] == event['context'] == driver['contexts'][0]['handle']
                and module['load_order'] < event['order']
                and (module['unload_order'] == 0 or event['exit_order'] < module['unload_order'])
                and function['extent'] == row['params'][-1][0] + row['params'][-1][1]
                and 0 <= event['shared'] <= 128 * 1024 and event['attributes'] == 0
                and len(event['grid']) == len(event['block']) == 3
                and all(type(value) is int and value > 0 for value in event['grid'] + event['block']),
                'actual live reflected full-image function and original finite launch controls')
        offset, size = event['args']
        calls.add(event['scope'])
        if offset == ABSENT:
            require(size == 0 and event['layout'] == ABSENT and event['api'] in ('cuLaunchKernel', 'cuLaunchKernel_ptsz')
                    and key not in layouts and function['prelaunch_binding_kind'] == 1,
                    'only actual absent declared layout; missing packets remain absent')
            getters = [item for item in events if item['api'] == 'cuModuleGetFunction'
                       and item['context'] == function['context'] and item['values'][3] == function['handle']
                       and item['values'][1] == function['native_module']
                       and item['exit_order'] == function['prelaunch_binding_order'] < event['order']]
            require(len(getters) == 1, 'one successful public same-context prelaunch function getter')
            getter = getters[0]
            name_offset, name_bytes = getter['data']
            require(type(name_offset) is type(name_bytes) is int and 0 < name_bytes <= 2048
                    and 0 <= name_offset <= arena['bytes'] - name_bytes
                    and os.pread(arena['fd'], name_bytes, name_offset) == function['name'].encode('ascii') + b'\0',
                    'actual successful public getter requested this exact reflected entry')
            loads = [item for item in events if item['api'] in ('cuModuleLoad', 'cuModuleLoadData',
                       'cuModuleLoadDataEx', 'cuModuleLoadFatBinary', 'cuLibraryGetModule') and item['context'] == module['context']
                     and item['values'][16] == function['native_module']
                     and item['order'] < module['load_order'] < item['exit_order'] < getter['order']]
            require(len(loads) == 1, 'one actual native load/resource/getter lineage before missing packet launch')
            load = loads[0]
            if load['api'] == 'cuLibraryGetModule':
                library = load['values'][1]
                origins = [item for item in events if item['api'] in ('cuLibraryLoadData', 'cuLibraryLoadFromFile')
                           and item['values'][16] == library > 0 and item['exit_order'] < load['order']]
                require(len(origins) == 1 and not any(item['api'] == 'cuLibraryUnload'
                        and item['values'][0] == library and origins[0]['exit_order'] < item['exit_order'] < event['exit_order']
                        for item in events), 'actual successful library owner remains live through module/getter/launch')
            require(sum(candidate['ready'] is True and candidate['context'] == module['context']
                        and load['order'] < candidate['load_order'] < load['exit_order']
                        for candidate in modules.values()) == 1,
                    'public load materializes exactly this whole executable resource')
            require(not any(item['api'] == 'cuModuleUnload' and item['context'] == function['context']
                            and item['values'][0] == function['native_module']
                            and getter['exit_order'] < item['exit_order'] < event['exit_order'] for item in events),
                    'same actual native module remains live through launch')
            require(key not in rows or rows[key] == row, 'consistent complete late-reflected ABI')
            rows[key] = row
            missing.append({'event': event['sequence'], 'function': function['index'], 'module': module['index'],
                            'getter_order': getter['order'], 'getter_exit_order': getter['exit_order'],
                            'load_order': load['order'], 'load_exit_order': load['exit_order']})
            continue
        require(key in layouts and layouts[key] == row and 0 <= event['layout'] < 128
                and size == function['extent'] and 0 <= offset <= arena['bytes'] - size,
                'all other actual launch packets have complete bounded original fields')
        raw = os.pread(arena['fd'], size, offset)
        require(len(raw) == size, 'complete actual present packet; no reconstruction')
        for at in range(0, size - 7, 4):
            value = int.from_bytes(raw[at:at + 8], 'little')
            if value:
                candidates += 1
                pointer = (event['scope'], value)
                if pointer in pointers:
                    if pointer in present:
                        duplicates += 1
                    present.add(pointer)
    require(bool(missing) and calls == set(range(8)), 'nonvacuous missing-layout-only failure after all8 natural calls')
    guards['validate_pointer_observations'](driver, 'capture', (candidates, duplicates, present))
    return {'schema': 'zg.vllm.blas-layouts.v1', 'layouts': [rows[key] for key in sorted(rows)]}, missing


class LateLayoutCapture(HeldFiles):
    def __init__(self, root, controller, controller_sha, pins_path, pins_sha, discoveries):
        super().__init__()
        self.root = root
        try:
            tokenized = request(self.document(root / 'tokenize/tokenized-request.json', limit=MIB), tokenized=True)
            discovery = matching_discovery(tokenized, discoveries)
            outer = self.document(controller, controller_sha, limit=MIB)
            step = outer['step']
            require(outer['complete'] is False and outer['inference_qualified'] is False and bool(outer['failure'])
                    and step['exit_code'] == step['reaped_exit_code'] == step['observed_unreaped_exit_code'] == 1
                    and step['leader_reaped'] and step['reaped'] and step['group_absent'] and not step['running']
                    and not step['retained_group'] and not step['cleanup_errors'] and step['failure'] is None
                    and step['group_term_before_reap'] and step['group_kill_before_reap']
                    and step['stdout_eof'] and step['stderr_eof'] and not step['timed_out'],
                    'explicit failed outer controller with complete actual physical closure')
            command = step['command']
            require(option(command, '--output') == str(root) and option(command, '--mode') == 'capture'
                    and option(command, '--discovery') == str(discovery.root),
                    'failed controller stage is this same-request packed capture')
            old = read_json(HERE / 'predecessor-pins.json')
            pinned = self.document(pins_path, pins_sha, limit=MIB)
            script = Path(command[2])
            require(script.name == 'run_reference.py' and all(pinned[str(script.parent / name)] == pin['sha256']
                    for name, pin in old.items()), 'explicit authenticated predecessor source pinset; never relabel new bytes')
            for path, digest in pinned.items():
                if Path(path).parent != script.parent:
                    self.hold(Path(path), expected_sha=digest)
            for name in ('stdout', 'stderr'):
                path = Path(step[name])
                require(path == controller.parent / ('command.' + name + '.log'), 'exact original controller stream routing')
                self.hold(path)
            require(self.content(Path(step['stderr'])).decode().rstrip().endswith(
                    'ValueError: owned original failed; retained complete failure scope'), 'exact original observed-stage refusal')
            require(tokenized == discovery.result['tokenized_request'], 'same original request/tokenizer/source/generation/batch')
            require(self.document(Path(option(command, '--request')), limit=MIB) == discovery.result['request'],
                    'failed controller used this exact original request input')
            guards = prepared(self, root, tokenized)
            owned_invocation(self, root / 'tokenize')
            owned_invocation(self, root / 'plain')
            observed = owned_invocation(self, root / 'observed', failed=True)
            require(observed['request'] == discovery.result['request'] and not observed.get('failure'),
                    'collector refusal after complete owned original, not unrelated controller error')
            for lane in ('plain', 'observed'):
                source = self.document(root / lane / 'main-source-after.json', limit=MIB)
                lifetime = self.document(root / lane / 'lifecycle.json')
                require(source == lifetime['source'] and source['manifest_sha256'] ==
                        'c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09'
                        and source['text_roles'] == 320 and source['all_tensors'] == 488
                        and source['full_payload_freshly_verified'] is True,
                        'same fully verified original BF16 source and complete original lifetime')
            require(self.content(root / 'observed/stderr.log').decode().rstrip().endswith(
                    'Exception: Call to collective_rpc method failed: inspect original collector refusal; never waive unknown/layout failures'),
                    'exact collector-only original RPC refusal')
            require(not (root / 'reference-result.json').exists() and not (root / 'observed/worker-report.json').exists()
                    and not (root / 'observed/result.json').exists(), 'missing final receipts remain missing and parent stays failed')
            catalog = self.document(root / 'layout-catalog.json', environment(observed['argv'], 'ZG_DRIVER_LAYOUT_CATALOG_SHA'))
            expected = compose_layouts(discoveries)
            require(catalog['schema'] == expected['schema'] and catalog['layouts'] == expected['layouts']
                    and all(catalog[key] is False for key in ('native_memory_qualified', 'mathematical_replay_qualified',
                                                               'model_qualified', 'performance_qualified')),
                    'failed run used the complete actually validated discovery layout union')
            driver = self.document(root / 'observed/driver/capture.json')
            result = self.document(root / 'observed/driver-result.json', limit=MIB)
            require(result['pass'] == 'capture' and result['collector_status'] == 1
                    and result['argument_fields_complete'] is False
                    and result['calls'] == [{'call': i, 'selected_model_head_sampler': True} for i in range(8)]
                    and result['observer_transport_excluded'] is True and result['observer_cuda_calls_inside_callbacks'] == 0
                    and result['native_memory_qualified'] is result['mathematical_replay_qualified'] is False,
                    'original explicit missing-packet refusal, never a packed capture pass')
            require(driver['image_arena'] == 256 * MIB and driver['metadata_arena'] == 64 * MIB
                    and type(driver['state_bytes']) is int and 0 < driver['state_bytes'] <= 60 * MIB
                    and 0 <= driver['data_bytes'] <= 64 * MIB - driver['state_bytes'], 'unchanged complete bounded collector arenas')
            arena = self.hold(root / 'observed/driver/data.bin', expected_bytes=driver['data_bytes'])
            self.layout, missing = late_facts(driver, catalog, arena, guards, tokenized['prompt_tokens'])
            modules = {item['index']: item for item in driver['modules']}
            for module in modules.values():
                require(module['ready'] is True, 'complete actual materialized executable cohort')
                self.hold(root / 'observed/driver' / ('module-%04d.cubin' % module['index']),
                          expected_sha=module['sha256'], expected_bytes=module['bytes'], limit=32 * MIB)
            for fact in missing:
                function = next(item for item in driver['functions'] if item['index'] == fact['function'])
                require(symbol(self.content(root / 'observed/driver' / ('module-%04d.cubin' % fact['module'])),
                               function['name']), 'late-reflected entry is actually defined in the held whole executable')
            progress = self.document(root / 'observed/final-norm-progress.json')
            require(len(progress['calls']) == 8 and [row['call'] for row in progress['calls']] == list(range(8))
                    and 0 < len(progress['sampler']) <= 8
                    and [row['call'] for row in progress['sampler']] == list(range(len(progress['sampler']))),
                    'actual ordered sampler progress stays partial; all8 complete trace scopes are separate evidence')
            for lane in ('plain', 'observed'):
                states = self.document(root / lane / 'all42/report.json')
                original_states = discovery.document(discovery.root / lane / 'all42/report.json')
                require(all(states[name] == original_states[name] for name in
                        ('active_domains', 'outputs', 'rows', 'sequences', 'bytes', 'prior_bytes',
                         'archives', 'prior_archives', 'initial_gdn_prior_consumed', 'initial_gdn_prior_bytes_observed')),
                        'complete actual all24/all42 and consumed-prior metadata equals same-request reference')
            current, expected, unpublished = {}, {}, []
            for key, original in discovery.raw.items():
                lane, kind, *layer = key
                if kind == 'common': path = root / lane / 'raw.bin'
                elif kind in ('parameters', 'final'): path = root / lane / kind / 'raw.bin'
                else: path = root / lane / ('all42' if kind == 'all42' else 'all42-prior') / str(layer[0]) / 'raw.bin'
                if key == ('observed', 'parameters') and path.stat().st_size == 0:
                    require(original['item']['bytes'] == 2048, 'only the original1024-wide norm parameter archive is unpublished')
                    self.hold(path, expected_bytes=0, expected_sha=hashlib.sha256(b'').hexdigest())
                    unpublished.append({'path': str(path), 'actual_bytes': 0, 'reference_bytes': 2048,
                                        'reconstructed': False})
                    continue
                item = self.hold(path, expected_bytes=original['item']['bytes'], expected_sha=original['item']['sha256'], limit=32 * MIB)
                # The reference metadata remains reference metadata. No absent
                # failed-worker report/record or missing native packet is minted.
                current[key] = {'item': item, 'records': original['records']}
                expected[key] = original
            require(len(current) >= 101 and len(unpublished) <= 1, 'every actually retained complete raw stream compared')
            compared_bytes = compare_archives(expected, current)
            self.facts = {'schema': 'zerogravity.qwen35-cuda-reference.failed-late-layout.v1',
                'failed_capture': str(root), 'controller': str(controller), 'controller_sha256': controller_sha,
                'source_pins': str(pins_path), 'source_pins_sha256': pins_sha, 'parent_failed': True,
                'original_collector_failure_bits': 64, 'missing_argument_launches': missing,
                'layout_count': len(self.layout['layouts']), 'same_request_discovery': str(discovery.root),
                'diagnostic_prompt_tokens': tokenized['prompt_tokens'],
                'exported_scope': 'exact callback/entry/whole-image SHA/reflected formals only; no numerical target transfer',
                'matched_complete_raw_streams': len(current), 'matched_complete_raw_bytes': compared_bytes,
                'unpublished_parameter_archives': unpublished,
                'worker_loaded_parameters_after_audit_complete': False, 'original_postvalidation_complete': False,
                'missing_worker_report_reconstructed': False,
                'sampler_progress_calls': len(progress['sampler']), 'completed_original_trace_calls': 8,
                'missing_argument_packets_reconstructed': False, 'native_recipes_qualified': False,
                'model_execution_qualified': False, 'performance_qualified': False}
            discovery.check()
            self.check()
        except BaseException:
            self.close()
            raise

    def provenance(self):
        return {**self.facts, 'held_inputs': self.pins()}
