"""Qualification-only actual GDN selections; pointer bits are not control contents."""
from collections import Counter

from common import require, sha
from inputs import manifest
from packets import controls, join_packet, select_native
from tuples import CACHED, REL, STAGES, pin, recipe, rendered_signature


def observer_facts(report, rows, observer_sha):
    steps = report['steps']
    require(len(steps) == 8 and [s['call'] for s in steps] == list(range(8)),
            'actual eight-call GDN observer metadata')
    facts = []
    for call, step in enumerate(steps):
        count = rows if call == 0 else 1
        require(step['slots'] == [1] and step['num_actual_tokens'] == count
                and step['num_prefills'] == (1 if call == 0 else 0)
                and step['num_decodes'] == (0 if call == 0 else 1),
                'actual source observer single-sequence/slot metadata')
        if call == 0:
            require(step['has_initial_state'] is False
                    and step['metadata_has_initial_state'] == [False]
                    and step['prefill_has_initial_state'] == [False]
                    and step['prefill_query_start_loc'] == step['non_spec_query_start_loc'] == [0, rows]
                    and step['chunk_indices'] == [[0, 0]] and step['chunk_offsets'] == [0, 1],
                    'actual source observer fresh one-chunk metadata')
        facts.append(dict(kind='AUTHENTICATED_OBSERVER_FACT', source='package/layer_oracle.py',
                          source_sha256=observer_sha, call=call, observed=step,
                          native_control_binding_proven=False,
                          producer_source_authenticated=False))
    return facts


def extract(held, root, rows, renderer, original, baseline_root):
    observed = root / 'observed'
    report = held.document(observed / 'worker-report.json')
    native = held.document(observed / 'native-metadata-validation.json')
    driver = held.document(observed / 'driver/capture.json')
    result = held.document(observed / 'driver-result.json', limit=1024**2)
    selected = held.document(observed / 'selected-native-input-validation.json')
    require(native['mode'] == result['pass'] == 'capture' and driver['mode'] == 2
            and driver['failures'] == result['collector_status'] == 0
            and native['complete_selected_argument_fields'] is True
            and result['argument_fields_complete'] is True
            and result['observer_cuda_calls_inside_callbacks'] == 0
            and result['observer_transport_excluded'] is True
            and selected['status'] == 'ACTUAL_SELECTED_TRITON_PACKET_IMAGE_AND_CONSUMED_CONTROLS_ONLY'
            and selected['prompt_rows'] == rows and selected['control_tensors'] == 216,
            'passing full-ABI packet/image/control capture')
    arena = held.hold(observed / 'driver/data.bin', expected_sha=native['parameter_arena_sha256'],
                      expected_bytes=driver['data_bytes'])
    launches, markers = report['launches'], report['native_launch_inputs']
    require(len(launches) == len(markers) == selected['complete_typed_launches'],
            'complete original Python launch-marker cohort')
    formals = manifest('original-formals.json')
    groups, consumed, selected_joins = {}, set(), {r['launch_index']: r for r in selected['joins']}
    require(len(selected_joins) == len(launches), 'all selected original joins uniquely recorded')
    for index, (launch, marker) in enumerate(zip(launches, markers)):
        require(marker['launch_index'] == index and marker['call'] == launch['call']
                and marker['kernel_hash'] == launch['kernel_hash'], 'same actual original launch marker')
        kernel = report['kernels'][launch['kernel_hash']]
        require(kernel['hash'] == launch['kernel_hash'] and type(launch['call']) is int
                and 0 <= launch['call'] < 8, 'actual selected call/kernel identity')
        mode = 'prefill' if launch['call'] == 0 else 'cached'
        stage = (STAGES if mode == 'prefill' else CACHED).get(kernel['name'])
        if stage is None:
            continue
        actual = select_native(native, marker)
        joined = join_packet(kernel, launch, marker, actual, driver, arena)
        require(joined['event'] not in consumed, 'one actual driver launch per original marker')
        consumed.add(joined['event'])
        old_join = selected_joins[index]
        require(old_join == dict(call=joined['call'], event=joined['event'], kernel_hash=kernel['hash'],
                                launch_index=index, module_sha256=joined['module_sha256']),
                'independently recomputed packet join agrees with original validator')
        held.hold(observed / 'driver' / ('module-%04d.cubin' % actual['module']),
                  expected_sha=marker['cubin_sha256'], expected_bytes=marker['cubin_bytes'], limit=32 * 1024**2)
        ptx_path = observed / 'kernels' / (kernel['hash'] + '.ptx')
        held.hold(ptx_path, expected_sha=kernel['ptx_sha256'], expected_bytes=kernel['ptx_bytes'])
        source, descriptor = renderer.render(held.content(ptx_path).decode(), kernel, launch)
        signature, bindings = rendered_signature(descriptor)
        require(bindings == formals[mode]['stages'][stage], 'same actual original ordered pointer ABI roles/types')
        source = source.encode()
        require(sha(source) == descriptor['source_sha256'], 'same actual renderer source bytes')
        group = groups.setdefault((mode, stage), dict(source=source, descriptor=descriptor,
            signature=signature, bindings=bindings, kernel=kernel, counts=Counter(), launches=[], controls=[]))
        require((group['signature'], group['bindings'], group['source']) == (signature, bindings, source),
                'actual all18 layer/all7 cached-call source/scalar/launch/formal agreement: ' + stage)
        group['counts'][launch['call']] += 1
        group['launches'].append(dict(launch_index=index, **joined))
        group['controls'].extend(dict(launch_index=index, event=joined['event'], **fact)
                                 for fact in controls(marker, launch, kernel, stage, rows))
    require(set(groups) == {('prefill', s) for s in STAGES.values()} | {('cached', s) for s in CACHED.values()},
            'all genuine prefill/cached GDN stages')
    for (mode, stage), group in groups.items():
        require(group['counts'] == {call: 18 for call in ((0,) if mode == 'prefill' else range(1, 8))},
                'exact eighteen GDN layer population per actual call: ' + stage)
    require(sum(len(g['controls']) for g in groups.values()) == 216, 'all actual captured GDN controls joined')
    cached = next(row for row in original if row['profile'] == 'Qwen08' and row['is_decode'])
    cached_pins = []
    for old in cached['pins']:
        group = groups['cached', old['stage']]
        current = pin(old['stage'], group['descriptor'], old['file'], group['source'])
        require(current == old and group['source'] == held.content(baseline_root / REL.parent / old['file']),
                'actual seven-call cached source/scalars/launch reuse the exact original tuple')
        cached_pins.append(current)
    require(recipe('Qwen08', 1, True, cached_pins)['recipe_sha256'] == cached['recipe_sha256'],
            'actual cached recipe unchanged')
    outputs, assets, pins = {}, {}, []
    for stage in STAGES.values():
        group = groups['prefill', stage]
        file = 'qwen08_%d/%s.cu' % (rows, stage.lower())
        outputs[file] = group['source']
        pins.append(pin(stage, group['descriptor'], file, group['source']))
    for (mode, stage), group in groups.items():
        assets[mode + '/' + stage] = dict(descriptor=group['descriptor'],
            actual_kernel_hash=group['kernel']['hash'], actual_metadata=group['kernel']['metadata'],
            original_arg_names=group['kernel']['arg_names'], original_signature=group['kernel']['signature'],
            original_constants=group['kernel']['constants'], call_counts=dict(group['counts']),
            actual_launches=group['launches'], actual_captured_controls=group['controls'])
    observer = held.hold(root / 'package/layer_oracle.py')
    coverage = dict(actual_captured_control_tensors=216, actual_complete_gdn_packets=len(consumed),
        authenticated_producer_facts=[],
        authenticated_observer_facts=observer_facts(report, rows, observer['sha256']),
        unresolved=[dict(roles=['CacheIndices', 'HasInitial', 'ConvBatch', 'ConvChunkOffsets', 'ConvQueryStarts'],
            reason='Native pointer/formal bits and observer metadata do not prove the consumed control bytes; '
                   'no authenticated executed producer or dominating complete H2D payload join supplied.')],
        all_consumed_control_contents_proven=False)
    return outputs, recipe('Qwen08', rows, False, pins), dict(
        prompt_tokens=rows, assets=assets, controls=coverage,
        actual_cached_recipe_sha256=cached['recipe_sha256'], cached_source_tuple_reuse_verified=True,
        actual_packet_image_formal_join_verified=True, native_executed=False,
        numerical_qualified=False, model_execution_qualified=False, performance_qualified=False)
