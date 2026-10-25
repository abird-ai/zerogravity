#!/usr/bin/env python3
"""Grouped GPU-free controls: typed semantic negatives, complete source/closure,
two-artifact mode binding, original TERM/2KILL/1Reap/EOF owner and 70-FD regression guard."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

CACHE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
GIB = 1024**3
MIB = 1024**2
ROW = 17920


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_fail(fn, label):
    try:
        fn()
    except BaseException:
        return
    raise AssertionError('accepted negative control: '+label)


def inventory_text(resident=True, native_device=None, native_host=None):
    """The exact source-derived cold inventory, per storage policy."""
    scratch = 2121728 if resident else 40960
    rows = 262144 if resident else 2048
    matrix = 4558946304 if resident else 0
    largest = 805306368 if resident else 1048576
    matrices = 277 if resident else 0
    windows = 4516 if resident else 0
    control = 18153472 if resident else 0
    capacity = 58 if resident else 0
    row_bytes = 17920 if resident else 0
    device = 8*GIB if native_device is None else native_device
    host = 8*GIB if native_host is None else native_host
    owned = (MIB + matrix + 2502656 + 1146880 + 114688 + scratch + 2560 + 2048)
    six = 14155776
    with_six = owned + six + 12336 + 6*256 + device
    return ('unique_source_matrices: 278 complete_source_bytes: 9256566784 '
            'source_page_bytes: 1048576 selected_row_cache_capacity: '+str(capacity)+' '
            'selected_row_cache_row_bytes: '+str(row_bytes)+' value_slot_bytes: 2502656 '
            'value_slots: 24 source_gamma_bytes: 1146880 math_scratch_bytes: 114688 '
            'projection_scratch_bytes: '+str(scratch)+' status_and_packet_bytes: 2560 '
            'original_greedy_workspace_bytes: 12336 single_root_payload_bytes: 2359296 '
            'six_root_device_bytes: '+str(six)+' owned_device_bytes: '+str(owned)+' '
            'device_with_six_roots_bytes: '+str(with_six)+' '
            'host_upper_bytes: '+str(host+4*GIB+2*MIB)+' native_host_bytes: '+str(host)+' '
            'native_device_bytes: '+str(device)+' largest_binding_bytes: '+str(largest)+' '
            'resident_matrices: '+str(matrices)+' resident_matrix_bytes: '+str(matrix)+' '
            'resident_upload_windows: '+str(windows)+' '
            'resident_control_host_bytes: '+str(control)+' root_pool_metadata_bytes: 256000 '
            'root_initialization_host_bytes: 7077888 projection_max_rows: '+str(rows)+' '
            'persistent_context_device_bytes: 4294967296 '
            'persistent_context_host_bytes: 2147483648 '
            'complete_physical_device_bytes: '+str(with_six+4*GIB))


def traffic(rows, uploads=None, hits=None, diagnostic=False):
    up = rows if uploads is None else uploads
    hit = 0 if hits is None else hits
    return {'uploads':up,'reuses':hit,'ple_row_h2d':up*ROW,'value_upload_total':up*ROW,
            'avoided_h2d':hit*ROW,'producer_d2h':4 if diagnostic else 0,
            'kv_diagnostic_d2h':4 if diagnostic else 0}


def expected_cells():
    cells = []
    for index in range(7):
        cells.append({'name':'cell'+str(index),'rows':2,'prompt_ids':[1,2],
                      'calls':[{'call':0,'base':0,'frontier':2,
                                'selected_id':10+index,'eos':True,'input_ids':[1,2]}]})
    return cells


MAX_BITS = 1065353216


def comparison(cell, sequence, rows=2, boundary=None):
    call = cell['calls'][0]
    return {'call':0,'exact':True,'mismatched_words':0,
            'boundary_events':70*rows if boundary is None else boundary,
            'first_difference':None,'actual_selected_id':call['selected_id'],
            'actual_max_bits':format(MAX_BITS,'08x'),
            'expected_selected_id':call['selected_id'],'frontier':call['frontier'],
            'all_block_kv_and_output_comparisons':True,'case':cell['name'],
            'phase':'proposal','schedule':0,'sequence':sequence,'input_ids':[1,2],
            'base':0,'source_bound':True,'unique_order':True,'prior_kv_comparisons':2,
            'final_kv_planes':30,'full_vocab_words':262144,'max_bits':MAX_BITS}


def parity_event(cell):
    call = cell['calls'][0]
    return {'kind':'resident-normal-parity','case':cell['name'],'phase':'proposal','call':0,
            'schedule':0,'input_ids':[1,2],'base':0,'frontier':call['frontier'],
            'selected_id':call['selected_id'],'actual_selected_id':call['selected_id'],
            'actual_max_bits':format(MAX_BITS,'08x'),'max_bits':MAX_BITS,'observer':False,
            'source_bound':True,'state_domains_exact':True,'producer_kv_d2h':0,
            'resident_weight_bytes':4558946304,'ple_row_h2d':2*ROW,
            'value_upload_total':2*ROW,'ple_row_uploads':2,'ple_row_reuses':0,
            'ple_row_avoided_h2d':0}


def reuse_event(cell, repeat, comparison_value, full_kv):
    call = cell['calls'][0]
    event = {'kind':'resident-ple-row-reuse','case':cell['name'],'phase':'proposal','call':0,
             'schedule':repeat,'repeat':repeat,'input_ids':[1,2],'base':0,
             'frontier':call['frontier'],'observer':repeat==1,'actual_rows':2,
             'resident_weight_bytes':4558946304,'source_bound':True,
             'state_domains_exact':True,'full_kv_exact':full_kv,
             'selected_id':call['selected_id'],'max_bits':MAX_BITS,
             'actual_max_bits':format(MAX_BITS,'08x'),'comparison':comparison_value}
    event.update(traffic(2,uploads=2 if repeat==0 else 0,
                         hits=0 if repeat==0 else 2,
                         diagnostic=repeat==1))
    return event


def matrix_result():
    cells = expected_cells()
    events = [{'kind':'pre-cancel','root':{'root':0,'frontier':0}},
              {'kind':'completed-Ready-drop','exact':True,'comparison':None,
               'root':{'root':0,'frontier':0}}]
    for sequence, cell in enumerate(cells):
        events.append({'kind':'actual-completed-proposal','case':cell['name'],
                       'phase':'proposal','call':0,'input_ids':[1,2],
                       'before':{'root':0,'frontier':0},'comparison':comparison(cell,sequence)})
        events.append({'kind':'actual-committed-call','case':cell['name'],'call':0,
                       'selected_id':cell['calls'][0]['selected_id'],'eos':True,
                       'root':{'root':sequence+1,'frontier':2}})
        events.append({'kind':'source-ID-decoded-output','case':cell['name'],'ids':[10],
                       'text':'x','sequence_decode_only':True})
    events.append({'kind':'actual-completed-late-cancel','comparison':None,
                   'root':{'root':7,'frontier':2},'error':'Gemma source-bound GPU call cancelled'})
    events.append({'kind':'actual-completed-refusal','comparison':None,
                   'root':{'root':7,'frontier':2},'error':'deliberate completed Gemma refusal'})
    events += [parity_event(cell) for cell in cells]
    events.append(reuse_event(cells[0],0,None,None))
    events.append(reuse_event(cells[0],1,comparison(cells[0],len(cells)),True))
    events.append(reuse_event(cells[0],2,None,True))
    return {'schema':'zg.gemma4-e2b-original-hip-complete-qualification.v1',
            'status':'PASS_EXACT_DIAGNOSTIC_MATRIX','exact':True,
            'managed_source_closed':True,'source_close_error':None,
            'storage':'ResidentProjections','source_page_policy':'PLE selected row page only',
            'physical_reserved':4*GIB,'host_reserved':2*GIB+32*MIB,
            'physical_fenced':False,'host_fenced':False,'events':events,
            'inventory':inventory_text(True),'failure':None,
            'trained_native_serving_or_performance_qualified':False,
            'context_retirement':'original HIP context retained for process lifetime',
            'diagnostic_host_reserved':32*MIB,
            'expected_persistent_context_device':4*GIB,
            'expected_persistent_context_host':2*GIB}


def unknown_result(mode):
    cached = mode == 'resident-row-unknown'
    fresh_row = 2*ROW if mode == 'resident-unknown' else 0
    return {'schema':'zg.gemma4-e2b-original-hip-unknown.v1',
            'status':'ACTUAL_UNKNOWN_RETAINED','cached_row_fault':cached,
            'complete_source_bytes':9256566784,
            'completed_producers':1,'warmup_calls':1 if cached else 0,
            'warmup_commits':1 if cached else 0,'faulted_call_commits':0,
            'selected_row_reuses':1 if cached else 0,
            'selected_row_avoided_h2d':ROW if cached else 0,
            'selected_row_uploads':0 if cached else (2 if mode == 'resident-unknown' else 0),
            'selected_row_h2d':0 if cached else fresh_row,
            'total_value_upload_h2d':0 if cached else (fresh_row or MIB),
            'published_roots':0,'source_close_refused':True,
            'source_close_error':'original managed source close refused under armed fault',
            'physical_fenced':True,'host_fenced':False,
            'physical_reserved':(20 if cached or mode == 'resident-unknown' else 16)*GIB,
            'host_reserved':2*GIB+32*MIB,
            'context_retirement':'original HIP context retained for process lifetime',
            'diagnostic_host_reserved':32*MIB,
            'inventory':inventory_text(mode != 'unknown'),
            'trained_native_serving_or_performance_qualified':False}


NATIVE_MODULES = 6
NATIVE_ELF = 1000
NATIVE_LOAD = 100
NATIVE_HOST = 200


def final_result(mode):
    fault = mode.endswith('release-fault')
    snapshot = mode.startswith('snapshot')
    load = NATIVE_MODULES*NATIVE_LOAD
    host = NATIVE_MODULES*NATIVE_HOST
    native_device = 8*GIB+load
    native_host = 8*GIB+host+3*MIB
    reader = {'wrapper':'Snapshot' if snapshot else 'Session','fault':fault,
              'candidate_retired':True,'native_admission_retained_after':fault,
              'original_service_retained_after':fault,'source':'source',
              'target_lease':'BufferLeaseId(7)',
              'candidate_owners_before_drop':1,
              'model_owners_before_drop':1 if snapshot else 2,
              'candidate_workspace_target_aliases':0,'target_final_reader_hits':1,
              'native_retirement_attempts':1,'reader_hits_before_native_retirement':1,
              'actual_native_modules_before':NATIVE_MODULES,
              'actual_native_modules_after':NATIVE_MODULES if fault else 0,
              'service_generation':1,'target_lease_bytes':12,'aliases_inside_only_wrapper':1,
              'actual_quarantined_handles_after':1 if fault else 0,
              'actual_native_PT_LOAD_bytes_before':load,
              'actual_native_host_bytes_before':host,
              'actual_native_device_bytes_before':native_device}
    return {'schema':'zg.gemma4-original-hip-final-feedback-owner.v1',
            'status':'PASS_ACTUAL_FINAL_OWNER','mode':mode,'failure':None,
            'numerical_replay_performed':False,'old_order_physical_negative_performed':False,
            'trained_native_natural_serving_or_performance_qualified':False,
            'source_frontier':0,'warmup_completed_full35_calls':1,'warmup_calls':1,
            'warmup_commits':1,'observed_completed_full35_calls':1,
            'total_completed_full35_calls':2,'actual_completed_layers':35,
            'actual_completed_proposed_state_planes':30,'actual_completed_fullV':1,
            'commits':2,'diagnostic_host_reserved':32*MIB,
            'final_feedback_on_completed_hit':True,
            'warmup':traffic(1),'final_hit':traffic(1,uploads=0,hits=1,diagnostic=True),
            'source_close_refused':fault,'physical_fenced':fault,'host_fenced':False,
            'physical_reserved':20*GIB if fault else 4*GIB,
            'host_reserved':2*GIB+32*MIB,
            'source_close_error':'refused' if fault else None,
            'root_transitions':{'warmup':bounding(1,0,1),'reset':bounding(2,1,0),
                                'final':bounding(3,1,1)},
            'original_final_owner':reader,
            'inventory':inventory_text(True,native_device,native_host),
            'prompt_text':'One','prompt_ids':[1],'actual_selected':[123,MAX_BITS]}


def bounding(root, epoch, frontier):
    return {'session':7,'root':root,'session_epoch':epoch,'frontier':frontier,
            'source_bound':True,'source':'SourceIdentity(1)'}


def native_log(count=NATIVE_MODULES, elf=NATIVE_ELF, load=NATIVE_LOAD,
               host=NATIVE_HOST, index=1, inventory=True):
    lines = []
    for step in range(count):
        lines.append('GEMMA_NATIVE_CODE module='+str(index+step)+' bytes='+str(elf)+
                     ' load='+str(load)+' retained_host='+str(host)+' sha256='+('a'*64))
    if inventory:
        lines.append('GEMMA_NATIVE_INVENTORY kernels='+str(count)+' host='+
                     str(8*GIB+count*host+3*MIB)+' device='+str(8*GIB+count*load)+
                     ' persistent_device=4294967296 persistent_host=2147483648')
    return '\n'.join(lines)+'\n'


def marker(mode):
    fault = mode.endswith('release-fault')
    return ('GEMMA_FINAL_OWNER_PASS mode='+mode+' real_producer=full35 target_final_reader=1 '
            'native_retirement_after_reader=true source_close_refused='+str(fault).lower()+
            ' process_retirement=true serving_performance=false')


def binding_value(closure_pin, remote_base):
    proof = {'bytes':100,'sha256':'b'*64}
    return {'schema':'zg.gemma-ple-source21-build-binding.v1',
            'source_base':str(remote_base),'source_closure':closure_pin,
            'hydrated_closure':proof,
            'artifacts':{'gemma4_hip_matrix':proof,'gemma4_hip_final_owner':proof},
            'mode_artifacts':{'resident':'gemma4_hip_matrix','resident-row-unknown':'gemma4_hip_matrix',
                              'unknown':'gemma4_hip_matrix','resident-unknown':'gemma4_hip_matrix',
                              'session-known':'gemma4_hip_final_owner',
                              'session-release-fault':'gemma4_hip_final_owner',
                              'snapshot-known':'gemma4_hip_final_owner',
                              'snapshot-release-fault':'gemma4_hip_final_owner'},
            'build_receipt':{'bytes':5,'sha256':'f'*64},'provenance':proof}


def owner_receipt(pid, cleanup, eof=True):
    return {'observed_exit':0,'errors':[],'pid':pid,'foreground_cleanup':cleanup,
            'log':{'retained':10,'total':10,'eof':eof,'closed':eof}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = args.payload
    archive_path = args.archive
    if not (payload.is_absolute() and payload.is_dir() and not payload.is_symlink()):
        raise ValueError('explicit prepared payload directory')
    if not (archive_path.is_absolute() and archive_path.is_file()
            and not archive_path.is_symlink()):
        raise ValueError('explicit prepared archive file')
    if not (args.output.is_absolute() and not os.path.lexists(args.output)):
        raise ValueError('fresh absolute control output')
    if Path(os.path.realpath(args.output)) != args.output:
        raise ValueError('canonical symlink-free control output path')
    if args.output.is_relative_to(Path(os.path.realpath(payload))):
        raise ValueError('control output outside the payload')
    args.output.mkdir(parents=True, exist_ok=False)
    prepared_path = payload.parent/'PREPARED.json'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(payload/'runner'), str(payload/'physical-runner')]
    from pins import parsed, pin, require
    import bootstrap
    import bootstrap_receipt as receipt_mod
    import fixture_semantic as semantic
    import joint_data
    import launch
    import modes as source_modes
    import run as physical_run
    owner = load('control_original_owner',
                 payload/'source/tools/qwen27b-full-window-reference/owned_container_v4.py')
    results = {'status':'UNQUALIFIED','checks':[],'errors':[],'GPU_or_model':False,
               'remote_effect':False,'network':False,'cargo':False}
    temp = Path(tempfile.mkdtemp(prefix='gemma-ple-source-control-'))

    def check(name, fn):
        try:
            fn()
            results['checks'].append(name)
        except BaseException as error:
            results['errors'].append(name+': '+repr(error))

    try:
        control, control_pin = parsed(payload/'control.json')
        closure, closure_pin = parsed(payload/'source-closure.json')
        manifest, manifest_pin = parsed(payload/'assembled-source.json', limit=16*1024**2)
        archive_pin = pin(archive_path)
        prepared = json.loads(prepared_path.read_text())
        actual = {p.relative_to(payload).as_posix(): pin(p)
                  for p in sorted(payload.rglob('*')) if p.is_file()}
        require(set(control['records']) == set(actual)-{'control.json','source-closure.json'}
                and control['records'] == {k: actual[k] for k in control['records']}
                and set(closure['records']) == set(actual)-{'source-closure.json'}
                and closure['records'] == {k: actual[k] for k in closure['records']},
                'control and closure records cover the prepared payload')
        require(prepared['source_archive'] == archive_pin
                and prepared['source_closure'] == closure_pin
                and prepared['controller'] == control_pin
                and prepared['source_manifest'] == manifest_pin == control['source_manifest']
                and prepared['source_files'] == len(manifest['files']) == closure['source_count'] > 0
                and prepared['tar_members'] == len(closure['records'])+1
                and prepared['payload_files'] == len(closure['records'])+1
                and prepared['all_source_bytes_modes_verified'] is True
                and prepared['all_payload_bytes_modes_verified'] is True
                and prepared['base'] == launch.BASE
                and prepared['changed_paths'] == {} == control['changed_paths']
                and prepared['no_closure_json_shipped'] is True,
                'PREPARED pins archive/closure/controller/complete source manifest')
        require(not (payload/'closure.json').exists(), 'offline hydration closure.json not shipped')

        def diagnostics():
            legal = {'uploads':0,'reuses':1,'ple_row_h2d':0,'avoided_h2d':ROW,
                     'producer_d2h':4,'kv_diagnostic_d2h':8}
            semantic.row_traffic(legal,'legal',rows=1,uploads=0,hits=1,diagnostic=True)
            semantic.diagnostic_bytes(legal,'producer_d2h')
            for key in ('producer_d2h','kv_diagnostic_d2h'):
                for value in (1,2,3,5,6,7,65*MIB,-4):
                    bad = dict(legal); bad[key] = value
                    expect_fail(lambda bad=bad: semantic.diagnostic_bytes(bad,key), key+' '+str(value))
                for value in (True,4.0,'4',None):
                    bad = dict(legal); bad[key] = value
                    expect_fail(lambda bad=bad: semantic.diagnostic_bytes(bad,key), key+' type')
            ordinary = {'uploads':1,'reuses':0,'ple_row_h2d':ROW,'avoided_h2d':0,
                        'producer_d2h':0,'kv_diagnostic_d2h':0}
            semantic.row_traffic(ordinary,'ordinary',rows=1,uploads=1,hits=0)
            for key in ('producer_d2h','kv_diagnostic_d2h'):
                bad = dict(ordinary); bad[key] = 4
                expect_fail(lambda bad=bad: semantic.row_traffic(bad,'bad',rows=1), 'ordinary '+key)
        check('typed-positive-4-byte-diagnostic-vs-exact-zero-ordinary', diagnostics)

        cells = expected_cells()
        expected_source = {'cells':cells}
        semantic.matrix_expected(expected_source)
        expect_fail(lambda: semantic.matrix_expected({'cells':cells+[cells[0]]}), 'duplicate cell')
        expect_fail(lambda: semantic.matrix_expected({'cells':cells[:-1]}), 'missing cell')
        log = semantic.MATRIX_MARKER+'\n'
        semantic.validate_fixture('resident', matrix_result(), log, expected_source)

        def events_of(result, kind):
            return [e for e in result['events'] if e['kind'] == kind]

        def matrix_negatives():
            base = matrix_result()
            def parity(result):
                return events_of(result, 'resident-normal-parity')
            def proposal(result):
                return events_of(result, 'actual-completed-proposal')
            for name, mutate in (
                ('duplicate parity', lambda r: r['events'].append(copy.deepcopy(parity(r)[0]))),
                ('missing parity', lambda r: r['events'].remove(parity(r)[0])),
                ('wrong selected', lambda r: parity(r)[0].__setitem__('selected_id',99)),
                ('wrong frontier', lambda r: parity(r)[0].__setitem__('frontier',3)),
                ('wrong row bytes', lambda r: parity(r)[0].__setitem__('ple_row_h2d',0)),
                ('row bytes not uploaded pages', lambda r: parity(r)[0].__setitem__('ple_row_uploads',1)),
                ('avoided not reused pages', lambda r: parity(r)[0].__setitem__('ple_row_avoided_h2d',ROW)),
                ('ordinary diagnostic positive', lambda r: parity(r)[0].__setitem__('producer_kv_d2h',4)),
                ('value upload distinct from rows', lambda r: parity(r)[0].__setitem__('value_upload_total',3*ROW)),
                ('missing proposal', lambda r: r['events'].remove(proposal(r)[0])),
                ('proposal mismatch', lambda r: proposal(r)[0]['comparison'].__setitem__('mismatched_words',1)),
                ('proposal other cell', lambda r: proposal(r)[0].__setitem__('case',cells[1]['name'])),
                ('proposal sequence gap', lambda r: proposal(r)[1]['comparison'].__setitem__('sequence',5)),
                ('proposal stale planes', lambda r: proposal(r)[0]['comparison'].__setitem__('final_kv_planes',29)),
                ('proposal hex mismatch', lambda r: proposal(r)[0]['comparison'].__setitem__('actual_max_bits','00000000')),
                ('proposal unbound order', lambda r: proposal(r)[0]['comparison'].__setitem__('unique_order',False)),
                ('report-phase host missing harness', lambda r: r.__setitem__('host_reserved',2*GIB)),
                ('inventory equation altered', lambda r: r.__setitem__('inventory',r['inventory'].replace('owned_device_bytes: 4565885440','owned_device_bytes: 4565885441'))),
                ('inventory native cap below load', lambda r: r.__setitem__('inventory',r['inventory'].replace('native_device_bytes: '+str(8*GIB),'native_device_bytes: '+str(8*GIB-4)))),
                ('inventory fake constants', lambda r: r.__setitem__('inventory',r['inventory'].replace('value_slots: 24','value_slots: 2'))),
                ('wrong storage', lambda r: r.__setitem__('storage','StreamingRows')),
                ('fenced healthy', lambda r: r.__setitem__('physical_fenced',True)),
                ('healthy report retains provider envelope', lambda r: r.__setitem__('physical_reserved',20*GIB)),
                ('invented failure', lambda r: r.__setitem__('failure','x'))):
                bad = copy.deepcopy(base); mutate(bad)
                expect_fail(lambda bad=bad: semantic.validate_fixture('resident',bad,log,expected_source), name)
            for name, mutate in (
                ('hit uploads one', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('uploads',1)),
                ('hit misses a row', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('reuses',1)),
                ('hit avoids less', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('avoided_h2d',ROW)),
                ('hit row bytes invented', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('ple_row_h2d',ROW)),
                ('hit d2h one byte', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('producer_d2h',1)),
                ('warm d2h nonzero', lambda r: events_of(r,'resident-ple-row-reuse')[0].__setitem__('kv_diagnostic_d2h',4)),
                ('warm relabeled hit', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('observer',False)),
                ('warm rows wrong', lambda r: events_of(r,'resident-ple-row-reuse')[0].__setitem__('actual_rows',1)),
                ('hit boundary incomplete', lambda r: events_of(r,'resident-ple-row-reuse')[1]['comparison'].__setitem__('boundary_events',0)),
                ('hit not fullKV', lambda r: events_of(r,'resident-ple-row-reuse')[1].__setitem__('full_kv_exact',False)),
                ('hit keeps diagnostic packet', lambda r: events_of(r,'resident-ple-row-reuse')[2].__setitem__('comparison',{}))):
                bad = copy.deepcopy(base); mutate(bad)
                expect_fail(lambda bad=bad: semantic.validate_fixture('resident',bad,log,expected_source), name)
            for name, mutate in (
                ('extra report member', lambda r: r.__setitem__('extra',1)),
                ('missing report member', lambda r: r.pop('expected_persistent_context_device')),
                ('unknown event kind', lambda r: r['events'].append({'kind':'invented','root':1})),
                ('extra event member', lambda r: parity(r)[0].__setitem__('extra',1)),
                ('dynamic event wrong shape', lambda r: events_of(r,'actual-completed-refusal')[0].pop('error'))):
                bad = copy.deepcopy(base); mutate(bad)
                expect_fail(lambda bad=bad: semantic.validate_fixture('resident',bad,log,expected_source), name)
            expect_fail(lambda: semantic.validate_fixture('resident',base,log+semantic.MATRIX_MARKER+'\n',expected_source), 'duplicate marker')
            expect_fail(lambda: semantic.validate_fixture('resident',base,semantic.UNKNOWN_MARKER+'\n',expected_source), 'wrong marker')
        check('seven-cell-exact-order/frontier/row/upload-hit/conservation-negatives', matrix_negatives)

        def unknowns():
            marker = semantic.UNKNOWN_MARKER+'\n'
            for mode in ('resident-row-unknown','unknown','resident-unknown'):
                result = unknown_result(mode)
                fresh = mode != 'resident-row-unknown'
                extra = expected_source if fresh else None
                semantic.validate_fixture(mode,result,marker,extra)
                def reject(bad,label):
                    expect_fail(lambda: semantic.validate_fixture(mode,bad,marker,extra), label)
                bad = copy.deepcopy(result); bad['warmup_calls'] = 0 if mode=='resident-row-unknown' else 1
                reject(bad, 'warm phase relabeled')
                bad = copy.deepcopy(result); bad['faulted_call_commits'] = 1
                reject(bad, 'faulted commit')
                bad = copy.deepcopy(result); bad['source_close_refused'] = False
                reject(bad, 'fault without source refusal')
                bad = copy.deepcopy(result); bad['host_reserved'] = 2*GIB
                reject(bad, 'fault report host baseline')
                bad = copy.deepcopy(result); bad['host_fenced'] = True
                reject(bad, 'fault report host fenced')
                bad = copy.deepcopy(result); bad['physical_reserved'] = 16*GIB if result['physical_reserved'] == 20*GIB else 20*GIB
                reject(bad, 'wrong retained provider')
                bad = copy.deepcopy(result); bad['diagnostic_host_reserved'] = 0
                reject(bad, 'diagnostic reservation dropped')
                if mode == 'resident-row-unknown':
                    bad = copy.deepcopy(result); bad['selected_row_h2d'] = ROW; bad['total_value_upload_h2d'] = ROW
                    reject(bad, 'cached hit uploaded')
                    bad = copy.deepcopy(result); bad['cached_row_fault'] = False
                    reject(bad, 'cached row relabeled fresh')
                else:
                    bad = copy.deepcopy(result); bad['cached_row_fault'] = True
                    reject(bad, 'fresh relabeled cached')
                    if mode == 'resident-unknown':
                        bad = copy.deepcopy(result); bad['selected_row_uploads'] = 0; bad['selected_row_h2d'] = 0
                        reject(bad, 'fresh resident row upload lost')
                    bad = copy.deepcopy(result); bad['total_value_upload_h2d'] = 0
                    reject(bad, 'whole-service value traffic invented zero')
                    if mode == 'unknown':
                        bad = copy.deepcopy(result); bad['selected_row_uploads'] = 1; bad['selected_row_h2d'] = ROW
                        reject(bad, 'streamed source row page invented')
                    else:
                        bad = copy.deepcopy(result); bad['selected_row_h2d'] = ROW
                        reject(bad, 'resident row bytes not the value total')
                expect_fail(lambda: semantic.validate_fixture(mode,result,semantic.MATRIX_MARKER+'\n',extra), 'unknown wrong marker')
                expect_fail(lambda: semantic.validate_fixture(mode,result,marker+marker,extra), 'unknown duplicate marker')
        check('cached-row/fresh-streamed/fresh-resident-unknown-negatives', unknowns)

        def finals():
            for mode in ('session-known','session-release-fault','snapshot-known','snapshot-release-fault'):
                result = final_result(mode)
                log = native_log()+marker(mode)+'\n'
                semantic.validate_fixture(mode,result,log)
                def reject(bad,label,log_value=log):
                    expect_fail(lambda: semantic.validate_fixture(mode,bad,log_value), label)
                bad = copy.deepcopy(result); bad['total_completed_full35_calls'] = 1
                reject(bad, 'old two-call count')
                bad = copy.deepcopy(result); bad['actual_completed_full35_calls'] = 1
                reject(bad, 'old observed-only label')
                bad = copy.deepcopy(result); bad['final_hit']['uploads'] = 1
                reject(bad, 'final hit upload')
                bad = copy.deepcopy(result); bad['final_hit']['producer_d2h'] = 1
                reject(bad, 'final hit 1-byte diagnostic')
                bad = copy.deepcopy(result); bad['final_hit']['value_upload_total'] = ROW
                reject(bad, 'final hit value upload invented')
                bad = copy.deepcopy(result); bad['warmup']['avoided_h2d'] = ROW
                reject(bad, 'warm warmup avoided upload invented')
                bad = copy.deepcopy(result); bad['numerical_replay_performed'] = True
                reject(bad, 'numeric replay claim')
                bad = copy.deepcopy(result); bad['final_feedback_on_completed_hit'] = False
                reject(bad, 'feedback not on hit')
                bad = copy.deepcopy(result); bad['source_close_refused'] = not bad['source_close_refused']
                reject(bad, 'wrong source fault')
                bad = copy.deepcopy(result); bad['physical_fenced'] = not bad['physical_fenced']
                reject(bad, 'wrong physical fence')
                bad = copy.deepcopy(result); bad['physical_reserved'] = 4*GIB if bad['physical_reserved'] != 4*GIB else 20*GIB
                reject(bad, 'wrong retained provider phase')
                bad = copy.deepcopy(result); bad['root_transitions']['reset']['root'] = 3
                reject(bad, 'root reset skipped')
                bad = copy.deepcopy(result); bad['root_transitions']['final']['frontier'] = 0
                reject(bad, 'final frontier reset')
                bad = copy.deepcopy(result); bad['root_transitions']['reset']['session_epoch'] = 0
                reject(bad, 'reset epoch not advanced')
                bad = copy.deepcopy(result); bad['original_final_owner']['target_final_reader_hits'] = 0
                reject(bad, 'reader hit dropped')
                bad = copy.deepcopy(result); bad['original_final_owner']['candidate_retired'] = False
                reject(bad, 'candidate not retired')
                bad = copy.deepcopy(result); bad['original_final_owner']['candidate_workspace_target_aliases'] = 1
                reject(bad, 'workspace alias invented')
                bad = copy.deepcopy(result); bad['original_final_owner']['aliases_inside_only_wrapper'] = 2
                reject(bad, 'wrapper alias invented')
                bad = copy.deepcopy(result); bad['original_final_owner']['actual_quarantined_handles_after'] = 0 if mode.endswith('release-fault') else 1
                reject(bad, 'quarantine accounting swapped')
                bad = copy.deepcopy(result); bad['original_final_owner']['actual_native_modules_after'] = 0 if mode.endswith('release-fault') else NATIVE_MODULES
                reject(bad, 'native retirement phase swapped')
                bad = copy.deepcopy(result); bad['inventory'] = bad['inventory'].replace(
                    'native_device_bytes: '+str(8*GIB+NATIVE_MODULES*NATIVE_LOAD),'native_device_bytes: '+str(8*GIB+NATIVE_MODULES*NATIVE_LOAD-1))
                reject(bad, 'native device not actual load sum')
                bad = copy.deepcopy(result); bad['inventory'] = bad['inventory'].replace(
                    'native_host_bytes: '+str(8*GIB+NATIVE_MODULES*NATIVE_HOST+3*MIB),'native_host_bytes: '+str(8*GIB-1))
                reject(bad, 'native host below actual module host')
                bad = copy.deepcopy(result); bad['original_final_owner']['actual_native_PT_LOAD_bytes_before'] = NATIVE_MODULES*NATIVE_LOAD+1
                reject(bad, 'PT_LOAD not the module sum')
                bad = copy.deepcopy(result); bad['prompt_text'] = 'one'
                reject(bad, 'tokenizer text invented')
                bad = copy.deepcopy(result); bad['actual_selected'] = [1,0x7fc00000]
                reject(bad, 'non-finite greedy score')
                expect_fail(lambda: semantic.validate_fixture(mode,result,native_log()), 'native inventory incomplete')
                expect_fail(lambda: semantic.validate_fixture(mode,result,marker(mode)+'\n'+marker(mode)+'\n'), 'duplicate final marker')
                expect_fail(lambda: semantic.validate_fixture(mode,result,semantic.MATRIX_MARKER+'\n'), 'wrong final marker')
                expect_fail(lambda: semantic.validate_fixture(mode,result,native_log(index=2)+marker(mode)+'\n'), 'non-contiguous modules')
                expect_fail(lambda: semantic.validate_fixture(mode,result,native_log(count=NATIVE_MODULES+1)+marker(mode)+'\n'), 'unextended native population')
                expect_fail(lambda: semantic.validate_fixture(mode,result,native_log(inventory=False)+marker(mode)+'\n'), 'missing native inventory')
                expect_fail(lambda: semantic.validate_fixture(mode,result,native_log(host=0)+marker(mode)+'\n'), 'zero module host')
                if mode.endswith('release-fault'):
                    bad = copy.deepcopy(result); bad['host_reserved'] = 2*GIB
                    reject(bad, 'fault host baseline')
                else:
                    bad = copy.deepcopy(result); bad['host_reserved'] = 2*GIB
                    reject(bad, 'known report host baseline')
                    bad = copy.deepcopy(result); bad['host_fenced'] = True
                    reject(bad, 'known host fenced')
        check('final-two-calls/warm-hit/known-fault-reader/native-inventory-negatives', finals)

        def seams():
            source_dir = payload/'source/crates/zerogravity/examples'
            proof = semantic.check_seams(source_dir)
            require(proof['reports'] == 3 and proof['events'] == 8
                    and proof['sources'] == list(semantic.SOURCE_FILES),
                    'three shipped report and eight event contracts')
            shipped = semantic.shipped_seams(source_dir)
            for schema, keys in semantic.REPORT_KEYS.items():
                require(shipped['reports'][schema] == sorted(keys), 'report seam '+schema)
            for kind, keys in semantic.EVENT_KEYS.items():
                require(sorted(shipped['events'][kind]) == sorted(keys), 'event seam '+kind)
            scratch = Path(temp)/'seams'
            shutil.copytree(source_dir, scratch)
            run_path = scratch/'gemma4_hip_matrix/run.rs'
            owner_path = scratch/'gemma4_hip_final_owner.rs'
            run_original, owner_original = run_path.read_text(), owner_path.read_text()
            run_path.write_text(run_original+'\nfn duplicated_seam() -> serde_json::Value {\n    json!({"schema":"zg.gemma4-e2b-original-hip-unknown.v1"})\n}\n')
            expect_fail(lambda: semantic.check_seams(scratch), 'second shipped report literal')
            run_path.write_text(run_original)
            run_path.write_text(run_original+'\nfn widened_seam() -> serde_json::Value {\n    json!({"kind":"pre-cancel","root":0,"invented":1})\n}\n')
            expect_fail(lambda: semantic.check_seams(scratch), 'second shipped event literal')
            run_path.write_text(run_original)
            owner_path.unlink()
            expect_fail(lambda: semantic.check_seams(scratch), 'missing shipped fixture source')
            owner_path.write_text(owner_original)
            semantic.check_seams(scratch)
            good = json.dumps(matrix_result()).encode()
            semantic.strict_report(good)
            for name, payload in (
                ('duplicate report member', b'{"schema":"a","schema":"b"}'),
                ('non-finite constant', b'{"schema":"a","x":NaN}'),
                ('array report', b'[1]'),
                ('trailing text', b'{"schema":"a"}x'),
                ('unbounded report', b'{"schema":"'+(4*MIB)*b'a'+b'"}')):
                expect_fail(lambda payload=payload: semantic.strict_report(payload), name)
            # planned_work.gemma_calls is the mode's actual prepare count, derived
            # from the accepted fixture semantics rather than a fixed 1.
            calls = physical_run.PLANNED_CALLS
            require(set(calls) == set(semantic.MODES) and calls['resident'] is None,
                    'planned calls cover the eight modes with resident source-cell derived')
            row = unknown_result('resident-row-unknown')
            require(calls['resident-row-unknown']
                    == row['warmup_calls'] + row['completed_producers'] == 2,
                    'cached-row warmup plus armed refusal is two actual calls')
            for mode in ('unknown', 'resident-unknown'):
                require(calls[mode] == unknown_result(mode)['completed_producers'] == 1,
                        'fresh armed refusal is one actual call: '+mode)
            for mode in ('session-known', 'session-release-fault',
                         'snapshot-known', 'snapshot-release-fault'):
                final = final_result(mode)
                require(calls[mode] == final['warmup_calls']
                        + final['observed_completed_full35_calls'] == 2,
                        'final warmup plus completed hit is two actual calls: '+mode)
            # Contextual fixture seams: finite actual feedback words, one
            # lowest-index strict-greater argmax joined to the committed packet,
            # and positive pre-commit Ready binding at every committed caller.
            owner_text = (payload/'source/crates/zerogravity/examples/'
                          'gemma4_hip_final_owner.rs').read_text()
            run_text = (payload/'source/crates/zerogravity/examples/'
                        'gemma4_hip_matrix/run.rs').read_text()
            require(owner_text.count('finite actual producer') == 1
                    and owner_text.count('if value > best.1') == 1
                    and owner_text.count('best.1.to_bits()') == 1
                    and owner_text.count('actual fullV winner equals this committed private packet') == 1
                    and owner_text.count('session.binding() == reset_binding') == 1,
                    'actual fullV finite scan/argmax/packet/Ready seams')
            require(run_text.count('Ready has not published root/private feedback') == 5,
                    'positive pre-commit Ready binding at every committed seam')
        check('shipped-fixture-schemas-drive-eight-mode-acceptance', seams)

        def capture_join():
            held = Path(temp)/'held'
            held.mkdir()
            request = {'cases':[{'name':'c'+str(i),'text':'t'+str(i),'rows':2} for i in range(7)]}
            metas = []
            for case in request['cases']:
                cell = {'schema':'zg.gemma4-e2b.ordinary-cell.v1',
                        'request':{'name':case['name'],'text':case['text'],'rows':case['rows']},
                        'prompt_ids':[1,2],
                        'calls':[{'call':0,'base':0,'frontier':2,'selected_id':7,
                                  'eos':True,'input_ids':[1,2]}]}
                raw = json.dumps(cell).encode()
                (held/(case['name']+'.json')).write_bytes(raw)
                metas.append({'metadata':{'name':case['name']+'.json','bytes':len(raw),
                                          'sha256':hashlib.sha256(raw).hexdigest()}})
            capture = {'schema':'zg.gemma4-e2b.ordinary-matrix.v1',
                       'request':request,'cells':metas,
                       'plain_vs_observed_equal':True,'trained_rust_qualified':False,
                       'native_serving_chat_or_performance_qualified':False}
            capture_raw = json.dumps(capture).encode()
            (held/'capture.json').write_bytes(capture_raw)
            invocation = {'status':'EXECUTED_ORIGINAL_GEMMA4_MATRIX_EXPECTED_ONLY',
                          'exit_code':0,'source_before':{'bytes':1,'sha256':'1'*64},
                          'source_after':{'bytes':1,'sha256':'1'*64},
                          'source_FDs_closed':True,'cleanup_errors':[],
                          'source_FD_cleanup_errors':[],
                          'foreground_cleanup':{'leader_reaped':True,'group_absent':True},
                          'capture':{'bytes':len(capture_raw),
                                     'sha256':hashlib.sha256(capture_raw).hexdigest()}}
            (held/'invocation.json').write_text(json.dumps(invocation))
            joined = semantic.expected_cells(held/'invocation.json')
            require(len(joined['cells']) == 7 and joined['cells'][1]['name'] == 'c1'
                    and joined['cells'][0]['calls'][0]['frontier'] == 2,
                    'seven authentic cells joined at case order')
            def reject_call(label):
                expect_fail(lambda: semantic.expected_cells(held/'invocation.json'), label)
            meta = copy.deepcopy(capture); meta['cells'][0]['metadata']['name'] = 'c1.json'
            meta['cells'][1]['metadata']['name'] = 'c0.json'
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('metadata order swapped')
            meta = copy.deepcopy(capture); meta['cells'][0]['metadata']['bytes'] += 1
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('cell byte pin')
            meta = copy.deepcopy(capture); meta['cells'][0]['metadata']['sha256'] = '0'*64
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('cell sha pin')
            meta = copy.deepcopy(capture); meta['cells'][0]['metadata']['extra'] = 1
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('metadata member set')
            meta = copy.deepcopy(capture); meta['cells'] = meta['cells'][:-1]
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('six-cell population')
            meta = copy.deepcopy(capture); meta['plain_vs_observed_equal'] = False
            (held/'capture.json').write_bytes(json.dumps(meta).encode())
            reject_call('plain/observed mismatch')
            (held/'capture.json').write_bytes(capture_raw)
            cell_path = held/'c0.json'
            cell_raw = cell_path.read_bytes()
            def rewrite_cell(text):
                raw = text.encode()
                cell_path.write_bytes(raw)
                patched = copy.deepcopy(capture)
                patched['cells'][0]['metadata']['bytes'] = len(raw)
                patched['cells'][0]['metadata']['sha256'] = hashlib.sha256(raw).hexdigest()
                return json.dumps(patched).encode()
            for name, mutate in (
                ('cell schema', lambda c: c.__setitem__('schema','other')),
                ('cell request name', lambda c: c['request'].__setitem__('name','zz')),
                ('cell rows', lambda c: c.__setitem__('prompt_ids',[1])),
                ('call base', lambda c: c['calls'][0].__setitem__('base',1)),
                ('call frontier jump', lambda c: c['calls'][0].__setitem__('frontier',5)),
                ('eos in the middle', lambda c: c['calls'].append({'call':1,'base':2,'frontier':3,
                                                                  'selected_id':7,'eos':True,
                                                                  'input_ids':[3]})),
                ('nine calls', lambda c: c['calls'].extend([{'call':n,'base':0,'frontier':1,
                                                             'selected_id':7,'eos':n==8,
                                                             'input_ids':[1]} for n in range(1,9)]))):
                cell = json.loads(cell_raw); mutate(cell)
                (held/'capture.json').write_bytes(rewrite_cell(json.dumps(cell)))
                reject_call(name)
            (held/'capture.json').write_bytes(capture_raw)
            cell_path.write_bytes(cell_raw)
            invocation['capture']['sha256'] = '0'*64
            (held/'invocation.json').write_text(json.dumps(invocation))
            reject_call('invocation capture pin')
            invocation['capture'] = {'bytes':len(capture_raw),
                                     'sha256':hashlib.sha256(capture_raw).hexdigest()}
            invocation['source_FDs_closed'] = False
            (held/'invocation.json').write_text(json.dumps(invocation))
            reject_call('source FD closure')
            invocation['source_FDs_closed'] = True
            invocation['foreground_cleanup']['group_absent'] = False
            (held/'invocation.json').write_text(json.dumps(invocation))
            reject_call('foreground group absent')
            invocation['foreground_cleanup']['group_absent'] = True
            (held/'invocation.json').write_text(json.dumps(invocation))
            semantic.expected_cells(held/'invocation.json')
        check('held-capture-metadata-name/bytes/sha-joined-to-case-call-order', capture_join)

        def bindings():
            remote_base = Path('/home/pvl/.cache/abird/zerogravity/qualification/'+launch.BASE)
            closure_pin = {'bytes':10,'sha256':'c'*64}
            value = binding_value(closure_pin,remote_base)
            launch.validate_binding(value,closure_pin,remote_base)
            proof = value['artifacts']['gemma4_hip_matrix']
            mode = 'resident'
            remote = {'mode':mode,'status':'CLOSED_PASS_SCOPED','control':{'bytes':1,'sha256':'d'*64},
                      'control_after':{'bytes':1,'sha256':'d'*64},
                      'source_closure':closure_pin,
                      'hydrated_closure':{'bytes':closure_pin['bytes']+1,'sha256':'e'*64},
                      'source_after':'UNCHANGED','observed_exit':0,'errors':[],'pid':77,
                      'foreground_cleanup':{'pgid':77,'term_sent':True,'kill_sent_before_reap':True,
                                            'exit_code':0,'leader_reaped':True,'group_absent':True,
                                            'kill_attempts_before_reap':2,'leader_reap_attempts':1,
                                            'all_nonzero_signals_before_reap':True,'stage_errors':[]},
                      'log':{'retained':10,'total':10,'eof':True,'closed':True},
                      'artifact_name':'gemma4_hip_matrix','binary':proof,
                      'build_binding':{'build_receipt':value['build_receipt']},
                      'held_data_files':70,'data_after':'UNCHANGED_HELD_FULL_SHA',
                      'semantic':{'mode':mode,'report':{'bytes':3,'sha256':'a'*64}}}
            binding = {'artifacts':value['artifacts'],'build_receipt':value['build_receipt']}
            path = temp/'remote-model.log'
            path.write_text(json.dumps(remote)+'\n')
            launch.remote_receipt(mode,path,closure_pin,remote['control'],binding)
            for name, mutate in (
                ('packaged==hydrated', lambda r: r.__setitem__('hydrated_closure',closure_pin)),
                ('selected artifact swapped', lambda r: r.__setitem__('artifact_name','gemma4_hip_final_owner')),
                ('wrong binary pin', lambda r: r.__setitem__('binary',{'bytes':1,'sha256':'0'*64})),
                ('69 held FDs', lambda r: r.__setitem__('held_data_files',69)),
                ('wrong semantic mode', lambda r: r['semantic'].__setitem__('mode','session-known')),
                ('receipt mismatch', lambda r: r['build_binding'].__setitem__('build_receipt',{'bytes':1,'sha256':'0'*64}))):
                bad = copy.deepcopy(remote); mutate(bad)
                (temp/'remote-bad.log').write_text(json.dumps(bad)+'\n')
                expect_fail(lambda bad=bad: launch.remote_receipt(mode,temp/'remote-bad.log',closure_pin,remote['control'],binding), name)
            for name, mutate in (
                ('missing artifact', lambda v: v['artifacts'].pop('gemma4_hip_final_owner')),
                ('swapped mode map', lambda v: v['mode_artifacts'].__setitem__('resident','gemma4_hip_final_owner')),
                ('local mutable metadata', lambda v: v.__setitem__('note','local')),
                ('wrong source base', lambda v: v.__setitem__('source_base','/tmp/other')),
                ('zero artifact pin', lambda v: v['artifacts']['gemma4_hip_matrix'].__setitem__('bytes',0)),
                ('hydrated string', lambda v: v.__setitem__('hydrated_closure','local'))):
                bad = copy.deepcopy(value); mutate(bad)
                expect_fail(lambda bad=bad: launch.validate_binding(bad,closure_pin,remote_base), name)
            build_remote = {'mode':'build','status':'CLOSED_PASS_SCOPED','control':remote['control'],
                            'control_after':remote['control'],'source_closure':closure_pin,
                            'hydrated_closure':remote['hydrated_closure'],'source_after':'UNCHANGED',
                            'observed_exit':0,'errors':[],'pid':77,
                            'foreground_cleanup':remote['foreground_cleanup'],
                            'log':remote['log'],'build_artifact_count':1,
                            'build_artifacts':{'gemma4_hip_matrix':{'path':'/tmp/x','proof':proof}},
                            'build_receipt':value['build_receipt']}
            (temp/'remote-build.log').write_text(json.dumps(build_remote)+'\n')
            expect_fail(lambda: launch.remote_receipt('build',temp/'remote-build.log',closure_pin,
                                                      remote['control'],None), 'one-artifact build')
        check('two-artifact-mode-map/pins/packaged-hydrated-mismatch-negatives', bindings)

        def packaged_hydrated():
            closure = {'source_count':5,'source_manifest':{'bytes':1,'sha256':'1'*64},
                       'parent_frozen_source_manifest':{'bytes':1,'sha256':'2'*64},
                       'files':100,'bytes':1000,'scope':'scope'}
            closure_pin = {'bytes':10,'sha256':'c'*64}
            child = {'files':101,'bytes':1001,'source_count':5,
                     'source_manifest':closure['source_manifest'],
                     'parent_frozen_source_manifest':closure['parent_frozen_source_manifest'],
                     'scope':'scope','reused_cache_closure_sha256':CACHE_SHA}
            remote = owner_receipt(4242,{'pgid':4242,'term_sent':True,'kill_sent_before_reap':True,
                                         'exit_code':0,'leader_reaped':True,'group_absent':True,
                                         'kill_attempts_before_reap':2,'leader_reap_attempts':1,
                                         'all_nonzero_signals_before_reap':True,'stage_errors':[]})
            remote.update({'status':'CLOSED_PASS_SCOPED','control':'control','control_after':'control',
                           'source_archive':{'bytes':1,'sha256':'0'*64},
                           'source_archive_after':'UNCHANGED_HELD_FULL_SHA',
                           'source_closure':closure_pin,
                           'hydrated_closure':{'bytes':11,'sha256':'d'*64},
                           'gpu_or_model_executed':False,'hydrate':child})
            raw = json.dumps(remote).encode()
            receipt_mod.parse_closed_bootstrap(raw,'control',remote['source_archive'],closure,
                                               closure_pin,CACHE_SHA)
            bad = copy.deepcopy(remote); bad['hydrated_closure'] = closure_pin
            expect_fail(lambda: receipt_mod.parse_closed_bootstrap(json.dumps(bad).encode(),'control',
                        bad['source_archive'],closure,closure_pin,CACHE_SHA), 'hydrated==packaged')
            bad = copy.deepcopy(remote); bad['hydrate']['files'] = 99
            expect_fail(lambda: receipt_mod.parse_closed_bootstrap(json.dumps(bad).encode(),'control',
                        bad['source_archive'],closure,closure_pin,CACHE_SHA), 'hydrated member count')
        check('offline-hydration-must-differ-from-packaged-closure', packaged_hydrated)

        def source_closure():
            scratch = temp/'extract'
            scratch.mkdir()
            umask = os.umask(0o022)
            try:
                with archive_path.open('rb') as raw:
                    with tarfile.open(fileobj=raw, mode='r:') as tar:
                        seen = bootstrap.extract_archive(tar,scratch,
                                                         ('pins.py','bootstrap.py','source-payload.tar'))
                with (scratch/'.plain-probe').open('xb'):
                    pass
            finally:
                os.umask(umask)
            require(seen == set(closure['records']) | {'source-closure.json'},
                    'archive member union exact')
            require(stat.S_IMODE((scratch/'.plain-probe').stat().st_mode) == 0o644,
                    'plain xb loses authenticated modes under umask022')
            source_modes.check_source(scratch)
            for name in ('README.md','install.sh'):
                if name in manifest['files']:
                    require(stat.S_IMODE((scratch/'source'/name).stat().st_mode)
                            == manifest['files'][name]['mode'],
                            'real extracted manifest mode: '+name)
            fixture = scratch/'source/crates/zerogravity/examples/gemma4_hip_matrix/run.rs'
            original = fixture.read_bytes()
            fixture.write_bytes(original+b'X')
            expect_fail(lambda: source_modes.check_source(scratch), 'changed fixture bytes')
            fixture.write_bytes(original)
            mode_rel = ('README.md' if 'README.md' in manifest['files']
                        else sorted(manifest['files'])[0])
            mode = stat.S_IMODE((scratch/'source'/mode_rel).stat().st_mode)
            os.chmod(scratch/'source'/mode_rel,0o600)
            expect_fail(lambda: source_modes.check_source(scratch), 'wrong source mode')
            os.chmod(scratch/'source'/mode_rel,mode)
            (scratch/'source/UNEXPECTED').write_text('x')
            expect_fail(lambda: source_modes.check_source(scratch), 'extra source file')
            (scratch/'source/UNEXPECTED').unlink()
            (scratch/'source/UNEXPECTED-LINK').symlink_to('/etc/passwd')
            expect_fail(lambda: source_modes.check_source(scratch), 'source symlink')
            (scratch/'source/UNEXPECTED-LINK').unlink()
            def remove_and_reject(rel, label):
                path = scratch/'source'/rel
                raw = path.read_bytes()
                kept = stat.S_IMODE(path.stat().st_mode)
                path.unlink()
                expect_fail(lambda: source_modes.check_source(scratch), label)
                path.write_bytes(raw)
                os.chmod(path,kept)
            remove_and_reject('Cargo.lock', 'missing required config input')
            other = next((name for name in sorted(manifest['files'])
                          if name.startswith('crates/zerogravity-model-qwen35/')), None)
            require(other is not None, 'qwen27b unit present in complete source closure')
            remove_and_reject(other, 'missing other integration unit member')
            source_modes.check_source(scratch)
        check('real-archive-extract-complete-source-modes/missing-extra-and-negatives', source_closure)

        def cleanup_owner():
            child = subprocess.Popen(['sh','-c','trap "exit 0" TERM; while :; do sleep 0.05; done'],
                                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                                     start_new_session=True)
            time.sleep(0.3)
            receipt = {'pid':child.pid}
            owner.cleanup_group(child,receipt)
            state = receipt['foreground_cleanup']
            require(state == {'pgid':receipt['pid'],'term_sent':True,'kill_sent_before_reap':True,
                              'leader_reaped':True,'group_absent':True,
                              'kill_attempts_before_reap':2,'leader_reap_attempts':1,
                              'all_nonzero_signals_before_reap':True,'exit_code':0,
                              'stage_errors':[]}, 'real TERM/twoKILL/oneReap/group_absent')
            require(not owner.group_exists(receipt['pid']), 'real group absent')
            receipt_mod._owner(owner_receipt(receipt['pid'],state))
            for name, mutate in (
                ('one kill', lambda c: c.__setitem__('kill_attempts_before_reap',1)),
                ('reap before kill', lambda c: c.__setitem__('kill_sent_before_reap',False)),
                ('leader not reaped', lambda c: c.__setitem__('leader_reaped',False)),
                ('group remains', lambda c: c.__setitem__('group_absent',False)),
                ('kill exit', lambda c: c.__setitem__('exit_code',-15))):
                bad = dict(state); mutate(bad)
                expect_fail(lambda bad=bad: receipt_mod._owner(owner_receipt(receipt['pid'],bad)), name)
            bad = owner_receipt(receipt['pid'],state); bad['log']['eof'] = False
            expect_fail(lambda: receipt_mod._owner(bad), 'incomplete EOF')
        check('real-original-TERM-twoKILL-before-oneReap-group-absent/EOF', cleanup_owner)

        def log_eof():
            script = ('import json,sys;'
                      'sys.stdout.write('+repr(receipt_mod.BANNER.decode())+'+json.dumps({"a":1})+"\\n");'
                      'sys.stdout.flush()')
            child = subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            log = owner.Log(child.stdout,temp/'eof.log')
            code = log.observe(child,30)
            owner.cleanup_group(child,{'pid':child.pid})
            log.finish()
            require(code == 0 and log.retained == log.total > 0 and log.eof and log.closed,
                    'real bounded full-EOF drain')
            require((temp/'eof.log').read_bytes().startswith(receipt_mod.BANNER),
                    'child banner bytes')
        check('real-child-log-full-EOF', log_eof)

        def fd70_guard():
            text = (payload/'physical-runner/joint_data.py').read_text()
            runtext = (payload/'physical-runner/run.py').read_text()
            require('len(self.held)==70' in text
                    and 'len({v.before[:2] for v in self.held})==70' in text,
                    'original 70 distinct source/reference FD invariant present')
            require('len(held_data) == 70' in runtext
                    and 'len({v.before[:2] for v in held_data}) == 70' in runtext,
                    'no standalone 35-FD regression in physical run')
            require(OWNER_SHA == '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
                    'original owner hash')
            require(text.count("HELD_BINARY = '/tmp/zg-hip-joint-held-binary'") == 1
                    and "['--proc','/proc','--tmpfs','/tmp','--ro-bind-fd',str(binary.file.fileno()),HELD_BINARY]" in text
                    and '/tmp/zg-gemma-ple-held-binary' not in text + runtext,
                    'one held-binary mount operand used by the loader exactly once')
            require(runtext.count('executable = HELD_BINARY') == 1
                    and "'/tmp/zg-hip-joint-held-binary'" in text,
                    'caller executes the single mounted held binary path')
        check('original-70-distinct-FD-invariant-not-35', fd70_guard)

        def banner():
            source = (payload/'source/scripts/native-gpu.py').read_text()
            require("LLVM verification not requested" in source
                    and "mandatory user/network isolation " in source,
                    'pinned native-gpu banner fragments')
            require(receipt_mod.BANNER.endswith(b'via unshare (setup failure aborts the command)\n'),
                    'carried banner literal')
        check('pinned-native-gpu-banner-literal', banner)
    except BaseException as error:
        results['errors'].append('setup: '+repr(error))
    finally:
        results['status'] = ('PASS_GPU_FREE_GROUPED_CONTROL' if not results['errors']
                              else 'FAIL_GPU_FREE_GROUPED_CONTROL')
        results['checks_count'] = len(results['checks'])
        (args.output/'CONTROL-RESULT.json').write_text(json.dumps(results, indent=2)+'\n')
        shutil.rmtree(temp, ignore_errors=True)
    print('CONTROL-CHECKS '+json.dumps({'status':results['status'],
                                        'checks':results['checks'],
                                        'errors':results['errors']}, separators=(',',':')))
    return 0 if results['status'] == 'PASS_GPU_FREE_GROUPED_CONTROL' else 1


if __name__ == '__main__':
    sys.exit(main())
