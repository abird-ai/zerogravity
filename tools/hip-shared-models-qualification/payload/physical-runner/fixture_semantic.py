"""Finite actual same-actor source/state/native lifecycle acceptance; no timing grant."""
import re
from pins import require
GIB=1024**3
PENDING_MODES=('source-page-query-unknown','source-page-wait-unknown','source-page-destroy-unknown')
MODES=('known-qwen-reload','qwen-final-reader-unknown','gemma-feedback-unknown')+PENDING_MODES
def integer(v,k,n=None,lo=0,hi=2**63-1):
    x=v[k]; require(type(x) is int and lo<=x<=hi and (n is None or x==n),'actual integer '+k); return x

def sets(n,expected,compiler):
    integer(n,'compiler_host_reserved',compiler*GIB)
    require(n['default_stream_refused'] is True and type(n['sets']) is list
            and sorted(s['non_null_modules'] for s in n['sets'])==sorted(expected),'actual selected native sets')
    for s in n['sets']:
        integer(s,'set_key',lo=1); integer(s,'logical_stream',s['set_key'])
        count=integer(s,'non_null_modules',lo=1)
        integer(s,'catalogue_keys',count); integer(s,'admitted_modules',count)
        require(s['sealed'] is True and s['sealed_miss_refused'] is True,'sealed actual native catalogue')
        integer(s,'scoped_metadata_bytes',lo=1,hi=2*1024**2);integer(s,'PT_LOAD_bytes',lo=1,hi=GIB); integer(s,'retained_code_host_bytes',lo=1,hi=2*GIB)

def source_ownership_joins(r,log,first,identities,generation_count,receipt=None,target_g=False,final_native=None):
    """Join actual per-generation ELF, first/final sets, inventories and cold bounds."""
    records=re.findall(r'^HIP_SCOPED_NATIVE_CODE source=(.+) module=(\d+) bytes=(\d+) load=(\d+) retained_host=(\d+) sha256=([0-9a-f]{64})$',log,re.M)
    generations=[]
    for source,module,size,load,host,sha in records:
        if int(module)==1: generations.append({'source':source,'rows':[]})
        require(generations and generations[-1]['source']==source,'chronological source-bound native generation')
        generations[-1]['rows'].append((int(module),int(size),int(load),int(host),sha))
    require(len(generations)==generation_count,'exact real compiler preparations; fourth never reaches RTC')
    for index,(gen,count,identity) in enumerate(zip(generations,[117,184,117][:generation_count],identities)):
        rows=gen['rows']; require(gen['source']==identity['source'] and [x[0] for x in rows]==list(range(1,count+1)),
                                 'all per-source native module indices')
        totals=[sum(x[j] for x in rows) for j in (1,2,3)]
        require(all(0<x<=limit for x,limit in zip(totals,[GIB,GIB,2*GIB])),'original actual ELF/catalogue host bounds')
        gen.update(modules=count,ELF_bytes=totals[0],PT_LOAD_bytes=totals[1],retained_host_bytes=totals[2])
        del gen['rows']
        if index<2:
            actual=next(s for s in first['native']['sets'] if s['set_key']==identity['set_key'])
            require(actual['PT_LOAD_bytes']==totals[1] and actual['retained_code_host_bytes']==totals[2],
                    'actual ELF records join same original source-set admission')
            if final_native is not None:
                retained=next(s for s in final_native['sets'] if s['set_key']==identity['set_key'])
                require(retained['PT_LOAD_bytes']==totals[1] and retained['retained_code_host_bytes']==totals[2],
                        'actual ELF records join first/final original source-set admission')
        if receipt is not None and ((index==1 and target_g) or (index==0 and not target_g)):
            integer(receipt,'native_code_bytes_before',totals[1]);integer(receipt,'native_host_bytes_before',totals[2]);integer(receipt,'native_device_bytes_before',8*GIB+totals[1])
    require(sum(line.startswith('GEMMA_NATIVE_CODE module=') for line in log.splitlines())==len(records),
            'legacy and new source-bound code records agree; no hidden native catalogue')
    inventories={}
    for role,modules,quota in [('qwen',117,12*GIB),('gemma',184,16*GIB)]:
        v={k:int(n) for k,n in re.findall(r'(\w+): (\d+)',first[role+'_actual'])}
        gen=generations[0 if role=='qwen' else 1]
        integer(v,'native_device_bytes',8*GIB+gen['PT_LOAD_bytes'])
        observed=next(row for row in first['native']['sets'] if row['set_key']==first[role]['set_key'])
        integer(v,'native_host_bytes',8*GIB+gen['retained_host_bytes']+observed['scoped_metadata_bytes']+2*1024**2)
        integer(v,'host_upper_bytes',lo=1,hi=32*GIB)
        integer(v,'complete_device_bytes' if role=='qwen' else 'device_with_six_roots_bytes',lo=1,hi=quota)
        if role=='qwen':
            integer(v,'admitted_host_domain_bytes',32*GIB);integer(v,'six_root_payload_bytes',954728448)
        else:
            for k,n in [('resident_matrices',277),('resident_matrix_bytes',4558946304),('resident_upload_windows',4516),
                        ('original_greedy_workspace_bytes',12336),('six_root_device_bytes',14155776),
                        ('persistent_context_device_bytes',4*GIB),('persistent_context_host_bytes',2*GIB)]:integer(v,k,n)
            integer(v,'device_with_six_roots_bytes',v['owned_device_bytes']+v['six_root_device_bytes']+v['native_device_bytes']+12336+6*256)
            integer(v,'complete_physical_device_bytes',v['device_with_six_roots_bytes']+4*GIB)
        inventories[role]=v
    integer(r['cold'],'provider_upper',lo=1,hi=32*GIB)
    integer(r['cold'],'host_upper',lo=1,hi=32*GIB-128*1024**2-512*1024**2)
    return generations,inventories

def validate_fixture(mode,r,log):
    require(mode in MODES and r['mode']==mode and r['schema']=='zg.hip-two-source-original-lifecycle.v1'
            and r['status']=='PASS_FINITE_SHARED_SOURCE_SCOPE' and r['failure'] is None
            and r['trained_serving_speed_or_general_fit_qualified'] is False,'finite actual original sources only')
    if mode in PENDING_MODES:
        events=r.get('events')
        require(type(events) is list and events and isinstance(events[0],dict)
                and events[0].get('kind')=='both-original-sources-live',
                'genuine first lifecycle event for pending source-page probe')
        first=events[0]
        pending=validate_pending_mode(mode,r,first)
        q=first['qwen']; g=first['gemma']
        require(r['physical_fenced'] is True and r['qwen_source_close_refused'] is True
                and r['gemma_source_close_refused'] is True and type(r['host_fenced']) is bool,
                'same global physical/source fence')
        integer(r,'physical_reserved',32*GIB)
        integer(r,'persistent_compiler_host',16*GIB)
        integer(r,'diagnostic_host',128*1024**2)
        integer(r,'host_reserved',None,lo=2*GIB,hi=32*GIB)
        first_native=first['native']; final_native=r.get('final_native')
        require(isinstance(final_native,dict),'actual retained final native observation')
        sets(first_native,[117,184],16); integer(first_native,'original_key_overlap',lo=1)
        sets(final_native,[117,184],16)
        require(all(s['usable'] is False for s in final_native['sets']),'global unknown rejects BOTH source views')
        first_rows=first_native['sets']; final_rows=final_native['sets']
        keys=sorted((q['set_key'],g['set_key']))
        require(sorted(s['set_key'] for s in first_rows)==keys
                and sorted(s['set_key'] for s in final_rows)==keys,
                'first/final native rows retain both original scoped set keys')
        for identity in (q,g):
            actual=next(s for s in first_rows if s['set_key']==identity['set_key'])
            retained=next(s for s in final_rows if s['set_key']==identity['set_key'])
            count=117 if identity['set_key']==q['set_key'] else 184
            require(actual['logical_stream']==identity['logical_stream']==retained['logical_stream']
                    and actual['non_null_modules']==count==retained['non_null_modules']
                    and actual['catalogue_keys']==count==retained['catalogue_keys']
                    and actual['admitted_modules']==count==retained['admitted_modules']
                    and actual['scoped_metadata_bytes']==retained['scoped_metadata_bytes']
                    and actual['usable'] is True and retained['usable'] is False,
                    'first/final native rows join same original source set/stream')
        probe=pending['source_page_probe']; outcome=probe['outcome']
        require(outcome['qwen_close_refused'] is r['qwen_source_close_refused']
                and outcome['gemma_close_refused'] is r['gemma_source_close_refused'],
                'probe and top-level original source close refusal agree')
        generations,inventories=source_ownership_joins(r,log,first,[q,g],2,final_native=final_native)
        return {'status':r['status'],'mode':mode,'global_unknown':True,
                'pending_event_probe':True,'source_page_probe':probe,
                'native_generations':generations,'cold':r['cold'],
                'actual_family_inventories':inventories,
                'physical_reserved':r['physical_reserved'],'host_reserved':r['host_reserved'],
                'math_state_owner_scope_only':True,'serving_speed_or_general_fit':False}
    fault=mode!='known-qwen-reload'; qfault=mode=='qwen-final-reader-unknown'
    for key,count in [('actual_completed_full64_calls',1 if fault else 3),
                      ('actual_qwen_commits',0 if qfault else 1 if fault else 3),
                      ('actual_completed_full35_calls',1 if fault else 4),
                      ('actual_gemma_commits',1 if fault else 4)]: integer(r,key,count)
    require(r['physical_fenced'] is fault and r['qwen_source_close_refused'] is fault
            and r['gemma_source_close_refused'] is fault and type(r['host_fenced']) is bool,'same global physical/source fence')
    integer(r,'physical_reserved',(32 if fault else 4)*GIB)
    integer(r,'persistent_compiler_host',(16 if fault else 24)*GIB)
    integer(r,'diagnostic_host',128*1024**2)
    integer(r,'host_reserved',None if fault else 26*GIB+64*1024+128*1024**2,lo=2*GIB,hi=32*GIB)
    if not fault: require(r['host_fenced'] is False,'known host unfenced')
    events=r['events']; require(type(events) is list and events,'complete ordered actual lifecycle events')
    kinds=[v['kind'] for v in events]
    expected=['both-original-sources-live']
    if qfault: expected+=['gemma-completed-committed','actual-qwen-final-reader-fault']
    elif fault: expected+=['qwen-completed-committed','gemma-completed-committed','actual-gemma-final-feedback-fault']
    else: expected+=['qwen-completed-committed','gemma-completed-committed','qwen-completed-committed','known-qwen-retired','gemma-completed-committed','actual-qwen-reloaded','qwen-completed-committed','gemma-completed-committed','fourth-preparation-refused-preeffect','gemma-completed-committed']
    require(kinds==expected,'entire genuine original-call/lifecycle order')
    first=events[0]; q=first['qwen']; g=first['gemma']
    require(q['actor']==g['actor'] and q['source']!=g['source']
            and q['source_service_generation']!=g['source_service_generation']
            and q['set_key']!=g['set_key'] and q['logical_stream']==q['set_key']
            and g['logical_stream']==g['set_key'],'one actor with distinct source/service/full scoped identity')
    sets(first['native'],[117,184],16)
    integer(first['native'],'original_key_overlap',lo=1)
    for e in events:
        if e['kind'].endswith('completed-committed'):
            c=e['comparison']; normal=c.get('normal_observer_none',False)
            require((c.get('exact') is True and c['mismatches' if e['kind'].startswith('qwen') else 'mismatched_words']==0) if not normal else
                    c['packet_exact'] is True and c['ordinary_chunk_bytes']==16,'every actual producer comparison')
            require(e['before']['root']+1==e['after']['root']
                    and e['before']['session']==e['after']['session'],'independent actual StateRoot publication')
    reader=next(e for e in events if 'receipt' in e)
    p=reader['receipt']; target_g=mode=='gemma-feedback-unknown'; modules=184 if target_g else 117
    require(p['wrapper']==('Session' if target_g else 'Pending' if qfault else 'Snapshot')
            and p['fault'] is fault and p['candidate_retired'] is True
            and p['native_admission_retained_after'] is fault
            and p['original_service_retained_after'] is fault,'true last reader and original native/source owner')
    for key,n in [('candidate_owners_before_drop',1),('model_owners_before_drop',2 if target_g else 1),
                  ('candidate_workspace_target_aliases',0),('release_hits',1),('native_retirement_attempts',1),
                  ('reader_hits_before_native_retirement',1),('native_modules_before',modules),
                  ('native_modules_after',modules if fault else 0)]: integer(p,key,n)
    integer(p,'lease_bytes',12 if target_g else None,lo=1)
    integer(p,'aliases_inside_wrapper',1 if target_g else None,lo=1,hi=512)
    integer(p,'quarantined_handles_after',None if fault else 0,lo=1 if fault else 0)
    if qfault: require(reader['comparison']['exact'] is True and reader['comparison']['mismatches']==0,'uncommitted full64 Ready exact')
    if fault:
        sets(reader['native'],[117,184],16); sets(r['final_native'],[117,184],16)
        require(all(s['usable'] is False for s in r['final_native']['sets']),'global unknown rejects BOTH source views')
    else:
        sets(reader['native'],[184],16); integer(reader['native'],'available_model_capacity',12*GIB)
        reloaded=next(e for e in events if e['kind']=='actual-qwen-reloaded')
        ri=reloaded['identity']; require(ri['source']==q['source'] and ri['actor']==q['actor']
              and ri['set_key']>max(q['set_key'],g['set_key']) and ri['logical_stream']==ri['set_key']
              and ri['source_service_generation']!=q['source_service_generation'],'actual same-source cold reload')
        sets(reloaded['native'],[117,184],24)
        refused=next(e for e in events if e['kind']=='fourth-preparation-refused-preeffect')
        sets(refused['native'],[184],24)
        require(refused['native']['sets'][0]['usable'] is True and r['final_native']['sets']==[],
                'known ordinary compiler refusal leaves sibling usable and final native empty')
    # Actual loaded ELF records and first-set/inventory joins, shared with pending modes.
    generations,inventories=source_ownership_joins(r,log,first,[q,g,q],2 if fault else 3,
                                                  receipt=p,target_g=target_g)
    return {'status':r['status'],'mode':mode,'global_unknown':fault,'native_generations':generations,
            'cold':r['cold'],'actual_family_inventories':inventories,'physical_reserved':r['physical_reserved'],'host_reserved':r['host_reserved'],
            'math_state_owner_scope_only':True,'serving_speed_or_general_fit':False}


def _probe_object(value, key):
    item=value.get(key); require(isinstance(item,dict),'actual probe object '+key); return item


def _probe_int(value, key, n=None, lo=0, hi=2**64-1):
    item=value.get(key); require(type(item) is int and lo<=item<=hi and (n is None or item==n),
                                'actual probe u64 '+key); return item


def _probe_bool(value, key, expected=None):
    item=value.get(key); require(type(item) is bool and (expected is None or item is expected),
                                'actual probe bool '+key); return item


def _probe_i32(value, key):
    item=value.get(key); require(item is None or (type(item) is int and -2**31<=item<=2**31-1),
                                'actual nullable i32 '+key); return item


def _ledger_domain(value, label):
    require(isinstance(value,dict) and set(value)=={'host','device'},
            'actual original ledger nested domain '+label)
    domains={}
    for name in ('host','device'):
        domain=value[name]
        require(isinstance(domain,dict) and set(domain)=={'limit','reserved','fenced'},
                'actual typed ledger domain '+label+'/'+name)
        limit=_probe_int(domain,'limit')
        reserved=_probe_int(domain,'reserved')
        fenced=_probe_bool(domain,'fenced')
        require(reserved<=limit,
                'actual original ledger reserved within limit '+label+'/'+name)
        domains[name]={'limit':limit,'reserved':reserved,'fenced':fenced}
    return domains


def validate_pending_mode(mode,r,first):
    """Validate the exact source-page probe contract; no invented driver fault claim."""
    _probe_int(r,'actual_completed_full64_calls',0)
    _probe_int(r,'actual_qwen_commits',0)
    _probe_int(r,'actual_completed_full35_calls',0)
    _probe_int(r,'actual_gemma_commits',0)
    probe=_probe_object(r,'source_page_probe')
    require(probe.get('schema')=='zg.hip-source-page-event-probe.v1'
            and probe.get('mode')==mode,'actual selected source-page probe')
    identity=_probe_object(probe,'identity')
    for key in ('source','service','actor'):
        require(isinstance(identity.get(key),str) and identity[key],'probe identity '+key)
    for key in ('generation','set_key','logical_stream','page_allocation_id'):
        _probe_int(identity,key)
    q=first.get('qwen'); g=first.get('gemma')
    require(isinstance(q,dict) and isinstance(g,dict),
            'genuine first-event Qwen/Gemma identities')
    for key in ('source','actor'):
        require(isinstance(q.get(key),str) and q[key],'actual first-event Qwen '+key)
        require(isinstance(g.get(key),str) and g[key],'actual first-event Gemma '+key)
    for key in ('source_service_generation','set_key','logical_stream'):
        _probe_int(q,key); _probe_int(g,key)
    require(q['actor']==g['actor'] and q['source']!=g['source']
            and q['source_service_generation']!=g['source_service_generation']
            and q['set_key']!=g['set_key']
            and q['logical_stream']==q['set_key']
            and g['logical_stream']==g['set_key'],
            'one actor with distinct source/service/full scoped identity')
    require(identity['source']==q['source']
            and identity['actor']==q['actor']
            and identity['generation']==q['source_service_generation']
            and identity['set_key']==q['set_key']
            and identity['logical_stream']==q['logical_stream'],
            'probe identity joins genuine first-event Qwen source/generation/actor/set/stream')
    if 'service' in q:
        require(identity['service']==q['service'],
                'probe identity service joins genuine first-event Qwen service')
    else:
        # The first-event qwen identity exposes its original actor/service string
        # as `actor`; no separate `service` field exists in that contract row.
        require(identity['service']==q['actor'],
                'probe identity service joins genuine first-event Qwen actor/service')
    sets(first['native'],[117,184],16)
    integer(first['native'],'original_key_overlap',lo=1)
    event=_probe_object(probe,'event')
    _probe_int(event,'identity')
    first_serial=_probe_int(event,'first_serial')
    current_serial=_probe_int(event,'current_serial')
    require(current_serial>=first_serial,'actual event serial lineage')
    window_bytes=_probe_int(event,'window_payload_bytes')
    charge_bytes=_probe_int(event,'source_window_charge_bytes')
    reached=_probe_object(probe,'reached')
    for key in ('copy','consumer','record'):
        _probe_bool(reached,key,True)
    raw=_probe_object(probe,'raw')
    query_first=_probe_i32(raw,'query_first')
    query_last=_probe_i32(raw,'query_last')
    for key,value in (('query_first',query_first),('query_last',query_last)):
        require(value is None or value in (0,600),
                'actual raw query Success0/NotReady600 '+key)
    success=_probe_int(raw,'query_success_count')
    not_ready=_probe_int(raw,'query_not_ready_count')
    _probe_int(raw,'query_error_count',0)
    synchronize=_probe_i32(raw,'synchronize')
    destroy=_probe_i32(raw,'destroy')
    require(synchronize is None or synchronize==0,
            'no driver synchronization error in a controlled pass')
    require(destroy is None,
            'actual event is deliberately not destroyed in a controlled pending probe')
    injection=_probe_object(probe,'injection')
    _probe_int(injection,'hits',1)
    require(injection.get('stage')==('destroy' if mode=='source-page-destroy-unknown'
                                     else 'wait' if mode=='source-page-wait-unknown' else 'query'),
            'probe injection stage joins the selected mode')
    require(injection.get('result')==('injected-destroy-refused' if mode=='source-page-destroy-unknown'
                                      else 'injected-policy-unknown'),'probe injection result')
    _probe_int(injection,'retirement_hold_count')
    counters=_probe_object(probe,'counters')
    retire=_probe_int(counters,'selected_retire_success')
    recycle=_probe_int(counters,'selected_recycle_success')
    destroy_calls=_probe_int(counters,'selected_destroy_calls')
    destroy_success=_probe_int(counters,'selected_destroy_success')
    witnesses=_probe_object(probe,'witnesses')
    run_witness=_probe_bool(witnesses,'run_owner_after_public_drop')
    window_witness=_probe_bool(witnesses,'window_after_public_drop')
    pending_required=_probe_bool(witnesses,'pending_window_required')
    ledger=_probe_object(probe,'ledger')
    before=_ledger_domain(_probe_object(ledger,'before'),'before')
    fault=_ledger_domain(_probe_object(ledger,'fault'),'fault')
    after=_ledger_domain(_probe_object(ledger,'after_public_drop'),'after_public_drop')
    for name in ('host','device'):
        require(before[name]['limit']==fault[name]['limit']==after[name]['limit'],
                'original ledger limit unchanged '+name)
    require(fault['device']['fenced'] is True and after['device']['fenced'] is True,
            'original device ledger fenced at/after injected fault')
    outcome=_probe_object(probe,'outcome')
    for key in ('original_service_fenced','original_native_fenced',
                'qwen_reentry_refused','gemma_reentry_refused',
                'qwen_close_refused','gemma_close_refused'):
        _probe_bool(outcome,key,True)
    _probe_bool(outcome,'finish_failed',True if mode=='source-page-destroy-unknown' else None)
    _probe_bool(outcome,'publication',False); _probe_bool(outcome,'commit',False)
    if mode=='source-page-destroy-unknown':
        require(destroy_calls==0 and destroy_success==0,
                'actual hipEventDestroy skipped and raw status null')
        require(retire>=1, 'retired successfully before the injected destroy refusal')
        require(pending_required is False,
                'destroy probe does not require a pending window')
        # Weak run/window may legitimately be false after ordinary retirement.
    else:
        require(query_first is not None and query_last is not None
                and success+not_ready>=1,
                'actual raw query Success/NotReady observation')
        if mode=='source-page-wait-unknown':
            require(synchronize==0,'actual raw synchronize Success')
        require(retire==0 and recycle==0 and destroy_calls==0 and destroy_success==0,
                'no selected retirement/recycle/destroy success after injected error')
        require(run_witness is True and window_witness is True
                and pending_required is True,
                'actual pending run/window witnesses after public drop')
        require(window_bytes>0 and charge_bytes>0,
                'actual pending window payload/charge extent')
        require(after['host']['reserved']>=charge_bytes,
                'retained original pending source-window charge')
    return {'status':r['status'],'mode':mode,'global_unknown':True,
            'pending_event_probe':True,'source_page_probe':probe,
            'math_state_owner_scope_only':True,'serving_speed_or_general_fit':False}
