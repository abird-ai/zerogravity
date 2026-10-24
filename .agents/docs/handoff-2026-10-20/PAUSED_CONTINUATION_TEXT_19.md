# Paused continuation text — volume 19

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4989; SHA256 `5be8a53cf66f04d8b7319b1e75d945befb9e5d79d5682f06fffbbcf2e1593fb2`.

````text
"""ROOT-only source/input binding. Import performs no hashing/provider/model calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PRIMITIVE=HERE.parent/'numerics-and-context-bulk-f32-primitive-20261020'
CASES=('av','alpha')
MODELS={'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
        'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def require(ok,message):
    if not ok: raise ValueError(message)

def digest(path,cap=None):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode)and(cap is None or before.st_size<=cap),'bounded regular file')
        h=hashlib.sha256();count=0
        while raw:=f.read(1024*1024):h.update(raw);count+=len(raw)
        after=os.fstat(f.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and count==before.st_size,
                'same immutable FD')
        return {'bytes':count,'sha256':h.hexdigest()}

def read(path,cap=1024*1024):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:
        require(os.fstat(f.fileno()).st_size<=cap,'bounded JSON')
        raw=f.read(cap+1)
    require(len(raw)<=cap,'actual JSON cap')
    def unique(rows):
        out={}
        for key,value in rows:require(key not in out,'duplicate key');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique)

def write(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    require(len(raw)<=1024*1024,'bounded report')
    with Path(path).open('xb')as f:os.chmod(path,0o600);f.write(raw)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

def original():
    """Bind unchanged original prepare.py to ITS unchanged contracts, not ours."""
    old=load(PRIMITIVE/'contracts.py','typed_bulk_immutable_primitive_contracts')
    prior=sys.modules.get('contracts')
    sys.modules['contracts']=old
    try:prep=load(PRIMITIVE/'prepare.py','typed_bulk_immutable_primitive_prepare')
    finally:
        if prior is None:sys.modules.pop('contracts',None)
        else:sys.modules['contracts']=prior
    return old,prep

def frozen():
    pin=read(HERE/'freeze-manifest.json')
    for row in pin['files']:
        require(digest(HERE/row['path'],2*1024*1024)==row['digest'],'reviewed proposal/helper changed')
    for row in pin['source_pins']:
        require(digest(row['path'],2*1024*1024)==row['digest'],'original source/metadata changed')
    return digest(HERE/'freeze-manifest.json')['sha256']

def parents():
    """ROOT ONLY: original full authentic source/readsets include model/DSO hashes."""
    old,prep=original();old.frozen();av,alpha=old.parents()
    return old,prep,av,alpha

def verify_transport(folder):
    old,prep=original()
    manifest=prep.verify_transport(folder)
    typed=read(folder/'typed-inputs.json')
    require(typed['status']=='AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY'
            and typed['arithmetic_executed']is False and typed['sourceprofile']=='GGUF_ORIGINAL'
            and typed['base']==0 and typed['capacity']==8448 and typed['teachers']==64,'closed pure64teacher scope')
    require(all(type(typed[k])is int for k in('base','capacity','teachers')),'exact teacher geometry types')
    require(digest(folder/'teacher64.u32le',256)==typed['teacher_digest'],'actual64 teacher bits')
    for row,case in zip(typed['models'],CASES,strict=True):
        require(row['case']==case and row['gguf']==str(REPO/'target/models'/('qwen35-0.8b'if case=='av'else'qwen35-2b')/'model.gguf')
                and row['gguf_digest']['sha256']==MODELS[case],'original model source namespace')
        # ROOT ONLY full source verification; no operator/model computation.
        require(digest(row['gguf'])==row['gguf_digest']and digest(row['config'],1024*1024)==row['config_digest'],
                'actual immutable whole GGUF/config')
    av_ids=old.AV.parent/'numerics-and-context-bulk-causality-20261020/root-first64-ar1-budget/input_ids.u32le'
    alpha=load(old.ALPHA/'prepare.py','typed_bulk_original_alpha_metadata')
    alpha_ids=alpha.CAPTURE/'input_ids.u32le'
    require(digest(av_ids,32768)==typed['teacher_sources'][0]['digest']and
            digest(alpha_ids,32768)==typed['teacher_sources'][1]['digest'],'original teacher source bindings')
    a=av_ids.read_bytes();b=alpha_ids.read_bytes()
    require(len(a)==len(b)==32768 and a==b and a[:256]==(folder/'teacher64.u32le').read_bytes(),
            'same actual full8192 teachers; first64 belongs to BOTH original captures')
    return manifest,typed

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/execution_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1013; SHA256 `7c9eb08206c46ef8f339848534c197babf7277d48c3c07753bcf3064011c10dc`.

````text
"""Additive runner binding only; original V1 math/source freeze remains exact."""
from contracts import HERE, digest, frozen, read, require

FILES=('sigterm_guard_v2.py','execution_v2.py','run_replay_v2.py','validate_v2.py','test_execution_v2.py')

def verify_execution():
    root=HERE/'followup-signal-v2';m=read(root/'freeze-manifest.json')
    require(frozen()==m['original_source_freeze']=='0d11d476e2955a40eeb1dbe45c478b4502dede72a92ebb2e4f092e7481e7a6e1',
            'same immutable V1 source/math/build contract')
    require([row['file']for row in m['execution_files']]==list(FILES),'closed execution file list')
    for row in m['execution_files']:
        require(digest(HERE/row['file'],1024*1024)==row['digest'],'exact reviewed execution guard/runner')
    require(digest(HERE/'sigterm_guard_v2.py')==m['accepted_guard']['digest']==digest(m['accepted_guard']['path'],1024*1024),
            'unchanged accepted two-signal record-only guard')
    return digest(root/'freeze-manifest.json')['sha256']

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/run_replay_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7029; SHA256 `edbc7f3e43b44cfb938db9a19ac007fe50ec40bec49501e41b5f006b585fa3b9`.

````text
"""ROOT ONLY: bounded owned SAME engine pure operator replay; no model forward."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
from contracts import HERE, digest, frozen, load, parents, read, require, write
from contracts import verify_transport
from sigterm_guard_v2 import SigtermCancellation
from execution_v2 import verify_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--build-attestation', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    execution_pin = verify_execution()
    build = read(args.build_attestation)
    binary = digest(args.binary, 64 * 1024 * 1024)
    require(args.binary.is_absolute() and binary['sha256'] == args.binary_sha256,
            'actual bounded fresh ROOT binary')
    require(build['status'] == 'ROOT_COMPILED_SAME_ENGINE_BULK_F32_ONLY' and build['build_exit_code'] == 0
            and build['freeze_sha256'] == pin and build['binary'] == str(args.binary)
            and build['binary_digest'] == binary and len(build['effective_lock_sha256']) == 64
            and build['main_digest'] == digest(HERE / 'main.rs'), 'actual ROOT build/source binding')
    proposed = HERE / 'proposed'
    actual_files = []
    for path in sorted(proposed.rglob('*.rs')):
        actual = HERE.parents[2] / path.relative_to(proposed)
        expected = digest(path)
        require(digest(actual) == expected, 'ROOT actual merged primitive bytes')
        actual_files.append({'path':str(actual),'digest':expected})
    primitive=HERE.parent/'numerics-and-context-bulk-f32-primitive-20261020'
    dependencies=[]
    for relative in ('crates/zerogravity-cpu/src/math.rs','crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs',
                     'crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs'):
        frozen_source=primitive/('followup-attribution/proposed' if relative.endswith('haswell_bulk_f32.rs')else'proposed')/relative
        expected=digest(frozen_source);actual=HERE.parents[2]/relative
        require(digest(actual)==expected,'actual unchanged primitive + Mozilla attribution')
        dependencies.append({'path':str(actual),'digest':expected})
    require(build['production_files'] == actual_files and build['primitive_dependency_files']==dependencies,
            'actual ROOT compiled production/dependency source pins')
    require(args.fixtures.is_absolute() and args.fixtures.parent.resolve() == HERE,
            'own authentic fixture transport')
    parents()
    _,inputs = verify_transport(args.fixtures)
    fixture = digest(args.fixtures / 'manifest.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    require(args.new_run.is_absolute() and args.new_run.parent.resolve() == HERE
            and not args.new_run.exists(), 'fresh own run')
    args.new_run.mkdir(mode=0o700)
    output = args.new_run / 'output'
    model_args=[]
    for row in inputs['models']: model_args.extend((row['gguf'],row['config']))
    argv = [str(args.binary),str(args.fixtures),str(output),*model_args,str(args.fixtures/'teacher64.u32le')]
    receipt = {'status':'UNQUALIFIED','freeze_sha256':pin,'argv':argv,'binary':binary,
               'build_attestation':build,'fixture_manifest':fixture,'actual_affinity_expected':cpus,
               'wall_seconds':300,'as_limit_bytes':4*1024**3,'math_scope':'SAME HostMath pure matrix; original source materialized; no forward/session',
               'provider_model_GPU_execution':False,'execution_freeze_sha256':execution_pin,
               'record_only_cancellation_signals':['SIGTERM','SIGINT']}
    write(args.new_run / 'prepared-invocation.json', receipt)
    h = load(HERE / 'owned_lifetime.py','bulk_f32_owned_lifetime')
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','RAYON_NUM_THREADS':'4',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'-1',
           'NVIDIA_VISIBLE_DEVICES':'void','LD_PRELOAD':'','LD_AUDIT':''}
    def limits():
        os.sched_setaffinity(0,cpus)
        for kind,value in ((resource.RLIMIT_AS,4*1024**3),(resource.RLIMIT_CPU,1200),
                           (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0),
                           (resource.RLIMIT_FSIZE,8*1024**2),(resource.RLIMIT_STACK,8*1024**2)):
            resource.setrlimit(kind,(value,value))
    child = log = failure = None
    start = time.monotonic()
    with SigtermCancellation() as cancellation:
      try:
        cancellation.check()
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        receipt['owned_cli_pid'] = child.pid
        cancellation.check()
        log = h.Log(child.stdout,args.new_run / 'execution.log')
        receipt['exit_code'] = cancellation.observe(log,child,300)
      except BaseException as error:
        failure = error
      finally:
        errors = receipt['cleanup_errors'] = []
        if child is not None:
            try:
                h.cleanup_group(child,receipt)
            except BaseException as error:
                errors.append('owned foreground:'+repr(error))
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                errors.append('log:'+repr(error))
            finally:
                receipt['log_cleanup'] = {'pipe_eof':log.eof,'all_closed':log.closed,
                    'total_bytes':log.total,'retained_bytes':log.retained}
        elif child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as error:
                errors.append('uninitialized pipe:'+repr(error))
        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-start
        receipt['sigterm_received'] = cancellation.received
    if failure is not None:
        receipt['failure'] = repr(failure)
    good = failure is None and receipt.get('exit_code') == 0 and not receipt['cleanup_errors'] and not receipt['sigterm_received']
    if good:
        receipt['status'] = 'EXECUTED_SAME_ENGINE_PURE_BULK_REQUIRES_VALIDATION'
    write(args.new_run / 'invocation.json', receipt)
    require(good, 'failed source/control/cleanup attempt retained; no retry/fallback')
    require(digest(args.binary) == binary and digest(args.fixtures / 'manifest.json') == fixture,
            'unchanged binary/transport after known retirement')
    parents()
    verify_transport(args.fixtures)
    require(frozen() == pin and verify_execution() == execution_pin, 'unchanged source/execution after replay')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/validate.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7060; SHA256 `cff5bf92b15b2e50eb189b0d2dcdec6520d430476907e0fc3e95f70a3bfb6cb4`.

````text
"""ROOT ONLY independent whole-output/three-node/source/ledger validator."""
import argparse
from pathlib import Path
from contracts import HERE, CASES, MODELS, digest, frozen, parents, read, require, verify_transport, write

META=4*1024**2
CACHE=16*1024**2
CONTEXT=16384
TRANSPORT=32*1024**2
WORDS={'av':131072,'alpha':1024}
INPUTS={'av':(4317184+131072)*4,'alpha':131072*4}

def ledger(row,*,metadata=0,staging=0,output=0,weights=0,reservations=0):
    scratch=256 if output else 0
    require(set(row)=={'reserved','reservations','fenced','output','scratch','staging','metadata','weights'},'exact live ledger fields')
    require(all(type(row[k])is int for k in row if k!='fenced')and row['fenced']is False,'actual ledger scalar types')
    require(row=={'reserved':metadata+staging+output+scratch+weights,'reservations':reservations,'fenced':False,
                  'metadata':metadata,'staging':staging,'output':output,'scratch':scratch,'weights':weights},
            'actual class/retained owner/charge conservation')

def report(value):
    require(value['status']=='PASS_SAME_ENGINE_PURE_BULK_F32_ONLY'and type(value['words'])is int and value['words']==132096,
            'complete pure operator scope')
    for key in('default_forward_changed','fullmodel_or_StateRoot_qualified','historical_prefix_recovered',
               'actual_mid_operation_cancel_qualified','physical_RAM_admission_qualified'):
        require(value[key]is False,'no broader capability/schedule/history/admission claim')
    cpus=value['actual_affinity'];require(len(cpus)==4 and len(set(cpus))==4 and all(type(n)is int and 0<=n<1024 for n in cpus),'actual four CPU set')
    for key in('mxcsr_before','mxcsr_after'):
        require(type(value[key])is int and value[key]&0xffc0==0x1f80,'unchanged RNE/noFTZ/noDAZ controls')
    ledger(value['before']);ledger(value['final'])
    require([r['case']for r in value['cases']]==list(CASES),'complete distinct authentic population')
    for row in value['cases']:
        case=row['case'];size=WORDS[case]*4
        require(row['model_sha256']==MODELS[case]and type(row['words'])is int and row['words']==WORDS[case]
                and type(row['unequal_words'])is int and row['unequal_words']==0,'all unchanged default raw output words')
        require(type(row['source_bytes'])is int and 0<row['source_bytes']<2*1024**3,'actual bounded source envelope')
        require(row['source_bound']is True and row['expected_opened_after_math']is True and row['output_clone_retains_source']is True,
                'actual source/result lifetime and expected isolation')
        for key in('pre_cancel_constructor','pre_cancel_execution','real_one_byte_pressure_refused'):
            require(row[key]is True,'actual cancellation/admission refusal')
        require(type(row['provider_callbacks'])is int and row['provider_callbacks']==0,'unaware provider not called')
        require(type(row['plan_nodes'])is int and row['plan_nodes']==3 and type(row['frontier_delta'])is int and row['frontier_delta']==0
                and type(row['state_domains'])is int and row['state_domains']==0,'same three-node no-state plan')
        require(all(type(n)is int for key in('left_shape','left_strides','right_shape')for n in row[key]),'exact integer geometry')
        require(row['left_shape']==([256,256,2,1]if case=='av'else[2048,16,1,1])
                and row['left_strides']==([1,8448,2162688,4325376]if case=='av'else[1,2048,32768,32768])
                and row['right_shape']==([256,64,8,1]if case=='av'else[2048,64,1,1]),'actual fullK/original stride/N shapes')
        ledger(row['live'],metadata=META+CACHE+CONTEXT,staging=TRANSPORT+INPUTS[case],output=size,weights=row['source_bytes'],reservations=6)
        ledger(row['retained_output'],metadata=META+CACHE+CONTEXT,output=size,weights=row['source_bytes'],reservations=4)
        ledger(row['after_case'],metadata=META,reservations=1)

def invocation(value):
    require(value['status']=='EXECUTED_SAME_ENGINE_PURE_BULK_REQUIRES_VALIDATION'and value['exit_code']==0
            and value['cleanup_errors']==[]and value['sigterm_received']is False
            and value['provider_model_GPU_execution']is False and value['wall_seconds']==300 and value['as_limit_bytes']==4*1024**3
            and value['math_scope']=='SAME HostMath pure matrix; original source materialized; no forward/session','actual bounded invocation')
    g=value['foreground_cleanup'];log=value['log_cleanup']
    require(g['pgid']==value['owned_cli_pid']and g['leader_reaped']is True and g['group_absent']is True and g['stage_errors']==[]
            and g['all_nonzero_signals_before_reap']is True and g['kill_attempts_before_reap']==2 and g['leader_reap_attempts']in(1,2),
            'original unreaped owner/group retirement')
    require(log['pipe_eof']is True and log['all_closed']is True and log['total_bytes']==log['retained_bytes']<=4*1024**2,
            'bounded complete owned pipe drain/close')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args();pin=frozen();parents();manifest,inputs=verify_transport(a.fixtures)
    require(a.run.is_absolute()and a.run.parent.resolve()==HERE and a.fixtures.is_absolute()and a.fixtures.parent.resolve()==HERE,'own ROOT artifacts')
    called=read(a.run/'invocation.json');invocation(called)
    expected_argv=[str(a.fixtures),str(a.run/'output')]
    for row in inputs['models']:expected_argv.extend((row['gguf'],row['config']))
    expected_argv.append(str(a.fixtures/'teacher64.u32le'))
    require(called['freeze_sha256']==pin and called['fixture_manifest']==digest(a.fixtures/'manifest.json')
            and called['binary']==digest(called['argv'][0],64*1024**2)and called['argv'][1:]==expected_argv,'actual full invocation/source binding')
    value=read(a.run/'output/report.json');report(value)
    require(all(row['source_bytes']==model['gguf_digest']['bytes']for row,model in zip(value['cases'],inputs['models'],strict=True)),
            'actual materialized source envelope equals authenticated file length')
    require(value['actual_affinity']==called['actual_affinity_expected'],'same actual CPU set')
    for row in manifest['cases']:
        actual=a.run/'output'/f"{row['case']}.actual.f32le";expected=a.fixtures/row['expected']['file']
        require(digest(actual,1024**2)==row['expected']['digest']==digest(expected,1024**2)
                and actual.read_bytes()==expected.read_bytes(),'complete independent DEFAULT raw bytes; no tolerance')
    require(frozen()==pin,'unchanged reviewed source')
    write(a.run/'validation.json',{'status':'PASS_SAME_ENGINE_PURE_BULK_F32_ONLY','freeze_sha256':pin,'words':132096,
          'cases':list(CASES),'invocation':digest(a.run/'invocation.json'),'report':digest(a.run/'output/report.json'),
          'fullmodel_default_or_state_qualification':False,'historical_scope_qualified':False,'outer_RAM_admission_qualified':False})
if __name__=='__main__':main()

````

## Source: .agents/runs/numerics-and-context-default-bulk-isolated-20261020/baseline/validate_v2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1203; SHA256 `d54e35c1be96fd47f1a75707c256690ff024e4c16399f0c3831c4254828c3bfb`.

````text
"""ROOT ONLY V1 complete math validation +additive two-signal execution binding."""
import argparse
from pathlib import Path
import validate as original
from contracts import digest,read,require,write
from execution_v2 import verify_execution

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args();pin=verify_execution();called=read(a.run/'invocation.json')
    require(called['execution_freeze_sha256']==pin and called['record_only_cancellation_signals']==['SIGTERM','SIGINT'],
            'actual fresh runner receipt binds both record-only signals')
    original.main() # Unchanged complete source/bytes/ledger/lifecycle validator.
    require(verify_execution()==pin,'same execution source after independent validation')
    write(a.run/'validation-execution-v2.json',{'status':'PASS_BOUND_EXECUTION_ONLY_V2','execution_freeze_sha256':pin,
          'original_math_validation':digest(a.run/'validation.json'),'invocation':digest(a.run/'invocation.json'),
          'actual_signal_injection_qualified':False,'math_source_changed':False})
if __name__=='__main__':main()

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1110; SHA256 `097cca66357b43e91563f216f67d64746df57a8649288b762d671c494692289c`.

````text
{
  "schema": "source_advisory_freeze_v1",
  "owner": "spine_review",
  "status": "SOURCE_ONLY_NOT_PHYSICALLY_QUALIFIED",
  "files": [
    {
      "path": "API_FACTS.md",
      "bytes": 5681,
      "sha256": "db0611585928ce815ca3b3cdb67aaa05cb7b7a4cd86445094a40f661daddab4f"
    },
    {
      "path": "DESIGN.md",
      "bytes": 6455,
      "sha256": "a088031ff7bd088959f0938785dffc9a4baa8e56991eb3877de958ceccf83f6a"
    },
    {
      "path": "README.md",
      "bytes": 1497,
      "sha256": "f1cbbbdb76240a781e71237d2e84e42c6b371af8cd3f9e847bd7c2b7f48899d7"
    },
    {
      "path": "ROOT_GATES.md",
      "bytes": 2983,
      "sha256": "77a903d628bb06da5c46e1ec8190f7defcb6311a81d3a8c1e2431f680199be8d"
    },
    {
      "path": "TODO.md",
      "bytes": 1147,
      "sha256": "141c9c9ad4f2decdf626e742b93d9aa23ef2eee14bacd1e5968641a82f375dcd"
    },
    {
      "path": "source-pins.json",
      "bytes": 6613,
      "sha256": "9fa2195c0d8fb274dd8aa6238f290f9d4c7fa791ad892eb7d7b67178732fbcb6"
    }
  ],
  "source_pins": "source-pins.json",
  "source_files_count": 15,
  "author_no_runtime": true
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/frozen/source-pins.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6613; SHA256 `9fa2195c0d8fb274dd8aa6238f290f9d4c7fa791ad892eb7d7b67178732fbcb6`.

````text
{
  "schema": "ordered_silu_loaded_sass_source_advisory_v1",
  "runtime_executed": false,
  "source_files": [
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/context.rs",
      "bytes": 25136,
      "sha256": "af203385958ca07d9c78c984ae0f333e5c29c76f4161aa66959d6db061b7d98d",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-gpu/vendor/cubecl-cuda/src/compute/driver.rs",
      "bytes": 9429,
      "sha256": "21f82dc010a50fa99305b9266ff80d259d3670a944e73cb2c02ed8149d18a8a2",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/include/cupti_callbacks.h",
      "bytes": 29689,
      "sha256": "699f9213460c15f4fd47e521e4c1dba0f834ca91cc8de492009c37cdd3fb382b",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/include/cupti_activity.h",
      "bytes": 232296,
      "sha256": "8f1be5852d80b06abbe2e5be103ae897a8c16477146968f04fe7b137007f12e6",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/include/generated_cuda_meta.h",
      "bytes": 100043,
      "sha256": "f478caf7935cb90c1666a3b615f9d7b14fbf7c655d2a574b76fd6b5982f71f9b",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/include/cupti_driver_cbid.h",
      "bytes": 77476,
      "sha256": "7f1db3d6cf6566283d6d32d55c0472adf63d70ce8b086dd532d6190178199fe1",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/include/cupti_version.h",
      "bytes": 4569,
      "sha256": "f4e267df16488bf98980f77f3cde772c64a5edc5d76bdab61de0aabb8f7cce17",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/include/cuda.h",
      "bytes": 1168299,
      "sha256": "fda3285d055a9f05e40cd76352f9f073d0ab45d42fb5bb0db4bb33b0df506d28",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/capacity-v1/collector_capacity.cpp",
      "bytes": 18769,
      "sha256": "83c63c0431d867fc437d876b9bf48b0e7455dfa5b91ce2ced9188c9488803838",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-cupti-activity-20261020/container-v1/parent-bindings.json",
      "bytes": 1703,
      "sha256": "2a6204cca537c7d76d1b6536e3c357149bae21f8eae14ac83d83832c42999a66",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/proposed/crates/zerogravity-gpu/src/ordered_silu.rs",
      "bytes": 14175,
      "sha256": "3d789c06acfca8ea77ca1bf79051005fc621162387e45b75478352e7497b6bc0",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-scalar-input-20261020/freeze-manifest.json",
      "bytes": 6869,
      "sha256": "16198590093e2dcf23c4376dc4358f05e3dcdd2527de4332e1e86546914213b8",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-vllm-gguf-baseline-20261020/root-build-2/image-inspect.json",
      "bytes": 59681,
      "sha256": "2bd9a3c02231fcea4aa60c10cb1c18180ad8989d179e7bf2af6d5d5080a5e878",
      "scope": "source_header_or_existing_metadata_only"
    },
    {
      "path": "/home/pvl/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.10/src/driver/result.rs",
      "bytes": 64277,
      "sha256": "a6e108c005942ab93b0f5a84d0913763ae633a72740e33e77e18cb82ddfcf504",
      "scope": "source_or_existing_dependency_metadata_only"
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/Cargo.lock",
      "bytes": 139446,
      "sha256": "d46da60a19caa05aeeca07315a837709117b9a84e5bec980c4fb2591dd87d4ec",
      "scope": "source_or_existing_dependency_metadata_only"
    }
  ],
  "approved_library_metadata_only": {
    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/7940bf3c960e46bcf6f5848ed00b3dadaa7d399d891836cd12dcd4746cb4c8cf/diff/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib/libcupti.so.12",
    "bytes": 7604800,
    "sha256": "2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a",
    "identity_from": "existing container-v1/parent-bindings.json; binary not read or rehashed by author"
  },
  "web_primary_urls": [
    "https://docs.nvidia.com/cupti/12.9/main/main.html",
    "https://docs.nvidia.com/cupti/12.9/api/structCUpti__ModuleResourceData.html",
    "https://docs.nvidia.com/cupti/12.9/api/structCUpti__ResourceData.html",
    "https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__CALLBACK__API.html"
  ],
  "not_closed": [
    "exact_module_callback_payload_dispatch",
    "actual_module_function_launch_association",
    "actual_copied_cubin",
    "actual_tool_file_and_closure",
    "physical_transparency",
    "actual_SASS_inspection"
  ]
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/fetch_primary_source.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6870; SHA256 `4e1a0658a600dd9a148b01fe961b0644e7ac615a62ec2518fc368f1d899d546e`.

````text
#!/usr/bin/env python3
"""Fetch only official ZIP metadata and named source/license entries; no SDK."""

from pathlib import Path
import hashlib
import json
import struct
import urllib.request
import zlib


BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST_URL = BASE + "redistrib_12.9.1.json"
ARCHIVE_URL = BASE + (
    "cuda_cupti/windows-x86_64/"
    "cuda_cupti-windows-x86_64-12.9.79-archive.zip"
)
ARCHIVE_SIZE = 13_359_912
ARCHIVE_PUBLISHED_SHA = (
    "c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa"
)
OUT = Path(__file__).resolve().parent
MAX_SOURCE = 1 << 20
MAX_COMPRESSED = 1 << 19
events = []


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("redirect refused")


opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)


def save(name, raw):
    with (OUT / name).open("xb") as stream:
        stream.write(raw)


def get(url, limit, start=None, end=None):
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        assert 0 <= start <= end < ARCHIVE_SIZE
        headers["Range"] = f"bytes={start}-{end}"
        assert end - start + 1 <= limit
    with opener.open(urllib.request.Request(url, headers=headers), timeout=20) as response:
        assert response.url == url
        if start is None:
            assert response.status == 200
        else:
            # Never consume a full-archive response if the server ignores Range.
            assert response.status == 206, "Range unsupported; no archive consumed"
            assert response.headers["Content-Range"] == (
                f"bytes {start}-{end}/{ARCHIVE_SIZE}"
            )
            assert int(response.headers["Content-Length"]) == end - start + 1
        assert response.headers.get("Content-Encoding", "identity") == "identity"
        raw = response.read(limit + 1)
        assert len(raw) <= limit
        if start is not None:
            assert len(raw) == end - start + 1
        events.append({
            "url": url, "range_start": start, "range_end": end,
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        })
        return raw


manifest = get(MANIFEST_URL, 1 << 18)
save("nvidia-redistrib-12.9.1.json", manifest)
component = json.loads(manifest)["cuda_cupti"]
assert component["version"] == "12.9.79"
item = component["windows-x86_64"]
assert BASE + item["relative_path"] == ARCHIVE_URL
assert int(item["size"]) == ARCHIVE_SIZE
assert item["sha256"] == ARCHIVE_PUBLISHED_SHA

tail_start = ARCHIVE_SIZE - 22
tail = get(ARCHIVE_URL, 22, tail_start, ARCHIVE_SIZE - 1)
save("zip-eocd-metadata.bin", tail)
eocd = 0
# Refuse a commented/ZIP64 archive instead of sweeping a tail that might also
# contain compressed library bytes. This request is exactly the EOCD candidate.
assert tail[:4] == b"PK\x05\x06"
_, disk, cd_disk, disk_entries, entries, cd_size, cd_start, comment_size = (
    struct.unpack_from("<4s4H2IH", tail, eocd)
)
assert disk == cd_disk == 0 and disk_entries == entries
assert 0 < entries <= 4096 and 0 < cd_size <= 1 << 20
assert eocd + 22 + comment_size == len(tail)
assert cd_start + cd_size == tail_start + eocd
directory = get(ARCHIVE_URL, 1 << 20, cd_start, cd_start + cd_size - 1)
save("zip-central-directory-metadata.bin", directory)
records = []
cursor = 0
for _ in range(entries):
    assert cursor + 46 <= len(directory)
    fields = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
    assert fields[0] == b"PK\x01\x02"
    flags, method, crc, compressed, size = (
        fields[3], fields[4], fields[7], fields[8], fields[9]
    )
    name_len, extra_len, comment_len = fields[10:13]
    assert fields[13] == 0 and 0 < name_len <= 1024
    stop = cursor + 46 + name_len + extra_len + comment_len
    assert stop <= len(directory)
    name = directory[cursor + 46:cursor + 46 + name_len].decode("utf-8", "strict")
    assert not name.startswith("/") and ".." not in name.split("/")
    records.append({"name": name, "flags": flags, "method": method,
                    "crc32": crc, "compressed": compressed, "bytes": size,
                    "local_header": fields[16]})
    cursor = stop
assert cursor == len(directory)
save("zip-source-directory.json", (json.dumps(records, indent=2) + "\n").encode())

selected = [r for r in records if (
    r["name"].endswith("/samples/sass_source_map/sass_source_map.cu")
    or r["name"].count("/") == 1 and r["name"].endswith("/LICENSE")
    or r["name"].count("/") == 1 and r["name"].endswith("/LICENSE.txt")
)]
assert sum(r["name"].endswith("/sass_source_map.cu") for r in selected) == 1
assert len(selected) <= 3
for row in selected:
    assert not row["flags"] & 1 and row["method"] in (0, 8)
    assert 0 < row["bytes"] <= MAX_SOURCE
    assert 0 < row["compressed"] <= MAX_COMPRESSED
    start = row["local_header"]
    fixed = get(ARCHIVE_URL, 30, start, start + 29)
    header = struct.unpack("<4s5H3I2H", fixed)
    assert header[0] == b"PK\x03\x04"
    assert header[2] == row["flags"] and header[3] == row["method"]
    name_len, extra_len = header[-2:]
    assert 0 < name_len <= 1024 and extra_len <= 65_535
    variable = get(ARCHIVE_URL, 66_559, start + 30,
                   start + 30 + name_len + extra_len - 1)
    assert variable[:name_len].decode("utf-8", "strict") == row["name"]
    data_start = start + 30 + name_len + extra_len
    compressed = get(ARCHIVE_URL, MAX_COMPRESSED, data_start,
                     data_start + row["compressed"] - 1)
    if row["method"] == 8:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(compressed, row["bytes"] + 1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    else:
        raw = compressed
    assert len(raw) == row["bytes"]
    assert zlib.crc32(raw) & 0xFFFFFFFF == row["crc32"]
    raw.decode("utf-8", "strict")
    row["source_sha256"] = hashlib.sha256(raw).hexdigest()
    save(Path(row["name"]).name, raw)

receipt = {
    "scope": "exact_NVIDIA_12.9.79_source_entries_and_ZIP_metadata_only",
    "archive_url": ARCHIVE_URL, "archive_published_bytes": ARCHIVE_SIZE,
    "archive_published_sha256": ARCHIVE_PUBLISHED_SHA,
    "whole_archive_fetched_or_hash_verified": False,
    "library_or_tool_entries_fetched": False,
    "requests": events, "selected_source_entries": selected,
    "total_HTTP_body_bytes": sum(r["bytes"] for r in events),
    "runtime_or_build_executed": False,
}
save("source-acquisition.json", (json.dumps(receipt, indent=2) + "\n").encode())
print(json.dumps({"selected": selected,
                  "total_HTTP_body_bytes": receipt["total_HTTP_body_bytes"]}, indent=2))

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1877; SHA256 `b4b7f125344fdb1c1425adb3c9a965573a29c34e3a6bd1c3368cdf1ba83a37f6`.

````text
{
  "schema": "NVIDIA_exact_CUPTI_12_9_79_source_ABI_closure_v2",
  "scope": "source_only_no_physical_acceptance",
  "files": [
    {
      "path": "CLOSURE.md",
      "bytes": 3145,
      "sha256": "ae0f2c96dc5300568c8be88cc8d9dbaed6248ed0e2905641330dc884d95491bc"
    },
    {
      "path": "LICENSE",
      "bytes": 63522,
      "sha256": "213f2f1cc4bfecd53eb1579edc89d6e3a024ddd2f77413b69b40e2bf3c114d84"
    },
    {
      "path": "TODO.md",
      "bytes": 545,
      "sha256": "10213943ccdcdbed30775e37f2114cc6181768b9716585fb31425d923bb87fb2"
    },
    {
      "path": "fetch_primary_source.py",
      "bytes": 6870,
      "sha256": "4e1a0658a600dd9a148b01fe961b0644e7ac615a62ec2518fc368f1d899d546e"
    },
    {
      "path": "nvidia-redistrib-12.9.1.json",
      "bytes": 54256,
      "sha256": "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630"
    },
    {
      "path": "sass_source_map.cu",
      "bytes": 6152,
      "sha256": "4fff611da3ae8e1e9d886dcd1d79229c8a752b8403a6e0b46a03ecb52854940e"
    },
    {
      "path": "source-acquisition.json",
      "bytes": 5207,
      "sha256": "50de8543bcb6c7b30e19285efcf632d6daf71332ec63075e1afabeb6bfc5e89a"
    },
    {
      "path": "zip-central-directory-metadata.bin",
      "bytes": 83669,
      "sha256": "6ef34da41a920fdc6d0158b4ac0e4f3b90d7481a43e958ea2e0daf960d576b16"
    },
    {
      "path": "zip-eocd-metadata.bin",
      "bytes": 22,
      "sha256": "b898efeeeb4d13e895a6ebb77f791a2bba4a9495cbc9180d5ad5c53200fee4f2"
    },
    {
      "path": "zip-source-directory.json",
      "bytes": 118494,
      "sha256": "66613a7fdbe8406af5f44b4bcec750e86e4a9d718108f9560991a60cbab616bf"
    }
  ],
  "original_advisory_manifest": "097cca66357b43e91563f216f67d64746df57a8649288b762d671c494692289c",
  "actual_source_only_HTTP_body_bytes": 154973,
  "whole_archive_or_binary_member_read": false
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/frozen/fetch_primary_source.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6870; SHA256 `4e1a0658a600dd9a148b01fe961b0644e7ac615a62ec2518fc368f1d899d546e`.

````text
#!/usr/bin/env python3
"""Fetch only official ZIP metadata and named source/license entries; no SDK."""

from pathlib import Path
import hashlib
import json
import struct
import urllib.request
import zlib


BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST_URL = BASE + "redistrib_12.9.1.json"
ARCHIVE_URL = BASE + (
    "cuda_cupti/windows-x86_64/"
    "cuda_cupti-windows-x86_64-12.9.79-archive.zip"
)
ARCHIVE_SIZE = 13_359_912
ARCHIVE_PUBLISHED_SHA = (
    "c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa"
)
OUT = Path(__file__).resolve().parent
MAX_SOURCE = 1 << 20
MAX_COMPRESSED = 1 << 19
events = []


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("redirect refused")


opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), NoRedirect()
)


def save(name, raw):
    with (OUT / name).open("xb") as stream:
        stream.write(raw)


def get(url, limit, start=None, end=None):
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        assert 0 <= start <= end < ARCHIVE_SIZE
        headers["Range"] = f"bytes={start}-{end}"
        assert end - start + 1 <= limit
    with opener.open(urllib.request.Request(url, headers=headers), timeout=20) as response:
        assert response.url == url
        if start is None:
            assert response.status == 200
        else:
            # Never consume a full-archive response if the server ignores Range.
            assert response.status == 206, "Range unsupported; no archive consumed"
            assert response.headers["Content-Range"] == (
                f"bytes {start}-{end}/{ARCHIVE_SIZE}"
            )
            assert int(response.headers["Content-Length"]) == end - start + 1
        assert response.headers.get("Content-Encoding", "identity") == "identity"
        raw = response.read(limit + 1)
        assert len(raw) <= limit
        if start is not None:
            assert len(raw) == end - start + 1
        events.append({
            "url": url, "range_start": start, "range_end": end,
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        })
        return raw


manifest = get(MANIFEST_URL, 1 << 18)
save("nvidia-redistrib-12.9.1.json", manifest)
component = json.loads(manifest)["cuda_cupti"]
assert component["version"] == "12.9.79"
item = component["windows-x86_64"]
assert BASE + item["relative_path"] == ARCHIVE_URL
assert int(item["size"]) == ARCHIVE_SIZE
assert item["sha256"] == ARCHIVE_PUBLISHED_SHA

tail_start = ARCHIVE_SIZE - 22
tail = get(ARCHIVE_URL, 22, tail_start, ARCHIVE_SIZE - 1)
save("zip-eocd-metadata.bin", tail)
eocd = 0
# Refuse a commented/ZIP64 archive instead of sweeping a tail that might also
# contain compressed library bytes. This request is exactly the EOCD candidate.
assert tail[:4] == b"PK\x05\x06"
_, disk, cd_disk, disk_entries, entries, cd_size, cd_start, comment_size = (
    struct.unpack_from("<4s4H2IH", tail, eocd)
)
assert disk == cd_disk == 0 and disk_entries == entries
assert 0 < entries <= 4096 and 0 < cd_size <= 1 << 20
assert eocd + 22 + comment_size == len(tail)
assert cd_start + cd_size == tail_start + eocd
directory = get(ARCHIVE_URL, 1 << 20, cd_start, cd_start + cd_size - 1)
save("zip-central-directory-metadata.bin", directory)
records = []
cursor = 0
for _ in range(entries):
    assert cursor + 46 <= len(directory)
    fields = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
    assert fields[0] == b"PK\x01\x02"
    flags, method, crc, compressed, size = (
        fields[3], fields[4], fields[7], fields[8], fields[9]
    )
    name_len, extra_len, comment_len = fields[10:13]
    assert fields[13] == 0 and 0 < name_len <= 1024
    stop = cursor + 46 + name_len + extra_len + comment_len
    assert stop <= len(directory)
    name = directory[cursor + 46:cursor + 46 + name_len].decode("utf-8", "strict")
    assert not name.startswith("/") and ".." not in name.split("/")
    records.append({"name": name, "flags": flags, "method": method,
                    "crc32": crc, "compressed": compressed, "bytes": size,
                    "local_header": fields[16]})
    cursor = stop
assert cursor == len(directory)
save("zip-source-directory.json", (json.dumps(records, indent=2) + "\n").encode())

selected = [r for r in records if (
    r["name"].endswith("/samples/sass_source_map/sass_source_map.cu")
    or r["name"].count("/") == 1 and r["name"].endswith("/LICENSE")
    or r["name"].count("/") == 1 and r["name"].endswith("/LICENSE.txt")
)]
assert sum(r["name"].endswith("/sass_source_map.cu") for r in selected) == 1
assert len(selected) <= 3
for row in selected:
    assert not row["flags"] & 1 and row["method"] in (0, 8)
    assert 0 < row["bytes"] <= MAX_SOURCE
    assert 0 < row["compressed"] <= MAX_COMPRESSED
    start = row["local_header"]
    fixed = get(ARCHIVE_URL, 30, start, start + 29)
    header = struct.unpack("<4s5H3I2H", fixed)
    assert header[0] == b"PK\x03\x04"
    assert header[2] == row["flags"] and header[3] == row["method"]
    name_len, extra_len = header[-2:]
    assert 0 < name_len <= 1024 and extra_len <= 65_535
    variable = get(ARCHIVE_URL, 66_559, start + 30,
                   start + 30 + name_len + extra_len - 1)
    assert variable[:name_len].decode("utf-8", "strict") == row["name"]
    data_start = start + 30 + name_len + extra_len
    compressed = get(ARCHIVE_URL, MAX_COMPRESSED, data_start,
                     data_start + row["compressed"] - 1)
    if row["method"] == 8:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(compressed, row["bytes"] + 1)
        assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    else:
        raw = compressed
    assert len(raw) == row["bytes"]
    assert zlib.crc32(raw) & 0xFFFFFFFF == row["crc32"]
    raw.decode("utf-8", "strict")
    row["source_sha256"] = hashlib.sha256(raw).hexdigest()
    save(Path(row["name"]).name, raw)

receipt = {
    "scope": "exact_NVIDIA_12.9.79_source_entries_and_ZIP_metadata_only",
    "archive_url": ARCHIVE_URL, "archive_published_bytes": ARCHIVE_SIZE,
    "archive_published_sha256": ARCHIVE_PUBLISHED_SHA,
    "whole_archive_fetched_or_hash_verified": False,
    "library_or_tool_entries_fetched": False,
    "requests": events, "selected_source_entries": selected,
    "total_HTTP_body_bytes": sum(r["bytes"] for r in events),
    "runtime_or_build_executed": False,
}
save("source-acquisition.json", (json.dumps(receipt, indent=2) + "\n").encode())
print(json.dumps({"selected": selected,
                  "total_HTTP_body_bytes": receipt["total_HTTP_body_bytes"]}, indent=2))

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/frozen/nvidia-redistrib-12.9.1.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 54256; SHA256 `8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630`.

````text
{
    "release_date": "2025-06-05",
    "release_label": "12.9.1",
    "release_product": "cuda",
    "collectx_bringup": {
        "name": "UFM telemetry CollectX Bringup",
        "license": "NVIDIA Proprietary",
        "license_path": "collectx_bringup/LICENSE.txt",
        "version": "1.19.13",
        "linux-x86_64": {
            "relative_path": "collectx_bringup/linux-x86_64/collectx_bringup-linux-x86_64-1.19.13-archive.tar.xz",
            "sha256": "4965fe096fca645a6d4565fd4d12366ae0ec31ec54ca20274910f34eea74b41a",
            "md5": "d48b8c177f312c8774c54988594f725c",
            "size": "125307508"
        },
        "linux-sbsa": {
            "relative_path": "collectx_bringup/linux-sbsa/collectx_bringup-linux-sbsa-1.19.13-archive.tar.xz",
            "sha256": "5d716fbb80ef7b9ac89f9e3fa8b900a29d4e6c0be487420dc4ee7a05340b57a9",
            "md5": "dc2c0bfedf790ba7dc21f684510e592e",
            "size": "120696772"
        }
    },
    "cuda_cccl": {
        "name": "CXX Core Compute Libraries",
        "license": "CUDA Toolkit",
        "license_path": "cuda_cccl/LICENSE.txt",
        "version": "12.9.27",
        "linux-x86_64": {
            "relative_path": "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz",
            "sha256": "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa",
            "md5": "59e3cb66aeb96423dcca73218c85bc02",
            "size": "997888"
        },
        "linux-sbsa": {
            "relative_path": "cuda_cccl/linux-sbsa/cuda_cccl-linux-sbsa-12.9.27-archive.tar.xz",
            "sha256": "8c3da24801b500f1d9217d191bb4b63e5d2096c8e7d0b7695e876853180ba82f",
            "md5": "b39d03cfac57a3b2ded8e9b0a6b0e782",
            "size": "997840"
        },
        "windows-x86_64": {
            "relative_path": "cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.9.27-archive.zip",
            "sha256": "17aaa7c6b8f94a417d8f3261780b7e34b9cbdfab7513bce86768623b06aa28b5",
            "md5": "afdfa89a35951f67b5bd70c513f5ef09",
            "size": "3127658"
        },
        "linux-aarch64": {
            "relative_path": "cuda_cccl/linux-aarch64/cuda_cccl-linux-aarch64-12.9.27-archive.tar.xz",
            "sha256": "8c87c2db67130108861609eacd40d30ee109656a7765e5172eab71dd6da4c453",
            "md5": "25b1a257ed35393052577bd9ee432d3f",
            "size": "998228"
        }
    },
    "cuda_compat": {
        "name": "CUDA compat L4T",
        "license": "CUDA Toolkit",
        "license_path": "cuda_compat/LICENSE.txt",
        "version": "12.9.40580548",
        "linux-aarch64": {
            "relative_path": "cuda_compat/linux-aarch64/cuda_compat-linux-aarch64-12.9.40580548-archive.tar.xz",
            "sha256": "c03fe18b9d23881f068ac97b45001a47778f83d2361728427a3f92b551b1c898",
            "md5": "3ca26403b8b36c5cc18104ee174ec874",
            "size": "37422044"
        }
    },
    "cuda_cudart": {
        "name": "CUDA Runtime (cudart)",
        "license": "CUDA Toolkit",
        "license_path": "cuda_cudart/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99",
            "md5": "6153e2b6a43389e5be3d68b04c6488f5",
            "size": "1514676"
        },
        "linux-sbsa": {
            "relative_path": "cuda_cudart/linux-sbsa/cuda_cudart-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "8b422a3b2cb8452cb678181b0bf9d7aa7342df168b5319c5488ae3b8514101fc",
            "md5": "c4b3e0d206f5dc3aceaddb78a32424d3",
            "size": "1521596"
        },
        "windows-x86_64": {
            "relative_path": "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip",
            "sha256": "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8",
            "md5": "22e500ce68dc9e099ec18f57e1309808",
            "size": "3521238"
        },
        "linux-aarch64": {
            "relative_path": "cuda_cudart/linux-aarch64/cuda_cudart-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "f893e2c72be81b0d0e3bc33827c785a96db15fd5aafdc51cc187f3df6fdbb657",
            "md5": "8970e9d40a7edada8cd9270906b1daa3",
            "size": "1223440"
        }
    },
    "cuda_cuobjdump": {
        "name": "cuobjdump",
        "license": "CUDA Toolkit",
        "license_path": "cuda_cuobjdump/LICENSE.txt",
        "version": "12.9.82",
        "linux-x86_64": {
            "relative_path": "cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.9.82-archive.tar.xz",
            "sha256": "ee0de40e8c18068bfcc53e73510e7e7a1a80555205347940df67fa525d24452f",
            "md5": "fb4ceb446f3ca58c4ca5381a2c0eee06",
            "size": "214896"
        },
        "linux-sbsa": {
            "relative_path": "cuda_cuobjdump/linux-sbsa/cuda_cuobjdump-linux-sbsa-12.9.82-archive.tar.xz",
            "sha256": "2d39ae9309f81a7980df296a93c0e49215c6fb22d316fd9daae211d29a731d27",
            "md5": "9ec739cea851b184b55017d0a6185304",
            "size": "208936"
        },
        "windows-x86_64": {
            "relative_path": "cuda_cuobjdump/windows-x86_64/cuda_cuobjdump-windows-x86_64-12.9.82-archive.zip",
            "sha256": "1eda43a76a2eac25fce5bdb4b68673b5bda737d54cca5513148c36362ab7c811",
            "md5": "0a1c8b1f69537bc605c0d5fc28110258",
            "size": "6219088"
        },
        "linux-aarch64": {
            "relative_path": "cuda_cuobjdump/linux-aarch64/cuda_cuobjdump-linux-aarch64-12.9.82-archive.tar.xz",
            "sha256": "91a9ae2935b411d136c2c61ccbbc0df3a83f17cce122d12d262be202217163ee",
            "md5": "ec5905b13be30f8f39e18d1f41dbcf18",
            "size": "197604"
        }
    },
    "cuda_cupti": {
        "name": "CUPTI",
        "license": "CUDA Toolkit",
        "license_path": "cuda_cupti/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_cupti/linux-x86_64/cuda_cupti-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "f779a24e8f0177b4ce45cbf118cb470139fb5107858df96689d0a0aa01f0fba1",
            "md5": "5fe49729261d1f899bd0948026b755ae",
            "size": "16799436"
        },
        "linux-sbsa": {
            "relative_path": "cuda_cupti/linux-sbsa/cuda_cupti-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "84f8657375c23a425b2d107c3cde1ce75777bff9798b744b78e146fdc02d2de0",
            "md5": "ff4c04b183cc173aa2937a21a3569b5b",
            "size": "12805704"
        },
        "windows-x86_64": {
            "relative_path": "cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
            "sha256": "c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa",
            "md5": "1cecfef05160d4e6b0f2549534fe7973",
            "size": "13359912"
        },
        "linux-aarch64": {
            "relative_path": "cuda_cupti/linux-aarch64/cuda_cupti-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "74a884691b2c7e5c8db8bd2a38143fc82f7502f0dd484051809df7953f7ae719",
            "md5": "7f92adb9aed6fdb047ebe3c7617c7179",
            "size": "4781260"
        }
    },
    "cuda_cuxxfilt": {
        "name": "CUDA cuxxfilt (demangler)",
        "license": "CUDA Toolkit",
        "license_path": "cuda_cuxxfilt/LICENSE.txt",
        "version": "12.9.82",
        "linux-x86_64": {
            "relative_path": "cuda_cuxxfilt/linux-x86_64/cuda_cuxxfilt-linux-x86_64-12.9.82-archive.tar.xz",
            "sha256": "833d7e56351d032717f217212577d369d230e284b2ded4bf151403cc11213add",
            "md5": "a7f4a446e4fb00657f5bd7eb1dd814a7",
            "size": "188692"
        },
        "linux-sbsa": {
            "relative_path": "cuda_cuxxfilt/linux-sbsa/cuda_cuxxfilt-linux-sbsa-12.9.82-archive.tar.xz",
            "sha256": "2175ef96b523eb8fea5d6ccfd21b70d6afaee623d00dbab70f11e8aa892c2828",
            "md5": "fde8b2db7d7e4afb08e7c703fa735440",
            "size": "177440"
        },
        "windows-x86_64": {
            "relative_path": "cuda_cuxxfilt/windows-x86_64/cuda_cuxxfilt-windows-x86_64-12.9.82-archive.zip",
            "sha256": "ec5911d680394d90968c480c7359e7f5a4025b9c4806dad673d489e57585afd3",
            "md5": "537cd82a0945fe20275d4791e353c3f5",
            "size": "178570"
        },
        "linux-aarch64": {
            "relative_path": "cuda_cuxxfilt/linux-aarch64/cuda_cuxxfilt-linux-aarch64-12.9.82-archive.tar.xz",
            "sha256": "37acae6576874c3781e844ea268a9fe076d13aaab3acda6f65bbe885a3726a3c",
            "md5": "c8e23bff8cc13fb58a57c90d69e7b374",
            "size": "170436"
        }
    },
    "cuda_demo_suite": {
        "name": "CUDA Demo Suite",
        "license": "CUDA Toolkit",
        "license_path": "cuda_demo_suite/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_demo_suite/linux-x86_64/cuda_demo_suite-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "92e0b324bf02d9705b8dd5bf1d36149c06512c720ceda7cc7d791fc3ecf64f0e",
            "md5": "5030cddcab7101704bb8a9b53c6137ad",
            "size": "4036604"
        },
        "windows-x86_64": {
            "relative_path": "cuda_demo_suite/windows-x86_64/cuda_demo_suite-windows-x86_64-12.9.79-archive.zip",
            "sha256": "adffd0b0f7cca262025f171dcaa73f62af020de5d9a0d6e0fbe7dbd827dd77a8",
            "md5": "9a8c3bbd696392e768ae3278e269cbd9",
            "size": "5164004"
        }
    },
    "cuda_documentation": {
        "name": "CUDA Documentation",
        "license": "CUDA Toolkit",
        "license_path": "cuda_documentation/LICENSE.txt",
        "version": "12.9.88",
        "linux-x86_64": {
            "relative_path": "cuda_documentation/linux-x86_64/cuda_documentation-linux-x86_64-12.9.88-archive.tar.xz",
            "sha256": "d6e44c454654016a3650f3101688231ba111cd00647b726b34d18ff0bd721933",
            "md5": "17436d82e589824bbf3f4ff669201d81",
            "size": "67920"
        },
        "linux-sbsa": {
            "relative_path": "cuda_documentation/linux-sbsa/cuda_documentation-linux-sbsa-12.9.88-archive.tar.xz",
            "sha256": "e1c47f32279a827baf8b4339c1737d7bd8e54c73388835e5fae3e2db83db2479",
            "md5": "995f5692be016e44821fb85ebc3a62d9",
            "size": "67924"
        },
        "windows-x86_64": {
            "relative_path": "cuda_documentation/windows-x86_64/cuda_documentation-windows-x86_64-12.9.88-archive.zip",
            "sha256": "660f2f71b94c13a021850a8bb6227dfb4eb9a8dd8e0f6bc9feb0b993dea180fd",
            "md5": "3d21d1b9056629557bb2745f962af636",
            "size": "107626"
        },
        "linux-aarch64": {
            "relative_path": "cuda_documentation/linux-aarch64/cuda_documentation-linux-aarch64-12.9.88-archive.tar.xz",
            "sha256": "830476a755f9a8c9780a6876d0ed562c073f29968c3e0e91962d0fd14b3f3618",
            "md5": "098d318ebb7700a6513e57e41edf20c8",
            "size": "67948"
        }
    },
    "cuda_gdb": {
        "name": "CUDA GDB",
        "license": "CUDA Toolkit",
        "license_path": "cuda_gdb/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_gdb/linux-x86_64/cuda_gdb-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "3dbb657683cd4f2989c30bdad31b3f4493795a9acc8a2214f48df3cfca7c37c0",
            "md5": "1405f9cf4a5d2c03785a69aefa05de16",
            "size": "67691620"
        },
        "linux-sbsa": {
            "relative_path": "cuda_gdb/linux-sbsa/cuda_gdb-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "251b9bd86ffcce0cdc4a5480ef3d8d609ca3a276cbbb5c793ca9978721d31faf",
            "md5": "b76ab824a5b651ecec31978e6e2f41bf",
            "size": "45695592"
        },
        "linux-aarch64": {
            "relative_path": "cuda_gdb/linux-aarch64/cuda_gdb-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "5473c27f1ec03fe064a09399e52b251e6ca5d3b9a67f21c1a26a02a26102f165",
            "md5": "95325479bec751d72d4f063d8d244049",
            "size": "45662012"
        }
    },
    "cuda_nsight": {
        "name": "Nsight Eclipse Edition Plugin",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nsight/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_nsight/linux-x86_64/cuda_nsight-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "5c3994440ce7aa6ac5ab120e45f25f5f5eaa631af0f793fab860a5edbc499518",
            "md5": "3bce6195222da1c1d549fbd2d83422e8",
            "size": "118693300"
        }
    },
    "cuda_nvcc": {
        "name": "CUDA NVCC",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvcc/LICENSE.txt",
        "version": "12.9.86",
        "linux-x86_64": {
            "relative_path": "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz",
            "sha256": "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35",
            "md5": "ac1871c955070267e117bd985f6dbb36",
            "size": "81100244"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvcc/linux-sbsa/cuda_nvcc-linux-sbsa-12.9.86-archive.tar.xz",
            "sha256": "0aa1fce92dbae76c059c27eefb9d0ffb58e1291151e44ff7c7f1fc2dd9376c0d",
            "md5": "1322a14e4d6482f69d4b8a14a7d5f4c5",
            "size": "73407012"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.9.86-archive.zip",
            "sha256": "227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85",
            "md5": "ff9b2942aaa3a1dfb487da6767bc689f",
            "size": "126917884"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvcc/linux-aarch64/cuda_nvcc-linux-aarch64-12.9.86-archive.tar.xz",
            "sha256": "2432ef8a7c12d0a9ce3332a8af42b123c07f256390b3390802b1b2c6254c6c74",
            "md5": "f846e79e53be949484a4968880eb9dd9",
            "size": "77162776"
        }
    },
    "cuda_nvdisasm": {
        "name": "CUDA nvdisasm",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvdisasm/LICENSE.txt",
        "version": "12.9.88",
        "linux-x86_64": {
            "relative_path": "cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.9.88-archive.tar.xz",
            "sha256": "49296dd550e05434185a8588ec639f1325b2de413e2321ddd7e56c5182a476ff",
            "md5": "87ec016e430618bf724a3e8332bfcca6",
            "size": "5483340"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvdisasm/linux-sbsa/cuda_nvdisasm-linux-sbsa-12.9.88-archive.tar.xz",
            "sha256": "28b2597f0901cfafcd050cba0877c1eb5edcd7ebd8164aea356cec832e636ee3",
            "md5": "0d6f0b8883e55ca07d64cbb746b80e8c",
            "size": "5408212"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvdisasm/windows-x86_64/cuda_nvdisasm-windows-x86_64-12.9.88-archive.zip",
            "sha256": "955ba1f52f7115031f10408ce3cec4c745df41dba8fdf6024c3983d899e9fbbc",
            "md5": "43cd76e59014be18d96ce03e2ed09d6b",
            "size": "5791897"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvdisasm/linux-aarch64/cuda_nvdisasm-linux-aarch64-12.9.88-archive.tar.xz",
            "sha256": "99bf70149423c42c8dc2649f76f8983101dcf9879d355cfb2582b2adbed84614",
            "md5": "cb9f7798d824ef1696fdde67f9506aa4",
            "size": "5416148"
        }
    },
    "cuda_nvml_dev": {
        "name": "CUDA NVML Headers",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvml_dev/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_nvml_dev/linux-x86_64/cuda_nvml_dev-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "1ad0866dbfff6a9e2661f5348e4ca4c2a4e40882b90014ab127f2734856ecccb",
            "md5": "7e615aa7b4525d3cc1b28379abcf6ce5",
            "size": "129244"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvml_dev/linux-sbsa/cuda_nvml_dev-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "e97e668ead7ebc1fb7e93ffe303019660d0119c4e4d0e8ef26ce012bbbea9b34",
            "md5": "d48bcaa83a7299b836d381b32e94a99d",
            "size": "130480"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvml_dev/windows-x86_64/cuda_nvml_dev-windows-x86_64-12.9.79-archive.zip",
            "sha256": "1894b70c5487a739c740929263fa3fbca80e53790647abc02b74eac024b97be8",
            "md5": "213d773bb297b15e9f697ec8f1363bd9",
            "size": "144955"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvml_dev/linux-aarch64/cuda_nvml_dev-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "531ac94b8b83209657cd14b28d23660df252393fb34c24f5d8919c56c977477e",
            "md5": "3fe82d3f6042a22ac3d119192577db79",
            "size": "130544"
        }
    },
    "cuda_nvprof": {
        "name": "CUDA nvprof",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvprof/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_nvprof/linux-x86_64/cuda_nvprof-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "8d8d1a9004710bad8d7a653769f088064b0285a06a80b46c4da7598115a0c6a2",
            "md5": "df8f768c6c21b06cb048b9a4b0ac39e7",
            "size": "2402440"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvprof/windows-x86_64/cuda_nvprof-windows-x86_64-12.9.79-archive.zip",
            "sha256": "81aefd12ab0a24f88feee10303b814fcf21887b3947d5e73523ed14338ef4e2b",
            "md5": "69ada7050609c5860a2396a6354665e0",
            "size": "1705346"
        }
    },
    "cuda_nvprune": {
        "name": "CUDA nvprune",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvprune/LICENSE.txt",
        "version": "12.9.82",
        "linux-x86_64": {
            "relative_path": "cuda_nvprune/linux-x86_64/cuda_nvprune-linux-x86_64-12.9.82-archive.tar.xz",
            "sha256": "a06f0e2959a4dd3dbb62a984dbe77b813397022596f5c62d74ddd83b238571f2",
            "md5": "db647ad5946c1e4cb0289c16e5ec92d8",
            "size": "59372"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvprune/linux-sbsa/cuda_nvprune-linux-sbsa-12.9.82-archive.tar.xz",
            "sha256": "15e1d6527bf04c162950251940b10b8b8254f68028e2ffc0bfb7ed84bb2e1382",
            "md5": "3d51536a3582379389d821afa05becfd",
            "size": "51052"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvprune/windows-x86_64/cuda_nvprune-windows-x86_64-12.9.82-archive.zip",
            "sha256": "be23018507f015ca948c503a43a3c48449c0dc1ceaab1e721caf01f024727312",
            "md5": "f5c1eaa20c7c8c67a564f04cec87fca2",
            "size": "142316"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvprune/linux-aarch64/cuda_nvprune-linux-aarch64-12.9.82-archive.tar.xz",
            "sha256": "69fa4cf2cede678825e0c8032ccd629e17de1b9d8667b05b6702d873fd2fb926",
            "md5": "00c370560287a3dcdeedb32fe13e7e0e",
            "size": "52876"
        }
    },
    "cuda_nvrtc": {
        "name": "CUDA NVRTC",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvrtc/LICENSE.txt",
        "version": "12.9.86",
        "linux-x86_64": {
            "relative_path": "cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz",
            "sha256": "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af",
            "md5": "8ed74b72c44ee50759f820dee875589c",
            "size": "114231976"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvrtc/linux-sbsa/cuda_nvrtc-linux-sbsa-12.9.86-archive.tar.xz",
            "sha256": "fb2d50c791465f333fc2236d2419170cf7a7886f48dd9b967a10f8233c686029",
            "md5": "483dc56df046c9c03eb17cd637bb8fc9",
            "size": "53265740"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.9.86-archive.zip",
            "sha256": "1aa0644fa53c8ca34cdc73db17bcc73530557bdd3f582c7bfdbd7916c8b48f65",
            "md5": "76317f5006ce9de6720cea0b3bdc7db6",
            "size": "314608748"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvrtc/linux-aarch64/cuda_nvrtc-linux-aarch64-12.9.86-archive.tar.xz",
            "sha256": "785b0952c0f77dd0feb9a674f3fcca3fcf05541bbe27f70a5926da9f85339152",
            "md5": "b49aeba8583248a4cd628154361d313d",
            "size": "57633224"
        }
    },
    "cuda_nvtx": {
        "name": "CUDA NVTX",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvtx/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "819bc39192955e6ba2067de39b85f30e157de462945e54b12bfdeda429d793fb",
            "md5": "3a56b3c5c1e20d99cff4c354e57ea30f",
            "size": "64496"
        },
        "linux-sbsa": {
            "relative_path": "cuda_nvtx/linux-sbsa/cuda_nvtx-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "dae359c2c51f83a5cd402468f481a82aeb6d32d79dc707d3625607e83cf97ceb",
            "md5": "d783ac2926973ad682c69270d047c3f3",
            "size": "64856"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvtx/windows-x86_64/cuda_nvtx-windows-x86_64-12.9.79-archive.zip",
            "sha256": "b9d506ce9ba056bf171b60e9dada06fb3d8bed5453a6399d0541960bf9b81659",
            "md5": "907cfcf3f3b637912ded30349ecfd515",
            "size": "87029"
        },
        "linux-aarch64": {
            "relative_path": "cuda_nvtx/linux-aarch64/cuda_nvtx-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "f1a4366e45dbfd0a4e6210a5a4b9b51501aaebb6b58845d3f1b271ec3d7f3e15",
            "md5": "6b40a46f73f46a7d8bac37b212989007",
            "size": "65396"
        }
    },
    "cuda_nvvp": {
        "name": "CUDA NVVP",
        "license": "CUDA Toolkit",
        "license_path": "cuda_nvvp/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_nvvp/linux-x86_64/cuda_nvvp-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "6cdf9196373a848856c1afc6f1df1023649fb5fe77b896967ecc0b014b200003",
            "md5": "0ded9b80cf8fc2c2233b28d582f2334e",
            "size": "117732648"
        },
        "windows-x86_64": {
            "relative_path": "cuda_nvvp/windows-x86_64/cuda_nvvp-windows-x86_64-12.9.79-archive.zip",
            "sha256": "a7e1ac0de34c69afc288283ea75314b67f109b08d046c3622405dff8f66b0720",
            "md5": "dd673788d3fae14c38af5c1506fae11d",
            "size": "120342251"
        }
    },
    "cuda_opencl": {
        "name": "CUDA OpenCL",
        "license": "CUDA Toolkit",
        "license_path": "cuda_opencl/LICENSE.txt",
        "version": "12.9.19",
        "linux-x86_64": {
            "relative_path": "cuda_opencl/linux-x86_64/cuda_opencl-linux-x86_64-12.9.19-archive.tar.xz",
            "sha256": "1bddacc2f0de77901faad5fecfad8fef8136ea8dd708e02177d10eef97d85b78",
            "md5": "d0a85b602afc8957374d3f536dfd0306",
            "size": "94140"
        },
        "windows-x86_64": {
            "relative_path": "cuda_opencl/windows-x86_64/cuda_opencl-windows-x86_64-12.9.19-archive.zip",
            "sha256": "7c78e05697a3f74510c4154b175ce4597ed0719c7793236ca7f3989eccae0ff6",
            "md5": "e2dec7d45f8d87045c4a37bb882e9cf4",
            "size": "139663"
        }
    },
    "cuda_profiler_api": {
        "name": "CUDA Profiler API",
        "license": "CUDA Toolkit",
        "license_path": "cuda_profiler_api/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "8c50636bfb97e9420905aa795b9fa6e3ad0b30ec6a6c8b0b8db519beb9241ce6",
            "md5": "483998e0e5b012d976bf62ada20bd4fa",
            "size": "16316"
        },
        "linux-sbsa": {
            "relative_path": "cuda_profiler_api/linux-sbsa/cuda_profiler_api-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "e07f47ef3aeb3a3ca995e9070d77d98ad79460216bf2075c9f9018962ae1d03b",
            "md5": "55345cd640f299f8da43ef946dd8aa89",
            "size": "16316"
        },
        "windows-x86_64": {
            "relative_path": "cuda_profiler_api/windows-x86_64/cuda_profiler_api-windows-x86_64-12.9.79-archive.zip",
            "sha256": "b05519f8ce4f02167bfc859358f62eb771d89ac2af3a6952a82317f2af4bc5bd",
            "md5": "f5da20e1160f547602ec1c558dd0bc64",
            "size": "20352"
        },
        "linux-aarch64": {
            "relative_path": "cuda_profiler_api/linux-aarch64/cuda_profiler_api-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "e26d82b4ce994dc445dc0191b1a5fe68dacc7290bfb9c2133b4979fe90e05433",
            "md5": "738fe1e26407ab88e0fd3209b4da1cfa",
            "size": "16308"
        }
    },
    "cuda_sandbox_dev": {
        "name": "CUDA nvsandboxutils Headers",
        "license": "CUDA Toolkit",
        "license_path": "cuda_sandbox_dev/LICENSE.txt",
        "version": "12.9.19",
        "linux-x86_64": {
            "relative_path": "cuda_sandbox_dev/linux-x86_64/cuda_sandbox_dev-linux-x86_64-12.9.19-archive.tar.xz",
            "sha256": "058c9616f9bb4e57c58996a053cf9a87f655c139dc2fa11af7bed74432bd8153",
            "md5": "ce4326bc03b26b88a552db8b163d7f5f",
            "size": "29172"
        }
    },
    "cuda_sanitizer_api": {
        "name": "CUDA Compute Sanitizer API",
        "license": "CUDA Toolkit",
        "license_path": "cuda_sanitizer_api/LICENSE.txt",
        "version": "12.9.79",
        "linux-x86_64": {
            "relative_path": "cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-12.9.79-archive.tar.xz",
            "sha256": "e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf",
            "md5": "9d138eb96c11e1b9293408e5b7a97114",
            "size": "9758036"
        },
        "linux-sbsa": {
            "relative_path": "cuda_sanitizer_api/linux-sbsa/cuda_sanitizer_api-linux-sbsa-12.9.79-archive.tar.xz",
            "sha256": "281538927b6818d4687fad102c0603ab7b389513c9c129f3e0de8c61ac7f474d",
            "md5": "3192ef7e50255f80430471ae6add0ebf",
            "size": "7753752"
        },
        "windows-x86_64": {
            "relative_path": "cuda_sanitizer_api/windows-x86_64/cuda_sanitizer_api-windows-x86_64-12.9.79-archive.zip",
            "sha256": "b1f366312cb52164dfe7b78463ab085b742f052e74b89e9da08561a8ca8b06e9",
            "md5": "1821743a8a46f3e4bab0fe96dac813f8",
            "size": "10103190"
        },
        "linux-aarch64": {
            "relative_path": "cuda_sanitizer_api/linux-aarch64/cuda_sanitizer_api-linux-aarch64-12.9.79-archive.tar.xz",
            "sha256": "2baaba3ff47eceb529ddcc866fc7e647fd02427ef3dbb8bec06689f919c14c69",
            "md5": "e48b00b6ed2124ac4a3a0b20201ba4dd",
            "size": "4808620"
        }
    },
    "driver_assistant": {
        "name": "NVIDIA Driver Assistant",
        "license": "MIT",
        "license_path": "driver_assistant/LICENSE.txt",
        "version": "0.21.57.08",
        "linux-all": {
            "relative_path": "driver_assistant/source/driver_assistant-0.21.57.08-archive.tar.xz",
            "sha256": "4904c08e1de1aa790d200f36528aaa672ab6b3c6620bbfe5e221312b9ef1120a",
            "md5": "ebb37f4ad248ddba8071c570f5267da3",
            "size": "38644"
        }
    },
    "fabricmanager": {
        "name": "NVIDIA Fabric Manager",
        "license": "NVIDIA Driver",
        "license_path": "fabricmanager/LICENSE.txt",
        "version": "575.57.08",
        "linux-x86_64": {
            "relative_path": "fabricmanager/linux-x86_64/fabricmanager-linux-x86_64-575.57.08-archive.tar.xz",
            "sha256": "2e7dc4db6788e618af345fb50dee63115b5116ccb57d6d27d93c1ff5e2c3a8cf",
            "md5": "39f4aa35a3f7cda55297bb73c91a5481",
            "size": "8077772"
        },
        "linux-sbsa": {
            "relative_path": "fabricmanager/linux-sbsa/fabricmanager-linux-sbsa-575.57.08-archive.tar.xz",
            "sha256": "239a7f8406987fb6d44cd4f7f6ba6e843b3f3acb7b1732e6ab5c0d370690ee85",
            "md5": "4d288febfed83ff3d532ebbc692f65ec",
            "size": "7336832"
        }
    },
    "imex": {
        "name": "Nvidia-Imex",
        "license": "NVIDIA Proprietary",
        "license_path": "imex/LICENSE.txt",
        "version": "575.57.08",
        "linux-x86_64": {
            "relative_path": "imex/linux-x86_64/nvidia-imex-linux-x86_64-575.57.08-archive.tar.xz",
            "sha256": "5dfe195429b0967788b7e9b4bab85d4936220d6d2f6cf394a61bd3439b437506",
            "md5": "909fd5da3b23f54e39484efd187f4c3e",
            "size": "7718644"
        },
        "linux-sbsa": {
            "relative_path": "imex/linux-sbsa/nvidia-imex-linux-sbsa-575.57.08-archive.tar.xz",
            "sha256": "36d1a49fb52294ef5352eeea457d131f3d14320fca0a56633d7798ab32eead18",
            "md5": "bd3d1ce8d91e1fd7d26191c05292698e",
            "size": "7149236"
        }
    },
    "libcublas": {
        "name": "CUDA cuBLAS",
        "license": "CUDA Toolkit",
        "license_path": "libcublas/LICENSE.txt",
        "version": "12.9.1.4",
        "linux-x86_64": {
            "relative_path": "libcublas/linux-x86_64/libcublas-linux-x86_64-12.9.1.4-archive.tar.xz",
            "sha256": "546addc4a9d82b8f23aa9ba9274b6bc0429a63008a31c759884ac24880796057",
            "md5": "9b07f0c3d94534e56e003c6016b9771c",
            "size": "933611504"
        },
        "linux-sbsa": {
            "relative_path": "libcublas/linux-sbsa/libcublas-linux-sbsa-12.9.1.4-archive.tar.xz",
            "sha256": "e99b074e6f66034e563508118804599d7579f73afc8424c55ad5fd8d12e085a5",
            "md5": "4c7cdc43283c0ba62e665b02e9ec7540",
            "size": "932528492"
        },
        "windows-x86_64": {
            "relative_path": "libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip",
            "sha256": "d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed",
            "md5": "3f092963b0767968fe6f02febc1b0cbf",
            "size": "549755186"
        },
        "linux-aarch64": {
            "relative_path": "libcublas/linux-aarch64/libcublas-linux-aarch64-12.9.1.4-archive.tar.xz",
            "sha256": "f75938d72153902cf08d1666311f35851e7eb6a16af743b2346bc68d7ba6b341",
            "md5": "15f6396431885647e961af811a90cc06",
            "size": "493899648"
        }
    },
    "libcudla": {
        "name": "cuDLA",
        "license": "CUDA Toolkit",
        "license_path": "libcudla/LICENSE.txt",
        "version": "12.9.19",
        "linux-aarch64": {
            "relative_path": "libcudla/linux-aarch64/libcudla-linux-aarch64-12.9.19-archive.tar.xz",
            "sha256": "ef7c1a05d9927c53aa089ec2217a1e999ffac0e776145d798bf784279fc79d40",
            "md5": "dd1fc18bd0fe3ed38b87a453d31575a0",
            "size": "38548"
        }
    },
    "libcufft": {
        "name": "CUDA cuFFT",
        "license": "CUDA Toolkit",
        "license_path": "libcufft/LICENSE.txt",
        "version": "11.4.1.4",
        "linux-x86_64": {
            "relative_path": "libcufft/linux-x86_64/libcufft-linux-x86_64-11.4.1.4-archive.tar.xz",
            "sha256": "b0e65af59b0c2f6c8ed9f5552a9b375890855b7926ae2c0404d15dcf2565bda4",
            "md5": "3ffadd77d5b084470aa8d9215d760e76",
            "size": "470942192"
        },
        "linux-sbsa": {
            "relative_path": "libcufft/linux-sbsa/libcufft-linux-sbsa-11.4.1.4-archive.tar.xz",
            "sha256": "b87637db96e485f4793d7ca8bd2cf07250eca5c86f6c56744a36683418359c03",
            "md5": "4a0906639fd293f529b8f949bc3ad9eb",
            "size": "471560648"
        },
        "windows-x86_64": {
            "relative_path": "libcufft/windows-x86_64/libcufft-windows-x86_64-11.4.1.4-archive.zip",
            "sha256": "f26f80bb9abff3269c548e1559e8c2b4ba58ccb8acc6095bbc6404fc962d4b80",
            "md5": "504ad2e94fd0d923b8a7082659813431",
            "size": "198361265"
        },
        "linux-aarch64": {
            "relative_path": "libcufft/linux-aarch64/libcufft-linux-aarch64-11.4.1.4-archive.tar.xz",
            "sha256": "5d992b98f0d3d294e339ed2f65a477a587803c1567598a120349fae52596bf20",
            "md5": "1f10a57d45e9078e940340cdeff26478",
            "size": "471529132"
        }
    },
    "libcufile": {
        "name": "CUDA cuFile",
        "license": "CUDA Toolkit",
        "license_path": "libcufile/LICENSE.txt",
        "version": "1.14.1.1",
        "linux-x86_64": {
            "relative_path": "libcufile/linux-x86_64/libcufile-linux-x86_64-1.14.1.1-archive.tar.xz",
            "sha256": "7ba9834b8dc2f8cdb1710a49f3de3f627bbcd4cc1f8a754019c66c8c80fdaee7",
            "md5": "e59833cbd8e423948fa5814a276315e3",
            "size": "42129088"
        },
        "linux-sbsa": {
            "relative_path": "libcufile/linux-sbsa/libcufile-linux-sbsa-1.14.1.1-archive.tar.xz",
            "sha256": "6b1d2a771bd822fdd06a6286eb59acba179b13fe063ae5b0de8fc0f4991a39d8",
            "md5": "5a8499cdc22754fe5c9335065d708ea4",
            "size": "41676328"
        },
        "linux-aarch64": {
            "relative_path": "libcufile/linux-aarch64/libcufile-linux-aarch64-1.14.1.1-archive.tar.xz",
            "sha256": "196b61a1bf02b85e76c21bdfe414a3f4db4380df41d9212c9eb6d0aa92eee1ce",
            "md5": "c7e269a01ecd2dc427941c5bc775c2ef",
            "size": "41655988"
        }
    },
    "libcurand": {
        "name": "CUDA cuRAND",
        "license": "CUDA Toolkit",
        "license_path": "libcurand/LICENSE.txt",
        "version": "10.3.10.19",
        "linux-x86_64": {
            "relative_path": "libcurand/linux-x86_64/libcurand-linux-x86_64-10.3.10.19-archive.tar.xz",
            "sha256": "48281b4caadb1cf790d44ac76b23c77d06f474c0b1799814f314aafec9258ad6",
            "md5": "e2fd0b8dd197f3bccc0ee19e5640fc93",
            "size": "89582256"
        },
        "linux-sbsa": {
            "relative_path": "libcurand/linux-sbsa/libcurand-linux-sbsa-10.3.10.19-archive.tar.xz",
            "sha256": "078afec842c99b3a953d62cc76bd74afa2d883dc436e6d642e6440bb1e85eb8e",
            "md5": "fae5b6ce60678f452a2c760e8bf27563",
            "size": "89544732"
        },
        "windows-x86_64": {
            "relative_path": "libcurand/windows-x86_64/libcurand-windows-x86_64-10.3.10.19-archive.zip",
            "sha256": "d0411f0b8c07e90d0fb6e01bfa7a54c9cb80f2ddf67e4ded2d96a50e19aadad6",
            "md5": "6ccce2f4f1f6d9592c908fa1cb91536d",
            "size": "67904600"
        },
        "linux-aarch64": {
            "relative_path": "libcurand/linux-aarch64/libcurand-linux-aarch64-10.3.10.19-archive.tar.xz",
            "sha256": "e5087640b5c9cd8bc173efb6a21f8388b24da59511cc3d57e60afc5e05d14b50",
            "md5": "71ed8ff7a7e539c0be8ca4b699f2f3e1",
            "size": "79761424"
        }
    },
    "libcusolver": {
        "name": "CUDA cuSOLVER",
        "license": "CUDA Toolkit",
        "license_path": "libcusolver/LICENSE.txt",
        "version": "11.7.5.82",
        "linux-x86_64": {
            "relative_path": "libcusolver/linux-x86_64/libcusolver-linux-x86_64-11.7.5.82-archive.tar.xz",
            "sha256": "3d3b96f3087dbc43893a28691b172f31725b316d524f5a3c1e6837257c898d06",
            "md5": "efcc5d100623563d1a56dff59b45b65d",
            "size": "324393196"
        },
        "linux-sbsa": {
            "relative_path": "libcusolver/linux-sbsa/libcusolver-linux-sbsa-11.7.5.82-archive.tar.xz",
            "sha256": "db463593ffcbc78f542a7f1ef808da43bf742acae654d970d99a47289c2a83e5",
            "md5": "63113f462dc11289f996eec8bab1db74",
            "size": "323573816"
        },
        "windows-x86_64": {
            "relative_path": "libcusolver/windows-x86_64/libcusolver-windows-x86_64-11.7.5.82-archive.zip",
            "sha256": "e991d64a0bbe3e0bb69ce2adce000244288002b0341894729666d66adb9b4f25",
            "md5": "e2d79987f6e38f63cc64816f393a8e51",
            "size": "322216184"
        },
        "linux-aarch64": {
            "relative_path": "libcusolver/linux-aarch64/libcusolver-linux-aarch64-11.7.5.82-archive.tar.xz",
            "sha256": "45e2fefeee5797c0492d8a9ee26c3dbb7af0cdcefe6f1cd25bd586daa633d9a0",
            "md5": "e311d62bfc65c6afc29152f6e923f78c",
            "size": "113258352"
        }
    },
    "libcusparse": {
        "name": "CUDA cuSPARSE",
        "license": "CUDA Toolkit",
        "license_path": "libcusparse/LICENSE.txt",
        "version": "12.5.10.65",
        "linux-x86_64": {
            "relative_path": "libcusparse/linux-x86_64/libcusparse-linux-x86_64-12.5.10.65-archive.tar.xz",
            "sha256": "a83415dcd3e1183afe363d4740f9f0309cfe560c6c08016c2a61468304f4b848",
            "md5": "c7ed952d12dec0f50a4386e5d09203f6",
            "size": "398914580"
        },
        "linux-sbsa": {
            "relative_path": "libcusparse/linux-sbsa/libcusparse-linux-sbsa-12.5.10.65-archive.tar.xz",
            "sha256": "8d1c8a57ba3eaecc3f7c11e29fed275a2f3dca5cea51dd2a24d07ab5d9998583",
            "md5": "2d48c3aecca07be0a458a977350ce175",
            "size": "398606816"
        },
        "windows-x86_64": {
            "relative_path": "libcusparse/windows-x86_64/libcusparse-windows-x86_64-12.5.10.65-archive.zip",
            "sha256": "abb4bfc01198f82fbc956773ccb98c578c03027b6ad425e829355be0c0a11a4a",
            "md5": "33ce31c3d880596c31304c965dbfcdce",
            "size": "358114928"
        },
        "linux-aarch64": {
            "relative_path": "libcusparse/linux-aarch64/libcusparse-linux-aarch64-12.5.10.65-archive.tar.xz",
            "sha256": "b294b6cd0acf5e68078d2fe4d41d95a4073fed780805adca7774bce7cbbe5b65",
            "md5": "ce4d363d3affd40fe751f065ec2bd4cf",
            "size": "131075108"
        }
    },
    "libnpp": {
        "name": "CUDA NPP",
        "license": "CUDA Toolkit",
        "license_path": "libnpp/LICENSE.txt",
        "version": "12.4.1.87",
        "linux-x86_64": {
            "relative_path": "libnpp/linux-x86_64/libnpp-linux-x86_64-12.4.1.87-archive.tar.xz",
            "sha256": "49351448d896854284ec708c14506eaaceb92aa01fbe35c91d5c52ad482e17ae",
            "md5": "03de546e34ad54d0119594bc237c5e87",
            "size": "331733872"
        },
        "linux-sbsa": {
            "relative_path": "libnpp/linux-sbsa/libnpp-linux-sbsa-12.4.1.87-archive.tar.xz",
            "sha256": "992d461905366cec4243b26ce7bfe997c0c0eabf53a001333025930a1a0c7237",
            "md5": "8de3be3d9043a07011b0de46c49ea668",
            "size": "330804724"
        },
        "windows-x86_64": {
            "relative_path": "libnpp/windows-x86_64/libnpp-windows-x86_64-12.4.1.87-archive.zip",
            "sha256": "cfcfbf59e4e5ce71113c058bd4eba3dd56e4db080932146d4047c0d44b4a558e",
            "md5": "7549e1191510ae5ca9291e5f13c70296",
            "size": "274109870"
        },
        "linux-aarch64": {
            "relative_path": "libnpp/linux-aarch64/libnpp-linux-aarch64-12.4.1.87-archive.tar.xz",
            "sha256": "50c9f80592eaf4f246c82475143768b848333d9f540782257d42023f5fe68fdf",
            "md5": "869efb32d5ab9468fda4cb94848cf90f",
            "size": "104647240"
        }
    },
    "libnvfatbin": {
        "name": "NVIDIA compiler library for fatbin interaction",
        "license": "CUDA Toolkit",
        "license_path": "libnvfatbin/LICENSE.txt",
        "version": "12.9.82",
        "linux-x86_64": {
            "relative_path": "libnvfatbin/linux-x86_64/libnvfatbin-linux-x86_64-12.9.82-archive.tar.xz",
            "sha256": "315be969a303437329bf72d7141babed024fc54f90a10aa748b03be8f826d57b",
            "md5": "d5e2dc19c01ab62dbf437a1ef80847b1",
            "size": "930820"
        },
        "linux-sbsa": {
            "relative_path": "libnvfatbin/linux-sbsa/libnvfatbin-linux-sbsa-12.9.82-archive.tar.xz",
            "sha256": "87bf71288ea7390d039b246fa794cad2ddd3b164ce9566f8542bb15039432cdb",
            "md5": "91902ce86f6f55a4ecc948f95bf1c032",
            "size": "836312"
        },
        "windows-x86_64": {
            "relative_path": "libnvfatbin/windows-x86_64/libnvfatbin-windows-x86_64-12.9.82-archive.zip",
            "sha256": "86563b25096bbc21d74b1c043701ec8596499f76b40e32f9ec9179fc10404d00",
            "md5": "c0963a07d7f2cbffd34b0b1ec6cae5c0",
            "size": "2217262"
        },
        "linux-aarch64": {
            "relative_path": "libnvfatbin/linux-aarch64/libnvfatbin-linux-aarch64-12.9.82-archive.tar.xz",
            "sha256": "248e77ede9afbd6060cda7de0475d222731ac021e22e696ce75ed72b426dfca9",
            "md5": "814b2ff13e944a07669580eb85c0e653",
            "size": "808780"
        }
    },
    "libnvidia_nscq": {
        "name": "NVIDIA NSCQ API",
        "license": "NVIDIA Driver",
        "license_path": "libnvidia_nscq/LICENSE.txt",
        "version": "575.57.08",
        "linux-x86_64": {
            "relative_path": "libnvidia_nscq/linux-x86_64/libnvidia_nscq-linux-x86_64-575.57.08-archive.tar.xz",
            "sha256": "7c54e959ee50212be8595e01ee76581c1f3a13c19b2279424b55a8d26385c41a",
            "md5": "b86a87c4f9fd58a79ced07cf82cec271",
            "size": "455504"
        },
        "linux-sbsa": {
            "relative_path": "libnvidia_nscq/linux-sbsa/libnvidia_nscq-linux-sbsa-575.57.08-archive.tar.xz",
            "sha256": "931004b8b2062249016a9bded2499f70331d27b265226ed879428b8bd7e4bb20",
            "md5": "790ca3dd29b03a71f32ad1dd05611a9e",
            "size": "446980"
        }
    },
    "libnvjitlink": {
        "name": "NVIDIA compiler library for JIT LTO functionality",
        "license": "CUDA Toolkit",
        "license_path": "libnvjitlink/LICENSE.txt",
        "version": "12.9.86",
        "linux-x86_64": {
            "relative_path": "libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-12.9.86-archive.tar.xz",
            "sha256": "392cac3144b52ba14900bc7259ea6405ae6da88a8c704eab9bbbcc9ba4824b07",
            "md5": "9156156146001536d1c8346783991d1b",
            "size": "54021940"
        },
        "linux-sbsa": {
            "relative_path": "libnvjitlink/linux-sbsa/libnvjitlink-linux-sbsa-12.9.86-archive.tar.xz",
            "sha256": "9c9227c1e9122fd8448cafced3b32bc69f40d3c041d25034ea23611a1262852f",
            "md5": "7fdf2864352ac613415327ebde38598b",
            "size": "50981988"
        },
        "windows-x86_64": {
            "relative_path": "libnvjitlink/windows-x86_64/libnvjitlink-windows-x86_64-12.9.86-archive.zip",
            "sha256": "ee7175da9628d47ccc92dce6d28d57ca77633e79079a2aee90e2a645edcd1384",
            "md5": "4cdb56f88316a5ade3edf69a1e7192e8",
            "size": "265666466"
        },
        "linux-aarch64": {
            "relative_path": "libnvjitlink/linux-aarch64/libnvjitlink-linux-aarch64-12.9.86-archive.tar.xz",
            "sha256": "a24842165d98660d4ba9fd753395f7c9834445552110fabce4baa4c211fd2c52",
            "md5": "548a35d56f08d07b1dc294a3c8172f71",
            "size": "54783780"
        }
    },
    "libnvjpeg": {
        "name": "CUDA nvJPEG",
        "license": "CUDA Toolkit",
        "license_path": "libnvjpeg/LICENSE.txt",
        "version": "12.4.0.76",
        "linux-x86_64": {
            "relative_path": "libnvjpeg/linux-x86_64/libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz",
            "sha256": "ddd8245b2803f5b55211261d7e5d7abf803c05f3b032238d0feaa6e09ea9401d",
            "md5": "b9f7806fb279c9cdd0b4a40e4171b87b",
            "size": "6378624"
        },
        "linux-sbsa": {
            "relative_path": "libnvjpeg/linux-sbsa/libnvjpeg-linux-sbsa-12.4.0.76-archive.tar.xz",
            "sha256": "405b5627ffd772d2837ae4ece123fdee841c34894dba2180a1c8e1b84c0f2665",
            "md5": "bad1da3ec8865dbd3955a757bb6b7ba0",
            "size": "6179324"
        },
        "windows-x86_64": {
            "relative_path": "libnvjpeg/windows-x86_64/libnvjpeg-windows-x86_64-12.4.0.76-archive.zip",
            "sha256": "b253241fc88bf30947b8ee068101aca8930960f113d8ee4a9583de021a79ffa1",
            "md5": "d19d13f9bd3dbc9c9361a3c1fee99dd5",
            "size": "4667529"
        },
        "linux-aarch64": {
            "relative_path": "libnvjpeg/linux-aarch64/libnvjpeg-linux-aarch64-12.4.0.76-archive.tar.xz",
            "sha256": "a9841bff40e577bec81352054e05c5d98720ae8e2ccfc4863eaac9dd35444c27",
            "md5": "0a7197330938f8b5363d38bdde63562c",
            "size": "1821640"
        }
    },
    "libnvsdm": {
        "name": "LIBNVSDM",
        "license": "NVIDIA",
        "license_path": "libnvsdm/LICENSE.txt",
        "version": "575.57.08",
        "linux-x86_64": {
            "relative_path": "libnvsdm/linux-x86_64/libnvsdm-linux-x86_64-575.57.08-archive.tar.xz",
            "sha256": "7a6ee934d5c328f9c05e967b63053c211f80485e0443ac119d887612a510ad53",
            "md5": "788f7c59de54096d84dc7f645964ad2a",
            "size": "500100"
        }
    },
    "mft": {
        "name": "NVLink 5 MFT",
        "license": "NVIDIA Proprietary",
        "license_path": "mft/LICENSE.txt",
        "version": "4.30.1.510",
        "linux-x86_64": {
            "relative_path": "mft/linux-x86_64/mft-linux-x86_64-4.30.1.510-archive.tar.xz",
            "sha256": "512714ab076d90d88550e60f5da65181363336cee94c67bff8821face8ef10b4",
            "md5": "30698cf8741826df5ca3d4d3bb3a982c",
            "size": "45536612"
        },
        "linux-sbsa": {
            "relative_path": "mft/linux-sbsa/mft-linux-sbsa-4.30.1.510-archive.tar.xz",
            "sha256": "95c05e1bc03ff81dffa58bfff26a1dd59c6300d6d5922662ae59c7af5788a310",
            "md5": "78ef93686e91302f7b751fed8f83d9be",
            "size": "44222004"
        }
    },
    "mft_autocomplete": {
        "name": "NVLink 5 MFT AUTOCOMPLETE",
        "license": "NVIDIA Proprietary",
        "license_path": "mft_autocomplete/LICENSE.txt",
        "version": "4.30.1.510",
        "linux-x86_64": {
            "relative_path": "mft_autocomplete/linux-x86_64/mft_autocomplete-linux-x86_64-4.30.1.510-archive.tar.xz",
            "sha256": "dc9069baa888c7ca0c3e55a3043d27b0100c29c81e6ed3346907dd49dabb1325",
            "md5": "b65cb40bacfe276b3444920e8eef1f2f",
            "size": "11884"
        },
        "linux-sbsa": {
            "relative_path": "mft_autocomplete/linux-sbsa/mft_autocomplete-linux-sbsa-4.30.1.510-archive.tar.xz",
            "sha256": "64647ce2a4de62e535ea7060d20d58c2613937071008c2a93894e1995facce2b",
            "md5": "bf034fbf9f887d62434f83dc9727db80",
            "size": "11880"
        }
    },
    "mft_oem": {
        "name": "NVLink 5 MFT OEM",
        "license": "NVIDIA Proprietary",
        "license_path": "mft_oem/LICENSE.txt",
        "version": "4.30.1.510",
        "linux-x86_64": {
            "relative_path": "mft_oem/linux-x86_64/mft_oem-linux-x86_64-4.30.1.510-archive.tar.xz",
            "sha256": "7a4729f8b91ba5c179820583189a4c95ffa9669312a28fe7ac03feebb726bbc3",
            "md5": "7329d76d0e5b976f88ef1f2484d962de",
            "size": "4708108"
        },
        "linux-sbsa": {
            "relative_path": "mft_oem/linux-sbsa/mft_oem-linux-sbsa-4.30.1.510-archive.tar.xz",
            "sha256": "b71bb03130b6dd3388a3e909cdf91cc9bca8b8098a8b80b5f376f5726e5b5e1c",
            "md5": "e55a4d898553c2cf92be57724a8204e1",
            "size": "4429188"
        }
    },
    "nsight_compute": {
        "name": "Nsight Compute",
        "license": "NVIDIA SLA",
        "license_path": "nsight_compute/LICENSE.txt",
        "version": "2025.2.1.3",
        "linux-x86_64": {
            "relative_path": "nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-2025.2.1.3-archive.tar.xz",
            "sha256": "02ab8867197aaf6a6ae3171293d70b6a9ddb20296be94ff4287741338cc2e1df",
            "md5": "e81faf319af6b21f241bae535d3d1907",
            "size": "300347320"
        },
        "linux-sbsa": {
            "relative_path": "nsight_compute/linux-sbsa/nsight_compute-linux-sbsa-2025.2.1.3-archive.tar.xz",
            "sha256": "0e336a5139f76778b8ad70fdb49fb43817ec1bb3ee6e7425d59d99d8c455d976",
            "md5": "18334b109fe6f10fdba2c2681726ba96",
            "size": "107009604"
        },
        "windows-x86_64": {
            "relative_path": "nsight_compute/windows-x86_64/nsight_compute-windows-x86_64-2025.2.1.3-archive.zip",
            "sha256": "e9d558654c98d83049969d133b98922b53ab8f4e3ba9e0a37bdb5e2ff300b7de",
            "md5": "7cb9cff3e0718212866c7fbb4365dee5",
            "size": "341265370"
        },
        "linux-aarch64": {
            "relative_path": "nsight_compute/linux-aarch64/nsight_compute-linux-aarch64-2025.2.1.3-archive.tar.xz",
            "sha256": "51c7762110c34728acd37878d4340f13612401fa622be0031b1a9e2ce112cfb2",
            "md5": "355babae2f49f3f2c7ec1cf1c0b07966",
            "size": "221171904"
        }
    },
    "nsight_systems": {
        "name": "Nsight Systems",
        "license": "NVIDIA SLA",
        "license_path": "nsight_systems/LICENSE.txt",
        "version": "2025.1.3.140",
        "linux-x86_64": {
            "relative_path": "nsight_systems/linux-x86_64/nsight_systems-linux-x86_64-2025.1.3.140-archive.tar.xz",
            "sha256": "dded4227619340307be0ba5bc4e23bcbc966e2df3763170ebb20410c2b54754e",
            "md5": "0b16c35f6e63a4fc33d14795e65bcd61",
            "size": "1075732108"
        },
        "linux-sbsa": {
            "relative_path": "nsight_systems/linux-sbsa/nsight_systems-linux-sbsa-2025.1.3.140-archive.tar.xz",
            "sha256": "479c46de1c459f7c760ef1ea5e5bbe61bfe7f4ba525ce4778d5ba25d874b8e1f",
            "md5": "6d70406b53a94eba94d7c3552b938fe6",
            "size": "980282844"
        },
        "windows-x86_64": {
            "relative_path": "nsight_systems/windows-x86_64/nsight_systems-windows-x86_64-2025.1.3.140-archive.zip",
            "sha256": "8d61f266a8a1bc1ababd44de3ee38f7b85c8cdfcdaebace14c8f065d7624d0b3",
            "md5": "b741046881531cbf6e3ccedb0b403432",
            "size": "403084779"
        }
    },
    "nsight_vse": {
        "name": "Nsight Visual Studio Edition (VSE)",
        "license": "NVIDIA SLA",
        "license_path": "nsight_vse/LICENSE.txt",
        "version": "2025.2.1.25125",
        "windows-x86_64": {
            "relative_path": "nsight_vse/windows-x86_64/nsight_vse-windows-x86_64-2025.2.1.25125-archive.zip",
            "sha256": "3f2b6ca8705929c97e0f931f0559921b110b9337cf11af5e017d0c95b5b31feb",
            "md5": "7e74f479c09aece6e5b46c97d0d4cc86",
            "size": "136237151"
        }
    },
    "nvidia_driver": {
        "name": "NVIDIA Linux Driver",
        "license": "NVIDIA Driver",
        "license_path": "nvidia_driver/LICENSE.txt",
        "version": "575.57.08",
        "linux-x86_64": {
            "relative_path": "nvidia_driver/linux-x86_64/nvidia_driver-linux-x86_64-575.57.08-archive.tar.xz",
            "sha256": "27ddfabde120a107527cbf88a6c96d1f81ee5c977462ce1a793051a3e678f552",
            "md5": "9007dddd1e2d48f897958b676fbe30ff",
            "size": "471616744"
        },
        "linux-sbsa": {
            "relative_path": "nvidia_driver/linux-sbsa/nvidia_driver-linux-sbsa-575.57.08-archive.tar.xz",
            "sha256": "4a7d1accb257f2539ff7dabb02319339fc3a8db622073e5dde33a5446910346a",
            "md5": "3c2b686f392789d8f3a1db1ae2ad62be",
            "size": "349127420"
        }
    },
    "nvidia_fs": {
        "name": "NVIDIA filesystem",
        "license": "CUDA Toolkit",
        "license_path": "nvidia_fs/LICENSE.txt",
        "version": "2.25.7",
        "linux-x86_64": {
            "relative_path": "nvidia_fs/linux-x86_64/nvidia_fs-linux-x86_64-2.25.7-archive.tar.xz",
            "sha256": "6e11ec0b885177ab21f4b864ff7417ea3cc1b0994f1c318767984557cd5b29d1",
            "md5": "60d2fa54d55b739302e3db38a8a389c9",
            "size": "60280"
        },
        "linux-sbsa": {
            "relative_path": "nvidia_fs/linux-sbsa/nvidia_fs-linux-sbsa-2.25.7-archive.tar.xz",
            "sha256": "62baf0beac2d446c0ff033fb3780cf348ab9c090d7903c210106dc98bb29d7eb",
            "md5": "3859bed4b5e330cde0e2316f9cdcdaeb",
            "size": "60296"
        },
        "linux-aarch64": {
            "relative_path": "nvidia_fs/linux-aarch64/nvidia_fs-linux-aarch64-2.25.7-archive.tar.xz",
            "sha256": "b8c67e28e9e23453183fb7d12ed9fed449427b2c11a21b1de4e7046a57d8d366",
            "md5": "5128c63c80c1089340b65807445d921a",
            "size": "60272"
        }
    },
    "nvlsm": {
        "name": "NVLSM SM component",
        "license": "NVIDIA Proprietary",
        "license_path": "nvlsm/LICENSE.txt",
        "version": "2025.03.1",
        "linux-x86_64": {
            "relative_path": "nvlsm/linux-x86_64/nvlsm-linux-x86_64-2025.03.1-archive.tar.xz",
            "sha256": "b034dad10a3154359e244b85206cd73f0fbce8e1cdf76058417b7b562c337388",
            "md5": "6e09669fe8e4fb26e5334a1c4aa9dabb",
            "size": "6890300"
        },
        "linux-sbsa": {
            "relative_path": "nvlsm/linux-sbsa/nvlsm-linux-sbsa-2025.03.1-archive.tar.xz",
            "sha256": "e260285ec01c6beb562a14625e9564b96374bb824ec62cc9866066a48710fa54",
            "md5": "bc90989f567c27d8b39e607f719acba8",
            "size": "6175832"
        }
    },
    "visual_studio_integration": {
        "name": "CUDA Visual Studio Integration",
        "license": "CUDA Toolkit",
        "license_path": "visual_studio_integration/LICENSE.txt",
        "version": "12.9.79",
        "windows-x86_64": {
            "relative_path": "visual_studio_integration/windows-x86_64/visual_studio_integration-windows-x86_64-12.9.79-archive.zip",
            "sha256": "d5b6514866395f52f04107da1aa523954f8dc76daf18a9f8699c3d564f294449",
            "md5": "680fbdf0f345619c1fa3de5f1fc79090",
            "size": "864204"
        }
    }
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-loaded-sass-design-20261020/primary-abi-v2/frozen/source-acquisition.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5207; SHA256 `50de8543bcb6c7b30e19285efcf632d6daf71332ec63075e1afabeb6bfc5e89a`.

````text
{
  "scope": "exact_NVIDIA_12.9.79_source_entries_and_ZIP_metadata_only",
  "archive_url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
  "archive_published_bytes": 13359912,
  "archive_published_sha256": "c96018456dc4db6af6d69d6db0170a2ccc656ddde8c8ce6ee05682c2c5569daa",
  "whole_archive_fetched_or_hash_verified": false,
  "library_or_tool_entries_fetched": false,
  "requests": [
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json",
      "range_start": null,
      "range_end": null,
      "bytes": 54256,
      "sha256": "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630",
      "etag": "\"ae61f89f456975e43815b8ee514d8e6f:1749159793.986914\"",
      "last_modified": "Thu, 05 Jun 2025 19:49:45 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 13359890,
      "range_end": 13359911,
      "bytes": 22,
      "sha256": "b898efeeeb4d13e895a6ebb77f791a2bba4a9495cbc9180d5ad5c53200fee4f2",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 13276221,
      "range_end": 13359889,
      "bytes": 83669,
      "sha256": "6ef34da41a920fdc6d0158b4ac0e4f3b90d7481a43e958ea2e0daf960d576b16",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 72,
      "range_end": 101,
      "bytes": 30,
      "sha256": "b8ff547154422a85f14ae4c712fb8b84596d36a46cc9f2f7e00475048a8ccdd4",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 102,
      "range_end": 150,
      "bytes": 49,
      "sha256": "c0376ae951c91e13f135945ea6d37ab72334a0b5fd44a18f2f22ec8f47c5a82d",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 151,
      "range_end": 14972,
      "bytes": 14822,
      "sha256": "d918c0b30a33682af30e0bd38a1a16968fd25f8b0e4ba74374e07adbdf1d5b72",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 13265812,
      "range_end": 13265841,
      "bytes": 30,
      "sha256": "33236d97b731904e3d8dae8663dc67e1c619200c3d92707c6c5fc719137bea9b",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 13265842,
      "range_end": 13265925,
      "bytes": 84,
      "sha256": "d05a4bca0b59c7440a94cb1ce480a07095914c7d69fec41a3b59c0dfbd0e64a6",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    },
    {
      "url": "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.9.79-archive.zip",
      "range_start": 13265926,
      "range_end": 13267936,
      "bytes": 2011,
      "sha256": "5b8942919d634cfc36b83bd0df4aa8c8b36de914ae699dffb94b0b3a5f8a108c",
      "etag": "\"1cecfef05160d4e6b0f2549534fe7973:1749159801.636538\"",
      "last_modified": "Sat, 31 May 2025 21:24:34 GMT"
    }
  ],
  "selected_source_entries": [
    {
      "name": "cuda_cupti-windows-x86_64-12.9.79-archive/LICENSE",
      "flags": 0,
      "method": 8,
      "crc32": 1292129296,
      "compressed": 14822,
      "bytes": 63522,
      "local_header": 72,
      "source_sha256": "213f2f1cc4bfecd53eb1579edc89d6e3a024ddd2f77413b69b40e2bf3c114d84"
    },
    {
      "name": "cuda_cupti-windows-x86_64-12.9.79-archive/samples/sass_source_map/sass_source_map.cu",
      "flags": 0,
      "method": 8,
      "crc32": 3901770132,
      "compressed": 2011,
      "bytes": 6152,
      "local_header": 13265812,
      "source_sha256": "4fff611da3ae8e1e9d886dcd1d79229c8a752b8403a6e0b46a03ecb52854940e"
    }
  ],
  "total_HTTP_body_bytes": 154973,
  "runtime_or_build_executed": false
}

````

