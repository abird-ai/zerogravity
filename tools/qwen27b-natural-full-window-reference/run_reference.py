"""ROOT-only bounded public CPU capture; no model construction on import."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import uuid
from contract import HERE,IMAGE,IMAGE_DIGEST,require,metadata,write,sha,digest,fingerprint,fd_sha,package,specs

MANAGED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)
_LIVE = {}


class _Hold:
    """Record managed signals only; mandatory closes must finish."""
    def __init__(self, state):
        self.state = state
    def __enter__(self):
        self.state['depth'] = self.state.get('depth', 0)+1
        self.state['critical'] = True
        return self
    def __exit__(self, kind, value, traceback):
        self.state['depth'] = max(0, self.state.get('depth', 1)-1)
        self.state['critical'] = self.state['depth'] > 0
        return False


class _Ownership(_Hold):
    """Record through child/resource ownership; check pending once it is held."""
    def __exit__(self, kind, value, traceback):
        super().__exit__(kind, value, traceback)
        if kind is None:
            checkpoint(self.state)
        return False


def checkpoint(state):
    pending = state['pending']
    if pending:
        sig = pending.pop(0)
        if sig == signal.SIGALRM:
            raise TimeoutError('bounded whole source/capture/lifecycle deadline')
        raise InterruptedError('capture cancelled; cleanup owned child only')


def install(state):
    previous = {}
    def managed(sig, _frame):
        if state['critical']:
            state['pending'].append(sig)
            return
        if sig == signal.SIGALRM:
            raise TimeoutError('bounded whole source/capture/lifecycle deadline')
        state['pending'].append(sig)
    for sig in MANAGED:
        previous[sig] = signal.signal(sig, managed)
    state['previous'] = previous


def restore(state):
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig, previous in state['previous'].items():
        try:signal.signal(sig, previous)
        except BaseException:pass


def source_open(plan,pins,state,fds=None):
    # Register into the caller-owned map before hashing; the caller's outer
    # finally owns every FD even if the return-to-assignment gap is interrupted.
    if fds is None:fds={}
    files=specs(plan,pins);rows=[]
    try:
        for name,pin in files.items():
            require(Path(name).name==name,'contained authentic source filename')
            fd=None
            with _Hold(state):
                fd=os.open(Path(plan['source_directory'])/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
                fds[name]=fd
            checkpoint(state)
            m=os.fstat(fd)
            require(stat.S_ISREG(m.st_mode)and m.st_size==pin['bytes']and fd_sha(fd,0,pin['bytes'])==pin['sha256']
                    and os.pread(fd,1,pin['bytes'])==b'' and fingerprint(m)==fingerprint(os.fstat(fd))
                    and fingerprint(os.stat(Path(plan['source_directory'])/name,follow_symlinks=False))==fingerprint(m),
                    'held complete original source/EOF/current path before math')
            rows.append({'name':name,'bytes':pin['bytes'],'sha256':pin['sha256'],'fd_identity':fingerprint(m)})
        return fds,rows
    except BaseException:
        # Close every partial FD independently, protected from managed signals.
        state['critical']=True
        with _Hold(state):
            for name,fd in list(fds.items()):
                try:os.close(fd)
                except BaseException:pass
                fds.pop(name,None)
        raise

def source_after(fds,rows,directory):
    for row in rows:
        fd=fds[row['name']]
        require(fingerprint(os.fstat(fd))==row['fd_identity']and fd_sha(fd,0,row['bytes'])==row['sha256']
                and os.pread(fd,1,row['bytes'])==b'' and fingerprint(os.fstat(fd))==row['fd_identity']
                and fingerprint(os.stat(Path(directory)/row['name'],follow_symlinks=False))==row['fd_identity'],
                'same whole original source/EOF/current path after owned completion')
    return rows.copy()

def environment(cpus,identity):return {'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2','ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_PACKAGE_SHA':digest(json.dumps(identity,sort_keys=True).encode()),'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE','PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}

def argv(helper,output,plan,env,name,nonce):
    args=helper.PODMAN+['run','--name',name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(output/'owned.cid'),'--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges','--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256','--ulimit','fsize=1342177280:1342177280','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
    for k,v in env.items():args+=['-e',k+'='+v]
    return args+['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro','-v',str(output)+':/output:rw','--entrypoint','python3',IMAGE,'/probe/capture.py']

def owned_helper():
    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d','reviewed original owned lifecycle helper')
    spec=importlib.util.spec_from_file_location('qwen27_full_owned',HERE/'owned_container_v4.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def output_path(path,plan):
    require(path.is_absolute()and not path.exists()and path.parent.is_dir(),'fresh absolute output with existing parent')
    actual=path.parent.resolve()/path.name
    for old in [HERE,Path(plan['source_directory'])]:require(actual!=old and old not in actual.parents and actual not in old.parents,'output disjoint from source/package')
    return actual

def _emergency_close():
    """Outer last resort: close every already-owned object, then restore."""
    state=_LIVE.get('state')
    if state is None:return
    old=state.get('critical')
    state['critical']=True
    try:
        child=_LIVE.get('child');log=_LIVE.get('log');helper=_LIVE.get('helper')
        if child is not None and not _LIVE.get('reaped') and helper is not None:
            try:helper.cleanup_group(child,_LIVE.get('receipt') or {})
            except BaseException:pass
        if log is not None:
            try:log.finish()
            except BaseException:pass
        elif child is not None and getattr(child,'stdout',None) is not None:
            try:child.stdout.close()
            except BaseException:pass
        owned=_LIVE.get('fds') or {}
        for name,fd in list(owned.items()):
            try:os.close(fd)
            except BaseException:pass
            owned.pop(name,None)
    finally:
        restore(state)
        state['critical']=bool(old)


def main():
    _LIVE.clear()
    try:
        return _main()
    finally:
        _emergency_close()


def _main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-output',type=Path,required=True);args=p.parse_args()
    state={'critical':False,'depth':0,'pending':[],'previous':{}}
    _LIVE['state']=state
    install(state)
    try:
        signal.alarm(10800)
        identity=package();plan=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');specs(plan,pins);output=output_path(args.new_output,plan);output.mkdir(mode=0o700)
        cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus);helper=owned_helper()
        nonce=uuid.uuid4().hex;name='zg-original27-natural-full-window-'+nonce
        r={'status':'UNQUALIFIED','package':identity,'expected_cpus':cpus,'owned_name':name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,'scope':plan['scope']};fds={};child=log=None;failure=None;started=time.monotonic();receipt_written=False
        _LIVE.update(receipt=r,fds=fds,helper=helper)
    except BaseException:
        restore(state)
        raise
    try:
        fds,rows=source_open(plan,pins,state,fds);r['source_before']=rows
        checkpoint(state)
        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15);require(inspected.returncode==0,'approved immutable image already exists; no pull')
        checkpoint(state)
        (output/'image-inspect.json').write_bytes(inspected.stdout);image,=json.loads(inspected.stdout);require(image['Id']==IMAGE.removeprefix('sha256:')and image['Digest']==IMAGE_DIGEST,'actual immutable image identity')
        env=environment(cpus,identity);command=argv(helper,output,plan,env,name,nonce);r['argv']=command;write(output/'prepared-invocation.json',r)
        try:
            checkpoint(state)
            original_drain=None
            try:
                with _Ownership(state):
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);r['owned_cli_pid']=child.pid
                    _LIVE['child']=child
                    log=helper.Log(child.stdout,output/'execution.log')
                    _LIVE['log']=log
                    original_drain=log.drain
                    def checked_drain(timeout):
                        original_drain(timeout)
                        checkpoint(state)
                    log.drain=checked_drain
                r['exit_code']=log.observe(child,7200)
            finally:
                if original_drain is not None:log.drain=original_drain
            checkpoint(state)
        except BaseException as e:failure=e
        finally:
            r['cleanup_errors']=[]
            state['critical']=True
            with _Hold(state):
                if child is not None:
                    try:r['cleanup_errors']=helper.cleanup(child,name,nonce,output/'owned.cid',r)
                    except BaseException as e:r['cleanup_errors'].append('owner:'+repr(e))
                    finally:_LIVE['child']=None
                if log is not None:
                    try:log.finish()
                    except BaseException as e:r['cleanup_errors'].append('log:'+repr(e))
                    finally:
                        r['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
                        _LIVE['log']=None
                elif child is not None and child.stdout is not None:
                    try:child.stdout.close()
                    except BaseException as e:r['cleanup_errors'].append('stdout:'+repr(e))
            try:checkpoint(state)
            except BaseException as e:r['deadline_error']=repr(e);failure=failure or e
        checkpoint(state)
        require(failure is None and r.get('exit_code')==0 and r['cleanup_errors']==[],'actual owned original capture/cleanup healthy')
        r['source_after']=source_after(fds,rows,plan['source_directory']);require(package()==identity,'same source package after owned retirement');r['status']='EXECUTED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION'
    except BaseException as e:failure=failure or e
    finally:
        # The fallback is established before collection/EOF/notes: an ALRM at
        # any later point still closes caller-registered FDs and restores.
        try:
            errors=[]
            state['critical']=True
            with _Hold(state):
                for name,fd in list(fds.items()):
                    try:os.close(fd)
                    except BaseException as e:errors.append(repr(e))
                    fds.pop(name,None)
                try:checkpoint(state)
                except BaseException as e:r['deadline_error']=repr(e);failure=failure or e
            r['source_FD_cleanup_errors']=errors
            r['source_FDs_closed']=not errors
            r['elapsed_seconds_not_benchmark']=time.monotonic()-started
            if failure is not None or errors:r.update(status='FAIL_UNQUALIFIED',failure=repr(failure))
            try:
                signal.setitimer(signal.ITIMER_REAL,5)
                write(output/'invocation.json',r)
                receipt_written=True
            except BaseException as e:
                r['receipt_error']=repr(e);r['status']='FAIL_UNQUALIFIED'
        except BaseException as e:
            failure=failure or e
            r['status']='FAIL_UNQUALIFIED'
            try:
                signal.setitimer(signal.ITIMER_REAL,5)
                write(output/'invocation.json',r)
                receipt_written=True
            except BaseException as receipt_error:
                r['receipt_error']=repr(receipt_error)
        finally:
            # Independent close of every already-owned FD, then handlers/timer.
            state['critical']=True
            with _Hold(state):
                for fd in fds.values():
                    try:os.close(fd)
                    except BaseException:pass
            restore(state)
    require(receipt_written and r['status']=='EXECUTED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION','capture refused; attempt preserved')
if __name__=='__main__':main()
