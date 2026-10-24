"""Explicit ROOT-owned original CPU reference; no work/model import on import."""
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
from contract import validate
from support import HERE, IMAGE, IMAGE_DIGEST, metadata, write, sha, fd_sha, fingerprint, require, read

HELPER_SHA='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'

def tool_identity():
    return {p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and
            p.suffix in ('.py','.json')}

def source_open(directory, pins):
    handles={}; rows=[]
    try:
        for row in pins['files']:
            name=row['name']; require(Path(name).name==name, 'contained actual source member')
            fd=os.open(directory/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            handles[name]=fd; before=os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
                    'full authentic held original source before invocation')
            rows.append(dict(row,fd_identity=fingerprint(before)))
        return handles,rows
    except BaseException:
        for fd in handles.values():os.close(fd)
        raise

def source_after(directory, handles, rows):
    for row in rows:
        fd=handles[row['name']]
        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
                fingerprint(os.stat(directory/row['name'],follow_symlinks=False))==row['fd_identity'] and
                fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(os.fstat(fd))==row['fd_identity'] and fingerprint(os.stat(directory/row['name'],follow_symlinks=False))==row['fd_identity'], 'same source owner/path/full SHA after original final hash')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--new-output',required=True,type=Path)
    args=parser.parse_args()
    require(args.source.is_absolute() and args.new_output.is_absolute() and
            not args.new_output.exists(), 'explicit existing source and fresh output')
    source=args.source.resolve(strict=True); parent=args.new_output.parent.resolve(strict=True)
    require(not parent.is_relative_to(source) and not parent.is_relative_to(HERE), 'output outside original source/tooling')
    dest=parent/args.new_output.name; dest.mkdir(mode=0o700)
    def stop(signum,_frame):raise InterruptedError('original reference stopped; owned cleanup remains mandatory')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    pins=metadata(HERE/'source-pin.json'); identity=tool_identity()
    require(sha(HERE/'owned_container_v4.py')==HELPER_SHA, 'unchanged reviewed owned lifecycle')
    spec=importlib.util.spec_from_file_location('gemma4_owned',HERE/'owned_container_v4.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    cpus=sorted(os.sched_getaffinity(0))[:4]; require(len(cpus)==4,'four available CPUs')
    os.sched_setaffinity(0,cpus)
    nonce=uuid.uuid4().hex; name='zg-original-gemma4-text-'+nonce; cid=dest/'owned.cid'
    receipt={'status':'FAIL_UNQUALIFIED','image_id':IMAGE,'image_digest':IMAGE_DIGEST,
             'source':pins,'source_directory':str(source),'tool_files':identity,'expected_cpus':cpus,
             'owned_name':name,'owned_nonce':nonce,'trained_rust_qualified':False,'whole_ordinary_matrix':True}
    handles={}; child=log=None; error=None; start=time.monotonic()
    try:
        handles,before=source_open(source,pins);receipt['source_before']=before
        result=owner.command(owner.PODMAN+['image','inspect',IMAGE],15)
        require(result.returncode==0, 'approved existing offline image; no pull')
        (dest/'image-inspect.json').write_bytes(result.stdout)
        image,=json.loads(result.stdout)
        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'exact original image')
        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
             'OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE','ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'PYTHONPATH':'/probe',
             'PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
        argv=owner.PODMAN+['run','--name',name,'--label',owner.LABEL+'='+nonce,'--cidfile',str(cid),
             '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
             '--memory=12g','--memory-swap=12g','--cpus=4',
             '--pids-limit=256','--ulimit','fsize=33554432:33554432','--tmpfs','/tmp:rw,nosuid,nodev,size=256m']
        for key,value in env.items():argv+=['-e',key+'='+value]
        argv+=['-v',str(HERE)+':/probe:ro','-v',str(source)+':/source:ro',
               '-v',str(dest)+':/output:rw','--entrypoint','python3',IMAGE,'/probe/capture.py']
        receipt['argv']=argv
        write(dest/'invocation.prepared.json',receipt)
        try:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['owned_cli_pid']=child.pid;log=owner.Log(child.stdout,dest/'execution.log')
            receipt['exit_code']=log.observe(child,7200)
        except BaseException as failure:error=failure
        finally:
            receipt['cleanup_errors']=owner.cleanup(child,name,nonce,cid,receipt) if child is not None else []
            if log is not None:
                try:log.finish()
                except BaseException as failure:receipt['cleanup_errors'].append(repr(failure))
                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
            elif child is not None and child.stdout is not None:child.stdout.close()
        require(error is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[], 'actual original/owned closure failed; preserve attempt')
        capture=metadata(dest/'capture.json',1024*1024)
        def cell_read(pin):
            require(Path(pin['name']).name==pin['name']and pin['bytes']<=1024*1024,'bounded contained source cell')
            return metadata(dest/pin['name'],1024*1024,pin)
        def raw_read(name,at,n):
            require(Path(name).name==name and type(at)is int and type(n)is int and 0<=at and 0<n<=32*1024*1024,'bounded contained raw source read')
            fd=os.open(dest/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            try:
                before=fingerprint(os.fstat(fd));require(at+n<=before[2]<=32*1024*1024,'whole source archive extent')
                raw=os.pread(fd,n,at);require(len(raw)==n and fingerprint(os.fstat(fd))==before,'same raw FD after bounded read');return raw
            finally:os.close(fd)
        validate(capture,pins,metadata(HERE/'request.json'),cell_read,raw_read)
        receipt['capture']={'bytes':(dest/'capture.json').stat().st_size,'sha256':sha(dest/'capture.json')}
        require(tool_identity()==identity,'all original capture tools unchanged')
    except BaseException as failure:
        if error is None:error=failure
    finally:
        # Required even when model load/validation fails, once full source ownership began.
        if handles:
            try:receipt['source_after']=source_after(source,handles,receipt['source_before'])
            except BaseException as failure:
                if error is None:error=failure
        close_errors=[]
        for fd in handles.values():
            try:os.close(fd)
            except OSError as failure:close_errors.append(repr(failure))
        receipt['source_FDs_closed']=not close_errors;receipt['source_FD_cleanup_errors']=close_errors
        if error is None and not close_errors:receipt['status']='EXECUTED_ORIGINAL_GEMMA4_MATRIX_EXPECTED_ONLY'
        if error is not None:receipt['failure']=repr(error)
        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-start
        write(dest/'invocation.json',receipt)
    require(receipt['status']=='EXECUTED_ORIGINAL_GEMMA4_MATRIX_EXPECTED_ONLY','original attempt failed/unqualified')

if __name__=='__main__':main()
