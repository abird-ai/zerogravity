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
from contract import HERE,IMAGE,IMAGE_DIGEST,require,metadata,write,sha,digest,fingerprint,fd_sha,package,specs,scope

def source_open(plan,pins):
    files=specs(plan,pins);fds={};rows=[]
    try:
        for name,pin in files.items():
            require(Path(name).name==name,'contained authentic source filename');fd=os.open(Path(plan['source_directory'])/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);fds[name]=fd;m=os.fstat(fd)
            require(stat.S_ISREG(m.st_mode)and m.st_size==pin['bytes']and fd_sha(fd,0,pin['bytes'])==pin['sha256']
                    and os.pread(fd,1,pin['bytes'])==b'' and fingerprint(m)==fingerprint(os.fstat(fd))
                    and fingerprint(os.stat(Path(plan['source_directory'])/name,follow_symlinks=False))==fingerprint(m),
                    'held complete original source/EOF/current path before math')
            rows.append({'name':name,'bytes':pin['bytes'],'sha256':pin['sha256'],'fd_identity':fingerprint(m)})
        return fds,rows
    except BaseException:
        for fd in fds.values():os.close(fd)
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

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-output',type=Path,required=True);args=p.parse_args()
    def deadline(_s,_f):raise TimeoutError('bounded whole source/capture/lifecycle deadline')
    def stop(_s,_f):raise InterruptedError('capture cancelled; cleanup owned child only')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,stop);signal.alarm(10800)
    identity=package();plan=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');specs(plan,pins);output=output_path(args.new_output,plan);output.mkdir(mode=0o700)
    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus);helper=owned_helper()
    nonce=uuid.uuid4().hex;name='zg-original27-layer9-prefix-'+nonce
    r={'status':'UNQUALIFIED','package':identity,'expected_cpus':cpus,'owned_name':name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,'scope':scope()};fds={};child=log=None;failure=None;started=time.monotonic()
    try:
        fds,rows=source_open(plan,pins);r['source_before']=rows
        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15);require(inspected.returncode==0,'approved immutable image already exists; no pull')
        (output/'image-inspect.json').write_bytes(inspected.stdout);image,=json.loads(inspected.stdout);require(image['Id']==IMAGE.removeprefix('sha256:')and image['Digest']==IMAGE_DIGEST,'actual immutable image identity')
        env=environment(cpus,identity);command=argv(helper,output,plan,env,name,nonce);r['argv']=command;write(output/'prepared-invocation.json',r)
        try:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);r['owned_cli_pid']=child.pid
            log=helper.Log(child.stdout,output/'execution.log');r['exit_code']=log.observe(child,7200)
        except BaseException as e:failure=e
        finally:
            r['cleanup_errors']=helper.cleanup(child,name,nonce,output/'owned.cid',r)if child is not None else []
            if log is not None:
                try:log.finish()
                except BaseException as e:r['cleanup_errors'].append('log:'+repr(e))
                finally:r['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
            elif child is not None and child.stdout is not None:child.stdout.close()
        require(failure is None and r.get('exit_code')==0 and r['cleanup_errors']==[],'actual owned original capture/cleanup healthy')
        r['source_after']=source_after(fds,rows,plan['source_directory']);require(package()==identity,'same source package after owned retirement');r['status']='EXECUTED_ORIGINAL27B_LAYER9_PREFIX_REQUIRES_VALIDATION'
    except BaseException as e:failure=failure or e
    finally:
        errors=[]
        for fd in fds.values():
            try:os.close(fd)
            except BaseException as e:errors.append(repr(e))
        r['source_FD_cleanup_errors']=errors;r['source_FDs_closed']=not errors;r['elapsed_seconds_not_benchmark']=time.monotonic()-started
        if failure is not None or errors:r.update(status='FAIL_UNQUALIFIED',failure=repr(failure))
        write(output/'invocation.json',r)
    require(r['status']=='EXECUTED_ORIGINAL27B_LAYER9_PREFIX_REQUIRES_VALIDATION','capture refused; attempt preserved')
if __name__=='__main__':main()
