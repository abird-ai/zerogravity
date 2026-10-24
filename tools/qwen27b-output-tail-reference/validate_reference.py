"""ROOT-only independent original source/raw/owned validation; expected data only."""
import argparse
from contextlib import ExitStack
import json
import stat
import os
from pathlib import Path
from contract import HERE,STAGES,IMAGE,IMAGE_DIGEST,MODEL_SHA,TORCH_GIT,require,metadata,read,write,sha,probe,finite,specs,package,fd_sha,fingerprint,source_header,role_view,unique
from run_reference import argv,environment,owned_helper,source_open,source_after

class HeldReference:
    """The validated snapshot stays joined to the receipt through source scans."""
    def __init__(self,path,cap):
        self.path=Path(path);self.fd=os.open(self.path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        try:
            before=os.fstat(self.fd);self.before=fingerprint(before);self.bytes=before.st_size
            require(stat.S_ISREG(before.st_mode) and 0<self.bytes<=cap,'bounded held reference')
            with os.fdopen(os.dup(self.fd),'rb')as stream:self.data=stream.read(cap+1)
            require(len(self.data)==self.bytes,'complete held reference/EOF')
            self.sha=__import__('hashlib').sha256(self.data).hexdigest();self.verify()
        except BaseException:
            os.close(self.fd);raise
    def __enter__(self):return self
    def __exit__(self,*_):os.close(self.fd)
    def metadata(self):return json.loads(self.data,object_pairs_hook=unique)
    def verify(self):
        require(fingerprint(os.fstat(self.fd))==self.before,'same held reference before verification')
        current=os.stat(self.path,follow_symlinks=False)
        require(stat.S_ISREG(current.st_mode) and fingerprint(current)==self.before,'same reference path/held FD')
        require(fd_sha(self.fd,0,self.bytes)==self.sha and os.pread(self.fd,1,self.bytes)==b'' and fingerprint(os.fstat(self.fd))==self.before,'same complete reference SHA/EOF after validation')
        require(fingerprint(os.stat(self.path,follow_symlinks=False))==self.before,'reference path unchanged after verification')
    def pin(self):
        self.verify()
        return {'path':str(self.path),'bytes':self.bytes,'sha256':self.sha}

def records(plan,report,raw):
    expected=[(p,r,label,n)for p in plan['probes']for r in range(2)for label,n in STAGES.items()]
    require(len(report['records'])==len(expected)==72,'complete four-row/two-repeat nine-stage cohort');offset=0;prior={}
    for row,(p,repeat,label,count)in zip(report['records'],expected):
        require({k:row[k]for k in p}==p and row['repeat']==repeat and row['layer']==63 and row['label']==label and row['shape']==[1,count]and row['stride']==[count,1]and row['dtype']=='torch.bfloat16'and row['device']=='cpu'and row['offset']==offset and row['bytes']==count*2,'exact actual stage schema/order/extent')
        pin={'bytes':count*2,'sha256':row['sha256']};b=raw[offset:offset+pin['bytes']];require(len(b)==pin['bytes']and __import__('hashlib').sha256(b).hexdigest()==pin['sha256']and finite(b),'complete finite actual stage bytes')
        key=(p['cohort'],p['row'],label)
        if repeat==0:prior[key]=b
        else:require(prior[key]==b,'both original complete stage calls exact')
        if label=='input':require(b==probe(p['seed']),'declared external input only, no model producer')
        offset+=len(b)
    require(offset==len(raw)==report['raw']['bytes']==5218304,'complete archive coverage/EOF')
    return offset//2

def validate(output,receipt):
    with ExitStack()as held:return validate_held(output,receipt,held)

def validate_held(output,receipt,held):
    identity=package();plan=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');files=specs(plan,pins)
    require(output.is_absolute()and output.is_dir()and receipt.is_absolute()and not receipt.exists()and receipt.parent.is_dir(),'actual retired output/fresh receipt')
    receipt=receipt.parent.resolve()/receipt.name
    for old in [HERE,output,Path(plan['source_directory'])]:require(receipt!=old and old not in receipt.parents and receipt not in old.parents,'fresh receipt disjoint from immutable source/capture/package')
    invocation_file=held.enter_context(HeldReference(output/'invocation.json',4*1024**2));manifest_file=held.enter_context(HeldReference(output/'captured/manifest.json',4*1024**2))
    inv=invocation_file.metadata();r=manifest_file.metadata()
    require(inv['status']=='EXECUTED_ORIGINAL27B_FINAL_TAIL_REQUIRES_VALIDATION'and inv['package']==identity and inv['source_before']==inv['source_after']and inv['source_FDs_closed']is True and inv['source_FD_cleanup_errors']==[]and inv['cleanup_errors']==[]and inv['exit_code']==0 and inv['owned_container_absent_after_reap']is True,'original full source/lifecycle healthy')
    g=inv['foreground_cleanup'];log=inv['log_cleanup'];require(g['leader_reaped']is g['group_absent']is g['all_nonzero_signals_before_reap']is True and g['stage_errors']==[]and g['kill_attempts_before_reap']==2 and log['pipe_eof']is log['all_closed']is True and log['total_bytes']==log['retained_bytes']<=4*1024**2,'real owned TERM/KILL-before-reap/EOF no truncation')
    cpus=inv['expected_cpus'];require(len(cpus)==len(set(cpus))==4 and all(type(v)is int and v>=0 for v in cpus),'actual bounded CPU affinity')
    nonce=inv['owned_nonce'];require(len(nonce)==32 and all(c in '0123456789abcdef'for c in nonce)and inv['owned_name']=='zg-original27-tail-'+nonce,'exact owned nonce/name')
    helper=owned_helper();require(inv['argv']==argv(helper,output,plan,environment(cpus,identity),inv['owned_name'],nonce),'actual immutable image/entrypoint/env/mount/resource argv')
    inspected=metadata(output/'image-inspect.json');require(len(inspected)==1 and inspected[0]['Id']==IMAGE.removeprefix('sha256:')and inspected[0]['Digest']==IMAGE_DIGEST and inv['image_id']==IMAGE and inv['image_digest']==IMAGE_DIGEST,'actual original immutable image')
    require(r['schema']=='zg.original27b-final-tail-capture.v1'and r['status']=='CAPTURED_ORIGINAL27B_FINAL_TAIL_REQUIRES_VALIDATION'and r['package']==identity and r['scope']==plan['scope']and r['snapshot']==plan['snapshot']and r['repository']==plan['repository']and r['revision']==plan['revision'],'exact captured source/scope')
    require(r['torch']=='2.13.0+cu129'and r['torch_git']==TORCH_GIT and r['modeling_sha256']==MODEL_SHA and r['affinity']==cpus and r['threads']==4 and r['interop_threads']==1 and r['independent_rows']is True and r['torch_call_shape']==[1,5120]and r['all_original_source_and_mapped_bytes_unchanged']is r['source_owner_retained_until_process_exit']is True,'actual public CPU profile and retained original source')
    require(r['raw']['file']=='raw.bin','fixed contained raw name');raw_file=held.enter_context(HeldReference(output/'captured/raw.bin',8*1024**2))
    require(raw_file.bytes==r['raw']['bytes'] and raw_file.sha==r['raw']['sha256'],'same complete captured raw identity');words=records(plan,r,raw_file.data)
    require(set(p.name for p in (output/'captured').iterdir())=={'raw.bin','manifest.json'},'no untracked captured records')
    fds,rows=source_open(plan,pins)
    try:
        original=inv['source_before'];require(len(original)==len(rows)==31 and {v['name']:(v['bytes'],v['sha256'],v['fd_identity'])for v in original}=={v['name']:(v['bytes'],v['sha256'],v['fd_identity'])for v in rows},'fresh full held source agrees actual original before/after owner')
        require(set(r['source_readsets'])==set(r['source_views'])=={v['name']for v in plan['roles']},'all six actual learned parameter intervals')
        for role in plan['roles']:
            name=role['name'];shard=role['shard'];start,header=source_header(fds[shard],files[shard]);view=role_view(start,header,role,files[shard]['bytes'])
            require(view==r['source_views'][name]and fd_sha(fds[shard],view['absolute'],view['bytes'])==r['source_readsets'][name]and fingerprint(os.fstat(fds[shard]))==r['source_FDs'][shard],'independent complete source/header/readset/owner join')
        source_after(fds,rows)
    finally:
        for fd in fds.values():os.close(fd)
    require(package()==identity,'package identity after validation')
    def package_pin(name):return {'path':str(HERE/name),**identity[name]}
    result={'schema':'zg.actual-original27b-final-tail-reference.v1','status':'ACTUAL_ORIGINAL27B_FINAL_TAIL_REFERENCE_ONLY','snapshot':plan['snapshot'],'repository':plan['repository'],'revision':plan['revision'],'config_sha256':plan['config_sha256'],'index_sha256':plan['index_sha256'],'scope':plan['scope'],'cases':4,'independent_cohorts':[1,3],'original_repetitions':2,'records':72,'compared_bf16_words':words,'input_trained_full_model_producer_qualified':False,'rust_numerical_parity_qualified':False,'whole_model_state_gpu_serving_or_performance_qualified':False,'original_source_and_owned_lifetime_closed':True,'manifest':manifest_file.pin(),'raw':raw_file.pin(),'invocation':invocation_file.pin(),'source_pin':package_pin('source-pin.json'),'inputs':package_pin('inputs.json'),'package':identity,'source_files':rows}
    write(receipt,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();validate(a.output,a.receipt)
