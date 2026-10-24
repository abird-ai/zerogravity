"""ROOT-only public original final63 MLP/residual/norm/head, external row probes."""
from pathlib import Path
import importlib.metadata
import json
import mmap
import os
import resource
import stat
from contract import HERE,STAGES,MODEL_SHA,TORCH_GIT,require,metadata,read,write,sha,digest,fd_sha,fingerprint,probe,finite,specs,source_header,role_view,package
from tensor_bits import tensor_bits

class SourceOwner:
    def __init__(self,plan,pins):
        self.plan=plan;self.files=specs(plan,pins);self.fds={};self.before={};self.headers={};self.maps={};self.views={};self.values={};self.versions={};self.readsets={}
        for name in ['config.json','model.safetensors.index.json']:read(Path('/source')/name,pin=self.files[name])
        self.config=metadata('/source/config.json')['text_config'];index=metadata('/source/model.safetensors.index.json')['weight_map']
        t=self.config
        require(t['hidden_size']==5120 and t['intermediate_size']==17408 and t['vocab_size']==248320 and t['num_hidden_layers']==64 and t['rms_norm_eps']==1e-6 and t['hidden_act']=='silu' and t['tie_word_embeddings']is False,'exact original final63 CPU geometry/activation')
        for name in plan['selected_shards']:
            pin=self.files[name];fd=os.open(Path('/source')/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);self.fds[name]=fd;m=os.fstat(fd)
            require(stat.S_ISREG(m.st_mode)and m.st_size==pin['bytes'],'selected source extent before any full scan');self.before[name]=fingerprint(m)
            require(fd_sha(fd,0,pin['bytes'])==pin['sha256'] and fingerprint(m)==fingerprint(os.fstat(fd)),'actual full selected source SHA before import')
            self.headers[name]=source_header(fd,pin)
        for role in plan['roles']:
            name=role['name'];shard=role['shard'];require(index[name]==shard,'same actual index→shard role')
            start,header=self.headers[shard];self.views[name]=role_view(start,header,role,self.files[shard]['bytes'])
    def map(self,torch):
        require(sum(v['mapping_bytes']for v in self.views.values())<4*1024**3,'bounded source-only mappings')
        for role in self.plan['roles']:
            name=role['name'];v=self.views[name];m=mmap.mmap(self.fds[role['shard']],v['mapping_bytes'],access=mmap.ACCESS_COPY,offset=v['aligned']);self.maps[name]=m
            x=torch.frombuffer(m,dtype=torch.bfloat16,count=v['bytes']//2,offset=v['view_offset']).reshape(role['shape']);require(x.is_contiguous()and x.device.type=='cpu','authentic full original role view')
            self.values[role['role']]=x;self.versions[role['role']]=x._version;self.readsets[name]=fd_sha(self.fds[role['shard']],v['absolute'],v['bytes'])
    def finish(self):
        for role in self.plan['roles']:
            name=role['name'];v=self.views[name];require(self.values[role['role']]._version==self.versions[role['role']],'source tensor version unchanged')
            h=__import__('hashlib').sha256();view=memoryview(self.maps[name])
            try:
                for i in range(0,v['bytes'],1024**2):h.update(view[v['view_offset']+i:v['view_offset']+min(v['bytes'],i+1024**2)])
            finally:view.release()
            require(h.hexdigest()==self.readsets[name],'mapped source bytes unchanged')
        for name,fd in self.fds.items():require(fingerprint(os.fstat(fd))==self.before[name]and fd_sha(fd,0,self.files[name]['bytes'])==self.files[name]['sha256'],'same full source FD/bytes after math')
OWNER=None

def main():
    global OWNER
    out=Path('/output/captured');require(not out.exists(),'fresh original reference');out.mkdir(mode=0o700)
    report={'schema':'zg.original27b-final-tail-capture.v1','status':'UNQUALIFIED','records':[]}
    try:
        cpus=[int(x)for x in os.environ['ZG_ALLOWED_CPUS'].split(',')];require(len(cpus)==len(set(cpus))==4,'four exact CPUs');os.sched_setaffinity(0,cpus)
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void'and os.environ.get('ATEN_CPU_CAPABILITY')=='avx2'and not os.environ.get('LD_PRELOAD')and not os.environ.get('LD_AUDIT'),'original CPU-only policy')
        initial=package();require(digest(json.dumps(initial,sort_keys=True).encode())==os.environ['ZG_PACKAGE_SHA'],'exact worker package')
        plan=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');specs(plan,pins)
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3));OWNER=SourceOwner(plan,pins)
        import torch
        from torch.nn import functional as F
        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
        require(torch.__version__=='2.13.0+cu129'and torch.version.git_version==TORCH_GIT and importlib.metadata.version('transformers')=='5.17.0'and sha(original.__file__)==MODEL_SHA and not torch.cuda.is_available(),'pinned original CPU source/runtime')
        torch.set_num_threads(4);torch.set_num_interop_threads(1);require(torch.backends.cpu.get_cpu_capability()=='AVX2','actual CPU profile')
        OWNER.map(torch);config=original.Qwen3_5TextConfig(**OWNER.config)
        with torch.device('meta'):
            ffn_norm=original.Qwen3_5RMSNorm(5120,eps=1e-6);mlp=original.Qwen3_5MLP(config,17408);final_norm=original.Qwen3_5RMSNorm(5120,eps=1e-6)
        ffn_norm.load_state_dict({'weight':OWNER.values['ffn_norm']},strict=True,assign=True);final_norm.load_state_dict({'weight':OWNER.values['final_norm']},strict=True,assign=True)
        mlp.load_state_dict({k+'_proj.weight':OWNER.values[k]for k in ['gate','up','down']},strict=True,assign=True)
        for module in [ffn_norm,mlp,final_norm]:module.eval()
        for module,role in [(ffn_norm,'ffn_norm'),(final_norm,'final_norm'),(mlp.gate_proj,'gate'),(mlp.up_proj,'up'),(mlp.down_proj,'down')]:require(module.weight.data_ptr()==OWNER.values[role].data_ptr()and module.weight.dtype==torch.bfloat16,'no learned cast/copy/provider replacement')
        active={}
        def hook(label):
            def retain(_module,_args,result):require(label not in active,'original producer called once');active[label]=result
            return retain
        hooks=[mlp.gate_proj.register_forward_hook(hook('gate')),mlp.up_proj.register_forward_hook(hook('up')),mlp.down_proj.register_forward_hook(hook('branch'))]
        def retain_gated(_module,args):require(len(args)==1 and 'gated'not in active,'actual Down input once');active['gated']=args[0]
        hooks.append(mlp.down_proj.register_forward_pre_hook(retain_gated))
        total=0
        with (out/'raw.bin').open('xb')as archive,torch.inference_mode():
            for case in plan['probes']:
                raw=probe(case['seed']);x=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(1,5120);repeats=[]
                for _ in range(2):
                    active.clear();normalized_ffn=ffn_norm(x);branch=mlp(normalized_ffn);residual=x+branch;normalized=final_norm(residual);logits=F.linear(normalized,OWNER.values['head'])
                    require(set(active)=={'gate','up','gated','branch'},'exact original MLP producer hooks')
                    repeats.append({'input':x,'normalized_ffn':normalized_ffn,**active,'residual':residual,'normalized':normalized,'logits':logits})
                for repeat,values in enumerate(repeats):
                    for stage,count in STAGES.items():
                        value=values[stage];require(value.dtype==torch.bfloat16 and list(value.shape)==[1,count]and value.device.type=='cpu'and value.is_contiguous(),'exact complete stage schema')
                        bits=tensor_bits(value,torch);require(len(bits)==count*2<=plan['record_cap']and finite(bits),'finite complete original stage')
                        require(bits==tensor_bits(repeats[1-repeat][stage],torch),'both original calls stable all words')
                        if stage=='input':require(bits==raw,'external boundary unchanged')
                        require(total+len(bits)<=plan['archive_cap'],'archive cap before transport');archive.write(bits)
                        report['records'].append({**case,'repeat':repeat,'layer':63,'label':stage,'shape':[1,count],'stride':[count,1],'dtype':'torch.bfloat16','device':'cpu','offset':total,'bytes':len(bits),'sha256':digest(bits)});total+=len(bits)
        for h in hooks:h.remove()
        OWNER.finish();require(package()==initial,'worker source package unchanged')
        report.update(status='CAPTURED_ORIGINAL27B_FINAL_TAIL_REQUIRES_VALIDATION',snapshot=plan['snapshot'],repository=plan['repository'],revision=plan['revision'],package=initial,scope=plan['scope'],source_readsets=OWNER.readsets,source_views=OWNER.views,source_FDs=OWNER.before,all_original_source_and_mapped_bytes_unchanged=True,source_owner_retained_until_process_exit=True,raw={'file':'raw.bin','bytes':total,'sha256':sha(out/'raw.bin')},torch=torch.__version__,torch_git=torch.version.git_version,modeling_sha256=MODEL_SHA,affinity=sorted(os.sched_getaffinity(0)),threads=4,interop_threads=1,torch_call_shape=[1,5120],independent_rows=True)
    except BaseException as e:report['failure']=repr(e);raise
    finally:write(out/'manifest.json',report)
if __name__=='__main__':main()
