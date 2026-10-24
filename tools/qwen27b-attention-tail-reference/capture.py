"""ROOT-only authentic final63 attention/cache/FFN/head; no full-model loading."""
from pathlib import Path
import importlib.metadata
import json
import os
import resource
from contract import HERE,MODEL_SHA,TORCH_GIT,require,metadata,write,sha,digest,probe,finite,specs,package,stage_counts,archive_words
from tensor_bits import tensor_bits
from source_owner import SourceOwner

OWNER=None

def main():
    global OWNER
    out=Path('/output/captured');require(not out.exists(),'fresh original reference');out.mkdir(mode=0o700)
    report={'schema':'zg.original27b-attention-tail-capture.v1','status':'UNQUALIFIED','records':[]}
    try:
        cpus=[int(x)for x in os.environ['ZG_ALLOWED_CPUS'].split(',')]
        require(len(cpus)==len(set(cpus))==4,'four exact CPUs');os.sched_setaffinity(0,cpus)
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void'
            and os.environ.get('ATEN_CPU_CAPABILITY')=='avx2'and not os.environ.get('LD_PRELOAD')and not os.environ.get('LD_AUDIT'),'CPU-only reference')
        initial=package();require(digest(json.dumps(initial,sort_keys=True).encode())==os.environ['ZG_PACKAGE_SHA'],'exact package')
        plan=metadata(HERE/'inputs.json');pins=metadata(HERE/'source-pin.json');specs(plan,pins)
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3));OWNER=SourceOwner(plan,pins)
        import torch
        from torch.nn import functional as F
        from torch.utils._python_dispatch import TorchDispatchMode
        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
        from transformers.cache_utils import DynamicCache
        require(torch.__version__=='2.13.0+cu129'and torch.version.git_version==TORCH_GIT
            and importlib.metadata.version('transformers')=='5.17.0'and sha(original.__file__)==MODEL_SHA and not torch.cuda.is_available(),'pinned public original CPU code')
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        require(torch.backends.cpu.get_cpu_capability()=='AVX2','actual CPU profile')
        OWNER.map(torch);config=original.Qwen3_5TextConfig(**OWNER.config);config._attn_implementation='eager'
        with torch.device('meta'):
            input_norm=original.Qwen3_5RMSNorm(5120,eps=1e-6)
            attn=original.Qwen3_5Attention(config,63)
            ffn_norm=original.Qwen3_5RMSNorm(5120,eps=1e-6)
            mlp=original.Qwen3_5MLP(config,17408)
            final_norm=original.Qwen3_5RMSNorm(5120,eps=1e-6)
        norms=[(input_norm,'input_norm'),(ffn_norm,'ffn_norm'),(final_norm,'final_norm')]
        for module,role in norms:module.load_state_dict({'weight':OWNER.values[role]},strict=True,assign=True)
        attn.load_state_dict({**{p+'_proj.weight':OWNER.values[r]for p,r in [('q','query'),('k','key'),('v','value'),('o','attention_out')]},
            'q_norm.weight':OWNER.values['query_norm'],'k_norm.weight':OWNER.values['key_norm']},strict=True,assign=True)
        mlp.load_state_dict({p+'_proj.weight':OWNER.values[p]for p in ('gate','up','down')},strict=True,assign=True)
        for module in (input_norm,attn,ffn_norm,mlp,final_norm):module.eval()
        rotary=original.Qwen3_5TextRotaryEmbedding(config,device='cpu')
        modules=norms+[(attn.q_proj,'query'),(attn.k_proj,'key'),(attn.v_proj,'value'),(attn.o_proj,'attention_out'),
            (attn.q_norm,'query_norm'),(attn.k_norm,'key_norm'),(mlp.gate_proj,'gate'),(mlp.up_proj,'up'),(mlp.down_proj,'down')]
        for module,role in modules:require(module.weight.data_ptr()==OWNER.values[role].data_ptr()
            and module.weight.dtype==torch.bfloat16,'same original learned storage, no casts/copies')
        active={};hooks=[];observing=False;bmm=[]
        def once(label,value):require(label not in active,'one original '+label);active[label]=value
        def post(label):
            def save(_module,_args,result):once(label,result)
            return save
        def pre(label):
            def save(_module,args):require(len(args)==1,'unary actual producer');once(label,args[0])
            return save
        for module,label in [(attn.q_proj,'query_gate_projection'),(attn.k_proj,'key_projection'),(attn.v_proj,'value_projection'),
            (attn.q_norm,'normalized_query'),(attn.k_norm,'normalized_key'),(mlp.gate_proj,'ffn_gate'),(mlp.up_proj,'ffn_up'),(mlp.down_proj,'ffn_branch')]:
            hooks.append(module.register_forward_hook(post(label)))
        hooks.append(attn.q_norm.register_forward_pre_hook(pre('query')))
        hooks.append(attn.o_proj.register_forward_pre_hook(pre('gated_attention')))
        hooks.append(mlp.down_proj.register_forward_pre_hook(pre('ffn_gated')))
        original_rope=original.apply_rotary_pos_emb
        def observed_rope(*args,**kwargs):
            result=original_rope(*args,**kwargs);once('rotated_query',result[0]);once('rotated_key',result[1]);return result
        original.apply_rotary_pos_emb=observed_rope
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                if observing:
                    if func==torch.ops.aten.bmm.default:bmm.append(result)
                    elif func==torch.ops.aten.sigmoid.default:once('gate',args[0])
                    elif func==torch.ops.aten.mul.Tensor and len(bmm)==1 and list(result.shape[:3])==[1,24,1]:once('scores',result)
                return result
        completed={}
        with torch.inference_mode(),Observe():
            for repeat in range(2):
                for cohort in (0,1):
                    cache=DynamicCache(config=config)
                    for case in (c for c in plan['probes']if c['cohort']==cohort):
                        active.clear();bmm.clear();position=case['row']
                        x=torch.frombuffer(bytearray(probe(case['seed'])),dtype=torch.bfloat16).reshape(1,1,5120)
                        normalized_input=input_norm(x)
                        positions=torch.full((3,1,1),position,dtype=torch.long)
                        embeddings=rotary(normalized_input,positions)
                        observing=True
                        try:branch,probabilities=attn(normalized_input,embeddings,attention_mask=None,past_key_values=cache)
                        finally:observing=False
                        require(len(bmm)==2,'original eager QK and value each once')
                        active['dots']=bmm[0];active['context']=bmm[1];active['probabilities']=probabilities
                        layer=cache.layers[63]
                        active['keys']=layer.keys.transpose(1,2).contiguous()
                        active['values']=layer.values.transpose(1,2).contiguous()
                        require(layer.keys.shape[-2]==position+1,'genuine causal original cache progression')
                        post_attention=x+branch;normalized_ffn=ffn_norm(post_attention)
                        residual=post_attention+mlp(normalized_ffn);normalized=final_norm(residual)
                        logits=F.linear(normalized,OWNER.values['head'])
                        values={'input':x,'normalized_input':normalized_input,**active,'attention_branch':branch,
                            'post_attention':post_attention,'normalized_ffn':normalized_ffn,'residual':residual,'normalized':normalized,'logits':logits}
                        require(set(values)==set(stage_counts(position)),'exact complete same original forward region')
                        completed[(case['cohort'],position,repeat)]=values
        original.apply_rotary_pos_emb=original_rope
        for hook in hooks:hook.remove()
        total=0
        with (out/'raw.bin').open('xb')as archive:
            for case in plan['probes']:
                for repeat in range(2):
                    values=completed[(case['cohort'],case['row'],repeat)]
                    for label,count in stage_counts(case['row']).items():
                        value=values[label];require(value.dtype==torch.bfloat16 and value.device.type=='cpu'and value.numel()==count,'actual full producer dtype/extent')
                        bits=tensor_bits(value,torch);other=tensor_bits(completed[(case['cohort'],case['row'],1-repeat)][label],torch)
                        require(len(bits)==count*2<=plan['record_cap']and finite(bits)and bits==other,'complete finite stable original words')
                        if label=='input':require(bits==probe(case['seed']),'external input unchanged')
                        require(total+len(bits)<=plan['archive_cap'],'archive admission before transport');archive.write(bits)
                        report['records'].append({**case,'repeat':repeat,'layer':63,'label':label,'position':case['row'],
                            'shape':[1,count],'stride':[count,1],'original_shape':list(value.shape),'original_stride':list(value.stride()),
                            'dtype':'torch.bfloat16','device':'cpu','offset':total,'bytes':len(bits),'sha256':digest(bits)})
                        total+=len(bits)
        require(total==2*archive_words(plan),'whole bounded archive extent')
        OWNER.finish();require(package()==initial,'worker source unchanged')
        report.update(status='CAPTURED_ORIGINAL27B_ATTENTION_TAIL_REQUIRES_VALIDATION',snapshot=plan['snapshot'],repository=plan['repository'],revision=plan['revision'],
            package=initial,scope=plan['scope'],source_readsets=OWNER.readsets,source_views=OWNER.views,source_FDs=OWNER.before,
            all_original_source_and_mapped_bytes_unchanged=True,source_owner_retained_until_process_exit=True,raw={'file':'raw.bin','bytes':total,'sha256':sha(out/'raw.bin')},
            torch=torch.__version__,torch_git=torch.version.git_version,modeling_sha256=MODEL_SHA,affinity=sorted(os.sched_getaffinity(0)),threads=4,interop_threads=1,
            cohorts_are_independent=True,rows_within_cohort_are_causal=True,torch_call_shape=[1,5120])
    except BaseException as e:report['failure']=repr(e);raise
    finally:write(out/'manifest.json',report)
if __name__=='__main__':main()
