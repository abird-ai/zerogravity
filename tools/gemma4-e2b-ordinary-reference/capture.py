"""One original model, seven genuine text trajectories, two independent caches each."""
import hashlib,inspect,os,sys
from pathlib import Path
from support import metadata,require,write,sha

class Archive:
    def __init__(self,path,limit):
        self.file=Path(path).open('x+b');self.limit=limit;self.records=[];self.cursor=0;self.repeat=0
    def begin(self,repeat):self.repeat=repeat;self.cursor=0
    def record(self,context,tensor,torch):
        require(tensor.device.type=='cpu'and tensor.dtype==torch.bfloat16 and bool(tensor.isfinite().all()),'actual finite original BF16 source')
        # Lossless raw storage: reinterpret original BF16 words, NEVER numeric cast.
        blob=tensor.detach().contiguous().view(torch.uint16).numpy().tobytes(order='C')
        require(0<len(blob)<=524288 and len(blob)==tensor.numel()*2,'whole authentic BF16 plane')
        record=dict(context,shape=list(tensor.shape),bytes=len(blob),dtype='bf16-le',sha256=hashlib.sha256(blob).hexdigest())
        if self.repeat==0:
            require(self.file.tell()+len(blob)<=self.limit and len(self.records)<4096,'unchanged archive byte/record cap')
            record['offset']=self.file.tell();self.file.write(blob);self.records.append(record)
        else:
            require(self.cursor<len(self.records),'complete independently repeated archive')
            expected=self.records[self.cursor]
            require(record=={k:v for k,v in expected.items()if k!='offset'}and os.pread(self.file.fileno(),len(blob),expected['offset'])==blob,
                'fresh original cache all source bytes equal; never expected operands')
        self.cursor+=1
    def end(self):
        if self.repeat:require(self.cursor==len(self.records),'whole second original trajectory')
        self.file.flush()
    def close(self):
        self.file.flush();name=Path(self.file.name);self.file.close()
        return dict(name=name.name,bytes=name.stat().st_size,sha256=sha(name),records=self.records)


def main():
    cpus=[int(x)for x in os.environ['ZG_ALLOWED_CPUS'].split(',')];require(len(set(cpus))==4,'four original CPUs');os.sched_setaffinity(0,cpus)
    import torch,transformers,tokenizers
    from transformers import AutoTokenizer,Gemma4ForConditionalGeneration
    from transformers.models.gemma4 import modeling_gemma4
    from transformers import cache_utils
    require(sys.byteorder=='little'and transformers.__version__=='5.17.0'and torch.version.git_version=='cf30153c4c131c8164ee7798e5022d810682e2cb','exact original profile')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    request=metadata('/probe/request.json');pins=metadata('/probe/source-pin.json')
    require(request['schema']=='zg.gemma4-e2b.ordinary-request.v1'and request['max_output']==8 and request['add_special_tokens']is True and len(request['cases'])==7,'whole genuine plain source matrix')
    tokenizer=AutoTokenizer.from_pretrained('/source',local_files_only=True,trust_remote_code=False)
    inputs=[]
    for case in request['cases']:
        ids=tokenizer.encode(case['text'],add_special_tokens=True)
        require(len(ids)==case['rows']and 1<=len(ids)<=57 and all(type(i)is int and 0<=i<262144 for i in ids),'actual tokenizer geometry, no token forcing/padding')
        inputs.append(ids)
    model,loading=Gemma4ForConditionalGeneration.from_pretrained('/source',local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
    for key,value in list(loading.items()):
        if type(value)is set:require(all(type(x)is str for x in value),'loading diagnostics');loading[key]=sorted(value)
    require(not loading.get('missing_keys')and not loading.get('mismatched_keys')and not loading.get('error_msgs'),'exact trained source loading')
    unused={f'model.language_model.layers.{i}.self_attn.{name}'for i in range(15,35)for name in ('k_proj.weight','v_proj.weight','k_norm.weight')}
    require(set(loading.get('unexpected_keys',[]))<=unused,'only original redundant shared-KV roles omitted')
    model.eval();text=model.model.language_model;require(len(text.layers)==35 and model.lm_head.weight.data_ptr()==text.embed_tokens.weight.data_ptr(),'whole35 and authentic tied head')
    eos=model.generation_config.eos_token_id;require(eos==[1,106,50],'actual three source stop IDs')
    active=None;handles=[]
    for layer,block in enumerate(text.layers):
        def pre(_module,args,kwargs,layer=layer):
            if active is None:return
            x=kwargs.get('hidden_states',args[0]if args else None)
            require(x.shape==(1,active['rows'],1536),'whole original block input')
            active['archive'].record(dict(call=active['call'],base=active['base'],layer=layer,label='block.input'),x[0],torch)
        def post(_module,_args,_kwargs,x,layer=layer):
            if active is None:return
            require(x.shape==(1,active['rows'],1536),'whole original block output')
            active['archive'].record(dict(call=active['call'],base=active['base'],layer=layer,label='block.output'),x[0],torch)
        handles+=[block.register_forward_pre_hook(pre,with_kwargs=True),block.register_forward_hook(post,with_kwargs=True)]
    cells=[]
    try:
        for case,ids in zip(request['cases'],inputs):
            raw=Archive('/output/'+case['name']+'.bin',16*1024*1024);boundaries=Archive('/output/'+case['name']+'-boundaries.bin',32*1024*1024)
            calls=None
            for repeat in range(2):
                cache=None;current=ids;frontier=0;previous={};actual_calls=[];raw.begin(repeat)
                if repeat==0:boundaries.begin(0)
                with torch.inference_mode():
                    for call in range(8):
                        base=frontier;frontier+=len(current);require(frontier<=64,'source finite context')
                        active=dict(archive=boundaries,call=call,base=base,rows=len(current))if repeat==0 else None
                        try:out=model(input_ids=torch.tensor([current],dtype=torch.long),attention_mask=torch.ones((1,frontier),dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                        finally:active=None
                        cache=out.past_key_values;require(out.logits.shape==(1,1,262144),'full original selected-last vocabulary')
                        selected=int(out.logits[0,0].argmax().item());raw.record(dict(call=call,label='logits',frontier=frontier),out.logits[0,0],torch)
                        next_prior={}
                        for owner in range(15):
                            width=512 if owner%5==4 else 256
                            for label,x in [('key',cache.layers[owner].keys),('value',cache.layers[owner].values)]:
                                require(x.shape==(1,1,frontier,width),'all15 genuine owner geometries')
                                if call:require(torch.equal(previous[(owner,label)],x[0,0,:base]),'actual consumed-prior continuity')
                                raw.record(dict(call=call,label=f'kv.{owner}.{label}',frontier=frontier),x[0,0],torch)
                                next_prior[(owner,label)]=x[0,0].clone()
                        previous=next_prior
                        actual_calls.append(dict(call=call,input_ids=current,base=base,frontier=frontier,selected_id=selected,eos=selected in eos))
                        if selected in eos:break
                        current=[selected]
                raw.end()
                if repeat==0:boundaries.end()
                if repeat==0:calls=actual_calls
                else:require(calls==actual_calls,'independent real natural feedback/stop equality')
                del cache
            r=raw.close();b=boundaries.close()
            cell=dict(schema='zg.gemma4-e2b.ordinary-cell.v1',request=case,prompt_ids=ids,calls=calls,raw=r,boundaries=b,boundary_repetitions=1,repetitions=2,repetition_equal=True)
            write('/output/'+case['name']+'.json',cell);cells.append(dict(name=case['name'],metadata=dict(name=case['name']+'.json',bytes=Path('/output/'+case['name']+'.json').stat().st_size,sha256=sha('/output/'+case['name']+'.json'))))
    finally:
        for handle in handles:handle.remove()
    write('/output/capture.json',dict(schema='zg.gemma4-e2b.ordinary-matrix.v1',source=pins,request=request,cells=cells,eos_ids=eos,loading=loading,
        torch=torch.__version__,torch_git=torch.version.git_version,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        original_code_sha256={name:sha(inspect.getfile(module))for name,module in [('modeling_gemma4.py',modeling_gemma4),('cache_utils.py',cache_utils)]},
        profile='HfTorchCpuBf16Avx2Eager5.17.0',plain_vs_observed_equal=True,trained_rust_qualified=False,native_serving_chat_or_performance_qualified=False))
    print('ORIGINAL_GEMMA4_ORDINARY_MATRIX_COMPLETE',flush=True)
if __name__=='__main__':main()
