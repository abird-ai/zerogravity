"""Original HF CPU invocation only. No model load at import; no Rust operands."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from support import require, metadata, write, sha


def main():
    allowed=[int(x) for x in os.environ['ZG_ALLOWED_CPUS'].split(',')]
    require(len(allowed)==len(set(allowed))==4, 'four original admitted CPU identities')
    os.sched_setaffinity(0,allowed)
    require(sorted(os.sched_getaffinity(0))==sorted(allowed), 'actual CPU affinity before framework import')
    import torch
    import transformers
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from transformers.models.gemma4 import modeling_gemma4
    from transformers import cache_utils
    require(torch.__version__ == '2.13.0+cu129' and torch.version.git_version ==
            'cf30153c4c131c8164ee7798e5022d810682e2cb' and transformers.__version__ == '5.17.0',
            'existing approved original Torch/Transformers profile')
    require(sys.byteorder == 'little', 'exact LE original CPU archive')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and not torch.cuda.is_available(), 'CPU only')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = metadata(Path('/probe/source-pin.json'))
    request = metadata(Path('/probe/request.json'))
    require(request == {'text': 'One plus one is', 'add_special_tokens': True, 'max_output': 3},
            'one bounded real tokenizer invocation with natural continuation')
    tokenizer = AutoTokenizer.from_pretrained('/source', local_files_only=True, trust_remote_code=False)
    encoded = tokenizer(request['text'], add_special_tokens=True, return_tensors='pt')
    ids = encoded['input_ids'].tolist()[0]
    require(0 < len(ids) <= 64 and all(type(x) is int and 0 <= x < 262144 for x in ids), 'actual tokenizer rows')
    model, loading = Gemma4ForConditionalGeneration.from_pretrained('/source', local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager', output_loading_info=True)
    model.eval()
    # Transformers5.17 returns sets for loading-key diagnostics; retain exact sorted members.
    for key,value in list(loading.items()):
        if type(value) is set:
            require(all(type(item) is str for item in value), 'exact source loading-key diagnostics')
            loading[key]=sorted(value)
    require(not loading.get('missing_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs'),
            'no missing/mismatched original checkpoint roles')
    unused = {f'model.language_model.layers.{i}.self_attn.{name}'
              for i in range(15,35) for name in ('k_proj.weight','v_proj.weight','k_norm.weight')}
    require(set(loading.get('unexpected_keys', [])) <= unused, 'only exact redundant shared-KV source roles unused')
    text = model.config.get_text_config()
    require(text.hidden_size == 1536 and text.num_hidden_layers == 35 and text.vocab_size == 262144 and
            text.num_kv_shared_layers == 20, 'actual full E2B geometry')
    require(model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr(),
            'actual tied original output source')
    eos = model.generation_config.eos_token_id
    eos = [eos] if type(eos) is int else eos
    require(type(eos) is list and all(type(x) is int and 0 <= x < 262144 for x in eos), 'original stop IDs')
    trace=None; producer=None
    if os.environ.get('ZG_GEMMA_PRODUCERS')=='1':
        from producers import ProducerTrace
        producer=ProducerTrace(torch,modeling_gemma4,model,len(ids))
    all_records=[]; calls=[]; raw_path=Path('/output/raw.bin'); prior_repeat=None
    with raw_path.open('xb') as raw:
        def record(call, label, tensor):
            require(tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16 and bool(torch.isfinite(tensor).all()),
                    'complete original finite BF16 CPU result')
            x=tensor.detach().to(torch.float32).contiguous()
            blob=x.numpy().tobytes(order='C')
            require(len(blob) == x.numel()*4, 'full F32-widened original words')
            row={'call':call,'label':label,'shape':list(x.shape),'offset':raw.tell(),'bytes':len(blob),
                 'dtype':'original-bf16-widened-f32-le','sha256':hashlib.sha256(blob).hexdigest()}
            raw.write(blob); all_records.append(row)
            return blob
        for repeat in range(2):
            cache=None; current=ids; frontier=0; previous={}; repeat_blobs=[]; actual_calls=[]
            with torch.inference_mode():
                for call in range(request['max_output']):
                    x=torch.tensor([current],dtype=torch.long)
                    if producer is not None and repeat==0:
                        producer.begin_call(call,len(current),frontier)
                    frontier+=len(current)
                    output=model(input_ids=x,attention_mask=torch.ones((1,frontier),dtype=torch.long),
                                 past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                    cache=output.past_key_values
                    require(output.logits.shape == (1,1,262144), 'actual selected-last-row full vocabulary')
                    if producer is not None and repeat==0:
                        producer.finish_call(output.logits)
                    selected=int(output.logits[0,0].argmax().item())
                    state={}
                    tensors=[('logits', output.logits[0,0])]
                    for owner in range(15):
                        width=512 if owner%5 == 4 else 256
                        layer=cache.layers[owner]
                        for name,tensor in [('key',layer.keys),('value',layer.values)]:
                            require(tensor.shape == (1,1,frontier,width), 'actual owner KV token/head extent')
                            row=tensor[0,0]
                            if call:
                                require(torch.equal(previous[(owner,name)],row[:frontier-1]), 'real consumed-prior continuity')
                            state[(owner,name)]=row.clone()
                            tensors.append((f'kv.{owner}.{name}',row))
                    if repeat == 0:
                        for label,tensor in tensors: repeat_blobs.append(record(call,label,tensor))
                        actual_calls.append({'call':call,'input_ids':current,'frontier':frontier,'selected_id':selected,
                                             'eos':selected in eos})
                    else:
                        for label,tensor in tensors:
                            blob=tensor.to(torch.float32).contiguous().numpy().tobytes(order='C')
                            require(blob == prior_repeat[len(repeat_blobs)], 'independent fresh-cache repetition equality')
                            repeat_blobs.append(blob)
                        require(actual_calls == [], 'no expected feedback injection')
                        require(current == calls[call]['input_ids'] and selected == calls[call]['selected_id'],
                                'repeat actual feedback equality')
                    previous=state
                    if selected in eos: break
                    current=[selected]
            if repeat == 0:
                prior_repeat=repeat_blobs; calls=actual_calls
                if producer is not None:trace=producer.finish_trajectory(plan)
            else: require(len(repeat_blobs) == len(prior_repeat), 'whole repeated trajectory extent')
    require(0 < raw_path.stat().st_size <= 16*1024*1024, 'bounded complete original logits/KV archive')
    write(Path('/output/capture.json'),{'schema':'zg.gemma4-e2b.original-text.v1',
        'profile':'HfTorchCpuBf16Avx2Eager5.17.0','rust_profile':'AnalyticF32TextV1_UNQUALIFIED',
        'source':plan,'request':request,'prompt_ids':ids,'eos_ids':eos,'calls':calls,'records':all_records,
        'raw':{'name':'raw.bin','bytes':raw_path.stat().st_size,'sha256':sha(raw_path)},
        'repetitions':2,'repetition_equal':True,'loading':loading,
        'producers':trace,
        'original_code_sha256':{'modeling_gemma4.py':sha(Path(inspect.getfile(modeling_gemma4))),
                                'cache_utils.py':sha(Path(inspect.getfile(cache_utils)))},
        'torch':torch.__version__,'torch_git':torch.version.git_version,'transformers':transformers.__version__,
        'tokenizer_class':type(tokenizer).__name__,'trained_rust_qualified':False})
    print('ORIGINAL_GEMMA4_E2B_TEXT_REFERENCE_COMPLETE',flush=True)

if __name__ == '__main__': main()
