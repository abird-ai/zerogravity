"""Borrow original producer results. Wrappers return original objects unchanged."""
import hashlib
import json
import os
from pathlib import Path
from support import require, sha
from producer_contract import validate_producers


class ProducerTrace:
    def __init__(self, torch, module, model, prompt_rows):
        require(prompt_rows == 4, 'source-tokenized four-row diagnostic geometry')
        self.torch, self.module = torch, module
        self.prompt_rows = prompt_rows
        self.rows, self.base, self.selected = 0, 0, []
        self.calls = []
        self.records, self.keys, self.handles = [], set(), []
        self.raw = Path('/output/producers.bin').open('xb')
        self.active = False
        self.layer = None
        self.rotaries = 0
        self.original_rotary = module.apply_rotary_pos_emb
        self.original_attention = module.eager_attention_forward
        text = model.model.language_model
        self.post(text.embed_tokens, None, 'embedding')
        self.post(text.embed_tokens_per_layer, None, 'ple.embedding')
        self.post(text.per_layer_model_projection, None, 'ple.projection')
        self.post(text.per_layer_projection_norm, None, 'ple.norm')
        self.post(text.norm, None, 'final_norm', last_only=True)
        self.post(model.lm_head, None, 'head', last_only=True)
        for layer, block in enumerate(text.layers):
            self.pre(block, layer, 'block.input')
            def local(_m, args, kwargs, layer=layer):
                if self.active:
                    value=kwargs.get('per_layer_input', args[1] if len(args)>1 else None)
                    require(value is not None, 'actual whole-layer PLE argument')
                    self.tokens(layer,'ple.input',value)
            self.handles.append(block.register_forward_pre_hook(local, with_kwargs=True))
            self.post(block, layer, 'block.output')
            att=block.self_attn
            def enter(_m,_a,_k,layer=layer):
                if self.active:
                    require(self.layer is None, 'non-nested original attention scope')
                    self.layer=layer; self.rotaries=0
            def leave(_m,_a,_k,_o,layer=layer):
                if self.active:
                    require(self.layer==layer and self.rotaries==(2 if layer<15 else 1),
                            'exact owner/shared rotary source calls')
                    self.layer=None
            self.handles.append(att.register_forward_pre_hook(enter,with_kwargs=True))
            self.handles.append(att.register_forward_hook(leave,with_kwargs=True))
            for child,label in [('input_layernorm','input.norm'),('post_attention_layernorm','attention.norm'),
                                ('pre_feedforward_layernorm','ffn.norm'),('post_feedforward_layernorm','ffn.post_norm'),
                                ('per_layer_input_gate','ple.gate'),('per_layer_projection','ple.projection'),
                                ('post_per_layer_input_norm','ple.post_norm')]:
                self.post(getattr(block,child),layer,label)
            self.pre(block.pre_feedforward_layernorm,layer,'attention.residual')
            self.pre(block.per_layer_input_gate,layer,'ffn.residual')
            self.pre(block.per_layer_projection,layer,'ple.activation')
            self.post(block.act_fn,layer,'ple.gelu')
            for child,label in [('q_proj','query.projection'),('q_norm','query.norm'),('o_proj','attention.projection')]:
                self.post(getattr(att,child),layer,label)
            self.pre(att.o_proj,layer,'attention.pv')
            if layer<15:
                for child,label in [('k_proj','key.projection'),('k_norm','key.norm'),('v_norm','value.norm')]:
                    self.post(getattr(att,child),layer,label)
                if att.v_proj is not None:self.post(att.v_proj,layer,'value.projection')
            for child,label in [('gate_proj','ffn.gate'),('up_proj','ffn.up'),('down_proj','ffn.down'),('act_fn','ffn.gelu')]:
                self.post(getattr(block.mlp,child),layer,label)
            self.pre(block.mlp.down_proj,layer,'ffn.activation')
        module.apply_rotary_pos_emb=self.rotary
        module.eager_attention_forward=self.attention

    def begin_call(self,call,input_rows,base):
        require(not self.active and self.layer is None and call==len(self.calls) and call<3 and
                input_rows==(4 if call==0 else 1) and base==(0 if call==0 else call+3),
                'actual natural call producer context, no retagged prefix')
        self.rows,self.base=input_rows,base
        self.selected=[0,3] if call==0 else [0]
        self.active=True

    def finish_call(self,logits):
        require(self.active and self.layer is None and logits.shape==(1,1,262144),
                'completed original whole35 call and selected-last full vocabulary')
        self.record(self.base+self.rows-1,None,'logits',logits[0,0])
        self.calls.append({'call':len(self.calls),'base':self.base,'input_rows':self.rows,
                           'positions':[self.base+position for position in self.selected]})
        self.active=False

    def record(self, position, layer, label, tensor):
        t=self.torch
        require(tensor.dtype==t.bfloat16 and tensor.device.type=='cpu' and bool(t.isfinite(tensor).all()),
                'actual finite original BF16 producer')
        key=(position,layer,label)
        require(key not in self.keys and len(self.records)<4096,'unique bounded original producer')
        x=tensor.detach().contiguous().reshape(-1)
        blob=x.view(t.uint16).numpy().tobytes(order='C')
        require(0<len(blob)<=262144*2 and self.raw.tell()+len(blob)<=32*1024*1024,
                'bounded full selected-row producer, original file limit unchanged')
        self.records.append({'position':position,'layer':layer,'label':label,'elements':x.numel(),
                             'offset':self.raw.tell(),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        self.raw.write(blob);self.keys.add(key)

    def tokens(self,layer,label,tensor,last_only=False):
        if not self.active:return
        require(tensor.ndim>=2 and tensor.shape[0]==1,'actual source batch1 producer')
        if last_only:
            if tensor.shape[1]==1:self.record(self.base+self.rows-1,layer,label,tensor[0,0])
            else:
                require(tensor.shape[1]==self.rows,'actual initial sequence extent')
                self.record(self.base+self.rows-1,layer,label,tensor[0,-1])
        else:
            require(tensor.shape[1]==self.rows,'actual current-call token-axis producer')
            for position in self.selected:self.record(self.base+position,layer,label,tensor[0,position])

    def pre(self,module,layer,label):
        def hook(_m,args,kwargs):
            if self.active:self.tokens(layer,label,kwargs.get('hidden_states',args[0] if args else None))
        self.handles.append(module.register_forward_pre_hook(hook,with_kwargs=True))

    def post(self,module,layer,label,last_only=False):
        def hook(_m,_a,_k,output):
            if self.active:self.tokens(layer,label,output,last_only)
        self.handles.append(module.register_forward_hook(hook,with_kwargs=True))

    def rotary(self,*args,**kwargs):
        output=self.original_rotary(*args,**kwargs)
        if self.active:
            require(self.layer is not None and self.rotaries<2,'actual text rotary context')
            self.tokens(self.layer,'query.rotary' if self.rotaries==0 else 'key.rotary',output)
            self.rotaries+=1
        return output

    def attention(self,*args,**kwargs):
        if not self.active:return self.original_attention(*args,**kwargs)
        require(self.layer is not None and kwargs.get('scaling')==1.0,'original E2B unit-scale attention')
        original=self.torch.matmul; original_softmax=self.torch.nn.functional.softmax
        calls=softmax_calls=0
        def matmul(a,b,*rest,**options):
            nonlocal calls
            output=original(a,b,*rest,**options)
            require(calls<2,'exact two original eager attention matmuls')
            calls+=1
            return output
        def softmax(x,*rest,**options):
            nonlocal softmax_calls
            require(softmax_calls==0 and x.shape[:3]==(1,8,self.rows) and
                    options.get('dtype')==self.torch.float32,'actual masked QK before FP32 softmax')
            for position in self.selected:
                # Actual mask/scaling results, restricted to the real causal prefix.
                prefix=self.base+position+1
                require(x.shape[3]==self.base+self.rows,'actual current-root context extent')
                self.record(self.base+position,self.layer,'attention.qk',x[0,:,position,:prefix])
            softmax_calls+=1
            return original_softmax(x,*rest,**options)
        self.torch.matmul=matmul;self.torch.nn.functional.softmax=softmax
        try:output=self.original_attention(*args,**kwargs)
        finally:
            self.torch.matmul=original;self.torch.nn.functional.softmax=original_softmax
        require(calls==2 and softmax_calls==1,'complete actual QK/softmax/PV source scope')
        probabilities=output[1]
        for position in self.selected:
            self.record(self.base+position,self.layer,'attention.softmax',
                        probabilities[0,:,position,:self.base+position+1])
        return output

    def finish_trajectory(self,source):
        require(not self.active and self.layer is None and 1<=len(self.calls)<=3,
                'whole naturally completed producer trajectory')
        for handle in self.handles:handle.remove()
        self.handles=[]
        self.module.apply_rotary_pos_emb=self.original_rotary
        self.module.eager_attention_forward=self.original_attention
        self.raw.close()
        path=Path('/output/producers.bin')
        result={'schema':'zg.gemma4-e2b.producers.v2','source':source,'prompt_rows':self.prompt_rows,
                'call_count':len(self.calls),'calls':self.calls,
                'positions':[position for call in self.calls for position in call['positions']],
                'records':self.records,'raw':{'name':'producers.bin',
                'bytes':path.stat().st_size,'sha256':sha(path)},'original_results_unchanged':True,
                'scope':'initial rows0/last and each actual natural cached call, full35',
                'trained_rust_qualified':False}
        self.write_metadata(Path('/output/producers.prepared.json'),result)
        with path.open('rb') as raw:
            validate_producers(result,source,lambda at,n:(raw.seek(at),raw.read(n))[1])
        self.write_metadata(Path('/output/producers.json'),result)
        return {'name':'producers.json','bytes':Path('/output/producers.json').stat().st_size,
                'sha256':sha(Path('/output/producers.json'))}

    @staticmethod
    def write_metadata(path,result):
        # Compact metadata keeps the same8MiB Rust control envelope; raw words stream separately.
        blob=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        require(len(blob)<=1024*1024,'complete three-call producer metadata within1MiB')
        with path.open('xb') as file:
            os.fchmod(file.fileno(),0o600)
            file.write(blob)
