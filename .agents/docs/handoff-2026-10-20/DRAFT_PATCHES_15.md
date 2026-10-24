# Draft Patches — volume 15

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/NEW_PREPARATION.patch

Original UTF-8 bytes: 107301; SHA256 `33d829e90187836239b7ad69a22416eafa0970353b0ff2448ea90e32a36013bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/README.md
@@ -0,0 +1,85 @@
+# Original27B layer0 GDN public reference — prepared only
+
+Author: spine_review. Parent plan: model-execution.
+
+- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
+  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
+  original module; no architecture inference from the model name.
+- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
+  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
+  real cached recurrence using that first return through original DynamicCache.
+  No whole-model constructor/traversal,55GB materialization or112-domain claim.
+- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
+  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
+  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
+  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
+  official-loader blanket cast or conversion from GGUF learned values.
+- Genuine original source public forward called unchanged once per invocation.
+  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
+  fallback must identify original Torch code or refuse. No function replacement,
+  scalar arithmetic port, reconstructed state or native exp eligibility.
+- Independent fresh plain/observed caches, same immutable weights. Every token's
+  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
+  must be bit-identical. True initial return must equal cached prior in all words.
+  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
+- Original call/line/return frame hooks reuse accepted source line maps. Capture
+  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
+  actual conv operands/return, projection input/output, and gated RMS stages.
+  Scoped TorchDispatchMode records original coefficient operations, five real
+  initial exps, cached exp(g) and norm args/results; forwards each original op
+  once and returns its actual result. Observer clones are suppressed.
+- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
+  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
+- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
+  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
+  captured values and published outputs must be finite. No tolerance widening.
+- Six actual embedding/inputnorm/QKV outputs also match the independently
+  accepted original27B public prefix reference in every word.
+
+## Bounds and source ownership
+
+- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
+  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
+  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
+- Each COW window maps page-aligned offset and only leading alignment+complete
+  original tensor bytes. Full embedding raw view2.543GB VA retains its original
+  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
+- Before library initialization, derive separate virtual and physical-envelope
+  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
+  captured payload,2GiB runtime resident allowance,64MiB control allowance.
+  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
+ 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
+  These estimates/OS containment do NOT prove native ResourceArbiter admission
+  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
+- ROOT alone hashes ALL18 original shards/all source files before and after via
+  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
+  verified before importing Torch and rechecked after; COW parameter readsets,
+  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
+  owners remain until the owned process exits; no premature mmap close.
+- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
+  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
+  All postspawn paths use original independent cleanup stages and EOF proof.
+  No next provider/process may start after cleanup/evidence failure.
+- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
+  that a configured limit or historical RSS is native physical admission.
+
+## ROOT execution, after independent source review
+
+All directories fresh. ROOT owns every test/container/operator execution.
+
+```bash
+python3 "$RUN/test_contracts.py"
+python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
+python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
+```
+
+Output: untouched plain archive +two prevalidation observed archives, full
+source/runtime/parameter/cleanup records. Failure preserves actual metadata and
+raw frames before semantic validation. Final validator compares all raw inputs,
+intermediates, initial/cached prior, source gamma rounding and transparency.
+H48 cached inputs804960 words and outputs798720 words become reference artifacts
+only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
+future source-kind admission and native H48 replay require separate review.
+
+No author new source payload hashes, model/provider/operator/library/container,
+Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/TODO.md
@@ -0,0 +1,16 @@
+Parent: `.agents/plans/model-execution/`. Owner: spine_review.
+
+- [x] Read actual original config, index and bounded selected source headers.
+- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
+- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
+- [x] Prepare original module/meta strict assignment, real DynamicCache, two
+  teacher plain/observed reference, original intermediate/dispatch observers.
+- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
+  and independent raw/state/source/provider/cleanup validator; no math replacement.
+- [x] Complete author static self-review/AST/freeze; no runtime or tests.
+- [ ] ROOT independent full source review.
+- [ ] ROOT source-free fixtures, actual original public reference and validator.
+- [ ] Separate H48 source-kind/native cached slice qualification after reference.
+
+No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
+exp/GPU admission from this source-only preparation.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/capture.py
@@ -0,0 +1,315 @@
+"""ROOT ONLY original27B layer0 GDN module; never a full-model constructor."""
+import gc
+import importlib.metadata
+import inspect
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import struct
+import sys
+import time
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, metadata, read, require, sha, unique, write
+
+SOURCE = Path('/source')
+OUTPUT = Path('/output/captured')
+EMBED = 'model.language_model.embed_tokens.weight'
+INPUT_NORM = 'model.language_model.layers.0.input_layernorm.weight'
+PREFIX = 'model.language_model.layers.0.linear_attn.'
+SOURCE_OWNER = None # Original FD/map/storage owner retained until owned process exit.
+
+class SourceOwner:
+    def __init__(self, plan):
+        self.plan, self.fds, self.headers, self.before = plan, {}, {}, {}
+        self.maps, self.tensors, self.readsets, self.versions = {}, {}, {}, {}
+        for name,pin in plan['metadata'].items():
+            read(SOURCE/name,pin=pin)
+        self.index=metadata(SOURCE/'model.safetensors.index.json')
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(SOURCE/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            self.fds[shard]=fd
+            self.before[shard]=fingerprint(os.fstat(fd))
+            require(os.fstat(fd).st_size==pin['file']['bytes'] and
+                    fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],
+                    'complete same opened original selected shard before library initialization')
+            n,=struct.unpack('<Q',os.pread(fd,8,0))
+            raw=os.pread(fd,n,8)
+            require(n==pin['header_bytes'] and len(raw)==n and n<=8*1024*1024 and
+                    digest(raw)==pin['header_sha256'],'exact original selected shard header')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+
+    def load_views(self,torch):
+        for name,item in self.plan['tensors'].items():
+            shard=item['shard'];begin,header=self.headers[shard];entry=header[name]
+            require(self.index['weight_map'][name]==shard and entry['dtype']=='BF16' and
+                    entry['shape']==item['shape'] and begin+entry['data_offsets'][0]==item['absolute_offset'] and
+                    entry['data_offsets'][1]-entry['data_offsets'][0]==item['bytes'],
+                    'actual exact source role/name/geometry/full-byte extent')
+            # COW protects source files; every window is only its exact parameter
+            # plus the leading alignment bytes, not a whole-shard mapping.
+            mapping=mmap.mmap(self.fds[shard],item['mapping_bytes'],access=mmap.ACCESS_COPY,
+                              offset=item['page_aligned_offset'])
+            self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=item['bytes']//2,
+                                   offset=item['view_offset']).reshape(item['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu' and not value.is_meta,
+                    'authentic contiguous BF16 source view')
+            self.tensors[name]=value
+            self.versions[name]=value._version
+            if name==EMBED:
+                self.readsets[name]={str(token):fd_sha(self.fds[shard],item['absolute_offset']+token*10240,10240)
+                                     for token in (11,353)}
+            else:
+                self.readsets[name]=fd_sha(self.fds[shard],item['absolute_offset'],item['bytes'])
+
+    def mapped_sha(self,name,begin,count):
+        value=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+        try:
+            for offset in range(0,count,1024*1024):
+                value.update(view[begin+offset:begin+min(count,offset+1024*1024)])
+        finally:view.release()
+        return value.hexdigest()
+
+    def finish(self):
+        for name,item in self.plan['tensors'].items():
+            require(self.tensors[name]._version==self.versions[name],'original source tensor version unchanged')
+            if name==EMBED:
+                actual={str(t):self.mapped_sha(name,item['view_offset']+t*10240,10240) for t in (11,353)}
+            else:actual=self.mapped_sha(name,item['view_offset'],item['bytes'])
+            require(actual==self.readsets[name],'original mapped learned payload unchanged: '+name)
+        after={}
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            after[shard]=fingerprint(os.fstat(fd))
+            require(after[shard]==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same selected full source FD unchanged after original operators')
+        return after
+
+def main():
+    global SOURCE_OWNER
+    started=time.monotonic()
+    require(not OUTPUT.exists(),'exclusive output directory');OUTPUT.mkdir(mode=0o700)
+    report={'schema':'zerogravity.original27B.layer0-gdn.public-reference.v1','status':'UNQUALIFIED',
+            'steps':[],'tokens':[11,353],'parameter_profile':'original27B-raw-source-bf16-v1',
+            'full_model_StateRoot_native_exp_GPU_eligible':False}
+    try:
+        require(sys.byteorder=='little','original little-endian source')
+        cpus=[int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'four explicit CPUs');os.sched_setaffinity(0,cpus)
+        require(sorted(os.sched_getaffinity(0))==sorted(cpus),'actual affinity before importing Torch')
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and os.environ.get('USE_HUB_KERNELS')=='0' and
+                not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'declared original CPU reference/no external math interposition')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'exact source metadata plan')
+        plan=metadata(HERE/'source-plan.json')
+        derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        report.update(source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],
+                      source_revision=plan['revision'],snapshot=plan['snapshot'],pre_library_bounds=derived,
+                      native_ResourceArbiter_admission_qualified=False)
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        SOURCE_OWNER=SourceOwner(plan)
+        # No library/model initialization or parameter view precedes the bounds.
+        import torch
+        from torch.nn import functional as F
+        from safetensors.torch import save_file
+        from transformers import Qwen3_5Config, DynamicCache
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        from observer import Observation, ScopedDispatch, info, total_bytes
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and
+                importlib.metadata.version('safetensors')=='0.8.0' and not torch.cuda.is_available(),
+                'exact existing pinned CPU runtime')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.backends.mkldnn.enabled,
+                'original CPU AVX2 policy')
+        require(sha(Path(original.__file__))==MODEL_SHA,'exact original declared architecture source')
+        actual_modules={'modeling_qwen3_5.py':original,
+                        'configuration_qwen3_5.py':sys.modules[Qwen3_5Config.__module__],
+                        'cache_utils.py':sys.modules[DynamicCache.__module__],
+                        'hub_kernels.py':sys.modules['transformers.integrations.hub_kernels']}
+        for name,module in actual_modules.items():
+            require(sha(Path(module.__file__))==plan['runtime_source'][name]['sha256'],
+                    'pinned original module '+name)
+        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
+                     'causal_conv1d_fn','causal_conv1d_update'):
+            function=getattr(original,name);found=False
+            while function is not None:
+                closure=inspect.getclosurevars(function).nonlocals
+                if 'is_new_implementation' in closure:
+                    require(closure['is_new_implementation'] is False and
+                            Path(closure['implementation'].__code__.co_filename).resolve()==Path(original.__file__).resolve(),
+                            'refuse optional non-original package implementation: '+name)
+                    found=True
+                function=getattr(function,'__wrapped__',None)
+            require(found,'authentic original public function wrapper: '+name)
+        config=metadata(SOURCE/'config.json')
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration'] and config['model_type']=='qwen3_5' and
+                config['text_config']['model_type']=='qwen3_5_text','actual27B declared original architecture')
+        cfg=Qwen3_5Config.from_dict(config).text_config
+        require((cfg.hidden_size,cfg.linear_num_key_heads,cfg.linear_num_value_heads,cfg.linear_key_head_dim,
+                 cfg.linear_value_head_dim,cfg.linear_conv_kernel_dim)==(5120,16,48,128,128,4),
+                'original27B GDN exact geometry')
+        SOURCE_OWNER.load_views(torch)
+        with torch.device('meta'):
+            gdn=original.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)
+            input_norm=original.Qwen3_5RMSNorm(5120,eps=cfg.rms_norm_eps)
+        parameters={name.removeprefix(PREFIX):value for name,value in SOURCE_OWNER.tensors.items() if name.startswith(PREFIX)}
+        require(set(parameters)==set(dict(gdn.named_parameters())) and len(parameters)==9 and
+                not dict(gdn.named_buffers()),'every original GDN parameter supplied, no random/unfilled fallback')
+        loaded=gdn.load_state_dict(parameters,strict=True,assign=True)
+        require(not loaded.missing_keys and not loaded.unexpected_keys,'strict complete source parameter assignment')
+        input_norm.load_state_dict({'weight':SOURCE_OWNER.tensors[INPUT_NORM]},strict=True,assign=True)
+        gdn.eval();input_norm.eval()
+        parameter_audit={}
+        for name,value in gdn.named_parameters():
+            raw=SOURCE_OWNER.tensors[PREFIX+name]
+            require(value.dtype==torch.bfloat16 and not value.is_meta and value.data_ptr()==raw.data_ptr() and
+                    tuple(value.shape)==tuple(raw.shape),'exact raw BF16 original parameter, no cast/copy: '+name)
+            parameter_audit[name]={**info(value),'pointer':value.data_ptr(),'version':value._version,
+                                   'source_readset_sha256':SOURCE_OWNER.readsets[PREFIX+name]}
+        require(input_norm.weight.data_ptr()==SOURCE_OWNER.tensors[INPUT_NORM].data_ptr() and
+                input_norm.weight.dtype==torch.bfloat16 and gdn.norm.weight.dtype==torch.bfloat16 and
+                gdn.A_log.dtype==gdn.dt_bias.dtype==torch.bfloat16,
+                'original BF16 gamma/A_log/dt_bias/inputnorm stay untouched')
+        baseline={};observations=[];hooks=[];active=[None]
+        def bits(value):return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
+        def same(a,b,message):
+            require(a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bits(a)==bits(b),message)
+        def projection_hook(name):
+            def hook(module,args,result):
+                if active[0] is not None:
+                    active[0].tensor('projection.'+name+'.input',args[0])
+                    active[0].tensor('projection.'+name+'.output',result)
+            return hook
+        def kept(name,value):
+            count=value.numel()*value.element_size()
+            require(total_bytes[0]+count<=128*1024**2,'bounded combined plain/observed tensor payload')
+            baseline[name]=value.detach().contiguous().clone();total_bytes[0]+=count
+        try:
+            kept('source.A_log',gdn.A_log)
+            kept('source.dt_bias',gdn.dt_bias)
+            kept('source.norm.weight',gdn.norm.weight)
+            with torch.inference_mode():
+                for mode in ('plain','observed'):
+                    cache=DynamicCache(config=cfg)
+                    require(len(cache.layers)==64 and not cache.has_previous_state(0,0),'real original cache starts empty')
+                    if mode=='observed':
+                        for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
+                            hooks.append(gdn.get_submodule(name).register_forward_hook(projection_hook(name)))
+                    for step,token in enumerate((11,353)):
+                        embedding=F.embedding(torch.tensor([[token]],dtype=torch.int64),SOURCE_OWNER.tensors[EMBED])
+                        normalized=input_norm(embedding)
+                        require(embedding.shape==normalized.shape==(1,1,5120),'actual trained original embedding/InputNorm')
+                        require(cache.has_previous_state(0,0) is (step==1),'real initial then cached source dispatch')
+                        if mode=='plain':
+                            output=gdn(normalized,cache_params=cache)
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                kept(f'step-{step}.{name}',value)
+                        else:
+                            obs=Observation(step,gdn);observations.append(obs);active[0]=obs
+                            obs.tensor('prefix.embedding',embedding);obs.tensor('prefix.input_norm',normalized)
+                            require(sys.gettrace() is None,'do not replace an existing tracer')
+                            previous=sys.gettrace();mode_observer=ScopedDispatch(obs);obs.mode=mode_observer
+                            try:
+                                sys.settrace(obs.trace)
+                                with mode_observer:
+                                    output=gdn(normalized,cache_params=cache)
+                            finally:
+                                sys.settrace(previous);obs.mode=None;active[0]=None
+                            obs.tensor('cache.after.conv',cache.layers[0].conv_states[0])
+                            obs.tensor('cache.after.state',cache.layers[0].recurrent_states[0])
+                            obs.tensor('module.output',output)
+                            directory=OUTPUT/f'step-{step}';directory.mkdir()
+                            archive=directory/'actual-gdn.safetensors'
+                            save_file(obs.tensors,str(archive),metadata={'scope':'actual original27B layer0 frames; prevalidation only'})
+                            for name,value in obs.tensors.items():obs.meta[name]['raw_sha256']=digest(bits(value))
+                            record={'step':step,'teacher_token':token,'actual_calls':dict(obs.calls),'events':obs.events,
+                                    'dispatch':obs.dispatch,'tensors':obs.meta,'payload_bytes':obs.bytes,
+                                    'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
+                                    'plain_observed_all_outputs_and_state_exact':False}
+                            write(directory/'prevalidation.json',record)
+                            phase='chunk' if step==0 else 'recurrent'
+                            expected={'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,
+                                      'l2norm':2,'torch_'+phase+'_gated_delta_rule':1,
+                                      'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1}
+                            require(dict(obs.calls)==expected,'actual unchanged original initial/cached call schedule')
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                same(baseline[f'step-{step}.{name}'],value,'plain/observed complete actual '+name)
+                            same(obs.tensors[phase+'.return.actual_return_1'],cache.layers[0].recurrent_states[0],
+                                 'original public returned state equals real original cache')
+                            if step:
+                                same(observations[0].tensors['chunk.return.actual_return_1'],
+                                     obs.tensors['recurrent.input.initial_state'],'actual initial return supplies real cached prior')
+                            record['plain_observed_all_outputs_and_state_exact']=True
+                            report['steps'].append(record)
+                            total_bytes[0]+=obs.bytes
+                        for layer in cache.layers[1:]:
+                            require(not getattr(layer,'is_conv_states_initialized',[]) or
+                                    not any(layer.is_conv_states_initialized),'no unrelated layer state allocated')
+        finally:
+            active[0]=None
+            for hook in hooks:hook.remove()
+            require(sys.gettrace() is None,'original tracer restored')
+        save_file(baseline,str(OUTPUT/'plain.safetensors'),metadata={'scope':'genuine unobserved two-teacher module outputs/cache'})
+        prior=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prior['source_repo']==plan['repository'] and prior['source_revision']==plan['revision'] and
+                prior['snapshot']==plan['snapshot'],'independent original27B prefix reference source binding')
+        prefix_checks=[]
+        for step,token in enumerate((11,353)):
+            old=prior['steps'][step]
+            require(old['token']==token,'independent prefix teacher binding')
+            for stage,actual in [('embedding',baseline[f'step-{step}.embedding']),
+                                 ('normalized',baseline[f'step-{step}.input_norm']),
+                                 ('projected',observations[step].tensors['projection.in_proj_qkv.output'])]:
+                row=old[stage];expected=read(Path('/prefix-reference')/row['file'],pin=row)
+                widened=b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',bits(actual)))
+                require(actual.dtype==torch.bfloat16 and len(widened)==row['bytes'] and widened==expected,
+                        'all prior genuine original27B prefix words: '+stage)
+                prefix_checks.append({'step':step,'stage':stage,'words':actual.numel(),'bit_exact':True,'prior':row})
+        for name,value in gdn.named_parameters():
+            require(value.data_ptr()==parameter_audit[name]['pointer'] and value._version==parameter_audit[name]['version'],
+                    'original source parameter storage/version unchanged')
+        after=SOURCE_OWNER.finish()
+        maps=Path('/proc/self/maps').read_text()
+        require(len(maps.encode())<=2*1024**2 and '/nix/store/' not in maps,'bounded actual container mapping closure')
+        objects=sorted({line.split(None,5)[5] for line in maps.splitlines() if len(line.split(None,5))==6 and
+                        line.split(None,5)[5].startswith('/') and '.so' in Path(line.split(None,5)[5]).name})
+        require(len(objects)<=256,'bounded actual loaded-library count')
+        dsos={};total_dso=0
+        for path in objects:
+            size=Path(path).stat().st_size;total_dso+=size
+            require(size<=2*1024**3 and total_dso<=16*1024**3 and ' (deleted)' not in path,'bounded actual mapped provider closure')
+            dsos[path]={'bytes':size,'sha256':sha(path)}
+        report.update(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_FDs_before=SOURCE_OWNER.before,source_FDs_after=after,source_readsets=SOURCE_OWNER.readsets,
+            all_selected_FDs_and_mapped_parameters_unchanged=True,all_source_parameters_BF16=True,
+            parameter_audit=parameter_audit,source_owner_retained_until_owned_process_exit=True,
+            norm_profile={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+                          'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+                          'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16',
+                          'epsilon':gdn.norm.variance_epsilon,'activation':gdn.norm.activation},
+            payload_bytes=total_bytes[0],
+            plain_archive={'bytes':(OUTPUT/'plain.safetensors').stat().st_size,'sha256':sha(OUTPUT/'plain.safetensors')},
+            hooks_removed=len(hooks),runtime={'torch':torch.__version__,'torch_git':torch.version.git_version,
+                'transformers':importlib.metadata.version('transformers'),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
+                'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
+                'affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'modeling_sha256':MODEL_SHA,
+                'mapped_shared_objects':dsos,'proc_status':Path('/proc/self/status').read_text(),
+                'proc_cgroup':Path('/proc/self/cgroup').read_text(),'proc_mountinfo':Path('/proc/self/mountinfo').read_text()},
+            scope='Original27B private layer0 embedding/InputNorm/GDN two-teacher public reference only; all raw source BF16. No full-model logits/112 StateRoots, native exp/provider allocation eligibility, production memory admission, GPU or performance.')
+        report['prior_prefix_checks']=prefix_checks
+        require(report['payload_bytes']<=128*1024**2,'bounded total actual plain/observed payload')
+    except BaseException as error:
+        report['failure']=repr(error)
+        raise
+    finally:
+        report['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+        write(OUTPUT/'manifest.json',report)
+    print('ORIGINAL27B_LAYER0_GDN_CAPTURE_ONLY',flush=True)
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/observer.py
@@ -0,0 +1,249 @@
+"""Original frame clones and exactly-once selected public-op observation only."""
+import collections
+import pathlib
+import sys
+import torch
+from torch.utils._python_dispatch import TorchDispatchMode
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+from support import require
+model_source = pathlib.Path(official.__file__).resolve()
+MAX_TENSOR_BYTES = 128 * 1024 * 1024
+total_bytes = [0]
+
+def info(value):
+    return {"dtype":str(value.dtype),"shape":list(value.shape),"stride":list(value.stride()),
+            "contiguous":value.is_contiguous(),"numel":value.numel(),
+            "bytes":value.numel()*value.element_size(),"device":str(value.device)}
+
+class Observation:
+    def __init__(self, step, gdn):
+        self.step = step
+        self.gdn = gdn
+        self.norm = gdn.norm
+        self.tensors = {}
+        self.meta = {}
+        self.events = []
+        self.calls = collections.Counter()
+        self.frames = {}
+        self.points_seen = set()
+        self.bytes = 0
+        self.dispatch = []
+        self.mode = None
+        self.norm_entered = 0
+        self.norm_exited = 0
+
+    def tensor(self, name, value):
+        assert name not in self.tensors, name
+        assert value.device.type == "cpu"
+        self.bytes += value.numel() * value.element_size()
+        assert total_bytes[0] + self.bytes <= MAX_TENSOR_BYTES
+        self.meta[name] = info(value)
+        mode = self.mode
+        previous = mode.suppressed if mode is not None else False
+        if mode is not None:
+            mode.suppressed = True
+        try:
+            self.tensors[name] = value.detach().contiguous().clone()
+        finally:
+            if mode is not None:
+                mode.suppressed = previous
+
+    def observe(self, frame, label, names, returned=None):
+        metadata = {"function": frame.f_code.co_name, "line": frame.f_lineno,
+                    "label": label, "step": self.step, "tensors": []}
+        for name in names:
+            value = frame.f_locals.get(name)
+            if isinstance(value, torch.Tensor):
+                key = label + "." + name
+                self.tensor(key, value)
+                metadata["tensors"].append(key)
+            elif value is None or isinstance(value, (bool, int, float, str)):
+                metadata[name] = value
+            elif isinstance(value, torch.dtype):
+                metadata[name] = str(value)
+        if returned is not None:
+            values = returned if isinstance(returned, tuple) else (returned,)
+            for index, value in enumerate(values):
+                if isinstance(value, torch.Tensor):
+                    key = label + f".actual_return_{index}"
+                    self.tensor(key, value)
+                    metadata["tensors"].append(key)
+                else:
+                    metadata[f"actual_return_{index}"] = value
+        self.events.append(metadata)
+
+    def trace(self, frame, event, arg):
+        if frame.f_code.co_filename != str(model_source):
+            return None
+        name = frame.f_code.co_name
+        if event == "call":
+            if name == "forward" and frame.f_locals.get("self") is self.gdn:
+                self.calls["Qwen3_5GatedDeltaNet.forward"] += 1
+                self.frames[id(frame)] = "gdn"
+                self.observe(frame, "gdn.input", ["hidden_states"])
+                return self.trace
+            if name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                self.calls[name] += 1
+                self.frames[id(frame)] = "conv"
+                self.observe(frame, "conv.input", ["hidden_states", "conv_state", "weight", "bias", "activation"])
+                return self.trace
+            if name == "forward" and frame.f_locals.get("self") is self.norm:
+                self.calls["Qwen3_5RMSNormGated.forward"] += 1
+                self.frames[id(frame)] = "norm"
+                self.observe(frame, "norm.input", ["hidden_states", "gate"])
+                self.tensor("norm.input.weight", self.norm.weight)
+                self.events[-1]["weight"] = "norm.input.weight"
+                self.events[-1]["variance_epsilon"] = self.norm.variance_epsilon
+                self.events[-1]["activation"] = self.norm.activation
+                return self.trace
+            if name not in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "l2norm"}:
+                return None
+            caller = frame.f_back
+            while caller is not None:
+                owner = caller.f_locals.get("self")
+                if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                    if owner.layer_idx != 0:
+                        return None
+                    break
+                caller = caller.f_back
+            else:
+                return None
+            self.calls[name] += 1
+            if name == "l2norm":
+                self.frames[id(frame)] = "l2_" + ("query" if self.calls[name] == 1 else "key")
+                self.observe(frame, self.frames[id(frame)] + ".input", ["x", "dim", "eps"])
+            else:
+                self.frames[id(frame)] = "chunk" if name == "torch_chunk_gated_delta_rule" else "recurrent"
+                self.observe(frame, self.frames[id(frame)] + ".input", ["query", "key", "value", "g", "beta", "initial_state", "chunk_size", "output_final_state", "use_qk_l2norm_in_kernel"])
+            return self.trace
+        if id(frame) not in self.frames:
+            return None
+        phase = self.frames[id(frame)]
+        if event == "return":
+            if phase == "norm":
+                self.observe(frame, "norm.return", ["hidden_states", "gate", "input_dtype", "variance"], arg)
+            elif phase == "gdn":
+                self.observe(frame, "gdn.return", ["core_attn_out", "z", "output"], arg)
+            elif phase == "conv":
+                self.observe(frame, "conv.return", ["hidden_states", "hidden_states_new", "conv_state", "out"], arg)
+            elif phase.startswith("l2_"):
+                self.observe(frame, phase + ".return", ["inv_norm"], arg)
+            else:
+                self.observe(frame, phase + ".return", ["query", "key", "value", "beta", "decay", "initial_dtype", "pad_size", "num_chunks", "last_recurrent_state", "core_attn_out"], arg)
+            del self.frames[id(frame)]
+            return None
+        if event != "line":
+            return self.trace
+        # A line event sees the actual locals BEFORE the named source line.
+        points = CHUNK_POINTS if phase == "chunk" else RECURRENT_POINTS if phase == "recurrent" else NORM_POINTS if phase == "norm" else GDN_POINTS if phase == "gdn" else {}
+        if frame.f_lineno in points:
+            label, names = points[frame.f_lineno]
+            point = (id(frame), label)
+            if point not in self.points_seen:
+                self.points_seen.add(point)
+                self.observe(frame, phase + "." + label, names)
+        return self.trace
+
+
+NORM_POINTS = {
+    228:("core_fp32",["hidden_states","input_dtype"]),
+    230:("mean",["hidden_states","variance","input_dtype"]),
+    231:("normalized_fp32",["hidden_states","variance","input_dtype"]),
+    232:("weighted_bf16",["hidden_states","gate","input_dtype"]),
+    234:("product_fp32",["hidden_states","gate","input_dtype"]),
+}
+
+
+CHUNK_POINTS = {
+    343: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    347: ("qk_normalized", ["query", "key"]),
+    351: ("query_scaled", ["query", "scaling"]),
+    355: ("padded64", ["query", "key", "value", "beta", "decay", "pad_size", "sequence_length"]),
+    364: ("beta_products", ["v_beta", "k_beta"]),
+    379: ("cumulative_decay", ["decay", "cum_decay"]),
+    384: ("pairwise_decay", ["pairwise_decay"]),
+    386: ("intra_chunk_products", ["ut_system", "intra_chunk_attn"]),
+    406: ("triangular_solutions", ["decayed_k_beta", "new_values", "k_cumdecay"]),
+    413: ("initial_state", ["last_recurrent_state"]),
+    418: ("chunk_decayed_inputs", ["query", "key", "chunk_decay", "num_chunks"]),
+    422: ("v_new", ["v_new"]),
+    425: ("fp32_core_before_state_update", ["core_attn_out", "inter_chunk_attn"]),
+}
+RECURRENT_POINTS = {
+    463: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    468: ("qk_normalized", ["query", "key"]),
+    472: ("query_scaled", ["query"]),
+    484: ("decay_and_prior", ["decay_t", "last_recurrent_state"]),
+    486: ("decayed_state", ["last_recurrent_state"]),
+    488: ("memory_prediction", ["q_t", "k_t", "v_t", "beta_t", "kv_mem"]),
+    489: ("delta", ["delta"]),
+    491: ("proposed_state", ["last_recurrent_state"]),
+    494: ("fp32_core", ["core_attn_out"]),
+}
+
+
+GDN_POINTS = {
+    574:("projections",["mixed_qkv","z","a","b","use_precomputed_states"]),
+    617:("pre_beta",["query","key","value","a","b"]),
+    620:("coefficients",["a","b","beta","g"]),
+    624:("repeated",["query","key","value","beta","g"]),
+    653:("core_return",["core_attn_out","last_recurrent_state"]),
+    660:("norm_return",["core_attn_out","z"]),
+}
+
+class ScopedDispatch(TorchDispatchMode):
+    def __init__(self, observation):
+        super().__init__()
+        self.observation = observation
+        self.suppressed = False
+
+    def source(self):
+        frame = sys._getframe()
+        while frame is not None:
+            if frame.f_code.co_filename == str(model_source):
+                owner = frame.f_locals.get("self")
+                if owner is self.observation.norm:
+                    return "norm",frame.f_lineno
+                if owner is self.observation.gdn:
+                    return "gdn",frame.f_lineno
+                if frame.f_code.co_name in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule"):
+                    return frame.f_code.co_name,frame.f_lineno
+            frame=frame.f_back
+        return None,None
+
+    def tree(self, value, label, budget):
+        budget[0]+=1
+        require(budget[0]<=256,"bounded operation argument tree")
+        if isinstance(value,torch.Tensor):
+            self.observation.tensor(label,value)
+            return {"tensor":label}
+        if isinstance(value,(list,tuple)):
+            require(len(value)<=32,"bounded argument sequence")
+            return [self.tree(v,label+"."+str(i),budget) for i,v in enumerate(value)]
+        if isinstance(value,dict):
+            require(len(value)<=32 and all(isinstance(k,str) for k in value),"bounded argument mapping")
+            return {k:self.tree(v,label+"."+k,budget) for k,v in value.items()}
+        if value is None or isinstance(value,(bool,int,float,str)):
+            return value
+        if isinstance(value,(torch.dtype,torch.device,torch.layout,torch.memory_format)):
+            return {"torch_value":str(value)}
+        raise ValueError("unhandled original operation argument: "+type(value).__name__)
+
+    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
+        kwargs={} if kwargs is None else kwargs
+        if self.suppressed:
+            return func(*args,**kwargs)
+        scope,line=self.source()
+        selected = scope=="norm" or scope=="gdn" and line in (617,619) or \
+            scope in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule") and str(func)=="aten.exp.default"
+        if not selected:
+            return func(*args,**kwargs)
+        require(len(self.observation.dispatch)<128,"bounded selected original operations")
+        index=len(self.observation.dispatch);label=f"dispatch.{index:03d}"
+        record={"index":index,"operation":str(func),"schema":str(func._schema),"scope":scope,
+                "source_line":line,"arguments":self.tree(args,label+".args",[0]),
+                "keyword_arguments":self.tree(kwargs,label+".kwargs",[0])}
+        result=func(*args,**kwargs) # EXACT original call once; original result returned.
+        record["result"]=self.tree(result,label+".result",[0])
+        self.observation.dispatch.append(record)
+        return result
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/run_reference.py
@@ -0,0 +1,134 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=256*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    def cancelled(_signal,_frame):raise InterruptedError('owned reference cancelled; retire only owned container/process')
+    signal.signal(signal.SIGTERM,cancelled)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('gdn27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-gdn-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json
@@ -0,0 +1,449 @@
+{
+  "bounds_before_library_or_views": {
+    "CPU_count": 4,
+    "aggregate_window_mapping_bytes": 2774582792,
+    "capture_payload_cap_bytes": 134217728,
+    "configured_container_RAM_bytes": 12884901888,
+    "conservative_operator_working_envelope_bytes": 2147483648,
+    "container_wall_seconds": 600,
+    "control_allowance_bytes": 67108864,
+    "derived_touched_source_cap_bytes": 268435456,
+    "mapped_VA_cap_bytes": 3221225472,
+    "native_ResourceArbiter_admission_qualified": false,
+    "outer_wall_seconds": 2400,
+    "page_size": 4096,
+    "process_AS_cap_bytes": 12884901888,
+    "runtime_VA_allowance_bytes": 6442450944,
+    "runtime_resident_allowance_bytes": 2147483648,
+    "selected_parameter_bytes": 231752128,
+    "touched_source_payload_bytes": 231782848,
+    "whole_artifact_cap_bytes": 268435456
+  },
+  "geometry": {
+    "conv_channels": 10240,
+    "conv_kernel": 4,
+    "head_width": 128,
+    "hidden": 5120,
+    "key_heads": 16,
+    "layers": 64,
+    "rms_epsilon": 1e-06,
+    "value_heads": 48
+  },
+  "metadata": {
+    "config.json": {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    "model.safetensors.index.json": {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    }
+  },
+  "original_parameters": "ALL raw source BF16, including A_log/dt_bias/norm.gamma; no official-loader cast or restored-F32 replacement",
+  "prior_full_source_verification": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "prior_public_prefix": {
+    "bytes": 65417,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+    "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+  },
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "runtime_source": {
+    "cache_utils.py": {
+      "bytes": 104677,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/cache_utils.py",
+      "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c"
+    },
+    "configuration_qwen3_5.py": {
+      "bytes": 8395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/configuration_qwen3_5.py",
+      "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da"
+    },
+    "hub_kernels.py": {
+      "bytes": 44974,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/hub_kernels.py",
+      "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95"
+    },
+    "modeling_qwen3_5.py": {
+      "bytes": 95685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    }
+  },
+  "schema": "zerogravity.original27B.layer0-gdn.source-plan.v1",
+  "selected_shards": {
+    "model-00001-of-00018.safetensors": {
+      "file": {
+        "bytes": 3966730552,
+        "git_blob_verified": false,
+        "name": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
+    },
+    "model-00003-of-00018.safetensors": {
+      "file": {
+        "bytes": 2542796952,
+        "git_blob_verified": false,
+        "name": "model-00003-of-00018.safetensors",
+        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+      },
+      "header_bytes": 144,
+      "header_sha256": "26785014f43e18cd545cebad85d003854b94eb64b90079852ba7efeb8041093c"
+    }
+  },
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "source_files": [
+    {
+      "bytes": 11544,
+      "git_blob_verified": true,
+      "name": "LICENSE",
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+    },
+    {
+      "bytes": 65012,
+      "git_blob_verified": true,
+      "name": "README.md",
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+    },
+    {
+      "bytes": 8952,
+      "git_blob_verified": true,
+      "name": "chat_template.jinja",
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+    },
+    {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    {
+      "bytes": 238,
+      "git_blob_verified": true,
+      "name": "crc32.txt",
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+    },
+    {
+      "bytes": 202,
+      "git_blob_verified": true,
+      "name": "generation_config.json",
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+    },
+    {
+      "bytes": 3353259,
+      "git_blob_verified": true,
+      "name": "merges.txt",
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+    },
+    {
+      "bytes": 3966730552,
+      "git_blob_verified": false,
+      "name": "model-00001-of-00018.safetensors",
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+    },
+    {
+      "bytes": 3043080328,
+      "git_blob_verified": false,
+      "name": "model-00002-of-00018.safetensors",
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+    },
+    {
+      "bytes": 2542796952,
+      "git_blob_verified": false,
+      "name": "model-00003-of-00018.safetensors",
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+    },
+    {
+      "bytes": 3988973152,
+      "git_blob_verified": false,
+      "name": "model-00004-of-00018.safetensors",
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+    },
+    {
+      "bytes": 2099339864,
+      "git_blob_verified": false,
+      "name": "model-00005-of-00018.safetensors",
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00006-of-00018.safetensors",
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00007-of-00018.safetensors",
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00008-of-00018.safetensors",
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00009-of-00018.safetensors",
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00010-of-00018.safetensors",
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00011-of-00018.safetensors",
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00012-of-00018.safetensors",
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00013-of-00018.safetensors",
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00014-of-00018.safetensors",
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00015-of-00018.safetensors",
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+    },
+    {
+      "bytes": 3979564040,
+      "git_blob_verified": false,
+      "name": "model-00016-of-00018.safetensors",
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00017-of-00018.safetensors",
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+    },
+    {
+      "bytes": 3392197344,
+      "git_blob_verified": false,
+      "name": "model-00018-of-00018.safetensors",
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+    },
+    {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    },
+    {
+      "bytes": 390,
+      "git_blob_verified": true,
+      "name": "preprocessor_config.json",
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+    },
+    {
+      "bytes": 12809320,
+      "git_blob_verified": false,
+      "name": "tokenizer.json",
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+    },
+    {
+      "bytes": 17928,
+      "git_blob_verified": true,
+      "name": "tokenizer_config.json",
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+    },
+    {
+      "bytes": 385,
+      "git_blob_verified": true,
+      "name": "video_preprocessor_config.json",
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+    },
+    {
+      "bytes": 6722759,
+      "git_blob_verified": true,
+      "name": "vocab.json",
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+    }
+  ],
+  "status": "SOURCE_HEADER_ONLY_NO_OPERATOR_EXECUTION",
+  "tensors": {
+    "model.language_model.embed_tokens.weight": {
+      "absolute_offset": 152,
+      "bytes": 2542796800,
+      "dtype": "BF16",
+      "mapping_bytes": 2542796952,
+      "name": "model.language_model.embed_tokens.weight",
+      "page_aligned_offset": 0,
+      "shape": [
+        248320,
+        5120
+      ],
+      "shard": "model-00003-of-00018.safetensors",
+      "view_offset": 152
+    },
+    "model.language_model.layers.0.input_layernorm.weight": {
+      "absolute_offset": 45400,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "mapping_bytes": 10584,
+      "name": "model.language_model.layers.0.input_layernorm.weight",
+      "page_aligned_offset": 45056,
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 344
+    },
+    "model.language_model.layers.0.linear_attn.A_log": {
+      "absolute_offset": 55640,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2488,
+      "name": "model.language_model.layers.0.linear_attn.A_log",
+      "page_aligned_offset": 53248,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2392
+    },
+    "model.language_model.layers.0.linear_attn.conv1d.weight": {
+      "absolute_offset": 55736,
+      "bytes": 81920,
+      "dtype": "BF16",
+      "mapping_bytes": 84408,
+      "name": "model.language_model.layers.0.linear_attn.conv1d.weight",
+      "page_aligned_offset": 53248,
+      "shape": [
+        10240,
+        1,
+        4
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.dt_bias": {
+      "absolute_offset": 137656,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2584,
+      "name": "model.language_model.layers.0.linear_attn.dt_bias",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_a.weight": {
+      "absolute_offset": 137752,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_b.weight": {
+      "absolute_offset": 629272,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
+      "page_aligned_offset": 626688,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
+      "absolute_offset": 1120792,
+      "bytes": 104857600,
+      "dtype": "BF16",
+      "mapping_bytes": 104860184,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+      "page_aligned_offset": 1118208,
+      "shape": [
+        10240,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_z.weight": {
+      "absolute_offset": 105978392,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917144,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
+      "page_aligned_offset": 105975808,
+      "shape": [
+        6144,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.norm.weight": {
+      "absolute_offset": 168892952,
+      "bytes": 256,
+      "dtype": "BF16",
+      "mapping_bytes": 2840,
+      "name": "model.language_model.layers.0.linear_attn.norm.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        128
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.out_proj.weight": {
+      "absolute_offset": 168893208,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917400,
+      "name": "model.language_model.layers.0.linear_attn.out_proj.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    }
+  },
+  "weight_bytes": 55563006776
+}
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/support.py
@@ -0,0 +1,110 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def bounds(plan, page):
+    """Integer geometry before importing libraries or allocating parameter maps."""
+    limits = plan['bounds_before_library_or_views']
+    require(page == limits['page_size'] == 4096, 'pinned page geometry')
+    require(len(plan['tensors']) == 11 and len(plan['selected_shards']) == 2, 'exact complete selected roles')
+    mapped, params = 0, 0
+    for name, item in plan['tensors'].items():
+        require(item['dtype'] == 'BF16' and type(item['shape']) is list, 'original BF16 source policy')
+        count = 1
+        for axis in item['shape']:
+            require(type(axis) is int and 0 < axis <= 248320, 'bounded original tensor axis')
+            count *= axis
+        start, length = item['absolute_offset'], item['bytes']
+        require(type(start) is type(length) is int and start >= 8 and length == count*2,
+                'full original typed extent')
+        aligned = start-start%page
+        require(item['page_aligned_offset'] == aligned and item['view_offset'] == start-aligned and
+                item['mapping_bytes'] == length+start-aligned and
+                start+length <= plan['selected_shards'][item['shard']]['file']['bytes'],
+                'exact bounded parameter window; no whole shard mapping')
+        mapped += item['mapping_bytes']
+        if '.linear_attn.' in name:
+            params += length
+    touched = params + 10240 + 20480
+    require(params == limits['selected_parameter_bytes'] == 231752128 and
+            mapped == limits['aggregate_window_mapping_bytes'] == 2774582792 and
+            touched == limits['touched_source_payload_bytes'] == 231782848 and
+            mapped <= limits['mapped_VA_cap_bytes'] == 3*1024**3 and
+            touched <= limits['derived_touched_source_cap_bytes'] == 256*1024**2,
+            'pre-library exact source windows and selected resident payload bound')
+    virtual_envelope=mapped+limits['runtime_VA_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']
+    physical_estimate=touched+limits['runtime_resident_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']+limits['control_allowance_bytes']
+    require(virtual_envelope < limits['process_AS_cap_bytes'] == 12*1024**3 and
+            physical_estimate < limits['configured_container_RAM_bytes'] and
+            limits['configured_container_RAM_bytes'] == 12*1024**3,
+            'finite explicit reference containment; not native physical admission')
+    return {'mapped_bytes':mapped,'selected_parameter_bytes':params,'touched_source_bytes':touched,
+            'virtual_envelope_bytes':virtual_envelope,'physical_envelope_estimate_bytes':physical_estimate,
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    manifest = metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:
+        read(row['path'],pin=row)
+    return sha(HERE/'prepared-manifest.json')
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/test_contracts.py
@@ -0,0 +1,43 @@
+"""ROOT-only source-free geometry/archive fixtures; no library/model import."""
+import copy
+from pathlib import Path
+import json
+import struct
+import tempfile
+import unittest
+from support import HERE, bounds, digest, metadata
+from validate_reference import archive
+
+class Contracts(unittest.TestCase):
+    def test_exact_parameter_windows_and_full_source_kind_refuse(self):
+        plan=metadata(HERE/'source-plan.json')
+        value=bounds(plan,4096)
+        self.assertEqual(value['mapped_bytes'],2774582792)
+        for field in ('absolute_offset','mapping_bytes','view_offset','bytes'):
+            wrong=copy.deepcopy(plan)
+            first=next(iter(wrong['tensors'].values()))
+            first[field]+=1
+            with self.assertRaises(ValueError):bounds(wrong,4096)
+        wrong=copy.deepcopy(plan);next(iter(wrong['tensors'].values()))['dtype']='F32'
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_native_provider_admission_not_inferred_from_containment(self):
+        plan=metadata(HERE/'source-plan.json')
+        self.assertIs(bounds(plan,4096)['native_ResourceArbiter_admission_qualified'],False)
+        with self.assertRaises(ValueError):bounds(plan,8192)
+        wrong=copy.deepcopy(plan);wrong['bounds_before_library_or_views']['runtime_VA_allowance_bytes']=16*1024**3
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_archive_preserves_actual_negative_inf_mask_and_rejects_tail(self):
+        # Transport-only fixture: no candidate result or numerical expectation.
+        payload=bytes.fromhex('000080ff0000008001000000')
+        header=json.dumps({'x':{'dtype':'F32','shape':[3],'data_offsets':[0,12]}}).encode()
+        raw=struct.pack('<Q',len(header))+header+payload
+        with tempfile.TemporaryDirectory() as root:
+            path=Path(root)/'actual.safetensors';path.write_bytes(raw)
+            pin={'bytes':len(raw),'sha256':digest(raw)}
+            self.assertEqual(archive(path,pin)['x']['raw'],payload)
+            path.write_bytes(raw+b'\0')
+            with self.assertRaises(ValueError):archive(path,{'bytes':len(raw)+1,'sha256':digest(raw+b'\0')})
+
+if __name__=='__main__':unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/validate_reference.py
@@ -0,0 +1,251 @@
+"""ROOT ONLY stdlib full-byte/source/lifecycle validation; no tensor arithmetic."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, frozen, metadata, read, require, sha, unique, write
+
+PREFIX='model.language_model.layers.0.linear_attn.'
+
+def source_readsets(plan,report):
+    """ROOT-only actual indexed source bytes, bounded read; no Torch or math."""
+    require(set(report['source_readsets'])==set(plan['tensors']),'all actual source raw readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd));n,=struct.unpack('<Q',os.pread(fd,8,0))
+            require(before[2]==pin['file']['bytes'] and n==pin['header_bytes'] and
+                    digest(os.pread(fd,n,8))==pin['header_sha256'],'same authenticated selected source header')
+            for name,item in plan['tensors'].items():
+                if item['shard']!=shard:continue
+                if name=='model.language_model.embed_tokens.weight':
+                    actual={str(t):fd_sha(fd,item['absolute_offset']+t*10240,10240) for t in (11,353)}
+                else:actual=fd_sha(fd,item['absolute_offset'],item['bytes'])
+                require(actual==report['source_readsets'][name],'independent actual raw source readset: '+name)
+            require(fingerprint(os.fstat(fd))==before,'original FD unchanged during independent source-range validation')
+        finally:os.close(fd)
+
+def archive(path,pin):
+    raw=read(path,128*1024**2,pin);n,=struct.unpack_from('<Q',raw)
+    require(1<n<=8*1024**2 and 8+n<len(raw),'bounded complete raw archive header')
+    head=json.loads(raw[8:8+n],object_pairs_hook=unique);tensors={};spans=[]
+    for name,item in head.items():
+        if name=='__metadata__':continue
+        dtype,shape=item['dtype'],item['shape'];require(dtype in ('F32','BF16'),'original raw archive dtype')
+        count=1
+        require(type(shape) is list and len(shape)<=8,'bounded actual rank')
+        for dim in shape:
+            require(type(dim) is int and 0<dim<=248320,'bounded actual dimension');count*=dim
+        lo,hi=item['data_offsets'];size=2 if dtype=='BF16' else 4
+        require(type(lo) is type(hi) is int and 0<=lo<hi and hi-lo==count*size and 8+n+hi<=len(raw),
+                'complete original dtype/extent geometry')
+        tensors[name]={'dtype':dtype,'shape':shape,'raw':raw[8+n+lo:8+n+hi]};spans.append((lo,hi))
+    cursor=0
+    for lo,hi in sorted(spans):require(lo==cursor,'all disjoint complete archive bytes');cursor=hi
+    require(8+n+cursor==len(raw),'no trailing/unbound archive bytes')
+    return tensors
+
+def tensor(tensors,name,dtype,shape=None):
+    value=tensors[name]
+    require(value['dtype']==dtype and (shape is None or value['shape']==shape),'actual stage dtype/shape: '+name)
+    return value['raw']
+
+def widened(raw):
+    return b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',raw))
+
+def operation(rows,scope,line,name):
+    selected=[row for row in rows if row['scope']==scope and row['source_line']==line and row['operation']==name]
+    require(len(selected)==1,'exact actual selected operator: '+str((scope,line,name)))
+    return selected[0]
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');derived=bounds(plan,4096)
+    invocation=metadata(args.output/'invocation.json');directory=args.output/'captured';report=metadata(directory/'manifest.json')
+    require(invocation['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['freeze_manifest_sha256']==freeze and invocation['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            invocation['exit_code']==0 and invocation['cleanup_errors']==[] and invocation['source_FD_cleanup_errors']==[] and
+            invocation['source_FDs_closed'] is True and invocation.get('owned_container_absent_after_reap') is True,
+            'actual original source/reference/owned container all retired')
+    cleanup=invocation['foreground_cleanup'];logs=invocation['log_cleanup']
+    require(cleanup['leader_reaped'] is True and cleanup['group_absent'] is True and cleanup['exit_code']==0 and
+            cleanup['all_nonzero_signals_before_reap'] is True and cleanup['stage_errors']==[] and
+            logs['pipe_eof'] is logs['all_closed'] is True and logs['total_bytes']==logs['retained_bytes']<=4*1024**2,
+            'original finite owned process/PGID/EOF cleanup and complete evidence')
+    expected={v['name']:(v['bytes'],v['sha256']) for v in plan['source_files']}
+    require(invocation['source_before']==invocation['source_after'] and
+            {v['name']:(v['bytes'],v['sha256']) for v in invocation['source_before']}==expected and
+            len(invocation['source_before'])==len(expected),'all original source files before/after full SHA coverage')
+    require(report['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            report['schema']=='zerogravity.original27B.layer0-gdn.public-reference.v1' and report['tokens']==[11,353] and
+            report['parameter_profile']=='original27B-raw-source-bf16-v1' and
+            report['source_plan_sha256']==sha(HERE/'source-plan.json') and report['source_repo']==plan['repository'] and
+            report['source_revision']==plan['revision'] and report['snapshot']==plan['snapshot'] and
+            report['pre_library_bounds']==derived and report['native_ResourceArbiter_admission_qualified'] is False and
+            report['full_model_StateRoot_native_exp_GPU_eligible'] is False and report['all_source_parameters_BF16'] is True and
+            report['source_owner_retained_until_owned_process_exit'] is True and report['hooks_removed']==5,
+            'authentic private original27B profile and unchanged eligibility guards')
+    require(report['norm_profile']=={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+        'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+        'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16','epsilon':1e-6,'activation':'silu'},
+        'explicit original27B BF16 gamma round profile')
+    require(report['all_selected_FDs_and_mapped_parameters_unchanged'] is True and
+            report['source_FDs_before']==report['source_FDs_after'] and
+            set(report['source_FDs_before'])==set(plan['selected_shards']),'same exact selected original FD owners')
+    source_readsets(plan,report)
+    expected_parameters={name.removeprefix(PREFIX):item for name,item in plan['tensors'].items() if name.startswith(PREFIX)}
+    require(set(report['parameter_audit'])==set(expected_parameters) and len(expected_parameters)==9,
+            'all and only original learned GDN parameters')
+    for name,item in expected_parameters.items():
+        actual=report['parameter_audit'][name]
+        require(actual['dtype']=='torch.bfloat16' and actual['shape']==item['shape'] and
+                actual['device']=='cpu' and actual['contiguous'] is True and actual['bytes']==item['bytes'] and
+                type(actual['pointer']) is int and actual['pointer']>0 and
+                actual['source_readset_sha256']==report['source_readsets'][PREFIX+name],
+                'actual complete original raw parameter binding: '+name)
+    runtime=report['runtime'];require(runtime['torch']=='2.13.0+cu129' and runtime['torch_git']==TORCH_GIT and
+        runtime['transformers']=='5.17.0' and runtime['cpu_capability']=='AVX2' and runtime['threads']==4 and
+        runtime['interop_threads']==1 and runtime['affinity']==invocation['expected_cpus'] and
+        len(set(runtime['affinity']))==4 and runtime['cuda_available'] is False and runtime['modeling_sha256']==MODEL_SHA,
+        'actual pinned unchanged CPU runtime and four CPUs')
+    dsos=runtime['mapped_shared_objects']
+    require(dsos['/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so']['sha256']==
+            '1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab' and
+            dsos['/usr/lib/x86_64-linux-gnu/libm.so.6']['sha256']==
+            'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
+            'actual original exp/CPU/scalar-provider mappings, not native admission')
+    plain=archive(directory/'plain.safetensors',report['plain_archive']);steps=[];payload=sum(len(t['raw']) for t in plain.values())
+    for name,shape in [('A_log',[48]),('dt_bias',[48]),('norm.weight',[128])]:
+        require(digest(tensor(plain,'source.'+name,'BF16',shape))==report['source_readsets'][PREFIX+name],
+                'all original BF16 coefficient/gamma source bytes in reference')
+    require(len(report['steps'])==2,'both true source teachers')
+    for step,record in enumerate(report['steps']):
+        phase='chunk' if step==0 else 'recurrent'
+        require(record['step']==step and record['teacher_token']==[11,353][step] and
+                record['plain_observed_all_outputs_and_state_exact'] is True and record['actual_calls']=={
+                    'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,'l2norm':2,
+                    'torch_'+phase+'_gated_delta_rule':1,'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1},
+                'one original chunk64 first, one actual cached call second')
+        calls=[event for event in record['events'] if event['label']==phase+'.input']
+        require(len(calls)==1 and calls[0]['output_final_state'] is True and
+                calls[0]['use_qk_l2norm_in_kernel'] is True and
+                (step==1 or calls[0]['initial_state'] is None and calls[0]['chunk_size']==64),
+                'actual original phase/initial-state/padded-chunk contract')
+        tensors=archive(directory/f'step-{step}/actual-gdn.safetensors',
+                        {'bytes':record['archive_bytes'],'sha256':record['archive_sha256']});steps.append(tensors)
+        before=metadata(directory/f'step-{step}/prevalidation.json')
+        require(before==dict(record,plain_observed_all_outputs_and_state_exact=False),
+                'raw original diagnostics retained before semantic acceptance')
+        require(set(tensors)==set(record['tensors']),'all original raw observed stages bound')
+        for name,item in tensors.items():
+            meta=record['tensors'][name]
+            require(meta['raw_sha256']==digest(item['raw']) and meta['bytes']==len(item['raw']) and
+                    meta['shape']==item['shape'] and meta['dtype']==('torch.bfloat16' if item['dtype']=='BF16' else 'torch.float32') and
+                    meta['device']=='cpu','actual complete observer payload: '+name)
+            payload+=len(item['raw'])
+        require(sum(len(t['raw']) for t in tensors.values())==record['payload_bytes'],'all actual per-step payload counted')
+        for original,observed,dtype,shape in [('embedding','prefix.embedding','BF16',[1,1,5120]),
+                ('input_norm','prefix.input_norm','BF16',[1,1,5120]),('output','module.output','BF16',[1,1,5120]),
+                ('conv','cache.after.conv','BF16',[1,10240,4]),('state','cache.after.state','F32',[1,48,128,128])]:
+            require(tensor(plain,f'step-{step}.'+original,dtype,shape)==tensor(tensors,observed,dtype,shape),
+                    'all plain/observed source outputs and full actual state: '+original)
+        require(tensor(tensors,phase+'.return.actual_return_1','F32',[1,48,128,128])==
+                tensor(tensors,'cache.after.state','F32',[1,48,128,128]),'actual original returned state equals actual cache')
+        require(tensor(tensors,phase+'.return.actual_return_0','BF16',[1,1,48,128])==
+                tensor(tensors,'norm.input.hidden_states','BF16',[48,128]),'original core return feeds actual gated norm')
+        for name in ('query','key'):
+            before=tensor(tensors,'gdn.pre_beta.'+name,'BF16',[1,1,16,128]);after=tensor(tensors,'gdn.repeated.'+name,'BF16',[1,1,48,128])
+            require(after==b''.join(before[h*256:(h+1)*256]*3 for h in range(16)),
+                    'actual original Q/K repeat_interleave(3), no inferred permutation')
+        rows=record['dispatch'];coef=[v for v in rows if v['scope']=='gdn']
+        require([v['operation'] for v in coef]==['aten.sigmoid.default','aten.to.dtype','aten.exp.default','aten.neg.default',
+                'aten.to.dtype','aten.add.Tensor','aten.softplus.default','aten.mul.Tensor'] and
+                [v['source_line'] for v in coef]==[617]+[619]*7,'actual BF16 A_log .float/coefficient source order')
+        def op_raw(row,which='result',index=0):
+            ref=row[which][index] if which=='arguments' else row[which]
+            require(type(ref) is dict and set(ref)=={'tensor'},'actual selected tensor result/argument')
+            return tensors[ref['tensor']]['raw']
+        require(op_raw(coef[0],'arguments')==tensor(tensors,'projection.in_proj_b.output','BF16',[1,1,48]) and
+                op_raw(coef[0])==tensor(tensors,'gdn.coefficients.beta','BF16',[1,1,48]) and
+                op_raw(coef[1],'arguments')==tensor(plain,'source.A_log','BF16',[48]) and
+                op_raw(coef[1])==widened(tensor(plain,'source.A_log','BF16',[48])) and
+                op_raw(coef[4],'arguments')==tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48]) and
+                op_raw(coef[4])==widened(tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48])) and
+                op_raw(coef[5],'arguments',1)==tensor(plain,'source.dt_bias','BF16',[48]) and
+                op_raw(coef[-1])==tensor(tensors,phase+'.input.g','F32',[1,1,48]),
+                'genuine source coefficient producer inputs/results, no reference lookup')
+        beta=tensor(tensors,phase+'.input.beta','BF16',[1,1,48])
+        require(all(0.0<=struct.unpack('<f',(word<<16).to_bytes(4,'little'))[0]<=1.0 for word, in struct.iter_unpack('<H',beta)) and
+                all(value<=0.0 for value, in struct.iter_unpack('<f',tensor(tensors,phase+'.input.g','F32',[1,1,48]))),
+                'actual beta and original negative log-decay ranges')
+        norm={}
+        require(len([row for row in rows if row['scope']=='norm'])==12,'complete unchanged original gated norm operation count')
+        for stage,op,line,dtype,shape in [
+            ('core','aten.to.dtype',227,'F32',[48,128]),('squares','aten.pow.Tensor_Scalar',228,'F32',[48,128]),
+            ('mean','aten.mean.dim',228,'F32',[48,1]),('denominator','aten.add.Tensor',230,'F32',[48,1]),
+            ('inverse','aten.rsqrt.default',230,'F32',[48,1]),('normalized','aten.mul.Tensor',230,'F32',[48,128]),
+            ('normalized_bf16','aten.to.dtype',231,'BF16',[48,128]),('weighted_bf16','aten.mul.Tensor',231,'BF16',[48,128]),
+            ('gate','aten.to.dtype',232,'F32',[48,128]),('silu','aten.silu.default',232,'F32',[48,128]),
+            ('product','aten.mul.Tensor',232,'F32',[48,128]),('return','aten.to.dtype',234,'BF16',[48,128])]:
+            actual=operation(rows,'norm',line,op);name=actual['result']['tensor'];norm[stage]=tensor(tensors,name,dtype,shape)
+        require(tensor(tensors,'norm.input.weight','BF16',[128])==tensor(plain,'source.norm.weight','BF16',[128]) and
+                norm['weighted_bf16']==tensor(tensors,'norm.weighted_bf16.hidden_states','BF16',[48,128]) and
+                norm['return']==tensor(tensors,'norm.return.actual_return_0','BF16',[48,128]),
+                'actual source BF16 gamma product rounds BEFORE FP32 SiLU gate')
+        exp=[v for v in rows if v['scope']=='torch_'+phase+'_gated_delta_rule']
+        require([v['source_line'] for v in exp]==([381,386,413,414,415] if step==0 else [483]) and
+                all(v['operation']=='aten.exp.default' for v in exp),'all actual original initial/cached exponentials')
+        masked=exp[0]['arguments'][0]['tensor'] if step==0 else None
+        for name,item in tensors.items():
+            if name==masked:
+                require(item['shape']==[1,48,1,64,64] and item['dtype']=='F32','actual upper-triangle exp input geometry')
+                for index,(word,) in enumerate(struct.iter_unpack('<I',item['raw'])):
+                    i,j=(index//64)%64,index%64
+                    require(word==0xff800000 if i<j else word&0x7f800000!=0x7f800000,
+                            'ONLY source strictly-upper exp mask may contain canonical negative Inf')
+            else:
+                mask=0x7f80 if item['dtype']=='BF16' else 0x7f800000
+                fmt='<H' if item['dtype']=='BF16' else '<I'
+                require(all(word&mask!=mask for word, in struct.iter_unpack(fmt,item['raw'])),'all other actual raw words finite: '+name)
+        if step:
+            require(tensor(steps[0],'chunk.return.actual_return_1','F32',[1,48,128,128])==
+                    tensor(tensors,'recurrent.input.initial_state','F32',[1,48,128,128]) and
+                    op_raw(exp[0])==tensor(tensors,'recurrent.decay_and_prior.decay_t','F32',[1,48,1,1]),
+                    'actual full initial chunk prior and actual cached exp(g), never zero replacement')
+        else:
+            for name in ('query','key','value','beta','decay'):
+                value=tensors['chunk.padded64.'+name];width=128 if name in ('query','key','value') else 1
+                require(value['dtype']=='F32' and value['shape']==([1,48,64,128] if width==128 else [1,48,64]),
+                        'actual first-token source padded64 layout')
+                for head in range(48):
+                    require(value['raw'][(head*64+1)*width*4:(head+1)*64*width*4]==bytes(63*width*4),
+                            'all63 genuine source padding lanes canonical positive zero')
+    require(payload==report['payload_bytes'] and payload<=128*1024**2,'all actual plain/observed payload bytes counted')
+    require(len(report['prior_prefix_checks'])==6 and all(v['bit_exact'] is True for v in report['prior_prefix_checks']),
+            'independent previously accepted original27B embedding/norm/QKV prefix closure')
+    prior=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    for step in range(2):
+        for stage,name,dataset in [('embedding',f'step-{step}.embedding',plain),
+                                   ('normalized',f'step-{step}.input_norm',plain),
+                                   ('projected','projection.in_proj_qkv.output',steps[step])]:
+            row=prior['steps'][step][stage]
+            require(widened(tensor(dataset,name,'BF16'))==read(Path(plan['prior_public_prefix']['path']).parent/row['file'],pin=row),
+                    'independent every genuine original prefix word')
+    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}=={
+        'manifest.json','plain.safetensors','step-0/actual-gdn.safetensors','step-0/prevalidation.json',
+        'step-1/actual-gdn.safetensors','step-1/prevalidation.json'},'all captured source/byte artifacts accounted')
+    total_artifacts=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no unknown artifact symlink')
+        if path.is_file():total_artifacts+=path.stat().st_size
+    require(total_artifacts<=256*1024**2-128*1024,'independent whole attempt bound including reserved validation record')
+    require(frozen()==freeze,'all immutable source preparation unchanged')
+    result={'status':'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY','teachers':[11,353],
+            'heads':48,'cached_input_words_available':804960,'cached_output_words_available':798720,
+            'source_plan_sha256':sha(HERE/'source-plan.json'),'manifest_sha256':sha(directory/'manifest.json'),
+            'invocation_sha256':sha(args.output/'invocation.json'),'payload_bytes':payload,
+            'full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified':False}
+    write(args.output/'validation.json',result);print(json.dumps(result,sort_keys=True))
+
+if __name__=='__main__':main()

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/frozen-prepared/NEW_PREPARATION.patch

Original UTF-8 bytes: 107301; SHA256 `33d829e90187836239b7ad69a22416eafa0970353b0ff2448ea90e32a36013bb`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/README.md
@@ -0,0 +1,85 @@
+# Original27B layer0 GDN public reference — prepared only
+
+Author: spine_review. Parent plan: model-execution.
+
+- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
+  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
+  original module; no architecture inference from the model name.
+- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
+  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
+  real cached recurrence using that first return through original DynamicCache.
+  No whole-model constructor/traversal,55GB materialization or112-domain claim.
+- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
+  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
+  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
+  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
+  official-loader blanket cast or conversion from GGUF learned values.
+- Genuine original source public forward called unchanged once per invocation.
+  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
+  fallback must identify original Torch code or refuse. No function replacement,
+  scalar arithmetic port, reconstructed state or native exp eligibility.
+- Independent fresh plain/observed caches, same immutable weights. Every token's
+  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
+  must be bit-identical. True initial return must equal cached prior in all words.
+  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
+- Original call/line/return frame hooks reuse accepted source line maps. Capture
+  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
+  actual conv operands/return, projection input/output, and gated RMS stages.
+  Scoped TorchDispatchMode records original coefficient operations, five real
+  initial exps, cached exp(g) and norm args/results; forwards each original op
+  once and returns its actual result. Observer clones are suppressed.
+- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
+  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
+- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
+  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
+  captured values and published outputs must be finite. No tolerance widening.
+- Six actual embedding/inputnorm/QKV outputs also match the independently
+  accepted original27B public prefix reference in every word.
+
+## Bounds and source ownership
+
+- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
+  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
+  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
+- Each COW window maps page-aligned offset and only leading alignment+complete
+  original tensor bytes. Full embedding raw view2.543GB VA retains its original
+  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
+- Before library initialization, derive separate virtual and physical-envelope
+  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
+  captured payload,2GiB runtime resident allowance,64MiB control allowance.
+  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
+ 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
+  These estimates/OS containment do NOT prove native ResourceArbiter admission
+  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
+- ROOT alone hashes ALL18 original shards/all source files before and after via
+  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
+  verified before importing Torch and rechecked after; COW parameter readsets,
+  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
+  owners remain until the owned process exits; no premature mmap close.
+- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
+  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
+  All postspawn paths use original independent cleanup stages and EOF proof.
+  No next provider/process may start after cleanup/evidence failure.
+- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
+  that a configured limit or historical RSS is native physical admission.
+
+## ROOT execution, after independent source review
+
+All directories fresh. ROOT owns every test/container/operator execution.
+
+```bash
+python3 "$RUN/test_contracts.py"
+python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
+python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
+```
+
+Output: untouched plain archive +two prevalidation observed archives, full
+source/runtime/parameter/cleanup records. Failure preserves actual metadata and
+raw frames before semantic validation. Final validator compares all raw inputs,
+intermediates, initial/cached prior, source gamma rounding and transparency.
+H48 cached inputs804960 words and outputs798720 words become reference artifacts
+only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
+future source-kind admission and native H48 replay require separate review.
+
+No author new source payload hashes, model/provider/operator/library/container,
+Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/TODO.md
@@ -0,0 +1,16 @@
+Parent: `.agents/plans/model-execution/`. Owner: spine_review.
+
+- [x] Read actual original config, index and bounded selected source headers.
+- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
+- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
+- [x] Prepare original module/meta strict assignment, real DynamicCache, two
+  teacher plain/observed reference, original intermediate/dispatch observers.
+- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
+  and independent raw/state/source/provider/cleanup validator; no math replacement.
+- [x] Complete author static self-review/AST/freeze; no runtime or tests.
+- [ ] ROOT independent full source review.
+- [ ] ROOT source-free fixtures, actual original public reference and validator.
+- [ ] Separate H48 source-kind/native cached slice qualification after reference.
+
+No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
+exp/GPU admission from this source-only preparation.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/capture.py
@@ -0,0 +1,315 @@
+"""ROOT ONLY original27B layer0 GDN module; never a full-model constructor."""
+import gc
+import importlib.metadata
+import inspect
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import struct
+import sys
+import time
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, metadata, read, require, sha, unique, write
+
+SOURCE = Path('/source')
+OUTPUT = Path('/output/captured')
+EMBED = 'model.language_model.embed_tokens.weight'
+INPUT_NORM = 'model.language_model.layers.0.input_layernorm.weight'
+PREFIX = 'model.language_model.layers.0.linear_attn.'
+SOURCE_OWNER = None # Original FD/map/storage owner retained until owned process exit.
+
+class SourceOwner:
+    def __init__(self, plan):
+        self.plan, self.fds, self.headers, self.before = plan, {}, {}, {}
+        self.maps, self.tensors, self.readsets, self.versions = {}, {}, {}, {}
+        for name,pin in plan['metadata'].items():
+            read(SOURCE/name,pin=pin)
+        self.index=metadata(SOURCE/'model.safetensors.index.json')
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(SOURCE/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            self.fds[shard]=fd
+            self.before[shard]=fingerprint(os.fstat(fd))
+            require(os.fstat(fd).st_size==pin['file']['bytes'] and
+                    fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],
+                    'complete same opened original selected shard before library initialization')
+            n,=struct.unpack('<Q',os.pread(fd,8,0))
+            raw=os.pread(fd,n,8)
+            require(n==pin['header_bytes'] and len(raw)==n and n<=8*1024*1024 and
+                    digest(raw)==pin['header_sha256'],'exact original selected shard header')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+
+    def load_views(self,torch):
+        for name,item in self.plan['tensors'].items():
+            shard=item['shard'];begin,header=self.headers[shard];entry=header[name]
+            require(self.index['weight_map'][name]==shard and entry['dtype']=='BF16' and
+                    entry['shape']==item['shape'] and begin+entry['data_offsets'][0]==item['absolute_offset'] and
+                    entry['data_offsets'][1]-entry['data_offsets'][0]==item['bytes'],
+                    'actual exact source role/name/geometry/full-byte extent')
+            # COW protects source files; every window is only its exact parameter
+            # plus the leading alignment bytes, not a whole-shard mapping.
+            mapping=mmap.mmap(self.fds[shard],item['mapping_bytes'],access=mmap.ACCESS_COPY,
+                              offset=item['page_aligned_offset'])
+            self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=item['bytes']//2,
+                                   offset=item['view_offset']).reshape(item['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu' and not value.is_meta,
+                    'authentic contiguous BF16 source view')
+            self.tensors[name]=value
+            self.versions[name]=value._version
+            if name==EMBED:
+                self.readsets[name]={str(token):fd_sha(self.fds[shard],item['absolute_offset']+token*10240,10240)
+                                     for token in (11,353)}
+            else:
+                self.readsets[name]=fd_sha(self.fds[shard],item['absolute_offset'],item['bytes'])
+
+    def mapped_sha(self,name,begin,count):
+        value=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+        try:
+            for offset in range(0,count,1024*1024):
+                value.update(view[begin+offset:begin+min(count,offset+1024*1024)])
+        finally:view.release()
+        return value.hexdigest()
+
+    def finish(self):
+        for name,item in self.plan['tensors'].items():
+            require(self.tensors[name]._version==self.versions[name],'original source tensor version unchanged')
+            if name==EMBED:
+                actual={str(t):self.mapped_sha(name,item['view_offset']+t*10240,10240) for t in (11,353)}
+            else:actual=self.mapped_sha(name,item['view_offset'],item['bytes'])
+            require(actual==self.readsets[name],'original mapped learned payload unchanged: '+name)
+        after={}
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            after[shard]=fingerprint(os.fstat(fd))
+            require(after[shard]==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same selected full source FD unchanged after original operators')
+        return after
+
+def main():
+    global SOURCE_OWNER
+    started=time.monotonic()
+    require(not OUTPUT.exists(),'exclusive output directory');OUTPUT.mkdir(mode=0o700)
+    report={'schema':'zerogravity.original27B.layer0-gdn.public-reference.v1','status':'UNQUALIFIED',
+            'steps':[],'tokens':[11,353],'parameter_profile':'original27B-raw-source-bf16-v1',
+            'full_model_StateRoot_native_exp_GPU_eligible':False}
+    try:
+        require(sys.byteorder=='little','original little-endian source')
+        cpus=[int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'four explicit CPUs');os.sched_setaffinity(0,cpus)
+        require(sorted(os.sched_getaffinity(0))==sorted(cpus),'actual affinity before importing Torch')
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and os.environ.get('USE_HUB_KERNELS')=='0' and
+                not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'declared original CPU reference/no external math interposition')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'exact source metadata plan')
+        plan=metadata(HERE/'source-plan.json')
+        derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        report.update(source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],
+                      source_revision=plan['revision'],snapshot=plan['snapshot'],pre_library_bounds=derived,
+                      native_ResourceArbiter_admission_qualified=False)
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        SOURCE_OWNER=SourceOwner(plan)
+        # No library/model initialization or parameter view precedes the bounds.
+        import torch
+        from torch.nn import functional as F
+        from safetensors.torch import save_file
+        from transformers import Qwen3_5Config, DynamicCache
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        from observer import Observation, ScopedDispatch, info, total_bytes
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and
+                importlib.metadata.version('safetensors')=='0.8.0' and not torch.cuda.is_available(),
+                'exact existing pinned CPU runtime')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.backends.mkldnn.enabled,
+                'original CPU AVX2 policy')
+        require(sha(Path(original.__file__))==MODEL_SHA,'exact original declared architecture source')
+        actual_modules={'modeling_qwen3_5.py':original,
+                        'configuration_qwen3_5.py':sys.modules[Qwen3_5Config.__module__],
+                        'cache_utils.py':sys.modules[DynamicCache.__module__],
+                        'hub_kernels.py':sys.modules['transformers.integrations.hub_kernels']}
+        for name,module in actual_modules.items():
+            require(sha(Path(module.__file__))==plan['runtime_source'][name]['sha256'],
+                    'pinned original module '+name)
+        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
+                     'causal_conv1d_fn','causal_conv1d_update'):
+            function=getattr(original,name);found=False
+            while function is not None:
+                closure=inspect.getclosurevars(function).nonlocals
+                if 'is_new_implementation' in closure:
+                    require(closure['is_new_implementation'] is False and
+                            Path(closure['implementation'].__code__.co_filename).resolve()==Path(original.__file__).resolve(),
+                            'refuse optional non-original package implementation: '+name)
+                    found=True
+                function=getattr(function,'__wrapped__',None)
+            require(found,'authentic original public function wrapper: '+name)
+        config=metadata(SOURCE/'config.json')
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration'] and config['model_type']=='qwen3_5' and
+                config['text_config']['model_type']=='qwen3_5_text','actual27B declared original architecture')
+        cfg=Qwen3_5Config.from_dict(config).text_config
+        require((cfg.hidden_size,cfg.linear_num_key_heads,cfg.linear_num_value_heads,cfg.linear_key_head_dim,
+                 cfg.linear_value_head_dim,cfg.linear_conv_kernel_dim)==(5120,16,48,128,128,4),
+                'original27B GDN exact geometry')
+        SOURCE_OWNER.load_views(torch)
+        with torch.device('meta'):
+            gdn=original.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)
+            input_norm=original.Qwen3_5RMSNorm(5120,eps=cfg.rms_norm_eps)
+        parameters={name.removeprefix(PREFIX):value for name,value in SOURCE_OWNER.tensors.items() if name.startswith(PREFIX)}
+        require(set(parameters)==set(dict(gdn.named_parameters())) and len(parameters)==9 and
+                not dict(gdn.named_buffers()),'every original GDN parameter supplied, no random/unfilled fallback')
+        loaded=gdn.load_state_dict(parameters,strict=True,assign=True)
+        require(not loaded.missing_keys and not loaded.unexpected_keys,'strict complete source parameter assignment')
+        input_norm.load_state_dict({'weight':SOURCE_OWNER.tensors[INPUT_NORM]},strict=True,assign=True)
+        gdn.eval();input_norm.eval()
+        parameter_audit={}
+        for name,value in gdn.named_parameters():
+            raw=SOURCE_OWNER.tensors[PREFIX+name]
+            require(value.dtype==torch.bfloat16 and not value.is_meta and value.data_ptr()==raw.data_ptr() and
+                    tuple(value.shape)==tuple(raw.shape),'exact raw BF16 original parameter, no cast/copy: '+name)
+            parameter_audit[name]={**info(value),'pointer':value.data_ptr(),'version':value._version,
+                                   'source_readset_sha256':SOURCE_OWNER.readsets[PREFIX+name]}
+        require(input_norm.weight.data_ptr()==SOURCE_OWNER.tensors[INPUT_NORM].data_ptr() and
+                input_norm.weight.dtype==torch.bfloat16 and gdn.norm.weight.dtype==torch.bfloat16 and
+                gdn.A_log.dtype==gdn.dt_bias.dtype==torch.bfloat16,
+                'original BF16 gamma/A_log/dt_bias/inputnorm stay untouched')
+        baseline={};observations=[];hooks=[];active=[None]
+        def bits(value):return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
+        def same(a,b,message):
+            require(a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bits(a)==bits(b),message)
+        def projection_hook(name):
+            def hook(module,args,result):
+                if active[0] is not None:
+                    active[0].tensor('projection.'+name+'.input',args[0])
+                    active[0].tensor('projection.'+name+'.output',result)
+            return hook
+        def kept(name,value):
+            count=value.numel()*value.element_size()
+            require(total_bytes[0]+count<=128*1024**2,'bounded combined plain/observed tensor payload')
+            baseline[name]=value.detach().contiguous().clone();total_bytes[0]+=count
+        try:
+            kept('source.A_log',gdn.A_log)
+            kept('source.dt_bias',gdn.dt_bias)
+            kept('source.norm.weight',gdn.norm.weight)
+            with torch.inference_mode():
+                for mode in ('plain','observed'):
+                    cache=DynamicCache(config=cfg)
+                    require(len(cache.layers)==64 and not cache.has_previous_state(0,0),'real original cache starts empty')
+                    if mode=='observed':
+                        for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
+                            hooks.append(gdn.get_submodule(name).register_forward_hook(projection_hook(name)))
+                    for step,token in enumerate((11,353)):
+                        embedding=F.embedding(torch.tensor([[token]],dtype=torch.int64),SOURCE_OWNER.tensors[EMBED])
+                        normalized=input_norm(embedding)
+                        require(embedding.shape==normalized.shape==(1,1,5120),'actual trained original embedding/InputNorm')
+                        require(cache.has_previous_state(0,0) is (step==1),'real initial then cached source dispatch')
+                        if mode=='plain':
+                            output=gdn(normalized,cache_params=cache)
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                kept(f'step-{step}.{name}',value)
+                        else:
+                            obs=Observation(step,gdn);observations.append(obs);active[0]=obs
+                            obs.tensor('prefix.embedding',embedding);obs.tensor('prefix.input_norm',normalized)
+                            require(sys.gettrace() is None,'do not replace an existing tracer')
+                            previous=sys.gettrace();mode_observer=ScopedDispatch(obs);obs.mode=mode_observer
+                            try:
+                                sys.settrace(obs.trace)
+                                with mode_observer:
+                                    output=gdn(normalized,cache_params=cache)
+                            finally:
+                                sys.settrace(previous);obs.mode=None;active[0]=None
+                            obs.tensor('cache.after.conv',cache.layers[0].conv_states[0])
+                            obs.tensor('cache.after.state',cache.layers[0].recurrent_states[0])
+                            obs.tensor('module.output',output)
+                            directory=OUTPUT/f'step-{step}';directory.mkdir()
+                            archive=directory/'actual-gdn.safetensors'
+                            save_file(obs.tensors,str(archive),metadata={'scope':'actual original27B layer0 frames; prevalidation only'})
+                            for name,value in obs.tensors.items():obs.meta[name]['raw_sha256']=digest(bits(value))
+                            record={'step':step,'teacher_token':token,'actual_calls':dict(obs.calls),'events':obs.events,
+                                    'dispatch':obs.dispatch,'tensors':obs.meta,'payload_bytes':obs.bytes,
+                                    'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
+                                    'plain_observed_all_outputs_and_state_exact':False}
+                            write(directory/'prevalidation.json',record)
+                            phase='chunk' if step==0 else 'recurrent'
+                            expected={'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,
+                                      'l2norm':2,'torch_'+phase+'_gated_delta_rule':1,
+                                      'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1}
+                            require(dict(obs.calls)==expected,'actual unchanged original initial/cached call schedule')
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                same(baseline[f'step-{step}.{name}'],value,'plain/observed complete actual '+name)
+                            same(obs.tensors[phase+'.return.actual_return_1'],cache.layers[0].recurrent_states[0],
+                                 'original public returned state equals real original cache')
+                            if step:
+                                same(observations[0].tensors['chunk.return.actual_return_1'],
+                                     obs.tensors['recurrent.input.initial_state'],'actual initial return supplies real cached prior')
+                            record['plain_observed_all_outputs_and_state_exact']=True
+                            report['steps'].append(record)
+                            total_bytes[0]+=obs.bytes
+                        for layer in cache.layers[1:]:
+                            require(not getattr(layer,'is_conv_states_initialized',[]) or
+                                    not any(layer.is_conv_states_initialized),'no unrelated layer state allocated')
+        finally:
+            active[0]=None
+            for hook in hooks:hook.remove()
+            require(sys.gettrace() is None,'original tracer restored')
+        save_file(baseline,str(OUTPUT/'plain.safetensors'),metadata={'scope':'genuine unobserved two-teacher module outputs/cache'})
+        prior=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prior['source_repo']==plan['repository'] and prior['source_revision']==plan['revision'] and
+                prior['snapshot']==plan['snapshot'],'independent original27B prefix reference source binding')
+        prefix_checks=[]
+        for step,token in enumerate((11,353)):
+            old=prior['steps'][step]
+            require(old['token']==token,'independent prefix teacher binding')
+            for stage,actual in [('embedding',baseline[f'step-{step}.embedding']),
+                                 ('normalized',baseline[f'step-{step}.input_norm']),
+                                 ('projected',observations[step].tensors['projection.in_proj_qkv.output'])]:
+                row=old[stage];expected=read(Path('/prefix-reference')/row['file'],pin=row)
+                widened=b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',bits(actual)))
+                require(actual.dtype==torch.bfloat16 and len(widened)==row['bytes'] and widened==expected,
+                        'all prior genuine original27B prefix words: '+stage)
+                prefix_checks.append({'step':step,'stage':stage,'words':actual.numel(),'bit_exact':True,'prior':row})
+        for name,value in gdn.named_parameters():
+            require(value.data_ptr()==parameter_audit[name]['pointer'] and value._version==parameter_audit[name]['version'],
+                    'original source parameter storage/version unchanged')
+        after=SOURCE_OWNER.finish()
+        maps=Path('/proc/self/maps').read_text()
+        require(len(maps.encode())<=2*1024**2 and '/nix/store/' not in maps,'bounded actual container mapping closure')
+        objects=sorted({line.split(None,5)[5] for line in maps.splitlines() if len(line.split(None,5))==6 and
+                        line.split(None,5)[5].startswith('/') and '.so' in Path(line.split(None,5)[5]).name})
+        require(len(objects)<=256,'bounded actual loaded-library count')
+        dsos={};total_dso=0
+        for path in objects:
+            size=Path(path).stat().st_size;total_dso+=size
+            require(size<=2*1024**3 and total_dso<=16*1024**3 and ' (deleted)' not in path,'bounded actual mapped provider closure')
+            dsos[path]={'bytes':size,'sha256':sha(path)}
+        report.update(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_FDs_before=SOURCE_OWNER.before,source_FDs_after=after,source_readsets=SOURCE_OWNER.readsets,
+            all_selected_FDs_and_mapped_parameters_unchanged=True,all_source_parameters_BF16=True,
+            parameter_audit=parameter_audit,source_owner_retained_until_owned_process_exit=True,
+            norm_profile={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+                          'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+                          'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16',
+                          'epsilon':gdn.norm.variance_epsilon,'activation':gdn.norm.activation},
+            payload_bytes=total_bytes[0],
+            plain_archive={'bytes':(OUTPUT/'plain.safetensors').stat().st_size,'sha256':sha(OUTPUT/'plain.safetensors')},
+            hooks_removed=len(hooks),runtime={'torch':torch.__version__,'torch_git':torch.version.git_version,
+                'transformers':importlib.metadata.version('transformers'),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
+                'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
+                'affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'modeling_sha256':MODEL_SHA,
+                'mapped_shared_objects':dsos,'proc_status':Path('/proc/self/status').read_text(),
+                'proc_cgroup':Path('/proc/self/cgroup').read_text(),'proc_mountinfo':Path('/proc/self/mountinfo').read_text()},
+            scope='Original27B private layer0 embedding/InputNorm/GDN two-teacher public reference only; all raw source BF16. No full-model logits/112 StateRoots, native exp/provider allocation eligibility, production memory admission, GPU or performance.')
+        report['prior_prefix_checks']=prefix_checks
+        require(report['payload_bytes']<=128*1024**2,'bounded total actual plain/observed payload')
+    except BaseException as error:
+        report['failure']=repr(error)
+        raise
+    finally:
+        report['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+        write(OUTPUT/'manifest.json',report)
+    print('ORIGINAL27B_LAYER0_GDN_CAPTURE_ONLY',flush=True)
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/observer.py
@@ -0,0 +1,249 @@
+"""Original frame clones and exactly-once selected public-op observation only."""
+import collections
+import pathlib
+import sys
+import torch
+from torch.utils._python_dispatch import TorchDispatchMode
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+from support import require
+model_source = pathlib.Path(official.__file__).resolve()
+MAX_TENSOR_BYTES = 128 * 1024 * 1024
+total_bytes = [0]
+
+def info(value):
+    return {"dtype":str(value.dtype),"shape":list(value.shape),"stride":list(value.stride()),
+            "contiguous":value.is_contiguous(),"numel":value.numel(),
+            "bytes":value.numel()*value.element_size(),"device":str(value.device)}
+
+class Observation:
+    def __init__(self, step, gdn):
+        self.step = step
+        self.gdn = gdn
+        self.norm = gdn.norm
+        self.tensors = {}
+        self.meta = {}
+        self.events = []
+        self.calls = collections.Counter()
+        self.frames = {}
+        self.points_seen = set()
+        self.bytes = 0
+        self.dispatch = []
+        self.mode = None
+        self.norm_entered = 0
+        self.norm_exited = 0
+
+    def tensor(self, name, value):
+        assert name not in self.tensors, name
+        assert value.device.type == "cpu"
+        self.bytes += value.numel() * value.element_size()
+        assert total_bytes[0] + self.bytes <= MAX_TENSOR_BYTES
+        self.meta[name] = info(value)
+        mode = self.mode
+        previous = mode.suppressed if mode is not None else False
+        if mode is not None:
+            mode.suppressed = True
+        try:
+            self.tensors[name] = value.detach().contiguous().clone()
+        finally:
+            if mode is not None:
+                mode.suppressed = previous
+
+    def observe(self, frame, label, names, returned=None):
+        metadata = {"function": frame.f_code.co_name, "line": frame.f_lineno,
+                    "label": label, "step": self.step, "tensors": []}
+        for name in names:
+            value = frame.f_locals.get(name)
+            if isinstance(value, torch.Tensor):
+                key = label + "." + name
+                self.tensor(key, value)
+                metadata["tensors"].append(key)
+            elif value is None or isinstance(value, (bool, int, float, str)):
+                metadata[name] = value
+            elif isinstance(value, torch.dtype):
+                metadata[name] = str(value)
+        if returned is not None:
+            values = returned if isinstance(returned, tuple) else (returned,)
+            for index, value in enumerate(values):
+                if isinstance(value, torch.Tensor):
+                    key = label + f".actual_return_{index}"
+                    self.tensor(key, value)
+                    metadata["tensors"].append(key)
+                else:
+                    metadata[f"actual_return_{index}"] = value
+        self.events.append(metadata)
+
+    def trace(self, frame, event, arg):
+        if frame.f_code.co_filename != str(model_source):
+            return None
+        name = frame.f_code.co_name
+        if event == "call":
+            if name == "forward" and frame.f_locals.get("self") is self.gdn:
+                self.calls["Qwen3_5GatedDeltaNet.forward"] += 1
+                self.frames[id(frame)] = "gdn"
+                self.observe(frame, "gdn.input", ["hidden_states"])
+                return self.trace
+            if name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                self.calls[name] += 1
+                self.frames[id(frame)] = "conv"
+                self.observe(frame, "conv.input", ["hidden_states", "conv_state", "weight", "bias", "activation"])
+                return self.trace
+            if name == "forward" and frame.f_locals.get("self") is self.norm:
+                self.calls["Qwen3_5RMSNormGated.forward"] += 1
+                self.frames[id(frame)] = "norm"
+                self.observe(frame, "norm.input", ["hidden_states", "gate"])
+                self.tensor("norm.input.weight", self.norm.weight)
+                self.events[-1]["weight"] = "norm.input.weight"
+                self.events[-1]["variance_epsilon"] = self.norm.variance_epsilon
+                self.events[-1]["activation"] = self.norm.activation
+                return self.trace
+            if name not in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "l2norm"}:
+                return None
+            caller = frame.f_back
+            while caller is not None:
+                owner = caller.f_locals.get("self")
+                if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                    if owner.layer_idx != 0:
+                        return None
+                    break
+                caller = caller.f_back
+            else:
+                return None
+            self.calls[name] += 1
+            if name == "l2norm":
+                self.frames[id(frame)] = "l2_" + ("query" if self.calls[name] == 1 else "key")
+                self.observe(frame, self.frames[id(frame)] + ".input", ["x", "dim", "eps"])
+            else:
+                self.frames[id(frame)] = "chunk" if name == "torch_chunk_gated_delta_rule" else "recurrent"
+                self.observe(frame, self.frames[id(frame)] + ".input", ["query", "key", "value", "g", "beta", "initial_state", "chunk_size", "output_final_state", "use_qk_l2norm_in_kernel"])
+            return self.trace
+        if id(frame) not in self.frames:
+            return None
+        phase = self.frames[id(frame)]
+        if event == "return":
+            if phase == "norm":
+                self.observe(frame, "norm.return", ["hidden_states", "gate", "input_dtype", "variance"], arg)
+            elif phase == "gdn":
+                self.observe(frame, "gdn.return", ["core_attn_out", "z", "output"], arg)
+            elif phase == "conv":
+                self.observe(frame, "conv.return", ["hidden_states", "hidden_states_new", "conv_state", "out"], arg)
+            elif phase.startswith("l2_"):
+                self.observe(frame, phase + ".return", ["inv_norm"], arg)
+            else:
+                self.observe(frame, phase + ".return", ["query", "key", "value", "beta", "decay", "initial_dtype", "pad_size", "num_chunks", "last_recurrent_state", "core_attn_out"], arg)
+            del self.frames[id(frame)]
+            return None
+        if event != "line":
+            return self.trace
+        # A line event sees the actual locals BEFORE the named source line.
+        points = CHUNK_POINTS if phase == "chunk" else RECURRENT_POINTS if phase == "recurrent" else NORM_POINTS if phase == "norm" else GDN_POINTS if phase == "gdn" else {}
+        if frame.f_lineno in points:
+            label, names = points[frame.f_lineno]
+            point = (id(frame), label)
+            if point not in self.points_seen:
+                self.points_seen.add(point)
+                self.observe(frame, phase + "." + label, names)
+        return self.trace
+
+
+NORM_POINTS = {
+    228:("core_fp32",["hidden_states","input_dtype"]),
+    230:("mean",["hidden_states","variance","input_dtype"]),
+    231:("normalized_fp32",["hidden_states","variance","input_dtype"]),
+    232:("weighted_bf16",["hidden_states","gate","input_dtype"]),
+    234:("product_fp32",["hidden_states","gate","input_dtype"]),
+}
+
+
+CHUNK_POINTS = {
+    343: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    347: ("qk_normalized", ["query", "key"]),
+    351: ("query_scaled", ["query", "scaling"]),
+    355: ("padded64", ["query", "key", "value", "beta", "decay", "pad_size", "sequence_length"]),
+    364: ("beta_products", ["v_beta", "k_beta"]),
+    379: ("cumulative_decay", ["decay", "cum_decay"]),
+    384: ("pairwise_decay", ["pairwise_decay"]),
+    386: ("intra_chunk_products", ["ut_system", "intra_chunk_attn"]),
+    406: ("triangular_solutions", ["decayed_k_beta", "new_values", "k_cumdecay"]),
+    413: ("initial_state", ["last_recurrent_state"]),
+    418: ("chunk_decayed_inputs", ["query", "key", "chunk_decay", "num_chunks"]),
+    422: ("v_new", ["v_new"]),
+    425: ("fp32_core_before_state_update", ["core_attn_out", "inter_chunk_attn"]),
+}
+RECURRENT_POINTS = {
+    463: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    468: ("qk_normalized", ["query", "key"]),
+    472: ("query_scaled", ["query"]),
+    484: ("decay_and_prior", ["decay_t", "last_recurrent_state"]),
+    486: ("decayed_state", ["last_recurrent_state"]),
+    488: ("memory_prediction", ["q_t", "k_t", "v_t", "beta_t", "kv_mem"]),
+    489: ("delta", ["delta"]),
+    491: ("proposed_state", ["last_recurrent_state"]),
+    494: ("fp32_core", ["core_attn_out"]),
+}
+
+
+GDN_POINTS = {
+    574:("projections",["mixed_qkv","z","a","b","use_precomputed_states"]),
+    617:("pre_beta",["query","key","value","a","b"]),
+    620:("coefficients",["a","b","beta","g"]),
+    624:("repeated",["query","key","value","beta","g"]),
+    653:("core_return",["core_attn_out","last_recurrent_state"]),
+    660:("norm_return",["core_attn_out","z"]),
+}
+
+class ScopedDispatch(TorchDispatchMode):
+    def __init__(self, observation):
+        super().__init__()
+        self.observation = observation
+        self.suppressed = False
+
+    def source(self):
+        frame = sys._getframe()
+        while frame is not None:
+            if frame.f_code.co_filename == str(model_source):
+                owner = frame.f_locals.get("self")
+                if owner is self.observation.norm:
+                    return "norm",frame.f_lineno
+                if owner is self.observation.gdn:
+                    return "gdn",frame.f_lineno
+                if frame.f_code.co_name in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule"):
+                    return frame.f_code.co_name,frame.f_lineno
+            frame=frame.f_back
+        return None,None
+
+    def tree(self, value, label, budget):
+        budget[0]+=1
+        require(budget[0]<=256,"bounded operation argument tree")
+        if isinstance(value,torch.Tensor):
+            self.observation.tensor(label,value)
+            return {"tensor":label}
+        if isinstance(value,(list,tuple)):
+            require(len(value)<=32,"bounded argument sequence")
+            return [self.tree(v,label+"."+str(i),budget) for i,v in enumerate(value)]
+        if isinstance(value,dict):
+            require(len(value)<=32 and all(isinstance(k,str) for k in value),"bounded argument mapping")
+            return {k:self.tree(v,label+"."+k,budget) for k,v in value.items()}
+        if value is None or isinstance(value,(bool,int,float,str)):
+            return value
+        if isinstance(value,(torch.dtype,torch.device,torch.layout,torch.memory_format)):
+            return {"torch_value":str(value)}
+        raise ValueError("unhandled original operation argument: "+type(value).__name__)
+
+    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
+        kwargs={} if kwargs is None else kwargs
+        if self.suppressed:
+            return func(*args,**kwargs)
+        scope,line=self.source()
+        selected = scope=="norm" or scope=="gdn" and line in (617,619) or \
+            scope in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule") and str(func)=="aten.exp.default"
+        if not selected:
+            return func(*args,**kwargs)
+        require(len(self.observation.dispatch)<128,"bounded selected original operations")
+        index=len(self.observation.dispatch);label=f"dispatch.{index:03d}"
+        record={"index":index,"operation":str(func),"schema":str(func._schema),"scope":scope,
+                "source_line":line,"arguments":self.tree(args,label+".args",[0]),
+                "keyword_arguments":self.tree(kwargs,label+".kwargs",[0])}
+        result=func(*args,**kwargs) # EXACT original call once; original result returned.
+        record["result"]=self.tree(result,label+".result",[0])
+        self.observation.dispatch.append(record)
+        return result
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/run_reference.py
@@ -0,0 +1,134 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=256*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    def cancelled(_signal,_frame):raise InterruptedError('owned reference cancelled; retire only owned container/process')
+    signal.signal(signal.SIGTERM,cancelled)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('gdn27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-gdn-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json
@@ -0,0 +1,449 @@
+{
+  "bounds_before_library_or_views": {
+    "CPU_count": 4,
+    "aggregate_window_mapping_bytes": 2774582792,
+    "capture_payload_cap_bytes": 134217728,
+    "configured_container_RAM_bytes": 12884901888,
+    "conservative_operator_working_envelope_bytes": 2147483648,
+    "container_wall_seconds": 600,
+    "control_allowance_bytes": 67108864,
+    "derived_touched_source_cap_bytes": 268435456,
+    "mapped_VA_cap_bytes": 3221225472,
+    "native_ResourceArbiter_admission_qualified": false,
+    "outer_wall_seconds": 2400,
+    "page_size": 4096,
+    "process_AS_cap_bytes": 12884901888,
+    "runtime_VA_allowance_bytes": 6442450944,
+    "runtime_resident_allowance_bytes": 2147483648,
+    "selected_parameter_bytes": 231752128,
+    "touched_source_payload_bytes": 231782848,
+    "whole_artifact_cap_bytes": 268435456
+  },
+  "geometry": {
+    "conv_channels": 10240,
+    "conv_kernel": 4,
+    "head_width": 128,
+    "hidden": 5120,
+    "key_heads": 16,
+    "layers": 64,
+    "rms_epsilon": 1e-06,
+    "value_heads": 48
+  },
+  "metadata": {
+    "config.json": {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    "model.safetensors.index.json": {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    }
+  },
+  "original_parameters": "ALL raw source BF16, including A_log/dt_bias/norm.gamma; no official-loader cast or restored-F32 replacement",
+  "prior_full_source_verification": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "prior_public_prefix": {
+    "bytes": 65417,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+    "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+  },
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "runtime_source": {
+    "cache_utils.py": {
+      "bytes": 104677,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/cache_utils.py",
+      "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c"
+    },
+    "configuration_qwen3_5.py": {
+      "bytes": 8395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/configuration_qwen3_5.py",
+      "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da"
+    },
+    "hub_kernels.py": {
+      "bytes": 44974,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/hub_kernels.py",
+      "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95"
+    },
+    "modeling_qwen3_5.py": {
+      "bytes": 95685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    }
+  },
+  "schema": "zerogravity.original27B.layer0-gdn.source-plan.v1",
+  "selected_shards": {
+    "model-00001-of-00018.safetensors": {
+      "file": {
+        "bytes": 3966730552,
+        "git_blob_verified": false,
+        "name": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
+    },
+    "model-00003-of-00018.safetensors": {
+      "file": {
+        "bytes": 2542796952,
+        "git_blob_verified": false,
+        "name": "model-00003-of-00018.safetensors",
+        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+      },
+      "header_bytes": 144,
+      "header_sha256": "26785014f43e18cd545cebad85d003854b94eb64b90079852ba7efeb8041093c"
+    }
+  },
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "source_files": [
+    {
+      "bytes": 11544,
+      "git_blob_verified": true,
+      "name": "LICENSE",
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+    },
+    {
+      "bytes": 65012,
+      "git_blob_verified": true,
+      "name": "README.md",
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+    },
+    {
+      "bytes": 8952,
+      "git_blob_verified": true,
+      "name": "chat_template.jinja",
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+    },
+    {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    {
+      "bytes": 238,
+      "git_blob_verified": true,
+      "name": "crc32.txt",
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+    },
+    {
+      "bytes": 202,
+      "git_blob_verified": true,
+      "name": "generation_config.json",
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+    },
+    {
+      "bytes": 3353259,
+      "git_blob_verified": true,
+      "name": "merges.txt",
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+    },
+    {
+      "bytes": 3966730552,
+      "git_blob_verified": false,
+      "name": "model-00001-of-00018.safetensors",
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+    },
+    {
+      "bytes": 3043080328,
+      "git_blob_verified": false,
+      "name": "model-00002-of-00018.safetensors",
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+    },
+    {
+      "bytes": 2542796952,
+      "git_blob_verified": false,
+      "name": "model-00003-of-00018.safetensors",
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+    },
+    {
+      "bytes": 3988973152,
+      "git_blob_verified": false,
+      "name": "model-00004-of-00018.safetensors",
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+    },
+    {
+      "bytes": 2099339864,
+      "git_blob_verified": false,
+      "name": "model-00005-of-00018.safetensors",
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00006-of-00018.safetensors",
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00007-of-00018.safetensors",
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00008-of-00018.safetensors",
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00009-of-00018.safetensors",
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00010-of-00018.safetensors",
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00011-of-00018.safetensors",
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00012-of-00018.safetensors",
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00013-of-00018.safetensors",
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00014-of-00018.safetensors",
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00015-of-00018.safetensors",
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+    },
+    {
+      "bytes": 3979564040,
+      "git_blob_verified": false,
+      "name": "model-00016-of-00018.safetensors",
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00017-of-00018.safetensors",
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+    },
+    {
+      "bytes": 3392197344,
+      "git_blob_verified": false,
+      "name": "model-00018-of-00018.safetensors",
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+    },
+    {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    },
+    {
+      "bytes": 390,
+      "git_blob_verified": true,
+      "name": "preprocessor_config.json",
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+    },
+    {
+      "bytes": 12809320,
+      "git_blob_verified": false,
+      "name": "tokenizer.json",
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+    },
+    {
+      "bytes": 17928,
+      "git_blob_verified": true,
+      "name": "tokenizer_config.json",
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+    },
+    {
+      "bytes": 385,
+      "git_blob_verified": true,
+      "name": "video_preprocessor_config.json",
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+    },
+    {
+      "bytes": 6722759,
+      "git_blob_verified": true,
+      "name": "vocab.json",
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+    }
+  ],
+  "status": "SOURCE_HEADER_ONLY_NO_OPERATOR_EXECUTION",
+  "tensors": {
+    "model.language_model.embed_tokens.weight": {
+      "absolute_offset": 152,
+      "bytes": 2542796800,
+      "dtype": "BF16",
+      "mapping_bytes": 2542796952,
+      "name": "model.language_model.embed_tokens.weight",
+      "page_aligned_offset": 0,
+      "shape": [
+        248320,
+        5120
+      ],
+      "shard": "model-00003-of-00018.safetensors",
+      "view_offset": 152
+    },
+    "model.language_model.layers.0.input_layernorm.weight": {
+      "absolute_offset": 45400,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "mapping_bytes": 10584,
+      "name": "model.language_model.layers.0.input_layernorm.weight",
+      "page_aligned_offset": 45056,
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 344
+    },
+    "model.language_model.layers.0.linear_attn.A_log": {
+      "absolute_offset": 55640,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2488,
+      "name": "model.language_model.layers.0.linear_attn.A_log",
+      "page_aligned_offset": 53248,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2392
+    },
+    "model.language_model.layers.0.linear_attn.conv1d.weight": {
+      "absolute_offset": 55736,
+      "bytes": 81920,
+      "dtype": "BF16",
+      "mapping_bytes": 84408,
+      "name": "model.language_model.layers.0.linear_attn.conv1d.weight",
+      "page_aligned_offset": 53248,
+      "shape": [
+        10240,
+        1,
+        4
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.dt_bias": {
+      "absolute_offset": 137656,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2584,
+      "name": "model.language_model.layers.0.linear_attn.dt_bias",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_a.weight": {
+      "absolute_offset": 137752,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_b.weight": {
+      "absolute_offset": 629272,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
+      "page_aligned_offset": 626688,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
+      "absolute_offset": 1120792,
+      "bytes": 104857600,
+      "dtype": "BF16",
+      "mapping_bytes": 104860184,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+      "page_aligned_offset": 1118208,
+      "shape": [
+        10240,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_z.weight": {
+      "absolute_offset": 105978392,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917144,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
+      "page_aligned_offset": 105975808,
+      "shape": [
+        6144,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.norm.weight": {
+      "absolute_offset": 168892952,
+      "bytes": 256,
+      "dtype": "BF16",
+      "mapping_bytes": 2840,
+      "name": "model.language_model.layers.0.linear_attn.norm.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        128
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.out_proj.weight": {
+      "absolute_offset": 168893208,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917400,
+      "name": "model.language_model.layers.0.linear_attn.out_proj.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    }
+  },
+  "weight_bytes": 55563006776
+}
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/support.py
@@ -0,0 +1,110 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def bounds(plan, page):
+    """Integer geometry before importing libraries or allocating parameter maps."""
+    limits = plan['bounds_before_library_or_views']
+    require(page == limits['page_size'] == 4096, 'pinned page geometry')
+    require(len(plan['tensors']) == 11 and len(plan['selected_shards']) == 2, 'exact complete selected roles')
+    mapped, params = 0, 0
+    for name, item in plan['tensors'].items():
+        require(item['dtype'] == 'BF16' and type(item['shape']) is list, 'original BF16 source policy')
+        count = 1
+        for axis in item['shape']:
+            require(type(axis) is int and 0 < axis <= 248320, 'bounded original tensor axis')
+            count *= axis
+        start, length = item['absolute_offset'], item['bytes']
+        require(type(start) is type(length) is int and start >= 8 and length == count*2,
+                'full original typed extent')
+        aligned = start-start%page
+        require(item['page_aligned_offset'] == aligned and item['view_offset'] == start-aligned and
+                item['mapping_bytes'] == length+start-aligned and
+                start+length <= plan['selected_shards'][item['shard']]['file']['bytes'],
+                'exact bounded parameter window; no whole shard mapping')
+        mapped += item['mapping_bytes']
+        if '.linear_attn.' in name:
+            params += length
+    touched = params + 10240 + 20480
+    require(params == limits['selected_parameter_bytes'] == 231752128 and
+            mapped == limits['aggregate_window_mapping_bytes'] == 2774582792 and
+            touched == limits['touched_source_payload_bytes'] == 231782848 and
+            mapped <= limits['mapped_VA_cap_bytes'] == 3*1024**3 and
+            touched <= limits['derived_touched_source_cap_bytes'] == 256*1024**2,
+            'pre-library exact source windows and selected resident payload bound')
+    virtual_envelope=mapped+limits['runtime_VA_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']
+    physical_estimate=touched+limits['runtime_resident_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']+limits['control_allowance_bytes']
+    require(virtual_envelope < limits['process_AS_cap_bytes'] == 12*1024**3 and
+            physical_estimate < limits['configured_container_RAM_bytes'] and
+            limits['configured_container_RAM_bytes'] == 12*1024**3,
+            'finite explicit reference containment; not native physical admission')
+    return {'mapped_bytes':mapped,'selected_parameter_bytes':params,'touched_source_bytes':touched,
+            'virtual_envelope_bytes':virtual_envelope,'physical_envelope_estimate_bytes':physical_estimate,
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    manifest = metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:
+        read(row['path'],pin=row)
+    return sha(HERE/'prepared-manifest.json')
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/test_contracts.py
@@ -0,0 +1,43 @@
+"""ROOT-only source-free geometry/archive fixtures; no library/model import."""
+import copy
+from pathlib import Path
+import json
+import struct
+import tempfile
+import unittest
+from support import HERE, bounds, digest, metadata
+from validate_reference import archive
+
+class Contracts(unittest.TestCase):
+    def test_exact_parameter_windows_and_full_source_kind_refuse(self):
+        plan=metadata(HERE/'source-plan.json')
+        value=bounds(plan,4096)
+        self.assertEqual(value['mapped_bytes'],2774582792)
+        for field in ('absolute_offset','mapping_bytes','view_offset','bytes'):
+            wrong=copy.deepcopy(plan)
+            first=next(iter(wrong['tensors'].values()))
+            first[field]+=1
+            with self.assertRaises(ValueError):bounds(wrong,4096)
+        wrong=copy.deepcopy(plan);next(iter(wrong['tensors'].values()))['dtype']='F32'
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_native_provider_admission_not_inferred_from_containment(self):
+        plan=metadata(HERE/'source-plan.json')
+        self.assertIs(bounds(plan,4096)['native_ResourceArbiter_admission_qualified'],False)
+        with self.assertRaises(ValueError):bounds(plan,8192)
+        wrong=copy.deepcopy(plan);wrong['bounds_before_library_or_views']['runtime_VA_allowance_bytes']=16*1024**3
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_archive_preserves_actual_negative_inf_mask_and_rejects_tail(self):
+        # Transport-only fixture: no candidate result or numerical expectation.
+        payload=bytes.fromhex('000080ff0000008001000000')
+        header=json.dumps({'x':{'dtype':'F32','shape':[3],'data_offsets':[0,12]}}).encode()
+        raw=struct.pack('<Q',len(header))+header+payload
+        with tempfile.TemporaryDirectory() as root:
+            path=Path(root)/'actual.safetensors';path.write_bytes(raw)
+            pin={'bytes':len(raw),'sha256':digest(raw)}
+            self.assertEqual(archive(path,pin)['x']['raw'],payload)
+            path.write_bytes(raw+b'\0')
+            with self.assertRaises(ValueError):archive(path,{'bytes':len(raw)+1,'sha256':digest(raw+b'\0')})
+
+if __name__=='__main__':unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/validate_reference.py
@@ -0,0 +1,251 @@
+"""ROOT ONLY stdlib full-byte/source/lifecycle validation; no tensor arithmetic."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, frozen, metadata, read, require, sha, unique, write
+
+PREFIX='model.language_model.layers.0.linear_attn.'
+
+def source_readsets(plan,report):
+    """ROOT-only actual indexed source bytes, bounded read; no Torch or math."""
+    require(set(report['source_readsets'])==set(plan['tensors']),'all actual source raw readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd));n,=struct.unpack('<Q',os.pread(fd,8,0))
+            require(before[2]==pin['file']['bytes'] and n==pin['header_bytes'] and
+                    digest(os.pread(fd,n,8))==pin['header_sha256'],'same authenticated selected source header')
+            for name,item in plan['tensors'].items():
+                if item['shard']!=shard:continue
+                if name=='model.language_model.embed_tokens.weight':
+                    actual={str(t):fd_sha(fd,item['absolute_offset']+t*10240,10240) for t in (11,353)}
+                else:actual=fd_sha(fd,item['absolute_offset'],item['bytes'])
+                require(actual==report['source_readsets'][name],'independent actual raw source readset: '+name)
+            require(fingerprint(os.fstat(fd))==before,'original FD unchanged during independent source-range validation')
+        finally:os.close(fd)
+
+def archive(path,pin):
+    raw=read(path,128*1024**2,pin);n,=struct.unpack_from('<Q',raw)
+    require(1<n<=8*1024**2 and 8+n<len(raw),'bounded complete raw archive header')
+    head=json.loads(raw[8:8+n],object_pairs_hook=unique);tensors={};spans=[]
+    for name,item in head.items():
+        if name=='__metadata__':continue
+        dtype,shape=item['dtype'],item['shape'];require(dtype in ('F32','BF16'),'original raw archive dtype')
+        count=1
+        require(type(shape) is list and len(shape)<=8,'bounded actual rank')
+        for dim in shape:
+            require(type(dim) is int and 0<dim<=248320,'bounded actual dimension');count*=dim
+        lo,hi=item['data_offsets'];size=2 if dtype=='BF16' else 4
+        require(type(lo) is type(hi) is int and 0<=lo<hi and hi-lo==count*size and 8+n+hi<=len(raw),
+                'complete original dtype/extent geometry')
+        tensors[name]={'dtype':dtype,'shape':shape,'raw':raw[8+n+lo:8+n+hi]};spans.append((lo,hi))
+    cursor=0
+    for lo,hi in sorted(spans):require(lo==cursor,'all disjoint complete archive bytes');cursor=hi
+    require(8+n+cursor==len(raw),'no trailing/unbound archive bytes')
+    return tensors
+
+def tensor(tensors,name,dtype,shape=None):
+    value=tensors[name]
+    require(value['dtype']==dtype and (shape is None or value['shape']==shape),'actual stage dtype/shape: '+name)
+    return value['raw']
+
+def widened(raw):
+    return b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',raw))
+
+def operation(rows,scope,line,name):
+    selected=[row for row in rows if row['scope']==scope and row['source_line']==line and row['operation']==name]
+    require(len(selected)==1,'exact actual selected operator: '+str((scope,line,name)))
+    return selected[0]
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');derived=bounds(plan,4096)
+    invocation=metadata(args.output/'invocation.json');directory=args.output/'captured';report=metadata(directory/'manifest.json')
+    require(invocation['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['freeze_manifest_sha256']==freeze and invocation['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            invocation['exit_code']==0 and invocation['cleanup_errors']==[] and invocation['source_FD_cleanup_errors']==[] and
+            invocation['source_FDs_closed'] is True and invocation.get('owned_container_absent_after_reap') is True,
+            'actual original source/reference/owned container all retired')
+    cleanup=invocation['foreground_cleanup'];logs=invocation['log_cleanup']
+    require(cleanup['leader_reaped'] is True and cleanup['group_absent'] is True and cleanup['exit_code']==0 and
+            cleanup['all_nonzero_signals_before_reap'] is True and cleanup['stage_errors']==[] and
+            logs['pipe_eof'] is logs['all_closed'] is True and logs['total_bytes']==logs['retained_bytes']<=4*1024**2,
+            'original finite owned process/PGID/EOF cleanup and complete evidence')
+    expected={v['name']:(v['bytes'],v['sha256']) for v in plan['source_files']}
+    require(invocation['source_before']==invocation['source_after'] and
+            {v['name']:(v['bytes'],v['sha256']) for v in invocation['source_before']}==expected and
+            len(invocation['source_before'])==len(expected),'all original source files before/after full SHA coverage')
+    require(report['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            report['schema']=='zerogravity.original27B.layer0-gdn.public-reference.v1' and report['tokens']==[11,353] and
+            report['parameter_profile']=='original27B-raw-source-bf16-v1' and
+            report['source_plan_sha256']==sha(HERE/'source-plan.json') and report['source_repo']==plan['repository'] and
+            report['source_revision']==plan['revision'] and report['snapshot']==plan['snapshot'] and
+            report['pre_library_bounds']==derived and report['native_ResourceArbiter_admission_qualified'] is False and
+            report['full_model_StateRoot_native_exp_GPU_eligible'] is False and report['all_source_parameters_BF16'] is True and
+            report['source_owner_retained_until_owned_process_exit'] is True and report['hooks_removed']==5,
+            'authentic private original27B profile and unchanged eligibility guards')
+    require(report['norm_profile']=={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+        'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+        'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16','epsilon':1e-6,'activation':'silu'},
+        'explicit original27B BF16 gamma round profile')
+    require(report['all_selected_FDs_and_mapped_parameters_unchanged'] is True and
+            report['source_FDs_before']==report['source_FDs_after'] and
+            set(report['source_FDs_before'])==set(plan['selected_shards']),'same exact selected original FD owners')
+    source_readsets(plan,report)
+    expected_parameters={name.removeprefix(PREFIX):item for name,item in plan['tensors'].items() if name.startswith(PREFIX)}
+    require(set(report['parameter_audit'])==set(expected_parameters) and len(expected_parameters)==9,
+            'all and only original learned GDN parameters')
+    for name,item in expected_parameters.items():
+        actual=report['parameter_audit'][name]
+        require(actual['dtype']=='torch.bfloat16' and actual['shape']==item['shape'] and
+                actual['device']=='cpu' and actual['contiguous'] is True and actual['bytes']==item['bytes'] and
+                type(actual['pointer']) is int and actual['pointer']>0 and
+                actual['source_readset_sha256']==report['source_readsets'][PREFIX+name],
+                'actual complete original raw parameter binding: '+name)
+    runtime=report['runtime'];require(runtime['torch']=='2.13.0+cu129' and runtime['torch_git']==TORCH_GIT and
+        runtime['transformers']=='5.17.0' and runtime['cpu_capability']=='AVX2' and runtime['threads']==4 and
+        runtime['interop_threads']==1 and runtime['affinity']==invocation['expected_cpus'] and
+        len(set(runtime['affinity']))==4 and runtime['cuda_available'] is False and runtime['modeling_sha256']==MODEL_SHA,
+        'actual pinned unchanged CPU runtime and four CPUs')
+    dsos=runtime['mapped_shared_objects']
+    require(dsos['/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so']['sha256']==
+            '1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab' and
+            dsos['/usr/lib/x86_64-linux-gnu/libm.so.6']['sha256']==
+            'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
+            'actual original exp/CPU/scalar-provider mappings, not native admission')
+    plain=archive(directory/'plain.safetensors',report['plain_archive']);steps=[];payload=sum(len(t['raw']) for t in plain.values())
+    for name,shape in [('A_log',[48]),('dt_bias',[48]),('norm.weight',[128])]:
+        require(digest(tensor(plain,'source.'+name,'BF16',shape))==report['source_readsets'][PREFIX+name],
+                'all original BF16 coefficient/gamma source bytes in reference')
+    require(len(report['steps'])==2,'both true source teachers')
+    for step,record in enumerate(report['steps']):
+        phase='chunk' if step==0 else 'recurrent'
+        require(record['step']==step and record['teacher_token']==[11,353][step] and
+                record['plain_observed_all_outputs_and_state_exact'] is True and record['actual_calls']=={
+                    'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,'l2norm':2,
+                    'torch_'+phase+'_gated_delta_rule':1,'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1},
+                'one original chunk64 first, one actual cached call second')
+        calls=[event for event in record['events'] if event['label']==phase+'.input']
+        require(len(calls)==1 and calls[0]['output_final_state'] is True and
+                calls[0]['use_qk_l2norm_in_kernel'] is True and
+                (step==1 or calls[0]['initial_state'] is None and calls[0]['chunk_size']==64),
+                'actual original phase/initial-state/padded-chunk contract')
+        tensors=archive(directory/f'step-{step}/actual-gdn.safetensors',
+                        {'bytes':record['archive_bytes'],'sha256':record['archive_sha256']});steps.append(tensors)
+        before=metadata(directory/f'step-{step}/prevalidation.json')
+        require(before==dict(record,plain_observed_all_outputs_and_state_exact=False),
+                'raw original diagnostics retained before semantic acceptance')
+        require(set(tensors)==set(record['tensors']),'all original raw observed stages bound')
+        for name,item in tensors.items():
+            meta=record['tensors'][name]
+            require(meta['raw_sha256']==digest(item['raw']) and meta['bytes']==len(item['raw']) and
+                    meta['shape']==item['shape'] and meta['dtype']==('torch.bfloat16' if item['dtype']=='BF16' else 'torch.float32') and
+                    meta['device']=='cpu','actual complete observer payload: '+name)
+            payload+=len(item['raw'])
+        require(sum(len(t['raw']) for t in tensors.values())==record['payload_bytes'],'all actual per-step payload counted')
+        for original,observed,dtype,shape in [('embedding','prefix.embedding','BF16',[1,1,5120]),
+                ('input_norm','prefix.input_norm','BF16',[1,1,5120]),('output','module.output','BF16',[1,1,5120]),
+                ('conv','cache.after.conv','BF16',[1,10240,4]),('state','cache.after.state','F32',[1,48,128,128])]:
+            require(tensor(plain,f'step-{step}.'+original,dtype,shape)==tensor(tensors,observed,dtype,shape),
+                    'all plain/observed source outputs and full actual state: '+original)
+        require(tensor(tensors,phase+'.return.actual_return_1','F32',[1,48,128,128])==
+                tensor(tensors,'cache.after.state','F32',[1,48,128,128]),'actual original returned state equals actual cache')
+        require(tensor(tensors,phase+'.return.actual_return_0','BF16',[1,1,48,128])==
+                tensor(tensors,'norm.input.hidden_states','BF16',[48,128]),'original core return feeds actual gated norm')
+        for name in ('query','key'):
+            before=tensor(tensors,'gdn.pre_beta.'+name,'BF16',[1,1,16,128]);after=tensor(tensors,'gdn.repeated.'+name,'BF16',[1,1,48,128])
+            require(after==b''.join(before[h*256:(h+1)*256]*3 for h in range(16)),
+                    'actual original Q/K repeat_interleave(3), no inferred permutation')
+        rows=record['dispatch'];coef=[v for v in rows if v['scope']=='gdn']
+        require([v['operation'] for v in coef]==['aten.sigmoid.default','aten.to.dtype','aten.exp.default','aten.neg.default',
+                'aten.to.dtype','aten.add.Tensor','aten.softplus.default','aten.mul.Tensor'] and
+                [v['source_line'] for v in coef]==[617]+[619]*7,'actual BF16 A_log .float/coefficient source order')
+        def op_raw(row,which='result',index=0):
+            ref=row[which][index] if which=='arguments' else row[which]
+            require(type(ref) is dict and set(ref)=={'tensor'},'actual selected tensor result/argument')
+            return tensors[ref['tensor']]['raw']
+        require(op_raw(coef[0],'arguments')==tensor(tensors,'projection.in_proj_b.output','BF16',[1,1,48]) and
+                op_raw(coef[0])==tensor(tensors,'gdn.coefficients.beta','BF16',[1,1,48]) and
+                op_raw(coef[1],'arguments')==tensor(plain,'source.A_log','BF16',[48]) and
+                op_raw(coef[1])==widened(tensor(plain,'source.A_log','BF16',[48])) and
+                op_raw(coef[4],'arguments')==tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48]) and
+                op_raw(coef[4])==widened(tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48])) and
+                op_raw(coef[5],'arguments',1)==tensor(plain,'source.dt_bias','BF16',[48]) and
+                op_raw(coef[-1])==tensor(tensors,phase+'.input.g','F32',[1,1,48]),
+                'genuine source coefficient producer inputs/results, no reference lookup')
+        beta=tensor(tensors,phase+'.input.beta','BF16',[1,1,48])
+        require(all(0.0<=struct.unpack('<f',(word<<16).to_bytes(4,'little'))[0]<=1.0 for word, in struct.iter_unpack('<H',beta)) and
+                all(value<=0.0 for value, in struct.iter_unpack('<f',tensor(tensors,phase+'.input.g','F32',[1,1,48]))),
+                'actual beta and original negative log-decay ranges')
+        norm={}
+        require(len([row for row in rows if row['scope']=='norm'])==12,'complete unchanged original gated norm operation count')
+        for stage,op,line,dtype,shape in [
+            ('core','aten.to.dtype',227,'F32',[48,128]),('squares','aten.pow.Tensor_Scalar',228,'F32',[48,128]),
+            ('mean','aten.mean.dim',228,'F32',[48,1]),('denominator','aten.add.Tensor',230,'F32',[48,1]),
+            ('inverse','aten.rsqrt.default',230,'F32',[48,1]),('normalized','aten.mul.Tensor',230,'F32',[48,128]),
+            ('normalized_bf16','aten.to.dtype',231,'BF16',[48,128]),('weighted_bf16','aten.mul.Tensor',231,'BF16',[48,128]),
+            ('gate','aten.to.dtype',232,'F32',[48,128]),('silu','aten.silu.default',232,'F32',[48,128]),
+            ('product','aten.mul.Tensor',232,'F32',[48,128]),('return','aten.to.dtype',234,'BF16',[48,128])]:
+            actual=operation(rows,'norm',line,op);name=actual['result']['tensor'];norm[stage]=tensor(tensors,name,dtype,shape)
+        require(tensor(tensors,'norm.input.weight','BF16',[128])==tensor(plain,'source.norm.weight','BF16',[128]) and
+                norm['weighted_bf16']==tensor(tensors,'norm.weighted_bf16.hidden_states','BF16',[48,128]) and
+                norm['return']==tensor(tensors,'norm.return.actual_return_0','BF16',[48,128]),
+                'actual source BF16 gamma product rounds BEFORE FP32 SiLU gate')
+        exp=[v for v in rows if v['scope']=='torch_'+phase+'_gated_delta_rule']
+        require([v['source_line'] for v in exp]==([381,386,413,414,415] if step==0 else [483]) and
+                all(v['operation']=='aten.exp.default' for v in exp),'all actual original initial/cached exponentials')
+        masked=exp[0]['arguments'][0]['tensor'] if step==0 else None
+        for name,item in tensors.items():
+            if name==masked:
+                require(item['shape']==[1,48,1,64,64] and item['dtype']=='F32','actual upper-triangle exp input geometry')
+                for index,(word,) in enumerate(struct.iter_unpack('<I',item['raw'])):
+                    i,j=(index//64)%64,index%64
+                    require(word==0xff800000 if i<j else word&0x7f800000!=0x7f800000,
+                            'ONLY source strictly-upper exp mask may contain canonical negative Inf')
+            else:
+                mask=0x7f80 if item['dtype']=='BF16' else 0x7f800000
+                fmt='<H' if item['dtype']=='BF16' else '<I'
+                require(all(word&mask!=mask for word, in struct.iter_unpack(fmt,item['raw'])),'all other actual raw words finite: '+name)
+        if step:
+            require(tensor(steps[0],'chunk.return.actual_return_1','F32',[1,48,128,128])==
+                    tensor(tensors,'recurrent.input.initial_state','F32',[1,48,128,128]) and
+                    op_raw(exp[0])==tensor(tensors,'recurrent.decay_and_prior.decay_t','F32',[1,48,1,1]),
+                    'actual full initial chunk prior and actual cached exp(g), never zero replacement')
+        else:
+            for name in ('query','key','value','beta','decay'):
+                value=tensors['chunk.padded64.'+name];width=128 if name in ('query','key','value') else 1
+                require(value['dtype']=='F32' and value['shape']==([1,48,64,128] if width==128 else [1,48,64]),
+                        'actual first-token source padded64 layout')
+                for head in range(48):
+                    require(value['raw'][(head*64+1)*width*4:(head+1)*64*width*4]==bytes(63*width*4),
+                            'all63 genuine source padding lanes canonical positive zero')
+    require(payload==report['payload_bytes'] and payload<=128*1024**2,'all actual plain/observed payload bytes counted')
+    require(len(report['prior_prefix_checks'])==6 and all(v['bit_exact'] is True for v in report['prior_prefix_checks']),
+            'independent previously accepted original27B embedding/norm/QKV prefix closure')
+    prior=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    for step in range(2):
+        for stage,name,dataset in [('embedding',f'step-{step}.embedding',plain),
+                                   ('normalized',f'step-{step}.input_norm',plain),
+                                   ('projected','projection.in_proj_qkv.output',steps[step])]:
+            row=prior['steps'][step][stage]
+            require(widened(tensor(dataset,name,'BF16'))==read(Path(plan['prior_public_prefix']['path']).parent/row['file'],pin=row),
+                    'independent every genuine original prefix word')
+    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}=={
+        'manifest.json','plain.safetensors','step-0/actual-gdn.safetensors','step-0/prevalidation.json',
+        'step-1/actual-gdn.safetensors','step-1/prevalidation.json'},'all captured source/byte artifacts accounted')
+    total_artifacts=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no unknown artifact symlink')
+        if path.is_file():total_artifacts+=path.stat().st_size
+    require(total_artifacts<=256*1024**2-128*1024,'independent whole attempt bound including reserved validation record')
+    require(frozen()==freeze,'all immutable source preparation unchanged')
+    result={'status':'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY','teachers':[11,353],
+            'heads':48,'cached_input_words_available':804960,'cached_output_words_available':798720,
+            'source_plan_sha256':sha(HERE/'source-plan.json'),'manifest_sha256':sha(directory/'manifest.json'),
+            'invocation_sha256':sha(args.output/'invocation.json'),'payload_bytes':payload,
+            'full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified':False}
+    write(args.output/'validation.json',result);print(json.dumps(result,sort_keys=True))
+
+if __name__=='__main__':main()

````

## Source: .agents/runs/model-execution-qwen38-gdn-reference-20261020/internal-draft-1/NEW_PREPARATION.patch

Original UTF-8 bytes: 107132; SHA256 `4fb68a4e03d8d9191b15f643bada1ad3cebe04df02479a4b93bbc76bc70d354e`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/README.md
@@ -0,0 +1,85 @@
+# Original27B layer0 GDN public reference — prepared only
+
+Author: spine_review. Parent plan: model-execution.
+
+- Actual original `Qwen/Qwen3.8-27B` snapshot98cea13c config explicitly declares
+  Qwen3_5ForConditionalGeneration/qwen3_5/qwen3_5_text. Use that exact pinned
+  original module; no architecture inference from the model name.
+- Original F.embedding →original Qwen3_5RMSNorm →original Qwen3_5GatedDeltaNet
+  layer0. Only two teachers:11 enters real padded initial chunk64;353 enters
+  real cached recurrence using that first return through original DynamicCache.
+  No whole-model constructor/traversal,55GB materialization or112-domain claim.
+- Exact nine GDN parameters +inputnorm +embedding bind by original index, shape,
+  raw BF16 kind and same opened source FD/range. All parameters assigned strictly
+  into meta-constructed ORIGINAL classes; no random/unfilled parameter may run.
+  A_log,dt_bias and gated norm gamma remain ORIGINAL BF16. No source36F32 restore,
+  official-loader blanket cast or conversion from GGUF learned values.
+- Genuine original source public forward called unchanged once per invocation.
+  Public USE_HUB_KERNELS=0 declares CPU reference profile; optional package
+  fallback must identify original Torch code or refuse. No function replacement,
+  scalar arithmetic port, reconstructed state or native exp eligibility.
+- Independent fresh plain/observed caches, same immutable weights. Every token's
+  complete output5120, conv10240×4, recurrent48×128×128, embedding and inputnorm
+  must be bit-identical. True initial return must equal cached prior in all words.
+  Real source Q/K16→V48 repeat3 is compared directly from captured actual bytes.
+- Original call/line/return frame hooks reuse accepted source line maps. Capture
+  q/k/v/beta/g, initial/prior/proposed/returned states, all old GDN intermediates,
+  actual conv operands/return, projection input/output, and gated RMS stages.
+  Scoped TorchDispatchMode records original coefficient operations, five real
+  initial exps, cached exp(g) and norm args/results; forwards each original op
+  once and returns its actual result. Observer clones are suppressed.
+- Gated gamma product must be actual BF16 BEFORE FP32 SiLU multiplication. This
+  differs from the `.8B` source36F32 gamma profile. No new native dtype eligibility.
+- Legitimate initial pairwise exp input contains canonical negative Inf ONLY in
+  the exact strictly-upper64×64 source mask. Preserve those raw bits; all other
+  captured values and published outputs must be finite. No tolerance widening.
+- Six actual embedding/inputnorm/QKV outputs also match the independently
+  accepted original27B public prefix reference in every word.
+
+## Bounds and source ownership
+
+- BEFORE importing Torch or creating maps: page4096, eleven exact parameter
+  windows, selected GDN payload231,752,128B; touched source incl inputnorm and
+  two embedding rows231,782,848B; aggregate mapped VA2,774,582,792B (<3GiB).
+- Each COW window maps page-aligned offset and only leading alignment+complete
+  original tensor bytes. Full embedding raw view2.543GB VA retains its original
+  geometry; only trained rows11/353 are read. No whole-shard/18-shard mmap.
+- Before library initialization, derive separate virtual and physical-envelope
+  estimates:6GiB runtime VA allowance,2GiB operator working estimate,128MiB
+  captured payload,2GiB runtime resident allowance,64MiB control allowance.
+  Finite RLIMIT_AS12GiB +configured container12GiB RAM/totalRAM+swap,4CPUs,
+ 600s foreground/2400s outer wall,128MiB per-file,256MiB whole artifact cap.
+  These estimates/OS containment do NOT prove native ResourceArbiter admission
+  or opaque allocator bounds. Any actual cap/refusal remains UNQUALIFIED.
+- ROOT alone hashes ALL18 original shards/all source files before and after via
+  retained FDs and bounded1MiB reads. Inside, selected original FDs are fully
+  verified before importing Torch and rechecked after; COW parameter readsets,
+  exact pointers/dtypes/shapes and versions are unchanged. Source FD/map/tensor
+  owners remain until the owned process exits; no premature mmap close.
+- Exact accepted owned_container_v4.py53ee4270, fresh nonce/CID, networknone,
+  readonly source/probe/root image, no GPU/CDI, inside affinity before Torch.
+  All postspawn paths use original independent cleanup stages and EOF proof.
+  No next provider/process may start after cleanup/evidence failure.
+- Configured OS caps and raw proc/cgroup/mount metadata are retained; no claim
+  that a configured limit or historical RSS is native physical admission.
+
+## ROOT execution, after independent source review
+
+All directories fresh. ROOT owns every test/container/operator execution.
+
+```bash
+python3 "$RUN/test_contracts.py"
+python3 "$RUN/run_reference.py" --new-output "$RUN/root-reference-1"
+python3 "$RUN/validate_reference.py" --output "$RUN/root-reference-1"
+```
+
+Output: untouched plain archive +two prevalidation observed archives, full
+source/runtime/parameter/cleanup records. Failure preserves actual metadata and
+raw frames before semantic validation. Final validator compares all raw inputs,
+intermediates, initial/cached prior, source gamma rounding and transparency.
+H48 cached inputs804960 words and outputs798720 words become reference artifacts
+only after actual ROOT PASS. Existing model slice F32 gamma guard stays CLOSED;
+future source-kind admission and native H48 replay require separate review.
+
+No author new source payload hashes, model/provider/operator/library/container,
+Cargo/GPU execution, MAIN/shared-plan edits or commits. AST/metadata only.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/TODO.md
@@ -0,0 +1,16 @@
+Parent: `.agents/plans/model-execution/`. Owner: spine_review.
+
+- [x] Read actual original config, index and bounded selected source headers.
+- [x] Verify declared Qwen3_5 architecture and all11 raw BF16 role descriptors.
+- [x] Derive exact parameter-window VA/touched-payload bounds before libraries.
+- [x] Prepare original module/meta strict assignment, real DynamicCache, two
+  teacher plain/observed reference, original intermediate/dispatch observers.
+- [x] Prepare whole-source/FD/map/readset/lifetime checks, original owned runner
+  and independent raw/state/source/provider/cleanup validator; no math replacement.
+- [x] Complete author static self-review/AST/freeze; no runtime or tests.
+- [ ] ROOT independent full source review.
+- [ ] ROOT source-free fixtures, actual original public reference and validator.
+- [ ] Separate H48 source-kind/native cached slice qualification after reference.
+
+No author runtime/tests/new payload hashes. No full-model/112 StateRoot/native
+exp/GPU admission from this source-only preparation.
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/capture.py
@@ -0,0 +1,315 @@
+"""ROOT ONLY original27B layer0 GDN module; never a full-model constructor."""
+import gc
+import importlib.metadata
+import inspect
+import json
+import mmap
+import os
+from pathlib import Path
+import resource
+import struct
+import sys
+import time
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, metadata, read, require, sha, unique, write
+
+SOURCE = Path('/source')
+OUTPUT = Path('/output/captured')
+EMBED = 'model.language_model.embed_tokens.weight'
+INPUT_NORM = 'model.language_model.layers.0.input_layernorm.weight'
+PREFIX = 'model.language_model.layers.0.linear_attn.'
+SOURCE_OWNER = None # Original FD/map/storage owner retained until owned process exit.
+
+class SourceOwner:
+    def __init__(self, plan):
+        self.plan, self.fds, self.headers, self.before = plan, {}, {}, {}
+        self.maps, self.tensors, self.readsets, self.versions = {}, {}, {}, {}
+        for name,pin in plan['metadata'].items():
+            read(SOURCE/name,pin=pin)
+        self.index=metadata(SOURCE/'model.safetensors.index.json')
+        for shard,pin in plan['selected_shards'].items():
+            fd=os.open(SOURCE/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            self.fds[shard]=fd
+            self.before[shard]=fingerprint(os.fstat(fd))
+            require(os.fstat(fd).st_size==pin['file']['bytes'] and
+                    fd_sha(fd,0,pin['file']['bytes'])==pin['file']['sha256'],
+                    'complete same opened original selected shard before library initialization')
+            n,=struct.unpack('<Q',os.pread(fd,8,0))
+            raw=os.pread(fd,n,8)
+            require(n==pin['header_bytes'] and len(raw)==n and n<=8*1024*1024 and
+                    digest(raw)==pin['header_sha256'],'exact original selected shard header')
+            self.headers[shard]=(8+n,json.loads(raw,object_pairs_hook=unique))
+
+    def load_views(self,torch):
+        for name,item in self.plan['tensors'].items():
+            shard=item['shard'];begin,header=self.headers[shard];entry=header[name]
+            require(self.index['weight_map'][name]==shard and entry['dtype']=='BF16' and
+                    entry['shape']==item['shape'] and begin+entry['data_offsets'][0]==item['absolute_offset'] and
+                    entry['data_offsets'][1]-entry['data_offsets'][0]==item['bytes'],
+                    'actual exact source role/name/geometry/full-byte extent')
+            # COW protects source files; every window is only its exact parameter
+            # plus the leading alignment bytes, not a whole-shard mapping.
+            mapping=mmap.mmap(self.fds[shard],item['mapping_bytes'],access=mmap.ACCESS_COPY,
+                              offset=item['page_aligned_offset'])
+            self.maps[name]=mapping
+            value=torch.frombuffer(mapping,dtype=torch.bfloat16,count=item['bytes']//2,
+                                   offset=item['view_offset']).reshape(item['shape'])
+            require(value.is_contiguous() and value.device.type=='cpu' and not value.is_meta,
+                    'authentic contiguous BF16 source view')
+            self.tensors[name]=value
+            self.versions[name]=value._version
+            if name==EMBED:
+                self.readsets[name]={str(token):fd_sha(self.fds[shard],item['absolute_offset']+token*10240,10240)
+                                     for token in (11,353)}
+            else:
+                self.readsets[name]=fd_sha(self.fds[shard],item['absolute_offset'],item['bytes'])
+
+    def mapped_sha(self,name,begin,count):
+        value=__import__('hashlib').sha256();view=memoryview(self.maps[name])
+        try:
+            for offset in range(0,count,1024*1024):
+                value.update(view[begin+offset:begin+min(count,offset+1024*1024)])
+        finally:view.release()
+        return value.hexdigest()
+
+    def finish(self):
+        for name,item in self.plan['tensors'].items():
+            require(self.tensors[name]._version==self.versions[name],'original source tensor version unchanged')
+            if name==EMBED:
+                actual={str(t):self.mapped_sha(name,item['view_offset']+t*10240,10240) for t in (11,353)}
+            else:actual=self.mapped_sha(name,item['view_offset'],item['bytes'])
+            require(actual==self.readsets[name],'original mapped learned payload unchanged: '+name)
+        after={}
+        for shard,fd in self.fds.items():
+            pin=self.plan['selected_shards'][shard]['file']
+            after[shard]=fingerprint(os.fstat(fd))
+            require(after[shard]==self.before[shard] and fd_sha(fd,0,pin['bytes'])==pin['sha256'],
+                    'same selected full source FD unchanged after original operators')
+        return after
+
+def main():
+    global SOURCE_OWNER
+    started=time.monotonic()
+    require(not OUTPUT.exists(),'exclusive output directory');OUTPUT.mkdir(mode=0o700)
+    report={'schema':'zerogravity.original27B.layer0-gdn.public-reference.v1','status':'UNQUALIFIED',
+            'steps':[],'tokens':[11,353],'parameter_profile':'original27B-raw-source-bf16-v1',
+            'full_model_StateRoot_native_exp_GPU_eligible':False}
+    try:
+        require(sys.byteorder=='little','original little-endian source')
+        cpus=[int(v) for v in os.environ['ZG_ALLOWED_CPUS'].split(',')]
+        require(len(cpus)==len(set(cpus))==4,'four explicit CPUs');os.sched_setaffinity(0,cpus)
+        require(sorted(os.sched_getaffinity(0))==sorted(cpus),'actual affinity before importing Torch')
+        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and
+                os.environ.get('ATEN_CPU_CAPABILITY')=='avx2' and os.environ.get('USE_HUB_KERNELS')=='0' and
+                not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
+                'declared original CPU reference/no external math interposition')
+        require(sha(HERE/'source-plan.json')==os.environ['ZG_SOURCE_PLAN_SHA'],'exact source metadata plan')
+        plan=metadata(HERE/'source-plan.json')
+        derived=bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+        report.update(source_plan_sha256=sha(HERE/'source-plan.json'),source_repo=plan['repository'],
+                      source_revision=plan['revision'],snapshot=plan['snapshot'],pre_library_bounds=derived,
+                      native_ResourceArbiter_admission_qualified=False)
+        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
+        SOURCE_OWNER=SourceOwner(plan)
+        # No library/model initialization or parameter view precedes the bounds.
+        import torch
+        from torch.nn import functional as F
+        from safetensors.torch import save_file
+        from transformers import Qwen3_5Config, DynamicCache
+        from transformers.models.qwen3_5 import modeling_qwen3_5 as original
+        from observer import Observation, ScopedDispatch, info, total_bytes
+        require(torch.__version__=='2.13.0+cu129' and torch.version.git_version==TORCH_GIT and
+                importlib.metadata.version('transformers')=='5.17.0' and
+                importlib.metadata.version('safetensors')=='0.8.0' and not torch.cuda.is_available(),
+                'exact existing pinned CPU runtime')
+        torch.set_num_threads(4);torch.set_num_interop_threads(1)
+        require(torch.backends.cpu.get_cpu_capability()=='AVX2' and torch.backends.mkldnn.enabled,
+                'original CPU AVX2 policy')
+        require(sha(Path(original.__file__))==MODEL_SHA,'exact original declared architecture source')
+        actual_modules={'modeling_qwen3_5.py':original,
+                        'configuration_qwen3_5.py':sys.modules[Qwen3_5Config.__module__],
+                        'cache_utils.py':sys.modules[DynamicCache.__module__],
+                        'hub_kernels.py':sys.modules['transformers.integrations.hub_kernels']}
+        for name,module in actual_modules.items():
+            require(sha(Path(module.__file__))==plan['runtime_source'][name]['sha256'],
+                    'pinned original module '+name)
+        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
+                     'causal_conv1d_fn','causal_conv1d_update'):
+            function=getattr(original,name);found=False
+            while function is not None:
+                closure=inspect.getclosurevars(function).nonlocals
+                if 'is_new_implementation' in closure:
+                    require(closure['is_new_implementation'] is False and
+                            Path(closure['implementation'].__code__.co_filename).resolve()==Path(original.__file__).resolve(),
+                            'refuse optional non-original package implementation: '+name)
+                    found=True
+                function=getattr(function,'__wrapped__',None)
+            require(found,'authentic original public function wrapper: '+name)
+        config=metadata(SOURCE/'config.json')
+        require(config['architectures']==['Qwen3_5ForConditionalGeneration'] and config['model_type']=='qwen3_5' and
+                config['text_config']['model_type']=='qwen3_5_text','actual27B declared original architecture')
+        cfg=Qwen3_5Config.from_dict(config).text_config
+        require((cfg.hidden_size,cfg.linear_num_key_heads,cfg.linear_num_value_heads,cfg.linear_key_head_dim,
+                 cfg.linear_value_head_dim,cfg.linear_conv_kernel_dim)==(5120,16,48,128,128,4),
+                'original27B GDN exact geometry')
+        SOURCE_OWNER.load_views(torch)
+        with torch.device('meta'):
+            gdn=original.Qwen3_5GatedDeltaNet(cfg,layer_idx=0)
+            input_norm=original.Qwen3_5RMSNorm(5120,eps=cfg.rms_norm_eps)
+        parameters={name.removeprefix(PREFIX):value for name,value in SOURCE_OWNER.tensors.items() if name.startswith(PREFIX)}
+        require(set(parameters)==set(dict(gdn.named_parameters())) and len(parameters)==9 and
+                not dict(gdn.named_buffers()),'every original GDN parameter supplied, no random/unfilled fallback')
+        loaded=gdn.load_state_dict(parameters,strict=True,assign=True)
+        require(not loaded.missing_keys and not loaded.unexpected_keys,'strict complete source parameter assignment')
+        input_norm.load_state_dict({'weight':SOURCE_OWNER.tensors[INPUT_NORM]},strict=True,assign=True)
+        gdn.eval();input_norm.eval()
+        parameter_audit={}
+        for name,value in gdn.named_parameters():
+            raw=SOURCE_OWNER.tensors[PREFIX+name]
+            require(value.dtype==torch.bfloat16 and not value.is_meta and value.data_ptr()==raw.data_ptr() and
+                    tuple(value.shape)==tuple(raw.shape),'exact raw BF16 original parameter, no cast/copy: '+name)
+            parameter_audit[name]={**info(value),'pointer':value.data_ptr(),'version':value._version,
+                                   'source_readset_sha256':SOURCE_OWNER.readsets[PREFIX+name]}
+        require(input_norm.weight.data_ptr()==SOURCE_OWNER.tensors[INPUT_NORM].data_ptr() and
+                input_norm.weight.dtype==torch.bfloat16 and gdn.norm.weight.dtype==torch.bfloat16 and
+                gdn.A_log.dtype==gdn.dt_bias.dtype==torch.bfloat16,
+                'original BF16 gamma/A_log/dt_bias/inputnorm stay untouched')
+        baseline={};observations=[];hooks=[];active=[None]
+        def bits(value):return value.detach().contiguous().view(torch.uint8).numpy().tobytes()
+        def same(a,b,message):
+            require(a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bits(a)==bits(b),message)
+        def projection_hook(name):
+            def hook(module,args,result):
+                if active[0] is not None:
+                    active[0].tensor('projection.'+name+'.input',args[0])
+                    active[0].tensor('projection.'+name+'.output',result)
+            return hook
+        def kept(name,value):
+            count=value.numel()*value.element_size()
+            require(total_bytes[0]+count<=128*1024**2,'bounded combined plain/observed tensor payload')
+            baseline[name]=value.detach().contiguous().clone();total_bytes[0]+=count
+        try:
+            kept('source.A_log',gdn.A_log)
+            kept('source.dt_bias',gdn.dt_bias)
+            kept('source.norm.weight',gdn.norm.weight)
+            with torch.inference_mode():
+                for mode in ('plain','observed'):
+                    cache=DynamicCache(config=cfg)
+                    require(len(cache.layers)==64 and not cache.has_previous_state(0,0),'real original cache starts empty')
+                    if mode=='observed':
+                        for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
+                            hooks.append(gdn.get_submodule(name).register_forward_hook(projection_hook(name)))
+                    for step,token in enumerate((11,353)):
+                        embedding=F.embedding(torch.tensor([[token]],dtype=torch.int64),SOURCE_OWNER.tensors[EMBED])
+                        normalized=input_norm(embedding)
+                        require(embedding.shape==normalized.shape==(1,1,5120),'actual trained original embedding/InputNorm')
+                        require(cache.has_previous_state(0,0) is (step==1),'real initial then cached source dispatch')
+                        if mode=='plain':
+                            output=gdn(normalized,cache_params=cache)
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                kept(f'step-{step}.{name}',value)
+                        else:
+                            obs=Observation(step,gdn);observations.append(obs);active[0]=obs
+                            obs.tensor('prefix.embedding',embedding);obs.tensor('prefix.input_norm',normalized)
+                            require(sys.gettrace() is None,'do not replace an existing tracer')
+                            previous=sys.gettrace();mode_observer=ScopedDispatch(obs);obs.mode=mode_observer
+                            try:
+                                sys.settrace(obs.trace)
+                                with mode_observer:
+                                    output=gdn(normalized,cache_params=cache)
+                            finally:
+                                sys.settrace(previous);obs.mode=None;active[0]=None
+                            obs.tensor('cache.after.conv',cache.layers[0].conv_states[0])
+                            obs.tensor('cache.after.state',cache.layers[0].recurrent_states[0])
+                            obs.tensor('module.output',output)
+                            directory=OUTPUT/f'step-{step}';directory.mkdir()
+                            archive=directory/'actual-gdn.safetensors'
+                            save_file(obs.tensors,str(archive),metadata={'scope':'actual original27B layer0 frames; prevalidation only'})
+                            for name,value in obs.tensors.items():obs.meta[name]['raw_sha256']=digest(bits(value))
+                            record={'step':step,'teacher_token':token,'actual_calls':dict(obs.calls),'events':obs.events,
+                                    'dispatch':obs.dispatch,'tensors':obs.meta,'payload_bytes':obs.bytes,
+                                    'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
+                                    'plain_observed_all_outputs_and_state_exact':False}
+                            write(directory/'prevalidation.json',record)
+                            phase='chunk' if step==0 else 'recurrent'
+                            expected={'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,
+                                      'l2norm':2,'torch_'+phase+'_gated_delta_rule':1,
+                                      'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1}
+                            require(dict(obs.calls)==expected,'actual unchanged original initial/cached call schedule')
+                            for name,value in [('embedding',embedding),('input_norm',normalized),('output',output),
+                                               ('conv',cache.layers[0].conv_states[0]),('state',cache.layers[0].recurrent_states[0])]:
+                                same(baseline[f'step-{step}.{name}'],value,'plain/observed complete actual '+name)
+                            same(obs.tensors[phase+'.return.actual_return_1'],cache.layers[0].recurrent_states[0],
+                                 'original public returned state equals real original cache')
+                            if step:
+                                same(observations[0].tensors['chunk.return.actual_return_1'],
+                                     obs.tensors['recurrent.input.initial_state'],'actual initial return supplies real cached prior')
+                            record['plain_observed_all_outputs_and_state_exact']=True
+                            report['steps'].append(record)
+                            total_bytes[0]+=obs.bytes
+                        for layer in cache.layers[1:]:
+                            require(not getattr(layer,'is_conv_states_initialized',[]) or
+                                    not any(layer.is_conv_states_initialized),'no unrelated layer state allocated')
+        finally:
+            active[0]=None
+            for hook in hooks:hook.remove()
+            require(sys.gettrace() is None,'original tracer restored')
+        save_file(baseline,str(OUTPUT/'plain.safetensors'),metadata={'scope':'genuine unobserved two-teacher module outputs/cache'})
+        prior=metadata(Path('/prefix-reference/reference.json'),pin=plan['prior_public_prefix'])
+        require(prior['source_repo']==plan['repository'] and prior['source_revision']==plan['revision'] and
+                prior['snapshot']==plan['snapshot'],'independent original27B prefix reference source binding')
+        prefix_checks=[]
+        for step,token in enumerate((11,353)):
+            old=prior['steps'][step]
+            require(old['token']==token,'independent prefix teacher binding')
+            for stage,actual in [('embedding',baseline[f'step-{step}.embedding']),
+                                 ('normalized',baseline[f'step-{step}.input_norm']),
+                                 ('projected',observations[step].tensors['projection.in_proj_qkv.output'])]:
+                row=old[stage];expected=read(Path('/prefix-reference')/row['file'],pin=row)
+                widened=b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',bits(actual)))
+                require(actual.dtype==torch.bfloat16 and len(widened)==row['bytes'] and widened==expected,
+                        'all prior genuine original27B prefix words: '+stage)
+                prefix_checks.append({'step':step,'stage':stage,'words':actual.numel(),'bit_exact':True,'prior':row})
+        for name,value in gdn.named_parameters():
+            require(value.data_ptr()==parameter_audit[name]['pointer'] and value._version==parameter_audit[name]['version'],
+                    'original source parameter storage/version unchanged')
+        after=SOURCE_OWNER.finish()
+        maps=Path('/proc/self/maps').read_text()
+        require(len(maps.encode())<=2*1024**2 and '/nix/store/' not in maps,'bounded actual container mapping closure')
+        objects=sorted({line.split(None,5)[5] for line in maps.splitlines() if len(line.split(None,5))==6 and
+                        line.split(None,5)[5].startswith('/') and '.so' in Path(line.split(None,5)[5]).name})
+        require(len(objects)<=256,'bounded actual loaded-library count')
+        dsos={};total_dso=0
+        for path in objects:
+            size=Path(path).stat().st_size;total_dso+=size
+            require(size<=2*1024**3 and total_dso<=16*1024**3 and ' (deleted)' not in path,'bounded actual mapped provider closure')
+            dsos[path]={'bytes':size,'sha256':sha(path)}
+        report.update(status='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION',
+            source_FDs_before=SOURCE_OWNER.before,source_FDs_after=after,source_readsets=SOURCE_OWNER.readsets,
+            all_selected_FDs_and_mapped_parameters_unchanged=True,all_source_parameters_BF16=True,
+            parameter_audit=parameter_audit,source_owner_retained_until_owned_process_exit=True,
+            norm_profile={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+                          'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+                          'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16',
+                          'epsilon':gdn.norm.variance_epsilon,'activation':gdn.norm.activation},
+            payload_bytes=total_bytes[0],
+            plain_archive={'bytes':(OUTPUT/'plain.safetensors').stat().st_size,'sha256':sha(OUTPUT/'plain.safetensors')},
+            hooks_removed=len(hooks),runtime={'torch':torch.__version__,'torch_git':torch.version.git_version,
+                'transformers':importlib.metadata.version('transformers'),'cpu_capability':torch.backends.cpu.get_cpu_capability(),
+                'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
+                'affinity':sorted(os.sched_getaffinity(0)),'cuda_available':False,'modeling_sha256':MODEL_SHA,
+                'mapped_shared_objects':dsos,'proc_status':Path('/proc/self/status').read_text(),
+                'proc_cgroup':Path('/proc/self/cgroup').read_text(),'proc_mountinfo':Path('/proc/self/mountinfo').read_text()},
+            scope='Original27B private layer0 embedding/InputNorm/GDN two-teacher public reference only; all raw source BF16. No full-model logits/112 StateRoots, native exp/provider allocation eligibility, production memory admission, GPU or performance.')
+        report['prior_prefix_checks']=prefix_checks
+        require(report['payload_bytes']<=128*1024**2,'bounded total actual plain/observed payload')
+    except BaseException as error:
+        report['failure']=repr(error)
+        raise
+    finally:
+        report['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+        write(OUTPUT/'manifest.json',report)
+    print('ORIGINAL27B_LAYER0_GDN_CAPTURE_ONLY',flush=True)
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/observer.py
@@ -0,0 +1,249 @@
+"""Original frame clones and exactly-once selected public-op observation only."""
+import collections
+import pathlib
+import sys
+import torch
+from torch.utils._python_dispatch import TorchDispatchMode
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+from support import require
+model_source = pathlib.Path(official.__file__).resolve()
+MAX_TENSOR_BYTES = 128 * 1024 * 1024
+total_bytes = [0]
+
+def info(value):
+    return {"dtype":str(value.dtype),"shape":list(value.shape),"stride":list(value.stride()),
+            "contiguous":value.is_contiguous(),"numel":value.numel(),
+            "bytes":value.numel()*value.element_size(),"device":str(value.device)}
+
+class Observation:
+    def __init__(self, step, gdn):
+        self.step = step
+        self.gdn = gdn
+        self.norm = gdn.norm
+        self.tensors = {}
+        self.meta = {}
+        self.events = []
+        self.calls = collections.Counter()
+        self.frames = {}
+        self.points_seen = set()
+        self.bytes = 0
+        self.dispatch = []
+        self.mode = None
+        self.norm_entered = 0
+        self.norm_exited = 0
+
+    def tensor(self, name, value):
+        assert name not in self.tensors, name
+        assert value.device.type == "cpu"
+        self.bytes += value.numel() * value.element_size()
+        assert total_bytes[0] + self.bytes <= MAX_TENSOR_BYTES
+        self.meta[name] = info(value)
+        mode = self.mode
+        previous = mode.suppressed if mode is not None else False
+        if mode is not None:
+            mode.suppressed = True
+        try:
+            self.tensors[name] = value.detach().contiguous().clone()
+        finally:
+            if mode is not None:
+                mode.suppressed = previous
+
+    def observe(self, frame, label, names, returned=None):
+        metadata = {"function": frame.f_code.co_name, "line": frame.f_lineno,
+                    "label": label, "step": self.step, "tensors": []}
+        for name in names:
+            value = frame.f_locals.get(name)
+            if isinstance(value, torch.Tensor):
+                key = label + "." + name
+                self.tensor(key, value)
+                metadata["tensors"].append(key)
+            elif value is None or isinstance(value, (bool, int, float, str)):
+                metadata[name] = value
+            elif isinstance(value, torch.dtype):
+                metadata[name] = str(value)
+        if returned is not None:
+            values = returned if isinstance(returned, tuple) else (returned,)
+            for index, value in enumerate(values):
+                if isinstance(value, torch.Tensor):
+                    key = label + f".actual_return_{index}"
+                    self.tensor(key, value)
+                    metadata["tensors"].append(key)
+                else:
+                    metadata[f"actual_return_{index}"] = value
+        self.events.append(metadata)
+
+    def trace(self, frame, event, arg):
+        if frame.f_code.co_filename != str(model_source):
+            return None
+        name = frame.f_code.co_name
+        if event == "call":
+            if name == "forward" and frame.f_locals.get("self") is self.gdn:
+                self.calls["Qwen3_5GatedDeltaNet.forward"] += 1
+                self.frames[id(frame)] = "gdn"
+                self.observe(frame, "gdn.input", ["hidden_states"])
+                return self.trace
+            if name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                self.calls[name] += 1
+                self.frames[id(frame)] = "conv"
+                self.observe(frame, "conv.input", ["hidden_states", "conv_state", "weight", "bias", "activation"])
+                return self.trace
+            if name == "forward" and frame.f_locals.get("self") is self.norm:
+                self.calls["Qwen3_5RMSNormGated.forward"] += 1
+                self.frames[id(frame)] = "norm"
+                self.observe(frame, "norm.input", ["hidden_states", "gate"])
+                self.tensor("norm.input.weight", self.norm.weight)
+                self.events[-1]["weight"] = "norm.input.weight"
+                self.events[-1]["variance_epsilon"] = self.norm.variance_epsilon
+                self.events[-1]["activation"] = self.norm.activation
+                return self.trace
+            if name not in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "l2norm"}:
+                return None
+            caller = frame.f_back
+            while caller is not None:
+                owner = caller.f_locals.get("self")
+                if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                    if owner.layer_idx != 0:
+                        return None
+                    break
+                caller = caller.f_back
+            else:
+                return None
+            self.calls[name] += 1
+            if name == "l2norm":
+                self.frames[id(frame)] = "l2_" + ("query" if self.calls[name] == 1 else "key")
+                self.observe(frame, self.frames[id(frame)] + ".input", ["x", "dim", "eps"])
+            else:
+                self.frames[id(frame)] = "chunk" if name == "torch_chunk_gated_delta_rule" else "recurrent"
+                self.observe(frame, self.frames[id(frame)] + ".input", ["query", "key", "value", "g", "beta", "initial_state", "chunk_size", "output_final_state", "use_qk_l2norm_in_kernel"])
+            return self.trace
+        if id(frame) not in self.frames:
+            return None
+        phase = self.frames[id(frame)]
+        if event == "return":
+            if phase == "norm":
+                self.observe(frame, "norm.return", ["hidden_states", "gate", "input_dtype", "variance"], arg)
+            elif phase == "gdn":
+                self.observe(frame, "gdn.return", ["core_attn_out", "z", "output"], arg)
+            elif phase == "conv":
+                self.observe(frame, "conv.return", ["hidden_states", "hidden_states_new", "conv_state", "out"], arg)
+            elif phase.startswith("l2_"):
+                self.observe(frame, phase + ".return", ["inv_norm"], arg)
+            else:
+                self.observe(frame, phase + ".return", ["query", "key", "value", "beta", "decay", "initial_dtype", "pad_size", "num_chunks", "last_recurrent_state", "core_attn_out"], arg)
+            del self.frames[id(frame)]
+            return None
+        if event != "line":
+            return self.trace
+        # A line event sees the actual locals BEFORE the named source line.
+        points = CHUNK_POINTS if phase == "chunk" else RECURRENT_POINTS if phase == "recurrent" else NORM_POINTS if phase == "norm" else GDN_POINTS if phase == "gdn" else {}
+        if frame.f_lineno in points:
+            label, names = points[frame.f_lineno]
+            point = (id(frame), label)
+            if point not in self.points_seen:
+                self.points_seen.add(point)
+                self.observe(frame, phase + "." + label, names)
+        return self.trace
+
+
+NORM_POINTS = {
+    228:("core_fp32",["hidden_states","input_dtype"]),
+    230:("mean",["hidden_states","variance","input_dtype"]),
+    231:("normalized_fp32",["hidden_states","variance","input_dtype"]),
+    232:("weighted_bf16",["hidden_states","gate","input_dtype"]),
+    234:("product_fp32",["hidden_states","gate","input_dtype"]),
+}
+
+
+CHUNK_POINTS = {
+    343: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    347: ("qk_normalized", ["query", "key"]),
+    351: ("query_scaled", ["query", "scaling"]),
+    355: ("padded64", ["query", "key", "value", "beta", "decay", "pad_size", "sequence_length"]),
+    364: ("beta_products", ["v_beta", "k_beta"]),
+    379: ("cumulative_decay", ["decay", "cum_decay"]),
+    384: ("pairwise_decay", ["pairwise_decay"]),
+    386: ("intra_chunk_products", ["ut_system", "intra_chunk_attn"]),
+    406: ("triangular_solutions", ["decayed_k_beta", "new_values", "k_cumdecay"]),
+    413: ("initial_state", ["last_recurrent_state"]),
+    418: ("chunk_decayed_inputs", ["query", "key", "chunk_decay", "num_chunks"]),
+    422: ("v_new", ["v_new"]),
+    425: ("fp32_core_before_state_update", ["core_attn_out", "inter_chunk_attn"]),
+}
+RECURRENT_POINTS = {
+    463: ("fp32_transposed", ["query", "key", "value", "beta", "decay"]),
+    468: ("qk_normalized", ["query", "key"]),
+    472: ("query_scaled", ["query"]),
+    484: ("decay_and_prior", ["decay_t", "last_recurrent_state"]),
+    486: ("decayed_state", ["last_recurrent_state"]),
+    488: ("memory_prediction", ["q_t", "k_t", "v_t", "beta_t", "kv_mem"]),
+    489: ("delta", ["delta"]),
+    491: ("proposed_state", ["last_recurrent_state"]),
+    494: ("fp32_core", ["core_attn_out"]),
+}
+
+
+GDN_POINTS = {
+    574:("projections",["mixed_qkv","z","a","b","use_precomputed_states"]),
+    617:("pre_beta",["query","key","value","a","b"]),
+    620:("coefficients",["a","b","beta","g"]),
+    624:("repeated",["query","key","value","beta","g"]),
+    653:("core_return",["core_attn_out","last_recurrent_state"]),
+    660:("norm_return",["core_attn_out","z"]),
+}
+
+class ScopedDispatch(TorchDispatchMode):
+    def __init__(self, observation):
+        super().__init__()
+        self.observation = observation
+        self.suppressed = False
+
+    def source(self):
+        frame = sys._getframe()
+        while frame is not None:
+            if frame.f_code.co_filename == str(model_source):
+                owner = frame.f_locals.get("self")
+                if owner is self.observation.norm:
+                    return "norm",frame.f_lineno
+                if owner is self.observation.gdn:
+                    return "gdn",frame.f_lineno
+                if frame.f_code.co_name in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule"):
+                    return frame.f_code.co_name,frame.f_lineno
+            frame=frame.f_back
+        return None,None
+
+    def tree(self, value, label, budget):
+        budget[0]+=1
+        require(budget[0]<=256,"bounded operation argument tree")
+        if isinstance(value,torch.Tensor):
+            self.observation.tensor(label,value)
+            return {"tensor":label}
+        if isinstance(value,(list,tuple)):
+            require(len(value)<=32,"bounded argument sequence")
+            return [self.tree(v,label+"."+str(i),budget) for i,v in enumerate(value)]
+        if isinstance(value,dict):
+            require(len(value)<=32 and all(isinstance(k,str) for k in value),"bounded argument mapping")
+            return {k:self.tree(v,label+"."+k,budget) for k,v in value.items()}
+        if value is None or isinstance(value,(bool,int,float,str)):
+            return value
+        if isinstance(value,(torch.dtype,torch.device,torch.layout,torch.memory_format)):
+            return {"torch_value":str(value)}
+        raise ValueError("unhandled original operation argument: "+type(value).__name__)
+
+    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
+        kwargs={} if kwargs is None else kwargs
+        if self.suppressed:
+            return func(*args,**kwargs)
+        scope,line=self.source()
+        selected = scope=="norm" or scope=="gdn" and line in (617,619) or \
+            scope in ("torch_chunk_gated_delta_rule","torch_recurrent_gated_delta_rule") and str(func)=="aten.exp.default"
+        if not selected:
+            return func(*args,**kwargs)
+        require(len(self.observation.dispatch)<128,"bounded selected original operations")
+        index=len(self.observation.dispatch);label=f"dispatch.{index:03d}"
+        record={"index":index,"operation":str(func),"schema":str(func._schema),"scope":scope,
+                "source_line":line,"arguments":self.tree(args,label+".args",[0]),
+                "keyword_arguments":self.tree(kwargs,label+".kwargs",[0])}
+        result=func(*args,**kwargs) # EXACT original call once; original result returned.
+        record["result"]=self.tree(result,label+".result",[0])
+        self.observation.dispatch.append(record)
+        return result
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/run_reference.py
@@ -0,0 +1,132 @@
+"""ROOT ONLY owned CPU reference, whole source before/after, no model on import."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import signal
+import stat
+import subprocess
+import time
+import uuid
+from support import HERE, IMAGE, IMAGE_DIGEST, bounds, fd_sha, fingerprint, frozen, metadata, read, require, sha, write
+
+def source_open(plan):
+    """ROOT-only complete source hashes, bounded chunks; same FDs kept open."""
+    handles, rows = {}, []
+    try:
+        for row in plan['source_files']:
+            require(Path(row['name']).name==row['name'],'contained full source filename')
+            fd=os.open(Path(plan['source_directory'])/row['name'],os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+            handles[row['name']]=fd
+            before=os.fstat(fd)
+            require(stat.S_ISREG(before.st_mode) and before.st_size==row['bytes'] and
+                    fd_sha(fd,0,row['bytes'])==row['sha256'] and fingerprint(before)==fingerprint(os.fstat(fd)),
+                    'actual full original source bytes/FD immutable before reference')
+            rows.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256'],
+                         'fd_identity':fingerprint(before)})
+        require(sum(row['bytes'] for row in plan['source_files'] if row['name'].endswith('.safetensors'))
+                ==plan['weight_bytes']==55563006776,'all18 authentic original weight shards')
+        return handles,rows
+    except BaseException:
+        for fd in handles.values():os.close(fd)
+        raise
+
+def source_after(handles,rows):
+    after=[]
+    for row in rows:
+        fd=handles[row['name']]
+        require(fingerprint(os.fstat(fd))==row['fd_identity'] and
+                fd_sha(fd,0,row['bytes'])==row['sha256'],'same full original source FD after reference')
+        after.append(dict(row))
+    return after
+
+def tree_size(root):
+    total=0
+    for path in root.rglob('*'):
+        require(not path.is_symlink(),'no output artifact symlink')
+        if path.is_file():
+            require(stat.S_ISREG(path.stat().st_mode),'regular actual artifact')
+            total+=path.stat().st_size
+    require(total<=256*1024**2-128*1024,'bounded whole attempt with reserved completion metadata')
+    return total
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__)
+    parser.add_argument('--new-output',type=Path,required=True)
+    args=parser.parse_args()
+    def deadline(_signal,_frame):raise TimeoutError('finite whole source/reference/lifecycle deadline')
+    signal.signal(signal.SIGALRM,deadline);signal.alarm(2400)
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');bounds(plan,os.sysconf('SC_PAGE_SIZE'))
+    require(args.new_output.is_absolute() and args.new_output.parent.resolve()==HERE,
+            'exclusive owned attempt directory');args.new_output.mkdir(mode=0o700)
+    cpus=sorted(os.sched_getaffinity(0))[:4];require(len(cpus)==4,'four available CPUs');os.sched_setaffinity(0,cpus)
+    require(sha(HERE/'owned_container_v4.py')=='53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d',
+            'byte-identical reviewed owned container lifecycle')
+    spec=importlib.util.spec_from_file_location('gdn27_owned',HERE/'owned_container_v4.py')
+    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
+    nonce=uuid.uuid4().hex;owned_name='zg-original27-gdn-'+nonce;cid=args.new_output/'owned.cid'
+    receipt={'status':'UNQUALIFIED','freeze_manifest_sha256':freeze,'source_plan_sha256':sha(HERE/'source-plan.json'),
+             'expected_cpus':cpus,'owned_name':owned_name,'owned_nonce':nonce,'image_id':IMAGE,'image_digest':IMAGE_DIGEST,
+             'no_GPU_or_full_model_or_native_provider_admission':True}
+    handles={};child=log=None;failure=None;started=time.monotonic()
+    try:
+        handles,before=source_open(plan);receipt['source_before']=before
+        inspected=helper.command(helper.PODMAN+['image','inspect',IMAGE],15)
+        require(inspected.returncode==0,'existing approved image; no pull')
+        (args.new_output/'image-inspect.json').write_bytes(inspected.stdout)
+        image,=json.loads(inspected.stdout)
+        require(image['Id']==IMAGE.removeprefix('sha256:') and image['Digest']==IMAGE_DIGEST,'pinned actual image')
+        prefix=Path(plan['prior_public_prefix']['path']).parent
+        env={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'void','ATEN_CPU_CAPABILITY':'avx2',
+             'ZG_ALLOWED_CPUS':','.join(map(str,cpus)),'ZG_SOURCE_PLAN_SHA':sha(HERE/'source-plan.json'),
+             'USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
+             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','MKL_DYNAMIC':'FALSE',
+             'PYTHONPATH':'/probe','PYTHONDONTWRITEBYTECODE':'1','LD_PRELOAD':'','LD_AUDIT':'','VLLM_PLUGINS':''}
+        argv=helper.PODMAN+['run','--name',owned_name,'--label',helper.LABEL+'='+nonce,'--cidfile',str(cid),
+            '--rm','--pull=never','--network=none','--read-only','--cap-drop=all','--security-opt=no-new-privileges',
+            '--memory=12g','--memory-swap=12g','--cpus=4','--pids-limit=256',
+            '--ulimit','fsize=134217728:134217728','--tmpfs','/tmp:rw,nosuid,nodev,size=1g']
+        for key,value in env.items():argv+=['-e',key+'='+value]
+        argv+=['-v',str(HERE)+':/probe:ro','-v',plan['source_directory']+':/source:ro',
+               '-v',str(prefix)+':/prefix-reference:ro','-v',str(args.new_output)+':/output:rw',
+               '--entrypoint','python3',IMAGE,'/probe/capture.py']
+        receipt['argv']=argv;write(args.new_output/'prepared-invocation.json',receipt)
+        try:
+            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
+            receipt['owned_cli_pid']=child.pid
+            log=helper.Log(child.stdout,args.new_output/'execution.log')
+            receipt['exit_code']=log.observe(child,600)
+        except BaseException as error:failure=error
+        finally:
+            receipt['cleanup_errors']=helper.cleanup(child,owned_name,nonce,cid,receipt) if child is not None else []
+            if log is not None:
+                try:log.finish()
+                except BaseException as error:receipt['cleanup_errors'].append('log cleanup: '+repr(error))
+                finally:receipt['log_cleanup']={'pipe_eof':log.eof,'all_closed':log.closed,'total_bytes':log.total,'retained_bytes':log.retained}
+            elif child is not None and child.stdout is not None:
+                try:child.stdout.close()
+                except BaseException as error:receipt['cleanup_errors'].append('uninitialized pipe close: '+repr(error))
+        require(failure is None and receipt.get('exit_code')==0 and receipt['cleanup_errors']==[],
+                'owned original public reference/cleanup failed; preserve this attempt')
+        receipt['source_after']=source_after(handles,before)
+        require(receipt['source_after']==before and frozen()==freeze,'all source/preparation files unchanged')
+        receipt['precompletion_artifact_bytes']=tree_size(args.new_output)
+        receipt['status']='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION'
+    except BaseException as error:
+        if failure is None:failure=error
+    finally:
+        failures=[]
+        for fd in handles.values():
+            try:os.close(fd)
+            except BaseException as error:failures.append(repr(error))
+        receipt['source_FD_cleanup_errors']=failures
+        receipt['source_FDs_closed']=not failures
+        receipt['elapsed_seconds_not_benchmark']=time.monotonic()-started
+        if failure is not None:receipt['failure']=repr(failure)
+        if failure is not None or failures:receipt['status']='FAIL_UNQUALIFIED'
+        write(args.new_output/'invocation.json',receipt)
+    require(receipt['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION',
+            'reference attempt unqualified; retained original evidence')
+
+if __name__=='__main__':main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json
@@ -0,0 +1,449 @@
+{
+  "bounds_before_library_or_views": {
+    "CPU_count": 4,
+    "aggregate_window_mapping_bytes": 2774582792,
+    "capture_payload_cap_bytes": 134217728,
+    "configured_container_RAM_bytes": 12884901888,
+    "conservative_operator_working_envelope_bytes": 2147483648,
+    "container_wall_seconds": 600,
+    "control_allowance_bytes": 67108864,
+    "derived_touched_source_cap_bytes": 268435456,
+    "mapped_VA_cap_bytes": 3221225472,
+    "native_ResourceArbiter_admission_qualified": false,
+    "outer_wall_seconds": 2400,
+    "page_size": 4096,
+    "process_AS_cap_bytes": 12884901888,
+    "runtime_VA_allowance_bytes": 6442450944,
+    "runtime_resident_allowance_bytes": 2147483648,
+    "selected_parameter_bytes": 231752128,
+    "touched_source_payload_bytes": 231782848,
+    "whole_artifact_cap_bytes": 268435456
+  },
+  "geometry": {
+    "conv_channels": 10240,
+    "conv_kernel": 4,
+    "head_width": 128,
+    "hidden": 5120,
+    "key_heads": 16,
+    "layers": 64,
+    "rms_epsilon": 1e-06,
+    "value_heads": 48
+  },
+  "metadata": {
+    "config.json": {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    "model.safetensors.index.json": {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    }
+  },
+  "original_parameters": "ALL raw source BF16, including A_log/dt_bias/norm.gamma; no official-loader cast or restored-F32 replacement",
+  "prior_full_source_verification": {
+    "bytes": 7058,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-owned-window-20261020/source-pin.json",
+    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
+  },
+  "prior_public_prefix": {
+    "bytes": 65417,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-typed-oracle-20261020/root-reference-1/captured/reference.json",
+    "sha256": "794765650ff40a52b821cebf85889cb46593b337ea79b94b50f4bad4d244e0aa"
+  },
+  "repository": "Qwen/Qwen3.8-27B",
+  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
+  "runtime_source": {
+    "cache_utils.py": {
+      "bytes": 104677,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/cache_utils.py",
+      "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c"
+    },
+    "configuration_qwen3_5.py": {
+      "bytes": 8395,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/configuration_qwen3_5.py",
+      "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da"
+    },
+    "hub_kernels.py": {
+      "bytes": 44974,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/hub_kernels.py",
+      "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95"
+    },
+    "modeling_qwen3_5.py": {
+      "bytes": 95685,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/pinned-runtime-source/modeling_qwen3_5.py",
+      "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+    }
+  },
+  "schema": "zerogravity.original27B.layer0-gdn.source-plan.v1",
+  "selected_shards": {
+    "model-00001-of-00018.safetensors": {
+      "file": {
+        "bytes": 3966730552,
+        "git_blob_verified": false,
+        "name": "model-00001-of-00018.safetensors",
+        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+      },
+      "header_bytes": 45392,
+      "header_sha256": "7bdeb03df566aa804dca2cb48e420908cb1864e5222858ad04918299883287e0"
+    },
+    "model-00003-of-00018.safetensors": {
+      "file": {
+        "bytes": 2542796952,
+        "git_blob_verified": false,
+        "name": "model-00003-of-00018.safetensors",
+        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+      },
+      "header_bytes": 144,
+      "header_sha256": "26785014f43e18cd545cebad85d003854b94eb64b90079852ba7efeb8041093c"
+    }
+  },
+  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
+  "source_files": [
+    {
+      "bytes": 11544,
+      "git_blob_verified": true,
+      "name": "LICENSE",
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
+    },
+    {
+      "bytes": 65012,
+      "git_blob_verified": true,
+      "name": "README.md",
+      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
+    },
+    {
+      "bytes": 8952,
+      "git_blob_verified": true,
+      "name": "chat_template.jinja",
+      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
+    },
+    {
+      "bytes": 4312,
+      "git_blob_verified": true,
+      "name": "config.json",
+      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
+    },
+    {
+      "bytes": 238,
+      "git_blob_verified": true,
+      "name": "crc32.txt",
+      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
+    },
+    {
+      "bytes": 202,
+      "git_blob_verified": true,
+      "name": "generation_config.json",
+      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
+    },
+    {
+      "bytes": 3353259,
+      "git_blob_verified": true,
+      "name": "merges.txt",
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
+    },
+    {
+      "bytes": 3966730552,
+      "git_blob_verified": false,
+      "name": "model-00001-of-00018.safetensors",
+      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
+    },
+    {
+      "bytes": 3043080328,
+      "git_blob_verified": false,
+      "name": "model-00002-of-00018.safetensors",
+      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
+    },
+    {
+      "bytes": 2542796952,
+      "git_blob_verified": false,
+      "name": "model-00003-of-00018.safetensors",
+      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
+    },
+    {
+      "bytes": 3988973152,
+      "git_blob_verified": false,
+      "name": "model-00004-of-00018.safetensors",
+      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
+    },
+    {
+      "bytes": 2099339864,
+      "git_blob_verified": false,
+      "name": "model-00005-of-00018.safetensors",
+      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00006-of-00018.safetensors",
+      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00007-of-00018.safetensors",
+      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00008-of-00018.safetensors",
+      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00009-of-00018.safetensors",
+      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00010-of-00018.safetensors",
+      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00011-of-00018.safetensors",
+      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00012-of-00018.safetensors",
+      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00013-of-00018.safetensors",
+      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
+    },
+    {
+      "bytes": 3979553696,
+      "git_blob_verified": false,
+      "name": "model-00014-of-00018.safetensors",
+      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00015-of-00018.safetensors",
+      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
+    },
+    {
+      "bytes": 3979564040,
+      "git_blob_verified": false,
+      "name": "model-00016-of-00018.safetensors",
+      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
+    },
+    {
+      "bytes": 2108759344,
+      "git_blob_verified": false,
+      "name": "model-00017-of-00018.safetensors",
+      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
+    },
+    {
+      "bytes": 3392197344,
+      "git_blob_verified": false,
+      "name": "model-00018-of-00018.safetensors",
+      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
+    },
+    {
+      "bytes": 112216,
+      "git_blob_verified": true,
+      "name": "model.safetensors.index.json",
+      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
+    },
+    {
+      "bytes": 390,
+      "git_blob_verified": true,
+      "name": "preprocessor_config.json",
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
+    },
+    {
+      "bytes": 12809320,
+      "git_blob_verified": false,
+      "name": "tokenizer.json",
+      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
+    },
+    {
+      "bytes": 17928,
+      "git_blob_verified": true,
+      "name": "tokenizer_config.json",
+      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
+    },
+    {
+      "bytes": 385,
+      "git_blob_verified": true,
+      "name": "video_preprocessor_config.json",
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
+    },
+    {
+      "bytes": 6722759,
+      "git_blob_verified": true,
+      "name": "vocab.json",
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
+    }
+  ],
+  "status": "SOURCE_HEADER_ONLY_NO_OPERATOR_EXECUTION",
+  "tensors": {
+    "model.language_model.embed_tokens.weight": {
+      "absolute_offset": 152,
+      "bytes": 2542796800,
+      "dtype": "BF16",
+      "mapping_bytes": 2542796952,
+      "name": "model.language_model.embed_tokens.weight",
+      "page_aligned_offset": 0,
+      "shape": [
+        248320,
+        5120
+      ],
+      "shard": "model-00003-of-00018.safetensors",
+      "view_offset": 152
+    },
+    "model.language_model.layers.0.input_layernorm.weight": {
+      "absolute_offset": 45400,
+      "bytes": 10240,
+      "dtype": "BF16",
+      "mapping_bytes": 10584,
+      "name": "model.language_model.layers.0.input_layernorm.weight",
+      "page_aligned_offset": 45056,
+      "shape": [
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 344
+    },
+    "model.language_model.layers.0.linear_attn.A_log": {
+      "absolute_offset": 55640,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2488,
+      "name": "model.language_model.layers.0.linear_attn.A_log",
+      "page_aligned_offset": 53248,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2392
+    },
+    "model.language_model.layers.0.linear_attn.conv1d.weight": {
+      "absolute_offset": 55736,
+      "bytes": 81920,
+      "dtype": "BF16",
+      "mapping_bytes": 84408,
+      "name": "model.language_model.layers.0.linear_attn.conv1d.weight",
+      "page_aligned_offset": 53248,
+      "shape": [
+        10240,
+        1,
+        4
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.dt_bias": {
+      "absolute_offset": 137656,
+      "bytes": 96,
+      "dtype": "BF16",
+      "mapping_bytes": 2584,
+      "name": "model.language_model.layers.0.linear_attn.dt_bias",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2488
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_a.weight": {
+      "absolute_offset": 137752,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
+      "page_aligned_offset": 135168,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_b.weight": {
+      "absolute_offset": 629272,
+      "bytes": 491520,
+      "dtype": "BF16",
+      "mapping_bytes": 494104,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
+      "page_aligned_offset": 626688,
+      "shape": [
+        48,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
+      "absolute_offset": 1120792,
+      "bytes": 104857600,
+      "dtype": "BF16",
+      "mapping_bytes": 104860184,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
+      "page_aligned_offset": 1118208,
+      "shape": [
+        10240,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.in_proj_z.weight": {
+      "absolute_offset": 105978392,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917144,
+      "name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
+      "page_aligned_offset": 105975808,
+      "shape": [
+        6144,
+        5120
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.norm.weight": {
+      "absolute_offset": 168892952,
+      "bytes": 256,
+      "dtype": "BF16",
+      "mapping_bytes": 2840,
+      "name": "model.language_model.layers.0.linear_attn.norm.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        128
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2584
+    },
+    "model.language_model.layers.0.linear_attn.out_proj.weight": {
+      "absolute_offset": 168893208,
+      "bytes": 62914560,
+      "dtype": "BF16",
+      "mapping_bytes": 62917400,
+      "name": "model.language_model.layers.0.linear_attn.out_proj.weight",
+      "page_aligned_offset": 168890368,
+      "shape": [
+        5120,
+        6144
+      ],
+      "shard": "model-00001-of-00018.safetensors",
+      "view_offset": 2840
+    }
+  },
+  "weight_bytes": 55563006776
+}
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/support.py
@@ -0,0 +1,110 @@
+"""Bounded metadata/source transport; no Torch/provider/model import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+import stat
+
+HERE = Path(__file__).resolve().parent
+IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
+IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
+MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
+TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
+
+def require(ok, message):
+    if not ok:
+        raise ValueError(message)
+
+def digest(raw):
+    return hashlib.sha256(raw).hexdigest()
+
+def sha(path):
+    with Path(path).open('rb') as file:
+        return hashlib.file_digest(file, 'sha256').hexdigest()
+
+def unique(pairs):
+    result = {}
+    for key, value in pairs:
+        require(key not in result, 'duplicate JSON key')
+        result[key] = value
+    return result
+
+def read(path, cap=4*1024*1024, pin=None):
+    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
+        before = os.fstat(file.fileno())
+        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
+        raw = file.read(cap+1)
+        after = os.fstat(file.fileno())
+    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
+    if pin is not None:
+        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
+    return raw
+
+def metadata(path, cap=4*1024*1024, pin=None):
+    return json.loads(read(path, cap, pin), object_pairs_hook=unique)
+
+def write(path, value):
+    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
+    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
+    with Path(path).open('xb') as file:
+        os.fchmod(file.fileno(),0o600)
+        file.write(raw)
+
+def fingerprint(meta):
+    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]
+
+def fd_sha(fd, begin, count):
+    value = hashlib.sha256()
+    for offset in range(0,count,1024*1024):
+        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
+        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
+        value.update(raw)
+    return value.hexdigest()
+
+def bounds(plan, page):
+    """Integer geometry before importing libraries or allocating parameter maps."""
+    limits = plan['bounds_before_library_or_views']
+    require(page == limits['page_size'] == 4096, 'pinned page geometry')
+    require(len(plan['tensors']) == 11 and len(plan['selected_shards']) == 2, 'exact complete selected roles')
+    mapped, params = 0, 0
+    for name, item in plan['tensors'].items():
+        require(item['dtype'] == 'BF16' and type(item['shape']) is list, 'original BF16 source policy')
+        count = 1
+        for axis in item['shape']:
+            require(type(axis) is int and 0 < axis <= 248320, 'bounded original tensor axis')
+            count *= axis
+        start, length = item['absolute_offset'], item['bytes']
+        require(type(start) is type(length) is int and start >= 8 and length == count*2,
+                'full original typed extent')
+        aligned = start-start%page
+        require(item['page_aligned_offset'] == aligned and item['view_offset'] == start-aligned and
+                item['mapping_bytes'] == length+start-aligned and
+                start+length <= plan['selected_shards'][item['shard']]['file']['bytes'],
+                'exact bounded parameter window; no whole shard mapping')
+        mapped += item['mapping_bytes']
+        if '.linear_attn.' in name:
+            params += length
+    touched = params + 10240 + 20480
+    require(params == limits['selected_parameter_bytes'] == 231752128 and
+            mapped == limits['aggregate_window_mapping_bytes'] == 2774582792 and
+            touched == limits['touched_source_payload_bytes'] == 231782848 and
+            mapped <= limits['mapped_VA_cap_bytes'] == 3*1024**3 and
+            touched <= limits['derived_touched_source_cap_bytes'] == 256*1024**2,
+            'pre-library exact source windows and selected resident payload bound')
+    virtual_envelope=mapped+limits['runtime_VA_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']
+    physical_estimate=touched+limits['runtime_resident_allowance_bytes']+limits['conservative_operator_working_envelope_bytes']+limits['capture_payload_cap_bytes']+limits['control_allowance_bytes']
+    require(virtual_envelope < limits['process_AS_cap_bytes'] == 12*1024**3 and
+            physical_estimate < limits['configured_container_RAM_bytes'] and
+            limits['configured_container_RAM_bytes'] == 12*1024**3,
+            'finite explicit reference containment; not native physical admission')
+    return {'mapped_bytes':mapped,'selected_parameter_bytes':params,'touched_source_bytes':touched,
+            'virtual_envelope_bytes':virtual_envelope,'physical_envelope_estimate_bytes':physical_estimate,
+            'native_ResourceArbiter_admission_qualified':False}
+
+def frozen():
+    manifest = metadata(HERE/'prepared-manifest.json')
+    for row in manifest['files']:
+        read(HERE/row['file'],pin=row)
+    for row in manifest['parent_metadata']:
+        read(row['path'],pin=row)
+    return sha(HERE/'prepared-manifest.json')
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/test_contracts.py
@@ -0,0 +1,43 @@
+"""ROOT-only source-free geometry/archive fixtures; no library/model import."""
+import copy
+from pathlib import Path
+import json
+import struct
+import tempfile
+import unittest
+from support import HERE, bounds, digest, metadata
+from validate_reference import archive
+
+class Contracts(unittest.TestCase):
+    def test_exact_parameter_windows_and_full_source_kind_refuse(self):
+        plan=metadata(HERE/'source-plan.json')
+        value=bounds(plan,4096)
+        self.assertEqual(value['mapped_bytes'],2774582792)
+        for field in ('absolute_offset','mapping_bytes','view_offset','bytes'):
+            wrong=copy.deepcopy(plan)
+            first=next(iter(wrong['tensors'].values()))
+            first[field]+=1
+            with self.assertRaises(ValueError):bounds(wrong,4096)
+        wrong=copy.deepcopy(plan);next(iter(wrong['tensors'].values()))['dtype']='F32'
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_native_provider_admission_not_inferred_from_containment(self):
+        plan=metadata(HERE/'source-plan.json')
+        self.assertIs(bounds(plan,4096)['native_ResourceArbiter_admission_qualified'],False)
+        with self.assertRaises(ValueError):bounds(plan,8192)
+        wrong=copy.deepcopy(plan);wrong['bounds_before_library_or_views']['runtime_VA_allowance_bytes']=16*1024**3
+        with self.assertRaises(ValueError):bounds(wrong,4096)
+
+    def test_archive_preserves_actual_negative_inf_mask_and_rejects_tail(self):
+        # Transport-only fixture: no candidate result or numerical expectation.
+        payload=bytes.fromhex('000080ff0000008001000000')
+        header=json.dumps({'x':{'dtype':'F32','shape':[3],'data_offsets':[0,12]}}).encode()
+        raw=struct.pack('<Q',len(header))+header+payload
+        with tempfile.TemporaryDirectory() as root:
+            path=Path(root)/'actual.safetensors';path.write_bytes(raw)
+            pin={'bytes':len(raw),'sha256':digest(raw)}
+            self.assertEqual(archive(path,pin)['x']['raw'],payload)
+            path.write_bytes(raw+b'\0')
+            with self.assertRaises(ValueError):archive(path,{'bytes':len(raw)+1,'sha256':digest(raw+b'\0')})
+
+if __name__=='__main__':unittest.main()
--- /dev/null
+++ b/.agents/runs/model-execution-qwen38-gdn-reference-20261020/validate_reference.py
@@ -0,0 +1,251 @@
+"""ROOT ONLY stdlib full-byte/source/lifecycle validation; no tensor arithmetic."""
+import argparse
+import json
+import os
+from pathlib import Path
+import struct
+from support import HERE, MODEL_SHA, TORCH_GIT, bounds, digest, fd_sha, fingerprint, frozen, metadata, read, require, sha, unique, write
+
+PREFIX='model.language_model.layers.0.linear_attn.'
+
+def source_readsets(plan,report):
+    """ROOT-only actual indexed source bytes, bounded read; no Torch or math."""
+    require(set(report['source_readsets'])==set(plan['tensors']),'all actual source raw readsets')
+    for shard,pin in plan['selected_shards'].items():
+        fd=os.open(Path(plan['source_directory'])/shard,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
+        try:
+            before=fingerprint(os.fstat(fd));n,=struct.unpack('<Q',os.pread(fd,8,0))
+            require(before[2]==pin['file']['bytes'] and n==pin['header_bytes'] and
+                    digest(os.pread(fd,n,8))==pin['header_sha256'],'same authenticated selected source header')
+            for name,item in plan['tensors'].items():
+                if item['shard']!=shard:continue
+                if name=='model.language_model.embed_tokens.weight':
+                    actual={str(t):fd_sha(fd,item['absolute_offset']+t*10240,10240) for t in (11,353)}
+                else:actual=fd_sha(fd,item['absolute_offset'],item['bytes'])
+                require(actual==report['source_readsets'][name],'independent actual raw source readset: '+name)
+            require(fingerprint(os.fstat(fd))==before,'original FD unchanged during independent source-range validation')
+        finally:os.close(fd)
+
+def archive(path,pin):
+    raw=read(path,128*1024**2,pin);n,=struct.unpack_from('<Q',raw)
+    require(1<n<=8*1024**2 and 8+n<len(raw),'bounded complete raw archive header')
+    head=json.loads(raw[8:8+n],object_pairs_hook=unique);tensors={};spans=[]
+    for name,item in head.items():
+        if name=='__metadata__':continue
+        dtype,shape=item['dtype'],item['shape'];require(dtype in ('F32','BF16'),'original raw archive dtype')
+        count=1
+        require(type(shape) is list and len(shape)<=8,'bounded actual rank')
+        for dim in shape:
+            require(type(dim) is int and 0<dim<=248320,'bounded actual dimension');count*=dim
+        lo,hi=item['data_offsets'];size=2 if dtype=='BF16' else 4
+        require(type(lo) is type(hi) is int and 0<=lo<hi and hi-lo==count*size and 8+n+hi<=len(raw),
+                'complete original dtype/extent geometry')
+        tensors[name]={'dtype':dtype,'shape':shape,'raw':raw[8+n+lo:8+n+hi]};spans.append((lo,hi))
+    cursor=0
+    for lo,hi in sorted(spans):require(lo==cursor,'all disjoint complete archive bytes');cursor=hi
+    require(8+n+cursor==len(raw),'no trailing/unbound archive bytes')
+    return tensors
+
+def tensor(tensors,name,dtype,shape=None):
+    value=tensors[name]
+    require(value['dtype']==dtype and (shape is None or value['shape']==shape),'actual stage dtype/shape: '+name)
+    return value['raw']
+
+def widened(raw):
+    return b''.join((word<<16).to_bytes(4,'little') for word, in struct.iter_unpack('<H',raw))
+
+def operation(rows,scope,line,name):
+    selected=[row for row in rows if row['scope']==scope and row['source_line']==line and row['operation']==name]
+    require(len(selected)==1,'exact actual selected operator: '+str((scope,line,name)))
+    return selected[0]
+
+def main():
+    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
+    freeze=frozen();plan=metadata(HERE/'source-plan.json');derived=bounds(plan,4096)
+    invocation=metadata(args.output/'invocation.json');directory=args.output/'captured';report=metadata(directory/'manifest.json')
+    require(invocation['status']=='EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
+            invocation['freeze_manifest_sha256']==freeze and invocation['source_plan_sha256']==sha(HERE/'source-plan.json') and
+            invocation['exit_code']==0 and invocation['cleanup_errors']==[] and invocation['source_FD_cleanup_errors']==[] and
+            invocation['source_FDs_closed'] is True and invocation.get('owned_container_absent_after_reap') is True,
+            'actual original source/reference/owned container all retired')
+    cleanup=invocation['foreground_cleanup'];logs=invocation['log_cleanup']
+    require(cleanup['leader_reaped'] is True and cleanup['group_absent'] is True and cleanup['exit_code']==0 and
+            cleanup['all_nonzero_signals_before_reap'] is True and cleanup['stage_errors']==[] and
+            logs['pipe_eof'] is logs['all_closed'] is True and logs['total_bytes']==logs['retained_bytes']<=4*1024**2,
+            'original finite owned process/PGID/EOF cleanup and complete evidence')
+    expected={v['name']:(v['bytes'],v['sha256']) for v in plan['source_files']}
+    require(invocation['source_before']==invocation['source_after'] and
+            {v['name']:(v['bytes'],v['sha256']) for v in invocation['source_before']}==expected and
+            len(invocation['source_before'])==len(expected),'all original source files before/after full SHA coverage')
+    require(report['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
+            report['schema']=='zerogravity.original27B.layer0-gdn.public-reference.v1' and report['tokens']==[11,353] and
+            report['parameter_profile']=='original27B-raw-source-bf16-v1' and
+            report['source_plan_sha256']==sha(HERE/'source-plan.json') and report['source_repo']==plan['repository'] and
+            report['source_revision']==plan['revision'] and report['snapshot']==plan['snapshot'] and
+            report['pre_library_bounds']==derived and report['native_ResourceArbiter_admission_qualified'] is False and
+            report['full_model_StateRoot_native_exp_GPU_eligible'] is False and report['all_source_parameters_BF16'] is True and
+            report['source_owner_retained_until_owned_process_exit'] is True and report['hooks_removed']==5,
+            'authentic private original27B profile and unchanged eligibility guards')
+    require(report['norm_profile']=={'class':'Qwen3_5RMSNormGated','heads':48,'head_width':128,
+        'raw_gamma_dtype':'torch.bfloat16','weighted_dtype':'torch.bfloat16',
+        'gate_product_dtype':'torch.float32','return_dtype':'torch.bfloat16','epsilon':1e-6,'activation':'silu'},
+        'explicit original27B BF16 gamma round profile')
+    require(report['all_selected_FDs_and_mapped_parameters_unchanged'] is True and
+            report['source_FDs_before']==report['source_FDs_after'] and
+            set(report['source_FDs_before'])==set(plan['selected_shards']),'same exact selected original FD owners')
+    source_readsets(plan,report)
+    expected_parameters={name.removeprefix(PREFIX):item for name,item in plan['tensors'].items() if name.startswith(PREFIX)}
+    require(set(report['parameter_audit'])==set(expected_parameters) and len(expected_parameters)==9,
+            'all and only original learned GDN parameters')
+    for name,item in expected_parameters.items():
+        actual=report['parameter_audit'][name]
+        require(actual['dtype']=='torch.bfloat16' and actual['shape']==item['shape'] and
+                actual['device']=='cpu' and actual['contiguous'] is True and actual['bytes']==item['bytes'] and
+                type(actual['pointer']) is int and actual['pointer']>0 and
+                actual['source_readset_sha256']==report['source_readsets'][PREFIX+name],
+                'actual complete original raw parameter binding: '+name)
+    runtime=report['runtime'];require(runtime['torch']=='2.13.0+cu129' and runtime['torch_git']==TORCH_GIT and
+        runtime['transformers']=='5.17.0' and runtime['cpu_capability']=='AVX2' and runtime['threads']==4 and
+        runtime['interop_threads']==1 and runtime['affinity']==invocation['expected_cpus'] and
+        len(set(runtime['affinity']))==4 and runtime['cuda_available'] is False and runtime['modeling_sha256']==MODEL_SHA,
+        'actual pinned unchanged CPU runtime and four CPUs')
+    dsos=runtime['mapped_shared_objects']
+    require(dsos['/usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so']['sha256']==
+            '1b4ed646ce51a4c341808552426a0f9e2e2c04961955759701066d335c1631ab' and
+            dsos['/usr/lib/x86_64-linux-gnu/libm.so.6']['sha256']==
+            'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',
+            'actual original exp/CPU/scalar-provider mappings, not native admission')
+    plain=archive(directory/'plain.safetensors',report['plain_archive']);steps=[];payload=sum(len(t['raw']) for t in plain.values())
+    for name,shape in [('A_log',[48]),('dt_bias',[48]),('norm.weight',[128])]:
+        require(digest(tensor(plain,'source.'+name,'BF16',shape))==report['source_readsets'][PREFIX+name],
+                'all original BF16 coefficient/gamma source bytes in reference')
+    require(len(report['steps'])==2,'both true source teachers')
+    for step,record in enumerate(report['steps']):
+        phase='chunk' if step==0 else 'recurrent'
+        require(record['step']==step and record['teacher_token']==[11,353][step] and
+                record['plain_observed_all_outputs_and_state_exact'] is True and record['actual_calls']=={
+                    'Qwen3_5GatedDeltaNet.forward':1,'Qwen3_5RMSNormGated.forward':1,'l2norm':2,
+                    'torch_'+phase+'_gated_delta_rule':1,'causal_conv1d_fn' if step==0 else 'causal_conv1d_update':1},
+                'one original chunk64 first, one actual cached call second')
+        calls=[event for event in record['events'] if event['label']==phase+'.input']
+        require(len(calls)==1 and calls[0]['output_final_state'] is True and
+                calls[0]['use_qk_l2norm_in_kernel'] is True and
+                (step==1 or calls[0]['initial_state'] is None and calls[0]['chunk_size']==64),
+                'actual original phase/initial-state/padded-chunk contract')
+        tensors=archive(directory/f'step-{step}/actual-gdn.safetensors',
+                        {'bytes':record['archive_bytes'],'sha256':record['archive_sha256']});steps.append(tensors)
+        before=metadata(directory/f'step-{step}/prevalidation.json')
+        require(before==dict(record,plain_observed_all_outputs_and_state_exact=False),
+                'raw original diagnostics retained before semantic acceptance')
+        require(set(tensors)==set(record['tensors']),'all original raw observed stages bound')
+        for name,item in tensors.items():
+            meta=record['tensors'][name]
+            require(meta['raw_sha256']==digest(item['raw']) and meta['bytes']==len(item['raw']) and
+                    meta['shape']==item['shape'] and meta['dtype']==('torch.bfloat16' if item['dtype']=='BF16' else 'torch.float32') and
+                    meta['device']=='cpu','actual complete observer payload: '+name)
+            payload+=len(item['raw'])
+        require(sum(len(t['raw']) for t in tensors.values())==record['payload_bytes'],'all actual per-step payload counted')
+        for original,observed,dtype,shape in [('embedding','prefix.embedding','BF16',[1,1,5120]),
+                ('input_norm','prefix.input_norm','BF16',[1,1,5120]),('output','module.output','BF16',[1,1,5120]),
+                ('conv','cache.after.conv','BF16',[1,10240,4]),('state','cache.after.state','F32',[1,48,128,128])]:
+            require(tensor(plain,f'step-{step}.'+original,dtype,shape)==tensor(tensors,observed,dtype,shape),
+                    'all plain/observed source outputs and full actual state: '+original)
+        require(tensor(tensors,phase+'.return.actual_return_1','F32',[1,48,128,128])==
+                tensor(tensors,'cache.after.state','F32',[1,48,128,128]),'actual original returned state equals actual cache')
+        require(tensor(tensors,phase+'.return.actual_return_0','BF16',[1,1,48,128])==
+                tensor(tensors,'norm.input.hidden_states','BF16',[48,128]),'original core return feeds actual gated norm')
+        for name in ('query','key'):
+            before=tensor(tensors,'gdn.pre_beta.'+name,'BF16',[1,1,16,128]);after=tensor(tensors,'gdn.repeated.'+name,'BF16',[1,1,48,128])
+            require(after==b''.join(before[h*256:(h+1)*256]*3 for h in range(16)),
+                    'actual original Q/K repeat_interleave(3), no inferred permutation')
+        rows=record['dispatch'];coef=[v for v in rows if v['scope']=='gdn']
+        require([v['operation'] for v in coef]==['aten.sigmoid.default','aten.to.dtype','aten.exp.default','aten.neg.default',
+                'aten.to.dtype','aten.add.Tensor','aten.softplus.default','aten.mul.Tensor'] and
+                [v['source_line'] for v in coef]==[617]+[619]*7,'actual BF16 A_log .float/coefficient source order')
+        def op_raw(row,which='result',index=0):
+            ref=row[which][index] if which=='arguments' else row[which]
+            require(type(ref) is dict and set(ref)=={'tensor'},'actual selected tensor result/argument')
+            return tensors[ref['tensor']]['raw']
+        require(op_raw(coef[0],'arguments')==tensor(tensors,'projection.in_proj_b.output','BF16',[1,1,48]) and
+                op_raw(coef[0])==tensor(tensors,'gdn.coefficients.beta','BF16',[1,1,48]) and
+                op_raw(coef[1],'arguments')==tensor(plain,'source.A_log','BF16',[48]) and
+                op_raw(coef[1])==widened(tensor(plain,'source.A_log','BF16',[48])) and
+                op_raw(coef[4],'arguments')==tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48]) and
+                op_raw(coef[4])==widened(tensor(tensors,'projection.in_proj_a.output','BF16',[1,1,48])) and
+                op_raw(coef[5],'arguments',1)==tensor(plain,'source.dt_bias','BF16',[48]) and
+                op_raw(coef[-1])==tensor(tensors,phase+'.input.g','F32',[1,1,48]),
+                'genuine source coefficient producer inputs/results, no reference lookup')
+        beta=tensor(tensors,phase+'.input.beta','BF16',[1,1,48])
+        require(all(0.0<=struct.unpack('<f',(word<<16).to_bytes(4,'little'))[0]<=1.0 for word, in struct.iter_unpack('<H',beta)) and
+                all(value<=0.0 for value, in struct.iter_unpack('<f',tensor(tensors,phase+'.input.g','F32',[1,1,48]))),
+                'actual beta and original negative log-decay ranges')
+        norm={}
+        require(len([row for row in rows if row['scope']=='norm'])==12,'complete unchanged original gated norm operation count')
+        for stage,op,line,dtype,shape in [
+            ('core','aten.to.dtype',227,'F32',[48,128]),('squares','aten.pow.Tensor_Scalar',228,'F32',[48,128]),
+            ('mean','aten.mean.dim',228,'F32',[48,1]),('denominator','aten.add.Tensor',230,'F32',[48,1]),
+            ('inverse','aten.rsqrt.default',230,'F32',[48,1]),('normalized','aten.mul.Tensor',230,'F32',[48,128]),
+            ('normalized_bf16','aten.to.dtype',231,'BF16',[48,128]),('weighted_bf16','aten.mul.Tensor',231,'BF16',[48,128]),
+            ('gate','aten.to.dtype',232,'F32',[48,128]),('silu','aten.silu.default',232,'F32',[48,128]),
+            ('product','aten.mul.Tensor',232,'F32',[48,128]),('return','aten.to.dtype',234,'BF16',[48,128])]:
+            actual=operation(rows,'norm',line,op);name=actual['result']['tensor'];norm[stage]=tensor(tensors,name,dtype,shape)
+        require(tensor(tensors,'norm.input.weight','BF16',[128])==tensor(plain,'source.norm.weight','BF16',[128]) and
+                norm['weighted_bf16']==tensor(tensors,'norm.weighted_bf16.hidden_states','BF16',[48,128]) and
+                norm['return']==tensor(tensors,'norm.return.actual_return_0','BF16',[48,128]),
+                'actual source BF16 gamma product rounds BEFORE FP32 SiLU gate')
+        exp=[v for v in rows if v['scope']=='torch_'+phase+'_gated_delta_rule']
+        require([v['source_line'] for v in exp]==([381,386,413,414,415] if step==0 else [483]) and
+                all(v['operation']=='aten.exp.default' for v in exp),'all actual original initial/cached exponentials')
+        masked=exp[0]['arguments'][0]['tensor'] if step==0 else None
+        for name,item in tensors.items():
+            if name==masked:
+                require(item['shape']==[1,48,1,64,64] and item['dtype']=='F32','actual upper-triangle exp input geometry')
+                for index,(word,) in enumerate(struct.iter_unpack('<I',item['raw'])):
+                    i,j=(index//64)%64,index%64
+                    require(word==0xff800000 if i<j else word&0x7f800000!=0x7f800000,
+                            'ONLY source strictly-upper exp mask may contain canonical negative Inf')
+            else:
+                mask=0x7f80 if item['dtype']=='BF16' else 0x7f800000
+                fmt='<H' if item['dtype']=='BF16' else '<I'
+                require(all(word&mask!=mask for word, in struct.iter_unpack(fmt,item['raw'])),'all other actual raw words finite: '+name)
+        if step:
+            require(tensor(steps[0],'chunk.return.actual_return_1','F32',[1,48,128,128])==
+                    tensor(tensors,'recurrent.input.initial_state','F32',[1,48,128,128]) and
+                    op_raw(exp[0])==tensor(tensors,'recurrent.decay_and_prior.decay_t','F32',[1,48,1,1]),
+                    'actual full initial chunk prior and actual cached exp(g), never zero replacement')
+        else:
+            for name in ('query','key','value','beta','decay'):
+                value=tensors['chunk.padded64.'+name];width=128 if name in ('query','key','value') else 1
+                require(value['dtype']=='F32' and value['shape']==([1,48,64,128] if width==128 else [1,48,64]),
+                        'actual first-token source padded64 layout')
+                for head in range(48):
+                    require(value['raw'][(head*64+1)*width*4:(head+1)*64*width*4]==bytes(63*width*4),
+                            'all63 genuine source padding lanes canonical positive zero')
+    require(payload==report['payload_bytes'] and payload<=128*1024**2,'all actual plain/observed payload bytes counted')
+    require(len(report['prior_prefix_checks'])==6 and all(v['bit_exact'] is True for v in report['prior_prefix_checks']),
+            'independent previously accepted original27B embedding/norm/QKV prefix closure')
+    prior=metadata(plan['prior_public_prefix']['path'],pin=plan['prior_public_prefix'])
+    for step in range(2):
+        for stage,name,dataset in [('embedding',f'step-{step}.embedding',plain),
+                                   ('normalized',f'step-{step}.input_norm',plain),
+                                   ('projected','projection.in_proj_qkv.output',steps[step])]:
+            row=prior['steps'][step][stage]
+            require(widened(tensor(dataset,name,'BF16'))==read(Path(plan['prior_public_prefix']['path']).parent/row['file'],pin=row),
+                    'independent every genuine original prefix word')
+    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}=={
+        'manifest.json','plain.safetensors','step-0/actual-gdn.safetensors','step-0/prevalidation.json',
+        'step-1/actual-gdn.safetensors','step-1/prevalidation.json'},'all captured source/byte artifacts accounted')
+    total_artifacts=0
+    for path in args.output.rglob('*'):
+        require(not path.is_symlink(),'no unknown artifact symlink')
+        if path.is_file():total_artifacts+=path.stat().st_size
+    require(total_artifacts<=256*1024**2-128*1024,'independent whole attempt bound including reserved validation record')
+    require(frozen()==freeze,'all immutable source preparation unchanged')
+    result={'status':'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY','teachers':[11,353],
+            'heads':48,'cached_input_words_available':804960,'cached_output_words_available':798720,
+            'source_plan_sha256':sha(HERE/'source-plan.json'),'manifest_sha256':sha(directory/'manifest.json'),
+            'invocation_sha256':sha(args.output/'invocation.json'),'payload_bytes':payload,
+            'full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified':False}
+    write(args.output/'validation.json',result);print(json.dumps(result,sort_keys=True))
+
+if __name__=='__main__':main()

````

